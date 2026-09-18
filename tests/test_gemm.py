"""文件职责：验证 GEMM 算子的 Dispatcher 契约、数值正确性、边界行为和错误处理。"""

import torch
import my_ops
import pytest


def test_gemm_schema_is_registered():
    schema = torch._C._dispatch_find_schema_or_throw("my_ops::gemm", "").schema()
    assert str(schema) == "my_ops::gemm(Tensor a, Tensor b, bool transpose_b=False) -> Tensor"


CUDA_REQUIRED = pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 NVIDIA CUDA GPU")


def gemm(a: torch.Tensor, b: torch.Tensor, transpose_b: bool = False) -> torch.Tensor:
    # 测试始终经过公开 Dispatcher 入口。
    return torch.ops.my_ops.gemm(a, b, transpose_b)


@CUDA_REQUIRED
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("transpose_b", [False, True])
@pytest.mark.parametrize("shape", [(1, 1, 1), (17, 19, 23), (32, 32, 32), (35, 37, 29), (128, 128, 32), (129, 131, 33), (256, 256, 64), (17, 257, 257), (1024, 1024, 32), (1025, 1025, 33)])
def test_gemm_matches_torch(shape, transpose_b, dtype):
    M, N, K = shape
    torch.manual_seed(0)

    a = torch.randn((M, K), device="cuda", dtype=dtype)
    b_shape = (N, K) if transpose_b else (K, N)
    b = torch.randn(b_shape, device="cuda", dtype=dtype)

    actual = gemm(a, b, transpose_b)
    expected_b = b.float().transpose(0, 1) if transpose_b else b.float()
    expected = a.float() @ expected_b

    assert actual.shape == (M, N)
    assert actual.dtype == torch.float32
    assert actual.device == a.device
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


@CUDA_REQUIRED
@pytest.mark.parametrize("shape", [(0, 7, 5), (3, 0, 5)])
def test_gemm_handles_empty_output(shape):
    M, N, K = shape
    a = torch.empty((M, K), device="cuda", dtype=torch.float32)
    b = torch.empty((K, N), device="cuda", dtype=torch.float32)

    actual = gemm(a, b)

    assert actual.shape == (M, N)
    assert actual.dtype == torch.float32
    assert actual.numel() == 0


@CUDA_REQUIRED
def test_gemm_rejects_mismatched_k():
    a = torch.randn((3, 5), device="cuda")
    b = torch.randn((7, 4), device="cuda")

    with pytest.raises(RuntimeError, match="K 维度不匹配"):
        gemm(a, b)


@CUDA_REQUIRED
def test_gemm_uses_non_default_stream():
    default_stream = torch.cuda.current_stream()
    stream = torch.cuda.Stream()

    a = torch.randn((35, 29), device="cuda")
    b = torch.randn((29, 37), device="cuda")

    with torch.cuda.stream(stream):
        stream.wait_stream(default_stream)
        actual = gemm(a, b)
        expected = a @ b

    stream.synchronize()
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


@CUDA_REQUIRED
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("transpose_b", [False, True])
def test_gemm_opcheck(dtype, transpose_b):
    a = torch.randn((17, 29), device="cuda", dtype=dtype)
    b_shape = (23, 29) if transpose_b else (29, 23)
    b = torch.randn(b_shape, device="cuda", dtype=dtype)

    torch.library.opcheck(torch.ops.my_ops.gemm.default, (a, b, transpose_b))


@CUDA_REQUIRED
def test_gemm_unaligned_fp16_input_uses_tiled_fallback():
    # 该规模在地址对齐时会进入 small Tensor Core；未对齐是回退 v1 的唯一原因。
    M, N, K = 256, 256, 64

    # storage_offset=1 使 Tensor 保持 contiguous，但 FP16 data_ptr 偏移 2 Byte。
    a_storage = torch.randn(M * K + 1, device="cuda", dtype=torch.float16)
    b_storage = torch.randn(K * N + 1, device="cuda", dtype=torch.float16)
    a = a_storage[1:].view(M, K)
    b = b_storage[1:].view(K, N)

    assert a.is_contiguous()
    assert b.is_contiguous()
    assert a.data_ptr() % 16 != 0
    assert b.data_ptr() % 16 != 0

    actual = gemm(a, b)
    expected = a.float() @ b.float()

    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


@CUDA_REQUIRED
def test_gemm_tensor_core_uses_non_default_stream():
    default_stream = torch.cuda.current_stream()
    stream = torch.cuda.Stream()

    # 256x256x64 超过小工作量阈值，并满足 small aligned 路径条件。
    a = torch.randn((256, 64), device="cuda", dtype=torch.float16)
    b = torch.randn((64, 256), device="cuda", dtype=torch.float16)

    with torch.cuda.stream(stream):
        stream.wait_stream(default_stream)
        actual = gemm(a, b)
        expected = a.float() @ b.float()

    stream.synchronize()
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
