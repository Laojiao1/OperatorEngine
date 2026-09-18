"""文件职责：验证 Softmax 的 Dispatcher 契约、数值、边界和错误处理。"""

import pytest
import torch
import my_ops


def test_softmax_schema_is_registered():
    schema = torch.ops.my_ops.softmax.default._schema

    assert str(schema) == (
        "my_ops::softmax(Tensor input, int dim) -> Tensor"
    )

CUDA_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="需要 NVIDIA CUDA GPU",
)


@CUDA_REQUIRED
@pytest.mark.parametrize("shape", [(0,), (0, 128), (2, 0, 3)])
def test_softmax_empty_tensor_returns_empty_output(shape):
    input = torch.empty(
        shape,
        device="cuda",
        dtype=torch.float32,
    )

    output = torch.ops.my_ops.softmax(input, -1)

    assert output.shape == input.shape
    assert output.dtype == input.dtype
    assert output.device == input.device
    assert output.numel() == 0


@CUDA_REQUIRED
@pytest.mark.parametrize("dim", [0, -2])
def test_softmax_rejects_non_last_dimension(dim):
    input = torch.randn(
        (4, 7),
        device="cuda",
        dtype=torch.float32,
    )

    with pytest.raises(
        RuntimeError,
        match="只支持沿最后一维",
    ):
        torch.ops.my_ops.softmax(input, dim)


@CUDA_REQUIRED
def test_softmax_rejects_unsupported_dtype():
    input = torch.randn(
        (4, 7),
        device="cuda",
        dtype=torch.bfloat16,
    )
    with pytest.raises(RuntimeError, match="FP32 和 FP16"):
        torch.ops.my_ops.softmax(input, -1)


def test_softmax_rejects_cpu_input():
    input = torch.randn(4, 7)

    with pytest.raises(NotImplementedError, match="CPU"):
        torch.ops.my_ops.softmax(input, -1)


@CUDA_REQUIRED
def test_softmax_rejects_scalar_input():
    input = torch.tensor(1.0, device="cuda")

    with pytest.raises(RuntimeError, match="至少必须是一维"):
        torch.ops.my_ops.softmax(input, -1)


@CUDA_REQUIRED
def test_softmax_rejects_noncontiguous_input():
    input = torch.randn(
        (4, 7),
        device="cuda",
        dtype=torch.float32,
    ).transpose(0, 1)

    assert not input.is_contiguous()

    with pytest.raises(RuntimeError, match="contiguous"):
        torch.ops.my_ops.softmax(input, -1)


@CUDA_REQUIRED
def test_softmax_rejects_autograd_input():
    input = torch.randn(
        (4, 7),
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )

    with pytest.raises(RuntimeError, match="autograd"):
        torch.ops.my_ops.softmax(input, -1)


@CUDA_REQUIRED
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(1, 1), (7, 127), (32, 1024), (16, 1024 + 123), (2, 3, 257)])
def test_softmax_correctness(shape, dtype):
    """覆盖最小行、非 2 次幂 reduction、对齐、尾部和多维 Tensor。"""
    torch.manual_seed(0)
    input = torch.randn(*shape, device="cuda", dtype=dtype)

    # 极大值会让不减 row_max 的实现溢出，同时验证数值稳定路径。
    input.flatten()[0] = 1000.0

    expected = torch.softmax(input, dim=-1)
    actual = torch.ops.my_ops.softmax(input, -1)

    if dtype == torch.float16:
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
    else:
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@CUDA_REQUIRED
def test_softmax_accepts_positive_last_dimension():
    input = torch.randn(4, 7, device="cuda", dtype=torch.float32)

    actual = torch.ops.my_ops.softmax(input, 1)
    expected = torch.softmax(input, dim=1)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@CUDA_REQUIRED
def test_softmax_contiguous_but_unaligned_storage_offset():
    # contiguous 只保证逻辑步长连续，不保证带 storage_offset 的 data_ptr 仍然满足 float4 对齐。
    storage = torch.randn(4 * 1024 + 1, device="cuda", dtype=torch.float32)
    input = storage[1:].view(4, 1024)

    assert input.is_contiguous()
    assert input.data_ptr() % 16 != 0

    actual = torch.ops.my_ops.softmax(input, -1)
    expected = torch.softmax(input, dim=-1)

    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@CUDA_REQUIRED
def test_softmax_uses_non_default_stream():
    default_stream = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    input = torch.randn(32, 1024, device="cuda", dtype=torch.float32)

    with torch.cuda.stream(stream):
        stream.wait_stream(default_stream)
        actual = torch.ops.my_ops.softmax(input, -1)
        expected = torch.softmax(input, dim=-1)

    stream.synchronize()
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@CUDA_REQUIRED
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_softmax_opcheck(dtype):
    input = torch.randn(17, 1024, device="cuda", dtype=dtype)

    # FP32 对齐输入进入 v4，FP16 输入进入 v1，可以同时覆盖两条真实 CUDA 路径。
    torch.library.opcheck(torch.ops.my_ops.softmax.default, (input, -1))


@CUDA_REQUIRED
@pytest.mark.parametrize(
    "values",
    [
        [1000.0, 999.0, 998.0],
        [-1000.0, -1000.0, -1000.0],
    ],
)
def test_softmax_is_numerically_stable(values):
    input = torch.tensor([values], device="cuda", dtype=torch.float32)

    actual = torch.ops.my_ops.softmax(input, -1)
    expected = torch.softmax(input, dim=-1)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    # 除了逐元素对齐 reference，还要明确验证没有溢出或下溢产生 NaN/Inf。
    assert torch.isfinite(actual).all()

    # Softmax 每行应形成概率分布，行和必须接近 1。
    torch.testing.assert_close(
        actual.sum(dim=-1),
        torch.ones(1, device="cuda"),
        atol=1e-6,
        rtol=1e-6,
    )