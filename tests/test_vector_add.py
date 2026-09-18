"""文件职责：测试文件，包含正确性测试、非法输入测试和 PyTorch 契约测试
验证 Vector Add 的注册契约、数值、边界、错误和 stream 语义。"""

import pytest
import torch

import my_ops


CUDA_REQUIRED = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires an NVIDIA CUDA GPU"
)


def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # 测试始终走公开 Dispatcher 入口，不绕过注册直接调用 C++ 符号。
    return torch.ops.my_ops.vector_add(a, b)


def test_import_loads_compiled_extension():
    assert my_ops._C is not None


def test_vector_add_schema_is_registered():
    assert str(torch.ops.my_ops.vector_add.default._schema) == (
        "my_ops::vector_add(Tensor a, Tensor b) -> Tensor"
    )


@CUDA_REQUIRED
@pytest.mark.parametrize("numel", [1024, 1003])
def test_vector_add_1d_matches_torch(numel):
    # 1024 覆盖整除 grid，1003 专门覆盖最后一个 block 的边界判断。
    torch.manual_seed(0)
    a = torch.randn(numel, device="cuda", dtype=torch.float32)
    b = torch.randn_like(a)

    # 使用 PyTorch 官方测试函数进行对比
    actual = vector_add(a, b)
    torch.testing.assert_close(actual, a + b)

    assert actual.shape == a.shape
    assert actual.dtype == a.dtype
    assert actual.device == a.device


# 特殊边界覆盖
@CUDA_REQUIRED
@pytest.mark.parametrize("shape", [(17, 59), (3, 7, 11)])
def test_vector_add_handles_multidimensional_tensors(shape):
    a = torch.randn(shape, device="cuda", dtype=torch.float32)
    b = torch.randn_like(a)

    torch.testing.assert_close(vector_add(a, b), a + b)


@CUDA_REQUIRED
def test_vector_add_handles_empty_tensor():
    a = torch.empty((2, 0, 3), device="cuda", dtype=torch.float32)
    b = torch.empty_like(a)

    actual = vector_add(a, b)

    assert actual.shape == a.shape
    assert actual.numel() == 0


@CUDA_REQUIRED
def test_vector_add_rejects_mismatched_shape():
    a = torch.randn(8, device="cuda", dtype=torch.float32)
    b = torch.randn(9, device="cuda", dtype=torch.float32)

    with pytest.raises(RuntimeError, match="same shape"):
        vector_add(a, b)


def test_vector_add_rejects_cpu_tensors():
    a = torch.randn(8, dtype=torch.float32)
    b = torch.randn_like(a)

    with pytest.raises(NotImplementedError, match="CPU"):
        vector_add(a, b)


@CUDA_REQUIRED
def test_vector_add_rejects_non_fp32_dtype():
    a = torch.randn(8, device="cuda", dtype=torch.float16)
    b = torch.randn_like(a)

    with pytest.raises(RuntimeError, match="torch.float32"):
        vector_add(a, b)


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="需要至少两张 CUDA GPU 验证跨设备输入"
)
def test_vector_add_rejects_different_cuda_devices():
    # Dispatcher 会因任一 CUDA 输入路由到 wrapper；完整的同设备契约仍由 wrapper 检查。
    a = torch.randn(8, device="cuda:0", dtype=torch.float32)
    b = torch.randn(8, device="cuda:1", dtype=torch.float32)

    with pytest.raises(RuntimeError, match="same CUDA device"):
        vector_add(a, b)


@CUDA_REQUIRED
def test_vector_add_rejects_noncontiguous_tensor():
    a = torch.randn((8, 16), device="cuda", dtype=torch.float32).transpose(0, 1)
    b = torch.randn(a.shape, device="cuda", dtype=torch.float32)

    assert not a.is_contiguous()
    with pytest.raises(RuntimeError, match="contiguous"):
        vector_add(a, b)


@CUDA_REQUIRED
def test_vector_add_rejects_autograd_input():
    a = torch.randn(8, device="cuda", dtype=torch.float32, requires_grad=True)
    b = torch.randn_like(a)

    with pytest.raises(RuntimeError, match="autograd is not supported"):
        vector_add(a, b)


@CUDA_REQUIRED
def test_vector_add_uses_non_default_stream():
    default_stream = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    a = torch.randn(1003, device="cuda", dtype=torch.float32)
    b = torch.randn_like(a)

    with torch.cuda.stream(stream):
        # 输入在 default stream 上创建，先显式建立跨 stream 依赖。
        stream.wait_stream(default_stream)
        # wrapper 必须读取这里的 current stream，不能把 kernel 偷偷发到 default stream。
        actual = vector_add(a, b)
        expected = a + b

    stream.synchronize()
    torch.testing.assert_close(actual, expected)


@CUDA_REQUIRED
def test_vector_add_opcheck():
    a = torch.randn(1003, device="cuda", dtype=torch.float32)
    b = torch.randn_like(a)

    # opcheck 验证 schema、FakeTensor 和编译契约；数值正确性由前面的 reference 测试负责。
    torch.library.opcheck(torch.ops.my_ops.vector_add.default, (a, b))
