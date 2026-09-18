"""文件职责：验证 Attention schema、PyTorch reference、数值语义和后续 CUDA 实现。"""

import math

import pytest
import torch
import torch.nn.functional as F

import my_ops


CUDA_REQUIRED = pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 NVIDIA CUDA GPU")

CUSTOM_KERNEL_CASES = [
    pytest.param((1, 2, 17, 64), id="cuda_core_short_d64"),
    pytest.param((2, 3, 32, 128), id="cuda_core_d128"),
    pytest.param((2, 3, 32, 64), id="tensor_core_aligned"),
    pytest.param((1, 2, 33, 64), id="tensor_core_boundary"),
]


def attention_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False) -> torch.Tensor:
    """使用显式 FP32 Scores/Softmax/PV 定义第一版 Attention 数学语义。"""
    head_dim = q.shape[-1]
    sequence_length = q.shape[-2]

    scores = torch.matmul(q.float(), k.float().transpose(-2, -1))
    scores = scores * (1.0 / math.sqrt(head_dim))

    if causal:
        future_mask = torch.triu(torch.ones((sequence_length, sequence_length), device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(future_mask, float("-inf"))

    probabilities = torch.softmax(scores, dim=-1)
    output = torch.matmul(probabilities, v.float())

    # 算子内部使用 FP32 累加，最终输出恢复为输入 FP16 dtype。
    return output.to(q.dtype)


def test_attention_schema_is_registered():
    schema = torch.ops.my_ops.attention.default._schema
    assert str(schema) == "my_ops::attention(Tensor q, Tensor k, Tensor v, bool causal=False) -> Tensor"


@CUDA_REQUIRED
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("shape", [(1, 2, 17, 64), (2, 3, 32, 128)])
def test_attention_reference_matches_sdpa(shape, causal):
    torch.manual_seed(0)

    q = torch.randn(shape, device="cuda", dtype=torch.float16)
    k = torch.randn(shape, device="cuda", dtype=torch.float16)
    v = torch.randn(shape, device="cuda", dtype=torch.float16)

    actual = attention_reference(q, k, v, causal)
    expected = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=causal)

    assert actual.shape == shape
    assert actual.dtype == torch.float16
    assert actual.device == q.device
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@CUDA_REQUIRED
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("shape", CUSTOM_KERNEL_CASES)
def test_attention_custom_kernels_match_reference(shape, causal):
    torch.manual_seed(1)

    q = torch.randn(shape, device="cuda", dtype=torch.float16)
    k = torch.randn(shape, device="cuda", dtype=torch.float16)
    v = torch.randn(shape, device="cuda", dtype=torch.float16)

    actual = torch.ops.my_ops.attention(q, k, v, causal)
    expected = attention_reference(q, k, v, causal)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@CUDA_REQUIRED
@pytest.mark.parametrize("causal", [False, True])
def test_attention_long_sequence_falls_back_to_sdpa(causal):
    torch.manual_seed(2)

    # CUDA Core 第一轮迁移只接管 N<=1024；1025 应显式走 PyTorch SDPA。
    shape = (1, 1, 1025, 64)
    q = torch.randn(shape, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    actual = torch.ops.my_ops.attention(q, k, v, causal)
    expected = attention_reference(q, k, v, causal)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@CUDA_REQUIRED
def test_attention_handles_empty_sequence():
    q = torch.empty((2, 3, 0, 64), device="cuda", dtype=torch.float16)

    actual = torch.ops.my_ops.attention(q, q, q, True)

    assert actual.shape == q.shape
    assert actual.dtype == q.dtype
    assert actual.numel() == 0


@CUDA_REQUIRED
def test_attention_rejects_unsupported_head_dim():
    q = torch.randn((1, 2, 8, 32), device="cuda", dtype=torch.float16)

    with pytest.raises(RuntimeError, match="head_dim=64 或 128"):
        torch.ops.my_ops.attention(q, q, q, False)


@CUDA_REQUIRED
def test_attention_rejects_noncontiguous_input():
    q = torch.randn((1, 2, 64, 16), device="cuda", dtype=torch.float16).transpose(-2, -1)
    assert q.shape == (1, 2, 16, 64)
    assert not q.is_contiguous()

    with pytest.raises(RuntimeError, match="contiguous"):
        torch.ops.my_ops.attention(q, q, q, False)


@CUDA_REQUIRED
def test_attention_rejects_autograd_input():
    q = torch.randn((1, 2, 8, 64), device="cuda", dtype=torch.float16, requires_grad=True)

    with pytest.raises(RuntimeError, match="autograd"):
        torch.ops.my_ops.attention(q, q, q, False)


def test_attention_rejects_cpu_input():
    q = torch.randn((1, 2, 8, 64), dtype=torch.float16)

    with pytest.raises(NotImplementedError, match="CPU"):
        torch.ops.my_ops.attention(q, q, q, False)


@CUDA_REQUIRED
@pytest.mark.parametrize("shape", [(1, 2, 17, 64), (1, 2, 32, 64), (1, 2, 32, 128)])
def test_attention_uses_non_default_stream(shape):
    default_stream = torch.cuda.current_stream()
    stream = torch.cuda.Stream()

    q = torch.randn(shape, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    with torch.cuda.stream(stream):
        stream.wait_stream(default_stream)
        actual = torch.ops.my_ops.attention(q, k, v, True)
        expected = attention_reference(q, k, v, True)

    stream.synchronize()
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@CUDA_REQUIRED
@pytest.mark.parametrize("causal", [False, True])
def test_attention_opcheck(causal):
    q = torch.randn((1, 2, 17, 64), device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    torch.library.opcheck(torch.ops.my_ops.attention.default, (q, k, v, causal))
