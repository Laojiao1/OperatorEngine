"""文件职责：独立验证 Triton FlashAttention 的数值、边界、输入契约和 current stream。"""

import math

import pytest
import torch

pytest.importorskip("triton")

from my_ops.attention_triton import _should_skip_causal_future_tiles, triton_attention


CUDA_REQUIRED = pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 NVIDIA CUDA GPU")


def attention_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if causal:
        sequence_length = q.shape[-2]
        mask = torch.triu(torch.ones((sequence_length, sequence_length), device=q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), v.float()).to(q.dtype)


@CUDA_REQUIRED
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("shape", [(1, 2, 17, 64), (2, 3, 32, 128), (1, 2, 65, 64)])
def test_triton_attention_matches_reference(shape, causal):
    torch.manual_seed(3)
    q = torch.randn(shape, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    actual = triton_attention(q, k, v, causal)
    expected = attention_reference(q, k, v, causal)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@CUDA_REQUIRED
@pytest.mark.parametrize("shape", [(0, 2, 17, 64), (1, 0, 17, 64), (1, 2, 0, 64)])
def test_triton_attention_handles_empty_tensor(shape):
    q = torch.empty(shape, device="cuda", dtype=torch.float16)
    actual = triton_attention(q, q, q, False)

    assert actual.shape == q.shape
    assert actual.numel() == 0


@CUDA_REQUIRED
def test_triton_attention_causal_skip_path_matches_reference():
    torch.manual_seed(4)
    q = torch.randn((1, 1, 512, 64), device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    actual = triton_attention(q, k, v, True)
    expected = attention_reference(q, k, v, True)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def test_triton_attention_causal_skip_threshold():
    assert not _should_skip_causal_future_tiles(511, True)
    assert _should_skip_causal_future_tiles(512, True)
    assert not _should_skip_causal_future_tiles(1024, False)


@CUDA_REQUIRED
def test_triton_attention_uses_non_default_stream():
    default_stream = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    q = torch.randn((1, 2, 65, 64), device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    with torch.cuda.stream(stream):
        stream.wait_stream(default_stream)
        actual = triton_attention(q, k, v, True)
        expected = attention_reference(q, k, v, True)

    stream.synchronize()
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@CUDA_REQUIRED
def test_triton_attention_rejects_unsupported_input():
    q = torch.randn((1, 2, 17, 64), device="cuda", dtype=torch.float32)
    with pytest.raises(TypeError, match="只支持 FP16"):
        triton_attention(q, q, q, False)
