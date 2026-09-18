"""文件职责：验证统一 Attention Python 接口的 provider 选择、数值语义和输入契约。"""

import pytest
import torch
import torch.nn.functional as F

import my_ops
import my_ops.functional as functional


CUDA_REQUIRED = pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 NVIDIA CUDA GPU")


@pytest.mark.parametrize(
    ("sequence_length", "causal", "expected"),
    [
        (17, False, "triton"),
        (512, False, "triton"),
        (513, False, "sdpa"),
        (1024, False, "sdpa"),
        (1024, True, "triton"),
    ],
)
def test_attention_auto_selector_boundaries(sequence_length, causal, expected):
    assert functional._select_attention_provider(sequence_length, causal) == expected


@CUDA_REQUIRED
@pytest.mark.parametrize("provider", ["auto", "cpp", "triton", "sdpa"])
@pytest.mark.parametrize("causal", [False, True])
def test_attention_providers_match_sdpa(provider, causal):
    torch.manual_seed(5)
    q = torch.randn((1, 2, 33, 64), device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    actual = my_ops.attention(q, k, v, causal=causal, provider=provider)
    expected = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=causal)

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


@CUDA_REQUIRED
@pytest.mark.parametrize(
    ("sequence_length", "causal", "expected_provider"),
    [(17, False, "triton"), (513, False, "sdpa"), (513, True, "triton")],
)
def test_attention_auto_calls_selected_provider(monkeypatch, sequence_length, causal, expected_provider):
    calls: list[str] = []

    def fake_triton(q, k, v, causal=False):
        calls.append("triton")
        return torch.empty_like(q)

    def fake_sdpa(q, k, v, dropout_p=0.0, is_causal=False):
        calls.append("sdpa")
        return torch.empty_like(q)

    monkeypatch.setattr(functional, "triton_attention", fake_triton)
    monkeypatch.setattr(functional.F, "scaled_dot_product_attention", fake_sdpa)

    q = torch.empty((1, 1, sequence_length, 64), device="cuda", dtype=torch.float16)
    my_ops.attention(q, q, q, causal=causal, provider="auto")

    assert calls == [expected_provider]


@CUDA_REQUIRED
@pytest.mark.parametrize("provider", ["auto", "cpp", "triton", "sdpa"])
def test_attention_providers_handle_empty_tensor(provider):
    q = torch.empty((1, 2, 0, 64), device="cuda", dtype=torch.float16)
    actual = my_ops.attention(q, q, q, provider=provider)

    assert actual.shape == q.shape
    assert actual.numel() == 0


@CUDA_REQUIRED
def test_attention_rejects_unknown_provider():
    q = torch.empty((1, 1, 17, 64), device="cuda", dtype=torch.float16)

    with pytest.raises(ValueError, match="provider 必须是"):
        my_ops.attention(q, q, q, provider="unknown")
