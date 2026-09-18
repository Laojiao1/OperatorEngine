"""文件职责：提供 OperatorEngine 的统一 Python 函数接口，并在 C++、Triton 和 PyTorch 后端之间调度。"""

from typing import Literal

import torch
import torch.nn.functional as F

from .attention_triton import _check_inputs, triton_attention


AttentionProvider = Literal["auto", "cpp", "triton", "sdpa"]
_SUPPORTED_PROVIDERS = ("auto", "cpp", "triton", "sdpa")


def _select_attention_provider(sequence_length: int, causal: bool) -> str:
    """根据当前 benchmark 结论返回 auto 模式的预期后端。"""
    if causal or sequence_length <= 512:
        return "triton"
    return "sdpa"


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False, provider: AttentionProvider = "auto") -> torch.Tensor:
    """执行四维 FP16 Attention forward。

    参数：
        q、k、v：形状为 [B, H, N, D] 的 contiguous CUDA FP16 Tensor。
        causal：是否应用 causal mask。
        provider：
            auto：根据当前 benchmark 结论自动选择；
            cpp：进入 torch.ops.my_ops.attention；
            triton：显式调用 Triton FlashAttention；
            sdpa：显式调用 PyTorch SDPA。
    """
    _, _, sequence_length, _ = _check_inputs(q, k, v, causal)

    if provider not in _SUPPORTED_PROVIDERS:
        raise ValueError(f"attention: provider 必须是 {_SUPPORTED_PROVIDERS} 之一，实际得到 {provider!r}")

    # 所有 provider 对空 Tensor 使用相同语义，避免向后端提交零维 grid。
    if q.numel() == 0:
        return torch.empty_like(q)

    selected_provider = _select_attention_provider(sequence_length, causal) if provider == "auto" else provider

    if selected_provider == "cpp":
        return torch.ops.my_ops.attention(q, k, v, causal)

    if selected_provider == "triton":
        return triton_attention(q, k, v, causal)

    # SDPA 默认使用 1/sqrt(D)，dropout 固定为 0，符合 inference forward 范围。
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=causal)