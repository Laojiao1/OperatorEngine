"""文件职责：实现纯 PyTorch Transformer Block 基线，以及逐项替换 OperatorEngine 算子的对照版本。"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """使用 FP32 计算均方根，再恢复输入 dtype。"""

    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states_fp32 = hidden_states.float()
        variance = hidden_states_fp32.pow(2).mean(dim=-1, keepdim=True)
        normalized = hidden_states_fp32 * torch.rsqrt(variance + self.eps)
        return (normalized * self.weight.float()).to(input_dtype)


def apply_rotary_embedding(q: torch.Tensor, k: torch.Tensor, theta: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """对 Q/K 最后一维相邻的偶数、奇数通道应用 RoPE。"""
    if q.shape != k.shape:
        raise ValueError("RoPE: q 和 k 的 shape 必须相同")

    head_dim = q.shape[-1]
    if head_dim % 2 != 0:
        raise ValueError("RoPE: head_dim 必须是偶数")

    sequence_length = q.shape[-2]
    positions = torch.arange(sequence_length, device=q.device, dtype=torch.float32)
    dimension_indices = torch.arange(0, head_dim, 2, device=q.device, dtype=torch.float32)
    inverse_frequency = 1.0 / (theta ** (dimension_indices / head_dim))
    frequencies = torch.outer(positions, inverse_frequency)

    cos = frequencies.cos()[None, None, :, :]
    sin = frequencies.sin()[None, None, :, :]

    def rotate(input: torch.Tensor) -> torch.Tensor:
        even = input[..., 0::2].float()
        odd = input[..., 1::2].float()
        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2).to(input.dtype)

    return rotate(q), rotate(k)


class TorchTransformerBlock(nn.Module):
    """纯 PyTorch、causal、pre-norm decoder block。"""

    def _attention_forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """纯 PyTorch Attention 基线。"""
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=True)

    def _linear_forward(self, hidden_states: torch.Tensor, projection: nn.Linear) -> torch.Tensor:
        """纯 PyTorch Linear 基线。"""
        return projection(hidden_states)

    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int, rms_norm_eps: float = 1e-5, rope_theta: float = 10000.0):
        super().__init__()

        if hidden_size <= 0 or num_heads <= 0 or intermediate_size <= 0:
            raise ValueError("hidden_size、num_heads 和 intermediate_size 必须大于 0")
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size 必须能被 num_heads 整除")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.rope_theta = rope_theta

        # 后续自定义 Attention 第一版只支持 D=64/128，因此基线也锁定相同范围。
        if self.head_dim not in (64, 128):
            raise ValueError("第一版 Transformer Block 只支持 head_dim=64 或 128")

        self.input_norm = RMSNorm(hidden_size, rms_norm_eps)
        self.qkv_projection = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.output_projection = nn.Linear(hidden_size, hidden_size, bias=False)

        self.post_attention_norm = RMSNorm(hidden_size, rms_norm_eps)
        self.gate_projection = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_projection = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_projection = nn.Linear(intermediate_size, hidden_size, bias=False)


    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dim() != 3:
            raise ValueError("hidden_states 必须是三维 Tensor [B, N, hidden_size]")
        if hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("hidden_states 的最后一维必须等于 hidden_size")

        batch_size, sequence_length, _ = hidden_states.shape

        # Attention 子层：Pre-Norm -> QKV -> RoPE -> SDPA -> 输出投影 -> Residual。
        residual = hidden_states
        normalized = self.input_norm(hidden_states)
        qkv = self._linear_forward(normalized, self.qkv_projection)
        q, k, v = qkv.split(self.hidden_size, dim=-1)

        q = q.reshape(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        k = k.reshape(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        v = v.reshape(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

        q, k = apply_rotary_embedding(q, k, self.rope_theta)
        attention_output = self._attention_forward(q, k, v)
        attention_output = attention_output.transpose(1, 2).contiguous().reshape(batch_size, sequence_length, self.hidden_size)

        hidden_states = residual + self._linear_forward(attention_output, self.output_projection)

        # MLP 子层：Pre-Norm -> SwiGLU -> Down projection -> Residual。
        residual = hidden_states
        normalized = self.post_attention_norm(hidden_states)
        gate = self._linear_forward(normalized, self.gate_projection)
        up = self._linear_forward(normalized, self.up_projection)
        gated = F.silu(gate) * up
        hidden_states = residual + self._linear_forward(gated, self.down_projection)

        return hidden_states

    
class OperatorEngineAttentionBlock(TorchTransformerBlock):
    """只将 PyTorch SDPA 替换为 OperatorEngine Attention，其余计算与基线完全相同。"""

    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int, attention_provider: str = "auto", rms_norm_eps: float = 1e-5, rope_theta: float = 10000.0):
        super().__init__(hidden_size, num_heads, intermediate_size, rms_norm_eps, rope_theta)

        if attention_provider not in ("auto", "cpp", "triton", "sdpa"):
            raise ValueError("attention_provider 必须是 auto、cpp、triton 或 sdpa")

        self.attention_provider = attention_provider

    def _attention_forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # 局部导入保证纯 PyTorch 基线本身不依赖 OperatorEngine 扩展。
        from my_ops import attention

        return attention(q, k, v, causal=True, provider=self.attention_provider)


class OperatorEngineGemmBlock(TorchTransformerBlock):
    """只将无 bias Linear 替换为 OperatorEngine GEMM，Attention 仍使用 PyTorch SDPA。"""

    def _linear_forward(self, hidden_states: torch.Tensor, projection: nn.Linear) -> torch.Tensor:
        # 局部导入只负责加载扩展并完成 Dispatcher 注册，纯 PyTorch 基线不依赖它。
        import my_ops

        if projection.bias is not None:
            raise ValueError("OperatorEngine GEMM 第一版只替换无 bias Linear")

        input_shape = hidden_states.shape
        flattened_input = hidden_states.reshape(-1, input_shape[-1]).contiguous()

        # Parameter 即使处于 inference_mode，requires_grad 属性仍可能为 True；
        # detach 只解除 autograd 关系，不复制权重存储。
        weight = projection.weight.detach()

        # nn.Linear 的 weight 为 [out_features, in_features]，
        # 因此使用 transpose_b=True 计算 input @ weight.T。
        output_fp32 = torch.ops.my_ops.gemm(flattened_input, weight, True)

        # 当前 GEMM 固定输出 FP32；Block 其余计算使用 FP16，因此显式恢复输入 dtype。
        output = output_fp32.to(hidden_states.dtype)
        return output.reshape(*input_shape[:-1], projection.out_features)


class OperatorEngineTransformerBlock(OperatorEngineAttentionBlock, OperatorEngineGemmBlock):
    """同时使用 OperatorEngine Attention 和 GEMM 的完整对照版本。"""

    pass


class TorchUnfusedAttentionBlock(TorchTransformerBlock):
    """使用显式 QK^T、causal mask、Softmax 和 PV 实现未融合 Attention。"""

    def _softmax_forward(self, scores: torch.Tensor) -> torch.Tensor:
        """未融合 Attention 使用的 PyTorch Softmax 基线。"""
        return torch.softmax(scores, dim=-1)

    def _attention_forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        sequence_length = q.shape[-2]
        scale = q.shape[-1] ** -0.5

        # 显式生成 Attention score，作为 fused SDPA 的未融合对照。
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        # 上三角区域代表当前 token 之后的未来位置，需要在 Softmax 前屏蔽。
        causal_mask = torch.ones(
            (sequence_length, sequence_length),
            device=q.device,
            dtype=torch.bool,
        ).triu(diagonal=1)
        scores = scores.masked_fill(causal_mask, float("-inf"))

        probabilities = self._softmax_forward(scores.contiguous())
        return torch.matmul(probabilities, v)


class OperatorEngineSoftmaxBlock(TorchUnfusedAttentionBlock):
    """仅将未融合 Attention 中的 PyTorch Softmax 替换为 OperatorEngine Softmax。"""

    def _softmax_forward(self, scores: torch.Tensor) -> torch.Tensor:
        # 局部导入负责加载扩展并完成 Dispatcher 注册。
        import my_ops

        return torch.ops.my_ops.softmax(scores, -1)