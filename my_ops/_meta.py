"""文件职责：定义自定义算子的 FakeTensor 元数据与输入契约，给 PyTorch 编译器（torch.compile）看的 Python 函数，用于静态推导输出的形状（Shape）和数据类型（Dtype），不执行 GPU 计算"""

import torch


"""
当 PyTorch 处于图追踪阶段（Tracing）或者运行契约检查 torch.library.opcheck 时，不会真正启动 GPU 计算，而是调用这个 Python 函数。
这个函数接收虚拟的 Tensor（FakeTensor，不占真实显存），验证形状、类型，并返回一个尺寸相同的新 FakeTensor。
"""
@torch.library.register_fake("my_ops::vector_add")
def _vector_add_fake(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # FakeTensor 不执行真实 CUDA kernel，但必须复现真实算子的输入约束和输出元数据。
    # 这使 torch.compile/opcheck 能在不分配真实数据的情况下推导 shape、dtype 和 device。
    torch._check(a.device == b.device, lambda: "a and b must be on the same device")
    torch._check(
        a.dtype == torch.float32 and b.dtype == torch.float32,
        lambda: "a and b must have dtype torch.float32",
    )
    torch._check(a.dim() == b.dim(), lambda: "a and b must have the same rank")
    for a_size, b_size in zip(a.shape, b.shape):
        torch._check(a_size == b_size, lambda: "a and b must have the same shape")
    torch._check(
        a.is_contiguous() and b.is_contiguous(),
        lambda: "a and b must be contiguous",
    )

    # vector_add 是 functional operator，Fake 实现同样返回一个不与输入 alias 的新 Tensor。
    return torch.empty_like(a)


@torch.library.register_fake("my_ops::softmax")
def _softmax_fake(input: torch.Tensor, dim: int) -> torch.Tensor:
    # FakeTensor 不执行 CUDA kernel，因此必须在 Python 中复现 C++ wrapper 的契约。
    torch._check(input.is_cuda, lambda: "softmax: input 必须是 CUDA Tensor")
    torch._check(input.dim() >= 1, lambda: "softmax: input 至少必须是一维 Tensor")

    ndim = input.dim()
    wrapped_dim = dim + ndim if dim < 0 else dim

    torch._check(wrapped_dim == ndim - 1, lambda: "softmax: 第一版只支持沿最后一维计算")
    torch._check(input.dtype in (torch.float16, torch.float32), lambda: "softmax: 第一版只支持 FP32 和 FP16")
    torch._check(input.is_contiguous(), lambda: "softmax: input 必须是 contiguous Tensor")
    torch._check(not input.requires_grad, lambda: "softmax: 第一版不支持 autograd")

    # empty_like 返回新的 FakeTensor，保持输入的 shape、dtype 和虚拟 device。
    return torch.empty_like(input)


@torch.library.register_fake("my_ops::gemm")
def _gemm_fake(a: torch.Tensor, b: torch.Tensor, transpose_b: bool = False) -> torch.Tensor:
    # FakeTensor 实现必须复现真实 wrapper 的 shape、dtype、device 和 layout 契约。
    torch._check(a.is_cuda, lambda: "gemm: a 必须是 CUDA Tensor")
    torch._check(b.is_cuda, lambda: "gemm: b 必须是 CUDA Tensor")
    torch._check(a.dim() == 2, lambda: "gemm: a 必须是二维 Tensor")
    torch._check(b.dim() == 2, lambda: "gemm: b 必须是二维 Tensor")
    torch._check(a.device == b.device, lambda: "gemm: a 和 b 必须位于同一 CUDA device")
    torch._check(a.dtype == b.dtype, lambda: "gemm: a 和 b 的 dtype 必须相同")
    torch._check(a.dtype in (torch.float16, torch.float32), lambda: "gemm: 第一版只支持 FP32 和 FP16")
    torch._check(a.is_contiguous(), lambda: "gemm: a 必须是 contiguous Tensor")
    torch._check(b.is_contiguous(), lambda: "gemm: b 必须是 contiguous Tensor")
    torch._check(not a.requires_grad and not b.requires_grad, lambda: "gemm: 第一版不支持 autograd")

    M = a.shape[0]
    K = a.shape[1]
    N = b.shape[0] if transpose_b else b.shape[1]
    b_k = b.shape[1] if transpose_b else b.shape[0]

    torch._check(K == b_k, lambda: "gemm: a 和 b 的 K 维度不匹配")
    torch._check(K > 0, lambda: "gemm: 第一版要求 K 大于 0")

    return torch.empty((M, N), device=a.device, dtype=torch.float32)


@torch.library.register_fake("my_ops::attention")
def _attention_fake(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False) -> torch.Tensor:
    torch._check(q.is_cuda and k.is_cuda and v.is_cuda, lambda: "attention: q、k、v 必须是 CUDA Tensor")
    torch._check(q.dim() == 4 and k.dim() == 4 and v.dim() == 4, lambda: "attention: q、k、v 必须是四维 Tensor [B, H, N, D]")

    for q_size, k_size, v_size in zip(q.shape, k.shape, v.shape):
        torch._check(q_size == k_size and q_size == v_size, lambda: "attention: q、k、v 的 shape 必须完全相同")

    torch._check(q.device == k.device and q.device == v.device, lambda: "attention: q、k、v 必须位于同一 CUDA device")
    torch._check(q.dtype == torch.float16 and k.dtype == torch.float16 and v.dtype == torch.float16, lambda: "attention: 第一版只支持 FP16")
    torch._check(q.is_contiguous() and k.is_contiguous() and v.is_contiguous(), lambda: "attention: q、k、v 必须是 contiguous Tensor")
    torch._check(not q.requires_grad and not k.requires_grad and not v.requires_grad, lambda: "attention: 第一版不支持 autograd")
    torch._check(q.shape[-1] == 64 or q.shape[-1] == 128, lambda: "attention: 第一版只支持 head_dim=64 或 128")

    return torch.empty_like(q)
