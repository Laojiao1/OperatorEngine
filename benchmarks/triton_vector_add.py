"""文件职责：提供从阶段三旧算法迁移、兼容当前 Triton 的 benchmark baseline。"""

import torch
import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(
    a_ptr,
    b_ptr,
    output_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    # 每个 Triton program 负责一个连续 tile，与阶段三旧实现保持相同映射。
    program_id = tl.program_id(axis=0)
    offsets = program_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)

    # 当前 Triton 的 tl.store 不接受 other；同一 mask 已足以保护尾块写入。
    tl.store(output_ptr + offsets, a + b, mask=mask)


def add(a: torch.Tensor, b: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    """以固定 tile 大小启动迁移后的 Triton Vector Add。"""
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("a 和 b 必须位于 CUDA 设备")
    if a.device != b.device or a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError("a 和 b 的 device、shape 和 dtype 必须一致")
    if a.dtype != torch.float32:
        raise TypeError("benchmark baseline 仅支持 torch.float32")
    if not a.is_contiguous() or not b.is_contiguous():
        raise ValueError("a 和 b 必须连续")

    output = torch.empty_like(a)
    if a.numel() == 0:
        return output

    # cdiv 覆盖全部元素，最后一个 program 依靠 kernel 内 mask 处理越界 lane。
    grid = (triton.cdiv(a.numel(), block_size),)
    vector_add_kernel[grid](
        a, b, output, a.numel(), BLOCK_SIZE=block_size
    )
    return output
