"""文件职责：提供从阶段三旧算法迁移、兼容当前 Triton 的 Softmax benchmark baseline。"""

import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    # 一个 Triton Program 承包一整行，与阶段三旧实现保持相同映射
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # 越界位置填充负无穷，不影响后续 Max 和 ExpSum
    row = tl.load(row_start_ptr + col_offsets, mask=mask, other=-float("inf"))
    row_max = tl.max(row, axis=0)
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    tl.store(output_row_start_ptr + col_offsets, softmax_output, mask=mask)


def softmax(input: torch.Tensor) -> torch.Tensor:
    """沿二维 CUDA Tensor 的最后一维执行阶段三 Triton Softmax。"""
    if not input.is_cuda or input.ndim != 2:
        raise ValueError("Triton benchmark baseline 只接受二维 CUDA Tensor")
    if input.dtype not in (torch.float16, torch.float32):
        raise TypeError("Triton benchmark baseline 只支持 FP16 和 FP32")
    if not input.is_contiguous():
        raise ValueError("Triton benchmark baseline 要求 contiguous Tensor")

    output = torch.empty_like(input)
    if input.numel() == 0:
        return output

    n_rows, n_cols = input.shape
    block_size = triton.next_power_of_2(n_cols)
    if block_size > 65536:
        raise ValueError("最后一维超过当前 Triton单 Program 的支持范围")

    softmax_kernel[(n_rows,)](output, input, input.stride(0), output.stride(0), n_cols, BLOCK_SIZE=block_size)
    return output
