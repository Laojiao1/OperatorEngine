"""文件职责：实现 OperatorEngine 的 Triton FlashAttention 候选后端，供显式调用和后续统一 selector 使用。"""

import math

import torch
import triton
import triton.language as tl

@triton.jit
def _attention_triton_kernel(
    q_ptr, k_ptr, v_ptr, output_ptr,
    sequence_length,
    head_dim,
    num_heads,
    stride_qb, stride_qh, stride_qn, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_on, stride_od,
    scale,
    CAUSAL: tl.constexpr,
    SKIP_CAUSAL_FUTURE_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """一个 Program 固定一个 batch/head/query tile，并遍历所有 K/V tile。"""
    # Grid：(query tile, batch*head), Key tile 是 Program 内部循环
    query_block_idx = tl.program_id(0)
    batch_head_idx = tl.program_id(1) # B*H

    # 把 B*H 映射回 batch 和 head。
    batch_idx = batch_head_idx // num_heads
    head_idx = batch_head_idx % num_heads

    query_offsets = query_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    feature_offsets = tl.arange(0, BLOCK_D)

    query_mask = query_offsets < sequence_length
    feature_mask = feature_offsets < head_dim

    q_base = q_ptr + batch_idx * stride_qb + head_idx * stride_qh
    k_base = k_ptr + batch_idx * stride_kb + head_idx * stride_kh
    v_base = v_ptr + batch_idx * stride_vb + head_idx * stride_vh

    # 加载数据 Q tile [BLOCK_M, BLOCK_D] 在所有 K/V 循环中复用，只加载一次
    q_ptrs = q_base + query_offsets[:, None] * stride_qn + feature_offsets[None, :] * stride_qd
    q_tile = tl.load(q_ptrs,
        mask=query_mask[:, None] & feature_mask[None, :],
        other=0.0
    )

    # 每个 query 独立维护在线 Softmax 状态：
    # m = max(score)，l = sum(exp(score-m))，acc = sum(exp(score - m) * V)。
    m_running_max = tl.where(query_mask, -float("inf"), 0.0).to(tl.float32)
    denominator = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    # 处理因果跳块
    key_block_count = tl.cdiv(sequence_length, BLOCK_N)
    if CAUSAL:
        if SKIP_CAUSAL_FUTURE_TILES:
            # 只遍历起点不超过当前 Query Tile 最大位置的 Key Tile
            query_tile_end = tl.minimum(
                (query_block_idx + 1) * BLOCK_M,
                sequence_length,
            )
            key_block_count = tl.cdiv(query_tile_end, BLOCK_N)

    key_offsets_in_block = tl.arange(0, BLOCK_N)
    for key_block_idx in range(0, key_block_count):
        key_offsets = key_block_idx * BLOCK_N + key_offsets_in_block
        key_mask = key_offsets < sequence_length

        # 通过索引广播直接加载 K^T tile [BLOCK_D,BLOCK_N]
        k_ptrs = (
            k_base
            + feature_offsets[:, None] * stride_kd
            + key_offsets[None, :] * stride_kn
        )
        v_ptrs = (
            v_base
            + key_offsets[:, None] * stride_vn
            + feature_offsets[None, :] * stride_vd
        )

        # 加载数据 K^T tile 和 v tile
        k_T_tile = tl.load(
            k_ptrs,
            mask=feature_mask[:, None] & key_mask[None, :],
            other=0.0,
        )
        v_tile = tl.load(
            v_ptrs,
            mask=key_mask[:, None] & feature_mask[None, :],
            other=0.0,
        )

        # 计算 Score 在当前循环直接计算出结果，不写回显存
        scores = tl.dot(q_tile, k_T_tile, input_precision="ieee") * scale
        score_mask = query_mask[:, None] & key_mask[None, :]

        # 处理因果跳块
        if CAUSAL:
            # 行 i query，列 j key, 只保留 j <= i
            score_mask = score_mask & (key_offsets[None, :] <= query_offsets[:, None])
        scores = tl.where(score_mask, scores, -float("inf"))

        # 维护最大值 m_max
        m_block_max = tl.max(scores, axis=1)
        m_new_max = tl.maximum(m_running_max, m_block_max)
        # 最大值变化后，旧 l/acc 必须乘同一个 alpha，切换到新基准
        alpha = tl.exp(m_running_max - m_new_max)

        # 计算当前块的未归一化权重
        numerator = tl.exp(scores - m_new_max[:, None])
        numerator = tl.where(score_mask, numerator, 0.0)

        # l_new = l_old * alpha + sum(numerator)
        denominator = denominator * alpha + tl.sum(numerator, axis=1)
        acc = acc * alpha[:, None]

        # 第一版输入固定为 FP16；P 转为 FP16 进入 Tensor Core，结果继续累加到 FP32 acc。
        acc = tl.dot(numerator.to(tl.float16), v_tile, acc)
        m_running_max = m_new_max

    # acc 是未归一化加权和，除以 l 后得到 output = softmax(score)@V
    safe_denominator = tl.where(denominator > 0.0, denominator, 1.0)
    output_tile = acc / safe_denominator[:, None]

    output_ptrs = (
        output_ptr
        + batch_idx * stride_ob
        + head_idx * stride_oh
        + query_offsets[:, None] * stride_on
        + feature_offsets[None, :] * stride_od
    )
    tl.store(
        output_ptrs,
        output_tile,
        mask=query_mask[:, None] & feature_mask[None, :]
    )


"""根据当前实验结论决定正式入口是否启用因果跳块"""
def _should_skip_causal_future_tiles(
    sequence_length: int,
    causal: bool,
) -> bool:
    return causal and sequence_length >= 512


def _check_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> tuple[int, int, int, int]:
    if not isinstance(causal, bool):
        raise TypeError("attention: causal 必须是 bool")

    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"attention: {name} 必须是 torch.Tensor")
        if not tensor.is_cuda:
            raise ValueError(f"attention: {name} 必须是 CUDA Tensor")
        if tensor.dim() != 4:
            raise ValueError(f"attention: {name} 必须是四维 Tensor [B, H, N, D]")

    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("attention: q、k、v 的 shape 必须完全相同")
    if q.device != k.device or q.device != v.device:
        raise ValueError("attention: q、k、v 必须位于同一 CUDA device")
    if q.dtype != torch.float16 or k.dtype != torch.float16 or v.dtype != torch.float16:
        raise TypeError("attention: Triton 第一版只支持 FP16")
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        raise ValueError("attention: q、k、v 必须是 contiguous Tensor")
    if q.requires_grad or k.requires_grad or v.requires_grad:
        raise ValueError("attention: 第一版不支持 autograd")

    batch_size, num_heads, sequence_length, head_dim = q.shape
    if head_dim not in (64, 128):
        raise ValueError("attention: 第一版只支持 head_dim=64 或 128")

    return batch_size, num_heads, sequence_length, head_dim


"""使用已验证的输入元数据启动 Kernel，避免不同入口重复验证"""
def triton_attention(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    causal: bool = False) -> torch.Tensor:
    batch_size, num_heads, sequence_length, head_dim = _check_inputs(q, k, v, causal)
    
    output = torch.empty_like(q)
    if output.numel() == 0:
        return output

    block_m = 64
    block_n = 32
    block_d = triton.next_power_of_2(head_dim)
    skip_causal_future_tiles = _should_skip_causal_future_tiles(sequence_length, causal)

    grid = (
        triton.cdiv(sequence_length, block_m),
        batch_size * num_heads,
    )

    # device context 与 current stream 都由 PyTorch/Triton 当前上下文继承。
    with torch.cuda.device(q.device):
        _attention_triton_kernel[grid](
            q,
            k,
            v,
            output,
            sequence_length,
            head_dim,
            num_heads,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            v.stride(3),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            output.stride(3),
            1.0 / math.sqrt(head_dim),
            CAUSAL=causal,
            SKIP_CAUSAL_FUTURE_TILES=skip_causal_future_tiles,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            BLOCK_D=block_d,
        )

    return output
