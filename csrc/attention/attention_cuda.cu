// 文件职责：实现 FP16 CUDA Core 与 Tensor Core Online Attention，并以 FP32 完成 Softmax 和输出累加。

#include "attention/attention.h"

#include <cuda_fp16.h>

#include <cfloat>
#include <cstddef>
#include <cmath>

#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>

#include <cstdint>
#include <limits>
#include <mma.h>

namespace my_ops {
namespace {

constexpr int kWarpSize = 32;
constexpr int kQueryRows = 16;
constexpr int kKeyCols = 32;
constexpr int kMaxHeadDim = 128;
constexpr int kThreads = kQueryRows * kWarpSize;
constexpr int kOutputValuesPerLane = (kMaxHeadDim + kWarpSize - 1) / kWarpSize;
constexpr int kKeySharedStride = kKeyCols + 1;

constexpr int kTensorCoreQueryRows = 32;
constexpr int kTensorCoreKeyCols = 32;
constexpr int kTensorCoreHeadDim = 64;
constexpr int kWmmaTile = 16;
constexpr int kTensorCoreQkWarps = (kTensorCoreQueryRows / kWmmaTile) * (kTensorCoreKeyCols / kWmmaTile);
constexpr int kTensorCorePvWarps = (kTensorCoreQueryRows / kWmmaTile) * (kTensorCoreHeadDim / kWmmaTile);
constexpr int kTensorCoreWarps = kTensorCoreQkWarps > kTensorCorePvWarps ? kTensorCoreQkWarps : kTensorCorePvWarps;
constexpr int kTensorCoreThreads = kTensorCoreWarps * kWarpSize;

static_assert(kKeyCols == kWarpSize, "每个 warp lane 必须对应一个 key");
static_assert(kThreads <= 1024, "Attention block 线程数超过 CUDA 上限");
static_assert(kTensorCoreThreads <= 1024, "Tensor Core Attention block 线程数超过 CUDA 上限");


// ----- 两大规约操作：sum，max -----
// 通过 shuffle 指令，直接绕过 shared mem，快速找出32个线程中的 sum 和 max
__device__ __forceinline__ float warp_reduce_sum(float value) {
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ float warp_reduce_max(float value) {
    #pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
    }
    return value;
}


// ===== 针对 CUDA Core 的 Flash Attention kernel =====
template <int HeadDim, bool Causal>
__global__ void flash_attention_online_kernel(
    const half* __restrict__ q,
    const half* __restrict__ k,
    const half* __restrict__ v,
    half* __restrict__ output,
    int n,
    int query_blocks,
    float scale) {

    // 第一版只实例化 HeadDim=64/128，使编译器能够展开依赖 d 的循环。
    constexpr int d = HeadDim;

    // ----- 阶段二：动态 Shared Mem 布局与转置 Padding -----

    // 2. 声明动态 Shared Memory 指针
    extern __shared__ float shared[];

    // 3. 将 Shared Memory 空间顺序切割给 Q, K, V, P 四个 Tile
    float* q_tile = shared; // [Br,d]
    float* k_tile = q_tile + kQueryRows * d;
    float* v_tile = k_tile + kKeySharedStride * d; // [Bc, d]
    // 这里只在 Shared Memory 暂存当前 tile 的概率，绝不写到 HBM 全局显存中
    float* p_tile = v_tile + kKeyCols * d; // [Br, Bc]

    // ----- 阶段三：全局内存加载与 Causal Mask 跳块优化 -----
    const int tid = threadIdx.x;
    const int warp_id = tid / kWarpSize;
    const int lane = tid % kWarpSize;

    // grid.x 同时展开 B×H 和 Query Tile。
    // 每个 block 仍然只处理一个 [N,D] 矩阵中的连续 16 行 Q。
    const int block_index = static_cast<int>(blockIdx.x);
    const int query_block_index = block_index % query_blocks;
    const int batch_head_index = block_index / query_blocks;
    const std::size_t tensor_offset = static_cast<std::size_t>(batch_head_index) * n * d;

    q += tensor_offset;
    k += tensor_offset;
    v += tensor_offset;
    output += tensor_offset;

    const int query_start = query_block_index * kQueryRows;
    const int query_index = query_start + warp_id;
    const bool valid_query = query_index < n;

    // 块内所有线程协作将 Q 矩阵加载到 Shared Memory (q_tile)
    for (int index = tid; index < kQueryRows * d; index += blockDim.x) {
        const int row = index / d;
        const int col = index % d;
        const int global_row = query_start + row;
        q_tile[index] = global_row < n ? __half2float(q[global_row * d + col]) : 0.0f;
    }
    __syncthreads();

    // 对当前 K/V tile，Online Softmax 递推为：
    //   m_new = max(m_old, max(S_tile))
    //   alpha = exp(m_old - m_new)
    //   l_new = alpha * l_old + sum(exp(S_tile - m_new))
    //   O_acc = alpha * O_acc + exp(S_tile - m_new) * V_tile
    // 循环结束后只做一次 O = O_acc / l，因此中途不需要写出 S/P。
    // lane 只负责 lane, lane+32, ... 四个输出维度，避免旧版本每线程
    // float o[64] 带来的寄存器压力与 local-memory spill。
    float row_max = -FLT_MAX;
    float row_sum = 0.0f;
    float o_reg[kOutputValuesPerLane] = {0.0f};

    // 1. 算出如果不管 Causal，全局一共有多少个 Key Tile
    const int all_key_tiles = (n + kKeyCols - 1) / kKeyCols;

    // 2. 找到当前 Block 所负责的所有 Query 中，最靠后（最未来）的那一行 Query 的后一位开区间界限
    const int last_query_exclusive = min(n, query_start + kQueryRows);

    // 3. 核心：算出当前 Block 实际上最多只需要处理到第几个 Key Tile
    const int causal_key_tiles = (last_query_exclusive + kKeyCols - 1) / kKeyCols;

    // 4. 三元表达式：如果开启了 Causal，就用剪枝后的上限；没开启就处理所有 Tile
    const int key_tiles = Causal ? causal_key_tiles : all_key_tiles;

    // 5. 最外层 Key Tile 循环：直接用计算出的 key_tiles 作为终止条件！
    for (int key_tile_index = 0; key_tile_index < key_tiles; ++key_tile_index) {
        const int key_start = key_tile_index * kKeyCols;

        for (int index = tid; index < kKeyCols * d; index += blockDim.x) {
            const int row = index / d;
            const int col = index % d;
            const int global_row = key_start + row;
            const bool valid_key = global_row < n;

            // K 在 Shared Memory 中转置为 [d, Bc+1]。这样计算 QK^T 时 一个 warp 的 32 个 lane 会读取连续 key。
            // +1 padding 消除转置写入时因 stride=32 产生的 shared-memory bank conflict
            k_tile[col * kKeySharedStride + row] = valid_key ? __half2float(k[global_row * d + col]) : 0.0f;
            v_tile[index] = valid_key ? __half2float(v[global_row * d + col]) : 0.0f;
        }
        __syncthreads();

        if (valid_query) {
            // lane 与当前 tile 的 key 一一对应。旧映射按 key 串行推进，每个 score 都要做一次 warp reduction；现在 32 个 score 并行，整个 tile 只为 Softmax 的 max/sum 各规约一次。
            const int key_index = key_start + lane;
            const bool keep = key_index < n && (!Causal || key_index <= query_index);

            // ----- 阶段四：最内层点积计算与 4 路指令级并行（ILP）展开 -----
            float score = -FLT_MAX;
            if (keep) {
                // 优化传统单累加器的流水线阻塞的问题，采用 4 路累加器打断依赖链
                // 当算 dot3 时，dot0 的结果正好算完出来
                float dot0 = 0.0f;
                float dot1 = 0.0f;
                float dot2 = 0.0f;
                float dot3 = 0.0f;
                int inner = 0;
                for (; inner + 3 < d; inner += 4) {
                    dot0 += q_tile[warp_id * d + inner + 0] *
                            k_tile[(inner + 0) * kKeySharedStride + lane];
                    dot1 += q_tile[warp_id * d + inner + 1] *
                            k_tile[(inner + 1) * kKeySharedStride + lane];
                    dot2 += q_tile[warp_id * d + inner + 2] *
                            k_tile[(inner + 2) * kKeySharedStride + lane];
                    dot3 += q_tile[warp_id * d + inner + 3] *
                            k_tile[(inner + 3) * kKeySharedStride + lane];
                }
                // 两两归并累加：循环结束后，把 4 个局部累加器的结果加起来，得到完整的向量点积
                float dot = (dot0 + dot1) + (dot2 + dot3);

                // 处理剩余维数
                for (; inner < d; ++inner) {
                    dot += q_tile[warp_id * d + inner] *
                           k_tile[inner * kKeySharedStride + lane];
                }
                // 乘上缩放因子
                score = dot * scale;
            }

            // ----- 阶段五：Online Softmax 递推更新、o_reg 缩放与最终归一化写出 -----
            // 1. 求当前 Tile 的局部最大值，并广播给整个 Warp
            float tile_max = warp_reduce_max(score);
            tile_max = __shfl_sync(0xffffffffu, tile_max, 0);

            // 2. 更新全局最大值并计算旧状态缩放因子 alpha
            const float new_max = fmaxf(row_max, tile_max);
            const float alpha = expf(row_max - new_max);

            // 3. 计算当前 Tile 概率，存入 p_tile，并求当前 Tile 分母和
            const float probability = keep ? expf(score - new_max) : 0.0f;
            p_tile[warp_id * kKeyCols + lane] = probability;
            float tile_sum = warp_reduce_sum(probability);
            tile_sum = __shfl_sync(0xffffffffu, tile_sum, 0);
            const float new_sum = alpha * row_sum + tile_sum;

            __syncwarp();

            // 4. 用 alpha 重缩放旧输出 o_reg，并加上 P_tile * V_tile
            #pragma unroll
            for (int slot = 0; slot < kOutputValuesPerLane; ++slot) {
                const int output_col = lane + slot * kWarpSize;
                if (output_col < d) {
                    float pv = 0.0f;
                    for (int key_offset = 0; key_offset < kKeyCols; ++key_offset) {
                        pv += p_tile[warp_id * kKeyCols + key_offset] *
                              v_tile[key_offset * d + output_col];
                    }
                    o_reg[slot] = alpha * o_reg[slot] + pv;
                }
            }
            row_max = new_max;
            row_sum = new_sum;
        }
        // 所有 warp 用完 K/V 后才能覆盖 Shared Memory 进入下一 tile。
        __syncthreads();
    }

    // 当最外层遍历完全部 K/V Tile 后，在循环体外面做全流程唯一一次归一化除法
    if (valid_query) {
        const float inverse_sum = 1.0f / row_sum;
        #pragma unroll
        for (int slot = 0; slot < kOutputValuesPerLane; ++slot) {
            const int output_col = lane + slot * kWarpSize;
            if (output_col < d) {
                // 将未归一化的累加器除以最终的总分母分母，得出最终归一化好的注意力结果，写回 HBM 全局显存
                output[query_index * d + output_col] = __float2half(o_reg[slot] * inverse_sum);
            }
        }
    }
}

// ===== 针对 Tensor Core 的 Flash Attention kernel =====
template <bool Causal>
__global__ void flash_attention_tensor_core_kernel(
    const half* __restrict__ q,
    const half* __restrict__ k,
    const half* __restrict__ v,
    half* __restrict__ output,
    int n,
    int query_blocks,
    float scale) {

    // ----- 阶段 1：WMMA 维度常量推导 与 Shared Memory 对齐布局 -----
    namespace wmma = nvcuda::wmma;

    constexpr int Br = kTensorCoreQueryRows;
    constexpr int Bc = kTensorCoreKeyCols;
    constexpr int D = kTensorCoreHeadDim;
    constexpr int WmmaTile = kWmmaTile;
    constexpr int Warps = kTensorCoreWarps;
    constexpr int QueryTiles = Br / WmmaTile;
    constexpr int KeyTiles = Bc / WmmaTile;
    constexpr int OutputTiles = D / WmmaTile;
    constexpr int QkWarps = QueryTiles * KeyTiles;

    // half 的 WMMA leading dimension 必须保持 16-byte 粒度，且需要为16的整数倍，让相邻行错开从而解决Bank冲突，float store 同理。所以半字需要 +8，字需要 +4
    constexpr int QkvStride = D + 8;
    constexpr int PStride = Bc + 8;
    constexpr int ScoreStride = Bc + 4;
    constexpr int OutputStride = D + 4;

    // Q/K/V/P 用 half 降低 Shared Memory 流量并满足 Tensor Core 输入要求；
    // score、Online 状态与 O 始终用 FP32，避免跨 tile 递推不断损失精度。
    __shared__ __align__(16) half q_tile[Br * QkvStride];
    __shared__ __align__(16) half k_tile[Bc * QkvStride];
    __shared__ __align__(16) half v_tile[Bc * QkvStride];
    __shared__ __align__(16) half p_tile[Br * PStride];
    __shared__ __align__(16) float scores[Br * ScoreStride];
    __shared__ __align__(16) float output_acc[Br * OutputStride];
    __shared__ __align__(16) float pv_tile[Br * OutputStride];
    __shared__ float row_max[Br];
    __shared__ float row_sum[Br];
    __shared__ float alpha[Br];

    // ----- 阶段 2：全局内存加载、数据格式转换与状态初始化 ----- 
    const int tid = threadIdx.x;
    const int warp_id = tid / kWarpSize;
    const int lane = tid % kWarpSize;
    
    const int block_index = static_cast<int>(blockIdx.x);
    const int query_block_index = block_index % query_blocks;
    const int batch_head_index = block_index / query_blocks;
    const std::size_t tensor_offset = static_cast<std::size_t>(batch_head_index) * n * D;

    q += tensor_offset;
    k += tensor_offset;
    v += tensor_offset;
    output += tensor_offset;

    const int query_start = query_block_index * Br;

    // 1. 线程协作将全局内存中的 FP16 Q Tile 搬入 Shared Memory。
    for (int index = tid; index < Br * D; index += blockDim.x) {
        const int row = index / D;
        const int col = index % D;
        const int global_row = query_start + row;
        q_tile[row * QkvStride + col] = global_row < n ? q[global_row * D + col] : __float2half_rn(0.0f);
        output_acc[row * OutputStride + col] = 0.0f;
    }
    // 2. Online Softmax 状态数组初始化 (维持 FP32 精度)
    if (tid < Br) {
        row_max[tid] = -FLT_MAX;
        row_sum[tid] = 0.0f;
        alpha[tid] = 0.0f;
    }
    __syncthreads();

    // ----- 阶段 3：K/V 协作加载与 Causal 跳块 -----
    // 跳块计算（与之前 CUDA Core 实现的逻辑一致）
    const int all_key_tiles = (n + Bc - 1) / Bc;
    const int last_query_exclusive = min(n, query_start + Br);
    const int causal_key_tiles = (last_query_exclusive + Bc - 1) / Bc;
    const int key_tiles = Causal ? causal_key_tiles : all_key_tiles;

    // 协作搬运
    for (int key_tile_index = 0; key_tile_index < key_tiles; ++key_tile_index) {
        const int key_start = key_tile_index * Bc;
        for (int index = tid; index < Bc * D; index += blockDim.x) {
            const int row = index / D;
            const int col = index % D;
            const int global_row = key_start + row;
            const bool valid = global_row < n;
            k_tile[row * QkvStride + col] = valid ? k[global_row * D + col] : __float2half_rn(0.0f);
            v_tile[row * QkvStride + col] = valid ? v[global_row * D + col] : __float2half_rn(0.0f);
        }
        __syncthreads();

        // ----- 阶段 4：用 Tensor Core WMMA 加速算子 1（计算 $S = Q \cdot K^T$ 得分矩阵） -----
        // 每个 warp 产生一个 16x16 S 子块。K 原本按 [key,d] 存储，
        // 作为 col-major 的 [d,key] 矩阵读取即可得到 K^T 视图。
        if (warp_id < QkWarps) {
            // 子块坐标计算
            const int query_offset = (warp_id / KeyTiles) * WmmaTile; // 行 Q
            const int key_offset = (warp_id % KeyTiles) * WmmaTile; // 列 K

            // Fragment：一个由 Warp 内 32 个线程共同分担持有的“不透明矩阵寄存器块”
            wmma::fragment<wmma::accumulator, WmmaTile, WmmaTile, WmmaTile, float> score_fragment;
            // 把这 16 * 16 的矩阵块清零
            wmma::fill_fragment(score_fragment, 0.0f);

            for (int inner = 0; inner < D; inner += WmmaTile) { // inner 每次跳 16
                wmma::fragment<wmma::matrix_a, WmmaTile, WmmaTile, WmmaTile,
                               half, wmma::row_major> q_fragment;
                wmma::fragment<wmma::matrix_b, WmmaTile, WmmaTile, WmmaTile,
                               half, wmma::col_major> k_fragment; // 这里直接按照列优先存储，实现转置操作
                // Warp 内 32 个线程并发从 Shared Memory 中各自加载一部分数据到各自的寄存器中，组合拼装出 16*16 的 q_fragment 和 k_fragment。传入的步长 QkvStride (72) 确保了 16 字节对齐
                // 注意这里的指针二维偏移计算，满足：BasePtr + RowIndex * Stride + ColIndex
                wmma::load_matrix_sync(
                    q_fragment, q_tile + query_offset * QkvStride + inner,
                    QkvStride);
                wmma::load_matrix_sync(
                    k_fragment, k_tile + key_offset * QkvStride + inner,
                    QkvStride);

                // 执行乘累加 score_fragment = q_fragment * k_fragment + score_fragment
                wmma::mma_sync(
                    score_fragment, q_fragment, k_fragment, score_fragment);
            }
            // 把处于寄存器 Fragment 中的 16*16 结果矩阵写回 Shared Memory 的 scores 数组中（步长为 ScoreStride = 36）
            wmma::store_matrix_sync(
                scores + query_offset * ScoreStride + key_offset,
                score_fragment, ScoreStride,
                wmma::mem_row_major);
        }
        __syncthreads();

        // ----- 阶段 5：Softmax 状态递推、细粒度 Causal Mask 与 P_tile(FP16) 生成 -----
        // 四个 warp 各处理 4 行；lane 与 Bc=32 的 key 一一对应
        // 这里只做两次 shuffle reduction，softmax 状态仍保持 FP32
        #pragma unroll
        for (int group = 0; group < Br / Warps; ++group) {
            const int row = warp_id + group * Warps;
            const int query_index = query_start + row;
            const int key_index = key_start + lane;
            const bool valid_query = query_index < n;

            // 细粒度 Causal Mask 与缩放
            const bool keep = valid_query && key_index < n &&
                              (!Causal || key_index <= query_index);
            const float score =
                keep ? scores[row * ScoreStride + lane] * scale : -FLT_MAX;

            // Online Softmax 规约与 p_tile (FP16) 转换写入
            if (valid_query) {
                // Warp 内 Shuffle 规约：
                // 32 个线程求出当前行的最大值 tile_max，并通过 __shfl_sync 广播给 Warp 内所有线程
                float tile_max = warp_reduce_max(score);
                tile_max = __shfl_sync(0xffffffffu, tile_max, 0);

                // 状态更新：算出 m_new，缩放因子 alpha，和 P
                const float new_max = fmaxf(row_max[row], tile_max);
                const float row_alpha = expf(row_max[row] - new_max);
                const float probability = keep ? expf(score - new_max) : 0.0f;

                // 关键点：写入 p_tile 时转为 FP16
                // 因为下一个阶段，p_tile 将作为矩阵 A 喂给 Tensor Core 算 PV，而 Tensor Core 必须要求输入是 FP16
                p_tile[row * PStride + lane] = __float2half_rn(probability);

                float tile_sum = warp_reduce_sum(probability);
                tile_sum = __shfl_sync(0xffffffffu, tile_sum, 0);
                
                // 确保当前Warp内所有32个线程都读完了旧的row_max[row]状态，并完成了p_tile的写入
                __syncwarp(); 

                // lane 0 将新数据写回 Shared Mem 中
                if (lane == 0) {
                    alpha[row] = row_alpha;
                    row_sum[row] = row_alpha * row_sum[row] + tile_sum;
                    row_max[row] = new_max;
                }
            } else {
                p_tile[row * PStride + lane] = __float2half_rn(0.0f);
                if (lane == 0) {
                    alpha[row] = 0.0f;
                }
            }
        }
        __syncthreads();

        // ----- 阶段 6：用 WMMA 加速算子 2（计算 PV 矩阵）以及全局 Online Rescale 重缩放 -----
        const int query_offset = (warp_id / OutputTiles) * WmmaTile;
        const int output_col = (warp_id % OutputTiles) * WmmaTile;
        wmma::fragment<wmma::accumulator, WmmaTile, WmmaTile, WmmaTile, float>
            pv_fragment;
        wmma::fill_fragment(pv_fragment, 0.0f);
        for (int inner = 0; inner < Bc; inner += WmmaTile) {
            wmma::fragment<wmma::matrix_a, WmmaTile, WmmaTile, WmmaTile,
                           half, wmma::row_major> p_fragment;
            wmma::fragment<wmma::matrix_b, WmmaTile, WmmaTile, WmmaTile,
                           half, wmma::row_major> v_fragment;
            wmma::load_matrix_sync(
                p_fragment, p_tile + query_offset * PStride + inner, PStride);
            wmma::load_matrix_sync(
                v_fragment, v_tile + inner * QkvStride + output_col,
                QkvStride);
            wmma::mma_sync(pv_fragment, p_fragment, v_fragment, pv_fragment);
        }
        wmma::store_matrix_sync(
            pv_tile + query_offset * OutputStride + output_col,
            pv_fragment, OutputStride,
            wmma::mem_row_major);
        __syncthreads();

        // alpha 每行不同，无法直接作用于 WMMA fragment
        // 所以需要存回 Shared Memory 后逐元素完成 Online rescale，再进入下一个 K/V tile
        for (int index = tid; index < Br * D; index += blockDim.x) {
            const int row = index / D;
            const int col = index % D;
            output_acc[row * OutputStride + col] =
                alpha[row] * output_acc[row * OutputStride + col] +
                pv_tile[row * OutputStride + col];
        }
        __syncthreads();
    }

    // 代码在 Kernel 底部做全流程唯一一次除法归一化，并写回全局显存
    for (int index = tid; index < Br * D; index += blockDim.x) {
        const int row = index / D;
        const int col = index % D;
        const int global_row = query_start + row;
        if (global_row < n) {
            output[global_row * D + col] = 
                __float2half_rn(output_acc[row * OutputStride + col] / row_sum[row]);
        }
    }
}


template <int HeadDim, bool Causal>
void launch_attention_specialization(const half* q, const half* k, const half* v, half* output, int n, int query_blocks, int grid_blocks, std::size_t shared_bytes, float scale, cudaStream_t stream) {
    flash_attention_online_kernel<HeadDim, Causal><<<grid_blocks, kThreads, shared_bytes, stream>>>(q, k, v, output, n, query_blocks, scale);
}

template <bool Causal>
void launch_attention_tensor_core_specialization(const half* q, const half* k, const half* v, half* output, int n, int query_blocks, int grid_blocks, float scale, cudaStream_t stream) {
    flash_attention_tensor_core_kernel<Causal><<<grid_blocks, kTensorCoreThreads, 0, stream>>>(q, k, v, output, n, query_blocks, scale);
}

}  // namespace

// 公开 launcher
void launch_attention_cuda_core(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, at::Tensor& output, bool causal, cudaStream_t stream) {
    const std::int64_t B = q.size(0);
    const std::int64_t H = q.size(1);
    const std::int64_t N = q.size(2);
    const std::int64_t D = q.size(3);

    TORCH_CHECK(N <= std::numeric_limits<int>::max(), "attention: sequence length 超过 int 支持范围");

    const std::int64_t query_blocks_64 = (N + kQueryRows - 1) / kQueryRows;
    const std::int64_t grid_blocks_64 = B * H * query_blocks_64;

    TORCH_CHECK(query_blocks_64 <= std::numeric_limits<int>::max(), "attention: query tile 数量超过 int 支持范围");
    TORCH_CHECK(grid_blocks_64 <= std::numeric_limits<int>::max(), "attention: CUDA grid.x 超过当前实现的支持范围");

    const int n = static_cast<int>(N);
    const int head_dim = static_cast<int>(D);
    const int query_blocks = static_cast<int>(query_blocks_64);
    const int grid_blocks = static_cast<int>(grid_blocks_64);

    const std::size_t shared_bytes = static_cast<std::size_t>(
        kQueryRows * head_dim +
        kKeySharedStride * head_dim +
        kKeyCols * head_dim +
        kQueryRows * kKeyCols
    ) * sizeof(float);

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    const half* q_ptr = reinterpret_cast<const half*>(q.const_data_ptr<at::Half>());
    const half* k_ptr = reinterpret_cast<const half*>(k.const_data_ptr<at::Half>());
    const half* v_ptr = reinterpret_cast<const half*>(v.const_data_ptr<at::Half>());
    half* output_ptr = reinterpret_cast<half*>(output.mutable_data_ptr<at::Half>());

    if (head_dim == 64) {
        if (causal) {
            launch_attention_specialization<64, true>(q_ptr, k_ptr, v_ptr, output_ptr, n, query_blocks, grid_blocks, shared_bytes, scale, stream);
        } else {
            launch_attention_specialization<64, false>(q_ptr, k_ptr, v_ptr, output_ptr, n, query_blocks, grid_blocks, shared_bytes, scale, stream);
        }
    } else {
        if (causal) {
            launch_attention_specialization<128, true>(q_ptr, k_ptr, v_ptr, output_ptr, n, query_blocks, grid_blocks, shared_bytes, scale, stream);
        } else {
            launch_attention_specialization<128, false>(q_ptr, k_ptr, v_ptr, output_ptr, n, query_blocks, grid_blocks, shared_bytes, scale, stream);
        }
    }

    // 只检查异步 launch 错误，不在算子内部执行全设备同步。
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void launch_attention_tensor_core(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, at::Tensor& output, bool causal, cudaStream_t stream) {
    const std::int64_t B = q.size(0);
    const std::int64_t H = q.size(1);
    const std::int64_t N = q.size(2);
    const std::int64_t D = q.size(3);

    TORCH_CHECK(D == kTensorCoreHeadDim, "attention: Tensor Core 路径只支持 head_dim=64");
    TORCH_CHECK(N <= std::numeric_limits<int>::max(), "attention: sequence length 超过 int 支持范围");

    const std::int64_t query_blocks_64 = (N + kTensorCoreQueryRows - 1) / kTensorCoreQueryRows;
    const std::int64_t grid_blocks_64 = B * H * query_blocks_64;

    TORCH_CHECK(query_blocks_64 <= std::numeric_limits<int>::max(), "attention: Tensor Core query tile 数量超过 int 支持范围");
    TORCH_CHECK(grid_blocks_64 <= std::numeric_limits<int>::max(), "attention: Tensor Core CUDA grid.x 超过当前实现的支持范围");

    const int n = static_cast<int>(N);
    const int query_blocks = static_cast<int>(query_blocks_64);
    const int grid_blocks = static_cast<int>(grid_blocks_64);
    const float scale = 1.0f / std::sqrt(static_cast<float>(kTensorCoreHeadDim));

    const half* q_ptr = reinterpret_cast<const half*>(q.const_data_ptr<at::Half>());
    const half* k_ptr = reinterpret_cast<const half*>(k.const_data_ptr<at::Half>());
    const half* v_ptr = reinterpret_cast<const half*>(v.const_data_ptr<at::Half>());
    half* output_ptr = reinterpret_cast<half*>(output.mutable_data_ptr<at::Half>());

    if (causal) {
        launch_attention_tensor_core_specialization<true>(q_ptr, k_ptr, v_ptr, output_ptr, n, query_blocks, grid_blocks, scale, stream);
    } else {
        launch_attention_tensor_core_specialization<false>(q_ptr, k_ptr, v_ptr, output_ptr, n, query_blocks, grid_blocks, scale, stream);
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace my_ops
