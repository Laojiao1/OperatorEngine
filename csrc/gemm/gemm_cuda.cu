// 文件职责：实现 GEMM 的 CUDA kernel、dtype dispatch、kernel selector 和 launch error 检查。

#include "gemm/gemm.h"

#include <cstddef>
#include <cstdint>
#include <limits>

#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>


namespace my_ops {
namespace {

constexpr int TILE_SIZE = 32;

using namespace nvcuda;


// 一个 Thread Block 计算 C 中 128x128 的 tile，并沿 K 轴每次推进 32。
constexpr int BM = 128;
constexpr int BN = 128;
constexpr int BK = 32;

// 空间换时间：
// 通过 NCU 性能分析发现 B fragment 的 LDSM.16.MT88.4 存在严重 Bank Conflict。
// 将行跨度从 128 改为 136 个 half，使相邻行的 Bank 起点产生偏移。
// 136 仍是 8 的倍数，满足 WMMA API 的对齐约束。
constexpr int BN_PADDED = BN + 8;

// 当前 WMMA API 使用 m16n16k16：一个 warp 协作完成一个 16x16 输出 tile。
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// Small/Medium GEMM 使用更小的输出 tile，增加可并行 Thread Block 数量。
constexpr int SMALL_BM = 64;
constexpr int SMALL_BN = 64;
constexpr int SMALL_BK = 32;

// 与大 tile 相同，为 B 的 Shared Memory 行跨度增加 8 个 half，错开 Bank 起点。
constexpr int SMALL_BN_PADDED = SMALL_BN + 8;

// cp.async 绕过寄存器，实现 Global Memory 到 Shared Memory 的硬件级异步搬运。
// 发射后 warp 可以继续执行独立的 HMMA 指令，直到显式 wait 时才要求拷贝完成。
__device__ __forceinline__ void copy_global_to_shared_async_16(half* shared_dst, const half* global_src) {
    const unsigned shared_address = static_cast<unsigned>(__cvta_generic_to_shared(shared_dst));

    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :
        : "r"(shared_address), "l"(global_src)
        : "memory");
}


__device__ __forceinline__ void commit_async_copy_group() {
    // 当前线程此前发射的 cp.async 被归入同一个等待组。
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}


__device__ __forceinline__ void wait_for_all_async_copies() {
    // wait_all 只保证本线程发射的异步拷贝完成；随后仍需 __syncthreads，
    // 才能保证整个 Block 都完成搬运并安全读取完整 tile。
    asm volatile("cp.async.wait_all;\n" ::: "memory");
}


__device__ __forceinline__ void load_aligned_tile_async(half* As_stage, half* Bs_stage, const half* A, const half* B, int N, int K, int bk, int block_m, int block_n, int tid) {
    // 每个线程搬运 A/B 各 16 个 half，每一侧拆成两条 16-byte cp.async。
    const int a_load_row = tid / 2;
    const int a_load_col = (tid % 2) * 16;
    const std::size_t a_index = static_cast<std::size_t>(block_m * BM + a_load_row) * K + bk + a_load_col;

    const int b_load_row = tid / 8;
    const int b_load_col = (tid % 8) * 16;
    const std::size_t b_index = static_cast<std::size_t>(bk + b_load_row) * N + block_n * BN + b_load_col;

    #pragma unroll
    for (int vec = 0; vec < 16; vec += 8) {
        copy_global_to_shared_async_16(&As_stage[a_load_row * BK + a_load_col + vec], &A[a_index + vec]);
        copy_global_to_shared_async_16(&Bs_stage[b_load_row * BN_PADDED + b_load_col + vec], &B[b_index + vec]);
    }

    commit_async_copy_group();
}


__device__ __forceinline__ void load_small_aligned_tile_async(half* As_stage, half* Bs_stage, const half* A, const half* B, int N, int K, int bk, int block_m, int block_n, int tid) {
    // A tile 为 64x32，共 2048 个 half。128 个线程每线程搬运 16 个 half。
    const int a_load_row = tid / 2;
    const int a_load_col = (tid % 2) * 16;
    const std::size_t a_index = static_cast<std::size_t>(block_m * SMALL_BM + a_load_row) * K + bk + a_load_col;

    // B tile 为 32x64，共 2048 个 half。128 个线程每线程搬运 16 个 half。
    const int b_load_row = tid / 4;
    const int b_load_col = (tid % 4) * 16;
    const std::size_t b_index = static_cast<std::size_t>(bk + b_load_row) * N + block_n * SMALL_BN + b_load_col;

    #pragma unroll
    for (int vec = 0; vec < 16; vec += 8) {
        copy_global_to_shared_async_16(&As_stage[a_load_row * SMALL_BK + a_load_col + vec], &A[a_index + vec]);
        copy_global_to_shared_async_16(&Bs_stage[b_load_row * SMALL_BN_PADDED + b_load_col + vec], &B[b_index + vec]);
    }

    commit_async_copy_group();
}

/**
 * C = A * B。A/B 为 row-major FP16，C 为 row-major FP32。
 *
 * Aligned=true 是规则尺寸快路径：编译期删除边界分支，使用 cp.async 双缓冲
 * 重叠下一 K tile 的搬运与当前 tile 的 HMMA，并让 WMMA 直接写 C。
 * Aligned=false 是通用路径：尾块逐元素补零；输出先经过 shared memory，
 * 再由各 lane 对合法位置做掩码写回，避免 WMMA 整块写导致越界。
 */
template <bool Aligned>
__global__ void gemm_tensor_core_kernel(const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    // 快路径需要两个 stage 做 ping-pong；通用路径只实例化一个 stage，
    // 因此不会为不使用的双缓冲额外占用 Shared Memory。
    constexpr int Stages = Aligned ? 2 : 1;
    // 只让 Aligned 实例使用 padding；通用路径仍保持 BN=128，
    // 避免为不是 Benchmark 主路径的尾块 Kernel 增加 Shared Memory。
    constexpr int BSharedStride = Aligned ? BN_PADDED : BN;
    __shared__ half As[Stages][BM][BK];
    __shared__ half Bs[Stages][BK][BSharedStride];

    // Block 使用 16x16=256 个线程，即 8 个 warp。
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // 8 个 warp 排成 4x2：每个 warp 负责 C 中 32x64 的区域。
    // 32x64 又被拆成 2x4 个 16x16 WMMA tile。
    const int warp_m = warp_id / 2;
    const int warp_n = warp_id % 2;

    // 每个 warp 持有 8 个 FP32 累加 fragment。它们贯穿整个 K 循环，
    // 避免把中间结果写回 Shared/Global Memory，但也会带来较高寄存器压力。
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag[4];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[2][4];

    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    // A/B tile 各有 4096 个 half，256 个线程平均每线程搬运 16 个 half。
    // 16 个 half 拆成两次 float4 搬运，每次传输 8 个 half = 128 bit。
    const int a_load_row = tid / 2;
    const int a_load_col = (tid % 2) * 16;
    const int b_load_row = tid / 8;
    const int b_load_col = (tid % 8) * 16;

    int read_stage = 0;
    int write_stage = 1;

    if constexpr (Aligned) {
        // Prologue：在进入主循环前先把 tile 0 放入 stage 0。
        load_aligned_tile_async(&As[0][0][0], &Bs[0][0][0], A, B, N, K, 0, blockIdx.y, blockIdx.x, tid);
        wait_for_all_async_copies();
        __syncthreads();
    }

    // K 轴流水线：快路径在计算当前 read_stage 时，把下一 tile 异步写入
    // write_stage。通用路径仍使用 stage 0 做同步加载和边界补零。
    for (int bk = 0; bk < K; bk += BK) {
        if constexpr (Aligned) {
            if (bk + BK < K) {
                load_aligned_tile_async(&As[write_stage][0][0], &Bs[write_stage][0][0], A, B, N, K, bk + BK, blockIdx.y, blockIdx.x, tid);
            }
        } else {
            const int global_a_row = blockIdx.y * BM + a_load_row;
            const int global_a_col = bk + a_load_col;
            const std::size_t a_index =
                static_cast<std::size_t>(global_a_row) * K + global_a_col;

            // 通用路径只有在连续 16 个 half 全部合法且对齐时才能使用 float4。
            // 尾块或行首未对齐时逐元素读取，越界位置补 0，不影响 GEMM 结果。
            const bool a_vector_ok = global_a_row < M &&
                                     global_a_col + 15 < K &&
                                     (a_index % 8 == 0);
            if (a_vector_ok) {
                #pragma unroll
                for (int vec = 0; vec < 16; vec += 8) {
                    *reinterpret_cast<float4*>(&As[0][a_load_row][a_load_col + vec]) =
                        *reinterpret_cast<const float4*>(&A[a_index + vec]);
                }
            } else {
                #pragma unroll
                for (int v = 0; v < 16; ++v) {
                    const int col = global_a_col + v;
                    As[0][a_load_row][a_load_col + v] =
                        (global_a_row < M && col < K)
                            ? A[static_cast<std::size_t>(global_a_row) * K + col]
                            : __float2half(0.0f);
                }
            }

            const int global_b_row = bk + b_load_row;
            const int global_b_col = blockIdx.x * BN + b_load_col;
            const std::size_t b_index =
                static_cast<std::size_t>(global_b_row) * N + global_b_col;
            const bool b_vector_ok = global_b_row < K &&
                                     global_b_col + 15 < N &&
                                     (b_index % 8 == 0);
            if (b_vector_ok) {
                #pragma unroll
                for (int vec = 0; vec < 16; vec += 8) {
                    *reinterpret_cast<float4*>(&Bs[0][b_load_row][b_load_col + vec]) =
                        *reinterpret_cast<const float4*>(&B[b_index + vec]);
                }
            } else {
                #pragma unroll
                for (int v = 0; v < 16; ++v) {
                    const int col = global_b_col + v;
                    Bs[0][b_load_row][b_load_col + v] =
                        (global_b_row < K && col < N)
                            ? B[static_cast<std::size_t>(global_b_row) * N + col]
                            : __float2half(0.0f);
                }
            }

            // 同步路径必须等整个 Block 完成协作加载后才能读取 stage 0。
            __syncthreads();
        }

        // BK=32 包含两个 k16 子阶段。每个阶段先装载 fragment，
        // 再执行 2x4=8 次 warp 级 mma_sync。
        #pragma unroll
        for (int k_step = 0; k_step < BK / WMMA_K; ++k_step) {
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int row = warp_m * 32 + i * WMMA_M;
                wmma::load_matrix_sync(a_frag[i], &As[read_stage][row][k_step * WMMA_K], BK);
            }

            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int col = warp_n * 64 + j * WMMA_N;
                // ldm 必须与 B 在 Shared Memory 中的真实行跨度一致。
                // Aligned 路径传入 136，通用路径仍传入 128。
                wmma::load_matrix_sync(b_frag[j], &Bs[read_stage][k_step * WMMA_K][col], BSharedStride);
            }

            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
                }
            }
        }

        if constexpr (Aligned) {
            if (bk + BK < K) {
                // 当前 tile 计算完后等待下一 tile，再交换读写 stage。
                wait_for_all_async_copies();
                __syncthreads();
                const int old_read_stage = read_stage;
                read_stage = write_stage;
                write_stage = old_read_stage;
            }
        } else {
            // 通用路径下一轮会覆盖 stage 0，必须等所有 warp 消费完毕。
            __syncthreads();
        }
    }

    if constexpr (Aligned) {
        // 快路径中每个 16x16 输出 tile 都完全位于矩阵内，可由 WMMA 直接写回。
        // 这会绕过通用路径的 8 KiB shared-memory 输出暂存区和逐元素判断。
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int row = blockIdx.y * BM + warp_m * 32 + i * WMMA_M;
                const int col = blockIdx.x * BN + warp_n * 64 + j * WMMA_N;
                wmma::store_matrix_sync(&C[static_cast<std::size_t>(row) * N + col], c_frag[i][j], N, wmma::mem_row_major);
            }
        }
    } else {
        // WMMA store 没有逐元素 mask，只能一次写完整 16x16 tile。
        // 每个 warp 使用独占暂存区，随后由 32 个 lane 合作写回合法元素。
        __shared__ __align__(32) float C_tile[8][WMMA_M][WMMA_N];
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                wmma::store_matrix_sync(&C_tile[warp_id][0][0], c_frag[i][j], WMMA_N, wmma::mem_row_major);
                // store_matrix_sync 是 warp 集体操作；写回 C 前确保暂存数据可见。
                __syncwarp();

                const int tile_row = blockIdx.y * BM + warp_m * 32 + i * WMMA_M;
                const int tile_col = blockIdx.x * BN + warp_n * 64 + j * WMMA_N;
                for (int index = lane_id; index < WMMA_M * WMMA_N; index += 32) {
                    const int row = tile_row + index / WMMA_N;
                    const int col = tile_col + index % WMMA_N;
                    if (row < M && col < N) {
                        C[static_cast<std::size_t>(row) * N + col] =
                            C_tile[warp_id][index / WMMA_N][index % WMMA_N];
                    }
                }
                // 同一 warp 的下一 fragment 会复用这块暂存区。
                __syncwarp();
            }
        }
    }
}


/**
 * Small/Medium GEMM Tensor Core Kernel。
 *
 * 每个 Block 计算 64x64 输出 tile，使用 4 个 warp。相比 128x128 大 tile，
 * 它会生成更多 Block、减少尾块浪费，并降低每个 warp 的 accumulator 数量。
 */
template <bool Aligned>
__global__ void gemm_tensor_core_small_kernel(const half* __restrict__ A, const half* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    constexpr int Stages = Aligned ? 2 : 1;
    constexpr int BSharedStride = Aligned ? SMALL_BN_PADDED : SMALL_BN;
    __shared__ half As[Stages][SMALL_BM][SMALL_BK];
    __shared__ half Bs[Stages][SMALL_BK][BSharedStride];

    // Block 使用 16x8=128 个线程，即 4 个 warp。
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    // 4 个 warp 排成 2x2，每个 warp 负责 32x32 输出区域。
    const int warp_m = warp_id / 2;
    const int warp_n = warp_id % 2;

    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag[2];
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag[2];
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag[2][2];

    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 2; ++j) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    // A/B tile 各有 2048 个 half，128 个线程平均每线程搬运 16 个 half。
    const int a_load_row = tid / 2;
    const int a_load_col = (tid % 2) * 16;
    const int b_load_row = tid / 4;
    const int b_load_col = (tid % 4) * 16;

    int read_stage = 0;
    int write_stage = 1;

    if constexpr (Aligned) {
        load_small_aligned_tile_async(&As[0][0][0], &Bs[0][0][0], A, B, N, K, 0, blockIdx.y, blockIdx.x, tid);
        wait_for_all_async_copies();
        __syncthreads();
    }

    for (int bk = 0; bk < K; bk += SMALL_BK) {
        if constexpr (Aligned) {
            if (bk + SMALL_BK < K) {
                load_small_aligned_tile_async(&As[write_stage][0][0], &Bs[write_stage][0][0], A, B, N, K, bk + SMALL_BK, blockIdx.y, blockIdx.x, tid);
            }
        } else {
            const int global_a_row = blockIdx.y * SMALL_BM + a_load_row;
            const int global_a_col = bk + a_load_col;
            const std::size_t a_index = static_cast<std::size_t>(global_a_row) * K + global_a_col;

            const bool a_vector_ok = global_a_row < M && global_a_col + 15 < K && a_index % 8 == 0;
            if (a_vector_ok) {
                #pragma unroll
                for (int vec = 0; vec < 16; vec += 8) {
                    *reinterpret_cast<float4*>(&As[0][a_load_row][a_load_col + vec]) = *reinterpret_cast<const float4*>(&A[a_index + vec]);
                }
            } else {
                #pragma unroll
                for (int v = 0; v < 16; ++v) {
                    const int col = global_a_col + v;
                    As[0][a_load_row][a_load_col + v] = global_a_row < M && col < K ? A[static_cast<std::size_t>(global_a_row) * K + col] : __float2half(0.0f);
                }
            }

            const int global_b_row = bk + b_load_row;
            const int global_b_col = blockIdx.x * SMALL_BN + b_load_col;
            const std::size_t b_index = static_cast<std::size_t>(global_b_row) * N + global_b_col;
            const bool b_vector_ok = global_b_row < K && global_b_col + 15 < N && b_index % 8 == 0;
            if (b_vector_ok) {
                #pragma unroll
                for (int vec = 0; vec < 16; vec += 8) {
                    *reinterpret_cast<float4*>(&Bs[0][b_load_row][b_load_col + vec]) = *reinterpret_cast<const float4*>(&B[b_index + vec]);
                }
            } else {
                #pragma unroll
                for (int v = 0; v < 16; ++v) {
                    const int col = global_b_col + v;
                    Bs[0][b_load_row][b_load_col + v] = global_b_row < K && col < N ? B[static_cast<std::size_t>(global_b_row) * N + col] : __float2half(0.0f);
                }
            }

            __syncthreads();
        }

        #pragma unroll
        for (int k_step = 0; k_step < SMALL_BK / WMMA_K; ++k_step) {
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int row = warp_m * 32 + i * WMMA_M;
                wmma::load_matrix_sync(a_frag[i], &As[read_stage][row][k_step * WMMA_K], SMALL_BK);
            }

            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int col = warp_n * 32 + j * WMMA_N;
                wmma::load_matrix_sync(b_frag[j], &Bs[read_stage][k_step * WMMA_K][col], BSharedStride);
            }

            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                #pragma unroll
                for (int j = 0; j < 2; ++j) {
                    wmma::mma_sync(c_frag[i][j], a_frag[i], b_frag[j], c_frag[i][j]);
                }
            }
        }

        if constexpr (Aligned) {
            if (bk + SMALL_BK < K) {
                wait_for_all_async_copies();
                __syncthreads();
                const int old_read_stage = read_stage;
                read_stage = write_stage;
                write_stage = old_read_stage;
            }
        } else {
            __syncthreads();
        }
    }

    if constexpr (Aligned) {
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int row = blockIdx.y * SMALL_BM + warp_m * 32 + i * WMMA_M;
                const int col = blockIdx.x * SMALL_BN + warp_n * 32 + j * WMMA_N;
                wmma::store_matrix_sync(&C[static_cast<std::size_t>(row) * N + col], c_frag[i][j], N, wmma::mem_row_major);
            }
        }
    } else {
        __shared__ __align__(32) float C_tile[4][WMMA_M][WMMA_N];
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 2; ++j) {
                wmma::store_matrix_sync(&C_tile[warp_id][0][0], c_frag[i][j], WMMA_N, wmma::mem_row_major);
                __syncwarp();

                const int tile_row = blockIdx.y * SMALL_BM + warp_m * 32 + i * WMMA_M;
                const int tile_col = blockIdx.x * SMALL_BN + warp_n * 32 + j * WMMA_N;
                for (int index = lane_id; index < WMMA_M * WMMA_N; index += 32) {
                    const int row = tile_row + index / WMMA_N;
                    const int col = tile_col + index % WMMA_N;
                    if (row < M && col < N) {
                        C[static_cast<std::size_t>(row) * N + col] = C_tile[warp_id][index / WMMA_N][index % WMMA_N];
                    }
                }
                __syncwarp();
            }
        }
    }
}


// Shared Memory Tiling GEMM Kernel: C = A * B
template <typename scalar_t>
__global__ void gemm_v1_tiled(const scalar_t* __restrict__ A, const scalar_t* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    // 静态分配 Shared Memory
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // 当前线程负责的 C 矩阵全局行列号
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;

    // 沿 K 维度按 TILE_SIZE 分块推进
    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    for (int i = 0; i < num_tiles; ++i) {
        // 协作加载 A 和 B 的 Tile 到 Shared Memory，并处理边界
        if (row < M && i * TILE_SIZE + tx < K) {
            As[ty][tx] = static_cast<float>(A[row * K + i * TILE_SIZE + tx]);
        } else {
            As[ty][tx] = 0.0f;
        }

        if (i * TILE_SIZE + ty < K && col < N) {
            Bs[ty][tx] = static_cast<float>(B[(i * TILE_SIZE + ty) * N + col]);
        } else {
            Bs[ty][tx] = 0.0f;
        }

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += As[ty][k] * Bs[k][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}


// B 按 [N, K] 行主序保存，在加载 Shared Memory 时直接实现 B 的转置视图。
template <typename scalar_t>
__global__ void gemm_v1_tiled_transpose_b(const scalar_t* __restrict__ A, const scalar_t* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TILE_SIZE + ty;
    const int col = blockIdx.x * TILE_SIZE + tx;

    float sum = 0.0f;

    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; ++tile) {
        const int a_col = tile * TILE_SIZE + tx;
        const int b_col = tile * TILE_SIZE + ty;

        As[ty][tx] = row < M && a_col < K ? static_cast<float>(A[row * K + a_col]) : 0.0f;

        // Bs[k_local][n_local] 对应 B[n_global][k_global]。
        Bs[ty][tx] = col < N && b_col < K ? static_cast<float>(B[col * K + b_col]) : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int inner = 0; inner < TILE_SIZE; ++inner) {
            sum += As[ty][inner] * Bs[inner][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

}  // namespace


void launch_gemm_cuda(const at::Tensor& a, const at::Tensor& b, at::Tensor& output, std::int64_t M, std::int64_t N, std::int64_t K, bool transpose_b, cudaStream_t stream) {
    TORCH_CHECK(M <= std::numeric_limits<int>::max(), "gemm: M 超过 int 支持范围");
    TORCH_CHECK(N <= std::numeric_limits<int>::max(), "gemm: N 超过 int 支持范围");
    TORCH_CHECK(K <= std::numeric_limits<int>::max(), "gemm: K 超过 int 支持范围");

    const int rows = static_cast<int>(M);
    const int columns = static_cast<int>(N);
    const int reduction_size = static_cast<int>(K);

    // Tensor Core kernel 当前只支持 FP16、非转置 B。通用路径内部也可能使用
    // float4，因此必须检查 PyTorch Tensor 带 storage_offset 后的实际地址。
    if (a.scalar_type() == at::kHalf && !transpose_b) {
        const half* A = reinterpret_cast<const half*>(a.const_data_ptr<at::Half>());
        const half* B = reinterpret_cast<const half*>(b.const_data_ptr<at::Half>());
        float* C = output.mutable_data_ptr<float>();

        const std::uintptr_t a_address = reinterpret_cast<std::uintptr_t>(A);
        const std::uintptr_t b_address = reinterpret_cast<std::uintptr_t>(B);
        const bool input_address_aligned = a_address % alignof(float4) == 0 && b_address % alignof(float4) == 0;

        // 同轮 crossover benchmark 表明约 1M 以下的工作量仍由 v1 tiled 更合适。
        const double logical_work = static_cast<double>(rows) * columns * reduction_size;
        constexpr double MIN_TENSOR_CORE_WORK = 1024.0 * 1024.0;

        if (input_address_aligned && logical_work >= MIN_TENSOR_CORE_WORK) {
            const std::uintptr_t output_address = reinterpret_cast<std::uintptr_t>(C);

            // 任一输出维较小时使用 64x64 tile，增加并行 Block 数量并减少尾块浪费。
            const bool use_small_kernel = rows < 1024 || columns < 1024;
            if (use_small_kernel) {
                const dim3 small_block(16, 8);
                const dim3 small_grid((columns + SMALL_BN - 1) / SMALL_BN, (rows + SMALL_BM - 1) / SMALL_BM);
                const bool small_aligned = rows % SMALL_BM == 0 && columns % SMALL_BN == 0 && reduction_size % SMALL_BK == 0 && output_address % 32 == 0;
                if (small_aligned) {
                    gemm_tensor_core_small_kernel<true><<<small_grid, small_block, 0, stream>>>(A, B, C, rows, columns, reduction_size);
                } else {
                    gemm_tensor_core_small_kernel<false><<<small_grid, small_block, 0, stream>>>(A, B, C, rows, columns, reduction_size);
                }
            } else {
                const dim3 large_block(16, 16);
                const dim3 large_grid((columns + BN - 1) / BN, (rows + BM - 1) / BM);
                const bool large_aligned = rows % BM == 0 && columns % BN == 0 && reduction_size % BK == 0 && output_address % 32 == 0;
                if (large_aligned) {
                    gemm_tensor_core_kernel<true><<<large_grid, large_block, 0, stream>>>(A, B, C, rows, columns, reduction_size);
                } else {
                    gemm_tensor_core_kernel<false><<<large_grid, large_block, 0, stream>>>(A, B, C, rows, columns, reduction_size);
                }
            }

            C10_CUDA_KERNEL_LAUNCH_CHECK();
            return;
        }
    }

    const dim3 block(TILE_SIZE, TILE_SIZE);
    const dim3 grid((columns + TILE_SIZE - 1) / TILE_SIZE, (rows + TILE_SIZE - 1) / TILE_SIZE);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(a.scalar_type(), "gemm_v1_tiled_cuda", [&] {
        if (transpose_b) {
            gemm_v1_tiled_transpose_b<scalar_t><<<grid, block, 0, stream>>>(a.const_data_ptr<scalar_t>(), b.const_data_ptr<scalar_t>(), output.mutable_data_ptr<float>(), rows, columns, reduction_size);
        } else {
            gemm_v1_tiled<scalar_t><<<grid, block, 0, stream>>>(a.const_data_ptr<scalar_t>(), b.const_data_ptr<scalar_t>(), output.mutable_data_ptr<float>(), rows, columns, reduction_size);
        }
    });

    // 只检查异步 launch 错误，不进行全设备同步。
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace my_ops
