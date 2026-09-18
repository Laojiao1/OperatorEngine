// 文件职责：迁移阶段四的 Shared Memory Block Softmax，并接入 PyTorch CUDA Stream。

#include "softmax/softmax.h"

#include <cfloat>
#include <cstdint>
#include <limits>

#include <ATen/Dispatch.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>


namespace my_ops {
namespace {

// Single Block Row-wise Softmax Kernel
// 1 个 Block 负责 1 行 (M 行 = M 个 Blocks)
// 使用 Shared Memory 进行 Block 内的树状规约 (Tree Reduction)
template <typename scalar_t>
__global__ void softmax_block_kernel(const scalar_t* __restrict__ input, scalar_t* __restrict__ output, int M, int N) {
    // 动态分配 Shared Memory，用于树状规约中间数据存储
    extern __shared__ float sdata[];

    int row = blockIdx.x; // 1 个 Block 对应 1 行
    int tid = threadIdx.x; // Block 内部的线程 ID
    int block_size = blockDim.x;

    if (row >= M) return;

    const scalar_t* row_input = input + row * N;
    scalar_t* row_output = output + row * N;

    // ----- 阶段 1: 寻找行最大值 (Row Max) -----
    // 每个线程首先在寄存器中计算自己负责的元素的局部最大值 (Block-stride Loop)
    float thread_max = -FLT_MAX;
    for (int col = tid; col < N; col += block_size) {
        thread_max = fmaxf(thread_max, static_cast<float>(row_input[col]));
    }

    // 将线程局部最大值写入 Shared Memory
    sdata[tid] = thread_max;
    __syncthreads(); // 必须同步，确保所有线程都将 thread_max 写入了 SMem

    // 在 Shared Memory 中进行折半树状规约 (Tree Reduction)
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads(); // 每一层规约都需要块内同步，避免读写冲突 (Race Condition)
    }

    // 规约结束后，sdata[0] 即为该行的全局最大值
    float row_max = sdata[0];


    // ----- 阶段 2: 计算 exp(x - row_max) 并求行和 (Row Sum) -----
    // 每个线程再次遍历自己负责的数据，在寄存器中累加求和 (FP32 Accumulation)
    float thread_sum = 0.0f;
    for (int col = tid; col < N; col += block_size) {
        thread_sum += expf(static_cast<float>(row_input[col]) - row_max);
    }

    // 将线程局部和写入 Shared Memory
    sdata[tid] = thread_sum;
    __syncthreads();

    // 在 Shared Memory 中再次进行折半树状规约求 Sum
    for (int stride = block_size / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // 规约结束后，sdata[0] 即为该行的 ExpSum
    float row_sum = sdata[0];
    float inv_sum = 1.0f / row_sum; // 预先计算倒数，减少重复除法运算


    // ----- 阶段 3: 归一化并写回 Global Memory -----
    for (int col = tid; col < N; col += block_size) {
        row_output[col] = static_cast<scalar_t>(expf(static_cast<float>(row_input[col]) - row_max) * inv_sum);
    }
}

// 定义 Online Softmax 的状态结构体 (Max, Sum)
struct MDState {
    float m; // 局部/全局最大值
    float d; // 局部/全局 Exp 累加和
};

// Online Softmax 的状态合并函数 (Online Reduction)
// 结合两个局部状态 (m_a, d_a) 与 (m_b, d_b) -> 合并为 (m_new, d_new)
__device__ __forceinline__ MDState combine_md(MDState a, MDState b) {
    if (a.m < b.m) {
        return {b.m, a.d * expf(a.m - b.m) + b.d};
    } else {
        return {a.m, b.d * expf(b.m - a.m) + a.d};
    }
}

// Warp 级的 Online MDState 规约 (基于 Register Shuffle)
__device__ __forceinline__ MDState warp_reduce_md(MDState state) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        MDState other;
        other.m = __shfl_down_sync(0xffffffff, state.m, mask);
        other.d = __shfl_down_sync(0xffffffff, state.d, mask);
        state = combine_md(state, other);
    }
    return state;
}

// Block 级的 Online MDState 规约
__device__ __forceinline__ MDState block_reduce_md(MDState state, MDState* s_mem) {
    int lane_id = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;

    // 1. Warp 内 Online 规约
    state = warp_reduce_md(state);

    // 2. 各 Warp 的 0 号线程写入 SMem
    if (lane_id == 0) {
        s_mem[warp_id] = state;
    }
    __syncthreads();

    // 3. 由第 0 个 Warp 汇总各 Warp 的 MDState
    int num_warps = blockDim.x / 32;
    MDState block_state = (threadIdx.x < num_warps) ? s_mem[lane_id] : MDState{-FLT_MAX, 0.0f};

    if (warp_id == 0) {
        block_state = warp_reduce_md(block_state);
    }

    if (threadIdx.x == 0) {
        s_mem[0] = block_state;
    }
    __syncthreads();

    return s_mem[0];
}

// Single-Pass State Tracking Online Softmax Kernel
__global__ void softmax_online_kernel(const float* __restrict__ input, float* __restrict__ output, int M, int N) {
    extern __shared__ MDState s_mem_md[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int block_size = blockDim.x;

    if (row >= M) return;

    const float* row_input = input + row * N;
    float* row_output = output + row * N;

    int num_vec = N / 4;
    int tail_start = num_vec * 4;

    // 阶段 1: 单次遍历 (Single-Pass Load)，流式更新线程局部的 (m, d) 状态
    MDState thread_md = {-FLT_MAX, 0.0f};

    const float4* vec_input = reinterpret_cast<const float4*>(row_input);
    for (int idx = tid; idx < num_vec; idx += block_size) {
        float4 val4 = vec_input[idx];

        // 依次将 4 个矢量化元素流式融合进 thread_md
        thread_md = combine_md(thread_md, {val4.x, 1.0f});
        thread_md = combine_md(thread_md, {val4.y, 1.0f});
        thread_md = combine_md(thread_md, {val4.z, 1.0f});
        thread_md = combine_md(thread_md, {val4.w, 1.0f});
    }

    // 处理尾部元素
    for (int col = tail_start + tid; col < N; col += block_size) {
        thread_md = combine_md(thread_md, {row_input[col], 1.0f});
    }

    // 阶段 2: 块级分层规约，得到整行的全局 (row_max, row_sum) 状态
    MDState row_md = block_reduce_md(thread_md, s_mem_md);

    float row_max = row_md.m;
    float inv_sum = 1.0f / row_md.d;

    // 阶段 3: 利用最终状态进行归一化并写回
    float4* vec_output = reinterpret_cast<float4*>(row_output);
    for (int idx = tid; idx < num_vec; idx += block_size) {
        float4 val4 = vec_input[idx];
        float4 out4;
        out4.x = expf(val4.x - row_max) * inv_sum;
        out4.y = expf(val4.y - row_max) * inv_sum;
        out4.z = expf(val4.z - row_max) * inv_sum;
        out4.w = expf(val4.w - row_max) * inv_sum;
        vec_output[idx] = out4;
    }

    for (int col = tail_start + tid; col < N; col += block_size) {
        row_output[col] = expf(row_input[col] - row_max) * inv_sum;
    }
}

}  // namespace


void launch_softmax_cuda(const at::Tensor& input, at::Tensor& output, std::int64_t M, std::int64_t N, cudaStream_t stream) {
    TORCH_CHECK(M <= std::numeric_limits<int>::max(), "softmax: 行数超过 CUDA grid.x 的支持范围");
    TORCH_CHECK(N <= std::numeric_limits<int>::max(), "softmax: 最后一维超过 int 的支持范围");

    // 1 个 Block 负责 1 行，总共 M 个 Block
    const int threads_per_block = 256;
    const int blocks_per_grid = static_cast<int>(M);
    const int rows = static_cast<int>(M);
    const int columns = static_cast<int>(N);

    // 旧 v4 使用 float4，只有 FP32、列数为 4 的倍数且 Tensor 起点 16 Byte 对齐时才能安全进入
    bool use_online_kernel = false;
    if (input.scalar_type() == at::kFloat && columns >= 1024 && columns % 4 == 0) {
        const std::uintptr_t input_address = reinterpret_cast<std::uintptr_t>(input.const_data_ptr<float>());
        const std::uintptr_t output_address = reinterpret_cast<std::uintptr_t>(output.mutable_data_ptr<float>());
        use_online_kernel = input_address % alignof(float4) == 0 && output_address % alignof(float4) == 0;
    }

    if (use_online_kernel) {
        const size_t shared_mem_size = (threads_per_block / 32) * sizeof(MDState);
        softmax_online_kernel<<<blocks_per_grid, threads_per_block, shared_mem_size, stream>>>(input.const_data_ptr<float>(), output.mutable_data_ptr<float>(), rows, columns);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }

    // 其他输入走旧 v1 Shared Memory Kernel；FP16 的归约过程仍使用 FP32 Accumulation
    const size_t shared_mem_size = threads_per_block * sizeof(float);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "softmax_block_cuda", [&] {
        softmax_block_kernel<scalar_t><<<blocks_per_grid, threads_per_block, shared_mem_size, stream>>>(input.const_data_ptr<scalar_t>(), output.mutable_data_ptr<scalar_t>(), rows, columns);
    });

    // 只检查 Kernel 启动配置等同步前可发现的错误，不调用 cudaDeviceSynchronize
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace my_ops
