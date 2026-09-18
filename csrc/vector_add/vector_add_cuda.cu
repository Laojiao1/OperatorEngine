// 文件职责：CUDA 源码。包含运行在 GPU 上的核函数（Kernel）和启动配置（Launcher）。
// 实现标量 FP32 Vector Add CUDA kernel、launch 配置和错误检查。

#include "vector_add/vector_add.h"

#include <limits>

#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>


namespace my_ops {
namespace {

constexpr int kThreadsPerBlock = 256;

// GPU 的流式多处理器（SM）接收到指令，创建线程并开始并行执行：CUDA Kernel：向量加法
__global__ void vector_add_kernel(const float* A, const float* B, float* C, std::int64_t N) {
    // 计算全局线程编号
    const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    // 防止最后一个 Block 中多余线程越界访问
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}
// 结果返回：C++ 函数将封装好的 output Tensor 对象原路返回给 PyTorch Dispatcher，Dispatcher 返回给 Python 环境中的变量 out

}  // namespace

void launch_vector_add_cuda(const float* A, const float* B, float* C, std::int64_t N, cudaStream_t stream) {
    // 向上取整计算需要的 Block 数量，确保总线程数 >= N
    const std::int64_t blocks = (N + kThreadsPerBlock - 1) / kThreadsPerBlock;

    TORCH_CHECK(blocks <= std::numeric_limits<int>::max(), "vector_add: tensor is too large for the one-thread-per-element grid");

    // 启动 Kernel 时显式传入 PyTorch 当前 CUDA Stream
    // 这里的第三个参数 0 表示核函数使用 0 字节的动态共享内存（Dynamic Shared Memory）
    vector_add_kernel<<<static_cast<int>(blocks), kThreadsPerBlock, 0, stream>>>(A, B, C, N);

    // 检查刚才这一条启动指令本身是否有错误（例如参数是否传错、Block 数量是否超出了硬件限制）
    // 注意：这里绝不调用 cudaDeviceSynchronize()，它只检查启动命令是否成功推送到 GPU 队列，保持 GPU 和 CPU 的异步非阻塞运行
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace my_ops
