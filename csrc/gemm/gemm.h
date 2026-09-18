// 文件职责：声明 GEMM 的 Dispatcher C++ wrapper 与 CUDA launcher 接口。

#pragma once

#include <cstdint>

#include <ATen/core/Tensor.h>
#include <cuda_runtime_api.h>


namespace my_ops {

// Dispatcher 调用的 CUDA wrapper：负责检查 Tensor 契约、设置 device、分配输出和获取 current stream。
at::Tensor gemm_cuda(const at::Tensor& a, const at::Tensor& b, bool transpose_b);

// CUDA launcher：负责 dtype dispatch、kernel selector、grid/block 配置和 kernel launch。
void launch_gemm_cuda(const at::Tensor& a, const at::Tensor& b, at::Tensor& output, std::int64_t M, std::int64_t N, std::int64_t K, bool transpose_b, cudaStream_t stream);

}  // namespace my_ops