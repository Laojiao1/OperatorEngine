// 文件职责：头文件。声明 C++ 函数与 CUDA 启动函数的接口

#pragma once

#include <cstdint>

#include <ATen/core/Tensor.h>
#include <cuda_runtime_api.h>


namespace my_ops {

// PyTorch Dispatcher 调用的 C++ wrapper：负责 Tensor 契约和资源生命周期。
at::Tensor vector_add_cuda(const at::Tensor& a, const at::Tensor& b);

// wrapper 与 CUDA translation unit 之间的窄接口：只传裸指针、元素数和 stream。
void launch_vector_add_cuda(const float* A, const float* B, float* C, std::int64_t N, cudaStream_t stream);

}  // namespace my_ops
