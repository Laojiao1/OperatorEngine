// 文件职责：声明 Softmax 的 Dispatcher C++ wrapper 接口。

#pragma once

#include <cstdint>

#include <ATen/core/Tensor.h>
#include <cuda_runtime_api.h>


namespace my_ops {

at::Tensor softmax_cuda(const at::Tensor& input, std::int64_t dim);

// launcher 负责实现选择、dtype dispatch、grid/block 配置和 launch error。
void launch_softmax_cuda(const at::Tensor& input, at::Tensor& output, std::int64_t M, std::int64_t N, cudaStream_t stream);

}  // namespace my_ops
