// 文件职责：声明 Attention 的 Dispatcher C++ wrapper 接口。

#pragma once

#include <ATen/core/Tensor.h>

#include <cuda_runtime_api.h>


namespace my_ops {

// Dispatcher CUDA wrapper：检查四维契约，并在 CUDA Core kernel 与 PyTorch SDPA 之间选择。
at::Tensor attention_cuda(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, bool causal);

// CUDA Core launcher：负责模板分派、grid/block 配置、共享内存大小和 kernel 启动。
void launch_attention_cuda_core(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, at::Tensor& output, bool causal, cudaStream_t stream);

// Tensor Core launcher：当前只支持 FP16、head_dim=64。
void launch_attention_tensor_core(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, at::Tensor& output, bool causal, cudaStream_t stream);

}  // namespace my_ops
