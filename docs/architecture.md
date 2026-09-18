# OperatorEngine 架构与环境

## 1. Phase 0 环境快照

记录日期：2026-09-12，时区 Asia/Shanghai。

项目构建环境固定为 WSL2 中 VS Code 已配置的 `triton` conda 环境：

| 项目 | 当前值 |
| --- | --- |
| Host OS | Windows 11，build 26200 |
| WSL | WSL2，kernel `6.6.87.2-microsoft-standard-WSL2` |
| Distribution | Ubuntu 20.04.6 LTS |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| Compute capability | 12.0 (`sm_120`) |
| NVIDIA driver | 576.65 |
| Python | 3.10.20，`/home/laojiao/miniconda3/envs/triton/bin/python` |
| PyTorch | 2.11.0+cu128 |
| PyTorch CUDA runtime | 12.8 |
| Triton | 3.6.0 |
| CUDA Toolkit | 12.8，`/usr/local/cuda-12.8` |
| nvcc | 12.8.61，`/usr/local/cuda/bin/nvcc` |
| Host C++ compiler | GCC/G++ 9.4.0；PyTorch ABI compatibility check 通过 |
| setuptools / pytest | 78.1.0 / 9.1.1 |

构建时从 WSL 进入仓库：

```bash
cd /mnt/e/Code/C++Compiler/OperatorEngine
/home/laojiao/miniconda3/envs/triton/bin/python -m pip install -e . --no-build-isolation
```

注意：PowerShell 默认命中的 `python.exe` 是不可执行的 Windows Store 占位符。Windows 的 `E:\conda_envs\pytorch` 环境虽然能看到 GPU，但它是 PyTorch 2.5.1+cu121，并且该 wheel 不包含 `sm_120` 支持；它不是本项目的构建环境。

## 2. 第一条调用链

```text
Python: torch.ops.my_ops.vector_add(a, b)
  -> PyTorch Dispatcher
     根据 schema 校验调用形式，并根据 Tensor dispatch key 选择 CUDA implementation
  -> C++ wrapper: vector_add_cuda(a, b)
     检查契约、保护输入 device、分配 output、处理 empty Tensor
  -> CUDA launcher
     取得并传入 PyTorch current CUDA stream，配置 grid/block，启动 kernel
  -> CUDA kernel
     以线性 numel 处理连续 FP32 元素，尾块用 index < numel 防越界
  -> output torch.Tensor
     Dispatcher 将结果原路返回 Python
  -> pytest correctness + torch.library.opcheck
```

各层职责必须分开：

- Dispatcher 负责 schema 与后端路由，不负责数值正确性。
- C++ wrapper 负责 Tensor 契约、device 和输出生命周期，不生成测试数据。
- launcher 负责 launch 配置、stream 和 launch error，不做无条件同步。
- kernel 只处理已经通过契约检查的裸指针与元素数量。
- correctness test 与 PyTorch reference 比数值；`opcheck` 检查注册、FakeTensor/元数据等算子契约，两者不能互相替代。

## 3. current stream 为什么是接口的一部分

PyTorch 的 CUDA 运算是异步排队的。若调用者在 non-default stream 上先写入 `a`、`b`，自定义 kernel 却偷偷启动到 default stream，kernel 与前序写入之间就没有正确的同流顺序，可能读取旧数据。反过来，输出也可能在消费者读取之前尚未完成。

因此 launcher 必须接收 `cudaStream_t`，C++ wrapper 必须在输入所在 device 上取得 PyTorch 的 current CUDA stream。正常路径不能用 `cudaDeviceSynchronize()` 掩盖 stream 错误；non-default stream 测试会专门验证这一点。

## 4. 旧 Vector Add 实现盘点

### Triton 版本

`3_Triton/01_vector_add/kernel.py` 的 kernel 参数为 `x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE`。一维 grid 使用 `ceil_div(n_elements, BLOCK_SIZE)`；每个 program 生成连续 offsets，并用 `offsets < n_elements` 同时保护 load/store。Python wrapper 已检查 CUDA、同 device/shape/dtype、支持 dtype、contiguous 和空 Tensor。

它当前支持 FP16/BF16/FP32，但 OperatorEngine 第一版主动收窄为 FP32，减少 dtype dispatch 对工程闭环的干扰。

### CUDA 版本

`4_CUDA/00_cuda_prerequisite/Vector_add.cu` 使用最直接的一线程一元素映射：

```text
index = blockIdx.x * blockDim.x + threadIdx.x
if index < n: output[index] = a[index] + b[index]
```

`4_CUDA/02_cuda_minimal/01_execution_model/vector_add.cu` 进一步展示了 grid-stride loop；`02_memory/vector_add_float4.cu` 展示了标量和 `float4` 路径、尾部处理及 benchmark。

Phase 1 第一版选择标量 FP32、256 threads/block、`ceil_div(numel, 256)` 和 `index < numel`。这个版本最容易审查“一个线程对应一个元素”的正确性，也天然覆盖 1003 这类非整除长度。暂不迁移 `float4`：PyTorch Tensor 可能是带 storage offset 的 contiguous view，不能仅根据 allocator 基地址就假设传入的 `data_ptr()` 始终满足 16-byte alignment。

旧 `.cu` 文件中的 `main()`、`cudaMalloc/cudaMemcpy/cudaFree`、benchmark、随机数据和 `cudaDeviceSynchronize()` 都不迁移。Tensor 的存储由 PyTorch 管理，扩展只接收 Tensor、分配输出并把 kernel 排入 current stream。

## 5. Phase 1 接口边界

第一版 schema：

```text
vector_add(Tensor a, Tensor b) -> Tensor
```

第一版约束：CUDA only、同 device、FP32、同 shape、contiguous。输出是新 Tensor，继承输入的 shape、dtype 和 device；空 Tensor 返回已分配的空输出，不启动零 block kernel；autograd 不在支持范围内。

以上是 Phase 1 当时主动收窄的接口边界。当前项目已继续完成 Softmax、GEMM、Attention 和最小 Transformer Block；各算子的最新契约与 selector 见 `docs/dispatch_policy.md`，整体限制见 `docs/limitations.md`。

## 6. 当前完整调用边界

四个低层算子都通过同一个扩展完成 Dispatcher 注册：

```text
torch.ops.my_ops.vector_add
torch.ops.my_ops.softmax
torch.ops.my_ops.gemm
torch.ops.my_ops.attention
```

`my_ops.attention` 在 Python 层进一步统一 Triton、C++ CUDA 和 PyTorch SDPA provider。`examples/transformer_block.py` 只组合公开算子接口，不直接依赖任何 CUDA launcher 或 kernel 符号。这样算子注册、实现选择和模型集成保持清晰的职责边界。
