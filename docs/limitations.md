# OperatorEngine 已知限制

## 通用限制

- 只支持 NVIDIA CUDA，当前验证环境为 WSL2、PyTorch 2.11.0+cu128、CUDA 12.8 和 RTX 5060 Laptop GPU（sm_120）。
- 第一版只实现 inference forward，不支持 autograd/backward。
- 不支持 CPU 高性能后端、ROCm、多 GPU、NCCL、TP、PP、Paged Attention 或完整推理框架接入。
- selector 阈值来自当前硬件的离线 benchmark，不是跨 GPU 通用的硬件定律。
- 正常算子路径只检查异步 kernel launch error，不做全设备同步；调用者必须遵守 PyTorch stream 语义。

## Attention 限制

- Q、K、V 必须是相同 shape/device/dtype 的四维 contiguous CUDA Tensor `[B,H,N,D]`。
- 只支持 FP16 和 `D=64/128`，输出为 FP16，Softmax 与输出累加使用 FP32。
- 不支持不同的 Q/K/V 序列长度、attention mask、dropout、GQA/MQA、KV cache、ragged 或 paged layout。
- C++ Tensor Core kernel 当前固定 `D=64`，静态 Shared Memory 约 38.8 KiB；D128 使用 CUDA Core 或上层 fallback。
- C++ CUDA Core/Tensor Core kernel 已通过 correctness，但多数中大 shape 尚未达到 PyTorch SDPA 性能。
- Python auto selector 当前只按 causal 与 sequence length 选择 Triton/SDPA。阈值来自有限 case，后续需要更多模型 shape 校准。
- `my_ops.attention` 是普通 Python 统一入口，不是新的 Dispatcher schema；低层注册算子仍是 `torch.ops.my_ops.attention`。
- FakeTensor/opcheck 当前验证低层 C++ Dispatcher 算子。Python/Triton 统一入口尚未声明为 `torch.library.triton_op`，也未验收 `torch.compile`。
- 包导入会加载 Triton Python 模块；当前项目环境必须安装与 PyTorch/CUDA 匹配的 Triton。

## Softmax 与 GEMM 限制

- Softmax 只支持 contiguous FP16/FP32 Tensor 的最后一维，不支持任意 dim、stride 或 autograd。
- Softmax FP16 当前全部进入 v1 Shared Memory Block；v4 Online/float4 只用于满足长度和实际地址对齐要求的 FP32 输入。
- GEMM 只支持二维 contiguous FP16/FP32 输入，输出固定为 FP32，不支持 bias、batched GEMM 或 autograd。
- GEMM 的 `transpose_b=True` 当前进入 CUDA Core tiled fallback，不能使用现有 Tensor Core 路径；这使直接适配 `nn.Linear` 权重布局时明显慢于 PyTorch Linear/cuBLAS。
- 当前 GEMM selector 没有调用 cuBLAS fallback。Transformer Block 中的 GEMM-only 类是工程对照，不是推荐的高性能模型入口。

## Transformer Block 限制

- 示例仅覆盖 causal、pre-norm、FP16 inference forward，head dim 固定为 64 或 128。
- 未实现 batch/sequence padding mask、dropout、缓存式 decode、KV cache 或训练反向。
- Softmax-only 版本会显式物化 `[B,H,N,N]` score/probability，只用于 unfused 对照，不适合长序列生产推理。
- benchmark 中的 Attention/GEMM/Softmax 路径标签是 selector 推导元数据，不是 profiler 观测证据。

## 性能解释限制

- benchmark 报告的是 warmup 后 CUDA Event 延迟，不包含首次 Triton JIT 编译时间。
- causal Attention 的 `dense_tflops` 按完整稠密 QK/PV 工作量计算；会跳过未来 tile 的实现实际执行工作量更少，因此应优先比较同一 case 的 latency。
- `expected_dispatch_path` 由当前规则推导。若需要证明实际 kernel，必须使用 profiler 或额外 instrumentation。
- 小于约 1 ms 的 Block benchmark 容易受到 GPU 时钟、功耗状态和 provider 测量顺序影响；当前数据用于工程判断，不宣称跨运行稳定领先。
