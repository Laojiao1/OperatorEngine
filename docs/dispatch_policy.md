# OperatorEngine Dispatch Policy

## Vector Add

当前只有一个 FP32 标量 CUDA kernel：

```text
CUDA + FP32 + same shape/device + contiguous
-> scalar vector_add kernel
```

每个线程处理一个元素，block size 固定为 256，grid 使用 ceil-div。Vector Add 暂时没有多 kernel selector。

## Softmax

### 输入契约

- CUDA only。
- FP32 或 FP16。
- contiguous。
- `ndim >= 1`。
- 只沿最后一维计算。
- 不支持 autograd。
- 空 Tensor 直接返回，不启动 kernel。

### 当前 selector

```text
FP32
+ N >= 1024
+ N % 4 == 0
+ input.data_ptr 16 Byte aligned
+ output.data_ptr 16 Byte aligned
-> v4 Online + Warp Shuffle + float4

其他合法输入
-> v1 Shared Memory Block Reduction
```

检查实际 `data_ptr()` 而不只检查 `contiguous`，是因为带非零 storage offset 的 contiguous view 仍可能破坏 `float4` 所需的 16 Byte 对齐。

FP16 当前全部进入 v1：v4 旧实现只提供 FP32 `float4` 路径。为了避免隐式转换或错误 reinterpret cast，selector 不允许 FP16 进入 v4。

### 两条路径的工程差异

v1 保留阶段四的三阶段结构：

```text
读取 input 求 row_max
-> 再读 input 求 exp_sum
-> 再读 input 归一化并写 output
```

它使用 256 个线程和动态 Shared Memory 完成两次树形规约。按算法访问次数计算，FP32 最多涉及三次输入读取和一次输出写入；实际 DRAM 流量会受到 cache 影响。

v4 保留阶段四的 Online `(m, d)` 状态、warp shuffle 和 `float4`：

```text
第一次读取 input 在线合并 max/sum 状态
-> block 级合并
-> 第二次读取 input 并归一化写 output
```

它减少一次算法输入遍历，但增加 Online 状态合并中的 `expf` 和依赖链，因此并非所有 shape 都稳定快于 v1。

### 首轮证据与限制

在约 8M 总元素的 FP32 benchmark 中：

| N | v4 Online | v1 unaligned fallback | 首轮差异 |
| ---: | ---: | ---: | ---: |
| 1024 | 0.2299 ms | 0.2372 ms | v4 快约 3.1% |
| 2048 | 0.2283 ms | 0.2230 ms | v4 慢约 2.4% |
| 4096 | 0.2271 ms | 0.2328 ms | v4 快约 2.5% |
| 8192 | 0.2225 ms | 0.2304 ms | v4 快约 3.4% |

差距接近运行波动，而且 v1 对照使用未对齐 input，当前阈值属于可解释的初始策略，不是最终最优结论。后续只有经过重复 benchmark 或 profiler 证据，才能修改 selector。

完整数据见 `benchmarks/results/softmax.csv` 和 `docs/performance_report.md`。

## GEMM

### 输入与输出契约

- A/B 均为二维、CUDA、contiguous、同 device、同 dtype Tensor。
- 支持 FP16 和 FP32输入，输出统一为 FP32。
- `transpose_b=false` 时 B 为 `[K, N]`；`transpose_b=true` 时 B 以 `[N, K]` 存储并计算 `A @ B^T`。
- 第一版要求 `K > 0`，不支持 autograd。
- `M=0` 或 `N=0` 时直接返回空输出，不启动 kernel。

### 当前 selector

```text
FP16 + transpose_b=false + A/B data_ptr 16 Byte aligned
├─ M*N*K < 1M
│  -> v1 Shared Memory tiled
├─ M < 1024 或 N < 1024
│  ├─ M%64==0 + N%64==0 + K%32==0 + C data_ptr 32 Byte aligned
│  │  -> 64x64 Tensor Core aligned
│  └─ 其他 shape -> 64x64 Tensor Core general
└─ M >= 1024 且 N >= 1024
   ├─ M%128==0 + N%128==0 + K%32==0 + C data_ptr 32 Byte aligned
   │  -> 128x128 Tensor Core aligned
   └─ 其他 shape -> 128x128 Tensor Core general

FP32、transpose_b=true 或 FP16 输入地址未对齐
-> v1 Shared Memory tiled fallback
```

selector 检查实际 `data_ptr`，因为 contiguous Tensor 仍可能带非零 storage offset。Tensor Core general 内部也可能执行 16 Byte 向量化读取，不能只检查逻辑 stride。

`1M` 工作量阈值来自同轮 crossover：约 `0.56M` 工作量时 v1 更快，`128^3≈2.10M` 时 small Tensor Core 比强制 v1 快约 1.14 倍。它是当前硬件上的经验策略，不是硬件定律。

64x64 small kernel 使用 128 个线程和 4 个 warp，每个 warp 持有 2x2 个 accumulator fragment。它用于增加中小矩阵的并行 Block 数量并减少 M/N 尾块浪费。128x128 large kernel仍用于大矩阵，以保持更高数据复用和更少 Block 调度开销。

复测中 small kernel 相对旧 128x128 路径使 `192³~512³` 延迟降低约 8%--18%，并使 decode `M=1/16` 延迟降低约 41%--43%。完整数据见 `benchmarks/results/gemm_small_tile.csv`。

## Attention

### 输入契约

- Q、K、V 均为 `[B,H,N,D]` 的 CUDA Tensor，shape、device 和 dtype 完全相同。
- 第一版只支持 FP16、contiguous、`D=64/128`，中间 Softmax 和输出累加使用 FP32。
- 支持 causal 与 non-causal inference forward，不支持 autograd。
- 任意维度为 0 时返回相同 shape 的空输出，不启动 kernel。

### C++ Dispatcher 路径

`torch.ops.my_ops.attention` 保留完整的 Dispatcher、C++ wrapper、current stream 和自定义 CUDA 学习闭环：

```text
N > 1024
-> PyTorch SDPA fallback

N <= 1024 + D=64 + N>=32
-> Tensor Core Online Attention

其他合法输入
-> CUDA Core Online Attention
```

CUDA Core 路径使用标量 FP16 全局读写、FP32 Shared Memory 和 FP32 Online Softmax，不要求额外的向量地址对齐。Tensor Core 路径固定 `Br=32/Bc=32/D=64`、256 threads，Q/K/V/P 使用 FP16 Shared Memory，scores、Online 状态和输出累加使用 FP32；静态 Shared Memory 约 38.8 KiB。WMMA 只读取 16 Byte 对齐的静态 Shared Memory tile，不直接从任意 global `data_ptr` 执行 WMMA load。

当前 Tensor Core 资源配置已在项目目标 GPU `sm_120` 上验证。它不是面向所有 NVIDIA GPU 的自动资源探测策略；迁移到共享内存上限更低的设备时，应增加设备能力检查并 fallback。

### 统一 Python selector

推荐入口是 `my_ops.attention(..., provider="auto")`：

```text
causal=True
-> Triton FlashAttention

causal=False + N<=512
-> Triton FlashAttention

causal=False + N>512
-> PyTorch SDPA
```

也可以显式指定：

- `provider="cpp"`：调用低层 C++ Dispatcher 算子。
- `provider="triton"`：调用 Triton FlashAttention。
- `provider="sdpa"`：调用 PyTorch SDPA。

所有 provider 在选择前共享同一套 Python 输入检查，因此显式 SDPA 不会绕过 OperatorEngine 的 FP16、D64/128、contiguous 和 no-autograd 契约。

### 首轮性能证据

RTX 5060 Laptop GPU 上的首轮候选 benchmark 表明：Triton 在 8 组 case 中有 7 组快于 SDPA，约为 `1.13x~1.74x`；例外是 non-causal `B=1,H=16,N=1024,D=128`，Triton 为 `0.4614 ms`，SDPA 为 `0.2893 ms`。因此 auto 使用保守的 causal/sequence-length 规则，而不是全量选择 Triton。

C++ 自定义路径用于工程学习与后续优化实验。除短序列 D64 外，首轮数据普遍慢于 SDPA；最明显的 `N=1024,D=128` CUDA Core case 为 `5.0829 ms`，SDPA 为 `0.2893 ms`。该性能欠账按项目约定延后到所有算子闭环完成后处理。

完整数据见 `benchmarks/results/attention_candidates.csv`。CSV 中的 `expected_dispatch_path` 是依据 selector 推导的预期路径，不等同于 profiler 观测证据。
