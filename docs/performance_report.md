# Vector Add 性能记录

## 测试环境

- 日期：2026-09-12。
- GPU：NVIDIA GeForce RTX 5060 Laptop GPU，compute capability 12.0。
- PyTorch：2.11.0+cu128。
- Triton：3.6.0。
- CUDA Toolkit / nvcc：12.8 / 12.8.61。
- dtype：FP32。
- warmup / repeat：100 ms / 300 ms。

## 对比口径

三条路径使用相同输入、shape、dtype 和当前 CUDA stream：

- `torch.add`。
- `my_ops::vector_add` 标量 CUDA kernel。
- 从阶段三 Vector Add 算法迁移并适配 Triton 3.6 API 的 Triton baseline。

每个 shape 在计时前都与 `a + b` 做 correctness 检查。逻辑有效带宽按每个元素读取两个 FP32、写入一个 FP32，即 `12 * N` Bytes 计算。这不是硬件计数器测得的实际 DRAM 流量。

## 首轮结果

| 规模 | N | 实现 | latency (ms) | effective GB/s |
| --- | ---: | --- | ---: | ---: |
| small | 4,096 | torch.add | 0.0080 | 6.12 |
| small | 4,096 | my_ops CUDA scalar | 0.0071 | 6.95 |
| small | 4,096 | Triton adapted | 0.0073 | 6.70 |
| medium | 1,048,576 | torch.add | 0.0527 | 238.62 |
| medium | 1,048,576 | my_ops CUDA scalar | 0.0507 | 248.33 |
| medium | 1,048,576 | Triton adapted | 0.0525 | 239.61 |
| large | 16,777,216 | torch.add | 0.7055 | 285.35 |
| large | 16,777,216 | my_ops CUDA scalar | 0.6791 | 296.45 |
| large | 16,777,216 | Triton adapted | 0.6929 | 290.55 |

原始数据保存在 `benchmarks/results/vector_add.csv`。

## 结论与限制

- 小输入只有约 7--8 微秒，固定 launch、Dispatcher 和输出分配开销占主导，有效带宽很低。
- 输入增大后，三个实现进入相近的带宽区间，符合 Vector Add 算术强度低、主要受内存通路限制的预期。
- 首轮中自定义 CUDA scalar 数值略好，但差距只有约 4%，尚未做跨次运行、功耗状态和误差分布控制，不能表述为稳定领先。
- 阶段三旧实现向 `tl.store` 传递了当前 Triton 3.6 不接受的 `other` 参数。旧学习档案保持不动；benchmark 使用相同 grid/mask 算法的本地兼容版本，并明确标注为 adapted。
- Nsight、向量化和 block-size 消融不属于 Phase 1 的正确性闭环，后续只有在提出具体性能问题时再展开。

# Softmax 首轮性能记录

## 对比配置

- 总元素数固定约为 `8M`，列宽扫描 `128/512/1024/2048/4096/8192`。
- dtype：FP32、FP16。
- 对比：`torch.softmax`、`my_ops::softmax`、阶段三 Triton adapted baseline。
- warmup / repeat：100 ms / 300 ms。
- 统一有效带宽按 Softmax 语义的一次输入读取和一次输出写回计算，不代表真实 DRAM 流量。

完整原始数据保存在 `benchmarks/results/softmax.csv`。

## 当前 selector 观察

FP32 对齐输入中，v4 Online 相对强制走 v1 的未对齐输入：

| N | v4 Online (ms) | v1 Block fallback (ms) | 首轮观察 |
| ---: | ---: | ---: | --- |
| 1024 | 0.2299 | 0.2372 | v4 快约 3.1% |
| 2048 | 0.2283 | 0.2230 | v4 慢约 2.4% |
| 4096 | 0.2271 | 0.2328 | v4 快约 2.5% |
| 8192 | 0.2225 | 0.2304 | v4 快约 3.4% |

差距整体只有几个百分点，而且 v1 对照使用了未对齐 storage offset，不能把单轮结果解释成纯 kernel 算法差异。当前 `N >= 1024` 的 v4 selector 暂时保留，后续需要多轮重复、对齐条件可控的内部实现对比再固化阈值。

## Shape-dependent 结论

- FP32 `N=128` 时 v1 为 `1.0003 ms`，明显慢于 PyTorch 的 `0.2308 ms`。固定 256-thread block 在大量短行上成本过高，短行应成为后续 warp-specialized 路径的重点。
- FP32 `N=1024~8192` 时自定义实现约为 `0.2225~0.2299 ms`，与 PyTorch/Triton 处于同一区间。
- FP16 v1 从 `N=128` 的 `1.0585 ms` 逐步改善到 `N=8192` 的 `0.1265 ms`；短行仍明显落后 PyTorch，长行则具有竞争力。
- 这些数据只是一轮运行，用于暴露趋势和选择下一项实验，不作为稳定 speedup 结论。

# GEMM 首轮性能记录

## 对比配置

- A/B 为 row-major contiguous Tensor，输出统一为 FP32。
- FP16 非转置输入走 Tensor Core aligned/general selector。
- FP16 转置和 FP32 输入走 v1 Shared Memory tiled fallback。
- PyTorch FP16 reference 使用 `torch.mm(..., out_dtype=torch.float32)`，保持输出 dtype 一致。
- warmup / repeat：100 ms / 300 ms。

完整 Tensor Core 数据保存在 `benchmarks/results/gemm_tensor_core.csv`。

## Tensor Core 与 PyTorch

| case | M | N | K | my_ops path | torch.mm (ms) | my_ops (ms) | my_ops TFLOPS | my_ops / torch |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| aligned small | 128 | 128 | 32 | aligned | 0.0091 | 0.0107 | 0.098 | 0.851x |
| general boundary | 129 | 131 | 33 | general | 0.0153 | 0.0166 | 0.067 | 0.919x |
| square 512 | 512 | 512 | 512 | aligned | 0.0217 | 0.0291 | 9.233 | 0.747x |
| square 2048 | 2048 | 2048 | 2048 | aligned | 0.6298 | 0.6377 | 26.938 | 0.987x |
| decode M=1 | 1 | 4096 | 4096 | general | 0.1617 | 0.4023 | 0.083 | 0.402x |
| decode M=16 | 16 | 4096 | 4096 | general | 0.1203 | 0.4014 | 1.338 | 0.300x |

## 与 v1 tiled 的迁移对照

第一次 benchmark 时 launcher 漏接 Tensor Core selector，因此 Python 标注为 Tensor Core 的自定义数据实际全部来自 v1 tiled。这次意外运行提供了同一环境下的阶段性 baseline，但原 CSV 中的路径标签不是真实 profiler 观测值。

| case | v1 tiled (ms) | Tensor Core (ms) | Tensor Core 相对 v1 |
| --- | ---: | ---: | ---: |
| aligned small | 0.0101 | 0.0107 | 0.95x |
| general boundary | 0.0126 | 0.0166 | 0.76x |
| square 512 | 0.2182 | 0.0291 | 7.50x |
| square 2048 | 11.5282 | 0.6377 | 18.08x |
| decode M=1 | 0.9027 | 0.4023 | 2.24x |
| decode M=16 | 0.9018 | 0.4014 | 2.25x |

## 当前结论

- `2048^3` aligned Tensor Core 达到约 `26.94 TFLOPS`，本轮只比 `torch.mm` 慢约 1.3%，证明迁移后的 WMMA、padding 和 cp.async 双缓冲路径确实生效。
- `512^3` 相对 v1 提升约 7.5 倍，但仍比 PyTorch 慢约 34%；大矩阵吞吐不能直接外推到中小矩阵。
- `M=1/16` 时两个 case 都约为 0.40 ms，说明 128 行输出 tile 在小 M 下存在明显无效计算。Tensor Core 仍比当前 v1 快约 2.25 倍，因此在没有专用 small-M kernel 前不能简单回退 v1。
- FP16 transpose 只能走 v1，约比 PyTorch 慢 12.4 倍；FP32 v1 在 `512^3` 约慢 5.1 倍。这两类输入是当前支持范围中的主要性能缺口。
- 十几微秒的小矩阵主要受 Dispatcher、输出分配和 kernel launch 固定开销影响。v1 与 Tensor Core 的差异来自两次独立运行，暂不足以固化小矩阵阈值。

## 64x64 Small Tensor Core 复测

在同一进程中使用未对齐 contiguous FP16 输入强制走 v1 tiled，并与正常对齐输入的 Tensor Core 路径比较。完整数据保存在 `benchmarks/results/gemm_small_tile.csv`。

| case | 旧 128 tile (ms) | 新 64 tile (ms) | 延迟改善 | 新实现 / torch |
| --- | ---: | ---: | ---: | ---: |
| 128³ | 0.0149 | 0.0140 | 6.2% | 0.645x |
| 192³ | 0.0213 | 0.0175 | 18.0% | 0.889x |
| 256³ | 0.0190 | 0.0168 | 11.7% | 0.772x |
| 384³ | 0.0233 | 0.0208 | 10.8% | 0.890x |
| 512³ | 0.0286 | 0.0262 | 8.5% | 0.795x |
| decode M=1 | 0.4023 | 0.2365 | 41.2% | 0.673x |
| decode M=16 | 0.4014 | 0.2283 | 43.1% | 0.552x |

64x64 tile 的收益来自更多并行 Block、更少 M/N 尾块浪费，以及每个 warp 更少的 accumulator fragment。decode 两个 case 的延迟仍很接近，因为两者都只占用 64 行 tile 中的一部分，但浪费已由 128 行降为 64 行。

同轮 Tensor Core 相对强制 v1 的收益从 `128³` 的约 1.14 倍增长到 `512³` 的约 7.81 倍，支持当前 `1M` 工作量阈值。低于阈值的 `64³`、`128x128x32` 和 `129x131x33` 保持 v1。

`2048³` 本轮 large kernel 为 `0.5907 ms / 29.08 TFLOPS`，PyTorch 为 `0.5311 ms / 32.35 TFLOPS`。两者都比前一轮更快，说明跨轮绝对时间受到 GPU 时钟和功耗状态影响；selector 判断应优先使用同轮路径对比，而不是把跨轮差值全部归因于代码。

# Attention 与 Transformer Block 性能记录

Attention 候选后端的原始数据保存在 `benchmarks/results/attention_candidates.csv`，完整 Phase 4 结论见 `docs/phase4_acceptance.md`。Triton 在多数首轮 case 中优于 C++ 学习路径，但并非所有 shape 都优于 PyTorch SDPA，因此统一 Python auto selector 保留显式 fallback。

Transformer Block 的逐项替换数据保存在：

- `benchmarks/results/transformer_block_attention.csv`
- `benchmarks/results/transformer_block_gemm.csv`
- `benchmarks/results/transformer_block_combined.csv`
- `benchmarks/results/transformer_block_softmax.csv`

最终同轮组合对照中，Attention-only 相对 PyTorch baseline 为约 `0.968x~1.037x`；GEMM-only 为 `0.142x~0.837x`；Attention + GEMM 为 `0.140x~0.820x`。组合版本与 GEMM-only 基本重合，说明当前五次 `transpose_b=True` tiled GEMM 的成本主导整个 Block。

Softmax-only 对照保持 QK^T、causal mask 和 PV 完全相同。自定义 Softmax 相对 Torch unfused 版本在五个 case 中为 `0.868x~1.395x`，表现依赖 score 的行数和列宽；除 tiny case 外只有小幅收益或退化。中长序列下，fused SDPA 相对 Torch unfused Attention 约快 `1.03x~1.23x`。

这些结果同时保留成功与失败实验：Attention 在部分 shape 有小幅 Block 收益，Softmax 没有普遍优势，当前 GEMM 直接替换 Linear 则显著退化。详细验收口径、逐 case 数据与延后事项见 `docs/phase5_acceptance.md`。
