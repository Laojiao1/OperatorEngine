# Phase 5 最小 Transformer Block 验收记录

验收日期：2026-09-18。

## 完成范围

- 纯 PyTorch causal pre-norm decoder block：RMSNorm、QKV projection、RoPE、SDPA、output projection、SwiGLU MLP 和 residual。
- Attention-only：只将 PyTorch SDPA 替换为 `my_ops.attention(provider="auto")`。
- GEMM-only：将五处无 bias Linear 替换为 `torch.ops.my_ops.gemm`，Attention 仍使用 PyTorch SDPA。
- Attention + GEMM：通过两个已验证 hook 的组合，让两类自定义算子同时进入真实计算图。
- Softmax-only unfused 对照：保持显式 QK^T、causal mask 和 PV 不变，只比较 `torch.softmax` 与 `torch.ops.my_ops.softmax`。
- 所有对照使用相同输入、相同权重和相同 FP16 输出契约。

## 正确性验收

用户在实际 WSL PyTorch/CUDA 环境中运行：

```bash
pytest tests/test_transformer_block.py -v
```

结果：`32 passed`。

测试覆盖基础数学组件、causal 语义、输出 shape/dtype/device、参数结构、方法路由，以及 Attention、GEMM、组合和 Softmax-only 版本的同权重端到端数值比较。

## 端到端性能记录

逐项替换的最终同轮对照见 `benchmarks/results/transformer_block_combined.csv`：

| Case | Attention-only | GEMM-only | Attention + GEMM |
| --- | ---: | ---: | ---: |
| tiny_n17_d64 | 1.022x | 0.837x | 0.820x |
| prefill_n128_d64 | 1.011x | 0.508x | 0.510x |
| prefill_n512_d64 | 0.987x | 0.210x | 0.211x |
| prefill_n128_d128 | 0.968x | 0.431x | 0.437x |
| prefill_n1024_d128 | 1.037x | 0.142x | 0.140x |

表中数值均为相对同轮 PyTorch baseline 的 Block 级 speedup。Attention-only 的变化约在 -3.2% 到 +3.7% 内；组合版本与 GEMM-only 基本重合，说明当前端到端性能由五次 GEMM 主导。

Softmax 未融合对照见 `benchmarks/results/transformer_block_softmax.csv`。相对完全相同的 Torch unfused 图，自定义 Softmax 的 speedup 依次为：

```text
N17/D64    1.395x
N128/D64   0.881x
N512/D64   1.036x
N128/D128  0.868x
N1024/D128 0.955x
```

除 tiny case 外，自定义 Softmax 在 Block 级只表现为小幅收益或退化，不能据单点结果建立普遍更快的结论。中长序列下，fused SDPA 相对 Torch unfused Attention 约快 1.03x 到 1.23x。

## 性能解释

- 当前 Linear 接入必须使用 `transpose_b=True` 适配 `[out_features, in_features]` 权重，因此进入 `v1_tiled_transpose_b`，没有使用 Tensor Core 快路径。
- 自定义 GEMM 固定输出 FP32，Block 每次 Linear 后还会执行 FP32 到 FP16 转换；五次调用会累计该成本。
- PyTorch Linear 使用成熟的 cuBLAS/cuBLASLt 实现，当前自定义 transpose-b tiled kernel 不具备竞争力。
- unfused Attention 需要显式物化 score、causal mask 和 probability，并分别启动 QK^T、Softmax、PV；单独 Softmax 的局部变化不会等比例转化为 Block 收益。
- Attention-only 的局部收益占 Block 总时间比例有限，因此符合 Amdahl 定律：kernel 局部加速不会直接变成相同倍数的端到端加速。
- CSV 中的 `expected_*_path` 来自当前 selector 和输入契约推导，不是 profiler 对真实 kernel 执行的观测证据。

## 验收结论

Phase 5 已完成。自定义 Attention、Softmax 和 GEMM 均已进入真实 Transformer Block，并具有逐项 correctness 与端到端性能记录。实验同时保留了成功和失败结果：Attention 在部分 shape 有小幅收益，Softmax 收益依赖 shape，当前 GEMM 路径则应在正式接口中主动回退 PyTorch Linear/cuBLAS。

## 延后事项

- 为 Linear 权重建立适配 Tensor Core 的预转置或预打包布局。
- 消除 GEMM FP32 输出到 FP16 的额外转换。
- 使用 profiler 验证真实 dispatch path、kernel time、layout conversion 和各子算子占比。
- 对 benchmark 做随机顺序、多轮统计和 GPU 时钟控制，降低小延迟 case 的顺序与频率波动。
- backward/autograd、KV cache、continuous batching 和完整模型集成。

这些性能优化不阻塞 Phase 5 工程闭环，统一留到工程收尾后再处理。
