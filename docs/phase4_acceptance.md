# Phase 4 Attention 验收记录

验收日期：2026-09-17。

## 完成范围

- Dispatcher schema：`attention(Tensor q, Tensor k, Tensor v, bool causal=False) -> Tensor`。
- 输入接口：Q/K/V 为 `[B,H,N,D]`，FP16、contiguous、D64/128，支持 causal/non-causal inference forward。
- C++ wrapper：输入契约、CUDA device guard、输出分配、空 Tensor、PyTorch current CUDA stream。
- CUDA Core：从旧 FP32 `[N,D]` Online Attention 迁移为 FP16 `[B,H,N,D]`，FP32 Softmax/output accumulation。
- Tensor Core：迁移固定 D64 的 WMMA QK/PV 路径，并适配 B/H、FP16 输入输出和 current stream。
- Triton：迁移 FlashAttention 候选后端，修复旧实现中的状态变量和 store mask 错误。
- fallback：C++ 长序列显式进入 PyTorch SDPA。
- 统一 Python API：`my_ops.attention(..., provider="auto|cpp|triton|sdpa")`。
- FakeTensor、`torch.library.opcheck`、边界长度、空 Tensor、非法输入和 non-default stream 测试。

## 当前调度

低层 `torch.ops.my_ops.attention`：

```text
N > 1024                         -> PyTorch SDPA fallback
N <= 1024 且 D=64 且 N>=32      -> Tensor Core
其他合法输入                     -> CUDA Core
```

推荐的 `my_ops.attention(provider="auto")`：

```text
causal=True                      -> Triton
causal=False 且 N<=512           -> Triton
causal=False 且 N>512            -> PyTorch SDPA
```

详细规则与证据见 `docs/dispatch_policy.md`，不支持范围见 `docs/limitations.md`。

## 验收结果

用户在实际 WSL PyTorch/CUDA 环境中完成以下统一测试：

```bash
pytest \
  tests/test_attention.py \
  tests/test_attention_triton.py \
  tests/test_attention_functional.py \
  -v
```

结果：`59 passed`。

新版 `benchmarks/benchmark_attention.py` 已完成运行，结果写入 `benchmarks/results/attention_candidates.csv`。该版本同时覆盖 PyTorch SDPA、低层 C++ 自动路径、显式 Triton 和统一 Python auto provider。

## 延后事项

- CUDA Core/Tensor Core 中大 shape 性能优化。
- Triton 与 SDPA 更细粒度的 crossover 校准。
- 跨 GPU 的共享内存能力探测与 selector 校准。
- backward/autograd、GQA/MQA、KV cache、ragged/paged Attention。
- `torch.compile` 与 `torch.library.triton_op` 集成验证。

这些事项不阻塞 Phase 4 工程闭环，统一留到全部算子与 Transformer Block 完成后处理。
