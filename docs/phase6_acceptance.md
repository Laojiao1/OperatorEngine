# Phase 6 工程收尾验收记录

验收日期：2026-09-18。

## 安装与全量测试

用户在实际 WSL/CUDA 环境中重新执行 editable 安装：

```bash
pip install -e . --no-build-isolation
```

安装成功。随后使用单条命令运行完整测试：

```bash
pytest -q
```

结果：`184 passed, 1 skipped in 6.52s`。

唯一跳过项是需要至少两张 CUDA GPU 的跨设备输入检查。当前项目使用单张 RTX 5060 Laptop GPU，该跳过原因明确，不影响单 GPU 首版验收。

## 一键核心 benchmark

新增统一入口：

```bash
python benchmarks/run_core_benchmarks.py
```

用户已完成实际运行，五份结果均成功写入 `benchmarks/results/core/`：

| 文件 | 行数 | 覆盖重点 |
| --- | ---: | --- |
| `vector_add.csv` | 9 | PyTorch、CUDA scalar、Triton |
| `softmax.csv` | 40 | v1、v4、未对齐 fallback |
| `gemm.csv` | 58 | v1、small/large Tensor Core、aligned/general |
| `attention.csv` | 32 | CUDA Core、Tensor Core、SDPA fallback、Triton/SDPA auto |
| `transformer_block.csv` | 20 | baseline、Attention-only、GEMM-only、组合版本 |

所有记录的 latency 均为有限正数，没有空 CSV、NaN、零延迟或负延迟。

## 工程与文档检查

- README 已包含项目目标、Vector Add/Softmax/GEMM/Attention 支持矩阵、构建方式、完整测试命令、Transformer Block 示例和核心 benchmark 入口。
- `docs/architecture.md` 解释 Dispatcher、C++ wrapper、selector、launcher、kernel 和模型集成的职责边界。
- `docs/dispatch_policy.md` 记录 Softmax、GEMM 和 Attention 的当前规则及 benchmark 证据。
- `docs/performance_report.md` 同时保留成功与失败实验，没有只展示最优结果。
- `docs/limitations.md` 记录 dtype、layout、autograd、GEMM transpose-b、unfused Attention 和硬件适用边界。
- C++/CUDA 扩展中不存在 benchmark `main()` 或无条件 `cudaDeviceSynchronize()`；kernel launch 统一使用 PyTorch/CUDA launch check。
- 所有代码文件开头均具有文件职责说明，关键工程边界使用中文注释。

## 工作区清理

- `.gitignore` 已覆盖 `__pycache__/`、`.pytest_cache/`、`build/`、`dist/`、`*.egg-info/`、`*.so`。
- 最终保留 `my_ops/_C...so`，因为它是当前 editable 安装实际加载的原生扩展，而不是无用缓存。
- 测试和 benchmark 产生的 Python/pytest 缓存在最终验收后删除；它们均可安全重建。

## 最终结论

OperatorEngine Phase 0 到 Phase 6 已完成。项目已从独立 CUDA/Triton 实验形成可安装、可调度、可测试、可 benchmark、可组合进 Transformer Block 的单 GPU PyTorch 算子库。

当前性能结果同样明确：Attention 在部分 shape 有小幅端到端收益；Softmax 收益依赖 shape；直接用 `transpose_b=True` tiled GEMM 替换 Linear 会显著退化。后续优化应以 profiler 证据为起点，优先处理 Linear 权重布局、Tensor Core 路径和不必要的 FP32 到 FP16 转换，而不是继续增加未经端到端验证的孤立 kernel。
