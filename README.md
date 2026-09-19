# OperatorEngine

OperatorEngine 是一个用于学习 PyTorch 算子工程化的 C++/CUDA 扩展项目。第一条闭环从 Vector Add 开始，目标是把已有的 kernel 实验组织成可安装、可调度、可测试的 PyTorch 自定义算子，而不是继续堆叠独立的 `.cu` 示例。

当前里程碑：项目已经形成安装、Dispatcher、current CUDA stream、correctness/opcheck、benchmark 和最小 Transformer Block 的完整闭环。

## 首版支持矩阵

| 维度 | Vector Add 第一版 |
| --- | --- |
| Device | NVIDIA CUDA only |
| 输入 | 两个 shape 相同的 `torch.Tensor` |
| Dtype | `torch.float32` only |
| Layout | contiguous only |
| 输出 | 与输入 shape、dtype、device 相同的新 Tensor |
| Autograd | not supported |
| Mutation / aliasing | 不修改输入，输出不与输入 alias |
| CPU | 不提供实现，调用应明确报错 |

不支持的 dtype、layout、device 或 shape 组合必须明确报错，不能静默转换或计算错误。

Softmax 当前支持矩阵：

| 维度 | Softmax 第一版 |
| --- | --- |
| Schema | `softmax(Tensor input, int dim) -> Tensor` |
| Device | NVIDIA CUDA only |
| Dtype | FP32、FP16；归约使用 FP32 accumulation |
| Dim | 只支持最后一维，接受 `-1` 或对应正索引 |
| Rank | `ndim >= 1`，前导维度展平成 row |
| Layout | contiguous only |
| Autograd | not supported |
| 实现 | v1 Shared Memory Block、v4 Online/Warp Shuffle/float4 |
| 编译契约 | Python FakeTensor implementation + `torch.library.opcheck` |

Softmax 的当前选择规则和证据见 `docs/dispatch_policy.md`。

GEMM 当前支持矩阵：

| 维度 | GEMM 第一版 |
| --- | --- |
| Schema | `gemm(Tensor a, Tensor b, bool transpose_b=False) -> Tensor` |
| Device | NVIDIA CUDA only，A/B 必须位于同一设备 |
| Dtype | FP16、FP32 输入，统一输出 FP32 |
| Shape | A/B 均为二维；支持 `A @ B` 与 `A @ B.T` |
| Layout | contiguous only，Tensor Core 路径额外检查实际地址对齐 |
| 后端 | CUDA Core tiled、Tensor Core aligned/general |
| Autograd | not supported |

GEMM 的当前 selector、真实 Transformer shape 数据和性能缺口见 `docs/dispatch_policy.md` 与 `docs/performance_report.md`。

Attention 当前支持矩阵：

| 维度 | Attention 第一版 |
| --- | --- |
| Schema | `attention(Tensor q, Tensor k, Tensor v, bool causal=False) -> Tensor` |
| Python API | `my_ops.attention(q, k, v, causal=False, provider="auto")` |
| Shape | Q/K/V 均为 `[B,H,N,D]`，shape 完全相同 |
| Dtype | FP16 输入输出；FP32 Softmax/output accumulation |
| Head dim | 64 或 128 |
| Layout | contiguous only |
| 模式 | causal / non-causal inference forward |
| 后端 | CUDA Core、Tensor Core、Triton、PyTorch SDPA fallback |
| Provider | `auto`、`cpp`、`triton`、`sdpa` |
| Autograd | not supported |

推荐调用：

```python
import my_ops

output = my_ops.attention(q, k, v, causal=True, provider="auto")
```

Attention 的 C++ 与 Python 两层选择规则、性能证据和限制见 `docs/dispatch_policy.md` 与 `docs/limitations.md`。

## Transformer Block 集成

`examples/transformer_block.py` 提供纯 PyTorch baseline，以及 Attention-only、GEMM-only、Softmax-only unfused 和 Attention + GEMM 对照版本。Phase 5 的正确性、端到端数据和性能解释见 `docs/phase5_acceptance.md`。

运行 Block 测试：

```bash
pytest tests/test_transformer_block.py -v
```

运行逐项替换的最终对照：

```bash
python benchmarks/benchmark_transformer_block_combined.py \
  --output benchmarks/results/transformer_block_combined.csv
```

用一条命令运行五组核心 benchmark：

```bash
python benchmarks/run_core_benchmarks.py
```

该命令依次运行 Vector Add、Softmax、GEMM、Attention 和最终 Transformer Block 对照，结果写入 `benchmarks/results/core/`。可用 `--benchmark` 选择子集，或用 `--warmup-ms`、`--repeat-ms` 调整测量时长。

## 为什么先做 Vector Add

Vector Add 的数学和 kernel 很简单，因此可以把注意力放在真正的工程边界上：Dispatcher schema、C++ 输入检查、Tensor 生命周期、CUDA device guard、PyTorch current CUDA stream、launch error，以及 correctness/opcheck 的职责区别。等这条链路可解释且可测试后，再迁移 Softmax、GEMM 和 Attention。

## 计划中的调用形式

以 vector_add 算子为例：

```python
import torch
import my_ops

a = torch.randn(1003, device="cuda", dtype=torch.float32)
b = torch.randn_like(a)
out = torch.ops.my_ops.vector_add(a, b)
torch.testing.assert_close(out, a + b)
```

## 构建与验证

在 WSL 中构建并运行完整测试：

```bash
cd /mnt/e/Code/C++Compiler/OperatorEngine

pip install -e . --no-build-isolation

pytest -v

python benchmarks/benchmark_vector_add.py
```

当前单 GPU 环境下，跨设备输入测试会明确跳过；其余测试必须通过。不要使用 Windows `pytorch` conda 环境构建，原因见 `docs/architecture.md`。

指令解释：

- **`pip install -e .`（Editable 模式）**： 不会把代码复制到 Python 的全局 `site-packages` 目录中，而是在 `site-packages` 中写入一个指向当前仓库路径的文件（`.pth` 文件）。这样 Python 在 `import my_ops` 时，直接加载当前工程目录下的代码和刚生成的 `.so` 文件。

- **`--no-build-isolation`（关闭构建隔离）**： 默认情况下，pip 会新建一个空环境去下载依赖。加上这个参数后，pip 直接使用当前 Conda 环境里已经安装的 PyTorch 和系统的 CUDA 工具链。这样能保证编译时使用的 `libtorch` 版本和运行时的 PyTorch 版本完全一致。
