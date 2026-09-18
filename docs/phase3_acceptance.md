# Phase 3 GEMM 验收记录

验收状态：已通过（2026-09-16）。

## 已完成

- Dispatcher schema：`gemm(Tensor a, Tensor b, bool transpose_b=False) -> Tensor`。
- 二维 row-major contiguous 契约、`transpose_b` 语义和 FP32 输出约定。
- CUDA device guard、PyTorch current CUDA stream 和异步 launch error 检查。
- FP32/FP16 v1 Shared Memory tiled 通用路径。
- FP16 64x64 与 128x128 Tensor Core aligned/general 路径。
- dtype、shape、alignment 和工作量 selector。
- FakeTensor、opcheck、空输出、非默认 stream、未对齐地址和边界 shape 测试。
- PyTorch `torch.mm` correctness/performance 对比。
- 方阵、非规则边界、prefill-like 和 decode-like benchmark。

## 最终验收命令

```bash
pip install -e . --no-build-isolation
pytest -q
python benchmarks/benchmark_gemm.py --case qkv_prefill --case output_prefill --case mlp_up_prefill --case mlp_down_prefill --case qkv_decode --case mlp_up_decode --case mlp_down_decode --output benchmarks/results/gemm_transformer_shapes.csv
```

通过条件：

- 全量 pytest 通过。
- 每个 Transformer case 在计时前通过 PyTorch reference correctness。
- benchmark CSV 同时保存 PyTorch、自定义 selector 路径和强制 v1 对照。

实际结果：用户确认全量测试与 Transformer shape benchmark 成功；结果保存在 `benchmarks/results/gemm_transformer_shapes.csv`。

## 已知性能边界

- FP16 大方阵 Tensor Core 已接近 PyTorch native library 路径。
- 64x64 small kernel 改善中小方阵和 decode，但 small-M 仍落后 PyTorch。
- FP32 与 `transpose_b=true` 当前使用 v1 tiled，性能明显落后原生库。
- Python 侧的 `expected_dispatch_path` 是 selector 规则镜像，不是 profiler 对 kernel 名称的运行时观测。

## 按项目优先级延后的工作

以下项目属于性能优化，不再阻塞 Phase 4：

- 迁移 FP32 register-blocking/vectorized v2.5。
- 为 FP16 `transpose_b=true` 增加 Tensor Core 路径。
- 增加更小行 tile 的 decode/GEMV kernel。
- 根据性能策略将部分 shape 显式 fallback 到 cuBLAS，而不只是做 benchmark 对比。
- 使用 Nsight Compute 固化 occupancy、Tensor Pipe、Shared Memory 和寄存器证据。

这些延期项是对原 Phase 3 范围的显式裁剪，原因是当前项目先完成所有算子的工程闭环，再统一进入性能优化阶段。
