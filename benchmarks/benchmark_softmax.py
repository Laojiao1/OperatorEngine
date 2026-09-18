"""文件职责：比较 PyTorch、OperatorEngine CUDA 与阶段三 Triton Softmax 的性能。"""

import argparse
import csv
from pathlib import Path
from typing import Callable

import torch
import triton.testing

import my_ops
from triton_softmax import softmax as triton_softmax


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "softmax.csv"


def effective_bandwidth_gb_s(input: torch.Tensor, latency_ms: float) -> float:
    """按一次输入读取和一次输出写回计算统一的逻辑有效带宽。"""
    logical_bytes = 2 * input.numel() * input.element_size()
    return logical_bytes / (latency_ms * 1.0e-3) / 1.0e9


def benchmark_provider(function: Callable[[], torch.Tensor], warmup_ms: int, repeat_ms: int) -> float:
    """使用 CUDA Event 驱动的 Triton helper 返回 GPU latency 中位数。"""
    return float(triton.testing.do_bench(function, warmup=warmup_ms, rep=repeat_ms))


def expected_dispatch_path(input: torch.Tensor) -> str:
    """按当前 C++ selector 规则标注预期路径，便于检查 benchmark 覆盖。"""
    columns = input.shape[-1]
    if input.dtype == torch.float32 and columns >= 1024 and columns % 4 == 0 and input.data_ptr() % 16 == 0:
        return "v4_online"
    return "v1_block"


def make_unaligned_input(rows: int, columns: int) -> torch.Tensor:
    """构造逻辑连续但起始地址偏移 4 Byte 的 FP32 Tensor，强制验证 v1 fallback。"""
    storage = torch.randn(rows * columns + 1, device="cuda", dtype=torch.float32)
    input = storage[1:].view(rows, columns)
    assert input.is_contiguous() and input.data_ptr() % 16 != 0
    return input


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--repeat-ms", type=int, default=300)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark 需要 NVIDIA CUDA GPU")

    columns_list = [128, 512, 1024, 2048, 4096, 8192]
    dtypes = [torch.float32, torch.float16]
    target_numel = 8 * 1024 * 1024
    rows_data: list[dict[str, object]] = []

    torch.manual_seed(0)
    for dtype in dtypes:
        for columns in columns_list:
            rows = target_numel // columns
            input = torch.randn(rows, columns, device="cuda", dtype=dtype)
            expected = torch.softmax(input, dim=-1)

            # 所有被计时实现必须先通过相同 PyTorch Reference。
            torch.testing.assert_close(torch.ops.my_ops.softmax(input, -1), expected, atol=2e-3 if dtype == torch.float16 else 1e-5, rtol=2e-3 if dtype == torch.float16 else 1e-5)
            torch.testing.assert_close(triton_softmax(input), expected, atol=2e-3 if dtype == torch.float16 else 1e-5, rtol=2e-3 if dtype == torch.float16 else 1e-5)

            providers = {
                "torch.softmax": lambda: torch.softmax(input, dim=-1),
                "my_ops.softmax": lambda: torch.ops.my_ops.softmax(input, -1),
                "triton_phase3_adapted": lambda: triton_softmax(input),
            }

            for provider, function in providers.items():
                latency_ms = benchmark_provider(function, args.warmup_ms, args.repeat_ms)
                path = expected_dispatch_path(input) if provider == "my_ops.softmax" else "native"
                bandwidth = effective_bandwidth_gb_s(input, latency_ms)
                rows_data.append({"rows": rows, "columns": columns, "dtype": str(dtype).removeprefix("torch."), "provider": provider, "dispatch_path": path, "latency_ms": latency_ms, "effective_gb_s": bandwidth})
                print(f"dtype={str(dtype).removeprefix('torch.'):<7} M={rows:>6} N={columns:>5} {provider:<23} {path:<10} {latency_ms:8.4f} ms {bandwidth:8.2f} GB/s")

            # 同 shape 下用未对齐地址强制走 v1，估计 v4 selector 相对 v1 fallback 的收益。
            if dtype == torch.float32 and columns >= 1024 and columns % 4 == 0:
                unaligned_input = make_unaligned_input(rows, columns)
                unaligned_expected = torch.softmax(unaligned_input, dim=-1)
                torch.testing.assert_close(torch.ops.my_ops.softmax(unaligned_input, -1), unaligned_expected, atol=1e-5, rtol=1e-5)
                latency_ms = benchmark_provider(lambda: torch.ops.my_ops.softmax(unaligned_input, -1), args.warmup_ms, args.repeat_ms)
                bandwidth = effective_bandwidth_gb_s(unaligned_input, latency_ms)
                rows_data.append({"rows": rows, "columns": columns, "dtype": "float32", "provider": "my_ops.softmax_unaligned", "dispatch_path": "v1_block", "latency_ms": latency_ms, "effective_gb_s": bandwidth})
                print(f"dtype=float32 M={rows:>6} N={columns:>5} {'my_ops.softmax_unaligned':<23} {'v1_block':<10} {latency_ms:8.4f} ms {bandwidth:8.2f} GB/s")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=rows_data[0].keys())
        writer.writeheader()
        writer.writerows(rows_data)

    print(f"结果已写入：{args.output}")


if __name__ == "__main__":
    main()
