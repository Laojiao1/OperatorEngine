"""文件职责：比较 PyTorch、OperatorEngine CUDA 和 Triton Vector Add 性能。"""

import argparse
import csv
from pathlib import Path
from typing import Callable

import torch
import triton.testing

import my_ops
from triton_vector_add import add as triton_add


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "vector_add.csv"


def effective_bandwidth_gb_s(numel: int, latency_ms: float) -> float:
    """按两次 FP32 读取和一次 FP32 写入计算逻辑有效带宽。"""
    logical_bytes = 3 * numel * torch.float32.itemsize
    return logical_bytes / (latency_ms * 1.0e-3) / 1.0e9


def benchmark_provider(
    function: Callable[[], torch.Tensor], warmup_ms: int, repeat_ms: int
) -> float:
    """用 CUDA Event 驱动的 Triton helper 测量 GPU latency 中位数。"""
    return float(triton.testing.do_bench(function, warmup=warmup_ms, rep=repeat_ms))


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

    shapes = {
        "small": 1 << 12,
        "medium": 1 << 20,
        "large": 1 << 24,
    }
    rows: list[dict[str, object]] = []

    torch.manual_seed(0)
    for size_class, numel in shapes.items():
        a = torch.randn(numel, device="cuda", dtype=torch.float32)
        b = torch.randn_like(a)

        # benchmark 前先做一次独立 correctness，避免给错误 kernel 计出漂亮数据。
        torch.testing.assert_close(torch.ops.my_ops.vector_add(a, b), a + b)
        torch.testing.assert_close(triton_add(a, b), a + b)

        providers = {
            "torch.add": lambda: torch.add(a, b),
            "my_ops_cuda_scalar": lambda: torch.ops.my_ops.vector_add(a, b),
            "triton_phase3_adapted": lambda: triton_add(
                a, b, block_size=1024
            ),
        }

        for provider, function in providers.items():
            latency_ms = benchmark_provider(
                function, args.warmup_ms, args.repeat_ms
            )
            bandwidth = effective_bandwidth_gb_s(numel, latency_ms)
            rows.append(
                {
                    "size_class": size_class,
                    "numel": numel,
                    "dtype": "float32",
                    "provider": provider,
                    "latency_ms": latency_ms,
                    "effective_gb_s": bandwidth,
                }
            )
            print(
                f"{size_class:>6}  N={numel:>9}  {provider:<20} "
                f"{latency_ms:8.4f} ms  {bandwidth:8.2f} GB/s"
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"结果已写入：{args.output}")


if __name__ == "__main__":
    main()
