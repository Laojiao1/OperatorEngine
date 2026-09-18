"""文件职责：比较 PyTorch 与 OperatorEngine GEMM 在不同 shape 和 dispatch 路径下的性能。"""

import argparse
import csv
from pathlib import Path
from typing import Callable

import torch
import triton.testing

import my_ops


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "gemm.csv"

CASES = {
    "aligned_small": (128, 128, 32, torch.float16, False),
    "general_boundary": (129, 131, 33, torch.float16, False),
    "crossover_64": (64, 64, 64, torch.float16, False),
    "crossover_128": (128, 128, 128, torch.float16, False),
    "crossover_192": (192, 192, 192, torch.float16, False),
    "crossover_256": (256, 256, 256, torch.float16, False),
    "crossover_384": (384, 384, 384, torch.float16, False),
    "square_512": (512, 512, 512, torch.float16, False),
    "square_2048": (2048, 2048, 2048, torch.float16, False),
    "decode_m1": (1, 4096, 4096, torch.float16, False),
    "decode_m16": (16, 4096, 4096, torch.float16, False),
    # LLaMA 类 4096 hidden / 11008 intermediate 的真实投影 shape。
    "qkv_prefill": (512, 4096, 4096, torch.float16, False),
    "output_prefill": (512, 4096, 4096, torch.float16, False),
    "mlp_up_prefill": (512, 11008, 4096, torch.float16, False),
    "mlp_down_prefill": (512, 4096, 11008, torch.float16, False),
    "qkv_decode": (1, 4096, 4096, torch.float16, False),
    "mlp_up_decode": (1, 11008, 4096, torch.float16, False),
    "mlp_down_decode": (1, 4096, 11008, torch.float16, False),
    "fp16_transpose_b": (512, 512, 512, torch.float16, True),
    "fp32_square": (512, 512, 512, torch.float32, False),
}


def benchmark_provider(function: Callable[[], torch.Tensor], warmup_ms: int, repeat_ms: int) -> float:
    """使用 CUDA Event 驱动的 Triton helper 返回 GPU latency 中位数。"""
    return float(triton.testing.do_bench(function, warmup=warmup_ms, rep=repeat_ms))


def torch_gemm(a: torch.Tensor, b: torch.Tensor, transpose_b: bool) -> torch.Tensor:
    """调用与自定义算子具有相同逻辑布局和输出 dtype 的 PyTorch reference。"""
    logical_b = b.transpose(0, 1) if transpose_b else b
    if a.dtype == torch.float16:
        return torch.mm(a, logical_b, out_dtype=torch.float32)
    return torch.mm(a, logical_b)


def expected_dispatch_path(a: torch.Tensor, b: torch.Tensor, M: int, N: int, K: int, transpose_b: bool) -> str:
    """按当前 C++ selector 标注预期路径，帮助确认所有分支都被 benchmark 覆盖。"""
    tensor_core_eligible = a.dtype == torch.float16 and not transpose_b and a.data_ptr() % 16 == 0 and b.data_ptr() % 16 == 0
    if not tensor_core_eligible or M * N * K < 1024 * 1024:
        return "v1_tiled"
    if M < 1024 or N < 1024:
        if M % 64 == 0 and N % 64 == 0 and K % 32 == 0:
            return "tensor_core_small_aligned"
        return "tensor_core_small_general"
    if M % 128 == 0 and N % 128 == 0 and K % 32 == 0:
        return "tensor_core_large_aligned"
    return "tensor_core_large_general"


def make_unaligned_copy(input: torch.Tensor) -> torch.Tensor:
    """构造数值相同且 contiguous，但起始地址偏移一个元素的 Tensor。"""
    storage = torch.empty(input.numel() + 1, device=input.device, dtype=input.dtype)
    output = storage[1:].view_as(input)
    output.copy_(input)
    assert output.is_contiguous() and output.data_ptr() % 16 != 0
    return output


def tflops(M: int, N: int, K: int, latency_ms: float) -> float:
    """GEMM 按 2*M*N*K 次浮点操作计算吞吐量。"""
    return 2.0 * M * N * K / (latency_ms * 1.0e-3) / 1.0e12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", choices=CASES, help="只运行指定 case；可重复传入")
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--repeat-ms", type=int, default=300)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark 需要 NVIDIA CUDA GPU")

    selected_cases = args.case if args.case else list(CASES)
    rows: list[dict[str, object]] = []
    torch.manual_seed(0)

    for case_name in selected_cases:
        M, N, K, dtype, transpose_b = CASES[case_name]
        a = torch.randn((M, K), device="cuda", dtype=dtype)
        b_shape = (N, K) if transpose_b else (K, N)
        b = torch.randn(b_shape, device="cuda", dtype=dtype)

        # 所有实现必须先通过相同 reference，再进入计时。
        expected = torch_gemm(a, b, transpose_b)
        actual = torch.ops.my_ops.gemm(a, b, transpose_b)
        tolerance = 2e-2 if dtype == torch.float16 else 1e-3
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=2e-3 if dtype == torch.float16 else 1e-3)

        providers = {
            "torch.mm": lambda: torch_gemm(a, b, transpose_b),
            "my_ops.gemm": lambda: torch.ops.my_ops.gemm(a, b, transpose_b),
        }

        # 不增加 benchmark-only 公共算子：通过合法但未对齐的 contiguous Tensor，
        # 让真实 C++ selector 在同一轮运行中强制进入 v1 tiled fallback。
        forced_tiled_inputs = None
        if dtype == torch.float16 and not transpose_b:
            unaligned_a = make_unaligned_copy(a)
            unaligned_b = make_unaligned_copy(b)
            forced_tiled_inputs = (unaligned_a, unaligned_b)
            forced_tiled_actual = torch.ops.my_ops.gemm(unaligned_a, unaligned_b, False)
            torch.testing.assert_close(forced_tiled_actual, expected, atol=tolerance, rtol=2e-3)
            providers["my_ops.gemm_forced_tiled"] = lambda: torch.ops.my_ops.gemm(unaligned_a, unaligned_b, False)

        latencies = {provider: benchmark_provider(function, args.warmup_ms, args.repeat_ms) for provider, function in providers.items()}

        for provider, latency_ms in latencies.items():
            if provider == "torch.mm":
                expected_path = "native"
            elif provider == "my_ops.gemm_forced_tiled":
                expected_path = expected_dispatch_path(forced_tiled_inputs[0], forced_tiled_inputs[1], M, N, K, False)
            else:
                expected_path = expected_dispatch_path(a, b, M, N, K, transpose_b)

            speedup_vs_torch = latencies["torch.mm"] / latency_ms
            speedup_vs_forced_tiled = latencies.get("my_ops.gemm_forced_tiled", latency_ms) / latency_ms
            rows.append({"case": case_name, "M": M, "N": N, "K": K, "dtype": str(dtype).removeprefix("torch."), "transpose_b": transpose_b, "provider": provider, "expected_dispatch_path": expected_path, "latency_ms": latency_ms, "tflops": tflops(M, N, K, latency_ms), "speedup_vs_torch": speedup_vs_torch, "speedup_vs_forced_tiled": speedup_vs_forced_tiled})
            print(f"{case_name:<18} M={M:>4} N={N:>4} K={K:>4} {provider:<12} {expected_path:<20} {latency_ms:8.4f} ms {tflops(M, N, K, latency_ms):8.3f} TFLOPS")

        print(f"{'':18} my_ops / torch speedup: {latencies['torch.mm'] / latencies['my_ops.gemm']:.3f}x")
        if "my_ops.gemm_forced_tiled" in latencies:
            print(f"{'':18} Tensor Core / forced v1 speedup: {latencies['my_ops.gemm_forced_tiled'] / latencies['my_ops.gemm']:.3f}x")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"结果已写入：{args.output}")


if __name__ == "__main__":
    main()
