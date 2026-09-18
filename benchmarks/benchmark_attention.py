"""文件职责：比较 PyTorch SDPA、OperatorEngine C++ Attention 与 Triton FlashAttention 的正确性和性能。"""

import argparse
import csv
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
import triton.testing

import my_ops
from my_ops.attention_triton import triton_attention


DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "attention_candidates.csv"

# case: B, H, N, D, causal
CASES = {
    "short_d64": (1, 2, 17, 64, False),
    "short_d64_causal": (1, 2, 17, 64, True),
    "tensor_core_boundary": (1, 2, 33, 64, False),
    "tensor_core_causal": (1, 8, 128, 64, True),
    "d128_boundary": (1, 8, 129, 128, False),
    "prefill_512_causal": (1, 16, 512, 64, True),
    "prefill_1024_d128": (1, 16, 1024, 128, False),
    "sdpa_fallback": (1, 4, 1025, 64, True),
}


def benchmark_provider(function: Callable[[], torch.Tensor], warmup_ms: int, repeat_ms: int) -> float:
    """使用 CUDA Event 驱动的 Triton helper 返回 GPU latency 中位数。"""
    return float(triton.testing.do_bench(function, warmup=warmup_ms, rep=repeat_ms))


def torch_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    """调用与第一版算子语义一致的 PyTorch SDPA。"""
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=causal)


def expected_cpp_dispatch_path(sequence_length: int, head_dim: int) -> str:
    """按当前 C++ selector 标注预期路径；该标签不是 profiler 观测结果。"""
    if sequence_length > 1024:
        return "pytorch_sdpa_fallback"
    if head_dim == 64 and sequence_length >= 32:
        return "tensor_core"
    return "cuda_core"


def dense_tflops(batch_size: int, num_heads: int, sequence_length: int, head_dim: int, latency_ms: float) -> float:
    """按完整 QK^T 与 PV 的 4*B*H*N*N*D 次运算计算稠密等效吞吐量。"""
    operations = 4.0 * batch_size * num_heads * sequence_length * sequence_length * head_dim
    return operations / (latency_ms * 1.0e-3) / 1.0e12


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
        batch_size, num_heads, sequence_length, head_dim, causal = CASES[case_name]
        shape = (batch_size, num_heads, sequence_length, head_dim)
        q = torch.randn(shape, device="cuda", dtype=torch.float16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        expected = torch_attention(q, k, v, causal)
        cpp_actual = torch.ops.my_ops.attention(q, k, v, causal)
        triton_actual = triton_attention(q, k, v, causal)
        auto_actual = my_ops.attention(q, k, v, causal=causal, provider="auto")
        torch.testing.assert_close(cpp_actual, expected, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(triton_actual, expected, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(auto_actual, expected, atol=2e-2, rtol=2e-2)

        providers = {
            "torch.sdpa": lambda: torch_attention(q, k, v, causal),
            "my_ops.attention": lambda: torch.ops.my_ops.attention(q, k, v, causal),
            "triton_attention": lambda: triton_attention(q, k, v, causal),
            "my_ops.attention_auto": lambda: my_ops.attention(q, k, v, causal=causal, provider="auto"),
        }
        latencies = {provider: benchmark_provider(function, args.warmup_ms, args.repeat_ms) for provider, function in providers.items()}

        for provider, latency_ms in latencies.items():
            if provider == "torch.sdpa":
                expected_path = "native"
            elif provider == "my_ops.attention":
                expected_path = expected_cpp_dispatch_path(sequence_length, head_dim)
            elif provider == "my_ops.attention_auto":
                expected_path = "auto_triton" if causal or sequence_length <= 512 else "auto_sdpa"
            else:
                expected_path = "triton_flash_attention"

            throughput = dense_tflops(batch_size, num_heads, sequence_length, head_dim, latency_ms)
            speedup_vs_sdpa = latencies["torch.sdpa"] / latency_ms
            rows.append({"case": case_name, "B": batch_size, "H": num_heads, "N": sequence_length, "D": head_dim, "dtype": "float16", "causal": causal, "provider": provider, "expected_dispatch_path": expected_path, "latency_ms": latency_ms, "dense_tflops": throughput, "speedup_vs_sdpa": speedup_vs_sdpa})
            print(f"{case_name:<22} B={batch_size:>2} H={num_heads:>2} N={sequence_length:>4} D={head_dim:>3} causal={str(causal):<5} {provider:<18} {expected_path:<22} {latency_ms:8.4f} ms {throughput:8.3f} TFLOPS")

        print(f"{'':22} C++ / SDPA speedup: {latencies['torch.sdpa'] / latencies['my_ops.attention']:.3f}x")
        print(f"{'':22} Triton / SDPA speedup: {latencies['torch.sdpa'] / latencies['triton_attention']:.3f}x")
        print(f"{'':22} Auto / SDPA speedup: {latencies['torch.sdpa'] / latencies['my_ops.attention_auto']:.3f}x")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"结果已写入：{args.output}")


if __name__ == "__main__":
    main()
