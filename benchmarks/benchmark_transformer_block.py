"""文件职责：比较纯 PyTorch Block 与只替换 Attention 后的端到端延迟。"""

import argparse
import csv
import importlib.util
from pathlib import Path
from typing import Callable

import torch
import triton.testing

import my_ops


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "transformer_block_attention.csv"

# case: B, N, hidden_size, num_heads, intermediate_size
CASES = {
    "tiny_n17_d64": (1, 17, 256, 4, 512),
    "prefill_n128_d64": (1, 128, 512, 8, 1024),
    "prefill_n512_d64": (1, 512, 512, 8, 1024),
    "prefill_n128_d128": (1, 128, 512, 4, 1024),
    "prefill_n1024_d128": (1, 1024, 512, 4, 1024),
}


def load_transformer_block_module():
    """按文件路径加载示例，避免把 examples 目录安装为 Python 包。"""
    module_path = ROOT / "examples" / "transformer_block.py"
    spec = importlib.util.spec_from_file_location("operator_engine_transformer_block_benchmark", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 Transformer Block 示例：{module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def benchmark_provider(function: Callable[[], torch.Tensor], warmup_ms: int, repeat_ms: int) -> float:
    """使用 CUDA Event 驱动的 Triton helper 返回端到端 GPU latency 中位数。"""
    return float(triton.testing.do_bench(function, warmup=warmup_ms, rep=repeat_ms))


def expected_cpp_attention_path(sequence_length: int, head_dim: int) -> str:
    """按当前低层 C++ selector 返回预期路径，不作为 profiler 观测证据。"""
    if sequence_length > 1024:
        return "cpp_sdpa_fallback"
    if head_dim == 64 and sequence_length >= 32:
        return "cpp_tensor_core"
    return "cpp_cuda_core"


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

    transformer_module = load_transformer_block_module()
    TorchTransformerBlock = transformer_module.TorchTransformerBlock
    OperatorEngineAttentionBlock = transformer_module.OperatorEngineAttentionBlock

    selected_cases = args.case if args.case else list(CASES)
    rows: list[dict[str, object]] = []

    for case_index, case_name in enumerate(selected_cases):
        batch_size, sequence_length, hidden_size, num_heads, intermediate_size = CASES[case_name]
        head_dim = hidden_size // num_heads
        torch.manual_seed(case_index)

        baseline = TorchTransformerBlock(hidden_size, num_heads, intermediate_size).cuda().half().eval()
        auto_block = OperatorEngineAttentionBlock(hidden_size, num_heads, intermediate_size, attention_provider="auto").cuda().half().eval()
        triton_block = OperatorEngineAttentionBlock(hidden_size, num_heads, intermediate_size, attention_provider="triton").cuda().half().eval()
        cpp_block = OperatorEngineAttentionBlock(hidden_size, num_heads, intermediate_size, attention_provider="cpp").cuda().half().eval()

        for block in (auto_block, triton_block, cpp_block):
            block.load_state_dict(baseline.state_dict())

        hidden_states = torch.randn((batch_size, sequence_length, hidden_size), device="cuda", dtype=torch.float16)

        with torch.inference_mode():
            expected = baseline(hidden_states)
            auto_actual = auto_block(hidden_states)
            triton_actual = triton_block(hidden_states)
            cpp_actual = cpp_block(hidden_states)
            torch.testing.assert_close(auto_actual, expected, atol=3e-2, rtol=3e-2)
            torch.testing.assert_close(triton_actual, expected, atol=3e-2, rtol=3e-2)
            torch.testing.assert_close(cpp_actual, expected, atol=3e-2, rtol=3e-2)

            providers = {
                "torch_baseline": lambda: baseline(hidden_states),
                "attention_auto": lambda: auto_block(hidden_states),
                "attention_triton": lambda: triton_block(hidden_states),
                "attention_cpp": lambda: cpp_block(hidden_states),
            }
            latencies = {provider: benchmark_provider(function, args.warmup_ms, args.repeat_ms) for provider, function in providers.items()}

        for provider, latency_ms in latencies.items():
            if provider == "torch_baseline":
                expected_path = "pytorch_sdpa"
            elif provider == "attention_cpp":
                expected_path = expected_cpp_attention_path(sequence_length, head_dim)
            else:
                expected_path = "triton"

            speedup_vs_torch = latencies["torch_baseline"] / latency_ms
            rows.append({"case": case_name, "B": batch_size, "N": sequence_length, "hidden_size": hidden_size, "num_heads": num_heads, "head_dim": head_dim, "intermediate_size": intermediate_size, "provider": provider, "expected_attention_path": expected_path, "latency_ms": latency_ms, "speedup_vs_torch": speedup_vs_torch})
            print(f"{case_name:<22} B={batch_size:>2} N={sequence_length:>4} hidden={hidden_size:>4} H={num_heads:>2} D={head_dim:>3} {provider:<18} {expected_path:<20} {latency_ms:8.4f} ms {speedup_vs_torch:7.3f}x")

        print(f"{'':22} Auto block speedup: {latencies['torch_baseline'] / latencies['attention_auto']:.3f}x")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"结果已写入：{args.output}")


if __name__ == "__main__":
    main()
