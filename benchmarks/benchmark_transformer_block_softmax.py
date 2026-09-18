"""文件职责：比较 fused SDPA、Torch 未融合 Attention 与替换 Softmax 后的 Block 延迟。"""

import argparse
import csv
import importlib.util
from pathlib import Path
from typing import Callable

import torch
import triton.testing

import my_ops


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "transformer_block_softmax.csv"

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
    spec = importlib.util.spec_from_file_location("operator_engine_transformer_block_softmax_benchmark", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 Transformer Block 示例：{module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def benchmark_provider(function: Callable[[], torch.Tensor], warmup_ms: int, repeat_ms: int) -> float:
    """使用 CUDA Event 驱动的 Triton helper 返回端到端 GPU latency 中位数。"""
    return float(triton.testing.do_bench(function, warmup=warmup_ms, rep=repeat_ms))


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
    TorchUnfusedAttentionBlock = transformer_module.TorchUnfusedAttentionBlock
    OperatorEngineSoftmaxBlock = transformer_module.OperatorEngineSoftmaxBlock

    selected_cases = args.case if args.case else list(CASES)
    rows: list[dict[str, object]] = []

    for case_index, case_name in enumerate(selected_cases):
        batch_size, sequence_length, hidden_size, num_heads, intermediate_size = CASES[case_name]
        head_dim = hidden_size // num_heads
        torch.manual_seed(case_index)

        baseline = TorchTransformerBlock(hidden_size, num_heads, intermediate_size).cuda().half().eval()
        torch_unfused = TorchUnfusedAttentionBlock(hidden_size, num_heads, intermediate_size).cuda().half().eval()
        custom_softmax = OperatorEngineSoftmaxBlock(hidden_size, num_heads, intermediate_size).cuda().half().eval()

        for block in (torch_unfused, custom_softmax):
            block.load_state_dict(baseline.state_dict())

        hidden_states = torch.randn((batch_size, sequence_length, hidden_size), device="cuda", dtype=torch.float16)

        with torch.inference_mode():
            expected = baseline(hidden_states)
            torch_unfused_actual = torch_unfused(hidden_states)
            custom_softmax_actual = custom_softmax(hidden_states)
            torch.testing.assert_close(torch_unfused_actual, expected, atol=3e-2, rtol=3e-2)
            torch.testing.assert_close(custom_softmax_actual, torch_unfused_actual, atol=3e-2, rtol=3e-2)

            providers = {
                "torch_fused_sdpa": lambda: baseline(hidden_states),
                "torch_unfused": lambda: torch_unfused(hidden_states),
                "custom_softmax_unfused": lambda: custom_softmax(hidden_states),
            }
            latencies = {provider: benchmark_provider(function, args.warmup_ms, args.repeat_ms) for provider, function in providers.items()}

        for provider, latency_ms in latencies.items():
            if provider == "torch_fused_sdpa":
                attention_form = "fused"
                expected_softmax_path = "inside_sdpa"
            elif provider == "torch_unfused":
                attention_form = "unfused"
                expected_softmax_path = "torch.softmax"
            else:
                attention_form = "unfused"
                expected_softmax_path = "v1_block_fp16"

            speedup_vs_torch = latencies["torch_fused_sdpa"] / latency_ms
            speedup_vs_torch_unfused = latencies["torch_unfused"] / latency_ms
            rows.append({"case": case_name, "B": batch_size, "N": sequence_length, "hidden_size": hidden_size, "num_heads": num_heads, "head_dim": head_dim, "intermediate_size": intermediate_size, "provider": provider, "attention_form": attention_form, "expected_softmax_path": expected_softmax_path, "latency_ms": latency_ms, "speedup_vs_torch": speedup_vs_torch, "speedup_vs_torch_unfused": speedup_vs_torch_unfused})
            print(f"{case_name:<22} B={batch_size:>2} N={sequence_length:>4} hidden={hidden_size:>4} H={num_heads:>2} D={head_dim:>3} {provider:<24} {attention_form:<7} {expected_softmax_path:<15} {latency_ms:8.4f} ms torch={speedup_vs_torch:7.3f}x unfused={speedup_vs_torch_unfused:7.3f}x")

        print(f"{'':22} Fused SDPA / Torch unfused: {latencies['torch_unfused'] / latencies['torch_fused_sdpa']:.3f}x")
        print(f"{'':22} Custom / Torch Softmax:     {latencies['torch_unfused'] / latencies['custom_softmax_unfused']:.3f}x")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"结果已写入：{args.output}")


if __name__ == "__main__":
    main()
