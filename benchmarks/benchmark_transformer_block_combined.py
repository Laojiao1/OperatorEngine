"""文件职责：在相同输入和权重下比较 Transformer Block 的逐项算子替换效果。"""

import argparse
import csv
import importlib.util
from pathlib import Path
from typing import Callable

import torch
import triton.testing

import my_ops


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "transformer_block_combined.csv"

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
    spec = importlib.util.spec_from_file_location("operator_engine_transformer_block_combined_benchmark", module_path)
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
    OperatorEngineAttentionBlock = transformer_module.OperatorEngineAttentionBlock
    OperatorEngineGemmBlock = transformer_module.OperatorEngineGemmBlock
    OperatorEngineTransformerBlock = transformer_module.OperatorEngineTransformerBlock

    selected_cases = args.case if args.case else list(CASES)
    rows: list[dict[str, object]] = []

    for case_index, case_name in enumerate(selected_cases):
        batch_size, sequence_length, hidden_size, num_heads, intermediate_size = CASES[case_name]
        head_dim = hidden_size // num_heads
        torch.manual_seed(case_index)

        baseline = TorchTransformerBlock(hidden_size, num_heads, intermediate_size).cuda().half().eval()
        attention_block = OperatorEngineAttentionBlock(hidden_size, num_heads, intermediate_size, attention_provider="auto").cuda().half().eval()
        gemm_block = OperatorEngineGemmBlock(hidden_size, num_heads, intermediate_size).cuda().half().eval()
        combined_block = OperatorEngineTransformerBlock(hidden_size, num_heads, intermediate_size, attention_provider="auto").cuda().half().eval()

        for block in (attention_block, gemm_block, combined_block):
            block.load_state_dict(baseline.state_dict())

        hidden_states = torch.randn((batch_size, sequence_length, hidden_size), device="cuda", dtype=torch.float16)

        with torch.inference_mode():
            expected = baseline(hidden_states)
            attention_actual = attention_block(hidden_states)
            gemm_actual = gemm_block(hidden_states)
            combined_actual = combined_block(hidden_states)
            torch.testing.assert_close(attention_actual, expected, atol=3e-2, rtol=3e-2)
            torch.testing.assert_close(gemm_actual, expected, atol=3e-2, rtol=3e-2)
            torch.testing.assert_close(combined_actual, expected, atol=3e-2, rtol=3e-2)

            providers = {
                "torch_baseline": lambda: baseline(hidden_states),
                "attention_only": lambda: attention_block(hidden_states),
                "gemm_only": lambda: gemm_block(hidden_states),
                "attention_and_gemm": lambda: combined_block(hidden_states),
            }
            latencies = {provider: benchmark_provider(function, args.warmup_ms, args.repeat_ms) for provider, function in providers.items()}

        for provider, latency_ms in latencies.items():
            uses_custom_attention = provider in ("attention_only", "attention_and_gemm")
            uses_custom_gemm = provider in ("gemm_only", "attention_and_gemm")
            expected_attention_path = "triton" if uses_custom_attention else "pytorch_sdpa"
            expected_gemm_path = "v1_tiled_transpose_b" if uses_custom_gemm else "pytorch_linear"
            speedup_vs_torch = latencies["torch_baseline"] / latency_ms
            rows.append({"case": case_name, "B": batch_size, "N": sequence_length, "hidden_size": hidden_size, "num_heads": num_heads, "head_dim": head_dim, "intermediate_size": intermediate_size, "provider": provider, "expected_attention_path": expected_attention_path, "expected_gemm_path": expected_gemm_path, "latency_ms": latency_ms, "speedup_vs_torch": speedup_vs_torch})
            print(f"{case_name:<22} B={batch_size:>2} N={sequence_length:>4} hidden={hidden_size:>4} H={num_heads:>2} D={head_dim:>3} {provider:<19} attn={expected_attention_path:<12} gemm={expected_gemm_path:<21} {latency_ms:8.4f} ms {speedup_vs_torch:7.3f}x")

        print(f"{'':22} Attention-only speedup: {latencies['torch_baseline'] / latencies['attention_only']:.3f}x")
        print(f"{'':22} GEMM-only speedup:      {latencies['torch_baseline'] / latencies['gemm_only']:.3f}x")
        print(f"{'':22} Combined speedup:       {latencies['torch_baseline'] / latencies['attention_and_gemm']:.3f}x")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"结果已写入：{args.output}")


if __name__ == "__main__":
    main()
