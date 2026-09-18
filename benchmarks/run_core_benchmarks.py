"""文件职责：用一个命令顺序运行 OperatorEngine 的五组核心 benchmark。"""

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = BENCHMARK_DIR / "results" / "core"

# 每个条目只负责编排现有 benchmark，不改变其 case、正确性检查或计时方法。
CORE_BENCHMARKS = {
    "vector_add": ("benchmark_vector_add.py", "vector_add.csv"),
    "softmax": ("benchmark_softmax.py", "softmax.csv"),
    "gemm": ("benchmark_gemm.py", "gemm.csv"),
    "attention": ("benchmark_attention.py", "attention.csv"),
    "transformer_block": ("benchmark_transformer_block_combined.py", "transformer_block.csv"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="append", choices=CORE_BENCHMARKS, help="只运行指定 benchmark；可重复传入")
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--repeat-ms", type=int, default=300)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_benchmarks = args.benchmark if args.benchmark else list(CORE_BENCHMARKS)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for index, benchmark_name in enumerate(selected_benchmarks, start=1):
        script_name, output_name = CORE_BENCHMARKS[benchmark_name]
        command = [
            sys.executable,
            str(BENCHMARK_DIR / script_name),
            "--warmup-ms",
            str(args.warmup_ms),
            "--repeat-ms",
            str(args.repeat_ms),
            "--output",
            str(output_dir / output_name),
        ]

        print(f"\n[{index}/{len(selected_benchmarks)}] 运行 {benchmark_name}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)

    print(f"\n全部核心 benchmark 已完成，结果目录：{output_dir}")


if __name__ == "__main__":
    main()
