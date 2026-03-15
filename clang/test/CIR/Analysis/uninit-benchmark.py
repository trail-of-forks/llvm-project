#!/usr/bin/env python3
"""Benchmark CIR vs CFG uninitialized variable analysis.

Usage:
    python3 uninit-benchmark.py <clang-binary> [num-functions]

Generates a synthetic C file with many functions exhibiting different
initialization patterns, then times both analysis paths.
"""

import subprocess
import sys
import tempfile
import time


def generate_bench(n_funcs: int) -> str:
    lines = [
        "// Auto-generated benchmark for uninit variable analysis timing",
        f"// Functions: {n_funcs}",
        "",
    ]
    for i in range(n_funcs):
        kind = i % 5
        lines.append(f"int bench_{i}(int cond, int a, int b) {{")
        if kind == 0:
            lines.append("  int x;")
            lines.append("  if (cond) x = 1; else x = 2;")
            lines.append("  return x;")
        elif kind == 1:
            lines.append("  int x;")
            lines.append("  if (cond) x = a + b;")
            lines.append("  return x;")
        elif kind == 2:
            lines.append("  int x;")
            lines.append("  { { x = a * b; } }")
            lines.append("  return x;")
        elif kind == 3:
            lines.append("  int x, y, z;")
            lines.append("  x = a; y = b;")
            lines.append("  if (cond) z = x + y; else z = 0;")
            lines.append("  return z;")
        elif kind == 4:
            lines.append("  int x;")
            lines.append("  if (a > 0) {")
            lines.append("    if (b > 0) x = 1;")
            lines.append("    else x = 2;")
            lines.append("  } else {")
            lines.append("    x = 3;")
            lines.append("  }")
            lines.append("  return x;")
        lines.append("}")
        lines.append("")
    return "\n".join(lines)


def time_cmd(args: list[str], warmup: bool = True) -> float:
    if warmup:
        subprocess.run(args, capture_output=True)
    times = []
    for _ in range(5):
        start = time.perf_counter()
        subprocess.run(args, capture_output=True)
        times.append(time.perf_counter() - start)
    return sum(sorted(times)[1:4]) / 3  # median of middle 3


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <clang-binary> [num-functions]")
        sys.exit(1)

    clang = sys.argv[1]
    n_funcs = int(sys.argv[2]) if len(sys.argv) > 2 else 2000

    bench_src = generate_bench(n_funcs)
    with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as f:
        f.write(bench_src)
        bench_file = f.name

    n_lines = bench_src.count("\n")
    print(f"Benchmark: {n_funcs} functions, {n_lines} lines")
    print()

    base_args = [clang, "-cc1", "-triple", "x86_64-unknown-linux-gnu"]
    warn_args = ["-Wuninitialized", "-Wconditional-uninitialized"]

    t_baseline = time_cmd(
        base_args + ["-fsyntax-only", bench_file]
    )
    t_cfg = time_cmd(
        base_args + warn_args + ["-fsyntax-only", bench_file]
    )
    t_cir_no_analysis = time_cmd(
        base_args + ["-fclangir", "-emit-llvm", bench_file, "-o", "/dev/null"]
    )
    t_cir = time_cmd(
        base_args
        + ["-fclangir", "-fclangir-analysis=uninit"]
        + warn_args
        + ["-emit-llvm", bench_file, "-o", "/dev/null"]
    )

    cfg_analysis = t_cfg - t_baseline
    cir_analysis = t_cir - t_cir_no_analysis

    print(f"{'Component':<45} {'Time (ms)':>10}")
    print("-" * 57)
    print(f"{'Baseline (parse + sema)':<45} {t_baseline*1000:>10.1f}")
    print(f"{'CFG path (parse + sema + CFG uninit)':<45} {t_cfg*1000:>10.1f}")
    print(f"{'CIR pipeline (no analysis)':<45} {t_cir_no_analysis*1000:>10.1f}")
    print(f"{'CIR path (pipeline + CIR uninit)':<45} {t_cir*1000:>10.1f}")
    print()
    print(f"{'Analysis-only overhead:':<45}")
    print(f"{'  CFG uninit analysis':<45} {cfg_analysis*1000:>10.1f}")
    print(f"{'  CIR uninit analysis':<45} {cir_analysis*1000:>10.1f}")
    if cfg_analysis > 0:
        ratio = cir_analysis / cfg_analysis
        print(f"{'  Ratio (CIR / CFG)':<45} {ratio:>10.2f}x")


if __name__ == "__main__":
    main()
