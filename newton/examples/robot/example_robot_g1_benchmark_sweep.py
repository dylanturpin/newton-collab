#!/usr/bin/env python
import argparse
import csv
import datetime as dt
import re
import subprocess
import sys
from pathlib import Path

# Path to your benchmark script, relative to this file or the repo root.
BENCH_SCRIPT = "newton/examples/robot/example_robot_g1_benchmark.py"

# Regexes to scrape the summary block
RE_ELAPSED = re.compile(r"Elapsed time \(s\):\s*([0-9.]+)")
RE_ENVFPS = re.compile(r"Env-FPS \(env/s\):\s*([0-9,\.]+)")
RE_GPU_USED = re.compile(r"GPU memory used \(GB\):\s*([0-9.]+)")
RE_GPU_TOTAL = re.compile(r"GPU memory total \(GB\):\s*([0-9.]+)")
RE_WORLDS = re.compile(r"Worlds:\s*([0-9]+)")
RE_SOLVER = re.compile(r"Solver:\s*([A-Za-z0-9_]+)")


def parse_summary(text: str) -> dict:
    """Extract key numbers from the printed benchmark summary."""
    def get_float(regex, default=None):
        m = regex.search(text)
        if not m:
            return default
        # Strip commas for things like "541,284.57"
        return float(m.group(1).replace(",", ""))

    def get_int(regex, default=None):
        m = regex.search(text)
        if not m:
            return default
        return int(m.group(1))

    return {
        "solver": RE_SOLVER.search(text).group(1) if RE_SOLVER.search(text) else None,
        "num_worlds": get_int(RE_WORLDS),
        "elapsed_s": get_float(RE_ELAPSED),
        "env_fps": get_float(RE_ENVFPS),
        "gpu_used_gb": get_float(RE_GPU_USED),
        "gpu_total_gb": get_float(RE_GPU_TOTAL),
    }


def run_one(solver: str, num_worlds: int, args) -> dict:
    """Run a single benchmark and return parsed metrics."""
    cmd = [
        "uv",
        "run",
        BENCH_SCRIPT,
        "--solver", solver,
        "--benchmark",
        "--viewer", "null",
        "--num-worlds", str(num_worlds),
        "--sim-substeps", str(args.sim_substeps),
        "--warmup-frames", str(args.warmup_frames),
        "--measure-frames", str(args.measure_frames),
    ]

    print(f"\n=== Running {solver} with num-worlds={num_worlds} ===")
    print(" ".join(cmd))

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )

    # Stream stderr so you see any issues
    if proc.stderr:
        sys.stderr.write(proc.stderr)

    if proc.returncode != 0:
        print(f"Run failed (return code {proc.returncode}), skipping.")
        return {
            "solver": solver,
            "num_worlds": num_worlds,
            "elapsed_s": None,
            "env_fps": None,
            "gpu_used_gb": None,
            "gpu_total_gb": None,
            "ok": False,
        }

    metrics = parse_summary(proc.stdout)
    metrics["ok"] = True
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Sweep G1 benchmark over num-worlds and solvers.")
    parser.add_argument(
        "--solvers",
        type=str,
        default="feather_pgs,mujoco",
        help="Comma-separated list of solvers to run (default: feather_pgs,mujoco).",
    )
    parser.add_argument(
        "--min-log2-worlds",
        type=int,
        default=10,
        help="Minimum log2(num_worlds). 10 => 1024.",
    )
    parser.add_argument(
        "--max-log2-worlds",
        type=int,
        default=16,
        help="Maximum log2(num_worlds). 16 => 65536.",
    )
    parser.add_argument(
        "--sim-substeps",
        type=int,
        default=2,
        help="Simulation substeps per frame (passed through).",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=16,
        help="Warmup frames (passed through).",
    )
    parser.add_argument(
        "--measure-frames",
        type=int,
        default=64,
        help="Measured frames (passed through).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV file. Default: g1_bench_{timestamp}.csv",
    )

    args = parser.parse_args()

    solvers = [s.strip() for s in args.solvers.split(",") if s.strip()]
    log2_values = range(args.min_log2_worlds, args.max_log2_worlds + 1)
    num_worlds_list = [2 ** k for k in log2_values]

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out or f"g1_bench_{timestamp}.csv")

    rows = []
    for solver in solvers:
        for n in num_worlds_list:
            metrics = run_one(solver, n, args)
            row = {
                "solver": solver,
                "num_worlds": n,
                "elapsed_s": metrics.get("elapsed_s"),
                "env_fps": metrics.get("env_fps"),
                "gpu_used_gb": metrics.get("gpu_used_gb"),
                "gpu_total_gb": metrics.get("gpu_total_gb"),
                "ok": metrics.get("ok", False),
                "sim_substeps": args.sim_substeps,
                "warmup_frames": args.warmup_frames,
                "measure_frames": args.measure_frames,
            }
            rows.append(row)

    # Write CSV
    fieldnames = [
        "solver",
        "num_worlds",
        "elapsed_s",
        "env_fps",
        "gpu_used_gb",
        "gpu_total_gb",
        "ok",
        "sim_substeps",
        "warmup_frames",
        "measure_frames",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rows to {out_path}")
    print("Quick preview:")
    for r in rows:
        print(
            f"{r['solver']:12s}  worlds={r['num_worlds']:6d}  "
            f"env_fps={r['env_fps']!s:>10}  gpu_used={r['gpu_used_gb']!s:>6}"
        )


if __name__ == "__main__":
    main()
