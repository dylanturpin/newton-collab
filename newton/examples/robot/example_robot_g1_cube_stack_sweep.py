# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Sweep benchmark for G1 cube stack example.

Sweeps over:
  - Solvers (feather_pgs, mujoco)
  - Tiled modes (tiled, non_tiled) - only for feather_pgs
  - Number of worlds (powers of 2)

Results are written incrementally to CSV to avoid losing progress.

Usage examples:

Sweep tiled vs non-tiled for feather_pgs:
    uv run newton/examples/robot/example_robot_g1_cube_stack_sweep.py \
        --solvers feather_pgs --tiled-modes tiled,non_tiled \
        --min-log2-worlds 10 --max-log2-worlds 14

Sweep all solvers:
    uv run newton/examples/robot/example_robot_g1_cube_stack_sweep.py \
        --solvers feather_pgs,mujoco --tiled-modes tiled,non_tiled \
        --min-log2-worlds 10 --max-log2-worlds 14
"""

import argparse
import csv
import datetime as dt
import re
import subprocess
import sys
from pathlib import Path

# Path to the benchmark script, relative to the repo root.
BENCH_SCRIPT = "newton/examples/robot/example_robot_g1_cube_stack.py"

# Regexes to scrape the summary block (same as g1 benchmark sweep)
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


def run_one(solver: str, use_tiled: bool | None, num_worlds: int, args) -> dict:
    """
    Run a single benchmark and return parsed metrics.

    Args:
        solver: "feather_pgs" or "mujoco"
        use_tiled: True/False for feather_pgs, None for mujoco
        num_worlds: Number of parallel worlds
        args: CLI arguments
    """
    cmd = [
        "uv",
        "run",
        BENCH_SCRIPT,
        "--solver",
        solver,
        "--benchmark",
        "--viewer",
        "null",
        "--num-worlds",
        str(num_worlds),
        "--sim-substeps",
        str(args.sim_substeps),
        "--warmup-frames",
        str(args.warmup_frames),
        "--measure-frames",
        str(args.measure_frames),
    ]

    # FeatherPGS-specific options
    if solver == "feather_pgs":
        cmd.extend(
            [
                "--pgs-iterations",
                str(args.pgs_iterations),
                "--dense-max-constraints",
                str(args.dense_max_constraints),
                "--update-mass-matrix-interval",
                str(1),
                "--pgs-warmstart",
                "--pgs-omega",
                str(1.4),
            ]
        )
        # Tiled flag
        if use_tiled:
            cmd.append("--use-tiled")
        else:
            cmd.append("--no-tiled")

    # Build solver label for CSV
    if solver == "feather_pgs":
        solver_label = f"feather_pgs_{'tiled' if use_tiled else 'non_tiled'}"
    else:
        solver_label = solver  # "mujoco"

    print(f"\n=== Running {solver_label} with num-worlds={num_worlds} ===")
    print(" ".join(cmd))

    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )

    # Stream stderr so you see any issues
    if proc.stderr:
        sys.stderr.write(proc.stderr)

    if proc.returncode != 0:
        print(f"Run failed (return code {proc.returncode}), skipping.")
        return {
            "solver": solver_label,
            "num_worlds": num_worlds,
            "elapsed_s": None,
            "env_fps": None,
            "gpu_used_gb": None,
            "gpu_total_gb": None,
            "ok": False,
        }

    metrics = parse_summary(proc.stdout)
    metrics["solver"] = solver_label  # Override with our label
    metrics["ok"] = True
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Sweep G1 cube stack benchmark over solvers, tiled modes, and num-worlds."
    )
    parser.add_argument(
        "--solvers",
        type=str,
        default="feather_pgs,mujoco",
        help="Comma-separated list of solvers (default: feather_pgs,mujoco). Options: feather_pgs, mujoco",
    )
    parser.add_argument(
        "--tiled-modes",
        type=str,
        default="tiled,non_tiled",
        help="Comma-separated tiled modes for feather_pgs (default: tiled,non_tiled).",
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
        default=14,
        help="Maximum log2(num_worlds). 14 => 16384.",
    )
    parser.add_argument(
        "--sim-substeps",
        type=int,
        default=2,
        help="Simulation substeps per frame.",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=16,
        help="Warmup frames.",
    )
    parser.add_argument(
        "--measure-frames",
        type=int,
        default=64,
        help="Measured frames.",
    )
    parser.add_argument(
        "--pgs-iterations",
        type=int,
        default=4,
        help="PGS iterations (feather_pgs only).",
    )
    parser.add_argument(
        "--dense-max-constraints",
        type=int,
        default=64,
        help="Max constraints per world (feather_pgs only).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output CSV file. Default: g1_cube_stack_bench_{timestamp}.csv",
    )

    args = parser.parse_args()

    solvers = [s.strip() for s in args.solvers.split(",") if s.strip()]
    tiled_modes = [t.strip() for t in args.tiled_modes.split(",") if t.strip()]
    num_worlds_list = [2**k for k in range(args.min_log2_worlds, args.max_log2_worlds + 1)]

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.out or f"g1_cube_stack_bench_{timestamp}.csv")

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

    # Write CSV header immediately
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    rows = []

    # Build list of (solver, use_tiled) configurations
    configs = []
    for solver in solvers:
        if solver == "feather_pgs":
            for tiled_mode in tiled_modes:
                use_tiled = tiled_mode == "tiled"
                configs.append((solver, use_tiled))
        else:
            configs.append((solver, None))  # mujoco doesn't use tiled flag

    for solver, use_tiled in configs:
        for n in num_worlds_list:
            metrics = run_one(solver, use_tiled, n, args)
            row = {
                "solver": metrics.get("solver"),
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

            # Append row immediately (incremental save)
            with out_path.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(row)

            print(f"  -> Saved: {row['solver']}, worlds={n}, env_fps={row['env_fps']}")

    # Final summary
    print(f"\nWrote {len(rows)} rows to {out_path}")
    print("Quick preview:")
    for r in rows:
        print(
            f"  {r['solver']:24s}  worlds={r['num_worlds']:6d}  "
            f"env_fps={r['env_fps']!s:>12}  gpu_used={r['gpu_used_gb']!s:>6}"
        )


if __name__ == "__main__":
    main()
