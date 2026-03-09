# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run one nightly worker under Nsight Systems and emit compact trace artifacts."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from benchmarks.nightly.nsys_trace import write_perfetto_trace


def run_profiled_worker(
    *,
    results_dir: Path,
    worker_command: list[str],
    measure_frames: int,
    cuda_graph_trace: str,
    nsys_bin: str | None = None,
) -> int:
    """Run one worker command under Nsight Systems and emit report + trace files."""
    resolved_nsys = _resolve_nsys_bin(nsys_bin)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_base = results_dir / "profile"

    profile_command = [
        resolved_nsys,
        "profile",
        "--sample=none",
        "--trace=cuda,nvtx",
        "--cuda-graph-trace",
        cuda_graph_trace,
        "--force-overwrite=true",
        "--output",
        str(out_base),
        *worker_command,
    ]
    completed = subprocess.run(profile_command, capture_output=True, text=True, check=False)
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=os.sys.stderr)
    if completed.returncode != 0:
        return completed.returncode

    report_path = results_dir / "profile.nsys-rep"
    if not report_path.is_file():
        raise RuntimeError(f"Nsight Systems did not produce {report_path.name}")

    trace_path = results_dir / "profile.trace.json"
    tmp_root = Path(os.environ.get("TMPDIR") or tempfile.gettempdir())
    tmp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nightly-nsys-", dir=str(tmp_root)) as tmp_dir:
        sqlite_base = Path(tmp_dir) / "profile"
        export_command = [
            resolved_nsys,
            "export",
            "--force-overwrite=true",
            "--type",
            "sqlite",
            "--output",
            str(sqlite_base),
            str(report_path),
        ]
        exported = subprocess.run(export_command, capture_output=True, text=True, check=False)
        if exported.stdout:
            print(exported.stdout, end="")
        if exported.stderr:
            print(exported.stderr, end="", file=os.sys.stderr)
        if exported.returncode != 0:
            return exported.returncode

        sqlite_path = sqlite_base if sqlite_base.is_file() else sqlite_base.with_suffix(".sqlite")
        if not sqlite_path.is_file():
            raise RuntimeError("Nsight Systems export did not produce a SQLite file.")
        write_perfetto_trace(sqlite_path, trace_path, measure_frames=measure_frames)

    meta_path = results_dir / "profile_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "tool": "nsys",
                "cuda_graph_trace": cuda_graph_trace,
                "measure_frames": measure_frames,
                "report_name": report_path.name,
                "trace_name": trace_path.name,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


def _resolve_nsys_bin(nsys_bin: str | None) -> str:
    if nsys_bin:
        candidate = Path(os.path.expanduser(nsys_bin))
        if candidate.is_file():
            return str(candidate)
    env_value = os.environ.get("NSYS_BIN")
    if env_value:
        candidate = Path(os.path.expanduser(env_value))
        if candidate.is_file():
            return str(candidate)
    bundled = Path.home() / "opt" / "nsight-systems" / "2025.3.2" / "bin" / "nsys"
    if bundled.is_file():
        return str(bundled)
    discovered = shutil.which("nsys")
    if discovered:
        return discovered
    raise RuntimeError("NSYS_BIN is not set and no usable 'nsys' binary was found.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True, help="Job results directory.")
    parser.add_argument("--measure-frames", type=int, required=True, help="Measured frame count for trimming.")
    parser.add_argument(
        "--cuda-graph-trace",
        choices=("graph", "node"),
        default="node",
        help="Nsight Systems CUDA graph trace mode.",
    )
    parser.add_argument("--nsys-bin", type=str, default=None, help="Override path to the nsys binary.")
    parser.add_argument("worker_command", nargs=argparse.REMAINDER, help="Worker command after --.")
    args = parser.parse_args()

    worker_command = list(args.worker_command)
    if worker_command and worker_command[0] == "--":
        worker_command = worker_command[1:]
    if not worker_command:
        parser.error("worker_command must be provided after --")

    return run_profiled_worker(
        results_dir=args.results_dir,
        worker_command=worker_command,
        measure_frames=args.measure_frames,
        cuda_graph_trace=args.cuda_graph_trace,
        nsys_bin=args.nsys_bin,
    )


if __name__ == "__main__":
    raise SystemExit(main())
