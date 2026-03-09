# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Derive compact kernel summaries from nightly Nsight reports."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from benchmarks.nightly.nsys_trace import summarize_kernels
from benchmarks.nightly.profiled_worker import _resolve_nsys_bin


def summarize_run_profiles(
    run_dir: Path | str,
    *,
    nsys_bin: str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Export profiled jobs in a run to kernel summary JSON artifacts."""
    resolved_run_dir = Path(run_dir)
    processed = 0
    skipped = 0

    for results_dir in sorted(resolved_run_dir.glob("tasks/*/jobs/*/results")):
        profile_meta_path = results_dir / "profile_meta.json"
        if not profile_meta_path.is_file():
            continue
        summary_path = results_dir / "profile_summary.json"
        if summary_path.is_file() and not force:
            skipped += 1
            continue

        profile_meta = json.loads(profile_meta_path.read_text(encoding="utf-8"))
        report_name = profile_meta.get("report_name")
        if not isinstance(report_name, str) or not report_name:
            continue
        report_path = results_dir / report_name
        if not report_path.is_file():
            continue

        measure_frames = int(profile_meta.get("measure_frames") or 0)
        summary = _summarize_report(report_path, measure_frames=measure_frames, nsys_bin=nsys_bin)
        summary["source_report"] = report_name
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        processed += 1

    return {
        "run_dir": str(resolved_run_dir),
        "processed_profiles": processed,
        "skipped_profiles": skipped,
    }


def _summarize_report(report_path: Path, *, measure_frames: int, nsys_bin: str | None) -> dict[str, Any]:
    tmp_root = Path(os.environ.get("TMPDIR") or tempfile.gettempdir())
    tmp_root.mkdir(parents=True, exist_ok=True)
    resolved_nsys = _resolve_nsys_bin(nsys_bin)
    with tempfile.TemporaryDirectory(prefix="nightly-nsys-summary-", dir=str(tmp_root)) as tmp_dir:
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
            raise RuntimeError(f"Nsight Systems export failed for {report_path}: {exported.returncode}")

        sqlite_path = sqlite_base if sqlite_base.is_file() else sqlite_base.with_suffix(".sqlite")
        if not sqlite_path.is_file():
            raise RuntimeError(f"Nsight Systems export did not produce a SQLite file for {report_path}")
        return summarize_kernels(sqlite_path, measure_frames=measure_frames)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Nightly run directory to enrich.")
    parser.add_argument("--nsys-bin", type=str, default=None, help="Override the nsys binary path.")
    parser.add_argument("--force", action="store_true", help="Rewrite existing profile_summary.json files.")
    args = parser.parse_args()

    summary = summarize_run_profiles(args.run_dir, nsys_bin=args.nsys_bin, force=args.force)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
