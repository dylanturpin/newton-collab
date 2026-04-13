#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bootstrap manifest files for the FeatherPGS explainer artifacts.

This script fixes the artifact contract for later passes without requiring a
runtime benchmark capture yet. It derives the comparison surface from the
checked-in benchmark scenario and preset definitions and emits a manifest under
``.agent/data/fpgs-matrix-free-dense-explainer/``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path

from newton.tools.solver_benchmark import SCENARIOS, SOLVER_PRESETS

ARTIFACT_ROOT = Path(".agent/data/fpgs-matrix-free-dense-explainer")
MANIFEST_PATH = ARTIFACT_ROOT / "m2-capture-manifest.json"
SCHEMA_DIR = ARTIFACT_ROOT / "schema"
SCHEMA_VERSION = "1.0.0"

TARGET_SCENARIOS = ("g1_flat", "h1_tabletop")
COMPARISON_PRESETS = (
    "fpgs_dense_loop",
    "fpgs_dense_row",
    "fpgs_dense_streaming",
    "fpgs_split",
    "fpgs_matrix_free",
)


def get_git_commit() -> str:
    """Return the current short git commit or ``unknown``."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def build_scenario_entry(name: str) -> dict[str, object]:
    """Build the manifest entry for one required scenario."""
    cfg = SCENARIOS[name]
    return {
        "name": name,
        "description": cfg["description"],
        "robot": cfg["robot"],
        "environment": cfg["environment"],
        "default_substeps": cfg["default_substeps"],
        "default_pgs_iterations": cfg["default_pgs_iterations"],
        "default_dense_max_constraints": cfg["default_dense_max_constraints"],
        "mujoco_settings": cfg["mujoco_settings"],
        "planned_outputs": [
            f"scenarios/{name}/fpgs_dense_loop.json",
            f"scenarios/{name}/fpgs_dense_row.json",
            f"scenarios/{name}/fpgs_dense_streaming.json",
            f"scenarios/{name}/fpgs_split.json",
            f"scenarios/{name}/fpgs_matrix_free.json",
        ],
    }


def build_preset_entry(name: str) -> dict[str, object]:
    """Build the manifest entry for one comparison preset."""
    cfg = SOLVER_PRESETS[name]
    settings = {key: value for key, value in cfg.items() if key != "type"}
    return {
        "name": name,
        "type": cfg["type"],
        "pgs_mode": cfg.get("pgs_mode"),
        "settings": settings,
    }


def build_manifest() -> dict[str, object]:
    """Construct the bootstrap manifest."""
    generated_at = dt.datetime.now(tz=dt.UTC).isoformat().replace("+00:00", "Z")
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": generated_at,
        "git_commit": get_git_commit(),
        "artifact_root": str(ARTIFACT_ROOT),
        "source_files": [
            "newton/tools/solver_benchmark.py",
            "newton/_src/solvers/feather_pgs/solver_feather_pgs.py",
        ],
        "schemas": {
            "scenario_sizing": str(SCHEMA_DIR / "scenario-sizing.schema.json"),
            "kernel_memory_analysis": str(SCHEMA_DIR / "kernel-memory-analysis.schema.json"),
        },
        "required_scenarios": [build_scenario_entry(name) for name in TARGET_SCENARIOS],
        "comparison_presets": [build_preset_entry(name) for name in COMPARISON_PRESETS],
        "planned_kernel_outputs": [
            "kernels/dense-loop.json",
            "kernels/dense-row.json",
            "kernels/dense-streaming.json",
            "kernels/matrix-free-gs.json",
            "kernels/tiled-contact-fused.json",
        ],
        "notes": [
            "Scenario-backed capture files are deferred to M3.",
            "Kernel memory-layout artifacts are deferred to M4.",
            "This manifest is generated from the checked-in benchmark surface and should be regenerated if scenario or preset names change.",
        ],
    }


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=MANIFEST_PATH,
        help="Destination JSON manifest path.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate the manifest."""
    args = parse_args()
    manifest = build_manifest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
