#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Capture scenario-backed FeatherPGS sizing artifacts for the explainer lane."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace

import warp as wp

from newton.tools.solver_benchmark import SCENARIOS, SOLVER_PRESETS, build_model, create_solver

ARTIFACT_ROOT = Path(".agent/data/fpgs-matrix-free-dense-explainer")
SCHEMA_VERSION = "1.0.0"
DEFAULT_SCENARIOS = ("g1_flat", "h1_tabletop")
DEFAULT_PRESETS = ("fpgs_dense_row", "fpgs_matrix_free")
SOURCE_FILES = [
    "newton/tools/solver_benchmark.py",
    "newton/_src/solvers/feather_pgs/solver_feather_pgs.py",
    "scripts/analysis/capture_fpgs_scenario_sizing.py",
]


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


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", action="append", choices=sorted(SCENARIOS), dest="scenarios")
    parser.add_argument("--preset", action="append", choices=sorted(SOLVER_PRESETS), dest="presets")
    parser.add_argument("--num-worlds", type=int, default=1, help="Number of worlds to build for each capture.")
    parser.add_argument("--warmup-frames", type=int, default=30, help="Frames to simulate before recording counts.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ARTIFACT_ROOT / "scenarios",
        help="Root directory for scenario output files.",
    )
    return parser.parse_args()


def make_solver_args(*, scenario: str, preset: str, num_worlds: int) -> SimpleNamespace:
    """Build the benchmark-style args namespace for one capture."""
    scenario_cfg = SCENARIOS[scenario]
    return SimpleNamespace(
        scenario=scenario,
        solver=preset,
        num_worlds=num_worlds,
        substeps=scenario_cfg["default_substeps"],
        pgs_iterations=scenario_cfg["default_pgs_iterations"],
        dense_max_constraints=scenario_cfg["default_dense_max_constraints"],
        pgs_beta=0.1,
        pgs_cfm=1.0e-6,
        pgs_omega=1.0,
        pgs_warmstart=False,
        pgs_mode="split",
        cholesky_kernel="auto",
        trisolve_kernel="auto",
        hinv_jt_kernel="auto",
        delassus_kernel="auto",
        pgs_kernel="tiled_contact",
        delassus_chunk_size=None,
        pgs_chunk_size=None,
        use_parallel_streams=True,
        double_buffer=True,
        nvtx=False,
        pgs_debug=False,
        override_scenario_defaults=False,
        mj_solver=None,
        mj_integrator=None,
        mj_njmax=None,
        mj_nconmax=None,
    )


def dtype_name(value: object) -> str:
    """Return a stable dtype name for a Warp array."""
    dtype = getattr(value, "dtype", None)
    if dtype is None:
        return "unknown"
    return str(dtype)


def element_size_bytes(value: object) -> int:
    """Estimate element size in bytes for a Warp array."""
    dtype = dtype_name(value)
    if "float32" in dtype or "int32" in dtype:
        return 4
    if "float64" in dtype or "int64" in dtype:
        return 8
    if "spatial_vector" in dtype or "vec6" in dtype:
        return 24
    if "spatial_matrix" in dtype:
        return 144
    if "transform" in dtype:
        return 28
    if "vec3" in dtype:
        return 12
    if "quat" in dtype:
        return 16
    raise ValueError(f"Unhandled dtype for byte-size estimation: {dtype}")


def shape_list(value: object) -> list[int]:
    """Convert a Warp array shape into plain ints."""
    return [int(dim) for dim in getattr(value, "shape", ())]


def element_count(value: object) -> int:
    """Return the number of elements in a Warp array."""
    shape = shape_list(value)
    if not shape:
        return 1
    return math.prod(shape)


def per_world_elements(total_elements: int, world_count: int) -> int | None:
    """Return a per-world element count when it divides evenly."""
    if world_count <= 0:
        return None
    if total_elements % world_count != 0:
        return None
    return total_elements // world_count


def rigid_contact_count(contacts: object) -> int | None:
    """Best-effort rigid contact count extraction."""
    value = getattr(contacts, "rigid_contact_count", None)
    if value is None:
        return None
    try:
        numpy_value = value.numpy()
    except Exception:
        return None
    if getattr(numpy_value, "shape", ()) == ():
        return int(numpy_value)
    flat = numpy_value.reshape(-1)
    return int(flat[0]) if len(flat) else 0


def solver_world_dofs(solver: object, model: object) -> int:
    """Compute realized world DOFs even when the solver did not allocate J_world/Y_world."""
    art_to_world = getattr(solver, "art_to_world", None)
    art_dof_start = getattr(solver, "articulation_dof_start", None)
    art_h_rows = getattr(solver, "articulation_H_rows", None)
    world_count = getattr(solver, "world_count", 0)
    if art_to_world is None or art_dof_start is None or art_h_rows is None or world_count == 0:
        return 0

    art_to_world_np = art_to_world.numpy()
    art_dof_start_np = art_dof_start.numpy()
    art_h_rows_np = art_h_rows.numpy()

    world_starts = [None] * world_count
    world_ends = [0] * world_count
    for art_idx in range(model.articulation_count):
        world = int(art_to_world_np[art_idx])
        dof_start = int(art_dof_start_np[art_idx])
        dof_end = dof_start + int(art_h_rows_np[art_idx])
        world_starts[world] = dof_start if world_starts[world] is None else min(world_starts[world], dof_start)
        world_ends[world] = max(world_ends[world], dof_end)

    counts = [
        0 if world_starts[world] is None else world_ends[world] - int(world_starts[world])
        for world in range(world_count)
    ]
    return max(counts, default=0)


def add_buffer(
    logical_buffers: list[dict[str, object]],
    *,
    name: str,
    solver_scope: str,
    value: object | None,
    source: str,
    world_count: int,
    notes: str | None = None,
) -> None:
    """Append one logical buffer entry when the buffer exists."""
    if value is None:
        return
    total_elements = element_count(value)
    bytes_per_element = element_size_bytes(value)
    shape = shape_list(value)
    total_bytes = total_elements * bytes_per_element
    logical_buffers.append(
        {
            "name": name,
            "solver_scope": solver_scope,
            "storage": "global",
            "dtype": dtype_name(value),
            "logical_shape": shape,
            "per_world_elements": per_world_elements(total_elements, world_count),
            "total_elements": total_elements,
            "bytes_per_element": bytes_per_element,
            "total_bytes": total_bytes,
            "populated_when": ["allocated in solver initialization", "scenario stepped through warmup frames"],
            "source": source,
            "notes": notes,
        }
    )


def collect_grouped_buffers(
    logical_buffers: list[dict[str, object]],
    grouped: dict[int, object],
    *,
    prefix: str,
    solver_scope: str,
    source_prefix: str,
    world_count: int,
    notes: str | None = None,
) -> None:
    """Record one logical buffer per size-group entry."""
    for size in sorted(grouped):
        add_buffer(
            logical_buffers,
            name=f"{prefix}[{size}]",
            solver_scope=solver_scope,
            value=grouped[size],
            source=f"{source_prefix}[{size}]",
            world_count=world_count,
            notes=notes,
        )


def collect_logical_buffers(solver: object) -> list[dict[str, object]]:
    """Collect the main dense and matrix-free solver buffers."""
    logical_buffers: list[dict[str, object]] = []
    collect_grouped_buffers(
        logical_buffers,
        getattr(solver, "H_by_size", {}),
        prefix="H_by_size",
        solver_scope="shared",
        source_prefix="solver.H_by_size",
        world_count=solver.world_count,
        notes="Per-size grouped reduced mass blocks H.",
    )
    collect_grouped_buffers(
        logical_buffers,
        getattr(solver, "L_by_size", {}),
        prefix="L_by_size",
        solver_scope="shared",
        source_prefix="solver.L_by_size",
        world_count=solver.world_count,
        notes="Per-size grouped Cholesky factors of H.",
    )
    collect_grouped_buffers(
        logical_buffers,
        getattr(solver, "J_by_size", {}),
        prefix="J_by_size",
        solver_scope="dense",
        source_prefix="solver.J_by_size",
        world_count=solver.world_count,
        notes="Dense articulation rows grouped by DOF size.",
    )
    collect_grouped_buffers(
        logical_buffers,
        getattr(solver, "Y_by_size", {}),
        prefix="Y_by_size",
        solver_scope="dense",
        source_prefix="solver.Y_by_size",
        world_count=solver.world_count,
        notes="Per-size H^{-1} J^T rows grouped by articulation DOF count.",
    )
    add_buffer(
        logical_buffers,
        name="C",
        solver_scope="dense",
        value=getattr(solver, "C", None),
        source="solver.C",
        world_count=solver.world_count,
    )
    add_buffer(
        logical_buffers,
        name="J_world",
        solver_scope="matrix_free",
        value=getattr(solver, "J_world", None),
        source="solver.J_world",
        world_count=solver.world_count,
        notes="World-consolidated Jacobian rows used by articulated matrix-free PGS.",
    )
    add_buffer(
        logical_buffers,
        name="Y_world",
        solver_scope="matrix_free",
        value=getattr(solver, "Y_world", None),
        source="solver.Y_world",
        world_count=solver.world_count,
        notes="World-consolidated H^{-1} J^T rows used by articulated matrix-free PGS.",
    )
    add_buffer(
        logical_buffers,
        name="rhs",
        solver_scope="dense",
        value=getattr(solver, "rhs", None),
        source="solver.rhs",
        world_count=solver.world_count,
    )
    add_buffer(
        logical_buffers,
        name="diag",
        solver_scope="dense",
        value=getattr(solver, "diag", None),
        source="solver.diag",
        world_count=solver.world_count,
        notes="Dense path diagonal, reused as the articulated preconditioner in matrix-free mode.",
    )
    add_buffer(
        logical_buffers,
        name="impulses",
        solver_scope="dense",
        value=getattr(solver, "impulses", None),
        source="solver.impulses",
        world_count=solver.world_count,
    )
    add_buffer(
        logical_buffers,
        name="mf_J_a",
        solver_scope="matrix_free",
        value=getattr(solver, "mf_J_a", None),
        source="solver.mf_J_a",
        world_count=solver.world_count,
    )
    add_buffer(
        logical_buffers,
        name="mf_J_b",
        solver_scope="matrix_free",
        value=getattr(solver, "mf_J_b", None),
        source="solver.mf_J_b",
        world_count=solver.world_count,
    )
    add_buffer(
        logical_buffers,
        name="mf_MiJt_a",
        solver_scope="matrix_free",
        value=getattr(solver, "mf_MiJt_a", None),
        source="solver.mf_MiJt_a",
        world_count=solver.world_count,
    )
    add_buffer(
        logical_buffers,
        name="mf_MiJt_b",
        solver_scope="matrix_free",
        value=getattr(solver, "mf_MiJt_b", None),
        source="solver.mf_MiJt_b",
        world_count=solver.world_count,
    )
    add_buffer(
        logical_buffers,
        name="mf_rhs",
        solver_scope="matrix_free",
        value=getattr(solver, "mf_rhs", None),
        source="solver.mf_rhs",
        world_count=solver.world_count,
    )
    add_buffer(
        logical_buffers,
        name="mf_impulses",
        solver_scope="matrix_free",
        value=getattr(solver, "mf_impulses", None),
        source="solver.mf_impulses",
        world_count=solver.world_count,
    )
    add_buffer(
        logical_buffers,
        name="mf_eff_mass_inv",
        solver_scope="matrix_free",
        value=getattr(solver, "mf_eff_mass_inv", None),
        source="solver.mf_eff_mass_inv",
        world_count=solver.world_count,
    )
    add_buffer(
        logical_buffers,
        name="mf_meta_packed",
        solver_scope="matrix_free",
        value=getattr(solver, "mf_meta_packed", None),
        source="solver.mf_meta_packed",
        world_count=solver.world_count,
        notes="Packed matrix-free metadata consumed by the fused tiled-contact kernel.",
    )
    return logical_buffers


def run_capture(*, scenario: str, preset: str, num_worlds: int, warmup_frames: int) -> dict[str, object]:
    """Run one capture and return the artifact payload."""
    args = make_solver_args(scenario=scenario, preset=preset, num_worlds=num_worlds)
    scenario_cfg = SCENARIOS[scenario]
    preset_cfg = SOLVER_PRESETS[preset]

    model = build_model(args, scenario_cfg)
    solver = create_solver(model, args, scenario_cfg)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.collide(state_0)
    sim_dt = (1.0 / 60.0) / float(args.substeps)

    for _ in range(warmup_frames):
        model.collide(state_0, contacts)
        for _ in range(args.substeps):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, contacts, sim_dt)
            state_0, state_1 = state_1, state_0

    wp.synchronize_device()

    dense_counts = getattr(solver, "constraint_count", None)
    dense_rows = int(dense_counts.numpy().reshape(-1)[0]) if dense_counts is not None else 0
    mf_counts = getattr(solver, "mf_constraint_count", None)
    mf_rows = int(mf_counts.numpy().reshape(-1)[0]) if mf_counts is not None else 0

    notes = [
        f"Captured after {warmup_frames} warmup frames at 60 Hz with {args.substeps} substeps/frame.",
        "Artifacts reflect solver allocation sizes plus realized dense/matrix-free row counts in the warmed scenario state.",
    ]
    if preset == "fpgs_dense_row":
        notes.append("Dense-row capture uses the dense articulated path only, so matrix-free row counts may be absent.")
    if preset == "fpgs_matrix_free":
        notes.append(
            "Matrix-free capture still retains dense articulation buffers for grouped H/J/Y data and the diagonal preconditioner."
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "provenance": {
            "generator": "scripts/analysis/capture_fpgs_scenario_sizing.py",
            "git_commit": get_git_commit(),
            "generated_at_utc": dt.datetime.now(tz=dt.UTC).isoformat().replace("+00:00", "Z"),
            "source_files": SOURCE_FILES,
        },
        "scenario": {
            "name": scenario,
            "description": scenario_cfg["description"],
            "robot": scenario_cfg["robot"],
            "environment": scenario_cfg.get("environment"),
            "default_substeps": scenario_cfg["default_substeps"],
            "default_pgs_iterations": scenario_cfg["default_pgs_iterations"],
            "default_dense_max_constraints": scenario_cfg["default_dense_max_constraints"],
            "mujoco_settings": scenario_cfg["mujoco_settings"],
        },
        "solver_preset": {
            "name": preset,
            "type": preset_cfg["type"],
            "pgs_mode": preset_cfg.get("pgs_mode"),
            "settings": {key: value for key, value in preset_cfg.items() if key != "type"},
        },
        "capture": {
            "num_worlds": num_worlds,
            "device": str(model.device),
            "pgs_iterations": args.pgs_iterations,
            "substeps": args.substeps,
        },
        "world_counts": {
            "articulations": model.articulation_count // num_worlds,
            "rigid_bodies": model.body_count // num_worlds,
            "contacts_total": rigid_contact_count(contacts),
            "dense_constraint_rows": dense_rows,
            "matrix_free_constraint_rows": mf_rows,
            "articulation_dofs": int(model.joint_dof_count // num_worlds if num_worlds else model.joint_dof_count),
            "world_dofs": solver_world_dofs(solver, model),
        },
        "logical_buffers": collect_logical_buffers(solver),
        "notes": notes,
    }


def main() -> None:
    """Capture one or more scenario sizing artifacts."""
    args = parse_args()
    scenarios = tuple(args.scenarios or DEFAULT_SCENARIOS)
    presets = tuple(args.presets or DEFAULT_PRESETS)

    wp.config.enable_backward = False

    for scenario in scenarios:
        scenario_dir = args.output_root / scenario
        scenario_dir.mkdir(parents=True, exist_ok=True)
        for preset in presets:
            artifact = run_capture(
                scenario=scenario,
                preset=preset,
                num_worlds=args.num_worlds,
                warmup_frames=args.warmup_frames,
            )
            output_path = scenario_dir / f"{preset}.json"
            output_path.write_text(json.dumps(artifact, indent=2, sort_keys=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
