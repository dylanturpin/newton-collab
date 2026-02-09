# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""
Unified solver benchmark script for comparing FeatherPGS and MuJoCo solvers.

Supports multiple scenarios, sweep/ablation modes, kernel timing, plotting, and JSONL outputs.

Usage examples:

    # Interactive run
    uv run newton/tools/solver_benchmark.py --scenario g1_flat --num-worlds 1

    # Single benchmark run
    uv run newton/tools/solver_benchmark.py --scenario h1_tabletop --solver fpgs_tiled \\
        --num-worlds 4096 --benchmark

    # Sweep over num_worlds with plotting
    uv run newton/tools/solver_benchmark.py --scenario h1_tabletop --sweep \\
        --solvers fpgs_tiled,fpgs_loop,mujoco --plot

    # Ablation study
    uv run newton/tools/solver_benchmark.py --scenario g1_cube_stack --ablation \\
        --num-worlds 4096 --plot

    # Compare kernel timings
    uv run newton/tools/solver_benchmark.py --compare run1.json run2.json
"""

import argparse
import datetime as dt
import json
import os
import platform
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


# =============================================================================
# Scenario Definitions
# =============================================================================

SCENARIOS = {
    "g1_flat": {
        "description": "G1 humanoid on flat ground",
        "robot": "g1",
        "environment": None,
        "storage": "flat",
        "default_substeps": 2,
        "default_pgs_iterations": 4,
        "default_pgs_max_constraints": 32,
        "mujoco_settings": {
            "njmax": 210,
            "nconmax": 35,
            "solver": "newton",
            "integrator": "implicitfast",
            "cone": "pyramidal",
            "ls_parallel": True,
        },
    },
    "g1_cube_stack": {
        "description": "G1 on stack of cubes (multi-articulation)",
        "robot": "g1",
        "environment": "cube_stack",
        "storage": "batched",
        "default_substeps": 2,
        "default_pgs_iterations": 8,
        "default_pgs_max_constraints": 128,
        "mujoco_settings": {
            "njmax": 256,
            "nconmax": 42,
            "solver": "newton",
            "integrator": "implicit",
            "cone": "pyramidal",
            "ls_parallel": True,
        },
    },
    "h1_flat": {
        "description": "H1 humanoid on flat ground",
        "robot": "h1",
        "environment": None,
        "storage": "flat",
        "default_substeps": 4,
        "default_pgs_iterations": 8,
        "default_pgs_max_constraints": 32,
        "mujoco_settings": {
            "njmax": 65,
            "nconmax": 15,
            "solver": "newton",
            "integrator": "implicitfast",
            "cone": "pyramidal",
            "ls_parallel": True,
        },
    },
    "h1_tabletop": {
        "description": "H1 humanoid with tabletop objects",
        "robot": "h1",
        "environment": "tabletop",
        "storage": "batched",
        "default_substeps": 8,
        "default_pgs_iterations": 8,
        "default_pgs_max_constraints": 384,
        # High contact count (~150) exceeds tiled kernel shared memory limits.
        # Use streaming kernel which streams block-rows from global memory.
        "default_cholesky_kernel": "tiled",
        "default_trisolve_kernel": "tiled",
        "default_hinv_jt_kernel": "par_row",
        "default_delassus_kernel": "tiled",
        "default_pgs_kernel": "streaming",
        "ablation_sequence": "streaming",
        "mujoco_settings": {
            "njmax": 512,
            "nconmax": 128,
            "solver": "newton",
            "integrator": "implicitfast",
            "cone": "pyramidal",
            "ls_parallel": True,
        },
    },
}

# =============================================================================
# Solver Presets
# =============================================================================

SOLVER_PRESETS = {
    "mujoco": {
        "type": "mujoco",
    },
    "fpgs_loop": {
        "type": "feather_pgs",
        "cholesky_kernel": "loop",
        "trisolve_kernel": "loop",
        "hinv_jt_kernel": "par_row",
        "delassus_kernel": "par_row_col",
        "pgs_kernel": "loop",
        "use_parallel_streams": False,
    },
    "fpgs_tiled": {
        "type": "feather_pgs",
        "cholesky_kernel": "tiled",
        "trisolve_kernel": "tiled",
        "hinv_jt_kernel": "tiled",
        "delassus_kernel": "tiled",
        "pgs_kernel": "tiled_contact",
        "use_parallel_streams": True,
    },
    "fpgs_tiled_row": {
        "type": "feather_pgs",
        "cholesky_kernel": "tiled",
        "trisolve_kernel": "tiled",
        "hinv_jt_kernel": "tiled",
        "delassus_kernel": "tiled",
        "pgs_kernel": "tiled_row",
        "use_parallel_streams": True,
    },
    "fpgs_tiled_contact": {
        "type": "feather_pgs",
        "cholesky_kernel": "tiled",
        "trisolve_kernel": "tiled",
        "hinv_jt_kernel": "tiled",
        "delassus_kernel": "tiled",
        "pgs_kernel": "tiled_contact",
        "use_parallel_streams": True,
    },
    "fpgs_streaming": {
        "type": "feather_pgs",
        "cholesky_kernel": "tiled",
        "trisolve_kernel": "tiled",
        "hinv_jt_kernel": "par_row",
        "delassus_kernel": "tiled",
        "pgs_kernel": "streaming",
        "use_parallel_streams": False,
    },
    "feather_pgs": {
        "type": "feather_pgs",
        # Use defaults from CLI or scenario
    },
}

# =============================================================================
# Ablation Sequence
# =============================================================================

ABLATION_SEQUENCES = {
    # Default: for low-constraint scenarios where tiled hinv_jt and tiled PGS fit in shared memory
    # (e.g. g1_flat with 32 constraints, g1_cube_stack with 128 constraints)
    "default": [
        {
            "label": "baseline (all loop)",
            "cholesky_kernel": "loop",
            "trisolve_kernel": "loop",
            "hinv_jt_kernel": "par_row",
            "delassus_kernel": "par_row_col",
            "pgs_kernel": "loop",
            "use_parallel_streams": False,
        },
        {
            "label": "+ tiled cholesky",
            "cholesky_kernel": "tiled",
            "trisolve_kernel": "loop",
            "hinv_jt_kernel": "par_row",
            "delassus_kernel": "par_row_col",
            "pgs_kernel": "loop",
            "use_parallel_streams": False,
        },
        {
            "label": "+ tiled trisolve",
            "cholesky_kernel": "tiled",
            "trisolve_kernel": "tiled",
            "hinv_jt_kernel": "par_row",
            "delassus_kernel": "par_row_col",
            "pgs_kernel": "loop",
            "use_parallel_streams": False,
        },
        {
            "label": "+ tiled hinv_jt",
            "cholesky_kernel": "tiled",
            "trisolve_kernel": "tiled",
            "hinv_jt_kernel": "tiled",
            "delassus_kernel": "tiled",
            "pgs_kernel": "loop",
            "use_parallel_streams": False,
        },
        {
            "label": "+ tiled PGS",
            "cholesky_kernel": "tiled",
            "trisolve_kernel": "tiled",
            "hinv_jt_kernel": "tiled",
            "delassus_kernel": "tiled",
            "pgs_kernel": "tiled_contact",  # Will be overridden by --ablation-pgs if specified
            "use_parallel_streams": False,
        },
        {
            "label": "+ parallel streams",
            "cholesky_kernel": "tiled",
            "trisolve_kernel": "tiled",
            "hinv_jt_kernel": "tiled",
            "delassus_kernel": "tiled",
            "pgs_kernel": "tiled_contact",
            "use_parallel_streams": True,
        },
    ],
    # Streaming: for high-constraint scenarios where tiled hinv_jt and tiled PGS exceed shared
    # memory limits (e.g. h1_tabletop with 1024 constraints). Uses par_row hinv_jt, streaming
    # Delassus (chunked shared memory), and streaming PGS throughout.
    "streaming": [
        {
            "label": "baseline (all loop)",
            "cholesky_kernel": "loop",
            "trisolve_kernel": "loop",
            "hinv_jt_kernel": "par_row",
            "delassus_kernel": "par_row_col",
            "pgs_kernel": "loop",
            "use_parallel_streams": False,
        },
        {
            "label": "+ tiled cholesky",
            "cholesky_kernel": "tiled",
            "trisolve_kernel": "loop",
            "hinv_jt_kernel": "par_row",
            "delassus_kernel": "par_row_col",
            "pgs_kernel": "loop",
            "use_parallel_streams": False,
        },
        {
            "label": "+ tiled trisolve",
            "cholesky_kernel": "tiled",
            "trisolve_kernel": "tiled",
            "hinv_jt_kernel": "par_row",
            "delassus_kernel": "par_row_col",
            "pgs_kernel": "loop",
            "use_parallel_streams": False,
        },
        {
            "label": "+ tiled delassus",
            "cholesky_kernel": "tiled",
            "trisolve_kernel": "tiled",
            "hinv_jt_kernel": "par_row",
            "delassus_kernel": "tiled",
            "pgs_kernel": "loop",
            "use_parallel_streams": False,
        },
        {
            "label": "+ streaming PGS",
            "cholesky_kernel": "tiled",
            "trisolve_kernel": "tiled",
            "hinv_jt_kernel": "par_row",
            "delassus_kernel": "tiled",
            "pgs_kernel": "streaming",  # Will be overridden by --ablation-pgs if specified
            "use_parallel_streams": False,
        },
        {
            "label": "+ parallel streams",
            "cholesky_kernel": "tiled",
            "trisolve_kernel": "tiled",
            "hinv_jt_kernel": "par_row",
            "delassus_kernel": "tiled",
            "pgs_kernel": "streaming",
            "use_parallel_streams": True,
        },
    ],
}

# =============================================================================
# Kernel to Stage Mapping
# =============================================================================

STAGE_PATTERNS = {
    "0_collision": [
        # Shared - broadphase and contact pair generation
        "broadphase_collision_pairs", "generate_handle_contact_pairs",
        # MuJoCo - narrowphase
        "_primitive_narrowphase", "_nxn_broadphase",
    ],
    "1_fk_id": [
        # FeatherPGS
        "eval_rigid_fk", "eval_rigid_id", "eval_rigid_mass", "compute_com", "compute_spatial",
        "compute_composite_inertia",
        # MuJoCo
        "_kinematics_level",
    ],
    "1_drives": [
        # FeatherPGS
        "eval_joint_drives", "eval_rigid_tau", "clamp_joint_tau",
        # MuJoCo - actuator velocity from joint velocities
        "_actuator_velocity", "_qderiv_actuator",
    ],
    "1_crba": [
        # FeatherPGS
        "crba_",
        # MuJoCo - mass matrix setup
        "update_gradient_set_h_qM",
    ],
    "2_cholesky": [
        # FeatherPGS
        "cholesky_",
        # MuJoCo - sparse Cholesky factorization accumulation
        "_qLD_acc",
        # MuJoCo - sparse L*D*L^T accumulation sweeps (part of factorization)
        "_solve_LD_sparse_x_acc",
    ],
    "3_trisolve": [
        # FeatherPGS
        "trisolve_",
        # MuJoCo - forward/back substitution for L*x=y and L^T*x=y
        "_solve_LD_sparse",
    ],
    "3_v_hat": [
        # FeatherPGS
        "compute_velocity_predictor", "compute_v_hat",
    ],
    "4_contact_build": [
        # FeatherPGS
        "build_contact_row", "build_augmented", "allocate_world_contact", "clamp_contact",
        "populate_world_J",
        # MuJoCo - contact constraint generation with pyramidal friction
        "_efc_contact", "update_constraint_efc",
    ],
    "4_hinv_jt": [
        # FeatherPGS - compute H^-1 * J^T
        "hinv_jt_",
    ],
    "4_delassus": [
        # FeatherPGS - Delassus matrix G = J * H^-1 * J^T (constraint space)
        "delassus_", "finalize_world_diag_cfm",
    ],
    "4_hessian": [
        # MuJoCo - build system Hessian: H + J^T * D * J (DOF space)
        "update_gradient_JTDAJ", "mul_m_sparse",
    ],
    "4_rhs": [
        # FeatherPGS
        "compute_world_contact_bias", "contact_bias_", "compute_rhs", "rhs_accum",
    ],
    "5_solver": [
        # FeatherPGS (PGS iterations)
        "pgs_solve", "prepare_impulses", "prepare_world_impulses",
        # MuJoCo (Newton solver iterations with line search)
        "linesearch_jv", "linesearch_parallel", "update_gradient_cholesky",
        "update_gradient_grad", "solve_init_jaref", "solve_search_update",
    ],
    "6_apply": [
        # FeatherPGS
        "apply_impulse", "apply_augmented",
        # MuJoCo - map constraint forces to DOF space: qfrc += J^T * efc_force
        "update_constraint_init_qfrc",
    ],
    "6_integrate": [
        # FeatherPGS
        "integrate_", "update_qdd", "update_body_qd", "scatter_qdd", "eval_fk",
    ],
}


def classify_kernel(kernel_name: str) -> str:
    """Classify a kernel name into a stage."""
    name_lower = kernel_name.lower()
    for stage, patterns in STAGE_PATTERNS.items():
        for pattern in patterns:
            if pattern.lower() in name_lower:
                return stage
    return "other"


# =============================================================================
# Output Parsing
# =============================================================================

RE_ELAPSED = re.compile(r"Elapsed time \(s\):\s*([0-9.]+)")
RE_ENVFPS = re.compile(r"Env-FPS \(env/s\):\s*([0-9,\.]+)")
RE_GPU_USED = re.compile(r"GPU memory used \(GB\):\s*([0-9.]+)")
RE_GPU_TOTAL = re.compile(r"GPU memory total \(GB\):\s*([0-9.]+)")
RE_WORLDS = re.compile(r"Worlds:\s*([0-9]+)")
RE_SOLVER = re.compile(r"Solver:\s*([A-Za-z0-9_]+)")


def parse_benchmark_output(text: str) -> dict:
    """Parse benchmark summary from stdout."""
    def get_float(regex, default=None):
        m = regex.search(text)
        if not m:
            return default
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


def parse_kernel_timing(text: str) -> dict:
    """Parse kernel timing from --summary-timer output."""
    # Look for kernel timing lines in the format:
    # kernel_name    TIME    PERCENT    CUMULATIVE
    kernels = {}
    lines = text.split("\n")
    in_timing_section = False

    for line in lines:
        # Detect timing table
        if "Kernel" in line and "Time" in line and "%" in line:
            in_timing_section = True
            continue
        if in_timing_section:
            if line.strip().startswith("-"):
                continue
            if not line.strip():
                in_timing_section = False
                continue
            # Parse kernel line
            parts = line.split()
            if len(parts) >= 2:
                try:
                    kernel_name = parts[0]
                    time_ms = float(parts[1])
                    kernels[kernel_name] = time_ms
                except (ValueError, IndexError):
                    pass

    return kernels


# =============================================================================
# Metadata Collection
# =============================================================================

def get_gpu_info() -> tuple[str, float]:
    """Get GPU name and total memory."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            _, total = torch.cuda.mem_get_info()
            return name, total / 1024**3
    except ImportError:
        pass
    return "Unknown", 0.0


def get_git_hash() -> str:
    """Get current git hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def collect_metadata(args, scenario_cfg: dict) -> dict:
    """Collect run metadata."""
    gpu_name, gpu_total = get_gpu_info()
    return {
        "scenario": args.scenario,
        "scenario_description": scenario_cfg["description"],
        "timestamp": dt.datetime.now().isoformat(),
        "gpu": gpu_name,
        "gpu_memory_total_gb": gpu_total,
        "platform": platform.system(),
        "python_version": platform.python_version(),
        "newton_git_hash": get_git_hash(),
        "substeps": args.substeps,
        "pgs_iterations": args.pgs_iterations,
        "warmup_frames": args.warmup_frames,
        "measure_frames": args.measure_frames,
    }


# =============================================================================
# Subprocess Runner
# =============================================================================

def build_run_command(args, solver_config: dict, num_worlds: int, substeps: int | None = None) -> list[str]:
    """Build command line for a single benchmark run."""
    scenario_cfg = SCENARIOS[args.scenario]
    substeps = substeps if substeps is not None else args.substeps

    cmd = [
        sys.executable, "-m", "newton.tools.solver_benchmark",
        "--scenario", args.scenario,
        "--num-worlds", str(num_worlds),
        "--substeps", str(substeps),
        "--warmup-frames", str(args.warmup_frames),
        "--measure-frames", str(args.measure_frames),
        "--benchmark",
        "--viewer", "null",
    ]

    solver_type = solver_config.get("type", "feather_pgs")

    if solver_type == "mujoco":
        cmd.extend(["--solver", "mujoco"])
        # Add mujoco-specific settings from scenario
        mj = scenario_cfg.get("mujoco_settings", {})
        if mj.get("njmax"):
            cmd.extend(["--mj-njmax", str(mj["njmax"])])
        if mj.get("nconmax"):
            cmd.extend(["--mj-nconmax", str(mj["nconmax"])])
        if mj.get("solver"):
            cmd.extend(["--mj-solver", mj["solver"]])
        if mj.get("integrator"):
            cmd.extend(["--mj-integrator", mj["integrator"]])
    else:
        cmd.extend(["--solver", "feather_pgs"])

        # When kernel configs are explicitly specified (e.g. ablation steps),
        # override scenario defaults so the CLI args take effect.
        if any(k.endswith("_kernel") for k in solver_config):
            cmd.append("--override-scenario-defaults")

        # Storage
        storage = solver_config.get("storage") or scenario_cfg.get("storage", "batched")
        cmd.extend(["--storage", storage])

        # Kernel options
        if "cholesky_kernel" in solver_config:
            cmd.extend(["--cholesky-kernel", solver_config["cholesky_kernel"]])
        if "trisolve_kernel" in solver_config:
            cmd.extend(["--trisolve-kernel", solver_config["trisolve_kernel"]])
        if "hinv_jt_kernel" in solver_config:
            cmd.extend(["--hinv-jt-kernel", solver_config["hinv_jt_kernel"]])
        if "delassus_kernel" in solver_config:
            cmd.extend(["--delassus-kernel", solver_config["delassus_kernel"]])
        if "pgs_kernel" in solver_config:
            cmd.extend(["--pgs-kernel", solver_config["pgs_kernel"]])

        # Streams
        if solver_config.get("use_parallel_streams", False):
            cmd.append("--use-parallel-streams")
        else:
            cmd.append("--no-parallel-streams")

        # PGS params
        cmd.extend(["--pgs-iterations", str(args.pgs_iterations)])
        cmd.extend(["--pgs-max-constraints", str(args.pgs_max_constraints)])
        cmd.extend(["--pgs-beta", str(args.pgs_beta)])
        cmd.extend(["--pgs-cfm", str(args.pgs_cfm)])
        cmd.extend(["--pgs-omega", str(args.pgs_omega)])
        if args.pgs_warmstart:
            cmd.append("--pgs-warmstart")

        # Chunk sizes: solver_config overrides CLI
        delassus_chunk = solver_config.get("delassus_chunk_size", getattr(args, "delassus_chunk_size", None))
        if delassus_chunk is not None:
            cmd.extend(["--delassus-chunk-size", str(delassus_chunk)])
        pgs_chunk = solver_config.get("pgs_chunk_size", getattr(args, "pgs_chunk_size", None))
        if pgs_chunk is not None:
            cmd.extend(["--pgs-chunk-size", str(pgs_chunk)])

    if args.timing_out or args.ablation:
        cmd.append("--summary-timer")

    return cmd


def run_subprocess(cmd: list[str], label: str) -> dict:
    """Run a benchmark subprocess and parse results."""
    print(f"\n{'='*60}")
    print(f"Running: {label}")
    print(f"{'='*60}")
    print(" ".join(cmd[:10]) + " ...")

    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.stderr:
        # Print any warnings/errors
        for line in proc.stderr.split("\n"):
            if "error" in line.lower() or "warning" in line.lower():
                print(f"  {line}")

    if proc.returncode != 0:
        print(f"  FAILED (return code {proc.returncode})")
        return {"ok": False, "label": label}

    metrics = parse_benchmark_output(proc.stdout)
    metrics["ok"] = True
    metrics["label"] = label
    metrics["kernels"] = parse_kernel_timing(proc.stdout)
    metrics["stdout"] = proc.stdout

    if metrics["env_fps"]:
        print(f"  env_fps: {metrics['env_fps']:,.0f}")
        print(f"  gpu_used: {metrics.get('gpu_used_gb', 'N/A')} GB")

    return metrics


# =============================================================================
# Sweep Mode
# =============================================================================

def run_sweep(args):
    """Run sweep over num_worlds for multiple solvers (and optionally substeps)."""
    scenario_cfg = SCENARIOS[args.scenario]

    # Parse solvers
    solver_names = [s.strip() for s in args.solvers.split(",")]
    solver_configs = []
    for name in solver_names:
        if name in SOLVER_PRESETS:
            config = SOLVER_PRESETS[name].copy()
            config["name"] = name
            solver_configs.append(config)
        else:
            print(f"Warning: Unknown solver preset '{name}', skipping")

    # Parse substeps list
    if args.substeps_list:
        substeps_values = [int(s.strip()) for s in args.substeps_list.split(",")]
    else:
        substeps_values = [args.substeps]

    # World counts
    world_counts = [2**k for k in range(args.min_log2_worlds, args.max_log2_worlds + 1)]

    # Output directory
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else Path(f"results/{args.scenario}_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSONL setup
    jsonl_path = out_dir / "sweep.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()

    results = []
    multi_substeps = len(substeps_values) > 1

    for solver_config in solver_configs:
        for substeps in substeps_values:
            for num_worlds in world_counts:
                cmd = build_run_command(args, solver_config, num_worlds, substeps=substeps)
                if multi_substeps:
                    label = f"{solver_config['name']} ({substeps} sub) @ {num_worlds} worlds"
                else:
                    label = f"{solver_config['name']} @ {num_worlds} worlds"
                metrics = run_subprocess(cmd, label)

                row = {
                    "solver": solver_config["name"],
                    "substeps": substeps,
                    "num_worlds": num_worlds,
                    "env_fps": metrics.get("env_fps"),
                    "gpu_used_gb": metrics.get("gpu_used_gb"),
                    "elapsed_s": metrics.get("elapsed_s"),
                    "ok": metrics.get("ok", False),
                    "label": metrics.get("label"),
                }
                results.append(row)

                # Append to JSONL
                with open(jsonl_path, "a") as f:
                    f.write(json.dumps(row) + "\n")

    # Save metadata
    metadata = collect_metadata(args, scenario_cfg)
    metadata["solvers"] = solver_names
    metadata["substeps_values"] = substeps_values
    metadata["world_counts"] = world_counts
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nResults saved to: {out_dir}")
    print(f"  sweep.jsonl: {jsonl_path}")

    # Plot if requested
    if args.plot:
        plot_sweep(jsonl_path, out_dir / "sweep.png", args.scenario, metadata.get("gpu", ""))

    return results


# =============================================================================
# Ablation Mode
# =============================================================================

def run_ablation(args):
    """Run ablation study."""
    scenario_cfg = SCENARIOS[args.scenario]

    # Output directory
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else Path(f"results/{args.scenario}_ablation_{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "kernels").mkdir(exist_ok=True)

    # Select ablation sequence based on scenario
    seq_name = scenario_cfg.get("ablation_sequence", "default")
    ablation_seq = ABLATION_SEQUENCES[seq_name]

    # Prepare ablation sequence
    ablation_steps = []
    for step in ablation_seq:
        step_config = step.copy()
        step_config["type"] = "feather_pgs"
        # Override PGS kernel if specified
        if args.ablation_pgs != "auto":
            if "PGS" in step_config["label"] or "parallel streams" in step_config["label"]:
                step_config["pgs_kernel"] = args.ablation_pgs
        ablation_steps.append(step_config)

    # Add MuJoCo baseline
    ablation_steps.append({
        "label": "MuJoCo baseline",
        "type": "mujoco",
    })

    # Run ablation
    results = []
    for i, step_config in enumerate(ablation_steps):
        cmd = build_run_command(args, step_config, args.num_worlds)
        metrics = run_subprocess(cmd, step_config["label"])

        metrics["step_index"] = i
        metrics["config"] = step_config
        results.append(metrics)

        # Save kernel timing
        if metrics.get("kernels"):
            kernel_file = out_dir / "kernels" / f"step_{i}_{step_config['label'].replace(' ', '_').replace('+', '')[:30]}.json"
            kernel_data = {
                "label": step_config["label"],
                "config": step_config,
                "env_fps": metrics.get("env_fps"),
                "kernels": metrics.get("kernels", {}),
            }
            with open(kernel_file, "w") as f:
                json.dump(kernel_data, f, indent=2)

    # Print ablation table
    print("\n" + "=" * 70)
    print(f"ABLATION RESULTS: {args.scenario} @ {args.num_worlds} worlds")
    print("=" * 70)
    print(f"{'Configuration':<45} {'Env-FPS':>12} {'vs baseline':>12}")
    print("-" * 70)

    baseline_fps = None
    for r in results:
        fps = r.get("env_fps")
        label = r.get("label", "?")

        if fps is None:
            print(f"{label:<45} {'FAILED':>12}")
            continue

        if baseline_fps is None:
            baseline_fps = fps
            delta_str = "-"
        else:
            delta_pct = (fps - baseline_fps) / baseline_fps * 100
            delta_str = f"+{delta_pct:.1f}%" if delta_pct >= 0 else f"{delta_pct:.1f}%"

        print(f"{label:<45} {fps:>12,.0f} {delta_str:>12}")

    print("=" * 70)

    # Save results
    ablation_jsonl = out_dir / "ablation.jsonl"
    if ablation_jsonl.exists():
        ablation_jsonl.unlink()
    with open(ablation_jsonl, "w") as f:
        for r in results:
            f.write(json.dumps({
                "label": r.get("label"),
                "env_fps": r.get("env_fps"),
                "gpu_used_gb": r.get("gpu_used_gb"),
                "ok": r.get("ok"),
                "solver": r.get("solver"),
                "num_worlds": r.get("num_worlds"),
                "step_index": r.get("step_index"),
            }) + "\n")

    # Save metadata
    metadata = collect_metadata(args, scenario_cfg)
    metadata["num_worlds"] = args.num_worlds
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nResults saved to: {out_dir}")

    # Plot if requested
    if args.plot:
        plot_ablation(results, out_dir / "ablation.png", args.scenario, args.num_worlds, metadata.get("gpu", ""))

    return results


# =============================================================================
# Compare Mode
# =============================================================================

def run_compare(args):
    """Compare kernel timing from multiple JSON files."""
    files = args.compare

    if len(files) < 2:
        print("Need at least 2 files to compare")
        return

    # Load data
    data = []
    for f in files:
        with open(f) as fp:
            d = json.load(fp)
            d["file"] = f
            data.append(d)

    # Collect all stages
    all_kernels = set()
    for d in data:
        all_kernels.update(d.get("kernels", {}).keys())

    # Group by stage
    stage_kernels = defaultdict(set)
    for k in all_kernels:
        stage = classify_kernel(k)
        stage_kernels[stage].add(k)

    # Print comparison
    print("\n" + "=" * 80)
    print("KERNEL COMPARISON BY STAGE")
    print("=" * 80)

    # Header
    labels = [d.get("label", Path(d["file"]).stem)[:20] for d in data]
    col_width_hdr = 22
    header = f"{'Kernel':<40}"
    for label in labels:
        header += f" {label:>{col_width_hdr}}"
    print(header)
    print("-" * (40 + (col_width_hdr + 1) * len(labels)))

    stage_order = ["1_fk_id", "1_drives", "1_crba", "2_cholesky", "3_trisolve", "3_v_hat",
                   "4_contact_build", "4_hinv_jt", "4_delassus", "4_hessian", "4_rhs", "5_solver",
                   "6_apply", "6_integrate", "other"]

    stage_totals = {label: defaultdict(float) for label in labels}

    # Accumulate totals first
    for stage in stage_order:
        for kernel in stage_kernels.get(stage, []):
            for i, d in enumerate(data):
                time_ms = d.get("kernels", {}).get(kernel)
                if time_ms is not None:
                    stage_totals[labels[i]][stage] += time_ms

    grand_totals_kernel = {label: sum(stage_totals[label].values()) for label in labels}

    col_width = 22
    for stage in stage_order:
        kernels = sorted(stage_kernels.get(stage, []))
        if not kernels:
            continue

        print(f"\n[{stage}]")
        for kernel in kernels:
            name = kernel if len(kernel) <= 38 else kernel[:35] + "..."
            row = f"  {name:<38}"
            for i, d in enumerate(data):
                time_ms = d.get("kernels", {}).get(kernel)
                if time_ms is not None:
                    gt = grand_totals_kernel[labels[i]]
                    pct = time_ms / gt * 100 if gt > 0 else 0
                    row += f" {f'{time_ms:.1f}ms ({pct:.0f}%)':>{col_width}}"
                else:
                    row += f" {'-':>{col_width}}"
            print(row)

    # Stage totals
    grand_totals = {label: sum(stage_totals[label].values()) for label in labels}

    print("\n" + "=" * 80)
    print("STAGE TOTALS")
    print("=" * 80)
    col_width = 22
    print(f"{'Stage':<20}", end="")
    for label in labels:
        print(f" {label:>{col_width}}", end="")
    if len(labels) == 2:
        print(f" {'Speedup':>10}", end="")
    print()
    print("-" * (20 + (col_width + 1) * len(labels) + (11 if len(labels) == 2 else 0)))

    for stage in stage_order:
        has_data = any(stage_totals[label].get(stage, 0) > 0 for label in labels)
        if not has_data:
            continue
        print(f"{stage:<20}", end="")
        times = []
        for label in labels:
            t = stage_totals[label].get(stage, 0)
            times.append(t)
            if t > 0 and grand_totals[label] > 0:
                pct = t / grand_totals[label] * 100
                print(f" {f'{t:.1f}ms ({pct:.0f}%)':>{col_width}}", end="")
            else:
                print(f" {'-':>{col_width}}", end="")
        if len(times) == 2 and times[1] > 0:
            speedup = times[1] / times[0] if times[0] > 0 else 0
            print(f" {speedup:>9.2f}x", end="")
        print()

    # Grand total row
    print(f"{'TOTAL':<20}", end="")
    for label in labels:
        gt = grand_totals[label]
        print(f" {f'{gt:.1f}ms':>{col_width}}", end="")
    if len(labels) == 2:
        gt0, gt1 = grand_totals[labels[0]], grand_totals[labels[1]]
        if gt0 > 0:
            print(f" {gt1 / gt0:>9.2f}x", end="")
    print()

    print("=" * (20 + (col_width + 1) * len(labels) + (11 if len(labels) == 2 else 0)))


# =============================================================================
# Plotting
# =============================================================================

def _load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file into list of dicts."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def plot_sweep(jsonl_path: Path, out_path: Path, scenario: str, gpu_name: str):
    """Generate sweep plot with support for multiple substeps values."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    # Load data keyed by (solver, substeps)
    series = {}
    all_substeps = set()
    for row in _load_jsonl(jsonl_path):
        solver = row.get("solver")
        if not solver:
            continue
        substeps = int(row.get("substeps", 0)) if row.get("substeps") is not None else 0
        all_substeps.add(substeps)
        key = (solver, substeps)
        if key not in series:
            series[key] = {"num_worlds": [], "env_fps": [], "gpu_used_gb": []}
        try:
            series[key]["num_worlds"].append(int(row["num_worlds"]))
            series[key]["env_fps"].append(float(row["env_fps"]) if row.get("env_fps") is not None else None)
            series[key]["gpu_used_gb"].append(float(row["gpu_used_gb"]) if row.get("gpu_used_gb") is not None else None)
        except (ValueError, TypeError):
            continue

    if not series:
        print("No valid data to plot")
        return

    # Determine if we have multiple substeps values
    all_substeps.discard(0)  # Remove placeholder
    multi_substeps = len(all_substeps) > 1
    sorted_substeps = sorted(all_substeps) if all_substeps else [0]

    # Colors by solver
    COLORS = {
        "fpgs_tiled": "#1f77b4",
        "fpgs_tiled_contact": "#1f77b4",
        "fpgs_tiled_row": "#2ca02c",
        "fpgs_loop": "#ff7f0e",
        "mujoco": "#9467bd",
    }

    SOLVER_LABELS = {
        "fpgs_tiled": "FeatherPGS",
        "fpgs_tiled_contact": "FeatherPGS",
        "fpgs_tiled_row": "FeatherPGS (tiled_row)",
        "fpgs_loop": "FeatherPGS (loop)",
        "mujoco": "MJWarp",
    }

    # Line styles by substeps index (solid for first, dashed for second, etc.)
    LINE_STYLES = ["-", "--", ":", "-."]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for (solver, substeps), data in sorted(series.items()):
        color = COLORS.get(solver, "#333333")
        base_label = SOLVER_LABELS.get(solver, solver)

        # Determine line style based on substeps index
        if multi_substeps and substeps in sorted_substeps:
            substeps_idx = sorted_substeps.index(substeps)
            linestyle = LINE_STYLES[substeps_idx % len(LINE_STYLES)]
            label = f"{base_label} ({substeps} substeps)"
        else:
            linestyle = "-"
            label = base_label

        ax1.plot(data["num_worlds"], data["env_fps"], marker="o", linestyle=linestyle,
                 color=color, label=label, markersize=6)
        ax2.plot(data["num_worlds"], data["gpu_used_gb"], marker="o", linestyle=linestyle,
                 color=color, label=label, markersize=6)

    ax1.set_xlabel("num_envs")
    ax1.set_ylabel("Env-FPS")
    ax1.set_title("FPS vs Num Envs")
    ax1.set_ylim(bottom=0)
    ax1.ticklabel_format(style="plain", axis="both")
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("num_envs")
    ax2.set_ylabel("GPU Memory (GB)")
    ax2.set_title("GPU Memory Usage vs Num Envs")
    ax2.set_ylim(bottom=0)
    ax2.ticklabel_format(style="plain", axis="x")
    ax2.grid(True, alpha=0.3)

    # Build title
    title = f"FeatherPGS vs. MJWarp ({SCENARIOS[scenario]['description']}"
    if gpu_name:
        title += f", {gpu_name}"
    title += ")"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    handles, labels_list = ax1.get_legend_handles_labels()
    ncol = min(len(handles), 4)
    fig.legend(handles, labels_list, loc="lower center", ncol=ncol, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.18)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Plot saved: {out_path}")
    plt.close()


def plot_ablation(results: list, out_path: Path, scenario: str, num_worlds: int, gpu_name: str = ""):
    """Generate ablation bar chart."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    labels = []
    fps_values = []
    colors = []

    for r in results:
        if r.get("env_fps") is None:
            continue
        label = r.get("label", "?")
        # Shorten label
        if label.startswith("+ "):
            label = label[2:]
        labels.append(label[:25])
        fps_values.append(r["env_fps"])
        # Color: blue for feather_pgs steps, purple for mujoco
        if "mujoco" in label.lower():
            colors.append("#9467bd")
        else:
            colors.append("#1f77b4")

    if not labels:
        print("No valid data to plot")
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(labels))
    bars = ax.bar(x, fps_values, color=colors)

    ax.set_xlabel("Configuration")
    ax.set_ylabel("Env-FPS")
    title = f"Ablation: {SCENARIOS[scenario]['description']} @ {num_worlds} worlds"
    if gpu_name:
        title += f" ({gpu_name})"
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(bottom=0)
    ax.ticklabel_format(style="plain", axis="y")
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, val in zip(bars, fps_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:,.0f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Plot saved: {out_path}")
    plt.close()


# =============================================================================
# Model and Solver Creation (shared between run modes)
# =============================================================================

def build_model(args, scenario_cfg: dict):
    """Build and finalize the simulation model for a scenario.

    Returns:
        The finalized newton.Model.
    """
    import warp as wp

    import newton
    import newton.examples
    import newton.utils

    # Build articulation
    articulation_builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(articulation_builder)

    robot = scenario_cfg["robot"]
    env = scenario_cfg.get("environment")

    # Robot setup
    if robot == "g1":
        articulation_builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            limit_ke=0.0, limit_kd=0.0, friction=0.0
        )
        articulation_builder.default_shape_cfg.ke = 5.0e4
        articulation_builder.default_shape_cfg.kd = 5.0e2
        articulation_builder.default_shape_cfg.kf = 1.0e3
        articulation_builder.default_shape_cfg.mu = 0.75

        asset_path = newton.utils.download_asset("unitree_g1")
        articulation_builder.add_usd(
            str(asset_path / "usd" / "g1_isaac.usd"),
            xform=wp.transform(wp.vec3(0, 0, 0.8)),
            collapse_fixed_joints=True,
            enable_self_collisions=False,
            hide_collision_shapes=True,
            skip_mesh_approximation=True,
        )
        for i in range(6, articulation_builder.joint_dof_count):
            articulation_builder.joint_target_ke[i] = 1000.0
            articulation_builder.joint_target_kd[i] = 5.0
        articulation_builder.approximate_meshes("bounding_box")

    elif robot == "h1":
        articulation_builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5
        )
        articulation_builder.default_shape_cfg.ke = 5.0e4
        articulation_builder.default_shape_cfg.kd = 5.0e2
        articulation_builder.default_shape_cfg.kf = 1.0e3
        articulation_builder.default_shape_cfg.mu = 0.75

        asset_path = newton.utils.download_asset("unitree_h1")
        asset_file = str(asset_path / "usd" / "h1_minimal.usda")
        articulation_builder.add_usd(
            asset_file,
            ignore_paths=["/GroundPlane"],
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            hide_collision_shapes=True,
        )
        articulation_builder.approximate_meshes("bounding_box")
        for i in range(articulation_builder.joint_dof_count):
            articulation_builder.joint_target_ke[i] = 150
            articulation_builder.joint_target_kd[i] = 5

    # Tabletop: H1 + free objects as separate articulations (per-world setup)
    # Objects match tabletop.xml from ASV benchmark
    if env == "tabletop":
        tabletop_objects = [
            # 4 spheres
            {"type": "sphere", "pos": (0.6, -0.1, 1.06), "radius": 0.05},
            {"type": "sphere", "pos": (0.675, -0.025, 1.06), "radius": 0.05},
            {"type": "sphere", "pos": (0.75, 0.05, 1.06), "radius": 0.05},
            {"type": "sphere", "pos": (0.825, 0.125, 1.06), "radius": 0.05},
            # 2 capsules
            {"type": "capsule", "pos": (0.6, -0.15, 1.16), "quat": (1, 0.5, 0, 0), "radius": 0.04, "half_height": 0.04},
            {"type": "capsule", "pos": (0.8, -0.15, 1.16), "quat": (1, 0.5, 0, 0), "radius": 0.04, "half_height": 0.04},
            # 4 spheres (original MJCF has ellipsoids, but BOX-ELLIPSOID collision unsupported)
            {"type": "sphere", "pos": (0.7, -0.15, 1.41), "radius": 0.04},
            {"type": "sphere", "pos": (0.7, -0.05, 1.41), "radius": 0.04},
            {"type": "sphere", "pos": (0.7, 0.05, 1.41), "radius": 0.04},
            {"type": "sphere", "pos": (0.7, 0.15, 1.41), "radius": 0.04},
            # 2 capsules (original MJCF has cylinders, but BOX-CYLINDER collision unsupported)
            {"type": "capsule", "pos": (0.55, 0, 1.2), "quat": (0, 0.5, 0, 1), "radius": 0.04, "half_height": 0.04},
            {"type": "capsule", "pos": (0.85, 0, 1.2), "quat": (0, 0.5, 0, 1), "radius": 0.04, "half_height": 0.04},
            # 6 boxes (two stacks of 3)
            {"type": "box", "pos": (0.65, 0, 1.19), "half_size": 0.03},
            {"type": "box", "pos": (0.65, 0, 1.25), "half_size": 0.03},
            {"type": "box", "pos": (0.65, 0, 1.31), "half_size": 0.03},
            {"type": "box", "pos": (0.75, 0, 1.19), "half_size": 0.03},
            {"type": "box", "pos": (0.75, 0, 1.25), "half_size": 0.03},
            {"type": "box", "pos": (0.75, 0, 1.31), "half_size": 0.03},
        ]

        obj_shape_cfg = newton.ModelBuilder.ShapeConfig(mu=0.5)

        builder = newton.ModelBuilder()
        for _ in range(args.num_worlds):
            builder.begin_world()
            builder.add_builder(articulation_builder)

            for obj in tabletop_objects:
                pos = obj["pos"]
                quat = obj.get("quat", (1, 0, 0, 0))
                qlen = (quat[0]**2 + quat[1]**2 + quat[2]**2 + quat[3]**2) ** 0.5
                quat = (quat[0]/qlen, quat[1]/qlen, quat[2]/qlen, quat[3]/qlen)

                body_idx = builder.add_body(
                    xform=wp.transform(wp.vec3(pos[0], pos[1], pos[2]), wp.quat(quat[0], quat[1], quat[2], quat[3])),
                )

                if obj["type"] == "sphere":
                    builder.add_shape_sphere(body_idx, radius=obj["radius"], cfg=obj_shape_cfg)
                elif obj["type"] == "box":
                    hs = obj["half_size"]
                    builder.add_shape_box(body_idx, hx=hs, hy=hs, hz=hs, cfg=obj_shape_cfg)
                elif obj["type"] == "capsule":
                    builder.add_shape_capsule(body_idx, radius=obj["radius"], half_height=obj["half_height"], cfg=obj_shape_cfg)
                elif obj["type"] == "ellipsoid":
                    builder.add_shape_ellipsoid(body_idx, a=obj["a"], b=obj["b"], c=obj["c"], cfg=obj_shape_cfg)
                elif obj["type"] == "cylinder":
                    builder.add_shape_cylinder(body_idx, radius=obj["radius"], half_height=obj["half_height"], cfg=obj_shape_cfg)

            builder.end_world()

        # Static table and container walls (body=-1 for static geometry)
        table_shape_cfg = newton.ModelBuilder.ShapeConfig(mu=0.5)
        builder.add_shape_box(-1, xform=wp.transform((0.8, 0.0, 0.75), wp.quat_identity()), hx=0.5, hy=1.0, hz=0.01, cfg=table_shape_cfg)
        builder.add_shape_box(-1, xform=wp.transform((0.9, 0.0, 0.86), wp.quat_identity()), hx=0.01, hy=0.21, hz=0.1, cfg=table_shape_cfg)
        builder.add_shape_box(-1, xform=wp.transform((0.5, 0.0, 0.86), wp.quat_identity()), hx=0.01, hy=0.21, hz=0.1, cfg=table_shape_cfg)
        builder.add_shape_box(-1, xform=wp.transform((0.7, -0.2, 0.86), wp.quat_identity()), hx=0.21, hy=0.01, hz=0.1, cfg=table_shape_cfg)
        builder.add_shape_box(-1, xform=wp.transform((0.7, 0.2, 0.86), wp.quat_identity()), hx=0.21, hy=0.01, hz=0.1, cfg=table_shape_cfg)

        builder.add_ground_plane()
        model = builder.finalize()

    # Cube stack: G1 on stacked cubes as separate articulations (per-world setup)
    elif env == "cube_stack":
        cube_shape_cfg = newton.ModelBuilder.ShapeConfig(mu=0.3)
        cube_configs = [
            {"half_size": 0.35, "mass": 40.0, "offset": (0.0, 0.0), "rot_deg": 0.0},
            {"half_size": 0.35, "mass": 40.0, "offset": (0.0, 0.0), "rot_deg": 15.0},
            {"half_size": 0.35, "mass": 40.0, "offset": (0.0, 0.0), "rot_deg": -12.0},
        ]
        G1_HEIGHT_OFFSET = 0.92

        builder = newton.ModelBuilder()
        for _ in range(args.num_worlds):
            builder.begin_world()

            # Stack cubes
            current_z = 0.0
            for cube_cfg in cube_configs:
                half_size = cube_cfg["half_size"]
                cube_z = current_z + half_size
                rot_rad = float(np.radians(cube_cfg["rot_deg"]))
                quat = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), rot_rad)
                body_idx = builder.add_body(
                    xform=wp.transform(wp.vec3(cube_cfg["offset"][0], cube_cfg["offset"][1], cube_z), quat),
                    mass=cube_cfg["mass"],
                )
                builder.add_shape_box(body_idx, hx=half_size, hy=half_size, hz=half_size, cfg=cube_shape_cfg)
                current_z += 2.0 * half_size

            # Add G1 on top of cube stack
            platform_height = current_z
            g1_xform = wp.transform(wp.vec3(0.0, 0.0, platform_height + G1_HEIGHT_OFFSET))
            builder.add_builder(articulation_builder, xform=g1_xform)

            builder.end_world()

        builder.add_ground_plane()
        model = builder.finalize()
    else:
        # Standard replicate for non-cube_stack environments
        builder = newton.ModelBuilder()
        builder.replicate(articulation_builder, args.num_worlds)
        builder.add_ground_plane()
        model = builder.finalize()
    
    model.shape_contact_margin.fill_(0.001)

    return model


def create_solver(model, args, scenario_cfg: dict):
    """Create the appropriate solver based on args.

    Returns:
        A solver instance (SolverMuJoCo or SolverFeatherPGS).
    """
    import newton

    # Resolve solver preset
    preset = SOLVER_PRESETS.get(args.solver, {})
    solver_type = preset.get("type", args.solver)

    if solver_type == "mujoco":
        mj_cfg = scenario_cfg.get("mujoco_settings", {})
        return newton.solvers.SolverMuJoCo(
            model,
            use_mujoco_cpu=False,
            solver=args.mj_solver or mj_cfg.get("solver", "newton"),
            integrator=args.mj_integrator or mj_cfg.get("integrator", "implicitfast"),
            njmax=args.mj_njmax or mj_cfg.get("njmax", 100),
            nconmax=args.mj_nconmax or mj_cfg.get("nconmax", 20),
            ls_parallel=mj_cfg.get("ls_parallel", True),
            iterations=100,
            ls_iterations=50,
        )
    else:
        # Priority: preset > scenario default > CLI arg
        # (--override-scenario-defaults skips scenario defaults, used by ablation subprocesses)
        def get_kernel(name, cli_val):
            if name in preset:
                return preset[name]
            if not args.override_scenario_defaults:
                scenario_key = f"default_{name}"
                if scenario_key in scenario_cfg:
                    return scenario_cfg[scenario_key]
            return cli_val

        cholesky = get_kernel("cholesky_kernel", args.cholesky_kernel)
        trisolve = get_kernel("trisolve_kernel", args.trisolve_kernel)
        hinv_jt = get_kernel("hinv_jt_kernel", args.hinv_jt_kernel)
        delassus = get_kernel("delassus_kernel", args.delassus_kernel)
        pgs = get_kernel("pgs_kernel", args.pgs_kernel)
        delassus_chunk = get_kernel("delassus_chunk_size", args.delassus_chunk_size)
        pgs_chunk = get_kernel("pgs_chunk_size", args.pgs_chunk_size)
        parallel_streams = preset.get("use_parallel_streams", args.use_parallel_streams)

        solver_kwargs = {
            "update_mass_matrix_interval": 1,
            "pgs_iterations": args.pgs_iterations,
            "pgs_beta": args.pgs_beta,
            "pgs_cfm": args.pgs_cfm,
            "pgs_omega": args.pgs_omega,
            "pgs_max_constraints": args.pgs_max_constraints,
            "pgs_warmstart": args.pgs_warmstart,
            "enable_contact_friction": True,
            "storage": args.storage or scenario_cfg.get("storage", "batched"),
            "cholesky_kernel": cholesky,
            "trisolve_kernel": trisolve,
            "hinv_jt_kernel": hinv_jt,
            "delassus_kernel": delassus,
            "pgs_kernel": pgs,
            "delassus_chunk_size": delassus_chunk,
            "pgs_chunk_size": pgs_chunk,
            "small_dof_threshold": 12,
            "use_parallel_streams": parallel_streams,
        }
        return newton.solvers.SolverFeatherPGS(model, **solver_kwargs)


# =============================================================================
# Direct Run Mode (called by subprocesses)
# =============================================================================

def run_direct(args):
    """Run a single benchmark directly (not as subprocess)."""
    import warp as wp
    wp.config.enable_backward = False

    scenario_cfg = SCENARIOS[args.scenario]

    model = build_model(args, scenario_cfg)
    solver = create_solver(model, args, scenario_cfg)

    # States
    fps = 60.0
    frame_dt = 1.0 / fps
    sim_dt = frame_dt / float(args.substeps)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.collide(state_0)

    # CUDA graph capture (if not timing)
    # Handle odd substeps: with CUDA graphs, we need state_0 to end up in the
    # same buffer it started in. With odd substeps, normal swapping leaves it
    # in the wrong buffer. Fix: copy on the last iteration instead of swap.
    graph = None
    need_odd_substep_fix = args.substeps % 2 == 1

    if not args.summary_timer:
        device = wp.get_device()
        if device.is_cuda:
            def simulate():
                for i in range(args.substeps):
                    nonlocal contacts, state_0, state_1
                    contacts = model.collide(state_0)
                    state_0.clear_forces()
                    solver.step(state_0, state_1, control, contacts, sim_dt)
                    # Handle odd substeps for CUDA graph compatibility
                    if need_odd_substep_fix and i == args.substeps - 1:
                        state_0.assign(state_1)
                    else:
                        state_0, state_1 = state_1, state_0

            with wp.ScopedCapture() as capture:
                simulate()
            graph = capture.graph

    def step():
        nonlocal contacts, state_0, state_1
        if graph is not None:
            wp.capture_launch(graph)
        else:
            for i in range(args.substeps):
                contacts = model.collide(state_0)
                state_0.clear_forces()
                solver.step(state_0, state_1, control, contacts, sim_dt)
                # Handle odd substeps consistently (even without graph)
                if need_odd_substep_fix and i == args.substeps - 1:
                    state_0.assign(state_1)
                else:
                    state_0, state_1 = state_1, state_0

    # Warmup
    for _ in range(args.warmup_frames):
        step()
    wp.synchronize_device()

    # Benchmark
    total_env_frames = args.num_worlds * args.measure_frames

    def run_benchmark():
        t_start = time.time()
        for _ in range(args.measure_frames):
            step()
        wp.synchronize_device()
        return time.time() - t_start

    if args.summary_timer:
        with wp.ScopedTimer(
            "benchmark",
            cuda_filter=wp.TIMING_ALL,
            synchronize=True,
            report_func=print_kernel_summary,
        ):
            elapsed = run_benchmark()
    else:
        elapsed = run_benchmark()

    fps_env = total_env_frames / elapsed if elapsed > 0 else 0

    # GPU memory
    gpu_used_gb = None
    gpu_total_gb = None
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            gpu_used_gb = (total - free) / 1024**3
            gpu_total_gb = total / 1024**3
    except ImportError:
        pass

    # Print summary
    solver_name = args.solver if args.solver == "mujoco" else "feather_pgs"
    print("\n=== Benchmark Summary ===")
    print(f"Solver:              {solver_name}")
    print(f"Worlds:              {args.num_worlds}")
    print(f"Sim substeps/frame:  {args.substeps}")
    print(f"Warmup frames:       {args.warmup_frames}")
    print(f"Measured frames:     {args.measure_frames}")
    print(f"Total env-frames:    {total_env_frames}")
    print(f"Elapsed time (s):    {elapsed:.6f}")
    print(f"Env-FPS (env/s):     {fps_env:,.2f}")
    if gpu_used_gb is not None:
        print(f"GPU memory used (GB):   {gpu_used_gb:.3f}")
        print(f"GPU memory total (GB):  {gpu_total_gb:.3f}")
    print("=========================\n")


def print_kernel_summary(results, indent: str = ""):
    """Print kernel timing summary."""
    import string
    kernel_results = [r for r in results if r.name.startswith(("forward kernel", "backward kernel"))]
    if not kernel_results:
        print(f"{indent}No kernel activity recorded.")
        return

    def normalize_kernel_name(name: str) -> str:
        if " kernel " in name:
            _, rest = name.split(" kernel ", 1)
        else:
            rest = name
        parts = rest.split("_")
        if len(parts) > 1 and len(parts[-1]) == 8 and all(c in string.hexdigits for c in parts[-1]):
            parts = parts[:-1]
        return "_".join(parts)

    totals = defaultdict(float)
    for r in kernel_results:
        totals[normalize_kernel_name(r.name)] += r.elapsed

    total_time = sum(totals.values())
    if total_time <= 0:
        print(f"{indent}No kernel time recorded.")
        return

    sorted_items = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)

    rows = []
    cumulative = 0.0
    for name, t in sorted_items[:15]:
        pct = (t / total_time) * 100
        cumulative += pct
        rows.append((name, t, pct, cumulative))

    name_width = min(max(len(name) for name, *_ in rows), 45)
    print(f"\n{indent}{'Kernel':<{name_width}}  {'Time (ms)':>10}  {'% total':>8}  {'Cumul.':>8}")
    print(f"{indent}{'-' * (name_width + 32)}")
    for name, t, pct, cum in rows:
        label = name if len(name) <= name_width else name[:name_width - 3] + "..."
        print(f"{indent}{label:<{name_width}}  {round(t):>10}  {pct:>7.1f}%  {cum:>7.1f}%")


# =============================================================================
# Interactive Mode
# =============================================================================

def run_interactive(args):
    """Run interactive mode with viewer."""
    import warp as wp
    wp.config.enable_backward = False

    import newton.viewer

    scenario_cfg = SCENARIOS[args.scenario]
    print(f"Interactive mode: {scenario_cfg['description']}")
    print(f"Solver: {args.solver}")

    # Create viewer
    viewer = newton.viewer.ViewerGL()

    # Build model and solver using shared functions
    model = build_model(args, scenario_cfg)
    solver = create_solver(model, args, scenario_cfg)

    # States
    fps = 60.0
    frame_dt = 1.0 / fps
    sim_dt = frame_dt / float(args.substeps)
    sim_time = 0.0

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.collide(state_0)

    # Set up viewer
    viewer.set_model(model)

    print("Starting interactive simulation. Close window to exit.")

    # Main loop
    while viewer.is_running():
        if not viewer.is_paused():
            for _ in range(args.substeps):
                contacts = model.collide(state_0)
                state_0.clear_forces()
                viewer.apply_forces(state_0)
                solver.step(state_0, state_1, control, contacts, sim_dt)
                state_0, state_1 = state_1, state_0
            sim_time += frame_dt

        # Render
        viewer.begin_frame(sim_time)
        viewer.log_state(state_0)
        viewer.log_contacts(contacts, state_0)
        viewer.end_frame()

    viewer.close()


# =============================================================================
# Replot Mode
# =============================================================================

def run_replot(results_dir: Path):
    """Regenerate plot from a previous results directory."""
    metadata_path = results_dir / "metadata.json"
    if not metadata_path.exists():
        print(f"Error: no metadata.json found in {results_dir}")
        return

    with open(metadata_path) as f:
        metadata = json.load(f)

    scenario = metadata.get("scenario", "unknown")
    gpu_name = metadata.get("gpu", "")

    # Detect sweep vs ablation from JSONL files
    sweep_jsonl = results_dir / "sweep.jsonl"
    ablation_jsonl = results_dir / "ablation.jsonl"

    if sweep_jsonl.exists():
        out_path = results_dir / "sweep.png"
        plot_sweep(sweep_jsonl, out_path, scenario, gpu_name)
    elif ablation_jsonl.exists():
        results = _load_jsonl(ablation_jsonl)
        num_worlds = metadata.get("num_worlds", 0)
        out_path = results_dir / "ablation.png"
        plot_ablation(results, out_path, scenario, num_worlds, gpu_name)
    else:
        print(f"Error: no sweep.jsonl or ablation.jsonl found in {results_dir}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified solver benchmark for Newton",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Mode selection
    parser.add_argument("--benchmark", action="store_true", help="Run single headless benchmark")
    parser.add_argument("--sweep", action="store_true", help="Sweep over num_worlds")
    parser.add_argument("--ablation", action="store_true", help="Run ablation study")
    parser.add_argument("--compare", nargs="+", metavar="FILE", help="Compare kernel timing files")
    parser.add_argument("--replot", type=str, metavar="DIR", help="Regenerate plot from a previous results directory")

    # Scenario
    parser.add_argument("--scenario", type=str, choices=list(SCENARIOS.keys()),
                        default="g1_flat", help="Scenario to run")

    # Common options
    parser.add_argument("--num-worlds", type=int, default=4096, help="Number of parallel worlds")
    parser.add_argument("--substeps", type=int, default=None, help="Sim substeps per frame (default: from scenario)")
    parser.add_argument("--warmup-frames", type=int, default=16, help="Warmup frames")
    parser.add_argument("--measure-frames", type=int, default=64, help="Measurement frames")
    parser.add_argument("--viewer", type=str, default="gl", choices=["gl", "null"], help="Viewer type")
    parser.add_argument("--out", type=str, help="Output directory")
    parser.add_argument("--plot", action="store_true", help="Generate plots")

    # Solver selection
    parser.add_argument("--solver", type=str, default="feather_pgs",
                        help="Solver: mujoco, feather_pgs, or preset (fpgs_tiled, fpgs_loop)")
    parser.add_argument("--solvers", type=str, default="fpgs_tiled,fpgs_loop,mujoco",
                        help="Comma-separated solvers for sweep mode")

    # FeatherPGS options
    parser.add_argument("--storage", type=str, choices=["flat", "batched"], help="Storage mode")
    parser.add_argument("--cholesky-kernel", type=str, default="auto",
                        choices=["loop", "tiled", "auto"], help="Cholesky kernel")
    parser.add_argument("--trisolve-kernel", type=str, default="auto",
                        choices=["loop", "tiled", "auto"], help="Trisolve kernel")
    parser.add_argument("--hinv-jt-kernel", type=str, default="auto",
                        choices=["par_row", "tiled", "auto"], help="H^-1 J^T kernel")
    parser.add_argument("--delassus-kernel", type=str, default="auto",
                        choices=["par_row_col", "tiled", "auto"], help="Delassus kernel")
    parser.add_argument("--pgs-kernel", type=str, default="tiled_contact",
                        choices=["loop", "tiled_row", "tiled_contact", "streaming"], help="PGS kernel")
    parser.add_argument("--pgs-iterations", type=int, default=8, help="PGS iterations")
    parser.add_argument("--pgs-max-constraints", type=int, default=64, help="Max constraints per world")
    parser.add_argument("--pgs-beta", type=float, default=0.1, help="PGS position correction factor (ERP)")
    parser.add_argument("--pgs-cfm", type=float, default=1.0e-6, help="PGS constraint force mixing (regularization)")
    parser.add_argument("--pgs-omega", type=float, default=1.0, help="PGS relaxation factor (SOR)")
    parser.add_argument("--pgs-warmstart", action="store_true", help="Enable warmstart")
    parser.add_argument("--delassus-chunk-size", type=int, default=None,
                        help="Chunk size (constraint rows) for streaming Delassus kernel (None=auto)")
    parser.add_argument("--pgs-chunk-size", type=int, default=None,
                        help="Chunk size (contacts) for streaming PGS kernel (None=1)")
    parser.add_argument("--use-parallel-streams", action="store_true", dest="use_parallel_streams")
    parser.add_argument("--no-parallel-streams", action="store_false", dest="use_parallel_streams")
    parser.set_defaults(use_parallel_streams=True)
    parser.add_argument("--override-scenario-defaults", action="store_true",
                        help="CLI kernel args override scenario defaults (used by ablation)")

    # MuJoCo options
    parser.add_argument("--mj-solver", type=str, choices=["cg", "newton"])
    parser.add_argument("--mj-integrator", type=str, choices=["euler", "rk4", "implicit", "implicitfast"])
    parser.add_argument("--mj-njmax", type=int)
    parser.add_argument("--mj-nconmax", type=int)

    # Sweep options
    parser.add_argument("--min-log2-worlds", type=int, default=10, help="Min worlds = 2^N")
    parser.add_argument("--max-log2-worlds", type=int, default=14, help="Max worlds = 2^N")
    parser.add_argument("--substeps-list", type=str, default=None,
                        help="Comma-separated substeps to sweep (e.g., '2,4'). Overrides --substeps in sweep mode.")

    # Ablation options
    parser.add_argument("--ablation-pgs", type=str, default="auto",
                        choices=["auto", "tiled_row", "tiled_contact"],
                        help="PGS kernel for ablation final steps")

    # Timing
    parser.add_argument("--timing-out", type=str, help="Output kernel timing to JSON")
    parser.add_argument("--summary-timer", action="store_true", help="Print kernel summary")

    args = parser.parse_args()

    # Apply scenario defaults for unspecified options
    if args.scenario in SCENARIOS:
        scenario_cfg = SCENARIOS[args.scenario]
        if args.substeps is None:
            args.substeps = scenario_cfg.get("default_substeps", 4)
        if args.pgs_iterations == 8:  # default value, check if scenario has different
            args.pgs_iterations = scenario_cfg.get("default_pgs_iterations", args.pgs_iterations)
        if args.pgs_max_constraints == 64:  # default value
            args.pgs_max_constraints = scenario_cfg.get("default_pgs_max_constraints", args.pgs_max_constraints)

    # Route to appropriate handler
    if args.replot:
        run_replot(Path(args.replot))
        return
    elif args.compare:
        run_compare(args)
    elif args.sweep:
        run_sweep(args)
    elif args.ablation:
        run_ablation(args)
    elif args.benchmark:
        run_direct(args)
    else:
        # Interactive mode with viewer
        run_interactive(args)


if __name__ == "__main__":
    main()
