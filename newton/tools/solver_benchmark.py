# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Concrete benchmark and render worker used by nightly orchestration.

Usage examples:

    # Interactive run
    uv run newton/tools/solver_benchmark.py --scenario g1_flat --num-worlds 1

    # Single benchmark run with job-owned artifacts
    uv run newton/tools/solver_benchmark.py --scenario h1_tabletop --solver fpgs_tiled \\
        --num-worlds 4096 --benchmark --summary-timer --out results/job_0001

    # Render a short video of a scenario (headless, outputs mp4 via ffmpeg)
    uv run newton/tools/solver_benchmark.py --scenario h1_tabletop --solver fpgs_tiled \\
        --render --render-frames 300 --out renders/job_0002
"""

import argparse
import datetime as dt
import json
import platform
import string
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

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
        "default_dense_max_constraints": 32,
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
        "default_dense_max_constraints": 128,
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
        "default_dense_max_constraints": 32,
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
        "default_dense_max_constraints": 128,
        # With MF enabled, rigid body contacts are routed to the matrix-free path,
        # leaving only ~90 dense robot constraints. 128 max_constraints suffices and
        # fits entirely in shared memory for tiled kernels.
        "default_cholesky_kernel": "tiled",
        "default_trisolve_kernel": "tiled",
        "default_hinv_jt_kernel": "tiled",
        "default_delassus_kernel": "tiled",
        "default_pgs_kernel": "tiled_contact",
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
    "fpgs_mf": {
        "type": "feather_pgs",
        "storage": "batched",
        "pgs_mode": "velocity",
        "cholesky_kernel": "tiled",
        "trisolve_kernel": "tiled",
        "hinv_jt_kernel": "tiled",
        "delassus_kernel": "tiled",
        "pgs_kernel": "tiled_contact",
        "dense_max_constraints": 128,
        "use_parallel_streams": True,
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
    "smoke": [
        {
            "label": "FeatherPGS baseline",
            "cholesky_kernel": "loop",
            "trisolve_kernel": "loop",
            "hinv_jt_kernel": "par_row",
            "delassus_kernel": "par_row_col",
            "pgs_kernel": "loop",
            "use_parallel_streams": False,
        },
    ],
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
            "pgs_kernel": "tiled_contact",
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
        {
            "label": "+ full MF GS",
            "storage": "batched",
            "cholesky_kernel": "tiled",
            "trisolve_kernel": "tiled",
            "hinv_jt_kernel": "tiled",
            "delassus_kernel": "tiled",
            "pgs_kernel": "tiled_contact",
            "dense_max_constraints": 128,
            "pgs_mode": "velocity",
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
            "dense_max_constraints": 396,
            "pgs_mode": "delassus",
            "use_parallel_streams": False,
        },
        {
            "label": "+ tiled cholesky",
            "cholesky_kernel": "tiled",
            "trisolve_kernel": "loop",
            "hinv_jt_kernel": "par_row",
            "delassus_kernel": "par_row_col",
            "pgs_kernel": "loop",
            "dense_max_constraints": 396,
            "pgs_mode": "delassus",
            "use_parallel_streams": False,
        },
        {
            "label": "+ tiled trisolve",
            "cholesky_kernel": "tiled",
            "trisolve_kernel": "tiled",
            "hinv_jt_kernel": "par_row",
            "delassus_kernel": "par_row_col",
            "pgs_kernel": "loop",
            "dense_max_constraints": 396,
            "pgs_mode": "delassus",
            "use_parallel_streams": False,
        },
        {
            "label": "+ tiled delassus",
            "cholesky_kernel": "tiled",
            "trisolve_kernel": "tiled",
            "hinv_jt_kernel": "par_row",
            "delassus_kernel": "tiled",
            "pgs_kernel": "loop",
            "dense_max_constraints": 396,
            "pgs_mode": "delassus",
            "use_parallel_streams": False,
        },
        {
            "label": "+ streaming PGS",
            "cholesky_kernel": "tiled",
            "trisolve_kernel": "tiled",
            "hinv_jt_kernel": "par_row",
            "delassus_kernel": "tiled",
            "pgs_kernel": "streaming",
            "dense_max_constraints": 396,
            "pgs_mode": "delassus",
            "use_parallel_streams": False,
        },
        {
            "label": "+ parallel streams",
            "cholesky_kernel": "tiled",
            "trisolve_kernel": "tiled",
            "hinv_jt_kernel": "par_row",
            "delassus_kernel": "tiled",
            "pgs_kernel": "streaming",
            "dense_max_constraints": 396,
            "pgs_mode": "delassus",
            "use_parallel_streams": True,
        },
        # --- MF comparison configs ---
        {
            "label": "hybrid (dense + MF rigid)",
            "cholesky_kernel": "tiled",
            "trisolve_kernel": "tiled",
            "hinv_jt_kernel": "tiled",
            "delassus_kernel": "tiled",
            "pgs_kernel": "tiled_contact",
            "dense_max_constraints": 128,
            "pgs_mode": "hybrid",
            "use_parallel_streams": True,
        },
        {
            "label": "fully matrix-free GS",
            "cholesky_kernel": "tiled",
            "trisolve_kernel": "tiled",
            "hinv_jt_kernel": "tiled",
            "delassus_kernel": "tiled",
            "pgs_kernel": "tiled_contact",
            "dense_max_constraints": 128,
            "pgs_mode": "velocity",
            "use_parallel_streams": True,
        },
    ],
}

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
        result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except Exception:
        return "unknown"


def collect_metadata(args, scenario_cfg: dict) -> dict:
    """Collect run metadata."""
    gpu_name, gpu_total = get_gpu_info()
    return {
        "scenario": args.scenario,
        "scenario_description": scenario_cfg["description"],
        "solver": args.solver,
        "num_worlds": args.num_worlds,
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
        "viewer": args.viewer,
    }


def build_measurement_row(
    *,
    args,
    elapsed_s: float,
    env_fps: float,
    gpu_used_gb: float | None,
    gpu_total_gb: float | None,
    kernels: dict[str, float] | None = None,
) -> dict[str, object]:
    """Build the structured measurement row for one benchmark job."""
    return {
        "solver": args.solver,
        "substeps": args.substeps,
        "num_worlds": args.num_worlds,
        "env_fps": env_fps,
        "gpu_used_gb": gpu_used_gb,
        "gpu_total_gb": gpu_total_gb,
        "elapsed_s": elapsed_s,
        "ok": True,
        "kernels": kernels or {},
    }


def write_benchmark_artifacts(out_dir: Path, measurement: dict[str, object], metadata: dict[str, object]) -> None:
    """Write job-owned benchmark artifacts to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "measurements.jsonl").write_text(json.dumps(measurement) + "\n", encoding="utf-8")
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def build_render_metadata(args, *, render_fps: float, width: int, height: int, video_name: str) -> dict[str, object]:
    """Build the structured render metadata for one render job."""
    solver_name = args.solver if args.solver == "mujoco" else "feather_pgs"
    return {
        "scenario": args.scenario,
        "solver": args.solver,
        "substeps": args.substeps,
        "num_worlds": 1,
        "fps": int(render_fps),
        "frames": args.render_frames,
        "width": width,
        "height": height,
        "video": video_name,
        "label": f"{SOLVER_LABEL_MAP.get(args.solver, solver_name)}, {args.substeps} substeps",
    }


def write_render_artifacts(
    out_dir: Path,
    metadata: dict[str, object],
    render_metadata: dict[str, object],
) -> None:
    """Write job-owned render artifacts to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (out_dir / "render_meta.json").write_text(json.dumps(render_metadata, indent=2) + "\n", encoding="utf-8")


# =============================================================================
# Subprocess Runner
# =============================================================================


def build_run_command(args, solver_config: dict, num_worlds: int, substeps: int | None = None) -> list[str]:
    """Build command line for a single benchmark run."""
    scenario_cfg = SCENARIOS[args.scenario]
    substeps = substeps if substeps is not None else args.substeps

    cmd = [
        sys.executable,
        "-m",
        "newton.tools.solver_benchmark",
        "--scenario",
        args.scenario,
        "--num-worlds",
        str(num_worlds),
        "--substeps",
        str(substeps),
        "--warmup-frames",
        str(args.warmup_frames),
        "--measure-frames",
        str(args.measure_frames),
        "--benchmark",
        "--viewer",
        "null",
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
        pgs_max_c = solver_config.get("dense_max_constraints", args.dense_max_constraints)
        cmd.extend(["--dense-max-constraints", str(pgs_max_c)])
        cmd.extend(["--pgs-beta", str(args.pgs_beta)])
        cmd.extend(["--pgs-cfm", str(args.pgs_cfm)])
        cmd.extend(["--pgs-omega", str(args.pgs_omega)])
        if args.pgs_warmstart:
            cmd.append("--pgs-warmstart")
        pgs_mode = solver_config.get("pgs_mode", args.pgs_mode)
        if pgs_mode != "hybrid":
            cmd.extend(["--pgs-mode", pgs_mode])

        # Chunk sizes: solver_config overrides CLI
        delassus_chunk = solver_config.get("delassus_chunk_size", getattr(args, "delassus_chunk_size", None))
        if delassus_chunk is not None:
            cmd.extend(["--delassus-chunk-size", str(delassus_chunk)])
        pgs_chunk = solver_config.get("pgs_chunk_size", getattr(args, "pgs_chunk_size", None))
        if pgs_chunk is not None:
            cmd.extend(["--pgs-chunk-size", str(pgs_chunk)])

    if getattr(args, "summary_timer", False):
        cmd.append("--summary-timer")

    return cmd


# =============================================================================
# Model and Solver Creation (shared between run modes)
# =============================================================================


def build_model(args, scenario_cfg: dict):
    """Build and finalize the simulation model for a scenario.

    Returns:
        The finalized newton.Model.
    """
    import warp as wp  # noqa: PLC0415

    import newton  # noqa: PLC0415
    import newton.examples  # noqa: PLC0415
    import newton.utils  # noqa: PLC0415

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
                qlen = (quat[0] ** 2 + quat[1] ** 2 + quat[2] ** 2 + quat[3] ** 2) ** 0.5
                quat = (quat[0] / qlen, quat[1] / qlen, quat[2] / qlen, quat[3] / qlen)

                body_idx = builder.add_body(
                    xform=wp.transform(wp.vec3(pos[0], pos[1], pos[2]), wp.quat(quat[0], quat[1], quat[2], quat[3])),
                )

                if obj["type"] == "sphere":
                    builder.add_shape_sphere(body_idx, radius=obj["radius"], cfg=obj_shape_cfg)
                elif obj["type"] == "box":
                    hs = obj["half_size"]
                    builder.add_shape_box(body_idx, hx=hs, hy=hs, hz=hs, cfg=obj_shape_cfg)
                elif obj["type"] == "capsule":
                    builder.add_shape_capsule(
                        body_idx, radius=obj["radius"], half_height=obj["half_height"], cfg=obj_shape_cfg
                    )
                elif obj["type"] == "ellipsoid":
                    builder.add_shape_ellipsoid(body_idx, a=obj["a"], b=obj["b"], c=obj["c"], cfg=obj_shape_cfg)
                elif obj["type"] == "cylinder":
                    builder.add_shape_cylinder(
                        body_idx, radius=obj["radius"], half_height=obj["half_height"], cfg=obj_shape_cfg
                    )

            builder.end_world()

        # Static table and container walls (body=-1 for static geometry)
        table_shape_cfg = newton.ModelBuilder.ShapeConfig(mu=0.5)
        builder.add_shape_box(
            -1, xform=wp.transform((0.8, 0.0, 0.75), wp.quat_identity()), hx=0.5, hy=1.0, hz=0.01, cfg=table_shape_cfg
        )
        builder.add_shape_box(
            -1, xform=wp.transform((0.9, 0.0, 0.86), wp.quat_identity()), hx=0.01, hy=0.21, hz=0.1, cfg=table_shape_cfg
        )
        builder.add_shape_box(
            -1, xform=wp.transform((0.5, 0.0, 0.86), wp.quat_identity()), hx=0.01, hy=0.21, hz=0.1, cfg=table_shape_cfg
        )
        builder.add_shape_box(
            -1, xform=wp.transform((0.7, -0.2, 0.86), wp.quat_identity()), hx=0.21, hy=0.01, hz=0.1, cfg=table_shape_cfg
        )
        builder.add_shape_box(
            -1, xform=wp.transform((0.7, 0.2, 0.86), wp.quat_identity()), hx=0.21, hy=0.01, hz=0.1, cfg=table_shape_cfg
        )

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

    model.shape_margin.fill_(0.001)

    return model


def create_solver(model, args, scenario_cfg: dict):
    """Create the appropriate solver based on args.

    Returns:
        A solver instance (SolverMuJoCo or SolverFeatherPGS).
    """
    import newton  # noqa: PLC0415

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
        storage = preset.get("storage", args.storage or scenario_cfg.get("storage", "batched"))
        pgs_iterations = preset.get("pgs_iterations", args.pgs_iterations)
        dense_max_constraints = preset.get("dense_max_constraints", args.dense_max_constraints)
        pgs_mode = preset.get("pgs_mode", args.pgs_mode)

        solver_kwargs = {
            "update_mass_matrix_interval": 1,
            "pgs_iterations": pgs_iterations,
            "pgs_beta": args.pgs_beta,
            "pgs_cfm": args.pgs_cfm,
            "pgs_omega": args.pgs_omega,
            "dense_max_constraints": dense_max_constraints,
            "pgs_warmstart": args.pgs_warmstart,
            "pgs_mode": pgs_mode,
            "enable_contact_friction": True,
            "storage": storage,
            "cholesky_kernel": cholesky,
            "trisolve_kernel": trisolve,
            "hinv_jt_kernel": hinv_jt,
            "delassus_kernel": delassus,
            "pgs_kernel": pgs,
            "delassus_chunk_size": delassus_chunk,
            "pgs_chunk_size": pgs_chunk,
            "small_dof_threshold": 12,
            "use_parallel_streams": parallel_streams,
            "double_buffer": args.double_buffer,
            "nvtx": args.nvtx,
        }
        from newton._src.solvers import SolverFeatherPGS  # noqa: PLC0415

        return SolverFeatherPGS(model, **solver_kwargs)


# =============================================================================
# Direct Run Mode (called by subprocesses)
# =============================================================================


def run_direct(args):
    """Run a single benchmark directly (not as subprocess)."""
    import warp as wp  # noqa: PLC0415

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

    # Pipeline collide: overlap collision detection with solver stages S1-S3
    _pipeline_collide = getattr(args, "pipeline_collide", False)
    collide_stream = None
    contacts_bufs = None
    collide_done_events = None
    collide_state = None
    if _pipeline_collide:
        _pc_dev = wp.get_device()
        if _pc_dev.is_cuda:
            collide_stream = wp.Stream(_pc_dev)
            contacts_bufs = [contacts, model.contacts()]
            collide_done_events = [None, None]
            collide_state = model.state()
        else:
            _pipeline_collide = False

    # CUDA graph capture (if not timing)
    # Handle odd substeps: with CUDA graphs, we need state_0 to end up in the
    # same buffer it started in. With odd substeps, normal swapping leaves it
    # in the wrong buffer. Fix: copy on the last iteration instead of swap.
    graph = None
    need_odd_substep_fix = args.substeps % 2 == 1

    _nvtx = args.nvtx

    if not args.summary_timer and not args.no_graph:
        device = wp.get_device()
        if device.is_cuda:
            if _pipeline_collide:

                def simulate():
                    nonlocal state_0, state_1
                    for i in range(args.substeps):
                        buf = i % 2
                        next_buf = 1 - buf
                        if need_odd_substep_fix and i == args.substeps - 1:
                            next_buf = 0
                        with wp.ScopedTimer("ClearForces", print=False, use_nvtx=_nvtx, synchronize=_nvtx):
                            state_0.clear_forces()
                        solver.step(
                            state_0,
                            state_1,
                            control,
                            contacts_bufs[buf],
                            sim_dt,
                            collide_done_event=collide_done_events[buf],
                        )
                        wp.copy(collide_state.body_q, state_1.body_q)
                        integrate_done = wp.get_stream(device).record_event()
                        with wp.ScopedStream(collide_stream):
                            collide_stream.wait_event(integrate_done)
                            with wp.ScopedTimer("Collide", print=False, use_nvtx=_nvtx, synchronize=_nvtx):
                                model.collide(collide_state, contacts_bufs[next_buf])
                            collide_done_events[next_buf] = collide_stream.record_event()
                        if need_odd_substep_fix and i == args.substeps - 1:
                            state_0.assign(state_1)
                        else:
                            state_0, state_1 = state_1, state_0

            else:

                def simulate():
                    for i in range(args.substeps):
                        nonlocal contacts, state_0, state_1
                        with wp.ScopedTimer("Collide", print=False, use_nvtx=_nvtx, synchronize=_nvtx):
                            contacts = model.collide(state_0)
                        with wp.ScopedTimer("ClearForces", print=False, use_nvtx=_nvtx, synchronize=_nvtx):
                            state_0.clear_forces()
                        solver.step(state_0, state_1, control, contacts, sim_dt)
                        if need_odd_substep_fix and i == args.substeps - 1:
                            state_0.assign(state_1)
                        else:
                            state_0, state_1 = state_1, state_0

            with wp.ScopedCapture() as capture:
                if hasattr(solver, "seed_double_buffer_events"):
                    solver.seed_double_buffer_events()
                if _pipeline_collide:
                    collide_done_events[0] = wp.get_stream(device).record_event()
                simulate()
            graph = capture.graph

    # Seed pipeline collide events for non-graph path
    if _pipeline_collide and graph is None:
        collide_done_events[0] = wp.get_stream(wp.get_device()).record_event()

    def step():
        nonlocal contacts, state_0, state_1
        if graph is not None:
            wp.capture_launch(graph)
        elif _pipeline_collide:
            _dev = wp.get_device()
            for i in range(args.substeps):
                buf = i % 2
                next_buf = 1 - buf
                if need_odd_substep_fix and i == args.substeps - 1:
                    next_buf = 0
                with wp.ScopedTimer("ClearForces", print=False, use_nvtx=_nvtx, synchronize=_nvtx):
                    state_0.clear_forces()
                solver.step(
                    state_0, state_1, control, contacts_bufs[buf], sim_dt, collide_done_event=collide_done_events[buf]
                )
                wp.copy(collide_state.body_q, state_1.body_q)
                integrate_done = wp.get_stream(_dev).record_event()
                with wp.ScopedStream(collide_stream):
                    collide_stream.wait_event(integrate_done)
                    with wp.ScopedTimer("Collide", print=False, use_nvtx=_nvtx, synchronize=_nvtx):
                        model.collide(collide_state, contacts_bufs[next_buf])
                    collide_done_events[next_buf] = collide_stream.record_event()
                if need_odd_substep_fix and i == args.substeps - 1:
                    state_0.assign(state_1)
                else:
                    state_0, state_1 = state_1, state_0
        else:
            for i in range(args.substeps):
                with wp.ScopedTimer("Collide", print=False, use_nvtx=_nvtx, synchronize=_nvtx):
                    contacts = model.collide(state_0)
                with wp.ScopedTimer("ClearForces", print=False, use_nvtx=_nvtx, synchronize=_nvtx):
                    state_0.clear_forces()
                solver.step(state_0, state_1, control, contacts, sim_dt)
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

    kernel_timings = {}
    if args.summary_timer:

        def report_kernel_summary(results):
            nonlocal kernel_timings
            kernel_timings = print_kernel_summary(results)

        with wp.ScopedTimer(
            "benchmark",
            cuda_filter=wp.TIMING_ALL,
            synchronize=True,
            report_func=report_kernel_summary,
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

    measurement = build_measurement_row(
        args=args,
        elapsed_s=elapsed,
        env_fps=fps_env,
        gpu_used_gb=gpu_used_gb,
        gpu_total_gb=gpu_total_gb,
        kernels=kernel_timings,
    )
    if args.out:
        metadata = collect_metadata(args, scenario_cfg)
        write_benchmark_artifacts(Path(args.out), measurement, metadata)

    return measurement


def print_kernel_summary(results, indent: str = "") -> dict[str, float]:
    """Print kernel timing summary and return per-kernel totals in milliseconds."""
    kernel_results = [r for r in results if r.name.startswith(("forward kernel", "backward kernel"))]
    if not kernel_results:
        print(f"{indent}No kernel activity recorded.")
        return {}

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
        return {}

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
        label = name if len(name) <= name_width else name[: name_width - 3] + "..."
        print(f"{indent}{label:<{name_width}}  {round(t):>10}  {pct:>7.1f}%  {cum:>7.1f}%")

    return {name: round(time_ms, 6) for name, time_ms in totals.items()}


# =============================================================================
# Render Mode (headless video capture via ffmpeg)
# =============================================================================

# Camera presets per scenario: (pos, pitch, yaw)
SOLVER_LABEL_MAP = {
    "fpgs_tiled": "FeatherPGS (tiled)",
    "fpgs_tiled_row": "FeatherPGS (tiled_row)",
    "fpgs_loop": "FeatherPGS (loop)",
    "fpgs_streaming": "FeatherPGS (streaming)",
    "feather_pgs": "FeatherPGS",
    "mujoco": "MJWarp",
}

CAMERA_PRESETS = {
    "g1_flat": ((2.5, -1.0, 1.2), -15.0, 155.0),
    "g1_cube_stack": ((3.0, -1.5, 2.5), -20.0, 155.0),
    "h1_flat": ((2.5, -1.0, 1.2), -15.0, 155.0),
    "h1_tabletop": ((2.77, -0.83, 2.40), -30.3, -198.6),
}


def run_render(args):
    """Render a short video of the scenario running headlessly via ffmpeg."""
    import shutil  # noqa: PLC0415

    import warp as wp  # noqa: PLC0415

    wp.config.enable_backward = False

    import newton.viewer  # noqa: PLC0415

    if not shutil.which("ffmpeg"):
        print("Error: ffmpeg is required for --render but was not found on PATH")
        sys.exit(1)

    scenario_cfg = SCENARIOS[args.scenario]

    # Force 1 world for rendering
    saved_num_worlds = args.num_worlds
    args.num_worlds = 1

    model = build_model(args, scenario_cfg)
    solver = create_solver(model, args, scenario_cfg)
    args.num_worlds = saved_num_worlds

    # Create headless viewer
    width, height = args.render_width, args.render_height
    viewer = newton.viewer.ViewerGL(width=width, height=height, headless=True)
    viewer.set_model(model)

    # Set camera (CLI overrides take priority over presets)
    cam_pos, cam_pitch, cam_yaw = CAMERA_PRESETS.get(args.scenario, ((3.0, -1.0, 1.5), -15.0, 155.0))
    if args.camera_pos is not None:
        cam_pos = tuple(float(x) for x in args.camera_pos.split(","))
    if args.camera_pitch is not None:
        cam_pitch = args.camera_pitch
    if args.camera_yaw is not None:
        cam_yaw = args.camera_yaw
    viewer.set_camera(wp.vec3(*cam_pos), cam_pitch, cam_yaw)

    # Simulation setup
    render_fps = 60.0
    frame_dt = 1.0 / render_fps
    sim_dt = frame_dt / float(args.substeps)
    sim_time = 0.0

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.collide(state_0)

    num_frames = args.render_frames
    out_dir = Path(args.out) if args.out else Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    video_path = out_dir / f"{args.scenario}.mp4"
    solver_name = args.solver if args.solver == "mujoco" else "feather_pgs"
    print(f"Rendering {num_frames} frames of {args.scenario} with {solver_name}")
    print(f"Resolution: {width}x{height}, substeps: {args.substeps}")
    print(f"Output: {video_path}")

    ffmpeg_proc = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            "rgb24",
            "-r",
            str(int(render_fps)),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            str(video_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Pre-allocate frame buffer on GPU
    frame_buf = wp.empty(shape=(height, width, 3), dtype=wp.uint8, device=wp.get_device())

    # Simulate and capture
    for i in range(num_frames):
        for _ in range(args.substeps):
            contacts = model.collide(state_0)
            state_0.clear_forces()
            solver.step(state_0, state_1, control, contacts, sim_dt)
            state_0, state_1 = state_1, state_0
        sim_time += frame_dt

        # Render and capture frame
        viewer.begin_frame(sim_time)
        viewer.log_state(state_0)
        viewer.log_contacts(contacts, state_0)
        viewer.end_frame()

        frame_buf = viewer.get_frame(target_image=frame_buf)
        ffmpeg_proc.stdin.write(frame_buf.numpy().tobytes())

        if (i + 1) % 60 == 0 or i == num_frames - 1:
            print(f"  frame {i + 1}/{num_frames}")

    # Finalize
    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()

    if ffmpeg_proc.returncode != 0:
        print(f"Error: ffmpeg exited with code {ffmpeg_proc.returncode}")
        sys.exit(1)

    file_size_mb = video_path.stat().st_size / (1024 * 1024)
    print(f"Video saved: {video_path} ({file_size_mb:.1f} MB)")

    # Write render metadata
    metadata = collect_metadata(args, scenario_cfg)
    metadata["num_worlds"] = 1
    metadata["render_frames"] = args.render_frames
    metadata["render_width"] = width
    metadata["render_height"] = height

    render_metadata = build_render_metadata(
        args,
        render_fps=render_fps,
        width=width,
        height=height,
        video_name=f"{args.scenario}.mp4",
    )
    write_render_artifacts(out_dir, metadata, render_metadata)
    print(f"Metadata saved: {out_dir / 'render_meta.json'}")

    viewer.close()
    return render_metadata


# =============================================================================
# Interactive Mode
# =============================================================================


def run_interactive(args):
    """Run interactive mode with viewer."""
    import warp as wp  # noqa: PLC0415

    wp.config.enable_backward = False

    import newton.viewer  # noqa: PLC0415

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
# Main
# =============================================================================


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Concrete benchmark and render worker for Newton nightly jobs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--benchmark", action="store_true", help="Run one headless benchmark job")
    mode_group.add_argument("--render", action="store_true", help="Render one headless video job")

    parser.add_argument(
        "--render-frames", type=int, default=300, help="Number of frames to render (default: 300 = 5s at 60fps)"
    )
    parser.add_argument("--render-width", type=int, default=1920, help="Render width in pixels")
    parser.add_argument("--render-height", type=int, default=1080, help="Render height in pixels")
    parser.add_argument("--camera-pos", type=str, default=None, help="Camera position as 'x,y,z' (overrides preset)")
    parser.add_argument("--camera-pitch", type=float, default=None, help="Camera pitch in degrees (overrides preset)")
    parser.add_argument("--camera-yaw", type=float, default=None, help="Camera yaw in degrees (overrides preset)")

    parser.add_argument(
        "--scenario", type=str, choices=list(SCENARIOS.keys()), default="g1_flat", help="Scenario to run"
    )
    parser.add_argument("--num-worlds", type=int, default=4096, help="Number of parallel worlds")
    parser.add_argument("--substeps", type=int, default=None, help="Sim substeps per frame (default: from scenario)")
    parser.add_argument("--warmup-frames", type=int, default=16, help="Warmup frames")
    parser.add_argument("--measure-frames", type=int, default=64, help="Measurement frames")
    parser.add_argument("--viewer", type=str, default="gl", choices=["gl", "null"], help="Viewer type")
    parser.add_argument("--out", type=str, help="Output directory for job-owned artifacts")

    parser.add_argument(
        "--solver",
        type=str,
        default="feather_pgs",
        help="Solver: mujoco, feather_pgs, or preset (fpgs_tiled, fpgs_loop, fpgs_mf, ...)",
    )

    parser.add_argument("--storage", type=str, choices=["flat", "batched"], help="Storage mode")
    parser.add_argument(
        "--cholesky-kernel", type=str, default="auto", choices=["loop", "tiled", "auto"], help="Cholesky kernel"
    )
    parser.add_argument(
        "--trisolve-kernel", type=str, default="auto", choices=["loop", "tiled", "auto"], help="Trisolve kernel"
    )
    parser.add_argument(
        "--hinv-jt-kernel", type=str, default="auto", choices=["par_row", "tiled", "auto"], help="H^-1 J^T kernel"
    )
    parser.add_argument(
        "--delassus-kernel", type=str, default="auto", choices=["par_row_col", "tiled", "auto"], help="Delassus kernel"
    )
    parser.add_argument(
        "--pgs-kernel",
        type=str,
        default="tiled_contact",
        choices=["loop", "tiled_row", "tiled_contact", "streaming"],
        help="PGS kernel",
    )
    parser.add_argument("--pgs-iterations", type=int, default=8, help="PGS iterations")
    parser.add_argument(
        "--dense-max-constraints", type=int, default=64, help="Max dense (articulation) constraint rows per world"
    )
    parser.add_argument("--pgs-beta", type=float, default=0.1, help="PGS position correction factor (ERP)")
    parser.add_argument("--pgs-cfm", type=float, default=1.0e-6, help="PGS constraint force mixing (regularization)")
    parser.add_argument("--pgs-omega", type=float, default=1.0, help="PGS relaxation factor (SOR)")
    parser.add_argument("--pgs-warmstart", action="store_true", help="Enable warmstart")
    parser.add_argument(
        "--pgs-mode",
        type=str,
        choices=["delassus", "hybrid", "velocity"],
        default="hybrid",
        help="PGS mode: delassus, hybrid, or velocity",
    )
    parser.add_argument(
        "--delassus-chunk-size",
        type=int,
        default=None,
        help="Chunk size (constraint rows) for streaming Delassus kernel (None=auto)",
    )
    parser.add_argument(
        "--pgs-chunk-size",
        type=int,
        default=None,
        help="Chunk size (contacts) for streaming PGS kernel (None=1)",
    )
    parser.add_argument("--use-parallel-streams", action="store_true", dest="use_parallel_streams")
    parser.add_argument("--no-parallel-streams", action="store_false", dest="use_parallel_streams")
    parser.set_defaults(use_parallel_streams=True)
    parser.add_argument("--double-buffer", action="store_true", dest="double_buffer")
    parser.add_argument("--no-double-buffer", action="store_false", dest="double_buffer")
    parser.set_defaults(double_buffer=True)
    parser.add_argument("--pipeline-collide", action="store_true", dest="pipeline_collide")
    parser.add_argument("--no-pipeline-collide", action="store_false", dest="pipeline_collide")
    parser.set_defaults(pipeline_collide=False)
    parser.add_argument("--nvtx", action="store_true", help="Enable NVTX markers in solver stages")
    parser.add_argument("--no-graph", action="store_true", help="Disable CUDA graph capture (for NVTX profiling)")
    parser.add_argument(
        "--override-scenario-defaults",
        action="store_true",
        help="CLI kernel args override scenario defaults",
    )

    parser.add_argument("--mj-solver", type=str, choices=["cg", "newton"])
    parser.add_argument("--mj-integrator", type=str, choices=["euler", "rk4", "implicit", "implicitfast"])
    parser.add_argument("--mj-njmax", type=int)
    parser.add_argument("--mj-nconmax", type=int)

    parser.add_argument("--summary-timer", action="store_true", help="Print kernel summary and store kernel timings")
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.scenario in SCENARIOS:
        scenario_cfg = SCENARIOS[args.scenario]
        if args.substeps is None:
            args.substeps = scenario_cfg.get("default_substeps", 4)
        if args.pgs_iterations == 8:
            args.pgs_iterations = scenario_cfg.get("default_pgs_iterations", args.pgs_iterations)
        if args.dense_max_constraints == 64:
            args.dense_max_constraints = scenario_cfg.get("default_dense_max_constraints", args.dense_max_constraints)

    if args.render:
        run_render(args)
    elif args.benchmark:
        run_direct(args)
    else:
        run_interactive(args)


if __name__ == "__main__":
    main()
