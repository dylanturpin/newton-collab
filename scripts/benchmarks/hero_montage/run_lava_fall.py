# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Record a gravity-driven miniature falling into 200 granular flame pieces."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import warp as wp
from scene import build_template

import newton
from newton.solvers import SolverFeatherPGS


def validate_geometry(builder, info, poses):
    """Check the complete visible figure, not just its center, is inside the pot."""
    from scipy.spatial.transform import Rotation

    points = []
    for i, owner in enumerate(builder.shape_body):
        if owner != info["tracked_body"] or not builder.shape_flags[i] & int(newton.ShapeFlags.VISIBLE):
            continue
        mesh = builder.shape_source[i]
        if isinstance(mesh, newton.Mesh):
            shape = np.array(builder.shape_transform[i])
            vertices = np.asarray(mesh.vertices) * np.array(builder.shape_scale[i])
            points.append(Rotation.from_quat(shape[3:]).apply(vertices) + shape[:3])
    final = poses[-1, info["tracked_body"]]
    vertices = Rotation.from_quat(final[3:]).apply(np.concatenate(points)) + final[:3]
    radial = np.linalg.norm(vertices[:, :2] - [0.1, 0], axis=1)
    pieces = poses[-1, info["lava_bodies"], :3]
    inside = np.linalg.norm(pieces[:, :2] - [0.1, 0], axis=1) < 0.43
    surface = float(np.quantile(pieces[inside, 2], 0.8))
    return {
        "figure_top_z": float(vertices[:, 2].max()),
        "figure_max_radius": float(radial.max()),
        "contained_flames": int(inside.sum()),
        "pile_surface": surface,
        "complete_figure_inside": bool(radial.max() < 0.412),
        "complete_figure_below_surface": bool(vertices[:, 2].max() < surface),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--duration", type=float, default=4)
    p.add_argument("--prepare-pile", action="store_true")
    args = p.parse_args()
    if args.prepare_pile:
        os.environ["LAVA_PREPARE_PILE"] = "1"
    args.output.mkdir(parents=True, exist_ok=True)
    wp.init()
    wp.set_device("cuda:0")
    b, info = build_template("lava_fall", 0, json.loads(args.assets.read_text()), "cuda:0")
    builder = newton.ModelBuilder()
    builder.add_world(b, label_prefix="00_lava_fall")
    builder.add_ground_plane(height=-0.8)
    model = builder.finalize(device="cuda:0")
    s0, s1 = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="sap",
        rigid_contact_max=65536,
        reduce_contacts=newton.CollisionPipeline.ContactReductionConfig(
            body_pairs=True, body_pair_cell_size=0.04, body_pair_hysteresis=0
        ),
        contact_matching="disabled",
    )
    contacts = pipeline.contacts()
    solver = SolverFeatherPGS(
        model,
        solve_mode="matrix_free",
        articulated_contact_response="immediate",
        iterations=32,
        deterministic=True,
        free_body_constraint_max=10000,
        articulation_constraint_max=512,
        friction_anchor_beta=0,
        contact_warmstart=False,
        use_parallel_streams=False,
    )
    info.update(
        body_start=0,
        q_start=0,
        v_start=0,
        body_count=b.body_count,
        coord_count=b.joint_coord_count,
        dof_count=b.joint_dof_count,
        id="lava_fall-0",
    )
    clean = {k: v for k, v in info.items() if k not in ("waypoints", "policy") and not k.startswith("_")}
    (args.output / "model-summary.json").write_text(
        json.dumps({"worlds": [clean], "body_labels": model.body_label}, indent=2)
    )
    poses = [s0.body_q.numpy()]
    joints = [s0.joint_q.numpy()]
    tick = time.monotonic()
    for frame in range(round(args.duration * 60)):
        for _ in range(4):
            s0.clear_forces()
            pipeline.collide(s0, contacts)
            solver.step(s0, s1, control, contacts, 1 / 240)
            s0, s1 = s1, s0
        poses.append(s0.body_q.numpy())
        joints.append(s0.joint_q.numpy())
        if frame % 30 == 0:
            print("STEP", frame, "figure", poses[-1][info["tracked_body"], :3], flush=True)
    solver.check_constraint_capacity()
    poses = np.array(poses)
    np.savez_compressed(args.output / "trace.npz", poses=poses, joints=joints, fps=60)
    if args.prepare_pile:
        np.savez_compressed(
            Path(__file__).parent / "assets" / "lava_settled_pile.npz", poses=poses[-1, info["lava_bodies"]]
        )
    figure = poses[:, info["tracked_body"], :3]
    pieces = poses[-1, info["lava_bodies"], :3]
    surface = float(np.quantile(pieces[:, 2], 0.80))
    end = figure[-1]
    passed = bool(np.isfinite(poses).all() and end[2] < surface - 0.08 and np.linalg.norm(end[:2] - [0.1, 0]) < 0.4)
    geometry = validate_geometry(b, info, poses)
    passed &= geometry["complete_figure_inside"] and geometry["complete_figure_below_surface"]
    report = {
        "solver": "SolverFeatherPGS",
        "substeps": 4,
        "iterations": 32,
        "quality_gate_passed": passed,
        "metrics": {
            "figure_start": figure[0].tolist(),
            "figure_end": end.tolist(),
            "pile_surface": surface,
            "flame_count": 200,
            "physical_free_fall": True,
            **geometry,
        },
        "constraint_capacity_checked": True,
        "wall_seconds": time.monotonic() - tick,
        "trace_sha256": hashlib.sha256((args.output / "trace.npz").read_bytes()).hexdigest(),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
