# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Record and validate the contact-driven yoyo scene with FPGS."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import warp as wp
from scene import build_template
from yoyo_task import check, string_force

import newton
from newton.solvers import SolverFeatherPGS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--duration", type=float, default=8)
    p.add_argument("--substeps", type=int, default=8)
    p.add_argument("--iterations", type=int, default=16)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    wp.init()
    wp.set_device("cuda:0")
    b, info = build_template("yoyo", 0, json.loads(args.assets.read_text()), "cuda:0")
    builder = newton.ModelBuilder()
    builder.add_world(b, label_prefix="00_yoyo")
    builder.add_ground_plane(height=-0.8)
    model = builder.finalize(device="cuda:0")
    s0, s1 = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    pipeline = newton.CollisionPipeline(model, broad_phase="sap", rigid_contact_max=32768)
    contacts = pipeline.contacts()
    solver = SolverFeatherPGS(
        model,
        solve_mode="matrix_free",
        articulated_contact_response="immediate",
        iterations=args.iterations,
        enable_joint_limits=True,
        enable_joint_velocity_limits=True,
        articulation_constraint_max=512,
        free_body_constraint_max=4096,
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
        id="yoyo-0",
    )
    clean = {k: v for k, v in info.items() if k not in ("waypoints", "policy") and not k.startswith("_")}
    (args.output / "model-summary.json").write_text(
        json.dumps({"worlds": [clean], "body_labels": model.body_label}, indent=2)
    )
    times = np.array([w[0] for w in info["waypoints"]])
    qs = np.array([w[1] for w in info["waypoints"]])
    targets = np.array(builder.joint_target_q, np.float32)
    from scipy.spatial.transform import Rotation

    anchor = np.array(info["yoyo_anchor"])

    def record(state, available=0.08):
        pose = state.body_q.numpy()
        h = pose[info["yoyo_hand"]]
        a = h[:3] + Rotation.from_quat(h[3:]).apply(anchor)
        c = pose[info["tracked_body"], :3]
        delta = c - a
        # Display a massless slack cord with its actual unwound length.
        u = np.linspace(0, 1, 161)[:, None]
        line = a + u * delta
        sag_direction = np.array([0.3, 0, -1.0])
        low, high = 0.0, max(available, 0.02)
        for _ in range(16):
            sag = (low + high) / 2
            curve = line + 4 * u * (1 - u) * sag * sag_direction
            length = np.linalg.norm(np.diff(curve, axis=0), axis=1).sum()
            if length < max(np.linalg.norm(delta), available):
                low = sag
            else:
                high = sag
        curve = line + 4 * u * (1 - u) * (low + high) / 2 * sag_direction
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
        samples = np.linspace(0, arc[-1], len(info["yoyo_cord"]) + 1)
        ends = np.column_stack([np.interp(samples, arc, curve[:, j]) for j in range(3)])
        for i, body in enumerate(info["yoyo_cord"]):
            direction = ends[i + 1] - ends[i]
            rotation = wp.quat_between_vectors(wp.vec3(0, 0, 1), wp.vec3(*(direction / np.linalg.norm(direction))))
            pose[body, :3] = (ends[i] + ends[i + 1]) / 2
            pose[body, 3:] = np.array(rotation)
        return pose

    initial = record(s0)
    h = initial[info["yoyo_hand"]]
    a = h[:3] + Rotation.from_quat(h[3:]).apply(anchor)
    distance = np.linalg.norm(initial[info["tracked_body"], :3] - a)
    angle = -(info["yoyo_string_length"] - max(distance, 0.08)) / info["yoyo_axle_radius"]
    winding = wp.array([float(angle)], dtype=float, device="cuda:0")
    metrics = wp.zeros(4, dtype=float, device="cuda:0")
    telemetry = []
    poses = [initial]
    joints = [s0.joint_q.numpy()]
    velocities = [s0.body_qd.numpy()]
    tick = time.monotonic()
    for frame in range(round(args.duration * 50)):
        for substep in range(args.substeps):
            t = (frame + substep / args.substeps) / 50
            ix = min(max(int(np.searchsorted(times, t, side="right")) - 1, 0), len(times) - 2)
            u = np.clip((t - times[ix]) / (times[ix + 1] - times[ix]), 0, 1)
            targets[: qs.shape[1]] = qs[ix] * (1 - u) + qs[ix + 1] * u
            control.joint_target_q.assign(targets)
            s0.clear_forces()
            wp.launch(
                string_force,
                dim=1,
                inputs=[
                    s0.body_q,
                    s0.body_qd,
                    s0.body_f,
                    model.body_com,
                    winding,
                    metrics,
                    info["yoyo_hand"],
                    info["tracked_body"],
                    wp.vec3(*anchor),
                    info["yoyo_axle_radius"],
                    info["yoyo_string_length"],
                    0.02 / args.substeps,
                ],
            )
            pipeline.collide(s0, contacts)
            solver.step(s0, s1, control, contacts, 0.02 / args.substeps)
            s0, s1 = s1, s0
        poses.append(
            record(
                s0,
                info["yoyo_string_length"]
                - info["yoyo_axle_radius"] * float(np.sqrt(winding.numpy()[0] ** 2 + 0.15**2)),
            )
        )
        telemetry.append([*metrics.numpy().tolist(), float(winding.numpy()[0])])
        joints.append(s0.joint_q.numpy())
        velocities.append(s0.body_qd.numpy())
        if frame % 100 == 0:
            print("STEP", frame, "yo-yo", poses[-1][info["tracked_body"], :3], flush=True)
    solver.check_constraint_capacity()
    pipeline.collide(s0, contacts)
    count = int(contacts.rigid_contact_count.numpy()[0])
    owners = model.shape_body.numpy()
    shapes0 = contacts.rigid_contact_shape0.numpy()[:count]
    shapes1 = contacts.rigid_contact_shape1.numpy()[:count]
    touched = {}
    for a, b in zip(shapes0, shapes1, strict=True):
        owner0, owner1 = int(owners[a]), int(owners[b])
        if info["tracked_body"] in (owner0, owner1):
            other = owner1 if owner0 == info["tracked_body"] else owner0
            label = model.body_label[other] if other >= 0 else "world"
            touched[label] = touched.get(label, 0) + 1
    print("FINAL_TOY_CONTACTS", touched, flush=True)
    poses = np.array(poses)
    np.savez_compressed(
        args.output / "trace.npz", poses=poses, velocities=velocities, joints=joints, fps=50, string_telemetry=telemetry
    )
    result = check(info, poses, 50)
    report = {
        "solver": "SolverFeatherPGS",
        "substeps": args.substeps,
        "iterations": args.iterations,
        "quality_gate_passed": result["pass"],
        "metrics": result,
        "constraint_capacity_checked": True,
        "wall_seconds": time.monotonic() - tick,
        "trace_sha256": hashlib.sha256((args.output / "trace.npz").read_bytes()).hexdigest(),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
