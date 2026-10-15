# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Record and validate the contact-driven wrecking crane with FPGS."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import warp as wp
from audit import demolition_check
from scene import build_template

import newton
from newton.solvers import SolverFeatherPGS


def main():
    from scipy.spatial.transform import Rotation

    p = argparse.ArgumentParser()
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--duration", type=float, default=15)
    p.add_argument("--substeps", type=int, default=4)
    p.add_argument("--iterations", type=int, default=32)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    wp.init()
    wp.set_device("cuda:0")
    b, info = build_template("gear", 0, json.loads(args.assets.read_text()), "cuda:0")
    assert not any(name.endswith(("crane_long_jaw", "handle_support", "grasp_handle")) for name in b.shape_label), (
        "Wrecking crane must use stock fingers and its original rear counterweight"
    )
    builder = newton.ModelBuilder()
    builder.add_world(b, label_prefix="00_gear")
    builder.add_ground_plane(height=-0.8)
    model = builder.finalize(device="cuda:0")
    s0, s1 = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="sap",
        rigid_contact_max=32768,
        reduce_contacts=newton.CollisionPipeline.ContactReductionConfig(
            body_pairs=True, body_pair_cell_size=0.06, body_pair_hysteresis=0.0
        ),
        contact_matching="disabled",
    )
    contacts = pipeline.contacts()
    solver = SolverFeatherPGS(
        model,
        solve_mode="matrix_free",
        articulated_contact_response="immediate",
        iterations=args.iterations,
        deterministic=True,
        enable_joint_friction=True,
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
        id="gear-0",
    )
    clean = {k: v for k, v in info.items() if k not in ("waypoints", "policy") and not k.startswith("_")}
    (args.output / "model-summary.json").write_text(
        json.dumps({"worlds": [clean], "body_labels": model.body_label}, indent=2)
    )
    times = np.array([w[0] for w in info["waypoints"]])
    qs = np.array([w[1] for w in info["waypoints"]])
    targets = np.array(builder.joint_target_q, np.float32)
    poses = [s0.body_q.numpy()]
    joints = [s0.joint_q.numpy()]
    tick = time.monotonic()
    integral = np.zeros(6)
    grasp_frames = 0
    impact = False
    owners = model.shape_body.numpy()
    for frame in range(round(max(args.duration, info.get("motion_duration", 0) + 0.5) * 60)):
        for substep in range(args.substeps):
            t = (frame + substep / args.substeps) / 60
            ix = min(max(int(np.searchsorted(times, t, side="right")) - 1, 0), len(times) - 2)
            u = np.clip((t - times[ix]) / (times[ix + 1] - times[ix]), 0, 1)
            targets[: qs.shape[1]] = qs[ix] * (1 - u) + qs[ix + 1] * u
            error = np.clip(targets[:6] - s0.joint_q.numpy()[:6], -0.08, 0.08)
            integral = np.clip(integral + 3 / (60 * args.substeps) * error, -0.12, 0.12)
            targets[:6] += integral
            control.joint_target_q.assign(targets)
            s0.clear_forces()
            pipeline.collide(s0, contacts)
            solver.step(s0, s1, control, contacts, 1 / (60 * args.substeps))
            s0, s1 = s1, s0
        count = int(contacts.rigid_contact_count.numpy()[0])
        aa = contacts.rigid_contact_shape0.numpy()[:count]
        bb = contacts.rigid_contact_shape1.numpy()[:count]
        pairs = {
            tuple(sorted((int(owners[a]), int(owners[b])))) for a, b in zip(aa, bb, strict=True) if a >= 0 and b >= 0
        }
        grasp_frames += int(
            all(tuple(sorted((finger, info["crane_grasp_body"]))) in pairs for finger in info["crane_fingers"])
        )
        impact |= any(tuple(sorted((info["tracked_body"], body))) in pairs for body in info["demolition_structure"])
        poses.append(s0.body_q.numpy())
        joints.append(s0.joint_q.numpy())
        if frame % 100 == 0:
            print("STEP", frame, "ball", poses[-1][info["tracked_body"], :3], flush=True)
    solver.check_constraint_capacity()
    poses = np.array(poses)
    np.savez_compressed(args.output / "trace.npz", poses=poses, joints=joints, fps=60)
    result = demolition_check(info, poses, 60)
    angles = np.unwrap(Rotation.from_quat(poses[:, info["crane_slew"], 3:]).as_euler("xyz")[:, 2])
    release = angles[round(6.3 * 60) :]
    drift = float(np.max(np.abs(release - release[0])))
    result.update(grasp_frames=grasp_frames, ball_contact_recorded=bool(impact), post_release_drift_rad=drift)
    result["pass"] = bool(result["pass"] and grasp_frames >= 20 and impact and drift < 0.05)
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
