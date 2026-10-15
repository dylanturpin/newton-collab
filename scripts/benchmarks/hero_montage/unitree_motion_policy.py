# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Run public G1 Moves tracking policies in FPGS with closed-loop observations."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import warp as wp
from scene import build_template

import newton
from newton.solvers import SolverFeatherPGS


@wp.kernel
def pd_torque(
    q: wp.array[float],
    qd: wp.array[float],
    qi: wp.array[int],
    vi: wp.array[int],
    target: wp.array[float],
    kp: wp.array[float],
    kd: wp.array[float],
    limit: wp.array[float],
    force: wp.array[float],
):
    i = wp.tid()
    force[vi[i]] = wp.clamp(kp[i] * (target[i] - q[qi[i]]) - kd[i] * qd[vi[i]], -limit[i], limit[i])


def main():
    from scipy.spatial.transform import Rotation

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=12)
    parser.add_argument("--start", type=float, default=0)
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--reference-only", action="store_true", help="Negative control: bypass policy feedback")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.policy), sess_options=options, providers=["CPUExecutionProvider"])
    metadata = session.get_modelmeta().custom_metadata_map

    def array(key):
        return np.fromstring(metadata[key], sep=",", dtype=np.float32)

    names = metadata["joint_names"].split(",")
    defaults = array("default_joint_pos")
    scale = array("action_scale")
    kp = array("joint_stiffness")
    kd = array("joint_damping")
    wp.init()
    wp.set_device("cuda:0")
    b, info = build_template("g1", 0, json.loads(args.assets.read_text()), "cuda:0")
    joint_names = {name.rsplit("/", 1)[-1]: i for i, name in enumerate(b.joint_label)}
    joints = [joint_names[n] for n in names]
    qi = np.array([b.joint_q_start[j] for j in joints])
    vi = np.array([b.joint_qd_start[j] for j in joints])
    effort = []
    for i, name in enumerate(names):
        if "hip_roll" in name or "knee" in name:
            arm, lim = 0.025101925, 139
        elif "hip_pitch" in name or "hip_yaw" in name or name == "waist_yaw_joint":
            arm, lim = 0.01017752004132231, 88
        elif "ankle" in name or name in ("waist_pitch_joint", "waist_roll_joint"):
            arm, lim = 0.00721945, 50
        elif "wrist_pitch" in name or "wrist_yaw" in name:
            arm, lim = 0.00425, 5
        else:
            arm, lim = 0.003609725, 25
        effort.append(lim)
        b.joint_armature[vi[i]] = arm
        b.joint_target_ke[vi[i]] = 0
        b.joint_target_kd[vi[i]] = 0
        b.joint_target_mode[vi[i]] = int(newton.JointTargetMode.NONE)
    zeros = np.zeros((1, 154), np.float32)

    def infer(obs, f):
        return session.run(None, {"obs": obs[None] if obs.ndim == 1 else obs, "time_step": np.array([[f]], np.float32)})

    first = infer(zeros, round(args.start * 50))
    root = first[3][0, 0]
    quat = first[4][0, 0]
    b.joint_q[:3] = [0, 0, float(root[2])]
    b.joint_q[3:7] = quat[[1, 2, 3, 0]].tolist()
    for i, j in enumerate(qi):
        b.joint_q[j] = float(first[1][0, i])
    for i, j in enumerate(vi):
        b.joint_qd[j] = float(first[2][0, i])
    b.joint_qd[:3] = first[5][0, 0].tolist()
    b.joint_qd[3:6] = first[6][0, 0].tolist()
    builder = newton.ModelBuilder()
    builder.add_world(b, label_prefix="00_g1")
    builder.add_ground_plane(height=-0.8)
    model = builder.finalize(device="cuda:0")
    model.set_gravity((0, 0, -9.81))
    s0, s1 = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    pipeline = newton.CollisionPipeline(model, broad_phase="sap", rigid_contact_max=4096)
    contacts = pipeline.contacts()
    solver = SolverFeatherPGS(
        model,
        solve_mode="matrix_free",
        articulated_contact_response="immediate",
        iterations=args.iterations,
        enable_joint_limits=True,
        enable_joint_velocity_limits=True,
        articulation_constraint_max=256,
        free_body_constraint_max=512,
        friction_anchor_beta=0,
        contact_warmstart=False,
        use_parallel_streams=False,
    )
    pelvis = next(i for i, n in enumerate(model.body_label) if n.rsplit("/", 1)[-1] == "pelvis")
    anchor = next(i for i, n in enumerate(model.body_label) if n.rsplit("/", 1)[-1] == metadata["anchor_body_name"])
    ref_anchor = metadata["body_names"].split(",").index(metadata["anchor_body_name"])
    arrays = [
        wp.array(qi, dtype=int),
        wp.array(vi, dtype=int),
        wp.zeros(29, dtype=float),
        wp.array(kp),
        wp.array(kd),
        wp.array(np.array(effort, np.float32)),
    ]
    poses = [s0.body_q.numpy()]
    qs = [s0.joint_q.numpy()]
    actions = []
    errors = []
    last = np.zeros(29, np.float32)
    tick = time.monotonic()
    frames = round(args.duration * 50)
    for f in range(frames):
        step = f + round(args.start * 50)
        reference = infer(zeros, step)
        q = s0.joint_q.numpy()
        v = s0.joint_qd.numpy()
        body = s0.body_q.numpy()
        vel = s0.body_qd.numpy()
        rel = Rotation.from_quat(body[anchor, 3:]).inv() * Rotation.from_quat(reference[4][0, ref_anchor][[1, 2, 3, 0]])
        omega = Rotation.from_quat(body[pelvis, 3:]).inv().apply(vel[pelvis, 3:])
        obs = np.concatenate(
            [reference[1][0], reference[2][0], rel.as_matrix()[:, :2].ravel(), omega, q[qi] - defaults, v[vi], last]
        ).astype(np.float32)
        action = infer(obs, step)[0][0]
        last = action.copy()
        target = defaults + scale * action
        if args.reference_only:
            target = reference[1][0].copy()
        arrays[2].assign(target)
        actions.append(action)
        errors.append(float(np.sqrt(np.mean((q[qi] - reference[1][0]) ** 2))))
        for _ in range(args.substeps):
            s0.clear_forces()
            wp.launch(pd_torque, dim=29, inputs=[s0.joint_q, s0.joint_qd, *arrays, control.joint_f])
            pipeline.collide(s0, contacts)
            solver.step(s0, s1, control, contacts, 0.02 / args.substeps)
            s0, s1 = s1, s0
        poses.append(s0.body_q.numpy())
        qs.append(s0.joint_q.numpy())
        if f % 100 == 0:
            print("STEP", f, "height", poses[-1][pelvis, 2], "rmse", errors[-1], flush=True)
        if not np.isfinite(poses[-1]).all() or poses[-1][pelvis, 2] < 0.35:
            break
    solver.check_constraint_capacity()
    poses = np.array(poses)
    qs = np.array(qs)
    np.savez_compressed(args.output / "trace.npz", poses=poses, joints=qs, fps=50)
    np.savez_compressed(args.output / "policy-actions.npz", actions=actions, tracking_rmse=errors)
    info.update(
        body_start=0,
        q_start=0,
        v_start=0,
        body_count=b.body_count,
        coord_count=b.joint_coord_count,
        dof_count=b.joint_dof_count,
        id="g1-0",
    )
    info = {k: v for k, v in info.items() if k not in ("waypoints", "policy") and not k.startswith("_")}
    (args.output / "model-summary.json").write_text(
        json.dumps({"worlds": [info], "body_labels": model.body_label}, indent=2)
    )
    upright = Rotation.from_quat(poses[:, pelvis, 3:]).as_matrix()[:, 2, 2]
    passed = (
        len(poses) == frames + 1
        and np.isfinite(poses).all()
        and poses[:, pelvis, 2].min() > 0.45
        and upright.min() > 0.5
        and np.mean(errors) < 0.6
    )
    report = {
        "solver": "SolverFeatherPGS",
        "constraint_capacity_checked": True,
        "substeps": args.substeps,
        "iterations": args.iterations,
        "duration": (len(poses) - 1) / 50,
        "requested_duration": args.duration,
        "quality_gate_passed": bool(passed),
        "min_pelvis_height": float(poses[:, pelvis, 2].min()),
        "min_upright": float(upright.min()),
        "mean_joint_tracking_rmse": float(np.mean(errors)),
        "max_xy_displacement": float(np.linalg.norm(poses[:, pelvis, :2] - poses[0, pelvis, :2], axis=1).max()),
        "policy": str(args.policy),
        "policy_sha256": hashlib.sha256(args.policy.read_bytes()).hexdigest(),
        "policy_metadata": metadata,
        "reference_only": args.reference_only,
        "trace_sha256": hashlib.sha256((args.output / "trace.npz").read_bytes()).hexdigest(),
        "wall_seconds": time.monotonic() - tick,
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
