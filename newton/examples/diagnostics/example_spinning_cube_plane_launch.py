# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""One-cube FeatherPGS spin-to-launch diagnostic.

This is a deliberately small repro for the Franka lift failure mode after the
hand has already created a high-spin cube state.  It uses no IsaacLab, no robot,
and no policy: one free box starts on a ground plane with a representative
post-onset cube velocity captured from the FPGS Franka rollout.

Command:
    python -m newton.examples.diagnostics.example_spinning_cube_plane_launch
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton._src.solvers import SolverFeatherPGS

DEFAULT_DT = 0.005
DEFAULT_STEPS = 80
DEFAULT_HALF_EXTENT = 0.02

# Representative FPGS post-step-1 cube state from the 2026-05-15 Franka
# pre-onset replay bundle, env 22.  The root position was env-origin shifted in
# the IL capture, so the raw Newton repro uses only the local height and twist.
CAPTURED_LINEAR_VEL = np.array([0.299165278673172, -1.654648780822754, 0.38417425751686096], dtype=np.float32)
CAPTURED_ANGULAR_VEL = np.array([13.849493980407715, 5.491700649261475, -4.771509647369385], dtype=np.float32)


def _make_solver(model: newton.Model, *, pgs_iterations: int, pgs_beta: float) -> SolverFeatherPGS:
    return SolverFeatherPGS(
        model,
        update_mass_matrix_interval=1,
        pgs_iterations=pgs_iterations,
        pgs_beta=pgs_beta,
        pgs_cfm=1.0e-6,
        pgs_omega=1.0,
        dense_max_constraints=512,
        pgs_warmstart=False,
        pgs_mode="matrix_free",
        enable_contact_friction=True,
        cholesky_kernel="auto",
        trisolve_kernel="auto",
        hinv_jt_kernel="auto",
        delassus_kernel="auto",
        pgs_kernel="tiled_contact",
        delassus_chunk_size=128,
        pgs_chunk_size=128,
        small_dof_threshold=12,
        use_parallel_streams=False,
        double_buffer=False,
        nvtx=False,
        pgs_debug=False,
        friction_mode="current",
        drive_mode="augmented",
    )


def build_model(args: argparse.Namespace) -> tuple[newton.Model, int]:
    builder = newton.ModelBuilder()
    builder.gravity = args.gravity
    builder.default_shape_cfg.mu = args.mu
    builder.default_shape_cfg.ke = args.ke
    builder.default_shape_cfg.kd = args.kd
    builder.default_shape_cfg.kf = args.kf

    half = args.half_extent
    cube = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, args.initial_z), wp.quat_identity()),
        mass=args.mass,
        label="spinning_cube",
    )
    builder.add_shape_box(cube, hx=half, hy=half, hz=half)
    builder.add_ground_plane()
    return builder.finalize(), cube


def seed_free_body_state(
    model: newton.Model,
    state: newton.State,
    *,
    initial_z: float,
    zero_vz: bool,
    scale_angular: float,
) -> np.ndarray:
    pose = np.array([[0.0, 0.0, initial_z, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    linear = CAPTURED_LINEAR_VEL.copy()
    if zero_vz:
        linear[2] = 0.0
    angular = CAPTURED_ANGULAR_VEL * np.float32(scale_angular)
    twist = np.concatenate((linear, angular)).astype(np.float32)

    state.body_q.assign(pose)
    state.joint_q.assign(pose.reshape(-1))
    model.body_q.assign(pose)
    model.joint_q.assign(pose.reshape(-1))
    state.body_qd.assign(twist.reshape(1, 6))
    state.joint_qd.assign(twist)
    model.body_qd.assign(twist.reshape(1, 6))
    model.joint_qd.assign(twist)
    return twist


def run(args: argparse.Namespace) -> dict[str, object]:
    model, cube = build_model(args)
    solver = _make_solver(model, pgs_iterations=args.pgs_iterations, pgs_beta=args.pgs_beta)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    collision_pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        contact_matching=args.contact_matching,
        reduce_contacts=False,
    )
    contacts = collision_pipeline.contacts()
    twist = seed_free_body_state(
        model,
        state_0,
        initial_z=args.initial_z,
        zero_vz=args.zero_vz,
        scale_angular=args.scale_angular,
    )

    z0 = float(state_0.body_q.numpy()[cube, 2])
    rows: list[dict[str, float]] = []
    max_z_delta = 0.0
    max_up_vel = float(twist[2])
    max_body_up_vel = float(twist[2])
    max_ang = float(np.linalg.norm(twist[3:6]))
    max_body_ang = float(np.linalg.norm(twist[3:6]))

    for step in range(args.steps):
        state_0.clear_forces()
        collision_pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, args.dt)
        if hasattr(solver, "update_contacts"):
            solver.update_contacts(contacts)
        state_0, state_1 = state_1, state_0

        q = state_0.body_q.numpy()[cube]
        qd = state_0.joint_qd.numpy()
        body_qd = state_0.body_qd.numpy()[cube]
        z_delta = float(q[2] - z0)
        up_vel = float(qd[2])
        lin_norm = float(np.linalg.norm(qd[:3]))
        ang_norm = float(np.linalg.norm(qd[3:6]))
        body_up_vel = float(body_qd[2])
        body_lin_norm = float(np.linalg.norm(body_qd[:3]))
        body_ang_norm = float(np.linalg.norm(body_qd[3:6]))
        contact_count = int(contacts.rigid_contact_count.numpy()[0])
        max_z_delta = max(max_z_delta, z_delta)
        max_up_vel = max(max_up_vel, up_vel)
        max_ang = max(max_ang, ang_norm)
        max_body_up_vel = max(max_body_up_vel, body_up_vel)
        max_body_ang = max(max_body_ang, body_ang_norm)
        rows.append(
            {
                "step": float(step + 1),
                "z_delta": z_delta,
                "up_vel": up_vel,
                "lin_norm": lin_norm,
                "ang_norm": ang_norm,
                "body_up_vel": body_up_vel,
                "body_lin_norm": body_lin_norm,
                "body_ang_norm": body_ang_norm,
                "contact_count": float(contact_count),
            }
        )

    summary = {
        "steps": args.steps,
        "dt": args.dt,
        "half_extent": args.half_extent,
        "initial_z": args.initial_z,
        "initial_twist": [float(x) for x in twist],
        "max_z_delta": max_z_delta,
        "max_up_vel": max_up_vel,
        "max_ang_norm": max_ang,
        "max_body_up_vel": max_body_up_vel,
        "max_body_ang_norm": max_body_ang,
        "final_z_delta": rows[-1]["z_delta"],
        "final_up_vel": rows[-1]["up_vel"],
        "final_body_up_vel": rows[-1]["body_up_vel"],
        "trace": rows,
    }
    return summary


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT)
    parser.add_argument("--half-extent", type=float, default=DEFAULT_HALF_EXTENT)
    parser.add_argument("--initial-z", type=float, default=DEFAULT_HALF_EXTENT, help="Cube center height.")
    parser.add_argument("--gravity", type=float, default=-9.81, help="Scalar gravity along the model up vector.")
    parser.add_argument("--mass", type=float, default=0.1)
    parser.add_argument("--mu", type=float, default=1.0)
    parser.add_argument("--ke", type=float, default=5.0e4)
    parser.add_argument("--kd", type=float, default=5.0e2)
    parser.add_argument("--kf", type=float, default=1.0e3)
    parser.add_argument("--pgs-iterations", type=int, default=16)
    parser.add_argument("--pgs-beta", type=float, default=0.05)
    parser.add_argument("--scale-angular", type=float, default=1.0)
    parser.add_argument("--zero-vz", action="store_true", help="Remove inherited upward velocity from the capture.")
    parser.add_argument("--contact-matching", choices=("disabled", "latest", "sticky"), default="sticky")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--print-every", type=int, default=10)
    return parser


def main() -> None:
    args = create_parser().parse_args()
    if args.initial_z < args.half_extent:
        raise ValueError("--initial-z must be at or above --half-extent for a cube resting on the plane.")
    summary = run(args)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True))

    print(
        "spinning_cube_plane_launch: "
        f"max_z_delta={summary['max_z_delta']:.6f}m "
        f"max_up_vel={summary['max_up_vel']:.6f}m/s "
        f"max_ang={summary['max_ang_norm']:.6f}rad/s"
    )
    for row in summary["trace"]:
        step = int(row["step"])
        if step <= 12 or step == args.steps or (args.print_every > 0 and step % args.print_every == 0):
            print(
                f"step={step:03d} "
                f"z_delta={row['z_delta']:+.6f} "
                f"up={row['up_vel']:+.6f} "
                f"lin={row['lin_norm']:.6f} "
                f"ang={row['ang_norm']:.6f} "
                f"body_up={row['body_up_vel']:+.6f} "
                f"body_lin={row['body_lin_norm']:.6f} "
                f"body_ang={row['body_ang_norm']:.6f} "
                f"contacts={int(row['contact_count'])}"
            )


if __name__ == "__main__":
    main()
