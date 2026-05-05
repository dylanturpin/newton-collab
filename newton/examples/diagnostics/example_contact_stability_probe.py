# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Contact Stability Probes
#
# Minimal FeatherPGS diagnostic scenes for contact and drive stability.
#
# Command:
#   uv run python newton/examples/diagnostics/example_contact_stability_probe.py \
#       --output-dir contact_stability_probe
#
###########################################################################

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton import JointTargetMode
from newton._src.solvers import SolverFeatherPGS

CSV_FIELDS = (
    "probe",
    "variant",
    "result",
    "mu",
    "pgs_iterations",
    "pgs_beta",
    "pgs_cfm",
    "frames",
    "substeps",
    "dt_s",
    "peak_contacts",
    "peak_contact_force_n",
    "peak_box_speed_mps",
    "peak_box_omega_radps",
    "final_x_m",
    "final_y_m",
    "final_z_m",
    "final_tilt_deg",
    "slip_m",
    "fall_m",
    "target_overlap_m",
    "start_clearance_m",
    "close_time_s",
    "release_time_s",
    "box_mass_kg",
    "box_density_kg_m3",
    "grip_drive_ke",
    "grip_effort_limit_n",
    "grip_velocity_limit_mps",
    "peak_grip_cmd_n",
    "mean_grip_cmd_n",
    "mean_squeeze_force_n",
    "peak_squeeze_force_n",
    "disturbance_accel_mps2",
    "shake_hz",
    "drive_ke",
    "drive_effort_limit_n",
    "drive_velocity_limit_mps",
    "peak_drive_qd_mps",
    "peak_drive_cmd_n",
    "peak_velocity_limit_violation_mps",
    "cube_peak_speed_mps",
)


@dataclass(frozen=True)
class SqueezeVariant:
    probe: str
    name: str
    mu: float
    pgs_iterations: int
    pgs_beta: float
    target_overlap_m: float
    grip_effort_limit_n: float = 40.0
    grip_drive_ke: float = 2.0e3
    grip_velocity_limit_mps: float = 1.0
    start_clearance_m: float = 0.025
    close_time_s: float = 0.35
    release_time_s: float = 0.50
    box_density_kg_m3: float = 90.0
    disturbance_accel_mps2: float = 0.0
    shake_hz: float = 0.0
    pgs_cfm: float = 1.0e-6


@dataclass(frozen=True)
class DriveVariant:
    name: str
    drive_ke: float
    effort_limit_n: float
    velocity_limit_mps: float
    pgs_iterations: int
    pgs_beta: float
    mu: float = 0.8
    pgs_cfm: float = 1.0e-6


SQUEEZE_VARIANTS = (
    SqueezeVariant(
        "squeeze_grasp",
        "baseline",
        mu=0.8,
        pgs_iterations=8,
        pgs_beta=0.10,
        target_overlap_m=0.030,
        grip_effort_limit_n=40.0,
    ),
    SqueezeVariant(
        "squeeze_grasp",
        "few_iterations",
        mu=0.8,
        pgs_iterations=2,
        pgs_beta=0.10,
        target_overlap_m=0.030,
        grip_effort_limit_n=40.0,
    ),
    SqueezeVariant(
        "squeeze_grasp",
        "weak_grip",
        mu=0.25,
        pgs_iterations=8,
        pgs_beta=0.10,
        target_overlap_m=0.030,
        grip_effort_limit_n=8.0,
    ),
)

DRIVE_VARIANTS = (
    DriveVariant("baseline", drive_ke=1.0e3, effort_limit_n=20.0, velocity_limit_mps=2.0, pgs_iterations=8, pgs_beta=0.10),
    DriveVariant("stiff_drive", drive_ke=1.0e4, effort_limit_n=100.0, velocity_limit_mps=5.0, pgs_iterations=8, pgs_beta=0.10),
    DriveVariant("low_effort", drive_ke=1.0e3, effort_limit_n=5.0, velocity_limit_mps=1.0, pgs_iterations=8, pgs_beta=0.10),
)

SHAKE_VARIANTS = (
    SqueezeVariant(
        "shaken_grasp",
        "baseline",
        mu=0.8,
        pgs_iterations=8,
        pgs_beta=0.10,
        target_overlap_m=0.030,
        grip_effort_limit_n=40.0,
        disturbance_accel_mps2=20.0,
        shake_hz=4.0,
    ),
    SqueezeVariant(
        "shaken_grasp",
        "low_friction",
        mu=0.25,
        pgs_iterations=8,
        pgs_beta=0.10,
        target_overlap_m=0.030,
        grip_effort_limit_n=16.0,
        disturbance_accel_mps2=45.0,
        shake_hz=4.0,
    ),
    SqueezeVariant(
        "shaken_grasp",
        "large_shake",
        mu=0.8,
        pgs_iterations=8,
        pgs_beta=0.10,
        target_overlap_m=0.030,
        grip_effort_limit_n=40.0,
        disturbance_accel_mps2=130.0,
        shake_hz=6.0,
    ),
)

SQUEEZE_SWEEP_VARIANTS = tuple(
    SqueezeVariant(
        "squeeze_sweep",
        f"mu{mu:.2f}_effort{effort_n:.0f}",
        mu=mu,
        pgs_iterations=8,
        pgs_beta=0.10,
        target_overlap_m=0.030,
        grip_effort_limit_n=effort_n,
        box_density_kg_m3=1.5 / (0.18**3),
    )
    for mu in (0.25, 0.50, 0.80)
    for effort_n in (8.0, 16.0, 32.0, 64.0)
)


def _box_tilt_deg(body_q_row: np.ndarray) -> float:
    qx, qy, _qz, _qw = body_q_row[3:7]
    local_up_dot_world_up = 1.0 - 2.0 * (qx * qx + qy * qy)
    local_up_dot_world_up = float(np.clip(local_up_dot_world_up, -1.0, 1.0))
    return math.degrees(math.acos(local_up_dot_world_up))


def _make_shape_cfg(mu: float, *, density: float) -> newton.ModelBuilder.ShapeConfig:
    cfg = newton.ModelBuilder.ShapeConfig(mu=mu)
    cfg.ke = 5.0e4
    cfg.kd = 5.0e2
    cfg.kf = 1.0e3
    cfg.density = density
    return cfg


def _make_solver(model: newton.Model, *, pgs_iterations: int, pgs_beta: float, pgs_cfm: float):
    return SolverFeatherPGS(
        model,
        update_mass_matrix_interval=1,
        pgs_mode="matrix_free",
        pgs_kernel="tiled_contact",
        pgs_iterations=pgs_iterations,
        pgs_beta=pgs_beta,
        pgs_cfm=pgs_cfm,
        pgs_omega=1.0,
        dense_max_constraints=64,
        mf_max_constraints=256,
        pgs_warmstart=False,
        enable_contact_friction=True,
        enable_joint_limits=True,
        enable_joint_velocity_limits=True,
        double_buffer=False,
        use_parallel_streams=False,
    )


def _contact_stats(contacts) -> tuple[int, float]:
    contact_count = int(contacts.rigid_contact_count.numpy()[0])
    peak_force = 0.0
    if contact_count > 0 and contacts.force is not None:
        forces = contacts.force.numpy()[:contact_count, :3]
        peak_force = float(np.linalg.norm(forces, axis=1).sum())
    return contact_count, peak_force


def _pad_box_contact_stats(contacts, *, pad_shapes: set[int], box_shape: int) -> tuple[int, float, float]:
    contact_count = int(contacts.rigid_contact_count.numpy()[0])
    if contact_count <= 0:
        return 0, 0.0, 0.0

    shape0 = contacts.rigid_contact_shape0.numpy()[:contact_count]
    shape1 = contacts.rigid_contact_shape1.numpy()[:contact_count]
    forces = contacts.rigid_contact_force.numpy()[:contact_count]
    pair_count = 0
    total_force = 0.0
    squeeze_force = 0.0
    for i, (raw_shape_a, raw_shape_b) in enumerate(zip(shape0, shape1, strict=True)):
        shape_a = int(raw_shape_a)
        shape_b = int(raw_shape_b)
        if not ((shape_a == box_shape and shape_b in pad_shapes) or (shape_b == box_shape and shape_a in pad_shapes)):
            continue
        force = forces[i]
        pair_count += 1
        total_force += float(np.linalg.norm(force))
        squeeze_force += abs(float(force[0]))
    return pair_count, total_force, squeeze_force


def _squeeze_geometry(variant: SqueezeVariant) -> tuple[float, float, float]:
    cube_hx = 0.09
    pad_radius = 0.11
    start_x = cube_hx + pad_radius + variant.start_clearance_m
    target_x = cube_hx + pad_radius - variant.target_overlap_m
    pad_z = 0.34
    return start_x, target_x, pad_z


def _build_squeeze_model(variant: SqueezeVariant):
    builder = newton.ModelBuilder(gravity=0.0)
    pad_cfg = _make_shape_cfg(variant.mu, density=600.0)
    box_cfg = _make_shape_cfg(variant.mu, density=variant.box_density_kg_m3)
    floor_cfg = _make_shape_cfg(variant.mu, density=0.0)
    start_x, _target_x, pad_z = _squeeze_geometry(variant)
    pad_drive_ke = variant.grip_drive_ke
    pad_drive_kd = 2.0 * math.sqrt(pad_drive_ke)

    left_body = builder.add_link(
        xform=wp.transform(wp.vec3(-start_x, 0.0, pad_z), wp.quat_identity()),
        label="left_pad",
    )
    left_shape = builder.add_shape_sphere(left_body, radius=0.11, cfg=pad_cfg, color=wp.vec3(0.15, 0.43, 0.85))
    left_joint = builder.add_joint_prismatic(
        parent=-1,
        child=left_body,
        parent_xform=wp.transform(wp.vec3(-start_x, 0.0, pad_z), wp.quat_identity()),
        child_xform=wp.transform_identity(),
        axis=wp.vec3(1.0, 0.0, 0.0),
        target_pos=0.0,
        target_ke=pad_drive_ke,
        target_kd=pad_drive_kd,
        effort_limit=variant.grip_effort_limit_n,
        velocity_limit=variant.grip_velocity_limit_mps,
        actuator_mode=JointTargetMode.POSITION,
        label="left_pad_x",
    )
    builder.add_articulation([left_joint], label="left_pad_articulation")

    right_body = builder.add_link(
        xform=wp.transform(wp.vec3(start_x, 0.0, pad_z), wp.quat_identity()),
        label="right_pad",
    )
    right_shape = builder.add_shape_sphere(right_body, radius=0.11, cfg=pad_cfg, color=wp.vec3(0.15, 0.43, 0.85))
    right_joint = builder.add_joint_prismatic(
        parent=-1,
        child=right_body,
        parent_xform=wp.transform(wp.vec3(start_x, 0.0, pad_z), wp.quat_identity()),
        child_xform=wp.transform_identity(),
        axis=wp.vec3(1.0, 0.0, 0.0),
        target_pos=0.0,
        target_ke=pad_drive_ke,
        target_kd=pad_drive_kd,
        effort_limit=variant.grip_effort_limit_n,
        velocity_limit=variant.grip_velocity_limit_mps,
        actuator_mode=JointTargetMode.POSITION,
        label="right_pad_x",
    )
    builder.add_articulation([right_joint], label="right_pad_articulation")

    box_body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, pad_z), wp.quat_identity()),
        label="grasped_box",
    )
    box_shape = builder.add_shape_box(
        box_body, hx=0.09, hy=0.09, hz=0.09, cfg=box_cfg, color=wp.vec3(0.95, 0.48, 0.16)
    )

    builder.add_shape_box(
        body=-1,
        xform=wp.transform(wp.vec3(0.0, 0.0, -0.04), wp.quat_identity()),
        hx=0.55,
        hy=0.35,
        hz=0.02,
        cfg=floor_cfg,
        color=wp.vec3(0.32, 0.33, 0.36),
    )

    model = builder.finalize()
    model.shape_margin.fill_(0.001)
    model.request_contact_attributes("force")
    return model, box_body, {left_shape, right_shape}, box_shape, left_joint, right_joint


def _start_viewer_video(
    model: newton.Model,
    path: Path,
    *,
    width: int,
    height: int,
    fps: float,
    camera_pos: tuple[float, float, float],
    camera_pitch: float,
    camera_yaw: float,
):
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required to write ViewerGL MP4 output")

    import newton.viewer  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    viewer = newton.viewer.ViewerGL(width=width, height=height, headless=True)
    viewer.set_model(model)
    viewer.set_camera(wp.vec3(*camera_pos), camera_pitch, camera_yaw)

    proc = subprocess.Popen(
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
            str(int(fps)),
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
            str(path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    frame_buf = wp.empty(shape=(height, width, 3), dtype=wp.uint8, device=wp.get_device())
    return viewer, proc, frame_buf


def _close_viewer(viewer) -> None:
    try:
        wp.synchronize()
    except Exception:
        pass
    try:
        invalidate = getattr(viewer, "_invalidate_pbo", None)
        if invalidate is not None:
            invalidate()
    except Exception:
        pass
    try:
        gc.collect()
    except Exception:
        pass
    try:
        viewer.close()
    except Exception:
        pass
    try:
        gc.collect()
    except Exception:
        pass


def run_squeeze_variant(
    variant: SqueezeVariant,
    *,
    frames: int,
    substeps: int,
    fps: float,
    render_path: Path | None = None,
    render_width: int = 960,
    render_height: int = 540,
) -> dict[str, object]:
    model, box_body, pad_shapes, box_shape, left_joint, right_joint = _build_squeeze_model(variant)
    solver = _make_solver(
        model,
        pgs_iterations=variant.pgs_iterations,
        pgs_beta=variant.pgs_beta,
        pgs_cfm=variant.pgs_cfm,
    )

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    dt = 1.0 / fps / substeps
    frame_dt = 1.0 / fps
    time_s = 0.0
    left_q_start = int(model.joint_q_start.numpy()[left_joint])
    right_q_start = int(model.joint_q_start.numpy()[right_joint])
    left_qd_start = int(model.joint_qd_start.numpy()[left_joint])
    right_qd_start = int(model.joint_qd_start.numpy()[right_joint])
    start_x, target_x, pad_z = _squeeze_geometry(variant)
    left_final_target = start_x - target_x
    right_final_target = target_x - start_x
    pad_drive_kd = 2.0 * math.sqrt(variant.grip_drive_ke)
    target_pos_np = control.joint_target_pos.numpy()

    viewer = None
    ffmpeg_proc = None
    frame_buf = None
    if render_path is not None:
        viewer, ffmpeg_proc, frame_buf = _start_viewer_video(
            model,
            render_path,
            width=render_width,
            height=render_height,
            fps=fps,
            camera_pos=(0.95, -1.05, 0.65),
            camera_pitch=-18.0,
            camera_yaw=138.0,
        )

    peak_speed = 0.0
    peak_omega = 0.0
    peak_contacts = 0
    peak_contact_force = 0.0
    peak_squeeze_force = 0.0
    peak_grip_cmd = 0.0
    settle_grip_cmd_sum = 0.0
    settle_grip_cmd_count = 0
    settle_squeeze_force_sum = 0.0
    settle_squeeze_force_count = 0
    max_slip = 0.0
    fall_m = 0.0
    box_mass = float(model.body_mass.numpy()[box_body])

    try:
        for _frame in range(frames):
            for substep in range(substeps):
                substep_time = time_s + substep * dt
                close_alpha = min(substep_time / variant.close_time_s, 1.0)
                target_pos_np[left_q_start] = left_final_target * close_alpha
                target_pos_np[right_q_start] = right_final_target * close_alpha
                control.joint_target_pos.assign(target_pos_np)
                state_0.clear_forces()
                accel_z = -9.81 if substep_time > variant.release_time_s else 0.0
                if variant.disturbance_accel_mps2 > 0.0 and substep_time > variant.release_time_s:
                    phase_t = substep_time - variant.release_time_s
                    accel_z += variant.disturbance_accel_mps2 * math.sin(2.0 * math.pi * variant.shake_hz * phase_t)
                if accel_z != 0.0:
                    body_f = state_0.body_f.numpy()
                    body_f[box_body, 2] += box_mass * accel_z
                    state_0.body_f.assign(body_f)
                model.collide(state_0, contacts)
                solver.step(state_0, state_1, control, contacts, dt)
                state_0, state_1 = state_1, state_0

            time_s += frame_dt
            solver.update_contacts(contacts)
            body_q = state_0.body_q.numpy()
            body_qd = state_0.body_qd.numpy()
            joint_q = state_0.joint_q.numpy()
            joint_qd = state_0.joint_qd.numpy()
            box_q = body_q[box_body]
            box_qd = body_qd[box_body]
            left_target = float(target_pos_np[left_q_start])
            right_target = float(target_pos_np[right_q_start])
            left_cmd = variant.grip_drive_ke * (left_target - float(joint_q[left_q_start])) - pad_drive_kd * float(
                joint_qd[left_qd_start]
            )
            right_cmd = variant.grip_drive_ke * (right_target - float(joint_q[right_q_start])) - pad_drive_kd * float(
                joint_qd[right_qd_start]
            )
            left_cmd = float(np.clip(left_cmd, -variant.grip_effort_limit_n, variant.grip_effort_limit_n))
            right_cmd = float(np.clip(right_cmd, -variant.grip_effort_limit_n, variant.grip_effort_limit_n))
            grip_cmd = abs(left_cmd) + abs(right_cmd)

            speed = float(np.linalg.norm(box_qd[:3]))
            omega = float(np.linalg.norm(box_qd[3:6]))
            peak_speed = max(peak_speed, speed)
            peak_omega = max(peak_omega, omega)
            contact_count, contact_force = _contact_stats(contacts)
            _pad_pair_count, _pad_total_force, squeeze_force = _pad_box_contact_stats(
                contacts,
                pad_shapes=pad_shapes,
                box_shape=box_shape,
            )
            peak_contacts = max(peak_contacts, contact_count)
            peak_contact_force = max(peak_contact_force, contact_force)
            peak_squeeze_force = max(peak_squeeze_force, squeeze_force)
            peak_grip_cmd = max(peak_grip_cmd, grip_cmd)
            if time_s > variant.release_time_s + 0.15:
                settle_grip_cmd_sum += grip_cmd
                settle_grip_cmd_count += 1
                settle_squeeze_force_sum += squeeze_force
                settle_squeeze_force_count += 1
            max_slip = max(max_slip, abs(float(box_q[2]) - pad_z))
            fall_m = max(fall_m, 0.0, pad_z - float(box_q[2]))

            if viewer is not None and ffmpeg_proc is not None and frame_buf is not None:
                viewer.begin_frame(time_s)
                viewer.log_state(state_0)
                viewer.log_contacts(contacts, state_0)
                viewer.end_frame()
                frame_buf = viewer.get_frame(target_image=frame_buf)
                assert ffmpeg_proc.stdin is not None
                ffmpeg_proc.stdin.write(frame_buf.numpy().tobytes())
    finally:
        if ffmpeg_proc is not None:
            assert ffmpeg_proc.stdin is not None
            ffmpeg_proc.stdin.close()
            ffmpeg_proc.wait()
        if viewer is not None:
            _close_viewer(viewer)

    body_q = state_0.body_q.numpy()
    box_q = body_q[box_body]
    retained = bool(abs(float(box_q[0])) < 0.06 and max_slip < 0.08 and float(box_q[2]) > 0.22)
    mean_squeeze_force = settle_squeeze_force_sum / settle_squeeze_force_count if settle_squeeze_force_count else 0.0
    mean_grip_cmd = settle_grip_cmd_sum / settle_grip_cmd_count if settle_grip_cmd_count else 0.0

    return {
        "probe": variant.probe,
        "variant": variant.name,
        "result": "retained" if retained else "dropped",
        "mu": variant.mu,
        "pgs_iterations": variant.pgs_iterations,
        "pgs_beta": variant.pgs_beta,
        "pgs_cfm": variant.pgs_cfm,
        "frames": frames,
        "substeps": substeps,
        "dt_s": dt,
        "peak_contacts": peak_contacts,
        "peak_contact_force_n": peak_contact_force,
        "peak_box_speed_mps": peak_speed,
        "peak_box_omega_radps": peak_omega,
        "final_x_m": float(box_q[0]),
        "final_y_m": float(box_q[1]),
        "final_z_m": float(box_q[2]),
        "final_tilt_deg": _box_tilt_deg(box_q),
        "slip_m": max_slip,
        "fall_m": fall_m,
        "target_overlap_m": variant.target_overlap_m,
        "start_clearance_m": variant.start_clearance_m,
        "close_time_s": variant.close_time_s,
        "release_time_s": variant.release_time_s,
        "box_mass_kg": box_mass,
        "box_density_kg_m3": variant.box_density_kg_m3,
        "grip_drive_ke": variant.grip_drive_ke,
        "grip_effort_limit_n": variant.grip_effort_limit_n,
        "grip_velocity_limit_mps": variant.grip_velocity_limit_mps,
        "peak_grip_cmd_n": peak_grip_cmd,
        "mean_grip_cmd_n": mean_grip_cmd,
        "mean_squeeze_force_n": mean_squeeze_force,
        "peak_squeeze_force_n": peak_squeeze_force,
        "disturbance_accel_mps2": variant.disturbance_accel_mps2,
        "shake_hz": variant.shake_hz,
    }


def _build_drive_model(variant: DriveVariant):
    builder = newton.ModelBuilder(gravity=0.0)
    wall_cfg = _make_shape_cfg(variant.mu, density=0.0)
    cube_cfg = _make_shape_cfg(variant.mu, density=180.0)
    pusher_cfg = _make_shape_cfg(variant.mu, density=500.0)

    builder.add_shape_box(
        body=-1,
        xform=wp.transform(wp.vec3(0.32, 0.0, 0.16), wp.quat_identity()),
        hx=0.025,
        hy=0.18,
        hz=0.16,
        cfg=wall_cfg,
        color=wp.vec3(0.36, 0.37, 0.40),
    )
    cube_body = builder.add_body(
        xform=wp.transform(wp.vec3(0.13, 0.0, 0.16), wp.quat_identity()),
        label="pushed_cube",
    )
    builder.add_shape_box(cube_body, hx=0.06, hy=0.06, hz=0.06, cfg=cube_cfg, color=wp.vec3(0.95, 0.48, 0.16))

    pusher_body = builder.add_link(
        xform=wp.transform(wp.vec3(-0.18, 0.0, 0.16), wp.quat_identity()),
        label="drive_pusher",
    )
    builder.add_shape_box(pusher_body, hx=0.05, hy=0.08, hz=0.08, cfg=pusher_cfg, color=wp.vec3(0.15, 0.43, 0.85))
    pusher_joint = builder.add_joint_prismatic(
        parent=-1,
        child=pusher_body,
        parent_xform=wp.transform(wp.vec3(-0.18, 0.0, 0.16), wp.quat_identity()),
        child_xform=wp.transform_identity(),
        axis=wp.vec3(1.0, 0.0, 0.0),
        target_pos=0.35,
        target_ke=variant.drive_ke,
        target_kd=2.0 * math.sqrt(variant.drive_ke),
        effort_limit=variant.effort_limit_n,
        velocity_limit=variant.velocity_limit_mps,
        actuator_mode=JointTargetMode.POSITION,
        label="pusher_x",
    )
    builder.add_articulation([pusher_joint], label="pusher_articulation")

    model = builder.finalize()
    model.shape_margin.fill_(0.001)
    model.request_contact_attributes("force")
    return model, cube_body, pusher_joint


def run_drive_variant(
    variant: DriveVariant,
    *,
    frames: int,
    substeps: int,
    fps: float,
    render_path: Path | None = None,
    render_width: int = 960,
    render_height: int = 540,
) -> dict[str, object]:
    model, cube_body, pusher_joint = _build_drive_model(variant)
    solver = _make_solver(
        model,
        pgs_iterations=variant.pgs_iterations,
        pgs_beta=variant.pgs_beta,
        pgs_cfm=variant.pgs_cfm,
    )

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    q_start = int(model.joint_q_start.numpy()[pusher_joint])
    qd_start = int(model.joint_qd_start.numpy()[pusher_joint])
    dt = 1.0 / fps / substeps
    frame_dt = 1.0 / fps
    time_s = 0.0

    viewer = None
    ffmpeg_proc = None
    frame_buf = None
    if render_path is not None:
        viewer, ffmpeg_proc, frame_buf = _start_viewer_video(
            model,
            render_path,
            width=render_width,
            height=render_height,
            fps=fps,
            camera_pos=(0.78, -0.85, 0.46),
            camera_pitch=-20.0,
            camera_yaw=132.0,
        )

    peak_contacts = 0
    peak_contact_force = 0.0
    peak_cube_speed = 0.0
    peak_drive_qd = 0.0
    peak_drive_cmd = 0.0
    peak_vlim_violation = 0.0
    kd = 2.0 * math.sqrt(variant.drive_ke)
    target = 0.35

    try:
        for _frame in range(frames):
            for _ in range(substeps):
                state_0.clear_forces()
                model.collide(state_0, contacts)
                solver.step(state_0, state_1, control, contacts, dt)
                state_0, state_1 = state_1, state_0

            time_s += frame_dt
            solver.update_contacts(contacts)
            body_qd = state_0.body_qd.numpy()
            joint_q = state_0.joint_q.numpy()
            joint_qd = state_0.joint_qd.numpy()
            cube_speed = float(np.linalg.norm(body_qd[cube_body, :3]))
            drive_q = float(joint_q[q_start])
            drive_qd = float(joint_qd[qd_start])
            cmd = variant.drive_ke * (target - drive_q) - kd * drive_qd
            cmd = float(np.clip(cmd, -variant.effort_limit_n, variant.effort_limit_n))

            peak_cube_speed = max(peak_cube_speed, cube_speed)
            peak_drive_qd = max(peak_drive_qd, abs(drive_qd))
            peak_drive_cmd = max(peak_drive_cmd, abs(cmd))
            peak_vlim_violation = max(peak_vlim_violation, 0.0, abs(drive_qd) - variant.velocity_limit_mps)
            contact_count, contact_force = _contact_stats(contacts)
            peak_contacts = max(peak_contacts, contact_count)
            peak_contact_force = max(peak_contact_force, contact_force)

            if viewer is not None and ffmpeg_proc is not None and frame_buf is not None:
                viewer.begin_frame(time_s)
                viewer.log_state(state_0)
                viewer.log_contacts(contacts, state_0)
                viewer.end_frame()
                frame_buf = viewer.get_frame(target_image=frame_buf)
                assert ffmpeg_proc.stdin is not None
                ffmpeg_proc.stdin.write(frame_buf.numpy().tobytes())
    finally:
        if ffmpeg_proc is not None:
            assert ffmpeg_proc.stdin is not None
            ffmpeg_proc.stdin.close()
            ffmpeg_proc.wait()
        if viewer is not None:
            _close_viewer(viewer)

    body_q = state_0.body_q.numpy()
    cube_q = body_q[cube_body]
    stable = bool(peak_vlim_violation < 0.05 and peak_cube_speed < 5.0)
    return {
        "probe": "drive_into_contact",
        "variant": variant.name,
        "result": "bounded" if stable else "spike",
        "mu": variant.mu,
        "pgs_iterations": variant.pgs_iterations,
        "pgs_beta": variant.pgs_beta,
        "pgs_cfm": variant.pgs_cfm,
        "frames": frames,
        "substeps": substeps,
        "dt_s": dt,
        "peak_contacts": peak_contacts,
        "peak_contact_force_n": peak_contact_force,
        "peak_box_speed_mps": peak_cube_speed,
        "peak_box_omega_radps": float(np.linalg.norm(state_0.body_qd.numpy()[cube_body, 3:6])),
        "final_x_m": float(cube_q[0]),
        "final_y_m": float(cube_q[1]),
        "final_z_m": float(cube_q[2]),
        "final_tilt_deg": _box_tilt_deg(cube_q),
        "drive_ke": variant.drive_ke,
        "drive_effort_limit_n": variant.effort_limit_n,
        "drive_velocity_limit_mps": variant.velocity_limit_mps,
        "peak_drive_qd_mps": peak_drive_qd,
        "peak_drive_cmd_n": peak_drive_cmd,
        "peak_velocity_limit_violation_mps": peak_vlim_violation,
        "cube_peak_speed_mps": peak_cube_speed,
    }


def write_results(rows: list[dict[str, object]], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metrics.csv"
    json_path = output_dir / "metrics.json"
    normalized_rows = [{field: row.get(field, "") for field in CSV_FIELDS} for row in rows]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(CSV_FIELDS), lineterminator="\n")
        writer.writeheader()
        writer.writerows(normalized_rows)
    json_path.write_text(json.dumps(normalized_rows, indent=2) + "\n")
    return csv_path, json_path


def write_squeeze_sweep_svg(rows: list[dict[str, object]], output_dir: Path) -> Path | None:
    sweep_rows = [row for row in rows if row.get("probe") == "squeeze_sweep"]
    if not sweep_rows:
        return None

    width = 720
    height = 390
    left = 70
    right = 26
    top = 26
    bottom = 58
    plot_w = width - left - right
    plot_h = height - top - bottom
    max_force = max(float(row.get("peak_squeeze_force_n") or 0.0) for row in sweep_rows)
    max_fall = max(float(row.get("fall_m") or 0.0) for row in sweep_rows)
    x_max = max(10.0, max_force * 1.15)
    y_max = max(0.05, max_fall * 1.15)

    def sx(force: float) -> float:
        return left + plot_w * force / x_max

    def sy(fall: float) -> float:
        return top + plot_h * (1.0 - fall / y_max)

    elems = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        'role="img" aria-label="Squeeze force versus fall distance">',
        '<rect width="720" height="390" fill="#ffffff"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222" stroke-width="1.2"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#222" stroke-width="1.2"/>',
    ]

    for i in range(5):
        y_val = y_max * i / 4.0
        y = sy(y_val)
        elems.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        elems.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="12" fill="#374151">{y_val:.2f}</text>'
        )
    for i in range(5):
        x_val = x_max * i / 4.0
        x = sx(x_val)
        elems.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h}" stroke="#f1f5f9"/>')
        elems.append(
            f'<text x="{x:.1f}" y="{top + plot_h + 20}" text-anchor="middle" font-size="12" fill="#374151">{x_val:.0f}</text>'
        )

    elems.append(
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 16}" text-anchor="middle" font-size="13" fill="#111827">'
        "peak pad-to-box squeeze force (N)</text>"
    )
    elems.append(
        f'<text x="18" y="{top + plot_h / 2:.1f}" transform="rotate(-90 18 {top + plot_h / 2:.1f})" '
        'text-anchor="middle" font-size="13" fill="#111827">max fall distance (m)</text>'
    )

    for row in sweep_rows:
        force = float(row.get("peak_squeeze_force_n") or 0.0)
        fall = float(row.get("fall_m") or 0.0)
        mu = float(row.get("mu") or 0.0)
        effort = float(row.get("grip_effort_limit_n") or 0.0)
        retained = row.get("result") == "retained"
        color = "#0f766e" if retained else "#c2410c"
        label = f"mu={mu:.2f}, cap={effort:.0f}"
        x = sx(force)
        y = sy(fall)
        elems.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}" opacity="0.88"/>')
        elems.append(f'<text x="{x + 8:.1f}" y="{y - 7:.1f}" font-size="11" fill="#111827">{label}</text>')

    elems.append('<circle cx="555" cy="28" r="5" fill="#0f766e"/><text x="566" y="32" font-size="12">retained</text>')
    elems.append('<circle cx="635" cy="28" r="5" fill="#c2410c"/><text x="646" y="32" font-size="12">dropped</text>')
    elems.append("</svg>")

    path = output_dir / "squeeze_force_vs_fall.svg"
    path.write_text("\n".join(elems) + "\n")
    return path


def render_single_video(
    video_name: str,
    *,
    output_dir: Path,
    frames: int,
    substeps: int,
    fps: float,
    render_width: int,
    render_height: int,
) -> Path:
    if video_name == "squeeze":
        path = output_dir / "squeeze_baseline.mp4"
        run_squeeze_variant(
            SQUEEZE_VARIANTS[0],
            frames=frames,
            substeps=substeps,
            fps=fps,
            render_path=path,
            render_width=render_width,
            render_height=render_height,
        )
    elif video_name == "drive":
        path = output_dir / "drive_stiff_drive.mp4"
        run_drive_variant(
            DRIVE_VARIANTS[1],
            frames=frames,
            substeps=substeps,
            fps=fps,
            render_path=path,
            render_width=render_width,
            render_height=render_height,
        )
    elif video_name == "shaken":
        path = output_dir / "shaken_large_shake.mp4"
        run_squeeze_variant(
            SHAKE_VARIANTS[2],
            frames=frames,
            substeps=substeps,
            fps=fps,
            render_path=path,
            render_width=render_width,
            render_height=render_height,
        )
    else:
        raise ValueError(f"unknown video name: {video_name}")
    return path


def render_videos_in_fresh_processes(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    script_path = Path(__file__).resolve()
    for video_name in ("squeeze", "drive", "shaken"):
        subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--output-dir",
                str(args.output_dir),
                "--frames",
                str(args.frames),
                "--fps",
                str(args.fps),
                "--substeps",
                str(args.substeps),
                "--render-width",
                str(args.render_width),
                "--render-height",
                str(args.render_height),
                "--video-only",
                video_name,
            ],
            check=True,
        )
        paths.append(args.output_dir / {
            "squeeze": "squeeze_baseline.mp4",
            "drive": "drive_stiff_drive.mp4",
            "shaken": "shaken_large_shake.mp4",
        }[video_name])
    return paths


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("contact_stability_probe"))
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--render-height", type=int, default=540)
    parser.add_argument("--no-video", action="store_true", help="Skip ViewerGL MP4 generation.")
    parser.add_argument("--video-only", choices=("squeeze", "drive", "shaken"), help=argparse.SUPPRESS)
    return parser


def main():
    args = create_parser().parse_args()
    if args.frames <= 0:
        raise ValueError("--frames must be positive")
    if args.substeps <= 0:
        raise ValueError("--substeps must be positive")
    if args.video_only is not None:
        path = render_single_video(
            args.video_only,
            output_dir=args.output_dir,
            frames=args.frames,
            substeps=args.substeps,
            fps=args.fps,
            render_width=args.render_width,
            render_height=args.render_height,
        )
        print(f"Wrote {path}")
        return

    rows: list[dict[str, object]] = []
    for variant in SQUEEZE_VARIANTS:
        rows.append(
            run_squeeze_variant(
                variant,
                frames=args.frames,
                substeps=args.substeps,
                fps=args.fps,
                render_width=args.render_width,
                render_height=args.render_height,
            )
        )
    for variant in DRIVE_VARIANTS:
        rows.append(
            run_drive_variant(
                variant,
                frames=args.frames,
                substeps=args.substeps,
                fps=args.fps,
                render_width=args.render_width,
                render_height=args.render_height,
            )
        )
    for variant in SHAKE_VARIANTS:
        rows.append(
            run_squeeze_variant(
                variant,
                frames=args.frames,
                substeps=args.substeps,
                fps=args.fps,
                render_width=args.render_width,
                render_height=args.render_height,
            )
        )
    for variant in SQUEEZE_SWEEP_VARIANTS:
        rows.append(run_squeeze_variant(variant, frames=args.frames, substeps=args.substeps, fps=args.fps))

    csv_path, json_path = write_results(rows, args.output_dir)
    svg_path = write_squeeze_sweep_svg(rows, args.output_dir)
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    if svg_path is not None:
        print(f"Wrote {svg_path}")
    if not args.no_video:
        for video_path in render_videos_in_fresh_processes(args):
            print(f"Wrote {video_path}")

    for row in rows:
        force = row.get("peak_squeeze_force_n") if row.get("probe") in {"squeeze_grasp", "squeeze_sweep", "shaken_grasp"} else row.get("peak_contact_force_n")
        print(
            f"{row['probe']}/{row['variant']}: result={row['result']} "
            f"peak_speed={float(row.get('peak_box_speed_mps') or 0.0):.3f} "
            f"peak_force={float(force or 0.0):.1f}"
        )


if __name__ == "__main__":
    main()
