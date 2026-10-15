# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Two fixed-aperture gripper tips reorient a free box through friction."""

import math

import numpy as np
import warp as wp
from robots import make_arm

import newton


def smooth(t, start, end):
    u = np.clip((t - start) / (end - start), 0, 1)
    return u * u * (3 - 2 * u)


def populate(b, info, variant, assets, device):
    from scene import plan_ik, tf  # noqa: PLC0415
    from scipy.spatial.transform import Rotation

    half = np.array([0.09, 0.075, 0.12])
    builders, paths = [], []
    for side in (-1, 1):
        arm = newton.ModelBuilder()
        arm.default_shape_cfg.gap = 0.0002
        ee, _, fingers = make_arm(arm, assets, device, robot="ur10", root=(-0.60, -0.35, 0))
        # Choose the equivalent wrist coordinate with room for the whole turn.
        # The shared planner still enforces position, orientation and speed limits.
        arm.joint_q[4] += math.tau
        base_rotation = Rotation.from_euler("y", -side * math.pi / 2) * Rotation.from_euler(
            "z", math.pi if side == -1 else 0
        )
        frame = Rotation.from_euler("z", 0 if side == -1 else math.pi)
        poses = []
        for t in np.linspace(0, 12, 121):
            roll = smooth(t, 3, 5.5) * math.pi / 2
            yaw = smooth(t, 6, 8.5) * math.pi / 2
            rotation = Rotation.from_euler("z", yaw) * Rotation.from_euler("x", roll)
            height = 0.121 + 0.20 * smooth(t, 1.5, 3) - 0.245 * smooth(t, 9, 10.5) + 0.18 * smooth(t, 10.7, 11.8)
            center = np.array([0, -0.12, height])
            distance = half[0] + 0.055 * (1 - smooth(t, 0, 1.5)) - 0.0025 + 0.09 * smooth(t, 10.7, 11.8)
            target = center + rotation.apply([side * distance, 0, 0])
            q = (frame.inv() * rotation * base_rotation).as_quat()
            target = frame.inv().apply(target)
            poses.append((float(t), target, 0.012, wp.quat(*q)))
        for i in range(6):
            arm.joint_target_ke[i], arm.joint_target_kd[i], arm.joint_effort_limit[i] = 8000, 180, 180
        for i in fingers:
            arm.joint_target_ke[i], arm.joint_target_kd[i], arm.joint_effort_limit[i] = 1500, 30, 30
        for si, owner in enumerate(arm.shape_body):
            if owner >= 0 and arm.body_label[owner].endswith(("/fr3_leftfinger", "/fr3_rightfinger")):
                arm.shape_material_mu[si] = 1.2
        paths.append(
            plan_ik(
                arm,
                ee,
                (0, 0, 0.1034),
                poses,
                device,
                desired_rotation=wp.quat(*(frame.inv() * base_rotation).as_quat()),
            )
        )
        builders.append(arm)
    times = sorted({t for path in paths for t, _ in path})
    values = []
    for path in paths:
        knots = np.array([t for t, _ in path])
        joints = np.stack([q for _, q in path])
        values.append(np.stack([np.interp(times, knots, joints[:, i]) for i in range(joints.shape[1])], axis=1))
    info["waypoints"] = list(zip(times, np.concatenate(values, axis=1), strict=True))
    for index, arm in enumerate(builders):
        b.add_builder(arm, xform=tf(q=wp.quat_rpy(0.0, 0.0, index * math.pi)), label_prefix=f"box_arm_{index}")
    body = b.add_body(xform=tf((0, -0.12, 0.121)), label="turn_box")
    cfg = b.default_shape_cfg.copy()
    cfg.density, cfg.mu, cfg.gap, cfg.restitution = 65, 1.0, 0.0002, 0
    b.add_shape_box(body, hx=half[0], hy=half[1], hz=half[2], cfg=cfg, color=(0.57, 0.43, 0.29), label="cardboard_box")
    visual = cfg.copy()
    visual.density = 0
    visual.flags = newton.ShapeFlags.VISIBLE
    for p, size, color in (
        ((0, 0, 0.1203), (0.016, 0.075, 0.0003), (0.15, 0.45, 0.62)),
        ((0, -0.0753, 0.02), (0.055, 0.0003, 0.035), (0.88, 0.87, 0.80)),
        ((0, -0.0757, 0.02), (0.028, 0.0003, 0.008), (0.12, 0.42, 0.57)),
    ):
        b.add_shape_box(
            body, xform=tf(p), hx=size[0], hy=size[1], hz=size[2], cfg=visual, color=color, label="box_marking"
        )
    info.update(
        robot="ur10+ur10",
        tracked_body=body,
        dense_waypoints=True,
        phase_delay=0,
        task="Touch opposite faces with fixed-aperture tips, flip 90 degrees and turn 90 degrees",
    )


def check(world, poses, fps):
    from scipy.spatial.transform import Rotation

    body = poses[:, world["body_start"] + world["tracked_body"]]
    rotation = Rotation.from_quat(body[:, 3:])
    flip_goal = Rotation.from_euler("x", math.pi / 2)
    final_goal = Rotation.from_euler("z", math.pi / 2) * flip_goal
    flip_error = float((flip_goal.inv() * rotation[min(len(body) - 1, round(5.8 * fps))]).magnitude())
    final_error = float((final_goal.inv() * rotation[-1]).magnitude())
    resting_motion = float(np.linalg.norm(body[-fps:, :3] - body[-1, :3], axis=1).max())
    result = {
        "flip_error_deg": math.degrees(flip_error),
        "final_error_deg": math.degrees(final_error),
        "resting_motion_m": resting_motion,
        "final_height_m": float(body[-1, 2]),
    }
    result["pass"] = bool(
        flip_error < 0.15 and final_error < 0.15 and resting_motion < 0.004 and abs(body[-1, 2] - 0.075) < 0.005
    )
    return result
