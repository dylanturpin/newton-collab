# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Responsive yo-yo using a compliant, no-slip string/axle coupling.

The reduced winding model assumes a constant axle radius and negligible string
mass. Extension energy couples translation to axle rotation; it is not a
prescribed yo-yo trajectory. The cord is visual-only geometry, drawn from the simulated available length.
"""

import math
from pathlib import Path

import numpy as np
import warp as wp

import newton


@wp.kernel
def string_force(
    poses: wp.array[wp.transform],
    velocities: wp.array[wp.spatial_vector],
    forces: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    winding: wp.array[float],
    metrics: wp.array[float],
    hand: int,
    toy: int,
    anchor_local: wp.vec3,
    radius: float,
    maximum_length: float,
    dt: float,
):
    hp = wp.transform_point(poses[hand], anchor_local)
    yp = wp.transform_get_translation(poses[toy])
    axis = wp.quat_rotate(wp.transform_get_rotation(poses[toy]), wp.vec3(0.0, 1.0, 0.0))
    omega = wp.dot(wp.spatial_bottom(velocities[toy]) - wp.spatial_bottom(velocities[hand]), axis)
    angle = winding[0] + omega * dt
    winding[0] = angle
    smooth = wp.sqrt(angle * angle + 0.15 * 0.15)
    length = maximum_length - radius * smooth
    derivative = -radius * angle / smooth
    delta = hp - yp
    distance = wp.length(delta)
    direction = delta / wp.max(distance, 1.0e-6)
    hand_offset = hp - wp.transform_point(poses[hand], body_com[hand])
    hv = wp.spatial_top(velocities[hand]) + wp.cross(wp.spatial_bottom(velocities[hand]), hand_offset)
    stretch_rate = wp.dot(hv - wp.spatial_top(velocities[toy]), direction) - derivative * omega
    tension = float(0.0)
    if distance > length:
        tension = wp.max(0.0, 250.0 * (distance - length) + 0.3 * stretch_rate)
    force = direction * tension
    torque = axis * (tension * derivative - 0.0000005 * omega)
    wp.atomic_add(forces, toy, wp.spatial_vector(force, torque))
    wp.atomic_add(forces, hand, wp.spatial_vector(-force, wp.cross(hand_offset, -force) - torque))
    metrics[0] = tension
    metrics[1] = distance
    metrics[2] = length
    metrics[3] = omega


def populate(b, info, variant, assets, device):
    from scene import tf  # noqa: PLC0415 -- scene helpers

    carrier = b.add_link(mass=0.1, inertia=wp.mat33(np.eye(3, dtype=np.float32) * 0.001), label="yoyo/wrist-carrier")
    joint = b.add_joint_prismatic(
        -1,
        carrier,
        axis=(0, 0, 1),
        limit_lower=-0.2,
        limit_upper=0.2,
        target_ke=8000,
        target_kd=150,
        effort_limit=300,
        label="yoyo/wrist-lift",
    )
    retreat = b.add_link(mass=0.1, inertia=wp.mat33(np.eye(3, dtype=np.float32) * 0.001), label="yoyo/wrist-retreat")
    slide = b.add_joint_prismatic(
        carrier,
        retreat,
        axis=(0, 1, 0),
        limit_lower=-0.15,
        limit_upper=0.15,
        target_ke=8000,
        target_kd=150,
        effort_limit=300,
        label="yoyo/wrist-slide",
    )
    b.add_articulation([joint, slide])
    palm_position = np.array([0.18, -0.08, 1.0])
    palm_rotation = wp.quat_rpy(math.pi / 2, 0.0, 0.0)
    root_rotation = (
        palm_rotation * wp.quat_rpy(0.0, 0.0, -math.pi / 2) * wp.quat_inverse(wp.normalize(wp.quat(1, 0, 1, 0)))
    )
    asset = Path(assets["wonik_allegro"]).parent / "shadow_hand/right_hand.xml"
    b.add_mjcf(
        str(asset),
        xform=tf(
            palm_position - np.asarray(wp.quat_rotate(palm_rotation, wp.vec3(0.0, -0.01, 0.2470098))), root_rotation
        ),
        parent_body=retreat,
        floating=False,
        enable_self_collisions=False,
    )
    values = {
        f"rh_{finger}J{joint}": angle
        for finger in ("FF", "MF", "RF", "LF")
        for joint, angle in ((3, 1.0), (2, 1.2), (1, 0.5))
    }
    values.update(rh_THJ5=-1.0)
    for j, label in enumerate(b.joint_label):
        if j in (joint, slide) or b.joint_type[j] == newton.JointType.FIXED:
            continue
        qi, vi = b.joint_q_start[j], b.joint_qd_start[j]
        short = label.rsplit("/", 1)[-1]
        b.joint_q[qi] = values.get(short, 0)
        b.joint_target_ke[vi], b.joint_target_kd[vi] = 40, 1.2
        b.joint_effort_limit[vi] = 3
        b.joint_target_mode[vi] = int(newton.JointTargetMode.POSITION)
        b.joint_armature[vi] = 0.001
    b.approximate_meshes("convex_hull", keep_visual_shapes=True)
    palm = next(i for i, label in enumerate(b.body_label) if label.endswith("rh_mfknuckle"))
    closed = np.array(b.joint_q).copy()
    opened = closed.copy()
    for j, label in enumerate(b.joint_label):
        if any(f"rh_{f}J" in label for f in ("FF", "MF", "RF", "LF")):
            opened[b.joint_q_start[j]] = 0.0
    points = []
    for t, z, pose in [
        (0, 0, closed),
        (1, 0, closed),
        (1.8, 0, opened),
        (2.15, 0, opened),
        (2.35, 0.07, opened),
        (2.8, 0.07, opened),
        (8, 0, opened),
    ]:
        q = pose.copy()
        q[0] = z
        points.append((t, q))
    info["waypoints"] = points
    b.joint_target_q[:] = closed
    center = np.add(palm_position, [0.011, -0.085, -0.055])
    toy = b.add_body(
        xform=tf(center, wp.quat_rpy(0.0, 0.0, -math.pi / 2)),
        mass=0.065,
        inertia=wp.mat33(0.000035, 0, 0, 0, 0.000064, 0, 0, 0, 0.000035),
        lock_inertia=True,
        label="yoyo/toy",
    )
    cfg = b.default_shape_cfg.copy()
    cfg.density = 0
    cfg.mu = 0.65
    cfg.gap = 0.00015
    axis = wp.quat_rpy(math.pi / 2, 0.0, 0.0)
    for side in [-1, 1]:
        b.add_shape_cylinder(
            toy,
            xform=tf((0, side * 0.015, 0), axis),
            radius=0.042,
            half_height=0.010,
            cfg=cfg,
            color=(0.035, 0.43, 0.78),
            label="yoyo/disc",
        )
        b.add_shape_cylinder(
            toy,
            xform=tf((0, side * 0.0255, 0), axis),
            radius=0.030,
            half_height=0.001,
            cfg=cfg,
            color=(0.02, 0.65, 0.47),
            label="yoyo/inset",
        )
        b.add_shape_cylinder(
            toy,
            xform=tf((0, side * 0.027, 0), axis),
            radius=0.009,
            half_height=0.0015,
            cfg=cfg,
            color=(0.96, 0.61, 0.065),
            label="yoyo/hub",
        )
        b.add_shape_box(
            toy,
            xform=tf((0.022, side * 0.027, 0)),
            hx=0.010,
            hy=0.001,
            hz=0.002,
            cfg=cfg,
            color=(0.96, 0.61, 0.065),
            label="yoyo/spin-marker",
        )
    b.add_shape_cylinder(
        toy,
        xform=tf((0, 0, 0), axis),
        radius=0.012,
        half_height=0.006,
        cfg=cfg,
        color=(0.5, 0.53, 0.55),
        label="yoyo/axle",
    )
    visual = cfg.copy()
    visual.has_shape_collision = visual.has_particle_collision = False
    for i in range(24):
        a = i * math.tau / 24
        b_angle = (i + 1) * math.tau / 24
        first = np.array([0.011 * math.cos(a), 0.011 * math.sin(a), 0])
        second = np.array([0.011 * math.cos(b_angle), 0.011 * math.sin(b_angle), 0])
        d = second - first
        b.add_shape_capsule(
            palm,
            xform=tf(
                (first + second) / 2, wp.quat_between_vectors(wp.vec3(0, 0, 1), wp.vec3(*(d / np.linalg.norm(d))))
            ),
            radius=0.0008,
            half_height=float(np.linalg.norm(d) / 2),
            cfg=visual,
            color=(0.90, 0.82, 0.60),
            label="yoyo/finger-loop",
        )
    cord = []
    for k in range(40):
        link = b.add_body(xform=tf(center), is_kinematic=True, label=f"yoyo/cord-visual-{k}")
        b.add_shape_capsule(
            link, radius=0.0008, half_height=0.008, cfg=visual, color=(0.90, 0.82, 0.60), label="yoyo/string"
        )
        cord.append(link)
    info.update(
        robot="shadow_hand",
        tracked_body=toy,
        yoyo_hand=palm,
        yoyo_anchor=[0.0, -0.012, 0.0],
        yoyo_cord=cord,
        yoyo_axle_radius=0.012,
        yoyo_string_length=0.62,
        dense_waypoints=True,
        phase_delay=0,
        task="Release a responsive yo-yo and tug its string to recover it",
    )


def check(world, poses, fps):
    p = poses[:, world["tracked_body"], :3]
    release = int(1.5 * fps)
    bottom = release + int(np.argmin(p[release : int(3.2 * fps), 2]))
    fall = float(p[release, 2] - p[bottom, 2])
    rise = float(p[bottom : int(4.5 * fps), 2].max() - p[bottom, 2])
    result = {"drop_m": fall, "return_m": rise, "bottom_time": bottom / fps, "finite": bool(np.isfinite(poses).all())}
    result["pass"] = bool(
        result["finite"]
        and fall > 0.25
        and rise > 0.20
        and p[:, 2].min() > 0.10
        and np.ptp(p[:, 0]) < 0.18
        and np.ptp(p[:, 1]) < 0.18
    )
    return result
