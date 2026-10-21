# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""An unpowered toy winch winds a fixed-length articulated rope."""

import math

import numpy as np
import warp as wp

from ._tabletop_common import box, plan_ik, relative_angles, rotate_vectors, tf


def populate(b, info, ee, down, fingers, device):

    center = np.array([0.18, 0.20, 0.76])
    crank_radius = 0.10
    grip_y = -0.055
    b.joint_q[5] = 5.8
    orientation = wp.quat_rpy(-math.pi / 2, 0.0, 0.0)
    poses = []
    for t in np.linspace(0, 10, 201):
        u = np.clip((t - 1.5) / 7.5, 0, 1)
        angle = -1.65 * math.tau * (u * u * (3 - 2 * u))
        pos = np.add(center, [crank_radius * math.cos(angle), grip_y - center[1], -crank_radius * math.sin(angle)])
        # Seat the jaws gradually with a light preload on the freely turning grip.
        aperture = float(0.025 - 0.010 * np.clip(t / 1.0, 0, 1))
        poses.append((float(t), pos, aperture, wp.quat_rpy(0.0, float(angle), 0.0) * orientation))
    for i in range(6):
        b.joint_target_ke[i], b.joint_target_kd[i] = 16000, 220
    for i in fingers:
        b.joint_target_ke[i], b.joint_target_kd[i], b.joint_effort_limit[i] = 1200, 20, 12
    info["waypoints"] = plan_ik(b, ee, (0, 0, 0.1034), poses, device, desired_rotation=orientation)
    for i, owner in enumerate(b.shape_body):
        if owner >= 0 and b.body_label[owner].endswith(("/fr3_leftfinger", "/fr3_rightfinger")):
            b.shape_material_mu[i] = 1.4
    blue, teal, gold, steel = (0.06, 0.38, 0.70), (0.06, 0.57, 0.43), (0.92, 0.66, 0.20), (0.30, 0.36, 0.40)
    axis = wp.quat_rpy(math.pi / 2, 0.0, 0.0)
    for y in (0.06, 0.34):
        box(b, (center[0], y, 0.365), (0.022, 0.018, 0.365), teal, label="well/upright")
        box(b, (center[0], y, 0.014), (0.16, 0.045, 0.014), blue, label="well/foot")
        b.add_shape_cylinder(
            -1,
            xform=tf((center[0], y, center[2]), axis),
            radius=0.036,
            half_height=0.025,
            color=steel,
            label="well/bearing",
        )
    first = b.joint_count
    drum = b.add_link(xform=tf(center), label="well/drum")
    b.add_joint_revolute(-1, drum, parent_xform=tf(center), axis=(0, 1, 0), damping=0.001, label="well/spool-bearing")
    cfg = b.default_shape_cfg.copy()
    cfg.density, cfg.mu, cfg.gap = 250, 0.65, 0.0001
    b.add_shape_cylinder(
        drum, xform=tf((0, 0, 0), axis), radius=0.030, half_height=0.075, cfg=cfg, color=steel, label="well/drum"
    )
    for y in (-0.075, 0.075):
        b.add_shape_cylinder(
            drum, xform=tf((0, y, 0), axis), radius=0.048, half_height=0.004, cfg=cfg, color=blue, label="well/flange"
        )
    b.add_shape_cylinder(
        drum, xform=tf((0, -0.125, 0), axis), radius=0.012, half_height=0.045, cfg=cfg, color=steel, label="well/axle"
    )
    box(b, (0.05, -0.17, 0), (0.062, 0.009, 0.012), blue, drum, "well/crank")
    b.add_shape_cylinder(
        drum,
        xform=tf((crank_radius, -0.21, 0), axis),
        radius=0.005,
        half_height=0.04,
        cfg=cfg,
        color=steel,
        label="well/handle-pin",
    )
    knob = b.add_link(xform=tf(np.add(center, [crank_radius, grip_y - center[1], 0])), label="well/knob")
    b.add_joint_revolute(
        drum,
        knob,
        parent_xform=tf((crank_radius, grip_y - center[1], 0)),
        axis=(0, 1, 0),
        damping=0.0001,
        label="well/handle-bearing",
    )
    grip_cfg = cfg.copy()
    grip_cfg.mu = 1.4
    b.add_shape_cylinder(
        knob,
        xform=tf((0, 0.012, 0), axis),
        radius=0.018,
        half_height=0.028,
        cfg=grip_cfg,
        color=gold,
        label="well/grip",
    )
    # Short capsules bend at passive planar hinges; their length never changes.
    length, count, radius = 0.012, 44, 0.003
    top = np.add(center, [0.034, 0, 0])
    previous, anchor = drum, np.array([0.034, 0, 0])
    anchors = []
    cfg.density, cfg.mu = 1400, 0.65
    for k in range(count):
        link = b.add_link(xform=tf(top - [0, 0, (k + 0.5) * length]), label=f"well/rope-{k}")
        child = [0, 0, length / 2]
        b.add_joint_revolute(
            previous,
            link,
            axis=(0, 1, 0),
            parent_xform=tf(anchor),
            child_xform=tf(child),
            damping=0.00001,
            armature=1e-6,
            label=f"well/rope-joint-{k}",
        )
        b.add_shape_capsule(
            link, radius=radius, half_height=length / 2, cfg=cfg, color=(0.82, 0.72, 0.48), label="well/rope"
        )
        anchors.append((previous, link, list(anchor), child))
        previous, anchor = link, [0, 0, -length / 2]
    bucket = b.add_link(xform=tf(top - [0, 0, count * length + 0.10]), label="well/bucket")
    b.add_joint_ball(
        previous,
        bucket,
        parent_xform=tf(anchor),
        child_xform=tf((0, 0, 0.10)),
        damping=0.0001,
        label="well/bucket-hook",
    )
    anchors.append((previous, bucket, anchor, [0, 0, 0.10]))
    cfg.density = 220
    b.add_shape_cylinder(bucket, radius=0.040, half_height=0.003, cfg=cfg, color=blue, label="well/bucket-bottom")
    for k in range(24):
        a = k * math.tau / 24
        b.add_shape_box(
            bucket,
            xform=tf((0.040 * math.cos(a), 0.040 * math.sin(a), 0.03), wp.quat_rpy(0.0, 0.0, a)),
            hx=0.003,
            hy=0.0055,
            hz=0.03,
            cfg=cfg,
            color=blue,
            label="well/bucket-wall",
        )
    for k in range(16):
        a, c = k * math.pi / 16, (k + 1) * math.pi / 16
        pa, pc = (
            np.array([0.04 * math.cos(a), 0, 0.06 + 0.04 * math.sin(a)]),
            np.array([0.04 * math.cos(c), 0, 0.06 + 0.04 * math.sin(c)]),
        )
        b.add_shape_capsule(
            bucket,
            xform=tf((pa + pc) / 2, wp.quat_between_vectors(wp.vec3(0, 0, 1), wp.vec3(*(pc - pa)))),
            radius=0.0025,
            half_height=float(np.linalg.norm(pc - pa) / 2),
            cfg=cfg,
            color=steel,
            label="well/bucket-bail",
        )
    b.add_articulation(list(range(first, b.joint_count)), label="well/mechanism")
    # The fixed bearing housing surrounds its axle; their hinge supplies support.
    bearings = [i for i, label in enumerate(b.shape_label) if label == "well/bearing"]
    for bearing in bearings:
        for i, owner in enumerate(b.shape_body):
            if owner == drum:
                b.add_shape_collision_filter_pair(bearing, i)
    info.update(
        tracked_body=bucket,
        well_drum=drum,
        well_contact_bodies=[
            knob,
            *[i for i, label in enumerate(b.body_label) if label.endswith(("/fr3_leftfinger", "/fr3_rightfinger"))],
        ],
        well_drum_radius=0.030,
        well_rope=list(range(drum + 2, bucket)),
        well_anchors=anchors,
        task="Wind an articulated rope onto a passive drum to lift a bucket",
    )


def check(info, poses, fps):
    """Measure the completed physical interaction from its recorded trajectory."""

    drum = poses[:, info["well_drum"]]
    angle = relative_angles(drum[:, 3:], 1)
    bucket = poses[:, info["tracked_body"], :3]
    errors = []
    for parent, child, pa, ca in info["well_anchors"]:
        p, c = poses[:, parent], poses[:, child]
        errors.append(
            np.max(
                np.linalg.norm(
                    p[:, :3] + rotate_vectors(p[:, 3:], pa) - c[:, :3] - rotate_vectors(c[:, 3:], ca),
                    axis=1,
                )
            )
        )
    lift = float(bucket[-1, 2] - bucket[0, 2])
    turns = float(-angle[-1] / math.tau)
    error = float(max(errors))
    rope = poses[-1, np.asarray(info["well_rope"]), :3]
    radial = np.linalg.norm((rope - drum[-1, :3])[:, [0, 2]], axis=1)
    wrapped_indices = np.flatnonzero((radial > 0.030) & (radial < 0.048))
    wrapped = int(len(wrapped_indices))
    rope_angle = np.unwrap(np.arctan2((rope - drum[-1, :3])[:, 2], (rope - drum[-1, :3])[:, 0]))
    rope_turns = float(abs(rope_angle[wrapped_indices[-1]] - rope_angle[0]) / math.tau) if wrapped else 0.0
    # Evaluate after the robot has stopped. Smooth pendulum swing is allowed;
    # rapid second differences expose contact chatter instead of slow drift.
    rest = poses[-int(fps) :, np.asarray(info["well_contact_bodies"]), :3]
    contact_travel = float(np.max(np.linalg.norm(np.ptp(rest, axis=0), axis=1)))
    contact_acceleration = float(np.sqrt(np.mean((np.diff(rest, n=2, axis=0) * fps**2) ** 2)))
    rope_rest = poses[-int(fps) :, np.asarray(info["well_rope"]), :3]
    rope_acceleration = float(np.sqrt(np.mean((np.diff(rope_rest, n=2, axis=0) * fps**2) ** 2)))
    return dict(
        drum_turns=turns,
        bucket_lift_m=lift,
        maximum_rope_joint_error_m=error,
        wrapped_segments=wrapped,
        rope_winding_turns=rope_turns,
        resting_contact_travel_m=contact_travel,
        resting_contact_acceleration_rms=contact_acceleration,
        resting_rope_acceleration_rms=rope_acceleration,
        **{
            "pass": bool(
                1.5 < turns < 1.8
                and lift > 0.30
                and error < 0.005
                and wrapped >= 28
                and rope_turns > 1.5
                and contact_travel < 0.0005
                and contact_acceleration < 0.05
                and rope_acceleration < 0.15
            )
        },
    )
