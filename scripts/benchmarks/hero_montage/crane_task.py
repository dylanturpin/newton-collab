# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Arm-operated wrecking crane and a fully dynamic miniature building."""

import math

import numpy as np
import warp as wp
from furnishings import skin_box
from task_assets import import_object


def building(b, center):
    """Dry-stacked columns, floor panels and furniture; no collapse animation."""
    from scene import box, tf  # noqa: PLC0415 -- shared scene helpers

    structure, half_sizes, upper_floors, furniture = [], [], [], []
    concrete = (0.68, 0.70, 0.69)
    column_cfg = b.default_shape_cfg.copy()
    column_cfg.density, column_cfg.mu, column_cfg.gap = 1500, 0.85, 0.00025
    slab_cfg = column_cfg.copy()
    slab_cfg.density = 800
    box(b, (*center, 0.004), (0.17, 0.14, 0.004), (0.31, 0.36, 0.40), label="demolition_pad")

    def piece(position, half, cfg, label, color=concrete):
        body = b.add_body(xform=tf(position), label=label)
        b.add_shape_box(body, hx=half[0], hy=half[1], hz=half[2], cfg=cfg, color=color, label=label)
        skin_box(b, b.shape_count - 1, "paint", color, radius=0.0008)
        structure.append(body)
        half_sizes.append(half)
        return body

    def furniture_body(position, parts, label):
        body = b.add_body(xform=tf(position), label=label)
        for local, half, color in parts:
            box(b, local, half, color, body, label)
            skin_box(b, b.shape_count - 1, "paint", color, radius=0.001)
        furniture.append(body)

    blue, green, wood = (0.035, 0.43, 0.78), (0.02, 0.65, 0.47), (0.72, 0.55, 0.34)
    for floor in range(3):
        z = 0.008 + floor * 0.11
        for x in (-0.10, -0.05, 0, 0.05, 0.10):
            for y in (-0.075, 0.075):
                piece(
                    (center[0] + x, center[1] + y, z + 0.049),
                    (0.007, 0.007, 0.049),
                    column_cfg,
                    f"demolition_column_{floor}",
                    (0.82, 0.83, 0.79),
                )
        # Three real prefabricated panels per floor may separate on impact.
        for x in (-0.087, 0, 0.087):
            panel = piece(
                (center[0] + x, center[1], z + 0.104), (0.0432, 0.106, 0.006), slab_cfg, f"demolition_floor_{floor}"
            )
            if floor > 0:
                upper_floors.append(panel)
        if floor == 0:
            furniture_body(
                (center[0] - 0.046, center[1] - 0.010, z),
                [
                    ((0, 0, 0.046), (0.033, 0.022, 0.003), wood),
                    *[((x, y, 0.0215), (0.003, 0.003, 0.0215), wood) for x in (-0.027, 0.027) for y in (-0.016, 0.016)],
                ],
                "demolition_dining_table",
            )
            for y in (-0.044, 0.044):
                furniture_body(
                    (center[0] - 0.046, center[1] + y, z),
                    [
                        ((0, 0, 0.028), (0.017, 0.016, 0.003), blue),
                        ((0, y / 4, 0.043), (0.017, 0.0025, 0.015), blue),
                        *[
                            ((x, yy, 0.0125), (0.0025, 0.0025, 0.0125), wood)
                            for x in (-0.012, 0.012)
                            for yy in (-0.011, 0.011)
                        ],
                    ],
                    "demolition_dining_chair",
                )
        elif floor == 1:
            furniture_body(
                (center[0] + 0.040, center[1] + 0.020, z),
                [
                    ((0, 0, 0.012), (0.038, 0.022, 0.012), green),
                    ((0, 0.018, 0.032), (0.038, 0.004, 0.020), green),
                    ((-0.034, 0, 0.029), (0.004, 0.022, 0.005), green),
                    ((0.034, 0, 0.029), (0.004, 0.022, 0.005), green),
                ],
                "demolition_sofa",
            )
        else:
            furniture_body(
                (center[0] - 0.040, center[1], z),
                [
                    ((0, 0, 0.009), (0.033, 0.046, 0.009), wood),
                    ((0, 0, 0.023), (0.032, 0.045, 0.005), blue),
                    ((0, 0.045, 0.030), (0.033, 0.003, 0.030), wood),
                    ((0, 0.026, 0.031), (0.023, 0.011, 0.003), (0.87, 0.88, 0.83)),
                ],
                "demolition_bed",
            )
    return structure, half_sizes, upper_floors, furniture


def populate(b, info, ee, down, device, variant):
    from scene import box, plan_ik, tf  # noqa: PLC0415 -- shared scene helpers

    base = np.array([0.18, 0.25, 0])
    for body, label in enumerate(b.body_label):
        if label.endswith(("/fr3_leftfinger", "/fr3_rightfinger")):
            box(b, (0, 0, 0.085), (0.006, 0.006, 0.055), (0.31, 0.36, 0.40), body, "crane_long_jaw")
            for shape, owner in enumerate(b.shape_body):
                if owner == body:
                    b.shape_material_mu[shape] = 1.3
    for vi in range(6):
        b.joint_target_ke[vi], b.joint_target_kd[vi] = 8000, 200
        b.joint_effort_limit[vi] = 250
    for j, label in enumerate(b.joint_label):
        if "finger" in label:
            vi = b.joint_qd_start[j]
            b.joint_target_ke[vi], b.joint_target_kd[vi] = 4000, 40
            b.joint_effort_limit[vi] = 40
    pivot = base + np.array([0, -0.06, 0.142])
    lever = np.array([0, -0.20, 0.268])

    def control_point(lift, slew):
        q = wp.quat_rpy(0.0, 0.0, float(slew))
        rotated_pivot = base + np.asarray(wp.quat_rotate(q, wp.vec3(*(pivot - base))))
        return rotated_pivot + np.asarray(wp.quat_rotate(wp.quat_rpy(float(lift), 0.0, float(slew)), wp.vec3(*lever)))

    first = control_point(0, 0)
    poses = [(0, first + np.array([0, 0, 0.17]), 0.018), (1, first, 0.018), (1.8, first, 0.004)]
    phases = [
        (3.2, -0.12, -0.4),
        (4.0, -0.12, -0.4),
        (5.4, -0.12, 1.65),
        (11.0, -0.12, 1.65),
    ]
    previous_time, previous_lift, previous_slew = 1.8, 0.0, 0.0
    for time, lift, slew in phases:
        count = max(1, math.ceil((time - previous_time) / 0.055))
        for j in range(1, count + 1):
            u = j / count
            ease = u * u * (3 - 2 * u)
            angle = previous_slew + ease * (slew - previous_slew)
            tilt = previous_lift + ease * (lift - previous_lift)
            poses.append(
                (
                    previous_time + (time - previous_time) * u,
                    control_point(tilt, angle),
                    0.004,
                    wp.quat_rpy(0.0, 0.0, angle) * down,
                )
            )
        previous_time, previous_lift, previous_slew = time, lift, slew
    turned = wp.quat_rpy(0.0, 0.0, 1.65) * down
    last = control_point(-0.12, 1.65)
    poses.extend(
        [
            (11.7, last, 0.018, turned),
            (13.0, last + np.array([0, 0, 0.18]), 0.018, turned),
            (15, last + np.array([0, 0, 0.18]), 0.018, turned),
        ]
    )
    info["waypoints"] = plan_ik(b, ee, (0, 0, 0.1934), poses, device, desired_rotation=down)
    info["dense_waypoints"] = True
    joint_start = b.joint_count
    mapping, _ = import_object(
        b,
        "crane",
        tf(base),
        scale=2,
        moving={"crane.slew", "boom.lift", "hook.swing"},
        variant=variant,
        register_articulations=False,
    )
    for j in range(joint_start, b.joint_count):
        vi = b.joint_qd_start[j]
        if b.joint_label[j] == "crane/boom.lift.q":
            b.joint_limit_upper[vi] = 0
            b.joint_damping[vi] = 0.05
        elif b.joint_label[j] in ("crane/crane.slew.q", "crane/hook.swing.q"):
            b.joint_damping[vi] = 0.006
    box(b, (first[0], first[1], first[2] - 0.025), (0.022, 0.024, 0.012), (0.31, 0.36, 0.40), label="crane_boom_stop")
    hook_local = np.array([0, -0.338, 0.416])
    top = base + hook_local
    count, link_length, radius = 5, 0.025, 0.049
    previous, anchor = mapping["hook.swing"], hook_local
    chain, anchors = [], []
    steel = b.default_shape_cfg.copy()
    steel.density, steel.mu, steel.gap = 7800, 0.3, 0.0002
    for k in range(count):
        body = b.add_link(xform=tf(top - [0, 0, (k + 0.5) * link_length]), label=f"wrecking_chain_{k}")
        child_anchor = np.array([0, 0, link_length / 2])
        b.add_joint_ball(
            previous,
            body,
            parent_xform=tf(anchor),
            child_xform=tf(child_anchor),
            damping=0.0,
            armature=1e-4,
            label=f"wrecking_chain_joint_{k}",
        )
        anchors.append((previous, body, anchor.tolist(), child_anchor.tolist()))
        # Alternating closed oval links; their joints are entirely passive.
        rotation = wp.quat_rpy(0.0, 0.0, (k % 2) * math.pi / 2)
        for segment in range(16):
            a, c = segment * math.tau / 16, (segment + 1) * math.tau / 16
            pa = np.array([0.005 * math.cos(a), 0, 0.017 * math.sin(a)])
            pc = np.array([0.005 * math.cos(c), 0, 0.017 * math.sin(c)])
            q = wp.quat_between_vectors(wp.vec3(0, 0, 1), wp.vec3(*(pc - pa)))
            local = tf(np.asarray(wp.quat_rotate(rotation, wp.vec3(*((pa + pc) / 2)))), rotation * q)
            b.add_shape_capsule(
                body,
                xform=local,
                radius=0.003,
                half_height=float(np.linalg.norm(pc - pa) / 2),
                color=(0.50, 0.56, 0.61),
                cfg=steel,
                label="wrecking_chain_link",
            )
        chain.append(body)
        previous, anchor = body, np.array([0, 0, -link_length / 2])
    ball = b.add_link(xform=tf(top - [0, 0, count * link_length + radius]), label="wrecking_ball")
    b.add_joint_ball(
        previous,
        ball,
        parent_xform=tf(anchor),
        child_xform=tf((0, 0, radius)),
        damping=0.0,
        armature=1e-5,
        label="wrecking_ball_joint",
    )
    anchors.append((previous, ball, anchor.tolist(), [0, 0, radius]))
    b.add_shape_sphere(ball, radius=radius, color=(0.26, 0.31, 0.36), cfg=steel, label="wrecking_ball")
    b.add_articulation(list(range(joint_start, b.joint_count)), label="wrecking_crane")
    structure, sizes, floors, furniture = building(b, (0.53, 0.26))
    info.update(
        tracked_body=ball,
        demolition=True,
        wrecking_chain=chain,
        wrecking_chain_anchors=anchors,
        wrecking_ball_radius=radius,
        demolition_structure=structure,
        demolition_half_sizes=sizes,
        demolition_upper_floors=floors,
        demolition_furniture=furniture,
        crane_slew=mapping["crane.slew"],
        crane_boom=mapping["boom.lift"],
        crane_hook=mapping["hook.swing"],
        demolition_windup_time=4.0,
        task="Operate a passive wrecking crane to swing a linked steel ball into a three-floor miniature building",
    )
