# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Grasp a loose dinner plate and lower it into a populated catalog rack."""

import math

import numpy as np
import warp as wp
from task_assets import COLORS, import_object, load


def populate(b, info, ee, fingers, device, variant):
    from scene import plan_ik, tf  # noqa: PLC0415 -- shared task helpers

    plate_rotation = wp.quat_rpy(math.pi / 2, 0.0, math.pi / 2)
    rack_origin = np.array([0.19, 0.27, 0.0])
    fit = load("plate_rack")[0]["features"]["dish_fit"]
    pitch = fit["slot_pitch_m"]
    slots = 4
    thickness_center = np.mean(np.asarray(fit["rotated_bounds"]), axis=0)[0]
    roots = [rack_origin + np.array([(i - (slots - 1) / 2) * pitch - thickness_center, 0, 0.106]) for i in range(slots)]
    source = np.array([0.12, -0.31, 0.118])
    # A narrow staging cradle leaves the upper rim accessible to both jaws.
    for y in (-0.085, 0.085):
        b.add_shape_cylinder(
            -1,
            xform=tf((source[0] + thickness_center, source[1] + y, 0.018), wp.quat_rpy(0.0, math.pi / 2, 0.0)),
            radius=0.008,
            half_height=0.030,
            color=(0.40, 0.46, 0.50),
        )
    for x in (-0.002, 0.026):
        b.add_shape_capsule(
            -1, xform=tf((source[0] + x, source[1], 0.065)), radius=0.003, half_height=0.057, color=(0.40, 0.46, 0.50)
        )
    grasp = np.array([0.0095, 0, 0.085])
    pickup, target = source + grasp, roots[3] + grasp
    down = wp.quat_rpy(0.0, 0.0, math.pi / 2) * wp.quat(1, 0, 0, 0)
    for finger in fingers:
        b.joint_target_ke[finger], b.joint_target_kd[finger] = 5000, 40
        b.joint_effort_limit[finger] = 50
    arm_count = b._hero_driven_dofs - 2
    for vi in range(arm_count):
        b.joint_target_ke[vi], b.joint_target_kd[vi] = 8000, 200
        b.joint_effort_limit[vi] = 200
    info["integral_drive_count"] = arm_count
    for shape, owner in enumerate(b.shape_body):
        if owner >= 0 and b.body_label[owner].endswith(("/fr3_leftfinger", "/fr3_rightfinger")):
            b.shape_material_mu[shape] = 2.0
    lift_height = 0.18 if variant == 3 else 0.22
    poses = [
        (0, pickup + np.array([0, 0, lift_height]), 0.04),
        (1.0, pickup + np.array([0, 0, 0.07]), 0.04),
        (2.0, pickup, 0.04),
        (2.8, pickup, 0.0),
        (4.3, pickup + np.array([0, 0, lift_height]), 0.0),
        (6.8, target + np.array([0, 0, lift_height]), 0.0),
        (8.3, target + np.array([0, 0, 0.045]), 0.0),
        (9.5, target + np.array([0, 0, -0.006]), 0.0),
        (10.5, target + np.array([0, 0, -0.006]), 0.04),
        (12.0, target + np.array([0, 0, lift_height]), 0.04),
        (15, pickup + np.array([0, 0, lift_height]), 0.04),
    ]
    # The pads seat against the dish's inclined rim. Correct the measured
    # 11-degree grasp pitch with the robot wrist before approaching the slots.
    seated_rotation = wp.quat_rpy(0.0, 0.19, 0.0) * down
    poses = [(*pose, seated_rotation) if pose[0] >= 4.3 else pose for pose in poses]
    # IK only sees the robot; add all dish bodies after its self filters.
    info["waypoints"] = plan_ik(b, ee, (0, 0, 0.1034), poses, device, desired_rotation=down, cartesian_step=0.12)
    import_object(b, "plate_rack", tf(rack_origin))
    resting = []
    for i in (0, 1):
        mapping, _ = import_object(
            b, "rack_plate", tf(roots[i], plate_rotation), free_root=True, variant=i, finish_color=COLORS[i]
        )
        resting.append(mapping[load("rack_plate")[0]["root"]])
    mapping, _ = import_object(
        b, "rack_plate", tf(source, plate_rotation), free_root=True, variant=2, finish_color=COLORS[2]
    )
    info.update(
        tracked_body=mapping[load("rack_plate")[0]["root"]],
        arm_dofs=len(info["waypoints"][0][1]),
        dense_waypoints=True,
        plate_rack=True,
        rack_origin=rack_origin.tolist(),
        plate_slot_x=float(roots[3][0] + thickness_center),
        plate_resting_bodies=resting,
        plate_goal_rotation=list(plate_rotation),
        plate_grasp_local=[0, 0.085, 0.0095],
        task="Lift a loose dinner plate by its rim and seat it in a proc-gen wire rack with two contact-supported plates",
    )
