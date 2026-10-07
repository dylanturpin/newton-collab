# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Manipulate removable pieces of stock proc-gen balance and knife sets."""

import math

import numpy as np
import warp as wp
from robots import make_arm
from task_assets import import_object, load


def populate(b, info, kind, variant, assets, device):
    from scene import COLORS, box, plan_ik, tf  # noqa: PLC0415 -- scene helpers

    robot = ("franka", "ur5", "xarm", "ur10")[variant % 4]
    ee, down, fingers = make_arm(b, assets, device, robot=robot, root=(-0.30, 0.10, 0))
    for finger in fingers:
        b.joint_target_ke[finger] = 1800
        b.joint_target_kd[finger] = 18
        b.joint_effort_limit[finger] = 25
    info["robot"] = robot

    if kind == "lift":
        scale = 2.5
        orientation = wp.quat_rpy(0.0, 0.0, math.pi / 2)
        source = (0.16, -0.635, -0.0955)
        destination = (0.16, 0.20, 0.02)
        pickup = (0.16, -0.40, 0.12)
        target = (0.16, 0.435, 0.32)
        closed = 0.032
        down = orientation * down
        poses = [
            (0, (*pickup[:2], 0.40), 0.04),
            (1.8, pickup, 0.04),
            (2.6, pickup, closed),
            (4.0, (*pickup[:2], 0.49), closed),
            (5.5, (*target[:2], 0.49), closed),
            (7.8, target, closed),
            (8.6, target, 0.04),
            (10, (*target[:2], 0.49), 0.04),
            (15, (*pickup[:2], 0.40), 0.04),
        ]
        info["waypoints"] = plan_ik(b, ee, (0, 0, 0.1034), poses, device, desired_rotation=down, cartesian_step=0.10)
        mapping, joints = import_object(
            b,
            "balance_board",
            tf(destination, orientation),
            scale=scale,
            body_poses={"weight.1.1": tf(source, orientation)},
            variant=variant,
        )
        # A passive torsion spring stabilizes this top-heavy catalog beam.
        # It never tracks a commanded angle: the missing weight tips the beam,
        # and replacing that weight restores its mechanical equilibrium.
        beam_joint = next(i for i, label in enumerate(b.joint_label) if label == "balance_board/balance.pivot.q")
        vi = b.joint_qd_start[beam_joint]
        b.joint_spring_stiffness[vi] = 4.0
        b.joint_spring_ref[vi] = 0.0
        b.joint_damping[vi] = 0.8
        info.update(
            catalog_task="balance",
            tracked_body=mapping["weight.1.1"],
            balance_beam=mapping["balance.pivot"],
            balance_q=joints["balance.pivot.q"],
            balance_weights=[mapping[name] for name in mapping if name.startswith("weight.")],
            balance_pose=list(destination),
            balance_weight_com=[0.094 * scale, 0, 0.0862 * scale],
            passive_balance_spring_nm_per_rad=4.0,
            task="Restore a stock balance board's equilibrium by placing its missing counterweight",
        )
        box(b, (*pickup[:2], 0.045), (0.060, 0.065, 0.045), (0.28, 0.33, 0.37), label="counterweight_source_stand")
        # Expose the spring housing at the bearing instead of implying a motor.
        b.add_shape_cylinder(
            mapping["fixed.walnut"],
            xform=tf((0, -0.16, 0.057 * scale), wp.quat_rpy(math.pi / 2, 0.0, 0.0)),
            radius=0.025,
            half_height=0.014,
            color=(0.40, 0.46, 0.51),
            label="passive_torsion_spring_housing",
        )
    else:
        for vi in range(7):
            b.joint_target_ke[vi], b.joint_target_kd[vi] = 10000, 300
        for finger in fingers:
            b.joint_target_ke[finger], b.joint_target_kd[finger] = 4000, 30
        for si, body in enumerate(b.shape_body):
            if body >= 0 and b.body_label[body].endswith(("fr3_leftfinger", "fr3_rightfinger")):
                b.shape_material_mu[si] = 1.2
        meta, arrays = load("knife_block_task")
        part = next(record for record in meta["bodies"] if record["id"] == "knife.0")
        destination = (0.15, 0.22, 0.0)
        source = (0.15, -0.32, -0.0265)
        handle_x = -0.0594
        pickup = (source[0] + handle_x, source[1], 0.248)
        target = (destination[0] + handle_x, destination[1], 0.275)
        closed = 0.009
        poses = [
            (0, (*pickup[:2], 0.47), 0.04),
            (1.8, pickup, 0.04),
            (2.6, pickup, closed),
            (4, (*pickup[:2], 0.49), closed),
            (5.4, (*target[:2], 0.49), closed),
            (6.2, (*target[:2], 0.46), closed),
            (10.0, target, closed),
            (10.8, target, 0.022),
            (12.0, (*target[:2], 0.48), 0.022),
            (12.8, (*target[:2], 0.48), 0.04),
            (15, (*pickup[:2], 0.47), 0.04),
        ]
        info["waypoints"] = plan_ik(b, ee, (0, 0, 0.1034), poses, device, desired_rotation=down, cartesian_step=0.10)
        mapping, _ = import_object(
            b, "knife_block_task", tf(destination), body_poses={"knife.0": tf(source)}, variant=variant
        )
        import_object(b, "knife_source_task", tf((0.1302, -0.32, 0)))
        info.update(
            catalog_task="knife",
            _servo_model=b._hero_ik_model,
            servo_ee=ee,
            tracked_body=mapping["knife.0"],
            knife_fixture=mapping["fixed.oak"],
            knife_goal_pose=list(destination),
            knife_slot_center_x=handle_x,
            knife_slot_width=0.005,
            knife_slot_depth=meta["features"]["slots"]["depth"],
            knife_blade_thickness=0.002218,
            knife_neck_seating_z=0.2142,
            knife_grasp_point=[handle_x, 0, 0.275],
            knife_grip_coords=fingers,
            knife_closed_aperture=closed,
            knife_blade_vertices=arrays["b1_v1_v"].tolist(),
            knife_blade_edges=np.unique(
                np.sort(arrays["b1_v1_f"][:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2), axis=1), axis=0
            ).tolist(),
            knife_neck_vertices=arrays["b1_v0_v"].tolist(),
            knife_neck_edges=np.unique(
                np.sort(arrays["b1_v0_f"][:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2), axis=1), axis=0
            ).tolist(),
            knife_bounds=part["bounds"],
            task="Lift a stock chef's knife and insert its blade through the knife block's 5 mm kerf",
        )
    info["dense_waypoints"] = True
    info["arm_dofs"] = len(info["waypoints"][0][1])
    for index, body in enumerate(b.shape_body):
        if body >= 0 and ("/weight." in b.body_label[body] or "/knife." in b.body_label[body]):
            if b.shape_label[index].startswith("procgen_visual") and b.shape_source[index].metallic < 0.1:
                b.shape_color[index] = np.asarray(COLORS[(body + variant) % 3], dtype=np.float32)
