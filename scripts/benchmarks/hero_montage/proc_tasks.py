# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Contact tasks using complete proc-gen-3d toys and tabletop objects."""

import math

import numpy as np
import warp as wp
from robots import make_arm
from task_assets import import_object, load


def populate(b, info, kind, variant, assets, device):
    from scene import (  # noqa: PLC0415 -- reuse task helpers after scene initialization
        COLORS,
        box,
        plan_ik,
        tf,
    )

    robot = (
        "ur10"
        if kind in ("gear", "pile") or (kind == "toy" and variant % 2 == 0)
        else (
            "ur5"
            if kind == "pile" or (kind == "toy" and variant % 2)
            else ("kinova", "kuka", "xarm", "ur5")[variant % 4]
        )
    )
    if kind == "serve" and variant == 3:
        robot = "franka"
    root = (-0.30, 0, 0) if kind == "serve" and variant == 3 else (-0.42, 0, 0)
    ee, down, _fingers = make_arm(b, assets, device, robot=robot, root=root, gripper=kind not in ("toy", "pile"))
    info["robot"] = robot
    for finger in _fingers:
        b.joint_target_ke[finger] = 2000
        b.joint_effort_limit[finger] = 25
    tcp_offset = (0, 0, 0.1034)
    if kind == "puzzle":
        # Narrow jaw extensions reach the small knob without the stock pads
        # touching the tile or surrounding socket walls.
        for body, label in enumerate(b.body_label):
            if label.endswith(("/fr3_leftfinger", "/fr3_rightfinger")):
                b.add_shape_box(
                    body,
                    xform=tf((0, 0, 0.062)),
                    hx=0.004,
                    hy=0.005,
                    hz=0.018,
                    color=(0.25, 0.29, 0.34),
                    label="precision_jaw",
                )
        tcp_offset = (0, 0, 0.1334)
        for finger in _fingers:
            b.joint_target_ke[finger] = 500
            b.joint_target_kd[finger] = 10
            b.joint_effort_limit[finger] = 8
    if kind == "serve":
        from plate_task import populate as populate_plate  # noqa: PLC0415 -- optional plate task

        populate_plate(b, info, ee, _fingers, device, variant)
    elif kind == "pile":
        # Sweep spare hardware off the elevated work deck into a lower bin.
        box(b, (0, 0, 0.13), (0.26, 0.008, 0.060), (0.43, 0.49, 0.53), ee, "bench_scraper")
        for vi in range(6):
            b.joint_target_ke[vi], b.joint_target_kd[vi] = 6000, 180
            b.joint_effort_limit[vi] = 150
        info["integral_drive_count"] = 6
        poses = [
            (0, (0.18, -0.72, 0.38), None),
            (1.5, (0.18, -0.72, 0.263), None),
            (3.0, (0.18, -0.47, 0.263), None),
            (9.0, (0.18, 0.44, 0.263), None),
            (10.0, (0.18, 0.44, 0.46), None),
            (15, (0.18, -0.50, 0.35), None),
        ]
        info["waypoints"] = plan_ik(b, ee, (0, 0, 0.13), poses, device, desired_rotation=down, cartesian_step=0.12)
        info["dense_waypoints"] = True
        info["pile_bodies"] = []
        info["pile_piece_centers"] = []
        from task_assets import collision_bounds  # noqa: PLC0415 -- spawn from actual collision geometry

        spawn_bounds = []
        for j in range(96):
            name = "hex_bolt" if j % 8 < 6 else "hex_nut" if j % 8 == 6 else "washer"
            rotation = wp.quat_rpy(0.2 * (j % 3), 0.45, j * 0.71)
            low, high = collision_bounds(name, 3.0, rotation)
            center = np.array((0.18 + ((j % 6) - 2.5) * 0.080, -0.20 + (((j // 6) % 8) - 3.5) * 0.080, 0.0))
            p = center - (low + high) / 2
            p[2] = 0.206 + (j // 48) * 0.080 - low[2]
            bounds = (p + low, p + high)
            assert all(
                np.any(bounds[0] > other[1] + 0.002) or np.any(other[0] > bounds[1] + 0.002) for other in spawn_bounds
            )
            spawn_bounds.append(bounds)
            mapping, _ = import_object(
                b,
                name,
                tf(p, rotation),
                scale=3.0,
                free_root=True,
                variant=j % 3,
                finish_color=COLORS[int(np.random.default_rng(20261006 + j).integers(3))],
            )
            info["pile_bodies"].append(mapping[load(name)[0]["root"]])
            local_low, local_high = collision_bounds(name, 3.0, wp.quat_identity())
            info["pile_piece_centers"].append(((local_low + local_high) / 2).tolist())
        box(b, (0.18, -0.22, 0.100), (0.27, 0.42, 0.100), (0.29, 0.34, 0.38), label="hardware_work_deck")
        b.shape_material_mu[-1] = 0.20
        for x in (-0.10, 0.46):
            box(b, (x, -0.22, 0.235), (0.008, 0.42, 0.035), COLORS[0])
            box(b, (x, 0.46, 0.080), (0.008, 0.26, 0.080), COLORS[1], label="collection_bin_side")
        box(b, (0.18, 0.46, 0.008), (0.27, 0.26, 0.008), (0.60, 0.67, 0.69), label="collection_bin_floor")
        box(b, (0.18, 0.72, 0.080), (0.28, 0.008, 0.080), COLORS[1], label="collection_bin_back")
        info["pile_deck_height"] = 0.20
        info["pile_required_count"] = 90
        # Bounds come from the inner faces of the physical bin walls/floor.
        info["pile_bin_bounds"] = [[-0.092, 0.20, 0.016], [0.452, 0.712, 0.16]]
        info["tracked_body"] = info["pile_bodies"][0]
        info["task"] = "Sweep a mixed pile of proc-gen nuts, bolts and washers off a raised deck into a collection bin"
    elif kind == "gear":
        from crane_task import populate as populate_crane  # noqa: PLC0415 -- optional mechanism

        populate_crane(b, info, ee, down, device, variant)
    elif kind == "toy":
        name, scale = ("train", 1.5) if variant % 2 == 0 else ("truck", 1.5)
        push_height = 0.033 if name == "train" else 0.084
        b.add_shape_box(
            ee,
            xform=tf((0, 0, 0.13)),
            hx=0.12,
            hy=0.018,
            hz=0.015,
            color=COLORS[variant % 3],
            label="soft_pushing_tool",
        )
        start_y = -0.61 if name == "train" else -0.52
        approach_y = -0.43 if name == "train" else -0.40
        source_y = -0.28 if name == "train" else -0.20
        poses = [
            (0, (0.18, start_y, 0.24), None),
            (1.3, (0.18, start_y, push_height), None),
            (3, (0.18, approach_y, push_height), None),
            (7, (0.18, 0.40, push_height), None),
            (9, (0.18, 0.40, 0.25), None),
            (15, (0.18, -0.45, 0.25), None),
        ]
        info["waypoints"] = plan_ik(b, ee, (0, 0, 0.13), poses, device, desired_rotation=down, cartesian_step=0.20)
        mapping, _joints = import_object(
            b,
            name,
            tf((0.18, source_y, 0.003)),
            scale=scale,
            free_root=True,
            variant=variant,
        )
        info["tracked_body"] = mapping[load(name)[0]["root"]]
        info["wheel_bodies"] = [body for key, body in mapping.items() if "wheel" in key]
        info["wheel_radius"] = 0.03 if name == "train" else 0.0375
        for x in (-0.01, 0.37):
            box(b, (x, 0.08, 0.013), (0.009, 0.75, 0.013), (0.20, 0.25, 0.29))
        box(b, (0.18, 0.86, 0.015), (0.19, 0.012, 0.015), COLORS[(variant + 1) % 3])
        info["task"] = f"Push the articulated toy {name} along a parking lane on its free wheels"
        info["procgen_asset"] = name
        info["travel_axis"] = 1
    else:
        source = np.array([0.11, -0.30, 0.0])
        dest = np.array([0.16, 0.25, 0.0])
        orientation = wp.quat_identity()
        closed = 0.010
        if kind == "kit":
            name = "viaduct" if variant % 2 == 0 else "spiral"
            meta, _ = load(name)
            part = "piece.2"
            record = next(r for r in meta["bodies"] if r["id"] == part)
            scale = 1.0
            low = record["bounds"][0][2]
            source[2] = -low * scale + 0.003
            grasp_local = np.array([0, 0, 0.300 if name == "viaduct" else 0.305]) * scale
            pickup = source + grasp_local
            target = dest + grasp_local
            moving = {part}
            if variant == 2:
                # Keep the xArm's tall tier upright during the transfer. The
                # stock pads otherwise let it roll and hang on a connector lip.
                closed = 0.006
                for finger in _fingers:
                    b.joint_target_ke[finger] = 3500
                    b.joint_target_kd[finger] = 35
                    b.joint_effort_limit[finger] = 40
                for shape, body in enumerate(b.shape_body):
                    if body >= 0 and b.body_label[body].endswith(("/fr3_leftfinger", "/fr3_rightfinger")):
                        b.shape_material_mu[shape] = 1.3
            info["task"] = f"Seat the removable upper {name} tier onto the kit's keyed tube connectors"
        elif kind in ("puzzle", "interlock"):
            name, scale = kind, 2.0
            part = "piece.2.lift" if kind == "puzzle" else "bar.1.free"
            local = [0.032, 0, 0.028] if kind == "puzzle" else [0, 0.050, 0.017]
            grasp_local = np.array(local) * scale
            source[2] = (-0.008 if kind == "puzzle" else -0.0057) * scale + 0.005
            pickup, target = source + grasp_local, dest + grasp_local
            moving = {part}
            closed = 0.007 if kind == "puzzle" else 0.003
            if kind == "interlock":
                source[2] += 0.020
                pickup = source + grasp_local
                box(b, (source[0], source[1], 0.01), (0.05, 0.16, 0.01), (0.25, 0.3, 0.34))
                down = wp.quat_rpy(0.0, 0.0, math.pi / 2) * down
            info["task"] = (
                "Fit the removable hexagonal tile into its matching puzzle socket"
                if kind == "puzzle"
                else "Assemble the two half-lap bars of a cross puzzle"
            )
        else:
            name = "bird" if variant % 2 == 0 else "mug"
            scale = 1.0 if name == "bird" else 0.85
            part = load(name)[0]["root"]
            source[2] = 0.004
            dest[2] = 0.014
            grasp_local = np.array([0, 0, 0.045] if name == "bird" else [0, 0, 0.047]) * scale
            if name == "bird":
                dest[2] = 0.1135
                source[2] = 0.1175
                down = wp.quat(0.5, 0.5, 0.5, 0.5)
                closed = 0.015
                for point in (source, dest):
                    b.add_shape_cylinder(
                        -1, xform=tf((point[0], point[1], 0.060)), radius=0.085, half_height=0.060, color=COLORS[1]
                    )
            pickup, target = source + grasp_local, dest + grasp_local
            moving = None
            box(b, (dest[0], dest[1], 0.006), (0.13, 0.12, 0.006), COLORS[(variant + 1) % 3])
            info["task"] = (
                "Place a molded toy bird on its display stand"
                if name == "bird"
                else "Place a handled ceramic mug onto a serving coaster"
            )
        if kind == "puzzle":
            # Account for the measured loaded finger-pad deflection before
            # approaching the 2.6 mm-clearance socket.
            target = target + np.array([-0.005, 0.003, 0.004])
        hover = 0.36 if kind == "kit" else 0.46
        approach_height = 0.06 if kind == "kit" else 0.10
        poses = [
            (0, (*pickup[:2], min(hover, 0.46)), 0.04),
            (1.1, (*pickup[:2], pickup[2] + 0.11), 0.04),
            (2, pickup, 0.04),
            (2.8, pickup, closed),
            (4.3, (*pickup[:2], hover), closed),
            (5.7, (*target[:2], hover), closed),
            (7, target + np.array([0, 0, approach_height]), closed),
            (7.6, target + np.array([0, 0, 0.035]), closed),
            (8.4, target + np.array([0, 0, 0.002]), closed),
            (9.5, target + np.array([0, 0, 0.002]), 0.04),
            (11, (*target[:2], hover), 0.04),
            (15, (*pickup[:2], min(hover, 0.46)), 0.04),
        ]
        info["waypoints"] = plan_ik(b, ee, tcp_offset, poses, device, desired_rotation=down, cartesian_step=0.20)
        if kind in ("kit", "gear", "puzzle", "interlock"):
            mapping, _joints = import_object(
                b,
                name,
                tf(dest),
                scale=scale,
                moving=moving,
                detached=moving,
                body_poses={part: tf(source)},
                variant=variant,
            )
            target_pose = dest
        else:
            mapping, _joints = import_object(
                b, name, tf(source, orientation), scale=scale, free_root=True, variant=variant
            )
            target_pose = dest
        info["tracked_body"] = mapping[part]
        info["placement_target"] = target_pose.tolist()
        info["procgen_asset"] = name
    info["arm_dofs"] = len(info["waypoints"][0][1])
