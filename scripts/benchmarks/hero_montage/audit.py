# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Audit recorded behavior independently of the simulation's summary flags."""

import argparse
import json
from pathlib import Path

import numpy as np


def pile_collection(world, poses, fps):
    """Measure geometry centers inside the actual bin, below the work deck."""
    from scipy.spatial.transform import Rotation

    pile = poses[:, [world["body_start"] + i for i in world["pile_bodies"]]]
    centers = pile[:, :, :3].copy()
    if "pile_piece_centers" in world:
        local = np.broadcast_to(np.asarray(world["pile_piece_centers"]), centers.shape)
        centers += Rotation.from_quat(pile[:, :, 3:].reshape(-1, 4)).apply(local.reshape(-1, 3)).reshape(centers.shape)
        deck = world["pile_deck_height"]
        dropped = (centers[2 * fps, :, 2] >= deck - 0.005) & (centers[-1, :, 2] < deck - 0.04)
    else:
        dropped = centers[2 * fps, :, 2] - centers[-1, :, 2] > 0.12
    bounds = np.asarray(world["pile_bin_bounds"])
    inside = np.all((centers[-1] >= bounds[0]) & (centers[-1] <= bounds[1]), axis=1)
    return {
        "displacements_m": np.linalg.norm(centers[-1] - centers[2 * fps], axis=1).tolist(),
        "dropped_into_bin_count": int(np.sum(inside & dropped)),
        "geometry_center_measurement": "pile_piece_centers" in world,
    }


def knife_slot_fit(world, pose):
    """Clip the native blade triangles at the block lip and measure clearance."""
    from scipy.spatial.transform import Rotation

    vertices = Rotation.from_quat(pose[3:]).apply(world["knife_blade_vertices"]) + pose[:3]
    top = world["knife_goal_pose"][2] + 0.19
    edges = vertices[np.asarray(world["knife_blade_edges"])]
    crosses = (edges[:, 0, 2] < top) != (edges[:, 1, 2] < top)
    selected = edges[crosses]
    alpha = (top - selected[:, 0, 2]) / (selected[:, 1, 2] - selected[:, 0, 2])
    points = np.vstack(
        (vertices[vertices[:, 2] <= top], selected[:, 0] + alpha[:, None] * (selected[:, 1] - selected[:, 0]))
    )
    if not len(points):
        return None
    x = world["knife_goal_pose"][0] + world["knife_slot_center_x"]
    y = world["knife_goal_pose"][1]
    side = min(points[:, 0].min() - (x - 0.0025), x + 0.0025 - points[:, 0].max())
    half_depth = world.get("knife_slot_depth", 0.108) / 2
    end = min(points[:, 1].min() - (y - half_depth), y + half_depth - points[:, 1].max())
    bottom = points[:, 2].min() - (world["knife_goal_pose"][2] + 0.008)
    return np.array((side, end, bottom)) * 1000


def knife_support_gap(world, pose):
    """Measure the native handle surface over the supporting slot lip."""
    from scipy.spatial.transform import Rotation

    vertices = Rotation.from_quat(pose[3:]).apply(world["knife_neck_vertices"]) + pose[:3]
    center = np.add(world["knife_goal_pose"][:2], (world["knife_slot_center_x"], 0))
    half_size = np.array((world["knife_slot_width"], world["knife_slot_depth"])) / 2
    outside = np.any(np.abs(vertices[:, :2] - center) >= half_size, axis=1)
    points = [vertices[outside]]
    edges = vertices[np.asarray(world["knife_neck_edges"])]
    for axis in (0, 1):
        for sign in (-1, 1):
            boundary = center[axis] + sign * half_size[axis]
            crosses = (edges[:, 0, axis] < boundary) != (edges[:, 1, axis] < boundary)
            selected = edges[crosses]
            alpha = (boundary - selected[:, 0, axis]) / (selected[:, 1, axis] - selected[:, 0, axis])
            points.append(selected[:, 0] + alpha[:, None] * (selected[:, 1] - selected[:, 0]))
    points = np.vstack(points)
    return float(1000 * (points[:, 2].min() - world["knife_goal_pose"][2] - 0.19))


def catalog_check(world, poses, fps):
    """Evaluate actual equilibrium and insertion, without using script targets."""
    from scipy.spatial.transform import Rotation

    start = world["body_start"]
    obj = poses[:, start + world["tracked_body"]]
    check = {}
    if world["catalog_task"] == "balance":
        beam = poses[:, start + world["balance_beam"]]
        local_rotation = Rotation.from_euler("z", -np.pi / 2) * Rotation.from_quat(beam[:, 3:])
        angles = local_rotation.as_euler("xyz")[:, 1]
        com = obj[:, :3] + Rotation.from_quat(obj[:, 3:]).apply(np.tile(world["balance_weight_com"], (len(obj), 1)))
        goal = np.add(world["balance_pose"], [0, 0.235, 0.2155])
        check.update(
            counterweight_lift_m=float(com[:, 2].max() - com[0, 2]),
            missing_weight_tilt_degrees=float(np.rad2deg(abs(angles[: 3 * fps]).max())),
            final_balance_error_degrees=float(np.rad2deg(abs(angles[-fps:]).max())),
            counterweight_seat_error_mm=float(1000 * np.linalg.norm(com[-1] - goal)),
            passive_balance=world.get("passive_balance_spring_nm_per_rad") == 4,
        )
        check["pass"] = bool(
            check["counterweight_lift_m"] > 0.20
            and check["missing_weight_tilt_degrees"] > 5
            and check["final_balance_error_degrees"] < 2.5
            and check["counterweight_seat_error_mm"] < 25
            and check["passive_balance"]
        )
    else:
        rotation = Rotation.from_quat(obj[:, 3:])
        goal = np.asarray(world["knife_goal_pose"])
        slot_x = goal[0] + world["knife_slot_center_x"]
        blade_tip = obj[:, :3] + rotation.apply(np.tile([-0.0594, 0, 0.0376], (len(obj), 1)))
        neck = obj[-1, :3] + rotation[-1].apply([-0.0594, 0, world["knife_neck_seating_z"]])
        check.update(
            blade_lift_m=float(blade_tip[:, 2].max() - blade_tip[0, 2]),
            inserted_depth_m=float(goal[2] + 0.19 - blade_tip[-1, 2]),
            slot_lateral_error_mm=float(1000 * abs(blade_tip[-1, 0] - slot_x)),
            slot_longitudinal_error_mm=float(1000 * abs(blade_tip[-1, 1] - goal[1])),
            knife_orientation_error_degrees=float(np.rad2deg(rotation[-1].magnitude())),
            neck_seating_error_mm=float(1000 * (neck[2] - goal[2] - 0.19)),
        )
        if "knife_blade_vertices" in world:
            fits = [fit for pose in obj[int(5.4 * fps) :] if (fit := knife_slot_fit(world, pose)) is not None]
            final_fit = knife_slot_fit(world, obj[-1])
            hand = poses[:, start + world["servo_ee"]]
            hand_rotation = Rotation.from_quat(hand[:, 3:])
            grasp = obj[:, :3] + rotation.apply(np.tile(world["knife_grasp_point"], (len(obj), 1)))
            relative_position = hand_rotation.inv().apply(grasp - hand[:, :3])
            relative_rotation = hand_rotation.inv() * rotation
            begin, end = int(5.4 * fps), int(world.get("measured_knife_release_time", 10.0) * fps)
            check.update(
                blade_wall_clearance_mm=float(final_fit[0]) if final_fit is not None else -np.inf,
                blade_end_clearance_mm=float(final_fit[1]) if final_fit is not None else -np.inf,
                blade_floor_clearance_mm=float(final_fit[2]) if final_fit is not None else -np.inf,
                minimum_insertion_clearance_mm=float(np.min(fits)) if fits else -np.inf,
                knife_grasp_slip_mm=float(
                    1000 * np.linalg.norm(relative_position[begin:end] - relative_position[begin], axis=1).max()
                ),
                knife_grasp_rotation_degrees=float(
                    np.rad2deg((relative_rotation[begin].inv() * relative_rotation[begin:end]).magnitude()).max()
                ),
                final_drift_mm=float(1000 * np.linalg.norm(obj[-fps:, :3] - obj[-1, :3], axis=1).max()),
            )
            if "knife_neck_vertices" in world:
                check["neck_support_gap_mm"] = knife_support_gap(world, obj[-1])
                check["released_after_seating"] = (
                    "measured_knife_release_time" in world
                    and world["measured_knife_release_time"] > world["measured_knife_seat_time"]
                    and world["measured_knife_release_time"] < len(obj) / fps - 2
                )
            aligned = (
                check["blade_wall_clearance_mm"] > -0.15
                and check["blade_end_clearance_mm"] > -0.15
                and check["blade_floor_clearance_mm"] > 0
                and check["minimum_insertion_clearance_mm"] > -0.25
                and check["knife_grasp_slip_mm"] < 6
                and check["knife_grasp_rotation_degrees"] < 5
                and check["final_drift_mm"] < 1
                and check.get("released_after_seating", True)
            )
        else:
            aligned = check["knife_orientation_error_degrees"] < 0.5
        check["pass"] = bool(
            check["blade_lift_m"] > 0.18
            and 0.14 < check["inserted_depth_m"] < 0.182
            and ("knife_blade_vertices" in world or check["slot_lateral_error_mm"] < 0.8)
            and ("knife_blade_vertices" in world or check["slot_longitudinal_error_mm"] < 8)
            and aligned
            and -0.25 < check.get("neck_support_gap_mm", check["neck_seating_error_mm"]) < 3
        )
    return check


def drawer_placement(world, poses):
    """Check the cutlery mesh footprint and resting height in the moving tray."""
    from scipy.spatial.transform import Rotation

    start = world["body_start"]
    drawer = poses[-1, start + world["drawer_body"]]
    fork = poses[-1, start + world["placed_bodies"][0]]
    corners = np.asarray(world["fork_contact_vertices"])
    points = (
        Rotation.from_quat(drawer[3:]).inv().apply(Rotation.from_quat(fork[3:]).apply(corners) + fork[:3] - drawer[:3])
    )
    low, high = points.min(axis=0), points.max(axis=0)
    in_pocket = bool(
        low[0] > -0.282
        and high[0] < 0.152
        and low[1] > -0.222
        and high[1] < -0.058
        and low[2] >= 0.0095
        and low[2] < 0.020
        and high[2] < 0.060
    )
    return {"fork_in_tray": in_pocket, "fork_tray_bounds": [low.tolist(), high.tolist()]}


def demolition_check(world, poses, fps):
    """Require a stable building, physical ball contact and subsequent collapse."""
    from scipy.spatial.transform import Rotation

    start = world["body_start"]
    ball = poses[:, start + world["tracked_body"]]
    structure = poses[:, [start + body for body in world["demolition_structure"]]]
    half_sizes = np.asarray(world["demolition_half_sizes"])
    relative = ball[:, None, :3] - structure[:, :, :3]
    local = Rotation.from_quat(structure[:, :, 3:].reshape(-1, 4)).inv().apply(relative.reshape(-1, 3))
    local = local.reshape(relative.shape)
    separation = np.linalg.norm(local - np.clip(local, -half_sizes, half_sizes), axis=2)
    separation -= world["wrecking_ball_radius"]
    contact = np.flatnonzero(separation.min(axis=1) < 0.002)
    impact_time = float(contact[0] / fps) if len(contact) else None
    stable_end = min(round(3.5 * fps), int(contact[0]) if len(contact) else len(ball))
    stability = float(np.linalg.norm(structure[:stable_end, :, :3] - structure[0, :, :3], axis=2).max())
    floors = poses[:, [start + body for body in world["demolition_upper_floors"]]]
    drops = floors[0, :, 2] - floors[-1, :, 2]
    rotation = Rotation.from_quat(floors[0, :, 3:]).inv() * Rotation.from_quat(floors[-1, :, 3:])
    speed = np.linalg.norm(np.diff(ball[:, :3], axis=0), axis=1) * fps
    errors = []
    for parent, child, pa, ca in world["wrecking_chain_anchors"]:
        pp, cp = poses[:, start + parent], poses[:, start + child]
        p = pp[:, :3] + Rotation.from_quat(pp[:, 3:]).apply(np.tile(pa, (len(pp), 1)))
        c = cp[:, :3] + Rotation.from_quat(cp[:, 3:]).apply(np.tile(ca, (len(cp), 1)))
        errors.append(float(np.linalg.norm(p - c, axis=1).max()))
    slew = np.unwrap(Rotation.from_quat(poses[:, start + world["crane_slew"], 3:]).as_euler("xyz")[:, 2])
    collapsed = (drops > 0.06) | (rotation.magnitude() > np.deg2rad(35))
    result = {
        "ball_building_contact_time_s": impact_time,
        "pre_impact_structure_drift_mm": 1000 * stability,
        "peak_ball_speed_m_s": float(speed.max()),
        "collapsed_upper_panels": int(collapsed.sum()),
        "upper_panel_drops_m": drops.tolist(),
        "maximum_chain_anchor_error_mm": 1000 * max(errors),
        "crane_slew_degrees": float(np.rad2deg(np.ptp(slew))),
    }
    result["pass"] = bool(
        impact_time is not None
        and impact_time > 3.5
        and stability < 0.003
        and speed.max() > 0.45
        and collapsed.sum() >= 3
        and max(errors) < 0.002
        and result["crane_slew_degrees"] > 70
    )
    return result


def audit(folder):
    from scipy.spatial.transform import Rotation

    data = np.load(folder / "trace.npz")
    poses, fps = data["poses"], int(data["fps"])
    summary = json.loads((folder / "model-summary.json").read_text())
    result = {"finite": bool(np.isfinite(poses).all()), "tasks": []}
    for world in summary["worlds"]:
        kind = world["kind"]
        start = world["body_start"]
        if "tracked_body" not in world:
            continue
        obj = poses[:, start + world["tracked_body"]]
        check = {"id": world["id"]}
        if not np.isfinite(poses[:, start : start + world["body_count"]]).all():
            check.update({"pass": False, "reason": "Non-finite world state"})
            result["tasks"].append(check)
            continue
        if world.get("demolition"):
            check.update(demolition_check(world, poses, fps))
        elif "catalog_task" in world:
            check.update(catalog_check(world, poses, fps))
        elif kind == "toy":
            chassis = Rotation.from_quat(obj[:, 3:])
            forward = chassis.apply(np.tile([0, 1, 0], (len(obj), 1)))
            distance = float(np.sum(np.diff(obj[:, :3], axis=0) * forward[:-1]))
            wheels = world.get("wheel_bodies") or [
                i - start
                for i, label in enumerate(summary["body_labels"])
                if start <= i < start + world["body_count"] and "/wheel." in label
            ]
            radius = world.get("wheel_radius", 0.03 if world["variant"] % 2 == 0 else 0.0375)
            angles = []
            for wheel in wheels:
                rotation = chassis.inv() * Rotation.from_quat(poses[:, start + wheel, 3:])
                steps = (rotation[:-1].inv() * rotation[1:]).as_rotvec()
                angles.append(float(np.sum(steps[:, 0])))
            slip = np.abs(np.asarray(angles) * radius + distance) / max(abs(distance), 0.01)
            check.update(
                forward_travel_m=distance,
                wheel_turns=(np.asarray(angles) / (2 * np.pi)).tolist(),
                rolling_error=slip.tolist(),
            )
            check["pass"] = bool(distance > 0.45 and np.max(slip) < 0.30)
        elif kind == "hand":
            palm = poses[:, start + world["palm_body"]]
            relative = Rotation.from_quat(palm[:, 3:]).inv() * Rotation.from_quat(obj[:, 3:])
            motion = (relative[2 * fps].inv() * relative[2 * fps :]).magnitude()
            check["finger_driven_rotation_degrees"] = float(np.rad2deg(np.max(motion)))
            increments = (relative[:-1].inv() * relative[1:]).as_rotvec()
            spin = np.cumsum(increments[:, 0])
            check["sustained_spin_degrees"] = float(np.rad2deg(abs(spin[-1] - spin[2 * fps])))
            check["wrist_travel_mm"] = float(1000 * np.max(np.linalg.norm(palm[:, :3] - palm[0, :3], axis=1)))
            check["wrist_rotation_degrees"] = float(
                np.rad2deg((Rotation.from_quat(palm[0, 3:]).inv() * Rotation.from_quat(palm[:, 3:])).magnitude().max())
            )
            check["pass"] = bool(
                check["finger_driven_rotation_degrees"] >= 120
                and check["sustained_spin_degrees"] >= 180
                and check["wrist_travel_mm"] < 15
                and check["wrist_rotation_degrees"] < 5
                and np.max(np.linalg.norm(obj[:, :3] - palm[:, :3], axis=1)) < 0.18
            )
        elif kind == "gear":
            if "crane_slew" not in world:
                check.update({"pass": False, "reason": "Old gear rotation does not perform a load-handling task"})
                result["tasks"].append(check)
                continue
            slew = Rotation.from_quat(poses[:, start + world["crane_slew"], 3:])
            angles = np.unwrap(slew.as_euler("xyz")[:, 2])
            lift = float(obj[:, 2].max() - obj[0, 2])
            travel = float(np.linalg.norm(obj[-1, :2] - obj[0, :2]))
            check.update(load_lift_m=lift, load_transfer_m=travel, crane_slew_degrees=float(np.rad2deg(np.ptp(angles))))
            goal = np.asarray(world.get("crane_load_goal", [0.518, 0.25, 0.361]))
            eye = obj[:, :3] + Rotation.from_quat(obj[:, 3:]).apply(np.tile([0, 0, 0.054], (len(obj), 1)))
            hook_body = poses[:, start + world["crane_hook"]]
            # Proc-gen bodies retain object-space mesh coordinates. The hook
            # bowl is offset from its link origin and must rotate with it.
            hook_local = world.get("crane_hook_contact_local", [0, -0.338, 0.416])
            hook = hook_body[:, :3] + Rotation.from_quat(hook_body[:, 3:]).apply(np.tile(hook_local, (len(obj), 1)))
            check["max_transport_hook_eye_distance_mm"] = float(
                1000 * np.linalg.norm(eye[3 * fps : 8 * fps] - hook[3 * fps : 8 * fps], axis=1).max()
            )
            check["supported_placement_error_mm"] = float(1000 * np.linalg.norm(obj[-1, :3] - goal))
            check["final_drift_mm"] = float(1000 * np.linalg.norm(obj[-fps:, :3] - obj[-1, :3], axis=1).max())
            check["pass"] = bool(
                lift > 0.06
                and travel > 0.25
                and check["crane_slew_degrees"] > 60
                and check["max_transport_hook_eye_distance_mm"] < 70
                and check["supported_placement_error_mm"] < 25
                and check["final_drift_mm"] < 2
            )
        elif world.get("lighter"):
            palm = poses[:, start + world["palm_body"]]
            body_rotation = Rotation.from_quat(obj[:, 3:])
            lid_rotation = Rotation.from_quat(poses[:, start + world["lighter_lid"], 3:])
            angle = world.get("lighter_axis_sign", 1) * np.unwrap(
                (body_rotation.inv() * lid_rotation).as_rotvec()
                @ np.asarray(world.get("lighter_hinge_axis", [1, 0, 0]))
            )
            relative = Rotation.from_quat(palm[:, 3:]).inv().apply(obj[:, :3] - palm[:, :3])
            check.update(
                case_tilt_degrees=float(
                    np.rad2deg(np.arccos(np.clip(body_rotation.as_matrix()[:, 2, 2], -1, 1))).max()
                ),
                lid_opening_degrees=float(np.rad2deg(angle.max())),
                final_lid_angle_degrees=float(np.rad2deg(angle[-1])),
                pre_action_lid_angle_degrees=float(np.rad2deg(np.abs(angle[: int(1.5 * fps)]).max())),
                case_grasp_slip_mm=float(1000 * np.linalg.norm(relative - relative[fps], axis=1).max()),
                unpowered_hinge=not world["lighter_hinge_actuated"] and not world.get("force_authoring"),
                force_authoring=world.get("force_authoring"),
            )
            check["pass"] = bool(
                not world.get("lighter_force_reference", False)
                and not world.get("lighter_force_fitted", False)
                and check["unpowered_hinge"]
                and check["lid_opening_degrees"] > 80
                and check["final_lid_angle_degrees"] > 75
                and check["pre_action_lid_angle_degrees"] < 8
                and check["case_grasp_slip_mm"] < 20
                and check["case_tilt_degrees"] < 30
            )
        elif world.get("plate_rack"):
            rotation = Rotation.from_quat(obj[:, 3:])
            normal = rotation.apply(np.tile([0, 0, 1], (len(obj), 1)))
            center = obj[:, :3] + rotation.apply(np.tile([0, 0, 0.012], (len(obj), 1)))
            rest = poses[-1, [start + i for i in world["plate_resting_bodies"]], :3]
            check.update(
                plate_lift_m=float(obj[:, 2].max() - obj[0, 2]),
                slot_center_error_mm=float(1000 * abs(center[-1, 0] - world["plate_slot_x"])),
                normal_tilt_degrees=float(np.rad2deg(np.arccos(np.clip(abs(normal[-1, 0]), 0, 1)))),
                final_drift_mm=float(1000 * np.linalg.norm(obj[-fps:, :3] - obj[-1, :3], axis=1).max()),
                two_preloaded_plates_retained=bool(
                    np.all(np.abs(rest[:, 1] - world["rack_origin"][1]) < 0.10)
                    and np.all((rest[:, 2] > 0.06) & (rest[:, 2] < 0.16))
                ),
            )
            check["pass"] = bool(
                check["plate_lift_m"] > 0.15
                and check["slot_center_error_mm"] < 12
                and check["normal_tilt_degrees"] < 18
                and abs(obj[-1, 1] - world["rack_origin"][1]) < 0.10
                and 0.06 < obj[-1, 2] < 0.16
                and check["final_drift_mm"] < 2
                and check["two_preloaded_plates_retained"]
            )
        elif kind == "shadow":
            if "hardware_bodies" not in world:
                check.update({"pass": False, "reason": "Old finger-motion scene has no pouring task"})
                result["tasks"].append(check)
                continue
            parts = poses[:, [start + i for i in world.get("poured_bodies", world.get("hardware_bodies", []))], :3]
            bounds = np.asarray(world.get("hardware_bin_bounds", [[-0.01, 0.06, 0.02], [0.37, 0.44, 0.17]]))
            inside = np.all((parts[-1] > bounds[0]) & (parts[-1] < bounds[1]), axis=1)
            check.update(mug_lift_m=float(obj[:, 2].max() - obj[0, 2]), hardware_poured_count=int(np.sum(inside)))
            palm = poses[:, start + world["palm_body"]]
            rotation = Rotation.from_quat(palm[:, 3:])
            relative_position = rotation.inv().apply(obj[:, :3] - palm[:, :3])
            relative_rotation = rotation.inv() * Rotation.from_quat(obj[:, 3:])
            hold = slice(int(2.6 * fps), len(obj))
            check["mug_grasp_slip_mm"] = float(
                1000 * np.linalg.norm(relative_position[hold] - relative_position[int(2.6 * fps)], axis=1).max()
            )
            check["mug_grasp_rotation_degrees"] = float(
                np.rad2deg((relative_rotation[int(2.6 * fps)].inv() * relative_rotation[hold]).magnitude().max())
            )
            check["pass"] = bool(
                check["mug_lift_m"] > 0.20
                and check["hardware_poured_count"] >= 5
                and check["mug_grasp_slip_mm"] < 30
                and check["mug_grasp_rotation_degrees"] < 20
            )
        elif kind == "spill":
            parts = poses[:, [start + i for i in world.get("poured_bodies", world.get("hardware_bodies", []))], :3]
            x = parts[-1, :, 0]
            lane_offset = np.min(np.abs(x[:, None] - np.array([0.04, 0.23, 0.42])), axis=1)
            inside = (lane_offset < 0.080) & (parts[-1, :, 1] > 0.69) & (parts[-1, :, 1] < 0.923)
            inside &= (parts[-1, :, 2] > 0.035) & (parts[-1, :, 2] < 0.12)
            check["terminal_pocket_arrivals"] = int(inside.sum())
            check["pass"] = bool(check["terminal_pocket_arrivals"] >= 12)
        elif kind == "insert":
            euler = Rotation.from_quat(obj[-1, 3:]).as_euler("xyz")
            twist = world.get("key_twist", 0.30 + world["variant"] * 0.07)
            period = 2 * np.pi / (5 + world["variant"] % 2)
            clocking = abs((euler[2] - twist + period / 2) % period - period / 2)
            check.update(
                lateral_error_mm=float(1000 * np.linalg.norm(obj[-1, :2] - [0.16, 0.23])),
                depth_error_mm=float(1000 * abs(obj[-1, 2] - 0.067)),
                tilt_degrees=float(np.rad2deg(np.linalg.norm(euler[:2]))),
                clocking_degrees=float(np.rad2deg(clocking)),
            )
            check["pass"] = bool(
                check["lateral_error_mm"] < 1.2
                and check["depth_error_mm"] < 1.5
                and check["tilt_degrees"] < 1
                and check["clocking_degrees"] < 1
            )
        elif kind == "pile":
            check.update(pile_collection(world, poses, fps))
            check["pass"] = bool(
                "pile_deck_height" in world and check["dropped_into_bin_count"] >= world.get("pile_required_count", 10)
            )
        elif kind == "drawer":
            drawer = poses[:, start + world["drawer_body"], :3]
            travel = float(drawer[0, 1] - np.min(drawer[: 5 * fps, 1]))
            check["handle_pulled_travel_m"] = travel
            check["unpowered_slide"] = world.get("drawer_actuator_stiffness") == 0
            fork = poses[:, start + world["placed_bodies"][0], :3]
            check["fork_lift_m"] = float(fork[:, 2].max() - fork[0, 2])
            check.update(drawer_placement(world, poses))
            if "drawer_ee" in world:
                hand = poses[:, start + world["drawer_ee"]]
                tcp = hand[:, :3] + Rotation.from_quat(hand[:, 3:]).apply(np.tile([0, 0, 0.1034], (len(hand), 1)))
                handle = drawer + world["drawer_handle_local"]
                error = tcp[2 * fps : 4 * fps] - handle[2 * fps : 4 * fps]
                # A bar permits grasping anywhere along its 200 mm length.
                # Measure distance to the physical bar axis, not its midpoint.
                error[:, 0] = np.maximum(np.abs(error[:, 0]) - 0.10, 0)
                check["max_handle_axis_error_mm"] = float(1000 * np.max(np.linalg.norm(error, axis=1)))
            check["pass"] = bool(
                check["unpowered_slide"]
                and travel > 0.40
                and check.get("max_handle_axis_error_mm", 999) < 15
                and check["fork_lift_m"] > 0.18
                and check["fork_in_tray"]
            )
        else:
            continue
        result["tasks"].append(check)
    result["pass"] = result["finite"] and all(t["pass"] for t in result["tasks"])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.folder)
    (args.output or args.folder / "strict-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
