# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""A floating authentic Shadow hand opens the passive lid of a free lighter."""

import json
import math
import os
from pathlib import Path

import numpy as np
import warp as wp
from task_assets import ROOT, import_object, load

import newton


def populate(b, info, assets, device):
    from scene import tf  # noqa: PLC0415 -- shared task helpers

    asset = Path(assets["wonik_allegro"]).parent / "shadow_hand" / "right_hand.xml"
    thumb_path = json.loads((ROOT / "lighter-thumb-path.json").read_text())
    candidates_path = os.environ.get("HERO_LIGHTER_CANDIDATES")
    if candidates_path:
        thumb_path = json.loads(Path(candidates_path).read_text())[info["variant"]]
    lighter_asset = thumb_path.get("lighter_asset", "retro_lighter")
    lighter_metadata, _ = load(lighter_asset)
    thumb_drive = thumb_path.get("thumb_drive", {"ke": 24, "kd": 0.7, "effort": 3})
    palm_position = np.array([0.18, -0.18, 0.37])
    # Native +X points toward the thumb and +Z along the four fingers.
    # Turn the whole palm sideways so the thumb points up and the fingers
    # wrap horizontally across the upright lighter.
    palm_rotation = wp.quat_rpy(0.0, -math.pi / 2, 0.0)
    root_rotation = (
        palm_rotation * wp.quat_rpy(0.0, 0.0, -math.pi / 2) * wp.quat_inverse(wp.normalize(wp.quat(1, 0, 1, 0)))
    )
    b.add_mjcf(
        str(asset),
        xform=tf(
            palm_position - np.asarray(wp.quat_rotate(palm_rotation, wp.vec3(0.0, -0.01, 0.2470098))), root_rotation
        ),
        floating=False,
        enable_self_collisions=False,
    )
    for j, label in enumerate(b.joint_label):
        if b.joint_type[j] == newton.JointType.FIXED:
            continue
        qi, vi = b.joint_q_start[j], b.joint_qd_start[j]
        short = label.rsplit("/", 1)[-1]
        value = (
            0.0
            if "WRJ" in short or short.endswith("J4")
            else 0.1
            if short.endswith("LFJ5")
            else {"3": 1.27, "2": 1.28, "1": 0.60}.get(short[-1], 0.0)
        )
        b.joint_q[qi] = np.clip(value, b.joint_limit_lower[vi], b.joint_limit_upper[vi])
        b.joint_target_ke[vi], b.joint_target_kd[vi] = (
            (500, 18) if "WRJ" in short else (thumb_drive["ke"], thumb_drive["kd"]) if "THJ" in short else (60, 1.2)
        )
        b.joint_effort_limit[vi] = 12 if "WRJ" in short else thumb_drive["effort"] if "THJ" in short else 3
        b.joint_armature[vi] = 0.001
        b.joint_target_mode[vi] = int(newton.JointTargetMode.POSITION)
    b.approximate_meshes("convex_hull", keep_visual_shapes=True)
    for si in range(b.shape_count):
        b.shape_material_mu[si] = 2.2
    palm = next(i for i, label in enumerate(b.body_label) if label.endswith("rh_palm"))
    thumb = next(
        i for i, label in enumerate(b.body_label) if label.endswith(thumb_path.get("contact_body", "rh_thdistal"))
    )
    indices = {
        label.rsplit("/", 1)[-1]: b.joint_q_start[j]
        for j, label in enumerate(b.joint_label)
        if b.joint_type[j] != newton.JointType.FIXED
    }
    for key, value in thumb_path.get("initial_grasp", {}).items():
        b.joint_q[indices[key]] = value
    points = []
    for item in thumb_path["waypoints"]:
        q = np.asarray(b.joint_q).copy()
        for key, value in thumb_path.get("closed_grasp", {}).items():
            q[indices[key]] = value
        for key, value in item["joints"].items():
            q[indices[key]] = value
        points.append((item["time"], q))
    if os.environ.get("HERO_LIGHTER_NO_THUMB") == "1":
        for _, q in points:
            for key in thumb_path["waypoints"][0]["joints"]:
                q[indices[key]] = points[0][1][indices[key]]
    for key, value in thumb_path["waypoints"][0]["joints"].items():
        b.joint_q[indices[key]] = value
    b.joint_target_q[:] = b.joint_q[:]
    if thumb_path.get("contact_feedback"):
        info["_thumb_ik_model"] = b.finalize(device=device)
        info["thumb_feedback"] = thumb_path["contact_feedback"]
        info["thumb_tip_point"] = thumb_path["contact_vertex"]
    count = b.joint_coord_count
    case_rotation = palm_rotation * wp.quat_rpy(
        thumb_path.get("case_pitch", 0.0), thumb_path.get("case_tilt", 0.0), thumb_path.get("case_yaw", 0.0)
    )
    case_rotation *= wp.quat_rpy(0.0, 0.0, float(thumb_path.get("case_clocking", 0.0)))
    mapping, qmap = import_object(
        b,
        lighter_asset,
        tf(
            palm_position + np.asarray(wp.quat_rotate(palm_rotation, wp.vec3(*thumb_path["case_local"]))), case_rotation
        ),
        free_root=True,
    )
    # A previously simulated, settled grasp can initialize the next episode.
    # This is applied once at construction, never during dynamics.
    settled = thumb_path.get("initial_joint_positions")
    if settled is not None:
        if len(settled) != b.joint_coord_count:
            raise ValueError("Settled lighter state does not match the model joint coordinates")
        b.joint_q[:] = settled
    for si, body in enumerate(b.shape_body):
        if body in mapping.values() and body != mapping["lid"]:
            b.shape_material_mu[si] = 1.4
    for j, label in enumerate(b.joint_label):
        if label == f"{lighter_asset}/lid.q":
            vi = b.joint_qd_start[j]
            b.joint_armature[vi] = 1e-5
            b.joint_target_ke[vi] = b.joint_target_kd[vi] = 0
            b.joint_damping[vi] = 0.0003
            b.joint_friction[vi] = 0.0002
            lid_v = vi
    thumb_bodies = [i for i, label in enumerate(b.body_label) if label.rsplit("/", 1)[-1].startswith("rh_th")]
    suppress_thumb_contact = os.environ.get("HERO_LIGHTER_NO_THUMB_CONTACT") == "1"
    if suppress_thumb_contact:
        b.shape_collision_filter_pairs.extend(
            (a, c)
            for a, body in enumerate(b.shape_body)
            if body in thumb_bodies
            for c, other in enumerate(b.shape_body)
            if other == mapping["lid"]
        )
    info.update(
        waypoints=points,
        phase_delay=thumb_path.get("phase_delay", info["variant"] * 0.25),
        arm_dofs=count,
        dense_waypoints=True,
        robot="shadow_hand",
        palm_body=palm,
        thumb_body=thumb,
        thumb_contact_bodies=thumb_bodies,
        thumb_lid_contact_disabled=suppress_thumb_contact,
        thumb_command_coordinates=[indices[f"rh_THJ{i}"] for i in (5, 4, 3, 2, 1)],
        thumb_command_rest=[thumb_path["waypoints"][0]["joints"][f"rh_THJ{i}"] for i in (5, 4, 3, 2, 1)],
        tracked_body=mapping[lighter_metadata["root"]],
        lighter=True,
        lighter_lid=mapping["lid"],
        lighter_hinge_actuated=False,
        contact_only_opening=True,
        lighter_configuration=thumb_path,
        thumb_frozen=os.environ.get("HERO_LIGHTER_NO_THUMB") == "1",
        lighter_axis_sign=1,
        lighter_hinge_axis=lighter_metadata["features"]["lid_axis"],
        lighter_hinge_anchor=lighter_metadata["features"]["lid_hinge"],
        lighter_lid_q=qmap["lid.q"],
        lighter_lid_v=lid_v,
        passive_cam={
            "spring_stiffness": 0.01,
            "open_rest_rad": 2.4,
            "closed_detent_torque": thumb_path.get("closed_detent_torque", 0.080),
            "detent_width_rad": 0.08,
            "damping": 0.001,
        },
        task="Open a free brass and lacquer lighter through thumb contact and friction",
    )
