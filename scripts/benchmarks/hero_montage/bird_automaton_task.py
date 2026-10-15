# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""A robot cranks the supplied bird through passive gear and cam contacts."""

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import warp as wp
from robots import make_arm

import newton

ASSET = Path(__file__).resolve().parent / "assets" / "bird_automaton"
SCALE = 2.5


def mechanism_xml():
    """Remove the motor and give the existing CAD handle its matching collider."""
    root = ET.parse(ASSET / "bird_contact_gears.xml").getroot()
    root.find("compiler").set("meshdir", str(ASSET / "cad"))
    root.remove(root.find("actuator"))
    # add_mjcf scales geometry and joint anchors, but keeps slide ranges in meters.
    for joint in root.findall(".//joint[@type='slide']"):
        joint.set("range", " ".join(str(float(value) * SCALE) for value in joint.get("range").split()))
    # Explicit disjoint masks preserve the supplied contact pairs in Newton.
    geometries = {g.get("name"): g for g in root.findall(".//worldbody//geom")}
    for bit, pair in enumerate(root.findall("contact/pair"), start=1):
        for key in ("geom1", "geom2"):
            geom = geometries[pair.get(key)]
            geom.set("contype", str(1 << bit))
            geom.set("conaffinity", str(1 << bit))
    for geom in geometries.values():
        if geom.get("name", "").endswith("_visual"):
            geom.set("contype", "0")
            geom.set("conaffinity", "0")
    # Match the handle already present in input_gear.stl: build_cad.py uses
    # cyl(4, 17, (-21, 0, -23), "x") in millimeters. No added visual or hub.
    rotor = root.find("worldbody/body[@name='input_rotor']")
    ET.SubElement(
        rotor,
        "geom",
        name="crank_handle_contact",
        type="cylinder",
        pos="-0.021 0 -0.023",
        size="0.004 0.0085",
        quat="0.7071067812 0 0.7071067812 0",
        rgba="0.5 0.5 0.5 0",
        group="3",
        contype="64",
        conaffinity="64",
        friction="1.2 0.001 0.0001",
        condim="3",
    )
    return ET.tostring(root, encoding="unicode")


def populate(b, info, variant, assets, device):
    from scene import plan_ik, tf  # noqa: PLC0415 -- shared scene helpers

    origin = np.array([-0.05, 0.20, 0.0])
    rotate = wp.quat_rpy(0.0, 0.0, math.pi / 2)
    crank_center = origin + np.array([0.100, -0.089, 0.065]) * SCALE
    radius = 0.023 * SCALE
    ee, _, fingers = make_arm(b, assets, device, robot="ur10", root=(-0.48, -0.30, 0), instrument=False)
    # Seed the elbow-up branch with enough wrist travel for the whole crank arc.
    b.joint_q[:6] = [2.9705, -0.9928, 1.9579, -0.9651, -0.1711, -4.7124]
    orientation = wp.quat_rpy(-math.pi / 2, 0.0, 0.0) * wp.quat_rpy(0.0, 0.0, -math.pi / 2)
    poses = []
    for t in np.linspace(0, 12, 241):
        u = np.clip((t - 1.5) / 8.5, 0, 1)
        angle = 1.5 * math.tau * (u * u * (3 - 2 * u))
        position = crank_center + np.array([-radius * math.sin(angle), 0, -radius * math.cos(angle)])
        aperture = float(0.025 - 0.018 * np.clip(t / 1.0, 0, 1))
        poses.append((float(t), position, aperture, wp.quat_rpy(0.0, float(angle), 0.0) * orientation))
    for i in range(6):
        b.joint_target_ke[i], b.joint_target_kd[i] = 16000, 220
    for i in fingers:
        b.joint_target_ke[i], b.joint_target_kd[i], b.joint_effort_limit[i] = 1200, 20, 12
    info["waypoints"] = plan_ik(b, ee, (0, 0, 0.1034), poses, device, desired_rotation=orientation)
    for i, owner in enumerate(b.shape_body):
        if owner >= 0 and b.body_label[owner].endswith(("/fr3_leftfinger", "/fr3_rightfinger")):
            b.shape_material_mu[i] = 1.2
    first_shape, first_dof = b.shape_count, b.joint_dof_count
    b.add_mjcf(
        mechanism_xml(),
        xform=tf(origin, rotate),
        scale=SCALE,
        parse_mujoco_options=False,
        enable_self_collisions=True,
        ignore_inertial_definitions=False,
    )
    # Only the robot is actuated. Every automaton bearing, slider, wing and jaw is passive.
    for i in range(first_dof, b.joint_dof_count):
        b.joint_target_ke[i] = 0
        b.joint_target_kd[i] = 0
        b.joint_target_mode[i] = int(newton.JointTargetMode.NONE)
    for i in range(first_shape, b.shape_count):
        b.shape_gap[i] = 0.000025
        if b.shape_flags[i] & int(newton.ShapeFlags.VISIBLE) and b.shape_type[i] == int(newton.GeoType.MESH):
            source = b.shape_source[i]
            vertices = np.asarray(source.vertices)[np.asarray(source.indices).reshape(-1, 3)]
            normals = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
            normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-12)
            b.shape_source[i] = newton.Mesh(
                vertices.reshape(-1, 3),
                np.arange(vertices.size // 3, dtype=np.int32),
                normals=np.repeat(normals, 3, axis=0),
                compute_inertia=False,
            )
        if b.shape_flags[i] & int(newton.ShapeFlags.COLLIDE_SHAPES):
            b.shape_flags[i] &= ~int(newton.ShapeFlags.VISIBLE)
            b.shape_material_mu[i] = 1.2 if b.shape_label[i].endswith("crank_handle_contact") else 0.03
    b.approximate_meshes(
        "convex_hull",
        keep_visual_shapes=False,
        shape_indices=[
            i for i in range(first_shape, b.shape_count) if b.shape_flags[i] & int(newton.ShapeFlags.COLLIDE_SHAPES)
        ],
    )
    bodies = {
        name: next(i for i, label in enumerate(b.body_label) if label.endswith(name))
        for name in ("input_rotor", "output_rotor", "wing_left", "wing_right", "jaw")
    }
    bodies["hand"] = ee
    info.update(
        robot="ur10",
        bird_handle_local=(np.array([-0.021, 0, -0.023]) * SCALE).tolist(),
        bird_tcp_local=[0, 0, 0.1034],
        tracked_body=bodies["output_rotor"],
        bird_bodies=bodies,
        dense_waypoints=True,
        phase_delay=0,
        task="Crank a bird automaton through gear teeth and passive cam followers",
    )


def check(world, poses, fps):
    from scipy.spatial.transform import Rotation

    def angle(name, axis):
        q = poses[:, world["body_start"] + world["bird_bodies"][name], 3:]
        rotation = Rotation.from_quat(q)
        relative = rotation[0].inv() * rotation
        # Quaternion twist about the body's local bearing axis, unwrapped.
        quat = relative.as_quat()
        return np.unwrap(2 * np.arctan2(quat[:, axis], quat[:, 3]))

    crank, output = angle("input_rotor", 0), angle("output_rotor", 0)
    crank_pose = poses[:, world["body_start"] + world["bird_bodies"]["input_rotor"]]
    hand_pose = poses[:, world["body_start"] + world["bird_bodies"]["hand"]]
    hand_point = hand_pose[:, :3] + Rotation.from_quat(hand_pose[:, 3:]).apply(world["bird_tcp_local"])
    handle_point = crank_pose[:, :3] + Rotation.from_quat(crank_pose[:, 3:]).apply(world["bird_handle_local"])
    grip_error = np.linalg.norm(hand_point - handle_point, axis=1)
    local_path = Rotation.from_quat(crank_pose[0, 3:]).inv().apply(hand_point - crank_pose[0, :3])
    orbit = np.unwrap(np.arctan2(local_path[:, 1], -local_path[:, 2]))
    orbit_radius = np.linalg.norm(local_path[:, 1:3], axis=1)
    wings = [float(np.rad2deg(np.ptp(angle(name, 0)))) for name in ("wing_left", "wing_right")]
    jaw = float(np.rad2deg(np.ptp(angle("jaw", 1))))
    result = {
        "crank_turns": float((crank[-1] - crank[0]) / math.tau),
        "output_turns": float((output[-1] - output[0]) / math.tau),
        "wing_sweeps_deg": wings,
        "jaw_sweep_deg": jaw,
        "gear_ratio_error_rad": float(np.max(np.abs(output - crank * 24 / 36))),
        "hand_orbit_turns": float((orbit[-1] - orbit[0]) / math.tau),
        "hand_orbit_radius_m": float(np.mean(orbit_radius)),
        "maximum_handle_grasp_error_m": float(np.max(grip_error[int(fps) :])),
    }
    result["pass"] = bool(
        1.4 < result["crank_turns"] < 1.6
        and min(wings) > 60
        and jaw > 25
        and result["gear_ratio_error_rad"] < 0.1
        and 1.4 < result["hand_orbit_turns"] < 1.6
        and 0.053 < result["hand_orbit_radius_m"] < 0.062
        and result["maximum_handle_grasp_error_m"] < 0.005
    )
    return result
