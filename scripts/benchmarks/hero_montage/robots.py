# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Interchangeable authentic arm models with a common physical gripper interface."""

import math
import xml.etree.ElementTree as ET
from functools import cache
from pathlib import Path

import warp as wp

import newton


@cache
def franka_hand_xml(asset_dir):
    """Extract the original hand subtree; preserve its meshes, inertia and joints."""
    source = ET.parse(Path(asset_dir) / "urdf/fr3_franka_hand.urdf").getroot()
    links = {"fr3_hand"}
    joints = []
    while True:
        added = [
            j
            for j in source.findall("joint")
            if j.find("parent").get("link") in links and j.find("child").get("link") not in links
        ]
        if not added:
            break
        joints.extend(added)
        links.update(j.find("child").get("link") for j in added)
    target = ET.Element("robot", name="franka_hand")
    for item in source:
        if item.tag == "material" or (item.tag == "link" and item.get("name") in links) or item in joints:
            target.append(item)
    for mesh in target.iter("mesh"):
        filename = mesh.get("filename")
        if filename.startswith("package://franka_emika_panda/"):
            mesh.set("filename", str(Path(asset_dir) / filename.split("franka_emika_panda/", 1)[1]))
    return ET.tostring(target, encoding="unicode")


def make_arm(b, assets, device, robot="franka", root=(-0.42, 0, 0), gripper=True, instrument=True):
    """Return end-effector body and control metadata for task-independent IK."""
    down = wp.quat(1.0, 0.0, 0.0, 0.0)
    if robot == "franka":
        b.add_urdf(
            str(Path(assets["franka_emika_panda"]) / "urdf/fr3_franka_hand.urdf"),
            xform=wp.transform(wp.vec3(*root), wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
        )
        b.joint_q[:9] = [0, -0.25, 0, -2.35, 0, 2.10, 0.7854, 0.04, 0.04]
        ee = next(i for i, label in enumerate(b.body_label) if label.endswith("/fr3_hand"))
        fingers = [7, 8]
    else:
        menagerie = Path(assets["ur5e_menagerie"]).parent
        choices = {
            "kuka": ("kuka_iiwa_14/iiwa14.xml", [0, 0.5, 0, -1.3, 0, 0.8, 0]),
            "ur5": ("universal_robots_ur5e/ur5e.xml", [0, -1.2, 1.6, -1.95, -1.57, 0]),
            "ur10": ("universal_robots_ur10e/ur10e.xml", [0, -1.2, 1.6, -1.95, -1.57, 0]),
            "kinova": ("kinova_gen3/gen3.xml", [0, 0.262, math.pi, -2.269, 0, 0.960, 1.571]),
            "xarm": ("ufactory_xarm7/xarm7_nohand.xml", [0, -0.247, 0, 0.909, 0, 1.15644, 0]),
        }
        filename, initial = choices[robot]
        b.add_mjcf(
            str(menagerie / filename),
            xform=wp.transform(wp.vec3(*root), wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
        )
        b.joint_q[: len(initial)] = initial
        ee = b.body_count - 1
        fingers = []
        # Use the model's authored attachment site, not the last link origin.
        source = ET.parse(menagerie / filename).getroot()
        site = next(item for item in source.iter("site") if item.get("name") in ("attachment_site", "pinch_site"))
        position = wp.vec3(*map(float, site.get("pos", "0 0 0").split()))
        qw, qx, qy, qz = map(float, site.get("quat", "1 0 0 0").split())
        mount = wp.normalize(wp.quat(qx, qy, qz, qw))
        position += wp.quat_rotate(mount, wp.vec3(0.0, 0.0, 0.015))
        b.add_shape_cylinder(
            ee,
            xform=wp.transform(position - wp.quat_rotate(mount, wp.vec3(0.0, 0.0, 0.0075)), mount),
            radius=0.027,
            half_height=0.0075,
            color=(0.22, 0.25, 0.29),
            label="tool_mount_adapter",
        )
        if gripper:
            first = b.joint_coord_count
            b.add_urdf(
                franka_hand_xml(assets["franka_emika_panda"]),
                parent_body=ee,
                xform=wp.transform(position, mount),
                floating=False,
                enable_self_collisions=False,
            )
            ee = next(i for i, label in enumerate(b.body_label) if label.endswith("/fr3_hand"))
            fingers = [first, first + 1]
            b.joint_q[first : first + 2] = [0.04, 0.04]
        else:
            b.add_urdf(
                '<robot name="instrument"><link name="tool_frame"><inertial><mass value="0.02"/><inertia ixx="0.00001" ixy="0" ixz="0" iyy="0.00001" iyz="0" izz="0.00001"/></inertial></link></robot>',
                parent_body=ee,
                xform=wp.transform(position, mount),
                floating=False,
                enable_self_collisions=False,
            )
            ee = b.body_count - 1
            if instrument:
                b.add_shape_cylinder(
                    ee,
                    xform=wp.transform(wp.vec3(0.0, 0.0, 0.060), wp.quat_identity()),
                    radius=0.012,
                    half_height=0.060,
                    color=(0.27, 0.30, 0.34),
                    label="instrument_shaft",
                )
    count = b.joint_dof_count
    for i in range(count):
        b.joint_target_ke[i] = 3000 if gripper else 1800
        b.joint_target_kd[i] = 100 if gripper else 80
        b.joint_armature[i] = 0.08
        b.joint_target_mode[i] = int(newton.JointTargetMode.POSITION)
        b.joint_effort_limit[i] = 100
    for i in fingers:
        b.joint_target_ke[i] = 1000
        b.joint_target_kd[i] = 18
        b.joint_effort_limit[i] = 30
    b.joint_target_q[:] = b.joint_q[:]
    b._hero_grip_dofs = fingers
    b._hero_driven_dofs = count
    b._hero_robot = robot
    b.approximate_meshes("convex_hull", keep_visual_shapes=True)
    return ee, down, fingers
