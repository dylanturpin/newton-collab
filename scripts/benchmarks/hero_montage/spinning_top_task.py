# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Spin a free weighted toy through a motorized chuck, then release it."""

import math
from itertools import pairwise

import numpy as np
from robots import franka_hand_xml, make_arm

import newton


def add_top(b, position):
    from scene import COLORS, tf  # noqa: PLC0415 -- shared task helpers
    from scipy.spatial import ConvexHull

    body = b.add_body(xform=tf(position), label="spinning_top/free_toy")
    cfg = b.default_shape_cfg.copy()
    cfg.gap, cfg.restitution, cfg.mu = 0.0001, 0.0, 0.35
    cfg.density = 7800
    tip = b.add_shape_sphere(body, xform=tf((0, 0, 0.004)), radius=0.004, cfg=cfg, color=(0.3, 0.34, 0.38))
    b.shape_flags[tip] &= ~int(newton.ShapeFlags.VISIBLE)

    def hull(points, density):
        vertices = np.asarray(points, dtype=np.float32)
        cooked = ConvexHull(vertices)
        faces = cooked.simplices.copy()
        for i, face in enumerate(faces):
            p = vertices[face]
            if np.dot(np.cross(p[1] - p[0], p[2] - p[0]), cooked.equations[i, :3]) < 0:
                faces[i] = face[::-1]
        cfg.density = density
        si = b.add_shape_mesh(body, mesh=newton.Mesh(vertices, faces.ravel()), cfg=cfg, label="top/collision")
        b.shape_flags[si] &= ~int(newton.ShapeFlags.VISIBLE)

    # The tapered foot holds the painted body visibly above the tabletop.
    hull(
        [
            (r * math.cos(a), r * math.sin(a), z)
            for r, z in ((0.003, 0.004), (0.008, 0.026))
            for a in np.linspace(0, math.tau, 25)[:-1]
        ],
        300,
    )
    # A light shell over a dense low annular rim: geometry determines the inertia.
    hull(
        [
            (r * math.cos(a), r * math.sin(a), z)
            for r, z in ((0.008, 0.026), (0.075, 0.028), (0.075, 0.031), (0.020, 0.058))
            for a in np.linspace(0, math.tau, 49)[:-1]
        ],
        80,
    )
    for i in range(24):
        hull(
            [
                (r * math.cos(a), r * math.sin(a), z)
                for z in (0.025, 0.031)
                for r in (0.060, 0.075)
                for a in (i * math.tau / 24, (i + 1) * math.tau / 24)
            ],
            7800,
        )
    cfg.density = 300
    stem = b.add_shape_cylinder(
        body, xform=tf((0, 0, 0.072)), radius=0.006, half_height=0.014, cfg=cfg, label="top/stem"
    )
    b.shape_flags[stem] &= ~int(newton.ShapeFlags.VISIBLE)
    visible = cfg.copy()
    visible.density = 0
    visible.has_shape_collision = False
    visible.has_particle_collision = False
    # Rounded tip is visible below the painted body.
    b.add_shape_sphere(body, xform=tf((0, 0, 0.004)), radius=0.004, cfg=visible, color=(0.3, 0.34, 0.38))
    profile = [
        (0.003, 0.004),
        (0.008, 0.026),
        (0.060, 0.025),
        (0.075, 0.025),
        (0.075, 0.031),
        (0.020, 0.058),
        (0.006, 0.058),
        (0.006, 0.086),
        (0, 0.086),
    ]
    for sector in range(3):
        vertices, normals, faces = [], [], []
        for a, c in pairwise(profile):
            tangent = np.subtract(c, a)
            normal = np.array([tangent[1], -tangent[0]])
            normal /= np.linalg.norm(normal)
            start = len(vertices)
            for r, z in (a, c):
                for j in range(49):
                    angle = (sector + j / 48) * math.tau / 3
                    vertices.append((r * math.cos(angle), r * math.sin(angle), z))
                    normals.append((normal[0] * math.cos(angle), normal[0] * math.sin(angle), normal[1]))
            for j in range(48):
                k = start + j
                faces.extend((k, k + 1, k + 50, k, k + 50, k + 49))
        mesh = newton.Mesh(
            np.asarray(vertices, dtype=np.float32),
            np.asarray(faces, dtype=np.int32),
            normals=np.asarray(normals, dtype=np.float32),
            compute_inertia=False,
            roughness=0.3,
        )
        b.add_shape_mesh(body, mesh=mesh, cfg=visible, color=COLORS[sector], label=f"top/painted-sector-{sector}")
    return body


def populate(b, info, variant, assets, device):
    from scene import plan_ik, tf  # noqa: PLC0415 -- shared task helpers

    ee, down, _ = make_arm(b, assets, device, robot="ur10", root=(-0.48, 0, 0), gripper=False, instrument=False)
    target = np.array([0.12, 0.04, 0.076])
    poses = [
        (0, target, None),
        (2.5, target, None),
        (3.4, target + np.array([0, 0, 0.23]), None),
        (4.4, target + np.array([-0.25, -0.25, 0.23]), None),
        (9, target + np.array([-0.25, -0.25, 0.23]), None),
    ]
    # IK handles only the arm; the independent rotary chuck is a real driven joint.
    path = plan_ik(b, ee, (0, 0, 0.1634), poses, device, desired_rotation=down, cartesian_step=0.1)
    rotor_q = b.joint_coord_count
    b.add_urdf(
        """<robot name="rotary_chuck">
      <link name="mount"><inertial><mass value="0.12"/><inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/></inertial>
      <visual><origin xyz="0 0 0.0225"/><geometry><cylinder radius="0.032" length="0.045"/></geometry><material name="housing"><color rgba="0.16 0.2 0.24 1"/></material></visual></link>
      <link name="rotor"><inertial><mass value="0.06"/><inertia ixx="0.00005" ixy="0" ixz="0" iyy="0.00005" iyz="0" izz="0.00005"/></inertial>
      <visual><geometry><cylinder radius="0.026" length="0.01"/></geometry><material name="collar"><color rgba="0.9 0.58 0.08 1"/></material></visual></link>
      <joint name="spindle" type="continuous"><parent link="mount"/><child link="rotor"/>
      <origin xyz="0 0 0.045"/><axis xyz="0 0 1"/><limit effort="1.0" velocity="30"/><dynamics damping="0.005"/></joint>
    </robot>""",
        parent_body=ee,
        floating=False,
        enable_self_collisions=False,
    )
    rotor = next(i for i, label in enumerate(b.body_label) if label.endswith("/rotor"))
    b.add_urdf(
        franka_hand_xml(assets["franka_emika_panda"]),
        parent_body=rotor,
        xform=tf((0, 0, 0.015)),
        floating=False,
        enable_self_collisions=False,
    )
    b.approximate_meshes("convex_hull", keep_visual_shapes=True)
    for vi in range(6):
        b.joint_target_ke[vi], b.joint_target_kd[vi] = 8000, 180
    b.joint_target_ke[rotor_q], b.joint_target_kd[rotor_q] = 8, 0.08
    b.joint_effort_limit[rotor_q] = 1
    b.joint_target_mode[rotor_q] = int(newton.JointTargetMode.POSITION)
    for vi in (rotor_q + 1, rotor_q + 2):
        b.joint_q[vi] = 0.0
        b.joint_target_ke[vi], b.joint_target_kd[vi], b.joint_effort_limit[vi] = 2000, 20, 30
        b.joint_target_mode[vi] = int(newton.JointTargetMode.POSITION)
    for si, owner in enumerate(b.shape_body):
        if owner >= 0 and b.body_label[owner].endswith(("/fr3_leftfinger", "/fr3_rightfinger")):
            b.shape_material_mu[si] = 1.3
    for i, a in enumerate([i for i, owner in enumerate(b.shape_body) if owner >= 0]):
        for c in [j for j, owner in enumerate(b.shape_body) if owner >= 0][i + 1 :]:
            b.add_shape_collision_filter_pair(a, c)
    spindle_path = [(0, 0.0)]
    for t in np.arange(0.6, 3.61, 0.02):
        if t <= 1.6:
            angle = 10 * (t - 0.6) ** 2
        elif t <= 2.6:
            angle = 10 + 20 * (t - 1.6)
        else:
            u = t - 2.6
            angle = 30 + 20 * u - 10 * u**2
        spindle_path.append((float(t), float(angle)))
    spindle_path.append((9, 40.0))
    info.update(
        waypoints=path,
        dense_waypoints=True,
        phase_delay=0,
        robot="ur10+rotary_chuck",
        extra_drives=[(rotor_q, spindle_path)],
    )
    for vi in (rotor_q + 1, rotor_q + 2):
        info["extra_drives"].append((vi, [(0, 0.0), (2.2, 0.0), (2.45, 0.04), (9, 0.04)]))
    toy = add_top(b, (0.12, 0.04, 0))
    info.update(
        tracked_body=toy,
        release_time=2.45,
        rotor_q=rotor_q,
        task="Spin a free weighted top by gripping its stem with a rotary chuck, then release it onto its tip",
    )


def check(world, poses, fps):
    from scipy.spatial.transform import Rotation

    top = poses[:, world["body_start"] + world["tracked_body"]]
    rotation = Rotation.from_quat(top[:, 3:])
    axis = rotation.apply(np.tile([0, 0, 1], (len(top), 1)))
    angles = np.unwrap(rotation.as_euler("xyz")[:, 2])
    start = round(3 * fps)
    turns = float((angles[-1] - angles[start]) / math.tau)
    result = {
        "free_spin_turns": turns,
        "minimum_free_axis_z": float(axis[start:, 2].min()),
        "max_free_tip_travel_m": float(np.linalg.norm(top[start:, :2] - top[start, :2], axis=1).max()),
    }
    result["pass"] = bool(
        abs(turns) > 3 and result["minimum_free_axis_z"] > 0.7 and result["max_free_tip_travel_m"] < 0.3
    )
    return result
