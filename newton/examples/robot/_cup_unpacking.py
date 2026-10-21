# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Recover a buried ceramic mug through contact with lightweight packing pieces."""

import math

import numpy as np
import warp as wp

import newton
import newton.examples

from ._tabletop_assets import import_object
from ._tabletop_common import box, plan_ik, rotate_vectors, tf


def foam_mesh(radius, half, seed):
    """A softly irregular extruded packing piece with indented end faces."""
    rng = np.random.default_rng(seed)
    rings, sides = 12, 16
    vertices, normals, indices = [], [], []
    for j in range(rings + 1):
        v = j / rings
        z = (2 * v - 1) * (half + radius)
        if j in (0, rings):
            z *= 0.98
        bend = 0.001 * math.sin(math.pi * v)
        cap_z = z - np.clip(z, -half, half)
        radial_limit = math.sqrt(max(0, radius**2 - cap_z**2))
        r = max(0, radial_limit - bend - 0.0001)
        for k in range(sides):
            a = k * math.tau / sides
            rough = rng.uniform(0.97, 1.0)
            vertices.append((bend + r * math.cos(a) * rough, r * math.sin(a) * rough, z))
            normals.append((radial_limit * math.cos(a) / radius, radial_limit * math.sin(a) / radius, cap_z / radius))
    for j in range(rings):
        for k in range(sides):
            a, c = j * sides + k, j * sides + (k + 1) % sides
            indices.extend((a, c, a + sides, c, c + sides, a + sides))
    for j, sign in ((0, -1), (rings, 1)):
        center = len(vertices)
        vertices.append((0, 0, sign * (half + radius * 0.85)))
        normals.append((0, 0, sign))
        for k in range(sides):
            a, c = j * sides + k, j * sides + (k + 1) % sides
            indices.extend((center, c, a) if sign < 0 else (center, a, c))
    return newton.Mesh(
        np.asarray(vertices, dtype=np.float32),
        np.asarray(indices, dtype=np.int32),
        normals=np.asarray(normals, dtype=np.float32),
        compute_inertia=False,
    )


def populate(b, info, ee, down, fingers, device):

    center = np.array([0.18, 0.18, 0.0])
    source = center + np.array([0, 0, 0.010])
    dest = np.array([0.18, -0.36, 0.014])
    pickup = np.add(source, [0, 0, 0.047 * 0.85])
    target = np.add(dest, [0, 0, 0.047 * 0.85])
    for i in range(6):
        b.joint_target_ke[i], b.joint_target_kd[i] = 10000, 200
    for i in fingers:
        b.joint_target_ke[i], b.joint_target_kd[i], b.joint_effort_limit[i] = 2200, 18, 30
    poses = [
        (0, (0.02, -0.08, 0.43), 0.04),
        (3, (0.02, -0.08, 0.43), 0.04),
        (4, (0.09, 0.18, 0.28), 0.04),
        (5, (0.09, 0.18, 0.14), 0.04),
        (6, (0.18, 0.18, 0.12), 0.04),
        (7.2, pickup, 0.04),
        (8, pickup, 0.027),
        (9.8, (0.18, 0.18, 0.38), 0.027),
        (11, (0.18, -0.36, 0.38), 0.027),
        (12.5, target, 0.027),
        (13.2, target, 0.04),
        (14.5, (0.04, -0.38, 0.35), 0.04),
        (15, (0.04, -0.38, 0.35), 0.04),
    ]
    info["waypoints"] = plan_ik(b, ee, (0, 0, 0.1034), poses, device, desired_rotation=down, waypoint_dt=0.12)
    for i, owner in enumerate(b.shape_body):
        if owner >= 0 and b.body_label[owner].endswith(("/fr3_leftfinger", "/fr3_rightfinger")):
            b.shape_material_mu[i] = 1.2
    cardboard = (0.64, 0.51, 0.34)
    box(b, (center[0], center[1], 0.004), (0.185, 0.175, 0.004), cardboard, label="unpack/carton-bottom")
    for side in (-1, 1):
        box(b, (center[0] + side * 0.182, center[1], 0.10), (0.003, 0.175, 0.10), cardboard, label="unpack/carton-wall")
        box(b, (center[0], center[1] + side * 0.172, 0.10), (0.179, 0.003, 0.10), cardboard, label="unpack/carton-wall")
        b.add_shape_box(
            -1,
            xform=tf((center[0] + side * 0.227, center[1], 0.225), wp.quat_rpy(0.0, -side * 0.5, 0.0)),
            hx=0.052,
            hy=0.17,
            hz=0.002,
            color=(0.70, 0.57, 0.39),
            label="unpack/side-flap",
        )
        b.add_shape_box(
            -1,
            xform=tf((center[0], center[1] + side * 0.215, 0.225), wp.quat_rpy(side * 0.5, 0.0, 0.0)),
            hx=0.178,
            hy=0.052,
            hz=0.002,
            color=(0.70, 0.57, 0.39),
            label="unpack/end-flap",
        )
    b.add_shape_cylinder(
        -1,
        xform=tf((dest[0], dest[1], 0.005)),
        radius=0.065,
        half_height=0.005,
        color=(0.28, 0.34, 0.38),
        label="unpack/coaster",
    )
    mapping = import_object(b, "mug", tf(source), free_root=True, finish_color=(0.035, 0.43, 0.78))
    mug = mapping["blue_glaze"]
    for i, owner in enumerate(b.shape_body):
        if owner == mug:
            b.shape_material_mu[i] = 0.8
    rng = np.random.default_rng(73)
    pieces = []
    cfg = b.default_shape_cfg.copy()
    cfg.density, cfg.mu, cfg.restitution, cfg.gap = 18, 0.35, 0.02, 0.00015
    visual = cfg.copy()
    visual.density = 0
    meshes = [foam_mesh(0.0155, 0.014, seed) for seed in range(12)]
    for layer in range(11):
        for row in range(5):
            for column in range(5):
                pos = center + np.array([(column - 2) * 0.061, (row - 2) * 0.061, 0.125 + layer * 0.034])
                pos[:2] += rng.uniform(-0.0005, 0.0005, 2)
                angle = float(rng.uniform(0, math.tau))
                rot = wp.quat_rpy(math.pi / 2, 0.0, angle)
                body = b.add_body(xform=tf(pos, rot), label=f"unpack/foam-{len(pieces)}")
                shape = b.add_shape_capsule(
                    body,
                    radius=0.0155,
                    half_height=0.014,
                    cfg=cfg,
                    color=(0.94, 0.94, 0.90),
                    label="unpack/foam-collision",
                )
                b.shape_flags[shape] &= ~int(newton.ShapeFlags.VISIBLE)
                shape = b.add_shape_mesh(
                    body, mesh=meshes[len(pieces) % 12], cfg=visual, color=(0.94, 0.94, 0.90), label="unpack/foam"
                )
                b.shape_flags[shape] = int(newton.ShapeFlags.VISIBLE)
                pieces.append(body)
    info.update(
        tracked_body=mug,
        unpack_pieces=pieces,
        placement_target=dest.tolist(),
        task="Clear lightweight packing and recover a buried ceramic mug",
    )


def check(info, poses, fps):
    """Measure the completed physical interaction from its recorded trajectory."""

    mug = poses[:, info["tracked_body"]]
    target = np.asarray(info["placement_target"])
    placement = float(np.linalg.norm(mug[-1, :3] - target))
    upright = float(rotate_vectors(mug[-1, 3:], [0, 0, 1])[2])
    foam = poses[:, np.asarray(info["unpack_pieces"]), :3]
    moved = int(np.count_nonzero(np.linalg.norm(foam[-1] - foam[3 * fps], axis=1) > 0.04))
    rise = float(np.max(mug[:, 2]) - mug[0, 2])
    return dict(
        placement_error_m=placement,
        upright=upright,
        packing_pieces_moved=moved,
        mug_lift_m=rise,
        **{"pass": bool(placement < 0.035 and upright > 0.95 and rise > 0.25 and moved > 8)},
    )
