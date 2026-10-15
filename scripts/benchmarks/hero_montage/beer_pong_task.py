# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Contact-driven grasp and release for a one-bounce table-tennis shot."""

import math

import numpy as np
import warp as wp
from robots import make_arm

import newton


def cup(b, center, color):
    """Build an open tapered cup, with matching convex wall sectors and rim."""
    from scene import tf  # noqa: PLC0415 -- shared task helpers
    from scipy.spatial import ConvexHull

    height, bottom, top, thickness = 0.12, 0.038, 0.058, 0.003
    cfg = b.default_shape_cfg.copy()
    cfg.restitution, cfg.mu, cfg.gap = 0.0, 0.3, 0.0002
    base = b.add_shape_cylinder(
        -1,
        xform=tf((*center, thickness / 2)),
        radius=bottom,
        half_height=thickness / 2,
        color=color,
        cfg=cfg,
        label="pong/cup-bottom",
    )
    b.shape_flags[base] &= ~int(newton.ShapeFlags.VISIBLE)
    sectors = 48
    for i in range(sectors):
        angles = [math.tau * i / sectors, math.tau * (i + 1) / sectors]
        vertices = np.array(
            [
                (r * math.cos(a), r * math.sin(a), z)
                for z, radius in ((thickness, bottom), (height, top))
                for r in (radius - thickness, radius)
                for a in angles
            ],
            dtype=np.float32,
        )
        # Each wedge is convex; the central cavity has no collision geometry.
        hull = ConvexHull(vertices)
        triangles = hull.simplices.copy()
        for j, face in enumerate(triangles):
            p = vertices[face]
            if np.dot(np.cross(p[1] - p[0], p[2] - p[0]), hull.equations[j, :3]) < 0:
                triangles[j] = face[::-1]
        mesh = newton.Mesh(vertices, triangles.ravel(), compute_inertia=False)
        shape = b.add_shape_mesh(-1, xform=tf((*center, 0)), mesh=mesh, color=color, cfg=cfg, label="pong/cup-wall")
        b.shape_flags[shape] &= ~int(newton.ShapeFlags.VISIBLE)
    # Continuous lathed surface avoids facets/seams in the rendered plastic.
    profile = [
        (bottom, 0),
        (bottom, thickness),
        (top, height),
        (top + 0.001, height + 0.0015),
        (top - thickness, height + 0.0015),
        (bottom - thickness, thickness),
        (0, thickness),
        (0, 0),
    ]
    vertices, normals, faces = [], [], []
    for a, c in zip(profile, profile[1:] + profile[:1], strict=True):
        tangent = np.subtract(c, a)
        normal = np.array([tangent[1], -tangent[0]], dtype=float)
        normal /= max(np.linalg.norm(normal), 1e-12)
        start = len(vertices)
        for radius, z in (a, c):
            for j in range(97):
                angle = j * math.tau / 96
                vertices.append((radius * math.cos(angle), radius * math.sin(angle), z))
                normals.append((normal[0] * math.cos(angle), normal[0] * math.sin(angle), normal[1]))
        for j in range(96):
            k = start + j
            faces.extend((k, k + 1, k + 98, k, k + 98, k + 97))
    visible = cfg.copy()
    visible.has_shape_collision = False
    visible.has_particle_collision = False
    mesh = newton.Mesh(
        np.array(vertices, dtype=np.float32),
        np.array(faces, dtype=np.int32),
        normals=np.array(normals, dtype=np.float32),
        compute_inertia=False,
        roughness=0.32,
    )
    b.add_shape_mesh(-1, xform=tf((*center, 0)), mesh=mesh, color=color, cfg=visible, label="pong/cup-shell")


def populate(b, info, variant, assets, device):
    from scene import COLORS, plan_ik, tf  # noqa: PLC0415 -- shared task helpers

    ee, _, fingers = make_arm(b, assets, device, robot="franka", root=(-0.42, -0.20, 0))
    down = wp.quat_rpy(0.0, 0.0, math.pi / 2) * wp.quat(1, 0, 0, 0)
    pickup = np.array([0.12, -0.40, 0.045])
    poses = [
        (0.0, pickup + np.array([0, 0, 0.13]), 0.04),
        (0.9, pickup, 0.04),
        (1.4, pickup, 0.015),
        (2.4, np.array([0.12, -0.50, 0.30]), 0.015),
        (3.2, np.array([0.12, -0.50, 0.30]), 0.015),
    ]
    # Sample one smooth throw continuously. Opening the jaws is the only release;
    # the ball never receives a prescribed pose, velocity, impulse or attachment.
    for t in np.linspace(3.2, 3.8, 31)[1:]:
        u = (t - 3.2) / 0.6
        ease = u * u * u * (10 + u * (-15 + 6 * u))
        y = -0.50 + 0.65 * ease
        grip = 0.015 if t < 3.48 else min(0.04, 0.015 + (t - 3.48) * 0.625)
        poses.append((float(t), np.array([0.12, y, 0.30]), grip))
    poses.extend(
        [
            (4.4, np.array([0.12, 0.15, 0.42]), 0.04),
            (5.5, np.array([-0.10, -0.25, 0.40]), 0.04),
            (7.0, np.array([-0.10, -0.25, 0.40]), 0.04),
        ]
    )
    for i in range(b._hero_driven_dofs - 2):
        b.joint_target_ke[i], b.joint_target_kd[i], b.joint_effort_limit[i] = 8000, 160, 150
    for i in fingers:
        b.joint_target_ke[i], b.joint_target_kd[i], b.joint_effort_limit[i] = 1200, 10, 15
    info["waypoints"] = plan_ik(b, ee, (0, 0, 0.1034), poses, device, desired_rotation=down)
    for i, owner in enumerate(b.shape_body):
        if owner >= 0 and b.body_label[owner].endswith(("/fr3_leftfinger", "/fr3_rightfinger")):
            b.shape_material_mu[i] = 1.2
    # A small tee lets both jaws reach below the ball's equator.
    cfg = b.default_shape_cfg.copy()
    cfg.mu, cfg.restitution, cfg.gap = 0.15, 0.0, 0.0002
    b.add_shape_cylinder(
        -1,
        xform=tf((pickup[0], pickup[1], 0.0125)),
        radius=0.008,
        half_height=0.0125,
        color=(0.35, 0.40, 0.44),
        cfg=cfg,
        label="pong/tee",
    )
    cfg.density = 0
    cfg.mu, cfg.restitution = 0.12, 0.6
    radius, mass = 0.02, 0.0027
    inertia = 2 * mass * radius**2 / 3
    ball = b.add_body(
        xform=tf(pickup),
        mass=mass,
        inertia=wp.mat33(inertia, 0, 0, 0, inertia, 0, 0, 0, inertia),
        lock_inertia=True,
        label="pong/ball",
    )
    b.add_shape_sphere(ball, radius=radius, cfg=cfg, color=(0.98, 0.77, 0.22), label="pong/ball")
    for i, label in enumerate(b.shape_label):
        if label == "plinth":
            b.shape_material_restitution[i] = 1.0
            b.shape_material_mu[i] = 0.12
            b.shape_gap[i] = 0.0002
    center = (0.12, 0.63)
    spacing = 0.125
    for row in range(4):
        for column in range(row + 1):
            x = center[0] + (column - row / 2) * spacing
            y = center[1] + row * spacing * math.sqrt(3) / 2
            cup(b, (x, y), COLORS[0] if row == 0 else COLORS[1])
    info.update(
        robot="franka",
        tracked_body=ball,
        dense_waypoints=True,
        phase_delay=0,
        pong_cup_center=list(center),
        pong_cup_rows=4,
        pong_cup_height=0.12,
        pong_ball_radius=radius,
        task="Grasp and throw a free table-tennis ball, bounce once, and land inside an open cup",
    )


def check(world, poses, fps):
    """Check the recorded free flight, single table bounce and retained landing."""
    p = poses[:, world["body_start"] + world["tracked_body"], :3]
    dz = np.diff(p[:, 2]) * fps
    candidates = np.flatnonzero((dz[:-1] < -0.2) & (dz[1:] > 0.2)) + 1
    bounces = [int(i) for i in candidates if i / fps > 3.0 and p[i, 2] < 0.04]
    cup_center = np.asarray(world["pong_cup_center"])
    radial = np.linalg.norm(p[:, :2] - cup_center, axis=1)
    slope = (0.058 - 0.038) / (0.12 - 0.003)
    clearance = 0.035 + slope * (p[:, 2] - 0.003) - world["pong_ball_radius"] * math.sqrt(1 + slope**2)
    inside = (radial < clearance + 0.0005) & (p[:, 2] > 0.0225) & (p[:, 2] < 0.08)
    retained = bool(np.all(inside[-fps:]))
    bounce_outside = [i for i in bounces if radial[i] > 0.07]
    result = {
        "table_bounce_count": len(bounce_outside),
        "bounce_times_s": [i / fps for i in bounce_outside],
        "retained_in_cup": retained,
        "final_ball_position": p[-1].tolist(),
        "lift_height_m": float(p[: 3 * fps, 2].max() - p[0, 2]),
    }
    gravity = 0.0
    if bounce_outside:
        end = bounce_outside[0] - 1
        start = end - round(0.14 * fps)
        times = np.arange(end - start) / fps
        gravity = float(-2 * np.polyfit(times, p[start:end, 2], 2)[0])
    result["free_flight_gravity_m_s2"] = gravity
    result["pass"] = bool(
        len(bounce_outside) == 1 and retained and result["lift_height_m"] > 0.15 and abs(gravity - 9.81) < 0.1
    )
    return result
