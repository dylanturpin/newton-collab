# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Small contact-driven mechanisms with explicitly matched geometry."""

import math

import numpy as np

import newton


def gear_trainer(b, base):
    """Build three freely hinged involute gears; coupling comes from contact."""
    from scene import COLORS, box, prism_mesh, tf  # noqa: PLC0415

    module = 0.006
    counts = [12, 16, 10]
    pitch = np.asarray(counts) * module / 2
    centers = np.array([[-0.084, 0.0, 0.062], [0.0, 0.0, 0.062], [0.078, 0.0, 0.062]]) + base
    box(b, (base[0], base[1], 0.012), (0.16, 0.12, 0.012), (0.24, 0.29, 0.33))
    root = b.add_link(label="gear_trainer_frame")
    joints = [b.add_joint_fixed(-1, root)]
    bodies = []
    cfg = b.default_shape_cfg.copy()
    cfg.gap = 0.00005
    cfg.mu = 0.18
    cfg.density = 1100
    for index, (n, radius, center) in enumerate(zip(counts, pitch, centers, strict=True)):
        body = b.add_link(xform=tf(center), label=f"involute_gear_{index}")
        bodies.append(body)
        joints.append(
            b.add_joint_revolute(
                root, body, parent_xform=tf(center), axis=(0, 0, 1), friction=0.0001, damping=0.001, armature=1e-6
            )
        )
        base_radius = radius * math.cos(math.radians(20))
        root_radius = radius - 1.25 * module
        tip_radius = radius + module
        pressure = math.radians(20)
        pitch_inv = math.tan(pressure) - pressure
        # 0.25 mm backlash at the pitch circle prevents initial tooth overlap.
        half_angle = math.pi / (2 * n) - 0.000125 / radius
        radial = np.linspace(max(root_radius, base_radius), tip_radius, 5)
        radial = np.unique(np.r_[root_radius, radial])
        widths = []
        for r in radial:
            a = math.acos(min(1.0, base_radius / r))
            widths.append(half_angle + pitch_inv - (math.tan(a) - a))
        phase = math.pi / n if index == 1 else 0.0
        b.add_shape_cylinder(body, radius=root_radius, half_height=0.009, color=COLORS[index], cfg=cfg)
        for tooth in range(n):
            angle = phase + tooth * 2 * math.pi / n
            for j in range(len(radial) - 1):
                points = [
                    (radial[k] * math.cos(angle + sign * widths[k]), radial[k] * math.sin(angle + sign * widths[k]))
                    for k, sign in ((j, -1), (j + 1, -1), (j + 1, 1), (j, 1))
                ]
                b.add_shape_mesh(body, mesh=prism_mesh(points, 0.009), color=COLORS[index], cfg=cfg)
                b.shape_type[-1] = newton.GeoType.CONVEX_MESH
        # Contrasting rotating hub and fixed bearing make the rotation legible.
        b.add_shape_cylinder(
            body,
            xform=tf((0, 0, 0.010)),
            radius=radius * 0.53,
            half_height=0.002,
            color=COLORS[(index + 1) % 3],
            cfg=cfg,
        )
        b.add_shape_cylinder(
            root, xform=tf(center - [0, 0, 0.021]), radius=0.007, half_height=0.019, color=(0.5, 0.55, 0.59), cfg=cfg
        )
        if index == 0:
            b.add_shape_cylinder(
                body, xform=tf((0.024, 0, 0.029)), radius=0.0075, half_height=0.025, color=(0.58, 0.62, 0.65), cfg=cfg
            )
    b.add_articulation(joints)
    return bodies, centers[0] + [0, 0, 0.046], counts
