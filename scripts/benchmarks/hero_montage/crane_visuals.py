# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Render-only detailing attached to the recorded crane's existing bodies."""

from itertools import pairwise

import numpy as np
import warp as wp
from furnishings import rounded_box, visual_cfg

import newton


def dress_crane(builder):
    """Add cab, track and hydraulic details without changing any rigid body."""
    import trimesh

    names = ("fixed.plastic", "fixed.rubber", "crane.slew", "boom.lift")
    bodies = {
        name: next(i for i, label in enumerate(builder.body_label) if label.endswith("crane/" + name)) for name in names
    }
    before = (builder.body_count, builder.joint_count, np.array(builder.body_mass).copy())
    first_shape = builder.shape_count
    blue, ochre = (0.035, 0.43, 0.78), (0.96, 0.61, 0.065)
    dark, steel, glass = (0.055, 0.065, 0.073), (0.42, 0.48, 0.51), (0.055, 0.16, 0.20)

    def add(body, mesh, color, p, q=None):
        builder.add_shape_mesh(
            bodies[body],
            mesh=mesh,
            color=color,
            cfg=visual_cfg(builder),
            xform=wp.transform(wp.vec3(*p), wp.quat_identity() if q is None else q),
            label=f"crane_detail/{builder.shape_count - first_shape}",
        )

    def panel(body, p, half, color, metal=False):
        add(body, rounded_box(tuple(half), "metal" if metal else "paint", radius=0.001), color, p)

    def cylinder(body, p, radius, height, color, axis=(0, 0, 1)):
        mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=24)
        # Split cap boundaries while smoothing the cylindrical wall.
        mesh = mesh.copy()
        mesh = trimesh.graph.smooth_shade(mesh, angle=np.deg2rad(40))
        shape = newton.Mesh(
            np.asarray(mesh.vertices, dtype="f4"),
            np.asarray(mesh.faces, dtype="i4").ravel(),
            normals=np.asarray(mesh.vertex_normals, dtype="f4"),
            compute_inertia=False,
            roughness=0.3,
            metallic=0.5,
        )
        add(body, shape, color, p, wp.quat_between_vectors(wp.vec3(0, 0, 1), wp.vec3(*axis)))

    def hose(body, points):
        for first, second in pairwise(points):
            a, b = np.asarray(first), np.asarray(second)
            delta = b - a
            cylinder(body, (a + b) / 2, 0.002, float(np.linalg.norm(delta)), dark, delta / np.linalg.norm(delta))

    # Recessed-looking glazing, door frame and engine grille on the rotating cab.
    panel("crane.slew", (-0.048, 0.0055, 0.146), (0.041, 0.0015, 0.030), dark)
    panel("crane.slew", (-0.048, 0.0038, 0.148), (0.035, 0.0007, 0.023), glass)
    for x in (-0.100, 0.004):
        panel("crane.slew", (x, 0.053, 0.15), (0.001, 0.035, 0.025), dark)
        panel("crane.slew", (x + np.sign(x) * 0.0011, 0.053, 0.152), (0.0006, 0.029, 0.018), glass)
        panel("crane.slew", (x, 0.053, 0.113), (0.001, 0.035, 0.009), blue)
        panel("crane.slew", (x + np.sign(x) * 0.002, 0.077, 0.122), (0.0015, 0.007, 0.0015), steel, True)
    cylinder("crane.slew", (-0.048, 0.052, 0.185), 0.011, 0.005, dark)
    cylinder("crane.slew", (-0.048, 0.052, 0.192), 0.007, 0.009, ochre)
    panel("crane.slew", (0, 0.143, 0.135), (0.07, 0.002, 0.025), blue)
    for z in np.linspace(0.116, 0.155, 7):
        panel("crane.slew", (0, 0.1455, z), (0.056, 0.0008, 0.0015), dark)
    for x in (-0.079, 0.079):
        panel("crane.slew", (x, 0.116, 0.133), (0.0015, 0.022, 0.029), blue)
        for y in (0.10, 0.132):
            for z in (0.112, 0.153):
                cylinder("crane.slew", (x * 1.025, y, z), 0.002, 0.002, steel, (1, 0, 0))
    # Rollers sit inside the existing track silhouette.
    for side in (-1, 1):
        for y in np.linspace(-0.115, 0.115, 7):
            cylinder("fixed.rubber", (side * 0.1335, y, 0.026), 0.014, 0.003, steel, (1, 0, 0))
            cylinder("fixed.rubber", (side * 0.1355, y, 0.026), 0.0085, 0.002, blue, (1, 0, 0))
            cylinder("fixed.rubber", (side * 0.137, y, 0.026), 0.003, 0.001, dark, (1, 0, 0))
        for y in np.linspace(-0.132, 0.132, 18):
            panel("fixed.rubber", (side * 0.11, y, 0.0445), (0.021, 0.002, 0.0008), dark)
        cylinder("boom.lift", (side * 0.054, -0.054, 0.14), 0.012, 0.004, ochre, (1, 0, 0))
        cylinder("boom.lift", (side * 0.057, -0.054, 0.14), 0.005, 0.003, steel, (1, 0, 0))
        hose(
            "boom.lift",
            [
                (side * 0.057, -0.060, 0.152),
                (side * 0.059, -0.075, 0.19),
                (side * 0.057, -0.17, 0.30),
                (side * 0.045, -0.245, 0.40),
            ],
        )
    panel("fixed.plastic", (0, -0.156, 0.045), (0.103, 0.0015, 0.012), dark)
    for x in np.linspace(-0.085, 0.085, 7):
        panel("fixed.plastic", (x, -0.158, 0.045), (0.007, 0.0006, 0.010), ochre)
    for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        cylinder("crane.slew", (0.087 * np.cos(angle), 0.087 * np.sin(angle), 0.0925), 0.0025, 0.002, steel)
    assert (builder.body_count, builder.joint_count) == before[:2]
    np.testing.assert_array_equal(builder.body_mass, before[2])
    assert all(not (flags & newton.ShapeFlags.COLLIDE_SHAPES) for flags in builder.shape_flags[first_shape:])
