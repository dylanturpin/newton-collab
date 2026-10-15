# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""A rigid miniature G1 toy falling into independent flame-shaped packing pieces."""

import math
import os
from pathlib import Path

import numpy as np
import warp as wp

import newton


def populate(b, info, variant, assets, device):
    from scene import box, prism_mesh, tf  # noqa: PLC0415 -- shared scene helpers
    from scipy.spatial.transform import Rotation

    # Freeze only the toy's assembly pose, not its world motion. All imported
    # collision shapes remain attached to one freely simulated rigid figure.
    toy = newton.ModelBuilder()
    toy.add_usd(
        str(Path(assets["unitree_g1"]) / "usd/g1_isaac.usd"),
        collapse_fixed_joints=False,
        enable_self_collisions=False,
        joint_ordering="dfs",
        hide_collision_shapes=True,
    )
    original = len(toy.shape_flags)
    toy.approximate_meshes("convex_hull", keep_visual_shapes=True)
    for i in range(original, len(toy.shape_flags)):
        toy.shape_flags[i] &= ~int(newton.ShapeFlags.VISIBLE)
    toy.joint_q[:] = [0.0] * len(toy.joint_q)
    toy.joint_q[6] = 1.0
    for j, label in enumerate(toy.joint_label):
        if "shoulder_roll" in label:
            toy.joint_q[toy.joint_q_start[j]] = 1.45 if "left" in label else -1.45
        elif "elbow_joint" in label:
            toy.joint_q[toy.joint_q_start[j]] = 1.45
    model = toy.finalize(device="cpu")
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    frames = state.body_q.numpy()
    scale = 0.30
    masses = model.body_mass.numpy()
    rotations = Rotation.from_quat(frames[:, 3:]).as_matrix()
    centers = frames[:, :3] + np.einsum("nij,nj->ni", rotations, model.body_com.numpy())
    center = np.average(centers, axis=0, weights=masses)
    inertia = np.zeros((3, 3))
    for m, r, c, local_inertia in zip(masses, rotations, centers, model.body_inertia.numpy(), strict=True):
        delta = c - center
        inertia += r @ local_inertia @ r.T + m * (np.dot(delta, delta) * np.eye(3) - np.outer(delta, delta))
    # Uniform scaling preserves the imported G1's material density.
    mass = float(masses.sum() * scale**3)
    inertia *= scale**5
    # G1 imported origin is the pelvis, feet are about 0.76 m below it.
    tilt = -0.15
    rotation = wp.quat_rpy(0.0, -tilt, 0.0) * wp.quat_rpy(0.0, 0.0, math.pi)
    body = b.add_body(
        xform=tf((3.0, 0, 2.0) if os.environ.get("LAVA_PREPARE_PILE") else (-0.23, 0, 0.865), rotation),
        mass=mass,
        com=center * scale,
        inertia=wp.mat33(*inertia.ravel()),
        lock_inertia=True,
        label="mini_g1/free_figure",
    )
    cfg = b.default_shape_cfg.copy()
    cfg.density, cfg.mu, cfg.restitution, cfg.gap = 0, 0.12, 0, 0.0002
    for i, owner in enumerate(toy.shape_body):
        if owner < 0:
            continue
        transform = wp.transform_multiply(
            wp.transform(wp.vec3(*frames[owner, :3]), wp.quat(*frames[owner, 3:])), toy.shape_transform[i]
        )
        pos = np.array(wp.transform_get_translation(transform)) * scale
        cfg.is_visible = bool(toy.shape_flags[i] & int(newton.ShapeFlags.VISIBLE))
        cfg.has_shape_collision = bool(toy.shape_flags[i] & int(newton.ShapeFlags.COLLIDE_SHAPES))
        label = toy.body_label[owner].rsplit("/", 1)[-1].lower()
        dark = any(x in label for x in ("pelvis", "waist", "hand_", "ankle", "shoulder_pitch", "hip_pitch"))
        b.add_shape(
            body=body,
            type=toy.shape_type[i],
            xform=tf(pos, wp.transform_get_rotation(transform)),
            cfg=cfg,
            scale=np.array(toy.shape_scale[i]) * scale,
            src=toy.shape_source[i],
            color=(0.09, 0.105, 0.12) if dark else (0.73, 0.76, 0.79),
            label="mini_g1/" + toy.shape_label[i],
        )
        if cfg.is_visible and isinstance(toy.shape_source[i], newton.Mesh):
            # Walking USD omits limb collisions. This toy contacts the pile with
            # every link, so use a separate hull for each detailed visible part.
            limb_cfg = cfg.copy()
            limb_cfg.is_visible, limb_cfg.has_shape_collision = False, True
            b.add_shape_convex_hull(
                body,
                mesh=toy.shape_source[i].compute_convex_hull(),
                xform=tf(pos, wp.transform_get_rotation(transform)),
                scale=np.array(toy.shape_scale[i]) * scale,
                cfg=limb_cfg,
                label="mini_g1/limb_collision/" + toy.shape_label[i],
            )
    # Deep open blue pot, composed of a floor and closely fitted wall panels.
    radius, height = 0.43, 0.57
    potcfg = b.default_shape_cfg.copy()
    potcfg.mu, potcfg.restitution = 0.04, 0
    b.add_shape_cylinder(
        -1,
        xform=tf((0.10, 0, 0.025)),
        radius=radius + 0.025,
        half_height=0.025,
        cfg=potcfg,
        color=(0.035, 0.23, 0.42),
        label="pot/base",
    )
    potcfg.is_visible = False
    for i in range(64):
        a = math.tau * i / 64
        p = (0.10 + radius * math.cos(a), radius * math.sin(a), height / 2)
        b.add_shape_box(
            -1,
            xform=tf(p, wp.quat_rpy(0.0, 0.0, a)),
            hx=0.018,
            hy=0.023,
            hz=height / 2,
            cfg=potcfg,
            color=(0.035, 0.23, 0.42),
            label=f"pot/wall{i}",
        )
        b.add_shape_box(
            -1,
            xform=tf((p[0], p[1], height), wp.quat_rpy(0.0, 0.0, a)),
            hx=0.025,
            hy=0.024,
            hz=0.014,
            cfg=potcfg,
            color=(0.72, 0.78, 0.80),
            label=f"pot/rim{i}",
        )
    # Smooth render shells share the collision wall's exact inner/outer radii.
    shellcfg = potcfg.copy()
    shellcfg.is_visible, shellcfg.has_shape_collision = True, False
    for name, rings, color in [
        (
            "shell",
            [(radius + 0.018, 0.05), (radius + 0.018, height), (radius - 0.018, height), (radius - 0.018, 0.05)],
            (0.035, 0.23, 0.42),
        ),
        (
            "rim",
            [
                (radius + 0.025, height - 0.014),
                (radius + 0.025, height + 0.014),
                (radius - 0.025, height + 0.014),
                (radius - 0.025, height - 0.014),
            ],
            (0.72, 0.78, 0.80),
        ),
    ]:
        vertices, normals, indices = [], [], []
        for edge in range(len(rings) - 1):
            r0, z0 = rings[edge]
            r1, z1 = rings[edge + 1]
            nr, nz = z1 - z0, r0 - r1
            length = math.hypot(nr, nz)
            nr, nz = nr / length, nz / length
            start = len(vertices)
            for angle in np.linspace(0, math.tau, 129):
                for r, z in [(r0, z0), (r1, z1)]:
                    vertices.append((0.10 + r * math.cos(angle), r * math.sin(angle), z))
                    normals.append((nr * math.cos(angle), nr * math.sin(angle), nz))
            for j in range(128):
                a = start + 2 * j
                indices.extend([a, a + 2, a + 1, a + 1, a + 2, a + 3])
        mesh = newton.Mesh(
            vertices=np.array(vertices, dtype=np.float32),
            indices=np.array(indices, dtype=np.int32),
            normals=np.array(normals, dtype=np.float32),
            roughness=0.36,
        )
        b.add_shape_mesh(-1, mesh=mesh, cfg=shellcfg, color=color, label="pot/visual_" + name)
    # Narrow support: COM is beyond the heel edge so gravity initiates tipping.
    box(b, (-0.565, 0, 0.30), (0.11, 0.17, 0.30), (0.19, 0.23, 0.27), label="stand/base")
    box(b, (-0.465, 0, 0.61), (0.21, 0.17, 0.015), (0.78, 0.65, 0.42), label="stand/top")
    # Flat flame silhouettes, with a smaller yellow lick on either face.
    profile = np.array(
        [
            (-0.034, -0.038),
            (0.012, -0.043),
            (0.036, -0.021),
            (0.033, 0.010),
            (0.017, -0.001),
            (0.012, 0.053),
            (-0.009, 0.029),
            (-0.023, 0.004),
            (-0.026, 0.021),
            (-0.039, -0.009),
        ]
    )
    # Convex collision envelope stays inside the visible silhouette's bounds.
    from scipy.spatial import ConvexHull

    hull = profile[ConvexHull(profile).vertices]
    profile *= 1.5
    hull *= 1.5
    flame = prism_mesh(hull, 0.033)
    visual = prism_mesh(profile, 0.033)
    inner = prism_mesh(profile * 0.52, 0.0008)
    pieces = []
    seed_path = Path(__file__).parent / "assets" / "lava_settled_pile.npz"
    settled = np.load(seed_path)["poses"] if seed_path.exists() and not os.environ.get("LAVA_PREPARE_PILE") else None
    centers = [(0.0, 0.0)]
    for count, radius in [(6, 0.16), (12, 0.32)]:
        centers.extend((radius * math.cos(a), radius * math.sin(a)) for a in np.arange(count) * math.tau / count)
    for i in range(200):
        layer, k = divmod(i, len(centers))
        x, y = centers[k]
        p = (0.10 + x, y, 0.09 + layer * 0.075)
        q = wp.quat_rpy(0.0, 0.0, float(math.atan2(y, x)))
        if settled is not None:
            p, q = settled[i, :3], wp.quat(*settled[i, 3:])
        part = b.add_body(
            xform=tf(p, q),
            mass=0.006,
            inertia=wp.mat33(6e-6, 0, 0, 0, 6e-6, 0, 0, 0, 6e-6),
            lock_inertia=True,
            label=f"lava/flame{i:03}",
        )
        c = cfg.copy()
        c.is_visible = False
        c.has_shape_collision = True
        c.mu = 0.025
        b.add_shape_mesh(part, mesh=flame, cfg=c, label=f"lava/collision{i}")
        c.is_visible = True
        c.has_shape_collision = False
        b.add_shape_mesh(part, mesh=visual, cfg=c, color=(1, 0.19 + 0.06 * (i % 3), 0.015), label=f"lava/outer{i}")
        for side in [-1, 1]:
            b.add_shape_mesh(
                part,
                mesh=inner,
                xform=tf((0, -0.010, side * 0.034)),
                cfg=c,
                color=(1, 0.72, 0.035),
                label=f"lava/yellow{i}/{side}",
            )
        pieces.append(part)
    info.update(
        tracked_body=body,
        lava_bodies=pieces,
        pot_radius=radius,
        pot_height=height,
        rigid_toy=True,
        miniature_scale=scale,
        figure_mass_kg=mass,
    )
