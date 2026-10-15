# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Load bread and depress a passive, spring-return toaster carriage."""

import math
from itertools import pairwise

import numpy as np
import warp as wp
from furnishings import skin_box
from robots import make_arm
from task_assets import import_object

import newton


def populate(b, info, variant, assets, device):
    from scene import plan_ik, tf  # noqa: PLC0415 -- shared scene construction
    from scipy.spatial import ConvexHull

    ee, down, fingers = make_arm(b, assets, device, robot="ur10", root=(-0.48, 0.02, 0), instrument=False)
    down = wp.quat_rpy(0.0, 0.0, math.pi / 2) * down * wp.quat_rpy(0.0, 0.0, math.pi)
    origin = np.array([0.24, 0.12, 0.0])
    source = np.array([-0.115, -0.28, 0.109])
    flat = wp.quat_rpy(0.0, -math.pi / 2, 0.0)
    pickup_rotation = flat * down
    loaded = np.add(origin, [0.0476, 0, 0.355])
    paddle = np.add(origin, [0, -0.196, 0.1833])
    poses = [
        (0, np.add(source, [-0.17, 0, 0.04]), 0.04),
        (1, np.add(source, [-0.17, 0, 0.04]), 0.04),
        (2.4, source, 0.04),
        (3.2, source, 0.010),
        (4.5, np.add(source, [0, 0, 0.34]), 0.010),
        (6, np.add(loaded, [0, 0, 0.26]), 0.010),
        (7.5, loaded, 0.010),
        (8.1, loaded, 0.04),
        (9.2, np.add(loaded, [0, 0, 0.18]), 0.04),
        (10.5, np.add(paddle, [0, -0.12, 0.14]), 0.004),
        (11.5, np.add(paddle, [0, 0, 0.07]), 0.004),
        (12.2, paddle, 0.004),
        (14.2, np.add(paddle, [0, 0, -0.1134]), 0.004),
        (15, np.add(paddle, [0, 0, -0.1134]), 0.004),
        (16.3, np.add(paddle, [0, -0.13, 0.13]), 0.004),
        (18, np.add(paddle, [0, -0.13, 0.13]), 0.004),
    ]
    side = wp.quat_rpy(math.pi / 2, 0.0, 0.0) * down
    poses = [
        (p[0] + (1 if p[0] >= 6 else 0), p[1], p[2], pickup_rotation if p[0] <= 4.5 else down if p[0] < 10 else side)
        for p in poses
    ]
    poses.append((5.4, np.add(source, [0.04, 0, 0.35]), 0.010, wp.quat_rpy(0.0, 0.0, 0.45) * pickup_rotation))
    poses.sort(key=lambda pose: pose[0])
    # Solve at a conservative planning rate, then time-parameterize the motor
    # path. Near the wrist singularity, rotate more slowly instead of jumping.
    planned = plan_ik(
        b,
        ee,
        (0, 0, 0.1034),
        [(4 * t, p, g, r) for t, p, g, r in poses],
        device,
        desired_rotation=down,
        cartesian_step=0.1,
    )
    timed = [(planned[0][0] / 4, planned[0][1])]
    for (ta, qa), (tb, qb) in pairwise(planned):
        delta = float(np.max(np.abs(qb[:6] - qa[:6])))
        if delta > 0.25:
            raise ValueError(f"Toaster IK branch jump: {delta:.3f} rad")
        timed.append((timed[-1][0] + max((tb - ta) / 4, delta / 2.5), qb))
    info["waypoints"] = timed
    info["motion_duration"] = timed[-1][0]
    for i in range(6):
        b.joint_target_ke[i], b.joint_target_kd[i], b.joint_effort_limit[i] = 12000, 180, 250
    for i in fingers:
        b.joint_target_ke[i], b.joint_target_kd[i], b.joint_effort_limit[i] = 1800, 25, 18
    for i, owner in enumerate(b.shape_body):
        if owner >= 0 and "finger" in b.body_label[owner]:
            b.shape_material_mu[i] = 1.2
    start = b.joint_count
    mapping, _ = import_object(b, "toaster", tf(origin), scale=1.4, moving={"bread_lift_0"})
    lift = mapping["bread_lift_0"]
    for j in range(start, b.joint_count):
        if b.joint_label[j] == "toaster/bread_lift_0.q":
            v, q = b.joint_qd_start[j], b.joint_q_start[j]
            b.joint_q[q] = 0.0224
            b.joint_spring_stiffness[v] = 35
            b.joint_spring_ref[v] = 0.20
            b.joint_damping[v] = 2
    cfg = b.default_shape_cfg.copy()
    cfg.density, cfg.mu, cfg.gap = 0, 0.3, 0.00015
    # A real rear ratchet: descending carriage deflects a spring-loaded wedge;
    # its flat underside catches the tab after the lever reaches the lower stop.
    b.add_shape_box(
        lift,
        xform=tf((0, 0.188, 0.1512)),
        hx=0.018,
        hy=0.025,
        hz=0.003,
        cfg=cfg,
        color=(0.3, 0.34, 0.36),
        label="toaster/latch-tab",
    )
    latch = b.add_link(mass=0.03, inertia=wp.mat33(np.eye(3, dtype=np.float32) * 0.00001), label="toaster/latch")
    j = b.add_joint_prismatic(
        -1,
        latch,
        parent_xform=tf(np.add(origin, [0, 0.215, 0.075])),
        axis=(0, 1, 0),
        limit_lower=0,
        limit_upper=0.035,
        spring_stiffness=180,
        spring_ref=-0.006,
        damping=1,
        label="toaster/latch-spring",
    )
    b.add_articulation([j])
    vertices = np.array(
        [[x, y, z] for x in [-0.02, 0.02] for y, z in [(-0.018, -0.009), (0.012, -0.009), (0.012, 0.023)]], np.float32
    )

    hull = ConvexHull(vertices)
    faces = hull.simplices.copy()
    for i, f in enumerate(faces):
        a = vertices[f]
        if np.dot(np.cross(a[1] - a[0], a[2] - a[0]), hull.equations[i, :3]) < 0:
            faces[i] = f[::-1]
    mesh = newton.Mesh(vertices, faces.ravel(), compute_inertia=False)
    b.add_shape_mesh(latch, mesh=mesh, cfg=cfg, color=(0.24, 0.28, 0.3), label="toaster/ratchet")
    b.shape_type[-1] = newton.GeoType.CONVEX_MESH
    # Tapered entry rails center the slices by contact as the carriage descends.
    for x in [-0.0476, 0.0476]:
        for side in [-1, 1]:
            for y in [-0.065, 0.065]:
                a = np.array([x + side * 0.022, y, 0.30])
                c = np.array([x + side * 0.016, y, 0.11])
                d = c - a
                b.add_shape_capsule(
                    -1,
                    xform=tf(
                        origin + (a + c) / 2,
                        wp.quat_between_vectors(wp.vec3(0, 0, 1), wp.vec3(*(d / np.linalg.norm(d)))),
                    ),
                    radius=0.001,
                    half_height=float(np.linalg.norm(d) / 2),
                    cfg=cfg,
                    color=(0.55, 0.59, 0.61),
                    label="toaster/centering-wire",
                )

    def bread(position, name, rotation=None):
        from PIL import Image, ImageDraw

        body = b.add_body(
            xform=tf(position, rotation),
            mass=0.035,
            inertia=wp.mat33(0.00012, 0, 0, 0, 0.00012, 0, 0, 0, 0.00005),
            lock_inertia=True,
            label=name,
        )
        outline = [(-0.070, -0.092), (0.070, -0.092), (0.078, -0.082), (0.078, 0.056)]
        outline += [(0.078 * math.cos(a), 0.056 + 0.044 * math.sin(a)) for a in np.linspace(0, math.pi, 17)[1:]]
        outline += [(-0.078, -0.082)]
        vertices = np.array([[x, y, z] for x in [-0.012, 0.012] for y, z in outline], np.float32)
        hull = ConvexHull(vertices)
        faces = hull.simplices.copy()
        for i, f in enumerate(faces):
            a = vertices[f]
            if np.dot(np.cross(a[1] - a[0], a[2] - a[0]), hull.equations[i, :3]) < 0:
                faces[i] = f[::-1]
        mesh = newton.Mesh(vertices, faces.ravel(), compute_inertia=False, roughness=0.8)
        bread_cfg = cfg.copy()
        bread_cfg.mu = 0.6
        b.add_shape_mesh(body, mesh=mesh, cfg=bread_cfg, color=(0.72, 0.43, 0.20), label=name + "/crust")
        b.shape_type[-1] = newton.GeoType.CONVEX_MESH
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 2, (512, 512, 1))
        texture = Image.fromarray(np.uint8(np.clip(np.array([238, 212, 166]) + noise, 0, 255)))
        draw = ImageDraw.Draw(texture)
        for _ in range(1600):
            x, y = rng.integers(0, 512, 2)
            r = int(rng.choice([1, 1, 2, 3, 4]))
            draw.ellipse((x - r, y - r, x + r, y + r), fill=(214, 183, 133))
            if r > 2:
                draw.ellipse((x - r + 1, y - r + 1, x + r - 1, y + r - 1), fill=(225, 198, 151))
        visual = cfg.copy()
        visual.has_shape_collision = visual.has_particle_collision = False
        for side in [-1, 1]:
            points = np.array(
                [[side * 0.0121, 0, 0]] + [[side * 0.0121, y * 0.91, z * 0.92] for y, z in outline], np.float32
            )
            faces = np.array([[0, i + 1, (i + 1) % len(outline) + 1] for i in range(len(outline))], np.int32)
            if side < 0:
                faces = faces[:, ::-1]
            uv = np.column_stack(((points[:, 1] + 0.08) / 0.16, (points[:, 2] + 0.10) / 0.20))
            mesh = newton.Mesh(
                points,
                faces.ravel(),
                normals=np.tile([side, 0, 0], (len(points), 1)).astype(np.float32),
                uvs=uv.astype(np.float32),
                texture=np.asarray(texture),
                compute_inertia=False,
                roughness=0.85,
            )
            b.add_shape_mesh(body, mesh=mesh, cfg=visual, color=(1, 1, 1), label=name + "/crumb")
        return body

    first = bread(np.add(origin, [-0.0476, 0, 0.2736]), "toaster/preloaded-bread")
    # The stagger exposes the pickup slice's edge for a real side pinch.
    board_cfg = cfg.copy()
    board_cfg.mu = 0.7
    board = b.add_shape_box(
        -1,
        xform=tf((0.04, -0.28, 0.011)),
        hx=0.205,
        hy=0.14,
        hz=0.011,
        cfg=board_cfg,
        color=(1, 1, 1),
        label="toaster/cutting-board",
    )
    skin_box(b, board, material="ash", radius=0.006)
    spare_breads = [
        bread([0.075 - i * 0.035, -0.28, 0.034 + i * 0.025], f"toaster/spare-bread-{i}", flat) for i in range(3)
    ]
    second = bread([-0.04, -0.28, 0.109], "toaster/inserted-bread", flat)
    info.update(
        robot="ur10",
        tracked_body=lift,
        breads=[first, second],
        spare_breads=spare_breads,
        toaster_origin=origin.tolist(),
        dense_waypoints=True,
        phase_delay=0,
        task="Insert a second bread slice and press the mechanical toaster lever",
    )


def check(world, poses, fps):
    from scipy.spatial.transform import Rotation

    origin = np.array(world["toaster_origin"])
    bread = poses[:, np.array(world["breads"]) + world["body_start"], :3]
    lift = poses[:, world["tracked_body"] + world["body_start"], :3]
    descent = float(lift[0, 2] - lift[-1, 2])
    placed = bool(np.all(np.abs(bread[-1, :, :2] - np.add(origin[:2], [[-0.0476, 0], [0.0476, 0]])) < 0.018))
    spare = poses[-1, np.array(world["spare_breads"]) + world["body_start"], :3]
    pickup = poses[0, world["breads"][1] + world["body_start"]]
    flat_start = abs(float(Rotation.from_quat(pickup[3:]).as_matrix()[2, 0]))
    spares_on_board = bool(
        np.all(
            (spare[:, 0] > -0.14)
            & (spare[:, 0] < 0.22)
            & (spare[:, 1] > -0.40)
            & (spare[:, 1] < -0.16)
            & (spare[:, 2] > 0.022)
            & (spare[:, 2] < 0.13)
        )
    )
    result = {
        "pickup_starts_flat_cosine": flat_start,
        "spare_slices_remain_on_board": spares_on_board,
        "carriage_descent_m": descent,
        "both_slices_in_slots": placed,
        "second_slice_lift_m": float(bread[:, 1, 2].max() - bread[0, 1, 2]),
        "final_bread_heights": bread[-1, :, 2].tolist(),
        "finite": bool(np.isfinite(poses).all()),
        "carriage_motion_after_release_m": float(np.ptp(lift[-fps:, 2])),
        "bread_support_error_m": float(np.max(np.abs(bread[-1, :, 2] - (lift[-1, 2] + 0.1512 + 0.0014 + 0.092)))),
        "bread_motion_after_release_m": float(np.linalg.norm(np.ptp(bread[-fps:], axis=0), axis=1).max()),
        "carriage_held_up_before_press": bool(lift[: 10 * fps, 2].min() > 0.015),
    }
    result["pass"] = bool(
        result["finite"]
        and placed
        and flat_start > 0.98
        and spares_on_board
        and descent > 0.09
        and np.all(bread[-1, :, 2] < 0.20)
        and result["second_slice_lift_m"] > 0.25
        and result["carriage_motion_after_release_m"] < 0.001
        and result["bread_support_error_m"] < 0.008
        and result["bread_motion_after_release_m"] < 0.003
        and result["carriage_held_up_before_press"]
    )
    return result
