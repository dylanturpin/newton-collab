# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Grasp an open carton and pour independent alphabet cereal into a bowl."""

import math
from functools import cache
from itertools import pairwise
from pathlib import Path

import numpy as np
import warp as wp
from robots import make_arm
from task_assets import import_object, load

import newton

LETTERS = {
    "S": [
        [
            (0.55, 0.8),
            (0.25, 1),
            (-0.4, 0.9),
            (-0.6, 0.5),
            (-0.35, 0.15),
            (0.35, -0.15),
            (0.6, -0.5),
            (0.4, -0.9),
            (-0.25, -1),
            (-0.55, -0.8),
        ]
    ],
    "K": [[(-0.55, -1), (-0.55, 1)], [(0.55, 1), (-0.55, 0), (0.55, -1)]],
    "I": [[(-0.5, 1), (0.5, 1)], [(0, 1), (0, -1)], [(-0.5, -1), (0.5, -1)]],
    "L": [[(-0.5, 1), (-0.5, -1), (0.6, -1)]],
    "D": [[(-0.55, -1), (-0.55, 1), (0.15, 1), (0.55, 0.65), (0.65, 0), (0.55, -0.65), (0.15, -1), (-0.55, -1)]],
    "E": [[(0.55, 1), (-0.55, 1), (-0.55, -1), (0.55, -1)], [(-0.55, 0), (0.4, 0)]],
    "T": [[(-0.65, 1), (0.65, 1)], [(0, 1), (0, -1)]],
}


@cache
def carton_texture():
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (768, 1152), (239, 231, 206))
    draw = ImageDraw.Draw(image)
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]

    font_path = next((p for p in candidates if Path(p).exists()), None)

    def font(size):
        return ImageFont.truetype(font_path, size) if font_path else ImageFont.load_default(size=size)

    draw.rectangle((0, 0, 768, 240), fill=(9, 111, 183))
    draw.text((48, 50), "SKILDET", font=font(134), fill=(250, 246, 230))
    draw.text((52, 258), "ALPHABET", font=font(74), fill=(14, 92, 75))
    draw.text((52, 342), "CRUNCH", font=font(89), fill=(14, 92, 75))
    draw.ellipse((72, 480, 696, 1000), fill=(17, 120, 175))
    draw.ellipse((94, 490, 674, 925), fill=(249, 245, 225))
    for i, ch in enumerate("SKILDET"):
        x, y = [(145, 560), (310, 520), (490, 575), (180, 720), (360, 690), (510, 735), (320, 820)][i]
        draw.text((x + 3, y + 5), ch, font=font(100), fill=(166, 109, 41))
        draw.text((x, y), ch, font=font(100), fill=(221, 169, 79))
    draw.text((65, 1032), "A BRIGHT START", font=font(52), fill=(14, 92, 75))
    return np.asarray(image.transpose(Image.Transpose.FLIP_TOP_BOTTOM))


def bowl(b, center):
    from scene import tf  # noqa: PLC0415 -- shared scene helpers
    from scipy.spatial import ConvexHull

    cfg = b.default_shape_cfg.copy()
    cfg.mu, cfg.gap, cfg.restitution = 0.35, 0.00015, 0.02
    b.add_shape_cylinder(
        -1,
        xform=tf((*center, 0.006)),
        radius=0.073,
        half_height=0.006,
        cfg=cfg,
        color=(0.035, 0.43, 0.78),
        label="cereal/bowl-base",
    )
    for i in range(40):
        vertices = np.array(
            [
                (r * math.cos(a), r * math.sin(a), z)
                for z, radius in ((0.009, 0.072), (0.085, 0.135))
                for r in (radius - 0.004, radius)
                for a in (i * math.tau / 40, (i + 1) * math.tau / 40)
            ],
            np.float32,
        )
        hull = ConvexHull(vertices)
        faces = hull.simplices.copy()
        for j, face in enumerate(faces):
            p = vertices[face]
            if np.dot(np.cross(p[1] - p[0], p[2] - p[0]), hull.equations[j, :3]) < 0:
                faces[j] = face[::-1]
        shape = b.add_shape_mesh(
            -1,
            xform=tf((*center, 0)),
            mesh=newton.Mesh(vertices, faces.ravel(), compute_inertia=False),
            cfg=cfg,
            label="cereal/bowl-wall",
        )
        b.shape_flags[shape] &= ~int(newton.ShapeFlags.VISIBLE)
    profile = [
        (0, 0.002),
        (0.072, 0.002),
        (0.078, 0.014),
        (0.137, 0.083),
        (0.137, 0.087),
        (0.133, 0.089),
        (0.129, 0.084),
        (0.068, 0.013),
        (0, 0.013),
    ]
    vertices, normals, indices = [], [], []
    for a, c in pairwise(profile):
        tangent = np.subtract(c, a)
        n = np.array([tangent[1], -tangent[0]]) / np.linalg.norm(tangent)
        base = len(vertices)
        for r, z in (a, c):
            for j in range(97):
                angle = j * math.tau / 96
                vertices.append((r * math.cos(angle), r * math.sin(angle), z))
                normals.append((n[0] * math.cos(angle), n[0] * math.sin(angle), n[1]))
        for j in range(96):
            k = base + j
            indices.extend((k, k + 1, k + 98, k, k + 98, k + 97))
    visual = cfg.copy()
    visual.density = 0
    visual.has_shape_collision = visual.has_particle_collision = False
    mesh = newton.Mesh(
        np.array(vertices, np.float32),
        np.array(indices, np.int32),
        normals=np.array(normals, np.float32),
        compute_inertia=False,
        roughness=0.22,
    )
    b.add_shape_mesh(
        -1, xform=tf((*center, 0)), mesh=mesh, cfg=visual, color=(0.035, 0.43, 0.78), label="cereal/ceramic-bowl"
    )


def populate(b, info, variant, assets, device):
    from scene import plan_ik, tf  # noqa: PLC0415 -- shared scene helpers

    ee, down, fingers = make_arm(b, assets, device, robot="ur10", root=(-0.48, 0.02, 0), instrument=False)
    down = wp.quat_rpy(0.0, -math.pi / 2, 0.0) * down
    source = np.array([0.20, 0.23, 0.12])
    pouring = np.array([0.14, -0.20, 0.34])
    tilt = wp.quat_rpy(0.0, math.radians(135), 0.0) * down
    poses = [
        (0, np.add(source, [-0.20, 0, 0.12]), 0.04, down),
        (1, np.add(source, [-0.20, 0, 0.12]), 0.04, down),
        (2.5, source, 0.04, down),
        (3.3, source, 0.035, down),
        (4.7, np.add(source, [0, 0, 0.22]), 0.035, down),
        (6, pouring, 0.035, down),
        (8.5, pouring, 0.035, tilt),
        (10.8, pouring, 0.035, tilt),
        (12.5, pouring, 0.035, down),
        (14, np.add(source, [0, 0, 0.22]), 0.035, down),
        (15.5, source, 0.035, down),
        (16.2, source, 0.04, down),
        (17.5, np.add(source, [-0.20, 0, 0.12]), 0.04, down),
        (18, np.add(source, [-0.20, 0, 0.12]), 0.04, down),
    ]
    info["waypoints"] = plan_ik(b, ee, (0, 0, 0.1034), poses, device, desired_rotation=down, cartesian_step=0.12)
    for i in range(6):
        b.joint_target_ke[i], b.joint_target_kd[i], b.joint_effort_limit[i] = 12000, 180, 250
    for i in fingers:
        b.joint_target_ke[i], b.joint_target_kd[i], b.joint_effort_limit[i] = 2500, 30, 35
    for i, owner in enumerate(b.shape_body):
        if owner >= 0 and b.body_label[owner].endswith(("/fr3_leftfinger", "/fr3_rightfinger")):
            b.shape_material_mu[i] = 1.1
    carton = b.add_body(
        xform=tf((source[0], source[1], 0.101)),
        mass=0.075,
        inertia=wp.mat33(0.00028, 0, 0, 0, 0.00036, 0, 0, 0, 0.00011),
        lock_inertia=True,
        label="cereal/carton",
    )
    cfg = b.default_shape_cfg.copy()
    cfg.density, cfg.mu, cfg.gap, cfg.restitution = 0, 0.45, 0.00015, 0.01
    walls = [
        ((0, 0, -0.099), (0.065, 0.039, 0.0015)),
        ((-0.0635, 0, 0), (0.0015, 0.039, 0.1)),
        ((0.0635, 0, 0), (0.0015, 0.039, 0.1)),
        ((0, -0.0375, 0), (0.062, 0.0015, 0.1)),
        ((0, 0.0375, 0), (0.062, 0.0015, 0.1)),
    ]
    for p, size in walls:
        b.add_shape_box(
            carton,
            xform=tf(p),
            hx=size[0],
            hy=size[1],
            hz=size[2],
            cfg=cfg,
            color=(0.72, 0.62, 0.44),
            label="cereal/cardboard",
        )
    visual = cfg.copy()
    visual.has_shape_collision = visual.has_particle_collision = False
    # Printed panels remain attached to the physical carton walls.
    for side in (-1, 1):
        vertices = np.array(
            [
                [-0.062, side * 0.0391, -0.098],
                [0.062, side * 0.0391, -0.098],
                [0.062, side * 0.0391, 0.098],
                [-0.062, side * 0.0391, 0.098],
            ],
            np.float32,
        )
        mesh = newton.Mesh(
            vertices,
            np.array([0, 1, 2, 0, 2, 3] if side < 0 else [0, 2, 1, 0, 3, 2], np.int32),
            uvs=np.array(
                [[0, 0], [1, 0], [1, 1], [0, 1]] if side < 0 else [[1, 0], [0, 0], [0, 1], [1, 1]], np.float32
            ),
            normals=np.tile([0, side, 0], (4, 1)).astype(np.float32),
            texture=carton_texture(),
            compute_inertia=False,
            roughness=0.7,
        )
        b.add_shape_mesh(carton, mesh=mesh, cfg=visual, color=(1, 1, 1), label="cereal/printed-carton")
    bowl_center = [0.27, -0.20]
    bowl(b, bowl_center)
    mapping, _ = import_object(
        b,
        "cutlery_spoon",
        tf((0.36, -0.20, 0.065), wp.quat_rpy(0.0, -0.35, 0.0)),
        free_root=True,
        finish_color=(0.02, 0.65, 0.47),
    )
    spoon = mapping[load("cutlery_spoon")[0]["root"]]
    rng = np.random.default_rng(203)
    pieces = []
    cfg = b.default_shape_cfg.copy()
    cfg.density, cfg.mu, cfg.gap, cfg.restitution = 160, 0.22, 0.0001, 0.04
    for i in range(168):
        letter = "SKILDET"[i % 7]
        p = [source[0] + ((i % 4) - 1.5) * 0.029, source[1] + ((i // 4) % 2 - 0.5) * 0.036, 0.009 + (i // 8) * 0.0075]
        body = b.add_body(
            xform=tf(p, wp.quat_rpy(0.0, 0.0, float(rng.uniform(-0.06, 0.06)))), label=f"cereal/letter-{letter}-{i:02d}"
        )
        color = (float(rng.uniform(0.77, 0.88)), float(rng.uniform(0.55, 0.66)), float(rng.uniform(0.26, 0.34)))
        for stroke in LETTERS[letter]:
            for first, second in pairwise(stroke):
                a = np.array([first[0] * 0.0144, first[1] * 0.0144, 0.0])
                c = np.array([second[0] * 0.0144, second[1] * 0.0144, 0.0])
                d = c - a
                b.add_shape_capsule(
                    body,
                    xform=tf((a + c) / 2, wp.quat_between_vectors(wp.vec3(0, 0, 1), wp.vec3(*(d / np.linalg.norm(d))))),
                    radius=0.00336,
                    half_height=float(np.linalg.norm(d) / 2),
                    cfg=cfg,
                    color=color,
                    label=f"cereal/{letter}",
                )
        pieces.append(body)
    info.update(
        robot="ur10",
        tracked_body=carton,
        cereal_pieces=pieces,
        cereal_spoon=spoon,
        bowl_center=bowl_center,
        dense_waypoints=True,
        phase_delay=0,
        task="Pick up an open alphabet cereal carton and pour into a bowl containing a spoon",
    )


def check(world, poses, fps):
    from scipy.spatial.transform import Rotation

    carton = poses[:, world["body_start"] + world["tracked_body"]]
    pieces = poses[:, np.array(world["cereal_pieces"]) + world["body_start"], :3]
    radial = np.linalg.norm(pieces[-1, :, :2] - world["bowl_center"], axis=1)
    inside = (radial < 0.13) & (pieces[-1, :, 2] > 0.005) & (pieces[-1, :, 2] < 0.095)
    lift = float(carton[:, 2].max() - carton[0, 2])
    spoon = poses[:, world["body_start"] + world["cereal_spoon"], :3]
    initial = Rotation.from_quat(carton[0, 3:]).inv().apply(pieces[0] - carton[0, :3])
    initially_inside = (
        (np.abs(initial[:, 0]) < 0.062) & (np.abs(initial[:, 1]) < 0.036) & (np.abs(initial[:, 2]) < 0.098)
    )
    upright = float(Rotation.from_quat(carton[-1, 3:]).as_matrix()[2, 2])
    resting_motion = float(np.linalg.norm(np.ptp(carton[-fps:, :3], axis=0)))
    spoon_in_bowl = bool(np.linalg.norm(spoon[-1, :2] - world["bowl_center"]) < 0.16 and 0.005 < spoon[-1, 2] < 0.10)
    # Estimate bulk fill from the settled top surface, including air between letters.
    piece_poses = poses[-1, np.array(world["cereal_pieces"]) + world["body_start"]]
    top_heights = []
    for i, pose in enumerate(piece_poses):
        points = np.array([[a[0] * 0.0144, a[1] * 0.0144, 0] for stroke in LETTERS["SKILDET"[i % 7]] for a in stroke])
        top_heights.append(float((Rotation.from_quat(pose[3:]).apply(points)[:, 2] + pose[2]).max() + 0.00336))
    fill_height = float(np.quantile(np.array(top_heights)[inside], 0.90)) if inside.any() else 0.013
    h = float(np.clip(fill_height - 0.013, 0, 0.071))
    radius = 0.068 + (0.129 - 0.068) * h / 0.071
    fill_fraction = h * (0.068**2 + 0.068 * radius + radius**2) / (0.071 * (0.068**2 + 0.068 * 0.129 + 0.129**2))
    result = {
        "bulk_fill_fraction_from_settled_height": fill_fraction,
        "settled_surface_height_m": fill_height,
        "letters_initially_in_carton": int(initially_inside.sum()),
        "spoon_remains_in_bowl": spoon_in_bowl,
        "carton_final_up_cosine": upright,
        "carton_resting_motion_m": resting_motion,
        "letters_total": len(radial),
        "letters_in_bowl": int(inside.sum()),
        "carton_lift_m": lift,
        "carton_final_height": float(carton[-1, 2]),
        "finite": bool(np.isfinite(poses).all()),
    }
    result["pass"] = bool(
        result["finite"]
        and lift > 0.16
        and inside.sum() >= 0.90 * len(radial)
        and fill_fraction >= 0.30
        and initially_inside.all()
        and spoon_in_bowl
        and upright > 0.98
        and resting_motion < 0.002
        and carton[-1, 2] > 0.085
    )
    return result
