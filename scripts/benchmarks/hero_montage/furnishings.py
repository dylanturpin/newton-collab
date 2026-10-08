# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Material-aware set dressing for the existing contact tasks."""

import json
import math
from functools import cache
from pathlib import Path

import numpy as np
import warp as wp

import newton

ROOT = Path(__file__).parent / "assets" / "furnishings"
WOOD = {"ash": (0.79, 0.67, 0.48), "oak": (0.65, 0.47, 0.28), "walnut": (0.42, 0.28, 0.17)}


@cache
def metadata():
    return json.loads((ROOT / "manifest.json").read_text())["assets"]


def footprint(name, p, angle, scale):
    bounds = np.asarray(metadata()[name]["bounds"])
    sx, sy = (scale, scale) if np.isscalar(scale) else scale[:2]
    points = np.array([[x * sx, y * sy] for x in bounds[:, 0] for y in bounds[:, 1]])
    rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    points = points @ rotation.T + np.asarray(p[:2])
    return points.min(axis=0), points.max(axis=0)


@cache
def tabletop(name):
    from scipy.spatial import ConvexHull

    with np.load(ROOT / f"{name}.npz") as data:
        vertices = np.concatenate([data[f"v{i}"] for i in range(len(metadata()[name]["regions"]))])
    surface = vertices[vertices[:, 2] > -0.006, :2]
    return ConvexHull(surface).equations


@cache
def wood_texture(material):
    """Offline wood fields fitted from the source repository's CC0 profiles."""
    from PIL import Image

    return np.asarray(Image.open(ROOT / f"{material}.png").convert("RGB"))


def visual_cfg(b):
    cfg = b.default_shape_cfg.copy()
    cfg.has_shape_collision = False
    cfg.has_particle_collision = False
    cfg.density = 0
    return cfg


@cache
def catalog(name):
    record = json.loads((ROOT / "manifest.json").read_text())["assets"][name]
    result = []
    with np.load(ROOT / f"{name}.npz") as src:
        for i, region in enumerate(record["regions"]):
            material = region["material"]
            wood = material in WOOD
            mesh = newton.Mesh(
                src[f"v{i}"],
                src[f"f{i}"].ravel(),
                normals=src[f"n{i}"],
                uvs=src[f"uv{i}"],
                compute_inertia=False,
                texture=wood_texture(material) if wood else None,
                roughness=0.46 if wood else 0.32,
                metallic=0.75 if material in ("chrome", "aluminum", "brass") else 0,
            )
            result.append((mesh, (1, 1, 1) if wood else tuple(region["color"])))
    return result


def asset(b, name, p=(0, 0, 0), angle=0, scale=1, body=-1):
    for i, (mesh, color) in enumerate(catalog(name)):
        b.add_shape_mesh(
            body,
            xform=wp.transform(wp.vec3(*p), wp.quat_rpy(0.0, 0.0, float(angle))),
            mesh=mesh,
            scale=(scale,) * 3 if np.isscalar(scale) else scale,
            color=color,
            cfg=visual_cfg(b),
            label=f"furnishing/{name}/{i}",
        )
    if body == -1 and not name.startswith("bench_"):
        if not hasattr(b, "_hero_decor_footprints"):
            b._hero_decor_footprints = []
        b._hero_decor_footprints.append(footprint(name, p, angle, scale))


@cache
def rounded_box(half, material, radius=0.0015):
    import trimesh

    half = np.array(half)
    radius = min(radius, float(half.min()) * 0.3)
    points = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                for axis in range(3):
                    p = half - radius
                    p[axis] = half[axis]
                    points.append(p * np.array([sx, sy, sz]))
    mesh = trimesh.Trimesh(vertices=points).convex_hull
    # Split the flat bevel faces: manufactured edges must remain visible.
    vertices = mesh.vertices[mesh.faces].reshape(-1, 3)
    normals = np.repeat(mesh.face_normals, 3, axis=0)
    uv = vertices[:, :2].copy()
    axis = np.abs(normals).argmax(axis=1)
    uv[axis == 0] = vertices[axis == 0][:, [1, 2]]
    uv[axis == 1] = vertices[axis == 1][:, [0, 2]]
    return newton.Mesh(
        vertices.astype("f4"),
        np.arange(len(vertices), dtype="i4"),
        normals=normals.astype("f4"),
        uvs=uv.astype("f4"),
        compute_inertia=False,
        texture=wood_texture(material) if material in WOOD else None,
        roughness=0.45,
        metallic=0.6 if material == "metal" else 0,
    )


def skin_box(b, index, material="ash", color=(1, 1, 1), radius=0.0015):
    b.shape_flags[index] &= ~int(newton.ShapeFlags.VISIBLE)
    mesh = rounded_box(tuple(b.shape_scale[index]), material, radius)
    if material in WOOD and np.allclose(b.shape_scale[index], (0.09, 0.026, 0.017), atol=1.0e-6):
        mesh = catalog("jenga")[0][0]
        color = (1, 1, 1) if material == "ash" else (0.90, 0.85, 0.77)
    b.add_shape_mesh(
        b.shape_body[index],
        xform=b.shape_transform[index],
        mesh=mesh,
        color=color,
        cfg=visual_cfg(b),
        label=f"furnishing/skin/{index}",
    )


@cache
def label_mesh(text, color, width=0.28, height=0.085):
    from PIL import Image, ImageDraw, ImageFont

    canvas = Image.new("RGB", (768, 256), (235, 233, 219))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, 22, 255), fill=tuple(round(c * 255) for c in color))
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        font = ImageFont.truetype(font_path, 38)
    except OSError:
        font = ImageFont.load_default(size=38)
    draw.text((48, 68), text, font=font, fill=(35, 46, 50))
    draw.text(
        (48, 145),
        "FPGS / CONTACT LAB",
        font=font.font_variant(size=22) if hasattr(font, "font_variant") else font,
        fill=(92, 104, 104),
    )
    return newton.Mesh(
        np.array(
            [
                [-width / 2, -height / 2, 0],
                [width / 2, -height / 2, 0],
                [width / 2, height / 2, 0],
                [-width / 2, height / 2, 0],
            ],
            dtype="f4",
        ),
        np.array([0, 1, 2, 0, 2, 3], dtype="i4"),
        normals=np.tile([0, 0, 1], (4, 1)).astype("f4"),
        uvs=np.array([[0, 1], [1, 1], [1, 0], [0, 0]], dtype="f4"),
        texture=np.array(canvas),
        compute_inertia=False,
        roughness=0.8,
    )


def dress(b, kind, variant, info, color):
    """Append visual-only assets after IK and physics bodies are established."""
    original_count = b.shape_count
    if kind in ("g1", "go2"):
        mesh = rounded_box((1.14, 1.14, 0.33), "paint", 0.02)
        b.add_shape_mesh(
            -1,
            xform=wp.transform(wp.vec3(0.0, 0.0, -0.46), wp.quat_identity()),
            mesh=mesh,
            color=(0.28, 0.34, 0.39),
            cfg=visual_cfg(b),
            label="furnishing/stage",
        )
        return
    for i, label in enumerate(b.shape_label[:original_count]):
        if label in ("plinth", "accent"):
            b.shape_flags[i] &= ~int(newton.ShapeFlags.VISIBLE)
    tables = {
        "lift": ("bench_ash", "bench_white"),
        "stack": ("bench_walnut", "bench_round"),
        "sort": ("bench_oak", "bench_white"),
        "drawer": ("bench_ash", "bench_white"),
        "insert": ("bench_white", "bench_ash"),
        "spill": ("bench_oak", "bench_white"),
        "hand": ("bench_white", "bench_lab"),
        "shadow": ("bench_compact", "bench_lab"),
        "kit": ("bench_ash", "bench_white"),
        "gear": ("bench_white", "bench_lab"),
        "serve": ("bench_oval", "bench_round"),
        "toy": ("bench_white", "bench_oak"),
        "puzzle": ("bench_compact", "bench_walnut"),
        "interlock": ("bench_round", "bench_compact"),
        "pile": ("bench_lab", "bench_compact"),
    }
    table = tables[kind][bool(variant)]
    table = {
        ("drawer", 1): "bench_walnut",
        ("lift", 1): "bench_oval",
        ("stack", 3): "bench_white",
        ("serve", 3): "bench_lab",
    }.get((kind, variant), table)
    footprint_size = 0.88 if kind in ("hand", "gear", "stack") else 1.0
    asset(b, table, p=(0, 0, -0.001), scale=(footprint_size, footprint_size, 1.0))
    # Furnish by activity. No accessory is stamped onto every workstation.
    if kind == "lift":
        asset(b, "office_clipboard", (0.66, -0.64, 0.002), angle=0.17, scale=0.75)
        asset(b, "parts_tray", (0.69, 0.73, 0), angle=0.2, scale=0.75)
    elif kind == "kit":
        asset(b, "mallet", (0.61, -0.66, 0.002), angle=-0.7)
        asset(b, "caddy", (-0.72, 0.63, 0), scale=0.65)
        asset(b, "plant", (0.74, 0.75, 0), scale=0.65)
    elif kind == "insert":
        asset(b, "plant", (-0.72, 0.70, 0), scale=0.55)
        asset(b, "mug", (0.70, 0.63, 0), angle=0.35, scale=0.85)
        asset(b, "office_clipboard", (0.70, -0.58, 0.002), angle=-0.2, scale=0.70)
    elif kind == "pile":
        asset(b, "tool_rail", (0.03, 0.98, 0), scale=0.85)
        asset(b, "fastener_set", (0.73, 0.62, 0.002), scale=0.75)
    elif kind == "gear":
        asset(b, "parts_tray", (0.68, 0.67, 0), angle=math.pi / 2, scale=0.8)
        asset(b, "wrench", (0.68, -0.57, 0.002), angle=-0.4)
        asset(b, "mallet", (-0.65, 0.64, 0.002), angle=0.3, scale=0.75)
    elif kind == "spill":
        asset(b, "wrench", (0.78, -0.50, 0.002), angle=0.35, scale=0.85)
        asset(b, "parts_tray", (0.76, 0.73, 0), scale=0.7)
    elif kind == "hand":
        asset(b, "lamp", (-0.83, 0.66, 0), angle=-0.4, scale=0.66)
        asset(b, "parts_tray", (0.74, 0.64, 0), angle=math.pi / 2, scale=0.75)
    elif kind == "shadow":
        asset(b, "caddy", (-0.65, 0.45, 0), scale=0.65)
    elif kind == "serve":
        if variant == 3:
            asset(b, "knife_block", (0.73, 0.66, 0), angle=0.5, scale=0.75)
            asset(b, "parts_tray", (-0.74, 0.64, 0), scale=0.65)
        elif variant % 2:
            asset(b, "caddy", (0.61, 0.43, 0), scale=0.7)
            asset(b, "plant", (-0.59, 0.64, 0), scale=0.7)
        else:
            asset(b, "plant", (-0.60, 0.62, 0), scale=0.9)
            asset(b, "parts_tray", (0.59, 0.43, 0), scale=0.55)
    elif kind == "toy":
        asset(b, "screwdriver", (0.74, -0.58, 0.002), angle=0.45)
        if variant % 2:
            asset(b, "wrench", (0.73, 0.52, 0.002), angle=0.3)
        else:
            asset(b, "parts_tray", (0.75, 0.60, 0), scale=0.75)
    elif kind == "puzzle":
        asset(b, "parts_tray", (0.65, 0.43, 0), angle=-0.4, scale=0.7)
        if variant % 2:
            asset(b, "plant", (-0.72, 0.67, 0), scale=0.75)
    elif kind == "interlock":
        asset(b, "mug", (-0.65, 0.48, 0), angle=0.6)
        asset(b, "mallet", (0.62, 0.42, 0.002), angle=1.0, scale=0.75)
    if kind == "drawer":
        asset(b, "knife_block", (0.78, 0.72, 0), angle=-0.2)
        asset(b, "caddy", (0.80, 0.09, 0), scale=0.8)
        asset(b, "plant", (-0.85, 0.91, 0), scale=0.60)
        for i in range(original_count):
            label = b.shape_label[i]
            if label in ("cabinet_base", "cabinet_side", "cabinet_back", "drawer_front", "drawer_floor"):
                skin_box(b, i, "walnut" if variant == 1 else "ash")
            elif label == "cabinet_worktop":
                skin_box(b, i, "paint", (0.86, 0.87, 0.84), 0.004)
            elif label == "drawer_handle":
                skin_box(b, i, "metal", (0.48, 0.54, 0.58), 0.004)
            elif b.shape_body[i] == info["drawer_body"] and b.shape_type[i] == newton.GeoType.BOX:
                skin_box(b, i, "walnut" if variant == 1 else "ash")
    if kind == "stack":
        asset(b, "mug", (-0.68, 0.55, 0), angle=2.6)
        if variant == 0:
            asset(b, "plant", (-0.77, 0.78, 0), scale=0.7)
        for i in range(original_count):
            body = b.shape_body[i]
            if body >= 0 and (b.body_label[body].startswith(("jenga_", "spare_jenga")) or body == info["tracked_body"]):
                if b.shape_type[i] == newton.GeoType.BOX and b.shape_scale[i][0] > 0.05:
                    skin_box(b, i, "ash" if body % 3 else "oak")
    if kind == "sort":
        # The cabinet's field/rails occupy the same dimensions as the colliders.
        for i in range(original_count):
            if b.shape_body[i] == -1 and i > 1 and b.shape_label[i] != "robot_pedestal":
                b.shape_flags[i] &= ~int(newton.ShapeFlags.VISIBLE)
        asset(b, "hockey", (0.18, 0.08, 0))
        asset(b, "mug", (-0.84, -0.67, 0))
    # Different, mildly untidy office clusters occupy the edges of each bench.
    # Keep the central manipulation corridor clear; these are visible set props.
    seed = sum((i + 1) * ord(char) for i, char in enumerate(kind)) + 73 * variant
    rng = np.random.default_rng(seed)
    clusters = [
        ("office_sticky_notes", "office_pen", "office_pencil", "office_paper_clip", "office_binder_clip"),
        ("office_index_cards", "office_highlighter", "office_eraser", "office_ruler", "office_paper_clip"),
        ("office_stapler", "office_pen", "office_tape_dispenser", "office_scissors", "office_binder_clip"),
        ("office_pen_cup", "office_document_tray", "office_index_cards", "office_pencil", "office_hole_punch"),
    ]
    decor = []
    surface = tabletop(table)
    for cluster_index, center in enumerate(((0.74, -0.67), (-0.74, 0.67))):
        selection = clusters[(seed + cluster_index) % len(clusters)]
        for name in selection:
            for _ in range(600):
                pos = np.add(center, rng.uniform(-0.20, 0.20, size=2))
                angle = float(rng.uniform(-0.55, 0.55))
                low, high = footprint(name, (*pos, 0), angle, 0.65)
                corners = np.array([[x, y] for x in (low[0], high[0]) for y in (low[1], high[1])])
                if np.max((corners / footprint_size) @ surface[:, :2].T + surface[:, 2]) > -0.015:
                    continue
                occupied = getattr(b, "_hero_decor_footprints", [])
                if any(np.all(high + 0.012 > a) and np.all(low - 0.012 < c) for a, c in occupied):
                    continue
                asset(b, name, (*pos, 0.002), angle=angle, scale=0.65)
                decor.append({"asset": name, "position": pos.tolist(), "angle": angle})
                break
    title = {
        "lift": "01 / BALANCE SCALE",
        "stack": "02 / BALANCE",
        "sort": "03 / AIR HOCKEY",
        "hand": "04 / ALLEGRO",
        "spill": "05 / MARBLE RUN",
        "insert": "06 / KNIFE INSERTION",
        "drawer": "07 / CUTLERY SORT",
        "shadow": "08 / SHADOW HAND",
        "kit": "09 / VIADUCT KIT",
        "gear": "10 / GEAR TRAINER",
        "serve": "11 / SERVING" if variant % 2 else "11 / TOY DISPLAY",
        "toy": "12 / TOY GARAGE",
        "puzzle": "13 / SHAPE PUZZLE",
        "interlock": "14 / CROSS PUZZLE",
        "pile": "15 / BUSSING STATION",
    }[kind]
    b.add_shape_mesh(
        -1,
        xform=wp.transform(wp.vec3(-0.55, -0.72, 0.0015), wp.quat_identity()),
        mesh=label_mesh(title, tuple(color)),
        color=(1, 1, 1),
        cfg=visual_cfg(b),
        label="furnishing/label",
    )
    info["furnishings"] = {
        "source": "proc-gen-3d",
        "table": table,
        "collision_scope": "visual-only set dressing",
        "office_clutter": decor,
    }
