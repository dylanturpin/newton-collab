# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Add tabletop visual set dressing to a saved recorded paper layout."""

import argparse
import json
import math
import shutil
from functools import cache
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).parent / "assets"
PALETTE = [(0.02, 0.64, 0.49), (0.05, 0.40, 0.78), (0.96, 0.68, 0.16)]


def dress(source, output):
    from scipy.spatial import ConvexHull
    from scipy.spatial.transform import Rotation

    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source / "textures", output / "textures", dirs_exist_ok=True)
    for name in ("positions.bin", "rotations.bin"):
        shutil.copy2(source / name, output / name)
    meta = json.loads((source / "scene.json").read_text())
    vertices = np.memmap(source / "vertices.bin", dtype="<f4", mode="r").reshape(-1, 16)
    indices = np.memmap(source / "indices.bin", dtype="<u4", mode="r")
    poses = np.fromfile(source / "positions.bin", dtype="<f4").reshape(-1, 4)
    rotations = np.fromfile(source / "rotations.bin", dtype="<f4").reshape(-1, 4)
    catalog = json.loads((ASSETS / "furnishings/manifest.json").read_text())["assets"]
    for wood in ("ash", "oak", "walnut"):
        shutil.copy2(ASSETS / "furnishings" / f"{wood}.png", output / "textures" / f"decor-{wood}.png")
    rng = np.random.default_rng(718)
    materials = list(meta["materials"])
    meshes = []
    material_cache = {}
    nv = meta["vertex_count"]
    ni = 0
    dressing = []
    shutil.copy2(source / "vertices.bin", output / "vertices.bin")
    output_vertices = np.memmap(output / "vertices.bin", dtype="<f4", mode="r+").reshape(-1, 16)

    @cache
    def asset(name):
        data = np.load(ASSETS / "furnishings" / f"{name}.npz")
        return [
            (data[f"v{i}"], data[f"n{i}"], data[f"f{i}"], data[f"uv{i}"], region)
            for i, region in enumerate(catalog[name]["regions"])
        ]

    def material(color, roughness=0.45, metallic=0, texture=None):
        spec = {"color": list(color), "roughness": roughness, "metallic": metallic, "texture": texture}
        key = json.dumps(spec, sort_keys=True)
        if key not in material_cache:
            materials.append(spec)
            material_cache[key] = len(materials)
        return material_cache[key]

    with (output / "vertices.bin").open("ab") as vf, (output / "indices.bin").open("wb") as inf:

        def emit(wi, name, points, normals, faces, uv, mat, p=(0, 0, 0), angle=0, scale=1):
            nonlocal nv, ni
            rotation = Rotation.from_euler("z", angle)
            scalars = np.broadcast_to(scale, 3)
            transformed = rotation.apply(points * scalars) + p
            normal = rotation.apply(normals / scalars)
            normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-8)
            attrs = np.zeros((len(points), 16), dtype="<f4")
            attrs[:, :3] = transformed
            attrs.view("<u4")[:, 3] = meta["recorded_body_count"] + wi
            attrs[:, 4:7] = normal
            attrs[:, 7] = materials[mat - 1]["roughness"]
            attrs[:, 8:12] = 1
            attrs[:, 12:14] = uv
            attrs[:, 14] = mat
            attrs.tofile(vf)
            (faces.ravel().astype("<u4") + nv).tofile(inf)
            meshes.append(
                {
                    "world": wi,
                    "body": meta["recorded_body_count"] + wi,
                    "name": f"paper-decor/{name}",
                    "first_index": ni,
                    "index_count": faces.size,
                }
            )
            nv += len(points)
            ni += faces.size

        def prop(wi, name, p, angle=0, scale=1, color=None):
            for i, (v, n, f, uv, region) in enumerate(asset(name)):
                wood = region["material"] in ("ash", "oak", "walnut")
                tintable = region["material"] in (
                    "plastic",
                    "ceramic",
                    "blue_glaze",
                    "ivory",
                    "paper_ink",
                    "cloth_linen",
                    "cloth_twill",
                )
                mat = material(
                    (1, 1, 1) if wood else color if color and tintable else region["color"],
                    roughness=region.get("roughness", 0.45),
                    metallic=0.65 if region["material"] in ("chrome", "brass", "aluminum") else 0,
                    texture=f"textures/decor-{region['material']}.png" if wood else None,
                )
                emit(wi, f"{name}/{i}", v, n, f, uv, mat, p, angle, scale)

        def fitted_prop(wi, name, x, y, angle, width=0.42, depth=0.34, z=0.002, color=None):
            bounds = np.asarray(catalog[name]["bounds"])
            center = (bounds[0] + bounds[1]) / 2
            extent = bounds[1] - bounds[0]
            scale = min(width / extent[0], depth / extent[1], 0.43 / extent[2], 1.8)
            xy = Rotation.from_euler("z", angle).apply([center[0] * scale, center[1] * scale, 0])
            prop(wi, name, (x - xy[0], y - xy[1], z), angle, scale, color)
            return float(extent[2] * scale)

        def task_prop(wi, name, p, scale=1, angle=0, color=PALETTE[0]):
            with np.load(ASSETS / "tasks" / f"{name}.npz") as data:
                for key in data.files:
                    if key.endswith("_v") and "_v" in key[:-2]:
                        prefix = key[:-2]
                        v = data[key].copy()
                        v[:, 2] -= v[:, 2].min()
                        emit(
                            wi,
                            name,
                            v,
                            data[prefix + "_n"],
                            data[prefix + "_f"],
                            v[:, :2],
                            material(color, 0.33, 0.15),
                            p,
                            angle,
                            scale,
                        )

        def mark(wi, x, y, width, depth, color):
            v = np.array(
                [
                    [x - width / 2, y - depth / 2, 0.0012],
                    [x + width / 2, y - depth / 2, 0.0012],
                    [x + width / 2, y + depth / 2, 0.0012],
                    [x - width / 2, y + depth / 2, 0.0012],
                ]
            )
            emit(
                wi,
                "guide-mark",
                v,
                np.tile([0, 0, 1], (4, 1)),
                np.array([[0, 1, 2], [0, 2, 3]]),
                v[:, :2],
                material(color, 0.8),
            )

        def cluster(wi, style, x, y, angle):
            color = PALETTE[wi % 3]
            if style == "plate_stack":
                for j in range(5):
                    task_prop(
                        wi,
                        "rack_plate",
                        (x, y, 0.003 + j * 0.021),
                        scale=1.3,
                        color=(0.85, 0.87, 0.83) if j % 2 else PALETTE[0],
                    )
            elif style == "book_stack":
                z = 0.002
                for j in range(3):
                    z += (
                        fitted_prop(
                            wi,
                            "book_journal",
                            x + j * 0.008,
                            y,
                            angle + j * 0.055,
                            width=0.26,
                            depth=0.32,
                            z=z,
                            color=PALETTE[(wi + j) % 3],
                        )
                        + 0.001
                    )
            elif style == "pantry_pair":
                for j in range(2):
                    fitted_prop(
                        wi,
                        "pantry_canister",
                        x + (j - 0.5) * 0.17,
                        y,
                        angle,
                        width=0.15,
                        depth=0.15,
                        color=(0.84, 0.86, 0.80) if j else PALETTE[0],
                    )
            elif style == "construction_stock":
                for j in range(8):
                    yaw = angle + (j // 2 % 2) * math.pi / 2
                    offset = Rotation.from_euler("z", yaw).apply([0, (j % 2 - 0.5) * 0.068, 0])
                    prop(
                        wi,
                        "jenga",
                        (x + offset[0], y + offset[1], 0.018 + j // 2 * 0.034),
                        yaw,
                    )
            elif style == "brick_bin":
                fitted_prop(wi, "crate_vented", x, y, angle, color=color)
                for j in range(9):
                    task_prop(
                        wi,
                        "brick_2x2",
                        (x + (j % 3 - 1) * 0.085, y + (j // 3 - 1) * 0.070, 0.012),
                        scale=2.0,
                        angle=j * 0.3,
                        color=PALETTE[j % 3],
                    )
            else:
                tint = PALETTE[0] if style.startswith("vase_") and wi % 2 else None
                if style in ("crate_vented", "tote_lidded"):
                    tint = [(0.24, 0.49, 0.43), (0.22, 0.40, 0.54)][wi % 2]
                elif style == "mixing_bowl":
                    tint = PALETTE[wi % 2]
                elif style == "desk_task_lamp":
                    tint = PALETTE[wi % 3]
                fitted_prop(wi, style, x, y, angle, color=tint)

        for wi, world in enumerate(meta["worlds"]):
            original_meshes = []
            removed = 0
            for mesh in meta["meshes"]:
                if mesh["world"] != wi:
                    continue
                if "/furnishing/" in mesh["name"]:
                    item = mesh["name"].split("/furnishing/")[1].split("/")[0]
                    if not item.startswith("bench_") and item not in ("skin", "hockey", "stage", "tool_rail"):
                        removed += 1
                        continue
                original_meshes.append(mesh)
            tables = [m for m in original_meshes if "/furnishing/bench_" in m["name"]]
            # Keep the approved per-table mix of wood, cream, gray, blue-gray,
            # and sage. Only the two close-up surfaces get explicit overrides.
            for mesh in original_meshes:
                chunk = indices[mesh["first_index"] : mesh["first_index"] + mesh["index_count"]]
                if mesh in tables:
                    old_material = materials[int(vertices[chunk[0], 14]) - 1]
                    color = np.asarray(old_material["color"])
                    surface = old_material.get("texture") or (color.sum() > 0.8 and old_material["metallic"] < 0.5)
                    override = None
                    if surface and wi in (6, 88):
                        override = material((0.48, 0.55, 0.62) if wi == 6 else (0.47, 0.55, 0.51), 0.53)
                    elif (
                        old_material.get("texture")
                        and color[0] > color[1] + 0.06
                        and (color[0] > color[2] + 0.04 or color[2] > color[1] + 0.06)
                    ):
                        # Remove the random red cast without replacing the wood
                        # scan or lifting its darker grain to pale ash.
                        override = material(
                            (float(color.mean()),) * 3,
                            old_material["roughness"],
                            old_material["metallic"],
                            old_material["texture"],
                        )
                    if override is not None:
                        output_vertices[np.unique(chunk), 14] = override
                        output_vertices[np.unique(chunk), 7] = materials[override - 1]["roughness"]
                chunk.tofile(inf)
                meshes.append({**mesh, "first_index": ni})
                ni += len(chunk)
            if not tables:
                dressing.append({"world": world["id"], "clusters": [], "surface": "locomotion deck"})
                continue
            table_vertices = np.concatenate(
                [
                    vertices[np.unique(indices[m["first_index"] : m["first_index"] + m["index_count"]]), :3]
                    for m in tables
                ]
            )
            top = table_vertices[table_vertices[:, 2] > table_vertices[:, 2].max() - 0.006, :2]
            hull = ConvexHull(top).equations
            occupied = []
            for mesh in original_meshes:
                if mesh in tables:
                    continue
                v = vertices[np.unique(indices[mesh["first_index"] : mesh["first_index"] + mesh["index_count"]]), :3]
                body = mesh["body"]
                v = Rotation.from_quat(rotations[body]).apply(v) + poses[body, :3] - world["display_offset"]
                if v[:, 2].max() < 0.008 or v[:, 2].min() > 0.43:
                    continue
                occupied.append((v[:, :2].min(axis=0) - 0.025, v[:, :2].max(axis=0) + 0.025))
            plans = {
                "serve": [
                    "kitchen_microwave" if wi % 2 else "kitchen_toaster",
                    "kitchen_kettle",
                    "plate_stack",
                ],
                "insert": ["cutting_board", "kitchen_crock", "pantry_pair"],
                "drawer": ["mixing_bowl", "pantry_pair", "kitchen_crock"],
                "gear": ["toy_excavator", "toy_dump_truck", "construction_stock"],
                "pile": ["crate_vented", "drill", "tote_lidded"],
                "lift": ["crate_vented", "tote_lidded", "drill"],
                "kit": ["toy_gear_kit", "drill", "crate_vented"],
                "toy": ["toy_train", "toy_puzzle", "brick_bin"],
                "spill": ["brick_bin", "toy_train", "tote_lidded"],
                "stack": ["book_stack", "ring_stack", "clock_twin"],
                "sort": ["clock_twin", "parts_box", "book_stack"],
                "hand": ["desk_fan", "toy_gear_kit", "book_stack"],
                "shadow": ["clock_mantel", "book_stack", "desk_task_lamp"],
            }
            candidates = [
                (x, y)
                for x in (-0.80, -0.66, -0.42, 0, 0.42, 0.66, 0.80)
                for y in (-0.80, -0.66, -0.42, 0, 0.42, 0.66, 0.80)
                if abs(x) > 0.65 or abs(y) > 0.65
            ]
            rng.shuffle(candidates)
            clusters = []
            for x, y in candidates:
                if len(clusters) >= 3:
                    break
                half = np.array([0.24, 0.21])
                lo = np.array([x, y]) - half
                hi = np.array([x, y]) + half
                corners = np.array([[xx, yy] for xx in (lo[0], hi[0]) for yy in (lo[1], hi[1])])
                if np.max(corners @ hull[:, :2].T + hull[:, 2]) > -0.025:
                    continue
                if any(np.all(hi > a) and np.all(lo < b) for a, b in occupied):
                    continue
                angle = float(rng.uniform(-0.15, 0.15))
                style = plans[world["kind"]][len(clusters)]
                cluster(wi, style, x, y, angle)
                occupied.append((lo, hi))
                clusters.append({"style": style, "position": [x, y], "angle": angle})
            if world["kind"] in ("hand", "shadow"):
                # Painted corner guides and converging stripes highlight the
                # small floating hand without altering its physical scale.
                for sx in (-1, 1):
                    for sy in (-1, 1):
                        mark(wi, sx * 0.31, sy * 0.28, 0.16, 0.018, PALETTE[1])
                        mark(wi, sx * 0.38, sy * 0.22, 0.018, 0.13, PALETTE[1])
                    for j in range(3):
                        mark(wi, sx * (0.47 + j * 0.048), 0, 0.025, 0.19 - j * 0.036, PALETTE[2])
            dressing.append(
                {
                    "world": world["id"],
                    "clusters": clusters,
                    "table": "approved-reference",
                    "hand_guides": world["kind"] in ("hand", "shadow"),
                    "removed_accessory_regions": removed,
                }
            )
            print(world["id"], len(clusters), "approved-reference", flush=True)
    output_vertices.flush()
    meta.update(
        materials=materials,
        meshes=meshes,
        vertex_count=nv,
        index_count=ni,
        paper_dressing=dressing,
        decoration_scope="Static render-only proc-gen set dressing; recorded transforms unchanged",
    )
    (output / "scene.json").write_text(json.dumps(meta, indent=2) + "\n")
    assert (output / "positions.bin").read_bytes() == (source / "positions.bin").read_bytes()
    assert (output / "rotations.bin").read_bytes() == (source / "rotations.bin").read_bytes()
    print("Added clusters:", sum(len(d["clusters"]) for d in dressing), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    dress(args.source, args.output)
