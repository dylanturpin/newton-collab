# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Create a varied 96-tile still from accepted recorded scene templates."""

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np


def expand(source, output, count=96, seed=96, yaw_range=0.0):
    import fast_simplification
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation

    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source / "textures", output / "textures", dirs_exist_ok=True)
    meta = json.loads((source / "scene.json").read_text())
    vertices = np.memmap(source / "vertices.bin", dtype="<f4", mode="r").reshape(-1, 16)
    indices = np.memmap(source / "indices.bin", dtype="<u4", mode="r")
    shape = (meta["sample_count"], meta["body_count"], 4)
    positions = np.fromfile(source / "positions.bin", "<f4").reshape(shape)
    rotations = np.fromfile(source / "rotations.bin", "<f4").reshape(shape)
    rng = np.random.default_rng(seed)
    # Balanced template counts, shuffled to avoid adjacent repeated rows.
    choices = np.arange(count) % len(meta["worlds"])
    rng.shuffle(choices)
    columns = 12
    body_total = sum(meta["worlds"][i]["body_count"] for i in choices)
    poses = np.ones((1, body_total + count, 4), dtype="<f4")
    quats = np.zeros_like(poses)
    meshes, materials, worlds, provenance = [], [], [], []
    geometry = {}
    for wi, world in enumerate(meta["worlds"]):
        geometry[wi] = []
        for mesh in meta["meshes"]:
            if mesh["world"] != wi:
                continue
            tri = indices[mesh["first_index"] : mesh["first_index"] + mesh["index_count"]]
            ids, faces = np.unique(tri, return_inverse=True)
            attrs = vertices[ids].copy()
            faces = faces.reshape(-1, 3)
            # Overview LOD only; close-up exports retain original meshes.
            if len(faces) > 1600:
                original_attrs, original_faces = attrs, faces

                def surface_area(points, triangles):
                    xyz = points[triangles]
                    return np.linalg.norm(np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1).sum()

                # Exported hard-normal seams duplicate positions. Weld them
                # before reduction so the simplifier sees a connected surface,
                # rather than deleting disconnected triangles independently.
                _, first, welded = np.unique(
                    np.round(attrs[:, :3], decimals=8), axis=0, return_index=True, return_inverse=True
                )
                xyz, faces = fast_simplification.simplify(
                    attrs[first, :3].astype("f8"),
                    welded[faces],
                    target_count=min(1600, max(300, len(faces) // 6)),
                )
                nearest = cKDTree(attrs[:, :3]).query(xyz)[1]
                attrs = attrs[nearest].copy()
                attrs[:, :3] = xyz
                original_area = surface_area(original_attrs[:, :3], original_faces)
                reduced_area = surface_area(attrs[:, :3], faces)
                if original_area > 1e-9 and not 0.9 <= reduced_area / original_area <= 1.08:
                    # Keep thin/open material regions intact when reduction
                    # cannot preserve their surface coverage.
                    attrs, faces = original_attrs, original_faces
            geometry[wi].append((mesh, attrs, faces.ravel()))
        print(f"Prepared overview geometry {world['id']}", flush=True)
    nv, ni, body_start = 0, 0, 0
    with (output / "vertices.bin").open("wb") as vf, (output / "indices.bin").open("wb") as inf:
        for cell, wi in enumerate(choices):
            world = meta["worlds"][wi]
            offset = np.array(
                [
                    (cell % columns - (columns - 1) / 2) * 2.65,
                    (cell // columns - (math.ceil(count / columns) - 1) / 2) * 2.65,
                    0.805,
                ]
            )
            offset[:2] += rng.uniform(-0.085, 0.085, 2)
            yaw = float(rng.uniform(-yaw_range, yaw_range))
            rotation = Rotation.from_euler("z", yaw)
            frame = int(rng.integers(40, min(meta["sample_count"], 276)))
            sb, nb = world["body_start"], world["body_count"]
            source_bodies = [*range(sb, sb + nb), meta["recorded_body_count"] + wi]
            target_bodies = [*range(body_start, body_start + nb), body_total + cell]
            body_map = dict(zip(source_bodies, target_bodies, strict=True))
            src = positions[frame, source_bodies, :3] - world["display_offset"]
            poses[0, target_bodies, :3] = rotation.apply(src) + offset
            quats[0, target_bodies] = (rotation * Rotation.from_quat(rotations[frame, source_bodies])).as_quat()
            np.testing.assert_allclose(rotation.inv().apply(poses[0, target_bodies, :3] - offset), src, atol=2e-6)
            tint = rng.uniform(0.82, 1.05, 3)
            paint = [(0.47, 0.55, 0.51), (0.62, 0.66, 0.66), (0.80, 0.75, 0.62), (0.48, 0.55, 0.62)][cell % 4]
            material_map = {}
            for mesh, attrs, faces in geometry[wi]:
                v = attrs.copy()
                v.view("<u4")[:, 3] = [body_map[int(b)] for b in v.view("<u4")[:, 3]]
                old_material = int(v[0, 14])
                bench = "/furnishing/bench_" in mesh["name"]
                key = (old_material, bench)
                if key not in material_map:
                    material = meta["materials"][old_material - 1].copy()
                    if bench and material.get("texture"):
                        material["color"] = (np.asarray(material["color"]) * tint).clip(0, 1).tolist()
                    elif bench and not mesh["name"].endswith("/1"):
                        material["color"] = list(paint)
                    materials.append(material)
                    material_map[key] = len(materials)
                v[:, 14] = material_map[key]
                v.tofile(vf)
                (faces.astype("<u4") + nv).tofile(inf)
                meshes.append(
                    {
                        **mesh,
                        "world": cell,
                        "body": body_map[mesh["body"]],
                        "first_index": ni,
                        "index_count": len(faces),
                    }
                )
                nv += len(v)
                ni += len(faces)
            name = f"tile-{cell:02d}-{world['id']}"
            worlds.append({**world, "id": name, "body_start": body_start, "display_offset": offset.tolist()})
            provenance.append(
                {
                    "tile": name,
                    "source_world": world["id"],
                    "source_frame": frame,
                    "source_time": frame / meta["recording_fps"],
                    "yaw_radians": yaw,
                    "source_export": str(source.resolve()),
                }
            )
            body_start += nb
    poses.tofile(output / "positions.bin")
    quats.tofile(output / "rotations.bin")
    result = {
        **meta,
        "worlds": worlds,
        "body_count": body_total + count,
        "recorded_body_count": body_total,
        "sample_count": 1,
        "vertex_count": nv,
        "index_count": ni,
        "materials": materials,
        "meshes": meshes,
        "replicated_recorded_worlds": True,
        "simultaneous_heterogeneous_batch": False,
        "replication_seed": seed,
        "replication_yaw_range": yaw_range,
        "replica_provenance": provenance,
        "overview_geometry_lod": True,
        "trace_sha256": hashlib.sha256(poses.tobytes() + quats.tobytes()).hexdigest(),
    }
    (output / "scene.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"{count} tiles, {nv} vertices, {ni // 3} triangles", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=96)
    parser.add_argument("--seed", type=int, default=96)
    parser.add_argument(
        "--yaw-range", type=float, default=0.0, help="Optional whole-tile yaw in radians; aligned by default"
    )
    args = parser.parse_args()
    expand(args.source, args.output, args.count, args.seed, args.yaw_range)
