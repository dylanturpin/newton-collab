# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Compact visible paper tiles and restore original geometry for Metal previews."""

import argparse
import collections
import json
import shutil
from pathlib import Path

import numpy as np


def prepare(original, dressed, camera_path, output):
    source = json.loads((original / "scene.json").read_text())
    meta = json.loads((dressed / "scene.json").read_text())
    cameras = json.loads(camera_path.read_text())
    camera = cameras[0]
    selected = set()
    for view in cameras:
        position, target = np.array(view["position"]), np.array(view["target"])
        forward = target - position
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0, 0, 1])
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        tangent = np.tan(np.deg2rad(view["fov"] / 2))
        for wi, world in enumerate(meta["worlds"]):
            p = np.asarray(world["display_offset"], dtype=float) + np.array([0, 0, 0.45]) - position
            z = p @ forward
            # Union sampled camera views for a move; padding retains nearby
            # edge geometry and shadow casters between those samples.
            if abs(p @ right) < z * tangent * view.get("aspect", 1.5) + 2 and abs(p @ up) < z * tangent + 2:
                selected.add(wi)
    sv = np.memmap(original / "vertices.bin", dtype="<f4", mode="r").reshape(-1, 16)
    si = np.memmap(original / "indices.bin", dtype="<u4", mode="r")
    dv = np.memmap(dressed / "vertices.bin", dtype="<f4", mode="r").reshape(-1, 16)
    di = np.memmap(dressed / "indices.bin", dtype="<u4", mode="r")
    lookup = collections.defaultdict(list)
    for m in source["meshes"]:
        lookup[(m["world"], m["body"], m["name"])].append(m)
    world_lookup = {w["id"]: (i, w) for i, w in enumerate(source["worlds"])}
    counters = collections.Counter()
    output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(dressed / "textures", output / "textures", dirs_exist_ok=True)
    for name in ("positions.bin", "rotations.bin"):
        shutil.copy2(dressed / name, output / name)
    meshes = []
    nv = 0
    ni = 0
    with (output / "vertices.bin").open("wb") as vf, (output / "indices.bin").open("wb") as inf:
        for mesh in meta["meshes"]:
            wi = mesh["world"]
            if wi not in selected:
                continue
            current_indices = di[mesh["first_index"] : mesh["first_index"] + mesh["index_count"]]
            material = int(dv[current_indices[0], 14])
            src_wi, src_world = world_lookup[meta["replica_provenance"][wi]["source_world"]]
            source_body = (
                mesh["body"] - meta["worlds"][wi]["body_start"] + src_world["body_start"]
                if mesh["body"] < meta["recorded_body_count"]
                else source["recorded_body_count"] + src_wi
            )
            key = (src_wi, source_body, mesh["name"])
            if lookup.get(key):
                original_mesh = lookup[key][counters[(wi, key)] % len(lookup[key])]
                counters[(wi, key)] += 1
                ix = si[original_mesh["first_index"] : original_mesh["first_index"] + original_mesh["index_count"]]
                used, faces = np.unique(ix, return_inverse=True)
                attrs = sv[used].copy()
            else:
                used, faces = np.unique(current_indices, return_inverse=True)
                attrs = dv[used].copy()
            attrs.view("<u4")[:, 3] = mesh["body"]
            attrs[:, 14] = material
            if "/retro_lighter" in mesh["name"]:
                spec = meta["materials"][material - 1]
                spec.update(color=[0.96, 0.66, 0.16] if spec["metallic"] > 0.5 else [0.03, 0.66, 0.51], roughness=0.25)
            attrs.tofile(vf)
            (faces.astype("<u4") + nv).tofile(inf)
            meshes.append({**mesh, "first_index": ni, "index_count": len(faces)})
            nv += len(attrs)
            ni += len(faces)
    meta.update(
        meshes=meshes,
        vertex_count=nv,
        index_count=ni,
        overview_geometry_lod=False,
        visible_tile_count=len(selected),
        view_culling_camera=camera,
        view_culling_samples=len(cameras),
        offscreen_tile_culling_margin=2,
    )
    (output / "scene.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{len(selected)} visible/padded tiles; {nv} vertices, {ni // 3} triangles")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("original", "dressed", "camera", "output"):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    prepare(a.original, a.dressed, a.camera, a.output)
