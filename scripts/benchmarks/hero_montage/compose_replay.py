# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Combine unchanged recorded worlds for a presentation-only overview.

The input JSON names a base Metal export, duration/start, and replacements with
source/world/target/start. This is explicitly not a simultaneous simulation.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


def compose(spec, output):
    output.mkdir(parents=True, exist_ok=True)
    (output / "textures").mkdir(exist_ok=True)
    cache = {}

    def load(path):
        path = Path(path).resolve()
        if path not in cache:
            scene = json.loads((path / "scene.json").read_text())
            shape = (scene["sample_count"], scene["body_count"], 4)
            cache[path] = (
                scene,
                np.fromfile(path / "vertices.bin", "<f4").reshape(-1, 16),
                np.fromfile(path / "indices.bin", "<u4"),
                np.fromfile(path / "positions.bin", "<f4").reshape(shape),
                np.fromfile(path / "rotations.bin", "<f4").reshape(shape),
            )
        return path, cache[path]

    base_path, base = load(spec["base"])
    meta = base[0]
    fps = meta["recording_fps"]
    samples = round(spec["duration"] * fps) + 1
    positions = np.zeros((samples, meta["body_count"], 4), dtype="<f4")
    rotations = np.zeros_like(positions)
    vertices, indices, meshes, materials, provenance = [], [], [], [], []
    nv, ni = 0, 0
    replacements = {r["target"]: r for r in spec["replacements"]}
    for wi, world in enumerate(meta["worlds"]):
        replacement = replacements.get(world["id"])
        path, src = load(replacement["source"]) if replacement else (base_path, base)
        scene, verts, inds, pos, rot = src
        source_id = replacement["world"] if replacement else world["id"]
        swi = next(i for i, w in enumerate(scene["worlds"]) if w["id"] == source_id)
        sw = scene["worlds"][swi]
        assert sw["body_count"] == world["body_count"]
        assert scene["recording_fps"] == fps
        start = replacement["start"] if replacement else spec["start"]
        first = round(start * fps)
        assert abs(first / fps - start) < 1e-7
        assert 0 <= first and first + samples <= scene["sample_count"]
        source_bodies = [
            *range(sw["body_start"], sw["body_start"] + sw["body_count"]),
            scene["recorded_body_count"] + swi,
        ]
        target_bodies = [
            *range(world["body_start"], world["body_start"] + world["body_count"]),
            meta["recorded_body_count"] + wi,
        ]
        body_map = dict(zip(source_bodies, target_bodies, strict=True))
        delta = np.array(world["display_offset"]) - sw["display_offset"]
        for sb, tb in body_map.items():
            positions[:, tb] = pos[first : first + samples, sb]
            positions[:, tb, :3] += delta
            rotations[:, tb] = rot[first : first + samples, sb]
            np.testing.assert_allclose(positions[:, tb, :3] - delta, pos[first : first + samples, sb, :3], atol=1e-6)
            np.testing.assert_array_equal(rotations[:, tb], rot[first : first + samples, sb])
        material_offset = len(materials)
        for source_material in scene["materials"]:
            material = source_material.copy()
            if material.get("texture"):
                texture = path / material["texture"]
                name = hashlib.sha256(texture.read_bytes()).hexdigest() + texture.suffix
                destination = output / "textures" / name
                if not destination.exists():
                    shutil.copy2(texture, destination)
                material["texture"] = "textures/" + name
            materials.append(material)
        for mesh in scene["meshes"]:
            if mesh["world"] != swi:
                continue
            tri = inds[mesh["first_index"] : mesh["first_index"] + mesh["index_count"]]
            unique, remap = np.unique(tri, return_inverse=True)
            v = verts[unique].copy()
            body_ids = v.view("<u4")[:, 3]
            body_ids[:] = [body_map[int(b)] for b in body_ids]
            v[:, 14] += material_offset
            vertices.append(v)
            indices.append((remap + nv).astype("<u4"))
            meshes.append({**mesh, "world": wi, "body": body_map[mesh["body"]], "first_index": ni})
            nv += len(v)
            ni += len(tri)
        provenance.append(
            {
                "world": world["id"],
                "source_world": source_id,
                "source": str(path),
                "trace_sha256": scene["trace_sha256"],
                "start": start,
                "acceptance": replacement.get("acceptance") if replacement else "existing accepted take",
            }
        )
    positions.tofile(output / "positions.bin")
    rotations.tofile(output / "rotations.bin")
    np.concatenate(vertices).tofile(output / "vertices.bin")
    np.concatenate(indices).tofile(output / "indices.bin")
    result = {
        **meta,
        "worlds": [
            {key: world[key] for key in ("id", "kind", "variant", "body_start", "body_count", "display_offset")}
            for world in meta["worlds"]
        ],
        "sample_count": samples,
        "vertex_count": nv,
        "index_count": ni,
        "materials": materials,
        "meshes": meshes,
        "recording_composite": True,
        "simultaneous_heterogeneous_batch": False,
        "source_recordings": provenance,
        "quality_gate_passed": False,
        "user_accepted_recordings": True,
        "trace_sha256": hashlib.sha256(positions.tobytes() + rotations.tobytes()).hexdigest(),
        "trace_hash_format": "positions.bin concatenated with rotations.bin; composed display-space replay",
        "simulation_steps_executed": 0,
        "composition_sha256": hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
    }
    (output / "scene.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compose(json.loads(args.spec.read_text()), args.output)
