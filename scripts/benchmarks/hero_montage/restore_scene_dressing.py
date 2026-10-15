# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Restore approved static furniture/decor onto a newly recorded single scene.

Only exported visual meshes change. Physics recordings and task geometry remain
untouched. This prevents legacy builder furnishings replacing approved dressing.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def restore(source, reference, output):
    if source.resolve() == output.resolve():
        raise ValueError("Write a separate export so the source remains recoverable")
    shutil.copytree(source, output, dirs_exist_ok=True)
    current = json.loads((source / "scene.json").read_text())
    approved = json.loads((reference / "scene.json").read_text())
    assert len(current["worlds"]) == len(approved["worlds"]) == 1

    def dressing(mesh, static_body):
        return mesh["body"] == static_body and (
            "/furnishing/" in mesh["name"] or mesh["name"].startswith("paper-decor/")
        )

    selected = [m for m in approved["meshes"] if dressing(m, approved["recorded_body_count"])]
    assert selected and all(m["body"] == approved["recorded_body_count"] for m in selected)
    removed = [m["name"] for m in current["meshes"] if dressing(m, current["recorded_body_count"])]
    current["meshes"] = [m for m in current["meshes"] if not dressing(m, current["recorded_body_count"])]
    material_offset = len(current["materials"])
    for approved_material in approved["materials"]:
        material = dict(approved_material)
        if material.get("texture"):
            old = reference / material["texture"]
            new = Path("textures") / ("approved-" + old.name)
            (output / new).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old, output / new)
            material["texture"] = str(new)
        current["materials"].append(material)
    vertices = np.fromfile(source / "vertices.bin", dtype="<f4").reshape(-1, 16)
    indices = np.fromfile(source / "indices.bin", dtype="<u4")
    old_vertices = np.fromfile(reference / "vertices.bin", dtype="<f4").reshape(-1, 16)
    old_indices = np.fromfile(reference / "indices.bin", dtype="<u4")
    for record in selected:
        index = old_indices[record["first_index"] : record["first_index"] + record["index_count"]]
        attrs = old_vertices[index].copy()
        attrs[:, 14] += material_offset
        attrs[:, 3].view("<u4")[:] = current["recorded_body_count"]
        first = len(indices)
        indices = np.concatenate((indices, np.arange(len(vertices), len(vertices) + len(attrs), dtype="<u4")))
        vertices = np.concatenate((vertices, attrs))
        current["meshes"].append(dict(record, body=current["recorded_body_count"], first_index=first, world=0))
    vertices.tofile(output / "vertices.bin")
    indices.tofile(output / "indices.bin")
    current.update(vertex_count=len(vertices), index_count=len(indices))
    current["approved_dressing_source"] = str(reference)
    (output / "scene.json").write_text(json.dumps(current, indent=2) + "\n")
    for name in ("positions.bin", "rotations.bin"):
        assert (source / name).read_bytes() == (output / name).read_bytes()
    report = {"removed": removed, "restored": [m["name"] for m in selected], "trace_sha256": current["trace_sha256"]}
    (output / "dressing-restoration.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "reference", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    restore(args.source, args.reference, args.output)
