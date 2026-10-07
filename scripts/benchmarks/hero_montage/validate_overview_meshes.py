# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check overview reduction preserves robot surface area rather than dropping faces."""

import argparse
import json
from pathlib import Path

import numpy as np


def validate(source, overview):
    def load(folder):
        scene = json.loads((folder / "scene.json").read_text())
        vertices = np.memmap(folder / "vertices.bin", dtype="<f4", mode="r").reshape(-1, 16)
        indices = np.memmap(folder / "indices.bin", dtype="<u4", mode="r")
        return scene, vertices, indices

    def area(vertices, indices, mesh):
        triangles = indices[mesh["first_index"] : mesh["first_index"] + mesh["index_count"]].reshape(-1, 3)
        xyz = vertices[triangles, :3]
        return float(np.linalg.norm(np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1).sum() / 2)

    original, ov, oi = load(source)
    reduced, rv, ri = load(overview)
    lookup = {mesh["name"]: mesh for mesh in original["meshes"]}
    seen, ratios = set(), []
    for mesh in reduced["meshes"]:
        name = mesh["name"]
        if name in seen or not any(part in name for part in ("/worldbody/", "/franka_hand/", "_description/", "/g1_")):
            continue
        seen.add(name)
        reference = lookup[name]
        if reference["index_count"] <= 4800:
            continue
        source_area = area(ov, oi, reference)
        if source_area > 1e-9:
            ratios.append({"mesh": name, "surface_area_ratio": area(rv, ri, mesh) / source_area})
    assert ratios, "No reduced robot meshes were inspected"
    report = {
        "meshes_checked": len(ratios),
        "minimum_surface_area_ratio": min(r["surface_area_ratio"] for r in ratios),
        "failures": [r for r in ratios if not 0.85 <= r["surface_area_ratio"] <= 1.1],
        "pass": all(0.85 <= r["surface_area_ratio"] <= 1.1 for r in ratios),
    }
    print(json.dumps(report, indent=2))
    assert report["pass"], "Overview robot mesh reduction lost surface coverage"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--overview", type=Path, required=True)
    args = parser.parse_args()
    validate(args.source, args.overview)
