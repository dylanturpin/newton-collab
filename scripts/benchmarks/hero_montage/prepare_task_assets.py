# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Export procedural manipulation objects with original convex cells and joints."""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np


def export_asset(name, family_id, p, program, output):
    """Save original material regions, physical bodies and convex proxies."""
    import trimesh
    from assetgen.assembly.materials import visual_regions  # noqa: PLC0415 -- loaded after source selection

    asset = program.compile()
    root, ordered = asset.graph()
    home = asset.forward_kinematics()
    metadata = {
        "name": name,
        "family": family_id,
        "parameters": p,
        "root": root,
        "joints": [dataclasses.asdict(j) for j in ordered],
        "bodies": [],
        "features": {k: v for k, v in program.features.items() if k != "polymer_finishes"},
    }
    if "assembly_kit" in metadata["features"]:
        metadata["features"]["assembly_kit"] = dict(metadata["features"]["assembly_kit"])
        metadata["features"]["assembly_kit"].pop("source_program", None)
    data = {}
    for bi, body in enumerate(asset.bodies):
        record = {
            "id": body.id,
            "mass": body.mass,
            "com": body.com,
            "inertia": body.inertia,
            "home": home[body.id].tolist(),
            "visuals": [],
            "collisions": [],
            "bounds": body.mesh.bounds.tolist(),
        }
        for vi, region in enumerate(visual_regions(body)):
            mesh = trimesh.graph.smooth_shade(region.mesh, angle=np.deg2rad(38), facet_minarea=None)
            v = np.asarray(mesh.vertices, dtype="f4")
            f = np.asarray(mesh.faces, dtype="i4")
            normals = np.asarray(mesh.vertex_normals).copy()
            tri = v[f].astype("f8")
            cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            area = np.linalg.norm(cross, axis=1)
            keep = area > 1.0e-16
            f, cross, area = f[keep], cross[keep], area[keep]
            for vertex in np.flatnonzero(np.linalg.norm(normals, axis=1) < 0.5):
                adjacent = np.flatnonzero((f == vertex).any(axis=1))
                if len(adjacent):
                    face = adjacent[np.argmax(area[adjacent])]
                    normals[vertex] = cross[face] / area[face]
            used, inverse = np.unique(f, return_inverse=True)
            v, normals, f = v[used], normals[used], inverse.reshape(-1, 3)
            normals /= np.linalg.norm(normals, axis=1)[:, None]
            prefix = f"b{bi}_v{vi}"
            data[prefix + "_v"], data[prefix + "_f"], data[prefix + "_n"] = v, f.astype("i4"), normals.astype("f4")
            record["visuals"].append(
                {"prefix": prefix, "material": region.material.id, "color": region.material.rgba[:3]}
            )
        for ci, proxy in enumerate(body.proxies):
            collision = dataclasses.asdict(proxy)
            prefix = f"b{bi}_c{ci}"
            data[prefix + "_v"] = np.asarray(collision.pop("vertices"), dtype="f4")
            data[prefix + "_f"] = np.asarray(collision.pop("faces"), dtype="i4")
            collision["prefix"] = prefix
            record["collisions"].append(collision)
        metadata["bodies"].append(record)
    np.savez_compressed(output / f"{name}.npz", **data)
    (output / f"{name}.json").write_text(
        json.dumps(metadata, indent=2, default=lambda x: np.asarray(x).tolist()) + "\n"
    )
    print(name, len(asset.bodies), "bodies", sum(len(b.proxies) for b in asset.bodies), "convex cells", flush=True)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "assets" / "tasks")
    args = parser.parse_args()
    sys.path[:0] = [str(args.source.resolve()), str(args.source.resolve() / "src")]
    from assetgen import registry  # noqa: PLC0415

    args.output.mkdir(parents=True, exist_ok=True)
    recipes = [
        (
            "viaduct",
            "assembly_toy_studio",
            "viaduct",
            {
                "tiers": 2,
                "height": 0.31,
                "span": 0.22,
                "connection": "insert",
                "clearance": 0.00045,
                "sequence": ["viaduct", "viaduct"],
            },
        ),
        (
            "spiral",
            "assembly_toy_studio",
            "spiral",
            {
                "tiers": 2,
                "height": 0.31,
                "span": 0.20,
                "connection": "insert",
                "clearance": 0.00045,
                "sequence": ["spiral", "orbital"],
            },
        ),
        ("train", "hard_toy_studio", "train", {"variant": 0}),
        ("truck", "hard_toy_studio", "dump_truck", {"variant": 1}),
        ("gear", "hard_toy_studio", "gear_kit", {"variant": 1}),
        ("puzzle", "hard_toy_studio", "shape_puzzle", {"variant": 1}),
        ("bird", "hard_toy_studio", "bird", {"variant": 0}),
        ("interlock", "hard_toy_studio", "interlocking_puzzle", {"variant": 0}),
        (
            "mug",
            "drinkware_studio",
            "mug",
            {
                "finish": "blue_glaze",
                "radius": 0.04,
                "height": 0.09,
                "artwork": "none",
                "serving_set": False,
                "fluted": False,
                "profile": "straight",
                "handle": "loop",
            },
        ),
    ]
    for name, family_id, branch, overrides in recipes:
        plugin = registry.load(family_id)
        p = plugin.sample(seed=4, mode="coverage", index=plugin.branches.index(branch))
        p.update(overrides)
        program = plugin.build_program(p, quality="standard")
        export_asset(name, family_id, p, program, args.output)


if __name__ == "__main__":
    main()
