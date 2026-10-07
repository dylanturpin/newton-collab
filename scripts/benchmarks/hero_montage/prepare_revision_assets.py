# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Author stock proc-gen cutlery and fasteners for contact manipulation."""

import argparse
import dataclasses
import json
import math
import sys
from pathlib import Path

import numpy as np
from prepare_task_assets import export_asset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "assets/tasks")
    args = parser.parse_args()
    sys.path[:0] = [str(args.source.resolve()), str(args.source.resolve() / "src")]
    from assetgen import registry  # noqa: PLC0415 -- user-selected authoring source
    from assetgen.design import Program  # noqa: PLC0415
    from assetgen.design.fasteners import FastenerSpec, hexagon, threaded_part  # noqa: PLC0415
    from families.smallwares.tools import build_tools  # noqa: PLC0415
    from families.tool_fastening import bolt, circle, ring  # noqa: PLC0415

    args.output.mkdir(parents=True, exist_ok=True)
    plugin = registry.load("cutlery_studio")
    for style, branch in (("fork", "fork"), ("spoon", "spoon"), ("knife", "table_knife")):
        p = plugin.sample(seed=4, mode="coverage", index=plugin.branches.index(branch), complexity="moderate")
        p.update(length=0.20, width=0.032, height=0.018, grip="fitted", grip_material="black")
        program = Program("cutlery_studio", p, version=plugin.version)
        build_tools(program, p, "cutlery_studio")
        # Stock tools point along Y. Put the grip center at the object origin.
        program.place_since(0, rotation=(0, 0, -90), translation=(-0.08, 0, 0))
        program = program.refined("standard")
        export_asset(f"cutlery_{style}", "cutlery_studio", p, program, args.output)

    plugin = registry.load("hard_toy_studio")
    p = plugin.sample(seed=4, mode="coverage", index=plugin.branches.index("crane_kit"))
    p.update(variant=2)
    export_asset("crane", "hard_toy_studio", p, plugin.build_program(p), args.output)

    # Full helical rendering uses the source generator. Bulk handling collision
    # keeps the smooth shank and open nut/socket, omitting thread microcontacts.
    spec = dataclasses.replace(FastenerSpec.metric(8), segments=24)
    for name in ("hex_bolt", "socket_bolt", "hex_nut", "washer"):
        p = {"kind": name, "metric_size": 8, "source_spec": spec.trace()}
        program = Program("tool_studio", p)
        if name.endswith("bolt"):
            bolt(program, spec, spec.pitch * 6, hex_head=name == "hex_bolt")
        elif name == "hex_nut":
            threaded_part(program, "nut.thread", spec, spec.pitch * 4, internal=True)
        else:
            ring(program, "washer", circle(0.009, 48), circle(0.0045, 32), 0.0018, 0)
        export_asset(name, "tool_studio", p, program, args.output)
        metadata = json.loads((args.output / f"{name}.json").read_text())
        arrays = dict(np.load(args.output / f"{name}.npz"))
        import trimesh

        for bi, body in enumerate(metadata["bodies"]):
            proxies = []
            if name.endswith("bolt"):
                shaft = trimesh.creation.cylinder(radius=0.004, height=0.0075, sections=24)
                shaft.apply_translation((0, 0, 0.00375))
                meshes = [shaft]
                floor = trimesh.creation.cylinder(radius=0.0065, height=0.0036, sections=24)
                floor.apply_translation((0, 0, 0.0093))
                meshes.append(floor)
                if name == "hex_bolt":
                    points = np.asarray(hexagon(spec.nut_af))
                    vertices = np.vstack([np.c_[points, np.full(6, z)] for z in (0.0103, 0.0155)])
                    meshes.append(trimesh.convex.convex_hull(vertices))
                else:
                    for j in range(12):
                        a, c = j * math.tau / 12, (j + 1) * math.tau / 12
                        pts = [
                            [r * math.cos(t), r * math.sin(t), z]
                            for z in (0.0107, 0.0155)
                            for r in (0.0036, 0.0065)
                            for t in (a, c)
                        ]
                        meshes.append(trimesh.convex.convex_hull(np.asarray(pts)))
            else:
                height, inner = (0.005, 0.0041) if name == "hex_nut" else (0.0018, 0.0045)
                meshes = []
                for j in range(12):
                    angles = [j * math.tau / 12, (j + 1) * math.tau / 12]
                    points = []
                    for z in (0, height):
                        for a in angles:
                            r = (0.0065 / math.cos((a % (math.pi / 3)) - math.pi / 6)) if name == "hex_nut" else 0.009
                            points.extend(
                                ((inner * math.cos(a), inner * math.sin(a), z), (r * math.cos(a), r * math.sin(a), z))
                            )
                    meshes.append(trimesh.convex.convex_hull(np.asarray(points)))
            for ci, mesh in enumerate(meshes):
                prefix = f"macro_b{bi}_c{ci}"
                arrays[prefix + "_v"] = np.asarray(mesh.vertices, dtype="f4")
                arrays[prefix + "_f"] = np.asarray(mesh.faces, dtype="i4")
                proxies.append({"prefix": prefix, "kind": "convex", "center": [0, 0, 0], "quaternion": [1, 0, 0, 0]})
            body["collisions"] = proxies
        metadata["collision_scope"] = (
            "Macro convex fastener surfaces with open bore/socket; thread microcontacts omitted for bulk handling"
        )
        (args.output / f"{name}.json").write_text(json.dumps(metadata, indent=2) + "\n")
        np.savez_compressed(args.output / f"{name}.npz", **arrays)


if __name__ == "__main__":
    main()
