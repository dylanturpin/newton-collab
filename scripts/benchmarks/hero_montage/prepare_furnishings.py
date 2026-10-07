# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Bake selected proc-gen-3d designs into portable Newton visual mesh buffers.

These are set-dressing assets. They do not replace the independently validated
task collision models or turn decorative furniture into simulated mechanisms.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def main():
    import trimesh

    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "assets" / "furnishings")
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    selected = set(args.only.split(",")) if args.only else None
    sys.path[:0] = [str(args.source.resolve()), str(args.source.resolve() / "src")]
    from assetgen import registry  # noqa: PLC0415 -- source checkout is selected above
    from assetgen.design import PALETTES, Program  # noqa: PLC0415
    from assetgen.design.preview import compile_visual  # noqa: PLC0415

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": "https://github.com/LuckyIYI/proc-gen-3d",
        "commit": subprocess.check_output(["git", "-C", str(args.source), "rev-parse", "HEAD"], text=True).strip(),
        "scope": "Original material surfaces for fixed set dressing; task physics is separate.",
        "assets": {},
    }
    if selected and (args.output / "manifest.json").exists():
        manifest = json.loads((args.output / "manifest.json").read_text())
    # Bake periodic wood fields from the repository's CC0 scan-fitted spectra.
    # Keep unit scale/zero rotation so the FFT fields repeat without crop seams.
    from PIL import Image

    material_source = args.source / "tools" / "procedural-materials"
    sys.path.insert(0, str(material_source.resolve()))
    from material_fields import generate  # noqa: PLC0415 -- optional authoring dependency
    from variation import sample_variant  # noqa: PLC0415

    manifest.setdefault("wood_textures", {})
    for wood, color in (("ash", [0.80, 0.70, 0.56]), ("oak", [0.68, 0.51, 0.32]), ("walnut", [0.48, 0.32, 0.20])):
        if selected:
            continue
        name = "wood_" + wood
        parameters = sample_variant(name, 19, index=0)
        parameters.update(scale=1.0, rotation=0.0, color=color, gamma=0.85)
        with np.load(material_source / "profiles" / f"{name}.npz") as profile:
            maps, trace = generate(profile, name, seed=19, parameters=parameters, contrast=0.65)
        Image.fromarray((maps["diffuse"] * 255).astype("uint8")).save(args.output / f"{wood}.png")
        manifest["wood_textures"][wood] = trace

    def save(name, program, align="floor"):
        if selected and name not in selected:
            return
        regions = compile_visual(program)
        low = np.min([r.mesh.bounds[0] for r in regions], axis=0)
        high = np.max([r.mesh.bounds[1] for r in regions], axis=0)
        origin = np.array([0, 0, high[2] if align == "top" else 0 if align == "none" else low[2]])
        data, records = {}, []
        for i, region in enumerate(regions):
            mesh = region.mesh.copy()
            mesh.vertices -= origin
            mesh = trimesh.graph.smooth_shade(mesh, angle=np.deg2rad(38), facet_minarea=None)
            v, n = np.asarray(mesh.vertices, dtype="f4"), np.asarray(mesh.vertex_normals).copy()
            faces = np.asarray(mesh.faces)
            # Boolean slivers can collapse at the simulator's float32 precision.
            triangle = v[faces].astype("f8")
            cross = np.cross(triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0])
            area = np.linalg.norm(cross, axis=1)
            keep = area > 1.0e-16
            faces, cross, area = faces[keep], cross[keep], area[keep]
            for vertex in np.flatnonzero(np.linalg.norm(n, axis=1) < 0.5):
                adjacent = np.flatnonzero((faces == vertex).any(axis=1))
                if len(adjacent):
                    face = adjacent[np.argmax(area[adjacent])]
                    n[vertex] = cross[face] / area[face]
            used, inverse = np.unique(faces, return_inverse=True)
            v, n, faces = v[used], n[used], inverse.reshape(-1, 3)
            n /= np.linalg.norm(n, axis=1)[:, None]
            # Box projection follows each surface; smooth manufactured curves
            # retain their crease-aware normals independently of UV seams.
            dominant = np.abs(n).argmax(axis=1)
            uv = v[:, :2].copy()
            uv[dominant == 0] = v[dominant == 0][:, [1, 2]]
            uv[dominant == 1] = v[dominant == 1][:, [0, 2]]
            data.update(
                {
                    f"v{i}": v.astype("f4"),
                    f"n{i}": n.astype("f4"),
                    f"f{i}": faces.astype("i4"),
                    f"uv{i}": uv.astype("f4"),
                }
            )
            color = list((region.color or PALETTES[region.material][1])[:3])
            records.append({"material": region.material, "color": color, "vertices": len(v), "triangles": len(faces)})
        np.savez_compressed(args.output / f"{name}.npz", **data)
        manifest["assets"][name] = {
            "family": program.family,
            "parameters": program.parameters,
            "regions": records,
            "bounds": [(low - origin).tolist(), (high - origin).tolist()],
        }
        print(name, sum(r["triangles"] for r in records), flush=True)

    def family(name, family_id, branch, overrides=None, seed=31, align="floor"):
        if selected and name not in selected:
            return
        plugin = registry.load(family_id)
        p = plugin.sample(seed=seed, mode="coverage", index=plugin.branches.index(branch))
        p.update(overrides or {})
        program = plugin.build_program(p, quality="standard")
        if name in ("hex_key", "fastener_set"):
            program.place_since(0, rotation=(90, 0, 0))
        save(name, program, align)

    for name, wood, support, support_material in [
        ("bench_ash", "ash", "trestle", "black"),
        ("bench_oak", "oak", "tapered", "black"),
        ("bench_lab", "ash", "hoop", "chrome"),
        ("bench_walnut", "walnut", "trestle", "black"),
    ]:
        family(
            name,
            "table_studio",
            "workbench",
            {
                "width": 2.25,
                "depth": 2.25,
                "height": 0.8,
                "top": "rounded_rectangle",
                "support": support,
                "wood": wood,
                "support_material": support_material,
                "storage": "shelf",
                "top_material": "wood",
                "corner_radius": 0.065,
            },
            align="top",
        )
    for name, top, support, wood, material, depth in [
        ("bench_white", "rounded_rectangle", "portal", "ash", "plastic", 2.0),
        ("bench_round", "round", "tulip", "walnut", "wood", 2.25),
        ("bench_oval", "oval", "radial", "oak", "wood", 2.25),
        ("bench_compact", "capsule", "hairpin", "ash", "wood", 1.85),
    ]:
        family(
            name,
            "table_studio",
            "dining",
            {
                "width": 2.25,
                "depth": depth,
                "height": 0.8,
                "top": top,
                "support": support,
                "wood": wood,
                "support_material": "black",
                "storage": "none",
                "top_material": material,
                "corner_radius": 0.08,
            },
            align="top",
        )
    family(
        "lamp",
        "lamp_studio",
        "anglepoise",
        {
            "height": 0.65,
            "shade_radius": 0.12,
            "finish": "black",
            "shade": "ivory",
        },
    )
    family(
        "mug",
        "drinkware_studio",
        "mug",
        {
            "radius": 0.043,
            "height": 0.105,
            "finish": "blue_glaze",
            "artwork": "none",
            "serving_set": False,
            "profile": "rounded",
            "handle": "ear",
            "fluted": False,
        },
    )
    for kind in ("mallet", "screwdriver", "wrench", "drill", "hex_key", "fastener_set"):
        family(kind, "tool_studio", kind, {"variant": 1, "wood": "ash", "finish_color": "petrol"})
    family("caddy", "utensil_holder_studio", "divided_caddy")
    family("knife_block", "utensil_holder_studio", "knife_block")
    family("parts_tray", "drawer_organizer_studio", "modular_bins")
    family("plant", "plant_studio", "rosette")
    for branch in (
        "pen",
        "pencil",
        "stapler",
        "paper_clip",
        "binder_clip",
        "ruler",
        "scissors",
        "tape_dispenser",
        "sticky_notes",
        "index_cards",
        "clipboard",
        "document_tray",
        "pen_cup",
        "eraser",
        "highlighter",
        "hole_punch",
    ):
        family("office_" + branch, "office_clutter_studio", branch, seed=43)

    # Reuse the game's perforated bed, aluminum rails, cabinet and goal details.
    # Loose pucks/strikers and the support legs are omitted: the existing task
    # owns its simulated puck and robot tool, and this is a tabletop appliance.
    game = registry.load("game_table_studio")
    p = game.sample(seed=23, mode="coverage", index=game.branches.index("air_hockey"))
    p.update(variant=0, layout=0, scale=1.0, finish="walnut")
    g = game.build_program(p, quality="standard")
    g.parts = [part for part in g.parts if not part.name.startswith(("base.", "striker.", "puck"))]
    # Map the authored field exactly onto the established 0.88 x 1.30 m task.
    for part in g.parts:
        part.solids = [
            solid.translate((0, 0, -0.77)).scale((0.88 / 1.02, 1.30 / 1.95, 1)).translate((0, 0, 0.028))
            for solid in part.solids
        ]
    save("hockey", g, align="none")

    # Authored workshop fixtures share the same rounded solid construction API.
    g = Program("hero_fixture", {})
    g.block("back", (1.5, 0.035, 0.35), (0, 0, 0.175), "black", radius=0.012)
    g.block("shelf", (1.5, 0.20, 0.024), (0, -0.08, 0.026), "ash", radius=0.006)
    for x in np.linspace(-0.65, 0.65, 14):
        for z in (0.12, 0.20, 0.28):
            start = len(g.parts)
            g.disc(f"hole.{x}.{z}", 0.004, 0.002, (0, 0, 0), "chrome")
            g.place_since(start, rotation=(90, 0, 0), translation=(x, -0.019, z))
    save("tool_rail", g)

    g = Program("hero_jenga", {})
    g.block("block", (0.18, 0.052, 0.034), (0, 0, 0), "ash", radius=0.0013)
    save("jenga", g, align="none")

    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
