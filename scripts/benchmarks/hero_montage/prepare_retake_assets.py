# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Freeze catalog plate/rack/jar/brick geometry and an articulated retro lighter."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from prepare_task_assets import export_asset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "assets" / "tasks")
    args = parser.parse_args()
    sys.path[:0] = [str(args.source), str(args.source / "src")]
    from assetgen import registry  # noqa: PLC0415 -- load the selected catalog source
    from assetgen.design import Program  # noqa: PLC0415 -- load the selected catalog source
    from assetgen.design.articulation import attach_motion  # noqa: PLC0415 -- load the selected catalog source
    from families.board_game_studio.generator import brick  # noqa: PLC0415 -- load the selected catalog source

    args.output.mkdir(parents=True, exist_ok=True)
    plates = registry.load("plate_studio")
    p = plates.sample(12, "coverage", plates.branches.index("dinner"))
    p.update(radius=0.12, depth=0.018, rim="plain", foot="flat", finish="plastic")
    plate = plates.build_program(p, quality="standard")
    trace = plate.trace()
    for part in trace["parts"]:
        if part["op"] == "vessel":
            part["sides"] = 64
    export_asset("rack_plate", plates.id, p, Program.from_trace(trace), args.output)
    racks = registry.load("dish_rack_studio")
    q = racks.sample(12, "coverage", racks.branches.index("plate_rack"))
    q.update(length=0.29, width=0.25, height=0.18, divisions=3, filled=False, dish_branch="dinner", rack_arch=0.18)
    # Match the stock grammar's measured slots to the exact plate recipe above.
    # Rebuild only the rack grammar with its child recipe, rather than guessing
    # slot pitch from a different random dinner plate.
    original_sample = plates.sample
    plates.sample = lambda *a, **kw: dict(p)
    try:
        rack = racks.build_program(q, quality="detailed")
    finally:
        plates.sample = original_sample
    export_asset("plate_rack", racks.id, q, rack, args.output)

    jars = registry.load("food_container_studio")
    p = jars.sample(9, "coverage", jars.branches.index("threaded_jar"))
    p.update(radius=0.049, height=0.10, form="straight", material="glass", lid_material="chrome")
    jar = jars.build_program(p, quality="detailed")
    # An uncapped stock jar, with its authored neck threads and hollow vessel.
    jar.parts = [part for part in jar.parts if not part.name.startswith("lid.")]
    jar.features.pop("motions", None)
    jar.features["closure_removed"] = True
    export_asset("pour_jar", jars.id, p, jar, args.output)

    for nx, ny in ((1, 1), (1, 2), (2, 2), (3, 1), (3, 2), (4, 1)):
        p = {"nx": nx, "ny": ny, "stud_pitch_m": 0.008}
        g = Program("board_game_studio", p, version="1.1.0")
        g.block("temporary_support", (0.001, 0.001, 0.001), (0, 0, -1), "ivory", radius=0.0001)
        brick(g, "brick", 0, 0, 0, nx, ny, "ochre")
        g.parts = [part for part in g.parts if part.name != "temporary_support"]
        # Export one complete object, imported later as a free root.
        g.features.pop("motions", None)
        export_asset(f"brick_{nx}x{ny}", "board_game_studio/brick", p, g, args.output)

    p = {"width": 0.042, "depth": 0.024, "body_height": 0.048, "lid_height": 0.017}
    g = Program("hero_retro_lighter", p, version="1.0.0")
    w, d, h, lh = p.values()
    from assetgen.design.curves import rounded_rectangle  # noqa: PLC0415 -- load the selected catalog source

    contour = np.asarray(rounded_rectangle(w, d, 0.002)) / (w / 2)
    g.vessel("brass_case", [(w / 2, 0), (w / 2, h)], 0.001, 0.0015, "brass", sides=len(contour), contour=contour)
    g.block("sealed_fuel_reservoir", (w - 0.002, d - 0.002, h - 0.003), (0, 0, h / 2), "ivory", radius=0.001)
    for side in (-1, 1):
        g.block(
            f"lacquer_panel.{side}",
            (w - 0.006, 0.001, h - 0.007),
            (0, side * (d / 2 + 0.0002), h / 2),
            "black",
            radius=0.001,
        )
    g.block("burner_deck", (w - 0.004, d - 0.004, 0.002), (0, 0, h + 0.001), "brass", radius=0.0005)
    g.rod("hinge_pin", (-w / 2, -d / 2, h), (-w / 2, d / 2, h), 0.0018, "brass", segments=32)
    g.rod("flint_wheel", (0.010, -0.005, h + 0.005), (0.010, 0.005, h + 0.005), 0.004, "chrome", segments=32)
    for i in range(7):
        g.block(f"deck_rib.{i}", (0.030, 0.0007, 0.001), (-0.004, (i - 3) * 0.0022, h + 0.0025), "brass", radius=0.0001)
    g.rod("burner", (-0.009, 0, h + 0.001), (-0.009, 0, h + 0.008), 0.0025, "brass", segments=24)
    start = len(g.parts)
    g.block("lid.crown", (w, d, 0.002), (0, 0, h + lh - 0.001), "brass", radius=0.0007)
    for side in (-1, 1):
        g.block(f"lid.end.{side}", (0.002, d, lh), (side * (w / 2 - 0.001), 0, h + lh / 2), "brass", radius=0.0006)
        g.block(
            f"lid.wall.{side}", (w - 0.004, 0.002, lh), (0, side * (d / 2 - 0.001), h + lh / 2), "brass", radius=0.0006
        )
        g.block(
            f"lid.lacquer.{side}",
            (w - 0.006, 0.0007, lh - 0.005),
            (0, side * (d / 2 + 0.0001), h + lh / 2),
            "black",
            radius=0.0006,
        )
    for i in range(12):
        g.block(
            f"lid.flute.{i}",
            (0.0007, d - 0.004, 0.0006),
            ((i - 5.5) * (w - 0.005) / 11, 0, h + lh + 0.0001),
            "brass",
            radius=0.0001,
        )
    attach_motion(
        g,
        "lid",
        [part.name for part in g.parts[start:]],
        kind="revolute",
        anchor=(-w / 2, 0, h),
        axis=(0, -1, 0),
        limits=(0, 2.18),
    )
    g.features.update(
        physical_scope="Free lighter case with a passive pin hinge; no combustion model",
        lid_hinge=(-w / 2, 0, h),
        lid_axis=(0, -1, 0),
    )
    export_asset("retro_lighter", g.family, p, g, args.output)
    (args.output / "retake-props-provenance.json").write_text(
        json.dumps(
            {
                "source_commit": "45739ebfef8a8e933ba08b0c087618cd14dc54c9",
                "source_repository": "LuckyIYI/proc-gen-3d",
                "notes": "Stock plate/rack/jar and brick grammar; custom physically articulated lighter",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
