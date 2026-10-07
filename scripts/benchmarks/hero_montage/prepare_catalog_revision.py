# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Compile practical insertion and balancing tasks from the stock catalog."""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from prepare_task_assets import export_asset


def fit_knife_slots(program, parameters):
    """Retain the catalog knives and outline, fitting longitudinal slot walls."""
    from assetgen.design import Program  # noqa: PLC0415 -- pinned source selected by main
    from assetgen.design.form_field import FormField  # noqa: PLC0415
    from manifold3d import CrossSection, triangulate  # noqa: PLC0415 -- optional asset compiler

    width, depth, height = (parameters[key] for key in ("length", "width", "height"))
    field = replace(
        FormField(**parameters["form"]),
        aspect=depth / width,
        amplitude=0,
        lean=0,
        twist=0,
        base=1.0,
        belly=1.0,
        shoulder=1.0,
        mouth=1.0,
    )
    outline = field.points(1, np.linspace(0, 2 * np.pi, 64, endpoint=False)) * width / 2
    slots = program.features["slots"]
    slots["depth"] = 0.038
    region = CrossSection([outline])
    for x in slots["centers_x"]:
        half_width, half_depth = slots["width"] / 2, slots["depth"] / 2
        region -= CrossSection(
            [
                np.array(
                    (
                        (x - half_width, -half_depth),
                        (x + half_width, -half_depth),
                        (x + half_width, half_depth),
                        (x - half_width, half_depth),
                    )
                )
            ]
        )
    loops = region.to_polygons()
    vertices = np.concatenate(loops)
    cells = [[[*vertices[i], z] for z in (0.006, height) for i in face] for face in triangulate(loops)]
    replacement = Program(program.family, parameters)
    replacement.convex_cells("slotted_corpus", cells, "oak")
    index = next(i for i, part in enumerate(program.parts) if part.name == "slotted_corpus")
    program.parts[index] = replacement.parts[0]
    program.features["task_adaptation"] = "38 mm slot depth fitted to the original 35.75 mm chef blade"
    return program


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--only", default="")
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "assets/tasks")
    args = parser.parse_args()
    sys.path[:0] = [str(args.source.resolve()), str(args.source.resolve() / "src")]
    from assetgen import registry  # noqa: PLC0415 -- pinned catalog checkout

    args.output.mkdir(parents=True, exist_ok=True)
    recipes = [
        ("balance_board", "board_game_studio", "balance_board", {"variant": 0, "wood": "walnut"}),
        (
            "knife_block_task",
            "utensil_holder_studio",
            "knife_block",
            {"length": 0.27, "width": 0.18, "height": 0.19, "divisions": 2, "filled": True},
        ),
        (
            "knife_source_task",
            "utensil_holder_studio",
            "knife_block",
            {"length": 0.18, "width": 0.14, "height": 0.187, "divisions": 2, "filled": False},
        ),
    ]
    for name, family, branch, overrides in recipes:
        if args.only and name not in args.only.split(","):
            continue
        plugin = registry.load(family)
        params = plugin.sample(seed=41, mode="coverage", index=plugin.branches.index(branch))
        params.update(overrides)
        program = plugin.build_program(params, quality="standard")
        if name in ("knife_block_task", "knife_source_task"):
            program = fit_knife_slots(program, params)
        export_asset(name, family, params, program, args.output)


if __name__ == "__main__":
    main()
