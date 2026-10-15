# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Export the open-well, articulated proc-gen toaster used by the hero scene."""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "assets" / "tasks")
    args = parser.parse_args()
    sys.path[:0] = [str(args.source.resolve()), str(args.source.resolve() / "src")]
    from assetgen import registry  # noqa: PLC0415 -- selected source checkout
    from prepare_task_assets import export_asset  # noqa: PLC0415

    family = registry.load("toaster_studio")
    parameters = family.sample(seed=71, mode="coverage", index=family.branches.index("retro"))
    parameters.update(finish_color="chalk", slot_width=0.036, slot_length=0.155, height=0.225)
    args.output.mkdir(parents=True, exist_ok=True)
    export_asset("toaster", "toaster_studio", parameters, family.build_program(parameters), args.output)


if __name__ == "__main__":
    main()
