# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Prepare machine-local paths for the archived hero scenes."""

import argparse
import json
import subprocess
from pathlib import Path

import newton

NEWTON_REF = "f8fb7abcbeba2318814a74f3eeb02780ad7925d6"
MENAGERIE_REF = "f054586a8e90465d49ee5be15335c4a0c7f57caf"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--menagerie", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.menagerie.resolve()
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != MENAGERIE_REF:
        parser.error(f"Menagerie must be checked out at {MENAGERIE_REF}, found {revision}")
    for name in (
        "kuka_iiwa_14",
        "wonik_allegro",
        "shadow_hand",
        "kinova_gen3",
        "ufactory_xarm7",
        "universal_robots_ur5e",
        "universal_robots_ur10e",
    ):
        if not (root / name).is_dir():
            parser.error(f"Missing robot directory: {root / name}")
    assets = {
        key: str(newton.utils.download_asset(key, ref=NEWTON_REF))
        for key in ("franka_emika_panda", "unitree_g1", "unitree_go2")
    }
    assets.update({key: str(root / key) for key in ("kuka_iiwa_14", "wonik_allegro")})
    assets["ur5e_menagerie"] = str(root / "universal_robots_ur5e")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(assets, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
