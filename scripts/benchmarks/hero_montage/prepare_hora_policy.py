# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Convert the public HORA checkpoint into numeric Warp inference weights."""

import argparse
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


def main():
    import torch

    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--grasps", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path(__file__).parent / "assets/hora")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    weights = {k: v.float().numpy() for k, v in checkpoint["model"].items()}
    for group in ("running_mean_std", "sa_mean_std"):
        weights.update({f"{group}.{k}": v.float().numpy() for k, v in checkpoint[group].items()})
    np.savez_compressed(args.output / "weights.npz", **weights)
    cache = np.load(args.grasps)
    # Keep distinct grasp candidates for regression against the original
    # learned morphology; no object attachment or forced pose trajectory.
    np.save(args.output / "grasps.npy", cache[:128])
    shutil.copytree(args.source / "assets/allegro/meshes", args.output / "meshes", dirs_exist_ok=True)
    tree = ET.parse(args.source / "assets/allegro/allegro_internal.urdf")
    for mesh in tree.getroot().iter("mesh"):
        mesh.set("filename", mesh.get("filename").replace("allegro/meshes/", "meshes/"))
    tree.write(args.output / "hand.urdf")
    (args.output / "provenance.json").write_text(
        json.dumps(
            {
                "source": "https://github.com/HaozhiQi/hora",
                "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                "controller_hz": 20,
                "object_scale": 0.8,
                "license": "MIT",
                "object_pose_animation": False,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
