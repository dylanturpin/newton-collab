# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Render recorded close-ups with the same Cycles settings as the overview.

Run with Blender --background --python SCRIPT -- SOURCE PAPER_FOLDER.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_instanced_paper import render


def render_panels(source, folder):
    source, folder = source.resolve(), folder.resolve()
    metadata = json.loads((source / "scene.json").read_text())
    cameras = json.loads((folder / "cycles-miniatures.json").read_text())
    for camera in cameras:
        poses = folder / "cycles-poses" / camera["name"]
        poses.mkdir(parents=True, exist_ok=True)
        arrangement = dict(metadata)
        arrangement["replica_provenance"] = [{"source_world": world["id"]} for world in metadata["worlds"]]
        (poses / "scene.json").write_text(json.dumps(arrangement))
        for filename in ("vertices.bin", "indices.bin"):
            if not (poses / filename).exists():
                (poses / filename).symlink_to(source / filename)
        frame = round(camera["startTime"] * metadata["recording_fps"])
        for filename in ("positions.bin", "rotations.bin"):
            recording = np.memmap(source / filename, dtype="<f4", mode="r").reshape(
                metadata["sample_count"], metadata["body_count"], 4
            )
            np.asarray(recording[frame]).tofile(poses / filename)
        camera_path = poses / "camera.json"
        camera_path.write_text(json.dumps([camera]))
        render(source, poses, camera_path, folder / "miniatures" / camera["name"] / "frame_0000.png", single_scene=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("folder", type=Path)
    arguments = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    render_panels(arguments.source, arguments.folder)
