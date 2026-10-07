# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Extract decorated paper close-ups at their accepted recorded times."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
from prepare_metal_paper_view import prepare


def prepare_closeups(source, dressed, cameras_path, output):
    source, dressed, output = (p.resolve() for p in (source, dressed, output))
    original = json.loads((source / "scene.json").read_text())
    arrangement = json.loads((dressed / "scene.json").read_text())
    cameras = json.loads(cameras_path.read_text())
    # Retain the approved table variants used for the decor inspections.
    tiles = {"shadow-0": 88, "pile-0": 6, "serve-1": 38}
    for camera in cameras:
        source_id = camera["worlds"][0]
        wi = tiles[source_id]
        tile = arrangement["worlds"][wi]
        provenance = arrangement["replica_provenance"][wi]
        assert provenance["source_world"] == source_id
        assert provenance["yaw_radians"] == 0
        source_wi, world = next((i, w) for i, w in enumerate(original["worlds"]) if w["id"] == source_id)
        frame = round(camera["startTime"] * original["recording_fps"])
        translation = np.asarray(provenance["task_centering_translation"])
        count = world["body_count"]
        snapshot = output / camera["name"] / "snapshot"
        snapshot.mkdir(parents=True, exist_ok=True)
        meta = copy.deepcopy(arrangement)
        meta.update(
            worlds=[{**world, "body_start": 0}],
            body_count=count + 1,
            recorded_body_count=count,
            sample_count=1,
            replica_provenance=[{**provenance, "source_frame": frame, "source_time": camera["startTime"]}],
            source_snapshot_time=camera["startTime"],
            meshes=[
                {
                    **m,
                    "world": 0,
                    "body": m["body"] - tile["body_start"] if m["body"] < arrangement["recorded_body_count"] else count,
                }
                for m in arrangement["meshes"]
                if m["world"] == wi
            ],
        )
        for filename in ("positions.bin", "rotations.bin"):
            recording = np.memmap(source / filename, dtype="<f4", mode="r").reshape(
                original["sample_count"], original["body_count"], 4
            )
            values = np.concatenate(
                (
                    recording[frame, world["body_start"] : world["body_start"] + count],
                    recording[frame, original["recorded_body_count"] + source_wi][None],
                )
            )
            if filename == "positions.bin":
                values[:count, :3] += translation
            values.tofile(snapshot / filename)
        for filename in ("vertices.bin", "indices.bin", "textures"):
            if not (snapshot / filename).exists():
                (snapshot / filename).symlink_to(dressed / filename)
        (snapshot / "scene.json").write_text(json.dumps(meta))
        snapshot_camera = {
            **camera,
            "startTime": 0,
            "position": (np.asarray(camera["position"]) + translation).tolist(),
            "target": (np.asarray(camera["target"]) + translation).tolist(),
        }
        camera_path = snapshot.parent / "camera.json"
        camera_path.write_text(json.dumps([snapshot_camera], indent=2) + "\n")
        prepare(source, snapshot, camera_path, snapshot.parent / "data")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "dressed", "cameras", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    prepare_closeups(args.source, args.dressed, args.cameras, args.output)
