# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Swap two dressed tiles, preserving recorded poses, materials, and neighbors."""

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np


def visible_tasks(meta, camera):
    eye = np.array(camera["position"])
    forward = np.array(camera["target"]) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    tangent = np.tan(np.deg2rad(camera["fov"] / 2))
    counts = Counter()
    for world in meta["worlds"]:
        point = np.array(world["display_offset"]) + np.array([0, 0, 0.25]) - eye
        depth = point @ forward
        if depth > 0 and abs(point @ right) < depth * tangent * 1.5 and abs(point @ up) < depth * tangent:
            counts[world["kind"]] += 1
    return dict(counts)


def swap(source, output, camera_path, first, second):
    assert source.resolve() != output.resolve()
    shutil.copytree(source, output, dirs_exist_ok=True)
    meta = json.loads((output / "scene.json").read_text())
    camera = json.loads(camera_path.read_text())[0]
    before_counts = visible_tasks(meta, camera)
    poses = np.fromfile(output / "positions.bin", "<f4").reshape(-1, 4)
    before = poses.copy()
    worlds = meta["worlds"]
    a, b = worlds[first], worlds[second]
    offsets = [np.array(b["display_offset"]), np.array(a["display_offset"])]
    changes = []
    for index, offset in zip((first, second), offsets, strict=True):
        world = worlds[index]
        delta = offset - world["display_offset"]
        bodies = list(range(world["body_start"], world["body_start"] + world["body_count"]))
        bodies.append(meta["recorded_body_count"] + index)
        poses[bodies, :3] += delta
        np.testing.assert_allclose(
            poses[bodies, :3] - before[bodies, :3], np.broadcast_to(delta, (len(bodies), 3)), atol=3e-6
        )
        changes.append({"tile": world["id"], "translation": delta.tolist()})
        world["display_offset"] = offset.tolist()
    a["grid_cell"], b["grid_cell"] = b["grid_cell"], a["grid_cell"]

    def kind(world):
        return "hand" if world["kind"] in ("hand", "shadow") else world["kind"]

    for i, world in enumerate(worlds):
        row, col = divmod(world["grid_cell"], 12)
        for other in worlds[:i]:
            other_row, other_col = divmod(other["grid_cell"], 12)
            assert kind(world) != kind(other) or max(abs(row - other_row), abs(col - other_col)) > 1
    after_counts = visible_tasks(meta, camera)
    assert set(after_counts) == {w["kind"] for w in worlds}, "A task is still missing from the overview"
    poses.tofile(output / "positions.bin")
    meta["trace_sha256"] = hashlib.sha256(poses.tobytes() + (output / "rotations.bin").read_bytes()).hexdigest()
    meta["paper_layout"]["visibility_swaps"] = changes
    (output / "scene.json").write_text(json.dumps(meta, indent=2) + "\n")
    (output / "visibility-changes.json").write_text(json.dumps(changes, indent=2) + "\n")
    report = {"before": before_counts, "after": after_counts, "swaps": changes}
    (output / "visibility-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output", "camera"):
        parser.add_argument(name, type=Path)
    parser.add_argument("first", type=int)
    parser.add_argument("second", type=int)
    args = parser.parse_args()
    swap(args.source, args.output, args.camera, args.first, args.second)
