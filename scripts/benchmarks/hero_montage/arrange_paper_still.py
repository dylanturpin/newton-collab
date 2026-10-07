# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Rearrange recorded tiles without repeated neighboring tasks; center hands."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def separated_order(kinds, columns, rng):
    """Preserve counts while separating matching tasks in all eight directions."""
    for _ in range(1000):
        remaining = list(rng.permutation(len(kinds)))
        result = []
        for cell in range(len(kinds)):
            row, col = divmod(cell, columns)
            neighbors = [
                result[r * columns + c]
                for r, c in ((row, col - 1), (row - 1, col - 1), (row - 1, col), (row - 1, col + 1))
                if r >= 0 and 0 <= c < columns and r * columns + c < len(result)
            ]
            valid = [i for i in remaining if all(kinds[i] != kinds[n] for n in neighbors)]
            if not valid:
                break
            choice = valid[0]
            remaining.remove(choice)
            result.append(choice)
        if not remaining:
            return result
    raise ValueError("Cannot separate these template counts on the requested grid")


def arrange(source, replicas, columns=12, seed=197):
    from scipy.spatial.transform import Rotation

    original = json.loads((source / "scene.json").read_text())
    meta = json.loads((replicas / "scene.json").read_text())
    poses = np.fromfile(replicas / "positions.bin", "<f4").reshape(-1, 4)
    quats = np.fromfile(replicas / "rotations.bin", "<f4").reshape(-1, 4)
    before = poses.copy()
    worlds = meta["worlds"]
    kinds = ["dexterous-hand" if w["kind"] in ("hand", "shadow") else w["kind"] for w in worlds]
    rng = np.random.default_rng(seed)
    order = separated_order(kinds, columns, rng)
    original_worlds = {w["id"]: (i, w) for i, w in enumerate(original["worlds"])}
    vertices = np.memmap(source / "vertices.bin", dtype="<f4", mode="r").reshape(-1, 16)
    indices = np.memmap(source / "indices.bin", dtype="<u4", mode="r")
    changes = []
    for cell, tile_index in enumerate(order):
        world = worlds[tile_index]
        provenance = meta["replica_provenance"][tile_index]
        old_offset = np.asarray(world["display_offset"])
        offset = np.array(
            [
                (cell % columns - (columns - 1) / 2) * 2.65,
                (cell // columns - (math.ceil(len(worlds) / columns) - 1) / 2) * 2.65,
                0.805,
            ]
        )
        offset[:2] += rng.uniform(-0.085, 0.085, 2)
        delta = offset - old_offset
        bodies = np.arange(world["body_start"], world["body_start"] + world["body_count"])
        static_body = meta["recorded_body_count"] + tile_index
        poses[bodies, :3] += delta
        poses[static_body, :3] += delta
        task_delta = np.zeros(3)
        if world["kind"] in ("hand", "shadow"):
            wi, source_world = original_worlds[provenance["source_world"]]
            points = []
            for mesh in original["meshes"]:
                is_hand = "/shape_" in mesh["name"] if world["kind"] == "hand" else "right_shadow_hand" in mesh["name"]
                if mesh["world"] != wi or not is_hand:
                    continue
                if world["kind"] == "hand" and mesh["body"] >= source_world["body_start"] + 29:
                    continue
                body = world["body_start"] + mesh["body"] - source_world["body_start"]
                used = np.unique(indices[mesh["first_index"] : mesh["first_index"] + mesh["index_count"]])
                points.append(Rotation.from_quat(quats[body]).apply(vertices[used, :3]) + poses[body, :3])
            points = np.concatenate(points)
            hand_center = (points.min(axis=0) + points.max(axis=0)) / 2
            task_delta[:2] = offset[:2] - hand_center[:2]
            poses[bodies, :3] += task_delta
            np.testing.assert_allclose(hand_center[:2] + task_delta[:2], offset[:2], atol=1e-6)
        # Relative transforms within the recorded task remain exactly intact.
        np.testing.assert_allclose(
            poses[bodies, :3] - before[bodies, :3], np.broadcast_to(delta + task_delta, (len(bodies), 3)), atol=3e-6
        )
        world.update(display_offset=offset.tolist(), grid_cell=cell)
        provenance["task_centering_translation"] = task_delta.tolist()
        changes.append(
            {
                "tile": world["id"],
                "translation": delta.tolist(),
                "task_translation": task_delta.tolist(),
                "source_world": provenance["source_world"],
            }
        )
    poses.tofile(replicas / "positions.bin")
    meta["trace_sha256"] = hashlib.sha256(poses.tobytes() + quats.tobytes()).hexdigest()
    meta["paper_layout"] = {
        "seed": seed,
        "columns": columns,
        "neighbor_matching_tasks": 0,
        "diagonal_neighbors_checked": True,
        "hand_centers_verified": True,
    }
    (replicas / "scene.json").write_text(json.dumps(meta, indent=2) + "\n")
    (replicas / "layout-changes.json").write_text(json.dumps(changes, indent=2) + "\n")
    print(json.dumps(meta["paper_layout"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("replicas", type=Path)
    args = parser.parse_args()
    arrange(args.source, args.replicas)
