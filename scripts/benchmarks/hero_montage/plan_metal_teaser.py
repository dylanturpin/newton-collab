# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Plan presentation-only cameras over the accepted simulation recording."""

import argparse
import json
import math
from itertools import product
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scene = json.loads((args.data / "scene.json").read_text())
    worlds = {world["id"]: world for world in scene["worlds"]}
    offsets = {name: np.array(world["display_offset"]) for name, world in worlds.items()}
    trace = np.load(args.trace)
    shots = []

    def shot(name, world, start, duration, target, eye, fov=43, drift=(0.05, 0.02, 0.01)):
        target = np.array(target) + offsets[world]
        eye_offset = np.array(eye)
        end_offset = eye_offset + np.array(drift)
        end_offset *= np.linalg.norm(eye_offset) / np.linalg.norm(end_offset)
        eye = target + eye_offset
        item = {
            "name": name,
            "worlds": [world],
            "startTime": start,
            "duration": duration,
            "position": eye.tolist(),
            "target": target.tolist(),
            "fov": fov,
            "endPosition": (target + end_offset).tolist(),
            "endTarget": target.tolist(),
            "moveStart": 0,
            "moveEnd": duration,
        }
        shots.append(item)
        return item

    center = np.mean(list(offsets.values()), axis=0) + np.array((0, 0, -0.1))
    far = center + np.array([0.30, -0.74, 0.84]) * 12.7 * 1.08
    corners = np.array(list(product((-1.1, 1.1), (-1.1, 1.1), (-0.8, 1.35))))
    bounds = (np.array(list(offsets.values()))[:, None, :] + corners[None, :, :]).reshape(-1, 3)
    distance = np.linalg.norm(far - center)
    forward = (center - far) / distance
    right = np.cross(forward, [0, 0, 1])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    tangent = math.tan(math.radians(43 / 2))
    for _ in range(16):
        relative = bounds - center
        depth = distance + relative @ forward
        x = relative @ right / depth / tangent / (16 / 9)
        y = relative @ up / depth / tangent
        center += right * (x.max() + x.min()) / 2 * tangent * (16 / 9) * distance
        center += up * (y.max() + y.min()) / 2 * tangent * distance
        distance *= max(np.abs(x).max() / 0.93, np.abs(y).max() / 0.90)
    far = center - forward * distance
    # Every task receives six seconds at native simulation speed. The only
    # grid reveal happens after the final individual task.
    duration = 6
    hand = worlds["hand-0"]
    palm_target = trace["poses"][int(4.5 * int(trace["fps"])), hand["body_start"] + hand["tracked_body"], :3]
    shot("01-allegro", "hand-0", 2.0, duration, palm_target, (0.34, -0.43, 0.31), 45)
    shot("02-balance-scale", "lift-0", 3.5, duration, (0.15, 0.05, 0.23), (0.75, -0.95, 0.84), 43)
    # Two real-speed cuts keep opening and completed placement in the
    # drawer task's same six-second allocation.
    shot("03a-drawer-open", "drawer-0", 1.4, 2.8, (0.14, -0.13, 0.21), (0.82, -1.10, 0.84), 45)
    shot("03b-drawer-place", "drawer-0", 7.6, 3.2, (0.14, -0.31, 0.24), (0.66, -0.89, 0.66), 45)
    shot("04-hardware-bin", "pile-0", 4.0, duration, (0.18, 0.14, 0.105), (0.60, -0.72, 1.20), 44)
    if worlds["gear-0"].get("demolition"):
        shot("05-wrecking-ball", "gear-0", 3.8, duration, (0.37, 0.17, 0.23), (0.72, -0.90, 0.65), 43)
    else:
        shot("05-crane", "gear-0", 3.0, duration, (0.30, 0.13, 0.30), (0.72, -0.96, 0.79), 43)
    shot("06-knife-insertion", "insert-0", 6.0, duration, (0.11, 0.22, 0.20), (0.43, -0.61, 0.47), 43)
    # Show actual finger closure and the loaded pour within the hand task's
    # same six-second allocation. A cut makes both interactions readable.
    if worlds["shadow-0"].get("lighter"):
        shot("07a-lighter-grip", "shadow-0", 1.0, 1.8, (0.13, -0.20, 0.39), (0.09, -0.30, 0.10), 38, drift=(0, 0, 0))
        shot("07b-lighter-flick", "shadow-0", 2.8, 2.2, (0.13, -0.20, 0.39), (0.09, -0.30, 0.10), 38, drift=(0, 0, 0))
        shot("07c-lighter-open", "shadow-0", 5.0, 2.0, (0.13, -0.20, 0.39), (0.09, -0.30, 0.10), 38, drift=(0, 0, 0))
    else:
        shot("07a-five-finger-grasp", "shadow-0", 1.0, 1.8, (0.18, -0.30, 0.638), (-0.34, -0.43, 0.34), 45)
        shot("07b-five-finger-pour", "shadow-0", 7.0, 2.2, (0.0, 0.48, 0.67), (-0.28, -0.42, 0.28), 45)
        shot("07c-hardware-catch", "shadow-0", 9.2, 2.0, (0.0, 0.40, 0.06), (-0.40, -0.52, 0.55), 43)
    if worlds["serve-1"].get("plate_rack"):
        shot("08-plate-rack", "serve-1", 5.0, duration, (0.16, 0.10, 0.19), (0.62, -0.80, 0.70), 43)
    else:
        shot("08-serving", "serve-1", 3.0, duration, (0.17, 0.06, 0.13), (0.61, -0.78, 0.69), 43)
    shot("09-toy-truck", "toy-1", 2.0, duration, (0.18, 0.07, 0.13), (0.78, -1.08, 0.93), 43)
    shot("10-hockey", "sort-0", 1.0, duration, (0.18, 0.08, 0.05), (0.75, -1.04, 1.05), 43)
    shot("11-jenga", "stack-0", 3.5, duration, (0.16, 0.07, 0.23), (0.70, -0.91, 0.77), 43)
    chute_name = "12-brick-chutes" if worlds["spill-1"].get("pour_jar") is not None else "12-hardware-chutes"
    shot(chute_name, "spill-1", 3.3, duration, (0.20, 0.34, 0.23), (0.79, -1.10, 0.95), 43)
    shot("13-toy-assembly", "kit-0", 3.0, duration, (0.16, -0.06, 0.20), (0.61, -0.79, 0.70), 43)
    shot("14-humanoid", "g1-0", 2.0, duration, (0, 0, 0.65), (1.45, -2.20, 1.05), 43)
    final_task = shot("15-quadruped", "go2-0", 3.0, duration, (0, 0, 0.38), (1.32, -1.88, 0.97), 43)
    finale = {
        "name": "16-batch-finale",
        "worlds": [],
        "startTime": 9,
        "duration": 6,
        "position": final_task["endPosition"],
        "target": final_task["target"],
        "fov": 43,
        "endPosition": far.tolist(),
        "endTarget": center.tolist(),
        "moveStart": 0.6,
        "moveEnd": 4.8,
        "keyframes": [
            {"time": 0, "position": final_task["endPosition"], "target": final_task["target"], "fov": 43},
            {"time": 0.6, "position": final_task["endPosition"], "target": final_task["target"], "fov": 43},
            {"time": 4.8, "position": far.tolist(), "target": center.tolist(), "fov": 43},
            {"time": 6, "position": (far + np.array((0.12, 0.03, 0))).tolist(), "target": center.tolist(), "fov": 43},
        ],
    }
    shots.append(finale)
    assert all(0 <= item["startTime"] and item["startTime"] + item["duration"] <= 15 for item in shots)
    task_times = {}
    for item in shots[:-1]:
        key = item["worlds"][0]
        task_times[key] = task_times.get(key, 0) + item["duration"]
    assert all(abs(time - duration) < 1.0e-6 for time in task_times.values())
    # Keep neighboring worlds visible across the last shot and the continuous
    # pullback so their geometry and the lighting cannot pop into existence.
    final_task["focusWorld"] = "go2-0"
    final_task["worlds"] = []
    args.output.write_text(json.dumps(shots, indent=2) + "\n")
    print(f"{len(shots)} shots, {sum(item['duration'] for item in shots)} seconds; all playback at 1x")


if __name__ == "__main__":
    main()
