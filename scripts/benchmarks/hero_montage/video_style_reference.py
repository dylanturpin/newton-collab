# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Shared video style: camera direction and editorial timing.

The user rejected negative world-Z azimuth on 2026-10-15. Use POSITIVE
world-Z camera azimuth around the subject. Do not reverse this on the basis
of apparent object rotation in the image. Preserve the original lighting,
AgX transform, exposure, materials, framing and recorded physics.
Use deliberate task/class decorations only; never add generic clutter,
stationery scatter, or fallback props when a decoration plan is missing.
Historical output/source snapshots are not current style instructions.
For an existing HQ scene, restore its APPROVED exported decorations with
restore_scene_dressing.py; builder defaults are not the editorial dressing.

CLI: uv run python video_style_reference.py input-camera.json output-camera.json
"""

import argparse
import json
import math
from pathlib import Path

INTRO_SECONDS = 10.0
EXCERPT_SECONDS = 3.6
CELL_OFFSET_SECONDS = 0.9
CELL_ORDER = ("top-left", "top-right", "bottom-left", "bottom-right")
# Earlier assemble_cycles_montage.py: 2 source seconds in 1.8 screen seconds;
# final demonstrations: 5 source seconds in 4.6 screen seconds.
QUICK_PLAYBACK_SPEED = 2.0 / 1.8
FINAL_PLAYBACK_SPEED = 5.0 / 4.6


def counterclockwise_camera(camera, degrees=5.0):
    """Return the corrected positive-azimuth camera without changing scene style."""
    if not 0 < degrees <= 10:
        raise ValueError("Use a subtle positive orbit of at most ten degrees")
    result = dict(camera)
    result.pop("keyframes", None)
    target = camera["target"]
    x, y, z = [a - b for a, b in zip(camera["position"], target, strict=True)]
    angle = math.radians(degrees)
    result.update(
        endPosition=[
            target[0] + math.cos(angle) * x - math.sin(angle) * y,
            target[1] + math.sin(angle) * x + math.cos(angle) * y,
            target[2] + z,
        ],
        endTarget=list(target),
        moveStart=min(2.5, camera["duration"] * 0.55),
        moveEnd=camera.get("moveEnd", max(3.0, camera["duration"] - 1)),
    )
    end = [a - b for a, b in zip(result["endPosition"], target, strict=True)]
    assert x * end[1] - y * end[0] > 0, "Camera orbit sign reversed"
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--degrees", type=float, default=5.0)
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(counterclockwise_camera(json.loads(args.source.read_text()), args.degrees), indent=2) + "\n"
    )
