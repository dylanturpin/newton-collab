# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check the recorded gripper opens, clears the crane, and settles after slew."""

import argparse
import json
from pathlib import Path

import numpy as np


def validate(run):
    summary = json.loads((run / "model-summary.json").read_text())
    world = next(w for w in summary["worlds"] if w.get("demolition"))
    labels = summary["body_labels"]
    start, count = world["body_start"], world["body_count"]

    def body(suffix):
        return next(i for i in range(start, start + count) if labels[i].endswith(suffix))

    trace = np.load(run / "trace.npz")
    poses, fps = trace["poses"], int(trace["fps"])
    window = poses[round(7.5 * fps) : round(10 * fps), :, :3]
    left, right, tcp = body("fr3_leftfinger"), body("fr3_rightfinger"), body("fr3_hand_tcp")
    metrics = {
        "minimum_opening_m": float(np.linalg.norm(window[:, left] - window[:, right], axis=1).min()),
        "minimum_retreat_m": float(window[:, tcp, 2].min() - poses[round(5.4 * fps), tcp, 2]),
        "maximum_settled_tcp_speed_m_s": float(np.linalg.norm(np.diff(window[:, tcp], axis=0) * fps, axis=1).max()),
    }
    metrics["pass"] = (
        metrics["minimum_opening_m"] > 0.065
        and metrics["minimum_retreat_m"] > 0.15
        and metrics["maximum_settled_tcp_speed_m_s"] < 0.015
    )
    print(json.dumps(metrics, indent=2))
    assert metrics["pass"], "Gripper must release, retreat, and settle after rotating the crane"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    validate(parser.parse_args().run)
