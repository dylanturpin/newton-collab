# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check that a lighter opens only with moving-thumb contact."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from audit import audit


def inspect(folder):
    summary = json.loads((folder / "model-summary.json").read_text())
    world = next(w for w in summary["worlds"] if w.get("lighter"))
    result = next(t for t in audit(folder)["tasks"] if t["id"] == world["id"])
    return world, result, np.load(folder / "executed-joint-targets.npz")["targets"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("positive", "frozen", "no-contact", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    positive, opening, targets = inspect(args.positive)
    frozen, held, held_targets = inspect(args.frozen)
    disabled, no_contact, disabled_targets = inspect(args.no_contact)
    assert opening["pass"], opening
    assert not positive.get("thumb_lid_contact_disabled", False)
    assert disabled["thumb_lid_contact_disabled"]
    for world, check in ((frozen, held), (disabled, no_contact)):
        assert not world.get("force_authoring") and not world["lighter_hinge_actuated"]
        assert abs(check["lid_opening_degrees"]) < 8 and abs(check["final_lid_angle_degrees"]) < 8, check
    np.testing.assert_array_equal(targets, disabled_targets)
    thumb = np.asarray(positive["thumb_command_coordinates"]) + positive["q_start"]
    other = np.ones(targets.shape[1], dtype=bool)
    other[thumb] = False
    np.testing.assert_array_equal(targets[:, other], held_targets[:, other])
    assert np.max(np.abs(targets[:, thumb] - held_targets[:, thumb])) > 0.25
    result = {
        "pass": True,
        "opening": opening,
        "stationary_thumb": held,
        "disabled_thumb_lid_contact": no_contact,
        "identical_active_targets_without_contact": True,
        "trace_sha256": {
            label: hashlib.sha256((folder / "trace.npz").read_bytes()).hexdigest()
            for label, folder in (("positive", args.positive), ("frozen", args.frozen), ("no_contact", args.no_contact))
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
