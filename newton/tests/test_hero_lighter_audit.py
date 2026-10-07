# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Reject assisted lighter openings even when their recorded motion looks valid."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.benchmarks.hero_montage.audit import audit


class TestHeroLighterAudit(unittest.TestCase):
    def test_reject_scripted_contact_force(self):
        """Reject an apparently successful opening driven by an external force."""
        fps = 50
        poses = np.zeros((301, 3, 7), dtype=np.float32)
        poses[:, :, 6] = 1
        angle = np.clip((np.arange(301) / fps - 2) / 2, 0, 1) * 1.7
        poses[:, 2, 3] = np.sin(angle / 2)
        poses[:, 2, 6] = np.cos(angle / 2)
        world = {
            "id": "lighter",
            "kind": "shadow",
            "body_start": 0,
            "body_count": 3,
            "tracked_body": 1,
            "palm_body": 0,
            "lighter_lid": 2,
            "lighter": True,
            "lighter_hinge_actuated": False,
            "force_authoring": {"method": "fitted finger force script"},
        }
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            np.savez(folder / "trace.npz", poses=poses, fps=fps)
            (folder / "model-summary.json").write_text(json.dumps({"worlds": [world]}))
            result = audit(folder)
        self.assertGreater(result["tasks"][0]["lid_opening_degrees"], 80)
        self.assertFalse(result["pass"])


if __name__ == "__main__":
    unittest.main()
