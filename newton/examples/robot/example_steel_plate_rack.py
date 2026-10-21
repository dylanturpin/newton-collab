# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Lift a free dinner plate and place it in a wire rack beside two resting plates.

Command: python -m newton.examples steel_plate_rack
"""

import numpy as np
import warp as wp

import newton.examples

from . import _plate_rack
from ._tabletop_demo import TabletopDemo


class Example(TabletopDemo):
    task = "plates"

    def populate(self, builder, info, ee, down, fingers):
        _plate_rack.populate(builder, info, ee, fingers, wp.get_device(), 3)

    def test_final(self):
        """Verify a real lift followed by a supported plate placement."""
        self.check_capacity()
        body = self.info["tracked_body"]
        pose = self.state_0.body_q.numpy()[body]
        assert self.peak_height > 0.25, f"Plate was not lifted: {self.peak_height}"
        assert abs(pose[0] - self.info["plate_slot_x"]) < 0.04, f"Wrong rack slot: {pose}"
        assert abs(pose[1] - self.info["rack_origin"][1]) < 0.04, f"Plate missed rack: {pose}"
        assert 0.07 < pose[2] < 0.16, f"Plate is unsupported: {pose}"
        normal = np.asarray(wp.quat_rotate(wp.quat(*pose[3:]), wp.vec3(0, 0, 1)))
        goal_normal = np.asarray(wp.quat_rotate(wp.quat(*self.info["plate_goal_rotation"]), wp.vec3(0, 0, 1)))
        assert abs(np.dot(normal, goal_normal)) > np.cos(np.deg2rad(15)), f"Plate is not upright: {pose}"
        speed = np.linalg.norm(self.state_0.body_qd.numpy()[body])
        assert speed < 0.25, f"Plate did not settle: {speed}"


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    example = Example(viewer, args)
    newton.examples.run(example, args)
