# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Wind an articulated rope onto a passive winch to raise a bucket.

Command: python -m newton.examples steel_toy_well

Drag bodies to apply forces, pause or scrub the robot path, and reset the scene.
Only the robot joints are position-driven; object motion comes from contacts.
"""

import numpy as np
import warp as wp

import newton.examples

from . import _toy_well
from ._tabletop_demo import TabletopDemo


class Example(TabletopDemo):
    task = "toy_well"
    robot_root = (-0.48, -0.3, 0)

    camera_position = (1.25, -1.65, 1.45)

    def populate(self, builder, info, ee, down, fingers):
        _toy_well.populate(builder, info, ee, down, fingers, wp.get_device())

    def reset(self):
        super().reset()
        self.test_poses = [self.initial.body_q.numpy()] if self.test_mode else []

    def test_post_step(self):
        """Reject invalid states and record the physical mechanism trajectory."""
        super().test_post_step()
        self.test_poses.append(self.state_0.body_q.numpy())

    def test_final(self):
        """Validate the completed manipulation and mechanism stability."""
        self.check_capacity()
        result = _toy_well.check(self.info, np.asarray(self.test_poses), self.fps)
        assert result["pass"], result


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    example = Example(viewer, args)
    newton.examples.run(example, args)
