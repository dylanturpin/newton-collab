# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Operate a physical fader to drive a closed-linkage pelican automaton.

Command: python -m newton.examples steel_pelican_bicycle

Drag bodies to apply forces, pause or scrub the robot path, and reset the scene.
Only the robot joints are position-driven; object motion comes from contacts.
"""

import numpy as np
import warp as wp

import newton.examples

from . import _pelican_bicycle
from ._tabletop_demo import TabletopDemo


class Example(TabletopDemo):
    task = "pelican_bicycle"
    robot_root = (-0.48, -0.3, 0)

    camera_position = (1.25, -1.65, 1.45)

    def populate(self, builder, info, ee, down, fingers):
        _pelican_bicycle.populate(builder, info, ee, down, fingers, wp.get_device())

    def update_control(self):
        wp.launch(
            _pelican_bicycle.fader_drive,
            dim=1,
            inputs=[
                self.info["pelican_fader_q"],
                self.info["pelican_wheel_v"],
                self.state_0.joint_q,
                self.control.joint_target_qd,
            ],
        )

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
        starts = self.model.joint_qd_start.numpy()
        lower = self.model.joint_limit_lower.numpy()
        upper = self.model.joint_limit_upper.numpy()
        for name, limits in (("fork_z", (-0.002, 0.022)), ("beak_slider_z", (-0.002, 0.011))):
            joint = next(
                index
                for index, label in enumerate(self.model.joint_label)
                if label == name or label.endswith("/" + name)
            )
            coordinate = starts[joint]
            np.testing.assert_allclose(
                [lower[coordinate], upper[coordinate]], np.asarray(limits) * _pelican_bicycle.SCALE, atol=1e-7
            )
        result = _pelican_bicycle.check(self.info, np.asarray(self.test_poses), self.fps)
        assert result["pass"], result


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    example = Example(viewer, args)
    newton.examples.run(example, args)
