# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Grasp and slew a passive toy crane to demolish a dry-stacked miniature building.

Command: python -m newton.examples steel_wrecking_ball
"""

import numpy as np
import warp as wp

import newton.examples

from . import _wrecking_crane
from ._tabletop_demo import TabletopDemo


class Example(TabletopDemo):
    task = "crane"

    @staticmethod
    def create_parser():
        parser = TabletopDemo.create_parser()
        # At 12 sweeps the under-converged chain-ball contacts pump spin into the
        # ball after the strike until it back-drives the slew brake.
        parser.set_defaults(iterations=32)
        return parser

    def populate(self, builder, info, ee, down, fingers):
        _wrecking_crane.populate(builder, info, ee, down, wp.get_device())

    def reset(self):
        super().reset()
        self.ball_struck_building = False
        self.slew_min = self.slew_max = 0.0
        self.grasp_frames = 0
        self.release_angle = None
        self.max_release_drift = 0.0

    def test_post_step(self):
        """Require grasp, slew and ball impact, not a spontaneous collapse."""
        super().test_post_step()
        poses = self.state_0.body_q.numpy()
        q = poses[self.info["crane_slew"], 3:]
        angle = float(2 * np.arctan2(q[2], q[3]))
        self.slew_min = min(self.slew_min, angle)
        self.slew_max = max(self.slew_max, angle)
        if self.sim_time >= 6.3:
            if self.release_angle is None:
                self.release_angle = angle
            self.max_release_drift = max(self.max_release_drift, abs(angle - self.release_angle))
        count = int(self.contacts.rigid_contact_count.numpy()[0])
        owners = self.model.shape_body.numpy()
        shapes_a = self.contacts.rigid_contact_shape0.numpy()[:count]
        shapes_b = self.contacts.rigid_contact_shape1.numpy()[:count]
        pairs = {(int(owners[a]), int(owners[b])) for a, b in zip(shapes_a, shapes_b, strict=True) if a >= 0 and b >= 0}

        def touching(a, b):
            return (a, b) in pairs or (b, a) in pairs

        if all(touching(finger, self.info["crane_handle"]) for finger in self.info["crane_fingers"]):
            self.grasp_frames += 1
        self.ball_struck_building |= any(
            touching(self.info["tracked_body"], body) for body in self.info["demolition_structure"]
        )
        if not self.ball_struck_building:
            floors = self.info["demolition_upper_floors"]
            drop = self.initial.body_q.numpy()[floors, 2] - poses[floors, 2]
            assert np.max(np.abs(drop)) < 0.025, f"Building moved before ball impact: {drop}"

    def test_final(self):
        """Verify that the ball stays attached and the upper floors collapse."""
        self.check_capacity()
        assert self.grasp_frames >= 20, f"Handle was not held between both fingers: {self.grasp_frames} frames"
        assert self.slew_max - self.slew_min > 1.0, "Crane did not rotate through the strike"
        assert self.ball_struck_building, "Ball never contacted the building"
        assert self.max_release_drift < 0.05, f"Ball back-drove the crane after release: {self.max_release_drift} rad"
        poses = self.state_0.body_q.numpy()
        floors = self.info["demolition_upper_floors"]
        drop = self.initial.body_q.numpy()[floors, 2] - poses[floors, 2]
        assert np.count_nonzero(drop > 0.07) >= 3, f"Upper floor drops: {drop}"
        for parent, child, a, b in self.info["wrecking_chain_anchors"]:
            pa = np.asarray(wp.transform_point(wp.transform(*poses[parent]), wp.vec3(*a)))
            pb = np.asarray(wp.transform_point(wp.transform(*poses[child]), wp.vec3(*b)))
            assert np.linalg.norm(pa - pb) < 0.005, "Chain joint detached"


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    example = Example(viewer, args)
    newton.examples.run(example, args)
