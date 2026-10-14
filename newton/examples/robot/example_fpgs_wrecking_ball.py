# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Grasp and slew a passive toy crane to demolish a dry-stacked miniature building.

Command: python -m newton.examples fpgs_wrecking_ball
"""

import numpy as np
import warp as wp

import newton.examples

from . import _wrecking_crane
from ._tabletop_demo import TabletopDemo


class Example(TabletopDemo):
    task = "crane"

    def populate(self, builder, info, ee, down, fingers):
        _wrecking_crane.populate(builder, info, ee, down, wp.get_device(), 0)

    def test_post_step(self):
        """Keep the building stable until the arm starts the strike."""
        super().test_post_step()
        if 3.8 < self.sim_time < 4.0:
            floors = self.info["demolition_upper_floors"]
            drop = self.initial.body_q.numpy()[floors, 2] - self.state_0.body_q.numpy()[floors, 2]
            assert np.max(np.abs(drop)) < 0.025, f"Building collapsed before impact: {drop}"

    def test_final(self):
        """Verify that the ball stays attached and the upper floors collapse."""
        self.check_capacity()
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
