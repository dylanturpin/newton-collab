# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Sweep 96 loose nuts, bolts and washers into a bin using a robot-mounted scraper.

Command: python -m newton.examples fpgs_bolt_sweep
"""

import numpy as np
import warp as wp

import newton.examples

from . import _bolt_sweep
from ._tabletop_demo import TabletopDemo


class Example(TabletopDemo):
    task = "sweep"

    @staticmethod
    def create_parser():
        parser = TabletopDemo.create_parser()
        parser.set_defaults(contact_response="propagation_colored")
        return parser

    def populate(self, builder, info, ee, down, fingers):
        _bolt_sweep.populate(builder, info, ee, down, wp.get_device())

    def test_final(self):
        """Verify that at least 90 fasteners reach the collection bin."""
        self.check_capacity()
        poses = self.state_0.body_q.numpy()
        lo, hi = np.asarray(self.info["pile_bin_bounds"])
        centers = np.array(
            [
                np.asarray(wp.transform_point(wp.transform(*poses[b]), wp.vec3(*c)))
                for b, c in zip(self.info["pile_bodies"], self.info["pile_piece_centers"], strict=True)
            ]
        )
        count = np.count_nonzero(np.all((centers >= lo) & (centers <= hi), axis=1))
        assert count >= 90, f"Only {count}/96 fasteners collected"


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    example = Example(viewer, args)
    newton.examples.run(example, args)
