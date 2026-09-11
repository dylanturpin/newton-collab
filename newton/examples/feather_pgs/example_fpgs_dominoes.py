# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example FeatherPGS Dominoes
#
# Eighty dominoes stand along a spiral that winds inward. A heavy rolling
# ball kicks the outermost one and the chain reaction runs the whole
# spiral; the turns sit further apart than a tile is tall, so every fall
# is handed on along the line and never across it. Each topple
# is a rolling contact that hands momentum to the next tile through
# friction, so a stall or a jump anywhere in the run is visible at once.
#
# Command: python -m newton.examples fpgs_dominoes
#
###########################################################################

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.feather_pgs._showreel import assert_finite, make_solver

COUNT = 80
HEIGHT, WIDTH, THICKNESS = 1.2, 0.65, 0.18
SPACING = 0.75
RADIUS0, RADIUS_GROWTH = 7.8, -0.07
BALL_RADIUS = 0.3


class Example:
    def __init__(self, viewer, args):
        newton.use_coord_layout_targets = True
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.ball_speed = 5.0

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.rigid_gap = 0.002
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
        tile = newton.ModelBuilder.ShapeConfig(density=800.0, mu=0.4, restitution=0.0)
        self.tiles = []
        a, radius = 0.0, RADIUS0
        first_pos, first_dir = None, None
        for k in range(COUNT):
            pos = wp.vec3(radius * math.cos(a), radius * math.sin(a), HEIGHT / 2.0)
            rot = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), a + math.pi / 2.0)
            body = builder.add_body(xform=wp.transform(pos, rot))
            t = k / (COUNT - 1)
            color = wp.vec3(0.95 - 0.7 * t, 0.3 + 0.5 * t, 0.25 + 0.6 * t)
            builder.add_shape_box(body, hx=THICKNESS / 2, hy=WIDTH / 2, hz=HEIGHT / 2, cfg=tile, color=color)
            self.tiles.append(body)
            if k == 0:
                first_pos = np.array([pos[0], pos[1], pos[2]])
                first_dir = np.array([-math.sin(a), math.cos(a), 0.0])
            # Advance by a fixed arc length so the tile spacing stays constant as the radius shrinks.
            a += SPACING / radius
            radius += RADIUS_GROWTH

        ball_cfg = newton.ModelBuilder.ShapeConfig(density=5000.0, mu=0.2, restitution=0.0)
        start = first_pos - first_dir * 2.0
        start[2] = BALL_RADIUS
        self.ball = builder.add_body(xform=wp.transform(wp.vec3(*start), wp.quat_identity()))
        builder.add_shape_sphere(self.ball, radius=BALL_RADIUS, cfg=ball_cfg, color=wp.vec3(0.2, 0.2, 0.22))
        self.kick_dir = first_dir

        self.model = builder.finalize()
        self.model.rigid_contact_max = 24 * (COUNT + 1)
        self.solver = make_solver(self.model, pgs_iterations=24, mf_max_constraints=4096)
        self.collision_pipeline = newton.examples.create_collision_pipeline(
            self.model, args, broad_phase="sap", rigid_contact_max=self.model.rigid_contact_max
        )
        self.contacts = self.collision_pipeline.contacts()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.kick()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(-9.0, -14.0, 9.0), pitch=-30.0, yaw=57.0)

    def kick(self):
        """Roll the starter ball at the first domino."""
        qd = self.state_0.body_qd.numpy()
        qd[self.ball, :3] = self.kick_dir * self.ball_speed
        qd[self.ball, 3:] = np.cross([0.0, 0.0, 1.0], self.kick_dir) * (self.ball_speed / BALL_RADIUS)
        self.state_0.body_qd.assign(qd)
        newton.eval_ik(self.model, self.state_0, self.state_0.joint_q, self.state_0.joint_qd)

    def fallen(self) -> int:
        q = self.state_0.body_q.numpy()[self.tiles]
        up = np.array([wp.quat_rotate(wp.quat(*row[3:7]), wp.vec3(0.0, 0.0, 1.0))[2] for row in q])
        return int(np.count_nonzero(up < 0.5))

    def step(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_scalar("dominoes fallen", self.fallen())
        self.viewer.end_frame()

    def gui(self, ui):
        ui.text(f"{COUNT} dominoes, {self.fallen()} fallen")
        _, self.ball_speed = ui.slider_float("Ball speed [m/s]", self.ball_speed, 2.0, 15.0)
        if ui.button("Kick again"):
            self.kick()

    def test_final(self):
        assert_finite(self.state_0.body_q, self.state_0.body_qd)
        # The full spiral is a 60 m run and takes about 40 s; the test checks that
        # the chain reaction is well under way and has not stalled.
        if self.fallen() < 50:
            raise ValueError(f"only {self.fallen()} of {COUNT} dominoes fell after {self.sim_time:.0f} s")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=900)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
