# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example FeatherPGS Galton Board
#
# Hundreds of balls stream through staggered rows of pegs into bins and
# pile up into the familiar bell curve. It is a throughput and robustness
# test for the matrix-free free-body route: many simultaneous ball-ball
# and ball-peg contacts, restitution, and deep piles that must settle
# without jitter.
#
# Command: python -m newton.examples fpgs_galton
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.feather_pgs._showreel import Stepper, assert_finite, make_solver

BALL_RADIUS = 0.01
PEG_RADIUS = 0.006
PITCH = 0.05
ROWS = 12
BINS = 14
BIN_HEIGHT = 0.35
TOP_ROW_Z = 1.2
SLOT_HALF = 0.011
BALLS = 300
SOLVER_OVERRIDES = {"pgs_iterations": 4, "mf_max_constraints": 8192}


class Example:
    def __init__(self, viewer, args):
        newton.use_coord_layout_targets = True
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.show_counts = True

        rng = np.random.default_rng(7)
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.rigid_gap = 0.001
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.4, restitution=0.2))
        static = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.3, restitution=0.4)
        glass = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.05, restitution=0.2, is_visible=False)
        peg_color = wp.vec3(0.75, 0.75, 0.8)
        wood = wp.vec3(0.45, 0.3, 0.15)

        half_width = (BINS / 2.0) * PITCH
        peg_rot = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), np.pi / 2.0)
        for row in range(ROWS):
            z = TOP_ROW_Z - row * PITCH
            count = BINS - 1 if row % 2 == 0 else BINS
            for k in range(count):
                x = (k - (count - 1) / 2.0) * PITCH
                builder.add_shape_capsule(
                    -1,
                    xform=wp.transform(wp.vec3(x, 0.0, z), peg_rot),
                    radius=PEG_RADIUS,
                    half_height=SLOT_HALF,
                    cfg=static,
                    color=peg_color,
                )
        for k in range(BINS + 1):
            x = (k - BINS / 2.0) * PITCH
            builder.add_shape_box(
                -1,
                xform=wp.transform(wp.vec3(x, 0.0, BIN_HEIGHT / 2.0), wp.quat_identity()),
                hx=0.002,
                hy=SLOT_HALF,
                hz=BIN_HEIGHT / 2.0,
                cfg=static,
                color=wood,
            )
        wall_h = TOP_ROW_Z + 0.9
        for sign in (-1.0, 1.0):
            builder.add_shape_box(
                -1,
                xform=wp.transform(wp.vec3(sign * (half_width + 0.01), 0.0, wall_h / 2.0), wp.quat_identity()),
                hx=0.01,
                hy=SLOT_HALF + 0.02,
                hz=wall_h / 2.0,
                cfg=static,
                color=wood,
            )
            builder.add_shape_box(
                -1,
                xform=wp.transform(wp.vec3(0.0, sign * (SLOT_HALF + 0.01), wall_h / 2.0), wp.quat_identity()),
                hx=half_width + 0.02,
                hy=0.01,
                hz=wall_h / 2.0,
                cfg=glass,
            )

        ball_cfg = newton.ModelBuilder.ShapeConfig(density=1200.0, mu=0.3, restitution=0.4)
        self.balls = []
        for i in range(BALLS):
            column = (i % 3) - 1
            x = column * 0.024 + rng.uniform(-0.002, 0.002)
            z = TOP_ROW_Z + 0.15 + (i // 3) * 0.022
            body = builder.add_body(xform=wp.transform(wp.vec3(x, 0.0, z), wp.quat_identity()))
            t = i / BALLS
            builder.add_shape_sphere(
                body, radius=BALL_RADIUS, cfg=ball_cfg, color=wp.vec3(0.95 - 0.6 * t, 0.35 + 0.5 * t, 0.2 + 0.7 * t)
            )
            self.balls.append(body)

        self.model = builder.finalize()
        self.model.rigid_contact_max = 12 * BALLS
        self.solver = make_solver(self.model, **SOLVER_OVERRIDES)
        self.collision_pipeline = newton.examples.create_collision_pipeline(
            self.model, args, broad_phase="sap", rigid_contact_max=12 * BALLS
        )
        self.contacts = self.collision_pipeline.contacts()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.stepper = Stepper(self, solver_overrides=SOLVER_OVERRIDES)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(0.0, -2.4, 0.9), pitch=-8.0, yaw=90.0)

    def step(self):
        self.stepper.step()

    def substep(self):
        self.state_0.clear_forces()
        self.viewer.apply_forces(self.state_0)
        self.collision_pipeline.collide(self.state_0, self.contacts)
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
        self.state_0, self.state_1 = self.state_1, self.state_0

    def bin_counts(self) -> np.ndarray:
        q = self.state_0.body_q.numpy()[self.balls]
        settled = q[:, 2] < BIN_HEIGHT
        index = np.floor(q[settled, 0] / PITCH + BINS / 2.0).astype(int)
        index = np.clip(index, 0, BINS - 1)
        return np.bincount(index, minlength=BINS)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        counts = self.bin_counts()
        self.viewer.log_scalar("balls in bins", int(counts.sum()))
        self.viewer.end_frame()

    def gui(self, ui):
        ui.text(f"Galton board: {BALLS} balls, {ROWS} rows")
        _, self.show_counts = ui.checkbox("Show bin counts", self.show_counts)
        if self.show_counts:
            counts = self.bin_counts()
            peak = max(int(counts.max()), 1)
            for k, c in enumerate(counts):
                ui.text(f"{k:2d} {'#' * int(24 * c / peak):24s} {int(c)}")
        self.stepper.gui(ui)

    def test_final(self):
        assert_finite(self.state_0.body_q, self.state_0.body_qd)
        counts = self.bin_counts()
        if counts.sum() < 0.6 * BALLS:
            raise ValueError(f"only {int(counts.sum())} of {BALLS} balls reached the bins")
        if np.count_nonzero(counts) < 6:
            raise ValueError("balls did not spread across the bins")
        q = self.state_0.body_q.numpy()[self.balls]
        if np.any(q[:, 2] < BALL_RADIUS - 0.005):
            raise ValueError("a ball fell through the floor")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=600)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
