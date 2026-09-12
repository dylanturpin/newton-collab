# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example FeatherPGS Pyramid
#
# A twelve-row pyramid of seventy-eight boxes settles under gravity, then
# a cannonball is fired into it. The stack is a stiff, redundant contact
# graph that exposes solver creep and jitter before the shot, and the shot
# is a hundred-body pile collapse afterwards.
#
# Command: python -m newton.examples fpgs_pyramid
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.feather_pgs._showreel import Stepper, assert_finite, make_solver

BASE = 12
BOX = 0.5
CANNONBALL_RADIUS = 0.35
FIRE_AT = 1.5
# With persistent friction patches six iterations hold the stack better than twelve
# did with point friction (top-box drift 31 mm vs 103 mm over six seconds at rest).
SOLVER_OVERRIDES = {"pgs_iterations": 6, "mf_max_constraints": 8192}


class Example:
    def __init__(self, viewer, args):
        newton.use_coord_layout_targets = True
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 2
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.test_mode = bool(getattr(args, "test", False))
        self.shot_speed = 25.0
        self.fired = False
        self.pre_shot_top = None

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.rigid_gap = 0.002
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.8))
        crate = newton.ModelBuilder.ShapeConfig(density=300.0, mu=0.8, restitution=0.0)
        half = 0.95 * BOX / 2
        # Rows touch vertically (plus a millimetre) so the stack starts settled; the
        # horizontal 5% gap keeps neighbours from pre-loading each other.
        row_h = 2 * half + 0.001
        self.boxes = []
        for j in range(BASE):
            for i in range(BASE - j):
                x = i * BOX + j * BOX / 2 - BASE * BOX / 2
                z = half + 0.001 + j * row_h
                body = builder.add_body(xform=wp.transform(wp.vec3(x, 0.0, z), wp.quat_identity()))
                t = j / (BASE - 1)
                builder.add_shape_box(
                    body,
                    hx=half,
                    hy=half,
                    hz=half,
                    cfg=crate,
                    color=wp.vec3(0.85 - 0.3 * t, 0.6 - 0.2 * t, 0.3 + 0.4 * t),
                )
                self.boxes.append(body)
        iron = newton.ModelBuilder.ShapeConfig(density=7800.0, mu=0.5, restitution=0.0)
        self.shot_start = np.array([-15.0, 0.0, 1.5])
        self.ball = builder.add_body(xform=wp.transform(wp.vec3(*self.shot_start), wp.quat_identity()))
        builder.add_shape_sphere(self.ball, radius=CANNONBALL_RADIUS, cfg=iron, color=wp.vec3(0.1, 0.1, 0.12))

        self.model = builder.finalize()
        self.model.rigid_contact_max = 48 * (len(self.boxes) + 1)
        self.solver = make_solver(self.model, **SOLVER_OVERRIDES)
        self.collision_pipeline = newton.examples.create_collision_pipeline(
            self.model, args, broad_phase="sap", rigid_contact_max=self.model.rigid_contact_max
        )
        self.contacts = self.collision_pipeline.contacts()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.stepper = Stepper(self, solver_overrides=SOLVER_OVERRIDES)
        self.initial_top = float(self.state_0.body_q.numpy()[self.boxes[-1], 2])

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(-9.0, -13.0, 5.5), pitch=-16.0, yaw=55.0)

    def fire(self):
        q = self.state_0.body_q.numpy()
        qd = self.state_0.body_qd.numpy()
        q[self.ball] = [*self.shot_start, 0.0, 0.0, 0.0, 1.0]
        qd[self.ball] = [self.shot_speed, 0.0, 2.0, 0.0, 0.0, 0.0]
        self.state_0.body_q.assign(q)
        self.state_0.body_qd.assign(qd)
        newton.eval_ik(self.model, self.state_0, self.state_0.joint_q, self.state_0.joint_qd)
        self.pre_shot_top = float(self.state_0.body_q.numpy()[self.boxes, 2].max())
        self.fired = True

    def step(self):
        if self.test_mode and not self.fired and self.sim_time >= FIRE_AT:
            self.fire()
        self.stepper.step()

    def substep(self):
        self.state_0.clear_forces()
        self.viewer.apply_forces(self.state_0)
        self.collision_pipeline.collide(self.state_0, self.contacts)
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
        self.state_0, self.state_1 = self.state_1, self.state_0

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        q = self.state_0.body_q.numpy()
        self.viewer.log_scalar("pyramid top height [m]", float(q[self.boxes, 2].max()))
        self.viewer.log_scalar("contacts", int(self.contacts.rigid_contact_count.numpy()[0]))
        self.viewer.end_frame()

    def gui(self, ui):
        ui.text(f"{len(self.boxes)} boxes, {BASE} rows")
        _, self.shot_speed = ui.slider_float("Shot speed [m/s]", self.shot_speed, 5.0, 60.0)
        if ui.button("Fire cannonball"):
            self.fire()
        q = self.state_0.body_q.numpy()
        ui.text(f"top box height {float(q[self.boxes, 2].max()):.2f} m (settled: {self.initial_top:.2f})")
        self.stepper.gui(ui)

    def test_final(self):
        assert_finite(self.state_0.body_q, self.state_0.body_qd)
        q = self.state_0.body_q.numpy()
        if np.any(q[:, 2] < -0.05):
            raise ValueError("a body fell through the ground")
        if self.pre_shot_top is None or abs(self.pre_shot_top - self.initial_top) > 0.05:
            raise ValueError(f"the pyramid did not stand still before the shot, top moved to {self.pre_shot_top}")
        if q[self.boxes, 2].max() > self.initial_top - 0.5:
            raise ValueError("the cannonball did not knock the pyramid down")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=480)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
