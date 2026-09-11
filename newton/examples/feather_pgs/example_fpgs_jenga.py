# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example FeatherPGS Jenga
#
# Eighteen levels of three blocks, alternating direction, held together by
# friction alone. Drag blocks out with the mouse, or use the panel to poke
# a block on a chosen level. A tall friction-only tower is the classic
# creep test: any tangential drift at the seams shows up as a lean long
# before anything falls.
#
# Command: python -m newton.examples fpgs_jenga
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.feather_pgs._showreel import assert_finite, make_solver

LEVELS = 18
BLOCK_L, BLOCK_W, BLOCK_H = 0.30, 0.10, 0.06


class Example:
    def __init__(self, viewer, args):
        newton.use_coord_layout_targets = True
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.test_mode = bool(getattr(args, "test", False))
        self.poke_level = 9
        self.poke_force = 160.0
        self.poke_duration = 1.5
        self.poke_speed_cap = 1.0
        self.poke_travel = 0.25
        self.poke_body = None
        self.poke_until = -1.0
        self.poked = False

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.rigid_gap = 0.001
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
        wood = newton.ModelBuilder.ShapeConfig(density=700.0, mu=0.45, restitution=0.0)
        self.blocks = []
        self.level_of = []
        for level in range(LEVELS):
            z = BLOCK_H / 2 + level * BLOCK_H
            for i in (-1, 0, 1):
                tint = 0.75 + 0.2 * np.random.default_rng(level * 3 + i + 1).random()
                color = wp.vec3(tint, tint * 0.78, tint * 0.5)
                if level % 2 == 0:
                    pos, hx, hy = wp.vec3(0.0, i * BLOCK_W, z), BLOCK_L / 2, 0.99 * BLOCK_W / 2
                else:
                    pos, hx, hy = wp.vec3(i * BLOCK_W, 0.0, z), 0.99 * BLOCK_W / 2, BLOCK_L / 2
                body = builder.add_body(xform=wp.transform(pos, wp.quat_identity()))
                builder.add_shape_box(body, hx=hx, hy=hy, hz=0.99 * BLOCK_H / 2, cfg=wood, color=color)
                self.blocks.append(body)
                self.level_of.append(level)

        self.model = builder.finalize()
        self.model.rigid_contact_max = 32 * len(self.blocks)
        self.solver = make_solver(self.model, pgs_iterations=32, mf_max_constraints=4096)
        self.collision_pipeline = newton.examples.create_collision_pipeline(
            self.model, args, broad_phase="sap", rigid_contact_max=self.model.rigid_contact_max
        )
        self.contacts = self.collision_pipeline.contacts()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.initial_q = self.state_0.body_q.numpy().copy()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(1.3, -1.6, 1.0), pitch=-12.0, yaw=129.0)

    def poke(self, level: int):
        """Push the middle block of ``level`` out along its long axis with a steady finger force."""
        self.poke_body = self.blocks[3 * level + 1]
        self.poke_direction = np.array([1.0, 0.0, 0.0]) if level % 2 == 0 else np.array([0.0, 1.0, 0.0])
        self.poke_until = self.sim_time + self.poke_duration
        self.poked = True

    def _apply_poke(self):
        """Push like a finger: bounded speed, and let go once the block has cleared the tower."""
        if self.poke_body is None or self.sim_time >= self.poke_until:
            return
        q = self.state_0.body_q.numpy()[self.poke_body]
        qd = self.state_0.body_qd.numpy()[self.poke_body]
        travel = float(np.dot(q[:3] - self.initial_q[self.poke_body, :3], self.poke_direction))
        if travel >= self.poke_travel:
            self.poke_until = -1.0
            return
        if float(np.dot(qd[:3], self.poke_direction)) >= self.poke_speed_cap:
            return
        f = np.zeros((self.model.body_count, 6), dtype=np.float32)
        f[self.poke_body, :3] = self.poke_direction * self.poke_force
        self.state_0.body_f.assign(f)

    def lean(self) -> float:
        """Horizontal drift of the top level relative to its start, in millimetres."""
        q = self.state_0.body_q.numpy()
        top = self.blocks[-3:]
        return float(1.0e3 * np.linalg.norm((q[top, :2] - self.initial_q[top, :2]).mean(axis=0)))

    def step(self):
        if self.test_mode and not self.poked and self.sim_time >= 2.0:
            self.poke(self.poke_level)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self._apply_poke()
            self.viewer.apply_forces(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_scalar("top-level lean [mm]", self.lean())
        self.viewer.end_frame()

    def gui(self, ui):
        ui.text(f"{LEVELS} levels, {len(self.blocks)} blocks. Drag blocks with the mouse.")
        _, self.poke_level = ui.slider_int("Poke level", self.poke_level, 0, LEVELS - 1)
        _, self.poke_force = ui.slider_float("Finger force [N]", self.poke_force, 20.0, 400.0)
        _, self.poke_speed_cap = ui.slider_float("Finger speed cap [m/s]", self.poke_speed_cap, 0.2, 3.0)
        if ui.button("Push middle block out"):
            self.poke(self.poke_level)
        ui.text(f"top-level lean {self.lean():.1f} mm")

    def test_final(self):
        assert_finite(self.state_0.body_q, self.state_0.body_qd)
        q = self.state_0.body_q.numpy()
        if np.any(q[:, 2] < -0.05):
            raise ValueError("a block fell through the ground")
        poked = self.blocks[3 * self.poke_level + 1]
        if np.linalg.norm(q[poked, :2] - self.initial_q[poked, :2]) < 0.1:
            raise ValueError("the poked block did not leave the tower")
        standing = int(np.count_nonzero(q[self.blocks, 2] > BLOCK_H))
        if standing < 0.7 * len(self.blocks):
            raise ValueError(f"the tower collapsed after one poke, {standing} blocks left standing")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=480)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
