# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example FeatherPGS Wrecking Ball
#
# A seven-tonne ball hangs from a crane on an articulated chain and is
# released from a high swing into a two-storey building of free-standing
# columns and slabs. The chain is a Featherstone articulation with a
# hundred-to-one mass ratio between the bob and its links; the building is
# a few dozen free bodies held up by friction alone.
#
# Command: python -m newton.examples fpgs_wrecking_ball
#
###########################################################################

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.feather_pgs._showreel import assert_finite, make_solver

FLOORS = 2
WIDTH, DEPTH = 6.0, 4.0
FLOOR_H, SLAB_T, COLUMN = 1.4, 0.15, 0.3
LINKS = 8
LINK_HALF, LINK_R = 0.3, 0.06
BALL_RADIUS = 0.6
ANCHOR = (-3.0, 0.0, 7.0)
RELEASE_ANGLE = math.radians(70.0)


class Example:
    def __init__(self, viewer, args):
        newton.use_coord_layout_targets = True
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        # Eight substeps: the seven-tonne impact on the articulated chain is stiff.
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.release_angle = RELEASE_ANGLE

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.rigid_gap = 0.003
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.9))
        concrete = newton.ModelBuilder.ShapeConfig(density=1600.0, mu=0.9, restitution=0.0)
        slab_cfg = newton.ModelBuilder.ShapeConfig(density=550.0, mu=0.9, restitution=0.0)
        self.building = []
        col_h = FLOOR_H - SLAB_T
        for floor in range(FLOORS):
            z0 = floor * FLOOR_H
            spots = [(-WIDTH / 2 + gx * WIDTH / 2, -DEPTH / 2 + gy * DEPTH) for gx in range(3) for gy in range(2)]
            spots += [
                (-WIDTH / 4, -DEPTH / 2),
                (WIDTH / 4, -DEPTH / 2),
                (-WIDTH / 4, DEPTH / 2),
                (WIDTH / 4, DEPTH / 2),
            ]
            for x, y in spots:
                body = builder.add_body(xform=wp.transform(wp.vec3(x, y, z0 + col_h / 2), wp.quat_identity()))
                builder.add_shape_box(
                    body, hx=COLUMN / 2, hy=COLUMN / 2, hz=col_h / 2, cfg=concrete, color=wp.vec3(0.75, 0.72, 0.68)
                )
                self.building.append(body)
            slab = builder.add_body(
                xform=wp.transform(wp.vec3(0.0, 0.0, z0 + FLOOR_H - SLAB_T / 2), wp.quat_identity())
            )
            builder.add_shape_box(
                slab, hx=WIDTH / 2 + 0.4, hy=DEPTH / 2 + 0.4, hz=SLAB_T / 2, cfg=slab_cfg, color=wp.vec3(0.6, 0.6, 0.62)
            )
            self.building.append(slab)

        # Crane chain: ball joints between capsule links, hanging from a fixed anchor.
        steel = newton.ModelBuilder.ShapeConfig(density=7800.0, mu=0.5, restitution=0.0)
        anchor = wp.vec3(*ANCHOR)
        joints, parent = [], -1
        self.links = []
        for i in range(LINKS):
            z = ANCHOR[2] - (2 * i + 1) * LINK_HALF
            link = builder.add_link(xform=wp.transform(wp.vec3(ANCHOR[0], ANCHOR[1], z), wp.quat_identity()))
            builder.add_shape_capsule(
                link, radius=LINK_R, half_height=LINK_HALF, cfg=steel, color=wp.vec3(0.3, 0.3, 0.32)
            )
            if parent < 0:
                joint = builder.add_joint_ball(
                    -1,
                    link,
                    parent_xform=wp.transform(anchor, wp.quat_identity()),
                    child_xform=wp.transform(wp.vec3(0.0, 0.0, LINK_HALF), wp.quat_identity()),
                )
            else:
                joint = builder.add_joint_ball(
                    parent,
                    link,
                    parent_xform=wp.transform(wp.vec3(0.0, 0.0, -LINK_HALF), wp.quat_identity()),
                    child_xform=wp.transform(wp.vec3(0.0, 0.0, LINK_HALF), wp.quat_identity()),
                )
            joints.append(joint)
            self.links.append(link)
            parent = link
        ball_z = ANCHOR[2] - 2 * LINKS * LINK_HALF - BALL_RADIUS
        self.ball = builder.add_link(xform=wp.transform(wp.vec3(ANCHOR[0], ANCHOR[1], ball_z), wp.quat_identity()))
        builder.add_shape_sphere(self.ball, radius=BALL_RADIUS, cfg=steel, color=wp.vec3(0.12, 0.12, 0.13))
        joints.append(
            builder.add_joint_ball(
                parent,
                self.ball,
                parent_xform=wp.transform(wp.vec3(0.0, 0.0, -LINK_HALF), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(0.0, 0.0, BALL_RADIUS), wp.quat_identity()),
            )
        )
        builder.add_articulation(joints, label="crane")
        self.root_joint = joints[0]

        self.model = builder.finalize()
        self.model.rigid_contact_max = 4096
        self.solver = make_solver(self.model, pgs_iterations=32, dense_max_constraints=2048, mf_max_constraints=4096)
        self.collision_pipeline = newton.examples.create_collision_pipeline(
            self.model, args, broad_phase="sap", rigid_contact_max=self.model.rigid_contact_max
        )
        self.contacts = self.collision_pipeline.contacts()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.release()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(-4.0, -15.0, 6.0), pitch=-16.0, yaw=75.0)

    def release(self):
        """Hang the chain swung back by the release angle and let go from rest."""
        q = self.model.joint_q.numpy().copy()
        start = int(self.model.joint_q_start.numpy()[self.root_joint])
        swing = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), self.release_angle)
        q[start : start + 4] = [swing[0], swing[1], swing[2], swing[3]]
        self.state_0.joint_q.assign(q)
        self.state_0.joint_qd.zero_()
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

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
        qd = self.state_0.body_qd.numpy()
        self.viewer.log_scalar("ball speed [m/s]", float(np.linalg.norm(qd[self.ball, :3])))
        self.viewer.log_scalar("contacts", int(self.contacts.rigid_contact_count.numpy()[0]))
        self.viewer.end_frame()

    def gui(self, ui):
        ui.text(f"{len(self.building)} building parts, {LINKS}-link chain")
        _, deg = ui.slider_float("Release angle [deg]", math.degrees(self.release_angle), 10.0, 85.0)
        self.release_angle = math.radians(deg)
        q = self.state_0.body_q.numpy()
        ui.text(f"ball at x={q[self.ball, 0]:+.2f} z={q[self.ball, 2]:.2f}")
        ui.text("Reset the example to swing again from the chosen angle")

    def test_final(self):
        assert_finite(self.state_0.body_q, self.state_0.body_qd)
        q = self.state_0.body_q.numpy()
        if np.any(q[:, 2] < -0.05):
            raise ValueError("a body fell through the ground")
        slabs = [self.building[i] for i in range(len(self.building)) if (i + 1) % 11 == 0]
        if q[slabs[-1], 2] > FLOORS * FLOOR_H - 0.5:
            raise ValueError("the wrecking ball did not bring the top slab down")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=480)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
