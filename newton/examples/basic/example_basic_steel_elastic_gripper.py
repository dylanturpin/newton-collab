# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Basic Steel Elastic Gripper
#
# A position-driven parallel gripper squeezes a soft rubber ball and lifts
# it off the ground with SolverFeatherPGS. The ball is a free reduced
# elastic body whose free-free FEM modes deform at the pad contacts.
#
# Command: python -m newton.examples basic_steel_elastic_gripper
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.basic._steel_elastic import box_solid, cube_to_ball, elastic_link_kwargs


class Example:
    def __init__(self, viewer, args=None):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer
        self.rigid = bool(getattr(args, "rigid", False))

        radius = 0.04
        self.radius = radius
        pad_half = (0.008, 0.035, 0.035)
        self.open_x = radius + pad_half[0] + 0.02
        self.closed_x = radius + pad_half[0] - 0.012
        self.lift_height = 0.12

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.num_rigid_contacts_per_world = 4096
        contact = newton.ModelBuilder.ShapeConfig(mu=1.0, gap=0.002, density=0.0)
        builder.add_ground_plane(cfg=contact)

        # Carriage lifted by a prismatic joint; two pads on prismatic joints along x.
        carriage = builder.add_link(
            xform=wp.transform((0.0, 0.0, radius), wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33(np.eye(3) * 1.0e-3),
            label="carriage",
        )
        visual = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False, has_particle_collision=False)
        builder.add_shape_box(
            carriage,
            xform=wp.transform((0.0, 0.0, 0.05), wp.quat_identity()),
            hx=self.open_x + 0.02,
            hy=0.012,
            hz=0.008,
            cfg=visual,
        )
        drive = {"target_ke": 2.0e4, "target_kd": 4.0e2}
        j_lift = builder.add_joint_prismatic(
            -1,
            carriage,
            axis=(0.0, 0.0, 1.0),
            parent_xform=wp.transform((0.0, 0.0, radius), wp.quat_identity()),
            target_pos=0.0,
            **drive,
        )
        self.pads = []
        joints = [j_lift]
        for side in (-1.0, 1.0):
            pad = builder.add_link(
                xform=wp.transform((side * self.open_x, 0.0, radius), wp.quat_identity()),
                mass=0.2,
                inertia=wp.mat33(np.eye(3) * 1.0e-4),
                label=f"pad_{int(side)}",
            )
            builder.add_shape_box(pad, hx=pad_half[0], hy=pad_half[1], hz=pad_half[2], cfg=contact)
            j_pad = builder.add_joint_prismatic(
                carriage, pad, axis=(side, 0.0, 0.0), target_pos=self.open_x, limit_lower=0.0, limit_upper=0.2, **drive
            )
            builder.joint_q[builder.joint_q_start[j_pad]] = self.open_x
            joints.append(j_pad)
            self.pads.append(pad)
        builder.add_articulation(joints, label="gripper")
        self.j_lift, self.j_pads = j_lift, joints[1:]

        ball_xform = wp.transform((0.0, 0.0, radius + 0.001), wp.quat_identity())
        if self.rigid:
            self.ball = builder.add_body(xform=ball_xform, label="ball")
            builder.add_shape_sphere(
                self.ball, radius=radius, cfg=newton.ModelBuilder.ShapeConfig(mu=1.0, gap=0.002, density=300.0)
            )
        else:
            solid = box_solid(
                (radius,) * 3,
                (6, 6, 6),
                young=2.0e4,
                poisson=0.35,
                density=300.0,
                warp_nodes=cube_to_ball(radius, radius),
            )
            basis = solid.modal_basis(mode_count=12, discard_mode_count=6, damping_ratio=0.05, label="ball")
            self.ball = builder.add_body_elastic(xform=ball_xform, label="ball", **elastic_link_kwargs(solid, basis))
            solid.add_render_mesh(builder, self.ball, contact)

        self.model = builder.finalize()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        self.solver = newton.solvers.SolverFeatherPGS(self.model, iterations=16)

        target_start = self.model.joint_q_start.numpy()
        self._lift_index = int(target_start[self.j_lift])
        self._pad_index = [int(target_start[j]) for j in self.j_pads]
        self._targets = self.control.joint_target_q.numpy()

        self.max_mode = 0.0
        self.max_ball_z = 0.0
        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(0.0, -0.36, 0.2), -14.0, 90.0)

    def _schedule(self, t):
        close = min(max((t - 0.2) / 0.6, 0.0), 1.0)
        lift = min(max((t - 1.2) / 1.0, 0.0), 1.0)
        pad = self.open_x + (self.closed_x - self.open_x) * (0.5 - 0.5 * np.cos(np.pi * close))
        return pad, self.lift_height * (0.5 - 0.5 * np.cos(np.pi * lift))

    def simulate(self):
        for _ in range(self.sim_substeps):
            pad, lift = self._schedule(self.sim_time)
            self._targets[self._lift_index] = lift
            for i in self._pad_index:
                self._targets[i] = pad
            self.control.joint_target_q.assign(self._targets)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt

    def step(self):
        self.simulate()
        self.solver.check_constraint_capacity()
        self.max_ball_z = max(self.max_ball_z, float(self.state_0.body_q.numpy()[self.ball][2]))
        if not self.rigid:
            q0 = int(self.model.elastic_mode_q_start.numpy()[0])
            modes = self.state_0.joint_q.numpy()[q0 : q0 + int(self.model.elastic_mode_count.numpy()[0])]
            self.max_mode = max(self.max_mode, float(np.abs(modes).max()))

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        q = self.state_0.joint_q.numpy()
        if not np.all(np.isfinite(q)):
            raise AssertionError("non-finite joint state")
        if self.max_ball_z < self.radius + 0.8 * self.lift_height:
            raise AssertionError(f"ball was not lifted: max z={self.max_ball_z}")
        if not self.rigid and self.max_mode <= 0.0:
            raise AssertionError("ball did not deform")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--rigid", action="store_true", help="Replace the elastic ball by a rigid sphere.")
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
