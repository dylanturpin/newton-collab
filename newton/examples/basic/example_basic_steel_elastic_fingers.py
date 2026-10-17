# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Basic Steel Elastic Fingers
#
# A parallel gripper with soft rubber fingers grips a rigid box and lifts
# it with SolverFeatherPGS. Each finger is a reduced elastic FEM beam
# clamped to a driven prismatic slider and bends where it presses the box.
#
# Command: python -m newton.examples basic_steel_elastic_fingers
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.basic._steel_elastic import box_solid, elastic_link_kwargs


class Example:
    def __init__(self, viewer, args=None):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer
        self.rigid = bool(getattr(args, "rigid", False))

        box_half = 0.025
        finger_half = (0.006, 0.012, 0.055)
        clamp_z = 0.116
        self.box_half = box_half
        self.open_x = box_half + finger_half[0] + 0.025
        self.closed_x = box_half + finger_half[0] - 0.012
        self.lift_height = 0.1

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.num_rigid_contacts_per_world = 4096
        contact = newton.ModelBuilder.ShapeConfig(mu=1.0, gap=0.002, density=0.0)
        visual = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False)
        builder.add_ground_plane(cfg=contact)

        carriage = builder.add_link(
            xform=wp.transform((0.0, 0.0, clamp_z), wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33(np.eye(3) * 1.0e-3),
            label="carriage",
        )
        builder.add_shape_box(
            carriage,
            xform=wp.transform((0.0, 0.0, 0.022), wp.quat_identity()),
            hx=self.open_x + 0.02,
            hy=0.016,
            hz=0.007,
            cfg=visual,
            color=(0.35, 0.37, 0.42),
        )
        drive = {"target_ke": 2.0e4, "target_kd": 4.0e2}
        j_lift = builder.add_joint_prismatic(
            -1,
            carriage,
            axis=(0.0, 0.0, 1.0),
            parent_xform=wp.transform((0.0, 0.0, clamp_z), wp.quat_identity()),
            target_pos=0.0,
            **drive,
        )
        joints = [j_lift]
        self.j_sliders = []

        solid = box_solid(finger_half, (3, 4, 16), young=4.0e6, poisson=0.45, density=1100.0)
        fixed = np.flatnonzero(solid.nodes[:, 2] > finger_half[2] - 1.0e-9)
        self._tip_nodes = np.flatnonzero(solid.nodes[:, 2] < -finger_half[2] + 1.0e-9)
        self.fingers = []
        self._bases = []
        for side in (-1.0, 1.0):
            slider = builder.add_link(
                xform=wp.transform((side * self.open_x, 0.0, clamp_z), wp.quat_identity()),
                mass=0.05,
                inertia=wp.mat33(np.eye(3) * 1.0e-5),
                label=f"slider_{int(side)}",
            )
            builder.add_shape_box(
                slider,
                xform=wp.transform((0.0, 0.0, 0.008), wp.quat_identity()),
                hx=0.011,
                hy=0.016,
                hz=0.008,
                cfg=visual,
                color=(0.62, 0.64, 0.68),
            )
            j_slider = builder.add_joint_prismatic(
                carriage,
                slider,
                axis=(side, 0.0, 0.0),
                target_pos=self.open_x,
                limit_lower=0.0,
                limit_upper=0.2,
                **drive,
            )
            builder.joint_q[builder.joint_q_start[j_slider]] = self.open_x
            joints.append(j_slider)
            self.j_sliders.append(j_slider)

            finger_xform = wp.transform((side * self.open_x, 0.0, clamp_z - finger_half[2]), wp.quat_identity())
            clamp = wp.transform((0.0, 0.0, finger_half[2]), wp.quat_identity())
            color = (0.78, 0.55, 0.3)
            if self.rigid:
                finger = builder.add_link(
                    xform=finger_xform,
                    mass=solid.mass,
                    com=wp.vec3(*solid.com),
                    inertia=wp.mat33(*solid.inertia().ravel()),
                    label=f"finger_{int(side)}",
                )
                builder.add_shape_box(
                    finger, hx=finger_half[0], hy=finger_half[1], hz=finger_half[2], cfg=contact, color=color
                )
                joints.append(
                    builder.add_joint_fixed(slider, finger, parent_xform=wp.transform_identity(), child_xform=clamp)
                )
            else:
                basis = solid.modal_basis(
                    mode_count=8, fixed_nodes=fixed, damping_ratio=0.05, label=f"finger_{int(side)}"
                )
                finger = builder.add_link_elastic(
                    xform=finger_xform,
                    label=f"finger_{int(side)}",
                    parent=slider,
                    parent_xform=wp.transform_identity(),
                    child_xform=clamp,
                    linear_axes=[],
                    angular_axes=[],
                    **elastic_link_kwargs(solid, basis),
                )
                mesh = newton.Mesh(
                    solid.surface_vertices.astype(np.float32), solid.surface_indices.astype(np.int32).ravel()
                )
                builder.add_shape_mesh(finger, mesh=mesh, cfg=contact, color=color)
                joints.append(builder.body_elastic_joint[finger])
                self._bases.append(basis)
            self.fingers.append(finger)
        builder.add_articulation(joints, label="gripper")
        self.j_lift = j_lift

        self.box = builder.add_body(xform=wp.transform((0.0, 0.0, box_half + 0.001), wp.quat_identity()), label="box")
        builder.add_shape_box(
            self.box,
            hx=box_half,
            hy=box_half,
            hz=box_half,
            cfg=newton.ModelBuilder.ShapeConfig(mu=1.0, gap=0.002, density=640.0),
            color=(0.2, 0.45, 0.9),
        )

        self.model = builder.finalize()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        self.solver = newton.solvers.SolverFeatherPGS(self.model, iterations=16)

        q_start = self.model.joint_q_start.numpy()
        self._lift_index = int(q_start[self.j_lift])
        self._slider_index = [int(q_start[j]) for j in self.j_sliders]
        self._targets = self.control.joint_target_q.numpy()
        if not self.rigid:
            self._mode_start = self.model.elastic_mode_q_start.numpy()
            self._mode_count = self.model.elastic_mode_count.numpy()
            self._tip_phi = [b.sample_phi[self._tip_nodes] for b in self._bases]

        self.max_tip_deflection = 0.0
        self.max_box_z = 0.0
        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(0.11, -0.34, 0.19), -10.0, 108.0)

    def _schedule(self, t):
        close = min(max((t - 0.3) / 0.7, 0.0), 1.0)
        lift = min(max((t - 1.4) / 1.2, 0.0), 1.0)
        slider = self.open_x + (self.closed_x - self.open_x) * (0.5 - 0.5 * np.cos(np.pi * close))
        return slider, self.lift_height * (0.5 - 0.5 * np.cos(np.pi * lift))

    def simulate(self):
        for _ in range(self.sim_substeps):
            slider, lift = self._schedule(self.sim_time)
            self._targets[self._lift_index] = lift
            for i in self._slider_index:
                self._targets[i] = slider
            self.control.joint_target_q.assign(self._targets)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt
        self.solver.check_constraint_capacity()

    def step(self):
        self.simulate()
        self.max_box_z = max(self.max_box_z, float(self.state_0.body_q.numpy()[self.box][2]))
        if not self.rigid:
            q = self.state_0.joint_q.numpy()
            for e, phi in enumerate(self._tip_phi):
                modes = q[self._mode_start[e] : self._mode_start[e] + self._mode_count[e]]
                tip = np.einsum("smk,m->sk", phi, modes)
                self.max_tip_deflection = max(self.max_tip_deflection, float(np.abs(tip[:, 0]).max()))

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        # Show the deformed meshes without the elastic centerline diagnostics.
        self.viewer.log_lines("/model/elastic_bodies/centerlines", None, None, None)
        self.viewer.log_points("/model/elastic_bodies/samples", None)
        self.viewer.log_points("/model/elastic_bodies/endpoints", None)
        self.viewer.end_frame()

    def test_final(self):
        if not np.all(np.isfinite(self.state_0.joint_q.numpy())):
            raise AssertionError("non-finite joint state")
        if self.max_box_z < self.box_half + 0.8 * self.lift_height:
            raise AssertionError(f"box was not lifted: max z={self.max_box_z}")
        if not self.rigid and self.max_tip_deflection < 0.003:
            raise AssertionError(f"fingers did not bend: tip deflection={self.max_tip_deflection}")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--rigid", action="store_true", help="Replace the elastic fingers by rigid boxes.")
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
