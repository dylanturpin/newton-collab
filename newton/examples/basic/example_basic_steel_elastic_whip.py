# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Basic Steel Elastic Whip
#
# A position-driven two-link arm swings a long soft whip with
# SolverFeatherPGS. The whip is a reduced elastic FEM beam clamped to the
# forearm: it lags the swing, whips through when the arm stops, and knocks
# a block off a post.
#
# Command: python -m newton.examples basic_steel_elastic_whip
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

        shoulder_z = 0.45
        upper_len, fore_len = 0.25, 0.2
        whip_half = (0.25, 0.004, 0.015)
        self.swing_start, self.swing_end = np.radians([-120.0, -10.0]), np.radians([60.0, 0.0])
        q1, q2 = self.swing_start

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.num_rigid_contacts_per_world = 4096
        contact = newton.ModelBuilder.ShapeConfig(mu=0.6, gap=0.003, density=0.0)
        visual = newton.ModelBuilder.ShapeConfig(density=0.0, has_shape_collision=False)
        builder.add_ground_plane(cfg=contact)
        builder.add_shape_cylinder(
            -1,
            xform=wp.transform((0.0, 0.0, 0.5 * shoulder_z - 0.01), wp.quat_identity()),
            radius=0.03,
            half_height=0.5 * shoulder_z - 0.01,
            cfg=visual,
            color=(0.3, 0.32, 0.36),
        )

        def yaw(angle):
            return wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(angle))

        def link_box(body, length, half_y, half_z, color):
            builder.add_shape_box(
                body,
                xform=wp.transform((0.5 * length, 0.0, 0.0), wp.quat_identity()),
                hx=0.5 * length,
                hy=half_y,
                hz=half_z,
                cfg=visual,
                color=color,
            )

        elbow = np.array([upper_len * np.cos(q1), upper_len * np.sin(q1), shoulder_z])
        wrist = elbow + fore_len * np.array([np.cos(q1 + q2), np.sin(q1 + q2), 0.0])
        upper = builder.add_link(
            xform=wp.transform((0.0, 0.0, shoulder_z), yaw(q1)),
            mass=0.4,
            com=wp.vec3(0.5 * upper_len, 0.0, 0.0),
            inertia=wp.mat33(np.eye(3) * 2.0e-3),
            label="upper_arm",
        )
        link_box(upper, upper_len, 0.02, 0.02, (0.55, 0.6, 0.68))
        fore = builder.add_link(
            xform=wp.transform(tuple(elbow), yaw(q1 + q2)),
            mass=0.25,
            com=wp.vec3(0.5 * fore_len, 0.0, 0.0),
            inertia=wp.mat33(np.eye(3) * 1.0e-3),
            label="forearm",
        )
        link_box(fore, fore_len, 0.016, 0.016, (0.7, 0.74, 0.8))
        drive = {"target_ke": 1.0e4, "target_kd": 2.0e2}
        j_shoulder = builder.add_joint_revolute(
            -1,
            upper,
            axis=(0.0, 0.0, 1.0),
            target_pos=float(q1),
            parent_xform=wp.transform((0.0, 0.0, shoulder_z), wp.quat_identity()),
            **drive,
        )
        j_elbow = builder.add_joint_revolute(
            upper,
            fore,
            axis=(0.0, 0.0, 1.0),
            target_pos=float(q2),
            parent_xform=wp.transform((upper_len, 0.0, 0.0), wp.quat_identity()),
            **drive,
        )
        builder.joint_q[builder.joint_q_start[j_shoulder]] = float(q1)
        builder.joint_q[builder.joint_q_start[j_elbow]] = float(q2)
        joints = [j_shoulder, j_elbow]

        solid = box_solid(whip_half, (30, 2, 4), young=1.2e7, poisson=0.45, density=1100.0)
        self._tip_nodes = np.flatnonzero(solid.nodes[:, 0] > whip_half[0] - 1.0e-9)
        self._tip_rest = solid.nodes[self._tip_nodes].mean(0)
        whip_xform = wp.transform(
            tuple(wrist + whip_half[0] * np.array([np.cos(q1 + q2), np.sin(q1 + q2), 0.0])), yaw(q1 + q2)
        )
        clamp = wp.transform((-whip_half[0], 0.0, 0.0), wp.quat_identity())
        wrist_frame = wp.transform((fore_len, 0.0, 0.0), wp.quat_identity())
        color = (0.78, 0.55, 0.3)
        if self.rigid:
            self.whip = builder.add_link(
                xform=whip_xform,
                mass=solid.mass,
                com=wp.vec3(*solid.com),
                inertia=wp.mat33(*solid.inertia().ravel()),
                label="whip",
            )
            builder.add_shape_box(
                self.whip, hx=whip_half[0], hy=whip_half[1], hz=whip_half[2], cfg=contact, color=color
            )
            joints.append(builder.add_joint_fixed(fore, self.whip, parent_xform=wrist_frame, child_xform=clamp))
        else:
            fixed = np.flatnonzero(solid.nodes[:, 0] < -whip_half[0] + 1.0e-9)
            basis = solid.modal_basis(mode_count=10, fixed_nodes=fixed, damping_ratio=0.03, label="whip")
            # Start at the static gravity sag so the whip does not bounce at t=0.
            sag = (basis.mode_coupling_linear @ np.array([0.0, 0.0, -9.81])) / basis.mode_stiffness
            self.whip = builder.add_link_elastic(
                xform=whip_xform,
                label="whip",
                parent=fore,
                parent_xform=wrist_frame,
                child_xform=clamp,
                linear_axes=[],
                angular_axes=[],
                mode_q=sag,
                **elastic_link_kwargs(solid, basis),
            )
            mesh = newton.Mesh(
                solid.surface_vertices.astype(np.float32), solid.surface_indices.astype(np.int32).ravel()
            )
            builder.add_shape_mesh(self.whip, mesh=mesh, cfg=contact, color=color)
            joints.append(builder.body_elastic_joint[self.whip])
            self._tip_phi = basis.sample_phi[self._tip_nodes].mean(0)
            self._sag = sag
        builder.add_articulation(joints, label="arm")
        self.j_shoulder, self.j_elbow = j_shoulder, j_elbow

        target_pos = np.array([0.66, 0.52, 0.0])
        post_top = shoulder_z - 0.07
        builder.add_shape_cylinder(
            -1,
            xform=wp.transform(tuple(target_pos + np.array([0.0, 0.0, 0.5 * post_top])), wp.quat_identity()),
            radius=0.02,
            half_height=0.5 * post_top,
            cfg=contact,
            color=(0.3, 0.32, 0.36),
        )
        self.block_half = 0.03
        self.block = builder.add_body(
            xform=wp.transform(
                tuple(target_pos + np.array([0.0, 0.0, post_top + self.block_half])), wp.quat_identity()
            ),
            label="block",
        )
        builder.add_shape_box(
            self.block,
            hx=self.block_half,
            hy=self.block_half,
            hz=self.block_half,
            cfg=newton.ModelBuilder.ShapeConfig(mu=0.6, gap=0.003, density=150.0),
            color=(0.98, 0.78, 0.2),
        )
        self._block_start = np.array(target_pos + np.array([0.0, 0.0, post_top + self.block_half]))

        self.model = builder.finalize()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()
        self.solver = newton.solvers.SolverFeatherPGS(self.model, iterations=16)

        q_start = self.model.joint_q_start.numpy()
        self._arm_index = [int(q_start[j_shoulder]), int(q_start[j_elbow])]
        self._targets = self.control.joint_target_q.numpy()
        if not self.rigid:
            self._mode_start = int(self.model.elastic_mode_q_start.numpy()[0])
            self._mode_count = int(self.model.elastic_mode_count.numpy()[0])

        self.max_tip_deflection = 0.0
        self.max_tracking_error = 0.0
        self.max_block_displacement = 0.0
        self.tip_path = []
        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(0.1, -1.9, 1.5), -30.0, 90.0)

    def _schedule(self, t):
        s = min(max((t - 0.3) / 1.2, 0.0), 1.0)
        shoulder = s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)
        e = min(max((t - 0.8) / 0.7, 0.0), 1.0)
        elbow = e * e * e * (10.0 - 15.0 * e + 6.0 * e * e)
        start, end = self.swing_start, self.swing_end
        return start + (end - start) * np.array([shoulder, elbow])

    def simulate(self):
        for _ in range(self.sim_substeps):
            targets = self._schedule(self.sim_time)
            for i, target in zip(self._arm_index, targets, strict=True):
                self._targets[i] = target
            self.control.joint_target_q.assign(self._targets)
            self.model.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt
        self.solver.check_constraint_capacity()

    def step(self):
        self.simulate()
        q = self.state_0.joint_q.numpy()
        error = np.abs(q[self._arm_index] - self._schedule(self.sim_time)).max()
        self.max_tracking_error = max(self.max_tracking_error, float(error))
        body_q = self.state_0.body_q.numpy()
        self.max_block_displacement = max(
            self.max_block_displacement, float(np.linalg.norm(body_q[self.block][:3] - self._block_start))
        )
        tip_local = self._tip_rest.copy()
        if not self.rigid:
            modes = q[self._mode_start : self._mode_start + self._mode_count]
            offset = self._tip_phi.T @ (modes - self._sag)
            self.max_tip_deflection = max(self.max_tip_deflection, float(np.linalg.norm(offset)))
            tip_local += self._tip_phi.T @ modes
        whip = body_q[self.whip]
        tip = np.array(wp.transform_point(wp.transform(whip[:3], whip[3:]), wp.vec3(*tip_local)))
        self.tip_path.append(tip)

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
        if self.max_tracking_error > 0.15:
            raise AssertionError(f"arm did not track its targets: max error={self.max_tracking_error} rad")
        if not self.rigid and self.max_tip_deflection < 0.05:
            raise AssertionError(f"whip did not bend: tip deflection={self.max_tip_deflection}")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--rigid", action="store_true", help="Replace the elastic whip by a rigid bar.")
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
