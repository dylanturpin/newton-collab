# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Basic Steel Elastic Bin
#
# A rigid paddle on a driven prismatic joint sweeps two soft rubber balls
# across a bin floor and squashes them against the far wall, simulated with
# SolverSteel. Each ball is a free reduced elastic body; they travel
# in separate lanes and never touch each other.
#
# Command: python -m newton.examples basic_steel_elastic_bin
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.basic._steel_elastic import box_solid, cube_to_ball, elastic_link_kwargs

# (kind, half size [m], start x [m], lane y [m], color)
OBJECTS = (
    ("ball", 0.060, -0.14, -0.15, (0.90, 0.30, 0.35)),
    ("ball", 0.060, -0.05, 0.15, (0.25, 0.60, 0.90)),
)


class Example:
    def __init__(self, viewer, args=None):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer
        self.rigid = bool(getattr(args, "rigid", False))

        self.wall_x = 0.30
        bin_half_y = 0.30
        wall_height, side_height = 0.16, 0.05
        paddle_half = (0.02, bin_half_y - 0.01, 0.05)
        self.paddle_start = -0.38
        self.paddle_end = self.wall_x - paddle_half[0] - 0.075

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.num_rigid_contacts_per_world = 8192
        contact = newton.ModelBuilder.ShapeConfig(mu=0.5, gap=0.004, density=0.0)
        builder.add_ground_plane(cfg=contact)

        wall_color = (0.42, 0.45, 0.50)
        for pos, half in (
            ((self.wall_x + 0.015, 0.0), (0.015, bin_half_y + 0.03, 0.5 * wall_height)),
            ((-0.05, bin_half_y + 0.015), (0.45, 0.015, 0.5 * side_height)),
            ((-0.05, -bin_half_y - 0.015), (0.45, 0.015, 0.5 * side_height)),
        ):
            xform = wp.transform((pos[0], pos[1], half[2]), wp.quat_identity())
            builder.add_shape_box(-1, xform=xform, hx=half[0], hy=half[1], hz=half[2], cfg=contact, color=wall_color)

        paddle_xform = wp.transform((0.0, 0.0, paddle_half[2] + 0.006), wp.quat_identity())
        start_xform = wp.transform((self.paddle_start, 0.0, paddle_half[2] + 0.006), wp.quat_identity())
        self.paddle = builder.add_link(
            xform=start_xform, mass=2.0, inertia=wp.mat33(np.eye(3) * 1.0e-2), label="paddle"
        )
        builder.add_shape_box(
            self.paddle, hx=paddle_half[0], hy=paddle_half[1], hz=paddle_half[2], cfg=contact, color=(0.30, 0.32, 0.36)
        )
        self.j_paddle = builder.add_joint_prismatic(
            -1,
            self.paddle,
            axis=(1.0, 0.0, 0.0),
            parent_xform=paddle_xform,
            target_pos=self.paddle_start,
            target_ke=4.0e4,
            target_kd=1.5e3,
        )
        builder.joint_q[builder.joint_q_start[self.j_paddle]] = self.paddle_start
        builder.add_articulation([self.j_paddle], label="paddle")

        self.soft = []
        self.blocks = []
        self.surface_phi = []
        self.surface_rest = []
        for kind, size, x, y, color in OBJECTS:
            xform = wp.transform((x, y, size + 0.002), wp.quat_identity())
            if kind == "block":
                body = builder.add_body(xform=xform, label=kind)
                cfg = newton.ModelBuilder.ShapeConfig(mu=0.5, gap=0.004, density=600.0)
                builder.add_shape_box(body, hx=size, hy=size, hz=size, cfg=cfg, color=color)
                self.blocks.append(body)
                continue
            if self.rigid:
                body = builder.add_body(xform=xform, label=kind)
                cfg = newton.ModelBuilder.ShapeConfig(mu=0.5, gap=0.004, density=400.0)
                builder.add_shape_sphere(body, radius=size, cfg=cfg, color=color)
            else:
                solid = box_solid(
                    (size,) * 3,
                    (3, 3, 3),
                    young=3.0e4,
                    poisson=0.35,
                    density=400.0,
                    warp_nodes=cube_to_ball(size, size),
                )
                basis = solid.modal_basis(mode_count=12, discard_mode_count=6, damping_ratio=0.1, label=kind)
                body = builder.add_body_elastic(xform=xform, label=kind, **elastic_link_kwargs(solid, basis))
                mesh = newton.Mesh(
                    solid.surface_vertices.astype(np.float32), solid.surface_indices.astype(np.int32).ravel()
                )
                builder.add_shape_mesh(body, mesh=mesh, cfg=contact, color=color)
                self.surface_phi.append(np.asarray(basis.sample_phi)[solid.surface_nodes])
                self.surface_rest.append(solid.surface_vertices)
            self.soft.append(body)

        self.model = builder.finalize()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.pipeline.contacts()
        # 128 dense rows per world keep the tiled-row PGS kernel within its shared-memory budget.
        self.solver = newton.solvers.SolverSteel(self.model, iterations=16, articulation_constraint_max=128)

        self._paddle_index = int(self.model.joint_q_start.numpy()[self.j_paddle])
        self._targets = self.control.joint_target_q.numpy()
        self.max_squash = np.zeros(len(self.soft))
        self.final_squash = np.zeros(len(self.soft))
        self.max_contacts = 0
        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(-0.06, -0.66, 0.84), -56.0, 90.0)

    def _paddle_target(self, t):
        push = min(max((t - 0.3) / 1.7, 0.0), 1.0)
        back = min(max((t - 2.8) / 1.0, 0.0), 1.0)
        s = (0.5 - 0.5 * np.cos(np.pi * push)) * (0.5 + 0.5 * np.cos(np.pi * back))
        return self.paddle_start + (self.paddle_end - self.paddle_start) * s

    def simulate(self):
        for _ in range(self.sim_substeps):
            self._targets[self._paddle_index] = self._paddle_target(self.sim_time)
            self.control.joint_target_q.assign(self._targets)
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt

    def step(self):
        self.simulate()
        self.solver.check_constraint_capacity()
        self.max_contacts = max(self.max_contacts, int(self.contacts.rigid_contact_count.numpy()[0]))
        if not self.rigid:
            joint_q = self.state_0.joint_q.numpy()
            mode_start = self.model.elastic_mode_q_start.numpy()
            mode_count = self.model.elastic_mode_count.numpy()
            for e, (phi, rest) in enumerate(zip(self.surface_phi, self.surface_rest, strict=True)):
                modes = joint_q[mode_start[e] : mode_start[e] + mode_count[e]]
                deformed = rest + np.einsum("smc,m->sc", phi, modes)
                # Smallest extent over the three body axes, relative to the rest extent.
                extent = (deformed.max(0) - deformed.min(0)) / (rest.max(0) - rest.min(0))
                self.final_squash[e] = 1.0 - extent.min()
                self.max_squash[e] = max(self.max_squash[e], self.final_squash[e])

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        # Show the deformed meshes without the elastic centerline diagnostics.
        self.viewer.log_lines("/model/elastic_bodies/centerlines", None, None, None)
        self.viewer.log_points("/model/elastic_bodies/samples", None)
        self.viewer.log_points("/model/elastic_bodies/endpoints", None)
        self.viewer.end_frame()

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        if not (np.all(np.isfinite(self.state_0.joint_q.numpy())) and np.all(np.isfinite(body_q))):
            raise AssertionError("non-finite state")
        pos = body_q[self.soft + self.blocks, :3]
        if pos[:, 0].max() > self.wall_x or np.abs(pos[:, 1]).max() > 0.3 or pos[:, 2].min() < 0.02:
            raise AssertionError(f"a body left the bin: {pos.round(3).tolist()}")
        if pos[:, 0].min() < 0.1:
            raise AssertionError(f"a body was not swept to the wall: x={pos[:, 0].round(3).tolist()}")
        if not self.rigid:
            if self.max_squash.min() < 0.1:
                raise AssertionError(f"bodies were not squashed: max squash={self.max_squash.round(3).tolist()}")
            if self.final_squash.max() > 0.05:
                raise AssertionError(f"bodies did not recover: final squash={self.final_squash.round(3).tolist()}")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--rigid", action="store_true", help="Replace the soft balls by rigid spheres.")
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
