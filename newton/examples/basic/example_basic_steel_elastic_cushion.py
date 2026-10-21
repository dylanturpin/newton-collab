# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Basic Steel Elastic Cushion
#
# Rigid blocks and spheres of mixed mass drop into a bin and pile up on a
# soft rubber mat stretched across it, simulated with SolverSteel. The
# mat is a reduced elastic body clamped to the world along its rim, so its
# fixed-base FEM modes dish under each impact and sag under the pile.
#
# Command: python -m newton.examples basic_steel_elastic_cushion
#
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.basic._steel_elastic import box_solid, elastic_link_kwargs

# (shape, half size [m], density [kg/m^3], drop position [m], color)
DROPS = (
    ("box", 0.045, 2500.0, (-0.10, -0.08, 0.36), (0.85, 0.33, 0.20)),
    ("sphere", 0.050, 7800.0, (0.10, 0.09, 0.46), (0.72, 0.74, 0.78)),
    ("box", 0.040, 600.0, (0.10, -0.10, 0.58), (0.95, 0.72, 0.22)),
    ("sphere", 0.060, 1200.0, (-0.09, 0.10, 0.70), (0.22, 0.52, 0.85)),
    ("box", 0.050, 1500.0, (0.0, 0.0, 0.84), (0.30, 0.65, 0.40)),
    ("sphere", 0.050, 7800.0, (0.02, -0.03, 0.98), (0.72, 0.74, 0.78)),
    ("box", 0.040, 2500.0, (-0.02, 0.03, 1.12), (0.58, 0.36, 0.72)),
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

        half = (0.28, 0.28, 0.02)
        mat_z = 0.14
        self.mat_top = mat_z + half[2]
        self.bin_half = half[0]
        wall_bottom, wall_top = mat_z - half[2], self.mat_top + 0.11

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.num_rigid_contacts_per_world = 16384
        contact = newton.ModelBuilder.ShapeConfig(mu=0.6, gap=0.004, density=0.0)
        builder.add_ground_plane(cfg=contact)

        walls = []
        for axis in (0, 1):
            for side in (-1.0, 1.0):
                pos = [0.0, 0.0, 0.5 * (wall_bottom + wall_top)]
                pos[axis] = side * (self.bin_half + 0.012)
                hx, hy = (0.012, self.bin_half + 0.024) if axis == 0 else (self.bin_half + 0.024, 0.012)
                xform = wp.transform(wp.vec3(*pos), wp.quat_identity())
                walls.append(
                    builder.add_shape_box(
                        -1,
                        xform=xform,
                        hx=hx,
                        hy=hy,
                        hz=0.5 * (wall_top - wall_bottom),
                        cfg=contact,
                        color=(0.42, 0.45, 0.50),
                    )
                )

        mat_color = (0.95, 0.62, 0.30)
        mat_xform = wp.transform((0.0, 0.0, mat_z), wp.quat_identity())
        if self.rigid:
            self.mat = builder.add_link(xform=mat_xform, mass=1.0, inertia=wp.mat33(np.eye(3) * 1.0e-2), label="mat")
            mat_shape = builder.add_shape_box(
                self.mat, hx=half[0], hy=half[1], hz=half[2], cfg=contact, color=mat_color
            )
            builder.add_articulation([builder.add_joint_fixed(-1, self.mat, parent_xform=mat_xform)], label="mat")
        else:
            solid = box_solid(half, (12, 12, 2), young=1.5e5, poisson=0.3, density=200.0)
            nodes = solid.nodes
            rim = np.flatnonzero((np.abs(nodes[:, 0]) > half[0] - 1.0e-6) | (np.abs(nodes[:, 1]) > half[1] - 1.0e-6))
            top = np.flatnonzero(nodes[:, 2] > half[2] - 1.0e-6)
            basis = solid.modal_basis(mode_count=12, fixed_nodes=rim, damping_ratio=0.3, label="mat")
            self.phi_top_z = np.asarray(basis.sample_phi)[top, :, 2]
            self.mat = builder.add_link_elastic(
                xform=mat_xform,
                label="mat",
                parent=-1,
                parent_xform=mat_xform,
                linear_axes=[],
                angular_axes=[],
                **elastic_link_kwargs(solid, basis),
            )
            mesh = newton.Mesh(
                solid.surface_vertices.astype(np.float32), solid.surface_indices.astype(np.int32).ravel()
            )
            mat_shape = builder.add_shape_mesh(self.mat, mesh=mesh, cfg=contact, color=mat_color)
            builder.add_articulation([builder.body_elastic_joint[self.mat]], label="mat")
        # The clamped rim touches the walls; those contacts carry no motion and would only take solver rows.
        for wall in walls:
            builder.add_shape_collision_filter_pair(wall, mat_shape)

        self.drops = []
        for kind, size, density, pos, color in DROPS:
            cfg = newton.ModelBuilder.ShapeConfig(mu=0.6, gap=0.004, density=density)
            rot = wp.quat_from_axis_angle(wp.normalize(wp.vec3(0.3, 1.0, 0.2)), 0.5 * len(self.drops))
            body = builder.add_body(xform=wp.transform(pos, rot), label=f"{kind}_{len(self.drops)}")
            if kind == "box":
                builder.add_shape_box(body, hx=size, hy=size, hz=size, cfg=cfg, color=color)
            else:
                builder.add_shape_sphere(body, radius=size, cfg=cfg, color=color)
            self.drops.append(body)

        self.model = builder.finalize()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.pipeline.contacts()
        # Kernel shared memory caps the dense rows per world at 128, about 40 contacts on the mat.
        self.solver = newton.solvers.SolverSteel(self.model, iterations=16, articulation_constraint_max=128)

        self.drop_mass = float(self.model.body_mass.numpy()[self.drops].sum())
        self.max_deflection = 0.0
        self.final_deflection = 0.0
        self.max_contacts = 0
        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(0.0, -0.66, 0.98), -52.0, 90.0)

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sim_time += self.sim_dt

    def step(self):
        self.simulate()
        self.solver.check_constraint_capacity()
        self.max_contacts = max(self.max_contacts, int(self.contacts.rigid_contact_count.numpy()[0]))
        if not self.rigid:
            q0 = int(self.model.elastic_mode_q_start.numpy()[0])
            modes = self.state_0.joint_q.numpy()[q0 : q0 + self.phi_top_z.shape[1]]
            self.final_deflection = float(-(self.phi_top_z @ modes).min())
            self.max_deflection = max(self.max_deflection, self.final_deflection)

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
        pos = body_q[self.drops, :3]
        if np.abs(pos[:, :2]).max() > self.bin_half or pos[:, 2].min() < self.mat_top - 0.05:
            raise AssertionError(f"object left the bin or fell through the mat: {pos.round(3).tolist()}")
        speed = np.linalg.norm(self.state_0.body_qd.numpy()[self.drops, :3], axis=1).max()
        if speed > 0.05:
            raise AssertionError(f"objects did not settle: max speed={speed:.3f} m/s")
        if not self.rigid and self.final_deflection < 0.01:
            raise AssertionError(f"mat did not sag under the pile: deflection={self.final_deflection:.4f} m")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--rigid", action="store_true", help="Replace the elastic mat by a rigid slab.")
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
