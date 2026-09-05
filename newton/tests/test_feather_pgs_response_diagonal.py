# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_LOCAL_SOLVE_OWNER_PAIR,
    accumulate_group_diag_worlds,
)
from newton._src.solvers.feather_pgs.solver_feather_pgs import (
    _FeatherPGSExecutionPlan,
    _get_build_independent_sparse_contact_groups_kernel,
    _get_direct_diagonal_inverse_mass_kernel,
    _get_hinv_jt_kernel,
    _get_mark_independent_sparse_contact_candidates_kernel,
    _get_partitioned_inverse_dynamics_kernels,
    _get_pgs_solve_sparse_diagonal_kernel,
)
from newton.solvers import SolverFeatherPGS


def _build_mixed_response_model(
    device, world_count=1, *, dof_count=13, friction=0.0, restitution=0.0, static_plane=False
):
    """Build one serial articulation contacting one free rigid body."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.default_shape_cfg.density = 1000.0
    builder.default_shape_cfg.ke = 1.0e5
    builder.default_shape_cfg.kd = 1.0e3
    builder.default_shape_cfg.mu = friction
    builder.default_shape_cfg.restitution = restitution
    builder.default_shape_cfg.margin = 0.0
    builder.default_shape_cfg.gap = 0.0

    arm = builder.add_link()
    builder.add_shape_box(arm, hx=0.4, hy=0.05, hz=0.02)
    joints = [
        builder.add_joint_revolute(
            parent=-1,
            child=arm,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.4, 0.0, 0.0), wp.quat_identity()),
        )
    ]
    parent = arm
    for index in range(dof_count - 1):
        child = builder.add_link(
            mass=0.05,
            inertia=wp.mat33(np.eye(3, dtype=np.float32) * 1.0e-3),
            lock_inertia=True,
        )
        joints.append(
            builder.add_joint_revolute(
                parent=parent,
                child=child,
                axis=(newton.Axis.X, newton.Axis.Y, newton.Axis.Z)[index % 3],
            )
        )
        parent = child
    builder.add_articulation(joints)
    builder.add_constraint_mimic(joint0=joints[1], joint1=joints[0], coef0=0.0, coef1=1.0)

    box = builder.add_link(xform=wp.transform(wp.vec3(0.7, 0.0, 0.5695), wp.quat_identity()))
    builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.05)
    builder.add_articulation([builder.add_joint_free(parent=-1, child=box)])
    if static_plane:
        builder.add_shape_plane(plane=(0.0, 0.0, 1.0, -0.53))
    if world_count == 1:
        return builder.finalize(device=device)
    replicated = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    replicated.replicate(builder, world_count, spacing=(3.0, 0.0, 0.0))
    return replicated.finalize(device=device)


def _run_mixed_response(
    kernel,
    *,
    warmstart,
    preelimination,
    dof_count=13,
    dense_max_constraints=32,
    inactive_joint_limit_capacity=False,
    friction=0.0,
    restitution=0.0,
    tangential_velocity=0.0,
    static_plane=False,
):
    """Run a short mixed-contact trajectory with one H-inverse implementation."""
    model = _build_mixed_response_model(
        "cuda:0", dof_count=dof_count, friction=friction, restitution=restitution, static_plane=static_plane
    )
    with mock.patch.object(SolverFeatherPGS, "_kernel_overrides", {"hinv_jt_kernel": kernel}):
        solver = SolverFeatherPGS(
            model,
            pgs_mode="matrix_free",
            pgs_warmstart=warmstart,
            enable_bilateral_preelimination=preelimination,
            enable_contact_friction=friction > 0.0,
            enable_joint_limits=inactive_joint_limit_capacity,
            joint_limit_activation_gap=0.0,
            pgs_iterations=8,
            dense_max_constraints=dense_max_constraints,
            mf_max_constraints=32,
        )
    state_in, state_out = model.state(), model.state()
    joint_qd = state_in.joint_qd.numpy()
    free_articulation = int(np.flatnonzero(solver._model_plan.is_free_rigid)[0])
    free_dof_start = int(solver._model_plan.articulation_dof_start[free_articulation])
    joint_qd[free_dof_start] = tangential_velocity
    joint_qd[free_dof_start + 2] = -3.0
    state_in.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)

    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        reduce_contacts=False,
        contact_matching="latest",
    )
    contacts = pipeline.contacts()
    control = model.control()
    samples = []
    for _ in range(4):
        state_in.clear_forces()
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, 1.0 / 240.0)
        constraint_count = int(solver.constraint_count.numpy()[0])
        samples.append(
            (
                constraint_count,
                solver.diag.numpy()[0, :constraint_count].copy(),
                solver.impulses.numpy()[0, :constraint_count].copy(),
                state_out.joint_q.numpy().copy(),
                state_out.joint_qd.numpy().copy(),
                int(solver._local_solve_owner.numpy()[0]),
                solver.row_type.numpy()[0, :constraint_count].copy(),
                int(solver.mf_constraint_count.numpy()[0]),
            )
        )
        state_in, state_out = state_out, state_in
    return solver, samples


class TestFeatherPGSResponseDiagonal(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "sparse diagonal GS requires CUDA")
    def test_sparse_diagonal_gs_matches_scalar_reference(self):
        """Match a scalar PGS reference with coupled limits and friction."""
        device = wp.get_device("cuda:0")
        max_constraints, world_dofs, dense_dofs = 8, 4, 2
        cfm, omega, iterations = 1.0e-6, 1.0, 8
        inverse_mass = np.array((0.5, 0.25, 0.75, 1.0), dtype=np.float32)
        initial_velocity = np.array((-0.4, 0.3, -0.2, 0.1), dtype=np.float32)
        jacobian = np.array(
            ((0.6, -0.4, 0.2, 0.0), (0.1, 0.5, -0.3, 0.4), (-0.2, 0.25, 0.1, -0.35)),
            dtype=np.float32,
        )
        response = jacobian * inverse_mass
        rhs = np.zeros((1, max_constraints), dtype=np.float32)
        rhs[0, :3] = (-0.1, 0.0, 0.0)
        diag = np.zeros_like(rhs)
        diag[0, :3] = np.sum(jacobian * response, axis=1) + cfm
        row_type = np.zeros((1, max_constraints), dtype=np.int32)
        row_type[0, :3] = (PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION, PGS_CONSTRAINT_TYPE_FRICTION)
        row_parent = np.full((1, max_constraints), -1, dtype=np.int32)
        row_parent[0, 1:3] = 0
        row_mu = np.zeros((1, max_constraints), dtype=np.float32)
        row_mu[0, 1:3] = 0.7
        dense_j = np.zeros((1, max_constraints, dense_dofs), dtype=np.float32)
        dense_y = np.zeros_like(dense_j)
        dense_j[0, :3] = jacobian[:, :dense_dofs]
        dense_y[0, :3] = response[:, :dense_dofs]
        sparse_dof = np.full((1, max_constraints, 2), -1, dtype=np.int32)
        sparse_jy = np.zeros((1, max_constraints, 4), dtype=np.float32)
        sparse_dof[0, :3] = (2, 3)
        sparse_jy[0, :3, 0] = jacobian[:, 2]
        sparse_jy[0, :3, 1] = response[:, 2]
        sparse_jy[0, :3, 2] = jacobian[:, 3]
        sparse_jy[0, :3, 3] = response[:, 3]
        limit_active = np.zeros((1, world_dofs), dtype=np.int32)
        limit_active[0, 2] = 1
        limit_lower_rhs = np.zeros((1, world_dofs), dtype=np.float32)
        limit_lower_rhs[0, 2] = -0.15

        expected_velocity = initial_velocity.copy()
        expected_impulses = np.zeros(3, dtype=np.float32)
        expected_limit_lambda = np.float32(0.0)
        for _ in range(iterations):
            old_limit = expected_limit_lambda
            residual = expected_velocity[2] + limit_lower_rhs[0, 2]
            expected_limit_lambda = np.maximum(0.0, old_limit - omega * residual / (inverse_mass[2] + cfm))
            expected_velocity[2] += inverse_mass[2] * (expected_limit_lambda - old_limit)
            for row in range(3):
                old_impulse = expected_impulses[row]
                new_impulse = old_impulse - omega * (jacobian[row] @ expected_velocity + rhs[0, row]) / diag[0, row]
                if row == 0:
                    new_impulse = max(new_impulse, 0.0)
                else:
                    radius = max(row_mu[0, row] * expected_impulses[0], 0.0)
                    sibling = 2 if row == 1 else 1
                    if radius <= 0.0:
                        new_impulse = 0.0
                    else:
                        expected_impulses[row] = new_impulse
                        sibling_old = expected_impulses[sibling]
                        magnitude = np.sqrt(new_impulse * new_impulse + sibling_old * sibling_old)
                        if magnitude > radius:
                            scale = radius / magnitude
                            new_impulse *= scale
                            sibling_new = sibling_old * scale
                            expected_impulses[sibling] = sibling_new
                            expected_velocity += response[sibling] * (sibling_new - sibling_old)
                expected_impulses[row] = new_impulse
                expected_velocity += response[row] * (new_impulse - old_impulse)

        impulses = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)
        lower_lambda = wp.zeros((1, world_dofs), dtype=wp.float32, device=device)
        upper_lambda = wp.zeros_like(lower_lambda)
        velocity = wp.array(initial_velocity, dtype=wp.float32, device=device)
        kernel = _get_pgs_solve_sparse_diagonal_kernel(
            max_constraints, world_dofs, dense_dofs, str(device.arch), contact_triples=True
        )
        wp.launch_tiled(
            kernel,
            dim=[1],
            inputs=[
                wp.array((3,), dtype=wp.int32, device=device),
                wp.array(np.arange(world_dofs, dtype=np.int32)[None, :], device=device),
                wp.array(rhs, device=device),
                wp.array(diag, device=device),
                impulses,
                wp.array(row_type, device=device),
                wp.array(row_parent, device=device),
                wp.array(row_mu, device=device),
                wp.zeros((1, 2), dtype=wp.int32, device=device),
                wp.zeros(1, dtype=wp.int32, device=device),
                wp.empty((1, world_dofs), dtype=wp.int32, device=device),
                wp.zeros(1, dtype=wp.int32, device=device),
                wp.empty((1, (max_constraints + 2) // 3), dtype=wp.int32, device=device),
                wp.array((0,), dtype=wp.int32, device=device),
                wp.array((0,), dtype=wp.int32, device=device),
                wp.array(dense_j, device=device),
                wp.array(dense_y, device=device),
                wp.array(sparse_dof, device=device),
                wp.array(sparse_jy, device=device),
                wp.array(limit_active, device=device),
                wp.array(limit_lower_rhs, device=device),
                wp.zeros((1, world_dofs), dtype=wp.float32, device=device),
                wp.array(inverse_mass, device=device),
                cfm,
                iterations,
                omega,
                0,
                0,
            ],
            outputs=[lower_lambda, upper_lambda, velocity],
            block_dim=32,
            device=device,
        )

        np.testing.assert_allclose(velocity.numpy(), expected_velocity, rtol=2.0e-5, atol=2.0e-6)
        np.testing.assert_allclose(impulses.numpy()[0, :3], expected_impulses, rtol=2.0e-5, atol=2.0e-6)
        np.testing.assert_allclose(lower_lambda.numpy()[0, 2], expected_limit_lambda, rtol=2.0e-5, atol=2.0e-6)
        self.assertEqual(float(upper_lambda.numpy()[0, 2]), 0.0)

    @unittest.skipUnless(wp.is_cuda_available(), "speculative contact batches require CUDA")
    def test_speculative_sparse_contact_batches_match_serial_reference(self):
        """Skip exact no-op prefixes without changing later serial contact updates."""
        device = wp.get_device("cuda:0")
        max_constraints, world_dofs, dense_dofs = 12, 8, 6
        cfm, omega, iterations = 1.0e-6, 1.0, 2
        inverse_mass = np.array((0.5, 0.25, 0.75, 1.0, 0.4, 0.6, 0.75, 0.8), dtype=np.float32)
        initial_velocity = np.array((0.3, -0.2, 0.1, 0.4, -0.1, 0.2, 0.5, 0.25), dtype=np.float32)
        jacobian = np.zeros((max_constraints, world_dofs), dtype=np.float32)
        jacobian[0, 6] = 1.0
        jacobian[3, 7] = 1.0
        jacobian[6, (0, 6)] = (-0.2, -1.0)
        jacobian[9, (0, 6)] = (0.3, 1.0)
        response = jacobian * inverse_mass
        rhs = np.zeros((1, max_constraints), dtype=np.float32)
        rhs[0, 6] = -0.1
        rhs[0, 9] = -0.2
        diag = np.zeros_like(rhs)
        normal_rows = np.array((0, 3, 6, 9), dtype=np.int32)
        diag[0, normal_rows] = np.sum(jacobian[normal_rows] * response[normal_rows], axis=1) + cfm
        row_type = np.full((1, max_constraints), PGS_CONSTRAINT_TYPE_FRICTION, dtype=np.int32)
        row_type[0, normal_rows] = PGS_CONSTRAINT_TYPE_CONTACT
        row_parent = np.full((1, max_constraints), -1, dtype=np.int32)
        for normal in normal_rows:
            row_parent[0, normal + 1 : normal + 3] = normal
        row_mu = np.zeros((1, max_constraints), dtype=np.float32)
        dense_j = jacobian[:, :dense_dofs][None, ...]
        dense_y = response[:, :dense_dofs][None, ...]
        sparse_dof = np.empty((1, max_constraints, 2), dtype=np.int32)
        sparse_dof[:, :, 0] = 6
        sparse_dof[:, :, 1] = 7
        sparse_jy = np.zeros((1, max_constraints, 4), dtype=np.float32)
        sparse_jy[0, :, 0] = jacobian[:, 6]
        sparse_jy[0, :, 1] = response[:, 6]
        sparse_jy[0, :, 2] = jacobian[:, 7]
        sparse_jy[0, :, 3] = response[:, 7]

        expected_velocity = initial_velocity.copy()
        expected_impulses = np.zeros(max_constraints, dtype=np.float32)
        for _ in range(iterations):
            for row in normal_rows:
                old_impulse = expected_impulses[row]
                residual = jacobian[row] @ expected_velocity + rhs[0, row]
                new_impulse = max(old_impulse - omega * residual / diag[0, row], 0.0)
                expected_impulses[row] = new_impulse
                expected_velocity += response[row] * (new_impulse - old_impulse)

        impulses = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)
        velocity = wp.array(initial_velocity, dtype=wp.float32, device=device)
        kernel = _get_pgs_solve_sparse_diagonal_kernel(
            max_constraints,
            world_dofs,
            dense_dofs,
            str(device.arch),
            contact_triples=True,
            speculative_contact_batches=True,
        )
        wp.launch_tiled(
            kernel,
            dim=[1],
            inputs=[
                wp.array((max_constraints,), dtype=wp.int32, device=device),
                wp.array(np.arange(world_dofs, dtype=np.int32)[None, :], device=device),
                wp.array(rhs, device=device),
                wp.array(diag, device=device),
                impulses,
                wp.array(row_type, device=device),
                wp.array(row_parent, device=device),
                wp.array(row_mu, device=device),
                wp.zeros((1, 2), dtype=wp.int32, device=device),
                wp.zeros(1, dtype=wp.int32, device=device),
                wp.empty((1, world_dofs), dtype=wp.int32, device=device),
                wp.array((len(normal_rows),), dtype=wp.int32, device=device),
                wp.array(normal_rows[None, :], device=device),
                wp.array((0,), dtype=wp.int32, device=device),
                wp.array((0,), dtype=wp.int32, device=device),
                wp.array(dense_j, device=device),
                wp.array(dense_y, device=device),
                wp.array(sparse_dof, device=device),
                wp.array(sparse_jy, device=device),
                wp.zeros((1, world_dofs), dtype=wp.int32, device=device),
                wp.zeros((1, world_dofs), dtype=wp.float32, device=device),
                wp.zeros((1, world_dofs), dtype=wp.float32, device=device),
                wp.array(inverse_mass, device=device),
                cfm,
                iterations,
                omega,
                0,
                0,
            ],
            outputs=[
                wp.zeros((1, world_dofs), dtype=wp.float32, device=device),
                wp.zeros((1, world_dofs), dtype=wp.float32, device=device),
                velocity,
            ],
            block_dim=32,
            device=device,
        )

        np.testing.assert_allclose(velocity.numpy(), expected_velocity, rtol=2.0e-5, atol=2.0e-6)
        np.testing.assert_allclose(impulses.numpy()[0], expected_impulses, rtol=2.0e-5, atol=2.0e-6)
        np.testing.assert_array_equal(impulses.numpy()[0, :6], np.zeros(6, dtype=np.float32))

    @unittest.skipUnless(wp.is_cuda_available(), "direct diagonal projection requires CUDA")
    def test_direct_diagonal_inverse_mass_matches_compact_inertia_terms(self):
        """Project a one-body branch directly from its compact inertia representation."""
        device = wp.get_device("cuda:0")
        mass = np.float32(2.5)
        com = np.array((0.2, -0.1, 0.3), dtype=np.float32)
        inertia = np.array(((0.8, 0.1, -0.05), (0.1, 1.1, 0.02), (-0.05, 0.02, 1.4)), dtype=np.float32)
        motion = np.array((0.3, -0.2, 0.4, 0.5, 0.1, -0.6), dtype=np.float32)
        armature = np.float32(0.25)
        drive_stiffness = np.float32(0.75)
        terms = np.concatenate((com, inertia.reshape(-1)))[None, :]
        inverse_mass = wp.zeros(1, dtype=wp.float32, device=device)
        kernel = _get_direct_diagonal_inverse_mass_kernel(1, str(device.arch))
        wp.launch(
            kernel,
            dim=1,
            inputs=[
                wp.array((0, 1), dtype=wp.int32, device=device),
                wp.array((0,), dtype=wp.int32, device=device),
                wp.array((1,), dtype=wp.int32, device=device),
                wp.array((0,), dtype=wp.int32, device=device),
                wp.array((wp.spatial_vector(*motion),), dtype=wp.spatial_vector, device=device),
                wp.array((mass,), dtype=wp.float32, device=device),
                wp.array(terms, dtype=wp.float32, device=device),
                wp.array((0,), dtype=wp.int32, device=device),
                wp.array((0,), dtype=wp.int32, device=device),
                wp.array(((armature,),), dtype=wp.float32, device=device),
                wp.array((0,), dtype=wp.int32, device=device),
                wp.array((drive_stiffness,), dtype=wp.float32, device=device),
            ],
            outputs=[inverse_mass],
            device=device,
        )
        wp.synchronize_device(device)

        linear = motion[:3]
        angular = motion[3:]
        response = np.concatenate(
            (
                mass * (linear - np.cross(com, angular)),
                mass * np.cross(com, linear) + inertia @ angular,
            )
        )
        expected = np.float32(1.0) / (np.dot(motion, response) + armature + drive_stiffness)
        np.testing.assert_allclose(inverse_mass.numpy()[0], expected, rtol=2.0e-6, atol=0.0)

    @unittest.skipUnless(wp.is_cuda_available(), "partitioned inverse dynamics requires CUDA")
    def test_direct_branch_inverse_dynamics_matches_articulation_path(self):
        """A one-body branch must produce the same torque through either schedule."""
        device = wp.get_device("cuda:0")
        spatial = wp.spatial_vector(0.4, -0.2, 0.6, 0.1, 0.3, -0.5)
        external = wp.spatial_vector(-0.3, 0.5, 0.2, -0.4, 0.1, 0.7)
        articulation_start = wp.array((0, 1), dtype=wp.int32, device=device)
        articulation_end = wp.array((1,), dtype=wp.int32, device=device)
        joint_type = wp.array((int(newton.JointType.PRISMATIC),), dtype=wp.int32, device=device)
        joint_parent = wp.array((-1,), dtype=wp.int32, device=device)
        joint_child = wp.array((0,), dtype=wp.int32, device=device)
        joint_articulation = wp.array((0,), dtype=wp.int32, device=device)
        starts = wp.array((0, 1), dtype=wp.int32, device=device)
        dof_dim = wp.array(((1, 0),), dtype=wp.int32, device=device)
        joint_f = wp.array((0.2,), dtype=wp.float32, device=device)
        joint_q = wp.array((0.15,), dtype=wp.float32, device=device)
        joint_qd = wp.array((-0.25,), dtype=wp.float32, device=device)
        stiffness = wp.array((1.4,), dtype=wp.float32, device=device)
        spring_ref = wp.array((0.05,), dtype=wp.float32, device=device)
        damping = wp.array((0.3,), dtype=wp.float32, device=device)
        joint_S_s = wp.array((spatial,), dtype=wp.spatial_vector, device=device)
        body_force = wp.array((spatial,), dtype=wp.spatial_vector, device=device)
        external_force = wp.array((external,), dtype=wp.spatial_vector, device=device)
        body_flags = wp.zeros(1, dtype=wp.int32, device=device)
        body_q = wp.array((wp.transform_identity(),), dtype=wp.transform, device=device)
        body_com = wp.array((wp.vec3(0.2, -0.1, 0.3),), dtype=wp.vec3, device=device)
        origin = wp.array((wp.vec3(),), dtype=wp.vec3, device=device)
        selected_tau = wp.array((0.7,), dtype=wp.float32, device=device)
        direct_tau = wp.clone(selected_tau)
        body_ft = wp.zeros(1, dtype=wp.spatial_vector, device=device)
        selected_kernel, direct_kernel = _get_partitioned_inverse_dynamics_kernels(1, str(device.arch))
        common = [
            articulation_start,
            articulation_end,
            joint_type,
            joint_parent,
            joint_child,
            joint_articulation,
            starts,
            starts,
            dof_dim,
            joint_f,
            joint_q,
            joint_qd,
            stiffness,
            spring_ref,
            damping,
            joint_S_s,
            body_force,
            external_force,
            body_flags,
            body_q,
            body_com,
            origin,
        ]
        wp.launch(
            selected_kernel,
            dim=1,
            inputs=[wp.array((0,), dtype=wp.int32, device=device), *common, 1],
            outputs=[body_ft, selected_tau],
            device=device,
        )
        wp.launch(
            direct_kernel,
            dim=1,
            inputs=[
                wp.array((0,), dtype=wp.int32, device=device),
                wp.array((0,), dtype=wp.int32, device=device),
                articulation_start,
                joint_type,
                joint_child,
                starts,
                starts,
                dof_dim,
                joint_f,
                joint_q,
                joint_qd,
                stiffness,
                spring_ref,
                damping,
                joint_S_s,
                body_force,
                external_force,
                body_flags,
                body_q,
                body_com,
                origin,
                1,
            ],
            outputs=[direct_tau],
            device=device,
        )
        wp.synchronize_device(device)
        np.testing.assert_array_equal(direct_tau.numpy(), selected_tau.numpy())

    @unittest.skipUnless(wp.is_cuda_available(), "independent contact groups require CUDA")
    def test_independent_sparse_contact_groups_exclude_coupled_coordinates(self):
        """Keep scalar contacts serial when a coupled contact shares their coordinate."""
        device = wp.get_device("cuda:0")
        max_constraints, max_world_dofs, contact_count = 12, 4, 4
        sparse_response_dofs = 108
        sparse_row_dof_np = np.full((1, max_constraints, 2), -1, dtype=np.int32)
        sparse_row_dof_np[0, 0:3, 0] = 2
        sparse_row_dof_np[0, 3:6, 0] = 3
        sparse_row_dof_np[0, 6:9, 0] = 3
        sparse_row_dof_np[0, 9:12, 0] = 2
        sparse_row_dof = wp.array(sparse_row_dof_np, device=device)
        group_count = wp.zeros(1, dtype=wp.int32, device=device)
        group_heads = wp.empty((1, max_world_dofs), dtype=wp.int32, device=device)
        serial_count = wp.zeros(1, dtype=wp.int32, device=device)
        serial_normals = wp.empty((1, (max_constraints + 2) // 3), dtype=wp.int32, device=device)

        marker = _get_mark_independent_sparse_contact_candidates_kernel(sparse_response_dofs, str(device.arch))
        wp.launch(
            marker,
            dim=contact_count,
            inputs=[
                wp.array((contact_count,), dtype=wp.int32, device=device),
                contact_count,
                wp.zeros(contact_count, dtype=wp.int32, device=device),
                wp.array((0, 3, 6, 9), dtype=wp.int32, device=device),
                wp.zeros(contact_count, dtype=wp.int32, device=device),
                wp.array((-1, 1, -1, -1), dtype=wp.int32, device=device),
                wp.zeros(contact_count, dtype=wp.int32, device=device),
                wp.full(contact_count, 3, dtype=wp.int32, device=device),
                wp.array((sparse_response_dofs, 6), dtype=wp.int32, device=device),
            ],
            outputs=[sparse_row_dof],
            device=device,
        )
        schedule = _get_build_independent_sparse_contact_groups_kernel(
            max_constraints, max_world_dofs, str(device.arch), build_serial_contacts=True
        )
        wp.launch(
            schedule,
            dim=32,
            inputs=[
                wp.array((max_constraints,), dtype=wp.int32, device=device),
                wp.zeros((1, 2), dtype=wp.int32, device=device),
            ],
            outputs=[sparse_row_dof, group_count, group_heads, serial_count, serial_normals],
            device=device,
        )
        wp.synchronize_device(device)

        self.assertEqual(int(group_count.numpy()[0]), 1)
        self.assertEqual(int(group_heads.numpy()[0, 0]), 0)
        self.assertEqual(int(serial_count.numpy()[0]), 2)
        np.testing.assert_array_equal(serial_normals.numpy()[0, :2], np.array((3, 6), dtype=np.int32))
        normal_links = sparse_row_dof.numpy()[0, ::3, 1]
        np.testing.assert_array_equal(normal_links, np.array((-12, -1, -1, -2), dtype=np.int32))

    def test_mixed_world_diagonal_map_includes_free_rigid_response(self):
        """Include free rigid articulations in every forced-tiled diagonal group."""
        solver = SolverFeatherPGS(_build_mixed_response_model("cpu", world_count=2), dense_max_constraints=32)
        self.assertEqual(set(solver.size_groups), {6, 13})

        response_dof_count = solver._model_plan.response_dof_count
        articulation_world = solver._model_plan.articulation_world
        free_mask = solver._model_plan.is_free_rigid != 0
        for size in solver.size_groups:
            size_arts = np.flatnonzero(response_dof_count == size)
            response_order = np.asarray(
                sorted(size_arts, key=lambda art: (int(articulation_world[art]), int(art))), dtype=np.int32
            )
            propagation_order = response_order[~free_mask[response_order]]
            response_counts = np.bincount(articulation_world[response_order], minlength=solver.world_count)
            propagation_counts = np.bincount(articulation_world[propagation_order], minlength=solver.world_count)
            response_starts = np.pad(np.cumsum(response_counts), (1, 0)).astype(np.int32)
            propagation_starts = np.pad(np.cumsum(propagation_counts), (1, 0)).astype(np.int32)

            np.testing.assert_array_equal(solver.world_response_group_art_start[size].numpy(), response_starts)
            np.testing.assert_array_equal(solver.world_response_group_to_art[size].numpy(), response_order)
            np.testing.assert_array_equal(solver.world_group_art_start[size].numpy(), propagation_starts)
            np.testing.assert_array_equal(solver.world_group_to_art[size].numpy(), propagation_order)

        free_articulations = np.flatnonzero(free_mask).astype(np.int32)
        np.testing.assert_array_equal(solver.world_group_art_start[6].numpy(), np.array((0, 0, 0), dtype=np.int32))
        np.testing.assert_array_equal(solver.world_group_to_art[6].numpy(), np.empty(0, dtype=np.int32))
        np.testing.assert_array_equal(
            solver.world_response_group_art_start[6].numpy(), np.array((0, 1, 2), dtype=np.int32)
        )
        np.testing.assert_array_equal(solver.world_response_group_to_art[6].numpy(), free_articulations)

        forced_tiled = _FeatherPGSExecutionPlan.build(
            solver.size_groups,
            max_constraints=solver.dense_max_constraints,
            max_shared_memory=101376,
            cholesky_kernel="auto",
            hinv_jt_kernel="tiled",
            small_dof_threshold=12,
            tile_threads=64,
        )
        self.assertEqual(forced_tiled.hinv_jt_tiled_sizes, frozenset((6, 13)))

    @unittest.skipUnless(wp.is_cuda_available(), "matrix-free response diagonal parity requires CUDA")
    def test_forced_tiled_mixed_world_matches_par_row_trajectory(self):
        """Match diagonal, impulses, and state when every response group is tiled."""
        for warmstart in (False, True):
            for preelimination in (False, True):
                with self.subTest(warmstart=warmstart, preelimination=preelimination):
                    reference_solver, reference = _run_mixed_response(
                        "par_row", warmstart=warmstart, preelimination=preelimination
                    )
                    tiled_solver, tiled = _run_mixed_response(
                        "tiled", warmstart=warmstart, preelimination=preelimination
                    )

                    self.assertEqual(reference_solver._hinv_jt_diag_sizes, frozenset())
                    self.assertEqual(reference_solver._preelim_active, preelimination)
                    self.assertEqual(tiled_solver._preelim_active, preelimination)
                    expected_diag_sizes = frozenset() if preelimination else frozenset((6, 13))
                    self.assertEqual(tiled_solver._hinv_jt_diag_sizes, expected_diag_sizes)
                    self.assertEqual(len(reference), len(tiled))
                    self.assertGreater(reference[0][0], 0, "mixed scene generated no dense constraint rows")
                    for step, (expected, actual) in enumerate(zip(reference, tiled, strict=True)):
                        self.assertEqual(actual[0], expected[0], f"constraint count differed at step {step}")
                        for label, expected_value, actual_value in zip(
                            ("diagonal", "impulses", "joint_q", "joint_qd"),
                            expected[1:5],
                            actual[1:5],
                            strict=True,
                        ):
                            np.testing.assert_allclose(
                                actual_value,
                                expected_value,
                                rtol=5.0e-4,
                                atol=2.0e-6,
                                err_msg=f"{label} differed at step {step}",
                            )

    @unittest.skipUnless(wp.is_cuda_available(), "paired response ownership requires CUDA")
    def test_paired_response_matches_general_23_dof_trajectory(self):
        """Match the general response when one warp owns a robot/free-body pair."""
        run_kwargs = {
            "warmstart": False,
            "preelimination": False,
            "dof_count": 23,
            "dense_max_constraints": 96,
            "inactive_joint_limit_capacity": True,
            "friction": 0.7,
            "restitution": 0.3,
            "tangential_velocity": 2.0,
        }
        reference_solver, reference = _run_mixed_response("par_row", **run_kwargs)
        paired_solver, paired = _run_mixed_response("auto", **run_kwargs)

        self.assertIsNone(reference_solver._paired_response_primary_size)
        self.assertEqual(paired_solver._paired_response_primary_size, 23)
        self.assertEqual(paired_solver._paired_response_secondary_size, 6)
        self.assertIsNotNone(paired_solver._paired_response_kernel)
        self.assertIsNotNone(paired_solver._paired_factor_solve_kernel)
        self.assertTrue(paired_solver._paired_factor_coordinates)
        self.assertTrue(paired_solver._factor_coordinate_contact_triples)
        self.assertGreater(reference[0][0], 0, "mixed scene generated no dense constraint rows")
        for step, (expected, actual) in enumerate(zip(reference, paired, strict=True)):
            self.assertEqual(actual[0], expected[0], f"constraint count differed at step {step}")
            np.testing.assert_array_equal(actual[6], expected[6], err_msg=f"row types differed at step {step}")
            for label, expected_value, actual_value in zip(
                ("diagonal", "impulses", "joint_q", "joint_qd"), expected[1:5], actual[1:5], strict=True
            ):
                np.testing.assert_allclose(
                    actual_value,
                    expected_value,
                    rtol=5.0e-4,
                    atol=1.0e-5,
                    err_msg=f"{label} differed at step {step}",
                )

    @unittest.skipUnless(wp.is_cuda_available(), "paired response fallback requires CUDA")
    def test_paired_response_falls_back_for_matrix_free_rows(self):
        """Keep mixed dense/matrix-free worlds in physical coordinates."""
        run_kwargs = {
            "warmstart": False,
            "preelimination": False,
            "dof_count": 23,
            "dense_max_constraints": 96,
            "friction": 0.7,
            "restitution": 0.3,
            "tangential_velocity": 2.0,
            "static_plane": True,
        }
        reference_solver, reference = _run_mixed_response("par_row", **run_kwargs)
        paired_solver, paired = _run_mixed_response("auto", **run_kwargs)

        self.assertIsNone(reference_solver._paired_response_primary_size)
        self.assertTrue(paired_solver._paired_factor_coordinates)
        self.assertTrue(any(sample[7] > 0 for sample in paired), "static contact generated no matrix-free rows")
        for step, (expected, actual) in enumerate(zip(reference, paired, strict=True)):
            self.assertEqual(actual[0], expected[0], f"constraint count differed at step {step}")
            np.testing.assert_array_equal(actual[6], expected[6], err_msg=f"row types differed at step {step}")
            for label, expected_value, actual_value in zip(
                ("diagonal", "impulses", "joint_q", "joint_qd"), expected[1:5], actual[1:5], strict=True
            ):
                np.testing.assert_allclose(
                    actual_value,
                    expected_value,
                    rtol=5.0e-4,
                    atol=1.0e-5,
                    err_msg=f"{label} differed at step {step}",
                )

    @unittest.skipUnless(wp.is_cuda_available(), "articulation-local mixed-world parity requires CUDA")
    def test_local_internal_mixed_world_matches_general_response(self):
        """Match the general response when an articulation contacts a free body."""
        general_solver, general = _run_mixed_response(
            "tiled", warmstart=False, preelimination=False, inactive_joint_limit_capacity=True
        )
        local_solver, local = _run_mixed_response(
            "par_row", warmstart=False, preelimination=False, inactive_joint_limit_capacity=True
        )

        self.assertFalse(general_solver._local_internal_fast_path)
        self.assertTrue(local_solver._local_internal_fast_path)
        self.assertEqual(len(general), len(local))
        self.assertGreater(general[0][0], 0, "mixed scene generated no dense constraint rows")
        self.assertTrue(
            any(sample[5] == PGS_LOCAL_SOLVE_OWNER_PAIR for sample in local),
            "mixed contact never selected the paired local owner",
        )

        free_articulation = int(np.flatnonzero(general_solver._model_plan.is_free_rigid)[0])
        free_dof_start = int(general_solver._model_plan.articulation_dof_start[free_articulation])
        self.assertGreater(
            abs(float(general[0][4][free_dof_start + 2]) + 3.0),
            1.0e-4,
            "general response did not couple the dense contact to the free body",
        )

        for step, (expected, actual) in enumerate(zip(general, local, strict=True)):
            self.assertEqual(actual[0], expected[0], f"constraint count differed at step {step}")
            for label, expected_value, actual_value in zip(
                ("diagonal", "impulses", "joint_q", "joint_qd"), expected[1:5], actual[1:5], strict=True
            ):
                np.testing.assert_allclose(
                    actual_value,
                    expected_value,
                    rtol=5.0e-4,
                    atol=2.0e-6,
                    err_msg=f"{label} differed at step {step}",
                )

    @unittest.skipUnless(wp.is_cuda_available(), "articulation-local mixed-world parity requires CUDA")
    def test_local_pair_matches_general_friction_and_restitution(self):
        """Preserve paired friction projection and restitution response."""
        run_kwargs = {
            "warmstart": False,
            "preelimination": False,
            "inactive_joint_limit_capacity": True,
            "friction": 0.7,
            "restitution": 0.3,
            "tangential_velocity": 2.0,
        }
        general_solver, general = _run_mixed_response("tiled", **run_kwargs)
        local_solver, local = _run_mixed_response("par_row", **run_kwargs)

        self.assertFalse(general_solver._local_internal_fast_path)
        self.assertTrue(local_solver._local_internal_fast_path)
        pair_samples = [sample for sample in local if sample[5] == PGS_LOCAL_SOLVE_OWNER_PAIR]
        self.assertTrue(pair_samples, "mixed friction contact never selected the paired local owner")
        self.assertTrue(
            any(
                np.any(np.abs(sample[2][sample[6] == PGS_CONSTRAINT_TYPE_FRICTION]) > 1.0e-5) for sample in pair_samples
            ),
            "paired solve produced no nonzero friction impulse",
        )

        for step, (expected, actual) in enumerate(zip(general, local, strict=True)):
            self.assertEqual(actual[0], expected[0], f"constraint count differed at step {step}")
            np.testing.assert_array_equal(actual[6], expected[6], err_msg=f"row types differed at step {step}")
            for label, expected_value, actual_value in zip(
                ("diagonal", "impulses", "joint_q", "joint_qd"), expected[1:5], actual[1:5], strict=True
            ):
                np.testing.assert_allclose(
                    actual_value,
                    expected_value,
                    rtol=5.0e-4,
                    atol=2.0e-6,
                    err_msg=f"{label} differed at step {step}",
                )

    @unittest.skipUnless(wp.is_cuda_available(), "tiled H-inverse response requires CUDA")
    def test_tiled_response_diagonal_matches_dense_reference(self):
        device = wp.get_device("cuda:0")
        rng = np.random.default_rng(41)
        num_dofs = 23
        max_constraints = 16
        num_articulations = 4
        world_constraint_count_np = np.array((13, 7), dtype=np.int32)
        articulation_world_np = np.array((0, 0, 1, 1), dtype=np.int32)
        articulation_world_dof_offset_np = np.array((0, num_dofs, 0, num_dofs), dtype=np.int32)

        factors = rng.normal(size=(num_articulations, num_dofs, num_dofs)).astype(np.float32)
        mass = factors @ np.transpose(factors, (0, 2, 1))
        mass += 2.0 * np.eye(num_dofs, dtype=np.float32)[None, :, :]
        cholesky = wp.array(np.linalg.cholesky(mass).astype(np.float32), device=device)

        jacobian_np = np.zeros((num_articulations, max_constraints, num_dofs), dtype=np.float32)
        for articulation, world in enumerate(articulation_world_np):
            count = int(world_constraint_count_np[world])
            jacobian_np[articulation, :count] = rng.normal(size=(count, num_dofs))
        jacobian = wp.array(jacobian_np, device=device)

        group_to_art = wp.array(np.arange(num_articulations, dtype=np.int32), device=device)
        art_to_world = wp.array(articulation_world_np, device=device)
        articulation_world_dof_offset = wp.array(articulation_world_dof_offset_np, device=device)
        constraint_count = wp.array(world_constraint_count_np, device=device)
        response = wp.zeros_like(jacobian)
        world_jacobian = wp.zeros((2, max_constraints, 2 * num_dofs), dtype=wp.float32, device=device)
        world_response = wp.zeros_like(world_jacobian)
        group_diag = wp.zeros((num_articulations, max_constraints), dtype=wp.float32, device=device)

        kernel = _get_hinv_jt_kernel(
            num_dofs,
            max_constraints,
            str(device.arch),
            constraint_chunk_size=8,
            write_world=True,
            write_group=False,
            compute_diag=True,
        )
        wp.launch_tiled(
            kernel,
            dim=(num_articulations, 2),
            inputs=[
                cholesky,
                jacobian,
                group_to_art,
                art_to_world,
                articulation_world_dof_offset,
                constraint_count,
            ],
            outputs=[response, world_jacobian, world_response, group_diag],
            block_dim=64,
            device=device,
        )

        world_diag = wp.zeros((2, max_constraints), dtype=wp.float32, device=device)
        wp.launch(
            accumulate_group_diag_worlds,
            dim=2 * max_constraints,
            inputs=[
                group_diag,
                wp.array((0, 2, 4), dtype=wp.int32, device=device),
                group_to_art,
                group_to_art,
                constraint_count,
                max_constraints,
            ],
            outputs=[world_diag],
            device=device,
        )
        wp.synchronize_device(device)

        response_ref = np.zeros_like(jacobian_np)
        group_diag_ref = np.zeros((num_articulations, max_constraints), dtype=np.float32)
        world_jacobian_ref = np.zeros((2, max_constraints, 2 * num_dofs), dtype=np.float32)
        world_response_ref = np.zeros_like(world_jacobian_ref)
        for articulation, world in enumerate(articulation_world_np):
            count = int(world_constraint_count_np[world])
            response_ref[articulation, :count] = np.linalg.solve(
                mass[articulation], jacobian_np[articulation, :count].T
            ).T
            group_diag_ref[articulation, :count] = np.sum(
                jacobian_np[articulation, :count] * response_ref[articulation, :count], axis=1
            )
            dof_offset = articulation_world_dof_offset_np[articulation]
            world_jacobian_ref[world, :count, dof_offset : dof_offset + num_dofs] = jacobian_np[articulation, :count]
            world_response_ref[world, :count, dof_offset : dof_offset + num_dofs] = response_ref[articulation, :count]
        world_diag_ref = np.stack((group_diag_ref[:2].sum(axis=0), group_diag_ref[2:].sum(axis=0)))

        np.testing.assert_array_equal(response.numpy(), np.zeros_like(response_ref))
        np.testing.assert_allclose(world_jacobian.numpy(), world_jacobian_ref, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(world_response.numpy(), world_response_ref, rtol=2.0e-5, atol=2.0e-6)
        np.testing.assert_allclose(group_diag.numpy(), group_diag_ref, rtol=2.0e-5, atol=2.0e-6)
        np.testing.assert_allclose(world_diag.numpy(), world_diag_ref, rtol=2.0e-5, atol=2.0e-6)


if __name__ == "__main__":
    unittest.main()
