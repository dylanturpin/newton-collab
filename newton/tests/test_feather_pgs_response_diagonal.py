# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
import newton._src.solvers.feather_pgs.solver_feather_pgs as feather_pgs_module
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    PGS_LOCAL_SOLVE_OWNER_PAIR,
    accumulate_group_diag_worlds,
    prepare_fused_diagonal_joint_limits,
)
from newton._src.solvers.feather_pgs.solver_feather_pgs import _FeatherPGSExecutionPlan, _get_hinv_jt_kernel
from newton.solvers import SolverFeatherPGS


def _build_mixed_response_model(
    device,
    world_count=1,
    *,
    dof_count=13,
    internal_constraint_count=1,
    friction=0.0,
    restitution=0.0,
    free_body=True,
):
    """Build one articulation contacting a free body or static geometry."""
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
    for index in range(internal_constraint_count):
        builder.add_constraint_mimic(
            joint0=joints[(index + 1) % len(joints)],
            joint1=joints[index % len(joints)],
            coef0=0.0,
            coef1=1.0,
        )

    box_xform = wp.transform(wp.vec3(0.7, 0.0, 0.5695), wp.quat_identity())
    if free_body:
        box = builder.add_link(xform=box_xform)
        builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.05)
        builder.add_articulation([builder.add_joint_free(parent=-1, child=box)])
    else:
        builder.add_shape_box(body=-1, xform=box_xform, hx=0.1, hy=0.1, hz=0.05)
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
    inactive_joint_limit_capacity=False,
    friction=0.0,
    restitution=0.0,
    tangential_velocity=0.0,
    dof_count=13,
    internal_constraint_count=1,
    dense_max_constraints=32,
    use_parallel_streams=True,
    pgs_iterations=8,
    free_body=True,
    contact_friction_gap_threshold=float("inf"),
    contact_friction_anchor_limit=0,
):
    """Run a short mixed-contact trajectory with one H-inverse implementation."""
    model = _build_mixed_response_model(
        "cuda:0",
        dof_count=dof_count,
        internal_constraint_count=internal_constraint_count,
        friction=friction,
        restitution=restitution,
        free_body=free_body,
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
            pgs_iterations=pgs_iterations,
            dense_max_constraints=dense_max_constraints,
            mf_max_constraints=32,
            use_parallel_streams=use_parallel_streams,
            contact_friction_gap_threshold=contact_friction_gap_threshold,
            contact_friction_anchor_limit=contact_friction_anchor_limit,
        )
    state_in, state_out = model.state(), model.state()
    joint_qd = state_in.joint_qd.numpy()
    if free_body:
        free_articulation = int(np.flatnonzero(solver._model_plan.is_free_rigid)[0])
        free_dof_start = int(solver._model_plan.articulation_dof_start[free_articulation])
        joint_qd[free_dof_start] = tangential_velocity
        joint_qd[free_dof_start + 2] = -3.0
    else:
        joint_qd[0] = tangential_velocity
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
            )
        )
        state_in, state_out = state_out, state_in
    return solver, samples


class TestFeatherPGSResponseDiagonal(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "fused diagonal limits require CUDA")
    def test_fused_diagonal_limits_match_dense_rows(self):
        """Match ordinary lower/upper rows when limits precede a coupled contact."""
        device = wp.get_device("cuda:0")
        max_constraints = 8
        max_world_dofs = 4
        inverse_mass_np = np.array((0.5, 0.25, 0.75, 1.0), dtype=np.float32)
        initial_velocity_np = np.array((-0.4, 0.3, -0.2, 0.1), dtype=np.float32)
        contact_j = np.array((0.6, -0.4, 0.2, 0.0), dtype=np.float32)
        contact_y = contact_j * inverse_mass_np
        cfm = 1.0e-6

        world_dof_indices = wp.array(np.arange(max_world_dofs, dtype=np.int32)[None, :], device=device)
        active_sides = wp.zeros((1, max_world_dofs), dtype=wp.int32, device=device)
        lower_rhs = wp.zeros((1, max_world_dofs), dtype=wp.float32, device=device)
        upper_rhs = wp.zeros((1, max_world_dofs), dtype=wp.float32, device=device)
        wp.launch(
            prepare_fused_diagonal_joint_limits,
            dim=max_world_dofs,
            inputs=[
                world_dof_indices,
                max_world_dofs,
                wp.array((1, 1, 0, 0), dtype=wp.int32, device=device),
                wp.array((0, 1, 2, 3), dtype=wp.int32, device=device),
                wp.array((0.0, -1.0, -1.0, -1.0), dtype=wp.float32, device=device),
                wp.array((1.0, 1.0, 1.0, 1.0), dtype=wp.float32, device=device),
                wp.array((-0.2, 1.2, 0.0, 0.0), dtype=wp.float32, device=device),
                0.1,
                0.05,
                0.01,
            ],
            outputs=[active_sides, lower_rhs, upper_rhs],
            device=device,
        )

        def launch_solve(
            *,
            fused: bool,
            sparse: bool = False,
            packed: bool = False,
            contact_triples: bool = False,
            specialized: bool = False,
            speculative: bool = False,
            contact_rhs: float = -0.1,
        ):
            if sparse and not fused:
                self.fail("sparse response requires fused diagonal limits")
            if packed and not sparse:
                self.fail("packed worlds require sparse response")
            if contact_triples and (not sparse or packed):
                self.fail("contact-triple coverage requires the one-world sparse response")
            if specialized and not contact_triples:
                self.fail("the specialized kernel requires contact triples")
            if speculative and not specialized:
                self.fail("speculative batches require the specialized kernel")
            row_count = 4 if contact_triples else (1 if fused else 3)
            rhs_np = np.zeros((1, max_constraints), dtype=np.float32)
            diag_np = np.zeros_like(rhs_np)
            row_type_np = np.zeros((1, max_constraints), dtype=np.int32)
            row_parent_np = np.full((1, max_constraints), -1, dtype=np.int32)
            row_mu_np = np.zeros((1, max_constraints), dtype=np.float32)
            jacobian_np = np.zeros((1, max_constraints, max_world_dofs), dtype=np.float32)
            response_np = np.zeros_like(jacobian_np)
            contact_row = 1 if contact_triples else (0 if fused else 2)
            if contact_triples:
                rhs_np[0, 0] = -0.15
                diag_np[0, 0] = inverse_mass_np[2] + cfm
                row_type_np[0, 0] = PGS_CONSTRAINT_TYPE_JOINT_LIMIT
                jacobian_np[0, 0, 2] = 1.0
                response_np[0, 0, 2] = inverse_mass_np[2]
            elif not fused:
                rhs_np[0, :2] = -1.0
                diag_np[0, :2] = inverse_mass_np[:2] + cfm
                row_type_np[0, :2] = PGS_CONSTRAINT_TYPE_JOINT_LIMIT
                jacobian_np[0, 0, 0] = 1.0
                response_np[0, 0, 0] = inverse_mass_np[0]
                jacobian_np[0, 1, 1] = -1.0
                response_np[0, 1, 1] = -inverse_mass_np[1]
            rhs_np[0, contact_row] = contact_rhs
            diag_np[0, contact_row] = float(contact_j @ contact_y) + cfm
            row_type_np[0, contact_row] = PGS_CONSTRAINT_TYPE_CONTACT
            jacobian_np[0, contact_row] = contact_j
            response_np[0, contact_row] = contact_y
            if contact_triples:
                tangent_j = np.array(((-0.3, 0.2, 0.5, -0.1), (0.1, 0.35, -0.2, 0.45)), dtype=np.float32)
                tangent_y = tangent_j * inverse_mass_np
                rhs_np[0, contact_row + 1 : contact_row + 3] = (0.08, -0.04)
                diag_np[0, contact_row + 1 : contact_row + 3] = np.sum(tangent_j * tangent_y, axis=1) + cfm
                row_type_np[0, contact_row + 1 : contact_row + 3] = PGS_CONSTRAINT_TYPE_FRICTION
                row_parent_np[0, contact_row + 1 : contact_row + 3] = contact_row
                row_mu_np[0, contact_row + 1 : contact_row + 3] = 0.6
                jacobian_np[0, contact_row + 1 : contact_row + 3] = tangent_j
                response_np[0, contact_row + 1 : contact_row + 3] = tangent_y

            float_rows = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)
            row_parent = wp.array(row_parent_np, device=device)
            row_mu = wp.array(row_mu_np, device=device)
            sparse_row_dof_np = np.full((1, max_constraints, 2), -1, dtype=np.int32)
            sparse_row_jy_np = np.zeros((1, max_constraints, 4), dtype=np.float32)
            if sparse:
                sparse_rows = range(contact_row, row_count) if contact_triples else (contact_row,)
                for row in sparse_rows:
                    sparse_row_dof_np[0, row] = (0, 1)
                    sparse_row_jy_np[0, row] = (
                        jacobian_np[0, row, 0],
                        response_np[0, row, 0],
                        jacobian_np[0, row, 1],
                        response_np[0, row, 1],
                    )
            dense_jacobian_np = np.zeros((1, max_constraints, 2), dtype=np.float32)
            dense_response_np = np.zeros_like(dense_jacobian_np)
            dense_jacobian_np[0, :row_count] = jacobian_np[0, :row_count, 2:]
            dense_response_np[0, :row_count] = response_np[0, :row_count, 2:]
            velocity = wp.array(initial_velocity_np, device=device)
            impulses = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)
            sparse_row_dof = wp.array(sparse_row_dof_np, device=device)
            fused_lower_lambda = wp.zeros_like(lower_rhs)
            fused_upper_lambda = wp.zeros_like(upper_rhs)
            sparse_limit_touched_count = wp.zeros(1, dtype=wp.int32, device=device)
            sparse_limit_touched_dof = wp.empty((1, max_world_dofs), dtype=wp.int32, device=device)
            sparse_limit_initial_changed = wp.zeros(1, dtype=wp.int32, device=device)
            if packed:
                prepare_kernel = feather_pgs_module._get_prepare_sparse_diagonal_limits_kernel(
                    max_constraints, max_world_dofs, str(device.arch)
                )
                wp.launch_tiled(
                    prepare_kernel,
                    dim=[1],
                    inputs=[
                        1,
                        wp.array((row_count,), dtype=wp.int32, device=device),
                        world_dof_indices,
                        sparse_row_dof,
                        active_sides,
                        lower_rhs,
                        upper_rhs,
                        wp.array(inverse_mass_np, device=device),
                        cfm,
                        8,
                        1.0,
                    ],
                    outputs=[
                        fused_lower_lambda,
                        fused_upper_lambda,
                        sparse_limit_touched_count,
                        sparse_limit_touched_dof,
                        sparse_limit_initial_changed,
                        velocity,
                    ],
                    block_dim=128,
                    device=device,
                )
            kernel = feather_pgs_module._get_pgs_solve_mf_gs_kernel(
                max_constraints,
                1,
                max_world_dofs,
                str(device.arch),
                has_drive_rows=False,
                has_dense_velocity_limit_rows=False,
                dense_only_mf_free=True,
                dense_only_interleaved=True,
                fuse_diagonal_joint_limits=fused,
                sparse_diagonal_dense_dofs=2 if sparse else 0,
                sparse_contact_triples=specialized,
                speculative_contact_batches=speculative,
                worlds_per_block=4 if packed else 1,
            )
            bucket_counts_np = np.zeros(feather_pgs_module._SPARSE_DIAGONAL_ROW_BUCKET_COUNT, dtype=np.int32)
            bucket_offsets_np = np.zeros(feather_pgs_module._SPARSE_DIAGONAL_ROW_BUCKET_COUNT + 1, dtype=np.int32)
            if packed:
                bucket_counts_np[0] = 1
                bucket_offsets_np[1:] = 1
            wp.launch_tiled(
                kernel,
                dim=[1],
                inputs=[
                    wp.zeros(1, dtype=wp.int32, device=device),
                    wp.zeros(1, dtype=wp.int32, device=device),
                    1,
                    0,
                    1,
                    wp.array(bucket_counts_np, device=device),
                    wp.array(bucket_offsets_np, device=device),
                    wp.zeros(
                        (feather_pgs_module._SPARSE_DIAGONAL_ROW_BUCKET_COUNT, 1),
                        dtype=wp.int32,
                        device=device,
                    ),
                    wp.array((row_count,), dtype=wp.int32, device=device),
                    wp.array(((contact_row, contact_row),), dtype=wp.int32, device=device)
                    if contact_triples
                    else wp.zeros((1, 2), dtype=wp.int32, device=device),
                    wp.zeros(1, dtype=wp.int32, device=device),
                    world_dof_indices,
                    wp.zeros((1, max_world_dofs), dtype=wp.int32, device=device),
                    wp.array(rhs_np, device=device),
                    wp.array(diag_np, device=device),
                    float_rows,
                    impulses,
                    wp.array(dense_jacobian_np if sparse else jacobian_np, device=device),
                    wp.array(dense_response_np if sparse else response_np, device=device),
                    wp.array(row_type_np, device=device),
                    row_parent,
                    row_mu,
                    float_rows,
                    float_rows,
                    float_rows,
                    float_rows,
                    float_rows,
                    active_sides if fused else wp.zeros_like(active_sides),
                    lower_rhs if fused else wp.zeros_like(lower_rhs),
                    upper_rhs if fused else wp.zeros_like(upper_rhs),
                    fused_lower_lambda,
                    fused_upper_lambda,
                    wp.array(inverse_mass_np, device=device),
                    cfm,
                    sparse_limit_touched_count,
                    sparse_limit_touched_dof,
                    sparse_limit_initial_changed,
                    wp.array((2 if sparse else 0,), dtype=wp.int32, device=device),
                    wp.zeros(1, dtype=wp.int32, device=device),
                    sparse_row_dof,
                    wp.array(sparse_row_jy_np, device=device),
                    wp.zeros(1, dtype=wp.int32, device=device),
                    wp.zeros(1, dtype=wp.int32, device=device),
                    wp.zeros((1, 4), dtype=wp.int32, device=device),
                    wp.zeros((1, 1), dtype=wp.float32, device=device),
                    wp.zeros((1, 1, 6), dtype=wp.float32, device=device),
                    wp.zeros((1, 1, 6), dtype=wp.float32, device=device),
                    wp.zeros((1, 1, 6), dtype=wp.float32, device=device),
                    wp.zeros((1, 1, 6), dtype=wp.float32, device=device),
                    wp.zeros((1, 1), dtype=wp.float32, device=device),
                    float_rows,
                    8,
                    1.0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
                outputs=[velocity],
                block_dim=32,
                device=device,
            )
            wp.synchronize_device(device)
            solved_impulses = impulses.numpy()[0]
            return velocity.numpy(), solved_impulses.copy() if contact_triples else solved_impulses[contact_row]

        np.testing.assert_array_equal(active_sides.numpy(), np.array(((1, 2, 0, 0),), dtype=np.int32))
        np.testing.assert_allclose(lower_rhs.numpy()[0, :2], (-1.0, 0.0), rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(upper_rhs.numpy()[0, :2], (0.0, -1.0), rtol=0.0, atol=1.0e-6)
        expected_velocity, expected_contact_impulse = launch_solve(fused=False)
        actual_velocity, actual_contact_impulse = launch_solve(fused=True)
        np.testing.assert_allclose(actual_velocity, expected_velocity, rtol=0.0, atol=2.0e-6)
        self.assertAlmostEqual(float(actual_contact_impulse), float(expected_contact_impulse), places=6)
        sparse_velocity, sparse_contact_impulse = launch_solve(fused=True, sparse=True)
        np.testing.assert_allclose(sparse_velocity, expected_velocity, rtol=0.0, atol=2.0e-6)
        self.assertAlmostEqual(float(sparse_contact_impulse), float(expected_contact_impulse), places=6)
        packed_velocity, packed_contact_impulse = launch_solve(fused=True, sparse=True, packed=True)
        np.testing.assert_allclose(packed_velocity, expected_velocity, rtol=0.0, atol=2.0e-6)
        self.assertAlmostEqual(float(packed_contact_impulse), float(expected_contact_impulse), places=6)
        generic_velocity, generic_impulses = launch_solve(fused=True, sparse=True, contact_triples=True)
        triple_velocity, triple_impulses = launch_solve(fused=True, sparse=True, contact_triples=True, specialized=True)
        np.testing.assert_array_equal(triple_velocity, generic_velocity)
        np.testing.assert_array_equal(triple_impulses, generic_impulses)
        speculative_velocity, speculative_impulses = launch_solve(
            fused=True, sparse=True, contact_triples=True, specialized=True, speculative=True
        )
        np.testing.assert_array_equal(speculative_velocity, generic_velocity)
        np.testing.assert_array_equal(speculative_impulses, generic_impulses)
        inactive_velocity, inactive_impulses = launch_solve(
            fused=True, sparse=True, contact_triples=True, contact_rhs=10.0
        )
        speculative_inactive_velocity, speculative_inactive_impulses = launch_solve(
            fused=True,
            sparse=True,
            contact_triples=True,
            specialized=True,
            speculative=True,
            contact_rhs=10.0,
        )
        np.testing.assert_array_equal(speculative_inactive_velocity, inactive_velocity)
        np.testing.assert_array_equal(speculative_inactive_impulses, inactive_impulses)

    @unittest.skipUnless(wp.is_cuda_available(), "independent contact groups require CUDA")
    def test_independent_sparse_contact_groups_exclude_coupled_coordinates(self):
        """Link scalar rows by coordinate while retaining rows that share a coupled contact."""
        device = wp.get_device("cuda:0")
        max_constraints = 12
        max_world_dofs = 4
        contact_count = 4
        sparse_response_dofs = 108
        sparse_row_dof_np = np.full((1, max_constraints, 2), -1, dtype=np.int32)
        sparse_row_dof_np[0, 0:3, 0] = 2
        sparse_row_dof_np[0, 3:6, 0] = 3
        sparse_row_dof_np[0, 6:9, 0] = 3
        sparse_row_dof_np[0, 9:12, 0] = 2
        sparse_row_dof = wp.array(sparse_row_dof_np, device=device)
        row_parent = wp.full((1, max_constraints), -1, dtype=wp.int32, device=device)
        group_count = wp.zeros(1, dtype=wp.int32, device=device)
        group_heads = wp.empty((1, max_world_dofs), dtype=wp.int32, device=device)
        serial_contact_head = wp.full(1, -1, dtype=wp.int32, device=device)

        marker = feather_pgs_module._get_mark_independent_sparse_contact_candidates_kernel(
            sparse_response_dofs, str(device.arch)
        )
        builder = feather_pgs_module._get_build_independent_sparse_contact_groups_kernel(
            max_constraints, max_world_dofs, str(device.arch)
        )
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
        wp.launch(
            builder,
            dim=32,
            inputs=[
                wp.array((max_constraints,), dtype=wp.int32, device=device),
                wp.zeros((1, 2), dtype=wp.int32, device=device),
            ],
            outputs=[sparse_row_dof, row_parent, group_count, group_heads, serial_contact_head],
            device=device,
        )
        wp.synchronize_device(device)

        self.assertEqual(int(group_count.numpy()[0]), 1)
        self.assertEqual(int(group_heads.numpy()[0, 0]), 0)
        self.assertEqual(int(serial_contact_head.numpy()[0]), 3)
        np.testing.assert_array_equal(row_parent.numpy()[0, (3, 6)], np.array((6, -1), dtype=np.int32))
        normal_links = sparse_row_dof.numpy()[0, ::3, 1]
        np.testing.assert_array_equal(normal_links, np.array((-12, -1, -1, -2), dtype=np.int32))

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
        kernel = feather_pgs_module._get_direct_diagonal_inverse_mass_kernel(1, str(device.arch))
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
        zero_spatial = wp.spatial_vector()
        articulation_start = wp.array((0, 1), dtype=wp.int32, device=device)
        articulation_end = wp.array((1,), dtype=wp.int32, device=device)
        articulation_dof_start = wp.array((0,), dtype=wp.int32, device=device)
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
        external_force = wp.array((zero_spatial,), dtype=wp.spatial_vector, device=device)
        body_flags = wp.zeros(1, dtype=wp.int32, device=device)
        body_q = wp.array((wp.transform_identity(),), dtype=wp.transform, device=device)
        body_com = wp.array((wp.vec3(0.2, -0.1, 0.3),), dtype=wp.vec3, device=device)
        origin = wp.array((wp.vec3(),), dtype=wp.vec3, device=device)
        selected_tau = wp.array((0.7,), dtype=wp.float32, device=device)
        direct_tau = wp.clone(selected_tau)
        body_ft = wp.zeros(1, dtype=wp.spatial_vector, device=device)
        selected_kernel, direct_kernel = feather_pgs_module._get_partitioned_inverse_dynamics_kernels(
            1, str(device.arch)
        )
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
                articulation_dof_start,
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

    @unittest.skipUnless(wp.is_cuda_available(), "dense contact partition requires CUDA")
    def test_dense_contact_partition_rejects_divisible_mixed_rows(self):
        """Reject mixed contact rows even when their total is divisible by three."""
        device = wp.get_device("cuda:0")
        row_types = wp.array(
            [
                [
                    PGS_CONSTRAINT_TYPE_CONTACT,
                    PGS_CONSTRAINT_TYPE_FRICTION,
                    PGS_CONSTRAINT_TYPE_FRICTION,
                    PGS_CONSTRAINT_TYPE_CONTACT,
                    PGS_CONSTRAINT_TYPE_CONTACT,
                    PGS_CONSTRAINT_TYPE_CONTACT,
                ]
            ],
            dtype=wp.int32,
            device=device,
        )
        world_slot_counter = wp.array([6], dtype=wp.int32, device=device)
        dense_phase_bounds = wp.zeros((1, 2), dtype=wp.int32, device=device)
        mf_constraint_count = wp.zeros(1, dtype=wp.int32, device=device)
        mf_contact_rows_end = wp.zeros(1, dtype=wp.int32, device=device)
        world_constraint_count = wp.zeros(1, dtype=wp.int32, device=device)
        triple_eligible = wp.zeros(1, dtype=wp.int32, device=device)
        mixed_world_count = wp.zeros(1, dtype=wp.int32, device=device)
        mixed_worlds = wp.zeros(1, dtype=wp.int32, device=device)
        fallback_world_count = wp.zeros(1, dtype=wp.int32, device=device)
        fallback_worlds = wp.zeros(1, dtype=wp.int32, device=device)

        wp.launch(
            feather_pgs_module._finalize_dense_contact_triple_partition,
            dim=1,
            inputs=[
                world_slot_counter,
                6,
                6,
                dense_phase_bounds,
                row_types,
                mf_constraint_count,
                mf_contact_rows_end,
            ],
            outputs=[
                world_constraint_count,
                triple_eligible,
                mixed_world_count,
                mixed_worlds,
                fallback_world_count,
                fallback_worlds,
            ],
            device=device,
        )

        self.assertEqual(int(world_constraint_count.numpy()[0]), 6)
        self.assertEqual(int(triple_eligible.numpy()[0]), 0)
        self.assertEqual(int(mixed_world_count.numpy()[0]), 0)
        self.assertEqual(int(fallback_world_count.numpy()[0]), 1)
        self.assertEqual(int(fallback_worlds.numpy()[0]), 0)

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
            diagonal_mass_sizes=frozenset(),
            cholesky_kernel="auto",
            hinv_jt_kernel="tiled",
            hinv_jt_compute_diag=solver._hinv_jt_computes_diag,
            small_dof_threshold=12,
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

    @unittest.skipUnless(wp.is_cuda_available(), "articulation-local mixed-world parity requires CUDA")
    def test_local_admission_preserves_general_response_for_row_overflow(self):
        """Preserve the general response when a local-capable world exceeds its row limit."""
        run_kwargs = {
            "warmstart": False,
            "preelimination": False,
            "dof_count": 16,
            "internal_constraint_count": 41,
            "dense_max_constraints": 96,
        }
        reference_solver, reference = _run_mixed_response("tiled", **run_kwargs)
        local_solver, local = _run_mixed_response("auto", **run_kwargs)

        self.assertFalse(reference_solver._local_internal_fast_path)
        self.assertTrue(local_solver._local_internal_fast_path)
        self.assertTrue(all(sample[5] == 0 for sample in local), "overflow world unexpectedly used a local owner")
        self.assertGreater(local[0][0], 40, "test world did not exceed the local row capacity")
        expected, actual = reference[0], local[0]
        self.assertEqual(actual[0], expected[0], "constraint count differed")
        np.testing.assert_allclose(np.sort(actual[1]), np.sort(expected[1]), rtol=5.0e-4, atol=2.0e-6)
        self.assertTrue(np.isfinite(actual[2]).all(), "overflow solve produced non-finite impulses")
        self.assertTrue(np.isfinite(actual[3]).all(), "overflow solve produced a non-finite position")
        self.assertTrue(np.isfinite(actual[4]).all(), "overflow solve produced a non-finite velocity")

    @unittest.skipUnless(wp.is_cuda_available(), "static-contact dense solve parity requires CUDA")
    def test_static_contact_dense_only_matches_general_response(self):
        """Match the general solver when the model topology cannot produce matrix-free rows."""
        run_kwargs = {
            "warmstart": False,
            "preelimination": False,
            "dof_count": 18,
            "internal_constraint_count": 0,
            "friction": 0.7,
            "restitution": 0.3,
            "tangential_velocity": 2.0,
            "dense_max_constraints": 96,
            "free_body": False,
        }
        kernel_factory = feather_pgs_module._get_pgs_solve_mf_gs_kernel

        def force_general_kernel(*args, **kwargs):
            kwargs["dense_only_mf_free"] = False
            kwargs["dense_only_interleaved"] = False
            return kernel_factory(*args, **kwargs)

        with (
            mock.patch.object(feather_pgs_module, "_get_pgs_solve_mf_gs_kernel", force_general_kernel),
            mock.patch.object(SolverFeatherPGS, "_detect_jy_world_identity", return_value=False),
        ):
            reference_solver, reference = _run_mixed_response("auto", **run_kwargs)
        specialized_solver, specialized = _run_mixed_response("auto", **run_kwargs)

        self.assertFalse(reference_solver._local_internal_fast_path)
        self.assertFalse(specialized_solver._local_internal_fast_path)
        self.assertFalse(reference_solver._jy_world_aliased)
        self.assertTrue(specialized_solver._jy_world_aliased)
        self.assertFalse(reference_solver._identity_contact_jacobian_fast_path)
        self.assertFalse(specialized_solver._identity_contact_jacobian_fast_path)
        self.assertTrue(specialized_solver._world_owned_contact_response)
        self.assertTrue(specialized_solver._factor_coordinate_contact_triples)
        self.assertTrue(specialized_solver._factor_coordinate_all_dense_worlds)
        self.assertEqual(specialized_solver._contact_response_primary_size, 18)
        self.assertEqual(specialized_solver._contact_response_secondary_size, 0)
        self.assertIsNone(reference_solver._hinv_jt_direct_world_diag_size)
        self.assertIsNone(specialized_solver._hinv_jt_direct_world_diag_size)
        self.assertEqual(specialized_solver._hinv_jt_diag_sizes, frozenset())
        self.assertNotIn("_dense0", reference_solver._pgs_solve_mf_gs_kernel.key)
        self.assertIn("_dense0", specialized_solver._pgs_solve_mf_gs_kernel.key)
        self.assertIn("_interleaved", specialized_solver._pgs_solve_mf_gs_kernel.key)
        self.assertIn("_factor_arbitrary", specialized_solver._world_owned_contact_response_kernel.key)
        self.assertIn("pgs_solve_factor_dense", specialized_solver._pgs_solve_factor_dense_kernel.key)
        self.assertIsNone(specialized_solver._pgs_solve_dense_contact_triples_kernel)
        self.assertFalse(specialized_solver._has_free_rigid_bodies)
        for step, (expected, actual) in enumerate(zip(reference, specialized, strict=True)):
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

    @unittest.skipUnless(wp.is_cuda_available(), "wide factor-coordinate response requires CUDA")
    def test_static_contact_wide_factor_matches_general_response(self):
        """Match physical-space PGS for a 43-DOF one-component world."""
        run_kwargs = {
            "warmstart": False,
            "preelimination": False,
            "dof_count": 43,
            "internal_constraint_count": 0,
            "friction": 0.7,
            "restitution": 0.3,
            "tangential_velocity": 2.0,
            "dense_max_constraints": 96,
            "free_body": False,
        }
        with mock.patch.object(SolverFeatherPGS, "_detect_jy_world_identity", return_value=False):
            reference_solver, reference = _run_mixed_response("auto", **run_kwargs)
        specialized_solver, specialized = _run_mixed_response("auto", **run_kwargs)

        self.assertFalse(reference_solver._factor_coordinate_contact_triples)
        self.assertFalse(specialized_solver._world_owned_contact_response)
        self.assertTrue(specialized_solver._factor_coordinate_tiled_response)
        self.assertTrue(specialized_solver._factor_coordinate_all_dense_worlds)
        self.assertEqual(specialized_solver._contact_response_primary_size, 43)
        self.assertIn("_factor", specialized_solver._hinv_jt_kernels_by_size[43].key)
        self.assertIn("pgs_solve_factor_dense", specialized_solver._pgs_solve_factor_dense_kernel.key)
        self.assertGreater(specialized[0][0], 0, "wide scene generated no dense rows")
        for step, (expected, actual) in enumerate(zip(reference, specialized, strict=True)):
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

    @unittest.skipUnless(wp.is_cuda_available(), "mixed-friction response parity requires CUDA")
    def test_world_owned_identity_keeps_factor_coordinates_for_mixed_friction_rows(self):
        """Preserve normal-only contacts without switching the world to physical coordinates."""
        run_kwargs = {
            "warmstart": False,
            "preelimination": False,
            "dof_count": 18,
            "internal_constraint_count": 0,
            "friction": 0.7,
            "restitution": 0.3,
            "tangential_velocity": 2.0,
            "dense_max_constraints": 96,
            "free_body": False,
            "contact_friction_anchor_limit": 2,
        }
        with mock.patch.object(SolverFeatherPGS, "_detect_jy_world_identity", return_value=False):
            reference_solver, reference = _run_mixed_response("auto", **run_kwargs)
        specialized_solver, specialized = _run_mixed_response("auto", **run_kwargs)

        self.assertFalse(reference_solver._world_owned_contact_response)
        self.assertTrue(specialized_solver._world_owned_contact_response)
        self.assertTrue(specialized_solver._factor_coordinate_all_dense_worlds)
        self.assertIsNone(specialized_solver._world_owned_contact_response_fallback_kernel)
        self.assertGreater(specialized[0][0], 0, "mixed-friction scene generated no dense rows")
        self.assertNotEqual(specialized[0][0] % 3, 0, "test scene unexpectedly retained only contact triples")
        for step, (expected, actual) in enumerate(zip(reference, specialized, strict=True)):
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

    @unittest.skipUnless(wp.is_cuda_available(), "world-owned contact problem parity requires CUDA")
    def test_world_owned_factor_pair_matches_general_response_for_23_dofs(self):
        """Match the general solve when a 23-DOF pair owns its complete contact problem."""
        run_kwargs = {
            "warmstart": False,
            "preelimination": False,
            "inactive_joint_limit_capacity": True,
            "friction": 0.7,
            "restitution": 0.3,
            "tangential_velocity": 2.0,
            "dof_count": 23,
            "internal_constraint_count": 0,
            "dense_max_constraints": 96,
        }
        reference_solver, reference = _run_mixed_response("par_row", **run_kwargs)
        paired_solver, paired = _run_mixed_response("auto", **run_kwargs)

        self.assertIsNone(reference_solver._paired_response_primary_size)
        self.assertEqual(paired_solver._paired_response_primary_size, 23)
        self.assertEqual(paired_solver._paired_response_secondary_size, 6)
        self.assertTrue(paired_solver._world_owned_contact_response)
        self.assertTrue(paired_solver._factor_coordinate_contact_triples)
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
