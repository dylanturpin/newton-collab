# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_LOCAL_SOLVE_OWNER_PAIR,
    accumulate_group_diag_worlds,
)
from newton._src.solvers.feather_pgs.solver_feather_pgs import _FeatherPGSExecutionPlan, _get_hinv_jt_kernel
from newton.solvers import SolverFeatherPGS


def _build_mixed_response_model(device, world_count=1, *, dof_count=13, friction=0.0, restitution=0.0):
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
):
    """Run a short mixed-contact trajectory with one H-inverse implementation."""
    model = _build_mixed_response_model("cuda:0", dof_count=dof_count, friction=friction, restitution=restitution)
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
            )
        )
        state_in, state_out = state_out, state_in
    return solver, samples


class TestFeatherPGSResponseDiagonal(unittest.TestCase):
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
