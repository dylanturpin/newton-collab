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
    PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
    PGS_LOCAL_SOLVE_OWNER_GENERAL,
    PGS_LOCAL_SOLVE_OWNER_PAIR,
    PGS_LOCAL_SOLVE_OWNER_PAIR_RESIDUAL,
    accumulate_group_diag_worlds,
)
from newton._src.solvers.feather_pgs.solver_feather_pgs import _FeatherPGSExecutionPlan, _get_hinv_jt_kernel
from newton.solvers import SolverFeatherPGS


def _build_mixed_response_model(
    device,
    world_count=1,
    *,
    dof_count=13,
    friction=0.0,
    restitution=0.0,
    static_plane=False,
    static_support=False,
    free_body_first=False,
    free_body_velocity_limit=None,
):
    """Build one serial articulation contacting one free rigid body.

    ``static_support`` rests the free body on a static ledge that clears the arm, so the free body also
    produces matrix-free contact rows. ``free_body_first`` builds the free body before the articulation,
    which packs the world DOFs free-body first. ``free_body_velocity_limit`` caps the free body's linear
    and angular velocity [m/s, rad/s], producing matrix-free velocity-limit rows once exceeded.
    ``static_plane`` adds a ground plane under the free body.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    if free_body_velocity_limit is not None:
        SolverFeatherPGS.register_custom_attributes(builder)
    builder.default_shape_cfg.density = 1000.0
    builder.default_shape_cfg.ke = 1.0e5
    builder.default_shape_cfg.kd = 1.0e3
    builder.default_shape_cfg.mu = friction
    builder.default_shape_cfg.restitution = restitution
    builder.default_shape_cfg.margin = 0.0
    builder.default_shape_cfg.gap = 0.0

    def add_free_body():
        custom_attributes = None
        if free_body_velocity_limit is not None:
            custom_attributes = {
                "rigid_body_max_linear_velocity": free_body_velocity_limit,
                "rigid_body_max_angular_velocity": free_body_velocity_limit,
            }
        box = builder.add_link(
            xform=wp.transform(wp.vec3(0.7, 0.0, 0.5695), wp.quat_identity()), custom_attributes=custom_attributes
        )
        builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.05)
        builder.add_articulation([builder.add_joint_free(parent=-1, child=box)])
        if static_support:
            builder.add_shape_box(
                -1,
                xform=wp.transform(wp.vec3(0.7, 0.11, 0.47), wp.quat_identity()),
                hx=0.05,
                hy=0.05,
                hz=0.05,
            )

    if free_body_first:
        add_free_body()
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

    if not free_body_first:
        add_free_body()
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
    contact_regularization=0.0,
    friction_anchor_beta=None,
    model_kwargs=None,
):
    """Run a short mixed-contact trajectory with one H-inverse implementation."""
    model = _build_mixed_response_model(
        "cuda:0",
        dof_count=dof_count,
        friction=friction,
        restitution=restitution,
        static_plane=static_plane,
        **(model_kwargs or {}),
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
            pgs_contact_regularization=contact_regularization,
            friction_anchor_beta=friction_anchor_beta,
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
        mf_count = int(solver.mf_constraint_count.numpy()[0])
        samples.append(
            (
                constraint_count,
                solver.diag.numpy()[0, :constraint_count].copy(),
                solver.impulses.numpy()[0, :constraint_count].copy(),
                state_out.joint_q.numpy().copy(),
                state_out.joint_qd.numpy().copy(),
                int(solver._local_solve_owner.numpy()[0]),
                solver.row_type.numpy()[0, :constraint_count].copy(),
                mf_count,
                solver.mf_impulses.numpy()[0, :mf_count].copy(),
                solver.mf_row_type.numpy()[0, :mf_count].copy(),
                solver.mf_row_parent.numpy()[0, :mf_count].copy(),
                solver.mf_J_a.numpy()[0, :mf_count].copy(),
                solver.mf_J_b.numpy()[0, :mf_count].copy(),
                solver.mf_MiJt_a.numpy()[0, :mf_count].copy(),
                solver.mf_MiJt_b.numpy()[0, :mf_count].copy(),
            )
        )
        state_in, state_out = state_out, state_in
    return solver, samples


def _assert_owner_parity(test, general, local):
    """Assert the local owner trajectory matches the general owner row for row."""
    test.assertEqual(len(general), len(local))
    test.assertGreater(general[0][0], 0, "mixed scene generated no dense constraint rows")
    for step, (expected, actual) in enumerate(zip(general, local, strict=True)):
        test.assertEqual(actual[0], expected[0], f"constraint count differed at step {step}")
        test.assertEqual(actual[7], expected[7], f"matrix-free row count differed at step {step}")
        np.testing.assert_array_equal(actual[6], expected[6], err_msg=f"row types differed at step {step}")
        np.testing.assert_array_equal(actual[9], expected[9], err_msg=f"matrix-free row types differed at step {step}")
        for label, expected_value, actual_value in zip(
            ("diagonal", "impulses", "joint_q", "joint_qd", "mf_impulses"),
            (*expected[1:5], expected[8]),
            (*actual[1:5], actual[8]),
            strict=True,
        ):
            np.testing.assert_allclose(
                actual_value,
                expected_value,
                rtol=5.0e-4,
                atol=2.0e-6,
                err_msg=f"{label} differed at step {step}",
            )


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
        """Match the general response when one warp owns a robot/free-body pair, with patch and point friction."""
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
        # Patch friction rows are not uniform triples, so the generic factor path pools the patch load; point
        # friction keeps the contact-triple layout.
        for label, friction_anchor_beta, triples in (("patch", None, False), ("point", 0.0, True)):
            with self.subTest(friction=label):
                reference_solver, reference = _run_mixed_response(
                    "par_row", friction_anchor_beta=friction_anchor_beta, **run_kwargs
                )
                paired_solver, paired = _run_mixed_response(
                    "auto", friction_anchor_beta=friction_anchor_beta, **run_kwargs
                )

                self.assertIsNone(reference_solver._paired_response_primary_size)
                self.assertEqual(paired_solver._friction_anchors_enabled, label == "patch")
                self.assertEqual(paired_solver._paired_response_primary_size, 23)
                self.assertEqual(paired_solver._paired_response_secondary_size, 6)
                self.assertIsNotNone(paired_solver._paired_response_kernel)
                self.assertIsNotNone(paired_solver._paired_factor_solve_kernel)
                self.assertTrue(paired_solver._paired_factor_coordinates)
                self.assertEqual(paired_solver._factor_coordinate_contact_triples, triples)
                self.assertGreater(reference[0][0], 0, "mixed scene generated no dense constraint rows")
                self.assertTrue(
                    any(np.any(sample[6] == PGS_CONSTRAINT_TYPE_FRICTION) for sample in paired),
                    "mixed scene generated no friction rows",
                )
                for step, (expected, actual) in enumerate(zip(reference, paired, strict=True)):
                    self.assertEqual(actual[0], expected[0], f"constraint count differed at step {step}")
                    np.testing.assert_array_equal(actual[6], expected[6], err_msg=f"row types differed at step {step}")
                    for label_value, expected_value, actual_value in zip(
                        ("diagonal", "impulses", "joint_q", "joint_qd"), expected[1:5], actual[1:5], strict=True
                    ):
                        np.testing.assert_allclose(
                            actual_value,
                            expected_value,
                            rtol=5.0e-4,
                            atol=1.0e-5,
                            err_msg=f"{label_value} differed at step {step}",
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

    @unittest.skipUnless(wp.is_cuda_available(), "paired response ownership requires CUDA")
    def test_paired_factor_coordinates_follow_patch_friction_and_torsion(self):
        """Keep factor coordinates with default patch friction and yield them to the general owner for torsion."""
        model = _build_mixed_response_model("cuda:0", dof_count=23, friction=0.7)
        solvers = {}
        with mock.patch.object(SolverFeatherPGS, "_kernel_overrides", {"hinv_jt_kernel": "auto"}):
            for radius in (0.0, 0.01):
                solvers[radius] = SolverFeatherPGS(
                    model,
                    pgs_mode="matrix_free",
                    enable_joint_limits=True,
                    joint_limit_activation_gap=0.0,
                    pgs_iterations=8,
                    dense_max_constraints=96,
                    mf_max_constraints=32,
                    contact_torsion_radius=radius,
                )
            point = SolverFeatherPGS(
                model,
                pgs_mode="matrix_free",
                enable_joint_limits=True,
                joint_limit_activation_gap=0.0,
                pgs_iterations=8,
                dense_max_constraints=96,
                mf_max_constraints=32,
                friction_anchor_beta=0.0,
            )
        self.assertTrue(solvers[0.0]._friction_anchors_enabled)
        self.assertTrue(solvers[0.0]._paired_factor_coordinates)
        # Patch rows are not uniform triples; the generic factor path pools the patch load instead.
        self.assertFalse(solvers[0.0]._factor_coordinate_contact_triples)
        self.assertFalse(point._friction_anchors_enabled)
        self.assertTrue(point._paired_factor_coordinates)
        self.assertTrue(point._factor_coordinate_contact_triples)
        self.assertTrue(solvers[0.01]._contact_torsion_enabled)
        self.assertIsNotNone(solvers[0.01]._paired_response_primary_size)
        self.assertFalse(solvers[0.01]._paired_factor_coordinates)
        self.assertFalse(solvers[0.01]._factor_coordinate_contact_triples)

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
    def test_local_owners_match_general_contact_regularization(self):
        """Apply the general owner's per-row regularization weights in the local owners."""
        run_kwargs = {
            "warmstart": False,
            "preelimination": False,
            "inactive_joint_limit_capacity": True,
            "friction": 0.7,
            "restitution": 0.3,
            "tangential_velocity": 2.0,
            "contact_regularization": 1.0,
            "model_kwargs": {"static_support": True},
        }
        general_solver, general = _run_mixed_response("tiled", **run_kwargs)
        local_solver, local = _run_mixed_response("par_row", **run_kwargs)

        self.assertTrue(general_solver._regularization_enabled)
        self.assertFalse(general_solver._local_internal_fast_path)
        self.assertTrue(local_solver._local_internal_fast_path)
        owners = [sample[5] for sample in local]
        self.assertIn(
            PGS_LOCAL_SOLVE_OWNER_PAIR_RESIDUAL, owners, "regularized mixed contact never selected the residual owner"
        )
        self.assertTrue(
            any(sample[7] > 0 and sample[5] != PGS_LOCAL_SOLVE_OWNER_GENERAL for sample in local),
            "no local owner solved matrix-free rows",
        )
        _assert_owner_parity(self, general, local)

    @unittest.skipUnless(wp.is_cuda_available(), "articulation-local mixed-world parity requires CUDA")
    def test_local_owner_matches_general_persistent_patch_friction(self):
        """Preserve pooled normal load and coupled tangent response under local ownership."""
        run_kwargs = {
            "warmstart": False,
            "preelimination": False,
            "inactive_joint_limit_capacity": True,
            "friction": 0.7,
            "restitution": 0.3,
            "tangential_velocity": 2.0,
            "model_kwargs": {"static_support": True},
        }
        general_solver, general = _run_mixed_response("tiled", **run_kwargs)
        local_solver, local = _run_mixed_response("par_row", **run_kwargs)

        self.assertTrue(general_solver._friction_anchors_enabled)
        self.assertTrue(local_solver._friction_anchors_enabled)
        self.assertFalse(general_solver._local_internal_fast_path)
        self.assertTrue(local_solver._local_internal_fast_path)
        self.assertIn(PGS_LOCAL_SOLVE_OWNER_PAIR_RESIDUAL, [sample[5] for sample in local])

        self.assertTrue(
            any(
                sample[10][row] >= 0 and sample[10][row] != row
                for sample in local
                for row in np.flatnonzero(sample[9] == PGS_CONSTRAINT_TYPE_CONTACT)
            ),
            "scene generated no linked multi-normal patch",
        )

        tangent_cross_terms = []
        for sample in local:
            row_type, row_parent = sample[9], sample[10]
            jacobian_a, jacobian_b = sample[11], sample[12]
            response_a, response_b = sample[13], sample[14]
            for row in np.flatnonzero(row_type == PGS_CONSTRAINT_TYPE_FRICTION):
                parent = row_parent[row]
                if row != parent + 1:
                    continue
                sibling = parent + 2
                tangent_cross_terms.append(
                    float(jacobian_a[row] @ response_a[sibling] + jacobian_b[row] @ response_b[sibling])
                )
        self.assertTrue(tangent_cross_terms, "scene generated no tangent pair")
        self.assertGreater(max(abs(value) for value in tangent_cross_terms), 1.0e-3)
        _assert_owner_parity(self, general, local)

    @unittest.skipUnless(wp.is_cuda_available(), "articulation-local mixed-world parity requires CUDA")
    def test_free_body_velocity_limit_rows_stay_on_general_owner(self):
        """Keep worlds with free-body velocity-limit rows on the general owner."""
        run_kwargs = {
            "warmstart": False,
            "preelimination": False,
            "inactive_joint_limit_capacity": True,
            "model_kwargs": {"free_body_velocity_limit": 1.0},
        }
        general_solver, general = _run_mixed_response("tiled", **run_kwargs)
        local_solver, local = _run_mixed_response("par_row", **run_kwargs)

        self.assertFalse(general_solver._local_internal_fast_path)
        self.assertTrue(local_solver._local_internal_fast_path)
        limited = [sample for sample in local if np.any(sample[9] == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)]
        self.assertTrue(limited, "free body never produced a velocity-limit row")
        # The residual owner is eligible for this world; only the velocity-limit rows keep it general.
        self.assertGreaterEqual(int(local_solver._local_residual_pair_articulation.numpy()[0]), 0)
        for sample in limited:
            self.assertEqual(sample[5], PGS_LOCAL_SOLVE_OWNER_GENERAL, "velocity-limit rows left the general owner")
        _assert_owner_parity(self, general, local)

    @unittest.skipUnless(wp.is_cuda_available(), "articulation-local mixed-world parity requires CUDA")
    def test_free_body_first_world_layout_matches_general(self):
        """Reject the residual owner when the world packs the free body before the articulation."""
        run_kwargs = {
            "warmstart": False,
            "preelimination": False,
            "inactive_joint_limit_capacity": True,
            "friction": 0.7,
            "restitution": 0.3,
            "tangential_velocity": 2.0,
            "model_kwargs": {"static_support": True, "free_body_first": True},
        }
        general_solver, general = _run_mixed_response("tiled", **run_kwargs)
        local_solver, local = _run_mixed_response("par_row", **run_kwargs)

        free_articulation = int(np.flatnonzero(local_solver._model_plan.is_free_rigid)[0])
        self.assertEqual(int(local_solver._model_plan.articulation_dof_start[free_articulation]), 0)
        self.assertFalse(general_solver._local_internal_fast_path)
        self.assertTrue(local_solver._local_internal_fast_path)
        np.testing.assert_array_equal(local_solver._local_residual_pair_articulation.numpy(), [-1])
        self.assertGreaterEqual(int(local_solver._local_pair_articulation.numpy()[0]), 0)
        self.assertTrue(any(sample[7] > 0 for sample in local), "free body generated no matrix-free rows")
        for sample in local:
            if sample[7] > 0:
                self.assertEqual(sample[5], PGS_LOCAL_SOLVE_OWNER_GENERAL, "matrix-free rows left the general owner")
        _assert_owner_parity(self, general, local)

    @unittest.skipUnless(wp.is_cuda_available(), "articulation-local mixed-world parity requires CUDA")
    def test_local_owners_stay_general_with_contact_torsion(self):
        """Keep torsion-eligible worlds on the general owner, which solves the appended torsion row."""
        model = _build_mixed_response_model("cuda:0", friction=0.7)
        solvers = {}
        with mock.patch.object(SolverFeatherPGS, "_kernel_overrides", {"hinv_jt_kernel": "par_row"}):
            for radius in (0.0, 0.01):
                solvers[radius] = SolverFeatherPGS(
                    model,
                    pgs_mode="matrix_free",
                    enable_joint_limits=True,
                    joint_limit_activation_gap=0.0,
                    pgs_iterations=8,
                    dense_max_constraints=32,
                    mf_max_constraints=32,
                    contact_torsion_radius=radius,
                )
        self.assertTrue(solvers[0.0]._local_internal_fast_path)
        self.assertTrue(solvers[0.01]._contact_torsion_enabled)
        self.assertFalse(solvers[0.01]._local_internal_fast_path)

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
