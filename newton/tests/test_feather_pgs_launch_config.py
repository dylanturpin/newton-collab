# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.solver_feather_pgs import (
    _DENSE_META_MAX_PARENT,
    _DENSE_META_ROW_TYPE_MASK,
    _FeatherPGSExecutionPlan,
    _select_hinv_jt_chunk_size,
    _use_resident_mfgs_metadata,
    _validate_dense_metadata_encoding,
)
from newton.solvers import SolverFeatherPGS


def _build_chain_model(num_links=3, num_worlds=2):
    chain = newton.ModelBuilder()
    hx = 0.3
    joints = []
    parent = -1
    root_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.45 * wp.pi)
    for _ in range(num_links):
        link = chain.add_link()
        chain.add_shape_box(link, hx=hx - 0.08, hy=0.05, hz=0.05)
        if parent == -1:
            parent_xform = wp.transform(p=wp.vec3(0.0, 0.0, 2.5), q=root_rot)
        else:
            parent_xform = wp.transform(p=wp.vec3(hx, 0.0, 0.0), q=wp.quat_identity())
        joints.append(
            chain.add_joint_revolute(
                parent=parent,
                child=link,
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent_xform=parent_xform,
                child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
            )
        )
        parent = link
    chain.add_articulation(joints)
    main = newton.ModelBuilder()
    main.replicate(chain, num_worlds, spacing=(3.0, 3.0, 0.0))
    return main.finalize()


def _build_heterogeneous_world_model():
    free_template = newton.ModelBuilder()
    free_body = free_template.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    free_joint = free_template.add_joint_free(parent=-1, child=free_body)
    free_template.add_articulation([free_joint])

    slider_template = newton.ModelBuilder()
    slider_body = slider_template.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    slider_joint = slider_template.add_joint_prismatic(parent=-1, child=slider_body, axis=newton.Axis.X)
    slider_template.add_articulation([slider_joint])

    builder = newton.ModelBuilder()
    builder.add_world(free_template)
    builder.add_world(slider_template)
    return builder.finalize()


class TestFeatherPGSLaunchConfig(unittest.TestCase):
    def test_defaults_preserved(self):
        model = newton.ModelBuilder().finalize()
        solver = SolverFeatherPGS(model)
        self.assertEqual(solver.serial_kernel_block_dim, 256)
        self.assertEqual(solver.tile_threads, 64)
        self.assertEqual(solver.articulated_contact_response, "immediate")

    def test_articulated_contact_response_validation(self):
        model = _build_chain_model(num_links=2, num_worlds=1)
        solver = SolverFeatherPGS(
            model,
            pgs_mode="matrix_free",
            articulated_contact_response="propagation",
        )
        self.assertEqual(solver.articulated_contact_response, "propagation")
        fused = SolverFeatherPGS(
            model,
            pgs_mode="matrix_free",
            articulated_contact_response="propagation-fused",
        )
        self.assertEqual(fused.articulated_contact_response, "propagation-fused")

        with self.assertRaisesRegex(ValueError, "articulated_contact_response"):
            SolverFeatherPGS(model, articulated_contact_response="bad")

        with self.assertRaisesRegex(NotImplementedError, "pgs_mode='matrix_free'"):
            SolverFeatherPGS(model, pgs_mode="split", articulated_contact_response="propagation")
        with self.assertRaisesRegex(NotImplementedError, "pgs_mode='matrix_free'"):
            SolverFeatherPGS(model, pgs_mode="split", articulated_contact_response="propagation-fused")

        with self.assertRaisesRegex(NotImplementedError, "friction_mode='current'"):
            SolverFeatherPGS(
                model,
                pgs_mode="matrix_free",
                articulated_contact_response="propagation",
                friction_mode="bisection",
            )
        propagation_debug = SolverFeatherPGS(
            model,
            pgs_mode="matrix_free",
            articulated_contact_response="propagation",
            pgs_debug=True,
        )
        self.assertEqual(propagation_debug.articulated_contact_response, "propagation")

    def test_default_kwargs_produce_identical_kernel_selection(self):
        model = _build_chain_model()
        implicit = SolverFeatherPGS(model)
        explicit = SolverFeatherPGS(model, serial_kernel_block_dim=256, tile_threads=64)
        for attr in (
            "_cholesky_kernels_by_size",
            "_triangular_solve_kernels_by_size",
            "_hinv_jt_kernels_by_size",
            "_hinv_jt_fused_kernels_by_size",
        ):
            implicit_kernels = getattr(implicit, attr)
            explicit_kernels = getattr(explicit, attr)
            self.assertEqual(set(implicit_kernels), set(explicit_kernels))
            for size, kernel in implicit_kernels.items():
                # The factories are functools.cache'd, so identical knob values
                # must resolve to the *same* kernel objects as the defaults.
                self.assertIs(kernel, explicit_kernels[size], f"{attr}[{size}]")

    def test_non_default_tile_threads_selects_distinct_kernels(self):
        model = _build_chain_model()
        default = SolverFeatherPGS(model)
        wide = SolverFeatherPGS(model, tile_threads=128)
        for size, kernel in default._cholesky_kernels_by_size.items():
            if kernel is None:
                continue
            other = wide._cholesky_kernels_by_size[size]
            self.assertIsNot(kernel, other)
            self.assertIn("_bd64", kernel.key)
            self.assertIn("_bd128", other.key)

    @unittest.skipUnless(wp.is_cuda_available(), "compact response mapping requires CUDA matrix-free mode")
    def test_compact_world_dof_mapping_pads_heterogeneous_worlds(self):
        solver = SolverFeatherPGS(_build_heterogeneous_world_model(), pgs_mode="matrix_free")
        self.assertEqual(solver.max_world_dofs, 6)
        np.testing.assert_array_equal(solver.world_dof_count.numpy(), np.array((6, 1), dtype=np.int32))
        indices = solver.world_dof_indices.numpy()
        np.testing.assert_array_equal(indices[0], np.arange(6, dtype=np.int32))
        self.assertGreaterEqual(int(indices[1, 0]), 0)
        np.testing.assert_array_equal(indices[1, 1:], np.full(5, -1, dtype=np.int32))

    def test_hinv_chunk_selection_respects_shared_memory(self):
        cases = (
            (23, 384, 101376, 64),
            (128, 384, 101376, 16),
            (160, 384, 101376, None),
            (23, 0, 101376, None),
            (23, 384, 8192, 16),
            (23, 384, 2500, None),
            (100, 1, 49152, 1),
        )
        for n_dofs, max_constraints, shared_memory, expected in cases:
            with self.subTest(n_dofs=n_dofs, max_constraints=max_constraints, shared_memory=shared_memory):
                self.assertEqual(_select_hinv_jt_chunk_size(n_dofs, max_constraints, shared_memory, 64), expected)

    def test_hinv_chunk_selection_accounts_for_tile_threads(self):
        self.assertEqual(_select_hinv_jt_chunk_size(50, 384, 49152, 64), 64)
        self.assertEqual(_select_hinv_jt_chunk_size(50, 384, 49152, 256), 32)

    def test_hinv_chunk_selection_caps_compact_articulations(self):
        """Cap compact articulation chunks without restricting larger groups."""
        self.assertEqual(_select_hinv_jt_chunk_size(20, 384, 101376, 64), 32)
        self.assertEqual(_select_hinv_jt_chunk_size(21, 384, 101376, 64), 64)

    def test_dense_metadata_encoding_bounds(self):
        _validate_dense_metadata_encoding(32)
        self.assertGreaterEqual(_DENSE_META_ROW_TYPE_MASK + 1, 5)
        with self.assertRaisesRegex(ValueError, "packed parent capacity"):
            _validate_dense_metadata_encoding(_DENSE_META_MAX_PARENT + 2)

    def test_hinv_execution_plan_falls_back_or_rejects_safely(self):
        fallback = _FeatherPGSExecutionPlan.build(
            [160],
            max_constraints=384,
            max_shared_memory=101376,
            hinv_jt_kernel="auto",
            small_dof_threshold=12,
            tile_threads=64,
        )
        self.assertFalse(fallback.use_tiled_hinv_jt(160))

        zero_capacity = _FeatherPGSExecutionPlan.build(
            [23],
            max_constraints=0,
            max_shared_memory=101376,
            hinv_jt_kernel="tiled",
            small_dof_threshold=12,
            tile_threads=64,
        )
        self.assertFalse(zero_capacity.use_tiled_hinv_jt(23))

        with self.assertRaisesRegex(ValueError, "hinv_jt_kernel='tiled'"):
            _FeatherPGSExecutionPlan.build(
                [160],
                max_constraints=384,
                max_shared_memory=101376,
                hinv_jt_kernel="tiled",
                small_dof_threshold=12,
                tile_threads=64,
            )

    def test_hinv_fusion_requires_full_working_set_to_fit(self):
        fitting = _FeatherPGSExecutionPlan.build(
            [23],
            max_constraints=64,
            max_shared_memory=101376,
            hinv_jt_kernel="auto",
            small_dof_threshold=12,
            tile_threads=64,
        )
        oversized = _FeatherPGSExecutionPlan.build(
            [23],
            max_constraints=384,
            max_shared_memory=101376,
            hinv_jt_kernel="auto",
            small_dof_threshold=12,
            tile_threads=64,
        )
        self.assertTrue(fitting.use_fused_hinv_jt(23))
        self.assertFalse(oversized.use_fused_hinv_jt(23))

    def test_mfgs_metadata_storage_respects_resource_budget(self):
        """Keep resident metadata only when its complete working set fits."""
        compact = _use_resident_mfgs_metadata(192, 64, 29, 101376, has_drive_rows=False, fuse_vel_limits=False)
        large = _use_resident_mfgs_metadata(1024, 4096, 604, 101376, has_drive_rows=False, fuse_vel_limits=False)
        overcommitted = _use_resident_mfgs_metadata(192, 64, 29, 4096, has_drive_rows=False, fuse_vel_limits=False)

        self.assertTrue(compact)
        self.assertFalse(large)
        self.assertFalse(overcommitted)

    def test_serial_kernel_block_dim_validation(self):
        model = newton.ModelBuilder().finalize()
        for bad in (0, -32, 1, 33, 100):
            with self.subTest(value=bad):
                with self.assertRaisesRegex(ValueError, "serial_kernel_block_dim"):
                    SolverFeatherPGS(model, serial_kernel_block_dim=bad)
        for good in (32, 64, 128, 256, 512):
            with self.subTest(value=good):
                solver = SolverFeatherPGS(model, serial_kernel_block_dim=good)
                self.assertEqual(solver.serial_kernel_block_dim, good)

    def test_tile_threads_validation(self):
        model = newton.ModelBuilder().finalize()
        for bad in (0, -64, 16, 48, 96, 512):
            with self.subTest(value=bad):
                with self.assertRaisesRegex(ValueError, "tile_threads"):
                    SolverFeatherPGS(model, tile_threads=bad)
        for good in (32, 64, 128, 256):
            with self.subTest(value=good):
                solver = SolverFeatherPGS(model, tile_threads=good)
                self.assertEqual(solver.tile_threads, good)

    @unittest.skipUnless(wp.is_cuda_available(), "tiled launch-config step test requires CUDA")
    def test_non_default_tile_threads_compiles_and_steps(self):
        model = _build_chain_model()
        solver = SolverFeatherPGS(
            model,
            serial_kernel_block_dim=64,
            tile_threads=128,
            cholesky_kernel="tiled",
            trisolve_kernel="tiled",
            hinv_jt_kernel="tiled",
            pgs_mode="dense",
            pgs_kernel="loop",
            dense_max_constraints=384,
        )
        self.assertTrue(all(solver._execution_plan.use_tiled_hinv_jt(size) for size in solver.size_groups))
        self.assertFalse(any(solver._execution_plan.use_fused_hinv_jt(size) for size in solver.size_groups))
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        for _ in range(5):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, None, 1.0 / 600.0)
            state_0, state_1 = state_1, state_0
        wp.synchronize()
        self.assertTrue(np.isfinite(state_0.joint_q.numpy()).all())
        self.assertTrue(np.isfinite(state_0.joint_qd.numpy()).all())

    @unittest.skipUnless(wp.is_cuda_available(), "propagation matrix-free step test requires CUDA")
    def test_propagation_matrix_free_compiles_and_steps_with_velocity_limit_rows(self):
        model = _build_chain_model(num_links=3, num_worlds=1)
        model.joint_velocity_limit.assign(np.full(model.joint_dof_count, 0.1, dtype=np.float32))

        solver = SolverFeatherPGS(
            model,
            pgs_mode="matrix_free",
            articulated_contact_response="propagation",
            enable_joint_velocity_limits=True,
            pgs_iterations=2,
            pgs_velocity_iterations=1,
            dense_max_constraints=16,
            mf_max_constraints=16,
        )
        state_0, state_1 = model.state(), model.state()
        state_0.joint_qd.assign(np.full(model.joint_dof_count, 1.0, dtype=np.float32))
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

        solver.step(state_0, state_1, model.control(), None, 1.0 / 600.0)
        wp.synchronize()

        self.assertGreater(int(solver.constraint_count.numpy()[0]), 0)
        self.assertTrue(np.isfinite(state_1.joint_q.numpy()).all())
        self.assertTrue(np.isfinite(state_1.joint_qd.numpy()).all())

    def test_fuse_joint_velocity_limits_validation(self):
        # fuse_joint_velocity_limits defaults to True but only engages in the
        # matrix_free + physx_pgs + velocity-limits formulation; anywhere else
        # it must be silently inert (no error), so the default doesn't
        # constrain unrelated solver modes.
        model = _build_chain_model(num_links=2, num_worlds=1)
        for kwargs in (
            {},  # neither matrix_free nor velocity limits nor physx drive
            {"pgs_mode": "matrix_free", "drive_mode": "physx_pgs"},  # no velocity limits
        ):
            with self.subTest(kwargs=kwargs):
                solver = SolverFeatherPGS(model, fuse_joint_velocity_limits=True, **kwargs)
                self.assertFalse(solver.fuse_joint_velocity_limits)
        if wp.is_cuda_available():
            # drive_mode='augmented' with velocity limits enabled is inert too
            # (matrix_free construction requires CUDA).
            solver = SolverFeatherPGS(
                model,
                pgs_mode="matrix_free",
                enable_joint_velocity_limits=True,
                drive_mode="augmented",
                fuse_joint_velocity_limits=True,
            )
            self.assertFalse(solver.fuse_joint_velocity_limits)
            # The applicable combination engages by default (no kwarg passed),
            # and an explicit False opts out.
            solver = SolverFeatherPGS(
                model,
                pgs_mode="matrix_free",
                enable_joint_velocity_limits=True,
                drive_mode="physx_pgs",
            )
            self.assertTrue(solver.fuse_joint_velocity_limits)
            solver = SolverFeatherPGS(
                model,
                pgs_mode="matrix_free",
                enable_joint_velocity_limits=True,
                drive_mode="physx_pgs",
                fuse_joint_velocity_limits=False,
            )
            self.assertFalse(solver.fuse_joint_velocity_limits)
            # velocity_limit_activation_fraction=inf is the explicit
            # never-activate kill-switch for velocity limits; the fused clamp
            # rides the drive-row visit (which the inf gate does not cover),
            # so an explicit opt-in must stay inert rather than re-enforce
            # limits the fraction disabled.
            solver = SolverFeatherPGS(
                model,
                pgs_mode="matrix_free",
                enable_joint_velocity_limits=True,
                drive_mode="physx_pgs",
                fuse_joint_velocity_limits=True,
                velocity_limit_activation_fraction=float("inf"),
            )
            self.assertFalse(solver.fuse_joint_velocity_limits)
            # The clamp pass runs at the end of each iteration, independent of
            # the drive-row visit, so frozen drive rows during velocity
            # iterations no longer disable fusion: the clamp keeps enforcing
            # the limit there, exactly like the dedicated rows it replaces.
            for velocity_drive_mode in ("freeze", "active"):
                solver = SolverFeatherPGS(
                    model,
                    pgs_mode="matrix_free",
                    enable_joint_velocity_limits=True,
                    drive_mode="physx_pgs",
                    fuse_joint_velocity_limits=True,
                    pgs_velocity_iterations=2,
                    pgs_velocity_drive_mode=velocity_drive_mode,
                )
                self.assertTrue(solver.fuse_joint_velocity_limits)

    @unittest.skipUnless(wp.is_cuda_available(), "fused velocity-limit clamp requires CUDA")
    def test_fuse_joint_velocity_limits_clamps_driven_dofs_without_rows(self):
        # A strongly driven chain with a low joint velocity limit: with
        # fuse_joint_velocity_limits=True the driven DOFs must (a) stay within
        # the limit plus a small GS-convergence tolerance and (b) use fewer
        # dense rows than the dedicated-row formulation (the two vel-limit
        # rows per driven DOF are replaced by a stateless end-of-iteration
        # clamp).
        num_links = 3
        qdot_max = 1.0

        def run(fuse):
            model = _build_chain_model(num_links=num_links, num_worlds=1)
            n = model.joint_dof_count
            model.joint_target_ke.assign(np.full(n, 200.0, dtype=np.float32))
            model.joint_target_kd.assign(np.full(n, 5.0, dtype=np.float32))
            model.joint_velocity_limit.assign(np.full(n, qdot_max, dtype=np.float32))
            solver = SolverFeatherPGS(
                model,
                pgs_mode="matrix_free",
                drive_mode="physx_pgs",
                enable_joint_velocity_limits=True,
                fuse_joint_velocity_limits=fuse,
                pgs_iterations=64,
                dense_max_constraints=16,
                mf_max_constraints=16,
            )
            state_0, state_1 = model.state(), model.state()
            control = model.control()
            control.joint_target_q.assign(np.full(n, 3.0, dtype=np.float32))
            newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
            max_speed = 0.0
            for _ in range(60):
                state_0.clear_forces()
                solver.step(state_0, state_1, control, None, 1.0 / 60.0)
                state_0, state_1 = state_1, state_0
                max_speed = max(max_speed, float(np.max(np.abs(state_0.joint_qd.numpy()))))
            wp.synchronize()
            return max_speed, int(solver.constraint_count.numpy()[0])

        fused_speed, fused_rows = run(True)
        dedicated_speed, dedicated_rows = run(False)

        # (a) the fused clamp reproduces the dedicated-row behavior it
        # replaces. Both are stateless passes at the end of each iteration,
        # and on a coupled chain both leave the same small GS residual above
        # the limit (clamping DOF i perturbs DOF j through the tree response,
        # and the sweep visits each DOF once per iteration) — dedicated rows
        # measure ~1.12x the limit here. Assert against that reference, not
        # an exact limit no last-word formulation achieves on this scene.
        self.assertLessEqual(fused_speed, dedicated_speed * 1.05)
        self.assertLessEqual(fused_speed, qdot_max * 1.25)
        # (b) the fused solve drops the two dedicated vel-limit rows per
        # driven DOF: 3 drive rows vs 3 drive + 6 vel-limit rows.
        self.assertGreater(fused_rows, 0)
        self.assertLess(fused_rows, dedicated_rows)
        self.assertEqual(dedicated_rows - fused_rows, 2 * num_links)


if __name__ == "__main__":
    unittest.main(verbosity=2)
