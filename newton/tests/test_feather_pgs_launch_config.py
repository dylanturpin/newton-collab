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
        import numpy as np  # noqa: PLC0415

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
