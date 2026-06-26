# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.kernels import (
    accumulate_deferred_dense_tau_from_J,
    accumulate_deferred_dense_tau_from_J_world,
    compute_dense_impulse_delta,
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


class TestFeatherPGSLaunchConfig(unittest.TestCase):
    def test_defaults_preserved(self):
        model = newton.ModelBuilder().finalize()
        solver = SolverFeatherPGS(model)
        self.assertEqual(solver.serial_kernel_block_dim, 256)
        self.assertEqual(solver.tile_threads, 64)
        self.assertEqual(solver.articulated_dense_response_mode, "immediate")

    def test_articulated_dense_response_mode_validation(self):
        model = _build_chain_model(num_links=2, num_worlds=1)
        solver = SolverFeatherPGS(
            model,
            pgs_mode="matrix_free",
            articulated_dense_response_mode="deferred_cholesky",
        )
        self.assertEqual(solver.articulated_dense_response_mode, "deferred_cholesky")
        compact = SolverFeatherPGS(
            model,
            pgs_mode="matrix_free",
            articulated_dense_response_mode="compact_tree",
        )
        self.assertEqual(compact.articulated_dense_response_mode, "compact_tree")
        compact_alias = SolverFeatherPGS(
            model,
            pgs_mode="matrix_free",
            articulated_dense_response_mode="compact_cholesky",
        )
        self.assertEqual(compact_alias.articulated_dense_response_mode, "compact_cholesky")

        with self.assertRaisesRegex(ValueError, "articulated_dense_response_mode"):
            SolverFeatherPGS(model, articulated_dense_response_mode="bad")

        with self.assertRaisesRegex(NotImplementedError, "pgs_mode='matrix_free'"):
            SolverFeatherPGS(model, pgs_mode="split", articulated_dense_response_mode="deferred_cholesky")
        with self.assertRaisesRegex(NotImplementedError, "pgs_mode='matrix_free'"):
            SolverFeatherPGS(model, pgs_mode="split", articulated_dense_response_mode="compact_cholesky")
        with self.assertRaisesRegex(NotImplementedError, "pgs_mode='matrix_free'"):
            SolverFeatherPGS(model, pgs_mode="split", articulated_dense_response_mode="compact_tree")

        with self.assertRaisesRegex(NotImplementedError, "friction_mode='current'"):
            SolverFeatherPGS(
                model,
                pgs_mode="matrix_free",
                articulated_dense_response_mode="compact_tree",
                friction_mode="bisection",
            )
        compact_debug = SolverFeatherPGS(
            model,
            pgs_mode="matrix_free",
            articulated_dense_response_mode="compact_tree",
            pgs_debug=True,
        )
        self.assertEqual(compact_debug.articulated_dense_response_mode, "compact_tree")

    def test_deferred_cholesky_accepts_matrix_free_friction_modes(self):
        model = _build_chain_model(num_links=2, num_worlds=1)
        for friction_mode in ("current", "bisection", "bisection_desaxce", "coulomb_newton"):
            with self.subTest(friction_mode=friction_mode):
                solver = SolverFeatherPGS(
                    model,
                    pgs_mode="matrix_free",
                    articulated_dense_response_mode="deferred_cholesky",
                    friction_mode=friction_mode,
                )
                self.assertEqual(solver.friction_mode, friction_mode)

    def test_deferred_dense_tau_kernels_accumulate_j_transpose_delta(self):
        device = "cpu"
        constraint_count = wp.array([2], dtype=wp.int32, device=device)
        impulses = wp.array([[1.0, -2.0, 7.0]], dtype=wp.float32, device=device)
        prev = wp.array([[0.25, -3.0, 100.0]], dtype=wp.float32, device=device)
        delta = wp.zeros((1, 3), dtype=wp.float32, device=device)

        wp.launch(
            compute_dense_impulse_delta,
            dim=(1, 3),
            inputs=[constraint_count, 3, impulses, prev],
            outputs=[delta],
            device=device,
        )

        np.testing.assert_allclose(delta.numpy(), [[0.75, 1.0, 0.0]], rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(prev.numpy(), [[1.0, -2.0, 0.0]], rtol=0.0, atol=1.0e-6)

        group_to_art = wp.array([0], dtype=wp.int32, device=device)
        art_to_world = wp.array([0], dtype=wp.int32, device=device)
        articulation_dof_start = wp.array([0], dtype=wp.int32, device=device)
        J_group = wp.array(
            [[[2.0, 3.0], [4.0, 5.0], [99.0, 99.0]]],
            dtype=wp.float32,
            device=device,
        )
        tau = wp.zeros((2,), dtype=wp.float32, device=device)
        wp.launch(
            accumulate_deferred_dense_tau_from_J,
            dim=1 * 3 * 2,
            inputs=[
                group_to_art,
                art_to_world,
                articulation_dof_start,
                constraint_count,
                delta,
                J_group,
                2,
                3,
                1,
            ],
            outputs=[tau],
            device=device,
        )

        np.testing.assert_allclose(tau.numpy(), [5.5, 7.25], rtol=0.0, atol=1.0e-6)

        world_dof_start = wp.array([0], dtype=wp.int32, device=device)
        world_dof_count = wp.array([2], dtype=wp.int32, device=device)
        world_deferred_dof_mask = wp.array([[1, 1]], dtype=wp.int32, device=device)
        J_world = wp.array(
            [[[2.0, 3.0], [4.0, 5.0], [99.0, 99.0]]],
            dtype=wp.float32,
            device=device,
        )
        tau_world = wp.zeros((2,), dtype=wp.float32, device=device)
        wp.launch(
            accumulate_deferred_dense_tau_from_J_world,
            dim=1 * 3 * 2,
            inputs=[
                world_dof_start,
                world_dof_count,
                world_deferred_dof_mask,
                constraint_count,
                delta,
                J_world,
                2,
                3,
            ],
            outputs=[tau_world],
            device=device,
        )

        np.testing.assert_allclose(tau_world.numpy(), [5.5, 7.25], rtol=0.0, atol=1.0e-6)

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
        )
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        for _ in range(5):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, None, 1.0 / 600.0)
            state_0, state_1 = state_1, state_0
        wp.synchronize()
        self.assertTrue(np.isfinite(state_0.joint_q.numpy()).all())
        self.assertTrue(np.isfinite(state_0.joint_qd.numpy()).all())

    @unittest.skipUnless(wp.is_cuda_available(), "deferred Cholesky matrix-free step test requires CUDA")
    def test_deferred_cholesky_matrix_free_compiles_and_steps_with_dense_rows(self):
        model = _build_chain_model(num_links=3, num_worlds=1)
        model.joint_velocity_limit.assign(np.full(model.joint_dof_count, 0.1, dtype=np.float32))

        solver = SolverFeatherPGS(
            model,
            pgs_mode="matrix_free",
            articulated_dense_response_mode="deferred_cholesky",
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
