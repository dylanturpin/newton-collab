# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import warp as wp

import newton
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
        import numpy as np  # noqa: PLC0415

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
