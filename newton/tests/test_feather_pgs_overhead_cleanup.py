# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Focused regressions for FeatherPGS inactive-path overhead cleanup."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _build_articulated_only_model(device: str) -> newton.Model:
    builder = newton.ModelBuilder(gravity=0.0)
    link = builder.add_link()
    joint = builder.add_joint_revolute(parent=-1, child=link, axis=wp.vec3(0.0, 0.0, 1.0))
    builder.add_articulation([joint])
    return builder.finalize(device=device)


def _build_mixed_free_rigid_contact_model(device: str) -> newton.Model:
    builder = newton.ModelBuilder(gravity=-9.81)

    link = builder.add_link()
    joint = builder.add_joint_revolute(parent=-1, child=link, axis=wp.vec3(0.0, 0.0, 1.0))
    builder.add_articulation([joint])

    sphere = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.15), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1),
    )
    builder.add_shape_sphere(sphere, radius=0.2)
    builder.add_ground_plane()
    return builder.finalize(device=device)


def _step_once(model: newton.Model, solver: SolverFeatherPGS, dt: float = 1.0 / 120.0) -> None:
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.collide(state_0)
    state_0.clear_forces()
    solver.step(state_0, state_1, control, contacts, dt)


class TestFeatherPGSOverheadCleanup(unittest.TestCase):
    pass


def run_matrix_free_articulated_only_skips_free_rigid_mf_work(test: TestFeatherPGSOverheadCleanup, device):
    model = _build_articulated_only_model(device)
    solver = SolverFeatherPGS(model, pgs_mode="matrix_free", pgs_iterations=2)

    test.assertFalse(solver._has_free_rigid_bodies)
    test.assertFalse(solver._has_full_mf_buffers)
    test.assertFalse(hasattr(solver, "mf_slot_counter"))
    test.assertIsNone(solver.drive_slot)
    test.assertEqual(solver.mf_rhs.shape, (solver.world_count, 1))

    _step_once(model, solver)

    np.testing.assert_array_equal(solver.mf_constraint_count.numpy(), np.zeros(solver.world_count, dtype=np.int32))


def run_mixed_free_rigid_scene_keeps_active_mf_rows(test: TestFeatherPGSOverheadCleanup, device):
    model = _build_mixed_free_rigid_contact_model(device)
    solver = SolverFeatherPGS(model, pgs_mode="matrix_free", pgs_iterations=2)

    test.assertTrue(solver._has_free_rigid_bodies)
    test.assertTrue(solver._has_full_mf_buffers)
    test.assertTrue(hasattr(solver, "mf_slot_counter"))
    test.assertIsNone(solver.rigid_velocity_limit_slot)
    test.assertEqual(solver.mf_rhs.shape, (solver.world_count, solver.mf_max_constraints))

    _step_once(model, solver)

    test.assertGreater(int(np.max(solver.mf_constraint_count.numpy())), 0)


def run_zero_velocity_iterations_skip_post_solve(test: TestFeatherPGSOverheadCleanup, device):
    model = _build_articulated_only_model(device)
    solver = SolverFeatherPGS(model, pgs_mode="matrix_free", pgs_iterations=1, pgs_velocity_iterations=0)
    calls = []

    def counted_velocity_post_solve():
        calls.append(1)

    solver._run_matrix_free_velocity_post_solve = counted_velocity_post_solve

    _step_once(model, solver)

    test.assertEqual(calls, [])


def run_pgs_debug_retains_supported_diagnostics(test: TestFeatherPGSOverheadCleanup, device):
    model = _build_articulated_only_model(device)
    solver = SolverFeatherPGS(model, pgs_mode="matrix_free", pgs_iterations=2, pgs_debug=True)

    _step_once(model, solver)

    test.assertFalse(hasattr(solver, "_debug_stage3_v_hat"))
    test.assertFalse(hasattr(solver, "_debug_position_rhs"))
    test.assertEqual(len(solver._pgs_convergence_log), 1)
    test.assertEqual(solver._pgs_convergence_log[0].shape, (2, 4))
    test.assertEqual(len(solver._pgs_ncp_residual_log), 1)
    test.assertEqual(solver._pgs_ncp_residual_log[0].shape, (2, solver.world_count, 6))
    test.assertTrue(np.all(np.isfinite(solver._pgs_convergence_log[0])))
    test.assertTrue(np.all(np.isfinite(solver._pgs_ncp_residual_log[0])))


devices = get_test_devices()
for device in devices:
    add_function_test(
        TestFeatherPGSOverheadCleanup,
        "test_matrix_free_articulated_only_skips_free_rigid_mf_work",
        run_matrix_free_articulated_only_skips_free_rigid_mf_work,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSOverheadCleanup,
        "test_mixed_free_rigid_scene_keeps_active_mf_rows",
        run_mixed_free_rigid_scene_keeps_active_mf_rows,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSOverheadCleanup,
        "test_zero_velocity_iterations_skip_post_solve",
        run_zero_velocity_iterations_skip_post_solve,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSOverheadCleanup,
        "test_pgs_debug_retains_supported_diagnostics",
        run_pgs_debug_retains_supported_diagnostics,
        devices=[device],
    )


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=False)
