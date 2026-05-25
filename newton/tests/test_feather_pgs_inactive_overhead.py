# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Focused regressions for inactive FeatherPGS matrix-free/debug overhead."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers import SolverFeatherPGS


def _device():
    device = wp.get_device()
    if device is None:
        raise unittest.SkipTest("No Warp device available")
    return device


def _build_articulated_only_model(device) -> newton.Model:
    builder = newton.ModelBuilder()
    link_a = builder.add_link()
    link_b = builder.add_link()
    joint_a = builder.add_joint_revolute(
        parent=-1,
        child=link_a,
        axis=wp.vec3(0.0, 0.0, 1.0),
    )
    joint_b = builder.add_joint_revolute(
        parent=link_a,
        child=link_b,
        axis=wp.vec3(0.0, 1.0, 0.0),
    )
    builder.add_articulation([joint_a, joint_b])
    return builder.finalize(device=device)


def _build_mixed_free_rigid_contact_model(device) -> newton.Model:
    builder = newton.ModelBuilder()

    link = builder.add_link()
    joint = builder.add_joint_revolute(
        parent=-1,
        child=link,
        axis=wp.vec3(0.0, 0.0, 1.0),
    )
    builder.add_articulation([joint])

    sphere = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.12), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1),
    )
    builder.add_shape_sphere(sphere, radius=0.2)
    builder.add_ground_plane()

    return builder.finalize(device=device)


def _step_once(model: newton.Model, solver: SolverFeatherPGS) -> None:
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.collide(state_0)
    state_0.clear_forces()
    solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)


class TestFeatherPGSInactiveOverhead(unittest.TestCase):
    def test_matrix_free_articulated_only_skips_free_rigid_mf_buffers(self):
        model = _build_articulated_only_model(_device())
        solver = SolverFeatherPGS(model, pgs_mode="matrix_free", pgs_iterations=2)

        self.assertFalse(solver._has_free_rigid_bodies)
        self.assertFalse(solver._mf_rows_active)
        self.assertFalse(hasattr(solver, "mf_constraint_count"))
        self.assertIsNotNone(solver._debug_stage3_v_hat)
        self.assertIsNotNone(solver._debug_position_constraint_count)

        _step_once(model, solver)

    def test_mixed_articulated_free_rigid_contact_keeps_active_mf_rows(self):
        model = _build_mixed_free_rigid_contact_model(_device())
        solver = SolverFeatherPGS(model, pgs_mode="matrix_free", pgs_iterations=2)

        self.assertTrue(solver._has_free_rigid_bodies)
        self.assertTrue(solver._mf_rows_active)
        self.assertTrue(hasattr(solver, "mf_constraint_count"))

        _step_once(model, solver)
        self.assertGreater(int(np.sum(solver.mf_constraint_count.numpy())), 0)

    def test_zero_velocity_iterations_does_not_run_velocity_post_solve(self):
        model = _build_articulated_only_model(_device())
        solver = SolverFeatherPGS(
            model,
            pgs_mode="matrix_free",
            pgs_iterations=2,
            pgs_velocity_iterations=0,
        )

        def fail_if_called():
            raise AssertionError("velocity post-solve should be skipped when pgs_velocity_iterations=0")

        solver._run_matrix_free_velocity_post_solve = fail_if_called
        _step_once(model, solver)

    def test_pgs_debug_retains_diagnostics(self):
        model = _build_mixed_free_rigid_contact_model(_device())
        solver = SolverFeatherPGS(
            model,
            pgs_mode="matrix_free",
            pgs_iterations=2,
            pgs_debug=True,
        )

        self.assertIsNotNone(solver._debug_stage3_v_hat)
        self.assertIsNotNone(solver._debug_position_constraint_count)
        self.assertIsNotNone(solver._debug_position_mf_constraint_count)

        _step_once(model, solver)

        self.assertEqual(len(solver._pgs_convergence_log), 1)
        self.assertEqual(solver._pgs_convergence_log[0].shape, (solver.pgs_iterations, 4))
        self.assertEqual(len(solver._pgs_ncp_residual_log), 1)
        self.assertEqual(solver._pgs_ncp_residual_log[0].shape[0], solver.pgs_iterations)
        self.assertEqual(solver._pgs_ncp_residual_log[0].shape[2], 6)


if __name__ == "__main__":
    unittest.main()
