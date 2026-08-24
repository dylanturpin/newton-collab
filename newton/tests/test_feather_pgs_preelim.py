# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for bilateral (mimic + connect) pre-elimination in SolverFeatherPGS."""

import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_CONNECT

from .test_feather_pgs_connect import _build_four_bar, _loop_anchor_gap


def _run_four_bar(steps: int = 720, crank_target: float = 0.6, pgs_iterations: int = 2, **solver_kwargs):
    """Drive the connect-test four-bar and track the worst loop-anchor gap.

    Deliberately runs at a LOW iteration count: with iterative closure rows the
    anchor gap is convergence-limited (measured ~0.5 mm at 2 iterations), while
    pre-elimination enforces the closure independently of the sweep budget.
    """
    model = _build_four_bar().finalize()
    solver = newton.solvers.SolverFeatherPGS(
        model, pgs_mode="matrix_free", pgs_iterations=pgs_iterations, pgs_beta=0.1, **solver_kwargs
    )
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    targets = model.joint_target_q.numpy().copy()
    targets[0] = crank_target
    control.joint_target_q.assign(targets)
    gap_max = 0.0
    for _ in range(steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0
        gap_max = max(gap_max, _loop_anchor_gap(model, state_0))
    return solver, model, state_0, gap_max


@unittest.skipUnless(wp.get_device().is_cuda, "SolverFeatherPGS matrix-free mode requires CUDA")
class TestFeatherPGSPreelimination(unittest.TestCase):
    def test_closure_exact_at_low_iterations(self):
        """Pre-elimination makes the loop closure iteration-independent.

        At 2 PGS iterations the iterative CONNECT rows cannot converge (the
        anchor gap plateaus near half a millimetre); the pre-eliminated solve
        must hold the closure 100x tighter through the same driven stroke.
        This is the regression-first differential: without the feature the
        ratio is 1 and the assertions below fail.
        """
        solver_off, _, _state_off, gap_off = _run_four_bar()
        self.assertFalse(solver_off._preelim_active)
        solver_on, _, state_on, gap_on = _run_four_bar(enable_bilateral_preelimination=True)
        self.assertTrue(solver_on._preelim_active)
        self.assertEqual(solver_on._preelim_count, 1)

        self.assertTrue(np.isfinite(state_on.body_q.numpy()).all())
        self.assertGreater(gap_off, 1.0e-4, "iterative baseline unexpectedly tight; differential is meaningless")
        self.assertLess(gap_on, 2.0e-5, f"pre-eliminated closure gap {gap_on:.2e} m not exact")
        self.assertLess(gap_on, 0.01 * gap_off, f"expected >100x tightening, got {gap_off / max(gap_on, 1e-12):.1f}x")

    def test_dense_warmstart_preserves_projected_closure(self):
        """Project the predictor after installing dense warm-start impulses.

        Seed a nonempty cache, reverse the drive, and disable the iterative
        sweeps for one step. This isolates the stage-6 initializer: moving the
        bilateral projection before warm-start application lets the latter
        overwrite it and opens the loop by millimetres.
        """
        model = _build_four_bar().finalize()
        solver = newton.solvers.SolverFeatherPGS(
            model,
            pgs_mode="matrix_free",
            pgs_iterations=2,
            pgs_beta=0.1,
            enable_bilateral_preelimination=True,
            pgs_warmstart=True,
        )
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        targets = model.joint_target_q.numpy().copy()
        targets[0] = 0.6
        control.joint_target_q.assign(targets)
        for _ in range(120):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, None, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0

        cache_peak = np.max(np.abs(solver.impulses.numpy()))
        targets[0] = -0.6
        control.joint_target_q.assign(targets)
        solver.pgs_iterations = 0
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0
        gap = _loop_anchor_gap(model, state_0)

        self.assertTrue(solver._preelim_active)
        self.assertTrue(solver.pgs_warmstart)
        self.assertTrue(np.isfinite(state_0.body_q.numpy()).all())
        self.assertGreater(cache_peak, 1.0e-4, "warm-start cache stayed empty")
        self.assertLess(gap, 2.0e-5, f"warm-started projection left a {gap:.2e} m closure gap")

    def test_mechanism_behavior_preserved(self):
        """The rocker still tracks the crank through the closed loop (parallel four-bar)."""
        _, _, state, _ = _run_four_bar(enable_bilateral_preelimination=True, pgs_iterations=16)
        q = state.joint_q.numpy()
        self.assertAlmostEqual(q[2], q[0], delta=0.08)
        self.assertGreater(abs(q[2]), 0.3, "rocker did not move — loop not transmitting")

    def test_rows_remain_allocated(self):
        """B rows keep their dense slots (layout/warm-start stability); they are neutralized, not removed."""
        solver, _, _, _ = _run_four_bar(steps=60, enable_bilateral_preelimination=True)
        counts = solver.constraint_count.numpy()
        rows = solver.row_type.numpy()
        seen = set()
        for w in range(rows.shape[0]):
            seen.update(rows[w, : counts[w]].tolist())
        self.assertIn(int(PGS_CONSTRAINT_TYPE_CONNECT), seen)

    def test_unsupported_mode_warns_and_falls_back(self):
        """Non-matrix-free modes warn and keep the iterative rows."""
        model = _build_four_bar().finalize()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            solver = newton.solvers.SolverFeatherPGS(
                model, pgs_mode="dense", pgs_iterations=16, enable_bilateral_preelimination=True
            )
        self.assertFalse(solver._preelim_active)
        self.assertTrue(any("matrix_free" in str(w.message) for w in caught))

    def test_velocity_iterations_warn_and_fall_back(self):
        """The TGS velocity pass is unsupported in v1: warn and fall back."""
        model = _build_four_bar().finalize()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            solver = newton.solvers.SolverFeatherPGS(
                model,
                pgs_mode="matrix_free",
                pgs_iterations=16,
                pgs_velocity_iterations=4,
                enable_bilateral_preelimination=True,
            )
        self.assertFalse(solver._preelim_active)
        self.assertTrue(any("velocity_iterations" in str(w.message) for w in caught))


if __name__ == "__main__":
    unittest.main()
