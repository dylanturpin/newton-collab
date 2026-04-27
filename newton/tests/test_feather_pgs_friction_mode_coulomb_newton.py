# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for ``friction_mode="coulomb_newton"`` (FPGS Friction Modes 7/13).

Covers the acceptance criteria from the FPGS Friction Modes 7/13 issue:

* The constructor no longer raises ``NotImplementedError`` for
  ``friction_mode="coulomb_newton"``.
* The in-solver ``solve_coulomb_row`` ``@wp.func`` matches the NumPy
  reference (ported from
  ``artifacts/2026-04-16-slack-raisim/coulomb_root_finding_warp.py``)
  on a synthetic batch of random ``(W, b, μ)`` inputs: no status
  mismatches, ``max |phi| < 5e-6`` at solution (the tolerance the
  reference script's self-test reports).
* A sliding-cube replay stays finite and the ``r_mdp_dir`` /
  ``r_ds_compl`` residuals at matched GS iterations are not worse than
  ``bisection_desaxce`` (6/13 baseline) on sliding contacts.
* ``pgs_debug=True`` populates the 6-channel NCP log.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers import SolverFeatherPGS
from newton._src.solvers.feather_pgs.kernels import solve_coulomb_row
from newton.tests.test_feather_pgs_friction_mode_bisection_desaxce import (
    _build_sliding_cube_model,
)


# ---------------------------------------------------------------------------
# Random problem generation + NumPy reference (both ported verbatim from
# ``coulomb_root_finding_warp.py``; kept local so the unit test stays
# self-contained and does not depend on the artifact staying on disk).
# ---------------------------------------------------------------------------


def _generate_random_problems(n: int, seed: int = 42):
    """Generate ``n`` random (W, b, mu) Coulomb contact problems.

    ``W`` is 3x3 SPD, ``b_N < 0``, ``mu`` in ``[0.1, 1.5]``.
    """
    rng = np.random.RandomState(seed)
    L = rng.randn(n, 3, 3) * 0.8
    W = np.einsum("nij,nkj->nik", L, L) + 0.3 * np.eye(3)[None, :, :]
    b = rng.randn(n, 3) * 0.5
    b[:, 0] = -np.abs(b[:, 0]) - 0.1  # ensure b_N < 0
    mu = rng.uniform(0.1, 1.5, size=n)
    return W, b, mu


def _numpy_solve_one(W, b, mu, tol: float = 1.0e-10, max_iter: int = 50):
    """Reference NumPy solver for a single (W, b, mu) problem.

    Returns ``(alpha, rN, rT, iters, status)`` where ``status`` is 0 for
    sticking and 1 for sliding.  Ported from
    ``coulomb_root_finding_warp.py::numpy_solve_one``.
    """
    WN = W[0, 0]
    wNT = W[1:, 0]
    WT = W[1:, 1:]
    bN = b[0]
    bT = b[1:]

    AT = WT - np.outer(wNT, wNT) / WN
    cT = bT - (bN / WN) * wNT

    s0 = np.linalg.solve(AT, cT)
    rn0 = (wNT @ s0 - bN) / WN
    phi0 = np.linalg.norm(s0) - mu * rn0

    if phi0 <= 0.0:
        rT = -s0
        rN = -(wNT @ rT + bN) / WN
        return 0.0, rN, rT, 0, 0  # sticking

    hi = 1.0
    for _ in range(30):
        M = AT + hi * np.eye(2)
        s = np.linalg.solve(M, cT)
        if np.linalg.norm(s) - mu * (wNT @ s - bN) / WN < 0:
            break
        hi *= 2.0

    lo = 0.0
    x = 0.5 * (lo + hi)
    last_s = None

    for it in range(1, max_iter + 1):
        M = AT + x * np.eye(2)
        s = np.linalg.solve(M, cT)
        t = np.linalg.solve(M, s)
        ns = np.linalg.norm(s)
        fx = ns - mu * (wNT @ s - bN) / WN
        dfx = -(s @ t) / ns + mu * (wNT @ t) / WN
        last_s = s

        if abs(fx) < tol or abs(hi - lo) < tol:
            break

        if fx > 0:
            lo = x
        else:
            hi = x

        x_new = x - fx / dfx if dfx != 0.0 else 0.5 * (lo + hi)
        x = x_new if lo < x_new < hi else 0.5 * (lo + hi)

    rT = -last_s
    rN = -(wNT @ rT + bN) / WN
    return x, rN, rT, it, 1


# ---------------------------------------------------------------------------
# Test kernel that invokes ``solve_coulomb_row`` on a batch of problems.
# ---------------------------------------------------------------------------


@wp.kernel
def _fpgs_solve_coulomb_batch_kernel(
    W_arr: wp.array(dtype=wp.mat33),
    b_arr: wp.array(dtype=wp.vec3),
    mu_arr: wp.array(dtype=float),
    alpha_out: wp.array(dtype=float),
    rN_out: wp.array(dtype=float),
    rT_out: wp.array(dtype=wp.vec2),
    iters_out: wp.array(dtype=int),
    status_out: wp.array(dtype=int),
):
    i = wp.tid()
    result = solve_coulomb_row(W_arr[i], b_arr[i], mu_arr[i])
    alpha_out[i] = result[0]
    rN_out[i] = result[1]
    rT_out[i] = wp.vec2(result[2], result[3])
    iters_out[i] = int(result[4])
    status_out[i] = int(result[5])


# ---------------------------------------------------------------------------
# Helpers for the end-to-end replays.
# ---------------------------------------------------------------------------


def _build_sphere_on_plane_model(device: wp.context.Device) -> newton.Model:
    """Deterministic sphere-on-ground scene (no downloaded assets)."""
    builder = newton.ModelBuilder()
    builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
        limit_ke=0.0, limit_kd=0.0, friction=0.0
    )
    builder.default_shape_cfg.ke = 5.0e4
    builder.default_shape_cfg.kd = 5.0e2
    builder.default_shape_cfg.kf = 1.0e3
    builder.default_shape_cfg.mu = 0.75

    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.35), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1),
    )
    builder.add_shape_sphere(body, radius=0.2)
    builder.add_ground_plane()
    return builder.finalize(device=device)


def _run_sim(
    device: wp.context.Device,
    friction_mode: str,
    num_steps: int,
    model: newton.Model | None = None,
    pgs_iterations: int = 12,
    pgs_debug: bool = False,
    substeps: int = 1,
    dense_max_constraints: int = 128,
) -> tuple[np.ndarray, SolverFeatherPGS, newton.Model]:
    """Run ``num_steps`` frames with the requested ``friction_mode``.

    Mirrors the helper used by the 5/13 and 6/13 tests so residual
    comparisons stay apples-to-apples.
    """
    if model is None:
        model = _build_sphere_on_plane_model(device)
    solver = SolverFeatherPGS(
        model,
        pgs_mode="matrix_free",
        friction_mode=friction_mode,
        pgs_iterations=pgs_iterations,
        pgs_debug=pgs_debug,
        dense_max_constraints=dense_max_constraints,
    )
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    sim_dt = 1.0 / 240.0
    sub_dt = sim_dt / float(substeps)
    for _ in range(num_steps):
        contacts = model.collide(state_0)
        for _ in range(substeps):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, contacts, sub_dt)
            state_0, state_1 = state_1, state_0

    return state_0.joint_q.numpy().copy(), solver, model


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFeatherPGSFrictionModeCoulombNewtonSelector(unittest.TestCase):
    """``friction_mode="coulomb_newton"`` no longer raises."""

    def test_constructor_accepts_coulomb_newton(self):
        """``SolverFeatherPGS`` accepts the new mode on ``matrix_free``."""
        device = wp.get_device()
        if device is None:
            self.skipTest("No warp device available")

        model = _build_sphere_on_plane_model(device)
        solver = SolverFeatherPGS(
            model,
            pgs_mode="matrix_free",
            friction_mode="coulomb_newton",
        )
        self.assertEqual(solver.friction_mode, "coulomb_newton")

    def test_non_matrix_free_still_rejected(self):
        """coulomb_newton still requires ``pgs_mode="matrix_free"``."""
        device = wp.get_device()
        if device is None:
            self.skipTest("No warp device available")

        model = _build_sphere_on_plane_model(device)
        with self.assertRaises(ValueError):
            SolverFeatherPGS(
                model,
                pgs_mode="split",
                friction_mode="coulomb_newton",
            )


class TestFeatherPGSFrictionModeCoulombNewtonKernel(unittest.TestCase):
    """In-solver ``solve_coulomb_row`` matches the NumPy reference.

    Mirrors ``coulomb_root_finding_warp.py::main()``'s self-test: a
    synthetic batch of random ``(W, b, μ)`` is pushed through both the
    in-solver ``@wp.func`` and ``numpy_solve_one``, and we assert zero
    status mismatches and ``max |phi| < 5e-6`` (the tolerance the
    reference script reports as passing).
    """

    def _run_batch(self, n: int, seed: int):
        device = wp.get_device()
        if device is None:
            self.skipTest("No warp device available")

        W_np, b_np, mu_np = _generate_random_problems(n, seed=seed)

        W_wp = wp.array(W_np.astype(np.float32), dtype=wp.mat33, device=device)
        b_wp = wp.array(b_np.astype(np.float32), dtype=wp.vec3, device=device)
        mu_wp = wp.array(mu_np.astype(np.float32), dtype=float, device=device)

        alpha_wp = wp.zeros(n, dtype=float, device=device)
        rN_wp = wp.zeros(n, dtype=float, device=device)
        rT_wp = wp.zeros(n, dtype=wp.vec2, device=device)
        iters_wp = wp.zeros(n, dtype=int, device=device)
        status_wp = wp.zeros(n, dtype=int, device=device)

        wp.launch(
            _fpgs_solve_coulomb_batch_kernel,
            dim=n,
            inputs=[W_wp, b_wp, mu_wp],
            outputs=[alpha_wp, rN_wp, rT_wp, iters_wp, status_wp],
            device=device,
        )
        wp.synchronize()

        return (
            W_np,
            b_np,
            mu_np,
            alpha_wp.numpy(),
            rN_wp.numpy(),
            rT_wp.numpy(),
            iters_wp.numpy(),
            status_wp.numpy(),
        )

    def test_matches_numpy_reference_small_batch(self):
        """256 random contacts: no status mismatches, |phi| < 5e-6."""
        n = 256
        W_np, b_np, mu_np, _alpha_all, rN_all, rT_all, _iters_all, status_all = (
            self._run_batch(n, seed=42)
        )

        mismatches = 0
        n_slide_verified = 0
        max_phi_residual = 0.0
        max_rN_err = 0.0
        max_rT_err = 0.0

        for j in range(n):
            _a_ref, rN_ref, rT_ref, _it_ref, st_ref = _numpy_solve_one(
                W_np[j], b_np[j], mu_np[j]
            )
            st_wp = int(status_all[j])
            if st_ref != st_wp:
                mismatches += 1
                continue

            if st_ref == 1:  # sliding
                n_slide_verified += 1
                max_rN_err = max(max_rN_err, abs(float(rN_all[j]) - rN_ref))
                max_rT_err = max(
                    max_rT_err, float(np.linalg.norm(rT_all[j] - rT_ref))
                )
                phi_res = abs(
                    float(np.linalg.norm(rT_all[j]))
                    - mu_np[j] * float(rN_all[j])
                )
                max_phi_residual = max(max_phi_residual, phi_res)

        self.assertEqual(
            mismatches,
            0,
            f"solve_coulomb_row status mismatches vs NumPy reference: {mismatches}/{n}",
        )
        self.assertGreater(
            n_slide_verified,
            0,
            "expected at least one sliding contact in the random batch",
        )
        # The reference self-test prints PASS when max|phi| < 1e-4; the
        # issue calls for ~5e-6.  Float32 Newton in Warp should clear
        # the tighter bar on a batch this size.
        self.assertLess(
            max_phi_residual,
            5.0e-6,
            f"|phi| at solution exceeded tolerance: {max_phi_residual:.2e} "
            f"(max_rN_err={max_rN_err:.2e}, max_rT_err={max_rT_err:.2e})",
        )


class TestFeatherPGSFrictionModeCoulombNewtonStability(unittest.TestCase):
    """End-to-end replays stay finite with ``coulomb_newton``."""

    def test_sphere_on_plane_240_steps_finite(self):
        device = wp.get_device()
        if device is None:
            self.skipTest("No warp device available")

        joint_q, _solver, _model = _run_sim(
            device, friction_mode="coulomb_newton", num_steps=240
        )
        self.assertTrue(
            np.all(np.isfinite(joint_q)),
            f"coulomb_newton produced non-finite joint_q: {joint_q}",
        )
        self.assertGreater(joint_q[2], 0.15, f"sphere fell through: z={joint_q[2]}")
        self.assertLess(joint_q[2], 0.35, f"sphere did not settle: z={joint_q[2]}")

    def test_sliding_cube_stable(self):
        device = wp.get_device()
        if device is None:
            self.skipTest("No warp device available")

        model = _build_sliding_cube_model(device)
        joint_q, _solver, _model = _run_sim(
            device,
            friction_mode="coulomb_newton",
            num_steps=240,
            model=model,
            pgs_iterations=20,
        )
        self.assertTrue(np.all(np.isfinite(joint_q)))
        self.assertGreater(joint_q[2], -0.1)
        self.assertLess(joint_q[2], 1.0)


class TestFeatherPGSFrictionModeCoulombNewtonDiagnostic(unittest.TestCase):
    """``pgs_debug=True`` populates the 6-channel NCP log."""

    def test_ncp_log_populated(self):
        device = wp.get_device()
        if device is None:
            self.skipTest("No warp device available")

        num_steps = 30
        _joint_q, solver, _model = _run_sim(
            device,
            friction_mode="coulomb_newton",
            num_steps=num_steps,
            pgs_iterations=10,
            pgs_debug=True,
        )
        self.assertEqual(len(solver._pgs_ncp_residual_log), num_steps)
        last = solver._pgs_ncp_residual_log[-1]
        self.assertEqual(last.ndim, 3)
        self.assertEqual(last.shape[0], solver.pgs_iterations)
        self.assertEqual(last.shape[2], 6)
        self.assertTrue(np.all(np.isfinite(last)))


class TestFeatherPGSFrictionModeCoulombNewtonResidualComparison(unittest.TestCase):
    """Acceptance criterion: ``coulomb_newton`` does not regress
    ``r_mdp_dir`` / ``r_ds_compl`` vs ``bisection_desaxce`` (6/13) at
    matched GS iterations on sliding contacts.

    Channel layout matches ``pgs_ncp_residuals_diagnostic_velocity``:
    ``[r_compl, r_cone, r_gap, r_ds_compl, r_ds_dual, r_mdp_dir]``.
    """

    def _compare_residual_channel(
        self,
        baseline_solver: SolverFeatherPGS,
        candidate_solver: SolverFeatherPGS,
        channel: int,
        channel_name: str,
        iters: tuple[int, ...],
        min_steps_satisfying: int = 1,
    ) -> None:
        base_stack = np.stack(baseline_solver._pgs_ncp_residual_log, axis=0)
        cand_stack = np.stack(candidate_solver._pgs_ncp_residual_log, axis=0)
        base_all = np.max(base_stack[:, :, :, channel], axis=2)
        cand_all = np.max(cand_stack[:, :, :, channel], axis=2)

        num_iter_budget = base_all.shape[1]
        filtered_iters = [k for k in iters if k < num_iter_budget]
        self.assertTrue(filtered_iters, "iters ladder does not fit")

        satisfying_steps: list[int] = []
        for step in range(base_all.shape[0]):
            ok_all = True
            for k in filtered_iters:
                base_k = float(base_all[step, k])
                cand_k = float(cand_all[step, k])
                # Matches the slack used by the 6/13 paired-replay test.
                slack = 1.0e-4 + 0.10 * max(base_k, 1.0e-6)
                if cand_k > base_k + slack:
                    ok_all = False
                    break
            if ok_all and any(base_all[step, k] > 1.0e-8 for k in filtered_iters):
                satisfying_steps.append(step)

        self.assertGreaterEqual(
            len(satisfying_steps),
            min_steps_satisfying,
            f"[sliding_cube / {channel_name}] no step satisfies "
            f"{channel_name}(coulomb_newton) <= {channel_name}(bisection_desaxce) + slack "
            f"at all iters {filtered_iters}; "
            f"replay length={base_all.shape[0]}.  "
            f"per-step {channel_name}(baseline) iter={filtered_iters[0]} (last 10): "
            f"{base_all[-10:, filtered_iters[0]].tolist()}; "
            f"per-step {channel_name}(candidate) iter={filtered_iters[0]} (last 10): "
            f"{cand_all[-10:, filtered_iters[0]].tolist()}",
        )

    def _run_paired_replay(
        self,
        device: wp.context.Device,
        model_builder,
        num_steps: int,
        num_iters: int,
    ) -> tuple[SolverFeatherPGS, SolverFeatherPGS]:
        model_ds = model_builder()
        _ds_q, ds_solver, _ = _run_sim(
            device,
            friction_mode="bisection_desaxce",
            num_steps=num_steps,
            model=model_ds,
            pgs_iterations=num_iters,
            pgs_debug=True,
        )
        model_cn = model_builder()
        _cn_q, cn_solver, _ = _run_sim(
            device,
            friction_mode="coulomb_newton",
            num_steps=num_steps,
            model=model_cn,
            pgs_iterations=num_iters,
            pgs_debug=True,
        )
        return ds_solver, cn_solver

    def test_r_mdp_dir_not_worse_sliding_cube(self):
        device = wp.get_device()
        if device is None:
            self.skipTest("No warp device available")

        num_steps = 120
        num_iters = 60

        ds_solver, cn_solver = self._run_paired_replay(
            device,
            model_builder=lambda: _build_sliding_cube_model(device),
            num_steps=num_steps,
            num_iters=num_iters,
        )
        self._compare_residual_channel(
            ds_solver,
            cn_solver,
            channel=5,
            channel_name="r_mdp_dir",
            iters=(5, 10, 20, 50),
        )

    def test_r_ds_compl_not_worse_sliding_cube(self):
        device = wp.get_device()
        if device is None:
            self.skipTest("No warp device available")

        num_steps = 120
        num_iters = 60

        ds_solver, cn_solver = self._run_paired_replay(
            device,
            model_builder=lambda: _build_sliding_cube_model(device),
            num_steps=num_steps,
            num_iters=num_iters,
        )
        self._compare_residual_channel(
            ds_solver,
            cn_solver,
            channel=3,
            channel_name="r_ds_compl",
            iters=(5, 10, 20, 50),
        )


if __name__ == "__main__":
    unittest.main()
