# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the identity-matched dense contact warm start in SolverFeatherPGS."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_CONTACT


def _build_press():
    """Build a 1-DOF prismatic press: an articulated box driven down onto the ground.

    The drive target sits below the contact height, so after touchdown the press
    stalls against the ground under a steady drive force — a persistent dense
    contact family (articulated vs. static routes to the dense path) whose
    converged impulse is constant per step. Exactly the regime warm starting
    carries impulses across, and the regime where the pre-fix carry diverged.
    """
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.add_ground_plane()
    body = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.30), wp.quat_identity()))
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    joint = builder.add_joint_prismatic(
        parent=-1,
        child=body,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.30), wp.quat_identity()),
        target_ke=2.0e3,
        target_kd=50.0,
    )
    builder.add_articulation([joint], label="press")
    return builder.finalize()


def _run_press(steps: int, warm_kwargs: dict, contact_matching: str | None = "sticky"):
    """Drive the press to a stall and record per-step contact-impulse sums and speeds.

    Returns (impulse_sum_per_step, |joint_qd|_per_step, final_state).
    """
    model = _build_press()
    solver = newton.solvers.SolverFeatherPGS(
        model,
        pgs_mode="matrix_free",
        pgs_iterations=8,
        pgs_beta=0.1,
        **warm_kwargs,
    )
    pipeline_kwargs = {} if contact_matching is None else {"contact_matching": contact_matching}
    pipeline = newton.CollisionPipeline(model, **pipeline_kwargs)
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    targets = model.joint_target_q.numpy().copy()
    targets[0] = -0.25  # 5 cm below touchdown: sustained press after the stall
    control.joint_target_q.assign(targets)

    impulse_sums = np.zeros(steps)
    speeds = np.zeros(steps)
    for i in range(steps):
        pipeline.collide(state_0, contacts)
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0
        counts = solver.constraint_count.numpy()
        rows = solver.row_type.numpy()[0, : counts[0]]
        lam = solver.impulses.numpy()[0, : counts[0]]
        impulse_sums[i] = np.abs(lam[rows == PGS_CONSTRAINT_TYPE_CONTACT]).sum()
        speeds[i] = np.abs(state_0.joint_qd.numpy()[0])
    return impulse_sums, speeds, state_0


@unittest.skipUnless(wp.get_device().is_cuda, "SolverFeatherPGS matrix-free mode requires CUDA")
class TestFeatherPGSMatchedWarmstart(unittest.TestCase):
    def test_matched_warmstart_holds_static_press(self):
        """Both warm-start paths keep a stalled press at the cold equilibrium.

        Regression net for the warm-start velocity accounting: carried impulses
        are installed into the starting velocity exactly once. A missing install
        accumulates the ledger (``lambda_{n+1} = lambda_n + lambda*``, the
        historical divergence); a duplicated install halves it (measured 0.5x
        cold when the matched path added a second fold on top of
        ``_stage6_apply_impulses_world``). The identity-matched path must also
        stay quiet and match the cold ledger while seeding by contact identity
        rather than raw slot index.
        """
        steps = 240
        stall = slice(120, None)  # well past touchdown + transient

        lam_cold, _speed_cold, _ = _run_press(steps, {})
        lam_legacy, speed_legacy, _ = _run_press(steps, {"pgs_warmstart": True})
        lam_matched, speed_matched, state = _run_press(steps, {"pgs_warmstart_matched": True})

        self.assertTrue(np.isfinite(state.body_q.numpy()).all())

        cold_end = lam_cold[-10:].mean()
        legacy_end = lam_legacy[-10:].mean()
        matched_end = lam_matched[-10:].mean()

        # The solver installs carried impulses into the starting velocity
        # (_stage6_apply_impulses_world under pgs_warmstart), so BOTH warm paths
        # must sit at the cold equilibrium. A duplicated velocity install shows
        # up here as a halved impulse ledger (measured 0.5x when the matched
        # path folded the seed a second time); a missing install shows up as an
        # accumulating ledger (lambda_{n+1} = lambda_n + lambda*).
        matched_growth = lam_matched[-10:].mean() / max(lam_matched[stall][:10].mean(), 1e-12)
        self.assertLess(matched_growth, 1.25, f"matched warm-start impulse grew x{matched_growth:.2f} at a stall")
        for name, end in (("legacy", legacy_end), ("matched", matched_end)):
            self.assertGreater(end, 0.7 * cold_end, f"{name} impulse ledger {end:.3f} below cold {cold_end:.3f}")
            self.assertLess(end, 1.4 * cold_end, f"{name} impulse ledger {end:.3f} above cold {cold_end:.3f}")
        for name, sp in (("legacy", speed_legacy), ("matched", speed_matched)):
            self.assertLess(
                sp[stall].max(),
                0.02,
                f"press not quiet under {name} warm start (peak |qd| {sp[stall].max():.3f} m/s)",
            )

    def test_matched_warmstart_matches_cold_equilibrium(self):
        """The matched warm start converges to the cold solve's stall pose, not a new one.

        Warm starting changes the sweep's starting point, never the physics: at
        the stall both runs must agree on the joint position to sub-millimetre.
        """
        _, _, state_cold = _run_press(240, {})
        _, _, state_warm = _run_press(240, {"pgs_warmstart_matched": True})
        q_cold = float(state_cold.joint_q.numpy()[0])
        q_warm = float(state_warm.joint_q.numpy()[0])
        self.assertAlmostEqual(q_warm, q_cold, delta=1.0e-3)

    def test_matched_warmstart_requires_contact_matching(self):
        """Constructing with pgs_warmstart_matched but stepping unmatched contacts raises.

        The identity gather needs ``Contacts.rigid_contact_match_index``; a
        pipeline built without ``contact_matching`` must fail loudly rather than
        silently reusing impulses by raw slot index.
        """
        with self.assertRaises(NotImplementedError):
            _run_press(3, {"pgs_warmstart_matched": True}, contact_matching=None)

    def test_matched_implies_warmstart_flag(self):
        """pgs_warmstart_matched=True implies the pgs_warmstart carry path."""
        model = _build_press()
        solver = newton.solvers.SolverFeatherPGS(
            model, pgs_mode="matrix_free", pgs_warmstart_matched=True, pgs_iterations=4
        )
        self.assertTrue(solver.pgs_warmstart)
        self.assertTrue(solver.pgs_warmstart_matched)


if __name__ == "__main__":
    unittest.main()
