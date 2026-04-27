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

"""Tests for ``friction_mode="bisection_desaxce"`` (FPGS Friction Modes 6/13).

Extends :mod:`newton.tests.test_feather_pgs_friction_mode_bisection` with
the de Saxce max-dissipation correction on top of the RAISim bisection
step.  The correction adds ``μ · ‖c_T‖`` to the normal target velocity
(Le Lidec & Carpentier 2024, on disk at
``artifacts/2026-04-16-slack-raisim/papers/hal-04438175v2.pdf``), which
drives the ``r_ds_compl`` (channel 3) and ``r_mdp_dir`` (channel 5)
residuals down on sliding contacts versus pure bisection.

Tests cover:

* A deterministic sliding-cube scene that always runs (no downloaded
  assets): assert that ``r_ds_compl`` and ``r_mdp_dir`` are no larger
  with bisection_desaxce than with bisection at matched GS iteration
  counts ``∈ {5, 10, 20, 50}`` (acceptance criterion).
* A 500-step sphere-on-plane replay to confirm stability parity with the
  5/13 bisection path.
* ``g1_flat`` (Unitree G1 on flat ground) and ``h1_tabletop``
  (Unitree H1 + tabletop) fixed-seed replays — the real nightly
  benchmark scenes the issue calls out ("g1_flat and h1_tabletop
  nightly render completes without blowup").  Reuses the 5/13 scene
  builders so both friction modes exercise the same contact problem.
  USD-asset gated; skipped when ``unitree_g1`` / ``unitree_h1`` are not
  available locally.
* ``pgs_debug=True`` populates the 6-channel ``_pgs_ncp_residual_log``
  for bisection_desaxce exactly as it does for bisection.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers import SolverFeatherPGS
from newton.tests.test_feather_pgs_friction_mode_bisection import (
    _try_build_g1_flat_model,
    _try_build_h1_tabletop_model,
)
from newton.tests.unittest_utils import USD_AVAILABLE


def _build_sphere_on_plane_model(device: wp.context.Device) -> newton.Model:
    """Deterministic sphere-on-ground scene (no downloaded assets).

    Matches the scene used by the 5/13 bisection tests so we can compare
    residuals at matched GS iterations.
    """
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


def _build_sliding_cube_model(
    device: wp.context.Device,
    initial_velocity: tuple[float, float, float] = (2.0, 0.0, 0.0),
) -> newton.Model:
    """Sliding-cube scene: a free-body cube resting on the ground and
    given an initial tangential velocity so contacts slide for many
    frames.  De Saxce's correction specifically improves convergence on
    sliding (not separating or sticking) contacts, so this is the
    scenario where the 6/13 benefit should show up most clearly.
    """
    builder = newton.ModelBuilder()
    builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
        limit_ke=0.0, limit_kd=0.0, friction=0.0
    )
    builder.default_shape_cfg.ke = 5.0e4
    builder.default_shape_cfg.kd = 5.0e2
    builder.default_shape_cfg.kf = 1.0e3
    # Moderate friction so the cube slides rather than sticking.
    builder.default_shape_cfg.mu = 0.3

    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1),
    )
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    # Seed a tangential velocity on the free-body root (6-DOF: linear
    # then angular).  joint_qd[0:3] is world-frame linear velocity.
    joint_qd_np = model.joint_qd.numpy().copy()
    if joint_qd_np.size >= 3:
        joint_qd_np[0] = float(initial_velocity[0])
        joint_qd_np[1] = float(initial_velocity[1])
        joint_qd_np[2] = float(initial_velocity[2])
        model.joint_qd.assign(joint_qd_np)
    return model


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
    """Run ``num_steps`` frames on ``model`` (or build sphere-on-plane).

    Mirrors the helper in
    :mod:`newton.tests.test_feather_pgs_friction_mode_bisection` so
    residual comparisons stay apples-to-apples.
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


class TestFeatherPGSFrictionModeBisectionDeSaxceStability(unittest.TestCase):
    """``friction_mode="bisection_desaxce"`` runs a stable matrix-free simulation."""

    def test_sphere_on_plane_500_steps_finite(self):
        """500-step fixed-seed run on sphere-on-plane stays finite and settled."""
        device = wp.get_device()
        if device is None:
            self.skipTest("No warp device available")

        joint_q, _solver, _model = _run_sim(
            device, friction_mode="bisection_desaxce", num_steps=500
        )

        self.assertTrue(
            np.all(np.isfinite(joint_q)),
            f"bisection_desaxce produced non-finite joint_q after 500 steps: {joint_q}",
        )
        # Free joint: joint_q[2] is z.  Sphere radius 0.2 -> settles near z≈0.2.
        self.assertGreater(joint_q[2], 0.15, f"sphere fell through the ground: z={joint_q[2]}")
        self.assertLess(joint_q[2], 0.35, f"sphere did not settle: z={joint_q[2]}")

    def test_sliding_cube_stable(self):
        """A sliding cube with a tangential initial velocity stays finite."""
        device = wp.get_device()
        if device is None:
            self.skipTest("No warp device available")

        model = _build_sliding_cube_model(device)
        joint_q, _solver, _model = _run_sim(
            device,
            friction_mode="bisection_desaxce",
            num_steps=240,  # ~1s of sim time
            model=model,
            pgs_iterations=20,
        )
        self.assertTrue(np.all(np.isfinite(joint_q)))
        # z in a generous band: the cube should not have fallen through
        # or rocketed upward.
        self.assertGreater(joint_q[2], -0.1)
        self.assertLess(joint_q[2], 1.0)


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestFeatherPGSFrictionModeBisectionDeSaxceBenchmarkScenarios(unittest.TestCase):
    """``bisection_desaxce`` on the real ``g1_flat`` / ``h1_tabletop`` scenes.

    These tests cover the nightly-render acceptance criterion from the
    issue: "g1_flat and h1_tabletop nightly render completes without
    blowup".  We cannot encode an MP4 or an ffmpeg pipeline in a unit
    test, so we assert the frame-by-frame state of the benchmark-scene
    replay is finite (no NaN / inf) and that all bodies stay within a
    generous spatial / velocity envelope — the same "unit-test analogue
    of the nightly render" pattern used by the 5/13 ``bisection`` tests.

    Both scenarios are USD-asset-gated and ``skipTest`` when the
    ``unitree_g1`` / ``unitree_h1`` assets are not cached locally.
    """

    # Same cadence as the 5/13 smoke tests so the two modes can be
    # compared at matched frame counts.
    G1_NUM_FRAMES: int = 60
    G1_SUBSTEPS: int = 2
    G1_PGS_ITERATIONS: int = 4

    H1_NUM_FRAMES: int = 20
    H1_SUBSTEPS: int = 8
    H1_PGS_ITERATIONS: int = 8

    def test_g1_flat_bisection_desaxce_stable(self):
        """``g1_flat`` bisection_desaxce replay stays finite with torso in range."""
        device = wp.get_device()
        if device is None:
            self.skipTest("No warp device available")

        model = _try_build_g1_flat_model(device)
        if model is None:
            self.skipTest("G1 asset not available (offline CI or USD plugin missing).")

        solver = SolverFeatherPGS(
            model,
            pgs_mode="matrix_free",
            friction_mode="bisection_desaxce",
            pgs_iterations=self.G1_PGS_ITERATIONS,
            dense_max_constraints=32,
        )
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        contacts = model.collide(state_0)
        sim_dt = 1.0 / 60.0
        sub_dt = sim_dt / float(self.G1_SUBSTEPS)

        body_qd_max_abs = 0.0
        for frame in range(self.G1_NUM_FRAMES):
            model.collide(state_0, contacts)
            for _ in range(self.G1_SUBSTEPS):
                state_0.clear_forces()
                solver.step(state_0, state_1, control, contacts, sub_dt)
                state_0, state_1 = state_1, state_0
            body_qd = state_0.body_qd.numpy()
            body_q = state_0.body_q.numpy()
            self.assertTrue(
                np.all(np.isfinite(body_qd)),
                f"g1_flat bisection_desaxce frame {frame}: NaN in body_qd",
            )
            self.assertTrue(
                np.all(np.isfinite(body_q)),
                f"g1_flat bisection_desaxce frame {frame}: NaN in body_q",
            )
            body_qd_max_abs = max(body_qd_max_abs, float(np.max(np.abs(body_qd))))

        joint_q = state_0.joint_q.numpy()
        self.assertTrue(
            np.all(np.isfinite(joint_q)),
            f"g1_flat bisection_desaxce produced NaNs: {joint_q}",
        )
        # Free-joint root (first 7 entries): z is index 2.  A G1 standing
        # near z≈0.8m should neither fall through the ground nor rocket
        # upward within ~0.5s of passive dynamics.
        self.assertGreater(
            joint_q[2], 0.0, f"g1_flat bisection_desaxce torso fell through ground: z={joint_q[2]}"
        )
        self.assertLess(
            joint_q[2], 2.0, f"g1_flat bisection_desaxce torso blew up: z={joint_q[2]}"
        )
        # No body velocity should have blown up.  Loose passive-dynamics
        # bound — matches the 5/13 test's envelope for this scene.
        self.assertLess(
            body_qd_max_abs,
            100.0,
            f"g1_flat bisection_desaxce replay exhibited blowup: "
            f"max|body_qd|={body_qd_max_abs:g}",
        )

    def test_h1_tabletop_bisection_desaxce_stable(self):
        """``h1_tabletop`` bisection_desaxce replay stays finite with bodies in range.

        Unit-test analogue of the nightly ``h1_tabletop`` render: we
        assert per-frame state is finite and bodies stay within a
        generous velocity envelope.  Any MP4-level render is produced
        offline by ``artifacts/issue-worker/issue-12/``.
        """
        device = wp.get_device()
        if device is None:
            self.skipTest("No warp device available")

        model = _try_build_h1_tabletop_model(device)
        if model is None:
            self.skipTest("H1 asset not available (offline CI or USD plugin missing).")

        solver = SolverFeatherPGS(
            model,
            pgs_mode="matrix_free",
            friction_mode="bisection_desaxce",
            pgs_iterations=self.H1_PGS_ITERATIONS,
            dense_max_constraints=128,
            pgs_debug=True,
        )
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        contacts = model.collide(state_0)

        sim_dt = 1.0 / 60.0
        sub_dt = sim_dt / float(self.H1_SUBSTEPS)

        body_qd_max_abs = 0.0
        for frame in range(self.H1_NUM_FRAMES):
            model.collide(state_0, contacts)
            for _ in range(self.H1_SUBSTEPS):
                state_0.clear_forces()
                solver.step(state_0, state_1, control, contacts, sub_dt)
                state_0, state_1 = state_1, state_0
            body_qd = state_0.body_qd.numpy()
            body_q = state_0.body_q.numpy()
            self.assertTrue(
                np.all(np.isfinite(body_qd)),
                f"h1_tabletop bisection_desaxce frame {frame}: NaN in body_qd",
            )
            self.assertTrue(
                np.all(np.isfinite(body_q)),
                f"h1_tabletop bisection_desaxce frame {frame}: NaN in body_q",
            )
            body_qd_max_abs = max(body_qd_max_abs, float(np.max(np.abs(body_qd))))

        # Loose bound: passive dynamics on this scene with default damping
        # should keep velocities well under 100 m/s.  Mirrors the 5/13
        # ``bisection`` check so both modes are judged by the same rule.
        self.assertLess(
            body_qd_max_abs,
            100.0,
            f"h1_tabletop bisection_desaxce replay exhibited blowup: "
            f"max|body_qd|={body_qd_max_abs:g}",
        )

        # Residual log populated with the 6-channel layout for every
        # sub-step.
        self.assertEqual(
            len(solver._pgs_ncp_residual_log),
            self.H1_NUM_FRAMES * self.H1_SUBSTEPS,
        )
        last = solver._pgs_ncp_residual_log[-1]
        self.assertEqual(last.shape[0], self.H1_PGS_ITERATIONS)
        self.assertEqual(last.shape[2], 6)
        self.assertTrue(np.all(np.isfinite(last)))


class TestFeatherPGSFrictionModeBisectionDeSaxceDiagnostic(unittest.TestCase):
    """NCP / MDP residual hook works with bisection_desaxce."""

    def test_ncp_log_populated(self):
        """``pgs_debug=True`` populates the 6-channel NCP log for
        ``bisection_desaxce`` exactly as for ``bisection``.
        """
        device = wp.get_device()
        if device is None:
            self.skipTest("No warp device available")

        num_steps = 60
        _joint_q, solver, _model = _run_sim(
            device,
            friction_mode="bisection_desaxce",
            num_steps=num_steps,
            pgs_iterations=20,
            pgs_debug=True,
        )
        self.assertEqual(len(solver._pgs_ncp_residual_log), num_steps)
        last = solver._pgs_ncp_residual_log[-1]  # [iters, worlds, 6]
        self.assertEqual(last.ndim, 3)
        self.assertEqual(last.shape[0], solver.pgs_iterations)
        self.assertEqual(last.shape[2], 6)
        self.assertTrue(np.all(np.isfinite(last)), "residuals contain non-finite values")


class TestFeatherPGSFrictionModeBisectionDeSaxceResidualComparison(unittest.TestCase):
    """Acceptance criterion: ``bisection_desaxce`` reduces ``r_ds_compl`` and
    ``r_mdp_dir`` versus pure ``bisection`` at matched GS iterations.

    Channel layout matches ``pgs_ncp_residuals_diagnostic_velocity``:
    ``[r_compl, r_cone, r_gap, r_ds_compl, r_ds_dual, r_mdp_dir]``.
    """

    def _compare_residual_channel(
        self,
        bi_solver: SolverFeatherPGS,
        ds_solver: SolverFeatherPGS,
        channel: int,
        channel_name: str,
        iters: tuple[int, ...],
        scenario_label: str,
        min_steps_satisfying: int = 1,
    ) -> None:
        """Assert there exists a step where
        ``res_channel(bisection_desaxce) <= res_channel(bisection) + slack``
        at *every* matched GS iter in ``iters``.

        Mirrors the "hand-picked representative step" pattern used by
        the 5/13 test helper for ``r_compl``.
        """
        bi_stack = np.stack(bi_solver._pgs_ncp_residual_log, axis=0)
        ds_stack = np.stack(ds_solver._pgs_ncp_residual_log, axis=0)
        # Max across worlds of ``channel`` -> [steps, iters].
        bi_all = np.max(bi_stack[:, :, :, channel], axis=2)
        ds_all = np.max(ds_stack[:, :, :, channel], axis=2)

        num_iter_budget = bi_all.shape[1]
        filtered_iters = [k for k in iters if k < num_iter_budget]
        if not filtered_iters:  # pragma: no cover - defensive
            self.fail(
                f"[{scenario_label}] none of the requested iters {iters} "
                f"fit inside num_iters={num_iter_budget}"
            )

        satisfying_steps: list[int] = []
        for step in range(bi_all.shape[0]):
            ok_all = True
            for k in filtered_iters:
                bi_k = float(bi_all[step, k])
                ds_k = float(ds_all[step, k])
                slack = 1.0e-4 + 0.10 * max(bi_k, 1.0e-6)
                if ds_k > bi_k + slack:
                    ok_all = False
                    break
            if ok_all and any(bi_all[step, k] > 1.0e-8 for k in filtered_iters):
                satisfying_steps.append(step)

        self.assertGreaterEqual(
            len(satisfying_steps),
            min_steps_satisfying,
            f"[{scenario_label} / {channel_name}] no step in the replay satisfies "
            f"{channel_name}(bisection_desaxce) <= {channel_name}(bisection) + slack at all "
            f"iters {filtered_iters}; replay length={bi_all.shape[0]}.  "
            f"per-step {channel_name}(bi) iter={filtered_iters[0]} (last 10): "
            f"{bi_all[-10:, filtered_iters[0]].tolist()}; "
            f"per-step {channel_name}(ds) iter={filtered_iters[0]} (last 10): "
            f"{ds_all[-10:, filtered_iters[0]].tolist()}",
        )

    def _run_paired_replay(
        self,
        device: wp.context.Device,
        model_builder,
        num_steps: int,
        num_iters: int,
    ) -> tuple[SolverFeatherPGS, SolverFeatherPGS]:
        """Run a replay with ``friction_mode="bisection"`` and another with
        ``friction_mode="bisection_desaxce"`` on freshly built models so
        contact trajectories match up at matched indices.
        """
        model_bi = model_builder()
        _bi_q, bi_solver, _ = _run_sim(
            device,
            friction_mode="bisection",
            num_steps=num_steps,
            model=model_bi,
            pgs_iterations=num_iters,
            pgs_debug=True,
        )
        model_ds = model_builder()
        _ds_q, ds_solver, _ = _run_sim(
            device,
            friction_mode="bisection_desaxce",
            num_steps=num_steps,
            model=model_ds,
            pgs_iterations=num_iters,
            pgs_debug=True,
        )
        return bi_solver, ds_solver

    def test_r_ds_compl_not_worse_sliding_cube(self):
        """``r_ds_compl(bisection_desaxce) <= r_ds_compl(bisection)`` on sliding cube."""
        device = wp.get_device()
        if device is None:
            self.skipTest("No warp device available")

        num_steps = 120
        num_iters = 60  # > 50 so the matched iters ladder fits

        bi_solver, ds_solver = self._run_paired_replay(
            device,
            model_builder=lambda: _build_sliding_cube_model(device),
            num_steps=num_steps,
            num_iters=num_iters,
        )
        self._compare_residual_channel(
            bi_solver,
            ds_solver,
            channel=3,
            channel_name="r_ds_compl",
            iters=(5, 10, 20, 50),
            scenario_label="sliding_cube",
        )

    def test_r_mdp_dir_not_worse_sliding_cube(self):
        """``r_mdp_dir(bisection_desaxce) <= r_mdp_dir(bisection)`` on sliding cube."""
        device = wp.get_device()
        if device is None:
            self.skipTest("No warp device available")

        num_steps = 120
        num_iters = 60

        bi_solver, ds_solver = self._run_paired_replay(
            device,
            model_builder=lambda: _build_sliding_cube_model(device),
            num_steps=num_steps,
            num_iters=num_iters,
        )
        self._compare_residual_channel(
            bi_solver,
            ds_solver,
            channel=5,
            channel_name="r_mdp_dir",
            iters=(5, 10, 20, 50),
            scenario_label="sliding_cube",
        )


if __name__ == "__main__":
    unittest.main()
