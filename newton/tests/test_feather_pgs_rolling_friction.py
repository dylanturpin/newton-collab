# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Validate opt-in per-contact torsional and rolling friction in FeatherPGS."""

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS

GRAVITY = 9.81
DT = 0.00125
MU = 0.8
RADIUS = 0.02
MU_ROLLING = 0.01
SOLVER_OPTIONS = {
    "pgs_mode": "matrix_free",
    "articulated_contact_response": "immediate",
    "pgs_iterations": 16,
    "friction_anchor_beta": 0.0,
    "angular_damping": 0.0,
    "dense_max_constraints": 32,
    "mf_max_constraints": 16,
    "double_buffer": False,
}


def ball(
    *,
    radius=RADIUS,
    mu=MU,
    mu_torsional=0.0,
    mu_rolling=MU_ROLLING,
    speed=1.0,
    rolling=True,
    spin=0.0,
    tilt=0.0,
    enable=True,
    cone="pyramidal",
    **solver_overrides,
):
    """Build a unit-mass solid sphere on the ground, articulated by slide x/z and hinge y/z joints.

    The joint chain keeps the slides world-aligned, so the generalized velocities are the
    center speed (0), vertical speed (1), rolling rate (2) and spin rate (3).
    """
    b = newton.ModelBuilder()
    material = newton.ModelBuilder.ShapeConfig(mu=mu, mu_torsional=mu_torsional, mu_rolling=mu_rolling)
    b.add_ground_plane(cfg=material)
    pose = wp.transform(wp.vec3(0.0, 0.0, radius), wp.quat_identity())
    tiny = wp.mat33(np.eye(3, dtype=np.float32) * 1.0e-8)
    carriers = [b.add_link(xform=pose, mass=1.0e-4, inertia=tiny) for _ in range(3)]
    inertia = wp.mat33(np.eye(3, dtype=np.float32) * 0.4 * radius * radius)
    body = b.add_link(xform=pose, mass=1.0, inertia=inertia)
    identity = wp.transform_identity()
    joints = [
        b.add_joint_prismatic(-1, carriers[0], parent_xform=pose, child_xform=identity, axis=(1.0, 0.0, 0.0)),
        b.add_joint_prismatic(
            carriers[0], carriers[1], parent_xform=identity, child_xform=identity, axis=(0.0, 0.0, 1.0)
        ),
        b.add_joint_revolute(
            carriers[1], carriers[2], parent_xform=identity, child_xform=identity, axis=(0.0, 1.0, 0.0)
        ),
        b.add_joint_revolute(carriers[2], body, parent_xform=identity, child_xform=identity, axis=(0.0, 0.0, 1.0)),
    ]
    b.add_articulation(joints)
    sphere = newton.ModelBuilder.ShapeConfig(density=0.0, mu=mu, mu_torsional=mu_torsional, mu_rolling=mu_rolling)
    b.add_shape_sphere(body, radius=radius, cfg=sphere)
    model = b.finalize(device="cuda:0")
    model.set_gravity((GRAVITY * math.sin(tilt), 0.0, -GRAVITY * math.cos(tilt)))
    state = model.state()
    state.joint_qd.assign(np.array([speed, 0.0, speed / radius if rolling else 0.0, spin], np.float32))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    options = dict(SOLVER_OPTIONS)
    options.update(solver_overrides)
    solver = SolverFeatherPGS(
        model, enable_torsional_rolling_friction=enable, torsional_rolling_friction_cone=cone, **options
    )
    return Scene(model, state, solver)


class Scene:
    """Step one model with its own collision pipeline and record generalized velocities."""

    def __init__(self, model, state, solver):
        self.model = model
        self.solver = solver
        self.state = state
        self.next = model.state()
        self.control = model.control()
        self.pipeline = newton.CollisionPipeline(model, rigid_contact_max=16, broad_phase="explicit")
        self.contacts = self.pipeline.contacts()

    def step(self):
        self.pipeline.collide(self.state, self.contacts)
        self.solver.step(self.state, self.next, self.control, self.contacts, DT)
        self.state, self.next = self.next, self.state

    def run(self, steps):
        """Return the generalized velocity after each step, shape (steps, dofs)."""
        out = []
        for _ in range(steps):
            self.step()
            out.append(self.state.joint_qd.numpy().copy())
        return np.array(out)


def rolling_deceleration(cone, radius=RADIUS, mu=MU, mu_rolling=MU_ROLLING):
    """Steady rolling deceleration of the unit-mass solid sphere for a saturated cone [m/s^2]."""
    sliding_per_torque = 5.0 / (7.0 * radius * mu)  # |f| / mu needed per unit rolling torque
    if cone == "pyramidal":
        torque = GRAVITY / (1.0 / mu_rolling + sliding_per_torque)
    else:
        torque = GRAVITY / math.hypot(1.0 / mu_rolling, sliding_per_torque)
    return 5.0 / 7.0 * torque / radius


def holding_slope(cone, radius=RADIUS, mu=MU, mu_rolling=MU_ROLLING):
    """Largest incline slope on which the cone holds the sphere at rest."""
    if cone == "pyramidal":
        return 1.0 / (1.0 / mu + radius / mu_rolling)
    return 1.0 / math.hypot(1.0 / mu, radius / mu_rolling)


def fit_deceleration(qd, start, stop):
    t = np.arange(1, len(qd) + 1) * DT
    return -np.polyfit(t[start:stop], qd[start:stop, 0], 1)[0]


@unittest.skipUnless(wp.is_cuda_available(), "Torsional/rolling friction requires CUDA")
class TestFeatherPGSRollingFriction(unittest.TestCase):
    def test_rolling_deceleration_matches_the_cone(self):
        for cone in ("pyramidal", "elliptic"):
            for radius, mu_rolling in ((0.02, 0.01), (0.1, 0.002)):
                with self.subTest(cone=cone, radius=radius, mu_rolling=mu_rolling):
                    qd = ball(radius=radius, mu_rolling=mu_rolling, cone=cone).run(80)
                    expected = rolling_deceleration(cone, radius, MU, mu_rolling)
                    self.assertAlmostEqual(fit_deceleration(qd, 8, 80), expected, delta=0.005 * expected)
                    # Pure rolling is kept: the contact slip stays at its first-contact value.
                    slip = qd[8:, 0] - qd[8:, 2] * radius
                    self.assertLess(np.ptp(slip), 1e-3)

    def test_rolling_stops_and_rests(self):
        qd = ball(speed=0.1).run(200)
        self.assertLess(abs(qd[-1, 0]), 1e-5)
        self.assertLess(abs(qd[-1, 2]), 1e-3)

    def test_incline_holds_below_and_rolls_above_threshold(self):
        for cone in ("pyramidal", "elliptic"):
            slope = holding_slope(cone)
            for factor, rolls in ((0.9, False), (1.1, True)):
                with self.subTest(cone=cone, factor=factor):
                    scene = ball(speed=0.0, tilt=math.atan(factor * slope), cone=cone)
                    qd = scene.run(400)
                    if rolls:
                        self.assertGreater(qd[-1, 0], 0.05)
                    else:
                        self.assertLess(abs(qd[-1, 0]), 1e-4)

    def test_spin_decays_at_the_torsional_bound(self):
        radius, mu_torsional = 0.1, 0.005
        for cone in ("pyramidal", "elliptic"):
            with self.subTest(cone=cone):
                scene = ball(radius=radius, mu_torsional=mu_torsional, mu_rolling=0.0, speed=0.0, spin=5.0, cone=cone)
                qd = scene.run(160)
                t = np.arange(1, 161) * DT
                rate = -np.polyfit(t[8:], qd[8:, 3], 1)[0]
                expected = mu_torsional * GRAVITY / (0.4 * radius * radius)
                self.assertAlmostEqual(rate, expected, delta=0.005 * expected)

    def test_creep_speed_softens_stiction_by_load_fraction(self):
        """Below the bound the coefficient times the angular rate settles at creep speed times the load fraction."""
        creep_speed, mu_torsional = 2.0e-3, 0.02
        for dof, coefficient in ((3, mu_torsional), (2, MU_ROLLING)):
            for fraction in (0.5, 0.9):
                with self.subTest(dof=dof, fraction=fraction):
                    scene = ball(
                        speed=0.0, mu_torsional=mu_torsional, torsional_rolling_friction_creep_speed=creep_speed
                    )
                    scene.run(40)
                    force = np.zeros(scene.model.joint_dof_count, np.float32)
                    force[dof] = fraction * coefficient * GRAVITY
                    scene.control.joint_f.assign(force)
                    qd = scene.run(200)
                    self.assertAlmostEqual(
                        qd[-1, dof], creep_speed * fraction / coefficient, delta=0.02 * creep_speed / coefficient
                    )
        rigid = ball(speed=0.0, mu_torsional=mu_torsional)
        rigid.run(40)
        force = np.zeros(rigid.model.joint_dof_count, np.float32)
        force[3] = 0.9 * mu_torsional * GRAVITY
        rigid.control.joint_f.assign(force)
        self.assertLess(abs(rigid.run(200)[-1, 3]), 1e-5)

    def test_sliding_saturates_the_pyramidal_cone_until_rolling(self):
        """A sliding sphere spends the whole budget on sliding, then rolls with the coupled resistance."""
        qd = ball(speed=1.0, rolling=False).run(160)
        roll_time = 2.0 / (7.0 * MU * GRAVITY)
        roll_step = int(roll_time / DT)
        sliding = fit_deceleration(qd, 2, roll_step - 2)
        self.assertAlmostEqual(sliding, MU * GRAVITY, delta=0.01 * MU * GRAVITY)
        expected = rolling_deceleration("pyramidal")
        self.assertAlmostEqual(fit_deceleration(qd, roll_step + 8, 160), expected, delta=0.01 * expected)

    def test_disabled_and_zero_coefficients_match_the_baseline_bitwise(self):
        baseline = ball(enable=False).run(60)
        self.assertAlmostEqual(baseline[-1, 0], 1.0, delta=1e-4)
        zero = ball(mu_torsional=0.0, mu_rolling=0.0).run(60)
        np.testing.assert_array_equal(zero, ball(mu_torsional=0.0, mu_rolling=0.0, enable=False).run(60))
        np.testing.assert_array_equal(baseline, ball(enable=False, mu_rolling=0.5).run(60))

    def test_zero_coefficients_allocate_no_rows(self):
        scene = ball(mu_torsional=0.0, mu_rolling=0.0)
        scene.step()
        self.assertEqual(int(scene.solver.constraint_count.numpy()[0]), 3)
        scene = ball()
        scene.step()
        self.assertEqual(int(scene.solver.constraint_count.numpy()[0]), 6)

    def test_runtime_coefficient_edits_and_array_replacement(self):
        scene = ball(mu_rolling=0.0)
        qd = scene.run(20)
        self.assertAlmostEqual(qd[-1, 0], 1.0, delta=1e-4)
        shapes = scene.model.shape_count
        scene.model.shape_material_mu_rolling.assign(np.full(shapes, MU_ROLLING, np.float32))
        scene.solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
        expected = rolling_deceleration("pyramidal")
        self.assertAlmostEqual(fit_deceleration(scene.run(60), 8, 60), expected, delta=0.005 * expected)
        scene.model.shape_material_mu_rolling = wp.zeros(shapes, dtype=wp.float32, device=scene.model.device)
        scene.solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
        qd = scene.run(20)
        self.assertAlmostEqual(qd[-1, 0], qd[0, 0], delta=1e-5)

    def test_graph_replay_matches_eager(self):
        eager = ball().run(40)
        scene = ball()
        scene.step()
        scene = ball()
        with wp.ScopedCapture(device=scene.model.device) as capture:
            scene.step()
            scene.step()
        replay = []
        for _ in range(20):
            wp.capture_launch(capture.graph)
            replay.append(scene.state.joint_qd.numpy().copy())
        np.testing.assert_array_equal(np.array(replay), eager[1::2])

    def test_reset_keeps_no_angular_history(self):
        fresh = ball().run(30)
        scene = ball()
        start_q = scene.state.joint_q.numpy().copy()
        start_qd = scene.state.joint_qd.numpy().copy()
        scene.run(30)
        scene.state.joint_q.assign(start_q)
        scene.state.joint_qd.assign(start_qd)
        newton.eval_fk(scene.model, scene.state.joint_q, scene.state.joint_qd, scene.state)
        scene.solver.reset(scene.state)
        np.testing.assert_array_equal(scene.run(30), fresh)

    def test_row_capacity_overflow_is_reported(self):
        scene = ball(dense_max_constraints=5, warn_constraint_overflow=False)
        scene.step()
        self.assertEqual(int(scene.solver.constraint_count.numpy()[0]), 0)
        self.assertTrue(bool(scene.solver.constraint_overflow.numpy()[0]))

    def test_unsupported_routes_are_rejected(self):
        rejected = (
            {"pgs_mode": "split"},
            {"articulated_contact_response": "propagation"},
            {"friction_anchor_beta": 0.2},
            {"pgs_warmstart": True},
            {"pgs_velocity_iterations": 2},
            {"friction_mode": "bisection"},
            {"pgs_schedule": "contact_then_internal"},
            {"contact_torsion_radius": 0.01},
            {"enable_sleeping": True},
        )
        for overrides in rejected:
            with self.subTest(**overrides), self.assertRaises(ValueError):
                ball(**overrides)
        with self.assertRaises(ValueError):
            ball(cone="cubic")
        with self.assertRaises(ValueError):
            ball(torsional_rolling_friction_creep_speed=-1.0)

    def test_free_rigid_coefficients_are_rejected(self):
        b = newton.ModelBuilder()
        b.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu_torsional=0.0, mu_rolling=0.0))
        body = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
        b.add_shape_sphere(body, radius=0.1)
        model = b.finalize(device="cuda:0")
        with self.assertRaises(ValueError):
            SolverFeatherPGS(model, enable_torsional_rolling_friction=True, **SOLVER_OPTIONS)
        model.shape_material_mu_torsional.zero_()
        model.shape_material_mu_rolling.zero_()
        solver = SolverFeatherPGS(model, enable_torsional_rolling_friction=True, **SOLVER_OPTIONS)
        model.shape_material_mu_rolling.fill_(0.01)
        with self.assertRaises(ValueError):
            solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)


if __name__ == "__main__":
    unittest.main(verbosity=2)
