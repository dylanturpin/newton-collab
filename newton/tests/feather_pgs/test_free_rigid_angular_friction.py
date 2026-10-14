# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental rolling and torsional friction rows on SolverFeatherPGS matrix-free free-rigid contacts."""

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_CONTACT_ROLLING,
    PGS_CONSTRAINT_TYPE_CONTACT_SPIN,
    PGS_CONSTRAINT_TYPE_FRICTION,
)
from newton.solvers import SolverFeatherPGS, SolverObservableFlags
from newton.tests.feather_pgs._fixtures import set_torsion_radius

_DT = 1.0 / 480.0
_RADIUS = 0.03
_MASS = 0.1
_GRAVITY = 9.81
_ROLLING = {"linear": (0.0, -0.3, 0.0), "angular": (10.0, 0.0, 0.0)}
# Principal moments 0.6, 1 and 1.4 with an x-z product: a physically valid anisotropic inertia.
_ANISOTROPIC = np.array([[1.0, 0.0, 0.4], [0.0, 1.0, 0.0], [0.4, 0.0, 1.0]])
_ON = {"enable_free_rigid_angular_friction": True, "free_rigid_torsional_friction": 0.0}


def _coulomb_stopping_distance(speed: float, rolling: float) -> float:
    """Return a solid sphere's rolling distance [m]: v^2 (1 + I / (m r^2)) = 2 d rolling g / r."""
    return speed**2 * 1.4 / (2.0 * rolling * _GRAVITY / _RADIUS)


def _scene(body: str, tilt_deg: float = 0.0, inertia=None, count: int = 1):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    SolverFeatherPGS.register_custom_attributes(builder)
    table = builder.add_body(xform=wp.transform((0.0, 0.0, -0.05), wp.quat_identity()), is_kinematic=True)
    builder.add_shape_box(table, hx=2.0, hy=2.0, hz=0.05)
    rotation = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.radians(tilt_deg))
    bodies = []
    massless = newton.ModelBuilder.ShapeConfig(density=0.0)
    for index in range(count):
        x = 0.2 * index
        if body == "sphere":
            moments = np.eye(3) * 0.4 * _MASS * _RADIUS**2 if inertia is None else np.asarray(inertia)
            z = _RADIUS
        else:
            moments = np.eye(3) * _MASS * (2.0 * _RADIUS) ** 2 / 6.0
            tilt = math.radians(tilt_deg)
            z = _RADIUS * (math.cos(tilt) + math.sin(tilt)) + 1.0e-4
        obj = builder.add_body(
            xform=wp.transform((x, 0.0, z), rotation),
            mass=_MASS,
            inertia=wp.mat33(*np.asarray(moments, dtype=np.float32).ravel()),
            lock_inertia=True,
        )
        if body == "sphere":
            builder.add_shape_sphere(obj, radius=_RADIUS, cfg=massless)
        else:
            builder.add_shape_box(obj, hx=_RADIUS, hy=_RADIUS, hz=_RADIUS, cfg=massless)
        bodies.append(obj)
    return builder.finalize(), bodies


def _simulate(
    body,
    seconds,
    *,
    tilt_deg=0.0,
    linear=(0.0, 0.0, 0.0),
    angular=(0.0, 0.0, 0.0),
    graph=False,
    torsion_radius=0.0,
    contact_matching="disabled",
    inertia=None,
    steps=None,
    rolling_material=None,
    count=1,
    support_angular=None,
    observe=None,
    parent_f=False,
    **solver_kwargs,
):
    model, bodies = _scene(body, tilt_deg, inertia, count)
    if torsion_radius:
        set_torsion_radius(model, torsion_radius)
    if rolling_material is not None:
        model.shape_material_mu_rolling.fill_(rolling_material)
    options = {"solve_mode": "matrix_free", "iterations": 32, "deterministic": True, **solver_kwargs}
    solver = SolverFeatherPGS(model, **options)
    pipeline = newton.CollisionPipeline(model, deterministic=True, contact_matching=contact_matching)
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    qd = state_0.joint_qd.numpy()
    for obj in bodies:
        joint = int(np.flatnonzero(model.joint_child.numpy() == obj)[0])
        dof = int(model.joint_qd_start.numpy()[joint])
        qd[dof : dof + 3] = linear
        qd[dof + 3 : dof + 6] = angular
    if support_angular is not None:
        joint = int(np.flatnonzero(model.joint_child.numpy() == 0)[0])
        dof = int(model.joint_qd_start.numpy()[joint])
        qd[dof + 3 : dof + 6] = support_angular
    state_0.joint_qd.assign(qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    control = model.control()
    obj = bodies[0]
    start = state_0.body_q.numpy()[obj][:3].copy()

    observables = solver.observables({SolverObservableFlags.BODY_PARENT_F}) if parent_f else None

    def step(state_in, state_out):
        state_in.clear_forces()
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, _DT, observables=observables)

    if steps is not None:
        for _ in range(steps):
            step(state_0, state_1)
            state_0, state_1 = state_1, state_0
    elif graph:
        with wp.ScopedCapture(device=model.device) as capture:
            step(state_0, state_1)
            step(state_1, state_0)
        for _ in range(int(seconds / _DT) // 2):
            wp.capture_launch(capture.graph)
    else:
        for _ in range(int(seconds / _DT) // 2):
            step(state_0, state_1)
            step(state_1, state_0)
    if observe is not None:
        observe(solver, contacts, observables)
    pose, twist = state_0.body_q.numpy()[obj], state_0.body_qd.numpy()[obj]
    return pose, twist, float(np.linalg.norm(pose[:2] - start[:2]))


@unittest.skipUnless(wp.is_cuda_available(), "FeatherPGS matrix-free requires CUDA")
class TestFreeRigidAngularFriction(unittest.TestCase):
    def test_rolling_sphere_stops_at_the_coulomb_distance(self):
        """Stop a rolling sphere within 5% of the analytic Coulomb distance; keep it rolling when off."""
        _, twist_free, travel_free = _simulate("sphere", 1.0, **_ROLLING)
        self.assertGreater(np.linalg.norm(twist_free[3:6]), 9.0)
        self.assertGreater(travel_free, 0.25)
        for rolling in (0.001, 0.002):
            with self.subTest(rolling=rolling):
                _, twist, travel = _simulate("sphere", 1.5, free_rigid_rolling_friction=rolling, **_ON, **_ROLLING)
                self.assertLess(np.linalg.norm(twist[3:6]), 1.0e-3)
                expected = _coulomb_stopping_distance(0.3, rolling)
                self.assertAlmostEqual(travel, expected, delta=0.05 * expected)

    def test_material_coefficient_matches_the_override(self):
        """Read the shape material rolling coefficient when no override is given."""
        material, _, _ = _simulate("sphere", 0.5, rolling_material=0.002, **_ON, **_ROLLING)
        override, _, _ = _simulate(
            "sphere", 0.5, rolling_material=0.0, free_rigid_rolling_friction=0.002, **_ON, **_ROLLING
        )
        np.testing.assert_array_equal(material, override)

    def test_spin_row_stops_a_spinning_sphere_at_the_coulomb_rate(self):
        """Decelerate a sphere spinning on its contact point at mu_s m g / I, and stop it."""
        torsional = 5.0e-4
        options = {
            "enable_free_rigid_angular_friction": True,
            "free_rigid_torsional_friction": torsional,
            "free_rigid_rolling_friction": 0.0,
        }
        _, free, _ = _simulate("sphere", 0.2, angular=(0.0, 0.0, 6.0))
        self.assertGreater(free[5], 5.9)
        rate = torsional * _MASS * _GRAVITY / (0.4 * _MASS * _RADIUS**2)
        _, twist, _ = _simulate("sphere", 0.2, angular=(0.0, 0.0, 6.0), **options)
        self.assertAlmostEqual(twist[5], 6.0 - rate * 0.2, delta=0.1)
        _, twist, _ = _simulate("sphere", 0.6, angular=(0.0, 0.0, 6.0), **options)
        self.assertLess(abs(twist[5]), 1.0e-3)

    def test_rows_never_add_energy(self):
        """End a contact step with no more kinetic energy than it started with, for anisotropic inertias."""
        rng = np.random.default_rng(3)
        cases = [(_ANISOTROPIC * 4.0e-5, (1.0, 0.0, -3.0), rolling) for rolling in (1.0, 1.0e-3, 1.0e-4)]
        for _ in range(6):
            axes, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            moments = np.sort(rng.uniform(0.2, 1.0, 3))
            moments[2] = min(moments[2], moments[0] + moments[1] - 1.0e-3)
            cases.append((axes @ np.diag(moments) @ axes.T * 4.0e-5, tuple(rng.normal(size=3) * 3.0), 1.0e-3))

        for index, (inertia, omega, rolling) in enumerate(cases):

            def energy(twist, inertia=inertia):
                return 0.5 * _MASS * float(twist[:3] @ twist[:3]) + 0.5 * float(twist[3:6] @ inertia @ twist[3:6])

            with self.subTest(case=index, rolling=rolling):
                # Settle at rest first so the contact carries the weight, then spin it for one step.
                kwargs = {"angular": omega, "inertia": inertia, "steps": 1}
                _, twist_free, _ = _simulate("sphere", 0.0, **kwargs)
                _, twist, _ = _simulate(
                    "sphere",
                    0.0,
                    free_rigid_rolling_friction=rolling,
                    free_rigid_torsional_friction=rolling,
                    enable_free_rigid_angular_friction=True,
                    **kwargs,
                )
                start = 0.5 * float(np.asarray(omega) @ inertia @ np.asarray(omega))
                # Gravity does at most m g |v_z| dt of work in one step.
                self.assertLessEqual(energy(twist), start + _MASS * _GRAVITY * 1.0e-3 * _DT)
                self.assertLessEqual(energy(twist), energy(twist_free) * (1.0 + 1.0e-4))

    def test_spin_is_relative_to_a_turning_support(self):
        """Spin a resting sphere up to a kinematic turntable's rate at mu_s m g / I; leave it still when off."""
        torsional = 5.0e-4
        table = (0.0, 0.0, 2.0)
        _, free, _ = _simulate("sphere", 0.5, support_angular=table)
        self.assertLess(abs(free[5]), 1.0e-3)
        options = {"free_rigid_torsional_friction": torsional, "free_rigid_rolling_friction": 0.0}
        rate = torsional * _GRAVITY / (0.4 * _RADIUS**2)
        _, twist, _ = _simulate(
            "sphere", 0.1, support_angular=table, enable_free_rigid_angular_friction=True, **options
        )
        self.assertAlmostEqual(twist[5], rate * 0.1, delta=0.05)
        _, twist, _ = _simulate(
            "sphere", 0.5, support_angular=table, enable_free_rigid_angular_friction=True, **options
        )
        self.assertAlmostEqual(twist[5], 2.0, delta=1.0e-3)

    def test_toppling_and_sliding_cube_are_unchanged(self):
        """Leave a toppling or sliding cube within 2 mm of the option-off pose; the spinning cube slows at most."""
        for kwargs in ({"tilt_deg": 40.0}, {"linear": (0.5, 0.0, 0.0)}, {"angular": (0.0, 0.0, 6.0)}):
            with self.subTest(**kwargs):
                pose_free, twist_free, _ = _simulate("box", 1.0, **kwargs)
                pose, twist, _ = _simulate("box", 1.0, free_rigid_rolling_friction=0.002, **_ON, **kwargs)
                np.testing.assert_allclose(pose[:3], pose_free[:3], atol=2.0e-3)
                self.assertGreater(abs(float(np.dot(pose[3:7], pose_free[3:7]))), 0.999)

    def test_composes_with_sleeping_anchors_torsion_warm_start_and_velocity_iterations(self):
        """Keep the Coulomb stopping distance with each production-style option."""
        expected = _coulomb_stopping_distance(0.3, 0.002)
        for options in (
            {"enable_sleeping": True},
            {"friction_anchor_beta": 0.2},
            {"torsion_radius": 0.01},
            {"enable_sleeping": True, "friction_anchor_beta": 0.2, "torsion_radius": 0.01},
            {"contact_warmstart": True, "contact_matching": "sticky"},
            {"velocity_iterations": 1},
        ):
            with self.subTest(**options):
                _, twist, travel = _simulate(
                    "sphere", 1.5, free_rigid_rolling_friction=0.002, **_ON, **_ROLLING, **options
                )
                self.assertLess(np.linalg.norm(twist[3:6]), 1.0e-3)
                self.assertAlmostEqual(travel, expected, delta=0.05 * expected)

    def test_patch_anchors_share_one_budget(self):
        """Divide a patch's rolling and spin coefficients among its anchors, like its sliding coefficient."""
        seen = {}

        def observe(solver, contacts, observables):
            count = int(contacts.rigid_contact_count.numpy()[0])
            slots = solver._contact_slots_needed.numpy()[:count]
            slot = solver._contact_slot.numpy()[:count]
            types = solver._mf_row_type.numpy()[0]
            mu = solver._mf_row_mu.numpy()[0]
            anchors = [int(s) for s, n in zip(slot, slots, strict=True) if n == 6]
            seen["anchors"] = len(anchors)
            for s in anchors:
                self.assertEqual(types[s], PGS_CONSTRAINT_TYPE_CONTACT)
                self.assertEqual(types[s + 1], PGS_CONSTRAINT_TYPE_FRICTION)
                self.assertEqual(types[s + 3], PGS_CONSTRAINT_TYPE_CONTACT_ROLLING)
                self.assertEqual(types[s + 4], PGS_CONSTRAINT_TYPE_CONTACT_ROLLING)
                self.assertEqual(types[s + 5], PGS_CONSTRAINT_TYPE_CONTACT_SPIN)
                self.assertAlmostEqual(mu[s + 3] * len(anchors), 0.002, places=7)
                self.assertAlmostEqual(mu[s + 5] * len(anchors), 0.001, places=7)

        _simulate(
            "box",
            0.0,
            steps=3,
            friction_anchor_beta=0.2,
            enable_free_rigid_angular_friction=True,
            free_rigid_rolling_friction=0.002,
            free_rigid_torsional_friction=0.001,
            observe=observe,
        )
        self.assertEqual(seen["anchors"], 2)

    def test_repeated_and_captured_runs_are_bitwise_identical(self):
        """Match two eager runs and a CUDA-graph replay bit for bit, with several worlds of bodies."""
        kwargs = {
            "free_rigid_rolling_friction": 0.001,
            "free_rigid_torsional_friction": 0.001,
            "friction_anchor_beta": 0.2,
            "enable_sleeping": True,
            "count": 3,
            **_ON,
            **_ROLLING,
        }
        kwargs["free_rigid_torsional_friction"] = 0.001
        first, _, _ = _simulate("sphere", 0.3, **kwargs)
        second, _, _ = _simulate("sphere", 0.3, **kwargs)
        captured, _, _ = _simulate("sphere", 0.3, graph=True, **kwargs)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first, captured)

    def test_off_and_zero_coefficients_match_the_default(self):
        """Match the default bit for bit when disabled, and add no rows for zero coefficients."""
        default, _, _ = _simulate("sphere", 0.3, **_ROLLING)
        disabled, _, _ = _simulate(
            "sphere", 0.3, enable_free_rigid_angular_friction=False, free_rigid_rolling_friction=0.01, **_ROLLING
        )
        zero, _, _ = _simulate(
            "sphere",
            0.3,
            enable_free_rigid_angular_friction=True,
            free_rigid_rolling_friction=0.0,
            free_rigid_torsional_friction=0.0,
            **_ROLLING,
        )
        np.testing.assert_array_equal(default, disabled)
        np.testing.assert_array_equal(default, zero)

    def test_body_parent_f_balances_the_rolling_torque(self):
        """Keep a rolling free sphere's parent wrench at zero: its contact wrench carries the rolling torque."""
        seen = {}

        def observe(solver, contacts, observables):
            slot = int(solver._contact_slot.numpy()[0])
            seen["rolling"] = float(np.linalg.norm(solver._mf_impulses.numpy()[0, slot + 3 : slot + 5])) / _DT
            seen["parent"] = observables.body_parent_f.numpy()[1]

        _simulate(
            "sphere", 0.0, steps=4, parent_f=True, free_rigid_rolling_friction=0.002, **_ON, observe=observe, **_ROLLING
        )
        torque = 0.002 * _MASS * _GRAVITY
        self.assertAlmostEqual(seen["rolling"], torque, delta=0.05 * torque)
        self.assertLess(float(np.linalg.norm(seen["parent"][3:])), 0.02 * torque)

    def test_invalid_options_raise(self):
        """Reject negative and non-finite coefficients and unsupported solve modes."""
        model, _ = _scene("sphere")
        for value in (-1.0e-3, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                SolverFeatherPGS(model, enable_free_rigid_angular_friction=True, free_rigid_rolling_friction=value)
        with self.assertRaises(ValueError):
            SolverFeatherPGS(model, solve_mode="split", enable_free_rigid_angular_friction=True)
        with self.assertRaises(ValueError):
            SolverFeatherPGS(
                model, solve_mode="matrix_free", friction_mode="bisection", enable_free_rigid_angular_friction=True
            )


if __name__ == "__main__":
    unittest.main()
