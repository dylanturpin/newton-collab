# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental free-rigid rolling resistance of SolverFeatherPGS."""

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.rolling_resistance import (
    SUPPORT_LOAD_FIELDS,
    SUPPORT_LOAD_SCALE,
    apply_free_root_rolling_resistance,
)
from newton.solvers import SolverFeatherPGS
from newton.tests.feather_pgs._fixtures import set_torsion_radius

_DT = 1.0 / 480.0
_RADIUS = 0.03
_MASS = 0.1
_ROLLING = {"linear": (0.0, -0.3, 0.0), "angular": (10.0, 0.0, 0.0)}


def _coulomb_stopping_distance(speed: float, resistance: float) -> float:
    """Rolling solid sphere: v^2 = 2 d (resistance g / r) / (1 + I / (m r^2)) [m]."""
    return speed**2 * 1.4 / (2.0 * resistance * 9.81 / _RADIUS)


def _scene(body: str, tilt_deg: float = 0.0):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    SolverFeatherPGS.register_custom_attributes(builder)
    table = builder.add_body(xform=wp.transform((0.0, 0.0, -0.05), wp.quat_identity()), is_kinematic=True)
    builder.add_shape_box(table, hx=2.0, hy=2.0, hz=0.05)
    rotation = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.radians(tilt_deg))
    if body == "sphere":
        obj = builder.add_body(xform=wp.transform((0.0, 0.0, _RADIUS), rotation), mass=_MASS)
        builder.add_shape_sphere(obj, radius=_RADIUS)
    else:
        tilt = math.radians(tilt_deg)
        z = _RADIUS * (math.cos(tilt) + math.sin(tilt)) + 1.0e-4
        obj = builder.add_body(xform=wp.transform((0.0, 0.0, z), rotation), mass=_MASS)
        builder.add_shape_box(obj, hx=_RADIUS, hy=_RADIUS, hz=_RADIUS)
    return builder.finalize(), obj


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
    **solver_kwargs,
):
    model, obj = _scene(body, tilt_deg)
    if torsion_radius:
        set_torsion_radius(model, torsion_radius)
    options = {"solve_mode": "matrix_free", "iterations": 32, "deterministic": True, **solver_kwargs}
    solver = SolverFeatherPGS(model, **options)
    pipeline = newton.CollisionPipeline(model, deterministic=True, contact_matching=contact_matching)
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    joint = int(np.flatnonzero(model.joint_child.numpy() == obj)[0])
    dof = int(model.joint_qd_start.numpy()[joint])
    qd = state_0.joint_qd.numpy()
    qd[dof : dof + 3] = linear
    qd[dof + 3 : dof + 6] = angular
    state_0.joint_qd.assign(qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    control = model.control()
    start = state_0.body_q.numpy()[obj][:3].copy()

    def step(state_in, state_out):
        state_in.clear_forces()
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, _DT)

    steps = int(seconds / _DT) // 2 * 2
    if graph:
        with wp.ScopedCapture(device=model.device) as capture:
            step(state_0, state_1)
            step(state_1, state_0)
        for _ in range(steps // 2):
            wp.capture_launch(capture.graph)
    else:
        for _ in range(steps // 2):
            step(state_0, state_1)
            step(state_1, state_0)
    pose, twist = state_0.body_q.numpy()[obj], state_0.body_qd.numpy()[obj]
    return pose, twist, float(np.linalg.norm(pose[:2] - start[:2]))


@unittest.skipUnless(wp.is_cuda_available(), "FeatherPGS matrix-free requires CUDA")
class TestFreeRigidRollingResistance(unittest.TestCase):
    def test_rolling_sphere_stops_at_the_coulomb_distance(self):
        _, twist_free, travel_free = _simulate("sphere", 1.0, **_ROLLING)
        self.assertGreater(np.linalg.norm(twist_free[3:6]), 9.0)
        self.assertGreater(travel_free, 0.25)
        for resistance in (0.001, 0.002):
            with self.subTest(resistance=resistance):
                _, twist, travel = _simulate("sphere", 1.5, free_rigid_rolling_resistance=resistance, **_ROLLING)
                self.assertLess(np.linalg.norm(twist[3:6]), 1.0e-3)
                expected = _coulomb_stopping_distance(0.3, resistance)
                self.assertAlmostEqual(travel, expected, delta=0.15 * expected)

    def test_split_mode_matches_the_coulomb_distance(self):
        _, twist, travel = _simulate("sphere", 1.0, solve_mode="split", free_rigid_rolling_resistance=0.002, **_ROLLING)
        self.assertLess(np.linalg.norm(twist[3:6]), 1.0e-3)
        expected = _coulomb_stopping_distance(0.3, 0.002)
        self.assertAlmostEqual(travel, expected, delta=0.15 * expected)

    def test_toppling_spin_and_sliding_cube_are_unchanged(self):
        for kwargs in ({"tilt_deg": 40.0}, {"angular": (0.0, 0.0, 6.0)}, {"linear": (0.5, 0.0, 0.0)}):
            with self.subTest(**kwargs):
                pose_free, _, _ = _simulate("box", 1.0, **kwargs)
                pose, _, _ = _simulate("box", 1.0, free_rigid_rolling_resistance=0.002, **kwargs)
                np.testing.assert_allclose(pose[:3], pose_free[:3], atol=2.0e-3)
                self.assertGreater(abs(float(np.dot(pose[3:7], pose_free[3:7]))), 0.999)

    def test_composes_with_sleeping_anchors_torsion_and_warm_start(self):
        """Each production-style option keeps the Coulomb stopping distance."""
        expected = _coulomb_stopping_distance(0.3, 0.002)
        for options in (
            {"enable_sleeping": True},
            {"friction_anchor_beta": 0.2},
            {"torsion_radius": 0.01},
            {"contact_warmstart": True, "contact_matching": "sticky"},
        ):
            with self.subTest(**options):
                _, twist, travel = _simulate("sphere", 1.5, free_rigid_rolling_resistance=0.002, **_ROLLING, **options)
                self.assertLess(np.linalg.norm(twist[3:6]), 1.0e-3)
                self.assertAlmostEqual(travel, expected, delta=0.15 * expected)

    def test_repeated_and_captured_runs_are_bitwise_identical(self):
        kwargs = {"free_rigid_rolling_resistance": 0.001, **_ROLLING}
        first, _, _ = _simulate("sphere", 0.3, **kwargs)
        second, _, _ = _simulate("sphere", 0.3, **kwargs)
        captured, _, _ = _simulate("sphere", 0.3, graph=True, **kwargs)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first, captured)

    def test_default_is_off(self):
        model, _ = _scene("sphere")
        self.assertIsNone(SolverFeatherPGS(model)._rolling_support_load)
        default, _, _ = _simulate("sphere", 0.3, **_ROLLING)
        explicit, _, _ = _simulate("sphere", 0.3, free_rigid_rolling_resistance=0.0, **_ROLLING)
        np.testing.assert_array_equal(default, explicit)

    def test_invalid_options_raise(self):
        model, _ = _scene("sphere")
        for value in (-1.0e-3, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                SolverFeatherPGS(model, free_rigid_rolling_resistance=value)
        with self.assertRaises(NotImplementedError):
            SolverFeatherPGS(
                model, solve_mode="matrix_free", velocity_iterations=1, free_rigid_rolling_resistance=1.0e-3
            )


@unittest.skipUnless(wp.is_cuda_available(), "the kernel test runs on CUDA")
class TestRollingResistanceLaw(unittest.TestCase):
    """One free root on one support: the impulse budget, the normal axis and the support's spin."""

    def _apply(self, omega, support_omega, normal_impulse, resistance, inertia=0.01):
        device = wp.get_device("cuda")
        load = np.zeros((1, SUPPORT_LOAD_FIELDS), dtype=np.int64)
        load[0, 2] = round(normal_impulse * SUPPORT_LOAD_SCALE)
        load[0, 3:6] = np.round(np.asarray(support_omega) * normal_impulse * SUPPORT_LOAD_SCALE)
        load[0, 6] = round(normal_impulse * SUPPORT_LOAD_SCALE)
        joint_qd = wp.array(np.array([0.0, 0.0, 0.0, *omega], dtype=np.float32), device=device)
        identity = wp.array([wp.transform_identity()], dtype=wp.transform, device=device)
        wp.launch(
            apply_free_root_rolling_resistance,
            dim=1,
            inputs=[
                wp.array([0, 1], dtype=int, device=device),
                wp.array([0], dtype=int, device=device),
                wp.array([0, 7], dtype=int, device=device),
                identity,
                identity,
                wp.zeros(1, dtype=wp.int32, device=device),
                wp.array([wp.mat33(np.eye(3) * inertia)], dtype=wp.mat33, device=device),
                wp.ones(1, dtype=int, device=device),
                wp.zeros(1, dtype=int, device=device),
                wp.ones(1, dtype=int, device=device),
                wp.array(load, dtype=wp.int64, device=device),
                resistance,
                wp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=float, device=device),
            ],
            outputs=[joint_qd],
            device=device,
        )
        return joint_qd.numpy()[3:6]

    def test_budget_bounds_the_removed_momentum(self):
        # |L| = 0.01 * 2 = 0.02 N m s; budget = 1e-3 m * 5 N s = 5e-3, so 25% of the rolling rate goes.
        np.testing.assert_allclose(self._apply((2.0, 0.0, 1.0), (0, 0, 0), 5.0, 1.0e-3), (1.5, 0.0, 1.0), rtol=1e-5)
        # A budget above |L| stops rolling and leaves spin about the normal.
        np.testing.assert_allclose(self._apply((2.0, 0.0, 1.0), (0, 0, 0), 50.0, 1.0e-3), (0.0, 0.0, 1.0), atol=1e-6)

    def test_rolling_is_relative_to_the_support(self):
        np.testing.assert_allclose(self._apply((2.0, 0.0, 0.0), (2.0, 0.0, 0.0), 50.0, 1.0e-3), (2.0, 0.0, 0.0))
        np.testing.assert_allclose(
            self._apply((3.0, 0.0, 0.0), (2.0, 0.0, 0.0), 50.0, 1.0e-3), (2.0, 0.0, 0.0), atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
