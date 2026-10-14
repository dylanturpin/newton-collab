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
    apply_free_rigid_rolling_resistance,
)
from newton.solvers import SolverFeatherPGS
from newton.tests.feather_pgs._fixtures import set_torsion_radius

_DT = 1.0 / 480.0
_RADIUS = 0.03
_MASS = 0.1
_ROLLING = {"linear": (0.0, -0.3, 0.0), "angular": (10.0, 0.0, 0.0)}
# Principal moments 0.6, 1 and 1.4 with an x-z product: a physically valid anisotropic inertia.
_ANISOTROPIC = np.array([[1.0, 0.0, 0.4], [0.0, 1.0, 0.0], [0.4, 0.0, 1.0]])


def _coulomb_stopping_distance(speed: float, resistance: float) -> float:
    """Return the rolling distance [m] of a solid sphere: v^2 = 2 d (resistance g / r) / (1 + I / (m r^2))."""
    return speed**2 * 1.4 / (2.0 * resistance * 9.81 / _RADIUS)


def _scene(body: str, tilt_deg: float = 0.0, inertia=None):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    SolverFeatherPGS.register_custom_attributes(builder)
    table = builder.add_body(xform=wp.transform((0.0, 0.0, -0.05), wp.quat_identity()), is_kinematic=True)
    builder.add_shape_box(table, hx=2.0, hy=2.0, hz=0.05)
    rotation = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.radians(tilt_deg))
    if body == "sphere":
        if inertia is None:
            obj = builder.add_body(xform=wp.transform((0.0, 0.0, _RADIUS), rotation), mass=_MASS)
            builder.add_shape_sphere(obj, radius=_RADIUS)
        else:
            obj = builder.add_body(
                xform=wp.transform((0.0, 0.0, _RADIUS), rotation),
                mass=_MASS,
                inertia=wp.mat33(*np.asarray(inertia, dtype=np.float32).ravel()),
                lock_inertia=True,
            )
            builder.add_shape_sphere(obj, radius=_RADIUS, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
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
    inertia=None,
    steps=None,
    **solver_kwargs,
):
    model, obj = _scene(body, tilt_deg, inertia)
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

    if steps == 1:
        step(state_0, state_1)
        state_0 = state_1
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
    pose, twist = state_0.body_q.numpy()[obj], state_0.body_qd.numpy()[obj]
    return pose, twist, float(np.linalg.norm(pose[:2] - start[:2]))


@unittest.skipUnless(wp.is_cuda_available(), "FeatherPGS matrix-free requires CUDA")
class TestFreeRigidRollingResistance(unittest.TestCase):
    def test_rolling_sphere_stops_at_the_coulomb_distance(self):
        """Stop a rolling sphere within 15% of the analytic Coulomb distance; keep it rolling when off."""
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
        """Apply the same law with solve_mode='split'."""
        _, twist, travel = _simulate("sphere", 1.0, solve_mode="split", free_rigid_rolling_resistance=0.002, **_ROLLING)
        self.assertLess(np.linalg.norm(twist[3:6]), 1.0e-3)
        expected = _coulomb_stopping_distance(0.3, 0.002)
        self.assertAlmostEqual(travel, expected, delta=0.15 * expected)

    def test_anisotropic_sphere_step_never_gains_energy(self):
        """Never end a step with more kinetic energy than the same step without resistance (anisotropic body)."""
        inertia = _ANISOTROPIC * 4.0e-5
        spin = {"angular": (1.0, 0.0, -3.0), "inertia": inertia, "steps": 1}
        _, twist_free, _ = _simulate("sphere", 0.0, **spin)

        def energy(twist):
            return 0.5 * _MASS * float(twist[:3] @ twist[:3]) + 0.5 * float(twist[3:6] @ inertia @ twist[3:6])

        # A full stop (large length) and a partial clamp (budget below the stopping impulse).
        for resistance in (1.0, 1.0e-3):
            with self.subTest(resistance=resistance):
                _, twist, _ = _simulate("sphere", 0.0, free_rigid_rolling_resistance=resistance, **spin)
                self.assertLessEqual(energy(twist), energy(twist_free) * (1.0 + 1.0e-5))
                self.assertLess(energy(twist), energy(twist_free))

    def test_toppling_spin_and_sliding_cube_are_unchanged(self):
        """Leave a toppling, spinning or sliding cube within 2 mm of the option-off pose."""
        for kwargs in ({"tilt_deg": 40.0}, {"angular": (0.0, 0.0, 6.0)}, {"linear": (0.5, 0.0, 0.0)}):
            with self.subTest(**kwargs):
                pose_free, _, _ = _simulate("box", 1.0, **kwargs)
                pose, _, _ = _simulate("box", 1.0, free_rigid_rolling_resistance=0.002, **kwargs)
                np.testing.assert_allclose(pose[:3], pose_free[:3], atol=2.0e-3)
                self.assertGreater(abs(float(np.dot(pose[3:7], pose_free[3:7]))), 0.999)

    def test_composes_with_sleeping_anchors_torsion_and_warm_start(self):
        """Keep the Coulomb stopping distance with each production-style option."""
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
        """Match two eager runs and a CUDA-graph replay bit for bit."""
        kwargs = {"free_rigid_rolling_resistance": 0.001, **_ROLLING}
        first, _, _ = _simulate("sphere", 0.3, **kwargs)
        second, _, _ = _simulate("sphere", 0.3, **kwargs)
        captured, _, _ = _simulate("sphere", 0.3, graph=True, **kwargs)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first, captured)

    def test_default_is_off(self):
        """Allocate nothing by default, and match an explicit zero bit for bit."""
        model, _ = _scene("sphere")
        self.assertIsNone(SolverFeatherPGS(model)._rolling_support_load)
        default, _, _ = _simulate("sphere", 0.3, **_ROLLING)
        explicit, _, _ = _simulate("sphere", 0.3, free_rigid_rolling_resistance=0.0, **_ROLLING)
        np.testing.assert_array_equal(default, explicit)

    def test_invalid_options_raise(self):
        """Reject negative and non-finite lengths, and velocity iterations in matrix-free mode."""
        model, _ = _scene("sphere")
        for value in (-1.0e-3, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                SolverFeatherPGS(model, free_rigid_rolling_resistance=value)
        with self.assertRaises(NotImplementedError):
            SolverFeatherPGS(
                model, solve_mode="matrix_free", velocity_iterations=1, free_rigid_rolling_resistance=1.0e-3
            )


class TestRollingResistanceLaw(unittest.TestCase):
    """One free rigid body on one support, through the production kernel on every available device."""

    devices = ("cpu", "cuda") if wp.is_cuda_available() else ("cpu",)

    def _apply(self, device, omega, inertia, budget, support_omega=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0)):
        normal_impulse = 1.0
        load = np.zeros((1, SUPPORT_LOAD_FIELDS), dtype=np.int64)
        load[0, 0:3] = np.round(np.asarray(normal) * normal_impulse * SUPPORT_LOAD_SCALE)
        load[0, 3:6] = np.round(np.asarray(support_omega) * normal_impulse * SUPPORT_LOAD_SCALE)
        load[0, 6] = round(normal_impulse * SUPPORT_LOAD_SCALE)
        joint_qd = wp.array(np.array([0.0, 0.0, 0.0, *omega], dtype=np.float32), device=device)
        identity = wp.array([wp.transform_identity()], dtype=wp.transform, device=device)
        inv_inertia = np.linalg.inv(np.asarray(inertia, dtype=np.float64)).astype(np.float32)
        wp.launch(
            apply_free_rigid_rolling_resistance,
            dim=1,
            inputs=[
                wp.array([0, 1], dtype=int, device=device),
                wp.array([0], dtype=int, device=device),
                wp.array([0, 7], dtype=int, device=device),
                identity,
                identity,
                wp.zeros(1, dtype=wp.int32, device=device),
                wp.array([wp.mat33(*inv_inertia.ravel())], dtype=wp.mat33, device=device),
                wp.ones(1, dtype=wp.int32, device=device),
                wp.zeros(1, dtype=int, device=device),
                wp.ones(1, dtype=int, device=device),
                wp.array(load, dtype=wp.int64, device=device),
                budget / normal_impulse,
                wp.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=float, device=device),
            ],
            outputs=[joint_qd],
            device=device,
        )
        return joint_qd.numpy()[3:6].astype(np.float64)

    def test_isotropic_budget_scales_the_rolling_rate(self):
        """Remove exactly the budget of tangent momentum, or stop rolling and keep spin when it suffices."""
        for device in self.devices:
            with self.subTest(device=device):
                inertia = np.eye(3) * 0.01
                # |L_t| = 0.02 and budget 5e-3 leave 75% of the rolling rate.
                np.testing.assert_allclose(
                    self._apply(device, (2.0, 0.0, 1.0), inertia, 5.0e-3), (1.5, 0.0, 1.0), rtol=1e-5
                )
                np.testing.assert_allclose(
                    self._apply(device, (2.0, 0.0, 1.0), inertia, 0.05), (0.0, 0.0, 1.0), atol=1e-6
                )

    def test_anisotropic_energy_never_increases(self):
        """Dissipate energy and bound the tangent impulse for valid anisotropic inertias, partial or full stop."""
        rng = np.random.default_rng(3)
        cases = [(_ANISOTROPIC, (1.0, 0.0, -3.0), budget) for budget in (1.0, 0.1, 1.0e-3)]
        for _ in range(20):
            axes, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            moments = np.sort(rng.uniform(0.2, 1.0, 3))
            moments[2] = min(moments[2], moments[0] + moments[1] - 1.0e-3)
            normal = rng.normal(size=3)
            cases.append(
                (axes @ np.diag(moments) @ axes.T, tuple(rng.normal(size=3) * 3.0), rng.uniform(0.01, 3.0), normal)
            )
        for device in self.devices:
            for index, case in enumerate(cases):
                inertia, omega, budget = case[:3]
                normal = case[3] / np.linalg.norm(case[3]) if len(case) > 3 else np.array([0.0, 0.0, 1.0])
                with self.subTest(device=device, case=index):
                    out = self._apply(device, omega, inertia, budget, normal=normal)
                    omega = np.asarray(omega)
                    impulse = inertia @ (out - omega)
                    self.assertLessEqual(0.5 * out @ inertia @ out, 0.5 * omega @ inertia @ omega * (1.0 + 1.0e-5))
                    self.assertLessEqual(np.linalg.norm(impulse), budget * (1.0 + 1.0e-4))
                    self.assertLess(abs(impulse @ normal), 1.0e-4 * max(budget, 1.0))

    def test_anisotropic_full_stop_zeroes_the_tangent_rate(self):
        """Stop the tangent angular velocity when the budget allows, letting the normal rate follow the inertia."""
        for device in self.devices:
            with self.subTest(device=device):
                out = self._apply(device, (1.0, 0.0, -3.0), _ANISOTROPIC, 10.0)
                np.testing.assert_allclose(out[:2], (0.0, 0.0), atol=1e-5)
                omega = np.array([1.0, 0.0, -3.0])
                self.assertLess(0.5 * out @ _ANISOTROPIC @ out, 0.5 * omega @ _ANISOTROPIC @ omega)

    def test_rolling_is_relative_to_the_support(self):
        """Leave co-rotation with a moving support alone and stop only the relative rate."""
        for device in self.devices:
            with self.subTest(device=device):
                inertia = np.eye(3) * 0.01
                np.testing.assert_allclose(
                    self._apply(device, (2.0, 0.0, 0.0), inertia, 1.0, (2.0, 0.0, 0.0)), (2.0, 0.0, 0.0)
                )
                np.testing.assert_allclose(
                    self._apply(device, (3.0, 0.0, 0.0), inertia, 1.0, (2.0, 0.0, 0.0)), (2.0, 0.0, 0.0), atol=1e-6
                )


if __name__ == "__main__":
    unittest.main()
