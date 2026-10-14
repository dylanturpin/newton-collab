# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental free-rigid rolling resistance of SolverFeatherPGS."""

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS

_DT = 1.0 / 480.0
_RADIUS = 0.03


def _scene(body: str, tilt_deg: float = 0.0):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    SolverFeatherPGS.register_custom_attributes(builder)
    table = builder.add_body(xform=wp.transform((0.0, 0.0, -0.05), wp.quat_identity()), is_kinematic=True)
    builder.add_shape_box(table, hx=2.0, hy=2.0, hz=0.05)
    rotation = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.radians(tilt_deg))
    if body == "sphere":
        obj = builder.add_body(xform=wp.transform((0.0, 0.0, _RADIUS), rotation), mass=0.1)
        builder.add_shape_sphere(obj, radius=_RADIUS)
    else:
        tilt = math.radians(tilt_deg)
        z = _RADIUS * (math.cos(tilt) + math.sin(tilt)) + 1.0e-4
        obj = builder.add_body(xform=wp.transform((0.0, 0.0, z), rotation), mass=0.1)
        builder.add_shape_box(obj, hx=_RADIUS, hy=_RADIUS, hz=_RADIUS)
    return builder.finalize(), obj


def _simulate(body, seconds, *, tilt_deg=0.0, linear=(0.0, 0.0, 0.0), angular=(0.0, 0.0, 0.0), **solver_kwargs):
    model, obj = _scene(body, tilt_deg)
    model.rigid_contact_max = 256
    solver = SolverFeatherPGS(model, solve_mode="matrix_free", iterations=32, deterministic=True, **solver_kwargs)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=256, deterministic=True)
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
    for _ in range(int(seconds / _DT)):
        state_0.clear_forces()
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, _DT)
        state_0, state_1 = state_1, state_0
    pose, twist = state_0.body_q.numpy()[obj], state_0.body_qd.numpy()[obj]
    return pose, twist, float(np.linalg.norm(pose[:2] - start[:2]))


@unittest.skipUnless(wp.is_cuda_available(), "FeatherPGS matrix-free requires CUDA")
class TestFreeRigidRollingResistance(unittest.TestCase):
    def test_rolling_sphere_stops_within_the_coulomb_distance(self):
        rolling = {"linear": (0.0, -0.3, 0.0), "angular": (10.0, 0.0, 0.0)}
        _, twist_free, travel_free = _simulate("sphere", 1.0, **rolling)
        _, twist, travel = _simulate("sphere", 1.0, free_rigid_rolling_resistance=0.002, **rolling)
        self.assertGreater(np.linalg.norm(twist_free[3:6]), 9.0)
        self.assertLess(np.linalg.norm(twist[3:6]), 1.0e-3)
        # Rolling ball: v^2 = 2 (mu_r g / r) / (1 + I / (m r^2)) d, so d = 0.3^2 * 1.4 / (2 * 0.002 * 9.81 / 0.03).
        self.assertAlmostEqual(travel, 0.3**2 * 1.4 / (2.0 * 0.002 * 9.81 / _RADIUS), delta=0.03)
        self.assertGreater(travel_free, 0.25)

    def test_toppling_spin_and_sliding_cube_are_unchanged(self):
        for kwargs in ({"tilt_deg": 40.0}, {"angular": (0.0, 0.0, 6.0)}, {"linear": (0.5, 0.0, 0.0)}):
            with self.subTest(**kwargs):
                pose_free, _, _ = _simulate("box", 1.0, **kwargs)
                pose, _, _ = _simulate("box", 1.0, free_rigid_rolling_resistance=0.002, **kwargs)
                np.testing.assert_allclose(pose[:3], pose_free[:3], atol=2.0e-3)
                self.assertGreater(abs(float(np.dot(pose[3:7], pose_free[3:7]))), 0.999)

    def test_repeated_runs_are_bitwise_identical(self):
        kwargs = {"linear": (0.0, -0.3, 0.0), "angular": (10.0, 0.0, 0.0), "free_rigid_rolling_resistance": 0.001}
        first, _, _ = _simulate("sphere", 0.3, **kwargs)
        second, _, _ = _simulate("sphere", 0.3, **kwargs)
        np.testing.assert_array_equal(first, second)

    def test_negative_coefficients_raise(self):
        model, _ = _scene("sphere")
        with self.assertRaises(ValueError):
            SolverFeatherPGS(model, free_rigid_rolling_resistance=-1.0)


if __name__ == "__main__":
    unittest.main()
