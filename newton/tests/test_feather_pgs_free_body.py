# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_test_devices


class TestFeatherPGSFreeBody(unittest.TestCase):
    pass


def test_free_root_slide_roll_has_no_spurious_vertical_acceleration(test, device):
    builder = newton.ModelBuilder(gravity=-9.81)
    builder.default_shape_cfg.density = 1000.0

    cube = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()))
    builder.add_shape_box(cube, hx=0.02, hy=0.02, hz=0.02)

    model = builder.finalize(device=device)
    solver = SolverFeatherPGS(
        model,
        pgs_mode="matrix_free",
        pgs_iterations=1,
        pgs_warmstart=False,
    )

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()
    dt = 0.005

    pose = np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    twist = np.array([0.0, -1.6546488, 0.0, 13.849494, 0.0, 0.0], dtype=np.float32)

    state_0.body_q.assign(pose)
    state_0.joint_q.assign(pose.reshape(-1))
    state_0.body_qd.assign(twist.reshape(1, 6))
    state_0.joint_qd.assign(twist)

    state_0.clear_forces()
    solver.step(state_0, state_1, control, contacts, dt)

    joint_qd = state_1.joint_qd.numpy()
    body_qd = state_1.body_qd.numpy()[cube]

    expected_vz = -9.81 * dt
    test.assertAlmostEqual(float(joint_qd[2]), expected_vz, delta=1.0e-4)
    test.assertAlmostEqual(float(body_qd[2]), expected_vz, delta=1.0e-4)


def test_free_root_angular_damping_is_applied(test, device):
    builder = newton.ModelBuilder(gravity=0.0)
    cube = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()))
    builder.add_shape_box(cube, hx=0.02, hy=0.02, hz=0.02)

    model = builder.finalize(device=device)
    damping = 10.0
    dt = 0.005
    solver = SolverFeatherPGS(
        model,
        angular_damping=damping,
        pgs_mode="matrix_free",
        pgs_iterations=1,
        pgs_warmstart=False,
    )

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()

    pose = np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    twist = np.array([0.0, 0.0, 0.0, 4.0, -3.0, 2.0], dtype=np.float32)

    state_0.body_q.assign(pose)
    state_0.joint_q.assign(pose.reshape(-1))
    state_0.body_qd.assign(twist.reshape(1, 6))
    state_0.joint_qd.assign(twist)

    state_0.clear_forces()
    solver.step(state_0, state_1, control, contacts, dt)

    expected_angular = twist[3:6] * (1.0 - damping * dt)
    joint_qd = state_1.joint_qd.numpy()
    body_qd = state_1.body_qd.numpy()[cube]

    np.testing.assert_allclose(joint_qd[3:6], expected_angular, rtol=1.0e-5, atol=1.0e-5)
    np.testing.assert_allclose(body_qd[3:6], expected_angular, rtol=1.0e-5, atol=1.0e-5)


devices = get_test_devices()
for device in devices:
    add_function_test(
        TestFeatherPGSFreeBody,
        "test_free_root_slide_roll_has_no_spurious_vertical_acceleration",
        test_free_root_slide_roll_has_no_spurious_vertical_acceleration,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSFreeBody,
        "test_free_root_angular_damping_is_applied",
        test_free_root_angular_damping_is_applied,
        devices=[device],
    )


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=False)
