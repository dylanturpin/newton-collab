# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Free rigid bodies must not gain energy from their gyroscopic bias."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _body_omega(state, body=0):
    q = wp.quat(*state.body_q.numpy()[body, 3:])
    omega = wp.vec3(*state.body_qd.numpy()[body, 3:])
    return np.asarray(wp.quat_rotate_inv(q, omega), dtype=np.float64)


def _model(device, inertia, omega, *, armature=0.0, mass=1.0):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.add_body(mass=mass, inertia=wp.mat33(inertia), com=wp.vec3(0.03, -0.02, 0.01))
    model = builder.finalize(device=device)
    model.joint_armature.fill_(armature)
    state, output = model.state(), model.state()
    state.joint_qd.assign(np.array([0.1, 0.2, 0.3, *omega], dtype=np.float32))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return model, state, output


def test_free_spin_energy(test, device):
    inertia = np.diag([0.001, 0.02, 0.0205])
    for mode in ("split", "matrix_free"):
        if mode == "matrix_free" and not wp.get_device(device).is_cuda:
            continue
        model, state, output = _model(device, inertia, (100.0, 20.0, 50.0))
        solver = newton.solvers.SolverFeatherPGS(model, pgs_mode=mode, angular_damping=0.0)
        control = model.control()
        omega = _body_omega(state)
        energy = 0.5 * omega @ inertia @ omega
        momentum = np.linalg.norm(inertia @ omega)
        with test.subTest(mode=mode):
            for step in range(1000):
                solver.step(state, output, control, None, 1.0 / 240.0)
                state, output = output, state
                omega = _body_omega(state)
                observed = 0.5 * omega @ inertia @ omega
                test.assertTrue(np.isfinite(observed), f"nonfinite energy at step {step}")
                test.assertAlmostEqual(observed / energy, 1.0, delta=1.0e-3, msg=f"step {step}")
            test.assertAlmostEqual(np.linalg.norm(inertia @ omega) / momentum, 1.0, delta=1.0e-3)
            np.testing.assert_allclose(state.body_qd.numpy()[0, :3], [0.1, 0.2, 0.3], atol=2.0e-4)


def test_principal_axis_torque(test, device):
    inertia = np.diag([0.04, 0.02, 0.05])
    for armature in (0.0, 0.03):
        model, state, output = _model(device, inertia, (0.0, 0.0, 2.0), armature=armature)
        solver = newton.solvers.SolverFeatherPGS(model, angular_damping=0.0)
        control = model.control()
        torque = 0.1
        dt = 1.0 / 240.0
        for _ in range(120):
            state.body_f.assign(np.array([[0.0, 0.0, 0.0, 0.0, 0.0, torque]], dtype=np.float32))
            solver.step(state, output, control, None, dt)
            state, output = output, state
        expected = 2.0 + 120 * dt * torque / (inertia[2, 2] + armature)
        test.assertAlmostEqual(float(state.body_qd.numpy()[0, 5]), expected, delta=1.0e-4)


def test_rotated_inertia_and_armature(test, device):
    rotation = wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 2.0, 3.0)), 0.7)
    basis = np.asarray(wp.quat_to_matrix(rotation), dtype=np.float64).reshape(3, 3)
    local_basis = np.asarray(wp.quat_to_matrix(wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.4))).reshape(3, 3)
    for scale, armature in ((1.0e-5, 0.0), (1.0, 0.003), (1.0e5, 0.0)):
        inertia = scale * local_basis @ np.diag([0.001, 0.02, 0.0205]) @ local_basis.T
        omega = basis @ local_basis @ np.array([20.0, 4.0, 10.0])
        model, state, output = _model(device, inertia, omega, mass=scale)
        rotor = armature * np.array([1.0, 2.0, 3.0])
        model.joint_armature.assign(np.array([0.0, 0.0, 0.0, *rotor], dtype=np.float32))
        q = state.joint_q.numpy()
        q[3:7] = np.asarray(rotation)
        state.joint_q.assign(q)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        solver = newton.solvers.SolverFeatherPGS(model, angular_damping=0.0)
        control = model.control()
        w = _body_omega(state)
        initial = w @ inertia @ w + np.dot(rotor, omega * omega)
        for _ in range(200):
            solver.step(state, output, control, None, 1.0 / 240.0)
            state, output = output, state
        w = _body_omega(state)
        omega = state.body_qd.numpy()[0, 3:].astype(np.float64)
        final = w @ inertia @ w + np.dot(rotor, omega * omega)
        test.assertAlmostEqual(final / initial, 1.0, delta=1.0e-3, msg=f"scale={scale}, armature={armature}")


def test_gyro_precession_converges(test, device):
    inertia = np.diag([0.001, 0.02, 0.0205])
    inverse = np.linalg.inv(inertia)
    initial = np.array([20.0, 4.0, 10.0])
    duration = 0.2

    def derivative(omega):
        return inverse @ np.cross(inertia @ omega, omega)

    # Independent double-precision Euler-equation reference in the body frame.
    reference = initial.copy()
    h = duration / 4096
    for _ in range(4096):
        k1 = derivative(reference)
        k2 = derivative(reference + 0.5 * h * k1)
        k3 = derivative(reference + 0.5 * h * k2)
        k4 = derivative(reference + h * k3)
        reference += h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    errors = []
    for steps in (48, 96):
        model, state, output = _model(device, inertia, initial)
        solver = newton.solvers.SolverFeatherPGS(model, angular_damping=0.0)
        control = model.control()
        for _ in range(steps):
            solver.step(state, output, control, None, duration / steps)
            state, output = output, state
        errors.append(np.linalg.norm(_body_omega(state) - reference))
    test.assertLess(errors[0], 0.1)
    test.assertLess(errors[1], 0.35 * errors[0])


class TestFeatherPGSGyroscopic(unittest.TestCase):
    pass


for _device in get_test_devices():
    add_function_test(TestFeatherPGSGyroscopic, "test_free_spin_energy", test_free_spin_energy, devices=[_device])
    add_function_test(
        TestFeatherPGSGyroscopic, "test_principal_axis_torque", test_principal_axis_torque, devices=[_device]
    )
    add_function_test(
        TestFeatherPGSGyroscopic,
        "test_rotated_inertia_and_armature",
        test_rotated_inertia_and_armature,
        devices=[_device],
    )
    add_function_test(
        TestFeatherPGSGyroscopic, "test_gyro_precession_converges", test_gyro_precession_converges, devices=[_device]
    )


if __name__ == "__main__":
    unittest.main()
