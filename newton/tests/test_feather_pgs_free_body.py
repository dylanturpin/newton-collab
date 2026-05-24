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


def _add_revolute_articulation(builder: newton.ModelBuilder) -> int:
    link = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.8), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1),
    )
    joint = builder.add_joint_revolute(parent=-1, child=link, axis=wp.vec3(0.0, 0.0, 1.0))
    builder.add_articulation([joint])
    return link


def _step_once(model: newton.Model, solver: SolverFeatherPGS, dt: float = 0.005):
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    contacts = model.collide(state_0)
    state_0.clear_forces()
    solver.step(state_0, state_1, control, contacts, dt)
    return state_1


def test_matrix_free_articulated_only_skips_free_rigid_mf_buffers(test, device):
    builder = newton.ModelBuilder(gravity=0.0)
    _add_revolute_articulation(builder)
    model = builder.finalize(device=device)

    solver = SolverFeatherPGS(
        model,
        pgs_mode="matrix_free",
        pgs_iterations=2,
        mf_max_constraints=64,
    )

    test.assertFalse(solver._has_free_rigid_bodies)
    test.assertEqual(solver.mf_body_a.shape, (solver.world_count, 1))
    test.assertEqual(solver.mf_meta_packed.shape, (solver.world_count, 4))
    test.assertIsNone(solver.mf_body_Hinv)
    test.assertFalse(hasattr(solver, "_debug_stage3_v_hat"))
    test.assertFalse(hasattr(solver, "_debug_position_rhs"))

    state_1 = _step_once(model, solver)
    test.assertTrue(np.all(np.isfinite(state_1.joint_qd.numpy())))


def test_matrix_free_mixed_scene_preserves_free_rigid_mf_rows(test, device):
    builder = newton.ModelBuilder(gravity=-9.81)
    _add_revolute_articulation(builder)
    cube = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.04), wp.quat_identity()), mass=1.0)
    builder.add_shape_box(cube, hx=0.05, hy=0.05, hz=0.05)
    builder.add_ground_plane()
    model = builder.finalize(device=device)

    solver = SolverFeatherPGS(
        model,
        pgs_mode="matrix_free",
        pgs_iterations=4,
        mf_max_constraints=64,
    )

    test.assertTrue(solver._has_free_rigid_bodies)
    test.assertTrue(solver._has_mixed_contacts)
    test.assertEqual(solver.mf_body_a.shape, (solver.world_count, 64))

    state_1 = _step_once(model, solver)
    test.assertTrue(np.all(np.isfinite(state_1.joint_qd.numpy())))
    test.assertGreater(int(np.max(solver.mf_constraint_count.numpy())), 0)


def test_zero_velocity_iterations_skip_velocity_post_solve(test, device):
    class NoVelocityPostSolver(SolverFeatherPGS):
        def _conclude_matrix_free_position_problem(self, dt: float) -> None:
            raise AssertionError("velocity-post setup should not run when pgs_velocity_iterations=0")

        def _run_matrix_free_velocity_post_solve(self) -> None:
            raise AssertionError("velocity-post solve should not run when pgs_velocity_iterations=0")

    builder = newton.ModelBuilder(gravity=0.0)
    _add_revolute_articulation(builder)
    model = builder.finalize(device=device)
    solver = NoVelocityPostSolver(
        model,
        pgs_mode="matrix_free",
        pgs_iterations=2,
        pgs_velocity_iterations=0,
    )

    _step_once(model, solver)


def test_pgs_debug_still_records_residual_logs(test, device):
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.ke = 5.0e4
    builder.default_shape_cfg.kd = 5.0e2
    builder.default_shape_cfg.mu = 0.75
    sphere = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.18), wp.quat_identity()), mass=1.0)
    builder.add_shape_sphere(sphere, radius=0.2)
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    solver = SolverFeatherPGS(
        model,
        pgs_mode="matrix_free",
        pgs_iterations=3,
        pgs_debug=True,
    )

    _step_once(model, solver)

    test.assertEqual(len(solver._pgs_convergence_log), 1)
    test.assertEqual(solver._pgs_convergence_log[0].shape, (3, 4))
    test.assertEqual(len(solver._pgs_ncp_residual_log), 1)
    test.assertEqual(solver._pgs_ncp_residual_log[0].shape, (3, solver.world_count, 6))
    test.assertFalse(hasattr(solver, "_debug_position_rhs"))


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
        "test_matrix_free_articulated_only_skips_free_rigid_mf_buffers",
        test_matrix_free_articulated_only_skips_free_rigid_mf_buffers,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSFreeBody,
        "test_matrix_free_mixed_scene_preserves_free_rigid_mf_rows",
        test_matrix_free_mixed_scene_preserves_free_rigid_mf_rows,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSFreeBody,
        "test_zero_velocity_iterations_skip_velocity_post_solve",
        test_zero_velocity_iterations_skip_velocity_post_solve,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSFreeBody,
        "test_pgs_debug_still_records_residual_logs",
        test_pgs_debug_still_records_residual_logs,
        devices=[device],
    )
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
