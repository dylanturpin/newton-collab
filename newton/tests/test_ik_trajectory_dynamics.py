# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for dynamics-aware trajectory IK objectives."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
import newton.ik as ik
from newton._src.sim.ik.ik_common import IKJacobianType, eval_fk_batched
from newton.tests.unittest_utils import add_function_test, assert_np_equal, get_test_devices

# Shared dimensions: keep identical across tests so the specialized tile
# kernels are compiled once per device (see test_ik_trajectory.py).
N_FRAMES = 8
DT = 0.05
EE_LINK = 2
EE_OFFSET = wp.vec3(0.25, 0.0, 0.0)


def _build_planar_vertical(device):
    """3 revolute-Y joints with links along +x: swings in the xz gravity plane."""
    builder = newton.ModelBuilder(up_axis="Z", gravity=-9.81)
    parent = -1
    joints = []
    for i in range(3):
        link = builder.add_link(mass=1.0)
        builder.add_shape_capsule(link, radius=0.04, half_height=0.25, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
        joints.append(
            builder.add_joint_revolute(
                parent,
                link,
                parent_xform=wp.transform(wp.vec3(0.5 if i else 0.0, 0.0, 0.0), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(-0.25, 0.0, 0.0), wp.quat_identity()),
                axis=wp.vec3(0.0, 1.0, 0.0),
            )
        )
        parent = link
    builder.add_articulation(joints)
    return builder.finalize(device=device)


def _build_free_plus_revolute(device):
    builder = newton.ModelBuilder(up_axis="Z", gravity=-9.81)
    root = builder.add_link(mass=2.0)
    builder.add_shape_box(root, hx=0.1, hy=0.1, hz=0.1, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
    j0 = builder.add_joint_free(root)
    link = builder.add_link(mass=1.0)
    builder.add_shape_capsule(link, radius=0.04, half_height=0.25, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
    j1 = builder.add_joint_revolute(
        root,
        link,
        parent_xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.25, 0.0, 0.0), wp.quat_identity()),
        axis=wp.vec3(0.0, 1.0, 0.0),
    )
    builder.add_articulation([j0, j1])
    return builder.finalize(device=device)


def _standalone_objective(model, n_frames, n_rows, device, weight=1.0):
    """Wire a gravity-torque objective the way IKSolverTrajectory would."""
    obj = ik.IKObjectiveGravityTorque(model, weight=weight)
    obj.set_batch_layout(model.joint_dof_count, 0, n_rows)
    obj.bind_device(device)
    obj.set_trajectory_layout(n_frames, max(n_rows // n_frames, 1))
    obj.init_buffers(model, IKJacobianType.ANALYTIC)
    return obj


def _objective_residuals(obj, model, joint_q_np, device):
    n_rows = joint_q_np.shape[0]
    joint_q = wp.array(joint_q_np.astype(np.float32), dtype=wp.float32, device=device)
    joint_qd = wp.zeros((n_rows, model.joint_dof_count), dtype=wp.float32, device=device)
    body_q = wp.zeros((n_rows, model.body_count), dtype=wp.transform, device=device)
    body_qd = wp.zeros((n_rows, model.body_count), dtype=wp.spatial_vector, device=device)
    eval_fk_batched(model, joint_q, joint_qd, body_q, body_qd)
    residuals = wp.zeros((n_rows, model.joint_dof_count), dtype=wp.float32, device=device)
    problem_idx = wp.zeros(n_rows, dtype=wp.int32, device=device)
    obj.compute_residuals(body_q, joint_q, model, residuals, 0, problem_idx)
    return residuals.numpy()


def _ee_positions(model, joint_q_np, device):
    n_rows = joint_q_np.shape[0]
    joint_q = wp.array(joint_q_np.astype(np.float32), dtype=wp.float32, device=device)
    joint_qd = wp.zeros((n_rows, model.joint_dof_count), dtype=wp.float32, device=device)
    body_q = wp.zeros((n_rows, model.body_count), dtype=wp.transform, device=device)
    body_qd = wp.zeros((n_rows, model.body_count), dtype=wp.spatial_vector, device=device)
    eval_fk_batched(model, joint_q, joint_qd, body_q, body_qd)
    bq = body_q.numpy()
    out = np.zeros((n_rows, 3), dtype=np.float64)
    off = np.array([EE_OFFSET[0], EE_OFFSET[1], EE_OFFSET[2]])
    for t in range(n_rows):
        x, y, z, w = bq[t, EE_LINK][3:7]
        uv = 2.0 * np.cross([x, y, z], off)
        out[t] = bq[t, EE_LINK][:3] + off + w * uv + np.cross([x, y, z], uv)
    return out


def test_gravity_torque_matches_inverse_dynamics(test, device):
    """Residual rows must equal eval_inverse_dynamics' gravity_force."""
    with wp.ScopedDevice(device):
        model = _build_planar_vertical(device)
        rng = np.random.default_rng(11)
        q_np = rng.uniform(-1.5, 1.5, size=(N_FRAMES, model.joint_coord_count))
        obj = _standalone_objective(model, N_FRAMES, N_FRAMES, device)
        res = _objective_residuals(obj, model, q_np, device)

        state = model.state()
        inv = model.inverse_dynamics()
        for t in range(N_FRAMES):
            state.joint_q.assign(q_np[t].astype(np.float32))
            state.joint_qd.zero_()
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            newton.eval_inverse_dynamics(model, state, newton.InverseDynamics.EvalType.GRAVITY_FORCE, inv)
            assert_np_equal(res[t], inv.gravity_force.numpy(), tol=1e-4)


def test_gravity_torque_coeffs_match_fd(test, device):
    """Analytic coefficient blocks must match finite differences of the residuals."""
    with wp.ScopedDevice(device):
        model = _build_planar_vertical(device)
        n_dofs = model.joint_dof_count
        rng = np.random.default_rng(13)
        q_np = rng.uniform(-1.5, 1.5, size=(N_FRAMES, model.joint_coord_count))
        obj = _standalone_objective(model, N_FRAMES, N_FRAMES, device)

        joint_q = wp.array(q_np.astype(np.float32), dtype=wp.float32, device=device)
        obj.compute_coeffs(joint_q)
        coeffs = obj.coeffs.numpy()[:, 0]

        eps = 1e-3
        fd = np.zeros((N_FRAMES, n_dofs, n_dofs))
        for a in range(n_dofs):
            qp = q_np.copy()
            qp[:, a] += eps
            qm = q_np.copy()
            qm[:, a] -= eps
            fd[:, :, a] = (
                _objective_residuals(obj, model, qp, device) - _objective_residuals(obj, model, qm, device)
            ) / (2 * eps)
        # tolerance dominated by fp32 round-off of ~20 N·m residuals under FD
        assert_np_equal(coeffs, fd.astype(np.float32), tol=5e-3)
        # the gravity Hessian is symmetric
        assert_np_equal(coeffs, np.transpose(coeffs, (0, 2, 1)), tol=1e-6)


def test_gravity_torque_free_joint(test, device):
    """Free-joint rows are zero; base-tangent columns match finite differences."""
    with wp.ScopedDevice(device):
        model = _build_free_plus_revolute(device)
        rng = np.random.default_rng(17)
        n_rows = 2
        q_np = np.zeros((n_rows, model.joint_coord_count), dtype=np.float64)
        q_np[:, 0:3] = rng.uniform(-0.5, 0.5, size=(n_rows, 3))
        quat = rng.normal(size=(n_rows, 4))
        quat /= np.linalg.norm(quat, axis=1, keepdims=True)
        q_np[:, 3:7] = quat
        q_np[:, 7] = rng.uniform(-1.5, 1.5, size=n_rows)

        obj = _standalone_objective(model, n_rows, n_rows, device)
        res = _objective_residuals(obj, model, q_np, device)
        test.assertEqual(np.abs(res[:, :6]).max(), 0.0)

        joint_q = wp.array(q_np.astype(np.float32), dtype=wp.float32, device=device)
        obj.compute_coeffs(joint_q)
        coeffs = obj.coeffs.numpy()[:, 0]

        def perturb(q, dof, eps):
            """Tangent perturbation matching the free-joint retraction
            (world-origin spatial velocity: p += eps * e_a x p, left-multiplied quat)."""
            qp = q.copy()
            if dof < 3:
                qp[dof] += eps
            elif dof < 6:
                axis = np.zeros(3)
                axis[dof - 3] = 1.0
                qp[0:3] = q[0:3] + eps * np.cross(axis, q[0:3])
                x1, y1, z1 = np.sin(eps / 2) * axis
                w1 = np.cos(eps / 2)
                x2, y2, z2, w2 = q[3:7]
                qp[3:7] = [
                    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                ]
            else:
                qp[7] += eps
            return qp

        eps = 1e-3
        for dof in range(model.joint_dof_count):
            qp = np.stack([perturb(q_np[t], dof, eps) for t in range(n_rows)])
            qm = np.stack([perturb(q_np[t], dof, -eps) for t in range(n_rows)])
            fd_col = (
                _objective_residuals(obj, model, qp, device)[:, 6] - _objective_residuals(obj, model, qm, device)[:, 6]
            ) / (2 * eps)
            assert_np_equal(coeffs[:, 6, dof], fd_col.astype(np.float32), tol=5e-3)


def test_gravity_torque_reduces_torque(test, device):
    """With redundancy, the objective trades a little tracking for much less torque."""
    with wp.ScopedDevice(device):
        model = _build_planar_vertical(device)
        target = np.array([0.7, 0.0, -0.5], dtype=np.float32)
        targets = wp.array(np.tile(target, (N_FRAMES, 1)), dtype=wp.vec3, device=device)
        seed = np.tile(np.array([0.3, 0.4, 0.4], dtype=np.float32), (N_FRAMES, 1))

        def solve(gravity_weight):
            # weight=0 keeps the residual layout (and kernel specialization)
            # identical to the active case
            objectives = [
                ik.IKObjectivePosition(EE_LINK, EE_OFFSET, targets, weight=5.0),
                ik.IKObjectiveSmoothness(model, derivative=1, dt=DT, weight=0.005),
                ik.IKObjectiveGravityTorque(model, weight=gravity_weight),
            ]
            solver = ik.IKSolverTrajectory(
                model,
                N_FRAMES,
                objectives,
                jacobian_mode=ik.IKJacobianType.ANALYTIC,
                linear_solver="direct",
            )
            joint_q = wp.array(seed.copy(), dtype=wp.float32, device=device)
            solver.step(joint_q, joint_q, iterations=60)
            return joint_q.numpy()

        eval_obj = _standalone_objective(model, N_FRAMES, N_FRAMES, device)
        q_base = solve(0.0)
        q_grav = solve(0.05)
        tau_base = np.abs(_objective_residuals(eval_obj, model, q_base, device)).sum(axis=1).mean()
        tau_grav = np.abs(_objective_residuals(eval_obj, model, q_grav, device)).sum(axis=1).mean()
        err_base = np.linalg.norm(_ee_positions(model, q_base, device) - target, axis=1).max()
        err_grav = np.linalg.norm(_ee_positions(model, q_grav, device) - target, axis=1).max()

        test.assertLess(err_base, 5e-3)
        test.assertLess(err_grav, 5e-2)
        test.assertLess(tau_grav, 0.7 * tau_base)


def _standalone_apparent_gravity(model, n_frames, device, dt, weight=1.0):
    obj = ik.IKObjectiveApparentGravity(
        model, EE_LINK, wp.vec3(EE_OFFSET[0], EE_OFFSET[1], EE_OFFSET[2]), dt=dt, weight=weight
    )
    obj.set_batch_layout(model.joint_dof_count, 0, n_frames)
    obj.bind_device(device)
    obj.set_trajectory_layout(n_frames, 1)
    obj.init_buffers(model, IKJacobianType.ANALYTIC)
    return obj


def test_apparent_gravity_coeffs_match_fd(test, device):
    """Waiter-objective blocks must match the full finite-difference Jacobian."""
    with wp.ScopedDevice(device):
        model = _build_planar_vertical(device)
        n_dofs = model.joint_dof_count
        rng = np.random.default_rng(23)
        q_np = rng.uniform(-1.0, 1.0, size=(N_FRAMES, model.joint_coord_count))
        obj = _standalone_apparent_gravity(model, N_FRAMES, device, dt=DT)

        res = _objective_residuals(obj, model, q_np, device)
        test.assertEqual(np.abs(res[N_FRAMES - 2 :, :]).max(), 0.0)
        test.assertEqual(np.abs(res[:, 2:]).max(), 0.0)

        joint_q = wp.array(q_np.astype(np.float32), dtype=wp.float32, device=device)
        obj.compute_coeffs(joint_q)
        coeffs = obj.coeffs.numpy()

        eps = 1e-4
        scale = max(np.abs(coeffs).max(), 1.0)
        for m in range(N_FRAMES):
            for a in range(n_dofs):
                qp = q_np.copy()
                qp[m, a] += eps
                qm = q_np.copy()
                qm[m, a] -= eps
                fd = (_objective_residuals(obj, model, qp, device) - _objective_residuals(obj, model, qm, device)) / (
                    2 * eps
                )
                for t in range(N_FRAMES):
                    j = m - t
                    analytic = coeffs[t, j, :2, a] if 0 <= j <= 2 else np.zeros(2)
                    # fp32 FD noise on accelerations scales with the coeff magnitude
                    assert_np_equal(analytic, fd[t, :2], tol=5e-3 * scale)


def test_apparent_gravity_static_tilt(test, device):
    """At rest the residual is the plate's tangential gravity component."""
    with wp.ScopedDevice(device):
        model = _build_planar_vertical(device)
        obj = _standalone_apparent_gravity(model, N_FRAMES, device, dt=DT)
        q_np = np.tile(np.array([0.3, -0.5, 0.4], dtype=np.float64), (N_FRAMES, 1))
        res = _objective_residuals(obj, model, q_np, device)

        joint_q = wp.array(q_np.astype(np.float32), dtype=wp.float32, device=device)
        joint_qd = wp.zeros((N_FRAMES, model.joint_dof_count), dtype=wp.float32, device=device)
        body_q = wp.zeros((N_FRAMES, model.body_count), dtype=wp.transform, device=device)
        body_qd = wp.zeros((N_FRAMES, model.body_count), dtype=wp.spatial_vector, device=device)
        eval_fk_batched(model, joint_q, joint_qd, body_q, body_qd)
        quat = body_q.numpy()[0, EE_LINK][3:7]

        def rotate(q, v):
            x, y, z, w = q
            uv = 2.0 * np.cross([x, y, z], v)
            return v + w * uv + np.cross([x, y, z], uv)

        up = np.array([0.0, 0.0, 9.81])
        for k, axis in enumerate((obj._tangent_u, obj._tangent_v)):
            world_axis = rotate(quat, np.array([axis[0], axis[1], axis[2]]))
            test.assertAlmostEqual(res[0, k], np.dot(world_axis, up), places=3)


def test_apparent_gravity_balances(test, device):
    """A fast dash with the objective carries far less tangential apparent force."""
    with wp.ScopedDevice(device):
        model = _build_planar_vertical(device)
        dt = DT
        # fast vertical-plane dash: down 0.5 m and back within ~0.6 s
        s = np.concatenate([np.linspace(0, 1, N_FRAMES // 2), np.linspace(1, 0, N_FRAMES - N_FRAMES // 2)])
        targets_np = np.stack(
            [0.9 - 0.25 * s, np.zeros_like(s), 0.6 * s - 0.2],
            axis=1,
        ).astype(np.float32)
        targets = wp.array(targets_np, dtype=wp.vec3, device=device)
        seed = np.tile(np.array([0.3, 0.4, 0.4], dtype=np.float32), (N_FRAMES, 1))

        def solve(weight):
            objectives = [
                ik.IKObjectivePosition(EE_LINK, EE_OFFSET, targets, weight=2.0),
                ik.IKObjectiveSmoothness(model, derivative=2, dt=dt, weight=0.001),
                ik.IKObjectiveApparentGravity(
                    model, EE_LINK, wp.vec3(EE_OFFSET[0], EE_OFFSET[1], EE_OFFSET[2]), dt=dt, weight=weight
                ),
            ]
            solver = ik.IKSolverTrajectory(
                model,
                N_FRAMES,
                objectives,
                jacobian_mode=ik.IKJacobianType.ANALYTIC,
                linear_solver="direct",
            )
            joint_q = wp.array(seed.copy(), dtype=wp.float32, device=device)
            solver.step(joint_q, joint_q, iterations=60)
            return joint_q.numpy()

        eval_obj = _standalone_apparent_gravity(model, N_FRAMES, device, dt=dt)
        res_base = _objective_residuals(eval_obj, model, solve(0.0), device)
        res_wtr = _objective_residuals(eval_obj, model, solve(0.05), device)
        tang_base = np.linalg.norm(res_base[: N_FRAMES - 2, :2], axis=1)
        tang_wtr = np.linalg.norm(res_wtr[: N_FRAMES - 2, :2], axis=1)
        test.assertLess(tang_wtr.mean(), 0.5 * tang_base.mean())


devices = get_test_devices()


class TestIKTrajectoryDynamics(unittest.TestCase):
    pass


add_function_test(
    TestIKTrajectoryDynamics,
    "test_apparent_gravity_coeffs_match_fd",
    test_apparent_gravity_coeffs_match_fd,
    devices=devices,
)
add_function_test(
    TestIKTrajectoryDynamics,
    "test_apparent_gravity_static_tilt",
    test_apparent_gravity_static_tilt,
    devices=devices,
)
add_function_test(
    TestIKTrajectoryDynamics,
    "test_apparent_gravity_balances",
    test_apparent_gravity_balances,
    devices=devices,
)


add_function_test(
    TestIKTrajectoryDynamics,
    "test_gravity_torque_matches_inverse_dynamics",
    test_gravity_torque_matches_inverse_dynamics,
    devices=devices,
)
add_function_test(
    TestIKTrajectoryDynamics,
    "test_gravity_torque_coeffs_match_fd",
    test_gravity_torque_coeffs_match_fd,
    devices=devices,
)
add_function_test(
    TestIKTrajectoryDynamics,
    "test_gravity_torque_free_joint",
    test_gravity_torque_free_joint,
    devices=devices,
)
add_function_test(
    TestIKTrajectoryDynamics,
    "test_gravity_torque_reduces_torque",
    test_gravity_torque_reduces_torque,
    devices=devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
