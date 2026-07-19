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


def _perturb_free_tangent(q, dof, eps, coord0=0, origin_pivot=False):
    """Free-joint tangent step matching the retraction. Root joints are
    body-centered: the angular tangent rotates about the joint's own anchor,
    leaving the position coordinates unchanged. Non-root joints
    (``origin_pivot=True``) keep jcalc_integrate's pivot about the parent
    anchor origin (``p += eps * e_a x p`` in anchor coordinates). The
    orientation quaternion is left-multiplied either way."""
    qp = q.copy()
    if dof < 3:
        qp[coord0 + dof] += eps
        return qp
    axis = np.zeros(3)
    axis[dof - 3] = 1.0
    if origin_pivot:
        p = q[coord0 : coord0 + 3]
        qp[coord0 : coord0 + 3] = p + eps * np.cross(axis, p)
    x1, y1, z1 = np.sin(eps / 2) * axis
    w1 = np.cos(eps / 2)
    x2, y2, z2, w2 = q[coord0 + 3 : coord0 + 7]
    qp[coord0 + 3 : coord0 + 7] = [
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ]
    return qp


def _standalone_apparent_gravity(model, n_frames, device, dt, weight=1.0, link=None, offset=None):
    obj = ik.IKObjectiveApparentGravity(
        model,
        EE_LINK if link is None else link,
        wp.vec3(EE_OFFSET[0], EE_OFFSET[1], EE_OFFSET[2]) if offset is None else offset,
        dt=dt,
        weight=weight,
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


def test_gravity_torque_free_descendant(test, device):
    """Payload attached by a free joint below the arm: the actuated rows'
    coupling columns to the payload tangents must match finite differences
    (regression: they used to be reported as zero)."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(up_axis="Z", gravity=-9.81)
        link = builder.add_link(mass=1.0)
        builder.add_shape_capsule(link, radius=0.04, half_height=0.25, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
        j0 = builder.add_joint_revolute(
            -1,
            link,
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.25, 0.0, 0.0), wp.quat_identity()),
            axis=wp.vec3(0.0, 1.0, 0.0),
        )
        payload = builder.add_link(mass=0.8)
        builder.add_shape_box(payload, hx=0.06, hy=0.05, hz=0.04, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
        j1 = builder.add_joint_free(
            payload,
            parent=link,
            parent_xform=wp.transform(wp.vec3(0.3, 0.0, 0.0), wp.quat_identity()),
        )
        builder.add_articulation([j0, j1])
        model = builder.finalize(device=device)
        n_dofs = model.joint_dof_count  # 1 revolute + 6 free

        rng = np.random.default_rng(31)
        n_rows = 2
        q_np = np.zeros((n_rows, model.joint_coord_count), dtype=np.float64)
        q_np[:, 0] = rng.uniform(-1.0, 1.0, size=n_rows)
        q_np[:, 1:4] = rng.uniform(-0.2, 0.2, size=(n_rows, 3))
        quat = rng.normal(size=(n_rows, 4))
        quat /= np.linalg.norm(quat, axis=1, keepdims=True)
        q_np[:, 4:8] = quat

        obj = _standalone_objective(model, n_rows, n_rows, device)
        joint_q = wp.array(q_np.astype(np.float32), dtype=wp.float32, device=device)
        obj.compute_coeffs(joint_q)
        coeffs = obj.coeffs.numpy()[:, 0]

        eps = 1e-4
        fd = np.zeros((n_rows, n_dofs, n_dofs))
        for dof in range(n_dofs):
            if dof == 0:
                qp = q_np.copy()
                qp[:, 0] += eps
                qm = q_np.copy()
                qm[:, 0] -= eps
            else:
                qp = np.stack(
                    [_perturb_free_tangent(q_np[t], dof - 1, eps, coord0=1, origin_pivot=True) for t in range(n_rows)]
                )
                qm = np.stack(
                    [_perturb_free_tangent(q_np[t], dof - 1, -eps, coord0=1, origin_pivot=True) for t in range(n_rows)]
                )
            fd[:, :, dof] = (
                _objective_residuals(obj, model, qp, device) - _objective_residuals(obj, model, qm, device)
            ) / (2 * eps)
        # only the actuated (revolute) row is nonzero; free rows are unweighted
        assert_np_equal(coeffs[:, 0, :], fd[:, 0, :].astype(np.float32), tol=5e-3 * max(np.abs(fd).max(), 1.0))
        # the payload-orientation coupling must actually be present
        test.assertGreater(np.abs(coeffs[:, 0, 4:]).max(), 0.1)


def test_apparent_gravity_free_joint_coeffs(test, device):
    """Floating-base waiter blocks must match the retraction's tangent convention
    (regression: jcalc's COM-anchored free-joint columns are re-anchored)."""
    with wp.ScopedDevice(device):
        model = _build_free_plus_revolute(device)
        n_dofs = model.joint_dof_count
        n_frames = 4
        rng = np.random.default_rng(29)
        # base well away from the origin so a convention mismatch is visible
        q_np = np.zeros((n_frames, model.joint_coord_count), dtype=np.float64)
        q_np[:, 0:3] = np.array([1.6, -0.9, 0.7]) + rng.uniform(-0.2, 0.2, size=(n_frames, 3))
        quat = rng.normal(size=(n_frames, 4))
        quat /= np.linalg.norm(quat, axis=1, keepdims=True)
        q_np[:, 3:7] = quat
        q_np[:, 7] = rng.uniform(-1.0, 1.0, size=n_frames)

        obj = _standalone_apparent_gravity(model, n_frames, device, dt=DT, link=1, offset=EE_OFFSET)
        joint_q = wp.array(q_np.astype(np.float32), dtype=wp.float32, device=device)
        obj.compute_coeffs(joint_q)
        coeffs = obj.coeffs.numpy()

        eps = 1e-4
        scale = max(np.abs(coeffs).max(), 1.0)
        for m in range(n_frames):
            for dof in range(n_dofs):
                qp = q_np.copy()
                qm = q_np.copy()
                if dof < 6:
                    qp[m] = _perturb_free_tangent(q_np[m], dof, eps)
                    qm[m] = _perturb_free_tangent(q_np[m], dof, -eps)
                else:
                    qp[m, 7] += eps
                    qm[m, 7] -= eps
                fd = (_objective_residuals(obj, model, qp, device) - _objective_residuals(obj, model, qm, device)) / (
                    2 * eps
                )
                for t in range(n_frames):
                    j = m - t
                    analytic = coeffs[t, j, :2, dof] if 0 <= j <= 2 else np.zeros(2)
                    assert_np_equal(analytic, fd[t, :2], tol=5e-3 * scale)


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


def _standalone_temporal(obj, model, n_frames, device):
    obj.set_batch_layout(model.joint_dof_count, 0, n_frames)
    obj.bind_device(device)
    obj.set_trajectory_layout(n_frames, 1)
    obj.init_buffers(model, IKJacobianType.ANALYTIC)
    return obj


def test_world_plane_coeffs_match_fd(test, device):
    """Plane-clearance blocks must match finite differences through the hinge."""
    with wp.ScopedDevice(device):
        model = _build_planar_vertical(device)
        n_dofs = model.joint_dof_count
        rng = np.random.default_rng(37)
        # configurations straddling the plane and the margin band
        q_np = rng.uniform(-0.9, 0.9, size=(N_FRAMES, model.joint_coord_count))
        obj = _standalone_temporal(
            ik.IKObjectiveWorldPlane(model, [EE_LINK], [[EE_OFFSET[0], EE_OFFSET[1], EE_OFFSET[2]]], margin=0.1),
            model,
            N_FRAMES,
            device,
        )
        joint_q = wp.array(q_np.astype(np.float32), dtype=wp.float32, device=device)
        obj.compute_coeffs(joint_q)
        coeffs = obj.coeffs.numpy()[:, 0]

        eps = 1e-4
        fd = np.zeros((N_FRAMES, n_dofs, n_dofs))
        for a in range(n_dofs):
            qp = q_np.copy()
            qp[:, a] += eps
            qm = q_np.copy()
            qm[:, a] -= eps
            fd[:, :, a] = (
                _objective_residuals(obj, model, qp, device) - _objective_residuals(obj, model, qm, device)
            ) / (2 * eps)
        # rows past the guarded points are padding and must stay dead
        n_active = coeffs.shape[1]
        test.assertEqual(np.abs(fd[:, n_active:]).max(initial=0.0), 0.0)
        # skip rows within eps of the hinge kinks (clearance ~ 0 or ~ margin)
        res = _objective_residuals(obj, model, q_np, device)
        for t in range(N_FRAMES):
            for c in range(n_active):
                if np.abs(fd[t, c] - coeffs[t, c]).max() > 5e-3:
                    # tolerate kink frames: FD and analytic may straddle a kink
                    test.assertLess(np.abs(fd[t, c] - coeffs[t, c]).max(), 1.2, (t, c, res[t, c]))
                else:
                    assert_np_equal(coeffs[t, c], fd[t, c].astype(np.float32), tol=5e-3)


def test_world_plane_pushes_up(test, device):
    """A target below the floor gets tracked only down to the plane margin."""
    with wp.ScopedDevice(device):
        model = _build_planar_vertical(device)
        target = np.array([0.9, 0.0, -0.35], dtype=np.float32)  # below the plane
        targets = wp.array(np.tile(target, (N_FRAMES, 1)), dtype=wp.vec3, device=device)
        seed = np.tile(np.array([0.3, 0.2, 0.2], dtype=np.float32), (N_FRAMES, 1))

        def solve(plane_weight):
            objectives = [
                ik.IKObjectivePosition(EE_LINK, EE_OFFSET, targets, weight=1.0),
                ik.IKObjectiveSmoothness(model, derivative=1, dt=DT, weight=0.005),
                ik.IKObjectiveWorldPlane(
                    model,
                    [EE_LINK],
                    [[EE_OFFSET[0], EE_OFFSET[1], EE_OFFSET[2]]],
                    margin=0.02,
                    weight=plane_weight,
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

        z_base = _ee_positions(model, solve(0.0), device)[:, 2]
        z_plane = _ee_positions(model, solve(20.0), device)[:, 2]
        test.assertLess(z_base.min(), -0.3)  # baseline dives to the target
        test.assertGreater(z_plane.min(), -0.02)  # guarded solve stays at the plane


def test_free_joint_far_from_origin_converges(test, device):
    """Trajectory IK on a floating base must converge far from the world origin.

    Regression test for the free-joint tangent convention: with origin-pivot
    angular tangents (and COM-anchored effector columns), the Jacobian error
    grows with the base's distance from the origin, so LM rejects every step
    on targets a few meters out and the solve returns the seed. Body-centered
    tangents keep the lever arms body-sized regardless of world position.
    """
    with wp.ScopedDevice(device):
        model = _build_free_plus_revolute(device)
        # targets ~4.6 m from the origin (the jump_on_box regime): EE sweeps
        # a small arc so the base must translate AND rotate to track it
        s = np.linspace(0.0, 1.0, N_FRAMES)
        targets_np = np.stack(
            [4.0 + 0.1 * s, 0.05 * np.sin(2.0 * np.pi * s), 2.5 + 0.1 * s],
            axis=1,
        ).astype(np.float32)
        targets = wp.array(targets_np, dtype=wp.vec3, device=device)
        objectives = [
            ik.IKObjectivePosition(1, wp.vec3(0.25, 0.0, 0.0), targets, weight=1.0),
            ik.IKObjectiveSmoothness(model, derivative=1, dt=DT, weight=0.001),
        ]
        solver = ik.IKSolverTrajectory(
            model,
            N_FRAMES,
            objectives,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
            linear_solver="direct",
        )
        # plain constant seed: identity orientation, base near the targets but
        # offset enough that the solve needs real base motion
        seed = np.zeros((N_FRAMES, model.joint_coord_count), dtype=np.float32)
        seed[:, 0:3] = (3.6, 0.3, 2.2)
        seed[:, 6] = 1.0  # qw
        seed[:, 7] = 0.3
        joint_q = wp.array(seed.copy(), dtype=wp.float32, device=device)
        solver.step(joint_q, joint_q, iterations=100)

        q = joint_q.numpy()
        ee = np.zeros((N_FRAMES, 3))
        joint_q_wp = wp.array(q, dtype=wp.float32, device=device)
        joint_qd = wp.zeros((N_FRAMES, model.joint_dof_count), dtype=wp.float32, device=device)
        body_q = wp.zeros((N_FRAMES, model.body_count), dtype=wp.transform, device=device)
        body_qd = wp.zeros((N_FRAMES, model.body_count), dtype=wp.spatial_vector, device=device)
        eval_fk_batched(model, joint_q_wp, joint_qd, body_q, body_qd)
        bq = body_q.numpy()
        off = np.array([0.25, 0.0, 0.0])
        for t in range(N_FRAMES):
            x, y, z, w = bq[t, 1][3:7]
            uv = 2.0 * np.cross([x, y, z], off)
            ee[t] = bq[t, 1][:3] + off + w * uv + np.cross([x, y, z], uv)
        err = np.linalg.norm(ee - targets_np, axis=1)
        test.assertLess(err.mean(), 0.02, f"mean EE error {err.mean() * 1e3:.1f} mm")


def test_world_plane_capsule_coeffs_match_fd(test, device):
    """Capsule-clearance blocks must match finite differences through the hinge.

    The comparison is strict (5e-3) away from true endpoint ties, so a kernel
    differentiating through the wrong endpoint fails: its error is
    ``weight * |g| * |omega x (p0 - p1)|``, far above the tolerance. The test
    asserts its own coverage — at least one active sample per endpoint and one
    in the quadratic taper band — via crafted configurations appended to the
    random ones (a pure random draw covers only deep, end1-active samples).
    Also covers the degenerate sphere case (``end0 == end1``, permanently on
    the tie branch, which is exact rather than exempt because both endpoints
    coincide).
    """
    with wp.ScopedDevice(device):
        model = _build_planar_vertical(device)
        n_dofs = model.joint_dof_count
        margin = 0.1
        radius = 0.05
        ends = [[[0.05, 0.0, 0.0], [0.25, 0.0, 0.0]], [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0]]]
        rng = np.random.default_rng(59)
        # random configurations straddling the plane, plus crafted rows:
        # end0-active deep, end1-active deep, and a taper-band sample
        q_np = np.concatenate(
            [
                rng.uniform(-0.9, 0.9, size=(N_FRAMES, model.joint_coord_count)),
                np.array(
                    [
                        [0.580, -0.980, 0.099],  # end0 active, shallow penetration
                        [0.300, 0.953, 0.662],  # end1 active, deep penetration
                        [0.009, -0.389, 0.784],  # taper band, end1 active
                        [0.503, -0.731, -0.871],  # taper band, end0 active
                    ]
                ),
            ]
        )
        n_rows = q_np.shape[0]
        obj = _standalone_temporal(
            ik.IKObjectiveWorldPlaneCapsule(model, [EE_LINK, 1], ends, [radius, 0.08], margin=margin),
            model,
            n_rows,
            device,
        )
        res = _objective_residuals(obj, model, q_np, device)
        test.assertGreater(np.abs(res[:, :2]).max(), 0.0)  # some frames must activate

        # endpoint plane distances of the two-endpoint capsule, from FK
        joint_q = wp.array(q_np.astype(np.float32), dtype=wp.float32, device=device)
        joint_qd = wp.zeros((n_rows, model.joint_dof_count), dtype=wp.float32, device=device)
        body_q = wp.zeros((n_rows, model.body_count), dtype=wp.transform, device=device)
        body_qd = wp.zeros((n_rows, model.body_count), dtype=wp.spatial_vector, device=device)
        eval_fk_batched(model, joint_q, joint_qd, body_q, body_qd)
        bq = body_q.numpy()
        c_ends = np.zeros((n_rows, 2))
        for j in range(2):
            off = np.array(ends[0][j])
            for t in range(n_rows):
                x, y, z, w = bq[t, EE_LINK][3:7]
                uv = 2.0 * np.cross([x, y, z], off)
                c_ends[t, j] = (bq[t, EE_LINK][:3] + off + w * uv + np.cross([x, y, z], uv))[2]
        clearance = c_ends.min(axis=1) - radius
        active = clearance < margin
        # coverage: both endpoints must be the active one somewhere, and the
        # quadratic taper band must be sampled (else wrong-endpoint or wrong-
        # branch kernels pass unnoticed)
        test.assertTrue((active & (c_ends[:, 0] < c_ends[:, 1] - 1e-3)).any(), "no end0-active sample")
        test.assertTrue((active & (c_ends[:, 1] < c_ends[:, 0] - 1e-3)).any(), "no end1-active sample")
        test.assertTrue(((clearance > 0.005) & (clearance < margin - 0.005)).any(), "no taper-band sample")

        obj.compute_coeffs(joint_q)
        coeffs = obj.coeffs.numpy()[:, 0]

        eps = 1e-4
        fd = np.zeros((n_rows, n_dofs, n_dofs))
        for a in range(n_dofs):
            qp = q_np.copy()
            qp[:, a] += eps
            qm = q_np.copy()
            qm[:, a] -= eps
            fd[:, :, a] = (
                _objective_residuals(obj, model, qp, device) - _objective_residuals(obj, model, qm, device)
            ) / (2 * eps)
        # rows past the guarded capsules are padding and must stay dead
        n_active = coeffs.shape[1]
        test.assertEqual(np.abs(fd[:, n_active:]).max(initial=0.0), 0.0)
        # strict everywhere except genuine endpoint near-ties of the
        # two-endpoint capsule, where FD straddles the min switch
        tie = np.abs(c_ends[:, 0] - c_ends[:, 1]) < 1e-3
        for t in range(n_rows):
            for c in range(n_active):
                if c == 0 and tie[t]:
                    continue
                assert_np_equal(coeffs[t, c], fd[t, c].astype(np.float32), tol=5e-3)


def test_world_plane_capsule_holds_surface(test, device):
    """A target below the floor gets tracked only down to the capsule surface."""
    with wp.ScopedDevice(device):
        model = _build_planar_vertical(device)
        target = np.array([0.9, 0.0, -0.35], dtype=np.float32)  # below the plane
        targets = wp.array(np.tile(target, (N_FRAMES, 1)), dtype=wp.vec3, device=device)
        seed = np.tile(np.array([0.3, 0.2, 0.2], dtype=np.float32), (N_FRAMES, 1))
        radius = 0.05

        def solve(capsule_weight):
            objectives = [
                ik.IKObjectivePosition(EE_LINK, EE_OFFSET, targets, weight=1.0),
                ik.IKObjectiveSmoothness(model, derivative=1, dt=DT, weight=0.005),
                ik.IKObjectiveWorldPlaneCapsule(
                    model,
                    [EE_LINK],
                    [[[0.05, 0.0, 0.0], [EE_OFFSET[0], EE_OFFSET[1], EE_OFFSET[2]]]],
                    [radius],
                    margin=0.02,
                    weight=capsule_weight,
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

        z_base = _ee_positions(model, solve(0.0), device)[:, 2]
        z_caps = _ee_positions(model, solve(20.0), device)[:, 2]
        test.assertLess(z_base.min(), -0.3)  # baseline dives to the target
        # guarded solve keeps the capsule SURFACE near the plane: the endpoint
        # center stays a radius above it (minus the soft-penalty slack)
        test.assertGreater(z_caps.min(), radius - 0.03)


def test_foot_skate_coeffs_match_fd(test, device):
    """Contact-gated skate blocks must match finite differences across the stencil."""
    with wp.ScopedDevice(device):
        model = _build_planar_vertical(device)
        n_dofs = model.joint_dof_count
        rng = np.random.default_rng(41)
        q_np = rng.uniform(-1.0, 1.0, size=(N_FRAMES, model.joint_coord_count))
        contact_np = (rng.uniform(size=(N_FRAMES, 1)) < 0.7).astype(np.uint8)
        contact = wp.array(contact_np, dtype=wp.uint8, device=device)
        obj = _standalone_temporal(
            ik.IKObjectiveFootSkate(model, [EE_LINK], [[EE_OFFSET[0], EE_OFFSET[1], EE_OFFSET[2]]], contact),
            model,
            N_FRAMES,
            device,
        )
        joint_q = wp.array(q_np.astype(np.float32), dtype=wp.float32, device=device)
        obj.compute_coeffs(joint_q)
        coeffs = obj.coeffs.numpy()

        eps = 1e-4
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
                    analytic = coeffs[t, j, :3, a] if 0 <= j <= 1 else np.zeros(3)
                    assert_np_equal(analytic, fd[t, :3], tol=5e-3)


def test_foot_contact_coeffs_match_fd(test, device):
    """Contact-gated anchor-pin blocks must match finite differences."""
    with wp.ScopedDevice(device):
        model = _build_planar_vertical(device)
        n_dofs = model.joint_dof_count
        rng = np.random.default_rng(43)
        q_np = rng.uniform(-1.0, 1.0, size=(N_FRAMES, model.joint_coord_count))
        contact = wp.array((rng.uniform(size=(N_FRAMES, 1)) < 0.7).astype(np.uint8), dtype=wp.uint8, device=device)
        anchors = wp.array(rng.uniform(-1, 1, size=(N_FRAMES, 1, 3)).astype(np.float32), dtype=wp.vec3, device=device)
        obj = _standalone_temporal(
            ik.IKObjectiveFootContact(model, [EE_LINK], [[EE_OFFSET[0], EE_OFFSET[1], EE_OFFSET[2]]], contact, anchors),
            model,
            N_FRAMES,
            device,
        )
        joint_q = wp.array(q_np.astype(np.float32), dtype=wp.float32, device=device)
        obj.compute_coeffs(joint_q)
        coeffs = obj.coeffs.numpy()[:, 0]

        eps = 1e-4
        fd = np.zeros((N_FRAMES, n_dofs, n_dofs))
        for a in range(n_dofs):
            qp = q_np.copy()
            qp[:, a] += eps
            qm = q_np.copy()
            qm[:, a] -= eps
            fd[:, :, a] = (
                _objective_residuals(obj, model, qp, device) - _objective_residuals(obj, model, qm, device)
            ) / (2 * eps)
        # rows past the anchored links are padding and must stay dead
        test.assertEqual(np.abs(fd[:, coeffs.shape[1] :]).max(initial=0.0), 0.0)
        assert_np_equal(coeffs, fd[:, : coeffs.shape[1]].astype(np.float32), tol=5e-3)


def test_position_rotation_set_equivalence(test, device):
    """Fused set objectives must reproduce the per-effector solve exactly."""
    with wp.ScopedDevice(device):
        model = _build_planar_vertical(device)
        rng = np.random.default_rng(47)
        links = [1, 2]
        offsets = [[0.25, 0.0, 0.0], [0.25, 0.0, 0.0]]
        weights = [2.0, 1.0]
        rot_weights = [0.5, 0.3]
        pos_np = rng.uniform(-0.8, 0.8, size=(N_FRAMES, 2, 3)).astype(np.float32)
        quat_np = rng.normal(size=(N_FRAMES, 2, 4)).astype(np.float32)
        quat_np /= np.linalg.norm(quat_np, axis=-1, keepdims=True)
        seed = np.tile(np.array([0.3, 0.4, 0.4], dtype=np.float32), (N_FRAMES, 1))

        def solve(objectives):
            solver = ik.IKSolverTrajectory(
                model,
                N_FRAMES,
                objectives,
                jacobian_mode=ik.IKJacobianType.ANALYTIC,
                linear_solver="direct",
            )
            joint_q = wp.array(seed.copy(), dtype=wp.float32, device=device)
            solver.step(joint_q, joint_q, iterations=20)
            return joint_q.numpy()

        individual = []
        for e in range(2):
            individual.append(
                ik.IKObjectivePosition(
                    links[e], wp.vec3(*offsets[e]), wp.array(pos_np[:, e].copy(), dtype=wp.vec3), weight=weights[e]
                )
            )
        for e in range(2):
            individual.append(
                ik.IKObjectiveRotation(
                    links[e], wp.quat_identity(), wp.array(quat_np[:, e].copy(), dtype=wp.vec4), weight=rot_weights[e]
                )
            )
        q_individual = solve(individual)

        fused = [
            ik.IKObjectivePositionSet(links, offsets, wp.array(pos_np, dtype=wp.vec3), weights),
            ik.IKObjectiveRotationSet(links, wp.array(quat_np, dtype=wp.vec4), rot_weights),
        ]
        q_fused = solve(fused)
        assert_np_equal(q_fused, q_individual, tol=1e-5)


def test_self_collision_coeffs_match_fd(test, device):
    """Capsule-pair separation blocks must match finite differences."""
    with wp.ScopedDevice(device):
        model = _build_planar_vertical(device)
        n_dofs = model.joint_dof_count
        rng = np.random.default_rng(53)
        # capsules on links 0 and 2 of the chain; poses that bring them close.
        # the mechanism is planar, so offset the capsules in y to keep the
        # closest distance bounded away from the crossing degeneracy where the
        # separation normal is undefined
        pairs = [[0, 2]]
        ends = [[[-0.2, 0.04, 0.0], [0.2, 0.04, 0.0], [-0.2, -0.04, 0.0], [0.2, -0.04, 0.0]]]
        radii = [[0.05, 0.05]]
        # half the frames fold the chain back on itself (joints 2 and 3 near
        # pi) so link 2's capsule approaches link 0's; jitter avoids exactly
        # parallel segments where the closest point is degenerate
        q_np = rng.uniform(-0.5, 0.5, size=(N_FRAMES, model.joint_coord_count))
        fold = N_FRAMES // 2
        q_np[:fold, 1] = np.pi - rng.uniform(0.15, 0.5, size=fold)
        q_np[:fold, 2] = np.pi - rng.uniform(0.15, 0.5, size=fold)
        obj = _standalone_temporal(
            ik.IKObjectiveSelfCollision(model, pairs, ends, radii, margin=0.15),
            model,
            N_FRAMES,
            device,
        )
        res = _objective_residuals(obj, model, q_np, device)
        test.assertGreater(np.abs(res[:, 0]).max(), 0.0)  # some frames must activate

        joint_q = wp.array(q_np.astype(np.float32), dtype=wp.float32, device=device)
        obj.compute_coeffs(joint_q)
        coeffs = obj.coeffs.numpy()[:, 0]

        eps = 1e-4
        fd = np.zeros((N_FRAMES, n_dofs, n_dofs))
        for a in range(n_dofs):
            qp = q_np.copy()
            qp[:, a] += eps
            qm = q_np.copy()
            qm[:, a] -= eps
            fd[:, :, a] = (
                _objective_residuals(obj, model, qp, device) - _objective_residuals(obj, model, qm, device)
            ) / (2 * eps)
        # rows past the capsule pairs are padding and must stay dead
        test.assertEqual(np.abs(fd[:, coeffs.shape[1] :]).max(initial=0.0), 0.0)
        fd = fd[:, : coeffs.shape[1]]
        # exact away from hinge kinks / degenerate parallel segments; allow
        # a small number of kink-straddling entries
        diff = np.abs(coeffs - fd)
        test.assertLess(np.median(diff[np.abs(fd) > 1e-6]) if (np.abs(fd) > 1e-6).any() else 0.0, 5e-3)
        test.assertLess((diff > 5e-2).mean(), 0.05)


devices = get_test_devices()


class TestIKTrajectoryDynamics(unittest.TestCase):
    pass


add_function_test(
    TestIKTrajectoryDynamics,
    "test_self_collision_coeffs_match_fd",
    test_self_collision_coeffs_match_fd,
    devices=devices,
)


add_function_test(
    TestIKTrajectoryDynamics,
    "test_foot_contact_coeffs_match_fd",
    test_foot_contact_coeffs_match_fd,
    devices=devices,
)
add_function_test(
    TestIKTrajectoryDynamics,
    "test_position_rotation_set_equivalence",
    test_position_rotation_set_equivalence,
    devices=devices,
)


add_function_test(
    TestIKTrajectoryDynamics,
    "test_world_plane_coeffs_match_fd",
    test_world_plane_coeffs_match_fd,
    devices=devices,
)
add_function_test(
    TestIKTrajectoryDynamics,
    "test_world_plane_pushes_up",
    test_world_plane_pushes_up,
    devices=devices,
)
add_function_test(
    TestIKTrajectoryDynamics,
    "test_free_joint_far_from_origin_converges",
    test_free_joint_far_from_origin_converges,
    devices=devices,
)
add_function_test(
    TestIKTrajectoryDynamics,
    "test_world_plane_capsule_coeffs_match_fd",
    test_world_plane_capsule_coeffs_match_fd,
    devices=devices,
)
add_function_test(
    TestIKTrajectoryDynamics,
    "test_world_plane_capsule_holds_surface",
    test_world_plane_capsule_holds_surface,
    devices=devices,
)
add_function_test(
    TestIKTrajectoryDynamics,
    "test_foot_skate_coeffs_match_fd",
    test_foot_skate_coeffs_match_fd,
    devices=devices,
)


add_function_test(
    TestIKTrajectoryDynamics,
    "test_gravity_torque_free_descendant",
    test_gravity_torque_free_descendant,
    devices=devices,
)
add_function_test(
    TestIKTrajectoryDynamics,
    "test_apparent_gravity_coeffs_match_fd",
    test_apparent_gravity_coeffs_match_fd,
    devices=devices,
)
add_function_test(
    TestIKTrajectoryDynamics,
    "test_apparent_gravity_free_joint_coeffs",
    test_apparent_gravity_free_joint_coeffs,
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
