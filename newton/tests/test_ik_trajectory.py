# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
import newton.ik as ik
from newton._src.sim.ik.ik_common import eval_fk_batched
from newton._src.sim.ik.ik_trajectory_solver import (
    _CG_DOT_TILE,
    _CG_SERIAL_DOT_MAX_LENGTH,
    _SegmentedTiledDot,
    _swap_cg_tiled_dot,
)
from newton.tests.unittest_utils import (
    add_function_test,
    assert_np_equal,
    get_selected_cuda_test_devices,
    get_test_devices,
)

# Shared trajectory dimensions. Keeping the model/objective dimensions
# identical across tests lets the tests reuse the same specialized tile
# kernels, which keeps compile time down.
N_FRAMES = 12
DT = 0.1
EE_LINK = 1
EE_OFFSET = wp.vec3(0.5, 0.0, 0.0)

# ----------------------------------------------------------------------------
# helpers: planar 2-revolute baseline
# ----------------------------------------------------------------------------


def _build_two_link_planar(device) -> newton.Model:
    """Returns a singleton model with one 2-DOF planar arm."""
    builder = newton.ModelBuilder()

    link1 = builder.add_link(
        xform=wp.transform([0.5, 0.0, 0.0], wp.quat_identity()),
        mass=1.0,
    )
    joint1 = builder.add_joint_revolute(
        parent=-1,
        child=link1,
        parent_xform=wp.transform([0.0, 0.0, 0.0], wp.quat_identity()),
        child_xform=wp.transform([-0.5, 0.0, 0.0], wp.quat_identity()),
        axis=[0.0, 0.0, 1.0],
    )

    link2 = builder.add_link(
        xform=wp.transform([1.5, 0.0, 0.0], wp.quat_identity()),
        mass=1.0,
    )
    joint2 = builder.add_joint_revolute(
        parent=link1,
        child=link2,
        parent_xform=wp.transform([0.5, 0.0, 0.0], wp.quat_identity()),
        child_xform=wp.transform([-0.5, 0.0, 0.0], wp.quat_identity()),
        axis=[0.0, 0.0, 1.0],
    )

    builder.add_articulation([joint1, joint2])

    model = builder.finalize(device=device, requires_grad=True)
    return model


def _build_free_plus_revolute(device) -> newton.Model:
    """Returns a model whose root link has a FREE joint followed by one REV link."""
    builder = newton.ModelBuilder()

    link1 = builder.add_link(
        xform=wp.transform([0.0, 0.0, 0.0], wp.quat_identity()),
        mass=1.0,
    )
    joint1 = builder.add_joint_free(
        parent=-1,
        child=link1,
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform_identity(),
    )

    link2 = builder.add_link(
        xform=wp.transform([1.0, 0.0, 0.0], wp.quat_identity()),
        mass=1.0,
    )
    joint2 = builder.add_joint_revolute(
        parent=link1,
        child=link2,
        parent_xform=wp.transform([0.5, 0.0, 0.0], wp.quat_identity()),
        child_xform=wp.transform([-0.5, 0.0, 0.0], wp.quat_identity()),
        axis=[0.0, 0.0, 1.0],
    )

    builder.add_articulation([joint1, joint2])

    model = builder.finalize(device=device, requires_grad=True)
    return model


# ----------------------------------------------------------------------------
# common utilities
# ----------------------------------------------------------------------------


def _arc_targets(n_frames: int, angle_start: float = 0.2, angle_end: float = 1.0, radius: float = 1.6) -> np.ndarray:
    """Returns (n_frames, 3) reachable end-effector positions along an arc."""
    angles = np.linspace(angle_start, angle_end, n_frames)
    return np.stack(
        [radius * np.cos(angles), radius * np.sin(angles), np.zeros(n_frames)],
        axis=1,
    ).astype(np.float32)


def _ee_positions(model: newton.Model, joint_q: wp.array, ee_link: int, ee_offset: wp.vec3) -> np.ndarray:
    """Returns (n_rows, 3) end-effector world positions for every trajectory row."""
    n_rows = joint_q.shape[0]
    joint_qd = wp.zeros((n_rows, model.joint_dof_count), dtype=wp.float32)
    body_q = wp.zeros((n_rows, model.body_count), dtype=wp.transform)
    body_qd = wp.zeros((n_rows, model.body_count), dtype=wp.spatial_vector)
    eval_fk_batched(model, joint_q, joint_qd, body_q, body_qd)

    body_q_np = body_q.numpy()
    positions = np.zeros((n_rows, 3), dtype=np.float32)
    for row in range(n_rows):
        tf = wp.transform(*body_q_np[row, ee_link])
        ee_world = wp.transform_point(tf, ee_offset)
        positions[row] = [ee_world[0], ee_world[1], ee_world[2]]
    return positions


def _make_arc_tracking_objectives(model: newton.Model, targets_np: np.ndarray) -> list[ik.IKObjective]:
    """Position tracking + first-order smoothness (residual layout shared across tests)."""
    targets = wp.array(targets_np, dtype=wp.vec3)
    pos_obj = ik.IKObjectivePosition(
        link_index=EE_LINK,
        link_offset=EE_OFFSET,
        target_positions=targets,
    )
    smooth_obj = ik.IKObjectiveSmoothness(model, derivative=1, dt=DT, weight=0.02)
    return [pos_obj, smooth_obj]


# ----------------------------------------------------------------------------
# 1.  Trajectory convergence (arc tracking)
# ----------------------------------------------------------------------------


def _trajectory_convergence(test, device, mode: ik.IKJacobianType):
    with wp.ScopedDevice(device):
        model = _build_two_link_planar(device)
        targets_np = _arc_targets(N_FRAMES)

        solver = ik.IKSolverTrajectory(
            model,
            N_FRAMES,
            _make_arc_tracking_objectives(model, targets_np),
            jacobian_mode=mode,
            linear_solver="direct",
            lambda_initial=1e-3,
        )

        requires_grad = mode in (ik.IKJacobianType.AUTODIFF, ik.IKJacobianType.MIXED)
        joint_q = wp.zeros((N_FRAMES, model.joint_coord_count), dtype=wp.float32, requires_grad=requires_grad)

        initial = _ee_positions(model, joint_q, EE_LINK, EE_OFFSET)
        solver.step(joint_q, joint_q, iterations=20)
        final = _ee_positions(model, joint_q, EE_LINK, EE_OFFSET)

        for t in range(N_FRAMES):
            err0 = np.linalg.norm(initial[t] - targets_np[t])
            err1 = np.linalg.norm(final[t] - targets_np[t])
            test.assertLess(err1, err0, f"mode {mode} frame {t} did not improve")
            test.assertLess(err1, 5e-3, f"mode {mode} frame {t} final error too high ({err1:.5f})")


def test_trajectory_convergence_analytic(test, device):
    _trajectory_convergence(test, device, ik.IKJacobianType.ANALYTIC)


def test_trajectory_convergence_autodiff(test, device):
    _trajectory_convergence(test, device, ik.IKJacobianType.AUTODIFF)


# ----------------------------------------------------------------------------
# 2.  Linear problem vs dense numpy least squares
# ----------------------------------------------------------------------------


def test_trajectory_linear_reference_matches_dense(test, device):
    """Revolute-only model with purely linear residuals: the trajectory solver
    must land on the exact least-squares optimum computed densely in numpy.
    This validates the finite-difference stencils, boundary masking, weights,
    and block-banded assembly (derivative=2 exercises the superblock path)."""
    with wp.ScopedDevice(device):
        model = _build_two_link_planar(device)
        n_coords = model.joint_coord_count  # revolute only: tangent == coordinates

        w_ref, w_vel, w_acc = 1.0, 0.3, 0.15
        rng = np.random.default_rng(42)
        reference_np = rng.uniform(-0.8, 0.8, size=(N_FRAMES, n_coords)).astype(np.float32)
        reference_q = wp.array(reference_np, dtype=wp.float32)

        objectives = [
            ik.IKObjectiveJointReference(model, reference_q, weight=w_ref),
            ik.IKObjectiveSmoothness(model, derivative=1, dt=DT, weight=w_vel),
            ik.IKObjectiveSmoothness(model, derivative=2, dt=DT, weight=w_acc),
        ]
        solver = ik.IKSolverTrajectory(
            model,
            N_FRAMES,
            objectives,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
            linear_solver="direct",
            lambda_initial=1e-3,
        )

        joint_q = wp.zeros((N_FRAMES, n_coords), dtype=wp.float32)
        solver.step(joint_q, joint_q, iterations=30)
        q_solver = joint_q.numpy()

        # dense least-squares reference in float64
        n_vars = N_FRAMES * n_coords

        def unit(t, d):
            e = np.zeros(n_vars)
            e[t * n_coords + d] = 1.0
            return e

        rows = []
        rhs = []
        for t in range(N_FRAMES):
            for d in range(n_coords):
                rows.append(w_ref * unit(t, d))
                rhs.append(w_ref * float(reference_np[t, d]))
        s_vel = w_vel / DT
        for t in range(N_FRAMES - 1):
            for d in range(n_coords):
                rows.append(s_vel * (unit(t + 1, d) - unit(t, d)))
                rhs.append(0.0)
        s_acc = w_acc / DT**2
        for t in range(N_FRAMES - 2):
            for d in range(n_coords):
                rows.append(s_acc * (unit(t + 2, d) - 2.0 * unit(t + 1, d) + unit(t, d)))
                rhs.append(0.0)

        a_mat = np.array(rows)
        b_vec = np.array(rhs)
        x_opt = np.linalg.solve(a_mat.T @ a_mat, a_mat.T @ b_vec).reshape(N_FRAMES, n_coords)

        assert_np_equal(q_solver, x_opt.astype(np.float32), tol=1e-4)


# ----------------------------------------------------------------------------
# 3.  CG backend matches direct backend
# ----------------------------------------------------------------------------


def test_trajectory_cg_matches_direct(test, device):
    with wp.ScopedDevice(device):
        model = _build_two_link_planar(device)
        targets_np = _arc_targets(N_FRAMES)

        def solve(linear_solver):
            solver = ik.IKSolverTrajectory(
                model,
                N_FRAMES,
                _make_arc_tracking_objectives(model, targets_np),
                jacobian_mode=ik.IKJacobianType.ANALYTIC,
                linear_solver=linear_solver,
                lambda_initial=1e-3,
            )
            joint_q = wp.zeros((N_FRAMES, model.joint_coord_count), dtype=wp.float32)
            solver.step(joint_q, joint_q, iterations=20)
            return joint_q.numpy()

        q_direct = solve("direct")
        q_cg = solve("cg")

        assert_np_equal(q_cg, q_direct, tol=1e-4)


def test_trajectory_spike_matches_direct(test, device):
    with wp.ScopedDevice(device):
        model = _build_two_link_planar(device)
        targets_np = _arc_targets(N_FRAMES)

        def solve(linear_solver, **kwargs):
            solver = ik.IKSolverTrajectory(
                model,
                N_FRAMES,
                _make_arc_tracking_objectives(model, targets_np),
                jacobian_mode=ik.IKJacobianType.ANALYTIC,
                linear_solver=linear_solver,
                lambda_initial=1e-3,
                **kwargs,
            )
            joint_q = wp.zeros((N_FRAMES, model.joint_coord_count), dtype=wp.float32)
            solver.step(joint_q, joint_q, iterations=20)
            return joint_q.numpy()

        q_direct = solve("direct")
        for n_parts in (2, 4):
            q_spike = solve("spike", spike_partitions=n_parts)
            assert_np_equal(q_spike, q_direct, tol=1e-4)


# ----------------------------------------------------------------------------
# 4.  Fixed frames stay at their seed values
# ----------------------------------------------------------------------------


def test_trajectory_fixed_frames(test, device):
    with wp.ScopedDevice(device):
        model = _build_two_link_planar(device)
        targets_np = _arc_targets(N_FRAMES)

        solver = ik.IKSolverTrajectory(
            model,
            N_FRAMES,
            _make_arc_tracking_objectives(model, targets_np),
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
            linear_solver="direct",
            lambda_initial=1e-3,
            fixed_frames=[0, N_FRAMES - 1],
        )

        rng = np.random.default_rng(7)
        seed_np = rng.uniform(-0.3, 0.3, size=(N_FRAMES, model.joint_coord_count)).astype(np.float32)
        joint_q = wp.array(seed_np, dtype=wp.float32)

        solver.step(joint_q, joint_q, iterations=20)
        q_final = joint_q.numpy()

        # fixed frames are bitwise-unchanged
        assert_np_equal(q_final[0], seed_np[0])
        assert_np_equal(q_final[N_FRAMES - 1], seed_np[N_FRAMES - 1])

        # every free frame moved toward the targets
        for t in range(1, N_FRAMES - 1):
            test.assertGreater(np.abs(q_final[t] - seed_np[t]).max(), 1e-4, f"free frame {t} did not move")


# ----------------------------------------------------------------------------
# 5.  Velocity limit objective reduces peak joint velocity
# ----------------------------------------------------------------------------


def test_trajectory_velocity_limit(test, device):
    with wp.ScopedDevice(device):
        model = _build_two_link_planar(device)
        # arc with frame-to-frame jumps that require ~1.04 rad/s on joint 1,
        # slightly above the 1.0 rad/s limit
        targets_np = _arc_targets(N_FRAMES, angle_start=0.3, angle_end=1.44)

        limit_np = np.full(model.joint_dof_count, 1.0, dtype=np.float32)
        # the "unlimited" baseline uses the same objective with an unreachable
        # limit so both solves share one kernel specialization
        no_limit_np = np.full(model.joint_dof_count, 1.0e6, dtype=np.float32)

        def solve(limits_np):
            targets = wp.array(targets_np, dtype=wp.vec3)
            pos_obj = ik.IKObjectivePosition(
                link_index=EE_LINK,
                link_offset=EE_OFFSET,
                target_positions=targets,
            )
            vel_obj = ik.IKObjectiveVelocityLimit(
                model,
                velocity_limits=wp.array(limits_np, dtype=wp.float32),
                dt=DT,
                weight=5.0,
            )
            solver = ik.IKSolverTrajectory(
                model,
                N_FRAMES,
                [pos_obj, vel_obj],
                jacobian_mode=ik.IKJacobianType.ANALYTIC,
                linear_solver="direct",
                lambda_initial=1e-3,
            )
            joint_q = wp.zeros((N_FRAMES, model.joint_coord_count), dtype=wp.float32)
            # the active hinge slows down the transient from the zero seed, so
            # this test needs more LM iterations than the pure tracking tests
            solver.step(joint_q, joint_q, iterations=80)
            return joint_q

        q_free = solve(no_limit_np)
        q_limited = solve(limit_np)

        vmax_free = np.abs(np.diff(q_free.numpy(), axis=0)).max() / DT
        vmax_limited = np.abs(np.diff(q_limited.numpy(), axis=0)).max() / DT

        test.assertGreater(vmax_free, 1.0, "baseline should exceed the velocity limit")
        test.assertLess(vmax_limited, vmax_free, "velocity limit objective did not reduce peak velocity")

        final = _ee_positions(model, q_limited, EE_LINK, EE_OFFSET)
        tracking_err = np.linalg.norm(final - targets_np, axis=1).max()
        test.assertLess(tracking_err, 5e-2, f"tracking error too high with velocity limit ({tracking_err:.4f})")


# ----------------------------------------------------------------------------
# 6.  Batched trajectories match a single trajectory
# ----------------------------------------------------------------------------


def test_trajectory_batched_matches_single(test, device):
    with wp.ScopedDevice(device):
        model = _build_two_link_planar(device)
        targets_np = _arc_targets(N_FRAMES)

        def solve(n_problems):
            solver = ik.IKSolverTrajectory(
                model,
                N_FRAMES,
                _make_arc_tracking_objectives(model, np.tile(targets_np, (n_problems, 1))),
                n_problems=n_problems,
                jacobian_mode=ik.IKJacobianType.ANALYTIC,
                linear_solver="direct",
                lambda_initial=1e-3,
            )
            joint_q = wp.zeros((n_problems * N_FRAMES, model.joint_coord_count), dtype=wp.float32)
            solver.step(joint_q, joint_q, iterations=20)
            return joint_q.numpy().reshape(n_problems, N_FRAMES, -1)

        q_single = solve(1)
        q_batched = solve(3)

        for prob in range(3):
            assert_np_equal(q_batched[prob], q_single[0], tol=1e-6)


def test_trajectory_cg_batched_long_horizon(test, device):
    """Multi-trajectory CG solves above the segmented-dot length threshold must
    engage the tree reduction (warp's batched dot accumulates serially per lane
    and under-converges long chains), stay bitwise-equal to the equivalent
    single-trajectory solve, and be bitwise deterministic run to run. The
    per-trajectory length sits inside the reduction's parity envelope
    (<= 512 * 512 scalar dofs), where the segmented tree has the same shape as
    warp's single-batch bounded tree, so batched == single holds exactly; with
    warp's serial batched reduction instead, most coordinates drift by up to
    ~1e-6 here, so the bitwise assertion discriminates the reduction
    numerically. The full convergence failure needs a production-scale
    articulation; this guards the reduction path itself."""
    # 2 dofs per frame: just above the serial-dot length threshold, and a
    # multiple of the reduction tile so both copies' windows stay aligned
    n_frames = _CG_SERIAL_DOT_MAX_LENGTH // 2 + 1024
    with wp.ScopedDevice(device):
        model = _build_two_link_planar(device)
        targets_np = _arc_targets(n_frames)

        def solve(n_problems):
            solver = ik.IKSolverTrajectory(
                model,
                n_frames,
                _make_arc_tracking_objectives(model, np.tile(targets_np, (n_problems, 1))),
                n_problems=n_problems,
                jacobian_mode=ik.IKJacobianType.ANALYTIC,
                linear_solver="cg",
                lambda_initial=1e-3,
            )
            if n_problems > 1:
                test.assertIsInstance(solver._cg_state._tiled_dot, _SegmentedTiledDot)
            joint_q = wp.zeros((n_problems * n_frames, model.joint_coord_count), dtype=wp.float32)
            solver.step(joint_q, joint_q, iterations=10)
            return joint_q.numpy().reshape(n_problems, n_frames, -1)

        q_single = solve(1)
        q_batched = solve(2)
        q_batched_rerun = solve(2)

        # fixed reduction order: bitwise identical across runs
        assert_np_equal(q_batched_rerun, q_batched)
        # inside the parity envelope every copy matches the single-trajectory
        # solve bitwise (fails with warp's serial batched reduction)
        for prob in range(2):
            assert_np_equal(q_batched[prob], q_single[0])


def test_trajectory_segmented_dot(test, device):
    """_SegmentedTiledDot must match an fp64 reference within a tree-level
    error bound across batch shapes, reduce every segment independently of its
    offset (identical segment data gives bitwise-identical per-segment
    results, equal to a singleton instance's), be bitwise deterministic across
    repeated computes, and honor the two-column ``compute``/``col`` interface
    CG uses (``col_offset=1`` for the ``p . Ap`` column)."""
    rng = np.random.default_rng(1234)
    # above the engagement threshold, with a partially filled tail block
    length = _CG_SERIAL_DOT_MAX_LENGTH + 3 * _CG_DOT_TILE + 13
    with wp.ScopedDevice(device):
        for batch_count in (2, 3, 4):
            offsets_np = np.arange(batch_count + 1, dtype=np.int32) * length
            offsets = wp.array(offsets_np, dtype=wp.int32)

            # accuracy vs fp64: a . a keeps the reference away from zero so
            # the relative bound is meaningful (a . b with independent signs
            # cancels toward zero and has no scale-free relative error)
            a_np = rng.standard_normal(batch_count * length).astype(np.float32)
            a = wp.array(a_np, dtype=wp.float32)
            dot = _SegmentedTiledDot(offsets, length, device)
            out = dot.compute(a, a).numpy()[0]
            ref = np.array(
                [np.dot(seg.astype(np.float64), seg.astype(np.float64)) for seg in np.split(a_np, batch_count)]
            )
            rel_err = np.abs(out - ref) / ref
            # observed tree-level error is ~1e-7 for this size; 1e-6 leaves headroom
            test.assertLess(rel_err.max(), 1e-6)

            # determinism: repeated computes are bitwise identical
            assert_np_equal(dot.compute(a, a).numpy()[0], out)

        # offset independence: identical data in every segment reduces to
        # bitwise-identical results, equal to a singleton instance's result
        seg_np = rng.standard_normal(length).astype(np.float32)
        seg = wp.array(seg_np, dtype=wp.float32)
        tiled = wp.array(np.tile(seg_np, 3), dtype=wp.float32)
        offsets3 = wp.array(np.arange(4, dtype=np.int32) * length, dtype=wp.int32)
        offsets1 = wp.array(np.array([0, length], dtype=np.int32), dtype=wp.int32)
        out3 = _SegmentedTiledDot(offsets3, length, device).compute(tiled, tiled).numpy()[0]
        out1 = _SegmentedTiledDot(offsets1, length, device).compute(seg, seg).numpy()[0]
        assert_np_equal(out3, np.full(3, out1[0], dtype=np.float32))

        # two-column interface as CG drives it: a two-column compute fills
        # cols 0-1 with per-column dots, then a col_offset=1 compute
        # overwrites col 1 (the p . Ap slot) and leaves col 0 untouched
        seg64, b_np = seg_np.astype(np.float64), rng.standard_normal(length).astype(np.float32)
        b, b64 = wp.array(b_np, dtype=wp.float32), b_np.astype(np.float64)
        dot = _SegmentedTiledDot(offsets1, length, device)
        two_col = wp.array(np.stack([seg_np, b_np]), dtype=wp.float32)
        dot.compute(two_col, two_col)
        col0, col1 = dot.col(0).numpy()[0], dot.col(1).numpy()[0]
        test.assertLess(abs(col0 - np.dot(seg64, seg64)), 1e-6 * np.dot(seg64, seg64))
        test.assertLess(abs(col1 - np.dot(b64, b64)), 1e-6 * np.dot(b64, b64))
        dot.compute(seg, b, col_offset=1)
        test.assertEqual(dot.col(0).numpy()[0], col0)  # col 0 untouched
        # mixed dot cancels toward zero, so bound its error by the scale
        # sum(|seg_i * b_i|) instead of the reference value
        scale = np.dot(np.abs(seg64), np.abs(b64))
        test.assertLess(abs(dot.col(1).numpy()[0] - np.dot(seg64, b64)), 1e-6 * scale)


def test_trajectory_cg_dot_swap_guard(test, device):
    """The dot-reduction swap must fail loudly at solver construction if warp's
    CG state stops exposing a compatible ``_tiled_dot`` (a plain assignment
    would silently create a dead attribute and revert the fix while tests stay
    green)."""
    device = wp.get_device(device)
    offsets = wp.array(np.array([0, 4], dtype=np.int32), dtype=wp.int32, device=device)

    class _FakeState:
        pass

    state = _FakeState()
    # attribute renamed/removed by a warp internals change
    with test.assertRaises(RuntimeError):
        _swap_cg_tiled_dot(state, offsets, 4, device)

    # attribute present but with an unexpected tile size
    class _FakeDot:
        tile_size = 256

    state._tiled_dot = _FakeDot()
    with test.assertRaises(RuntimeError):
        _swap_cg_tiled_dot(state, offsets, 4, device)

    # compatible attribute: the swap installs the segmented reduction
    _FakeDot.tile_size = _CG_DOT_TILE
    _swap_cg_tiled_dot(state, offsets, 4, device)
    test.assertIsInstance(state._tiled_dot, _SegmentedTiledDot)


# ----------------------------------------------------------------------------
# 7.  Free-joint trajectory (quaternion tangent path)
# ----------------------------------------------------------------------------


def test_trajectory_free_joint(test, device):
    with wp.ScopedDevice(device):
        model = _build_free_plus_revolute(device)

        # EE path starting at the seed EE position (1.5, 0, 0) so the fixed
        # first frame is consistent with the tracking task
        xs = np.linspace(0.0, 0.8, N_FRAMES)
        targets_np = np.stack([1.5 + xs, 0.3 * xs, np.zeros(N_FRAMES)], axis=1).astype(np.float32)
        targets = wp.array(targets_np, dtype=wp.vec3)

        pos_obj = ik.IKObjectivePosition(
            link_index=EE_LINK,
            link_offset=EE_OFFSET,
            target_positions=targets,
        )
        smooth_vel = ik.IKObjectiveSmoothness(model, derivative=1, dt=DT, weight=0.02)
        smooth_acc = ik.IKObjectiveSmoothness(model, derivative=2, dt=DT, weight=0.002)

        solver = ik.IKSolverTrajectory(
            model,
            N_FRAMES,
            [pos_obj, smooth_vel, smooth_acc],
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
            linear_solver="direct",
            lambda_initial=1e-2,
            fixed_frames=[0],
        )

        # free joint coords: px py pz qx qy qz qw
        seed_np = np.zeros((N_FRAMES, model.joint_coord_count), dtype=np.float32)
        seed_np[:, 6] = 1.0
        joint_q = wp.array(seed_np, dtype=wp.float32)

        solver.step(joint_q, joint_q, iterations=25)
        q_final = joint_q.numpy()

        # fixed frame is bitwise-unchanged
        assert_np_equal(q_final[0], seed_np[0])

        # quaternions stay normalized
        quat_norm_err = np.abs(np.linalg.norm(q_final[:, 3:7], axis=1) - 1.0).max()
        test.assertLess(quat_norm_err, 1e-5, f"quaternion drift too high ({quat_norm_err:.2e})")

        errs = np.linalg.norm(_ee_positions(model, joint_q, EE_LINK, EE_OFFSET) - targets_np, axis=1)
        test.assertLess(errs[0], 1e-6, "fixed frame 0 should match its target exactly")
        test.assertLess(errs[1:].max(), 2e-2, f"free-joint tracking error too high ({errs[1:].max():.4f})")


def test_trajectory_free_joint_world_offset(test, device):
    """Temporal objectives must stay well-posed for floating bases away from
    the world origin: free-joint residuals are plain position differences
    (origin-invariant) and root free-joint tangents are body-centered, so
    the position rows carry no angular coupling and convergence does not
    depend on where the trajectory sits in the world (cf.
    test_ik_trajectory_dynamics.test_free_joint_far_from_origin_converges
    for the far-origin regression)."""
    with wp.ScopedDevice(device):
        model = _build_free_plus_revolute(device)
        offset = np.array([10.0, -6.0, 0.0], dtype=np.float32)

        xs = np.linspace(0.0, 0.8, N_FRAMES)
        targets_np = np.stack([1.5 + xs, 0.3 * xs, np.zeros(N_FRAMES)], axis=1).astype(np.float32)
        targets_np += offset
        targets = wp.array(targets_np, dtype=wp.vec3)

        pos_obj = ik.IKObjectivePosition(
            link_index=EE_LINK,
            link_offset=EE_OFFSET,
            target_positions=targets,
        )
        smooth_vel = ik.IKObjectiveSmoothness(model, derivative=1, dt=DT, weight=0.02)
        smooth_acc = ik.IKObjectiveSmoothness(model, derivative=2, dt=DT, weight=0.002)

        solver = ik.IKSolverTrajectory(
            model,
            N_FRAMES,
            [pos_obj, smooth_vel, smooth_acc],
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
            linear_solver="direct",
            lambda_initial=1e-2,
            fixed_frames=[0],
        )

        seed_np = np.zeros((N_FRAMES, model.joint_coord_count), dtype=np.float32)
        seed_np[:, :3] = offset
        seed_np[:, 6] = 1.0
        joint_q = wp.array(seed_np, dtype=wp.float32)

        solver.step(joint_q, joint_q, iterations=60)

        errs = np.linalg.norm(_ee_positions(model, joint_q, EE_LINK, EE_OFFSET) - targets_np, axis=1)
        test.assertLess(errs[1:].max(), 2.5e-2, f"tracking error at world offset too high ({errs[1:].max():.4f})")


def test_trajectory_costs_fresh_after_step(test, device):
    """``trajectory_costs`` must reflect the returned trajectory, not the
    state at the start of the last LM iteration."""
    with wp.ScopedDevice(device):
        model = _build_two_link_planar(device)
        targets_np = _arc_targets(N_FRAMES)
        pos_obj = ik.IKObjectivePosition(
            link_index=EE_LINK,
            link_offset=EE_OFFSET,
            target_positions=wp.array(targets_np, dtype=wp.vec3),
        )
        smooth_obj = ik.IKObjectiveSmoothness(model, derivative=1, dt=DT, weight=0.02)
        solver = ik.IKSolverTrajectory(
            model,
            N_FRAMES,
            [pos_obj, smooth_obj],
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
            linear_solver="direct",
            lambda_initial=1e-3,
        )
        joint_q = wp.zeros((N_FRAMES, model.joint_coord_count), dtype=wp.float32)
        solver.step(joint_q, joint_q, iterations=1)
        reported = solver.trajectory_costs.numpy().copy()
        recomputed = solver.compute_trajectory_costs(joint_q).numpy()
        assert_np_equal(reported, recomputed, tol=1e-5)


# ----------------------------------------------------------------------------
# 8.  Temporal objectives require the trajectory solver
# ----------------------------------------------------------------------------


def test_temporal_objective_requires_trajectory_solver(test, device):
    with wp.ScopedDevice(device):
        model = _build_two_link_planar(device)
        smooth_obj = ik.IKObjectiveSmoothness(model, derivative=1, dt=DT, weight=0.02)

        with test.assertRaises(RuntimeError) as ctx:
            solver = ik.IKSolver(model, 1, [smooth_obj])
            joint_q = wp.zeros((1, model.joint_coord_count), dtype=wp.float32, requires_grad=True)
            solver.step(joint_q, joint_q, iterations=1)
        test.assertIn("IKSolverTrajectory", str(ctx.exception))


# ----------------------------------------------------------------------------
# 9.  CUDA graph capture
# ----------------------------------------------------------------------------


def test_trajectory_graph_capture(test, device):
    with wp.ScopedDevice(device):
        model = _build_two_link_planar(device)
        targets_np = _arc_targets(N_FRAMES)

        solver = ik.IKSolverTrajectory(
            model,
            N_FRAMES,
            _make_arc_tracking_objectives(model, targets_np),
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
            linear_solver="direct",
            lambda_initial=1e-3,
        )

        joint_q = wp.zeros((N_FRAMES, model.joint_coord_count), dtype=wp.float32)
        # warm up so all modules are loaded before capture
        solver.step(joint_q, joint_q, iterations=20)

        joint_q.zero_()
        with wp.ScopedCapture() as capture:
            solver.step(joint_q, joint_q, iterations=20)
        wp.capture_launch(capture.graph)

        costs = solver.compute_trajectory_costs(joint_q).numpy()
        test.assertTrue(np.all(np.isfinite(costs)), "trajectory costs not finite after graph replay")
        test.assertLess(costs.max(), 1e-2, f"trajectory cost too high after graph replay ({costs.max():.4f})")

        final = _ee_positions(model, joint_q, EE_LINK, EE_OFFSET)
        for t in range(N_FRAMES):
            err = np.linalg.norm(final[t] - targets_np[t])
            test.assertLess(err, 5e-3, f"frame {t} error too high after graph replay ({err:.5f})")


# ----------------------------------------------------------------------------
# 10.  Test-class registration per device
# ----------------------------------------------------------------------------

devices = get_test_devices()
cuda_devices = get_selected_cuda_test_devices()


class TestIKTrajectory(unittest.TestCase):
    pass


add_function_test(
    TestIKTrajectory, "test_trajectory_convergence_analytic", test_trajectory_convergence_analytic, devices
)
add_function_test(
    TestIKTrajectory, "test_trajectory_convergence_autodiff", test_trajectory_convergence_autodiff, devices
)
add_function_test(
    TestIKTrajectory,
    "test_trajectory_linear_reference_matches_dense",
    test_trajectory_linear_reference_matches_dense,
    devices,
)
add_function_test(TestIKTrajectory, "test_trajectory_cg_matches_direct", test_trajectory_cg_matches_direct, devices)
add_function_test(
    TestIKTrajectory, "test_trajectory_spike_matches_direct", test_trajectory_spike_matches_direct, devices
)
add_function_test(TestIKTrajectory, "test_trajectory_fixed_frames", test_trajectory_fixed_frames, devices)
add_function_test(TestIKTrajectory, "test_trajectory_velocity_limit", test_trajectory_velocity_limit, devices)
add_function_test(
    TestIKTrajectory, "test_trajectory_batched_matches_single", test_trajectory_batched_matches_single, devices
)
add_function_test(
    TestIKTrajectory,
    "test_trajectory_cg_batched_long_horizon",
    test_trajectory_cg_batched_long_horizon,
    cuda_devices,
)
add_function_test(TestIKTrajectory, "test_trajectory_segmented_dot", test_trajectory_segmented_dot, cuda_devices)
add_function_test(
    TestIKTrajectory, "test_trajectory_cg_dot_swap_guard", test_trajectory_cg_dot_swap_guard, cuda_devices
)
add_function_test(TestIKTrajectory, "test_trajectory_free_joint", test_trajectory_free_joint, devices)
add_function_test(
    TestIKTrajectory,
    "test_trajectory_free_joint_world_offset",
    test_trajectory_free_joint_world_offset,
    devices,
)
add_function_test(
    TestIKTrajectory,
    "test_trajectory_costs_fresh_after_step",
    test_trajectory_costs_fresh_after_step,
    devices,
)
add_function_test(
    TestIKTrajectory,
    "test_temporal_objective_requires_trajectory_solver",
    test_temporal_objective_requires_trajectory_solver,
    devices,
)
add_function_test(TestIKTrajectory, "test_trajectory_graph_capture", test_trajectory_graph_capture, cuda_devices)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
