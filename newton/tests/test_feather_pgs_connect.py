# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for connect (loop-closure) constraint rows in SolverFeatherPGS."""

import unittest

import numpy as np
import warp as wp

import newton


def _build_four_bar():
    """Build a planar four-bar: two grounded revolute chains closed by a BALL loop joint.

    Ground pivots at x=0 and x=0.4; crank and rocker hang down 0.2 m; the coupler
    connects the crank tip to the rocker tip through the loop closure. The crank is
    position-driven; the rocker is undriven, so any coherent motion it does comes from
    the closed loop.
    """
    b = newton.ModelBuilder(up_axis=newton.Axis.Z)
    z0 = 0.6

    crank = b.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, z0 - 0.1), wp.quat_identity()))
    b.add_shape_box(crank, hx=0.02, hy=0.02, hz=0.1)
    j_crank = b.add_joint_revolute(
        parent=-1,
        child=crank,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, z0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()),
    )
    coupler = b.add_link(xform=wp.transform(wp.vec3(0.2, 0.0, z0 - 0.2), wp.quat_identity()))
    b.add_shape_box(coupler, hx=0.2, hy=0.02, hz=0.02)
    j_coupler = b.add_joint_revolute(
        parent=crank,
        child=coupler,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, -0.1), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
    )
    rocker = b.add_link(xform=wp.transform(wp.vec3(0.4, 0.0, z0 - 0.1), wp.quat_identity()))
    b.add_shape_box(rocker, hx=0.02, hy=0.02, hz=0.1)
    j_rocker = b.add_joint_revolute(
        parent=-1,
        child=rocker,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.4, 0.0, z0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()),
    )
    b.add_articulation([j_crank, j_coupler, j_rocker], label="four_bar")

    # Loop closure: coupler tip pinned to rocker tip (a trailing BALL loop joint,
    # matching how the MJCF importer closes `connect` equalities).
    b.add_joint_ball(
        parent=coupler,
        child=rocker,
        parent_xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, -0.1), wp.quat_identity()),
    )

    # Drive the crank only.
    b.joint_target_ke[0] = 50.0
    b.joint_target_kd[0] = 5.0
    b.joint_target_mode[0] = int(newton.JointTargetMode.POSITION)
    return b


def _loop_anchor_gap(model, state) -> float:
    """World-space distance between the loop joint's parent and child anchors [m]."""
    bq = state.body_q.numpy().reshape(-1, 7).astype(np.float64)
    jt = model.joint_type.numpy()
    jp = model.joint_parent.numpy()
    jc = model.joint_child.numpy()
    Xp = model.joint_X_p.numpy()
    Xc = model.joint_X_c.numpy()
    j = int(np.nonzero(jt == int(newton.JointType.BALL))[0][-1])

    def anchor(body, X):
        t = wp.transform(wp.vec3(*bq[body, :3]), wp.quat(*bq[body, 3:]))
        a = wp.transform(wp.vec3(*X[:3]), wp.quat(*X[3:]))
        w = wp.transform_multiply(t, a)
        return np.array([w.p[0], w.p[1], w.p[2]])

    return float(np.linalg.norm(anchor(int(jp[j]), Xp[j]) - anchor(int(jc[j]), Xc[j])))


def _run(steps: int = 720, crank_target: float = 0.6, **solver_kwargs):
    builder = _build_four_bar()
    model = builder.finalize()
    solver = newton.solvers.SolverFeatherPGS(
        model, pgs_mode="matrix_free", pgs_iterations=16, pgs_beta=0.1, **solver_kwargs
    )
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    targets = model.joint_target_q.numpy().copy()
    targets[0] = crank_target
    control.joint_target_q.assign(targets)
    dt = 1.0 / 240.0
    for _ in range(steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0
    return solver, model, state_0


@unittest.skipUnless(wp.get_device().is_cuda, "SolverFeatherPGS matrix-free mode requires CUDA")
class TestFeatherPGSConnect(unittest.TestCase):
    def test_loop_joint_presence_is_stable(self):
        """Verify a trailing BALL loop joint no longer NaNs the articulation solve."""
        _, _, state = _run(steps=120, crank_target=0.0)
        self.assertTrue(np.isfinite(state.body_q.numpy()).all())
        self.assertTrue(np.isfinite(state.joint_qd.numpy()).all())

    def test_loop_closure_is_enforced(self):
        """Verify the connect rows hold the four-bar's loop anchors together under drive."""
        _, model, state = _run(steps=720, crank_target=0.6)
        self.assertTrue(np.isfinite(state.body_q.numpy()).all())
        gap = _loop_anchor_gap(model, state)
        self.assertLess(gap, 2.0e-3, f"loop anchor gap {gap * 1e3:.2f} mm — four-bar not closed")

    def test_connect_toggle_drops_closure(self):
        """Verify enable_connect_constraints=False stays stable but leaves the loop open.

        The loop joint must still be excluded from the dynamics tree (finite state), while
        the missing CONNECT rows let the anchors separate far beyond the enforced tolerance
        — the pre-connect solver behavior, kept reachable for A/B comparison.
        """
        with self.assertWarns(UserWarning):
            _, model, state = _run(steps=720, crank_target=0.6, enable_connect_constraints=False)
        self.assertTrue(np.isfinite(state.body_q.numpy()).all())
        gap = _loop_anchor_gap(model, state)
        self.assertGreater(gap, 1.0e-2, f"loop anchor gap {gap * 1e3:.2f} mm — closure still enforced?")

    def test_rocker_follows_crank(self):
        """Verify the undriven rocker moves coherently with the driven crank via the loop."""
        _, _model, state = _run(steps=720, crank_target=0.6)
        q = state.joint_q.numpy()
        # Parallel four-bar (equal crank/rocker lengths): rocker angle tracks crank angle.
        self.assertAlmostEqual(q[2], q[0], delta=0.08)
        self.assertGreater(abs(q[2]), 0.3, "rocker did not move — loop not transmitting")

    def test_connect_survives_foreign_joint_between_tree_and_loop(self):
        """Attribute loop joints by body ownership when another articulation interleaves.

        ``add_usd`` of a subtree containing a closed-loop robot AND a floating rigid
        body creates the body's FREE joint during traversal but appends the equality
        loop joints at the very end of the parse — so the closures no longer trail
        their articulation's joint-index range (the trailing sentinel range of the LAST
        articulation swallows them instead). Index-range attribution then silently
        produced ZERO connect rows and the closed linkage fell open (measured on the
        Robotiq 2F-85 through the IsaacLab scene path: 107 mm anchor gap, fingers
        collapsed). The closure must be attributed via its child body's articulation.
        This test reconstructs that joint ordering directly: four-bar tree, then a
        free-jointed object articulation, THEN the closure ball joint.
        """
        b = newton.ModelBuilder(up_axis=newton.Axis.Z)
        z0 = 0.6
        crank = b.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, z0 - 0.1), wp.quat_identity()))
        b.add_shape_box(crank, hx=0.02, hy=0.02, hz=0.1)
        j_crank = b.add_joint_revolute(
            parent=-1,
            child=crank,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, z0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()),
        )
        coupler = b.add_link(xform=wp.transform(wp.vec3(0.2, 0.0, z0 - 0.2), wp.quat_identity()))
        b.add_shape_box(coupler, hx=0.2, hy=0.02, hz=0.02)
        j_coupler = b.add_joint_revolute(
            parent=crank,
            child=coupler,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, -0.1), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
        )
        rocker = b.add_link(xform=wp.transform(wp.vec3(0.4, 0.0, z0 - 0.1), wp.quat_identity()))
        b.add_shape_box(rocker, hx=0.02, hy=0.02, hz=0.1)
        j_rocker = b.add_joint_revolute(
            parent=-1,
            child=rocker,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(wp.vec3(0.4, 0.0, z0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()),
        )
        b.add_articulation([j_crank, j_coupler, j_rocker], label="four_bar")

        # a second articulation's joint lands BETWEEN the tree and the loop joint
        body = b.add_body(xform=wp.transform(wp.vec3(1.0, 0.0, 0.5), wp.quat_identity()))
        b.add_shape_sphere(body, radius=0.05)
        j_free = b.add_joint_free(child=body)
        b.add_articulation([j_free], label="free_object")

        # the loop closure, appended last (the add_usd equality-parse ordering)
        b.add_joint_ball(
            parent=coupler,
            child=rocker,
            parent_xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, -0.1), wp.quat_identity()),
        )
        b.joint_target_ke[0] = 50.0
        b.joint_target_kd[0] = 5.0
        b.joint_target_mode[0] = int(newton.JointTargetMode.POSITION)
        model = b.finalize()

        solver = newton.solvers.SolverFeatherPGS(model, pgs_mode="matrix_free", pgs_iterations=16, pgs_beta=0.1)
        self.assertEqual(
            getattr(solver, "_connect_count", 0),
            1,
            "closure row lost — loop joint attributed to the wrong articulation",
        )

        state_0, state_1 = model.state(), model.state()
        control = model.control()
        targets = model.joint_target_q.numpy().copy()
        targets[0] = 0.6
        control.joint_target_q.assign(targets)
        gap_max = 0.0
        for _ in range(360):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, None, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0
            gap_max = max(gap_max, _loop_anchor_gap(model, state_0))
        self.assertTrue(np.isfinite(state_0.body_q.numpy()).all())
        self.assertLess(gap_max, 2.0e-3, f"loop closure not enforced (anchor gap {gap_max:.4f} m)")


if __name__ == "__main__":
    unittest.main()
