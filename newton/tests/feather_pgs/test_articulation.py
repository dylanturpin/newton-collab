# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Loop closures, mimic joints, pre-elimination, springs, free roots, gravity, parent wrenches and resets of SolverFeatherPGS."""

import functools
import inspect
import itertools
import unittest
import warnings
from itertools import pairwise
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs import kernels as feather_pgs_kernels
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONNECT,
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_MIMIC,
    accumulate_connect_wrenches_from_impulses,
    integrate_generalized_joints,
    populate_connect_J_for_size,
)
from newton._src.solvers.featherstone.kernels import eval_fk_with_velocity_conversion
from newton.solvers import SolverFeatherPGS, SolverFeatherstone, SolverMuJoCo, SolverObservableFlags
from newton.tests.feather_pgs._fixtures import _build_four_bar, _loop_anchor_gap, _sphere_on_ground, set_torsion_radius
from newton.tests.unittest_utils import USD_AVAILABLE, add_function_test, get_cuda_test_devices, get_test_devices
from newton.usd import SchemaResolverNewton, SchemaResolverPhysx


def _run(steps: int = 720, crank_target: float = 0.6, **solver_kwargs):
    builder = _build_four_bar()
    model = builder.finalize()
    solver = newton.solvers.SolverFeatherPGS(
        model, solve_mode="matrix_free", iterations=16, baumgarte=0.1, **solver_kwargs
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

    def test_rocker_follows_crank(self):
        """Verify the undriven rocker moves coherently with the driven crank via the loop."""
        _, _model, state = _run(steps=720, crank_target=0.6)
        q = state.joint_q.numpy()
        # Parallel four-bar (equal crank/rocker lengths): rocker angle tracks crank angle.
        self.assertAlmostEqual(q[2], q[0], delta=0.08)
        self.assertGreater(abs(q[2]), 0.3, "rocker did not move — loop not transmitting")

    def test_connect_ownership_survives_interleaved_articulation(self):
        """Keep a deferred closure with its child body's articulation."""
        b = newton.ModelBuilder(up_axis=newton.Axis.Z)

        left = b.add_link()
        b.add_shape_box(left, hx=0.05, hy=0.05, hz=0.1)
        j_left = b.add_joint_revolute(parent=-1, child=left, axis=newton.Axis.Y)
        right = b.add_link()
        b.add_shape_box(right, hx=0.05, hy=0.05, hz=0.1)
        j_right = b.add_joint_revolute(parent=-1, child=right, axis=newton.Axis.Y)
        b.add_articulation([j_left, j_right], label="closed_linkage")

        foreign = b.add_link()
        b.add_shape_sphere(foreign, radius=0.05)
        j_foreign = b.add_joint_free(child=foreign)
        b.add_articulation([j_foreign], label="foreign")

        loop_joint = b.add_joint_ball(parent=left, child=right)
        model = b.finalize()
        solver = newton.solvers.SolverFeatherPGS(model, solve_mode="matrix_free")

        self.assertEqual(int(model.joint_articulation.numpy()[loop_joint]), -1)
        np.testing.assert_array_equal(
            solver._model_plan.loop_joint_articulation,
            np.array([-1, -1, -1, 0], dtype=np.int32),
        )
        np.testing.assert_array_equal(solver._connect_art.numpy(), np.array([0], dtype=np.int32))

    def test_standalone_world_root_is_not_loop_joint(self):
        """Do not treat an unrelated unowned world-root joint as a closure."""
        b = newton.ModelBuilder(up_axis=newton.Axis.Z)

        articulated = b.add_link()
        b.add_shape_box(articulated, hx=0.05, hy=0.05, hz=0.1)
        tree_joint = b.add_joint_revolute(parent=-1, child=articulated, axis=newton.Axis.Y)
        b.add_articulation([tree_joint], label="articulated")

        standalone = b.add_link()
        b.add_shape_box(standalone, hx=0.05, hy=0.05, hz=0.1)
        standalone_joint = b.add_joint_fixed(parent=-1, child=standalone)

        model = b.finalize()
        self.assertEqual(model.joint_articulation.numpy().tolist(), [0, -1])
        self.assertEqual(standalone_joint, 1)

        solver = newton.solvers.SolverFeatherPGS(
            model,
            solve_mode="matrix_free",
            articulated_contact_response="propagation",
        )
        self.assertFalse(solver._has_loop_joints)
        self.assertEqual(solver._connect_count, 0)
        np.testing.assert_array_equal(
            solver._model_plan.loop_joint_articulation,
            np.array([-1, -1], dtype=np.int32),
        )

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
        body = b.add_link(xform=wp.transform(wp.vec3(1.0, 0.0, 0.5), wp.quat_identity()))
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

        solver = newton.solvers.SolverFeatherPGS(model, solve_mode="matrix_free", iterations=16, baumgarte=0.1)
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


_CHILD_ANCHORS = ((0.0, 0.0, 0.0), (0.05, 0.0, 0.0), (0.0, 0.05, 0.0))


_REL_P = np.array([0.0, 0.0, -0.2])


def _pgs_mode():
    return "matrix_free" if wp.get_device().is_cuda else "split"


def _build_carried_load(*, enabled: bool = True, world_parent: bool = False):
    """A kinematic carrier holding a dynamic box through three BALL loop joints.

    Three non-collinear point closures pin all six relative degrees of freedom, so the
    load must follow the carrier as if welded at ``_REL_P`` below it. With
    ``world_parent`` the carrier is the world and the anchors are world points.
    """
    b = newton.ModelBuilder(up_axis=newton.Axis.Z)
    carrier_p = np.array([0.0, 0.0, 1.0])
    carrier = -1
    if not world_parent:
        carrier = b.add_body(xform=wp.transform(wp.vec3(*carrier_p), wp.quat_identity()), is_kinematic=True)
        b.add_shape_box(carrier, hx=0.03, hy=0.03, hz=0.03)
    load = b.add_body(xform=wp.transform(wp.vec3(*(carrier_p + _REL_P)), wp.quat_identity()))
    b.add_shape_box(load, hx=0.04, hy=0.04, hz=0.04, cfg=newton.ModelBuilder.ShapeConfig(density=500.0))
    joints = []
    # The closures intentionally parallel the load's free joint and each other.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=r".*another joint already connects these bodies", category=UserWarning
        )
        for c in _CHILD_ANCHORS:
            p = (_REL_P if not world_parent else carrier_p + _REL_P) + np.array(c)
            joints.append(
                b.add_joint_ball(
                    parent=carrier,
                    child=load,
                    parent_xform=wp.transform(wp.vec3(*p), wp.quat_identity()),
                    child_xform=wp.transform(wp.vec3(*c), wp.quat_identity()),
                    enabled=enabled,
                )
            )
    return b, carrier, load, joints


def _anchor_gaps(model, state, joints):
    """World-space parent/child anchor distances [m] of the model's loop joint frames."""
    bq = state.body_q.numpy().astype(np.float64)
    jp = model.joint_parent.numpy()
    jc = model.joint_child.numpy()
    xp = model.joint_X_p.numpy()
    xc = model.joint_X_c.numpy()

    def anchor(body, local):
        if body < 0:
            return np.asarray(local, dtype=np.float64)
        t = wp.transform(wp.vec3(*bq[body, :3]), wp.quat(*bq[body, 3:]))
        w = wp.transform_point(t, wp.vec3(*local))
        return np.array([w[0], w[1], w[2]])

    return [float(np.linalg.norm(anchor(int(jp[j]), xp[j][:3]) - anchor(int(jc[j]), xc[j][:3]))) for j in joints]


def _set_joint_enabled(model, joints, enabled):
    """Write ``Model.joint_enabled`` in place, without a notification."""
    flags = model.joint_enabled.numpy()
    flags[np.asarray(joints, dtype=np.int64)] = enabled
    model.joint_enabled.assign(flags)


def _prescribe_carrier(model, solver, state, articulation, t, v, omega):
    """Write the carrier's free-joint pose/velocity for time ``t`` and refresh its FK."""
    q = state.joint_q.numpy()
    qd = state.joint_qd.numpy()
    angle = float(np.linalg.norm(omega)) * t
    axis = np.asarray(omega, dtype=np.float64) / max(float(np.linalg.norm(omega)), 1.0e-12)
    rot = wp.quat_from_axis_angle(wp.vec3(*axis), angle)
    q[0:3] = np.array([0.0, 0.0, 1.0]) + np.asarray(v) * t
    q[3:7] = [rot[0], rot[1], rot[2], rot[3]]
    qd[0:3] = v
    qd[3:6] = omega
    state.joint_q.assign(q)
    state.joint_qd.assign(qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state, indices=articulation)
    solver.notify_state_changed(articulation)


class TestFeatherPGSPrescribedParentConnect(unittest.TestCase):
    def test_kinematic_parent_closure_carries_load(self):
        """A load pinned to a moving kinematic carrier must ride along with it.

        Without prescribed-parent support the closure is ignored (it crosses
        articulations) and the load free-falls; with it, the three point closures hold
        the load at its relative pose while the carrier translates and spins.
        """
        b, carrier, load, joints = _build_carried_load()
        model = b.finalize()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            solver = newton.solvers.SolverFeatherPGS(model, solve_mode=_pgs_mode(), iterations=16, baumgarte=0.2)
        self.assertFalse([w for w in caught if "loop joint" in str(w.message)], [str(w.message) for w in caught])
        articulation = wp.array([int(solver._body_to_articulation.numpy()[carrier])], dtype=wp.int32)
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        dt = 1.0 / 240.0
        v = np.array([0.3, 0.0, 0.1])
        omega = np.array([0.0, 0.0, 1.5])
        for i in range(240):
            _prescribe_carrier(model, solver, state_0, articulation, i * dt, v, omega)
            state_0.clear_forces()
            solver.step(state_0, state_1, control, None, dt)
            state_0, state_1 = state_1, state_0
        _prescribe_carrier(model, solver, state_0, articulation, 240 * dt, v, omega)

        bq = state_0.body_q.numpy().astype(np.float64)
        self.assertTrue(np.isfinite(bq).all())
        carrier_t = wp.transform(wp.vec3(*bq[carrier, :3]), wp.quat(*bq[carrier, 3:]))
        expected = wp.transform_point(carrier_t, wp.vec3(*_REL_P))
        error = np.linalg.norm(bq[load, :3] - np.array([expected[0], expected[1], expected[2]]))
        self.assertLess(error, 3.0e-3, f"load lagged the carrier by {error:.4f} m")
        for gap in _anchor_gaps(model, state_0, joints):
            self.assertLess(gap, 3.0e-3)
        # The carrier actually moved, so the test exercised a non-trivial target velocity.
        self.assertGreater(np.linalg.norm(bq[carrier, :3] - np.array([0.0, 0.0, 1.0])), 0.25)

    def test_world_parent_closure_holds_hanging_load(self):
        b, _, load, joints = _build_carried_load(world_parent=True)
        model = b.finalize()
        solver = newton.solvers.SolverFeatherPGS(model, solve_mode=_pgs_mode(), iterations=16, baumgarte=0.2)
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        for _ in range(240):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, None, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0
        bq = state_0.body_q.numpy().astype(np.float64)
        self.assertTrue(np.isfinite(bq).all())
        self.assertLess(np.linalg.norm(bq[load, :3] - (np.array([0.0, 0.0, 1.0]) + _REL_P)), 3.0e-3)
        for gap in _anchor_gaps(model, state_0, joints):
            self.assertLess(gap, 3.0e-3)

    def test_model_edits_release_and_reanchor_closures(self):
        """Release closures through Model.joint_enabled, re-anchor them through joint_X_p and engage them again."""
        b, carrier, load, joints = _build_carried_load()
        model = b.finalize()
        solver = newton.solvers.SolverFeatherPGS(model, solve_mode=_pgs_mode(), iterations=16, baumgarte=0.2)
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        dt = 1.0 / 240.0

        def run(steps):
            nonlocal state_0, state_1
            for _ in range(steps):
                state_0.clear_forces()
                solver.step(state_0, state_1, control, None, dt)
                state_0, state_1 = state_1, state_0

        _set_joint_enabled(model, joints, False)
        z0 = float(state_0.body_q.numpy()[load, 2])
        run(48)
        bq = state_0.body_q.numpy().astype(np.float64)
        dropped = z0 - float(bq[load, 2])
        self.assertGreater(dropped, 0.15, "released load should free-fall")

        # Re-target the closures at the measured relative pose, then re-engage them.
        carrier_t = wp.transform(wp.vec3(*bq[carrier, :3]), wp.quat(*bq[carrier, 3:]))
        load_t = wp.transform(wp.vec3(*bq[load, :3]), wp.quat(*bq[load, 3:]))
        rel = wp.transform_multiply(wp.transform_inverse(carrier_t), load_t)
        joint_X_p = model.joint_X_p.numpy()
        for j, c in zip(joints, _CHILD_ANCHORS, strict=True):
            p = wp.transform_point(rel, wp.vec3(*c))
            joint_X_p[j][:3] = (p[0], p[1], p[2])
        model.joint_X_p.assign(joint_X_p)
        _set_joint_enabled(model, joints, True)
        solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
        held_z = float(bq[load, 2])
        run(120)
        bq = state_0.body_q.numpy().astype(np.float64)
        self.assertTrue(np.isfinite(bq).all())
        self.assertLess(abs(float(bq[load, 2]) - held_z), 3.0e-3, "re-engaged closure should hold the load")
        for gap in _anchor_gaps(model, state_0, joints):
            self.assertLess(gap, 3.0e-3)

        _set_joint_enabled(model, joints, False)
        run(48)
        self.assertGreater(held_z - float(state_0.body_q.numpy()[load, 2]), 0.1, "released again: must fall")


def _build_standalone_world_root():
    b = newton.ModelBuilder(up_axis=newton.Axis.Z)
    articulated = b.add_link()
    b.add_shape_box(articulated, hx=0.05, hy=0.05, hz=0.1)
    tree_joint = b.add_joint_revolute(parent=-1, child=articulated, axis=newton.Axis.Y)
    b.add_articulation([tree_joint], label="articulated")
    standalone = b.add_link()
    b.add_shape_box(standalone, hx=0.05, hy=0.05, hz=0.1)
    standalone_joint = b.add_joint_fixed(parent=-1, child=standalone)
    return b, standalone_joint


def test_four_bar_capture_matches_eager(test, device):
    """Replay the eager four-bar trajectory from a captured step."""
    model = _build_four_bar().finalize(device=device)
    control = model.control()
    targets = model.joint_target_q.numpy().copy()
    targets[0] = 0.6
    control.joint_target_q.assign(targets)
    eager = SolverFeatherPGS(model, solve_mode="matrix_free", iterations=16, baumgarte=0.1)
    captured = SolverFeatherPGS(model, solve_mode="matrix_free", iterations=16, baumgarte=0.1)
    e0, e1 = model.state(), model.state()
    c0, c1 = model.state(), model.state()

    def step_pair(solver, s0, s1):
        solver.step(s0, s1, control, None, 1.0 / 240.0)
        solver.step(s1, s0, control, None, 1.0 / 240.0)

    step_pair(captured, c0, c1)
    step_pair(eager, e0, e1)
    # Capturing records the launches without running them.
    with wp.ScopedCapture(device=device) as capture:
        step_pair(captured, c0, c1)
    for _ in range(60):
        wp.capture_launch(capture.graph)
        step_pair(eager, e0, e1)
    np.testing.assert_allclose(c0.joint_q.numpy(), e0.joint_q.numpy(), atol=2.0e-5)
    test.assertLess(_loop_anchor_gap(model, c0), 2.0e-3)


def check_standalone_world_root_is_not_loop_joint(test, device, **solver_kwargs):
    """An unrelated unowned world-root joint is not a closure."""
    b, standalone_joint = _build_standalone_world_root()
    model = b.finalize(device=device)
    test.assertEqual(model.joint_articulation.numpy().tolist(), [0, -1])
    test.assertEqual(standalone_joint, 1)
    solver = SolverFeatherPGS(model, solve_mode="matrix_free", **solver_kwargs)
    test.assertEqual(solver._connect_count, 0)
    np.testing.assert_array_equal(solver._model_plan.loop_joint_articulation, np.array([-1, -1], dtype=np.int32))


def test_propagation_standalone_world_root_is_not_loop_joint(test, device):
    """Do not treat an unowned world-root joint as a closure with propagation responses."""
    check_standalone_world_root_is_not_loop_joint(test, device, articulated_contact_response="propagation")


def test_disabled_loop_joint_starts_released(test, device):
    """Start a loop joint disabled in Model.joint_enabled released, and engage it later."""
    b, _, load, joints = _build_carried_load(enabled=False, world_parent=True)
    model = b.finalize(device=device)
    solver = SolverFeatherPGS(model, solve_mode="matrix_free", iterations=16, baumgarte=0.2)
    state_0, state_1 = model.state(), model.state()
    z0 = float(state_0.body_q.numpy()[load, 2])
    for _ in range(48):
        solver.step(state_0, state_1, model.control(), None, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0
    test.assertGreater(z0 - float(state_0.body_q.numpy()[load, 2]), 0.15)
    test.assertEqual(solver._connect_slot.numpy().tolist(), [-1, -1, -1])

    flags = model.joint_enabled.numpy()
    flags[np.asarray(joints, dtype=np.int64)] = True
    model.joint_enabled.assign(flags)
    solver.step(state_0, state_1, model.control(), None, 1.0 / 240.0)
    test.assertTrue(np.all(solver._connect_slot.numpy() >= 0))


def test_imported_connect_equality_is_enforced(test, device):
    """Enforce an imported MuJoCo CONNECT through its converted loop joint; reject it unconverted."""
    mjcf = """
    <mujoco>
      <worldbody>
        <body name="link" pos="0 0 1">
          <joint name="hinge" type="hinge" axis="0 1 0"/>
          <geom type="box" pos="0.3 0 0" size="0.3 0.05 0.05"/>
        </body>
      </worldbody>
      <equality><connect body1="link" anchor="0.6 0 0"/></equality>
    </mujoco>
    """
    for convert in (True, False):
        b = newton.ModelBuilder()
        if convert:
            # The converted CONNECT becomes a BALL loop joint parallel to the hinge.
            _expect_one_warning(
                test,
                UserWarning,
                "another joint already connects these bodies",
                functools.partial(b.add_mjcf, mjcf, convert_mjc_equality_constraints=True),
            )
        else:
            b.add_mjcf(mjcf, convert_mjc_equality_constraints=False)
        model = b.finalize(device=device)
        test.assertEqual(model.mujoco.equality_constraint_count, 1)
        if not convert:
            with test.assertRaisesRegex(NotImplementedError, "equality"):
                SolverFeatherPGS(model, solve_mode="matrix_free")
            continue
        solver = SolverFeatherPGS(model, solve_mode="matrix_free", iterations=16)
        test.assertEqual(solver._connect_count, 1)
        state_0, state_1 = model.state(), model.state()
        for _ in range(120):
            solver.step(state_0, state_1, model.control(), None, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0
        # The closure at the free end holds the hinge against gravity.
        test.assertLess(abs(float(state_0.joint_q.numpy()[0])), 1.0e-2)


def _expect_one_warning(test, category, pattern, call):
    """Return ``call()``, requiring it to emit exactly one warning, of ``category`` and matching ``pattern``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = call()
    test.assertEqual(len(caught), 1, [f"{w.category.__name__}: {w.message}" for w in caught])
    test.assertIs(caught[0].category, category)
    test.assertRegex(str(caught[0].message), pattern)
    return result


class TestFeatherPGSConnectUpstream(unittest.TestCase):
    pass


class TestFeatherPGSPrescribedParentConnectUpstream(unittest.TestCase):
    pass


class TestFeatherPGSConnectPropagation(unittest.TestCase):
    pass


cuda_devices = get_cuda_test_devices()


for _name in (
    "test_four_bar_capture_matches_eager",
    "test_imported_connect_equality_is_enforced",
):
    add_function_test(TestFeatherPGSConnectUpstream, _name, globals()[_name], devices=cuda_devices)


for _name in ("test_disabled_loop_joint_starts_released",):
    add_function_test(TestFeatherPGSPrescribedParentConnectUpstream, _name, globals()[_name], devices=cuda_devices)


add_function_test(
    TestFeatherPGSConnectPropagation,
    "test_propagation_standalone_world_root_is_not_loop_joint",
    test_propagation_standalone_world_root_is_not_loop_joint,
    devices=cuda_devices,
)


class TestPreeliminationSignature(unittest.TestCase):
    def test_selective_option_is_keyword_only(self):
        """Keep the new selective option out of the legacy positional API."""
        parameters = inspect.signature(newton.solvers.SolverFeatherPGS).parameters
        option = parameters["bilateral_preelimination_include_mimics"]
        self.assertEqual(option.kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIs(option.default, True)

    def test_legacy_joint_gap_position_is_preserved(self):
        """Bind a legacy positional gap without silently coercing it to a boolean."""
        signature = inspect.signature(newton.solvers.SolverFeatherPGS)
        positional = [p for p in signature.parameters.values() if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD]
        self.assertEqual(len(positional), 41)
        self.assertEqual(positional[9].name, "joint_limit_activation_gap")
        arguments = [object()] + [p.default for p in positional[1:9]] + [0.123]
        bound = signature.bind_partial(*arguments, bilateral_preelimination_include_mimics=False)
        self.assertEqual(bound.arguments["joint_limit_activation_gap"], 0.123)
        self.assertIs(bound.arguments["bilateral_preelimination_include_mimics"], False)


def _run_four_bar(steps: int = 720, crank_target: float = 0.6, iterations: int = 2, **solver_kwargs):
    """Drive the connect-test four-bar and track the worst loop-anchor gap.

    Deliberately runs at a LOW iteration count: with iterative closure rows the
    anchor gap is convergence-limited (measured ~0.5 mm at 2 iterations), while
    pre-elimination enforces the closure independently of the sweep budget.
    """
    model = _build_four_bar().finalize()
    solver = newton.solvers.SolverFeatherPGS(
        model, solve_mode="matrix_free", iterations=iterations, baumgarte=0.1, **solver_kwargs
    )
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    targets = model.joint_target_q.numpy().copy()
    targets[0] = crank_target
    control.joint_target_q.assign(targets)
    gap_max = 0.0
    for _ in range(steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0
        gap_max = max(gap_max, _loop_anchor_gap(model, state_0))
    return solver, model, state_0, gap_max


def _build_overcapacity_mixed_bilateral_model():
    """Build one articulation with three connect rows and six mimic rows."""
    builder = _build_four_bar()
    # Six rows on one follower need the deprecated sparse API; joint-owned mimics allow one per joint.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        for index in range(6):
            builder.add_constraint_mimic(joint0=1, joint1=0, coef0=0.0, coef1=1.0, label=f"mimic_{index}")
    return builder.finalize()


@unittest.skipUnless(wp.get_device().is_cuda, "SolverFeatherPGS matrix-free mode requires CUDA")
class TestFeatherPGSPreelimination(unittest.TestCase):
    def test_default_preserves_all_bilateral_trajectory(self):
        """Match the implicit default to explicit all-bilateral elimination."""
        builder = _build_four_bar()
        builder.set_joint_mimic(1, 0)
        model = builder.finalize()
        solvers = [
            newton.solvers.SolverFeatherPGS(
                model, solve_mode="matrix_free", iterations=8, enable_bilateral_preelimination=True, **options
            )
            for options in ({}, {"bilateral_preelimination_include_mimics": True})
        ]
        states = [(model.state(), model.state()) for _ in solvers]
        control = model.control()
        targets = model.joint_target_q.numpy().copy()
        targets[0] = 0.6
        control.joint_target_q.assign(targets)
        for _ in range(32):
            for index, solver in enumerate(solvers):
                state_in, state_out = states[index]
                state_in.clear_forces()
                solver.step(state_in, state_out, control, None, 1.0 / 240.0)
                states[index] = (state_out, state_in)
            for attribute in ("joint_q", "joint_qd", "body_q", "body_qd"):
                a, b = [getattr(pair[0], attribute).numpy() for pair in states]
                self.assertTrue(np.isfinite(a).all())
                np.testing.assert_array_equal(a, b)
        for solver in solvers:
            self.assertTrue(solver._bilateral_preelimination_include_mimics)
            self.assertTrue(solver._preelim_active)
            count = int(solver._preelim_nB.numpy()[0])
            slots = solver._preelim_slots.numpy()[:count]
            selected_types = solver._row_type.numpy()[0, slots]
            self.assertEqual(int(np.count_nonzero(selected_types == PGS_CONSTRAINT_TYPE_MIMIC)), 1)
            self.assertGreater(int(np.count_nonzero(selected_types == PGS_CONSTRAINT_TYPE_CONNECT)), 0)

    def test_mimics_can_remain_iterative_when_total_rows_exceed_capacity(self):
        """Pre-eliminate connect rows while leaving excess mimic rows iterative."""
        model = _build_overcapacity_mixed_bilateral_model()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            solver_all = newton.solvers.SolverFeatherPGS(
                model,
                solve_mode="matrix_free",
                enable_bilateral_preelimination=True,
            )
        self.assertFalse(solver_all._preelim_active)
        self.assertTrue(any("9 bilateral rows" in str(w.message) for w in caught))

        solver = newton.solvers.SolverFeatherPGS(
            model,
            solve_mode="matrix_free",
            iterations=8,
            enable_bilateral_preelimination=True,
            bilateral_preelimination_include_mimics=False,
        )
        self.assertTrue(solver._preelim_active)
        self.assertEqual(solver._preelim_count, 1)

        state_0, state_1 = model.state(), model.state()
        control = model.control()
        for _ in range(16):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, None, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0

        self.assertTrue(np.isfinite(state_0.body_q.numpy()).all())
        self.assertGreater(int(solver._preelim_nB.numpy()[0]), 0)
        self.assertLessEqual(int(solver._preelim_nB.numpy()[0]), 3)
        selected = solver._preelim_slots.numpy()[: int(solver._preelim_nB.numpy()[0])]
        selected_types = solver._row_type.numpy()[0, selected]
        np.testing.assert_array_equal(selected_types, np.full(len(selected), PGS_CONSTRAINT_TYPE_CONNECT))
        counts = solver._constraint_count.numpy()
        rows = solver._row_type.numpy()
        seen = {int(row) for world in range(rows.shape[0]) for row in rows[world, : counts[world]]}
        self.assertIn(int(PGS_CONSTRAINT_TYPE_CONNECT), seen)
        self.assertIn(int(PGS_CONSTRAINT_TYPE_MIMIC), seen)

    def test_closure_exact_at_low_iterations(self):
        """Pre-elimination makes the loop closure iteration-independent.

        At 2 PGS iterations the iterative CONNECT rows cannot converge (the
        anchor gap plateaus near half a millimetre); the pre-eliminated solve
        must hold the closure 100x tighter through the same driven stroke.
        This is the regression-first differential: without the feature the
        ratio is 1 and the assertions below fail.
        """
        solver_off, _, _state_off, gap_off = _run_four_bar()
        self.assertFalse(solver_off._preelim_active)
        solver_on, _, state_on, gap_on = _run_four_bar(enable_bilateral_preelimination=True)
        self.assertTrue(solver_on._preelim_active)
        self.assertEqual(solver_on._preelim_count, 1)

        self.assertTrue(np.isfinite(state_on.body_q.numpy()).all())
        self.assertGreater(gap_off, 1.0e-4, "iterative baseline unexpectedly tight; differential is meaningless")
        self.assertLess(gap_on, 2.0e-5, f"pre-eliminated closure gap {gap_on:.2e} m not exact")
        self.assertLess(gap_on, 0.01 * gap_off, f"expected >100x tightening, got {gap_off / max(gap_on, 1e-12):.1f}x")

    @unittest.skip(
        "Obsolete drive-only fixture: dense non-contact rows deliberately cold-start; "
        "test_noncontact_dense_cache_is_cold_initialized covers that contract. "
        "A contact-bearing fixture is required to exercise warm-start projection ordering."
    )
    def test_dense_warmstart_preserves_projected_closure(self):
        """Project loaded contact history and an unsaturated drive kick at zero sweeps."""
        builder = _build_four_bar()
        # An unlimited drive stays in the augmented predictor, so reversing it
        # still changes velocity in the final step with no PGS drive sweeps.
        builder.joint_effort_limit[0] = float("inf")
        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(-0.04, 0.0, 0.5), wp.quat_identity()),
            hx=0.02,
            hy=0.1,
            hz=0.1,
        )
        model = builder.finalize()
        solver = newton.solvers.SolverFeatherPGS(
            model,
            solve_mode="matrix_free",
            iterations=2,
            baumgarte=0.1,
            articulation_constraint_max=128,
            enable_bilateral_preelimination=True,
            contact_warmstart=True,
        )
        pipeline = newton.CollisionPipeline(model, contact_matching="latest")
        contacts = pipeline.contacts()
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        targets = model.joint_target_q.numpy().copy()
        targets[0] = 0.6
        control.joint_target_q.assign(targets)
        for _ in range(120):
            pipeline.collide(state_0, contacts)
            state_0.clear_forces()
            solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0

        count = int(solver._constraint_count.numpy()[0])
        contact_rows = solver._row_type.numpy()[0, :count] == PGS_CONSTRAINT_TYPE_CONTACT
        self.assertTrue(contact_rows.any())
        cache_peak = np.max(solver._impulses.numpy()[0, :count][contact_rows])
        self.assertGreater(cache_peak, 1.0e-4, "loaded contact cache stayed empty")
        previous_predictor = solver._v_hat.numpy().copy()
        targets[0] = -0.6
        control.joint_target_q.assign(targets)
        solver._iterations = 0
        pipeline.collide(state_0, contacts)
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
        gap = _loop_anchor_gap(model, state_1)
        self.assertTrue(solver._preelim_active)
        self.assertTrue(np.isfinite(state_1.body_q.numpy()).all())
        self.assertGreater(np.max(solver._impulses.numpy()), 1.0e-4)
        self.assertGreater(np.linalg.norm(solver._v_hat.numpy() - previous_predictor), 1.0e-3)
        self.assertLess(gap, 2.0e-5, f"warm-started projection left a {gap:.2e} m closure gap")

    def test_propagation_uses_iterative_bilateral_fallback(self):
        """Keep bilateral rows active when propagation lacks a projected response."""
        builder, _, _ = _build_two_revolute_chain(0.0, 1.0)
        model = builder.finalize()
        with self.assertWarnsRegex(UserWarning, "propagation"):
            solver = newton.solvers.SolverFeatherPGS(
                model,
                solve_mode="matrix_free",
                articulated_contact_response="propagation",
                enable_bilateral_preelimination=True,
            )
        self.assertFalse(solver._preelim_active)
        self.assertGreater(solver._mimic_count, 0)

    def test_mechanism_behavior_preserved(self):
        """The rocker still tracks the crank through the closed loop (parallel four-bar)."""
        _, _, state, _ = _run_four_bar(enable_bilateral_preelimination=True, iterations=16)
        q = state.joint_q.numpy()
        self.assertAlmostEqual(q[2], q[0], delta=0.08)
        self.assertGreater(abs(q[2]), 0.3, "rocker did not move — loop not transmitting")

    def test_rows_remain_allocated(self):
        """B rows keep their dense slots (layout/warm-start stability); they are neutralized, not removed."""
        solver, _, _, _ = _run_four_bar(steps=60, enable_bilateral_preelimination=True)
        counts = solver._constraint_count.numpy()
        rows = solver._row_type.numpy()
        seen = set()
        for w in range(rows.shape[0]):
            seen.update(rows[w, : counts[w]].tolist())
        self.assertIn(int(PGS_CONSTRAINT_TYPE_CONNECT), seen)

    def test_unsupported_mode_warns_and_falls_back(self):
        """Non-matrix-free modes warn and keep the iterative rows."""
        model = _build_four_bar().finalize()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            solver = newton.solvers.SolverFeatherPGS(
                model, solve_mode="split", iterations=16, enable_bilateral_preelimination=True
            )
        self.assertFalse(solver._preelim_active)
        self.assertTrue(any("matrix_free" in str(w.message) for w in caught))

    def test_velocity_iterations_warn_and_fall_back(self):
        """The TGS velocity pass is unsupported in v1: warn and fall back."""
        model = _build_four_bar().finalize()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            solver = newton.solvers.SolverFeatherPGS(
                model,
                solve_mode="matrix_free",
                iterations=16,
                velocity_iterations=4,
                enable_bilateral_preelimination=True,
            )
        self.assertFalse(solver._preelim_active)
        self.assertTrue(any("velocity_iterations" in str(w.message) for w in caught))


DT = 1.0 / 240.0


class TestPreeliminationKernels(unittest.TestCase):
    """Kernel-level contracts of the connect rows and the regularized elimination, on CPU."""

    def test_regularized_projection_leaves_documented_residual(self):
        """Leave the residual ``R (S + R)^-1 (J_B v + b_B)`` of the regularized block factor.

        With ``H = I``, a bilateral row ``J_B = [1, -1]`` (``S = 2``) and a second row
        ``[1, 0]``, the projection of ``v = [1, 0]`` and the corrected response of the second
        row both keep the factor ``R / (S + R)`` of their bilateral violation, with
        ``R = 1e-3 S + 1e-7``. The elimination is not exact.
        """
        device = "cpu"

        def arr(values, dtype=int):
            return wp.array(values, dtype=dtype, device=device)

        J = arr([[[1.0, -1.0], [1.0, 0.0]]], float)
        Y = arr(J.numpy(), float)
        slots = wp.full(8, -1, dtype=int, device=device)
        nB = wp.zeros(1, dtype=int, device=device)
        S = wp.zeros(64, dtype=float, device=device)
        reg = wp.zeros(8, dtype=float, device=device)
        LS = wp.zeros(64, dtype=float, device=device)
        reg_rel, reg_floor = 1.0e-3, 1.0e-7
        wp.launch(
            feather_pgs_kernels.preelim_setup_for_size,
            dim=1,
            inputs=[
                arr([0]),
                arr([0]),
                arr([0]),
                arr([0, 1]),
                arr([0]),
                1,
                arr([-1]),
                arr([-1]),
                0,
                J,
                Y,
                2,
                reg_rel,
                reg_floor,
            ],
            outputs=[slots, nB, S, reg, LS],
            device=device,
        )
        v = arr([1.0, 0.0], float)
        wp.launch(
            feather_pgs_kernels.preelim_project_velocity_for_size,
            dim=1,
            inputs=[
                arr([0]),
                arr([0]),
                arr([0]),
                arr([0]),
                slots,
                nB,
                LS,
                J,
                Y,
                wp.zeros((1, 2), dtype=float, device=device),
                2,
            ],
            outputs=[v],
            device=device,
        )
        wp.launch(
            feather_pgs_kernels.preelim_correct_Y_for_size,
            dim=2,
            inputs=[arr([0]), arr([0]), arr([0]), arr([2]), slots, nB, LS, J, 2, 2, 1],
            outputs=[Y],
            device=device,
        )
        s_block = 2.0
        regularizer = reg_rel * s_block + reg_floor
        self.assertAlmostEqual(float(reg.numpy()[0]), regularizer, places=7)
        expected = regularizer / (s_block + regularizer)
        residual = float(v.numpy()[0] - v.numpy()[1])
        leak = float(J.numpy()[0, 0] @ Y.numpy()[0, 1])
        np.testing.assert_allclose([residual, leak], [expected, expected], rtol=1.0e-3)
        self.assertGreater(residual, 0.0)


def _run_four_bar_preelim_upstream(
    device, steps: int = 720, crank_target: float = 0.6, iterations: int = 2, **solver_kwargs
):
    """Drive the four-bar and track the worst loop-anchor gap.

    Runs at a low iteration count on purpose: with iterative connect rows the anchor gap
    is limited by convergence (about 0.5 mm at 2 iterations), while pre-elimination
    keeps it small at the same budget.
    """
    model = _build_four_bar().finalize(device=device)
    solver = SolverFeatherPGS(model, solve_mode="matrix_free", iterations=iterations, baumgarte=0.1, **solver_kwargs)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    targets = model.joint_target_q.numpy().copy()
    targets[0] = crank_target
    control.joint_target_q.assign(targets)
    gap_max = 0.0
    for _ in range(steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, DT)
        state_0, state_1 = state_1, state_0
        gap_max = max(gap_max, _loop_anchor_gap(model, state_0))
    return solver, model, state_0, gap_max


def test_closure_held_at_low_iterations(test, device):
    """Hold the loop closure tight at a low iteration count with pre-elimination.

    At 2 iterations the iterative connect rows cannot converge (the anchor gap stays
    near half a millimetre); the regularized elimination holds the closure 100x tighter
    through the same driven stroke.
    """
    solver_off, _, _, gap_off = _run_four_bar_preelim_upstream(device)
    test.assertFalse(solver_off._preelim_active)
    solver_on, _, state_on, gap_on = _run_four_bar_preelim_upstream(device, enable_bilateral_preelimination=True)
    test.assertTrue(solver_on._preelim_active)
    test.assertEqual(solver_on._preelim_count, 1)

    test.assertTrue(np.isfinite(state_on.body_q.numpy()).all())
    test.assertGreater(gap_off, 1.0e-4, "iterative baseline unexpectedly tight; the comparison is meaningless")
    test.assertLess(gap_on, 2.0e-5, f"pre-eliminated closure gap {gap_on:.2e} m too loose")
    test.assertLess(gap_on, 0.01 * gap_off, f"expected >100x tightening, got {gap_off / max(gap_on, 1e-12):.1f}x")


def test_mimic_held_at_low_iterations(test, device):
    """Hold an eliminated mimic row coupled to a closure at one iteration.

    In the parallel four-bar the coupler joint angle is minus the crank angle; a mimic row
    stating the same relationship competes with the connect rows in an iterative sweep.
    """
    errors = []
    for enabled in (False, True):
        builder = _build_four_bar()
        builder.set_joint_mimic(1, 0, coeffs=(0.0, -1.0))
        model = builder.finalize(device=device)
        solver = SolverFeatherPGS(
            model, solve_mode="matrix_free", iterations=1, baumgarte=0.1, enable_bilateral_preelimination=enabled
        )
        test.assertEqual(solver._preelim_active, enabled)
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        targets = model.joint_target_q.numpy().copy()
        targets[0] = 0.6
        control.joint_target_q.assign(targets)
        worst = 0.0
        for _ in range(480):
            solver.step(state_0, state_1, control, None, DT)
            state_0, state_1 = state_1, state_0
            q = state_0.joint_q.numpy()
            worst = max(worst, abs(float(q[1] + q[0])))
        test.assertGreater(abs(float(q[0])), 0.5)
        errors.append(worst)
    test.assertGreater(errors[0], 1.0e-5, "iterative baseline unexpectedly tight; the comparison is meaningless")
    test.assertLess(errors[1], 1.0e-6)
    test.assertLess(errors[1], 0.01 * errors[0], f"iterative {errors[0]:.2e} vs eliminated {errors[1]:.2e}")


def test_preelimination_with_free_body_in_world(test, device):
    """Leave an eliminated, contacting four-bar unchanged by a free body elsewhere in its world.

    The four-bar's coupler rests on the ground, so its contact rows are corrected by the
    elimination. Adding a separate free box gives the world two size groups and the
    gathered (not aliased) world response; the correction must reach it.
    """
    trajectories = []
    for with_box in (False, True):
        builder = _build_four_bar()
        # The coupler's underside is at 0.38 m and starts in contact.
        builder.add_ground_plane(height=0.385)
        if with_box:
            box = builder.add_body(xform=wp.transform(wp.vec3(0.8, 0.0, 0.585), wp.quat_rpy(0.2, 0.1, 0.3)))
            builder.add_shape_box(box, hx=0.08, hy=0.08, hz=0.06)
        model = builder.finalize(device=device)
        solver = SolverFeatherPGS(
            model,
            solve_mode="matrix_free",
            iterations=2,
            baumgarte=0.1,
            articulation_constraint_max=64,
            enable_bilateral_preelimination=True,
        )
        test.assertTrue(solver._preelim_active)
        test.assertEqual(solver._jy_world_aliased, not with_box)
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        targets = model.joint_target_q.numpy().copy()
        targets[0] = 0.6
        control.joint_target_q.assign(targets)
        trajectory = []
        for _ in range(120):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
            trajectory.append(state_0.joint_q.numpy()[:3].copy())
        test.assertTrue(np.isfinite(state_0.body_q.numpy()).all())
        test.assertFalse(bool(solver.constraint_overflow.numpy().any()))
        trajectories.append(np.asarray(trajectory))
    np.testing.assert_allclose(trajectories[1], trajectories[0], atol=1.0e-4)


def test_preelimination_capture_matches_eager(test, device):
    """Replay the pre-eliminated four-bar's eager trajectory from a CUDA graph."""
    model = _build_four_bar().finalize(device=device)
    control = model.control()
    targets = model.joint_target_q.numpy().copy()
    targets[0] = 0.6
    control.joint_target_q.assign(targets)
    options = {"iterations": 2, "baumgarte": 0.1, "enable_bilateral_preelimination": True}
    eager, captured = (
        SolverFeatherPGS(model, solve_mode="matrix_free", **options),
        SolverFeatherPGS(model, solve_mode="matrix_free", **options),
    )
    e0, e1 = model.state(), model.state()
    c0, c1 = model.state(), model.state()

    def step_pair(solver, s0, s1):
        solver.step(s0, s1, control, None, DT)
        solver.step(s1, s0, control, None, DT)

    step_pair(captured, c0, c1)
    step_pair(eager, e0, e1)
    # Capturing records the launches without running them.
    with wp.ScopedCapture(device=device) as capture:
        step_pair(captured, c0, c1)
    for _ in range(60):
        wp.capture_launch(capture.graph)
        step_pair(eager, e0, e1)
    np.testing.assert_allclose(c0.joint_q.numpy(), e0.joint_q.numpy(), atol=2.0e-5)
    test.assertLess(_loop_anchor_gap(model, c0), 2.0e-5)


class TestFeatherPGSPreeliminationUpstream(unittest.TestCase):
    pass


for _name in (
    "test_closure_held_at_low_iterations",
    "test_mimic_held_at_low_iterations",
    "test_preelimination_with_free_body_in_world",
    "test_preelimination_capture_matches_eager",
):
    add_function_test(TestFeatherPGSPreeliminationUpstream, _name, globals()[_name], devices=cuda_devices)


class TestFeatherPGSMimicLayout(unittest.TestCase):
    def test_reject_quaternion_mimics_at_construction(self):
        """Reject joint-owned mimics whose position and velocity layouts differ."""
        for device in get_test_devices():
            for joint_type in (newton.JointType.BALL, newton.JointType.FREE, newton.JointType.DISTANCE):
                with self.subTest(device=device, joint_type=joint_type):
                    builder, _, follower = _build_mimic_layout(joint_type)
                    model = builder.finalize(device=device)
                    q_start = model.joint_q_start.numpy()
                    qd_start = model.joint_qd_start.numpy()
                    self.assertNotEqual(
                        q_start[follower + 1] - q_start[follower], qd_start[follower + 1] - qd_start[follower]
                    )
                    with self.assertRaisesRegex(ValueError, "joint-owned mimic.*position and velocity"):
                        newton.solvers.SolverFeatherPGS(model, solve_mode="split")

    def test_scalar_and_d6_mimic_coordinate_maps(self):
        """Preserve componentwise mimic maps after a quaternion-layout free base."""
        for device in get_test_devices():
            for joint_type, dimensions in (
                (newton.JointType.REVOLUTE, 1),
                (newton.JointType.PRISMATIC, 1),
                (newton.JointType.D6, 6),
            ):
                with self.subTest(device=device, joint_type=joint_type):
                    builder, leader, follower = _build_mimic_layout(joint_type)
                    model = builder.finalize(device=device)
                    solver = newton.solvers.SolverFeatherPGS(model, solve_mode="split")
                    self.assertEqual(solver._mimic_count, dimensions)
                    for suffix, start, joint in (
                        ("q0", model.joint_q_start, follower),
                        ("q1", model.joint_q_start, leader),
                        ("dof0", model.joint_qd_start, follower),
                        ("dof1", model.joint_qd_start, leader),
                    ):
                        expected = np.arange(int(start.numpy()[joint]), int(start.numpy()[joint]) + dimensions)
                        np.testing.assert_array_equal(getattr(solver, f"_mimic_{suffix}").numpy(), expected)
                    np.testing.assert_array_equal(solver._mimic_valid_np, np.ones(dimensions))

    def test_invalid_legacy_mimic_keeps_precedence(self):
        """Preserve masking of a legacy quaternion mimic over its joint-owned entry."""
        for device in get_test_devices():
            with self.subTest(device=device):
                builder, leader, follower = _build_mimic_layout(newton.JointType.BALL)
                with self.assertWarns(DeprecationWarning):
                    builder.add_constraint_mimic(joint0=follower, joint1=leader)
                model = builder.finalize(device=device)
                with self.assertWarnsRegex(UserWarning, "references a non-1-DoF"):
                    solver = newton.solvers.SolverFeatherPGS(model, solve_mode="split")
                self.assertEqual(solver._mimic_count, 1)
                self.assertEqual(solver._mimic_legacy.numpy().tolist(), [0])
                self.assertEqual(solver._mimic_valid_np.tolist(), [0])


def _build_mimic_layout(joint_type):
    """Build a joint-owned mimic pair after a free base in one articulation."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    base = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    joints = [builder.add_joint_free(child=base)]
    parent = base
    for _ in range(2):
        child = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        linear = []
        angular = []
        if joint_type == newton.JointType.PRISMATIC:
            linear = [newton.ModelBuilder.JointDofConfig(axis=newton.Axis.X)]
        elif joint_type == newton.JointType.REVOLUTE:
            angular = [newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Z)]
        elif joint_type == newton.JointType.D6:
            linear = [
                newton.ModelBuilder.JointDofConfig(axis=axis) for axis in (newton.Axis.X, newton.Axis.Y, newton.Axis.Z)
            ]
            angular = [
                newton.ModelBuilder.JointDofConfig(axis=axis) for axis in (newton.Axis.X, newton.Axis.Y, newton.Axis.Z)
            ]
        if joint_type == newton.JointType.BALL:
            joint = builder.add_joint_ball(parent=parent, child=child)
        elif joint_type == newton.JointType.FREE:
            joint = builder.add_joint_free(parent=parent, child=child)
        elif joint_type == newton.JointType.DISTANCE:
            joint = builder.add_joint_distance(parent=parent, child=child)
        else:
            joint = builder.add_joint(joint_type, parent=parent, child=child, linear_axes=linear, angular_axes=angular)
        joints.append(joint)
        parent = child
    builder.add_articulation(joints)
    builder.set_joint_mimic(joints[2], joints[1], coeffs=(0.1, -0.5))
    return builder, joints[1], joints[2]


def _build_two_revolute_chain(coef0: float, coef1: float, legacy: bool = False):
    """Build a fixed-base chain of two revolute Z-joints with a mimic between them.

    The leader joint is position-driven; the follower joint has no drive and no spring,
    so any tracking it does comes from the mimic constraint alone.
    """
    b = newton.ModelBuilder(up_axis=newton.Axis.Z)
    # add_link (not add_body): add_body eagerly wraps each body in its own
    # single-body free-joint articulation, which would split this chain.
    link_a = b.add_link(xform=wp.transform(wp.vec3(0.2, 0.0, 0.5), wp.quat_identity()))
    b.add_shape_box(link_a, hx=0.1, hy=0.02, hz=0.02)
    j_leader = b.add_joint_revolute(
        parent=-1,
        child=link_a,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
    )
    link_b = b.add_link(xform=wp.transform(wp.vec3(0.6, 0.0, 0.5), wp.quat_identity()))
    b.add_shape_box(link_b, hx=0.1, hy=0.02, hz=0.02)
    j_follower = b.add_joint_revolute(
        parent=link_a,
        child=link_b,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
    )
    b.add_articulation([j_leader, j_follower], label="mimic_chain")
    # Drive the leader only.
    b.joint_target_ke[0] = 50.0
    b.joint_target_kd[0] = 5.0
    b.joint_target_mode[0] = int(newton.JointTargetMode.POSITION)
    # follower: q_follower = coef0 + coef1 * q_leader
    if legacy:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            b.add_constraint_mimic(joint0=j_follower, joint1=j_leader, coef0=coef0, coef1=coef1)
    else:
        b.set_joint_mimic(j_follower, j_leader, coeffs=(coef0, coef1))
    return b, j_leader, j_follower


def _run_chain(
    coef0: float, coef1: float, leader_target: float, steps: int = 600, legacy: bool = False, **solver_kwargs
):
    builder, _, _ = _build_two_revolute_chain(coef0, coef1, legacy=legacy)
    model = builder.finalize()
    solver = newton.solvers.SolverFeatherPGS(
        model, solve_mode="matrix_free", iterations=16, baumgarte=0.1, **solver_kwargs
    )
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    targets = model.joint_target_q.numpy().copy()
    targets[0] = leader_target
    control.joint_target_q.assign(targets)
    dt = 1.0 / 240.0
    for _ in range(steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0
    return solver, state_0.joint_q.numpy()


@unittest.skipUnless(wp.get_device().is_cuda, "SolverFeatherPGS matrix-free mode requires CUDA")
class TestFeatherPGSMimic(unittest.TestCase):
    def test_identity_mimic_tracks_leader(self):
        """Verify a 1:1 mimic makes the undriven follower joint track the driven leader."""
        _, q = _run_chain(coef0=0.0, coef1=1.0, leader_target=0.5)
        self.assertAlmostEqual(q[0], 0.5, delta=0.05)
        self.assertAlmostEqual(q[1], q[0], delta=0.02)

    def test_legacy_constraint_mimic_tracks_leader(self):
        """Verify the deprecated sparse mimic constraints still couple the follower."""
        _, q = _run_chain(coef0=0.1, coef1=-0.5, leader_target=0.6, legacy=True)
        self.assertAlmostEqual(q[1], 0.1 - 0.5 * q[0], delta=0.02)

    def test_legacy_constraint_overrides_joint_mimic(self):
        """Verify a legacy constraint on the same follower replaces its joint-owned mimic, as in SolverMuJoCo."""
        builder, j_leader, j_follower = _build_two_revolute_chain(coef0=0.0, coef1=1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            builder.add_constraint_mimic(joint0=j_follower, joint1=j_leader, coef0=0.1, coef1=-0.5)
        solver = newton.solvers.SolverFeatherPGS(builder.finalize(), solve_mode="matrix_free")
        self.assertEqual(solver._mimic_count, 1)
        self.assertEqual(solver._mimic_legacy.numpy().tolist(), [0])

    def test_scaled_offset_mimic(self):
        """Verify q_follower converges to coef0 + coef1 * q_leader for a scaled, offset mimic."""
        _, q = _run_chain(coef0=0.1, coef1=-0.5, leader_target=0.6)
        self.assertAlmostEqual(q[1], 0.1 - 0.5 * q[0], delta=0.02)

    def test_mimic_row_is_assembled(self):
        """Verify the solver assembles a MIMIC constraint row that carries the coupling."""
        solver, _ = _run_chain(coef0=0.0, coef1=1.0, leader_target=0.5, steps=10)
        row_types = solver._row_type.numpy()
        counts = solver._constraint_count.numpy()
        rows = [int(t) for w in range(row_types.shape[0]) for t in row_types[w, : counts[w]]]
        self.assertIn(PGS_CONSTRAINT_TYPE_MIMIC, rows)

    def test_replicated_mimics_have_articulation_local_ranges(self):
        """Verify replicated mimic rows use one compact lookup range per articulation."""
        template, _, _ = _build_two_revolute_chain(coef0=0.0, coef1=1.0)
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.replicate(template, world_count=4)
        model = builder.finalize()
        solver = newton.solvers.SolverFeatherPGS(model, solve_mode="matrix_free")

        self.assertEqual(solver._mimic_art_start.numpy().tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(solver._mimic_art_list.numpy().tolist(), [0, 1, 2, 3])

    def test_disabled_mimic_is_ignored(self):
        """Verify a disabled legacy mimic constraint leaves the follower joint uncoupled."""
        builder, _, _ = _build_two_revolute_chain(0.0, 1.0, legacy=True)
        builder.constraint_mimic_enabled[0] = False
        model = builder.finalize()
        solver = newton.solvers.SolverFeatherPGS(model, solve_mode="matrix_free", iterations=16)
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        targets = model.joint_target_q.numpy().copy()
        targets[0] = 0.5
        control.joint_target_q.assign(targets)
        dt = 1.0 / 240.0
        for _ in range(600):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, None, dt)
            state_0, state_1 = state_1, state_0
        q = state_0.joint_q.numpy()
        self.assertAlmostEqual(q[0], 0.5, delta=0.05)
        # The undriven follower swings free under gravity; it must NOT sit at the leader angle.
        self.assertNotAlmostEqual(q[1], q[0], delta=0.02)

    def test_mimic_only_rows_size_propagation_dense_capacity(self):
        """Verify mimic rows alone reserve propagation dense capacity when there are no limit or drive rows."""
        mimic_count = 20
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        joints = []
        parent = -1
        for i in range(mimic_count + 1):
            link = builder.add_link(xform=wp.transform(wp.vec3(0.2 * (i + 1), 0.0, 0.5), wp.quat_identity()))
            builder.add_shape_box(link, hx=0.1, hy=0.02, hz=0.02)
            joints.append(
                builder.add_joint_revolute(
                    parent=parent,
                    child=link,
                    axis=wp.vec3(0.0, 0.0, 1.0),
                    parent_xform=wp.transform(
                        wp.vec3(0.2, 0.0, 0.0) if parent >= 0 else wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()
                    ),
                    child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
                )
            )
            parent = link
        builder.add_articulation(joints)
        for follower in joints[1:]:
            builder.set_joint_mimic(follower, joints[0], coeffs=(0.0, 1.0))
        scene = newton.ModelBuilder(up_axis=newton.Axis.Z)
        scene.replicate(builder, world_count=2)  # per-world row counts exclude the global world
        solver = newton.solvers.SolverFeatherPGS(
            scene.finalize(),
            solve_mode="matrix_free",
            articulated_contact_response="propagation",
            propagation_same_articulation_rows=True,
            articulation_constraint_max=64,
            enable_joint_limits=False,
        )
        self.assertEqual(solver._mimic_count, 2 * mimic_count)
        self.assertGreaterEqual(solver._dense_internal_max_rows, mimic_count)
        self.assertGreaterEqual(solver._articulation_constraint_max, mimic_count)


def _build_two_revolute_chain_mimic_upstream(coef0: float, coef1: float, legacy: bool = False):
    """Build a fixed-base chain of two revolute Z-joints with a mimic between them.

    The leader joint is position-driven; the follower joint has no drive and no spring,
    so any tracking it does comes from the mimic row alone.
    """
    b = newton.ModelBuilder(up_axis=newton.Axis.Z)
    # add_link (not add_body): add_body wraps each body in its own free-joint articulation.
    link_a = b.add_link(xform=wp.transform(wp.vec3(0.2, 0.0, 0.5), wp.quat_identity()))
    b.add_shape_box(link_a, hx=0.1, hy=0.02, hz=0.02)
    j_leader = b.add_joint_revolute(
        parent=-1,
        child=link_a,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
    )
    link_b = b.add_link(xform=wp.transform(wp.vec3(0.6, 0.0, 0.5), wp.quat_identity()))
    b.add_shape_box(link_b, hx=0.1, hy=0.02, hz=0.02)
    j_follower = b.add_joint_revolute(
        parent=link_a,
        child=link_b,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
    )
    b.add_articulation([j_leader, j_follower], label="mimic_chain")
    b.joint_target_ke[0] = 50.0
    b.joint_target_kd[0] = 5.0
    b.joint_target_mode[0] = int(newton.JointTargetMode.POSITION)
    # follower: q_follower = coef0 + coef1 * q_leader
    if legacy:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            b.add_constraint_mimic(joint0=j_follower, joint1=j_leader, coef0=coef0, coef1=coef1)
    else:
        b.set_joint_mimic(j_follower, j_leader, coeffs=(coef0, coef1))
    return b, j_leader, j_follower


def check_reject_quaternion_mimics_at_construction(test, device, **solver_kwargs):
    """Reject joint-owned mimics whose position and velocity layouts differ."""
    for joint_type in (newton.JointType.BALL, newton.JointType.FREE, newton.JointType.DISTANCE):
        with test.subTest(joint_type=joint_type):
            builder, _, follower = _build_mimic_layout(joint_type)
            model = builder.finalize(device=device)
            q_start = model.joint_q_start.numpy()
            qd_start = model.joint_qd_start.numpy()
            test.assertNotEqual(q_start[follower + 1] - q_start[follower], qd_start[follower + 1] - qd_start[follower])
            with test.assertRaisesRegex(NotImplementedError, "mimic joint .* position and velocity"):
                SolverFeatherPGS(model, **solver_kwargs)


def check_scalar_and_d6_mimic_coordinate_maps(test, device, **solver_kwargs):
    """Preserve componentwise mimic maps after a quaternion-layout free base."""
    for joint_type, dimensions in (
        (newton.JointType.REVOLUTE, 1),
        (newton.JointType.PRISMATIC, 1),
        (newton.JointType.D6, 6),
    ):
        with test.subTest(joint_type=joint_type):
            builder, leader, follower = _build_mimic_layout(joint_type)
            model = builder.finalize(device=device)
            solver = SolverFeatherPGS(model, **solver_kwargs)
            test.assertEqual(solver._mimic_count, dimensions)
            for suffix, start, joint in (
                ("q0", model.joint_q_start, follower),
                ("q1", model.joint_q_start, leader),
                ("dof0", model.joint_qd_start, follower),
                ("dof1", model.joint_qd_start, leader),
            ):
                expected = np.arange(int(start.numpy()[joint]), int(start.numpy()[joint]) + dimensions)
                np.testing.assert_array_equal(getattr(solver, f"_mimic_{suffix}").numpy(), expected)
            np.testing.assert_array_equal(solver._mimic_legacy.numpy(), np.full(dimensions, -1))


def check_invalid_legacy_mimic_keeps_precedence(test, device, **solver_kwargs):
    """A legacy mimic replaces the joint-owned mimic of its follower, also when it is rejected."""
    builder, leader, follower = _build_mimic_layout(newton.JointType.BALL)
    with test.assertWarns(DeprecationWarning):
        builder.add_constraint_mimic(joint0=follower, joint1=leader)
    model = builder.finalize(device=device)
    # The legacy entry is checked instead of the joint-owned BALL mimic it replaces.
    with test.assertRaisesRegex(NotImplementedError, "must couple REVOLUTE or PRISMATIC"):
        SolverFeatherPGS(model, **solver_kwargs)


def test_runtime_coefficient_change(test, device):
    """Read joint-owned coefficients every step, also from a captured graph."""
    builder, _, follower = _build_two_revolute_chain_mimic_upstream(0.0, 1.0)
    model = builder.finalize(device=device)
    solver = SolverFeatherPGS(model, solve_mode="matrix_free", iterations=16, baumgarte=0.1)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    targets = model.joint_target_q.numpy().copy()
    targets[0] = 0.6
    control.joint_target_q.assign(targets)

    def simulate():
        for _ in range(2):
            solver.step(state_0, state_1, control, None, DT)
            solver.step(state_1, state_0, control, None, DT)

    simulate()
    with wp.ScopedCapture(device=device) as capture:
        simulate()
    for _ in range(150):
        wp.capture_launch(capture.graph)
    q = state_0.joint_q.numpy()
    test.assertAlmostEqual(q[1], q[0], delta=0.02)

    coeffs = model.joint_mimic_coeffs.numpy()
    coeffs[follower] = (0.1, -0.5)
    model.joint_mimic_coeffs.assign(coeffs)
    for _ in range(150):
        wp.capture_launch(capture.graph)
    q = state_0.joint_q.numpy()
    test.assertTrue(np.isfinite(q).all())
    test.assertAlmostEqual(q[1], 0.1 - 0.5 * q[0], delta=0.02)


def test_mimic_row_overflow_is_reported(test, device):
    """Drop and flag a mimic row beyond articulation_constraint_max rather than keep part of it."""
    builder, _, _ = _build_two_revolute_chain_mimic_upstream(0.0, 1.0)
    model = builder.finalize(device=device)
    # A zero activation gap keeps the (far) joint-limit rows out of the single row slot.
    options = {"articulation_constraint_max": 1, "joint_limit_activation_gap": 0.0, "warn_constraint_overflow": False}
    solver = SolverFeatherPGS(model, solve_mode="matrix_free", **options)
    state_0, state_1 = model.state(), model.state()
    solver.step(state_0, state_1, model.control(), None, DT)
    test.assertEqual(int(solver._mimic_slot.numpy()[0]), 0)
    test.assertFalse(bool(solver.constraint_overflow.numpy()[0]))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        # A second, consistent row on the other joint (identity coupling in both directions).
        builder.add_constraint_mimic(joint0=0, joint1=1, coef0=0.0, coef1=1.0)
    model = builder.finalize(device=device)
    solver = SolverFeatherPGS(model, solve_mode="matrix_free", **options)
    state_0, state_1 = model.state(), model.state()
    solver.step(state_0, state_1, model.control(), None, DT)
    test.assertEqual(int(solver._constraint_count.numpy()[0]), 1)
    test.assertTrue(bool(solver.constraint_overflow.numpy()[0]))
    with test.assertRaises(RuntimeError):
        solver.check_constraint_capacity()


class TestFeatherPGSMimicUpstream(unittest.TestCase):
    pass


for _name in (
    "test_runtime_coefficient_change",
    "test_mimic_row_overflow_is_reported",
):
    add_function_test(TestFeatherPGSMimicUpstream, _name, globals()[_name], devices=cuda_devices)


def _build_hinge(spring_k: float, spring_ref: float, damping: float, limit_upper: float = 10.0):
    """Build a fixed-base link on a revolute Z joint (gravity exerts no torque about the axis).

    The joint has no drive; any motion toward ``spring_ref`` comes from the passive spring
    alone, and any velocity decay beyond numerics comes from the passive damping alone.
    """
    b = newton.ModelBuilder(up_axis=newton.Axis.Z)
    link = b.add_link(xform=wp.transform(wp.vec3(0.2, 0.0, 0.5), wp.quat_identity()))
    b.add_shape_box(link, hx=0.1, hy=0.02, hz=0.02)
    j = b.add_joint_revolute(
        parent=-1,
        child=link,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
        limit_lower=-10.0,
        limit_upper=limit_upper,
    )
    b.add_articulation([j], label="hinge")
    dof = b.joint_qd_start[j]
    b.joint_spring_stiffness[dof] = spring_k
    b.joint_spring_ref[dof] = spring_ref
    b.joint_damping[dof] = damping
    return b


def _run_springs(builder, steps: int = 1200, qd0: float = 0.0, **solver_kwargs):
    model = builder.finalize()
    solver = newton.solvers.SolverFeatherPGS(model, solve_mode="matrix_free", iterations=8, **solver_kwargs)
    state_0, state_1 = model.state(), model.state()
    if qd0 != 0.0:
        qd = state_0.joint_qd.numpy()
        qd[0] = qd0
        state_0.joint_qd.assign(qd)
    control = model.control()
    dt = 1.0 / 240.0
    for _ in range(steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0
    return state_0


@unittest.skipUnless(wp.get_device().is_cuda, "SolverFeatherPGS matrix-free mode requires CUDA")
class TestFeatherPGSSprings(unittest.TestCase):
    def test_spring_converges_to_reference(self):
        """Verify an undriven joint settles at the passive spring's reference position."""
        state = _run_springs(_build_hinge(spring_k=0.5, spring_ref=0.5, damping=0.05))
        q = state.joint_q.numpy()
        self.assertAlmostEqual(float(q[0]), 0.5, delta=0.02)

    def test_springref_preloads_against_limit(self):
        """Verify a spring reference beyond the joint range preloads the joint at its limit.

        This is the Robotiq spring_link pattern: ``springref`` is unreachable, so the
        spring presses the joint against its upper stop with constant preload torque.
        """
        state = _run_springs(
            _build_hinge(spring_k=0.5, spring_ref=2.62, damping=0.05, limit_upper=0.3),
            enable_joint_limits=True,
        )
        q = state.joint_q.numpy()
        self.assertAlmostEqual(float(q[0]), 0.3, delta=0.03)

    def test_passive_damping_decays_velocity(self):
        """Verify passive joint damping decays velocity that persists when damping is disabled."""
        damped = _run_springs(_build_hinge(spring_k=0.0, spring_ref=0.0, damping=0.02), steps=480, qd0=5.0)
        undamped = _run_springs(_build_hinge(spring_k=0.0, spring_ref=0.0, damping=0.0), steps=480, qd0=5.0)
        qd_damped = abs(float(damped.joint_qd.numpy()[0]))
        qd_undamped = abs(float(undamped.joint_qd.numpy()[0]))
        self.assertGreater(qd_undamped, 4.0, "undamped joint should keep coasting")
        self.assertLess(qd_damped, 0.2 * qd_undamped)


SPRING_KEYS = ("mujoco:dof_passive_stiffness", "mujoco:dof_springref", "mujoco:dof_ref")


_SPRING_WARNING = r"does not support passive joint springs yet .*#4516"


def _hinge_model(device, *, damping: float, spring_k: float | None = None, spring_ref: float = 0.5):
    """A fixed-base link on a revolute Z joint, so gravity exerts no torque about the axis.

    ``spring_k`` registers the MuJoCo spring attributes and sets the joint's stiffness.
    """
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    SolverFeatherPGS.register_custom_attributes(builder)
    custom = {}
    if spring_k is not None:
        SolverMuJoCo.register_custom_attributes(builder)
        custom = {"mujoco:dof_passive_stiffness": spring_k, "mujoco:dof_springref": spring_ref}
    link = builder.add_link(xform=wp.transform(wp.vec3(0.2, 0.0, 0.5), wp.quat_identity()))
    builder.add_shape_box(link, hx=0.1, hy=0.02, hz=0.02)
    joint = builder.add_joint_revolute(
        parent=-1,
        child=link,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
        damping=damping,
        custom_attributes=custom,
    )
    builder.add_articulation([joint])
    return builder.finalize(device=device)


def _construct(test, model, *, solve_mode, expect_warning):
    """Construct the solver, requiring exactly the spring warning or no warning."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver = SolverFeatherPGS(model, solve_mode=solve_mode, iterations=8)
    messages = [f"{w.category.__name__}: {w.message}" for w in caught]
    test.assertEqual(len(caught), int(expect_warning), messages)
    if expect_warning:
        test.assertIs(caught[0].category, UserWarning)
        test.assertRegex(str(caught[0].message), _SPRING_WARNING)
    return solver


def test_zero_or_absent_spring_stiffness_does_not_warn(test, device, solve_mode="matrix_free"):
    """Construct silently without the spring attributes, and with zero registered stiffness."""
    _construct(test, _hinge_model(device, damping=0.1), solve_mode=solve_mode, expect_warning=False)
    _construct(test, _hinge_model(device, damping=0.1, spring_k=0.0), solve_mode=solve_mode, expect_warning=False)


def test_spring_attribute_edits_leave_damping_refresh_intact(test, device, solve_mode="matrix_free"):
    """Refresh damping on a JOINT_DOF_PROPERTIES notify while still applying no spring torque."""
    model = _hinge_model(device, damping=0.0, spring_k=0.0, spring_ref=0.5)
    solver = _construct(test, model, solve_mode=solve_mode, expect_warning=False)
    model.mujoco.dof_passive_stiffness.assign([4.0])
    model.joint_damping.assign([3.0])
    solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
    state_0, state_1 = model.state(), model.state()
    state_0.joint_qd.fill_(1.0)
    solver.step(state_0, state_1, model.control(), None, DT)
    np.testing.assert_allclose(solver._joint_tau.numpy(), [-3.0], atol=1.0e-6)


def test_registration_adds_no_spring_attributes(test, device):
    """Register only FeatherPGS's own body attributes, not the MuJoCo spring attributes."""
    builder = newton.ModelBuilder()
    SolverFeatherPGS.register_custom_attributes(builder)
    test.assertFalse(any(key in builder.custom_attributes for key in SPRING_KEYS))


class TestFeatherPGSSpringsUpstream(unittest.TestCase):
    pass


for _name, _func in (
    ("test_zero_or_absent_spring_stiffness_does_not_warn", test_zero_or_absent_spring_stiffness_does_not_warn),
    (
        "test_spring_attribute_edits_leave_damping_refresh_intact",
        test_spring_attribute_edits_leave_damping_refresh_intact,
    ),
):
    add_function_test(TestFeatherPGSSpringsUpstream, _name, _func, devices=get_cuda_test_devices())
    add_function_test(
        TestFeatherPGSSpringsUpstream, f"{_name}_split", _func, devices=get_test_devices(), solve_mode="split"
    )


add_function_test(
    TestFeatherPGSSpringsUpstream,
    "test_registration_adds_no_spring_attributes",
    test_registration_adds_no_spring_attributes,
    devices=None,
)


DT_free_base_momentum = 1.0 / 200.0


STEPS = 60


# Spin about x, transverse to the stack axis, so the composite COM offset is swept through the
# rotation. A spin parallel to the offset cannot excite the term at all.
OMEGA = (20.0, 0.0, 0.0)


LINK_MASS = 1.0


LINK_HALF = 0.10  # half the stack spacing; the composite COM sits this far above the root COM


def _link_inertia(mass, radius=0.05, height=0.20):
    """Solid-cylinder inertia tensor about the link's own centre of mass."""
    i_xy = mass * (3.0 * radius**2 + height**2) / 12.0
    i_z = 0.5 * mass * radius**2
    return wp.mat33(i_xy, 0.0, 0.0, 0.0, i_xy, 0.0, 0.0, 0.0, i_z)


def _build_welded_pair(device, root_com_z=0.0):
    """Two links welded by a fixed joint: composite COM sits ``LINK_HALF`` above the root COM.

    ``root_com_z`` offsets the root link's own centre of mass from its origin, which is the case
    every real robot link has and no bare primitive does.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    root = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=LINK_MASS,
        com=wp.vec3(0.0, 0.0, root_com_z),
        inertia=_link_inertia(LINK_MASS),
    )
    upper = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 2.0 * LINK_HALF), wp.quat_identity()),
        mass=LINK_MASS,
        com=wp.vec3(0.0, 0.0, 0.0),
        inertia=_link_inertia(LINK_MASS),
    )
    free = builder.add_joint_free(parent=-1, child=root)
    weld = builder.add_joint_fixed(
        parent=root,
        child=upper,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 2.0 * LINK_HALF), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([free, weld])
    return builder.finalize(device=device)


def _build_equivalent_single(device):
    """One link carrying the welded pair's composite mass properties.

    Same total mass, same centre-of-mass offset from the body origin and same inertia about that
    centre of mass as :func:`_build_welded_pair`, so the two models are the same physical object.
    """
    total = 2.0 * LINK_MASS
    single = _link_inertia(LINK_MASS)
    i_xy = 2.0 * (single[0, 0] + LINK_MASS * LINK_HALF**2)
    i_z = 2.0 * single[2, 2]
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=total,
        com=wp.vec3(0.0, 0.0, LINK_HALF),
        inertia=wp.mat33(i_xy, 0.0, 0.0, 0.0, i_xy, 0.0, 0.0, 0.0, i_z),
    )
    joint = builder.add_joint_free(parent=-1, child=body)
    builder.add_articulation([joint])
    return builder.finalize(device=device)


def _build_serial_chain(device, depth=2):
    """Free root followed by ``depth`` revolute links in series, axes transverse to the spin."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    prev = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=LINK_MASS,
        com=wp.vec3(0.0, 0.0, 0.0),
        inertia=_link_inertia(LINK_MASS),
    )
    joints = [builder.add_joint_free(parent=-1, child=prev)]
    for level in range(depth):
        link = builder.add_link(
            xform=wp.transform(wp.vec3(0.0, 0.0, 2.0 * LINK_HALF * (level + 1)), wp.quat_identity()),
            mass=LINK_MASS,
            com=wp.vec3(0.0, 0.0, 0.0),
            inertia=_link_inertia(LINK_MASS),
        )
        joints.append(
            builder.add_joint_revolute(
                parent=prev,
                child=link,
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent_xform=wp.transform(wp.vec3(0.0, 0.0, 2.0 * LINK_HALF), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            )
        )
        prev = link
    builder.add_articulation(joints)
    return builder.finalize(device=device)


def _build_single_offset_com_tumbling(device):
    """One free link with a triaxial inertia and a centre of mass offset from its origin.

    Distinct principal moments let a non-principal spin tumble, so the angular velocity changes
    within every step. The public-to-internal shift's residual is ``(w_out - w_in) x com_offset``
    per step: it needs both the tumble and the offset to be nonzero, which is why bodies spun about
    a principal axis (and every zero-offset primitive) cannot see it. The offset magnitude matches
    a humanoid pelvis.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=2.0 * LINK_MASS,
        com=wp.vec3(0.0, 0.0, -0.076),
        inertia=wp.mat33(0.04, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.06),
    )
    joint = builder.add_joint_free(parent=-1, child=body)
    builder.add_articulation([joint])
    return builder.finalize(device=device)


def _com_velocity(model, state):
    """Centre-of-mass velocity of the whole articulation [m/s]."""
    mass = model.body_mass.numpy().astype(np.float64)
    body_qd = state.body_qd.numpy().astype(np.float64).reshape(-1, 6)
    return (mass[:, None] * body_qd[:, :3]).sum(axis=0) / mass.sum()


def _spin_free_articulation(model, omega=OMEGA):
    """Spin the articulation with its composite centre of mass at rest, then step.

    Returns the largest centre-of-mass velocity drift [m/s] seen over the run.
    """
    state_0, state_1 = model.state(), model.state()

    # The free joint's linear coordinate is the ROOT body's COM velocity, which for a multi-link
    # articulation is not the composite COM. Give it the orbital velocity that leaves the composite
    # COM at rest, so both models start from the same physical state.
    body_q = state_0.body_q.numpy().astype(np.float64).reshape(-1, 7)
    mass = model.body_mass.numpy().astype(np.float64)
    com_local = model.body_com.numpy().astype(np.float64).reshape(-1, 3)
    com_world = body_q[:, :3] + com_local  # identity orientation at build time
    composite = (mass[:, None] * com_world).sum(axis=0) / mass.sum()
    root_com = com_world[0]

    joint_qd = state_0.joint_qd.numpy()
    joint_qd[0:3] = np.cross(np.array(omega), root_com - composite)
    joint_qd[3:6] = omega
    state_0.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

    solver = SolverFeatherPGS(model)
    solver._rigid_body_angular_damping.zero_()
    control = model.control()
    reference = _com_velocity(model, state_0)
    worst = 0.0
    for _ in range(STEPS):
        solver.step(state_0, state_1, control, None, DT_free_base_momentum)
        state_0, state_1 = state_1, state_0
        worst = max(worst, float(np.linalg.norm(_com_velocity(model, state_0) - reference)))
    return worst


class TestFeatherPgsFreeBaseMomentum(unittest.TestCase):
    """Free articulations must not create linear momentum while they rotate."""

    def test_single_link_conserves_com_velocity(self):
        """A lone rotating free body in free fall holds its centre-of-mass velocity.

        With no gravity the public-to-internal shift cancels step to step, so this case passes even
        on the defective formulation; it pins the baseline, not the fix. The gravity variant below
        is the discriminating single-body case.
        """
        model = _build_equivalent_single(wp.get_device())
        self.assertLess(_spin_free_articulation(model), 1e-6)

    def test_single_link_offset_com_tumbling(self):
        """A lone tumbling free body with an offset centre of mass conserves its COM velocity.

        Tumbling changes the angular velocity within each step, so the public-to-internal shift's
        reference-point error stops cancelling step to step: on the defective formulation this body
        drifts at ~0.5 m/s within the run, versus ~1e-7 here, in every ``solve_mode`` and
        ``articulated_contact_response``. This is the single-body case the fix corrects — every
        real robot root link has an offset centre of mass; it fails without the fix.
        """
        model = _build_single_offset_com_tumbling(wp.get_device())
        self.assertLess(_spin_free_articulation(model, omega=(20.0, 0.5, 0.0)), 1e-3)

    def test_welded_pair_conserves_com_velocity(self):
        """A welded two-link articulation holds it to the integrator's own accuracy.

        The pair is the same physical object as the single link above -- same mass, centre of mass
        and inertia -- but its composite centre of mass sits away from the frame origin, so it
        exercises the reference-point bias. Drift over this run is ~0.20 m/s with the per-link
        frame correction and ~1.64 m/s without it; the bound sits between the two.
        """
        model = _build_welded_pair(wp.get_device())
        self.assertLess(_spin_free_articulation(model), 0.5)

    def test_welded_pair_drift_converges_with_timestep(self):
        """The residual drift is first-order in dt, not a fixed formulation error.

        Halving the step must roughly halve the drift. A formulation error would sit flat instead,
        which is what distinguished this defect from ordinary truncation error.
        """
        global DT_free_base_momentum  # noqa: PLW0603 - the step size is the quantity under test
        original = DT_free_base_momentum
        try:
            DT_free_base_momentum = 1.0 / 200.0
            coarse = _spin_free_articulation(_build_welded_pair(wp.get_device()))
            DT_free_base_momentum = 1.0 / 400.0
            fine = _spin_free_articulation(_build_welded_pair(wp.get_device()))
        finally:
            DT_free_base_momentum = original
        self.assertGreater(coarse, 0.0)
        self.assertLess(fine, 0.65 * coarse)

    def test_offset_root_com_conserves_com_velocity(self):
        """A root link whose own COM is offset from its origin conserves momentum too.

        This exercises the public-to-internal free-base shift, which is an identity only when the
        frame origin and the coordinate refer to the same point. It is a no-op for a bare primitive
        and the dominant error term for a real robot link.
        """
        model = _build_welded_pair(wp.get_device(), root_com_z=-0.076)
        self.assertLess(_spin_free_articulation(model), 0.5)

    def test_matches_featherstone_reference(self):
        """FeatherPGS tracks SolverFeatherstone on chains deep enough to expose the free base.

        Featherstone solves the same dynamics without FeatherPGS's frame re-centring machinery, so
        it is an independent reference. A depth-2 serial chain is the shallowest case that
        distinguishes a correct free-base bias from one that merely looks right on a welded pair.
        """
        model = _build_serial_chain(wp.get_device(), depth=2)
        trajectories = []
        for solver_cls in (SolverFeatherstone, SolverFeatherPGS):
            state_0, state_1 = model.state(), model.state()
            joint_qd = state_0.joint_qd.numpy()
            joint_qd[3:6] = OMEGA
            state_0.joint_qd.assign(joint_qd)
            newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
            if solver_cls is SolverFeatherPGS:
                solver = solver_cls(model)
                solver._rigid_body_angular_damping.zero_()
            else:
                solver = solver_cls(model, angular_damping=0.0)
            control = model.control()
            for _ in range(20):
                solver.step(state_0, state_1, control, None, DT_free_base_momentum)
                state_0, state_1 = state_1, state_0
            trajectories.append(state_0.body_qd.numpy().astype(np.float64).copy())
        self.assertLess(float(np.abs(trajectories[0] - trajectories[1]).max()), 1e-4)


DT_free_root_predictor = 1.0 / 200.0


STEPS_free_root_predictor = 80


RADIUS = 0.1


# Spin transverse to the slide so omega x v points along +/-z, straight into
# the contact normal; spin parallel to the slide cannot excite the term.
OMEGA_Y = 20.0


SLIDE_VX = 2.0


# (solve_mode, articulated_contact_response, velocity_iterations): the split
# velocity post-pass converts velocity to acceleration at two more patched
# sites, and each propagation response variant is a separately generated
# contact path
SLIDER_CONFIGS = [
    ("split", "immediate", 0),
]


SLIDER_CONFIGS_CUDA = [
    ("matrix_free", "immediate", 0),
    ("matrix_free", "immediate", 2),
    ("matrix_free", "propagation", 0),
    ("matrix_free", "propagation_fused", 0),
    ("matrix_free", "propagation_colored", 0),
]


@wp.kernel
def _record_slider_step(
    body_q: wp.array[wp.transform],
    rigid_contact_count: wp.array[int],
    step: int,
    # outputs
    heights: wp.array[float],
    peak_contacts: wp.array[int],
):
    """Record one slider sample without a per-step device synchronization."""
    heights[step] = wp.transform_get_translation(body_q[0])[2]
    peak_contacts[0] = wp.max(peak_contacts[0], rigid_contact_count[0])


def _build_slider(device, articulated=False):
    """A frictionless sphere resting on a frictionless ground plane."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    builder.default_shape_cfg.mu = 0.0
    body = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, RADIUS), wp.quat_identity()),
        mass=1.0,
        com=wp.vec3(0.0, 0.0, 0.0),
        inertia=wp.mat33(0.004, 0.0, 0.0, 0.0, 0.004, 0.0, 0.0, 0.0, 0.004),
    )
    builder.add_shape_sphere(body, radius=RADIUS)
    joints = [builder.add_joint_free(body)]
    if articulated:
        # Fused propagation deliberately keeps single free bodies on the
        # ordinary matrix-free path. A concentric fixed child makes this a
        # floating articulation without changing the frictionless invariant.
        child = builder.add_link(
            xform=wp.transform(wp.vec3(0.0, 0.0, RADIUS), wp.quat_identity()),
            mass=0.1,
            com=wp.vec3(0.0, 0.0, 0.0),
            inertia=wp.mat33(0.0004, 0.0, 0.0, 0.0, 0.0004, 0.0, 0.0, 0.0, 0.0004),
        )
        joints.append(builder.add_joint_fixed(body, child))
    builder.add_articulation(joints)
    builder.add_ground_plane()
    return builder.finalize(device=device)


def _slide_heights(model, omega_y, solve_mode, response="immediate", velocity_iterations=0):
    """Slide the sphere at ``SLIDE_VX`` while spinning at ``omega_y``.

    Returns per-step COM heights [m] and the peak live contact count, so the
    caller can prove the comparison exercised actual contact rows rather than
    two contact-free (and therefore trivially matching) runs.
    """
    state_0, state_1 = model.state(), model.state()
    joint_qd = state_0.joint_qd.numpy()
    joint_qd[0:3] = (SLIDE_VX, 0.0, 0.0)
    joint_qd[3:6] = (0.0, omega_y, 0.0)
    state_0.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

    solver = SolverFeatherPGS(
        model,
        solve_mode=solve_mode,
        articulated_contact_response=response,
        velocity_iterations=velocity_iterations,
    )
    solver._rigid_body_angular_damping.zero_()
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
    contacts = pipeline.contacts()
    control = model.control()
    heights = wp.empty(STEPS_free_root_predictor, dtype=wp.float32, device=model.device)
    peak_contacts = wp.zeros(1, dtype=wp.int32, device=model.device)
    if response == "propagation_fused":
        if not solver._propagation_contacts_enabled():
            raise AssertionError("propagation_fused test model did not activate propagation contacts")
        if solver._pgs_solve_propagation_full_iteration_kernel is None:
            raise AssertionError("propagation_fused test model did not construct the fused iteration kernel")
    for step in range(STEPS_free_root_predictor):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT_free_root_predictor)
        state_0, state_1 = state_1, state_0
        wp.launch(
            _record_slider_step,
            dim=1,
            inputs=[state_0.body_q, contacts.rigid_contact_count, step],
            outputs=[heights, peak_contacts],
            device=model.device,
        )
    if response == "propagation_fused" and int(solver._propagation_constraint_count.numpy().sum()) <= 0:
        raise AssertionError("propagation_fused test did not build any propagation contact rows")
    return heights.numpy(), int(peak_contacts.numpy()[0])


def _world_com(model, state):
    """World position of the root body's centre of mass [m]."""
    com_local = model.body_com.numpy().astype(np.float64).reshape(-1, 3)[0]
    q = state.body_q.numpy().astype(np.float64).reshape(-1, 7)[0]
    rot = wp.quat(*(float(c) for c in q[3:7]))
    return np.asarray(q[0:3]) + np.asarray(wp.quat_rotate(rot, wp.vec3(*com_local)))


def _com_positions(model, state_0, state_1, steps):
    """Step the model contact-free; return per-step world COM positions [m]."""
    solver = SolverFeatherPGS(model)
    solver._rigid_body_angular_damping.zero_()
    control = model.control()
    positions = []
    for _ in range(steps):
        solver.step(state_0, state_1, control, None, DT_free_root_predictor)
        state_0, state_1 = state_1, state_0
        positions.append(_world_com(model, state_0))
    return np.asarray(positions)


class TestFeatherPgsFreeRootPredictor(unittest.TestCase):
    """Constraint rows must see the COM velocity the integrator realizes."""

    def test_frictionless_slide_is_spin_invariant(self):
        """Spinning a frictionless sliding sphere must not change its height trajectory.

        Without friction the spin exerts no force, so the height trajectory is
        exactly that of the unspun sphere.  A predictor missing the
        ``omega x v`` term feeds the normal rows a phantom vertical velocity of
        ``dt * omega_y * vx`` (0.2 m/s here): the sphere's height diverges by
        millimetres within the run, in every ``solve_mode``.  With the conventions
        agreeing the trajectories match to solver precision.
        """
        device = wp.get_device()
        configs = list(SLIDER_CONFIGS)
        if device.is_cuda:
            configs += SLIDER_CONFIGS_CUDA
        for solve_mode, response, vel_iters in configs:
            with self.subTest(solve_mode=solve_mode, response=response, velocity_iterations=vel_iters):
                model = _build_slider(device, articulated=response == "propagation_fused")
                still, contacts_still = _slide_heights(model, 0.0, solve_mode, response, vel_iters)
                spinning, contacts_spin = _slide_heights(model, OMEGA_Y, solve_mode, response, vel_iters)
                # the comparison is vacuous unless both runs really ride on
                # active contact rows at the rest height
                self.assertGreater(min(contacts_still, contacts_spin), 0, "no contacts were generated")
                for label, h in (("still", still), ("spinning", spinning)):
                    self.assertLess(float(np.abs(h - RADIUS).max()), 2e-3, f"{label} run left the ground support band")
                divergence = float(np.abs(spinning - still).max())
                self.assertLess(divergence, 1e-4, f"spin changed a frictionless slide by {divergence} m")

    def _anchored_spinner_drift(self, child_xform, com, omega=(0.0, 0.0, 10.0)):
        """Max world COM drift [m] of a force-free spinner with an offset joint anchor."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        body = builder.add_link(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            mass=1.0,
            com=com,
            inertia=wp.mat33(0.01, 0.0, 0.0, 0.0, 0.012, 0.0, 0.0, 0.0, 0.014),
        )
        free = builder.add_joint_free(body, child_xform=child_xform)
        builder.add_articulation([free])
        model = builder.finalize(device=wp.get_device())

        state_0, state_1 = model.state(), model.state()
        joint_qd = state_0.joint_qd.numpy()
        joint_qd[3:6] = omega
        state_0.joint_qd.assign(joint_qd)
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
        start = _world_com(model, state_0)

        positions = _com_positions(model, state_0, state_1, STEPS_free_root_predictor)
        return float(np.linalg.norm(positions - start[None, :], axis=1).max())

    def test_anchored_spinner_com_stays_put(self):
        """A force-free spinning body with an offset joint anchor keeps its COM in place.

        The free joint's coordinate tracks the child ANCHOR frame, offset from
        the body's COM by ``child_xform``.  With zero COM velocity and no
        forces, the COM must stay put and the anchor must orbit it.  A lever
        built from ``body_com`` alone pins the anchor instead, so the COM
        orbits at the anchor-offset radius -- a 0.2 m excursion within half a
        turn, from nothing; advancing the anchor with the linearized lever
        velocity instead had ``O(omega^2 * r * dt^2)`` local error that
        accumulated into first-order global drift. The anchor is now
        reconstructed from the directly integrated COM
        (``p_new = x_com_new - R_new * r_ac``, as SolverFeatherstone does), so
        a force-free COM holds to roundoff at ANY step size and the bound can
        sit far below both failure modes.  The second configuration adds a
        rotated anchor and a non-collinear COM offset, so both transform
        compositions in the lever are exercised, with a tumbling-capable
        triaxial inertia.
        """
        configs = [
            ("plain-offset", wp.transform(wp.vec3(0.1, 0.0, 0.0), wp.quat_identity()), wp.vec3(0.0, 0.0, 0.0)),
            (
                "rotated-anchor-offset-com",
                wp.transform(wp.vec3(0.1, 0.02, 0.0), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.7)),
                wp.vec3(0.03, -0.05, 0.02),
            ),
        ]
        for label, child_xform, com in configs:
            with self.subTest(config=label):
                drift = self._anchored_spinner_drift(child_xform, com)
                self.assertLess(drift, 1e-4, f"COM of a force-free spinner drifted {drift} m")

    def test_integrator_transport_identities(self):
        """Pin the integrator's root transport and descendant pass-through directly.

        A single launch of ``integrate_generalized_joints`` with zero
        ``joint_qdd`` must realize the two velocity identities: the ROOT free
        joint's linear coordinate gains exactly ``omega x v * dt`` (the
        transport term the predictor and the qdd conversion mirror), and a
        DESCENDANT free joint's linear coordinate passes through unchanged --
        its coordinate is a relative twist in the parent anchor frame where
        the root rule is not derived, so the branch is deliberately
        component-wise.  (End-to-end descendant momentum is NOT asserted
        anywhere: a force-free nested free body's world COM velocity drifts
        through the dynamics stages independently of integration -- a
        pre-existing gap outside this fix's scope.)
        """
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        parent = builder.add_link(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
            mass=1.0,
            com=wp.vec3(0.0, 0.0, 0.0),
            inertia=wp.mat33(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01),
        )
        child = builder.add_link(
            xform=wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity()),
            mass=1.0,
            com=wp.vec3(0.0, 0.0, 0.0),
            inertia=wp.mat33(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01),
        )
        root = builder.add_joint_free(parent)
        nested = builder.add_joint_free(child, parent=parent)
        builder.add_articulation([root, nested])
        model = builder.finalize(device=wp.get_device())

        state = model.state()
        joint_qd = state.joint_qd.numpy()
        joint_qd[0:3] = (0.4, 0.0, 0.0)
        joint_qd[3:6] = (0.0, 0.0, 3.0)
        joint_qd[6:9] = (0.4, 0.0, 0.0)
        joint_qd[9:12] = (0.0, 0.0, 3.0)
        state.joint_qd.assign(joint_qd)

        device = wp.get_device()
        q_new = wp.zeros_like(state.joint_q)
        qd_new = wp.zeros_like(state.joint_qd)
        wp.launch(
            integrate_generalized_joints,
            dim=model.joint_count,
            inputs=[
                model.joint_type,
                model.joint_parent,
                model.joint_child,
                model.joint_q_start,
                model.joint_qd_start,
                wp.zeros(model.joint_count, dtype=wp.int32, device=device),
                model.joint_dof_dim,
                model.body_com,
                model.joint_X_c,
                state.joint_q,
                state.joint_qd,
                wp.zeros_like(state.joint_qd),  # joint_qdd = 0
                DT_free_root_predictor,
                wp.zeros(model.body_count, dtype=wp.float32, device=device),
                wp.ones(model.joint_count, dtype=wp.int32, device=device),
            ],
            outputs=[q_new, qd_new],
            device=device,
        )
        out = qd_new.numpy()
        transport = np.cross((0.0, 0.0, 3.0), (0.4, 0.0, 0.0)) * DT_free_root_predictor
        np.testing.assert_allclose(out[0:3], np.array((0.4, 0.0, 0.0)) + transport, rtol=1e-6)
        np.testing.assert_allclose(out[3:6], (0.0, 0.0, 3.0), atol=1e-7)
        np.testing.assert_allclose(out[6:9], (0.4, 0.0, 0.0), atol=1e-7, err_msg="descendant gained transport")
        np.testing.assert_allclose(out[9:12], (0.0, 0.0, 3.0), atol=1e-7)

    def test_predictor_and_qdd_transport_identities(self):
        """Make prediction and solved-velocity conversion match root integration."""
        device = wp.get_device()
        model = _build_slider(device)
        solver = SolverFeatherPGS(model)
        solver._rigid_body_angular_damping.zero_()
        self.assertEqual(solver._free_root_joint_count, 1)
        state_in, state_out = model.state(), model.state()
        state_aug = solver._prepare_augmented_state(state_in, state_out, model.control())

        qd_np = np.array((2.0, 0.0, 0.0, 0.0, 20.0, 0.0), dtype=np.float32)
        qdd_np = np.array((0.3, -0.4, 0.5, 0.1, -0.2, 0.25), dtype=np.float32)
        state_in.joint_qd.assign(qd_np)
        solver._qd_work.assign(qd_np)
        state_aug._joint_qdd.assign(qdd_np)
        solver._stage3_compute_v_hat(state_in, state_aug, DT_free_root_predictor, solver._qd_work)
        transport = np.cross(qd_np[3:6], qd_np[0:3])
        expected_v_hat = qd_np + qdd_np * DT_free_root_predictor
        expected_v_hat[0:3] += transport * DT_free_root_predictor
        np.testing.assert_allclose(solver._v_hat.numpy(), expected_v_hat, rtol=1e-6, atol=1e-7)

        v_out_np = np.array((1.8, 0.1, -0.15, 0.05, 19.5, 0.25), dtype=np.float32)
        solver._v_out.assign(v_out_np)
        solver._stage6_update_qdd(state_in, state_aug, DT_free_root_predictor)
        expected_qdd = (v_out_np - qd_np) / DT_free_root_predictor
        expected_qdd[0:3] -= transport
        np.testing.assert_allclose(state_aug._joint_qdd.numpy(), expected_qdd, rtol=1e-6, atol=1e-5)
        solver._stage6_integrate(state_in, state_aug, state_out, DT_free_root_predictor)
        np.testing.assert_allclose(state_out.joint_qd.numpy(), v_out_np, rtol=1e-6, atol=1e-6)

    def test_compact_root_metadata_includes_standalone_joint(self):
        """Keep every world root, but no descendant, in the compact launch."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        inertia = wp.mat33(0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01)
        root_a = builder.add_link(mass=1.0, inertia=inertia)
        child = builder.add_link(mass=1.0, inertia=inertia)
        root_b = builder.add_link(mass=1.0, inertia=inertia)
        joint_a = builder.add_joint_free(root_a)
        nested = builder.add_joint_free(child, parent=root_a)
        joint_b = builder.add_joint_free(root_b)
        builder.add_articulation([joint_a, nested])

        model = builder.finalize(device=wp.get_device())
        solver = SolverFeatherPGS(model)
        np.testing.assert_array_equal(solver._free_root_joint_indices.numpy(), (joint_a, joint_b))

    def test_final_velocity_conversion_stores_transport_adjusted_qdd(self):
        """Keep the final velocity-postpass conversion observable and wired."""
        device = wp.get_device()
        model = _build_slider(device)
        solver = SolverFeatherPGS(model, solve_mode="split")
        solver._rigid_body_angular_damping.zero_()
        state_in, state_out = model.state(), model.state()
        state_aug = solver._prepare_augmented_state(state_in, state_out, model.control())

        qd_np = np.array((2.0, 0.0, 0.0, 0.0, 20.0, 0.0), dtype=np.float32)
        target_np = np.array((1.8, 0.1, -0.15, 0.05, 19.5, 0.25), dtype=np.float32)
        state_in.joint_qd.assign(qd_np)
        solver._v_out.assign(target_np)
        solver._stage6_write_final_velocity(state_in, state_aug, state_out, DT_free_root_predictor)

        expected_qdd = (target_np - qd_np) / DT_free_root_predictor
        expected_qdd[0:3] -= np.cross(qd_np[3:6], qd_np[0:3])
        np.testing.assert_allclose(state_aug._joint_qdd.numpy(), expected_qdd, rtol=1e-6, atol=1e-5)
        np.testing.assert_allclose(state_out.joint_qd.numpy(), target_np, rtol=1e-6, atol=1e-7)

    def test_captured_step_matches_uncaptured(self):
        """Replaying a captured collide+step reproduces the uncaptured trajectory.

        The transport corrections are fixed-size device launches guarded by a
        host-side flag, so they must record and replay exactly; a divergence
        here means part of the correction ran host-side.  Skipped on CPU
        devices.
        """
        device = wp.get_device()
        if not device.is_cuda:
            self.skipTest("CUDA graph capture requires a CUDA device")
        heights_ref, _ = _slide_heights(_build_slider(device), OMEGA_Y, "matrix_free")

        model = _build_slider(device)
        state_0, state_1 = model.state(), model.state()
        joint_qd = state_0.joint_qd.numpy()
        joint_qd[0:3] = (SLIDE_VX, 0.0, 0.0)
        joint_qd[3:6] = (0.0, OMEGA_Y, 0.0)
        state_0.joint_qd.assign(joint_qd)
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
        solver = SolverFeatherPGS(model, solve_mode="matrix_free")
        solver._rigid_body_angular_damping.zero_()
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
        contacts = pipeline.contacts()
        control = model.control()

        def one_step():
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT_free_root_predictor)
            # copy back instead of swapping so the captured pointers are stable
            wp.copy(state_0.body_q, state_1.body_q)
            wp.copy(state_0.body_qd, state_1.body_qd)
            wp.copy(state_0.joint_q, state_1.joint_q)
            wp.copy(state_0.joint_qd, state_1.joint_qd)

        one_step()  # warm-up, matches the reference run's first step
        with wp.ScopedCapture(device) as capture:
            one_step()  # capture RECORDS this step without executing it
        heights = []
        for _ in range(STEPS_free_root_predictor - 1):
            wp.capture_launch(capture.graph)
            heights.append(float(state_0.body_q.numpy().reshape(-1, 7)[0, 2]))
        np.testing.assert_allclose(heights, heights_ref[1:], atol=1e-6)


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
        solver = newton.solvers.SolverFeatherPGS(model, solve_mode=mode)
        solver._rigid_body_angular_damping.zero_()
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
        solver = newton.solvers.SolverFeatherPGS(model)
        solver._rigid_body_angular_damping.zero_()
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
        solver = newton.solvers.SolverFeatherPGS(model)
        solver._rigid_body_angular_damping.zero_()
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
        solver = newton.solvers.SolverFeatherPGS(model)
        solver._rigid_body_angular_damping.zero_()
        control = model.control()
        for _ in range(steps):
            solver.step(state, output, control, None, duration / steps)
            state, output = output, state
        errors.append(np.linalg.norm(_body_omega(state) - reference))
    test.assertLess(errors[0], 0.1)
    test.assertLess(errors[1], 0.35 * errors[0])


def _mixed_scene(device, *, arm, totes):
    """Build two worlds with free bodies on both sides of a 19-DOF articulation."""
    template = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    if totes:
        template.add_body(
            xform=wp.transform(wp.vec3(-0.5, 0.0, 0.0), wp.quat_identity()),
            mass=1.0,
            inertia=wp.mat33(np.diag([0.001, 0.02, 0.0205])),
            com=wp.vec3(0.03, -0.02, 0.01),
        )
    if arm:
        joints = []
        parent = -1
        for _ in range(19):
            body = template.add_link(mass=0.5, inertia=wp.mat33(np.eye(3) * 0.02))
            joints.append(
                template.add_joint_revolute(
                    parent,
                    body,
                    parent_xform=wp.transform(wp.vec3(0.1, 0.0, 0.0), wp.quat_identity()),
                    axis=newton.Axis.Z,
                    target_ke=40.0,
                    target_kd=4.0,
                    armature=0.01,
                )
            )
            parent = body
        template.add_articulation(joints)
    if totes:
        template.add_body(
            xform=wp.transform(wp.vec3(0.5, 0.0, 0.0), wp.quat_identity()),
            mass=2.0,
            inertia=wp.mat33(np.diag([0.04, 0.002, 0.041])),
            com=wp.vec3(-0.02, 0.01, 0.04),
        )
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.replicate(template, 2, spacing=(3.0, 0.0, 0.0))
    model = builder.finalize(device=device)
    state, output = model.state(), model.state()
    velocity = state.joint_qd.numpy()
    free_joints = np.flatnonzero(model.joint_type.numpy() == newton.JointType.FREE)
    for i, joint in enumerate(free_joints):
        start = int(model.joint_qd_start.numpy()[joint])
        velocity[start + 3 : start + 6] = (100.0, 20.0, 50.0) if i % 2 == 0 else (20.0, 100.0, 50.0)
    state.joint_qd.assign(velocity)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    control = model.control()
    control.joint_target_q.fill_(0.1)
    solver = newton.solvers.SolverFeatherPGS(
        model, solve_mode="matrix_free" if wp.get_device(device).is_cuda else "split"
    )
    solver._rigid_body_angular_damping.zero_()
    return model, solver, control, [state, output]


def test_mixed_arm_and_spinning_bodies(test, device):
    """Preserve isolated arm and rotor dynamics with mixed response sizes and graph replay."""
    executions = ("eager", "graph") if wp.get_device(device).is_cuda else ("eager",)
    for execution in executions:
        with test.subTest(execution=execution):
            scenes = [
                _mixed_scene(device, arm=True, totes=True),
                _mixed_scene(device, arm=True, totes=False),
                _mixed_scene(device, arm=False, totes=True),
            ]
            model, solver, _, states = scenes[0]
            free_bodies = (0, 20, 21, 41)
            inertias = model.body_inertia.numpy().astype(np.float64)

            def energies(state, free_bodies=free_bodies, inertias=inertias):
                """Measure each free body's rotational energy in its inertia frame."""
                values = []
                for body in free_bodies:
                    omega = _body_omega(state, body)
                    values.append(omega @ inertias[body] @ omega)
                return np.asarray(values)

            initial_energy = energies(states[0])
            test.assertIn(6, solver._L_by_size)
            test.assertIn(19, solver._L_by_size)

            def step(scenes=scenes):
                """Advance the mixed scene and its independent controls together."""
                for _, scene_solver, control, pair in scenes:
                    scene_solver.step(pair[0], pair[1], control, None, 1.0 / 240.0)
                    pair[0], pair[1] = pair[1], pair[0]

            # Compile and initialize the same layouts before capture.
            step()
            graph = None
            if execution == "graph":
                with wp.ScopedCapture(device=device) as capture:
                    step()
                    step()
                graph = capture.graph
            for _ in range(100):
                if graph is None:
                    step()
                    step()
                else:
                    wp.capture_launch(graph)
                current = scenes[0][3][0]
                np.testing.assert_allclose(energies(current) / initial_energy, 1.0, atol=1.0e-3, rtol=0.0)
                for field in ("body_q", "body_qd"):
                    width = 7 if field == "body_q" else 6
                    mixed = getattr(current, field).numpy().reshape(2, 21, width)
                    arm = getattr(scenes[1][3][0], field).numpy().reshape(2, 19, width)
                    totes = getattr(scenes[2][3][0], field).numpy().reshape(2, 2, width)
                    np.testing.assert_allclose(mixed[:, 1:20], arm, atol=3.0e-5, rtol=1.0e-6)
                    np.testing.assert_allclose(mixed[:, [0, 20]], totes, atol=3.0e-5, rtol=1.0e-6)
            test.assertGreater(float(np.abs(scenes[1][3][0].joint_q.numpy()).max()), 0.01)


class TestFeatherPGSGyroscopic(unittest.TestCase):
    pass


for _device in get_test_devices():
    add_function_test(
        TestFeatherPGSGyroscopic,
        "test_mixed_arm_and_spinning_bodies",
        test_mixed_arm_and_spinning_bodies,
        devices=[_device],
    )
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


# Off-origin center of mass, so the tests distinguish the body COM from the body origin.
COM_LOCAL = (0.05, -0.03, 0.02)


OMEGA_joint_qd_convention = (0.0, 0.0, 3.0)


V_COM = (0.1, -0.2, 0.05)


def _build_free_body(origin, device):
    """Build a single gravity-free, contact-free free-root body placed at ``origin``."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(
        xform=wp.transform(wp.vec3(*origin), wp.quat_identity()),
        mass=1.0,
        com=wp.vec3(*COM_LOCAL),
        inertia=wp.mat33(0.02, 0.0, 0.0, 0.0, 0.03, 0.0, 0.0, 0.0, 0.04),
    )
    joint = builder.add_joint_free(parent=-1, child=body)
    builder.add_articulation([joint])
    return builder.finalize(device=device)


def _step_once(model):
    """Step FeatherPGS once from a seeded free-joint twist."""
    state_in, state_out = model.state(), model.state()
    joint_qd = state_in.joint_qd.numpy()
    joint_qd[0:3] = V_COM
    joint_qd[3:6] = OMEGA_joint_qd_convention
    state_in.joint_qd.assign(joint_qd)

    solver = SolverFeatherPGS(model)
    solver.step(state_in, state_out, control=model.control(), contacts=None, dt=DT)
    return solver, state_out


def _refresh_body_qd(model, source, helper):
    """Rebuild ``body_qd`` from ``source``'s joint state using ``helper``, as a reset would."""
    probe = model.state()
    probe.joint_q.assign(source.joint_q)
    probe.joint_qd.assign(source.joint_qd)
    helper(model, probe.joint_q, probe.joint_qd, probe)
    return probe.body_qd.numpy().copy()


def _consumer_refresh(solver, model, source):
    """Mimic an integration layer dispatching FK on the solver's declared convention."""
    if getattr(solver, "joint_qd_public_convention", False):
        return _refresh_body_qd(model, source, newton.eval_fk)
    return _refresh_body_qd(model, source, eval_fk_with_velocity_conversion)


def _com_world(model, state):
    """World-space center of mass of body 0 [m]."""
    body_q = state.body_q.numpy()[0]
    rotated = wp.quat_rotate(wp.quat(*body_q[3:7]), wp.vec3(*model.body_com.numpy()[0]))
    return np.array(body_q[0:3]) + np.array([rotated[0], rotated[1], rotated[2]])


class TestFeatherPGSJointQdConvention(unittest.TestCase):
    def test_solver_declares_public_joint_qd_convention(self):
        """The dispatch key is declared on FeatherPGS and absent on vanilla Featherstone."""
        self.assertTrue(SolverFeatherPGS.joint_qd_public_convention)

        model = _build_free_body((0.0, 0.0, 0.6), "cpu")
        self.assertTrue(SolverFeatherPGS(model).joint_qd_public_convention)

        # Featherstone must NOT opt in: its free-joint qd is an internal world-origin twist,
        # so a consumer's getattr(..., False) has to keep routing it to the conversion helper.
        self.assertFalse(getattr(SolverFeatherstone, "joint_qd_public_convention", False))

    def test_public_eval_fk_round_trips_solver_body_qd(self):
        """eval_fk on FeatherPGS joint state reproduces the solver's own body_qd."""
        model = _build_free_body((5.0, 0.0, 0.6), "cpu")
        _solver, state_out = _step_once(model)

        solver_body_qd = state_out.body_qd.numpy().copy()
        refreshed = _refresh_body_qd(model, state_out, newton.eval_fk)

        np.testing.assert_allclose(refreshed, solver_body_qd, rtol=1e-6, atol=1e-6)

    def test_featherstone_helper_injects_omega_cross_com_phantom_velocity(self):
        """The conversion helper's error is exactly omega x x_com_world."""
        model = _build_free_body((5.0, 0.0, 0.6), "cpu")
        _solver, state_out = _step_once(model)

        public = _refresh_body_qd(model, state_out, newton.eval_fk)
        converted = _refresh_body_qd(model, state_out, eval_fk_with_velocity_conversion)

        omega = public[0][3:6]
        expected_phantom = np.cross(omega, _com_world(model, state_out))
        actual_phantom = converted[0][0:3] - public[0][0:3]

        np.testing.assert_allclose(actual_phantom, expected_phantom, rtol=1e-5, atol=1e-5)
        # Angular velocity is untouched; only the linear part is re-referenced.
        np.testing.assert_allclose(converted[0][3:6], public[0][3:6], rtol=1e-6, atol=1e-6)

    def test_phantom_velocity_grows_with_distance_from_world_origin(self):
        """The phantom velocity scales with env-grid radius while eval_fk stays exact."""
        radii = (0.0, 1.0, 5.0, 20.0)
        magnitudes = []

        for radius in radii:
            with self.subTest(radius=radius):
                model = _build_free_body((radius, 0.0, 0.6), "cpu")
                _solver, state_out = _step_once(model)
                solver_body_qd = state_out.body_qd.numpy().copy()

                public = _refresh_body_qd(model, state_out, newton.eval_fk)
                converted = _refresh_body_qd(model, state_out, eval_fk_with_velocity_conversion)

                # The correct helper is radius-independent.
                np.testing.assert_allclose(public, solver_body_qd, rtol=1e-6, atol=1e-6)

                phantom = converted[0][0:3] - public[0][0:3]
                np.testing.assert_allclose(
                    phantom,
                    np.cross(public[0][3:6], _com_world(model, state_out)),
                    rtol=1e-5,
                    atol=1e-5,
                )
                magnitudes.append(float(np.linalg.norm(phantom)))

        for near, far in itertools.pairwise(magnitudes):
            self.assertGreater(far, near)

        # At a 20 m grid offset the phantom velocity dwarfs the true COM velocity.
        self.assertGreater(magnitudes[-1], 100.0 * float(np.linalg.norm(V_COM)))

    def test_consumer_dispatch_selects_correct_helper(self):
        """A consumer dispatching on the attribute is correct with it and wrong without it."""
        model = _build_free_body((20.0, 0.0, 0.6), "cpu")
        solver, state_out = _step_once(model)
        solver_body_qd = state_out.body_qd.numpy().copy()

        with_fix = _consumer_refresh(solver, model, state_out)
        np.testing.assert_allclose(with_fix, solver_body_qd, rtol=1e-6, atol=1e-6)

        # Pre-fix state: the solver never declared the convention, so getattr(..., False)
        # routed FeatherPGS through the Featherstone-internal helper.
        with mock.patch.object(SolverFeatherPGS, "joint_qd_public_convention", False):
            without_fix = _consumer_refresh(solver, model, state_out)

        phantom = without_fix[0][0:3] - solver_body_qd[0][0:3]
        np.testing.assert_allclose(
            phantom,
            np.cross(solver_body_qd[0][3:6], _com_world(model, state_out)),
            rtol=1e-5,
            atol=1e-5,
        )
        self.assertGreater(float(np.linalg.norm(phantom)), 50.0)


OMEGA_joint_qd_convention_upstream = (0.0, 0.0, 3.0)


ORIGINS = ((0.0, 0.0, 0.6), (5.0, 0.0, 0.6), (20.0, 0.0, 0.6))


def _build_free_body_joint_qd_convention_upstream(origin, device):
    """Build a single gravity-free, contact-free free-root body placed at ``origin``."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(
        xform=wp.transform(wp.vec3(*origin), wp.quat_identity()),
        mass=1.0,
        com=wp.vec3(*COM_LOCAL),
        inertia=wp.mat33(0.02, 0.0, 0.0, 0.0, 0.03, 0.0, 0.0, 0.0, 0.04),
    )
    builder.add_articulation([builder.add_joint_free(parent=-1, child=body)])
    return builder.finalize(device=device)


def _step_once_joint_qd_convention_upstream(model, solver):
    """Step ``solver`` once from the seeded public free-joint twist."""
    state_in, state_out = model.state(), model.state()
    joint_qd = state_in.joint_qd.numpy()
    joint_qd[0:3] = V_COM
    joint_qd[3:6] = OMEGA_joint_qd_convention_upstream
    state_in.joint_qd.assign(joint_qd)
    solver.step(state_in, state_out, model.control(), None, DT)
    return state_out


def test_joint_qd_matches_featherstone_far_from_origin(test, device, solve_mode="matrix_free"):
    """Match SolverFeatherstone's joint_qd and body_qd from the same public twist, far from the origin."""
    for origin in ORIGINS:
        with test.subTest(origin=origin):
            model = _build_free_body_joint_qd_convention_upstream(origin, device)
            reference = _step_once_joint_qd_convention_upstream(model, SolverFeatherstone(model))
            state_out = _step_once_joint_qd_convention_upstream(model, SolverFeatherPGS(model, solve_mode=solve_mode))
            # A convention mismatch would differ by omega x x_com_world, about 60 m/s at 20 m.
            np.testing.assert_allclose(state_out.joint_qd.numpy(), reference.joint_qd.numpy(), rtol=0.0, atol=1e-3)
            np.testing.assert_allclose(state_out.body_qd.numpy(), reference.body_qd.numpy(), rtol=0.0, atol=1e-3)


class TestFeatherPGSJointQdConventionUpstream(unittest.TestCase):
    pass


for _name, _func in (
    ("test_joint_qd_matches_featherstone_far_from_origin", test_joint_qd_matches_featherstone_far_from_origin),
):
    add_function_test(TestFeatherPGSJointQdConventionUpstream, _name, _func, devices=get_cuda_test_devices())
    add_function_test(
        TestFeatherPGSJointQdConventionUpstream, f"{_name}_split", _func, devices=get_test_devices(), solve_mode="split"
    )


def _build_two_world_free_model(device):
    template = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = template.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    joint = template.add_joint_free(parent=-1, child=body)
    template.add_articulation([joint])

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.replicate(template, 2)
    return builder.finalize(device=device)


def _make_solver(model, *, solve_mode: str, dense_warmstart: bool, mf_warmstart: bool):
    return SolverFeatherPGS(
        model,
        solve_mode=solve_mode,
        contact_warmstart=dense_warmstart or mf_warmstart,
        articulation_constraint_max=4,
        free_body_constraint_max=4,
    )


def _history_specs(solver):
    specs = []
    if solver._contact_warmstart:
        specs.append(("dense impulses", solver._impulses, 7.0, 0.0))
    if solver._ws_prev_mf_impulses is not None:
        specs.extend(
            (
                ("MF impulses", solver._ws_prev_mf_impulses, 11.0, 0.0),
                ("MF row types", solver._ws_prev_mf_row_type, 13, -1),
            )
        )
    return specs


def _poison(specs):
    for _name, array, poison, _cleared in specs:
        array.fill_(poison)


def _assert_worlds(test: unittest.TestCase, specs, reset_worlds: tuple[bool, bool]):
    for name, array, poison, cleared in specs:
        expected = np.full(array.shape, poison, dtype=array.numpy().dtype)
        for world, reset in enumerate(reset_worlds):
            if reset:
                expected[world] = cleared
        np.testing.assert_array_equal(array.numpy(), expected, err_msg=name)


class TestFeatherPGSReset(unittest.TestCase):
    def test_model_changes_invalidate_impulses_without_clearing_overflow(self):
        """Invalidate changed-model impulses while retaining invalid-transition status."""
        model = _build_two_world_free_model("cpu")
        solver = _make_solver(model, solve_mode="split", dense_warmstart=True, mf_warmstart=True)
        for flags in (
            newton.ModelFlags.BODY_INERTIAL_PROPERTIES,
            newton.ModelFlags.JOINT_DOF_PROPERTIES,
            newton.ModelFlags.SHAPE_PROPERTIES,
        ):
            with self.subTest(flags=flags):
                specs = _history_specs(solver)
                _poison(specs)
                solver.constraint_overflow.fill_(True)
                solver.notify_model_changed(flags)
                _assert_worlds(self, specs, (True, True))
                self.assertTrue(solver.constraint_overflow.numpy().all())

    def test_model_change_world_mask_limits_impulse_invalidation(self):
        """Invalidate impulses only in the masked worlds, in either mask layout."""
        model = _build_two_world_free_model("cpu")
        solver = _make_solver(model, solve_mode="split", dense_warmstart=True, mf_warmstart=True)
        for mask in ((False, True), (False, True, False)):
            with self.subTest(mask=mask):
                specs = _history_specs(solver)
                _poison(specs)
                world_mask = wp.array(mask, dtype=wp.bool, device=model.device)
                solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES, world_mask=world_mask)
                _assert_worlds(self, specs, (False, True))

    def test_reset_clears_enabled_histories_by_scope(self):
        cases = (
            ("dense selected", "split", True, False, (True, False)),
            ("MF selected", "split", False, True, (True, False)),
            ("combined selected", "split", True, True, (True, False)),
            ("combined all", "split", True, True, (True, True)),
        )
        for name, solve_mode, dense_warmstart, mf_warmstart, reset_worlds in cases:
            with self.subTest(name=name):
                model = _build_two_world_free_model("cpu")
                solver = _make_solver(
                    model,
                    solve_mode=solve_mode,
                    dense_warmstart=dense_warmstart,
                    mf_warmstart=mf_warmstart,
                )
                specs = _history_specs(solver)
                _poison(specs)
                mask = None
                if reset_worlds != (True, True):
                    mask = wp.array(reset_worlds, dtype=wp.bool, device=model.device)
                solver.reset(model.state(), mask)
                _assert_worlds(self, specs, reset_worlds)

    def test_reset_validates_mask_before_disabled_noop(self):
        model = _build_two_world_free_model("cpu")
        solver = _make_solver(model, solve_mode="split", dense_warmstart=False, mf_warmstart=False)
        solver._impulses.fill_(17.0)
        before = solver._impulses.numpy().copy()

        invalid_mask = wp.ones(4, dtype=wp.bool, device=model.device)
        with self.assertRaisesRegex(ValueError, "expected 2"):
            solver.reset(model.state(), invalid_mask)

        valid_mask = wp.array((True, False), dtype=wp.bool, device=model.device)
        solver.reset(model.state(), valid_mask)
        np.testing.assert_array_equal(solver._impulses.numpy(), before)
        self.assertIsNone(solver._ws_prev_mf_impulses)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA graph reset test requires CUDA")
    def test_reset_is_cuda_graph_capturable(self):
        device = wp.get_device("cuda:0")
        model = _build_two_world_free_model(device)
        solver = _make_solver(model, solve_mode="split", dense_warmstart=True, mf_warmstart=True)
        state = model.state()
        mask = wp.array((True, False), dtype=wp.bool, device=device)
        specs = _history_specs(solver)

        _poison(specs)
        solver.reset(state, mask)
        wp.synchronize_device(device)
        with wp.ScopedCapture(device=device) as capture:
            solver.reset(state, mask)

        _poison(specs)
        wp.capture_launch(capture.graph)
        _assert_worlds(self, specs, (True, False))


PLATFORM_TOP = 0.5


BOX_HALF = 0.1


def _stack_on_platform(device, num_worlds=2):
    """Two free boxes stacked on a position-driven platform slider, without a ground plane.

    The lower box couples the platform's dense rows to the box-box free-body rows, so the
    stack only rests if the two row families see each other's impulses.
    """
    world = newton.ModelBuilder()
    world.default_shape_cfg.mu = 0.5
    platform = world.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, PLATFORM_TOP - 0.05), wp.quat_identity()))
    world.add_shape_box(platform, hx=0.4, hy=0.4, hz=0.05)
    slider = world.add_joint_prismatic(
        -1,
        platform,
        axis=newton.Axis.Z,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, PLATFORM_TOP - 0.05), wp.quat_identity()),
        target_ke=5.0e4,
        target_kd=5.0e2,
    )
    world.add_articulation([slider])
    for level in range(2):
        z = PLATFORM_TOP + BOX_HALF * (2 * level + 1)
        body = world.add_body(xform=wp.transform(wp.vec3(0.02 * level, 0.0, z), wp.quat_identity()))
        world.add_shape_box(body, hx=BOX_HALF, hy=BOX_HALF, hz=BOX_HALF)
    builder = newton.ModelBuilder()
    builder.replicate(world, num_worlds, spacing=(2.0, 0.0, 0.0))
    return builder.finalize(device=device)


def _run_split(model, solver, steps):
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    for _ in range(steps):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0
    return state_0


def test_mixed_world_stack_rests(test, device):
    """Rest a box stack on an articulated platform: dense and free-body rows exchange impulses every sweep."""
    model = _stack_on_platform(device)
    solver = SolverFeatherPGS(model, solve_mode="split", iterations=16)
    test.assertTrue(solver._has_mixed_contacts)
    state = _run_split(model, solver, 240)
    solver.check_constraint_capacity()
    test.assertGreater(int(solver._constraint_count.numpy()[0]), 0)
    test.assertGreater(int(solver._mf_constraint_count.numpy()[0]), 0)
    # The drive spring sags under the load; the boxes rest on the platform and on each other.
    body_q = state.body_q.numpy().reshape(model.world_count, 3, 7)
    platform_top = body_q[:, 0, 2] + 0.05
    for level in range(2):
        expected = platform_top + BOX_HALF * (2 * level + 1)
        np.testing.assert_allclose(body_q[:, level + 1, 2], expected, atol=2.0e-3, err_msg=f"box {level}")
    test.assertLess(float(np.abs(state.body_qd.numpy()).max()), 2.0e-2)


def test_mixed_world_matches_matrix_free_solve(test, device):
    """Converge to the matrix-free solve's resting state on the same mixed scene."""
    final = {}
    for solve_mode in ("split", "matrix_free"):
        model = _stack_on_platform(device)
        solver = SolverFeatherPGS(model, solve_mode=solve_mode, iterations=32)
        final[solve_mode] = _run_split(model, solver, 120).body_q.numpy()
    np.testing.assert_allclose(final["split"], final["matrix_free"], rtol=0.0, atol=2.0e-3)


def test_captured_mixed_step_matches_eager(test, device):
    """Replay captured collide + split step on a mixed scene like eager stepping."""
    trajectories = []
    for capture in (False, True):
        model = _stack_on_platform(device)
        solver = SolverFeatherPGS(model, solve_mode="split")
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        state_0, state_1 = model.state(), model.state()
        control = model.control()

        def substep(
            solver=solver, pipeline=pipeline, contacts=contacts, state_0=state_0, state_1=state_1, control=control
        ):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
                wp.copy(getattr(state_0, name), getattr(state_1, name))

        substep()
        if capture:
            with wp.ScopedCapture(device=device) as graph:
                substep()
            for _ in range(59):
                wp.capture_launch(graph.graph)
        else:
            for _ in range(59):
                substep()
        trajectories.append(state_0.body_q.numpy())
    # The right-hand side accumulates the contributions of several articulations with atomics,
    # so two runs agree only to accumulated round-off (about 1e-5 here).
    np.testing.assert_allclose(trajectories[1][:, :3], trajectories[0][:, :3], rtol=0.0, atol=1.0e-4)
    np.testing.assert_allclose(trajectories[1][:, 3:], trajectories[0][:, 3:], rtol=0.0, atol=5.0e-4)


class TestFeatherPGSSplit(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "requires a CUDA device to compare against")
    def test_cpu_and_cuda_split_solves_agree(self):
        """Match the CPU split solve (scalar kernels) and the CUDA split solve (native kernels)."""
        final = {}
        for device in ("cpu", "cuda:0"):
            model = _stack_on_platform(device)
            final[device] = _run_split(model, SolverFeatherPGS(model, solve_mode="split"), 120).body_q.numpy()
        np.testing.assert_allclose(final["cpu"][:, :3], final["cuda:0"][:, :3], rtol=0.0, atol=1.0e-4)
        np.testing.assert_allclose(final["cpu"][:, 3:], final["cuda:0"][:, 3:], rtol=0.0, atol=5.0e-4)


for _name, _func, _devices in (
    ("test_mixed_world_stack_rests", test_mixed_world_stack_rests, get_test_devices()),
    ("test_mixed_world_matches_matrix_free_solve", test_mixed_world_matches_matrix_free_solve, get_cuda_test_devices()),
    ("test_captured_mixed_step_matches_eager", test_captured_mixed_step_matches_eager, get_cuda_test_devices()),
):
    add_function_test(TestFeatherPGSSplit, _name, _func, devices=_devices)


class TestFeatherPGSBodyGravity(unittest.TestCase):
    def test_parallel_tree_selective_world_gravity(self):
        """Match analytic axis accelerations with enabled and disabled tree leaves in distinct worlds."""
        if not wp.get_device().is_cuda:
            self.skipTest("Parallel tree execution requires CUDA")
        builder = newton.ModelBuilder(gravity=(2.0, -3.0, 1.0))
        for gravity in ((0.0, 0.0, -10.0), (4.0, -5.0, 2.0)):
            builder.add_world(_gravity_tree(gravity))
        _add_free_body(builder, "global", False)
        model = builder.finalize(device=wp.get_device())
        expected = np.zeros((model.body_count, 3), dtype=np.float32)
        gravity = model.gravity.numpy()[model.body_world.numpy()]
        for body in range(model.body_count - 1):
            leaf = body % 4 - 1
            if leaf >= 0:
                expected[body, leaf] = gravity[body, leaf] * 0.08
        expected[-1] = gravity[-1] * 0.08
        expected[model.body_disable_gravity.numpy()] = 0.0
        for mode in ("split", "matrix_free"):
            for parallel in (False, True):
                with self.subTest(mode=mode, parallel_tree=parallel):
                    state = _advance(model, SolverFeatherPGS, {"solve_mode": mode, "parallel_tree": parallel}, steps=8)
                    np.testing.assert_allclose(state.body_qd.numpy()[:, :3], expected, atol=2.0e-6)
                    np.testing.assert_allclose(state.body_qd.numpy()[:, 3:], 0.0, atol=1.0e-7)

    def test_parallel_tree_mixed_com_wrench(self):
        """Match branched selective gravity to external forces at enabled centers of mass."""
        if not wp.get_device().is_cuda:
            self.skipTest("Parallel tree execution requires CUDA")
        gravity = np.array((2.0, -3.0, -10.0), dtype=np.float32)
        model = _gravity_tree(gravity, floating=True).finalize(device=wp.get_device())
        reference = _gravity_tree((0.0, 0.0, 0.0), floating=True).finalize(device=wp.get_device())
        forces = np.zeros((model.body_count, 6), dtype=np.float32)
        forces[:, :3] = model.body_mass.numpy()[:, None] * gravity
        forces[model.body_disable_gravity.numpy()] = 0.0
        for mode in ("split", "matrix_free"):
            serial = None
            for parallel in (False, True):
                with self.subTest(mode=mode, parallel_tree=parallel):
                    options = {"solve_mode": mode, "parallel_tree": parallel}
                    state = _advance(model, SolverFeatherPGS, options, steps=8)
                    control = _advance(reference, SolverFeatherPGS, options, steps=8, forces=forces)
                    for name in ("body_q", "body_qd"):
                        np.testing.assert_allclose(
                            getattr(state, name).numpy(), getattr(control, name).numpy(), atol=3.0e-6
                        )
                        if serial is not None:
                            np.testing.assert_allclose(
                                getattr(state, name).numpy(), getattr(serial, name).numpy(), atol=3.0e-6
                            )
                    self.assertGreater(float(np.linalg.norm(state.body_qd.numpy()[:, 3:])), 0.01)
                    serial = state

    def test_parallel_tree_runtime_flags_and_graph(self):
        """Refresh selective tree gravity and world vectors during notified graph replay."""
        device = wp.get_device()
        if not device.is_cuda:
            self.skipTest("Parallel tree graph capture requires CUDA")
        for mode in ("split", "matrix_free"):
            with self.subTest(mode=mode):
                builder = newton.ModelBuilder()
                for gravity in ((1.0, -2.0, -10.0), (4.0, -5.0, 2.0)):
                    builder.add_world(_gravity_tree(gravity))
                model = builder.finalize(device=device)
                solvers, states, graphs, outputs, controls = [], [], [], [], []
                for parallel in (False, True):
                    options = {"solve_mode": mode, "parallel_tree": parallel}
                    _advance(model, SolverFeatherPGS, options, steps=2)
                    solver = SolverFeatherPGS(model, **options)
                    solver._rigid_body_angular_damping.zero_()
                    self.assertEqual(solver._tree_plan is not None, parallel)
                    self.assertTrue(solver._fk_id_cache_enabled)
                    state, output = model.state(), model.state()
                    control = model.control()
                    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                    with wp.ScopedCapture(device) as capture:
                        state.clear_forces()
                        solver.step(state, output, control, None, 0.01)
                        output.clear_forces()
                        solver.step(output, state, control, None, 0.01)
                    solvers.append(solver)
                    states.append(state)
                    graphs.append(capture.graph)
                    # Keep captured state and control allocations alive for replay.
                    outputs.append(output)
                    controls.append(control)
                for phase in range(3):
                    if phase == 1:
                        flags = model.body_disable_gravity.numpy()
                        flags[5:7] = ~flags[5:7]
                        model.body_disable_gravity.assign(flags)
                        for solver in solvers:
                            solver.notify_model_changed(newton.ModelFlags.BODY_PROPERTIES)
                    elif phase == 2:
                        gravity = model.gravity.numpy()
                        gravity[1] = (3.0, 7.0, -5.0)
                        model.gravity.assign(gravity)
                        for solver in solvers:
                            solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
                    before = states[0].body_qd.numpy().copy()
                    for graph in graphs:
                        wp.capture_launch(graph)
                    delta = np.zeros((model.body_count, 3), dtype=np.float32)
                    gravity = model.gravity.numpy()[model.body_world.numpy()]
                    for body in range(model.body_count):
                        leaf = body % 4 - 1
                        if leaf >= 0:
                            delta[body, leaf] = gravity[body, leaf] * 0.02
                    delta[model.body_disable_gravity.numpy()] = 0.0
                    for state in states:
                        np.testing.assert_allclose(state.body_qd.numpy()[:, :3] - before[:, :3], delta, atol=2.0e-6)
                    for name in ("body_q", "body_qd"):
                        np.testing.assert_allclose(
                            getattr(states[1], name).numpy(), getattr(states[0], name).numpy(), atol=3.0e-6
                        )

    @unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
    def test_usd_selective_gravity(self):
        """Keep a gravity-disabled body stationary while its neighbor falls."""
        # USD is an optional importer dependency.
        from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        scene = UsdPhysics.Scene.Define(stage, "/physicsScene")
        scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        scene.CreateGravityMagnitudeAttr().Set(10.0)
        for index, disabled in enumerate((True, False, None)):
            cube = UsdGeom.Cube.Define(stage, f"/body_{index}")
            cube.CreateSizeAttr().Set(0.1)
            cube.AddTranslateOp().Set(Gf.Vec3d(index, 0.0, 2.0))
            prim = cube.GetPrim()
            UsdPhysics.RigidBodyAPI.Apply(prim)
            UsdPhysics.CollisionAPI.Apply(prim)
            UsdPhysics.MassAPI.Apply(prim).CreateMassAttr().Set(1.0)
            if disabled is not None:
                prim.CreateAttribute("physxRigidBody:disableGravity", Sdf.ValueTypeNames.Bool).Set(disabled)
        builder = newton.ModelBuilder()
        SolverFeatherPGS.register_custom_attributes(builder)
        builder.add_usd(stage, schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()])
        self.assertEqual(builder.body_disable_gravity, [True, False, False])
        model = builder.finalize(device=wp.get_device())
        solver = SolverFeatherPGS(model)
        solver._rigid_body_angular_damping.zero_()
        state, output = model.state(), model.state()
        control = model.control()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        initial = state.body_q.numpy().copy()
        for _ in range(8):
            state.clear_forces()
            solver.step(state, output, control, None, 0.01)
            state, output = output, state
        frozen = model.body_label.index("/body_0")
        falling = model.body_label.index("/body_1")
        np.testing.assert_allclose(state.body_q.numpy()[frozen], initial[frozen], atol=1.0e-7)
        self.assertAlmostEqual(float(state.body_qd.numpy()[falling, 2]), -0.8, places=5)
        self.assertLess(float(state.body_q.numpy()[falling, 2]), float(initial[falling, 2]) - 0.02)
        absent = model.body_label.index("/body_2")
        np.testing.assert_allclose(state.body_qd.numpy()[absent], state.body_qd.numpy()[falling], atol=1.0e-7)
        UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath("/body_0"))
        for parent, child in ((0, 1), (1, 2)):
            joint = UsdPhysics.FixedJoint.Define(stage, f"/joint_{child}")
            joint.CreateBody0Rel().SetTargets([f"/body_{parent}"])
            joint.CreateBody1Rel().SetTargets([f"/body_{child}"])
        imported = newton.ModelBuilder()
        imported.add_usd(
            stage, collapse_fixed_joints=True, schema_resolvers=[SchemaResolverNewton(), SchemaResolverPhysx()]
        )
        self.assertEqual(imported.body_disable_gravity, [True, False])
        clones = newton.ModelBuilder()
        clones.replicate(imported, 4)
        np.testing.assert_array_equal(clones.finalize(device="cpu").body_disable_gravity.numpy(), [True, False] * 4)

    def test_world_gravity_and_replication(self):
        """Preserve selective gravity across distinct local worlds and global bodies."""
        builder = newton.ModelBuilder(gravity=(2.0, -3.0, 1.0))
        for gravity in ((0.0, 0.0, -10.0), (4.0, -5.0, 2.0)):
            source = _free_pair(gravity)
            builder.add_world(source)
        _add_free_body(builder, "global", False)
        self.assertEqual(builder.body_disable_gravity, [True, False, True, False, False])
        model = builder.finalize(device=wp.get_device())
        expected = model.gravity.numpy()[model.body_world.numpy()] * 0.08
        expected[model.body_disable_gravity.numpy()] = 0.0
        initial = model.state()
        newton.eval_fk(model, initial.joint_q, initial.joint_qd, initial)
        gravity_force = wp.zeros(model.joint_dof_count, dtype=float, device=model.device)
        newton.eval_inverse_dynamics_passive(model, initial, gravity_force=gravity_force)
        np.testing.assert_allclose(gravity_force.numpy().reshape(-1, 6)[:, :3], -expected / 0.08, atol=1.0e-6)
        for solver_type, options in _solver_options():
            with self.subTest(solver=solver_type.__name__, options=options):
                state = _advance(model, solver_type, options, steps=8)
                np.testing.assert_allclose(state.body_qd.numpy()[:, :3], expected, atol=2.0e-6)
                np.testing.assert_allclose(state.body_qd.numpy()[:, 3:], 0.0, atol=1.0e-7)
        replicas = newton.ModelBuilder()
        replicas.replicate(_free_pair(), 4)
        self.assertEqual(replicas.body_disable_gravity, [True, False] * 4)
        copied = newton.ModelBuilder()
        copied.add_builder(replicas)
        np.testing.assert_array_equal(copied.finalize(device="cpu").body_disable_gravity.numpy(), [True, False] * 4)

    def test_mixed_articulation_gravity_wrench(self):
        """Match selective gravity to an equivalent external force at the enabled COM."""
        gravity = np.array((2.0, -3.0, -10.0), dtype=np.float32)
        model = _mixed_articulation(gravity).finalize(device=wp.get_device())
        reference = _mixed_articulation((0.0, 0.0, 0.0)).finalize(device=wp.get_device())
        forces = np.zeros((2, 6), dtype=np.float32)
        forces[1, :3] = reference.body_mass.numpy()[1] * gravity
        for solver_type, options in _solver_options():
            with self.subTest(solver=solver_type.__name__, options=options):
                state = _advance(model, solver_type, options, steps=8)
                control = _advance(reference, solver_type, options, steps=8, forces=forces)
                np.testing.assert_allclose(state.body_q.numpy(), control.body_q.numpy(), atol=3.0e-6)
                np.testing.assert_allclose(state.body_qd.numpy(), control.body_qd.numpy(), atol=3.0e-6)
                self.assertGreater(float(np.linalg.norm(state.body_qd.numpy()[:, 3:])), 0.01)

    def test_collapse_preserves_mixed_gravity(self):
        """Retain mixed-gravity fixed joints and remap flags after equal-gravity merges."""
        for flags in ((False, False, True, True, False), (True, True, False, True, True)):
            builder = newton.ModelBuilder()
            world_fixed = builder.add_link(mass=1.0, label="world_fixed", disable_gravity=True)
            builder.add_articulation([builder.add_joint_fixed(-1, world_fixed)])
            joints = []
            parent = -1
            for index, disabled in enumerate(flags):
                body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), disable_gravity=disabled)
                joints.append(builder.add_joint_free(body) if index == 0 else builder.add_joint_fixed(parent, body))
                parent = body
            builder.add_articulation(joints)
            builder.collapse_fixed_joints()
            expected = [flags[0], *[right for left, right in pairwise(flags) if left != right]]
            self.assertEqual(builder.body_disable_gravity, expected)
            self.assertEqual(builder.body_count, len(expected))
            np.testing.assert_array_equal(builder.finalize(device="cpu").body_disable_gravity.numpy(), expected)

    def test_runtime_flags_and_graph(self):
        """Refresh selective gravity after notification during cached graph replay."""
        device = wp.get_device()
        if not device.is_cuda:
            self.skipTest("CUDA graph capture requires CUDA")
        for mode in ("split", "matrix_free"):
            for notify in (False, True):
                with self.subTest(mode=mode, notify=notify):
                    builder = newton.ModelBuilder()
                    builder.replicate(_free_pair(), 4)
                    model = builder.finalize(device=device)
                    _advance(model, SolverFeatherPGS, {"solve_mode": mode}, steps=2)
                    solver = SolverFeatherPGS(model, solve_mode=mode)
                    solver._rigid_body_angular_damping.zero_()
                    state, output = model.state(), model.state()
                    control = model.control()
                    newton.eval_fk(model, state.joint_q, state.joint_qd, state)

                    self.assertTrue(solver._fk_id_cache_enabled)
                    with wp.ScopedCapture(device) as capture:
                        state.clear_forces()
                        solver.step(state, output, control, None, 0.01)
                        output.clear_forces()
                        solver.step(output, state, control, None, 0.01)
                    self.assertIs(solver._fk_id_cache_source_state, state)
                    for _ in range(4):
                        wp.capture_launch(capture.graph)
                    expected = np.array([0.0, -0.8] * 4, dtype=np.float32)
                    np.testing.assert_allclose(state.body_qd.numpy()[:, 2], expected, atol=2.0e-6)
                    old_flags = model.body_disable_gravity.numpy()
                    flags = old_flags.copy()
                    flags[2:4] = [False, True]
                    model.body_disable_gravity.assign(flags)
                    np.testing.assert_array_equal(solver._fk_id_cache_valid.numpy(), 1)
                    if notify:
                        solver.notify_model_changed(newton.ModelFlags.BODY_PROPERTIES)
                        np.testing.assert_array_equal(solver._fk_id_cache_valid.numpy(), 0)
                    wp.capture_launch(capture.graph)
                    expected += np.where(flags, 0.0, -0.1)
                    expected += np.where(flags if notify else old_flags, 0.0, -0.1)
                    np.testing.assert_allclose(state.body_qd.numpy()[:, 2], expected, atol=2.0e-6)
                    old_gravity = model.gravity.numpy()
                    gravity = old_gravity.copy()
                    gravity[2] = [3.0, 0.0, -5.0]
                    model.gravity.assign(gravity)
                    if notify:
                        solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
                    before = state.body_qd.numpy().copy()
                    wp.capture_launch(capture.graph)
                    world = model.body_world.numpy()
                    delta = (gravity[world] + (gravity if notify else old_gravity)[world]) * 0.01
                    delta[flags] = 0.0
                    np.testing.assert_allclose(state.body_qd.numpy()[:, :3] - before[:, :3], delta, atol=2.0e-6)


def _add_free_body(builder, label, disabled):
    """Add an isolated body with explicit inertial properties."""
    return builder.add_body(
        label=label,
        mass=1.0,
        inertia=wp.mat33(np.eye(3)),
        disable_gravity=disabled,
        xform=wp.transform((0.0, 0.0, 2.0), wp.quat_identity()),
    )


def _free_pair(gravity=(0.0, 0.0, -10.0)):
    """Build one disabled and one enabled free body."""
    builder = newton.ModelBuilder(gravity=gravity)
    _add_free_body(builder, "disabled", True)
    _add_free_body(builder, "enabled", False)
    return builder


def _mixed_articulation(gravity):
    """Build a floating fixed pair with separated centers of mass."""
    builder = newton.ModelBuilder(gravity=gravity)
    root = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), disable_gravity=True)
    child = builder.add_link(mass=2.0, inertia=wp.mat33(np.eye(3)), com=(0.3, 0.2, -0.1))
    joints = [builder.add_joint_free(root)]
    joints.append(builder.add_joint_fixed(root, child, parent_xform=wp.transform((1.0, 0.0, 0.0), wp.quat_identity())))
    builder.add_articulation(joints)
    return builder


def _solver_options():
    """Exercise serial and cached dynamics in both Feather solvers."""
    choices = [(SolverFeatherstone, {}), (SolverFeatherPGS, {"solve_mode": "split"})]
    if wp.get_device().is_cuda:
        choices.append((SolverFeatherPGS, {"solve_mode": "matrix_free"}))
    return choices


def _advance(model, solver_type, options, *, steps, forces=None):
    """Step an isolated model with an optional fixed COM wrench."""
    if solver_type is SolverFeatherPGS:
        solver = solver_type(model, **options)
        solver._rigid_body_angular_damping.zero_()
    else:
        solver = solver_type(model, angular_damping=0.0, **options)
    if options.get("parallel_tree"):
        assert solver._tree_plan is not None
        assert any(group.lanes > 1 for group in solver._tree_plan.groups)
    state, output = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    for _ in range(steps):
        state.clear_forces()
        if forces is not None:
            state.body_f.assign(forces)
        solver.step(state, output, control, None, 0.01)
        state, output = output, state
    return state


def _gravity_tree(gravity, *, floating=False):
    """Build three independently scheduled leaves with alternating gravity flags."""
    builder = newton.ModelBuilder(gravity=gravity)
    root = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), disable_gravity=True)
    joints = [builder.add_joint_free(root) if floating else builder.add_joint_fixed(-1, root)]
    for index, axis in enumerate((newton.Axis.X, newton.Axis.Y, newton.Axis.Z)):
        child = builder.add_link(
            mass=2.0 + index,
            inertia=wp.mat33(np.eye(3)),
            com=(0.3, 0.2, -0.1) if floating else (0.0, 0.0, 0.0),
            disable_gravity=index == 1,
        )
        anchor = wp.transform((1.0, 0.4 * index, 0.1 * index), wp.quat_identity())
        if floating:
            joint = builder.add_joint_fixed(root, child, parent_xform=anchor)
        else:
            joint = builder.add_joint_prismatic(root, child, axis=axis, parent_xform=anchor)
        joints.append(joint)
    builder.add_articulation(joints)
    return builder


DT_parent_wrench = 1.0 / 120.0
FLAGS = {SolverObservableFlags.BODY_PARENT_F}


def _parent_wrench_scene(device):
    """Two free boxes resting on the ground, each carrying a smaller box welded on top."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    for x in (0.0, 0.5):
        base = builder.add_link(xform=wp.transform(wp.vec3(x, 0.0, 0.1), wp.quat_identity()))
        builder.add_shape_box(base, hx=0.1, hy=0.1, hz=0.1)
        top = builder.add_link(xform=wp.transform(wp.vec3(x, 0.0, 0.25), wp.quat_identity()))
        builder.add_shape_box(top, hx=0.05, hy=0.05, hz=0.05)
        free = builder.add_joint_free(base)
        weld = builder.add_joint_fixed(
            base, top, parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.15), wp.quat_identity())
        )
        builder.add_articulation([free, weld])
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=64)
    return model, pipeline


def _assert_resting_wrench(test, model, wrench):
    """Free roots transmit nothing; each weld carries its top box's weight."""
    weight = model.body_mass.numpy() * 9.81
    np.testing.assert_allclose(wrench[0::2], 0.0, atol=1.0e-3)
    np.testing.assert_allclose(wrench[1::2, 2], weight[1::2], rtol=1.0e-4)
    np.testing.assert_allclose(wrench[1::2, [0, 1, 3, 4, 5]], 0.0, atol=1.0e-3)


def _advance_parent_wrench(pipeline, solver, states, control, steps, *, observables=None):
    contacts = pipeline.contacts()
    for _ in range(steps):
        states[0].clear_forces()
        pipeline.collide(states[0], contacts)
        solver.step(states[0], states[1], control, contacts, DT_parent_wrench, observables=observables)
        states.reverse()


def _sleeping_wrench_solvers(model):
    solver = SolverFeatherPGS(
        model,
        solve_mode="matrix_free",
        enable_sleeping=True,
        sleep_quiet_time=0.05,
        friction_anchor_beta=0.0,
        articulation_constraint_max=128,
    )
    reference = SolverFeatherPGS(
        model, solve_mode="matrix_free", friction_anchor_beta=0.0, articulation_constraint_max=128
    )
    return solver, reference


def _resting_wrench(model, pipeline, reference, control):
    observables = reference.observables(FLAGS)
    _advance_parent_wrench(pipeline, reference, [model.state(), model.state()], control, 130, observables=observables)
    return observables.body_parent_f.numpy()


def test_sleeping_body_keeps_its_parent_wrench(test, device):
    """Publish the last awake joint wrench of a sleeping body instead of a skipped dynamics result."""
    model, pipeline = _parent_wrench_scene(device)
    solver, reference = _sleeping_wrench_solvers(model)
    observables = solver.observables(FLAGS)
    states, control = [model.state(), model.state()], model.control()
    _advance_parent_wrench(pipeline, solver, states, control, 120, observables=observables)
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0, 0, 0])
    asleep = observables.body_parent_f.numpy().copy()
    _advance_parent_wrench(pipeline, solver, states, control, 10, observables=observables)
    np.testing.assert_array_equal(observables.body_parent_f.numpy(), asleep)
    # The frozen value is the resting value an always-awake solver publishes.
    np.testing.assert_allclose(asleep, _resting_wrench(model, pipeline, reference, control), rtol=0.0, atol=1.0e-3)
    _assert_resting_wrench(test, model, asleep)


def test_sleeping_body_keeps_its_legacy_parent_wrench(test, device):
    """Freeze the deprecated State.body_parent_f output of a sleeping body as well."""
    model, pipeline = _parent_wrench_scene(device)
    with test.assertWarns(DeprecationWarning):
        model.request_state_attributes("body_parent_f")
    solver, _reference = _sleeping_wrench_solvers(model)
    states, control = [model.state(), model.state()], model.control()
    test.assertIsNotNone(states[0].body_parent_f)
    _advance_parent_wrench(pipeline, solver, states, control, 120)
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0, 0, 0])
    asleep = states[0].body_parent_f.numpy().copy()
    _assert_resting_wrench(test, model, asleep)
    _advance_parent_wrench(pipeline, solver, states, control, 10)
    np.testing.assert_array_equal(states[0].body_parent_f.numpy(), asleep)
    np.testing.assert_array_equal(states[1].body_parent_f.numpy(), asleep)


def test_sleeping_parent_wrench_is_independent_of_the_output(test, device):
    """Report the frozen wrench in a newly allocated or cleared output, in eager and captured steps."""
    model, pipeline = _parent_wrench_scene(device)
    solver, reference = _sleeping_wrench_solvers(model)
    first = solver.observables(FLAGS)
    states, control = [model.state(), model.state()], model.control()
    _advance_parent_wrench(pipeline, solver, states, control, 120, observables=first)
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0, 0, 0])
    asleep = first.body_parent_f.numpy().copy()
    np.testing.assert_allclose(asleep, _resting_wrench(model, pipeline, reference, control), rtol=0.0, atol=1.0e-3)
    _assert_resting_wrench(test, model, asleep)

    second = solver.observables(FLAGS)
    _advance_parent_wrench(pipeline, solver, states, control, 1, observables=second)
    np.testing.assert_array_equal(second.body_parent_f.numpy(), asleep)
    first.body_parent_f.fill_(wp.spatial_vector(-1.0))
    _advance_parent_wrench(pipeline, solver, states, control, 1, observables=first)
    np.testing.assert_array_equal(first.body_parent_f.numpy(), asleep)

    contacts = pipeline.contacts()

    def step(observables):
        states[0].clear_forces()
        pipeline.collide(states[0], contacts)
        solver.step(states[0], states[1], control, contacts, DT_parent_wrench, observables=observables)
        pipeline.collide(states[1], contacts)
        solver.step(states[1], states[0], control, contacts, DT_parent_wrench, observables=observables)

    step(second)
    third = solver.observables(FLAGS)
    with wp.ScopedCapture(device) as capture:
        step(third)
    for _ in range(3):
        wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0, 0, 0])
    np.testing.assert_array_equal(third.body_parent_f.numpy(), asleep)


def test_parent_wrench_first_requested_after_sleep(test, device):
    """Report the resting wrench when the first output is requested after the bodies fell asleep."""
    model, pipeline = _parent_wrench_scene(device)
    solver, reference = _sleeping_wrench_solvers(model)
    states, control = [model.state(), model.state()], model.control()
    _advance_parent_wrench(pipeline, solver, states, control, 120)
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0, 0, 0])
    observables = solver.observables(FLAGS)
    _advance_parent_wrench(pipeline, solver, states, control, 1, observables=observables)
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0, 0, 0])
    resting = _resting_wrench(model, pipeline, reference, control)
    np.testing.assert_allclose(observables.body_parent_f.numpy(), resting, rtol=0.0, atol=1.0e-3)
    # A reset wakes the bodies, which then publish freshly computed wrenches.
    solver.reset(states[0])
    _advance_parent_wrench(pipeline, solver, states, control, 1, observables=observables)
    np.testing.assert_allclose(observables.body_parent_f.numpy(), resting, rtol=0.0, atol=1.0e-3)


def test_sleeping_parent_wrench_from_legacy_to_observable(test, device):
    """Report the same frozen wrench in an observable after sleeping with only the legacy output."""
    model, pipeline = _parent_wrench_scene(device)
    with test.assertWarns(DeprecationWarning):
        model.request_state_attributes("body_parent_f")
    solver, _reference = _sleeping_wrench_solvers(model)
    states, control = [model.state(), model.state()], model.control()
    _advance_parent_wrench(pipeline, solver, states, control, 120)
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0, 0, 0])
    legacy = states[0].body_parent_f.numpy().copy()
    observables = solver.observables(FLAGS)
    _advance_parent_wrench(pipeline, solver, states, control, 1, observables=observables)
    np.testing.assert_array_equal(observables.body_parent_f.numpy(), legacy)
    np.testing.assert_array_equal(states[0].body_parent_f.numpy(), legacy)


def _branched_chains(device, world_count=2):
    """Two-branch revolute trees on a fixed base plus a free box, with gravity, revolute joint forces and velocities."""
    template = newton.ModelBuilder()
    root = template.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.01))
    template.add_shape_box(root, hx=0.05, hy=0.05, hz=0.05)
    joints = [template.add_joint_fixed(parent=-1, child=root)]
    for side in (-1.0, 1.0):
        parent = root
        for level in range(3):
            child = template.add_link(mass=0.5 + 0.1 * level, inertia=wp.mat33(np.eye(3) * 0.005))
            template.add_shape_capsule(child, radius=0.02, half_height=0.08)
            axis = (0.0, 1.0, 0.0) if level % 2 == 0 else (1.0, 0.0, 0.0)
            joints.append(
                template.add_joint_revolute(
                    parent=parent,
                    child=child,
                    axis=axis,
                    parent_xform=wp.transform(wp.vec3(side * 0.1, 0.0, -0.05 - 0.15 * level), wp.quat_identity()),
                    child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.08), wp.quat_identity()),
                )
            )
            parent = child
    template.add_articulation(joints)
    free = template.add_body(xform=wp.transform(wp.vec3(1.0, 0.0, 2.0), wp.quat_identity()))
    template.add_shape_box(free, hx=0.1, hy=0.1, hz=0.1)
    builder = newton.ModelBuilder()
    for _ in range(world_count):
        builder.add_world(template)
    model = builder.finalize(device=device)
    state = model.state()
    rng = np.random.default_rng(3)
    joint_q = model.joint_q.numpy()
    joint_qd = model.joint_qd.numpy()
    revolute = model.joint_type.numpy() == int(newton.JointType.REVOLUTE)
    q_start = model.joint_q_start.numpy()
    qd_start = model.joint_qd_start.numpy()
    joint_f = np.zeros(model.joint_dof_count, dtype=np.float32)
    for joint in np.flatnonzero(revolute):
        joint_q[q_start[joint]] = rng.uniform(-0.6, 0.6)
        joint_qd[qd_start[joint]] = rng.uniform(-1.0, 1.0)
        # Free-joint forces enter Featherstone's and FeatherPGS's root wrench differently.
        joint_f[qd_start[joint]] = rng.uniform(-0.5, 0.5)
    state.joint_q.assign(joint_q)
    state.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    control = model.control()
    control.joint_f.assign(joint_f)
    return model, state, control


def _first_step(model, state, control, solver):
    observables = solver.observables(FLAGS)
    state_out = model.state()
    solver.step(state, state_out, control, None, DT_parent_wrench, observables=observables)
    return observables.body_parent_f.numpy(), state_out


def _revolute_wrench_torques(model, state, wrench):
    """Project each revolute joint's published wrench onto its axis about its anchor."""
    body_q = state.body_q.numpy()
    body_com = model.body_com.numpy()
    joint_X_c = model.joint_X_c.numpy()
    joint_axis = model.joint_axis.numpy()
    torques = {}
    for joint in np.flatnonzero(model.joint_type.numpy() == int(newton.JointType.REVOLUTE)):
        child = model.joint_child.numpy()[joint]
        X_wc = wp.transform(*body_q[child])
        X_wj = X_wc * wp.transform(*joint_X_c[joint])
        anchor = np.array(wp.transform_get_translation(X_wj))
        axis = np.array(
            wp.quat_rotate(wp.transform_get_rotation(X_wj), wp.vec3(*joint_axis[model.joint_qd_start.numpy()[joint]]))
        )
        com = np.array(wp.transform_point(X_wc, wp.vec3(*body_com[child])))
        force, torque_com = wrench[child, :3], wrench[child, 3:]
        torques[joint] = float(np.dot(axis, torque_com + np.cross(com - anchor, force)))
    return torques


def test_parent_wrench_matches_inverse_dynamics_on_every_path(test, device):
    """Publish the joint wrench of the solved motion, whichever inverse-dynamics path a configuration selects.

    Every revolute wrench projects onto its axis as exactly the applied joint torque, and a free body in flight
    transmits nothing.
    """
    model, state, control = _branched_chains(device)
    configurations = [{"solve_mode": "split"}]
    if wp.get_device(device).is_cuda:
        configurations += [
            {"solve_mode": "matrix_free"},
            {"solve_mode": "matrix_free", "parallel_tree": True},
            {"solve_mode": "matrix_free", "use_parallel_streams": False},
            {"solve_mode": "matrix_free", "double_buffer": False},
        ]
    joint_f = control.joint_f.numpy()
    qd_start = model.joint_qd_start.numpy()
    free_bodies = model.joint_child.numpy()[model.joint_type.numpy() == int(newton.JointType.FREE)]
    reference = None
    for options in configurations:
        with test.subTest(**options):
            solver = SolverFeatherPGS(model, **options)
            solver._rigid_body_angular_damping.zero_()
            wrench, _ = _first_step(model, state, control, solver)
            test.assertGreater(np.abs(wrench).max(), 1.0)
            for joint, torque in _revolute_wrench_torques(model, state, wrench).items():
                test.assertAlmostEqual(torque, joint_f[qd_start[joint]], delta=1.0e-3)
            np.testing.assert_allclose(wrench[free_bodies], 0.0, atol=1.0e-3)
            if reference is None:
                reference = wrench
            np.testing.assert_allclose(wrench, reference, rtol=1.0e-4, atol=1.0e-4)


G = 9.81
CONTACT_DT = 1.0 / 480.0
M_CARRIAGE = 1.0
M_GRIP = 0.5
M_FINGER = 0.1
M_BOX = 0.5


def _gripper_scene(device, *, lift_z=0.5, lift_drive=True, flange_drive=True, load_box_x=None, pinch=False):
    """A held tray gripper: a fixed prismatic lift, a revolute flange about z and a 0.2 m square tray.

    Optionally a free box rests on the tray at ``load_box_x``, or two prismatic fingers pinch a free box below it.
    The tray's bottom face is 0.13 m below the lift's origin.
    """
    builder = newton.ModelBuilder()
    shape_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=1.0)
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=1.0))
    carriage = builder.add_link(mass=M_CARRIAGE, inertia=wp.mat33(np.eye(3) * 0.01))
    ke, kd = (1.0e5, 1.0e3) if lift_drive else (0.0, 0.0)
    lift = builder.add_joint_prismatic(
        -1,
        carriage,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform((0.0, 0.0, lift_z), wp.quat_identity()),
        target_ke=ke,
        target_kd=kd,
    )
    grip = builder.add_link(
        mass=M_GRIP, com=(0.0, 0.0, -0.02), inertia=wp.mat33(np.diag([0.002, 0.002, 0.003])), lock_inertia=True
    )
    builder.add_shape_box(
        grip, xform=wp.transform((0.0, 0.0, -0.02), wp.quat_identity()), hx=0.1, hy=0.1, hz=0.01, cfg=shape_cfg
    )
    flange = builder.add_joint_revolute(
        carriage,
        grip,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform((0.0, 0.0, -0.1), wp.quat_identity()),
        target_ke=1.0e4 if flange_drive else 0.0,
        target_kd=1.0e2 if flange_drive else 0.0,
    )
    joints = [lift, flange]
    if pinch:
        for side in (-1.0, 1.0):
            finger = builder.add_link(mass=M_FINGER, inertia=wp.mat33(np.eye(3) * 1.0e-4))
            builder.add_shape_box(
                finger,
                xform=wp.transform((0.0, 0.0, -0.08), wp.quat_identity()),
                hx=0.01,
                hy=0.03,
                hz=0.05,
                cfg=shape_cfg,
            )
            joints.append(
                builder.add_joint_prismatic(
                    grip,
                    finger,
                    axis=(side, 0.0, 0.0),
                    parent_xform=wp.transform((side * 0.06, 0.0, -0.03), wp.quat_identity()),
                    target_pos=-0.03,
                    target_ke=2000.0,
                    target_kd=20.0,
                    limit_lower=-0.05,
                    limit_upper=0.05,
                )
            )
    builder.add_articulation(joints)
    box = None
    box_z = None
    if load_box_x is not None:
        box_x, box_z = load_box_x, lift_z - 0.07
    elif pinch:
        box_x, box_z = 0.0, lift_z - 0.21
    if box_z is not None:
        box = builder.add_body(
            xform=wp.transform((box_x, 0.0, box_z), wp.quat_identity()),
            mass=M_BOX,
            inertia=wp.mat33(np.eye(3) * M_BOX * 0.0032 / 6.0),
            lock_inertia=True,
        )
        builder.add_shape_box(box, hx=0.04, hy=0.04, hz=0.04, cfg=shape_cfg)
    model = builder.finalize(device=device)
    return model, {"grip": grip, "lift": lift, "flange": flange, "box": box}


_CONTACT_CONFIGURATIONS = {
    # The production profile: patch friction, shared anchors, spin rows and sleeping.
    "production": {
        "solve_mode": "matrix_free",
        "iterations": 32,
        "contact_regularization": 0.01,
        "contact_shared_anchor": True,
        "contact_friction_shared_anchor": True,
        "friction_anchor_beta": 0.2,
        "contact_friction_gap_max": 0.001,
        "enable_joint_velocity_limits": True,
        "torsion_radius": 0.01,
        "contact_torsion_device": True,
        "enable_sleeping": True,
        "articulation_constraint_max": 256,
        "free_body_constraint_max": 256,
    },
    "point_friction": {"solve_mode": "matrix_free", "friction_anchor_beta": 0.0, "articulation_constraint_max": 256},
    "split": {"solve_mode": "split", "articulation_constraint_max": 64},
    "propagation": {
        "solve_mode": "matrix_free",
        "articulated_contact_response": "propagation",
        "articulation_constraint_max": 256,
    },
    "velocity_pass": {"solve_mode": "matrix_free", "velocity_iterations": 4, "articulation_constraint_max": 256},
}


def _settled_wrench(model, ids, configuration, steps, *, joint_f=None, capture=False):
    """Average the published wrenches over the last 30 of ``steps`` steps."""
    options = dict(_CONTACT_CONFIGURATIONS[configuration])
    set_torsion_radius(model, options.pop("torsion_radius", 0.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        solver = SolverFeatherPGS(model, **options)
    solver._rigid_body_angular_damping.zero_()
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=256, contact_matching="sticky")
    observables = solver.observables(FLAGS)
    states, control = [model.state(), model.state()], model.control()
    if joint_f is not None:
        values = np.zeros(model.joint_dof_count, dtype=np.float32)
        for joint, value in joint_f.items():
            values[model.joint_qd_start.numpy()[ids[joint]]] = value
        control.joint_f.assign(values)
    contacts = pipeline.contacts()

    def step():
        states[0].clear_forces()
        pipeline.collide(states[0], contacts)
        solver.step(states[0], states[1], control, contacts, CONTACT_DT, observables=observables)
        states.reverse()

    samples = []
    for i in range(steps):
        if capture and i == steps - 31:
            # Two steps per graph keep the state ping-pong in place across replays.
            with wp.ScopedCapture(model.device) as recording:
                step()
                step()
            for _ in range(15):
                wp.capture_launch(recording.graph)
                solver.check_constraint_capacity()
                samples.append(observables.body_parent_f.numpy())
            break
        step()
        if i >= steps - 30:
            samples.append(observables.body_parent_f.numpy())
    return np.mean(samples, axis=0)


def test_parent_wrench_carries_a_load_resting_on_the_gripper(test, device):
    """A box resting off-centre on the held tray reaches the flange as weight plus its moment."""
    box_x = 0.03
    model, ids = _gripper_scene(device, load_box_x=box_x)
    for configuration in _CONTACT_CONFIGURATIONS:
        with test.subTest(configuration=configuration):
            wrench = _settled_wrench(model, ids, configuration, 480)
            expected = [0.0, 0.0, (M_GRIP + M_BOX) * G, 0.0, -box_x * M_BOX * G, 0.0]
            np.testing.assert_allclose(wrench[ids["grip"]], expected, atol=2.0e-3)
            np.testing.assert_allclose(wrench[ids["box"]], 0.0, atol=2.0e-3)


def test_parent_wrench_reports_a_commanded_push(test, device):
    """A lift force pressing the tray into the ground reaches the flange as that force plus the carriage weight."""
    push = 20.0
    model, ids = _gripper_scene(device, lift_z=0.1305, lift_drive=False)
    for configuration in _CONTACT_CONFIGURATIONS:
        with test.subTest(configuration=configuration):
            wrench = _settled_wrench(
                model, ids, configuration, 480, joint_f={"lift": -push}, capture=configuration == "production"
            )
            # The four tray corners share the load indeterminately, so only the force is fixed.
            np.testing.assert_allclose(wrench[ids["grip"], :3], [0.0, 0.0, -(push + M_CARRIAGE * G)], atol=2.0e-3)


def test_parent_wrench_carries_a_pinched_load(test, device):
    """Friction-held payload weight reaches the flange; the squeeze stays inside the gripper subtree."""
    model, ids = _gripper_scene(device, pinch=True)
    for configuration in _CONTACT_CONFIGURATIONS:
        with test.subTest(configuration=configuration):
            wrench = _settled_wrench(model, ids, configuration, 720)
            expected = [0.0, 0.0, (M_GRIP + 2.0 * M_FINGER + M_BOX) * G, 0.0, 0.0, 0.0]
            np.testing.assert_allclose(wrench[ids["grip"]], expected, atol=1.0e-2)
            np.testing.assert_allclose(wrench[ids["box"]], 0.0, atol=1.0e-2)


def test_parent_wrench_includes_spin_rows(test, device):
    """A box riding a spinning tray transmits nothing through its free joint once spin-row torques are counted."""
    model, ids = _gripper_scene(device, flange_drive=False, load_box_x=0.0)
    wrench = _settled_wrench(model, ids, "production", 240, joint_f={"flange": 0.3})
    np.testing.assert_allclose(wrench[ids["box"]], 0.0, atol=1.0e-3)
    test.assertAlmostEqual(wrench[ids["grip"], 5], 0.3, delta=1.0e-3)


def test_parent_wrench_excludes_velocity_limit_prescaling(test, device):
    """A disk spun past its velocity limit is rescaled before the dynamics; the rescale transmits no wrench."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    hub = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.01))
    hub_joint = builder.add_joint_revolute(-1, hub, axis=(0.0, 0.0, 1.0))
    disk = builder.add_link(mass=1.0, inertia=wp.mat33(np.diag([0.015, 0.015, 0.02])), lock_inertia=True)
    disk_joint = builder.add_joint_revolute(
        hub,
        disk,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform((0.3, 0.0, 0.0), wp.quat_identity()),
        velocity_limit=1.0,
    )
    builder.add_articulation([hub_joint, disk_joint])
    model = builder.finalize(device=device)
    solver = SolverFeatherPGS(model, solve_mode="matrix_free", enable_joint_velocity_limits=True)
    solver._rigid_body_angular_damping.zero_()
    observables = solver.observables(FLAGS)
    state_in, state_out = model.state(), model.state()
    state_in.joint_qd.assign(np.array([0.0, 10.0], dtype=np.float32))
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    solver.step(state_in, state_out, None, None, CONTACT_DT, observables=observables)
    test.assertAlmostEqual(float(state_out.joint_qd.numpy()[1]), 1.0, places=4)
    np.testing.assert_allclose(observables.body_parent_f.numpy(), 0.0, atol=1.0e-4)


def test_parent_wrench_includes_loop_closure_reactions(test, device):
    """Loop-closure forces reach both closure links: an undriven joint in a four-bar transmits no axis torque."""
    builder = _build_four_bar()
    model = builder.finalize(device=device)
    solver = SolverFeatherPGS(model, solve_mode="matrix_free", iterations=16, baumgarte=0.1)
    solver._rigid_body_angular_damping.zero_()
    observables = solver.observables(FLAGS)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    targets = model.joint_target_q.numpy().copy()
    targets[0] = 0.6
    control.joint_target_q.assign(targets)
    for step in range(120):
        state_in = state_0
        solver.step(state_0, state_1, control, None, 1.0 / 240.0, observables=observables)
        state_0, state_1 = state_1, state_0
        if step % 20 == 19:
            wrench = observables.body_parent_f.numpy()
            test.assertGreater(np.abs(wrench).max(), 0.5)
            torques = _revolute_wrench_torques(model, state_in, wrench)
            # Joint 0 is the driven crank; the coupler and rocker joints are undriven.
            for joint in (1, 2):
                test.assertAlmostEqual(torques[joint], 0.0, delta=2.0e-3)


def test_parent_wrench_of_a_load_held_by_closures(test, device):
    """A FREE load hung from the world or a moving kinematic carrier by closures transmits nothing through its free joint."""
    for world_parent in (True, False):
        with test.subTest(world_parent=world_parent):
            builder, carrier, load, _joints = _build_carried_load(world_parent=world_parent)
            model = builder.finalize(device=device)
            solver = SolverFeatherPGS(model, solve_mode="matrix_free", iterations=16, baumgarte=0.2)
            observables = solver.observables(FLAGS)
            state_0, state_1 = model.state(), model.state()
            control = model.control()
            articulation = None
            if not world_parent:
                articulation = wp.array(
                    [int(solver._body_to_articulation.numpy()[carrier])], dtype=wp.int32, device=device
                )
            dt = 1.0 / 240.0
            for i in range(120):
                if articulation is not None:
                    _prescribe_carrier(
                        model,
                        solver,
                        state_0,
                        articulation,
                        i * dt,
                        np.array([0.3, 0.0, 0.1]),
                        np.array([0.0, 0.0, 1.5]),
                    )
                state_0.clear_forces()
                solver.step(state_0, state_1, control, None, dt, observables=observables)
                state_0, state_1 = state_1, state_0
            np.testing.assert_allclose(observables.body_parent_f.numpy()[load], 0.0, atol=2.0e-3)


NEAR_REDUNDANT_LEVERS = (0.02, 0.001, 0.0002, 0.00015)


def test_closure_force_uses_the_row_builder_norm(test, device):
    """A closure row whose shared-ancestor terms cancel maps its impulse back with the builder's own norm."""

    def ints(values):
        return wp.array(values, dtype=wp.int32, device=device)

    for lever in NEAR_REDUNDANT_LEVERS:
        with test.subTest(lever=lever):
            zero = ints([0])
            anchor_xform = wp.array(
                [wp.transform((1.0, 0.0, 0.0), wp.quat_identity())], dtype=wp.transform, device=device
            )
            body_q = wp.array([wp.transform_identity()] * 2, dtype=wp.transform, device=device)
            # A shared root hinge plus a child hinge whose closure anchor sits `lever` from its axis.
            motion = wp.array(
                [[0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, -(1.0 - lever), 0.0, 0.0, 0.0, 1.0]],
                dtype=wp.spatial_vector,
                device=device,
            )
            body_to_joint, ancestor, qd_start = ints([0, 1]), ints([-1, 0]), ints([0, 1, 2])
            origin = wp.zeros(1, dtype=wp.vec3, device=device)
            J = wp.zeros((1, 3, 2), dtype=float, device=device)
            row_type, row_parent = ints([[0, 0, 0]]), ints([[0, 0, 0]])
            row_fields = [wp.zeros((1, 3), dtype=float, device=device) for _ in range(5)]
            inv_norm = wp.zeros((1, 3), dtype=float, device=device)
            wp.launch(
                populate_connect_J_for_size,
                dim=1,
                inputs=[
                    zero,
                    zero,
                    zero,
                    2,
                    zero,
                    zero,
                    zero,
                    zero,
                    ints([1]),
                    anchor_xform,
                    anchor_xform,
                    zero,
                    body_q,
                    wp.zeros(2, dtype=wp.spatial_vector, device=device),
                    wp.zeros(2, dtype=wp.vec3, device=device),
                    body_to_joint,
                    ancestor,
                    qd_start,
                    motion,
                    origin,
                    0.1,
                    0.0,
                ],
                outputs=[J, row_type, row_parent, *row_fields, inv_norm],
                device=device,
            )
            wrench = wp.zeros(2, dtype=wp.spatial_vector, device=device)
            impulses = wp.array([[0.0, 1.0, 0.0]], dtype=wp.float32, device=device)
            wp.launch(
                accumulate_connect_wrenches_from_impulses,
                dim=1,
                inputs=[
                    zero,
                    zero,
                    zero,
                    ints([1]),
                    zero,
                    anchor_xform,
                    anchor_xform,
                    inv_norm,
                    zero,
                    body_q,
                    ints([0, 0]),
                    origin,
                    ints([3]),
                    row_type,
                    impulses,
                    1.0,
                ],
                outputs=[wrench],
                device=device,
            )
            np.testing.assert_allclose(J.numpy()[0, 1], [0.0, -1.0], atol=1.0e-6)
            float32_lever = 1.0 + float(motion.numpy()[1, 1])
            force = wrench.numpy()[:, 1]
            np.testing.assert_allclose(force, [1.0 / float32_lever, -1.0 / float32_lever], rtol=1.0e-5)


def test_parent_wrench_of_a_near_redundant_closure(test, device):
    """A child hinge braced by a closure anchored just off its axis reports the torque applied to it."""
    torque = 0.01
    for lever in NEAR_REDUNDANT_LEVERS:
        with test.subTest(lever=lever):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            root = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
            root_joint = builder.add_joint_revolute(-1, root, axis=(0.0, 0.0, 1.0), armature=0.0)
            carrier = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
            carrier_joint = builder.add_joint_fixed(root, carrier)
            child = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
            child_joint = builder.add_joint_revolute(
                carrier,
                child,
                axis=(0.0, 0.0, 1.0),
                parent_xform=wp.transform((1.0 - lever, 0.0, 0.0), wp.quat_identity()),
                armature=0.0,
            )
            builder.add_articulation([root_joint, carrier_joint, child_joint])
            builder.add_joint_ball(
                root,
                child,
                parent_xform=wp.transform((1.0, 0.0, 0.0), wp.quat_identity()),
                child_xform=wp.transform((lever, 0.0, 0.0), wp.quat_identity()),
            )
            model = builder.finalize(device=device)
            solver = SolverFeatherPGS(model, solve_mode="split", iterations=64, cfm=0.0)
            solver._rigid_body_angular_damping.zero_()
            state = model.state()
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            control = model.control()
            joint_f = np.zeros(model.joint_dof_count, dtype=np.float32)
            joint_f[model.joint_qd_start.numpy()[child_joint]] = torque
            control.joint_f.assign(joint_f)
            wrench, _state_out = _first_step(model, state, control, solver)
            test.assertFalse(bool(solver.constraint_overflow.numpy().any()))
            test.assertAlmostEqual(_revolute_wrench_torques(model, state, wrench)[child_joint], torque, delta=1.0e-5)


def test_parent_wrench_warns_without_preeliminated_closures(test, device):
    """Pre-eliminated loop closures have no stored impulses, so requesting the wrench warns once."""
    model = _build_four_bar().finalize(device=device)
    solver = SolverFeatherPGS(model, solve_mode="matrix_free", enable_bilateral_preelimination=True)
    observables = solver.observables(FLAGS)
    state_0, state_1 = model.state(), model.state()
    with test.assertWarnsRegex(UserWarning, "pre-eliminated loop-closure"):
        solver.step(state_0, state_1, None, None, 1.0 / 240.0, observables=observables)


def test_unattached_body_reports_zero(test, device):
    """A body outside every articulation overwrites its row with zero."""
    builder = newton.ModelBuilder()
    root = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.1))
    builder.add_articulation([builder.add_joint_revolute(-1, root, axis=(0.0, 1.0, 0.0))])
    builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    model = builder.finalize(device=device)
    solver = SolverFeatherPGS(model)
    solver._rigid_body_angular_damping.zero_()
    observables = solver.observables(FLAGS)
    observables.body_parent_f.fill_(wp.spatial_vector(123.0, 123.0, 123.0, 123.0, 123.0, 123.0))
    solver.step(model.state(), model.state(), None, None, DT_parent_wrench, observables=observables)
    np.testing.assert_array_equal(observables.body_parent_f.numpy()[1], 0.0)


ARMATURE_DT = 0.01


def _armature_scene(device, armature):
    """A FREE body, a revolute carrying a welded child, a prismatic and a ball, each joint with the given armature.

    The FREE body starts at rest at its center of mass, so its joint coordinates are its world-frame motion.
    """
    builder = newton.ModelBuilder()
    free = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 2.0), wp.quat_identity()), mass=1.0, inertia=wp.mat33(np.eye(3) * 0.2)
    )
    builder.add_articulation([builder.add_joint_free(free)])
    arm = builder.add_link(mass=1.0, com=wp.vec3(0.1, 0.0, 0.0), inertia=wp.mat33(np.diag([0.15, 0.2, 0.3])))
    welded = builder.add_link(mass=0.5, com=wp.vec3(0.0, 0.05, 0.0), inertia=wp.mat33(np.eye(3) * 0.01))
    builder.add_articulation(
        [
            builder.add_joint_revolute(
                -1, arm, axis=(0.0, 1.0, 0.0), parent_xform=wp.transform(wp.vec3(1.0, 0.0, 1.0), wp.quat_identity())
            ),
            builder.add_joint_fixed(arm, welded, parent_xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity())),
        ]
    )
    slider = builder.add_link(mass=2.0, inertia=wp.mat33(np.eye(3) * 0.1))
    builder.add_articulation(
        [
            builder.add_joint_prismatic(
                -1, slider, axis=(0.0, 0.0, 1.0), parent_xform=wp.transform(wp.vec3(2.0, 0.0, 1.0), wp.quat_identity())
            )
        ]
    )
    pendulum = builder.add_link(mass=1.0, com=wp.vec3(0.0, 0.0, -0.2), inertia=wp.mat33(np.eye(3) * 0.05))
    builder.add_articulation(
        [builder.add_joint_ball(-1, pendulum, parent_xform=wp.transform(wp.vec3(3.0, 0.0, 1.0), wp.quat_identity()))]
    )
    model = builder.finalize(device=device)
    model.joint_armature.fill_(armature)
    state = model.state()
    joint_qd = np.zeros(model.joint_dof_count, dtype=np.float32)
    joint_qd[6:] = (0.7, -0.2, 0.2, -0.1, 0.3)
    state.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    body_f = np.zeros((model.body_count, 6), dtype=np.float32)
    body_f[free] = (1.0, -2.0, 10.0, 0.5, -0.3, 0.2)
    state.body_f.assign(body_f)
    control = model.control()
    control.joint_f.assign(np.array([0.0] * 6 + [10.0, 5.0, 1.0, -2.0, 0.5], dtype=np.float32))
    return model, state, control


def _step_armature_scene(model, state, control, solver):
    """Step once; return the published wrenches and the joint accelerations of the step."""
    observables = solver.observables(FLAGS)
    state_out = model.state()
    solver.step(state, state_out, control, None, ARMATURE_DT, observables=observables)
    qdd = (state_out.joint_qd.numpy() - state.joint_qd.numpy()) / ARMATURE_DT
    return observables.body_parent_f.numpy(), qdd


def test_parent_wrench_excludes_joint_armature(test, device):
    """Reconstruct every joint type's wrench without its armature.

    Projected onto its motion subspace, every wrench is the applied joint force less ``armature * qdd``. The FREE
    body has no ``joint_f``, so it reports ``-armature * qdd``, and the weld reports its child's rigid-body wrench.
    """
    configurations = [{"solve_mode": "split"}]
    if wp.get_device(device).is_cuda:
        configurations.append({"solve_mode": "matrix_free"})
    for armature in (0.0, 9.0):
        model, state, control = _armature_scene(device, armature)
        joint_f = control.joint_f.numpy()
        com_world = [
            np.array(wp.transform_point(wp.transform(*X), wp.vec3(*com)))
            for X, com in zip(state.body_q.numpy(), model.body_com.numpy(), strict=True)
        ]
        for options in configurations:
            with test.subTest(armature=armature, **options):
                solver = SolverFeatherPGS(model, **options)
                solver._rigid_body_angular_damping.zero_()
                wrench, qdd = _step_armature_scene(model, state, control, solver)
                expected = joint_f - armature * qdd
                np.testing.assert_allclose(wrench[0], expected[:6], rtol=1.0e-4, atol=1.0e-4)
                if armature:
                    test.assertGreater(np.abs(wrench[0]).max(), 1.0)
                hinge = np.array([1.0, 0.0, 1.0])
                hinge_torque = wrench[1, 3:] + np.cross(com_world[1] - hinge, wrench[1, :3])
                test.assertAlmostEqual(
                    float(hinge_torque[1]), float(expected[6]), delta=1.0e-4 * abs(expected[6]) + 1.0e-4
                )
                test.assertAlmostEqual(
                    float(wrench[3, 2]), float(expected[7]), delta=1.0e-4 * abs(expected[7]) + 1.0e-4
                )
                socket = np.array([3.0, 0.0, 1.0])
                ball_torque = wrench[4, 3:] + np.cross(com_world[4] - socket, wrench[4, :3])
                np.testing.assert_allclose(ball_torque, expected[8:11], rtol=1.0e-4, atol=1.0e-4)
                # The weld carries its child's mass times its acceleration from the hinge's start velocity and qdd.
                omega = np.array([0.0, state.joint_qd.numpy()[6], 0.0])
                alpha = np.array([0.0, qdd[6], 0.0])
                lever = com_world[2] - hinge
                accel = np.cross(alpha, lever) + np.cross(omega, np.cross(omega, lever))
                weld_force = model.body_mass.numpy()[2] * (accel - np.array([0.0, 0.0, -9.81]))
                np.testing.assert_allclose(wrench[2, :3], weld_force, rtol=1.0e-4, atol=1.0e-4)
                np.testing.assert_allclose(wrench[2, 3:], 0.01 * alpha, rtol=1.0e-4, atol=1.0e-4)


def test_parent_wrench_of_a_contact_stop_with_armature(test, device, solve_mode="split"):
    """Report a contact-stopped FREE body's armature share of the impact as its joint wrench.

    The contact impulse stops the armature-augmented inertia; the body's own mass takes ``m * dv / dt`` of it and
    the FREE joint reports the remaining ``-armature * dv / dt``.
    """
    for armature in (0.0, 9.0):
        with test.subTest(armature=armature):
            model = _sphere_on_ground(device, armature, dense=False)
            solver = SolverFeatherPGS(model, solve_mode=solve_mode, cfm=0.0)
            pipeline = newton.CollisionPipeline(model)
            observables = solver.observables(FLAGS | {SolverObservableFlags.CONTACT_F})
            state_in, state_out = model.state(), model.state()
            qd = state_in.joint_qd.numpy()
            qd[2] = -1.0
            state_in.joint_qd.assign(qd)
            newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
            contacts = pipeline.contacts()
            pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, model.control(), contacts, ARMATURE_DT, observables=observables)
            count = int(contacts.rigid_contact_count.numpy()[0])
            contact_z = abs(float(observables.contact_f.numpy()[:count, 2].sum()))
            accel_z = float(state_out.joint_qd.numpy()[2] - qd[2]) / ARMATURE_DT
            wrench = observables.body_parent_f.numpy()[0]
            test.assertGreater(accel_z, 90.0)
            test.assertAlmostEqual(float(wrench[2]), -armature * accel_z, delta=1.0e-4 * armature * accel_z + 1.0e-3)
            np.testing.assert_allclose(wrench[[0, 1, 3, 4, 5]], 0.0, atol=1.0e-3)
            mass = float(model.body_mass.numpy()[0])
            test.assertAlmostEqual(contact_z + float(wrench[2]), mass * accel_z, delta=1.0e-3 * mass * accel_z)


def test_parent_wrench_of_a_commanded_free_joint(test, device):
    """Report a FREE joint's own generalized command less its armature reaction.

    A 1 kg, unit-inertia body at rest driven through ``joint_f`` by 10 N and 10 N·m accelerates at
    ``10 / (1 + armature)``, so it reports ``10 - armature * qdd``: 1 with armature 9 and 10 without.
    """
    for armature, expected in ((0.0, 10.0), (9.0, 1.0)):
        with test.subTest(armature=armature):
            builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
            body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
            builder.add_articulation([builder.add_joint_free(body)])
            model = builder.finalize(device=device)
            model.joint_armature.fill_(armature)
            state = model.state()
            control = model.control()
            control.joint_f.assign(np.array([10.0, 0.0, 0.0, 0.0, 0.0, 10.0], dtype=np.float32))
            solver = SolverFeatherPGS(model, solve_mode="split")
            solver._rigid_body_angular_damping.zero_()
            wrench, qdd = _step_armature_scene(model, state, control, solver)
            np.testing.assert_allclose(qdd[[0, 5]], 10.0 / (1.0 + armature), rtol=1.0e-5)
            np.testing.assert_allclose(wrench[0], [expected, 0.0, 0.0, 0.0, 0.0, expected], rtol=1.0e-4, atol=1.0e-4)


def test_parent_wrench_of_a_held_kinematic_joint(test, device):
    """Report the support load of a held kinematic hinge, whatever its armature.

    The 1 kg hinge body sits on the axis and carries a welded 1 kg child 1 m out, so the hinge holds 2 g of
    weight and the child's -g moment about its axis, with zero command and zero acceleration.
    """
    for armature in (0.0, 9.0):
        with test.subTest(armature=armature):
            builder = newton.ModelBuilder()
            hinge = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)), is_kinematic=True)
            child = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
            builder.add_articulation(
                [
                    builder.add_joint_revolute(-1, hinge, axis=(0.0, 1.0, 0.0)),
                    builder.add_joint_fixed(
                        hinge, child, parent_xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity())
                    ),
                ]
            )
            model = builder.finalize(device=device)
            model.joint_armature.fill_(armature)
            solver = SolverFeatherPGS(model, solve_mode="split")
            solver._rigid_body_angular_damping.zero_()
            wrench, qdd = _step_armature_scene(model, model.state(), model.control(), solver)
            test.assertEqual(float(qdd[0]), 0.0)
            np.testing.assert_allclose(wrench[0], [0.0, 0.0, 2.0 * 9.81, 0.0, -9.81, 0.0], rtol=1.0e-5, atol=1.0e-4)


def test_parent_wrench_matches_mujoco_with_armature(test, device):
    """Publish the same joint wrenches as SolverMuJoCo on one step of the armature scene, with and without armature."""
    for armature in (0.0, 9.0):
        with test.subTest(armature=armature):
            model, state, control = _armature_scene(device, armature)
            solver = SolverFeatherPGS(model, solve_mode="matrix_free")
            solver._rigid_body_angular_damping.zero_()
            featherpgs, _ = _step_armature_scene(model, state, control, solver)
            mujoco, _ = _step_armature_scene(
                model, state, control, newton.solvers.SolverMuJoCo(model, integrator="euler")
            )
            test.assertGreater(np.abs(mujoco).max(), 1.0)
            np.testing.assert_allclose(featherpgs, mujoco, rtol=1.0e-4, atol=1.0e-4)


class TestFeatherPGSParentWrench(unittest.TestCase):
    pass


for _name in (
    "test_sleeping_body_keeps_its_parent_wrench",
    "test_sleeping_body_keeps_its_legacy_parent_wrench",
    "test_sleeping_parent_wrench_is_independent_of_the_output",
    "test_parent_wrench_first_requested_after_sleep",
    "test_sleeping_parent_wrench_from_legacy_to_observable",
):
    add_function_test(TestFeatherPGSParentWrench, _name, globals()[_name], devices=get_cuda_test_devices())
add_function_test(
    TestFeatherPGSParentWrench,
    "test_unattached_body_reports_zero",
    test_unattached_body_reports_zero,
    devices=get_test_devices(),
)
for _name in ("test_closure_force_uses_the_row_builder_norm", "test_parent_wrench_of_a_near_redundant_closure"):
    add_function_test(TestFeatherPGSParentWrench, _name, globals()[_name], devices=get_test_devices())
add_function_test(
    TestFeatherPGSParentWrench,
    "test_parent_wrench_matches_inverse_dynamics_on_every_path",
    test_parent_wrench_matches_inverse_dynamics_on_every_path,
    devices=get_test_devices(),
)
for _name in (
    "test_parent_wrench_excludes_joint_armature",
    "test_parent_wrench_of_a_contact_stop_with_armature",
    "test_parent_wrench_of_a_commanded_free_joint",
    "test_parent_wrench_of_a_held_kinematic_joint",
):
    add_function_test(TestFeatherPGSParentWrench, _name, globals()[_name], devices=get_test_devices())
add_function_test(
    TestFeatherPGSParentWrench,
    "test_parent_wrench_of_a_contact_stop_with_armature_matrix_free",
    test_parent_wrench_of_a_contact_stop_with_armature,
    devices=get_cuda_test_devices(),
    solve_mode="matrix_free",
)
add_function_test(
    TestFeatherPGSParentWrench,
    "test_parent_wrench_matches_mujoco_with_armature",
    test_parent_wrench_matches_mujoco_with_armature,
    devices=get_cuda_test_devices(),
)
for _name in (
    "test_parent_wrench_carries_a_load_resting_on_the_gripper",
    "test_parent_wrench_reports_a_commanded_push",
    "test_parent_wrench_carries_a_pinched_load",
    "test_parent_wrench_includes_spin_rows",
    "test_parent_wrench_excludes_velocity_limit_prescaling",
    "test_parent_wrench_includes_loop_closure_reactions",
    "test_parent_wrench_of_a_load_held_by_closures",
    "test_parent_wrench_warns_without_preeliminated_closures",
):
    add_function_test(TestFeatherPGSParentWrench, _name, globals()[_name], devices=get_cuda_test_devices())


if __name__ == "__main__":
    unittest.main()
