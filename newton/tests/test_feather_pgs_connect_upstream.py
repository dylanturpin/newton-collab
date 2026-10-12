# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Connect (loop-closure) rows of SolverFeatherPGS."""

import functools
import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices


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


# -- Four-bar closure ------------------------------------------------------------------


def test_four_bar_capture_matches_eager(test, device):
    """Replay the eager four-bar trajectory from a captured step."""
    model = _build_four_bar().finalize(device=device)
    control = model.control()
    targets = model.joint_target_q.numpy().copy()
    targets[0] = 0.6
    control.joint_target_q.assign(targets)
    eager = SolverFeatherPGS(model, pgs_mode="matrix_free", pgs_iterations=16, pgs_beta=0.1)
    captured = SolverFeatherPGS(model, pgs_mode="matrix_free", pgs_iterations=16, pgs_beta=0.1)
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
    solver = SolverFeatherPGS(model, pgs_mode="matrix_free", **solver_kwargs)
    test.assertEqual(solver._connect_count, 0)
    np.testing.assert_array_equal(solver._model_plan.loop_joint_articulation, np.array([-1, -1], dtype=np.int32))


def test_propagation_standalone_world_root_is_not_loop_joint(test, device):
    """Do not treat an unowned world-root joint as a closure with propagation responses."""
    check_standalone_world_root_is_not_loop_joint(test, device, articulated_contact_response="propagation")


# ---------------------------------------------------------------------------
# Closures with a prescribed (kinematic or world) parent
# ---------------------------------------------------------------------------

_CHILD_ANCHORS = ((0.0, 0.0, 0.0), (0.05, 0.0, 0.0), (0.0, 0.05, 0.0))
_REL_P = np.array([0.0, 0.0, -0.2])


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


def test_disabled_loop_joint_starts_released(test, device):
    """Start a loop joint disabled in Model.joint_enabled released, and engage it later."""
    b, _, load, joints = _build_carried_load(enabled=False, world_parent=True)
    model = b.finalize(device=device)
    solver = SolverFeatherPGS(model, pgs_mode="matrix_free", pgs_iterations=16, pgs_beta=0.2)
    state_0, state_1 = model.state(), model.state()
    z0 = float(state_0.body_q.numpy()[load, 2])
    for _ in range(48):
        solver.step(state_0, state_1, model.control(), None, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0
    test.assertGreater(z0 - float(state_0.body_q.numpy()[load, 2]), 0.15)
    test.assertEqual(solver.connect_slot.numpy().tolist(), [-1, -1, -1])

    for j in joints:
        solver.set_loop_joint_enabled(j, True)
    solver.step(state_0, state_1, model.control(), None, 1.0 / 240.0)
    test.assertTrue(np.all(solver.connect_slot.numpy() >= 0))


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
                SolverFeatherPGS(model, pgs_mode="matrix_free")
            continue
        solver = SolverFeatherPGS(model, pgs_mode="matrix_free", pgs_iterations=16)
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


class TestFeatherPGSConnect(unittest.TestCase):
    pass


class TestFeatherPGSPrescribedParentConnect(unittest.TestCase):
    pass


class TestFeatherPGSConnectPropagation(unittest.TestCase):
    pass


cuda_devices = get_cuda_test_devices()
for _name in (
    "test_four_bar_capture_matches_eager",
    "test_imported_connect_equality_is_enforced",
):
    add_function_test(TestFeatherPGSConnect, _name, globals()[_name], devices=cuda_devices)
for _name in ("test_disabled_loop_joint_starts_released",):
    add_function_test(TestFeatherPGSPrescribedParentConnect, _name, globals()[_name], devices=cuda_devices)
add_function_test(
    TestFeatherPGSConnectPropagation,
    "test_propagation_standalone_world_root_is_not_loop_joint",
    test_propagation_standalone_world_root_is_not_loop_joint,
    devices=cuda_devices,
)


if __name__ == "__main__":
    unittest.main()
