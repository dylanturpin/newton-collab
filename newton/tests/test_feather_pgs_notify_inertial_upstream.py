# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Model-change notifications and state refresh of SolverFeatherPGS."""

import unittest

import numpy as np
import warp as wp

import newton
from newton import ModelFlags
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

DT = 1.0 / 60.0
INITIAL_JOINT_Q = 0.3
NEW_COM = (0.15, 0.0, 0.05)


def _build_model(device, com=None):
    """Single-link pendulum on a Y-axis revolute joint, box COM at the pivot.

    With the COM at the pivot gravity exerts no torque, so any swing after a COM change is
    attributable to the change alone.
    """
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.density = 1000.0
    link = builder.add_link()
    builder.add_shape_box(link, hx=0.25, hy=0.05, hz=0.05)
    joint = builder.add_joint_revolute(
        -1,
        link,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.8), wp.quat_identity()),
        axis=newton.Axis.Y,
    )
    builder.add_articulation([joint])
    builder.joint_q[0] = INITIAL_JOINT_Q
    model = builder.finalize(device=device)
    if com is not None:
        body_com = model.body_com.numpy()
        body_com[0] = com
        model.body_com.assign(body_com)
    return model


def test_step_refreshes_reused_state_after_joint_q_write(test, device, pgs_mode="matrix_free"):
    """Pick up a joint_q write into a state the solver produced on the previous step."""
    model = _build_model(device)
    solver = SolverFeatherPGS(model, pgs_mode=pgs_mode)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    solver.step(state_0, state_1, control, None, DT)
    joint_q = state_1.joint_q.numpy()
    joint_q[0] = -0.9
    state_1.joint_q.assign(joint_q)
    solver.step(state_1, state_0, control, None, DT)

    reference_model = _build_model(device)
    reference_state = reference_model.state()
    reference_state.joint_q.assign(state_1.joint_q)
    reference_state.joint_qd.assign(state_1.joint_qd)
    reference_out = reference_model.state()
    SolverFeatherPGS(reference_model, pgs_mode=pgs_mode).step(
        reference_state, reference_out, reference_model.control(), None, DT
    )
    np.testing.assert_allclose(state_0.joint_q.numpy(), reference_out.joint_q.numpy(), rtol=0.0, atol=1.0e-6)
    np.testing.assert_allclose(state_0.body_q.numpy(), reference_out.body_q.numpy(), rtol=0.0, atol=1.0e-6)


def test_kinematic_flag_change_is_picked_up(test, device, pgs_mode="matrix_free"):
    """Hold a body still once it is flagged kinematic and the solver is notified."""
    model = _build_model(device, com=NEW_COM)
    solver = SolverFeatherPGS(model, pgs_mode=pgs_mode)
    state_0, state_1 = model.state(), model.state()
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    control = model.control()
    flags = model.body_flags.numpy()
    flags[0] = int(newton.BodyFlags.KINEMATIC)
    model.body_flags.assign(flags)
    solver.notify_model_changed(ModelFlags.BODY_PROPERTIES)
    for _ in range(30):
        solver.step(state_0, state_1, control, None, DT)
        state_0, state_1 = state_1, state_0
    test.assertAlmostEqual(float(state_0.joint_q.numpy()[0]), INITIAL_JOINT_Q, places=5)


class TestFeatherPGSNotifyInertial(unittest.TestCase):
    pass


devices = get_cuda_test_devices()
for _name in (
    "test_step_refreshes_reused_state_after_joint_q_write",
    "test_kinematic_flag_change_is_picked_up",
):
    add_function_test(TestFeatherPGSNotifyInertial, _name, globals()[_name], devices=devices)
    add_function_test(
        TestFeatherPGSNotifyInertial, f"{_name}_split", globals()[_name], devices=get_test_devices(), pgs_mode="split"
    )


if __name__ == "__main__":
    unittest.main()
