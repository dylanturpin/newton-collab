# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Joint and free-body velocity-limit rows of SolverFeatherPGS."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.sim.enums import BodyFlags, JointType
from newton._src.solvers.feather_pgs.kernels import (
    allocate_joint_velocity_limit_slots,
    allocate_rigid_velocity_limit_slots,
)
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

QDOT_MAX = 0.5
# Slack on the limit for an under-converged sweep; the final pass leaves the DOF at the limit.
LIMIT_TOL = 1.05


def _allocated_joint_velocity_slots(device, qd: float, *, fraction: float, qdot_max: float = 1.0):
    velocity_limit_slot = wp.full((2,), -1, dtype=wp.int32, device=device)
    velocity_limit_sign = wp.zeros((2,), dtype=wp.float32, device=device)
    world_slot_counter = wp.zeros((1,), dtype=wp.int32, device=device)
    wp.launch(
        allocate_joint_velocity_limit_slots,
        dim=1,
        inputs=[
            wp.array([0, 1], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([1], dtype=wp.int32, device=device),
            wp.array([int(JointType.REVOLUTE)], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([[0, 1]], dtype=wp.int32, device=device),
            wp.array([qdot_max], dtype=wp.float32, device=device),
            wp.array([qd], dtype=wp.float32, device=device),
            fraction,
            wp.array([-1], dtype=wp.int32, device=device),
            0,
            wp.array([0], dtype=wp.int32, device=device),
            8,
            wp.ones(1, dtype=wp.int32, device=device),
        ],
        outputs=[velocity_limit_slot, velocity_limit_sign, world_slot_counter],
        device=device,
    )
    return (
        velocity_limit_slot.numpy().tolist(),
        velocity_limit_sign.numpy().tolist(),
        int(world_slot_counter.numpy()[0]),
    )


def _allocated_rigid_velocity_slots(device, qd6, *, fraction: float, lin_limit: float = 1.0, ang_limit: float = 1.0):
    rigid_velocity_limit_slot = wp.full((12,), -1, dtype=wp.int32, device=device)
    rigid_velocity_limit_sign = wp.zeros((12,), dtype=wp.float32, device=device)
    mf_slot_counter = wp.zeros((1,), dtype=wp.int32, device=device)
    wp.launch(
        allocate_rigid_velocity_limit_slots,
        dim=1,
        inputs=[
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([1], dtype=wp.int32, device=device),
            wp.array([int(BodyFlags.DYNAMIC)], dtype=wp.int32, device=device),
            wp.array([lin_limit], dtype=wp.float32, device=device),
            wp.array([ang_limit], dtype=wp.float32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array(list(qd6), dtype=wp.float32, device=device),
            fraction,
            64,
            wp.ones(1, dtype=wp.int32, device=device),
        ],
        outputs=[rigid_velocity_limit_slot, rigid_velocity_limit_sign, mf_slot_counter],
        device=device,
    )
    return (
        rigid_velocity_limit_slot.numpy().tolist(),
        rigid_velocity_limit_sign.numpy().tolist(),
        int(mf_slot_counter.numpy()[0]),
    )


def test_fraction_zero_allocates_every_joint_row(test, device):
    """Allocate both rows of every limited DOF regardless of its velocity when the fraction is zero."""
    for qd in (0.0, 0.5, -2.0):
        with test.subTest(qd=qd):
            slots, signs, count = _allocated_joint_velocity_slots(device, qd, fraction=0.0)
            test.assertEqual(slots, [0, 1])
            test.assertEqual(signs, [1.0, -1.0])
            test.assertEqual(count, 2)


def test_fraction_zero_allocates_every_rigid_row(test, device):
    """Allocate all twelve free-body velocity-limit rows when the fraction is zero."""
    slots, signs, count = _allocated_rigid_velocity_slots(device, [0.0] * 6, fraction=0.0)
    test.assertEqual(slots, list(range(12)))
    test.assertEqual(signs, [1.0, -1.0] * 6)
    test.assertEqual(count, 12)


def test_free_body_velocity_limits_hold(test, device, pgs_mode="matrix_free"):
    """Hold a free body's authored linear and angular speed limits."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    SolverFeatherPGS.register_custom_attributes(builder)
    body = builder.add_body(
        custom_attributes={"rigid_body_max_linear_velocity": 0.5, "rigid_body_max_angular_velocity": 1.0}
    )
    builder.add_shape_box(body, hx=0.1, hy=0.2, hz=0.3)
    model = builder.finalize(device=device)
    solver = SolverFeatherPGS(model, pgs_mode=pgs_mode)
    state_0, state_1 = model.state(), model.state()
    qd = state_0.joint_qd.numpy()
    qd[3:6] = (5.0, 0.0, 0.0)
    state_0.joint_qd.assign(qd)
    for _ in range(60):
        solver.step(state_0, state_1, model.control(), None, 1.0 / 60.0)
        state_0, state_1 = state_1, state_0
    final = state_0.joint_qd.numpy()
    test.assertLessEqual(float(np.max(np.abs(final[0:3]))), 0.5 * LIMIT_TOL)
    test.assertLessEqual(float(np.max(np.abs(final[3:6]))), 1.0 * LIMIT_TOL)


class TestFeatherPGSVelocityLimitActivationFraction(unittest.TestCase):
    pass


devices = get_cuda_test_devices()
for _name, _func in (
    ("test_fraction_zero_allocates_every_joint_row", test_fraction_zero_allocates_every_joint_row),
    ("test_fraction_zero_allocates_every_rigid_row", test_fraction_zero_allocates_every_rigid_row),
    ("test_free_body_velocity_limits_hold", test_free_body_velocity_limits_hold),
):
    add_function_test(TestFeatherPGSVelocityLimitActivationFraction, _name, _func, devices=devices)
# The allocators are mode-independent kernels; free-body velocity limits are also rows in split mode.
for _name, _func in (
    ("test_fraction_zero_allocates_every_joint_row", test_fraction_zero_allocates_every_joint_row),
    ("test_fraction_zero_allocates_every_rigid_row", test_fraction_zero_allocates_every_rigid_row),
):
    add_function_test(
        TestFeatherPGSVelocityLimitActivationFraction, _name, _func, devices=[d for d in get_test_devices() if d.is_cpu]
    )
add_function_test(
    TestFeatherPGSVelocityLimitActivationFraction,
    "test_free_body_velocity_limits_hold_split",
    test_free_body_velocity_limits_hold,
    devices=get_test_devices(),
    pgs_mode="split",
)


if __name__ == "__main__":
    unittest.main()
