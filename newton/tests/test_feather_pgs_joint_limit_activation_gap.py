# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import warp as wp

import newton
from newton._src.core.types import MAXVAL
from newton._src.sim.enums import JointType
from newton._src.solvers.feather_pgs.kernels import allocate_joint_limit_slots
from newton.solvers import SolverFeatherPGS


def _allocated_slots(q: float, *, gap: float, lower: float = -1.0, upper: float = 1.0):
    device = "cpu"
    limit_slot = wp.full((2,), -1, dtype=wp.int32, device=device)
    limit_sign = wp.zeros((2,), dtype=wp.float32, device=device)
    world_slot_counter = wp.zeros((1,), dtype=wp.int32, device=device)

    wp.launch(
        allocate_joint_limit_slots,
        dim=1,
        inputs=[
            wp.array([0, 1], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([1], dtype=wp.int32, device=device),
            wp.array([int(JointType.REVOLUTE)], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([[0, 1]], dtype=wp.int32, device=device),
            wp.array([lower], dtype=wp.float32, device=device),
            wp.array([upper], dtype=wp.float32, device=device),
            wp.array([q], dtype=wp.float32, device=device),
            gap,
            wp.array([0], dtype=wp.int32, device=device),
            8,
        ],
        outputs=[limit_slot, limit_sign, world_slot_counter],
        device=device,
    )
    wp.synchronize()
    return (
        limit_slot.numpy().tolist(),
        limit_sign.numpy().tolist(),
        int(world_slot_counter.numpy()[0]),
    )


class TestFeatherPGSJointLimitActivationGap(unittest.TestCase):
    def test_solver_rejects_invalid_joint_limit_activation_gap(self):
        model = newton.ModelBuilder().finalize()

        for value in (-0.1, float("nan")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "joint_limit_activation_gap"):
                    SolverFeatherPGS(model, joint_limit_activation_gap=value)

    def test_solver_accepts_finite_and_infinite_joint_limit_activation_gap(self):
        model = newton.ModelBuilder().finalize()

        self.assertEqual(SolverFeatherPGS(model, joint_limit_activation_gap=0.2).joint_limit_activation_gap, 0.2)
        self.assertEqual(
            SolverFeatherPGS(model, joint_limit_activation_gap=float("inf")).joint_limit_activation_gap,
            float("inf"),
        )

    def test_finite_gap_allocates_only_near_limits(self):
        slots, signs, count = _allocated_slots(0.0, gap=0.2)
        self.assertEqual(slots, [-1, -1])
        self.assertEqual(signs, [0.0, 0.0])
        self.assertEqual(count, 0)

        slots, signs, count = _allocated_slots(-0.85, gap=0.2)
        self.assertEqual(slots, [0, -1])
        self.assertEqual(signs, [1.0, 0.0])
        self.assertEqual(count, 1)

        slots, signs, count = _allocated_slots(0.85, gap=0.2)
        self.assertEqual(slots, [-1, 0])
        self.assertEqual(signs, [0.0, -1.0])
        self.assertEqual(count, 1)

    def test_finite_gap_does_not_activate_unlimited_sentinel_limits(self):
        slots, signs, count = _allocated_slots(0.0, gap=0.2, lower=-MAXVAL, upper=MAXVAL)

        self.assertEqual(slots, [-1, -1])
        self.assertEqual(signs, [0.0, 0.0])
        self.assertEqual(count, 0)

    def test_infinite_gap_preserves_historical_always_allocate_behavior(self):
        slots, signs, count = _allocated_slots(0.0, gap=float("inf"))

        self.assertEqual(slots, [0, 1])
        self.assertEqual(signs, [1.0, -1.0])
        self.assertEqual(count, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
