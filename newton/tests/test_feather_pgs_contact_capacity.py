# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for FeatherPGS contact-capacity handling."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.kernels import allocate_world_contact_slots
from newton.solvers import SolverFeatherPGS


def _build_model():
    """Build one articulated body for contact-capacity tests."""
    builder = newton.ModelBuilder()
    body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    joint = builder.add_joint_free(parent=-1, child=body)
    builder.add_articulation([joint])
    model = builder.finalize(device="cpu")
    model.rigid_contact_max = 1
    return model


class TestFeatherPGSContactCapacity(unittest.TestCase):
    def test_allocator_rejects_incomplete_contact_frame(self):
        """Ignore a frame whose reported contact count exceeds its materialized capacity."""
        capacity = 2
        device = "cpu"
        contact_slot = wp.full((capacity,), -7, dtype=wp.int32, device=device)
        contact_path = wp.full((capacity,), -7, dtype=wp.int32, device=device)
        world_slot_counter = wp.zeros((1,), dtype=wp.int32, device=device)

        wp.launch(
            allocate_world_contact_slots,
            dim=capacity,
            inputs=[
                wp.array([capacity + 1], dtype=wp.int32, device=device),
                wp.zeros((capacity,), dtype=wp.int32, device=device),
                wp.full((capacity,), -1, dtype=wp.int32, device=device),
                wp.zeros((capacity,), dtype=wp.vec3, device=device),
                wp.zeros((capacity,), dtype=wp.vec3, device=device),
                wp.array([wp.vec3(0.0, 0.0, 1.0)] * capacity, dtype=wp.vec3, device=device),
                wp.zeros((capacity,), dtype=wp.float32, device=device),
                wp.zeros((capacity,), dtype=wp.float32, device=device),
                wp.array([wp.transform_identity()], dtype=wp.transform, device=device),
                wp.array([wp.transform_identity()], dtype=wp.transform, device=device),
                wp.array([0], dtype=wp.int32, device=device),
                wp.array([0], dtype=wp.int32, device=device),
                wp.array([0], dtype=wp.int32, device=device),
                wp.array([6], dtype=wp.int32, device=device),
                wp.zeros((1,), dtype=wp.int32, device=device),
                wp.ones((1,), dtype=wp.int32, device=device),
                wp.zeros((1,), dtype=wp.int32, device=device),
                0,
                0,
                0,
                0,
                0.0,
                8,
                8,
                8,
                0,
                0.0,
                0,
            ],
            outputs=[
                wp.zeros((capacity,), dtype=wp.int32, device=device),
                contact_slot,
                wp.zeros((capacity,), dtype=wp.int32, device=device),
                wp.zeros((capacity,), dtype=wp.int32, device=device),
                world_slot_counter,
                contact_path,
                wp.zeros((1,), dtype=wp.int32, device=device),
                wp.zeros((1,), dtype=wp.int32, device=device),
                wp.zeros((1,), dtype=wp.int32, device=device),
                wp.zeros((capacity,), dtype=wp.int32, device=device),
            ],
            device=device,
        )

        np.testing.assert_array_equal(contact_slot.numpy(), np.full(capacity, -1, dtype=np.int32))
        np.testing.assert_array_equal(contact_path.numpy(), np.full(capacity, -1, dtype=np.int32))
        self.assertEqual(int(world_slot_counter.numpy()[0]), 0)

    def test_step_rejects_contacts_larger_than_scratch(self):
        """Reject oversized contact input before it can overwrite solver scratch."""
        model = _build_model()
        solver = SolverFeatherPGS(model)
        contacts = newton.Contacts(rigid_contact_max=2, soft_contact_max=0, device=model.device)

        with self.assertRaisesRegex(ValueError, "contact capacity"):
            solver.step(model.state(), model.state(), model.control(), contacts, 1.0 / 60.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
