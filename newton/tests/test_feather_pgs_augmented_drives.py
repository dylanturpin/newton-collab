# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from newton._src.sim.enums import JointType
from newton._src.solvers.feather_pgs.kernels import build_augmented_joint_rows_and_apply_tau


class TestFeatherPGSAugmentedDrives(unittest.TestCase):
    def test_fused_builder_augments_only_unlimited_drives(self):
        """Keep bounded drives out of the unrestricted augmented response."""
        device = wp.get_device()
        row_counts = wp.zeros(1, dtype=wp.int32, device=device)
        row_dof_index = wp.zeros(3, dtype=wp.int32, device=device)
        row_stiffness = wp.zeros(3, dtype=wp.float32, device=device)
        limit_counts = wp.zeros(1, dtype=wp.int32, device=device)
        joint_tau = wp.array([5.0, -4.0, 1.0], dtype=wp.float32, device=device)

        wp.launch(
            build_augmented_joint_rows_and_apply_tau,
            dim=1,
            inputs=[
                wp.array([0, 3], dtype=wp.int32, device=device),
                wp.array([0], dtype=wp.int32, device=device),
                wp.array([3], dtype=wp.int32, device=device),
                wp.array([JointType.REVOLUTE] * 3, dtype=wp.int32, device=device),
                wp.array([0, 1, 2], dtype=wp.int32, device=device),
                wp.array([0, 1, 2], dtype=wp.int32, device=device),
                wp.array(np.tile([0, 1], (3, 1)), dtype=wp.int32, device=device),
                wp.array([10.0, 0.0, 20.0], dtype=wp.float32, device=device),
                wp.array([1.0, 0.0, 2.0], dtype=wp.float32, device=device),
                wp.array([0.2, 0.3, -0.4], dtype=wp.float32, device=device),
                wp.array([0.5, 0.6, -0.7], dtype=wp.float32, device=device),
                wp.array([0.0, 0.0, 0.1], dtype=wp.float32, device=device),
                wp.array([0.0, 0.0, 0.2], dtype=wp.float32, device=device),
                wp.array([wp.inf, 10.0, 2.5], dtype=wp.float32, device=device),
                3,
                0.1,
            ],
            outputs=[row_counts, row_dof_index, row_stiffness, limit_counts, joint_tau],
            device=device,
        )

        np.testing.assert_array_equal(row_counts.numpy(), [1])
        np.testing.assert_array_equal(row_dof_index.numpy()[:1], [0])
        np.testing.assert_allclose(row_stiffness.numpy()[:1], [0.2], rtol=0.0, atol=1.0e-7)
        np.testing.assert_allclose(joint_tau.numpy(), [2.0, -4.0, 1.0], rtol=0.0, atol=1.0e-6)


if __name__ == "__main__":
    unittest.main()
