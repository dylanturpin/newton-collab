# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_JOINT_LIMIT, PGS_CONSTRAINT_TYPE_MIMIC
from newton._src.solvers.feather_pgs.solver_feather_pgs import _get_pgs_solve_local_internal_kernel


class TestFeatherPGSLocalInternalSolve(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "local internal solve requires CUDA")
    def test_local_kernel_matches_sequential_pgs_and_skips_contact_worlds(self):
        device = wp.get_cuda_device()
        dof_count = 3
        max_constraints = 4
        local_max_constraints = 2
        iterations = 4
        omega = 1.0

        lower = np.array([[2.0, 0.0, 0.0], [0.25, 1.5, 0.0], [-0.1, 0.2, 1.25]], dtype=np.float32)
        jacobian = np.zeros((2, max_constraints, dof_count), dtype=np.float32)
        jacobian[0, 0] = [1.0, 0.5, -0.25]
        jacobian[0, 1] = [-0.4, 0.3, 0.8]
        jacobian[1, 0] = [0.2, -0.1, 0.7]
        velocity = np.array([0.3, -0.5, 0.1, 9.0, 8.0, 7.0], dtype=np.float32)
        rhs = np.zeros((2, max_constraints), dtype=np.float32)
        rhs[0, :2] = [-0.2, 0.15]
        cfm = np.zeros((2, max_constraints), dtype=np.float32)
        cfm[0, :2] = [0.01, 0.02]
        row_type = np.zeros((2, max_constraints), dtype=np.int32)
        row_type[0, :2] = [PGS_CONSTRAINT_TYPE_MIMIC, PGS_CONSTRAINT_TYPE_JOINT_LIMIT]

        impulses = wp.zeros((2, max_constraints), dtype=wp.float32, device=device)
        velocity_out = wp.array(velocity, dtype=wp.float32, device=device)
        kernel = _get_pgs_solve_local_internal_kernel(
            max_constraints, local_max_constraints, dof_count, device.arch
        )
        wp.launch_tiled(
            kernel,
            dim=[2],
            inputs=[
                wp.array([0, 1], dtype=wp.int32, device=device),
                wp.array([0, 1], dtype=wp.int32, device=device),
                wp.array([0, 1], dtype=wp.int32, device=device),
                wp.array([0, 3], dtype=wp.int32, device=device),
                wp.array([2, 1], dtype=wp.int32, device=device),
                wp.array([[2, 2], [1, 1]], dtype=wp.int32, device=device),
                wp.array([0, 1], dtype=wp.int32, device=device),
                wp.array(np.stack([lower, np.eye(dof_count, dtype=np.float32)]), dtype=wp.float32, device=device),
                wp.array(jacobian, dtype=wp.float32, device=device),
                wp.array(rhs, dtype=wp.float32, device=device),
                wp.array(cfm, dtype=wp.float32, device=device),
                impulses,
                wp.array(row_type, dtype=wp.int32, device=device),
                iterations,
                omega,
            ],
            outputs=[velocity_out],
            block_dim=32,
            device=device,
        )
        wp.synchronize_device(device)

        mass = lower @ lower.T
        response = np.stack([np.linalg.solve(mass, row) for row in jacobian[0, :2]])
        diagonal = np.sum(jacobian[0, :2] * response, axis=1) + cfm[0, :2]
        expected_velocity = velocity[:dof_count].copy()
        expected_impulses = np.zeros(2, dtype=np.float32)
        for _ in range(iterations):
            for row in range(2):
                delta = -(jacobian[0, row] @ expected_velocity + rhs[0, row]) / diagonal[row]
                new_impulse = expected_impulses[row] + omega * delta
                if row_type[0, row] == PGS_CONSTRAINT_TYPE_JOINT_LIMIT:
                    new_impulse = max(new_impulse, 0.0)
                expected_velocity += response[row] * (new_impulse - expected_impulses[row])
                expected_impulses[row] = new_impulse

        np.testing.assert_allclose(velocity_out.numpy()[:dof_count], expected_velocity, rtol=0.0, atol=2.0e-6)
        np.testing.assert_allclose(impulses.numpy()[0, :2], expected_impulses, rtol=0.0, atol=2.0e-6)
        np.testing.assert_array_equal(velocity_out.numpy()[dof_count:], velocity[dof_count:])
        np.testing.assert_array_equal(impulses.numpy()[1], np.zeros(max_constraints, dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
