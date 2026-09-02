# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    PGS_CONSTRAINT_TYPE_MIMIC,
    PGS_LOCAL_SOLVE_OWNER_GENERAL,
    PGS_LOCAL_SOLVE_OWNER_PAIR,
    PGS_LOCAL_SOLVE_OWNER_SINGLE,
    classify_local_solve_worlds,
    compact_local_pair_candidates,
)
from newton._src.solvers.feather_pgs.solver_feather_pgs import _get_pgs_solve_local_owned_kernel


class TestFeatherPGSLocalInternalSolve(unittest.TestCase):
    def test_local_owner_classifier_rejects_unsupported_worlds(self):
        device = "cpu"
        constraint_count = wp.array([5, 8, 21, 8, 8, 0], dtype=wp.int32, device=device)
        phase_bounds = wp.array([[0, 5], [0, 2], [0, 2], [0, 2], [0, 2], [0, 0]], dtype=wp.int32, device=device)
        mf_count = wp.array([0, 0, 0, 1, 0, 0], dtype=wp.int32, device=device)
        primary = wp.array([0, 1, 2, 3, 4, 5], dtype=wp.int32, device=device)
        pair = wp.array([-1, 10, 11, 12, -1, 13], dtype=wp.int32, device=device)
        owner = wp.empty(6, dtype=wp.int32, device=device)
        general_count = wp.zeros(1, dtype=wp.int32, device=device)
        general_worlds = wp.empty(6, dtype=wp.int32, device=device)

        wp.launch(
            classify_local_solve_worlds,
            dim=6,
            inputs=[constraint_count, phase_bounds, mf_count, primary, pair, 20],
            outputs=[owner, general_count, general_worlds],
            device=device,
        )

        np.testing.assert_array_equal(
            owner.numpy(),
            [
                PGS_LOCAL_SOLVE_OWNER_SINGLE,
                PGS_LOCAL_SOLVE_OWNER_PAIR,
                PGS_LOCAL_SOLVE_OWNER_GENERAL,
                PGS_LOCAL_SOLVE_OWNER_GENERAL,
                PGS_LOCAL_SOLVE_OWNER_GENERAL,
                PGS_LOCAL_SOLVE_OWNER_GENERAL,
            ],
        )
        self.assertEqual(int(general_count.numpy()[0]), 3)
        np.testing.assert_array_equal(general_worlds.numpy()[:3], [2, 3, 4])

        active_count = wp.zeros(1, dtype=wp.int32, device=device)
        active_primary = wp.empty(6, dtype=wp.int32, device=device)
        active_secondary = wp.empty(6, dtype=wp.int32, device=device)
        wp.launch(
            compact_local_pair_candidates,
            dim=6,
            inputs=[primary, pair, wp.array(np.arange(14), dtype=wp.int32, device=device), owner],
            outputs=[active_count, active_primary, active_secondary],
            device=device,
        )
        self.assertEqual(int(active_count.numpy()[0]), 1)
        np.testing.assert_array_equal(active_primary.numpy()[:1], [1])
        np.testing.assert_array_equal(active_secondary.numpy()[:1], [10])

    @unittest.skipUnless(wp.is_cuda_available(), "local internal solve requires CUDA")
    def test_local_kernel_matches_sequential_pgs_and_respects_world_ownership(self):
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
        row_type[1, 0] = PGS_CONSTRAINT_TYPE_MIMIC

        impulses = wp.zeros((2, max_constraints), dtype=wp.float32, device=device)
        diagonal_out = wp.array(cfm, dtype=wp.float32, device=device)
        velocity_out = wp.array(velocity, dtype=wp.float32, device=device)
        candidate_articulations = wp.array([0, 1], dtype=wp.int32, device=device)
        lower_group = wp.array(np.stack([lower, np.eye(dof_count, dtype=np.float32)]), dtype=wp.float32, device=device)
        jacobian_group = wp.array(jacobian, dtype=wp.float32, device=device)
        kernel = _get_pgs_solve_local_owned_kernel(max_constraints, local_max_constraints, dof_count, device.arch)
        wp.launch_tiled(
            kernel,
            dim=[2],
            inputs=[
                candidate_articulations,
                candidate_articulations,
                1,
                wp.array([0, 1], dtype=wp.int32, device=device),
                wp.array([0, 1], dtype=wp.int32, device=device),
                wp.array([0, 3], dtype=wp.int32, device=device),
                wp.array([1, 0], dtype=wp.int32, device=device),
                wp.array([2, 1], dtype=wp.int32, device=device),
                lower_group,
                jacobian_group,
                lower_group,
                jacobian_group,
                wp.array(rhs, dtype=wp.float32, device=device),
                wp.array(row_type, dtype=wp.int32, device=device),
                wp.full((2, max_constraints), -1, dtype=wp.int32, device=device),
                wp.zeros((2, max_constraints), dtype=wp.float32, device=device),
                iterations,
                omega,
                0,
                0,
            ],
            outputs=[diagonal_out, impulses, velocity_out],
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

        def launch_variant(
            warps_per_block: int, *, contact_capable: bool = True, persistent_queue: bool = False, active_count: int = 2
        ):
            variant_diagonal = wp.array(cfm, dtype=wp.float32, device=device)
            variant_impulses = wp.zeros((2, max_constraints), dtype=wp.float32, device=device)
            variant_velocity = wp.array(velocity, dtype=wp.float32, device=device)
            variant_kernel = _get_pgs_solve_local_owned_kernel(
                max_constraints,
                local_max_constraints,
                dof_count,
                device.arch,
                persistent_queue=persistent_queue,
                warps_per_block=warps_per_block,
                contact_capable=contact_capable,
            )
            inputs = [candidate_articulations, candidate_articulations]
            if persistent_queue:
                inputs.extend([wp.array([active_count], dtype=wp.int32, device=device), 2])
            inputs.extend(
                [
                    PGS_LOCAL_SOLVE_OWNER_SINGLE,
                    wp.array([0, 1], dtype=wp.int32, device=device),
                    wp.array([0, 1], dtype=wp.int32, device=device),
                    wp.array([0, 3], dtype=wp.int32, device=device),
                    wp.array([PGS_LOCAL_SOLVE_OWNER_SINGLE] * 2, dtype=wp.int32, device=device),
                    wp.array([2, 1], dtype=wp.int32, device=device),
                    lower_group,
                    jacobian_group,
                    lower_group,
                    jacobian_group,
                    wp.array(rhs, dtype=wp.float32, device=device),
                    wp.array(row_type, dtype=wp.int32, device=device),
                    wp.full((2, max_constraints), -1, dtype=wp.int32, device=device),
                    wp.zeros((2, max_constraints), dtype=wp.float32, device=device),
                    iterations,
                    omega,
                    0,
                    0,
                ]
            )
            wp.launch_tiled(
                variant_kernel,
                dim=[2 // warps_per_block],
                inputs=inputs,
                outputs=[variant_diagonal, variant_impulses, variant_velocity],
                block_dim=32 * warps_per_block,
                device=device,
            )
            wp.synchronize_device(device)
            return variant_diagonal.numpy(), variant_impulses.numpy(), variant_velocity.numpy()

        one_warp_outputs = launch_variant(1)
        variants = [
            launch_variant(2),
            launch_variant(1, contact_capable=False),
            launch_variant(2, contact_capable=False),
        ]
        for variant_outputs in variants:
            for one_warp, variant in zip(one_warp_outputs, variant_outputs, strict=True):
                np.testing.assert_array_equal(variant, one_warp)

        one_warp_queue = launch_variant(1, persistent_queue=True, active_count=1)
        two_warp_queue = launch_variant(2, persistent_queue=True, active_count=1)
        for one_warp, two_warp in zip(one_warp_queue, two_warp_queue, strict=True):
            np.testing.assert_array_equal(two_warp, one_warp)

    @unittest.skipUnless(wp.is_cuda_available(), "local internal solve requires CUDA")
    def test_local_kernel_waits_for_delayed_friction_activation(self):
        device = wp.get_cuda_device()
        dof_count = 3
        max_constraints = 3
        candidate = wp.array([0], dtype=wp.int32, device=device)
        lower = wp.array(np.eye(dof_count, dtype=np.float32)[None], dtype=wp.float32, device=device)
        jacobian = wp.array(np.eye(dof_count, dtype=np.float32)[None], dtype=wp.float32, device=device)
        impulses = wp.array([[1.0, 0.0, 0.0]], dtype=wp.float32, device=device)
        diagonal = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)
        velocity = wp.array([0.0, 1.0, 0.0], dtype=wp.float32, device=device)
        kernel = _get_pgs_solve_local_owned_kernel(max_constraints, max_constraints, dof_count, device.arch)
        wp.launch_tiled(
            kernel,
            dim=[1],
            inputs=[
                candidate,
                candidate,
                1,
                wp.array([0], dtype=wp.int32, device=device),
                wp.array([0], dtype=wp.int32, device=device),
                wp.array([0], dtype=wp.int32, device=device),
                wp.array([1], dtype=wp.int32, device=device),
                wp.array([max_constraints], dtype=wp.int32, device=device),
                lower,
                jacobian,
                lower,
                jacobian,
                wp.zeros((1, max_constraints), dtype=wp.float32, device=device),
                wp.array(
                    [[PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION, PGS_CONSTRAINT_TYPE_FRICTION]],
                    dtype=wp.int32,
                    device=device,
                ),
                wp.array([[-1, 0, 0]], dtype=wp.int32, device=device),
                wp.array([[0.0, 0.5, 0.5]], dtype=wp.float32, device=device),
                3,
                1.0,
                2,
                0,
            ],
            outputs=[diagonal, impulses, velocity],
            block_dim=32,
            device=device,
        )
        wp.synchronize_device(device)

        np.testing.assert_allclose(diagonal.numpy()[0], np.ones(max_constraints), rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(impulses.numpy()[0], [1.0, -0.5, 0.0], rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(velocity.numpy(), [0.0, 0.5, 0.0], rtol=0.0, atol=1.0e-6)


if __name__ == "__main__":
    unittest.main()
