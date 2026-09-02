# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from newton._src.sim.enums import JointType
from newton._src.solvers.feather_pgs.kernels import (
    build_augmented_joint_rows_and_apply_tau,
    eval_rigid_tau,
    eval_rigid_tau_and_augmented_drives,
)


class TestFeatherPGSAugmentedDrives(unittest.TestCase):
    def test_fused_builder_clamps_and_accumulates_drive_torque(self):
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
                wp.array([1.5, 10.0, 2.5], dtype=wp.float32, device=device),
                3,
                0.1,
            ],
            outputs=[row_counts, row_dof_index, row_stiffness, limit_counts, joint_tau],
            device=device,
        )
        wp.synchronize_device(device)

        np.testing.assert_array_equal(row_counts.numpy(), [2])
        np.testing.assert_array_equal(row_dof_index.numpy()[:2], [0, 2])
        np.testing.assert_allclose(row_stiffness.numpy()[:2], [0.2, 0.4], rtol=0.0, atol=1.0e-7)
        np.testing.assert_allclose(joint_tau.numpy(), [3.5, -4.0, 3.5], rtol=0.0, atol=1.0e-6)

    def test_combined_tau_and_drive_kernel_matches_sequential_kernels(self):
        device = wp.get_device()
        articulation_start = wp.array([0, 2], dtype=wp.int32, device=device)
        articulation_joint_end = wp.array([2], dtype=wp.int32, device=device)
        articulation_dof_start = wp.array([0], dtype=wp.int32, device=device)
        articulation_dof_count = wp.array([2], dtype=wp.int32, device=device)
        joint_type = wp.array([JointType.REVOLUTE, JointType.REVOLUTE], dtype=wp.int32, device=device)
        joint_parent = wp.array([-1, 0], dtype=wp.int32, device=device)
        joint_child = wp.array([0, 1], dtype=wp.int32, device=device)
        joint_articulation = wp.zeros(2, dtype=wp.int32, device=device)
        joint_start = wp.array([0, 1], dtype=wp.int32, device=device)
        joint_dof_dim = wp.array([[0, 1], [0, 1]], dtype=wp.int32, device=device)
        joint_f = wp.array([0.4, -0.2], dtype=wp.float32, device=device)
        joint_q = wp.array([0.25, -0.35], dtype=wp.float32, device=device)
        joint_qd = wp.array([0.1, -0.2], dtype=wp.float32, device=device)
        spring_stiffness = wp.array([1.5, 0.75], dtype=wp.float32, device=device)
        spring_ref = wp.array([0.0, 0.1], dtype=wp.float32, device=device)
        damping = wp.array([0.2, 0.3], dtype=wp.float32, device=device)
        joint_S_s = wp.array(
            [wp.spatial_vector(0.0, 0.0, 1.0, 0.0, 0.0, 0.0)] * 2,
            dtype=wp.spatial_vector,
            device=device,
        )
        body_fb_s = wp.array(
            [
                wp.spatial_vector(0.1, 0.2, 0.3, 1.0, 1.5, -0.5),
                wp.spatial_vector(-0.2, 0.4, -0.1, -0.5, 0.25, 0.75),
            ],
            dtype=wp.spatial_vector,
            device=device,
        )
        body_f_ext = wp.array(
            [
                wp.spatial_vector(0.05, 0.0, -0.1, 0.0, 0.25, 0.0),
                wp.spatial_vector(0.0, -0.05, 0.1, 0.15, 0.0, -0.2),
            ],
            dtype=wp.spatial_vector,
            device=device,
        )
        body_flags = wp.zeros(2, dtype=wp.int32, device=device)
        body_q = wp.array([wp.transform_identity()] * 2, dtype=wp.transform, device=device)
        body_com = wp.array([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]], dtype=wp.vec3, device=device)
        articulation_origin = wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3, device=device)
        target_ke = wp.array([10.0, 20.0], dtype=wp.float32, device=device)
        target_kd = wp.array([1.0, 2.0], dtype=wp.float32, device=device)
        target_q = wp.array([0.1, -0.1], dtype=wp.float32, device=device)
        target_qd = wp.array([0.0, 0.05], dtype=wp.float32, device=device)
        effort_limit = wp.array([1.25, 2.5], dtype=wp.float32, device=device)

        sequential_body_ft = wp.zeros(2, dtype=wp.spatial_vector, device=device)
        sequential_tau = wp.zeros(2, dtype=wp.float32, device=device)
        sequential_counts = wp.zeros(1, dtype=wp.int32, device=device)
        sequential_indices = wp.zeros(2, dtype=wp.int32, device=device)
        sequential_stiffness = wp.zeros(2, dtype=wp.float32, device=device)
        sequential_limit_counts = wp.zeros(1, dtype=wp.int32, device=device)
        wp.launch(
            eval_rigid_tau,
            dim=1,
            inputs=[
                articulation_start,
                articulation_joint_end,
                joint_type,
                joint_parent,
                joint_child,
                joint_articulation,
                joint_start,
                joint_start,
                joint_dof_dim,
                joint_f,
                joint_q,
                joint_qd,
                spring_stiffness,
                spring_ref,
                damping,
                joint_S_s,
                body_fb_s,
                body_f_ext,
                body_flags,
                body_q,
                body_com,
                articulation_origin,
            ],
            outputs=[sequential_body_ft, sequential_tau],
            device=device,
        )
        wp.launch(
            build_augmented_joint_rows_and_apply_tau,
            dim=1,
            inputs=[
                articulation_start,
                articulation_dof_start,
                articulation_dof_count,
                joint_type,
                joint_start,
                joint_start,
                joint_dof_dim,
                target_ke,
                target_kd,
                joint_q,
                joint_qd,
                target_q,
                target_qd,
                effort_limit,
                2,
                0.1,
            ],
            outputs=[
                sequential_counts,
                sequential_indices,
                sequential_stiffness,
                sequential_limit_counts,
                sequential_tau,
            ],
            device=device,
        )

        combined_body_ft = wp.zeros_like(sequential_body_ft)
        combined_tau = wp.zeros_like(sequential_tau)
        combined_counts = wp.zeros_like(sequential_counts)
        combined_indices = wp.zeros_like(sequential_indices)
        combined_stiffness = wp.zeros_like(sequential_stiffness)
        combined_limit_counts = wp.zeros_like(sequential_limit_counts)
        wp.launch(
            eval_rigid_tau_and_augmented_drives,
            dim=1,
            inputs=[
                articulation_start,
                articulation_joint_end,
                articulation_dof_count,
                joint_type,
                joint_parent,
                joint_child,
                joint_articulation,
                joint_start,
                joint_start,
                joint_dof_dim,
                joint_f,
                joint_q,
                joint_qd,
                spring_stiffness,
                spring_ref,
                damping,
                joint_S_s,
                body_fb_s,
                body_f_ext,
                body_flags,
                body_q,
                body_com,
                articulation_origin,
                target_ke,
                target_kd,
                target_q,
                target_qd,
                effort_limit,
                2,
                0.1,
            ],
            outputs=[
                combined_body_ft,
                combined_counts,
                combined_indices,
                combined_stiffness,
                combined_limit_counts,
                combined_tau,
            ],
            device=device,
        )
        wp.synchronize_device(device)

        np.testing.assert_array_equal(combined_body_ft.numpy(), sequential_body_ft.numpy())
        np.testing.assert_array_equal(combined_tau.numpy(), sequential_tau.numpy())
        np.testing.assert_array_equal(combined_counts.numpy(), sequential_counts.numpy())
        np.testing.assert_array_equal(combined_indices.numpy(), sequential_indices.numpy())
        np.testing.assert_array_equal(combined_stiffness.numpy(), sequential_stiffness.numpy())
        np.testing.assert_array_equal(combined_limit_counts.numpy(), sequential_limit_counts.numpy())


if __name__ == "__main__":
    unittest.main()
