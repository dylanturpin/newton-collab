# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.sim.enums import JointType
from newton._src.solvers import SolverFeatherPGS
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_JOINT_DOF,
    PGS_JOINT_DOF_HAS_DRIVE,
    PGS_JOINT_DOF_HAS_LOWER_LIMIT,
    PGS_JOINT_DOF_HAS_UPPER_LIMIT,
    PGS_JOINT_DOF_HAS_VELOCITY_LIMIT,
    allocate_joint_dof_slots,
    compute_world_contact_bias,
    populate_joint_dof_J_for_size,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _one_dof_arrays(device, q: float = 0.0):
    return {
        "articulation_start": wp.array(np.array([0, 1], dtype=np.int32), dtype=int, device=device),
        "articulation_dof_start": wp.array(np.array([0], dtype=np.int32), dtype=int, device=device),
        "articulation_H_rows": wp.array(np.array([1], dtype=np.int32), dtype=int, device=device),
        "joint_type": wp.array(np.array([int(JointType.REVOLUTE)], dtype=np.int32), dtype=int, device=device),
        "joint_q_start": wp.array(np.array([0], dtype=np.int32), dtype=int, device=device),
        "joint_qd_start": wp.array(np.array([0], dtype=np.int32), dtype=int, device=device),
        "joint_dof_dim": wp.array(np.array([[0, 1]], dtype=np.int32), dtype=int, ndim=2, device=device),
        "joint_target_ke": wp.array(np.array([20.0], dtype=np.float32), dtype=float, device=device),
        "joint_target_kd": wp.array(np.array([3.0], dtype=np.float32), dtype=float, device=device),
        "joint_effort_limit": wp.array(np.array([100.0], dtype=np.float32), dtype=float, device=device),
        "joint_limit_lower": wp.array(np.array([-0.25], dtype=np.float32), dtype=float, device=device),
        "joint_limit_upper": wp.array(np.array([0.5], dtype=np.float32), dtype=float, device=device),
        "joint_velocity_limit": wp.array(np.array([1.5], dtype=np.float32), dtype=float, device=device),
        "joint_q": wp.array(np.array([q], dtype=np.float32), dtype=float, device=device),
        "joint_target_pos": wp.array(np.array([0.25], dtype=np.float32), dtype=float, device=device),
        "joint_target_vel": wp.array(np.array([0.75], dtype=np.float32), dtype=float, device=device),
        "art_to_world": wp.array(np.array([0], dtype=np.int32), dtype=int, device=device),
        "group_to_art": wp.array(np.array([0], dtype=np.int32), dtype=int, device=device),
    }


class TestFeatherPGSInternalDofSpike(unittest.TestCase):
    pass


def run_internal_dof_collapses_drive_limits_and_velocity_limit_to_one_row(test, device):
    data = _one_dof_arrays(device)
    max_constraints = 8
    joint_dof_slot = wp.full((1,), -1, dtype=wp.int32, device=device)
    joint_dof_flags = wp.zeros((1, max_constraints), dtype=wp.int32, device=device)
    slot_counter = wp.zeros((1,), dtype=wp.int32, device=device)

    wp.launch(
        allocate_joint_dof_slots,
        dim=1,
        inputs=[
            data["articulation_start"],
            data["articulation_dof_start"],
            data["articulation_H_rows"],
            data["joint_type"],
            data["joint_qd_start"],
            data["joint_dof_dim"],
            data["joint_target_ke"],
            data["joint_target_kd"],
            data["joint_limit_lower"],
            data["joint_limit_upper"],
            data["joint_velocity_limit"],
            data["art_to_world"],
            max_constraints,
            1,
            1,
            1,
        ],
        outputs=[joint_dof_slot, joint_dof_flags, slot_counter],
        device=device,
    )

    expected_flags = (
        PGS_JOINT_DOF_HAS_DRIVE
        | PGS_JOINT_DOF_HAS_LOWER_LIMIT
        | PGS_JOINT_DOF_HAS_UPPER_LIMIT
        | PGS_JOINT_DOF_HAS_VELOCITY_LIMIT
    )
    test.assertEqual(int(slot_counter.numpy()[0]), 1)
    test.assertEqual(int(joint_dof_slot.numpy()[0]), 0)
    test.assertEqual(int(joint_dof_flags.numpy()[0, 0]), int(expected_flags))


def run_internal_dof_populates_drive_limit_and_velocity_metadata(test, device):
    data = _one_dof_arrays(device, q=0.0)
    max_constraints = 8
    joint_dof_slot = wp.array(np.array([0], dtype=np.int32), dtype=int, device=device)

    J_group = wp.zeros((1, max_constraints, 1), dtype=float, device=device)
    row_type = wp.zeros((1, max_constraints), dtype=wp.int32, device=device)
    row_parent = wp.full((1, max_constraints), -1, dtype=wp.int32, device=device)
    row_mu = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)
    row_beta = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)
    row_cfm = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)
    phi = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)
    target_velocity = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)
    drive_stiffness = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)
    drive_damping = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)
    drive_geom_error = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)
    drive_max_force = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)
    joint_dof_index = wp.full((1, max_constraints), -1, dtype=wp.int32, device=device)
    joint_dof_flags = wp.array(
        np.array([[15, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int32),
        dtype=int,
        ndim=2,
        device=device,
    )
    joint_dof_lower_rhs = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)
    joint_dof_upper_rhs = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)
    joint_dof_velocity_limit = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)
    joint_dof_lower_impulse = wp.full((1, max_constraints), 2.0, dtype=wp.float32, device=device)
    joint_dof_upper_impulse = wp.full((1, max_constraints), -3.0, dtype=wp.float32, device=device)

    wp.launch(
        populate_joint_dof_J_for_size,
        dim=1,
        inputs=[
            data["articulation_start"],
            data["articulation_dof_start"],
            data["joint_type"],
            data["joint_q_start"],
            data["joint_qd_start"],
            data["joint_dof_dim"],
            data["joint_target_ke"],
            data["joint_target_kd"],
            data["joint_effort_limit"],
            data["joint_limit_lower"],
            data["joint_limit_upper"],
            data["joint_velocity_limit"],
            data["joint_q"],
            data["joint_target_pos"],
            data["joint_target_vel"],
            data["art_to_world"],
            joint_dof_slot,
            data["group_to_art"],
            0.2,
            1.0e-5,
        ],
        outputs=[
            J_group,
            row_type,
            row_parent,
            row_mu,
            row_beta,
            row_cfm,
            phi,
            target_velocity,
            drive_stiffness,
            drive_damping,
            drive_geom_error,
            drive_max_force,
            joint_dof_index,
            joint_dof_flags,
            joint_dof_lower_rhs,
            joint_dof_upper_rhs,
            joint_dof_velocity_limit,
            joint_dof_lower_impulse,
            joint_dof_upper_impulse,
        ],
        device=device,
    )

    test.assertEqual(int(row_type.numpy()[0, 0]), int(PGS_CONSTRAINT_TYPE_JOINT_DOF))
    test.assertEqual(int(joint_dof_index.numpy()[0, 0]), 0)
    np.testing.assert_allclose(J_group.numpy()[0, 0, 0], 1.0, atol=1e-7)
    np.testing.assert_allclose(drive_stiffness.numpy()[0, 0], 20.0, atol=1e-7)
    np.testing.assert_allclose(drive_damping.numpy()[0, 0], 3.0, atol=1e-7)
    np.testing.assert_allclose(drive_geom_error.numpy()[0, 0], 0.25, atol=1e-7)
    np.testing.assert_allclose(drive_max_force.numpy()[0, 0], 100.0, atol=1e-7)
    np.testing.assert_allclose(target_velocity.numpy()[0, 0], 0.75, atol=1e-7)
    np.testing.assert_allclose(joint_dof_lower_rhs.numpy()[0, 0], 0.25, atol=1e-7)
    np.testing.assert_allclose(joint_dof_upper_rhs.numpy()[0, 0], 0.5, atol=1e-7)
    np.testing.assert_allclose(joint_dof_velocity_limit.numpy()[0, 0], 1.5, atol=1e-7)


def run_internal_dof_limit_rhs_preserves_side_sign_convention(test, device):
    data = _one_dof_arrays(device, q=-0.5)
    max_constraints = 8
    constraint_count = wp.array(np.array([1], dtype=np.int32), dtype=int, device=device)
    row_type = wp.array(np.array([[int(PGS_CONSTRAINT_TYPE_JOINT_DOF)] + [0] * 7], dtype=np.int32), dtype=int, ndim=2, device=device)
    phi = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)
    row_beta = wp.full((1, max_constraints), 0.2, dtype=wp.float32, device=device)
    target_velocity = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)
    flags = wp.array(np.array([[6] + [0] * 7], dtype=np.int32), dtype=int, ndim=2, device=device)
    rhs = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)
    lower_rhs = wp.array(np.array([[-0.25] + [0.0] * 7], dtype=np.float32), dtype=float, ndim=2, device=device)
    upper_rhs = wp.array(np.array([[1.0] + [0.0] * 7], dtype=np.float32), dtype=float, ndim=2, device=device)

    wp.launch(
        compute_world_contact_bias,
        dim=1,
        inputs=[constraint_count, max_constraints, phi, row_beta, row_type, target_velocity, flags, 0.01, 1.0, 1.0, 1.0],
        outputs=[rhs, lower_rhs, upper_rhs],
        device=device,
    )

    # Lower violation produces a negative side residual, which the GS update
    # resolves with a non-negative low impulse. The separated upper side stays
    # positive and therefore does not create a negative high impulse.
    test.assertLess(float(lower_rhs.numpy()[0, 0]), 0.0)
    test.assertGreater(float(upper_rhs.numpy()[0, 0]), 0.0)
    np.testing.assert_allclose(rhs.numpy()[0, 0], 0.0, atol=1e-7)


def run_augmented_drive_without_limits_does_not_allocate_internal_row_buffer(test, device):
    builder = newton.ModelBuilder(gravity=0.0)
    link = builder.add_link()
    joint = builder.add_joint_revolute(
        parent=-1,
        child=link,
        axis=wp.vec3(0.0, 0.0, 1.0),
        target_ke=10.0,
        target_kd=1.0,
    )
    builder.add_articulation([joint])
    model = builder.finalize(device=device)
    solver = SolverFeatherPGS(model, pgs_mode="matrix_free", drive_mode="augmented")
    test.assertIsNone(solver.joint_dof_slot)


devices = get_test_devices()
for device in devices:
    add_function_test(
        TestFeatherPGSInternalDofSpike,
        f"test_internal_dof_collapses_drive_limits_and_velocity_limit_to_one_row_{device}",
        run_internal_dof_collapses_drive_limits_and_velocity_limit_to_one_row,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSInternalDofSpike,
        f"test_internal_dof_populates_drive_limit_and_velocity_metadata_{device}",
        run_internal_dof_populates_drive_limit_and_velocity_metadata,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSInternalDofSpike,
        f"test_internal_dof_limit_rhs_preserves_side_sign_convention_{device}",
        run_internal_dof_limit_rhs_preserves_side_sign_convention,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSInternalDofSpike,
        f"test_augmented_drive_without_limits_does_not_allocate_internal_row_buffer_{device}",
        run_augmented_drive_without_limits_does_not_allocate_internal_row_buffer,
        devices=[device],
    )


if __name__ == "__main__":
    unittest.main()
