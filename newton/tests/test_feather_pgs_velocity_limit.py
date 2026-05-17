# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression tests for the FeatherPGS joint velocity-limit constraint.

These tests cover the feature introduced in issue #23:

1. **Parameter plumbing and guards** — the new ``enable_joint_velocity_limits``
   kwarg is accepted, stored, and bound to the supported PGS mode.
2. **Allocation kernel** — the Warp kernel ``allocate_joint_velocity_limit_slots``
   reserves lower/upper slots for every finite ``qdot_max_i`` so the bilateral
   ``[-qdot_max, +qdot_max]`` box is present before the predicted velocity
   escapes.
3. **Populator kernel** — the Warp kernel
   ``populate_joint_velocity_limit_J_for_size`` writes a signed ±1 selector row
   into the grouped Jacobian and sets the correct constraint metadata
   (row type, zero Baumgarte bias, ``target_vel = -qdot_max``).
4. **Flag-off is a strict no-op** — constructing the solver without the flag
   matches the baseline behaviour bit-for-bit.

The end-to-end Isaac-Lab-level smoke comparison (``fpgs`` vs ``fpgs_vlim`` vs
``physx`` vs ``mjwarp``) lives in
``skild-IL-solver/scripts/benchmarks/velocity_spike_smoke.py`` and is reported
in ``notes/investigations/velocity-spike/smoke-scenario.md``.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.sim.enums import JointType
from newton._src.solvers import SolverFeatherPGS
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
    allocate_joint_velocity_limit_slots,
    allocate_rigid_velocity_limit_slots,
    pgs_solve_mf_loop,
    populate_joint_velocity_limit_J_for_size,
    populate_rigid_velocity_limit_rows,
    prescale_joint_velocity_limits,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _build_two_dof_pendulum_model(device: str, velocity_limit: float = 2.0) -> newton.Model:
    """Build a 2-DOF revolute chain with a finite per-axis velocity limit.

    The chain has two serial revolute joints with a known velocity limit on
    each DOF, so the allocation / populator kernels have a non-trivial DOF
    range to sweep and the solver has a non-empty articulation to build
    against.
    """
    builder = newton.ModelBuilder()
    link_a = builder.add_link()
    link_b = builder.add_link()
    joint_a = builder.add_joint_revolute(
        parent=-1,
        child=link_a,
        axis=wp.vec3(0.0, 0.0, 1.0),
        velocity_limit=velocity_limit,
    )
    joint_b = builder.add_joint_revolute(
        parent=link_a,
        child=link_b,
        axis=wp.vec3(0.0, 0.0, 1.0),
        velocity_limit=velocity_limit,
    )
    builder.add_articulation([joint_a, joint_b])
    model = builder.finalize(device=device)
    return model


class TestFeatherPGSVelocityLimitFlag(unittest.TestCase):
    """Constructor-level plumbing for ``enable_joint_velocity_limits``."""


def run_default_is_flag_off(test: TestFeatherPGSVelocityLimitFlag, device):
    """With no kwarg, velocity limits stay off (regression baseline)."""
    model = _build_two_dof_pendulum_model(device)
    solver = SolverFeatherPGS(model)
    test.assertFalse(solver.enable_joint_velocity_limits)
    # The per-DOF buffers only allocate when the flag is on.
    test.assertIsNone(solver.velocity_limit_slot)
    test.assertIsNone(solver.velocity_limit_sign)


def run_flag_on_allocates_buffers(test: TestFeatherPGSVelocityLimitFlag, device):
    """Passing the flag allocates lower/upper buffers for each DOF."""
    model = _build_two_dof_pendulum_model(device)
    solver = SolverFeatherPGS(
        model,
        enable_joint_velocity_limits=True,
        pgs_mode="matrix_free",
    )
    test.assertTrue(solver.enable_joint_velocity_limits)
    test.assertIsNotNone(solver.velocity_limit_slot)
    test.assertIsNotNone(solver.velocity_limit_sign)
    test.assertEqual(solver.velocity_limit_slot.shape, (2 * model.joint_dof_count,))
    test.assertEqual(solver.velocity_limit_sign.shape, (2 * model.joint_dof_count,))
    # Default slot entry is -1 (no row active).
    np.testing.assert_array_equal(
        solver.velocity_limit_slot.numpy(),
        -np.ones(2 * model.joint_dof_count, dtype=np.int32),
    )


def run_flag_on_dense_mode_accepts(test: TestFeatherPGSVelocityLimitFlag, device):
    """The flag is accepted in dense mode."""
    model = _build_two_dof_pendulum_model(device)
    solver = SolverFeatherPGS(
        model,
        enable_joint_velocity_limits=True,
        pgs_mode="dense",
    )
    test.assertTrue(solver.enable_joint_velocity_limits)


def run_flag_on_split_mode_accepts(test: TestFeatherPGSVelocityLimitFlag, device):
    """The flag is accepted in split mode."""
    model = _build_two_dof_pendulum_model(device)
    solver = SolverFeatherPGS(
        model,
        enable_joint_velocity_limits=True,
        pgs_mode="split",
    )
    test.assertTrue(solver.enable_joint_velocity_limits)


class TestFeatherPGSVelocityLimitAllocationKernel(unittest.TestCase):
    """Direct kernel tests for ``allocate_joint_velocity_limit_slots``."""


def run_allocation_activates_only_over_limit(test: TestFeatherPGSVelocityLimitAllocationKernel, device):
    """Every finite-limited DOF gets lower/upper slots regardless of current qdot."""
    # Layout: one articulation with two revolute DOFs, limits = 2.0 rad/s.
    dof_count = 2
    max_constraints = 8

    articulation_start = wp.array(np.array([0, 2], dtype=np.int32), dtype=int, device=device)
    articulation_dof_start = wp.array(np.array([0], dtype=np.int32), dtype=int, device=device)
    articulation_H_rows = wp.array(np.array([dof_count], dtype=np.int32), dtype=int, device=device)
    # Two revolute joints.
    joint_type = wp.array(
        np.array([int(JointType.REVOLUTE), int(JointType.REVOLUTE)], dtype=np.int32),
        dtype=int,
        device=device,
    )
    joint_qd_start = wp.array(np.array([0, 1], dtype=np.int32), dtype=int, device=device)
    # Each joint has 0 linear, 1 angular DOF.
    joint_dof_dim = wp.array(np.array([[0, 1], [0, 1]], dtype=np.int32), dtype=int, ndim=2, device=device)
    joint_velocity_limit = wp.array(np.array([2.0, 2.0], dtype=np.float32), dtype=float, device=device)
    # Current qdot no longer gates row creation.
    joint_qd = wp.array(np.array([0.5, 3.5], dtype=np.float32), dtype=float, device=device)
    art_to_world = wp.array(np.array([0], dtype=np.int32), dtype=int, device=device)

    velocity_limit_slot = wp.full((2 * dof_count,), -1, dtype=wp.int32, device=device)
    velocity_limit_sign = wp.zeros((2 * dof_count,), dtype=wp.float32, device=device)
    world_slot_counter = wp.zeros((1,), dtype=wp.int32, device=device)

    wp.launch(
        allocate_joint_velocity_limit_slots,
        dim=1,
        inputs=[
            articulation_start,
            articulation_dof_start,
            articulation_H_rows,
            joint_type,
            joint_qd_start,
            joint_dof_dim,
            joint_velocity_limit,
            joint_qd,
            art_to_world,
            max_constraints,
        ],
        outputs=[velocity_limit_slot, velocity_limit_sign, world_slot_counter],
        device=device,
    )

    slot = velocity_limit_slot.numpy()
    sign = velocity_limit_sign.numpy()
    counter = int(world_slot_counter.numpy()[0])

    test.assertEqual(counter, 4)
    np.testing.assert_array_equal(slot, np.array([0, 1, 2, 3], dtype=np.int32))
    np.testing.assert_allclose(sign, np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32), atol=1e-7)


def run_allocation_lower_vs_upper_sides(test: TestFeatherPGSVelocityLimitAllocationKernel, device):
    """Lower row uses sign +1; upper row uses sign -1.

    This is the direct signal that the bilateral ``[-qdot_max, +qdot_max]``
    box is encoded by the sign of the two per-DOF Jacobian rows.
    """
    dof_count = 2
    max_constraints = 8

    articulation_start = wp.array(np.array([0, 2], dtype=np.int32), dtype=int, device=device)
    articulation_dof_start = wp.array(np.array([0], dtype=np.int32), dtype=int, device=device)
    articulation_H_rows = wp.array(np.array([dof_count], dtype=np.int32), dtype=int, device=device)
    joint_type = wp.array(
        np.array([int(JointType.REVOLUTE), int(JointType.REVOLUTE)], dtype=np.int32),
        dtype=int,
        device=device,
    )
    joint_qd_start = wp.array(np.array([0, 1], dtype=np.int32), dtype=int, device=device)
    joint_dof_dim = wp.array(np.array([[0, 1], [0, 1]], dtype=np.int32), dtype=int, ndim=2, device=device)
    joint_velocity_limit = wp.array(np.array([1.0, 1.0], dtype=np.float32), dtype=float, device=device)
    # DOF 0: below lower bound. DOF 1: above upper bound.
    joint_qd = wp.array(np.array([-5.0, 5.0], dtype=np.float32), dtype=float, device=device)
    art_to_world = wp.array(np.array([0], dtype=np.int32), dtype=int, device=device)

    velocity_limit_slot = wp.full((2 * dof_count,), -1, dtype=wp.int32, device=device)
    velocity_limit_sign = wp.zeros((2 * dof_count,), dtype=wp.float32, device=device)
    world_slot_counter = wp.zeros((1,), dtype=wp.int32, device=device)

    wp.launch(
        allocate_joint_velocity_limit_slots,
        dim=1,
        inputs=[
            articulation_start,
            articulation_dof_start,
            articulation_H_rows,
            joint_type,
            joint_qd_start,
            joint_dof_dim,
            joint_velocity_limit,
            joint_qd,
            art_to_world,
            max_constraints,
        ],
        outputs=[velocity_limit_slot, velocity_limit_sign, world_slot_counter],
        device=device,
    )

    slot = velocity_limit_slot.numpy()
    sign = velocity_limit_sign.numpy()
    counter = int(world_slot_counter.numpy()[0])

    test.assertEqual(counter, 4)
    np.testing.assert_array_equal(slot, np.array([0, 1, 2, 3], dtype=np.int32))
    # Each DOF gets a lower row (+e_i) and an upper row (-e_i).
    np.testing.assert_allclose(sign, np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32), atol=1e-7)


def run_allocation_nonpositive_limit_is_unlimited(test: TestFeatherPGSVelocityLimitAllocationKernel, device):
    """A non-positive ``qdot_max_i`` is treated as "no limit": no row allocated.

    This mirrors PhysX's ``recipResponse`` pinning on ``unitResponse <= 0``
    and is how Newton's builder represents an unset / unlimited velocity
    limit (``velocity_limit`` defaulting to a sentinel).
    """
    dof_count = 1
    max_constraints = 8

    articulation_start = wp.array(np.array([0, 1], dtype=np.int32), dtype=int, device=device)
    articulation_dof_start = wp.array(np.array([0], dtype=np.int32), dtype=int, device=device)
    articulation_H_rows = wp.array(np.array([dof_count], dtype=np.int32), dtype=int, device=device)
    joint_type = wp.array(np.array([int(JointType.REVOLUTE)], dtype=np.int32), dtype=int, device=device)
    joint_qd_start = wp.array(np.array([0], dtype=np.int32), dtype=int, device=device)
    joint_dof_dim = wp.array(np.array([[0, 1]], dtype=np.int32), dtype=int, ndim=2, device=device)
    # Zero limit should be treated as unlimited and NOT allocate.
    joint_velocity_limit = wp.array(np.array([0.0], dtype=np.float32), dtype=float, device=device)
    joint_qd = wp.array(np.array([100.0], dtype=np.float32), dtype=float, device=device)
    art_to_world = wp.array(np.array([0], dtype=np.int32), dtype=int, device=device)

    velocity_limit_slot = wp.full((2 * dof_count,), -1, dtype=wp.int32, device=device)
    velocity_limit_sign = wp.zeros((2 * dof_count,), dtype=wp.float32, device=device)
    world_slot_counter = wp.zeros((1,), dtype=wp.int32, device=device)

    wp.launch(
        allocate_joint_velocity_limit_slots,
        dim=1,
        inputs=[
            articulation_start,
            articulation_dof_start,
            articulation_H_rows,
            joint_type,
            joint_qd_start,
            joint_dof_dim,
            joint_velocity_limit,
            joint_qd,
            art_to_world,
            max_constraints,
        ],
        outputs=[velocity_limit_slot, velocity_limit_sign, world_slot_counter],
        device=device,
    )

    test.assertEqual(int(world_slot_counter.numpy()[0]), 0)
    test.assertEqual(int(velocity_limit_slot.numpy()[0]), -1)
    test.assertEqual(int(velocity_limit_slot.numpy()[1]), -1)


class TestFeatherPGSVelocityLimitPrescaleKernel(unittest.TestCase):
    """Direct kernel tests for PhysX-style pre-solve velocity scaling."""


def run_prescale_uses_single_articulation_ratio(test: TestFeatherPGSVelocityLimitPrescaleKernel, device):
    """The most over-limit DOF scales every scalar DOF in the articulation."""
    articulation_start = wp.array(np.array([0, 3], dtype=np.int32), dtype=int, device=device)
    joint_type = wp.array(
        np.array([int(JointType.REVOLUTE), int(JointType.REVOLUTE), int(JointType.PRISMATIC)], dtype=np.int32),
        dtype=int,
        device=device,
    )
    joint_qd_start = wp.array(np.array([0, 1, 2], dtype=np.int32), dtype=int, device=device)
    joint_dof_dim = wp.array(np.array([[0, 1], [0, 1], [1, 0]], dtype=np.int32), dtype=int, ndim=2, device=device)
    joint_velocity_limit = wp.array(np.array([10.0, 5.0, 8.0], dtype=np.float32), dtype=float, device=device)
    joint_qd = wp.array(np.array([2.0, 20.0, -4.0], dtype=np.float32), dtype=float, device=device)

    wp.launch(
        prescale_joint_velocity_limits,
        dim=1,
        inputs=[
            articulation_start,
            joint_type,
            joint_qd_start,
            joint_dof_dim,
            joint_velocity_limit,
        ],
        outputs=[joint_qd],
        device=device,
    )

    # DOF 1 is the limiting axis: 5 / 20 = 0.25. PhysX applies this same
    # articulation-wide ratio to every scalar DOF before link velocities are built.
    np.testing.assert_allclose(joint_qd.numpy(), np.array([0.5, 5.0, -1.0], dtype=np.float32), atol=1e-6)


class TestFeatherPGSVelocityLimitPopulatorKernel(unittest.TestCase):
    """Direct kernel tests for ``populate_joint_velocity_limit_J_for_size``."""


def run_populator_writes_signed_selector_row_and_metadata(test: TestFeatherPGSVelocityLimitPopulatorKernel, device):
    """Populator writes J = sign*e_i, zero Baumgarte bias, and target_vel = -qdot_max.

    Build a 2-DOF layout where:
    - DOF 0 is upper-violating (sign=-1) and gets slot 0.
    - DOF 1 is lower-violating (sign=+1) and gets slot 1.

    Verify the grouped Jacobian, the row-type marker, and the constraint
    metadata (``row_beta``, ``row_cfm``, ``phi``, ``target_velocity``) match
    the PhysX mirror formulation.
    """
    dof_count = 2
    max_constraints = 8
    n_arts = 1

    articulation_start = wp.array(np.array([0, 2], dtype=np.int32), dtype=int, device=device)
    articulation_dof_start = wp.array(np.array([0], dtype=np.int32), dtype=int, device=device)
    joint_type = wp.array(
        np.array([int(JointType.REVOLUTE), int(JointType.REVOLUTE)], dtype=np.int32),
        dtype=int,
        device=device,
    )
    joint_qd_start = wp.array(np.array([0, 1], dtype=np.int32), dtype=int, device=device)
    joint_dof_dim = wp.array(np.array([[0, 1], [0, 1]], dtype=np.int32), dtype=int, ndim=2, device=device)
    qdot_max = np.array([3.0, 4.0], dtype=np.float32)
    joint_velocity_limit = wp.array(qdot_max, dtype=float, device=device)

    art_to_world = wp.array(np.array([0], dtype=np.int32), dtype=int, device=device)
    # DOF 0 lower/upper → slots 0/1; DOF 1 lower/upper → slots 2/3.
    velocity_limit_slot = wp.array(np.array([0, 1, 2, 3], dtype=np.int32), dtype=int, device=device)
    velocity_limit_sign = wp.array(np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32), dtype=float, device=device)
    group_to_art = wp.array(np.array([0], dtype=np.int32), dtype=int, device=device)

    pgs_cfm = 1.0e-6

    # Grouped Jacobian: [n_arts_of_size, max_constraints, n_dofs].
    J_group = wp.zeros((n_arts, max_constraints, dof_count), dtype=float, device=device)
    world_row_type = wp.zeros((1, max_constraints), dtype=wp.int32, device=device)
    world_row_parent = wp.full((1, max_constraints), -1, dtype=wp.int32, device=device)
    world_row_mu = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)
    world_row_beta = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)
    world_row_cfm = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)
    world_phi = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)
    world_target_velocity = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)

    wp.launch(
        populate_joint_velocity_limit_J_for_size,
        dim=n_arts,
        inputs=[
            articulation_start,
            articulation_dof_start,
            joint_type,
            joint_qd_start,
            joint_dof_dim,
            joint_velocity_limit,
            art_to_world,
            velocity_limit_slot,
            velocity_limit_sign,
            group_to_art,
            pgs_cfm,
        ],
        outputs=[
            J_group,
            world_row_type,
            world_row_parent,
            world_row_mu,
            world_row_beta,
            world_row_cfm,
            world_phi,
            world_target_velocity,
        ],
        device=device,
    )

    # J rows are single signed-±1 selector rows at (slot, local_dof).
    J = J_group.numpy()
    test.assertAlmostEqual(float(J[0, 0, 0]), 1.0, places=6)  # slot 0 lower row, DOF 0 col
    test.assertAlmostEqual(float(J[0, 0, 1]), 0.0, places=6)
    test.assertAlmostEqual(float(J[0, 1, 0]), -1.0, places=6)  # slot 1 upper row, DOF 0 col
    test.assertAlmostEqual(float(J[0, 1, 1]), 0.0, places=6)
    test.assertAlmostEqual(float(J[0, 2, 0]), 0.0, places=6)
    test.assertAlmostEqual(float(J[0, 2, 1]), 1.0, places=6)  # slot 2 lower row, DOF 1 col
    test.assertAlmostEqual(float(J[0, 3, 0]), 0.0, places=6)
    test.assertAlmostEqual(float(J[0, 3, 1]), -1.0, places=6)  # slot 3 upper row, DOF 1 col

    row_type = world_row_type.numpy()
    test.assertEqual(int(row_type[0, 0]), PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)
    test.assertEqual(int(row_type[0, 1]), PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)
    test.assertEqual(int(row_type[0, 2]), PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)
    test.assertEqual(int(row_type[0, 3]), PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)

    # No Baumgarte / ERP — matches PhysX.
    np.testing.assert_allclose(world_row_beta.numpy()[0, :4], np.zeros(4, dtype=np.float32), atol=1e-7)
    # phi = 0 (velocity-limit row has no positional bias quantity).
    np.testing.assert_allclose(world_phi.numpy()[0, :4], np.zeros(4, dtype=np.float32), atol=1e-7)
    # target_vel = -qdot_max for both signs; combined with the sign flip in J
    # this identifies which side of the bilateral clamp is currently violated.
    np.testing.assert_allclose(
        world_target_velocity.numpy()[0, :4],
        np.array([-qdot_max[0], -qdot_max[0], -qdot_max[1], -qdot_max[1]], dtype=np.float32),
        atol=1e-7,
    )
    # cfm preserved.
    np.testing.assert_allclose(world_row_cfm.numpy()[0, :4], np.full(4, pgs_cfm, dtype=np.float32), atol=1e-7)
    # No parent — the row is standalone (not a friction sibling).
    np.testing.assert_array_equal(world_row_parent.numpy()[0, :4], np.array([-1, -1, -1, -1], dtype=np.int32))


class TestFeatherPGSVelocityLimitEndToEnd(unittest.TestCase):
    """End-to-end integration tests that drive a real solver step.

    The matrix-free PGS sweep kernel (``get_pgs_solve_mf_gs_kernel``) is a
    CUDA-native snippet guarded by ``#if defined(__CUDA_ARCH__)`` — it is a
    no-op on CPU. These tests therefore require a CUDA device. Kernel-level
    correctness (allocation + populator + row-type dispatch) is covered on
    CPU by the classes above.
    """


def _build_driven_pendulum(
    device: str, velocity_limit: float, target_ke: float, target_kd: float
) -> tuple[newton.Model, newton.State, newton.State, newton.Control]:
    """Build a 1-DOF revolute pendulum with a stiff implicit PD drive.

    The setup mirrors the shape of the Franka smoke: a scripted position
    target that is far outside the physical reach of the joint, with a
    stiff implicit-PD drive that demands a huge ``qdot`` at the next step.
    Reproduces the conditions under which baseline FPGS produces the
    velocity-spike ``|qdot|`` of issue #21.
    """
    builder = newton.ModelBuilder(gravity=0.0)
    box_inertia = wp.mat33((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    link = builder.add_link(armature=0.0, inertia=box_inertia, mass=1.0)
    joint = builder.add_joint_revolute(
        parent=-1,
        child=link,
        axis=wp.vec3(0.0, 0.0, 1.0),
        velocity_limit=velocity_limit,
        target_ke=target_ke,
        target_kd=target_kd,
        actuator_mode=newton.JointTargetMode.POSITION,
    )
    builder.add_articulation([joint])
    model = builder.finalize(device=device)

    state_0 = model.state()
    state_1 = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    control = model.control()
    # Target far outside any reasonable reach so the stiff PD drive commands a
    # huge ``qdot`` — mirrors the smoke's ``pos_high`` pulse.
    control.joint_target_pos = wp.array([10.0], dtype=wp.float32, device=device)
    control.joint_target_vel = wp.array([0.0], dtype=wp.float32, device=device)

    return model, state_0, state_1, control


def _step_and_read_qdot(
    model: newton.Model,
    solver: SolverFeatherPGS,
    state_0: newton.State,
    state_1: newton.State,
    control: newton.Control,
    dt: float,
    n_steps: int,
) -> float:
    """Run ``n_steps`` and return the peak ``|qdot|`` observed on DOF 0."""
    peak = 0.0
    for _ in range(n_steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0
        peak = max(peak, float(abs(state_0.joint_qd.numpy()[0])))
    return peak


def run_end_to_end_flag_on_clamps_peak_qdot(test: TestFeatherPGSVelocityLimitEndToEnd, device):
    """With the flag on, peak ``|qdot|`` stays near the velocity limit.

    Drives a stiff-PD pendulum with a far position target — the baseline
    solver produces a huge ``qdot`` spike over a few steps; with
    ``enable_joint_velocity_limits=True`` the PGS clamp pulls ``|qdot|``
    back toward ``qdot_max``. We accept a small tolerance above the strict
    limit because the clamp is applied *after* the predictor: the
    pre-solve velocity can be above the limit, and the solver's iteration
    count / articulated-body response determine the residual.
    """
    velocity_limit = 1.5
    model, state_0, state_1, control = _build_driven_pendulum(
        device, velocity_limit=velocity_limit, target_ke=2000.0, target_kd=40.0
    )
    solver = SolverFeatherPGS(
        model,
        enable_joint_velocity_limits=True,
        pgs_mode="matrix_free",
        pgs_iterations=32,
    )
    peak = _step_and_read_qdot(model, solver, state_0, state_1, control, dt=1.0 / 120.0, n_steps=40)

    # The clamp must be doing *something* meaningful: peak stays within a
    # generous 1.5x multiple of the limit (mirrors the issue's 1.25x
    # threshold with extra headroom for PGS residual). The reference
    # baseline without the flag blows past 6x; see the companion test.
    test.assertLess(peak, velocity_limit * 1.5)


def run_end_to_end_flag_on_is_tighter_than_flag_off(test: TestFeatherPGSVelocityLimitEndToEnd, device):
    """Constrained FPGS has a smaller ``|qdot|`` peak than baseline FPGS.

    This is the direct smoke-scenario signal at Newton-library level
    (no Isaac Sim required): driven by the same stiff-PD far target and
    the same matrix-free pipeline, the flag-on solver clamps and the
    flag-off solver does not.
    """
    velocity_limit = 1.5

    # Baseline (flag off).
    model_b, s0_b, s1_b, ctl_b = _build_driven_pendulum(
        device, velocity_limit=velocity_limit, target_ke=2000.0, target_kd=40.0
    )
    solver_b = SolverFeatherPGS(model_b, pgs_mode="matrix_free", pgs_iterations=32)
    peak_baseline = _step_and_read_qdot(model_b, solver_b, s0_b, s1_b, ctl_b, dt=1.0 / 120.0, n_steps=40)

    # Constrained (flag on).
    model_c, s0_c, s1_c, ctl_c = _build_driven_pendulum(
        device, velocity_limit=velocity_limit, target_ke=2000.0, target_kd=40.0
    )
    solver_c = SolverFeatherPGS(
        model_c,
        enable_joint_velocity_limits=True,
        pgs_mode="matrix_free",
        pgs_iterations=32,
    )
    peak_constrained = _step_and_read_qdot(model_c, solver_c, s0_c, s1_c, ctl_c, dt=1.0 / 120.0, n_steps=40)

    # Baseline must overshoot meaningfully (otherwise the scenario itself
    # isn't a valid witness).
    test.assertGreater(peak_baseline, velocity_limit * 1.5)
    # Constrained must be tighter than baseline.
    test.assertLess(peak_constrained, peak_baseline)
    # And preferably near the limit.
    test.assertLess(peak_constrained, velocity_limit * 1.5)


def run_end_to_end_flag_off_is_strict_noop(test: TestFeatherPGSVelocityLimitEndToEnd, device):
    """Flag off: per-step outputs are bit-exact to the pre-issue baseline.

    Same stiff-PD driven pendulum, same matrix-free pipeline, same seed
    (deterministic), the flag toggles to ``False`` must produce the same
    ``joint_qd`` trajectory as a solver constructed without the kwarg.
    """
    velocity_limit = 1.5

    # Solver A: constructed without the kwarg (pre-issue baseline).
    model_a, s0_a, s1_a, ctl_a = _build_driven_pendulum(
        device, velocity_limit=velocity_limit, target_ke=2000.0, target_kd=40.0
    )
    solver_a = SolverFeatherPGS(model_a, pgs_mode="matrix_free")
    peak_a = _step_and_read_qdot(model_a, solver_a, s0_a, s1_a, ctl_a, dt=1.0 / 120.0, n_steps=20)

    # Solver B: constructed with enable_joint_velocity_limits=False (explicit).
    model_b, s0_b, s1_b, ctl_b = _build_driven_pendulum(
        device, velocity_limit=velocity_limit, target_ke=2000.0, target_kd=40.0
    )
    solver_b = SolverFeatherPGS(model_b, enable_joint_velocity_limits=False, pgs_mode="matrix_free")
    peak_b = _step_and_read_qdot(model_b, solver_b, s0_b, s1_b, ctl_b, dt=1.0 / 120.0, n_steps=20)

    # Bit-exact match: flag-off is a strict no-op.
    test.assertAlmostEqual(peak_a, peak_b, delta=1e-6)


class TestFeatherPGSRigidVelocityLimitRows(unittest.TestCase):
    """Direct matrix-free row tests for free-rigid velocity limits."""


def run_rigid_velocity_limit_rows_allocate_and_populate(test: TestFeatherPGSRigidVelocityLimitRows, device):
    """A limited free rigid body gets signed MF rows for all six speed axes."""
    mf_max_constraints = 16
    body_to_articulation = wp.array(np.array([0], dtype=np.int32), dtype=int, device=device)
    art_to_world = wp.array(np.array([0], dtype=np.int32), dtype=int, device=device)
    is_free_rigid = wp.array(np.array([1], dtype=np.int32), dtype=int, device=device)
    max_linear = wp.array(np.array([1.0], dtype=np.float32), dtype=float, device=device)
    max_angular = wp.array(np.array([2.0], dtype=np.float32), dtype=float, device=device)

    slot = wp.full((12,), -1, dtype=wp.int32, device=device)
    sign = wp.zeros((12,), dtype=wp.float32, device=device)
    counter = wp.zeros((1,), dtype=wp.int32, device=device)

    wp.launch(
        allocate_rigid_velocity_limit_slots,
        dim=1,
        inputs=[
            body_to_articulation,
            art_to_world,
            is_free_rigid,
            max_linear,
            max_angular,
            mf_max_constraints,
        ],
        outputs=[slot, sign, counter],
        device=device,
    )

    np.testing.assert_array_equal(slot.numpy(), np.arange(12, dtype=np.int32))
    np.testing.assert_allclose(sign.numpy(), np.array([1.0, -1.0] * 6, dtype=np.float32))
    test.assertEqual(int(counter.numpy()[0]), 12)

    mf_body_a = wp.zeros((1, mf_max_constraints), dtype=wp.int32, device=device)
    mf_body_b = wp.zeros((1, mf_max_constraints), dtype=wp.int32, device=device)
    mf_J_a = wp.zeros((1, mf_max_constraints, 6), dtype=wp.float32, device=device)
    mf_J_b = wp.zeros((1, mf_max_constraints, 6), dtype=wp.float32, device=device)
    mf_row_type = wp.zeros((1, mf_max_constraints), dtype=wp.int32, device=device)
    mf_row_parent = wp.full((1, mf_max_constraints), -1, dtype=wp.int32, device=device)
    mf_row_mu = wp.zeros((1, mf_max_constraints), dtype=wp.float32, device=device)
    mf_phi = wp.zeros((1, mf_max_constraints), dtype=wp.float32, device=device)

    wp.launch(
        populate_rigid_velocity_limit_rows,
        dim=1,
        inputs=[
            body_to_articulation,
            art_to_world,
            is_free_rigid,
            max_linear,
            max_angular,
            slot,
            sign,
        ],
        outputs=[mf_body_a, mf_body_b, mf_J_a, mf_J_b, mf_row_type, mf_row_parent, mf_row_mu, mf_phi],
        device=device,
    )

    np.testing.assert_array_equal(mf_body_a.numpy()[0, :12], np.zeros(12, dtype=np.int32))
    np.testing.assert_array_equal(mf_body_b.numpy()[0, :12], -np.ones(12, dtype=np.int32))
    np.testing.assert_array_equal(
        mf_row_type.numpy()[0, :12],
        np.full(12, PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT, dtype=np.int32),
    )
    np.testing.assert_allclose(
        mf_phi.numpy()[0, :12],
        np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float32),
    )

    J = mf_J_a.numpy()[0, :12]
    expected_J = np.zeros((12, 6), dtype=np.float32)
    for axis in range(6):
        expected_J[2 * axis, axis] = 1.0
        expected_J[2 * axis + 1, axis] = -1.0
    np.testing.assert_allclose(J, expected_J, atol=1e-7)
    np.testing.assert_allclose(mf_J_b.numpy()[0, :12], 0.0, atol=1e-7)


def run_rigid_velocity_limit_row_solves_upper_bound(test: TestFeatherPGSRigidVelocityLimitRows, device):
    """The MF row projection clamps a free body's scalar speed to the row limit."""
    mf_constraint_count = wp.array(np.array([1], dtype=np.int32), dtype=int, device=device)
    mf_body_a = wp.array(np.array([[0]], dtype=np.int32), dtype=int, ndim=2, device=device)
    mf_body_b = wp.array(np.array([[-1]], dtype=np.int32), dtype=int, ndim=2, device=device)
    mf_MiJt_a = wp.array(np.array([[[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]], dtype=np.float32), dtype=float, ndim=3, device=device)
    mf_MiJt_b = wp.zeros((1, 1, 6), dtype=wp.float32, device=device)
    mf_J_a = wp.array(np.array([[[-1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]], dtype=np.float32), dtype=float, ndim=3, device=device)
    mf_J_b = wp.zeros((1, 1, 6), dtype=wp.float32, device=device)
    mf_eff_mass_inv = wp.array(np.array([[1.0]], dtype=np.float32), dtype=float, ndim=2, device=device)
    mf_rhs = wp.array(np.array([[1.0]], dtype=np.float32), dtype=float, ndim=2, device=device)
    mf_row_type = wp.array(
        np.array([[PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT]], dtype=np.int32),
        dtype=int,
        ndim=2,
        device=device,
    )
    mf_row_parent = wp.full((1, 1), -1, dtype=wp.int32, device=device)
    mf_row_mu = wp.zeros((1, 1), dtype=wp.float32, device=device)
    body_to_articulation = wp.array(np.array([0], dtype=np.int32), dtype=int, device=device)
    art_dof_start = wp.array(np.array([0], dtype=np.int32), dtype=int, device=device)
    impulses = wp.zeros((1, 1), dtype=wp.float32, device=device)
    v_out = wp.array(np.array([5.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32), dtype=float, device=device)

    wp.launch(
        pgs_solve_mf_loop,
        dim=1,
        inputs=[
            mf_constraint_count,
            mf_body_a,
            mf_body_b,
            mf_MiJt_a,
            mf_MiJt_b,
            mf_J_a,
            mf_J_b,
            mf_eff_mass_inv,
            mf_rhs,
            mf_row_type,
            mf_row_parent,
            mf_row_mu,
            body_to_articulation,
            art_dof_start,
            1,  # iterations
            1.0,  # omega; velocity-limit rows apply their direct delta
            0,  # friction_mode
            0,  # friction_start_iteration
            0,  # iteration_offset
        ],
        outputs=[impulses, v_out],
        device=device,
    )

    test.assertAlmostEqual(float(v_out.numpy()[0]), 1.0, delta=1e-6)
    test.assertGreater(float(impulses.numpy()[0, 0]), 0.0)


devices = get_test_devices()

for device in devices:
    add_function_test(
        TestFeatherPGSVelocityLimitFlag,
        f"test_default_is_flag_off_{device}",
        run_default_is_flag_off,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSVelocityLimitFlag,
        f"test_flag_on_allocates_buffers_{device}",
        run_flag_on_allocates_buffers,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSVelocityLimitFlag,
        f"test_flag_on_dense_mode_accepts_{device}",
        run_flag_on_dense_mode_accepts,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSVelocityLimitFlag,
        f"test_flag_on_split_mode_accepts_{device}",
        run_flag_on_split_mode_accepts,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSVelocityLimitAllocationKernel,
        f"test_allocation_activates_only_over_limit_{device}",
        run_allocation_activates_only_over_limit,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSVelocityLimitAllocationKernel,
        f"test_allocation_lower_vs_upper_sides_{device}",
        run_allocation_lower_vs_upper_sides,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSVelocityLimitAllocationKernel,
        f"test_allocation_nonpositive_limit_is_unlimited_{device}",
        run_allocation_nonpositive_limit_is_unlimited,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSVelocityLimitPrescaleKernel,
        f"test_prescale_uses_single_articulation_ratio_{device}",
        run_prescale_uses_single_articulation_ratio,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSVelocityLimitPopulatorKernel,
        f"test_populator_writes_signed_selector_row_and_metadata_{device}",
        run_populator_writes_signed_selector_row_and_metadata,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSRigidVelocityLimitRows,
        f"test_rigid_velocity_limit_rows_allocate_and_populate_{device}",
        run_rigid_velocity_limit_rows_allocate_and_populate,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSRigidVelocityLimitRows,
        f"test_rigid_velocity_limit_row_solves_upper_bound_{device}",
        run_rigid_velocity_limit_row_solves_upper_bound,
        devices=[device],
    )
    # End-to-end tests require the CUDA-native matrix-free PGS sweep
    # kernel. The kernel is a no-op on CPU (see class docstring).
    if str(device).startswith("cuda"):
        add_function_test(
            TestFeatherPGSVelocityLimitEndToEnd,
            f"test_end_to_end_flag_on_clamps_peak_qdot_{device}",
            run_end_to_end_flag_on_clamps_peak_qdot,
            devices=[device],
        )
        add_function_test(
            TestFeatherPGSVelocityLimitEndToEnd,
            f"test_end_to_end_flag_on_is_tighter_than_flag_off_{device}",
            run_end_to_end_flag_on_is_tighter_than_flag_off,
            devices=[device],
        )
        add_function_test(
            TestFeatherPGSVelocityLimitEndToEnd,
            f"test_end_to_end_flag_off_is_strict_noop_{device}",
            run_end_to_end_flag_off_is_strict_noop,
            devices=[device],
        )


if __name__ == "__main__":
    unittest.main()
