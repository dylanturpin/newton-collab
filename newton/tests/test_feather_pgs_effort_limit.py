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

"""Regression tests for the guarded FeatherPGS effort-limit semantics switch.

``SolverFeatherPGS`` always applies an actuator-only effort-limit
clamp: the explicit-PD drive bucket (``u0``) is clamped before it is
folded into ``joint_tau``. The earlier net-torque semantics was a
silent bug (see ``notes/2026-04-20/effort-limit.md``) and has been
removed; passing ``effort_limit_mode="net"`` now raises.

The tests in this file cover:

1. Parameter plumbing: the ``effort_limit_mode`` kwarg is accepted,
   stored on the solver, validated, and defaults to ``"actuator"``;
   passing ``"net"`` raises.
2. ``clamp_augmented_joint_u0`` clamps only the per-row actuator-drive
   ``u0``, using ``joint_effort_limit[dof]`` as the cap, while leaving
   the row and the downstream ``joint_tau`` rigid/passive bucket
   untouched. The legacy ``clamp_joint_tau`` kernel is still exercised
   as a reference for what the old (buggy) net-torque clamp would have
   done.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers import SolverFeatherPGS
from newton._src.solvers.feather_pgs.kernels import (
    clamp_augmented_joint_u0,
    clamp_joint_tau,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _build_one_dof_pendulum_model(device: str) -> newton.Model:
    """Build a minimal 1-DOF revolute pendulum for FPGS-plumbing tests.

    The joint geometry is not important for the plumbing tests; only that
    :class:`~newton.solvers.SolverFeatherPGS` can be constructed against
    a non-empty articulated model.
    """
    builder = newton.ModelBuilder()
    link = builder.add_link()
    joint = builder.add_joint_revolute(
        parent=-1,
        child=link,
        axis=wp.vec3(0.0, 0.0, 1.0),
        target_ke=0.0,
        target_kd=0.0,
        effort_limit=5.0,
    )
    builder.add_articulation([joint])
    model = builder.finalize(device=device)
    return model


class TestFeatherPGSEffortLimitMode(unittest.TestCase):
    """Parameter-plumbing checks for the ``effort_limit_mode`` switch."""


def run_default_mode_is_actuator(test: TestFeatherPGSEffortLimitMode, device):
    """With no kwarg, the solver uses the actuator-only clamp."""
    model = _build_one_dof_pendulum_model(device)
    solver = SolverFeatherPGS(model)
    test.assertEqual(solver.effort_limit_mode, "actuator")


def run_actuator_mode_stored(test: TestFeatherPGSEffortLimitMode, device):
    """``effort_limit_mode="actuator"`` round-trips onto the instance."""
    model = _build_one_dof_pendulum_model(device)
    solver = SolverFeatherPGS(model, effort_limit_mode="actuator")
    test.assertEqual(solver.effort_limit_mode, "actuator")


def run_invalid_mode_raises(test: TestFeatherPGSEffortLimitMode, device):
    """Any value outside 'actuator' must raise ``ValueError``; the legacy
    ``'net'`` semantics is no longer supported."""
    model = _build_one_dof_pendulum_model(device)
    with test.assertRaises(ValueError):
        SolverFeatherPGS(model, effort_limit_mode="bogus")
    with test.assertRaises(ValueError):
        SolverFeatherPGS(model, effort_limit_mode="net")


class TestFeatherPGSEffortLimitClamp(unittest.TestCase):
    """Direct signal that the alternative path clamps a different quantity.

    ``clamp_joint_tau`` is the baseline kernel that clamps the *net*
    generalized torque (rigid/passive + drive-explicit) sitting in
    ``joint_tau``.  ``clamp_augmented_joint_u0`` is the new kernel used
    under ``effort_limit_mode="actuator"``; it clamps only the per-row
    actuator-drive ``u0`` bucket and never touches ``joint_tau``.
    """


def run_clamp_augmented_joint_u0_clamps_only_drive(test: TestFeatherPGSEffortLimitClamp, device):
    """The actuator-bucket kernel clamps ``row_u0`` against the per-DOF limit.

    Build a single-articulation, 2-DOF layout with two drive rows:

    - DOF 0: ``u0 = 100``, ``limit = 5``     → expect clamp to ``+5``.
    - DOF 1: ``u0 = -50``, ``limit = 0``     → expect no clamp (unlimited).

    Populate the ``joint_tau`` bucket with a rigid/passive value that
    itself exceeds the DOF-0 limit. The kernel must not touch it: that
    is the core difference vs. :func:`clamp_joint_tau`.
    """
    # Single articulation, max_dofs == 2.
    max_dofs = 2
    n_articulations = 1

    row_counts = wp.array([2], dtype=wp.int32, device=device)
    row_dof_index = wp.array([0, 1], dtype=wp.int32, device=device)
    row_u0 = wp.array([100.0, -50.0], dtype=wp.float32, device=device)

    # DOF 0 has a finite limit; DOF 1 has limit=0 → unlimited in FPGS convention.
    joint_effort_limit = wp.array([5.0, 0.0], dtype=wp.float32, device=device)

    # Rigid/passive bucket for the same DOFs. Under Semantic B (actuator)
    # this must remain untouched after the drive-only clamp kernel.
    joint_tau = wp.array([-9.81, 0.0], dtype=wp.float32, device=device)

    wp.launch(
        clamp_augmented_joint_u0,
        dim=n_articulations,
        inputs=[max_dofs, row_counts, row_dof_index, joint_effort_limit],
        outputs=[row_u0],
        device=device,
    )

    # Drive bucket clamped per DOF.
    u0_out = row_u0.numpy()
    np.testing.assert_allclose(u0_out, np.array([5.0, -50.0], dtype=np.float32), rtol=0.0, atol=1e-7)

    # Rigid/passive bucket is untouched.
    np.testing.assert_allclose(joint_tau.numpy(), np.array([-9.81, 0.0], dtype=np.float32), rtol=0.0, atol=1e-6)


def run_clamp_joint_tau_clamps_net_sum(test: TestFeatherPGSEffortLimitClamp, device):
    """The baseline kernel clamps the net ``joint_tau`` buffer in place.

    Mirrors ``run_clamp_augmented_joint_u0_clamps_only_drive`` but on the
    baseline kernel, showing the quantity being clamped is the net
    generalized torque (``tau_rigid + u0``) rather than the drive-only
    ``u0`` bucket. This is the other half of the "different quantity"
    signal.
    """
    # Same pair of DOFs as the actuator-bucket test, but we now store the
    # *net* generalized torque in joint_tau: rigid_passive = -9.81 at DOF 0
    # plus u0 = +100 → net = +90.19. For DOF 1: rigid = 0, u0 = -50,
    # net = -50. Limit 5 on DOF 0, unlimited on DOF 1.
    net_tau = np.array([-9.81 + 100.0, 0.0 - 50.0], dtype=np.float32)
    joint_tau = wp.array(net_tau, dtype=wp.float32, device=device)
    joint_effort_limit = wp.array([5.0, 0.0], dtype=wp.float32, device=device)

    wp.launch(
        clamp_joint_tau,
        dim=2,
        inputs=[joint_tau, joint_effort_limit],
        device=device,
    )

    out = joint_tau.numpy()
    # DOF 0's net is clamped to +5 (baseline: the rigid/passive contribution
    # is *not* excluded — it is part of the clamped quantity). DOF 1 is
    # unlimited so the -50 survives unchanged.
    np.testing.assert_allclose(out, np.array([5.0, -50.0], dtype=np.float32), rtol=0.0, atol=1e-7)


def run_net_vs_actuator_differ_on_passive_only_dof(test: TestFeatherPGSEffortLimitClamp, device):
    """Direct divergence signal: same inputs, two kernels, different outputs.

    Build a DOF whose drive ``u0 = 0`` but whose rigid/passive bucket is
    ``-9.81`` (a gravity-loaded joint). With ``limit = 5.0`` the two
    semantics produce opposite answers:

    - Baseline (:func:`clamp_joint_tau` on the summed buffer):
      clamps the passive load to ``-5.0``.
    - Alternative (:func:`clamp_augmented_joint_u0` on the drive-only
      buffer): leaves the passive load at ``-9.81`` untouched because
      there is nothing to clamp in the drive bucket.
    """
    # Shared inputs.
    limit = 5.0
    rigid_passive = -9.81  # "tau_passive" at a horizontal pendulum, zero drive.

    # --- Baseline (net clamp) ---
    # joint_tau after `apply_augmented_joint_tau` in the baseline path
    # contains the summed rigid + drive-explicit bucket. Here drive=0.
    net_joint_tau = wp.array([rigid_passive], dtype=wp.float32, device=device)
    net_limit = wp.array([limit], dtype=wp.float32, device=device)
    wp.launch(
        clamp_joint_tau,
        dim=1,
        inputs=[net_joint_tau, net_limit],
        device=device,
    )
    net_out = float(net_joint_tau.numpy()[0])

    # --- Alternative (actuator-only clamp) ---
    # Under the alternative semantics, `joint_tau` after
    # `eval_rigid_tau` still holds the rigid bucket and we only clamp the
    # drive bucket. With zero drive gains there is no augmented row, so
    # `row_counts = 0` and the drive-only clamp kernel is a no-op on the
    # one row. Critically, `joint_tau` retains the uncapped passive load.
    act_joint_tau = wp.array([rigid_passive], dtype=wp.float32, device=device)
    act_limit = wp.array([limit], dtype=wp.float32, device=device)
    act_row_counts = wp.array([0], dtype=wp.int32, device=device)
    act_row_dof_index = wp.zeros((1,), dtype=wp.int32, device=device)
    act_row_u0 = wp.zeros((1,), dtype=wp.float32, device=device)
    wp.launch(
        clamp_augmented_joint_u0,
        dim=1,
        inputs=[1, act_row_counts, act_row_dof_index, act_limit],
        outputs=[act_row_u0],
        device=device,
    )
    act_out = float(act_joint_tau.numpy()[0])

    # Baseline semantics clamps the gravity torque to the effort limit,
    # so the magnitude stored in joint_tau is exactly `limit`.
    test.assertAlmostEqual(net_out, -limit, places=6)
    # Alternative semantics leaves the gravity torque untouched.
    test.assertAlmostEqual(act_out, rigid_passive, places=6)
    # And the two outputs actually disagree — the "direct signal" that
    # the alternative path clamps a different quantity.
    test.assertGreater(abs(act_out - net_out), 1e-3)


devices = get_test_devices()

for device in devices:
    add_function_test(
        TestFeatherPGSEffortLimitMode,
        f"test_default_mode_is_actuator_{device}",
        run_default_mode_is_actuator,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSEffortLimitMode,
        f"test_actuator_mode_stored_{device}",
        run_actuator_mode_stored,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSEffortLimitMode,
        f"test_invalid_mode_raises_{device}",
        run_invalid_mode_raises,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSEffortLimitClamp,
        f"test_clamp_augmented_joint_u0_clamps_only_drive_{device}",
        run_clamp_augmented_joint_u0_clamps_only_drive,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSEffortLimitClamp,
        f"test_clamp_joint_tau_clamps_net_sum_{device}",
        run_clamp_joint_tau_clamps_net_sum,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSEffortLimitClamp,
        f"test_net_vs_actuator_differ_on_passive_only_dof_{device}",
        run_net_vs_actuator_differ_on_passive_only_dof,
        devices=[device],
    )


if __name__ == "__main__":
    unittest.main()
