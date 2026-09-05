# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise capacity, dispatch and history contracts on ordinary robot defaults."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS


def _driven_model(count=1, worlds=1, device="cpu"):
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    for _ in range(count):
        body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        joint = builder.add_joint_prismatic(
            -1, body, axis=newton.Axis.X, target_ke=10000.0, target_kd=0.0, armature=0.0, damping=0.0
        )
        builder.add_articulation([joint])
    if worlds > 1:
        replicated = newton.ModelBuilder(gravity=wp.vec3(0.0))
        replicated.replicate(builder, worlds, spacing=(1.0, 0.0, 0.0))
        builder = replicated
    return builder.finalize(device=device)


class TestFeatherPGSDriveCapacity(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "requires the native tiled CUDA solver")
    def test_partial_warp_capacity_preserves_world_boundaries(self):
        """Keep partial-warp loads and stores inside each world's row allocation."""
        device = wp.get_cuda_device()
        for capacity in (33, 35, 40):
            with self.subTest(capacity=capacity):
                model = _driven_model(worlds=3, device=device)
                solver = SolverFeatherPGS(
                    model, pgs_mode="split", pgs_kernel="tiled_row", dense_max_constraints=capacity
                )
                initial = np.zeros((3, capacity), dtype=np.float32)
                initial[2] = 1234.0  # Inactive world must remain untouched.
                solver.constraint_count.assign(np.array([capacity, 1, 0], dtype=np.int32))
                solver.impulses.assign(initial)
                solver.diag.fill_(2.0)
                solver.C.assign(np.tile(2.0 * np.eye(capacity, dtype=np.float32), (3, 1, 1)))
                solver.rhs.fill_(-1.0)
                solver.row_type.zero_()
                solver.row_parent.zero_()
                solver.row_mu.zero_()
                solver._dispatch_dense_pgs_solve(iterations=1, friction_start_iteration=0)
                expected = initial.copy()
                expected[0] = 0.5
                expected[1, 0] = 0.5
                np.testing.assert_array_equal(solver.impulses.numpy(), expected)

    def test_default_capacity_reserves_contact_rows_after_drives(self):
        """Retain all default-effort drives and the default contact row budget."""
        model = _driven_model(count=40)
        solver = SolverFeatherPGS(model)
        self.assertGreaterEqual(solver.dense_max_constraints, 40 + 32)
        state, output = model.state(), model.state()
        control = model.control()
        control.joint_target_q.fill_(0.1)
        solver.step(state, output, control, None, 0.01)
        self.assertLessEqual(int(solver.slot_counter.numpy().max()), solver.dense_max_constraints)
        np.testing.assert_allclose(output.joint_qd.numpy(), 5.0, atol=1.0e-4)

    def test_builder_default_effort_matches_unsaturated_implicit_drive(self):
        """Match the analytic unsaturated drive response at the builder effort default."""
        model = _driven_model()
        self.assertEqual(float(model.joint_effort_limit.numpy()[0]), 1.0e6)
        solver = SolverFeatherPGS(model, pgs_iterations=1)
        state, output = model.state(), model.state()
        control = model.control()
        control.joint_target_q.fill_(0.1)
        solver.step(state, output, control, None, 0.01)
        self.assertAlmostEqual(float(output.joint_qd.numpy()[0]), 5.0, delta=1.0e-4)

    def test_unsupported_drive_kernel_is_rejected(self):
        """Reject explicit kernel requests that cannot solve bounded drive rows."""
        model = _driven_model()
        for kernel in ("tiled_contact", "streaming"):
            with self.subTest(kernel=kernel), self.assertRaisesRegex(ValueError, "drive.*tiled_row"):
                SolverFeatherPGS(model, pgs_kernel=kernel)


if __name__ == "__main__":
    unittest.main()
