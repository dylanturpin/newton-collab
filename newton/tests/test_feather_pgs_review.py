# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise capacity, dispatch and history contracts on ordinary robot defaults."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.solver_feather_pgs import _get_pgs_solve_tiled_row_kernel
from newton.solvers import SolverFeatherPGS
from newton.tests.test_contact_reduction_body_pairs import _cylinder_foot


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


class TestFeatherPGSReview(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "requires the native tiled CUDA solver")
    def test_partial_warp_capacity_preserves_world_boundaries(self):
        """Keep partial-warp loads and stores inside each world's row allocation."""
        device = wp.get_cuda_device()
        for capacity in (33, 35, 40):
            with self.subTest(capacity=capacity):
                counts = wp.array([capacity, 1, 0], dtype=wp.int32, device=device)
                initial = np.zeros((3, capacity), dtype=np.float32)
                initial[2] = 1234.0  # Inactive world must remain untouched.
                impulses = wp.array(initial, dtype=wp.float32, device=device)
                diagonal = wp.full((3, capacity), 2.0, dtype=wp.float32, device=device)
                matrix = wp.array(np.tile(2.0 * np.eye(capacity), (3, 1, 1)), dtype=wp.float32, device=device)
                rhs = wp.full((3, capacity), -1.0, dtype=wp.float32, device=device)
                integer = wp.zeros((3, capacity), dtype=wp.int32, device=device)
                floating = wp.zeros((3, capacity), dtype=wp.float32, device=device)
                kernel = _get_pgs_solve_tiled_row_kernel(capacity, str(device.arch))
                wp.launch_tiled(
                    kernel,
                    dim=[3],
                    inputs=[
                        counts,
                        diagonal,
                        matrix,
                        rhs,
                        impulses,
                        1,
                        1.0,
                        integer,
                        integer,
                        floating,
                        0,
                        0,
                        floating,
                        floating,
                        floating,
                        floating,
                    ],
                    block_dim=32,
                    device=device,
                )
                expected = initial.copy()
                expected[0] = 0.5
                expected[1, 0] = 0.5
                np.testing.assert_array_equal(impulses.numpy(), expected)

    def test_default_capacity_reserves_contact_rows_after_drives(self):
        """Retain all default-effort drives and the default contact row budget."""
        model = _driven_model(count=40)
        solver = SolverFeatherPGS(model)
        self.assertGreaterEqual(solver.dense_max_constraints, 40 + 32)
        state, output = model.state(), model.state()
        control = model.control()
        control.joint_target_q.fill_(0.1)
        solver.step(state, output, control, None, 0.01)
        solver.check_constraint_capacity()
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

    def test_partial_reset_refreshes_only_selected_mass_factors(self):
        """Preserve factor reuse in worlds outside a partial episode reset."""
        devices = ["cpu", wp.get_cuda_device()] if wp.is_cuda_available() else ["cpu"]
        for device in devices:
            with self.subTest(device=str(device)):
                model = _driven_model(worlds=2, device=device)
                solver = SolverFeatherPGS(model, update_mass_matrix_interval=100)
                state, output = model.state(), model.state()
                control = model.control()
                solver.step(state, output, control, None, 0.01)
                mask = wp.array([True, False], dtype=wp.bool, device=device)
                solver.reset(state, mask)
                solver.step(state, output, control, None, 0.01)
                np.testing.assert_array_equal(solver.mass_update_mask.numpy(), [1, 0])
                if wp.get_device(device).is_cuda:
                    first_mask = wp.empty_like(solver.mass_update_mask)
                    with wp.ScopedCapture(device=device) as capture:
                        solver.seed_double_buffer_events()
                        solver.reset(state, mask)
                        solver.step(state, output, control, None, 0.01)
                        wp.copy(first_mask, solver.mass_update_mask)
                        solver.step(state, output, control, None, 0.01)
                    mask.assign(np.array([False, True]))
                    wp.capture_launch(capture.graph)
                    np.testing.assert_array_equal(first_mask.numpy(), [0, 1])
                    # The request is consumed by the first step; the next reuses both factors.
                    np.testing.assert_array_equal(solver.mass_update_mask.numpy(), [0, 0])

    def test_unmatched_pipeline_clears_stale_contact_identity(self):
        """Cold-start a matched buffer when its current producer disables matching."""
        builder = newton.ModelBuilder()
        _cylinder_foot(builder, wp.vec3(0.0, 0.0, 0.015))
        builder.add_ground_plane()
        model = builder.finalize(device="cpu")
        config = newton.CollisionPipeline.ContactReductionConfig(body_pairs=True)
        matched = newton.CollisionPipeline(model, contact_matching="latest", reduce_contacts=config)
        unmatched = newton.CollisionPipeline(model, deterministic=True, reduce_contacts=config)
        state, contacts = model.state(), matched.contacts()
        matched.collide(state, contacts)
        matched.collide(state, contacts)
        count = int(contacts.rigid_contact_count.numpy()[0])
        self.assertTrue(np.any(contacts.rigid_contact_match_index.numpy()[:count] >= 0))
        unmatched.collide(state, contacts)
        count = int(contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(count, 0)
        self.assertTrue(np.all(contacts.rigid_contact_match_index.numpy()[:count] == -1))

    def test_contact_reset_clears_reduction_without_matching(self):
        """Clear selected patch history even when the matcher is disabled."""
        template = newton.ModelBuilder()
        _cylinder_foot(template, wp.vec3(0.0, 0.0, 0.015))
        builder = newton.ModelBuilder()
        builder.replicate(template, 2, spacing=(1.0, 0.0, 0.0))
        builder.add_ground_plane()
        model = builder.finalize(device="cpu")
        pipeline = newton.CollisionPipeline(
            model,
            reduce_contacts=newton.CollisionPipeline.ContactReductionConfig(
                body_pairs=True, body_pair_hysteresis=0.001
            ),
        )
        contacts = pipeline.contacts()
        pipeline.collide(model.state(), contacts)
        before = pipeline._body_pair_reducer.history_generation.numpy().copy()
        pipeline.reset_contact_matching(wp.array([True, False, False], dtype=wp.bool, device=model.device))
        after = pipeline._body_pair_reducer.history_generation.numpy()
        self.assertGreater(after[0], before[0])
        self.assertEqual(after[1], before[1])


if __name__ == "__main__":
    unittest.main()
