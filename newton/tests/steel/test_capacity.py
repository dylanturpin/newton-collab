# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Row capacities, overflow reporting, watermarks and heterogeneous worlds of SolverSteel."""

from __future__ import annotations

import gc
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.steel._testing import kernel_overrides
from newton._src.solvers.steel.friction_patches import FrictionPatches
from newton._src.solvers.steel.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT_ROLLING,
    PGS_CONSTRAINT_TYPE_CONTACT_SPIN,
    PGS_CONSTRAINT_TYPE_TORSION,
    allocate_world_contact_slots,
    collect_propagation_units,
    finalize_mf_constraint_counts,
)
from newton._src.solvers.steel.solver_steel import (
    PROPAGATION_COLOR_TAIL,
    _get_color_propagation_prebuild_kernel,
)
from newton.solvers import SolverSteel
from newton.tests.steel._fixtures import set_torsion_radius
from newton.tests.steel.test_propagation import _add_arm_on_heap, _Stepper
from newton.tests.test_contact_reduction_body_pairs import _cylinder_foot
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices


class TestSteelCapacityStatus(unittest.TestCase):
    def test_capacity_failure_is_observable_without_watermarks(self):
        """Expose dropped contacts even when optional detailed telemetry is off."""
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device="cpu")
        solver = SolverSteel(model, free_body_constraint_max=1)
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        state_in, state_out = model.state(), model.state()
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, model.control(), contacts, 1.0 / 240.0)
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            solver.check_constraint_capacity()
        self.assertTrue(solver.constraint_overflow.numpy()[0])
        solver.reset(state_out)
        solver.check_constraint_capacity()


def _box_on_ground(device, worlds=1):
    template = newton.ModelBuilder()
    body = template.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
    template.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    builder = newton.ModelBuilder()
    builder.replicate(template, worlds, spacing=(1.0, 0.0, 0.0))
    builder.add_ground_plane()
    return builder.finalize(device=device)


def test_capacity_failure_is_observable(test, device, solve_mode="matrix_free"):
    """Flag dropped free-body contact rows until reset, without optional telemetry."""
    model = _box_on_ground(device)
    solver = SolverSteel(model, solve_mode=solve_mode, free_body_constraint_max=1, warn_constraint_overflow=False)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_in, state_out = model.state(), model.state()
    pipeline.collide(state_in, contacts)
    solver.step(state_in, state_out, model.control(), contacts, 1.0 / 240.0)
    with test.assertRaisesRegex(RuntimeError, "capacity"):
        solver.check_constraint_capacity()
    test.assertTrue(solver.constraint_overflow.numpy()[0])
    test.assertGreater(int(solver._row_dropped_mf.numpy()[0]), 0)
    solver.reset(state_out)
    solver.check_constraint_capacity()


def test_status_snapshot_reports_status_at_snapshot(test, device):
    """Report the status as copied by snapshot_constraint_status, unchanged by later steps and resets."""
    model = _box_on_ground(device)
    solver = SolverSteel(model, free_body_constraint_max=1, warn_constraint_overflow=False)
    with test.assertRaisesRegex(RuntimeError, "snapshot_constraint_status"):
        solver.check_constraint_capacity(snapshot=True)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_in, state_out = model.state(), model.state()
    solver.snapshot_constraint_status()
    pipeline.collide(state_in, contacts)
    solver.step(state_in, state_out, model.control(), contacts, 1.0 / 240.0)
    solver.check_constraint_capacity(snapshot=True)
    solver.snapshot_constraint_status()
    solver.reset(state_out)
    solver.check_constraint_capacity()
    with test.assertRaisesRegex(RuntimeError, "capacity"):
        solver.check_constraint_capacity(snapshot=True)
    solver.snapshot_constraint_status()
    solver.check_constraint_capacity(snapshot=True)
    solver._disabled_tree_joint.fill_(True)
    solver.snapshot_constraint_status()
    with test.assertRaisesRegex(RuntimeError, "articulation tree joint"):
        solver.check_constraint_capacity(snapshot=True)


def test_released_snapshot_solver_keeps_another_capture_valid(test, device):
    """Release a solver that took and read a snapshot while another graph is being captured."""
    model = _box_on_ground(device)
    solver = SolverSteel(model, free_body_constraint_max=1, warn_constraint_overflow=False)
    solver.snapshot_constraint_status()
    solver.check_constraint_capacity(snapshot=True)
    counter = wp.zeros(1, dtype=int, device=device)
    with wp.ScopedCapture(device) as capture:
        counter.fill_(1)
        del solver
        gc.collect()
        counter.fill_(2)
    wp.capture_launch(capture.graph)
    test.assertEqual(int(counter.numpy()[0]), 2)


class TestSteelCapacityStatusUpstream(unittest.TestCase):
    pass


devices = get_cuda_test_devices()


add_function_test(
    TestSteelCapacityStatusUpstream,
    "test_capacity_failure_is_observable",
    test_capacity_failure_is_observable,
    devices=devices,
)
add_function_test(
    TestSteelCapacityStatusUpstream,
    "test_released_snapshot_solver_keeps_another_capture_valid",
    test_released_snapshot_solver_keeps_another_capture_valid,
    devices=devices,
)
add_function_test(
    TestSteelCapacityStatusUpstream,
    "test_status_snapshot_reports_status_at_snapshot",
    test_status_snapshot_reports_status_at_snapshot,
    devices=get_test_devices(),
)


for _name in ("test_capacity_failure_is_observable",):
    add_function_test(
        TestSteelCapacityStatusUpstream,
        f"{_name}_split",
        globals()[_name],
        devices=get_test_devices(),
        check_output=_name != "test_overflow_warning_is_printed_once",
        solve_mode="split",
    )


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


class TestSteelSafety(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "requires the native tiled CUDA solver")
    def test_partial_warp_capacity_preserves_world_boundaries(self):
        """Keep partial-warp loads and stores inside each world's row allocation."""
        device = wp.get_cuda_device()
        for capacity in (33, 35, 40):
            with self.subTest(capacity=capacity):
                model = _driven_model(worlds=3, device=device)
                with kernel_overrides({"pgs_kernel": "tiled_row"}):
                    solver = SolverSteel(model, solve_mode="split", articulation_constraint_max=capacity)
                initial = np.zeros((3, capacity), dtype=np.float32)
                initial[2] = 1234.0  # Inactive world must remain untouched.
                solver._constraint_count.assign(np.array([capacity, 1, 0], dtype=np.int32))
                solver._impulses.assign(initial)
                solver._diag.fill_(2.0)
                solver._C.assign(np.tile(2.0 * np.eye(capacity, dtype=np.float32), (3, 1, 1)))
                solver._rhs.fill_(-1.0)
                solver._row_type.zero_()
                solver._row_parent.zero_()
                solver._row_mu.zero_()
                solver._dispatch_dense_pgs_solve(iterations=1, friction_start_iteration=0)
                expected = initial.copy()
                expected[0] = 0.5
                expected[1, 0] = 0.5
                np.testing.assert_array_equal(solver._impulses.numpy(), expected)

    @unittest.skipUnless(wp.is_cuda_available(), "requires the native tiled CUDA solver")
    def test_dense_capacity_beyond_static_shared_memory_builds(self):
        """Solve a dense capacity whose packed Delassus triangle exceeds static shared memory."""
        device = wp.get_cuda_device()
        capacity = 256
        model = _driven_model(worlds=2, device=device)
        solver = SolverSteel(model, solve_mode="split", articulation_constraint_max=capacity)
        self.assertEqual(solver._pgs_kernel, "loop")
        self.assertIsNone(solver._pgs_solve_tiled_row_kernel)
        state, output = model.state(), model.state()
        solver.step(state, output, model.control(), None, 0.01)
        self.assertTrue(np.isfinite(output.body_qd.numpy()).all())

        solver._constraint_count.assign(np.array([capacity, 1], dtype=np.int32))
        solver._impulses.zero_()
        solver._diag.fill_(2.0)
        solver._C.assign(np.tile(2.0 * np.eye(capacity, dtype=np.float32), (2, 1, 1)))
        solver._rhs.fill_(-1.0)
        solver._row_type.zero_()
        solver._row_parent.zero_()
        solver._row_mu.zero_()
        solver._dispatch_dense_pgs_solve(iterations=1, friction_start_iteration=0)
        expected = np.zeros((2, capacity), dtype=np.float32)
        expected[0] = 0.5
        expected[1, 0] = 0.5
        np.testing.assert_array_equal(solver._impulses.numpy(), expected)

        with self.assertRaisesRegex(ValueError, "shared memory"):
            with kernel_overrides({"pgs_kernel": "tiled_row"}):
                SolverSteel(model, solve_mode="split", articulation_constraint_max=capacity)

    def test_partial_reset_refreshes_only_selected_mass_factors(self):
        """Preserve factor reuse in worlds outside a partial episode reset."""
        devices = ["cpu", wp.get_cuda_device()] if wp.is_cuda_available() else ["cpu"]
        for device in devices:
            with self.subTest(device=str(device)):
                model = _driven_model(worlds=2, device=device)
                solver = SolverSteel(model, update_mass_matrix_interval=100)
                state, output = model.state(), model.state()
                control = model.control()
                solver.step(state, output, control, None, 0.01)
                mask = wp.array([True, False], dtype=wp.bool, device=device)
                solver.reset(state, mask)
                solver.step(state, output, control, None, 0.01)
                np.testing.assert_array_equal(solver._mass_update_mask.numpy(), [1, 0])
                if wp.get_device(device).is_cuda:
                    first_mask = wp.empty_like(solver._mass_update_mask)
                    with wp.ScopedCapture(device=device) as capture:
                        solver.reset(state, mask)
                        solver.step(state, output, control, None, 0.01)
                        wp.copy(first_mask, solver._mass_update_mask)
                        solver.step(state, output, control, None, 0.01)
                    mask.assign(np.array([False, True]))
                    wp.capture_launch(capture.graph)
                    np.testing.assert_array_equal(first_mask.numpy(), [0, 1])
                    # The request is consumed by the first step; the next reuses both factors.
                    np.testing.assert_array_equal(solver._mass_update_mask.numpy(), [0, 0])

    def test_unmatched_pipeline_clears_stale_contact_identity(self):
        """Cold-start a matched buffer when its current producer disables matching."""
        builder = newton.ModelBuilder()
        _cylinder_foot(builder, wp.vec3(0.0, 0.0, 0.015))
        builder.add_ground_plane()
        model = builder.finalize(device="cpu")
        config = newton.CollisionPipeline.ContactReductionConfig(body_pairs=True)
        matched = newton.CollisionPipeline(model, contact_matching="latest")
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


def _build_model():
    """Build one articulated body for contact-capacity tests."""
    builder = newton.ModelBuilder()
    body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    joint = builder.add_joint_free(parent=-1, child=body)
    builder.add_articulation([joint])
    model = builder.finalize(device="cpu")
    model.rigid_contact_max = 1
    return model


class TestSteelContactCapacity(unittest.TestCase):
    def _assert_mixed_units_are_all_scheduled(
        self,
        *,
        contact_phi: tuple[float, ...],
        friction_gap_threshold: float,
        patch_friction: bool,
    ):
        """Build and color a row-capacity-filling mixture of contact units."""
        device = wp.get_device("cuda:0")
        contact_count = len(contact_phi)
        row_capacity = 6
        color_entries = contact_count + 2  # one color per unit, plus the tail and total entries
        order_stride = 8  # power of two >= row_capacity

        contact_count_array = wp.array([contact_count], dtype=wp.int32, device=device)
        contact_shape0 = wp.zeros((contact_count,), dtype=wp.int32, device=device)
        contact_shape1 = wp.full((contact_count,), -1, dtype=wp.int32, device=device)
        contact_point0_np = np.zeros((contact_count, 3), dtype=np.float32)
        contact_point0_np[:, 2] = contact_phi
        contact_point0 = wp.array(contact_point0_np, dtype=wp.vec3, device=device)
        contact_point1 = wp.zeros((contact_count,), dtype=wp.vec3, device=device)
        contact_normal = wp.array(
            np.tile(np.array([0.0, 0.0, -1.0], dtype=np.float32), (contact_count, 1)),
            dtype=wp.vec3,
            device=device,
        )
        contact_slot = wp.full((contact_count,), -1, dtype=wp.int32, device=device)
        contact_path = wp.full((contact_count,), -1, dtype=wp.int32, device=device)
        contact_world = wp.zeros((contact_count,), dtype=wp.int32, device=device)
        contact_slots_needed = wp.zeros((contact_count,), dtype=wp.int32, device=device)
        propagation_slot_counter = wp.zeros((1,), dtype=wp.int32, device=device)

        patches = FrictionPatches()
        if patch_friction:
            patches.enabled = 1
            patches.weight = wp.array([1.0] + [0.0] * (contact_count - 1), dtype=float, device=device)

        wp.launch(
            allocate_world_contact_slots,
            dim=contact_count,
            inputs=[
                contact_count_array,
                contact_count,
                contact_shape0,
                contact_shape1,
                contact_point0,
                contact_point1,
                contact_normal,
                wp.zeros((contact_count,), dtype=wp.float32, device=device),
                wp.zeros((contact_count,), dtype=wp.float32, device=device),
                wp.array([wp.transform_identity()], dtype=wp.transform, device=device),
                wp.zeros((1,), dtype=wp.int32, device=device),
                wp.zeros((1,), dtype=wp.int32, device=device),
                wp.zeros((1,), dtype=wp.int32, device=device),
                wp.array([6], dtype=wp.int32, device=device),
                wp.zeros((1,), dtype=wp.int32, device=device),
                wp.ones((1,), dtype=wp.int32, device=device),
                wp.zeros((1,), dtype=wp.int32, device=device),
                0,
                1,
                1,
                0,
                float("inf"),
                row_capacity,
                row_capacity,
                row_capacity,
                friction_gap_threshold,
                patches,
                0,
                wp.zeros((1,), dtype=wp.float32, device=device),
                wp.zeros((1,), dtype=wp.float32, device=device),
                wp.zeros((1,), dtype=wp.float32, device=device),
                wp.zeros((1,), dtype=wp.float32, device=device),
                0,
                wp.zeros((1,), dtype=wp.float32, device=device),
                0,
            ],
            outputs=[
                contact_world,
                contact_slot,
                wp.full((contact_count,), -1, dtype=wp.int32, device=device),
                wp.full((contact_count,), -1, dtype=wp.int32, device=device),
                wp.zeros((1,), dtype=wp.int32, device=device),
                contact_path,
                wp.zeros((1,), dtype=wp.int32, device=device),
                propagation_slot_counter,
                wp.zeros((1,), dtype=wp.int32, device=device),
                contact_slots_needed,
                wp.zeros((1,), dtype=wp.int32, device=device),
                wp.zeros((1,), dtype=wp.int32, device=device),
                wp.zeros((1,), dtype=wp.int32, device=device),
                wp.full((1,), 2**31 - 1, dtype=wp.int32, device=device),
                wp.full((1,), 2**31 - 1, dtype=wp.int32, device=device),
                wp.full((1,), 2**31 - 1, dtype=wp.int32, device=device),
                wp.zeros((contact_count,), dtype=wp.int32, device=device),
            ],
            device=device,
        )

        unit_cursor = wp.zeros((1,), dtype=wp.int32, device=device)
        unit_contact = wp.full((row_capacity,), -1, dtype=wp.int32, device=device)
        unit_body_a = wp.full((row_capacity,), -1, dtype=wp.int32, device=device)
        unit_body_b = wp.full((row_capacity,), -1, dtype=wp.int32, device=device)
        unit_len = wp.zeros((row_capacity,), dtype=wp.int32, device=device)
        wp.launch(
            collect_propagation_units,
            dim=contact_count,
            inputs=[
                contact_count_array,
                contact_path,
                contact_world,
                contact_shape0,
                contact_shape1,
                wp.zeros((1,), dtype=wp.int32, device=device),
                wp.zeros((1,), dtype=wp.int32, device=device),  # body_prescribed: none here
                contact_slots_needed,
                row_capacity,
            ],
            outputs=[unit_cursor, unit_contact, unit_body_a, unit_body_b, unit_len],
            device=device,
        )

        row_order = wp.full((row_capacity,), -1, dtype=wp.int32, device=device)
        color_offsets = wp.full((color_entries,), -1, dtype=wp.int32, device=device)
        sorted_contact = wp.full((row_capacity,), -1, dtype=wp.int32, device=device)
        kernel = _get_color_propagation_prebuild_kernel(
            row_capacity,
            color_entries,
            64,
            order_stride,
            str(device.arch),
        )
        wp.launch_tiled(
            kernel,
            dim=[1],
            inputs=[
                1,
                unit_cursor,
                unit_contact,
                unit_body_a,
                unit_body_b,
                unit_len,
                wp.zeros(((PROPAGATION_COLOR_TAIL + 31) // 32,), dtype=wp.int32, device=device),  # body 0 mask
                wp.zeros((order_stride,), dtype=wp.int32, device=device),
                wp.zeros((row_capacity,), dtype=wp.int32, device=device),
                contact_slot,
                row_order,
                color_offsets,
                sorted_contact,
            ],
            block_dim=64,
            device=device,
        )

        slots_needed_np = contact_slots_needed.numpy()
        self.assertEqual(sorted(slots_needed_np.tolist()), [1, 1, 1, 3])
        np.testing.assert_array_equal(contact_path.numpy(), np.full(contact_count, 2, dtype=np.int32))
        self.assertEqual(int(propagation_slot_counter.numpy()[0]), row_capacity)
        self.assertEqual(int(unit_cursor.numpy()[0]), contact_count)

        # Every unit is colored (no staging cap, so nothing spills into the tail). All
        # units share body 0, so each takes its own color, in contact-index order.
        offsets_np = color_offsets.numpy()
        self.assertEqual(int(offsets_np[-1]), contact_count)
        self.assertEqual(int(offsets_np[-1] - offsets_np[-2]), 0)
        self.assertEqual(offsets_np[: contact_count + 1].tolist(), list(range(contact_count + 1)))

        sorted_contact_np = sorted_contact.numpy()[:contact_count]
        self.assertEqual(sorted_contact_np.tolist(), list(range(contact_count)))
        contact_slot_np = contact_slot.numpy()
        row_order_np = row_order.numpy()
        row_acc = 0
        for position, contact in enumerate(sorted_contact_np):
            self.assertEqual(int(row_order_np[position]), row_acc)
            self.assertEqual(int(contact_slot_np[contact]), row_acc)
            row_acc += int(slots_needed_np[contact])
        self.assertEqual(row_acc, row_capacity)

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
                capacity,
                wp.zeros((capacity,), dtype=wp.int32, device=device),
                wp.full((capacity,), -1, dtype=wp.int32, device=device),
                wp.zeros((capacity,), dtype=wp.vec3, device=device),
                wp.zeros((capacity,), dtype=wp.vec3, device=device),
                wp.array([wp.vec3(0.0, 0.0, 1.0)] * capacity, dtype=wp.vec3, device=device),
                wp.zeros((capacity,), dtype=wp.float32, device=device),
                wp.zeros((capacity,), dtype=wp.float32, device=device),
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
                float("inf"),
                8,
                8,
                8,
                0.0,
                FrictionPatches(),
                0,
                wp.zeros((1,), dtype=wp.float32, device=device),
                wp.zeros((1,), dtype=wp.float32, device=device),
                wp.zeros((1,), dtype=wp.float32, device=device),
                wp.zeros((1,), dtype=wp.float32, device=device),
                0,
                wp.zeros((1,), dtype=wp.float32, device=device),
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
                wp.zeros((1,), dtype=wp.int32, device=device),
                wp.zeros((1,), dtype=wp.int32, device=device),
                wp.zeros((1,), dtype=wp.int32, device=device),
                wp.full((1,), 2**31 - 1, dtype=wp.int32, device=device),
                wp.full((1,), 2**31 - 1, dtype=wp.int32, device=device),
                wp.full((1,), 2**31 - 1, dtype=wp.int32, device=device),
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
        solver = SolverSteel(model)
        contacts = newton.Contacts(rigid_contact_max=2, soft_contact_max=0, device=model.device)

        with self.assertRaisesRegex(ValueError, "contact capacity"):
            solver.step(model.state(), model.state(), model.control(), contacts, 1.0 / 60.0)

    @unittest.skipUnless(wp.is_cuda_available(), "propagation_colored prebuild requires CUDA")
    def test_colored_prebuild_preserves_mixed_contact_units(self):
        """Schedule friction-gap and patch-selected one-row units alongside three-row units."""
        scenarios = {
            "friction_gap": ((-0.01, 0.01, 0.02, 0.03), 0.0, 0),
            "patch_friction": ((-0.01, -0.01, -0.01, -0.01), float("inf"), 1),
        }
        for name, (contact_phi, friction_gap_threshold, patch_friction) in scenarios.items():
            with self.subTest(name=name):
                self._assert_mixed_units_are_all_scheduled(
                    contact_phi=contact_phi,
                    friction_gap_threshold=friction_gap_threshold,
                    patch_friction=patch_friction,
                )


_UNBOUNDED = 2**31 - 1


def _allocate_overflowing_contacts(device, contact_count, row_capacity):
    """Route ``contact_count`` contacts into one world with ``row_capacity`` rows.

    Contacts alternate between penetrating (three rows: normal plus two friction) and
    separated within the gap (one speculative normal row). Mixed reservation sizes are
    what let a rolled-back counter re-admit a contact behind a rejected one.
    """
    contact_shape0 = wp.zeros((contact_count,), dtype=wp.int32, device=device)
    contact_shape1 = wp.full((contact_count,), -1, dtype=wp.int32, device=device)
    points = np.zeros((contact_count, 3), dtype=np.float32)
    points[:, 2] = np.where(np.arange(contact_count) % 2 == 0, -0.01, 0.01)
    contact_point0 = wp.array(points, dtype=wp.vec3, device=device)
    contact_point1 = wp.zeros((contact_count,), dtype=wp.vec3, device=device)
    contact_normal = wp.array(
        np.tile([0.0, 0.0, -1.0], (contact_count, 1)).astype(np.float32), dtype=wp.vec3, device=device
    )
    contact_slot = wp.full((contact_count,), -1, dtype=wp.int32, device=device)
    contact_path = wp.full((contact_count,), -1, dtype=wp.int32, device=device)
    contact_world = wp.zeros((contact_count,), dtype=wp.int32, device=device)
    slots_needed = wp.zeros((contact_count,), dtype=wp.int32, device=device)
    counter = wp.zeros((1,), dtype=wp.int32, device=device)
    dropped = wp.zeros((1,), dtype=wp.int32, device=device)
    first_rejected = wp.full((1,), _UNBOUNDED, dtype=wp.int32, device=device)
    count = wp.zeros((1,), dtype=wp.int32, device=device)
    wp.launch(
        allocate_world_contact_slots,
        dim=contact_count,
        inputs=[
            wp.array([contact_count], dtype=wp.int32, device=device),
            contact_count,
            contact_shape0,
            contact_shape1,
            contact_point0,
            contact_point1,
            contact_normal,
            wp.zeros((contact_count,), dtype=wp.float32, device=device),
            wp.zeros((contact_count,), dtype=wp.float32, device=device),
            wp.array([wp.transform_identity()], dtype=wp.transform, device=device),
            wp.zeros((1,), dtype=wp.int32, device=device),
            wp.zeros((1,), dtype=wp.int32, device=device),
            wp.zeros((1,), dtype=wp.int32, device=device),
            wp.array([6], dtype=wp.int32, device=device),
            wp.zeros((1,), dtype=wp.int32, device=device),
            wp.ones((1,), dtype=wp.int32, device=device),
            wp.zeros((1,), dtype=wp.int32, device=device),
            0,
            1,
            1,
            0,
            float("inf"),
            row_capacity,
            row_capacity,
            row_capacity,
            0.0,
            FrictionPatches(),
            0,
            wp.zeros((1,), dtype=wp.float32, device=device),
            wp.zeros((1,), dtype=wp.float32, device=device),
            wp.zeros((1,), dtype=wp.float32, device=device),
            wp.zeros((1,), dtype=wp.float32, device=device),
            0,
            wp.zeros((1,), dtype=wp.float32, device=device),
            0,
        ],
        outputs=[
            contact_world,
            contact_slot,
            wp.full((contact_count,), -1, dtype=wp.int32, device=device),
            wp.full((contact_count,), -1, dtype=wp.int32, device=device),
            wp.zeros((1,), dtype=wp.int32, device=device),
            contact_path,
            wp.zeros((1,), dtype=wp.int32, device=device),
            counter,
            wp.zeros((1,), dtype=wp.int32, device=device),
            slots_needed,
            wp.zeros((1,), dtype=wp.int32, device=device),
            wp.zeros((1,), dtype=wp.int32, device=device),
            dropped,
            wp.full((1,), _UNBOUNDED, dtype=wp.int32, device=device),
            wp.full((1,), _UNBOUNDED, dtype=wp.int32, device=device),
            first_rejected,
            wp.zeros((contact_count,), dtype=wp.int32, device=device),
        ],
        device=device,
    )
    wp.launch(
        finalize_mf_constraint_counts,
        dim=1,
        inputs=[counter, row_capacity, first_rejected],
        outputs=[count],
        device=device,
    )
    return (
        contact_slot.numpy(),
        slots_needed.numpy(),
        int(count.numpy()[0]),
        int(counter.numpy()[0]),
        int(dropped.numpy()[0]),
    )


@unittest.skipUnless(wp.is_cuda_available(), "the allocation race needs concurrent GPU reservations")
class TestSteelRowAllocation(unittest.TestCase):
    def test_overflow_keeps_every_accepted_contact_below_the_count(self):
        """No accepted contact may hold rows at or past the finalized count.

        Rolling the shared counter back for a rejected reservation raced with concurrent
        reservations: a contact accepted at slot s could see the counter end below s, so
        its rows were built but never solved while patch links still chained into them.
        With 3000 mixed one- and three-row contacts into 48 rows nearly every reservation
        is rejected, which makes the race easy to hit.
        """
        device = wp.get_device()
        worst_excess = 0
        for _ in range(40):
            slot, needed, count, counter, dropped = _allocate_overflowing_contacts(device, 3000, 48)
            accepted = slot >= 0
            self.assertLessEqual(count, 48)
            self.assertEqual(
                counter,
                int(needed[accepted].sum()) + dropped,
                "counter must equal accepted rows plus rejected reservations",
            )
            excess = int(np.max(slot[accepted] + needed[accepted])) - count if accepted.any() else 0
            worst_excess = max(worst_excess, excess)
            self.assertEqual(int(np.sum(needed[accepted])), count, "accepted rows must fill exactly [0, count)")
        self.assertLessEqual(worst_excess, 0, f"accepted rows reach {worst_excess} past the row count")

    def test_dense_capacity_keeps_room_for_same_articulation_contacts(self):
        """Propagation modes must not shrink the dense budget when contacts can still route there.

        Contacts between two links of one articulation stay on the dense family unless
        ``propagation_same_articulation_rows`` routes them; a 16-row internal reserve
        drops most of them every step.
        """
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_ground_plane()
        root = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()))
        builder.add_shape_box(root, hx=0.1, hy=0.1, hz=0.1)
        j_root = builder.add_joint_revolute(parent=-1, child=root, axis=wp.vec3(0.0, 1.0, 0.0))
        link = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.8), wp.quat_identity()))
        builder.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1)
        j_link = builder.add_joint_revolute(
            parent=root,
            child=link,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.3), wp.quat_identity()),
        )
        builder.add_articulation([j_root, j_link], label="chain")
        model = builder.finalize()
        model.rigid_contact_max = 256
        dense_routed = newton.solvers.SolverSteel(
            model,
            solve_mode="matrix_free",
            articulated_contact_response="propagation_colored",
            articulation_constraint_max=320,
            free_body_constraint_max=256,
        )
        self.assertEqual(int(dense_routed._articulation_constraint_max), 320)
        propagation_routed = newton.solvers.SolverSteel(
            model,
            solve_mode="matrix_free",
            articulated_contact_response="propagation_colored",
            articulation_constraint_max=320,
            free_body_constraint_max=256,
            propagation_same_articulation_rows=True,
        )
        self.assertLess(int(propagation_routed._articulation_constraint_max), 320)

    def test_propagation_constraint_max_sizes_the_propagation_family(self):
        """``propagation_constraint_max`` sets the propagation row capacity on its own.

        In the propagation responses every contact row lives on the propagation family, so its
        capacity defaulted to the sum of the free-body and articulation budgets: shrinking the
        matrix-free budget to the rows it still holds (joint velocity limits) also starved the
        contact rows. The explicit capacity decouples the two.
        """
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_ground_plane()
        box = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
        builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize()
        model.rigid_contact_max = 256
        kwargs = {"solve_mode": "matrix_free", "articulated_contact_response": "propagation_colored"}
        summed = newton.solvers.SolverFeatherPGS(
            model, articulation_constraint_max=32, free_body_constraint_max=256, **kwargs
        )
        self.assertEqual(int(summed._propagation_max_constraints), 256 + 32)
        self.assertIsNone(summed.propagation_constraint_max)
        decoupled = newton.solvers.SolverFeatherPGS(
            model,
            articulation_constraint_max=32,
            free_body_constraint_max=16,
            propagation_constraint_max=288,
            **kwargs,
        )
        self.assertEqual(int(decoupled._propagation_max_constraints), 288)
        self.assertEqual(decoupled.propagation_constraint_max, 288)
        self.assertEqual(int(decoupled._free_body_constraint_max), 16)
        with self.assertRaises(ValueError):
            newton.solvers.SolverFeatherPGS(model, propagation_constraint_max=0, **kwargs)


def _scene(device):
    """Two worlds: a box on the ground next to a fixed-base slider resting on it, one world without contacts."""
    template = newton.ModelBuilder()
    box = template.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
    template.add_shape_box(box, hx=0.1, hy=0.1, hz=0.1)
    link = template.add_link(xform=wp.transform(wp.vec3(0.5, 0.0, 0.1), wp.quat_identity()))
    template.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1)
    template.add_articulation(
        [
            template.add_joint_prismatic(
                -1, link, axis=newton.Axis.Z, parent_xform=wp.transform(wp.vec3(0.5, 0.0, 0.1), wp.quat_identity())
            )
        ]
    )
    lifted = newton.ModelBuilder()
    lifted.add_shape_box(
        lifted.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 3.0), wp.quat_identity())), hx=0.1, hy=0.1, hz=0.1
    )
    builder = newton.ModelBuilder()
    builder.add_world(template)
    builder.add_world(lifted)
    builder.add_ground_plane()
    return builder.finalize(device=device)


def test_watermarks_accumulate_in_captured_steps(test, device):
    """Accumulate the same marks under CUDA graph replay as in eager steps."""
    results = []
    for capture in (False, True):
        model = _scene(device)
        solver = SolverSteel(model, row_watermark=True, friction_anchor_beta=0.0)
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        state_0, state_1 = model.state(), model.state()
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
        control = model.control()

        def substep(
            state_0=state_0, state_1=state_1, contacts=contacts, solver=solver, control=control, pipeline=pipeline
        ):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
            wp.copy(state_0.joint_q, state_1.joint_q)
            wp.copy(state_0.joint_qd, state_1.joint_qd)
            wp.copy(state_0.body_q, state_1.body_q)
            wp.copy(state_0.body_qd, state_1.body_qd)

        if capture:
            substep()
            with wp.ScopedCapture(device) as capture_scope:
                substep()
            for _ in range(9):
                wp.capture_launch(capture_scope.graph)
        else:
            for _ in range(11):
                substep()
        results.append(solver.constraint_row_watermarks())
    test.assertEqual(results[0], results[1])
    test.assertGreater(results[1]["articulation_high_water"], 0)


class TestSteelRowWatermark(unittest.TestCase):
    pass


cuda_devices = get_cuda_test_devices()


add_function_test(
    TestSteelRowWatermark,
    "test_watermarks_accumulate_in_captured_steps",
    test_watermarks_accumulate_in_captured_steps,
    devices=cuda_devices,
)


def _make_chain_world(n_links: int, root_height: float = 1.0, free_box: bool = False) -> newton.ModelBuilder:
    """A fixed-base serial chain of ``n_links`` revolute links (n_links DOFs), optionally with a free box."""
    builder = newton.ModelBuilder()
    joints = []
    prev = -1
    for _ in range(n_links):
        link = builder.add_link()
        builder.add_shape_box(link, hx=0.15, hy=0.03, hz=0.03)
        if prev == -1:
            parent_xform = wp.transform(p=wp.vec3(0.0, 0.0, root_height), q=wp.quat_identity())
        else:
            parent_xform = wp.transform(p=wp.vec3(0.15, 0.0, 0.0), q=wp.quat_identity())
        joints.append(
            builder.add_joint_revolute(
                parent=prev,
                child=link,
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent_xform=parent_xform,
                child_xform=wp.transform(p=wp.vec3(-0.15, 0.0, 0.0), q=wp.quat_identity()),
            )
        )
        prev = link
    builder.add_articulation(joints)
    if free_box:
        body = builder.add_body(xform=wp.transform(wp.vec3(-1.0, 0.0, 0.1), wp.quat_identity()))
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    return builder


def _build_chain_worlds(world_link_counts: list[int], device, root_height=1.0, free_box=None) -> newton.Model:
    """One chain world per entry in ``world_link_counts``, plus a ground plane.

    ``root_height`` and ``free_box`` are either one value for every world or one entry per world.
    """
    count = len(world_link_counts)
    heights = root_height if isinstance(root_height, list) else [root_height] * count
    boxes = free_box if free_box is not None else [False] * count
    scene = newton.ModelBuilder()
    for n_links, height, box in zip(world_link_counts, heights, boxes, strict=True):
        scene.add_world(_make_chain_world(n_links, height, box))
    scene.add_ground_plane()
    return scene.finalize(device=device)


def _final_joint_q(model, steps=60, solve_mode="matrix_free", response="immediate", from_fk=False):
    solver = SolverSteel(
        model, solve_mode=solve_mode, articulation_constraint_max=64, articulated_contact_response=response
    )
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    if from_fk:
        # Start from the joint-space pose instead of the builder's body placement.
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    control = model.control()
    for _ in range(steps):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1.0 / 120.0)
        state_0, state_1 = state_1, state_0
    return state_0.joint_q.numpy()


def test_hetero_matrix_free_constructs(test, device):
    """Construct on worlds whose DOF counts differ."""
    solver = SolverSteel(_build_chain_worlds([1, 3, 1, 3], device), solve_mode="matrix_free")
    np.testing.assert_array_equal(solver._world_dof_count.numpy(), [1, 3, 1, 3])


def test_homogeneous_matrix_free_constructs(test, device):
    """Construct on worlds with identical DOF counts."""
    solver = SolverSteel(_build_chain_worlds([3, 3, 3, 3], device), solve_mode="matrix_free")
    np.testing.assert_array_equal(solver._world_dof_count.numpy(), [3, 3, 3, 3])


def test_hetero_split_constructs(test, device):
    """Construct the split solve on worlds whose DOF counts differ."""
    solver = SolverSteel(_build_chain_worlds([1, 3, 1, 3], device), solve_mode="split")
    test.assertEqual(solver._solve_mode, "split")
    test.assertEqual(solver._size_groups, [3, 1])


def test_homogeneous_split_constructs(test, device):
    """Construct the split solve on worlds with identical DOF counts."""
    solver = SolverSteel(_build_chain_worlds([3, 3, 3, 3], device), solve_mode="split")
    test.assertEqual(solver._solve_mode, "split")
    test.assertEqual(solver._size_groups, [3])


def test_hetero_worlds_match_isolated_worlds(test, device, solve_mode="matrix_free"):
    """Simulate each world of a heterogeneous model exactly as it simulates alone."""
    combined = _final_joint_q(_build_chain_worlds([1, 3], device), solve_mode=solve_mode)
    single = _final_joint_q(_build_chain_worlds([1], device), solve_mode=solve_mode)
    triple = _final_joint_q(_build_chain_worlds([3], device), solve_mode=solve_mode)
    np.testing.assert_allclose(combined, np.concatenate([single, triple]), rtol=0.0, atol=1.0e-5)


def test_homogeneous_propagation_constructs(test, device):
    """Construct every propagation response on worlds with identical DOF counts."""
    model = _build_chain_worlds([3, 3, 3, 3], device)
    for response in ("propagation", "propagation_colored"):
        with test.subTest(response=response):
            solver = SolverSteel(model, solve_mode="matrix_free", articulated_contact_response=response)
            test.assertEqual(solver._articulated_contact_response, response)
            np.testing.assert_array_equal(solver._world_dof_count.numpy(), [3, 3, 3, 3])


def test_hetero_propagation_matches_isolated_worlds(test, device):
    """Simulate each world of a heterogeneous model with a propagation response exactly as it simulates alone.

    The chains hang low enough to hit the ground, so every world solves propagation contact rows.
    """

    def run(link_counts, free_box, response):
        model = _build_chain_worlds(link_counts, device, 0.35, free_box)
        return _final_joint_q(model, response=response, from_fk=True)

    for response in ("propagation", "propagation_colored"):
        with test.subTest(response=response):
            combined = run([1, 3], None, response)
            single = run([1], None, response)
            triple = run([3], None, response)
            # The cached response sizes its reductions by the model's body capacity, so the sums reassociate.
            np.testing.assert_allclose(combined, np.concatenate([single, triple]), rtol=0.0, atol=1.0e-4)
            # One articulation size, with a free box that still makes the world DOF counts differ.
            combined = run([3, 3], [False, True], response)
            plain = run([3], [False], response)
            boxed = run([3], [True], response)
            np.testing.assert_allclose(combined, np.concatenate([plain, boxed]), rtol=0.0, atol=1.0e-4)


def test_hetero_propagation_keeps_the_next_world_velocities(test, device):
    """Keep a busy 3-DOF world from writing over the next, 9-DOF world in the propagation responses.

    The first chain lies on the ground and solves many rows; the second hangs clear above a resting box.
    """

    def run(root_height, free_box, response):
        model = _build_chain_worlds([3] * len(free_box), device, root_height, free_box)
        return _final_joint_q(model, response=response, from_fk=True)

    for response in ("propagation", "propagation_colored"):
        with test.subTest(response=response):
            combined = run([0.03, 2.0], [False, True], response)
            lying = run([0.03], [False], response)
            hanging = run([2.0], [True], response)
            np.testing.assert_allclose(combined, np.concatenate([lying, hanging]), rtol=0.0, atol=1.0e-4)


class TestSteelHeteroWorlds(unittest.TestCase):
    pass


add_function_test(
    TestSteelHeteroWorlds, "test_hetero_matrix_free_constructs", test_hetero_matrix_free_constructs, devices=devices
)


add_function_test(
    TestSteelHeteroWorlds,
    "test_homogeneous_matrix_free_constructs",
    test_homogeneous_matrix_free_constructs,
    devices=devices,
)


add_function_test(
    TestSteelHeteroWorlds,
    "test_hetero_worlds_match_isolated_worlds",
    test_hetero_worlds_match_isolated_worlds,
    devices=devices,
)


split_devices = get_test_devices()


add_function_test(
    TestSteelHeteroWorlds, "test_hetero_split_constructs", test_hetero_split_constructs, devices=split_devices
)


add_function_test(
    TestSteelHeteroWorlds,
    "test_homogeneous_split_constructs",
    test_homogeneous_split_constructs,
    devices=split_devices,
)


add_function_test(
    TestSteelHeteroWorlds,
    "test_hetero_worlds_match_isolated_worlds_split",
    test_hetero_worlds_match_isolated_worlds,
    devices=split_devices,
    solve_mode="split",
)


add_function_test(
    TestSteelHeteroWorlds,
    "test_homogeneous_propagation_constructs",
    test_homogeneous_propagation_constructs,
    devices=devices,
)


add_function_test(
    TestSteelHeteroWorlds,
    "test_hetero_propagation_matches_isolated_worlds",
    test_hetero_propagation_matches_isolated_worlds,
    devices=devices,
)


add_function_test(
    TestSteelHeteroWorlds,
    "test_hetero_propagation_keeps_the_next_world_velocities",
    test_hetero_propagation_keeps_the_next_world_velocities,
    devices=devices,
)


# Arm lengths and heaps differ per world, so DOF and body counts differ too.
_HETERO_HEAPS = ({"links": 3, "heap": (2, 2, 1)}, {"links": 6, "heap": (4, 2, 1)})
_HETERO_HEAP_STEPS = 90
_HETERO_HEAP_FIELDS = ("body_q", "body_qd", "joint_q", "joint_qd")


def _build_hetero_heap_model(worlds: tuple[dict, ...]) -> newton.Model:
    scene = newton.ModelBuilder(up_axis=newton.Axis.Z)
    scene.rigid_gap = 0.005
    for world_kwargs in worlds:
        world = newton.ModelBuilder(up_axis=newton.Axis.Z)
        world.rigid_gap = 0.005
        _add_arm_on_heap(world, **world_kwargs)
        scene.add_world(world)
    scene.add_ground_plane()
    model = scene.finalize(device="cuda:0")
    model.rigid_contact_max = 4096 * len(worlds)
    set_torsion_radius(model, 0.01)
    return model


def _hetero_heap_rollout(worlds: tuple[dict, ...], capture_at: int | None = None):
    """Roll out the heap worlds with every colored contact feature on; return the final state and solver."""
    stepper = _Stepper(
        _build_hetero_heap_model(worlds),
        contact_torsion_device=True,
        free_rigid_torsional_friction=0.01,
        free_rigid_rolling_friction=0.01,
        enable_sleeping=True,
        sleep_quiet_time=0.05,
    )
    graph = None
    for step in range(_HETERO_HEAP_STEPS):
        if step == capture_at:
            with wp.ScopedCapture() as capture:
                stepper.step()
            graph = capture.graph
        if graph is not None:
            wp.capture_launch(graph)
        else:
            stepper.step()
    return stepper.snapshot(), stepper.solver


def _split_by_world(model: newton.Model, snapshot: dict) -> list[dict]:
    body_world = model.body_world.numpy()
    joint_world = model.joint_world.numpy()
    q_start, qd_start = model.joint_q_start.numpy(), model.joint_qd_start.numpy()
    worlds = []
    for world in range(model.world_count):
        joints = np.flatnonzero(joint_world == world)
        worlds.append(
            {
                "body_q": snapshot["body_q"][body_world == world],
                "body_qd": snapshot["body_qd"][body_world == world],
                "joint_q": np.concatenate([snapshot["joint_q"][q_start[j] : q_start[j + 1]] for j in joints]),
                "joint_qd": np.concatenate([snapshot["joint_qd"][qd_start[j] : qd_start[j + 1]] for j in joints]),
            }
        )
    return worlds


@unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
class TestSteelHeteroColored(unittest.TestCase):
    """Deterministic colored propagation on worlds whose DOF and body counts differ."""

    def test_rollouts_match_isolated_worlds_bitwise(self):
        """Match isolated worlds, a repeat and a captured replay bitwise, with torsion, angular rows and sleeping."""
        combined, solver = _hetero_heap_rollout(_HETERO_HEAPS)
        repeat, _ = _hetero_heap_rollout(_HETERO_HEAPS)
        captured, _ = _hetero_heap_rollout(_HETERO_HEAPS, capture_at=2)
        for name in _HETERO_HEAP_FIELDS:
            np.testing.assert_array_equal(repeat[name], combined[name], err_msg=f"repeat {name}")
            np.testing.assert_array_equal(captured[name], combined[name], err_msg=f"captured {name}")
        for world, (world_state, world_kwargs) in enumerate(
            zip(_split_by_world(solver.model, combined), _HETERO_HEAPS, strict=True)
        ):
            isolated, _ = _hetero_heap_rollout((world_kwargs,))
            for name in _HETERO_HEAP_FIELDS:
                np.testing.assert_array_equal(world_state[name], isolated[name], err_msg=f"world {world} {name}")

        counts = solver._world_dof_count.numpy()
        self.assertNotEqual(counts[0], counts[1], "worlds must differ in DOF count")
        rows = solver._propagation_constraint_count.numpy()
        row_types = np.concatenate([solver._propagation_row_type.numpy()[w, :n] for w, n in enumerate(rows)])
        for row_type in (
            PGS_CONSTRAINT_TYPE_TORSION,
            PGS_CONSTRAINT_TYPE_CONTACT_ROLLING,
            PGS_CONSTRAINT_TYPE_CONTACT_SPIN,
        ):
            self.assertGreater(int(np.sum(row_types == row_type)), 0, f"row type {row_type} must be active")
        self.assertGreater(int(np.sum(solver._sleeping.body_awake.numpy() == 0)), 0, "some bodies must sleep")


if __name__ == "__main__":
    unittest.main()
