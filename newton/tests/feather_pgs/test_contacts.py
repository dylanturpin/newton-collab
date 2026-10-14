# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact controls, contact forces, restitution and paired responses of SolverFeatherPGS."""

import inspect
import itertools
import unittest
import warnings

import numpy as np
import warp as wp

import newton
import newton._src.solvers.feather_pgs.kernels
from newton._src.geometry.sdf_hydroelastic import HydroelasticSDF
from newton._src.solvers.feather_pgs._testing import kernel_overrides
from newton._src.solvers.feather_pgs.friction_patches import FrictionPatches, contact_tangent_basis
from newton._src.solvers.feather_pgs.kernels import (
    allocate_world_contact_slots,
    apply_world_contact_restitution_accumulated,
    compute_mf_effective_mass_and_rhs,
    compute_mf_rhs_bias,
    compute_propagation_effective_mass_and_rhs,
    compute_world_contact_bias,
    populate_world_J_for_compact_size,
    populate_world_J_for_size,
    prepare_world_contact_rows,
)
from newton.solvers import SolverFeatherPGS, SolverObservableFlags
from newton.tests.feather_pgs._fixtures import (
    _build_sparse_contact_friction_model,
    _sphere_on_ground,
    set_torsion_radius,
)
from newton.tests.test_contact_reduction_body_pairs import _cylinder_foot
from newton.tests.unittest_utils import (
    StdOutCapture,
    add_function_test,
    get_cuda_test_devices,
    get_selected_cuda_test_devices,
    get_test_devices,
)

PATH_DENSE = 0


PATH_MATRIX_FREE = 1


PATH_PROPAGATION = 2


def _disabled_patches(device):
    patches = FrictionPatches()
    patches.phi = wp.zeros(1, dtype=wp.vec2, device=device)
    return patches


def _launch_contact_allocator(
    *,
    route: int,
    gap: float,
    gate: float | None,
    responsive: bool = True,
    enable_friction: bool = False,
    friction_gap: float = float("inf"),
):
    """Allocate one contact and return its route metadata and counters."""
    device = "cpu"
    outputs = {
        "world": wp.full((1,), -9, dtype=wp.int32, device=device),
        "slot": wp.full((1,), -9, dtype=wp.int32, device=device),
        "art_a": wp.full((1,), -9, dtype=wp.int32, device=device),
        "art_b": wp.full((1,), -9, dtype=wp.int32, device=device),
        "path": wp.full((1,), -9, dtype=wp.int32, device=device),
        "slots_needed": wp.full((1,), -9, dtype=wp.int32, device=device),
    }
    counters = {
        "dense_count": wp.zeros((1,), dtype=wp.int32, device=device),
        "mf_count": wp.zeros((1,), dtype=wp.int32, device=device),
        "propagation_count": wp.zeros((1,), dtype=wp.int32, device=device),
        "dense_world_flag": wp.zeros((1,), dtype=wp.int32, device=device),
        "dense_dropped": wp.zeros((1,), dtype=wp.int32, device=device),
        "mf_dropped": wp.zeros((1,), dtype=wp.int32, device=device),
        "propagation_dropped": wp.zeros((1,), dtype=wp.int32, device=device),
        "dense_first_rejected": wp.full((1,), 2**31 - 1, dtype=wp.int32, device=device),
        "mf_first_rejected": wp.full((1,), 2**31 - 1, dtype=wp.int32, device=device),
        "propagation_first_rejected": wp.full((1,), 2**31 - 1, dtype=wp.int32, device=device),
    }
    is_free = route == PATH_MATRIX_FREE
    propagation_enabled = route == PATH_PROPAGATION
    wp.launch(
        allocate_world_contact_slots,
        dim=1,
        inputs=[
            wp.array([1], dtype=wp.int32, device=device),
            1,
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([-1], dtype=wp.int32, device=device),
            wp.array([wp.vec3(gap, 0.0, 0.0)], dtype=wp.vec3, device=device),
            wp.array([wp.vec3(0.0)], dtype=wp.vec3, device=device),
            wp.array([wp.vec3(-1.0, 0.0, 0.0)], dtype=wp.vec3, device=device),
            wp.zeros((1,), dtype=wp.float32, device=device),
            wp.zeros((1,), dtype=wp.float32, device=device),
            wp.array([wp.transform_identity()], dtype=wp.transform, device=device),
            wp.array([wp.transform_identity()], dtype=wp.transform, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([1], dtype=wp.int32, device=device),
            wp.zeros((1,), dtype=wp.int32, device=device),
            wp.array([int(responsive)], dtype=wp.int32, device=device),
            wp.array([int(is_free)], dtype=wp.int32, device=device),
            1,
            int(propagation_enabled),
            0,
            0,
            float("inf") if gate is None else gate,
            8,
            8,
            8,
            friction_gap if enable_friction else -float("inf"),  # no friction rows
            FrictionPatches(),
            0,
            wp.zeros((1,), dtype=wp.float32, device=device),
            wp.zeros((1,), dtype=wp.float32, device=device),
            0,
        ],
        outputs=[
            outputs["world"],
            outputs["slot"],
            outputs["art_a"],
            outputs["art_b"],
            counters["dense_count"],
            outputs["path"],
            counters["mf_count"],
            counters["propagation_count"],
            counters["dense_world_flag"],
            outputs["slots_needed"],
            counters["dense_dropped"],
            counters["mf_dropped"],
            counters["propagation_dropped"],
            counters["dense_first_rejected"],
            counters["mf_first_rejected"],
            counters["propagation_first_rejected"],
        ],
        device=device,
    )
    wp.synchronize_device(device)
    return {name: int(array.numpy()[0]) for name, array in (outputs | counters).items()}


def _dense_speculative_rhs(scale: float) -> float:
    """Return the dense positive-gap RHS for a requested speculative scale."""
    rhs = wp.zeros((1, 1), dtype=wp.float32, device="cpu")
    wp.launch(
        compute_world_contact_bias,
        dim=1,
        inputs=[
            wp.array([1], dtype=wp.int32, device="cpu"),
            wp.array([[1.0]], dtype=wp.float32, device="cpu"),
            wp.array([[0.2]], dtype=wp.float32, device="cpu"),
            wp.array([[0]], dtype=wp.int32, device="cpu"),
            wp.zeros((1, 1), dtype=wp.float32, device="cpu"),
            0.5,
            1.0,
            scale,
            1.0,
            1.0,
        ],
        outputs=[rhs, wp.zeros((1, 1), dtype=wp.float32, device="cpu")],
        device="cpu",
    )
    return float(rhs.numpy()[0, 0])


def _mf_speculative_rhs(scale: float) -> float:
    """Return the MF setup RHS for a requested speculative scale."""
    rhs = wp.zeros((1, 1), dtype=wp.float32, device="cpu")
    wp.launch(
        compute_mf_effective_mass_and_rhs,
        dim=1,
        inputs=[
            wp.array([1], dtype=wp.int32, device="cpu"),
            wp.array([[-1]], dtype=wp.int32, device="cpu"),
            wp.array([[-1]], dtype=wp.int32, device="cpu"),
            wp.zeros((1, 1, 6), dtype=wp.float32, device="cpu"),
            wp.zeros((1, 1, 6), dtype=wp.float32, device="cpu"),
            wp.zeros((1,), dtype=wp.spatial_matrix, device="cpu"),
            wp.array([[1.0]], dtype=wp.float32, device="cpu"),
            wp.array([[0]], dtype=wp.int32, device="cpu"),
            wp.zeros((1, 1), dtype=wp.float32, device="cpu"),
            wp.zeros((1, 1), dtype=wp.float32, device="cpu"),
            0,
            wp.array([-1], dtype=wp.int32, device="cpu"),
            wp.array([0], dtype=wp.int32, device="cpu"),
            wp.zeros((1,), dtype=wp.float32, device="cpu"),
            wp.array([float("inf")], dtype=wp.float32, device="cpu"),
            1.0,
            0.2,
            1.0,
            0.5,
            scale,
            0.5,
            1,
        ],
        outputs=[
            wp.zeros((1, 1), dtype=wp.float32, device="cpu"),
            wp.zeros((1, 1, 6), dtype=wp.float32, device="cpu"),
            wp.zeros((1, 1, 6), dtype=wp.float32, device="cpu"),
            rhs,
            wp.zeros((1, 1), dtype=wp.float32, device="cpu"),
        ],
        device="cpu",
    )
    return float(rhs.numpy()[0, 0])


def _propagation_speculative_rhs(scale: float) -> float:
    """Return the propagation setup RHS for a requested speculative scale."""
    rhs = wp.zeros((1, 1), dtype=wp.float32, device="cpu")
    wp.launch(
        compute_propagation_effective_mass_and_rhs,
        dim=1,
        inputs=[
            wp.array([1], dtype=wp.int32, device="cpu"),
            wp.array([[-1]], dtype=wp.int32, device="cpu"),
            wp.array([[-1]], dtype=wp.int32, device="cpu"),
            wp.zeros((1, 1, 6), dtype=wp.float32, device="cpu"),
            wp.zeros((1, 1, 6), dtype=wp.float32, device="cpu"),
            wp.zeros((1, 6, 6), dtype=wp.float32, device="cpu"),
            wp.zeros((1,), dtype=wp.int32, device="cpu"),
            wp.ones((1,), dtype=wp.int32, device="cpu"),
            wp.array([[1.0]], dtype=wp.float32, device="cpu"),
            wp.array([[0]], dtype=wp.int32, device="cpu"),
            wp.zeros((1, 1), dtype=wp.float32, device="cpu"),
            wp.zeros((1, 6), dtype=wp.float32, device="cpu"),
            wp.array([float("inf")], dtype=wp.float32, device="cpu"),
            1.0,
            0.2,
            1.0,
            0.5,
            scale,
            0.5,
            1,
        ],
        outputs=[
            wp.zeros((1, 1), dtype=wp.float32, device="cpu"),
            wp.zeros((1, 1, 6), dtype=wp.float32, device="cpu"),
            wp.zeros((1, 1, 6), dtype=wp.float32, device="cpu"),
            rhs,
            wp.zeros((1, 1), dtype=wp.float32, device="cpu"),
            wp.zeros((1, 1), dtype=wp.float32, device="cpu"),
        ],
        device="cpu",
    )
    return float(rhs.numpy()[0, 0])


def _dense_restitution_rhs(scale: float) -> float:
    """Apply dense restitution after accumulating the scaled position RHS."""
    rhs = wp.array([[-2.0 + scale]], dtype=wp.float32, device="cpu")
    wp.launch(
        apply_world_contact_restitution_accumulated,
        dim=1,
        inputs=[
            wp.array([1], dtype=wp.int32, device="cpu"),
            1,
            wp.array([[1.0]], dtype=wp.float32, device="cpu"),
            wp.array([[0.2]], dtype=wp.float32, device="cpu"),
            wp.array([[0]], dtype=wp.int32, device="cpu"),
            wp.array([[0.5]], dtype=wp.float32, device="cpu"),
            1.0,
            scale,
            0.0,
            0,
        ],
        outputs=[rhs, wp.zeros((1, 1), dtype=wp.float32, device="cpu")],
        device="cpu",
    )
    return float(rhs.numpy()[0, 0])


def _contact_row_outputs(device: str) -> dict[str, wp.array]:
    """Allocate dense-contact Jacobian and metadata outputs."""
    return {
        "J": wp.zeros((1, 8, 3), dtype=wp.float32, device=device),
        "row_type": wp.full((1, 8), -9, dtype=wp.int32, device=device),
        "row_parent": wp.full((1, 8), -9, dtype=wp.int32, device=device),
        "row_mu": wp.full((1, 8), -9.0, dtype=wp.float32, device=device),
        "row_beta": wp.full((1, 8), -9.0, dtype=wp.float32, device=device),
        "row_cfm": wp.full((1, 8), -9.0, dtype=wp.float32, device=device),
        "phi": wp.full((1, 8), -9.0, dtype=wp.float32, device=device),
        "target_velocity": wp.full((1, 8), -9.0, dtype=wp.float32, device=device),
        "row_restitution": wp.full((1, 8), -9.0, dtype=wp.float32, device=device),
    }


def _launch_dense_contact_builders(device: str = "cpu") -> tuple[dict[str, wp.array], dict[str, wp.array]]:
    """Build one same-articulation contact through serial and compact paths."""
    contact_count = wp.array([1], dtype=wp.int32, device=device)
    point0 = wp.array([wp.vec3(0.2, 0.1, -0.05)], dtype=wp.vec3, device=device)
    point1 = wp.array([wp.vec3(-0.1, 0.05, 0.2)], dtype=wp.vec3, device=device)
    normal = wp.array([wp.vec3(0.0, 0.0, -1.0)], dtype=wp.vec3, device=device)
    shape0 = wp.array([0], dtype=wp.int32, device=device)
    shape1 = wp.array([1], dtype=wp.int32, device=device)
    thickness0 = wp.array([0.01], dtype=wp.float32, device=device)
    thickness1 = wp.array([0.02], dtype=wp.float32, device=device)
    contact_world = wp.array([0], dtype=wp.int32, device=device)
    contact_slot = wp.array([0], dtype=wp.int32, device=device)
    contact_art_a = wp.array([0], dtype=wp.int32, device=device)
    contact_art_b = wp.array([0], dtype=wp.int32, device=device)
    contact_path = wp.array([PATH_DENSE], dtype=wp.int32, device=device)
    contact_slots_needed = wp.array([3], dtype=wp.int32, device=device)
    response_dof_count = wp.array([3], dtype=wp.int32, device=device)
    art_group_idx = wp.array([0], dtype=wp.int32, device=device)
    art_dof_start = wp.array([0], dtype=wp.int32, device=device)
    articulation_origin = wp.array([wp.vec3(0.1, -0.2, 0.3)], dtype=wp.vec3, device=device)
    body_to_joint = wp.array([0, 1, 2], dtype=wp.int32, device=device)
    joint_ancestor = wp.array([-1, 0, 1], dtype=wp.int32, device=device)
    joint_qd_start = wp.array([0, 1, 2, 3], dtype=wp.int32, device=device)
    joint_s = wp.array(
        [
            wp.spatial_vector(1.0, 0.0, 0.0, 0.0, 0.0, 0.5),
            wp.spatial_vector(0.0, 1.0, 0.0, 0.25, 0.0, 0.0),
            wp.spatial_vector(0.0, 0.0, 1.0, 0.0, -0.5, 0.0),
        ],
        dtype=wp.spatial_vector,
        device=device,
    )
    shape_body = wp.array([2, 1], dtype=wp.int32, device=device)
    body_q = wp.array(
        [
            wp.transform(wp.vec3(0.0), wp.quat_identity()),
            wp.transform(wp.vec3(-0.2, 0.3, 0.1), wp.quat_identity()),
            wp.transform(wp.vec3(0.4, -0.1, 0.2), wp.quat_identity()),
        ],
        dtype=wp.transform,
        device=device,
    )
    body_v = wp.zeros((3,), dtype=wp.spatial_vector, device=device)
    prescribed = wp.zeros((1,), dtype=wp.int32, device=device)
    shape_transform = wp.array([wp.transform_identity(), wp.transform_identity()], dtype=wp.transform, device=device)
    material_mu = wp.array([0.6, 0.8], dtype=wp.float32, device=device)
    material_restitution = wp.array([0.2, 0.4], dtype=wp.float32, device=device)

    common_geometry = [
        contact_count,
        point0,
        point1,
        normal,
        shape0,
        shape1,
        thickness0,
        thickness1,
    ]
    serial = _contact_row_outputs(device)
    wp.launch(
        populate_world_J_for_size,
        dim=1,
        inputs=[
            contact_count,
            1,
            *common_geometry[1:],
            contact_world,
            contact_slot,
            contact_art_a,
            contact_art_b,
            contact_path,
            3,
            response_dof_count,
            art_group_idx,
            art_dof_start,
            articulation_origin,
            body_to_joint,
            joint_ancestor,
            joint_qd_start,
            joint_s,
            shape_body,
            body_q,
            body_v,
            prescribed,
            shape_transform,
            material_mu,
            material_restitution,
            1.0,
            1,
            wp.zeros((1,), dtype=wp.int32, device=device),
            1.0,
            0,
            0.05,
            1.0e-6,
            _disabled_patches(device),  # patch friction off
            0.0,  # friction_anchor_beta
        ],
        outputs=list(serial.values()),
        device=device,
    )

    compact = _contact_row_outputs(device)
    wp.launch(
        prepare_world_contact_rows,
        dim=1,
        inputs=[
            contact_count,
            1,
            *common_geometry[1:],
            contact_world,
            contact_slot,
            contact_art_a,
            contact_art_b,
            contact_path,
            contact_slots_needed,
            shape_body,
            body_q,
            body_v,
            prescribed,
            articulation_origin,
            material_mu,
            material_restitution,
            1,
            1.0,
            0,
            0.05,
            1.0e-6,
            _disabled_patches(device),  # patch friction off
            0.0,  # friction_anchor_beta
        ],
        outputs=list(compact.values())[1:],
        device=device,
    )
    wp.launch(
        populate_world_J_for_compact_size,
        dim=(1, 32),
        inputs=[
            contact_count,
            1,
            *common_geometry[1:],
            contact_slot,
            contact_art_a,
            contact_art_b,
            contact_path,
            contact_slots_needed,
            3,
            response_dof_count,
            art_group_idx,
            art_dof_start,
            articulation_origin,
            wp.array([1, 3, 7], dtype=wp.uint32, device=device),
            joint_s,
            shape_body,
            body_q,
            1,
            FrictionPatches(),
            0,
        ],
        outputs=[compact["J"]],
        device=device,
    )
    wp.synchronize_device(device)
    return serial, compact


class TestFeatherPGSContactControls(unittest.TestCase):
    def test_compact_contact_builder_matches_tree_walk(self):
        """Match dense Jacobians and metadata for a same-articulation contact."""
        serial, compact = _launch_dense_contact_builders()

        for name in serial:
            with self.subTest(output=name):
                np.testing.assert_allclose(compact[name].numpy(), serial[name].numpy(), rtol=0.0, atol=1.0e-6)

    def test_solver_exposes_documented_defaults(self):
        """Expose legacy-preserving defaults through the public constructor."""
        solver = SolverFeatherPGS(newton.ModelBuilder().finalize(device="cpu"))

        self.assertEqual(solver._contact_speculative_scale, 1.0)
        self.assertIsNone(solver._contact_gap_gate)
        self.assertIsNone(solver._contact_friction_gap_max)
        self.assertTrue(solver._warn_constraint_overflow)
        self.assertTrue(solver._track_row_capacity)
        self.assertFalse(solver._row_watermark)
        quiet_solver = SolverFeatherPGS(
            newton.ModelBuilder().finalize(device="cpu"),
            warn_constraint_overflow=False,
        )
        self.assertFalse(quiet_solver._warn_constraint_overflow)
        self.assertFalse(quiet_solver._track_row_capacity)
        self.assertIsNotNone(quiet_solver._row_overflow_warning_emitted)

    def test_new_contact_controls_preserve_legacy_positional_layout(self):
        """Append new controls without shifting established positional arguments."""
        parameters = tuple(inspect.signature(SolverFeatherPGS).parameters)
        for sequence in (
            ("contact_friction_shared_anchor", "contact_friction_scale", "contact_shared_anchor"),
            ("row_watermark", "restitution_velocity_threshold", "contact_speculative_scale", "contact_gap_gate"),
        ):
            start = parameters.index(sequence[0])
            self.assertEqual(parameters[start : start + len(sequence)], sequence)

        legacy_tail = parameters.index("contact_gap_gate")
        for name in ("enable_restitution", "warn_constraint_overflow"):
            self.assertGreater(parameters.index(name), legacy_tail)

    def test_solver_validates_and_stores_contact_controls(self):
        """Accept finite non-negative controls and reject malformed values."""
        model = newton.ModelBuilder().finalize(device="cpu")
        solver = SolverFeatherPGS(
            model,
            contact_speculative_scale=0.0,
            contact_gap_gate=0.001,
        )
        self.assertEqual(solver._contact_speculative_scale, 0.0)
        self.assertEqual(solver._contact_gap_gate, 0.001)

        for name in ("contact_speculative_scale", "contact_gap_gate"):
            for value in (-0.1, float("nan"), float("inf"), "invalid"):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        SolverFeatherPGS(model, **{name: value})

    def test_speculative_scale_controls_every_position_rhs_family(self):
        """Scale positive-gap position bias on dense, MF, and propagation rows."""
        rhs_families = {
            "dense": _dense_speculative_rhs,
            "matrix_free": _mf_speculative_rhs,
            "propagation": _propagation_speculative_rhs,
        }
        for name, compute_rhs in rhs_families.items():
            with self.subTest(family=name):
                self.assertEqual(compute_rhs(0.0), 0.0)
                self.assertAlmostEqual(compute_rhs(1.0), 2.0, places=6)

    def test_dense_restitution_removes_the_scaled_position_bias(self):
        """Recover the same incident speed before restitution at scales zero and one."""
        self.assertEqual(_dense_restitution_rhs(0.0), -3.0)
        self.assertEqual(_dense_restitution_rhs(1.0), -3.0)

    def test_gap_gate_prevents_all_route_allocations(self):
        """Drop positive gaps above the gate before any route reserves a slot; no gate keeps them."""
        for route in (PATH_DENSE, PATH_MATRIX_FREE, PATH_PROPAGATION):
            with self.subTest(route=route):
                allocated = _launch_contact_allocator(route=route, gap=0.002, gate=None)
                self.assertEqual(allocated["path"], route)
                self.assertEqual(allocated["slot"], 0)
                self.assertEqual(allocated["world"], 0)
                self.assertEqual(allocated["art_a"], 0)
                self.assertEqual(allocated["art_b"], -1)
                self.assertEqual(allocated["slots_needed"], 1)
                self.assertEqual(
                    (
                        allocated["dense_count"],
                        allocated["mf_count"],
                        allocated["propagation_count"],
                    ),
                    (
                        int(route == PATH_DENSE),
                        int(route == PATH_MATRIX_FREE),
                        int(route == PATH_PROPAGATION),
                    ),
                )

                dropped = _launch_contact_allocator(route=route, gap=0.002, gate=0.001)
                self.assertEqual(dropped["slot"], -1)
                self.assertEqual(dropped["path"], -1)
                self.assertEqual(dropped["dense_count"], 0)
                self.assertEqual(dropped["mf_count"], 0)
                self.assertEqual(dropped["propagation_count"], 0)
                self.assertEqual(dropped["dense_world_flag"], 0)

    def test_gap_gate_keeps_contacts_at_threshold(self):
        """Keep a contact whose gap equals the positive gate exactly."""
        for route in (PATH_DENSE, PATH_MATRIX_FREE, PATH_PROPAGATION):
            with self.subTest(route=route):
                result = _launch_contact_allocator(route=route, gap=0.001, gate=0.001)
                self.assertEqual(result["path"], route)
                self.assertEqual(result["slots_needed"], 1)

    def test_allocator_skips_contacts_without_response_dofs(self):
        """Do not allocate a row when neither contact body can change velocity."""
        result = _launch_contact_allocator(route=PATH_DENSE, gap=-0.001, gate=None, responsive=False)

        self.assertEqual(result["slot"], -1)
        self.assertEqual(result["path"], -1)
        self.assertEqual(result["dense_count"], 0)

    @unittest.skipUnless(wp.is_cuda_available(), "matrix-free solve requires CUDA")
    def test_friction_gap_max_keeps_the_constructed_row_layout(self):
        """Reject a switch to or from ``None`` after construction, and match a fresh solver after an allowed edit."""
        model = _build_sparse_contact_friction_model(device="cuda:0")
        # Tilted boxes make the triple and normal-only schedules give different motion.
        transforms = model.shape_transform.numpy()
        transforms[model.shape_body.numpy() >= 0, 3:] = np.asarray(wp.quat_rpy(0.2, 0.3, 0.1))
        model.shape_transform.assign(transforms)
        options = {
            "solve_mode": "matrix_free",
            "iterations": 8,
            "enable_joint_limits": True,
            "articulation_constraint_max": 96,
            "free_body_constraint_max": 32,
            "use_parallel_streams": True,
            "friction_anchor_beta": 0.0,
        }
        every_contact = SolverFeatherPGS(model, **options)
        for value in (-float("inf"), -1.0, 0.01):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "contact_friction_gap_max"):
                every_contact.contact_friction_gap_max = value
        every_contact.contact_friction_gap_max = float("inf")
        gated = SolverFeatherPGS(model, **options, contact_friction_gap_max=0.01)
        with self.assertRaisesRegex(ValueError, "contact_friction_gap_max"):
            gated.contact_friction_gap_max = None
        gated.contact_friction_gap_max = -float("inf")
        fresh = SolverFeatherPGS(model, **options, contact_friction_gap_max=-float("inf"))
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", reduce_contacts=False)
        contacts = pipeline.contacts()
        trajectories = []
        for solver in (gated, fresh):
            state, output = model.state(), model.state()
            qd = state.joint_qd.numpy()
            qd[:] = -0.3
            state.joint_qd.assign(qd)
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            trajectory = []
            for _ in range(4):
                pipeline.collide(state, contacts)
                solver.step(state, output, model.control(), contacts, 1.0 / 240.0)
                solver.check_constraint_capacity()
                state, output = output, state
                trajectory.append(np.concatenate([state.joint_q.numpy(), state.joint_qd.numpy()]))
            trajectories.append(np.stack(trajectory))
        np.testing.assert_array_equal(trajectories[0], trajectories[1])

    def test_overflow_warning_enabled_after_construction_latches(self):
        """Latch the dense-row overflow warning when warnings are enabled after a quiet overflowing step."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        scene = newton.ModelBuilder()
        scene.add_ground_plane()
        xform = wp.transform(wp.vec3(0.0, 0.0, 0.049), wp.quat_identity())
        body = scene.add_link(xform=xform)
        scene.add_shape_box(body, hx=0.15, hy=0.1, hz=0.05)
        scene.add_articulation([scene.add_joint_prismatic(-1, body, axis=newton.Axis.Z, parent_xform=xform)])
        model = scene.finalize(device=device)
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn", reduce_contacts=False)
        state, output = model.state(), model.state()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        solver = SolverFeatherPGS(
            model, solve_mode="split", articulation_constraint_max=1, warn_constraint_overflow=False
        )
        solver.step(state, output, model.control(), contacts, 1.0 / 240.0)
        self.assertTrue(solver.constraint_overflow.numpy().any())
        self.assertEqual(solver._row_overflow_warning_emitted.numpy().tolist(), [0, 0, 0])
        solver.warn_constraint_overflow = True
        capture = StdOutCapture()
        capture.begin()
        try:
            solver.step(state, output, model.control(), contacts, 1.0 / 240.0)
        finally:
            try:
                wp.synchronize()
            finally:
                printed = capture.end()
        self.assertRegex(printed, "constraint-row overflow")
        self.assertEqual(int(solver._row_overflow_warning_emitted.numpy()[0]), 1)
        self.assertTrue(np.isfinite(output.joint_q.numpy()).all())


def _free_body_model(device):
    builder = newton.ModelBuilder()
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()))
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    return builder.finalize(device=device)


def test_shared_anchors_warn_with_patch_friction(test, device):
    """Explain that patch anchors keep their friction points when shared anchors are requested."""
    model = _free_body_model(device)
    for flag in ("contact_shared_anchor", "contact_friction_shared_anchor"):
        with test.subTest(flag=flag), test.assertWarnsRegex(UserWarning, "friction_anchor_beta=0"):
            SolverFeatherPGS(model, **{flag: True})
        solver = SolverFeatherPGS(model, friction_anchor_beta=0.0, **{flag: True})
        test.assertTrue(getattr(solver, flag))


class TestFeatherPGSContactControlsUpstream(unittest.TestCase):
    pass


devices = get_cuda_test_devices()


for _fn in (test_shared_anchors_warn_with_patch_friction,):
    add_function_test(TestFeatherPGSContactControlsUpstream, _fn.__name__, _fn, devices=devices)


DT = 1.0 / 120.0


BOX_MASS = 2.0


def _box_on_ground(device, gravity=None, mu=None):
    """Box resting on the ground."""
    builder = newton.ModelBuilder() if gravity is None else newton.ModelBuilder(gravity=gravity)
    shape_cfg = newton.ModelBuilder.ShapeConfig(density=0.0)
    if mu is not None:
        shape_cfg.mu = mu
    # Solid-cube inertia for a 0.2 m box of mass BOX_MASS.
    inertia = wp.mat33(np.eye(3) * BOX_MASS * 0.04 / 6.0)
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()), mass=BOX_MASS, inertia=inertia
    )
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=shape_cfg)
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=shape_cfg.mu))
    return builder.finalize(device=device)


def test_resting_box_reports_weight_as_linear_force(test, device, solve_mode="matrix_free", response="immediate"):
    """Report the resting box's weight through the linear contact force and leave the torque zero."""
    model = _box_on_ground(device)
    solver = SolverFeatherPGS(model, solve_mode=solve_mode, iterations=32, articulated_contact_response=response)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    observables = solver.observables({SolverObservableFlags.CONTACT_F})
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    for _ in range(120):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT, observables=observables)
        state_0, state_1 = state_1, state_0

    count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(count, 0)
    shape0 = contacts.rigid_contact_shape0.numpy()[:count]
    # Force on shape0's body by shape1's; orient every record as the force on the box.
    box_shape = int(np.flatnonzero(model.shape_body.numpy() >= 0)[0])
    sign = np.where(shape0 == box_shape, 1.0, -1.0)[:, None]
    wrench = observables.contact_f.numpy()[:count]
    linear = wrench[:, :3] * sign
    weight = BOX_MASS * float(np.linalg.norm(model.gravity.numpy()[0]))
    np.testing.assert_allclose(linear.sum(axis=0), [0.0, 0.0, weight], rtol=2.0e-2, atol=2.0e-2 * weight)

    np.testing.assert_array_equal(wrench[:, 3:], np.zeros((count, 3)))


def _tilted_gravity_box_velocity(device, slope_angle, mu, steps, solve_mode="matrix_free", response="immediate"):
    """Box on the ground under gravity tilted by ``slope_angle``, as on an incline; return its x velocity."""
    g = 9.81
    gravity = (g * np.sin(slope_angle), 0.0, -g * np.cos(slope_angle))
    model = _box_on_ground(device, gravity=gravity, mu=mu)
    solver = SolverFeatherPGS(model, solve_mode=solve_mode, iterations=32, articulated_contact_response=response)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    for _ in range(steps):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0
    return float(state_0.body_qd.numpy()[0, 0])


def test_friction_holds_inside_the_cone(test, device, solve_mode="matrix_free", response="immediate"):
    """Hold the box at rest when the tangential load is inside the friction cone."""
    # tan(0.3) = 0.31 < mu = 0.5.
    velocity = _tilted_gravity_box_velocity(
        device, slope_angle=0.3, mu=0.5, steps=120, solve_mode=solve_mode, response=response
    )
    test.assertLess(abs(velocity), 1.0e-3)


def test_friction_slides_at_the_coulomb_bound(test, device, solve_mode="matrix_free", response="immediate"):
    """Accelerate the box at ``g (sin a - mu cos a)`` when the load exceeds the friction cone."""
    slope_angle, mu, steps = 0.6, 0.3, 120
    velocity = _tilted_gravity_box_velocity(
        device, slope_angle=slope_angle, mu=mu, steps=steps, solve_mode=solve_mode, response=response
    )
    expected = 9.81 * (np.sin(slope_angle) - mu * np.cos(slope_angle)) * steps * DT
    test.assertAlmostEqual(velocity, expected, delta=0.03 * expected)


ARMATURE = (9.0, 4.0, 2.0, 0.5, 1.5, 3.0)


def _impact(model, solver, joint_qd):
    """Step once from the given free-joint velocity; return the new velocity and summed contact force."""
    state_in, state_out = model.state(), model.state()
    qd = state_in.joint_qd.numpy()
    qd[:6] = joint_qd
    state_in.joint_qd.assign(qd)
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(state_in, contacts)
    observables = solver.observables({SolverObservableFlags.CONTACT_F})
    solver.step(state_in, state_out, model.control(), contacts, 0.01, observables=observables)
    count = int(contacts.rigid_contact_count.numpy()[0])
    return state_out.joint_qd.numpy()[:6], observables.contact_f.numpy()[:count, :3].sum(axis=0)


def test_free_body_contact_response_includes_armature(test, device, solve_mode="matrix_free", response="immediate"):
    """Respond on free-body rows with the same armature-augmented inertia as on articulated rows."""
    stop = (0.0, 0.0, -1.0, 0.0, 0.0, 0.0)
    for armature, expected in ((0.0, 100.0), (9.0, 1000.0)):
        with test.subTest(armature=armature):
            model = _sphere_on_ground(device, armature, dense=False)
            solver = SolverFeatherPGS(model, solve_mode=solve_mode, cfm=0.0, articulated_contact_response=response)
            qd, force = _impact(model, solver, stop)
            # The sphere starts 0.1 mm deep, so the depenetration bias adds 0.2 %.
            test.assertAlmostEqual(abs(float(force[2])), expected, delta=5.0e-3 * expected)
            test.assertAlmostEqual(float(qd[2]), 0.0, delta=5.0e-3)

    # An oblique frictional impact of an offset-COM body with per-axis armature.
    oblique = (0.3, -0.2, -1.0, 0.4, 0.5, -0.3)
    results = {}
    for dense in (False, True):
        model = _sphere_on_ground(device, ARMATURE, dense=dense, com=(0.03, -0.02, 0.01), mu=0.5)
        solver = SolverFeatherPGS(
            model, solve_mode=solve_mode, cfm=0.0, iterations=64, articulated_contact_response=response
        )
        results[dense] = _impact(model, solver, oblique)
    np.testing.assert_allclose(results[False][0], results[True][0], rtol=0.0, atol=1.0e-4)
    np.testing.assert_allclose(results[False][1], results[True][1], rtol=1.0e-3, atol=1.0e-2)

    # A notified armature change reaches the free-body response.
    model = _sphere_on_ground(device, 0.0, dense=False)
    solver = SolverFeatherPGS(model, solve_mode=solve_mode, cfm=0.0, articulated_contact_response=response)
    _impact(model, solver, stop)
    joint_armature = model.joint_armature.numpy()
    joint_armature[:6] = 9.0
    model.joint_armature.assign(joint_armature)
    solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
    _, force = _impact(model, solver, stop)
    test.assertAlmostEqual(abs(float(force[2])), 1000.0, delta=5.0)


def test_free_body_impact_shares_armature_momentum(test, device, solve_mode="matrix_free", response="immediate"):
    """Conserve the armature-augmented momentum in a frictionless plastic impact between free bodies."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.0)
    for x in (0.0, 0.1999):
        body = builder.add_body(
            xform=wp.transform(wp.vec3(x, 0.0, 1.0), wp.quat_identity()), mass=1.0, inertia=wp.mat33(np.eye(3) * 0.01)
        )
        builder.add_shape_sphere(body, radius=0.1, cfg=cfg)
    model = builder.finalize(device=device)
    joint_armature = model.joint_armature.numpy()
    joint_armature[:6] = 9.0
    model.joint_armature.assign(joint_armature)
    solver = SolverFeatherPGS(
        model, solve_mode=solve_mode, cfm=0.0, iterations=64, articulated_contact_response=response
    )
    state_in, state_out = model.state(), model.state()
    joint_qd = state_in.joint_qd.numpy()
    joint_qd[0] = 1.0
    state_in.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(state_in, contacts)
    test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
    solver.step(state_in, state_out, model.control(), contacts, 0.01)
    qd = state_out.joint_qd.numpy()
    # Effective masses 10 and 1: both move at 10 / 11 m/s (plus a small depenetration bias).
    test.assertAlmostEqual(float(qd[0]), 10.0 / 11.0, delta=2.0e-3)
    test.assertAlmostEqual(float(qd[6]), 10.0 / 11.0, delta=1.5e-2)
    test.assertAlmostEqual(float(10.0 * qd[0] + qd[6]), 10.0, delta=1.0e-3)


ANCHOR_ROTATIONS = (
    wp.quat_from_axis_angle(wp.normalize(wp.vec3(0.3, 1.0, 0.2)), 0.25 * np.pi),
    wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.5 * np.pi),
)


def _anchored_sphere(device, anchor_rotation, gravity=None, inertia=(0.004, 0.004, 0.004), max_speed=None):
    """An offset-COM sphere on the ground whose FREE joint has the given parent-anchor rotation."""
    builder = newton.ModelBuilder() if gravity is None else newton.ModelBuilder(gravity=gravity)
    SolverFeatherPGS.register_custom_attributes(builder)
    pose = wp.transform(wp.vec3(0.0, 0.0, 0.0999), wp.quat_identity())
    limits = {}
    if max_speed is not None:
        limits = {
            "feather_pgs:rigid_body_max_linear_velocity": max_speed[0],
            "feather_pgs:rigid_body_max_angular_velocity": max_speed[1],
        }
    body = builder.add_link(
        xform=pose,
        mass=1.0,
        com=wp.vec3(0.03, -0.02, 0.01),
        inertia=wp.mat33(np.diag(inertia)),
        custom_attributes=limits,
    )
    builder.add_articulation([builder.add_joint_free(body, parent_xform=wp.transform_identity())])
    builder.add_shape_sphere(body, radius=0.1, cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.5))
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
    model = builder.finalize(device=device)
    _set_free_anchor(model, anchor_rotation, (0.0,) * 6)
    return model


def _set_free_anchor(model, anchor_rotation, world_twist):
    """Move the FREE joint's parent anchor, keeping the body's world pose and giving it ``world_twist``."""
    pose = wp.transform(*model.body_q.numpy()[0])
    anchor = wp.transform(wp.vec3(0.2, -0.1, 0.3), anchor_rotation)
    joint_X_p = model.joint_X_p.numpy()
    joint_X_p[0] = [*anchor.p, *anchor.q]
    model.joint_X_p.assign(joint_X_p)
    local = wp.transform_inverse(anchor) * pose
    joint_q = model.joint_q.numpy()
    joint_q[:7] = [*local.p, *local.q]
    model.joint_q.assign(joint_q)
    # FREE joint velocities are expressed in the parent anchor frame.
    to_anchor = np.array(wp.quat_to_matrix(anchor_rotation)).reshape(3, 3).T
    joint_qd = model.joint_qd.numpy()
    joint_qd[:3] = to_anchor @ np.asarray(world_twist[:3])
    joint_qd[3:6] = to_anchor @ np.asarray(world_twist[3:])
    model.joint_qd.assign(joint_qd)


def _run_anchored_sphere(model, solver, steps, dt):
    """Step from the model's joint state and return the body's world twist."""
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    for _ in range(steps):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, model.control(), contacts, dt)
        state_0, state_1 = state_1, state_0
    return state_0.body_qd.numpy()[0]


def _anchored_sphere_cases():
    """An oblique frictional impact, and the body rolling off its offset COM under gravity."""
    return {
        "impact": ((0.3, -0.2, -1.0, 0.4, 0.5, -0.3), wp.vec3(0.0), 1, 0.01, None),
        "gravity": ((0.0,) * 6, None, 60, DT, None),
        # Box velocity limits act on world axes.
        "velocity_limits": ((0.3, -0.2, -1.0, 0.4, 0.5, -0.3), wp.vec3(0.0), 3, 0.01, (0.35, 0.3)),
    }


def test_rotated_anchor_free_body_matches_world_aligned(test, device, solve_mode="matrix_free", response="immediate"):
    """Give a FREE body the same world motion whatever the rotation of its parent anchor."""
    for inertia in ((0.004, 0.004, 0.004), (0.003, 0.004, 0.006)):
        for case, (world_twist, gravity, steps, dt, max_speed) in _anchored_sphere_cases().items():
            twists = []
            for rotation in (wp.quat_identity(), *ANCHOR_ROTATIONS):
                model = _anchored_sphere(device, rotation, gravity, inertia, max_speed)
                _set_free_anchor(model, rotation, world_twist)
                solver = SolverFeatherPGS(
                    model,
                    solve_mode=solve_mode,
                    cfm=0.0,
                    iterations=64,
                    articulated_contact_response=response,
                    velocity_limit_activation_fraction=0.5,
                )
                twists.append(_run_anchored_sphere(model, solver, steps, dt))
            for rotation, twist in zip(ANCHOR_ROTATIONS, twists[1:], strict=True):
                with test.subTest(inertia=inertia, case=case, rotation=tuple(rotation)):
                    np.testing.assert_allclose(twist, twists[0], rtol=0.0, atol=2.0e-4)


def test_free_anchor_rotated_after_construction(test, device, solve_mode="matrix_free", response="immediate"):
    """Follow a FREE body's parent anchor rotated after construction, eagerly and in a replayed graph."""
    world_twist, gravity, steps, dt, _ = _anchored_sphere_cases()["gravity"]
    inertia = (0.003, 0.004, 0.006)
    rotation = ANCHOR_ROTATIONS[0]
    # The world-aligned anchor gives the expected world motion.
    model = _anchored_sphere(device, wp.quat_identity(), gravity, inertia)
    solver = SolverFeatherPGS(model, solve_mode=solve_mode, iterations=64, articulated_contact_response=response)
    expected = _run_anchored_sphere(model, solver, steps, dt)

    model = _anchored_sphere(device, wp.quat_identity(), gravity, inertia)
    solver = SolverFeatherPGS(model, solve_mode=solve_mode, iterations=64, articulated_contact_response=response)
    _run_anchored_sphere(model, solver, 1, dt)
    _set_free_anchor(model, rotation, world_twist)
    solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
    np.testing.assert_allclose(_run_anchored_sphere(model, solver, steps, dt), expected, rtol=0.0, atol=2.0e-4)

    if not wp.get_device(device).is_cuda:
        return
    model = _anchored_sphere(device, wp.quat_identity(), gravity, inertia)
    solver = SolverFeatherPGS(model, solve_mode=solve_mode, iterations=64, articulated_contact_response=response)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    control = model.control()

    def step_pair():
        for _ in range(2):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, dt)
            state_0.assign(state_1)

    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    step_pair()
    with wp.ScopedCapture(device) as capture:
        step_pair()
    _set_free_anchor(model, rotation, world_twist)
    state_0.joint_q.assign(model.joint_q)
    state_0.joint_qd.assign(model.joint_qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    with wp.ScopedCapture(device) as notify:
        solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
    wp.capture_launch(notify.graph)
    for _ in range(steps // 2):
        wp.capture_launch(capture.graph)
    np.testing.assert_allclose(state_0.body_qd.numpy()[0], expected, rtol=0.0, atol=2.0e-4)


class TestFeatherPGSContactForce(unittest.TestCase):
    pass


split_devices = get_test_devices()


for _name in (
    "test_resting_box_reports_weight_as_linear_force",
    "test_friction_holds_inside_the_cone",
    "test_friction_slides_at_the_coulomb_bound",
    "test_free_body_contact_response_includes_armature",
    "test_free_body_impact_shares_armature_momentum",
    "test_rotated_anchor_free_body_matches_world_aligned",
    "test_free_anchor_rotated_after_construction",
):
    add_function_test(TestFeatherPGSContactForce, _name, globals()[_name], devices=devices)
    add_function_test(
        TestFeatherPGSContactForce, f"{_name}_split", globals()[_name], devices=split_devices, solve_mode="split"
    )
    # The propagation response solves these contacts as body-space rows.
    add_function_test(
        TestFeatherPGSContactForce, f"{_name}_propagation", globals()[_name], devices=devices, response="propagation"
    )


# Colored propagation builds the same body-space rows as propagation in a different order.
add_function_test(
    TestFeatherPGSContactForce,
    "test_rotated_anchor_free_body_matches_world_aligned_propagation_colored",
    test_rotated_anchor_free_body_matches_world_aligned,
    devices=devices,
    response="propagation_colored",
)


# The fused response only differs from the immediate one with an articulated (dense-path) body.
add_function_test(
    TestFeatherPGSContactForce,
    "test_free_body_contact_response_includes_armature_propagation_fused",
    test_free_body_contact_response_includes_armature,
    devices=devices,
    response="propagation_fused",
)


LOWER_MASS = 2.0


UPPER_MASS = 0.5


FILL = -7.0


def _stacked_boxes(device):
    """Two free boxes stacked on the ground: ground-box and box-box contacts."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    for z, half, mass in ((0.1, 0.1, LOWER_MASS), (0.25, 0.05, UPPER_MASS)):
        inertia = wp.mat33(np.eye(3) * mass * (2.0 * half) ** 2 / 6.0)
        body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, z), wp.quat_identity()), mass=mass, inertia=inertia
        )
        builder.add_shape_box(body, hx=half, hy=half, hz=half, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=64)
    return model, pipeline


def _net_body_forces(model, contacts, contact_f):
    """Sum each body's contact forces from the per-contact rows (force on shape 0's body)."""
    count = int(contacts.rigid_contact_count.numpy()[0])
    shape_body = model.shape_body.numpy()
    body0 = shape_body[contacts.rigid_contact_shape0.numpy()[:count]]
    body1 = shape_body[contacts.rigid_contact_shape1.numpy()[:count]]
    linear = contact_f[:count, :3]
    net = np.zeros((model.body_count, 3))
    for body in range(model.body_count):
        net[body] = linear[body0 == body].sum(axis=0) - linear[body1 == body].sum(axis=0)
    return net


def _check_rows(test, contacts, contact_f):
    """Live rows hold a finite linear force and no torque; every other row is zero."""
    count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(count, 0)
    test.assertTrue(np.isfinite(contact_f[:count, :3]).all())
    np.testing.assert_array_equal(contact_f[:count, 3:], 0.0)
    np.testing.assert_array_equal(contact_f[count:], 0.0)


def test_contact_f_matches_legacy_force_and_weight(test, device, steps=150, torsion_radius=None, **options):
    """Report linear-only per-contact forces that sum to each box's weight at rest."""
    model, pipeline = _stacked_boxes(device)
    if torsion_radius is not None:
        set_torsion_radius(model, torsion_radius)
    options.setdefault("solve_mode", "matrix_free")
    solver = SolverFeatherPGS(model, iterations=32, **options)
    observables = solver.observables({SolverObservableFlags.CONTACT_F})
    test.assertEqual(observables.contact_f.shape[0], model.rigid_contact_max + model.soft_contact_max)
    observables.contact_f.fill_(wp.spatial_vector(FILL, FILL, FILL, FILL, FILL, FILL))
    contacts = pipeline.contacts()
    states = [model.state(), model.state()]
    control = model.control()
    for _ in range(steps):
        pipeline.collide(states[0], contacts)
        solver.step(states[0], states[1], control, contacts, DT, observables=observables)
        states.reverse()
    contact_f = observables.contact_f.numpy()
    # The SolverBase signature, which also passes the state.
    _check_rows(test, contacts, contact_f)
    if options.get("free_body_constraint_max") is not None:
        # Rows beyond the capacity were dropped; those contacts report zero.
        count = int(contacts.rigid_contact_count.numpy()[0])
        dropped = solver._contact_slot.numpy()[:count] < 0
        test.assertTrue(dropped.any())
        np.testing.assert_array_equal(contact_f[:count][dropped], 0.0)
    else:
        weight = np.linalg.norm(model.gravity.numpy()[0])
        net = _net_body_forces(model, contacts, contact_f)
        np.testing.assert_allclose(net[0], [0.0, 0.0, LOWER_MASS * weight], atol=0.03 * LOWER_MASS * weight)
        np.testing.assert_allclose(net[1], [0.0, 0.0, UPPER_MASS * weight], atol=0.03 * UPPER_MASS * weight)


def test_unrequested_contact_f_is_left_untouched(test, device):
    """Write nothing when the observable is allocated but not selected for the step."""
    model, pipeline = _stacked_boxes(device)
    solver = SolverFeatherPGS(model, solve_mode="matrix_free" if wp.get_device(device).is_cuda else "split")
    observables = solver.observables({SolverObservableFlags.CONTACT_F})
    observables.contact_f.fill_(wp.spatial_vector(FILL, FILL, FILL, FILL, FILL, FILL))
    contacts = pipeline.contacts()
    state_in, state_out = model.state(), model.state()
    pipeline.collide(state_in, contacts)
    solver.step(state_in, state_out, model.control(), contacts, DT, observables=observables.select(set()))
    np.testing.assert_array_equal(observables.contact_f.numpy(), FILL)
    solver.step(state_in, state_out, model.control(), contacts, DT, observables=observables)
    test.assertFalse(np.any(observables.contact_f.numpy() == FILL))


def test_contact_f_in_a_captured_graph(test, device):
    """Write the observable from the replayed step's impulses in a captured graph."""
    model, pipeline = _stacked_boxes(device)
    solver = SolverFeatherPGS(model, solve_mode="matrix_free", iterations=32)
    observables = solver.observables({SolverObservableFlags.CONTACT_F})
    contacts = pipeline.contacts()
    states = [model.state(), model.state()]
    control = model.control()

    def step():
        pipeline.collide(states[0], contacts)
        solver.step(states[0], states[1], control, contacts, DT, observables=observables)
        pipeline.collide(states[1], contacts)
        solver.step(states[1], states[0], control, contacts, DT, observables=observables)

    step()
    with wp.ScopedCapture(device) as recorded:
        step()
    observables.contact_f.fill_(wp.spatial_vector(FILL, FILL, FILL, FILL, FILL, FILL))
    for _ in range(60):
        wp.capture_launch(recorded.graph)
    _check_rows(test, contacts, observables.contact_f.numpy())
    weight = np.linalg.norm(model.gravity.numpy()[0])
    net = _net_body_forces(model, contacts, observables.contact_f.numpy())
    np.testing.assert_allclose(net[1, 2], UPPER_MASS * weight, rtol=0.03)


def test_contact_f_of_sleeping_islands(test, device, skip_constraints=True):
    """Report the legacy force of a sleeping island's contacts: zero with skipped rows, else the resting force."""
    model, pipeline = _stacked_boxes(device)
    solver = SolverFeatherPGS(
        model,
        solve_mode="matrix_free",
        iterations=32,
        enable_sleeping=True,
        sleep_quiet_time=0.05,
        sleep_skip_constraints=skip_constraints,
        friction_anchor_beta=0.0,
    )
    observables = solver.observables({SolverObservableFlags.CONTACT_F})
    contacts = pipeline.contacts()
    states = [model.state(), model.state()]
    control = model.control()
    for step in range(150):
        if step == 149:
            observables.contact_f.fill_(wp.spatial_vector(FILL, FILL, FILL, FILL, FILL, FILL))
        states[0].clear_forces()
        pipeline.collide(states[0], contacts)
        solver.step(states[0], states[1], control, contacts, DT, observables=observables)
        states.reverse()
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0])
    contact_f = observables.contact_f.numpy()
    _check_rows(test, contacts, contact_f)
    net = _net_body_forces(model, contacts, contact_f)
    if skip_constraints:
        np.testing.assert_array_equal(net, 0.0)
    else:
        weight = np.linalg.norm(model.gravity.numpy()[0])
        np.testing.assert_allclose(net[0, 2], LOWER_MASS * weight, rtol=0.03)
        np.testing.assert_allclose(net[1, 2], UPPER_MASS * weight, rtol=0.03)


def test_contact_f_of_compliant_hydroelastic_rows(test, device):
    """Report the legacy force of hydroelastic contacts solved with contact compliance."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.default_shape_cfg = newton.ModelBuilder.ShapeConfig(
            mu=0.5,
            is_hydroelastic=True,
            sdf_max_resolution=32,
            sdf_narrow_band_range=(-0.01, 0.01),
            sdf_padding=0.006,
            gap=0.005,
            kh=1e7,
        )
        builder.add_shape_box(
            body=-1, hx=0.1, hy=0.1, hz=0.025, xform=wp.transform(wp.vec3(0, 0, -0.025), wp.quat_identity())
        )
        body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.048), wp.quat_identity()))
        cfg = builder.default_shape_cfg.copy()
        cfg.density = 0.3 / (4 / 3 * np.pi * 0.05**3)
        builder.add_shape_sphere(body, radius=0.05, cfg=cfg)
        model = builder.finalize()
        pipeline = newton.CollisionPipeline(
            model,
            rigid_contact_max=512,
            deterministic=True,
            sdf_hydroelastic_config=HydroelasticSDF.Config(
                reduce_contacts=True, moment_matching=True, anchor_contact=True, buffer_fraction=1.0
            ),
        )
        solver = SolverFeatherPGS(
            model,
            solve_mode="matrix_free",
            contact_compliance=True,
            enable_restitution=False,
            friction_anchor_beta=0.0,
            iterations=64,
            free_body_constraint_max=2048,
        )
        observables = solver.observables({SolverObservableFlags.CONTACT_F})
        contacts = pipeline.contacts()
        states = [model.state(), model.state()]
        newton.eval_fk(model, model.joint_q, model.joint_qd, states[0])
        for _ in range(5):
            pipeline.collide(states[0], contacts)
            solver.step(states[0], states[1], model.control(), contacts, 0.0025, observables=observables)
            states.reverse()
        test.assertGreater(solver.compliance_contact_count, 0)
        contact_f = observables.contact_f.numpy()
        _check_rows(test, contacts, contact_f)
        test.assertGreater(np.abs(contact_f[:, 2]).sum(), 0.0)


def test_overflowing_contact_count_stays_in_the_rigid_rows(test, device, capture=False):
    """Bound rigid rows by the capacity when the contact count overflows it, leaving soft rows zero."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.09), wp.quat_identity()))
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    model = builder.finalize(device=device)
    # Built before the pipeline, the solver's contact scratch is larger than the contact buffer.
    solve_mode = "matrix_free" if wp.get_device(device).is_cuda else "split"
    solver = SolverFeatherPGS(model, solve_mode=solve_mode, warn_constraint_overflow=False)
    rigid_capacity, soft_capacity = 4, 2
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=rigid_capacity, soft_contact_max=soft_capacity)
    observables = solver.observables({SolverObservableFlags.CONTACT_F})
    test.assertEqual(observables.contact_f.shape[0], rigid_capacity + soft_capacity)
    contacts = pipeline.contacts()
    state_in, state_out, control = model.state(), model.state(), model.control()
    overflow_count = wp.array([rigid_capacity + soft_capacity + 3], dtype=wp.int32, device=device)

    def step():
        pipeline.collide(state_in, contacts)
        # The collision counter keeps counting past the capacity when the buffer overflows.
        wp.copy(contacts.rigid_contact_count, overflow_count)
        solver.step(state_in, state_out, control, contacts, DT, observables=observables)

    pipeline.collide(state_in, contacts)
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), rigid_capacity)
    step()
    observables.contact_f.fill_(wp.spatial_vector(FILL, FILL, FILL, FILL, FILL, FILL))
    if capture:
        with wp.ScopedCapture(device) as recorded:
            step()
        observables.contact_f.fill_(wp.spatial_vector(FILL, FILL, FILL, FILL, FILL, FILL))
        wp.capture_launch(recorded.graph)
    else:
        step()
    contact_f = observables.contact_f.numpy()
    test.assertTrue(np.isfinite(contact_f[:rigid_capacity, :3]).all())
    test.assertFalse(np.any(contact_f[:rigid_capacity] == FILL))
    np.testing.assert_array_equal(contact_f[:rigid_capacity, 3:], 0.0)
    np.testing.assert_array_equal(contact_f[rigid_capacity:], 0.0)


def test_contact_force_kernel_reads_only_rigid_rows(test, device):
    """Read no rigid contact data past the capacity when the count overflows into the soft rows."""
    rigid_capacity, soft_capacity = 1, 2
    rows = rigid_capacity + soft_capacity
    backing = []

    def rigid_view(values, dtype):
        # Valid storage past the view makes a read beyond the rigid capacity deterministic.
        full = wp.array(values, dtype=dtype, device=device)
        backing.append(full)
        return wp.array(ptr=full.ptr, shape=(rigid_capacity,), dtype=dtype, device=device)

    impulses = wp.array([[3.0]], dtype=wp.float32, device=device)
    row_count = wp.array([1], dtype=wp.int32, device=device)
    row_type = wp.zeros((1, 1), dtype=wp.int32, device=device)
    row_parent = wp.zeros((1, 1), dtype=wp.int32, device=device)
    contact_f = wp.full(rows, wp.spatial_vector(FILL, FILL, FILL, FILL, FILL, FILL), device=device)
    wp.launch(
        newton._src.solvers.feather_pgs.kernels.compute_contact_spatial_force_from_impulses,
        dim=rows,
        inputs=[
            wp.array([rows + 4], dtype=wp.int32, device=device),
            rigid_view([[0.0, 0.0, -1.0]] * rows, wp.vec3),
            rigid_view([0] * rows, wp.int32),
            rigid_view([0] * rows, wp.int32),
            rigid_view([0] * rows, wp.int32),
            impulses,
            impulses,
            impulses,
            row_count,
            row_count,
            row_count,
            row_type,
            row_parent,
            row_type,
            row_parent,
            row_type,
            row_parent,
            10.0,
            rigid_capacity,
        ],
        outputs=[contact_f],
        device=device,
    )
    np.testing.assert_allclose(contact_f.numpy()[0], [0.0, 0.0, 30.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(contact_f.numpy()[rigid_capacity:], 0.0)


class TestFeatherPGSContactObservable(unittest.TestCase):
    pass


cuda_devices = get_cuda_test_devices()


all_devices = get_test_devices()


add_function_test(
    TestFeatherPGSContactObservable,
    "test_contact_f_matches_legacy_force_and_weight",
    test_contact_f_matches_legacy_force_and_weight,
    devices=cuda_devices,
)


add_function_test(
    TestFeatherPGSContactObservable,
    "test_contact_f_matches_legacy_force_and_weight_split",
    test_contact_f_matches_legacy_force_and_weight,
    devices=all_devices,
    solve_mode="split",
)


for _suffix, _options in (
    ("propagation", {"articulated_contact_response": "propagation"}),
    ("propagation_fused", {"articulated_contact_response": "propagation_fused"}),
    ("propagation_colored", {"articulated_contact_response": "propagation_colored"}),
    ("point_friction", {"friction_anchor_beta": 0.0}),
    ("torsion", {"torsion_radius": 0.01}),
    ("device_torsion", {"torsion_radius": 0.01, "contact_torsion_device": True}),
    ("double_buffer", {"double_buffer": True}),
    ("row_overflow", {"free_body_constraint_max": 6, "warn_constraint_overflow": False}),
):
    add_function_test(
        TestFeatherPGSContactObservable,
        f"test_contact_f_matches_legacy_force_and_weight_{_suffix}",
        test_contact_f_matches_legacy_force_and_weight,
        devices=cuda_devices,
        **_options,
    )


add_function_test(
    TestFeatherPGSContactObservable,
    "test_unrequested_contact_f_is_left_untouched",
    test_unrequested_contact_f_is_left_untouched,
    devices=all_devices,
)


add_function_test(
    TestFeatherPGSContactObservable,
    "test_overflowing_contact_count_stays_in_the_rigid_rows",
    test_overflowing_contact_count_stays_in_the_rigid_rows,
    devices=all_devices,
)


add_function_test(
    TestFeatherPGSContactObservable,
    "test_overflowing_contact_count_stays_in_the_rigid_rows_captured",
    test_overflowing_contact_count_stays_in_the_rigid_rows,
    devices=cuda_devices,
    capture=True,
)


add_function_test(
    TestFeatherPGSContactObservable,
    "test_contact_force_kernel_reads_only_rigid_rows",
    test_contact_force_kernel_reads_only_rigid_rows,
    devices=all_devices,
)


add_function_test(
    TestFeatherPGSContactObservable,
    "test_contact_f_in_a_captured_graph",
    test_contact_f_in_a_captured_graph,
    devices=cuda_devices,
)


add_function_test(
    TestFeatherPGSContactObservable,
    "test_contact_f_of_sleeping_islands",
    test_contact_f_of_sleeping_islands,
    devices=cuda_devices,
)


add_function_test(
    TestFeatherPGSContactObservable,
    "test_contact_f_of_sleeping_islands_without_skipped_rows",
    test_contact_f_of_sleeping_islands,
    devices=cuda_devices,
    skip_constraints=False,
)


add_function_test(
    TestFeatherPGSContactObservable,
    "test_contact_f_of_compliant_hydroelastic_rows",
    test_contact_f_of_compliant_hydroelastic_rows,
    devices=cuda_devices,
)


class TestFeatherPGSPersistentContacts(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "requires CUDA graph capture")
    def test_combined_history_reset_replays_with_changed_world_mask(self):
        """Capture reduction, matching and solver resets with a live device mask."""
        template = newton.ModelBuilder()
        _cylinder_foot(template, wp.vec3(0.0, 0.0, 0.015))
        builder = newton.ModelBuilder()
        builder.replicate(template, 2, spacing=(1.0, 0.0, 0.0))
        builder.add_ground_plane()
        model = builder.finalize(device=wp.get_cuda_device())
        pipeline = newton.CollisionPipeline(
            model,
            contact_matching="latest",
            reduce_contacts=newton.CollisionPipeline.ContactReductionConfig(
                body_pairs=True,
                body_pair_hysteresis=0.001,
            ),
        )
        solver = SolverFeatherPGS(model, solve_mode="matrix_free", contact_warmstart=True)
        state, output = model.state(), model.state()
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        solver.step(state, output, model.control(), contacts, 1.0 / 240.0)
        mask = wp.array([False, False, False], dtype=wp.bool, device=model.device)
        solver_mask = mask[:2]
        # Compile both reset paths before capture.
        solver.reset(state, solver_mask)
        pipeline.reset_contact_matching(mask)
        with wp.ScopedCapture(device=model.device) as capture:
            solver.reset(state, solver_mask)
            pipeline.reset_contact_matching(mask)
            pipeline.collide(state, contacts)
        for selected in (0, 1):
            values = np.zeros(3, dtype=bool)
            values[selected] = True
            mask.assign(values)
            solver._ws_prev_mf_impulses.fill_(1.0)
            wp.capture_launch(capture.graph)
            previous = solver._ws_prev_mf_impulses.numpy()
            self.assertTrue(np.all(previous[selected] == 0.0))
            self.assertTrue(np.all(previous[1 - selected] == 1.0))
            count = int(contacts.rigid_contact_count.numpy()[0])
            shape0 = contacts.rigid_contact_shape0.numpy()[:count]
            shape1 = contacts.rigid_contact_shape1.numpy()[:count]
            worlds = np.maximum(model.shape_world.numpy()[shape0], model.shape_world.numpy()[shape1])
            matches = contacts.rigid_contact_match_index.numpy()[:count]
            self.assertTrue(np.any(worlds == selected) and np.any(worlds == 1 - selected))
            self.assertTrue(np.all(matches[worlds == selected] < 0))
            self.assertTrue(np.all(matches[worlds == 1 - selected] >= 0))

    def _seed_rotating_contact(self, friction, *, oblique=False, rotated=False, device="cpu"):
        """Prepare a cached friction impulse and move its normal across a basis seam."""
        builder = newton.ModelBuilder(up_axis=newton.Axis.X, gravity=wp.vec3(0.0))
        builder.add_ground_plane()
        body = builder.add_body(xform=wp.transform(wp.vec3(0.1, 0.0, 0.0), wp.quat_identity()))
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device=device)
        model.shape_material_mu.fill_(1.0)
        solver = SolverFeatherPGS(model, contact_warmstart=True, iterations=0)
        pipeline = newton.CollisionPipeline(model, contact_matching="latest")
        contacts = pipeline.contacts()
        state_in, state_out = model.state(), model.state()
        newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
        pipeline.collide(state_in, contacts)
        self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
        contacts.rigid_contact_count.fill_(1)
        old_normal = wp.normalize(wp.vec3(-1.0, 0.001, 0.0))
        new_normal = wp.normalize(wp.vec3(-1.0, -0.001, 0.0))
        if oblique:
            old_normal = wp.normalize(wp.vec3(0.2, 0.3, 0.93))
            new_normal = wp.normalize(wp.vec3(0.3, 0.2, 0.93))
        if rotated:
            old_normal = wp.vec3(0.0, 0.0, 1.0)
            new_normal = wp.vec3(0.5 / np.sqrt(2.0), 0.5 / np.sqrt(2.0), np.sqrt(3.0) / 2.0)
        normals = contacts.rigid_contact_normal.numpy()
        normals[0] = old_normal
        contacts.rigid_contact_normal.assign(normals)
        solver.step(state_in, state_out, model.control(), contacts, 1.0 / 240.0)
        slot = int(solver._contact_slot.numpy()[0])
        self.assertGreaterEqual(slot, 0)
        jacobian = solver._mf_J_a if int(solver._mf_body_a.numpy()[0, slot]) >= 0 else solver._mf_J_b
        old_direction = jacobian.numpy()[0, slot + 1, :3].copy()
        previous = solver._ws_prev_mf_impulses.numpy()
        previous[0, slot : slot + 3] = (1.0, 0.5, 0.0)
        solver._ws_prev_mf_impulses.assign(previous)
        # Seed both owners of the previous solve's tangent impulse. Persistent
        # anchors take precedence over the contact-matched cache when carried.
        patches = solver._friction_patches
        tangent, _ = contact_tangent_basis(-old_normal)
        tangent *= 0.5
        if patches.current.flipped.numpy()[0]:
            tangent = -tangent
        anchor_body = int(patches.current.body_a.numpy()[0])
        if anchor_body >= 0:
            pose = state_in.body_q.numpy()[anchor_body]
            transform = wp.transform(wp.vec3(*pose[:3]), wp.quat(*pose[3:]))
            tangent = wp.transform_vector(wp.transform_inverse(transform), tangent)
        cached = patches.previous.tangent_impulse.numpy()
        cached[0] = tangent
        patches.previous.tangent_impulse.assign(cached)
        normals[0] = new_normal
        contacts.rigid_contact_normal.assign(normals)
        contacts.rigid_contact_match_index.fill_(-1)
        indices = contacts.rigid_contact_match_index.numpy()
        indices[0] = 0
        contacts.rigid_contact_match_index.assign(indices)
        model.shape_material_mu.fill_(friction)
        # Refresh the solver's friction copy without the history reset a SHAPE_PROPERTIES notify performs.
        solver._refresh_shape_materials()
        solver.step(state_in, state_out, model.control(), contacts, 1.0 / 240.0)
        slot = int(solver._contact_slot.numpy()[0])
        impulses = solver._mf_impulses.numpy()[0, slot : slot + 3]
        tangent_rows = jacobian.numpy()[0, slot + 1 : slot + 3, :3]
        world_tangent = impulses[1:] @ tangent_rows
        # Independent world-space projection; solver rows use B-to-A normals.
        n = np.asarray(new_normal)
        projected_direction = old_direction - n * np.dot(n, old_direction)
        return impulses, world_tangent, projected_direction

    def test_cached_friction_preserves_world_direction(self):
        """Transport a cached tangent impulse through a discontinuous tangent basis."""
        _, world_tangent, old_direction = self._seed_rotating_contact(1.0)
        np.testing.assert_allclose(world_tangent, 0.5 * old_direction, atol=1.0e-6)

    def test_cached_friction_uses_solver_normal_convention(self):
        """Project friction in the frame used by the actual contact Jacobians."""
        devices = ["cpu", wp.get_cuda_device()] if wp.is_cuda_available() else ["cpu"]
        for device in devices:
            with self.subTest(device=str(device)):
                _, world_tangent, projected_direction = self._seed_rotating_contact(1.0, oblique=True, device=device)
                np.testing.assert_allclose(world_tangent, 0.5 * projected_direction, atol=1.0e-6)

    def test_cached_friction_obeys_changed_material(self):
        """Clamp carried friction to the current material before applying velocity."""
        impulses, world_tangent, old_direction = self._seed_rotating_contact(0.1)
        self.assertLessEqual(float(np.linalg.norm(impulses[1:])), 0.1 * impulses[0] + 1.0e-6)
        np.testing.assert_allclose(world_tangent, 0.1 * old_direction, atol=1.0e-6)

    def test_rotating_contact_clamps_the_projected_friction_cone(self):
        """Project through a 30-degree normal change and saturate the current cone."""
        devices = ["cpu", wp.get_cuda_device()] if wp.is_cuda_available() else ["cpu"]
        for device in devices:
            with self.subTest(device=str(device)):
                impulses, tangent, projected = self._seed_rotating_contact(0.1, rotated=True, device=device)
                expected = 0.1 * projected / np.linalg.norm(projected)
                np.testing.assert_allclose(tangent, expected, atol=1.0e-6)
                self.assertAlmostEqual(float(np.linalg.norm(impulses[1:])), 0.1 * impulses[0], delta=1.0e-6)

    def test_matching_retains_world_distance_policy(self):
        """Moving beyond the existing world-distance threshold must cold-start."""
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        for z in (0.1, 0.29):
            body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, z), wp.quat_identity()))
            builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device="cpu")
        state = model.state()
        pipeline = newton.CollisionPipeline(model, contact_matching="latest")
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        count = int(contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(count, 0)
        poses = state.body_q.numpy()
        poses[:, 0] += 1.0
        state.body_q.assign(poses)
        pipeline.collide(state, contacts)
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), count)
        self.assertTrue(np.all(contacts.rigid_contact_match_index.numpy()[:count] < 0))

    def test_reduced_contacts_preserve_warmstart_identity(self):
        """Carry only retained contact identities through reduction and solver steps."""
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        _cylinder_foot(builder, wp.vec3(0.0, 0.0, 0.015))
        model = builder.finalize(device="cpu")
        solver = SolverFeatherPGS(model, contact_warmstart=True, iterations=8)
        pipeline = newton.CollisionPipeline(
            model,
            contact_matching="latest",
            reduce_contacts=newton.CollisionPipeline.ContactReductionConfig(body_pairs=True),
        )
        contacts = pipeline.contacts()
        state_in, state_out = model.state(), model.state()
        matched = 0
        for _ in range(40):
            pipeline.collide(state_in, contacts)
            count = int(contacts.rigid_contact_count.numpy()[0])
            indices = contacts.rigid_contact_match_index.numpy()[:count]
            valid = indices[indices >= 0]
            self.assertEqual(len(valid), len(np.unique(valid)), "a cached impulse was assigned more than once")
            matched += len(valid)
            self.assertEqual(int(pipeline._contact_matcher.prev_contact_count.numpy()[0]), count)
            state_in.clear_forces()
            solver.step(state_in, state_out, model.control(), contacts, 1.0 / 240.0)
            state_in, state_out = state_out, state_in
        stats = pipeline.body_pair_reduction_stats()
        self.assertGreater(stats["sum_contacts_in"], stats["sum_contacts_kept"])
        self.assertGreater(matched, 20)
        self.assertTrue(np.isfinite(state_in.body_q.numpy()).all())
        self.assertGreater(float(state_in.body_q.numpy()[0, 2]), 0.012)

    def test_matching_reset_also_invalidates_reduction_history(self):
        """Partial resets preserve other worlds; material changes start cold."""
        template = newton.ModelBuilder()
        _cylinder_foot(template, wp.vec3(0.0, 0.0, 0.015))
        builder = newton.ModelBuilder()
        builder.replicate(template, 2, spacing=(1.0, 0.0, 0.0))
        builder.add_ground_plane()
        model = builder.finalize(device="cpu")
        pipeline = newton.CollisionPipeline(
            model,
            contact_matching="latest",
            reduce_contacts=newton.CollisionPipeline.ContactReductionConfig(
                body_pairs=True,
                body_pair_hysteresis=0.001,
            ),
        )
        state, contacts = model.state(), pipeline.contacts()
        pipeline.collide(state, contacts)
        pipeline.collide(state, contacts)
        generations = pipeline._body_pair_reducer.history_generation.numpy().copy()
        pipeline.reset_contact_matching(wp.array([True, False, False], dtype=wp.bool, device=model.device))
        after = pipeline._body_pair_reducer.history_generation.numpy()
        self.assertGreater(after[0], generations[0])
        self.assertEqual(after[1], generations[1])
        pipeline.collide(state, contacts)
        count = int(contacts.rigid_contact_count.numpy()[0])
        shape0 = contacts.rigid_contact_shape0.numpy()[:count]
        shape1 = contacts.rigid_contact_shape1.numpy()[:count]
        worlds = np.maximum(model.shape_world.numpy()[shape0], model.shape_world.numpy()[shape1])
        matches = contacts.rigid_contact_match_index.numpy()[:count]
        self.assertTrue(np.any(worlds == 0) and np.any(worlds == 1))
        self.assertTrue(np.all(matches[worlds == 0] < 0))
        self.assertTrue(np.all(matches[worlds == 1] >= 0))
        model.shape_material_mu.fill_(0.1)
        pipeline.refresh_body_pair_reduction_groups()
        pipeline.collide(state, contacts)
        count = int(contacts.rigid_contact_count.numpy()[0])
        self.assertTrue(np.all(contacts.rigid_contact_match_index.numpy()[:count] < 0))


RADIUS = 0.05


CONTACT_GAP = 0.1


DEFAULT_DT = 1.0e-3


IMPACT_FRACTION = 0.4


# Values written into ``SolverFeatherPGS.contact_path``.
PATH_DENSE = 0


RESPONSE_MODES = ("immediate", "propagation", "propagation_fused", "propagation_colored")


EXPECTED_PATHS = {
    ("free", "immediate"): {PATH_MATRIX_FREE},
    ("free", "propagation"): {PATH_PROPAGATION},
    ("free", "propagation_fused"): {PATH_MATRIX_FREE},
    ("free", "propagation_colored"): {PATH_PROPAGATION},
    ("articulated", "immediate"): {PATH_DENSE},
    ("articulated", "propagation"): {PATH_PROPAGATION},
    ("articulated", "propagation_fused"): {PATH_PROPAGATION},
    ("articulated", "propagation_colored"): {PATH_PROPAGATION},
}


def _diagonal_inertia(value):
    """Return an isotropic inertia tensor."""
    return wp.mat33(value, 0.0, 0.0, 0.0, value, 0.0, 0.0, 0.0, value)


def _sphere_inertia(mass):
    """Return the inertia of a solid sphere about its centre."""
    return _diagonal_inertia(0.4 * mass * RADIUS * RADIUS)


def _shape_cfg(restitution):
    """Return an inertialess, frictionless test material."""
    return newton.ModelBuilder.ShapeConfig(
        density=0.0,
        mu=0.0,
        restitution=restitution,
        margin=0.0,
        gap=CONTACT_GAP,
    )


def _build_plane_model(
    device,
    *,
    separation,
    restitution,
    plane_restitution=None,
    mass=1.0,
    gravity=(0.0, 0.0, 0.0),
    scene="free",
):
    """Build one sphere above a plane, optionally as a two-link articulation."""
    builder = newton.ModelBuilder(gravity=gravity)
    xform = wp.transform(wp.vec3(0.0, 0.0, RADIUS + separation), wp.quat_identity())
    if scene == "free":
        body = builder.add_body(
            xform=xform,
            mass=mass,
            inertia=_sphere_inertia(mass),
            lock_inertia=True,
        )
    elif scene == "articulated":
        # A concentric, shape-free fixed child makes this a real floating
        # articulation without adding a second contact or a changing contact
        # point velocity to the analytical one-row problem.
        body = builder.add_link(
            xform=xform,
            mass=mass,
            inertia=_sphere_inertia(mass),
            lock_inertia=True,
        )
        child_mass = 0.25 * mass
        child = builder.add_link(
            xform=xform,
            mass=child_mass,
            inertia=_sphere_inertia(child_mass),
            lock_inertia=True,
        )
        root_joint = builder.add_joint_free(child=body)
        fixed_joint = builder.add_joint_fixed(parent=body, child=child)
        builder.add_articulation([root_joint, fixed_joint])
    else:
        raise ValueError(f"unknown scene {scene!r}")

    builder.add_shape_sphere(body, radius=RADIUS, cfg=_shape_cfg(restitution))
    builder.add_ground_plane(cfg=_shape_cfg(restitution if plane_restitution is None else plane_restitution))
    return builder.finalize(device=device), body


def _make_solver(
    model,
    *,
    enable_restitution=None,
    velocity_iterations=0,
    response="immediate",
    restitution_velocity_threshold=0.0,
    iterations=16,
    contact_warmstart=False,
    mf_warmstart=False,
    solve_mode="matrix_free",
):
    """Construct a deterministic frictionless FeatherPGS test solver."""
    kwargs = {
        "contact_friction_scale": 0.0,
        "solve_mode": solve_mode,
        "articulated_contact_response": response,
        "iterations": iterations,
        "velocity_iterations": velocity_iterations,
        "cfm": 0.0,
        "relaxation": 1.0,
        "contact_warmstart": contact_warmstart or mf_warmstart,
    }
    if enable_restitution is not None:
        kwargs["enable_restitution"] = enable_restitution
    if restitution_velocity_threshold is not None:
        kwargs["restitution_velocity_threshold"] = restitution_velocity_threshold
    solver = newton.solvers.SolverFeatherPGS(model, **kwargs)
    solver._rigid_body_angular_damping.zero_()
    return solver


def _reset_state(model, state, vertical_velocity):
    """Restore the authored pose and set a free root's world-z velocity."""
    wp.copy(state.joint_q, model.joint_q)
    joint_qd = np.zeros(model.joint_dof_count, dtype=np.float32)
    joint_qd[2] = vertical_velocity
    state.joint_qd.assign(joint_qd)
    state.clear_forces()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)


def _step_plane(
    device,
    *,
    separation,
    speed,
    restitution,
    dt=DEFAULT_DT,
    mass=1.0,
    plane_restitution=None,
    gravity=(0.0, 0.0, 0.0),
    initial_vertical_velocity=None,
    scene="free",
    response="immediate",
    enable_restitution=None,
    velocity_iterations=0,
    restitution_velocity_threshold=0.0,
    iterations=16,
    solve_mode="matrix_free",
):
    """Run one sphere-plane step and return observable contact and state data."""
    model, body = _build_plane_model(
        device,
        separation=separation,
        restitution=restitution,
        plane_restitution=plane_restitution,
        mass=mass,
        gravity=gravity,
        scene=scene,
    )
    solver = _make_solver(
        model,
        velocity_iterations=velocity_iterations,
        response=response,
        enable_restitution=enable_restitution,
        restitution_velocity_threshold=restitution_velocity_threshold,
        iterations=iterations,
        solve_mode=solve_mode,
    )
    state_in, state_out = model.state(), model.state()
    vz = -speed if initial_vertical_velocity is None else initial_vertical_velocity
    _reset_state(model, state_in, vz)
    before = float(state_in.body_qd.numpy()[body][2])

    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
    contacts = pipeline.contacts()
    pipeline.collide(state_in, contacts)
    solver.step(state_in, state_out, model.control(), contacts, dt)

    count = int(contacts.rigid_contact_count.numpy()[0])
    active_paths = {int(v) for v in solver._contact_path.numpy()[:count] if v >= 0}
    qd = state_out.body_qd.numpy()[body].astype(np.float64)
    q = state_out.body_q.numpy()[body].astype(np.float64)
    return {
        "before": before,
        "after": float(qd[2]),
        "body_qd": qd,
        "end_gap": float(q[2] - RADIUS),
        "contact_count": count,
        "paths": active_paths,
    }


def _assert_velocity(test, actual, expected, scale, label, *, rtol=2.0e-3, atol=2.0e-4):
    """Assert a scalar velocity with a scale-aware float32 tolerance."""
    test.assertAlmostEqual(
        actual,
        expected,
        delta=atol + rtol * max(abs(scale), abs(expected)),
        msg=f"{label}: got {actual:+.7f} m/s, expected {expected:+.7f} m/s",
    )


def _assert_crossing_setup(test, result, separation, speed, dt, label):
    """Prove that a test generated a positive-gap contact crossing this step."""
    test.assertGreater(separation, 0.0, f"{label}: contact did not start separated")
    test.assertLess(separation, speed * dt, f"{label}: contact could not reach the surface this step")
    test.assertEqual(result["contact_count"], 1, f"{label}: expected exactly one contact")
    _assert_velocity(test, result["before"], -speed, speed, f"{label}, incident state", rtol=1.0e-6, atol=1.0e-6)


def _assert_restitution_row_position(test, result, separation, speed, restitution, dt, label):
    """Assert the single-state PGS position law and its known time-of-impact error."""
    expected_gap = separation + restitution * speed * dt
    test.assertAlmostEqual(
        result["end_gap"],
        expected_gap,
        delta=max(5.0e-6, 3.0e-3 * expected_gap),
        msg=f"{label}: ordinary restitution row used an unexpected position trajectory",
    )
    physical_gap = restitution * (speed * dt - separation)
    expected_error = (1.0 + restitution) * separation
    test.assertAlmostEqual(
        result["end_gap"] - physical_gap,
        expected_error,
        delta=max(8.0e-6, 4.0e-3 * expected_error),
        msg=f"{label}: phase-dependent position error changed unexpectedly",
    )


def test_one_step_restitution_law_over_coefficient_speed_and_mass(test, device):
    """Enforce Newton's impact law across coefficient, speed, and mass scales."""
    cases = []
    cases.extend((f"e={e}", e, 4.0, 1.0) for e in (0.25, 0.6, 1.0))
    cases.extend((f"speed={speed}", 0.6, speed, 1.0) for speed in (0.05, 2.0, 20.0))
    cases.extend((f"mass={mass}", 0.6, 4.0, mass) for mass in (1.0e-6, 1.0, 1.0e6))

    for label, restitution, speed, mass in cases:
        with test.subTest(case=label):
            separation = IMPACT_FRACTION * speed * DEFAULT_DT
            result = _step_plane(
                device,
                separation=separation,
                speed=speed,
                restitution=restitution,
                mass=mass,
            )
            _assert_crossing_setup(test, result, separation, speed, DEFAULT_DT, label)
            expected = restitution * speed
            _assert_velocity(test, result["after"], expected, speed, label)
            _assert_restitution_row_position(test, result, separation, speed, restitution, DEFAULT_DT, label)
            energy_ratio = (result["after"] / speed) ** 2
            test.assertAlmostEqual(
                energy_ratio,
                restitution * restitution,
                delta=5.0e-3,
                msg=f"{label}: kinetic-energy ratio was {energy_ratio:.6f}",
            )


def test_crossing_result_is_invariant_to_dt_and_time_of_impact(test, device):
    """Use incident speed rather than gap-clipped velocity at every impact time."""
    restitution = 0.65
    speed = 3.0
    cases = (
        # At impact_fraction -> 0 the single-state position law coincides with
        # the analytical rebound (both e*speed*dt), so the first case anchors
        # the position assertions to exact ground truth; the rest pin the law
        # and its known (1+e)*separation deviation across the step phase.
        (1.0e-3, 0.01),
        (2.5e-4, 0.1),
        (2.5e-4, 0.9),
        (1.0e-3, 0.5),
        (1.0e-2, 0.1),
        (1.0e-2, 0.9),
    )
    for iterations in (1, 2, 8):
        for dt, impact_fraction in cases:
            label = f"iters={iterations}, dt={dt:g}, impact/dt={impact_fraction:g}"
            with test.subTest(iterations=iterations, dt=dt, impact_fraction=impact_fraction):
                separation = speed * dt * impact_fraction
                result = _step_plane(
                    device,
                    separation=separation,
                    speed=speed,
                    restitution=restitution,
                    dt=dt,
                    iterations=iterations,
                )
                _assert_crossing_setup(test, result, separation, speed, dt, label)
                _assert_velocity(test, result["after"], restitution * speed, speed, label)
                _assert_restitution_row_position(test, result, separation, speed, restitution, dt, label)


def test_penetrating_impact_replaces_baumgarte_with_restitution(test, device):
    """Keep penetration correction out of the qualifying restitution target."""
    restitution = 0.5
    speed = 2.0
    for penetration in (1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2):
        label = f"penetration={penetration:g}"
        with test.subTest(penetration=penetration):
            result = _step_plane(
                device,
                separation=-penetration,
                speed=speed,
                restitution=restitution,
                velocity_iterations=0,
                iterations=8,
                solve_mode="split",
            )
            test.assertEqual(result["contact_count"], 1, f"{label}: expected exactly one contact")
            test.assertEqual(result["paths"], {PATH_MATRIX_FREE}, f"{label}: wrong contact route")
            _assert_velocity(test, result["before"], -speed, speed, f"{label}, incident state")
            _assert_velocity(test, result["after"], restitution * speed, speed, label)


def test_restitution_uses_the_post_force_incident_predictor(test, device):
    """Include this step's known gravity impulse in the frozen incident speed."""
    restitution = 0.6
    initial_speed = 2.0
    gravity = 9.81
    dt = 1.0e-2
    incident_speed = initial_speed + gravity * dt
    separation = IMPACT_FRACTION * incident_speed * dt
    result = _step_plane(
        device,
        separation=separation,
        speed=initial_speed,
        restitution=restitution,
        dt=dt,
        gravity=(0.0, 0.0, -gravity),
    )

    test.assertEqual(result["contact_count"], 1, "gravity case generated no unique contact")
    _assert_velocity(test, result["before"], -initial_speed, initial_speed, "gravity, authored velocity")
    _assert_restitution_row_position(
        test, result, separation, incident_speed, restitution, dt, "gravity, post-force incident predictor"
    )
    _assert_velocity(
        test,
        result["after"],
        restitution * incident_speed,
        incident_speed,
        "gravity, post-force incident predictor",
    )


def test_non_crossing_speculative_contact_does_not_bounce(test, device):
    """Preserve free approach when a speculative contact cannot impact this step."""
    restitution = 1.0
    speed = 3.0
    separation = 1.25 * speed * DEFAULT_DT
    result = _step_plane(
        device,
        separation=separation,
        speed=speed,
        restitution=restitution,
    )
    test.assertGreater(result["contact_count"], 0, "the speculative contact was never generated")
    test.assertGreater(separation - speed * DEFAULT_DT, 0.0, "the setup unexpectedly crosses the surface")
    _assert_velocity(test, result["after"], -speed, speed, "non-crossing contact")
    test.assertGreater(result["end_gap"], 0.0, "the non-crossing body reached or crossed the plane")


def test_shape_restitution_uses_symmetric_arithmetic_average(test, device):
    """Average sanitized shape coefficients symmetrically, including authored zero."""
    speed = 4.0
    separation = IMPACT_FRACTION * speed * DEFAULT_DT
    cases = (
        (0.0, 0.8, 0.4),
        (0.2, 0.8, 0.5),
        (0.8, 0.2, 0.5),
        (2.0, 0.8, 0.9),
        (-0.2, 0.8, 0.4),
        (float("nan"), 0.8, 0.4),
    )
    for sphere_e, plane_e, mixed in cases:
        label = f"sphere={sphere_e}, plane={plane_e}"
        with test.subTest(sphere=sphere_e, plane=plane_e):
            result = _step_plane(
                device,
                separation=separation,
                speed=speed,
                restitution=sphere_e,
                plane_restitution=plane_e,
            )
            _assert_crossing_setup(test, result, separation, speed, DEFAULT_DT, label)
            _assert_velocity(test, result["after"], mixed * speed, speed, label)


def test_restitution_velocity_threshold_uses_incident_relative_speed(test, device):
    """Suppress low-speed bounce and activate immediately above the threshold."""
    threshold = 0.5
    restitution = 0.8
    for speed in (0.25, 0.49, 0.51, 2.0):
        label = f"speed={speed}, threshold={threshold}"
        with test.subTest(speed=speed):
            separation = IMPACT_FRACTION * speed * DEFAULT_DT
            result = _step_plane(
                device,
                separation=separation,
                speed=speed,
                restitution=restitution,
                restitution_velocity_threshold=threshold,
            )
            _assert_crossing_setup(test, result, separation, speed, DEFAULT_DT, label)
            expected = restitution * speed if speed > threshold else -separation / DEFAULT_DT
            _assert_velocity(test, result["after"], expected, speed, label)
            if speed > threshold:
                _assert_restitution_row_position(test, result, separation, speed, restitution, DEFAULT_DT, label)
            else:
                test.assertLess(abs(result["end_gap"]), 5.0e-6, f"{label}: speculative row missed the surface")

    # Both bodies move quickly in the world frame, but their closing speed is
    # below the threshold. This catches a gate applied to either body's
    # absolute speed instead of the contact's relative normal speed.
    out_a, out_b, _distance, count, _paths = _step_two_spheres(
        device,
        mass_a=1.0,
        mass_b=1.0,
        velocity_a=10.0,
        velocity_b=9.6,
        restitution=restitution,
        restitution_velocity_threshold=threshold,
    )
    test.assertEqual(count, 1, "relative-threshold test generated no unique contact")
    # Restitution is suppressed, but the pair still closes the remaining gap
    # under the speculative allowance and arrives touching with relative speed
    # separation/dt, the impulse shared equally between the equal masses -- the
    # same rule the single-body sub-cases assert above. Expecting a common 9.8
    # would freeze the bodies mid-air while still separated.
    residual_closing = IMPACT_FRACTION * (10.0 - 9.6)
    _assert_velocity(test, out_a, 9.8 + 0.5 * residual_closing, 0.4, "relative threshold, body A")
    _assert_velocity(test, out_b, 9.8 - 0.5 * residual_closing, 0.4, "relative threshold, body B")
    test.assertAlmostEqual(out_a + out_b, 19.6, delta=1.0e-4, msg="relative threshold: momentum drifted")


def test_bouncing_ball_settles_to_rest_under_gravity(test, device):
    """Settle a bouncing ball on the surface without perpetual micro-bounce or sinking.

    Restitution shares the contact row with the speculative/Baumgarte law, so
    the long-run interaction with resting contact is a distinct failure mode
    from any single-impact assertion: a bounce target that keeps firing at
    resting speeds produces a Zeno micro-bounce, while an over-eager gate
    lets the row sink. Impacts decay 2.0 -> 1.0 -> 0.5 m/s, the third landing
    falls at the 0.5 m/s threshold and is suppressed, and the tail of the
    rollout must be quiescent on the surface.
    """
    restitution = 0.5
    speed = 2.0
    gravity = 9.81
    separation = IMPACT_FRACTION * speed * DEFAULT_DT
    model, body = _build_plane_model(
        device,
        separation=separation,
        restitution=restitution,
        gravity=(0.0, 0.0, -gravity),
    )
    solver = _make_solver(model, restitution_velocity_threshold=0.5)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
    contacts = pipeline.contacts()
    state_in, state_out = model.state(), model.state()
    _reset_state(model, state_in, -speed)

    heights, speeds = [], []
    for _ in range(800):
        contacts.clear()
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, model.control(), contacts, DEFAULT_DT)
        state_in, state_out = state_out, state_in
        heights.append(float(state_in.body_q.numpy()[body][2]))
        speeds.append(float(state_in.body_qd.numpy()[body][2]))
    heights = np.array(heights)
    speeds = np.array(speeds)

    # Non-vacuity: the first impact must actually rebound at ~e*speed.
    test.assertGreater(float(speeds[:100].max()), 0.9 * restitution * speed, "the ball never bounced")
    test.assertLess(float(heights.min()), RADIUS + 1.0e-3, "the ball never reached the surface")
    test.assertGreater(float(heights.min()), RADIUS - 2.0e-3, "the ball sank through the plane")
    # Quiescence: by 0.6 s every impact is below the threshold, so the tail
    # must rest on the surface instead of bouncing or creeping.
    tail_heights = heights[-200:]
    tail_speeds = speeds[-200:]
    test.assertLess(float(np.abs(tail_speeds).max()), 0.1, "resting contact kept bouncing")
    test.assertLess(float(tail_heights.max()), RADIUS + 1.5e-3, "resting contact hovered above the surface")
    test.assertGreater(float(tail_heights.min()), RADIUS - 1.5e-3, "resting contact sank into the plane")


def _build_two_sphere_model(device, *, separation, restitution, mass_a, mass_b):
    """Build two free spheres closing along their line of centres."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    cfg = _shape_cfg(restitution)
    body_a = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        mass=mass_a,
        inertia=_sphere_inertia(mass_a),
        lock_inertia=True,
    )
    body_b = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 2.0 * RADIUS + separation), wp.quat_identity()),
        mass=mass_b,
        inertia=_sphere_inertia(mass_b),
        lock_inertia=True,
    )
    builder.add_shape_sphere(body_a, radius=RADIUS, cfg=cfg)
    builder.add_shape_sphere(body_b, radius=RADIUS, cfg=cfg)
    return builder.finalize(device=device), body_a, body_b


def _step_two_spheres(
    device,
    *,
    mass_a,
    mass_b,
    velocity_a,
    velocity_b,
    restitution,
    restitution_velocity_threshold=0.0,
):
    """Run one two-body impact and return the final normal velocities."""
    closing_speed = velocity_a - velocity_b
    separation = IMPACT_FRACTION * closing_speed * DEFAULT_DT
    model, body_a, body_b = _build_two_sphere_model(
        device,
        separation=separation,
        restitution=restitution,
        mass_a=mass_a,
        mass_b=mass_b,
    )
    solver = _make_solver(model, restitution_velocity_threshold=restitution_velocity_threshold)
    state_in, state_out = model.state(), model.state()
    joint_qd = np.zeros(model.joint_dof_count, dtype=np.float32)
    joint_qd[2] = velocity_a
    joint_qd[8] = velocity_b
    state_in.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
    contacts = pipeline.contacts()
    pipeline.collide(state_in, contacts)
    solver.step(state_in, state_out, model.control(), contacts, DEFAULT_DT)
    qd = state_out.body_qd.numpy()
    q = state_out.body_q.numpy()
    count = int(contacts.rigid_contact_count.numpy()[0])
    paths = {int(v) for v in solver._contact_path.numpy()[:count] if v >= 0}
    return float(qd[body_a][2]), float(qd[body_b][2]), float(q[body_b][2] - q[body_a][2]), count, paths


def test_two_body_impact_conserves_momentum_and_reverses_relative_speed(test, device):
    """Match the exact two-body impact solution over mass ratio and frame shift."""
    restitution = 0.6
    for mass_a, mass_b, frame_speed in ((1.0, 1.0, 0.0), (1.0, 4.0, 0.0), (4.0, 1.0, 0.0), (1.0, 4.0, 3.0)):
        velocity_a = 2.0 + frame_speed
        velocity_b = -1.0 + frame_speed
        label = f"m=({mass_a},{mass_b}), frame={frame_speed}"
        with test.subTest(mass_a=mass_a, mass_b=mass_b, frame_speed=frame_speed):
            out_a, out_b, distance, count, paths = _step_two_spheres(
                device,
                mass_a=mass_a,
                mass_b=mass_b,
                velocity_a=velocity_a,
                velocity_b=velocity_b,
                restitution=restitution,
            )
            total_mass = mass_a + mass_b
            expected_a = (
                (mass_a - restitution * mass_b) * velocity_a + (1.0 + restitution) * mass_b * velocity_b
            ) / total_mass
            expected_b = (
                (1.0 + restitution) * mass_a * velocity_a + (mass_b - restitution * mass_a) * velocity_b
            ) / total_mass
            test.assertEqual(count, 1, f"{label}: expected exactly one sphere-sphere contact")
            test.assertEqual(paths, {PATH_MATRIX_FREE}, f"{label}: unexpected contact route {paths}")
            closing_speed = velocity_a - velocity_b
            separation = IMPACT_FRACTION * closing_speed * DEFAULT_DT
            expected_distance = 2.0 * RADIUS + separation + restitution * closing_speed * DEFAULT_DT
            test.assertAlmostEqual(distance, expected_distance, delta=2.0e-4, msg=f"{label}: wrong end separation")
            _assert_velocity(test, out_a, expected_a, velocity_a - velocity_b, f"{label}, body A")
            _assert_velocity(test, out_b, expected_b, velocity_a - velocity_b, f"{label}, body B")

            momentum_before = mass_a * velocity_a + mass_b * velocity_b
            momentum_after = mass_a * out_a + mass_b * out_b
            test.assertAlmostEqual(
                momentum_after,
                momentum_before,
                delta=2.0e-3 * max(1.0, abs(momentum_before)),
                msg=f"{label}: normal momentum changed",
            )
            test.assertAlmostEqual(
                out_b - out_a,
                -restitution * (velocity_b - velocity_a),
                delta=5.0e-3,
                msg=f"{label}: relative impact law failed",
            )
            reduced_mass = mass_a * mass_b / total_mass
            expected_energy_loss = 0.5 * reduced_mass * (1.0 - restitution**2) * (velocity_a - velocity_b) ** 2
            energy_before = 0.5 * mass_a * velocity_a**2 + 0.5 * mass_b * velocity_b**2
            energy_after = 0.5 * mass_a * out_a**2 + 0.5 * mass_b * out_b**2
            test.assertAlmostEqual(
                energy_before - energy_after,
                expected_energy_loss,
                delta=5.0e-3 * max(1.0, energy_before),
                msg=f"{label}: collision energy loss was inconsistent with e",
            )


def test_restitution_is_relative_to_a_moving_kinematic_surface(test, device):
    """Measure rebound relative to prescribed motion on MF and propagation rows."""
    restitution = 0.6
    closing_speed = 3.0
    separation = IMPACT_FRACTION * closing_speed * DEFAULT_DT
    expected_path = {"immediate": PATH_MATRIX_FREE, "propagation": PATH_PROPAGATION}

    for response in ("immediate", "propagation"):
        for frame_speed in (0.0, 7.0):
            label = f"{response}, frame={frame_speed}"
            with test.subTest(response=response, frame_speed=frame_speed):
                builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
                cfg = _shape_cfg(restitution)
                dynamic_body = builder.add_body(
                    xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
                    mass=1.0,
                    inertia=_sphere_inertia(1.0),
                    lock_inertia=True,
                )
                kinematic_body = builder.add_body(
                    xform=wp.transform(wp.vec3(0.0, 0.0, 2.0 * RADIUS + separation), wp.quat_identity()),
                    mass=1.0,
                    inertia=_sphere_inertia(1.0),
                    lock_inertia=True,
                    is_kinematic=True,
                )
                builder.add_shape_sphere(dynamic_body, radius=RADIUS, cfg=cfg)
                builder.add_shape_sphere(kinematic_body, radius=RADIUS, cfg=cfg)
                model = builder.finalize(device=device)
                solver = _make_solver(model, response=response)
                state_in, state_out = model.state(), model.state()
                velocity_dynamic = frame_speed + 2.0
                velocity_surface = frame_speed - 1.0
                joint_qd = np.zeros(model.joint_dof_count, dtype=np.float32)
                joint_qd[2] = velocity_dynamic
                joint_qd[8] = velocity_surface
                state_in.joint_qd.assign(joint_qd)
                newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
                pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
                contacts = pipeline.contacts()
                pipeline.collide(state_in, contacts)
                solver.step(state_in, state_out, model.control(), contacts, DEFAULT_DT)

                count = int(contacts.rigid_contact_count.numpy()[0])
                paths = {int(v) for v in solver._contact_path.numpy()[:count] if v >= 0}
                velocity_after = float(state_out.body_qd.numpy()[dynamic_body][2])
                expected_after = velocity_surface - restitution * closing_speed
                test.assertEqual(count, 1, f"{label}: expected one sphere contact")
                test.assertEqual(paths, {expected_path[response]}, f"{label}: wrong contact route")
                _assert_velocity(test, velocity_after, expected_after, closing_speed, label, rtol=5.0e-3)
                test.assertAlmostEqual(
                    velocity_surface - velocity_after,
                    restitution * closing_speed,
                    delta=1.5e-2,
                    msg=f"{label}: restitution was not measured relative to the moving surface",
                )


def test_every_contact_response_route_enforces_restitution(test, device):
    """Enforce the same impact law on every measured internal response route."""
    restitution = 0.65
    speed = 3.0
    separation = IMPACT_FRACTION * speed * DEFAULT_DT
    for scene in ("free", "articulated"):
        for response in RESPONSE_MODES:
            label = f"{scene}/{response}"
            with test.subTest(scene=scene, response=response):
                result = _step_plane(
                    device,
                    separation=separation,
                    speed=speed,
                    restitution=restitution,
                    scene=scene,
                    response=response,
                    velocity_iterations=0,
                    iterations=16,
                )
                _assert_crossing_setup(test, result, separation, speed, DEFAULT_DT, label)
                test.assertEqual(result["paths"], EXPECTED_PATHS[(scene, response)], f"{label}: wrong route")
                _assert_velocity(test, result["after"], restitution * speed, speed, label, rtol=5.0e-3)


def test_dense_and_production_split_modes_enforce_restitution_without_velocity_iterations(test, device):
    """Apply restitution in dense and production split rows with explicit zero refinement."""
    restitution = 0.65
    speed = 3.0
    separation = IMPACT_FRACTION * speed * DEFAULT_DT
    cases = (
        ("split/free", "split", "free", PATH_MATRIX_FREE),
        ("split/articulated", "split", "articulated", PATH_DENSE),
    )
    for label, solve_mode, scene, expected_path in cases:
        with test.subTest(case=label):
            result = _step_plane(
                device,
                separation=separation,
                speed=speed,
                restitution=restitution,
                scene=scene,
                velocity_iterations=0,
                iterations=16,
                solve_mode=solve_mode,
            )
            _assert_crossing_setup(test, result, separation, speed, DEFAULT_DT, label)
            test.assertEqual(result["paths"], {expected_path}, f"{label}: wrong route")
            _assert_velocity(test, result["after"], restitution * speed, speed, label, rtol=5.0e-3)


def test_multiworld_restitution_keeps_case_data_isolated(test, device):
    """Apply each world's coefficient and crossing state without cross-world leakage."""
    # (e, speed, mass, toi/dt); the third case is deliberately non-crossing.
    cases = ((0.25, 2.0, 1.0, 0.4), (0.75, 4.0, 10.0, 0.7), (1.0, 3.0, 0.5, 1.25), (0.0, 5.0, 2.0, 0.2))
    scene = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    for restitution, speed, mass, impact_fraction in cases:
        world = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        separation = speed * DEFAULT_DT * impact_fraction
        body = world.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, RADIUS + separation), wp.quat_identity()),
            mass=mass,
            inertia=_sphere_inertia(mass),
            lock_inertia=True,
        )
        cfg = _shape_cfg(restitution)
        world.add_shape_sphere(body, radius=RADIUS, cfg=cfg)
        world.add_ground_plane(cfg=cfg)
        scene.add_world(world)
    model = scene.finalize(device=device)
    solver = _make_solver(model)
    state_in, state_out = model.state(), model.state()
    joint_qd = np.zeros((len(cases), 6), dtype=np.float32)
    joint_qd[:, 2] = [-case[1] for case in cases]
    state_in.joint_qd.assign(joint_qd.reshape(-1))
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
    contacts = pipeline.contacts()
    pipeline.collide(state_in, contacts)
    solver.step(state_in, state_out, model.control(), contacts, DEFAULT_DT)

    count = int(contacts.rigid_contact_count.numpy()[0])
    after = state_out.body_qd.numpy()[:, 2]
    end_gap = state_out.body_q.numpy()[:, 2] - RADIUS
    paths = solver._contact_path.numpy()[:count]
    test.assertEqual(count, len(cases), "expected exactly one contact in every world")
    test.assertTrue(np.all(paths == PATH_MATRIX_FREE), f"multiworld contacts used paths {paths.tolist()}")
    for world, (restitution, speed, _mass, impact_fraction) in enumerate(cases):
        with test.subTest(world=world):
            crossing = impact_fraction < 1.0
            if crossing and restitution > 0.0:
                expected = restitution * speed
            elif crossing:
                expected = -impact_fraction * speed
            else:
                expected = -speed
            _assert_velocity(test, float(after[world]), expected, speed, f"world {world}")
            if crossing and restitution > 0.0:
                expected_gap = impact_fraction * speed * DEFAULT_DT + restitution * speed * DEFAULT_DT
                test.assertAlmostEqual(
                    float(end_gap[world]), expected_gap, delta=2.0e-4, msg=f"world {world} wrong row trajectory"
                )
            elif crossing:
                test.assertLess(abs(float(end_gap[world])), 2.0e-4, f"world {world} missed the surface")
            else:
                test.assertGreater(float(end_gap[world]), 0.0, f"world {world} unexpectedly reached the surface")


def test_restitution_uses_ordinary_rows_and_validates_configuration(test, device):
    """Keep velocity iterations explicit while provisioning live restitution rows."""
    positive_model, _ = _build_plane_model(
        device,
        separation=1.0e-3,
        restitution=0.6,
    )
    solver = _make_solver(positive_model, velocity_iterations=0)
    test.assertEqual(solver._velocity_iterations, 0)
    test.assertFalse(hasattr(solver, "_velocity_post_iterations"), "zero iterations were silently overridden")
    test.assertEqual(solver._row_restitution.shape, (solver._world_count, solver._articulation_constraint_max))
    test.assertEqual(solver._mf_row_restitution.shape, (solver._world_count, solver._free_body_constraint_max))
    for invalid_threshold in (-1.0, float("nan"), float("inf")):
        with test.subTest(invalid_threshold=invalid_threshold):
            with test.assertRaisesRegex(ValueError, "restitution_velocity_threshold"):
                _make_solver(positive_model, restitution_velocity_threshold=invalid_threshold)

    zero_model, _ = _build_plane_model(
        device,
        separation=1.0e-3,
        restitution=0.0,
    )
    zero_solver = _make_solver(zero_model, velocity_iterations=0)
    test.assertEqual(zero_solver._velocity_iterations, 0)

    zero_model.shape_material_restitution.fill_(0.6)
    zero_solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)

    default_solver = _make_solver(positive_model, velocity_iterations=0, restitution_velocity_threshold=None)
    test.assertAlmostEqual(default_solver._restitution_velocity_threshold, 0.5)


def test_restitution_can_be_disabled_without_changing_materials(test, device):
    """Disable restitution while retaining authored material coefficients."""
    model, _ = _build_plane_model(device, separation=1.0e-3, restitution=0.8)
    solver = _make_solver(
        model,
        enable_restitution=False,
        restitution_velocity_threshold=0.75,
    )
    np.testing.assert_array_equal(
        solver._shape_restitution.numpy()[: model.shape_count], model.shape_material_restitution.numpy()
    )
    test.assertAlmostEqual(solver._restitution_velocity_threshold, 0.75)
    test.assertGreater(solver._rebound_velocity_threshold, 1.0e30)

    speed = 2.0
    separation = IMPACT_FRACTION * speed * DEFAULT_DT
    for scene, response in (("free", "immediate"), ("articulated", "immediate"), ("free", "propagation")):
        with test.subTest(scene=scene, response=response):
            result = _step_plane(
                device,
                separation=separation,
                speed=speed,
                restitution=0.8,
                scene=scene,
                response=response,
                enable_restitution=False,
            )
            _assert_crossing_setup(test, result, separation, speed, DEFAULT_DT, f"{scene}/{response}")
            _assert_velocity(
                test,
                result["after"],
                -IMPACT_FRACTION * speed,
                speed,
                f"{scene}/{response}",
            )


def test_velocity_iteration_counts_reach_the_same_single_row_solution(test, device):
    """Keep restitution correct while optional velocity iterations refine the row."""
    restitution = 0.7
    speed = 2.0
    separation = IMPACT_FRACTION * speed * DEFAULT_DT
    for iterations in (0, 1, 4, 12):
        with test.subTest(velocity_iterations=iterations):
            result = _step_plane(
                device,
                separation=separation,
                speed=speed,
                restitution=restitution,
                velocity_iterations=iterations,
            )
            _assert_crossing_setup(test, result, separation, speed, DEFAULT_DT, f"iterations={iterations}")
            _assert_velocity(test, result["after"], restitution * speed, speed, f"iterations={iterations}")


def test_warm_start_history_does_not_change_or_repeat_restitution(test, device):
    """Use the current incident velocity once under dense and identity warm starting."""
    restitution = 0.6
    # Both incident speeds used below must cross from this same authored pose.
    separation = IMPACT_FRACTION * 2.0 * DEFAULT_DT
    cases = (
        ("matrix-free MF", "matrix_free", "free", True, False, PATH_MATRIX_FREE),
        ("matrix-free dense", "matrix_free", "articulated", False, True, PATH_DENSE),
        ("split MF", "split", "free", True, False, PATH_MATRIX_FREE),
        ("split dense", "split", "articulated", False, True, PATH_DENSE),
    )
    for label, solve_mode, scene, mf_warmstart, contact_warmstart, expected_path in cases:
        with test.subTest(warmstart=label):
            model, body = _build_plane_model(
                device,
                separation=separation,
                restitution=restitution,
                scene=scene,
            )
            solver = _make_solver(
                model,
                velocity_iterations=0,
                mf_warmstart=mf_warmstart,
                contact_warmstart=contact_warmstart,
                solve_mode=solve_mode,
            )
            pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching="latest")
            contacts = pipeline.contacts()
            state_in, state_out = model.state(), model.state()

            for speed in (2.0, 5.0):
                _reset_state(model, state_in, -speed)
                contacts.clear()
                pipeline.collide(state_in, contacts)
                solver.step(state_in, state_out, model.control(), contacts, DEFAULT_DT)
                count = int(contacts.rigid_contact_count.numpy()[0])
                test.assertEqual(count, 1, f"{label}: reset impact generated {count} contacts")
                test.assertEqual(
                    {int(v) for v in solver._contact_path.numpy()[:count] if v >= 0},
                    {expected_path},
                    f"{label}: reset impact used the wrong route",
                )
                after = float(state_out.body_qd.numpy()[body][2])
                _assert_velocity(test, after, restitution * speed, speed, f"{label}, speed={speed}", rtol=5.0e-3)

            # The second impact ends on the plane and separates. A persistent
            # matched contact on the next step must not apply restitution a
            # second time from either warm-start impulse history or stale u^-.
            contacts.clear()
            pipeline.collide(state_out, contacts)
            test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0, f"{label}: contact did not persist")
            solver.step(state_out, state_in, model.control(), contacts, DEFAULT_DT)
            separating = float(state_in.body_qd.numpy()[body][2])
            _assert_velocity(test, separating, restitution * 5.0, 5.0, f"{label}, separating step", rtol=1.0e-2)


def test_redundant_box_contacts_do_not_multiply_restitution_energy(test, device):
    """Bounce a flat box without row-count-dependent speed, torque, or energy gain."""
    hx, hy, hz = 0.1, 0.08, 0.05
    mass = 2.0
    restitution = 0.6
    speed = 3.0
    separation = IMPACT_FRACTION * speed * DEFAULT_DT
    ix = mass * (hy * hy + hz * hz) / 3.0
    iy = mass * (hx * hx + hz * hz) / 3.0
    iz = mass * (hx * hx + hy * hy) / 3.0
    inertia = wp.mat33(ix, 0.0, 0.0, 0.0, iy, 0.0, 0.0, 0.0, iz)

    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, hz + separation), wp.quat_identity()),
        mass=mass,
        inertia=inertia,
        lock_inertia=True,
    )
    cfg = _shape_cfg(restitution)
    builder.add_shape_box(body, hx=hx, hy=hy, hz=hz, cfg=cfg)
    builder.add_ground_plane(cfg=cfg)
    model = builder.finalize(device=device)
    solver = _make_solver(model, velocity_iterations=32, iterations=32)
    state_in, state_out = model.state(), model.state()
    _reset_state(model, state_in, -speed)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", reduce_contacts=False)
    contacts = pipeline.contacts()
    pipeline.collide(state_in, contacts)
    solver.step(state_in, state_out, model.control(), contacts, DEFAULT_DT)

    count = int(contacts.rigid_contact_count.numpy()[0])
    qd = state_out.body_qd.numpy()[body].astype(np.float64)
    end_gap = float(state_out.body_q.numpy()[body][2] - hz)
    test.assertGreaterEqual(count, 4, f"box test generated only {count} contacts")
    _assert_velocity(test, float(qd[2]), restitution * speed, speed, "flat box", rtol=1.0e-2)
    test.assertLess(np.linalg.norm(qd[:2]), 1.0e-2, f"flat box gained tangential speed {qd[:2]}")
    test.assertLess(np.linalg.norm(qd[3:]), 2.0e-2, f"flat box gained angular speed {qd[3:]}")
    expected_gap = separation + restitution * speed * DEFAULT_DT
    test.assertAlmostEqual(end_gap, expected_gap, delta=3.0e-4, msg=f"flat box ended at gap {end_gap:+.3e} m")
    energy_before = 0.5 * mass * speed * speed
    energy_after = 0.5 * mass * float(np.dot(qd[:3], qd[:3])) + 0.5 * (
        ix * qd[3] ** 2 + iy * qd[4] ** 2 + iz * qd[5] ** 2
    )
    test.assertAlmostEqual(
        energy_after / energy_before,
        restitution * restitution,
        delta=2.0e-2,
        msg=f"flat-box energy ratio was {energy_after / energy_before:.6f}",
    )


def test_articulated_redundant_manifold_preserves_symmetric_rebound(test, device):
    """Bounce an articulated four-point foot without spurious spin or energy gain."""
    mass = 2.0
    child_mass = 0.25
    half = 0.08
    sphere_radius = 0.02
    foot_offset = 0.03
    restitution = 0.6
    speed = 3.0
    separation = IMPACT_FRACTION * speed * DEFAULT_DT
    root_height = foot_offset + sphere_radius + separation
    inertia = _diagonal_inertia(0.02)

    for joint_kind in ("fixed", "revolute"):
        for response in ("immediate", "propagation_fused", "propagation_colored"):
            with test.subTest(joint=joint_kind, response=response):
                builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
                xform = wp.transform(wp.vec3(0.0, 0.0, root_height), wp.quat_identity())
                root = builder.add_link(xform=xform, mass=mass, inertia=inertia, lock_inertia=True)
                child = builder.add_link(
                    xform=xform,
                    mass=child_mass,
                    inertia=_diagonal_inertia(0.002),
                    lock_inertia=True,
                )
                root_joint = builder.add_joint_free(child=root)
                if joint_kind == "fixed":
                    child_joint = builder.add_joint_fixed(parent=root, child=child)
                else:
                    child_joint = builder.add_joint_revolute(
                        parent=root,
                        child=child,
                        axis=wp.vec3(0.0, 0.0, 1.0),
                    )
                builder.add_articulation([root_joint, child_joint])
                cfg = _shape_cfg(restitution)
                for sx, sy in ((-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)):
                    builder.add_shape_sphere(
                        root,
                        xform=wp.transform(wp.vec3(sx * half, sy * half, -foot_offset), wp.quat_identity()),
                        radius=sphere_radius,
                        cfg=cfg,
                    )
                builder.add_ground_plane(cfg=cfg)
                model = builder.finalize(device=device)
                solver = _make_solver(
                    model,
                    response=response,
                    velocity_iterations=0,
                    iterations=16,
                )
                state_in, state_out = model.state(), model.state()
                _reset_state(model, state_in, -speed)
                pipeline = newton.CollisionPipeline(model, broad_phase="nxn", reduce_contacts=False)
                contacts = pipeline.contacts()
                pipeline.collide(state_in, contacts)
                solver.step(state_in, state_out, model.control(), contacts, DEFAULT_DT)

                count = int(contacts.rigid_contact_count.numpy()[0])
                paths = {int(v) for v in solver._contact_path.numpy()[:count] if v >= 0}
                qd = state_out.body_qd.numpy()[root].astype(np.float64)
                expected_path = PATH_DENSE if response == "immediate" else PATH_PROPAGATION
                label = f"{joint_kind}/{response}"
                test.assertGreaterEqual(count, 4, f"{label}: manifold had only {count} contacts")
                test.assertEqual(paths, {expected_path}, f"{label}: wrong contact route")
                _assert_velocity(test, float(qd[2]), restitution * speed, speed, label, rtol=7.5e-3)
                test.assertLess(np.linalg.norm(qd[3:]), 3.0e-2, f"{label}: symmetric foot spun at {qd[3:]}")


def test_cuda_graph_replay_reads_current_restitution_and_matches_eager(test, device):
    """Replay collide plus restitution with stable pointers and device-read material data."""
    speed = 3.0
    separation = IMPACT_FRACTION * speed * DEFAULT_DT

    def make_fixture():
        model, body = _build_plane_model(
            device,
            separation=separation,
            restitution=0.0,
        )
        solver = _make_solver(model, velocity_iterations=0)
        state_in, state_out = model.state(), model.state()
        pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
        contacts = pipeline.contacts()
        control = model.control()

        def one_step():
            pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, DEFAULT_DT)
            # Keep addresses stable for graph replay.
            wp.copy(state_in.body_q, state_out.body_q)
            wp.copy(state_in.body_qd, state_out.body_qd)
            wp.copy(state_in.joint_q, state_out.joint_q)
            wp.copy(state_in.joint_qd, state_out.joint_qd)

        return model, solver, body, state_in, contacts, one_step

    graph_model, graph_solver, graph_body, graph_state, graph_contacts, graph_step = make_fixture()
    eager_model, eager_solver, eager_body, eager_state, eager_contacts, eager_step = make_fixture()

    # Warm both fixtures through the same history. Besides compiling kernels,
    # this keeps the solver's double-buffer phase identical for the comparison.
    _reset_state(graph_model, graph_state, -speed)
    _reset_state(eager_model, eager_state, -speed)
    graph_step()
    eager_step()
    _reset_state(graph_model, graph_state, -speed)
    _reset_state(eager_model, eager_state, -speed)

    with wp.ScopedCapture(device) as capture:
        graph_step()

    # Change restitution from zero after capture. Row storage is independent
    # of construction-time values, and the coefficient must come from the
    # solver's device buffer rather than a captured host scalar.
    for model, solver in ((graph_model, graph_solver), (eager_model, eager_solver)):
        model.shape_material_restitution.fill_(0.75)
        solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
    wp.capture_launch(capture.graph)
    eager_step()

    graph_velocity = float(graph_state.body_qd.numpy()[graph_body][2])
    eager_velocity = float(eager_state.body_qd.numpy()[eager_body][2])
    graph_count = int(graph_contacts.rigid_contact_count.numpy()[0])
    eager_count = int(eager_contacts.rigid_contact_count.numpy()[0])
    test.assertEqual(graph_count, 1, "captured collision generated no unique contact")
    test.assertEqual(eager_count, 1, "eager collision generated no unique contact")
    _assert_velocity(test, graph_velocity, 0.75 * speed, speed, "captured step")
    test.assertAlmostEqual(graph_velocity, eager_velocity, delta=1.0e-6, msg="captured and eager steps diverged")


def test_notify_reads_replaced_and_in_place_restitution(test, device, solve_mode="matrix_free"):
    """Match a fresh solver after a SHAPE_PROPERTIES notify, eagerly and in a captured graph."""
    speed = 3.0
    separation = IMPACT_FRACTION * speed * DEFAULT_DT
    # Graph capture needs a CUDA device.
    captures = (False, True) if wp.get_device(device).is_cuda else (False,)
    for scene, old, replace, capture in itertools.product(("free", "articulated"), (0.0, 1.0), (False, True), captures):
        new = 1.0 - old
        label = f"{scene} e={old}->{new} {'replaced' if replace else 'in place'} {'captured' if capture else 'eager'}"
        with test.subTest(label):
            model, body = _build_plane_model(device, separation=separation, restitution=old, scene=scene)
            solver = _make_solver(model, solve_mode=solve_mode)
            state_in, state_out = model.state(), model.state()
            _reset_state(model, state_in, -speed)
            pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
            contacts = pipeline.contacts()
            control = model.control()
            pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, DEFAULT_DT)
            if capture:
                with wp.ScopedCapture(device) as graph:
                    pipeline.collide(state_in, contacts)
                    solver.step(state_in, state_out, control, contacts, DEFAULT_DT)
            if replace:
                model.shape_material_restitution = wp.full(model.shape_count, new, dtype=float, device=device)
            else:
                model.shape_material_restitution.fill_(new)
            solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
            if capture:
                wp.capture_launch(graph.graph)
            else:
                pipeline.collide(state_in, contacts)
                solver.step(state_in, state_out, control, contacts, DEFAULT_DT)

            reference = model.state()
            _make_solver(model, solve_mode=solve_mode).step(state_in, reference, control, contacts, DEFAULT_DT)
            count = int(contacts.rigid_contact_count.numpy()[0])
            paths = {int(v) for v in solver._contact_path.numpy()[:count] if v >= 0}
            test.assertEqual(count, 1, f"{label}: expected one contact")
            test.assertEqual(paths, EXPECTED_PATHS[(scene, "immediate")], f"{label}: wrong route")
            expected = float(reference.body_qd.numpy()[body][2])
            _assert_velocity(test, expected, new * speed if new else -separation / DEFAULT_DT, speed, label)
            test.assertAlmostEqual(float(state_out.body_qd.numpy()[body][2]), expected, delta=2.0e-5, msg=label)


# (label, solve_mode, scene, response, velocity_iterations)
THRESHOLD_EDIT_ROUTES = (
    ("split/free", "split", "free", "immediate", 0),
    ("split/articulated", "split", "articulated", "immediate", 0),
    ("matrix_free/free", "matrix_free", "free", "immediate", 0),
    ("matrix_free/articulated", "matrix_free", "articulated", "immediate", 0),
    ("matrix_free/free/velocity_pass", "matrix_free", "free", "immediate", 2),
    ("matrix_free/free/propagation", "matrix_free", "free", "propagation", 0),
    ("matrix_free/articulated/propagation", "matrix_free", "articulated", "propagation", 0),
)


def _threshold_bounce(device, route, threshold, edited=None, enable_restitution=True):
    """Return the rebound speed of a resting-gap 1 m/s sphere impact and the solver that produced it."""
    _label, solve_mode, scene, response, velocity_iterations = route
    model, body = _build_plane_model(device, separation=0.0, restitution=1.0, scene=scene)
    solver = _make_solver(
        model,
        solve_mode=solve_mode,
        response=response,
        velocity_iterations=velocity_iterations,
        enable_restitution=enable_restitution,
        restitution_velocity_threshold=threshold,
    )
    if edited is not None:
        solver.restitution_velocity_threshold = edited
    state_in, state_out = model.state(), model.state()
    _reset_state(model, state_in, -1.0)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
    contacts = pipeline.contacts()
    pipeline.collide(state_in, contacts)
    solver.step(state_in, state_out, model.control(), contacts, 0.01)
    return float(state_out.body_qd.numpy()[body][2]), solver


def test_edited_restitution_threshold_matches_a_fresh_solver(test, device):
    """Launch an edited threshold on every contact route exactly as a solver constructed with it."""
    # Matrix-free mode runs on CUDA only.
    routes = [r for r in THRESHOLD_EDIT_ROUTES if wp.get_device(device).is_cuda or r[1] != "matrix_free"]
    for route in routes:
        for old, new in ((0.5, 2.0), (2.0, 0.5)):
            label = f"{route[0]} {old}->{new}"
            with test.subTest(label):
                edited, solver = _threshold_bounce(device, route, old, edited=new)
                fresh_old, _ = _threshold_bounce(device, route, old)
                fresh_new, _ = _threshold_bounce(device, route, new)
                test.assertGreater(abs(fresh_new - fresh_old), 0.5, f"{label}: thresholds do not discriminate")
                test.assertAlmostEqual(edited, fresh_new, delta=2.0e-5, msg=label)
                test.assertEqual(solver.restitution_velocity_threshold, new)
                test.assertAlmostEqual(solver._rebound_velocity_threshold, new)

    # The enable switch is construction-only: an edited threshold leaves rebounds off.
    disabled, solver = _threshold_bounce(device, THRESHOLD_EDIT_ROUTES[0], 0.5, edited=0.0, enable_restitution=False)
    test.assertGreater(solver._rebound_velocity_threshold, 1.0e30)
    test.assertLess(disabled, 0.5)


def test_captured_graph_keeps_its_restitution_threshold_until_recaptured(test, device):
    """Replay a graph with its capture-time threshold; a recapture after the edit launches the new one."""
    for route in THRESHOLD_EDIT_ROUTES:
        label = route[0]
        with test.subTest(label):
            _label, solve_mode, scene, response, velocity_iterations = route
            model, body = _build_plane_model(device, separation=0.0, restitution=1.0, scene=scene)
            solver = _make_solver(
                model,
                solve_mode=solve_mode,
                response=response,
                velocity_iterations=velocity_iterations,
                enable_restitution=True,
                restitution_velocity_threshold=0.5,
            )
            state_in, state_out = model.state(), model.state()
            pipeline = newton.CollisionPipeline(model, broad_phase="nxn")
            contacts = pipeline.contacts()
            control = model.control()

            _reset_state(model, state_in, -1.0)
            pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, 0.01)
            with wp.ScopedCapture(device) as before_edit:
                pipeline.collide(state_in, contacts)
                solver.step(state_in, state_out, control, contacts, 0.01)
            solver.restitution_velocity_threshold = 2.0
            with wp.ScopedCapture(device) as after_edit:
                pipeline.collide(state_in, contacts)
                solver.step(state_in, state_out, control, contacts, 0.01)

            replays = {}
            for name, graph in (("before", before_edit.graph), ("after", after_edit.graph)):
                _reset_state(model, state_in, -1.0)
                wp.capture_launch(graph)
                replays[name] = float(state_out.body_qd.numpy()[body][2])
            fresh_old, _ = _threshold_bounce(device, route, 0.5)
            fresh_new, _ = _threshold_bounce(device, route, 2.0)
            test.assertGreater(abs(fresh_new - fresh_old), 0.5, f"{label}: thresholds do not discriminate")
            test.assertAlmostEqual(replays["before"], fresh_old, delta=2.0e-5, msg=f"{label}: capture-time graph")
            test.assertAlmostEqual(replays["after"], fresh_new, delta=2.0e-5, msg=f"{label}: recaptured graph")


devices_restitution = get_selected_cuda_test_devices()


class TestFeatherPGSRestitution(unittest.TestCase):
    pass


for _fn in (
    test_one_step_restitution_law_over_coefficient_speed_and_mass,
    test_crossing_result_is_invariant_to_dt_and_time_of_impact,
    test_penetrating_impact_replaces_baumgarte_with_restitution,
    test_restitution_uses_the_post_force_incident_predictor,
    test_non_crossing_speculative_contact_does_not_bounce,
    test_shape_restitution_uses_symmetric_arithmetic_average,
    test_restitution_velocity_threshold_uses_incident_relative_speed,
    test_bouncing_ball_settles_to_rest_under_gravity,
    test_two_body_impact_conserves_momentum_and_reverses_relative_speed,
    test_restitution_is_relative_to_a_moving_kinematic_surface,
    test_every_contact_response_route_enforces_restitution,
    test_dense_and_production_split_modes_enforce_restitution_without_velocity_iterations,
    test_multiworld_restitution_keeps_case_data_isolated,
    test_restitution_uses_ordinary_rows_and_validates_configuration,
    test_restitution_can_be_disabled_without_changing_materials,
    test_velocity_iteration_counts_reach_the_same_single_row_solution,
    test_warm_start_history_does_not_change_or_repeat_restitution,
    test_redundant_box_contacts_do_not_multiply_restitution_energy,
    test_articulated_redundant_manifold_preserves_symmetric_rebound,
    test_cuda_graph_replay_reads_current_restitution_and_matches_eager,
    test_notify_reads_replaced_and_in_place_restitution,
):
    add_function_test(TestFeatherPGSRestitution, _fn.__name__, _fn, devices=devices_restitution)


add_function_test(
    TestFeatherPGSRestitution,
    "test_edited_restitution_threshold_matches_a_fresh_solver",
    test_edited_restitution_threshold_matches_a_fresh_solver,
    devices=get_test_devices(),
)
add_function_test(
    TestFeatherPGSRestitution,
    "test_captured_graph_keeps_its_restitution_threshold_until_recaptured",
    test_captured_graph_keeps_its_restitution_threshold_until_recaptured,
    devices=get_selected_cuda_test_devices(),
)
add_function_test(
    TestFeatherPGSRestitution,
    "test_notify_reads_replaced_and_in_place_restitution_split",
    test_notify_reads_replaced_and_in_place_restitution,
    devices=get_test_devices(),
    solve_mode="split",
)


SIM_DT = 1.0 / 2000.0


DROP = 0.12


# The route each scene and response mode takes, measured rather than assumed.
# A lone free body never reaches dense rows, and "propagation_fused" only uses
# propagation rows once a real articulation exists -- with a free body it
# degrades to the matrix-free path -- so a single-sphere scene cannot claim to
# cover either. Asserting the exact set keeps the suite from silently claiming
# coverage of a path it never reaches.
EXPECTED_PATHS = {
    ("free", "immediate"): {PATH_MATRIX_FREE},
    ("free", "propagation"): {PATH_PROPAGATION},
    ("free", "propagation_fused"): {PATH_MATRIX_FREE},
    ("free", "propagation_colored"): {PATH_PROPAGATION},
    ("articulated", "immediate"): {PATH_DENSE},
    ("articulated", "propagation"): {PATH_PROPAGATION},
    ("articulated", "propagation_fused"): {PATH_PROPAGATION},
    ("articulated", "propagation_colored"): {PATH_PROPAGATION},
}


def _build(device, scene, mass=1.0):
    """Free sphere, or a two-link articulation whose root carries the contact."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.5)
    if scene == "free":
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, RADIUS + DROP), wp.quat_identity()))
        builder.add_shape_sphere(body, radius=RADIUS, cfg=cfg)
    else:
        # add_link + explicit joints + add_articulation: add_body would create a
        # standalone free articulation per call, so appending a revolute to two
        # of them yields two articulations and an unowned loop joint rather than
        # the single two-link chain this fixture is meant to exercise.
        body = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, RADIUS + DROP), wp.quat_identity()), mass=mass)
        builder.add_shape_sphere(body, radius=RADIUS, cfg=cfg)
        link = builder.add_link(xform=wp.transform(wp.vec3(0.12, 0.0, RADIUS + DROP), wp.quat_identity()), mass=mass)
        builder.add_shape_sphere(link, radius=RADIUS, cfg=cfg)
        root_joint = builder.add_joint_free(child=body)
        hinge = builder.add_joint_revolute(
            parent=body,
            child=link,
            axis=(0.0, 1.0, 0.0),
            parent_xform=wp.transform(wp.vec3(0.12, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        )
        builder.add_articulation([root_joint, hinge])
    builder.add_ground_plane(cfg=cfg)
    return builder.finalize(device=device), body


def _drop(
    device,
    velocity_iterations,
    response="immediate",
    steps=620,
    scene="free",
    *,
    contact_warmstart=False,
    mf_warmstart=False,
):
    """Drop a body onto a plane; return heights, velocities, contact counts, routes."""
    model, body = _build(device, scene)
    solver = newton.solvers.SolverFeatherPGS(
        model,
        solve_mode="matrix_free",
        velocity_iterations=velocity_iterations,
        articulated_contact_response=response,
        contact_warmstart=contact_warmstart or mf_warmstart,
    )
    solver._rigid_body_angular_damping.zero_()
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    heights, velocities, counts, routes = [], [], [], set()
    pipeline = (
        newton.CollisionPipeline(model, broad_phase="nxn", contact_matching="latest")
        if contact_warmstart or mf_warmstart
        else newton.CollisionPipeline(model)
    )
    contacts = pipeline.contacts()
    for _ in range(steps):
        contacts.clear()
        pipeline.collide(state_0, contacts)
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts, SIM_DT)
        state_0, state_1 = state_1, state_0
        heights.append(float(state_0.body_q.numpy()[body][2]) - RADIUS)
        velocities.append(float(state_0.body_qd.numpy()[body][2]))
        n = int(contacts.rigid_contact_count.numpy()[0])
        counts.append(n)
        if n:
            routes.update(int(v) for v in solver._contact_path.numpy()[:n] if v >= 0)
    return np.asarray(heights), np.asarray(velocities), np.asarray(counts), routes


def _assert_landed(test, heights, velocities, counts, label):
    """Assert the body came to rest ON the plane, not through it and not above it."""
    final_h, final_v = float(heights[-1]), float(velocities[-1])
    # Bounded on BOTH sides: a tunnelling body has negative heights and would
    # satisfy any upper bound by itself.
    test.assertLess(abs(final_h), 0.005, f"{label}: rested at {final_h:+.4f} m instead of on the surface")
    test.assertLess(abs(final_v), 0.02, f"{label}: still moving at {final_v:+.4f} m/s")
    test.assertGreater(float(heights[:80].min()), 0.05, f"{label}: body was not still falling early in the run")
    test.assertGreater(int(counts.max()), 0, f"{label}: no contact was ever generated")
    settled = heights[-150:]
    test.assertLess(float(settled.max() - settled.min()), 0.002, f"{label}: resting height was not stable")


def _single_step(device, gap, approach_speed, velocity_iterations=4, mass=1.0):
    """One step of a body ``gap`` above the plane closing at ``approach_speed``.

    Returns the velocity before and after, the contact count, the position-pass
    impulse, the start gap, and the linearized end gap on positive-gap rows, so
    a test can confirm which side of the velocity-pass classifier it exercised.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))  # no gravity: isolate the constraint
    # density=0 so the body's mass is exactly the requested value: the point of
    # the light-body case is that the position impulse scales with it.
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.0, density=0.0)
    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, RADIUS + gap), wp.quat_identity()),
        mass=mass,
        inertia=wp.mat33(np.eye(3) * 0.4 * mass * RADIUS**2),
    )
    builder.add_shape_sphere(body, radius=RADIUS, cfg=cfg)
    builder.add_ground_plane(cfg=cfg)
    # The light-body case sits below the inertia floor on purpose.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Inertia validation corrected", category=UserWarning)
        model = builder.finalize(device=device)
    joint_qd = model.joint_qd.numpy()
    joint_qd[2] = -abs(approach_speed)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()

    def step(iterations):
        solver = newton.solvers.SolverFeatherPGS(model, solve_mode="matrix_free", velocity_iterations=iterations)
        solver._rigid_body_angular_damping.zero_()
        state_0, state_1 = model.state(), model.state()
        # A free body is driven through joint_qd; writing body_qd alone is discarded
        # by the forward kinematics the solver runs from joint state.
        state_0.joint_qd.assign(joint_qd)
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
        pipeline.collide(state_0, contacts)
        state_0.clear_forces()
        solver.step(state_0, state_1, model.control(), contacts, SIM_DT)
        return solver, state_0, state_1

    # A solve without velocity iterations leaves the position pass's rows, impulses and velocity in place.
    position, state_0, _ = step(0)
    _, _, state_1 = step(velocity_iterations)
    before = float(state_0.body_qd.numpy()[body][2])
    n = int(contacts.rigid_contact_count.numpy()[0])
    rows = int(position._mf_constraint_count.numpy()[0])
    phi = position._mf_phi.numpy()[0][:rows]
    lam = position._mf_impulses.numpy()[0][:rows]
    row_type = position._mf_row_type.numpy()[0][:rows]
    contact_rows = np.flatnonzero((row_type == 0) & (phi > 0.0))
    position_impulse = float(lam[contact_rows].max()) if contact_rows.size else 0.0
    max_phi = float(phi[contact_rows].max()) if contact_rows.size else 0.0
    v_position = position._v_out.numpy()
    J_a = position._mf_J_a.numpy()[0]
    J_b = position._mf_J_b.numpy()[0]
    dof_a = position._mf_dof_a.numpy()[0]
    dof_b = position._mf_dof_b.numpy()[0]
    world_dofs = position._world_dof_indices.numpy()[0]
    end_gaps = []
    for row in contact_rows:
        jv = 0.0
        if dof_a[row] >= 0:
            indices = world_dofs[dof_a[row] : dof_a[row] + 6]
            jv += float(np.dot(J_a[row][indices >= 0], v_position[indices[indices >= 0]]))
        if dof_b[row] >= 0:
            indices = world_dofs[dof_b[row] : dof_b[row] + 6]
            jv += float(np.dot(J_b[row][indices >= 0], v_position[indices[indices >= 0]]))
        end_gaps.append(float(phi[row] + SIM_DT * jv))
    max_end_gap = max(end_gaps, default=0.0)
    return before, float(state_1.body_qd.numpy()[body][2]), n, position_impulse, max_phi, max_end_gap


def test_rhs_classifies_exact_end_gap_without_impulse(test, device):
    """Treat exactly and numerically reached surfaces as active without consulting impulse."""
    count = wp.array([1], dtype=wp.int32, device=device)
    dof_a = wp.array([[0]], dtype=wp.int32, device=device)
    dof_b = wp.array([[-1]], dtype=wp.int32, device=device)
    J_a = wp.array([[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]], dtype=wp.float32, device=device)
    J_b = wp.zeros((1, 1, 6), dtype=wp.float32, device=device)
    world_dofs = wp.array([[0, 1, 2, 3, 4, 5]], dtype=wp.int32, device=device)
    phi = wp.array([[1.0]], dtype=wp.float32, device=device)
    row_type = wp.array([[0]], dtype=wp.int32, device=device)
    target_velocity = wp.zeros((1, 1), dtype=wp.float32, device=device)
    row_restitution = wp.zeros((1, 1), dtype=wp.float32, device=device)
    position_velocity = wp.zeros((6,), dtype=wp.float32, device=device)
    rhs = wp.zeros((1, 1), dtype=wp.float32, device=device)

    def classify(normal_velocity, target=0.0, has_target=0):
        position_velocity.assign(np.array([normal_velocity, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32))
        target_velocity.assign(np.array([[target]], dtype=np.float32))
        wp.launch(
            compute_mf_rhs_bias,
            dim=1,
            inputs=[
                count,
                dof_a,
                dof_b,
                J_a,
                J_b,
                world_dofs,
                phi,
                row_type,
                target_velocity,
                row_restitution,
                has_target,
                0.5,
                position_velocity,
                position_velocity,
                0.5,
                1,
            ],
            outputs=[rhs],
            device=device,
        )
        return float(rhs.numpy()[0, 0])

    outside_velocity = np.float32(-1.999997)
    outside_end_gap = np.float32(1.0) + np.float32(0.5) * outside_velocity
    near_velocity = np.float32(-1.999999)
    near_end_gap = np.float32(1.0) + np.float32(0.5) * near_velocity

    test.assertGreater(float(outside_end_gap), 1.0e-6, "outside test case did not clear the activation slop")
    test.assertGreater(float(near_end_gap), 0.0, "near-boundary test case did not retain a positive residual")
    test.assertLessEqual(float(near_end_gap), 1.0e-6, "near-boundary test case exceeded the activation slop")

    test.assertEqual(classify(-1.0), 2.0, "a row ending 0.5 m away lost its speculative allowance")
    test.assertEqual(
        classify(float(outside_velocity)),
        2.0,
        "a row ending beyond the numerical boundary lost its speculative allowance",
    )
    test.assertEqual(
        classify(float(near_velocity)),
        0.0,
        "a binding row with a small positive residual retained its speculative allowance",
    )
    test.assertEqual(classify(-2.0), 0.0, "a row reaching the surface retained its speculative allowance")
    test.assertEqual(
        classify(-1.0, target=1.0, has_target=1),
        -1.0,
        "prescribed target velocity was omitted or used with the wrong sign",
    )


def test_velocity_iterations_do_not_halt_a_falling_body(test, device):
    """Land a falling body on the plane with the velocity pass enabled."""
    for iterations in (2, 8):
        h, v, n, _r = _drop(device, iterations)
        _assert_landed(test, h, v, n, f"velocity_iterations={iterations}")


def test_velocity_iterations_match_the_position_only_landing(test, device):
    """Land at the same height with and without velocity iterations."""
    baseline, base_v, base_n, _r0 = _drop(device, 0)
    with_velocity, vel_v, vel_n, _r4 = _drop(device, 4)
    _assert_landed(test, baseline, base_v, base_n, "velocity_iterations=0")
    _assert_landed(test, with_velocity, vel_v, vel_n, "velocity_iterations=4")
    test.assertAlmostEqual(
        float(with_velocity[-1]),
        float(baseline[-1]),
        delta=0.002,
        msg=(
            f"resting height moved from {float(baseline[-1]):.4f} m to "
            f"{float(with_velocity[-1]):.4f} m when velocity iterations were enabled"
        ),
    )


def test_every_contact_response_route_lands(test, device):
    """Land correctly on every internal contact-response route.

    Asserts the exact ``contact_path`` each combination took, so the suite
    cannot claim coverage of a route the scene never reaches.
    """
    for scene in ("free", "articulated"):
        for response in RESPONSE_MODES:
            label = f"{scene}/{response}"
            h, v, n, routes = _drop(device, 4, response=response, scene=scene)
            _assert_landed(test, h, v, n, label)
            test.assertEqual(
                routes,
                EXPECTED_PATHS[(scene, response)],
                f"{label}: exercised contact_path {sorted(routes)}, "
                f"expected {sorted(EXPECTED_PATHS[(scene, response)])}",
            )


def test_non_crossing_row_keeps_its_speculative_allowance(test, device):
    """Leave the velocity untouched when the body cannot reach the surface.

    With ``phi + h*u > 0`` the position trajectory remains separated. If
    the velocity pass dropped its ``phi/h`` allowance it would forbid any
    approach and brake a body still far from contact.
    """
    gap = 0.02
    before, after, n, impulse, phi, end_gap = _single_step(device, gap, 0.5 * gap / SIM_DT)
    test.assertLess(before, -1.0, "setup failed to give the body an approach velocity")
    test.assertGreater(n, 0, "no speculative contact was generated, so the branch was never reached")
    test.assertGreater(phi, 0.0, "the row under test was not a positive-gap speculative contact")
    test.assertEqual(impulse, 0.0, f"a non-crossing row took a position impulse of {impulse:.3e}")
    test.assertGreater(end_gap, 0.0, f"non-crossing row ended at gap {end_gap:+.3e} m")
    test.assertAlmostEqual(
        after, before, delta=0.02 * abs(before), msg=f"free approach was braked: {before:.3f} -> {after:.3f} m/s"
    )


def test_crossing_row_loses_its_speculative_allowance(test, device):
    """Arrest a body that would cross the surface within the step.

    A blanket "always retain phi/h" implementation leaves it closing at the gap
    rate instead, which this bound catches.
    """
    gap = 0.002
    before, after, n, impulse, phi, end_gap = _single_step(device, gap, 5.0 * gap / SIM_DT)
    test.assertLess(before, -1.0, "setup failed to give the body an approach velocity")
    test.assertGreater(n, 0, "no contact was generated")
    test.assertGreater(phi, 0.0, "the row under test was not a positive-gap speculative contact")
    test.assertGreater(impulse, 0.0, "a crossing row took no position impulse")
    test.assertLessEqual(end_gap, 1.0e-6, f"crossing row still ended at gap {end_gap:+.3e} m")
    # Magnitude, not a signed bound: "after <= 0.05*|before|" is satisfied by any
    # negative value, so a row that merely slowed from -20 to -4 m/s would pass.
    test.assertLess(
        abs(after),
        0.05 * abs(before),
        f"crossing contact kept approaching: {before:.3f} -> {after:.3f} m/s",
    )


def test_light_body_crossing_uses_end_gap_not_impulse_scale(test, device):
    """Arrest a very light crossing body independently of impulse scale.

    The impulse is effective mass times the velocity change, so it can be made
    arbitrarily small without changing the physics of the impact. At this mass it
    is ~2e-11, below the 1e-9 absolute threshold an impulse classifier once
    used. End-gap classification is invariant to that mass scaling. The mass is
    deliberately unphysical: the point is the classifier, not the scenario.
    """
    gap = 0.002
    before, after, n, impulse, phi, end_gap = _single_step(device, gap, 5.0 * gap / SIM_DT, mass=1.0e-12)
    test.assertGreater(n, 0, "no contact was generated")
    test.assertGreater(phi, 0.0, "the row under test was not a positive-gap speculative contact")
    # Straddles the 1e-9 cutoff this implementation once used: positive, so the
    # row reaches the surface, yet small enough that an absolute threshold misses it.
    test.assertGreater(impulse, 0.0, "light crossing row took no position impulse")
    test.assertLess(impulse, 1.0e-9, f"position impulse {impulse:.3e} does not straddle the old 1e-9 cutoff")
    test.assertLessEqual(end_gap, 1.0e-6, f"light crossing row still ended at gap {end_gap:+.3e} m")
    test.assertLess(
        abs(after),
        0.05 * abs(before),
        f"light crossing body kept approaching: {before:.3f} -> {after:.3f} m/s (position impulse {impulse:.3e})",
    )


def test_multiworld_end_gap_uses_each_world_position_velocity(test, device):
    """Classify speculative rows from the corresponding world's velocity."""
    gap = 0.002
    template = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.0)
    body = template.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, RADIUS + gap), wp.quat_identity()))
    template.add_shape_sphere(body, radius=RADIUS, cfg=cfg)
    template.add_ground_plane(cfg=cfg)
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.replicate(template, 2)
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverFeatherPGS(
        model,
        solve_mode="matrix_free",
        velocity_iterations=4,
    )
    solver._rigid_body_angular_damping.zero_()
    state_0, state_1 = model.state(), model.state()
    joint_qd = state_0.joint_qd.numpy().reshape(2, 6)
    joint_qd[0, 2] = -0.5 * gap / SIM_DT
    joint_qd[1, 2] = -5.0 * gap / SIM_DT
    state_0.joint_qd.assign(joint_qd.reshape(-1))
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(state_0, contacts)
    state_0.clear_forces()
    solver.step(state_0, state_1, model.control(), contacts, SIM_DT)
    after = state_1.body_qd.numpy()[:, 2]
    test.assertAlmostEqual(
        float(after[0]),
        float(joint_qd[0, 2]),
        delta=0.02 * abs(float(joint_qd[0, 2])),
        msg=f"non-crossing world was braked to {after[0]:.3f} m/s",
    )
    test.assertLess(
        abs(float(after[1])),
        0.05 * abs(float(joint_qd[1, 2])),
        f"crossing world kept approaching at {after[1]:.3f} m/s",
    )


def test_warm_start_with_velocity_iterations_is_supported(test, device):
    """Allow every warm-start entry point with end-gap classification."""
    model, _body = _build(device, "free")

    def make(**kwargs):
        return newton.solvers.SolverFeatherPGS(model, solve_mode="matrix_free", velocity_iterations=4, **kwargs)

    make(contact_warmstart=True)

    # Exercise dense slot warm starting and MF identity warm starting through
    # the same velocity pass, not just their constructor paths.
    h, v, n, routes = _drop(
        device,
        4,
        response="immediate",
        scene="articulated",
        contact_warmstart=True,
    )
    _assert_landed(test, h, v, n, "dense warm start")
    test.assertEqual(routes, {PATH_DENSE})

    h, v, n, routes = _drop(device, 4, mf_warmstart=True)
    _assert_landed(test, h, v, n, "matrix-free identity warm start")
    test.assertEqual(routes, {PATH_MATRIX_FREE})


devices_velocity_pass_speculative = get_selected_cuda_test_devices()


class TestFeatherPGSVelocityPassSpeculative(unittest.TestCase):
    pass


for _fn in (
    test_rhs_classifies_exact_end_gap_without_impulse,
    test_velocity_iterations_do_not_halt_a_falling_body,
    test_velocity_iterations_match_the_position_only_landing,
    test_every_contact_response_route_lands,
    test_non_crossing_row_keeps_its_speculative_allowance,
    test_crossing_row_loses_its_speculative_allowance,
    test_light_body_crossing_uses_end_gap_not_impulse_scale,
    test_multiworld_end_gap_uses_each_world_position_velocity,
    test_warm_start_with_velocity_iterations_is_supported,
):
    add_function_test(
        TestFeatherPGSVelocityPassSpeculative,
        _fn.__name__,
        _fn,
        devices=devices_velocity_pass_speculative,
    )


def _build_mixed_response_model(
    device,
    world_count=1,
    *,
    dof_count=13,
    friction=0.0,
    restitution=0.0,
    static_plane=False,
    static_support=False,
    free_body_first=False,
    free_body_velocity_limit=None,
):
    """Build one serial articulation contacting one free rigid body.

    ``static_support`` rests the free body on a static ledge that clears the arm, so the free body also
    produces matrix-free contact rows. ``free_body_first`` builds the free body before the articulation,
    which packs the world DOFs free-body first. ``free_body_velocity_limit`` caps the free body's linear
    and angular velocity [m/s, rad/s], producing matrix-free velocity-limit rows once exceeded.
    ``static_plane`` adds a ground plane under the free body.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    if free_body_velocity_limit is not None:
        SolverFeatherPGS.register_custom_attributes(builder)
    builder.default_shape_cfg.density = 1000.0
    builder.default_shape_cfg.ke = 1.0e5
    builder.default_shape_cfg.kd = 1.0e3
    builder.default_shape_cfg.mu = friction
    builder.default_shape_cfg.restitution = restitution
    builder.default_shape_cfg.margin = 0.0
    builder.default_shape_cfg.gap = 0.0

    def add_free_body():
        custom_attributes = None
        if free_body_velocity_limit is not None:
            custom_attributes = {
                "feather_pgs:rigid_body_max_linear_velocity": free_body_velocity_limit,
                "feather_pgs:rigid_body_max_angular_velocity": free_body_velocity_limit,
            }
        box = builder.add_link(
            xform=wp.transform(wp.vec3(0.7, 0.0, 0.5695), wp.quat_identity()), custom_attributes=custom_attributes
        )
        builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.05)
        builder.add_articulation([builder.add_joint_free(parent=-1, child=box)])
        if static_support:
            builder.add_shape_box(
                -1,
                xform=wp.transform(wp.vec3(0.7, 0.11, 0.47), wp.quat_identity()),
                hx=0.05,
                hy=0.05,
                hz=0.05,
            )

    if free_body_first:
        add_free_body()
    arm = builder.add_link()
    builder.add_shape_box(arm, hx=0.4, hy=0.05, hz=0.02)
    joints = [
        builder.add_joint_revolute(
            parent=-1,
            child=arm,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.4, 0.0, 0.0), wp.quat_identity()),
        )
    ]
    parent = arm
    for index in range(dof_count - 1):
        child = builder.add_link(
            mass=0.05,
            inertia=wp.mat33(np.eye(3, dtype=np.float32) * 1.0e-3),
            lock_inertia=True,
        )
        joints.append(
            builder.add_joint_revolute(
                parent=parent,
                child=child,
                axis=(newton.Axis.X, newton.Axis.Y, newton.Axis.Z)[index % 3],
            )
        )
        parent = child
    builder.add_articulation(joints)
    builder.set_joint_mimic(joints[1], joints[0])

    if not free_body_first:
        add_free_body()
    if static_plane:
        builder.add_shape_plane(plane=(0.0, 0.0, 1.0, -0.53))
    if world_count == 1:
        return builder.finalize(device=device)
    replicated = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    replicated.replicate(builder, world_count, spacing=(3.0, 0.0, 0.0))
    return replicated.finalize(device=device)


def _run_mixed_response(
    kernel,
    *,
    warmstart,
    preelimination,
    dof_count=13,
    articulation_constraint_max=32,
    inactive_joint_limit_capacity=False,
    friction=0.0,
    restitution=0.0,
    tangential_velocity=0.0,
    static_plane=False,
    contact_regularization=0.0,
    friction_anchor_beta=None,
    model_kwargs=None,
):
    """Run a short mixed-contact trajectory with one H-inverse implementation."""
    model = _build_mixed_response_model(
        "cuda:0",
        dof_count=dof_count,
        friction=friction,
        restitution=restitution,
        static_plane=static_plane,
        **(model_kwargs or {}),
    )
    with kernel_overrides({"hinv_jt_kernel": kernel}):
        solver = SolverFeatherPGS(
            model,
            solve_mode="matrix_free",
            contact_warmstart=warmstart,
            enable_bilateral_preelimination=preelimination,
            enable_joint_limits=inactive_joint_limit_capacity,
            joint_limit_activation_gap=0.0,
            iterations=8,
            contact_regularization=contact_regularization,
            friction_anchor_beta=friction_anchor_beta,
            articulation_constraint_max=articulation_constraint_max,
            free_body_constraint_max=32,
        )
    state_in, state_out = model.state(), model.state()
    joint_qd = state_in.joint_qd.numpy()
    free_articulation = int(np.flatnonzero(solver._model_plan.is_free_rigid)[0])
    free_dof_start = int(solver._model_plan.articulation_dof_start[free_articulation])
    joint_qd[free_dof_start] = tangential_velocity
    joint_qd[free_dof_start + 2] = -3.0
    state_in.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)

    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        reduce_contacts=False,
        contact_matching="latest",
    )
    contacts = pipeline.contacts()
    control = model.control()
    samples = []
    for _ in range(4):
        state_in.clear_forces()
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, 1.0 / 240.0)
        constraint_count = int(solver._constraint_count.numpy()[0])
        mf_count = int(solver._mf_constraint_count.numpy()[0])
        samples.append(
            (
                constraint_count,
                solver._diag.numpy()[0, :constraint_count].copy(),
                solver._impulses.numpy()[0, :constraint_count].copy(),
                state_out.joint_q.numpy().copy(),
                state_out.joint_qd.numpy().copy(),
                0,
                solver._row_type.numpy()[0, :constraint_count].copy(),
                mf_count,
                solver._mf_impulses.numpy()[0, :mf_count].copy(),
                solver._mf_row_type.numpy()[0, :mf_count].copy(),
                solver._mf_row_parent.numpy()[0, :mf_count].copy(),
                solver._mf_J_a.numpy()[0, :mf_count].copy(),
                solver._mf_J_b.numpy()[0, :mf_count].copy(),
                solver._mf_MiJt_a.numpy()[0, :mf_count].copy(),
                solver._mf_MiJt_b.numpy()[0, :mf_count].copy(),
            )
        )
        state_in, state_out = state_out, state_in
    return solver, samples


class TestFeatherPGSPairedResponse(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "paired response ownership requires CUDA")
    def test_paired_response_without_factor_coordinates_matches_general(self):
        """Match the general response with the explicit inverses that velocity limits keep in physical coordinates."""
        run_kwargs = {
            "warmstart": False,
            "preelimination": False,
            "dof_count": 23,
            "articulation_constraint_max": 96,
            "inactive_joint_limit_capacity": True,
            "friction": 0.7,
            "restitution": 0.3,
            "tangential_velocity": 2.0,
            "friction_anchor_beta": 0.0,
            "contact_regularization": 0.01,
        }
        reference_solver, reference = _run_mixed_response("par_row", **run_kwargs)
        paired_solver, paired = _run_mixed_response("auto", **run_kwargs)
        self.assertIsNone(reference_solver._paired_response_primary_size)
        self.assertEqual(paired_solver._paired_response_primary_size, 23)
        self.assertFalse(paired_solver._paired_factor_coordinates)
        self.assertIsNone(paired_solver._paired_factor_solve_kernel)
        self.assertIsNotNone(paired_solver._paired_cholesky_inverse_kernel)
        for step, (expected, actual) in enumerate(zip(reference, paired, strict=True)):
            self.assertEqual(actual[0], expected[0], f"constraint count differed at step {step}")
            for label, expected_value, actual_value in zip(
                ("diagonal", "impulses", "joint_q", "joint_qd"), expected[1:5], actual[1:5], strict=True
            ):
                np.testing.assert_allclose(
                    actual_value, expected_value, rtol=5.0e-4, atol=1.0e-5, err_msg=f"{label} differed at step {step}"
                )


if __name__ == "__main__":
    unittest.main()
