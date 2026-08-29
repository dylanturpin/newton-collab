# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for FeatherPGS speculative-contact controls."""

import inspect
import unittest

import warp as wp

import newton
from newton._src.solvers.feather_pgs.kernels import (
    allocate_world_contact_slots,
    apply_world_contact_restitution_accumulated,
    compute_mf_effective_mass_and_rhs,
    compute_propagation_effective_mass_and_rhs,
    compute_world_contact_bias,
)
from newton.solvers import SolverFeatherPGS

PATH_DENSE = 0
PATH_MATRIX_FREE = 1
PATH_PROPAGATION = 2


def _launch_contact_allocator(
    *, route: int, gap: float, gate: float, scoped_gate: float = 0.0, pair_gate: float = 0.0
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
    }
    is_free = route == PATH_MATRIX_FREE
    propagation_enabled = route == PATH_PROPAGATION
    wp.launch(
        allocate_world_contact_slots,
        dim=1,
        inputs=[
            wp.array([1], dtype=wp.int32, device=device),
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
            wp.array([int(is_free)], dtype=wp.int32, device=device),
            1,
            int(propagation_enabled),
            0,
            0,
            gate,
            scoped_gate,
            pair_gate,
            8,
            8,
            8,
            0,
            float("inf"),
            0,
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
        ],
        device=device,
    )
    wp.synchronize_device(device)
    return {name: int(array.numpy()[0]) for name, array in (outputs | counters).items()}


def _launch_same_articulation_contact_allocator(
    *, gap: float, scoped_gate: float, pair_gate: float = 0.0
):
    """Allocate one two-link contact from a non-free articulation."""
    device = "cpu"
    contact_slot = wp.full((1,), -9, dtype=wp.int32, device=device)
    contact_path = wp.full((1,), -9, dtype=wp.int32, device=device)
    dense_count = wp.zeros((1,), dtype=wp.int32, device=device)
    wp.launch(
        allocate_world_contact_slots,
        dim=1,
        inputs=[
            wp.array([1], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([1], dtype=wp.int32, device=device),
            wp.array([wp.vec3(gap, 0.0, 0.0)], dtype=wp.vec3, device=device),
            wp.array([wp.vec3(0.0)], dtype=wp.vec3, device=device),
            wp.array([wp.vec3(-1.0, 0.0, 0.0)], dtype=wp.vec3, device=device),
            wp.zeros((1,), dtype=wp.float32, device=device),
            wp.zeros((1,), dtype=wp.float32, device=device),
            wp.array([wp.transform_identity(), wp.transform_identity()], dtype=wp.transform, device=device),
            wp.array([wp.transform_identity(), wp.transform_identity()], dtype=wp.transform, device=device),
            wp.array([0, 1], dtype=wp.int32, device=device),
            wp.array([0, 0], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([1], dtype=wp.int32, device=device),
            wp.zeros((2,), dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            0,
            0,
            0,
            0,
            0.0,
            scoped_gate,
            pair_gate,
            8,
            8,
            8,
            0,
            float("inf"),
            0,
            0,
        ],
        outputs=[
            wp.full((1,), -9, dtype=wp.int32, device=device),
            contact_slot,
            wp.full((1,), -9, dtype=wp.int32, device=device),
            wp.full((1,), -9, dtype=wp.int32, device=device),
            dense_count,
            contact_path,
            wp.zeros((1,), dtype=wp.int32, device=device),
            wp.zeros((1,), dtype=wp.int32, device=device),
            wp.zeros((1,), dtype=wp.int32, device=device),
            wp.full((1,), -9, dtype=wp.int32, device=device),
            wp.zeros((1,), dtype=wp.int32, device=device),
            wp.zeros((1,), dtype=wp.int32, device=device),
            wp.zeros((1,), dtype=wp.int32, device=device),
        ],
        device=device,
    )
    wp.synchronize_device(device)
    return int(contact_slot.numpy()[0]), int(contact_path.numpy()[0]), int(dense_count.numpy()[0])


def _dense_speculative_rhs(scale: float) -> float:
    """Return the dense positive-gap RHS for a requested speculative scale."""
    rhs = wp.zeros((1, 1), dtype=wp.float32, device="cpu")
    wp.launch(
        compute_world_contact_bias,
        dim=1,
        inputs=[
            wp.array([1], dtype=wp.int32, device="cpu"),
            1,
            wp.array([[1.0]], dtype=wp.float32, device="cpu"),
            wp.array([[0.2]], dtype=wp.float32, device="cpu"),
            wp.array([[0]], dtype=wp.int32, device="cpu"),
            wp.zeros((1, 1), dtype=wp.float32, device="cpu"),
            0.5,
            1.0,
            scale,
            1.0,
        ],
        outputs=[rhs],
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
            wp.array([[1.0]], dtype=wp.float32, device="cpu"),
            wp.array([[0]], dtype=wp.int32, device="cpu"),
            wp.zeros((1, 1), dtype=wp.float32, device="cpu"),
            wp.zeros((1, 6), dtype=wp.float32, device="cpu"),
            wp.array([float("inf")], dtype=wp.float32, device="cpu"),
            1.0,
            0.2,
            0.0,
            0.0,
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
        ],
        outputs=[rhs],
        device="cpu",
    )
    return float(rhs.numpy()[0, 0])


class TestFeatherPGSContactControls(unittest.TestCase):
    def test_solver_exposes_documented_defaults(self):
        """Expose legacy-preserving defaults through the public constructor."""
        solver = SolverFeatherPGS(newton.ModelBuilder().finalize(device="cpu"))

        self.assertEqual(solver.contact_speculative_scale, 1.0)
        self.assertEqual(solver.contact_gap_gate, 0.0)
        self.assertEqual(solver.same_articulation_contact_gap_gate, 0.0)
        self.assertEqual(solver.articulation_pair_contact_gap_gate, 0.0)
        parameters = tuple(inspect.signature(SolverFeatherPGS).parameters)
        self.assertIn("same_articulation_contact_gap_gate", parameters)

    def test_solver_validates_and_stores_contact_controls(self):
        """Accept finite non-negative controls and reject malformed values."""
        model = newton.ModelBuilder().finalize(device="cpu")
        solver = SolverFeatherPGS(
            model,
            contact_speculative_scale=0.0,
            contact_gap_gate=0.001,
            same_articulation_contact_gap_gate=0.002,
            articulation_pair_contact_gap_gate=0.003,
        )
        self.assertEqual(solver.contact_speculative_scale, 0.0)
        self.assertEqual(solver.contact_gap_gate, 0.001)
        self.assertEqual(solver.same_articulation_contact_gap_gate, 0.002)
        self.assertEqual(solver.articulation_pair_contact_gap_gate, 0.003)

        for name in (
            "contact_speculative_scale",
            "contact_gap_gate",
            "same_articulation_contact_gap_gate",
            "articulation_pair_contact_gap_gate",
        ):
            for value in (-0.1, float("nan"), float("inf"), "invalid"):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        SolverFeatherPGS(model, **{name: value})

    def test_scoped_gap_gate_only_drops_distant_same_articulation_contact(self):
        """The scoped gate retains near self-contact while bounding its speculative tail."""
        self.assertEqual(
            _launch_same_articulation_contact_allocator(gap=0.002, scoped_gate=0.003),
            (0, PATH_DENSE, 1),
        )
        self.assertEqual(
            _launch_same_articulation_contact_allocator(gap=0.004, scoped_gate=0.003),
            (-1, -1, 0),
        )
        self.assertEqual(
            _launch_same_articulation_contact_allocator(gap=0.004, scoped_gate=0.0),
            (0, PATH_DENSE, 1),
        )

    def test_scoped_gap_gate_preserves_other_contact_routes(self):
        """Do not shorten predictive contacts for ground, free-body, or cross-articulation rows."""
        for route, counter in (
            (PATH_DENSE, "dense_count"),
            (PATH_MATRIX_FREE, "mf_count"),
            (PATH_PROPAGATION, "propagation_count"),
        ):
            with self.subTest(route=route):
                result = _launch_contact_allocator(
                    route=route,
                    gap=0.04,
                    gate=0.0,
                    scoped_gate=0.003,
                )
                self.assertEqual(result["path"], route)
                self.assertEqual(result[counter], 1)

    def test_articulation_pair_gap_gate_drops_distant_pair_contact(self):
        """The pair gate includes same-articulation contact without touching free-body routes."""
        self.assertEqual(
            _launch_same_articulation_contact_allocator(
                gap=0.004,
                scoped_gate=0.0,
                pair_gate=0.003,
            ),
            (-1, -1, 0),
        )
        result = _launch_contact_allocator(
            route=PATH_MATRIX_FREE,
            gap=0.04,
            gate=0.0,
            pair_gate=0.003,
        )
        self.assertEqual(result["path"], PATH_MATRIX_FREE)
        self.assertEqual(result["mf_count"], 1)

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
        """Drop positive gaps above the gate before any route reserves a slot."""
        for route in (PATH_DENSE, PATH_MATRIX_FREE, PATH_PROPAGATION):
            with self.subTest(route=route):
                allocated = _launch_contact_allocator(route=route, gap=0.002, gate=0.0)
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
