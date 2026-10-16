# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact torsion: per-shape spin friction, host and device preparation of SolverFeatherPGS."""

import itertools
import unittest
import warnings
from inspect import signature
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton import GeoType
from newton._src.solvers.feather_pgs import contact_torsion, solver_feather_pgs
from newton._src.solvers.feather_pgs import contact_torsion as host
from newton._src.solvers.feather_pgs.contact_torsion import _TORSION_SPIN_RETIRED, _contact_groups, _group_budget
from newton._src.solvers.feather_pgs.contact_torsion_device import (
    DeviceTorsionPreparation,
    _host_dot3,
    _host_transform_point,
    enable_device_torsion,
)
from newton._src.solvers.feather_pgs.friction_patches import link_patch_rows
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    PGS_CONSTRAINT_TYPE_JOINT_TARGET,
    PGS_CONSTRAINT_TYPE_TORSION,
)
from newton._src.solvers.feather_pgs.solver_feather_pgs import (
    _estimate_mfgs_shared_memory,
    _get_pgs_solve_mf_gs_kernel,
    _get_pgs_solve_propagation_contact_kernel,
    _use_mfgs_pair_cache,
    _use_mfgs_torsion_links,
)
from newton.solvers import SolverBase, SolverFeatherPGS
from newton.tests.feather_pgs._fixtures import (
    _build_four_bar,
    _loop_anchor_gap,
    _patch_fixture,
    fixture,
    set_torsion_radius,
)

# Persistent patches select their own friction locations; shared-anchor flags only warn.
PATCH_OPTIONS = {"friction_anchor_beta": 0.2, "contact_shared_anchor": False, "contact_friction_shared_anchor": False}


def held_box(radius, torque, *, steps=20, dt=0.0025, mu=0.5, mass=0.3, half=0.005):
    """Rest a small articulated box on the ground under gravity and a yaw torque with patch friction.

    The square footprint keeps the two patch anchors within ``sqrt(2) * half`` of the
    center, so their own twist capacity stays far below ``mu * N * radius``.
    """
    b = newton.ModelBuilder()
    b.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=mu, restitution=0))
    pose = wp.transform(wp.vec3(0, 0, 0.02), wp.quat_identity())
    body = b.add_link(xform=pose, mass=mass, inertia=wp.mat33(np.diag([3e-4, 3e-4, 3e-4]).astype(np.float32)))
    axes = [newton.ModelBuilder.JointDofConfig(axis=wp.vec3(*a)) for a in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
    joint = b.add_joint_d6(-1, body, parent_xform=pose, linear_axes=axes, angular_axes=axes)
    b.add_articulation([joint])
    b.add_shape_box(
        body, hx=half, hy=half, hz=0.02, cfg=newton.ModelBuilder.ShapeConfig(density=0, mu=mu, restitution=0)
    )
    model = b.finalize(device="cuda:0")
    set_torsion_radius(model, radius)
    solver = SolverFeatherPGS(
        model,
        solve_mode="matrix_free",
        articulated_contact_response="immediate",
        iterations=64,
        velocity_iterations=0,
        baumgarte=0.05,
        cfm=0,
        contact_regularization=0,
        contact_warmstart=False,
        articulation_constraint_max=64,
        free_body_constraint_max=16,
        **PATCH_OPTIONS,
    )
    solver._rigid_body_angular_damping.zero_()
    pipeline = newton.CollisionPipeline(
        model, contact_matching="latest", reduce_contacts=False, broad_phase="nxn", rigid_contact_max=64
    )
    contacts = pipeline.contacts()
    s0, s1 = model.state(), model.state()
    control = model.control()
    force = np.zeros(model.joint_dof_count, np.float32)
    force[5] = torque
    control.joint_f.assign(force)
    for _ in range(steps):
        s0.clear_forces()
        pipeline.collide(s0, contacts)
        solver.step(s0, s1, control, contacts, dt)
        s0, s1 = s1, s0
    return float(s0.body_qd.numpy()[0, 5]), solver


def patch_rows(points, *, slots, slots_needed, row_type, parents, mu, patches_enabled=True, shape_type=GeoType.BOX):
    """Link one CPU patch region into dense rows and return the torsion host view of it.

    Rows follow the dense builder's layout: every contact keeps its normal, and only
    anchors receive an adjacent tangent pair. ``link_patch_rows`` then closes the
    normal-parent load ring and divides the anchors' friction coefficient.
    """
    model, state, contacts, patches = _patch_fixture(points)
    model.shape_type = wp.full(3, int(shape_type), dtype=int, device="cpu")
    contacts.rigid_contact_max = len(points)
    contacts.rigid_contact_stiffness = None
    n = len(points)
    solver = SimpleNamespace(
        model=model,
        _contact_path=wp.zeros(n, dtype=int, device="cpu"),
        _contact_slot=wp.array(slots, dtype=int, device="cpu"),
        _contact_world=wp.zeros(n, dtype=int, device="cpu"),
        _contact_slots_needed=wp.array(slots_needed, dtype=int, device="cpu"),
        _row_type=wp.array([row_type], dtype=int, device="cpu"),
        _row_parent=wp.array([parents], dtype=int, device="cpu"),
        _row_mu=wp.array([mu], dtype=float, device="cpu"),
        _constraint_count=wp.array([len(row_type)], dtype=int, device="cpu"),
        _friction_anchors_enabled=patches_enabled,
        _friction_patches=patches,
        _contact_torsion_shape_radius=wp.full(model.shape_body.shape[0], 0.01, dtype=float, device="cpu"),
    )
    if patches_enabled:
        wp.launch(
            link_patch_rows,
            dim=n,
            inputs=[
                contacts.rigid_contact_count,
                patches.view,
                solver._contact_world,
                solver._contact_slot,
                solver._contact_path,
                solver._contact_slots_needed,
                0,
                solver._row_parent,
                solver._row_mu,
            ],
            device="cpu",
        )
    return solver, state, contacts


class TestContactTorsionPatchGrouping(unittest.TestCase):
    """Follow persistent patch regions on the host without a CUDA solve."""

    def test_two_anchor_region_pools_the_undivided_coefficient(self):
        """Budget one region by the full coefficient on every ring normal, consuming only anchor rows."""
        solver, state, contacts = patch_rows(
            [[-0.1, 0, 0], [0, 0, 0], [0.1, 0, 0]],
            slots=[0, 3, 4],
            slots_needed=[3, 1, 3],
            row_type=[0, 2, 2, 0, 0, 2, 2],
            parents=[-1, 0, 0, -1, -1, 4, 4],
            mu=[0.5] * 7,
        )
        np.testing.assert_array_equal(solver._row_parent.numpy()[0], [3, 0, 0, 4, 0, 4, 4])
        np.testing.assert_allclose(solver._row_mu.numpy()[0], [0.5, 0.25, 0.25, 0.5, 0.5, 0.25, 0.25])
        groups = _contact_groups(solver, state, contacts)
        self.assertEqual(len(groups), 1)
        self.assertEqual(sorted(w.slot for w in groups[0]), [0, 3, 4])
        coefficient, anchors, touching = _group_budget(solver, groups[0], solver._row_mu.numpy())
        self.assertAlmostEqual(coefficient, 0.5)
        self.assertEqual(sorted(w.slot for w in anchors), [0, 4])
        self.assertEqual(len(touching), 3)

    def test_single_anchor_region_keeps_the_coefficient(self):
        """Pool coincident normals behind one anchor without scaling the coefficient."""
        solver, state, contacts = patch_rows(
            [[0, 0, 0], [0, 0, 0]],
            slots=[0, 3],
            slots_needed=[3, 1],
            row_type=[0, 2, 2, 0],
            parents=[-1, 0, 0, -1],
            mu=[0.5] * 4,
        )
        np.testing.assert_array_equal(solver._row_parent.numpy()[0], [3, 0, 0, 0])
        groups = _contact_groups(solver, state, contacts)
        self.assertEqual(len(groups), 1)
        self.assertEqual(sorted(w.slot for w in groups[0]), [0, 3])
        coefficient, anchors, _touching = _group_budget(solver, groups[0], solver._row_mu.numpy())
        self.assertAlmostEqual(coefficient, 0.5)
        self.assertEqual([w.slot for w in anchors], [0])

    def test_point_friction_grouping_is_unchanged(self):
        """Keep every point-friction witness on its own tangent pair with the row coefficient."""
        solver, state, contacts = patch_rows(
            [[0, 0, 0], [0, 0, 0]],
            slots=[0, 3],
            slots_needed=[3, 3],
            row_type=[0, 2, 2, 0, 2, 2],
            parents=[-1, 0, 0, -1, 3, 3],
            mu=[0.5] * 6,
            patches_enabled=False,
        )
        groups = _contact_groups(solver, state, contacts)
        self.assertEqual(len(groups), 1)
        coefficient, anchors, _touching = _group_budget(solver, groups[0], solver._row_mu.numpy())
        self.assertAlmostEqual(coefficient, 0.5)
        self.assertEqual(sorted(w.slot for w in anchors), [0, 3])

    def test_region_without_anchor_rows_carries_no_spin(self):
        """Skip a region whose members all lost their tangent rows to friction filters."""
        solver, state, contacts = patch_rows(
            [[-0.1, 0, 0], [0.1, 0, 0]],
            slots=[0, 1],
            slots_needed=[1, 1],
            row_type=[0, 0],
            parents=[-1, -1],
            mu=[0.5] * 2,
        )
        groups = _contact_groups(solver, state, contacts)
        self.assertEqual(len(groups), 1)
        self.assertIsNone(_group_budget(solver, groups[0], solver._row_mu.numpy()))

    def test_region_with_inadmissible_member_is_skipped(self):
        """Skip the whole region rather than budget a partial ring."""
        solver, state, contacts = patch_rows(
            [[-0.1, 0, 0], [0.1, 0, 0]],
            slots=[0, 3],
            slots_needed=[3, 3],
            row_type=[0, 2, 2, 0, 2, 2],
            parents=[-1, 0, 0, -1, 3, 3],
            mu=[0.5] * 6,
            shape_type=GeoType.MESH,
        )
        self.assertEqual(_contact_groups(solver, state, contacts), [])

    def test_broken_load_ring_is_rejected(self):
        """Refuse a region whose normal-parent ring does not close over its own rows."""
        solver, state, contacts = patch_rows(
            [[-0.1, 0, 0], [0.1, 0, 0]],
            slots=[0, 3],
            slots_needed=[3, 3],
            row_type=[0, 2, 2, 0, 2, 2],
            parents=[-1, 0, 0, -1, 3, 3],
            mu=[0.5] * 6,
        )
        parents = solver._row_parent.numpy()
        parents[0, 3] = -1
        solver._row_parent.assign(parents)
        with self.assertRaisesRegex(RuntimeError, "load ring"):
            _contact_groups(solver, state, contacts)


@unittest.skipUnless(wp.get_device().is_cuda, "Contact torsion currently requires CUDA")
class TestContactTorsion(unittest.TestCase):
    """Exercise an explicitly assumed uniform-disk effective spin radius."""

    def test_default_off_equivalent(self):
        """Keep a model without torsion radii exactly equivalent to the zero-radius output."""
        reference, *_ = fixture(None)
        actual, *_ = fixture(0.0)
        for key in reference:
            np.testing.assert_array_equal(actual[key], reference[key], err_msg=key)

    def test_spin_stops_below_bound(self):
        """Remove spin below the prescribed disk's Coulomb torque capacity."""
        baseline, *_ = fixture(0.0, spin=1.0, center_only=True)
        self.assertGreater(np.max(np.abs(baseline["body_qd"][:, 5])), 0.9)
        result, solver, *_ = fixture(0.01, spin=1.0, center_only=True)
        self.assertLess(np.max(np.abs(result["body_qd"][:, 5])), 1e-4)
        self.assertGreater(solver._torsion_stats["rows"], 0)

    def test_zero_friction_and_zero_load(self):
        """Apply no torsion when friction or compressive normal load vanishes."""
        for options in ({"mu": 0.0}, {"closing": 0.0}, {"separation": 0.004}):
            actual, solver, *_ = fixture(0.01, **options)
            if "closing" in options:
                self.assertGreater(solver._torsion_stats["rows"], 0)
            else:
                self.assertEqual(solver._torsion_stats["rows"], 0)
            active = actual["row_type"] == 7
            self.assertLess(np.abs(actual["impulses"][active]).max(initial=0), 1e-9)

    def test_joint_response_and_coupled_budget(self):
        """Respect Coulomb sharing, articulated response and kinetic-energy bounds."""
        for sat in (False, True):
            for spin, sliding in ((1.0, 0.0), (100.0, 0.0), (10.0, 1.0), (100.0, 10.0)):
                actual, solver, _model, initial, _ = fixture(
                    0.01, spin=spin, sliding=sliding, sat=sat, center_only=not sat
                )
                # Evaluate contact response at its input pose, before D6
                # integration changes the motion-basis readback. The existing
                # predictor already changes lateral momentum at extreme mixed
                # velocity; this gate isolates the contact solve from that predictor.
                v0 = actual["v_hat"].reshape(2, 6)
                v1 = actual["v_out"].reshape(2, 6)
                inertia = np.array([0.00012, 0.00012, 0.00008])

                def energy(v, inertia=inertia):
                    return float(0.5 * 0.3 * np.sum(v[:, :3] ** 2) + 0.5 * np.sum(v[:, 3:] ** 2 * inertia))

                self.assertLessEqual(energy(v1), energy(v0) + 2e-6)
                np.testing.assert_allclose(np.sum(v1[:, :3], axis=0), np.sum(v0[:, :3], axis=0), atol=1e-5)
                positions = initial.body_q.numpy()[:, :3]

                def angular(v, inertia=inertia, positions=positions):
                    return np.sum(v[:, 3:] * inertia + np.cross(positions, 0.3 * v[:, :3]), axis=0)

                np.testing.assert_allclose(angular(v1), angular(v0), atol=2e-6)
                count = int(actual["count"][0])
                impulse = actual["impulses"][0, :count]
                response = actual["Y_world"][0, :count].T @ impulse
                np.testing.assert_allclose(actual["v_out"] - actual["v_hat"], response, atol=2e-5)
                for group in solver._torsion_stats["groups"]:
                    rows = group["normal_rows"]
                    budget = 0.5 * sum(max(float(impulse[r]), 0) for r in rows)
                    used = sum(float(np.linalg.norm(impulse[r + 1 : r + 3])) for r in rows)
                    used += abs(float(impulse[group["row"]])) / 0.01
                    self.assertLessEqual(used, budget + 2e-6)

    def test_witness_count_does_not_multiply_torque(self):
        """Share one footprint budget across repeated normal quadrature witnesses."""
        outputs = []
        for count in (1, 2, 4):
            result, solver, *_ = fixture(0.01, spin=100.0, center_only=True, center_count=count)
            self.assertEqual(solver._torsion_stats["rows"], 1)
            outputs.append(result["body_qd"])
        for output in outputs[1:]:
            np.testing.assert_allclose(output, outputs[0], atol=2e-5)

    def test_release_and_reset_carry_no_torque(self):
        """Forget spin impulse on contact loss and reproduce cold state after reset."""
        expected, solver, model, initial, contacts = fixture(0.01, spin=100.0)
        output = model.state()
        count = contacts.rigid_contact_count.numpy()
        contacts.rigid_contact_count.zero_()
        solver.step(initial, output, model.control(), contacts, 0.0025)
        self.assertEqual(solver._torsion_stats["rows"], 0)
        contacts.rigid_contact_count.assign(count)
        solver.reset(initial)
        solver.step(initial, output, model.control(), contacts, 0.0025)
        np.testing.assert_allclose(output.body_qd.numpy(), expected["body_qd"], atol=2e-5)

    def test_timestep_and_capacity(self):
        """Keep impulse-level overload behavior across dt and reject insufficient rows."""
        velocities = []
        for dt in (0.00125, 0.0025, 0.005):
            result, solver, *_ = fixture(0.01, spin=100.0, center_only=True, dt=dt)
            velocities.append(result["v_out"])
            self.assertEqual(solver.constraint_row_watermarks()["articulation_dropped_contact_rows_high_water"], 0)
            self.assertLessEqual(int(solver._constraint_count.numpy().max()), solver._articulation_constraint_max)
        for velocity in velocities[1:]:
            np.testing.assert_allclose(velocity, velocities[0], atol=2e-5)
        baseline, *_ = fixture(0.0, center_only=True)
        with self.assertRaisesRegex(RuntimeError, "capacity exceeded"):
            fixture(0.01, center_only=True, row_limit=int(baseline["count"][0]))

    def test_per_shape_radius_selects_contacts(self):
        """Apply torsion when either shape of the pair has a radius, using the larger one."""
        baseline, *_ = fixture(0.0, center_only=True)
        self.assertLess(0.01, float(np.max(np.abs(baseline["body_qd"][:, 5]))))
        excluded, *_ = fixture(0.01, center_only=True, torsion_shapes=())
        np.testing.assert_array_equal(baseline["body_qd"], excluded["body_qd"])
        for shapes in ((0,), (1,), (0, 1)):
            with self.subTest(shapes=shapes):
                result, solver, *_ = fixture(0.01, center_only=True, torsion_shapes=shapes)
                self.assertLess(np.max(np.abs(result["body_qd"][:, 5])), 1e-4)
                self.assertEqual(solver._torsion_stats["rows"], 1)
                self.assertEqual(solver._torsion_stats["groups"][0]["effective_radius_m"], np.float32(0.01))
        # A pair uses the larger radius, so a smaller second radius gives the same rows.
        reference, *_ = fixture(0.01, center_only=True)
        mixed, *_ = fixture(np.array([0.01, 1.0e-4], dtype=np.float32), center_only=True)
        for key in reference:
            np.testing.assert_array_equal(mixed[key], reference[key], err_msg=key)

    def test_invalid_input_and_unsupported_modes(self):
        """Reject invalid scopes and modes rather than silently ignoring spin friction."""
        for radius in (-1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                fixture(radius)
        for options in (
            {"contact_warmstart": True},
            {"velocity_iterations": 1, "enable_bilateral_preelimination": True},
            {"articulated_contact_response": "propagation"},
            {"friction_mode": "bisection"},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                fixture(0.01, **options)

    def test_compliance_combination_rejected_at_construction(self):
        """Fail before stepping when both experimental responses are requested."""
        with self.assertRaisesRegex(ValueError, "torsion.*contact_compliance"):
            fixture(
                0.01,
                contact_compliance=True,
                enable_restitution=False,
                contact_shared_anchor=False,
                contact_friction_shared_anchor=False,
            )

    def test_capture_is_explicitly_rejected(self):
        """Reject capture before host contact grouping is attempted."""
        _, solver, model, initial, contacts = fixture(0.01)
        output = model.state()
        with self.assertRaisesRegex(RuntimeError, "graph capture"):
            with wp.ScopedCapture():
                solver.step(initial, output, model.control(), contacts, 0.0025)

    def test_hydro_combination_is_rejected(self):
        """Reject actual hydro contact stiffness before combining contact mechanisms."""
        _, solver, model, initial, contacts = fixture(0.01)
        contacts.rigid_contact_stiffness = wp.ones(contacts.rigid_contact_max, device=model.device)
        with self.assertRaisesRegex(ValueError, "hydroelastic"):
            solver.step(initial, model.state(), model.control(), contacts, 0.0025)


@unittest.skipUnless(wp.is_cuda_available(), "Contact torsion currently requires CUDA")
class TestContactTorsionWithPatches(unittest.TestCase):
    """Bound one spin row per persistent friction patch region."""

    def test_constructor_accepts_torsion_with_patches(self):
        """Build torsion on top of default and explicit patch friction without friction warnings."""
        for options in ({"friction_anchor_beta": None}, {}):
            with self.subTest(options=options), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                _, solver, *_ = fixture(0.01, center_only=True, **{**PATCH_OPTIONS, **options})
            self.assertEqual([str(w.message) for w in caught if "friction" in str(w.message)], [])
            self.assertTrue(solver._contact_torsion_enabled)
            self.assertTrue(solver._friction_anchors_enabled)
            self.assertAlmostEqual(solver._friction_anchor_beta, 0.2)
            self.assertEqual(solver._torsion_stats["rows"], 1)

    def test_patch_regions_pool_the_spin_budget(self):
        """Share mu times the region's pooled normal load between anchor sliding and spin."""
        for center_only, anchors in ((True, 1), (False, 2)):
            with self.subTest(anchors=anchors):
                actual, solver, *_ = fixture(
                    0.01, spin=100.0, sliding=10.0, sat=not center_only, center_only=center_only, **PATCH_OPTIONS
                )
                self.assertEqual(solver._torsion_stats["rows"], 1)
                group = solver._torsion_stats["groups"][0]
                self.assertEqual(len(group["anchor_rows"]), anchors)
                self.assertGreaterEqual(len(group["normal_rows"]), anchors)
                self.assertAlmostEqual(group["mu"], 0.5, places=6)
                self.assertAlmostEqual(float(actual["row_mu"][0, group["row"]]), 0.5, places=6)
                for slot in group["anchor_rows"]:
                    self.assertAlmostEqual(float(actual["row_mu"][0, slot + 1]), 0.5 / anchors, places=6)
                    self.assertEqual(list(actual["row_type"][0, slot + 1 : slot + 3]), [2, 2])
                for slot in set(group["normal_rows"]) - set(group["anchor_rows"]):
                    self.assertNotEqual(int(actual["row_type"][0, slot + 1]), 2)
                impulse = actual["impulses"][0, : int(actual["count"][0])]
                pooled = sum(max(float(impulse[r]), 0.0) for r in group["normal_rows"])
                used = sum(float(np.linalg.norm(impulse[r + 1 : r + 3])) for r in group["anchor_rows"])
                used += abs(float(impulse[group["row"]])) / 0.01
                self.assertGreater(pooled, 0.0)
                self.assertLessEqual(used, 0.5 * pooled + 2e-6)

    def test_single_anchor_spin_stops_below_bound(self):
        """Stop a slow spin that a lone patch anchor cannot resist on its own."""
        baseline, solver, *_ = fixture(0.0, spin=1.0, center_only=True, **PATCH_OPTIONS)
        self.assertTrue(solver._friction_anchors_enabled)
        self.assertGreater(np.max(np.abs(baseline["body_qd"][:, 5])), 0.9)
        result, solver, *_ = fixture(0.01, spin=1.0, center_only=True, **PATCH_OPTIONS)
        self.assertLess(np.max(np.abs(result["body_qd"][:, 5])), 1e-4)
        self.assertEqual(len(solver._torsion_stats["groups"][0]["anchor_rows"]), 1)

    def test_held_box_yaw_torque_threshold(self):
        """Hold a resting box below mu * N * radius and let it spin above, with patch friction."""
        radius, mu, mass = 0.05, 0.5, 0.3
        bound = mu * mass * 9.81 * radius
        held, solver = held_box(radius, 0.5 * bound, mu=mu, mass=mass)
        self.assertEqual(solver._torsion_stats["rows"], 1)
        self.assertEqual(len(solver._torsion_stats["groups"][0]["anchor_rows"]), 2)
        self.assertLess(abs(held), 1e-3)
        released, _ = held_box(radius, 2.0 * bound, mu=mu, mass=mass)
        self.assertGreater(released, 1.0)
        anchors_only, _ = held_box(0.0, 0.5 * bound, mu=mu, mass=mass)
        self.assertGreater(anchors_only, 1.0)


@unittest.skipUnless(wp.is_cuda_available(), "Contact torsion requires CUDA")
class TestContactTorsionRegressions(unittest.TestCase):
    """Reject invalid storage/lifecycle paths rather than silently welding spin."""

    def test_radius_edits_apply_while_enablement_is_fixed(self):
        """Read edited torsion radii every step, but enable torsion only when a radius was positive at construction."""
        _, solver, model, initial, contacts = fixture(0.01, center_only=True)
        self.assertEqual(solver._torsion_stats["groups"][0]["effective_radius_m"], np.float32(0.01))
        set_torsion_radius(model, 0.02)
        solver.step(initial, model.state(), model.control(), contacts, 0.0025)
        self.assertEqual(solver._torsion_stats["groups"][0]["effective_radius_m"], np.float32(0.02))
        _, solver, model, initial, contacts = fixture(0.0, center_only=True)
        set_torsion_radius(model, 0.01)
        solver.step(initial, model.state(), model.control(), contacts, 0.0025)
        self.assertFalse(solver._contact_torsion_enabled)
        count = int(solver._constraint_count.numpy()[0])
        self.assertFalse(np.any(solver._row_type.numpy()[0, :count] == PGS_CONSTRAINT_TYPE_TORSION))

    def test_previous_friction_rows_are_not_current_rows(self):
        """Reject stale tangent metadata when a later step admits only a normal."""
        _, solver, model, initial, contacts = fixture(0.01, center_only=True, contact_friction_gap_max=1.0)
        self.assertEqual(solver._torsion_stats["rows"], 1)
        solver.contact_friction_gap_max = -1.0
        solver.step(initial, model.state(), model.control(), contacts, 0.0025)
        self.assertEqual(solver._torsion_stats["rows"], 0)

    def test_both_tangent_rows_must_belong_to_current_normal(self):
        """Reject a missing or mismatched second tangent, not just the first."""
        for invalid in ("count", "type", "parent"):
            with self.subTest(invalid=invalid):
                _, solver, _model, initial, contacts = fixture(0.01, center_only=True)
                row = solver._torsion_stats["groups"][0]["normal_rows"][0]
                if invalid == "count":
                    counts = solver._constraint_count.numpy()
                    counts[0] = row + 2
                    solver._constraint_count.assign(counts)
                else:
                    array = solver._row_type if invalid == "type" else solver._row_parent
                    values = array.numpy()
                    values[0, row + 2] = -1
                    array.assign(values)
                self.assertEqual(_contact_groups(solver, initial, contacts), [])

    def test_normal_parent_metadata_is_preserved(self):
        """Keep CONTACT parents available for pooled friction-patch load rings."""
        result, _solver, *_ = fixture(0.01, center_only=True)
        normal = result["row_type"][0, : result["count"][0]] == 0
        self.assertGreater(np.count_nonzero(normal), 0)
        np.testing.assert_array_equal(result["row_parent"][0, : result["count"][0]][normal], -1)

    def test_torsion_uses_cfm_floor(self):
        """Apply the same configured diagonal floor as other dense rows."""
        result, solver, *_ = fixture(0.01, center_only=True, cfm=0.0007)
        active = result["row_type"] == 7
        self.assertGreater(np.count_nonzero(active), 0)
        np.testing.assert_allclose(solver._row_cfm.numpy()[active], 0.0007)

    def test_torsion_respects_relaxation(self):
        """Apply half the unconstrained spin correction at omega one half."""
        result, solver, *_ = fixture(100.0, center_only=True, iterations=1, relaxation=0.5)
        self.assertEqual(solver._torsion_stats["rows"], 1)
        spin = np.abs(result["v_out"].reshape(2, 6)[:, 5])
        np.testing.assert_allclose(spin, 0.5, atol=1e-4)

    def test_torsion_keeps_joint_velocity_limits(self):
        """Let joint velocity limits have the last word over the spin rows of each sweep."""
        for (iterations, velocity_iterations), preparation, radius in _VELOCITY_LIMIT_CASES:
            with self.subTest(
                iterations=iterations, velocity_iterations=velocity_iterations, preparation=preparation, radius=radius
            ):
                _, solver, model, initial, contacts = fixture(
                    radius,
                    center_only=True,
                    spin=0.0,
                    iterations=iterations,
                    velocity_iterations=velocity_iterations,
                    contact_torsion_device=preparation != "host",
                )
                limits = np.full(model.joint_dof_count, np.inf, dtype=np.float32)
                limits[5] = _VELOCITY_LIMIT
                model.joint_velocity_limit.assign(limits)
                qd = _step_spinning_pad(solver, model, initial, contacts, preparation)
                self.assertEqual(_row_count(solver, PGS_CONSTRAINT_TYPE_TORSION), int(radius > 0.0))
                self.assertLessEqual(abs(float(qd[5])), _VELOCITY_LIMIT * (1.0 + 1e-5))
                if radius > 0.0:
                    # The spin row still resists the relative spin.
                    self.assertLess(float(qd[11]), 9.0)

    def test_torsion_keeps_fused_drive_velocity_limits(self):
        """Let the fused drive-row velocity clamp have the last word over the spin rows of each sweep."""
        dofs = 12
        limits = np.full(dofs, np.inf, dtype=np.float32)
        limits[5] = _VELOCITY_LIMIT
        damping = np.zeros(dofs, dtype=np.float32)
        damping[5] = 1.0e-3
        for (iterations, velocity_iterations), preparation, radius in _VELOCITY_LIMIT_CASES:
            with self.subTest(
                iterations=iterations, velocity_iterations=velocity_iterations, preparation=preparation, radius=radius
            ):
                _, solver, model, initial, contacts = fixture(
                    radius,
                    center_only=True,
                    spin=0.0,
                    model_arrays={"joint_velocity_limit": limits, "joint_target_kd": damping},
                    drive_mode="rows",
                    fuse_joint_velocity_limits=True,
                    iterations=iterations,
                    velocity_iterations=velocity_iterations,
                    contact_torsion_device=preparation != "host",
                )
                self.assertTrue(solver._velocity_limits_fused)
                qd = _step_spinning_pad(solver, model, initial, contacts, preparation)
                self.assertEqual(_row_count(solver, PGS_CONSTRAINT_TYPE_JOINT_TARGET), 1)
                self.assertEqual(_row_count(solver, PGS_CONSTRAINT_TYPE_TORSION), int(radius > 0.0))
                self.assertLessEqual(abs(float(qd[5])), _VELOCITY_LIMIT * (1.0 + 1e-5))
                if radius > 0.0:
                    self.assertLess(float(qd[11]), 9.0)

    def test_torsion_keeps_free_body_velocity_limits(self):
        """Let the free-body velocity-limit rows have the last word over a mixed contact's spin row."""
        for (iterations, velocity_iterations), preparation, radius in _VELOCITY_LIMIT_CASES:
            with self.subTest(
                iterations=iterations, velocity_iterations=velocity_iterations, preparation=preparation, radius=radius
            ):
                # A spinning articulated pad presses on a resting free body whose spin is limited.
                builder = newton.ModelBuilder(gravity=(0, 0, 0))
                SolverFeatherPGS.register_custom_attributes(builder)
                inertia = wp.mat33(np.diag([0.00012, 0.00012, 0.00008]).astype(np.float32))
                shape = newton.ModelBuilder.ShapeConfig(density=0, mu=0.5, restitution=0.0)
                pose = wp.transform(wp.vec3(0, 0, -0.025), wp.quat_identity())
                pad = builder.add_link(xform=pose, mass=0.3, inertia=inertia)
                axes = [newton.ModelBuilder.JointDofConfig(axis=wp.vec3(*a)) for a in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
                builder.add_articulation(
                    [builder.add_joint_d6(-1, pad, parent_xform=pose, linear_axes=axes, angular_axes=axes)]
                )
                builder.add_shape_box(pad, hx=0.02, hy=0.015, hz=0.025, cfg=shape)
                free = builder.add_body(
                    xform=wp.transform(wp.vec3(0, 0, 0.025), wp.quat_identity()),
                    mass=0.3,
                    inertia=inertia,
                    custom_attributes={"feather_pgs:rigid_body_max_angular_velocity": _VELOCITY_LIMIT},
                )
                builder.add_shape_box(free, hx=0.02, hy=0.015, hz=0.025, cfg=shape)
                model = builder.finalize(device="cuda:0")
                initial = model.state()
                initial.joint_qd.assign(np.array([0, 0, 0.1, 0, 0, 10.0, 0, 0, -0.1, 0, 0, 0], np.float32))
                newton.eval_fk(model, initial.joint_q, initial.joint_qd, initial)
                pipeline = newton.CollisionPipeline(
                    model, contact_matching="latest", reduce_contacts=False, broad_phase="nxn", rigid_contact_max=64
                )
                contacts = pipeline.contacts()
                pipeline.collide(initial, contacts)
                # One central witness: any free-body spin comes from the spin row.
                poses = initial.body_q.numpy()
                for side in (0, 1):
                    name = f"rigid_contact_point{side}"
                    body = model.shape_body.numpy()[getattr(contacts, f"rigid_contact_shape{side}").numpy()[0]]
                    points = getattr(contacts, name).numpy()
                    points[0] = -poses[body, :3]
                    getattr(contacts, name).assign(points)
                contacts.rigid_contact_count.assign(np.array([1], np.int32))
                set_torsion_radius(model, radius)
                solver = SolverFeatherPGS(
                    model,
                    solve_mode="matrix_free",
                    articulated_contact_response="immediate",
                    friction_anchor_beta=0.0,
                    iterations=iterations,
                    velocity_iterations=velocity_iterations,
                    baumgarte=0.05,
                    cfm=0,
                    contact_regularization=0,
                    contact_warmstart=False,
                    articulation_constraint_max=64,
                    free_body_constraint_max=16,
                    contact_torsion_device=preparation != "host",
                )
                solver._rigid_body_angular_damping.zero_()
                qd = _step(solver, model, initial, contacts, preparation)
                self.assertEqual(_row_count(solver, PGS_CONSTRAINT_TYPE_TORSION), int(radius > 0.0))
                self.assertGreater(int(solver._mf_constraint_count.numpy()[0]), 0)
                self.assertLessEqual(abs(float(qd[11])), _VELOCITY_LIMIT * (1.0 + 1e-5))
                if radius > 0.0:
                    self.assertLess(float(qd[5]), 9.0)

    def test_contact_row_loss_fails_without_diagnostics(self):
        """Detect rolled-back dropped contacts even with optional telemetry off."""
        baseline, *_ = fixture(0.0, center_only=True)
        limit = int(baseline["count"][0]) - 1
        with self.assertRaisesRegex(RuntimeError, "[Oo]verflow|[Dd]ropped|capacity"):
            fixture(0.01, center_only=True, row_limit=limit, row_watermark=False, warn_constraint_overflow=False)

    def test_selected_unsupported_shape_fails_at_construction(self):
        """Reject explicitly selected unsupported geometry instead of ignoring it."""
        _, _solver, model, *_ = fixture(0.0, center_only=True)
        types = model.shape_type.numpy()
        types[0] = int(GeoType.MESH)
        model.shape_type.assign(types)
        set_torsion_radius(model, 0.01, (0,))
        with self.assertRaisesRegex(ValueError, r"does not support the geometry of shapes \[0\]"):
            SolverFeatherPGS(model, solve_mode="matrix_free")

    def test_hydro_contact_input_is_rejected(self):
        """Reject actual positive hydro contact stiffness, not a synthetic solver flag."""
        for device_preparation, stepped in ((False, True), (False, False), (True, False)):
            _, solver, model, initial, contacts = fixture(
                0.01, center_only=True, contact_torsion_device=device_preparation, step=stepped
            )
            contacts.rigid_contact_stiffness = wp.ones(contacts.rigid_contact_max, device=model.device)
            with (
                self.subTest(device_preparation=device_preparation, stepped=stepped),
                self.assertRaisesRegex(ValueError, "hydroelastic"),
            ):
                solver.step(initial, model.state(), model.control(), contacts, 0.0025)

    def test_contact_input_overflow_is_rejected(self):
        """Reject a contact count beyond the buffer instead of grouping a truncated input."""
        for device_preparation in (False, True):
            _, solver, model, initial, contacts = fixture(
                0.01, center_only=True, contact_torsion_device=device_preparation, step=False
            )
            contacts.rigid_contact_count.fill_(contacts.rigid_contact_max + 1)
            with (
                self.subTest(device_preparation=device_preparation),
                self.assertRaisesRegex(RuntimeError, "Contact input overflow"),
            ):
                solver.step(initial, model.state(), model.control(), contacts, 0.0025)


_VELOCITY_LIMIT = 0.1


_VELOCITY_LIMIT_CASES = tuple(itertools.product(((1, 0), (4, 0), (1, 2)), ("host", "device", "graph"), (0.0, 1.0)))


def _step_spinning_pad(solver, model, initial, contacts, preparation):
    """Spin only the second pad, so any DOF 5 motion is transferred through the contact."""
    velocity = initial.joint_qd.numpy()
    velocity[5], velocity[11] = 0.0, 10.0
    initial.joint_qd.assign(velocity)
    newton.eval_fk(model, initial.joint_q, initial.joint_qd, initial)
    return _step(solver, model, initial, contacts, preparation)


def _step(solver, model, initial, contacts, preparation):
    """Step once with host, device or graph-captured torsion preparation and return joint_qd."""
    output = model.state()
    if preparation == "graph":
        with wp.ScopedCapture(device=model.device) as capture:
            solver.step(initial, output, model.control(), contacts, 0.0025)
        wp.capture_launch(capture.graph)
        solver.check_constraint_capacity()
    else:
        solver.step(initial, output, model.control(), contacts, 0.0025)
    return output.joint_qd.numpy()


def _row_count(solver, row_type):
    count = int(solver._constraint_count.numpy()[0])
    return int(np.count_nonzero(solver._row_type.numpy()[0, :count] == row_type))


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestTorsionCombined(unittest.TestCase):
    def check_pair(self, regularization, velocity, **options):
        """Check split position integration and the accumulated velocity correction."""
        common = dict(
            iterations=64,
            contact_regularization=regularization,
            enable_bilateral_preelimination=False,
            **options,
        )
        baseline, *_ = fixture(0.01, velocity_iterations=0, **common)
        result, solver, model, state, contacts = fixture(0.01, velocity_iterations=velocity, **common)
        np.testing.assert_array_equal(result["body_q"], baseline["body_q"])
        count = int(result["count"][0])
        delta = result["impulses"][0, :count] - baseline["impulses"][0, :count]
        np.testing.assert_allclose(
            result["v_out"],
            baseline["v_out"].astype(np.float64) + result["Y_world"][0, :count].astype(np.float64).T @ delta,
            atol=2e-5,
            rtol=8 * np.finfo(np.float32).eps,
        )
        weights = solver._row_w.numpy()[0, :count]
        types = result["row_type"][0, :count]
        np.testing.assert_allclose(weights[types == 0], 1 / (1 + regularization), atol=1e-7)
        TestTorsionVelocitySolve._assert_budget(self, result, solver)
        self.assertTrue(np.isfinite(result["body_qd"]).all())
        return baseline, result, solver, model, state, contacts

    def test_analytic_normal_load_position_soft_velocity_rigid(self):
        """Undo normal softness only in velocity cleanup and budget spin from final load."""
        for regularization in (0.01, 0.02, 0.5):
            for velocity in (1, 4, 16):
                with self.subTest(reg=regularization, velocity=velocity):
                    baseline, result, solver, *_ = self.check_pair(
                        regularization, velocity, center_only=True, spin=100.0
                    )
                    group = solver._torsion_stats["groups"][0]
                    normals = group["normal_rows"]
                    self.assertAlmostEqual(
                        float(baseline["impulses"][0, normals].sum()), 0.03 / (1 + regularization), delta=2e-7
                    )
                    normal = float(result["impulses"][0, normals].sum())
                    self.assertAlmostEqual(normal, 0.03, delta=2e-7)
                    self.assertAlmostEqual(
                        abs(float(result["impulses"][0, group["row"]])), 0.01 * 0.5 * normal, delta=2e-8
                    )
                    self.assertLess(abs(float(result["body_qd"][0, 5])), 100.0)

    def test_distributed_budget_and_export(self):
        """Keep point and patch budgets coherent across internal timesteps."""
        for patch_options in ({}, PATCH_OPTIONS):
            for dt in (0.00125, 0.0025, 0.005):
                with self.subTest(patch=bool(patch_options), dt=dt):
                    _, result, solver, _, _, contacts = self.check_pair(
                        0.01, 16, spin=10.0, sliding=1.0, dt=dt, **patch_options
                    )
                    count = int(contacts.rigid_contact_count.numpy()[0])
                    forces = result["contact_f"][:count, :3]
                    normals = contacts.rigid_contact_normal.numpy()[:count]
                    for index, slot in enumerate(solver._contact_slot.numpy()[:count]):
                        if slot >= 0:
                            self.assertAlmostEqual(
                                abs(float(forces[index] @ normals[index])) * dt,
                                float(result["impulses"][0, slot]),
                                delta=2e-7,
                            )

    def test_no_load_near_positive_gap_and_release(self):
        """Refund sliding and spin even for the narrow positive initial-gap tolerance."""
        for gap in (0.5e-6, 0.01):
            _, _, solver, *_ = self.check_pair(0.01, 16, center_only=True, spin=100.0, **PATCH_OPTIONS)
            group = solver._torsion_stats["groups"][0]
            n = int(solver._constraint_count.numpy()[0])
            before = solver._impulses.numpy()[0, :n].copy()
            phi = solver._phi.numpy()
            phi[0, group["normal_rows"]] = gap
            solver._phi.assign(phi)
            # Force a separating end-gap through the normal row's response vector.
            row = group["normal_rows"][0]
            separating = solver._Y_world.numpy()[0, row].copy()
            solver._v_out_snap.assign(separating)
            solver._v_hat.assign(separating)
            solver._v_out.assign(separating + solver._Y_world.numpy()[0, :n].T @ before)
            velocity_before = solver._v_out.numpy().copy()
            solver._conclude_matrix_free_position_problem(0.0025)
            solver._run_matrix_free_velocity_post_solve()
            after = solver._impulses.numpy()[0, :n]
            self.assertLess(np.abs(after[group["normal_rows"]]).max(initial=0), 2e-7)
            self.assertLess(abs(float(after[group["row"]])), 2e-7)
            for anchor in group["anchor_rows"]:
                self.assertLess(np.linalg.norm(after[anchor + 1 : anchor + 3]), 2e-7)
            np.testing.assert_allclose(
                solver._v_out.numpy(), velocity_before + solver._Y_world.numpy()[0, :n].T @ (after - before), atol=2e-5
            )


@wp.kernel(enable_backward=False, module="unique", module_options={"fuse_fp": False})
def _rounding_samples(
    poses: wp.array[wp.transform],
    points: wp.array[wp.vec3],
    directions: wp.array[wp.vec3],
    transformed: wp.array[wp.vec3],
    dotted: wp.array[float],
):
    i = wp.tid()
    transformed[i] = _host_transform_point(poses[i], points[i])
    dotted[i] = _host_dot3(points[i], directions[i])


def synthetic_fixture(*, worlds=3, witnesses=5, patches=True, seed=4, device="cpu", padding=7):
    """Create ordered/reordered contact worlds with unequal articulation DOFs."""
    rng = np.random.default_rng(seed)
    n = worlds * witnesses
    capacity = n + padding
    dense = witnesses * 3 + 8
    records = [(world, local) for world in range(worlds) for local in range(witnesses)]
    rng.shuffle(records)

    def arr(value, dtype=None):
        return wp.array(np.asarray(value), dtype=dtype, device=device)

    def zero(shape, dtype=float):
        return wp.zeros(shape, dtype=dtype, device=device)

    def full(shape, value, dtype=int):
        return wp.full(shape, value, dtype=dtype, device=device)

    count = np.full(worlds, witnesses * 3, np.int32)
    row_types = np.zeros((worlds, dense), np.int32)
    parents = np.full((worlds, dense), -1, np.int32)
    mu = np.zeros((worlds, dense), np.float32)
    world_ids = np.zeros(capacity, np.int32)
    slots = np.full(capacity, -1, np.int32)
    anchors = np.zeros(capacity, np.int32)
    sa = np.zeros(capacity, np.int32)
    sb = np.zeros(capacity, np.int32)
    owner = np.full(capacity, -1, np.int32)
    normals = np.zeros((capacity, 3), np.float32)
    points = np.zeros((capacity, 3), np.float32)
    region_seeds = {}
    for c, (world, local) in enumerate(records):
        slot = local * 3
        region = local % 2
        region_seeds.setdefault((world, region), c)
        world_ids[c], slots[c], anchors[c] = world, slot, 3
        sa[c], sb[c] = world * 4 + region, world * 4 + 2 + region
        owner[c] = region_seeds[world, region]
        normals[c] = (0.0, 0.0, -1.0)
        points[c] = (rng.uniform(-0.1, 0.1), rng.uniform(-0.1, 0.1), 0.0)
        row_types[world, slot : slot + 3] = (
            PGS_CONSTRAINT_TYPE_CONTACT,
            PGS_CONSTRAINT_TYPE_FRICTION,
            PGS_CONSTRAINT_TYPE_FRICTION,
        )
        parents[world, slot : slot + 3] = (slot, slot, slot)
        mu[world, slot + 1 : slot + 3] = 0.7
    if patches:
        for world in range(worlds):
            for region in range(2):
                members = [c for c, (w, local) in enumerate(records) if w == world and local % 2 == region]
                for i, c in enumerate(members):
                    parents[world, slots[c]] = slots[members[(i + 1) % len(members)]]
                    mu[world, slots[c] + 1 : slots[c] + 3] /= len(members)
    shape_body = np.repeat(np.arange(worlds * 2, dtype=np.int32), 2)
    response_sizes = np.tile([6, 3], worlds).astype(np.int32)
    starts = np.concatenate(([0], np.cumsum(response_sizes))).astype(np.int32)
    total_dofs = int(starts[-1])
    motion = rng.normal(size=(total_dofs, 6)).astype(np.float32)
    body_velocity = rng.normal(size=(worlds * 2, 6)).astype(np.float32)
    prescribed = np.zeros(worlds * 2, np.int32)
    if worlds > 1:
        prescribed[-1] = 1
    model = SimpleNamespace(
        device=wp.get_device(device),
        shape_count=worlds * 4,
        shape_body=arr(shape_body, int),
        shape_type=full(worlds * 4, int(GeoType.BOX)),
        joint_ancestor=full(worlds * 2, -1),
        joint_qd_start=arr(starts, int),
    )
    state = SimpleNamespace(
        body_q=arr(np.tile([0, 0, 0, 0, 0, 0, 1], (worlds * 2, 1)).astype(np.float32), wp.transform)
    )
    augmented = SimpleNamespace(
        _joint_S_s=arr(motion, wp.spatial_vector), _body_v_s=arr(body_velocity, wp.spatial_vector)
    )
    contacts = SimpleNamespace(
        rigid_contact_count=arr([n], int),
        rigid_contact_max=capacity,
        rigid_contact_stiffness=None,
        rigid_contact_shape0=arr(sa, int),
        rigid_contact_shape1=arr(sb, int),
        rigid_contact_normal=arr(normals, wp.vec3),
        rigid_contact_point0=arr(points, wp.vec3),
        rigid_contact_point1=arr(points, wp.vec3),
        rigid_contact_margin0=zero(capacity),
        rigid_contact_margin1=zero(capacity),
    )
    solver = SimpleNamespace(
        model=model,
        _world_count=worlds,
        _articulation_constraint_max=dense,
        _max_contacts_alloc=capacity,
        _contact_torsion_shape_radius=full(worlds * 4, 0.01, float),
        _friction_anchors_enabled=patches,
        _friction_patches=SimpleNamespace(current=SimpleNamespace(owner=arr(owner, int))),
        _contact_path=zero(capacity, int),
        _contact_slot=arr(slots, int),
        _contact_world=arr(world_ids, int),
        _contact_slots_needed=arr(anchors, int),
        _constraint_count=arr(count, int),
        _slot_counter=arr(count, int),
        _row_type=arr(row_types, int),
        _row_parent=arr(parents, int),
        _row_mu=arr(mu, float),
        _row_dropped_dense=zero(worlds, int),
        _contact_torsion_group=full((worlds, dense), -1),
        _cfm=0.003,
        _torsion_row_radius=zero((worlds, dense)),
        _body_to_joint=arr(np.arange(worlds * 2, dtype=np.int32), int),
        _body_to_articulation=arr(np.arange(worlds * 2, dtype=np.int32), int),
        _articulation_dof_start=arr(starts[:-1], int),
        _articulation_response_dof_count=arr(response_sizes, int),
        _art_group_idx=arr(np.repeat(np.arange(worlds, dtype=np.int32), 2), int),
        _art_to_world=arr(np.repeat(np.arange(worlds, dtype=np.int32), 2), int),
        _prescribed_articulation=arr(prescribed, int),
        _group_to_art={size: arr(np.flatnonzero(response_sizes == size).astype(np.int32), int) for size in (3, 6)},
        _J_by_size={size: zero((worlds, dense, size)) for size in (3, 6)},
        _propagation_torsion_rows=False,
        _contact_mf_origin=zero(capacity, int),
        _propagation_torsion_head=full(1, -1),
        _propagation_torsion_first=full((1, 1), -1),
        _propagation_torsion_next=full((1, 1), -1),
    )
    for name in ("_row_beta", "_row_cfm", "_phi", "_target_velocity", "_row_restitution"):
        setattr(solver, name, zero((worlds, dense)))
    return solver, state, augmented, contacts


def run_oracle(solver, state, augmented, contacts):
    """Invoke only the host preparation oracle, not a CPU solver backend."""
    with patch.object(host, "validate_torsion_step", return_value=None):
        host.prepare_torsion_rows(solver, state, augmented, contacts)


class TestDeviceTorsionPreparation(unittest.TestCase):
    """Exercise portable preparation kernels independently of a full CUDA solve."""

    def test_public_device_option_is_default_off(self):
        """Keep public device preparation opt-in."""
        self.assertIs(
            signature(solver_feather_pgs.SolverFeatherPGS).parameters["contact_torsion_device"].default, False
        )

    def compare(self, *, device="cpu", modify=None, **options):
        """Compare every row field, group membership and size-specific Jacobian."""
        expected, state, augmented, contacts = synthetic_fixture(device=device, **options)
        actual, state2, augmented2, contacts2 = synthetic_fixture(device=device, **options)
        if modify is not None:
            modify(expected, contacts)
            modify(actual, contacts2)
        run_oracle(expected, state, augmented, contacts)
        preparer = DeviceTorsionPreparation(actual)
        preparer.prepare(state2, augmented2, contacts2)
        for name in (
            "_constraint_count",
            "_slot_counter",
            "_row_type",
            "_row_parent",
            "_row_mu",
            "_row_beta",
            "_row_cfm",
            "_phi",
            "_target_velocity",
            "_row_restitution",
            "_contact_torsion_group",
        ):
            np.testing.assert_allclose(
                getattr(actual, name).numpy(), getattr(expected, name).numpy(), rtol=2e-6, atol=2e-6, err_msg=name
            )
        for size in expected._J_by_size:
            np.testing.assert_allclose(
                actual._J_by_size[size].numpy(),
                expected._J_by_size[size].numpy(),
                rtol=2e-6,
                atol=2e-6,
                err_msg=f"J{size}",
            )
        return actual, preparer

    def test_randomized_patch_and_point_oracle(self):
        """Preserve contact-order semantics across mixed shapes and unequal DOFs."""
        for patches in (False, True):
            for seed in range(5):
                with self.subTest(patches=patches, seed=seed):
                    self.compare(patches=patches, seed=seed)

    def test_host_math_rounding(self):
        """Match explicit dot and quaternion-transform rounding on CPU and CUDA."""
        rng = np.random.default_rng(84)
        poses = rng.normal(size=(1024, 7)).astype(np.float32)
        poses[:, 3:] /= np.linalg.norm(poses[:, 3:], axis=1, keepdims=True)
        points = rng.normal(size=(1024, 3)).astype(np.float32)
        directions = rng.normal(size=points.shape).astype(np.float32)
        expected_points = np.array([host._transform_point(q, p) for q, p in zip(poses, points, strict=True)])
        # Independent scalar reference: float32 products, sequential float64
        # sum, one float32 output rounding. Do not consult platform BLAS.
        expected_dot = np.array(
            [sum(float(np.float32(a[i] * b[i])) for i in range(3)) for a, b in zip(points, directions, strict=True)],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(
            np.array([host._dot3(a, b) for a, b in zip(points, directions, strict=True)]), expected_dot
        )
        for device in ["cpu", "cuda:0"] if wp.is_cuda_available() else ["cpu"]:
            transformed = wp.empty(1024, dtype=wp.vec3, device=device)
            dotted = wp.empty(1024, dtype=float, device=device)
            wp.launch(
                _rounding_samples,
                dim=1024,
                inputs=[
                    wp.array(poses, dtype=wp.transform, device=device),
                    wp.array(points, dtype=wp.vec3, device=device),
                    wp.array(directions, dtype=wp.vec3, device=device),
                    transformed,
                    dotted,
                ],
                device=device,
            )
            np.testing.assert_array_equal(transformed.numpy(), expected_points)
            np.testing.assert_array_equal(dotted.numpy(), expected_dot)

    def test_grouping_and_jacobians_do_not_depend_on_blas(self):
        """Build host groups and rows without platform-dependent NumPy dot."""
        for patches in (False, True):
            with self.subTest(patches=patches):
                with patch.object(host.np, "dot", side_effect=AssertionError("BLAS dot must not decide torsion rows")):
                    self.compare(patches=patches)

    def test_empty_and_larger_preallocated_buffer(self):
        """Keep zero contacts valid and distinguish configured from input capacity."""
        solver, state, augmented, contacts = synthetic_fixture()
        solver._max_contacts_alloc += 5
        device = DeviceTorsionPreparation(solver)
        device.prepare(state, augmented, contacts)
        self.assertEqual(int(np.count_nonzero(device.work.spin_row.numpy() >= 0)), 6)
        contacts.rigid_contact_count.zero_()
        device.prepare(state, augmented, contacts)
        self.assertTrue(np.all(solver._contact_torsion_group.numpy() == -1))

    def test_ring_and_capacity_errors_are_latched(self):
        """Reject incomplete rings and capacity failures before row writes."""
        for failure in ("ring", "dense", "input", "spin"):
            solver, state, augmented, contacts = synthetic_fixture()
            before = solver._row_type.numpy().copy()
            if failure == "ring":
                parents = solver._row_parent.numpy()
                parents[0, 0] = solver._articulation_constraint_max - 1
                solver._row_parent.assign(parents)
            elif failure == "dense":
                solver._row_dropped_dense.fill_(1)
            elif failure == "input":
                contacts.rigid_contact_count.fill_(contacts.rigid_contact_max + 1)
            else:
                solver._constraint_count.fill_(solver._articulation_constraint_max)
            device = DeviceTorsionPreparation(solver)
            with self.subTest(failure=failure), self.assertRaises(RuntimeError):
                device.prepare(state, augmented, contacts)
            np.testing.assert_array_equal(before, solver._row_type.numpy())
            self.assertNotEqual(int(device.work.status.numpy()[0]), 0)
            with self.assertRaises(RuntimeError):
                device.prepare(state, augmented, None)

    def test_patch_segments_match_host_oracle(self):
        """Match the host patch grouping under selection, rejection and negative owners."""
        devices = ["cpu", "cuda:0"] if wp.is_cuda_available() else ["cpu"]
        for device in devices:
            for seed in range(8):
                results = []
                for oracle in (True, False):
                    solver, state, augmented, contacts = synthetic_fixture(
                        worlds=4, witnesses=12, seed=seed, device=device
                    )
                    rng = np.random.default_rng(seed)
                    radii = np.where(rng.random(solver.model.shape_count) < 0.6, 0.01, 0.0).astype(np.float32)
                    solver._contact_torsion_shape_radius = wp.array(radii, dtype=float, device=device)
                    anchors = solver._contact_slots_needed.numpy()
                    anchors[rng.random(anchors.size) < 0.3] = 1
                    solver._contact_slots_needed.assign(anchors)
                    points = contacts.rigid_contact_point0.numpy()
                    points[rng.random(len(points)) < 0.3, 2] = 0.002
                    contacts.rigid_contact_point0.assign(points)
                    if seed % 2:
                        owner = solver._friction_patches.current.owner.numpy()
                        owner[owner == owner.max()] = -1
                        solver._friction_patches.current.owner.assign(owner)
                    try:
                        if oracle:
                            run_oracle(solver, state, augmented, contacts)
                        else:
                            DeviceTorsionPreparation(solver).prepare(state, augmented, contacts)
                        error = None
                    except RuntimeError as exc:
                        error = str(exc)
                    names = ("_constraint_count", "_row_type", "_row_parent", "_row_mu", "_target_velocity")
                    arrays = [getattr(solver, name).numpy() for name in (*names, "_contact_torsion_group")]
                    results.append((error, arrays))
                with self.subTest(device=device, seed=seed):
                    self.assertEqual(results[0][0] is None, results[1][0] is None)
                    if results[0][0] is None:
                        for expected, actual in zip(results[0][1], results[1][1], strict=True):
                            np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)

    def test_patch_eligibility_and_point_cluster_edges(self):
        """Match mixed-anchor patches, selector rejection and nontransitive clusters."""
        for scenario in ("nonanchor", "positive_gap", "selector", "nontransitive"):

            def modify(solver, contacts, scenario=scenario):
                if scenario == "nonanchor":
                    anchors = solver._contact_slots_needed.numpy()
                    anchors[::2] = 1
                    solver._contact_slots_needed.assign(anchors)
                elif scenario == "positive_gap":
                    points = contacts.rigid_contact_point0.numpy()
                    points[:, 2] = 0.002
                    contacts.rigid_contact_point0.assign(points)
                elif scenario == "selector":
                    radii = np.zeros(solver.model.shape_count, dtype=np.float32)
                    radii[0] = 0.01
                    solver._contact_torsion_shape_radius = wp.array(radii, dtype=float, device=solver.model.device)
                else:
                    # Adjacent normals match, but the endpoints do not. The
                    # greedy oracle tests every member, not only its seed.
                    angles = (np.arange(contacts.rigid_contact_max) % 3) * 0.035
                    normals = np.column_stack((np.sin(angles), np.zeros_like(angles), -np.cos(angles)))
                    contacts.rigid_contact_normal.assign(normals.astype(np.float32))
                    contacts.rigid_contact_point0.zero_()
                    contacts.rigid_contact_point1.zero_()

            for patches in (False, True):
                with self.subTest(scenario=scenario, patches=patches):
                    self.compare(patches=patches, modify=modify)

    def test_touching_threshold_preserves_host_rounding(self):
        """Keep admission identical when cancelling dot products straddle the gap gate."""

        def modify(_solver, contacts):
            rng = np.random.default_rng(18)
            shape = (contacts.rigid_contact_max, 3)
            normal = rng.normal(size=shape).astype(np.float32)
            normal /= np.linalg.norm(normal, axis=1, keepdims=True)
            point = rng.normal(size=shape).astype(np.float32) * 0.1
            point -= np.sum(normal * point, axis=1, keepdims=True) * normal
            point += np.float32(1e-5) * normal
            contacts.rigid_contact_normal.assign(-normal)
            contacts.rigid_contact_point0.assign(point)
            contacts.rigid_contact_point1.zero_()

        for device in ["cpu", "cuda:0"] if wp.is_cuda_available() else ["cpu"]:
            for patches in (False, True):
                with self.subTest(device=device, patches=patches):
                    self.compare(device=device, worlds=512, witnesses=1, patches=patches, modify=modify)

    def test_rejected_region_and_normal_clusters(self):
        """Preserve whole-region rejection and all-member coplanar clustering."""
        for patches in (False, True):
            pairs = [synthetic_fixture(patches=patches) for _ in range(2)]
            for solver, _state, _augmented, contacts in pairs:
                types = solver.model.shape_type.numpy()
                types[0] = int(GeoType.MESH)
                solver.model.shape_type.assign(types)
                normals = contacts.rigid_contact_normal.numpy()
                normals[3] = (1, 0, 0)
                contacts.rigid_contact_normal.assign(normals)
                points = contacts.rigid_contact_point0.numpy()
                points[6, 2] = 0.001
                contacts.rigid_contact_point0.assign(points)
            expected, state, augmented, contacts = pairs[0]
            actual, state2, augmented2, contacts2 = pairs[1]
            run_oracle(expected, state, augmented, contacts)
            device = DeviceTorsionPreparation(actual)
            device.prepare(state2, augmented2, contacts2)
            np.testing.assert_array_equal(
                actual._contact_torsion_group.numpy(), expected._contact_torsion_group.numpy()
            )
            np.testing.assert_array_equal(actual._row_parent.numpy(), expected._row_parent.numpy())

    def test_more_than_host_global_capacity(self):
        """Retain over 4096 witnesses against a decomposed host grouping oracle."""
        solver, state, augmented, contacts = synthetic_fixture(worlds=822, witnesses=5)
        with self.assertRaisesRegex(RuntimeError, "host grouping capacity"):
            host._contact_groups(solver, state, contacts)
        worlds = solver._contact_world.numpy()
        expected = set()
        for first, last in ((0, 411), (411, 822)):
            solver._contact_path.assign(np.where((worlds >= first) & (worlds < last), 0, -1).astype(np.int32))
            for group in host._contact_groups(solver, state, contacts):
                expected.add((group[0].world, tuple(sorted(w.slot for w in group))))
        solver._contact_path.zero_()
        device = DeviceTorsionPreparation(solver)
        device.prepare(state, augmented, contacts)
        membership = solver._contact_torsion_group.numpy()
        actual = set()
        for world in range(solver._world_count):
            for spin in np.unique(membership[world]):
                if spin >= 0:
                    actual.add((world, tuple(np.flatnonzero(membership[world] == spin))))
        self.assertEqual(expected, actual)
        self.assertEqual(len(actual), 1644)

    def test_deferred_failure_is_not_partial_success(self):
        """Latch failure, suppress row solves and restore public dynamic state."""
        solver, state, augmented, contacts = synthetic_fixture()
        out = SimpleNamespace(body_q=wp.clone(state.body_q))
        device = DeviceTorsionPreparation(solver, deferred_errors=True)
        device.begin_step(state, out)
        out.body_q.zero_()
        solver._row_dropped_dense.fill_(1)
        device.prepare(state, augmented, contacts)
        device.end_step(out)
        np.testing.assert_array_equal(out.body_q.numpy(), state.body_q.numpy())
        self.assertTrue(np.all(solver._constraint_count.numpy() == 0))
        self.assertTrue(np.all(solver._contact_torsion_group.numpy() == -1))
        with self.assertRaisesRegex(RuntimeError, "Dense contact row overflow"):
            device.validate()
        solver._row_dropped_dense.zero_()
        device.prepare(state, augmented, contacts)
        with self.assertRaises(RuntimeError):
            device.validate()

    def test_deferred_preparation_has_no_host_array_transfers(self):
        """Reject accidental full or scalar host transfers in deferred preparation."""
        solver, state, augmented, contacts = synthetic_fixture()
        device = DeviceTorsionPreparation(solver, deferred_errors=True)
        with (
            patch.object(wp.array, "numpy", side_effect=AssertionError("Unexpected device-to-host transfer")),
            patch.object(wp.array, "assign", side_effect=AssertionError("Unexpected host assignment")),
        ):
            device.prepare(state, augmented, contacts)
        device.validate()

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_cuda_oracle(self):
        """Repeat the full row/Jacobian oracle comparison on CUDA."""
        for patches in (False, True):
            for seed in range(3):
                self.compare(device="cuda:0", patches=patches, seed=seed)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_cuda_cancelling_angular_projection_matches_host_rounding(self):
        """Preserve float32 host rounding for nearly cancelling angular rows."""
        pairs = [synthetic_fixture(device="cuda:0") for _ in range(2)]
        normal = np.array([0.8962938785552979, 0.44297733902931213, 0.020699456334114075], dtype=np.float32)
        angular = np.array(
            [
                [0.16465608775615692, -0.29055461287498474, -0.9425849914550781],
                [-0.411504328250885, 0.8481746912002563, -0.33356380462646484],
                [0.41150417923927307, -0.8481748104095459, 0.3335646986961365],
                [0.4004249572753906, 0.6479430198669434, -0.647942841053009],
            ],
            dtype=np.float32,
        )
        for _solver, _state, augmented, contacts in pairs:
            contacts.rigid_contact_normal.assign(np.tile(-normal, (contacts.rigid_contact_max, 1)))
            motion = augmented._joint_S_s.numpy()
            motion[:4, 3:] = angular
            augmented._joint_S_s.assign(motion)
        expected, state, augmented, contacts = pairs[0]
        actual, state2, augmented2, contacts2 = pairs[1]
        run_oracle(expected, state, augmented, contacts)
        DeviceTorsionPreparation(actual).prepare(state2, augmented2, contacts2)
        np.testing.assert_array_equal(actual._J_by_size[6].numpy()[0, :, :4], expected._J_by_size[6].numpy()[0, :, :4])

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_cuda_loaded_closed_linkage(self):
        """Preserve loaded loop-closure behavior with active spin rows in both profiles."""
        for velocity, regularization in ((16, 0.0), (0, 0.01)):
            results = []
            for device_preparation in (False, True):
                builder = _build_four_bar()
                root_half = float(np.sqrt(0.5))
                builder.add_shape_plane(plane=(0.0, root_half, root_half, -0.362 * root_half))
                model = builder.finalize(device="cuda:0")
                set_torsion_radius(model, 0.01)
                solver = newton.solvers.SolverFeatherPGS(
                    model,
                    solve_mode="matrix_free",
                    iterations=64,
                    velocity_iterations=velocity,
                    contact_regularization=regularization,
                    enable_bilateral_preelimination=False,
                    articulation_constraint_max=128,
                    free_body_constraint_max=64,
                    row_watermark=True,
                    **PATCH_OPTIONS,
                )
                self.assertGreater(solver._connect_count, 0)
                if device_preparation:
                    enable_device_torsion(solver)
                a, z = model.state(), model.state()
                newton.eval_fk(model, a.joint_q, a.joint_qd, a)
                control = model.control()
                target = model.joint_target_q.numpy()
                target[0] = 0.03
                control.joint_target_q.assign(target)
                pipeline = newton.CollisionPipeline(
                    model, contact_matching="latest", reduce_contacts=False, broad_phase="nxn", rigid_contact_max=64
                )
                contacts = pipeline.contacts()
                spins = 0
                load = 0.0
                for _ in range(32):
                    a.clear_forces()
                    pipeline.collide(a, contacts)
                    solver.step(a, z, control, contacts, 0.0025)
                    types = solver._row_type.numpy()
                    spins = max(spins, int(np.count_nonzero(types == PGS_CONSTRAINT_TYPE_TORSION)))
                    load = max(
                        load, float(solver._impulses.numpy()[types == PGS_CONSTRAINT_TYPE_CONTACT].max(initial=0))
                    )
                    a, z = z, a
                self.assertGreater(spins, 0)
                self.assertGreater(load, 1e-8)
                self.assertTrue(np.isfinite(a.body_q.numpy()).all())
                results.append((a.body_q.numpy(), a.joint_q.numpy(), a.joint_qd.numpy(), _loop_anchor_gap(model, a)))
            for expected, actual in zip(*results, strict=True):
                np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-6)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_cuda_solver_profiles(self):
        """Match the host's solved state and impulses for both production profiles."""

        def device_prepare(solver, state, augmented, contacts):
            enable_device_torsion(solver).prepare(state, augmented, contacts)

        for patches in (False, True):
            for velocity, regularization in ((16, 0.0), (0, 0.01)):
                options = {
                    "iterations": 16,
                    "velocity_iterations": velocity,
                    "contact_regularization": regularization,
                    "enable_bilateral_preelimination": False,
                    "sliding": 0.03,
                    "spin": 0.4,
                }
                if patches:
                    options.update(PATCH_OPTIONS)
                expected, *_ = fixture(0.01, **options)
                with patch.object(solver_feather_pgs, "prepare_torsion_rows", device_prepare):
                    actual, *_ = fixture(0.01, **options)
                for key in expected:
                    np.testing.assert_allclose(
                        actual[key], expected[key], rtol=3e-5, atol=3e-6, err_msg=f"{patches=} {velocity=} {key=}"
                    )

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_cuda_preparation_graph_changes_and_failure(self):
        """Replay changing groups and fail-stop invalid rings without host reads."""
        solver, state, augmented, contacts = synthetic_fixture(device="cuda:0")
        device = DeviceTorsionPreparation(solver, deferred_errors=True)
        initial_count = wp.clone(solver._constraint_count)
        initial_type = wp.clone(solver._row_type)
        initial_parent = wp.clone(solver._row_parent)
        output = SimpleNamespace(body_q=wp.clone(state.body_q))
        device.begin_step(state, output)

        def operation():
            wp.copy(solver._constraint_count, initial_count)
            wp.copy(solver._row_type, initial_type)
            wp.copy(solver._row_parent, initial_parent)
            device.begin_step(state, output)
            output.body_q.zero_()
            device.prepare(state, augmented, contacts)
            device.end_step(output)

        operation()
        with wp.ScopedCapture(device="cuda:0") as capture:
            operation()
        for count in (0, 15, 0, 15):
            contacts.rigid_contact_count.fill_(count)
            wp.capture_launch(capture.graph)
            device.validate()
            self.assertEqual(np.count_nonzero(device.work.spin_row.numpy() >= 0), 0 if count == 0 else 6)
        bad = initial_parent.numpy()
        bad[0, 0] = solver._articulation_constraint_max - 1
        initial_parent.assign(bad)
        wp.capture_launch(capture.graph)
        with self.assertRaisesRegex(RuntimeError, "load ring"):
            device.validate()
        np.testing.assert_array_equal(output.body_q.numpy(), state.body_q.numpy())
        self.assertTrue(np.all(solver._constraint_count.numpy() == 0))

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_cuda_captured_capacity_errors(self):
        """Graph replay must not turn an input, dense or spin overflow into success."""
        for failure in ("input", "dense", "spin"):
            solver, state, augmented, contacts = synthetic_fixture(device="cuda:0")
            device = DeviceTorsionPreparation(solver, deferred_errors=True)
            output = SimpleNamespace(body_q=wp.clone(state.body_q))
            initial_count = wp.clone(solver._constraint_count)
            device.begin_step(state, output)

            def operation(
                solver=solver,
                state=state,
                augmented=augmented,
                contacts=contacts,
                device=device,
                output=output,
                initial_count=initial_count,
            ):
                wp.copy(solver._constraint_count, initial_count)
                device.begin_step(state, output)
                output.body_q.zero_()
                device.prepare(state, augmented, contacts)
                device.end_step(output)

            operation()
            with wp.ScopedCapture(device="cuda:0") as capture:
                operation()
            if failure == "input":
                contacts.rigid_contact_count.fill_(contacts.rigid_contact_max + 1)
            elif failure == "dense":
                solver._row_dropped_dense.fill_(1)
            else:
                initial_count.fill_(solver._articulation_constraint_max)
            for _ in range(2):
                wp.capture_launch(capture.graph)
                with self.subTest(failure=failure), self.assertRaises(RuntimeError):
                    device.validate()
                np.testing.assert_array_equal(output.body_q.numpy(), state.body_q.numpy())
                self.assertTrue(np.all(solver._constraint_count.numpy() == 0))
                self.assertTrue(np.all(solver._contact_torsion_group.numpy() == -1))

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_public_api_noop_and_fail_stop(self):
        """No radius means no new state; active host torsion cannot be captured."""
        _, plain, model, state, _ = fixture(0.0, contact_torsion_device=True, iterations=16)
        plain.check_constraint_capacity()
        self.assertIsNone(getattr(plain, "_device_torsion", None))

        _, reference, model, state, contacts = fixture(0.01, iterations=16)
        with self.assertRaisesRegex(RuntimeError, "requires contact_torsion_device"):
            with wp.ScopedCapture(device=model.device):
                reference.step(state, model.state(), model.control(), contacts, 0.0025)

        _, solver, model, state, contacts = fixture(0.01, contact_torsion_device=True, iterations=16)
        output = model.state()
        solver._device_torsion.work.status.assign(np.array([16], dtype=np.int32))
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, "load ring"):
                solver.step(state, output, model.control(), contacts, 0.0025)
            with self.assertRaisesRegex(RuntimeError, "load ring"):
                solver.check_constraint_capacity()

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_cuda_cold_capture_reports_errors_until_reset(self):
        """Capture the first step of a solver that never stepped, and report a replay's error until reset."""
        _, solver, model, state, contacts = fixture(0.01, contact_torsion_device=True, iterations=16, step=False)
        self.assertEqual(solver._step, 0)
        output = model.state()
        with wp.ScopedCapture(device=model.device) as capture:
            solver.step(state, output, model.control(), contacts, 0.0025)
        wp.capture_launch(capture.graph)
        solver.check_constraint_capacity()
        count = int(solver._constraint_count.numpy()[0])
        self.assertGreater(np.count_nonzero(solver._row_type.numpy()[0, :count] == PGS_CONSTRAINT_TYPE_TORSION), 0)
        self.assertFalse(np.array_equal(output.joint_q.numpy(), state.joint_q.numpy()))

        solver._device_torsion.work.status.assign(np.array([16], dtype=np.int32))
        for _ in range(2):
            wp.capture_launch(capture.graph)
            np.testing.assert_array_equal(output.joint_q.numpy(), state.joint_q.numpy())
            np.testing.assert_array_equal(output.joint_qd.numpy(), state.joint_qd.numpy())
            with self.assertRaisesRegex(RuntimeError, "load ring"):
                solver.check_constraint_capacity()
        solver.reset(state)
        wp.capture_launch(capture.graph)
        solver.check_constraint_capacity()
        self.assertFalse(np.array_equal(output.joint_q.numpy(), state.joint_q.numpy()))

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_cuda_status_snapshot_reports_a_replay_error(self):
        """Report a replay's torsion error from a snapshot taken after it, and not from an earlier one."""
        _, solver, model, state, contacts = fixture(0.01, contact_torsion_device=True, iterations=16)
        output = model.state()
        with wp.ScopedCapture(device=model.device) as capture:
            solver.step(state, output, model.control(), contacts, 0.0025)
        wp.capture_launch(capture.graph)
        solver.snapshot_constraint_status()
        solver._device_torsion.work.status.assign(np.array([16], dtype=np.int32))
        wp.capture_launch(capture.graph)
        solver.check_constraint_capacity(snapshot=True)
        solver.snapshot_constraint_status()
        with self.assertRaisesRegex(RuntimeError, "load ring"):
            solver.check_constraint_capacity(snapshot=True)
        np.testing.assert_array_equal(output.joint_q.numpy(), state.joint_q.numpy())
        solver.reset(state)
        wp.capture_launch(capture.graph)
        solver.snapshot_constraint_status()
        solver.check_constraint_capacity(snapshot=True)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_cuda_full_solver_graph(self):
        """Match captured and eager solves through the default-off public device API."""

        for velocity, regularization in ((16, 0.0), (0, 0.01)):
            results = []
            for captured in (False, True):
                _, solver, model, a, contacts = fixture(
                    0.01,
                    contact_torsion_device=True,
                    iterations=16,
                    velocity_iterations=velocity,
                    contact_regularization=regularization,
                    enable_bilateral_preelimination=False,
                    sliding=0.01,
                    spin=0.4,
                    **PATCH_OPTIONS,
                )
                z = model.state()
                control = model.control()
                pipeline = newton.CollisionPipeline(
                    model, contact_matching="latest", reduce_contacts=False, broad_phase="nxn", rigid_contact_max=64
                )

                def pair(a=a, z=z, pipeline=pipeline, solver=solver, control=control, contacts=contacts):
                    a.clear_forces()
                    pipeline.collide(a, contacts)
                    solver.step(a, z, control, contacts, 0.0025)
                    z.clear_forces()
                    pipeline.collide(z, contacts)
                    solver.step(z, a, control, contacts, 0.0025)

                pair()
                if captured:
                    with wp.ScopedCapture(device=model.device) as capture:
                        pair()
                    for _ in range(5):
                        wp.capture_launch(capture.graph)
                        solver.check_constraint_capacity()
                else:
                    for _ in range(5):
                        pair()
                        solver.check_constraint_capacity()
                results.append((a.joint_q.numpy(), a.joint_qd.numpy(), a.body_q.numpy()))
            for eager, graph in zip(*results, strict=True):
                np.testing.assert_allclose(graph, eager, rtol=3e-5, atol=3e-6)


@unittest.skipUnless(wp.is_cuda_available(), "Contact torsion requires CUDA")
class TestTorsionRegularization(unittest.TestCase):
    def check_budget(self, result, solver):
        """Bound final shared impulses and account for their entire velocity response."""
        count = int(result["count"][0])
        impulses = result["impulses"][0, :count]
        # Compare full velocities in float64 rather than subtracting two large
        # float32 spins. Repeated in-place updates at 100rad/s accumulate several
        # velocity ULPs even though the impulse budget and linear response agree.
        response = result["Y_world"][0, :count].astype(np.float64).T @ impulses.astype(np.float64)
        np.testing.assert_allclose(
            result["v_out"],
            result["v_hat"].astype(np.float64) + response,
            rtol=8 * np.finfo(np.float32).eps,
            atol=2e-5,
        )
        self.assertTrue(solver._torsion_stats["groups"])
        for group in solver._torsion_stats["groups"]:
            load = sum(max(float(impulses[n]), 0.0) for n in group["normal_rows"])
            sliding = sum(float(np.linalg.norm(impulses[n + 1 : n + 3])) for n in group["anchor_rows"])
            spin = abs(float(impulses[group["row"]])) / group["effective_radius_m"]
            self.assertLessEqual(sliding + spin, group["mu"] * load + 2e-6)
        self.assertEqual(solver.constraint_row_watermarks()["articulation_dropped_contact_rows_high_water"], 0)

    def test_analytic_regularized_load_and_saturated_spin(self):
        """Use the solved normal impulse once, without double-regularizing spin capacity."""
        for regularization in (0.01, 0.02, 0.5):
            with self.subTest(regularization=regularization):
                result, solver, *_ = fixture(
                    0.01,
                    center_only=True,
                    spin=100.0,
                    iterations=16,
                    contact_regularization=regularization,
                )
                group = solver._torsion_stats["groups"][0]
                impulse = result["impulses"][0]
                # Two equal 0.3kg bodies close at 0.1m/s each: rigid lambda=.03Ns.
                normal = sum(float(impulse[n]) for n in group["normal_rows"])
                self.assertAlmostEqual(normal, 0.03 / (1 + regularization), delta=2e-7)
                self.assertAlmostEqual(abs(float(impulse[group["row"]])), 0.01 * 0.5 * normal, delta=2e-8)
                before = result["v_hat"].reshape(2, 6)
                after = result["v_out"].reshape(2, 6)
                self.assertLess(abs(after[0, 5] - after[1, 5]), abs(before[0, 5] - before[1, 5]))
                inertia = np.array([0.3, 0.3, 0.3, 0.00012, 0.00012, 0.00008])
                self.assertLessEqual(float(np.sum(after**2 * inertia)), float(np.sum(before**2 * inertia)) + 2e-6)
                self.check_budget(result, solver)

    def test_final_load_budget_across_iterations(self):
        """Correct stale sliding and spin when distributed normal loads shrink."""
        for regularization in (0.01, 0.02, 0.5):
            for iterations in (1, 2, 4, 16, 64):
                for spin, sliding in ((0.0, 1.0), (10.0, 1.0), (100.0, 0.0)):
                    with self.subTest(reg=regularization, iterations=iterations, spin=spin):
                        result, solver, *_ = fixture(
                            0.0075,
                            iterations=iterations,
                            contact_regularization=regularization,
                            spin=spin,
                            sliding=sliding,
                            **PATCH_OPTIONS,
                        )
                        self.check_budget(result, solver)
                        count = int(result["count"][0])
                        types = result["row_type"][0, :count]
                        weights = solver._row_w.numpy()[0, :count]
                        normal_rows = types == 0
                        self.assertTrue(np.any(normal_rows))
                        np.testing.assert_allclose(weights[normal_rows], 1 / (1 + regularization), atol=1e-7)
                        # Torsion itself is a dry-friction row, not another soft normal.
                        np.testing.assert_array_equal(weights[types == 7], 1.0)

    def test_zero_controls_and_no_load(self):
        """Preserve excluded/radius-zero controls and forbid spin without compressive load."""
        for regularization in (0.0, 0.01, 0.02):
            base, *_ = fixture(0.0, contact_regularization=regularization)
            excluded, *_ = fixture(0.01, contact_regularization=regularization, torsion_shapes=())
            for name in base:
                np.testing.assert_array_equal(base[name], excluded[name], err_msg=name)
            for options in ({"closing": 0.0}, {"mu": 0.0}):
                result, *_ = fixture(0.01, contact_regularization=regularization, center_only=True, **options)
                self.assertLess(np.abs(result["impulses"][result["row_type"] == 7]).max(initial=0), 1e-9)

    def test_substep_force_export_and_release(self):
        """Export the actual solved normal load over dt and forget torque on release."""
        for substeps in (1, 2, 4):
            dt = 0.005 / substeps
            result, solver, model, initial, contacts = fixture(
                0.01,
                center_only=True,
                spin=100.0,
                dt=dt,
                contact_regularization=0.02,
                **PATCH_OPTIONS,
            )
            count = int(contacts.rigid_contact_count.numpy()[0])
            forces = result["contact_f"][:count, :3]
            normals = contacts.rigid_contact_normal.numpy()[:count]
            slots = solver._contact_slot.numpy()[:count]
            for contact, slot in enumerate(slots):
                if slot >= 0:
                    self.assertAlmostEqual(
                        abs(float(np.dot(forces[contact], normals[contact]))) * dt,
                        float(result["impulses"][0, slot]),
                        delta=2e-7,
                    )
            contacts.rigid_contact_count.zero_()
            solver.step(initial, model.state(), model.control(), contacts, dt)
            self.assertEqual(solver._torsion_stats["rows"], 0)


class TestTorsionVelocityAdmission(unittest.TestCase):
    def test_end_gap_retires_spin_but_preserves_target_and_carried_impulse(self):
        """Retire separated spin groups without deleting their required velocity correction."""
        for initial_gap, end_gap in ((-0.001, -0.001), (-0.001, 0.0001), (0.0001, -0.0001), (0.0001, 0.0002)):
            with self.subTest(initial_gap=initial_gap, end_gap=end_gap):
                count = wp.array([2], dtype=int, device="cpu")
                types = wp.array([[0, 7]], dtype=int, device="cpu")
                groups = wp.array([[1, -1]], dtype=int, device="cpu")
                phi = wp.array([[initial_gap, 0.0]], dtype=float, device="cpu")
                target = wp.array([[0.0, 0.25]], dtype=float, device="cpu")
                velocity = wp.array([(end_gap - initial_gap) / 0.01], dtype=float, device="cpu")
                dofs = wp.array([1], dtype=int, device="cpu")
                indices = wp.array([[0]], dtype=int, device="cpu")
                jacobian = wp.array(np.array([[[1.0], [0.0]]], np.float32), device="cpu")
                rhs = wp.array([[3.0, 9.0]], dtype=float, device="cpu")
                wp.launch(
                    contact_torsion._prepare_velocity_torsion_rows,
                    dim=2,
                    inputs=[count, 2, types, groups, phi, target, velocity, dofs, indices, jacobian, 0.01],
                    outputs=[rhs],
                    device="cpu",
                )
                retired = initial_gap > 1e-6 and end_gap > 1e-6
                self.assertEqual(int(groups.numpy()[0, 1]), contact_torsion._TORSION_SPIN_RETIRED if retired else -1)
                np.testing.assert_array_equal(rhs.numpy(), [[3.0, -0.25]])
                # Re-admission must clear a previous retirement, including when
                # these same buffers are reused by a captured graph replay.
                phi.assign(np.array([[-0.001, 0.0]], dtype=np.float32))
                wp.launch(
                    contact_torsion._prepare_velocity_torsion_rows,
                    dim=2,
                    inputs=[count, 2, types, groups, phi, target, velocity, dofs, indices, jacobian, 0.01],
                    outputs=[rhs],
                    device="cpu",
                )
                np.testing.assert_array_equal(groups.numpy(), [[1, -1]])


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestTorsionVelocitySolve(unittest.TestCase):
    def test_rebound_preserves_loaded_sliding_when_spin_retires(self):
        """Preserve ordinary sliding friction on an admitted rebounding contact."""
        for separation in (0.0, 5.0e-6):
            for restitution in (0.0, 0.5):
                with self.subTest(separation=separation, restitution=restitution):
                    options = dict(
                        separation=separation,
                        restitution=restitution,
                        closing=1.0,
                        sliding=1.0,
                        spin=0.0,
                        sat=True,
                        center_only=True,
                        iterations=64,
                        velocity_iterations=16,
                        enable_bilateral_preelimination=False,
                        **PATCH_OPTIONS,
                    )
                    baseline, *_ = fixture(0.0, **options)
                    actual, solver, *_ = fixture(0.01, **options)
                    normals = actual["row_type"] == 0
                    tangent = actual["row_type"] == 2
                    np.testing.assert_allclose(actual["impulses"][normals], baseline["impulses"][normals], atol=2e-6)
                    self.assertGreater(float(np.linalg.norm(baseline["impulses"][tangent])), 0.01)
                    np.testing.assert_allclose(actual["impulses"][tangent], baseline["impulses"][tangent], atol=2e-6)
                    np.testing.assert_allclose(actual["body_qd"], baseline["body_qd"], atol=3e-5)
                    self._assert_budget(actual, solver)

    def _assert_budget(self, result, solver):
        """Check the shared final-load budget and nonnegative normal impulses."""
        for group in solver._torsion_stats["groups"]:
            impulse = result["impulses"][group["world"]]
            normal = sum(max(float(impulse[n]), 0.0) for n in group["normal_rows"])
            sliding = sum(float(np.linalg.norm(impulse[n + 1 : n + 3])) for n in group["anchor_rows"])
            spin = abs(float(impulse[group["row"]])) / group["effective_radius_m"]
            self.assertLessEqual(sliding + spin, group["mu"] * normal + 3e-6)

    def test_spin_opposition_and_final_budget(self):
        """Dissipate angular slip and respect final normal load in each velocity count."""
        for velocity in (1, 4, 16):
            for spin, sliding in ((1.0, 0.0), (100.0, 0.0), (10.0, 1.0)):
                with self.subTest(velocity=velocity, spin=spin, sliding=sliding):
                    result, solver, *_ = fixture(
                        0.01,
                        iterations=16,
                        velocity_iterations=velocity,
                        spin=spin,
                        sliding=sliding,
                        center_only=sliding == 0,
                        enable_bilateral_preelimination=False,
                        **PATCH_OPTIONS,
                    )
                    self._assert_budget(result, solver)
                    self.assertTrue(np.isfinite(result["body_qd"]).all())
                    if sliding == 0:
                        self.assertLess(np.max(np.abs(result["body_qd"][:, 5])), spin)

    def test_postpass_accumulated_impulse_response_and_position_split(self):
        """Apply delta impulses once while retaining the biased position trajectory."""
        original = fixture(
            0.01,
            spin=100.0,
            closing=0.1,
            center_only=True,
            iterations=16,
            enable_bilateral_preelimination=False,
            **PATCH_OPTIONS,
        )
        result, solver, *_ = fixture(
            0.01,
            spin=100.0,
            closing=0.1,
            center_only=True,
            iterations=16,
            velocity_iterations=4,
            enable_bilateral_preelimination=False,
            **PATCH_OPTIONS,
        )
        baseline = original[0]
        np.testing.assert_array_equal(result["body_q"], baseline["body_q"])
        count = int(result["count"][0])
        delta = result["impulses"][0, :count] - baseline["impulses"][0, :count]
        np.testing.assert_allclose(
            result["v_out"] - baseline["v_out"], result["Y_world"][0, :count].T @ delta, atol=2e-5
        )
        self._assert_budget(result, solver)

    def test_zero_load_and_speculation(self):
        """Remove torque when normal load or current touching support disappears."""
        for options in ({"closing": 0.0}, {"closing": -0.1}, {"separation": 0.004}, {"mu": 0.0}):
            with self.subTest(options=options):
                result, solver, *_ = fixture(
                    0.01, velocity_iterations=4, enable_bilateral_preelimination=False, **PATCH_OPTIONS, **options
                )
                active = result["row_type"] == 7
                self.assertLess(np.abs(result["impulses"][active]).max(initial=0), 2e-7)
                self._assert_budget(result, solver)

    def test_carried_spin_is_refunded_when_postpass_retires_group(self):
        """Undo carried spin through its response column when final support is separated."""
        original, solver, model, state, contacts = fixture(
            0.01, spin=100.0, center_only=True, velocity_iterations=4, **PATCH_OPTIONS
        )
        group = solver._torsion_stats["groups"][0]
        count = int(solver._constraint_count.numpy()[0])
        before_lambda = solver._impulses.numpy()[0, :count].copy()
        self.assertGreater(abs(float(before_lambda[group["row"]])), 1e-8)
        before_velocity = solver._v_out.numpy().copy()
        phi = solver._phi.numpy()
        phi[0, group["normal_rows"]] = 0.01
        solver._phi.assign(phi)
        solver._v_out_snap.zero_()
        solver._conclude_matrix_free_position_problem(0.0025)
        self.assertEqual(float(solver._row_mu.numpy()[0, group["row"]]), group["mu"])
        self.assertEqual(
            int(solver._contact_torsion_group.numpy()[0, group["row"]]), contact_torsion._TORSION_SPIN_RETIRED
        )
        # Eligibility changes the admissible set, not the already-applied load.
        np.testing.assert_array_equal(solver._impulses.numpy()[0, :count], before_lambda)
        solver._run_matrix_free_velocity_post_solve()
        after_lambda = solver._impulses.numpy()[0, :count]
        self.assertLess(abs(float(after_lambda[group["row"]])), 1e-9)
        np.testing.assert_allclose(
            solver._v_out.numpy() - before_velocity,
            solver._Y_world.numpy()[0, :count].T @ (after_lambda - before_lambda),
            atol=2e-5,
        )
        # A new position solve must restore admission rather than inherit the
        # prior velocity-pass retirement in a reused membership buffer.
        output = model.state()
        solver.step(state, output, model.control(), contacts, 0.0025)
        group = solver._torsion_stats["groups"][0]
        self.assertEqual(int(solver._contact_torsion_group.numpy()[0, group["row"]]), -1)
        self.assertGreater(abs(float(solver._impulses.numpy()[0, group["row"]])), 1e-8)
        np.testing.assert_allclose(output.body_qd.numpy(), original["body_qd"], atol=2e-5)

    def test_internal_timestep_export_uses_final_linear_impulse(self):
        """Export final normal impulses in newtons at each internal substep duration."""
        for dt in (0.00125, 0.0025, 0.005):
            with self.subTest(dt=dt):
                result, solver, model, initial, contacts = fixture(
                    0.01, spin=100.0, center_only=True, dt=dt, velocity_iterations=4, **PATCH_OPTIONS
                )
                observables = solver.observables({SolverBase.ObservableKind.CONTACT_F}, contacts=contacts)
                for _ in range(4):
                    output = model.state()
                    solver.step(initial, output, model.control(), contacts, dt, observables=observables)
                    row = int(solver._contact_slot.numpy()[0])
                    normal = contacts.rigid_contact_normal.numpy()[0]
                    force = observables.contact_f.numpy()[0]
                    expected = max(float(solver._impulses.numpy()[0, row]), 0.0) / dt
                    self.assertAlmostEqual(abs(float(np.dot(force[:3], normal))), expected, delta=1e-4)
                    # Torque slots are deliberately placeholders, not solved spin wrench.
                    np.testing.assert_array_equal(force[3:], 0.0)
                    np.testing.assert_allclose(output.body_qd.numpy(), result["body_qd"], atol=2e-5)

    def test_selected_out_torsion_matches_zero_radius(self):
        """Keep the torsion-zero velocity path identical when no shapes are selected."""
        for velocity in (0, 4):
            expected, *_ = fixture(0.0, velocity_iterations=velocity, **PATCH_OPTIONS)
            actual, *_ = fixture(0.01, velocity_iterations=velocity, torsion_shapes=(), **PATCH_OPTIONS)
            for key in expected:
                np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)

    def test_runtime_incompatible_mutation_rejected(self):
        """Reject unsupported mutation before stepping an allocated torsion solver."""
        _, solver, model, state, contacts = fixture(0.01, velocity_iterations=4)
        solver._contact_warmstart = True
        with self.assertRaisesRegex(ValueError, "warmstart"):
            solver.step(state, model.state(), model.control(), contacts, 0.0025)


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestFinalPatchBudget(unittest.TestCase):
    def test_final_budget_and_velocity_response(self):
        """Bound final traction plus spin and apply every correction to velocity."""
        for kernel in ("loop", "tiled_row"):
            for iterations in (1, 2, 4, 8, 64):
                for spin, sliding in ((0.0, 1.0), (10.0, 1.0), (100.0, 0.0)):
                    with self.subTest(kernel=kernel, iterations=iterations, spin=spin, sliding=sliding):
                        result, solver, *_ = fixture(
                            0.0075,
                            friction_anchor_beta=0.2,
                            pgs_kernel=kernel,
                            iterations=iterations,
                            closing=0.1,
                            spin=spin,
                            sliding=sliding,
                            contact_shared_anchor=False,
                            contact_friction_shared_anchor=False,
                        )
                        count = int(result["count"][0])
                        response = result["Y_world"][0, :count].T @ result["impulses"][0, :count]
                        np.testing.assert_allclose(result["v_out"] - result["v_hat"], response, atol=2e-5)
                        groups = solver._torsion_stats["groups"]
                        self.assertTrue(groups, "The regression must exercise coupled patch/torsion rows")
                        for group in groups:
                            impulse = result["impulses"][group["world"]]
                            normal = sum(max(float(impulse[n]), 0.0) for n in group["normal_rows"])
                            traction = sum(float(np.linalg.norm(impulse[n + 1 : n + 3])) for n in group["anchor_rows"])
                            spin_used = abs(float(impulse[group["row"]])) / group["effective_radius_m"]
                            self.assertLessEqual(traction + spin_used, group["mu"] * normal + 2e-6)


_LINK_ROWS = 40
_LINK_DOFS = 12
_LINK_RADIUS = 0.01
# Tangent rows allow more traction than the group budget, so the sweep must rescale them.
_LINK_TANGENT_MU = 1.0
_LINK_GROUP_MU = 0.2


class TestTorsionLinkSharedMemory(unittest.TestCase):
    """Admit the shared row lists only within the kernel's static shared-memory limit."""

    def test_estimator_counts_lists(self):
        """Count two short or int list entries per dense row."""
        for rows, entry_bytes in ((1024, 2), (8192, 2), (32767, 2), (32768, 4)):
            with self.subTest(rows=rows):
                scan = _estimate_mfgs_shared_memory(rows, 512, 32, metadata_bytes=0, torsion_links=False)
                linked = _estimate_mfgs_shared_memory(rows, 512, 32, metadata_bytes=0, torsion_links=True)
                self.assertEqual(linked - scan, 2 * entry_bytes * rows)
        # Static arrays of the shape reported to exceed sm_75's limit once the lists were added.
        self.assertEqual(_estimate_mfgs_shared_memory(8192, 512, 32, metadata_bytes=0, torsion_links=True), 67840)

    def test_selection(self):
        """Fall back to the scan when the lists, plus resident metadata, exceed the limit."""
        options = {"has_drive_rows": False, "fuse_vel_limits": False}
        device_limit = 101376
        self.assertTrue(_use_mfgs_torsion_links(1024, 1024, 30, device_limit, shared_metadata=False, **options))
        self.assertFalse(_use_mfgs_torsion_links(8192, 512, 32, device_limit, shared_metadata=False, **options))
        self.assertFalse(_use_mfgs_torsion_links(64, 16, 12, 0, shared_metadata=False, **options))
        # 128 + 4 * (16 + 8 + rows) + 4 * rows plus the 2-KiB Warp reserve is 48 KiB at rows = 5860.
        self.assertTrue(_use_mfgs_torsion_links(5860, 8, 16, device_limit, shared_metadata=False, **options))
        self.assertFalse(_use_mfgs_torsion_links(5861, 8, 16, device_limit, shared_metadata=False, **options))
        self.assertFalse(_use_mfgs_torsion_links(5860, 8, 16, 48 * 1024 - 1, shared_metadata=False, **options))
        self.assertFalse(_use_mfgs_torsion_links(2048, 0, 16, device_limit, shared_metadata=True, **options))
        self.assertTrue(_use_mfgs_torsion_links(2048, 0, 16, device_limit, shared_metadata=False, **options))


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestTorsionSweepLinks(unittest.TestCase):
    """Run the linked and scanning sweeps on identical inputs and require bitwise equal outputs."""

    def test_boundary_layouts_match_scan(self):
        """Match the scan bitwise for empty, first-row, last-row, over-budget, invalid and retired groups."""
        layouts = _link_layouts()
        inputs = _link_kernel_inputs(layouts)
        for shared_metadata in (True, False):
            for friction_start in (0, 3):
                results = {}
                for linked in (False, True):
                    kernel = _get_pgs_solve_mf_gs_kernel(
                        _LINK_ROWS,
                        1,
                        _LINK_DOFS,
                        wp.get_device("cuda:0").arch,
                        shared_metadata=shared_metadata,
                        has_drive_rows=False,
                        has_dense_velocity_limit_rows=False,
                        contact_torsion=True,
                        torsion_links=linked,
                    )
                    self.assertEqual(kernel.key.endswith("_torsion_links"), linked)
                    results[linked] = _link_launch(kernel, inputs, friction_start)
                with self.subTest(shared_metadata=shared_metadata, friction_start=friction_start):
                    for name in ("impulses", "v_out"):
                        np.testing.assert_array_equal(
                            results[True][name].view(np.uint32), results[False][name].view(np.uint32)
                        )
                    if friction_start == 0:
                        _assert_link_layouts_exercised(self, layouts, results[True]["impulses"])

    def test_solver_selects_links_and_matches_fallback(self):
        """Select the lists at a compact capacity and match the forced fallback bitwise."""
        options = {
            "friction_anchor_beta": 0.2,
            "contact_shared_anchor": False,
            "contact_friction_shared_anchor": False,
            "iterations": 8,
            "spin": 10.0,
            "sliding": 1.0,
        }
        linked, solver, *_ = fixture(0.0075, **options)
        self.assertTrue(solver._mfgs_torsion_links)
        self.assertTrue(solver._pgs_solve_mf_gs_kernel.key.endswith("_torsion_links"))
        with patch.object(solver_feather_pgs, "_use_mfgs_torsion_links", return_value=False):
            scanned, solver, *_ = fixture(0.0075, **options)
        self.assertFalse(solver._mfgs_torsion_links)
        self.assertTrue(solver._pgs_solve_mf_gs_kernel.key.endswith("_torsion"))
        self.assertTrue(np.any(linked["row_type"] == PGS_CONSTRAINT_TYPE_TORSION))
        for name, value in linked.items():
            np.testing.assert_array_equal(np.asarray(value), np.asarray(scanned[name]), err_msg=name)

    def test_largest_admitted_capacity_compiles(self):
        """Compile the linked kernel at the largest capacity the selection admits."""
        options = {"has_drive_rows": False, "fuse_vel_limits": False}
        device = wp.get_device("cuda:0")
        limit = device.max_shared_memory_per_block
        self.assertTrue(_use_mfgs_torsion_links(5860, 8, 16, limit, shared_metadata=False, **options))
        kernel = _get_pgs_solve_mf_gs_kernel(
            5860,
            8,
            16,
            device.arch,
            shared_metadata=False,
            has_drive_rows=False,
            has_dense_velocity_limit_rows=False,
            contact_torsion=True,
            torsion_links=True,
        )
        wp.load_module(kernel.module, device=device)

    def test_large_capacity_falls_back(self):
        """Fall back to the scan when the lists exceed the static limit, and still solve torsion."""
        result, solver, *_ = fixture(
            0.0075, articulation_constraint_max=8192, free_body_constraint_max=512, spin=10.0, iterations=8
        )
        self.assertFalse(solver._mfgs_torsion_links)
        self.assertTrue(solver._pgs_solve_mf_gs_kernel.key.endswith("_torsion"))
        count = int(result["count"][0])
        torsion_rows = result["row_type"][0, :count] == PGS_CONSTRAINT_TYPE_TORSION
        self.assertTrue(np.any(torsion_rows))
        self.assertTrue(np.any(result["impulses"][0, :count][torsion_rows] != 0.0))
        for name in ("body_q", "body_qd", "impulses"):
            self.assertTrue(np.all(np.isfinite(result[name])), name)


class TestMfgsRowCaches(unittest.TestCase):
    """Solve with and without the per-launch friction-pair and row caches and require bitwise equal outputs."""

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_cached_solve_matches_uncached(self):
        """Match every fixture output bitwise with the caches selected and disabled, for both metadata modes."""
        options = {**PATCH_OPTIONS, "iterations": 8, "spin": 10.0, "sliding": 1.0}
        for resident in (True, False):
            with (
                self.subTest(resident_metadata=resident),
                patch.object(solver_feather_pgs, "_use_resident_mfgs_metadata", return_value=resident),
            ):
                cached, solver, *_ = fixture(0.0075, **options)
                self.assertTrue(solver._mfgs_pair_cache)
                self.assertIn("_paircache", solver._pgs_solve_mf_gs_kernel.key)
                with patch.object(solver_feather_pgs, "_use_mfgs_pair_cache", return_value=False):
                    plain, solver, *_ = fixture(0.0075, **options)
                self.assertFalse(solver._mfgs_pair_cache)
                self.assertNotIn("_paircache", solver._pgs_solve_mf_gs_kernel.key)
                self.assertTrue(np.any(cached["row_type"] == PGS_CONSTRAINT_TYPE_TORSION))
                for name, value in cached.items():
                    np.testing.assert_array_equal(np.asarray(value), np.asarray(plain[name]), err_msg=name)

    def test_selection_requires_static_fit_and_resident_worlds(self):
        """Select the caches only within the static shared-memory bound and the estimated resident world count."""
        options = {"shared_metadata": False, "torsion_links": True, "has_drive_rows": False, "fuse_vel_limits": False}
        limit = 101376
        self.assertTrue(_use_mfgs_pair_cache(1024, 4096, 30, limit, world_count=123, sm_count=188, **options))
        # Two cached blocks fit per SM, so 1024 worlds would queue.
        self.assertFalse(_use_mfgs_pair_cache(1024, 4096, 30, limit, world_count=1024, sm_count=188, **options))
        self.assertFalse(_use_mfgs_pair_cache(4096, 4096, 30, limit, world_count=1, sm_count=188, **options))
        self.assertFalse(_use_mfgs_pair_cache(1024, 4096, 30, 32 * 1024, world_count=1, sm_count=188, **options))
        self.assertTrue(_use_mfgs_pair_cache(2, 8, 2, limit, world_count=1, sm_count=1, **options))


def _link_layouts():
    """Return per-world (row types, parents, torsion groups, active rows, loaded torsion rows)."""

    def world(active):
        return (
            np.full(_LINK_ROWS, PGS_CONSTRAINT_TYPE_JOINT_LIMIT, np.int32),
            np.full(_LINK_ROWS, -1, np.int32),
            np.full(_LINK_ROWS, -1, np.int32),
            active,
        )

    def contact(types, parents, groups, row, spin, tangents=True):
        types[row] = PGS_CONSTRAINT_TYPE_CONTACT
        groups[row] = spin
        if tangents:
            types[row + 1 : row + 3] = PGS_CONSTRAINT_TYPE_FRICTION
            parents[row + 1 : row + 3] = row

    def torsion(types, groups, row, retired=False):
        types[row] = PGS_CONSTRAINT_TYPE_TORSION
        if retired:
            groups[row] = _TORSION_SPIN_RETIRED

    layouts = []
    # No torsion rows.
    types, parents, groups, active = world(_LINK_ROWS)
    for row in range(0, 36, 3):
        contact(types, parents, groups, row, -1)
    layouts.append((types, parents, groups, active, []))
    # Torsion row first, every member after it.
    types, parents, groups, active = world(_LINK_ROWS)
    torsion(types, groups, 0)
    for row in range(1, 37, 3):
        contact(types, parents, groups, row, 0)
    layouts.append((types, parents, groups, active, [0]))
    # Torsion row last at full capacity, after a normal-only member.
    types, parents, groups, active = world(_LINK_ROWS)
    for row in range(0, 36, 3):
        contact(types, parents, groups, row, _LINK_ROWS - 1)
    contact(types, parents, groups, 37, _LINK_ROWS - 1, tangents=False)
    torsion(types, groups, _LINK_ROWS - 1)
    layouts.append((types, parents, groups, active, [_LINK_ROWS - 1]))
    # Interleaved groups, empty and retired groups, invalid targets, and normal-only members on the
    # last active rows; inactive rows hold a torsion group that both sweeps must ignore.
    types, parents, groups, active = world(31)
    torsion(types, groups, 0)
    torsion(types, groups, 1)
    torsion(types, groups, 2)
    torsion(types, groups, 3, retired=True)
    targets = [0, 1, 0, 1, -1, 3, 4, 35]
    for row, spin in zip(range(4, 28, 3), targets, strict=True):
        contact(types, parents, groups, row, spin)
    contact(types, parents, groups, 28, 1, tangents=False)
    contact(types, parents, groups, 29, 0, tangents=False)
    contact(types, parents, groups, 30, 1, tangents=False)
    torsion(types, groups, 34)
    contact(types, parents, groups, 35, 34)
    layouts.append((types, parents, groups, active, [0, 1]))
    return layouts


def _link_kernel_inputs(layouts):
    rng = np.random.default_rng(7)
    worlds = len(layouts)
    types = np.stack([layout[0] for layout in layouts])
    parents = np.stack([layout[1] for layout in layouts])
    groups = np.stack([layout[2] for layout in layouts])
    counts = np.array([layout[3] for layout in layouts], np.int32)
    J = rng.normal(size=(worlds, _LINK_ROWS, _LINK_DOFS)).astype(np.float32)
    Y = (0.5 * J).astype(np.float32)
    diag = np.einsum("wrd,wrd->wr", J, Y).astype(np.float32) + np.float32(1.0e-3)
    rhs = rng.normal(scale=0.5, size=(worlds, _LINK_ROWS)).astype(np.float32)
    # Approaching normals load the groups; strong tangential slip saturates their tangent rows.
    rhs[types == PGS_CONSTRAINT_TYPE_CONTACT] = -np.abs(rhs[types == PGS_CONSTRAINT_TYPE_CONTACT]) - 0.5
    rhs[types == PGS_CONSTRAINT_TYPE_FRICTION] *= 8.0
    mu = np.zeros((worlds, _LINK_ROWS), np.float32)
    mu[types == PGS_CONSTRAINT_TYPE_FRICTION] = _LINK_TANGENT_MU
    mu[types == PGS_CONSTRAINT_TYPE_TORSION] = _LINK_GROUP_MU
    return {
        "worlds": worlds,
        "counts": counts,
        "types": types,
        "parents": parents,
        "groups": groups,
        "J": J,
        "Y": Y,
        "diag": diag,
        "rhs": rhs,
        "mu": mu,
        "v": rng.normal(size=worlds * _LINK_DOFS).astype(np.float32),
    }


def _link_launch(kernel, inputs, friction_start):
    worlds = inputs["worlds"]

    def arr(value, dtype=float):
        return wp.array(value, dtype=dtype, device="cuda:0")

    def zeros(shape, dtype=float):
        return wp.zeros(shape, dtype=dtype, device="cuda:0")

    impulses = zeros((worlds, _LINK_ROWS))
    v_out = arr(inputs["v"])
    dense = (worlds, _LINK_ROWS)
    wp.launch_tiled(
        kernel,
        dim=[worlds],
        inputs=[
            zeros(1, int),
            zeros(1, int),
            worlds,
            0,
            arr(inputs["counts"], int),
            zeros((worlds, 2), int),
            zeros(worlds, int),
            arr(np.arange(worlds * _LINK_DOFS, dtype=np.int32).reshape(worlds, _LINK_DOFS), int),
            arr(inputs["rhs"]),
            arr(inputs["diag"]),
            arr(np.ones(dense, np.float32)),
            impulses,
            arr(inputs["J"]),
            arr(inputs["Y"]),
            arr(inputs["types"], int),
            arr(inputs["parents"], int),
            arr(inputs["mu"]),
            arr(inputs["groups"], int),
            arr(np.full(dense, _LINK_RADIUS, np.float32)),
            zeros(dense),
            zeros(dense),
            zeros(dense),
            zeros(dense),
            zeros((1, 1)),
            zeros(worlds, int),
            zeros(worlds, int),
            zeros((worlds, 4), int),
            zeros((worlds, 1)),
            zeros((worlds, 1, 6)),
            zeros((worlds, 1, 6)),
            zeros((worlds, 1, 6)),
            zeros((worlds, 1, 6)),
            zeros((worlds, 1)),
            zeros((worlds, 1)),
            zeros((1, 1)),
            zeros((1, 1)),
            6,
            1.0,
            0,
            0,
            friction_start,
            0,
            0,
        ],
        outputs=[v_out],
        block_dim=32,
        device="cuda:0",
    )
    return {"impulses": impulses.numpy(), "v_out": v_out.numpy()}


def _assert_link_layouts_exercised(test, layouts, impulses):
    """Require loaded groups within budget, a rescaled group, a spin impulse and idle retired rows."""
    rescaled = spun = False
    for world, (types, parents, groups, active, spins) in enumerate(layouts):
        lam = impulses[world]
        torsion_rows = [row for row in range(active) if types[row] == PGS_CONSTRAINT_TYPE_TORSION]
        if not spins:
            test.assertFalse(np.any(lam[torsion_rows]))
        for spin in spins:
            members = [n for n in range(active) if types[n] == PGS_CONSTRAINT_TYPE_CONTACT and groups[n] == spin]
            load = sum(max(float(lam[n]), 0.0) for n in members)
            sliding = sum(
                float(np.hypot(lam[n + 1], lam[n + 2]))
                for n in members
                if n + 2 < active and parents[n + 1] == n and parents[n + 2] == n
            )
            budget = _LINK_GROUP_MU * load
            test.assertGreater(load, 0.0)
            test.assertLessEqual(sliding + abs(float(lam[spin])) / _LINK_RADIUS, budget * (1.0 + 1.0e-5) + 1.0e-6)
            rescaled |= sliding > 0.99 * budget
            spun |= abs(float(lam[spin])) / _LINK_RADIUS > 0.5 * budget
        retired = [row for row in torsion_rows if groups[row] == _TORSION_SPIN_RETIRED]
        test.assertFalse(np.any(lam[retired]))
    test.assertTrue(rescaled, "No group exceeded its friction budget")
    test.assertTrue(spun, "No group carried a spin impulse")


_PROPAGATION_TORSION = {"articulated_contact_response": "propagation", "contact_torsion_device": True}
_CENTER_SPIN = {"center_only": True, "spin": 100.0, "contact_friction_position_iterations": 8}
_CORNER_SPIN = {"sat": True, "spin": 10.0, "sliding": 0.2, "contact_friction_position_iterations": 8}


@unittest.skipUnless(wp.is_cuda_available(), "Contact torsion requires CUDA")
class TestPropagationTorsion(unittest.TestCase):
    """Contact torsion rows on the plain propagation response, against the immediate response."""

    def test_rows_take_the_propagation_route_and_host_spares(self):
        """Host each group's row in its last member's spare, keep other spares inert and match immediate's groups."""
        for options in ({"sat": True}, {"sat": True, **PATCH_OPTIONS}):
            with self.subTest(patches=bool(options.get("friction_anchor_beta"))):
                _, reference, *_ = fixture(0.01, spin=10.0, **options)
                _, solver, _model, _state, contacts = fixture(0.01, spin=10.0, **options, **_PROPAGATION_TORSION)
                count = int(contacts.rigid_contact_count.numpy()[0])
                self.assertGreater(count, 1)
                path = solver._contact_path.numpy()[:count]
                self.assertTrue(np.all(path == 2))
                self.assertTrue(np.all(solver._contact_mf_origin.numpy()[:count] == 0))
                rows = _propagation_torsion_rows(solver)
                self.assertEqual(len(rows), _row_count(reference, PGS_CONSTRAINT_TYPE_TORSION))
                self.assertEqual(len(rows), 1)
                spare_rows = _spare_rows(solver, count)
                row_type = solver._propagation_row_type.numpy()[0]
                body_a = solver._propagation_body_a.numpy()[0]
                J_a = solver._propagation_J_a.numpy()[0]
                J_b = solver._propagation_J_b.numpy()[0]
                normals = -contacts.rigid_contact_normal.numpy()[:count]
                slots = solver._contact_slot.numpy()[:count]
                for spin, members in rows.items():
                    contact_of = {int(slots[c]): c for c in range(count)}
                    host = max(contact_of[n] for n in members)
                    self.assertEqual(spin, spare_rows[host])
                    lead = contact_of[int(solver._propagation_row_parent.numpy()[0, spin])]
                    np.testing.assert_array_equal(J_a[spin, :3], 0.0)
                    np.testing.assert_allclose(J_a[spin, 3:], normals[lead], atol=1e-7)
                    np.testing.assert_allclose(J_b[spin, 3:], -normals[lead], atol=1e-7)
                    self.assertEqual(sorted(members), sorted(int(slots[c]) for c in range(count)))
                for row in set(spare_rows.values()) - set(rows):
                    self.assertEqual(row_type[row], PGS_CONSTRAINT_TYPE_TORSION)
                    self.assertEqual(body_a[row], -1)
                    np.testing.assert_array_equal(J_a[row], 0.0)
                self.assertEqual(
                    int(solver._propagation_constraint_count.numpy()[0]),
                    int(solver._contact_slots_needed.numpy()[:count].sum()),
                )

    def test_free_rigid_contacts_reserve_no_spare(self):
        """Give free-rigid contacts, which immediate solves without torsion, no spare and no torsion row."""
        b = newton.ModelBuilder()
        b.add_ground_plane()
        body = b.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0099), wp.quat_identity()))
        b.add_shape_box(body, hx=0.01, hy=0.01, hz=0.01)
        model = b.finalize(device="cuda:0")
        set_torsion_radius(model, 0.01)
        solver = SolverFeatherPGS(
            model, solve_mode="matrix_free", friction_anchor_beta=0.0, iterations=8, **_PROPAGATION_TORSION
        )
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        state = model.state()
        pipeline.collide(state, contacts)
        solver.step(state, model.state(), model.control(), contacts, 0.0025)
        count = int(contacts.rigid_contact_count.numpy()[0])
        self.assertGreater(count, 0)
        self.assertTrue(np.all(solver._contact_path.numpy()[:count] == 2))
        self.assertTrue(np.all(solver._contact_mf_origin.numpy()[:count] == 1))
        self.assertTrue(np.all(solver._contact_slots_needed.numpy()[:count] == 3))
        self.assertEqual(int(solver._propagation_torsion_head.numpy()[0]), -1)

    def test_budget_holds_after_every_sweep(self):
        """Keep each torsion impulse within its group's residual sliding budget after every plain sweep."""
        for patches in ({}, PATCH_OPTIONS):
            # A central witness leaves the spin to torsion; spinning corner witnesses slide on the whole budget.
            for case, options in (("center", _CENTER_SPIN), ("corners", _CORNER_SPIN)):
                with self.subTest(patches=bool(patches), case=case):
                    sweeps = _record_propagation_sweeps(0.01, **options, **patches)
                    self.assertTrue(any(sweep["offset"] < sweep["friction_start"] for sweep in sweeps))
                    saturated, sliding_saturated = _assert_torsion_budgets(self, sweeps)
                    if case == "center":
                        self.assertTrue(saturated, "No torsion row reached its residual budget")
                    else:
                        self.assertTrue(sliding_saturated, "No group's sliding pairs used the whole budget")

    def test_sweep_checks_catch_an_unbounded_torsion_row(self):
        """Trip the budget checks with the torsion-off sweep, which solves torsion rows as unbounded rows."""

        def unbounded(solver):
            solver._pgs_solve_propagation_kernel = _get_pgs_solve_propagation_contact_kernel(
                solver._propagation_max_constraints,
                str(solver.model.device.arch),
                worlds_per_block=solver._propagation_gs_worlds_per_block,
            )

        sweeps = _record_propagation_sweeps(0.01, on_solver=unbounded, **_CENTER_SPIN)
        with self.assertRaises(AssertionError):
            _assert_torsion_budgets(self, sweeps)

    def test_spin_decelerates_at_the_coulomb_rate_and_sticks(self):
        """Decelerate a sliding spin by r mu N / I and hold a slow spin, on both responses."""
        inertia_z = 0.00008
        for response in ("immediate", "propagation"):
            with self.subTest(response=response):
                options = {"articulated_contact_response": response, "contact_torsion_device": True}
                result, solver, *_ = fixture(0.01, spin=100.0, center_only=True, **options)
                if response == "immediate":
                    count = int(result["count"][0])
                    impulses, types = result["impulses"][0, :count], result["row_type"][0, :count]
                else:
                    count = int(solver._propagation_constraint_count.numpy()[0])
                    impulses = solver._propagation_impulses.numpy()[0, :count]
                    types = solver._propagation_row_type.numpy()[0, :count]
                load = float(impulses[types == PGS_CONSTRAINT_TYPE_CONTACT].sum())
                self.assertGreater(load, 0.0)
                expected = 0.01 * 0.5 * load / inertia_z
                deceleration = 100.0 - float(result["joint_qd"][5])
                self.assertAlmostEqual(deceleration, expected, delta=0.02 * expected)
                held, *_ = fixture(0.01, spin=1.0, center_only=True, **options)
                self.assertLess(np.max(np.abs(held["body_qd"][:, 5])), 1e-4)

    def test_spin_effect_tracks_immediate(self):
        """Require torsion to change a spin far beyond the response gap, and within the shared non-no-op contract."""
        x = {}
        for response in ("immediate", "propagation"):
            for radius in (0.0, 0.01):
                x[response, radius], _ = _spinner(response, 32, radius, spin=4.0)
        gap_off = float(np.abs(x["propagation", 0.0] - x["immediate", 0.0]).max())
        gap_on = float(np.abs(x["propagation", 0.01] - x["immediate", 0.01]).max())
        effect_immediate = float(np.abs(x["immediate", 0.01] - x["immediate", 0.0]).max())
        effect = float(np.abs(x["propagation", 0.01] - x["propagation", 0.0]).max())
        self.assertGreaterEqual(effect, 0.5 * effect_immediate)
        self.assertGreaterEqual(effect, 10.0 * gap_off)
        self.assertLessEqual(gap_on, 2.0 * gap_off + 1.0e-6)

    def test_converges_to_the_immediate_response(self):
        """Close the coupled three-foot gap to immediate as sweeps rise, with torsion deciding stick."""
        gaps = {}
        for iterations in (8, 32, 128, 512):
            expected, _ = _spinner("immediate", iterations, 0.01, spin=1.14)
            actual, solver = _spinner("propagation", iterations, 0.01, spin=1.14)
            gaps[iterations] = float(np.abs(actual - expected).max())
            self.assertEqual(len(_propagation_torsion_rows(solver)), 3)
        released, _ = _spinner("immediate", 128, 0.0, spin=1.14)
        held, _ = _spinner("immediate", 128, 0.01, spin=1.14)
        self.assertGreater(abs(float(released[1])), 10.0 * abs(float(held[1])) + 1.0e-3)
        self.assertLessEqual(gaps[128], 1.0e-4, gaps)
        self.assertLessEqual(gaps[128], gaps[32] + 1.0e-6, gaps)
        self.assertLessEqual(gaps[512], gaps[128] + 1.0e-6, gaps)

    def test_worlds_solve_as_if_isolated(self):
        """Solve each world of an odd world count, packed two per block, as the same world alone."""
        spins = (4.0, 1.14, 0.3)
        together, solver = _spinner("propagation", 32, 0.01, spin=spins, steps=2)
        self.assertEqual(solver._propagation_gs_worlds_per_block, 2)
        for world, spin in enumerate(spins):
            self.assertEqual(len(_propagation_torsion_rows(solver, world)), 3)
            alone, _ = _spinner("propagation", 32, 0.01, spin=spin, steps=2)
            np.testing.assert_allclose(together.reshape(-1, 2)[world], alone, atol=1.0e-6)

    def test_contact_order_changes_the_result_no_more_than_on_immediate(self):
        """Reorder the contact buffer: plain's change stays within immediate's order floor plus float noise."""
        results = {}
        for response in ("immediate", "propagation"):
            for reverse in (False, True):
                _, solver, model, state, contacts = fixture(
                    0.01,
                    spin=10.0,
                    sliding=0.2,
                    sat=True,
                    step=False,
                    articulated_contact_response=response,
                    contact_torsion_device=True,
                )
                count = int(contacts.rigid_contact_count.numpy()[0])
                self.assertGreater(count, 2)
                if reverse:
                    _permute_contacts(contacts, np.arange(count)[::-1])
                output = model.state()
                solver.step(state, output, model.control(), contacts, 0.0025)
                results[response, reverse] = output.body_qd.numpy()
        floor = float(np.abs(results["immediate", True] - results["immediate", False]).max())
        change = float(np.abs(results["propagation", True] - results["propagation", False]).max())
        self.assertLessEqual(change, floor + 1.0e-6, (change, floor))

    def test_dropped_members_leave_the_highest_kept_host(self):
        """Host a group whose members overflowed the row cap after its last kept member, budgeting kept members only."""
        _, solver, _model, _state, contacts = fixture(
            0.01, spin=10.0, sat=True, articulation_constraint_max=4, free_body_constraint_max=6, **_PROPAGATION_TORSION
        )
        count = int(contacts.rigid_contact_count.numpy()[0])
        slots = solver._contact_slot.numpy()[:count]
        kept = [c for c in range(count) if slots[c] >= 0]
        self.assertGreater(len(kept), 0)
        self.assertLess(len(kept), count)
        rows = _propagation_torsion_rows(solver)
        self.assertEqual(len(rows), 1)
        ((spin, members),) = rows.items()
        self.assertEqual(sorted(members), sorted(int(slots[c]) for c in kept))
        self.assertEqual(spin, _spare_rows(solver, count)[max(kept)])
        self.assertGreater(int(solver._row_dropped_propagation.numpy()[0]), 0)

    def test_persistent_patch_impulse_matches_immediate(self):
        """Give a patch region the same torsion row count and final impulse as immediate at 128 sweeps."""
        impulses = {}
        for response in ("immediate", "propagation"):
            options = {"articulated_contact_response": response, "contact_torsion_device": True, "iterations": 128}
            result, solver, *_ = fixture(0.01, spin=1.0, center_only=True, center_count=4, **options, **PATCH_OPTIONS)
            if response == "immediate":
                count = int(result["count"][0])
                torsion = result["impulses"][0, :count][result["row_type"][0, :count] == PGS_CONSTRAINT_TYPE_TORSION]
            else:
                rows = _propagation_torsion_rows(solver)
                self.assertEqual([len(members) for members in rows.values()], [4])
                torsion = solver._propagation_impulses.numpy()[0, sorted(rows)]
            impulses[response] = torsion
        self.assertEqual(len(impulses["propagation"]), len(impulses["immediate"]))
        self.assertEqual(len(impulses["immediate"]), 1)
        self.assertGreater(abs(float(impulses["immediate"][0])), 0.0)
        np.testing.assert_allclose(impulses["propagation"], impulses["immediate"], rtol=1.0e-4)

    def test_body_parent_f_balances_the_torsion_torque(self):
        """Publish no joint reaction for a spinning free-D6 sphere: the torsion torque enters the external wrench."""
        for response in ("immediate", "propagation"):
            with self.subTest(response=response):
                seen = _spinning_box_parent_wrench(response)
                self.assertEqual(seen["path"], 0 if response == "immediate" else 2)
                self.assertGreater(seen["torsion"], 0.0)
                self.assertLess(seen["parent"], 1.0e-4 * seen["torsion"] + 1.0e-7, seen)
        with patch.object(solver_feather_pgs, "accumulate_propagation_torsion_wrenches_from_impulses", _skip_wrenches):
            seen = _spinning_box_parent_wrench("propagation")
        self.assertAlmostEqual(seen["parent"], seen["torsion"], delta=1.0e-2 * seen["torsion"])

    def test_captured_steps_match_eager_steps_and_report_errors(self):
        """Replay captured plain steps as the eager steps do, and fail stop on a latched torsion error."""
        eager, _ = _spinner("propagation", 16, 0.01, spin=4.0, steps=3)
        captured, solver = _spinner("propagation", 16, 0.01, spin=4.0, steps=3, graph=True)
        np.testing.assert_allclose(captured, eager, atol=1.0e-6)
        _, solver, model, state, contacts = fixture(0.01, iterations=16, step=False, **_PROPAGATION_TORSION)
        output = model.state()
        with wp.ScopedCapture(device=model.device) as capture:
            solver.step(state, output, model.control(), contacts, 0.0025)
        wp.capture_launch(capture.graph)
        solver.check_constraint_capacity()
        self.assertEqual(len(_propagation_torsion_rows(solver)), 1)
        solver._device_torsion.work.status.assign(np.array([16], dtype=np.int32))
        wp.capture_launch(capture.graph)
        with self.assertRaisesRegex(RuntimeError, "load ring"):
            solver.check_constraint_capacity()
        np.testing.assert_array_equal(output.joint_q.numpy(), state.joint_q.numpy())
        self.assertTrue(np.all(solver._propagation_constraint_count.numpy() == 0))

    def test_unsupported_combinations_raise(self):
        """Keep torsion rejected where its rows are not solved or not validated."""
        for options, message in (
            ({"articulated_contact_response": "propagation_colored", "contact_torsion_device": True}, "requires"),
            ({"articulated_contact_response": "propagation_fused", "contact_torsion_device": True}, "requires"),
            ({**_PROPAGATION_TORSION, "contact_warmstart": True}, "warmstart"),
            ({**_PROPAGATION_TORSION, "velocity_iterations": 1}, "velocity_iterations=0"),
            ({"articulated_contact_response": "propagation"}, "contact_torsion_device=True"),
        ):
            with self.subTest(**options), self.assertRaisesRegex(ValueError, message):
                fixture(0.01, step=False, **options)


@wp.kernel
def _skip_wrenches(
    count: wp.array[wp.int32],
    row_type: wp.array2d[wp.int32],
    body_a: wp.array2d[wp.int32],
    body_b: wp.array2d[wp.int32],
    J_a: wp.array3d[wp.float32],
    J_b: wp.array3d[wp.float32],
    impulses: wp.array2d[wp.float32],
    inv_dt: float,
    body_contact_wrench: wp.array[wp.spatial_vector],
):
    pass


def _propagation_torsion_rows(solver, world=0):
    """Return world ``world``'s plain torsion rows mapped to their member normal rows, read from the sweep's lists."""
    head = solver._propagation_torsion_head.numpy()
    first = solver._propagation_torsion_first.numpy()[world]
    following = solver._propagation_torsion_next.numpy()[world]
    rows = {}
    spin = int(head[world])
    while spin >= 0:
        members, n = [], int(first[spin])
        while n >= 0:
            members.append(n)
            n = int(following[n])
        rows[spin] = members
        spin = int(following[spin])
    return rows


def _spare_rows(solver, count):
    """Return each dense-origin propagation contact's spare row, the last of its reserved rows."""
    slots = solver._contact_slot.numpy()[:count]
    needed = solver._contact_slots_needed.numpy()[:count]
    return {c: int(slots[c] + needed[c] - 1) for c in range(count) if slots[c] >= 0}


def _record_propagation_sweeps(radius, on_solver=None, **options):
    """Record world 0's plain propagation impulses after every sweep of one pad-fixture step."""
    sweeps = []
    _, solver, model, state, contacts = fixture(radius, step=False, **options, **_PROPAGATION_TORSION)
    if on_solver is not None:
        on_solver(solver)
    sweep_once = solver._propagation_pgs_solve_one_iteration

    def sweep(**kwargs):
        sweep_once(**kwargs)
        rows = _propagation_torsion_rows(solver)
        row_type = solver._propagation_row_type.numpy()[0]
        parent = solver._propagation_row_parent.numpy()[0]
        count = int(solver._propagation_constraint_count.numpy()[0])
        pairs = {
            spin: [
                n
                for n in members
                if n + 2 < count
                and all(row_type[n + k] == PGS_CONSTRAINT_TYPE_FRICTION and parent[n + k] == n for k in (1, 2))
            ]
            for spin, members in rows.items()
        }
        sweeps.append(
            {
                "offset": kwargs["iteration_offset"],
                "friction_start": kwargs["friction_start_iteration"],
                "after": solver._propagation_impulses.numpy()[0].astype(np.float64),
                "rows": rows,
                "pairs": pairs,
                "mu": solver._propagation_row_mu.numpy()[0],
                "radius": solver._propagation_torsion_row_radius.numpy()[0],
            }
        )

    solver._propagation_pgs_solve_one_iteration = sweep
    solver.step(state, model.state(), model.control(), contacts, 0.0025)
    return sweeps


def _assert_torsion_budgets(test, sweeps):
    """Check every torsion impulse against its group's residual budget; return whether a torsion row and a
    group's sliding pairs reached their bounds."""
    saturated = sliding_saturated = False
    for sweep in sweeps:
        impulses = sweep["after"]
        for spin, members in sweep["rows"].items():
            load = sum(max(float(impulses[n]), 0.0) for n in members)
            sliding = sum(float(np.hypot(impulses[n + 1], impulses[n + 2])) for n in sweep["pairs"][spin])
            budget = float(sweep["mu"][spin]) * load
            bound = float(sweep["radius"][spin]) * max(budget - sliding, 0.0)
            torsion = abs(float(impulses[spin]))
            if sweep["offset"] < sweep["friction_start"]:
                test.assertEqual(torsion, 0.0, f"row {spin} moved before friction start")
                continue
            test.assertLessEqual(sliding, budget * (1.0 + 1.0e-6) + 1.0e-12, f"group {spin} slides past its load")
            test.assertLessEqual(torsion, bound * (1.0 + 1.0e-6) + 1.0e-12, f"row {spin} exceeds its residual budget")
            saturated |= torsion > 0.99 * bound > 0.0
            sliding_saturated |= sliding > 0.99 * budget > 0.0
    return saturated, sliding_saturated


def _permute_contacts(contacts, order):
    """Reorder the first ``len(order)`` entries of every per-contact rigid array of ``contacts``."""
    for name, value in vars(contacts).items():
        if not name.startswith("rigid_contact_") or not isinstance(value, wp.array):
            continue
        if value.shape[0] != contacts.rigid_contact_max or name == "rigid_contact_count":
            continue
        values = value.numpy()
        values[: len(order)] = values[order]
        value.assign(values)


def _spinner(response, iterations, radius, *, spin, steps=1, graph=False):
    """Press three spheres fixed to one spinning slider onto the ground and return its joint velocity.

    The feet share the slider's two DOFs, so the propagation sweep splits their mass; the middle foot
    sits on the spin axis, where only its torsion row resists the spin. A sequence ``spin`` replicates
    one world per rate.
    """
    spins = np.atleast_1d(np.asarray(spin, dtype=np.float32))
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.default_shape_cfg.mu = 0.5
    slider = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
    builder.add_shape_box(
        slider, hx=0.3, hy=0.05, hz=0.02, cfg=newton.ModelBuilder.ShapeConfig(has_shape_collision=False)
    )
    z = newton.ModelBuilder.JointDofConfig(axis=wp.vec3(0.0, 0.0, 1.0))
    joints = [builder.add_joint_d6(-1, slider, linear_axes=[z], angular_axes=[z])]
    for x in (-0.2, 0.0, 0.2):
        foot = builder.add_link(xform=wp.transform(wp.vec3(x, 0.0, 0.05), wp.quat_identity()))
        builder.add_shape_sphere(foot, radius=0.05)
        joints.append(
            builder.add_joint_fixed(
                slider,
                foot,
                parent_xform=wp.transform(wp.vec3(x, 0.0, -0.05), wp.quat_identity()),
                child_xform=wp.transform_identity(),
            )
        )
    builder.add_articulation(joints)
    builder.add_ground_plane()
    if len(spins) > 1:
        scene = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        scene.replicate(builder, len(spins), spacing=(1.0, 0.0, 0.0))
        builder = scene
    model = builder.finalize(device="cuda:0")
    set_torsion_radius(model, radius)
    solver = SolverFeatherPGS(
        model,
        solve_mode="matrix_free",
        articulated_contact_response=response,
        friction_anchor_beta=0.0,
        articulation_constraint_max=96,
        free_body_constraint_max=96,
        iterations=iterations,
        baumgarte=0.0,
        cfm=0.0,
        contact_regularization=3.0,
        contact_torsion_device=True,
    )
    state, output = model.state(), model.state()
    state.joint_qd.assign(np.stack([np.full_like(spins, -1.0), spins], axis=1).reshape(-1))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    if graph:
        pipeline.collide(state, contacts)
        solver.step(state, output, model.control(), contacts, 1.0 / 240.0)
        state, output = output, state
        with wp.ScopedCapture(device=model.device) as capture:
            pipeline.collide(state, contacts)
            solver.step(state, output, model.control(), contacts, 1.0 / 240.0)
            pipeline.collide(output, contacts)
            solver.step(output, state, model.control(), contacts, 1.0 / 240.0)
        for _ in range((steps - 1) // 2):
            wp.capture_launch(capture.graph)
        solver.check_constraint_capacity()
        return state.joint_qd.numpy(), solver
    for _ in range(steps):
        pipeline.collide(state, contacts)
        solver.step(state, output, model.control(), contacts, 1.0 / 240.0)
        state, output = output, state
    return state.joint_qd.numpy(), solver


def _spinning_box_parent_wrench(response):
    """Spin a sphere on a free D6 joint against the ground and return its route, torsion torque and parent wrench."""
    b = newton.ModelBuilder()
    b.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5, restitution=0))
    pose = wp.transform(wp.vec3(0, 0, 0.0199), wp.quat_identity())
    body = b.add_link(xform=pose, mass=0.3, inertia=wp.mat33(np.diag([3e-4, 3e-4, 3e-4]).astype(np.float32)))
    axes = [newton.ModelBuilder.JointDofConfig(axis=wp.vec3(*a)) for a in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
    joint = b.add_joint_d6(-1, body, parent_xform=pose, linear_axes=axes, angular_axes=axes)
    b.add_articulation([joint])
    # One central witness slides on none of its budget, so the spin is resisted by torsion alone.
    b.add_shape_sphere(body, radius=0.02, cfg=newton.ModelBuilder.ShapeConfig(density=0, mu=0.5))
    model = b.finalize(device="cuda:0")
    set_torsion_radius(model, 0.01)
    solver = SolverFeatherPGS(
        model,
        solve_mode="matrix_free",
        articulated_contact_response=response,
        iterations=64,
        baumgarte=0.05,
        cfm=0,
        contact_regularization=0,
        articulation_constraint_max=64,
        free_body_constraint_max=16,
        contact_torsion_device=True,
        friction_anchor_beta=0.0,
    )
    pipeline = newton.CollisionPipeline(model, reduce_contacts=False, broad_phase="nxn", rigid_contact_max=64)
    contacts = pipeline.contacts()
    observables = solver.observables({SolverBase.ObservableKind.BODY_PARENT_F})
    state, output = model.state(), model.state()
    velocity = np.zeros(model.joint_dof_count, np.float32)
    velocity[5] = 20.0
    state.joint_qd.assign(velocity)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    dt = 0.0025
    for _ in range(4):
        pipeline.collide(state, contacts)
        solver.step(state, output, model.control(), contacts, dt, observables=observables)
        state, output = output, state
    if response == "immediate":
        count = int(solver._constraint_count.numpy()[0])
        types, impulses = solver._row_type.numpy()[0, :count], solver._impulses.numpy()[0, :count]
    else:
        count = int(solver._propagation_constraint_count.numpy()[0])
        types = solver._propagation_row_type.numpy()[0, :count]
        impulses = solver._propagation_impulses.numpy()[0, :count]
    return {
        "path": int(solver._contact_path.numpy()[0]),
        "torsion": abs(float(impulses[types == PGS_CONSTRAINT_TYPE_TORSION].sum())) / dt,
        "parent": abs(float(observables.body_parent_f.numpy()[0][5])),
    }


if __name__ == "__main__":
    unittest.main()
