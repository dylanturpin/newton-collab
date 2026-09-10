# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for persistent patch friction in FeatherPGS."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.friction_patches import (
    _FrictionPatchState,
    finish_patch_impulses,
    link_patch_rows,
    seed_patch_impulses,
)
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    contact_friction_residuals,
    pgs_solve_loop,
)
from newton._src.solvers.feather_pgs.solver_feather_pgs import _get_pgs_solve_tiled_row_kernel
from newton.tests.test_feather_pgs_friction_anchors import _run_squeeze


@wp.kernel(enable_backward=False)
def _friction_residual_probe(result: wp.array[wp.vec4]):
    result[0] = contact_friction_residuals(0.0, 4.0, 0.25, 0.0, wp.vec2(-1.0, 0.0), wp.vec2(1.0, 0.0))
    result[1] = contact_friction_residuals(0.0, 4.0, 0.25, 0.0, wp.vec2(-1.0, 0.0), wp.vec2(2.5, 0.0))
    result[2] = contact_friction_residuals(2.0, 2.0, 0.5, -0.1, wp.vec2(0.0), wp.vec2(0.0))


def _ground_box(device, **solver_kwargs):
    """Return a resting 1 kg box on the ground with patch friction enabled."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
    body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.1), wp.quat_identity()))
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=newton.ModelBuilder.ShapeConfig(density=125, mu=0.5))
    model = builder.finalize(device=device)
    kwargs = {
        "pgs_iterations": 32,
        "pgs_mode": "matrix_free" if model.device.is_cuda else "split",
    }
    kwargs.update(solver_kwargs)
    solver = newton.solvers.SolverFeatherPGS(model, **kwargs)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=64)
    contacts = pipeline.contacts()
    states = [model.state(), model.state()]
    control = model.control()

    def step():
        s0, s1 = states
        s0.clear_forces()
        pipeline.collide(s0, contacts)
        solver.step(s0, s1, control, contacts, 0.005)
        states.reverse()

    return model, solver, contacts, step


def _patch_fixture(
    points, *, shape0=None, normals=None, materials=(0.5, 0.5, 0.5), shape_bodies=(0, 1, 0), device="cpu"
):
    """Construct patch geometry without a collision matcher or a solver step."""
    n = len(points)
    body_count = max(shape_bodies) + 1
    model = SimpleNamespace(
        device=wp.get_device(device),
        body_count=body_count,
        body_world=wp.zeros(body_count, dtype=int, device=device),
        shape_body=wp.array(shape_bodies, dtype=int, device=device),
        shape_collision_radius=wp.array([0.2] * len(shape_bodies), dtype=float, device=device),
        shape_transform=wp.array([wp.transform_identity()] * len(shape_bodies), dtype=wp.transform, device=device),
        shape_scale=wp.ones(len(shape_bodies), dtype=wp.vec3, device=device),
        shape_type=wp.zeros(len(shape_bodies), dtype=int, device=device),
        shape_source_ptr=wp.zeros(len(shape_bodies), dtype=wp.uint64, device=device),
        shape_margin=wp.zeros(len(shape_bodies), dtype=float, device=device),
        shape_gap=wp.zeros(len(shape_bodies), dtype=float, device=device),
        shape_is_solid=wp.ones(len(shape_bodies), dtype=bool, device=device),
        shape_material_mu=wp.array(materials, dtype=float, device=device),
    )
    state = SimpleNamespace(body_q=wp.array([wp.transform_identity()] * body_count, dtype=wp.transform, device=device))
    contacts = SimpleNamespace(
        rigid_contact_count=wp.array([n], dtype=int, device=device),
        rigid_contact_shape0=wp.array(shape0 if shape0 is not None else [0] * n, dtype=int, device=device),
        rigid_contact_shape1=wp.ones(n, dtype=int, device=device),
        rigid_contact_point0=wp.array(points, dtype=wp.vec3, device=device),
        rigid_contact_point1=wp.array(points, dtype=wp.vec3, device=device),
        rigid_contact_normal=wp.array(
            normals if normals is not None else [[0.0, 0.0, -1.0]] * n, dtype=wp.vec3, device=device
        ),
        rigid_contact_margin0=wp.zeros(n, dtype=float, device=device),
        rigid_contact_margin1=wp.zeros(n, dtype=float, device=device),
    )
    patches = _FrictionPatchState(model, max(n, 8), True, wp.zeros(max(n, 8), dtype=wp.vec2, device=device))
    patches.build(model, state, contacts)
    return model, state, contacts, patches


class TestFrictionPatchHistory(unittest.TestCase):
    def test_shared_pad_randomization_preserves_material_regions(self):
        """Pool one sampled pad material while keeping genuinely different coefficients separate."""
        model, state, contacts, patches = _patch_fixture([[-0.1, 0, 0], [0.1, 0, 0]], shape0=[0, 2])
        for coefficient in np.random.default_rng(42).uniform(0.1, 1.0, 3):
            model.shape_material_mu.assign([coefficient, 0.5, coefficient])
            patches.update_geometry(model)
            patches.build(model, state, contacts)
            self.assertEqual(len(np.unique(patches.current.owner.numpy()[:2])), 1)
            np.testing.assert_allclose(patches.view.weight.numpy()[:2], [0.5, 0.5])
            patches.store(state)
        model.shape_material_mu.assign([0.2, 0.5, 0.8])
        patches.build(model, state, contacts)
        self.assertEqual(len(np.unique(patches.current.owner.numpy()[:2])), 2)
        np.testing.assert_allclose(patches.view.weight.numpy()[:2], [1, 1])

    def test_rocking_face_retires_the_lifted_anchor(self):
        """Keep the supported edge's history and replace the opposite anchor after rocking."""
        points = [[-0.1, -0.1, 0], [0.1, -0.1, 0], [-0.1, 0.1, 0], [0.1, 0.1, 0]]
        model, state, contacts, patches = _patch_fixture(points)
        patches.store(state)
        rotation = wp.quat_from_axis_angle(wp.vec3(0, 1, 0), -np.pi / 60.0)
        pivot = wp.vec3(-0.1, 0, 0)
        state.body_q.assign([wp.transform(pivot - wp.quat_rotate(rotation, pivot), rotation), wp.transform_identity()])
        contacts.rigid_contact_count.assign([2])
        edge = [points[0], points[2], points[0], points[2]]
        contacts.rigid_contact_point0.assign(edge)
        contacts.rigid_contact_point1.assign(edge)
        patches.build(model, state, contacts)
        active = patches.current.valid.numpy() != 0
        self.assertEqual(np.count_nonzero(active), 2)
        np.testing.assert_allclose(patches.current.anchor_a.numpy()[active, 0], -0.1, atol=1.0e-7)
        self.assertEqual(np.count_nonzero(patches.current.source.numpy()[active] >= 0), 1)

    def test_unloading_penetration_keeps_supported_anchors(self):
        """Retain history while decompression leaves the actual surfaces in contact."""
        model, state, contacts, patches = _patch_fixture([[-0.1, 0, 0], [0.1, 0, 0]])
        contacts.rigid_contact_point0.assign([[-0.1, 0, -0.005], [0.1, 0, -0.005]])
        patches.build(model, state, contacts)
        patches.store(state)
        state.body_q.assign([wp.transform(wp.vec3(0, 0, 0.003), wp.quat_identity()), wp.transform_identity()])
        patches.build(model, state, contacts)
        self.assertEqual(np.count_nonzero(patches.current.source.numpy() >= 0), 2)

    def test_support_witnesses_use_existing_contact_gap_limits(self):
        """Keep supported history within the contact envelope and respect tighter solver gates."""
        for limits, carried in (({}, True), ({"contact_gap_gate": 0.0001}, False), ({"friction_gap": 0.0001}, True)):
            with self.subTest(limits=limits):
                model, state, contacts, patches = _patch_fixture([[-0.1, 0, 0], [0.1, 0, 0]])
                model.shape_gap.fill_(0.0005)
                patches.store(state)
                state.body_q.assign([wp.transform(wp.vec3(0, 0, 0.0002), wp.quat_identity()), wp.transform_identity()])
                contacts.rigid_contact_point0.assign([[-0.1, 0, -0.0004], [0.1, 0, -0.0004]])
                patches.build(model, state, contacts, **limits)
                self.assertEqual(np.count_nonzero(patches.current.source.numpy() >= 0), 2 if carried else 0)
                # A fresh sample on the supported region cannot retain history
                # whose old footprint has lifted beyond the shape gap envelope.
                state.body_q.assign([wp.transform(wp.vec3(0, 0, 0.0012), wp.quat_identity()), wp.transform_identity()])
                contacts.rigid_contact_point0.assign([[-0.1, 0, -0.0014], [0.1, 0, -0.0014]])
                patches.build(model, state, contacts, **limits)
                self.assertEqual(np.count_nonzero(patches.current.source.numpy() >= 0), 0)

    def test_geometry_edits_retire_only_affected_history(self):
        """Invalidate edited surfaces across carrier seams while preserving unrelated patches."""
        for field in ("shape_transform", "shape_scale", "shape_margin", "shape_source_ptr"):
            for static in (False, True):
                with self.subTest(field=field, static=static):
                    model, state, contacts, patches = _patch_fixture(
                        [[0, 0, 0], [0, 0, 0]],
                        shape0=[0, 3],
                        shape_bodies=(0, -1 if static else 1, 0, 2, 3),
                        materials=(0.5,) * 5,
                    )
                    contacts.rigid_contact_shape1.assign([1, 4])
                    patches.build(model, state, contacts)
                    patches.store(state)
                    np.testing.assert_array_equal(patches.previous.valid.numpy()[:2], [1, 1])
                    # Static geometry is identified by shape. A dynamic edit on
                    # shape 2 must also retire shape 0's anchor on the same body.
                    edited = 1 if static else 2
                    values = getattr(model, field).numpy()
                    if field == "shape_transform":
                        values[edited, 0] += 1
                    else:
                        values[edited] += 1
                    getattr(model, field).assign(values)
                    patches.update_geometry(model)
                    np.testing.assert_array_equal(patches.previous.valid.numpy()[:2], [0, 1])

    def test_geometry_updates_allow_contacts_without_a_shape(self):
        """Preserve implicit static contacts when an unrelated shape changes."""
        model, state, contacts, patches = _patch_fixture([[0, 0, 0]], shape_bodies=(0, 1, 1))
        contacts.rigid_contact_shape1.assign([-1])
        patches.build(model, state, contacts)
        patches.store(state)
        self.assertEqual(int(patches.previous.valid.numpy()[0]), 1)
        # Editing the last shape must not alias the contact's -1 sentinel.
        transforms = model.shape_transform.numpy()
        transforms[-1, 0] = 1.0
        model.shape_transform.assign(transforms)
        patches.update_geometry(model)
        self.assertEqual(int(patches.previous.valid.numpy()[0]), 1)

    def test_reused_contacts_keep_anchors_across_substeps(self):
        """Preserve material points across substeps regardless of stale collision match indices."""
        model, state, contacts, patches = _patch_fixture([[-0.1, 0, 0], [0.1, 0, 0]])
        contacts.rigid_contact_match_index = wp.array([1, 0], dtype=int, device=model.device)
        active = patches.current.valid.numpy() != 0
        anchors = patches.current.anchor_a.numpy()[active].copy()
        patches.store(state)
        for substep in range(1, 4):
            displacement = substep * 0.001
            state.body_q.assign(
                [wp.transform(wp.vec3(displacement, 0, 0), wp.quat_identity()), wp.transform_identity()]
            )
            patches.build(model, state, contacts)
            active = patches.current.valid.numpy() != 0
            np.testing.assert_array_equal(patches.current.anchor_a.numpy()[active], anchors)
            np.testing.assert_allclose(
                np.linalg.norm(patches.view.phi.numpy()[active], axis=1), displacement, atol=1.0e-7
            )
            self.assertTrue(np.all(patches.current.source.numpy()[active] >= 0))
            patches.store(state)

    def test_connected_convex_chain_forms_one_region(self):
        """Join a chain into one friction region even when the two end shapes do not overlap."""
        model, state, contacts, patches = _patch_fixture(
            [[-0.15, 0, 0], [0.15, 0, 0], [-0.05, 0, 0], [0.05, 0, 0]],
            shape0=[0, 4, 2, 3],
            shape_bodies=(0, 1, 0, 0, 0),
            materials=(0.5,) * 5,
        )
        model.shape_transform.assign(
            [wp.transform(wp.vec3(x, 0, 0), wp.quat_identity()) for x in (-0.15, 0, -0.05, 0.05, 0.15)]
        )
        model.shape_collision_radius.assign([0.06, 0.3, 0.06, 0.06, 0.06])
        patches.update_geometry(model)
        patches.build(model, state, contacts)
        self.assertEqual(len(np.unique(patches.current.owner.numpy()[:4])), 1)
        np.testing.assert_allclose(patches.view.weight.numpy()[:4], [0.5, 0.5, 0, 0])

    def test_disconnected_convex_shapes_form_separate_regions(self):
        """Keep two separated pads on one body from pooling friction through empty space."""
        model, state, contacts, patches = _patch_fixture([[-0.15, 0, 0], [0.15, 0, 0]], shape0=[0, 2])
        model.shape_transform.assign(
            [
                wp.transform(wp.vec3(-0.15, 0, 0), wp.quat_identity()),
                wp.transform_identity(),
                wp.transform(wp.vec3(0.15, 0, 0), wp.quat_identity()),
            ]
        )
        model.shape_collision_radius.assign([0.03, 0.2, 0.03])
        patches.update_geometry(model)
        patches.build(model, state, contacts)
        self.assertEqual(len(np.unique(patches.current.owner.numpy()[:2])), 2)
        np.testing.assert_allclose(patches.view.weight.numpy()[:2], [1, 1])

    def test_residuals_use_patch_support_and_keep_normal_terms_local(self):
        """Distinguish supported friction, a cone violation, and normal error in the residual diagnostics."""
        residuals = wp.zeros(3, dtype=wp.vec4, device="cpu")
        wp.launch(_friction_residual_probe, dim=1, inputs=[residuals], device="cpu")
        np.testing.assert_allclose(residuals.numpy(), [[0, 0, 0, 0], [1.5, 1.5, 0, 1.5], [0, 0.2, 0.1, 0]], atol=1e-7)

    def test_incomplete_frame_cannot_create_anchor_history(self):
        """Reject an overflowing contact frame before selecting or carrying anchors."""
        model, state, contacts, patches = _patch_fixture([[-0.1, 0, 0], [0.1, 0, 0]])
        contacts.rigid_contact_count.assign([3])
        patches.build(model, state, contacts)
        self.assertEqual(np.count_nonzero(patches.current.valid.numpy()), 0)
        self.assertEqual(np.count_nonzero(patches.view.weight.numpy()), 0)

    def test_reversed_contact_orientation_uses_same_patch(self):
        """Give a body pair one identity regardless of collision shape ordering."""
        model, state, contacts, patches = _patch_fixture([[-0.1, 0, 0], [0.1, 0, 0]])
        contacts.rigid_contact_shape0.assign([0, 1])
        contacts.rigid_contact_shape1.assign([1, 0])
        contacts.rigid_contact_normal.assign([[0, 0, -1], [0, 0, 1]])
        patches.build(model, state, contacts)
        self.assertEqual(len(np.unique(patches.current.owner.numpy()[:2])), 1)
        np.testing.assert_allclose(patches.view.weight.numpy()[:2], [0.5, 0.5])

    def test_saturation_only_releases_an_anchor_with_sliding_motion(self):
        """Keep history under impending slip, and release it under motion against saturated friction."""
        _, state, contacts, patches = _patch_fixture([[0, 0, 0]])
        zeros = wp.zeros(1, dtype=int, device="cpu")
        lengths = wp.array([3], dtype=int, device="cpu")
        parents = wp.array([[0, 0, 0]], dtype=int, device="cpu")
        mu = wp.array([[0.5, 0.5, 0.5]], dtype=float, device="cpu")
        impulses = wp.array([[2, 1, 0]], dtype=float, device="cpu")
        velocity = wp.zeros(2, dtype=wp.spatial_vector, device="cpu")
        args = [
            contacts.rigid_contact_count,
            patches.current,
            state.body_q,
            state.body_q,
            velocity,
            wp.zeros(2, dtype=wp.vec3, device="cpu"),
            zeros,
            zeros,
            zeros,
            lengths,
            0,
            parents,
            mu,
            impulses,
            0.005,
        ]
        wp.launch(finish_patch_impulses, dim=1, inputs=args, device="cpu")
        self.assertEqual(patches.current.valid.numpy()[0], 1)
        velocity.assign([[0, -0.1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]])
        wp.launch(finish_patch_impulses, dim=1, inputs=args, device="cpu")
        self.assertEqual(patches.current.valid.numpy()[0], 0)

    def test_convex_shapes_share_one_body_patch(self):
        """Pool friction load across adjacent convex pieces with compatible materials."""
        _, _, _, patches = _patch_fixture(
            [[-0.1, -0.1, 0], [0.1, -0.1, 0], [-0.1, 0.1, 0], [0.1, 0.1, 0]], shape0=[0, 2, 0, 2]
        )
        self.assertEqual(np.count_nonzero(patches.view.weight.numpy()), 2)
        self.assertEqual(len(np.unique(patches.current.owner.numpy()[:4])), 1)
        self.assertAlmostEqual(float(patches.view.weight.numpy().sum()), 1.0)

    def test_material_normal_and_disconnected_regions_stay_separate(self):
        """Keep friction from pooling across incompatible or spatially separate regions."""
        cases = (
            {"points": [[0, 0, 0], [0.01, 0, 0]], "shape0": [0, 2], "materials": (0.5, 0.5, 0.8)},
            {"points": [[0, 0, 0], [0.01, 0, 0]], "normals": [[0, 0, -1], [0, -1, 0]]},
            {"points": [[0, 0, 0], [0, 0, 0.05]]},
            {"points": [[0, 0, 0], [1, 0, 0]]},
        )
        for case in cases:
            with self.subTest(case=case):
                _, _, _, patches = _patch_fixture(**case)
                self.assertEqual(len(np.unique(patches.current.owner.numpy()[:2])), 2)
                np.testing.assert_array_equal(patches.view.weight.numpy()[:2], [1, 1])

    def test_contact_churn_preserves_material_points_and_error(self):
        """Preserve a sticking region's history even when every contact sample changes."""
        points = np.array([[-0.1, -0.1, 0], [0.1, -0.1, 0], [-0.1, 0.1, 0], [0.1, 0.1, 0]], dtype=np.float32)
        model, state, contacts, patches = _patch_fixture(points)
        previous = patches.current.anchor_a.numpy()[patches.current.valid.numpy() != 0].copy()
        patches.store(state)
        points = points[[3, 1, 0, 2]] * 0.98
        contacts.rigid_contact_point0.assign(points)
        contacts.rigid_contact_point1.assign(points)
        state.body_q.assign([wp.transform(wp.vec3(0.001, 0, 0), wp.quat_identity()), wp.transform_identity()])
        patches.build(model, state, contacts)
        active = patches.current.valid.numpy() != 0
        current = patches.current.anchor_a.numpy()[active]
        np.testing.assert_allclose(sorted(map(tuple, current)), sorted(map(tuple, previous)), atol=1.0e-7)
        np.testing.assert_allclose(np.linalg.norm(patches.view.phi.numpy()[active], axis=1), 0.001, atol=1.0e-7)
        self.assertTrue(np.all(patches.current.source.numpy()[active] >= 0))

    def test_coincident_contacts_use_one_anchor(self):
        """Reject duplicate witnesses that would create two coincident friction constraints."""
        _, _, _, patches = _patch_fixture([[0, 0, 0]] * 4)
        self.assertEqual(np.count_nonzero(patches.view.weight.numpy()), 1)
        self.assertEqual(float(patches.view.weight.numpy().sum()), 1.0)

    def test_warmstart_uses_patch_load_when_anchor_normal_is_unloaded(self):
        """Keep a cached tangent supported when other normals carry the load."""
        model, state, contacts, patches = _patch_fixture([[-0.1, 0, 0], [0, 0, 0], [0.1, 0, 0]])
        patches.current.tangent_impulse.assign([[0.8, 0, 0], [0, 0, 0], [0.8, 0, 0]] + [[0, 0, 0]] * 5)
        patches.store(state)
        patches.build(model, state, contacts)
        slots = wp.array([0, 3, 4], dtype=int, device="cpu")
        worlds = wp.zeros(3, dtype=int, device="cpu")
        paths = wp.zeros(3, dtype=int, device="cpu")
        lengths = wp.array([3, 1, 3], dtype=int, device="cpu")
        parents = wp.array([[-1, 0, 0, -1, -1, 4, 4]], dtype=int, device="cpu")
        mu = wp.array([[0.5] * 7], dtype=float, device="cpu")
        impulses = wp.array([[0, 0, 0, 4, 0, 0, 0]], dtype=float, device="cpu")
        wp.launch(
            link_patch_rows,
            dim=3,
            inputs=[contacts.rigid_contact_count, patches.view, worlds, slots, paths, lengths, 0, parents, mu],
            device="cpu",
        )
        wp.launch(
            seed_patch_impulses,
            dim=3,
            inputs=[
                contacts.rigid_contact_count,
                patches.current,
                patches.previous,
                state.body_q,
                worlds,
                slots,
                paths,
                lengths,
                0,
                parents,
                mu,
                impulses,
                1.0,
            ],
            device="cpu",
        )
        result = impulses.numpy()[0]
        self.assertAlmostEqual(float(np.linalg.norm(result[1:3])), 0.8)
        self.assertAlmostEqual(float(np.linalg.norm(result[5:7])), 0.8)

    def test_seed_keeps_matched_warmstart_without_patch_history(self):
        """Keep the contact-matched friction warm start on an anchor that has no patch history yet."""
        _model, state, contacts, patches = _patch_fixture([[0, 0, 0]])
        self.assertEqual(int(patches.current.source.numpy()[0]), -1)
        slots = wp.zeros(1, dtype=int, device="cpu")
        lengths = wp.array([3], dtype=int, device="cpu")
        parents = wp.array([[-1] * 8], dtype=int, device="cpu")
        mu = wp.array([[0.5, 0.25, 0.25] + [0.0] * 5], dtype=float, device="cpu")
        seeded = [1.0, 0.3, -0.2] + [0.0] * 5
        impulses = wp.array([seeded], dtype=float, device="cpu")
        wp.launch(
            seed_patch_impulses,
            dim=1,
            inputs=[
                contacts.rigid_contact_count,
                patches.current,
                patches.previous,
                state.body_q,
                slots,
                slots,
                slots,
                lengths,
                0,
                parents,
                mu,
                impulses,
                1.0,
            ],
            device="cpu",
        )
        np.testing.assert_array_equal(impulses.numpy()[0], np.array(seeded, dtype=np.float32))

    def test_anchors_without_rows_keep_history(self):
        """Keep a valid anchor and its cached impulse when its region gets no friction rows this step."""
        _model, state, contacts, patches = _patch_fixture([[0, 0, 0]])
        cached = [[0.1, 0.2, 0.0]] + [[0.0, 0.0, 0.0]] * 7
        zeros = wp.zeros(1, dtype=int, device="cpu")
        parents = wp.array([[-1] * 8], dtype=int, device="cpu")
        mu = wp.zeros((1, 8), dtype=float, device="cpu")
        impulses = wp.zeros((1, 8), dtype=float, device="cpu")
        qd = wp.zeros(2, dtype=wp.spatial_vector, device="cpu")
        com = wp.zeros(2, dtype=wp.vec3, device="cpu")
        for slot, length in ((-1, 3), (0, 1)):
            with self.subTest(slot=slot, slots_needed=length):
                patches.current.valid.assign([1] + [0] * 7)
                patches.current.tangent_impulse.assign(cached)
                wp.launch(
                    finish_patch_impulses,
                    dim=1,
                    inputs=[
                        contacts.rigid_contact_count,
                        patches.current,
                        state.body_q,
                        state.body_q,
                        qd,
                        com,
                        zeros,
                        wp.array([slot], dtype=int, device="cpu"),
                        zeros,
                        wp.array([length], dtype=int, device="cpu"),
                        0,
                        parents,
                        mu,
                        impulses,
                        0.005,
                    ],
                    device="cpu",
                )
                self.assertEqual(int(patches.current.valid.numpy()[0]), 1)
                np.testing.assert_array_equal(
                    patches.current.tangent_impulse.numpy()[0], np.asarray(cached[0], dtype=np.float32)
                )

    def test_carried_anchors_copy_stored_material_points(self):
        """Carry stored body-local anchors verbatim instead of round-tripping them through world space."""
        model, state, contacts, patches = _patch_fixture([[0.02, -0.03, 0.0]])
        rotation = wp.quat_from_axis_angle(wp.normalize(wp.vec3(0.3, 0.5, 0.8)), 1.1)
        state.body_q.assign([wp.transform(wp.vec3(1.3, -0.7, 2.1), rotation), wp.transform_identity()])
        patches.build(model, state, contacts)
        patches.store(state)
        patches.build(model, state, contacts)
        source = int(patches.current.source.numpy()[0])
        self.assertGreaterEqual(source, 0)
        np.testing.assert_array_equal(patches.current.anchor_a.numpy()[0], patches.previous.anchor_a.numpy()[source])
        np.testing.assert_array_equal(patches.current.anchor_b.numpy()[0], patches.previous.anchor_b.numpy()[source])

    def test_filtered_members_carry_history_without_starting_it(self):
        """Carry existing anchors through a friction-filtered step with zero weight, and never create new ones."""
        model, state, contacts, patches = _patch_fixture([[-0.05, 0, 0], [0.05, 0, 0]])
        patches.store(state)
        patches.build(model, state, contacts, friction_gap=-1.0)
        self.assertEqual(int(patches.current.valid.numpy().sum()), 2)
        self.assertTrue((patches.current.source.numpy()[:2] >= 0).all())
        self.assertEqual(float(patches.view.weight.numpy().max()), 0.0)
        patches.previous.valid.zero_()
        patches.build(model, state, contacts, friction_gap=-1.0)
        self.assertEqual(int(patches.current.valid.numpy().sum()), 0)
        self.assertEqual(float(patches.view.weight.numpy().max()), 0.0)


class TestFeatherPGSFrictionPatches(unittest.TestCase):
    def test_default_friction_allows_a_sphere_to_keep_rolling(self):
        """Keep a freely rolling sphere moving without pinning its changing contact footprint."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.05), wp.quat_identity()))
        builder.add_shape_sphere(body, radius=0.05)
        model = builder.finalize(device=device)
        solver = newton.solvers.SolverFeatherPGS(model)
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        s0, s1 = model.state(), model.state()
        s0.joint_qd.assign([1, 0, 0, 0, 20, 0])
        newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)
        control = model.control()
        for _ in range(240):
            s0.clear_forces()
            pipeline.collide(s0, contacts)
            solver.step(s0, s1, control, contacts, 1.0 / 240.0)
            s0, s1 = s1, s0
        pose, velocity = s0.body_q.numpy()[0], s0.body_qd.numpy()[0]
        self.assertTrue(np.isfinite(pose).all() and np.isfinite(velocity).all())
        # This is a no-pinning smoke test, not a claim of energy-conserving
        # rolling: finite-step patch friction can dissipate rolling motion.
        self.assertAlmostEqual(float(pose[0]), 1.0, delta=0.1)
        self.assertGreater(float(velocity[0]), 0.85)
        self.assertLess(abs(float(velocity[0] - 0.05 * velocity[4])), 0.15)
        self.assertAlmostEqual(float(pose[2]), 0.05, delta=0.002)

    def test_default_friction_keeps_a_box_stack_at_rest(self):
        """Settle an ordinary stack without grasp-specific settings or anchor opt-in."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        for height in (0.05, 0.151, 0.252):
            body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, height), wp.quat_identity()))
            builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
        model = builder.finalize(device=device)
        solver = newton.solvers.SolverFeatherPGS(model)
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        s0, s1 = model.state(), model.state()
        control = model.control()
        for _ in range(240):
            s0.clear_forces()
            pipeline.collide(s0, contacts)
            solver.step(s0, s1, control, contacts, 1.0 / 240.0)
            s0, s1 = s1, s0
        poses, velocities = s0.body_q.numpy(), s0.body_qd.numpy()
        self.assertTrue(np.isfinite(poses).all() and np.isfinite(velocities).all())
        np.testing.assert_allclose(poses[:, 2], [0.05, 0.15, 0.25], atol=0.005)
        self.assertLess(float(np.max(np.abs(poses[:, :2]))), 0.005)
        self.assertLess(float(np.max(np.linalg.norm(velocities[:, :3], axis=1))), 0.05)
        self.assertGreater(np.count_nonzero(solver._friction_patches.previous.valid.numpy()), 0)

    def test_default_patch_friction_and_explicit_opt_out(self):
        """Build persistent patches by default while retaining an explicit velocity-only opt-out."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        for kwargs, enabled in (({}, True), ({"friction_anchor_beta": 0.0}, False)):
            with self.subTest(enabled=enabled):
                _, solver, _, step = _ground_box(device, **kwargs)
                for _ in range(3):
                    step()
                self.assertEqual(solver._friction_anchors_enabled, enabled)
                if enabled:
                    self.assertAlmostEqual(solver.friction_anchor_beta, 0.2)
                    self.assertEqual(np.count_nonzero(solver._friction_patches.current.source.numpy() >= 0), 2)
                else:
                    self.assertFalse(hasattr(solver._friction_patches, "current"))

    def test_rocking_cube_assigns_friction_only_to_supported_edge(self):
        """Retire the lifted face anchor before building rows for a cube resting on one edge."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        builder = newton.ModelBuilder(gravity=wp.vec3(0, 0, 0))
        builder.rigid_gap = 1.0e-4
        builder.add_ground_plane()
        body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.1), wp.quat_identity()))
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device=device)
        solver = newton.solvers.SolverFeatherPGS(model, friction_anchor_beta=0.2, contact_gap_gate=1.0e-4)
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=16)
        contacts = pipeline.contacts()
        s0, s1 = model.state(), model.state()
        pipeline.collide(s0, contacts)
        solver.step(s0, s1, model.control(), contacts, 0.005)
        self.assertEqual(np.count_nonzero(solver._friction_patches.previous.valid.numpy()), 2)
        rotation = wp.quat_from_axis_angle(wp.vec3(0, 1, 0), -np.pi / 60.0)
        translation = wp.vec3(-0.1, 0, 0) - wp.quat_rotate(rotation, wp.vec3(-0.1, 0, -0.1))
        transform = np.array([*translation, *rotation], dtype=np.float32)
        s1.joint_q.assign(transform)
        newton.eval_fk(model, s1.joint_q, s1.joint_qd, s1)
        pipeline.collide(s1, contacts)
        solver.step(s1, s0, model.control(), contacts, 0.005)
        patches = solver._friction_patches
        active = patches.view.weight.numpy() > 0
        self.assertEqual(np.count_nonzero(active), 2)
        np.testing.assert_allclose(patches.current.anchor_a.numpy()[active, 0], -0.1, atol=1.0e-6)

    def test_contact_churn_reports_linear_force_but_not_a_wrench(self):
        """Compare the linear export with solved rows under an off-centre load after churn."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        builder = newton.ModelBuilder()
        ground = builder.add_ground_plane()
        body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.1), wp.quat_identity()))
        box = builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device=device)
        solver = newton.solvers.SolverFeatherPGS(model, friction_anchor_beta=0.2, pgs_iterations=64)
        contacts = newton.Contacts(4, 0, requested_attributes=["force"], device=device)
        contacts.rigid_contact_count.assign([4])
        contacts.rigid_contact_shape0.assign([box] * 4)
        contacts.rigid_contact_shape1.assign([ground] * 4)
        contacts.rigid_contact_normal.assign([[0, 0, -1]] * 4)
        points = np.array([[-0.1, -0.1, 0], [0.1, -0.1, 0], [-0.1, 0.1, 0], [0.1, 0.1, 0]], dtype=np.float32)
        contacts.rigid_contact_point0.assign(points + np.array([0, 0, -0.1], dtype=np.float32))
        contacts.rigid_contact_point1.assign(points)
        s0, s1 = model.state(), model.state()
        dt = 0.005
        solver.step(s0, s1, model.control(), contacts, dt)
        points[:, :2] *= 0.8
        contacts.rigid_contact_point0.assign(points + np.array([0, 0, -0.1], dtype=np.float32))
        contacts.rigid_contact_point1.assign(points)
        # The x force and z torque correspond to an off-centre tangential load.
        s1.body_f.assign([[1, 0, 0, 0, 0, 0.05]])
        solver.step(s1, s0, model.control(), contacts, dt)
        self.assertGreater(np.count_nonzero(solver._friction_patches.current.source.numpy() >= 0), 0)
        count = int(solver.mf_constraint_count.numpy()[0])
        self.assertTrue(np.all(solver.contact_path.numpy()[:4] == 1))
        solved = np.einsum("ij,i->j", solver.mf_J_a.numpy()[0, :count], solver.mf_impulses.numpy()[0, :count]) / dt
        self.assertGreater(np.linalg.norm(solved[3:]), 0.01, "fixture must exercise a nonzero contact moment")
        solver.update_contacts(contacts)
        reported = contacts.force.numpy().sum(axis=0)
        np.testing.assert_allclose(reported[:3], solved[:3], atol=1.0e-4)
        # Torque is explicitly unsupported by this export, including on the base
        # branch. Preserve that limitation visibly instead of asserting a wrench.
        np.testing.assert_array_equal(reported[3:], [0, 0, 0])

    def test_shape_translation_retires_anchors_outside_new_geometry(self):
        """Retire the old contact footprint after changing a shape's local transform."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        builder = newton.ModelBuilder(gravity=wp.vec3(0, 0, 0))
        builder.add_ground_plane()
        body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.1), wp.quat_identity()))
        box = builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device=device)
        solver = newton.solvers.SolverFeatherPGS(model, friction_anchor_beta=0.2)
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=64)
        contacts = pipeline.contacts()
        s0, s1 = model.state(), model.state()
        pipeline.collide(s0, contacts)
        solver.step(s0, s1, model.control(), contacts, 0.005)
        patches = solver._friction_patches
        self.assertGreater(np.count_nonzero(patches.previous.valid.numpy()), 0)

        transforms = model.shape_transform.numpy()
        transforms[box, 0] = 1.0
        model.shape_transform.assign(transforms)
        solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
        pipeline.collide(s1, contacts)
        patches.build(model, s1, contacts)
        active = patches.current.valid.numpy() != 0
        self.assertEqual(np.count_nonzero(active), 2)
        np.testing.assert_array_equal(patches.current.source.numpy()[active], [-1, -1])
        local_x = patches.current.anchor_a.numpy()[active, 0]
        self.assertTrue(np.all(local_x >= 0.89), "friction still acts on the removed x=-0.1..0.1 footprint")

    def test_rejected_contact_history_keeps_correct_reset_world(self):
        """Keep world ownership valid when contact order changes and normal gates reject rows."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        template = newton.ModelBuilder(gravity=wp.vec3(0, 0, 0))
        template.add_ground_plane()
        body = template.add_body()
        template.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        builder = newton.ModelBuilder(gravity=wp.vec3(0, 0, 0))
        builder.replicate(template, 2)
        model = builder.finalize(device=device)
        solver = newton.solvers.SolverFeatherPGS(model, friction_anchor_beta=0.2, contact_gap_gate=0.001)
        contacts = newton.CollisionPipeline(model, rigid_contact_max=2).contacts()
        shape_body = model.shape_body.numpy()
        boxes = np.flatnonzero(shape_body >= 0)
        grounds = np.flatnonzero(shape_body < 0)
        contacts.rigid_contact_count.assign([2])
        contacts.rigid_contact_shape0.assign(boxes)
        contacts.rigid_contact_shape1.assign(grounds)
        contacts.rigid_contact_normal.assign([[0, 0, -1]] * 2)
        contacts.rigid_contact_point0.zero_()
        contacts.rigid_contact_point1.zero_()
        s0, s1 = model.state(), model.state()
        solver.step(s0, s1, model.control(), contacts, 0.005)
        patches = solver._friction_patches
        self.assertEqual(int(patches.previous.valid.numpy().sum()), 2)

        contacts.rigid_contact_shape0.assign(boxes[::-1].copy())
        contacts.rigid_contact_shape1.assign(grounds[::-1].copy())
        contacts.rigid_contact_point0.assign([[0, 0, 0.002]] * 2)
        solver.step(s1, s0, model.control(), contacts, 0.005)
        self.assertTrue((solver.contact_slot.numpy()[:2] == -1).all())
        bodies = patches.current.body_a.numpy()[:2]
        worlds = model.body_world.numpy()[bodies]
        np.testing.assert_array_equal(patches.previous_world.numpy()[:2], worlds)
        solver.reset(s0, wp.array([True, False], dtype=bool, device=device))
        valid = patches.previous.valid.numpy()[:2]
        np.testing.assert_array_equal(valid, (worlds != 0).astype(np.int32))

    def test_incompatible_point_solvers_are_rejected(self):
        """Reject coupled point solves that would silently consume a patch's shared normal load."""
        builder = newton.ModelBuilder()
        body = builder.add_body()
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device="cpu")
        with self.assertRaisesRegex(ValueError, "Patch friction requires"):
            newton.solvers.SolverFeatherPGS(model, friction_anchor_beta=0.2, friction_mode="bisection")
        # CPU resolves every native kernel selector to the scalar loop, so only CUDA rejects them.
        solver = newton.solvers.SolverFeatherPGS(model, friction_anchor_beta=0.2, pgs_kernel="tiled_contact")
        self.assertEqual(solver.pgs_kernel, "loop")
        if wp.is_cuda_available():
            cuda_model = builder.finalize(device="cuda:0")
            for kernel in ("tiled_contact", "streaming"):
                with self.subTest(pgs_kernel=kernel), self.assertRaisesRegex(ValueError, "Patch friction requires"):
                    newton.solvers.SolverFeatherPGS(cuda_model, friction_anchor_beta=0.2, pgs_kernel=kernel)

    def test_deprecated_anchor_limit_warns_instead_of_raising(self):
        """Ignore the deprecated anchor limit without overriding the default or explicit opt-out."""
        builder = newton.ModelBuilder()
        body = builder.add_body()
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device="cpu")
        with self.assertWarns(DeprecationWarning):
            solver = newton.solvers.SolverFeatherPGS(model, contact_friction_anchor_limit=2)
        self.assertTrue(solver._friction_anchors_enabled)
        self.assertAlmostEqual(solver.friction_anchor_beta, 0.2)
        if wp.is_cuda_available():
            # Coupled friction modes need the CUDA matrix-free route; the shim must warn and stay off there.
            cuda_model = builder.finalize(device="cuda:0")
            with self.assertWarns(DeprecationWarning):
                solver = newton.solvers.SolverFeatherPGS(
                    cuda_model,
                    contact_friction_anchor_limit=2,
                    friction_mode="bisection",
                    pgs_mode="matrix_free",
                    friction_anchor_beta=0.0,
                )
            self.assertFalse(solver._friction_anchors_enabled)
        with self.assertWarns(DeprecationWarning):
            solver = newton.solvers.SolverFeatherPGS(model, contact_friction_anchor_limit=2, pgs_kernel="tiled_contact")
        self.assertTrue(solver._friction_anchors_enabled)
        with self.assertWarns(DeprecationWarning):
            solver = newton.solvers.SolverFeatherPGS(model, contact_friction_anchor_limit=2, friction_anchor_beta=0.3)
        self.assertAlmostEqual(solver.friction_anchor_beta, 0.3)
        with self.assertWarns(DeprecationWarning):
            solver = newton.solvers.SolverFeatherPGS(model, contact_friction_anchor_limit=2, friction_anchor_beta=0.0)
        self.assertFalse(solver._friction_anchors_enabled)

    def test_anchor_selection_respects_contact_gap_filters(self):
        """Keep filtered extreme points from removing friction from a loaded middle contact."""
        for gate in ("contact_friction_gap_threshold", "contact_gap_gate"):
            with self.subTest(gate=gate), wp.ScopedDevice("cpu"):
                builder = newton.ModelBuilder()
                ground = builder.add_ground_plane()
                body = builder.add_body()
                box = builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
                model = builder.finalize()
                solver = newton.solvers.SolverFeatherPGS(model, friction_anchor_beta=0.2, **{gate: 0.001})
                contacts = newton.CollisionPipeline(model, rigid_contact_max=3).contacts()
                contacts.rigid_contact_count.assign([3])
                contacts.rigid_contact_shape0.assign([box] * 3)
                contacts.rigid_contact_shape1.assign([ground] * 3)
                contacts.rigid_contact_point0.assign([[-0.08, 0, 0.002], [0, 0, 0], [0.08, 0, 0.002]])
                contacts.rigid_contact_point1.assign([[-0.08, 0, 0], [0, 0, 0], [0.08, 0, 0]])
                contacts.rigid_contact_normal.assign([[0, 0, -1]] * 3)
                solver.step(model.state(), model.state(), model.control(), contacts, 0.005)
                rows = solver.mf_row_type.numpy()[0, : solver.mf_constraint_count.numpy()[0]]
                self.assertEqual(np.count_nonzero(rows == PGS_CONSTRAINT_TYPE_FRICTION), 2)
                np.testing.assert_allclose(solver._friction_patches.view.weight.numpy()[:3], [0, 1, 0])

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_propagation_variants_preserve_a_warmstarted_patch(self):
        """Retain a sticking articulated grasp across cached, fused, and colored propagation."""
        for response in ("propagation", "propagation-fused", "propagation-colored"):
            with self.subTest(response=response):
                drift, solver, state = _run_squeeze(
                    5.0,
                    220,
                    friction_anchor_beta=0.2,
                    pgs_warmstart=True,
                    contact_friction_position_iterations=-1,
                    pgs_contact_regularization=0.0,
                    articulated_contact_response=response,
                )
                self.assertTrue(np.isfinite(state.body_q.numpy()).all())
                self.assertLess(abs(drift), 2.0e-4)
                self.assertGreater(np.count_nonzero(solver._friction_patches.current.valid.numpy()), 0)

    def test_shape_updates_refresh_geometry_and_keep_history(self):
        """Refresh broadphase bounds without discarding anchors on unchanged geometry."""
        builder = newton.ModelBuilder()
        body = builder.add_body()
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device="cpu")
        solver = newton.solvers.SolverFeatherPGS(model, friction_anchor_beta=0.2)
        patches = solver._friction_patches
        radius = patches.body_radius.numpy().copy()
        patches.previous.valid.fill_(1)
        model.shape_collision_radius.assign(model.shape_collision_radius.numpy() * 2)
        solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
        np.testing.assert_allclose(patches.body_radius.numpy(), 2 * radius)
        self.assertEqual(np.count_nonzero(patches.previous.valid.numpy()), patches.previous.valid.shape[0])

    def test_patch_resists_twist_and_releases_above_its_limit(self):
        """Resist a static yaw torque with separated anchors, but permit a larger torque to spin."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        for torque in (0.2, 1.0):
            with self.subTest(torque=torque), wp.ScopedDevice(device):
                builder = newton.ModelBuilder()
                builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
                body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.1), wp.quat_identity()))
                builder.add_shape_box(
                    body, hx=0.1, hy=0.1, hz=0.1, cfg=newton.ModelBuilder.ShapeConfig(density=125, mu=0.5)
                )
                model = builder.finalize()
                solver = newton.solvers.SolverFeatherPGS(
                    model,
                    friction_anchor_beta=0.2,
                    pgs_iterations=64,
                    pgs_mode="matrix_free" if wp.is_cuda_available() else "split",
                )
                pipeline = newton.CollisionPipeline(model, rigid_contact_max=64)
                contacts = pipeline.contacts()
                s0, s1 = model.state(), model.state()
                control = model.control()
                for _ in range(100):
                    s0.body_f.assign([[0, 0, 0, 0, 0, torque]])
                    pipeline.collide(s0, contacts)
                    solver.step(s0, s1, control, contacts, 0.005)
                    s0, s1 = s1, s0
                angular_speed = abs(float(s0.body_qd.numpy()[0, 5]))
                self.assertTrue(np.isfinite(s0.body_q.numpy()).all())
                if torque == 0.2:
                    self.assertLess(angular_speed, 0.01)
                    self.assertLess(abs(float(s0.body_q.numpy()[0, 5])), 0.01)
                else:
                    self.assertGreater(angular_speed, 1.0)

    def test_pooled_coulomb_budget(self):
        """Share the loaded middle normal's budget across two unloaded anchor normals."""
        self._check_pooled_projection("cpu", native=False)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_native_pooled_coulomb_budget(self):
        """Use the complete patch's normal load in native CUDA projection."""
        self._check_pooled_projection("cuda:0", native=True)

    def _check_pooled_projection(self, device, native):
        capacity = 32  # Native row kernels use a full warp of storage.
        count = wp.array([7], dtype=int, device=device)
        diag = wp.ones((1, capacity), dtype=float, device=device)
        matrix = wp.array(np.eye(capacity, dtype=np.float32)[None], dtype=float, device=device)
        rhs = wp.array([[0, -100, 0, -4, 0, -100, 0] + [0] * 25], dtype=float, device=device)
        impulses = wp.zeros((1, capacity), dtype=float, device=device)
        rows = wp.array([[0, 2, 2, 0, 0, 2, 2] + [-1] * 25], dtype=int, device=device)
        parents = wp.array([[3, 0, 0, 4, 0, 4, 4] + [-1] * 25], dtype=int, device=device)
        mu = wp.array([[0.5, 0.25, 0.25, 0.5, 0.5, 0.25, 0.25] + [0] * 25], dtype=float, device=device)
        args = [count, diag, matrix, rhs, impulses, 2, 1.0, rows, parents, mu, 0, 0]
        if native:
            kernel = _get_pgs_solve_tiled_row_kernel(capacity, str(wp.get_device(device).arch))
            wp.launch_tiled(kernel, dim=[1], inputs=args, block_dim=32, device=device)
        else:
            wp.launch(pgs_solve_loop, dim=1, inputs=[count, capacity, *args[1:]], device=device)
        np.testing.assert_allclose(impulses.numpy()[0], [0, 1, 0, 4, 0, 1, 0] + [0] * 25, atol=1.0e-6)

    def test_planar_patch_reduces_friction_rows(self):
        """Retain a box face's four normals while using only two friction anchors."""
        self._check_planar_patch("cpu", "split")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_cuda_planar_patch_reduces_friction_rows(self):
        """Solve the same patch layout with native split and fused kernels."""
        for mode in ("split", "matrix_free"):
            with self.subTest(mode=mode):
                self._check_planar_patch("cuda:0", mode)

    def _check_planar_patch(self, device, mode):
        with wp.ScopedDevice(device):
            builder = newton.ModelBuilder()
            builder.add_ground_plane()
            body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.099), wp.quat_identity()))
            builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
            model = builder.finalize()
            solver = newton.solvers.SolverFeatherPGS(
                model, friction_anchor_beta=0.2, dense_max_constraints=64, pgs_mode=mode
            )
            pipeline = newton.CollisionPipeline(model, rigid_contact_max=64, contact_matching="latest")
            contacts = pipeline.contacts()
            state = model.state()
            pipeline.collide(state, contacts)
            solver.step(state, model.state(), model.control(), contacts, 0.005)
            rows = np.concatenate(
                [
                    solver.row_type.numpy()[0, : solver.constraint_count.numpy()[0]],
                    solver.mf_row_type.numpy()[0, : solver.mf_constraint_count.numpy()[0]],
                ]
            )
            normals = int(np.count_nonzero(rows == PGS_CONSTRAINT_TYPE_CONTACT))
            friction = int(np.count_nonzero(rows == PGS_CONSTRAINT_TYPE_FRICTION))
            self.assertEqual(normals, 4)
            self.assertEqual(friction, 4)

    def test_contact_forces_report_per_row_normal_load(self):
        """Report each contact's own normal force so a resting box's contact forces sum to its weight."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        with wp.ScopedDevice(device):
            model, solver, contacts, step = _ground_box(device)
            for _ in range(100):
                step()
            solver.update_contacts(contacts)
            count = int(contacts.rigid_contact_count.numpy()[0])
            forces = contacts.rigid_contact_force.numpy()[:count]
        weight = float(model.body_mass.numpy()[0]) * 9.81
        self.assertGreaterEqual(count, 3)
        self.assertAlmostEqual(float(abs(forces[:, 2].sum())), weight, delta=0.05 * weight)
        self.assertLess(float(np.abs(forces[:, 2]).max()), weight)

    def test_gap_filtered_step_keeps_anchor_history(self):
        """Carry a region's anchors through a step in which the gap filter removes every friction row."""
        device = "cuda:0" if wp.is_cuda_available() else "cpu"
        with wp.ScopedDevice(device):
            _model, solver, _contacts, step = _ground_box(device, contact_friction_gap_threshold=0.001)
            patches = solver._friction_patches
            for _ in range(40):
                step()
            self.assertGreater(int((patches.current.source.numpy() >= 0).sum()), 0)
            solver.contact_friction_gap_threshold = -1.0
            step()
            self.assertEqual(float(patches.view.weight.numpy().max()), 0.0)
            self.assertGreater(int(patches.previous.valid.numpy().sum()), 0)
            solver.contact_friction_gap_threshold = 0.001
            step()
            weights = patches.view.weight.numpy()
            sources = patches.current.source.numpy()
            self.assertGreater(int(((weights > 0) & (sources >= 0)).sum()), 0)


if __name__ == "__main__":
    unittest.main()
