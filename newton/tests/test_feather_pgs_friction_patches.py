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


def _patch_fixture(
    points, *, shape0=None, normals=None, materials=(0.5, 0.5, 0.5), shape_bodies=(0, 1, 0), device="cpu"
):
    """Construct patch geometry without a collision matcher or a solver step."""
    n = len(points)
    model = SimpleNamespace(
        device=wp.get_device(device),
        body_count=2,
        shape_body=wp.array(shape_bodies, dtype=int, device=device),
        shape_collision_radius=wp.array([0.2] * len(shape_bodies), dtype=float, device=device),
        shape_transform=wp.array([wp.transform_identity()] * len(shape_bodies), dtype=wp.transform, device=device),
        shape_material_mu=wp.array(materials, dtype=float, device=device),
    )
    state = SimpleNamespace(body_q=wp.array([wp.transform_identity()] * 2, dtype=wp.transform, device=device))
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
    def test_connected_convex_chain_forms_one_region(self):
        """Connectivity joins a chain even when the two end shapes do not overlap."""
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
        """Two separated pads on one body must not pool friction through empty space."""
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
        """Diagnostics distinguish supported friction, a cone violation, and normal error."""
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
        """A body pair has one identity regardless of collision shape ordering."""
        model, state, contacts, patches = _patch_fixture([[-0.1, 0, 0], [0.1, 0, 0]])
        contacts.rigid_contact_shape0.assign([0, 1])
        contacts.rigid_contact_shape1.assign([1, 0])
        contacts.rigid_contact_normal.assign([[0, 0, -1], [0, 0, 1]])
        patches.build(model, state, contacts)
        self.assertEqual(len(np.unique(patches.current.owner.numpy()[:2])), 1)
        np.testing.assert_allclose(patches.view.weight.numpy()[:2], [0.5, 0.5])

    def test_saturation_only_releases_an_anchor_with_sliding_motion(self):
        """Impending slip keeps history; motion against saturated friction releases it."""
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
        """Adjacent convex pieces with compatible materials pool their friction load."""
        _, _, _, patches = _patch_fixture(
            [[-0.1, -0.1, 0], [0.1, -0.1, 0], [-0.1, 0.1, 0], [0.1, 0.1, 0]], shape0=[0, 2, 0, 2]
        )
        self.assertEqual(np.count_nonzero(patches.view.weight.numpy()), 2)
        self.assertEqual(len(np.unique(patches.current.owner.numpy()[:4])), 1)
        self.assertAlmostEqual(float(patches.view.weight.numpy().sum()), 1.0)

    def test_material_normal_and_disconnected_regions_stay_separate(self):
        """Friction is not pooled across incompatible or spatially separate regions."""
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
        """Changing every contact sample does not erase a sticking region's history."""
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
        """Duplicate witnesses must not create two coincident friction constraints."""
        _, _, _, patches = _patch_fixture([[0, 0, 0]] * 4)
        self.assertEqual(np.count_nonzero(patches.view.weight.numpy()), 1)
        self.assertEqual(float(patches.view.weight.numpy().sum()), 1.0)

    def test_warmstart_uses_patch_load_when_anchor_normal_is_unloaded(self):
        """A cached tangent remains supported when other normals carry the load."""
        model, state, contacts, patches = _patch_fixture([[-0.1, 0, 0], [0, 0, 0], [0.1, 0, 0]])
        patches.current.tangent_impulse.assign([[0.8, 0, 0], [0, 0, 0], [0.8, 0, 0]] + [[0, 0, 0]] * 5)
        patches.store(state)
        patches.build(model, state, contacts)
        slots = wp.array([0, 3, 4], dtype=int, device="cpu")
        worlds = wp.zeros(3, dtype=int, device="cpu")
        paths = wp.zeros(3, dtype=int, device="cpu")
        lengths = wp.array([3, 1, 3], dtype=int, device="cpu")
        parents = wp.array([[-1, 0, 0, -1, -1, 4, 4]], dtype=int, device="cpu")
        mu = wp.array([[0.5, 0.25, 0.25, 0.5, 0.5, 0.25, 0.25]], dtype=float, device="cpu")
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


class TestFeatherPGSFrictionPatches(unittest.TestCase):
    def test_incompatible_point_solvers_are_rejected(self):
        """Coupled point solves cannot silently consume a patch's shared normal load."""
        builder = newton.ModelBuilder()
        body = builder.add_body()
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        model = builder.finalize(device="cpu")
        for config in ({"friction_mode": "bisection"}, {"pgs_kernel": "tiled_contact"}, {"pgs_kernel": "streaming"}):
            with self.subTest(config=config), self.assertRaisesRegex(ValueError, "Patch friction requires"):
                newton.solvers.SolverFeatherPGS(model, friction_anchor_beta=0.2, **config)

    def test_anchor_selection_respects_contact_gap_filters(self):
        """Filtered extreme points cannot remove friction from a loaded middle contact."""
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
        """Cached, fused, and colored propagation retain a sticking articulated grasp."""
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

    def test_shape_updates_refresh_geometry_and_clear_history(self):
        """Explicit shape edits cannot retain anchors correlated with old geometry."""
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
        self.assertEqual(np.count_nonzero(patches.previous.valid.numpy()), 0)

    def test_patch_resists_twist_and_releases_above_its_limit(self):
        """Separated anchors resist a static yaw torque but permit a larger torque to spin."""
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
        """Two unloaded anchor normals share the loaded middle normal's budget."""
        self._check_pooled_projection("cpu", native=False)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_native_pooled_coulomb_budget(self):
        """Native CUDA projection uses the complete patch's normal load."""
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
        """A box face retains its four normals and uses only two friction anchors."""
        self._check_planar_patch("cpu", "split")

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
    def test_cuda_planar_patch_reduces_friction_rows(self):
        """Native split and fused kernels solve the same patch layout."""
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


if __name__ == "__main__":
    unittest.main()
