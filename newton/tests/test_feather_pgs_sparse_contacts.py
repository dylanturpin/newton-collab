# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.friction_patches import FrictionPatches
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    apply_world_contact_restitution_matrix_free,
    populate_world_J_for_compact_size,
    prepare_world_contact_rows,
)
from newton._src.solvers.feather_pgs.sparse_contact import (
    apply_sparse_contact_restitution,
    apply_sparse_factor_velocity,
    build_sparse_joint_limit_rows,
    populate_sparse_contact_response,
)
from newton._src.solvers.feather_pgs.sparse_mass_matrix import _SparseMassMatrixPlan


def _array(value, dtype=float):
    return wp.array(value, dtype=dtype, device="cpu")


def _full(shape, value, dtype=float):
    return wp.full(shape, value, dtype=dtype, device="cpu")


def _fixture():
    """Make a two-branch articulation with a multi-DOF root and fixed endpoint."""
    plan = _SparseMassMatrixPlan.build([-1, 0, 0, 1], [3, 2, 1, 0])
    rng = np.random.default_rng(916)
    lower = np.zeros((6, 6))
    lower[plan.entry_rows, plan.columns] = rng.normal(scale=0.3, size=plan.nonzero_count)
    lower[np.diag_indices(6)] = 2.0
    inverse = np.linalg.inv(lower)
    patches = FrictionPatches()
    patches.enabled = 0
    patches.weight = _array([1.0, 1.0, 0.0])
    patches.next_contact = _array([-1, -1, -1], int)
    patches.point_a = _array(rng.normal(size=(3, 3)), wp.vec3)
    patches.point_b = _array(rng.normal(size=(3, 3)), wp.vec3)
    patches.phi = _array(np.zeros((3, 2)), wp.vec2)
    normals = rng.normal(size=(3, 3))
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    physical_masks = [sum(1 << int(plan.permutation[i]) for i in plan.endpoint_support(j)) for j in range(4)]
    return {
        "plan": plan,
        "indices": plan.to_device("cpu"),
        "inverse": inverse,
        "packed": _array(inverse[plan.entry_rows, plan.columns][None, :]),
        "patches": patches,
        "count": _array([3], int),
        "point0": _array(rng.normal(size=(3, 3)), wp.vec3),
        "point1": _array(rng.normal(size=(3, 3)), wp.vec3),
        "normal": _array(normals, wp.vec3),
        "shape0": _array([0, 0, 2], int),
        "shape1": _array([3, 1, 3], int),
        "thickness0": _array([0.01, 0.02, 0.03]),
        "thickness1": _array([0.03, 0.01, 0.02]),
        "world": _array([0, 0, 0], int),
        "slot": _array([0, 3, 6], int),
        "art_a": _array([0, 0, 0], int),
        "art_b": _array([-1, 0, -1], int),
        "path": _array([0, 0, 0], int),
        "needed": _array([3, 3, 1], int),
        "group": _array([0], int),
        "start": _array([0], int),
        "origin": _array([[0.3, -0.4, 0.7]], wp.vec3),
        "sparse_mask": _array(plan.joint_ancestor_mask, wp.uint64),
        "physical_mask": _array(physical_masks, wp.uint32),
        "motion": _array(rng.normal(size=(6, 6)), wp.spatial_vector),
        "shape_body": _array([1, 2, 3, -1], int),
        "body_q": _array(
            [wp.transform(wp.vec3(*p), wp.quat_identity()) for p in rng.normal(size=(4, 3))], wp.transform
        ),
        "velocity": _array(rng.normal(size=6)),
    }


class TestSparseContacts(unittest.TestCase):
    def test_contact_response_matches_dense_jacobian(self):
        """Match dense contact rows for static, same-articulation, and fixed endpoints."""
        f = _fixture()
        p = f["plan"]
        indices = f["indices"]
        geometry = [f[name] for name in ("point0", "point1", "normal", "shape0", "shape1", "thickness0", "thickness1")]
        for shared, friction_shared, patch in [(0, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1), (1, 1, 1)]:
            with self.subTest(shared=shared, friction_shared=friction_shared, patch=patch):
                f["patches"].enabled = patch
                dense = _full((1, 9, 6), 0.0)
                row_dof = _full((1, 9, 6), -9, int)
                row_factor = _full((1, 9, 6), -9.0)
                incident = _full((1, 9), -9.0)
                diagonal = _full((1, 9), -9.0)
                wp.launch(
                    populate_world_J_for_compact_size,
                    dim=(1, 32),
                    inputs=[
                        f["count"],
                        1,
                        *geometry,
                        f["slot"],
                        f["art_a"],
                        f["art_b"],
                        f["path"],
                        f["needed"],
                        6,
                        _array([6], int),
                        f["group"],
                        f["start"],
                        f["origin"],
                        f["physical_mask"],
                        f["motion"],
                        f["shape_body"],
                        f["body_q"],
                        friction_shared,
                        f["patches"],
                        shared,
                    ],
                    outputs=[dense],
                    device="cpu",
                )
                wp.launch(
                    populate_sparse_contact_response,
                    dim=(1, 3),
                    inputs=[
                        f["count"],
                        1,
                        *geometry,
                        f["world"],
                        f["slot"],
                        f["art_a"],
                        f["art_b"],
                        f["path"],
                        f["needed"],
                        f["group"],
                        f["start"],
                        f["origin"],
                        f["sparse_mask"],
                        f["motion"],
                        f["shape_body"],
                        f["body_q"],
                        friction_shared,
                        f["patches"],
                        shared,
                        indices.permutation,
                        indices.row_offsets,
                        indices.columns,
                        f["packed"],
                        f["velocity"],
                    ],
                    outputs=[row_dof, row_factor, incident, diagonal],
                    device="cpu",
                )
                jacobian = dense.numpy()[0, :7]
                expected = jacobian[:, p.permutation] @ f["inverse"].T
                actual = np.zeros_like(expected)
                for row in range(7):
                    dofs = row_dof.numpy()[0, row]
                    valid = dofs >= 0
                    self.assertEqual(len(set(dofs[valid])), int(np.count_nonzero(valid)))
                    actual[row, dofs[valid]] = row_factor.numpy()[0, row, valid]
                np.testing.assert_allclose(actual, expected, rtol=3.0e-6, atol=2.0e-6)
                np.testing.assert_allclose(
                    incident.numpy()[0, :7], jacobian @ f["velocity"].numpy(), rtol=3.0e-6, atol=2.0e-6
                )
                np.testing.assert_allclose(
                    diagonal.numpy()[0, :7], np.sum(expected**2, axis=1), rtol=3.0e-6, atol=2.0e-6
                )
                np.testing.assert_array_equal(row_dof.numpy()[0, 7:], -9)
                np.testing.assert_array_equal(row_factor.numpy()[0, 7:], -9.0)

    def test_contact_metadata_honors_allocated_extent(self):
        """Keep reserved one-row contacts from writing newly eligible friction rows."""
        f = _fixture()
        # Friction would pass a recomputed geometric/material predicate. The
        # allocator's one-row extent is authoritative despite that discrepancy.
        f["count"] = _array([1], int)
        f["needed"] = _array([1, 3, 1], int)
        row_type = _full((1, 9), -9, int)
        row_parent = _full((1, 9), -9, int)
        other = [_full((1, 9), -9.0) for _ in range(6)]
        wp.launch(
            prepare_world_contact_rows,
            dim=1,
            inputs=[
                f["count"],
                1,
                f["point0"],
                f["point1"],
                f["normal"],
                f["shape0"],
                f["shape1"],
                f["thickness0"],
                f["thickness1"],
                f["world"],
                f["slot"],
                f["art_a"],
                f["art_b"],
                f["path"],
                f["needed"],
                f["shape_body"],
                f["body_q"],
                _full(4, wp.spatial_vector(0.0), wp.spatial_vector),
                _array([0], int),
                f["origin"],
                _array([0.8] * 4),
                _array([0.4] * 4),
                0,
                1.0,
                0,
                0.05,
                1.0e-6,
                f["patches"],
                0.0,
            ],
            outputs=[row_type, row_parent, *other],
            device="cpu",
        )
        self.assertEqual(row_type.numpy()[0, 0], PGS_CONSTRAINT_TYPE_CONTACT)
        for output in [row_type, row_parent, *other]:
            np.testing.assert_array_equal(output.numpy()[0, 1:], -9)

    def test_joint_limit_rows_and_capacity(self):
        """Match signed inverse-factor columns and preserve limit allocation order."""
        f = _fixture()
        plan, indices = f["plan"], f["indices"]
        expected_rows = [(0, 1.0, -0.02), (1, -1.0, 0.01), (3, -1.0, -0.2), (4, 1.0, 0.0), (4, -1.0, 0.0)]
        for capacity in (6, 3):
            with self.subTest(capacity=capacity):
                count = _array([1], int)
                row_type, parent = [_full((1, capacity), -9, int) for _ in range(2)]
                mu, beta, cfm, phi, target, incident, diagonal = [_full((1, capacity), -9.0) for _ in range(7)]
                dof = _full((1, capacity, 6), -9, int)
                factor = _full((1, capacity, 6), -9.0)
                wp.launch(
                    build_sparse_joint_limit_rows,
                    dim=32,
                    inputs=[
                        f["group"],
                        _array([0], int),
                        f["start"],
                        _array([0, 1, -1, 3, 4, 5], int),
                        _array([-1.0, -1.0, -np.inf, -0.5, 0.0, -2.0]),
                        _array([1.0, 1.0, np.inf, 0.5, 0.0, 2.0]),
                        _array([-1.02, 0.99, 0.0, 0.7, 0.0, -1.0]),
                        0.02,
                        0.05,
                        1.0e-6,
                        indices.ancestor_mask,
                        indices.inverse_permutation,
                        indices.lookup,
                        f["packed"],
                        f["velocity"],
                    ],
                    outputs=[count, row_type, parent, mu, beta, cfm, phi, target, dof, factor, incident, diagonal],
                    block_dim=128,
                    device="cpu",
                )
                self.assertEqual(count.numpy()[0], 6)
                self.assertEqual(row_type.numpy()[0, 0], -9)
                for row, (physical, sign, gap) in enumerate(expected_rows[: capacity - 1], 1):
                    expected = sign * f["inverse"][:, plan.inverse_permutation[physical]]
                    actual = np.zeros(6)
                    active = dof.numpy()[0, row] >= 0
                    actual[dof.numpy()[0, row, active]] = factor.numpy()[0, row, active]
                    np.testing.assert_allclose(actual, expected, atol=1.0e-7)
                    self.assertEqual(row_type.numpy()[0, row], PGS_CONSTRAINT_TYPE_JOINT_LIMIT)
                    self.assertEqual(parent.numpy()[0, row], -1)
                    self.assertAlmostEqual(float(phi.numpy()[0, row]), gap, places=6)
                    self.assertAlmostEqual(
                        float(incident.numpy()[0, row]), sign * f["velocity"].numpy()[physical], places=6
                    )
                    self.assertAlmostEqual(float(diagonal.numpy()[0, row]), float(expected @ expected), places=6)

    def _check_joint_limit_parallel_scan(self, device, capacity):
        plan = _SparseMassMatrixPlan.build([-1, 0, 0], [2, 31, 31])
        dofs, worlds = plan.dof_count, 3
        rng = np.random.default_rng(923)
        lower_factor = np.zeros((dofs, dofs))
        lower_factor[plan.entry_rows, plan.columns] = rng.normal(scale=0.02, size=plan.nonzero_count)
        lower_factor[np.diag_indices(dofs)] = 1.5
        inverse = np.linalg.inv(lower_factor)
        packed = np.stack([(1.0 + group * 0.1) * inverse[plan.entry_rows, plan.columns] for group in range(worlds)])
        group_to_art = np.array([2, 0, 1], dtype=np.int32)
        art_to_world = np.array([1, 2, 0], dtype=np.int32)
        start = np.arange(worlds, dtype=np.int32) * dofs
        q_index = np.arange(worlds * dofs, dtype=np.int32)
        lower = np.full((worlds, dofs), -1.0, dtype=np.float32)
        upper = np.full((worlds, dofs), 1.0, dtype=np.float32)
        position = np.zeros((worlds, dofs), dtype=np.float32)
        # Articulation zero has no active rows. The others cross warp and 64-bit mask boundaries.
        position[1, [0, 15, 32]] = -1.0
        position[1, [16, 47]] = 1.0
        lower[1, [31, 63]] = upper[1, [31, 63]] = 0.0
        lower[2] = upper[2] = 0.0
        q_index[2 * dofs + 5] = -1
        lower[2, 7] = -np.inf
        upper[2, 8] = np.inf
        lower[2, 9] = np.nan
        velocity = rng.normal(size=worlds * dofs).astype(np.float32)
        initial_count = np.array([1, 2, 3], dtype=np.int32)

        def array(value, dtype=float):
            return wp.array(value, dtype=dtype, device=device)

        def full(shape, value, dtype=float):
            return wp.full(shape, value, dtype=dtype, device=device)

        indices = plan.to_device(device)
        counter = array(initial_count, int)
        row_type, parent = [full((worlds, capacity), -9, int) for _ in range(2)]
        mu, beta, cfm, phi, target, incident, diagonal = [full((worlds, capacity), -9.0) for _ in range(7)]
        row_dof = full((worlds, capacity, 34), -9, int)
        row_factor = full((worlds, capacity, 34), -9.0)
        outputs = [counter, row_type, parent, mu, beta, cfm, phi, target, row_dof, row_factor, incident, diagonal]
        wp.launch(
            build_sparse_joint_limit_rows,
            dim=worlds * 32,
            inputs=[
                array(group_to_art, int),
                array(art_to_world, int),
                array(start, int),
                array(q_index, int),
                array(lower.reshape(-1)),
                array(upper.reshape(-1)),
                array(position.reshape(-1)),
                0.01,
                0.05,
                1.0e-6,
                indices.ancestor_mask,
                indices.inverse_permutation,
                indices.lookup,
                array(packed),
                array(velocity),
            ],
            outputs=outputs,
            block_dim=128,
            device=device,
        )
        actual = [value.numpy() for value in outputs]
        for group, art in enumerate(group_to_art):
            world = art_to_world[art]
            first = initial_count[world]
            expected = []
            for local_dof in range(dofs):
                if q_index[start[art] + local_dof] < 0:
                    continue
                for side in (0, 1):
                    bound = (lower if side == 0 else upper)[art, local_dof]
                    q = position[art, local_dof]
                    active = q <= bound + 0.01 if side == 0 else q >= bound - 0.01
                    if np.isfinite(bound) and active:
                        expected.append((local_dof, 1.0 if side == 0 else -1.0, q - bound))
            self.assertEqual(actual[0][world], first + len(expected))
            for output in actual[1:]:
                np.testing.assert_array_equal(output[world, :first], -9)
                np.testing.assert_array_equal(output[world, first + len(expected) :], -9)
            for row, (local_dof, sign, gap) in enumerate(expected[: capacity - first], first):
                support = np.flatnonzero([int(plan.ancestor_mask[local_dof]) & (1 << bit) for bit in range(dofs)])
                np.testing.assert_array_equal(actual[8][world, row, : len(support)], support)
                np.testing.assert_array_equal(actual[8][world, row, len(support) :], -1)
                np.testing.assert_array_equal(actual[9][world, row, len(support) :], 0.0)
                expected_factor = sign * (1.0 + group * 0.1) * inverse[support, plan.inverse_permutation[local_dof]]
                np.testing.assert_allclose(actual[9][world, row, : len(support)], expected_factor, atol=1.0e-7)
                self.assertEqual(actual[1][world, row], PGS_CONSTRAINT_TYPE_JOINT_LIMIT)
                self.assertEqual(actual[2][world, row], -1)
                self.assertEqual(actual[3][world, row], 0.0)
                self.assertAlmostEqual(actual[4][world, row], 0.05, places=7)
                self.assertAlmostEqual(actual[5][world, row], 1.0e-6, places=9)
                self.assertAlmostEqual(actual[6][world, row], sign * gap, places=7)
                self.assertEqual(actual[7][world, row], 0.0)
                self.assertAlmostEqual(actual[10][world, row], sign * velocity[start[art] + local_dof], places=7)
                self.assertAlmostEqual(actual[11][world, row], expected_factor @ expected_factor, places=6)

    def test_joint_limit_parallel_scan_cpu(self):
        """Retain ordered rows, full overflow demand and zero-row worlds across 64 DOFs."""
        for capacity in (8, 140):
            with self.subTest(capacity=capacity):
                self._check_joint_limit_parallel_scan("cpu", capacity)

    @unittest.skipUnless(wp.is_cuda_available(), "cooperative limit construction requires CUDA")
    def test_joint_limit_parallel_scan_cuda(self):
        """Match cooperative ballot ordering and responses to the scalar CPU oracle."""
        for capacity in (8, 140):
            with self.subTest(capacity=capacity):
                self._check_joint_limit_parallel_scan("cuda:0", capacity)

    def test_restitution_matches_dense_incident_projection(self):
        """Preserve impact selection and target subtraction from the dense matrix-free law."""
        row_count = 7
        kinds = _array([[PGS_CONSTRAINT_TYPE_CONTACT] * 5 + [PGS_CONSTRAINT_TYPE_FRICTION] * 2], int)
        phi = _array([[-0.01, 0.01, 0.5, -0.01, -0.01, -0.01, -0.01]])
        incident = _array([[-2.0, -2.0, -2.0, -0.05, 0.5, -3.0, -4.0]])
        target = _array([[0.0, 0.2, 0.0, 0.0, 1.0, 0.0, 0.0]])
        restitution = _array([[0.4] * row_count])
        count = _array([6], int)
        dense_rhs = _full((1, row_count), -7.0)
        sparse_rhs = _full((1, row_count), -7.0)
        wp.launch(
            apply_world_contact_restitution_matrix_free,
            dim=row_count,
            inputs=[
                count,
                row_count,
                _array([row_count], int),
                phi,
                kinds,
                target,
                restitution,
                _array(incident.numpy()[0]),
                _array([np.arange(row_count)], int),
                _array(np.eye(row_count)[None]),
                0.01,
                0.1,
                0,
            ],
            outputs=[dense_rhs, _full((1, row_count), 1.0)],
            device="cpu",
        )
        wp.launch(
            apply_sparse_contact_restitution,
            dim=(1, row_count),
            inputs=[
                count,
                phi,
                kinds,
                target,
                restitution,
                incident,
                0.01,
                0.1,
            ],
            outputs=[sparse_rhs],
            device="cpu",
        )
        np.testing.assert_array_equal(sparse_rhs.numpy(), dense_rhs.numpy())
        np.testing.assert_allclose(sparse_rhs.numpy()[0], [-0.8, -1.08, -7.0, -7.0, -1.2, -7.0, -7.0], atol=1.0e-6)

    def test_decode_factor_velocity(self):
        """Decode accumulated factor impulses into the original physical DOF order."""
        f = _fixture()
        delta = np.arange(6, dtype=np.float32) * 0.17 - 0.4
        out = _full(6, -9.0)
        wp.launch(
            apply_sparse_factor_velocity,
            dim=(1, 6),
            inputs=[
                f["group"],
                _array([0], int),
                f["start"],
                f["indices"].permutation,
                f["indices"].lookup,
                f["packed"],
                _array(delta[None]),
                f["velocity"],
            ],
            outputs=[out],
            device="cpu",
        )
        expected = f["velocity"].numpy().copy()
        expected[f["plan"].permutation] += f["inverse"].T @ delta
        np.testing.assert_allclose(out.numpy(), expected, rtol=2.0e-6, atol=2.0e-7)


if __name__ == "__main__":
    unittest.main()
