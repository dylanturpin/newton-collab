# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare sparse factor-coordinate sweeps with physical-coordinate PGS."""

import unittest

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs.friction import friction_pair_candidate
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
)
from newton._src.solvers.feather_pgs.sparse_pgs import _get_pgs_solve_sparse_kernel


def _sweep(jacobian, response, velocity, bias, diagonal, types, parents, mu, impulses, friction_start=0):
    velocity, impulses = velocity.copy(), impulses.copy()
    for iteration in range(8):
        for row, kind in enumerate(types):
            if kind == PGS_CONSTRAINT_TYPE_FRICTION and iteration < friction_start:
                impulses[row] = 0.0
                continue
            if kind == PGS_CONSTRAINT_TYPE_FRICTION and row != parents[row] + 1:
                continue
            residual = jacobian[row] @ velocity + bias[row]
            old = impulses[row]
            if kind == PGS_CONSTRAINT_TYPE_FRICTION:
                parent, sibling = parents[row], row + 1
                load = impulses[parent]
                next_row = parents[parent]
                while next_row >= 0 and next_row != parent:
                    load += impulses[next_row]
                    next_row = parents[next_row]
                radius = max(mu[row] * load, 0.0)
                pair = friction_pair_candidate(
                    float(diagonal[row]),
                    float(jacobian[row] @ response[sibling]),
                    float(diagonal[sibling]),
                    wp.vec2(float(residual), float(jacobian[sibling] @ velocity + bias[sibling])),
                    wp.vec2(float(old), float(impulses[sibling])),
                    float(radius),
                    1.0,
                )
                pair = np.asarray(pair, dtype=np.float64)
                magnitude = np.linalg.norm(pair)
                if magnitude > radius:
                    pair *= radius / magnitude
                velocity += response[sibling] * (pair[1] - impulses[sibling])
                impulses[sibling] = pair[1]
                value = pair[0]
            else:
                value = old - residual / diagonal[row]
                if kind in (PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_JOINT_LIMIT):
                    value = max(value, 0.0)
            velocity += response[row] * (value - old)
            impulses[row] = value
    return velocity, impulses


def _problem(dofs=43, support=18, seeded=False):
    rng = np.random.default_rng(471)
    factor = np.tril(rng.normal(scale=0.04, size=(dofs, dofs))) + np.eye(dofs)
    indices = np.sort(rng.choice(dofs, support, replace=False))
    rows = np.zeros((8, dofs))
    rows[:, indices] = rng.normal(scale=0.3, size=(8, support))
    rows[0] = 0.0
    rows[0, indices[0]] = 0.8
    initial = rng.normal(scale=0.2, size=dofs)
    bias = np.array((-0.2, -0.1, 0.04, -0.03, -0.12, -0.15, 0.02, 0.04))
    types = np.array(
        (
            PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
            PGS_CONSTRAINT_TYPE_CONTACT,
            PGS_CONSTRAINT_TYPE_FRICTION,
            PGS_CONSTRAINT_TYPE_FRICTION,
            PGS_CONSTRAINT_TYPE_CONTACT,
            PGS_CONSTRAINT_TYPE_CONTACT,
            PGS_CONSTRAINT_TYPE_FRICTION,
            PGS_CONSTRAINT_TYPE_FRICTION,
        ),
        dtype=np.int32,
    )
    parents = np.array((-1, 4, 1, 1, 1, -1, 5, 5), dtype=np.int32)
    mu = np.array((0, 0, 0.6, 0.6, 0, 0, 0.4, 0.4), dtype=np.float64)
    impulses = np.array((0, 0.4, 0.02, -0.01, 0.3, 0.2, -0.03, 0.01)) if seeded else np.zeros(8)
    jacobian = rows @ factor.T
    response = np.linalg.solve(factor.T, rows.T).T
    incident = jacobian @ initial
    diagonal = np.sum(rows * rows, axis=1) + 1.0e-6
    return factor, indices, rows, initial, bias, types, parents, mu, impulses, jacobian, response, incident, diagonal


class TestFeatherPGSSparsePGS(unittest.TestCase):
    def test_factor_coordinate_residual_and_cross_term(self):
        """Preserve physical residuals and paired-tangent effective mass."""
        for dofs, support in ((7, 6), (43, 18), (75, 39)):
            with self.subTest(dofs=dofs, support=support):
                factor, _, rows, initial, _, _, _, _, _, jacobian, response, incident, _ = _problem(dofs, support)
                delta = np.linspace(-0.3, 0.2, dofs)
                physical = initial + np.linalg.solve(factor.T, delta)
                np.testing.assert_allclose(jacobian @ physical, incident + rows @ delta, atol=1.0e-12)
                np.testing.assert_allclose(jacobian @ response.T, rows @ rows.T, atol=1.0e-12)

    def test_factor_sweeps_match_physical_sweeps(self):
        """Preserve joint limits, pooled friction loads and seeded-impulse semantics."""
        for seeded in (False, True):
            for friction_start in (0, 2):
                with self.subTest(seeded=seeded, friction_start=friction_start):
                    factor, _, rows, initial, bias, types, parents, mu, impulses, jacobian, response, incident, diag = (
                        _problem(seeded=seeded)
                    )
                    expected, expected_impulses = _sweep(
                        jacobian, response, initial, bias, diag, types, parents, mu, impulses, friction_start
                    )
                    delta, actual_impulses = _sweep(
                        rows,
                        rows,
                        np.zeros_like(initial),
                        incident + bias,
                        diag,
                        types,
                        parents,
                        mu,
                        impulses,
                        friction_start,
                    )
                    actual = initial + np.linalg.solve(factor.T, delta)
                    np.testing.assert_allclose(actual, expected, atol=2.0e-6, rtol=2.0e-5)
                    np.testing.assert_allclose(actual_impulses, expected_impulses, atol=2.0e-6, rtol=2.0e-5)

    @unittest.skipUnless(wp.is_cuda_available(), "sparse PGS requires CUDA")
    def test_cuda_sparse_sweeps_match_reference(self):
        """Match serial reference sweeps, including padding and zero-row worlds."""
        device = wp.get_device("cuda:0")
        for dofs, support in ((43, 18), (75, 39)):
            for seeded in (False, True):
                with self.subTest(dofs=dofs, support=support, seeded=seeded):
                    _, indices, rows, initial, bias, types, parents, mu, impulses, _, _, incident, diag = _problem(
                        dofs, support, seeded
                    )
                    expected, expected_impulses = _sweep(
                        rows, rows, np.zeros_like(initial), incident + bias, diag, types, parents, mu, impulses
                    )
                    worlds, capacity = 3, 10
                    row_dof = np.full((worlds, capacity, support), -1, dtype=np.int32)
                    row_dof[:, :8] = indices
                    row_dof[:, 0, 1:] = -1
                    row_factor = np.zeros((worlds, capacity, support), dtype=np.float32)
                    row_factor[:, :8] = rows[:, indices]

                    def padded(array, dtype=np.float32, shape=(worlds, capacity)):
                        result = np.zeros(shape, dtype=dtype)
                        result[:, :8] = array
                        return wp.array(result, device=device)

                    actual_impulses = padded(impulses)
                    delta = wp.full((worlds, dofs), float("nan"), device=device)
                    kernel = _get_pgs_solve_sparse_kernel(capacity, dofs, support)
                    wp.launch_tiled(
                        kernel,
                        dim=[2],
                        inputs=[
                            worlds,
                            wp.array([8, 0, 8], dtype=int, device=device),
                            padded(bias),
                            padded(diag),
                            actual_impulses,
                            wp.array(row_dof, device=device),
                            wp.array(row_factor, device=device),
                            padded(incident),
                            padded(types, np.int32),
                            padded(parents, np.int32),
                            padded(mu),
                            8,
                            1.0,
                            0,
                            0,
                        ],
                        outputs=[delta],
                        block_dim=64,
                        device=device,
                    )
                    actual = delta.numpy()
                    result_impulses = actual_impulses.numpy()
                    np.testing.assert_array_equal(actual[1], 0.0)
                    for world in (0, 2):
                        np.testing.assert_allclose(actual[world], expected, atol=2.0e-5, rtol=2.0e-4)
                        np.testing.assert_allclose(
                            result_impulses[world, :8], expected_impulses, atol=2.0e-5, rtol=2.0e-4
                        )


if __name__ == "__main__":
    unittest.main()
