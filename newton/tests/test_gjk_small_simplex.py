# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""A small, nondegenerate triangle must retain its interior closest point."""

import unittest

import numpy as np
import warp as wp

from newton import GeoType
from newton._src.geometry.simplex_solver import create_solve_closest_distance
from newton._src.geometry.support_function import (
    GenericShapeData,
    GeoTypeEx,
    SupportMapDataProvider,
    support_map,
)


@wp.kernel
def _query_triangle_point(a: wp.vec3, b: wp.vec3, c: wp.vec3, out: wp.array[float]):
    """Query a triangle against a zero-radius sphere at the origin."""
    triangle = GenericShapeData()
    triangle.shape_type = int(GeoTypeEx.TRIANGLE)
    triangle.scale = b - a
    triangle.auxiliary = c - a
    triangle.center = (b + c - 2.0 * a) / 3.0
    point = GenericShapeData()
    point.shape_type = int(GeoType.SPHERE)
    point.scale = wp.vec3(0.0)
    separated, _, _, normal, distance = wp.static(create_solve_closest_distance(support_map).core)(
        triangle, point, wp.quat_identity(), -a, 0.0, SupportMapDataProvider()
    )
    out[0] = float(separated)
    out[1] = distance
    for axis in range(3):
        out[2 + axis] = normal[axis]


class TestGJKSmallSimplex(unittest.TestCase):
    """Preserve small, nondegenerate simplex faces during distance queries."""

    def test_triangle_interior_distance_is_scale_relative(self):
        """Return the face distance instead of a farther edge at small scales."""
        # A right triangle 6 mm by 7 mm, 40 micrometers above the point.
        # Its squared area is only 1.764e-9 m^4, but its squared sine is 1.
        vertices = np.array(
            [
                [-0.002, -0.003, 0.00004],
                [0.004, -0.003, 0.00004],
                [-0.002, 0.004, 0.00004],
            ],
            dtype=np.float64,
        )
        self._check_vertices(vertices)

    def test_stalled_edge_does_not_become_zero_normal(self):
        """Keep a positive sub-tolerance face gap and its unit normal."""
        # A second small, nondegenerate simplex whose edge distance falls below
        # the convergence tolerance. Discarding its face incorrectly classifies
        # the positive face gap as overlap and returns a zero normal.
        self._check_vertices(
            np.array(
                [
                    [-0.00225319340825, -0.00759682059288, 0.00803601182997],
                    [-0.00664979964495, -0.0111039578915, 0.0170838627964],
                    [0.00106162205338, 0.00387127697468, -0.00424785725772],
                ],
                dtype=np.float64,
            )
        )

    def _check_vertices(self, vertices):
        """Compare each scale with the analytic interior triangle projection."""
        for scale in (0.5, 1.0, 2.0):
            a, b, c = vertices * scale
            normal = np.cross(b - a, c - a)
            normal /= np.linalg.norm(normal)
            signed_distance = float(normal @ a)
            closest = signed_distance * normal
            weights = np.linalg.lstsq(np.stack([b - a, c - a], axis=1), closest - a, rcond=None)[0]
            self.assertGreater(float(weights.min()), 0.0)
            self.assertLess(float(weights.sum()), 1.0)
            expected_normal = -closest / np.linalg.norm(closest)
            for device in wp.get_devices():
                with self.subTest(scale=scale, device=str(device)):
                    output = wp.zeros(5, dtype=float, device=device)
                    wp.launch(
                        _query_triangle_point,
                        dim=1,
                        inputs=[wp.vec3(*x) for x in (a, b, c)],
                        outputs=[output],
                        device=device,
                    )
                    actual = output.numpy()
                    self.assertEqual(actual[0], 1.0)
                    self.assertAlmostEqual(float(actual[1]), abs(signed_distance), delta=1e-7 * scale)
                    np.testing.assert_allclose(actual[2:5], expected_normal, atol=2e-5)


if __name__ == "__main__":
    unittest.main()
