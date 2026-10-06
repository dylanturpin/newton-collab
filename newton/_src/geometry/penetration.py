# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Find box penetration directions with constant-storage separating axes."""

from typing import Any

import warp as wp

from .support_function import unpack_mesh_ptr
from .types import GeoType


@wp.func
def _box_polyhedron_pair(a: Any, b: Any) -> bool:
    """Select box pairs with a complete finite separating-axis basis."""
    return (
        a.shape_type == int(GeoType.BOX)
        and (b.shape_type == int(GeoType.BOX) or b.shape_type == int(GeoType.CONVEX_MESH))
    ) or (a.shape_type == int(GeoType.CONVEX_MESH) and b.shape_type == int(GeoType.BOX))


@wp.func
def _face_count(geometry: Any) -> int:
    if geometry.shape_type == int(GeoType.BOX):
        return 3
    mesh = wp.mesh_get(unpack_mesh_ptr(geometry.auxiliary))
    return mesh.indices.shape[0] / 3


@wp.func
def _edge_count(geometry: Any) -> int:
    if geometry.shape_type == int(GeoType.BOX):
        return 3
    mesh = wp.mesh_get(unpack_mesh_ptr(geometry.auxiliary))
    return mesh.indices.shape[0]


@wp.func
def _edge(geometry: Any, index: int) -> wp.vec3:
    if geometry.shape_type == int(GeoType.BOX):
        axis = wp.vec3(0.0)
        axis[index] = 1.0
        return axis
    mesh = wp.mesh_get(unpack_mesh_ptr(geometry.auxiliary))
    triangle = index / 3
    corner = index % 3
    a = mesh.points[mesh.indices[3 * triangle + corner]]
    b = mesh.points[mesh.indices[3 * triangle + (corner + 1) % 3]]
    return wp.cw_mul(b - a, geometry.scale)


@wp.func
def _face(geometry: Any, index: int) -> wp.vec3:
    if geometry.shape_type == int(GeoType.BOX):
        return _edge(geometry, index)
    return wp.cross(_edge(geometry, 3 * index), _edge(geometry, 3 * index + 1))


@wp.func
def _box_segment_depth(half: wp.vec3, first: wp.vec3, second: wp.vec3) -> float:
    """Bound depth using the box minus an actual segment of its partner."""
    center = 0.5 * (first + second)
    segment = 0.5 * (second - first)
    depth = float(1.0e30)
    for i in range(3):
        axis = wp.vec3(0.0)
        axis[i] = 1.0
        depth = wp.min(depth, half[i] + wp.abs(segment[i]) - wp.abs(center[i]))
        cross = wp.cross(axis, segment)
        length = wp.length(cross)
        if length > 0.0:
            bound = wp.dot(wp.abs(cross), half) + wp.abs(wp.dot(cross, segment)) - wp.abs(wp.dot(cross, center))
            depth = wp.min(depth, bound / length)
    return depth


def create_solve_box_penetration(support: Any):
    """Find minimum depth for a box against a box or convex mesh.

    Face normals and edge cross products include every Minkowski facet
    normal. Both signs are tested for asymmetric shapes. Streaming triangle
    edges also tests redundant diagonals, which cannot reduce the depth below
    the true minimum. Storage is constant and the axis count is linear in the
    mesh's triangle count; no expanding polytope is constructed.
    """

    @wp.func
    def test_axis(
        a: Any,
        b: Any,
        rotation: wp.quat,
        position: wp.vec3,
        provider: Any,
        axis: wp.vec3,
        best_normal: wp.vec3,
        best_depth: float,
        best_vertex: wp.vec3,
        recent_vertex: wp.vec3,
    ) -> tuple[wp.vec3, float, wp.vec3, wp.vec3]:
        length = wp.length(axis)
        if length > 0.0:
            direction = axis / length
            for sign in range(2):
                normal = direction if sign == 0 else -direction
                # Genuine support points bound this direction from below.
                # Repeated or inferior axes need no further support query.
                lower = wp.max(wp.dot(best_vertex, normal), wp.dot(recent_vertex, normal))
                if lower < best_depth:
                    vertex = support(a, b, normal, rotation, position, 0.0, provider)
                    recent_vertex = vertex.BtoA
                    depth = wp.dot(vertex.BtoA, normal)
                    if depth < best_depth:
                        best_depth = depth
                        best_normal = normal
                        best_vertex = vertex.BtoA
        return best_normal, best_depth, best_vertex, recent_vertex

    @wp.func
    def solve(a: Any, b: Any, rotation: wp.quat, position: wp.vec3, provider: Any) -> tuple[wp.vec3, float]:
        normal = wp.vec3(1.0, 0.0, 0.0)
        depth = float(1.0e30)
        if a.shape_type == int(GeoType.BOX) and b.shape_type == int(GeoType.BOX):
            if rotation[0] == 0.0 and rotation[1] == 0.0 and rotation[2] == 0.0:
                # Identical frames give an exact box Minkowski sum.
                depths = a.scale + b.scale - wp.abs(position)
                if depths[0] <= depths[1] and depths[0] <= depths[2]:
                    return wp.vec3(1.0 if position[0] >= 0.0 else -1.0, 0.0, 0.0), depths[0]
                elif depths[1] <= depths[2]:
                    return wp.vec3(0.0, 1.0 if position[1] >= 0.0 else -1.0, 0.0), depths[1]
                else:
                    return wp.vec3(0.0, 0.0, 1.0 if position[2] >= 0.0 else -1.0), depths[2]
        box_a = a.shape_type == int(GeoType.BOX)
        half = a.scale if box_a else b.scale
        offset = position if box_a else wp.quat_rotate_inv(rotation, -position)
        distances = half - wp.abs(offset)
        # Visit the nearby box face first. This only orders the search;
        # acceptance still requires the inner-body certificate below.
        if distances[0] <= distances[1] and distances[0] <= distances[2]:
            normal = wp.vec3(1.0, 0.0, 0.0)
        elif distances[1] <= distances[2]:
            normal = wp.vec3(0.0, 1.0, 0.0)
        else:
            normal = wp.vec3(0.0, 0.0, 1.0)
        if not box_a:
            normal = wp.quat_rotate(rotation, normal)
        first = support(a, b, normal, rotation, position, 0.0, provider)
        opposite = support(a, b, -normal, rotation, position, 0.0, provider)
        best_vertex = first.BtoA
        recent_vertex = opposite.BtoA
        depth = wp.dot(best_vertex, normal)
        opposite_depth = wp.dot(opposite.BtoA, -normal)
        if opposite_depth < depth:
            depth = opposite_depth
            normal = -normal
            best_vertex = opposite.BtoA
            recent_vertex = first.BtoA
        if depth < 0.0:
            return normal, depth
        # A box plus a segment is a six-facet-direction zonotope. Since the
        # segment joins genuine partner points, it gives an inner Minkowski
        # body. Matching its depth to a support-plane upper bound certifies
        # the global minimum without visiting the mesh's edges.
        lower = float(0.0)
        if box_a:
            lower = _box_segment_depth(a.scale, first.B, opposite.B)
        else:
            point_a = wp.quat_rotate_inv(rotation, first.B + first.BtoA - position)
            point_b = wp.quat_rotate_inv(rotation, opposite.B + opposite.BtoA - position)
            lower = _box_segment_depth(b.scale, point_a, point_b)
        if lower > 0.0 and depth - lower <= 1.0e-6:
            return normal, depth
        for face in range(_face_count(a)):
            normal, depth, best_vertex, recent_vertex = test_axis(
                a, b, rotation, position, provider, _face(a, face), normal, depth, best_vertex, recent_vertex
            )
        for face in range(_face_count(b)):
            axis = wp.quat_rotate(rotation, _face(b, face))
            normal, depth, best_vertex, recent_vertex = test_axis(
                a, b, rotation, position, provider, axis, normal, depth, best_vertex, recent_vertex
            )
        for edge_a in range(_edge_count(a)):
            axis_a = _edge(a, edge_a)
            for edge_b in range(_edge_count(b)):
                axis_b = wp.quat_rotate(rotation, _edge(b, edge_b))
                normal, depth, best_vertex, recent_vertex = test_axis(
                    a,
                    b,
                    rotation,
                    position,
                    provider,
                    wp.cross(axis_a, axis_b),
                    normal,
                    depth,
                    best_vertex,
                    recent_vertex,
                )
        return normal, depth

    return solve
