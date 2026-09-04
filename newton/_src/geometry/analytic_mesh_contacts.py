# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Closed-form contacts between a mesh triangle and an exact analytic primitive.

Replaces GJK/MPR in the mesh-convex narrow phase for box, sphere, capsule, cylinder and
cone. Per triangle: the primitive's support point against the face, the three vertices,
and the edges whose vertex gradients bracket an interior distance minimum. Ellipsoids
are not routed because their signed distance is a first-order approximation.
"""

from typing import Any

import warp as wp

from .contact_data import ContactData
from .kernels import eval_analytic_sdf, eval_analytic_sdf_grad
from .support_function import GenericShapeData, SupportMapDataProvider, support_map
from .types import GeoType

# golden-section iterations per edge minimum; fixed so the loop is graph capturable
EDGE_MIN_ITERS = 24

# barycentric slack for the face test; inclusive so a support point on a shared edge is kept
FACE_INTERIOR_EPS = -1.0e-6

# Sample around the exact support near an axis to retain flat face manifolds.
FACE_ON_TOL = 1.0e-3

# per-triangle contact slots in the sort key: 3 vertices, 3 edges, up to 8 face points
_SLOT_VERTEX = 0
_SLOT_EDGE = 3
_SLOT_FACE = 6
_SLOT_BITS = 4


@wp.func
def box_corner(k: int, half: wp.vec3) -> wp.vec3:
    """Corner ``k`` of 8 of a box with the given half extents."""
    sx = wp.where((k & 1) != 0, 1.0, -1.0)
    sy = wp.where((k & 2) != 0, 1.0, -1.0)
    sz = wp.where((k & 4) != 0, 1.0, -1.0)
    return wp.vec3(sx * half[0], sy * half[1], sz * half[2])


@wp.func
def face_support_count(geo: int, n: wp.vec3) -> int:
    """Number of support candidates along ``-n``: box corners, rim or segment samples, else one."""
    if geo == GeoType.BOX:
        return 8
    if geo == GeoType.CYLINDER:
        if wp.abs(n[2]) > 1.0 - FACE_ON_TOL:
            return 4
    if geo == GeoType.CONE:
        if n[2] > 1.0 - FACE_ON_TOL:
            return 4
    if geo == GeoType.CAPSULE:
        if wp.abs(n[2]) < FACE_ON_TOL:
            return 2
    return 1


@wp.func
def face_support_point(
    geo: int,
    scale: wp.vec3,
    n: wp.vec3,
    k: int,
    geom: GenericShapeData,
    provider: SupportMapDataProvider,
) -> wp.vec3:
    """Candidate ``k`` of the primitive's support along ``-n``, in the primitive's frame."""
    if geo == GeoType.BOX:
        return box_corner(k, scale)
    # must mirror face_support_count exactly
    rim = False
    if geo == GeoType.CYLINDER:
        if wp.abs(n[2]) > 1.0 - FACE_ON_TOL:
            rim = True
    elif geo == GeoType.CONE:
        if n[2] > 1.0 - FACE_ON_TOL:
            rim = True
    if rim:
        # Cardinal rim samples alone miss the true support at small tilts.
        # Rotations preserve the surface of revolution, including barrel sides.
        s = support_map(geom, -n, provider)
        if k == 1:
            return wp.vec3(-s[1], s[0], s[2])
        elif k == 2:
            return wp.vec3(-s[0], -s[1], s[2])
        elif k == 3:
            return wp.vec3(s[1], -s[0], s[2])
        return s
    if geo == GeoType.CAPSULE:
        if wp.abs(n[2]) < FACE_ON_TOL:
            z_end = wp.where(k == 0, -scale[1], scale[1])
            return wp.vec3(0.0, 0.0, z_end) - n * scale[0]
    return support_map(geom, -n, provider)


@wp.func
def edge_min_coordinate(geo: int, scale: wp.vec3, a: wp.vec3, b: wp.vec3) -> float:
    """``t`` in [0, 1] minimising the signed distance along ``a``-``b``; closed form for a sphere."""
    d = b - a
    dd = wp.length_sq(d)
    if dd <= 0.0:
        return 0.0
    if geo == GeoType.SPHERE:
        return wp.clamp(-wp.dot(a, d) / dd, 0.0, 1.0)

    inv_phi = 0.6180339887498949
    inv_phi2 = 0.3819660112501051
    lo = float(0.0)
    hi = float(1.0)
    h = hi - lo
    c = lo + inv_phi2 * h
    e = lo + inv_phi * h
    fc = eval_analytic_sdf(geo, scale, a + d * c)
    fe = eval_analytic_sdf(geo, scale, a + d * e)
    for _i in range(EDGE_MIN_ITERS):
        if fc < fe:
            hi = e
            e = c
            fe = fc
            h = inv_phi * h
            c = lo + inv_phi2 * h
            fc = eval_analytic_sdf(geo, scale, a + d * c)
        else:
            lo = c
            c = e
            fc = fe
            h = inv_phi * h
            e = lo + inv_phi * h
            fe = eval_analytic_sdf(geo, scale, a + d * e)
    if fc < fe:
        return 0.5 * (lo + e)
    return 0.5 * (c + hi)


@wp.func
def minkowski_radius(geo: int, scale: wp.vec3) -> float:
    """Radius carried in ``radius_eff`` rather than in the distance (sphere and capsule)."""
    if geo == GeoType.SPHERE or geo == GeoType.CAPSULE:
        return scale[0]
    return 0.0


def create_triangle_analytic_contacts(writer_func: Any):
    """Build the per-triangle exact contact function for a given contact writer."""

    @wp.func
    def emit(
        witness_mesh: wp.vec3,
        distance: float,
        normal_local: wp.vec3,
        X_prim_ws: wp.transform,
        radius_eff_prim: float,
        gap_sum: float,
        margin_mesh: float,
        margin_prim: float,
        shape_mesh: int,
        shape_prim: int,
        key: int,
        writer_data: Any,
    ):
        """Write one contact; ``normal_local`` points from the mesh toward the primitive."""
        normal_world = wp.transform_vector(X_prim_ws, normal_local)
        length = wp.length(normal_world)
        if length <= 0.0:
            return
        normal_world = normal_world / length
        center_local = witness_mesh + normal_local * (0.5 * distance)

        contact = ContactData()
        contact.contact_point_center = wp.transform_point(X_prim_ws, center_local)
        contact.contact_normal_a_to_b = normal_world
        contact.contact_distance = distance
        contact.radius_eff_a = 0.0
        contact.radius_eff_b = radius_eff_prim
        contact.margin_a = margin_mesh
        contact.margin_b = margin_prim
        contact.shape_a = shape_mesh
        contact.shape_b = shape_prim
        contact.gap_sum = gap_sum
        contact.sort_sub_key = key
        writer_func(contact, writer_data, -1)

    @wp.func
    def triangle_analytic_contacts(
        v0: wp.vec3,
        v1: wp.vec3,
        v2: wp.vec3,
        geo: int,
        prim_scale: wp.vec3,
        X_prim_ws: wp.transform,
        gap_sum: float,
        margin_mesh: float,
        margin_prim: float,
        shape_mesh: int,
        shape_prim: int,
        tri_idx: int,
        writer_data: Any,
    ):
        """Emit every contact between one triangle, given in the primitive's frame, and the primitive."""
        # sphere and capsule carry their radius in radius_eff, matching the GJK/MPR path
        accept = gap_sum + margin_mesh + margin_prim
        radius_eff = minkowski_radius(geo, prim_scale)
        key_base = tri_idx << _SLOT_BITS

        # vertices; the gradients are reused by the edge tests
        d0, g0 = eval_analytic_sdf_grad(geo, prim_scale, v0)
        d1, g1 = eval_analytic_sdf_grad(geo, prim_scale, v1)
        d2, g2 = eval_analytic_sdf_grad(geo, prim_scale, v2)
        if d0 < accept:
            emit(
                v0,
                d0,
                -g0,
                X_prim_ws,
                radius_eff,
                gap_sum,
                margin_mesh,
                margin_prim,
                shape_mesh,
                shape_prim,
                ((key_base | (_SLOT_VERTEX + 0)) << 1) | 1,
                writer_data,
            )
        if d1 < accept:
            emit(
                v1,
                d1,
                -g1,
                X_prim_ws,
                radius_eff,
                gap_sum,
                margin_mesh,
                margin_prim,
                shape_mesh,
                shape_prim,
                ((key_base | (_SLOT_VERTEX + 1)) << 1) | 1,
                writer_data,
            )
        if d2 < accept:
            emit(
                v2,
                d2,
                -g2,
                X_prim_ws,
                radius_eff,
                gap_sum,
                margin_mesh,
                margin_prim,
                shape_mesh,
                shape_prim,
                ((key_base | (_SLOT_VERTEX + 2)) << 1) | 1,
                writer_data,
            )

        # edges: only where the vertex gradients bracket an interior minimum
        for k in range(3):
            a = v0
            b = v1
            ga = g0
            gb = g1
            if k == 1:
                a = v1
                b = v2
                ga = g1
                gb = g2
            elif k == 2:
                a = v2
                b = v0
                ga = g2
                gb = g0
            d = b - a
            if wp.dot(ga, d) < 0.0 and wp.dot(gb, d) > 0.0:
                t = edge_min_coordinate(geo, prim_scale, a, b)
                x = a + d * t
                dist, grad = eval_analytic_sdf_grad(geo, prim_scale, x)
                if dist < accept:
                    emit(
                        x,
                        dist,
                        -grad,
                        X_prim_ws,
                        radius_eff,
                        gap_sum,
                        margin_mesh,
                        margin_prim,
                        shape_mesh,
                        shape_prim,
                        ((key_base | (_SLOT_EDGE + k)) << 1) | 1,
                        writer_data,
                    )

        # face against the primitive's support point along -n
        n = wp.cross(v1 - v0, v2 - v0)
        n_len_sq = wp.length_sq(n)
        if n_len_sq <= 0.0:
            return
        n = n / wp.sqrt(n_len_sq)
        # back-face cull: the primitive's origin is this frame's origin
        if wp.dot(n, v0) > 0.0:
            return

        geom = GenericShapeData()
        geom.shape_type = geo
        geom.scale = prim_scale
        geom.auxiliary = wp.vec3(0.0, 0.0, 0.0)
        geom.center = wp.vec3(0.0, 0.0, 0.0)
        provider = SupportMapDataProvider()

        count = face_support_count(geo, n)
        # only the candidates tied for the support touch; the rest lie behind
        min_dot = float(1.0e30)
        for k in range(count):
            s = face_support_point(geo, prim_scale, n, k, geom, provider)
            min_dot = wp.min(min_dot, wp.dot(s, n))
        tie_eps = 1.0e-5 * (wp.abs(prim_scale[0]) + wp.abs(prim_scale[1]) + wp.abs(prim_scale[2]))

        e1 = v1 - v0
        e2 = v2 - v0
        d11 = wp.dot(e1, e1)
        d12 = wp.dot(e1, e2)
        d22 = wp.dot(e2, e2)
        denom = d11 * d22 - d12 * d12
        if denom <= 0.0:
            return

        for k in range(count):
            s = face_support_point(geo, prim_scale, n, k, geom, provider)
            if wp.dot(s, n) > min_dot + tie_eps:
                continue
            sep = wp.dot(s - v0, n)
            if sep >= accept:
                continue
            # keep interior projections only; the boundary belongs to the edge tests
            proj = s - n * sep
            w = proj - v0
            dp1 = wp.dot(w, e1)
            dp2 = wp.dot(w, e2)
            bv = (d22 * dp1 - d12 * dp2) / denom
            bw = (d11 * dp2 - d12 * dp1) / denom
            bu = 1.0 - bv - bw
            if bu < FACE_INTERIOR_EPS or bv < FACE_INTERIOR_EPS or bw < FACE_INTERIOR_EPS:
                continue

            emit(
                proj,
                sep,
                n,
                X_prim_ws,
                radius_eff,
                gap_sum,
                margin_mesh,
                margin_prim,
                shape_mesh,
                shape_prim,
                ((key_base | (_SLOT_FACE + k)) << 1) | 1,
                writer_data,
            )

    return triangle_analytic_contacts
