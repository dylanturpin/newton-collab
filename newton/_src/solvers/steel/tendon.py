# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""MuJoCo-style tendons for :class:`~newton.solvers.SolverSteel`.

A tendon is a scalar length ``L(q)`` read from the ``mujoco:tendon_*`` custom attributes the
MJCF and USD importers write (:meth:`~newton.solvers.SolverMuJoCo.register_custom_attributes`):

- a *fixed* tendon is a linear combination of 1-DoF joint coordinates, ``L = sum_i c_i q_i``;
- a *spatial* tendon is a path through sites, optionally wrapping spheres and cylinders and split
  into branches by pulley elements, ``L = sum_s |p_1 - p_0| / divisor_s``. Its generalized Jacobian
  is the sum of point-Jacobian projections onto each straight span (the wrap arcs contribute length
  only: with the tangent points held on the wrap body the arc derivative vanishes).

``range`` limits and ``frictionloss`` become dense rows with the Jacobian ``J = dL/dq`` up to sign: two
one-sided rows ``L - lo >= 0`` and ``hi - L >= 0`` solved like joint position limits, and one row boxed to
``[-f dt, f dt]`` like joint dry friction. Armature, the ``stiffness`` / ``damping`` spring-damper
(``springlength`` dead band) and actuators with a tendon transmission are folded into the mass matrix along
``J`` and into the joint forces, the way Steel treats joint drives (:func:`apply_tendon_mass_terms_for_size`).

Every touched DOF must belong to one articulation; sites on the world, or on kinematic bodies outside
that articulation, are prescribed and enter the row target through their velocity. MuJoCo's
solref/solimp softening is not modeled.
"""

from __future__ import annotations

import warnings

import numpy as np
import warp as wp

from ...core.reset import reset_world_selected
from ...exceptions import NewtonWarning
from ...geometry import GeoType
from ...sim import BodyFlags
from .kernels import (
    PGS_CONSTRAINT_TYPE_JOINT_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    WARP_DETERMINISM_OFF,
    _reserve_bilateral_rows,
)

# Steel orders its own reductions; Warp's deterministic lowering stays off (see test_determinism).
wp.set_module_options({"enable_backward": False, **WARP_DETERMINISM_OFF})

TENDON_ROW_LIMIT_LOWER = wp.constant(1)
TENDON_ROW_LIMIT_UPPER = wp.constant(2)
TENDON_ROW_FRICTION = wp.constant(4)

TENDON_TYPE_FIXED = wp.constant(0)
TENDON_WRAP_SITE = wp.constant(0)
TENDON_WRAP_GEOM = wp.constant(1)
TENDON_WRAP_PULLEY = wp.constant(2)
_TRN_TENDON = 2  # SolverMuJoCo.TrnType.TENDON

_WRAP_MINVAL = wp.constant(1.0e-15)
_WRAP_MAXVAL = wp.constant(1.0e10)
_GEO_SPHERE = wp.constant(int(GeoType.SPHERE))
_GEO_CYLINDER = wp.constant(int(GeoType.CYLINDER))


# -----------------------------------------------------------------------------
# Wrap geometry: a port of MuJoCo's ``mju_wrap`` (via mujoco_warp ``util_misc``).
# -----------------------------------------------------------------------------


@wp.func
def _segments_intersect(p1: wp.vec2, p2: wp.vec2, p3: wp.vec2, p4: wp.vec2) -> bool:
    det = (p4[1] - p3[1]) * (p2[0] - p1[0]) - (p4[0] - p3[0]) * (p2[1] - p1[1])
    if wp.abs(det) < _WRAP_MINVAL:
        return False
    a = ((p4[0] - p3[0]) * (p1[1] - p3[1]) - (p4[1] - p3[1]) * (p1[0] - p3[0])) / det
    b = ((p2[0] - p1[0]) * (p1[1] - p3[1]) - (p2[1] - p1[1]) * (p1[0] - p3[0])) / det
    return a >= 0.0 and a <= 1.0 and b >= 0.0 and b <= 1.0


@wp.func
def _safe_div(x: float, y: float) -> float:
    if y == 0.0:
        return x / _WRAP_MINVAL
    return x / y


@wp.func
def _normalized(v: wp.vec2) -> wp.vec2:
    n = wp.length(v)
    if n < _WRAP_MINVAL:
        return v
    return v / n


@wp.func
def _circle_arc_length(p0: wp.vec2, p1: wp.vec2, ind: int, radius: float) -> float:
    angle = wp.acos(wp.clamp(wp.dot(_normalized(p0), _normalized(p1)), -1.0, 1.0))
    cross = p0[1] * p1[0] - p0[0] * p1[1]
    if (cross > 0.0 and ind != 0) or (cross < 0.0 and ind == 0):
        angle = 2.0 * wp.pi - angle
    return radius * angle


@wp.func
def _wrap_circle(end: wp.vec4, side: wp.vec2, radius: float) -> tuple[float, wp.vec2, wp.vec2]:
    """2D wrap of the segment ``end`` around a circle at the origin; length ``-1`` when the span is clear."""
    none = wp.vec2(_WRAP_MAXVAL, _WRAP_MAXVAL)
    valid_side = wp.length(side) < _WRAP_MAXVAL
    end0 = wp.vec2(end[0], end[1])
    end1 = wp.vec2(end[2], end[3])
    sqlen0 = wp.dot(end0, end0)
    sqlen1 = wp.dot(end1, end1)
    sqrad = radius * radius
    if sqlen0 < sqrad or sqlen1 < sqrad or radius < _WRAP_MINVAL:
        return -1.0, none, none
    dif = end1 - end0
    dd = wp.dot(dif, dif)
    if dd < _WRAP_MINVAL:
        return -1.0, none, none
    a = wp.clamp(-wp.dot(dif, end0) / dd, 0.0, 1.0)
    tmp = a * dif + end0
    if wp.dot(tmp, tmp) > sqrad and (not valid_side or wp.dot(side, tmp) >= 0.0):
        return -1.0, none, none

    sqrt0 = wp.sqrt(sqlen0 - sqrad)
    sqrt1 = wp.sqrt(sqlen1 - sqrad)
    sol00 = wp.vec2(
        _safe_div(end[0] * sqrad + radius * end[1] * sqrt0, sqlen0),
        _safe_div(end[1] * sqrad - radius * end[0] * sqrt0, sqlen0),
    )
    sol01 = wp.vec2(
        _safe_div(end[2] * sqrad - radius * end[3] * sqrt1, sqlen1),
        _safe_div(end[3] * sqrad + radius * end[2] * sqrt1, sqlen1),
    )
    sol10 = wp.vec2(
        _safe_div(end[0] * sqrad - radius * end[1] * sqrt0, sqlen0),
        _safe_div(end[1] * sqrad + radius * end[0] * sqrt0, sqlen0),
    )
    sol11 = wp.vec2(
        _safe_div(end[2] * sqrad + radius * end[3] * sqrt1, sqlen1),
        _safe_div(end[3] * sqrad - radius * end[2] * sqrt1, sqlen1),
    )
    good0 = float(0.0)
    good1 = float(0.0)
    if valid_side:
        good0 = wp.dot(_normalized(sol00 + sol01), side)
        good1 = wp.dot(_normalized(sol10 + sol11), side)
    else:
        good0 = -wp.dot(sol00 - sol01, sol00 - sol01)
        good1 = -wp.dot(sol10 - sol11, sol10 - sol11)
    if _segments_intersect(end0, sol00, end1, sol01):
        good0 = -10000.0
    if _segments_intersect(end0, sol10, end1, sol11):
        good1 = -10000.0
    pnt0 = sol10
    pnt1 = sol11
    ind = int(1)
    if good0 > good1:
        pnt0 = sol00
        pnt1 = sol01
        ind = 0
    if _segments_intersect(end0, pnt0, end1, pnt1):
        return -1.0, none, none
    return _circle_arc_length(pnt0, pnt1, ind, radius), pnt0, pnt1


@wp.func
def _wrap_inside(end: wp.vec4, radius: float) -> tuple[float, wp.vec2, wp.vec2]:
    """2D wrap with the side point inside the circle: the path touches the circle at one point."""
    none = wp.vec2(_WRAP_MAXVAL, _WRAP_MAXVAL)
    end0 = wp.vec2(end[0], end[1])
    end1 = wp.vec2(end[2], end[3])
    len0 = wp.length(end0)
    len1 = wp.length(end1)
    dif = end1 - end0
    dd = wp.dot(dif, dif)
    if len0 <= radius or len1 <= radius or radius < _WRAP_MINVAL or len0 < _WRAP_MINVAL or len1 < _WRAP_MINVAL:
        return -1.0, none, none
    if dd > _WRAP_MINVAL:
        a = -wp.dot(dif, end0) / dd
        if a > 0.0 and a < 1.0:
            if wp.length(end0 + a * dif) <= radius:
                return -1.0, none, none

    pnt = _normalized(0.5 * (end0 + end1)) * radius
    A = _safe_div(radius, len0)
    B = _safe_div(radius, len1)
    cos_g = _safe_div(len0 * len0 + len1 * len1 - dd, 2.0 * len0 * len1)
    if cos_g < -1.0 + _WRAP_MINVAL:
        return -1.0, pnt, pnt
    if cos_g > 1.0 - _WRAP_MINVAL:
        return 0.0, pnt, pnt
    G = wp.acos(cos_g)
    z = float(1.0 - 1.0e-7)
    f = wp.asin(A * z) + wp.asin(B * z) - 2.0 * wp.asin(z) + G
    if f > 0.0:
        return 0.0, pnt, pnt
    it = int(0)
    converged = bool(True)
    while it < 20 and wp.abs(f) > 1.0e-6:
        sq_z = z * z
        df = (
            A / wp.max(_WRAP_MINVAL, wp.sqrt(1.0 - sq_z * A * A))
            + B / wp.max(_WRAP_MINVAL, wp.sqrt(1.0 - sq_z * B * B))
            - 2.0 / wp.max(_WRAP_MINVAL, wp.sqrt(1.0 - sq_z))
        )
        if df > -_WRAP_MINVAL:
            converged = False
            break
        z1 = z - _safe_div(f, df)
        if z1 > z:
            converged = False
            break
        z = z1
        f = wp.asin(A * z) + wp.asin(B * z) - 2.0 * wp.asin(z) + G
        if f > 1.0e-6:
            converged = False
            break
        it += 1
    if not converged or it >= 20:
        return 0.0, pnt, pnt
    vec = end1
    ang = wp.asin(z) - wp.asin(B * z)
    if end[0] * end[3] - end[1] * end[2] > 0.0:
        vec = end0
        ang = wp.asin(z) - wp.asin(A * z)
    vec = _normalized(vec)
    pnt = wp.vec2(
        radius * (wp.cos(ang) * vec[0] - wp.sin(ang) * vec[1]),
        radius * (wp.sin(ang) * vec[0] + wp.cos(ang) * vec[1]),
    )
    return 0.0, pnt, pnt


@wp.func
def tendon_wrap(
    x0: wp.vec3, x1: wp.vec3, X_geom: wp.transform, radius: float, is_cylinder: bool, side: wp.vec3
) -> tuple[float, wp.vec3, wp.vec3]:
    """Wrap the span ``x0 -> x1`` around a sphere or a Z-axis cylinder posed at ``X_geom`` [m].

    Returns the arc length (``-1`` when the straight span clears the geom) and the two world-frame
    tangent points. ``side`` is the world side-site position, or a component of ``1e10`` for none.
    """
    none = wp.vec3(_WRAP_MAXVAL, _WRAP_MAXVAL, _WRAP_MAXVAL)
    p0 = wp.transform_point(wp.transform_inverse(X_geom), x0)
    p1 = wp.transform_point(wp.transform_inverse(X_geom), x1)
    if wp.length(p0) < _WRAP_MINVAL or wp.length(p1) < _WRAP_MINVAL:
        return -1.0, none, none

    axis0 = wp.vec3(1.0, 0.0, 0.0)
    axis1 = wp.vec3(0.0, 1.0, 0.0)
    if not is_cylinder:
        axis0 = wp.normalize(p0)
        normal = wp.cross(p0, p1)
        nrm = wp.length(normal)
        if nrm < _WRAP_MINVAL:
            # (p0, p1) parallel: pick a normal off the largest component of axis0
            a = wp.vec3(wp.abs(axis0[0]), wp.abs(axis0[1]), wp.abs(axis0[2]))
            i = int(0)
            if a[1] > a[0] and a[1] > a[2]:
                i = 1
            if a[2] > a[0] and a[2] > a[1]:
                i = 2
            helper = wp.vec3(1.0, 1.0, 1.0)
            helper[i] = 0.0
            normal = wp.normalize(wp.cross(axis0, helper))
        else:
            normal = normal / nrm
        axis1 = wp.normalize(wp.cross(normal, axis0))

    end = wp.vec4(wp.dot(p0, axis0), wp.dot(p0, axis1), wp.dot(p1, axis0), wp.dot(p1, axis1))
    valid_side = wp.length(side) < _WRAP_MAXVAL
    sidepnt = wp.vec3(0.0, 0.0, 0.0)
    side2 = wp.vec2(_WRAP_MAXVAL, _WRAP_MAXVAL)
    if valid_side:
        sidepnt = wp.transform_point(wp.transform_inverse(X_geom), side)
        side2 = _normalized(wp.vec2(wp.dot(sidepnt, axis0), wp.dot(sidepnt, axis1))) * radius

    wlen = float(0.0)
    pnt0 = wp.vec2(0.0, 0.0)
    pnt1 = wp.vec2(0.0, 0.0)
    if valid_side and wp.length(sidepnt) < radius:
        wlen, pnt0, pnt1 = _wrap_inside(end, radius)
    else:
        wlen, pnt0, pnt1 = _wrap_circle(end, side2, radius)
    if wlen < 0.0:
        return -1.0, none, none

    res0 = axis0 * pnt0[0] + axis1 * pnt0[1]
    res1 = axis0 * pnt1[0] + axis1 * pnt1[1]
    if is_cylinder:
        # helix: spread the height change over the path proportionally to planar length
        l0 = wp.sqrt((p0[0] - res0[0]) * (p0[0] - res0[0]) + (p0[1] - res0[1]) * (p0[1] - res0[1]))
        l1 = wp.sqrt((p1[0] - res1[0]) * (p1[0] - res1[0]) + (p1[1] - res1[1]) * (p1[1] - res1[1]))
        res0[2] = p0[2] + (p1[2] - p0[2]) * _safe_div(l0, l0 + wlen + l1)
        res1[2] = p0[2] + (p1[2] - p0[2]) * _safe_div(l0 + wlen, l0 + wlen + l1)
        height = wp.abs(res1[2] - res0[2])
        wlen = wp.sqrt(wlen * wlen + height * height)
    return wlen, wp.transform_point(X_geom, res0), wp.transform_point(X_geom, res1)


# -----------------------------------------------------------------------------
# Length, Jacobian, and rows
# -----------------------------------------------------------------------------


@wp.func
def _shape_world_transform(
    shape: int, shape_body: wp.array[int], shape_transform: wp.array[wp.transform], body_q: wp.array[wp.transform]
):
    body = shape_body[shape]
    if body < 0:
        return shape_transform[shape]
    return body_q[body] * shape_transform[shape]


@wp.func
def _point_velocity(
    body: int,
    point: wp.vec3,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
):
    if body < 0:
        return wp.vec3(0.0, 0.0, 0.0)
    twist = body_qd[body]
    x_com = wp.transform_point(body_q[body], body_com[body])
    return wp.spatial_top(twist) + wp.cross(wp.spatial_bottom(twist), point - x_com)


@wp.struct
class TendonRowContext:
    """Per-thread inputs for the tendon Jacobian accumulation (one articulation row).

    ``accumulate`` selects what :func:`evaluate_tendon` computes besides the length: ``0`` nothing, ``1`` the
    Jacobian ``dL/dq`` into ``J_group``, ``2`` the rate ``J qd`` added to the returned prescribed rate.
    """

    accumulate: int
    art: int
    group_idx: int
    slot: int
    dof_start: int
    origin: wp.vec3


@wp.func
def _accumulate_span(
    ctx: TendonRowContext,
    p0: wp.vec3,
    body0: int,
    p1: wp.vec3,
    body1: int,
    scale: float,
    body_to_articulation: wp.array[int],
    body_to_joint: wp.array[int],
    joint_ancestor: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_S_s: wp.array[wp.spatial_vector],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    qd: wp.array[float],
    J_group: wp.array3d[float],
) -> tuple[float, float]:
    """Add one straight span: returns its scaled length [m] and its prescribed rate [m/s].

    The span's generalized Jacobian ``scale * u . (Jp(body1, p1) - Jp(body0, p0))`` is added to the row,
    walking each end's ancestor joints; an end outside the articulation (the world, or a kinematic body)
    contributes its point velocity to the returned prescribed rate instead.
    """
    d = p1 - p0
    length = wp.length(d)
    rate = float(0.0)
    if ctx.accumulate == 0 or body0 == body1 or length < _WRAP_MINVAL:
        return scale * length, rate
    u = d / length
    for end in range(2):
        body = body0
        point = p0
        sign = -scale
        if end == 1:
            body = body1
            point = p1
            sign = scale
        if body < 0:
            continue
        if body_to_articulation[body] != ctx.art:
            rate += sign * wp.dot(u, _point_velocity(body, point, body_q, body_qd, body_com))
            continue
        rel = point - ctx.origin
        curr = body_to_joint[body]
        while curr != -1:
            for dof in range(joint_qd_start[curr], joint_qd_start[curr + 1]):
                S = joint_S_s[dof]
                lin = wp.vec3(S[0], S[1], S[2])
                ang = wp.vec3(S[3], S[4], S[5])
                jac = sign * wp.dot(u, lin + wp.cross(ang, rel))
                if ctx.accumulate == 1:
                    J_group[ctx.group_idx, ctx.slot, dof - ctx.dof_start] += jac
                else:
                    rate += jac * qd[dof]
            curr = joint_ancestor[curr]
    return scale * length, rate


@wp.func
def _site_position(
    shape: int, shape_body: wp.array[int], shape_transform: wp.array[wp.transform], body_q: wp.array[wp.transform]
) -> wp.vec3:
    return wp.transform_get_translation(_shape_world_transform(shape, shape_body, shape_transform, body_q))


@wp.func
def evaluate_tendon(
    ctx: TendonRowContext,
    tendon: int,
    tendon_type: wp.array[int],
    tendon_joint_adr: wp.array[int],
    tendon_joint_num: wp.array[int],
    tendon_joint: wp.array[int],
    tendon_coef: wp.array[float],
    tendon_wrap_adr: wp.array[int],
    tendon_wrap_num: wp.array[int],
    tendon_wrap_type: wp.array[int],
    tendon_wrap_shape: wp.array[int],
    tendon_wrap_sidesite: wp.array[int],
    tendon_wrap_prm: wp.array[float],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    shape_body: wp.array[int],
    shape_transform: wp.array[wp.transform],
    shape_type: wp.array[int],
    shape_scale: wp.array[wp.vec3],
    body_com: wp.array[wp.vec3],
    body_to_articulation: wp.array[int],
    body_to_joint: wp.array[int],
    joint_ancestor: wp.array[int],
    joint_S_s: wp.array[wp.spatial_vector],
    joint_q: wp.array[float],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    qd: wp.array[float],
    J_group: wp.array3d[float],
) -> tuple[float, float]:
    """Return the tendon length [m or rad] and its rate; accumulate ``dL/dq`` into the row (see ``accumulate``)."""
    length = float(0.0)
    rate = float(0.0)
    if tendon_type[tendon] == TENDON_TYPE_FIXED:
        adr = tendon_joint_adr[tendon]
        for e in range(adr, adr + tendon_joint_num[tendon]):
            joint = tendon_joint[e]
            coef = tendon_coef[e]
            length += coef * joint_q[joint_q_start[joint]]
            if ctx.accumulate == 1:
                J_group[ctx.group_idx, ctx.slot, joint_qd_start[joint] - ctx.dof_start] += coef
            elif ctx.accumulate == 2:
                rate += coef * qd[joint_qd_start[joint]]
        return length, rate

    adr = tendon_wrap_adr[tendon]
    num = tendon_wrap_num[tendon]
    scale = float(1.0)
    j = int(0)
    while j < num - 1:
        e = adr + j
        type0 = tendon_wrap_type[e]
        type1 = tendon_wrap_type[e + 1]
        if type0 == TENDON_WRAP_PULLEY:
            scale = 1.0 / tendon_wrap_prm[e]
            j += 1
            continue
        if type1 == TENDON_WRAP_PULLEY:
            j += 1
            continue
        site0 = tendon_wrap_shape[e]
        p0 = _site_position(site0, shape_body, shape_transform, body_q)
        b0 = shape_body[site0]
        if type1 == TENDON_WRAP_GEOM and j + 2 < num:
            geom = tendon_wrap_shape[e + 1]
            site1 = tendon_wrap_shape[e + 2]
            p1 = _site_position(site1, shape_body, shape_transform, body_q)
            b1 = shape_body[site1]
            bg = shape_body[geom]
            gtype = shape_type[geom]
            wlen = float(-1.0)
            g0 = p0
            g1 = p1
            if gtype == _GEO_SPHERE or gtype == _GEO_CYLINDER:
                side = wp.vec3(_WRAP_MAXVAL, _WRAP_MAXVAL, _WRAP_MAXVAL)
                sidesite = tendon_wrap_sidesite[e + 1]
                if sidesite >= 0:
                    side = _site_position(sidesite, shape_body, shape_transform, body_q)
                X_geom = _shape_world_transform(geom, shape_body, shape_transform, body_q)
                wlen, g0, g1 = tendon_wrap(p0, p1, X_geom, shape_scale[geom][0], gtype == _GEO_CYLINDER, side)
            if wlen >= 0.0:
                l0, r0 = _accumulate_span(
                    ctx, p0, b0, g0, bg, scale, body_to_articulation, body_to_joint, joint_ancestor,
                    joint_qd_start, joint_S_s, body_q, body_qd, body_com, qd, J_group,
                )  # fmt: skip
                l1, r1 = _accumulate_span(
                    ctx, g1, bg, p1, b1, scale, body_to_articulation, body_to_joint, joint_ancestor,
                    joint_qd_start, joint_S_s, body_q, body_qd, body_com, qd, J_group,
                )  # fmt: skip
                length += l0 + scale * wlen + l1
                rate += r0 + r1
            else:
                l0, r0 = _accumulate_span(
                    ctx, p0, b0, p1, b1, scale, body_to_articulation, body_to_joint, joint_ancestor,
                    joint_qd_start, joint_S_s, body_q, body_qd, body_com, qd, J_group,
                )  # fmt: skip
                length += l0
                rate += r0
            j += 2
            continue
        site1 = tendon_wrap_shape[e + 1]
        p1 = _site_position(site1, shape_body, shape_transform, body_q)
        l0, r0 = _accumulate_span(
            ctx, p0, b0, p1, shape_body[site1], scale, body_to_articulation, body_to_joint, joint_ancestor,
            joint_qd_start, joint_S_s, body_q, body_qd, body_com, qd, J_group,
        )  # fmt: skip
        length += l0
        rate += r0
        j += 1
    return length, rate


@wp.kernel
def eval_tendon_lengths(
    tendon_index: wp.array[int],
    tendon_type: wp.array[int],
    tendon_joint_adr: wp.array[int],
    tendon_joint_num: wp.array[int],
    tendon_joint: wp.array[int],
    tendon_coef: wp.array[float],
    tendon_wrap_adr: wp.array[int],
    tendon_wrap_num: wp.array[int],
    tendon_wrap_type: wp.array[int],
    tendon_wrap_shape: wp.array[int],
    tendon_wrap_sidesite: wp.array[int],
    tendon_wrap_prm: wp.array[float],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    shape_body: wp.array[int],
    shape_transform: wp.array[wp.transform],
    shape_type: wp.array[int],
    shape_scale: wp.array[wp.vec3],
    body_com: wp.array[wp.vec3],
    joint_q: wp.array[float],
    body_q: wp.array[wp.transform],
    no_int: wp.array[int],
    no_S: wp.array[wp.spatial_vector],
    no_J: wp.array3d[float],
    # outputs
    tendon_length: wp.array[float],
):
    """Evaluate the listed tendons' lengths [m or rad] from a joint/body state (no Jacobian; ``no_*`` are unread)."""
    tendon = tendon_index[wp.tid()]
    ctx = TendonRowContext()
    ctx.accumulate = 0
    length, _rate = evaluate_tendon(
        ctx, tendon, tendon_type, tendon_joint_adr, tendon_joint_num, tendon_joint, tendon_coef,
        tendon_wrap_adr, tendon_wrap_num, tendon_wrap_type, tendon_wrap_shape, tendon_wrap_sidesite,
        tendon_wrap_prm, joint_q_start, joint_qd_start, shape_body, shape_transform, shape_type, shape_scale,
        body_com, no_int, no_int, no_int, no_S, joint_q, body_q, no_S, joint_q, no_J,
    )  # fmt: skip
    tendon_length[tendon] = length


@wp.func
def _gate_limit_flags(
    flags: int, length: float, speed: float, gap: float, one_side: int, speed_scale: float, dt: float, bounds: wp.vec2
) -> tuple[int, int]:
    """Return a tendon's row flags and row count for this step; see :func:`gate_tendon_limit_rows`."""
    reach = float(gap)
    if speed_scale > 0.0 and not wp.isinf(gap):
        reach += speed_scale * dt * wp.abs(speed)
    lo = bounds[0]
    hi = bounds[1]
    lower = length <= lo + reach
    upper = length >= hi - reach
    if lower and upper and one_side != 0:
        predicted = length + dt * speed
        if predicted - lo <= hi - predicted:
            upper = False
        else:
            lower = False
    out = flags & ~(TENDON_ROW_LIMIT_LOWER | TENDON_ROW_LIMIT_UPPER)
    rows = int(0)
    if lower:
        out = out | TENDON_ROW_LIMIT_LOWER
        rows += 1
    if upper:
        out = out | TENDON_ROW_LIMIT_UPPER
        rows += 1
    if (out & TENDON_ROW_FRICTION) != 0:
        rows += 1
    return out, rows


@wp.kernel
def gate_tendon_limit_rows(
    tendon_list: wp.array[int],
    tendon_art: wp.array[int],
    tendon_flags: wp.array[int],
    tendon_gap: wp.array[float],
    tendon_one_side: wp.array[int],
    speed_scale: float,
    dt: float,
    tendon_range: wp.array[wp.vec2],
    articulation_origin: wp.array[wp.vec3],
    tendon_type: wp.array[int],
    tendon_joint_adr: wp.array[int],
    tendon_joint_num: wp.array[int],
    tendon_joint: wp.array[int],
    tendon_coef: wp.array[float],
    tendon_wrap_adr: wp.array[int],
    tendon_wrap_num: wp.array[int],
    tendon_wrap_type: wp.array[int],
    tendon_wrap_shape: wp.array[int],
    tendon_wrap_sidesite: wp.array[int],
    tendon_wrap_prm: wp.array[float],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    shape_body: wp.array[int],
    shape_transform: wp.array[wp.transform],
    shape_type: wp.array[int],
    shape_scale: wp.array[wp.vec3],
    body_com: wp.array[wp.vec3],
    body_to_articulation: wp.array[int],
    body_to_joint: wp.array[int],
    joint_ancestor: wp.array[int],
    joint_S_s: wp.array[wp.spatial_vector],
    joint_q: wp.array[float],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    v_hat: wp.array[float],
    no_J: wp.array3d[float],
    # outputs
    step_flags: wp.array[int],
    step_rows: wp.array[int],
    tendon_length: wp.array[float],
):
    """Choose the limit rows of this step for the listed limited tendons (one thread per tendon).

    A bound gets its row when the length is within the tendon's gap of it, widened by
    ``speed_scale * dt * |dL/dt|`` with the rate taken at the unconstrained velocity ``v_hat``. A
    one-sided tendon (range wider than twice its gap, so at most one limit row is reserved) keeps
    only the bound nearer the predicted end-of-step length. The friction-loss row is kept as is. Writes the
    length [m or rad], which the row writer skips for a tendon left without rows.
    """
    k = tendon_list[wp.tid()]
    flags = tendon_flags[k]
    ctx = TendonRowContext()
    ctx.accumulate = 2
    ctx.art = tendon_art[k]
    ctx.origin = articulation_origin[ctx.art]
    length, speed = evaluate_tendon(
        ctx, k, tendon_type, tendon_joint_adr, tendon_joint_num, tendon_joint, tendon_coef,
        tendon_wrap_adr, tendon_wrap_num, tendon_wrap_type, tendon_wrap_shape, tendon_wrap_sidesite,
        tendon_wrap_prm, joint_q_start, joint_qd_start, shape_body, shape_transform, shape_type, shape_scale,
        body_com, body_to_articulation, body_to_joint, joint_ancestor, joint_S_s, joint_q, body_q, body_qd,
        v_hat, no_J,
    )  # fmt: skip
    tendon_length[k] = length
    new_flags, rows = _gate_limit_flags(
        flags, length, speed, tendon_gap[k], tendon_one_side[k], speed_scale, dt, tendon_range[k]
    )
    step_flags[k] = new_flags
    step_rows[k] = rows


@wp.kernel
def gate_fixed_tendon_limit_rows(
    tendon_list: wp.array[int],
    tendon_flags: wp.array[int],
    tendon_gap: wp.array[float],
    tendon_one_side: wp.array[int],
    speed_scale: float,
    dt: float,
    tendon_range: wp.array[wp.vec2],
    tendon_joint_adr: wp.array[int],
    tendon_joint_num: wp.array[int],
    tendon_joint: wp.array[int],
    tendon_coef: wp.array[float],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_q: wp.array[float],
    v_hat: wp.array[float],
    # outputs
    step_flags: wp.array[int],
    step_rows: wp.array[int],
    tendon_length: wp.array[float],
):
    """:func:`gate_tendon_limit_rows` for fixed tendons: ``L = sum c q`` and ``dL/dt = sum c v_hat``."""
    k = tendon_list[wp.tid()]
    length = float(0.0)
    speed = float(0.0)
    adr = tendon_joint_adr[k]
    for e in range(adr, adr + tendon_joint_num[k]):
        joint = tendon_joint[e]
        coef = tendon_coef[e]
        length += coef * joint_q[joint_q_start[joint]]
        speed += coef * v_hat[joint_qd_start[joint]]
    tendon_length[k] = length
    flags, rows = _gate_limit_flags(
        tendon_flags[k], length, speed, tendon_gap[k], tendon_one_side[k], speed_scale, dt, tendon_range[k]
    )
    step_flags[k] = flags
    step_rows[k] = rows


@wp.kernel
def allocate_tendon_slots(
    tendon_rows: wp.array[int],
    tendon_world: wp.array[int],
    max_constraints: int,
    # outputs
    tendon_slot: wp.array[int],
    world_slot_counter: wp.array[int],
    first_rejected_slot: wp.array[int],
    dropped_rows: wp.array[int],
    row_slot_mode: int,
    element_row_cursor: wp.array[int],
):
    """Reserve consecutive dense rows per tendon; a tendon that does not fit is dropped whole and counted."""
    k = wp.tid()
    tendon_slot[k] = -1
    rows = tendon_rows[k]
    if rows == 0:
        return
    tendon_slot[k] = _reserve_bilateral_rows(
        tendon_world[k],
        rows,
        max_constraints,
        world_slot_counter,
        first_rejected_slot,
        dropped_rows,
        row_slot_mode,
        element_row_cursor,
        k,
    )


@wp.func
def _write_tendon_row(
    world: int,
    slot: int,
    row_type: int,
    mu: float,
    phi: float,
    target: float,
    world_row_type: wp.array2d[int],
    world_row_parent: wp.array2d[int],
    world_row_mu: wp.array2d[float],
    world_phi: wp.array2d[float],
    world_target_velocity: wp.array2d[float],
):
    world_row_type[world, slot] = row_type
    world_row_parent[world, slot] = -1
    world_row_mu[world, slot] = mu
    world_phi[world, slot] = phi
    world_target_velocity[world, slot] = target


@wp.func
def _copy_row(J_group: wp.array3d[float], group_idx: int, src: int, dst: int, n_dofs: int, scale: float):
    for d in range(n_dofs):
        J_group[group_idx, dst, d] = scale * J_group[group_idx, src, d]


_vec6 = wp.types.vector(length=6, dtype=wp.float32)
_vec10 = wp.types.vector(length=10, dtype=wp.float32)
ACTUATOR_CTRL_LIMITED = wp.constant(1)
ACTUATOR_FORCE_LIMITED = wp.constant(2)
_AFFINE = wp.constant(1)


@wp.struct
class TendonActuators:
    """Device tables for the actuators driving tendons (``mujoco:actuator_*`` with a TENDON transmission)."""

    start: wp.array[int]
    """Per tendon, the first index into ``actuator``; ``start[k + 1]`` ends the range."""
    actuator: wp.array[int]
    flags: wp.array[int]
    gear: wp.array[_vec6]
    gainprm: wp.array[_vec10]
    biasprm: wp.array[_vec10]
    gaintype: wp.array[int]
    biastype: wp.array[int]
    ctrlrange: wp.array[wp.vec2]
    forcerange: wp.array[wp.vec2]
    ctrl: wp.array[float]


@wp.kernel
def populate_tendon_rows_for_size(
    tendon_list: wp.array[int],
    tendon_art: wp.array[int],
    tendon_slot: wp.array[int],
    tendon_flags: wp.array[int],
    art_group_idx: wp.array[int],
    art_to_world: wp.array[int],
    articulation_dof_start: wp.array[int],
    articulation_origin: wp.array[wp.vec3],
    n_dofs: int,
    tendon_type: wp.array[int],
    tendon_joint_adr: wp.array[int],
    tendon_joint_num: wp.array[int],
    tendon_joint: wp.array[int],
    tendon_coef: wp.array[float],
    tendon_wrap_adr: wp.array[int],
    tendon_wrap_num: wp.array[int],
    tendon_wrap_type: wp.array[int],
    tendon_wrap_shape: wp.array[int],
    tendon_wrap_sidesite: wp.array[int],
    tendon_wrap_prm: wp.array[float],
    tendon_range: wp.array[wp.vec2],
    tendon_frictionloss: wp.array[float],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    shape_body: wp.array[int],
    shape_transform: wp.array[wp.transform],
    shape_type: wp.array[int],
    shape_scale: wp.array[wp.vec3],
    body_com: wp.array[wp.vec3],
    body_to_articulation: wp.array[int],
    body_to_joint: wp.array[int],
    joint_ancestor: wp.array[int],
    joint_S_s: wp.array[wp.spatial_vector],
    joint_q: wp.array[float],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    dt: float,
    # outputs
    J_group: wp.array3d[float],
    world_row_type: wp.array2d[int],
    world_row_parent: wp.array2d[int],
    world_row_mu: wp.array2d[float],
    world_phi: wp.array2d[float],
    world_target_velocity: wp.array2d[float],
    tendon_length: wp.array[float],
):
    """Write the limit and friction-loss rows of one size group's tendons (one thread per tendon).

    The Jacobian ``J = dL/dq`` is accumulated into the tendon's first slot and copied, with each row's sign,
    into the others. Rows come in the order lower limit, upper limit, friction loss. Limit rows bias like
    joint limits (Baumgarte on violation, speculative otherwise); the friction row is boxed to ``[-f dt, f dt]``.
    ``rate`` is the prescribed (non-DOF) part of ``J v`` and enters each row's target.
    """
    i = wp.tid()
    k = tendon_list[i]
    base = tendon_slot[k]
    if base < 0:
        return
    art = tendon_art[k]
    world = art_to_world[art]
    ctx = TendonRowContext()
    ctx.accumulate = 1
    ctx.art = art
    ctx.group_idx = art_group_idx[art]
    ctx.slot = base
    ctx.dof_start = articulation_dof_start[art]
    ctx.origin = articulation_origin[art]
    for d in range(n_dofs):
        J_group[ctx.group_idx, base, d] = 0.0
    length, rate = evaluate_tendon(
        ctx, k, tendon_type, tendon_joint_adr, tendon_joint_num, tendon_joint, tendon_coef,
        tendon_wrap_adr, tendon_wrap_num, tendon_wrap_type, tendon_wrap_shape, tendon_wrap_sidesite,
        tendon_wrap_prm, joint_q_start, joint_qd_start, shape_body, shape_transform, shape_type, shape_scale,
        body_com, body_to_articulation, body_to_joint, joint_ancestor, joint_S_s, joint_q, body_q, body_qd,
        joint_q, J_group,
    )  # fmt: skip
    tendon_length[k] = length

    g = ctx.group_idx
    flags = tendon_flags[k]
    slot = base
    # The base row holds +J. A lower limit keeps it; every other row is a signed copy. When the gate leaves only
    # the upper limit, that row negates the base slot in place, and later copies undo the sign.
    base_sign = float(1.0)
    if (flags & TENDON_ROW_LIMIT_LOWER) != 0:
        lo = tendon_range[k][0]
        _write_tendon_row(
            world, slot, PGS_CONSTRAINT_TYPE_JOINT_LIMIT, 0.0, length - lo, -rate,
            world_row_type, world_row_parent, world_row_mu, world_phi, world_target_velocity,
        )  # fmt: skip
        slot += 1
    if (flags & TENDON_ROW_LIMIT_UPPER) != 0:
        hi = tendon_range[k][1]
        _copy_row(J_group, g, base, slot, n_dofs, -1.0)
        if slot == base:
            base_sign = -1.0
        _write_tendon_row(
            world, slot, PGS_CONSTRAINT_TYPE_JOINT_LIMIT, 0.0, hi - length, rate,
            world_row_type, world_row_parent, world_row_mu, world_phi, world_target_velocity,
        )  # fmt: skip
        slot += 1
    if (flags & TENDON_ROW_FRICTION) != 0:
        if slot != base:
            _copy_row(J_group, g, base, slot, n_dofs, base_sign)
        bound = wp.max(tendon_frictionloss[k], 0.0) * dt
        _write_tendon_row(
            world, slot, PGS_CONSTRAINT_TYPE_JOINT_FRICTION, bound, 0.0, -rate,
            world_row_type, world_row_parent, world_row_mu, world_phi, world_target_velocity,
        )  # fmt: skip


@wp.kernel
def apply_tendon_mass_terms_for_size(
    tendon_list: wp.array[int],
    tendon_art: wp.array[int],
    tendon_armature: wp.array[float],
    tendon_stiffness: wp.array[float],
    tendon_damping: wp.array[float],
    tendon_spring_length: wp.array[wp.vec2],
    actuators: TendonActuators,
    art_group_idx: wp.array[int],
    articulation_dof_start: wp.array[int],
    articulation_origin: wp.array[wp.vec3],
    mass_update_mask: wp.array[int],
    n_dofs: int,
    tendon_type: wp.array[int],
    tendon_joint_adr: wp.array[int],
    tendon_joint_num: wp.array[int],
    tendon_joint: wp.array[int],
    tendon_coef: wp.array[float],
    tendon_wrap_adr: wp.array[int],
    tendon_wrap_num: wp.array[int],
    tendon_wrap_type: wp.array[int],
    tendon_wrap_shape: wp.array[int],
    tendon_wrap_sidesite: wp.array[int],
    tendon_wrap_prm: wp.array[float],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    shape_body: wp.array[int],
    shape_transform: wp.array[wp.transform],
    shape_type: wp.array[int],
    shape_scale: wp.array[wp.vec3],
    body_com: wp.array[wp.vec3],
    body_to_articulation: wp.array[int],
    body_to_joint: wp.array[int],
    joint_ancestor: wp.array[int],
    joint_S_s: wp.array[wp.spatial_vector],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    dt: float,
    implicit_spring: int,
    scratch_index: wp.array[int],
    # in/out
    J_scratch: wp.array3d[float],
    J_prev: wp.array2d[float],
    has_prev: wp.array[int],
    H_group: wp.array3d[float],
    joint_tau: wp.array[float],
    tendon_length: wp.array[float],
    tendon_force: wp.array2d[float],
    actuator_force: wp.array[float],
):
    """Fold a tendon's armature, spring-damper and actuators into the mass matrix and the joint forces.

    This is the joint-style treatment (Steel's augmented drives and implicit passive joints) carried along
    the tendon Jacobian ``J``: every term adds ``K J^T J`` to the rebuilt mass matrix and ``J^T f`` to the
    joint forces, so the factorization solves the implicit step and no PGS row is needed.

    - Armature ``a``: ``K += a`` and ``f -= a dJ/dt qd`` (backward difference of ``J qd``, zero on the first
      step), as MuJoCo's tendon armature.
    - Spring-damper: ``f = -k (L - L_s) - b dL/dt`` (dead band ``[L_lo, L_hi]``); the damping is implicit
      (``K += b dt``) and, with ``implicit_spring``, the stiffness too (``K += k dt^2``, ``f -= k dt dL/dt``),
      as Steel's passive joint modes. Inside a dead band the spring is slack and adds no implicit stiffness.
    - Actuators on ``x = gear L``: the affine force ``f = c0 + c1 x + c2 dx/dt`` with its restoring part
      implicit like a joint drive (``K += gear^2 (k dt^2 + b dt)``, ``f -= k dt dx/dt``, ``k = max(-c1, 0)``,
      ``b = max(-c2, 0)``), the explicit force clamped to the force range as a drive's effort limit.

    The mass term is added only where CRBA rebuilt ``H`` this step; the forces every step.
    """
    i = wp.tid()
    k = tendon_list[i]
    art = tendon_art[k]
    ctx = TendonRowContext()
    ctx.accumulate = 1
    ctx.art = art
    # Row of this tendon in the spatial-only Jacobian scratch (-1 for fixed tendons, which never read it).
    s = scratch_index[k]
    ctx.group_idx = s
    ctx.slot = 0
    ctx.dof_start = articulation_dof_start[art]
    ctx.origin = articulation_origin[art]
    armature = tendon_armature[k]
    # A fixed tendon's Jacobian is its constant coefficient list: use it sparsely, without the dense scratch
    # row, and its armature has no dJ/dt term.
    fixed = tendon_type[k] == TENDON_TYPE_FIXED
    j_adr = tendon_joint_adr[k]
    j_end = j_adr + tendon_joint_num[k]
    length = float(0.0)
    rate = float(0.0)
    jqd = float(0.0)
    jqd_prev = float(0.0)
    if fixed:
        for e in range(j_adr, j_end):
            joint = tendon_joint[e]
            coef = tendon_coef[e]
            length += coef * joint_q[joint_q_start[joint]]
            jqd += coef * joint_qd[joint_qd_start[joint]]
    else:
        for d in range(n_dofs):
            J_scratch[s, 0, d] = 0.0
        length, rate = evaluate_tendon(
            ctx, k, tendon_type, tendon_joint_adr, tendon_joint_num, tendon_joint, tendon_coef,
            tendon_wrap_adr, tendon_wrap_num, tendon_wrap_type, tendon_wrap_shape, tendon_wrap_sidesite,
            tendon_wrap_prm, joint_q_start, joint_qd_start, shape_body, shape_transform, shape_type, shape_scale,
            body_com, body_to_articulation, body_to_joint, joint_ancestor, joint_S_s, joint_q, body_q, body_qd,
            joint_q, J_scratch,
        )  # fmt: skip
        for d in range(n_dofs):
            jd = J_scratch[s, 0, d]
            if jd != 0.0:
                jqd += jd * joint_qd[ctx.dof_start + d]
        if armature != 0.0 and has_prev[k] != 0:
            for d in range(n_dofs):
                jqd_prev += J_prev[s, d] * joint_qd[ctx.dof_start + d]
    tendon_length[k] = length
    velocity = jqd + rate

    K = armature
    force = float(0.0)
    if not fixed and armature != 0.0 and has_prev[k] != 0:
        force -= armature * (jqd - jqd_prev) / dt

    stiffness = wp.max(tendon_stiffness[k], 0.0)
    damping = wp.max(tendon_damping[k], 0.0)
    if stiffness > 0.0 or damping > 0.0:
        band = tendon_spring_length[k]
        stretch = float(0.0)
        if length < band[0]:
            stretch = length - band[0]
        elif length > band[1]:
            stretch = length - band[1]
        spring = -stiffness * stretch - damping * velocity
        K += damping * dt
        # Inside a dead band [L_lo, L_hi] with L_lo < L_hi the spring is slack, so it adds no implicit stiffness.
        slack = band[0] < band[1] and length >= band[0] and length <= band[1]
        if implicit_spring != 0 and not slack:
            K += stiffness * dt * dt
            spring -= stiffness * dt * velocity
        tendon_force[k, 1] = spring
        force += spring

    for e in range(actuators.start[k], actuators.start[k + 1]):
        act = actuators.actuator[e]
        gear = actuators.gear[act][0]
        x = gear * length
        dx = gear * velocity
        u = actuators.ctrl[act]
        act_flags = actuators.flags[act]
        if (act_flags & ACTUATOR_CTRL_LIMITED) != 0:
            u = wp.clamp(u, actuators.ctrlrange[act][0], actuators.ctrlrange[act][1])
        gp = actuators.gainprm[act]
        c0 = gp[0] * u
        c1 = float(0.0)
        c2 = float(0.0)
        if actuators.gaintype[act] == _AFFINE:
            c1 += gp[1] * u
            c2 += gp[2] * u
        if actuators.biastype[act] == _AFFINE:
            bp = actuators.biasprm[act]
            c0 += bp[0]
            c1 += bp[1]
            c2 += bp[2]
        k_act = wp.max(-c1, 0.0)
        b_act = wp.max(-c2, 0.0)
        drive = c0 + c1 * x + c2 * dx - k_act * dt * dx
        if (act_flags & ACTUATOR_FORCE_LIMITED) != 0:
            drive = wp.clamp(drive, actuators.forcerange[act][0], actuators.forcerange[act][1])
        actuator_force[act] = drive
        K += gear * gear * (k_act * dt * dt + b_act * dt)
        force += gear * drive

    # Every world's tendons add into the same H and joint forces, so the atomic adds contend: skip the zero
    # entries of J (a fixed tendon spans a few DOFs) rather than add zeros.
    g = art_group_idx[art]
    if fixed:
        if K != 0.0 and mass_update_mask[art] != 0:
            for e in range(j_adr, j_end):
                r = joint_qd_start[tendon_joint[e]] - ctx.dof_start
                jr = K * tendon_coef[e]
                for f in range(j_adr, j_end):
                    H_group[g, r, joint_qd_start[tendon_joint[f]] - ctx.dof_start] += jr * tendon_coef[f]
        if force != 0.0:
            for e in range(j_adr, j_end):
                joint_tau[joint_qd_start[tendon_joint[e]]] += tendon_coef[e] * force
        return
    if K != 0.0 and mass_update_mask[art] != 0:
        for r in range(n_dofs):
            jr = J_scratch[s, 0, r]
            if jr != 0.0:
                jr *= K
                for c in range(n_dofs):
                    jc = J_scratch[s, 0, c]
                    if jc != 0.0:
                        H_group[g, r, c] += jr * jc
    if force != 0.0:
        for d in range(n_dofs):
            jd = J_scratch[s, 0, d]
            if jd != 0.0:
                joint_tau[ctx.dof_start + d] += jd * force
    # Only armature reads the previous Jacobian; without it, restart the backward difference cold.
    if armature != 0.0:
        for d in range(n_dofs):
            J_prev[s, d] = J_scratch[s, 0, d]
        has_prev[k] = 1
    else:
        has_prev[k] = 0


@wp.kernel
def clear_tendon_armature_history(
    world_mask: wp.array[wp.bool],
    world_count: int,
    tendon_art: wp.array[int],
    articulation_world: wp.array[int],
    # outputs
    has_prev: wp.array[int],
):
    """Restart the armature ``dJ/dt`` backward difference of the tendons in the worlds ``world_mask`` selects."""
    k = wp.tid()
    art = tendon_art[k]
    if art >= 0 and reset_world_selected(articulation_world[art], world_mask, world_count):
        has_prev[k] = 0


@wp.kernel
def gather_tendon_forces(
    tendon_slot: wp.array[int],
    tendon_world: wp.array[int],
    tendon_flags: wp.array[int],
    world_impulses: wp.array2d[float],
    inv_dt: float,
    # outputs
    tendon_force: wp.array2d[float],
):
    """Split each tendon's solved impulses into limit, spring, and friction forces [N or N·m].

    Every column is the force along the tendon length (positive lengthens it); the limit column is the
    lower minus the upper limit row's push. The spring column is written by
    :func:`apply_tendon_mass_terms_for_size`, which applies the spring-damper.
    """
    k = wp.tid()
    tendon_force[k, 0] = 0.0
    tendon_force[k, 2] = 0.0
    slot = tendon_slot[k]
    if slot < 0:
        return
    world = tendon_world[k]
    flags = tendon_flags[k]
    limit = float(0.0)
    if (flags & TENDON_ROW_LIMIT_LOWER) != 0:
        limit += world_impulses[world, slot] * inv_dt
        slot += 1
    if (flags & TENDON_ROW_LIMIT_UPPER) != 0:
        limit -= world_impulses[world, slot] * inv_dt
        slot += 1
    tendon_force[k, 0] = limit
    if (flags & TENDON_ROW_FRICTION) != 0:
        tendon_force[k, 2] = world_impulses[world, slot] * inv_dt


class TendonPlan:
    """Static per-tendon tables for the tendon row family (host validation, device lookup arrays)."""

    def __init__(self):
        self.count = 0
        self.actuator_count = 0
        self.length = None
        self.length_index = None
        self.sizes: dict[int, wp.array] = {}
        self.mass_sizes: dict[int, wp.array] = {}
        self.has_rows = False
        self.has_friction_rows = False
        self.gated = False
        self.gate_fixed_index = None
        self.gate_spatial_index = None

    @classmethod
    def build(
        cls,
        model,
        *,
        body_to_articulation: np.ndarray | None,
        art_to_world: np.ndarray | None,
        response_dof_count: np.ndarray,
    ) -> TendonPlan:
        """Validate the model's ``mujoco:tendon_*`` and tendon ``mujoco:actuator_*`` attributes and build tables.

        Tendons whose DOFs span several dynamic articulations, or that touch multi-DoF joints in a
        fixed tendon, warn and are ignored, with their actuators. Range limits and friction loss become
        rows; armature, the spring-damper and actuators become mass-matrix terms (``mass_sizes``). Every
        valid tendon's length is reported, including tendons without rows or mass terms.
        """
        plan = cls()
        mujoco = getattr(model, "mujoco", None)
        tendon_type_attr = getattr(mujoco, "tendon_type", None) if mujoco is not None else None
        if tendon_type_attr is None or tendon_type_attr.shape[0] == 0 or body_to_articulation is None:
            return plan
        n = int(tendon_type_attr.shape[0])

        def host(name, default=None):
            attr = getattr(mujoco, name, None)
            return attr.numpy() if attr is not None and attr.shape[0] else default

        ttype = tendon_type_attr.numpy()
        joint_adr, joint_num = host("tendon_joint_adr"), host("tendon_joint_num")
        tjoint = host("tendon_joint", np.zeros(0, dtype=np.int32))
        wrap_adr, wrap_num = host("tendon_wrap_adr"), host("tendon_wrap_num")
        wrap_type = host("tendon_wrap_type", np.zeros(0, dtype=np.int32))
        wrap_shape = host("tendon_wrap_shape", np.zeros(0, dtype=np.int32))
        wrap_prm = host("tendon_wrap_prm", np.zeros(0, dtype=np.float32))
        limited = host("tendon_limited", np.full(n, 2, dtype=np.int32))
        trange = host("tendon_range", np.zeros((n, 2), dtype=np.float32))
        stiffness = host("tendon_stiffness", np.zeros(n, dtype=np.float32))
        damping = host("tendon_damping", np.zeros(n, dtype=np.float32))
        frictionloss = host("tendon_frictionloss", np.zeros(n, dtype=np.float32))
        armature = host("tendon_armature", np.zeros(n, dtype=np.float32))
        springlength = host("tendon_springlength", np.full((n, 2), -1.0, dtype=np.float32))
        labels = list(getattr(mujoco, "tendon_label", None) or [])
        tendon_actuators = cls._tendon_actuators(mujoco, n)

        joint_articulation = model.joint_articulation.numpy()
        joint_q_start = model.joint_q_start.numpy()
        joint_qd_start = model.joint_qd_start.numpy()
        shape_body = model.shape_body.numpy()
        kinematic = (model.body_flags.numpy() & int(BodyFlags.KINEMATIC)) != 0

        art = np.full(n, -1, dtype=np.int32)
        valid = np.zeros(n, dtype=bool)
        flags = np.zeros(n, dtype=np.int32)
        has_mass = np.zeros(n, dtype=bool)
        for k in range(n):
            name = labels[k] if k < len(labels) and labels[k] else f"tendon_{k}"
            arts = set()
            if int(ttype[k]) == 0:
                joints = tjoint[int(joint_adr[k]) : int(joint_adr[k]) + int(joint_num[k])]
                bad = [
                    int(j)
                    for j in joints
                    if joint_qd_start[j + 1] - joint_qd_start[j] != 1 or joint_q_start[j + 1] - joint_q_start[j] != 1
                ]
                if bad:
                    warnings.warn(
                        f"SolverSteel: fixed tendon '{name}' references multi-DoF joints {bad}; ignored.",
                        NewtonWarning,
                        stacklevel=3,
                    )
                    continue
                arts = {int(joint_articulation[j]) for j in joints}
            else:
                start = int(wrap_adr[k])
                elements = range(start, start + int(wrap_num[k]))
                if any(int(wrap_type[e]) == 2 and float(wrap_prm[e]) <= 0.0 for e in elements):
                    warnings.warn(
                        f"SolverSteel: tendon '{name}' has a non-positive pulley divisor; ignored.",
                        NewtonWarning,
                        stacklevel=3,
                    )
                    continue
                for e in elements:
                    if int(wrap_type[e]) == 2:
                        continue
                    body = int(shape_body[int(wrap_shape[e])])
                    if body >= 0 and not kinematic[body]:
                        arts.add(int(body_to_articulation[body]))
            if len(arts) != 1 or -1 in arts:
                warnings.warn(
                    f"SolverSteel: tendon '{name}' must move DOFs of exactly one articulation "
                    f"(touches {sorted(arts)}); cross-articulation and DOF-free tendons are ignored.",
                    NewtonWarning,
                    stacklevel=3,
                )
                continue
            a = arts.pop()
            if response_dof_count[a] <= 0:
                continue
            valid[k] = True
            lim = int(limited[k])
            if lim == 1 or (lim == 2 and float(trange[k, 0]) < float(trange[k, 1])):
                flags[k] |= int(TENDON_ROW_LIMIT_LOWER) | int(TENDON_ROW_LIMIT_UPPER)
            if float(frictionloss[k]) > 0.0:
                flags[k] |= int(TENDON_ROW_FRICTION)
            has_mass[k] = bool(
                float(armature[k]) != 0.0 or float(stiffness[k]) > 0.0 or float(damping[k]) > 0.0 or tendon_actuators[k]
            )
            if flags[k] or has_mass[k]:
                art[k] = a

        active = np.flatnonzero(art >= 0)
        # Tendons without rows or mass terms get their length from a length-only pass every step.
        length_only = np.flatnonzero(valid & (flags == 0) & ~has_mass)
        device = model.device
        if active.size or length_only.size:
            plan.length = wp.zeros(n, dtype=float, device=device)
        if length_only.size:
            plan.length_index = wp.array(length_only.astype(np.int32), dtype=wp.int32, device=device)
        if active.size == 0:
            return plan
        actuator_count = tendon_actuators.actuator_count
        actuator_start = np.zeros(n + 1, dtype=np.int32)
        actuator_list: list[int] = []
        rows = np.zeros(n, dtype=np.int32)
        for k in range(n):
            if art[k] >= 0:
                rows[k] = bin(int(flags[k])).count("1")
                for act in tendon_actuators[k]:
                    actuator_list.append(act)
            actuator_start[k + 1] = len(actuator_list)
        world = np.zeros(n, dtype=np.int32)
        world[active] = art_to_world[art[active]]

        plan.count = n
        plan.actuator_count = actuator_count
        plan.art_np = art
        plan.world_np = world
        plan.rows_np = rows
        plan.flags_np = flags
        plan.range_np = trange
        plan.springlength_np = springlength
        plan._paths = (ttype, joint_adr, joint_num, tjoint, wrap_adr, wrap_num, wrap_type, wrap_shape)
        plan.art = wp.array(art, dtype=wp.int32, device=device)
        plan.world = wp.array(world, dtype=wp.int32, device=device)
        plan.rows = wp.array(rows, dtype=wp.int32, device=device)
        plan.flags = wp.array(flags, dtype=wp.int32, device=device)
        plan.slot = wp.full(n, -1, dtype=wp.int32, device=device)
        plan.force = wp.zeros((n, 3), dtype=float, device=device)
        plan.spring_length = wp.zeros(n, dtype=wp.vec2, device=device)
        plan.actuator_start = wp.array(actuator_start, dtype=wp.int32, device=device)
        plan.actuator_list = wp.array(np.asarray(actuator_list or [0], dtype=np.int32), dtype=wp.int32, device=device)
        plan.actuator_flags = wp.array(tendon_actuators.flags, dtype=wp.int32, device=device)
        plan.actuator_force = wp.zeros(max(actuator_count, 1), dtype=float, device=device)
        sizes = response_dof_count[art[active]]
        plan.sizes = {
            int(size): wp.array(active[sizes == size].astype(np.int32), dtype=wp.int32, device=device)
            for size in np.unique(sizes)
        }
        mass_tendons = active[has_mass[active]]
        mass_sizes = response_dof_count[art[mass_tendons]]
        plan.mass_sizes = {
            int(size): wp.array(mass_tendons[mass_sizes == size].astype(np.int32), dtype=wp.int32, device=device)
            for size in np.unique(mass_sizes)
        }
        plan.has_rows = bool(rows.sum())
        plan.has_friction_rows = bool(np.any(flags[active] & int(TENDON_ROW_FRICTION)))
        plan.reserve_np = rows
        plan.step_flags = plan.flags
        plan.step_rows = plan.rows
        if plan.mass_sizes:
            # Only spatial tendons use the dense Jacobian scratch rows; a fixed tendon's Jacobian is its coefficients.
            spatial = mass_tendons[ttype[mass_tendons] != int(TENDON_TYPE_FIXED)]
            scratch_index = np.full(n, -1, dtype=np.int32)
            scratch_index[spatial] = np.arange(spatial.size, dtype=np.int32)
            max_size = int(response_dof_count[art[spatial]].max()) if spatial.size else 1
            rows_scratch = max(int(spatial.size), 1)
            plan.mass_scratch_index = wp.array(scratch_index, dtype=wp.int32, device=device)
            plan.mass_J = wp.zeros((rows_scratch, 1, max_size), dtype=float, device=device)
            plan.mass_J_prev = wp.zeros((rows_scratch, max_size), dtype=float, device=device)
            plan.mass_has_prev = wp.zeros(n, dtype=wp.int32, device=device)
        return plan

    def limited_np(self) -> np.ndarray:
        """Mask of the enforced tendons with range-limit rows."""
        limit = int(TENDON_ROW_LIMIT_LOWER) | int(TENDON_ROW_LIMIT_UPPER)
        return (self.art_np >= 0) & ((self.flags_np & limit) != 0)

    def touched_joints(self, model, k: int) -> set[int]:
        """Joints whose DOFs tendon ``k``'s length depends on: a fixed tendon's joints, or the tree joints above a
        spatial tendon's site and wrap bodies."""
        ttype, joint_adr, joint_num, tjoint, wrap_adr, wrap_num, wrap_type, wrap_shape = self._paths
        if int(ttype[k]) == 0:
            return {int(j) for j in tjoint[int(joint_adr[k]) : int(joint_adr[k]) + int(joint_num[k])]}
        if getattr(self, "_parent_joint", None) is None:
            tree = np.flatnonzero(model.joint_articulation.numpy() >= 0)
            self._parent_joint = dict(zip(model.joint_child.numpy()[tree].tolist(), tree.tolist(), strict=True))
            self._joint_parent = model.joint_parent.numpy()
            self._shape_body = model.shape_body.numpy()
        joints = set()
        for e in range(int(wrap_adr[k]), int(wrap_adr[k]) + int(wrap_num[k])):
            if int(wrap_type[e]) == 2:
                continue
            body = int(self._shape_body[int(wrap_shape[e])])
            while body in self._parent_joint and self._parent_joint[body] not in joints:
                joint = self._parent_joint[body]
                joints.add(joint)
                body = int(self._joint_parent[joint])
        return joints

    def configure_limit_gate(self, gap: np.ndarray, device) -> None:
        """Gate the limit rows of the tendons with a finite ``gap`` [m or rad] (``inf`` keeps both rows every step).

        A gated tendon whose range is wider than twice its gap reserves one limit row instead of two: both bounds
        can be within the gap only when the range is at most twice it, so :func:`gate_tendon_limit_rows` keeps the
        bound nearer the predicted length. ``reserve_np`` is each tendon's most rows in any step.
        """
        if not self.count:
            return
        gated = self.limited_np() & np.isfinite(gap)
        if not gated.any():
            return
        width = self.range_np[:, 1].astype(np.float64) - self.range_np[:, 0]
        one_side = gated & (width > 2.0 * np.where(gated, gap, 0.0))
        self.gated = True
        self.reserve_np = self.rows_np - one_side.astype(np.int32)
        fixed = self._paths[0] == 0
        self.gate_fixed_index = self._index_array(gated & fixed, device)
        self.gate_spatial_index = self._index_array(gated & ~fixed, device)
        self.gap = wp.array(np.where(gated, gap, np.inf).astype(np.float32), dtype=float, device=device)
        self.one_side = wp.array(one_side.astype(np.int32), dtype=wp.int32, device=device)
        self.step_flags = wp.array(self.flags_np, dtype=wp.int32, device=device)
        self.step_rows = wp.array(self.rows_np, dtype=wp.int32, device=device)
        self.no_J = wp.zeros((1, 1, 1), dtype=float, device=device)

    @staticmethod
    def _index_array(mask: np.ndarray, device) -> wp.array | None:
        return wp.array(np.flatnonzero(mask).astype(np.int32), dtype=wp.int32, device=device) if mask.any() else None

    class _ActuatorIndex(list):
        """Per tendon, the indices of the supported actuators that drive it."""

        actuator_count = 0
        flags = None

    @classmethod
    def _tendon_actuators(cls, mujoco, tendon_count: int) -> _ActuatorIndex:
        """Collect the TENDON-transmission actuators Steel supports; warn about the ones it cannot model."""
        index = cls._ActuatorIndex([] for _ in range(tendon_count))
        trntype_attr = getattr(mujoco, "actuator_trntype", None)
        if trntype_attr is None or trntype_attr.shape[0] == 0:
            index.flags = np.zeros(1, dtype=np.int32)
            return index
        count = int(trntype_attr.shape[0])
        index.actuator_count = count
        trntype = trntype_attr.numpy()
        trnid = mujoco.actuator_trnid.numpy()
        dyntype = mujoco.actuator_dyntype.numpy()
        gaintype = mujoco.actuator_gaintype.numpy()
        biastype = mujoco.actuator_biastype.numpy()
        labels = list(getattr(mujoco, "actuator_label", None) or [])
        autolimits_attr = getattr(mujoco, "autolimits", None)
        autolimits = bool(autolimits_attr.numpy()[0]) if autolimits_attr is not None else True

        def limited(flag, has_range):
            return flag == 1 or (flag == 2 and autolimits and has_range != 0)

        ctrllimited = mujoco.actuator_ctrllimited.numpy()
        has_ctrlrange = mujoco.actuator_has_ctrlrange.numpy()
        forcelimited = mujoco.actuator_forcelimited.numpy()
        has_forcerange = mujoco.actuator_has_forcerange.numpy()
        flags = np.zeros(count, dtype=np.int32)
        for act in range(count):
            if int(trntype[act]) != _TRN_TENDON:
                continue
            name = labels[act] if act < len(labels) and labels[act] else f"actuator_{act}"
            if int(dyntype[act]) != 0 or int(gaintype[act]) not in (0, 1) or int(biastype[act]) not in (0, 1):
                warnings.warn(
                    f"SolverSteel: tendon actuator '{name}' needs activation dynamics or a non-affine gain/bias, "
                    "which Steel does not model; ignored.",
                    NewtonWarning,
                    stacklevel=4,
                )
                continue
            tendon = int(trnid[act][0])
            if not 0 <= tendon < tendon_count:
                continue
            if limited(int(ctrllimited[act]), int(has_ctrlrange[act])):
                flags[act] |= int(ACTUATOR_CTRL_LIMITED)
            if limited(int(forcelimited[act]), int(has_forcerange[act])):
                flags[act] |= int(ACTUATOR_FORCE_LIMITED)
            index[tendon].append(act)
        index.flags = flags
        return index

    def actuator_inputs(self, model, control) -> TendonActuators:
        """Device actuator tables for one step; ``ctrl`` comes from ``control.mujoco.ctrl`` (zero when absent)."""
        device = model.device
        actuators = TendonActuators()
        actuators.start = self.actuator_start
        actuators.actuator = self.actuator_list
        actuators.flags = self.actuator_flags
        m = getattr(model, "mujoco", None)
        if self.actuator_count and m is not None:
            actuators.gear = m.actuator_gear
            actuators.gainprm = m.actuator_gainprm
            actuators.biasprm = m.actuator_biasprm
            actuators.gaintype = m.actuator_gaintype
            actuators.biastype = m.actuator_biastype
            actuators.ctrlrange = m.actuator_ctrlrange
            actuators.forcerange = m.actuator_forcerange
            ctrl = getattr(getattr(control, "mujoco", None), "ctrl", None) if control is not None else None
            if ctrl is None:
                if getattr(self, "_zero_ctrl", None) is None:
                    self._zero_ctrl = wp.zeros(self.actuator_count, dtype=float, device=device)
                ctrl = self._zero_ctrl
            actuators.ctrl = ctrl
        else:
            if getattr(self, "_empty_actuators", None) is None:
                self._empty_actuators = (
                    wp.zeros(1, dtype=_vec6, device=device),
                    wp.zeros(1, dtype=_vec10, device=device),
                    wp.zeros(1, dtype=wp.int32, device=device),
                    wp.zeros(1, dtype=wp.vec2, device=device),
                    wp.zeros(1, dtype=float, device=device),
                )
            vec6, vec10, ints, vec2, floats = self._empty_actuators
            actuators.gear = vec6
            actuators.gainprm = vec10
            actuators.biasprm = vec10
            actuators.gaintype = ints
            actuators.biastype = ints
            actuators.ctrlrange = vec2
            actuators.forcerange = vec2
            actuators.ctrl = floats
        return actuators

    def _placeholder(self, device, shape, dtype) -> wp.array:
        """A cached zero array standing in for an absent kernel input, so steps allocate nothing."""
        cache = self.__dict__.setdefault("_placeholders", {})
        key = (str(device), shape, dtype)
        if key not in cache:
            cache[key] = wp.zeros(shape, dtype=dtype, device=device)
        return cache[key]

    def model_inputs(self, model) -> list:
        """The ``mujoco:tendon_*`` arrays every tendon kernel reads, in kernel argument order."""
        m = model.mujoco

        def arr(name, dtype):
            attr = getattr(m, name, None)
            if attr is None or attr.shape[0] == 0:
                return self._placeholder(model.device, 1, dtype)
            return attr

        return [
            m.tendon_type,
            arr("tendon_joint_adr", wp.int32),
            arr("tendon_joint_num", wp.int32),
            arr("tendon_joint", wp.int32),
            arr("tendon_coef", wp.float32),
            arr("tendon_wrap_adr", wp.int32),
            arr("tendon_wrap_num", wp.int32),
            arr("tendon_wrap_type", wp.int32),
            arr("tendon_wrap_shape", wp.int32),
            arr("tendon_wrap_sidesite", wp.int32),
            arr("tendon_wrap_prm", wp.float32),
        ]

    def eval_lengths(self, model, joint_q: wp.array, body_q: wp.array, tendon_index: wp.array) -> None:
        """Write the lengths of the tendons in ``tendon_index`` at the given joint/body state into ``length``."""
        wp.launch(
            eval_tendon_lengths,
            dim=tendon_index.shape[0],
            inputs=[
                tendon_index,
                *self.model_inputs(model),
                model.joint_q_start,
                model.joint_qd_start,
                model.shape_body,
                model.shape_transform,
                model.shape_type,
                model.shape_scale,
                model.body_com,
                joint_q,
                body_q,
                self._placeholder(model.device, 1, wp.int32),
                self._placeholder(model.device, 1, wp.spatial_vector),
                self._placeholder(model.device, (1, 1, 1), float),
            ],
            outputs=[self.length],
            device=model.device,
        )

    def resolve_spring_lengths(self, model) -> None:
        """Fix each spring's rest band from ``springlength``; ``-1`` means the length at ``Model.joint_q``."""
        if not self.count:
            return
        self.eval_lengths(
            model, model.joint_q, model.body_q, wp.array(np.arange(self.count, dtype=np.int32), device=model.device)
        )
        length0 = self.length.numpy()
        band = np.empty((self.count, 2), dtype=np.float32)
        for k in range(self.count):
            lo, hi = (float(v) for v in self.springlength_np[k])
            if lo == -1.0:
                lo = hi = float(length0[k])
            elif hi < lo:
                hi = lo
            band[k] = (lo, hi)
        self.spring_length.assign(band)
