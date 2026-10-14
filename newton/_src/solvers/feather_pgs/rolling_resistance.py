# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Experimental Coulomb rolling resistance for free rigid bodies on static or kinematic supports.

After the contact solve, each dynamic body sums the solved normal impulse ``lambda_n`` of its
contacts with static or kinematic shapes, their load-weighted normal ``n`` and the supports'
load-weighted angular velocity. A single-body free rigid articulation then receives an angular
impulse ``h`` in the plane tangent to ``n``, with ``|h| <= rolling_resistance * lambda_n``, that
minimizes its kinetic energy relative to the supports through its world inverse inertia: the
impulse that stops the tangent relative angular velocity when the bound allows, otherwise the
energy minimizer on the bound, found by a fixed 48-step bisection. For finite inputs with positive-definite
inertia, static supports never gain kinetic energy in exact arithmetic; a bound below ``MIN_BUDGET_FRACTION``
of the full-stop impulse, or a non-finite intermediate, leaves the velocity unchanged.

Sums use fixed-point int64 atomics, so they do not depend on contact order while every sum stays
below ``2**63 / SUPPORT_LOAD_SCALE``: about 9.2e6 N s of normal impulse, and 9.2e6 N s rad/s of
impulse-weighted support angular velocity, per body and step.
"""

import warp as wp

from ...sim.enums import BodyFlags
from .kernels import contact_linear_force_from_impulses

wp.set_module_options({"enable_backward": False})

# Fixed-point scale of the accumulated sums [counts per N s] and [counts per N s rad/s].
SUPPORT_LOAD_SCALE = 1.0e12
SUPPORT_LOAD_FIELDS = 7
# Bounds below this fraction of the full-stop impulse leave the velocity unchanged.
MIN_BUDGET_FRACTION = 1.0e-6


@wp.func
def _is_support_body(body: int, body_flags: wp.array[wp.int32]):
    if body < 0:
        return True
    return (body_flags[body] & BodyFlags.KINEMATIC) != 0


@wp.func
def _support_omega(body: int, body_qd: wp.array[wp.spatial_vector]):
    if body < 0:
        return wp.vec3(0.0)
    return wp.spatial_bottom(body_qd[body])


@wp.func
def _add_load(load: wp.array2d[wp.int64], body: int, normal: wp.vec3, support_omega: wp.vec3, normal_impulse: float):
    for axis in range(3):
        wp.atomic_add(load, body, axis, wp.int64(normal[axis] * normal_impulse * SUPPORT_LOAD_SCALE))
        wp.atomic_add(load, body, 3 + axis, wp.int64(support_omega[axis] * normal_impulse * SUPPORT_LOAD_SCALE))
    wp.atomic_add(load, body, 6, wp.int64(normal_impulse * SUPPORT_LOAD_SCALE))


@wp.kernel
def accumulate_support_load(
    contact_count: wp.array[wp.int32],
    contact_shape0: wp.array[wp.int32],
    contact_shape1: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    body_flags: wp.array[wp.int32],
    body_qd: wp.array[wp.spatial_vector],
    contact_normal: wp.array[wp.vec3],
    contact_world: wp.array[wp.int32],
    contact_slot: wp.array[wp.int32],
    contact_path: wp.array[wp.int32],
    world_impulses: wp.array2d[wp.float32],
    mf_impulses: wp.array2d[wp.float32],
    propagation_impulses: wp.array2d[wp.float32],
    world_constraint_count: wp.array[wp.int32],
    mf_constraint_count: wp.array[wp.int32],
    propagation_constraint_count: wp.array[wp.int32],
    world_row_type: wp.array2d[wp.int32],
    world_row_parent: wp.array2d[wp.int32],
    mf_row_type: wp.array2d[wp.int32],
    mf_row_parent: wp.array2d[wp.int32],
    propagation_row_type: wp.array2d[wp.int32],
    propagation_row_parent: wp.array2d[wp.int32],
    # outputs
    load: wp.array2d[wp.int64],
):
    """Accumulate each dynamic body's solved normal impulse [N s] from static or kinematic supports."""
    c = wp.tid()
    if c >= contact_count[0]:
        return
    shape0 = contact_shape0[c]
    shape1 = contact_shape1[c]
    if shape0 < 0 or shape1 < 0:
        return
    body0 = shape_body[shape0]
    body1 = shape_body[shape1]
    support0 = _is_support_body(body0, body_flags)
    support1 = _is_support_body(body1, body_flags)
    if support0 == support1:
        return
    impulse = contact_linear_force_from_impulses(
        c,
        contact_normal,
        contact_world,
        contact_slot,
        contact_path,
        world_impulses,
        mf_impulses,
        propagation_impulses,
        world_constraint_count,
        mf_constraint_count,
        propagation_constraint_count,
        world_row_type,
        world_row_parent,
        mf_row_type,
        mf_row_parent,
        propagation_row_type,
        propagation_row_parent,
        1.0,
    )
    # The contact normal points from shape 0 to shape 1; orient it from the support toward the body.
    normal = contact_normal[c]
    normal_impulse = wp.abs(wp.dot(impulse, normal))
    if normal_impulse <= 0.0:
        return
    if support1:
        _add_load(load, body0, -normal, _support_omega(body1, body_qd), normal_impulse)
    else:
        _add_load(load, body1, normal, _support_omega(body0, body_qd), normal_impulse)


@wp.func
def _tangent_basis(normal: wp.vec3):
    """Return two unit vectors spanning the plane orthogonal to ``normal``."""
    reference = wp.vec3(1.0, 0.0, 0.0)
    if wp.abs(normal[0]) > 0.57735:
        reference = wp.vec3(0.0, 1.0, 0.0)
    t1 = wp.normalize(wp.cross(normal, reference))
    return t1, wp.cross(normal, t1)


@wp.func
def _solve_shifted(a: wp.mat22, shift: float, b: wp.vec2):
    return wp.inverse(a + wp.mat22(shift, 0.0, 0.0, shift)) * b


@wp.kernel
def apply_free_rigid_rolling_resistance(
    articulation_start: wp.array[int],
    joint_child: wp.array[int],
    joint_q_start: wp.array[int],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    body_flags: wp.array[wp.int32],
    body_inv_inertia: wp.array[wp.mat33],
    is_free_rigid: wp.array[wp.int32],
    articulation_root_dof_start: wp.array[int],
    articulation_active: wp.array[int],
    load: wp.array2d[wp.int64],
    rolling_resistance: float,
    joint_q: wp.array[float],
    # in/out
    joint_qd: wp.array[float],
):
    """Apply the bounded tangent angular impulse that minimizes support-relative kinetic energy."""
    art = wp.tid()
    if articulation_active[art] == 0 or is_free_rigid[art] == 0:
        return
    root_joint = articulation_start[art]
    body = joint_child[root_joint]
    if body < 0 or (body_flags[body] & BodyFlags.KINEMATIC) != 0:
        return
    normal_impulse = float(load[body, 6]) / SUPPORT_LOAD_SCALE
    if normal_impulse <= 0.0:
        return
    weighted_normal = wp.vec3(float(load[body, 0]), float(load[body, 1]), float(load[body, 2])) / SUPPORT_LOAD_SCALE
    weighted_length = wp.length(weighted_normal)
    if weighted_length <= 1.0e-6 * normal_impulse:
        return
    normal = weighted_normal / weighted_length
    support_omega = wp.vec3(float(load[body, 3]), float(load[body, 4]), float(load[body, 5])) / (
        SUPPORT_LOAD_SCALE * normal_impulse
    )

    ds = articulation_root_dof_start[art]
    omega = wp.vec3(joint_qd[ds + 3], joint_qd[ds + 4], joint_qd[ds + 5])
    t1, t2 = _tangent_basis(normal)
    relative = omega - support_omega
    # Impulse coordinates (h1, h2) along (t1, t2); the full stop solves a h = b.
    b = wp.vec2(-wp.dot(t1, relative), -wp.dot(t2, relative))
    if b[0] == 0.0 and b[1] == 0.0:
        return

    qs = joint_q_start[root_joint]
    anchor = wp.quat(joint_q[qs + 3], joint_q[qs + 4], joint_q[qs + 5], joint_q[qs + 6])
    orientation = (
        wp.transform_get_rotation(joint_X_p[root_joint])
        * anchor
        * wp.quat_inverse(wp.transform_get_rotation(joint_X_c[root_joint]))
    )
    rotation = wp.quat_to_matrix(orientation)
    inv_inertia = rotation * body_inv_inertia[body] * wp.transpose(rotation)
    r1 = inv_inertia * t1
    r2 = inv_inertia * t2
    a = wp.mat22(wp.dot(t1, r1), wp.dot(t1, r2), wp.dot(t2, r1), wp.dot(t2, r2))
    if wp.determinant(a) <= 0.0:
        return

    budget = rolling_resistance * normal_impulse
    h = wp.inverse(a) * b
    stop_impulse = wp.length(h)
    if stop_impulse > budget:
        # A budget below this fraction of the stopping impulse changes nothing representable; skip it.
        if budget <= MIN_BUDGET_FRACTION * stop_impulse:
            return
        # The bounded energy minimizer is (a + mu I)^-1 b with |h| = budget; |h| falls as mu grows.
        # Bisect on mu / scale with a / scale, so the bracket stays O(1 / MIN_BUDGET_FRACTION).
        scale = 0.5 * (a[0, 0] + a[1, 1])
        unit = a / scale
        low = float(0.0)
        high = wp.length(b) / (scale * budget)
        for _ in range(48):
            mid = 0.5 * (low + high)
            if wp.length(_solve_shifted(unit, mid, b)) > budget * scale:
                low = mid
            else:
                high = mid
        h = _solve_shifted(unit, high, b) / scale
    if not (wp.isfinite(h[0]) and wp.isfinite(h[1])):
        return
    omega = omega + r1 * h[0] + r2 * h[1]
    joint_qd[ds + 3] = omega[0]
    joint_qd[ds + 4] = omega[1]
    joint_qd[ds + 5] = omega[2]
