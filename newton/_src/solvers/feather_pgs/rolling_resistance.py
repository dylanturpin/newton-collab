# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Experimental rolling resistance for free rigid bodies resting on static or kinematic geometry.

After the contact solve, each free root body sums the solved normal impulse of its contacts with
static or kinematic shapes. Its angular velocity about axes perpendicular to the load-weighted
contact normal is then reduced by at most ``rolling_resistance * lambda_n`` of angular impulse
(Coulomb rolling resistance, a lever arm in meters), and optionally by a viscous ``rolling_damping``
rate [1/s]. Linear velocity is left to the next step's friction rows. Sums use fixed-point int64
atomics so the result is independent of contact order.
"""

import warp as wp

from ...sim.enums import BodyFlags
from .kernels import contact_linear_force_from_impulses

wp.set_module_options({"enable_backward": False})

ROLLING_LOAD_SCALE = 1.0e12


@wp.func
def _is_support_body(body: int, body_flags: wp.array[wp.int32]):
    if body < 0:
        return True
    return (body_flags[body] & BodyFlags.KINEMATIC) != 0


@wp.func
def _add_load(load: wp.array2d[wp.int64], body: int, weighted_normal: wp.vec3, normal_impulse: float):
    wp.atomic_add(load, body, 0, wp.int64(weighted_normal[0] * ROLLING_LOAD_SCALE))
    wp.atomic_add(load, body, 1, wp.int64(weighted_normal[1] * ROLLING_LOAD_SCALE))
    wp.atomic_add(load, body, 2, wp.int64(weighted_normal[2] * ROLLING_LOAD_SCALE))
    wp.atomic_add(load, body, 3, wp.int64(normal_impulse * ROLLING_LOAD_SCALE))


@wp.kernel
def accumulate_support_load(
    contact_count: wp.array[wp.int32],
    contact_shape0: wp.array[wp.int32],
    contact_shape1: wp.array[wp.int32],
    shape_body: wp.array[wp.int32],
    body_flags: wp.array[wp.int32],
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
    """Sum each dynamic body's normal impulse [N s] from static/kinematic partners, and its normal-weighted sum."""
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
    force = contact_linear_force_from_impulses(
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
    # The normal points from shape 0 to shape 1; orient it from the support toward the body.
    normal = contact_normal[c]
    normal_impulse = wp.abs(wp.dot(force, normal))
    if normal_impulse <= 0.0:
        return
    if support1:
        _add_load(load, body0, -normal * normal_impulse, normal_impulse)
    else:
        _add_load(load, body1, normal * normal_impulse, normal_impulse)


@wp.kernel
def apply_free_root_rolling_resistance(
    articulation_start: wp.array[int],
    joint_child: wp.array[int],
    joint_q_start: wp.array[int],
    joint_X_c: wp.array[wp.transform],
    body_flags: wp.array[wp.int32],
    body_inertia: wp.array[wp.mat33],
    articulation_root_is_free: wp.array[int],
    articulation_root_dof_start: wp.array[int],
    articulation_active: wp.array[int],
    load: wp.array2d[wp.int64],
    rolling_resistance: float,
    rolling_damping: float,
    dt: float,
    joint_q: wp.array[float],
    # in/out
    joint_qd: wp.array[float],
):
    """Remove up to ``rolling_resistance * lambda_n`` of angular momentum about the support's tangent axes."""
    art = wp.tid()
    if articulation_active[art] == 0 or articulation_root_is_free[art] == 0:
        return
    root_joint = articulation_start[art]
    body = joint_child[root_joint]
    if body < 0 or (body_flags[body] & BodyFlags.KINEMATIC) != 0:
        return
    normal_impulse = float(load[body, 3]) / ROLLING_LOAD_SCALE
    if normal_impulse <= 0.0:
        return
    weighted = wp.vec3(float(load[body, 0]), float(load[body, 1]), float(load[body, 2])) / ROLLING_LOAD_SCALE
    weighted_length = wp.length(weighted)
    if weighted_length <= 1.0e-6 * normal_impulse:
        return
    normal = weighted / weighted_length

    ds = articulation_root_dof_start[art]
    omega = wp.vec3(joint_qd[ds + 3], joint_qd[ds + 4], joint_qd[ds + 5])
    omega_roll = omega - wp.dot(omega, normal) * normal
    if wp.length(omega_roll) <= 0.0:
        return

    keep = float(1.0)
    if rolling_damping > 0.0:
        keep = wp.max(0.0, 1.0 - rolling_damping * dt)
    if rolling_resistance > 0.0:
        qs = joint_q_start[root_joint]
        anchor = wp.quat(joint_q[qs + 3], joint_q[qs + 4], joint_q[qs + 5], joint_q[qs + 6])
        rotation = wp.quat_to_matrix(anchor * wp.quat_inverse(wp.transform_get_rotation(joint_X_c[root_joint])))
        inertia = rotation * body_inertia[body] * wp.transpose(rotation)
        momentum = inertia * (omega_roll * keep)
        momentum = momentum - wp.dot(momentum, normal) * normal
        magnitude = wp.length(momentum)
        budget = rolling_resistance * normal_impulse
        if magnitude <= budget:
            keep = 0.0
        else:
            keep = keep * (1.0 - budget / magnitude)
    omega = omega - omega_roll * (1.0 - keep)
    joint_qd[ds + 3] = omega[0]
    joint_qd[ds + 4] = omega[1]
    joint_qd[ds + 5] = omega[2]
