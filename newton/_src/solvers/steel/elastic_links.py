# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reduced elastic link coordinates in Steel.

An ELASTIC tree joint carries a rigid frame (FREE or a D6 axis subset) followed by modal
coordinates. Modal motion-subspace columns are zero, so stock CRBA leaves the modal rows and
columns of ``H`` empty; these kernels fill them from the basis' floating-frame inertia
integrals, add the modal stiffness implicitly, and extend contact rows with the modes.
"""

import warp as wp

from .friction_patches import contact_tangent_basis


@wp.func
def _mode_coupling_wrench(R: wp.mat33, lever: wp.vec3, S: wp.vec3, G: wp.vec3):
    """Spatial modal momentum coupling about the articulation origin."""
    RS = R * S
    return wp.spatial_vector(RS, wp.cross(lever, RS) - R * G)


@wp.kernel(enable_backward=False)
def elastic_link_bias(
    elastic_body: wp.array[wp.int32],
    elastic_art: wp.array[wp.int32],
    elastic_mode_q0: wp.array[wp.int32],
    elastic_mode_qd0: wp.array[wp.int32],
    elastic_mode_start: wp.array[wp.int32],
    elastic_mode_count: wp.array[wp.int32],
    max_modes: int,
    coupling_linear: wp.array[wp.vec3],
    coupling_angular: wp.array[wp.vec3],
    coupling_centrifugal: wp.array[wp.mat33],
    coupling_coriolis: wp.array[wp.vec3],
    articulation_origin: wp.array[wp.vec3],
    articulation_active: wp.array[int],
    body_disable_gravity: wp.array[bool],
    body_world: wp.array[int],
    gravity: wp.array[wp.vec3],
    joint_qd: wp.array[float],
    body_q: wp.array[wp.transform],
    body_v_s: wp.array[wp.spatial_vector],
    body_a_s: wp.array[wp.spatial_vector],
    # outputs
    body_f_s: wp.array[wp.spatial_vector],
    mode_bias: wp.array[float],
):
    """Velocity-product and gravity terms of the modal equations, and the modes' frame wrench."""
    e = wp.tid()
    art = elastic_art[e]
    if art < 0 or articulation_active[art] == 0:
        return
    k = elastic_body[e]
    X = body_q[k]
    R = wp.quat_to_matrix(wp.transform_get_rotation(X))
    lever = wp.transform_get_translation(X) - articulation_origin[art]
    v = body_v_s[k]
    a = body_a_s[k]
    w = wp.spatial_bottom(v)
    v_O = wp.spatial_top(v) + wp.cross(w, lever)
    w_b = wp.transpose(R) * w
    w_skew = wp.skew(w_b)
    w_skew_sq = w_skew * w_skew
    g_b = wp.vec3()
    if not body_disable_gravity[k]:
        g_b = wp.transpose(R) * gravity[wp.max(body_world[k], 0)]

    start = elastic_mode_start[e]
    count = elastic_mode_count[e]
    qd0 = elastic_mode_qd0[e]
    s_qd = wp.vec3()
    g_qd = wp.vec3()
    for i in range(count):
        eta_dot = joint_qd[qd0 + i]
        s_qd += R * coupling_linear[start + i] * eta_dot
        g_qd += R * coupling_angular[start + i] * eta_dot

    cor_start = e * max_modes * max_modes
    for i in range(count):
        mode = start + i
        gamma = _mode_coupling_wrench(R, lever, coupling_linear[mode], coupling_angular[mode])
        cor = wp.vec3()
        for j in range(count):
            cor += coupling_coriolis[cor_start + i * max_modes + j] * joint_qd[qd0 + j]
        bias = wp.dot(gamma, a)
        bias += wp.dot(wp.spatial_top(gamma), wp.cross(w, v_O))
        bias += wp.ddot(w_skew_sq, coupling_centrifugal[mode])
        bias += 2.0 * wp.dot(w_b, cor)
        bias -= wp.dot(coupling_linear[mode], g_b)
        mode_bias[mode] = bias

    w_cross_s = wp.cross(w, s_qd)
    torque = wp.cross(v_O, s_qd) + wp.cross(lever, w_cross_s) - wp.cross(w, g_qd)
    body_f_s[k] = body_f_s[k] + wp.spatial_vector(w_cross_s, torque)


@wp.kernel(enable_backward=False)
def elastic_link_tau(
    elastic_art: wp.array[wp.int32],
    elastic_mode_q0: wp.array[wp.int32],
    elastic_mode_qd0: wp.array[wp.int32],
    elastic_mode_start: wp.array[wp.int32],
    elastic_mode_count: wp.array[wp.int32],
    mode_stiffness: wp.array[float],
    mode_damping: wp.array[float],
    mode_bias: wp.array[float],
    articulation_active: wp.array[int],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    dt: float,
    # in/out
    tau: wp.array[float],
):
    """Add implicit modal spring/damper forces and the modal bias to the generalized force."""
    e = wp.tid()
    art = elastic_art[e]
    if art < 0 or articulation_active[art] == 0:
        return
    start = elastic_mode_start[e]
    q0 = elastic_mode_q0[e]
    qd0 = elastic_mode_qd0[e]
    for i in range(elastic_mode_count[e]):
        k = mode_stiffness[start + i]
        c = mode_damping[start + i]
        eta = joint_q[q0 + i]
        eta_dot = joint_qd[qd0 + i]
        tau[qd0 + i] = tau[qd0 + i] - k * eta - (c + dt * k) * eta_dot - mode_bias[start + i]


@wp.kernel(enable_backward=False)
def elastic_link_mass(
    group_elastic: wp.array[wp.int32],
    group_index: wp.array[wp.int32],
    elastic_body: wp.array[wp.int32],
    elastic_joint: wp.array[wp.int32],
    elastic_art: wp.array[wp.int32],
    elastic_frame_dofs: wp.array[wp.int32],
    elastic_mode_qd0: wp.array[wp.int32],
    elastic_mode_start: wp.array[wp.int32],
    elastic_mode_count: wp.array[wp.int32],
    mode_mass: wp.array[float],
    mode_stiffness: wp.array[float],
    mode_damping: wp.array[float],
    coupling_linear: wp.array[wp.vec3],
    coupling_angular: wp.array[wp.vec3],
    articulation_dof_start: wp.array[int],
    articulation_origin: wp.array[wp.vec3],
    mass_update_mask: wp.array[int],
    joint_ancestor: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_S_s: wp.array[wp.spatial_vector],
    body_q: wp.array[wp.transform],
    dt: float,
    n_dofs: int,
    # in/out
    H_group: wp.array3d[float],
):
    """Fill the modal rows/columns of one size group's H after CRBA."""
    t = wp.tid()
    e = group_elastic[t]
    art = elastic_art[e]
    if mass_update_mask[art] == 0:
        return
    g = group_index[t]
    k = elastic_body[e]
    X = body_q[k]
    R = wp.quat_to_matrix(wp.transform_get_rotation(X))
    lever = wp.transform_get_translation(X) - articulation_origin[art]
    dof_start = articulation_dof_start[art]
    start = elastic_mode_start[e]
    qd0 = elastic_mode_qd0[e]
    joint = elastic_joint[e]
    for i in range(elastic_mode_count[e]):
        mode = start + i
        gamma = _mode_coupling_wrench(R, lever, coupling_linear[mode], coupling_angular[mode])
        col = qd0 + i - dof_start
        H_group[g, col, col] = mode_mass[mode] + dt * mode_damping[mode] + dt * dt * mode_stiffness[mode]
        curr = joint
        while curr >= 0:
            d_begin = joint_qd_start[curr]
            d_end = joint_qd_start[curr + 1]
            if curr == joint:
                d_end = d_begin + elastic_frame_dofs[e]
            for d in range(d_begin, d_end):
                row = d - dof_start
                if row >= 0 and row < n_dofs:
                    val = wp.dot(joint_S_s[d], gamma)
                    H_group[g, row, col] = val
                    H_group[g, col, row] = val
            curr = joint_ancestor[curr]


@wp.kernel(enable_backward=False)
def elastic_contact_points(
    contact_count: wp.array[int],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_sample0: wp.array[wp.int32],
    contact_sample1: wp.array[wp.int32],
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    shape_body: wp.array[int],
    body_elastic_index: wp.array[wp.int32],
    elastic_mode_q0: wp.array[wp.int32],
    elastic_mode_count: wp.array[wp.int32],
    max_modes: int,
    vertex_phi: wp.array[wp.vec3],
    joint_q: wp.array[float],
    # outputs
    point0_out: wp.array[wp.vec3],
    point1_out: wp.array[wp.vec3],
):
    """Write body-local contact points with elastic samples displaced by the current modes."""
    c = wp.tid()
    if c >= wp.min(contact_count[0], contact_point0.shape[0]):
        return
    point0_out[c] = _deformed_point(
        contact_point0[c],
        contact_sample0[c],
        contact_shape0[c],
        shape_body,
        body_elastic_index,
        elastic_mode_q0,
        elastic_mode_count,
        max_modes,
        vertex_phi,
        joint_q,
    )
    point1_out[c] = _deformed_point(
        contact_point1[c],
        contact_sample1[c],
        contact_shape1[c],
        shape_body,
        body_elastic_index,
        elastic_mode_q0,
        elastic_mode_count,
        max_modes,
        vertex_phi,
        joint_q,
    )


@wp.kernel(enable_backward=False)
def elastic_contact_jacobian(
    contact_count: wp.array[int],
    contact_normal: wp.array[wp.vec3],
    contact_sample0: wp.array[wp.int32],
    contact_sample1: wp.array[wp.int32],
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    contact_slot: wp.array[int],
    contact_slots_needed: wp.array[int],
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    contact_path: wp.array[int],
    target_size: int,
    articulation_response_dof_count: wp.array[int],
    art_group_idx: wp.array[int],
    art_dof_start: wp.array[int],
    shape_body: wp.array[int],
    body_elastic_index: wp.array[wp.int32],
    elastic_mode_qd0: wp.array[wp.int32],
    elastic_mode_count: wp.array[wp.int32],
    max_modes: int,
    vertex_phi: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    # in/out
    J_group: wp.array3d[float],
):
    """Add modal columns ``n · R φ`` to the dense normal and friction rows of elastic contacts."""
    c = wp.tid()
    if c >= wp.min(contact_count[0], contact_normal.shape[0]):
        return
    if contact_path[c] != 0 or contact_slot[c] < 0:
        return
    for side in range(2):
        sample = contact_sample0[c]
        shape = contact_shape0[c]
        art = contact_art_a[c]
        sign = 1.0
        if side == 1:
            sample = contact_sample1[c]
            shape = contact_shape1[c]
            art = contact_art_b[c]
            sign = -1.0
        if sample < 0 or shape < 0 or art < 0:
            continue
        if articulation_response_dof_count[art] != target_size:
            continue
        body = shape_body[shape]
        e = body_elastic_index[body]
        if e < 0:
            continue
        slot = contact_slot[c]
        normal = -contact_normal[c]
        t0, t1 = contact_tangent_basis(normal)
        # A contact owns its tangent rows exactly when it reserved them (patch non-anchors and gap contacts do not).
        has_friction = contact_slots_needed[c] >= 3
        R = wp.quat_to_matrix(wp.transform_get_rotation(body_q[body]))
        g = art_group_idx[art]
        qd0 = elastic_mode_qd0[e]
        for i in range(elastic_mode_count[e]):
            col = qd0 + i - art_dof_start[art]
            u = R * vertex_phi[sample * max_modes + i]
            J_group[g, slot, col] += sign * wp.dot(normal, u)
            if has_friction:
                J_group[g, slot + 1, col] += sign * wp.dot(t0, u)
                J_group[g, slot + 2, col] += sign * wp.dot(t1, u)


@wp.func
def _deformed_point(
    point: wp.vec3,
    sample: int,
    shape: int,
    shape_body: wp.array[int],
    body_elastic_index: wp.array[wp.int32],
    elastic_mode_q0: wp.array[wp.int32],
    elastic_mode_count: wp.array[wp.int32],
    max_modes: int,
    vertex_phi: wp.array[wp.vec3],
    joint_q: wp.array[float],
):
    if sample < 0 or shape < 0:
        return point
    body = shape_body[shape]
    if body < 0:
        return point
    e = body_elastic_index[body]
    if e < 0:
        return point
    q0 = elastic_mode_q0[e]
    out = point
    for i in range(elastic_mode_count[e]):
        out += vertex_phi[sample * max_modes + i] * joint_q[q0 + i]
    return out
