# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build contact responses in the sparse mass factor's coordinates."""

import warp as wp

from .friction_patches import FrictionPatches
from .kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    contact_restitution_fires,
    contact_tangent_basis,
)


@wp.func
def _contact_jacobian_value(
    motion: wp.spatial_vector,
    point: wp.vec3,
    origin: wp.vec3,
    direction: wp.vec3,
) -> float:
    linear = wp.vec3(motion[0], motion[1], motion[2])
    angular = wp.vec3(motion[3], motion[4], motion[5])
    return wp.dot(direction, linear + wp.cross(angular, point - origin))


@wp.kernel(enable_backward=False)
def populate_sparse_contact_response(
    contact_count: wp.array[int],
    total_num_workers: int,
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_shape0: wp.array[int],
    contact_shape1: wp.array[int],
    contact_thickness0: wp.array[float],
    contact_thickness1: wp.array[float],
    contact_world: wp.array[int],
    contact_slot: wp.array[int],
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    contact_path: wp.array[int],
    contact_slots_needed: wp.array[int],
    art_group_idx: wp.array[int],
    articulation_dof_start: wp.array[int],
    articulation_origin: wp.array[wp.vec3],
    body_dof_mask: wp.array[wp.uint64],
    joint_S_s: wp.array[wp.spatial_vector],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    contact_friction_shared_anchor: int,
    friction_patches: FrictionPatches,
    contact_shared_anchor: int,
    permutation: wp.array[int],
    factor_row_offsets: wp.array[int],
    factor_columns: wp.array[int],
    inverse_factor: wp.array2d[float],
    v_hat: wp.array[float],
    row_dof: wp.array3d[int],
    row_factor: wp.array3d[float],
    row_incident: wp.array2d[float],
    diagonal: wp.array2d[float],
):
    """Project the same contact geometry without constructing dense J or Y."""
    worker, row = wp.tid()
    total_contacts = wp.min(contact_count[0], contact_point0.shape[0])
    for c in range(worker, total_contacts, total_num_workers):
        if contact_path[c] != 0 or contact_slot[c] < 0 or row >= contact_slots_needed[c]:
            continue
        body_a = int(-1)
        body_b = int(-1)
        if contact_shape0[c] >= 0:
            body_a = shape_body[contact_shape0[c]]
        if contact_shape1[c] >= 0:
            body_b = shape_body[contact_shape1[c]]
        art_a = contact_art_a[c]
        art_b = contact_art_b[c]
        art = wp.max(art_a, art_b)
        group = art_group_idx[art]
        world = contact_world[c]
        slot = contact_slot[c] + row
        dof_start = articulation_dof_start[art]
        mask_a = wp.uint64(0)
        mask_b = wp.uint64(0)
        if body_a >= 0 and art_a >= 0:
            mask_a = body_dof_mask[body_a]
        if body_b >= 0 and art_b >= 0:
            mask_b = body_dof_mask[body_b]
        mask = mask_a | mask_b

        normal = -contact_normal[c]
        point_a = contact_point0[c] - contact_thickness0[c] * normal
        point_b = contact_point1[c] + contact_thickness1[c] * normal
        if body_a >= 0:
            point_a = wp.transform_point(body_q[body_a], contact_point0[c]) - contact_thickness0[c] * normal
        if body_b >= 0:
            point_b = wp.transform_point(body_q[body_b], contact_point1[c]) + contact_thickness1[c] * normal
        midpoint = 0.5 * (point_a + point_b)
        direction = normal
        if contact_shared_anchor != 0:
            point_a = midpoint
            point_b = midpoint
        if row != 0:
            tangent0, tangent1 = contact_tangent_basis(normal)
            direction = tangent0
            if row == 2:
                direction = tangent1
            if contact_friction_shared_anchor != 0:
                point_a = midpoint
                point_b = midpoint
            if friction_patches.enabled != 0:
                point_a = friction_patches.point_a[c]
                point_b = friction_patches.point_b[c]

        count = int(0)
        norm = float(0.0)
        incident = float(0.0)
        for node in range(permutation.shape[0]):
            bit = wp.uint64(1) << wp.uint64(node)
            if (mask & bit) == wp.uint64(0):
                continue
            value = float(0.0)
            for entry in range(factor_row_offsets[node], factor_row_offsets[node + 1]):
                column = factor_columns[entry]
                column_bit = wp.uint64(1) << wp.uint64(column)
                if (mask & column_bit) == wp.uint64(0):
                    continue
                motion = joint_S_s[dof_start + permutation[column]]
                jacobian = float(0.0)
                if (mask_a & column_bit) != wp.uint64(0):
                    jacobian += _contact_jacobian_value(motion, point_a, articulation_origin[art], direction)
                if (mask_b & column_bit) != wp.uint64(0):
                    jacobian -= _contact_jacobian_value(motion, point_b, articulation_origin[art], direction)
                value += inverse_factor[group, entry] * jacobian
            row_dof[world, slot, count] = node
            row_factor[world, slot, count] = value
            norm += value * value
            motion = joint_S_s[dof_start + permutation[node]]
            jacobian = float(0.0)
            if (mask_a & bit) != wp.uint64(0):
                jacobian += _contact_jacobian_value(motion, point_a, articulation_origin[art], direction)
            if (mask_b & bit) != wp.uint64(0):
                jacobian -= _contact_jacobian_value(motion, point_b, articulation_origin[art], direction)
            incident += jacobian * v_hat[dof_start + permutation[node]]
            count += 1
        for index in range(count, row_dof.shape[2]):
            row_dof[world, slot, index] = -1
            row_factor[world, slot, index] = 0.0
        row_incident[world, slot] = incident
        diagonal[world, slot] = norm


@wp.func_native(
    """
    const int group = tid >> 5;
    if (group >= group_to_art.shape[0]) return;
#if defined(__CUDA_ARCH__)
    constexpr unsigned MASK = 0xffffffffu;
    const int lane = tid & 31;
    const int width = 32;
#else
    if ((tid & 31) != 0) return;
    const int lane = 0;
    const int width = 1;
#endif
    const auto bit_count = [](unsigned long long mask) {
#if defined(__CUDA_ARCH__)
        return __popcll(mask);
#else
        int count = 0;
        while (mask != 0) { mask &= mask - 1; ++count; }
        return count;
#endif
    };
    const int art = group_to_art.data[group];
    const int world = art_to_world.data[art];
    const int start = articulation_dof_start.data[art];
    const int dofs = inverse_permutation.shape[0];
    const int capacity = row_type.shape[1];
    const int support_capacity = row_dof.shape[2];
    const int factor_base = group * inverse_factor.shape[1];
    for (int base = 0; base < 2 * dofs; base += width) {
        const int candidate = base + lane;
        const int local_dof = candidate >> 1;
        const int side = candidate & 1;
        const int dof = start + local_dof;
        const int q_index = candidate < 2 * dofs ? limit_q_index.data[dof] : -1;
        float gap = 0.0f;
        int active = 0;
        if (q_index >= 0) {
            const float position = joint_q.data[q_index];
            const float bound = side == 0 ? joint_limit_lower.data[dof] : joint_limit_upper.data[dof];
            gap = side == 0 ? position - bound : bound - position;
            active = wp::isfinite(bound)
                && (side == 0 ? position <= bound + activation_gap : position >= bound - activation_gap);
        }
#if defined(__CUDA_ARCH__)
        const unsigned active_mask = __ballot_sync(MASK, active != 0);
        const int active_count = __popc(active_mask);
        int first_slot = 0;
        if (lane == 0 && active_count != 0)
            first_slot = atomicAdd(&slot_counter.data[world], active_count);
        first_slot = __shfl_sync(MASK, first_slot, 0);
#else
        const unsigned active_mask = active != 0 ? 1u : 0u;
        const int first_slot = slot_counter.data[world];
        slot_counter.data[world] += active;
#endif
        // Visit the ballot in lane order, matching the existing lower/upper DOF order.
        unsigned pending = active_mask;
        int rank = 0;
        while (pending != 0) {
            const int row = first_slot + rank;
            // The reservation above still counts every over-capacity row.
            if (row >= capacity) break;
#if defined(__CUDA_ARCH__)
            const int source = __ffs(pending) - 1;
            const int selected_dof = __shfl_sync(MASK, local_dof, source);
            const int selected_side = __shfl_sync(MASK, side, source);
            const float selected_gap = __shfl_sync(MASK, gap, source);
#else
            const int selected_dof = local_dof;
            const int selected_side = side;
            const float selected_gap = gap;
#endif
            const float sign = selected_side == 0 ? 1.0f : -1.0f;
            const int column = inverse_permutation.data[selected_dof];
            const unsigned long long support_mask = dof_mask.data[selected_dof];
            const int row_offset = world * capacity + row;
            const int response_offset = row_offset * support_capacity;
            float norm = 0.0f;
            for (int node = lane; node < dofs; node += width) {
                const unsigned long long bit = 1ull << node;
                if ((support_mask & bit) == 0) continue;
                const int index = bit_count(support_mask & (bit - 1ull));
                const int entry = factor_lookup.data[node * dofs + column];
                const float value = entry >= 0 ? sign * inverse_factor.data[factor_base + entry] : 0.0f;
                row_dof.data[response_offset + index] = node;
                row_factor.data[response_offset + index] = value;
                norm += value * value;
            }
            const int support_count = bit_count(support_mask);
            for (int index = support_count + lane; index < support_capacity; index += width) {
                row_dof.data[response_offset + index] = -1;
                row_factor.data[response_offset + index] = 0.0f;
            }
#if defined(__CUDA_ARCH__)
            for (int shift = 16; shift > 0; shift >>= 1) norm += __shfl_down_sync(MASK, norm, shift);
#endif
            if (lane == 0) {
                row_incident.data[row_offset] = sign * v_hat.data[start + selected_dof];
                diagonal.data[row_offset] = norm;
                row_type.data[row_offset] = $LIMIT_TYPE;
                row_parent.data[row_offset] = -1;
                row_mu.data[row_offset] = 0.0f;
                row_beta.data[row_offset] = pgs_beta;
                row_cfm.data[row_offset] = pgs_cfm;
                phi.data[row_offset] = selected_gap;
                target_velocity.data[row_offset] = 0.0f;
            }
            pending &= pending - 1;
            ++rank;
        }
    }
""".replace("$LIMIT_TYPE", str(PGS_CONSTRAINT_TYPE_JOINT_LIMIT))
)
def _build_sparse_joint_limit_rows(
    tid: int,
    group_to_art: wp.array[int],
    art_to_world: wp.array[int],
    articulation_dof_start: wp.array[int],
    limit_q_index: wp.array[int],
    joint_limit_lower: wp.array[float],
    joint_limit_upper: wp.array[float],
    joint_q: wp.array[float],
    activation_gap: float,
    pgs_beta: float,
    pgs_cfm: float,
    dof_mask: wp.array[wp.uint64],
    inverse_permutation: wp.array[int],
    factor_lookup: wp.array2d[int],
    inverse_factor: wp.array2d[float],
    v_hat: wp.array[float],
    slot_counter: wp.array[int],
    row_type: wp.array2d[int],
    row_parent: wp.array2d[int],
    row_mu: wp.array2d[float],
    row_beta: wp.array2d[float],
    row_cfm: wp.array2d[float],
    phi: wp.array2d[float],
    target_velocity: wp.array2d[float],
    row_dof: wp.array3d[int],
    row_factor: wp.array3d[float],
    row_incident: wp.array2d[float],
    diagonal: wp.array2d[float],
): ...


@wp.kernel(enable_backward=False)
def build_sparse_joint_limit_rows(
    group_to_art: wp.array[int],
    art_to_world: wp.array[int],
    articulation_dof_start: wp.array[int],
    limit_q_index: wp.array[int],
    joint_limit_lower: wp.array[float],
    joint_limit_upper: wp.array[float],
    joint_q: wp.array[float],
    activation_gap: float,
    pgs_beta: float,
    pgs_cfm: float,
    dof_mask: wp.array[wp.uint64],
    inverse_permutation: wp.array[int],
    factor_lookup: wp.array2d[int],
    inverse_factor: wp.array2d[float],
    v_hat: wp.array[float],
    slot_counter: wp.array[int],
    row_type: wp.array2d[int],
    row_parent: wp.array2d[int],
    row_mu: wp.array2d[float],
    row_beta: wp.array2d[float],
    row_cfm: wp.array2d[float],
    phi: wp.array2d[float],
    target_velocity: wp.array2d[float],
    row_dof: wp.array3d[int],
    row_factor: wp.array3d[float],
    row_incident: wp.array2d[float],
    diagonal: wp.array2d[float],
):
    """Build ordered limit rows and sparse responses with one warp per articulation.

    Launch ``32 * group_count`` threads in blocks divisible by 32. Active row
    reservations include over-capacity demand; writes stay within row capacity.
    The admitted topology has one responsive articulation per world.
    """
    _build_sparse_joint_limit_rows(
        wp.tid(),
        group_to_art,
        art_to_world,
        articulation_dof_start,
        limit_q_index,
        joint_limit_lower,
        joint_limit_upper,
        joint_q,
        activation_gap,
        pgs_beta,
        pgs_cfm,
        dof_mask,
        inverse_permutation,
        factor_lookup,
        inverse_factor,
        v_hat,
        slot_counter,
        row_type,
        row_parent,
        row_mu,
        row_beta,
        row_cfm,
        phi,
        target_velocity,
        row_dof,
        row_factor,
        row_incident,
        diagonal,
    )


@wp.kernel(enable_backward=False)
def apply_sparse_contact_restitution(
    constraint_count: wp.array[int],
    phi: wp.array2d[float],
    row_type: wp.array2d[int],
    target_velocity: wp.array2d[float],
    row_restitution: wp.array2d[float],
    row_incident: wp.array2d[float],
    dt: float,
    restitution_velocity_threshold: float,
    rhs: wp.array2d[float],
):
    """Use the already projected incident velocity for the unchanged impact law."""
    world, row = wp.tid()
    if row >= constraint_count[world] or row_type[world, row] != PGS_CONSTRAINT_TYPE_CONTACT:
        return
    restitution = row_restitution[world, row]
    relative_incident = row_incident[world, row] - target_velocity[world, row]
    if restitution > 0.0 and contact_restitution_fires(
        phi[world, row], relative_incident, dt, restitution_velocity_threshold
    ):
        rhs[world, row] = -target_velocity[world, row] + restitution * relative_incident


@wp.kernel(enable_backward=False)
def apply_sparse_factor_velocity(
    group_to_art: wp.array[int],
    art_to_world: wp.array[int],
    articulation_dof_start: wp.array[int],
    permutation: wp.array[int],
    factor_lookup: wp.array2d[int],
    inverse_factor: wp.array2d[float],
    factor_velocity_delta: wp.array2d[float],
    v_hat: wp.array[float],
    v_out: wp.array[float],
):
    """Decode the factor-coordinate impulse once per world after its sweeps."""
    group, column = wp.tid()
    art = group_to_art[group]
    world = art_to_world[art]
    delta = float(0.0)
    for row in range(column, permutation.shape[0]):
        entry = factor_lookup[row, column]
        if entry >= 0:
            delta += inverse_factor[group, entry] * factor_velocity_delta[world, row]
    dof = articulation_dof_start[art] + permutation[column]
    v_out[dof] = v_hat[dof] + delta
