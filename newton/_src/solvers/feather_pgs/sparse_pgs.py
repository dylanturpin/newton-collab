# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Serial PGS over ancestor-sparse, mass-whitened constraint rows."""

from functools import cache

import warp as wp

from .friction import FRICTION_PAIR_CUDA
from .kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
)


@cache
def _get_pgs_solve_sparse_kernel(max_constraints: int, max_world_dofs: int, max_row_dofs: int) -> wp.Kernel:
    """Build a CUDA sweep using ``Z = J P^T L^-T`` and a factor-velocity delta.

    Rows contain unique factor-coordinate indices, padded with -1. Both
    tangent rows of a contact must have identical, ordered support. The caller
    supplies ``row_incident = J v_hat`` and decodes the output once as
    ``v_out = v_hat + P^T L^-T factor_velocity_delta``. No physical response rows or
    Delassus matrix are needed.

    This uses the existing interleaved row order and paired friction law,
    including linked patch normal loads. The caller admits only augmented
    drives, immediate response, and no warm start, torsion or velocity-limit
    rows. Seeded patch impulses retain the ordinary solver's delta semantics.

    Launch with ``ceil(world_count / 2)`` tiles and 64 threads per tile.
    """
    M, D, S = int(max_constraints), int(max_world_dofs), int(max_row_dofs)
    if M <= 0 or D <= 0 or S <= 0 or S > D:
        raise ValueError("sparse PGS requires positive capacities and max_row_dofs <= max_world_dofs")
    W = 2
    Q = (S + 31) // 32
    contact_type = int(PGS_CONSTRAINT_TYPE_CONTACT)
    friction_type = int(PGS_CONSTRAINT_TYPE_FRICTION)
    limit_type = int(PGS_CONSTRAINT_TYPE_JOINT_LIMIT)
    snippet = (
        "#if defined(__CUDA_ARCH__)\n"
        + FRICTION_PAIR_CUDA
        + f"""
    __shared__ float delta_storage[{W * D}];
    __shared__ float impulse_storage[{W * M}];
    __shared__ float rhs_storage[{W * M}];
    __shared__ float diagonal_storage[{W * M}];
    __shared__ unsigned char type_storage[{W * M}];
    constexpr unsigned MASK = 0xffffffffu;
    const int lane = threadIdx.x & 31;
    const int slot = threadIdx.x >> 5;
    const int offset = world * {M};
    const int row_base = offset * {S};
    const int count = min(world_constraint_count.data[world], {M});
    float* delta = delta_storage + slot * {D};
    float* impulse = impulse_storage + slot * {M};
    float* rhs = rhs_storage + slot * {M};
    float* diagonal = diagonal_storage + slot * {M};
    unsigned char* type = type_storage + slot * {M};

    for (int d = lane; d < {D}; d += 32) delta[d] = 0.0f;
    for (int i = lane; i < count; i += 32) {{
        impulse[i] = world_impulses.data[offset + i];
        rhs[i] = rhs_bias.data[offset + i] + row_incident.data[offset + i];
        diagonal[i] = world_diag.data[offset + i];
        type[i] = static_cast<unsigned char>(world_row_type.data[offset + i]);
    }}
    __syncwarp(MASK);

    for (int iteration = 0; iteration < iterations; ++iteration) {{
        const int global_iteration = iteration_offset + iteration;
        int changed = 0;
        for (int row = 0; row < count; ++row) {{
            const int row_type = static_cast<int>(type[row]);
            const bool friction = row_type == {friction_type};
            if (friction && global_iteration < friction_start_iteration) {{
                if (lane == 0) impulse[row] = 0.0f;
                __syncwarp(MASK);
                continue;
            }}
            int parent = -1;
            int sibling = -1;
            if (friction) {{
                parent = world_row_parent.data[offset + row];
                if (row != parent + 1) continue;
                sibling = parent + 2;
            }}
            const float denominator = diagonal[row];
            if (!friction && denominator <= 0.0f) continue;

            int indices[{Q}];
            float values[{Q}];
            float sibling_values[{Q}];
            float dot = 0.0f;
            float sibling_dot = 0.0f;
            float cross = 0.0f;
            #pragma unroll
            for (int q = 0; q < {Q}; ++q) {{
                const int entry = lane + q * 32;
                const int index = entry < {S} ? row_dof.data[row_base + row * {S} + entry] : -1;
                const float value = index >= 0 ? row_factor.data[row_base + row * {S} + entry] : 0.0f;
                const float sibling_value = friction && index >= 0
                    ? row_factor.data[row_base + sibling * {S} + entry] : 0.0f;
                indices[q] = index;
                values[q] = value;
                sibling_values[q] = sibling_value;
                if (index >= 0) {{
                    const float velocity = delta[index];
                    dot += value * velocity;
                    sibling_dot += sibling_value * velocity;
                    cross += value * sibling_value;
                }}
            }}
            for (int shift = 16; shift > 0; shift >>= 1) {{
                dot += __shfl_down_sync(MASK, dot, shift);
                sibling_dot += __shfl_down_sync(MASK, sibling_dot, shift);
                cross += __shfl_down_sync(MASK, cross, shift);
            }}
            const float residual = __shfl_sync(MASK, dot, 0) + rhs[row];
            const float old_impulse = impulse[row];
            float new_impulse = old_impulse;
            float sibling_change = 0.0f;
            if (friction) {{
                float normal_load = impulse[parent];
                for (int next = world_row_parent.data[offset + parent]; next >= 0 && next != parent;
                     next = world_row_parent.data[offset + next])
                    normal_load += impulse[next];
                const float radius = fmaxf(world_row_mu.data[offset + row] * normal_load, 0.0f);
                const float old_sibling = impulse[sibling];
                float2 pair = make_float2(0.0f, 0.0f);
                if (radius > 0.0f) {{
                    const float sibling_residual = __shfl_sync(MASK, sibling_dot, 0) + rhs[sibling];
                    pair = friction_pair_candidate(denominator, __shfl_sync(MASK, cross, 0),
                        diagonal[sibling], residual, sibling_residual, old_impulse, old_sibling, radius, omega);
                }}
                const float magnitude = sqrtf(pair.x * pair.x + pair.y * pair.y);
                const float scale = magnitude > radius ? radius / magnitude : 1.0f;
                new_impulse = pair.x * scale;
                const float new_sibling = pair.y * scale;
                sibling_change = new_sibling - old_sibling;
                if (lane == 0) impulse[sibling] = new_sibling;
            }} else {{
                new_impulse = old_impulse - omega * residual / denominator;
                if (row_type == {contact_type} || row_type == {limit_type})
                    new_impulse = fmaxf(new_impulse, 0.0f);
            }}
            const float change = new_impulse - old_impulse;
            if (lane == 0) impulse[row] = new_impulse;
            if (change != 0.0f || sibling_change != 0.0f) {{
                changed = 1;
                #pragma unroll
                for (int q = 0; q < {Q}; ++q) {{
                    if (indices[q] >= 0) {{
                        // Match the existing paired-tangent response order.
                        float value = delta[indices[q]];
                        if (sibling_change != 0.0f) value += sibling_values[q] * sibling_change;
                        if (change != 0.0f) value += values[q] * change;
                        delta[indices[q]] = value;
                    }}
                }}
            }}
            __syncwarp(MASK);
        }}
        if (global_iteration >= friction_start_iteration && __ballot_sync(MASK, changed != 0) == 0u) break;
    }}
    for (int d = lane; d < {D}; d += 32) factor_velocity_delta.data[world * {D} + d] = delta[d];
    for (int i = lane; i < count; i += 32) world_impulses.data[offset + i] = impulse[i];
#endif
"""
    )

    @wp.func_native(snippet)
    def solve_native(
        world: int,
        world_constraint_count: wp.array[int],
        rhs_bias: wp.array2d[float],
        world_diag: wp.array2d[float],
        world_impulses: wp.array2d[float],
        row_dof: wp.array3d[int],
        row_factor: wp.array3d[float],
        row_incident: wp.array2d[float],
        world_row_type: wp.array2d[int],
        world_row_parent: wp.array2d[int],
        world_row_mu: wp.array2d[float],
        iterations: int,
        omega: float,
        friction_start_iteration: int,
        iteration_offset: int,
        factor_velocity_delta: wp.array2d[float],
    ): ...

    def solve(
        world_count: int,
        world_constraint_count: wp.array[int],
        rhs_bias: wp.array2d[float],
        world_diag: wp.array2d[float],
        world_impulses: wp.array2d[float],
        row_dof: wp.array3d[int],
        row_factor: wp.array3d[float],
        row_incident: wp.array2d[float],
        world_row_type: wp.array2d[int],
        world_row_parent: wp.array2d[int],
        world_row_mu: wp.array2d[float],
        iterations: int,
        omega: float,
        friction_start_iteration: int,
        iteration_offset: int,
        factor_velocity_delta: wp.array2d[float],
    ):
        block, lane = wp.tid()
        world = block * W + lane // 32
        if world < world_count:
            solve_native(
                world,
                world_constraint_count,
                rhs_bias,
                world_diag,
                world_impulses,
                row_dof,
                row_factor,
                row_incident,
                world_row_type,
                world_row_parent,
                world_row_mu,
                iterations,
                omega,
                friction_start_iteration,
                iteration_offset,
                factor_velocity_delta,
            )

    solve.__name__ = f"pgs_solve_sparse_{M}_{D}_{S}"
    solve.__qualname__ = solve.__name__
    return wp.kernel(enable_backward=False, module="unique")(solve)
