# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Per-contact torsional and rolling friction from the shape material coefficients.

Each dense contact that receives a sliding pair and has a positive pair coefficient
reserves three more contiguous rows: ``[normal, t1, t2, spin, roll1, roll2]``. The
spin row resists relative rotation about the normal and the rolling pair about the
contact tangents. With the normal impulse fixed, each Gauss-Seidel visit solves the
five friction rows of a contact as one block: it minimizes the block's quadratic
over a joint cone in coefficient-normalized impulses ``(f_t / mu, tau_s / mu_s,
tau_r / mu_r)``, using accelerated projected gradient with exact Euclidean projection:

- ``"elliptic"``: ``(|f_t| / mu)^2 + (tau_s / mu_s)^2 + (|tau_r| / mu_r)^2 <= lambda_n^2``;
- ``"pyramidal"``: ``|f_t| / mu + |tau_s| / mu_s + |tau_r| / mu_r <= lambda_n``, an L1 norm
  over the three blocks with a disk inside the sliding and rolling blocks. This is not
  MuJoCo's component-wise pyramid.

An optional creep speed ``s`` [m/s] adds a compliance to the angular rows: below the
bound the coefficient times the relative angular rate settles at ``s`` times the load
fraction ``|tau| / (mu_i lambda_n)``. Rows are rebuilt every step and carry no history.
"""

import numpy as np
import warp as wp

from .kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT_ANGULAR_FRICTION,
    populate_world_angular_friction_rows,
)

ANGULAR_FRICTION_CONES = ("pyramidal", "elliptic")


def configure_torsional_rolling_friction(solver, enabled: bool, cone: str, creep_speed: float) -> None:
    """Record the opt-in, its cone and creep speed before any route selection reads them."""
    if cone not in ANGULAR_FRICTION_CONES:
        raise ValueError(f"torsional_rolling_friction_cone must be one of {ANGULAR_FRICTION_CONES}, got {cone!r}")
    creep_speed = float(creep_speed)
    if not np.isfinite(creep_speed) or creep_speed < 0.0:
        raise ValueError("torsional_rolling_friction_creep_speed must be finite and non-negative [m/s]")
    solver.enable_torsional_rolling_friction = bool(enabled)
    solver.torsional_rolling_friction_cone = cone
    solver.torsional_rolling_friction_creep_speed = creep_speed


def validate_torsional_rolling_friction_mode(solver) -> None:
    """Reject solver options whose row routes do not solve the angular friction rows."""
    if not solver.enable_torsional_rolling_friction:
        return
    model = solver.model
    if (
        not model.device.is_cuda
        or model.requires_grad
        or solver.pgs_mode != "matrix_free"
        or solver.articulated_contact_response != "immediate"
        or solver.pgs_schedule != "interleaved"
        or solver.friction_mode != "current"
        or solver._friction_anchors_enabled
        or solver.pgs_warmstart
        or solver.pgs_velocity_iterations != 0
        or solver.pgs_debug
        or solver.contact_compliance
        or solver._contact_torsion_enabled
        or solver.sleeping is not None
        or not solver.enable_contact_friction
    ):
        raise ValueError(
            "enable_torsional_rolling_friction requires CUDA non-differentiable matrix_free/immediate/interleaved "
            "solving with friction_mode='current', point friction (friction_anchor_beta=0), and no warm start, "
            "velocity iterations, debug, contact compliance, contact torsion radius, or sleeping"
        )


def validate_torsional_rolling_friction_coefficients(solver) -> None:
    """Reject positive coefficients that could reach the matrix-free free-rigid contact route."""
    if not solver.enable_torsional_rolling_friction or not solver._has_free_rigid_bodies:
        return
    model = solver.model
    if not model.shape_count:
        return
    coefficients = np.maximum(
        np.nan_to_num(model.shape_material_mu_torsional.numpy(), nan=0.0, posinf=0.0),
        np.nan_to_num(model.shape_material_mu_rolling.numpy(), nan=0.0, posinf=0.0),
    )
    shape_body = model.shape_body.numpy()
    body_art = solver.body_to_articulation.numpy()
    free = solver.is_free_rigid.numpy()
    response_dofs = solver.articulation_response_dof_count.numpy()
    art = np.where(shape_body >= 0, body_art[np.maximum(shape_body, 0)], -1)
    safe_art = np.maximum(art, 0)
    # The allocator's matrix-free compatibility: no articulation, no response DOFs (e.g. fixed), or free rigid.
    free_route = (art < 0) | (response_dofs[safe_art] == 0) | (free[safe_art] != 0)
    shapes = np.flatnonzero(free_route & (coefficients > 0.0))
    if shapes.size:
        raise ValueError(
            f"Torsional/rolling friction is unsupported on free-rigid contacts; zero the coefficients of shapes {shapes.tolist()}"
        )


def launch_angular_friction_rows(solver, state_in, state_aug, contacts, threads: int) -> None:
    """Fill the reserved spin and rolling rows for every size group after the contact rows."""
    model = solver.model
    for index, size in enumerate(solver.size_groups):
        wp.launch(
            populate_world_angular_friction_rows,
            dim=threads,
            inputs=[
                contacts.rigid_contact_count,
                threads,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                solver.contact_world,
                solver.contact_slot,
                solver.contact_art_a,
                solver.contact_art_b,
                solver.contact_path,
                solver.contact_slots_needed,
                size,
                solver.articulation_response_dof_count,
                solver.art_group_idx,
                solver.articulation_dof_start,
                solver.body_to_joint,
                model.joint_ancestor,
                model.joint_qd_start,
                state_aug.joint_S_s,
                model.shape_body,
                state_aug.body_v_s,
                solver._prescribed_articulation,
                solver._shape_mu_torsional,
                solver._shape_mu_rolling,
                solver.pgs_cfm,
                int(index == 0),
            ],
            outputs=[
                solver.J_by_size[size],
                solver.row_type,
                solver.row_parent,
                solver.row_mu,
                solver.row_beta,
                solver.row_cfm,
                solver.phi,
                solver.target_velocity,
                solver.row_restitution,
            ],
            device=model.device,
        )


def angular_friction_gs_sources(cone: str, creep_speed: float, dofs: int) -> dict[str, str]:
    """Return the CUDA fragments that the fused matrix-free GS kernel splices in for ``cone`` and ``creep_speed``."""
    angular = PGS_CONSTRAINT_TYPE_CONTACT_ANGULAR_FRICTION
    if cone == "elliptic":
        projection = """
        float norm = 0.0f;
        for (int k = 0; k < 5; ++k) norm += y[k] * y[k];
        norm = sqrtf(norm);
        if (norm > load) {
            float scale = load / norm;
            for (int k = 0; k < 5; ++k) y[k] *= scale;
        }"""
    else:
        projection = """
        float group[3] = {sqrtf(y[0] * y[0] + y[1] * y[1]), fabsf(y[2]), sqrtf(y[3] * y[3] + y[4] * y[4])};
        if (group[0] + group[1] + group[2] > load) {
            // Project the group norms onto the simplex of radius load, then rescale each group.
            float sorted[3] = {group[0], group[1], group[2]};
            for (int a = 0; a < 2; ++a)
                for (int b = a + 1; b < 3; ++b)
                    if (sorted[b] > sorted[a]) { float t = sorted[a]; sorted[a] = sorted[b]; sorted[b] = t; }
            float theta = 0.0f, cumulative = 0.0f;
            for (int a = 0; a < 3; ++a) {
                cumulative += sorted[a];
                float candidate = (cumulative - load) / (float)(a + 1);
                if (sorted[a] - candidate > 0.0f) theta = candidate;
            }
            const int first[3] = {0, 2, 3};
            const int count[3] = {2, 1, 2};
            for (int g = 0; g < 3; ++g) {
                float scale = group[g] > 0.0f ? fmaxf(group[g] - theta, 0.0f) / group[g] : 0.0f;
                for (int k = first[g]; k < first[g] + count[g]; ++k) y[k] *= scale;
            }
        }"""
    helpers = f"""
    // Exact Euclidean projection of normalized friction impulses onto the joint cone of radius load.
    const auto angular_cone_projection = [](float* y, float load) {{
        load = fmaxf(load, 0.0f);{projection}
    }};
"""
    block_open = f"""
                    if (parent_idx + 5 < m_dense &&
                        world_row_type.data[off_dense + parent_idx + 3] == {angular} &&
                        world_row_parent.data[off_dense + parent_idx + 3] == parent_idx) {{
                        // Block solve of the five friction rows [t1, t2, spin, roll1, roll2] at fixed normal load.
                        const int first_row = parent_idx + 1;
                        float sums[20];
                        for (int q = 0; q < 20; ++q) sums[q] = 0.0f;
                        for (int d = lane; d < {dofs}; d += 32) {{
                            float jr[5], yr[5];
                            for (int k = 0; k < 5; ++k) {{
                                jr[k] = J_world.data[jy_world_base + (first_row + k) * {dofs} + d];
                                yr[k] = Y_world.data[jy_world_base + (first_row + k) * {dofs} + d];
                            }}
                            int q = 5;
                            for (int k = 0; k < 5; ++k) {{
                                sums[k] += jr[k] * s_v[d];
                                for (int l = k; l < 5; ++l) sums[q++] += jr[k] * yr[l];
                            }}
                        }}
                        for (int q = 0; q < 20; ++q) {{
                            float value = sums[q];
                            for (int offset = 16; offset > 0; offset >>= 1)
                                value += __shfl_down_sync(MASK, value, offset);
                            sums[q] = __shfl_sync(MASK, value, 0);
                        }}
                        float load = fmaxf(s_lam_dense[parent_idx], 0.0f);
                        float x0[5], gradient0[5], mu[5], H[5][5];
                        int q = 5;
                        for (int k = 0; k < 5; ++k) {{
                            x0[k] = s_lam_dense[first_row + k];
                            mu[k] = fmaxf(s_mu_dense[first_row + k], 0.0f);
                            gradient0[k] = sums[k] + s_rhs_dense[first_row + k];
                            for (int l = k; l < 5; ++l) {{ H[k][l] = sums[q]; H[l][k] = sums[q]; ++q; }}
                        }}
                        for (int k = 2; k < 5; ++k) {{
                            // Creep compliance c adds 0.5 c x^2: c = creep_speed / (mu^2 lambda_n).
                            float compliance = mu[k] > 0.0f && load > 0.0f ? {float(creep_speed)!r}f / (mu[k] * mu[k] * load) : 0.0f;
                            H[k][k] += compliance;
                            gradient0[k] += compliance * x0[k];
                        }}
                        // Normalized coordinates y = x / mu; rows with mu == 0 stay at zero.
                        float y0[5], y[5], z[5], lipschitz = 0.0f;
                        for (int k = 0; k < 5; ++k) {{
                            y0[k] = mu[k] > 0.0f ? x0[k] / mu[k] : 0.0f;
                            gradient0[k] *= mu[k];
                            float row_sum = 0.0f;
                            for (int l = 0; l < 5; ++l) {{
                                H[k][l] *= mu[k] * mu[l];
                                row_sum += fabsf(H[k][l]);
                            }}
                            lipschitz = fmaxf(lipschitz, row_sum);
                        }}
                        // Sticking: the unconstrained block minimizer, when it lies inside the cone, is exact.
                        bool solved = false;
                        {{
                            float L[5][5], step[5], largest_pivot = 0.0f;
                            for (int k = 0; k < 5; ++k) largest_pivot = fmaxf(largest_pivot, mu[k] > 0.0f ? H[k][k] : 0.0f);
                            // Near-singular blocks (rows without response) take the projected-gradient path.
                            bool positive = largest_pivot > 0.0f;
                            for (int k = 0; k < 5 && positive; ++k) {{
                                for (int l = 0; l <= k; ++l) {{
                                    float value = mu[k] > 0.0f && mu[l] > 0.0f ? H[k][l] : (k == l ? 1.0f : 0.0f);
                                    for (int m = 0; m < l; ++m) value -= L[k][m] * L[l][m];
                                    if (k == l) {{
                                        if (!(value > 1.0e-5f * largest_pivot)) positive = false;
                                        L[k][k] = sqrtf(fmaxf(value, 0.0f));
                                    }} else {{
                                        L[k][l] = value / L[l][l];
                                    }}
                                }}
                            }}
                            if (positive) {{
                                for (int k = 0; k < 5; ++k) {{
                                    float value = mu[k] > 0.0f ? -gradient0[k] : 0.0f;
                                    for (int m = 0; m < k; ++m) value -= L[k][m] * step[m];
                                    step[k] = value / L[k][k];
                                }}
                                for (int k = 4; k >= 0; --k) {{
                                    float value = step[k];
                                    for (int m = k + 1; m < 5; ++m) value -= L[m][k] * step[m];
                                    step[k] = value / L[k][k];
                                }}
                                for (int k = 0; k < 5; ++k) y[k] = mu[k] > 0.0f ? y0[k] + step[k] : 0.0f;
                                float trial[5];
                                for (int k = 0; k < 5; ++k) trial[k] = y[k];
                                angular_cone_projection(trial, load);
                                solved = true;
                                for (int k = 0; k < 5; ++k) solved = solved && trial[k] == y[k];
                            }}
                        }}
                        if (!solved) {{
                            for (int k = 0; k < 5; ++k) y[k] = y0[k];
                            angular_cone_projection(y, load);
                        }}
                        if (!solved && lipschitz > 0.0f) {{
                            for (int k = 0; k < 5; ++k) z[k] = y[k];
                            float momentum = 1.0f;
                            for (int iteration = 0; iteration < 32; ++iteration) {{
                                float next[5];
                                for (int k = 0; k < 5; ++k) {{
                                    float gradient = gradient0[k];
                                    for (int l = 0; l < 5; ++l) gradient += H[k][l] * (z[l] - y0[l]);
                                    next[k] = mu[k] > 0.0f ? z[k] - gradient / lipschitz : 0.0f;
                                }}
                                angular_cone_projection(next, load);
                                float next_momentum = 0.5f * (1.0f + sqrtf(1.0f + 4.0f * momentum * momentum));
                                float blend = (momentum - 1.0f) / next_momentum;
                                float change = 0.0f;
                                for (int k = 0; k < 5; ++k) {{
                                    change = fmaxf(change, fabsf(next[k] - y[k]));
                                    z[k] = next[k] + blend * (next[k] - y[k]);
                                    y[k] = next[k];
                                }}
                                momentum = next_momentum;
                                // Outer sweeps warm-start the block, so stop once an iterate stalls.
                                if (change <= 1.0e-6f * load) break;
                            }}
                        }}
                        float x[5];
                        for (int k = 0; k < 5; ++k) x[k] = x0[k] + omega * (mu[k] * y[k] - x0[k]);
                        if (omega != 1.0f) {{
                            for (int k = 0; k < 5; ++k) y[k] = mu[k] > 0.0f ? x[k] / mu[k] : 0.0f;
                            angular_cone_projection(y, load);
                            for (int k = 0; k < 5; ++k) x[k] = mu[k] * y[k];
                        }}
                        for (int k = 1; k < 5; ++k) {{
                            float block_delta = x[k] - s_lam_dense[first_row + k];
                            if (block_delta != 0.0f) {{
                                iteration_changed = 1;
                                for (int d = lane; d < {dofs}; d += 32)
                                    s_v[d] += Y_world.data[jy_world_base + (first_row + k) * {dofs} + d] * block_delta;
                            }}
                            s_lam_dense[first_row + k] = x[k];
                        }}
                        new_impulse = x[0];
                    }} else {{"""
    block_close = """
                    }"""
    row_block = f"""
            }} else if (row_type == {angular}) {{
                // The contact's first sliding row solves these rows in its friction block.
                new_impulse = old_impulse;
                delta_impulse = 0.0f;"""
    return {"helpers": helpers, "block_open": block_open, "block_close": block_close, "row_block": row_block}
