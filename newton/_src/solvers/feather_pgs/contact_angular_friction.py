# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Per-contact torsional and rolling friction from the shape material coefficients.

Each dense contact that receives a sliding pair and has a positive pair coefficient
reserves three more contiguous rows: ``[normal, t1, t2, spin, roll1, roll2]``, and its
five friction rows take the angular-friction row type so that one solve owns them. The
spin row resists relative rotation about the normal and the rolling pair about the
contact tangents. With the normal impulse fixed, each Gauss-Seidel visit solves the
five friction rows of a contact as one block: it minimizes the block's quadratic over a
joint cone in coefficient-normalized impulses ``(f_t / mu, tau_s / mu_s, tau_r / mu_r)``.
Rows with no coefficient or no response stay at zero, and the unconstrained minimizer is
taken when it lies inside the cone.

- ``"elliptic"``: ``(|f_t| / mu)^2 + (tau_s / mu_s)^2 + (|tau_r| / mu_r)^2 <= lambda_n^2``.
  The block is a trust-region subproblem, solved by safeguarded Newton on its multiplier
  until the boundary residual is below a relative tolerance.
- ``"pyramidal"``: ``|f_t| / mu + |tau_s| / mu_s + |tau_r| / mu_r <= lambda_n``, an L1 norm
  over the three blocks with a disk inside the sliding and rolling blocks. This is not
  MuJoCo's component-wise pyramid. Accelerated projected gradient in group-scaled
  coordinates finds the face, then reweighted trust-region solves refine on it. The
  optimum may share the budget between blocks or stick inside the cone; in quadruped
  locomotion tests pivoting stance feet put most of it on rolling.

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
    elliptic = cone == "elliptic"
    helpers = (
        _BLOCK_SOURCE.replace("__PROJECT__", _ELLIPTIC_PROJECT if elliptic else _PYRAMIDAL_PROJECT)
        .replace("__CONSTRAINED__", _ELLIPTIC_CONSTRAINED if elliptic else _PYRAMIDAL_CONSTRAINED)
        .replace("__DOFS__", str(dofs))
        .replace("__CREEP__", f"{float(creep_speed)!r}f")
    )
    row_block = f"""
            }} else if (row_type == {angular}) {{
                // Rows [t1, t2, spin, roll1, roll2] of one contact; the first solves the whole block.
                int parent_idx = (s_meta_dense[i] >> __DENSE_META_ROW_TYPE_BITS__) - 1;
                if (i == parent_idx + 1) {{
                    new_impulse = AngularFrictionBlock::solve(
                        s_v, &s_lam_dense[i], &s_mu_dense[i], &s_rhs_dense[i],
                        &J_world.data[jy_world_base + i * {dofs}], &Y_world.data[jy_world_base + i * {dofs}],
                        lane, MASK, s_lam_dense[parent_idx], omega, &iteration_changed);
                }} else {{
                    new_impulse = old_impulse;
                }}
                delta_impulse = new_impulse - old_impulse;"""
    return {"helpers": helpers, "row_block": row_block}


_ELLIPTIC_PROJECT = """
        static __device__ float cone_norm(const float* y) {
            float norm = 0.0f;
            for (int k = 0; k < 5; ++k) norm += y[k] * y[k];
            return sqrtf(norm);
        }

        static __device__ void project(float* y, const float* w, float load) {
            float norm = cone_norm(y);
            if (norm > load) {
                float scale = load / norm;
                for (int k = 0; k < 5; ++k) y[k] *= scale;
            }
        }"""

_PYRAMIDAL_PROJECT = """
        static __device__ float cone_norm(const float* y) {
            return sqrtf(y[0] * y[0] + y[1] * y[1]) + fabsf(y[2]) + sqrtf(y[3] * y[3] + y[4] * y[4]);
        }

        // Projection onto sum_g w_g |y_g| <= load over the groups (t1, t2), (spin), (roll1, roll2).
        static __device__ void project(float* y, const float* w, float load) {
            const int first[3] = {0, 2, 3};
            const int count[3] = {2, 1, 2};
            float norm[3] = {sqrtf(y[0] * y[0] + y[1] * y[1]), fabsf(y[2]), sqrtf(y[3] * y[3] + y[4] * y[4])};
            if (w[0] * norm[0] + w[1] * norm[1] + w[2] * norm[2] <= load) return;
            int order[3] = {0, 1, 2};
            for (int a = 0; a < 2; ++a)
                for (int b = a + 1; b < 3; ++b)
                    if (norm[order[b]] * w[order[a]] > norm[order[a]] * w[order[b]]) {
                        int t = order[a]; order[a] = order[b]; order[b] = t;
                    }
            // The shrink solving sum_g w_g max(|y_g| - theta w_g, 0) = load is the largest prefix candidate;
            // when rounding swallows load it is the largest breakpoint, which zeroes every group.
            float theta = 0.0f, linear = 0.0f, quadratic = 0.0f;
            for (int a = 0; a < 3; ++a) {
                int g = order[a];
                linear += w[g] * norm[g];
                quadratic += w[g] * w[g];
                theta = fmaxf(theta, (linear - load) / quadratic);
            }
            float total = 0.0f;
            for (int g = 0; g < 3; ++g) {
                float scale = norm[g] > 0.0f ? fmaxf(norm[g] - theta * w[g], 0.0f) / norm[g] : 0.0f;
                for (int k = first[g]; k < first[g] + count[g]; ++k) y[k] *= scale;
                total += w[g] * norm[g] * scale;
            }
            if (total > load) {
                float scale = load / total;
                for (int k = 0; k < 5; ++k) y[k] *= scale;
            }
        }
"""

_ELLIPTIC_CONSTRAINED = """
                    float c[5];
                    for (int k = 0; k < 5; ++k) {
                        c[k] = gradient0[k];
                        for (int l = 0; l < 5; ++l) c[k] -= H[k][l] * y0[l];
                    }
                    trust_region(H, c, active, load, y);"""

_PYRAMIDAL_CONSTRAINED = """
                    // Accelerated projected gradient in group-scaled coordinates u_g = s_g y_g, which give the
                    // sliding, spin and rolling groups comparable curvature; stops on the gradient-mapping residual.
                    const int first[3] = {0, 2, 3};
                    const int count[3] = {2, 1, 2};
                    float c[5], s[5], w[3], Hu[5][5], cu[5], u[5], z[5], lipschitz = 0.0f;
                    for (int k = 0; k < 5; ++k) {
                        c[k] = gradient0[k];
                        for (int l = 0; l < 5; ++l) c[k] -= H[k][l] * y0[l];
                        c[k] = active[k] ? c[k] : 0.0f;
                    }
                    for (int g = 0; g < 3; ++g) {
                        float curvature = 0.0f;
                        for (int k = first[g]; k < first[g] + count[g]; ++k)
                            curvature = fmaxf(curvature, active[k] ? H[k][k] : 0.0f);
                        float scale = curvature > 0.0f ? sqrtf(curvature) : 1.0f;
                        w[g] = 1.0f / scale;
                        for (int k = first[g]; k < first[g] + count[g]; ++k) s[k] = scale;
                    }
                    for (int k = 0; k < 5; ++k) {
                        float row_sum = 0.0f;
                        for (int l = 0; l < 5; ++l) {
                            Hu[k][l] = H[k][l] / (s[k] * s[l]);
                            row_sum += fabsf(Hu[k][l]);
                        }
                        lipschitz = fmaxf(lipschitz, active[k] ? row_sum : 0.0f);
                        cu[k] = c[k] / s[k];
                        u[k] = y0[k] * s[k];
                    }
                    project(u, w, load);
                    for (int k = 0; k < 5; ++k) z[k] = u[k];
                    float momentum = 1.0f;
                    for (int iteration = 0; iteration < 64 && lipschitz > 0.0f; ++iteration) {
                        float next[5];
                        for (int k = 0; k < 5; ++k) {
                            float gradient = cu[k];
                            for (int l = 0; l < 5; ++l) gradient += Hu[k][l] * z[l];
                            next[k] = active[k] ? z[k] - gradient / lipschitz : 0.0f;
                        }
                        project(next, w, load);
                        // Residual of the projected-gradient fixed point at z, per group in y units.
                        float residual = 0.0f, ascent = 0.0f;
                        for (int g = 0; g < 3; ++g) {
                            float squared = 0.0f;
                            for (int k = first[g]; k < first[g] + count[g]; ++k)
                                squared += (next[k] - z[k]) * (next[k] - z[k]);
                            residual = fmaxf(residual, sqrtf(squared) * w[g]);
                        }
                        for (int k = 0; k < 5; ++k) ascent += (z[k] - next[k]) * (next[k] - u[k]);
                        momentum = ascent > 0.0f ? 1.0f : momentum;
                        float next_momentum = 0.5f * (1.0f + sqrtf(1.0f + 4.0f * momentum * momentum));
                        float blend = (momentum - 1.0f) / next_momentum;
                        for (int k = 0; k < 5; ++k) {
                            z[k] = next[k] + blend * (next[k] - u[k]);
                            u[k] = next[k];
                        }
                        momentum = next_momentum;
                        if (residual <= 1.0e-5f * load) break;
                    }
                    for (int k = 0; k < 5; ++k) y[k] = u[k] / s[k];
                    // On the face found above, sum_g |y_g| <= load is sum_g |y_g|^2 / t_g <= load^2 at t_g = |y_g| / sum |y|.
                    // Alternating the weighted trust-region solve with that t converges to the optimum.
                    for (int iteration = 0; iteration < 8; ++iteration) {
                        float norm[3], total = 0.0f;
                        for (int g = 0; g < 3; ++g) {
                            float squared = 0.0f;
                            for (int k = first[g]; k < first[g] + count[g]; ++k) squared += y[k] * y[k];
                            norm[g] = sqrtf(squared);
                            total += norm[g];
                        }
                        if (!(total > 0.0f)) break;
                        float r[5], cr[5];
                        bool face[5];
                        for (int g = 0; g < 3; ++g)
                            for (int k = first[g]; k < first[g] + count[g]; ++k) {
                                r[k] = sqrtf(norm[g] / total);
                                face[k] = active[k] && r[k] > 0.0f;
                                cr[k] = r[k] * c[k];
                            }
                        for (int k = 0; k < 5; ++k)
                            for (int l = 0; l < 5; ++l) Hu[k][l] = r[k] * r[l] * H[k][l];
                        trust_region(Hu, cr, face, load, y);
                        for (int k = 0; k < 5; ++k) y[k] *= r[k];
                    }"""

_BLOCK_SOURCE = """
    // Kept out of line so contacts without angular rows keep the sweep's register budget.
    struct AngularFrictionBlock {
        typedef float Row[5];
__PROJECT__

        // Cholesky of H + alpha I over the active rows; fails when a pivot keeps under 1e-6 of its diagonal.
        static __device__ bool factor(const Row* H, const bool* active, float alpha, Row* L) {
            for (int k = 0; k < 5; ++k) {
                for (int l = 0; l <= k; ++l) {
                    float value = active[k] && active[l] ? H[k][l] : 0.0f;
                    if (k == l) value = active[k] ? value + alpha : 1.0f;
                    for (int m = 0; m < l; ++m) value -= L[k][m] * L[l][m];
                    if (k == l) {
                        if (!(value > 1.0e-6f * (active[k] ? H[k][k] + alpha : 1.0f))) return false;
                        L[k][k] = sqrtf(value);
                    } else {
                        L[k][l] = value / L[l][l];
                    }
                }
            }
            return true;
        }

        static __device__ void lower(const Row* L, float* x) {
            for (int k = 0; k < 5; ++k) {
                for (int m = 0; m < k; ++m) x[k] -= L[k][m] * x[m];
                x[k] /= L[k][k];
            }
        }

        static __device__ void upper(const Row* L, float* x) {
            for (int k = 4; k >= 0; --k) {
                for (int m = k + 1; m < 5; ++m) x[k] -= L[m][k] * x[m];
                x[k] /= L[k][k];
            }
        }

        // Minimize 0.5 y'Hy + c'y over |y| <= load on the active rows: y = -(H + alpha I)^-1 c, with alpha found by
        // Newton on 1 / |y(alpha)| (More-Sorensen) inside a bracket, stopping on the boundary residual.
        static __device__ void trust_region(const Row* H, const float* c, const bool* active, float load, float* y) {
            float L[5][5], b[5], c_norm = 0.0f;
            for (int k = 0; k < 5; ++k) {
                b[k] = active[k] ? c[k] : 0.0f;
                c_norm += b[k] * b[k];
                y[k] = 0.0f;
            }
            c_norm = sqrtf(c_norm);
            if (!(load > 0.0f && c_norm > 0.0f)) return;
            // |y(hi)| <= |c| / hi = load because H is positive semidefinite.
            float lo = 0.0f, hi = c_norm / load, alpha = 0.0f;
            for (int iteration = 0; iteration < 40; ++iteration) {
                float next = -1.0f;
                if (factor(H, active, alpha, L)) {
                    for (int k = 0; k < 5; ++k) y[k] = -b[k];
                    lower(L, y);
                    upper(L, y);
                    float norm = 0.0f;
                    for (int k = 0; k < 5; ++k) norm += y[k] * y[k];
                    norm = sqrtf(norm);
                    if (alpha == 0.0f && norm <= load) return;
                    if (fabsf(norm - load) <= 1.0e-5f * load) break;
                    if (norm < load) hi = alpha; else lo = alpha;
                    float w[5], w_norm = 0.0f;
                    for (int k = 0; k < 5; ++k) w[k] = y[k];
                    lower(L, w);
                    for (int k = 0; k < 5; ++k) w_norm += w[k] * w[k];
                    next = alpha + norm * norm / w_norm * (norm - load) / load;
                } else {
                    lo = alpha;
                }
                // A singular block whose minimizer is interior closes the bracket from below.
                if (hi - lo <= 1.0e-4f * hi) break;
                alpha = next > lo && next < hi ? next : (lo > 0.0f ? sqrtf(lo * hi) : 1.0e-3f * hi);
            }
            float norm = 0.0f;
            for (int k = 0; k < 5; ++k) norm += y[k] * y[k];
            norm = sqrtf(norm);
            if (norm > load) {
                float scale = load / norm;
                for (int k = 0; k < 5; ++k) y[k] *= scale;
            }
        }

        // Minimize the five-row block quadratic over the cone; update velocities and rows 1..4, return row 0.
        static __device__ __noinline__ float solve(
            float* v, float* lam, const float* mu_rows, const float* rhs_rows, const float* J, const float* Y,
            int lane, unsigned MASK, float load, float omega, int* changed) {
                float sums[20];
                for (int q = 0; q < 20; ++q) sums[q] = 0.0f;
                for (int d = lane; d < __DOFS__; d += 32) {
                    float jr[5], yr[5];
                    for (int k = 0; k < 5; ++k) {
                        jr[k] = J[k * __DOFS__ + d];
                        yr[k] = Y[k * __DOFS__ + d];
                    }
                    int q = 5;
                    for (int k = 0; k < 5; ++k) {
                        sums[k] += jr[k] * v[d];
                        for (int l = k; l < 5; ++l) sums[q++] += jr[k] * yr[l];
                    }
                }
                for (int q = 0; q < 20; ++q) {
                    float value = sums[q];
                    for (int offset = 16; offset > 0; offset >>= 1)
                        value += __shfl_down_sync(MASK, value, offset);
                    sums[q] = __shfl_sync(MASK, value, 0);
                }
                load = fmaxf(load, 0.0f);
                float x0[5], gradient0[5], mu[5], H[5][5], largest = 0.0f;
                int q = 5;
                for (int k = 0; k < 5; ++k) {
                    x0[k] = lam[k];
                    mu[k] = fmaxf(mu_rows[k], 0.0f);
                    gradient0[k] = sums[k] + rhs_rows[k];
                    for (int l = k; l < 5; ++l) { H[k][l] = sums[q]; H[l][k] = sums[q]; ++q; }
                    largest = fmaxf(largest, H[k][k]);
                }
                // Rows without a coefficient, or whose response is round-off, stay at zero.
                bool active[5];
                for (int k = 0; k < 5; ++k) active[k] = mu[k] > 0.0f && H[k][k] > 1.0e-12f * largest;
                for (int k = 2; k < 5; ++k) {
                    // Creep compliance c adds 0.5 c x^2: c = creep_speed / (mu^2 lambda_n).
                    float compliance = active[k] && load > 0.0f ? __CREEP__ / (mu[k] * mu[k] * load) : 0.0f;
                    H[k][k] += compliance;
                    gradient0[k] += compliance * x0[k];
                }
                // Normalized coordinates y = x / mu, in which the cone has radius load.
                float y0[5], y[5], L[5][5];
                for (int k = 0; k < 5; ++k) {
                    y0[k] = active[k] ? x0[k] / mu[k] : 0.0f;
                    gradient0[k] = active[k] ? gradient0[k] * mu[k] : 0.0f;
                    for (int l = 0; l < 5; ++l) H[k][l] *= mu[k] * mu[l];
                }
                // Sticking: the unconstrained minimizer is exact when it lies inside the cone.
                bool sticking = false;
                if (factor(H, active, 0.0f, L)) {
                    for (int k = 0; k < 5; ++k) y[k] = -gradient0[k];
                    lower(L, y);
                    upper(L, y);
                    for (int k = 0; k < 5; ++k) y[k] = active[k] ? y0[k] + y[k] : 0.0f;
                    sticking = cone_norm(y) <= load;
                }
                const float unit[3] = {1.0f, 1.0f, 1.0f};
                if (!sticking) {
__CONSTRAINED__
                    project(y, unit, load);
                }
                float x[5];
                for (int k = 0; k < 5; ++k) x[k] = x0[k] + omega * (mu[k] * y[k] - x0[k]);
                if (omega != 1.0f) {
                    for (int k = 0; k < 5; ++k) y[k] = active[k] ? x[k] / mu[k] : 0.0f;
                    project(y, unit, load);
                    for (int k = 0; k < 5; ++k) x[k] = mu[k] * y[k];
                }
                for (int k = 1; k < 5; ++k) {
                    float block_delta = x[k] - lam[k];
                    if (block_delta != 0.0f) {
                        *changed = 1;
                        for (int d = lane; d < __DOFS__; d += 32)
                            v[d] += Y[k * __DOFS__ + d] * block_delta;
                    }
                    lam[k] = x[k];
                }
                return x[0];
        }
    };
"""
