# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Per-contact torsional and rolling friction from the shape material coefficients.

Each dense contact that receives a sliding pair and has a positive pair coefficient
reserves three more contiguous rows: ``[normal, t1, t2, spin, roll1, roll2]``. The
spin row resists relative rotation about the normal and the rolling pair about the
contact tangents. Their impulses [N m s] are bounded by the coefficient [m] times
the normal impulse left after the other friction blocks, as MuJoCo's friction cones
couple sliding, torsional and rolling friction:

- ``"pyramidal"``: ``|f_t| / mu + |tau_s| / mu_s + |tau_r| / mu_r <= lambda_n``;
- ``"elliptic"``: ``(|f_t| / mu)^2 + (tau_s / mu_s)^2 + (|tau_r| / mu_r)^2 <= lambda_n^2``.

Each block keeps a disk within itself. An optional creep speed ``s`` [m/s] softens
stiction as MuJoCo's soft constraints do: below the bound the coefficient times the
relative angular rate settles at ``s`` times the load fraction ``|tau| / (mu_i lambda_n)``.
Rows are rebuilt every step and carry no history.
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
    art = np.where(shape_body >= 0, body_art[np.maximum(shape_body, 0)], -1)
    # Free bodies and static shapes are the shapes a free-rigid route contact can pair.
    free_route = (art < 0) | (free[np.maximum(art, 0)] != 0)
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
    if cone == "pyramidal":
        residual_load = "fmaxf(load - a - b, 0.0f)"
    else:
        residual_load = "sqrtf(fmaxf(load * load - a * a - b * b, 0.0f))"
    angular = PGS_CONSTRAINT_TYPE_CONTACT_ANGULAR_FRICTION
    helpers = f"""
    // Normal load left for one friction block after the other two blocks' normalized usage.
    const auto angular_residual_load = [](float load, float a, float b) {{
        load = fmaxf(load, 0.0f);
        return {residual_load};
    }};
    const auto angular_usage = [](float magnitude, float mu) {{
        return mu > 0.0f ? magnitude / mu : 0.0f;
    }};
    // Row compliance giving a creep rate of creep_speed / mu at the bound of a load lambda_n.
    const auto angular_compliance = [](float mu, float lambda_n) {{
        return mu > 0.0f && lambda_n > 0.0f ? {float(creep_speed)!r}f / (mu * mu * lambda_n) : 0.0f;
    }};
"""
    sliding_radius = f"""
                    if (parent_idx + 5 < m_dense &&
                        world_row_type.data[off_dense + parent_idx + 3] == {angular} &&
                        world_row_parent.data[off_dense + parent_idx + 3] == parent_idx) {{
                        float spin_usage = angular_usage(fabsf(s_lam_dense[parent_idx + 3]), s_mu_dense[parent_idx + 3]);
                        float roll_a = s_lam_dense[parent_idx + 4], roll_b = s_lam_dense[parent_idx + 5];
                        float roll_usage = angular_usage(sqrtf(roll_a * roll_a + roll_b * roll_b), s_mu_dense[parent_idx + 4]);
                        radius = fmaxf(s_mu_dense[i], 0.0f) * angular_residual_load(lambda_n, spin_usage, roll_usage);
                    }}"""
    row_block = f"""
            }} else if (row_type == {angular}) {{
                int parent_idx = (s_meta_dense[i] >> __DENSE_META_ROW_TYPE_BITS__) - 1;
                float lambda_n = s_lam_dense[parent_idx];
                float slide_a = s_lam_dense[parent_idx + 1], slide_b = s_lam_dense[parent_idx + 2];
                float slide_usage = angular_usage(sqrtf(slide_a * slide_a + slide_b * slide_b), s_mu_dense[parent_idx + 1]);
                float spin_usage = angular_usage(fabsf(s_lam_dense[parent_idx + 3]), s_mu_dense[parent_idx + 3]);
                float roll_a = s_lam_dense[parent_idx + 4], roll_b = s_lam_dense[parent_idx + 5];
                float roll_usage = angular_usage(sqrtf(roll_a * roll_a + roll_b * roll_b), s_mu_dense[parent_idx + 4]);
                if (i == parent_idx + 3) {{
                    float bound = fmaxf(s_mu_dense[i], 0.0f) * angular_residual_load(lambda_n, slide_usage, roll_usage);
                    float compliance = angular_compliance(s_mu_dense[i], lambda_n);
                    if (compliance > 0.0f)
                        new_impulse = old_impulse - omega * (residual + compliance * old_impulse) / (fmaxf(denom, 0.0f) + compliance);
                    new_impulse = fminf(fmaxf(new_impulse, -bound), bound);
                }} else if (i == parent_idx + 4) {{
                    int sib = i + 1;
                    int sib_row_base = jy_world_base + sib * {dofs};
                    float radius = fmaxf(s_mu_dense[i], 0.0f) * angular_residual_load(lambda_n, slide_usage, spin_usage);
                    float sibling_residual = 0.0f;
                    for (int d = lane; d < {dofs}; d += 32)
                        sibling_residual += J_world.data[sib_row_base + d] * s_v[d];
                    for (int offset = 16; offset > 0; offset >>= 1)
                        sibling_residual += __shfl_down_sync(MASK, sibling_residual, offset);
                    sibling_residual = __shfl_sync(MASK, sibling_residual, 0) + s_rhs_dense[sib];
                    float cross = 0.0f;
                    for (int d = lane; d < {dofs}; d += 32)
                        cross += J_world.data[jy_world_base + i * {dofs} + d] * Y_world.data[sib_row_base + d];
                    for (int offset = 16; offset > 0; offset >>= 1)
                        cross += __shfl_down_sync(MASK, cross, offset);
                    cross = __shfl_sync(MASK, cross, 0);
                    float compliance = angular_compliance(s_mu_dense[i], lambda_n);
                    float2 pair = friction_pair_candidate(denom + compliance, cross, s_diag_dense[sib] + compliance,
                        residual + compliance * old_impulse, sibling_residual + compliance * s_lam_dense[sib],
                        old_impulse, s_lam_dense[sib], radius, omega);
                    float a = pair.x;
                    float b = pair.y;
                    float mag = sqrtf(a * a + b * b);
                    float scale = mag > radius ? radius / mag : 1.0f;
                    new_impulse = a * scale;
                    float sib_delta = b * scale - s_lam_dense[sib];
                    s_lam_dense[sib] = b * scale;
                    if (sib_delta != 0.0f) iteration_changed = 1;
                    __SIB_V_UPDATE__
                }} else {{
                    new_impulse = old_impulse;
                }}
                delta_impulse = new_impulse - old_impulse;"""
    return {"helpers": helpers, "sliding_radius": sliding_radius, "row_block": row_block}
