<!-- SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# FeatherPGS

```{important}
For skimming: FeatherPGS used to admit a clean "this is a Delassus solve" description. That is no longer quite true. We now have three closely related paths: `dense`, `split`, and `matrix_free`. `dense` is still the old explicit Delassus story. `split` is "Delassus for articulated rows, matrix-free for free rigid contacts." `matrix_free` is not a different articulation model; it is still the same reduced-coordinate machinery, but with the solve written more directly as repeated velocity-space updates rather than as sweeps over a fully assembled dense `J \tilde{H}^{-1} J^T`. The implementation is faster this way, but the conceptual packaging is less tidy than it used to be. The clean long-term direction is probably to present this as one Featherstone solver with pluggable constraint handling, rather than as a collection of solver names that sound more different than they really are.
```

## Motivation and Background

FeatherPGS is Newton's reduced-coordinate rigid-body contact solver built around Featherstone-style articulated dynamics, a semi-implicit step, and a projected Gauss-Seidel (PGS) contact solve. The core formulation uses a Schur-complement approach:

$$
\tilde{H} = H + \Delta t K_d + \Delta t^2 K_p,
\qquad
C = J \tilde{H}^{-1} J^T,
$$

followed by a PGS solve in impulse space. This describes the `dense` operating mode and is the most direct way to explain the articulation-side math. The other modes (`split`, `matrix_free`) keep the same reduced-coordinate foundation but avoid materializing the full dense Delassus operator when that cost dominates — especially in scenes mixing large articulated systems with many free-rigid-body contacts.

## Problem Setup and Notation

Let $q \in \mathbb{R}^n$ be generalized coordinates, $\dot{q}$ generalized velocities, and $\ddot{q}$ generalized accelerations for one articulation. FeatherPGS works in reduced coordinates internally, but it still maintains maximal-coordinate body state because collision detection and downstream consumers use body poses and spatial velocities.

At a high level the solver step is:

1. evaluate forward kinematics and inverse dynamics terms;
2. build the articulated mass operator and fold implicit PD terms into it;
3. compute an unconstrained velocity predictor $\hat{v}$;
4. build contact and optional joint-limit rows;
5. solve a unilateral/frictional contact problem with PGS;
6. recover accelerations, integrate, and update maximal state.

The current code uses `v_hat` and `v_out` for the unconstrained and corrected generalized velocities respectively; see [`solver_feather_pgs.py`](https://github.com/dylanturpin/newton-collab/blob/feather_pgs/newton/_src/solvers/feather_pgs/solver_feather_pgs.py). In compact form:

$$
\hat{\dot{q}} = \tilde{H}^{-1} b,
\qquad
\dot{q}_{t+1} = \hat{\dot{q}} + \tilde{H}^{-1} J^T p.
$$

This matches the dense path. In the `split` and `matrix_free` paths, the same objects are present, but the solver avoids forming the full dense matrix $C$.

## Frame Conventions in FeatherPGS

FeatherPGS currently uses a mixed velocity convention, and it is worth stating that explicitly because it is easy to misread the code otherwise.

- At the public API boundary, free-root `joint_qd` follows the CoM convention: the linear term is the root body's center-of-mass velocity.
- Inside the articulated solve, stages 1 through 6 use the root-body-origin linear term instead. On entry, FeatherPGS converts

$$
v_{\mathrm{local}} = v_{\mathrm{com}} - \omega \times r_{\mathrm{com}},
$$

where $r_{\mathrm{com}}$ is the root-body COM offset in world orientation. In the current code this is the `convert_root_free_qd_world_to_local()` / `convert_root_free_qd_local_to_world()` pair, using `articulation_root_com_offset` rather than `articulation_origin`.
- The internal Featherstone algebra is otherwise unchanged: inverse dynamics, CRBA, Cholesky, triangular solve, and the articulated PGS paths all run on that internal root-origin linear term.
- At stage 7, integration uses the public CoM-based `joint_qd` and `joint_qdd`, but FREE and DISTANCE joints reconstruct origin translation velocity before integrating position. In other words, position update uses

$$
v_{\mathrm{origin}} = v_{\mathrm{com}} - \omega \times r_{\mathrm{com,world}},
$$

while the stored velocity after integration remains CoM-based.
- FK writeback then restores the public maximal-coordinate contract: `body_qd` stores CoM velocity again.

There is one additional convention detail for contacts. `articulation_origin` is currently the root-body CoM world position, not the articulation frame origin. Dense articulated contact rows are still assembled against the articulated generalized coordinates in the usual way, but the free-rigid matrix-free rows use contact offsets relative to that root CoM world point. This is why the current code can keep the internal articulated math unchanged while still building free-rigid Jacobian rows in the expected CoM-centered form.

This matches the direction of upstream PR 2206, which keeps Featherstone internal math unchanged and makes the CoM convention explicit at the public boundary.

## Dense-Delassus View

The dense path is a straightforward Schur-complement solver:

$$
0 \le p \perp (Cp + r) \ge 0,
\qquad
C = J \tilde{H}^{-1} J^T.
$$

In implementation terms, this corresponds to the dense path:

- Stage 1 computes articulated kinematics and dynamics terms.
- Stage 2 factorizes $\tilde{H}$ with Cholesky.
- Stage 3 solves for the predictor velocity $\hat{v}$.
- Stage 4 builds contact Jacobian rows, computes $Y = \tilde{H}^{-1} J^T$, and then assembles the Delassus operator.
- Stage 5/6 runs PGS on the explicit dense contact system.

The dense branch in `step()` follows this structure:

- `_stage4_build_rows(...)` builds the world-level constraint rows;
- `_stage4_hinv_jt_*` computes $Y = \tilde{H}^{-1} J^T$;
- `_stage4_delassus_*` assembles $C = JY$;
- `_stage4_compute_rhs_world(...)` and `_stage4_accumulate_rhs_world(...)` form the right-hand side;
- `_dispatch_dense_pgs_solve(...)` runs the chosen dense PGS kernel;
- `_stage6_apply_impulses_world(...)` applies the resulting correction back to generalized velocity.

## The Matrix-Free Path

In the current code, `pgs_mode="matrix_free"` does **not** mean "no Jacobians are stored" and it does **not** mean "the solver works only in body-space without articulation information." Instead it means:

- compute $Y = \tilde{H}^{-1} J^T$ as before;
- compute only the diagonal of $JY$ with `diag_from_JY_par_art`;
- gather `J` and `Y` into world-indexed buffers (`J_world`, `Y_world`);
- keep `rhs` as a bias term rather than baking in $J \hat{v}$ once;
- during PGS, recompute the current $Jv$ from the live velocity vector each iteration instead of consulting a preassembled dense matrix.

So the shift is from an explicit Delassus matrix to an explicit velocity path:

$$
\text{dense: } w_i = r_i + \sum_j C_{ij} p_j,
\qquad
\text{current matrix-free direction: } w_i = \text{bias}_i + J_i v.
$$

The second form avoids storing and streaming the full $C$ tensor. It still uses the same articulated dynamics factorization and the same reduced-coordinate Jacobians, but the per-iteration working set is smaller and better aligned with the actual dependency structure of the solve.

### Specialized matrix-free handling for free rigid bodies

For contacts whose two sides are free rigid bodies or ground, the solver builds spatial Jacobian rows directly against the free body state using [`build_mf_contact_rows`](https://github.com/dylanturpin/newton-collab/blob/feather_pgs/newton/_src/solvers/feather_pgs/kernels.py), computes per-body $H^{-1}$ from the spatial inertia blocks with [`compute_mf_body_Hinv`](https://github.com/dylanturpin/newton-collab/blob/feather_pgs/newton/_src/solvers/feather_pgs/kernels.py), and forms only per-row effective masses and bias terms with [`compute_mf_effective_mass_and_rhs`](https://github.com/dylanturpin/newton-collab/blob/feather_pgs/newton/_src/solvers/feather_pgs/kernels.py). In that free-rigid branch the row offsets are taken relative to the root-body CoM world point stored in `articulation_origin`, not a separate articulation-frame origin.

This avoids both:

- dense world-level Delassus assembly, and
- articulated gather/scatter through a larger joint-space contact operator

for the subset of contacts that do not need it.

The resulting free-rigid solve uses live velocity updates inside the PGS sweep (`pgs_solve_mf_loop` in the simple path, or the fused two-phase tiled kernel in the main `matrix_free` path).

## Current Solver Architecture

The current implementation is best understood as a shared articulation pipeline followed by one of several contact-system realizations.

```{mermaid}
flowchart TD
    A[Shared stages 1-3<br/>articulation dynamics, factorization, v_hat] --> B[Stage 4 row build]
    B --> C{Contact structure and pgs_mode}
    C --> D[dense<br/>assemble J H^-1 J^T for all rows]
    C --> E[split<br/>articulated rows dense<br/>free rigid rows matrix-free]
    C --> F[matrix_free<br/>articulated rows from J/Y/diag<br/>free rigid rows matrix-free]
    D --> G[Dense PGS sweep]
    E --> H[Interleaved dense plus free-rigid MF sweep]
    F --> I[Velocity-path articulated PGS plus free-rigid MF sweep]
    G --> J[Apply impulses and integrate]
    H --> J
    I --> J
```

### Shared articulation pipeline

The front half of the solver is common across modes:

- `eval_rigid_fk`, `eval_rigid_id`, and CRBA-style mass construction produce the reduced-coordinate dynamics terms.
- The effective operator includes implicit drive terms, so joint PD stays in the operator rather than appearing as extra constraint rows.
- Cholesky and triangular solves are selected per articulation size group, with loop and tiled kernels available.
- `compute_velocity_predictor` produces `v_hat`, the unconstrained generalized velocity used as the base point for contact resolution.

The main divergence between modes happens after row construction.

### Contact-row construction and path split

The row builder classifies contacts before they enter the solve. The key classifier is [`allocate_world_contact_slots`](https://github.com/dylanturpin/newton-collab/blob/feather_pgs/newton/_src/solvers/feather_pgs/kernels.py), which assigns each contact either:

- `contact_path == 0`: dense articulated/world path;
- `contact_path == 1`: matrix-free free-rigid path;
- `contact_path == -1`: skipped or overflowed.

The current heuristic is structural rather than adaptive: if both sides of a contact are free rigid bodies or ground, the contact can be routed to the cheaper matrix-free path. Contacts involving nontrivial articulations stay on the dense articulated path. This is why the code talks about "free rigid bodies" separately from general articulations: a single free root body admits a much cheaper $6 \times 6$ effective-mass treatment than an arbitrary articulated subtree.

Joint limits are also handled as unilateral rows when enabled, but they are still built into the dense articulated row set rather than the free-rigid matrix-free path.

### Mixed worlds and the current compromise

The code also supports mixed worlds, where some rows belong to articulated contacts and others belong to free-rigid contacts. In `split` mode with mixed contacts, the solver interleaves one dense iteration with one matrix-free iteration and updates the dense right-hand side with the incremental free-rigid velocity correction. This is not the cleanest mathematical presentation, but it is faithful to the current implementation and it avoids forcing all contacts into the more expensive dense representation.

## Mode Taxonomy

The core solver exposes three `pgs_mode` values:

- `dense`: build the full Delassus operator and solve the entire contact system in the dense articulated path.
- `split`: use the dense path for articulated contacts and the specialized matrix-free path for contacts involving only free rigid bodies or ground.
- `matrix_free`: do not assemble the full Delassus operator for articulated rows; solve articulated rows from `J`, `Y`, and the diagonal while also using the free-rigid matrix-free path where available.

These are defined on [`SolverFeatherPGS`](https://github.com/dylanturpin/newton-collab/blob/feather_pgs/newton/_src/solvers/feather_pgs/solver_feather_pgs.py).

## Mapping From Math to Code

### Articulated operator and predictor

- $\tilde{H}$: grouped per-articulation matrices in `H_by_size`, factorized into `L_by_size`.
- $\hat{v}$ / $\hat{\dot{q}}$: `v_hat`, produced after `_stage3_trisolve_*` and `compute_velocity_predictor`.
- corrected velocity $v_{t+1}$: `v_out`.

Relevant code paths:

- [`solver_feather_pgs.py`](https://github.com/dylanturpin/newton-collab/blob/feather_pgs/newton/_src/solvers/feather_pgs/solver_feather_pgs.py)
- [`kernels.py`](https://github.com/dylanturpin/newton-collab/blob/feather_pgs/newton/_src/solvers/feather_pgs/kernels.py)

### Contact Jacobians and metadata

- $J$: grouped dense articulated rows in `J_by_size`; gathered world form in `J_world` for the matrix-free articulated path.
- row metadata: `row_type`, `row_parent`, `row_mu`, `row_beta`, `row_cfm`, `phi`, `target_velocity`.
- free-rigid contact Jacobians: `mf_J_a`, `mf_J_b`.

Relevant row builders:

- `allocate_world_contact_slots`
- `populate_world_J_for_size`
- `populate_joint_limit_J_for_size`
- `build_mf_contact_rows`

### $Y = \tilde{H}^{-1} J^T$, Delassus, and diagonal terms

- $Y$: `Y_by_size`, or world-gathered `Y_world` in the articulated matrix-free path.
- dense $C = JY$: `C`.
- diagonal-only articulated approximation for the matrix-free path: `diag`, built by `diag_from_JY_par_art` plus regularization/compliance.
- free-rigid effective mass inverse: `mf_eff_mass_inv`.

Relevant kernels:

- `_stage4_hinv_jt_tiled`
- `_stage4_hinv_jt_par_row`
- `_stage4_delassus_tiled`
- `_stage4_delassus_par_row_col`
- `diag_from_JY_par_art`
- `compute_mf_effective_mass_and_rhs`

### Right-hand side and PGS state

- dense right-hand side $r$: `rhs`, built from bias terms and optionally accumulated $J \hat{v}$.
- dense impulses $p$: `impulses`.
- free-rigid impulses: `mf_impulses`.

One detail that matters for interpreting the current code: in dense mode the solver builds `rhs = bias + J v_hat`, while in the articulated `matrix_free` path it keeps only the bias in `rhs` and recomputes $Jv$ against the live `v_out` inside the PGS loop.

## Practical Features

### Implicit drives instead of explicit motor constraints

FeatherPGS keeps joint PD in the reduced-coordinate operator rather than inflating the constraint system with additional solver rows for ordinary drives.

### Joint limits as on-demand unilateral rows

Joint limits remain optional and row-based. When `enable_joint_limits=True`, violated limits allocate unilateral rows and enter the dense articulated solve with their own Jacobian and bias terms. They are not part of the free-rigid matrix-free specialization.
