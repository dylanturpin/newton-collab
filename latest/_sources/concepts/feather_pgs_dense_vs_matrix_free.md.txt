<!-- SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# FeatherPGS: Dense vs Matrix-Free

This page is the data-heavy companion to [FeatherPGS](feather_pgs.md). The overview page covers the formulation and mode taxonomy; this page focuses on what the current dense and matrix-free implementations actually materialize, stream, and update in the main PGS kernels.

## Comparison Setup

The two paths share the same articulated dynamics front-end:

$$
\tilde{H} = H + \Delta t K_d + \Delta t^2 K_p,
\qquad
\hat{v} = \tilde{H}^{-1} b,
\qquad
Y = \tilde{H}^{-1} J^T.
$$

The main difference is how the contact coupling is presented to the Gauss-Seidel sweep.

Dense path:

$$
C = JY = J \tilde{H}^{-1} J^T,
\qquad
w_i = rhs_i + \sum_j C_{ij}\lambda_j.
$$

Matrix-free articulated path:

$$
w_i = rhs_i + J_i v,
\qquad
v \leftarrow v + Y_i \Delta \lambda_i.
$$

The matrix-free path still depends on the same articulated operator. It does not remove `J` or `Y`; it removes the explicit dense Delassus matrix `C` and reconstructs off-diagonal coupling from the live velocity vector.

For free-rigid contacts, the matrix-free path uses a second branch with spatial Jacobians and per-row effective masses:

$$
w_i = mf\_rhs_i + J_i v,
\qquad
v \leftarrow v + MiJt_i \Delta \lambda_i.
$$

## Logical Work Per Path

| Path | Materialized arrays | Per-row update | Main implication |
| --- | --- | --- | --- |
| Dense `fpgs_dense_row` | `J`, `Y`, dense `C`, `rhs`, `diag`, `impulses` | `w_i = rhs_i + \sum_j C_{ij}\lambda_j` | Pays the up-front `C` build and storage cost, then reuses staged tiles of `C` inside the sweep. |
| Matrix-free articulated phase | `J_world`, `Y_world`, `diag`, `rhs`, `impulses`, live `v_out` | `w_i = rhs_i + J_i v`; then `v += Y_i \Delta\lambda_i` | Avoids storing dense `C`, but still stores world-indexed Jacobian and `H^{-1}J^T` rows. |
| Matrix-free free-rigid phase | `mf_J_*`, `mf_MiJt_*`, `mf_meta_packed`, `mf_impulses` | `w_i = mf_rhs_i + J_i v`; then `v += MiJt_i \Delta\lambda_i` | Keeps mixed free-rigid contacts out of the articulated dense system. |

## Scenario-Backed Sizing

Two checked-in benchmark scenarios anchor the comparison:

- `g1_flat`: a control case with one articulation on flat ground.
- `h1_tabletop`: a mixed world with articulated contacts plus many free-rigid tabletop contacts.

| Scenario | Preset | World DOFs | Active dense rows | Active MF rows | Key stored objects |
| --- | --- | ---: | ---: | ---: | --- |
| `g1_flat` | `fpgs_dense_row` | 49 | 24 | 0 | `C`: 4096 B; `rhs` + `diag` + `impulses`: 384 B |
| `g1_flat` | `fpgs_matrix_free` | 49 | 24 | 0 | `J_world` + `Y_world`: 50176 B; scalar row state: 1536 B |
| `h1_tabletop` | `fpgs_dense_row` | 133 | 117 | 0 | `C`: 65536 B; `rhs` + `diag` + `impulses`: 1536 B |
| `h1_tabletop` | `fpgs_matrix_free` | 133 | 30 | 102 | `J_world` + `Y_world`: 136192 B; free-rigid `mf_*`: 63488 B |

Three points matter in practice:

1. `g1_flat` is mostly a control. The matrix-free preset allocates its larger world-indexed buffers but gets no free-rigid row-routing benefit because the warm contact state has zero matrix-free rows.
2. `h1_tabletop` is the mixed-world case that explains the current result. Dense keeps all 117 active rows inside the explicit Delassus system, while matrix-free leaves only 30 articulated rows on that side and routes 102 rows through the free-rigid branch.
3. Matrix-free is not "smaller everywhere." In `h1_tabletop`, `J_world + Y_world` is larger than dense `C`. The benefit comes from avoiding dense mixed-world coupling and from how the solve kernel uses that data, not from a universal byte-count reduction.

## What The Kernels Actually Stage

### Dense tiled-row PGS

The dense tiled-row kernel is a classic explicit-operator Gauss-Seidel sweep:

- global memory holds the full dense `C`, plus `rhs`, `diag`, impulse state, and row metadata;
- shared memory stages the packed lower triangle of `C` as `s_Ctri`, plus `s_lam`, `s_rhs`, `s_diag`, and row metadata;
- registers hold the lane-local dot-product accumulators and packed-triangle indices.

| Storage class | Dominant payload | Why it matters |
| --- | --- | --- |
| Global | `C[world, M, M]`, `rhs`, `diag`, warm-start impulses | The kernel still requires the full Delassus matrix to be built before the sweep begins. |
| Shared | packed lower triangle of `C`, scalar row data, row metadata | Shared memory removes repeated global reads of `C` during the sweep. |
| Registers | `w_i`, `\Delta \lambda_i`, dot-product partial sums | Each row still recomputes the full `\sum_j C_{ij}\lambda_j` from staged `C`. |

The important tradeoff is that shared memory helps only after the explicit Delassus matrix already exists.

### Fused articulated plus free-rigid matrix-free GS

The matrix-free kernel stages a different object: the live velocity vector.

- global memory streams `J_world` and `Y_world` for articulated rows, then `mf_J_*` and `mf_MiJt_*` for free-rigid rows;
- shared memory keeps `s_v`, the live world velocity vector, plus articulated and free-rigid impulse state;
- registers hold prefetched `J`/`Y` lane slices, row residuals, and decoded packed metadata.

| Storage class | Dominant payload | Why it matters |
| --- | --- | --- |
| Global | `J_world`, `Y_world`, `mf_J_*`, `mf_MiJt_*`, row scalars | The kernel streams row data instead of reading a preassembled dense `C`. |
| Shared | live `v_out` slice, dense row scalars, `mf` impulse state | One shared velocity vector couples the articulated and free-rigid phases. |
| Registers | prefetched `J`/`Y` fragments, `J_i v`, `\Delta \lambda_i` | Off-diagonal couplings are reconstructed on the fly. |

This is the main reason the matrix-free path can perform well without dedicating shared memory to a Delassus tile: shared memory is spent on the velocity state, while coupling is recovered through `J_i v` and `Y_i \Delta \lambda_i`.

## Why `h1_tabletop` Favors Matrix-Free

`h1_tabletop` is the clearest current example because it exercises both phases of the matrix-free kernel:

- the dense preset keeps all 117 active rows in one explicit world-level Delassus system;
- the matrix-free preset keeps only 30 articulated rows in the articulated phase;
- 102 rows move to the free-rigid path, which never enters the dense articulated `C` build.

That shift changes more than storage:

- dense pays for `J`, `Y`, and `C`, then performs row updates against staged `C`;
- matrix-free pays for `J_world` and `Y_world`, but updates the live velocity vector directly and lets the free-rigid rows stay out of the articulated dense system;
- the kernel no longer needs a shared-memory copy of a dense mixed-world operator.

So the current observation is not "matrix-free wins because it uses less memory" and not "matrix-free wins because shared memory disappears." The better explanation is narrower: the current matrix-free implementation avoids materializing dense mixed-world coupling and spends its hot working set on `v`, scalar row state, and streamed Jacobian data instead.

## Control Case: `g1_flat`

`g1_flat` matters because it prevents over-claiming:

- both presets warm to the same 24 dense rows and 49 world DOFs;
- the matrix-free preset has no free-rigid rows to route away from the articulated phase;
- it still allocates `J_world` and `Y_world`, so this case is mostly a sanity check for equivalence rather than the scenario that motivates the matrix-free path.

This is why the explainer uses `g1_flat` as a control and `h1_tabletop` as the realistic mixed-world sizing example.

## Open Questions

- The current evidence explains why the fused matrix-free kernel is competitive against the dense tiled-row kernel, but it does not yet fully quantify where the dense streaming variant closes or fails to close the gap.
- The current checked-in scenarios are single-world captures. A later scaling pass may still be useful to show how the same storage choices behave as `num_worlds` grows.
- `g1_flat` shows that matrix-free can carry extra world-indexed storage without a routing benefit. The remaining performance question is exactly when the free-rigid row split is large enough to dominate those extra buffers.
