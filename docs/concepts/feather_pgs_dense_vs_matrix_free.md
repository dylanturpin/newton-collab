<!-- SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# FeatherPGS: Dense vs Matrix-Free

This page is the data-heavy companion to [FeatherPGS](feather_pgs.md). The overview page covers the formulation and mode taxonomy; this page focuses on what the current dense and matrix-free implementations actually materialize, stream, and update in the main PGS kernels.

## Comparison Setup

The two paths share the same articulated dynamics front-end:

$$
\tilde{H} = H + \Delta t K_d + \Delta t^2 K_p,
\qquad
\hat{\dot{q}} = \tilde{H}^{-1} b,
\qquad
Y = \tilde{H}^{-1} J^T.
$$

The main difference is how the contact coupling is presented to the Gauss-Seidel sweep.

Dense path:

$$
C = JY = J \tilde{H}^{-1} J^T,
\qquad
w_i = r_i + \sum_j C_{ij} p_j.
$$

Matrix-free articulated path:

$$
w_i = \mathrm{bias}_i + J_i \dot{q},
\qquad
\dot{q} \leftarrow \dot{q} + Y_i \Delta p_i.
$$

The matrix-free path still depends on the same articulated operator. It does not remove `J` or `Y`; it removes the explicit dense Delassus matrix `C` and reconstructs off-diagonal coupling from the live corrected velocity vector (`v_out` in code).

For free-rigid contacts, the matrix-free path uses a second branch with spatial Jacobians and per-row effective masses:

$$
w_i = mf\_rhs_i + J_i v,
\qquad
v \leftarrow v + MiJt_i \Delta p_i.
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

| Scenario | Preset | World DOFs | Warm contacts | Articulated rows | Free-rigid MF rows | Articulated storage | Free-rigid storage | Row-state scalars |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| `g1_flat` | `fpgs_dense_row` | 49 | 8 | 24 | 0 | `C`: 4.0 KiB | none | `rhs` + `diag` + `impulses`: 0.4 KiB |
| `g1_flat` | `fpgs_matrix_free` | 49 | 8 | 24 | 0 | `J_world` + `Y_world`: 49.0 KiB | none | `rhs` + `diag` + `impulses`: 1.5 KiB |
| `h1_tabletop` | `fpgs_dense_row` | 133 | 229 | 117 | 0 | `C`: 64.0 KiB | none | `rhs` + `diag` + `impulses`: 1.5 KiB |
| `h1_tabletop` | `fpgs_matrix_free` | 133 | 235 | 30 | 102 | `J_world` + `Y_world`: 133.0 KiB | free-rigid `mf_*`: 62.0 KiB | `rhs` + `diag` + `impulses`: 1.5 KiB |

Here "articulated rows" means the rows that stay on the articulation-side solve (`C` in dense mode, `J_world`/`Y_world` in articulated matrix-free mode). "Free-rigid MF rows" means contacts routed into the specialized free-rigid matrix-free branch. The `h1_tabletop` totals do not add up the same across presets because these captures came from separate warmed states: the checked-in dense snapshot has 229 warm contacts, while the matrix-free snapshot has 235. Routing is part of the difference, but warm-state contact count also changed between captures.

Three points matter in practice:

1. `g1_flat` is mostly a control. The matrix-free preset allocates its larger world-indexed buffers but gets no free-rigid row-routing benefit because the warm contact state has zero matrix-free rows.
2. `h1_tabletop` is the mixed-world case that explains the current result. Dense keeps all 117 articulated rows inside the explicit Delassus system, while matrix-free leaves only 30 articulated rows on that side and routes 102 rows through the free-rigid branch.
3. Matrix-free is not "smaller everywhere." In `h1_tabletop`, `J_world + Y_world` alone is larger than dense `C`. The benefit comes from avoiding dense mixed-world coupling and from how the solve kernel uses that data, not from a universal byte-count reduction.

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

This is the main reason the matrix-free path can perform well without dedicating shared memory to a Delassus tile: shared memory is spent on the velocity state, while coupling is recovered through `J_i v` and `Y_i \Delta p_i`.

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

## Why This Is Not Only A "Rigid MF" Story

The immediate objection to the `h1_tabletop` table is fair: most of the visible row-count change comes from the 102 free-rigid rows that leave the articulated solve, so maybe the result says only "the rigid matrix-free branch helps" rather than "the dense articulated solve is the wrong long-term center."

The current codebase already contains the most relevant counterexample: `split`. That path keeps the articulated rows on the dense Delassus side and sends only free-rigid rows through matrix-free. So if the whole story were "just route rigid contacts away," `split` would already match the fused `matrix_free` path.

The current implementation does not let it do that cheaply. In `split` mode, each PGS iteration interleaves:

- a dense articulated solve kernel (`pgs_solve_tiled_contact_*`);
- dense impulse application back into world velocity (`apply_impulses_world_*`);
- a right-hand-side refresh (`rhs_accum_world_*`);
- a separate free-rigid matrix-free solve (`pgs_solve_mf_*`);
- another correction accumulation step before the next dense iteration.

That means `split` still reloads the dense articulated working set every iteration while also paying for a separate free-rigid kernel launch sequence.

The fetched `origin/gh-pages` profiler artifacts show this directly for `h1_tabletop` on the RTX 5090 run `nightly/runs/full-3gpu-20260309f`:

- `split` trace `.../h1_tabletop_ablation_5090__0007--h1-tabletop--split--s8--n8192.trace.json` spends about 7.33 s of measured kernel time in solver-related kernels.
- Within that window, `split` spends about 2.41 s in `pgs_solve_mf_*`, 0.89 s in the dense `pgs_solve_tiled_contact_*`, 1.34 s in repeated `rhs_accum_world_*`, and 0.42 s in repeated `apply_impulses_world_*`.
- `matrix_free` trace `.../h1_tabletop_ablation_5090__0008--h1-tabletop--matrix-free--s8--n8192.trace.json` spends about 3.46 s in solver-related kernels over the same measured window.
- In that fused path, the dominant solve kernel is `pgs_solve_mf_gs_*` at about 1.27 s total, and the repeated `rhs_accum_world_*` / `apply_impulses_world_*` pair from `split` disappears from the hot loop.

So the evidence is narrower than "articulated matrix-free alone already wins," but stronger than "only rigid routing matters." The current fused path wins because it combines the articulated velocity-path update with the free-rigid matrix-free branch inside one solve kernel family, instead of alternating dense and matrix-free launches around a refreshed articulated right-hand side.

This also explains why `split` remains a useful diagnostic path but not a convincing end state. It isolates the free-rigid routing idea, but it does not isolate it under the same kernel structure as `matrix_free`.

## Fused-Kernel Follow-Up

The remaining open question is whether a denser fused hybrid could recover the best of both worlds:

- keep explicit dense Delassus coupling for articulated rows;
- keep free-rigid rows on the matrix-free branch;
- fuse the two phases tightly enough that the relaunch and right-hand-side refresh costs seen in `split` disappear.

That is plausible in principle, but the occupancy tradeoff is real. The dense tiled-row kernel wants shared memory for the packed `C` tile, while the current fused matrix-free kernel benefits from spending shared memory on the live velocity vector and running at high occupancy. Combining both could reduce residency enough to erase the benefit.

The current checked-in profiler artifacts do not answer that occupancy question. They show timing and launch structure, not achieved occupancy. There is also an anecdotal observation from prior local experiments that artificially lowering matrix-free occupancy with an unused shared-memory allocation did not change throughput, but that result is not checked in and should not be treated as established evidence. The right next experiment is a profiler-backed retry of a fused dense+matrix-free kernel with explicit occupancy measurements.

## Code-Path Ablation Recommendation

The current codebase still carries several FeatherPGS solve variants, but the checked-in data narrows the decision materially.

Artifact references:

- [Scenario sizing: `g1_flat` dense rows](https://github.com/dylanturpin/newton-collab/blob/feather_pgs/.agent/data/fpgs-matrix-free-dense-explainer/scenarios/g1_flat/fpgs_dense_row.json)
- [Scenario sizing: `g1_flat` matrix-free](https://github.com/dylanturpin/newton-collab/blob/feather_pgs/.agent/data/fpgs-matrix-free-dense-explainer/scenarios/g1_flat/fpgs_matrix_free.json)
- [Scenario sizing: `h1_tabletop` dense rows](https://github.com/dylanturpin/newton-collab/blob/feather_pgs/.agent/data/fpgs-matrix-free-dense-explainer/scenarios/h1_tabletop/fpgs_dense_row.json)
- [Scenario sizing: `h1_tabletop` matrix-free](https://github.com/dylanturpin/newton-collab/blob/feather_pgs/.agent/data/fpgs-matrix-free-dense-explainer/scenarios/h1_tabletop/fpgs_matrix_free.json)
- [Kernel memory analysis: dense tiled-row](https://github.com/dylanturpin/newton-collab/blob/feather_pgs/.agent/data/fpgs-matrix-free-dense-explainer/kernels/dense-row.json)
- [Kernel memory analysis: fused matrix-free GS](https://github.com/dylanturpin/newton-collab/blob/feather_pgs/.agent/data/fpgs-matrix-free-dense-explainer/kernels/matrix-free-gs.json)

| Code path | Recommendation | Reason |
| --- | --- | --- |
| `dense_loop` | REMOVE | It keeps the full explicit Delassus formulation but does not offer a compelling maintenance or performance role relative to the tiled dense path. |
| `dense_row` | DEPRECATE | It is still useful as the clearest explicit-operator baseline, but `h1_tabletop` shows the mixed-world cost of keeping all 117 active rows inside dense `C`. |
| `dense_streaming` | DEPRECATE | It is the only remaining dense variant with a plausible optimization story, but the current evidence still points to matrix-free as the cleaner long-term path. |
| `split` | REMOVE | It preserves branch complexity without changing the main conclusion: the matrix-free fused path is the version that explains the observed win. |
| `matrix_free` | KEEP | It matches the current mixed-world result, keeps free-rigid rows out of the articulated dense system, and already has implementation proof from the private API cleanup branch. |

Three observations drive the recommendation:

1. The scenario data says the important case is not `g1_flat`, where both presets warm to the same 24 dense rows, but `h1_tabletop`, where dense keeps 117 active rows in `C` while matrix-free keeps only 30 articulated dense rows and routes 102 rows through the free-rigid matrix-free branch.
2. The kernel analysis says the dense path spends its hot working set on staged Delassus tiles, while the fused matrix-free path spends it on the live velocity vector plus streamed `J` and `Y` data. That is the implementation-level reason the current matrix-free path performs well without a shared-memory Delassus tile.
3. The private API cleanup branch `dturpin/fpgs-private-api-matrix-free` already collapsed the public FeatherPGS surface to matrix-free-only. That does not prove every private dense kernel should disappear immediately, but it does prove the main user-facing collapse is feasible.

This recommendation is intentionally orthogonal to the velocity spike investigation. The spike study on branch `dt/velocity-spike-claude` found that the dominant spike class is unconstrained `v_hat`, not PGS divergence, and that PGS converges within 8 iterations on the reproduced traces. That means the dense-vs-matrix-free code-path choice should be made on solver complexity, kernel behavior, and mixed-world cost, not on the spike symptom itself.

For a legacy branch, keep only the minimum dense baselines needed for regression checks and equivalence debugging:

- keep `dense_row` as the explicit Delassus reference path;
- keep `dense_streaming` only if a follow-up performance study still needs it;
- drop `dense_loop` and `split` first, because they add branch surface without carrying distinct decision value in the current evidence set.

## Open Questions

- The current evidence explains why the fused matrix-free kernel is competitive against the dense tiled-row kernel, but it does not yet fully quantify where the dense streaming variant closes or fails to close the gap.
- The current checked-in scenarios are single-world captures. A later scaling pass may still be useful to show how the same storage choices behave as `num_worlds` grows.
- `g1_flat` shows that matrix-free can carry extra world-indexed storage without a routing benefit. The remaining performance question is exactly when the free-rigid row split is large enough to dominate those extra buffers.
