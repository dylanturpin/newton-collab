<!-- SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# FeatherPGS: Dense vs Matrix-Free

This page is the data-heavy companion to [FeatherPGS](feather_pgs.md). The overview page covers the formulation and mode taxonomy; this page focuses on what the current dense and matrix-free implementations actually materialize, stream, and update in the main PGS kernels.

## Comparison Setup

The two paths share the same articulated dynamics front-end:

$$
\tilde{H} = H + \Delta t\, K_d + \Delta t^2\, K_p,
\qquad
\hat{\dot{q}} = \tilde{H}^{-1}\, b,
\qquad
Y = \tilde{H}^{-1}\, J^T.
$$

The main difference is how the contact coupling is presented to the Gauss-Seidel sweep.

Dense path:

$$
C = J\, \tilde{H}^{-1}\, J^T,
\qquad
w_i = r_i + \sum_j C_{ij}\, p_j.
$$

Matrix-free articulated path:

$$
w_i = \text{bias}_i + J_i\, \dot{q},
\qquad
\dot{q} \leftarrow \dot{q} + Y_i\, \Delta p_i.
$$

The matrix-free path still depends on the same articulated operator. It does not remove $J$ or $Y$; it removes the explicit dense Delassus matrix $C$ and reconstructs off-diagonal coupling from the live corrected velocity vector (`v_out` in code).

For free-rigid contacts, the matrix-free path uses a second branch with spatial Jacobians and per-row effective masses:

$$
w_i = r_i^{\text{mf}} + J_i\, v,
\qquad
v \leftarrow v + M_i^{-1}\, J_i^T\, \Delta p_i.
$$

## Logical Work Per Path

| Path | Materialized arrays | Per-row update | Main implication |
| --- | --- | --- | --- |
| Dense `fpgs_dense_row` | $J$, $Y$, dense $C$, `rhs`, `diag`, `impulses` | $w_i = r_i + \sum_j C_{ij}\, p_j$ | Pays the up-front $C$ build and storage cost, then reuses staged tiles of $C$ inside the sweep. |
| Matrix-free articulated phase | `J_world`, `Y_world`, `diag`, `rhs`, `impulses`, live `v_out` | $w_i = r_i + J_i\, v$; then $v \mathrel{+}= Y_i\, \Delta p_i$ | Avoids storing dense $C$, but still stores world-indexed Jacobian and $\tilde{H}^{-1} J^T$ rows. |
| Matrix-free free-rigid phase | `mf_J_*`, `mf_MiJt_*`, `mf_meta_packed`, `mf_impulses` | $w_i = r_i^{\text{mf}} + J_i\, v$; then $v \mathrel{+}= M_i^{-1}\, J_i^T\, \Delta p_i$ | Keeps free-rigid contacts out of the articulated dense system. |

## Scenario-Backed Sizing

Two checked-in benchmark scenarios anchor the comparison: `g1_flat` (one articulation on flat ground, used as a control) and `h1_tabletop` (articulated contacts plus many free-rigid tabletop contacts, the mixed-world case).

**Articulated-side storage**

| Scenario | Preset | World DOFs | Warm contacts | Art. rows | MF rows | Storage |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `g1_flat` | `dense_row` | 49 | 8 | 24 | 0 | $C$: 4.0 KiB |
| `g1_flat` | `matrix_free` | 49 | 8 | 24 | 0 | $J_w + Y_w$: 49.0 KiB |
| `h1_tabletop` | `dense_row` | 133 | 229 | 117 | 0 | $C$: 64.0 KiB |
| `h1_tabletop` | `matrix_free` | 133 | 235 | 30 | 102 | $J_w + Y_w$: 133.0 KiB |

**Free-rigid and row-state storage**

| Scenario | Preset | Free-rigid storage | Row-state scalars |
| --- | --- | --- | --- |
| `g1_flat` | `dense_row` | none | 0.4 KiB |
| `g1_flat` | `matrix_free` | none | 1.5 KiB |
| `h1_tabletop` | `dense_row` | none | 1.5 KiB |
| `h1_tabletop` | `matrix_free` | `mf_*`: 62.0 KiB | 1.5 KiB |

The `h1_tabletop` totals differ across presets because these captures came from separate warmed states (229 vs 235 warm contacts). Routing differences and warm-state contact count both contribute.

Three points matter in practice:

1. `g1_flat` is a control. Matrix-free allocates its world-indexed buffers but gets no free-rigid routing benefit (zero MF rows).
2. `h1_tabletop` is the mixed-world case. Dense keeps all 117 articulated rows inside the explicit Delassus system; matrix-free leaves only 30 articulated rows and routes 102 through the free-rigid branch.
3. Matrix-free is not smaller everywhere. In `h1_tabletop`, $J_w + Y_w$ alone is larger than dense $C$. The benefit comes from avoiding dense mixed-world coupling and from how the solve kernel uses that data, not from a universal byte-count reduction.

## What The Kernels Actually Stage

### Dense tiled-row PGS

The dense tiled-row kernel is a classic explicit-operator Gauss-Seidel sweep:

- global memory holds the full dense $C$, plus `rhs`, `diag`, impulse state, and row metadata;
- shared memory stages the packed lower triangle of $C$ as `s_Ctri`, plus `s_lam`, `s_rhs`, `s_diag`, and row metadata;
- registers hold the lane-local dot-product accumulators and packed-triangle indices.

| Storage class | Dominant payload | Why it matters |
| --- | --- | --- |
| Global | $C[\text{world}, M, M]$, `rhs`, `diag`, warm-start impulses | The kernel requires the full Delassus matrix to be built before the sweep begins. |
| Shared | packed lower triangle of $C$, scalar row data, row metadata | Shared memory removes repeated global reads of $C$ during the sweep. |
| Registers | $w_i$, $\Delta p_i$, dot-product partial sums | Each row recomputes $\sum_j C_{ij}\, p_j$ from staged $C$. |

The important tradeoff is that shared memory helps only after the explicit Delassus matrix already exists.

### Fused articulated plus free-rigid matrix-free GS

The matrix-free kernel stages a different object: the live velocity vector.

- global memory streams `J_world` and `Y_world` for articulated rows, then `mf_J_*` and `mf_MiJt_*` for free-rigid rows;
- shared memory keeps `s_v`, the live world velocity vector, plus articulated and free-rigid impulse state;
- registers hold prefetched $J$/$Y$ lane slices, row residuals, and decoded packed metadata.

| Storage class | Dominant payload | Why it matters |
| --- | --- | --- |
| Global | `J_world`, `Y_world`, `mf_J_*`, `mf_MiJt_*`, row scalars | The kernel streams row data instead of reading a preassembled dense $C$. |
| Shared | live `v_out` slice, dense row scalars, `mf` impulse state | One shared velocity vector couples the articulated and free-rigid phases. |
| Registers | prefetched $J$/$Y$ fragments, $J_i\, v$, $\Delta p_i$ | Off-diagonal couplings are reconstructed on the fly. |

This is the main reason the matrix-free path can perform well without dedicating shared memory to a Delassus tile: shared memory is spent on the velocity state, while coupling is recovered through $J_i\, v$ and $Y_i\, \Delta p_i$.

## Measured Performance: `split` vs `matrix_free`

The `split` path keeps articulated rows on the dense Delassus side and sends only free-rigid rows through matrix-free. If the entire benefit came from routing rigid contacts away from the dense system, `split` would already match fused `matrix_free`. It does not, and the profiler data shows why.

In `split` mode, each PGS iteration interleaves:

- a dense articulated solve kernel (`pgs_solve_tiled_contact_*`);
- dense impulse application back into world velocity (`apply_impulses_world_*`);
- a right-hand-side refresh (`rhs_accum_world_*`);
- a separate free-rigid matrix-free solve (`pgs_solve_mf_*`);
- another correction accumulation step before the next dense iteration.

Profiler artifacts for `h1_tabletop` on RTX 5090 (run `nightly/runs/full-3gpu-20260309f`):

| Path | Solver kernel time | Dominant components |
| --- | ---: | --- |
| `split` | ~7.33 s | `pgs_solve_mf_*` 2.41 s, `pgs_solve_tiled_contact_*` 0.89 s, `rhs_accum_world_*` 1.34 s, `apply_impulses_world_*` 0.42 s |
| `matrix_free` | ~3.46 s | `pgs_solve_mf_gs_*` 1.27 s; interleaved rhs/apply kernels eliminated |

The fused path wins because it combines the articulated velocity-path update with the free-rigid branch inside one kernel family, eliminating the per-iteration relaunch and right-hand-side refresh overhead.

However, the ~2× gap between `split` and `matrix_free` reflects kernel launch overhead in the interleaved approach, not an inherent cost of dense articulated coupling. This is a critical distinction: a tightly fused kernel that keeps dense articulated rows *and* matrix-free rigid rows could potentially match or beat the current fully matrix-free path. That experiment has not been run yet.

## Key Experiment: Fused Dense-Articulated + Rigid Matrix-Free Kernel

!!! warning "Open experiment — not yet implemented"

    The following experiment is needed before concluding that the fully matrix-free approach is the right long-term center:

    1. **Fused dense-articulated + rigid-MF kernel**: keep explicit dense Delassus coupling for articulated rows, keep free-rigid rows on the matrix-free branch, and fuse the two phases into a single kernel family — eliminating the relaunch overhead that makes `split` slow.
    2. **Streaming the dense Delassus matrix**: instead of staging the full packed $C$ triangle in shared memory, stream $C$ rows from global memory in the fused kernel. This trades shared-memory pressure for bandwidth.

    Until both are tried with profiler-backed occupancy measurements, we cannot distinguish "dense articulated coupling is inherently slower" from "the current `split` implementation has unnecessary overhead."

### Occupancy tradeoff

The core tension is shared-memory pressure. The dense tiled-row kernel wants shared memory for the packed $C$ tile. The current fused matrix-free kernel spends shared memory on the live velocity vector and runs at high occupancy. Combining both in a single kernel could reduce occupancy enough to erase the benefit.

To make this concrete, consider `h1_tabletop` sizing:

- The dense articulated phase has 30 rows after routing (in the matrix-free preset). The packed lower triangle of a 30-row $C$ tile is $30 \times 31 / 2 \times 4 = 1{,}860$ bytes. Even with row scalars and impulse state, this fits comfortably alongside the velocity vector in shared memory.
- The matrix-free kernel currently stages a velocity vector of $133 \times 4 = 532$ bytes (world DOFs × float32), plus row scalars. Total shared-memory use is well under 8 KiB per block.
- A fused kernel staging both the packed $C$ tile and the velocity vector would use roughly 2–3 KiB of shared memory — still small enough for high occupancy on current hardware.
- The streaming variant (reading $C$ rows from global memory instead of staging the full tile) would remove the $C$ tile from shared memory entirely, at the cost of additional global reads per iteration.

These are rough estimates. The actual occupancy impact depends on register pressure, block size, and the specific GPU. The point is that the numbers are small enough to be worth trying.

### What the experiment would show

If the fused dense-articulated + rigid-MF kernel matches or beats the current fully matrix-free path, the implication is that dense Delassus coupling is the better choice for articulated rows (exact off-diagonal terms, no velocity-path approximation artifacts) and matrix-free is the better choice for free-rigid rows (cheap per-row effective mass, no need to enter the articulated system). The current fully matrix-free path would then be understood as a good default that happened to win because the alternative (`split`) had unnecessary overhead, not because velocity-path coupling is inherently superior for articulated rows.

If the fused kernel is slower — because occupancy drops, or because the streaming variant cannot hide latency — then the fully matrix-free path is genuinely the right center, and the dense articulated kernels can be retired with confidence.

There is also an anecdotal observation from prior local experiments that artificially lowering matrix-free occupancy with an unused shared-memory allocation did not change throughput, but that result is not checked in and should not be treated as established evidence.
