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

We already tried a `split` mode that keeps articulated rows on the dense Delassus side and sends only free-rigid rows through matrix-free. Profiler results on `h1_tabletop` (RTX 5090, run `nightly/runs/full-3gpu-20260309f`):

| Path | Solver kernel time | Note |
| --- | ---: | --- |
| `split` | ~7.33 s | Two kernel families per PGS iteration (dense articulated + separate MF rigid), plus inter-kernel rhs refresh and impulse-apply passes |
| `matrix_free` | ~3.46 s | Single fused kernel family; inter-kernel overhead eliminated |

The ~2× gap is kernel-launch and rhs-refresh overhead from the interleaved approach, not an inherent cost of dense articulated coupling.

## Key Experiment: Fused Dense-Articulated + Rigid Matrix-Free Kernel

```{admonition} Open experiment — not yet implemented
:class: warning

A fused kernel that keeps explicit dense Delassus coupling for articulated rows *and* matrix-free handling for rigid rows — eliminating the relaunch overhead that makes `split` slow — has not been tried yet. A streaming variant (reading $C$ rows from global instead of staging the full tile in shared memory) would further reduce shared-memory pressure.

Until this is profiled we cannot distinguish "dense articulated coupling is inherently slower" from "the `split` implementation just had unnecessary overhead."
```

### Occupancy tradeoff

For `h1_tabletop`: the packed lower triangle of a 30-row $C$ tile is ~1.9 KiB; the velocity vector is ~0.5 KiB. A fused kernel staging both would use roughly 2–3 KiB of shared memory — small enough for high occupancy on current hardware. The streaming variant would remove the $C$ tile entirely at the cost of extra global reads. Whether this trades favorably depends on register pressure, block size, and the specific GPU, but the numbers are small enough to be worth trying.

If the fused kernel matches the fully matrix-free path, dense Delassus is the better choice for articulated rows (exact off-diagonals) and matrix-free for rigid rows. If it's slower even with streaming, the fully matrix-free path is genuinely the right center and dense articulated kernels can be retired.
