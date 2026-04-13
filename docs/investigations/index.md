<!-- SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Investigations

This journal tracks solver investigations that feed into Newton design decisions. It is intentionally short-form: each dated entry points to the deeper artifact or branch where the full analysis lives.

## 2026-04-13

### [Dense vs matrix-free FeatherPGS explainer](../concepts/feather_pgs_dense_vs_matrix_free.md)

The main dense-vs-matrix-free report now ties the current FeatherPGS code paths to checked-in sizing artifacts for `g1_flat` and `h1_tabletop`, plus kernel-level memory-layout analysis. Its new ablation section turns that evidence into a concrete keep/deprecate/remove recommendation for the live solver variants.

### [Velocity spike root-cause analysis](velocity_spikes.md)

The dominant spike class is unconstrained `v_hat`, not a PGS convergence failure. On the reproduced traces, PGS converges in 8 iterations; the recommended mitigation is post-solve velocity clamping plus compliance-oriented follow-up work rather than a dense-vs-matrix-free path change.

### Private API matrix-free cleanup

Branch: `dturpin/fpgs-private-api-matrix-free`

The private API cleanup collapsed the public FeatherPGS surface to matrix-free-only and removed now-dead solver branching around the public constructor knobs. That branch is the implementation proof that the matrix-free-only direction is operationally feasible, not just a docs-level recommendation.

### Warp kernel migration (Zach Corse)

Zach converted the FeatherPGS kernels from raw CUDA strings to Warp, which is significantly easier to maintain, understand, and debug. The initial conversion showed a ~40% throughput reduction on `h1_tabletop`, but after targeted Warp-side optimizations that gap has narrowed to ~9% — likely an acceptable tradeoff for the maintainability and debuggability gains. Release timing for this depends on landing the required Warp PRs upstream, and needs to be coordinated with the private FeatherPGS API changes in Newton.

### [In-flight] TGS feasibility study

Branch: `dt/tgs-feather-pgs-study`

Just kicked off. The TGS-style smaller-step path (doubled FeatherPGS solves per action at `sim.dt=0.0025`, `decimation=8`) shows equivalent physics at roughly 2× wall-clock cost. Early training comparisons suggest the velocity guard is the dominant training blocker — removing it allows learning to start — but the absolute reward numbers are low and need validation against a proper baseline. Assumptions and setup still need human verification before drawing conclusions.
