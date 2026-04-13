<!-- SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Velocity Spike Root-Cause Analysis

## Problem

During Isaac Lab training of the Franka cube-lift task, FeatherPGS produces transient velocity spikes exceeding the environment's termination threshold (1.25 × joint velocity soft limits), causing premature episode resets and degrading RL reward curves.

## Replay Infrastructure

We built capture hooks in `SolverFeatherPGS.step()` that snapshot full solver state (positions, velocities, Delassus matrix, impulses, constraint metadata) to `.npz` files when post-solve velocity exceeds a threshold. A pure-numpy PGS replay harness reproduces the Warp kernel with zero drift, enabling offline classification and fix experiments without GPU or Isaac Sim.

## Spike Classification

Replay of captured spike frames revealed four distinct classes:

| Class | Mechanism | Peak velocity | Frequency | PGS-tunable? |
|---|---|---|---|---|
| 1 — Unconstrained $\hat{v}$ | Drive torque pushes predictor above limits before any constraint fires | **5.20 rad/s** (2.4× limit) | ~60% of spikes | **No** |
| 2 — Contact impulse | Contact Baumgarte correction kicks wrist DOFs | **2.70 rad/s** (just above limit) | Moderate | Partially |
| 3 — Joint-limit coupling | Limit correction propagates via $Y$ off-diagonals to neighboring DOFs | **1.94 rad/s** (below limit) | Mild | No (already converged at 8 iters) |
| 4 — Coupled limit+contact | Mixed constraints | **0.97 rad/s** (well below) | Rare | N/A |

The dominant finding: Class 1 spikes originate *upstream* of the PGS solve — $v_{\text{out}} = \hat{v}$ with negligible impulses. PGS parameter tuning (omega, CFM, iterations) cannot address the primary problem.

## Fix Experiments

Three fixes tested on all four artifact classes via the replay harness:

**Post-solve velocity clamping** — clamp $v_{\text{out}}$ per-DOF to `factor × vel_limit`:

| Class | Baseline | factor=1.25 |
|---|---|---|
| 1 (dominant) | 5.20 | **2.72 (−47.7%)** |
| 2–4 | ≤2.70 | unchanged (below clamp) |

Only fix effective against Class 1. At 1.25× the peak drops to exactly the termination threshold.

**Reduced $\beta$ (Baumgarte)** — helps Class 2 contacts (−38.6%) but *worsens* Class 3 joint limits (+64% to +123%). Rejected.

**Contact compliance** ($c$ = 0.0001) — reduces Class 2 contact spikes by ~28% and Class 4 by ~46%. No effect on joint limits.

## Proposed Mitigation

1. **Post-solve velocity clamp at 1.25×** — addresses Class 1 (dominant); no harm to other classes. Non-physical (discards kinetic energy) but fires only during transient spikes.
2. **Contact compliance at 0.0001** — addresses Class 2; no effect on joint limits.

Both are implemented on branch `dt/velocity-spike-claude`. Full training validation is pending — queued for dedicated hardware.
