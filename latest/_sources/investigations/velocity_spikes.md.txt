<!-- SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Velocity Spike Root-Cause Analysis

## Problem

During Isaac Lab training of the Franka cube-lift task, FeatherPGS produces transient velocity spikes exceeding the environment's termination threshold (1.25 × joint velocity soft limits), causing premature episode resets and degrading RL reward curves.

## Replay Infrastructure

We added opt-in capture hooks in `SolverFeatherPGS.step()` that snapshot full solver state (positions, velocities, Delassus matrix $C$, impulses $\lambda$, constraint metadata, $Y = \tilde{H}^{-1} J^T$) to `.npz` files whenever post-solve velocity exceeds a configurable threshold. Capture is off by default (zero overhead) and capped at 100 artifacts per session. A pure-numpy PGS replay harness reproduces captured frames with zero drift, enabling offline classification and fix experiments without GPU or Isaac Sim.

## Spike Classification

Replay of captured spike frames revealed three classes that matter.

### Class 1: Unconstrained $\hat{v}$ Spike (DOMINANT, ~60% of observed spikes)

The unconstrained velocity predictor $\hat{v}$ already exceeds soft velocity limits *before* the PGS solve runs. The PGS has minimal effect because active constraints (e.g., cube-table contact) do not project onto the spiking arm DOFs.

$$
\hat{v}_i = \dot{q}_i + \frac{\tau_i}{H_{ii}} \Delta t
$$

With Franka shoulder inertia ~0.65 kg·m² and max drive torque 87 N·m, $\Delta\hat{v}$ can reach 0.67 rad/s per substep. When the PD controller tracks a distant target and gravity compounds the drive torque, $\hat{v}$ reaches 4–6 rad/s within a few substeps.

**Peak:** 5.20 rad/s (2.4× shoulder soft limit). $v_{\text{out}} = \hat{v}$ with negligible impulses (max|impulse| = 0.006).

PGS parameter tuning (omega, CFM, iterations) **cannot** address this class.

### Class 2: Contact Impulse Spike (MODERATE)

When a wrist link contacts a surface, the contact Jacobian has lever arms through multiple arm DOFs. The Baumgarte correction from penetration ($\phi$ = −0.05 m) produces a meaningful impulse ($\lambda$ = 9.34) that maps to wrist DOFs. The PGS complementarity constraint ($\lambda \geq 0$) bounds the impulse, so the spike is moderate.

**Peak:** 2.70 rad/s (just above the 2.61 wrist limit). Partially PGS-tunable.

### Class 3: Joint-Limit Cross-Coupling (MILD)

When a joint overshoots its position limit, the correction impulse correctly stops the violating DOF, but mass matrix off-diagonal coupling ($Y$ matrix) propagates the impulse to neighboring DOFs. For the Franka's low-inertia wrist joints (0.03–0.05 kg·m²), even small limit violations produce non-trivial cross-coupled velocities.

A 100-combination parameter sweep (omega 1.0→0.6, CFM 0→10⁻², iterations 8→64) produces **identical results** (impulse variance < 10⁻⁴). The PGS is already converged at 8 iterations. These spikes are the mathematically correct solution to the formulated constraint problem.

**Peak:** 1.94 rad/s (below soft limit). Not PGS-tunable.

### Summary

| Class | Mechanism | Peak $|v_{\text{out}}|$ | vs. Threshold | PGS-tunable? |
|---|---|---|---|---|
| 1. Unconstrained $\hat{v}$ | Drive torque → large $\hat{v}$, no opposing constraint | **5.20 rad/s** | 1.9× over | **No** |
| 2. Contact impulse | Contact Baumgarte → wrist DOF kick | **2.70 rad/s** | 1.04× over | Partially |
| 3. Joint-limit coupling | Limit correction → $Y$ off-diagonals | **1.94 rad/s** | Below | No (converged) |

---

## Fix Experiments

Three candidate fixes tested on all three spike classes via the replay harness.

### Fix 1: Post-Solve Velocity Clamping

Clamp $v_{\text{out}}$ per-DOF to `factor × vel_limit` after the PGS solve, before integration:

$$
v_{\text{out},i} = \text{clip}\!\left(v_{\text{out},i},\; -f \cdot v_{\lim,i},\; +f \cdot v_{\lim,i}\right)
$$

| Artifact (Class) | Baseline | factor=1.25 | factor=1.5 | factor=2.0 |
|---|---|---|---|---|
| Unconstrained $\hat{v}$ (1) | 5.200 | **2.719 (−47.7%)** | 3.262 (−37.3%) | 4.350 (−16.3%) |
| Contact impulse (2) | 2.699 | 2.699 (0%) | 2.699 (0%) | 2.699 (0%) |
| Joint-limit coupling (3) | 1.944 | 1.944 (0%) | 1.944 (0%) | 1.944 (0%) |

Only fix effective against Class 1. At `factor=1.25`, the dominant spike drops from 5.20 to 2.72 rad/s, exactly at the termination threshold (47.7% reduction). Non-physical (discards kinetic energy) but fires only during transient spikes.

### Fix 2: Reduced $\beta$ (Baumgarte Correction)

Reduce the Baumgarte position correction factor ($\text{rhs} = \beta \cdot \phi / \Delta t + J \cdot \hat{v}$):

| Artifact (Class) | Baseline | $\beta$=0.02 | $\beta$=0.01 |
|---|---|---|---|
| Unconstrained $\hat{v}$ (1) | 5.200 | 5.200 (0%) | 5.200 (0%) |
| Contact impulse (2) | 2.699 | **1.658 (−38.6%)** | 1.727 (−36.0%) |
| Joint-limit coupling (3) | 1.944 | 3.190 (**+64.1%**) | 4.345 (**+123.5%**) |

Mixed effect: helps contact spikes (Class 2) but **worsens** joint-limit spikes (Class 3) significantly. Rejected.

### Fix 3: Contact Compliance (Diagonal Softening)

Add compliance $\alpha = c / \Delta t^2$ to the Delassus diagonal for contact-normal rows:

| Artifact (Class) | Baseline | $c$=0.0001 | $c$=0.001 | $c$=0.01 |
|---|---|---|---|---|
| Unconstrained $\hat{v}$ (1) | 5.200 | 5.200 (0%) | 5.200 (0%) | 5.200 (0%) |
| Contact impulse (2) | 2.699 | 1.940 (−28.1%) | **1.670 (−38.1%)** | 1.784 (−33.9%) |
| Joint-limit coupling (3) | 1.944 | 1.944 (0%) | 1.944 (0%) | 1.944 (0%) |

Reduces Class 2 contact spikes without affecting joint limits. Conservative $c$ = 0.0001 recommended.

### Fix Summary

| Spike Class | Best Fix | Effect |
|---|---|---|
| 1: Unconstrained $\hat{v}$ | **Velocity clamp (1.25×)** | 5.20 → 2.72 (−47.7%) |
| 2: Contact impulse | **Compliance ($c$=0.001)** | 2.70 → 1.67 (−38.1%) |
| 3: Joint-limit coupling | **None** | 1.94 (unchanged; correct solution) |

---

## Possible Mitigation

1. **Post-solve velocity clamp at 1.25×** addresses Class 1 (dominant); no harm to other classes.
2. **Contact compliance at 0.0001** addresses Class 2; no effect on joint limits.

Both are implemented on branch `dt/velocity-spike-claude`.
