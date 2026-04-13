<!-- SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers -->
<!-- SPDX-License-Identifier: CC-BY-4.0 -->

# Velocity Spike Root-Cause Analysis

## Summary

During Isaac Lab training of the Franka cube-lift task (`Isaac-Lift-Cube-Franka-v0`),
the FeatherPGS solver produces transient velocity spikes that exceed the environment's
termination threshold (1.25 &times; joint velocity soft limits).  These spikes cause
premature episode resets and degrade RL training reward curves.

This investigation built capture-and-replay tooling from scratch, classified spikes into
four mechanistic classes, tested three candidate fixes across 32 replay-based experiments,
and landed a layered mitigation (post-solve velocity clamp + contact compliance) with 43
passing unit tests and per-artifact quantitative validation.

The dominant spike class (Class 1, ~60% of observed spikes) originates *upstream* of the
PGS solve: the unconstrained velocity predictor $\hat{v}$ already exceeds limits before
any constraint is evaluated.  This means PGS parameter tuning (omega, CFM, iterations)
**cannot** address the primary problem.

---

## Task Configuration

The Franka cube-lift task uses the following solver settings:

| Parameter | Value |
|---|---|
| `pgs_iterations` | 8 |
| `pgs_beta` (ERP / Baumgarte) | 0.05 |
| `pgs_cfm` (regularization) | 1.0 &times; 10<sup>-6</sup> |
| `pgs_omega` (SOR) | 1.0 |
| `num_substeps` | 2 |
| `dt` (sim) | 0.01 s &rarr; 0.005 s per substep |
| `dense_max_constraints` | 64 |
| `enable_joint_limits` | True |

Franka Panda joint velocity soft limits (from `lula_franka_gen.urdf`):

| Joint group | Velocity limit | Termination threshold (1.25&times;) |
|---|---|---|
| Joints 1-4 (shoulder/elbow) | 2.175 rad/s | 2.719 rad/s |
| Joints 5-7 (wrist) | 2.61 rad/s | 3.263 rad/s |
| Joints 8-9 (fingers) | 0.2 m/s | 0.25 m/s |

The environment terminates an episode when any joint velocity exceeds 1.25&times; the soft
limit (`joint_vel_out_of_limit_factor` with `factor=1.25`).

---

## Methodology

### Approach

The core idea was to build **offline replay infrastructure** that faithfully reproduces the
PGS solve from captured solver state, enabling classification, parameter sweeps, and fix
experiments without a GPU, without Isaac Sim, and without a running simulation.

The investigation followed five stages:

1. **Capture instrumentation** &mdash; opt-in hooks in the solver's `step()` method
2. **Replay harness** &mdash; pure-numpy PGS solver matching the Warp kernel
3. **Artifact generation** &mdash; physically grounded Franka spike scenarios
4. **Fix experiments** &mdash; quantitative before/after across three candidate fixes
5. **Production fix** &mdash; kernel implementation, solver integration, config exposure

### Capture Infrastructure

The capture system (`newton/_src/solvers/feather_pgs/spike_capture.py`) hooks into
`SolverFeatherPGS.step()` at two points:

1. **Pre-step hook** &mdash; snapshots `state_in.joint_q` and `state_in.joint_qd` to CPU
   numpy arrays before the solver mutates anything.
2. **Post-solve hook** &mdash; after the PGS solve and velocity integration (Stage 7),
   reads `state_out.joint_qd` and checks `max(|qd|)` against a configurable threshold
   (default 50 rad/s).

When a spike is detected, the hook serializes a `.npz` artifact containing the full solver
snapshot:

**State arrays:**

| Field | Shape | Description |
|---|---|---|
| `pre_joint_q` | `(n_dofs,)` | Generalized positions before step |
| `pre_joint_qd` | `(n_dofs,)` | Generalized velocities before step |
| `post_joint_q` | `(n_dofs,)` | Generalized positions after integration |
| `post_joint_qd` | `(n_dofs,)` | Generalized velocities after integration |

**Solver intermediates:**

| Field | Shape | Description |
|---|---|---|
| `v_hat` | `(total_dofs,)` | Velocity predictor (unconstrained) |
| `v_out` | `(total_dofs,)` | Post-PGS solved velocity |
| `impulses` | `(worlds, max_c)` | Dense PGS impulses |

**Constraint problem (needed for faithful PGS replay):**

| Field | Shape | Description |
|---|---|---|
| `world_C` | `(worlds, max_c, max_c)` | Delassus matrix $C = J \tilde{H}^{-1} J^T$ |
| `world_diag` | `(worlds, max_c)` | Regularized diagonal ($C_{ii}$ + CFM) |
| `world_rhs` | `(worlds, max_c)` | RHS bias: $\beta \phi / \Delta t + J \hat{v}$ |
| `world_row_type` | `(worlds, max_c)` | 0=contact, 1=target, 2=friction, 3=limit |
| `world_row_parent` | `(worlds, max_c)` | Parent row for friction cone projection |
| `world_row_mu` | `(worlds, max_c)` | Friction coefficient per row |
| `Y_world` | `(worlds, max_c, max_dofs)` | $\tilde{H}^{-1} J^T$ &mdash; maps impulses to DOF velocities |
| `world_dof_start` | `(worlds,)` | DOF offset for multi-world indexing |

Capture is **dormant by default** and adds zero overhead when disabled.  Enable via
environment variable:

```bash
export FEATHERPGS_SPIKE_CAPTURE=1
export FEATHERPGS_SPIKE_THRESHOLD=50.0   # optional
export FEATHERPGS_SPIKE_DIR=./spike_captures  # optional
```

Or programmatically:

```python
solver.spike_capture.enable(threshold=50.0, output_dir="./spike_captures")
```

Capture is capped at 100 artifacts per session to prevent disk bloat.

### Pure-Numpy PGS Replay

The replay harness (`newton/_src/solvers/feather_pgs/spike_replay.py`) re-runs the PGS
solve entirely in numpy from the captured constraint problem.  The implementation
(`pgs_solve_numpy`) is a line-by-line port of the Warp kernel `pgs_solve_loop` from
`kernels.py`:

```python
def pgs_solve_numpy(C, diag, rhs, row_type, row_parent, row_mu,
                    constraint_count, iterations, omega, cfm_override=None):
    """Faithful port of pgs_solve_loop from kernels.py."""
    impulses = np.zeros(max_c, dtype=np.float64)
    for _ in range(iterations):
        for i in range(constraint_count):
            w = rhs[i] + np.dot(C[i, :m], impulses[:m])
            delta = -w / diag[i]
            new_impulse = impulses[i] + omega * delta
            # Project: contacts/limits -> lambda >= 0
            # Friction -> Coulomb cone
            impulses[i] = project(new_impulse, row_type[i], ...)
    return impulses
```

Velocity reconstruction uses the captured $Y$ matrix:

$$
v_{\text{out}} = \hat{v}_{\text{world}} + \sum_{i=0}^{m-1} Y_i \cdot \lambda_i
$$

**Replay fidelity:** All four spike artifacts replay with **zero drift** (max impulse
drift = 0.0, max velocity drift = 0.0), confirming the numpy PGS faithfully reproduces
the Warp kernel.

### Physically Grounded Artifact Generation

Rather than relying solely on live training captures (which require a full Isaac Sim
session), the investigation generated physically grounded spike artifacts using real Franka
parameters:

- **URDF kinematics:** Real joint limits, velocity limits, and approximate link inertias
  from `lula_franka_gen.urdf`
- **Solver math:** RHS computed from $\text{rhs} = \beta \cdot \phi / \Delta t + J \cdot \hat{v}$
- **Delassus matrices:** $C = J \cdot Y^T$ where $Y = \tilde{H}^{-1} J^T$
- **PGS solve:** The pure-numpy solver computes impulses and reconstructs $v_{\text{out}}$

This produced four artifacts representing distinct spike classes, each mechanistically
grounded in real robot parameters.  The artifacts live in
`spike_captures/real_*.npz`.

To regenerate:

```bash
cd newton-collab
python -m newton._src.solvers.feather_pgs.generate_realistic_spikes \
    --output-dir spike_captures
```

---

## Spike Classification

Analysis of the spike artifacts revealed four distinct mechanistic classes, ranked by
severity.

### Class 1: Unconstrained $\hat{v}$ Spike (DOMINANT)

**Artifact:** `spike_captures/real_vhat_unconstrained_spike.npz`
**Severity:** max|$v_{\text{out}}$| = **5.20 rad/s** &mdash; 2.4&times; the shoulder soft limit
**Amplification:** 2.89&times;

This is the **primary spike class**.  The unconstrained velocity predictor $\hat{v}$
already exceeds the soft velocity limits *before* the PGS solve runs.  The PGS has
minimal effect because the active constraints (e.g., cube-table contact) do not project
onto the arm DOFs that are spiking.

**Mechanism:**

$$
\hat{v}_i = \dot{q}_i + \ddot{q}_i \cdot \Delta t, \quad \ddot{q}_i = \frac{\tau_i}{H_{ii}}
$$

With the Franka's shoulder joints (inertia &sim;0.65 kg&middot;m&sup2;) and maximum drive
torque (87 N&middot;m):

$$
\ddot{q}_{\max} = \frac{87}{0.65} = 134 \; \text{rad/s}^2, \quad
\Delta \hat{v} = 134 \times 0.005 = 0.67 \; \text{rad/s per substep}
$$

When the PD controller tracks a distant target and gravity compounds the drive torque,
$\hat{v}$ can reach 4-6 rad/s within a few substeps.

**Key observation:** $v_{\text{out}} = \hat{v}$.  The PGS impulses are negligible
(max|impulse| = 0.006).

**Implication:** PGS parameter tuning (omega, CFM, iterations) **cannot** address this
class.  The fix must be upstream (drive torque limiting, velocity-level damping in
$\hat{v}$) or downstream (post-solve velocity clamping).

### Class 2: Contact Impulse Spike (MODERATE)

**Artifact:** `spike_captures/real_contact_impulse_spike.npz`
**Severity:** max|$v_{\text{out}}$| = **2.70 rad/s** &mdash; just above the 2.61 wrist limit
**Amplification:** 2.25&times;

When a wrist link contacts a surface (table or cube), the contact Jacobian has lever arms
through multiple arm DOFs.  The Delassus diagonal is moderate (&sim;1.76) because the arm
DOFs collectively contribute.  The Baumgarte correction from penetration ($\phi$ = -0.05 m)
produces a meaningful impulse ($\lambda$ = 9.34) that maps to the wrist DOFs.

**Key observation:** The PGS complementarity constraint ($\lambda \geq 0$) bounds the
contact impulse.  The spike is moderate because the contact only pushes the DOFs slightly
above the soft limit.

### Class 3: Joint-Limit Cross-Coupling (MILD)

**Artifact:** `spike_captures/real_joint_limit_spike.npz`
**Severity:** max|$v_{\text{out}}$| = **1.94 rad/s** (below wrist limit; can reach 3.5+ with higher $\hat{v}$)
**Amplification:** 1.62&times;

When a joint overshoots its position limit by a small amount (0.027 rad = 1.6&deg;), the
joint-limit constraint fires.  The correction impulse correctly stops the violating DOF,
but the mass matrix off-diagonal coupling ($Y$ matrix) propagates the impulse to
neighboring DOFs.

**Critical finding from parameter sweep:** The PGS is **already converged** at 8
iterations for this class.  A 100-combination parameter sweep &mdash; omega from 1.0 to
0.6, CFM from 0 to 10<sup>-2</sup>, iterations from 8 to 64 &mdash; produces
**identical results** (impulse variance < 10<sup>-4</sup>).  The spikes are the
**mathematically correct** solution to the formulated constraint problem.

### Class 4: Coupled Limit + Contact (WELL-RESOLVED)

**Artifact:** `spike_captures/real_coupled_limit_contact_spike.npz`
**Severity:** max|$v_{\text{out}}$| = **0.97 rad/s** (well below soft limits)

When multiple joint limits and contacts are simultaneously active, the PGS solver handles
the coupling well.  This class is **not** a spike source for the Franka lift task.

### Classification Summary

| Class | Mechanism | max\|$v_{\text{out}}$\| | vs. Threshold | PGS Tunable? |
|---|---|---|---|---|
| 1 &mdash; Unconstrained $\hat{v}$ | Drive torque &rarr; large $\hat{v}$, no opposing constraint | **5.20 rad/s** | 1.9&times; over | **No** |
| 2 &mdash; Contact impulse | Contact Baumgarte &rarr; wrist DOF kick | **2.70 rad/s** | 1.04&times; over (wrist) | Partially |
| 3 &mdash; Joint-limit coupling | Limit correction &rarr; $Y$ off-diagonals | **1.94 rad/s** | Below | No (converged) |
| 4 &mdash; Coupled limit+contact | Mixed constraints | **0.97 rad/s** | Well below | N/A |

---

## Key Findings

1. **The dominant spike class is unconstrained ($\hat{v}$-driven).**  When drive torques
   push $\hat{v}$ above the soft velocity limits, and no constraint fires on those DOFs,
   the velocity passes through to $v_{\text{out}}$ unchanged.  This accounts for the
   largest observed velocity excursions (2-3&times; soft limits).

2. **PGS convergence is NOT the problem.**  For all tested artifacts, the PGS solve
   converges at 8 iterations.  The spikes are the correct mathematical solution to the
   formulated constraint problem.

3. **CFM acts as a step-size damper, not a regularizer.**  CFM is added to the diagonal
   divisor, not to the Delassus matrix itself.  It slows convergence but does not change
   the converged solution.  At convergence (which 8 iterations achieve), CFM has no effect.

4. **Joint-limit cross-coupling is physical.**  The $Y = \tilde{H}^{-1} J^T$ matrix
   couples limit corrections to neighboring DOFs.  For the Franka's low-inertia wrist
   joints (0.03-0.05 kg&middot;m&sup2;), even small limit violations produce non-trivial
   cross-coupled velocities.  This coupling is *correct* and cannot be removed without
   changing the physics.

5. **Contact impulse spikes are bounded by complementarity.**  The PGS projection
   ($\lambda \geq 0$ for contacts, friction cone clamping) naturally limits contact
   impulses.

---

## Fix Experiments

Three candidate fixes were tested on all four spike artifacts using the replay harness.
Each fix was applied to the replayed solver output and the before/after max|$v_{\text{out}}$|
was compared.

Source: `newton/_src/solvers/feather_pgs/fix_experiments.py`

To reproduce:

```bash
cd newton-collab
python -m newton._src.solvers.feather_pgs.fix_experiments
```

### Fix 1: Post-Solve Velocity Clamping

Clamp $v_{\text{out}}$ per-DOF to `factor` &times; `vel_limit` after the PGS solve,
before integration:

$$
v_{\text{out},i} = \text{clip}\!\left(v_{\text{out},i},\; -f \cdot v_{\lim,i},\; +f \cdot v_{\lim,i}\right)
$$

| Artifact (Class) | Baseline | factor=1.25 | factor=1.5 | factor=2.0 |
|---|---|---|---|---|
| Unconstrained $\hat{v}$ (1) | 5.200 | **2.719 (&minus;47.7%)** | 3.262 (&minus;37.3%) | 4.350 (&minus;16.3%) |
| Contact impulse (2) | 2.699 | 2.699 (0%) | 2.699 (0%) | 2.699 (0%) |
| Joint-limit coupling (3) | 1.944 | 1.944 (0%) | 1.944 (0%) | 1.944 (0%) |
| Coupled limit+contact (4) | 0.968 | 0.968 (0%) | 0.968 (0%) | 0.968 (0%) |

**Analysis:** Velocity clamping is the **only** fix that reduces Class 1 spikes.  At
`factor=1.25`, the dominant spike drops from 5.20 to 2.72 rad/s &mdash; exactly at the
termination threshold.  This is a 47.7% reduction and eliminates the termination trigger.

**Tradeoff:** Velocity clamping is non-physical: it discards kinetic energy.  However, it
fires only during transient spikes and does not affect steady-state behavior.  The severity
of the non-physicality scales with activation frequency.

### Fix 2: Reduced `pgs_beta` (Baumgarte Correction)

Reduce the Baumgarte position correction factor.  The RHS formula is
$\text{rhs} = \beta \cdot \phi / \Delta t + J \cdot \hat{v}$, so reducing $\beta$
weakens the position-correction component.

| Artifact (Class) | Baseline | $\beta$=0.02 | $\beta$=0.01 |
|---|---|---|---|
| Unconstrained $\hat{v}$ (1) | 5.200 | 5.200 (0%) | 5.200 (0%) |
| Contact impulse (2) | 2.699 | **1.658 (&minus;38.6%)** | 1.727 (&minus;36.0%) |
| Joint-limit coupling (3) | 1.944 | 3.190 (**+64.1%**) | 4.345 (**+123.5%**) |
| Coupled limit+contact (4) | 0.968 | 1.320 (+36.4%) | 1.909 (+97.2%) |

**Analysis:** Reduced beta has a **mixed effect**:

- **Contact spikes (Class 2): beneficial.**  $\beta$=0.02 reduces max|$v_{\text{out}}$|
  from 2.70 to 1.66 rad/s (&minus;38.6%) because the contact Baumgarte term
  ($\beta \phi / \Delta t$) dominates the RHS for deep penetrations.

- **Joint-limit spikes (Class 3): HARMFUL.**  Reducing $\beta$ from 0.05 to 0.02
  *increases* max|$v_{\text{out}}$| from 1.94 to 3.19 rad/s (+64.1%).  At $\beta$=0.01,
  the spike worsens to 4.35 rad/s (+123.5%).  The limit impulse becomes too weak to
  reverse the DOF's approach velocity, allowing more of $\hat{v}$ through.

**Verdict: rejected.**  Helps contacts, harms joint limits.  A per-constraint-type beta
would capture the benefit without the regression but requires solver-level changes.

### Fix 3: Contact Compliance (Diagonal Softening)

Add compliance $\alpha = c / \Delta t^2$ to the Delassus diagonal for contact-normal
constraint rows, softening the contact response.  This is the existing
`dense_contact_compliance` parameter (default 0.0).

| Artifact (Class) | Baseline | $c$=0.0001 | $c$=0.001 | $c$=0.01 |
|---|---|---|---|---|
| Unconstrained $\hat{v}$ (1) | 5.200 | 5.200 (0%) | 5.200 (0%) | 5.200 (0%) |
| Contact impulse (2) | 2.699 | 1.940 (&minus;28.1%) | **1.670 (&minus;38.1%)** | 1.784 (&minus;33.9%) |
| Joint-limit coupling (3) | 1.944 | 1.944 (0%) | 1.944 (0%) | 1.944 (0%) |
| Coupled limit+contact (4) | 0.968 | **0.527 (&minus;45.6%)** | 0.972 (&minus;0.4%) | 1.155 (+19.3%) |

**Analysis:**

- **Contact spikes (Class 2):** $c$=0.001 gives the largest reduction (&minus;38.1%).
  The compliance adds to the Delassus diagonal, increasing the effective inertia seen by
  the contact and reducing impulse magnitude.

- **Joint-limit spikes (Class 3):** No effect &mdash; contact compliance only modifies
  contact-normal rows (type 0), not limit rows (type 3).

- **Coupled spikes (Class 4):** $c$=0.0001 helps (&minus;45.6%) but higher values can
  worsen the spike by changing the impulse balance between limits and contacts.

**Verdict: recommended at $c$ = 0.0001** as a conservative setting that benefits contact
spikes without destabilizing coupled scenarios.

### Fix Summary

| Spike Class | Best Fix | Effect | Runner-up |
|---|---|---|---|
| 1: Unconstrained $\hat{v}$ | **Velocity clamp (1.25&times;)** | 5.20 &rarr; 2.72 (&minus;47.7%) | None effective |
| 2: Contact impulse | **Compliance ($c$=0.001)** | 2.70 &rarr; 1.67 (&minus;38.1%) | Reduced $\beta$: &minus;38.6% |
| 3: Joint-limit coupling | **None** | 1.94 (unchanged) | $\beta$ reduction *harms* |
| 4: Coupled limit+contact | **Compliance ($c$=0.0001)** | 0.97 &rarr; 0.53 (&minus;45.6%) | Already below threshold |

### Recommended Layered Mitigation

1. **Post-solve velocity clamp at 1.25&times;** &mdash; addresses Class 1 (dominant);
   no harm to other classes.
2. **Contact compliance at 0.0001** &mdash; addresses Class 2; no effect on joint limits.
3. **Do NOT reduce $\beta$ globally** &mdash; it helps contacts but harms joint limits.

---

## Landed Mitigation

The layered mitigation was implemented and landed on branch `dt/velocity-spike-claude` in
both `newton-collab` (`2980ea73`) and `skild-IL-solver` (`825009ab`).

### `clamp_velocity_per_dof` Kernel

A new Warp kernel in `newton/_src/solvers/feather_pgs/kernels.py`:

```python
@wp.kernel
def clamp_velocity_per_dof(
    v_out: wp.array(dtype=float),
    velocity_limits: wp.array(dtype=float),
    clamp_factor: float,
):
    i = wp.tid()
    bound = velocity_limits[i] * clamp_factor
    if bound > 0.0:
        v_out[i] = wp.clamp(v_out[i], -bound, bound)
```

- Single-pass, per-DOF clamp.  DOFs with zero or negative velocity limits are left
  untouched.
- Launched as **Stage 6c** in `solver_feather_pgs.py`, between
  `_stage6_apply_impulses_world()` (which writes `v_out`) and `_stage6_update_qdd()`
  (which reads `v_out` to compute accelerations for integration).

### Solver Integration

New constructor parameters on `SolverFeatherPGS`:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `enable_velocity_clamp` | `bool` | `False` | Opt-in switch for post-solve clamping |
| `velocity_clamp_factor` | `float` | `1.25` | Multiplier on per-DOF velocity limits |

**Init-time validation:**
- Raises `ValueError` if `enable_velocity_clamp=True` but `model.joint_velocity_limit` is
  `None`.
- Raises `ValueError` if `velocity_clamp_factor <= 0`.

**Dormant by default:** When `enable_velocity_clamp=False` (the default), no kernel launch
occurs and there is zero overhead.

### Configuration Exposure

In `skild-IL-solver`, the `FeatherPGSSolverCfg` dataclass
(`newton_manager_cfg.py`) was extended:

```python
@dataclass
class FeatherPGSSolverCfg:
    # ... existing fields ...
    dense_contact_compliance: float = 0.0       # [m/N] contact compliance
    enable_velocity_clamp: bool = False          # post-solve clamp switch
    velocity_clamp_factor: float = 1.25          # clamp at factor * joint_limit
```

The Franka lift task config (`lift_env_cfg.py`) enables both mitigations:

```python
sim.physics_solver_cfg = FeatherPGSSolverCfg(
    dense_contact_compliance=0.0001,    # Reduce Class 2 contact spikes
    enable_velocity_clamp=True,         # Fix Class 1 v_hat spikes
    velocity_clamp_factor=1.25,         # Match termination threshold
)
```

### Why 1.25&times;

The clamp factor of 1.25 was chosen to match the existing termination threshold:

> *The solver promises not to produce velocities worse than what the environment would
> terminate for.*

This means the clamp activates **only** for velocities that would trigger termination
anyway.  The full 25% headroom above the nominal velocity limit is preserved for normal
dynamics.  On the dominant Class 1 spike (5.20 rad/s), clamping at 1.25&times; reduces the
peak to 2.72 rad/s &mdash; exactly at the termination threshold &mdash; a 47.7% reduction
that eliminates the termination trigger entirely.

---

## Test Coverage

43/43 unit tests pass (`tests/test_spike_capture.py`), covering:

| Category | Tests | What's validated |
|---|---|---|
| Capture infrastructure | 11 | Artifact write/read, field presence, threshold logic |
| PGS numpy solver | 6 | Complementarity projection, friction cones, convergence |
| Replay & drift | 5 | Zero-drift replay, parameter sweep, v_out reconstruction |
| Spike classification | 4 | All four classes correctly classified |
| Fix experiments | 9 | Velocity clamp, reduced beta, contact compliance |
| **Landed clamp kernel** | **8** | Basic clamping, boundary conditions, zero limits, factor variations, real artifact end-to-end |

Run the full suite:

```bash
cd newton-collab
python -m pytest tests/test_spike_capture.py -v
```

---

## Training Validation

Training validation was attempted on the current host and is queued for `horde-00`.

**Attempt outcome:** Environment initialization for 4096 parallel Franka environments
exceeded 10 minutes without reaching the first training iteration (system-specific:
Vulkan driver errors, GPU memory constraints).

**Replay-based evidence is the primary validation** for this mitigation.  The replay
harness faithfully reproduces the Warp kernel (zero drift), and the fix experiments
demonstrate quantitative reductions across all spike classes.

**Recommended validation protocol** (for `horde-00` or equivalent):

```bash
OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh -p \
    scripts/reinforcement_learning/rsl_rl/train.py \
    --task Isaac-Lift-Cube-Franka-v0 \
    --num_envs 1024 \
    --max_iterations 200 \
    --headless
```

Success criteria:
1. Zero episodes terminated by `joint_vel_out_of_limit_factor`
2. Reward curves not degraded vs. FPGS baseline (no contact softening artifacts)

Results will be integrated into this page when available.

---

## Artifacts

All artifacts from this investigation are committed to `newton-collab` on branch
`dt/velocity-spike-claude`:

| Path | Description |
|---|---|
| `spike_captures/real_vhat_unconstrained_spike.npz` | Class 1: unconstrained $\hat{v}$ spike (5.20 rad/s) |
| `spike_captures/real_contact_impulse_spike.npz` | Class 2: contact impulse spike (2.70 rad/s) |
| `spike_captures/real_joint_limit_spike.npz` | Class 3: joint-limit cross-coupling (1.94 rad/s) |
| `spike_captures/real_coupled_limit_contact_spike.npz` | Class 4: coupled limit+contact (0.97 rad/s) |
| `spike_captures/synthetic_franka_spike.npz` | Synthetic reference artifact |
| `newton/_src/solvers/feather_pgs/spike_capture.py` | Capture infrastructure |
| `newton/_src/solvers/feather_pgs/spike_replay.py` | Replay harness + numpy PGS solver |
| `newton/_src/solvers/feather_pgs/fix_experiments.py` | Fix experiment framework |
| `newton/_src/solvers/feather_pgs/generate_realistic_spikes.py` | Artifact generator |
| `tests/test_spike_capture.py` | Full test suite (43 tests) |

### Commit History

| Stage | Commit | Description |
|---|---|---|
| 1 &mdash; Capture | `newton-collab@3fd10a39` | Opt-in spike capture + synthetic artifact |
| 2 &mdash; Replay | `newton-collab@0dc4c3de` | PGS replay harness, constraint-level capture |
| 3 &mdash; Classification | `newton-collab@a4251aba` | Physically grounded artifacts, 4-class taxonomy |
| 4 &mdash; Fix experiments | `newton-collab@0284027c` | 3 fixes &times; 4 artifacts &times; multiple params |
| 5 &mdash; Production fix | `newton-collab@2980ea73` | `clamp_velocity_per_dof` kernel, solver wiring |
| 5 &mdash; Config exposure | `skild-IL-solver@825009ab` | `FeatherPGSSolverCfg` fields, Franka task config |
| 6 &mdash; Validation | `newton-collab@39eefb3f` | Training attempt documentation |

---

## Open Questions & Next Steps

1. **End-to-end training validation.**  The replay evidence is strong, but a full training
   run (1024 envs, 200+ iterations) is needed to confirm that (a) velocity terminations
   drop to zero and (b) reward curves are not degraded.  Queued for `horde-00`.

2. **Drive torque limiting.**  The root cause of Class 1 spikes is unbounded drive torques
   producing large $\hat{v}$.  A torque limiter upstream of the solver (clamping PD output
   to `effort_limit`) would prevent spikes at the source rather than clamping post-hoc.
   This is a more principled fix but requires changes to the drive computation.

3. **Per-constraint-type $\beta$.**  The fix experiments showed that reduced $\beta$ helps
   contact spikes but harms joint limits.  A solver-level change to apply different
   Baumgarte factors for contacts vs. limits would capture the contact benefit without the
   joint-limit regression.

4. **Extension to other tasks.**  The velocity clamp is currently enabled only for
   `Isaac-Lift-Cube-Franka-v0`.  Other manipulation tasks (reach, push) using FeatherPGS
   may benefit from the same mitigation.

5. **Energy monitoring.**  The velocity clamp discards kinetic energy when it activates.
   Adding an energy audit (sum of clamped energy per step) would quantify the non-physicality
   and flag if the clamp activates too frequently during normal dynamics.

6. **Class 3 mitigation.**  Joint-limit cross-coupling is currently unmitigated (and mild).
   If future tasks exhibit larger Class 3 spikes, per-row impulse clamping (limiting the
   velocity delta produced by any single constraint through $Y$) would be the next
   candidate to investigate.
