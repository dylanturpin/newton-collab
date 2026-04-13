# FeatherPGS Velocity Spike Investigation Report

## Overview

This report documents the investigation of velocity spikes in the FeatherPGS solver when running the `Isaac-Lift-Cube-Franka-v0` task.  A velocity spike is a simulation step where the post-integration generalized velocity (`joint_qd`) reaches extreme magnitudes that would not occur in a physically faithful simulation.  In the Franka lift task, the environment terminates episodes when joint velocities exceed 1.25x the soft velocity limits (`joint_vel_out_of_limit_factor` with `factor=1.25`), and spikes that exceed this bound cause premature resets that degrade training.

## Task Configuration

The Franka cube lift task is defined in:

- `skild-IL-solver/source/isaaclab_tasks/.../lift/lift_env_cfg.py`
- `skild-IL-solver/source/isaaclab_tasks/.../lift/config/franka/joint_pos_env_cfg.py`

Key physics settings from `LiftPhysicsCfg.feather_pgs`:

| Parameter                      | Value     |
|-------------------------------|-----------|
| `pgs_iterations`              | 8         |
| `pgs_beta` (ERP)              | 0.05      |
| `pgs_cfm` (regularization)    | 1.0e-6    |
| `pgs_omega` (SOR)             | 1.0       |
| `pgs_warmstart`               | False     |
| `dense_max_constraints`       | 64        |
| `enable_joint_limits`         | True      |
| `update_mass_matrix_interval` | 1         |
| `num_substeps`                | 2         |
| `dt` (sim)                    | 0.01 s    |

Termination checks (from `FrankaCubeLiftEnvCfg`):

- `joint_vel_out_of_limit_factor` with `factor=1.25` on `panda_joint.*`
- `joint_pos_out_of_limit_factor` with `factor=1.25` on `panda_joint.*`
- `joint_effort_out_of_limit_factor` with `factor=1.25` on `panda_joint.*`

## Solver Architecture (Relevant to Spikes)

FeatherPGS uses a 7-stage pipeline per substep:

1. **FK/ID + Drives** – Forward kinematics, inverse dynamics, and drive torques.
2. **CRBA** – Composite Rigid Body Algorithm builds the articulation mass matrix H.
3. **Cholesky** – Factorize H for each articulation.
4. **Trisolve + v_hat** – Solve H * qdd = tau, then compute the velocity predictor `v_hat = qd + qdd * dt`.  This is the unconstrained velocity prediction; it already contains gravity and drive contributions but no contact or limit corrections.
5. **Build contact/limit problem** – Allocate constraint rows for contacts and joint limits, populate Jacobians J, compute Delassus matrix or diagonal.
6. **PGS Solve** – Iterative Projected Gauss-Seidel over the constraint system.  Produces `v_out`, the solved velocity in DOF space.
7. **Integrate** – Compute `qdd = (v_out - qd) / dt`, then integrate positions and velocities via `integrate_generalized_joints`.

A velocity spike can originate at several points:

- **v_hat already large**: Unconstrained dynamics (gravity, external forces, or large drive torques) predict extreme velocities before constraints are even considered.
- **PGS divergence**: The iterative PGS solve amplifies velocity rather than damping it.  This can happen with insufficient iterations, poor conditioning of the Delassus matrix, too-small `pgs_cfm`, or `pgs_omega > 1.0`.
- **Contact impulse spike**: A contact configuration produces impulses large enough to launch the articulation.
- **Joint limit interaction**: A joint near or past its limit receives a correction impulse that, combined with other forces, overshoots.

## Capture Path

Capture is implemented in `newton/_src/solvers/feather_pgs/spike_capture.py`.  It is **dormant by default** and adds no overhead when disabled.  When enabled (via environment variable or API call), it:

1. At the top of `SolverFeatherPGS.step()`, snapshots `state_in.joint_q` and `state_in.joint_qd` to CPU numpy arrays.
2. After stage 7 (integration), reads `state_out.joint_qd` and checks `max(|qd|)` against a configurable threshold (default 50.0).
3. If a spike is detected, writes a `.npz` artifact containing:

| Field              | Description                                             |
|--------------------|---------------------------------------------------------|
| `pre_joint_q`      | Generalized positions before the step                  |
| `pre_joint_qd`     | Generalized velocities before the step                 |
| `post_joint_q`     | Generalized positions after integration                |
| `post_joint_qd`    | Generalized velocities after integration               |
| `v_out`            | Post-PGS solved velocity (DOF space)                   |
| `v_hat`            | Velocity predictor (unconstrained)                     |
| `impulses`         | Dense PGS impulses                                     |
| `constraint_count` | Active constraint count per world                      |
| `solver_params`    | dt, pgs_iterations, pgs_beta, pgs_cfm, pgs_omega, etc |
| `meta`             | step_index, timestamp, max_abs_qd, capture_index       |

Each captured field matters because it lets us pinpoint where the spike originated.  Comparing `v_hat` to `v_out` separates unconstrained instability from PGS amplification.  Comparing `v_out` to `post_joint_qd` confirms integration did not introduce additional error.  The impulses and constraint counts reveal whether the spike is contact-driven or limit-driven.

### Enabling Capture

    export FEATHERPGS_SPIKE_CAPTURE=1
    export FEATHERPGS_SPIKE_THRESHOLD=50.0   # optional, default 50.0
    export FEATHERPGS_SPIKE_DIR=./spike_captures  # optional

Or programmatically:

    solver.spike_capture.enable(threshold=50.0, output_dir="./spike_captures")

### Analyzing Artifacts

    python -m newton._src.solvers.feather_pgs.spike_replay spike_captures/spike_0000_step42_maxqd150.3.npz

This prints a summary including the heuristic classification (one of `unconstrained`, `pgs_divergence`, `contact_impulse`, `joint_limit`, or `unknown`), the top DOFs by velocity magnitude, and the velocity predictor vs solved velocity comparison.

## Spike Taxonomy (Preliminary)

Based on the solver architecture, we expect these categories:

1. **Unconstrained (v_hat)** – The velocity predictor is already extreme.  Causes: large drive torques, gravity acting on poorly supported configurations, or numerical issues in FK/ID.
2. **PGS divergence** – v_hat is moderate but v_out is much larger.  Causes: ill-conditioned Delassus diagonal, insufficient iterations, `pgs_cfm` too small, `pgs_omega > 1.0`.
3. **Contact impulse** – Large impulses from contact resolution.  Causes: deep penetrations resolved in one step, thin objects, high-stiffness contacts with low compliance.
4. **Joint limit** – Concentrated velocity spike on few DOFs near their position limits.  Causes: aggressive position correction (`pgs_beta` too high) or limit constraint fighting against other forces.

Actual classification requires real spike captures from training runs.  The capture system includes a `classify_spike()` heuristic that identifies these patterns.

## Candidate Fixes (To Be Validated After Capture)

These are evidence-backed stabilization ideas to test on replay once spikes are captured:

- **Increase `pgs_cfm`** (e.g., 1e-4): More regularization on the constraint diagonal prevents near-singular behavior.
- **Lower `pgs_omega`** (e.g., 0.8): Under-relaxation damps PGS oscillation at the cost of slower convergence.
- **Lower `pgs_beta`** (e.g., 0.01): Less aggressive position correction reduces impulse magnitudes from penetration recovery.
- **Increase `pgs_iterations`** (e.g., 16): More iterations for better convergence.
- **Velocity clamping post-solve**: Clamp `v_out` before integration as a safety net (may violate energy conservation).
- **Dense contact compliance > 0**: The `dense_contact_compliance` parameter adds normal compliance to articulated contact rows.

## Status

- [x] Branch `dt/velocity-spike-claude` created in both repos
- [x] Capture instrumentation implemented and tested (11/11 unit tests pass)
- [x] Replay/analysis tool implemented
- [ ] Real spike captures from training runs
- [ ] Spike classification from real data
- [ ] Fix experiments on replay
- [ ] Longer-run validation

## Files Changed

- `newton/_src/solvers/feather_pgs/spike_capture.py` – New: capture module
- `newton/_src/solvers/feather_pgs/spike_replay.py` – New: replay/analysis tool
- `newton/_src/solvers/feather_pgs/solver_feather_pgs.py` – Modified: import + hooks in `step()`
- `tests/test_spike_capture.py` – New: unit tests
- `report.md` – This file
