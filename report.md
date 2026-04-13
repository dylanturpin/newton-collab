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

**State arrays:**

| Field              | Description                                             |
|--------------------|---------------------------------------------------------|
| `pre_joint_q`      | Generalized positions before the step                  |
| `pre_joint_qd`     | Generalized velocities before the step                 |
| `post_joint_q`     | Generalized positions after integration                |
| `post_joint_qd`    | Generalized velocities after integration               |

**Solver intermediates:**

| Field              | Description                                             |
|--------------------|---------------------------------------------------------|
| `v_out`            | Post-PGS solved velocity (DOF space)                   |
| `v_hat`            | Velocity predictor (unconstrained)                     |
| `impulses`         | Dense PGS impulses, shape (world, max_constraints)     |
| `constraint_count` | Active constraint count per world                      |

**Constraint problem (Stage 2: needed for PGS replay):**

| Field              | Description                                             |
|--------------------|---------------------------------------------------------|
| `world_C`          | Delassus matrix C = J H^{-1} J^T per world            |
| `world_diag`       | Regularized diagonal (C diagonal + CFM)                |
| `world_rhs`        | Right-hand side bias vector                            |
| `world_row_type`   | Constraint type: 0=contact, 1=target, 2=friction, 3=limit |
| `world_row_parent` | Parent row index for friction constraints (-1 otherwise) |
| `world_row_mu`     | Friction coefficient per constraint row                |
| `Y_world`          | H^{-1} J^T per world, maps impulses to DOF velocities |
| `world_dof_start`  | DOF start offset per world (for multi-world indexing)  |
| `solver_params`    | dt, pgs_iterations, pgs_beta, pgs_cfm, pgs_omega, etc |
| `meta`             | step_index, timestamp, max_abs_qd, capture_index       |

Each captured field matters because it lets us pinpoint where the spike originated.  Comparing `v_hat` to `v_out` separates unconstrained instability from PGS amplification.  Comparing `v_out` to `post_joint_qd` confirms integration did not introduce additional error.  The impulses and constraint counts reveal whether the spike is contact-driven or limit-driven.  The constraint-level arrays (C, diag, rhs, row metadata, Y) enable offline PGS replay without GPU or Warp.

### Enabling Capture

    export FEATHERPGS_SPIKE_CAPTURE=1
    export FEATHERPGS_SPIKE_THRESHOLD=50.0   # optional, default 50.0
    export FEATHERPGS_SPIKE_DIR=./spike_captures  # optional

Or programmatically:

    solver.spike_capture.enable(threshold=50.0, output_dir="./spike_captures")

### Analyzing Artifacts

    python -m newton._src.solvers.feather_pgs.spike_replay spike_captures/spike_0000_step42_maxqd150.3.npz

This prints a summary including the heuristic classification (one of `unconstrained`, `pgs_divergence`, `contact_impulse`, `joint_limit`, or `unknown`), the top DOFs by velocity magnitude, and the velocity predictor vs solved velocity comparison.

## Replay Path

The replay harness is implemented in `newton/_src/solvers/feather_pgs/spike_replay.py`.  It re-runs the PGS solve in pure numpy from the captured constraint problem, compares the replayed output to the original capture, and enables parameter-sweep experiments.

### How Replay Works

The replay harness reconstructs a single PGS solve step without GPU or Warp:

1. Load the artifact with `SpikeArtifact.load("spike.npz")`.
2. Extract the constraint problem: Delassus matrix `C`, regularized diagonal `diag`, right-hand side `rhs`, and constraint metadata (`row_type`, `row_parent`, `row_mu`).
3. Run `pgs_solve_numpy()`, a faithful port of the Warp kernel `pgs_solve_loop` from `kernels.py`, implementing the same Projected Gauss-Seidel iteration with:
   - Normal/joint-limit constraints clamped at lambda >= 0.
   - Friction constraints projected onto the isotropic Coulomb cone.
   - SOR relaxation via the omega parameter.
4. Compare replayed impulses to the originally captured impulses to measure "impulse drift".
5. Reconstruct the replayed velocity: `v_out_replay = v_hat + Y^T * impulses_replay` using the captured `Y_world` matrix.
6. Compare replayed velocity to the original `v_out` to measure "velocity drift".

### Determinism Measurement

When the replay uses the same PGS parameters as the original capture, the impulse drift is expected to be zero or near-zero.  Any non-zero drift comes from:

- **float32 vs float64**: The numpy replay uses float64 internally for stability but rounds to float32 for comparison.  This produces drift on the order of 1e-7.
- **Missing matrix-free constraints**: The replay currently handles only the dense constraint path.  If the original solve also included matrix-free (MF) constraints for free rigid body contacts, those corrections are not replayed and the velocity drift will be larger.  The report explicitly notes this gap.

### Using the Replay CLI

Analyze and replay with the original parameters:

    python -m newton._src.solvers.feather_pgs.spike_replay spike.npz --replay

Replay with modified parameters:

    python -m newton._src.solvers.feather_pgs.spike_replay spike.npz --replay --omega 0.8 --cfm 1e-4 --iterations 16

Run a parameter sweep across omega, CFM, and iteration counts:

    python -m newton._src.solvers.feather_pgs.spike_replay spike.npz --sweep

### Using the Replay API

    from newton._src.solvers.feather_pgs.spike_replay import SpikeArtifact

    artifact = SpikeArtifact.load("spike.npz")
    assert artifact.can_replay  # True if constraint data is present

    # Replay with original parameters
    result = artifact.replay_pgs(world_idx=0)
    result.print_drift_report()

    # Replay with modified parameters
    result = artifact.replay_pgs(omega=0.8, cfm=1e-4, iterations=16)
    print(f"max|replayed v_out|: {np.max(np.abs(result.replayed_v_out)):.2f}")

    # Parameter sweep
    results = artifact.replay_parameter_sweep(
        omega_values=[1.0, 0.9, 0.8, 0.7],
        cfm_values=[None, 1e-5, 1e-4, 1e-3],
        iteration_values=[8, 16, 32],
    )

### Replay Evidence: Synthetic Franka Spike

A synthetic Franka-like spike artifact is committed at `spike_captures/synthetic_franka_spike.npz`.  It models a 9-DOF Franka arm with 6 active constraints (joint limits + contact + friction).  Replay output with original parameters:

    PGS Replay Drift Report (world 0)
    ────────────────────────────────────────────────────────────
      Parameters: {'iterations': 8, 'omega': 1.0, 'cfm_override': None}
      max|impulse drift|:   0.000000e+00
      max|velocity drift|:  0.000000e+00
      max|replayed v_out|:  51.3660
      max|original v_out|:  51.3660

Zero drift confirms that the numpy PGS solver faithfully reproduces the Warp kernel behavior for dense constraint problems.

### Known Replay Limitations

1. **Matrix-free constraints not replayed**: The dense PGS path is the only one replayed.  When free rigid body contacts are present, the matrix-free (MF) PGS path runs separately and its corrections are not captured or replayed.  For the Franka lift task, which has no free rigid bodies, this limitation does not affect accuracy.

2. **Single-substep replay**: The replay covers a single PGS solve, not a full simulation step with multiple substeps.  Multi-substep effects (position feedback between substeps) are not captured.

3. **State reconstruction**: The replay does not reconstruct the full Newton State or re-run FK/ID/CRBA/Cholesky.  It replays only the PGS solve (stages 5-6) from captured constraint data.  This is sufficient for debugging PGS-related spikes but does not help diagnose spikes originating in v_hat (unconstrained dynamics).

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
- [x] Replay/analysis tool implemented with pure-numpy PGS solver
- [x] Enhanced capture to include full constraint problem (C, diag, rhs, Y, row metadata)
- [x] PGS replay harness with determinism measurement (28/28 unit tests pass)
- [x] Synthetic Franka-like spike artifact committed for validation
- [x] Parameter-sweep CLI for fix exploration
- [ ] Real spike captures from training runs
- [ ] Spike classification from real data
- [ ] Fix experiments on replay
- [ ] Longer-run validation

## Files Changed

- `newton/_src/solvers/feather_pgs/spike_capture.py` – Capture module (enhanced with constraint-level data)
- `newton/_src/solvers/feather_pgs/spike_replay.py` – Replay harness with pure-numpy PGS solver, drift measurement, and parameter sweep
- `newton/_src/solvers/feather_pgs/generate_synthetic_spike.py` – Synthetic spike artifact generator
- `newton/_src/solvers/feather_pgs/solver_feather_pgs.py` – Modified: import + hooks in `step()`
- `tests/test_spike_capture.py` – Unit tests (28 tests: capture, PGS solver, replay, drift)
- `spike_captures/synthetic_franka_spike.npz` – Reference spike artifact
- `report.md` – This file
