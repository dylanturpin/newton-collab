# FeatherPGS Velocity Spike Investigation Report

This report is the working notebook for the `dt/velocity-spike-codex` branch. It will track how real FeatherPGS velocity spikes are captured, replayed, classified, and fixed. The current pass completes the first solver-side capture slice in `newton-collab`; it does not yet include a real Franka spike artifact or a replay harness.

## Current Status

The branch now includes an opt-in FeatherPGS capture path inside `newton/_src/solvers/feather_pgs/solver_feather_pgs.py`. The hook snapshots solver inputs immediately before the first projected Gauss-Seidel iteration in the active solve path:

- `dense`: just before the dense PGS loop starts.
- `split_dense_then_mf`: just before the dense phase begins in split mode when free-rigid mixed contacts are not interleaved.
- `split_mixed`: after matrix-free metadata is prepared and before the interleaved dense/matrix-free iteration loop starts.
- `matrix_free`: after world-space Jacobian gather and matrix-free metadata packing, before the solver kernel runs.

Capture is disabled by default. When enabled, the solver keeps one pending pre-solve snapshot in host memory and flushes it to disk only if the post-solve maximum absolute generalized velocity exceeds the configured threshold, or immediately for every frame if no threshold is configured.

## Capture Controls

Set these environment variables before launching the Isaac or Newton process that constructs `SolverFeatherPGS`:

- `NEWTON_FEATHER_PGS_CAPTURE_DIR=/abs/path/to/captures`
  Required. Enables capture and selects the output directory.
- `NEWTON_FEATHER_PGS_CAPTURE_VELOCITY_THRESHOLD=80`
  Optional. If set, only frames whose post-solve `max(abs(v_out))` reaches or exceeds this value are written.
- `NEWTON_FEATHER_PGS_CAPTURE_MAX_FRAMES=1`
  Optional. Limits the number of written frames for the current process. Defaults to `1`.

Example:

    export NEWTON_FEATHER_PGS_CAPTURE_DIR=/tmp/fpgs-captures
    export NEWTON_FEATHER_PGS_CAPTURE_VELOCITY_THRESHOLD=80
    export NEWTON_FEATHER_PGS_CAPTURE_MAX_FRAMES=2

The solver prints a short line when it writes a capture artifact, including the `.npz` payload path and matching `.json` metadata path.

## Artifact Format

Each flushed frame writes two files with a shared stem:

- `fpgs_capture_step000123_frame000.json`
- `fpgs_capture_step000123_frame000.npz`

The JSON sidecar holds capture metadata:

- solver step index and UTC creation time
- `dt`
- solver mode and solve-path label
- PGS parameters (`pgs_beta`, `pgs_cfm`, `pgs_omega`, `pgs_iterations`, `pgs_kernel`)
- constraint capacities (`dense_max_constraints`, `mf_max_constraints`)
- whether contact friction and joint limits were enabled
- the trigger threshold and the observed `max_abs_velocity`

The `.npz` payload stores the arrays needed to understand or later replay the solve inputs. The current slice captures:

- incoming state context: `state_in_joint_q`, `state_in_joint_qd`, `state_in_body_q`, `state_in_body_qd`, `qd_work`, `v_hat`
- row-routing and per-world solve buffers: `constraint_count`, `slot_counter`, `rhs`, `diag`, `impulses_pre_solve`, `row_type`, `row_parent`, `row_mu`, `row_beta`, `row_cfm`, `phi`, `target_velocity`
- dense contact-routing outputs: `contact_world`, `contact_slot`, `contact_art_a`, `contact_art_b`, `contact_path`
- per-size dense matrices: `J_by_size_<dof>`, `Y_by_size_<dof>`, `group_to_art_<dof>`
- dense world matrix when present: `C`
- world-space Jacobian gather when present: `J_world`, `Y_world`
- matrix-free buffers when present: `mf_constraint_count`, `mf_body_a`, `mf_body_b`, `mf_J_a`, `mf_J_b`, `mf_MiJt_a`, `mf_MiJt_b`, `mf_rhs`, `mf_eff_mass_inv`, `mf_row_type`, `mf_row_parent`, `mf_row_mu`, `mf_phi`, `mf_dof_a`, `mf_dof_b`, `mf_meta_packed`, body-map helpers, and pre/post-solve matrix-free impulses
- model and mapping context that affects row ordering: `art_to_world`, `art_size`, `art_group_idx`, `articulation_dof_start`, `articulation_H_rows`, `articulation_origin`, `body_to_articulation`, `body_to_joint`, `joint_ancestor`, `joint_qd_start`, `shape_body`, `shape_material_mu`
- raw rigid-contact inputs when available: shape ids, contact points, normals, margins, and the raw contact-count buffer
- post-solve outputs currently needed for spike triage: `v_out`, `impulses_post_solve`, and `mf_impulses_post_solve` when matrix-free buffers exist

## Known Gaps

This slice does not yet ship a replay script. The payload is designed so Stage 2 can choose between:

- rebuilding only the solve step from already prepared dense/matrix-free buffers, or
- rebuilding row assembly from the preserved contact and mapping context to verify row ordering.

This slice also does not yet include an Isaac-side invocation recipe because no real spike has been captured yet from `Isaac-Lift-Cube-Franka-v0`. That is the next step before Stage 1 can be considered complete.

## Validation Completed In This Pass

The branch includes `newton/tests/test_feather_pgs_capture.py`, which validates:

- environment parsing for the capture configuration
- rejection of invalid `MAX_FRAMES`
- creation of the paired `.json` and `.npz` artifacts

Further validation in later passes must confirm that a real run writes a capture when the configured threshold is crossed.
