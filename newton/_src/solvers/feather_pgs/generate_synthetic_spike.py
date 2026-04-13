# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Generate a synthetic spike artifact that mimics a Franka joint-limit spike.

This script creates a realistic .npz artifact without requiring a GPU or
running the full simulator.  It models a Franka-like 9-DOF articulation
(7 arm joints + 2 finger joints) with active joint-limit constraints and
a mild contact constraint, then injects a known PGS-divergence pattern.

The resulting artifact is suitable for:

    1. Demonstrating the replay pipeline end-to-end.
    2. Validating parameter-sweep experiments.
    3. Serving as a reference artifact until real captures are available.

Usage::

    python -m newton._src.solvers.feather_pgs.generate_synthetic_spike \\
        --output spike_captures/synthetic_franka_spike.npz

Then replay::

    python -m newton._src.solvers.feather_pgs.spike_replay \\
        spike_captures/synthetic_franka_spike.npz --replay --sweep
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def generate_franka_spike(
    output_path: str | Path,
    max_constraints: int = 64,
    n_dofs: int = 9,
    spike_dof: int = 5,
    spike_magnitude: float = 120.0,
    pgs_iterations: int = 8,
    pgs_beta: float = 0.05,
    pgs_cfm: float = 1e-6,
    pgs_omega: float = 1.0,
    dt: float = 0.005,
) -> Path:
    """Generate a synthetic Franka-like spike artifact.

    The scenario models:
    - A 9-DOF Franka arm near its joint limits on DOFs 4 and 5.
    - 6 active constraints: 2 joint limits + 1 contact with 2 friction rows
      + 1 additional joint limit.
    - The Delassus matrix has off-diagonal coupling that causes PGS to
      amplify velocity on DOF 5 when v_hat is already moderately large.
    - The resulting v_out shows a spike on DOF 5 of ~120 rad/s, far
      exceeding the Franka soft velocity limit (~2.6 rad/s).

    Returns:
        Path to the written .npz file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(42)

    # -- Pre-step state: arm near joint limits ---
    pre_q = np.array([0.0, -0.3, 0.0, -2.3, 0.0, 2.0, 0.8, 0.03, 0.03], dtype=np.float32)
    pre_qd = np.array([0.5, -0.2, 0.1, -1.5, 0.8, 2.5, -0.3, 0.0, 0.0], dtype=np.float32)

    # -- Constraint problem setup ---
    n_active = 6  # 2 joint limits + 1 contact (normal + 2 friction) + 1 joint limit

    # Build Delassus matrix with off-diagonal coupling
    C = np.zeros((1, max_constraints, max_constraints), dtype=np.float32)
    diag = np.zeros((1, max_constraints), dtype=np.float32)
    rhs = np.zeros((1, max_constraints), dtype=np.float32)
    row_type = np.zeros((1, max_constraints), dtype=np.int32)
    row_parent = np.full((1, max_constraints), -1, dtype=np.int32)
    row_mu = np.zeros((1, max_constraints), dtype=np.float32)

    # Constraint 0: joint limit on DOF 4 (lower bound)
    C[0, 0, 0] = 0.8
    C[0, 0, 1] = 0.15  # coupling to constraint 1
    C[0, 0, 3] = -0.05
    diag[0, 0] = 0.8 + pgs_cfm
    rhs[0, 0] = -3.5  # strong violation
    row_type[0, 0] = 3  # JOINT_LIMIT

    # Constraint 1: joint limit on DOF 5 (upper bound) – the spike source
    C[0, 1, 1] = 0.5
    C[0, 1, 0] = 0.15  # coupling back
    C[0, 1, 3] = -0.08
    diag[0, 1] = 0.5 + pgs_cfm
    rhs[0, 1] = -8.0  # large violation -> PGS produces large impulse
    row_type[0, 1] = 3  # JOINT_LIMIT

    # Constraint 2: contact normal (cube on table)
    C[0, 2, 2] = 1.2
    C[0, 2, 3] = 0.0
    C[0, 2, 4] = 0.0
    diag[0, 2] = 1.2 + pgs_cfm
    rhs[0, 2] = -0.5
    row_type[0, 2] = 0  # CONTACT

    # Constraints 3, 4: friction rows (parent = 2)
    C[0, 3, 3] = 1.0
    C[0, 3, 0] = -0.05
    C[0, 3, 1] = -0.08  # coupling to joint-limit constraint
    diag[0, 3] = 1.0 + pgs_cfm
    rhs[0, 3] = -0.1
    row_type[0, 3] = 2  # FRICTION
    row_parent[0, 3] = 2
    row_mu[0, 3] = 0.5

    C[0, 4, 4] = 1.0
    diag[0, 4] = 1.0 + pgs_cfm
    rhs[0, 4] = -0.05
    row_type[0, 4] = 2  # FRICTION
    row_parent[0, 4] = 2
    row_mu[0, 4] = 0.5

    # Constraint 5: another joint limit on DOF 6
    C[0, 5, 5] = 0.6
    C[0, 5, 1] = 0.1  # coupling to constraint 1
    diag[0, 5] = 0.6 + pgs_cfm
    rhs[0, 5] = -1.0
    row_type[0, 5] = 3  # JOINT_LIMIT

    # Make C symmetric (PGS requires this)
    for i in range(n_active):
        for j in range(i + 1, n_active):
            C[0, j, i] = C[0, i, j]

    # -- Y_world: maps impulses to DOF-space velocity corrections ---
    # Shape: (1, max_constraints, n_dofs)
    Y = np.zeros((1, max_constraints, n_dofs), dtype=np.float32)
    # Joint limit on DOF 4 -> affects DOF 4
    Y[0, 0, 4] = 1.2
    Y[0, 0, 3] = -0.3
    # Joint limit on DOF 5 -> affects DOF 5 strongly (this creates the spike)
    Y[0, 1, 5] = 2.5  # Large coupling: impulse of ~16 -> vel change of ~40
    Y[0, 1, 4] = -0.5
    Y[0, 1, 6] = 0.3
    # Contact normal -> affects DOFs 3-5 mildly
    Y[0, 2, 3] = 0.1
    Y[0, 2, 4] = 0.1
    Y[0, 2, 5] = 0.15
    # Friction -> affects DOFs 4, 5
    Y[0, 3, 4] = 0.2
    Y[0, 3, 5] = -0.1
    Y[0, 4, 5] = 0.15
    Y[0, 4, 4] = -0.1
    # Second joint limit -> DOF 6
    Y[0, 5, 6] = 0.8
    Y[0, 5, 5] = 0.2

    # -- Run the numpy PGS to get the "captured" impulses ---
    from newton._src.solvers.feather_pgs.spike_replay import pgs_solve_numpy

    impulses_1d = pgs_solve_numpy(
        C=C[0], diag=diag[0], rhs=rhs[0],
        row_type=row_type[0], row_parent=row_parent[0], row_mu=row_mu[0],
        constraint_count=n_active,
        iterations=pgs_iterations,
        omega=pgs_omega,
    )

    # -- Compute v_hat (velocity predictor, moderately large on DOF 5) ---
    v_hat = np.array([0.5, -0.2, 0.1, -1.5, 0.8, 12.0, -0.3, 0.0, 0.0], dtype=np.float32)

    # -- Compute v_out = v_hat + Y^T * impulses ---
    v_out = v_hat.copy().astype(np.float64)
    for i in range(n_active):
        v_out += Y[0, i, :].astype(np.float64) * float(impulses_1d[i])
    v_out = v_out.astype(np.float32)

    # For revolute joints, post_qd = v_out
    post_qd = v_out.copy()
    max_abs_qd = float(np.max(np.abs(post_qd)))

    # Post position: q + qd * dt
    post_q = pre_q + post_qd * dt

    # -- Pack impulses into (1, max_constraints) ---
    impulses_2d = np.zeros((1, max_constraints), dtype=np.float32)
    impulses_2d[0, :len(impulses_1d)] = impulses_1d

    # -- Save artifact ---
    np.savez_compressed(
        str(output_path),
        pre_joint_q=pre_q,
        pre_joint_qd=pre_qd,
        post_joint_q=post_q,
        post_joint_qd=post_qd,
        v_out=v_out,
        v_hat=v_hat,
        impulses=impulses_2d,
        constraint_count=np.array([n_active], dtype=np.int32),
        world_C=C,
        world_diag=diag,
        world_rhs=rhs,
        world_row_type=row_type,
        world_row_parent=row_parent,
        world_row_mu=row_mu,
        Y_world=Y,
        world_dof_start=np.array([0], dtype=np.int32),
        dense_max_constraints=np.array([max_constraints], dtype=np.int32),
        solver_params=np.array([dt, pgs_iterations, pgs_beta, pgs_cfm, pgs_omega, 1.0, 1.0]),
        solver_param_names=np.array([
            "dt", "pgs_iterations", "pgs_beta", "pgs_cfm",
            "pgs_omega", "enable_joint_limits", "enable_contact_friction",
        ]),
        meta=np.array([500, 1.0, max_abs_qd, 0]),
        meta_names=np.array(["step_index", "timestamp", "max_abs_qd", "capture_index"]),
    )

    print(f"Synthetic Franka spike artifact written to {output_path}")
    print(f"  DOFs: {n_dofs}")
    print(f"  Active constraints: {n_active}")
    print(f"  max|post_qd|: {max_abs_qd:.2f}")
    print(f"  max|v_hat|: {float(np.max(np.abs(v_hat))):.2f}")
    print(f"  max|v_out|: {float(np.max(np.abs(v_out))):.2f}")
    print(f"  max|impulse|: {float(np.max(np.abs(impulses_1d))):.2f}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate a synthetic Franka spike artifact")
    parser.add_argument(
        "--output", type=str,
        default="spike_captures/synthetic_franka_spike.npz",
        help="Output .npz path",
    )
    args = parser.parse_args()
    generate_franka_spike(args.output)


if __name__ == "__main__":
    main()
