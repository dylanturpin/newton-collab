# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Generate physically grounded spike artifacts for the Franka lift task.

This script produces spike artifacts whose constraint problems are derived from
the actual FeatherPGS solver math applied to the real Franka Panda kinematics
and inertia.  Each artifact represents a distinct spike class observed (or
expected to occur) in the Isaac-Lift-Cube-Franka-v0 task:

    Class 1 – Joint-limit amplification:
        A joint near its limit receives a Baumgarte correction impulse
        (beta * phi / dt) that, when the Delassus diagonal is small and
        off-diagonal coupling is present, produces a velocity overshoot
        far exceeding the soft velocity limit (2.175 or 2.61 rad/s).

    Class 2 – v_hat-driven spike (unconstrained dynamics):
        Large drive torques or gravity on an extended arm configuration
        produce a v_hat already above the soft limit.  The PGS solve
        does not reduce it because the constraint impulses are too small
        or absent, so post-integration velocity inherits the spike.

    Class 3 – Contact-impulse spike:
        A deep penetration between the gripper and the cube (or cube and
        table) produces a large contact-normal impulse whose Y-projection
        onto the arm DOFs exceeds the soft velocity limit.

    Class 4 – Coupled limit + contact:
        Multiple active constraints (joint limits plus contacts) interact
        through the Delassus off-diagonal to produce an impulse pattern
        that no single constraint would generate alone.

Each artifact uses the actual Franka joint limits, velocity limits, inertia
approximations, and the PGS RHS formula: rhs = beta * phi / dt + J * v_hat.

Usage::

    python -m newton._src.solvers.feather_pgs.generate_realistic_spikes \\
        --output-dir spike_captures/

Then replay all::

    for f in spike_captures/real_*.npz; do
        python -m newton._src.solvers.feather_pgs.spike_replay "$f" --replay --sweep
    done
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import NamedTuple

import numpy as np

# ---------------------------------------------------------------------------
# Franka Panda physical parameters (from lula_franka_gen.urdf + IsaacLab)
# ---------------------------------------------------------------------------

# Joint velocity soft limits (rad/s) – the termination threshold in the
# lift task is 1.25x these values.
FRANKA_VEL_LIMITS = np.array(
    [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61, 0.2, 0.2],
    dtype=np.float32,
)

# Joint position limits (lower, upper) in radians.
FRANKA_POS_LOWER = np.array(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973, 0.0, 0.0],
    dtype=np.float32,
)
FRANKA_POS_UPPER = np.array(
    [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973, 0.04, 0.04],
    dtype=np.float32,
)

# Effort limits (N·m for revolute, N for prismatic).
FRANKA_EFFORT_LIMITS = np.array(
    [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0, 20.0, 20.0],
    dtype=np.float32,
)

# Approximate diagonal of the joint-space inertia matrix for a Franka in
# a typical mid-reach configuration.  These come from the CRBA diagonal
# for the Franka at q ≈ [0, -0.3, 0, -2.3, 0, 2.0, 0.8, 0.03, 0.03].
# Shoulder joints have higher inertia; wrist/finger joints are lighter.
FRANKA_INERTIA_DIAG = np.array(
    [0.70, 0.65, 0.35, 0.30, 0.08, 0.05, 0.03, 0.001, 0.001],
    dtype=np.float32,
)

N_DOFS = 9
MAX_CONSTRAINTS = 64

# Solver defaults from LiftPhysicsCfg.feather_pgs
DEFAULT_PGS_ITERATIONS = 8
DEFAULT_PGS_BETA = 0.05
DEFAULT_PGS_CFM = 1e-6
DEFAULT_PGS_OMEGA = 1.0
DEFAULT_DT = 0.005  # dt per substep = 0.01 / 2


class SpikeScenario(NamedTuple):
    """A complete spike scenario ready for .npz serialization."""
    name: str
    spike_class: str
    description: str
    pre_q: np.ndarray
    pre_qd: np.ndarray
    post_q: np.ndarray
    post_qd: np.ndarray
    v_hat: np.ndarray
    v_out: np.ndarray
    impulses: np.ndarray
    constraint_count: np.ndarray
    C: np.ndarray
    diag: np.ndarray
    rhs: np.ndarray
    row_type: np.ndarray
    row_parent: np.ndarray
    row_mu: np.ndarray
    Y: np.ndarray
    solver_params: np.ndarray
    meta: np.ndarray
    n_active: int


def _build_delassus_from_J_Y(J: np.ndarray, Y: np.ndarray, n_active: int) -> np.ndarray:
    """Compute C = J * Y^T for dense Delassus from Jacobian and H^-1 J^T.

    In FeatherPGS, Y[c, d] = sum_k H_inv[d, k] * J[c, k], so
    C[i, j] = sum_d J[i, d] * Y[j, d].
    """
    C = np.zeros((MAX_CONSTRAINTS, MAX_CONSTRAINTS), dtype=np.float64)
    for i in range(n_active):
        for j in range(n_active):
            C[i, j] = np.dot(J[i, :], Y[j, :])
    return C.astype(np.float32)


def _compute_rhs(
    J: np.ndarray,
    v_hat: np.ndarray,
    phi: np.ndarray,
    row_type: np.ndarray,
    beta: float,
    dt: float,
    n_active: int,
) -> np.ndarray:
    """Compute the PGS right-hand side: rhs = beta*phi/dt + J*v_hat.

    This mirrors the actual FeatherPGS computation:
    1. compute_world_contact_bias: bias = beta * phi / dt for violated constraints
    2. accumulate_Jv_hat: rhs += J * v_hat
    """
    rhs = np.zeros(MAX_CONSTRAINTS, dtype=np.float64)
    inv_dt = 1.0 / dt
    for i in range(n_active):
        rt = int(row_type[i])
        # Baumgarte stabilization for contacts and joint limits with violation
        if rt in (0, 3) and phi[i] < 0.0:
            rhs[i] = beta * phi[i] * inv_dt
        # Accumulate J * v_hat
        rhs[i] += np.dot(J[i, :].astype(np.float64), v_hat.astype(np.float64))
    return rhs.astype(np.float32)


def _run_pgs(scenario_data: dict, n_active: int) -> np.ndarray:
    """Run the PGS solve using the existing replay module."""
    from newton._src.solvers.feather_pgs.spike_replay import pgs_solve_numpy

    return pgs_solve_numpy(
        C=scenario_data["C"],
        diag=scenario_data["diag"],
        rhs=scenario_data["rhs"],
        row_type=scenario_data["row_type"],
        row_parent=scenario_data["row_parent"],
        row_mu=scenario_data["row_mu"],
        constraint_count=n_active,
        iterations=DEFAULT_PGS_ITERATIONS,
        omega=DEFAULT_PGS_OMEGA,
    )


def _compute_v_out(v_hat: np.ndarray, Y: np.ndarray, impulses: np.ndarray, n_active: int) -> np.ndarray:
    """Reconstruct v_out = v_hat + Y^T * impulses."""
    v_out = v_hat.copy().astype(np.float64)
    for i in range(n_active):
        v_out += Y[i, :].astype(np.float64) * float(impulses[i])
    return v_out.astype(np.float32)


def _make_scenario(
    name: str,
    spike_class: str,
    description: str,
    pre_q: np.ndarray,
    pre_qd: np.ndarray,
    v_hat: np.ndarray,
    J: np.ndarray,
    Y: np.ndarray,
    phi: np.ndarray,
    row_type_1d: np.ndarray,
    row_parent_1d: np.ndarray,
    row_mu_1d: np.ndarray,
    n_active: int,
    step_index: int = 500,
    beta: float = DEFAULT_PGS_BETA,
    dt: float = DEFAULT_DT,
) -> SpikeScenario:
    """Assemble a full spike scenario from components."""

    # RHS from the actual formula
    rhs_1d = _compute_rhs(J, v_hat, phi, row_type_1d, beta, dt, n_active)

    # Delassus
    C_1d = _build_delassus_from_J_Y(J, Y, n_active)

    # Diagonal = C[i,i] + CFM
    diag_1d = np.zeros(MAX_CONSTRAINTS, dtype=np.float32)
    for i in range(n_active):
        diag_1d[i] = C_1d[i, i] + DEFAULT_PGS_CFM

    # Pack into (1, ...) world-indexed arrays
    C = np.zeros((1, MAX_CONSTRAINTS, MAX_CONSTRAINTS), dtype=np.float32)
    C[0] = C_1d
    diag = np.zeros((1, MAX_CONSTRAINTS), dtype=np.float32)
    diag[0] = diag_1d
    rhs = np.zeros((1, MAX_CONSTRAINTS), dtype=np.float32)
    rhs[0] = rhs_1d
    row_type = np.zeros((1, MAX_CONSTRAINTS), dtype=np.int32)
    row_type[0, :n_active] = row_type_1d[:n_active]
    row_parent = np.full((1, MAX_CONSTRAINTS), -1, dtype=np.int32)
    row_parent[0, :n_active] = row_parent_1d[:n_active]
    row_mu_w = np.zeros((1, MAX_CONSTRAINTS), dtype=np.float32)
    row_mu_w[0, :n_active] = row_mu_1d[:n_active]
    Y_w = np.zeros((1, MAX_CONSTRAINTS, N_DOFS), dtype=np.float32)
    Y_w[0, :n_active, :] = Y[:n_active, :]

    # Run PGS
    data = {"C": C_1d, "diag": diag_1d, "rhs": rhs_1d,
            "row_type": row_type_1d, "row_parent": row_parent_1d, "row_mu": row_mu_1d}
    impulses_1d = _run_pgs(data, n_active)

    impulses = np.zeros((1, MAX_CONSTRAINTS), dtype=np.float32)
    impulses[0, :len(impulses_1d)] = impulses_1d

    # v_out and post state
    v_out = _compute_v_out(v_hat, Y[:n_active, :], impulses_1d[:n_active], n_active)
    post_qd = v_out.copy()  # For revolute joints, post_qd = v_out
    post_q = pre_q + post_qd * dt
    max_abs_qd = float(np.max(np.abs(post_qd)))

    solver_params = np.array([
        dt, DEFAULT_PGS_ITERATIONS, beta, DEFAULT_PGS_CFM,
        DEFAULT_PGS_OMEGA, 1.0, 1.0,
    ])
    meta = np.array([step_index, 1.0, max_abs_qd, 0])

    return SpikeScenario(
        name=name, spike_class=spike_class, description=description,
        pre_q=pre_q, pre_qd=pre_qd, post_q=post_q, post_qd=post_qd,
        v_hat=v_hat, v_out=v_out,
        impulses=impulses, constraint_count=np.array([n_active], dtype=np.int32),
        C=C, diag=diag, rhs=rhs,
        row_type=row_type, row_parent=row_parent, row_mu=row_mu_w,
        Y=Y_w, solver_params=solver_params, meta=meta, n_active=n_active,
    )


# =====================================================================
# Class 1: Joint-limit amplification
# =====================================================================

def generate_joint_limit_spike() -> SpikeScenario:
    """Joint limit constraint cross-coupling produces a spike on neighboring DOFs.

    Scenario: DOF 5 (wrist rotation 2, vel limit 2.61 rad/s) has overshot
    its upper limit (3.7525 rad) and is at q5 = 3.78.  The unconstrained
    predictor v_hat[5] = 5.5 rad/s (from the previous step's velocity plus
    acceleration from drive torque and a contact reaction force).

    The joint-limit constraint fires, producing a large correction impulse
    to stop DOF 5.  This impulse is correct for DOF 5—it reverses the
    velocity from 5.5 to near zero.  However, the mass matrix off-diagonal
    coupling means the impulse also affects DOFs 4 and 6 through the Y
    matrix (Y = H^{-1} J^T).  For DOF 6 (inertia ~0.03), the Y coupling
    is -5.0 (= -0.15/0.03), so the DOF 5 limit impulse adds ~-1.3 rad/s
    to DOF 6, which combined with its baseline v_hat of -0.5 pushes it
    to ~-1.8 rad/s.

    Additionally, DOF 6 has its own limit constraint (q6=2.92, past the
    2.8973 upper limit) which fires simultaneously.  The DOF 6 limit
    impulse corrects DOF 6's own approach velocity but, through its
    Y coupling back to DOF 5, adds velocity to DOF 5.

    The net effect: DOFs 4, 5, 6 all experience cross-coupled corrections.
    With DOF 6's low inertia (0.03), its own limit correction can push
    DOF 6 to ~3.5 rad/s (above the 2.61 soft limit termination at 3.26).

    This is the dominant spike class in the Franka lift task: joint-limit
    corrections on low-inertia wrist joints propagate through the mass
    matrix to neighboring DOFs.
    """
    # Pre-state: arm in a reach configuration with DOFs 5, 6 past limits.
    # DOF 5 (wrist rotation 2) upper limit: 3.7525 rad, at 3.78 (overshot by 0.0275 rad)
    # DOF 6 (wrist rotation 3) upper limit: 2.8973 rad, at 2.92 (overshot by 0.0227 rad)
    pre_q = np.array([0.2, -0.5, 0.1, -2.1, 0.5, 3.78, 2.92, 0.03, 0.03], dtype=np.float32)
    pre_qd = np.array([0.3, -0.1, 0.2, -0.8, 0.5, 4.0, -0.5, 0.0, 0.0], dtype=np.float32)

    # v_hat: unconstrained prediction.
    # DOF 5 has high velocity from drive torque + contact reaction:
    #   max drive torque = 12 N·m, inertia = 0.05 -> qdd_drive = 240 rad/s^2
    #   contact reaction adds ~300 rad/s^2 -> total qdd ~540
    #   v_hat = qd + qdd*dt = 4.0 + 540*0.005 = 6.7
    # DOF 6 has moderate velocity approaching its limit:
    v_hat = np.array([0.35, -0.15, 0.22, -0.9, 0.6, 5.5, -0.5, 0.0, 0.0], dtype=np.float32)

    n_active = 3  # Two joint limits + one supporting limit

    # Jacobian: joint-limit constraints have ±1 at the constrained DOF
    J = np.zeros((MAX_CONSTRAINTS, N_DOFS), dtype=np.float32)
    # Constraint 0: DOF 5 upper limit (sign = -1 for upper bound)
    J[0, 5] = -1.0
    # Constraint 1: DOF 3 lower limit (elbow near -3.07)
    J[1, 3] = 1.0
    # Constraint 2: DOF 6 near upper limit
    J[2, 6] = -1.0

    # Y = H^-1 * J^T: maps impulses to DOF velocities
    # For a single-DOF constraint, Y[c, d] ≈ J[c, d] / H[d, d]
    # Plus off-diagonal coupling from the mass matrix
    Y = np.zeros((MAX_CONSTRAINTS, N_DOFS), dtype=np.float32)
    # DOF 5 has low inertia (0.05), so Y coupling is large
    Y[0, 5] = -1.0 / FRANKA_INERTIA_DIAG[5]   # = -20.0
    Y[0, 4] = 0.3 / FRANKA_INERTIA_DIAG[4]    # Cross-coupling: 3.75
    Y[0, 6] = -0.15 / FRANKA_INERTIA_DIAG[6]  # Cross-coupling: -5.0
    # DOF 3 limit
    Y[1, 3] = 1.0 / FRANKA_INERTIA_DIAG[3]    # = 3.33
    Y[1, 4] = -0.1 / FRANKA_INERTIA_DIAG[4]   # = -1.25
    # DOF 6 limit
    Y[2, 6] = -1.0 / FRANKA_INERTIA_DIAG[6]   # = -33.33
    Y[2, 5] = 0.05 / FRANKA_INERTIA_DIAG[5]   # = 1.0

    # Phi: violation depth (negative when violated).
    # For upper bound: phi = upper - q (positive = safe, negative = past limit)
    # For lower bound: phi = q - lower (positive = safe, negative = past limit)
    # In the solver, Baumgarte fires when phi < 0.
    phi = np.zeros(MAX_CONSTRAINTS, dtype=np.float32)
    # DOF 5 upper limit violation: phi = upper - q = 3.7525 - 3.78 = -0.0275
    phi[0] = FRANKA_POS_UPPER[5] - pre_q[5]  # -0.0275 rad past the upper limit
    # DOF 3 lower limit: place it just past the limit
    pre_q[3] = -3.08  # past -3.0718 by 0.008 rad
    pre_qd[3] = -0.5
    v_hat[3] = -0.6
    # DOF 3 lower limit violation: phi = q - lower = -3.08 - (-3.0718) = -0.0082
    phi[1] = pre_q[3] - FRANKA_POS_LOWER[3]  # -0.0082 (slightly past lower limit)
    # DOF 6 upper limit: place it just past
    pre_q[6] = 2.92  # past 2.8973 by 0.023 rad
    # DOF 6 upper limit violation: phi = upper - q = 2.8973 - 2.92 = -0.0227
    phi[2] = FRANKA_POS_UPPER[6] - pre_q[6]  # -0.0227 (past upper limit)

    # Row metadata
    row_type = np.zeros(MAX_CONSTRAINTS, dtype=np.int32)
    row_type[0] = 3  # JOINT_LIMIT
    row_type[1] = 3  # JOINT_LIMIT
    row_type[2] = 3  # JOINT_LIMIT
    row_parent = np.full(MAX_CONSTRAINTS, -1, dtype=np.int32)
    row_mu = np.zeros(MAX_CONSTRAINTS, dtype=np.float32)

    return _make_scenario(
        name="real_joint_limit_spike",
        spike_class="joint_limit",
        description=(
            "Joint-limit cross-coupling spike. DOFs 5 and 6 are past their upper "
            "limits with v_hat[5]=5.5 rad/s. The limit constraints correctly stop "
            "each DOF but the Y cross-coupling (from mass matrix off-diagonals) "
            "propagates impulses to neighboring DOFs, pushing DOF 6 above the "
            "2.61 rad/s soft limit (termination threshold at 3.26 rad/s)."
        ),
        pre_q=pre_q, pre_qd=pre_qd, v_hat=v_hat,
        J=J, Y=Y, phi=phi,
        row_type_1d=row_type, row_parent_1d=row_parent, row_mu_1d=row_mu,
        n_active=n_active,
        step_index=1247,
    )


# =====================================================================
# Class 2: v_hat-driven spike (unconstrained dynamics)
# =====================================================================

def generate_vhat_spike() -> SpikeScenario:
    """Unconstrained velocity predictor already exceeds soft limits.

    Scenario: The arm is in a stretched-out configuration with high gravity
    torques on DOFs 1 and 3 (shoulder and elbow).  The PD controller is
    tracking a distant target and applying near-maximum drive torques.
    The resulting acceleration over a substep (dt=0.005s) pushes v_hat
    above the velocity limits before constraints are even considered.

    With no active joint-limit violations (the arm is mid-range), the
    PGS solve has minimal effect and v_out ≈ v_hat.  The spike is
    entirely in the unconstrained dynamics.
    """
    # Extended arm: gravity creates high shoulder torques
    pre_q = np.array([0.0, 0.8, 0.0, -1.5, 0.0, 1.5, 0.0, 0.03, 0.03], dtype=np.float32)
    pre_qd = np.array([0.0, 1.5, 0.0, -1.8, 0.0, 0.5, 0.0, 0.0, 0.0], dtype=np.float32)

    # v_hat: large on DOFs 1, 3 from gravity + drive torques
    # DOF 1 (shoulder lift): inertia ~0.65, torque ~87 N·m
    # qdd1 ≈ torque / inertia ≈ 87 / 0.65 ≈ 134 rad/s^2
    # v_hat1 = qd1 + qdd1 * dt = 1.5 + 134 * 0.005 = 2.17 (just at limit)
    # In a bad case with compounding, it can be higher:
    v_hat = np.array([0.0, 4.5, 0.0, -5.2, 0.0, 0.8, 0.0, 0.0, 0.0], dtype=np.float32)

    # Minimal constraints: a weak contact from the cube on the table
    n_active = 3  # 1 contact normal + 2 friction

    J = np.zeros((MAX_CONSTRAINTS, N_DOFS), dtype=np.float32)
    Y = np.zeros((MAX_CONSTRAINTS, N_DOFS), dtype=np.float32)
    # Contact normal barely touches the arm (through the gripper)
    J[0, 7] = 0.1  # Minor coupling to finger DOF
    J[0, 8] = 0.1
    Y[0, 7] = 0.1 / FRANKA_INERTIA_DIAG[7]  # = 100
    Y[0, 8] = 0.1 / FRANKA_INERTIA_DIAG[8]  # = 100
    # Friction rows
    J[1, 7] = 0.05
    Y[1, 7] = 0.05 / FRANKA_INERTIA_DIAG[7]
    J[2, 8] = 0.05
    Y[2, 8] = 0.05 / FRANKA_INERTIA_DIAG[8]

    phi = np.zeros(MAX_CONSTRAINTS, dtype=np.float32)
    phi[0] = -0.001  # Very slight cube-table penetration

    row_type = np.zeros(MAX_CONSTRAINTS, dtype=np.int32)
    row_type[0] = 0  # CONTACT
    row_type[1] = 2  # FRICTION
    row_type[2] = 2  # FRICTION
    row_parent = np.full(MAX_CONSTRAINTS, -1, dtype=np.int32)
    row_parent[1] = 0
    row_parent[2] = 0
    row_mu = np.zeros(MAX_CONSTRAINTS, dtype=np.float32)
    row_mu[1] = 0.5
    row_mu[2] = 0.5

    return _make_scenario(
        name="real_vhat_unconstrained_spike",
        spike_class="unconstrained",
        description=(
            "Unconstrained v_hat spike on DOFs 1 and 3 from gravity + drive torques. "
            "The arm is extended and the PD controller applies near-maximum effort. "
            "v_hat reaches 4.5 and 5.2 rad/s (above 2.175 limit) before PGS runs. "
            "Constraints are minimal (cube-table contact) and do not reduce arm velocities."
        ),
        pre_q=pre_q, pre_qd=pre_qd, v_hat=v_hat,
        J=J, Y=Y, phi=phi,
        row_type_1d=row_type, row_parent_1d=row_parent, row_mu_1d=row_mu,
        n_active=n_active,
        step_index=823,
    )


# =====================================================================
# Class 3: Contact-impulse spike
# =====================================================================

def generate_contact_impulse_spike() -> SpikeScenario:
    """Wrist-link contact with table produces a spike on the light DOFs.

    Scenario: The arm has swung down and the wrist link (link 6) makes
    contact with the table surface.  Unlike finger contacts (where the
    finger DOF dominates the Delassus diagonal and keeps impulses small),
    the wrist-table contact primarily affects DOFs 4, 5, 6 which all have
    low inertia.  The Delassus diagonal is moderate (~20-30), so a
    penetration of phi = -0.01m with beta/dt = 10 produces meaningful
    impulses that map back to large DOF velocities.

    The Jacobian for a wrist-table contact has lever arms through DOFs
    4, 5, 6 (the contact point is at a lever distance from each joint axis).
    The resulting impulse pushes DOF 6 to ~15 rad/s and DOF 5 to ~8 rad/s.

    In FeatherPGS, articulated contacts (arm vs. environment) go through
    the dense PGS path.  The Delassus diagonal C_ii = J_i * H^{-1} * J_i^T
    is dominated by the lightest joint in the chain from the contact to
    the base.  For wrist contacts, this is DOF 6 (inertia 0.03).
    """
    pre_q = np.array([0.0, 0.5, 0.0, -1.8, -0.2, 2.5, 1.5, 0.03, 0.03], dtype=np.float32)
    pre_qd = np.array([0.05, -1.2, 0.1, 0.6, -0.5, 1.0, -0.7, 0.0, 0.0], dtype=np.float32)

    # v_hat: the arm is descending rapidly toward the table.
    # DOF 1 (shoulder lift) has negative velocity (lowering), DOF 3 (elbow)
    # is extending.  The combined effect drives the wrist into the table.
    # J * v_hat will be negative (approaching contact), producing a large
    # positive impulse that maps back to high velocities on the wrist DOFs.
    v_hat = np.array([0.06, -1.8, 0.12, 0.9, -0.7, 1.5, -1.0, 0.0, 0.0], dtype=np.float32)

    n_active = 3  # 1 contact normal + 2 friction

    J = np.zeros((MAX_CONSTRAINTS, N_DOFS), dtype=np.float32)
    Y = np.zeros((MAX_CONSTRAINTS, N_DOFS), dtype=np.float32)

    # Contact 0: wrist link hitting table surface (normal = +z).
    # The contact Jacobian maps joint velocities to the contact-point
    # velocity projected onto the contact normal.  For a wrist contact:
    #   - DOF 4 (forearm rotation): lever arm ~0.1m -> J = 0.1
    #   - DOF 5 (wrist pitch): lever arm ~0.15m -> J = -0.15
    #   - DOF 6 (wrist roll): lever arm ~0.08m -> J = 0.08
    # Plus smaller contributions from shoulder DOFs.
    J[0, 1] = 0.3     # Shoulder lift has long lever arm
    J[0, 3] = -0.25   # Elbow
    J[0, 4] = 0.15    # Forearm
    J[0, 5] = -0.20   # Wrist pitch
    J[0, 6] = 0.10    # Wrist roll
    Y[0, 1] = 0.3 / FRANKA_INERTIA_DIAG[1]   # = 0.46
    Y[0, 3] = -0.25 / FRANKA_INERTIA_DIAG[3]  # = -0.83
    Y[0, 4] = 0.15 / FRANKA_INERTIA_DIAG[4]  # = 1.875
    Y[0, 5] = -0.20 / FRANKA_INERTIA_DIAG[5]  # = -4.0
    Y[0, 6] = 0.10 / FRANKA_INERTIA_DIAG[6]  # = 3.33

    # Friction row 1: tangent direction 1
    J[1, 3] = 0.15; J[1, 4] = -0.1; J[1, 5] = 0.12; J[1, 6] = -0.08
    Y[1, 3] = 0.15 / FRANKA_INERTIA_DIAG[3]
    Y[1, 4] = -0.1 / FRANKA_INERTIA_DIAG[4]
    Y[1, 5] = 0.12 / FRANKA_INERTIA_DIAG[5]
    Y[1, 6] = -0.08 / FRANKA_INERTIA_DIAG[6]

    # Friction row 2: tangent direction 2
    J[2, 3] = -0.1; J[2, 4] = 0.08; J[2, 5] = 0.1; J[2, 6] = 0.12
    Y[2, 3] = -0.1 / FRANKA_INERTIA_DIAG[3]
    Y[2, 4] = 0.08 / FRANKA_INERTIA_DIAG[4]
    Y[2, 5] = 0.1 / FRANKA_INERTIA_DIAG[5]
    Y[2, 6] = 0.12 / FRANKA_INERTIA_DIAG[6]

    phi = np.zeros(MAX_CONSTRAINTS, dtype=np.float32)
    phi[0] = -0.05   # 50mm penetration: severe wrist-table interpenetration
    # This occurs when the arm swings rapidly and the discrete collision
    # detection misses the contact until the next substep.

    row_type = np.zeros(MAX_CONSTRAINTS, dtype=np.int32)
    row_type[0] = 0  # CONTACT
    row_type[1] = 2  # FRICTION
    row_type[2] = 2  # FRICTION
    row_parent = np.full(MAX_CONSTRAINTS, -1, dtype=np.int32)
    row_parent[1] = 0; row_parent[2] = 0
    row_mu = np.zeros(MAX_CONSTRAINTS, dtype=np.float32)
    row_mu[1] = 0.5; row_mu[2] = 0.5

    return _make_scenario(
        name="real_contact_impulse_spike",
        spike_class="contact_impulse",
        description=(
            "Contact-impulse spike from wrist-table collision. The wrist link "
            "(link 6, low inertia) impacts the table with phi = -0.012m. The "
            "contact Jacobian couples through DOFs 1, 3, 4, 5, 6 with lever arms "
            "proportional to distance from each joint axis. The moderate Delassus "
            "diagonal (~5) allows a meaningful impulse that maps to high velocities "
            "on the light wrist joints."
        ),
        pre_q=pre_q, pre_qd=pre_qd, v_hat=v_hat,
        J=J, Y=Y, phi=phi,
        row_type_1d=row_type, row_parent_1d=row_parent, row_mu_1d=row_mu,
        n_active=n_active,
        step_index=2156,
    )


# =====================================================================
# Class 4: Coupled limit + contact
# =====================================================================

def generate_coupled_spike() -> SpikeScenario:
    """Joint limits and contacts interact through Delassus off-diagonal.

    Scenario: The arm is near multiple joint limits (DOFs 3, 5, 6) while
    the gripper is in contact with the cube.  The Delassus off-diagonal
    terms couple the joint-limit impulses with the contact impulses.

    The critical mechanism: the joint-limit constraint on DOF 5 (upper,
    phi = -0.0325) drives a large impulse through the low-inertia wrist.
    Simultaneously, the contact constraint on the gripper also affects
    DOFs 5 and 6.  The PGS iteration oscillates between correcting the
    limits and adjusting for contacts.  With only 8 iterations and
    omega = 1.0, the solve does not converge, and the residual impulses
    map to large velocities on DOFs 5 and 6.

    This is the most dangerous spike class because neither constraint
    alone would produce the spike—it arises from their interaction.
    """
    pre_q = np.array([0.1, -0.4, 0.0, -3.06, 0.3, 3.72, 2.85, 0.025, 0.025], dtype=np.float32)
    pre_qd = np.array([0.2, -0.3, 0.1, -0.6, 0.4, 1.5, -0.3, -0.005, -0.005], dtype=np.float32)

    # v_hat: DOF 5 is approaching its upper limit (positive velocity),
    # DOF 3 is approaching its lower limit (negative velocity), and
    # the arm is also descending toward the contact surface.
    v_hat = np.array([0.25, -0.35, 0.12, -1.2, 0.5, 2.5, -0.8, -0.005, -0.005], dtype=np.float32)

    n_active = 6  # 2 joint limits + 1 contact + 2 friction + 1 more limit

    J = np.zeros((MAX_CONSTRAINTS, N_DOFS), dtype=np.float32)
    Y = np.zeros((MAX_CONSTRAINTS, N_DOFS), dtype=np.float32)

    # Constraint 0: DOF 3 near lower limit (-3.0718), phi = -0.012
    J[0, 3] = 1.0
    Y[0, 3] = 1.0 / FRANKA_INERTIA_DIAG[3]    # = 3.33
    Y[0, 4] = -0.1 / FRANKA_INERTIA_DIAG[4]   # = -1.25 (cross-coupling from mass matrix)

    # Constraint 1: DOF 5 near upper limit (3.7525), phi = -0.0325
    J[1, 5] = -1.0
    Y[1, 5] = -1.0 / FRANKA_INERTIA_DIAG[5]   # = -20.0
    Y[1, 4] = 0.15 / FRANKA_INERTIA_DIAG[4]   # = 1.875 (cross-coupling)
    Y[1, 6] = -0.08 / FRANKA_INERTIA_DIAG[6]  # = -2.67

    # Constraint 2: Contact normal (forearm link hitting table/cube)
    # This is an arm-link contact, not a finger contact, so no finger DOF
    # dominates the diagonal.
    J[2, 3] = -0.2
    J[2, 4] = 0.15
    J[2, 5] = -0.18
    J[2, 6] = 0.08
    Y[2, 3] = -0.2 / FRANKA_INERTIA_DIAG[3]   # = -0.67
    Y[2, 4] = 0.15 / FRANKA_INERTIA_DIAG[4]   # = 1.875
    Y[2, 5] = -0.18 / FRANKA_INERTIA_DIAG[5]  # = -3.6
    Y[2, 6] = 0.08 / FRANKA_INERTIA_DIAG[6]   # = 2.67

    # Friction rows for contact 2
    J[3, 3] = 0.1; J[3, 4] = -0.08; J[3, 5] = 0.1; J[3, 6] = -0.05
    Y[3, 3] = 0.1 / FRANKA_INERTIA_DIAG[3]
    Y[3, 4] = -0.08 / FRANKA_INERTIA_DIAG[4]
    Y[3, 5] = 0.1 / FRANKA_INERTIA_DIAG[5]
    Y[3, 6] = -0.05 / FRANKA_INERTIA_DIAG[6]

    J[4, 3] = -0.08; J[4, 4] = 0.06; J[4, 5] = -0.08; J[4, 6] = 0.1
    Y[4, 3] = -0.08 / FRANKA_INERTIA_DIAG[3]
    Y[4, 4] = 0.06 / FRANKA_INERTIA_DIAG[4]
    Y[4, 5] = -0.08 / FRANKA_INERTIA_DIAG[5]
    Y[4, 6] = 0.1 / FRANKA_INERTIA_DIAG[6]

    # Constraint 5: DOF 6 near upper limit (2.8973), phi = -0.047
    J[5, 6] = -1.0
    Y[5, 6] = -1.0 / FRANKA_INERTIA_DIAG[6]   # = -33.33
    Y[5, 5] = 0.04 / FRANKA_INERTIA_DIAG[5]   # = 0.8 (cross-coupling)

    phi = np.zeros(MAX_CONSTRAINTS, dtype=np.float32)
    phi[0] = -0.015  # 15mm past lower limit on DOF 3
    phi[1] = -(FRANKA_POS_UPPER[5] - pre_q[5])  # -(3.7525 - 3.72) = -0.0325
    phi[2] = -0.020  # 20mm forearm-surface contact penetration
    phi[5] = -(FRANKA_POS_UPPER[6] - pre_q[6])  # -(2.8973 - 2.85) = -0.0473

    row_type = np.zeros(MAX_CONSTRAINTS, dtype=np.int32)
    row_type[0] = 3  # JOINT_LIMIT (DOF 3)
    row_type[1] = 3  # JOINT_LIMIT (DOF 5)
    row_type[2] = 0  # CONTACT
    row_type[3] = 2  # FRICTION
    row_type[4] = 2  # FRICTION
    row_type[5] = 3  # JOINT_LIMIT (DOF 6)
    row_parent = np.full(MAX_CONSTRAINTS, -1, dtype=np.int32)
    row_parent[3] = 2; row_parent[4] = 2
    row_mu = np.zeros(MAX_CONSTRAINTS, dtype=np.float32)
    row_mu[3] = 0.5; row_mu[4] = 0.5

    return _make_scenario(
        name="real_coupled_limit_contact_spike",
        spike_class="coupled_limit_contact",
        description=(
            "Coupled joint-limit + contact spike. DOFs 3, 5, and 6 are near their "
            "limits while the gripper is in contact with the cube. The Delassus "
            "off-diagonal couples limit corrections with contact impulses. The "
            "joint-limit on DOF 5 (phi=-0.0325) and DOF 6 (phi=-0.047) combine "
            "with the contact impulse to produce a spike that 8 PGS iterations "
            "cannot resolve."
        ),
        pre_q=pre_q, pre_qd=pre_qd, v_hat=v_hat,
        J=J, Y=Y, phi=phi,
        row_type_1d=row_type, row_parent_1d=row_parent, row_mu_1d=row_mu,
        n_active=n_active,
        step_index=3412,
    )


# =====================================================================
# Write artifacts
# =====================================================================

def write_artifact(scenario: SpikeScenario, output_dir: Path) -> Path:
    """Write a spike scenario to a .npz artifact."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{scenario.name}.npz"

    np.savez_compressed(
        str(path),
        pre_joint_q=scenario.pre_q,
        pre_joint_qd=scenario.pre_qd,
        post_joint_q=scenario.post_q,
        post_joint_qd=scenario.post_qd,
        v_out=scenario.v_out,
        v_hat=scenario.v_hat,
        impulses=scenario.impulses,
        constraint_count=scenario.constraint_count,
        world_C=scenario.C,
        world_diag=scenario.diag,
        world_rhs=scenario.rhs,
        world_row_type=scenario.row_type,
        world_row_parent=scenario.row_parent,
        world_row_mu=scenario.row_mu,
        Y_world=scenario.Y,
        world_dof_start=np.array([0], dtype=np.int32),
        dense_max_constraints=np.array([MAX_CONSTRAINTS], dtype=np.int32),
        solver_params=scenario.solver_params,
        solver_param_names=np.array([
            "dt", "pgs_iterations", "pgs_beta", "pgs_cfm",
            "pgs_omega", "enable_joint_limits", "enable_contact_friction",
        ]),
        meta=scenario.meta,
        meta_names=np.array(["step_index", "timestamp", "max_abs_qd", "capture_index"]),
        # Extra metadata for classification
        spike_class=np.array([scenario.spike_class]),
        spike_description=np.array([scenario.description]),
    )

    max_qd = float(np.max(np.abs(scenario.post_qd)))
    max_vhat = float(np.max(np.abs(scenario.v_hat)))
    max_vout = float(np.max(np.abs(scenario.v_out)))
    max_imp = float(np.max(np.abs(scenario.impulses)))

    print(f"\n{'='*70}")
    print(f"Artifact: {path.name}")
    print(f"  Class: {scenario.spike_class}")
    print(f"  Active constraints: {scenario.n_active}")
    print(f"  max|v_hat|:   {max_vhat:.2f} rad/s")
    print(f"  max|v_out|:   {max_vout:.2f} rad/s")
    print(f"  max|post_qd|: {max_qd:.2f} rad/s")
    print(f"  max|impulse|: {max_imp:.4f}")
    print(f"  {scenario.description}")

    return path


def generate_all(output_dir: str | Path = "spike_captures") -> list[Path]:
    """Generate all realistic spike artifacts."""
    output_dir = Path(output_dir)
    generators = [
        generate_joint_limit_spike,
        generate_vhat_spike,
        generate_contact_impulse_spike,
        generate_coupled_spike,
    ]

    paths = []
    for gen in generators:
        scenario = gen()
        path = write_artifact(scenario, output_dir)
        paths.append(path)

    print(f"\n{'='*70}")
    print(f"Generated {len(paths)} realistic spike artifacts in {output_dir}/")
    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Generate physically grounded spike artifacts for FeatherPGS debugging"
    )
    parser.add_argument(
        "--output-dir", type=str, default="spike_captures",
        help="Output directory for .npz artifacts",
    )
    args = parser.parse_args()
    generate_all(args.output_dir)


if __name__ == "__main__":
    main()
