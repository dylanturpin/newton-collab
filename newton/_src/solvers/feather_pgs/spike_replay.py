# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Offline replay and analysis of captured velocity spike artifacts.

This module provides two capabilities:

1. **Analysis**: Load a captured spike artifact and inspect solver state
   (velocity profiles, classification heuristics, impulse magnitudes).

2. **Replay**: Re-run the PGS solve in pure numpy from the captured
   constraint problem (Delassus matrix C, diagonal, RHS, row metadata).
   This reproduces the solver step without GPU or Warp, compares the
   replayed output to the original capture, and enables parameter-sweep
   experiments (e.g. different omega, CFM, iteration counts).

Usage as a CLI tool::

    # Analyze a spike artifact
    python -m newton._src.solvers.feather_pgs.spike_replay path/to/spike.npz

    # Replay with modified PGS parameters
    python -m newton._src.solvers.feather_pgs.spike_replay path/to/spike.npz \\
        --replay --omega 0.8 --cfm 1e-4 --iterations 16

Or as a library::

    from newton._src.solvers.feather_pgs.spike_replay import SpikeArtifact
    artifact = SpikeArtifact.load("spike_0000.npz")
    artifact.print_summary()

    # Re-run PGS with different parameters
    result = artifact.replay_pgs(omega=0.8, cfm=1e-4, iterations=16)
    result.print_drift_report()
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# PGS constraint type constants – must match kernels.py
PGS_CONSTRAINT_TYPE_CONTACT = 0
PGS_CONSTRAINT_TYPE_JOINT_TARGET = 1
PGS_CONSTRAINT_TYPE_FRICTION = 2
PGS_CONSTRAINT_TYPE_JOINT_LIMIT = 3


# ------------------------------------------------------------------
# Pure-numpy PGS solver (mirrors kernels.py pgs_solve_loop)
# ------------------------------------------------------------------

def pgs_solve_numpy(
    C: np.ndarray,
    diag: np.ndarray,
    rhs: np.ndarray,
    row_type: np.ndarray,
    row_parent: np.ndarray,
    row_mu: np.ndarray,
    constraint_count: int,
    iterations: int,
    omega: float,
    cfm_override: float | None = None,
) -> np.ndarray:
    """Run Projected Gauss-Seidel in pure numpy for a single world.

    This is a faithful port of ``pgs_solve_loop`` from ``kernels.py``.
    It operates on a single world's constraint problem.

    Args:
        C: Delassus matrix, shape ``(max_c, max_c)``.
        diag: Regularized diagonal, shape ``(max_c,)``.
        rhs: Right-hand side bias, shape ``(max_c,)``.
        row_type: Constraint type per row, shape ``(max_c,)``.
        row_parent: Parent row index (for friction), shape ``(max_c,)``.
        row_mu: Friction coefficient per row, shape ``(max_c,)``.
        constraint_count: Number of active constraints in this world.
        iterations: Number of PGS iterations.
        omega: SOR relaxation factor (1.0 = standard GS).
        cfm_override: If not None, add this to the diagonal (on top of
            whatever regularization the captured diagonal already has).

    Returns:
        Solved impulse vector, shape ``(max_c,)``.
    """
    m = constraint_count
    if m == 0:
        return np.zeros_like(rhs)

    max_c = len(rhs)
    impulses = np.zeros(max_c, dtype=np.float64)

    # Work in float64 for numerical stability in replay
    C64 = C.astype(np.float64)
    diag64 = diag.astype(np.float64)
    rhs64 = rhs.astype(np.float64)

    if cfm_override is not None and cfm_override > 0.0:
        diag64[:m] += cfm_override

    for _ in range(iterations):
        for i in range(m):
            # Compute residual: w = rhs_i + sum_j C_ij * lambda_j
            w = rhs64[i] + np.dot(C64[i, :m], impulses[:m])

            denom = diag64[i]
            if denom <= 0.0:
                continue

            delta = -w / denom
            new_impulse = impulses[i] + omega * delta
            rt = int(row_type[i])

            # Normal contact or joint limit: lambda >= 0
            if rt == PGS_CONSTRAINT_TYPE_CONTACT or rt == PGS_CONSTRAINT_TYPE_JOINT_LIMIT:
                if new_impulse < 0.0:
                    new_impulse = 0.0
                impulses[i] = new_impulse

            # Friction: project onto Coulomb cone
            elif rt == PGS_CONSTRAINT_TYPE_FRICTION:
                parent_idx = int(row_parent[i])
                lambda_n = impulses[parent_idx]
                mu = float(row_mu[i])
                radius = max(mu * lambda_n, 0.0)

                if radius <= 0.0:
                    impulses[i] = 0.0
                    continue

                impulses[i] = new_impulse

                # Sibling friction row: [normal, friction1, friction2]
                if i == parent_idx + 1:
                    sib = parent_idx + 2
                else:
                    sib = parent_idx + 1

                a = impulses[i]
                b = impulses[sib] if sib < max_c else 0.0
                mag = np.sqrt(a * a + b * b)
                if mag > radius:
                    scale = radius / mag
                    impulses[i] = a * scale
                    if sib < max_c:
                        impulses[sib] = b * scale

            else:
                # Joint target or other: unclamped
                impulses[i] = new_impulse

    return impulses.astype(np.float32)


# ------------------------------------------------------------------
# Replay result
# ------------------------------------------------------------------

@dataclass
class ReplayResult:
    """Result of replaying a PGS solve from a captured artifact.

    Attributes:
        world_idx: Which world this replay covers.
        replayed_impulses: Impulse vector from the numpy PGS solve.
        original_impulses: Impulse vector from the original capture.
        replayed_v_out: Velocity reconstructed from replayed impulses
            via ``v_out = v_hat[world_dofs] + Y^T * impulses``.
        original_v_out: The originally captured ``v_out``.
        v_hat_slice: The v_hat slice for this world.
        impulse_drift: Per-constraint absolute difference in impulses.
        velocity_drift: Per-DOF absolute difference in v_out.
        max_impulse_drift: Scalar max impulse drift.
        max_velocity_drift: Scalar max velocity drift.
        params_used: Dict of PGS parameters used for this replay.
    """
    world_idx: int = 0
    replayed_impulses: np.ndarray = field(default_factory=lambda: np.array([]))
    original_impulses: np.ndarray = field(default_factory=lambda: np.array([]))
    replayed_v_out: np.ndarray | None = None
    original_v_out: np.ndarray | None = None
    v_hat_slice: np.ndarray | None = None
    impulse_drift: np.ndarray = field(default_factory=lambda: np.array([]))
    velocity_drift: np.ndarray | None = None
    max_impulse_drift: float = 0.0
    max_velocity_drift: float = 0.0
    params_used: dict = field(default_factory=dict)

    def print_drift_report(self, file=None) -> None:
        """Print a human-readable report of replay-vs-capture drift."""
        f = file or sys.stdout
        print(f"\n{'─'*60}", file=f)
        print(f"PGS Replay Drift Report (world {self.world_idx})", file=f)
        print(f"{'─'*60}", file=f)
        print(f"  Parameters: {self.params_used}", file=f)
        print(f"  max|impulse drift|:   {self.max_impulse_drift:.6e}", file=f)
        print(f"  max|velocity drift|:  {self.max_velocity_drift:.6e}", file=f)

        if self.replayed_v_out is not None:
            print(f"  max|replayed v_out|:  {float(np.max(np.abs(self.replayed_v_out))):.4f}", file=f)
        if self.original_v_out is not None:
            print(f"  max|original v_out|:  {float(np.max(np.abs(self.original_v_out))):.4f}", file=f)

        # Show worst drifting constraints
        if len(self.impulse_drift) > 0:
            n_active = np.count_nonzero(
                np.abs(self.original_impulses) + np.abs(self.replayed_impulses) > 1e-12
            )
            print(f"  Active constraints:   {n_active}", file=f)
            top_idx = np.argsort(self.impulse_drift)[::-1][:5]
            print(f"\n  Top 5 constraint drifts:", file=f)
            print(f"    {'Row':>5}  {'orig':>12}  {'replay':>12}  {'drift':>12}", file=f)
            for idx in top_idx:
                if self.impulse_drift[idx] < 1e-12:
                    break
                print(
                    f"    {idx:>5}  {self.original_impulses[idx]:>12.6f}  "
                    f"{self.replayed_impulses[idx]:>12.6f}  "
                    f"{self.impulse_drift[idx]:>12.6e}",
                    file=f,
                )

        # Show worst drifting DOFs
        if self.velocity_drift is not None and len(self.velocity_drift) > 0:
            top_v = np.argsort(self.velocity_drift)[::-1][:5]
            print(f"\n  Top 5 velocity drifts:", file=f)
            print(f"    {'DOF':>5}  {'orig':>12}  {'replay':>12}  {'drift':>12}", file=f)
            for idx in top_v:
                if self.velocity_drift[idx] < 1e-12:
                    break
                orig = self.original_v_out[idx] if self.original_v_out is not None else float("nan")
                rep = self.replayed_v_out[idx] if self.replayed_v_out is not None else float("nan")
                print(
                    f"    {idx:>5}  {orig:>12.6f}  {rep:>12.6f}  "
                    f"{self.velocity_drift[idx]:>12.6e}",
                    file=f,
                )


# ------------------------------------------------------------------
# Spike artifact (load + analyze + replay)
# ------------------------------------------------------------------

class SpikeArtifact:
    """Loaded spike capture artifact with analysis and replay helpers.

    Attributes:
        path: Path the artifact was loaded from.
        pre_joint_q: Generalized positions before the step.
        pre_joint_qd: Generalized velocities before the step.
        post_joint_q: Generalized positions after integration.
        post_joint_qd: Generalized velocities after integration.
        v_out: Post-PGS solved velocity (before integration), if available.
        v_hat: Velocity predictor (unconstrained), if available.
        impulses: Dense PGS impulses, if available.
        constraint_count: Constraint counts per world, if available.
        world_C: Delassus matrix per world, if available.
        world_diag: Regularized diagonal per world, if available.
        world_rhs: RHS bias per world, if available.
        world_row_type: Constraint type per row, if available.
        world_row_parent: Parent row for friction, if available.
        world_row_mu: Friction coefficient per row, if available.
        Y_world: Inverse-mass Jacobian transpose, if available.
        world_dof_start: DOF start offset per world, if available.
        solver_params: Dict of solver parameter name -> value.
        meta: Dict of metadata name -> value.
    """

    def __init__(self, path: str | Path, data: dict):
        self.path = Path(path)
        self.pre_joint_q: np.ndarray = data["pre_joint_q"]
        self.pre_joint_qd: np.ndarray = data["pre_joint_qd"]
        self.post_joint_q: np.ndarray = data["post_joint_q"]
        self.post_joint_qd: np.ndarray = data["post_joint_qd"]
        self.v_out: np.ndarray | None = data.get("v_out")
        self.v_hat: np.ndarray | None = data.get("v_hat")
        self.impulses: np.ndarray | None = data.get("impulses")
        self.constraint_count: np.ndarray | None = data.get("constraint_count")

        # Constraint problem arrays (Stage 2 additions)
        self.world_C: np.ndarray | None = data.get("world_C")
        self.world_diag: np.ndarray | None = data.get("world_diag")
        self.world_rhs: np.ndarray | None = data.get("world_rhs")
        self.world_row_type: np.ndarray | None = data.get("world_row_type")
        self.world_row_parent: np.ndarray | None = data.get("world_row_parent")
        self.world_row_mu: np.ndarray | None = data.get("world_row_mu")
        self.Y_world: np.ndarray | None = data.get("Y_world")
        self.world_dof_start: np.ndarray | None = data.get("world_dof_start")

        dmc = data.get("dense_max_constraints")
        self.dense_max_constraints: int | None = int(dmc[0]) if dmc is not None else None

        # Unpack solver params
        self.solver_params: dict[str, float] = {}
        if "solver_params" in data and "solver_param_names" in data:
            for name, val in zip(data["solver_param_names"], data["solver_params"]):
                self.solver_params[str(name)] = float(val)

        # Unpack metadata
        self.meta: dict[str, float] = {}
        if "meta" in data and "meta_names" in data:
            for name, val in zip(data["meta_names"], data["meta"]):
                self.meta[str(name)] = float(val)

    @classmethod
    def load(cls, path: str | Path) -> "SpikeArtifact":
        """Load a spike artifact from an .npz file."""
        path = Path(path)
        data = dict(np.load(str(path), allow_pickle=True))
        return cls(path, data)

    # ------------------------------------------------------------------
    # Replay capability
    # ------------------------------------------------------------------

    @property
    def can_replay(self) -> bool:
        """Whether this artifact has enough data for a faithful PGS replay."""
        return (
            self.world_C is not None
            and self.world_diag is not None
            and self.world_rhs is not None
            and self.world_row_type is not None
            and self.world_row_parent is not None
            and self.world_row_mu is not None
            and self.constraint_count is not None
            and self.impulses is not None
        )

    def replay_pgs(
        self,
        world_idx: int = 0,
        iterations: int | None = None,
        omega: float | None = None,
        cfm: float | None = None,
    ) -> ReplayResult:
        """Re-run the PGS solve from captured constraint data.

        This calls the pure-numpy PGS implementation with the captured
        Delassus matrix, diagonal, RHS, and constraint metadata.  The
        result is compared against the original captured impulses to
        measure replay drift.

        Args:
            world_idx: Which world to replay (default 0).
            iterations: PGS iteration count (default: use original).
            omega: SOR factor (default: use original).
            cfm: Additional CFM to add to diagonal (default: None = no change).

        Returns:
            ReplayResult with impulse and velocity drift measurements.
        """
        if not self.can_replay:
            raise ValueError(
                "Artifact does not contain constraint-level data needed for replay. "
                "Re-capture with the Stage 2 enhanced spike_capture module."
            )

        # Resolve parameters
        orig_iters = int(self.solver_params.get("pgs_iterations", 8))
        orig_omega = float(self.solver_params.get("pgs_omega", 1.0))
        iters = iterations if iterations is not None else orig_iters
        om = omega if omega is not None else orig_omega

        m = int(self.constraint_count[world_idx])

        # Run numpy PGS
        replayed = pgs_solve_numpy(
            C=self.world_C[world_idx],
            diag=self.world_diag[world_idx],
            rhs=self.world_rhs[world_idx],
            row_type=self.world_row_type[world_idx],
            row_parent=self.world_row_parent[world_idx],
            row_mu=self.world_row_mu[world_idx],
            constraint_count=m,
            iterations=iters,
            omega=om,
            cfm_override=cfm,
        )

        original = self.impulses[world_idx]
        impulse_drift = np.abs(replayed.astype(np.float64) - original.astype(np.float64)).astype(np.float32)

        # Reconstruct v_out from replayed impulses using Y_world
        replayed_v_out = None
        original_v_out_slice = None
        velocity_drift = None
        v_hat_slice = None

        if self.Y_world is not None and self.v_hat is not None:
            Y = self.Y_world[world_idx]  # (max_constraints, max_world_dofs)
            n_dofs = Y.shape[1]

            # Determine v_hat slice for this world
            if self.world_dof_start is not None:
                dof_start = int(self.world_dof_start[world_idx])
                v_hat_slice = self.v_hat[dof_start: dof_start + n_dofs]
            elif len(self.v_hat) == n_dofs:
                v_hat_slice = self.v_hat[:n_dofs]
                dof_start = 0
            else:
                # Single world, use first n_dofs
                v_hat_slice = self.v_hat[:n_dofs]
                dof_start = 0

            # v_out = v_hat + Y^T * impulses
            # Y shape: (max_constraints, max_world_dofs)
            # impulses shape: (max_constraints,)
            replayed_v_out = v_hat_slice.copy().astype(np.float64)
            for i in range(m):
                replayed_v_out += Y[i, :].astype(np.float64) * float(replayed[i])
            replayed_v_out = replayed_v_out.astype(np.float32)

            # Also get original v_out slice
            if self.v_out is not None:
                original_v_out_slice = self.v_out[dof_start: dof_start + n_dofs]
                velocity_drift = np.abs(
                    replayed_v_out.astype(np.float64) - original_v_out_slice.astype(np.float64)
                ).astype(np.float32)

        params_used = {"iterations": iters, "omega": om, "cfm_override": cfm}

        return ReplayResult(
            world_idx=world_idx,
            replayed_impulses=replayed,
            original_impulses=original,
            replayed_v_out=replayed_v_out,
            original_v_out=original_v_out_slice,
            v_hat_slice=v_hat_slice,
            impulse_drift=impulse_drift,
            velocity_drift=velocity_drift,
            max_impulse_drift=float(np.max(impulse_drift[:m])) if m > 0 else 0.0,
            max_velocity_drift=float(np.max(velocity_drift)) if velocity_drift is not None else 0.0,
            params_used=params_used,
        )

    def replay_parameter_sweep(
        self,
        world_idx: int = 0,
        omega_values: list[float] | None = None,
        cfm_values: list[float] | None = None,
        iteration_values: list[int] | None = None,
    ) -> list[ReplayResult]:
        """Run PGS replay with multiple parameter combinations.

        Returns a list of ReplayResult objects, one per combination.
        Useful for finding which parameters reduce spikes most effectively.
        """
        if omega_values is None:
            omega_values = [float(self.solver_params.get("pgs_omega", 1.0))]
        if cfm_values is None:
            cfm_values = [None]
        if iteration_values is None:
            iteration_values = [int(self.solver_params.get("pgs_iterations", 8))]

        results = []
        for om in omega_values:
            for cfm in cfm_values:
                for iters in iteration_values:
                    r = self.replay_pgs(
                        world_idx=world_idx,
                        iterations=iters,
                        omega=om,
                        cfm=cfm,
                    )
                    results.append(r)
        return results

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    @property
    def max_abs_post_qd(self) -> float:
        return float(np.max(np.abs(self.post_joint_qd)))

    @property
    def max_abs_pre_qd(self) -> float:
        return float(np.max(np.abs(self.pre_joint_qd)))

    @property
    def num_dofs(self) -> int:
        return len(self.pre_joint_qd)

    @property
    def velocity_amplification(self) -> float:
        """Ratio of max post-solve velocity to max pre-solve velocity."""
        pre = self.max_abs_pre_qd
        if pre < 1e-12:
            return float("inf")
        return self.max_abs_post_qd / pre

    def top_spike_dofs(self, n: int = 10) -> list[tuple[int, float, float]]:
        """Return the top-n DOFs by absolute post-solve velocity.

        Returns:
            List of (dof_index, post_qd, pre_qd) tuples sorted descending by |post_qd|.
        """
        abs_post = np.abs(self.post_joint_qd)
        top_idx = np.argsort(abs_post)[::-1][:n]
        return [
            (int(i), float(self.post_joint_qd[i]), float(self.pre_joint_qd[i]))
            for i in top_idx
        ]

    def velocity_delta(self) -> np.ndarray:
        """Per-DOF velocity change: post_qd - pre_qd."""
        return self.post_joint_qd - self.pre_joint_qd

    def classify_spike(self) -> str:
        """Heuristic classification of the spike type.

        Returns one of:
            "contact_impulse"  - large impulses with many active constraints
            "joint_limit"      - spike concentrated on few DOFs near limits
            "unconstrained"    - v_hat already large (external force / gravity)
            "pgs_divergence"   - v_out much larger than v_hat
            "unknown"          - does not match any heuristic
        """
        delta = self.velocity_delta()
        top = self.top_spike_dofs(5)

        # Check if v_hat already shows the spike (unconstrained instability)
        if self.v_hat is not None:
            max_v_hat = float(np.max(np.abs(self.v_hat)))
            if max_v_hat > self.meta.get("max_abs_qd", 50.0) * 0.8:
                return "unconstrained"

            # Check if PGS amplified beyond v_hat
            if self.v_out is not None:
                max_v_out = float(np.max(np.abs(self.v_out)))
                if max_v_out > max_v_hat * 5.0 and max_v_out > 50.0:
                    return "pgs_divergence"

        # Check constraint activity
        if self.constraint_count is not None:
            total_constraints = int(np.sum(self.constraint_count))
            if total_constraints > 10 and self.impulses is not None:
                max_impulse = float(np.max(np.abs(self.impulses)))
                if max_impulse > 100.0:
                    return "contact_impulse"

        # Check concentration on few DOFs (joint limit pattern)
        if len(top) >= 3:
            top_delta = abs(delta[top[0][0]])
            third_delta = abs(delta[top[2][0]])
            if top_delta > 10.0 * third_delta and top_delta > 10.0:
                return "joint_limit"

        return "unknown"

    # ------------------------------------------------------------------
    # Printing
    # ------------------------------------------------------------------

    def print_summary(self, file=None) -> None:
        """Print a human-readable summary of the spike."""
        f = file or sys.stdout
        print(f"\n{'='*70}", file=f)
        print(f"Spike Artifact: {self.path.name}", file=f)
        print(f"{'='*70}", file=f)
        print(f"  Step index:         {self.meta.get('step_index', '?'):.0f}", file=f)
        print(f"  Capture index:      {self.meta.get('capture_index', '?'):.0f}", file=f)
        print(f"  DOFs:               {self.num_dofs}", file=f)
        print(f"  max|pre_qd|:        {self.max_abs_pre_qd:.4f}", file=f)
        print(f"  max|post_qd|:       {self.max_abs_post_qd:.4f}", file=f)
        print(f"  Amplification:      {self.velocity_amplification:.2f}x", file=f)
        print(f"  Classification:     {self.classify_spike()}", file=f)
        print(f"  Replay capable:     {self.can_replay}", file=f)
        print(f"\n  Solver params:", file=f)
        for k, v in self.solver_params.items():
            print(f"    {k}: {v}", file=f)

        if self.v_hat is not None:
            print(f"\n  max|v_hat|:         {float(np.max(np.abs(self.v_hat))):.4f}", file=f)
        if self.v_out is not None:
            print(f"  max|v_out|:         {float(np.max(np.abs(self.v_out))):.4f}", file=f)
        if self.impulses is not None:
            print(f"  max|impulses|:      {float(np.max(np.abs(self.impulses))):.4f}", file=f)
        if self.constraint_count is not None:
            print(f"  Total constraints:  {int(np.sum(self.constraint_count))}", file=f)

    def print_velocity_profile(self, top_n: int = 10, file=None) -> None:
        """Print per-DOF velocity details for the worst offenders."""
        f = file or sys.stdout
        print(f"\nTop {top_n} DOFs by |post_qd|:", file=f)
        print(f"  {'DOF':>5}  {'pre_qd':>12}  {'post_qd':>12}  {'delta':>12}", file=f)
        print(f"  {'---':>5}  {'---':>12}  {'---':>12}  {'---':>12}", file=f)
        for dof, post, pre in self.top_spike_dofs(top_n):
            delta = post - pre
            print(f"  {dof:>5}  {pre:>12.4f}  {post:>12.4f}  {delta:>12.4f}", file=f)

        if self.v_hat is not None:
            print(f"\nTop {top_n} DOFs by |v_hat| (unconstrained predictor):", file=f)
            abs_vhat = np.abs(self.v_hat)
            top_vh = np.argsort(abs_vhat)[::-1][:top_n]
            print(f"  {'DOF':>5}  {'v_hat':>12}  {'v_out':>12}  {'post_qd':>12}", file=f)
            print(f"  {'---':>5}  {'---':>12}  {'---':>12}  {'---':>12}", file=f)
            for i in top_vh:
                vh = float(self.v_hat[i])
                vo = float(self.v_out[i]) if self.v_out is not None else float("nan")
                pq = float(self.post_joint_qd[i])
                print(f"  {int(i):>5}  {vh:>12.4f}  {vo:>12.4f}  {pq:>12.4f}", file=f)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze and replay FeatherPGS velocity spike artifacts"
    )
    parser.add_argument("artifact", type=str, help="Path to .npz spike artifact")
    parser.add_argument("--top", type=int, default=10, help="Number of top DOFs to show")
    parser.add_argument("--replay", action="store_true", help="Run PGS replay and show drift")
    parser.add_argument("--omega", type=float, default=None, help="Override SOR omega for replay")
    parser.add_argument("--cfm", type=float, default=None, help="Additional CFM for replay")
    parser.add_argument("--iterations", type=int, default=None, help="Override PGS iterations for replay")
    parser.add_argument(
        "--sweep", action="store_true",
        help="Run a parameter sweep with predefined omega/cfm/iteration values"
    )
    args = parser.parse_args()

    artifact = SpikeArtifact.load(args.artifact)
    artifact.print_summary()
    artifact.print_velocity_profile(top_n=args.top)

    if args.replay and artifact.can_replay:
        result = artifact.replay_pgs(
            omega=args.omega,
            cfm=args.cfm,
            iterations=args.iterations,
        )
        result.print_drift_report()
    elif args.replay:
        print(
            "\n[WARNING] Artifact lacks constraint-level data for replay. "
            "Re-capture with Stage 2 enhanced spike_capture.",
            file=sys.stderr,
        )

    if args.sweep and artifact.can_replay:
        print(f"\n{'='*60}")
        print("Parameter Sweep Results")
        print(f"{'='*60}")
        results = artifact.replay_parameter_sweep(
            omega_values=[1.0, 0.9, 0.8, 0.7],
            cfm_values=[None, 1e-5, 1e-4, 1e-3],
            iteration_values=[8, 16, 32],
        )
        print(
            f"  {'omega':>6}  {'cfm':>10}  {'iters':>6}  "
            f"{'max|imp_drift|':>14}  {'max|vel_drift|':>14}  "
            f"{'max|replay_v|':>14}"
        )
        for r in results:
            cfm_str = f"{r.params_used['cfm_override']:.0e}" if r.params_used["cfm_override"] else "none"
            max_rv = float(np.max(np.abs(r.replayed_v_out))) if r.replayed_v_out is not None else float("nan")
            print(
                f"  {r.params_used['omega']:>6.2f}  {cfm_str:>10}  "
                f"{r.params_used['iterations']:>6d}  "
                f"{r.max_impulse_drift:>14.6e}  {r.max_velocity_drift:>14.6e}  "
                f"{max_rv:>14.4f}"
            )


if __name__ == "__main__":
    main()
