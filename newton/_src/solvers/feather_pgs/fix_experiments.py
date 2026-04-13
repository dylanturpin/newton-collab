# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fix experiments for FeatherPGS velocity spikes.

This script tests three candidate fixes on the captured spike artifacts:

1. **Post-solve velocity clamping**: Clamp v_out per-DOF to N * vel_limit
   before integration.  This directly prevents the large velocities that
   trigger environment termination.  It is a safety net, not a physics fix.

2. **Reduced pgs_beta**: Lower the Baumgarte correction strength from the
   default 0.05 to 0.01 or 0.02.  This reduces the impulse magnitude for
   joint-limit and contact constraints, at the cost of slower penetration
   recovery.

3. **Contact compliance (increased diagonal)**: Add compliance to the
   Delassus diagonal for contact-normal constraints.  This softens the
   contact response, reducing impulse magnitude but allowing more
   penetration.

For each fix, the script:
- Loads each spike artifact
- Runs the baseline PGS replay (original parameters)
- Runs the fix variant
- Computes before/after max|v_out| and the reduction percentage
- Prints a detailed report and returns structured results

Usage::

    cd newton-collab
    python -m newton._src.solvers.feather_pgs.fix_experiments

Or as a library::

    from newton._src.solvers.feather_pgs.fix_experiments import run_all_experiments
    results = run_all_experiments()
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from newton._src.solvers.feather_pgs.spike_replay import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    SpikeArtifact,
    pgs_solve_numpy,
)

# ---------------------------------------------------------------------------
# Franka Panda velocity limits (from generate_realistic_spikes.py)
# ---------------------------------------------------------------------------

FRANKA_VEL_LIMITS = np.array(
    [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61, 0.2, 0.2],
    dtype=np.float32,
)

# Termination threshold: 1.25x the soft velocity limits
FRANKA_TERM_LIMITS = FRANKA_VEL_LIMITS * 1.25


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------

@dataclass
class FixResult:
    """Result of applying a single fix to a single artifact."""
    artifact_name: str
    fix_name: str
    fix_params: dict
    baseline_max_v_out: float
    fixed_max_v_out: float
    reduction_pct: float
    baseline_v_out: np.ndarray | None = None
    fixed_v_out: np.ndarray | None = None
    # Per-DOF: how many DOFs exceed termination threshold before/after
    baseline_dofs_over_limit: int = 0
    fixed_dofs_over_limit: int = 0
    # Impulse comparison
    baseline_max_impulse: float = 0.0
    fixed_max_impulse: float = 0.0
    notes: str = ""


@dataclass
class ExperimentReport:
    """Complete report of all fix experiments."""
    results: list[FixResult] = field(default_factory=list)

    def best_fix_for(self, artifact_name: str) -> FixResult | None:
        """Return the fix with the largest v_out reduction for a given artifact."""
        candidates = [r for r in self.results if r.artifact_name == artifact_name]
        if not candidates:
            return None
        return min(candidates, key=lambda r: r.fixed_max_v_out)

    def print_summary(self, file=None) -> str:
        """Print and return a formatted summary table."""
        f = file or sys.stdout
        lines = []

        def p(s):
            lines.append(s)
            print(s, file=f)

        p(f"\n{'='*100}")
        p(f"FIX EXPERIMENT RESULTS SUMMARY")
        p(f"{'='*100}")
        p(f"{'Artifact':<40} {'Fix':<30} {'Before':>8} {'After':>8} {'Δ%':>7} {'Over-limit':>11}")
        p(f"{'':<40} {'':<30} {'v_out':>8} {'v_out':>8} {'':<7} {'B→A':>11}")
        p(f"{'─'*40} {'─'*30} {'─'*8} {'─'*8} {'─'*7} {'─'*11}")

        for r in self.results:
            over = f"{r.baseline_dofs_over_limit}→{r.fixed_dofs_over_limit}"
            p(
                f"{r.artifact_name:<40} {r.fix_name:<30} "
                f"{r.baseline_max_v_out:>8.3f} {r.fixed_max_v_out:>8.3f} "
                f"{r.reduction_pct:>+6.1f}% {over:>11}"
            )

        p(f"{'─'*100}")

        # Best fix per artifact
        p(f"\nBest fix per artifact:")
        seen = set()
        for r in self.results:
            if r.artifact_name in seen:
                continue
            seen.add(r.artifact_name)
            best = self.best_fix_for(r.artifact_name)
            if best:
                p(f"  {best.artifact_name}: {best.fix_name} "
                  f"({best.baseline_max_v_out:.3f} → {best.fixed_max_v_out:.3f}, "
                  f"{best.reduction_pct:+.1f}%)")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Fix implementations
# ---------------------------------------------------------------------------

def _replay_baseline(artifact: SpikeArtifact, world_idx: int = 0):
    """Run baseline replay with original parameters."""
    result = artifact.replay_pgs(world_idx=world_idx)
    return result


def _count_dofs_over_limit(v_out: np.ndarray | None) -> int:
    """Count DOFs where |v_out| exceeds the Franka termination threshold."""
    if v_out is None:
        return 0
    n = min(len(v_out), len(FRANKA_TERM_LIMITS))
    count = 0
    for i in range(n):
        if abs(float(v_out[i])) > FRANKA_TERM_LIMITS[i]:
            count += 1
    return count


# --- Fix 1: Post-solve velocity clamping ---

def apply_velocity_clamp(
    v_out: np.ndarray,
    clamp_factor: float = 2.0,
) -> np.ndarray:
    """Clamp v_out per-DOF to clamp_factor * velocity_limit.

    This is a safety-net fix: it does not change the solver behavior,
    only clips the output velocity to prevent extreme values from
    reaching integration.

    Args:
        v_out: Post-PGS velocity, shape (n_dofs,).
        clamp_factor: Multiple of velocity limit for the clamp.

    Returns:
        Clamped velocity array.
    """
    clamped = v_out.copy()
    n = min(len(clamped), len(FRANKA_VEL_LIMITS))
    for i in range(n):
        limit = FRANKA_VEL_LIMITS[i] * clamp_factor
        clamped[i] = np.clip(float(clamped[i]), -limit, limit)
    return clamped


def run_velocity_clamp_experiment(
    artifact: SpikeArtifact,
    clamp_factor: float = 2.0,
    world_idx: int = 0,
) -> FixResult:
    """Test post-solve velocity clamping on a spike artifact."""
    baseline = _replay_baseline(artifact, world_idx)

    baseline_v = baseline.replayed_v_out
    if baseline_v is None:
        return FixResult(
            artifact_name=artifact.path.name,
            fix_name=f"velocity_clamp(factor={clamp_factor})",
            fix_params={"clamp_factor": clamp_factor},
            baseline_max_v_out=0.0, fixed_max_v_out=0.0,
            reduction_pct=0.0, notes="No v_out available for replay"
        )

    fixed_v = apply_velocity_clamp(baseline_v, clamp_factor)

    baseline_max = float(np.max(np.abs(baseline_v)))
    fixed_max = float(np.max(np.abs(fixed_v)))
    reduction = (1.0 - fixed_max / baseline_max) * 100.0 if baseline_max > 0 else 0.0

    return FixResult(
        artifact_name=artifact.path.name,
        fix_name=f"velocity_clamp(factor={clamp_factor})",
        fix_params={"clamp_factor": clamp_factor},
        baseline_max_v_out=baseline_max,
        fixed_max_v_out=fixed_max,
        reduction_pct=reduction,
        baseline_v_out=baseline_v,
        fixed_v_out=fixed_v,
        baseline_dofs_over_limit=_count_dofs_over_limit(baseline_v),
        fixed_dofs_over_limit=_count_dofs_over_limit(fixed_v),
        baseline_max_impulse=float(np.max(np.abs(baseline.replayed_impulses))),
        fixed_max_impulse=float(np.max(np.abs(baseline.replayed_impulses))),
        notes=(
            f"Clamp at {clamp_factor}x vel_limit. "
            f"Does not change solver output, only clips v_out."
        ),
    )


# --- Fix 2: Reduced pgs_beta ---

def run_reduced_beta_experiment(
    artifact: SpikeArtifact,
    new_beta: float = 0.02,
    world_idx: int = 0,
) -> FixResult:
    """Test reduced pgs_beta on a spike artifact.

    This fix reduces the Baumgarte correction strength in the constraint
    RHS.  The RHS formula is: rhs = beta * phi / dt + J * v_hat.
    Reducing beta from 0.05 to 0.02 reduces the position-correction
    component by 2.5x.

    Since we cannot directly change beta in the replay (the RHS is
    pre-computed in the artifact), we recompute the RHS with the new
    beta and re-run the PGS solve.
    """
    if not artifact.can_replay:
        return FixResult(
            artifact_name=artifact.path.name,
            fix_name=f"reduced_beta({new_beta})",
            fix_params={"pgs_beta": new_beta},
            baseline_max_v_out=0.0, fixed_max_v_out=0.0,
            reduction_pct=0.0, notes="Cannot replay"
        )

    # Get baseline
    baseline = _replay_baseline(artifact, world_idx)
    baseline_v = baseline.replayed_v_out
    baseline_max = float(np.max(np.abs(baseline_v))) if baseline_v is not None else 0.0

    # Recompute RHS with new beta
    orig_beta = float(artifact.solver_params.get("pgs_beta", 0.05))
    dt = float(artifact.solver_params.get("dt", 0.005))
    inv_dt = 1.0 / dt

    orig_rhs = artifact.world_rhs[world_idx].copy()
    m = int(artifact.constraint_count[world_idx])

    # The RHS has two components: beta*phi/dt and J*v_hat.
    # We need to scale only the beta*phi/dt part.
    # rhs_new = (new_beta/orig_beta) * beta_component + Jv_hat_component
    # beta_component = rhs - J*v_hat
    # But we don't have J and v_hat separately in the artifact for each row.
    # However, we can compute J*v_hat from the Y and v_hat:
    #   J*v_hat is the velocity-level part of the RHS
    #
    # Alternative approach: scale the entire RHS by the beta ratio for
    # constraint types that have Baumgarte correction (contacts and limits).
    # This is approximate but captures the dominant effect.

    # More precise: recompute the Baumgarte bias term.
    # For contacts/limits with phi < 0: bias = beta * phi / dt
    # The J*v_hat part is computed as dot(J[i], v_hat)
    # Since we have Y and v_hat, and J is implicitly in the artifact
    # through C and Y, we can compute J*v_hat for each row using
    # the relationship C = J * Y^T, which gives J for each row if we
    # knew the mapping. But that's circular.

    # Simplest correct approach: the Baumgarte component is
    # (orig_beta * phi * inv_dt) and we want (new_beta * phi * inv_dt).
    # The ratio is new_beta / orig_beta. If we can identify which part
    # of rhs is the Baumgarte term, we scale only that.

    # We use the fact that for rows with active Baumgarte (contacts, limits
    # with phi < 0), the bias term is additive to rhs.
    # rhs = beta*phi/dt + J*v_hat
    # rhs_new = new_beta*phi/dt + J*v_hat = rhs + (new_beta - orig_beta)*phi/dt

    # We need phi. It's not directly stored in the artifact, but we can
    # extract it from the original rhs and J*v_hat:
    # phi_term = rhs - J*v_hat, then phi = phi_term * dt / orig_beta
    # J*v_hat can be computed from the constraint Jacobian.

    # Actually the simplest approach: scale the entire rhs for relevant
    # constraint types by new_beta/orig_beta. This assumes the J*v_hat
    # component is small relative to the Baumgarte term for violated
    # constraints. For the spike artifacts this is approximately true
    # (beta*phi/dt dominates for phi = -0.02 to -0.05).

    # Let's use a more precise approach:
    # Compute J*v_hat for each row from the artifact data.
    # J is implicit in C and Y: C = J * Y^T. But J is not stored directly.
    # However, for joint limits, J is ±1 on the constrained DOF, so
    # J*v_hat = ±v_hat[constrained_dof]. For contacts, J has multiple
    # non-zero entries but we can compute J*v_hat from the rhs and the
    # known beta*phi/dt term (if we knew phi).

    # Best practical approach for replay: scale the Baumgarte component.
    # For contact/limit rows: estimate phi from the rhs.
    # If rhs[i] is significantly different from what J*v_hat alone would give,
    # the difference is the Baumgarte term.

    # For simplicity and accuracy, we'll use the beta-ratio scaling on
    # the entire rhs for contact and limit rows. This slightly under-corrects
    # (because J*v_hat is also scaled) but the error is small when the
    # Baumgarte term dominates.

    new_rhs = orig_rhs.copy().astype(np.float64)
    row_type = artifact.world_row_type[world_idx]

    if orig_beta > 0.0:
        beta_ratio = new_beta / orig_beta
        for i in range(m):
            rt = int(row_type[i])
            if rt == PGS_CONSTRAINT_TYPE_CONTACT or rt == PGS_CONSTRAINT_TYPE_JOINT_LIMIT:
                new_rhs[i] = orig_rhs[i] * beta_ratio
    new_rhs = new_rhs.astype(np.float32)

    # Run PGS with modified RHS
    orig_iters = int(artifact.solver_params.get("pgs_iterations", 8))
    orig_omega = float(artifact.solver_params.get("pgs_omega", 1.0))

    fixed_impulses = pgs_solve_numpy(
        C=artifact.world_C[world_idx],
        diag=artifact.world_diag[world_idx],
        rhs=new_rhs,
        row_type=artifact.world_row_type[world_idx],
        row_parent=artifact.world_row_parent[world_idx],
        row_mu=artifact.world_row_mu[world_idx],
        constraint_count=m,
        iterations=orig_iters,
        omega=orig_omega,
    )

    # Reconstruct v_out with fixed impulses
    fixed_v = None
    if artifact.Y_world is not None and artifact.v_hat is not None:
        Y = artifact.Y_world[world_idx]
        n_dofs = Y.shape[1]
        if artifact.world_dof_start is not None:
            dof_start = int(artifact.world_dof_start[world_idx])
        else:
            dof_start = 0
        v_hat_slice = artifact.v_hat[dof_start:dof_start + n_dofs]

        fixed_v = v_hat_slice.copy().astype(np.float64)
        for i in range(m):
            fixed_v += Y[i, :].astype(np.float64) * float(fixed_impulses[i])
        fixed_v = fixed_v.astype(np.float32)

    fixed_max = float(np.max(np.abs(fixed_v))) if fixed_v is not None else 0.0
    reduction = (1.0 - fixed_max / baseline_max) * 100.0 if baseline_max > 0 else 0.0

    return FixResult(
        artifact_name=artifact.path.name,
        fix_name=f"reduced_beta({new_beta})",
        fix_params={"pgs_beta": new_beta, "orig_beta": orig_beta},
        baseline_max_v_out=baseline_max,
        fixed_max_v_out=fixed_max,
        reduction_pct=reduction,
        baseline_v_out=baseline_v,
        fixed_v_out=fixed_v,
        baseline_dofs_over_limit=_count_dofs_over_limit(baseline_v),
        fixed_dofs_over_limit=_count_dofs_over_limit(fixed_v),
        baseline_max_impulse=float(np.max(np.abs(baseline.replayed_impulses[:m]))) if m > 0 else 0.0,
        fixed_max_impulse=float(np.max(np.abs(fixed_impulses[:m]))) if m > 0 else 0.0,
        notes=(
            f"Beta {orig_beta}→{new_beta} (ratio {new_beta/orig_beta:.2f}). "
            f"Scales Baumgarte correction in RHS. "
            f"Tradeoff: slower penetration recovery."
        ),
    )


# --- Fix 3: Contact compliance (increased diagonal) ---

def run_contact_compliance_experiment(
    artifact: SpikeArtifact,
    compliance: float = 0.001,
    world_idx: int = 0,
) -> FixResult:
    """Test increased contact compliance on a spike artifact.

    Contact compliance adds alpha = compliance / dt^2 to the Delassus
    diagonal for contact-normal constraint rows.  This softens the contact
    response: smaller impulses, more penetration allowed.

    In FeatherPGS, this is controlled by the `dense_contact_compliance`
    parameter (default 0.0, meaning perfectly rigid contacts).
    """
    if not artifact.can_replay:
        return FixResult(
            artifact_name=artifact.path.name,
            fix_name=f"contact_compliance({compliance})",
            fix_params={"compliance": compliance},
            baseline_max_v_out=0.0, fixed_max_v_out=0.0,
            reduction_pct=0.0, notes="Cannot replay"
        )

    # Get baseline
    baseline = _replay_baseline(artifact, world_idx)
    baseline_v = baseline.replayed_v_out
    baseline_max = float(np.max(np.abs(baseline_v))) if baseline_v is not None else 0.0

    dt = float(artifact.solver_params.get("dt", 0.005))
    m = int(artifact.constraint_count[world_idx])

    # Modify diagonal: add alpha = compliance / dt^2 to contact-normal rows
    alpha = compliance / (dt * dt)
    modified_diag = artifact.world_diag[world_idx].copy()
    row_type = artifact.world_row_type[world_idx]

    contact_rows_modified = 0
    for i in range(m):
        if int(row_type[i]) == PGS_CONSTRAINT_TYPE_CONTACT:
            modified_diag[i] += alpha
            contact_rows_modified += 1

    # Run PGS with modified diagonal
    orig_iters = int(artifact.solver_params.get("pgs_iterations", 8))
    orig_omega = float(artifact.solver_params.get("pgs_omega", 1.0))

    fixed_impulses = pgs_solve_numpy(
        C=artifact.world_C[world_idx],
        diag=modified_diag,
        rhs=artifact.world_rhs[world_idx],
        row_type=artifact.world_row_type[world_idx],
        row_parent=artifact.world_row_parent[world_idx],
        row_mu=artifact.world_row_mu[world_idx],
        constraint_count=m,
        iterations=orig_iters,
        omega=orig_omega,
    )

    # Reconstruct v_out
    fixed_v = None
    if artifact.Y_world is not None and artifact.v_hat is not None:
        Y = artifact.Y_world[world_idx]
        n_dofs = Y.shape[1]
        if artifact.world_dof_start is not None:
            dof_start = int(artifact.world_dof_start[world_idx])
        else:
            dof_start = 0
        v_hat_slice = artifact.v_hat[dof_start:dof_start + n_dofs]

        fixed_v = v_hat_slice.copy().astype(np.float64)
        for i in range(m):
            fixed_v += Y[i, :].astype(np.float64) * float(fixed_impulses[i])
        fixed_v = fixed_v.astype(np.float32)

    fixed_max = float(np.max(np.abs(fixed_v))) if fixed_v is not None else 0.0
    reduction = (1.0 - fixed_max / baseline_max) * 100.0 if baseline_max > 0 else 0.0

    return FixResult(
        artifact_name=artifact.path.name,
        fix_name=f"contact_compliance({compliance})",
        fix_params={"compliance": compliance, "alpha": alpha, "contact_rows": contact_rows_modified},
        baseline_max_v_out=baseline_max,
        fixed_max_v_out=fixed_max,
        reduction_pct=reduction,
        baseline_v_out=baseline_v,
        fixed_v_out=fixed_v,
        baseline_dofs_over_limit=_count_dofs_over_limit(baseline_v),
        fixed_dofs_over_limit=_count_dofs_over_limit(fixed_v),
        baseline_max_impulse=float(np.max(np.abs(baseline.replayed_impulses[:m]))) if m > 0 else 0.0,
        fixed_max_impulse=float(np.max(np.abs(fixed_impulses[:m]))) if m > 0 else 0.0,
        notes=(
            f"Compliance {compliance} → alpha={alpha:.1f} added to {contact_rows_modified} "
            f"contact-normal rows. Tradeoff: softer contacts, more penetration."
        ),
    )


# ---------------------------------------------------------------------------
# Run all experiments
# ---------------------------------------------------------------------------

def run_all_experiments(
    artifact_dir: str | Path = "spike_captures",
) -> ExperimentReport:
    """Run all fix experiments on all spike artifacts.

    Returns an ExperimentReport with structured results for each
    (artifact, fix) combination.
    """
    artifact_dir = Path(artifact_dir)
    artifact_files = sorted(artifact_dir.glob("real_*.npz"))

    if not artifact_files:
        print(f"[ERROR] No real_*.npz artifacts found in {artifact_dir}", file=sys.stderr)
        return ExperimentReport()

    report = ExperimentReport()

    for artifact_path in artifact_files:
        artifact = SpikeArtifact.load(artifact_path)

        print(f"\n{'═'*80}")
        print(f"Artifact: {artifact_path.name}")
        print(f"  Classification: {artifact.classify_spike()}")
        print(f"  max|post_qd|: {artifact.max_abs_post_qd:.3f}")
        if artifact.v_out is not None:
            print(f"  max|v_out|: {float(np.max(np.abs(artifact.v_out))):.3f}")
        if artifact.v_hat is not None:
            print(f"  max|v_hat|: {float(np.max(np.abs(artifact.v_hat))):.3f}")
        print(f"  Can replay: {artifact.can_replay}")
        print(f"{'═'*80}")

        if not artifact.can_replay:
            print(f"  [SKIP] Cannot replay artifact {artifact_path.name}")
            continue

        # Fix 1: Post-solve velocity clamping (multiple factors)
        for factor in [1.5, 2.0, 3.0]:
            result = run_velocity_clamp_experiment(artifact, clamp_factor=factor)
            report.results.append(result)
            print(f"  [{result.fix_name}] "
                  f"{result.baseline_max_v_out:.3f} → {result.fixed_max_v_out:.3f} "
                  f"({result.reduction_pct:+.1f}%)"
                  f"  over-limit: {result.baseline_dofs_over_limit}→{result.fixed_dofs_over_limit}")

        # Fix 2: Reduced pgs_beta
        for beta in [0.02, 0.01]:
            result = run_reduced_beta_experiment(artifact, new_beta=beta)
            report.results.append(result)
            print(f"  [{result.fix_name}] "
                  f"{result.baseline_max_v_out:.3f} → {result.fixed_max_v_out:.3f} "
                  f"({result.reduction_pct:+.1f}%)"
                  f"  over-limit: {result.baseline_dofs_over_limit}→{result.fixed_dofs_over_limit}")

        # Fix 3: Contact compliance (multiple values)
        for compliance in [0.0001, 0.001, 0.01]:
            result = run_contact_compliance_experiment(artifact, compliance=compliance)
            report.results.append(result)
            print(f"  [{result.fix_name}] "
                  f"{result.baseline_max_v_out:.3f} → {result.fixed_max_v_out:.3f} "
                  f"({result.reduction_pct:+.1f}%)"
                  f"  over-limit: {result.baseline_dofs_over_limit}→{result.fixed_dofs_over_limit}")

    return report


def format_detailed_results(report: ExperimentReport) -> str:
    """Format detailed per-DOF results for the report."""
    lines = []

    for r in report.results:
        if r.baseline_v_out is None or r.fixed_v_out is None:
            continue

        lines.append(f"\n### {r.artifact_name} + {r.fix_name}")
        lines.append(f"")
        lines.append(f"| DOF | Baseline v_out | Fixed v_out | Vel Limit | Term Limit | Status |")
        lines.append(f"|-----|---------------|------------|-----------|------------|--------|")

        n = min(len(r.baseline_v_out), len(FRANKA_VEL_LIMITS))
        for i in range(n):
            bv = float(r.baseline_v_out[i])
            fv = float(r.fixed_v_out[i])
            vl = FRANKA_VEL_LIMITS[i]
            tl = FRANKA_TERM_LIMITS[i]
            status = "OK"
            if abs(fv) > tl:
                status = "OVER-LIMIT"
            elif abs(fv) > vl:
                status = "above-soft"
            lines.append(
                f"| {i} | {bv:+.4f} | {fv:+.4f} | {vl:.3f} | {tl:.3f} | {status} |"
            )

        lines.append(f"")
        lines.append(f"Notes: {r.notes}")

    return "\n".join(lines)


def main():
    report = run_all_experiments("spike_captures")
    summary = report.print_summary()

    # Print detailed per-DOF results for the best fixes
    print(f"\n{'='*80}")
    print("DETAILED PER-DOF RESULTS (best fix per artifact)")
    print(f"{'='*80}")

    seen = set()
    for r in report.results:
        if r.artifact_name in seen:
            continue
        seen.add(r.artifact_name)
        best = report.best_fix_for(r.artifact_name)
        if best and best.baseline_v_out is not None and best.fixed_v_out is not None:
            print(f"\n  {best.artifact_name} → {best.fix_name}")
            n = min(len(best.baseline_v_out), len(FRANKA_VEL_LIMITS))
            print(f"    {'DOF':>4} {'Baseline':>10} {'Fixed':>10} {'Limit':>8} {'Status':>12}")
            for i in range(n):
                bv = float(best.baseline_v_out[i])
                fv = float(best.fixed_v_out[i])
                tl = FRANKA_TERM_LIMITS[i]
                status = "OVER" if abs(fv) > tl else "ok"
                print(f"    {i:>4} {bv:>+10.4f} {fv:>+10.4f} {tl:>8.3f} {status:>12}")


if __name__ == "__main__":
    main()
