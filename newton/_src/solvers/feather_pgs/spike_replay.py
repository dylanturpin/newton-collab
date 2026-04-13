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

Usage as a CLI tool::

    python -m newton._src.solvers.feather_pgs.spike_replay path/to/spike_0000.npz

Or as a library::

    from newton._src.solvers.feather_pgs.spike_replay import SpikeArtifact
    artifact = SpikeArtifact.load("spike_0000.npz")
    artifact.print_summary()
    artifact.print_velocity_profile()
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


class SpikeArtifact:
    """Loaded spike capture artifact with analysis helpers.

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
        solver_params: Dict of solver parameter name -> value.
        meta: Dict of metadata name -> value (step_index, timestamp, etc.).
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
            "contact_impulse"  – large impulses with many active constraints
            "joint_limit"      – spike concentrated on few DOFs near limits
            "unconstrained"    – v_hat already large (external force / gravity)
            "pgs_divergence"   – v_out much larger than v_hat
            "unknown"          – does not match any heuristic
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
    parser = argparse.ArgumentParser(description="Analyze a FeatherPGS velocity spike artifact")
    parser.add_argument("artifact", type=str, help="Path to .npz spike artifact")
    parser.add_argument("--top", type=int, default=10, help="Number of top DOFs to show")
    args = parser.parse_args()

    artifact = SpikeArtifact.load(args.artifact)
    artifact.print_summary()
    artifact.print_velocity_profile(top_n=args.top)


if __name__ == "__main__":
    main()
