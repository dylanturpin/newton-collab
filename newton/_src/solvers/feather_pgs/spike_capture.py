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

"""Opt-in velocity spike capture for FeatherPGS debugging.

This module provides a lightweight capture system that detects and records
velocity spikes during FeatherPGS solver steps.  It is dormant by default
and must be explicitly enabled via environment variable or API call.

A "velocity spike" is defined as a step where the maximum absolute
generalized velocity (joint_qd) after integration exceeds a configurable
threshold.  When a spike is detected the module serializes a snapshot of
the solver inputs and outputs to a NumPy .npz file for later replay and
analysis.

Enable capture by setting the environment variable::

    FEATHERPGS_SPIKE_CAPTURE=1

Or programmatically::

    solver.spike_capture.enable()

Optional environment variables::

    FEATHERPGS_SPIKE_THRESHOLD    – max |qd| to consider a spike (default 50.0)
    FEATHERPGS_SPIKE_DIR          – directory for .npz artifacts (default ./spike_captures)
    FEATHERPGS_SPIKE_MAX_CAPTURES – stop capturing after this many (default 100)
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np


class SpikeCaptureConfig:
    """Configuration for velocity spike capture, read from environment."""

    def __init__(self):
        self.enabled: bool = os.environ.get("FEATHERPGS_SPIKE_CAPTURE", "0") == "1"
        self.threshold: float = float(os.environ.get("FEATHERPGS_SPIKE_THRESHOLD", "50.0"))
        self.output_dir: Path = Path(os.environ.get("FEATHERPGS_SPIKE_DIR", "./spike_captures"))
        self.max_captures: int = int(os.environ.get("FEATHERPGS_SPIKE_MAX_CAPTURES", "100"))

    def __repr__(self) -> str:
        return (
            f"SpikeCaptureConfig(enabled={self.enabled}, threshold={self.threshold}, "
            f"output_dir={self.output_dir!r}, max_captures={self.max_captures})"
        )


class SpikeCapture:
    """Velocity spike detector and artifact writer for FeatherPGS.

    This object is meant to be attached to a SolverFeatherPGS instance.
    It exposes two hooks that the solver calls at well-defined points:

    1. ``pre_step(state_in, dt)`` – called at the top of ``step()`` to
       snapshot the input state before the solver modifies anything.

    2. ``post_solve(solver, state_in, state_out, dt)`` – called after the
       PGS solve and velocity integration (after stage 7) but before
       the method returns.  This is where spike detection and artifact
       writing happen.

    The capture is designed to be zero-cost when disabled: the hooks
    return immediately without touching GPU memory.
    """

    def __init__(self, config: SpikeCaptureConfig | None = None):
        self.config = config or SpikeCaptureConfig()
        self._capture_count: int = 0
        self._step_index: int = 0
        # Pre-step snapshot (numpy copies, taken on CPU)
        self._pre_joint_q: np.ndarray | None = None
        self._pre_joint_qd: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enable(self, threshold: float | None = None, output_dir: str | Path | None = None) -> None:
        """Enable capture programmatically."""
        self.config.enabled = True
        if threshold is not None:
            self.config.threshold = threshold
        if output_dir is not None:
            self.config.output_dir = Path(output_dir)

    def disable(self) -> None:
        """Disable capture."""
        self.config.enabled = False

    @property
    def is_active(self) -> bool:
        return self.config.enabled and self._capture_count < self.config.max_captures

    # ------------------------------------------------------------------
    # Solver hooks
    # ------------------------------------------------------------------

    def pre_step(self, state_in, dt: float) -> None:
        """Snapshot input state before the solver mutates anything.

        Args:
            state_in: Newton State with joint_q and joint_qd arrays.
            dt: The substep timestep.
        """
        if not self.is_active:
            return

        # Copy to CPU / numpy.  These are small (num_dofs) arrays so the
        # sync cost is negligible relative to the solver work.
        self._pre_joint_q = state_in.joint_q.numpy().copy()
        self._pre_joint_qd = state_in.joint_qd.numpy().copy()
        self._step_index += 1

    def post_solve(self, solver, state_in, state_out, dt: float) -> str | None:
        """Detect spikes and write capture artifacts.

        Args:
            solver: The SolverFeatherPGS instance (for v_out, impulses, params).
            state_in: Newton State (input to the step).
            state_out: Newton State (output of the step, after integration).
            dt: The substep timestep.

        Returns:
            Path to the written .npz file if a spike was captured, else None.
        """
        if not self.is_active:
            return None
        if self._pre_joint_qd is None:
            return None

        # Read the post-solve generalized velocity
        post_qd = state_out.joint_qd.numpy()
        max_abs_qd = float(np.max(np.abs(post_qd)))

        if max_abs_qd <= self.config.threshold:
            return None

        # --- Spike detected! ---
        return self._write_artifact(solver, state_in, state_out, post_qd, max_abs_qd, dt)

    # ------------------------------------------------------------------
    # Artifact writing
    # ------------------------------------------------------------------

    def _write_artifact(
        self,
        solver,
        state_in,
        state_out,
        post_qd: np.ndarray,
        max_abs_qd: float,
        dt: float,
    ) -> str:
        """Serialize a spike snapshot to an .npz file.

        The artifact contains everything needed for an offline replay,
        including the full dense constraint problem that allows re-running
        the PGS solve in pure numpy without Warp or GPU.

        State arrays:

        - **pre_joint_q**: generalized positions before the step
        - **pre_joint_qd**: generalized velocities before the step
        - **post_joint_q**: generalized positions after integration
        - **post_joint_qd**: generalized velocities after integration

        Solver intermediates:

        - **v_out**: the solved velocity from PGS (before integration, in DOF space)
        - **v_hat**: the velocity predictor (unconstrained predicted velocity)
        - **impulses**: the dense PGS impulses, shape ``(world_count, max_constraints)``

        Constraint problem (needed for faithful PGS replay):

        - **world_C**: Delassus matrix, shape ``(world_count, max_constraints, max_constraints)``
        - **world_diag**: regularized diagonal, shape ``(world_count, max_constraints)``
        - **world_rhs**: right-hand side bias, shape ``(world_count, max_constraints)``
        - **world_row_type**: constraint type per row (0=contact, 1=target, 2=friction, 3=limit)
        - **world_row_parent**: parent row index for friction rows (-1 otherwise)
        - **world_row_mu**: friction coefficient per row
        - **constraint_count**: active constraint count per world
        - **Y_world**: inverse-mass–weighted Jacobian transpose,
          shape ``(world_count, max_constraints, max_world_dofs)``,
          used to reconstruct ``v_out = v_hat + Y^T impulses``

        Scalar parameters and metadata:

        - **solver_params**: dt, pgs_iterations, pgs_beta, pgs_cfm, pgs_omega, etc.
        - **meta**: step_index, timestamp, max_abs_qd, capture_index
        """
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        tag = f"spike_{self._capture_count:04d}_step{self._step_index}_maxqd{max_abs_qd:.1f}"
        path = self.config.output_dir / f"{tag}.npz"

        # Collect arrays – all synced to CPU
        data = {}
        data["pre_joint_q"] = self._pre_joint_q
        data["pre_joint_qd"] = self._pre_joint_qd
        data["post_joint_q"] = state_out.joint_q.numpy()
        data["post_joint_qd"] = post_qd

        # v_out holds the post-PGS velocity in DOF-space (before integration)
        if hasattr(solver, "v_out") and solver.v_out is not None:
            data["v_out"] = solver.v_out.numpy()

        # v_hat is the unconstrained velocity predictor: v + M^-1 * f * dt
        if hasattr(solver, "v_hat") and solver.v_hat is not None:
            data["v_hat"] = solver.v_hat.numpy()

        # Dense PGS impulses
        if hasattr(solver, "impulses") and solver.impulses is not None:
            data["impulses"] = solver.impulses.numpy()

        # Constraint counts per world
        if hasattr(solver, "constraint_count") and solver.constraint_count is not None:
            data["constraint_count"] = solver.constraint_count.numpy()

        # ------------------------------------------------------------------
        # Constraint problem arrays (Stage 2: needed for faithful PGS replay)
        # ------------------------------------------------------------------
        # Delassus matrix C = J * H^{-1} * J^T, shape (world, max_c, max_c)
        if hasattr(solver, "C") and solver.C is not None:
            data["world_C"] = solver.C.numpy()

        # Regularized diagonal of C (after CFM addition)
        if hasattr(solver, "diag") and solver.diag is not None:
            data["world_diag"] = solver.diag.numpy()

        # Right-hand side bias vector
        if hasattr(solver, "rhs") and solver.rhs is not None:
            data["world_rhs"] = solver.rhs.numpy()

        # Constraint metadata arrays
        if hasattr(solver, "row_type") and solver.row_type is not None:
            data["world_row_type"] = solver.row_type.numpy()
        if hasattr(solver, "row_parent") and solver.row_parent is not None:
            data["world_row_parent"] = solver.row_parent.numpy()
        if hasattr(solver, "row_mu") and solver.row_mu is not None:
            data["world_row_mu"] = solver.row_mu.numpy()

        # Y_world = H^{-1} J^T, shape (world, max_constraints, max_world_dofs)
        # Needed to reconstruct v_out = v_hat + Y^T * impulses
        if hasattr(solver, "Y_world") and solver.Y_world is not None:
            data["Y_world"] = solver.Y_world.numpy()

        # Dense max constraints (scalar, needed to interpret shapes)
        if hasattr(solver, "dense_max_constraints"):
            data["dense_max_constraints"] = np.array([solver.dense_max_constraints], dtype=np.int32)

        # World DOF start offsets (for multi-world indexing into v_hat/v_out)
        if hasattr(solver, "world_dof_start") and solver.world_dof_start is not None:
            data["world_dof_start"] = solver.world_dof_start.numpy()

        # Scalar solver parameters packed as a structured dict
        data["solver_params"] = np.array([
            dt,
            solver.pgs_iterations,
            solver.pgs_beta,
            solver.pgs_cfm,
            solver.pgs_omega,
            float(solver.enable_joint_limits),
            float(solver.enable_contact_friction),
        ])
        data["solver_param_names"] = np.array([
            "dt", "pgs_iterations", "pgs_beta", "pgs_cfm",
            "pgs_omega", "enable_joint_limits", "enable_contact_friction",
        ])

        # Metadata
        data["meta"] = np.array([
            self._step_index,
            time.time(),
            max_abs_qd,
            self._capture_count,
        ])
        data["meta_names"] = np.array(["step_index", "timestamp", "max_abs_qd", "capture_index"])

        np.savez_compressed(str(path), **data)

        self._capture_count += 1
        print(
            f"[SpikeCapture] Spike #{self._capture_count} at step {self._step_index}: "
            f"max|qd|={max_abs_qd:.2f} > threshold={self.config.threshold:.2f}  "
            f"-> {path}"
        )
        return str(path)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of captures so far."""
        return (
            f"SpikeCapture: {self._capture_count} captures in {self._step_index} steps, "
            f"threshold={self.config.threshold}, dir={self.config.output_dir}"
        )
