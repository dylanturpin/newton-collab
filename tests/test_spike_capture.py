# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the velocity spike capture, replay, and PGS solver modules.

These tests validate the capture/replay tooling in isolation, without
requiring a GPU, Warp, or a full Newton simulation.  They use mock
objects that mimic the minimal interface expected by SpikeCapture.

The tests are organized into five groups:

1. SpikeCaptureConfig – environment-driven configuration.
2. SpikeCapture – spike detection and artifact writing.
3. SpikeArtifact (analysis) – loading, classification, velocity profiles.
4. pgs_solve_numpy – pure-numpy PGS solver correctness.
5. SpikeArtifact (replay) – end-to-end replay with drift measurement.
"""

import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from newton._src.solvers.feather_pgs.spike_capture import SpikeCapture, SpikeCaptureConfig
from newton._src.solvers.feather_pgs.spike_replay import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    ReplayResult,
    SpikeArtifact,
    pgs_solve_numpy,
)


class MockArray:
    """Mimics a warp array with .numpy() returning a pre-set ndarray."""

    def __init__(self, data: np.ndarray):
        self._data = data

    def numpy(self):
        return self._data.copy()


def _make_state(joint_q, joint_qd):
    """Create a mock state object with joint_q and joint_qd."""
    return SimpleNamespace(
        joint_q=MockArray(np.array(joint_q, dtype=np.float32)),
        joint_qd=MockArray(np.array(joint_qd, dtype=np.float32)),
    )


def _make_solver(
    v_out=None,
    v_hat=None,
    impulses=None,
    constraint_count=None,
    C=None,
    diag=None,
    rhs=None,
    row_type=None,
    row_parent=None,
    row_mu=None,
    Y_world=None,
    world_dof_start=None,
    dense_max_constraints=4,
):
    """Create a mock solver with the arrays that SpikeCapture reads.

    Accepts optional constraint-level arrays for Stage 2 replay support.
    """
    s = SimpleNamespace(
        pgs_iterations=8,
        pgs_beta=0.05,
        pgs_cfm=1e-6,
        pgs_omega=1.0,
        enable_joint_limits=True,
        enable_contact_friction=True,
        dense_max_constraints=dense_max_constraints,
        v_out=MockArray(np.array(v_out, dtype=np.float32)) if v_out is not None else None,
        v_hat=MockArray(np.array(v_hat, dtype=np.float32)) if v_hat is not None else None,
        impulses=MockArray(np.array(impulses, dtype=np.float32)) if impulses is not None else None,
        constraint_count=MockArray(np.array(constraint_count, dtype=np.int32)) if constraint_count is not None else None,
        C=MockArray(np.array(C, dtype=np.float32)) if C is not None else None,
        diag=MockArray(np.array(diag, dtype=np.float32)) if diag is not None else None,
        rhs=MockArray(np.array(rhs, dtype=np.float32)) if rhs is not None else None,
        row_type=MockArray(np.array(row_type, dtype=np.int32)) if row_type is not None else None,
        row_parent=MockArray(np.array(row_parent, dtype=np.int32)) if row_parent is not None else None,
        row_mu=MockArray(np.array(row_mu, dtype=np.float32)) if row_mu is not None else None,
        Y_world=MockArray(np.array(Y_world, dtype=np.float32)) if Y_world is not None else None,
        world_dof_start=MockArray(np.array(world_dof_start, dtype=np.int32)) if world_dof_start is not None else None,
    )
    return s


@pytest.fixture
def tmp_capture_dir():
    d = tempfile.mkdtemp(prefix="spike_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════
# 1. SpikeCaptureConfig
# ══════════════════════════════════════════════════════════════════


class TestSpikeCaptureConfig:
    def test_defaults(self):
        cfg = SpikeCaptureConfig()
        assert cfg.threshold == 50.0
        assert cfg.max_captures == 100
        assert not cfg.enabled  # disabled by default

    def test_repr(self):
        cfg = SpikeCaptureConfig()
        r = repr(cfg)
        assert "enabled=False" in r
        assert "threshold=50.0" in r


# ══════════════════════════════════════════════════════════════════
# 2. SpikeCapture
# ══════════════════════════════════════════════════════════════════


class TestSpikeCapture:
    def test_no_capture_when_disabled(self, tmp_capture_dir):
        cfg = SpikeCaptureConfig()
        cfg.enabled = False
        cfg.output_dir = tmp_capture_dir
        sc = SpikeCapture(cfg)

        state_in = _make_state([0.0, 0.1], [0.0, 0.5])
        state_out = _make_state([0.0, 0.1], [0.0, 100.0])  # would be a spike
        solver = _make_solver()

        sc.pre_step(state_in, 0.01)
        result = sc.post_solve(solver, state_in, state_out, 0.01)
        assert result is None
        assert sc._capture_count == 0

    def test_no_capture_below_threshold(self, tmp_capture_dir):
        cfg = SpikeCaptureConfig()
        cfg.enabled = True
        cfg.threshold = 50.0
        cfg.output_dir = tmp_capture_dir
        sc = SpikeCapture(cfg)

        state_in = _make_state([0.0, 0.1], [0.0, 0.5])
        state_out = _make_state([0.0, 0.1], [0.0, 10.0])  # below threshold
        solver = _make_solver(v_out=[0.0, 10.0], v_hat=[0.0, 9.5])

        sc.pre_step(state_in, 0.01)
        result = sc.post_solve(solver, state_in, state_out, 0.01)
        assert result is None
        assert sc._capture_count == 0

    def test_capture_above_threshold(self, tmp_capture_dir):
        cfg = SpikeCaptureConfig()
        cfg.enabled = True
        cfg.threshold = 50.0
        cfg.output_dir = tmp_capture_dir
        sc = SpikeCapture(cfg)

        state_in = _make_state([0.0, 0.1, 0.2], [1.0, 2.0, 3.0])
        state_out = _make_state([0.0, 0.1, 0.2], [1.0, 2.0, 200.0])  # spike!
        solver = _make_solver(
            v_out=[1.0, 2.0, 200.0],
            v_hat=[1.0, 2.0, 5.0],
            impulses=[[0.0, 0.1, 50.0, 0.0]],
            constraint_count=[2],
        )

        sc.pre_step(state_in, 0.01)
        result = sc.post_solve(solver, state_in, state_out, 0.01)

        assert result is not None
        assert Path(result).exists()
        assert sc._capture_count == 1

        # Verify artifact is loadable and correct
        artifact = SpikeArtifact.load(result)
        assert artifact.max_abs_post_qd == 200.0
        assert artifact.num_dofs == 3
        np.testing.assert_array_almost_equal(artifact.pre_joint_qd, [1.0, 2.0, 3.0])
        np.testing.assert_array_almost_equal(artifact.post_joint_qd, [1.0, 2.0, 200.0])

    def test_capture_includes_constraint_data(self, tmp_capture_dir):
        """Stage 2: verify that constraint-level arrays are captured."""
        cfg = SpikeCaptureConfig()
        cfg.enabled = True
        cfg.threshold = 50.0
        cfg.output_dir = tmp_capture_dir
        sc = SpikeCapture(cfg)

        # Build a small constraint problem: 1 world, 2 constraints, 3 DOFs
        max_c = 4
        n_dofs = 3
        C = np.eye(max_c, dtype=np.float32).reshape(1, max_c, max_c) * 2.0
        diag = np.full((1, max_c), 2.0, dtype=np.float32)
        rhs = np.array([[0.1, 0.2, 0.0, 0.0]], dtype=np.float32)
        row_type = np.array([[PGS_CONSTRAINT_TYPE_JOINT_LIMIT, PGS_CONSTRAINT_TYPE_JOINT_LIMIT, 0, 0]], dtype=np.int32)
        row_parent = np.full((1, max_c), -1, dtype=np.int32)
        row_mu = np.zeros((1, max_c), dtype=np.float32)
        Y = np.eye(max_c, n_dofs, dtype=np.float32).reshape(1, max_c, n_dofs)

        state_in = _make_state([0.0, 0.1, 0.2], [1.0, 2.0, 3.0])
        state_out = _make_state([0.0, 0.1, 0.2], [1.0, 2.0, 200.0])
        solver = _make_solver(
            v_out=[1.0, 2.0, 200.0],
            v_hat=[1.0, 2.0, 5.0],
            impulses=[[0.0, 0.1, 0.0, 0.0]],
            constraint_count=[2],
            C=C, diag=diag, rhs=rhs,
            row_type=row_type, row_parent=row_parent, row_mu=row_mu,
            Y_world=Y, world_dof_start=[0],
            dense_max_constraints=max_c,
        )

        sc.pre_step(state_in, 0.01)
        result = sc.post_solve(solver, state_in, state_out, 0.01)
        assert result is not None

        artifact = SpikeArtifact.load(result)
        assert artifact.can_replay
        assert artifact.world_C is not None
        assert artifact.world_diag is not None
        assert artifact.world_rhs is not None
        assert artifact.world_row_type is not None
        assert artifact.world_row_parent is not None
        assert artifact.world_row_mu is not None
        assert artifact.Y_world is not None
        assert artifact.dense_max_constraints == max_c

    def test_max_captures_respected(self, tmp_capture_dir):
        cfg = SpikeCaptureConfig()
        cfg.enabled = True
        cfg.threshold = 1.0
        cfg.max_captures = 2
        cfg.output_dir = tmp_capture_dir
        sc = SpikeCapture(cfg)

        for i in range(5):
            state_in = _make_state([0.0], [0.0])
            state_out = _make_state([0.0], [100.0])
            solver = _make_solver(v_out=[100.0], v_hat=[0.5])
            sc.pre_step(state_in, 0.01)
            sc.post_solve(solver, state_in, state_out, 0.01)

        assert sc._capture_count == 2  # capped at max

    def test_enable_disable_api(self, tmp_capture_dir):
        sc = SpikeCapture()
        assert not sc.is_active

        sc.enable(threshold=25.0, output_dir=tmp_capture_dir)
        assert sc.is_active
        assert sc.config.threshold == 25.0

        sc.disable()
        assert not sc.is_active


# ══════════════════════════════════════════════════════════════════
# 3. SpikeArtifact (analysis)
# ══════════════════════════════════════════════════════════════════


class TestSpikeArtifact:
    def test_load_and_analyze(self, tmp_capture_dir):
        # Create a synthetic artifact
        path = tmp_capture_dir / "test_spike.npz"
        np.savez_compressed(
            str(path),
            pre_joint_q=np.array([0.0, 0.1, 0.2], dtype=np.float32),
            pre_joint_qd=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            post_joint_q=np.array([0.0, 0.1, 0.2], dtype=np.float32),
            post_joint_qd=np.array([1.0, 2.0, 300.0], dtype=np.float32),
            v_out=np.array([1.0, 2.0, 300.0], dtype=np.float32),
            v_hat=np.array([1.0, 2.0, 5.0], dtype=np.float32),
            impulses=np.array([[0.0, 0.1, 0.0, 0.0]], dtype=np.float32),
            constraint_count=np.array([1], dtype=np.int32),
            solver_params=np.array([0.01, 8, 0.05, 1e-6, 1.0, 1.0, 1.0]),
            solver_param_names=np.array(["dt", "pgs_iterations", "pgs_beta", "pgs_cfm", "pgs_omega", "enable_joint_limits", "enable_contact_friction"]),
            meta=np.array([100, 1.0, 300.0, 0]),
            meta_names=np.array(["step_index", "timestamp", "max_abs_qd", "capture_index"]),
        )

        artifact = SpikeArtifact.load(path)
        assert artifact.max_abs_post_qd == 300.0
        assert artifact.max_abs_pre_qd == 3.0
        assert artifact.velocity_amplification == 100.0
        assert artifact.num_dofs == 3

        top = artifact.top_spike_dofs(2)
        assert top[0][0] == 2  # DOF 2 has the spike
        assert top[0][1] == 300.0

    def test_classify_pgs_divergence(self, tmp_capture_dir):
        path = tmp_capture_dir / "pgs_div.npz"
        np.savez_compressed(
            str(path),
            pre_joint_q=np.zeros(5, dtype=np.float32),
            pre_joint_qd=np.ones(5, dtype=np.float32),
            post_joint_q=np.zeros(5, dtype=np.float32),
            post_joint_qd=np.array([1, 1, 1, 1, 500], dtype=np.float32),
            v_out=np.array([1, 1, 1, 1, 500], dtype=np.float32),
            v_hat=np.array([1, 1, 1, 1, 5], dtype=np.float32),
            meta=np.array([10, 1.0, 500.0, 0]),
            meta_names=np.array(["step_index", "timestamp", "max_abs_qd", "capture_index"]),
        )
        artifact = SpikeArtifact.load(path)
        assert artifact.classify_spike() == "pgs_divergence"

    def test_classify_unconstrained(self, tmp_capture_dir):
        path = tmp_capture_dir / "uncon.npz"
        np.savez_compressed(
            str(path),
            pre_joint_q=np.zeros(5, dtype=np.float32),
            pre_joint_qd=np.ones(5, dtype=np.float32),
            post_joint_q=np.zeros(5, dtype=np.float32),
            post_joint_qd=np.array([1, 1, 1, 1, 100], dtype=np.float32),
            v_out=np.array([1, 1, 1, 1, 100], dtype=np.float32),
            v_hat=np.array([1, 1, 1, 1, 90], dtype=np.float32),
            meta=np.array([10, 1.0, 100.0, 0]),
            meta_names=np.array(["step_index", "timestamp", "max_abs_qd", "capture_index"]),
        )
        artifact = SpikeArtifact.load(path)
        assert artifact.classify_spike() == "unconstrained"

    def test_can_replay_false_without_constraint_data(self, tmp_capture_dir):
        """Artifacts without Stage 2 data should report can_replay=False."""
        path = tmp_capture_dir / "no_replay.npz"
        np.savez_compressed(
            str(path),
            pre_joint_q=np.zeros(3, dtype=np.float32),
            pre_joint_qd=np.ones(3, dtype=np.float32),
            post_joint_q=np.zeros(3, dtype=np.float32),
            post_joint_qd=np.array([1, 1, 100], dtype=np.float32),
            v_out=np.array([1, 1, 100], dtype=np.float32),
            v_hat=np.array([1, 1, 5], dtype=np.float32),
            impulses=np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            constraint_count=np.array([0], dtype=np.int32),
            meta=np.array([10, 1.0, 100.0, 0]),
            meta_names=np.array(["step_index", "timestamp", "max_abs_qd", "capture_index"]),
        )
        artifact = SpikeArtifact.load(path)
        assert not artifact.can_replay

    def test_print_summary(self, tmp_capture_dir, capsys):
        path = tmp_capture_dir / "summary_test.npz"
        np.savez_compressed(
            str(path),
            pre_joint_q=np.zeros(3, dtype=np.float32),
            pre_joint_qd=np.array([1.0, 2.0, 3.0], dtype=np.float32),
            post_joint_q=np.zeros(3, dtype=np.float32),
            post_joint_qd=np.array([1.0, 2.0, 300.0], dtype=np.float32),
            v_out=np.array([1.0, 2.0, 300.0], dtype=np.float32),
            v_hat=np.array([1.0, 2.0, 5.0], dtype=np.float32),
            solver_params=np.array([0.01, 8, 0.05, 1e-6, 1.0, 1.0, 1.0]),
            solver_param_names=np.array(["dt", "pgs_iterations", "pgs_beta", "pgs_cfm", "pgs_omega", "enable_joint_limits", "enable_contact_friction"]),
            meta=np.array([100, 1.0, 300.0, 0]),
            meta_names=np.array(["step_index", "timestamp", "max_abs_qd", "capture_index"]),
        )
        artifact = SpikeArtifact.load(path)
        artifact.print_summary()
        captured = capsys.readouterr().out
        assert "300.0" in captured
        assert "Amplification" in captured
        assert "Replay capable" in captured


# ══════════════════════════════════════════════════════════════════
# 4. pgs_solve_numpy (pure-numpy PGS solver)
# ══════════════════════════════════════════════════════════════════


class TestPgsSolveNumpy:
    """Validate the pure-numpy PGS solver against known analytical solutions."""

    def test_no_constraints(self):
        """Zero active constraints should return zero impulses."""
        max_c = 4
        result = pgs_solve_numpy(
            C=np.eye(max_c, dtype=np.float32),
            diag=np.ones(max_c, dtype=np.float32),
            rhs=np.zeros(max_c, dtype=np.float32),
            row_type=np.zeros(max_c, dtype=np.int32),
            row_parent=np.full(max_c, -1, dtype=np.int32),
            row_mu=np.zeros(max_c, dtype=np.float32),
            constraint_count=0,
            iterations=8,
            omega=1.0,
        )
        np.testing.assert_array_equal(result, 0.0)

    def test_single_contact_constraint(self):
        """A single contact constraint with rhs < 0 should produce lambda > 0.

        The system is: C * lambda + rhs = 0, lambda >= 0.
        With C = [[d]], rhs = [-b], the solution is lambda = b / d.
        """
        max_c = 4
        d = 2.0
        b = 1.0  # rhs = -b => lambda = b/d = 0.5
        C = np.zeros((max_c, max_c), dtype=np.float32)
        C[0, 0] = d
        diag = np.zeros(max_c, dtype=np.float32)
        diag[0] = d
        rhs = np.zeros(max_c, dtype=np.float32)
        rhs[0] = -b

        result = pgs_solve_numpy(
            C=C, diag=diag, rhs=rhs,
            row_type=np.array([PGS_CONSTRAINT_TYPE_CONTACT, 0, 0, 0], dtype=np.int32),
            row_parent=np.full(max_c, -1, dtype=np.int32),
            row_mu=np.zeros(max_c, dtype=np.float32),
            constraint_count=1,
            iterations=10,
            omega=1.0,
        )
        np.testing.assert_allclose(result[0], 0.5, atol=1e-6)

    def test_contact_constraint_clamped_at_zero(self):
        """A contact constraint with rhs > 0 should clamp impulse to 0."""
        max_c = 4
        C = np.eye(max_c, dtype=np.float32) * 2.0
        diag = np.full(max_c, 2.0, dtype=np.float32)
        rhs = np.zeros(max_c, dtype=np.float32)
        rhs[0] = 1.0  # Positive rhs -> solution would be negative -> clamp to 0

        result = pgs_solve_numpy(
            C=C, diag=diag, rhs=rhs,
            row_type=np.array([PGS_CONSTRAINT_TYPE_CONTACT, 0, 0, 0], dtype=np.int32),
            row_parent=np.full(max_c, -1, dtype=np.int32),
            row_mu=np.zeros(max_c, dtype=np.float32),
            constraint_count=1,
            iterations=10,
            omega=1.0,
        )
        assert result[0] == 0.0

    def test_joint_limit_constraint(self):
        """Joint limit constraints should also clamp at zero."""
        max_c = 4
        C = np.eye(max_c, dtype=np.float32)
        diag = np.ones(max_c, dtype=np.float32)
        rhs = np.zeros(max_c, dtype=np.float32)
        rhs[0] = -3.0  # lambda should be 3.0

        result = pgs_solve_numpy(
            C=C, diag=diag, rhs=rhs,
            row_type=np.array([PGS_CONSTRAINT_TYPE_JOINT_LIMIT, 0, 0, 0], dtype=np.int32),
            row_parent=np.full(max_c, -1, dtype=np.int32),
            row_mu=np.zeros(max_c, dtype=np.float32),
            constraint_count=1,
            iterations=10,
            omega=1.0,
        )
        np.testing.assert_allclose(result[0], 3.0, atol=1e-6)

    def test_friction_cone_projection(self):
        """Friction impulses should be projected onto the friction cone."""
        max_c = 4
        mu = 0.5

        # Setup: normal contact at row 0, friction rows at 1 and 2
        C = np.eye(max_c, dtype=np.float32)
        diag = np.ones(max_c, dtype=np.float32)
        rhs = np.zeros(max_c, dtype=np.float32)
        rhs[0] = -2.0   # Normal: lambda_n = 2.0
        rhs[1] = -10.0  # Friction1: would be 10.0 unclamped
        rhs[2] = -10.0  # Friction2: would be 10.0 unclamped

        row_type = np.array([
            PGS_CONSTRAINT_TYPE_CONTACT,
            PGS_CONSTRAINT_TYPE_FRICTION,
            PGS_CONSTRAINT_TYPE_FRICTION,
            0,
        ], dtype=np.int32)
        row_parent = np.array([0, 0, 0, -1], dtype=np.int32)
        row_mu = np.array([0.0, mu, mu, 0.0], dtype=np.float32)

        result = pgs_solve_numpy(
            C=C, diag=diag, rhs=rhs,
            row_type=row_type, row_parent=row_parent, row_mu=row_mu,
            constraint_count=3,
            iterations=50,  # Need more iterations for coupled system
            omega=1.0,
        )

        # Normal impulse should converge to 2.0
        np.testing.assert_allclose(result[0], 2.0, atol=0.1)

        # Friction magnitude should be <= mu * lambda_n = 1.0
        fric_mag = np.sqrt(result[1]**2 + result[2]**2)
        assert fric_mag <= mu * result[0] + 0.01, f"Friction {fric_mag} > cone {mu * result[0]}"

    def test_omega_under_relaxation(self):
        """Under-relaxation (omega < 1) should converge more slowly but still converge."""
        max_c = 4
        C = np.eye(max_c, dtype=np.float32)
        diag = np.ones(max_c, dtype=np.float32)
        rhs = np.zeros(max_c, dtype=np.float32)
        rhs[0] = -1.0

        # With omega=1.0, converges in 1 iteration
        result_1 = pgs_solve_numpy(
            C=C, diag=diag, rhs=rhs,
            row_type=np.array([PGS_CONSTRAINT_TYPE_CONTACT, 0, 0, 0], dtype=np.int32),
            row_parent=np.full(max_c, -1, dtype=np.int32),
            row_mu=np.zeros(max_c, dtype=np.float32),
            constraint_count=1, iterations=1, omega=1.0,
        )

        # With omega=0.5, needs more iterations
        result_half = pgs_solve_numpy(
            C=C, diag=diag, rhs=rhs,
            row_type=np.array([PGS_CONSTRAINT_TYPE_CONTACT, 0, 0, 0], dtype=np.int32),
            row_parent=np.full(max_c, -1, dtype=np.int32),
            row_mu=np.zeros(max_c, dtype=np.float32),
            constraint_count=1, iterations=1, omega=0.5,
        )

        assert result_1[0] == 1.0
        assert result_half[0] == 0.5  # Only half-step toward solution

    def test_cfm_override(self):
        """CFM override should reduce impulse magnitude by softening the diagonal.

        In PGS, CFM is added to the diagonal divisor only (not to the
        Delassus matrix itself).  For a single constraint with C[0,0]=d,
        diag=d, rhs=-b, after adding cfm to diag the GS iteration does:

            w = rhs + C * lambda = -b + d * lambda
            delta = -w / (d + cfm)
            lambda_new = lambda + delta

        At convergence: d * lambda - b = 0  is NOT the fixed point because
        the divisor is (d + cfm) while the residual uses C[0,0] = d.
        The effective fixed point is lambda = b / (d + cfm) when starting
        from zero and running enough iterations.  But for GS the first-
        iteration update is lambda = b / (d + cfm), and further iterations
        observe a non-zero residual because C[0,0]*lambda != -rhs.

        For a diagonal C, the converged solution with CFM in the divisor
        only is lambda = b * d / (d*(d+cfm)) which drifts depending on
        iterations.  We just check that CFM reduces the impulse and that
        more iterations converge toward the no-CFM solution.
        """
        max_c = 4
        C = np.eye(max_c, dtype=np.float32)
        diag = np.ones(max_c, dtype=np.float32)
        rhs = np.zeros(max_c, dtype=np.float32)
        rhs[0] = -1.0

        # Without CFM override
        result_no_cfm = pgs_solve_numpy(
            C=C, diag=diag, rhs=rhs,
            row_type=np.array([PGS_CONSTRAINT_TYPE_CONTACT, 0, 0, 0], dtype=np.int32),
            row_parent=np.full(max_c, -1, dtype=np.int32),
            row_mu=np.zeros(max_c, dtype=np.float32),
            constraint_count=1, iterations=10, omega=1.0,
        )
        np.testing.assert_allclose(result_no_cfm[0], 1.0, atol=1e-6)

        # With CFM override (1 iteration): first update is b/(d+cfm) = 1/2
        result_cfm_1 = pgs_solve_numpy(
            C=C, diag=diag, rhs=rhs,
            row_type=np.array([PGS_CONSTRAINT_TYPE_CONTACT, 0, 0, 0], dtype=np.int32),
            row_parent=np.full(max_c, -1, dtype=np.int32),
            row_mu=np.zeros(max_c, dtype=np.float32),
            constraint_count=1, iterations=1, omega=1.0,
            cfm_override=1.0,
        )
        np.testing.assert_allclose(result_cfm_1[0], 0.5, atol=1e-6)

        # With many iterations the CFM-in-divisor GS converges to the
        # same fixed point (C*lambda + rhs = 0) since CFM only changes
        # the step size, not the residual.  So the main effect of CFM
        # is slower convergence (damping).  With fewer iterations, the
        # impulse is smaller because GS hasn't fully converged.
        result_cfm_few = pgs_solve_numpy(
            C=C, diag=diag, rhs=rhs,
            row_type=np.array([PGS_CONSTRAINT_TYPE_CONTACT, 0, 0, 0], dtype=np.int32),
            row_parent=np.full(max_c, -1, dtype=np.int32),
            row_mu=np.zeros(max_c, dtype=np.float32),
            constraint_count=1, iterations=4, omega=1.0,
            cfm_override=1.0,
        )
        # With 4 iterations and CFM=1.0, impulse should be less than
        # the fully converged value of 1.0 but more than the 1-iteration value
        assert 0.5 < result_cfm_few[0] < 1.0

    def test_two_coupled_constraints(self):
        """Two coupled contact constraints should converge to the correct solution."""
        max_c = 4
        C = np.array([
            [2.0, 0.5, 0.0, 0.0],
            [0.5, 2.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32)
        diag = np.array([2.0, 2.0, 1.0, 1.0], dtype=np.float32)
        rhs = np.array([-1.0, -1.0, 0.0, 0.0], dtype=np.float32)

        result = pgs_solve_numpy(
            C=C, diag=diag, rhs=rhs,
            row_type=np.array([PGS_CONSTRAINT_TYPE_CONTACT] * 2 + [0, 0], dtype=np.int32),
            row_parent=np.full(max_c, -1, dtype=np.int32),
            row_mu=np.zeros(max_c, dtype=np.float32),
            constraint_count=2,
            iterations=100,
            omega=1.0,
        )

        # Analytical solution: C[:2,:2] * x = [1, 1] -> x = C^{-1} [1, 1]
        C_sub = C[:2, :2]
        x_exact = np.linalg.solve(C_sub, [1.0, 1.0])
        np.testing.assert_allclose(result[:2], x_exact, atol=1e-4)


# ══════════════════════════════════════════════════════════════════
# 5. SpikeArtifact (replay)
# ══════════════════════════════════════════════════════════════════


def _make_replay_artifact(tmp_dir, max_c=4, n_dofs=3, n_active=2):
    """Create a synthetic replay-capable artifact with known solution.

    The constraint problem is:
      - 2 joint-limit constraints on an identity Delassus matrix
      - RHS = [-1, -2, 0, 0] -> solution impulses = [1, 2, 0, 0]
      - Y = identity (top-left block) so v_out = v_hat + impulses[:n_dofs]
      - v_hat = [0, 0, 0] -> v_out = [1, 2, 0]
    """
    C = np.eye(max_c, dtype=np.float32).reshape(1, max_c, max_c)
    diag = np.ones((1, max_c), dtype=np.float32)
    rhs = np.zeros((1, max_c), dtype=np.float32)
    rhs[0, 0] = -1.0
    rhs[0, 1] = -2.0
    row_type = np.zeros((1, max_c), dtype=np.int32)
    row_type[0, 0] = PGS_CONSTRAINT_TYPE_JOINT_LIMIT
    row_type[0, 1] = PGS_CONSTRAINT_TYPE_JOINT_LIMIT
    row_parent = np.full((1, max_c), -1, dtype=np.int32)
    row_mu = np.zeros((1, max_c), dtype=np.float32)

    # Y: (1, max_c, n_dofs) – identity top-left block
    Y = np.zeros((1, max_c, n_dofs), dtype=np.float32)
    for i in range(min(max_c, n_dofs)):
        Y[0, i, i] = 1.0

    v_hat = np.zeros(n_dofs, dtype=np.float32)
    # Original solver produced impulses [1, 2, 0, 0] and v_out [1, 2, 0]
    orig_impulses = np.array([[1.0, 2.0, 0.0, 0.0]], dtype=np.float32)
    v_out = np.array([1.0, 2.0, 0.0], dtype=np.float32)

    path = tmp_dir / "replay_test.npz"
    np.savez_compressed(
        str(path),
        pre_joint_q=np.zeros(n_dofs, dtype=np.float32),
        pre_joint_qd=np.zeros(n_dofs, dtype=np.float32),
        post_joint_q=np.zeros(n_dofs, dtype=np.float32),
        post_joint_qd=v_out.copy(),
        v_out=v_out,
        v_hat=v_hat,
        impulses=orig_impulses,
        constraint_count=np.array([n_active], dtype=np.int32),
        world_C=C,
        world_diag=diag,
        world_rhs=rhs,
        world_row_type=row_type,
        world_row_parent=row_parent,
        world_row_mu=row_mu,
        Y_world=Y,
        world_dof_start=np.array([0], dtype=np.int32),
        dense_max_constraints=np.array([max_c], dtype=np.int32),
        solver_params=np.array([0.01, 8, 0.05, 1e-6, 1.0, 1.0, 1.0]),
        solver_param_names=np.array([
            "dt", "pgs_iterations", "pgs_beta", "pgs_cfm",
            "pgs_omega", "enable_joint_limits", "enable_contact_friction",
        ]),
        meta=np.array([42, 1.0, 2.0, 0]),
        meta_names=np.array(["step_index", "timestamp", "max_abs_qd", "capture_index"]),
    )
    return path


class TestSpikeArtifactReplay:
    def test_replay_matches_original(self, tmp_capture_dir):
        """Replay with original parameters should reproduce the captured impulses."""
        path = _make_replay_artifact(tmp_capture_dir)
        artifact = SpikeArtifact.load(path)
        assert artifact.can_replay

        result = artifact.replay_pgs(world_idx=0)

        # Impulses should match closely (identity Delassus, simple problem)
        np.testing.assert_allclose(result.replayed_impulses[:2], [1.0, 2.0], atol=1e-5)
        assert result.max_impulse_drift < 1e-5

        # Velocity should also match
        assert result.replayed_v_out is not None
        np.testing.assert_allclose(result.replayed_v_out, [1.0, 2.0, 0.0], atol=1e-5)
        assert result.max_velocity_drift < 1e-5

    def test_replay_with_different_omega(self, tmp_capture_dir):
        """Replay with omega < 1 should produce different (smaller) impulses."""
        path = _make_replay_artifact(tmp_capture_dir)
        artifact = SpikeArtifact.load(path)

        result = artifact.replay_pgs(omega=0.5, iterations=1)

        # With omega=0.5 and 1 iteration, impulses should be half
        np.testing.assert_allclose(result.replayed_impulses[:2], [0.5, 1.0], atol=1e-5)

        # Drift should be non-zero since we changed parameters
        assert result.max_impulse_drift > 0.4

    def test_replay_with_cfm_override(self, tmp_capture_dir):
        """CFM override should reduce impulse magnitude vs baseline."""
        path = _make_replay_artifact(tmp_capture_dir)
        artifact = SpikeArtifact.load(path)

        result_baseline = artifact.replay_pgs()
        # Use only 1 iteration so the effect is clean: first GS step
        # with cfm=1.0 gives lambda = b/(d+cfm) = 1/2 instead of 1
        result_cfm = artifact.replay_pgs(cfm=1.0, iterations=1)

        np.testing.assert_allclose(result_cfm.replayed_impulses[0], 0.5, atol=1e-5)
        assert result_cfm.replayed_impulses[0] < result_baseline.replayed_impulses[0]

    def test_replay_velocity_reconstruction(self, tmp_capture_dir):
        """Replayed v_out should equal v_hat + Y^T * impulses."""
        path = _make_replay_artifact(tmp_capture_dir)
        artifact = SpikeArtifact.load(path)

        result = artifact.replay_pgs()
        assert result.replayed_v_out is not None
        assert result.original_v_out is not None
        assert result.v_hat_slice is not None

        # v_out = v_hat + Y^T * impulses; v_hat = 0, Y = I, impulses = [1, 2, 0, 0]
        # -> v_out = [1, 2, 0]
        np.testing.assert_allclose(result.replayed_v_out, [1.0, 2.0, 0.0], atol=1e-5)

    def test_replay_parameter_sweep(self, tmp_capture_dir):
        """Parameter sweep should return multiple results."""
        path = _make_replay_artifact(tmp_capture_dir)
        artifact = SpikeArtifact.load(path)

        results = artifact.replay_parameter_sweep(
            omega_values=[1.0, 0.8],
            cfm_values=[None, 1e-3],
            iteration_values=[8],
        )

        assert len(results) == 4  # 2 omega * 2 cfm * 1 iter
        # Each result should have valid drift measurements
        for r in results:
            assert isinstance(r, ReplayResult)
            assert r.max_impulse_drift >= 0.0

    def test_replay_drift_report_prints(self, tmp_capture_dir, capsys):
        """Drift report should print without errors."""
        path = _make_replay_artifact(tmp_capture_dir)
        artifact = SpikeArtifact.load(path)

        result = artifact.replay_pgs()
        result.print_drift_report()

        captured = capsys.readouterr().out
        assert "Drift Report" in captured
        assert "impulse drift" in captured
        assert "velocity drift" in captured

    def test_replay_not_possible_raises(self, tmp_capture_dir):
        """Replaying an artifact without constraint data should raise ValueError."""
        path = tmp_capture_dir / "no_constraint.npz"
        np.savez_compressed(
            str(path),
            pre_joint_q=np.zeros(3, dtype=np.float32),
            pre_joint_qd=np.ones(3, dtype=np.float32),
            post_joint_q=np.zeros(3, dtype=np.float32),
            post_joint_qd=np.array([1, 1, 100], dtype=np.float32),
            meta=np.array([10, 1.0, 100.0, 0]),
            meta_names=np.array(["step_index", "timestamp", "max_abs_qd", "capture_index"]),
        )
        artifact = SpikeArtifact.load(path)
        assert not artifact.can_replay
        with pytest.raises(ValueError, match="constraint-level data"):
            artifact.replay_pgs()
