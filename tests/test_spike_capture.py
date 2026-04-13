# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the velocity spike capture and replay modules.

These tests validate the capture/replay tooling in isolation, without
requiring a GPU, Warp, or a full Newton simulation.  They use mock
objects that mimic the minimal interface expected by SpikeCapture.
"""

import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from newton._src.solvers.feather_pgs.spike_capture import SpikeCapture, SpikeCaptureConfig
from newton._src.solvers.feather_pgs.spike_replay import SpikeArtifact


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


def _make_solver(v_out=None, v_hat=None, impulses=None, constraint_count=None):
    """Create a mock solver with the arrays that SpikeCapture reads."""
    s = SimpleNamespace(
        pgs_iterations=8,
        pgs_beta=0.05,
        pgs_cfm=1e-6,
        pgs_omega=1.0,
        enable_joint_limits=True,
        enable_contact_friction=True,
        v_out=MockArray(np.array(v_out, dtype=np.float32)) if v_out is not None else None,
        v_hat=MockArray(np.array(v_hat, dtype=np.float32)) if v_hat is not None else None,
        impulses=MockArray(np.array(impulses, dtype=np.float32)) if impulses is not None else None,
        constraint_count=MockArray(np.array(constraint_count, dtype=np.int32)) if constraint_count is not None else None,
    )
    return s


@pytest.fixture
def tmp_capture_dir():
    d = tempfile.mkdtemp(prefix="spike_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


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
            impulses=[0.0, 0.1, 50.0, 0.0],
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
            impulses=np.array([0.0, 0.1, 0.0, 0.0], dtype=np.float32),
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
