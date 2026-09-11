# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the FeatherPGS rigid-body showreel examples."""

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton


def make_solver(model: newton.Model, **overrides) -> newton.solvers.SolverFeatherPGS:
    """Build a FeatherPGS solver with showreel defaults for the model's device.

    CUDA uses the matrix-free route so free bodies and articulations share one
    pass; CPU falls back to the split mode, which is the only CPU-capable mode.
    """
    kwargs = {
        "pgs_mode": "matrix_free" if model.device.is_cuda else "split",
        "pgs_iterations": 32,
        "pgs_beta": 0.2,
        "angular_damping": 0.0,
    }
    kwargs.update(overrides)
    return newton.solvers.SolverFeatherPGS(model, **kwargs)


def circle_segments(center, radius: float, segments: int = 96, axis: str = "z"):
    """Return (starts, ends) Warp vec3 arrays tracing a circle for ``viewer.log_lines``."""
    angles = np.linspace(0.0, 2.0 * math.pi, segments + 1, dtype=np.float32)
    ring = np.zeros((segments + 1, 3), dtype=np.float32)
    if axis == "z":
        ring[:, 0] = np.cos(angles) * radius
        ring[:, 1] = np.sin(angles) * radius
    else:
        ring[:, 1] = np.cos(angles) * radius
        ring[:, 2] = np.sin(angles) * radius
    ring += np.asarray(center, dtype=np.float32)
    starts = wp.array(ring[:-1], dtype=wp.vec3)
    ends = wp.array(ring[1:], dtype=wp.vec3)
    return starts, ends


def body_axis_world(body_q: np.ndarray, body: int, axis) -> np.ndarray:
    """Rotate a body-frame unit axis into the world frame from a ``body_q`` row."""
    q = wp.quat(*body_q[body, 3:7])
    return np.asarray(wp.quat_rotate(q, wp.vec3(*axis)), dtype=np.float64)


def assert_finite(*arrays) -> None:
    for array in arrays:
        values = array.numpy() if hasattr(array, "numpy") else np.asarray(array)
        if not np.isfinite(values).all():
            raise ValueError("non-finite values in simulation state")


@wp.kernel
def drive_revolute(dof: int, angle: float, rate: float, joint_q: wp.array[float], joint_qd: wp.array[float]):
    """Prescribe one revolute coordinate and rate, used for kinematic turntables."""
    joint_q[dof] = angle
    joint_qd[dof] = rate
