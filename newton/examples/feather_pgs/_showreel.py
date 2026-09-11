# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the FeatherPGS rigid-body showreel examples.

Every scene drives its substeps through :class:`Stepper`, which captures the
collide-and-solve loop into one CUDA graph (re-captured whenever a solver
setting changes) and exposes a common solver panel: iterations, substeps,
contact route, graph capture, and the measured frame time.
"""

from __future__ import annotations

import math
import time

import numpy as np
import warp as wp

import newton

ROUTES = ("propagation-colored", "immediate")


def make_solver(model: newton.Model, *, route: str = ROUTES[0], **overrides) -> newton.solvers.SolverFeatherPGS:
    """Build a FeatherPGS solver with showreel defaults for the model's device.

    On CUDA the matrix-free mode is used. ``"propagation-colored"`` places every
    contact row on graph-colored batches that solve thread-per-row, which is what
    a single large world needs; ``"immediate"`` is the serial per-world sweep.
    CPU falls back to the split mode, the only CPU-capable mode.
    """
    kwargs = {"pgs_iterations": 24, "pgs_beta": 0.2, "angular_damping": 0.0}
    if model.device.is_cuda:
        kwargs["pgs_mode"] = "matrix_free"
        kwargs["articulated_contact_response"] = route
    else:
        kwargs["pgs_mode"] = "split"
    kwargs.update(overrides)
    return newton.solvers.SolverFeatherPGS(model, **kwargs)


class Stepper:
    """Run a scene's substeps, through a CUDA graph when possible, and own the solver panel.

    The example provides ``substep()`` (clear forces, collide, solve, swap states).
    The captured loop must leave the newest state in the buffer it read from, so
    an odd ``sim_substeps`` ends with a device copy back into that buffer.
    """

    def __init__(self, example, *, solver_overrides: dict | None = None):
        self.example = example
        self.solver_overrides = dict(solver_overrides or {})
        self.route = ROUTES[0] if example.model.device.is_cuda else "split"
        self.use_graph = example.model.device.is_cuda
        self.graph = None
        self.frame_ms = 0.0
        self.iterations = int(example.solver.pgs_iterations)

    def invalidate(self):
        self.graph = None

    def set_route(self, route: str):
        if route == self.route:
            return
        self.route = route
        ex = self.example
        ex.solver = make_solver(ex.model, route=route, **{**self.solver_overrides, "pgs_iterations": self.iterations})
        self.invalidate()

    def _capture(self):
        ex = self.example
        # Warm up once so every kernel module is loaded, then capture. FeatherPGS
        # double-buffers across streams and must record its memset-done events
        # inside the capture before the first captured step.
        for _ in range(ex.sim_substeps):
            ex.substep()
        wp.synchronize_device(ex.model.device)
        with wp.ScopedCapture(device=ex.model.device) as capture:
            ex.solver.seed_double_buffer_events()
            for _ in range(ex.sim_substeps):
                ex.substep()
            if ex.sim_substeps % 2:
                # The newest state sits in the other buffer: copy it back so every
                # replay reads from and delivers into the same buffer.
                for name in ("body_q", "body_qd", "joint_q", "joint_qd"):
                    src, dst = getattr(ex.state_0, name), getattr(ex.state_1, name)
                    if src is not None:
                        wp.copy(dst, src)
                ex.state_0, ex.state_1 = ex.state_1, ex.state_0
        self.graph = capture.graph

    def step(self):
        ex = self.example
        start = time.perf_counter()
        if self.use_graph:
            if self.graph is None:
                self._capture()
            wp.capture_launch(self.graph)
        else:
            for _ in range(ex.sim_substeps):
                ex.substep()
        wp.synchronize_device(ex.model.device)
        ms = (time.perf_counter() - start) * 1.0e3
        self.frame_ms = ms if self.frame_ms == 0.0 else 0.9 * self.frame_ms + 0.1 * ms
        ex.sim_time += ex.frame_dt

    def gui(self, ui):
        """Draw the shared solver panel. Call from the example's ``gui``."""
        ex = self.example
        ui.separator()
        ui.text("Solver")
        ui.text(f"{self.frame_ms:5.2f} ms / frame  ({(1000.0 / self.frame_ms) if self.frame_ms else 0.0:5.1f} fps)")
        contacts = int(ex.contacts.rigid_contact_count.numpy()[0]) if ex.contacts is not None else 0
        ui.text(f"{ex.model.body_count} bodies, {contacts} contacts, dt = 1/{round(1.0 / ex.sim_dt)} s")
        changed, iterations = ui.slider_int("PGS iterations", self.iterations, 2, 128)
        if changed:
            self.iterations = iterations
            ex.solver.pgs_iterations = iterations
            self.invalidate()
        changed, substeps = ui.slider_int("Substeps / frame", ex.sim_substeps, 1, 32)
        if changed:
            ex.sim_substeps = max(1, substeps)
            ex.sim_dt = ex.frame_dt / ex.sim_substeps
            self.invalidate()
        if ex.model.device.is_cuda:
            index = ROUTES.index(self.route) if self.route in ROUTES else 0
            changed, index = ui.combo("Contact route", index, list(ROUTES))
            if changed:
                self.set_route(ROUTES[index])
            changed, use_graph = ui.checkbox("CUDA graph capture", self.use_graph)
            if changed:
                self.use_graph = use_graph
                self.invalidate()


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
    return wp.array(ring[:-1], dtype=wp.vec3), wp.array(ring[1:], dtype=wp.vec3)


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


@wp.kernel
def push_body(
    params: wp.array[float],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_f: wp.array[wp.spatial_vector],
):
    """Finger-like push, graph-safe: ``params`` = [body, dir xyz, force, speed cap, travel cap, start xyz].

    The force is applied while the body is below the speed cap along the push
    direction and has travelled less than the cap from its start; a negative
    body index disables the push.
    """
    body = int(params[0])
    if body < 0:
        return
    direction = wp.vec3(params[1], params[2], params[3])
    start = wp.vec3(params[7], params[8], params[9])
    travel = wp.dot(wp.transform_get_translation(body_q[body]) - start, direction)
    if travel >= params[6]:
        return
    if wp.dot(wp.spatial_top(body_qd[body]), direction) >= params[5]:
        return
    body_f[body] = wp.spatial_vector(direction * params[4], wp.vec3(0.0))
