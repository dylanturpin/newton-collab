# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the rigid-body showreel examples.

Every scene drives its substeps through :class:`Stepper`, which captures the
collide-and-solve loop into one CUDA graph (re-captured whenever a solver
setting changes) and exposes a common solver panel: the solver itself
(FeatherPGS, VBD or MuJoCo), its iteration count, substeps, graph capture, and
the measured frame time. Each scene supplies per-solver settings through
``SOLVERS``; :func:`solver_configs` fills in the shared defaults.
"""

from __future__ import annotations

import math
import time

import numpy as np
import warp as wp

import newton

ROUTES = ("propagation-colored", "immediate")
SOLVERS = ("feather_pgs", "mujoco")
#: Solvers that run their own broad and narrow phase and ignore a Contacts buffer.
NATIVE_CONTACT_SOLVERS = frozenset({"mujoco"})
#: Name of each solver's iteration count, for the shared panel slider.
ITERATION_ARG = {"feather_pgs": "pgs_iterations", "mujoco": "iterations"}

# Shared per-solver defaults. A scene's SOLVERS table overrides any of these and
# adds "substeps"; everything else is passed to the solver constructor.
_DEFAULTS = {
    # friction_anchor_beta is the positional correction gain of the persistent friction
    # patches. At 1.0 a resting eighteen-level jenga tower leans 56 mm after forty
    # seconds at four iterations; at the solver default of 0.2 it leans 195 mm after
    # twenty and eventually topples. No scene pays anything for it.
    "feather_pgs": {
        "pgs_iterations": 4,
        "pgs_beta": 0.2,
        "angular_damping": 0.0,
        "friction_anchor_beta": 1.0,
        "shape_ke": 2.5e3,
        "shape_kd": 100.0,
    },
    # An elliptic cone matches Coulomb friction; the pyramidal default over-grips
    # along the cone's corners, which is visible as boxes that refuse to slide.
    "mujoco": {
        "cone": "elliptic",
        "iterations": 20,
        "ls_iterations": 10,
        "njmax": 4096,
        "nconmax": 2048,
        "shape_ke": 2.5e3,
        "shape_kd": 100.0,
    },
}


def solver_configs(scene: dict[str, dict]) -> dict[str, dict]:
    """Merge a scene's per-solver settings onto the shared defaults."""
    return {name: {**_DEFAULTS[name], **scene.get(name, {})} for name in SOLVERS}


def make_solver(model: newton.Model, *, solver: str = SOLVERS[0], route: str = ROUTES[0], **overrides):
    """Build one of the showreel's solvers with defaults suited to the model's device.

    FeatherPGS runs matrix-free on CUDA. ``"propagation-colored"`` places every
    contact row on graph-colored batches that solve thread-per-row, which is what
    a single large world needs; ``"immediate"`` is the serial per-world sweep.
    CPU falls back to the split mode, the only CPU-capable mode.
    """
    kwargs = {**_DEFAULTS[solver], **overrides}
    for key in ("substeps", "shape_ke", "shape_kd"):
        kwargs.pop(key, None)
    if solver == "mujoco":
        return newton.solvers.SolverMuJoCo(model, **kwargs)
    if model.device.is_cuda:
        kwargs["pgs_mode"] = "matrix_free"
        kwargs["articulated_contact_response"] = route
    else:
        kwargs["pgs_mode"] = "split"
    return newton.solvers.SolverFeatherPGS(model, **kwargs)


class Stepper:
    """Run a scene's substeps, through a CUDA graph when possible, and own the solver panel.

    The example provides ``substep()``, which clears forces, calls :meth:`collide`
    and :meth:`solve`, and nothing else that depends on the solver. The captured
    loop must leave the newest state in the buffer it read from, so an odd
    ``sim_substeps`` ends with a device copy back into that buffer.
    """

    def __init__(self, example, *, solver_overrides: dict | None = None, solver: str = SOLVERS[0]):
        self.example = example
        self.configs = solver_configs(solver_overrides or {})
        self.solver_name = solver
        self.route = ROUTES[0] if example.model.device.is_cuda else "split"
        self.use_graph = example.model.device.is_cuda
        self.graph = None
        self.frame_ms = 0.0
        self.error = ""
        self.rebuild()

    @property
    def config(self) -> dict:
        return self.configs[self.solver_name]

    @property
    def iterations(self) -> int:
        return int(self.config[ITERATION_ARG[self.solver_name]])

    @property
    def substeps(self) -> int:
        return int(self.config.get("substeps", 4))

    @property
    def native_contacts(self) -> bool:
        return self.solver_name in NATIVE_CONTACT_SOLVERS

    def invalidate(self):
        self.graph = None

    def rebuild(self):
        """Recreate the solver from the current configuration."""
        ex = self.example
        ex.sim_substeps = self.substeps
        ex.sim_dt = ex.frame_dt / ex.sim_substeps
        # Contact stiffness lives on the model and the solvers read it differently:
        # MuJoCo turns it into solref, FeatherPGS ignores it unless compliance is
        # enabled. Rewrite it so a switch takes the new value.
        for key, attr in (("shape_ke", "shape_material_ke"), ("shape_kd", "shape_material_kd")):
            value = self.config.get(key)
            array = getattr(ex.model, attr, None)
            if value is not None and array is not None:
                array.fill_(float(value))
        try:
            ex.solver = make_solver(ex.model, solver=self.solver_name, route=self.route, **self.config)
            self.error = ""
        except Exception as err:
            self.error = f"{self.solver_name}: {type(err).__name__}: {err}"[:200]
        self.invalidate()

    def set_solver(self, name: str):
        if name == self.solver_name:
            return
        self.solver_name = name
        self.rebuild()

    def set_route(self, route: str):
        if route == self.route:
            return
        self.route = route
        if self.solver_name == "feather_pgs":
            self.rebuild()

    def reset_scene(self):
        """Put the scene back to its built pose so a trigger can be tried again.

        The scene's ``on_reset`` re-arms whatever it scripts (a shot, a poke, the
        crane's release); everything else comes from the model's own defaults.
        """
        ex = self.example
        for state in (ex.state_0, ex.state_1):
            for name in ("body_q", "joint_q"):
                array, default = getattr(state, name, None), getattr(ex.model, name, None)
                if array is not None and default is not None:
                    wp.copy(array, default)
            for name in ("body_qd", "joint_qd", "body_f"):
                array = getattr(state, name, None)
                if array is not None:
                    array.zero_()
        ex.sim_time = 0.0
        if hasattr(ex, "on_reset"):
            ex.on_reset()
        self.notify_state_edit()

    def notify_state_edit(self):
        """Tell the solver the state was teleported, not integrated.

        A solver that keeps the previous body pose to difference velocities from
        otherwise reads an edit such as firing a cannonball as a displacement over
        one substep, which is a velocity of hundreds of metres per second.
        ``flags=0`` keeps the pose that was just authored and clears only history.
        """
        ex = self.example
        ex.solver.reset(ex.state_0, flags=0)

    def collide(self):
        """Refresh the contact buffer, unless the solver finds its own contacts."""
        ex = self.example
        if not self.native_contacts:
            ex.collision_pipeline.collide(ex.state_0, ex.contacts)

    def solve(self):
        """Advance one substep and swap the state pair."""
        ex = self.example
        contacts = None if self.native_contacts else ex.contacts
        ex.solver.step(ex.state_0, ex.state_1, ex.control, contacts, ex.sim_dt)
        ex.state_0, ex.state_1 = ex.state_1, ex.state_0

    def _capture(self):
        ex = self.example
        # Warm up once so every kernel module is loaded, then capture. FeatherPGS
        # double-buffers across streams and must record its memset-done events
        # inside the capture before the first captured step.
        for _ in range(ex.sim_substeps):
            ex.substep()
        wp.synchronize_device(ex.model.device)
        with wp.ScopedCapture(device=ex.model.device) as capture:
            if hasattr(ex.solver, "seed_double_buffer_events"):
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
        contacts = 0 if ex.contacts is None or self.native_contacts else int(ex.contacts.rigid_contact_count.numpy()[0])
        detail = "solver contacts" if self.native_contacts else f"{contacts} contacts"
        ui.text(f"{ex.model.body_count} bodies, {detail}, dt = 1/{round(1.0 / ex.sim_dt)} s")
        if self.error:
            ui.text(f"! {self.error}")
        index = SOLVERS.index(self.solver_name)
        changed, index = ui.combo("Solver", index, list(SOLVERS))
        if changed:
            self.set_solver(SOLVERS[index])
        changed, iterations = ui.slider_int("Iterations", self.iterations, 1, 64)
        if changed:
            self.config[ITERATION_ARG[self.solver_name]] = iterations
            if self.solver_name == "feather_pgs":
                ex.solver.pgs_iterations = iterations
                self.invalidate()
            else:
                # MuJoCo bakes the count into its model, so rebuild the solver.
                self.rebuild()
        if self.solver_name == "mujoco":
            for label, key, lo, hi in (("Line-search iters", "ls_iterations", 1, 50),):
                changed, value = ui.slider_int(label, int(self.config[key]), lo, hi)
                if changed:
                    self.config[key] = value
                    self.rebuild()
            changed, impratio = ui.slider_float("Friction impedance", float(self.config["impratio"]), 0.1, 100.0)
            if changed:
                self.config["impratio"] = impratio
                self.rebuild()
            cones = ["elliptic", "pyramidal"]
            index = cones.index(str(self.config["cone"]))
            changed, index = ui.combo("Friction cone", index, cones)
            if changed:
                self.config["cone"] = cones[index]
                self.rebuild()
        changed, ke = ui.slider_float("Contact stiffness", float(self.config["shape_ke"]), 1.0e2, 1.0e7)
        if changed:
            self.config["shape_ke"] = ke
            self.rebuild()
        changed, substeps = ui.slider_int("Substeps / frame", ex.sim_substeps, 1, 32)
        if changed:
            self.config["substeps"] = max(1, substeps)
            ex.sim_substeps = self.substeps
            ex.sim_dt = ex.frame_dt / ex.sim_substeps
            self.invalidate()
        if ex.model.device.is_cuda and self.solver_name == "feather_pgs":
            index = ROUTES.index(self.route) if self.route in ROUTES else 0
            changed, index = ui.combo("Contact route", index, list(ROUTES))
            if changed:
                self.set_route(ROUTES[index])
        if ex.model.device.is_cuda:
            changed, use_graph = ui.checkbox("CUDA graph capture", self.use_graph)
            if changed:
                self.use_graph = use_graph
                self.invalidate()
        if ui.button("Reset scene"):
            self.reset_scene()


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
