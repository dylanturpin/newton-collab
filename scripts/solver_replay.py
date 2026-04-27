#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""FeatherPGS solver snapshot / replay + frame-selection harness.

Tool for the FPGS Friction Modes 8/13 slice (issue #14).  The plotting
step (``metric vs gs_iterations``) is separate (10/13); this script only
produces the raw residual traces and a ranked list of "hard" frames.

Two modes:

* ``--mode replay`` (default): simulate forward ``--step`` warmup steps
  on a chosen scenario with ``friction_mode="current"``, snapshot the
  state just before the target step, then replay the single target step
  for each ``--friction-mode`` value across a sweep of
  ``--gs-iterations`` values.  For every replay we capture the
  per-iteration six-residual vector that the matrix-free PGS debug path
  populates when ``pgs_debug=True``
  (``(r_compl, r_cone, r_gap, r_ds_compl, r_ds_dual, r_mdp_dir)``).
* ``--mode select-frame``: scan a range of steps, run a fixed-GS probe
  at each, and emit a ranked list of "hard" candidate frames ordered by
  post-solve ``r_compl`` (highest first), with the total contact row
  count and a sliding-row heuristic as secondary / tertiary metrics.

Determinism:

* Replay uses bit-identical snapshot state (NumPy round-trip).
* Each friction-mode replay constructs a fresh :class:`SolverFeatherPGS`
  so warmstart / internal counters cannot leak between runs.
* Warp CUDA kernels are deterministic on a single GPU given identical
  inputs; bitwise identity across runs is **not** guaranteed but the
  residual traces match closely enough to compare convergence curves.
  A ``--self-check`` flag reruns the replay twice and asserts
  ``max(rel_delta) < 1e-2`` on the residual stack before writing output.

Dependencies: Warp, NumPy, stdlib only.  The optional PNG snapshot uses
the existing Newton GL renderer (``newton.viewer.ViewerGL``) in headless
mode; if the renderer cannot be initialised (for example, no GL/EGL
libraries in the environment) the tool logs a warning and skips the
PNG without failing.

Example::

    python newton-collab/scripts/solver_replay.py \\
        --scenario sliding_cube --step 20 \\
        --friction-modes current,bisection,bisection_desaxce \\
        --gs-iterations 1,2,5,10,20,50,100 \\
        --out /tmp/replay/ --self-check
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Allow running directly from the repo without an editable install: add the
# ``newton-collab/`` package root (the parent of this ``scripts/`` directory)
# to ``sys.path`` before importing the ``newton`` package.  This keeps the
# documented invocation ``python scripts/solver_replay.py ...`` working when
# invoked from ``newton-collab/``.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
import warp as wp  # noqa: E402

import newton  # noqa: E402
import newton.utils  # noqa: E402
from newton._src.solvers import SolverFeatherPGS  # noqa: E402

# =============================================================================
# Scenario builders
# =============================================================================
#
# Mirror the scene builders in the 5/13 and 6/13 friction-mode test
# modules so the replay tool exercises the same contact problems the
# rest of the FPGS Friction Modes series already validates.  Keeping
# them inline here avoids importing test modules from a scripts/ tool
# (the test modules depend on ``unittest`` fixtures we don't want to
# pull into the tool).

G1_ASSET_FOLDER = "unitree_g1"
G1_USD_RELPATH = ("usd", "g1_isaac.usd")

H1_ASSET_FOLDER = "unitree_h1"
H1_USD_RELPATH = ("usd", "h1_minimal.usda")


def _build_sliding_cube_model(
    device: wp.context.Device,
    initial_velocity: tuple[float, float, float] = (2.0, 0.0, 0.0),
) -> newton.Model:
    """A free-body cube resting on the ground with tangential initial velocity.

    Matches the scene in
    ``newton.tests.test_feather_pgs_friction_mode_bisection_desaxce``.
    Always available (no downloaded assets).
    """
    builder = newton.ModelBuilder()
    builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(limit_ke=0.0, limit_kd=0.0, friction=0.0)
    builder.default_shape_cfg.ke = 5.0e4
    builder.default_shape_cfg.kd = 5.0e2
    builder.default_shape_cfg.kf = 1.0e3
    builder.default_shape_cfg.mu = 0.3

    body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()),
        mass=1.0,
        inertia=wp.mat33(0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1),
    )
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    joint_qd_np = model.joint_qd.numpy().copy()
    if joint_qd_np.size >= 3:
        joint_qd_np[0] = float(initial_velocity[0])
        joint_qd_np[1] = float(initial_velocity[1])
        joint_qd_np[2] = float(initial_velocity[2])
        model.joint_qd.assign(joint_qd_np)
    return model


def _try_build_g1_flat_model(device: wp.context.Device) -> newton.Model | None:
    """Build the ``g1_flat`` scenario, or return ``None`` when the USD asset is missing.

    Mirrors the ``g1_flat`` scenario in ``newton.tools.solver_benchmark``.
    """
    try:
        asset_path = newton.utils.download_asset(G1_ASSET_FOLDER)
    except Exception:
        return None
    usd_path = asset_path
    for part in G1_USD_RELPATH:
        usd_path = usd_path / part
    if not usd_path.exists():
        return None

    g1 = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(g1)
    g1.default_joint_cfg = newton.ModelBuilder.JointDofConfig(limit_ke=0.0, limit_kd=0.0, friction=0.0)
    g1.default_shape_cfg.ke = 5.0e4
    g1.default_shape_cfg.kd = 5.0e2
    g1.default_shape_cfg.kf = 1.0e3
    g1.default_shape_cfg.mu = 0.75

    try:
        g1.add_usd(
            str(usd_path),
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.8)),
            collapse_fixed_joints=True,
            enable_self_collisions=False,
            hide_collision_shapes=True,
            skip_mesh_approximation=True,
        )
    except Exception:
        return None

    for i in range(6, g1.joint_dof_count):
        g1.joint_target_ke[i] = 1000.0
        g1.joint_target_kd[i] = 5.0
    try:
        g1.approximate_meshes("bounding_box")
    except Exception:
        pass

    builder = newton.ModelBuilder()
    builder.replicate(g1, 1)
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    model.shape_margin.fill_(0.001)
    return model


def _try_build_h1_tabletop_model(device: wp.context.Device) -> newton.Model | None:
    """Build the ``h1_tabletop`` scenario, or return ``None`` when the USD asset is missing.

    Mirrors the ``h1_tabletop`` scenario in ``newton.tools.solver_benchmark``.
    """
    try:
        asset_path = newton.utils.download_asset(H1_ASSET_FOLDER)
    except Exception:
        return None
    usd_path = asset_path
    for part in H1_USD_RELPATH:
        usd_path = usd_path / part
    if not usd_path.exists():
        return None

    h1 = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(h1)
    h1.default_joint_cfg = newton.ModelBuilder.JointDofConfig(limit_ke=1.0e3, limit_kd=1.0e1, friction=1e-5)
    h1.default_shape_cfg.ke = 5.0e4
    h1.default_shape_cfg.kd = 5.0e2
    h1.default_shape_cfg.kf = 1.0e3
    h1.default_shape_cfg.mu = 0.75

    try:
        h1.add_usd(
            str(usd_path),
            ignore_paths=["/GroundPlane"],
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            hide_collision_shapes=True,
        )
    except Exception:
        return None
    try:
        h1.approximate_meshes("bounding_box")
    except Exception:
        pass
    for i in range(h1.joint_dof_count):
        h1.joint_target_ke[i] = 150
        h1.joint_target_kd[i] = 5

    tabletop_objects = [
        {"type": "sphere", "pos": (0.6, -0.1, 1.06), "radius": 0.05},
        {"type": "sphere", "pos": (0.675, -0.025, 1.06), "radius": 0.05},
        {"type": "sphere", "pos": (0.75, 0.05, 1.06), "radius": 0.05},
        {"type": "sphere", "pos": (0.825, 0.125, 1.06), "radius": 0.05},
        {"type": "capsule", "pos": (0.6, -0.15, 1.16), "quat": (1, 0.5, 0, 0), "radius": 0.04, "half_height": 0.04},
        {"type": "capsule", "pos": (0.8, -0.15, 1.16), "quat": (1, 0.5, 0, 0), "radius": 0.04, "half_height": 0.04},
        {"type": "sphere", "pos": (0.7, -0.15, 1.41), "radius": 0.04},
        {"type": "sphere", "pos": (0.7, -0.05, 1.41), "radius": 0.04},
        {"type": "sphere", "pos": (0.7, 0.05, 1.41), "radius": 0.04},
        {"type": "sphere", "pos": (0.7, 0.15, 1.41), "radius": 0.04},
        {"type": "capsule", "pos": (0.55, 0, 1.2), "quat": (0, 0.5, 0, 1), "radius": 0.04, "half_height": 0.04},
        {"type": "capsule", "pos": (0.85, 0, 1.2), "quat": (0, 0.5, 0, 1), "radius": 0.04, "half_height": 0.04},
        {"type": "box", "pos": (0.65, 0, 1.19), "half_size": 0.03},
        {"type": "box", "pos": (0.65, 0, 1.25), "half_size": 0.03},
        {"type": "box", "pos": (0.65, 0, 1.31), "half_size": 0.03},
        {"type": "box", "pos": (0.75, 0, 1.19), "half_size": 0.03},
        {"type": "box", "pos": (0.75, 0, 1.25), "half_size": 0.03},
        {"type": "box", "pos": (0.75, 0, 1.31), "half_size": 0.03},
    ]

    obj_shape_cfg = newton.ModelBuilder.ShapeConfig(mu=0.5)

    builder = newton.ModelBuilder()
    builder.begin_world()
    builder.add_builder(h1)

    for obj in tabletop_objects:
        pos = obj["pos"]
        quat = obj.get("quat", (1, 0, 0, 0))
        qlen = (quat[0] ** 2 + quat[1] ** 2 + quat[2] ** 2 + quat[3] ** 2) ** 0.5
        quat = (quat[0] / qlen, quat[1] / qlen, quat[2] / qlen, quat[3] / qlen)
        body_idx = builder.add_body(
            xform=wp.transform(
                wp.vec3(pos[0], pos[1], pos[2]),
                wp.quat(quat[0], quat[1], quat[2], quat[3]),
            ),
        )
        if obj["type"] == "sphere":
            builder.add_shape_sphere(body_idx, radius=obj["radius"], cfg=obj_shape_cfg)
        elif obj["type"] == "box":
            hs = obj["half_size"]
            builder.add_shape_box(body_idx, hx=hs, hy=hs, hz=hs, cfg=obj_shape_cfg)
        elif obj["type"] == "capsule":
            builder.add_shape_capsule(
                body_idx,
                radius=obj["radius"],
                half_height=obj["half_height"],
                cfg=obj_shape_cfg,
            )

    builder.end_world()

    table_shape_cfg = newton.ModelBuilder.ShapeConfig(mu=0.5)
    builder.add_shape_box(
        -1, xform=wp.transform((0.8, 0.0, 0.75), wp.quat_identity()), hx=0.5, hy=1.0, hz=0.01, cfg=table_shape_cfg
    )
    builder.add_shape_box(
        -1, xform=wp.transform((0.9, 0.0, 0.86), wp.quat_identity()), hx=0.01, hy=0.21, hz=0.1, cfg=table_shape_cfg
    )
    builder.add_shape_box(
        -1, xform=wp.transform((0.5, 0.0, 0.86), wp.quat_identity()), hx=0.01, hy=0.21, hz=0.1, cfg=table_shape_cfg
    )
    builder.add_shape_box(
        -1, xform=wp.transform((0.7, -0.2, 0.86), wp.quat_identity()), hx=0.21, hy=0.01, hz=0.1, cfg=table_shape_cfg
    )
    builder.add_shape_box(
        -1, xform=wp.transform((0.7, 0.2, 0.86), wp.quat_identity()), hx=0.21, hy=0.01, hz=0.1, cfg=table_shape_cfg
    )

    builder.add_ground_plane()
    model = builder.finalize(device=device)
    model.shape_margin.fill_(0.001)
    return model


# Scenario metadata: (substeps, pgs_iterations_warmup, dense_max_constraints,
# builder_callable).  ``pgs_iterations_warmup`` is only used for the warmup
# phase (friction_mode="current"); the replay sweeps its own iteration
# budget.  These defaults mirror ``newton.tools.solver_benchmark.SCENARIOS``.
_SCENARIO_DEFAULTS: dict[str, dict] = {
    "sliding_cube": {
        "substeps": 1,
        "pgs_iterations_warmup": 20,
        "dense_max_constraints": 128,
        "builder": _build_sliding_cube_model,
    },
    "g1_flat": {
        "substeps": 2,
        "pgs_iterations_warmup": 4,
        "dense_max_constraints": 32,
        "builder": _try_build_g1_flat_model,
    },
    "h1_tabletop": {
        "substeps": 8,
        "pgs_iterations_warmup": 8,
        "dense_max_constraints": 128,
        "builder": _try_build_h1_tabletop_model,
    },
}


# =============================================================================
# Friction mode CLI helpers
# =============================================================================

_VALID_FRICTION_MODES = ("current", "bisection", "bisection_desaxce", "coulomb_newton")

# Channel order that matches ``pgs_ncp_residuals_diagnostic_velocity``.
_RESIDUAL_CHANNELS: tuple[str, ...] = (
    "r_compl",
    "r_cone",
    "r_gap",
    "r_ds_compl",
    "r_ds_dual",
    "r_mdp_dir",
)


def _parse_csv_str_list(value: str, valid: tuple[str, ...] | None = None) -> list[str]:
    """Parse a comma-separated CLI list into a de-duplicated ordered list."""
    items: list[str] = []
    seen: set[str] = set()
    for raw in value.split(","):
        tok = raw.strip()
        if not tok:
            continue
        if tok in seen:
            continue
        if valid is not None and tok not in valid:
            raise argparse.ArgumentTypeError(f"invalid value {tok!r}; expected one of {list(valid)}")
        seen.add(tok)
        items.append(tok)
    if not items:
        raise argparse.ArgumentTypeError("empty list")
    return items


def _parse_csv_int_list(value: str) -> list[int]:
    """Parse a comma-separated CLI list into a sorted unique positive-int list."""
    vals: set[int] = set()
    for raw in value.split(","):
        tok = raw.strip()
        if not tok:
            continue
        try:
            v = int(tok)
        except ValueError as exc:  # pragma: no cover - argparse reports nicely
            raise argparse.ArgumentTypeError(f"not an int: {tok!r}") from exc
        if v <= 0:
            raise argparse.ArgumentTypeError(f"must be >= 1, got {v}")
        vals.add(v)
    if not vals:
        raise argparse.ArgumentTypeError("empty list")
    return sorted(vals)


def _parse_step_range(value: str) -> tuple[int, int]:
    """Parse ``"a..b"`` into ``(a, b)`` with ``0 <= a < b``."""
    if ".." not in value:
        raise argparse.ArgumentTypeError("step range must be LO..HI (e.g. 50..200)")
    lo_s, hi_s = value.split("..", 1)
    try:
        lo = int(lo_s)
        hi = int(hi_s)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"step range not integers: {value!r}") from exc
    if lo < 0 or hi <= lo:
        raise argparse.ArgumentTypeError(f"step range must satisfy 0 <= LO < HI, got {value!r}")
    return lo, hi


# =============================================================================
# Snapshot
# =============================================================================


# The state attributes we snapshot.  ``joint_q`` / ``joint_qd`` define
# the articulation configuration, ``body_q`` / ``body_qd`` are recomputed
# by forward kinematics inside the solver but we also copy them so that
# ``model.collide(state_0)`` sees the correct contact geometry before
# the first step.
_SNAPSHOT_STATE_ATTRS: tuple[str, ...] = ("joint_q", "joint_qd", "body_q", "body_qd")

# Control attributes we snapshot (all optional on :class:`newton.Control`).
_SNAPSHOT_CONTROL_ATTRS: tuple[str, ...] = (
    "joint_f",
    "joint_target_pos",
    "joint_target_vel",
    "joint_act",
    "tri_activations",
    "tet_activations",
    "muscle_activations",
)

# Contact attributes we snapshot off the :class:`newton.sim.Contacts` object.
# We capture both rigid and soft contact buffers plus the two counters so a
# replay can be performed either from the captured contacts or by
# re-running ``model.collide(state_0)`` (which is deterministic for the
# same state input).
_SNAPSHOT_CONTACT_ATTRS: tuple[str, ...] = (
    "rigid_contact_count",
    "rigid_contact_point_id",
    "rigid_contact_shape0",
    "rigid_contact_shape1",
    "rigid_contact_point0",
    "rigid_contact_point1",
    "rigid_contact_offset0",
    "rigid_contact_offset1",
    "rigid_contact_normal",
    "rigid_contact_margin0",
    "rigid_contact_margin1",
    "rigid_contact_force",
    "rigid_contact_stiffness",
    "rigid_contact_damping",
    "rigid_contact_friction",
    "soft_contact_count",
    "soft_contact_particle",
    "soft_contact_shape",
    "soft_contact_body_pos",
    "soft_contact_body_vel",
    "soft_contact_normal",
)


@dataclass
class _Snapshot:
    """Off-device snapshot of the bits a FeatherPGS replay needs.

    The issue calls for snapshotting ``(state, control, contacts,
    rng_state, Model)``.  The Warp :class:`~newton.Model` is re-used
    across replays in a single invocation (it's a heavy build for G1/H1),
    so it is kept live on :attr:`model_ref` rather than copied; every
    other component is copied off-device so that the replay is a
    round-trip through NumPy and therefore bit-identical on re-entry.

    ``rng_state`` is carried for API completeness: the FeatherPGS solver
    and the Newton collision pipeline are deterministic and do not pull
    from any RNG, so this field is always ``None`` today.  It exists so
    future stochastic paths (e.g. randomised warmstart, probabilistic
    contact filtering) can be snapshotted without a schema change.
    """

    scenario: str
    target_step: int
    substeps: int
    sim_dt: float
    state_arrays: dict[str, np.ndarray] = field(default_factory=dict)
    control_arrays: dict[str, np.ndarray] = field(default_factory=dict)
    contact_arrays: dict[str, np.ndarray] = field(default_factory=dict)
    rng_state: object | None = None
    model_ref: newton.Model | None = None

    def populate_state(self, state: newton.State) -> None:
        """Write snapshot state arrays into a freshly allocated state object."""
        for name, value in self.state_arrays.items():
            arr = getattr(state, name, None)
            if arr is None:
                continue
            arr.assign(value)

    def populate_control(self, control: newton.Control) -> None:
        """Write snapshot control arrays into a freshly allocated control object."""
        for name, value in self.control_arrays.items():
            arr = getattr(control, name, None)
            if arr is None:
                continue
            arr.assign(value)

    def populate_contacts(self, contacts) -> None:
        """Write snapshot contact buffers into a :class:`newton.sim.Contacts` object.

        Useful when the caller wants to replay without re-running
        collision detection; the standard replay path instead calls
        ``model.collide(state_0)`` (deterministic for identical state)
        and therefore skips this method.
        """
        for name, value in self.contact_arrays.items():
            arr = getattr(contacts, name, None)
            if arr is None:
                continue
            arr.assign(value)


def _snapshot_arrays_from(obj, attrs: tuple[str, ...]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for attr in attrs:
        val = getattr(obj, attr, None)
        if val is None:
            continue
        if not hasattr(val, "numpy"):
            continue
        arrays[attr] = np.ascontiguousarray(val.numpy().copy())
    return arrays


def _snapshot_state(
    state: newton.State,
    *,
    control: newton.Control | None,
    contacts: object | None,
    model: newton.Model | None,
    scenario: str,
    target_step: int,
    substeps: int,
    sim_dt: float,
) -> _Snapshot:
    """Capture state + control + contacts + rng_state + Model reference.

    Args:
        state: Simulation state just before ``target_step``.
        control: Per-frame control inputs driving the solver at this step.
        contacts: The :class:`newton.sim.Contacts` object that was fed to
            the solver for this step, or ``None`` if only state is known.
        model: The live :class:`newton.Model` to carry alongside the
            off-device buffers (see :attr:`_Snapshot.model_ref`).
        scenario: Scenario name for provenance.
        target_step: Step index the snapshot was taken at.
        substeps: Number of substeps per outer frame.
        sim_dt: Outer-frame ``dt`` in seconds.
    """
    state_arrays = _snapshot_arrays_from(state, _SNAPSHOT_STATE_ATTRS)
    control_arrays = _snapshot_arrays_from(control, _SNAPSHOT_CONTROL_ATTRS) if control is not None else {}
    contact_arrays = _snapshot_arrays_from(contacts, _SNAPSHOT_CONTACT_ATTRS) if contacts is not None else {}
    return _Snapshot(
        scenario=scenario,
        target_step=target_step,
        substeps=substeps,
        sim_dt=sim_dt,
        state_arrays=state_arrays,
        control_arrays=control_arrays,
        contact_arrays=contact_arrays,
        # Newton's collision pipeline + FeatherPGS have no stochastic
        # paths today; recorded as ``None`` rather than elided so the
        # schema stays forward-compatible with issue #14's contract.
        rng_state=None,
        model_ref=model,
    )


# =============================================================================
# Core: warmup + replay
# =============================================================================


def _build_model_or_raise(scenario: str, device: wp.context.Device) -> newton.Model:
    cfg = _SCENARIO_DEFAULTS[scenario]
    model = cfg["builder"](device)
    if model is None:
        raise SystemExit(
            f"scenario {scenario!r} is USD-asset gated and the asset is not available "
            f"locally; skipping.  Download the asset with "
            f"``python -c \"import newton.utils; newton.utils.download_asset('{G1_ASSET_FOLDER if scenario == 'g1_flat' else H1_ASSET_FOLDER}')\"``."
        )
    return model


def _warmup_and_snapshot(
    *,
    scenario: str,
    warmup_steps: int,
    sim_dt: float,
    device: wp.context.Device,
) -> tuple[newton.Model, _Snapshot]:
    """Build the scenario, step it forward ``warmup_steps`` times, and snapshot the pre-step state."""
    cfg = _SCENARIO_DEFAULTS[scenario]
    substeps = int(cfg["substeps"])
    pgs_iterations_warmup = int(cfg["pgs_iterations_warmup"])
    dense_max_constraints = int(cfg["dense_max_constraints"])

    model = _build_model_or_raise(scenario, device)
    solver = SolverFeatherPGS(
        model,
        pgs_mode="matrix_free",
        friction_mode="current",
        pgs_iterations=pgs_iterations_warmup,
        pgs_debug=False,
        dense_max_constraints=dense_max_constraints,
    )
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    sub_dt = sim_dt / float(substeps)
    contacts = None
    for _step in range(warmup_steps):
        contacts = model.collide(state_0)
        for _sub in range(substeps):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, contacts, sub_dt)
            state_0, state_1 = state_1, state_0

    # Take a pre-step snapshot of everything needed to replay the next
    # frame deterministically: state, control, contacts (captured from
    # the final warmup call to ``model.collide``), an ``rng_state``
    # placeholder, and a reference to the live Model.  See
    # :class:`_Snapshot`.
    final_contacts = model.collide(state_0)
    snapshot = _snapshot_state(
        state_0,
        control=control,
        contacts=final_contacts,
        model=model,
        scenario=scenario,
        target_step=warmup_steps,
        substeps=substeps,
        sim_dt=sim_dt,
    )
    return model, snapshot


def _build_replay_solver(
    *,
    model: newton.Model,
    friction_mode: str,
    pgs_iterations: int,
    dense_max_constraints: int,
) -> SolverFeatherPGS:
    """Construct a fresh matrix-free SolverFeatherPGS tuned for residual capture."""
    return SolverFeatherPGS(
        model,
        pgs_mode="matrix_free",
        friction_mode=friction_mode,
        pgs_iterations=pgs_iterations,
        pgs_debug=True,
        pgs_warmstart=False,
        dense_max_constraints=dense_max_constraints,
    )


def _replay_single_mode(
    *,
    model: newton.Model,
    snapshot: _Snapshot,
    friction_mode: str,
    gs_sweep: list[int],
    dense_max_constraints: int,
) -> dict[str, object]:
    """Run the replay for one ``friction_mode`` across every value in ``gs_sweep``.

    Returns a JSON-serializable dict ``{"status": "...", "gs_sweeps": {...}}``.
    """
    # ``coulomb_newton`` currently raises NotImplementedError at construction
    # time.  Treat that as an explicit "skip" rather than a hard error so the
    # CLI example in the issue (which lists all four modes) still completes
    # with exit 0.
    try:
        _probe = _build_replay_solver(
            model=model,
            friction_mode=friction_mode,
            pgs_iterations=max(gs_sweep),
            dense_max_constraints=dense_max_constraints,
        )
    except NotImplementedError as exc:
        return {"status": "unsupported", "reason": str(exc), "gs_sweeps": {}}
    except ValueError as exc:
        return {"status": "invalid", "reason": str(exc), "gs_sweeps": {}}

    solver = _probe  # reuse: we change pgs_iterations and reset logs per sweep.
    sub_dt = snapshot.sim_dt / float(snapshot.substeps)
    substeps = snapshot.substeps

    gs_sweeps: dict[str, dict[str, object]] = {}
    for gs in gs_sweep:
        solver.pgs_iterations = int(gs)
        solver.reset_diagnostic_logs()

        # Fresh State/Control objects so the replay starts from an
        # identical kinematic + control configuration each time.  The
        # snapshot round-trip through NumPy guarantees bit-identical
        # restoration; Contacts are regenerated via ``model.collide``
        # (deterministic for the same state input) once per outer
        # frame and then reused across substeps — matching the repo's
        # canonical stepping pattern (see ``newton/tools/solver_benchmark.py``
        # and the friction-mode tests).  Calling ``collide`` inside the
        # substep loop would diverge from that semantic and skew the
        # residual traces for multi-substep scenarios (``g1_flat``,
        # ``h1_tabletop``).
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        snapshot.populate_state(state_0)
        snapshot.populate_control(control)

        # One full frame's worth of substeps, same as warmup: collide
        # once, then reuse the ``contacts`` object across substeps.
        contacts = model.collide(state_0)
        for _sub in range(substeps):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, contacts, sub_dt)
            state_0, state_1 = state_1, state_0

        # Capture the residual log from the *final* substep (most
        # scenarios use 1 or 2 substeps per frame).  Shape:
        # [gs, world_count, 6] -> serialized as plain lists.
        if solver._pgs_ncp_residual_log:
            ncp_stack = np.asarray(solver._pgs_ncp_residual_log[-1])
            residuals = ncp_stack.tolist()
        else:
            residuals = []

        gs_sweeps[str(gs)] = {
            "gs_iterations": int(gs),
            "channels": list(_RESIDUAL_CHANNELS),
            # residuals shape: [gs, world_count, 6] (or empty)
            "residuals": residuals,
        }

    return {"status": "ok", "gs_sweeps": gs_sweeps}


def _run_replay(
    *,
    scenario: str,
    target_step: int,
    friction_modes: list[str],
    gs_sweep: list[int],
    sim_dt: float,
    out_dir: Path,
    self_check: bool,
    device: wp.context.Device,
) -> dict[str, object]:
    """High-level ``--mode replay`` entry point; writes output files and returns the summary."""
    model, snapshot = _warmup_and_snapshot(scenario=scenario, warmup_steps=target_step, sim_dt=sim_dt, device=device)
    dense_max_constraints = int(_SCENARIO_DEFAULTS[scenario]["dense_max_constraints"])

    per_mode: dict[str, dict[str, object]] = {}
    for mode in friction_modes:
        per_mode[mode] = _replay_single_mode(
            model=model,
            snapshot=snapshot,
            friction_mode=mode,
            gs_sweep=gs_sweep,
            dense_max_constraints=dense_max_constraints,
        )

    # 2x4 contact-mix breakdown for the snapshot frame.  Captured once
    # from a probe solver with ``friction_mode="current"`` + a high
    # gs-iteration budget so the sticking-vs-sliding split is stable.
    # Consumed by ``scripts/plot_residuals.py`` to render the
    # ``Contact mix`` table the 10/13 slice asks for.
    contact_mix = _capture_contact_mix(
        model=model,
        snapshot=snapshot,
        gs_probe=max(gs_sweep),
        dense_max_constraints=dense_max_constraints,
    )

    summary: dict[str, object] = {
        "schema_version": "1.2",
        "tool": "solver_replay",
        "mode": "replay",
        "scenario": scenario,
        "target_step": int(target_step),
        "substeps": int(snapshot.substeps),
        "sim_dt": float(snapshot.sim_dt),
        "dense_max_constraints": int(dense_max_constraints),
        "friction_modes_requested": list(friction_modes),
        "gs_iterations": list(gs_sweep),
        "channels": list(_RESIDUAL_CHANNELS),
        "friction_modes": per_mode,
        "contact_mix": contact_mix,
        # Record what the snapshot carries so downstream tooling can
        # verify the ``(state, control, contacts, rng_state, Model)``
        # contract from issue #14.
        "snapshot_contents": {
            "state_arrays": sorted(snapshot.state_arrays),
            "control_arrays": sorted(snapshot.control_arrays),
            "contact_arrays": sorted(snapshot.contact_arrays),
            "rng_state": snapshot.rng_state,
            "model": snapshot.scenario,
        },
    }

    if self_check:
        # Re-run the whole replay once and compare residuals channel-wise.
        model2, snapshot2 = _warmup_and_snapshot(
            scenario=scenario, warmup_steps=target_step, sim_dt=sim_dt, device=device
        )
        per_mode2: dict[str, dict[str, object]] = {}
        for mode in friction_modes:
            per_mode2[mode] = _replay_single_mode(
                model=model2,
                snapshot=snapshot2,
                friction_mode=mode,
                gs_sweep=gs_sweep,
                dense_max_constraints=dense_max_constraints,
            )
        _check_two_run_stability(per_mode, per_mode2, tolerance=1.0e-2)
        summary["self_check"] = {"passed": True, "tolerance": 1.0e-2}

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "replay.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)

    # Optional PNG (best-effort; logs + skips on any error).
    png_path = out_dir / "snapshot.png"
    _try_save_snapshot_png(snapshot=snapshot, model=model, out_path=png_path)

    return summary


# =============================================================================
# Contact-mix breakdown (for 10/13 report tables)
# =============================================================================


# Column keys used both in-file and by scripts/plot_residuals.py (10/13).
_CONTACT_MIX_COLUMNS: tuple[str, ...] = (
    "normal_only",
    "sticking_friction",
    "sliding_friction",
    "joint_limit",
)

# Row keys used both in-file and by the downstream markdown renderer.
_CONTACT_MIX_ROWS: tuple[str, ...] = ("articulated", "free_rigid")


def _classify_path_rows(
    *,
    impulses_np: np.ndarray,
    row_type_np: np.ndarray,
    row_parent_np: np.ndarray,
    row_mu_np: np.ndarray,
    count_np: np.ndarray,
    cone_tolerance: float = 0.05,
) -> dict[str, int]:
    """Return a {normal_only, sticking_friction, sliding_friction, joint_limit} count dict.

    For each world we scan all active rows (``slot < count``), identify
    normal contact rows (``row_type == 0``) and their two paired friction
    rows (``row_type == 2`` with ``row_parent == normal_slot``).  A
    contact with no friction rows counts as ``normal_only``; contacts
    whose paired friction magnitude is at (or within ``cone_tolerance``
    of) the ``mu * lambda_n`` Coulomb bound count as
    ``sliding_friction``; otherwise ``sticking_friction``.
    ``row_type == 3`` rows are counted separately as ``joint_limit``
    regardless of their friction status.
    """
    buckets = dict.fromkeys(_CONTACT_MIX_COLUMNS, 0)
    worlds = impulses_np.shape[0]
    for world in range(worlds):
        n_rows = int(count_np[world])
        if n_rows <= 0:
            continue
        row_type = row_type_np[world, :n_rows]
        row_parent = row_parent_np[world, :n_rows]
        row_mu = row_mu_np[world, :n_rows]
        lam = impulses_np[world, :n_rows]

        tangent_sq: dict[int, float] = {}
        for i in range(n_rows):
            if int(row_type[i]) == 2:  # PGS_CONSTRAINT_TYPE_FRICTION
                parent = int(row_parent[i])
                if parent < 0:
                    continue
                tangent_sq[parent] = tangent_sq.get(parent, 0.0) + float(lam[i]) ** 2

        for i in range(n_rows):
            rt = int(row_type[i])
            if rt == 3:  # PGS_CONSTRAINT_TYPE_JOINT_LIMIT
                buckets["joint_limit"] += 1
                continue
            if rt != 0:  # only normal contact rows drive normal-vs-friction buckets
                continue
            t2 = tangent_sq.get(i)
            if t2 is None:
                buckets["normal_only"] += 1
                continue
            lam_n = float(lam[i])
            mu = float(row_mu[i])
            if lam_n <= 0.0 or mu <= 0.0:
                # No active normal impulse — treat as normal-only (no cone bound to test against).
                buckets["normal_only"] += 1
                continue
            bound = mu * lam_n
            if t2 >= ((1.0 - cone_tolerance) * bound) ** 2:
                buckets["sliding_friction"] += 1
            else:
                buckets["sticking_friction"] += 1
    return buckets


def _capture_contact_mix(
    *,
    model: newton.Model,
    snapshot: _Snapshot,
    gs_probe: int,
    dense_max_constraints: int,
) -> dict[str, object]:
    """Classify the snapshot frame into the 2x4 contact-mix table.

    Dense-path rows are attributed to the ``articulated`` row of the
    table (they live inside articulations — including mixed contacts
    that couple an articulation with a free-rigid body), while
    matrix-free-path rows are attributed to the ``free_rigid`` row.
    The four columns are ``{normal_only, sticking_friction,
    sliding_friction, joint_limit}``.  Totals are added on write.

    The probe runs ``friction_mode="current"`` at ``gs_probe`` iterations
    to populate the solver row buffers, then classifies them with
    :func:`_classify_path_rows`.  Returns a dict shaped::

        {
            "cone_tolerance": float,
            "gs_probe": int,
            "rows": ["articulated", "free_rigid"],
            "columns": ["normal_only", ...],
            "counts": {"articulated": {...}, "free_rigid": {...}},
            "totals": {"articulated": int, "free_rigid": int, "grand_total": int},
        }
    """
    solver = _build_replay_solver(
        model=model,
        friction_mode="current",
        pgs_iterations=int(gs_probe),
        dense_max_constraints=dense_max_constraints,
    )
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    snapshot.populate_state(state_0)
    snapshot.populate_control(control)
    sub_dt = snapshot.sim_dt / float(snapshot.substeps)
    contacts = model.collide(state_0)
    for _sub in range(int(snapshot.substeps)):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts, sub_dt)
        state_0, state_1 = state_1, state_0

    cone_tolerance = 0.05
    articulated = dict.fromkeys(_CONTACT_MIX_COLUMNS, 0)
    try:
        articulated = _classify_path_rows(
            impulses_np=solver.impulses.numpy(),
            row_type_np=solver.row_type.numpy(),
            row_parent_np=solver.row_parent.numpy(),
            row_mu_np=solver.row_mu.numpy(),
            count_np=solver.constraint_count.numpy(),
            cone_tolerance=cone_tolerance,
        )
    except Exception as exc:  # pragma: no cover - buffer-shape dependent
        sys.stderr.write(f"[solver_replay] contact_mix: dense-path classification failed ({exc}).\n")

    free_rigid = dict.fromkeys(_CONTACT_MIX_COLUMNS, 0)
    mf_impulses = getattr(solver, "mf_impulses", None)
    if mf_impulses is not None:
        try:
            free_rigid = _classify_path_rows(
                impulses_np=mf_impulses.numpy(),
                row_type_np=solver.mf_row_type.numpy(),
                row_parent_np=solver.mf_row_parent.numpy(),
                row_mu_np=solver.mf_row_mu.numpy(),
                count_np=solver.mf_constraint_count.numpy(),
                cone_tolerance=cone_tolerance,
            )
        except Exception as exc:  # pragma: no cover - buffer-shape dependent
            sys.stderr.write(f"[solver_replay] contact_mix: matrix-free classification failed ({exc}).\n")

    row_totals = {
        "articulated": int(sum(articulated.values())),
        "free_rigid": int(sum(free_rigid.values())),
    }
    grand_total = row_totals["articulated"] + row_totals["free_rigid"]

    return {
        "cone_tolerance": cone_tolerance,
        "gs_probe": int(gs_probe),
        "rows": list(_CONTACT_MIX_ROWS),
        "columns": list(_CONTACT_MIX_COLUMNS),
        "counts": {
            "articulated": articulated,
            "free_rigid": free_rigid,
        },
        "totals": {
            **row_totals,
            "grand_total": int(grand_total),
        },
    }


def _check_two_run_stability(
    run_a: dict[str, dict[str, object]],
    run_b: dict[str, dict[str, object]],
    *,
    tolerance: float,
) -> None:
    """Raise ``SystemExit`` if ``max(rel_delta)`` on any residual exceeds ``tolerance``."""
    for mode, entry_a in run_a.items():
        entry_b = run_b.get(mode)
        if entry_b is None or entry_a.get("status") != "ok" or entry_b.get("status") != "ok":
            continue
        sweeps_a = entry_a.get("gs_sweeps", {})
        sweeps_b = entry_b.get("gs_sweeps", {})
        for gs_key, sweep_a in sweeps_a.items():
            sweep_b = sweeps_b.get(gs_key)
            if sweep_b is None:
                continue
            arr_a = np.asarray(sweep_a.get("residuals", []), dtype=np.float64)
            arr_b = np.asarray(sweep_b.get("residuals", []), dtype=np.float64)
            if arr_a.size == 0 or arr_b.size == 0 or arr_a.shape != arr_b.shape:
                continue
            denom = np.maximum(np.abs(arr_a), 1.0e-12)
            rel = np.abs(arr_a - arr_b) / denom
            # Ignore cells where both values are tiny — noise-level deltas
            # aren't meaningful for a "compare-paths" check.
            mask = (np.abs(arr_a) > 1.0e-8) | (np.abs(arr_b) > 1.0e-8)
            if not mask.any():
                continue
            rel_max = float(np.max(rel[mask]))
            if rel_max > tolerance:
                raise SystemExit(
                    f"self-check failed: {mode}/gs={gs_key}: max rel-delta {rel_max:.3e} > {tolerance:.3e}"
                )


# =============================================================================
# Optional PNG capture (ViewerGL headless)
# =============================================================================


def _try_save_snapshot_png(*, snapshot: _Snapshot, model: newton.Model, out_path: Path) -> bool:
    """Render the snapshot frame to a PNG via :class:`newton.viewer.ViewerGL`.

    Issue #14 calls for "a PNG screenshot of the snapshot frame via the
    existing GL viewer / renderer".  We restore the snapshot state into
    a fresh :class:`newton.State`, feed it to a headless
    :class:`~newton.viewer.ViewerGL`, and write the captured RGB frame
    out via ``PIL`` (already a transitive dep of the viewer stack) or
    falling back to ``imageio`` if PIL is missing.

    The viewer is instantiated with ``headless=True`` so it works on
    EGL-only machines.  On any initialisation failure — missing OpenGL /
    EGL libraries, no suitable pyglet backend, a PIL-free environment
    without imageio — the PNG is skipped and a warning is printed.  The
    JSON residual trace is the primary deliverable; the PNG is best-effort.
    Returns ``True`` when the PNG is written, ``False`` otherwise.
    """
    try:
        from newton.viewer import ViewerGL  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - environment-dependent
        sys.stderr.write(f"[solver_replay] PNG skipped: newton.viewer.ViewerGL import failed ({exc}).\n")
        return False

    try:
        viewer = ViewerGL(width=960, height=720, headless=True)
    except Exception as exc:  # pragma: no cover - environment-dependent
        sys.stderr.write(f"[solver_replay] PNG skipped: ViewerGL headless init failed ({exc}).\n")
        return False

    try:
        viewer.set_model(model)
        state = model.state()
        snapshot.populate_state(state)

        viewer.begin_frame(time=0.0)
        viewer.log_state(state)
        viewer.end_frame()

        frame = viewer.get_frame()
        frame_np = frame.numpy() if hasattr(frame, "numpy") else np.asarray(frame)

        # Normalise to uint8 HxWx3 RGB for PNG writers.  ``get_frame``
        # returns uint8 already, but guard against future changes.
        if frame_np.dtype != np.uint8:
            frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
        if frame_np.ndim == 3 and frame_np.shape[2] == 4:
            frame_np = frame_np[..., :3]

        # Drop our local references to the Warp-side frame buffer *before*
        # the viewer's GL context is torn down.  ``get_frame`` returns a
        # Warp array backed by a GPU allocation; keeping it alive past
        # ``viewer.close()`` is fine, but the viewer's internal
        # :class:`wp.RegisteredGLBuffer` (the PBO used for readback) holds
        # a CUDA/GL interop resource that must be unregistered while the
        # GL context is still alive.  Otherwise Warp emits
        # ``CUDA error 219: invalid OpenGL or DirectX context`` to stderr
        # during interpreter teardown.
        del frame
        del state

        if not _write_rgb_png(frame_np, out_path):
            return False
    except Exception as exc:  # pragma: no cover - render-path dependent
        sys.stderr.write(f"[solver_replay] PNG skipped: viewer render failed ({exc}).\n")
        return False
    finally:
        _shutdown_viewer_cleanly(viewer)
    return True


def _shutdown_viewer_cleanly(viewer: newton.viewer.ViewerGL) -> None:
    """Tear down a :class:`~newton.viewer.ViewerGL` without CUDA-GL noise.

    The viewer lazily creates a CUDA-registered GL pixel-buffer object in
    :meth:`newton.viewer.ViewerGL.get_frame`.  If that
    :class:`wp.RegisteredGLBuffer` is garbage-collected *after* the GL
    context has been destroyed, Warp writes
    ``Warp CUDA error 219: invalid OpenGL or DirectX context`` to
    ``stderr`` from the ``__del__`` hook.  To leave stderr clean on a
    successful PNG capture we:

    1. Flush any pending CUDA work so Warp's GL-interop resources are
       not in-use.
    2. Invalidate the viewer's readback PBO — this drops the
       :class:`wp.RegisteredGLBuffer` reference held by the viewer
       while the GL context is still alive.
    3. Force a full ``gc.collect()`` so the
       :meth:`wp.RegisteredGLBuffer.__del__` finaliser runs under the
       live CUDA-GL context rather than during interpreter shutdown.
    4. Finally close the viewer (tears down the pyglet window).

    All steps are best-effort — any exception is swallowed so teardown
    never masks a primary success.
    """
    try:
        wp.synchronize()
    except Exception:
        pass
    # Step (2): drop CUDA-registered PBO refs while GL context is live.
    try:
        invalidate = getattr(viewer, "_invalidate_pbo", None)
        if invalidate is not None:
            invalidate()
    except Exception:
        pass
    # Step (3): run __del__ of any RegisteredGLBuffer whose refcount
    # just hit zero so Warp's interop unregister runs on a live context.
    try:
        gc.collect()
    except Exception:
        pass
    # Step (4): finally close the window / renderer.
    try:
        viewer.close()
    except Exception:
        pass
    # One more gc.collect() catches any lingering pyglet/gl finalisers
    # so they don't surface on interpreter shutdown.
    try:
        gc.collect()
    except Exception:
        pass


def _write_rgb_png(frame_np: np.ndarray, out_path: Path) -> bool:
    """Write an ``HxWx3`` ``uint8`` RGB buffer to ``out_path`` as PNG.

    Prefers Pillow (used throughout the Newton viewer stack); falls back
    to ``imageio``; logs + returns ``False`` if neither is available.
    """
    try:
        from PIL import Image

        Image.fromarray(frame_np, mode="RGB").save(out_path)
        return True
    except Exception as pil_exc:
        try:
            import imageio.v2 as imageio  # noqa: PLC0415

            imageio.imwrite(str(out_path), frame_np)
            return True
        except Exception as imageio_exc:  # pragma: no cover
            sys.stderr.write(
                f"[solver_replay] PNG skipped: no PNG writer available (pillow: {pil_exc}; imageio: {imageio_exc}).\n"
            )
            return False


# =============================================================================
# select-frame
# =============================================================================


def _count_sliding_rows_generic(
    *,
    impulses_np: np.ndarray,
    row_type_np: np.ndarray,
    row_parent_np: np.ndarray,
    row_mu_np: np.ndarray,
    count_np: np.ndarray,
    cone_tolerance: float,
) -> int:
    """Shared core for dense / matrix-free sliding-row estimates.

    For each friction row (``row_type==2``) whose parent is a contact
    row, we compare the tangential impulse magnitude against the
    friction-cone bound ``mu * |lambda_n|`` and call the row "sliding"
    when ``|lambda_t_pair| >= (1 - cone_tolerance) * mu * |lambda_n|``.
    Friction rows come in pairs (two tangent directions share the same
    parent normal row), so we sum the two tangents into one per-contact
    magnitude.  Rows with zero normal impulse are excluded.
    """
    total = 0
    worlds = impulses_np.shape[0]
    for world in range(worlds):
        n_rows = int(count_np[world])
        if n_rows <= 0:
            continue
        row_type = row_type_np[world, :n_rows]
        row_parent = row_parent_np[world, :n_rows]
        row_mu = row_mu_np[world, :n_rows]
        lam = impulses_np[world, :n_rows]

        tangent_sq: dict[int, float] = {}
        for i in range(n_rows):
            if int(row_type[i]) == 2:  # PGS_CONSTRAINT_TYPE_FRICTION
                parent = int(row_parent[i])
                if parent < 0:
                    continue
                tangent_sq[parent] = tangent_sq.get(parent, 0.0) + float(lam[i]) ** 2

        for parent, t2 in tangent_sq.items():
            if parent >= n_rows:
                continue
            lam_n = float(lam[parent])
            mu = float(row_mu[parent])
            if lam_n <= 0.0 or mu <= 0.0:
                continue
            bound = mu * lam_n
            if t2 >= ((1.0 - cone_tolerance) * bound) ** 2:
                total += 1
    return total


def _estimate_sliding_rows(solver: SolverFeatherPGS, *, cone_tolerance: float = 0.05) -> int:
    """Count sliding friction rows across both dense and matrix-free paths.

    FeatherPGS routes contacts through two parallel buffers: the dense
    path (articulation-local rows, ``solver.impulses`` / ``row_type``)
    and the matrix-free path (free-body rows, ``solver.mf_impulses`` /
    ``mf_row_type``).  Both use the same ``PGS_CONSTRAINT_TYPE_FRICTION``
    convention with ``row_parent`` pointing at the paired normal row.
    We count sliding rows in both and sum.
    """
    total = 0
    try:
        total += _count_sliding_rows_generic(
            impulses_np=solver.impulses.numpy(),
            row_type_np=solver.row_type.numpy(),
            row_parent_np=solver.row_parent.numpy(),
            row_mu_np=solver.row_mu.numpy(),
            count_np=solver.constraint_count.numpy(),
            cone_tolerance=cone_tolerance,
        )
    except Exception:
        pass
    mf_impulses = getattr(solver, "mf_impulses", None)
    if mf_impulses is not None:
        try:
            total += _count_sliding_rows_generic(
                impulses_np=mf_impulses.numpy(),
                row_type_np=solver.mf_row_type.numpy(),
                row_parent_np=solver.mf_row_parent.numpy(),
                row_mu_np=solver.mf_row_mu.numpy(),
                count_np=solver.mf_constraint_count.numpy(),
                cone_tolerance=cone_tolerance,
            )
        except Exception:
            pass
    return total


def _run_select_frame(
    *,
    scenario: str,
    step_range: tuple[int, int],
    step_stride: int,
    gs_probe: int,
    sim_dt: float,
    out_dir: Path,
    device: wp.context.Device,
) -> dict[str, object]:
    """Advance the scenario and probe each candidate step for "hardness"."""
    cfg = _SCENARIO_DEFAULTS[scenario]
    substeps = int(cfg["substeps"])
    dense_max_constraints = int(cfg["dense_max_constraints"])

    model = _build_model_or_raise(scenario, device)

    # Warm-up solver does plain stepping with friction_mode="current".  We
    # snapshot before each candidate step and run a one-frame probe with a
    # fresh pgs_debug=True solver so the main solver's state is not
    # perturbed by the debug path.
    warm_solver = SolverFeatherPGS(
        model,
        pgs_mode="matrix_free",
        friction_mode="current",
        pgs_iterations=int(cfg["pgs_iterations_warmup"]),
        pgs_debug=False,
        dense_max_constraints=dense_max_constraints,
    )
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    sub_dt = sim_dt / float(substeps)

    lo, hi = step_range
    candidates: list[dict[str, object]] = []

    for step in range(hi):
        is_candidate = step >= lo and ((step - lo) % step_stride) == 0
        if is_candidate:
            # Snapshot state_0 (+ control + contacts + model ref) and run a probe.
            probe_contacts = model.collide(state_0)
            snapshot = _snapshot_state(
                state_0,
                control=control,
                contacts=probe_contacts,
                model=model,
                scenario=scenario,
                target_step=step,
                substeps=substeps,
                sim_dt=sim_dt,
            )
            probe_metrics = _probe_step_hardness(
                model=model,
                snapshot=snapshot,
                gs_probe=gs_probe,
                dense_max_constraints=dense_max_constraints,
            )
            probe_metrics["step"] = int(step)
            candidates.append(probe_metrics)

        # Advance warm solver by one frame.
        contacts = model.collide(state_0)
        for _sub in range(substeps):
            state_0.clear_forces()
            warm_solver.step(state_0, state_1, control, contacts, sub_dt)
            state_0, state_1 = state_1, state_0

    # Rank: r_compl descending (primary), then mixed rows desc, then
    # sliding rows desc.  The issue calls for contact-type aware ranking;
    # mixed rows (articulated ↔ free-rigid contacts) are usually the
    # hardest cases because they couple the dense and matrix-free PGS
    # paths, so we prioritise them as the first tie-breaker before the
    # sliding-row heuristic.
    candidates.sort(
        key=lambda c: (
            -float(c.get("r_compl_final", 0.0)),
            -int(c.get("mixed_contact_rows", 0)),
            -int(c.get("articulated_contact_rows", 0)),
            -int(c.get("free_rigid_contact_rows", 0)),
            -int(c.get("sliding_rows", 0)),
        )
    )

    summary: dict[str, object] = {
        "schema_version": "1.1",
        "tool": "solver_replay",
        "mode": "select-frame",
        "scenario": scenario,
        "step_range": [int(step_range[0]), int(step_range[1])],
        "step_stride": int(step_stride),
        "gs_probe": int(gs_probe),
        "sim_dt": float(sim_dt),
        "substeps": substeps,
        # Per-candidate row breakdown keys: articulated_contact_rows,
        # mixed_contact_rows, free_rigid_contact_rows — the issue's
        # requested "mixed / free-rigid / articulated" split.  Legacy
        # aggregate fields (contact_rows, dense_contact_rows,
        # mf_contact_rows) remain for back-compat.
        "ranking": candidates,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "frame_ranking.json"
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)

    # Print a human-readable top-10 summary.
    print(f"[solver_replay] select-frame top {min(10, len(candidates))} (scenario={scenario}):")
    for i, entry in enumerate(candidates[:10]):
        print(
            f"  #{i + 1:2d} step={entry['step']:5d}  "
            f"r_compl={float(entry['r_compl_final']):.4e}  "
            f"art={int(entry['articulated_contact_rows']):3d}  "
            f"mixed={int(entry['mixed_contact_rows']):3d}  "
            f"free_rigid={int(entry['free_rigid_contact_rows']):3d}  "
            f"sliding={int(entry['sliding_rows']):3d}"
        )

    return summary


def _probe_step_hardness(
    *,
    model: newton.Model,
    snapshot: _Snapshot,
    gs_probe: int,
    dense_max_constraints: int,
) -> dict[str, object]:
    """Run a pgs_debug probe across one full frame and extract per-frame hardness metrics.

    The probe covers all substeps of the scenario's frame (not just the
    first substep) so multi-substep scenarios such as ``g1_flat`` or
    ``h1_tabletop`` are ranked against the full frame difficulty the
    replay mode will later reproduce.
    """
    solver = _build_replay_solver(
        model=model,
        friction_mode="current",
        pgs_iterations=gs_probe,
        dense_max_constraints=dense_max_constraints,
    )
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    snapshot.populate_state(state_0)
    snapshot.populate_control(control)

    sub_dt = snapshot.sim_dt / float(snapshot.substeps)
    # Advance the probe a full frame's worth of substeps — multi-substep
    # scenarios (``g1_flat`` / ``h1_tabletop``) would otherwise rank against
    # only their first substep, which underreports frame-level "hardness".
    # The residual log accumulates one ``[iters, world, 6]`` entry per
    # substep, so we inspect all of them below.
    #
    # Collide exactly once per outer frame and reuse the ``contacts``
    # object across all substeps — this matches the canonical stepping
    # pattern in ``newton/tools/solver_benchmark.py`` and the friction-mode
    # tests.  Re-colliding every substep would diverge from the replay /
    # warmup semantics and bias the ranking for multi-substep scenarios.
    contacts = model.collide(state_0)
    for _sub in range(int(snapshot.substeps)):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts, sub_dt)
        state_0, state_1 = state_1, state_0

    # r_compl final: last iter per substep, max over worlds, then max over
    # substeps to represent the hardest moment in the frame.
    r_compl_final = 0.0
    if solver._pgs_ncp_residual_log:
        for entry in solver._pgs_ncp_residual_log:
            stack = np.asarray(entry)
            # stack: [iters, world, 6]
            if stack.size == 0:
                continue
            r_compl_final = max(r_compl_final, float(np.max(stack[-1, :, 0])))

    # Contact rows: dense + matrix-free, from the final substep (the solver
    # buffers only hold the most recently solved substep's counts).
    constraint_count_np = solver.constraint_count.numpy()
    mf_constraint_count_np = solver.mf_constraint_count.numpy()
    dense_contact_rows = int(np.sum(constraint_count_np))
    mf_contact_rows = int(np.sum(mf_constraint_count_np))
    contact_rows = dense_contact_rows + mf_contact_rows

    # Row-type breakdown: classify each active contact as articulated /
    # mixed / free-rigid per the issue's select-frame contract.  Uses the
    # per-contact routing arrays ``contact_path`` (1 = matrix-free, 0 =
    # dense) and the per-articulation ``is_free_rigid`` mask that
    # :meth:`SolverFeatherPGS._classify_free_rigid_bodies` builds.
    row_breakdown = _classify_contact_row_types(solver)

    # Sliding rows: heuristic count across both the dense (articulation)
    # and matrix-free (free rigid body) paths — at the cone boundary the
    # tangent impulse saturates, and those are the rows the plotting
    # harness in 10/13 is most interested in.
    try:
        sliding_rows = _estimate_sliding_rows(solver)
    except Exception as exc:
        # Non-fatal: sliding count is a heuristic; don't let it break
        # the ranking if any buffer is missing.
        sys.stderr.write(f"[solver_replay] sliding_rows estimate unavailable: {exc}\n")
        sliding_rows = 0

    return {
        "r_compl_final": r_compl_final,
        "contact_rows": contact_rows,
        "dense_contact_rows": dense_contact_rows,
        "mf_contact_rows": mf_contact_rows,
        "articulated_contact_rows": int(row_breakdown["articulated"]),
        "mixed_contact_rows": int(row_breakdown["mixed"]),
        "free_rigid_contact_rows": int(row_breakdown["free_rigid"]),
        "sliding_rows": int(sliding_rows),
    }


def _classify_contact_row_types(solver: SolverFeatherPGS) -> dict[str, int]:
    """Classify active contacts into articulated / mixed / free-rigid buckets.

    FeatherPGS routes every rigid contact into either the dense path
    (articulation-local rows, used whenever at least one side is a
    non-free-rigid articulation) or the matrix-free path (both sides are
    free-rigid or ground).  Within the dense path we further distinguish
    *mixed* contacts — one free-rigid side, one articulated side — from
    fully *articulated* contacts where both sides are non-free-rigid.

    The buckets are counted per active contact (not per constraint row,
    so the count is independent of whether the normal + two friction
    rows are present).  Ground is treated as "free-rigid" for
    classification purposes, matching the routing logic in
    ``allocate_world_contact_slots``.

    Returns:
        dict: ``{"articulated": int, "mixed": int, "free_rigid": int}``.
    """
    zero = {"articulated": 0, "mixed": 0, "free_rigid": 0}
    try:
        contact_path = solver.contact_path.numpy()
        contact_art_a = solver.contact_art_a.numpy()
        contact_art_b = solver.contact_art_b.numpy()
    except Exception:
        return zero

    is_free_rigid_arr = getattr(solver, "is_free_rigid", None)
    if is_free_rigid_arr is None:
        # Pure articulated simulation (no free-rigid bodies): every
        # active contact is articulated by definition.
        active = np.count_nonzero(contact_path >= 0)
        return {"articulated": int(active), "mixed": 0, "free_rigid": 0}
    is_free_rigid = is_free_rigid_arr.numpy()

    def _side_is_free(art_idx: int) -> bool:
        # Ground (art_idx < 0) is treated as a free-rigid side for
        # routing / classification, matching the solver's allocator.
        return art_idx < 0 or bool(is_free_rigid[int(art_idx)])

    counts = {"articulated": 0, "mixed": 0, "free_rigid": 0}
    n = contact_path.shape[0]
    for c in range(n):
        path = int(contact_path[c])
        if path < 0:
            # Slot was allocated but skipped (e.g. no active contact this frame).
            continue
        a_free = _side_is_free(contact_art_a[c])
        b_free = _side_is_free(contact_art_b[c])
        if a_free and b_free:
            counts["free_rigid"] += 1
        elif not a_free and not b_free:
            counts["articulated"] += 1
        else:
            counts["mixed"] += 1
    return counts


# =============================================================================
# CLI
# =============================================================================


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=("replay", "select-frame"),
        default="replay",
        help="Tool sub-mode.  ``replay`` (default) captures residual traces at a chosen step. "
        "``select-frame`` ranks candidate hard frames in a step range.",
    )
    parser.add_argument(
        "--scenario",
        choices=sorted(_SCENARIO_DEFAULTS),
        required=True,
        help="Scenario name (shared with ``newton.tools.solver_benchmark``).",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=100,
        help="(replay) Target step — number of warmup frames to take before the snapshot.",
    )
    parser.add_argument(
        "--friction-modes",
        default="current",
        help=f"(replay) Comma-separated list of friction_mode values. Valid: {','.join(_VALID_FRICTION_MODES)}.",
    )
    parser.add_argument(
        "--gs-iterations",
        default="1,2,5,10,20,50,100",
        help="(replay) Comma-separated list of PGS iteration counts to sweep. "
        "(select-frame) Single value: the fixed GS iteration budget used by the probe.",
    )
    parser.add_argument(
        "--sim-dt",
        type=float,
        default=1.0 / 240.0,
        help="Outer simulation ``dt`` per frame (seconds).  Substeps per frame are taken from scenario defaults.",
    )
    parser.add_argument(
        "--step-range",
        type=_parse_step_range,
        default=None,
        help="(select-frame) Candidate step range LO..HI (HI exclusive).",
    )
    parser.add_argument(
        "--step-stride",
        type=int,
        default=5,
        help="(select-frame) Stride between candidate steps within ``--step-range``.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory.  Will be created if it doesn't exist.",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="(replay) After the primary run, rerun the replay and assert max rel-delta < 1e-2 on residuals.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Warp device to use (e.g. ``cuda:0``).  Default: current Warp device.",
    )
    return parser


def _resolve_device(device_arg: str | None) -> wp.context.Device:
    if device_arg is None:
        device = wp.get_device()
    else:
        device = wp.get_device(device_arg)
    if device is None:
        raise SystemExit("no Warp device available; cannot run solver_replay.")
    return device


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    device = _resolve_device(args.device)

    out_dir: Path = args.out
    t0 = time.perf_counter()
    if args.mode == "replay":
        friction_modes = _parse_csv_str_list(args.friction_modes, _VALID_FRICTION_MODES)
        gs_sweep = _parse_csv_int_list(args.gs_iterations)
        _run_replay(
            scenario=args.scenario,
            target_step=int(args.step),
            friction_modes=friction_modes,
            gs_sweep=gs_sweep,
            sim_dt=float(args.sim_dt),
            out_dir=out_dir,
            self_check=bool(args.self_check),
            device=device,
        )
    else:
        if args.step_range is None:
            parser.error("--step-range is required for --mode select-frame")
        gs_list = _parse_csv_int_list(args.gs_iterations)
        if len(gs_list) != 1:
            parser.error("--gs-iterations must be a single integer for --mode select-frame")
        _run_select_frame(
            scenario=args.scenario,
            step_range=args.step_range,
            step_stride=int(args.step_stride),
            gs_probe=int(gs_list[0]),
            sim_dt=float(args.sim_dt),
            out_dir=out_dir,
            device=device,
        )
    dt = time.perf_counter() - t0
    print(f"[solver_replay] done in {dt:.1f}s; output dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
