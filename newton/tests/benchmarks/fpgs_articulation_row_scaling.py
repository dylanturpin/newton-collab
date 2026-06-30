# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Production FeatherPGS articulated-response scaling benchmark.

This benchmark intentionally uses the public SolverFeatherPGS.step path only.
The only compared modes are the current matrix-free articulated response
("immediate") and the compact tree articulated-contact response. There are
no private generated-kernel calls and no reduced row-solver substitutes here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

try:
    import torch
except Exception:  # pragma: no cover - optional benchmark instrumentation
    torch = None

_NEWTON_REPO = Path(__file__).resolve().parents[3]
if str(_NEWTON_REPO) not in sys.path:
    sys.path.insert(0, str(_NEWTON_REPO))

import newton  # noqa: E402
from newton.solvers import SolverFeatherPGS  # noqa: E402


COMPACT_PATHS = ("compact_tree", "compact_cholesky")


@dataclass(frozen=True)
class BenchCase:
    sweep: str
    label: str
    case_kind: str
    world_count: int = 1
    free_pairs: int = 0
    articulations: int = 0
    links: int = 0
    contacts_per_articulation: int = 0
    joint_armature: float = 0.0


@dataclass
class RunResult:
    sweep: str
    label: str
    case_kind: str
    path: str
    compact_active: bool
    requested_world_count: int
    free_pairs: int
    articulations: int
    links: int
    contacts_per_articulation: int
    world_count: int
    body_count: int
    joint_dof_count: int
    dense_row_capacity: int
    mf_row_capacity: int
    compact_row_capacity: int
    dense_rows_total: int
    dense_rows_max_world: int
    mf_rows_total: int
    mf_rows_max_world: int
    compact_rows_total: int
    compact_rows_max_world: int
    rigid_contacts: int
    ms_per_step: float
    peak_gpu_mem_bytes: int
    row_solver_mib: float
    propagation_extra_mib: float
    joint_q_rel_l2: float | None = None
    joint_qd_rel_l2: float | None = None
    joint_qd_abs_l2: float | None = None
    joint_qd_abs_linf: float | None = None
    body_q_rel_l2: float | None = None
    body_qd_rel_l2: float | None = None
    body_qd_abs_l2: float | None = None
    body_qd_abs_linf: float | None = None
    state_linf: float | None = None


def _next_multiple(value: int, multiple: int) -> int:
    return int(math.ceil(max(value, 1) / multiple) * multiple)


def _planned_row_capacity(case: BenchCase) -> int:
    if case.case_kind == "free_free":
        contacts = max(case.free_pairs, 1)
    else:
        contacts = max(case.articulations * case.contacts_per_articulation, 1)

    # Box/box contact commonly emits up to four contact points. Each point can
    # become one normal plus two friction rows. Keep a small margin for contact
    # manifold changes while still letting capacity scale with the requested case.
    return _next_multiple(contacts * 16, 16)


def _selected_contact_links(num_links: int, count: int) -> list[int]:
    if count <= 0:
        return []
    if count == 1:
        return [num_links - 1]
    if count >= num_links:
        return list(range(num_links))

    out: list[int] = []
    for value in np.linspace(0, num_links - 1, count):
        idx = int(round(float(value)))
        if idx not in out:
            out.append(idx)
    while len(out) < count:
        idx = min(num_links - 1, len(out))
        if idx not in out:
            out.append(idx)
        else:
            break
    return out


def _max_contact_slots_per_link(num_links: int, count: int) -> int:
    if num_links <= 0 or count <= 0:
        return 0
    return int(math.ceil(count / num_links))


def _contact_link_slots(num_links: int, count: int) -> list[tuple[int, float]]:
    if count <= num_links:
        return [(link_idx, 0.0) for link_idx in _selected_contact_links(num_links, count)]

    slots_by_link = [[] for _ in range(num_links)]
    for contact_idx in range(count):
        slots_by_link[contact_idx % num_links].append(contact_idx // num_links)

    out: list[tuple[int, float]] = []
    slot_spacing = 0.09
    for link_idx, slots in enumerate(slots_by_link):
        n_slots = len(slots)
        if n_slots == 0:
            continue
        for slot_idx in range(n_slots):
            y_offset = 0.0 if n_slots == 1 else (slot_idx - 0.5 * (n_slots - 1)) * slot_spacing
            out.append((link_idx, y_offset))
    return out


def _configure_shape_defaults(builder: newton.ModelBuilder) -> None:
    builder.default_shape_cfg.density = 1000.0
    builder.default_shape_cfg.ke = 1.0e5
    builder.default_shape_cfg.kd = 1.0e3
    builder.default_shape_cfg.kf = 1.0e3
    builder.default_shape_cfg.mu = 0.75
    builder.default_shape_cfg.margin = 0.0
    builder.default_shape_cfg.gap = 0.0


def _build_free_free_builder(case: BenchCase) -> tuple[newton.ModelBuilder, tuple[float, float, float]]:
    builder = newton.ModelBuilder(gravity=0.0)
    _configure_shape_defaults(builder)

    hx = hy = hz = 0.05
    penetration = 0.012
    spacing = 0.45
    for i in range(case.free_pairs):
        x = i * spacing
        lower = builder.add_body(xform=wp.transform(wp.vec3(x, 0.0, 0.5), wp.quat_identity()))
        upper = builder.add_body(
            xform=wp.transform(wp.vec3(x, 0.0, 0.5 + 2.0 * hz - penetration), wp.quat_identity())
        )
        builder.add_shape_box(lower, hx=hx, hy=hy, hz=hz)
        builder.add_shape_box(upper, hx=hx, hy=hy, hz=hz)

    return builder, (max(case.free_pairs, 1) * spacing + 0.5, 0.0, 0.0)


def _replicate_or_finalize(
    template: newton.ModelBuilder,
    world_count: int,
    spacing: tuple[float, float, float],
    device: str,
) -> newton.Model:
    if world_count <= 1:
        return template.finalize(device=device)
    builder = newton.ModelBuilder(gravity=0.0)
    builder.replicate(template, world_count, spacing=spacing)
    return builder.finalize(device=device)


def _build_free_free_model(case: BenchCase, device: str) -> newton.Model:
    template, spacing = _build_free_free_builder(case)
    return _replicate_or_finalize(template, case.world_count, spacing, device)


def _build_articulated_free_builder(case: BenchCase) -> tuple[newton.ModelBuilder, tuple[float, float, float]]:
    builder = newton.ModelBuilder(gravity=0.0)
    _configure_shape_defaults(builder)

    cube_h = 0.04
    contact_slot_spacing = 0.09
    max_slots_per_link = _max_contact_slots_per_link(case.links, case.contacts_per_articulation)
    link_hx = 0.15
    link_hy = 0.06
    if max_slots_per_link > 2:
        link_hy = max(link_hy, 0.5 * (max_slots_per_link - 1) * contact_slot_spacing + cube_h + 0.01)
    link_hz = 0.045
    penetration = 0.012
    art_spacing_y = 0.55
    base_z = 0.5

    for art_idx in range(case.articulations):
        origin = wp.vec3(0.0, art_idx * art_spacing_y, base_z)
        parent = -1
        joints: list[int] = []
        link_bodies: list[int] = []

        for link_idx in range(case.links):
            link = builder.add_link()
            link_bodies.append(link)
            builder.add_shape_box(link, hx=link_hx, hy=link_hy, hz=link_hz)

            if parent == -1:
                parent_xform = wp.transform(origin, wp.quat_identity())
            else:
                parent_xform = wp.transform(wp.vec3(link_hx, 0.0, 0.0), wp.quat_identity())

            joints.append(
                builder.add_joint_revolute(
                    parent=parent,
                    child=link,
                    axis=wp.vec3(0.0, 1.0, 0.0),
                    parent_xform=parent_xform,
                    child_xform=wp.transform(wp.vec3(-link_hx, 0.0, 0.0), wp.quat_identity()),
                    armature=case.joint_armature,
                )
            )
            parent = link

        builder.add_articulation(joints)

        for link_idx, y_offset in _contact_link_slots(case.links, case.contacts_per_articulation):
            x = float(origin[0]) + (2.0 * link_idx + 1.0) * link_hx
            y = float(origin[1]) + y_offset
            z = base_z + link_hz + cube_h - penetration
            cube = builder.add_body(xform=wp.transform(wp.vec3(x, y, z), wp.quat_identity()))
            builder.add_shape_box(cube, hx=cube_h, hy=cube_h, hz=cube_h)

    world_spacing_y = max(case.articulations + 2, 1) * art_spacing_y
    return builder, (0.0, world_spacing_y, 0.0)


def _build_articulated_free_model(case: BenchCase, device: str) -> newton.Model:
    template, spacing = _build_articulated_free_builder(case)
    return _replicate_or_finalize(template, case.world_count, spacing, device)


def _build_model(case: BenchCase, device: str) -> newton.Model:
    if case.case_kind == "free_free":
        return _build_free_free_model(case, device)
    if case.case_kind == "articulated_free":
        return _build_articulated_free_model(case, device)
    raise ValueError(f"unknown case kind {case.case_kind!r}")


def _snapshot_state(state: newton.State) -> dict[str, np.ndarray]:
    return {
        "joint_q": state.joint_q.numpy().copy(),
        "joint_qd": state.joint_qd.numpy().copy(),
        "body_q": state.body_q.numpy().copy(),
        "body_qd": state.body_qd.numpy().copy(),
    }


def _restore_state(model: newton.Model, state: newton.State, snapshot: dict[str, np.ndarray]) -> None:
    state.joint_q.assign(snapshot["joint_q"])
    state.joint_qd.assign(snapshot["joint_qd"])
    state.body_q.assign(snapshot["body_q"])
    state.body_qd.assign(snapshot["body_qd"])
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)


def _array_nbytes(array: Any) -> int:
    if array is None:
        return 0
    shape = getattr(array, "shape", None)
    if shape is None:
        return 0
    count = 1
    for dim in shape:
        count *= int(dim)
    dtype = getattr(array, "dtype", None)
    if dtype in (wp.float64, wp.int64):
        itemsize = 8
    elif dtype in (wp.int8, wp.uint8):
        itemsize = 1
    elif dtype in (wp.int16, wp.uint16):
        itemsize = 2
    else:
        itemsize = 4
    return count * itemsize


def _row_solver_bytes(solver: SolverFeatherPGS) -> int:
    names = (
        "J_world",
        "Y_world",
        "diag",
        "rhs",
        "rhs_unbiased",
        "impulses",
        "row_type",
        "row_parent",
        "row_mu",
        "row_cfm",
        "phi",
        "mf_rhs",
        "mf_rhs_unbiased",
        "mf_impulses",
        "mf_eff_mass_inv",
        "mf_row_type",
        "mf_row_parent",
        "mf_row_mu",
        "mf_J_a",
        "mf_J_b",
        "mf_MiJt_a",
        "mf_MiJt_b",
        "mf_dof_a",
        "mf_dof_b",
        "compact_rhs",
        "compact_rhs_unbiased",
        "compact_impulses",
        "compact_eff_mass_inv",
        "compact_row_type",
        "compact_row_parent",
        "compact_row_mu",
        "compact_phi",
        "compact_J_a",
        "compact_J_b",
        "compact_MiJt_a",
        "compact_MiJt_b",
        "compact_body_a",
        "compact_body_b",
        "world_deferred_dof_mask",
    )
    return sum(_array_nbytes(getattr(solver, name, None)) for name in names)


def _propagation_extra_bytes(solver: SolverFeatherPGS) -> int:
    names = (
        "_deferred_dense_prev_impulses",
        "_deferred_dense_delta_impulses",
        "_deferred_dense_tau",
        "_deferred_dense_qd_delta",
        "compact_body_response",
        "compact_body_qd",
        "compact_body_impulses",
        "compact_tree_Ia",
        "compact_tree_U",
        "compact_tree_D_chol",
        "compact_tree_D_inv",
        "compact_tree_pA",
        "compact_tree_u",
        "compact_tree_qdd",
        "compact_tree_body_delta",
    )
    return sum(_array_nbytes(getattr(solver, name, None)) for name in names)


def _gpu_used_bytes() -> int:
    if torch is None or not torch.cuda.is_available():
        return 0
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return int(total_bytes - free_bytes)


def _finite_rel_l2(value: np.ndarray, reference: np.ndarray) -> float:
    diff = np.asarray(value, dtype=np.float64).reshape(-1) - np.asarray(reference, dtype=np.float64).reshape(-1)
    denom = np.linalg.norm(np.asarray(reference, dtype=np.float64).reshape(-1))
    if denom == 0.0:
        denom = 1.0
    return float(np.linalg.norm(diff) / denom)


def _abs_l2(value: np.ndarray, reference: np.ndarray) -> float:
    diff = np.asarray(value, dtype=np.float64).reshape(-1) - np.asarray(reference, dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(diff))


def _abs_linf(value: np.ndarray, reference: np.ndarray) -> float:
    diff = np.asarray(value, dtype=np.float64).reshape(-1) - np.asarray(reference, dtype=np.float64).reshape(-1)
    return float(np.max(np.abs(diff))) if diff.size else 0.0


def _state_linf(value: dict[str, np.ndarray], reference: dict[str, np.ndarray]) -> float:
    max_diff = 0.0
    for key, ref in reference.items():
        diff = np.max(np.abs(np.asarray(value[key], dtype=np.float64) - np.asarray(ref, dtype=np.float64)))
        max_diff = max(max_diff, float(diff))
    return max_diff


def _run_step_loop(
    model: newton.Model,
    solver: SolverFeatherPGS,
    contacts: newton.Contacts,
    initial: dict[str, np.ndarray],
    *,
    repeats: int,
    warmups: int,
    dt: float,
) -> tuple[float, dict[str, np.ndarray], int]:
    state_in = model.state()
    state_out = model.state()
    control = model.control()
    peak_gpu_mem_bytes = _gpu_used_bytes()

    for _ in range(warmups):
        _restore_state(model, state_in, initial)
        _restore_state(model, state_out, initial)
        state_in.clear_forces()
        state_out.clear_forces()
        solver.step(state_in, state_out, control, contacts, dt)
        peak_gpu_mem_bytes = max(peak_gpu_mem_bytes, _gpu_used_bytes())
    wp.synchronize()
    peak_gpu_mem_bytes = max(peak_gpu_mem_bytes, _gpu_used_bytes())

    start = time.perf_counter()
    for _ in range(repeats):
        _restore_state(model, state_in, initial)
        _restore_state(model, state_out, initial)
        state_in.clear_forces()
        state_out.clear_forces()
        solver.step(state_in, state_out, control, contacts, dt)
        peak_gpu_mem_bytes = max(peak_gpu_mem_bytes, _gpu_used_bytes())
    wp.synchronize()
    peak_gpu_mem_bytes = max(peak_gpu_mem_bytes, _gpu_used_bytes())

    elapsed = time.perf_counter() - start
    return (elapsed * 1000.0 / repeats), _snapshot_state(state_out), peak_gpu_mem_bytes


def _prepare_case_run(case: BenchCase, device: str) -> tuple[newton.Model, dict[str, np.ndarray], newton.Contacts]:
    model = _build_model(case, device)
    initial_state = model.state()
    newton.eval_fk(model, initial_state.joint_q, initial_state.joint_qd, initial_state)
    initial_state.clear_forces()
    initial = _snapshot_state(initial_state)
    contacts = model.contacts()
    model.collide(initial_state, contacts)
    wp.synchronize()
    return model, initial, contacts


def _run_case_path(
    case: BenchCase,
    path: str,
    *,
    device: str,
    repeats: int,
    warmups: int,
    dt: float,
    pgs_iterations: int,
    pgs_velocity_iterations: int,
    enable_contact_friction: bool,
    contact_friction_position_iterations: int,
    compact_fast_body_map: bool,
    compact_existing_row_phases: str,
    compact_shared_row_solver: bool,
    compact_warp_propagation: bool,
    compact_max_constraints_override: int | None,
    model: newton.Model | None = None,
    initial: dict[str, np.ndarray] | None = None,
    contacts: newton.Contacts | None = None,
) -> tuple[RunResult, dict[str, np.ndarray]]:
    if model is None or initial is None or contacts is None:
        model, initial, contacts = _prepare_case_run(case, device)

    row_capacity = _planned_row_capacity(case)
    dense_capacity = row_capacity
    mf_capacity = row_capacity
    compact_capacity = row_capacity
    if compact_max_constraints_override is not None:
        compact_capacity = compact_max_constraints_override
    if case.case_kind == "articulated_free":
        # These scenes intentionally contain articulated/free contacts but no
        # free/free contact pairs. Keep MF capacity at a small internal margin
        # for both paths so the comparison focuses on dense D-wide contact rows
        # versus compact contact rows.
        mf_capacity = 16
    if path in ("compact_cholesky", "compact_tree") and case.case_kind == "articulated_free":
        # Compact articulated contacts do not need D-wide dense contact rows.
        # Keep a small dense capacity for non-contact rows such as drives or
        # joint limits while the compact capacity scales with requested contacts.
        dense_capacity = 16
    response_mode = "immediate"
    if path in ("compact_cholesky", "compact_tree"):
        response_mode = path
    peak_gpu_mem_bytes = _gpu_used_bytes()
    solver = SolverFeatherPGS(
        model,
        pgs_mode="matrix_free",
        articulated_dense_response_mode=response_mode,
        hinv_jt_kernel="par_row",
        pgs_iterations=pgs_iterations,
        pgs_velocity_iterations=pgs_velocity_iterations,
        enable_contact_friction=enable_contact_friction,
        contact_friction_position_iterations=contact_friction_position_iterations,
        dense_max_constraints=dense_capacity,
        mf_max_constraints=mf_capacity,
        compact_max_constraints=compact_capacity,
        compact_fast_body_map=compact_fast_body_map,
        compact_existing_row_phases=compact_existing_row_phases if path == "compact_tree" else "auto",
        compact_shared_row_solver=compact_shared_row_solver if path == "compact_tree" else False,
        compact_warp_propagation=compact_warp_propagation if path == "compact_tree" else False,
        pgs_warmstart=False,
        mf_warmstart=False,
    )
    wp.synchronize()
    peak_gpu_mem_bytes = max(peak_gpu_mem_bytes, _gpu_used_bytes())
    ms_per_step, final_state, loop_peak_gpu_mem_bytes = _run_step_loop(
        model,
        solver,
        contacts,
        initial,
        repeats=repeats,
        warmups=warmups,
        dt=dt,
    )
    peak_gpu_mem_bytes = max(peak_gpu_mem_bytes, loop_peak_gpu_mem_bytes)

    dense_counts = solver.constraint_count.numpy().astype(np.int64, copy=False)
    mf_counts = solver.mf_constraint_count.numpy().astype(np.int64, copy=False)
    compact_count_array = getattr(solver, "compact_constraint_count", None)
    if compact_count_array is None:
        compact_counts = np.zeros_like(dense_counts)
    else:
        compact_counts = compact_count_array.numpy().astype(np.int64, copy=False)
    rigid_contacts = int(contacts.rigid_contact_count.numpy()[0])
    compact_active = bool(
        solver.articulated_dense_response_mode in ("compact_cholesky", "compact_tree")
        and np.sum(compact_counts) > 0
    )

    result = RunResult(
        sweep=case.sweep,
        label=case.label,
        case_kind=case.case_kind,
        path=path,
        compact_active=compact_active,
        requested_world_count=case.world_count,
        free_pairs=case.free_pairs,
        articulations=case.articulations,
        links=case.links,
        contacts_per_articulation=case.contacts_per_articulation,
        world_count=model.world_count,
        body_count=model.body_count,
        joint_dof_count=model.joint_dof_count,
        dense_row_capacity=dense_capacity,
        mf_row_capacity=mf_capacity,
        compact_row_capacity=compact_capacity,
        dense_rows_total=int(np.sum(dense_counts)),
        dense_rows_max_world=int(np.max(dense_counts)) if dense_counts.size else 0,
        mf_rows_total=int(np.sum(mf_counts)),
        mf_rows_max_world=int(np.max(mf_counts)) if mf_counts.size else 0,
        compact_rows_total=int(np.sum(compact_counts)),
        compact_rows_max_world=int(np.max(compact_counts)) if compact_counts.size else 0,
        rigid_contacts=rigid_contacts,
        ms_per_step=float(ms_per_step),
        peak_gpu_mem_bytes=peak_gpu_mem_bytes,
        row_solver_mib=_row_solver_bytes(solver) / (1024.0 * 1024.0),
        propagation_extra_mib=_propagation_extra_bytes(solver) / (1024.0 * 1024.0),
    )
    return result, final_state


def _cases_for_preset(preset: str) -> list[BenchCase]:
    if preset == "smoke":
        return [
            BenchCase("free_free_dof", "free_pairs=2", "free_free", free_pairs=2),
            BenchCase(
                "articulated_contact_rows",
                "L=4,boxes=1",
                "articulated_free",
                articulations=1,
                links=4,
                contacts_per_articulation=1,
            ),
            BenchCase(
                "articulated_contact_rows",
                "L=4,boxes=2",
                "articulated_free",
                articulations=1,
                links=4,
                contacts_per_articulation=2,
            ),
            BenchCase("chain_depth", "L=2,boxes=2", "articulated_free", articulations=1, links=2, contacts_per_articulation=2),
            BenchCase("chain_depth", "L=4,boxes=4", "articulated_free", articulations=1, links=4, contacts_per_articulation=4),
            BenchCase("articulation_count", "A=1,L=4", "articulated_free", articulations=1, links=4, contacts_per_articulation=1),
            BenchCase("articulation_count", "A=2,L=4", "articulated_free", articulations=2, links=4, contacts_per_articulation=1),
        ]
    if preset == "default":
        return [
            BenchCase("free_free_dof", "free_pairs=2", "free_free", free_pairs=2),
            BenchCase("free_free_dof", "free_pairs=8", "free_free", free_pairs=8),
            BenchCase("free_free_dof", "free_pairs=21", "free_free", free_pairs=21),
            BenchCase(
                "articulated_contact_rows",
                "L=4,boxes=4",
                "articulated_free",
                articulations=1,
                links=4,
                contacts_per_articulation=4,
            ),
            BenchCase(
                "articulated_contact_rows",
                "L=16,boxes=8",
                "articulated_free",
                articulations=1,
                links=16,
                contacts_per_articulation=8,
            ),
            BenchCase(
                "articulated_contact_rows",
                "L=64,boxes=32",
                "articulated_free",
                articulations=1,
                links=64,
                contacts_per_articulation=32,
            ),
            BenchCase("chain_depth", "L=2,boxes=2", "articulated_free", articulations=1, links=2, contacts_per_articulation=2),
            BenchCase("chain_depth", "L=8,boxes=8", "articulated_free", articulations=1, links=8, contacts_per_articulation=8),
            BenchCase("chain_depth", "L=32,boxes=32", "articulated_free", articulations=1, links=32, contacts_per_articulation=32),
            BenchCase("chain_depth", "L=64,boxes=64", "articulated_free", articulations=1, links=64, contacts_per_articulation=64),
            BenchCase("articulation_count", "A=1,L=16", "articulated_free", articulations=1, links=16, contacts_per_articulation=1),
            BenchCase("articulation_count", "A=2,L=16", "articulated_free", articulations=2, links=16, contacts_per_articulation=1),
            BenchCase("articulation_count", "A=16,L=16", "articulated_free", articulations=16, links=16, contacts_per_articulation=1),
            BenchCase("articulation_count", "A=64,L=16", "articulated_free", articulations=64, links=16, contacts_per_articulation=1),
        ]
    if preset == "stress":
        cases: list[BenchCase] = []
        for free_pairs in (1, 2, 4, 8, 16, 21):
            cases.append(
                BenchCase(
                    "free_free_dof",
                    f"free_pairs={free_pairs}",
                    "free_free",
                    free_pairs=free_pairs,
                )
            )

        contact_row_cases = (
            (4, 4),
            (16, 8),
            (16, 16),
            (64, 8),
            (64, 32),
            (64, 64),
        )
        for links, contact_boxes in contact_row_cases:
            cases.append(
                BenchCase(
                    "articulated_contact_rows",
                    f"L={links},boxes={contact_boxes}",
                    "articulated_free",
                    articulations=1,
                    links=links,
                    contacts_per_articulation=contact_boxes,
                )
            )

        for links in (2, 16, 32, 64):
            cases.append(
                BenchCase(
                    "chain_depth",
                    f"L={links},boxes={links}",
                    "articulated_free",
                    articulations=1,
                    links=links,
                    contacts_per_articulation=links,
                )
            )

        for articulations in (1, 2, 16, 32, 64, 128):
            cases.append(
                BenchCase(
                    "articulation_count",
                    f"A={articulations},L=16",
                    "articulated_free",
                    articulations=articulations,
                    links=16,
                    contacts_per_articulation=1,
                )
            )
        return cases
    if preset == "env_scale":
        world_counts = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
        a16_l16_c64 = [
            BenchCase(
                "env_count",
                f"envs={world_count}",
                "articulated_free",
                world_count=world_count,
                articulations=16,
                links=16,
                contacts_per_articulation=1,
            )
            for world_count in world_counts
        ]
        a2_l16_c256 = [
            BenchCase(
                "env_count_a2_l16_c256",
                f"envs={world_count}",
                "articulated_free",
                world_count=world_count,
                articulations=2,
                links=16,
                contacts_per_articulation=32,
            )
            for world_count in world_counts
        ]
        a16_l4_c256 = [
            BenchCase(
                "env_count_a16_l4_c256",
                f"envs={world_count}",
                "articulated_free",
                world_count=world_count,
                articulations=16,
                links=4,
                contacts_per_articulation=4,
            )
            for world_count in world_counts
        ]
        a2_l4_c64 = [
            BenchCase(
                "env_count_a2_l4_c64",
                f"envs={world_count}",
                "articulated_free",
                world_count=world_count,
                articulations=2,
                links=4,
                contacts_per_articulation=8,
            )
            for world_count in world_counts
        ]
        a1_l4_c256 = [
            BenchCase(
                "env_count_a1_l4_c256",
                f"envs={world_count}",
                "articulated_free",
                world_count=world_count,
                articulations=1,
                links=4,
                contacts_per_articulation=64,
            )
            for world_count in world_counts
        ]
        return a16_l16_c64 + a2_l16_c256 + a16_l4_c256 + a2_l4_c64 + a1_l4_c256
    raise ValueError(f"unknown preset {preset!r}")


def _add_error_columns(results: list[RunResult], final_states: dict[tuple[str, str, str], dict[str, np.ndarray]]) -> None:
    for result in results:
        if result.case_kind == "free_free":
            # Free/free controls execute the same existing MF kernel in both
            # modes. At large contact counts that kernel is order-sensitive, so
            # state deltas across separate runs are reproducibility noise rather
            # than compact-method error.
            result.joint_q_rel_l2 = None
            result.joint_qd_rel_l2 = None
            result.joint_qd_abs_l2 = None
            result.joint_qd_abs_linf = None
            result.body_q_rel_l2 = None
            result.body_qd_rel_l2 = None
            result.body_qd_abs_l2 = None
            result.body_qd_abs_linf = None
            result.state_linf = None
            continue

        key = (result.sweep, result.label, result.case_kind)
        current = final_states.get((key[0], key[1], "mf_immediate"))
        if current is None:
            result.joint_q_rel_l2 = None
            result.joint_qd_rel_l2 = None
            result.joint_qd_abs_l2 = None
            result.joint_qd_abs_linf = None
            result.body_q_rel_l2 = None
            result.body_qd_rel_l2 = None
            result.body_qd_abs_l2 = None
            result.body_qd_abs_linf = None
            result.state_linf = None
            continue
        value = final_states[(key[0], key[1], result.path)]
        if result.path == "mf_immediate":
            result.joint_q_rel_l2 = 0.0
            result.joint_qd_rel_l2 = 0.0
            result.joint_qd_abs_l2 = 0.0
            result.joint_qd_abs_linf = 0.0
            result.body_q_rel_l2 = 0.0
            result.body_qd_rel_l2 = 0.0
            result.body_qd_abs_l2 = 0.0
            result.body_qd_abs_linf = 0.0
            result.state_linf = 0.0
        else:
            result.joint_q_rel_l2 = _finite_rel_l2(value["joint_q"], current["joint_q"])
            result.joint_qd_rel_l2 = _finite_rel_l2(value["joint_qd"], current["joint_qd"])
            result.joint_qd_abs_l2 = _abs_l2(value["joint_qd"], current["joint_qd"])
            result.joint_qd_abs_linf = _abs_linf(value["joint_qd"], current["joint_qd"])
            result.body_q_rel_l2 = _finite_rel_l2(value["body_q"], current["body_q"])
            result.body_qd_rel_l2 = _finite_rel_l2(value["body_qd"], current["body_qd"])
            result.body_qd_abs_l2 = _abs_l2(value["body_qd"], current["body_qd"])
            result.body_qd_abs_linf = _abs_linf(value["body_qd"], current["body_qd"])
            result.state_linf = _state_linf(value, current)


def _write_csv(path: Path, results: list[RunResult]) -> None:
    rows = [asdict(result) for result in results]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: float | None) -> str:
    if value is None:
        return ""
    if abs(value) >= 100.0:
        return f"{value:.1f}"
    if abs(value) >= 1.0:
        return f"{value:.3f}"
    return f"{value:.6g}"


def _write_summary(path: Path, results: list[RunResult], args: argparse.Namespace) -> None:
    lines: list[str] = []
    lines.append("# FPGS production articulated response benchmark")
    lines.append("")
    lines.append("This report uses real `SolverFeatherPGS.step` runs only. Compared paths:")
    lines.append("- `mf_immediate`: current production matrix-free path.")
    lines.append("- `compact_tree`: fixed-size articulated contact rows plus tree response/propagation.")
    lines.append("- `compact_cholesky`: fixed-size contact rows with a dense Cholesky response oracle for diagnostics.")
    lines.append("")
    lines.append(
        f"Config: preset `{args.preset}`, repeats {args.repeats}, warmups {args.warmups}, "
        f"dt {args.dt}, PGS iterations {args.pgs_iterations}, velocity iterations {args.pgs_velocity_iterations}, "
        f"contact friction {'off' if args.no_friction else 'on'}, "
        f"contact_friction_position_iterations={args.contact_friction_position_iterations}, "
        f"`hinv_jt_kernel=par_row`, "
        f"compact_fast_body_map={args.compact_fast_body_map}, "
        f"compact_existing_row_phases={args.compact_existing_row_phases}, "
        f"compact_shared_row_solver={args.compact_shared_row_solver}, "
        f"compact_warp_propagation={args.compact_warp_propagation}, "
        f"compact_max_constraints={args.compact_max_constraints}, "
        f"compact_path={args.compact_path}, "
        f"joint_armature={args.joint_armature}."
    )
    lines.append("")
    lines.append(
        "`dense_row_capacity`/`mf_row_capacity`/`compact_row_capacity` are configured capacities. "
        "`dense_rows_total`/`mf_rows_total`/`compact_rows_total` are produced by the real contact/row setup. "
        "`peak GPU MB` is total CUDA memory in use around solver allocation and stepping. "
        "Contacts are generated once per case and reused for both solver paths so path comparisons see identical contact arrays."
    )
    lines.append(
        "State-error columns are omitted for free/free controls because compact rows are inactive there and both rows execute "
        "the same order-sensitive MF kernel; articulated/free rows report error against `mf_immediate`."
    )
    if args.preset == "stress":
        lines.append(
            "The 128-link chain-depth case is omitted for now: the current tiled Cholesky factorization exceeds "
            "shared memory on this GPU for a 128-DOF articulation group (OOSM)."
        )
        lines.append(
            "The `chain_depth` sweep uses one contact box per link (`boxes=L`), so both scalar contact rows and "
            "articulation DOF grow with chain depth."
        )

    for sweep in dict.fromkeys(result.sweep for result in results):
        lines.append("")
        lines.append(f"## {sweep}")
        lines.append("")
        lines.append(
            "| case | path | D | cap dense/mf/compact | rows dense/mf/compact | contacts | ms/step | peak GPU MB | qd rel L2 | qd abs Linf | state Linf |"
        )
        lines.append(
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        )
        for result in results:
            if result.sweep != sweep:
                continue
            lines.append(
                "| "
                + " | ".join(
                    [
                        result.label,
                        result.path,
                        str(result.joint_dof_count),
                        f"{result.dense_row_capacity}/{result.mf_row_capacity}/{result.compact_row_capacity}",
                        f"{result.dense_rows_total}/{result.mf_rows_total}/{result.compact_rows_total}",
                        str(result.rigid_contacts),
                        _fmt(result.ms_per_step),
                        _fmt(result.peak_gpu_mem_bytes / 1.0e6 if result.peak_gpu_mem_bytes else None),
                        _fmt(result.joint_qd_rel_l2),
                        _fmt(result.joint_qd_abs_linf),
                        _fmt(result.state_linf),
                    ]
                )
                + " |"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plot(path: Path, results: list[RunResult]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    sweeps = list(dict.fromkeys(result.sweep for result in results))
    fig, axes = plt.subplots(len(sweeps), 2, figsize=(13, 4 * len(sweeps)), squeeze=False)

    for row_idx, sweep in enumerate(sweeps):
        sweep_results = [result for result in results if result.sweep == sweep]
        labels = list(dict.fromkeys(result.label for result in sweep_results))
        x = np.arange(len(labels))
        width = 0.35
        paths = list(dict.fromkeys(result.path for result in sweep_results))
        width = min(0.8 / max(len(paths), 1), 0.35)
        for path_idx, path_name in enumerate(paths):
            path_results = {
                result.label: result for result in sweep_results if result.path == path_name
            }
            values = [
                path_results[label].ms_per_step if label in path_results else math.nan for label in labels
            ]
            offset = (path_idx - 0.5 * (len(paths) - 1)) * width
            axes[row_idx][0].bar(x + offset, values, width=width, label=path_name)
            mem = [
                path_results[label].peak_gpu_mem_bytes / 1.0e6
                if label in path_results and path_results[label].peak_gpu_mem_bytes
                else math.nan
                for label in labels
            ]
            axes[row_idx][1].bar(x + offset, mem, width=width, label=path_name)

        axes[row_idx][0].set_title(f"{sweep}: time")
        axes[row_idx][0].set_ylabel("ms / step")
        axes[row_idx][1].set_title(f"{sweep}: total GPU memory")
        axes[row_idx][1].set_ylabel("MB")
        for axis in axes[row_idx]:
            axis.set_xticks(x)
            axis.set_xticklabels(labels, rotation=30, ha="right")
            axis.legend()
            axis.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("smoke", "default", "stress", "env_scale"), default="default")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--warmups", type=int, default=None)
    parser.add_argument("--dt", type=float, default=1.0 / 240.0)
    parser.add_argument("--pgs-iterations", type=int, default=None)
    parser.add_argument("--pgs-velocity-iterations", type=int, default=0)
    parser.add_argument("--contact-friction-position-iterations", type=int, default=-1)
    parser.add_argument("--joint-armature", type=float, default=0.0)
    parser.add_argument(
        "--compact-path",
        choices=COMPACT_PATHS,
        default="compact_tree",
        help="Compact path to compare against mf_immediate.",
    )
    parser.add_argument(
        "--path",
        action="append",
        choices=("mf_immediate", *COMPACT_PATHS),
        default=None,
        help="Run only selected path(s). Defaults to mf_immediate plus --compact-path.",
    )
    parser.add_argument("--no-friction", action="store_true")
    parser.add_argument("--compact-fast-body-map", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--compact-existing-row-phases",
        choices=("auto", "always", "skip"),
        default="skip",
        help="Temporary compact-tree profiling control. Use 'skip' only for known compact-only scenes.",
    )
    parser.add_argument("--compact-shared-row-solver", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compact-warp-propagation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--compact-max-constraints",
        type=int,
        default=None,
        help="Override compact_max_constraints for capacity-sensitivity profiling.",
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help=(
            "Run only matching cases. Use either an exact label, e.g. 'A=64,L=16', "
            "or 'sweep|label', e.g. 'articulation_count|A=64,L=16'. May be repeated."
        ),
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wp.init()
    device = wp.get_device(args.device)
    if not device.is_cuda:
        raise RuntimeError("This benchmark is intended for CUDA devices")

    if args.out_dir is None:
        args.out_dir = Path("artifacts/fpgs_articulation_row_scaling") / args.preset
    if args.repeats is None:
        args.repeats = 1 if args.preset == "smoke" else 2 if args.preset == "env_scale" else 4 if args.preset == "stress" else 8
    if args.warmups is None:
        args.warmups = 1 if args.preset in ("smoke", "env_scale") else 2
    if args.pgs_iterations is None:
        args.pgs_iterations = 2 if args.preset == "smoke" else 8

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cases = _cases_for_preset(args.preset)
    if args.joint_armature != 0.0:
        cases = [
            replace(case, joint_armature=args.joint_armature)
            if case.case_kind == "articulated_free"
            else case
            for case in cases
        ]
    if args.case:
        requested = set(args.case)
        cases = [
            case
            for case in cases
            if case.label in requested or f"{case.sweep}|{case.label}" in requested
        ]
        if not cases:
            raise ValueError(f"--case filters did not match any {args.preset!r} cases: {sorted(requested)}")

    results: list[RunResult] = []
    final_states: dict[tuple[str, str, str], dict[str, np.ndarray]] = {}
    paths = tuple(dict.fromkeys(args.path)) if args.path else ("mf_immediate", args.compact_path)
    for case in cases:
        model, initial, contacts = _prepare_case_run(case, args.device)
        for path_name in paths:
            result, final_state = _run_case_path(
                case,
                path_name,
                device=args.device,
                repeats=args.repeats,
                warmups=args.warmups,
                dt=args.dt,
                pgs_iterations=args.pgs_iterations,
                pgs_velocity_iterations=args.pgs_velocity_iterations,
                enable_contact_friction=not args.no_friction,
                contact_friction_position_iterations=args.contact_friction_position_iterations,
                compact_fast_body_map=args.compact_fast_body_map,
                compact_existing_row_phases=args.compact_existing_row_phases,
                compact_shared_row_solver=args.compact_shared_row_solver,
                compact_warp_propagation=args.compact_warp_propagation,
                compact_max_constraints_override=args.compact_max_constraints,
                model=model,
                initial=initial,
                contacts=contacts,
            )
            results.append(result)
            final_states[(case.sweep, case.label, path_name)] = final_state

    _add_error_columns(results, final_states)
    rows = [asdict(result) for result in results]
    (args.out_dir / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    _write_csv(args.out_dir / "results.csv", results)
    _write_summary(args.out_dir / "summary.md", results, args)
    if not args.no_plots:
        _write_plot(args.out_dir / "fpgs_articulation_row_scaling.png", results)

    print(f"Wrote {args.out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
