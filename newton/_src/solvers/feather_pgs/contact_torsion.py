# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Bounded, local-only spin resistance with an explicit effective radius.

No material radius is enabled implicitly. This diagnostic adds one angular
row per coherent exact-shape-pair contact group, not one full torque budget
per witness. It uses the same articulated response and bilateral projection
as existing dense rows. All state is rebuilt per step; there is no anchor or
elastic orientation memory. Host grouping precludes graphs and throughput use.
"""

import re
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import warp as wp

from ...geometry import GeoType

_TOUCH_TOLERANCE = 1.0e-5
_NORMAL_COSINE = 0.999
_MAX_GROUP_CONTACTS = 4096


def configure_contact_torsion(solver, radius, indices, patterns):
    """Validate the experimental opt-in and resolve stable shape-name selectors."""
    radius = float(radius)
    if not np.isfinite(radius) or radius < 0:
        raise ValueError("contact_torsion_radius must be finite and non-negative [m]")
    if indices is not None and patterns is not None:
        raise ValueError("Use either contact_torsion_shape_indices or contact_torsion_shape_patterns")
    selected = None
    if indices is not None:
        indices = tuple(indices)
        if any(
            not isinstance(i, Integral) or isinstance(i, bool) or i < 0 or i >= solver.model.shape_count
            for i in indices
        ):
            raise ValueError("contact_torsion_shape_indices contains invalid global shape indices")
        indices = tuple(int(i) for i in indices)
        selected = frozenset(indices)
    if patterns is not None:
        if isinstance(patterns, str):
            raise ValueError("contact_torsion_shape_patterns must be a sequence, not a string")
        patterns = tuple(patterns)
        selected = set()
        for pattern in patterns:
            if not isinstance(pattern, str):
                raise ValueError("contact_torsion_shape_patterns entries must be strings")
            try:
                regex = re.compile(pattern)
            except re.error as error:
                raise ValueError("Invalid contact_torsion_shape_patterns regex") from error
            matches = {i for i, label in enumerate(solver.model.shape_label) if regex.fullmatch(label)}
            if not matches:
                raise ValueError(f"contact_torsion_shape_patterns matched no shapes: {pattern!r}")
            selected.update(matches)
        selected = frozenset(selected)
    solver.contact_torsion_radius = radius
    solver.contact_torsion_shape_indices = indices
    solver.contact_torsion_shape_patterns = patterns
    solver._contact_torsion_shape_set = selected
    solver._torsion_stats = {}
    if radius > 0 and (
        not solver.model.device.is_cuda
        or solver.model.requires_grad
        or solver.pgs_mode != "matrix_free"
        or solver.articulated_contact_response != "immediate"
        or solver.friction_mode != "current"
        or solver.pgs_schedule != "interleaved"
        or solver.pgs_velocity_iterations
        or solver.pgs_warmstart
        or solver.pgs_debug
        or solver.pgs_contact_regularization != 0.0
        or not solver.enable_contact_friction
    ):
        raise ValueError(
            "Contact torsion requires CUDA non-differentiable matrix_free/immediate/current/"
            "interleaved, without warmstart, regularization, debug, or velocity post-passes"
        )


@dataclass
class _Witness:
    """Represent one currently touching, friction-enabled dense contact."""

    world: int
    slot: int
    shape_a: int
    shape_b: int
    body_a: int
    body_b: int
    normal: np.ndarray
    point: np.ndarray
    gap: float


def _transform_point(pose, point):
    """Transform a body-frame witness using scalar-last quaternion storage."""
    q = pose[3:]
    cross = 2.0 * np.cross(q[:3], point)
    return point + q[3] * cross + np.cross(q[:3], cross) + pose[:3]


def _contact_groups(solver, state, contacts):
    """Group compatible current witnesses without anchor or impulse history."""
    count = int(contacts.rigid_contact_count.numpy()[0])
    if count > contacts.rigid_contact_max:
        raise RuntimeError("Contact input overflow before contact torsion")
    stiffness = contacts.rigid_contact_stiffness
    if stiffness is not None and np.any(stiffness.numpy()[:count] > 0):
        raise ValueError("Contact torsion does not support hydroelastic contacts")
    paths = solver.contact_path.numpy()[:count]
    slots = solver.contact_slot.numpy()[:count]
    worlds = solver.contact_world.numpy()[:count]
    shape_a = contacts.rigid_contact_shape0.numpy()[:count]
    shape_b = contacts.rigid_contact_shape1.numpy()[:count]
    normals = -contacts.rigid_contact_normal.numpy()[:count]
    points_a = contacts.rigid_contact_point0.numpy()[:count]
    points_b = contacts.rigid_contact_point1.numpy()[:count]
    margins_a = contacts.rigid_contact_margin0.numpy()[:count]
    margins_b = contacts.rigid_contact_margin1.numpy()[:count]
    shape_body = solver.model.shape_body.numpy()
    shape_types = solver.model.shape_type.numpy()
    poses = state.body_q.numpy()
    row_types = solver.row_type.numpy()
    parents = solver.row_parent.numpy()
    supported = (
        GeoType.SPHERE,
        GeoType.BOX,
        GeoType.CAPSULE,
        GeoType.CYLINDER,
        GeoType.CONE,
        GeoType.ELLIPSOID,
        GeoType.PLANE,
        GeoType.CONVEX_MESH,
    )
    groups = {}
    selected = solver._contact_torsion_shape_set
    admitted = 0
    for c in range(count):
        a, b = int(shape_a[c]), int(shape_b[c])
        world, slot = int(worlds[c]), int(slots[c])
        if a < 0 or b < 0 or (selected is not None and a not in selected and b not in selected):
            continue
        if paths[c] != 0 or slot < 0:
            continue
        if shape_types[a] not in supported or shape_types[b] not in supported:
            continue
        if slot + 2 >= row_types.shape[1] or row_types[world, slot + 1] != 2 or parents[world, slot + 1] != slot:
            continue
        ba, bb = int(shape_body[a]), int(shape_body[b])
        pa, pb = points_a[c].copy(), points_b[c].copy()
        if ba >= 0:
            pa = _transform_point(poses[ba], pa)
        if bb >= 0:
            pb = _transform_point(poses[bb], pb)
        pa -= margins_a[c] * normals[c]
        pb += margins_b[c] * normals[c]
        gap = float(np.dot(normals[c], pa - pb))
        if gap > _TOUCH_TOLERANCE:
            continue
        admitted += 1
        if admitted > _MAX_GROUP_CONTACTS:
            raise RuntimeError("Contact torsion host grouping capacity exceeded; refusing dropped work")
        witness = _Witness(world, slot, a, b, ba, bb, normals[c], (pa + pb) * 0.5, gap)
        clusters = groups.setdefault((world, a, b, ba, bb), [])
        for cluster in clusters:
            if all(
                np.dot(witness.normal, other.normal) >= _NORMAL_COSINE
                and abs(np.dot(witness.normal, witness.point - other.point)) <= _TOUCH_TOLERANCE
                for other in cluster
            ):
                cluster.append(witness)
                break
        else:
            clusters.append([witness])
    return [cluster for clusters in groups.values() for cluster in clusters]


def prepare_torsion_rows(solver, state, augmented_state, contacts):
    """Append current touching-group angular rows before H-inverse/J response."""
    if getattr(solver, "contact_compliance", False):
        raise ValueError("Contact torsion combined with contact compliance is not supported")
    if wp.get_stream(solver.model.device).is_capturing:
        raise RuntimeError("Experimental torsion host grouping does not support CUDA graph capture")
    solver._torsion_stats = {"rows": 0, "groups": []}
    if contacts is None:
        return
    groups = _contact_groups(solver, state, contacts)
    count = solver.constraint_count.numpy()
    raw_count = solver.slot_counter.numpy()
    if np.any(raw_count > solver.dense_max_constraints):
        raise RuntimeError("Dense input overflow before experimental torsion; refusing lost rows")
    fields = {
        key: getattr(solver, key).numpy()
        for key in (
            "row_type",
            "row_parent",
            "row_mu",
            "row_beta",
            "row_cfm",
            "phi",
            "target_velocity",
            "row_restitution",
        )
    }
    jacobians = {size: array.numpy() for size, array in solver.J_by_size.items()}
    body_joint = solver.body_to_joint.numpy()
    body_art = solver.body_to_articulation.numpy()
    ancestor = solver.model.joint_ancestor.numpy()
    qd_start = solver.model.joint_qd_start.numpy()
    art_start = solver.articulation_dof_start.numpy()
    art_size = solver.articulation_response_dof_count.numpy()
    art_group = solver.art_group_idx.numpy()
    motions = augmented_state.joint_S_s.numpy()
    prescribed = solver._prescribed_articulation.numpy()
    body_velocities = augmented_state.body_v_s.numpy()
    for group in groups:
        world = group[0].world
        coefficient = float(fields["row_mu"][world, group[0].slot + 1])
        if coefficient <= 0:
            continue
        row = int(count[world])
        if row >= solver.dense_max_constraints:
            raise RuntimeError("Experimental torsion row capacity exceeded; refusing dropped work")
        count[world] += 1
        for field in fields.values():
            field[world, row] = 0
        fields["row_type"][world, row] = 7
        fields["row_parent"][world, row] = group[0].slot
        fields["row_mu"][world, row] = solver.contact_torsion_radius
        for witness in group:
            fields["row_parent"][world, witness.slot] = row
        normal = group[0].normal
        for body, sign in ((group[0].body_a, 1.0), (group[0].body_b, -1.0)):
            if body < 0:
                continue
            art = int(body_art[body])
            if art < 0:
                continue
            if prescribed[art]:
                fields["target_velocity"][world, row] -= sign * np.dot(normal, body_velocities[body, 3:])
                continue
            size, index = int(art_size[art]), int(art_group[art])
            joint = int(body_joint[body])
            while joint >= 0:
                for global_dof in range(int(qd_start[joint]), int(qd_start[joint + 1])):
                    local = global_dof - int(art_start[art])
                    if 0 <= local < size:
                        jacobians[size][index, row, local] += sign * np.dot(normal, motions[global_dof, 3:])
                joint = int(ancestor[joint])
        solver._torsion_stats["groups"].append(
            {
                "world": int(world),
                "row": row,
                "normal_rows": [w.slot for w in group],
                "shape_pair": [group[0].shape_a, group[0].shape_b],
                "effective_radius_m": solver.contact_torsion_radius,
                "mu": coefficient,
                "gap_max_m": max(w.gap for w in group),
            }
        )
    solver._torsion_stats["rows"] = len(solver._torsion_stats["groups"])
    if not solver._torsion_stats["rows"]:
        return
    for name, values in fields.items():
        getattr(solver, name).assign(values)
    for size, values in jacobians.items():
        solver.J_by_size[size].assign(values)
    solver.constraint_count.assign(count)
    solver.slot_counter.assign(count)


def torque_sweep_source(dofs):
    """Return the coupled residual-budget angular solve inside each PGS sweep."""
    return f"""
        if (row_phase == 0 || row_phase == 1 || row_phase == 4) {{
            for (int spin = 0; spin < m_dense; ++spin) {{
                if (world_row_type.data[off_dense + spin] != 7) continue;
                float radius = world_row_mu.data[off_dense + spin];
                float normal_budget = 0.0f;
                float sliding_used = 0.0f;
                for (int n = 0; n + 2 < m_dense; ++n) {{
                    if (world_row_type.data[off_dense + n] != 0 ||
                        world_row_parent.data[off_dense + n] != spin) continue;
                    float mu = fmaxf(world_row_mu.data[off_dense + n + 1], 0.0f);
                    normal_budget += mu * fmaxf(s_lam_dense[n], 0.0f);
                    float a = s_lam_dense[n + 1], b = s_lam_dense[n + 2];
                    sliding_used += sqrtf(a * a + b * b);
                }}
                float bound = radius * fmaxf(normal_budget - sliding_used, 0.0f);
                if (global_iter < friction_start_iteration) bound = 0.0f;
                float dot = 0.0f;
                for (int d = lane; d < {dofs}; d += 32)
                    dot += J_world.data[jy_world_base + spin * {dofs} + d] * s_v[d];
                dot += __shfl_down_sync(MASK, dot, 16);
                dot += __shfl_down_sync(MASK, dot, 8);
                dot += __shfl_down_sync(MASK, dot, 4);
                dot += __shfl_down_sync(MASK, dot, 2);
                dot += __shfl_down_sync(MASK, dot, 1);
                float residual = __shfl_sync(MASK, dot, 0) + rhs_bias.data[off_dense + spin];
                float old = s_lam_dense[spin];
                float diagonal = world_diag.data[off_dense + spin];
                float trial = diagonal > 0.0f ? old - residual / diagonal : 0.0f;
                float next = fminf(fmaxf(trial, -bound), bound);
                float delta = next - old;
                if (delta != 0.0f) {{
                    iteration_changed = 1;
                    if (lane == 0) s_lam_dense[spin] = next;
                    for (int d = lane; d < {dofs}; d += 32)
                        s_v[d] += Y_world.data[jy_world_base + spin * {dofs} + d] * delta;
                }}
                __syncwarp();
            }}
        }}
"""
