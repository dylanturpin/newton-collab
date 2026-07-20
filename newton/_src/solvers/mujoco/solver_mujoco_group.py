# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
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

"""Grouped MuJoCo solver for structurally heterogeneous multi-world models.

``mujoco_warp`` batches one MuJoCo model across ``nworld`` data rows, which
requires every world to be structurally identical. A heterogeneous Newton
model (different articulations in different worlds) cannot be simulated by a
single :class:`SolverMuJoCo`. It can, however, be partitioned into groups of
structurally identical worlds — worlds never interact, so simulating each
group with its own :class:`SolverMuJoCo` over a world subset is exact.

:class:`SolverMuJoCoGroup` performs that partition and fans the solver API out
to one child solver per group. All children share the same Newton model,
states, and control arrays; each child reads and writes only the flat-array
segments belonging to its worlds.
"""

from __future__ import annotations

import os
import warnings

import numpy as np

import warp as wp

from ...core.types import MAXVAL, override
from ...geometry import ShapeFlags
from ..coupled.interface import CouplingInterface
from ..solver import SolverBase
from .solver_mujoco import SolverMuJoCo


def compute_structural_world_groups(model, *, include_shapes: bool = True) -> list[list[int]]:
    """Partition model worlds into groups of structurally identical worlds.

    Two worlds belong to the same group when they have identical body, joint,
    equality-constraint and mimic-constraint counts and identical joint type
    sequences — and, when :paramref:`include_shapes` is set, identical shape
    counts and shape-type sequences. Worlds without any body are skipped
    (there is nothing to simulate; static shapes live in the global world).

    Args:
        model: The Newton model to partition.
        include_shapes: Whether collision-shape census is part of the
            structural key. Shapes only shape the MuJoCo solve when the solver
            runs MuJoCo's internal collision pipeline; with Newton-pipeline
            contacts (``use_mujoco_contacts=False``) worlds differing only in
            collision shells can share one solver and this should be
            ``False``.

    Returns:
        List of world-index groups, each sorted ascending, ordered by first
        occurrence.
    """
    world_count = model.world_count
    if world_count <= 1:
        return [[0]] if world_count == 1 else []

    body_world = model.body_world.numpy()
    joint_world = model.joint_world.numpy()
    shape_world = model.shape_world.numpy()
    joint_type = model.joint_type.numpy()
    shape_type = model.shape_type.numpy()
    mujoco_attrs = getattr(model, "mujoco", None)
    eq_world = None
    if mujoco_attrs is not None and getattr(mujoco_attrs, "equality_constraint_count", 0) > 0:
        eq_world = mujoco_attrs.equality_constraint_world.numpy()
    mimic_world = model.constraint_mimic_world.numpy() if model.constraint_mimic_count > 0 else None

    shape_body = model.shape_body.numpy() if not include_shapes else None
    shape_flags = model.shape_flags.numpy() if not include_shapes else None

    groups: dict[tuple, list[int]] = {}
    for w in range(world_count):
        body_ids = np.flatnonzero(body_world == w)
        if len(body_ids) == 0:
            continue
        joint_ids = np.flatnonzero(joint_world == w)
        shape_ids = np.flatnonzero(shape_world == w)
        if include_shapes:
            shape_key: tuple = (len(shape_ids), shape_type[shape_ids].tobytes())
        else:
            # Shape census may diverge (MuJoCo does not collide in this
            # mode), but injected contacts still resolve their geom through
            # the shape's body: every body that carries shapes in one world
            # must carry shapes in all of them, so the per-body shape
            # PRESENCE pattern stays part of the key.
            colliding = shape_ids[(shape_flags[shape_ids] & int(ShapeFlags.COLLIDE_SHAPES)) != 0]
            has_shape = np.zeros(len(body_ids), dtype=bool)
            if len(colliding):
                has_shape[np.searchsorted(body_ids, shape_body[colliding])] = True
            shape_key = (-1, has_shape.tobytes())
        key = (
            len(body_ids),
            len(joint_ids),
            *shape_key,
            joint_type[joint_ids].tobytes(),
            int(np.count_nonzero(eq_world == w)) if eq_world is not None else 0,
            int(np.count_nonzero(mimic_world == w)) if mimic_world is not None else 0,
        )
        groups.setdefault(key, []).append(w)
    return list(groups.values())


# Classification of per-entity Model value arrays for cross-world transport in
# :class:`SolverMuJoCo`. "Transported" values are expanded to ``[nworld, ...]``
# mjw-model fields and written per world by the ``notify_model_changed`` sync
# kernels at construction; worlds in one group may diverge in them freely.
_TRANSPORTED_VALUES = {
    "body_mass",
    "body_inv_mass",
    "body_inertia",
    "body_inv_inertia",
    "body_com",
    "joint_X_p",
    "joint_X_c",
    "joint_axis",
    "joint_limit_lower",
    "joint_limit_upper",
    "joint_limit_ke",
    "joint_limit_kd",
    "joint_target_ke",
    "joint_target_kd",
    "joint_armature",
    "joint_friction",
    "joint_effort_limit",
}
# State and control values: synced per world every step (or at reset), never
# read from the template.
_STATE_VALUES = {
    "body_q",
    "body_qd",
    "joint_q",
    "joint_qd",
    "joint_f",
    "joint_target_pos",
    "joint_target_vel",
    "joint_target_q",
    "joint_target_qd",
}
# Attributes SolverMuJoCo documents as unsupported: divergence never changes
# this solver's behavior.
_UNSUPPORTED_VALUES = {"joint_velocity_limit", "joint_enabled"}
# Divergence here is semi-structural: the template decides which MuJoCo
# actuators exist, so a per-world mode change cannot be transported.
_SEMI_STRUCTURAL_VALUES = {"joint_target_mode"}
# Index/topology arrays are covered by the structural fingerprint; internal
# acceleration structures are never consumed by MuJoCo.
_SKIP_NAME_PREFIXES = ("bvh_",)
_SKIP_NAME_SUFFIXES = ("_start", "_range", "_world", "_key", "_dim")
_INDEX_VALUES = {"joint_parent", "joint_child", "joint_ancestor", "joint_articulation", "shape_body", "shape_flags"}


def audit_group_value_transport(model, world_groups: list[list[int]], *, use_mujoco_contacts: bool = False):
    """Audit whether per-world value differences within groups reach the solver.

    Each group runs one template :class:`SolverMuJoCo` model; member worlds may
    diverge from the template only in values the solver transports per world.
    This audit compares every per-entity numeric Model array across each
    group's worlds and classifies the differences:

    - ``ok``: transported per world (or state/solver-unsupported) — safe.
    - ``veto``: the divergence is silently collapsed to the template's value
      (e.g. joint limited-ness when the template joint is unlimited, geom
      bounding radii under MuJoCo contacts, semi-structural actuator modes).
    - ``warn``: an array this audit cannot classify differs; transport is
      unverified.

    Args:
        model: The Newton model the groups partition.
        world_groups: World-index groups, each group's first world being the
            template (matching :class:`SolverMuJoCo`'s ``worlds`` semantics).
        use_mujoco_contacts: Whether the solvers run MuJoCo's internal
            collision pipeline. Shape values are irrelevant to the MuJoCo
            solve when ``False``; when ``True`` they are transported except
            for the template-derived ``geom_rbound``.

    Returns:
        List of finding dicts with keys ``group``, ``template``, ``world``,
        ``field``, ``verdict`` (``ok``/``warn``/``veto``), ``detail``.
    """
    body_world = model.body_world.numpy()
    joint_world = model.joint_world.numpy()
    shape_world = model.shape_world.numpy()
    dof_world = np.repeat(joint_world, np.diff(model.joint_qd_start.numpy()))
    coord_world = np.repeat(joint_world, np.diff(model.joint_q_start.numpy()))
    entity_worlds: dict[int, np.ndarray] = {}
    for count, wmap in (
        (model.body_count, body_world),
        (model.joint_count, joint_world),
        (model.shape_count, shape_world),
        (model.joint_dof_count, dof_world),
        (model.joint_coord_count, coord_world),
    ):
        entity_worlds.setdefault(count, wmap)

    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in dir(model):
        if name.startswith("_") or name.startswith(_SKIP_NAME_PREFIXES) or name.endswith(_SKIP_NAME_SUFFIXES):
            continue
        if name in _INDEX_VALUES or name in _STATE_VALUES:
            continue
        try:
            # dir() sweeps deprecated alias properties too; touching them must
            # not surface their deprecation warnings to the caller.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                value = getattr(model, name)
        except (AttributeError, RuntimeError):
            continue
        if not isinstance(value, wp.array) or value.ndim < 1 or len(value) not in entity_worlds or len(value) == 0:
            continue
        host = value.numpy()
        if host.dtype.kind not in "fiu?":
            continue
        arrays[name] = (host, entity_worlds[len(value)])

    findings = []

    def add(group_index, template, world, field, verdict, detail):
        findings.append(
            {
                "group": group_index,
                "template": int(template),
                "world": int(world),
                "field": field,
                "verdict": verdict,
                "detail": detail,
            }
        )

    limit_lower, lower_wmap = arrays.get("joint_limit_lower", (None, None))
    limit_upper, _ = arrays.get("joint_limit_upper", (None, None))

    for group_index, group in enumerate(world_groups):
        template = group[0]
        for world in group[1:]:
            for name, (host, wmap) in arrays.items():
                t_ids = np.flatnonzero(wmap == template)
                w_ids = np.flatnonzero(wmap == world)
                if len(t_ids) != len(w_ids):
                    if name.startswith("shape_") and not use_mujoco_contacts:
                        add(group_index, template, world, name, "ok", "shape census diverges; unused by MuJoCo solve")
                    else:
                        add(group_index, template, world, name, "veto", "entity count mismatch")
                    continue
                a, b = host[t_ids], host[w_ids]
                if a.dtype.kind == "f":
                    # matching infinities (limit sentinels) subtract to nan; treat as equal
                    with np.errstate(invalid="ignore"):
                        differs = bool(np.any(np.abs(a.astype(np.float64) - b.astype(np.float64)) > 1e-9))
                else:
                    differs = bool(np.any(a != b))
                if not differs:
                    continue
                if name.startswith("shape_"):
                    if not use_mujoco_contacts:
                        add(group_index, template, world, name, "ok", "shape values unused by MuJoCo solve")
                    elif name in ("shape_scale", "shape_transform"):
                        add(
                            group_index,
                            template,
                            world,
                            name,
                            "veto",
                            "geom_rbound is derived from the template geometry and not re-synced",
                        )
                    else:
                        add(group_index, template, world, name, "ok", "transported via geom properties sync")
                elif name in _TRANSPORTED_VALUES:
                    add(group_index, template, world, name, "ok", "transported per world at solver construction")
                elif name in _UNSUPPORTED_VALUES:
                    add(group_index, template, world, name, "ok", "attribute unsupported by SolverMuJoCo")
                elif name in _SEMI_STRUCTURAL_VALUES:
                    add(
                        group_index,
                        template,
                        world,
                        name,
                        "veto",
                        "actuator existence is decided by the template world",
                    )
                else:
                    add(group_index, template, world, name, "warn", "transport unverified for this array")
            # Limited-ness is baked from the template: a member with finite
            # limits under an unlimited template joint silently loses them.
            if limit_lower is not None:
                t_ids = np.flatnonzero(lower_wmap == template)
                w_ids = np.flatnonzero(lower_wmap == world)
                if len(t_ids) == len(w_ids):
                    template_unlimited = (limit_lower[t_ids] <= -MAXVAL) & (limit_upper[t_ids] >= MAXVAL)
                    member_limited = ~((limit_lower[w_ids] <= -MAXVAL) & (limit_upper[w_ids] >= MAXVAL))
                    lost = np.flatnonzero(template_unlimited & member_limited)
                    if len(lost):
                        add(
                            group_index,
                            template,
                            world,
                            "joint_limit_lower/upper",
                            "veto",
                            f"{len(lost)} dof(s) have finite limits but the template joint is unlimited;"
                            " jnt_limited is baked from the template and the member limits are ignored",
                        )
    return findings


class SolverMuJoCoGroup(SolverBase, CouplingInterface):
    """One :class:`SolverMuJoCo` per structurally-identical world group.

    Presents the solver interface used by callers of :class:`SolverMuJoCo`
    (``step``, ``reset``, ``notify_model_changed``, ``get_max_contact_count``)
    and fans each call out to the child solvers. Worlds never interact, so
    sequential child stepping is exact; children share the model's flat state
    arrays and write disjoint segments.

    Limitations (raise :class:`NotImplementedError`): particle/cloth coupling
    hooks and MuJoCo-data contact reporting (``update_contacts``), both of
    which would need cross-child merging.
    """

    def __init__(self, model, world_groups: list[list[int]] | None = None, **kwargs):
        """
        Args:
            model: The (possibly structurally heterogeneous) multi-world model.
            world_groups: Optional explicit partition of world indices. When
                ``None``, worlds are grouped by structural signature via
                :func:`compute_structural_world_groups`.
            **kwargs: Forwarded to every child :class:`SolverMuJoCo`.
        """
        super().__init__(model)
        if world_groups is None:
            world_groups = compute_structural_world_groups(
                model, include_shapes=bool(kwargs.get("use_mujoco_contacts", False))
            )
        if not world_groups:
            raise ValueError("SolverMuJoCoGroup: no non-empty worlds to simulate.")
        if kwargs.get("use_mujoco_cpu"):
            raise ValueError("SolverMuJoCoGroup requires the GPU (mujoco_warp) backend.")
        self.world_groups = [sorted(int(w) for w in group) for group in world_groups]
        audit = audit_group_value_transport(
            model, self.world_groups, use_mujoco_contacts=bool(kwargs.get("use_mujoco_contacts", False))
        )
        for finding in audit:
            if finding["verdict"] == "veto":
                warnings.warn(
                    f"SolverMuJoCoGroup: world {finding['world']} diverges from template world"
                    f" {finding['template']} in '{finding['field']}' but the difference does not"
                    f" reach the solver ({finding['detail']}); world {finding['world']} will"
                    " silently use the template's value.",
                    UserWarning,
                    stacklevel=2,
                )
        self.solvers = [SolverMuJoCo(model, worlds=group, **kwargs) for group in self.world_groups]
        # One stream per child group: the children write disjoint per-world
        # rows of the shared state, so their solves are independent and can
        # overlap. Events fork the children off the caller's stream and join
        # them back, which is also the CUDA-graph-capture-compatible pattern
        # (capture follows event edges into parallel graph branches).
        self._device = model.device
        streams_enabled = os.environ.get("NEWTON_GROUP_STREAMS", "1") != "0"
        if streams_enabled and wp.get_device(self._device).is_cuda and len(self.solvers) > 1:
            self._streams = [wp.Stream(self._device) for _ in self.solvers]
            self._fork_event = wp.Event(self._device)
            self._join_events = [wp.Event(self._device) for _ in self.solvers]
        else:
            self._streams = None
        # Per-child step graphs captured lazily on the first eager step (the
        # embedder's capture warmup). The parallel path launches these on the
        # child streams: graph launches are capture-legal (recorded as child
        # nodes), so an embedder capturing the whole step records the fan-out
        # — whereas capturing raw mjwarp solves on forked streams is not
        # possible (capture_while pauses capture, legal only on its origin
        # stream).
        self._child_graphs = None

    @property
    def use_mujoco_cpu(self) -> bool:
        return False

    @property
    def steps_with_stream_fanout(self) -> bool:
        """``True`` when :meth:`step` overlaps its groups on child streams.

        Such a step must NOT be recorded into an embedder's CUDA graph
        (child-graph launches are not capturable); the embedder keeps the
        solver stage outside its captured span and calls :meth:`step`
        directly — the per-group child graphs make that call cheap.
        """
        return self._streams is not None

    @property
    def mjw_model(self):
        """First child's MuJoCo Warp model (debug/diagnostics only)."""
        return self.solvers[0].mjw_model

    @property
    def mjw_data(self):
        """First child's MuJoCo Warp data (debug/diagnostics only)."""
        return self.solvers[0].mjw_data

    @property
    def update_data_interval(self) -> int:
        return self.solvers[0].update_data_interval

    @override
    def step(self, state_in, state_out, control, contacts, dt: float) -> None:
        if self._streams is None:
            for solver in self.solvers:
                solver.step(state_in, state_out, control, contacts, dt)
            return

        device = wp.get_device(self._device)
        if self._child_graphs is None:
            if device.is_capturing:
                # An embedder began capturing before any eager warmup step:
                # record the children sequentially into its capture.
                for solver in self.solvers:
                    solver.step(state_in, state_out, control, contacts, dt)
                return
            # First eager step: capture each child's step once (all buffer
            # addresses are stable), then run it to actually advance state —
            # from here on every step overlaps the groups on child streams.
            graphs = []
            for solver in self.solvers:
                with wp.ScopedCapture(device=self._device) as capture:
                    solver.step(state_in, state_out, control, contacts, dt)
                graphs.append(capture.graph)
            self._child_graphs = graphs

        main_stream = wp.get_stream(self._device)
        main_stream.record_event(self._fork_event)
        for graph, stream, join_event in zip(self._child_graphs, self._streams, self._join_events):
            stream.wait_event(self._fork_event)
            wp.capture_launch(graph, stream=stream)
            stream.record_event(join_event)
            main_stream.wait_event(join_event)

    @override
    def notify_model_changed(self, flags) -> None:
        for solver in self.solvers:
            solver.notify_model_changed(flags)
        # a model change may rebuild solver-internal buffers: recapture
        self._child_graphs = None

    def reset(self, state, world_mask: wp.array | None = None, flags=None) -> None:
        """Fan out reset; each child gathers its local rows from *world_mask*."""
        for solver in self.solvers:
            solver.reset(state, world_mask=world_mask, flags=flags)

    def get_max_contact_count(self) -> int:
        """Total contact capacity across all child solvers."""
        return sum(solver.get_max_contact_count() for solver in self.solvers)

    def update_contacts(self, contacts, state) -> None:
        raise NotImplementedError(
            "SolverMuJoCoGroup does not support MuJoCo-data contact reporting yet;"
            " child solvers would need to merge into one Contacts buffer."
        )

    @override
    def coupling_eval_gravity_acceleration(self, out_body_acceleration, out_particle_acceleration) -> None:
        raise NotImplementedError("SolverMuJoCoGroup does not support particle coupling yet.")

    @override
    def coupling_eval_effective_mass(self, endpoint_kind, endpoint_index, endpoint_local_pos, out) -> None:
        raise NotImplementedError("SolverMuJoCoGroup does not support particle coupling yet.")

    @override
    def coupling_eval_effective_mass_block(
        self, endpoint_kind, endpoint_index, endpoint_local_pos, out_mass, out_inertia=None
    ) -> None:
        raise NotImplementedError("SolverMuJoCoGroup does not support particle coupling yet.")
