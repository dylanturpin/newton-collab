# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Body-pair contact reduction: post-narrow-phase compaction of redundant contacts.

Multi-shape bodies multiply narrow-phase output: a foot approximated by 7
cylinders emits up to 28 candidate contacts against a plane and up to 49
against another such foot, while the underlying physics is one flat patch that
is fully described by its deepest point plus the extremes of its footprint
(any interior point's force is a convex combination of forces at the hull
points, so discarding interior points preserves the feasible contact wrench
exactly).

This pass runs after the narrow phase has written the ``Contacts`` buffer and
compacts it **per body pair and normal bin**:

* contacts are grouped by ``(group0, group1, normal bin)`` where the group is
  the body (or the shape itself for static geometry, so distinct static
  colliders never merge) and the bin is a polyhedron face from
  :mod:`contact_reduction`'s normal binning;
* per group, the deepest contact always survives, and contacts within
  ``depth_window`` of the group's deepest gap compete for
  ``NUM_SPATIAL_DIRECTIONS`` spatial-extreme slots (the footprint hull);
* everything else -- interior points of the patch and candidates far shallower
  than the load-bearing set -- is discarded, and the ``Contacts`` arrays are
  compacted in place so ``rigid_contact_count`` itself drops.

The depth gate is **relative** to the group's own deepest contact rather than
an absolute threshold: witness/margin conventions differ between narrow-phase
contact modes, so absolute gaps are only comparable within a group, and an
absolute gate was observed to starve loaded patches of support points when a
pair's convention shifted mid-landing.

Reduction never crosses a normal bin, so multi-patch configurations (a body
touching floor and wall at once) keep representatives of every patch.  On
hashtable overflow the pass fails open: a contact that cannot be registered is
kept, never dropped.

All launches are fixed-size and the pass is CUDA-graph-capture compatible.
"""

from __future__ import annotations

import warp as wp

from ..sim.contacts import contact_surface_separation
from .contact_reduction import (
    NUM_SPATIAL_DIRECTIONS,
    get_slot,
    get_spatial_direction_2d,
    project_point_to_plane,
)
from .contact_reduction_global import make_contact_value
from .hashtable import HashTable, hashtable_find_or_insert

# Value slots per (body pair, normal bin, spatial cell) entry: extremes + one deepest.
BODY_PAIR_REDUCTION_SLOTS = NUM_SPATIAL_DIRECTIONS + 1


# Bit budget of the 63-bit group key. Group ids are asserted against this at
# pipeline construction: aliasing two groups could evict a patch's deepest
# contact, which would break the strict keep-deepest guarantee.
GROUP_ID_BITS = 21
MAX_GROUP_ID = (1 << GROUP_ID_BITS) - 1
# Cell coordinates are packed EXACTLY as two signed 8-bit values (+/-127 cells
# from the origin on the bin's face plane); positions beyond that range clamp
# to the border cell, which merges only the far periphery (beyond ~32 m at the
# default cell size) and only ever over-competes -- the deepest of the merged
# region is still kept.
CELL_COORD_MAX = 127


@wp.func
def _make_group_key(group_a: int, group_b: int, bin_id: int, cx: int, cy: int) -> wp.uint64:
    """Pack (group_a, group_b, normal bin, exact spatial cell) into 63 bits.

    Layout: ``[62:42] group_a (21b) | [41:21] group_b (21b) | [20:16] bin (5b)
    | [15:8] cx (8b) | [7:0] cy (8b)``.  All fields are exact within their
    asserted/clamped ranges, so two distinct groups can never alias.
    """
    ux = wp.uint64(wp.clamp(cx, -CELL_COORD_MAX, CELL_COORD_MAX) + CELL_COORD_MAX)
    uy = wp.uint64(wp.clamp(cy, -CELL_COORD_MAX, CELL_COORD_MAX) + CELL_COORD_MAX)
    return (
        (wp.uint64(group_a) << wp.uint64(42))
        | (wp.uint64(group_b) << wp.uint64(21))
        | ((wp.uint64(bin_id) & wp.uint64(0x1F)) << wp.uint64(16))
        | (ux << wp.uint64(8))
        | uy
    )


@wp.func_native("""
uint32_t mask = ((i >> 31) - 1u) | 0x80000000u;
uint32_t r = i ^ mask;
return reinterpret_cast<float&>(r);
""")
def float_unflip(i: wp.uint32) -> float: ...


@wp.func
def _contact_group_key(
    i: int,
    contact_shape0: wp.array[wp.int32],
    contact_shape1: wp.array[wp.int32],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_margin0: wp.array[wp.float32],
    contact_margin1: wp.array[wp.float32],
    body_q: wp.array[wp.transform],
    shape_body: wp.array[wp.int32],
    shape_count: int,
    cell_size: float,
):
    """Compute (key, gap, center, bin_id) for contact ``i``.

    The gap is :func:`newton._src.sim.contacts.contact_surface_separation` --
    the one canonical signed-separation convention the solvers consume
    (normal points shape0 -> shape1, positive = gap).  Depth must be ranked
    with exactly that formula: a hand-rolled variant with the opposite point
    order ranked SHALLOWEST as deepest and made the pass keep hovering
    candidates while discarding the load-bearing contacts.
    The group id is the body for dynamic shapes (all shapes of a body form one
    group) and the shape itself for static geometry (distinct static colliders
    never merge).  The key also carries a spatial cell -- the contact's
    position quantized on the bin's face plane by ``cell_size`` -- so multiple
    same-normal patches far apart on ONE shape pair (a long body across two
    regions of a terrain collider) each get their own deepest + extremes
    instead of competing for a single slot set.
    """
    s0 = contact_shape0[i]
    s1 = contact_shape1[i]

    b0 = shape_body[s0]
    b1 = shape_body[s1]

    p0_w = contact_point0[i]
    if b0 >= 0:
        p0_w = wp.transform_point(body_q[b0], p0_w)
    p1_w = contact_point1[i]
    if b1 >= 0:
        p1_w = wp.transform_point(body_q[b1], p1_w)

    n = contact_normal[i]
    gap = contact_surface_separation(p0_w, p1_w, n, contact_margin0[i], contact_margin1[i])
    center = 0.5 * (p0_w + p1_w)

    g0 = s0
    if b0 >= 0:
        g0 = shape_count + b0
    g1 = s1
    if b1 >= 0:
        g1 = shape_count + b1
    ga = wp.min(g0, g1)
    gb = wp.max(g0, g1)

    bin_id = get_slot(n)
    pos_2d = project_point_to_plane(bin_id, center)
    cx = wp.int32(wp.floor(pos_2d[0] / cell_size))
    cy = wp.int32(wp.floor(pos_2d[1] / cell_size))
    key = _make_group_key(ga, gb, bin_id, cx, cy)
    return key, gap, center, bin_id


@wp.kernel(enable_backward=False)
def _insert_deepest_kernel(
    contact_count: wp.array[wp.int32],
    contact_shape0: wp.array[wp.int32],
    contact_shape1: wp.array[wp.int32],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_margin0: wp.array[wp.float32],
    contact_margin1: wp.array[wp.float32],
    body_q: wp.array[wp.transform],
    shape_body: wp.array[wp.int32],
    shape_count: int,
    cell_size: float,
    ht_keys: wp.array[wp.uint64],
    ht_active_slots: wp.array[wp.int32],
    ht_values: wp.array[wp.uint64],
    # outputs
    keep_flags: wp.array[wp.int32],
):
    """Pass 1: register every contact's depth in its group's deepest slot.

    A contact that cannot be registered (hashtable full or detached shape) is
    kept unconditionally -- reduction must fail open, never drop silently.
    """
    i = wp.tid()
    if i >= contact_count[0]:
        return
    if contact_shape0[i] < 0 or contact_shape1[i] < 0:
        keep_flags[i] = 1
        return

    key, gap, _center, _bin_id = _contact_group_key(
        i,
        contact_shape0,
        contact_shape1,
        contact_point0,
        contact_point1,
        contact_normal,
        contact_margin0,
        contact_margin1,
        body_q,
        shape_body,
        shape_count,
        cell_size,
    )
    entry_idx = hashtable_find_or_insert(key, ht_keys, ht_active_slots)
    if entry_idx < 0:
        keep_flags[i] = 1
        return

    ht_capacity = ht_keys.shape[0]
    depth_value = make_contact_value(-gap, i, i, 0)
    slot_idx = wp.static(NUM_SPATIAL_DIRECTIONS) * ht_capacity + entry_idx
    if ht_values[slot_idx] < depth_value:
        wp.atomic_max(ht_values, slot_idx, depth_value)


@wp.kernel(enable_backward=False)
def _insert_spatial_kernel(
    contact_count: wp.array[wp.int32],
    contact_shape0: wp.array[wp.int32],
    contact_shape1: wp.array[wp.int32],
    contact_point0: wp.array[wp.vec3],
    contact_point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    contact_margin0: wp.array[wp.float32],
    contact_margin1: wp.array[wp.float32],
    body_q: wp.array[wp.transform],
    shape_body: wp.array[wp.int32],
    shape_count: int,
    cell_size: float,
    depth_window: float,
    ht_keys: wp.array[wp.uint64],
    ht_active_slots: wp.array[wp.int32],
    # in/out
    ht_values: wp.array[wp.uint64],
):
    """Pass 2: contacts near their group's deepest gap compete for the footprint extremes.

    Runs after pass 1 so every group's deepest slot is final.  The gate is
    relative -- ``gap <= deepest_gap + depth_window`` -- which is invariant to
    the per-contact-mode witness/margin offsets that make absolute gaps
    incomparable across groups.
    """
    i = wp.tid()
    if i >= contact_count[0]:
        return
    if contact_shape0[i] < 0 or contact_shape1[i] < 0:
        return

    key, gap, center, bin_id = _contact_group_key(
        i,
        contact_shape0,
        contact_shape1,
        contact_point0,
        contact_point1,
        contact_normal,
        contact_margin0,
        contact_margin1,
        body_q,
        shape_body,
        shape_count,
        cell_size,
    )
    entry_idx = hashtable_find_or_insert(key, ht_keys, ht_active_slots)
    if entry_idx < 0:
        return

    ht_capacity = ht_keys.shape[0]
    deepest_value = ht_values[wp.static(NUM_SPATIAL_DIRECTIONS) * ht_capacity + entry_idx]
    # high 32 bits hold float_flip(-deepest_gap)
    deepest_gap = -float_unflip(wp.uint32(deepest_value >> wp.uint64(32)))
    if gap > deepest_gap + depth_window:
        return

    pos_2d = project_point_to_plane(bin_id, center)
    for dir_i in range(wp.static(NUM_SPATIAL_DIRECTIONS)):
        dir_2d = get_spatial_direction_2d(dir_i)
        score = wp.dot(pos_2d, dir_2d)
        value = make_contact_value(score, i, i, 0)
        slot_idx = dir_i * ht_capacity + entry_idx
        if ht_values[slot_idx] < value:
            wp.atomic_max(ht_values, slot_idx, value)


@wp.kernel(enable_backward=False)
def _clear_active_values_kernel(
    ht_active_slots: wp.array[wp.int32],
    ht_capacity: int,
    # outputs
    ht_values: wp.array[wp.uint64],
):
    """Zero the value slots of every active hashtable entry from the last step."""
    t = wp.tid()
    if t >= ht_active_slots[ht_capacity]:
        return
    entry_idx = ht_active_slots[t]
    for slot in range(wp.static(BODY_PAIR_REDUCTION_SLOTS)):
        ht_values[slot * ht_capacity + entry_idx] = wp.uint64(0)


@wp.kernel(enable_backward=False)
def _export_keep_flags_kernel(
    ht_active_slots: wp.array[wp.int32],
    ht_values: wp.array[wp.uint64],
    ht_capacity: int,
    # outputs
    keep_flags: wp.array[wp.int32],
):
    """Flag the winning contact of every populated slot as kept.

    Flag writes are idempotent, so no per-entry deduplication is needed.  A
    packed value of zero means the slot never received a contact (a real value
    cannot be zero for finite scores because the flipped-float score occupies
    the high bits).
    """
    t = wp.tid()
    if t >= ht_active_slots[ht_capacity]:
        return
    entry_idx = ht_active_slots[t]
    for slot in range(wp.static(BODY_PAIR_REDUCTION_SLOTS)):
        value = ht_values[slot * ht_capacity + entry_idx]
        if value != wp.uint64(0):
            keep_flags[wp.int32(value & wp.uint64(0xFFFFFFFF))] = 1


@wp.kernel(enable_backward=False)
def _gather_kept_contacts_kernel(
    contact_count: wp.array[wp.int32],
    keep_flags: wp.array[wp.int32],
    keep_scan: wp.array[wp.int32],
    src_point_id: wp.array[wp.int32],
    src_shape0: wp.array[wp.int32],
    src_shape1: wp.array[wp.int32],
    src_point0: wp.array[wp.vec3],
    src_point1: wp.array[wp.vec3],
    src_offset0: wp.array[wp.vec3],
    src_offset1: wp.array[wp.vec3],
    src_normal: wp.array[wp.vec3],
    src_margin0: wp.array[wp.float32],
    src_margin1: wp.array[wp.float32],
    src_tids: wp.array[wp.int32],
    has_material: int,
    src_stiffness: wp.array[wp.float32],
    src_damping: wp.array[wp.float32],
    src_friction: wp.array[wp.float32],
    # outputs
    dst_point_id: wp.array[wp.int32],
    dst_shape0: wp.array[wp.int32],
    dst_shape1: wp.array[wp.int32],
    dst_point0: wp.array[wp.vec3],
    dst_point1: wp.array[wp.vec3],
    dst_offset0: wp.array[wp.vec3],
    dst_offset1: wp.array[wp.vec3],
    dst_normal: wp.array[wp.vec3],
    dst_margin0: wp.array[wp.float32],
    dst_margin1: wp.array[wp.float32],
    dst_tids: wp.array[wp.int32],
    dst_stiffness: wp.array[wp.float32],
    dst_damping: wp.array[wp.float32],
    dst_friction: wp.array[wp.float32],
):
    """Stable-compact kept contacts into the scratch arrays."""
    i = wp.tid()
    if i >= contact_count[0]:
        return
    if keep_flags[i] == 0:
        return
    dst = keep_scan[i] - 1  # inclusive scan -> 0-based position
    dst_point_id[dst] = src_point_id[i]
    dst_shape0[dst] = src_shape0[i]
    dst_shape1[dst] = src_shape1[i]
    dst_point0[dst] = src_point0[i]
    dst_point1[dst] = src_point1[i]
    dst_offset0[dst] = src_offset0[i]
    dst_offset1[dst] = src_offset1[i]
    dst_normal[dst] = src_normal[i]
    dst_margin0[dst] = src_margin0[i]
    dst_margin1[dst] = src_margin1[i]
    dst_tids[dst] = src_tids[i]
    if has_material != 0:
        dst_stiffness[dst] = src_stiffness[i]
        dst_damping[dst] = src_damping[i]
        dst_friction[dst] = src_friction[i]


@wp.kernel(enable_backward=False)
def _write_reduced_count_kernel(
    keep_scan: wp.array[wp.int32],
    # in/out
    contact_count: wp.array[wp.int32],
):
    """Replace the contact count with the number of kept contacts."""
    old_count = contact_count[0]
    if old_count > 0:
        contact_count[0] = keep_scan[old_count - 1]


class BodyPairContactReducer:
    """Owns the buffers and launch sequence for body-pair contact reduction.

    Created by :class:`newton.CollisionPipeline` when
    ``reduce_contacts_body_pairs=True``; not part of the public API.

    Args:
        rigid_contact_max: Capacity of the ``Contacts`` buffer being reduced.
        depth_window: Gap band [m] above a group's deepest contact within
            which contacts compete for the spatial-extreme (footprint hull)
            slots. Contacts shallower than that only survive if they are their
            group's deepest.
        cell_size: Spatial cell edge [m] on the normal bin's face plane. Each
            (body pair, bin, cell) keeps its own deepest + extremes, so
            same-normal patches farther apart than a cell never compete.
        device: Warp device.
    """

    def __init__(
        self, rigid_contact_max: int, depth_window: float, cell_size: float, device, hashtable_factor: float = 0.25
    ):
        self.rigid_contact_max = rigid_contact_max
        self.depth_window = float(depth_window)
        self.cell_size = float(cell_size)
        self.device = device
        # One entry per (body pair, bin, cell) actually touched -- far fewer
        # than contacts. Undersizing is safe: on a full table the insert
        # kernels keep the contact unconditionally (fail open), so the factor
        # trades memory for reduction coverage, never for dropped contacts.
        self.hashtable = HashTable(max(1024, int(rigid_contact_max * hashtable_factor)), device=device)
        self.ht_values = wp.zeros(BODY_PAIR_REDUCTION_SLOTS * self.hashtable.capacity, dtype=wp.uint64, device=device)
        self.keep_flags = wp.zeros(rigid_contact_max, dtype=wp.int32, device=device)
        self.keep_scan = wp.zeros(rigid_contact_max, dtype=wp.int32, device=device)
        self._scratch = None

    def _ensure_scratch(self):
        if self._scratch is not None:
            return
        n = self.rigid_contact_max
        dev = self.device
        self._scratch = {
            "point_id": wp.zeros(n, dtype=wp.int32, device=dev),
            "shape0": wp.zeros(n, dtype=wp.int32, device=dev),
            "shape1": wp.zeros(n, dtype=wp.int32, device=dev),
            "point0": wp.zeros(n, dtype=wp.vec3, device=dev),
            "point1": wp.zeros(n, dtype=wp.vec3, device=dev),
            "offset0": wp.zeros(n, dtype=wp.vec3, device=dev),
            "offset1": wp.zeros(n, dtype=wp.vec3, device=dev),
            "normal": wp.zeros(n, dtype=wp.vec3, device=dev),
            "margin0": wp.zeros(n, dtype=wp.float32, device=dev),
            "margin1": wp.zeros(n, dtype=wp.float32, device=dev),
            "tids": wp.zeros(n, dtype=wp.int32, device=dev),
            "stiffness": wp.zeros(n, dtype=wp.float32, device=dev),
            "damping": wp.zeros(n, dtype=wp.float32, device=dev),
            "friction": wp.zeros(n, dtype=wp.float32, device=dev),
        }

    def reduce(self, model, state, contacts):
        """Compact ``contacts`` in place, dropping patch-redundant candidates."""
        self._ensure_scratch()
        sc = self._scratch
        n = self.rigid_contact_max

        self.keep_flags.zero_()
        # Clear only the previously-active entries' value slots, then their
        # keys (order matters: the value clear reads the active list that the
        # key clear resets). Zeroing the full slot array instead costs a
        # ~100 MB memset per collide at large env counts.
        wp.launch(
            _clear_active_values_kernel,
            dim=self.hashtable.capacity,
            inputs=[self.hashtable.active_slots, self.hashtable.capacity],
            outputs=[self.ht_values],
            device=self.device,
            record_tape=False,
        )
        self.hashtable.clear_active()

        geom_inputs = [
            contacts.rigid_contact_count,
            contacts.rigid_contact_shape0,
            contacts.rigid_contact_shape1,
            contacts.rigid_contact_point0,
            contacts.rigid_contact_point1,
            contacts.rigid_contact_normal,
            contacts.rigid_contact_margin0,
            contacts.rigid_contact_margin1,
            state.body_q,
            model.shape_body,
            model.shape_count,
        ]
        wp.launch(
            _insert_deepest_kernel,
            dim=n,
            inputs=[*geom_inputs, self.cell_size, self.hashtable.keys, self.hashtable.active_slots, self.ht_values],
            outputs=[self.keep_flags],
            device=self.device,
            record_tape=False,
        )
        wp.launch(
            _insert_spatial_kernel,
            dim=n,
            inputs=[
                *geom_inputs,
                self.cell_size,
                self.depth_window,
                self.hashtable.keys,
                self.hashtable.active_slots,
                self.ht_values,
            ],
            device=self.device,
            record_tape=False,
        )
        wp.launch(
            _export_keep_flags_kernel,
            dim=self.hashtable.capacity,
            inputs=[self.hashtable.active_slots, self.ht_values, self.hashtable.capacity],
            outputs=[self.keep_flags],
            device=self.device,
            record_tape=False,
        )
        wp.utils.array_scan(self.keep_flags, self.keep_scan, True)

        has_material = int(contacts.rigid_contact_stiffness is not None)
        mat_src = (
            (contacts.rigid_contact_stiffness, contacts.rigid_contact_damping, contacts.rigid_contact_friction)
            if has_material
            else (sc["stiffness"], sc["damping"], sc["friction"])
        )
        wp.launch(
            _gather_kept_contacts_kernel,
            dim=n,
            inputs=[
                contacts.rigid_contact_count,
                self.keep_flags,
                self.keep_scan,
                contacts.rigid_contact_point_id,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_offset0,
                contacts.rigid_contact_offset1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1,
                contacts.rigid_contact_tids,
                has_material,
                mat_src[0],
                mat_src[1],
                mat_src[2],
            ],
            outputs=[
                sc["point_id"],
                sc["shape0"],
                sc["shape1"],
                sc["point0"],
                sc["point1"],
                sc["offset0"],
                sc["offset1"],
                sc["normal"],
                sc["margin0"],
                sc["margin1"],
                sc["tids"],
                sc["stiffness"],
                sc["damping"],
                sc["friction"],
            ],
            device=self.device,
            record_tape=False,
        )
        wp.launch(
            _write_reduced_count_kernel,
            dim=1,
            inputs=[self.keep_scan],
            outputs=[contacts.rigid_contact_count],
            device=self.device,
            record_tape=False,
        )
        wp.copy(contacts.rigid_contact_point_id, sc["point_id"])
        wp.copy(contacts.rigid_contact_shape0, sc["shape0"])
        wp.copy(contacts.rigid_contact_shape1, sc["shape1"])
        wp.copy(contacts.rigid_contact_point0, sc["point0"])
        wp.copy(contacts.rigid_contact_point1, sc["point1"])
        wp.copy(contacts.rigid_contact_offset0, sc["offset0"])
        wp.copy(contacts.rigid_contact_offset1, sc["offset1"])
        wp.copy(contacts.rigid_contact_normal, sc["normal"])
        wp.copy(contacts.rigid_contact_margin0, sc["margin0"])
        wp.copy(contacts.rigid_contact_margin1, sc["margin1"])
        wp.copy(contacts.rigid_contact_tids, sc["tids"])
        if has_material:
            wp.copy(contacts.rigid_contact_stiffness, sc["stiffness"])
            wp.copy(contacts.rigid_contact_damping, sc["damping"])
            wp.copy(contacts.rigid_contact_friction, sc["friction"])
