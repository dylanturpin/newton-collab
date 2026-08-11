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
* per group, the deepest contact always survives and every contact competes for
  one spatial-extreme slot per scan direction on projection alone, so each slot
  ends up holding that direction's true extreme (see
  :data:`BODY_PAIR_REDUCTION_SLOTS` for the policies this replaced);
* everything else -- interior points of the patch and candidates far shallower
  than the load-bearing set -- is discarded, and the ``Contacts`` arrays are
  compacted in place so ``rigid_contact_count`` itself drops.

Depth is ranked with the canonical
:func:`newton._src.sim.contacts.contact_surface_separation`, but only to choose
each group's single deepest survivor: depth never gates or biases the spatial
slots.  The policy therefore carries no tunable length scale at all, and no
classification can leave a patch without footprint support.

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
    float_flip,
    get_slot,
    project_point_to_plane,
)
from .hashtable import HashTable, hashtable_find_or_insert

# Scan directions used to pick footprint extremes on a normal bin's face plane.
# Deliberately independent of the mesh reducer's own direction count so tuning
# that path cannot silently change body-pair behaviour.
#
# Six is a measured floor, not a value carried over. Each extra direction costs a
# slot, an atomic per contact, and a kept contact whose row the solver carries,
# so fewer was tried first: on the randomized primitive piles of the property
# test, peak body speed against the unreduced run's 95 m/s comes out at 95 with
# six directions, 199 with four, and diverges outright with five. Fewer
# directions under-sample a patch's hull and leave a face rocking on too few
# support points.
BODY_PAIR_NUM_DIRECTIONS = 6


@wp.func
def _direction_2d(dir_idx: int) -> wp.vec2:
    """Unit 2D direction at ``dir_idx * 2pi / BODY_PAIR_NUM_DIRECTIONS``."""
    angle = float(dir_idx) * (2.0 * wp.pi / float(wp.static(BODY_PAIR_NUM_DIRECTIONS)))
    return wp.vec2(wp.cos(angle), wp.sin(angle))


# Value slots per (body pair, normal bin, spatial cell) entry: one spatial
# extreme per scan direction, plus the group's deepest contact.
#
# Every contact competes for every spatial slot on projection alone. Two richer
# policies were implemented and measured against both a walking humanoid and
# randomized primitive piles, and neither earned its cost:
#
# * gating slot entry on a depth window starves any patch whose gap spread
#   exceeds the window -- a tilted box face keeps only its deepest corner,
#   pivots on that point contact and diverges;
# * adding a second, depth-gated family of slots alongside this one changed
#   neither kept counts (p50 300 rows either way) nor trajectories on any
#   scene, because a patch's load-bearing contacts are already its spatial
#   extremes; it only doubled the slot memory and clearing work.
DEEPEST_SLOT = BODY_PAIR_NUM_DIRECTIONS
BODY_PAIR_REDUCTION_SLOTS = BODY_PAIR_NUM_DIRECTIONS + 1


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


@wp.func
def _pack_score(score: float, fingerprint: wp.uint32) -> wp.uint64:
    """Pack ``(score, content fingerprint)`` for ``atomic_max`` competition.

    Layout (bit 63 kept zero so the value orders identically read as signed or
    unsigned)::

        [62:31] float_flip(score)   (32 bits)
        [30:0]  content fingerprint (31 bits, never zero)

    Every contact of a group competes for every spatial slot on projection
    alone -- there is no depth gate or depth preference.  Both were tried and
    both starve patches: a gate drops every contact whose gap spread exceeds
    the window, and a high-order "near" preference is won outright by the
    group's deepest contact (trivially near), which then takes all slots.
    Either way a tilted box face collapses to single-point support and
    diverges.  Pure projection competition gives each direction slot to that
    direction's true spatial extreme, so footprint support is preserved by
    construction and no tuning parameter can remove it.

    No buffer index is stored: winners identify THEMSELVES in the selection
    pass by comparing their own packed value against the slot.  Ties on score
    and fingerprint mean identical content -- both contacts are kept, which
    fails open.  Because the value is a pure function of contact content, the
    winning SET is invariant to thread scheduling, buffer order, and buffer
    capacity.  The fingerprint is forced nonzero so a real value is never
    confused with the empty-slot sentinel 0.
    """
    return (wp.uint64(float_flip(score)) << wp.uint64(31)) | (wp.uint64(fingerprint) & wp.uint64(0x7FFFFFFF))


@wp.func
def _slot_gap(ht_values: wp.array[wp.uint64], ht_capacity: int, entry_idx: int) -> float:
    """Decode the group's deepest gap from its depth slot."""
    v = ht_values[wp.static(DEEPEST_SLOT) * ht_capacity + entry_idx]
    return -float_unflip(wp.uint32((v >> wp.uint64(31)) & wp.uint64(0xFFFFFFFF)))


@wp.func
def _key_fingerprint(sort_keys: wp.array[wp.int64], i: int) -> wp.uint32:
    """Fold the content-derived narrow-phase sort key to a 32-bit fingerprint."""
    k = wp.uint64(sort_keys[i])
    # bit 0 forced set: a zero fingerprint could collide with the empty-slot sentinel
    return wp.uint32(((k ^ (k >> wp.uint64(31))) & wp.uint64(0x7FFFFFFF)) | wp.uint64(1))


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
    sort_keys: wp.array[wp.int64],
    ht_keys: wp.array[wp.uint64],
    ht_active_slots: wp.array[wp.int32],
    ht_values: wp.array[wp.uint64],
    # outputs
    keep_flags: wp.array[wp.int32],
    contact_entry: wp.array[wp.int32],
    contact_gap: wp.array[wp.float32],
    contact_pos2d: wp.array[wp.vec2],
    stats: wp.array[wp.int32],
):
    """Pass 1: register every contact's depth in its group's deepest slot.

    Also caches the contact's hashtable entry, gap, and face-plane position so
    the later passes never recompute geometry or re-probe the table.  A
    contact that cannot be registered (hashtable full or detached shape) is
    kept unconditionally -- reduction must fail open, never drop silently.
    """
    i = wp.tid()
    if i >= contact_count[0]:
        return
    if contact_shape0[i] < 0 or contact_shape1[i] < 0:
        keep_flags[i] = 1
        contact_entry[i] = -1
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
    contact_entry[i] = entry_idx
    if entry_idx < 0:
        keep_flags[i] = 1
        wp.atomic_add(stats, 1, 1)
        return
    contact_gap[i] = gap
    contact_pos2d[i] = project_point_to_plane(bin_id, center)

    ht_capacity = ht_keys.shape[0]
    depth_value = _pack_score(-gap, _key_fingerprint(sort_keys, i))
    slot_idx = wp.static(DEEPEST_SLOT) * ht_capacity + entry_idx
    if ht_values[slot_idx] < depth_value:
        wp.atomic_max(ht_values, slot_idx, depth_value)


@wp.kernel(enable_backward=False)
def _insert_spatial_kernel(
    contact_count: wp.array[wp.int32],
    contact_entry: wp.array[wp.int32],
    contact_gap: wp.array[wp.float32],
    contact_pos2d: wp.array[wp.vec2],
    sort_keys: wp.array[wp.int64],
    ht_capacity: int,
    # in/out
    ht_values: wp.array[wp.uint64],
):
    """Pass 2: every contact competes for its group's footprint-extreme slots.

    Reads the entry/position cache pass 1 wrote instead of recomputing
    geometry or re-probing the hashtable.  Competition is on spatial
    projection alone, so each direction slot ends up holding that direction's
    true extreme contact.
    """
    i = wp.tid()
    if i >= contact_count[0]:
        return
    entry_idx = contact_entry[i]
    if entry_idx < 0:
        return

    pos_2d = contact_pos2d[i]
    fp = _key_fingerprint(sort_keys, i)
    for dir_i in range(wp.static(BODY_PAIR_NUM_DIRECTIONS)):
        dir_2d = _direction_2d(dir_i)
        value = _pack_score(wp.dot(pos_2d, dir_2d), fp)
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
def _select_winners_kernel(
    contact_count: wp.array[wp.int32],
    contact_entry: wp.array[wp.int32],
    contact_gap: wp.array[wp.float32],
    contact_pos2d: wp.array[wp.vec2],
    sort_keys: wp.array[wp.int64],
    ht_capacity: int,
    ht_values: wp.array[wp.uint64],
    # outputs
    keep_flags: wp.array[wp.int32],
):
    """Pass 3: every contact checks whether its own packed value won a slot.

    Winner self-identification: the packed values carry no buffer index, so a
    contact is kept iff one of the values it submitted equals the slot's final
    winner. Two contacts with identical content (equal score AND fingerprint)
    both match and are both kept -- reduction fails open on true ties. The
    kept set is therefore a pure function of contact content: invariant to
    thread scheduling, buffer order, and buffer capacity.
    """
    i = wp.tid()
    if i >= contact_count[0]:
        return
    entry_idx = contact_entry[i]
    if entry_idx < 0:
        return  # keep_flags[i] already set by pass 1 (fail open)

    fp = _key_fingerprint(sort_keys, i)
    gap = contact_gap[i]

    # NOTE: no early ``return`` inside the loop -- Warp does not reliably honor
    # returns from within a kernel for-loop, so accumulate and flag once.
    won = ht_values[wp.static(DEEPEST_SLOT) * ht_capacity + entry_idx] == _pack_score(-gap, fp)
    pos_2d = contact_pos2d[i]
    for dir_i in range(wp.static(BODY_PAIR_NUM_DIRECTIONS)):
        value = _pack_score(wp.dot(pos_2d, _direction_2d(dir_i)), fp)
        if ht_values[dir_i * ht_capacity + entry_idx] == value:
            won = True
    if won:
        keep_flags[i] = 1


@wp.kernel(enable_backward=False)
def _verify_invariant_kernel(
    contact_count: wp.array[wp.int32],
    contact_entry: wp.array[wp.int32],
    contact_gap: wp.array[wp.float32],
    contact_pos2d: wp.array[wp.vec2],
    sort_keys: wp.array[wp.int64],
    ht_capacity: int,
    ht_values: wp.array[wp.uint64],
    keep_flags: wp.array[wp.int32],
    # outputs
    violations: wp.array[wp.int32],
):
    """Certificate mode: re-derive the keep/discard decision and count disagreements.

    For every contact, recompute its packed slot values and check the
    invariant the pass promises: a DISCARDED contact must not beat any final
    slot winner it was eligible for, and a KEPT registered contact must
    actually match at least one winner. Any disagreement means a slot race,
    clearing bug, or ranking regression -- counted, never silent.
    """
    i = wp.tid()
    if i >= contact_count[0]:
        return
    entry_idx = contact_entry[i]
    if entry_idx < 0:
        return  # fail-open contacts are kept by definition

    fp = _key_fingerprint(sort_keys, i)
    gap = contact_gap[i]
    kept = keep_flags[i] != 0

    matched = False
    beaten = False

    deepest_value = ht_values[wp.static(DEEPEST_SLOT) * ht_capacity + entry_idx]
    my_depth = _pack_score(-gap, fp)
    if my_depth == deepest_value:
        matched = True
    elif my_depth > deepest_value:
        beaten = True  # I out-rank the recorded winner: the slot missed me

    pos_2d = contact_pos2d[i]
    for dir_i in range(wp.static(BODY_PAIR_NUM_DIRECTIONS)):
        dir_2d = _direction_2d(dir_i)
        value = _pack_score(wp.dot(pos_2d, dir_2d), fp)
        slot_value = ht_values[dir_i * ht_capacity + entry_idx]
        if value == slot_value:
            matched = True
        elif value > slot_value:
            beaten = True

    if kept and not matched:
        wp.atomic_add(violations, 0, 1)  # kept without winning: selection too permissive
    if (not kept) and matched:
        wp.atomic_add(violations, 0, 1)  # winner discarded: selection missed it
    if (not kept) and (not matched) and beaten:
        wp.atomic_add(violations, 2, 1)  # out-ranks a slot winner: atomic lost an update


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
def _scatter_back_kernel(
    contact_count: wp.array[wp.int32],
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
    """Write the compacted contacts back into the live arrays.

    Runs after ``_write_reduced_count_kernel``, so ``contact_count`` already
    holds the kept count: only the live range is touched, instead of copying
    entire capacity-sized arrays back.
    """
    i = wp.tid()
    if i >= contact_count[0]:
        return
    dst_point_id[i] = src_point_id[i]
    dst_shape0[i] = src_shape0[i]
    dst_shape1[i] = src_shape1[i]
    dst_point0[i] = src_point0[i]
    dst_point1[i] = src_point1[i]
    dst_offset0[i] = src_offset0[i]
    dst_offset1[i] = src_offset1[i]
    dst_normal[i] = src_normal[i]
    dst_margin0[i] = src_margin0[i]
    dst_margin1[i] = src_margin1[i]
    dst_tids[i] = src_tids[i]
    if has_material != 0:
        dst_stiffness[i] = src_stiffness[i]
        dst_damping[i] = src_damping[i]
        dst_friction[i] = src_friction[i]


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
        cell_size: Spatial cell edge [m] on the normal bin's face plane. Each
            (body pair, bin, cell) keeps its own deepest + extremes, so
            same-normal patches farther apart than a cell never compete.
        device: Warp device.
    """

    def __init__(
        self,
        rigid_contact_max: int,
            cell_size: float,
        device,
        hashtable_factor: float = 0.25,
        borrowed_scratch: dict | None = None,
        verify: bool = False,
    ):
        self.rigid_contact_max = rigid_contact_max
        self.cell_size = float(cell_size)
        self.device = device
        # Full-size gather scratch borrowed from the deterministic sorter (see
        # ContactSorter.borrow_full_scratch): the two stages run strictly
        # sequentially inside collide(), so sharing halves the pipeline's
        # scratch footprint. Fields the sorter did not allocate (zero-length
        # material arrays, point_id) are allocated locally on first use.
        self._borrowed_scratch = borrowed_scratch
        self.verify = bool(verify)
        # Telemetry counters, read via stats(): [0] invariant violations
        # (verify mode only), [1] fail-open keeps (hashtable full), never
        # reset automatically -- whole-run accumulators like the row
        # watermarks.
        self._stats = wp.zeros(3, dtype=wp.int32, device=device)
        # One entry per (body pair, bin, cell) actually touched -- far fewer
        # than contacts. Undersizing is safe: on a full table the insert
        # kernels keep the contact unconditionally (fail open), so the factor
        # trades memory for reduction coverage, never for dropped contacts.
        self.hashtable = HashTable(max(1024, int(rigid_contact_max * hashtable_factor)), device=device)
        self.ht_values = wp.zeros(BODY_PAIR_REDUCTION_SLOTS * self.hashtable.capacity, dtype=wp.uint64, device=device)
        self.keep_flags = wp.zeros(rigid_contact_max, dtype=wp.int32, device=device)
        self.keep_scan = wp.zeros(rigid_contact_max, dtype=wp.int32, device=device)
        # pass-1 cache read by passes 2 and 3: hashtable entry, canonical gap,
        # face-plane position
        self.contact_entry = wp.zeros(rigid_contact_max, dtype=wp.int32, device=device)
        self.contact_gap = wp.zeros(rigid_contact_max, dtype=wp.float32, device=device)
        self.contact_pos2d = wp.zeros(rigid_contact_max, dtype=wp.vec2, device=device)
        self._scratch = None

    def _ensure_scratch(self, need_material: bool):
        if self._scratch is None:
            n = self.rigid_contact_max
            dev = self.device
            dtypes = {
                "shape0": wp.int32,
                "shape1": wp.int32,
                "point0": wp.vec3,
                "point1": wp.vec3,
                "offset0": wp.vec3,
                "offset1": wp.vec3,
                "normal": wp.vec3,
                "margin0": wp.float32,
                "margin1": wp.float32,
                "tids": wp.int32,
            }
            borrowed = self._borrowed_scratch or {}
            self._scratch = {
                name: (
                    borrowed[name]
                    if name in borrowed and borrowed[name].shape[0] >= n
                    else wp.zeros(n, dtype=dtype, device=dev)
                )
                for name, dtype in dtypes.items()
            }
            self._scratch["point_id"] = wp.zeros(n, dtype=wp.int32, device=dev)
        if need_material and "stiffness" not in self._scratch:
            n = self.rigid_contact_max
            dev = self.device
            borrowed = self._borrowed_scratch or {}
            for name in ("stiffness", "damping", "friction"):
                arr = borrowed.get(name)
                self._scratch[name] = (
                    arr if arr is not None and arr.shape[0] >= n else wp.zeros(n, dtype=wp.float32, device=dev)
                )
        elif "stiffness" not in self._scratch:
            zero = wp.zeros(0, dtype=wp.float32, device=self.device)
            for name in ("stiffness", "damping", "friction"):
                self._scratch[name] = zero

    def reduce(self, model, state, contacts, sort_keys):
        """Compact ``contacts`` in place, dropping patch-redundant candidates.

        Args:
            model: The simulation model.
            state: Current state (body transforms for witness-point math).
            contacts: The contacts buffer to compact.
            sort_keys: Per-contact content-derived narrow-phase sort keys;
                folded into the fingerprints that make winner selection
                deterministic without any sorting.
        """
        has_material = contacts.rigid_contact_stiffness is not None
        self._ensure_scratch(has_material)
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
            inputs=[
                *geom_inputs,
                self.cell_size,
                sort_keys,
                self.hashtable.keys,
                self.hashtable.active_slots,
                self.ht_values,
            ],
            outputs=[
                self.keep_flags,
                self.contact_entry,
                self.contact_gap,
                self.contact_pos2d,
                self._stats,
            ],
            device=self.device,
            record_tape=False,
        )
        wp.launch(
            _insert_spatial_kernel,
            dim=n,
            inputs=[
                contacts.rigid_contact_count,
                self.contact_entry,
                self.contact_gap,
                self.contact_pos2d,
                sort_keys,
                self.hashtable.capacity,
                self.ht_values,
            ],
            device=self.device,
            record_tape=False,
        )
        wp.launch(
            _select_winners_kernel,
            dim=n,
            inputs=[
                contacts.rigid_contact_count,
                self.contact_entry,
                self.contact_gap,
                self.contact_pos2d,
                sort_keys,
                self.hashtable.capacity,
                self.ht_values,
            ],
            outputs=[self.keep_flags],
            device=self.device,
            record_tape=False,
        )
        if self.verify:
            wp.launch(
                _verify_invariant_kernel,
                dim=n,
                inputs=[
                    contacts.rigid_contact_count,
                    self.contact_entry,
                    self.contact_gap,
                        self.contact_pos2d,
                    sort_keys,
                        self.hashtable.capacity,
                    self.ht_values,
                    self.keep_flags,
                ],
                outputs=[self._stats],
                device=self.device,
                record_tape=False,
            )
        wp.utils.array_scan(self.keep_flags, self.keep_scan, True)

        has_material = int(has_material)
        mat_dst = (
            (contacts.rigid_contact_stiffness, contacts.rigid_contact_damping, contacts.rigid_contact_friction)
            if has_material
            else (sc["stiffness"], sc["damping"], sc["friction"])
        )
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
        wp.launch(
            _scatter_back_kernel,
            dim=n,
            inputs=[
                contacts.rigid_contact_count,
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
                has_material,
                sc["stiffness"],
                sc["damping"],
                sc["friction"],
            ],
            outputs=[
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
                mat_dst[0],
                mat_dst[1],
                mat_dst[2],
            ],
            device=self.device,
            record_tape=False,
        )

    def stats(self) -> dict:
        """Whole-run reduction telemetry (forces a device sync; read outside capture).

        Returns:
            ``invariant_violations``: disagreements found by the opt-in
            ``verify`` re-derivation (0 when verify is off).
            ``fail_open_keeps``: contacts kept unconditionally because the
            hashtable was full -- sustained non-zero values mean the
            ``hashtable_factor`` is too small for the scene.
        """
        v = self._stats.numpy()
        return {
            "invariant_violations": int(v[0]),
            "fail_open_keeps": int(v[1]),
            "outranked_discards": int(v[2]),
        }
