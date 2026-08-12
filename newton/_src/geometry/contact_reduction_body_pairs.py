# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Body-pair contact reduction: post-narrow-phase compaction of redundant contacts.

Multi-shape bodies multiply narrow-phase output: a foot approximated by 7
cylinders emits up to 28 candidate contacts against a plane and up to 49
against another such foot, while the underlying physics is one flat patch that
is described by its deepest point plus the extremes of its footprint.  An
interior point's NORMAL force is a convex combination of hull-point normal
forces, so support and tipping statics are preserved exactly, and translational
friction capacity (``sum mu*N_i``) is position-independent and preserved too.
The one wrench component this argument does NOT cover is torsional friction --
see the known-characteristic note below.

This pass runs after the narrow phase has written the ``Contacts`` buffer and
compacts it in two kernels over the live contact range -- register, then select:

* contacts are grouped by ``(group0, group1, normal bin, spatial cell)`` where
  the group is the body (or the shape itself for static geometry, so distinct
  static colliders never merge), the bin is a face of this module's own
  Z-aligned icosahedron (see :data:`BP_FACE_NORMALS` for why it is not the
  shared table), and the cell is the contact's position on that face plane
  quantized relative to the pair's own reference body (see
  :func:`_contact_group_key`);
* every contact enters its group's depth slot and all
  :data:`BODY_PAIR_NUM_DIRECTIONS` spatial-extreme slots, competing on
  projection alone, so each slot ends up holding that direction's true extreme
  (see :data:`BODY_PAIR_REDUCTION_SLOTS` for the policies this replaced);
* each contact then checks whether any value it submitted won its slot, and the
  survivors are compacted in place so ``rigid_contact_count`` itself drops.
  Everything else -- interior points of a patch, whose force is a convex
  combination of the hull points' -- is discarded.

**When this pays.** Only bodies whose collision is decomposed into several
primitives generate more candidates per pair than the slot budget. On a walking
G1 (7 cylinders per foot) the pass costs ~4% of the step and returns ~14% of
throughput. On a scene of single-box bodies there is nothing to discard and it is
pure overhead -- measured 5% slower on 1024x250 falling cubes. It is opt-in for
that reason; ``stats()`` reports the achieved ratio, and a ratio near 1.0 means
this scene does not want it.

Depth is ranked with the canonical
:func:`newton._src.sim.contacts.contact_surface_separation`, but only to choose
each group's single deepest survivor: depth never gates or biases the spatial
slots.  The policy therefore carries no tunable length scale at all, and no
classification can leave a patch without footprint support.

Reduction never crosses a normal bin, so multi-patch configurations (a body
touching floor and wall at once) keep representatives of every patch.  On
hashtable overflow the pass fails open: a contact that cannot be registered is
kept, never dropped.

**Temporal hysteresis.** Winner selection alone is memoryless, and on curved or
mesh support (flat analytic planes are immune) the score differences between a
patch's near-symmetric contacts are curvature-sized, so the argmax reorders
continuously: the kept set churns every step and a resting multi-collider body
sustains a bounded rocking limit cycle instead of settling.  With hysteresis
enabled (the default), each step snapshots the winners, and a contact that held
a slot keeps a score bonus of the hysteresis margin for that slot -- a
challenger better by more than the margin still wins immediately, incumbency
follows the contact's quantized pair-anchored position so it disengages by
itself under real sliding, and the kept set becomes a function of contact
geometry plus the previous step's winners.  Set the margin to ``0`` for the
exact, memoryless behavior.

**Known characteristic -- torsional patch friction.** During twist, friction at
every contact is saturated at ``mu*N_i``, so the resisting torque is
``mu * sum(N_i * r_i)`` -- fixed by WHERE the normal load sits, with no
remaining freedom for the solver to redistribute.  Keeping only rim extremes
forces the load to the largest lever arms, the maximizer of that sum: a
spinning multi-collider disc stops ~25% sooner reduced than unreduced
(adversarial A/B, mu 0.3 and 0.6; FPGS measures 1.34x the uniform-pressure
torque against a geometric ceiling of 1.5x).  A dense point set approximates
the uniform-pressure integral only by accident of even loading -- nobody
reweights in either case -- and no placement of boundary points can match both
translational capacity (``sum mu*N_i``) and the continuum torsion
(``2/3 mu*W*R``) simultaneously: the bias is inherent to any reduction that
keeps a patch's boundary points.  Bounded, smooth, and documented rather than
patched; the principled fix, if yaw transfer measurably suffers, is an
explicit per-patch torsion row in the solver, not point placement.  Watch
yaw-tracking error when validating policies whose feet pivot.

All launches are fixed-size and the pass is CUDA-graph-capture compatible.
"""

from __future__ import annotations

import math

import warp as wp

from ..sim.contacts import contact_surface_separation
from .contact_reduction import float_flip
from .hashtable import (
    _HASHTABLE_EMPTY_KEY_VALUE,
    HASHTABLE_EMPTY_KEY,
    HashTable,
    _hashtable_hash,
)

# Normal-binning table PRIVATE to body-pair reduction: the icosahedron from
# :mod:`contact_reduction`, rotated so a face CENTER points at world +Z and the
# yaw chosen to maximize the minimum bin margin at +/-X and +/-Y.
#
# The shared table stores the icosahedron Y-up, which puts +Z on the boundary
# between two equatorial faces to within 7e-8 of dot product. A flat plane is
# immune (its normal is bit-identical every step, so the argmax is stable), but
# on any curved or mesh surface in a Z-up world the patch normals wobble across
# that boundary, contacts regroup between bins step to step, and the kept set
# churns: measured 20x angular-velocity noise and a permanent rocking limit
# cycle that never settles. With this table the margin at +/-Z is 0.2546 -- a
# ground normal must tilt more than 20 degrees to change bins -- and 0.066 at
# +/-X/+/-Y (walls, ~4 degrees). Boundaries must exist somewhere on the sphere;
# they just must not pass through the one direction every scene is built
# around. Private copy on purpose: the mesh reducer consumes the shared table,
# and re-orienting that path is out of scope (same reasoning as
# BODY_PAIR_NUM_DIRECTIONS).
#
# Derivation (offline, float64): R0 aligns shared-table face 10 with +Z; yaw
# 55.75 deg about Z then maximizes min(margin(+/-X), margin(+/-Y)); rows
# renormalized. The two axis-aligned rows are written exactly.
# fmt: off
_BP_FACE_NORMALS_DATA = (
    -0.57095543,  0.75026579,  0.33333337,
    -0.64404845,  0.17218052,  0.74535593,
    -0.86918069, -0.36525859,  0.33333330,
    -0.93522697, -0.11932883, -0.33333339,
    -0.75091368,  0.57010308, -0.33333336,
     0.36427171,  0.86959472, -0.33333337,
    -0.17291149,  0.64385240, -0.74535609,
     0.47113684,  0.47167205,  0.74535599,
     0.11826704,  0.93536185,  0.33333335,
     0.17291155, -0.64385245,  0.74535603,
     0.0,         0.0,         1.0,
    -0.11826707, -0.93536184, -0.33333335,
    -0.36427169, -0.86959473,  0.33333337,
     0.0,         0.0,        -1.0,
    -0.47113677, -0.47167196, -0.74535609,
     0.93522695,  0.11932898,  0.33333339,
     0.75091367, -0.57010306,  0.33333340,
     0.57095545, -0.75026579, -0.33333334,
     0.64404840, -0.17218051, -0.74535597,
     0.86918063,  0.36525872, -0.33333332,
)
# fmt: on
BP_NUM_NORMAL_BINS = 20
_bp_face_normals_mat = wp.types.matrix(shape=(BP_NUM_NORMAL_BINS, 3), dtype=wp.float32)
BP_FACE_NORMALS = _bp_face_normals_mat(*_BP_FACE_NORMALS_DATA)


@wp.func
def _bp_get_slot(normal: wp.vec3) -> int:
    """Return the body-pair bin whose face normal best matches ``normal``.

    Full linear scan over the rotated table; the shared reducer's Y-cap pruning
    does not apply to a rotated orientation, and 20 dot products are noise next
    to the hashtable probe this feeds.
    """
    best_slot = int(0)
    max_dot = wp.dot(normal, wp.vec3(BP_FACE_NORMALS[0, 0], BP_FACE_NORMALS[0, 1], BP_FACE_NORMALS[0, 2]))
    for i in range(1, wp.static(BP_NUM_NORMAL_BINS)):
        d = wp.dot(normal, wp.vec3(BP_FACE_NORMALS[i, 0], BP_FACE_NORMALS[i, 1], BP_FACE_NORMALS[i, 2]))
        if d > max_dot:
            max_dot = d
            best_slot = i
    return best_slot


@wp.func
def _bp_project_point_to_plane(bin_normal_idx: int, point: wp.vec3) -> wp.vec2:
    """Project ``point`` onto the local 2D frame of a body-pair bin face."""
    face_normal = wp.vec3(
        BP_FACE_NORMALS[bin_normal_idx, 0],
        BP_FACE_NORMALS[bin_normal_idx, 1],
        BP_FACE_NORMALS[bin_normal_idx, 2],
    )
    if wp.abs(face_normal[1]) < 0.9:
        ref = wp.vec3(0.0, 1.0, 0.0)
    else:
        ref = wp.vec3(1.0, 0.0, 0.0)
    u = wp.normalize(ref - wp.dot(ref, face_normal) * face_normal)
    v = wp.cross(face_normal, u)
    return wp.vec2(wp.dot(point, u), wp.dot(point, v))


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
# Cell coordinates are packed EXACTLY as two signed 8-bit values, measured from
# the pair's own reference body (see _contact_group_key), so the range covers a
# +/-127-cell span ACROSS ONE BODY PAIR -- 32 m at the default cell size, far
# more than any single pair of colliders spans. Beyond it the coordinate clamps
# to the border cell, which only ever over-competes (the deepest of the merged
# region is still kept) and is counted in the clamp telemetry.
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


# Telemetry slots in the reducer's stats array. Whole-run accumulators (never
# reset), read via BodyPairContactReducer.stats().
STAT_VIOLATIONS = 0
STAT_FAIL_OPEN = 1
STAT_OUTRANKED = 2
STAT_ENTRY_WATERMARK = 3
STAT_CELL_CLAMPS = 4
STAT_CONTACTS_IN = 5
STAT_CONTACTS_KEPT = 6
STAT_INPUT_OVERFLOWS = 7
STAT_FALLBACK_FRAMES = 8
STAT_COUNT = 9

# Per-frame device state, reset by the preparation kernel each reduce.
FRAME_WORK_COUNT = 0  # validated loop bound for every reducer kernel
FRAME_INPUT_OVERFLOW = 1  # narrow phase emitted more contacts than capacity
FRAME_FALLBACK = 2  # hashtable budget exceeded: keep every contact this frame
FRAME_STATE_COUNT = 3

# Linear-probe budget for the reduction's hashtable accesses. The table is
# provisioned for load far below saturation (measured 0.08 on the reference
# scene), where expected probe chains are a handful of slots; a chain this long
# means the table is effectively saturated for this frame and the pass must
# fail open as a whole rather than let CUDA scheduling decide which groups get
# the last entries (a scheduling-dependent kept set) or let probes walk the
# entire capacity (a quadratic-cost cliff).
PROBE_BUDGET = 128


@wp.func
def _bp_find_or_insert(
    key: wp.uint64,
    keys: wp.array[wp.uint64],
    active_slots: wp.array[wp.int32],
) -> int:
    """``hashtable_find_or_insert`` with the probe budget applied."""
    capacity = keys.shape[0]
    capacity_mask = capacity - 1
    idx = _hashtable_hash(key, capacity_mask)

    for _i in range(wp.static(PROBE_BUDGET)):
        stored_key = keys[idx]
        if stored_key == key:
            return idx
        if stored_key == HASHTABLE_EMPTY_KEY:
            old_key = wp.atomic_cas(keys, idx, HASHTABLE_EMPTY_KEY, key)
            if old_key == HASHTABLE_EMPTY_KEY:
                active_idx = wp.atomic_add(active_slots, capacity, 1)
                if active_idx < capacity:
                    active_slots[active_idx] = idx
                return idx
            elif old_key == key:
                return idx
        idx = (idx + 1) & capacity_mask
    return -1


@wp.func
def _bp_find(key: wp.uint64, keys: wp.array[wp.uint64]) -> int:
    """``hashtable_find`` with the probe budget applied (read-only lookup)."""
    capacity = keys.shape[0]
    capacity_mask = capacity - 1
    idx = _hashtable_hash(key, capacity_mask)

    for _i in range(wp.static(PROBE_BUDGET)):
        stored_key = keys[idx]
        if stored_key == key:
            return idx
        if stored_key == HASHTABLE_EMPTY_KEY:
            return -1
        idx = (idx + 1) & capacity_mask
    return -1


@wp.kernel(enable_backward=False)
def _prepare_frame_kernel(
    contact_count: wp.array[wp.int32],
    reducer_capacity: int,
    # outputs
    frame_state: wp.array[wp.int32],
    stats: wp.array[wp.int32],
):
    """Derive this frame's validated work count and reset the per-frame flags.

    Newton's narrow phase reserves contact indices with an atomic add BEFORE
    checking capacity, so ``rigid_contact_count`` can legitimately exceed
    ``rigid_contact_max`` on overflow (the excess batches are simply not
    written).  The raw counter is therefore NOT a safe array bound.  Every
    reducer kernel loops over the work count derived here instead: on overflow
    it is zero, the whole pass becomes a no-op, and the counter and the
    materialized contact prefix reach the solver exactly as an unreduced
    pipeline would deliver them -- preserving both memory safety and the
    engine's overflow diagnostics.  Reducing a clamped prefix instead would be
    wrong: batched reservations can leave unwritten holes where a batch
    crossed capacity.
    """
    raw = contact_count[0]
    if raw >= 0 and raw <= reducer_capacity:
        frame_state[wp.static(FRAME_WORK_COUNT)] = raw
        frame_state[wp.static(FRAME_INPUT_OVERFLOW)] = 0
    else:
        frame_state[wp.static(FRAME_WORK_COUNT)] = 0
        frame_state[wp.static(FRAME_INPUT_OVERFLOW)] = 1
        wp.atomic_add(stats, wp.static(STAT_INPUT_OVERFLOWS), 1)
    frame_state[wp.static(FRAME_FALLBACK)] = 0


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
    """Compute (key, gap, face-plane position, cell_clamped) for contact ``i``.

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

    Both the cell and the returned face-plane position are measured from the
    pair's OWN reference body, not the world origin, because
    :func:`project_point_to_plane` is a pure linear map and the cell field is
    only 8 bits per axis: absolute coordinates pin every contact past ~32 m to
    the border cell, which silently disables the subdivision for all but the
    envs nearest the origin.  Subtracting a per-group constant cannot change
    which contact is extreme in a direction, so the shift is free, and it also
    recovers the float32 precision that large world offsets spend on the
    exponent.
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

    # Cell origin: the pair's lowest-indexed dynamic body, or the world origin
    # when both sides are static. Chosen from body indices rather than from the
    # contact's own shape order so the origin is the same for every contact of
    # the group -- otherwise contacts of one patch would land in different cells.
    ref = wp.vec3(0.0, 0.0, 0.0)
    if b0 >= 0 and (b1 < 0 or b0 <= b1):
        ref = wp.transform_get_translation(body_q[b0])
    elif b1 >= 0:
        ref = wp.transform_get_translation(body_q[b1])

    bin_id = _bp_get_slot(n)
    pos_2d = _bp_project_point_to_plane(bin_id, center - ref)
    # Round to nearest, not floor: that puts the reference body's origin at the
    # CENTER of cell 0 rather than on its corner. Flooring relative coordinates
    # would place the origin exactly on a cell boundary, so a patch centered
    # under the body -- the common case, e.g. a foot's colliders -- would always
    # straddle up to four cells and get four slot sets instead of one.
    # Clamp in FLOAT before converting: an extreme finite coordinate would
    # overflow the int32 conversion itself and corrupt both the grouping and
    # the clamp telemetry. The clamp bound is one cell beyond the packed range
    # so the out-of-range condition stays observable after conversion.
    fx = wp.floor(pos_2d[0] / cell_size + 0.5)
    fy = wp.floor(pos_2d[1] / cell_size + 0.5)
    bound = float(CELL_COORD_MAX + 1)
    cx = wp.int32(wp.clamp(fx, -bound, bound))
    cy = wp.int32(wp.clamp(fy, -bound, bound))
    key = _make_group_key(ga, gb, bin_id, cx, cy)
    # Report the clamp rather than hide it: clamped cells merge distant regions
    # of one shape pair, which only over-competes, but a sustained nonzero count
    # means one body pair spans more than +/-127 cells and reduction quality
    # across it is no longer what the cell size promises.
    clamped = wp.abs(cx) > CELL_COORD_MAX or wp.abs(cy) > CELL_COORD_MAX
    return key, gap, pos_2d, clamped


@wp.func
def _pos_key_31(pos_2d: wp.vec2) -> wp.uint64:
    """Quantize a face-plane position to a 31-bit, never-zero identity key.

    0.5 mm resolution; x covers +/-16.4 m (16 bits), y +/-4.1 m (14 bits),
    both measured from the pair's reference body, and bit 0 is always set so a
    real packed value can never equal the empty-slot sentinel 0.  Positions
    beyond the range clamp, which can only merge far-apart duplicates into a
    tie -- and ties are kept, never dropped.
    """
    # Clamp in FLOAT before converting so extreme coordinates cannot overflow
    # the int32 conversion (they would corrupt the identity, not just clamp it).
    qx = wp.int32(wp.clamp(wp.floor(pos_2d[0] * 2000.0), -32768.0, 32767.0)) + 32768
    qy = wp.int32(wp.clamp(wp.floor(pos_2d[1] * 2000.0), -8192.0, 8191.0)) + 8192
    return (wp.uint64(qx) << wp.uint64(15)) | (wp.uint64(qy) << wp.uint64(1)) | wp.uint64(1)


@wp.func
def _pack_score(primary: float, pos_key: wp.uint64) -> wp.uint64:
    """Pack ``(score, position identity)`` for ``atomic_max`` competition.

    Layout (bit 63 kept zero so the value orders identically read as signed or
    unsigned)::

        [62:31] float_flip(primary)              (32 bits)
        [30:0]  quantized face-plane position    (31 bits, never zero)

    Every contact of a group competes for every spatial slot on projection
    alone -- there is no depth gate or depth preference.  Both were tried and
    both starve patches: a gate drops every contact whose gap spread exceeds
    the window, and a high-order "near" preference is won outright by the
    group's deepest contact (trivially near), which then takes all slots.
    Either way a tilted box face collapses to single-point support and
    diverges.  Pure projection competition gives each direction slot to that
    direction's true spatial extreme, so footprint support is preserved by
    construction and no tuning parameter can remove it.

    The low bits are the GEOMETRIC tie-break: the contact's own quantized
    position on the pair-anchored face plane.  Symmetric collider layouts tie
    on the primary score constantly -- a row of foot cylinders shares a
    coordinate exactly -- and the previous tie-break, a hash of contact
    content, flipped winners whenever any input float moved (a static shape's
    witness point is world-space, so it changes under pure translation),
    churning the kept set of a body sliding in a straight line.  The position
    key cannot see translation, and unlike a single scalar it identifies the
    contact (up to 0.5 mm co-location), so a tie means two physically
    coincident contacts -- both are kept, and the <= 7-per-group bound holds
    for all genuinely distinct geometry.

    No buffer index is stored: winners identify THEMSELVES in the selection
    pass by comparing their own packed value against the slot, so the winning
    SET is a pure function of contact geometry: invariant to thread
    scheduling, buffer order, buffer capacity, and rigid translation.
    """
    return (wp.uint64(float_flip(primary)) << wp.uint64(31)) | pos_key


@wp.func
def _slot_index(entry_idx: int, slot: int) -> int:
    """Address of one value slot, entry-major.

    An entry's slots are adjacent, so each pass that touches all slots of one
    entry -- clearing, spatial competition, winner selection -- works on a
    single contiguous 56-byte run.  Slot-major addressing
    (``slot * capacity + entry``) instead put an entry's slots one whole
    capacity apart, costing one cache line per slot: 33 MB of stride between
    consecutive accesses at G1's 4.2M-entry table.
    """
    return entry_idx * wp.static(BODY_PAIR_REDUCTION_SLOTS) + slot


# Launch width for the passes that walk the contact buffer. Every one of them
# grid-strides over the LIVE contact count, so the launch only has to be wide
# enough to fill the device; sizing it to rigid_contact_max instead spends most
# of its threads on an early-exit test, because a training scene provisions the
# contact buffer far above the count it actually reaches (measured on G1: 102080
# capacity against a peak of 7937 live contacts at 64 envs, a 12.9x margin that
# holds as both scale with env count).
REDUCTION_MAX_THREADS = 262144


@wp.func
def _incumbent_mask(
    key: wp.uint64,
    pos_key: wp.uint64,
    prev_keys: wp.array[wp.uint64],
    prev_winner_pos: wp.array[wp.uint32],
) -> int:
    """Return the bitmask of slots this contact won on the PREVIOUS step.

    Probes the previous step's snapshot with the contact's group key and
    compares the contact's own position identity against each slot's recorded
    winner.  Identity is the quantized pair-anchored position, so a persisting
    near-static contact keeps matching itself across steps, while a contact
    sliding faster than the 0.5 mm quantum per step naturally stops matching --
    hysteresis engages exactly where kept-set churn lives (near-degenerate,
    slowly-varying patches) and disengages under real motion, where fresh
    extremes must win.
    """
    mask = int(0)
    prev_entry = _bp_find(key, prev_keys)
    if prev_entry >= 0:
        me = wp.uint32(pos_key)
        for slot in range(wp.static(BODY_PAIR_REDUCTION_SLOTS)):
            if prev_winner_pos[_slot_index(prev_entry, slot)] == me:
                mask |= 1 << slot
    return mask


@wp.func
def _biased_primary(primary: float, hysteresis: float, mask: int, slot: int) -> float:
    """Add the hysteresis bonus to ``primary`` if this contact holds ``slot``.

    The bonus is added to the raw score BEFORE packing, identically in the
    register, select, and verify passes, so winner self-identification still
    works by exact value equality.  A challenger better by more than
    ``hysteresis`` still wins: the bias resolves only near-degenerate handoffs,
    which is the churn that keeps multi-collider bodies on curved support from
    ever coming to rest (winner argmax is otherwise memoryless, and
    curvature-induced score differences between symmetric contacts reorder
    continuously).
    """
    if (mask >> slot) & 1 != 0:
        return primary + hysteresis
    return primary


@wp.kernel(enable_backward=False)
def _clear_prev_snapshot_kernel(
    ht_capacity: int,
    num_threads: int,
    # in/out
    prev_active_slots: wp.array[wp.int32],
    prev_keys: wp.array[wp.uint64],
):
    """Erase the previous snapshot's keys so stale groups cannot be probed.

    Only the keys need clearing: an entry's winner positions are unreachable
    once its key is gone, and a re-snapshotted entry overwrites all its slots.
    """
    count = prev_active_slots[ht_capacity]
    t = wp.tid()
    while t < count:
        prev_keys[prev_active_slots[t]] = HASHTABLE_EMPTY_KEY
        t += num_threads


@wp.kernel(enable_backward=False)
def _snapshot_prev_kernel(
    ht_keys: wp.array[wp.uint64],
    ht_active_slots: wp.array[wp.int32],
    ht_values: wp.array[wp.uint64],
    ht_capacity: int,
    frame_state: wp.array[wp.int32],
    num_threads: int,
    # outputs
    prev_keys: wp.array[wp.uint64],
    prev_winner_pos: wp.array[wp.uint32],
    prev_active_slots: wp.array[wp.int32],
):
    """Snapshot last step's winners before the live table is cleared.

    Copies each active entry's group key and, per slot, the winner's position
    identity (the low 31 bits of the packed value; 0 for slots no contact won).
    Entries keep their table index, so the snapshot is probed with the same
    hash layout as the live table.  Copying instead of pointer-swapping keeps
    every kernel argument fixed across steps, which CUDA graph capture
    requires.
    """
    count = ht_active_slots[ht_capacity]
    if frame_state[wp.static(FRAME_FALLBACK)] != 0 or frame_state[wp.static(FRAME_INPUT_OVERFLOW)] != 0:
        # The last frame kept everything (or reduced nothing): its slot values
        # are partial and must not seed incumbency.
        count = 0
    t = wp.tid()
    if t == 0:
        prev_active_slots[ht_capacity] = count
    while t < count:
        entry_idx = ht_active_slots[t]
        prev_active_slots[t] = entry_idx
        prev_keys[entry_idx] = ht_keys[entry_idx]
        for slot in range(wp.static(BODY_PAIR_REDUCTION_SLOTS)):
            idx = _slot_index(entry_idx, slot)
            prev_winner_pos[idx] = wp.uint32(ht_values[idx] & wp.uint64(0x7FFFFFFF))
        t += num_threads


@wp.func
def _register_contact_one(
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
    ht_keys: wp.array[wp.uint64],
    ht_active_slots: wp.array[wp.int32],
    ht_values: wp.array[wp.uint64],
    hysteresis: float,
    prev_keys: wp.array[wp.uint64],
    prev_winner_pos: wp.array[wp.uint32],
    frame_state: wp.array[wp.int32],
    keep_flags: wp.array[wp.int32],
    contact_entry: wp.array[wp.int32],
    contact_gap: wp.array[wp.float32],
    contact_pos2d: wp.array[wp.vec2],
    contact_incumbent: wp.array[wp.int32],
    stats: wp.array[wp.int32],
):
    """Enter contact ``i`` into every slot of its group: depth and all extremes.

    Depth and the spatial extremes go in together because they are independent
    ``atomic_max`` accumulations into disjoint slots of the same entry.  They
    were two passes only while slot entry was gated on the group's deepest
    value; once that gate was removed the split cost an extra launch and an
    extra read of this cache for nothing.

    The per-item body lives in a ``wp.func`` so the grid-stride kernel around
    it can bail out of one item without leaving the loop: Warp does not reliably
    honor ``return`` from inside a kernel for-loop, and an earlier version of
    the selection pass silently stopped flagging winners because of it.
    """
    # Initialize the flag here rather than with a capacity-wide memset: this
    # pass already visits exactly the live range, and rigid_contact_max is an
    # order of magnitude larger than that.
    keep_flags[i] = 0
    if contact_shape0[i] < 0 or contact_shape1[i] < 0:
        keep_flags[i] = 1
        contact_entry[i] = -1
        return

    key, gap, pos_2d, cell_clamped = _contact_group_key(
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
    if cell_clamped:
        wp.atomic_add(stats, wp.static(STAT_CELL_CLAMPS), 1)
    entry_idx = _bp_find_or_insert(key, ht_keys, ht_active_slots)
    contact_entry[i] = entry_idx
    if entry_idx < 0:
        # Table budget exhausted. Keeping just THIS contact would make the
        # kept set depend on which threads claimed the last entries, so the
        # whole frame falls back to the unreduced set (see the select pass).
        keep_flags[i] = 1
        frame_state[wp.static(FRAME_FALLBACK)] = 1
        wp.atomic_add(stats, wp.static(STAT_FAIL_OPEN), 1)
        return
    contact_gap[i] = gap
    contact_pos2d[i] = pos_2d

    pos_key = _pos_key_31(pos_2d)
    mask = int(0)
    if hysteresis > 0.0:
        mask = _incumbent_mask(key, pos_key, prev_keys, prev_winner_pos)
        contact_incumbent[i] = mask

    depth_value = _pack_score(_biased_primary(-gap, hysteresis, mask, wp.static(DEEPEST_SLOT)), pos_key)
    slot_idx = _slot_index(entry_idx, wp.static(DEEPEST_SLOT))
    if ht_values[slot_idx] < depth_value:
        wp.atomic_max(ht_values, slot_idx, depth_value)

    for dir_i in range(wp.static(BODY_PAIR_NUM_DIRECTIONS)):
        primary = _biased_primary(wp.dot(pos_2d, _direction_2d(dir_i)), hysteresis, mask, dir_i)
        value = _pack_score(primary, pos_key)
        dir_slot = _slot_index(entry_idx, dir_i)
        if ht_values[dir_slot] < value:
            wp.atomic_max(ht_values, dir_slot, value)


@wp.kernel(enable_backward=False)
def _register_contacts_kernel(
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
    hysteresis: float,
    prev_keys: wp.array[wp.uint64],
    prev_winner_pos: wp.array[wp.uint32],
    frame_state: wp.array[wp.int32],
    num_threads: int,
    # outputs
    keep_flags: wp.array[wp.int32],
    contact_entry: wp.array[wp.int32],
    contact_gap: wp.array[wp.float32],
    contact_pos2d: wp.array[wp.vec2],
    contact_incumbent: wp.array[wp.int32],
    stats: wp.array[wp.int32],
):
    """Pass 1 of 2: enter every contact into all slots of its group.

    Caches each contact's hashtable entry, gap, face-plane position, and
    incumbency mask so the selection pass neither recomputes geometry nor
    re-probes either table.  A contact that cannot be registered (hashtable
    full or detached shape) is kept unconditionally -- reduction must fail
    open, never drop silently.
    """
    count = frame_state[wp.static(FRAME_WORK_COUNT)]
    i = wp.tid()
    while i < count:
        _register_contact_one(
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
            ht_keys,
            ht_active_slots,
            ht_values,
            hysteresis,
            prev_keys,
            prev_winner_pos,
            frame_state,
            keep_flags,
            contact_entry,
            contact_gap,
            contact_pos2d,
            contact_incumbent,
            stats,
        )
        i += num_threads


@wp.kernel(enable_backward=False)
def _clear_active_values_kernel(
    ht_active_slots: wp.array[wp.int32],
    ht_capacity: int,
    num_threads: int,
    # outputs
    ht_values: wp.array[wp.uint64],
):
    """Zero the value slots of every active hashtable entry from the last step.

    Grid-strides over the ACTIVE entry list, not the table capacity: the table
    is deliberately provisioned well above the group count a scene reaches
    (G1 runs at load 0.08), so a capacity-wide launch spent over 90% of its
    threads deciding they had nothing to clear.
    """
    count = ht_active_slots[ht_capacity]
    t = wp.tid()
    while t < count:
        entry_idx = ht_active_slots[t]
        for slot in range(wp.static(BODY_PAIR_REDUCTION_SLOTS)):
            ht_values[_slot_index(entry_idx, slot)] = wp.uint64(0)
        t += num_threads


@wp.func
def _select_winner_one(
    i: int,
    contact_entry: wp.array[wp.int32],
    contact_gap: wp.array[wp.float32],
    contact_pos2d: wp.array[wp.vec2],
    contact_incumbent: wp.array[wp.int32],
    hysteresis: float,
    ht_values: wp.array[wp.uint64],
    keep_flags: wp.array[wp.int32],
):
    """Flag contact ``i`` if one of the values it submitted won its slot."""
    entry_idx = contact_entry[i]
    if entry_idx < 0:
        return  # keep_flags[i] already set by the register pass (fail open)

    gap = contact_gap[i]
    pos_2d = contact_pos2d[i]
    mask = int(0)
    if hysteresis > 0.0:
        mask = contact_incumbent[i]

    # NOTE: no early ``return`` inside the loop -- Warp does not reliably honor
    # returns from within a for-loop, so accumulate and flag once.
    pos_key = _pos_key_31(pos_2d)
    depth_primary = _biased_primary(-gap, hysteresis, mask, wp.static(DEEPEST_SLOT))
    won = ht_values[_slot_index(entry_idx, wp.static(DEEPEST_SLOT))] == _pack_score(depth_primary, pos_key)
    for dir_i in range(wp.static(BODY_PAIR_NUM_DIRECTIONS)):
        primary = _biased_primary(wp.dot(pos_2d, _direction_2d(dir_i)), hysteresis, mask, dir_i)
        if ht_values[_slot_index(entry_idx, dir_i)] == _pack_score(primary, pos_key):
            won = True
    if won:
        keep_flags[i] = 1


@wp.kernel(enable_backward=False)
def _select_winners_kernel(
    contact_count: wp.array[wp.int32],
    contact_entry: wp.array[wp.int32],
    contact_gap: wp.array[wp.float32],
    contact_pos2d: wp.array[wp.vec2],
    contact_incumbent: wp.array[wp.int32],
    hysteresis: float,
    ht_values: wp.array[wp.uint64],
    frame_state: wp.array[wp.int32],
    num_threads: int,
    # outputs
    keep_flags: wp.array[wp.int32],
):
    """Pass 2 of 2: every contact checks whether its own packed value won a slot.

    Winner self-identification: the packed values carry no buffer index, so a
    contact is kept iff one of the values it submitted equals the slot's final
    winner. Two contacts with identical content (equal biased score AND
    position identity) both match and are both kept -- reduction fails open on
    true ties. The kept set is therefore a pure function of contact geometry
    plus, when hysteresis is enabled, the previous step's winners: invariant to
    thread scheduling, buffer order, and buffer capacity.
    """
    count = frame_state[wp.static(FRAME_WORK_COUNT)]
    keep_all = frame_state[wp.static(FRAME_FALLBACK)] != 0
    i = wp.tid()
    while i < count:
        if keep_all:
            # Table budget was exceeded somewhere this frame: the only
            # deterministic result is the whole unreduced set.
            keep_flags[i] = 1
        else:
            _select_winner_one(
                i, contact_entry, contact_gap, contact_pos2d, contact_incumbent, hysteresis, ht_values, keep_flags
            )
        i += num_threads


@wp.func
def _verify_invariant_one(
    i: int,
    contact_entry: wp.array[wp.int32],
    contact_gap: wp.array[wp.float32],
    contact_pos2d: wp.array[wp.vec2],
    contact_incumbent: wp.array[wp.int32],
    hysteresis: float,
    ht_values: wp.array[wp.uint64],
    keep_flags: wp.array[wp.int32],
    violations: wp.array[wp.int32],
):
    """Re-derive contact ``i``'s keep/discard decision and count disagreements."""
    entry_idx = contact_entry[i]
    if entry_idx < 0:
        return  # fail-open contacts are kept by definition

    gap = contact_gap[i]
    pos_2d = contact_pos2d[i]
    kept = keep_flags[i] != 0
    mask = int(0)
    if hysteresis > 0.0:
        mask = contact_incumbent[i]

    matched = False
    beaten = False

    pos_key = _pos_key_31(pos_2d)
    deepest_value = ht_values[_slot_index(entry_idx, wp.static(DEEPEST_SLOT))]
    my_depth = _pack_score(_biased_primary(-gap, hysteresis, mask, wp.static(DEEPEST_SLOT)), pos_key)
    if my_depth == deepest_value:
        matched = True
    elif my_depth > deepest_value:
        beaten = True  # I out-rank the recorded winner: the slot missed me

    for dir_i in range(wp.static(BODY_PAIR_NUM_DIRECTIONS)):
        primary = _biased_primary(wp.dot(pos_2d, _direction_2d(dir_i)), hysteresis, mask, dir_i)
        value = _pack_score(primary, pos_key)
        slot_value = ht_values[_slot_index(entry_idx, dir_i)]
        if value == slot_value:
            matched = True
        elif value > slot_value:
            beaten = True

    if kept and not matched:
        wp.atomic_add(violations, wp.static(STAT_VIOLATIONS), 1)  # kept without winning: too permissive
    if (not kept) and matched:
        wp.atomic_add(violations, wp.static(STAT_VIOLATIONS), 1)  # winner discarded: selection missed it
    if (not kept) and (not matched) and beaten:
        wp.atomic_add(violations, wp.static(STAT_OUTRANKED), 1)  # out-ranks a winner: atomic lost an update


@wp.kernel(enable_backward=False)
def _verify_invariant_kernel(
    contact_count: wp.array[wp.int32],
    contact_entry: wp.array[wp.int32],
    contact_gap: wp.array[wp.float32],
    contact_pos2d: wp.array[wp.vec2],
    contact_incumbent: wp.array[wp.int32],
    hysteresis: float,
    ht_values: wp.array[wp.uint64],
    keep_flags: wp.array[wp.int32],
    frame_state: wp.array[wp.int32],
    num_threads: int,
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
    count = frame_state[wp.static(FRAME_WORK_COUNT)]
    if frame_state[wp.static(FRAME_FALLBACK)] != 0:
        count = 0  # keep-all frames are trivially consistent; slots are partial
    i = wp.tid()
    while i < count:
        _verify_invariant_one(
            i,
            contact_entry,
            contact_gap,
            contact_pos2d,
            contact_incumbent,
            hysteresis,
            ht_values,
            keep_flags,
            violations,
        )
        i += num_threads


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
    frame_state: wp.array[wp.int32],
    num_threads: int,
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
    count = frame_state[wp.static(FRAME_WORK_COUNT)]
    i = wp.tid()
    while i < count:
        if keep_flags[i] != 0:
            dst = keep_scan[i] - 1  # inclusive scan -> 0-based position
            _copy_contact(
                i,
                dst,
                src_point_id,
                src_shape0,
                src_shape1,
                src_point0,
                src_point1,
                src_offset0,
                src_offset1,
                src_normal,
                src_margin0,
                src_margin1,
                src_tids,
                has_material,
                src_stiffness,
                src_damping,
                src_friction,
                dst_point_id,
                dst_shape0,
                dst_shape1,
                dst_point0,
                dst_point1,
                dst_offset0,
                dst_offset1,
                dst_normal,
                dst_margin0,
                dst_margin1,
                dst_tids,
                dst_stiffness,
                dst_damping,
                dst_friction,
            )
        i += num_threads


@wp.func
def _copy_contact(
    i: int,
    dst: int,
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
    """Copy one contact record from index ``i`` to index ``dst``."""
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
    frame_state: wp.array[wp.int32],
    num_threads: int,
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
    holds the KEPT count: only that range is touched, instead of copying
    entire capacity-sized arrays back.  On an input-overflow frame the counter
    was deliberately left at its raw (over-capacity) value, so the bound comes
    from the frame state instead: zero, and nothing is written back.
    """
    count = contact_count[0]
    if frame_state[wp.static(FRAME_INPUT_OVERFLOW)] != 0:
        count = 0
    i = wp.tid()
    while i < count:
        _copy_contact(
            i,
            i,
            src_point_id,
            src_shape0,
            src_shape1,
            src_point0,
            src_point1,
            src_offset0,
            src_offset1,
            src_normal,
            src_margin0,
            src_margin1,
            src_tids,
            has_material,
            src_stiffness,
            src_damping,
            src_friction,
            dst_point_id,
            dst_shape0,
            dst_shape1,
            dst_point0,
            dst_point1,
            dst_offset0,
            dst_offset1,
            dst_normal,
            dst_margin0,
            dst_margin1,
            dst_tids,
            dst_stiffness,
            dst_damping,
            dst_friction,
        )
        i += num_threads


@wp.kernel(enable_backward=False)
def _write_reduced_count_kernel(
    keep_scan: wp.array[wp.int32],
    ht_active_slots: wp.array[wp.int32],
    ht_capacity: int,
    frame_state: wp.array[wp.int32],
    # in/out
    contact_count: wp.array[wp.int32],
    stats: wp.array[wp.int32],
):
    """Replace the contact count with the number of kept contacts.

    Also records the watermarks that size the pass: how many contacts arrived,
    how many survived, and how many hashtable entries the scene actually needed.
    Occupancy is what says whether ``hashtable_factor`` is generous or one busy
    step away from failing open.
    """
    wp.atomic_max(stats, wp.static(STAT_CONTACTS_IN), contact_count[0])
    wp.atomic_max(stats, wp.static(STAT_ENTRY_WATERMARK), ht_active_slots[ht_capacity])
    if frame_state[wp.static(FRAME_INPUT_OVERFLOW)] != 0:
        # Leave the raw counter untouched: the solver receives exactly what an
        # unreduced pipeline would deliver, and overflow stays observable.
        return
    if frame_state[wp.static(FRAME_FALLBACK)] != 0:
        wp.atomic_add(stats, wp.static(STAT_FALLBACK_FRAMES), 1)
    work = frame_state[wp.static(FRAME_WORK_COUNT)]
    if work > 0:
        kept = keep_scan[work - 1]
        contact_count[0] = kept
        wp.atomic_max(stats, wp.static(STAT_CONTACTS_KEPT), kept)


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
        hysteresis: Temporal hysteresis margin [m]. A contact that won a slot
            on the previous step receives this bonus on its raw score for that
            slot, so near-degenerate winners stop handing off every step -- the
            churn that keeps multi-collider bodies on curved or mesh support
            oscillating instead of coming to rest.  A challenger better by more
            than the margin still wins immediately, and incumbency follows the
            contact's quantized pair-anchored position, so it disengages by
            itself once a contact slides faster than 0.5 mm per step.  Also
            softens two exact guarantees by at most the margin: the depth
            representative may be a contact within ``hysteresis`` of the true
            deepest, and a spatial slot may hold a contact within ``hysteresis``
            of the true extreme.  ``0`` disables the mechanism entirely and
            restores the exact, memoryless behavior.
    """

    def __init__(
        self,
        rigid_contact_max: int,
        cell_size: float,
        device,
        hashtable_factor: float = 0.25,
        borrowed_scratch: dict | None = None,
        verify: bool = False,
        hysteresis: float = 0.001,
    ):
        self.rigid_contact_max = rigid_contact_max
        self.cell_size = float(cell_size)
        if not (math.isfinite(self.cell_size) and self.cell_size > 0.0):
            raise ValueError(f"cell_size must be finite and positive, got {cell_size}")
        self.device = device
        self.hysteresis = float(hysteresis)
        if not (math.isfinite(self.hysteresis) and self.hysteresis >= 0.0):
            raise ValueError(f"hysteresis must be finite and non-negative, got {hysteresis}")
        hashtable_factor = float(hashtable_factor)
        if not (math.isfinite(hashtable_factor) and hashtable_factor > 0.0):
            raise ValueError(f"hashtable_factor must be finite and positive, got {hashtable_factor}")
        # Full-size gather scratch borrowed from the deterministic sorter (see
        # ContactSorter.borrow_full_scratch): the two stages run strictly
        # sequentially inside collide(), so sharing halves the pipeline's
        # scratch footprint. Fields the sorter did not allocate (zero-length
        # material arrays, point_id) are allocated locally on first use.
        self._borrowed_scratch = borrowed_scratch
        self.verify = bool(verify)
        # Whole-run telemetry accumulators; see stats() for the slot meanings.
        self._stats = wp.zeros(STAT_COUNT, dtype=wp.int32, device=device)
        # Per-frame device state: validated work count + overflow/fallback
        # flags, reset by the preparation kernel each reduce.
        self._frame_state = wp.zeros(FRAME_STATE_COUNT, dtype=wp.int32, device=device)
        # One entry per (body pair, bin, cell) actually touched -- far fewer
        # than contacts. Undersizing is safe: on a full table the insert
        # kernels keep the contact unconditionally (fail open), so the factor
        # trades memory for reduction coverage, never for dropped contacts.
        self.hashtable = HashTable(max(1024, int(rigid_contact_max * hashtable_factor)), device=device)
        self.ht_values = wp.zeros(BODY_PAIR_REDUCTION_SLOTS * self.hashtable.capacity, dtype=wp.uint64, device=device)
        # Fixed launch width for every pass; the kernels grid-stride over the
        # live count, so this only has to fill the device. See
        # REDUCTION_MAX_THREADS.
        self.stride_threads = min(rigid_contact_max, REDUCTION_MAX_THREADS)
        self.entry_stride_threads = min(self.hashtable.capacity, REDUCTION_MAX_THREADS)
        self.keep_flags = wp.zeros(rigid_contact_max, dtype=wp.int32, device=device)
        self.keep_scan = wp.zeros(rigid_contact_max, dtype=wp.int32, device=device)
        # pass-1 cache read by the selection pass: hashtable entry, canonical
        # gap, face-plane position
        self.contact_entry = wp.zeros(rigid_contact_max, dtype=wp.int32, device=device)
        self.contact_gap = wp.zeros(rigid_contact_max, dtype=wp.float32, device=device)
        self.contact_pos2d = wp.zeros(rigid_contact_max, dtype=wp.vec2, device=device)
        # Previous-step winner snapshot for temporal hysteresis: group keys,
        # per-slot winner position identities, and the active list needed to
        # erase the snapshot next step. Rebuilt by copy each reduce (never
        # pointer-swapped) so every kernel argument stays fixed for CUDA graph
        # capture. Zero-length when hysteresis is off: the kernels take the
        # arrays unconditionally but only touch them behind a hysteresis > 0
        # guard.
        if self.hysteresis > 0.0:
            cap = self.hashtable.capacity
            self.prev_keys = wp.full(cap, _HASHTABLE_EMPTY_KEY_VALUE, dtype=wp.uint64, device=device)
            self.prev_winner_pos = wp.zeros(BODY_PAIR_REDUCTION_SLOTS * cap, dtype=wp.uint32, device=device)
            self.prev_active_slots = wp.zeros(cap + 1, dtype=wp.int32, device=device)
            self.contact_incumbent = wp.zeros(rigid_contact_max, dtype=wp.int32, device=device)
        else:
            self.prev_keys = wp.zeros(0, dtype=wp.uint64, device=device)
            self.prev_winner_pos = wp.zeros(0, dtype=wp.uint32, device=device)
            self.prev_active_slots = wp.zeros(0, dtype=wp.int32, device=device)
            self.contact_incumbent = wp.zeros(0, dtype=wp.int32, device=device)
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

    def reduce(self, model, state, contacts):
        """Compact ``contacts`` in place, dropping patch-redundant candidates.

        Args:
            model: The simulation model.
            state: Current state (body transforms for witness-point math).
            contacts: The contacts buffer to compact.
        """
        has_material = contacts.rigid_contact_stiffness is not None
        self._ensure_scratch(has_material)
        sc = self._scratch

        # Snapshot last step's winners BEFORE anything clears the live table:
        # erase the old snapshot's keys, then copy the current active entries'
        # keys and winner position identities across. Both kernels walk only
        # the (small) active lists.
        if self.hysteresis > 0.0:
            wp.launch(
                _clear_prev_snapshot_kernel,
                dim=self.entry_stride_threads,
                inputs=[self.hashtable.capacity, self.entry_stride_threads],
                outputs=[self.prev_active_slots, self.prev_keys],
                device=self.device,
                record_tape=False,
            )
            wp.launch(
                _snapshot_prev_kernel,
                dim=self.entry_stride_threads,
                inputs=[
                    self.hashtable.keys,
                    self.hashtable.active_slots,
                    self.ht_values,
                    self.hashtable.capacity,
                    self._frame_state,
                    self.entry_stride_threads,
                ],
                outputs=[self.prev_keys, self.prev_winner_pos, self.prev_active_slots],
                device=self.device,
                record_tape=False,
            )

        # Derive this frame's validated work count (see the kernel docstring:
        # the raw narrow-phase counter is NOT a safe array bound on overflow).
        # Must run after the history snapshot, which reads last frame's flags.
        wp.launch(
            _prepare_frame_kernel,
            dim=1,
            inputs=[contacts.rigid_contact_count, self.rigid_contact_max],
            outputs=[self._frame_state, self._stats],
            device=self.device,
            record_tape=False,
        )

        # keep_flags needs no memset: pass 1 writes every live entry, and
        # positions past the live count are never read (the scan output above
        # the count is unused, and the gather only walks the live range).
        #
        # Clear only the previously-active entries' value slots, then their keys
        # (order matters: the value clear reads the active list that the key
        # clear resets). Zeroing the full slot array instead costs a ~200 MB
        # memset per collide at large env counts.
        wp.launch(
            _clear_active_values_kernel,
            dim=self.entry_stride_threads,
            inputs=[self.hashtable.active_slots, self.hashtable.capacity, self.entry_stride_threads],
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
            _register_contacts_kernel,
            dim=self.stride_threads,
            inputs=[
                *geom_inputs,
                self.cell_size,
                self.hashtable.keys,
                self.hashtable.active_slots,
                self.ht_values,
                self.hysteresis,
                self.prev_keys,
                self.prev_winner_pos,
                self._frame_state,
                self.stride_threads,
            ],
            outputs=[
                self.keep_flags,
                self.contact_entry,
                self.contact_gap,
                self.contact_pos2d,
                self.contact_incumbent,
                self._stats,
            ],
            device=self.device,
            record_tape=False,
        )
        wp.launch(
            _select_winners_kernel,
            dim=self.stride_threads,
            inputs=[
                contacts.rigid_contact_count,
                self.contact_entry,
                self.contact_gap,
                self.contact_pos2d,
                self.contact_incumbent,
                self.hysteresis,
                self.ht_values,
                self._frame_state,
                self.stride_threads,
            ],
            outputs=[self.keep_flags],
            device=self.device,
            record_tape=False,
        )
        if self.verify:
            wp.launch(
                _verify_invariant_kernel,
                dim=self.stride_threads,
                inputs=[
                    contacts.rigid_contact_count,
                    self.contact_entry,
                    self.contact_gap,
                    self.contact_pos2d,
                    self.contact_incumbent,
                    self.hysteresis,
                    self.ht_values,
                    self.keep_flags,
                    self._frame_state,
                    self.stride_threads,
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
            dim=self.stride_threads,
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
                self._frame_state,
                self.stride_threads,
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
            inputs=[self.keep_scan, self.hashtable.active_slots, self.hashtable.capacity, self._frame_state],
            outputs=[contacts.rigid_contact_count, self._stats],
            device=self.device,
            record_tape=False,
        )
        wp.launch(
            _scatter_back_kernel,
            dim=self.stride_threads,
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
                self._frame_state,
                self.stride_threads,
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

        Every value is a whole-run accumulator -- a max watermark or a total --
        never reset between steps, so one read at the end of a run characterizes
        it. The three sizing watermarks exist so a scene that has outgrown its
        provisioning says so instead of quietly reducing less well.

        Returns:
            ``invariant_violations``: disagreements found by the opt-in
            ``verify`` re-derivation (0 when verify is off).
            ``fail_open_keeps``: contacts kept unconditionally because the
            hashtable was full -- sustained non-zero values mean the
            ``hashtable_factor`` is too small for the scene.
            ``outranked_discards``: discarded contacts that out-rank their
            slot's recorded winner (verify mode only); non-zero means an atomic
            lost an update.
            ``cell_clamp_events``: contacts whose spatial cell hit the packed
            +/-127 range; non-zero means distant regions of one shape pair are
            merging on the periphery.
            ``max_contacts_in`` / ``max_contacts_kept``: peak live contact count
            before and after reduction, i.e. the achieved reduction ratio and
            the capacity the ``Contacts`` buffer actually needs.
            ``max_hashtable_entries`` / ``hashtable_capacity``: peak distinct
            (body pair, bin, cell) groups against the table that holds them.
            ``hashtable_load``: the ratio of those two -- linear probing degrades
            well past ~0.7, and 1.0 means entries were refused and kept open.
        """
        v = self._stats.numpy()
        entries = int(v[STAT_ENTRY_WATERMARK])
        return {
            "invariant_violations": int(v[STAT_VIOLATIONS]),
            "fail_open_keeps": int(v[STAT_FAIL_OPEN]),
            "outranked_discards": int(v[STAT_OUTRANKED]),
            "cell_clamp_events": int(v[STAT_CELL_CLAMPS]),
            "max_contacts_in": int(v[STAT_CONTACTS_IN]),
            "max_contacts_kept": int(v[STAT_CONTACTS_KEPT]),
            "max_hashtable_entries": entries,
            "hashtable_capacity": self.hashtable.capacity,
            "hashtable_load": entries / self.hashtable.capacity,
            "input_overflow_frames": int(v[STAT_INPUT_OVERFLOWS]),
            "fallback_frames": int(v[STAT_FALLBACK_FRAMES]),
        }

    def describe(self) -> dict:
        """Static footprint of the pass: buffer bytes by role, and capacities.

        Reported per role rather than as one total so an over-provisioned
        ``rigid_contact_max`` is distinguishable from the pass's own overhead;
        ``gather_scratch_owned_bytes`` counts only arrays this pass allocated
        itself, excluding any borrowed from the deterministic sorter.
        """
        n = self.rigid_contact_max
        owned = 0
        borrowed_ids = {id(a) for a in (self._borrowed_scratch or {}).values()}
        for arr in (self._scratch or {}).values():
            if id(arr) not in borrowed_ids:
                owned += arr.size * wp.types.type_size_in_bytes(arr.dtype)
        return {
            "rigid_contact_max": n,
            "hashtable_capacity": self.hashtable.capacity,
            "slots_per_entry": BODY_PAIR_REDUCTION_SLOTS,
            "hashtable_bytes": self.hashtable.keys.size * 8 + self.hashtable.active_slots.size * 4,
            "slot_value_bytes": self.ht_values.size * 8,
            "per_contact_cache_bytes": n * (4 + 4 + 4 + 4 + 8) + self.contact_incumbent.size * 4,
            "hysteresis_snapshot_bytes": (
                self.prev_keys.size * 8 + self.prev_winner_pos.size * 4 + self.prev_active_slots.size * 4
            ),
            "gather_scratch_owned_bytes": owned,
        }
