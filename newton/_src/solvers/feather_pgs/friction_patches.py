# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Solver-owned friction regions, independent of collision contact identities.

Normal contacts are never reduced here. Each compatible region selects up to two
friction locations, with an equal share of the region's total normal impulse.
Regions require matching friction materials, nearly aligned normals, nearby
contact planes, and connected shape bounding spheres on each body. The latter
is a conservative test across convex seams, not an exact surface-connectivity
query. Length tolerances scale with the smaller body's bounding radius; angular
tolerance is about six degrees. Normals remain individual unilateral rows.

Only body-local anchors and their tangent impulses persist across frames. Pair
identity and geometric compatibility correlate them without a collision matcher.
Saturated friction with opposing slip releases history; saturation alone does
not. All matching tolerances are internal so correction strength is the only
new user control.

Sorting body pairs makes construction local to a pair and preserves CUDA graph
capture: all buffers and the radix-sort workspace have fixed capacity.
"""

import numpy as np
import warp as wp


@wp.struct
class FrictionPatches:
    """Per-contact view consumed by the existing contact row builders."""

    enabled: int
    weight: wp.array[float]
    next_contact: wp.array[int]
    point_a: wp.array[wp.vec3]
    point_b: wp.array[wp.vec3]
    phi: wp.array[wp.vec2]


@wp.struct
class _PatchFrame:
    keys: wp.array[wp.int64]
    indices: wp.array[int]
    center: wp.array[wp.vec3]
    normal: wp.array[wp.vec3]
    mu: wp.array[wp.vec2]
    radius: wp.array[float]
    body_a: wp.array[int]
    body_b: wp.array[int]
    shape_a: wp.array[int]
    shape_b: wp.array[int]
    flipped: wp.array[int]
    owner: wp.array[int]
    anchor_a: wp.array[wp.vec3]
    anchor_b: wp.array[wp.vec3]
    valid: wp.array[int]
    used: wp.array[int]
    source: wp.array[int]
    eligible: wp.array[int]
    tangent_impulse: wp.array[wp.vec3]


@wp.func
def _world_point(q: wp.array[wp.transform], body: int, point: wp.vec3):
    result = point
    if body >= 0:
        result = wp.transform_point(q[body], point)
    return result


@wp.func
def _local_point(q: wp.array[wp.transform], body: int, point: wp.vec3):
    result = point
    if body >= 0:
        result = wp.transform_point(wp.transform_inverse(q[body]), point)
    return result


@wp.kernel(enable_backward=False)
def _prepare(
    count: wp.array[int],
    shape0: wp.array[int],
    shape1: wp.array[int],
    point0: wp.array[wp.vec3],
    point1: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    margin0: wp.array[float],
    margin1: wp.array[float],
    shape_body: wp.array[int],
    shape_mu: wp.array[float],
    body_radius: wp.array[float],
    q: wp.array[wp.transform],
    body_to_articulation: wp.array[int],
    is_free_rigid: wp.array[int],
    contact_gap_gate: float,
    same_articulation_gap_gate: float,
    articulation_pair_gap_gate: float,
    friction_gap: float,
    friction_articulation_pairs_only: int,
    frame: _PatchFrame,
    patches: FrictionPatches,
):
    c = wp.tid()
    frame.indices[c] = c
    frame.keys[c] = wp.int64(0x7FFFFFFFFFFFFFFF)
    frame.owner[c] = -1
    frame.valid[c] = 0
    frame.used[c] = 0
    frame.source[c] = -1
    frame.eligible[c] = 0
    frame.tangent_impulse[c] = wp.vec3(0.0)
    patches.weight[c] = 0.0
    patches.next_contact[c] = -1
    patches.phi[c] = wp.vec2(0.0)
    if count[0] > shape0.shape[0] or c >= count[0] or c >= shape0.shape[0]:
        return
    sa = shape0[c]
    sb = shape1[c]
    a = int(-1)
    b = int(-1)
    if sa >= 0:
        a = shape_body[sa]
    if sb >= 0:
        b = shape_body[sb]
    ka = a + 1
    kb = b + 1
    if a < 0:
        ka = q.shape[0] + (sa if sa >= 0 else shape_body.shape[0]) + 1
    if b < 0:
        kb = q.shape[0] + (sb if sb >= 0 else shape_body.shape[0]) + 1
    n = -normal[c]
    pa = _world_point(q, a, point0[c]) - margin0[c] * n
    pb = _world_point(q, b, point1[c]) + margin1[c] * n
    gap = wp.dot(n, pa - pb)
    art_a = int(-1)
    art_b = int(-1)
    if a >= 0 and a < body_to_articulation.shape[0]:
        art_a = body_to_articulation[a]
    if b >= 0 and b < body_to_articulation.shape[0]:
        art_b = body_to_articulation[b]
    a_non_free = art_a >= 0 and is_free_rigid[art_a] == 0
    b_non_free = art_b >= 0 and is_free_rigid[art_b] == 0
    # Match the allocator's gap filters before choosing anchors. Normals that
    # remain eligible can still support the patch without carrying an anchor.
    eligible = contact_gap_gate <= 0.0 or gap <= contact_gap_gate
    if a_non_free and b_non_free:
        eligible = eligible and (articulation_pair_gap_gate <= 0.0 or gap <= articulation_pair_gap_gate)
        if art_a == art_b:
            eligible = eligible and (same_articulation_gap_gate <= 0.0 or gap <= same_articulation_gap_gate)
    if friction_articulation_pairs_only == 0 or (a_non_free and b_non_free):
        eligible = eligible and gap <= friction_gap
    frame.eligible[c] = int(eligible)
    flip = int(ka > kb)
    if flip != 0:
        t = a
        a = b
        b = t
        t = ka
        ka = kb
        kb = t
        t = sa
        sa = sb
        sb = t
        n = -n
    frame.keys[c] = (wp.int64(ka) << wp.int64(32)) | wp.int64(kb)
    frame.body_a[c] = a
    frame.body_b[c] = b
    frame.shape_a[c] = sa
    frame.shape_b[c] = sb
    frame.flipped[c] = flip
    frame.center[c] = 0.5 * (pa + pb)
    frame.normal[c] = n
    mu_a = float(0.0)
    mu_b = float(0.0)
    if sa >= 0:
        mu_a = shape_mu[sa]
    if sb >= 0:
        mu_b = shape_mu[sb]
    frame.mu[c] = wp.vec2(mu_a, mu_b)
    r = float(1.0e10)
    if a >= 0:
        r = wp.min(r, body_radius[a])
    if b >= 0:
        r = wp.min(r, body_radius[b])
    frame.radius[c] = wp.max(r, 1.0e-4)


@wp.func
def _compatible(frame: _PatchFrame, seed: int, c: int):
    d = frame.center[c] - frame.center[seed]
    r = frame.radius[seed]
    return (
        wp.dot(frame.normal[seed], frame.normal[c]) >= 0.995
        and frame.mu[seed][0] == frame.mu[c][0]
        and frame.mu[seed][1] == frame.mu[c][1]
        and wp.abs(wp.dot(d, frame.normal[seed])) <= 0.02 * r
        and wp.length_sq(d) <= 4.0 * r * r
    )


@wp.func
def _shapes_adjacent(a: int, b: int, transforms: wp.array[wp.transform], radii: wp.array[float]):
    if a == b:
        return True
    if a < 0 or b < 0:
        return False
    radius = radii[a] + radii[b]
    delta = wp.transform_get_translation(transforms[a]) - wp.transform_get_translation(transforms[b])
    return wp.length_sq(delta) <= (1.0 + 1.0e-5) * radius * radius


@wp.func
def _geometry_adjacent(sa: int, sb: int, ta: int, tb: int, transforms: wp.array[wp.transform], radii: wp.array[float]):
    """Conservative adjacency in each body's local frame, including convex seams."""
    return _shapes_adjacent(sa, ta, transforms, radii) and _shapes_adjacent(sb, tb, transforms, radii)


@wp.kernel(enable_backward=False)
def _build(
    count: wp.array[int],
    q: wp.array[wp.transform],
    shape_transform: wp.array[wp.transform],
    shape_radius: wp.array[float],
    frame: _PatchFrame,
    prev: _PatchFrame,
    patches: FrictionPatches,
):
    """One thread per sorted body pair; writes disjoint current/previous records."""
    start = wp.tid()
    end = wp.min(count[0], frame.center.shape[0])
    if start >= end or (start > 0 and frame.keys[start - 1] == frame.keys[start]):
        return
    key = frame.keys[start]
    if key == wp.int64(0x7FFFFFFFFFFFFFFF):
        return
    stop = start + 1
    while stop < end and frame.keys[stop] == key:
        stop += 1

    # Binary search the previous pair, avoiding a scan over other worlds/pairs.
    lo = int(0)
    hi = prev.center.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if prev.keys[mid] < key:
            lo = mid + 1
        else:
            hi = mid
    prev_start = lo
    prev_stop = lo
    while prev_stop < prev.center.shape[0] and prev.keys[prev_stop] == key:
        prev.used[prev.indices[prev_stop]] = 0
        prev_stop += 1

    remaining = stop - start
    for index in range(start, stop):
        seed = frame.indices[index]
        if frame.owner[seed] >= 0:
            continue
        frame.owner[seed] = seed
        remaining -= 1
        tail = seed
        first = int(-1)
        if frame.eligible[seed] != 0:
            first = seed
        cursor = seed
        expanded_a = int(-2)
        expanded_b = int(-2)
        # Flood through adjacent convex pieces. The queue is the contact list
        # itself; no allocation, atomics, or fixed patch-count limit is needed.
        while cursor >= 0 and remaining > 0:
            sa = frame.shape_a[cursor]
            sb = frame.shape_b[cursor]
            if sa != expanded_a or sb != expanded_b:
                for j in range(index + 1, stop):
                    c = frame.indices[j]
                    if frame.owner[c] < 0 and _compatible(frame, seed, c):
                        if _geometry_adjacent(
                            sa, sb, frame.shape_a[c], frame.shape_b[c], shape_transform, shape_radius
                        ):
                            frame.owner[c] = seed
                            remaining -= 1
                            patches.next_contact[tail] = c
                            tail = c
                            if frame.eligible[c] != 0:
                                if first < 0:
                                    first = c
                                else:
                                    position = frame.center[c]
                                    p0 = frame.center[first]
                                    if position[0] < p0[0] or (
                                        position[0] == p0[0]
                                        and (position[1] < p0[1] or (position[1] == p0[1] and position[2] < p0[2]))
                                    ):
                                        first = c
                expanded_a = sa
                expanded_b = sb
            cursor = patches.next_contact[cursor]
        patches.next_contact[tail] = seed
        if first < 0:
            continue
        second = first
        separation = float(0.0)
        for j in range(index, stop):
            c = frame.indices[j]
            if frame.owner[c] == seed and frame.eligible[c] != 0:
                distance = wp.length_sq(frame.center[c] - frame.center[first])
                if distance > separation:
                    separation = distance
                    second = c
        anchors = int(1)
        if separation > 1.0e-8 * frame.radius[seed] * frame.radius[seed]:
            anchors = 2
        for aidx in range(anchors):
            c = first
            if aidx == 1:
                c = second
            a = frame.body_a[c]
            b = frame.body_b[c]
            n = frame.normal[c]
            r = frame.radius[c]
            chosen = int(-1)
            nearest = float(1.0e30)
            for j in range(prev_start, prev_stop):
                p = prev.indices[j]
                if (
                    prev.valid[p] == 0
                    or prev.used[p] != 0
                    or prev.mu[p][0] != frame.mu[c][0]
                    or prev.mu[p][1] != frame.mu[c][1]
                ):
                    continue
                connected = _geometry_adjacent(
                    prev.shape_a[p], prev.shape_b[p], frame.shape_a[c], frame.shape_b[c], shape_transform, shape_radius
                )
                if not connected:
                    for k in range(index, stop):
                        member = frame.indices[k]
                        if frame.owner[member] == seed and _geometry_adjacent(
                            prev.shape_a[p],
                            prev.shape_b[p],
                            frame.shape_a[member],
                            frame.shape_b[member],
                            shape_transform,
                            shape_radius,
                        ):
                            connected = True
                            break
                if not connected:
                    continue
                old_n = prev.normal[p]
                if a >= 0:
                    old_n = wp.transform_vector(q[a], old_n)
                pa = _world_point(q, a, prev.anchor_a[p])
                pb = _world_point(q, b, prev.anchor_b[p])
                delta = pa - pb
                tangent_delta = delta - n * wp.dot(delta, n)
                distance = wp.length_sq(0.5 * (pa + pb) - frame.center[c])
                if (
                    wp.dot(old_n, n) >= 0.995
                    and wp.length_sq(tangent_delta) <= 0.0004 * r * r
                    and wp.abs(wp.dot(delta, n)) <= 0.1 * r
                    and wp.abs(wp.dot(0.5 * (pa + pb) - frame.center[c], n)) <= 0.05 * r
                    and distance <= 4.0 * r * r
                    and distance < nearest
                ):
                    chosen = p
                    nearest = distance
            pa = frame.center[c]
            pb = pa
            if chosen >= 0:
                prev.used[chosen] = 1
                pa = _world_point(q, a, prev.anchor_a[chosen])
                pb = _world_point(q, b, prev.anchor_b[chosen])
            frame.source[c] = chosen
            frame.anchor_a[c] = _local_point(q, a, pa)
            frame.anchor_b[c] = _local_point(q, b, pb)
            frame.valid[c] = 1
            patches.weight[c] = 1.0 / float(anchors)
            if frame.flipped[c] != 0:
                temp = pa
                pa = pb
                pb = temp
                n = -n
            patches.point_a[c] = pa
            patches.point_b[c] = pb
            t0 = wp.cross(n, wp.vec3(1.0, 0.0, 0.0))
            if wp.length_sq(t0) < 1.0e-12:
                t0 = wp.cross(n, wp.vec3(0.0, 1.0, 0.0))
            t0 = wp.normalize(t0)
            t1 = wp.normalize(wp.cross(n, t0))
            patches.phi[c] = wp.vec2(wp.dot(t0, pa - pb), wp.dot(t1, pa - pb))


@wp.kernel(enable_backward=False)
def _store_history(q: wp.array[wp.transform], frame: _PatchFrame, prev: _PatchFrame):
    """Carry fixed-capacity history in one launch, including its sorted pair index."""
    c = wp.tid()
    normal = frame.normal[c]
    if frame.valid[c] != 0 and frame.body_a[c] >= 0:
        normal = wp.transform_vector(wp.transform_inverse(q[frame.body_a[c]]), normal)
    prev.keys[c] = frame.keys[c]
    prev.indices[c] = frame.indices[c]
    prev.normal[c] = normal
    prev.mu[c] = frame.mu[c]
    prev.shape_a[c] = frame.shape_a[c]
    prev.shape_b[c] = frame.shape_b[c]
    prev.anchor_a[c] = frame.anchor_a[c]
    prev.anchor_b[c] = frame.anchor_b[c]
    prev.valid[c] = frame.valid[c]
    prev.tangent_impulse[c] = frame.tangent_impulse[c]


class _FrictionPatchState:
    """Own preallocated patch frames; never read device counters on the host."""

    def __init__(self, model, capacity, enabled, phi):
        self.view = FrictionPatches()
        self.view.enabled = int(enabled)
        self.view.phi = phi
        self.capacity = capacity
        device = model.device
        n = capacity if enabled else 0
        self.view.weight = wp.zeros(n, dtype=float, device=device)
        self.view.next_contact = wp.full(n, -1, dtype=int, device=device)
        self.view.point_a = wp.zeros(n, dtype=wp.vec3, device=device)
        self.view.point_b = wp.zeros(n, dtype=wp.vec3, device=device)
        if not enabled:
            return
        self.body_radius = wp.zeros(model.body_count, dtype=float, device=device)
        self.update_geometry(model)
        self.current = self._frame(capacity, device)
        self.previous = self._frame(capacity, device)
        self.previous_world = wp.full(capacity, -1, dtype=int, device=device)

    def update_geometry(self, model):
        """Refresh geometry scales after explicit shape-property updates."""
        radii = np.zeros(model.body_count, dtype=np.float32)
        for body, radius, transform in zip(
            model.shape_body.numpy(), model.shape_collision_radius.numpy(), model.shape_transform.numpy(), strict=True
        ):
            if body >= 0:
                radii[body] = max(radii[body], radius + np.linalg.norm(transform[:3]))
        self.body_radius.assign(radii)

    @staticmethod
    def _frame(n, device):
        frame = _PatchFrame()
        frame.keys = wp.full(2 * n, 0x7FFFFFFFFFFFFFFF, dtype=wp.int64, device=device)
        frame.indices = wp.zeros(2 * n, dtype=int, device=device)
        for field in ("center", "normal", "anchor_a", "anchor_b", "tangent_impulse"):
            setattr(frame, field, wp.zeros(n, dtype=wp.vec3, device=device))
        frame.mu = wp.zeros(n, dtype=wp.vec2, device=device)
        frame.radius = wp.zeros(n, dtype=float, device=device)
        for field in (
            "body_a",
            "body_b",
            "shape_a",
            "shape_b",
            "flipped",
            "owner",
            "valid",
            "used",
            "source",
            "eligible",
        ):
            setattr(frame, field, wp.zeros(n, dtype=int, device=device))
        return frame

    def build(
        self,
        model,
        state,
        contacts,
        *,
        body_to_articulation=None,
        is_free_rigid=None,
        contact_gap_gate=0.0,
        same_articulation_gap_gate=0.0,
        articulation_pair_gap_gate=0.0,
        friction_gap=float("inf"),
        friction_articulation_pairs_only=False,
    ):
        wp.launch(
            _prepare,
            dim=self.capacity,
            inputs=[
                contacts.rigid_contact_count,
                contacts.rigid_contact_shape0,
                contacts.rigid_contact_shape1,
                contacts.rigid_contact_point0,
                contacts.rigid_contact_point1,
                contacts.rigid_contact_normal,
                contacts.rigid_contact_margin0,
                contacts.rigid_contact_margin1,
                model.shape_body,
                model.shape_material_mu,
                self.body_radius,
                state.body_q,
                body_to_articulation,
                is_free_rigid,
                contact_gap_gate,
                same_articulation_gap_gate,
                articulation_pair_gap_gate,
                friction_gap,
                int(friction_articulation_pairs_only),
                self.current,
                self.view,
            ],
            device=model.device,
        )
        wp.utils.radix_sort_pairs(self.current.keys, self.current.indices, self.capacity)
        wp.launch(
            _build,
            dim=self.capacity,
            inputs=[
                contacts.rigid_contact_count,
                state.body_q,
                model.shape_transform,
                model.shape_collision_radius,
                self.current,
                self.previous,
                self.view,
            ],
            device=model.device,
        )

    def store(self, state):
        wp.launch(
            _store_history,
            dim=self.capacity,
            inputs=[state.body_q, self.current, self.previous],
            device=state.body_q.device,
        )


@wp.func
def patch_normal_load(parents: wp.array2d[int], impulses: wp.array2d[float], world: int, first: int):
    """Sum a region's live normal impulses; -1 terminates ordinary point contacts.

    A contact row's parent field links the next normal in its region (a circular
    list). Friction rows still point to their adjacent normal row. Other row
    families retain their original parent semantics.
    """
    load = impulses[world, first]
    row = parents[world, first]
    while row >= 0 and row != first:
        load += impulses[world, row]
        row = parents[world, row]
    return load


@wp.kernel(enable_backward=False)
def link_patch_rows(
    count: wp.array[int],
    patches: FrictionPatches,
    world: wp.array[int],
    slot: wp.array[int],
    path: wp.array[int],
    slots_needed: wp.array[int],
    route: int,
    parents: wp.array2d[int],
    mu: wp.array2d[float],
):
    """Link only allocated rows and divide the load among surviving anchors."""
    c = wp.tid()
    if c >= count[0] or slot[c] < 0 or path[c] != route:
        return
    row = patches.next_contact[c]
    if row < 0:
        return
    next_slot = slot[c]
    anchors = int(patches.weight[c] > 0.0 and slots_needed[c] == 3)
    while row != c:
        if slot[row] >= 0 and path[row] == route and world[row] == world[c]:
            if next_slot == slot[c]:
                next_slot = slot[row]
            if patches.weight[row] > 0.0 and slots_needed[row] == 3:
                anchors += 1
            if patches.weight[c] == 0.0:
                break
        row = patches.next_contact[row]
    parents[world[c], slot[c]] = next_slot
    if patches.weight[c] > 0.0 and slots_needed[c] == 3 and anchors > 0:
        for t in range(1, 3):
            mu[world[c], slot[c] + t] /= patches.weight[c] * float(anchors)


@wp.func
def _tangents(n: wp.vec3):
    t0 = wp.cross(n, wp.vec3(1.0, 0.0, 0.0))
    if wp.length_sq(t0) < 1.0e-12:
        t0 = wp.cross(n, wp.vec3(0.0, 1.0, 0.0))
    t0 = wp.normalize(t0)
    return t0, wp.normalize(wp.cross(n, t0))


@wp.kernel(enable_backward=False)
def seed_patch_impulses(
    count: wp.array[int],
    frame: _PatchFrame,
    prev: _PatchFrame,
    q: wp.array[wp.transform],
    world: wp.array[int],
    slot: wp.array[int],
    path: wp.array[int],
    slots_needed: wp.array[int],
    route: int,
    parents: wp.array2d[int],
    mu: wp.array2d[float],
    impulses: wp.array2d[float],
    scale: float,
):
    """Transport cached patch impulses after all contact normals have been seeded."""
    c = wp.tid()
    if c >= count[0] or path[c] != route or slot[c] < 0 or slots_needed[c] != 3:
        return
    tangent = wp.vec3(0.0)
    source = frame.source[c]
    if source >= 0:
        tangent = prev.tangent_impulse[source] * scale
        if frame.body_a[c] >= 0:
            tangent = wp.transform_vector(q[frame.body_a[c]], tangent)
    n = frame.normal[c]
    if frame.flipped[c] != 0:
        n = -n
        tangent = -tangent
    t0, t1 = _tangents(n)
    value = wp.vec2(wp.dot(tangent, t0), wp.dot(tangent, t1))
    w = world[c]
    s = slot[c]
    radius = wp.max(mu[w, s + 1] * patch_normal_load(parents, impulses, w, s), 0.0)
    magnitude = wp.length(value)
    if magnitude > radius and magnitude > 0.0:
        value *= radius / magnitude
    impulses[w, s + 1] = value[0]
    impulses[w, s + 2] = value[1]


@wp.func
def _point_velocity(
    q: wp.array[wp.transform], qd: wp.array[wp.spatial_vector], com: wp.array[wp.vec3], body: int, anchor: wp.vec3
):
    velocity = wp.vec3(0.0)
    if body >= 0:
        offset = wp.transform_vector(q[body], anchor - com[body])
        velocity = wp.spatial_top(qd[body]) + wp.cross(wp.spatial_bottom(qd[body]), offset)
    return velocity


@wp.kernel(enable_backward=False)
def finish_patch_impulses(
    count: wp.array[int],
    frame: _PatchFrame,
    q: wp.array[wp.transform],
    q_out: wp.array[wp.transform],
    qd_out: wp.array[wp.spatial_vector],
    com: wp.array[wp.vec3],
    world: wp.array[int],
    slot: wp.array[int],
    path: wp.array[int],
    slots_needed: wp.array[int],
    route: int,
    parents: wp.array2d[int],
    mu: wp.array2d[float],
    impulses: wp.array2d[float],
    dt: float,
):
    """Cache solved friction and release saturated anchors with tangential motion.

    The tolerance is relative to geometry and timestep. Merely resting at the
    Coulomb limit does not prove sliding, so it must not erase static history.
    """
    c = wp.tid()
    if c >= count[0] or frame.valid[c] == 0:
        return
    if slot[c] < 0 or slots_needed[c] != 3:
        frame.valid[c] = 0
        return
    if path[c] != route:
        return
    w = world[c]
    s = slot[c]
    n = frame.normal[c]
    t0, t1 = _tangents(n)
    if frame.flipped[c] != 0:
        t0, t1 = _tangents(-n)
    tangent = impulses[w, s + 1] * t0 + impulses[w, s + 2] * t1
    if frame.flipped[c] != 0:
        tangent = -tangent
    stored = tangent
    a = frame.body_a[c]
    b = frame.body_b[c]
    if a >= 0:
        stored = wp.transform_vector(wp.transform_inverse(q[a]), tangent)
    frame.tangent_impulse[c] = stored
    radius = wp.max(mu[w, s + 1] * patch_normal_load(parents, impulses, w, s), 0.0)
    speed = _point_velocity(q_out, qd_out, com, a, frame.anchor_a[c]) - _point_velocity(
        q_out, qd_out, com, b, frame.anchor_b[c]
    )
    # Check opposing tangential motion to distinguish slipping from impending slip.
    if radius > 0.0 and wp.length(tangent) >= (1.0 - 1.0e-5) * radius:
        if wp.dot(speed, tangent) < -radius * (1.0e-5 * frame.radius[c] / dt):
            frame.valid[c] = 0
