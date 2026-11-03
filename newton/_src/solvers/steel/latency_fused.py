# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Latency mode for the propagation-colored matrix-free PGS loop: one launch per iteration loop.

Today's loop launches about a dozen kernels per PGS iteration (row phases 3 and 4, tree and free-body
velocity refreshes, the cache snapshot, the colored contact sweep, the cached-response GEMV, the tree
impulse propagation, the free-body flush, row phase 5). With few worlds each launch holds a handful of
warps, so the frame is a long chain of small kernels. Every one of those passes reads and writes only its
own world's data, so one block per world can run the whole loop in a single launch.

The block's warps take fixed roles and call the existing device functions (each factory exposes its
``kernel.native``), so the math is the same code:

* warp 0: row phases (3, 4, then 5), the cache snapshot, and the GEMV; every warp the colored sweep
  (a color wider than one warp spreads over the block);
* warp 1 + i: tree variant ``i`` (a size group's warp kernels), refresh and impulse propagation for the
  world's articulations of that group, one after another;
* the last warp: free-rigid body velocity refresh and flush.

Per iteration the order is the chain's, with block barriers between dependent stages:

    w0: phase 3, phase 4 | sync | trees, free bodies: refresh | sync | w0: snapshot, colored sweep | sync
    | w0: GEMV, trees: propagate, free bodies: flush (disjoint data) | sync | w0: phase 5

Global writes are visible block-wide after ``__syncthreads``, and the device functions read data written
in the loop only through plain loads, so results match the chain bit for bit.
"""

from __future__ import annotations

import hashlib
from typing import Any

import warp as wp

# Tree variants (size groups with warp kernels) one fused kernel can carry.
MAX_TREE_SLOTS = 4


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__syncthreads();
#endif
""")
def _block_sync(): ...


@wp.struct
class ColoredArgs:
    constraint_count: wp.array(dtype=int)
    body_a: wp.array2d(dtype=int)
    body_b: wp.array2d(dtype=int)
    MiJt_a: wp.array3d(dtype=float)
    MiJt_b: wp.array3d(dtype=float)
    J_a: wp.array3d(dtype=float)
    J_b: wp.array3d(dtype=float)
    eff_mass_inv: wp.array2d(dtype=float)
    rhs: wp.array2d(dtype=float)
    row_w: wp.array2d(dtype=float)
    row_type: wp.array2d(dtype=int)
    row_parent: wp.array2d(dtype=int)
    row_mu: wp.array2d(dtype=float)
    compliance_gamma: wp.array2d(dtype=float)
    torsion_first: wp.array2d(dtype=int)
    torsion_next: wp.array2d(dtype=int)
    torsion_row_radius: wp.array2d(dtype=float)
    row_order: wp.array(dtype=int)
    color_offsets: wp.array(dtype=int)
    body_list: wp.array2d(dtype=int)
    body_count: wp.array(dtype=int)
    body_local_slot: wp.array(dtype=int)
    unit_meta: wp.array(dtype=int)
    unit_friction_setup: wp.array(dtype=float)
    unit_payload: wp.array(dtype=float)
    ring_overflow: wp.array(dtype=int)
    impulses: wp.array2d(dtype=float)


@wp.struct
class RowArgs:
    constraint_count: wp.array(dtype=int)
    dense_phase_bounds: wp.array2d(dtype=int)
    local_solve_owner: wp.array(dtype=int)
    world_dof_indices: wp.array2d(dtype=int)
    rhs: wp.array2d(dtype=float)
    diag: wp.array2d(dtype=float)
    row_w: wp.array2d(dtype=float)
    impulses: wp.array2d(dtype=float)
    J_world: wp.array3d(dtype=float)
    Y_world: wp.array3d(dtype=float)
    row_type: wp.array2d(dtype=int)
    row_parent: wp.array2d(dtype=int)
    row_mu: wp.array2d(dtype=float)
    torsion_group: wp.array2d(dtype=int)
    torsion_row_radius: wp.array2d(dtype=float)
    drive_target_vel_bias: wp.array2d(dtype=float)
    drive_vel_multiplier: wp.array2d(dtype=float)
    drive_impulse_multiplier: wp.array2d(dtype=float)
    drive_max_impulse: wp.array2d(dtype=float)
    drive_vel_limit: wp.array2d(dtype=float)
    mf_constraint_count: wp.array(dtype=int)
    mf_contact_rows_end: wp.array(dtype=int)
    mf_meta: wp.array2d(dtype=int)
    mf_impulses: wp.array2d(dtype=float)
    mf_J_a: wp.array3d(dtype=float)
    mf_J_b: wp.array3d(dtype=float)
    mf_MiJt_a: wp.array3d(dtype=float)
    mf_MiJt_b: wp.array3d(dtype=float)
    mf_row_mu: wp.array2d(dtype=float)
    mf_row_w: wp.array2d(dtype=float)
    dense_gamma: wp.array2d(dtype=float)
    mf_gamma: wp.array2d(dtype=float)
    momentum_scratch: wp.array2d(dtype=float)


@wp.struct
class CacheArgs:
    world_flag: wp.array(dtype=int)
    body_count: wp.array(dtype=int)
    art_eligible: wp.array(dtype=int)
    R: wp.array4d(dtype=float)
    B: wp.array4d(dtype=float)
    qd_base: wp.array3d(dtype=float)


@wp.struct
class TreeArgs:
    art_to_world: wp.array(dtype=int)
    dense_contact_world_flag: wp.array(dtype=int)
    articulation_start: wp.array(dtype=int)
    joint_parent: wp.array(dtype=int)
    joint_child: wp.array(dtype=int)
    joint_qd_start: wp.array(dtype=int)
    joint_parent_slot: wp.array(dtype=int)
    S_flat: wp.array2d(dtype=float)
    com_rel: wp.array2d(dtype=float)
    U: wp.array2d(dtype=float)
    D_inv: wp.array3d(dtype=float)


@wp.struct
class TreeSlot:
    """One size group with warp tree kernels: its group-to-articulation map and per-world CSR."""

    group_to_art: wp.array(dtype=int)
    guard_flag: wp.array(dtype=int)
    world_offsets: wp.array(dtype=int)
    world_groups: wp.array(dtype=int)


@wp.struct
class FreeArgs:
    """Free-rigid bodies per world (CSR) and their root DOFs."""

    world_offsets: wp.array(dtype=int)
    bodies: wp.array(dtype=int)
    root_dof: wp.array(dtype=int)


@wp.func
def _free_refresh(
    world: int, lane: int, free: FreeArgs, v_out: wp.array(dtype=float), body_qd: wp.array2d(dtype=float)
):
    """``refresh_propagation_free_body_qd_from_vout`` for one world's free-rigid bodies."""
    for i in range(free.world_offsets[world] + lane, free.world_offsets[world + 1], 32):
        body = free.bodies[i]
        dof = free.root_dof[i]
        for r in range(6):
            body_qd[body, r] = v_out[dof + r]


@wp.func
def _free_flush(
    world: int,
    lane: int,
    free: FreeArgs,
    body_qd: wp.array2d(dtype=float),
    body_impulses: wp.array2d(dtype=float),
    v_out: wp.array(dtype=float),
):
    """``flush_propagation_free_body_qd_to_vout`` for one world's free-rigid bodies."""
    for i in range(free.world_offsets[world] + lane, free.world_offsets[world + 1], 32):
        body = free.bodies[i]
        dof = free.root_dof[i]
        for r in range(6):
            v_out[dof + r] = body_qd[body, r]
        for r in range(6):
            body_impulses[body, r] = 0.0


@wp.func
def _cache_snapshot(
    world: int,
    lane: int,
    cache: CacheArgs,
    cache_max_bodies: int,
    body_list: wp.array2d(dtype=int),
    body_qd: wp.array2d(dtype=float),
):
    """``snapshot_propagation_cache_qd_base`` for one world."""
    if cache.world_flag[world] != 0:
        n = wp.min(cache.body_count[world], cache_max_bodies)
        for slot in range(lane, n, 32):
            body = body_list[world, slot]
            if body >= 0:
                for r in range(6):
                    cache.qd_base[world, slot, r] = body_qd[body, r]


def warp_count(n_tree_slots: int) -> int:
    """Warps per block: the solver warp, one per tree variant, and the free-body warp."""
    return 2 + n_tree_slots


def get_fused_iteration_kernel(
    colored_native: Any,
    row_native: Any,
    gemv_native: Any | None,
    tree_natives: tuple[tuple[Any, Any], ...],
    has_phase3: bool,
    has_phase5: bool,
) -> wp.Kernel:
    """Build the fused loop kernel for one solver configuration.

    ``colored_native`` is the block-wide variant (built with ``block_threads`` equal to the launch's block
    size), which every warp calls so wide colors use the whole block.
    ``tree_natives`` holds ``(propagate, refresh)`` device functions per tree variant (at most
    :data:`MAX_TREE_SLOTS`); ``gemv_native`` is ``None`` without cached responses. Launch it with
    ``wp.launch_tiled(dim=[world_count], block_dim=32 * warp_count(len(tree_natives)))``.
    """
    n_slots = len(tree_natives)
    assert 0 <= n_slots <= MAX_TREE_SLOTS
    # unused slots get placeholders; their branches compile out
    filler = tree_natives[0] if n_slots else (_free_refresh, _free_refresh)
    natives = list(tree_natives) + [filler] * (MAX_TREE_SLOTS - n_slots)
    has_cache = gemv_native is not None
    if gemv_native is None:
        gemv_native = colored_native  # placeholder; the GEMV branch compiles out
    free_warp = 1 + n_slots

    def fused_template(
        world_count: int,
        iterations: int,
        iteration_offset: int,
        friction_start_iteration: int,
        omega: float,
        colored_regularize: int,
        row_regularize: int,
        freeze_drive_rows: int,
        force_refresh: int,
        cache_max_bodies: int,
        colored: ColoredArgs,
        rows: RowArgs,
        cache: CacheArgs,
        tree: TreeArgs,
        slot0: TreeSlot,
        slot1: TreeSlot,
        slot2: TreeSlot,
        slot3: TreeSlot,
        free: FreeArgs,
        body_to_articulation: wp.array(dtype=int),
        body_qd: wp.array2d(dtype=float),
        body_impulses: wp.array2d(dtype=float),
        v_out: wp.array(dtype=float),
    ):
        world, t = wp.tid()
        warp_id = t // 32
        lane = t - warp_id * 32
        for i in range(iterations):
            g = iteration_offset + i
            # row phases 3 (drives, position limits, joint friction) and 4 (dense / matrix-free contacts)
            if warp_id == 0:
                if wp.static(has_phase3):
                    wp.static(row_native)(
                        world, rows.constraint_count, rows.dense_phase_bounds, rows.local_solve_owner,
                        rows.world_dof_indices, rows.rhs, rows.diag, rows.row_w, rows.impulses, rows.J_world,
                        rows.Y_world, rows.row_type, rows.row_parent, rows.row_mu, rows.torsion_group,
                        rows.torsion_row_radius, rows.drive_target_vel_bias, rows.drive_vel_multiplier,
                        rows.drive_impulse_multiplier, rows.drive_max_impulse, rows.drive_vel_limit,
                        rows.mf_constraint_count, rows.mf_contact_rows_end, rows.mf_meta, rows.mf_impulses,
                        rows.mf_J_a, rows.mf_J_b, rows.mf_MiJt_a, rows.mf_MiJt_b, rows.mf_row_mu, rows.mf_row_w,
                        rows.dense_gamma, rows.mf_gamma, 1, omega, row_regularize, 3, friction_start_iteration, g,
                        freeze_drive_rows, rows.momentum_scratch, v_out,
                    )  # fmt: skip
                wp.static(row_native)(
                    world, rows.constraint_count, rows.dense_phase_bounds, rows.local_solve_owner,
                    rows.world_dof_indices, rows.rhs, rows.diag, rows.row_w, rows.impulses, rows.J_world,
                    rows.Y_world, rows.row_type, rows.row_parent, rows.row_mu, rows.torsion_group,
                    rows.torsion_row_radius, rows.drive_target_vel_bias, rows.drive_vel_multiplier,
                    rows.drive_impulse_multiplier, rows.drive_max_impulse, rows.drive_vel_limit,
                    rows.mf_constraint_count, rows.mf_contact_rows_end, rows.mf_meta, rows.mf_impulses,
                    rows.mf_J_a, rows.mf_J_b, rows.mf_MiJt_a, rows.mf_MiJt_b, rows.mf_row_mu, rows.mf_row_w,
                    rows.dense_gamma, rows.mf_gamma, 1, omega, row_regularize, 4, friction_start_iteration, g,
                    freeze_drive_rows, rows.momentum_scratch, v_out,
                )  # fmt: skip
            _block_sync()
            # live body twists from the generalized velocities
            if wp.static(n_slots > 0):
                if warp_id == 1:
                    for k in range(slot0.world_offsets[world], slot0.world_offsets[world + 1]):
                        wp.static(natives[0][1])(
                            slot0.world_groups[k], slot0.group_to_art, tree.art_to_world,
                            tree.dense_contact_world_flag, force_refresh, tree.articulation_start, tree.joint_parent,
                            tree.joint_child, tree.joint_qd_start, tree.joint_parent_slot, tree.S_flat, tree.com_rel,
                            v_out, body_qd,
                        )  # fmt: skip
            if wp.static(n_slots > 1):
                if warp_id == 2:
                    for k in range(slot1.world_offsets[world], slot1.world_offsets[world + 1]):
                        wp.static(natives[1][1])(
                            slot1.world_groups[k], slot1.group_to_art, tree.art_to_world,
                            tree.dense_contact_world_flag, force_refresh, tree.articulation_start, tree.joint_parent,
                            tree.joint_child, tree.joint_qd_start, tree.joint_parent_slot, tree.S_flat, tree.com_rel,
                            v_out, body_qd,
                        )  # fmt: skip
            if wp.static(n_slots > 2):
                if warp_id == 3:
                    for k in range(slot2.world_offsets[world], slot2.world_offsets[world + 1]):
                        wp.static(natives[2][1])(
                            slot2.world_groups[k], slot2.group_to_art, tree.art_to_world,
                            tree.dense_contact_world_flag, force_refresh, tree.articulation_start, tree.joint_parent,
                            tree.joint_child, tree.joint_qd_start, tree.joint_parent_slot, tree.S_flat, tree.com_rel,
                            v_out, body_qd,
                        )  # fmt: skip
            if wp.static(n_slots > 3):
                if warp_id == 4:
                    for k in range(slot3.world_offsets[world], slot3.world_offsets[world + 1]):
                        wp.static(natives[3][1])(
                            slot3.world_groups[k], slot3.group_to_art, tree.art_to_world,
                            tree.dense_contact_world_flag, force_refresh, tree.articulation_start, tree.joint_parent,
                            tree.joint_child, tree.joint_qd_start, tree.joint_parent_slot, tree.S_flat, tree.com_rel,
                            v_out, body_qd,
                        )  # fmt: skip
            if warp_id == wp.static(free_warp):
                _free_refresh(world, lane, free, v_out, body_qd)
            _block_sync()
            # colored contact sweep (with the pre-sweep snapshot the GEMV rebuilds from; the snapshot only
            # reads the twists the sweep stages)
            if warp_id == 0:
                if wp.static(has_cache):
                    _cache_snapshot(world, lane, cache, cache_max_bodies, colored.body_list, body_qd)
            # every warp: a color wider than one warp spreads over the block
            wp.static(colored_native)(
                world, world_count, colored.constraint_count, colored.body_a, colored.body_b, colored.MiJt_a,
                colored.MiJt_b, colored.J_a, colored.J_b, colored.eff_mass_inv, colored.rhs, colored.row_w,
                colored.row_type, colored.row_parent, colored.row_mu, colored.compliance_gamma,
                colored.torsion_first, colored.torsion_next, colored.torsion_row_radius, colored.row_order,
                colored.color_offsets, colored.body_list, colored.body_count, colored.body_local_slot,
                colored.unit_meta, colored.unit_friction_setup, colored.unit_payload, colored.ring_overflow, omega,
                colored_regularize, friction_start_iteration, g, colored.impulses, body_qd, body_impulses,
            )  # fmt: skip
            _block_sync()
            # push the deferred impulses into generalized velocities: GEMV, trees and free bodies own
            # disjoint DOFs, bodies and impulses, so they run side by side
            if wp.static(has_cache):
                if warp_id == 0:
                    wp.static(gemv_native)(
                        world, world_count, cache.world_flag, cache.body_count, colored.body_list,
                        cache.art_eligible, body_to_articulation, tree.articulation_start, tree.joint_qd_start,
                        cache.R, cache.B, cache.qd_base, body_impulses, body_qd, v_out,
                    )  # fmt: skip
            if wp.static(n_slots > 0):
                if warp_id == 1:
                    for k in range(slot0.world_offsets[world], slot0.world_offsets[world + 1]):
                        wp.static(natives[0][0])(
                            slot0.world_groups[k], slot0.group_to_art, tree.art_to_world, slot0.guard_flag,
                            tree.articulation_start, tree.joint_parent, tree.joint_child, tree.joint_qd_start,
                            tree.joint_parent_slot, tree.S_flat, tree.com_rel, tree.U, tree.D_inv, body_impulses,
                            body_qd, v_out,
                        )  # fmt: skip
            if wp.static(n_slots > 1):
                if warp_id == 2:
                    for k in range(slot1.world_offsets[world], slot1.world_offsets[world + 1]):
                        wp.static(natives[1][0])(
                            slot1.world_groups[k], slot1.group_to_art, tree.art_to_world, slot1.guard_flag,
                            tree.articulation_start, tree.joint_parent, tree.joint_child, tree.joint_qd_start,
                            tree.joint_parent_slot, tree.S_flat, tree.com_rel, tree.U, tree.D_inv, body_impulses,
                            body_qd, v_out,
                        )  # fmt: skip
            if wp.static(n_slots > 2):
                if warp_id == 3:
                    for k in range(slot2.world_offsets[world], slot2.world_offsets[world + 1]):
                        wp.static(natives[2][0])(
                            slot2.world_groups[k], slot2.group_to_art, tree.art_to_world, slot2.guard_flag,
                            tree.articulation_start, tree.joint_parent, tree.joint_child, tree.joint_qd_start,
                            tree.joint_parent_slot, tree.S_flat, tree.com_rel, tree.U, tree.D_inv, body_impulses,
                            body_qd, v_out,
                        )  # fmt: skip
            if wp.static(n_slots > 3):
                if warp_id == 4:
                    for k in range(slot3.world_offsets[world], slot3.world_offsets[world + 1]):
                        wp.static(natives[3][0])(
                            slot3.world_groups[k], slot3.group_to_art, tree.art_to_world, slot3.guard_flag,
                            tree.articulation_start, tree.joint_parent, tree.joint_child, tree.joint_qd_start,
                            tree.joint_parent_slot, tree.S_flat, tree.com_rel, tree.U, tree.D_inv, body_impulses,
                            body_qd, v_out,
                        )  # fmt: skip
            if warp_id == wp.static(free_warp):
                _free_flush(world, lane, free, body_qd, body_impulses, v_out)
            _block_sync()
            # row phase 5 (velocity limits); the next iteration's phase 3 follows on the same warp
            if wp.static(has_phase5):
                if warp_id == 0:
                    wp.static(row_native)(
                        world, rows.constraint_count, rows.dense_phase_bounds, rows.local_solve_owner,
                        rows.world_dof_indices, rows.rhs, rows.diag, rows.row_w, rows.impulses, rows.J_world,
                        rows.Y_world, rows.row_type, rows.row_parent, rows.row_mu, rows.torsion_group,
                        rows.torsion_row_radius, rows.drive_target_vel_bias, rows.drive_vel_multiplier,
                        rows.drive_impulse_multiplier, rows.drive_max_impulse, rows.drive_vel_limit,
                        rows.mf_constraint_count, rows.mf_contact_rows_end, rows.mf_meta, rows.mf_impulses,
                        rows.mf_J_a, rows.mf_J_b, rows.mf_MiJt_a, rows.mf_MiJt_b, rows.mf_row_mu, rows.mf_row_w,
                        rows.dense_gamma, rows.mf_gamma, 1, omega, row_regularize, 5, friction_start_iteration, g,
                        freeze_drive_rows, rows.momentum_scratch, v_out,
                    )  # fmt: skip
            # phase 5 and the next iteration's phases 3 and 4 share warp 0; the barrier after phase 4
            # orders them against the other warps

    # the device functions' keys identify the variant; hash them to keep the module name short
    parts = [colored_native, row_native, gemv_native, *(f for pair in tree_natives for f in pair)]
    digest = hashlib.sha1("|".join(getattr(f, "key", repr(f)) for f in parts).encode()).hexdigest()[:10]
    key = f"p{int(has_phase3)}{int(has_phase5)}_c{int(has_cache)}_t{n_slots}_{digest}"
    fused_template.__name__ = f"steel_fused_iteration_{key}"
    fused_template.__qualname__ = fused_template.__name__
    return wp.kernel(enable_backward=False, module="unique")(fused_template)
