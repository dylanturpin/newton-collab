# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Latency mode of the colored propagation solve: the PGS iteration loop as one launch.

Each iteration of ``SolverSteel._launch_propagation_solve`` launches about a dozen kernels: row phases 3
and 4, the tree and free-body twist refreshes, the cache snapshot, the colored sweep, the cached-response
GEMV, the tree impulse propagation, the free-body flush and row phase 5. With few worlds each launch holds
a handful of warps, so a substep is a long chain of small launches. Every pass reads and writes only its
own world, so one block per world runs the whole loop in a single launch.

The block's warps take fixed roles and call the device functions of the per-pass kernels (each factory
exposes ``kernel.native``):

* warp 0: row phases 3, 4 and 5, the cache snapshot and the GEMV;
* warp 1 + i: tree group ``i``, refresh and propagation of the world's articulations of that group, one
  after another;
* the last warp: the free-rigid twist refresh and flush;
* every warp: the colored sweep, where a color wider than one warp spreads over the block.

Block barriers order the stages as the launch chain does. The device functions read the data written in
the loop through plain loads after a barrier, so the results match the chain bit for bit.

The structs fixed at setup (cache, trees, free bodies) reach the kernel through one-element device arrays,
which keeps its parameters under 4 KB: some drivers crash capturing a launch with more into a CUDA graph.
"""

from __future__ import annotations

from typing import Any

import warp as wp

from .kernels import WARP_DETERMINISM_OFF

# Tree groups one fused kernel can carry, one warp each.
MAX_TREE_SLOTS = 4

# Fields in the order of ``SolverSteel._mf_gs_inputs``, which fills them.
ROW_FIELDS = (
    "constraint_count",
    "world_dof_indices",
    "rhs",
    "diag",
    "impulses",
    "J_world",
    "Y_world",
    "row_type",
    "row_parent",
    "row_mu",
    "drive_target_vel_bias",
    "drive_vel_multiplier",
    "drive_impulse_multiplier",
    "drive_max_impulse",
    "drive_vel_limit",
    "velocity_limit_slot",
    "velocity_limit_sign",
    "mf_constraint_count",
    "mf_contact_rows_end",
    "mf_meta",
    "mf_impulses",
    "mf_J_a",
    "mf_J_b",
    "mf_MiJt_a",
    "mf_MiJt_b",
    "mf_row_mu",
    "row_w",
    "mf_row_w",
    "torsion_group",
    "torsion_row_radius",
    "dense_gamma",
    "mf_gamma",
)

# Fields in the order of ``SolverSteel._colored_sweep_inputs``, which fills them.
COLORED_FIELDS = (
    "constraint_count",
    "body_a",
    "body_b",
    "MiJt_a",
    "MiJt_b",
    "J_a",
    "J_b",
    "eff_mass_inv",
    "rhs",
    "row_w",
    "row_type",
    "row_parent",
    "row_mu",
    "torsion_first",
    "torsion_next",
    "torsion_row_radius",
    "compliance_gamma",
    "row_order",
    "color_offsets",
    "body_list",
    "body_count",
    "body_local_slot",
    "unit_meta",
    "unit_friction_setup",
    "unit_payload",
    "ring_overflow",
)


@wp.func_native("""
#if defined(__CUDA_ARCH__)
__syncthreads();
#endif
""")
def _block_sync(): ...


@wp.struct
class RowArgs:
    constraint_count: wp.array[int]
    world_dof_indices: wp.array2d[int]
    rhs: wp.array2d[float]
    diag: wp.array2d[float]
    impulses: wp.array2d[float]
    J_world: wp.array3d[float]
    Y_world: wp.array3d[float]
    row_type: wp.array2d[int]
    row_parent: wp.array2d[int]
    row_mu: wp.array2d[float]
    drive_target_vel_bias: wp.array2d[float]
    drive_vel_multiplier: wp.array2d[float]
    drive_impulse_multiplier: wp.array2d[float]
    drive_max_impulse: wp.array2d[float]
    drive_vel_limit: wp.array2d[float]
    velocity_limit_slot: wp.array[int]
    velocity_limit_sign: wp.array[float]
    mf_constraint_count: wp.array[int]
    mf_contact_rows_end: wp.array[int]
    mf_meta: wp.array2d[int]
    mf_impulses: wp.array2d[float]
    mf_J_a: wp.array3d[float]
    mf_J_b: wp.array3d[float]
    mf_MiJt_a: wp.array3d[float]
    mf_MiJt_b: wp.array3d[float]
    mf_row_mu: wp.array2d[float]
    row_w: wp.array2d[float]
    mf_row_w: wp.array2d[float]
    torsion_group: wp.array2d[int]
    torsion_row_radius: wp.array2d[float]
    dense_gamma: wp.array2d[float]
    mf_gamma: wp.array2d[float]


@wp.struct
class ColoredArgs:
    constraint_count: wp.array[int]
    body_a: wp.array2d[int]
    body_b: wp.array2d[int]
    MiJt_a: wp.array3d[float]
    MiJt_b: wp.array3d[float]
    J_a: wp.array3d[float]
    J_b: wp.array3d[float]
    eff_mass_inv: wp.array2d[float]
    rhs: wp.array2d[float]
    row_w: wp.array2d[float]
    row_type: wp.array2d[int]
    row_parent: wp.array2d[int]
    row_mu: wp.array2d[float]
    torsion_first: wp.array2d[int]
    torsion_next: wp.array2d[int]
    torsion_row_radius: wp.array2d[float]
    compliance_gamma: wp.array2d[float]
    row_order: wp.array[int]
    color_offsets: wp.array[int]
    body_list: wp.array2d[int]
    body_count: wp.array[int]
    body_local_slot: wp.array[int]
    unit_meta: wp.array[int]
    unit_friction_setup: wp.array[float]
    unit_payload: wp.array[float]
    ring_overflow: wp.array[int]


@wp.struct
class CacheArgs:
    world_flag: wp.array[int]
    body_count: wp.array[int]
    art_eligible: wp.array[int]
    R: wp.array4d[float]
    B: wp.array4d[float]
    qd_base: wp.array3d[float]


@wp.struct
class TreeArgs:
    """Arrays every tree group's refresh and propagation read."""

    art_to_world: wp.array[int]
    dense_contact_world_flag: wp.array[int]
    articulation_start: wp.array[int]
    joint_parent: wp.array[int]
    joint_child: wp.array[int]
    joint_qd_start: wp.array[int]
    joint_parent_slot: wp.array[int]
    # Depth-level joint order, read by the level-walk refresh variants.
    level_joint: wp.array[int]
    level_offsets: wp.array[int]
    level_art_start: wp.array[int]
    S_flat: wp.array2d[float]
    com_rel: wp.array2d[float]
    U: wp.array2d[float]
    D_inv: wp.array3d[float]


@wp.struct
class TreeSlot:
    """One tree group: its articulations, its propagation guard and its articulations per world (CSR)."""

    group_to_art: wp.array[int]
    guard_flag: wp.array[int]
    world_offsets: wp.array[int]
    world_groups: wp.array[int]


@wp.struct
class FreeArgs:
    """Free-rigid bodies per world (CSR) and their first DOFs."""

    world_offsets: wp.array[int]
    bodies: wp.array[int]
    dof_start: wp.array[int]


@wp.func
def _free_refresh(world: int, lane: int, free: FreeArgs, v_out: wp.array[float], body_qd: wp.array2d[float]):
    """``refresh_propagation_free_body_qd_from_vout`` for one world's free-rigid bodies."""
    for i in range(free.world_offsets[world] + lane, free.world_offsets[world + 1], 32):
        body = free.bodies[i]
        dof = free.dof_start[i]
        for r in range(6):
            body_qd[body, r] = v_out[dof + r]


@wp.func
def _free_flush(
    world: int,
    lane: int,
    free: FreeArgs,
    body_qd: wp.array2d[float],
    body_impulses: wp.array2d[float],
    v_out: wp.array[float],
):
    """``flush_propagation_free_body_qd_to_vout`` for one world's free-rigid bodies."""
    for i in range(free.world_offsets[world] + lane, free.world_offsets[world + 1], 32):
        body = free.bodies[i]
        dof = free.dof_start[i]
        for r in range(6):
            v_out[dof + r] = body_qd[body, r]
            body_impulses[body, r] = 0.0


@wp.func
def _cache_snapshot(
    world: int,
    lane: int,
    cache: CacheArgs,
    cache_max_bodies: int,
    body_list: wp.array2d[int],
    body_qd: wp.array2d[float],
):
    """``snapshot_propagation_cache_qd_base`` for one world."""
    if cache.world_flag[world] != 0:
        n = wp.min(cache.body_count[world], cache_max_bodies)
        for slot in range(lane, n, 32):
            body = body_list[world, slot]
            if body >= 0:
                for r in range(6):
                    cache.qd_base[world, slot, r] = body_qd[body, r]


def warp_count(tree_slots: int) -> int:
    """Warps per block: the row warp, one per tree group and the free-body warp."""
    return 2 + tree_slots


def get_fused_iteration_kernel(
    colored_native: Any,
    row_native: Any,
    gemv_native: Any | None,
    tree_natives: tuple[tuple[Any, Any, bool], ...],
    has_phase3: bool,
    has_phase5: bool,
) -> wp.Kernel:
    """Build the fused loop kernel for one solver configuration.

    ``colored_native`` is the block-wide sweep built for the launch's block size, ``tree_natives`` holds
    ``(propagate, refresh, refresh_takes_levels)`` per tree group in the order of the ``slots`` argument (a
    level-walk refresh also reads ``TreeArgs``' level order), and ``gemv_native`` is
    ``None`` without cached responses. Launch with ``dim=[world_count]`` and ``32 * warp_count(...)`` threads.
    """
    n_slots = len(tree_natives)
    free_warp = 1 + n_slots

    @wp.func(module="unique")
    def row_phase(
        world: int,
        phase: int,
        rows: RowArgs,
        omega: float,
        regularize: int,
        freeze_drive_rows: int,
        friction_start_iteration: int,
        iteration: int,
        v_out: wp.array[float],
    ):
        wp.static(row_native)(
            world, rows.constraint_count, rows.world_dof_indices, rows.rhs, rows.diag, rows.impulses,
            rows.J_world, rows.Y_world, rows.row_type, rows.row_parent, rows.row_mu, rows.drive_target_vel_bias,
            rows.drive_vel_multiplier, rows.drive_impulse_multiplier, rows.drive_max_impulse, rows.drive_vel_limit,
            rows.velocity_limit_slot, rows.velocity_limit_sign, rows.mf_constraint_count, rows.mf_contact_rows_end, rows.mf_meta, rows.mf_impulses, rows.mf_J_a,
            rows.mf_J_b, rows.mf_MiJt_a, rows.mf_MiJt_b, rows.mf_row_mu, rows.row_w, rows.mf_row_w,
            rows.torsion_group, rows.torsion_row_radius, rows.dense_gamma, rows.mf_gamma, 1, omega, regularize,
            freeze_drive_rows, phase, friction_start_iteration, iteration, v_out,
        )  # fmt: skip

    def steel_fused_propagation_loop(
        world_count: int,
        iterations: int,
        iteration_offset: int,
        friction_start_iteration: int,
        omega: float,
        regularize: int,
        freeze_drive_rows: int,
        force_refresh: int,
        cache_max_bodies: int,
        colored: ColoredArgs,
        rows: RowArgs,
        cache: wp.array[CacheArgs],
        tree: wp.array[TreeArgs],
        slots: wp.array[TreeSlot],
        free: wp.array[FreeArgs],
        articulation_active: wp.array[int],
        body_to_articulation: wp.array[int],
        propagation_impulses: wp.array2d[float],
        body_qd: wp.array2d[float],
        body_impulses: wp.array2d[float],
        v_out: wp.array[float],
    ):
        world, t = wp.tid()
        warp_id = t // 32
        lane = t - warp_id * 32
        for i in range(iterations):
            g = iteration_offset + i
            # row phases 3 (internal rows) and 4 (dense and free-body contacts)
            if warp_id == 0:
                if wp.static(has_phase3):
                    row_phase(world, 3, rows, omega, regularize, freeze_drive_rows, friction_start_iteration, g, v_out)
                row_phase(world, 4, rows, omega, regularize, freeze_drive_rows, friction_start_iteration, g, v_out)
            _block_sync()
            # live twists from the generalized velocities
            for s in range(wp.static(n_slots)):
                if warp_id == 1 + s:
                    slot = slots[s]
                    for k in range(slot.world_offsets[world], slot.world_offsets[world + 1]):
                        if wp.static(tree_natives[s][2]):
                            wp.static(tree_natives[s][1])(
                                slot.world_groups[k], slot.group_to_art, articulation_active, tree[0].art_to_world,
                                tree[0].dense_contact_world_flag, force_refresh, tree[0].articulation_start,
                                tree[0].joint_parent, tree[0].joint_child, tree[0].joint_qd_start,
                                tree[0].joint_parent_slot, tree[0].level_joint, tree[0].level_offsets,
                                tree[0].level_art_start, tree[0].S_flat, tree[0].com_rel, v_out, body_qd,
                            )  # fmt: skip
                        else:
                            wp.static(tree_natives[s][1])(
                                slot.world_groups[k], slot.group_to_art, articulation_active, tree[0].art_to_world,
                                tree[0].dense_contact_world_flag, force_refresh, tree[0].articulation_start,
                                tree[0].joint_parent, tree[0].joint_child, tree[0].joint_qd_start,
                                tree[0].joint_parent_slot, tree[0].S_flat, tree[0].com_rel, v_out, body_qd,
                            )  # fmt: skip
            if warp_id == wp.static(free_warp):
                _free_refresh(world, lane, free[0], v_out, body_qd)
            _block_sync()
            # The snapshot overlaps the sweep's start: the sweep only stages the twists in shared memory before
            # a block barrier that warp 0 reaches after the snapshot.
            if wp.static(gemv_native is not None):
                if warp_id == 0:
                    _cache_snapshot(world, lane, cache[0], cache_max_bodies, colored.body_list, body_qd)
            wp.static(colored_native)(
                world, world_count, colored.constraint_count, colored.body_a, colored.body_b, colored.MiJt_a,
                colored.MiJt_b, colored.J_a, colored.J_b, colored.eff_mass_inv, colored.rhs, colored.row_w,
                colored.row_type, colored.row_parent, colored.row_mu, colored.torsion_first, colored.torsion_next,
                colored.torsion_row_radius, colored.compliance_gamma, colored.row_order, colored.color_offsets,
                colored.body_list, colored.body_count, colored.body_local_slot, colored.unit_meta,
                colored.unit_friction_setup, colored.unit_payload, colored.ring_overflow, omega, regularize,
                friction_start_iteration, g, propagation_impulses, body_qd, body_impulses,
            )  # fmt: skip
            _block_sync()
            # The GEMV, the trees and the free bodies own disjoint DOFs, bodies and impulses.
            if wp.static(gemv_native is not None):
                if warp_id == 0:
                    wp.static(gemv_native)(
                        world, world_count, cache[0].world_flag, cache[0].body_count, colored.body_list,
                        cache[0].art_eligible, body_to_articulation, tree[0].articulation_start,
                        tree[0].joint_qd_start, cache[0].R, cache[0].B, cache[0].qd_base, body_impulses, body_qd,
                        v_out,
                    )  # fmt: skip
            for s in range(wp.static(n_slots)):
                if warp_id == 1 + s:
                    slot = slots[s]
                    for k in range(slot.world_offsets[world], slot.world_offsets[world + 1]):
                        wp.static(tree_natives[s][0])(
                            slot.world_groups[k], slot.group_to_art, articulation_active, tree[0].art_to_world,
                            slot.guard_flag, tree[0].articulation_start, tree[0].joint_parent, tree[0].joint_child,
                            tree[0].joint_qd_start, tree[0].joint_parent_slot, tree[0].S_flat, tree[0].com_rel,
                            tree[0].U, tree[0].D_inv, body_impulses, body_qd, v_out,
                        )  # fmt: skip
            if warp_id == wp.static(free_warp):
                _free_flush(world, lane, free[0], body_qd, body_impulses, v_out)
            _block_sync()
            # row phase 5 (velocity limits); warp 0 runs it before the next iteration's phases 3 and 4
            if wp.static(has_phase5):
                if warp_id == 0:
                    row_phase(world, 5, rows, omega, regularize, freeze_drive_rows, friction_start_iteration, g, v_out)

    return wp.kernel(enable_backward=False, module="unique", module_options=WARP_DETERMINISM_OFF)(
        steel_fused_propagation_loop
    )
