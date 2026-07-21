# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Trajectory inverse-kinematics solver with a block-banded global solve.

Solves all frames of one or more joint-space trajectories jointly as a
single nonlinear least-squares problem. Per-frame objectives
(:class:`~newton.ik.IKObjectivePosition`, ...) contribute block-diagonal
Gauss-Newton terms and are reused unchanged from the per-frame IK module;
temporal objectives (:class:`~newton.ik.IKObjectiveSmoothness`, ...)
contribute the banded coupling between frames. The resulting normal
equations are block-banded and are solved either with a batched
block-tridiagonal Cholesky factorization (one CUDA block per trajectory,
sequential over frames inside the kernel) or with a preconditioned
conjugate-gradient iteration on a BSR matrix (parallel over frames).
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import Any, ClassVar

import numpy as np
import warp as wp
from warp.optim.linear import LinearOperator, aslinearoperator, cg
from warp.sparse import bsr_block_index, bsr_from_triplets

from ..model import Model
from .ik_common import IKJacobianType, compute_costs
from .ik_lm_optimizer import IKOptimizerLM
from .ik_objectives import IKObjective
from .ik_trajectory_objectives import IKObjectiveTemporal

# solver kernels are never differentiated
wp.set_module_options({"enable_backward": False})


class IKLinearSolver(str, Enum):
    """Linear-solver backends supported by :class:`~newton.ik.IKSolverTrajectory`."""

    DIRECT = "direct"
    """Batched block-tridiagonal Cholesky (block Thomas algorithm).

    Exact solve, sequential over frames inside each trajectory's CUDA
    block, parallel across trajectories. The factorization streams through
    global workspaces so at most three superblock tiles live in shared
    memory at once. Preferred for moderate horizon lengths and larger
    trajectory batches.
    """

    CG = "cg"
    """Block-Jacobi-preconditioned conjugate gradient on a BSR system.

    Inexact iterative solve, parallel over frames. Preferred for very long
    horizons with a small number of trajectories.
    """

    SPIKE = "spike"
    """SPIKE-style parallel-in-time direct solve (Schur-complement variant).

    The frame chain is split into partitions separated by single interface
    blocks. Interior partitions factorize in parallel (one CUDA block per
    partition per trajectory); the symmetric Schur complement on the
    interfaces forms a small block-tridiagonal system solved with the same
    sequential kernel as :attr:`DIRECT`; interiors then recover in parallel.
    Exact like :attr:`DIRECT`, but with ``O(T / partitions)`` sequential
    depth — preferred for long horizons with few trajectories when an exact
    solve is wanted. The classic SPIKE reduced system is nonsymmetric; the
    Schur-complement reduction used here preserves symmetric positive
    definiteness so every factorization stays a Cholesky.

    The factorization streams through global workspaces in several lean
    passes, so its largest kernel holds five superblock tiles in CUDA
    shared memory (``5 * (k * n_dofs)^2`` fp32); problems whose
    factorization kernels exceed the device's shared-memory limit raise
    :class:`~newton.ik.IKSharedMemoryError` at construction and should use
    :attr:`CG`.
    """


class IKSharedMemoryError(RuntimeError):
    """Specialized tile kernels exceed the CUDA device's shared-memory limit.

    Raised at construction by :class:`~newton.ik.IKSolverTrajectory` when the
    :attr:`IKLinearSolver.DIRECT` or :attr:`IKLinearSolver.SPIKE` backend is
    selected but a factorization kernel's dynamic shared-memory footprint —
    proportional to the squared superblock size ``(k * n_dofs)^2`` set by the
    objective stack's temporal stencil width — does not fit the device. Catch
    it to fall back to :attr:`IKLinearSolver.CG`.
    """


# Per-trajectory reductions run in two stages so long horizons do not
# serialize on one thread per trajectory (dominant at small batch sizes).
@wp.kernel
def _reduce_costs_partial(
    costs_rows: wp.array[wp.float32],  # (n_rows,)
    n_frames: int,
    chunk: int,
    # outputs
    partials: wp.array2d[wp.float32],  # (n_trajectories, n_chunks)
):
    p, c = wp.tid()
    start = c * chunk
    end = wp.min(start + chunk, n_frames)
    acc = float(0.0)
    for t in range(start, end):
        acc += costs_rows[p * n_frames + t]
    partials[p, c] = acc


@wp.kernel
def _reduce_partials(
    partials: wp.array2d[wp.float32],  # (n_trajectories, n_chunks)
    n_chunks: int,
    # outputs
    out: wp.array[wp.float32],  # (n_trajectories,)
):
    p = wp.tid()
    acc = float(0.0)
    for c in range(n_chunks):
        acc += partials[p, c]
    out[p] = acc


@wp.kernel
def _accept_reject_trajectory(
    cost_curr: wp.array[wp.float32],  # (n_trajectories,)
    cost_prop: wp.array[wp.float32],  # (n_trajectories,)
    pred_red: wp.array[wp.float32],  # (n_trajectories,)
    rho_min: float,
    # outputs
    accept: wp.array[wp.int32],
):
    p = wp.tid()
    # a non-positive predicted reduction (possible with an inexact CG solve)
    # means the quadratic model found no descent direction: always reject
    reduction = cost_curr[p] - cost_prop[p]
    ok = pred_red[p] > 0.0 and reduction >= rho_min * pred_red[p]
    accept[p] = wp.int32(1) if ok else wp.int32(0)


@wp.kernel
def _update_trajectory_rows(
    joint_q_proposed: wp.array2d[wp.float32],
    residuals_proposed: wp.array2d[wp.float32],
    accept: wp.array[wp.int32],  # (n_trajectories,)
    n_frames: int,
    n_coords: int,
    num_residuals: int,
    # outputs
    joint_q_current: wp.array2d[wp.float32],
    residuals_current: wp.array2d[wp.float32],
):
    row = wp.tid()
    p = row // n_frames
    if accept[p] == 1:
        for i in range(n_coords):
            joint_q_current[row, i] = joint_q_proposed[row, i]
        for i in range(num_residuals):
            residuals_current[row, i] = residuals_proposed[row, i]


@wp.kernel
def _update_trajectory_scalars(
    accept: wp.array[wp.int32],  # (n_trajectories,)
    costs_proposed: wp.array[wp.float32],  # (n_trajectories,)
    lambda_factor: float,
    lambda_min: float,
    lambda_max: float,
    # outputs
    lambda_traj: wp.array[wp.float32],
    costs_traj: wp.array[wp.float32],
):
    p = wp.tid()
    if accept[p] == 1:
        lambda_traj[p] = lambda_traj[p] / lambda_factor
        costs_traj[p] = costs_proposed[p]
    else:
        lambda_traj[p] = wp.clamp(lambda_traj[p] * lambda_factor, lambda_min, lambda_max)


@wp.kernel
def _accumulate_temporal_band(
    coeffs: wp.array4d[wp.float32],  # (n_rows, width + 1, n_coeff_rows, n_dofs)
    width: int,
    n_frames: int,
    n_coeff_rows: int,
    # outputs
    band: wp.array4d[wp.float32],  # (n_rows, band_count, n_dofs, n_dofs)
):
    row, a, b = wp.tid()
    t = row % n_frames

    # H(t, t + d) blocks, gathered from residual rows t - i:
    # H(t, t+d)[a, b] = sum_i sum_c dr[t-i, c]/du[t, a] * dr[t-i, c]/du[t+d, b]
    for d in range(width + 1):
        acc = float(0.0)
        for i in range(width - d + 1):
            if i <= t:
                rs = row - i
                for c in range(n_coeff_rows):
                    acc += coeffs[rs, i, c, a] * coeffs[rs, i + d, c, b]
        band[row, d, a, b] += acc


@wp.kernel
def _accumulate_temporal_grad(
    coeffs: wp.array4d[wp.float32],  # (n_rows, width + 1, n_coeff_rows, n_dofs)
    residuals: wp.array2d[wp.float32],  # (n_rows, n_residuals)
    start_idx: int,
    width: int,
    n_frames: int,
    n_coeff_rows: int,
    # outputs
    grad: wp.array2d[wp.float32],  # (n_rows, n_dofs)
):
    row, a = wp.tid()
    t = row % n_frames

    # gradient J^T r, gathered from residual rows t - j
    gacc = float(0.0)
    for j in range(width + 1):
        if j <= t:
            rs = row - j
            for c in range(n_coeff_rows):
                gacc += coeffs[rs, j, c, a] * residuals[rs, start_idx + c]
    grad[row, a] += gacc


@wp.kernel
def _gather_perframe_residuals(
    residuals: wp.array2d[wp.float32],  # (n_rows, n_residuals)
    # outputs
    residuals_3d: wp.array3d[wp.float32],  # (n_rows, n_perframe, 1)
):
    row, i = wp.tid()
    residuals_3d[row, i, 0] = residuals[row, i]


@wp.kernel
def _gather_block_diag(
    jtj: wp.array3d[wp.float32],  # (n_rows, n_dofs, n_dofs)
    band: wp.array4d[wp.float32],  # (n_rows, band_count, n_dofs, n_dofs)
    lambda_traj: wp.array[wp.float32],  # (n_trajectories,)
    fixed_mask: wp.array[wp.uint8],  # (n_frames,)
    n_frames: int,
    kb: int,
    n_dofs: int,
    band_count: int,
    # outputs
    d_bar: wp.array4d[wp.float32],  # (n_trajectories, n_super, m, m)
):
    p, g, a, b = wp.tid()
    fa = g * kb + a // n_dofs
    fb = g * kb + b // n_dofs
    da = a % n_dofs
    db = b % n_dofs

    val = float(0.0)
    if fa >= n_frames or fb >= n_frames:
        # identity padding for the partial trailing superblock
        val = 1.0 if a == b else 0.0
    elif fixed_mask[fa] != 0 or fixed_mask[fb] != 0:
        val = 1.0 if a == b else 0.0
    elif fa == fb:
        row = p * n_frames + fa
        val = jtj[row, da, db] + band[row, 0, da, db]
        if da == db:
            val += lambda_traj[p]
    else:
        # band stores H(f, f + d); read the transpose for fa > fb
        if fa < fb:
            d = fb - fa
            if d < band_count:
                val = band[p * n_frames + fa, d, da, db]
        else:
            d = fa - fb
            if d < band_count:
                val = band[p * n_frames + fb, d, db, da]

    d_bar[p, g, a, b] = val


@wp.kernel
def _gather_block_offdiag(
    band: wp.array4d[wp.float32],  # (n_rows, band_count, n_dofs, n_dofs)
    fixed_mask: wp.array[wp.uint8],  # (n_frames,)
    n_frames: int,
    kb: int,
    n_dofs: int,
    band_count: int,
    # outputs
    l_bar: wp.array4d[wp.float32],  # (n_trajectories, n_super, m, m) — block (g, g - 1)
):
    p, g, a, b = wp.tid()
    val = float(0.0)
    if g > 0:
        fa = g * kb + a // n_dofs
        fb = (g - 1) * kb + b // n_dofs
        da = a % n_dofs
        db = b % n_dofs
        if fa < n_frames and fixed_mask[fa] == 0 and fixed_mask[fb] == 0:
            d = fa - fb  # always > 0: read the transpose of H(fb, fb + d)
            if d < band_count:
                val = band[p * n_frames + fb, d, db, da]
    l_bar[p, g, a, b] = val


@wp.kernel
def _gather_rhs(
    grad: wp.array2d[wp.float32],  # (n_rows, n_dofs)
    fixed_mask: wp.array[wp.uint8],  # (n_frames,)
    n_frames: int,
    kb: int,
    n_dofs: int,
    # outputs
    b_bar: wp.array3d[wp.float32],  # (n_trajectories, n_super, m)
):
    p, g, a = wp.tid()
    fa = g * kb + a // n_dofs
    val = float(0.0)
    if fa < n_frames and fixed_mask[fa] == 0:
        val = -grad[p * n_frames + fa, a % n_dofs]
    b_bar[p, g, a] = val


@wp.kernel
def _scatter_delta(
    x_bar: wp.array3d[wp.float32],  # (n_trajectories, n_super, m)
    fixed_mask: wp.array[wp.uint8],  # (n_frames,)
    n_frames: int,
    kb: int,
    n_dofs: int,
    # outputs
    dq_dof: wp.array2d[wp.float32],  # (n_rows, n_dofs)
):
    row, dof = wp.tid()
    t = row % n_frames
    if fixed_mask[t] != 0:
        dq_dof[row, dof] = 0.0
        return
    p = row // n_frames
    g = t // kb
    a = (t % kb) * n_dofs + dof
    dq_dof[row, dof] = x_bar[p, g, a]


@wp.kernel
def _pred_reduction_partial(
    dq_dof: wp.array2d[wp.float32],  # (n_rows, n_dofs)
    grad: wp.array2d[wp.float32],  # (n_rows, n_dofs)
    lambda_traj: wp.array[wp.float32],  # (n_trajectories,)
    n_frames: int,
    n_dofs: int,
    chunk: int,
    # outputs
    partials: wp.array2d[wp.float32],  # (n_trajectories, n_chunks)
):
    p, c = wp.tid()
    lam = lambda_traj[p]
    start = c * chunk
    end = wp.min(start + chunk, n_frames)
    acc = float(0.0)
    for t in range(start, end):
        row = p * n_frames + t
        for d in range(n_dofs):
            dq = dq_dof[row, d]
            acc += dq * (lam * dq - grad[row, d])
    partials[p, c] = 0.5 * acc


@wp.kernel
def _fill_bsr_values(
    jtj: wp.array3d[wp.float32],  # (n_rows, n_dofs, n_dofs)
    band: wp.array4d[wp.float32],  # (n_rows, band_count, n_dofs, n_dofs)
    lambda_traj: wp.array[wp.float32],  # (n_trajectories,)
    fixed_mask: wp.array[wp.uint8],  # (n_frames,)
    n_frames: int,
    n_dofs: int,
    band_width: int,
    bsr_offsets: wp.array[wp.int32],
    bsr_columns: wp.array[wp.int32],
    # outputs
    values: wp.array3d[wp.float32],  # (nnz, n_dofs, n_dofs)
):
    row, s, a = wp.tid()
    t = row % n_frames
    p = row // n_frames
    tc = t + s - band_width
    if tc < 0 or tc >= n_frames:
        return
    col = p * n_frames + tc
    idx = bsr_block_index(row, col, bsr_offsets, bsr_columns)
    if idx < 0:
        return

    if s == band_width:  # diagonal block
        for b in range(n_dofs):
            val = float(0.0)
            if fixed_mask[t] != 0:
                val = 1.0 if a == b else 0.0
            else:
                val = jtj[row, a, b] + band[row, 0, a, b]
                if a == b:
                    val += lambda_traj[p]
            values[idx, a, b] = val
    else:
        d = s - band_width
        active = fixed_mask[t] == 0 and fixed_mask[tc] == 0
        for b in range(n_dofs):
            val = float(0.0)
            if active:
                # band stores H(f, f + d); read the transpose below the diagonal
                if d > 0:
                    val = band[row, d, a, b]
                else:
                    val = band[col, -d, b, a]
            values[idx, a, b] = val


@wp.kernel
def _gather_rhs_flat(
    grad: wp.array2d[wp.float32],  # (n_rows, n_dofs)
    fixed_mask: wp.array[wp.uint8],  # (n_frames,)
    n_frames: int,
    n_dofs: int,
    # outputs
    rhs: wp.array[wp.float32],  # (n_rows * n_dofs,)
):
    row, dof = wp.tid()
    val = float(0.0)
    if fixed_mask[row % n_frames] == 0:
        val = -grad[row, dof]
    rhs[row * n_dofs + dof] = val


@wp.kernel
def _mask_fixed_delta(
    fixed_mask: wp.array[wp.uint8],  # (n_frames,)
    n_frames: int,
    # outputs
    dq_dof: wp.array2d[wp.float32],
):
    row, dof = wp.tid()
    if fixed_mask[row % n_frames] != 0:
        dq_dof[row, dof] = 0.0


@wp.kernel
def _find_diag_block_index(
    bsr_offsets: wp.array[wp.int32],
    bsr_columns: wp.array[wp.int32],
    # outputs
    diag_idx: wp.array[wp.int32],  # (n_rows,)
):
    row = wp.tid()
    diag_idx[row] = bsr_block_index(row, row, bsr_offsets, bsr_columns)


@wp.kernel
def _block_jacobi_apply(
    minv: wp.array3d[wp.float32],  # (n_rows, n_dofs, n_dofs)
    x: wp.array[wp.float32],
    y: wp.array[wp.float32],
    alpha: wp.float32,
    beta: wp.float32,
    n_dofs: int,
    # outputs
    z: wp.array[wp.float32],
):
    row, di = wp.tid()
    acc = float(0.0)
    for j in range(n_dofs):
        acc += minv[row, di, j] * x[row * n_dofs + j]
    i = row * n_dofs + di
    z[i] = alpha * acc + beta * y[i]


_CG_DOT_TILE = 512
# length above which warp's single-batch dot path abandons the serial-lane
# reduction for its tiled tree; _SegmentedTiledDot applies the same policy to
# the batched path (warp's batched kernel stays serial at every length)
_CG_SERIAL_DOT_MAX_LENGTH = 128 * _CG_DOT_TILE


@wp.kernel
def _batch_dot_partials(
    a: wp.array2d[wp.float32],
    b: wp.array2d[wp.float32],
    batch_offsets: wp.array[wp.int32],
    blocks_per_batch: int,
    # outputs
    partials: wp.array2d[wp.float32],  # (n_columns, batch_count * blocks_per_batch)
):
    col, block, lane = wp.tid()
    batch_id = block // blocks_per_batch
    i = batch_offsets[batch_id] + (block - batch_id * blocks_per_batch) * wp.block_dim() + lane
    acc = float(0.0)
    if i < batch_offsets[batch_id + 1]:
        acc = a[col, i] * b[col, i]
    partial = wp.tile_sum(wp.tile(acc))
    wp.tile_store(partials[col], partial, offset=block)


@wp.kernel
def _batch_dot_combine(
    partials: wp.array2d[wp.float32],  # (n_columns, batch_count * blocks_per_batch)
    blocks_per_batch: int,
    # outputs
    result: wp.array2d[wp.float32],  # (n_columns, batch_count)
):
    col, batch_id, lane = wp.tid()
    acc = float(0.0)
    for j in range(batch_id * blocks_per_batch + lane, (batch_id + 1) * blocks_per_batch, wp.block_dim()):
        acc += partials[col, j]
    total = wp.tile_sum(wp.tile(acc))
    wp.tile_store(result[col], total, offset=batch_id)


class _SegmentedTiledDot:
    """Per-trajectory CG dot products with a segmented tile-tree reduction.

    Drop-in for warp CG's use of the ``compute``/``col``/``cols`` interface
    of its internal ``TiledDot``, restricted to flat fp32 scalar arrays (no
    vector-dtype handling) over a uniform per-subproblem block grid.

    warp's batched path (``batch_count > 1``) reduces each subproblem with a
    single block whose lanes accumulate their strided share serially in the
    payload dtype — ``subproblem_length / block_dim`` fp32 additions per
    lane. The O(n) rounding growth this injects into every CG scalar (rho,
    the step denominator ``p . Ap``, the stopping-test residual norms)
    stalls convergence of long trajectory chains at iteration budgets that
    are ample for the same trajectory solved alone (warp reduces single
    batches longer than ``_CG_SERIAL_DOT_MAX_LENGTH`` with a tiled tree).
    Here every 512-entry block reduces through ``wp.tile_sum`` and one
    further tile reduction per subproblem combines the block partials,
    keeping the accumulation depth logarithmic.

    Parity envelope: for subproblems up to ``_CG_DOT_TILE**2`` (512 * 512 =
    262,144) scalar dofs this reduction has the same shape as warp's
    single-batch bounded tree — block indexing is relative to each
    subproblem's own offset — so batched CG solves stay bitwise-equal to
    the equivalent single-trajectory solves. Above that, the combine stage
    folds partials serially per lane: accuracy stays tree-like, but bitwise
    parity with the single-batch path ends. The reduction order is fixed
    (no atomics), so results are deterministic run to run.

    Args:
        batch_offsets: Scalar-dof prefix offsets, shape ``[batch_count + 1]``,
            partitioning the flat dof vector into per-subproblem segments.
        batch_length: Maximum per-subproblem scalar length. The block grid
            is uniform across subproblems, so every segment must satisfy
            ``batch_offsets[i + 1] - batch_offsets[i] <= batch_length``.
        device: Device on which to allocate scratch memory and launch
            kernels.
        max_column_count: Maximum number of simultaneous dot products (CG
            computes two at once).
    """

    def __init__(
        self,
        batch_offsets: wp.array[wp.int32],
        batch_length: int,
        device,
        max_column_count: int = 2,
    ):
        self.batch_count = batch_offsets.shape[0] - 1
        self._batch_offsets = batch_offsets
        self._device = device
        self._blocks_per_batch = -(-batch_length // _CG_DOT_TILE)
        self._partials = wp.zeros(
            (max_column_count, self.batch_count * self._blocks_per_batch),
            dtype=wp.float32,
            device=device,
        )
        self._output = wp.zeros((max_column_count, self.batch_count), dtype=wp.float32, device=device)

    def compute(self, a: wp.array, b: wp.array, col_offset: int = 0) -> wp.array:
        if a.ndim == 1:
            a = a.reshape((1, -1))
        if b.ndim == 1:
            b = b.reshape((1, -1))
        column_count = a.shape[0]
        out = self._output[col_offset : col_offset + column_count]
        wp.launch(
            _batch_dot_partials,
            dim=(column_count, self.batch_count * self._blocks_per_batch, _CG_DOT_TILE),
            inputs=[a, b, self._batch_offsets, self._blocks_per_batch],
            outputs=[self._partials],
            block_dim=_CG_DOT_TILE,
            device=self._device,
        )
        wp.launch(
            _batch_dot_combine,
            dim=(column_count, self.batch_count, _CG_DOT_TILE),
            inputs=[self._partials, self._blocks_per_batch],
            outputs=[out],
            block_dim=_CG_DOT_TILE,
            device=self._device,
        )
        return out

    def col(self, col: int = 0) -> wp.array:
        return self._output[col][: self.batch_count]

    def cols(self, count: int, start: int = 0) -> wp.array:
        return self._output[start : start + count, : self.batch_count]


def _swap_cg_tiled_dot(cg_state, batch_offsets: wp.array[wp.int32], batch_length: int, device) -> None:
    """Replace a warp CG state's dot reduction with :class:`_SegmentedTiledDot`.

    warp exposes no public hook for the CG dot reduction, so the swap writes
    the private ``_tiled_dot`` attribute. The attribute is validated first:
    a plain assignment would silently create a dead attribute if warp renamed
    it, reverting the fix while tests stay green. The check is duck-typed
    because warp does not export ``TiledDot`` from ``warp.optim.linear``.

    Args:
        cg_state: warp CG solver state created with ``run=False``.
        batch_offsets: Scalar-dof prefix offsets, shape ``[batch_count + 1]``.
        batch_length: Maximum per-subproblem scalar length.
        device: Device on which to allocate scratch memory and launch
            kernels.

    Raises:
        RuntimeError: If the CG state does not carry a ``_tiled_dot`` with
            the expected ``_CG_DOT_TILE``-lane tile size.
    """
    if not hasattr(cg_state, "_tiled_dot") or getattr(cg_state._tiled_dot, "tile_size", None) != _CG_DOT_TILE:
        raise RuntimeError(
            f"warp's CG solver state does not expose a _tiled_dot attribute with tile_size {_CG_DOT_TILE}; "
            "its internals have changed and the segmented batched-dot replacement (_SegmentedTiledDot) "
            "can no longer be installed safely. Re-validate the replacement against this warp version, "
            "or remove it if warp's batched dot reduction no longer accumulates serially."
        )
    cg_state._tiled_dot = _SegmentedTiledDot(batch_offsets, batch_length, device)


@wp.kernel
def _spike_recover(
    y_int: wp.array3d[wp.float32],  # (n_traj * n_parts, l_max, m)
    u_int: wp.array4d[wp.float32],  # (n_traj * n_parts, l_max, m, m)
    v_int: wp.array4d[wp.float32],  # (n_traj * n_parts, l_max, m, m)
    x_sep: wp.array3d[wp.float32],  # (n_traj, n_parts - 1, m)
    g_kind: wp.array[wp.int32],  # (n_super,) 0 = interior, 1 = separator
    g_idx: wp.array[wp.int32],  # (n_super,) partition / separator index
    g_loc: wp.array[wp.int32],  # (n_super,) local offset within the partition
    n_parts: int,
    m: int,
    # outputs
    x_bar: wp.array3d[wp.float32],  # (n_traj, n_super, m)
):
    p, g, a = wp.tid()
    if g_kind[g] == 1:
        x_bar[p, g, a] = x_sep[p, g_idx[g], a]
        return
    part = g_idx[g]
    loc = g_loc[g]
    row = p * n_parts + part
    acc = y_int[row, loc, a]
    # x_interior = y - U x_sep_left - V x_sep_right
    if part > 0:
        for c in range(m):
            acc -= u_int[row, loc, a, c] * x_sep[p, part - 1, c]
    if part < n_parts - 1:
        for c in range(m):
            acc -= v_int[row, loc, a, c] * x_sep[p, part, c]
    x_bar[p, g, a] = acc


@wp.kernel
def _refine_residual_f64(
    d_bar: wp.array4d[wp.float32],  # (n_traj, n_super, m, m)
    l_bar: wp.array4d[wp.float32],  # (n_traj, n_super, m, m), block (g, g - 1)
    b_bar: wp.array3d[wp.float32],  # (n_traj, n_super, m)
    x_bar: wp.array3d[wp.float32],  # (n_traj, n_super, m)
    n_super: int,
    m: int,
    # outputs
    r_bar: wp.array3d[wp.float32],  # (n_traj, n_super, m)
):
    """Banded residual r = b - A x accumulated in float64 (iterative refinement)."""
    p, g, a = wp.tid()
    acc = wp.float64(b_bar[p, g, a])
    for c in range(m):
        acc -= wp.float64(d_bar[p, g, a, c]) * wp.float64(x_bar[p, g, c])
    if g > 0:
        for c in range(m):
            acc -= wp.float64(l_bar[p, g, a, c]) * wp.float64(x_bar[p, g - 1, c])
    if g < n_super - 1:
        for c in range(m):
            acc -= wp.float64(l_bar[p, g + 1, c, a]) * wp.float64(x_bar[p, g + 1, c])
    r_bar[p, g, a] = wp.float32(acc)


@wp.kernel
def _add_refine_delta(
    dx_bar: wp.array3d[wp.float32],  # (n_traj, n_super, m)
    # outputs
    x_bar: wp.array3d[wp.float32],  # (n_traj, n_super, m)
):
    p, g, a = wp.tid()
    x_bar[p, g, a] += dx_bar[p, g, a]


class IKSolverTrajectory(IKOptimizerLM):
    """Levenberg-Marquardt trajectory IK with a block-banded global solve.

    The solver optimizes ``n_problems`` trajectories of ``n_frames`` frames
    each. Evaluation rows are laid out frame-major: row ``p * n_frames + t``
    holds frame ``t`` of trajectory ``p``. Per-frame objectives size their
    target arrays by ``n_problems * n_frames`` (one target per frame);
    temporal objectives couple consecutive frames and define the bandwidth
    of the Gauss-Newton system.

    Damping and step acceptance are per trajectory: a trajectory accepts or
    rejects the joint update of all of its frames atomically, based on the
    total cost across frames.

    Args:
        model: Shared articulation model.
        n_frames: Number of frames per trajectory.
        objectives: Ordered IK objectives; per-frame and temporal objectives
            may be mixed freely.
        n_problems: Number of trajectories optimized together.
        jacobian_mode: Jacobian backend for the per-frame objectives.
            Temporal objectives always evaluate analytically.
        linear_solver: Backend used to solve the block-banded normal
            equations.
        fixed_frames: Frame indices whose configurations are held fixed at
            their seed values (in every trajectory), e.g. ``[0]`` to anchor
            the start of the trajectory.
        lambda_initial: Initial LM damping factor for each trajectory.
        lambda_factor: LM damping update factor.
        lambda_min: Minimum LM damping value.
        lambda_max: Maximum LM damping value.
        rho_min: Minimum LM acceptance ratio.
        cg_iterations: Maximum conjugate-gradient iterations per LM step
            (CG backend only).
        cg_tol: Relative residual tolerance of the conjugate-gradient solve
            (CG backend only).
        spike_partitions: Number of parallel partitions of the frame chain
            (SPIKE backend only). ``None`` picks roughly one partition per
            16 superblocks, clamped to a valid range.
        refine_iterations: Number of float64-residual iterative-refinement
            passes applied after each linear solve, reusing the stored
            factors (DIRECT and SPIKE backends only; forced to ``0`` for
            CG, which does not factorize). Typically one pass drives the
            fp32 solution of an IK-shaped system to its correctly rounded
            value (near-flat spectra can need two), and the refined fixed
            point is independent of the backend choice (bitwise); ``0``
            keeps the raw fp32 factorization result.
    """

    TILE_M_SUPER = None
    _cache: ClassVar[dict[tuple[int, int, int, str], type]] = {}

    def __new__(
        cls,
        model: Model,
        n_frames: int,
        objectives: Sequence[IKObjective],
        n_problems: int = 1,
        *a: Any,
        **kw: Any,
    ) -> IKSolverTrajectory:
        n_dofs = model.joint_dof_count
        n_residuals = sum(o.residual_dim() for o in objectives)
        n_perframe = sum(o.residual_dim() for o in objectives if not isinstance(o, IKObjectiveTemporal))
        band_width = max((o.stencil_width() for o in objectives if isinstance(o, IKObjectiveTemporal)), default=0)
        kb = max(band_width, 1)
        arch = model.device.arch
        key = (n_dofs, n_residuals, n_perframe, kb, arch)

        spec_cls = cls._cache.get(key)
        if spec_cls is None:
            spec_cls = cls._build_specialized(key)
            cls._cache[key] = spec_cls

        return object.__new__(spec_cls)

    def __init__(
        self,
        model: Model,
        n_frames: int,
        objectives: Sequence[IKObjective],
        n_problems: int = 1,
        *,
        jacobian_mode: IKJacobianType | str = IKJacobianType.AUTODIFF,
        linear_solver: IKLinearSolver | str = IKLinearSolver.DIRECT,
        fixed_frames: Sequence[int] | None = None,
        lambda_initial: float = 0.1,
        lambda_factor: float = 2.0,
        lambda_min: float = 1e-5,
        lambda_max: float = 1e10,
        rho_min: float = 1e-3,
        cg_iterations: int = 64,
        cg_tol: float = 1e-6,
        spike_partitions: int | None = None,
        refine_iterations: int = 0,
    ) -> None:
        if isinstance(jacobian_mode, str):
            jacobian_mode = IKJacobianType(jacobian_mode)
        if isinstance(linear_solver, str):
            linear_solver = IKLinearSolver(linear_solver)
        if n_frames < 2:
            raise ValueError("n_frames must be >= 2")
        if n_problems < 1:
            raise ValueError("n_problems must be >= 1")
        if refine_iterations < 0:
            raise ValueError("refine_iterations must be >= 0")

        self.n_frames = n_frames
        self.n_trajectories = n_problems
        self.linear_solver = linear_solver
        self.cg_iterations = cg_iterations
        self.cg_tol = cg_tol
        self.refine_iterations = refine_iterations if linear_solver is not IKLinearSolver.CG else 0

        self.temporal_objectives = [o for o in objectives if isinstance(o, IKObjectiveTemporal)]
        self._temporal_uses_fk = any(o.uses_fk for o in self.temporal_objectives)
        self.band_width = max((o.stencil_width() for o in self.temporal_objectives), default=0)
        self.kb = max(self.band_width, 1)
        self.n_superblocks = (n_frames + self.kb - 1) // self.kb
        # temporal objectives report Jacobians through their stencil
        # coefficients, so the dense per-frame Jacobian and its J^T J tile
        # kernel only carry the per-frame objectives' rows
        self.n_perframe_residuals = sum(o.residual_dim() for o in objectives if not isinstance(o, IKObjectiveTemporal))

        if self.linear_solver is IKLinearSolver.SPIKE:
            self._plan_spike_partitions(spike_partitions)

        mask = np.zeros(n_frames, dtype=np.uint8)
        if fixed_frames is not None:
            for f in fixed_frames:
                if not 0 <= f < n_frames:
                    raise ValueError(f"fixed frame index {f} out of range [0, {n_frames})")
                mask[f] = 1
        self._fixed_mask_np = mask

        super().__init__(
            model,
            n_problems * n_frames,
            objectives,
            lambda_initial=lambda_initial,
            jacobian_mode=jacobian_mode,
            lambda_factor=lambda_factor,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            rho_min=rho_min,
        )

    # ------------------------------------------------------------------
    # setup
    # ------------------------------------------------------------------

    def _plan_spike_partitions(self, spike_partitions: int | None) -> None:
        """Split the superblock chain into interiors separated by interface blocks."""
        n_super = self.n_superblocks
        if n_super < 3:
            raise ValueError("the spike backend requires at least 3 superblocks; use the direct backend")
        n_parts = spike_partitions if spike_partitions is not None else max(2, min(64, n_super // 16))
        n_parts = max(2, min(n_parts, (n_super + 1) // 2))

        base, rem = divmod(n_super - (n_parts - 1), n_parts)
        lens = [base + (1 if i < rem else 0) for i in range(n_parts)]
        starts = []
        kind = np.zeros(n_super, dtype=np.int32)  # 0 = interior, 1 = separator
        idx = np.zeros(n_super, dtype=np.int32)
        loc = np.zeros(n_super, dtype=np.int32)
        g = 0
        for i, length in enumerate(lens):
            starts.append(g)
            for t in range(length):
                kind[g], idx[g], loc[g] = 0, i, t
                g += 1
            if i < n_parts - 1:
                kind[g], idx[g], loc[g] = 1, i, 0
                g += 1

        self._spike_n_parts = n_parts
        self._spike_l_max = max(lens)
        self._spike_istart_np = np.array(starts, dtype=np.int32)
        self._spike_ilen_np = np.array(lens, dtype=np.int32)
        self._spike_kind_np, self._spike_idx_np, self._spike_loc_np = kind, idx, loc

    def _init_objectives(self) -> None:
        for obj in self.temporal_objectives:
            obj.set_trajectory_layout(self.n_frames, self.n_trajectories)
        super()._init_objectives()

    def _build_residual_offsets(self) -> None:
        # per-frame objectives first so their rows form the leading
        # contiguous block consumed by the J^T J tile kernel; temporal rows
        # follow (their Jacobians live in the banded coefficients instead)
        offsets: list[int] = []
        offset = 0
        for obj in self.objectives:
            if not isinstance(obj, IKObjectiveTemporal):
                offsets.append(offset)
                offset += obj.residual_dim()
            else:
                offsets.append(-1)
        for i, obj in enumerate(self.objectives):
            if isinstance(obj, IKObjectiveTemporal):
                offsets[i] = offset
                offset += obj.residual_dim()
        self.residual_offsets = offsets

    def _jacobian_row_count(self) -> int:
        # at least one row so the buffers stay valid for temporal-only stacks
        return max(1, self.n_perframe_residuals)

    def _alloc_solver_buffers(self, grad: bool) -> None:
        super()._alloc_solver_buffers(grad)

        device = self.device
        n_rows = self.n_batch
        n_dofs = self.n_dofs
        n_traj = self.n_trajectories
        n_super = self.n_superblocks
        m = self.kb * n_dofs

        # pure-AUTODIFF solves compute no per-frame motion subspace; FK-using
        # temporal objectives then share this buffer instead (cf. _step)
        self._temporal_S_s = (
            wp.zeros((n_rows, n_dofs), dtype=wp.spatial_vector, device=device)
            if self._temporal_uses_fk and self.jacobian_mode == IKJacobianType.AUTODIFF
            else None
        )

        self.jtj = wp.zeros((n_rows, n_dofs, n_dofs), dtype=wp.float32, device=device)
        self.grad3 = wp.zeros((n_rows, n_dofs, 1), dtype=wp.float32, device=device)
        self.grad = self.grad3.reshape((n_rows, n_dofs))
        self.band = wp.zeros((n_rows, self.band_width + 1, n_dofs, n_dofs), dtype=wp.float32, device=device)
        self.fixed_mask = wp.array(self._fixed_mask_np, dtype=wp.uint8, device=device)

        # two-stage per-trajectory reduction: chunk count capped so both
        # stages stay parallel at million-frame horizons
        self._red_chunk = max(16, -(-self.n_frames // 4096))
        self._red_n_chunks = -(-self.n_frames // self._red_chunk)
        self._traj_partials = wp.zeros((n_traj, self._red_n_chunks), dtype=wp.float32, device=device)

        self.costs_traj = wp.zeros(n_traj, dtype=wp.float32, device=device)
        self.costs_traj_proposed = wp.zeros(n_traj, dtype=wp.float32, device=device)
        self.lambda_traj = wp.zeros(n_traj, dtype=wp.float32, device=device)
        self.accept_traj = wp.zeros(n_traj, dtype=wp.int32, device=device)
        self.pred_reduction_traj = wp.zeros(n_traj, dtype=wp.float32, device=device)

        if self.linear_solver in (IKLinearSolver.DIRECT, IKLinearSolver.SPIKE):
            self.d_bar = wp.zeros((n_traj, n_super, m, m), dtype=wp.float32, device=device)
            self.l_bar = wp.zeros((n_traj, n_super, m, m), dtype=wp.float32, device=device)
            self.b_bar = wp.zeros((n_traj, n_super, m), dtype=wp.float32, device=device)
            self.x_bar = wp.zeros((n_traj, n_super, m), dtype=wp.float32, device=device)
            if self.refine_iterations > 0:
                self._ref_r_bar = wp.zeros((n_traj, n_super, m), dtype=wp.float32, device=device)
                self._ref_dx_bar = wp.zeros((n_traj, n_super, m), dtype=wp.float32, device=device)
        if self.linear_solver is IKLinearSolver.DIRECT:
            self._chol_ws = wp.zeros((n_traj, n_super, m, m), dtype=wp.float32, device=device)
            self._coupling_ws = wp.zeros((n_traj, n_super, m, m), dtype=wp.float32, device=device)
            # the factorization/substitution kernels walk one partition per
            # block; the whole chain is a single partition here
            self._thomas_istart = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device=device)
            self._thomas_ilen = wp.array(np.array([n_super], dtype=np.int32), dtype=wp.int32, device=device)
        elif self.linear_solver is IKLinearSolver.SPIKE:
            self._alloc_spike_buffers()
        else:
            self._alloc_cg_buffers()
        self._check_tile_shared_memory()

    def _alloc_spike_buffers(self) -> None:
        device = self.device
        n_traj = self.n_trajectories
        m = self.kb * self.n_dofs
        n_parts = self._spike_n_parts
        l_max = self._spike_l_max
        n_sep = n_parts - 1
        rows = n_traj * n_parts

        self._sp_istart = wp.array(self._spike_istart_np, dtype=wp.int32, device=device)
        self._sp_ilen = wp.array(self._spike_ilen_np, dtype=wp.int32, device=device)
        self._sp_kind = wp.array(self._spike_kind_np, dtype=wp.int32, device=device)
        self._sp_idx = wp.array(self._spike_idx_np, dtype=wp.int32, device=device)
        self._sp_loc = wp.array(self._spike_loc_np, dtype=wp.int32, device=device)

        # per-interior local solution and the two spike columns
        self._sp_y = wp.zeros((rows, l_max, m), dtype=wp.float32, device=device)
        self._sp_u = wp.zeros((rows, l_max, m, m), dtype=wp.float32, device=device)
        self._sp_v = wp.zeros((rows, l_max, m, m), dtype=wp.float32, device=device)
        self._sp_chol = wp.zeros((rows, l_max, m, m), dtype=wp.float32, device=device)
        self._sp_coup = wp.zeros((rows, l_max, m, m), dtype=wp.float32, device=device)

        # reduced (Schur) block-tridiagonal system on the separators
        self._sp_d_sep = wp.zeros((n_traj, n_sep, m, m), dtype=wp.float32, device=device)
        self._sp_l_sep = wp.zeros((n_traj, n_sep, m, m), dtype=wp.float32, device=device)
        self._sp_b_sep = wp.zeros((n_traj, n_sep, m), dtype=wp.float32, device=device)
        self._sp_x_sep = wp.zeros((n_traj, n_sep, m), dtype=wp.float32, device=device)
        self._sp_chol_sep = wp.zeros((n_traj, n_sep, m, m), dtype=wp.float32, device=device)
        self._sp_coup_sep = wp.zeros((n_traj, n_sep, m, m), dtype=wp.float32, device=device)
        self._sp_sep_istart = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device=device)
        self._sp_sep_ilen = wp.array(np.array([n_sep], dtype=np.int32), dtype=wp.int32, device=device)

    def _alloc_cg_buffers(self) -> None:
        device = self.device
        n_rows = self.n_batch
        n_dofs = self.n_dofs
        n_frames = self.n_frames
        k = self.band_width

        # fixed block-banded topology, built once on the host
        rows = []
        cols = []
        for p in range(self.n_trajectories):
            for t in range(n_frames):
                row = p * n_frames + t
                for tc in range(max(0, t - k), min(n_frames, t + k + 1)):
                    rows.append(row)
                    cols.append(p * n_frames + tc)
        nnz = len(rows)
        values = wp.zeros((nnz, n_dofs, n_dofs), dtype=wp.float32, device=device)
        self._hessian = bsr_from_triplets(
            n_rows,
            n_rows,
            wp.array(np.array(rows, dtype=np.int32), dtype=wp.int32, device=device),
            wp.array(np.array(cols, dtype=np.int32), dtype=wp.int32, device=device),
            values,
            prune_numerical_zeros=False,
        )

        self._diag_block_idx = wp.zeros(n_rows, dtype=wp.int32, device=device)
        wp.launch(
            _find_diag_block_index,
            dim=n_rows,
            inputs=[self._hessian.offsets, self._hessian.columns],
            outputs=[self._diag_block_idx],
            device=device,
        )

        self._minv = wp.zeros((n_rows, n_dofs, n_dofs), dtype=wp.float32, device=device)
        self._identity = wp.array(np.eye(n_dofs, dtype=np.float32), dtype=wp.float32, device=device)
        self._cg_rhs = wp.zeros(n_rows * n_dofs, dtype=wp.float32, device=device)
        self._dq_flat = self.dq_dof.reshape((n_rows * n_dofs,))

        batch_offsets_np = np.arange(self.n_trajectories + 1, dtype=np.int32) * self.n_frames * n_dofs
        self._batch_offsets = wp.array(batch_offsets_np, dtype=wp.int32, device=device)

        def _matvec(x, y, z, alpha, beta):
            wp.launch(
                _block_jacobi_apply,
                dim=[n_rows, n_dofs],
                inputs=[self._minv, x, y, wp.float32(alpha), wp.float32(beta), n_dofs],
                outputs=[z],
                device=device,
            )

        precond = LinearOperator(
            shape=self._hessian.shape,
            dtype=self._hessian.scalar_type,
            device=device,
            matvec=_matvec,
        )
        # check_every=0 keeps the whole solve on-device (CUDA-graph friendly).
        # ScopedDevice works around warp <= 1.15 binding its internal reduction
        # launches to the default device instead of the operand device (fixed
        # on warp main).
        with wp.ScopedDevice(self.device):
            self._cg_state = cg(
                A=aslinearoperator(self._hessian, batch_offsets=self._batch_offsets),
                b=self._cg_rhs,
                x=self._dq_flat,
                M=precond,
                maxiter=self.cg_iterations,
                tol=self.cg_tol,
                check_every=0,
                run=False,
            )

        # warp's batched dot kernel accumulates each subproblem serially per
        # lane at every length, which under-converges CG on long trajectory
        # chains (see _SegmentedTiledDot); swap in the segmented tree above
        # the same threshold at which warp's single-batch path switches to
        # its tree, so shorter multi-trajectory solves keep warp's reduction
        # (and stay bitwise-equal to their single-trajectory counterparts).
        # CPU is deliberately left on warp's reduction: its batched dot is a
        # single serial lane per subproblem — the same O(n) issue in
        # principle — but unmeasured there and outside this fix's scope.
        # TODO: drop once warp's TiledDot batches with a multi-block tree.
        if self.n_trajectories > 1 and device.is_cuda and n_frames * n_dofs > _CG_SERIAL_DOT_MAX_LENGTH:
            _swap_cg_tiled_dot(self._cg_state, self._batch_offsets, n_frames * n_dofs, device)

    # ------------------------------------------------------------------
    # solve
    # ------------------------------------------------------------------

    def _reduce_costs_traj(self, costs_rows: wp.array[wp.float32], out: wp.array[wp.float32]) -> None:
        wp.launch(
            _reduce_costs_partial,
            dim=[self.n_trajectories, self._red_n_chunks],
            inputs=[costs_rows, self.n_frames, self._red_chunk],
            outputs=[self._traj_partials],
            device=self.device,
        )
        wp.launch(
            _reduce_partials,
            dim=self.n_trajectories,
            inputs=[self._traj_partials, self._red_n_chunks],
            outputs=[out],
            device=self.device,
        )

    def step(
        self,
        joint_q_in: wp.array2d[wp.float32],
        joint_q_out: wp.array2d[wp.float32],
        iterations: int = 10,
        step_size: float = 1.0,
    ) -> None:
        """Run several LM iterations on a batch of joint trajectories.

        Args:
            joint_q_in: Input joint coordinates [m or rad], shape
                [n_problems * n_frames, joint_coord_count], frame-major
                within each trajectory.
            joint_q_out: Output buffer for the optimized coordinates, same
                shape as ``joint_q_in``. It may alias ``joint_q_in``.
            iterations: Number of LM iterations to execute.
            step_size: Scalar applied to each computed update before
                integration.
        """
        if joint_q_in.shape != (self.n_batch, self.n_coords):
            raise ValueError("joint_q_in has incompatible shape")
        if joint_q_out.shape != (self.n_batch, self.n_coords):
            raise ValueError("joint_q_out has incompatible shape")

        if joint_q_in.ptr != joint_q_out.ptr:
            wp.copy(joint_q_out, joint_q_in)

        self.lambda_traj.fill_(self.lambda_initial)
        for i in range(iterations):
            self._step(joint_q_out, step_size=step_size, iteration=i)

    def _step(
        self,
        joint_q: wp.array2d[wp.float32],
        step_size: float = 1.0,
        iteration: int = 0,
    ) -> None:
        """Execute one trajectory-LM iteration with per-trajectory damping."""

        ctx_curr = self._ctx_solver(joint_q)

        # AUTODIFF/MIXED refresh FK inside the Jacobian tape; pure ANALYTIC
        # must re-evaluate here so a rejected proposal's FK left in body_q
        # does not corrupt the next linearization
        if self.jacobian_mode in (IKJacobianType.AUTODIFF, IKJacobianType.MIXED):
            if iteration == 0:
                self._residuals_autodiff(ctx_curr)
        else:
            self._residuals_analytic(ctx_curr)

        wp.launch(
            compute_costs,
            dim=self.n_batch,
            inputs=[ctx_curr.residuals, self.n_residuals],
            outputs=[self.costs],
            device=self.device,
        )
        self._reduce_costs_traj(self.costs, self.costs_traj)

        # dense per-frame Jacobian of the per-frame objectives (also refreshes
        # FK state for the AUTODIFF/MIXED modes)
        self._jacobian_at(ctx_curr)

        # block-diagonal J^T J and gradient of the per-frame objectives,
        # whose residual rows lead the buffer (cf. _build_residual_offsets)
        if self.n_perframe_residuals > 0:
            wp.launch(
                _gather_perframe_residuals,
                dim=[self.n_batch, self.n_perframe_residuals],
                inputs=[ctx_curr.residuals],
                outputs=[self.residuals_3d],
                device=self.device,
            )
            self._jtj_grad_tiled(ctx_curr.jacobian_out, self.residuals_3d, self.jtj, self.grad3)
        else:
            self.jtj.zero_()
            self.grad3.zero_()

        # banded coupling and gradient of the temporal objectives; body_q is
        # fresh at joint_q here in every mode (_residuals_analytic above, or
        # the Jacobian tape's FK), so FK-using objectives share it along with
        # the motion-subspace rows instead of re-evaluating their own
        shared_S_s = None
        if self._temporal_uses_fk:
            if self.joint_S_s is not None:
                # recomputed from body_q by _jacobian_analytic this iteration
                shared_S_s = self.joint_S_s
            else:
                self._compute_motion_subspace(
                    joint_q_in=joint_q,
                    body_q=self.body_q,
                    joint_S_s_out=self._temporal_S_s,
                )
                shared_S_s = self._temporal_S_s
        self.band.zero_()
        for obj, offset in zip(self.objectives, self.residual_offsets, strict=False):
            if isinstance(obj, IKObjectiveTemporal):
                obj.compute_coeffs(joint_q, body_q=self.body_q, joint_S_s=shared_S_s)
                n_coeff_rows = obj.coeffs.shape[2]
                wp.launch(
                    _accumulate_temporal_band,
                    dim=[self.n_batch, self.n_dofs, self.n_dofs],
                    inputs=[obj.coeffs, obj.stencil_width(), self.n_frames, n_coeff_rows],
                    outputs=[self.band],
                    device=self.device,
                )
                wp.launch(
                    _accumulate_temporal_grad,
                    dim=[self.n_batch, self.n_dofs],
                    inputs=[
                        obj.coeffs,
                        ctx_curr.residuals,
                        offset,
                        obj.stencil_width(),
                        self.n_frames,
                        n_coeff_rows,
                    ],
                    outputs=[self.grad],
                    device=self.device,
                )

        if self.linear_solver is IKLinearSolver.DIRECT:
            self._solve_direct()
        elif self.linear_solver is IKLinearSolver.SPIKE:
            self._solve_spike()
        else:
            self._solve_cg()

        wp.launch(
            _pred_reduction_partial,
            dim=[self.n_trajectories, self._red_n_chunks],
            inputs=[self.dq_dof, self.grad, self.lambda_traj, self.n_frames, self.n_dofs, self._red_chunk],
            outputs=[self._traj_partials],
            device=self.device,
        )
        wp.launch(
            _reduce_partials,
            dim=self.n_trajectories,
            inputs=[self._traj_partials, self._red_n_chunks],
            outputs=[self.pred_reduction_traj],
            device=self.device,
        )

        self._integrate_dq(
            joint_q,
            dq_in=self.dq_dof,
            joint_q_out=self.joint_q_proposed,
            joint_qd_out=self.qd_zero,
            step_size=step_size,
        )

        ctx_prop = self._ctx_solver(self.joint_q_proposed, residuals=self.residuals_proposed)
        if self.jacobian_mode in (IKJacobianType.AUTODIFF, IKJacobianType.MIXED):
            self._residuals_autodiff(ctx_prop)
        else:
            self._residuals_analytic(ctx_prop)

        wp.launch(
            compute_costs,
            dim=self.n_batch,
            inputs=[self.residuals_proposed, self.n_residuals],
            outputs=[self.costs_proposed],
            device=self.device,
        )
        self._reduce_costs_traj(self.costs_proposed, self.costs_traj_proposed)

        wp.launch(
            _accept_reject_trajectory,
            dim=self.n_trajectories,
            inputs=[self.costs_traj, self.costs_traj_proposed, self.pred_reduction_traj, self.rho_min],
            outputs=[self.accept_traj],
            device=self.device,
        )
        wp.launch(
            _update_trajectory_rows,
            dim=self.n_batch,
            inputs=[
                self.joint_q_proposed,
                self.residuals_proposed,
                self.accept_traj,
                self.n_frames,
                self.n_coords,
                self.n_residuals,
            ],
            outputs=[joint_q, self.residuals],
            device=self.device,
        )
        wp.launch(
            _update_trajectory_scalars,
            dim=self.n_trajectories,
            inputs=[
                self.accept_traj,
                self.costs_traj_proposed,
                self.lambda_factor,
                self.lambda_min,
                self.lambda_max,
            ],
            outputs=[self.lambda_traj, self.costs_traj],
            device=self.device,
        )

    def _gather_banded(self) -> None:
        """Gather jtj/band/grad into the superblocked (d_bar, l_bar, b_bar) arrays."""
        n_dofs = self.n_dofs
        m = self.kb * n_dofs
        dims = [self.n_trajectories, self.n_superblocks, m, m]
        wp.launch(
            _gather_block_diag,
            dim=dims,
            inputs=[
                self.jtj,
                self.band,
                self.lambda_traj,
                self.fixed_mask,
                self.n_frames,
                self.kb,
                n_dofs,
                self.band_width + 1,
            ],
            outputs=[self.d_bar],
            device=self.device,
        )
        wp.launch(
            _gather_block_offdiag,
            dim=dims,
            inputs=[
                self.band,
                self.fixed_mask,
                self.n_frames,
                self.kb,
                n_dofs,
                self.band_width + 1,
            ],
            outputs=[self.l_bar],
            device=self.device,
        )
        wp.launch(
            _gather_rhs,
            dim=[self.n_trajectories, self.n_superblocks, m],
            inputs=[self.grad, self.fixed_mask, self.n_frames, self.kb, n_dofs],
            outputs=[self.b_bar],
            device=self.device,
        )

    def _refine_step(self, substitute) -> None:
        """One float64-residual refinement pass; `substitute` solves A dx = r."""
        m = self.kb * self.n_dofs
        wp.launch(
            _refine_residual_f64,
            dim=[self.n_trajectories, self.n_superblocks, m],
            inputs=[self.d_bar, self.l_bar, self.b_bar, self.x_bar, self.n_superblocks, m],
            outputs=[self._ref_r_bar],
            device=self.device,
        )
        substitute()
        wp.launch(
            _add_refine_delta,
            dim=[self.n_trajectories, self.n_superblocks, m],
            inputs=[self._ref_dx_bar],
            outputs=[self.x_bar],
            device=self.device,
        )

    def _direct_factor_solve(self) -> None:
        """Solve the gathered banded system into ``x_bar`` (DIRECT backend)."""
        self._block_thomas_solve(
            self.d_bar,
            self.l_bar,
            self.b_bar,
            self._thomas_istart,
            self._thomas_ilen,
            self.x_bar,
            self._chol_ws,
            self._coupling_ws,
        )

        def substitute():
            self._banded_substitute(
                self._ref_r_bar,
                self._thomas_istart,
                self._thomas_ilen,
                1,
                self._chol_ws,
                self._coupling_ws,
                self._ref_dx_bar,
                self.n_trajectories,
            )

        for _ in range(self.refine_iterations):
            self._refine_step(substitute)

    def _solve_direct(self) -> None:
        n_dofs = self.n_dofs
        self._gather_banded()
        self._direct_factor_solve()

        wp.launch(
            _scatter_delta,
            dim=[self.n_batch, n_dofs],
            inputs=[self.x_bar, self.fixed_mask, self.n_frames, self.kb, n_dofs],
            outputs=[self.dq_dof],
            device=self.device,
        )

    def _spike_factor_solve(self) -> None:
        """Solve the gathered banded system into ``x_bar`` (SPIKE backend)."""
        m = self.kb * self.n_dofs
        n_parts = self._spike_n_parts

        # factor every interior in parallel; solve for the local rhs and the
        # left/right spike columns in one pass
        self._spike_interior_solve(
            self.d_bar,
            self.l_bar,
            self.b_bar,
            self._sp_istart,
            self._sp_ilen,
            n_parts,
            self._sp_y,
            self._sp_u,
            self._sp_v,
            self._sp_chol,
            self._sp_coup,
        )

        # symmetric Schur complement on the separator blocks
        self._spike_schur_assemble(
            self.d_bar,
            self.l_bar,
            self.b_bar,
            self._sp_istart,
            self._sp_ilen,
            n_parts,
            self._sp_y,
            self._sp_u,
            self._sp_v,
            self._sp_d_sep,
            self._sp_l_sep,
            self._sp_b_sep,
        )

        # the reduced system is block-tridiagonal with the same block size:
        # reuse the sequential Thomas kernels (n_parts - 1 blocks, cheap)
        self._block_thomas_solve(
            self._sp_d_sep,
            self._sp_l_sep,
            self._sp_b_sep,
            self._sp_sep_istart,
            self._sp_sep_ilen,
            self._sp_x_sep,
            self._sp_chol_sep,
            self._sp_coup_sep,
        )

        # recover the interiors in parallel from the separator solution
        def recover(x_out):
            wp.launch(
                _spike_recover,
                dim=[self.n_trajectories, self.n_superblocks, m],
                inputs=[
                    self._sp_y,
                    self._sp_u,
                    self._sp_v,
                    self._sp_x_sep,
                    self._sp_kind,
                    self._sp_idx,
                    self._sp_loc,
                    n_parts,
                    m,
                ],
                outputs=[x_out],
                device=self.device,
            )

        recover(self.x_bar)

        def substitute():
            # interior substitution of the residual, then the same Schur
            # rhs/reduced-solve/recovery chain as the primary solve
            self._banded_substitute(
                self._ref_r_bar,
                self._sp_istart,
                self._sp_ilen,
                n_parts,
                self._sp_chol,
                self._sp_coup,
                self._sp_y,
                self.n_trajectories * n_parts,
            )
            self._spike_schur_rhs(
                self._ref_r_bar,
                self.l_bar,
                self._sp_istart,
                self._sp_ilen,
                n_parts,
                self._sp_y,
                self._sp_b_sep,
            )
            self._banded_substitute(
                self._sp_b_sep,
                self._sp_sep_istart,
                self._sp_sep_ilen,
                1,
                self._sp_chol_sep,
                self._sp_coup_sep,
                self._sp_x_sep,
                self.n_trajectories,
            )
            recover(self._ref_dx_bar)

        for _ in range(self.refine_iterations):
            self._refine_step(substitute)

    def _solve_spike(self) -> None:
        n_dofs = self.n_dofs
        self._gather_banded()
        self._spike_factor_solve()

        wp.launch(
            _scatter_delta,
            dim=[self.n_batch, n_dofs],
            inputs=[self.x_bar, self.fixed_mask, self.n_frames, self.kb, n_dofs],
            outputs=[self.dq_dof],
            device=self.device,
        )

    def _solve_cg(self) -> None:
        n_dofs = self.n_dofs
        wp.launch(
            _fill_bsr_values,
            dim=[self.n_batch, 2 * self.band_width + 1, n_dofs],
            inputs=[
                self.jtj,
                self.band,
                self.lambda_traj,
                self.fixed_mask,
                self.n_frames,
                n_dofs,
                self.band_width,
                self._hessian.offsets,
                self._hessian.columns,
            ],
            outputs=[self._hessian.scalar_values],
            device=self.device,
        )
        wp.launch(
            _gather_rhs_flat,
            dim=[self.n_batch, n_dofs],
            inputs=[self.grad, self.fixed_mask, self.n_frames, n_dofs],
            outputs=[self._cg_rhs],
            device=self.device,
        )
        # block-Jacobi preconditioner: invert the diagonal blocks
        self._invert_diag_blocks(self._hessian.scalar_values, self._diag_block_idx, self._identity, self._minv)

        # warm-started from the previous iteration's update. ScopedDevice works
        # around warp <= 1.15 launching its reduction kernels on the default
        # device instead of the operand device (fixed on warp main).
        with wp.ScopedDevice(self.device):
            self._cg_state()

        wp.launch(
            _mask_fixed_delta,
            dim=[self.n_batch, n_dofs],
            inputs=[self.fixed_mask, self.n_frames],
            outputs=[self.dq_dof],
            device=self.device,
        )

    # ------------------------------------------------------------------
    # results
    # ------------------------------------------------------------------

    @property
    def trajectory_costs(self) -> wp.array[wp.float32]:
        """Total objective costs of the most recent solve, shape [n_problems]."""
        return self.costs_traj

    def compute_trajectory_costs(self, joint_q: wp.array2d[wp.float32]) -> wp.array[wp.float32]:
        """Evaluate total squared residual costs per trajectory.

        Args:
            joint_q: Joint coordinates to evaluate, shape
                [n_problems * n_frames, joint_coord_count].

        Returns:
            Costs for each trajectory, shape [n_problems].
        """
        super().compute_costs(joint_q)
        self._reduce_costs_traj(self.costs, self.costs_traj)
        return self.costs_traj

    def reset(self) -> None:
        """Clear LM damping and accept/reject state before a new solve."""
        super().reset()
        self.lambda_traj.zero_()
        self.accept_traj.zero_()

    # ------------------------------------------------------------------
    # specialization
    # ------------------------------------------------------------------

    def _jtj_grad_tiled(self, jacobian, residuals_3d, jtj_out, grad_out) -> None:
        raise NotImplementedError("This method should be overridden by specialized solver")

    def _block_thomas_solve(self, d_bar, l_bar, b_bar, istart, ilen, x_bar, chol_ws, coupling_ws) -> None:
        raise NotImplementedError("This method should be overridden by specialized solver")

    def _banded_substitute(self, rhs, istart, ilen, n_parts, chol_ws, coupling_ws, out, dim) -> None:
        raise NotImplementedError("This method should be overridden by specialized solver")

    def _spike_interior_solve(self, d_bar, l_bar, b_bar, istart, ilen, n_parts, y, u, v, chol, coup) -> None:
        raise NotImplementedError("This method should be overridden by specialized solver")

    def _spike_schur_assemble(self, d_bar, l_bar, b_bar, istart, ilen, n_parts, y, u, v, d_sep, l_sep, b_sep) -> None:
        raise NotImplementedError("This method should be overridden by specialized solver")

    def _spike_schur_rhs(self, rhs, l_bar, istart, ilen, n_parts, y, b_sep) -> None:
        raise NotImplementedError("This method should be overridden by specialized solver")

    def _invert_diag_blocks(self, values, diag_idx, identity, minv) -> None:
        raise NotImplementedError("This method should be overridden by specialized solver")

    def _solver_tile_kernels(self) -> list:
        raise NotImplementedError("This method should be overridden by specialized solver")

    def _check_tile_shared_memory(self) -> None:
        """Raise :class:`IKSharedMemoryError` if a tile kernel cannot fit the device.

        Warp sizes a tile kernel's dynamic shared memory from its owner-tile
        expression sites at compile time, so the footprint is known before any
        launch; checking here turns a launch-time CUDA error inside the first
        :meth:`step` into a typed construction-time error. When warp's
        per-kernel footprint is unavailable, the check degrades to the
        compile-free analytic three-tile lower bound.
        """
        device = self.device
        if not device.is_cuda:
            return
        limit = device.max_shared_memory_per_block
        kernels = self._solver_tile_kernels()
        if not kernels:
            return
        # cheap lower bound before compiling anything: every factorization
        # kernel keeps at least three owner MBxMB fp32 tiles live
        m = self.kb * self.n_dofs
        min_bytes = 3 * m * m * 4
        if min_bytes > limit:
            objectives = ", ".join(type(o).__name__ for o in self.objectives)
            raise IKSharedMemoryError(
                f"linear_solver='{self.linear_solver.value}' needs at least {min_bytes} B of dynamic "
                f"shared memory per block, but device {device.alias} allows {limit} B. "
                f"The superblock size k * n_dofs = {self.kb} * {self.n_dofs} = {m} "
                f"set by the objective stack [{objectives}] is too large for this device; "
                f"use linear_solver='cg' instead."
            )
        for kernel in kernels:
            # TODO: replace this reach into warp's module metadata with a
            # supported "will this kernel launch" query once warp provides
            # one (https://github.com/NVIDIA/warp/issues/1699); until then,
            # degrade to the analytic bound above whenever the metadata is
            # unavailable.
            module_exec = kernel.module.load(device, block_dim=int(self.THOMAS_THREADS))
            if module_exec is None:
                # warp returns None for a previously failed build rather
                # than raising again
                continue
            smem_meta = module_exec.meta.get(kernel.get_mangled_name() + "_cuda_kernel_forward_smem_bytes")
            if smem_meta is None:
                continue
            smem = int(smem_meta)
            if smem > limit:
                objectives = ", ".join(type(o).__name__ for o in self.objectives)
                raise IKSharedMemoryError(
                    f"linear_solver='{self.linear_solver.value}' needs {smem} B of dynamic shared memory "
                    f"per block in kernel '{kernel.key}', but device {device.alias} allows {limit} B. "
                    f"The superblock size k * n_dofs = {self.kb} * {self.n_dofs} = {self.kb * self.n_dofs} "
                    f"set by the objective stack [{objectives}] is too large for this device; "
                    f"use linear_solver='cg' instead."
                )

    @classmethod
    def _build_specialized(cls, key: tuple[int, int, int, int, str]) -> type[IKSolverTrajectory]:
        """Build a specialized subclass with tiled kernels for the given dimensions."""
        n_dofs, n_residuals, n_perframe, kb, arch = key

        base_key = (n_dofs, n_residuals, arch)
        base_cls = IKOptimizerLM._cache.get(base_key)
        if base_cls is None:
            base_cls = IKOptimizerLM._build_specialized(base_key)
            IKOptimizerLM._cache[base_key] = base_cls

        DOF = wp.constant(n_dofs)
        # the tile J^T J only sees the per-frame objectives' residual rows;
        # temporal rows enter through the banded stencil coefficients
        RES = wp.constant(max(1, n_perframe))
        MB = wp.constant(kb * n_dofs)

        def _jtj_grad_template(
            jacobians: wp.array3d[wp.float32],  # (n_rows, n_residuals, n_dofs)
            residuals: wp.array3d[wp.float32],  # (n_rows, n_residuals, 1)
            # outputs
            jtj_out: wp.array3d[wp.float32],  # (n_rows, n_dofs, n_dofs)
            grad_out: wp.array3d[wp.float32],  # (n_rows, n_dofs, 1)
        ):
            row = wp.tid()
            J = wp.tile_load(jacobians[row], shape=(RES, DOF))
            r = wp.tile_load(residuals[row], shape=(RES, 1))
            Jt = wp.tile_transpose(J)
            JtJ = wp.tile_zeros(shape=(DOF, DOF), dtype=wp.float32)
            wp.tile_matmul(Jt, J, JtJ)
            wp.tile_store(jtj_out[row], JtJ)
            g = wp.tile_zeros(shape=(DOF, 1), dtype=wp.float32)
            wp.tile_matmul(Jt, r, g)
            wp.tile_store(grad_out[row], g)

        _jtj_grad_template.__name__ = f"_trajik_jtj_grad_{n_dofs}_{n_perframe}"
        _jtj_grad_kernel = wp.kernel(enable_backward=False, module="unique")(_jtj_grad_template)

        # The factorization/substitution kernels below keep at most five
        # owner MBxMB tiles live per kernel: warp charges dynamic shared
        # memory for every owner-tile expression site in a kernel (there is
        # no live-range reuse), so the fused single-kernel forms exceed the
        # shared-memory limit of consumer devices already at MB ~ 48 (e.g.
        # 219,056 B at MB = 70 vs the 101,376 B sm_89 ceiling). In-place
        # tile Cholesky/triangular solves, subtract-accumulate GEMMs
        # (alpha=-1, beta=1), transpose views, and splitting program phases
        # into separate streaming kernels over the global workspaces keep
        # each kernel inside the ceiling without changing the math.

        def _fwd_factor_template(
            d_bar: wp.array4d[wp.float32],  # (n_traj, n_super, m, m)
            l_bar: wp.array4d[wp.float32],  # (n_traj, n_super, m, m), block (g, g - 1)
            b_bar: wp.array3d[wp.float32],  # (n_traj, n_super, m)
            istart: wp.array[wp.int32],  # (n_parts,)
            ilen: wp.array[wp.int32],  # (n_parts,)
            n_parts: int,
            # outputs
            chol_ws: wp.array4d[wp.float32],  # (n_traj * n_parts, l_max, m, m)
            coup_ws: wp.array4d[wp.float32],  # W_t = E_t L_{t-1}^{-T}
            fwd_ws: wp.array3d[wp.float32],  # (n_traj * n_parts, l_max, m)
        ):
            """Streamed block-Cholesky factorization + forward substitution
            over one partition [s, s + len) of one trajectory's chain."""
            row = wp.tid()
            p = row // n_parts
            part = row - p * n_parts
            s = istart[part]
            length = ilen[part]

            A = wp.tile_load(d_bar[p, s], shape=(MB, MB))
            wp.tile_cholesky_inplace(A)
            wp.tile_store(chol_ws[row, 0], A)
            y0 = wp.tile_load(b_bar[p, s], shape=MB)
            wp.tile_lower_solve_inplace(A, y0)
            wp.tile_store(fwd_ws[row, 0], y0)
            for t in range(1, length):
                # W^T = L_{t-1}^{-1} E^T solved in place into the transposed
                # view of E; E's storage then holds W
                E = wp.tile_load(l_bar[p, s + t], shape=(MB, MB))
                ET = wp.tile_transpose(E)
                wp.tile_lower_solve_inplace(A, ET)
                wp.tile_store(coup_ws[row, t], E)
                # S_t = D_t - W W^T
                A = wp.tile_load(d_bar[p, s + t], shape=(MB, MB))
                wp.tile_matmul(E, ET, A, alpha=-1.0, beta=1.0)
                wp.tile_cholesky_inplace(A)
                wp.tile_store(chol_ws[row, t], A)
                # y_t = L_t^{-1} (b_t - W y_{t-1})
                yp = wp.tile_load(fwd_ws[row, t - 1], shape=MB)
                bt = wp.tile_load(b_bar[p, s + t], shape=MB)
                wp.tile_matmul(
                    E, wp.tile_reshape(yp, shape=(MB, 1)), wp.tile_reshape(bt, shape=(MB, 1)), alpha=-1.0, beta=1.0
                )
                wp.tile_lower_solve_inplace(A, bt)
                wp.tile_store(fwd_ws[row, t], bt)

        def _fwd_subst_template(
            rhs: wp.array3d[wp.float32],  # (n_traj, n_chain, m), indexed at s + t
            istart: wp.array[wp.int32],
            ilen: wp.array[wp.int32],
            n_parts: int,
            chol_ws: wp.array4d[wp.float32],
            coup_ws: wp.array4d[wp.float32],
            # outputs
            fwd_ws: wp.array3d[wp.float32],
        ):
            """Forward substitution only, on the stored factors (refinement)."""
            row = wp.tid()
            p = row // n_parts
            part = row - p * n_parts
            s = istart[part]
            length = ilen[part]
            for t in range(length):
                bt = wp.tile_load(rhs[p, s + t], shape=MB)
                if t > 0:
                    E = wp.tile_load(coup_ws[row, t], shape=(MB, MB))
                    yp = wp.tile_load(fwd_ws[row, t - 1], shape=MB)
                    wp.tile_matmul(
                        E, wp.tile_reshape(yp, shape=(MB, 1)), wp.tile_reshape(bt, shape=(MB, 1)), alpha=-1.0, beta=1.0
                    )
                A = wp.tile_load(chol_ws[row, t], shape=(MB, MB))
                wp.tile_lower_solve_inplace(A, bt)
                wp.tile_store(fwd_ws[row, t], bt)

        def _bwd_subst_template(
            istart: wp.array[wp.int32],
            ilen: wp.array[wp.int32],
            n_parts: int,
            chol_ws: wp.array4d[wp.float32],
            coup_ws: wp.array4d[wp.float32],
            # outputs (in place)
            y_ws: wp.array3d[wp.float32],
        ):
            """Backward substitution in place over the forward values."""
            row = wp.tid()
            part = row - (row // n_parts) * n_parts
            length = ilen[part]
            for i in range(length):
                t = length - 1 - i
                yt = wp.tile_load(y_ws[row, t], shape=MB)
                if t < length - 1:
                    E = wp.tile_load(coup_ws[row, t + 1], shape=(MB, MB))
                    xn = wp.tile_load(y_ws[row, t + 1], shape=MB)
                    wp.tile_matmul(
                        wp.tile_transpose(E),
                        wp.tile_reshape(xn, shape=(MB, 1)),
                        wp.tile_reshape(yt, shape=(MB, 1)),
                        alpha=-1.0,
                        beta=1.0,
                    )
                A = wp.tile_load(chol_ws[row, t], shape=(MB, MB))
                wp.tile_upper_solve_inplace(wp.tile_transpose(A), yt)
                wp.tile_store(y_ws[row, t], yt)

        _fwd_factor_template.__name__ = f"_trajik_thomas_fwd_{n_dofs}_{kb}"
        _fwd_factor_kernel = wp.kernel(enable_backward=False, module="unique")(_fwd_factor_template)
        _fwd_subst_template.__name__ = f"_trajik_thomas_fwd_subst_{n_dofs}_{kb}"
        _fwd_subst_kernel = wp.kernel(enable_backward=False, module="unique")(_fwd_subst_template)
        _bwd_subst_template.__name__ = f"_trajik_thomas_bwd_{n_dofs}_{kb}"
        _bwd_subst_kernel = wp.kernel(enable_backward=False, module="unique")(_bwd_subst_template)

        def _u_fwd_template(
            l_bar: wp.array4d[wp.float32],
            istart: wp.array[wp.int32],
            ilen: wp.array[wp.int32],
            n_parts: int,
            chol_ws: wp.array4d[wp.float32],
            coup_ws: wp.array4d[wp.float32],
            # outputs
            u_int: wp.array4d[wp.float32],
        ):
            """Left-spike forward recurrence U_0 = L_0^{-1} Ru,
            U_t = -L_t^{-1} (W_t U_{t-1}); Ru = l_bar[p, s] is the coupling of
            the interior's first row to the separator on its left (a zero
            block for the first partition, since l_bar[p, 0] is zero)."""
            row = wp.tid()
            p = row // n_parts
            part = row - p * n_parts
            s = istart[part]
            length = ilen[part]
            # loop-carried spike tile: T holds U_{t-1} entering each iteration
            T = wp.tile_load(l_bar[p, s], shape=(MB, MB))
            A = wp.tile_load(chol_ws[row, 0], shape=(MB, MB))
            wp.tile_lower_solve_inplace(A, T)
            wp.tile_store(u_int[row, 0], T)
            for t in range(1, length):
                W = wp.tile_load(coup_ws[row, t], shape=(MB, MB))
                T = wp.tile_matmul(W, T, alpha=-1.0)
                A = wp.tile_load(chol_ws[row, t], shape=(MB, MB))
                wp.tile_lower_solve_inplace(A, T)
                wp.tile_store(u_int[row, t], T)

        def _v_last_template(
            l_bar: wp.array4d[wp.float32],
            istart: wp.array[wp.int32],
            ilen: wp.array[wp.int32],
            n_parts: int,
            n_super: int,
            chol_ws: wp.array4d[wp.float32],
            # outputs
            v_int: wp.array4d[wp.float32],
        ):
            """Right-spike forward values are structurally zero below the last
            interior row (V_prev = 0 propagates exact zeros through the
            recurrence); only V[len-1] = L_{len-1}^{-1} Rv is nonzero, with
            Rv = l_bar[p, s + len]^T the coupling of the interior's last row
            to the separator on its right. The caller zeroes v_int before
            this kernel; the clamped out-of-range spike of the last
            partition is never read."""
            row = wp.tid()
            p = row // n_parts
            part = row - p * n_parts
            s = istart[part]
            length = ilen[part]
            v_row = wp.min(s + length, n_super - 1)
            Rv = wp.tile_load(l_bar[p, v_row], shape=(MB, MB))
            A = wp.tile_load(chol_ws[row, length - 1], shape=(MB, MB))
            T = wp.tile_lower_solve(A, wp.tile_transpose(Rv))
            wp.tile_store(v_int[row, length - 1], T)

        def _spike_bwd_template(
            istart: wp.array[wp.int32],
            ilen: wp.array[wp.int32],
            n_parts: int,
            chol_ws: wp.array4d[wp.float32],
            coup_ws: wp.array4d[wp.float32],
            # outputs (in place)
            s_int: wp.array4d[wp.float32],
        ):
            """Backward substitution in place over spike forward values
            (matrix right-hand side; used for both U and V)."""
            row = wp.tid()
            part = row - (row // n_parts) * n_parts
            length = ilen[part]
            for i in range(length):
                t = length - 1 - i
                T = wp.tile_load(s_int[row, t], shape=(MB, MB))
                if t < length - 1:
                    E = wp.tile_load(coup_ws[row, t + 1], shape=(MB, MB))
                    Xn = wp.tile_load(s_int[row, t + 1], shape=(MB, MB))
                    wp.tile_matmul(wp.tile_transpose(E), Xn, T, alpha=-1.0, beta=1.0)
                A = wp.tile_load(chol_ws[row, t], shape=(MB, MB))
                wp.tile_upper_solve_inplace(wp.tile_transpose(A), T)
                wp.tile_store(s_int[row, t], T)

        _u_fwd_template.__name__ = f"_trajik_spike_u_fwd_{n_dofs}_{kb}"
        _u_fwd_kernel = wp.kernel(enable_backward=False, module="unique")(_u_fwd_template)
        _v_last_template.__name__ = f"_trajik_spike_v_last_{n_dofs}_{kb}"
        _v_last_kernel = wp.kernel(enable_backward=False, module="unique")(_v_last_template)
        _spike_bwd_template.__name__ = f"_trajik_spike_bwd_{n_dofs}_{kb}"
        _spike_bwd_kernel = wp.kernel(enable_backward=False, module="unique")(_spike_bwd_template)

        def _schur_diag_template(
            d_bar: wp.array4d[wp.float32],
            l_bar: wp.array4d[wp.float32],
            istart: wp.array[wp.int32],
            ilen: wp.array[wp.int32],
            n_parts: int,
            u_int: wp.array4d[wp.float32],
            v_int: wp.array4d[wp.float32],
            # outputs
            d_sep: wp.array4d[wp.float32],  # (n_traj, n_parts - 1, m, m)
        ):
            """Schur diagonal S_j = D_s - CL V_j[last] - CR^T U_{j+1}[first]."""
            tid = wp.tid()
            n_sep = n_parts - 1
            p = tid // n_sep
            j = tid - p * n_sep
            s = istart[j] + ilen[j]  # global index of separator j
            last = ilen[j] - 1

            A = wp.tile_load(d_bar[p, s], shape=(MB, MB))
            B = wp.tile_load(l_bar[p, s], shape=(MB, MB))  # CL = block (s, s - 1)
            C = wp.tile_load(v_int[p * n_parts + j, last], shape=(MB, MB))
            wp.tile_matmul(B, C, A, alpha=-1.0, beta=1.0)
            B = wp.tile_load(l_bar[p, s + 1], shape=(MB, MB))  # CR = block (s + 1, s)
            C = wp.tile_load(u_int[p * n_parts + j + 1, 0], shape=(MB, MB))
            wp.tile_matmul(wp.tile_transpose(B), C, A, alpha=-1.0, beta=1.0)
            wp.tile_store(d_sep[p, j], A)

        def _schur_lower_template(
            l_bar: wp.array4d[wp.float32],
            istart: wp.array[wp.int32],
            ilen: wp.array[wp.int32],
            n_parts: int,
            u_int: wp.array4d[wp.float32],
            # outputs
            l_sep: wp.array4d[wp.float32],
        ):
            """Schur sub-diagonal -CL U_j[last] (never read for j = 0: U_0 is
            exactly zero because l_bar[p, 0] is zero)."""
            tid = wp.tid()
            n_sep = n_parts - 1
            p = tid // n_sep
            j = tid - p * n_sep
            s = istart[j] + ilen[j]
            last = ilen[j] - 1
            B = wp.tile_load(l_bar[p, s], shape=(MB, MB))
            C = wp.tile_load(u_int[p * n_parts + j, last], shape=(MB, MB))
            T = wp.tile_matmul(B, C, alpha=-1.0)
            wp.tile_store(l_sep[p, j], T)

        def _schur_rhs_template(
            rhs: wp.array3d[wp.float32],
            l_bar: wp.array4d[wp.float32],
            istart: wp.array[wp.int32],
            ilen: wp.array[wp.int32],
            n_parts: int,
            y_int: wp.array3d[wp.float32],
            # outputs
            b_sep: wp.array3d[wp.float32],
        ):
            """Schur right-hand side b_s - CL y_j[last] - CR^T y_{j+1}[first]."""
            tid = wp.tid()
            n_sep = n_parts - 1
            p = tid // n_sep
            j = tid - p * n_sep
            s = istart[j] + ilen[j]
            last = ilen[j] - 1
            bt = wp.tile_load(rhs[p, s], shape=MB)
            B = wp.tile_load(l_bar[p, s], shape=(MB, MB))
            yl = wp.tile_load(y_int[p * n_parts + j, last], shape=MB)
            wp.tile_matmul(
                B, wp.tile_reshape(yl, shape=(MB, 1)), wp.tile_reshape(bt, shape=(MB, 1)), alpha=-1.0, beta=1.0
            )
            B = wp.tile_load(l_bar[p, s + 1], shape=(MB, MB))
            yr = wp.tile_load(y_int[p * n_parts + j + 1, 0], shape=MB)
            wp.tile_matmul(
                wp.tile_transpose(B),
                wp.tile_reshape(yr, shape=(MB, 1)),
                wp.tile_reshape(bt, shape=(MB, 1)),
                alpha=-1.0,
                beta=1.0,
            )
            wp.tile_store(b_sep[p, j], bt)

        _schur_diag_template.__name__ = f"_trajik_schur_diag_{n_dofs}_{kb}"
        _schur_diag_kernel = wp.kernel(enable_backward=False, module="unique")(_schur_diag_template)
        _schur_lower_template.__name__ = f"_trajik_schur_lower_{n_dofs}_{kb}"
        _schur_lower_kernel = wp.kernel(enable_backward=False, module="unique")(_schur_lower_template)
        _schur_rhs_template.__name__ = f"_trajik_schur_rhs_{n_dofs}_{kb}"
        _schur_rhs_kernel = wp.kernel(enable_backward=False, module="unique")(_schur_rhs_template)

        def _inv_diag_template(
            values: wp.array3d[wp.float32],  # (nnz, n_dofs, n_dofs)
            diag_idx: wp.array[wp.int32],  # (n_rows,)
            identity: wp.array2d[wp.float32],  # (n_dofs, n_dofs)
            # outputs
            minv: wp.array3d[wp.float32],  # (n_rows, n_dofs, n_dofs)
        ):
            row = wp.tid()
            idx = diag_idx[row]
            A = wp.tile_load(values[idx], shape=(DOF, DOF))
            L = wp.tile_cholesky(A)
            eye = wp.tile_load(identity, shape=(DOF, DOF))
            X = wp.tile_cholesky_solve(L, eye)
            wp.tile_store(minv[row], X)

        _inv_diag_template.__name__ = f"_trajik_inv_diag_{n_dofs}"
        _inv_diag_kernel = wp.kernel(enable_backward=False, module="unique")(_inv_diag_template)

        class _Specialized(IKSolverTrajectory):
            TILE_N_DOFS = wp.constant(n_dofs)
            TILE_N_RESIDUALS = wp.constant(n_residuals)
            TILE_M_SUPER = wp.constant(kb * n_dofs)
            TILE_THREADS = wp.constant(32)
            # block size of the factorization/substitution kernels; measured
            # fastest of {64, 128, 256} at production size (MB = 70, sm_89)
            # for both the single-trajectory SPIKE and batched DIRECT shapes
            # (e.g. 16 x 3,991 frames: 436 -> 263 ms/solve vs 64 threads)
            THOMAS_THREADS = wp.constant(256)

            def _jtj_grad_tiled(self, jacobian, residuals_3d, jtj_out, grad_out) -> None:
                wp.launch_tiled(
                    _jtj_grad_kernel,
                    dim=[self.n_batch],
                    inputs=[jacobian, residuals_3d],
                    outputs=[jtj_out, grad_out],
                    block_dim=self.TILE_THREADS,
                    device=self.device,
                )

            def _block_thomas_solve(self, d_bar, l_bar, b_bar, istart, ilen, x_bar, chol_ws, coupling_ws) -> None:
                # forward pass writes its values into x_bar; the backward
                # pass then solves in place
                wp.launch_tiled(
                    _fwd_factor_kernel,
                    dim=[self.n_trajectories],
                    inputs=[d_bar, l_bar, b_bar, istart, ilen, 1],
                    outputs=[chol_ws, coupling_ws, x_bar],
                    block_dim=self.THOMAS_THREADS,
                    device=self.device,
                )
                wp.launch_tiled(
                    _bwd_subst_kernel,
                    dim=[self.n_trajectories],
                    inputs=[istart, ilen, 1, chol_ws, coupling_ws],
                    outputs=[x_bar],
                    block_dim=self.THOMAS_THREADS,
                    device=self.device,
                )

            def _banded_substitute(self, rhs, istart, ilen, n_parts, chol_ws, coupling_ws, out, dim) -> None:
                wp.launch_tiled(
                    _fwd_subst_kernel,
                    dim=[dim],
                    inputs=[rhs, istart, ilen, n_parts, chol_ws, coupling_ws],
                    outputs=[out],
                    block_dim=self.THOMAS_THREADS,
                    device=self.device,
                )
                wp.launch_tiled(
                    _bwd_subst_kernel,
                    dim=[dim],
                    inputs=[istart, ilen, n_parts, chol_ws, coupling_ws],
                    outputs=[out],
                    block_dim=self.THOMAS_THREADS,
                    device=self.device,
                )

            def _spike_interior_solve(self, d_bar, l_bar, b_bar, istart, ilen, n_parts, y, u, v, chol, coup) -> None:
                rows = self.n_trajectories * n_parts
                # the lean right-spike kernel writes only the structurally
                # nonzero last interior row
                v.zero_()
                wp.launch_tiled(
                    _fwd_factor_kernel,
                    dim=[rows],
                    inputs=[d_bar, l_bar, b_bar, istart, ilen, n_parts],
                    outputs=[chol, coup, y],
                    block_dim=self.THOMAS_THREADS,
                    device=self.device,
                )
                wp.launch_tiled(
                    _u_fwd_kernel,
                    dim=[rows],
                    inputs=[l_bar, istart, ilen, n_parts, chol, coup],
                    outputs=[u],
                    block_dim=self.THOMAS_THREADS,
                    device=self.device,
                )
                wp.launch_tiled(
                    _v_last_kernel,
                    dim=[rows],
                    inputs=[l_bar, istart, ilen, n_parts, int(l_bar.shape[1]), chol],
                    outputs=[v],
                    block_dim=self.THOMAS_THREADS,
                    device=self.device,
                )
                wp.launch_tiled(
                    _bwd_subst_kernel,
                    dim=[rows],
                    inputs=[istart, ilen, n_parts, chol, coup],
                    outputs=[y],
                    block_dim=self.THOMAS_THREADS,
                    device=self.device,
                )
                for spike in (u, v):
                    wp.launch_tiled(
                        _spike_bwd_kernel,
                        dim=[rows],
                        inputs=[istart, ilen, n_parts, chol, coup],
                        outputs=[spike],
                        block_dim=self.THOMAS_THREADS,
                        device=self.device,
                    )

            def _spike_schur_assemble(
                self, d_bar, l_bar, b_bar, istart, ilen, n_parts, y, u, v, d_sep, l_sep, b_sep
            ) -> None:
                n_sep_rows = self.n_trajectories * (n_parts - 1)
                wp.launch_tiled(
                    _schur_diag_kernel,
                    dim=[n_sep_rows],
                    inputs=[d_bar, l_bar, istart, ilen, n_parts, u, v],
                    outputs=[d_sep],
                    block_dim=self.THOMAS_THREADS,
                    device=self.device,
                )
                wp.launch_tiled(
                    _schur_lower_kernel,
                    dim=[n_sep_rows],
                    inputs=[l_bar, istart, ilen, n_parts, u],
                    outputs=[l_sep],
                    block_dim=self.THOMAS_THREADS,
                    device=self.device,
                )
                self._spike_schur_rhs(b_bar, l_bar, istart, ilen, n_parts, y, b_sep)

            def _spike_schur_rhs(self, rhs, l_bar, istart, ilen, n_parts, y, b_sep) -> None:
                wp.launch_tiled(
                    _schur_rhs_kernel,
                    dim=[self.n_trajectories * (n_parts - 1)],
                    inputs=[rhs, l_bar, istart, ilen, n_parts, y],
                    outputs=[b_sep],
                    block_dim=self.THOMAS_THREADS,
                    device=self.device,
                )

            def _solver_tile_kernels(self) -> list:
                if self.linear_solver is IKLinearSolver.DIRECT:
                    kernels = [_fwd_factor_kernel, _bwd_subst_kernel]
                elif self.linear_solver is IKLinearSolver.SPIKE:
                    kernels = [
                        _fwd_factor_kernel,
                        _bwd_subst_kernel,
                        _u_fwd_kernel,
                        _v_last_kernel,
                        _spike_bwd_kernel,
                        _schur_diag_kernel,
                        _schur_lower_kernel,
                        _schur_rhs_kernel,
                    ]
                else:
                    return []
                if self.refine_iterations > 0:
                    kernels.append(_fwd_subst_kernel)
                return kernels

            def _invert_diag_blocks(self, values, diag_idx, identity, minv) -> None:
                wp.launch_tiled(
                    _inv_diag_kernel,
                    dim=[self.n_batch],
                    inputs=[values, diag_idx, identity],
                    outputs=[minv],
                    block_dim=self.TILE_THREADS,
                    device=self.device,
                )

        _Specialized.__name__ = f"IKTraj_{n_dofs}x{n_residuals}pf{n_perframe}_k{kb}"
        _Specialized._integrate_dq_dof = staticmethod(base_cls._integrate_dq_dof)
        _Specialized._compute_motion_subspace_2d = staticmethod(base_cls._compute_motion_subspace_2d)
        _Specialized._fk_two_pass = staticmethod(base_cls._fk_two_pass)
        return _Specialized
