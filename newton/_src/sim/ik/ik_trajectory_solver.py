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

    Exact solve, sequential over frames inside a single kernel, parallel
    across trajectories. Preferred for moderate horizon lengths and larger
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

    The interior factorization carries three simultaneous right-hand sides,
    so it reaches the tile shared-memory limit at smaller superblock sizes
    than :attr:`DIRECT` (roughly ``k * n_dofs <= 24–32`` fp32); larger
    problems should use :attr:`DIRECT` or :attr:`CG`.
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
    coeffs: wp.array4d[wp.float32],  # (n_rows, width + 1, n_dofs, n_dofs)
    width: int,
    n_frames: int,
    n_dofs: int,
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
                for c in range(n_dofs):
                    acc += coeffs[rs, i, c, a] * coeffs[rs, i + d, c, b]
        band[row, d, a, b] += acc


@wp.kernel
def _accumulate_temporal_grad(
    coeffs: wp.array4d[wp.float32],  # (n_rows, width + 1, n_dofs, n_dofs)
    residuals: wp.array2d[wp.float32],  # (n_rows, n_residuals)
    start_idx: int,
    width: int,
    n_frames: int,
    n_dofs: int,
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
            for c in range(n_dofs):
                gacc += coeffs[rs, j, c, a] * residuals[rs, start_idx + c]
    grad[row, a] += gacc


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
    row, s = wp.tid()
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
        for a in range(n_dofs):
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
        for a in range(n_dofs):
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
        band_width = max((o.stencil_width() for o in objectives if isinstance(o, IKObjectiveTemporal)), default=0)
        kb = max(band_width, 1)
        arch = model.device.arch
        key = (n_dofs, n_residuals, kb, arch)

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
    ) -> None:
        if isinstance(jacobian_mode, str):
            jacobian_mode = IKJacobianType(jacobian_mode)
        if isinstance(linear_solver, str):
            linear_solver = IKLinearSolver(linear_solver)
        if n_frames < 2:
            raise ValueError("n_frames must be >= 2")
        if n_problems < 1:
            raise ValueError("n_problems must be >= 1")

        self.n_frames = n_frames
        self.n_trajectories = n_problems
        self.linear_solver = linear_solver
        self.cg_iterations = cg_iterations
        self.cg_tol = cg_tol

        self.temporal_objectives = [o for o in objectives if isinstance(o, IKObjectiveTemporal)]
        self.band_width = max((o.stencil_width() for o in self.temporal_objectives), default=0)
        self.kb = max(self.band_width, 1)
        self.n_superblocks = (n_frames + self.kb - 1) // self.kb

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

    def _alloc_solver_buffers(self, grad: bool) -> None:
        super()._alloc_solver_buffers(grad)

        device = self.device
        n_rows = self.n_batch
        n_dofs = self.n_dofs
        n_traj = self.n_trajectories
        n_super = self.n_superblocks
        m = self.kb * n_dofs

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
        if self.linear_solver is IKLinearSolver.DIRECT:
            self._chol_ws = wp.zeros((n_traj, n_super, m, m), dtype=wp.float32, device=device)
            self._coupling_ws = wp.zeros((n_traj, n_super, m, m), dtype=wp.float32, device=device)
            self._fwd_ws = wp.zeros((n_traj, n_super, m), dtype=wp.float32, device=device)
        elif self.linear_solver is IKLinearSolver.SPIKE:
            self._alloc_spike_buffers()
        else:
            self._alloc_cg_buffers()

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
        self._sp_fwd_sep = wp.zeros((n_traj, n_sep, m), dtype=wp.float32, device=device)

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

        # dense per-frame Jacobian of the per-frame objectives
        self._jacobian_at(ctx_curr)

        residuals_flat = ctx_curr.residuals.flatten()
        residuals_3d_flat = self.residuals_3d.flatten()
        wp.copy(residuals_3d_flat, residuals_flat)

        # block-diagonal J^T J and gradient of the per-frame objectives
        self._jtj_grad_tiled(ctx_curr.jacobian_out, self.residuals_3d, self.jtj, self.grad3)

        # banded coupling and gradient of the temporal objectives
        self.band.zero_()
        for obj, offset in zip(self.objectives, self.residual_offsets, strict=False):
            if isinstance(obj, IKObjectiveTemporal):
                obj.compute_coeffs(joint_q)
                wp.launch(
                    _accumulate_temporal_band,
                    dim=[self.n_batch, self.n_dofs, self.n_dofs],
                    inputs=[obj.coeffs, obj.stencil_width(), self.n_frames, self.n_dofs],
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
                        self.n_dofs,
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

    def _solve_direct(self) -> None:
        n_dofs = self.n_dofs
        self._gather_banded()

        self._block_thomas_solve(
            self.d_bar,
            self.l_bar,
            self.b_bar,
            self.n_superblocks,
            self.x_bar,
            self._chol_ws,
            self._coupling_ws,
            self._fwd_ws,
        )

        wp.launch(
            _scatter_delta,
            dim=[self.n_batch, n_dofs],
            inputs=[self.x_bar, self.fixed_mask, self.n_frames, self.kb, n_dofs],
            outputs=[self.dq_dof],
            device=self.device,
        )

    def _solve_spike(self) -> None:
        n_dofs = self.n_dofs
        m = self.kb * n_dofs
        n_parts = self._spike_n_parts

        self._gather_banded()

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
        # reuse the sequential Thomas kernel (n_parts - 1 blocks, cheap)
        self._block_thomas_solve(
            self._sp_d_sep,
            self._sp_l_sep,
            self._sp_b_sep,
            n_parts - 1,
            self._sp_x_sep,
            self._sp_chol_sep,
            self._sp_coup_sep,
            self._sp_fwd_sep,
        )

        # recover the interiors in parallel from the separator solution
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
            outputs=[self.x_bar],
            device=self.device,
        )

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
            dim=[self.n_batch, 2 * self.band_width + 1],
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

    def _block_thomas_solve(self, d_bar, l_bar, b_bar, n_super, x_bar, chol_ws, coupling_ws, fwd_ws) -> None:
        raise NotImplementedError("This method should be overridden by specialized solver")

    def _spike_interior_solve(self, d_bar, l_bar, b_bar, istart, ilen, n_parts, y, u, v, chol, coup) -> None:
        raise NotImplementedError("This method should be overridden by specialized solver")

    def _spike_schur_assemble(self, d_bar, l_bar, b_bar, istart, ilen, n_parts, y, u, v, d_sep, l_sep, b_sep) -> None:
        raise NotImplementedError("This method should be overridden by specialized solver")

    def _invert_diag_blocks(self, values, diag_idx, identity, minv) -> None:
        raise NotImplementedError("This method should be overridden by specialized solver")

    @classmethod
    def _build_specialized(cls, key: tuple[int, int, int, str]) -> type[IKSolverTrajectory]:
        """Build a specialized subclass with tiled kernels for the given dimensions."""
        n_dofs, n_residuals, kb, arch = key

        base_key = (n_dofs, n_residuals, arch)
        base_cls = IKOptimizerLM._cache.get(base_key)
        if base_cls is None:
            base_cls = IKOptimizerLM._build_specialized(base_key)
            IKOptimizerLM._cache[base_key] = base_cls

        DOF = wp.constant(n_dofs)
        RES = wp.constant(n_residuals)
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

        _jtj_grad_template.__name__ = f"_trajik_jtj_grad_{n_dofs}_{n_residuals}"
        _jtj_grad_kernel = wp.kernel(enable_backward=False, module="unique")(_jtj_grad_template)

        def _thomas_template(
            d_bar: wp.array4d[wp.float32],  # (n_traj, n_super, m, m)
            l_bar: wp.array4d[wp.float32],  # (n_traj, n_super, m, m), block (g, g - 1)
            b_bar: wp.array3d[wp.float32],  # (n_traj, n_super, m)
            n_super: int,
            # outputs
            x_bar: wp.array3d[wp.float32],  # (n_traj, n_super, m)
            chol_ws: wp.array4d[wp.float32],  # Cholesky factors workspace
            coupling_ws: wp.array4d[wp.float32],  # coupling factors workspace
            fwd_ws: wp.array3d[wp.float32],  # forward-substitution workspace
        ):
            p = wp.tid()

            # forward factorization + forward substitution
            Dt = wp.tile_load(d_bar[p, 0], shape=(MB, MB))
            Ct = wp.tile_cholesky(Dt)
            wp.tile_store(chol_ws[p, 0], Ct)
            bt = wp.tile_load(b_bar[p, 0], shape=MB)
            yt = wp.tile_lower_solve(Ct, bt)
            wp.tile_store(fwd_ws[p, 0], yt)

            for t in range(1, n_super):
                Cprev = wp.tile_load(chol_ws[p, t - 1], shape=(MB, MB))
                Lt = wp.tile_load(l_bar[p, t], shape=(MB, MB))
                # W = L C^-T from C W^T = L^T
                WtT = wp.tile_lower_solve(Cprev, wp.tile_transpose(Lt))
                Wt = wp.tile_transpose(WtT)
                wp.tile_store(coupling_ws[p, t], Wt)

                Dt = wp.tile_load(d_bar[p, t], shape=(MB, MB))
                WWt = wp.tile_zeros(shape=(MB, MB), dtype=wp.float32)
                wp.tile_matmul(Wt, wp.tile_transpose(Wt), WWt)
                S = wp.tile_map(wp.sub, Dt, WWt)
                Ct = wp.tile_cholesky(S)
                wp.tile_store(chol_ws[p, t], Ct)

                yprev = wp.tile_load(fwd_ws[p, t - 1], shape=MB)
                bt = wp.tile_load(b_bar[p, t], shape=MB)
                Wy = wp.tile_zeros(shape=(MB, 1), dtype=wp.float32)
                wp.tile_matmul(Wt, wp.tile_reshape(yprev, shape=(MB, 1)), Wy)
                rhs = wp.tile_map(wp.sub, bt, wp.tile_reshape(Wy, shape=(MB,)))
                yt = wp.tile_lower_solve(Ct, rhs)
                wp.tile_store(fwd_ws[p, t], yt)

            # back substitution
            Clast = wp.tile_load(chol_ws[p, n_super - 1], shape=(MB, MB))
            ylast = wp.tile_load(fwd_ws[p, n_super - 1], shape=MB)
            xt = wp.tile_upper_solve(wp.tile_transpose(Clast), ylast)
            wp.tile_store(x_bar[p, n_super - 1], xt)

            for i in range(1, n_super):
                t = n_super - 1 - i
                Ct = wp.tile_load(chol_ws[p, t], shape=(MB, MB))
                Wnext = wp.tile_load(coupling_ws[p, t + 1], shape=(MB, MB))
                xnext = wp.tile_load(x_bar[p, t + 1], shape=MB)
                yt = wp.tile_load(fwd_ws[p, t], shape=MB)
                Wx = wp.tile_zeros(shape=(MB, 1), dtype=wp.float32)
                wp.tile_matmul(wp.tile_transpose(Wnext), wp.tile_reshape(xnext, shape=(MB, 1)), Wx)
                rhs = wp.tile_map(wp.sub, yt, wp.tile_reshape(Wx, shape=(MB,)))
                xt = wp.tile_upper_solve(wp.tile_transpose(Ct), rhs)
                wp.tile_store(x_bar[p, t], xt)

        _thomas_template.__name__ = f"_trajik_block_thomas_{n_dofs}_{kb}"
        _thomas_kernel = wp.kernel(enable_backward=False, module="unique")(_thomas_template)

        def _spike_interior_template(
            d_bar: wp.array4d[wp.float32],  # (n_traj, n_super, m, m)
            l_bar: wp.array4d[wp.float32],  # (n_traj, n_super, m, m), block (g, g - 1)
            b_bar: wp.array3d[wp.float32],  # (n_traj, n_super, m)
            istart: wp.array[wp.int32],  # (n_parts,)
            ilen: wp.array[wp.int32],  # (n_parts,)
            n_parts: int,
            # outputs
            y_int: wp.array3d[wp.float32],  # (n_traj * n_parts, l_max, m)
            u_int: wp.array4d[wp.float32],  # (n_traj * n_parts, l_max, m, m)
            v_int: wp.array4d[wp.float32],  # (n_traj * n_parts, l_max, m, m)
            chol_ws: wp.array4d[wp.float32],
            coup_ws: wp.array4d[wp.float32],
        ):
            tid = wp.tid()
            p = tid // n_parts
            part = tid - p * n_parts
            row = tid
            s = istart[part]
            length = ilen[part]

            # ----- forward factorization + forward substitution -----
            Dt = wp.tile_load(d_bar[p, s], shape=(MB, MB))
            Ct = wp.tile_cholesky(Dt)
            wp.tile_store(chol_ws[row, 0], Ct)

            bt = wp.tile_load(b_bar[p, s], shape=MB)
            yt = wp.tile_lower_solve(Ct, bt)
            wp.tile_store(y_int[row, 0], yt)

            # Branchless masking: the gather kernels write zero blocks at
            # l_bar[p, 0], so loading row 0 yields a zero rhs; the clamped
            # out-of-range spike of the last partition is never read.
            n_super = l_bar.shape[1]
            v_row = wp.min(s + length, n_super - 1)

            # left spike rhs: coupling of the interior's first row to the
            # separator on its left (block (s, s - 1) = l_bar[s]; zero for
            # the first partition since l_bar[p, 0] is zero)
            Ru = wp.tile_load(l_bar[p, s], shape=(MB, MB))
            Ut = wp.tile_lower_solve(Ct, Ru)
            wp.tile_store(u_int[row, 0], Ut)

            # right spike rhs: coupling of the interior's last row to the
            # separator on its right (block (s+L-1, s+L) = l_bar[s+L]^T);
            # nonzero only on the interior's last row
            # same expression form as the loop body so the tile layouts of
            # the reassigned variables agree across iterations
            rv0 = wp.where(length == 1, v_row, 0)
            Rv = wp.tile_transpose(wp.tile_load(l_bar[p, rv0], shape=(MB, MB)))
            WV = wp.tile_zeros(shape=(MB, MB), dtype=wp.float32)
            Vt = wp.tile_lower_solve(Ct, wp.tile_map(wp.sub, Rv, WV))
            wp.tile_store(v_int[row, 0], Vt)

            for t in range(1, length):
                Cprev = wp.tile_load(chol_ws[row, t - 1], shape=(MB, MB))
                Lt = wp.tile_load(l_bar[p, s + t], shape=(MB, MB))
                WtT = wp.tile_lower_solve(Cprev, wp.tile_transpose(Lt))
                Wt = wp.tile_transpose(WtT)
                wp.tile_store(coup_ws[row, t], Wt)

                Dt = wp.tile_load(d_bar[p, s + t], shape=(MB, MB))
                WWt = wp.tile_zeros(shape=(MB, MB), dtype=wp.float32)
                wp.tile_matmul(Wt, wp.tile_transpose(Wt), WWt)
                St = wp.tile_map(wp.sub, Dt, WWt)
                Ct = wp.tile_cholesky(St)
                wp.tile_store(chol_ws[row, t], Ct)

                yprev = wp.tile_load(y_int[row, t - 1], shape=MB)
                bt = wp.tile_load(b_bar[p, s + t], shape=MB)
                Wy = wp.tile_zeros(shape=(MB, 1), dtype=wp.float32)
                wp.tile_matmul(Wt, wp.tile_reshape(yprev, shape=(MB, 1)), Wy)
                rhs = wp.tile_map(wp.sub, bt, wp.tile_reshape(Wy, shape=(MB,)))
                yt = wp.tile_lower_solve(Ct, rhs)
                wp.tile_store(y_int[row, t], yt)

                # left spike: zero rhs past the first row
                Uprev = wp.tile_load(u_int[row, t - 1], shape=(MB, MB))
                WU = wp.tile_zeros(shape=(MB, MB), dtype=wp.float32)
                wp.tile_matmul(Wt, Uprev, WU)
                Ut = wp.tile_lower_solve(Ct, wp.tile_map(wp.neg, WU))
                wp.tile_store(u_int[row, t], Ut)

                # right spike: rhs only on the last row (fwd stays zero
                # before, so the recursion term W V_prev vanishes there)
                rvt = wp.where(t == length - 1, v_row, 0)
                Rv = wp.tile_transpose(wp.tile_load(l_bar[p, rvt], shape=(MB, MB)))
                Vprev = wp.tile_load(v_int[row, t - 1], shape=(MB, MB))
                WV2 = wp.tile_zeros(shape=(MB, MB), dtype=wp.float32)
                wp.tile_matmul(Wt, Vprev, WV2)
                Vt = wp.tile_lower_solve(Ct, wp.tile_map(wp.sub, Rv, WV2))
                wp.tile_store(v_int[row, t], Vt)

            # ----- back substitution (in place over the forward values) -----
            Clast = wp.tile_load(chol_ws[row, length - 1], shape=(MB, MB))
            ylast = wp.tile_load(y_int[row, length - 1], shape=MB)
            xt = wp.tile_upper_solve(wp.tile_transpose(Clast), ylast)
            wp.tile_store(y_int[row, length - 1], xt)
            Ulast = wp.tile_load(u_int[row, length - 1], shape=(MB, MB))
            Ux = wp.tile_upper_solve(wp.tile_transpose(Clast), Ulast)
            wp.tile_store(u_int[row, length - 1], Ux)
            Vlast = wp.tile_load(v_int[row, length - 1], shape=(MB, MB))
            Vx = wp.tile_upper_solve(wp.tile_transpose(Clast), Vlast)
            wp.tile_store(v_int[row, length - 1], Vx)

            for i in range(1, length):
                t = length - 1 - i
                Ct = wp.tile_load(chol_ws[row, t], shape=(MB, MB))
                Wnext = wp.tile_load(coup_ws[row, t + 1], shape=(MB, MB))
                WnT = wp.tile_transpose(Wnext)

                ynext = wp.tile_load(y_int[row, t + 1], shape=MB)
                yfwd = wp.tile_load(y_int[row, t], shape=MB)
                Wx = wp.tile_zeros(shape=(MB, 1), dtype=wp.float32)
                wp.tile_matmul(WnT, wp.tile_reshape(ynext, shape=(MB, 1)), Wx)
                rhs = wp.tile_map(wp.sub, yfwd, wp.tile_reshape(Wx, shape=(MB,)))
                xt = wp.tile_upper_solve(wp.tile_transpose(Ct), rhs)
                wp.tile_store(y_int[row, t], xt)

                Unext = wp.tile_load(u_int[row, t + 1], shape=(MB, MB))
                Ufwd = wp.tile_load(u_int[row, t], shape=(MB, MB))
                WUm = wp.tile_zeros(shape=(MB, MB), dtype=wp.float32)
                wp.tile_matmul(WnT, Unext, WUm)
                Ux = wp.tile_upper_solve(wp.tile_transpose(Ct), wp.tile_map(wp.sub, Ufwd, WUm))
                wp.tile_store(u_int[row, t], Ux)

                Vnext = wp.tile_load(v_int[row, t + 1], shape=(MB, MB))
                Vfwd = wp.tile_load(v_int[row, t], shape=(MB, MB))
                WVm = wp.tile_zeros(shape=(MB, MB), dtype=wp.float32)
                wp.tile_matmul(WnT, Vnext, WVm)
                Vx = wp.tile_upper_solve(wp.tile_transpose(Ct), wp.tile_map(wp.sub, Vfwd, WVm))
                wp.tile_store(v_int[row, t], Vx)

        _spike_interior_template.__name__ = f"_trajik_spike_interior_{n_dofs}_{kb}"
        _spike_interior_kernel = wp.kernel(enable_backward=False, module="unique")(_spike_interior_template)

        def _spike_schur_template(
            d_bar: wp.array4d[wp.float32],
            l_bar: wp.array4d[wp.float32],
            b_bar: wp.array3d[wp.float32],
            istart: wp.array[wp.int32],
            ilen: wp.array[wp.int32],
            n_parts: int,
            y_int: wp.array3d[wp.float32],
            u_int: wp.array4d[wp.float32],
            v_int: wp.array4d[wp.float32],
            # outputs
            d_sep: wp.array4d[wp.float32],  # (n_traj, n_parts - 1, m, m)
            l_sep: wp.array4d[wp.float32],
            b_sep: wp.array3d[wp.float32],
        ):
            tid = wp.tid()
            n_sep = n_parts - 1
            p = tid // n_sep
            j = tid - p * n_sep
            s = istart[j] + ilen[j]  # global index of separator j
            last = ilen[j] - 1

            CL = wp.tile_load(l_bar[p, s], shape=(MB, MB))  # block (s, s - 1)
            CRt = wp.tile_transpose(wp.tile_load(l_bar[p, s + 1], shape=(MB, MB)))  # block (s, s + 1)

            # S_j = D_s - CL V_j[last] - CR^T U_{j+1}[first]
            Dt = wp.tile_load(d_bar[p, s], shape=(MB, MB))
            Vl = wp.tile_load(v_int[p * n_parts + j, last], shape=(MB, MB))
            Uf = wp.tile_load(u_int[p * n_parts + j + 1, 0], shape=(MB, MB))
            T1 = wp.tile_zeros(shape=(MB, MB), dtype=wp.float32)
            wp.tile_matmul(CL, Vl, T1)
            T2 = wp.tile_zeros(shape=(MB, MB), dtype=wp.float32)
            wp.tile_matmul(CRt, Uf, T2)
            St = wp.tile_map(wp.sub, wp.tile_map(wp.sub, Dt, T1), T2)
            wp.tile_store(d_sep[p, j], St)

            # sub-diagonal of the reduced system: -CL U_j[last] (never read for j = 0)
            Ul = wp.tile_load(u_int[p * n_parts + j, last], shape=(MB, MB))
            T3 = wp.tile_zeros(shape=(MB, MB), dtype=wp.float32)
            wp.tile_matmul(CL, Ul, T3)
            wp.tile_store(l_sep[p, j], wp.tile_map(wp.neg, T3))

            # rhs: b_s - CL y_j[last] - CR^T y_{j+1}[first]
            yl = wp.tile_load(y_int[p * n_parts + j, last], shape=MB)
            yr = wp.tile_load(y_int[p * n_parts + j + 1, 0], shape=MB)
            t4 = wp.tile_zeros(shape=(MB, 1), dtype=wp.float32)
            wp.tile_matmul(CL, wp.tile_reshape(yl, shape=(MB, 1)), t4)
            t5 = wp.tile_zeros(shape=(MB, 1), dtype=wp.float32)
            wp.tile_matmul(CRt, wp.tile_reshape(yr, shape=(MB, 1)), t5)
            bs = wp.tile_load(b_bar[p, s], shape=MB)
            res = wp.tile_map(
                wp.sub, wp.tile_map(wp.sub, bs, wp.tile_reshape(t4, shape=(MB,))), wp.tile_reshape(t5, shape=(MB,))
            )
            wp.tile_store(b_sep[p, j], res)

        _spike_schur_template.__name__ = f"_trajik_spike_schur_{n_dofs}_{kb}"
        _spike_schur_kernel = wp.kernel(enable_backward=False, module="unique")(_spike_schur_template)

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
            THOMAS_THREADS = wp.constant(64)

            def _jtj_grad_tiled(self, jacobian, residuals_3d, jtj_out, grad_out) -> None:
                wp.launch_tiled(
                    _jtj_grad_kernel,
                    dim=[self.n_batch],
                    inputs=[jacobian, residuals_3d],
                    outputs=[jtj_out, grad_out],
                    block_dim=self.TILE_THREADS,
                    device=self.device,
                )

            def _block_thomas_solve(self, d_bar, l_bar, b_bar, n_super, x_bar, chol_ws, coupling_ws, fwd_ws) -> None:
                wp.launch_tiled(
                    _thomas_kernel,
                    dim=[self.n_trajectories],
                    inputs=[d_bar, l_bar, b_bar, n_super],
                    outputs=[x_bar, chol_ws, coupling_ws, fwd_ws],
                    block_dim=self.THOMAS_THREADS,
                    device=self.device,
                )

            def _spike_interior_solve(self, d_bar, l_bar, b_bar, istart, ilen, n_parts, y, u, v, chol, coup) -> None:
                wp.launch_tiled(
                    _spike_interior_kernel,
                    dim=[self.n_trajectories * n_parts],
                    inputs=[d_bar, l_bar, b_bar, istart, ilen, n_parts],
                    outputs=[y, u, v, chol, coup],
                    block_dim=self.THOMAS_THREADS,
                    device=self.device,
                )

            def _spike_schur_assemble(
                self, d_bar, l_bar, b_bar, istart, ilen, n_parts, y, u, v, d_sep, l_sep, b_sep
            ) -> None:
                wp.launch_tiled(
                    _spike_schur_kernel,
                    dim=[self.n_trajectories * (n_parts - 1)],
                    inputs=[d_bar, l_bar, b_bar, istart, ilen, n_parts, y, u, v],
                    outputs=[d_sep, l_sep, b_sep],
                    block_dim=self.THOMAS_THREADS,
                    device=self.device,
                )

            def _invert_diag_blocks(self, values, diag_idx, identity, minv) -> None:
                wp.launch_tiled(
                    _inv_diag_kernel,
                    dim=[self.n_batch],
                    inputs=[values, diag_idx, identity],
                    outputs=[minv],
                    block_dim=self.TILE_THREADS,
                    device=self.device,
                )

        _Specialized.__name__ = f"IKTraj_{n_dofs}x{n_residuals}_k{kb}"
        _Specialized._integrate_dq_dof = staticmethod(base_cls._integrate_dq_dof)
        _Specialized._compute_motion_subspace_2d = staticmethod(base_cls._compute_motion_subspace_2d)
        _Specialized._fk_two_pass = staticmethod(base_cls._fk_two_pass)
        return _Specialized
