# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Implicit-function adjoint of the differentiable Steel PGS solve.

At a converged PGS point each row satisfies its complementarity condition. Linearizing those conditions on the
active set gives ``M dlambda = -B (drhs + dC lambda) + ds``, and the adjoint is one dense solve ``M^T u = lambda_bar``
per world. Row equations:

- inactive normal or limit row (``lambda = 0``), zero-radius friction, unsupported kinds: ``dlambda_i = 0``
- active normal or limit row: ``dw_i + eps_i dlambda_i = 0``
- smooth normal row: ``dlambda_i = ds_i``
- sticking friction pair: ``dw_t + eps dlambda_t = 0`` on both rows
- sliding friction pair, ``lambda_t = mu lambda_n d``: ``d^T dlambda_t = mu dlambda_n`` and
  ``(alpha + eps) p^T dlambda_t + p^T dw_t = 0`` with ``p`` perpendicular to ``d`` and ``w_t = -alpha lambda_t``

where ``w = rhs + C lambda`` and ``eps_i = regularization * C_ii`` acts as a small compliance on the equality rows.
"""

from __future__ import annotations

import warp as wp

from .kernels import PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION, PGS_CONSTRAINT_TYPE_JOINT_LIMIT

wp.set_module_options({"enable_backward": False})

ROW_FROZEN = wp.constant(0)
ROW_ACTIVE = wp.constant(1)
ROW_SMOOTH = wp.constant(2)
ROW_STICK = wp.constant(3)
ROW_SLIDE_RADIAL = wp.constant(4)
ROW_SLIDE_TANGENT = wp.constant(5)
ROW_UNSUPPORTED = wp.constant(6)


@wp.kernel
def pgs_implicit_adjoint(
    row_count: wp.array[int],
    row_type: wp.array2d[int],
    row_parent: wp.array2d[int],
    row_mu: wp.array2d[float],
    C: wp.array3d[float],
    rhs: wp.array2d[float],
    lam: wp.array2d[float],
    lam_bar: wp.array2d[float],
    smooth_normal: int,
    regularization: float,
    slide_tolerance: float,
    M: wp.array3d[float],
    row_class: wp.array2d[int],
    perm: wp.array2d[int],
    u: wp.array2d[float],
    # outputs, accumulated
    C_bar: wp.array3d[float],
    rhs_bar: wp.array2d[float],
    smooth_bar: wp.array2d[float],
    status: wp.array[int],
):
    """One world: build M, factor it with partial pivoting, solve M^T u = lambda_bar, scatter -B^T u."""
    world = wp.tid()
    m = row_count[world]
    if m == 0:
        return
    unsupported = int(0)
    for i in range(m):
        for j in range(m):
            M[world, i, j] = 0.0
        row_class[world, i] = ROW_FROZEN
    for i in range(m):
        kind = row_type[world, i]
        if kind == PGS_CONSTRAINT_TYPE_CONTACT or kind == PGS_CONSTRAINT_TYPE_JOINT_LIMIT:
            if kind == PGS_CONSTRAINT_TYPE_CONTACT and smooth_normal != 0:
                M[world, i, i] = 1.0
                row_class[world, i] = ROW_SMOOTH
            elif lam[world, i] > 0.0:
                for j in range(m):
                    M[world, i, j] = C[world, i, j]
                M[world, i, i] = M[world, i, i] + regularization * C[world, i, i]
                row_class[world, i] = ROW_ACTIVE
            else:
                M[world, i, i] = 1.0
        elif kind == PGS_CONSTRAINT_TYPE_FRICTION:
            parent = row_parent[world, i]
            if i == parent + 1 and i + 1 < m:
                sib = i + 1
                load = lam[world, parent]
                mu = row_mu[world, i]
                radius = mu * load
                t0 = lam[world, i]
                t1 = lam[world, sib]
                rho = wp.sqrt(t0 * t0 + t1 * t1)
                if radius <= 0.0 or load <= 0.0:
                    M[world, i, i] = 1.0
                    M[world, sib, sib] = 1.0
                elif rho < radius * (1.0 - slide_tolerance):
                    for j in range(m):
                        M[world, i, j] = C[world, i, j]
                        M[world, sib, j] = C[world, sib, j]
                    M[world, i, i] = M[world, i, i] + regularization * C[world, i, i]
                    M[world, sib, sib] = M[world, sib, sib] + regularization * C[world, sib, sib]
                    row_class[world, i] = ROW_STICK
                    row_class[world, sib] = ROW_STICK
                else:
                    d0 = t0 / rho
                    d1 = t1 / rho
                    p0 = -d1
                    p1 = d0
                    w0 = rhs[world, i]
                    w1 = rhs[world, sib]
                    for j in range(m):
                        w0 += C[world, i, j] * lam[world, j]
                        w1 += C[world, sib, j] * lam[world, j]
                    alpha = wp.max(-(d0 * w0 + d1 * w1), 0.0) / rho
                    alpha += regularization * 0.5 * (C[world, i, i] + C[world, sib, sib])
                    M[world, i, i] = d0
                    M[world, i, sib] = d1
                    M[world, i, parent] = M[world, i, parent] - mu
                    for j in range(m):
                        M[world, sib, j] = p0 * C[world, i, j] + p1 * C[world, sib, j]
                    M[world, sib, i] = M[world, sib, i] + alpha * p0
                    M[world, sib, sib] = M[world, sib, sib] + alpha * p1
                    row_class[world, i] = ROW_SLIDE_RADIAL
                    row_class[world, sib] = ROW_SLIDE_TANGENT
        else:
            M[world, i, i] = 1.0
            row_class[world, i] = ROW_UNSUPPORTED
            unsupported += 1

    # In-place LU with partial pivoting: row k of the factors is original row perm[k].
    for i in range(m):
        perm[world, i] = i
    singular = int(0)
    for k in range(m):
        pivot = k
        best = wp.abs(M[world, k, k])
        for i in range(k + 1, m):
            value = wp.abs(M[world, i, k])
            if value > best:
                best = value
                pivot = i
        if pivot != k:
            for j in range(m):
                tmp = M[world, k, j]
                M[world, k, j] = M[world, pivot, j]
                M[world, pivot, j] = tmp
            tmp_p = perm[world, k]
            perm[world, k] = perm[world, pivot]
            perm[world, pivot] = tmp_p
        diag = M[world, k, k]
        if diag == 0.0:
            singular += 1
            M[world, k, k] = 1.0
            diag = 1.0
        inv = 1.0 / diag
        for i in range(k + 1, m):
            factor = M[world, i, k] * inv
            M[world, i, k] = factor
            if factor != 0.0:
                for j in range(k + 1, m):
                    M[world, i, j] = M[world, i, j] - factor * M[world, k, j]

    # M^T u = lam_bar with P M = L U: U^T y = lam_bar, then L^T z = y in place, then u[perm[k]] = z[k].
    for k in range(m):
        value = lam_bar[world, k]
        for i in range(k):
            value -= M[world, i, k] * u[world, i]
        u[world, k] = value / M[world, k, k]
    for kk in range(m):
        k = m - 1 - kk
        value = u[world, k]
        for i in range(k + 1, m):
            value -= M[world, i, k] * u[world, i]
        u[world, k] = value
    # M's first column is free after the solve; stage z there to undo the permutation.
    for k in range(m):
        M[world, perm[world, k], 0] = u[world, k]
    for k in range(m):
        u[world, k] = M[world, k, 0]

    # b_bar = -B^T u; rhs_bar += b_bar; C_bar += b_bar lambda^T; smooth_bar += u on smooth rows.
    for i in range(m):
        cls = row_class[world, i]
        b = float(0.0)
        if cls == ROW_ACTIVE or cls == ROW_STICK:
            b = -u[world, i]
        elif cls == ROW_SLIDE_RADIAL:
            t0 = lam[world, i]
            t1 = lam[world, i + 1]
            rho = wp.sqrt(t0 * t0 + t1 * t1)
            b = (t1 / rho) * u[world, i + 1]
        elif cls == ROW_SLIDE_TANGENT:
            t0 = lam[world, i - 1]
            t1 = lam[world, i]
            rho = wp.sqrt(t0 * t0 + t1 * t1)
            b = -(t0 / rho) * u[world, i]
        elif cls == ROW_SMOOTH:
            smooth_bar[world, i] = smooth_bar[world, i] + u[world, i]
        if b != 0.0:
            rhs_bar[world, i] = rhs_bar[world, i] + b
            for j in range(m):
                C_bar[world, i, j] = C_bar[world, i, j] + b * lam[world, j]
    if unsupported > 0 or singular > 0:
        wp.atomic_add(status, 0, unsupported)
        wp.atomic_add(status, 1, singular)


@wp.kernel
def cholesky_adjoint_phi(L: wp.array3d[float], L_bar: wp.array3d[float], n: int, phi: wp.array4d[float]):
    """phi = lower(L^T L_bar) with a halved diagonal, one entry per thread."""
    g, i, j = wp.tid()
    value = float(0.0)
    if j <= i:
        for k in range(i, n):
            value += L[g, k, i] * L_bar[g, k, j]
        if i == j:
            value *= 0.5
    phi[g, 0, i, j] = value


@wp.kernel
def cholesky_adjoint_left(L: wp.array3d[float], n: int, phi: wp.array4d[float]):
    """phi[1] = L^-T phi[0], one column per thread."""
    g, j = wp.tid()
    for ii in range(n):
        i = n - 1 - ii
        value = phi[g, 0, i, j]
        for k in range(i + 1, n):
            value -= L[g, k, i] * phi[g, 1, k, j]
        phi[g, 1, i, j] = value / L[g, i, i]


@wp.kernel
def cholesky_adjoint_right(L: wp.array3d[float], n: int, phi: wp.array4d[float]):
    """phi[0] = phi[1] L^-1, one row per thread."""
    g, i = wp.tid()
    for jj in range(n):
        j = n - 1 - jj
        value = phi[g, 1, i, j]
        for k in range(j + 1, n):
            value -= phi[g, 0, i, k] * L[g, k, j]
        phi[g, 0, i, j] = value / L[g, j, j]


@wp.kernel
def cholesky_adjoint_scatter(phi: wp.array4d[float], H_bar: wp.array3d[float], L_bar: wp.array3d[float]):
    """Accumulate the symmetric H_bar on the lower triangle the factorization reads, and consume L_bar."""
    g, i, j = wp.tid()
    if j < i:
        H_bar[g, i, j] = H_bar[g, i, j] + phi[g, 0, i, j] + phi[g, 0, j, i]
        L_bar[g, i, j] = 0.0
    elif j == i:
        H_bar[g, i, i] = H_bar[g, i, i] + phi[g, 0, i, i]
        L_bar[g, i, i] = 0.0
