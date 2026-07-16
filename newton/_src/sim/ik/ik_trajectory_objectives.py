# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Temporal objectives for trajectory inverse kinematics.

Temporal objectives couple joint coordinates of consecutive frames of a
trajectory (velocity/acceleration/jerk smoothness, velocity limits,
per-frame joint-space references). They contribute residual rows like any
:class:`~newton.ik.IKObjective`, but their Jacobians are diagonal
finite-difference stencils in the frame dimension. Instead of writing into
the dense per-frame Jacobian, they report per-DoF stencil coefficients that
:class:`~newton.ik.IKSolverTrajectory` accumulates directly into the
block-banded Gauss-Newton system.
"""

from __future__ import annotations

import math

import numpy as np
import warp as wp

from ..articulation import jcalc_motion_subspace
from ..enums import JointType
from ..model import Model
from .ik_common import IKJacobianType, eval_fk_batched
from .ik_objectives import IKObjective


@wp.func
def _quat_log(q: wp.quat) -> wp.vec3:
    """Axis-angle vector of a unit quaternion (2 * log map)."""
    v_norm = wp.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2])
    angle = 2.0 * wp.atan2(v_norm, q[3])
    if v_norm > 1.0e-8:
        return wp.vec3(q[0] / v_norm * angle, q[1] / v_norm * angle, q[2] / v_norm * angle)
    return wp.vec3(2.0 * q[0], 2.0 * q[1], 2.0 * q[2])


@wp.func
def _accumulate_joint_tangent_diff(
    joint_idx: int,
    q_a: wp.array[wp.float32],
    q_b: wp.array[wp.float32],
    scale: float,
    out_offset: int,
    joint_type: wp.array[wp.int32],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_dof_dim: wp.array2d[wp.int32],
    out_row: wp.array[wp.float32],
):
    """Accumulate ``scale * (q_b ⊖ q_a)`` for one joint into ``out_row``.

    Angular components are the axis-angle of the left-difference quaternion,
    matching the retraction used by the IK optimizers (``jcalc_integrate``).
    Free-joint linear components are the plain position difference, which is
    invariant to where the trajectory sits in the world; its coupling to the
    angular tangent coordinates (the ``w x p`` lever arm of the spatial
    velocity convention) is reported exactly through the temporal objectives'
    coefficient blocks.
    """
    t = joint_type[joint_idx]
    if t == JointType.FIXED:
        return

    coord0 = joint_q_start[joint_idx]
    dof0 = joint_qd_start[joint_idx]

    if t == JointType.BALL:
        qa = wp.quat(q_a[coord0 + 0], q_a[coord0 + 1], q_a[coord0 + 2], q_a[coord0 + 3])
        qb = wp.quat(q_b[coord0 + 0], q_b[coord0 + 1], q_b[coord0 + 2], q_b[coord0 + 3])
        if wp.dot(qa, qb) < 0.0:
            qb = -qb
        w = _quat_log(qb * wp.quat_inverse(qa))
        out_row[out_offset + dof0 + 0] += scale * w[0]
        out_row[out_offset + dof0 + 1] += scale * w[1]
        out_row[out_offset + dof0 + 2] += scale * w[2]
        return

    if t == JointType.FREE or t == JointType.DISTANCE:
        pa = wp.vec3(q_a[coord0 + 0], q_a[coord0 + 1], q_a[coord0 + 2])
        pb = wp.vec3(q_b[coord0 + 0], q_b[coord0 + 1], q_b[coord0 + 2])
        qa = wp.quat(q_a[coord0 + 3], q_a[coord0 + 4], q_a[coord0 + 5], q_a[coord0 + 6])
        qb = wp.quat(q_b[coord0 + 3], q_b[coord0 + 4], q_b[coord0 + 5], q_b[coord0 + 6])
        if wp.dot(qa, qb) < 0.0:
            qb = -qb
        w = _quat_log(qb * wp.quat_inverse(qa))
        dlin = pb - pa
        out_row[out_offset + dof0 + 0] += scale * dlin[0]
        out_row[out_offset + dof0 + 1] += scale * dlin[1]
        out_row[out_offset + dof0 + 2] += scale * dlin[2]
        out_row[out_offset + dof0 + 3] += scale * w[0]
        out_row[out_offset + dof0 + 4] += scale * w[1]
        out_row[out_offset + dof0 + 5] += scale * w[2]
        return

    # revolute / prismatic / D6: coordinates equal DoFs
    n_axes = joint_dof_dim[joint_idx, 0] + joint_dof_dim[joint_idx, 1]
    for i in range(n_axes):
        out_row[out_offset + dof0 + i] += scale * (q_b[coord0 + i] - q_a[coord0 + i])


@wp.kernel(enable_backward=False)
def _stencil_diff_residuals(
    joint_q: wp.array2d[wp.float32],  # (n_rows, n_coords)
    n_frames: int,
    width: int,
    diff_coeffs: wp.array[wp.float32],  # (width,)
    start_idx: int,
    joint_type: wp.array[wp.int32],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_dof_dim: wp.array2d[wp.int32],
    # outputs
    residuals: wp.array2d[wp.float32],  # (n_rows, n_residuals)
):
    row, joint_idx = wp.tid()
    t = row % n_frames
    if t + width >= n_frames:
        return

    for i in range(width):
        c = diff_coeffs[i]
        if c != 0.0:
            _accumulate_joint_tangent_diff(
                joint_idx,
                joint_q[row + i],
                joint_q[row + i + 1],
                c,
                start_idx,
                joint_type,
                joint_q_start,
                joint_qd_start,
                joint_dof_dim,
                residuals[row],
            )


@wp.kernel(enable_backward=False)
def _scale_residual_rows(
    n_frames: int,
    width: int,
    scale: float,
    dof_weights: wp.array[wp.float32],  # (n_dofs,)
    frame_weights: wp.array[wp.float32],  # (n_frames,) or empty
    start_idx: int,
    n_dofs: int,
    # outputs
    residuals: wp.array2d[wp.float32],
):
    row, dof = wp.tid()
    t = row % n_frames
    if t + width >= n_frames:
        return
    s = scale * dof_weights[dof]
    if frame_weights.shape[0] > 0:
        s = s * frame_weights[t]
    residuals[row, start_idx + dof] = residuals[row, start_idx + dof] * s


@wp.kernel(enable_backward=False)
def _stencil_diag_coeffs(
    n_frames: int,
    width: int,
    jac_coeffs: wp.array[wp.float32],  # (width + 1,)
    scale: float,
    dof_weights: wp.array[wp.float32],  # (n_dofs,)
    frame_weights: wp.array[wp.float32],  # (n_frames,) or empty
    # outputs
    coeffs: wp.array4d[wp.float32],  # (n_rows, width + 1, n_dofs, n_dofs), zeroed by caller
):
    row, dof = wp.tid()
    t = row % n_frames
    if t + width >= n_frames:
        return
    s = scale * dof_weights[dof]
    if frame_weights.shape[0] > 0:
        s = s * frame_weights[t]
    for j in range(width + 1):
        coeffs[row, j, dof, dof] = jac_coeffs[j] * s


@wp.func
def _skew_entry(p: wp.vec3, a: int, b: int) -> float:
    """Entry (a, b) of the cross-product matrix [p]x."""
    if a == 0:
        return wp.where(b == 1, -p[2], wp.where(b == 2, p[1], 0.0))
    if a == 1:
        return wp.where(b == 0, p[2], wp.where(b == 2, -p[0], 0.0))
    return wp.where(b == 0, -p[1], wp.where(b == 1, p[0], 0.0))


@wp.kernel(enable_backward=False)
def _stencil_free_lever_coeffs(
    joint_q: wp.array2d[wp.float32],  # (n_rows, n_coords)
    n_frames: int,
    width: int,
    jac_coeffs: wp.array[wp.float32],  # (width + 1,)
    scale: float,
    dof_weights: wp.array[wp.float32],  # (n_dofs,)
    frame_weights: wp.array[wp.float32],  # (n_frames,) or empty
    joint_type: wp.array[wp.int32],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    # outputs
    coeffs: wp.array4d[wp.float32],  # (n_rows, width + 1, n_dofs, n_dofs)
):
    """Free-joint lever-arm coefficient blocks for position-difference stencils.

    The retraction moves a free joint's position by ``delta_lin + delta_ang x p``
    (spatial velocity convention), so the linear residual rows couple to the
    angular tangent coordinates: d r_lin / d delta_ang(t + j) = -c_j [p_{t+j}]x.
    """
    row, joint_idx = wp.tid()
    jt = joint_type[joint_idx]
    if jt != JointType.FREE and jt != JointType.DISTANCE:
        return
    t = row % n_frames
    if t + width >= n_frames:
        return

    coord0 = joint_q_start[joint_idx]
    dof0 = joint_qd_start[joint_idx]

    fw = float(1.0)
    if frame_weights.shape[0] > 0:
        fw = frame_weights[t]

    for j in range(width + 1):
        rj = row + j
        p = wp.vec3(joint_q[rj, coord0 + 0], joint_q[rj, coord0 + 1], joint_q[rj, coord0 + 2])
        c = jac_coeffs[j]
        for a in range(3):
            s_row = scale * dof_weights[dof0 + a] * fw
            for b in range(3):
                coeffs[row, j, dof0 + a, dof0 + 3 + b] = -c * s_row * _skew_entry(p, a, b)


@wp.kernel(enable_backward=False)
def _reference_residuals(
    joint_q: wp.array2d[wp.float32],  # (n_rows, n_coords)
    reference_q: wp.array2d[wp.float32],  # (n_rows, n_coords)
    start_idx: int,
    joint_type: wp.array[wp.int32],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_dof_dim: wp.array2d[wp.int32],
    # outputs
    residuals: wp.array2d[wp.float32],
):
    row, joint_idx = wp.tid()
    _accumulate_joint_tangent_diff(
        joint_idx,
        reference_q[row],
        joint_q[row],
        1.0,
        start_idx,
        joint_type,
        joint_q_start,
        joint_qd_start,
        joint_dof_dim,
        residuals[row],
    )


@wp.kernel(enable_backward=False)
def _velocity_scratch(
    joint_q: wp.array2d[wp.float32],  # (n_rows, n_coords)
    n_frames: int,
    inv_dt: float,
    joint_type: wp.array[wp.int32],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_dof_dim: wp.array2d[wp.int32],
    # outputs
    velocity: wp.array2d[wp.float32],  # (n_rows, n_dofs)
):
    row, joint_idx = wp.tid()
    t = row % n_frames
    if t + 1 >= n_frames:
        return
    _accumulate_joint_tangent_diff(
        joint_idx,
        joint_q[row],
        joint_q[row + 1],
        inv_dt,
        0,
        joint_type,
        joint_q_start,
        joint_qd_start,
        joint_dof_dim,
        velocity[row],
    )


@wp.kernel(enable_backward=False)
def _velocity_limit_residuals(
    velocity: wp.array2d[wp.float32],  # (n_rows, n_dofs)
    velocity_limits: wp.array[wp.float32],  # (n_dofs,)
    n_frames: int,
    weight: float,
    start_idx: int,
    # outputs
    residuals: wp.array2d[wp.float32],
):
    row, dof = wp.tid()
    t = row % n_frames
    if t + 1 >= n_frames:
        return
    v = velocity[row, dof]
    viol = wp.abs(v) - velocity_limits[dof]
    residuals[row, start_idx + dof] = weight * wp.max(viol, 0.0)


@wp.kernel(enable_backward=False)
def _velocity_limit_coeffs(
    velocity: wp.array2d[wp.float32],  # (n_rows, n_dofs)
    velocity_limits: wp.array[wp.float32],  # (n_dofs,)
    n_frames: int,
    weight: float,
    inv_dt: float,
    # outputs
    coeffs: wp.array4d[wp.float32],  # (n_rows, 2, n_dofs, n_dofs), zeroed by caller
):
    row, dof = wp.tid()
    t = row % n_frames
    if t + 1 >= n_frames:
        return
    v = velocity[row, dof]
    if wp.abs(v) > velocity_limits[dof]:
        grad = wp.where(v > 0.0, weight * inv_dt, -weight * inv_dt)
        coeffs[row, 0, dof, dof] = -grad
        coeffs[row, 1, dof, dof] = grad


@wp.kernel(enable_backward=False)
def _velocity_limit_free_lever_coeffs(
    joint_q: wp.array2d[wp.float32],  # (n_rows, n_coords)
    velocity: wp.array2d[wp.float32],  # (n_rows, n_dofs)
    velocity_limits: wp.array[wp.float32],  # (n_dofs,)
    n_frames: int,
    weight: float,
    inv_dt: float,
    joint_type: wp.array[wp.int32],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    # outputs
    coeffs: wp.array4d[wp.float32],  # (n_rows, 2, n_dofs, n_dofs)
):
    row, joint_idx = wp.tid()
    jt = joint_type[joint_idx]
    if jt != JointType.FREE and jt != JointType.DISTANCE:
        return
    t = row % n_frames
    if t + 1 >= n_frames:
        return

    coord0 = joint_q_start[joint_idx]
    dof0 = joint_qd_start[joint_idx]

    for j in range(2):
        rj = row + j
        p = wp.vec3(joint_q[rj, coord0 + 0], joint_q[rj, coord0 + 1], joint_q[rj, coord0 + 2])
        # stencil coefficients of the hinge residual: (-grad, +grad) per row
        c_sign = wp.where(j == 0, -1.0, 1.0)
        for a in range(3):
            v = velocity[row, dof0 + a]
            grad = float(0.0)
            if wp.abs(v) > velocity_limits[dof0 + a]:
                grad = wp.where(v > 0.0, weight * inv_dt, -weight * inv_dt)
            c = c_sign * grad
            for b in range(3):
                coeffs[row, j, dof0 + a, dof0 + 3 + b] = -c * _skew_entry(p, a, b)


class IKObjectiveTemporal(IKObjective):
    """Base class for objectives whose residuals couple consecutive frames.

    A temporal objective contributes ``joint_dof_count`` residual rows per
    frame. Residual rows of frame ``t`` may depend on the tangent coordinates
    of frames ``t .. t + stencil_width()``, with Jacobian blocks reported
    through :meth:`compute_coeffs` as per-frame-offset coefficient blocks.
    This bounded-stencil structure is what allows
    :class:`~newton.ik.IKSolverTrajectory` to accumulate the objective's
    Gauss-Newton contribution directly into the block-banded system without
    materializing a global cross-frame Jacobian.

    Temporal objectives are only supported by
    :class:`~newton.ik.IKSolverTrajectory`; the per-frame IK solvers reject
    them.
    """

    def __init__(self, model: Model) -> None:
        super().__init__()
        self.model = model
        self.n_dofs = model.joint_dof_count
        self.n_frames = None
        self.n_trajectories = None
        self.coeffs = None

    def set_trajectory_layout(self, n_frames: int, n_trajectories: int) -> None:
        """Register the frame layout of the owning trajectory solver.

        Args:
            n_frames: Number of frames per trajectory.
            n_trajectories: Number of trajectories optimized together.
        """
        self.n_frames = n_frames
        self.n_trajectories = n_trajectories

    def _require_trajectory_layout(self) -> None:
        if self.n_frames is None:
            raise RuntimeError(f"{type(self).__name__} requires a trajectory layout; use it with IKSolverTrajectory")

    def stencil_width(self) -> int:
        """Return the largest forward frame offset referenced by a residual row."""
        raise NotImplementedError

    def residual_dim(self) -> int:
        """Return one residual row per joint DoF."""
        return self.n_dofs

    def init_buffers(self, model: Model, jacobian_mode: IKJacobianType) -> None:
        """Allocate the stencil-coefficient buffer.

        Args:
            model: Shared articulation model.
            jacobian_mode: Ignored; temporal objectives always evaluate their
                stencil coefficients analytically.
        """
        self._require_batch_layout()
        self._require_trajectory_layout()
        self.coeffs = wp.zeros(
            (self.n_batch, self.stencil_width() + 1, self.n_dofs, self.n_dofs),
            dtype=wp.float32,
            device=self.device,
        )

    def supports_analytic(self) -> bool:
        """Return ``True``; stencil coefficients are always analytic."""
        return True

    def compute_jacobian_autodiff(self, tape, model, jacobian, start_idx, dq_dof) -> None:
        """No-op: temporal Jacobians are reported via :meth:`compute_coeffs`."""

    def compute_jacobian_analytic(self, body_q, joint_q, model, jacobian, joint_S_s, start_idx) -> None:
        """No-op: temporal Jacobians are reported via :meth:`compute_coeffs`."""

    def compute_coeffs(self, joint_q: wp.array2d[wp.float32]) -> None:
        """Write the stencil-coefficient blocks at the given configuration.

        After this call, ``self.coeffs[row, j, c, a]`` must equal the partial
        derivative of residual row ``(row, c)`` with respect to tangent
        coordinate ``a`` of frame ``row + j``. Most joints only populate the
        diagonal ``c == a``; free-joint linear rows additionally carry the
        exact ``[p]x`` lever-arm coupling to the angular tangent coordinates.

        Args:
            joint_q: Batched joint coordinates, shape [n_rows, joint_coord_count].
        """
        raise NotImplementedError


class IKObjectiveSmoothness(IKObjectiveTemporal):
    """Penalize finite-difference velocity, acceleration, or jerk.

    Residual row ``t`` of DoF ``d`` is the order-``derivative`` forward
    difference of the joint tangent coordinates, scaled by
    ``weight / dt**derivative``. Quaternion joints (ball, free) use
    tangent-space differences consistent with the solver's retraction.

    Args:
        model: Shared articulation model.
        derivative: Finite-difference order: 1 (velocity), 2 (acceleration),
            or 3 (jerk).
        dt: Time step between consecutive frames [s].
        weight: Scalar multiplier applied to the residual rows.
        dof_weights: Optional per-DoF multipliers, shape [joint_dof_count].
    """

    def __init__(
        self,
        model: Model,
        derivative: int = 1,
        dt: float = 1.0,
        weight: float = 1.0,
        dof_weights: wp.array[wp.float32] | None = None,
    ) -> None:
        super().__init__(model)
        if derivative not in (1, 2, 3):
            raise ValueError("derivative must be 1, 2, or 3")
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.derivative = derivative
        self.dt = dt
        self.weight = weight
        self.dof_weights = dof_weights

        # Jacobian stencil: alternating binomial coefficients, e.g. (1, -2, 1)
        jac = np.array([(-1.0) ** (derivative - j) * float(math.comb(derivative, j)) for j in range(derivative + 1)])
        # first-difference form: r = sum_i s_i * (u_{t+i+1} - u_{t+i})
        diff = np.array([jac[i + 1 :].sum() for i in range(derivative)])
        self._jac_coeffs_np = jac.astype(np.float32)
        self._diff_coeffs_np = diff.astype(np.float32)
        self._jac_coeffs = None
        self._diff_coeffs = None
        self._dof_weights = None
        self._frame_weights = None

    def stencil_width(self) -> int:
        """Return the finite-difference order."""
        return self.derivative

    def init_buffers(self, model: Model, jacobian_mode: IKJacobianType) -> None:
        super().init_buffers(model, jacobian_mode)
        self._jac_coeffs = wp.array(self._jac_coeffs_np, dtype=wp.float32, device=self.device)
        self._diff_coeffs = wp.array(self._diff_coeffs_np, dtype=wp.float32, device=self.device)
        if self.dof_weights is not None:
            self._dof_weights = self.dof_weights
        else:
            self._dof_weights = wp.ones(self.n_dofs, dtype=wp.float32, device=self.device)
        self._frame_weights = wp.empty(0, dtype=wp.float32, device=self.device)

    @property
    def _scale(self) -> float:
        return self.weight / self.dt**self.derivative

    def compute_residuals(
        self,
        body_q: wp.array2d[wp.transform],
        joint_q: wp.array2d[wp.float32],
        model: Model,
        residuals: wp.array2d[wp.float32],
        start_idx: int,
        problem_idx: wp.array[wp.int32],
    ) -> None:
        """Write weighted finite-difference residuals into the global buffer.

        Args:
            body_q: Batched body transforms. Present for interface
                compatibility and not used by this objective.
            joint_q: Batched joint coordinates, shape [n_rows, joint_coord_count].
            model: Shared articulation model.
            residuals: Global residual buffer, shape [n_rows, total_residual_count].
            start_idx: First residual row reserved for this objective.
            problem_idx: Present for interface compatibility; frames are
                addressed through the trajectory layout instead.
        """
        self._require_trajectory_layout()
        n_rows = joint_q.shape[0]
        wp.launch(
            _stencil_diff_residuals,
            dim=[n_rows, model.joint_count],
            inputs=[
                joint_q,
                self.n_frames,
                self.derivative,
                self._diff_coeffs,
                start_idx,
                model.joint_type,
                model.joint_q_start,
                model.joint_qd_start,
                model.joint_dof_dim,
            ],
            outputs=[residuals],
            device=self.device,
        )
        wp.launch(
            _scale_residual_rows,
            dim=[n_rows, self.n_dofs],
            inputs=[
                self.n_frames,
                self.derivative,
                self._scale,
                self._dof_weights,
                self._frame_weights,
                start_idx,
                self.n_dofs,
            ],
            outputs=[residuals],
            device=self.device,
        )

    def compute_coeffs(self, joint_q: wp.array2d[wp.float32]) -> None:
        self.coeffs.zero_()
        wp.launch(
            _stencil_diag_coeffs,
            dim=[self.n_batch, self.n_dofs],
            inputs=[
                self.n_frames,
                self.derivative,
                self._jac_coeffs,
                self._scale,
                self._dof_weights,
                self._frame_weights,
            ],
            outputs=[self.coeffs],
            device=self.device,
        )
        wp.launch(
            _stencil_free_lever_coeffs,
            dim=[self.n_batch, self.model.joint_count],
            inputs=[
                joint_q,
                self.n_frames,
                self.derivative,
                self._jac_coeffs,
                self._scale,
                self._dof_weights,
                self._frame_weights,
                self.model.joint_type,
                self.model.joint_q_start,
                self.model.joint_qd_start,
            ],
            outputs=[self.coeffs],
            device=self.device,
        )


class IKObjectiveVelocityLimit(IKObjectiveTemporal):
    """Penalize finite-difference joint velocities exceeding per-DoF limits.

    Residual row ``t`` of DoF ``d`` is ``weight * max(0, |v| - limit_d)``
    with ``v = (u_{t+1} - u_t) / dt`` in tangent space.

    Args:
        model: Shared articulation model.
        velocity_limits: Per-DoF velocity limits [m/s or rad/s], shape
            [joint_dof_count]. Defaults to ``model.joint_velocity_limit``.
        dt: Time step between consecutive frames [s].
        weight: Scalar multiplier applied to the residual rows.
    """

    def __init__(
        self,
        model: Model,
        velocity_limits: wp.array[wp.float32] | None = None,
        dt: float = 1.0,
        weight: float = 1.0,
    ) -> None:
        super().__init__(model)
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if velocity_limits is None:
            velocity_limits = model.joint_velocity_limit
        if velocity_limits is None:
            raise ValueError("velocity_limits must be provided when the model defines none")
        self.velocity_limits = velocity_limits
        self.dt = dt
        self.weight = weight
        self._velocity = None

    def stencil_width(self) -> int:
        """Return 1: each residual row couples frames ``t`` and ``t + 1``."""
        return 1

    def init_buffers(self, model: Model, jacobian_mode: IKJacobianType) -> None:
        super().init_buffers(model, jacobian_mode)
        self._velocity = wp.zeros((self.n_batch, self.n_dofs), dtype=wp.float32, device=self.device)

    def _compute_velocity(self, joint_q: wp.array2d[wp.float32], model: Model) -> None:
        self._velocity.zero_()
        wp.launch(
            _velocity_scratch,
            dim=[joint_q.shape[0], model.joint_count],
            inputs=[
                joint_q,
                self.n_frames,
                1.0 / self.dt,
                model.joint_type,
                model.joint_q_start,
                model.joint_qd_start,
                model.joint_dof_dim,
            ],
            outputs=[self._velocity],
            device=self.device,
        )

    def compute_residuals(
        self,
        body_q: wp.array2d[wp.transform],
        joint_q: wp.array2d[wp.float32],
        model: Model,
        residuals: wp.array2d[wp.float32],
        start_idx: int,
        problem_idx: wp.array[wp.int32],
    ) -> None:
        """Write weighted velocity-limit violations into the global buffer.

        Args:
            body_q: Batched body transforms. Present for interface
                compatibility and not used by this objective.
            joint_q: Batched joint coordinates, shape [n_rows, joint_coord_count].
            model: Shared articulation model.
            residuals: Global residual buffer, shape [n_rows, total_residual_count].
            start_idx: First residual row reserved for this objective.
            problem_idx: Present for interface compatibility; frames are
                addressed through the trajectory layout instead.
        """
        self._require_trajectory_layout()
        self._compute_velocity(joint_q, model)
        wp.launch(
            _velocity_limit_residuals,
            dim=[joint_q.shape[0], self.n_dofs],
            inputs=[
                self._velocity,
                self.velocity_limits,
                self.n_frames,
                self.weight,
                start_idx,
            ],
            outputs=[residuals],
            device=self.device,
        )

    def compute_coeffs(self, joint_q: wp.array2d[wp.float32]) -> None:
        self._compute_velocity(joint_q, self.model)
        self.coeffs.zero_()
        wp.launch(
            _velocity_limit_coeffs,
            dim=[self.n_batch, self.n_dofs],
            inputs=[
                self._velocity,
                self.velocity_limits,
                self.n_frames,
                self.weight,
                1.0 / self.dt,
            ],
            outputs=[self.coeffs],
            device=self.device,
        )
        wp.launch(
            _velocity_limit_free_lever_coeffs,
            dim=[self.n_batch, self.model.joint_count],
            inputs=[
                joint_q,
                self._velocity,
                self.velocity_limits,
                self.n_frames,
                self.weight,
                1.0 / self.dt,
                self.model.joint_type,
                self.model.joint_q_start,
                self.model.joint_qd_start,
            ],
            outputs=[self.coeffs],
            device=self.device,
        )


class IKObjectiveJointReference(IKObjectiveTemporal):
    """Pull each frame toward a reference joint configuration.

    Residual row ``t`` of DoF ``d`` is
    ``weight * frame_weights[t] * (u_t ⊖ u_ref,t)`` in tangent space. Use a
    small uniform weight as a posture regularizer, or per-frame weights to
    softly anchor selected frames (e.g. the start of the trajectory) to
    known configurations.

    Args:
        model: Shared articulation model.
        reference_q: Reference joint coordinates [m or rad], shape
            [n_trajectories * n_frames, joint_coord_count].
        weight: Scalar multiplier applied to the residual rows.
        frame_weights: Optional per-frame multipliers, shape [n_frames],
            shared by all trajectories.
        dof_weights: Optional per-DoF multipliers, shape [joint_dof_count].
    """

    def __init__(
        self,
        model: Model,
        reference_q: wp.array2d[wp.float32],
        weight: float = 1.0,
        frame_weights: wp.array[wp.float32] | None = None,
        dof_weights: wp.array[wp.float32] | None = None,
    ) -> None:
        super().__init__(model)
        self.reference_q = reference_q
        self.weight = weight
        self.frame_weights = frame_weights
        self.dof_weights = dof_weights
        self._dof_weights = None
        self._frame_weights = None
        self._unit_jac = None

    def stencil_width(self) -> int:
        """Return 0: each residual row only involves its own frame."""
        return 0

    def init_buffers(self, model: Model, jacobian_mode: IKJacobianType) -> None:
        super().init_buffers(model, jacobian_mode)
        self._unit_jac = wp.array([1.0], dtype=wp.float32, device=self.device)
        if self.reference_q.shape != (self.n_batch, model.joint_coord_count):
            raise ValueError(
                f"reference_q has shape {self.reference_q.shape}, expected {(self.n_batch, model.joint_coord_count)}"
            )
        if self.dof_weights is not None:
            self._dof_weights = self.dof_weights
        else:
            self._dof_weights = wp.ones(self.n_dofs, dtype=wp.float32, device=self.device)
        if self.frame_weights is not None:
            self._frame_weights = self.frame_weights
        else:
            self._frame_weights = wp.empty(0, dtype=wp.float32, device=self.device)

    def compute_residuals(
        self,
        body_q: wp.array2d[wp.transform],
        joint_q: wp.array2d[wp.float32],
        model: Model,
        residuals: wp.array2d[wp.float32],
        start_idx: int,
        problem_idx: wp.array[wp.int32],
    ) -> None:
        """Write weighted reference deviations into the global buffer.

        Args:
            body_q: Batched body transforms. Present for interface
                compatibility and not used by this objective.
            joint_q: Batched joint coordinates, shape [n_rows, joint_coord_count].
            model: Shared articulation model.
            residuals: Global residual buffer, shape [n_rows, total_residual_count].
            start_idx: First residual row reserved for this objective.
            problem_idx: Present for interface compatibility; frames are
                addressed through the trajectory layout instead.
        """
        self._require_trajectory_layout()
        n_rows = joint_q.shape[0]
        wp.launch(
            _reference_residuals,
            dim=[n_rows, model.joint_count],
            inputs=[
                joint_q,
                self.reference_q,
                start_idx,
                model.joint_type,
                model.joint_q_start,
                model.joint_qd_start,
                model.joint_dof_dim,
            ],
            outputs=[residuals],
            device=self.device,
        )
        wp.launch(
            _scale_residual_rows,
            dim=[n_rows, self.n_dofs],
            inputs=[
                self.n_frames,
                0,
                self.weight,
                self._dof_weights,
                self._frame_weights,
                start_idx,
                self.n_dofs,
            ],
            outputs=[residuals],
            device=self.device,
        )

    def compute_coeffs(self, joint_q: wp.array2d[wp.float32]) -> None:
        self.coeffs.zero_()
        wp.launch(
            _stencil_diag_coeffs,
            dim=[self.n_batch, self.n_dofs],
            inputs=[
                self.n_frames,
                0,
                self._unit_jac,
                self.weight,
                self._dof_weights,
                self._frame_weights,
            ],
            outputs=[self.coeffs],
            device=self.device,
        )
        wp.launch(
            _stencil_free_lever_coeffs,
            dim=[self.n_batch, self.model.joint_count],
            inputs=[
                joint_q,
                self.n_frames,
                0,
                self._unit_jac,
                self.weight,
                self._dof_weights,
                self._frame_weights,
                self.model.joint_type,
                self.model.joint_q_start,
                self.model.joint_qd_start,
            ],
            outputs=[self.coeffs],
            device=self.device,
        )


@wp.kernel(enable_backward=False)
def _motion_subspace_rows(
    joint_type: wp.array[wp.int32],
    joint_parent: wp.array[wp.int32],
    joint_child: wp.array[wp.int32],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    joint_q: wp.array2d[wp.float32],  # (n_rows, n_coords)
    joint_axis: wp.array[wp.vec3],
    joint_dof_dim: wp.array2d[wp.int32],
    body_q: wp.array2d[wp.transform],  # (n_rows, n_bodies)
    body_com: wp.array[wp.vec3],
    joint_X_p: wp.array[wp.transform],
    # outputs
    joint_S_s: wp.array2d[wp.spatial_vector],  # (n_rows, n_dofs)
):
    """Batched world-frame motion subspace, layout ``(v_origin[0:3], omega[3:6])``."""
    row, joint_idx = wp.tid()

    t = joint_type[joint_idx]
    parent = joint_parent[joint_idx]
    child = joint_child[joint_idx]
    q_start = joint_q_start[joint_idx]
    qd_start = joint_qd_start[joint_idx]

    X_pj = joint_X_p[joint_idx]
    X_wpj = X_pj
    if parent >= 0:
        X_wpj = body_q[row, parent] * X_pj

    lin_axis_count = joint_dof_dim[joint_idx, 0]
    ang_axis_count = joint_dof_dim[joint_idx, 1]

    joint_q_1d = joint_q[row]
    S_s_out = joint_S_s[row]

    if t == JointType.FREE or t == JointType.DISTANCE:
        jcalc_motion_subspace(
            t,
            joint_axis,
            joint_q_1d,
            lin_axis_count,
            ang_axis_count,
            X_wpj,
            body_q[row, child],
            body_com[child],
            q_start,
            qd_start,
            S_s_out,
        )
        # jcalc anchors the free-joint angular columns at the child COM, but
        # the solver's retraction (jcalc_integrate) rotates about the joint
        # frame's origin — the world origin for a root joint, the parent
        # anchor otherwise (cf. _stencil_free_lever_coeffs). Rewrite the
        # linear parts so downstream point Jacobians dp = v + omega x p
        # match the tangent convention.
        anchor = wp.transform_get_translation(X_wpj)
        for k in range(3):
            S = S_s_out[qd_start + 3 + k]
            omega = wp.vec3(S[3], S[4], S[5])
            v = wp.cross(anchor, omega)
            S_s_out[qd_start + 3 + k] = wp.spatial_vector(v[0], v[1], v[2], S[3], S[4], S[5])
    else:
        jcalc_motion_subspace(
            t,
            joint_axis,
            joint_q_1d,
            lin_axis_count,
            ang_axis_count,
            X_wpj,
            wp.transform_identity(),
            wp.vec3(),
            q_start,
            qd_start,
            S_s_out,
        )


@wp.kernel(enable_backward=False)
def _world_com_scratch(
    body_q: wp.array2d[wp.transform],  # (n_rows, n_bodies)
    body_com: wp.array[wp.vec3],
    # outputs
    com_w: wp.array2d[wp.vec3],  # (n_rows, n_bodies)
):
    row, body = wp.tid()
    com_w[row, body] = wp.transform_point(body_q[row, body], body_com[body])


@wp.kernel(enable_backward=False)
def _gravity_torque_u(
    com_w: wp.array2d[wp.vec3],  # (n_rows, n_bodies)
    body_mass: wp.array[wp.float32],
    joint_S_s: wp.array2d[wp.spatial_vector],  # (n_rows, n_dofs)
    subtree_body: wp.array2d[wp.uint8],  # (n_dofs, n_bodies)
    n_bodies: int,
    # outputs
    u: wp.array2d[wp.vec3],  # (n_rows, n_dofs)
):
    """Subtree gravity moment direction ``u_d = M_d v_d + omega_d x P_d``.

    ``M_d`` / ``P_d`` are the total mass and mass-weighted world COM of the
    bodies moved by DoF ``d``, and ``(v_d, omega_d)`` its world motion
    subspace at the origin, so ``tau_g[d] = -g . u_d``.
    """
    row, dof = wp.tid()
    S = joint_S_s[row, dof]
    v = wp.vec3(S[0], S[1], S[2])
    w = wp.vec3(S[3], S[4], S[5])
    m_sum = float(0.0)
    p_sum = wp.vec3(0.0)
    for b in range(n_bodies):
        if subtree_body[dof, b] != 0:
            m = body_mass[b]
            m_sum += m
            p_sum += m * com_w[row, b]
    u[row, dof] = m_sum * v + wp.cross(w, p_sum)


@wp.kernel(enable_backward=False)
def _gravity_torque_residuals(
    u: wp.array2d[wp.vec3],  # (n_rows, n_dofs)
    gravity: wp.vec3,
    dof_scale: wp.array[wp.float32],  # (n_dofs,)
    start_idx: int,
    # outputs
    residuals: wp.array2d[wp.float32],
):
    row, dof = wp.tid()
    s = dof_scale[dof]
    if s == 0.0:
        return
    residuals[row, start_idx + dof] = -s * wp.dot(gravity, u[row, dof])


@wp.kernel(enable_backward=False)
def _gravity_torque_coeffs(
    u: wp.array2d[wp.vec3],  # (n_rows, n_dofs)
    joint_S_s: wp.array2d[wp.spatial_vector],  # (n_rows, n_dofs)
    dof_ancestor: wp.array2d[wp.uint8],  # (n_dofs, n_dofs); [d, a] = a ancestor-or-self of d
    gravity: wp.vec3,
    dof_scale: wp.array[wp.float32],  # (n_dofs,)
    # outputs
    coeffs: wp.array4d[wp.float32],  # (n_rows, 1, n_dofs, n_dofs), zeroed by caller
):
    """Gravity-torque Jacobian ``d tau_g[c] / d u_a = -g . (omega_ancestor x u_descendant)``.

    Rigid transport gives ``d u_descendant / d q_ancestor = omega_ancestor x u_descendant`` for any
    ancestor/descendant DoF pair on one chain (exact for revolute, prismatic,
    and free-joint tangents; Gauss-Newton-approximate between rotational DoFs
    that share one ball or D6 joint). Unrelated DoF pairs are zero.
    """
    row, c, a = wp.tid()
    s = dof_scale[c]
    if s == 0.0:
        return
    if dof_ancestor[c, a] != 0:
        S_ancestor = joint_S_s[row, a]
        u_descendant = u[row, c]
    elif dof_ancestor[a, c] != 0:
        S_ancestor = joint_S_s[row, c]
        u_descendant = u[row, a]
    else:
        return
    w_ancestor = wp.vec3(S_ancestor[3], S_ancestor[4], S_ancestor[5])
    coeffs[row, 0, c, a] = -s * wp.dot(gravity, wp.cross(w_ancestor, u_descendant))


@wp.func
def _link_point(body_q: wp.array2d[wp.transform], row: int, link: int, offset: wp.vec3) -> wp.vec3:
    tf = body_q[row, link]
    rot = wp.quat(tf[3], tf[4], tf[5], tf[6])
    return wp.vec3(tf[0], tf[1], tf[2]) + wp.quat_rotate(rot, offset)


@wp.func
def _link_axis(body_q: wp.array2d[wp.transform], row: int, link: int, axis: wp.vec3) -> wp.vec3:
    tf = body_q[row, link]
    rot = wp.quat(tf[3], tf[4], tf[5], tf[6])
    return wp.quat_rotate(rot, axis)


@wp.kernel(enable_backward=False)
def _apparent_gravity_scratch(
    body_q: wp.array2d[wp.transform],  # (n_rows, n_bodies)
    n_frames: int,
    link_index: int,
    link_offset: wp.vec3,
    tangent_u: wp.vec3,  # link-local plate tangent axes
    tangent_v: wp.vec3,
    gravity: wp.vec3,
    inv_dt2: float,
    # outputs
    f_app: wp.array[wp.vec3],  # (n_rows,) apparent specific force a_ee - g
    t_u: wp.array[wp.vec3],  # (n_rows,) world tangent axes at frame t + 1
    t_v: wp.array[wp.vec3],
):
    row = wp.tid()
    t = row % n_frames
    if t + 2 >= n_frames:
        return
    p0 = _link_point(body_q, row, link_index, link_offset)
    p1 = _link_point(body_q, row + 1, link_index, link_offset)
    p2 = _link_point(body_q, row + 2, link_index, link_offset)
    a_ee = (p0 - 2.0 * p1 + p2) * inv_dt2
    f_app[row] = a_ee - gravity
    t_u[row] = _link_axis(body_q, row + 1, link_index, tangent_u)
    t_v[row] = _link_axis(body_q, row + 1, link_index, tangent_v)


@wp.kernel(enable_backward=False)
def _apparent_gravity_residuals(
    f_app: wp.array[wp.vec3],  # (n_rows,)
    t_u: wp.array[wp.vec3],
    t_v: wp.array[wp.vec3],
    n_frames: int,
    weight: float,
    start_idx: int,
    # outputs
    residuals: wp.array2d[wp.float32],
):
    row = wp.tid()
    t = row % n_frames
    if t + 2 >= n_frames:
        return
    residuals[row, start_idx + 0] = weight * wp.dot(t_u[row], f_app[row])
    residuals[row, start_idx + 1] = weight * wp.dot(t_v[row], f_app[row])


@wp.kernel(enable_backward=False)
def _apparent_gravity_coeffs(
    body_q: wp.array2d[wp.transform],  # (n_rows, n_bodies)
    joint_S_s: wp.array2d[wp.spatial_vector],  # (n_rows, n_dofs)
    f_app: wp.array[wp.vec3],
    t_u: wp.array[wp.vec3],
    t_v: wp.array[wp.vec3],
    affects_dof: wp.array[wp.uint8],  # (n_dofs,)
    n_frames: int,
    link_index: int,
    link_offset: wp.vec3,
    weight: float,
    inv_dt2: float,
    # outputs
    coeffs: wp.array4d[wp.float32],  # (n_rows, 3, n_dofs, n_dofs), zeroed by caller
):
    """Waiter-objective Jacobian blocks.

    Row k of frame t couples to tangent coordinate ``a`` of frames
    ``t + j`` through the acceleration stencil ``s = (1, -2, 1)/dt^2``
    projected on the plate tangent axis, plus (at ``j = 1``, where the
    tangent axes are evaluated) the plate-tilt term
    ``(omega_a x t_k) . f_app``.
    """
    row, dof = wp.tid()
    t = row % n_frames
    if t + 2 >= n_frames:
        return
    if affects_dof[dof] == 0:
        return

    for j in range(3):
        rj = row + j
        s_j = wp.where(j == 1, -2.0, 1.0) * inv_dt2
        p = _link_point(body_q, rj, link_index, link_offset)
        S = joint_S_s[rj, dof]
        v = wp.vec3(S[0], S[1], S[2])
        omega = wp.vec3(S[3], S[4], S[5])
        dp = v + wp.cross(omega, p)
        c_u = s_j * wp.dot(t_u[row], dp)
        c_v = s_j * wp.dot(t_v[row], dp)
        if j == 1:
            c_u += wp.dot(wp.cross(omega, t_u[row]), f_app[row])
            c_v += wp.dot(wp.cross(omega, t_v[row]), f_app[row])
        coeffs[row, j, 0, dof] = weight * c_u
        coeffs[row, j, 1, dof] = weight * c_v


class IKObjectiveApparentGravity(IKObjectiveTemporal):
    """Keep apparent gravity aligned with a carried surface's normal.

    The "waiter" objective: for a plate (tray, glass, ball support) rigidly
    carried by ``link_index``, an object resting on it stays put when the
    apparent specific force ``f = a_ee - g`` [m/s²] felt in the plate frame
    points along the plate normal. Residual rows 0 and 1 of frame ``t`` are
    the two plate-tangential components ``weight * t_k . f`` with the
    end-effector acceleration ``a_ee`` taken as the forward second
    difference of frames ``t .. t + 2`` and the tangent axes ``t_k``
    evaluated at frame ``t + 1``. The remaining ``joint_dof_count - 2``
    rows stay zero (the banded assembly requires temporal residual blocks
    of that size).

    Minimizing it makes the solver both smooth the carried point's
    acceleration and bank the plate into the residual acceleration — the
    same strategy a waiter uses with a loaded tray. It penalizes sliding
    (tangential force) only; it does not enforce a unilateral contact
    (``f . n > 0``) or a friction cone.

    Args:
        model: Shared articulation model.
        link_index: Link carrying the surface.
        link_offset: Surface reference point in the link frame [m].
        plate_axis: Surface normal in the link frame (need not be unit).
        dt: Time step between consecutive frames [s].
        weight: Scalar multiplier applied to the residual rows.
    """

    def __init__(
        self,
        model: Model,
        link_index: int,
        link_offset: wp.vec3,
        plate_axis: wp.vec3 = wp.vec3(0.0, 0.0, 1.0),
        dt: float = 1.0,
        weight: float = 1.0,
    ) -> None:
        super().__init__(model)
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        self.link_index = link_index
        self.link_offset = link_offset
        self.dt = dt
        self.weight = weight

        axis = np.array([plate_axis[0], plate_axis[1], plate_axis[2]], dtype=np.float64)
        norm = np.linalg.norm(axis)
        if norm < 1e-9:
            raise ValueError("plate_axis must be non-zero")
        axis /= norm
        helper = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(axis, helper)
        u /= np.linalg.norm(u)
        v = np.cross(axis, u)
        self._tangent_u = wp.vec3(*u)
        self._tangent_v = wp.vec3(*v)

        self._affects_dof = None
        self._gravity = None
        self._f_app = None
        self._t_u = None
        self._t_v = None
        self._joint_S_s = None
        self._body_q_fk = None
        self._body_qd_fk = None
        self._qd_zero = None

    def stencil_width(self) -> int:
        """Return 2: the acceleration stencil couples frames ``t .. t + 2``."""
        return 2

    def init_buffers(self, model: Model, jacobian_mode: IKJacobianType) -> None:
        super().init_buffers(model, jacobian_mode)
        n_bodies = model.body_count

        # per-DoF mask of ancestors of the carrying link
        joint_qd_start_np = model.joint_qd_start.numpy()
        joint_parent_np = model.joint_parent.numpy()
        joint_child_np = model.joint_child.numpy()
        body_to_joint = np.full(n_bodies, -1, dtype=np.int32)
        for j in range(model.joint_count):
            child = joint_child_np[j]
            if child != -1:
                body_to_joint[child] = j
        affects = np.zeros(self.n_dofs, dtype=np.uint8)
        body = self.link_index
        while body != -1:
            j = body_to_joint[body]
            if j == -1:
                break
            affects[joint_qd_start_np[j] : joint_qd_start_np[j + 1]] = 1
            body = joint_parent_np[j]
        self._affects_dof = wp.array(affects, dtype=wp.uint8, device=self.device)

        self._gravity = wp.vec3(*model.gravity.numpy()[0])
        self._f_app = wp.zeros(self.n_batch, dtype=wp.vec3, device=self.device)
        self._t_u = wp.zeros(self.n_batch, dtype=wp.vec3, device=self.device)
        self._t_v = wp.zeros(self.n_batch, dtype=wp.vec3, device=self.device)
        self._joint_S_s = wp.zeros((self.n_batch, self.n_dofs), dtype=wp.spatial_vector, device=self.device)
        self._body_q_fk = wp.zeros((self.n_batch, n_bodies), dtype=wp.transform, device=self.device)
        self._body_qd_fk = wp.zeros((self.n_batch, n_bodies), dtype=wp.spatial_vector, device=self.device)
        self._qd_zero = wp.zeros((self.n_batch, self.n_dofs), dtype=wp.float32, device=self.device)

    def _compute_scratch(self, body_q: wp.array2d[wp.transform], n_rows: int) -> None:
        wp.launch(
            _apparent_gravity_scratch,
            dim=[n_rows],
            inputs=[
                body_q,
                self.n_frames,
                self.link_index,
                self.link_offset,
                self._tangent_u,
                self._tangent_v,
                self._gravity,
                1.0 / self.dt**2,
            ],
            outputs=[self._f_app, self._t_u, self._t_v],
            device=self.device,
        )

    def compute_residuals(
        self,
        body_q: wp.array2d[wp.transform],
        joint_q: wp.array2d[wp.float32],
        model: Model,
        residuals: wp.array2d[wp.float32],
        start_idx: int,
        problem_idx: wp.array[wp.int32],
    ) -> None:
        """Write weighted tangential apparent-gravity components into the buffer.

        Args:
            body_q: Batched body transforms, shape [n_rows, body_count].
            joint_q: Batched joint coordinates, shape [n_rows, joint_coord_count].
            model: Shared articulation model.
            residuals: Global residual buffer, shape [n_rows, total_residual_count].
            start_idx: First residual row reserved for this objective.
            problem_idx: Present for interface compatibility; frames are
                addressed through the trajectory layout instead.
        """
        self._require_trajectory_layout()
        n_rows = joint_q.shape[0]
        self._compute_scratch(body_q, n_rows)
        wp.launch(
            _apparent_gravity_residuals,
            dim=[n_rows],
            inputs=[self._f_app, self._t_u, self._t_v, self.n_frames, self.weight, start_idx],
            outputs=[residuals],
            device=self.device,
        )

    def compute_coeffs(self, joint_q: wp.array2d[wp.float32]) -> None:
        eval_fk_batched(self.model, joint_q, self._qd_zero, self._body_q_fk, self._body_qd_fk)
        model = self.model
        wp.launch(
            _motion_subspace_rows,
            dim=[self.n_batch, model.joint_count],
            inputs=[
                model.joint_type,
                model.joint_parent,
                model.joint_child,
                model.joint_q_start,
                model.joint_qd_start,
                joint_q,
                model.joint_axis,
                model.joint_dof_dim,
                self._body_q_fk,
                model.body_com,
                model.joint_X_p,
            ],
            outputs=[self._joint_S_s],
            device=self.device,
        )
        self._compute_scratch(self._body_q_fk, self.n_batch)
        self.coeffs.zero_()
        wp.launch(
            _apparent_gravity_coeffs,
            dim=[self.n_batch, self.n_dofs],
            inputs=[
                self._body_q_fk,
                self._joint_S_s,
                self._f_app,
                self._t_u,
                self._t_v,
                self._affects_dof,
                self.n_frames,
                self.link_index,
                self.link_offset,
                self.weight,
                1.0 / self.dt**2,
            ],
            outputs=[self.coeffs],
            device=self.device,
        )


class IKObjectiveGravityTorque(IKObjectiveTemporal):
    """Penalize the static gravity-compensation torque at every frame.

    Residual row ``t`` of DoF ``d`` is
    ``weight * dof_weights[d] * tau_g,d(q_t)`` [N or N·m], where
    ``tau_g(q) = dU/dq`` is the joint force that holds the articulation
    static under gravity (the ``gravity_force`` convention of
    :func:`~newton.eval_inverse_dynamics`). Minimizing it steers redundant
    DoFs toward gravity-friendly postures — e.g. an elbow hanging below the
    wrist instead of winging sideways — which is most visible with a heavy
    payload attached to the end effector.

    DoFs of free and distance joints are unactuated and always contribute
    zero residual rows; their coefficient columns (how base motion changes
    the actuated torques) are still reported exactly. Gravity is read from
    world 0 of the model.

    Args:
        model: Shared articulation model.
        weight: Scalar multiplier applied to the residual rows.
        dof_weights: Optional per-DoF multipliers, shape [joint_dof_count].
            Read once when the owning solver initializes the objective.
    """

    def __init__(
        self,
        model: Model,
        weight: float = 1.0,
        dof_weights: wp.array[wp.float32] | None = None,
    ) -> None:
        super().__init__(model)
        self.weight = weight
        self.dof_weights = dof_weights
        self._dof_scale = None
        self._subtree_body = None
        self._dof_ancestor = None
        self._gravity = None
        self._joint_S_s = None
        self._com_w = None
        self._u = None
        self._body_q_fk = None
        self._body_qd_fk = None
        self._qd_zero = None

    def stencil_width(self) -> int:
        """Return 0: each residual row only involves its own frame."""
        return 0

    def init_buffers(self, model: Model, jacobian_mode: IKJacobianType) -> None:
        super().init_buffers(model, jacobian_mode)
        n_bodies = model.body_count
        n_joints = model.joint_count

        joint_qd_start_np = model.joint_qd_start.numpy()
        joint_parent_np = model.joint_parent.numpy()
        joint_child_np = model.joint_child.numpy()
        joint_type_np = model.joint_type.numpy()

        body_to_joint = np.full(n_bodies, -1, dtype=np.int32)
        for j in range(n_joints):
            child = joint_child_np[j]
            if child != -1:
                body_to_joint[child] = j

        dof_to_joint = np.full(self.n_dofs, -1, dtype=np.int32)
        for j in range(n_joints):
            dof_to_joint[joint_qd_start_np[j] : joint_qd_start_np[j + 1]] = j

        # subtree_body[d, b] = 1 iff DoF d's joint lies on the chain from the
        # root to body b (i.e. moving d moves b)
        subtree_body = np.zeros((self.n_dofs, n_bodies), dtype=np.uint8)
        joint_ancestor = np.zeros((n_joints, n_joints), dtype=np.uint8)
        for b in range(n_bodies):
            body = b
            while body != -1:
                j = body_to_joint[body]
                if j == -1:
                    break
                subtree_body[joint_qd_start_np[j] : joint_qd_start_np[j + 1], b] = 1
                if body_to_joint[b] != -1:
                    joint_ancestor[body_to_joint[b], j] = 1
                body = joint_parent_np[j]

        # dof_ancestor[c, a] = 1 iff a is ancestor-or-self of c; DoFs of one
        # joint are ordered (earlier axes carry later ones, cf. jcalc FK)
        dof_ancestor = np.zeros((self.n_dofs, self.n_dofs), dtype=np.uint8)
        for c in range(self.n_dofs):
            jc = dof_to_joint[c]
            if jc == -1:
                continue
            for a in range(self.n_dofs):
                ja = dof_to_joint[a]
                if ja == -1:
                    continue
                if jc == ja:
                    dof_ancestor[c, a] = 1 if a <= c else 0
                elif joint_ancestor[jc, ja]:
                    dof_ancestor[c, a] = 1

        dof_scale = np.full(self.n_dofs, self.weight, dtype=np.float32)
        for j in range(n_joints):
            if joint_type_np[j] in (int(JointType.FREE), int(JointType.DISTANCE)):
                dof_scale[joint_qd_start_np[j] : joint_qd_start_np[j + 1]] = 0.0
        if self.dof_weights is not None:
            dof_scale *= self.dof_weights.numpy().astype(np.float32)

        self._subtree_body = wp.array(subtree_body, dtype=wp.uint8, device=self.device)
        self._dof_ancestor = wp.array(dof_ancestor, dtype=wp.uint8, device=self.device)
        self._dof_scale = wp.array(dof_scale, dtype=wp.float32, device=self.device)
        self._gravity = wp.vec3(*model.gravity.numpy()[0])

        self._joint_S_s = wp.zeros((self.n_batch, self.n_dofs), dtype=wp.spatial_vector, device=self.device)
        self._com_w = wp.zeros((self.n_batch, n_bodies), dtype=wp.vec3, device=self.device)
        self._u = wp.zeros((self.n_batch, self.n_dofs), dtype=wp.vec3, device=self.device)
        self._body_q_fk = wp.zeros((self.n_batch, n_bodies), dtype=wp.transform, device=self.device)
        self._body_qd_fk = wp.zeros((self.n_batch, n_bodies), dtype=wp.spatial_vector, device=self.device)
        self._qd_zero = wp.zeros((self.n_batch, self.n_dofs), dtype=wp.float32, device=self.device)

    def _compute_u(self, body_q: wp.array2d[wp.transform], joint_q: wp.array2d[wp.float32]) -> None:
        n_rows = joint_q.shape[0]
        model = self.model
        wp.launch(
            _world_com_scratch,
            dim=[n_rows, model.body_count],
            inputs=[body_q, model.body_com],
            outputs=[self._com_w],
            device=self.device,
        )
        wp.launch(
            _motion_subspace_rows,
            dim=[n_rows, model.joint_count],
            inputs=[
                model.joint_type,
                model.joint_parent,
                model.joint_child,
                model.joint_q_start,
                model.joint_qd_start,
                joint_q,
                model.joint_axis,
                model.joint_dof_dim,
                body_q,
                model.body_com,
                model.joint_X_p,
            ],
            outputs=[self._joint_S_s],
            device=self.device,
        )
        wp.launch(
            _gravity_torque_u,
            dim=[n_rows, self.n_dofs],
            inputs=[
                self._com_w,
                model.body_mass,
                self._joint_S_s,
                self._subtree_body,
                model.body_count,
            ],
            outputs=[self._u],
            device=self.device,
        )

    def compute_residuals(
        self,
        body_q: wp.array2d[wp.transform],
        joint_q: wp.array2d[wp.float32],
        model: Model,
        residuals: wp.array2d[wp.float32],
        start_idx: int,
        problem_idx: wp.array[wp.int32],
    ) -> None:
        """Write weighted gravity-compensation torques into the global buffer.

        Args:
            body_q: Batched body transforms, shape [n_rows, body_count].
            joint_q: Batched joint coordinates, shape [n_rows, joint_coord_count].
            model: Shared articulation model.
            residuals: Global residual buffer, shape [n_rows, total_residual_count].
            start_idx: First residual row reserved for this objective.
            problem_idx: Present for interface compatibility; frames are
                addressed through the trajectory layout instead.
        """
        self._require_trajectory_layout()
        self._compute_u(body_q, joint_q)
        wp.launch(
            _gravity_torque_residuals,
            dim=[joint_q.shape[0], self.n_dofs],
            inputs=[self._u, self._gravity, self._dof_scale, start_idx],
            outputs=[residuals],
            device=self.device,
        )

    def compute_coeffs(self, joint_q: wp.array2d[wp.float32]) -> None:
        eval_fk_batched(self.model, joint_q, self._qd_zero, self._body_q_fk, self._body_qd_fk)
        self._compute_u(self._body_q_fk, joint_q)
        self.coeffs.zero_()
        wp.launch(
            _gravity_torque_coeffs,
            dim=[self.n_batch, self.n_dofs, self.n_dofs],
            inputs=[
                self._u,
                self._joint_S_s,
                self._dof_ancestor,
                self._gravity,
                self._dof_scale,
            ],
            outputs=[self.coeffs],
            device=self.device,
        )
