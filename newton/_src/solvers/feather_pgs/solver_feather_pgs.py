# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

import warp as wp

from ...core.types import override
from ...sim import Contacts, Control, Model, State, eval_fk
from ..semi_implicit.kernels_contact import (
    eval_body_contact,
    eval_particle_body_contact_forces,
    eval_particle_contact_forces,
)
from ..semi_implicit.kernels_muscle import (
    eval_muscle_forces,
)
from ..semi_implicit.kernels_particle import (
    eval_bending_forces,
    eval_spring_forces,
    eval_tetrahedra_forces,
    eval_triangle_forces,
)
from ..solver import SolverBase
from .kernels import (
    accumulate_contact_velocity,
    apply_augmented_joint_tau,
    apply_augmented_mass_diagonal,
    apply_hinv_Jt_multi_rhs,
    apply_hinv_Jt_multi_rhs_tiled,
    build_augmented_joint_rows,
    build_contact_rows_normal,
    build_joint_target_rows,
    build_mass_update_mask,
    clamp_contact_counts,
    clamp_joint_tau,
    compute_com_transforms,
    compute_contact_bias,
    compute_spatial_inertia,
    compute_velocity_predictor,
    copy_int_array_masked,
    detect_limit_count_changes,
    eval_crba,
    eval_dense_cholesky_batched,
    eval_dense_solve_batched,
    eval_rigid_fk,
    eval_rigid_id,
    eval_rigid_jacobian,
    eval_rigid_mass,
    eval_rigid_tau,
    form_contact_matrix,
    form_contact_matrix_tiled,
    integrate_generalized_joints,
    pgs_solve_contacts,
    prepare_impulses,
    update_qdd_from_velocity,
    TILE_DOF,
    TILE_CONSTRAINTS,
    TILE_THREADS,
)


class SolverFeatherPGS(SolverBase):
    """A semi-implicit integrator using symplectic Euler that operates
    on reduced (also called generalized) coordinates to simulate articulated rigid body dynamics
    based on Featherstone's composite rigid body algorithm (CRBA).

    See: Featherstone, Roy. Rigid Body Dynamics Algorithms. Springer US, 2014.

    Instead of maximal coordinates :attr:`~newton.State.body_q` (rigid body positions) and :attr:`~newton.State.body_qd`
    (rigid body velocities) as is the case in :class:`~newton.solvers.SolverSemiImplicit` and :class:`~newton.solvers.SolverXPBD`,
    :class:`~newton.solvers.SolverFeatherPGS` uses :attr:`~newton.State.joint_q` and :attr:`~newton.State.joint_qd` to represent
    the positions and velocities of joints without allowing any redundant degrees of freedom.

    After constructing :class:`~newton.Model` and :class:`~newton.State` objects this time-integrator
    may be used to advance the simulation state forward in time.

    Note:
        Unlike :class:`~newton.solvers.SolverSemiImplicit` and :class:`~newton.solvers.SolverXPBD`, :class:`~newton.solvers.SolverFeatherPGS`
        does not simulate rigid bodies with nonzero mass as floating bodies if they are not connected through any joints.
        Floating-base systems require an explicit free joint with which the body is connected to the world,
        see :meth:`newton.ModelBuilder.add_joint_free`.

    Semi-implicit time integration is a variational integrator that
    preserves energy, however it not unconditionally stable, and requires a time-step
    small enough to support the required stiffness and damping forces.

    See: https://en.wikipedia.org/wiki/Semi-implicit_Euler_method

    This solver uses the routines from :class:`~newton.solvers.SolverSemiImplicit` to simulate particles, cloth, and soft bodies.

    Example
    -------

    .. code-block:: python

        solver = newton.solvers.SolverFeatherPGS(model)

        # simulation loop
        for i in range(100):
            solver.step(state_in, state_out, control, contacts, dt)
            state_in, state_out = state_out, state_in

    """

    def __init__(
        self,
        model: Model,
        angular_damping: float = 0.05,
        update_mass_matrix_interval: int = 1,
        friction_smoothing: float = 1.0,
        enable_contact_friction: bool = True,
        pgs_iterations: int = 12,
        pgs_beta: float = 0.2,
        pgs_cfm: float = 1.0e-6,
        pgs_omega: float = 1.0,
        pgs_max_constraints: int = 32,
        pgs_warmstart: bool = False,
        pgs_use_joint_targets: bool = False,
        pgs_joint_target_mode: Optional[str] = None,
        pgs_joint_beta: Optional[float] = None,
        pgs_joint_cfm: Optional[float] = None,
        enable_timers: bool = False,
        use_tiled_contact_build: bool = True,
    ):
        """
        Args:
            model (Model): the model to be simulated.
            angular_damping (float, optional): Angular damping factor. Defaults to 0.05.
            update_mass_matrix_interval (int, optional): How often to update the mass matrix (every n-th time the :meth:`step` function gets called). Defaults to 1.
            friction_smoothing (float, optional): The delta value for the Huber norm (see :func:`warp.math.norm_huber`) used for the friction velocity normalization. Defaults to 1.0.
            enable_contact_friction (bool, optional): Enables Coulomb friction contacts inside the PGS solve. Defaults to True.
            pgs_iterations (int, optional): Number of Gauss-Seidel iterations to apply per frame. Defaults to 12.
            pgs_beta (float, optional): ERP style position correction factor. Defaults to 0.2.
            pgs_cfm (float, optional): Compliance/regularization added to the Delassus diagonal. Defaults to 1.0e-6.
            pgs_omega (float, optional): Successive over-relaxation factor for the PGS sweep. Defaults to 1.0.
            pgs_max_constraints (int, optional): Maximum number of contact constraints stored per articulation. Defaults to 32.
            pgs_warmstart (bool, optional): Re-use impulses from the previous frame when contacts persist. Defaults to False.
            pgs_use_joint_targets (bool, optional): Whether to include joint drive targets as PGS equality constraints. Defaults to False.
            pgs_joint_target_mode (str, optional): Override for how joint targets are enforced (\"off\", \"pgs\", or \"augmented\"). Defaults to deriving from ``pgs_use_joint_targets``.
            pgs_joint_beta (float, optional): ERP override for joint-target constraints (defaults to auto-mapped from gains when None).
            pgs_joint_cfm (float, optional): CFM override for joint-target constraints (defaults to auto-mapped from gains when None).
            enable_timers (bool, optional): Enable NVTX profiling ranges for solver sub-sections. Defaults to False.
            use_tiled_contact_build (bool, optional): Enable use of fast tiled kernels for building constraint matrix. Defaults to True.
        """
        super().__init__(model)

        self.angular_damping = angular_damping
        self.update_mass_matrix_interval = update_mass_matrix_interval
        self.friction_smoothing = friction_smoothing
        self.enable_contact_friction = enable_contact_friction
        self.pgs_iterations = pgs_iterations
        self.pgs_beta = pgs_beta
        self.pgs_cfm = pgs_cfm
        self.pgs_omega = pgs_omega
        self.pgs_max_constraints = pgs_max_constraints
        if pgs_joint_target_mode is None:
            pgs_joint_target_mode = "pgs" if pgs_use_joint_targets else "off"
        if pgs_joint_target_mode not in ("off", "pgs", "augmented"):
            raise ValueError(f"Invalid joint target mode '{pgs_joint_target_mode}'")

        self.pgs_warmstart = pgs_warmstart
        self.pgs_joint_target_mode = pgs_joint_target_mode
        self.pgs_use_joint_targets = pgs_joint_target_mode == "pgs"
        self.pgs_use_augmented_targets = pgs_joint_target_mode == "augmented"
        self.pgs_joint_beta = pgs_joint_beta
        self.pgs_joint_cfm = pgs_joint_cfm
        self.enable_timers = enable_timers
        self.use_tiled_contact_build = use_tiled_contact_build

        self._step = 0
        self._force_mass_update = False
        self._last_step_dt = None

        self.compute_articulation_indices(model)
        # Validate tile-based path constraints (if enabled)
        if self.use_tiled_contact_build and model.articulation_count:
            if self.articulation_max_dofs > int(TILE_DOF):
                raise ValueError(
                    f"articulation_max_dofs={self.articulation_max_dofs} exceeds TILE_DOF={int(TILE_DOF)} "
                    "for tiled contact system build. Increase TILE_DOF or disable use_tiled_contact_build."
                )
            if self.pgs_max_constraints > int(TILE_CONSTRAINTS):
                raise ValueError(
                    f"pgs_max_constraints={self.pgs_max_constraints} exceeds TILE_CONSTRAINTS={int(TILE_CONSTRAINTS)} "
                    "for tiled contact system build. Increase TILE_CONSTRAINTS or reduce pgs_max_constraints."
                )

        self.build_body_maps(model)
        self.allocate_model_aux_vars(model)
        if model.shape_material_mu is not None:
            self.shape_material_mu = model.shape_material_mu
        else:
            self.shape_material_mu = wp.zeros(
                (1,), dtype=wp.float32, device=model.device, requires_grad=model.requires_grad
            )

    def _timer(self, label: str):
        return wp.ScopedTimer(label, active=self.enable_timers, use_nvtx=True, synchronize=True)

    def compute_articulation_indices(self, model):
        # calculate total size and offsets of Jacobian and mass matrices for entire system
        if model.joint_count:
            self.J_size = 0
            self.M_size = wp.int64(0)
            self.H_size = 0

            articulation_J_start = []
            articulation_M_start = []
            articulation_H_start = []

            articulation_M_rows = []
            articulation_H_rows = []
            articulation_J_rows = []
            articulation_J_cols = []

            articulation_dof_start = []
            articulation_coord_start = []

            articulation_start = model.articulation_start.numpy()
            joint_q_start = model.joint_q_start.numpy()
            joint_qd_start = model.joint_qd_start.numpy()

            for i in range(model.articulation_count):
                first_joint = articulation_start[i]
                last_joint = articulation_start[i + 1]

                first_coord = joint_q_start[first_joint]

                first_dof = joint_qd_start[first_joint]
                last_dof = joint_qd_start[last_joint]

                joint_count = last_joint - first_joint
                dof_count = last_dof - first_dof

                articulation_J_start.append(self.J_size)
                articulation_M_start.append(int(self.M_size))
                articulation_H_start.append(self.H_size)
                articulation_dof_start.append(first_dof)
                articulation_coord_start.append(first_coord)

                # bit of data duplication here, but will leave it as such for clarity
                articulation_M_rows.append(joint_count * 6)
                articulation_H_rows.append(dof_count)
                articulation_J_rows.append(joint_count * 6)
                articulation_J_cols.append(dof_count)

                self.J_size += 6 * joint_count * dof_count
                self.M_size = wp.int64(self.M_size + wp.int64(joint_count * 36))
                self.H_size += dof_count * dof_count

            # matrix offsets for batched gemm
            self.articulation_J_start = wp.array(articulation_J_start, dtype=wp.int32, device=model.device)
            self.articulation_M_start = wp.array(articulation_M_start, dtype=wp.int32, device=model.device)
            self.articulation_H_start = wp.array(articulation_H_start, dtype=wp.int32, device=model.device)

            self.articulation_M_rows = wp.array(articulation_M_rows, dtype=wp.int32, device=model.device)
            self.articulation_H_rows = wp.array(articulation_H_rows, dtype=wp.int32, device=model.device)
            self.articulation_J_rows = wp.array(articulation_J_rows, dtype=wp.int32, device=model.device)
            self.articulation_J_cols = wp.array(articulation_J_cols, dtype=wp.int32, device=model.device)

            self.articulation_dof_start = wp.array(articulation_dof_start, dtype=wp.int32, device=model.device)
            self.articulation_coord_start = wp.array(articulation_coord_start, dtype=wp.int32, device=model.device)

            self.articulation_max_dofs = int(max(articulation_H_rows)) if articulation_H_rows else 0
            self.M_size = int(self.M_size)
        else:
            self.M_size = 0
            self.articulation_max_dofs = 0

    def build_body_maps(self, model):
        if not model.body_count or not model.articulation_count:
            self.body_to_joint = None
            self.body_to_articulation = None
            return

        joint_child = model.joint_child.numpy()
        articulation_start = model.articulation_start.numpy()

        body_to_joint = [-1] * model.body_count
        body_to_articulation = [-1] * model.body_count

        for articulation in range(model.articulation_count):
            joint_start = articulation_start[articulation]
            joint_end = articulation_start[articulation + 1]

            for joint_index in range(joint_start, joint_end):
                child = joint_child[joint_index]
                if child < 0:
                    continue

                body_to_joint[child] = joint_index
                body_to_articulation[child] = articulation

        device = model.device
        self.body_to_joint = wp.array(body_to_joint, dtype=wp.int32, device=device)
        self.body_to_articulation = wp.array(body_to_articulation, dtype=wp.int32, device=device)


    def allocate_model_aux_vars(self, model):
        # allocate mass, Jacobian matrices, and other auxiliary variables pertaining to the model
        if model.joint_count:
            # system matrices
            self.M_blocks = wp.zeros((self.M_size,), dtype=wp.float32, device=model.device, requires_grad=model.requires_grad)
            self.J = wp.zeros((self.J_size,), dtype=wp.float32, device=model.device, requires_grad=model.requires_grad)
            self.H = wp.empty((self.H_size,), dtype=wp.float32, device=model.device, requires_grad=model.requires_grad)

            # zero since only upper triangle is set which can trigger NaN detection
            self.L = wp.zeros_like(self.H)
            self.mass_update_mask = wp.zeros(
                (model.articulation_count,), dtype=wp.int32, device=model.device, requires_grad=model.requires_grad
            )
            self.limit_change_mask = wp.zeros(
                (model.articulation_count,), dtype=wp.int32, device=model.device, requires_grad=model.requires_grad
            )

        if model.body_count:
            self.body_I_m = wp.empty(
                (model.body_count,), dtype=wp.spatial_matrix, device=model.device, requires_grad=model.requires_grad
            )
            wp.launch(
                compute_spatial_inertia,
                model.body_count,
                inputs=[model.body_inertia, model.body_mass],
                outputs=[self.body_I_m],
                device=model.device,
            )
            self.body_X_com = wp.empty(
                (model.body_count,), dtype=wp.transform, device=model.device, requires_grad=model.requires_grad
            )
            wp.launch(
                compute_com_transforms,
                model.body_count,
                inputs=[model.body_com],
                outputs=[self.body_X_com],
                device=model.device,
            )

        self.allocate_pgs_buffers(model)
        self.allocate_augmented_joint_buffers(model)

    def allocate_pgs_buffers(self, model):
        if not model.articulation_count or not model.joint_count:
            return

        max_dofs = self.articulation_max_dofs
        if max_dofs == 0:
            return

        device = model.device
        requires_grad = model.requires_grad
        articulation_count = model.articulation_count
        constraint_capacity = self.pgs_max_constraints

        self.pgs_row_stride = constraint_capacity * max_dofs
        self.pgs_matrix_stride = constraint_capacity * constraint_capacity

        total_rows = articulation_count * self.pgs_row_stride
        total_constraints = articulation_count * constraint_capacity
        total_matrix = articulation_count * self.pgs_matrix_stride

        self.pgs_counts = wp.zeros(
            (articulation_count,), dtype=wp.int32, device=device, requires_grad=requires_grad
        )
        self.pgs_phi = wp.zeros(
            (total_constraints,), dtype=wp.float32, device=device, requires_grad=requires_grad
        )
        self.pgs_diag = wp.zeros(
            (total_constraints,), dtype=wp.float32, device=device, requires_grad=requires_grad
        )
        self.pgs_rhs = wp.zeros(
            (total_constraints,), dtype=wp.float32, device=device, requires_grad=requires_grad
        )
        self.pgs_impulses = wp.zeros(
            (total_constraints,), dtype=wp.float32, device=device, requires_grad=requires_grad
        )
        self.pgs_types = wp.zeros(
            (total_constraints,), dtype=wp.int32, device=device, requires_grad=requires_grad
        )
        self.pgs_row_beta = wp.zeros(
            (total_constraints,), dtype=wp.float32, device=device, requires_grad=requires_grad
        )
        self.pgs_row_cfm = wp.zeros(
            (total_constraints,), dtype=wp.float32, device=device, requires_grad=requires_grad
        )
        self.pgs_target_velocity = wp.zeros(
            (total_constraints,), dtype=wp.float32, device=device, requires_grad=requires_grad
        )
        self.pgs_parent = wp.full(
            (total_constraints,), -1, dtype=wp.int32, device=device, requires_grad=requires_grad
        )
        self.pgs_mu = wp.zeros(
            (total_constraints,), dtype=wp.float32, device=device, requires_grad=requires_grad
        )
        self.pgs_contact_matrix = wp.zeros(
            (total_matrix,), dtype=wp.float32, device=device, requires_grad=requires_grad
        )
        self.pgs_Jc = wp.zeros(
            (total_rows,), dtype=wp.float32, device=device, requires_grad=requires_grad
        )
        self.pgs_Y = wp.zeros(
            (total_rows,), dtype=wp.float32, device=device, requires_grad=requires_grad
        )
        self.pgs_v_hat = wp.zeros_like(model.joint_qd, requires_grad=requires_grad)
        self.pgs_v_out = wp.zeros_like(model.joint_qd, requires_grad=requires_grad)

    def allocate_augmented_joint_buffers(self, model):
        if not self.pgs_use_augmented_targets:
            return
        if not model.articulation_count or not model.joint_count:
            return

        max_dofs = self.articulation_max_dofs
        if max_dofs == 0:
            return

        device = model.device
        requires_grad = model.requires_grad
        articulation_count = model.articulation_count
        total_rows = articulation_count * max_dofs

        self.aug_row_counts = wp.zeros(
            (articulation_count,), dtype=wp.int32, device=device, requires_grad=requires_grad
        )
        self.aug_limit_counts = wp.zeros(
            (articulation_count,), dtype=wp.int32, device=device, requires_grad=requires_grad
        )
        self.aug_prev_limit_counts = wp.zeros_like(self.aug_limit_counts)
        self.limit_change_mask = wp.zeros_like(self.aug_limit_counts)
        self.aug_row_dof_index = wp.zeros(
            (total_rows,), dtype=wp.int32, device=device, requires_grad=requires_grad
        )
        self.aug_row_K = wp.zeros(
            (total_rows,), dtype=wp.float32, device=device, requires_grad=requires_grad
        )
        self.aug_row_u0 = wp.zeros(
            (total_rows,), dtype=wp.float32, device=device, requires_grad=requires_grad
        )

    def build_augmented_joint_targets(self, state_in: State, control: Control, dt: float):
        if not self.pgs_use_augmented_targets:
            return

        model = self.model
        if model.articulation_count == 0 or self.articulation_max_dofs == 0:
            return
        if dt <= 0.0:
            self.aug_row_counts.zero_()
            self.aug_limit_counts.zero_()
            return

        device = model.device

        self.aug_row_counts.zero_()
        self.aug_limit_counts.zero_()

        wp.launch(
            build_augmented_joint_rows,
            dim=model.articulation_count,
            inputs=[
                model.articulation_start,
                self.articulation_dof_start,
                self.articulation_H_rows,
                model.joint_type,
                model.joint_q_start,
                model.joint_qd_start,
                model.joint_dof_dim,
                model.joint_target_ke,
                model.joint_target_kd,
                state_in.joint_q,
                state_in.joint_qd,
                control.joint_target_pos,
                control.joint_target_vel,
                self.articulation_max_dofs,
                dt,
            ],
            outputs=[
                self.aug_row_counts,
                self.aug_row_dof_index,
                self.aug_row_K,
                self.aug_row_u0,
                self.aug_limit_counts,
            ],
            device=device,
        )

        wp.launch(
            detect_limit_count_changes,
            dim=model.articulation_count,
            inputs=[
                self.aug_limit_counts,
                self.aug_prev_limit_counts,
            ],
            outputs=[
                self.limit_change_mask,
            ],
            device=device,
        )

    def apply_augmented_joint_tau(self, state_in: State, state_aug: State, dt: float):
        if not self.pgs_use_augmented_targets:
            return
        if dt <= 0.0:
            return

        model = self.model
        if model.articulation_count == 0 or self.articulation_max_dofs == 0:
            return

        wp.launch(
            apply_augmented_joint_tau,
            dim=model.articulation_count,
            inputs=[
                self.articulation_max_dofs,
                self.aug_row_counts,
                self.aug_row_dof_index,
                self.aug_row_u0,
            ],
            outputs=[state_aug.joint_tau],
            device=model.device,
        )

    def solve_contacts_pgs(self, state_in: State, state_aug, control: Control, contacts: Contacts, dt: float):
        model = self.model
        if not model.joint_count or model.articulation_count == 0:
            return

        device = model.device

        with self._timer("Contact assembly"):
            wp.launch(
                compute_velocity_predictor,
                dim=model.joint_dof_count,
                inputs=[
                    state_in.joint_qd,
                    state_aug.joint_qdd,
                    dt,
                ],
                outputs=[self.pgs_v_hat],
                device=device,
            )

            self.pgs_counts.zero_()

            if (
                contacts is not None
                and getattr(contacts, "rigid_contact_count", None) is not None
                and contacts.rigid_contact_max > 0
            ):
                enable_friction_flag = 1 if self.enable_contact_friction else 0
                wp.launch(
                    build_contact_rows_normal,
                    dim=contacts.rigid_contact_max,
                    inputs=[
                        contacts.rigid_contact_count,
                        contacts.rigid_contact_point0,
                        contacts.rigid_contact_point1,
                        contacts.rigid_contact_normal,
                        contacts.rigid_contact_shape0,
                        contacts.rigid_contact_shape1,
                        contacts.rigid_contact_thickness0,
                        contacts.rigid_contact_thickness1,
                        model.shape_body,
                        state_in.body_q,
                        model.shape_transform,
                        self.shape_material_mu,
                        model.articulation_start,
                        self.articulation_J_start,
                        self.articulation_H_rows,
                        self.articulation_dof_start,
                        self.body_to_joint,
                        self.body_to_articulation,
                        model.joint_ancestor,
                        self.J,
                        self.pgs_max_constraints,
                        self.articulation_max_dofs,
                        self.pgs_beta,
                        self.pgs_cfm,
                        enable_friction_flag,
                    ],
                    outputs=[
                        self.pgs_counts,
                        self.pgs_Jc,
                        self.pgs_phi,
                        self.pgs_row_beta,
                        self.pgs_row_cfm,
                        self.pgs_types,
                        self.pgs_target_velocity,
                        self.pgs_parent,
                        self.pgs_mu,
                    ],
                    device=device,
                )

            if self.pgs_use_joint_targets and control is not None and self.articulation_max_dofs > 0:
                beta_override = self.pgs_joint_beta if self.pgs_joint_beta is not None else -1.0
                cfm_override = self.pgs_joint_cfm if self.pgs_joint_cfm is not None else -1.0
                wp.launch(
                    build_joint_target_rows,
                    dim=model.articulation_count,
                    inputs=[
                        model.articulation_start,
                        self.articulation_H_rows,
                        self.articulation_dof_start,
                        model.joint_type,
                        model.joint_q_start,
                        model.joint_qd_start,
                        model.joint_dof_dim,
                        model.joint_target_ke,
                        model.joint_target_kd,
                        state_in.joint_q,
                        control.joint_target_pos,
                        control.joint_target_vel,
                        self.pgs_counts,
                        self.pgs_max_constraints,
                        self.articulation_max_dofs,
                        self.pgs_row_beta,
                        self.pgs_row_cfm,
                        self.pgs_types,
                        self.pgs_target_velocity,
                        self.pgs_phi,
                        self.pgs_Jc,
                        self.pgs_parent,
                        self.pgs_mu,
                        dt,
                        self.pgs_beta,
                        self.pgs_cfm,
                        beta_override,
                        cfm_override,
                    ],
                    device=device,
                )

            wp.launch(
                clamp_contact_counts,
                dim=model.articulation_count,
                inputs=[self.pgs_counts, self.pgs_max_constraints],
                device=device,
            )

        with self._timer("Contact system build (HinvJt + K + bias)"):
            if self.use_tiled_contact_build:
                wp.launch_tiled(
                    apply_hinv_Jt_multi_rhs_tiled,
                    dim=[model.articulation_count],
                    inputs=[
                        self.articulation_H_start,
                        self.articulation_H_rows,
                        self.pgs_max_constraints,
                        self.articulation_max_dofs,
                        self.pgs_counts,
                        self.L,
                        self.pgs_Jc,
                    ],
                    outputs=[self.pgs_Y],
                    block_dim=TILE_THREADS,
                    device=device,
                )

                wp.launch_tiled(
                    form_contact_matrix_tiled,
                    dim=[model.articulation_count],
                    inputs=[
                        self.articulation_H_rows,
                        self.pgs_max_constraints,
                        self.articulation_max_dofs,
                        self.pgs_counts,
                        self.pgs_Jc,
                        self.pgs_Y,
                        self.pgs_row_cfm,
                    ],
                    outputs=[
                        self.pgs_diag,
                        self.pgs_contact_matrix,
                    ],
                    block_dim=TILE_THREADS,
                    device=device,
                )
            else:
                wp.launch(
                    apply_hinv_Jt_multi_rhs,
                    dim=model.articulation_count * self.pgs_max_constraints,
                    inputs=[
                        self.articulation_H_start,
                        self.articulation_H_rows,
                        self.pgs_max_constraints,
                        self.articulation_max_dofs,
                        self.pgs_counts,
                        self.L,
                        self.pgs_Jc,
                    ],
                    outputs=[self.pgs_Y],
                    device=device,
                )

                wp.launch(
                    form_contact_matrix,
                    #dim=model.articulation_count,
                    dim=model.articulation_count * self.pgs_max_constraints,
                    inputs=[
                        self.articulation_H_rows,
                        self.pgs_max_constraints,
                        self.articulation_max_dofs,
                        self.pgs_counts,
                        self.pgs_Jc,
                        self.pgs_Y,
                        self.pgs_row_cfm,
                    ],
                    outputs=[
                        self.pgs_diag,
                        self.pgs_contact_matrix,
                    ],
                    device=device,
                )

            wp.launch(
                compute_contact_bias,
                dim=model.articulation_count,
                inputs=[
                    self.articulation_dof_start,
                    self.articulation_H_rows,
                    self.pgs_counts,
                    self.pgs_max_constraints,
                    self.articulation_max_dofs,
                    self.pgs_Jc,
                    self.pgs_v_hat,
                    self.pgs_phi,
                    self.pgs_row_beta,
                    self.pgs_types,
                    self.pgs_target_velocity,
                    dt,
                ],
                outputs=[self.pgs_rhs],
                device=device,
            )

        warmstart_flag = 1 if self.pgs_warmstart else 0
        with self._timer("PGS iterations"):
            wp.launch(
                prepare_impulses,
                dim=model.articulation_count,
                inputs=[self.pgs_counts, self.pgs_max_constraints, warmstart_flag, self.pgs_impulses],
                device=device,
            )

            wp.launch(
                pgs_solve_contacts,
                dim=model.articulation_count,
                inputs=[
                    self.pgs_counts,
                    self.pgs_max_constraints,
                    self.pgs_diag,
                    self.pgs_contact_matrix,
                    self.pgs_rhs,
                    self.pgs_impulses,
                    self.pgs_iterations,
                    self.pgs_omega,
                    self.pgs_types,
                    self.pgs_parent,
                    self.pgs_mu,
                ],
                device=device,
            )

        with self._timer("Contact apply (v, qdd)"):
            wp.launch(
                accumulate_contact_velocity,
                dim=model.articulation_count,
                inputs=[
                    self.articulation_dof_start,
                    self.articulation_H_rows,
                    self.pgs_counts,
                    self.pgs_max_constraints,
                    self.articulation_max_dofs,
                    self.pgs_Y,
                    self.pgs_v_hat,
                    self.pgs_impulses,
                ],
                outputs=[self.pgs_v_out],
                device=device,
            )

            wp.launch(
                update_qdd_from_velocity,
                dim=model.joint_dof_count,
                inputs=[state_in.joint_qd, self.pgs_v_out, 1.0 / dt],
                outputs=[state_aug.joint_qdd],
                device=device,
            )

    def allocate_state_aux_vars(self, model, target, requires_grad):
        # allocate auxiliary variables that vary with state
        if model.body_count:
            # joints
            target.joint_qdd = wp.zeros_like(model.joint_qd, requires_grad=requires_grad)
            target.joint_tau = wp.empty_like(model.joint_qd, requires_grad=requires_grad)
            if requires_grad:
                # used in the custom grad implementation of eval_dense_solve_batched
                target.joint_solve_tmp = wp.zeros_like(model.joint_qd, requires_grad=True)
            else:
                target.joint_solve_tmp = None
            target.joint_S_s = wp.empty(
                (model.joint_dof_count,),
                dtype=wp.spatial_vector,
                device=model.device,
                requires_grad=requires_grad,
            )

            # derived rigid body data (maximal coordinates)
            target.body_q_com = wp.empty_like(model.body_q, requires_grad=requires_grad)
            target.body_I_s = wp.empty(
                (model.body_count,), dtype=wp.spatial_matrix, device=model.device, requires_grad=requires_grad
            )
            target.body_v_s = wp.empty(
                (model.body_count,), dtype=wp.spatial_vector, device=model.device, requires_grad=requires_grad
            )
            target.body_a_s = wp.empty(
                (model.body_count,), dtype=wp.spatial_vector, device=model.device, requires_grad=requires_grad
            )
            target.body_f_s = wp.zeros(
                (model.body_count,), dtype=wp.spatial_vector, device=model.device, requires_grad=requires_grad
            )
            target.body_ft_s = wp.zeros(
                (model.body_count,), dtype=wp.spatial_vector, device=model.device, requires_grad=requires_grad
            )

            target._featherstone_augmented = True

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        contacts: Contacts,
        dt: float,
    ):
        requires_grad = state_in.requires_grad

        if self.pgs_use_augmented_targets:
            if self._last_step_dt is None:
                self._last_step_dt = dt
            elif abs(self._last_step_dt - dt) > 1.0e-8:
                self._force_mass_update = True
                self._last_step_dt = dt
            else:
                self._last_step_dt = dt

        # optionally create dynamical auxiliary variables
        if requires_grad:
            state_aug = state_out
        else:
            state_aug = self

        model = self.model

        if not getattr(state_aug, "_featherstone_augmented", False):
            self.allocate_state_aux_vars(model, state_aug, requires_grad)
        if control is None:
            control = model.control(clone_variables=False)

        with wp.ScopedTimer("simulate", False):
            particle_f = None
            body_f = None

            if state_in.particle_count:
                particle_f = state_in.particle_f

            if state_in.body_count:
                body_f = state_in.body_f

            with self._timer("Unconstrained dynamics"):
                # damped springs
                eval_spring_forces(model, state_in, particle_f)

                # triangle elastic and lift/drag forces
                eval_triangle_forces(model, state_in, control, particle_f)

                # triangle bending
                eval_bending_forces(model, state_in, particle_f)

                # tetrahedral FEM
                eval_tetrahedra_forces(model, state_in, control, particle_f)

                # particle-particle interactions
                eval_particle_contact_forces(model, state_in, particle_f)

                # particle shape contact
                eval_particle_body_contact_forces(model, state_in, contacts, particle_f, body_f, body_f_in_world_frame=True)

                # muscles
                if False:
                    eval_muscle_forces(model, state_in, control, body_f)

            # ----------------------------
            # articulations

            if model.joint_count:
                with self._timer("Articulation dynamics (FK/ID/drives)"):
                    with self._timer("FK + ID"):
                        # evaluate body transforms
                        wp.launch(
                            eval_rigid_fk,
                            dim=model.articulation_count,
                            inputs=[
                                model.articulation_start,
                                model.joint_type,
                                model.joint_parent,
                                model.joint_child,
                                model.joint_q_start,
                                model.joint_qd_start,
                                state_in.joint_q,
                                model.joint_X_p,
                                model.joint_X_c,
                                self.body_X_com,
                                model.joint_axis,
                                model.joint_dof_dim,
                            ],
                            outputs=[state_in.body_q, state_aug.body_q_com],
                            device=model.device,
                        )

                        # print("body_X_sc:")
                        # print(state_in.body_q.numpy())

                        # evaluate joint inertias, motion vectors, and forces
                        state_aug.body_f_s.zero_()

                        wp.launch(
                            eval_rigid_id,
                            dim=model.articulation_count,
                            inputs=[
                                model.articulation_start,
                                model.joint_type,
                                model.joint_parent,
                                model.joint_child,
                                model.joint_qd_start,
                                state_in.joint_qd,
                                model.joint_axis,
                                model.joint_dof_dim,
                                self.body_I_m,
                                state_in.body_q,
                                state_aug.body_q_com,
                                model.joint_X_p,
                                model.gravity,
                            ],
                            outputs=[
                                state_aug.joint_S_s,
                                state_aug.body_I_s,
                                state_aug.body_v_s,
                                state_aug.body_f_s,
                                state_aug.body_a_s,
                            ],
                            device=model.device,
                        )

                    if model.articulation_count:
                        with self._timer("Drives + augmentation"):
                            # evaluate joint torques
                            state_aug.body_ft_s.zero_()
                            use_joint_targets_flag = 0 if self.pgs_joint_target_mode in ("pgs", "augmented") else 1
                            wp.launch(
                                eval_rigid_tau,
                                dim=model.articulation_count,
                                inputs=[
                                    model.articulation_start,
                                    model.joint_type,
                                    model.joint_parent,
                                    model.joint_child,
                                    model.joint_q_start,
                                    model.joint_qd_start,
                                    model.joint_dof_dim,
                                    control.joint_target_pos,
                                    control.joint_target_vel,
                                    state_in.joint_q,
                                    state_in.joint_qd,
                                    control.joint_f,
                                    model.joint_target_ke,
                                    model.joint_target_kd,
                                    model.joint_limit_lower,
                                    model.joint_limit_upper,
                                    model.joint_limit_ke,
                                    model.joint_limit_kd,
                                    state_aug.joint_S_s,
                                    state_aug.body_f_s,
                                    body_f,
                                    use_joint_targets_flag,
                                ],
                                outputs=[
                                    state_aug.body_ft_s,
                                    state_aug.joint_tau,
                                ],
                                device=model.device,
                            )

                            # Clamp explicit PD torques to joint effort limits (MuJoCo-style forcerange)
                            if self.pgs_joint_target_mode == "off":
                                wp.launch(
                                    clamp_joint_tau,
                                    dim=model.joint_dof_count,
                                    inputs=[state_aug.joint_tau, model.joint_effort_limit],
                                    device=model.device,
                                )


                            if self.pgs_use_augmented_targets:
                                self.build_augmented_joint_targets(state_in, control, dt)
                                self.apply_augmented_joint_tau(state_in, state_aug, dt)

                                wp.launch(
                                    clamp_joint_tau,
                                    dim=model.joint_dof_count,
                                    inputs=[state_aug.joint_tau, model.joint_effort_limit],
                                    device=model.device,
                                )


                            # print("joint_tau:")
                            # print(state_aug.joint_tau.numpy())
                            # print("body_q:")
                            # print(state_in.body_q.numpy())
                            # print("body_qd:")
                            # print(state_in.body_qd.numpy())

                global_flag = 1 if ((self._step % self.update_mass_matrix_interval) == 0 or self._force_mass_update) else 0
                if self.pgs_use_augmented_targets:
                    mass_update = True
                else:
                    mass_update = bool(global_flag)
                if mass_update:
                    with self._timer("Mass matrix build (CRBA + prep)"):
                        wp.launch(
                            build_mass_update_mask,
                            dim=model.articulation_count,
                            inputs=[
                                global_flag,
                                self.limit_change_mask,
                            ],
                            outputs=[self.mass_update_mask],
                            device=model.device,
                        )
                        # build J
                        wp.launch(
                            eval_rigid_jacobian,
                            dim=model.articulation_count,
                            inputs=[
                                model.articulation_start,
                                self.articulation_J_start,
                                self.mass_update_mask,
                                model.joint_ancestor,
                                model.joint_qd_start,
                                state_aug.joint_S_s,
                            ],
                            outputs=[self.J],
                            device=model.device,
                        )

                        # build M
                        wp.launch(
                            eval_rigid_mass,
                            dim=model.articulation_count,
                            inputs=[
                                model.articulation_start,
                                self.articulation_M_start,
                                self.mass_update_mask,
                                state_aug.body_I_s,
                            ],
                            outputs=[self.M_blocks],
                            device=model.device,
                        )

                        # form H using CRBA
                        wp.launch(
                            eval_crba,
                            dim=model.articulation_count,
                            inputs=[
                                model.articulation_start,
                                self.mass_update_mask,
                                model.joint_ancestor,
                                model.joint_qd_start,
                                self.articulation_H_start,
                                self.articulation_H_rows,
                                state_aug.joint_S_s,
                                state_aug.body_I_s,
                            ],
                            outputs=[self.H],
                            device=model.device,
                        )

                        if self.pgs_use_augmented_targets:
                            wp.launch(
                                apply_augmented_mass_diagonal,
                                dim=model.articulation_count,
                            inputs=[
                                self.articulation_H_start,
                                self.articulation_H_rows,
                                self.articulation_dof_start,
                                self.articulation_max_dofs,
                                self.mass_update_mask,
                                self.aug_row_counts,
                                self.aug_row_dof_index,
                                self.aug_row_K,
                                self.H,
                            ],
                            device=model.device,
                        )

                            wp.launch(
                                copy_int_array_masked,
                                dim=model.articulation_count,
                                inputs=[self.aug_limit_counts, self.mass_update_mask],
                                outputs=[self.aug_prev_limit_counts],
                                device=model.device,
                            )

                        self._force_mass_update = False

                        # print("joint_target:")
                        # print(control.joint_target.numpy())
                        # print("joint_tau:")
                        # print(state_aug.joint_tau.numpy())
                        # print("H:")
                        # print(self.H.numpy())
                        # print("L:")
                        # print(self.L.numpy())

                with self._timer("Mass solve (Cholesky + backsolve)"):
                    if mass_update:
                        wp.launch(
                            eval_dense_cholesky_batched,
                            dim=model.articulation_count,
                            inputs=[
                                self.articulation_H_start,
                                self.articulation_H_rows,
                                self.H,
                                model.joint_armature,
                                self.mass_update_mask,
                            ],
                            outputs=[self.L],
                            device=model.device,
                        )

                    # solve for qdd
                    state_aug.joint_qdd.zero_()
                    wp.launch(
                        eval_dense_solve_batched,
                        dim=model.articulation_count,
                        inputs=[
                            self.articulation_H_start,
                            self.articulation_H_rows,
                            self.articulation_dof_start,
                            self.H,
                            self.L,
                            state_aug.joint_tau,
                        ],
                        outputs=[
                            state_aug.joint_qdd,
                            state_aug.joint_solve_tmp,
                        ],
                        device=model.device,
                    )

                self.solve_contacts_pgs(state_in, state_aug, control, contacts, dt)
                # print("joint_qdd:")
                # print(state_aug.joint_qdd.numpy())
                # print("\n\n")

        # -------------------------------------
        # integrate bodies

        if model.joint_count:
            with self._timer("Integration + FK_out"):
                wp.launch(
                    kernel=integrate_generalized_joints,
                    dim=model.joint_count,
                    inputs=[
                        model.joint_type,
                        model.joint_q_start,
                        model.joint_qd_start,
                        model.joint_dof_dim,
                        state_in.joint_q,
                        state_in.joint_qd,
                        state_aug.joint_qdd,
                        dt,
                    ],
                    outputs=[state_out.joint_q, state_out.joint_qd],
                    device=model.device,
                )

                # update maximal coordinates
                eval_fk(model, state_out.joint_q, state_out.joint_qd, state_out)

        self.integrate_particles(model, state_in, state_out, dt)

        self._step += 1

        return state_out
