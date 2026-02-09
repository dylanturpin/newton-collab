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

from contextlib import contextmanager
from typing import ClassVar

import numpy as np
import warp as wp

from ...core.types import override
from ...sim import Contacts, Control, Model, State, eval_fk
from ..semi_implicit.kernels_contact import (
    eval_particle_body_contact_forces,
    eval_particle_contact_forces,
)
from ..semi_implicit.kernels_particle import (
    eval_bending_forces,
    eval_spring_forces,
    eval_tetrahedra_forces,
    eval_triangle_forces,
)
from ..solver import SolverBase
from .kernels import (
    TILE_CONSTRAINTS,
    TILE_DOF,
    TILE_THREADS,
    allocate_world_contact_slots,
    apply_augmented_joint_tau,
    apply_augmented_mass_diagonal,
    apply_augmented_mass_diagonal_grouped,
    apply_impulses_flat_par_dof,
    apply_impulses_world_par_dof,
    build_augmented_joint_rows,
    build_contact_rows_normal,
    build_mass_update_mask,
    cholesky_batched_loop,
    cholesky_flat_loop,
    clamp_contact_counts,
    clamp_joint_tau,
    compute_com_transforms,
    compute_composite_inertia,
    compute_spatial_inertia,
    compute_velocity_predictor,
    compute_world_contact_bias,
    contact_bias_flat_par_row,
    copy_int_array_masked,
    crba_fill_batched_par_dof,
    crba_fill_flat_par_dof,
    delassus_batched_par_row_col,
    delassus_flat_par_row,
    detect_limit_count_changes,
    eval_rigid_fk,
    eval_rigid_id,
    eval_rigid_mass,
    eval_rigid_tau,
    finalize_world_constraint_counts,
    finalize_world_diag_cfm,
    gather_tau_to_groups,
    hinv_jt_batched_par_row,
    hinv_jt_flat_par_row,
    integrate_generalized_joints,
    pgs_solve_loop,
    populate_world_J_for_size,
    prepare_impulses,
    prepare_world_impulses,
    rhs_accum_world_par_art,
    scatter_qdd_from_groups,
    trisolve_batched_loop,
    trisolve_flat_loop,
    update_body_qd_from_featherstone,
    update_qdd_from_velocity,
)


@wp.kernel
def localize_parent_indices(
    counts: wp.array(dtype=int),
    max_constraints: int,
    parent_flat: wp.array(dtype=int),
    parent_local_flat: wp.array(dtype=int),
):
    art = wp.tid()
    m = counts[art]
    base = art * max_constraints

    for i in range(m):
        idx = base + i
        p = parent_flat[idx]
        if p >= 0:
            parent_local_flat[idx] = p - base
        else:
            parent_local_flat[idx] = -1


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
        # Storage path
        storage: str = "batched",
        # Kernel selection per operation
        cholesky_kernel: str = "auto",
        trisolve_kernel: str = "auto",
        hinv_jt_kernel: str = "auto",
        delassus_kernel: str = "auto",
        pgs_kernel: str = "tiled_row",
        # Streaming kernel chunk sizes (None = auto-select)
        delassus_chunk_size: int | None = None,
        pgs_chunk_size: int | None = None,
        # Auto selection threshold (batched path)
        small_dof_threshold: int = 12,
        # Parallelism options
        use_parallel_streams: bool = True,
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
            storage (str, optional): Storage layout. "batched" groups by DOF size into 3D arrays, "flat" uses
                offset-indexed 1D arrays. Defaults to "batched".
            cholesky_kernel (str, optional): "tiled", "loop", or "auto" for Cholesky factorization. Defaults to "auto".
            trisolve_kernel (str, optional): "tiled", "loop", or "auto" for triangular solve. Defaults to "auto".
            hinv_jt_kernel (str, optional): "tiled", "par_row", or "auto" for H^{-1}J^T. Defaults to "auto".
            delassus_kernel (str, optional): "tiled", "par_row_col", or "auto" for Delassus accumulation
                (C = J * H^{-1} * J^T). "tiled" uses a streaming CUDA kernel that chunks shared memory
                and scales to any constraint count. "par_row_col" launches one thread per matrix element.
                "auto" selects "tiled" in the batched path. Defaults to "auto".
            pgs_kernel (str, optional): "loop", "tiled_row", or "tiled_contact" for PGS solve. Defaults to "tiled_row".
            delassus_chunk_size (int, optional): Chunk size (in constraint rows) for the streaming Delassus
                kernel. Controls how many rows of J and Y are loaded into shared memory at once.
                None selects automatically based on shared memory heuristics. Defaults to None.
            pgs_chunk_size (int, optional): Chunk size (in contacts, i.e. groups of 3 constraint rows)
                for the streaming PGS kernel. Controls how many block-rows of the Delassus matrix are
                preloaded into shared memory at once. 1 = current streaming behavior (one block-row
                at a time). None defaults to 1. Defaults to None.
            small_dof_threshold (int, optional): DOF threshold for "auto" selection in batched path. Defaults to 12.
            use_parallel_streams (bool, optional): Dispatch size groups on separate CUDA streams (batched path only).
                Defaults to True.

        Auto selection behavior:
            - Batched + auto: size > threshold -> tiled (or par_row for hinv_jt), else loop/par_row.
            - Flat + auto: if homogeneous and within tile limits -> tiled, else loop/par_row.
            - Flat + tiled: only valid when homogeneous and within TILE_DOF / TILE_CONSTRAINTS limits.
            - Delassus batched + auto/tiled: streaming kernel (handles any constraint count via chunking).
            - Current limitation: storage="flat" requires one articulation per world.

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
        self.pgs_warmstart = pgs_warmstart

        valid_storage = {"batched", "flat"}
        if storage not in valid_storage:
            raise ValueError(f"storage must be one of {sorted(valid_storage)}")

        valid_cholesky = {"tiled", "loop", "auto"}
        if cholesky_kernel not in valid_cholesky:
            raise ValueError(f"cholesky_kernel must be one of {sorted(valid_cholesky)}")

        valid_trisolve = {"tiled", "loop", "auto"}
        if trisolve_kernel not in valid_trisolve:
            raise ValueError(f"trisolve_kernel must be one of {sorted(valid_trisolve)}")

        valid_hinv_jt = {"tiled", "par_row", "auto"}
        if hinv_jt_kernel not in valid_hinv_jt:
            raise ValueError(f"hinv_jt_kernel must be one of {sorted(valid_hinv_jt)}")

        valid_delassus = {"tiled", "par_row_col", "auto"}
        if delassus_kernel not in valid_delassus:
            raise ValueError(f"delassus_kernel must be one of {sorted(valid_delassus)}")

        valid_pgs = {"loop", "tiled_row", "tiled_contact", "streaming"}
        if pgs_kernel not in valid_pgs:
            raise ValueError(f"pgs_kernel must be one of {sorted(valid_pgs)}")

        self.storage = storage
        self.cholesky_kernel = cholesky_kernel
        self.trisolve_kernel = trisolve_kernel
        self.hinv_jt_kernel = hinv_jt_kernel
        self.delassus_kernel = delassus_kernel
        self.pgs_kernel = pgs_kernel
        self.delassus_chunk_size = delassus_chunk_size
        self.pgs_chunk_size = pgs_chunk_size if pgs_chunk_size is not None else 1
        self.small_dof_threshold = small_dof_threshold
        self.use_parallel_streams = use_parallel_streams

        self._step = 0
        self._force_mass_update = False
        self._last_step_dt = None

        self._compute_articulation_metadata(model)

        if self.storage == "flat" and self._is_multi_articulation:
            raise ValueError("storage='flat' is not supported for multiple articulations per world yet.")

        if self.storage == "flat" and model.articulation_count:
            if not self._is_homogeneous and (
                self.cholesky_kernel == "tiled" or self.trisolve_kernel == "tiled" or self.hinv_jt_kernel == "tiled"
            ):
                raise ValueError("Flat tiled kernels require homogeneous articulation sizes.")
            if self.cholesky_kernel == "tiled" and self.articulation_max_dofs > int(TILE_DOF):
                raise ValueError(
                    f"articulation_max_dofs={self.articulation_max_dofs} exceeds TILE_DOF={int(TILE_DOF)} "
                    "for flat tiled Cholesky. Increase TILE_DOF or choose cholesky_kernel='loop'."
                )
            if self.trisolve_kernel == "tiled" and self.articulation_max_dofs > int(TILE_DOF):
                raise ValueError(
                    f"articulation_max_dofs={self.articulation_max_dofs} exceeds TILE_DOF={int(TILE_DOF)} "
                    "for flat tiled triangular solve. Increase TILE_DOF or choose trisolve_kernel='loop'."
                )
            if self.hinv_jt_kernel == "tiled" and (
                self.articulation_max_dofs > int(TILE_DOF) or self.pgs_max_constraints > int(TILE_CONSTRAINTS)
            ):
                raise ValueError(
                    "Flat tiled H^{-1}J^T requires articulation_max_dofs <= TILE_DOF and "
                    "pgs_max_constraints <= TILE_CONSTRAINTS."
                )
            if self.pgs_kernel in ("tiled_row", "tiled_contact") and self.pgs_max_constraints > int(
                TILE_CONSTRAINTS
            ):
                raise ValueError(
                    f"pgs_max_constraints={self.pgs_max_constraints} exceeds TILE_CONSTRAINTS={int(TILE_CONSTRAINTS)} "
                    "for tiled PGS. Increase TILE_CONSTRAINTS or choose pgs_kernel='loop'."
                )
        self._allocate_common_buffers(model)
        if self.storage == "flat":
            self._allocate_flat_buffers(model)
        else:
            self._allocate_batched_buffers(model)
            self._allocate_world_buffers(model)
            self._scatter_armature_to_groups(model)
            self._init_size_group_streams(model)

        if model.shape_material_mu is not None:
            self.shape_material_mu = model.shape_material_mu
        else:
            self.shape_material_mu = wp.zeros(
                (1,), dtype=wp.float32, device=model.device, requires_grad=model.requires_grad
            )

    def _compute_articulation_metadata(self, model):
        self._compute_articulation_indices(model)
        self._setup_size_grouping(model)
        self._setup_world_mapping(model)
        self._is_one_art_per_world = self.world_count == model.articulation_count
        self._is_homogeneous = (len(self.size_groups) == 1) if self.size_groups else True
        self._build_body_maps(model)

    def _compute_articulation_indices(self, model):
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

    def _setup_size_grouping(self, model):
        """Set up size-grouped storage and indirection arrays for multi-articulation support.

        This enables efficient handling of articulations with different DOF counts by grouping
        them by size, allowing optimized tiled kernel launches for each size group.
        """
        if not model.articulation_count or not model.joint_count:
            self.size_groups = []
            self.n_arts_by_size = {}
            return

        device = model.device

        # Get DOF counts per articulation
        articulation_start = model.articulation_start.numpy()
        joint_qd_start = model.joint_qd_start.numpy()

        articulation_dof_counts = np.zeros(model.articulation_count, dtype=np.int32)
        for art_idx in range(model.articulation_count):
            first_joint = articulation_start[art_idx]
            last_joint = articulation_start[art_idx + 1]
            first_dof = joint_qd_start[first_joint]
            last_dof = joint_qd_start[last_joint]
            articulation_dof_counts[art_idx] = last_dof - first_dof

        # Determine unique sizes (sorted descending for largest first)
        unique_sizes = sorted(set(articulation_dof_counts), reverse=True)
        self.size_groups = unique_sizes
        self.n_arts_by_size = {size: int(np.sum(articulation_dof_counts == size)) for size in unique_sizes}

        # Build indirection arrays
        art_size_np = articulation_dof_counts.copy()
        art_group_idx_np = np.zeros(model.articulation_count, dtype=np.int32)
        group_to_art_np = {size: np.zeros(self.n_arts_by_size[size], dtype=np.int32) for size in unique_sizes}

        # Track current index within each size group
        size_counters = dict.fromkeys(unique_sizes, 0)

        for art_idx in range(model.articulation_count):
            size = articulation_dof_counts[art_idx]
            group_idx = size_counters[size]

            art_group_idx_np[art_idx] = group_idx
            group_to_art_np[size][group_idx] = art_idx

            size_counters[size] += 1

        # Copy to GPU
        self.art_size = wp.array(art_size_np, dtype=wp.int32, device=device)
        self.art_group_idx = wp.array(art_group_idx_np, dtype=wp.int32, device=device)
        self.group_to_art = {
            size: wp.array(group_to_art_np[size], dtype=wp.int32, device=device) for size in unique_sizes
        }

    def _setup_world_mapping(self, model):
        """Set up world-level mapping for multi-articulation support.

        Maps articulations to worlds and computes per-world articulation ranges.
        """
        if not model.articulation_count:
            self.world_count = 0
            self.art_to_world = None
            self.world_art_start = None
            self._is_multi_articulation = False
            self._max_arts_per_world = 0
            return

        device = model.device

        # Get articulation-to-world mapping from model
        if model.articulation_world is not None:
            art_to_world_np = model.articulation_world.numpy().astype(np.int32)
            # Handle -1 (global) by mapping to world 0
            art_to_world_np = np.where(art_to_world_np < 0, 0, art_to_world_np)
            self.world_count = int(np.max(art_to_world_np)) + 1
        else:
            # Default: one articulation per world (current behavior)
            art_to_world_np = np.arange(model.articulation_count, dtype=np.int32)
            self.world_count = model.articulation_count

        self.art_to_world = wp.array(art_to_world_np, dtype=wp.int32, device=device)

        # Compute per-world articulation ranges
        # Count articulations per world
        world_art_counts = np.zeros(self.world_count, dtype=np.int32)
        for world_idx in art_to_world_np:
            world_art_counts[world_idx] += 1

        # Compute start indices (exclusive prefix sum)
        world_art_start_np = np.zeros(self.world_count + 1, dtype=np.int32)
        world_art_start_np[1:] = np.cumsum(world_art_counts)

        self.world_art_start = wp.array(world_art_start_np, dtype=wp.int32, device=device)

        # Detect if we have multiple articulations per world
        self._max_arts_per_world = int(np.max(world_art_counts)) if len(world_art_counts) > 0 else 0
        self._is_multi_articulation = self._max_arts_per_world > 1

    def _build_body_maps(self, model):
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

    def _allocate_common_buffers(self, model):
        if model.joint_count:
            self.M_blocks = wp.zeros(
                (self.M_size,), dtype=wp.float32, device=model.device, requires_grad=model.requires_grad
            )
            self.mass_update_mask = wp.zeros(
                (model.articulation_count,), dtype=wp.int32, device=model.device, requires_grad=model.requires_grad
            )
            self.v_hat = wp.zeros_like(model.joint_qd, requires_grad=model.requires_grad)
            self.v_out = wp.zeros_like(model.joint_qd, requires_grad=model.requires_grad)
        else:
            self.M_blocks = None
            self.mass_update_mask = None
            self.v_hat = None
            self.v_out = None

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
            self.body_I_c = wp.empty(
                (model.body_count,), dtype=wp.spatial_matrix, device=model.device, requires_grad=model.requires_grad
            )
        else:
            self.body_I_m = None
            self.body_X_com = None
            self.body_I_c = None

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
        self.aug_row_dof_index = wp.zeros((total_rows,), dtype=wp.int32, device=device, requires_grad=requires_grad)
        self.aug_row_K = wp.zeros((total_rows,), dtype=wp.float32, device=device, requires_grad=requires_grad)
        self.aug_row_u0 = wp.zeros((total_rows,), dtype=wp.float32, device=device, requires_grad=requires_grad)

    def _allocate_flat_buffers(self, model):
        if not model.articulation_count or not model.joint_count:
            self.H_flat = None
            self.L_flat = None
            self.constraint_count_flat = None
            self.phi_flat = None
            self.diag_flat = None
            self.rhs_flat = None
            self.impulses_flat = None
            self.row_type_flat = None
            self.row_beta_flat = None
            self.row_cfm_flat = None
            self.target_velocity_flat = None
            self.row_parent_flat = None
            self.row_mu_flat = None
            self.C_flat = None
            self.J_flat = None
            self.Y_flat = None
            self.row_parent_local_flat = None
            self.H_by_size = {}
            self.L_by_size = {}
            self.J_by_size = {}
            self.Y_by_size = {}
            self.R_by_size = {}
            self.tau_by_size = {}
            self.qdd_by_size = {}
            return

        max_dofs = self.articulation_max_dofs
        if max_dofs == 0:
            return

        device = model.device
        requires_grad = model.requires_grad
        articulation_count = model.articulation_count
        constraint_capacity = self.pgs_max_constraints

        self.H_flat = wp.empty((self.H_size,), dtype=wp.float32, device=device, requires_grad=requires_grad)
        self.L_flat = wp.zeros_like(self.H_flat)

        self.pgs_row_stride = constraint_capacity * max_dofs
        self.pgs_matrix_stride = constraint_capacity * constraint_capacity

        total_rows = articulation_count * self.pgs_row_stride
        total_constraints = articulation_count * constraint_capacity
        total_matrix = articulation_count * self.pgs_matrix_stride

        self.constraint_count_flat = wp.zeros(
            (articulation_count,), dtype=wp.int32, device=device, requires_grad=requires_grad
        )
        self.phi_flat = wp.zeros((total_constraints,), dtype=wp.float32, device=device, requires_grad=requires_grad)
        self.diag_flat = wp.zeros((total_constraints,), dtype=wp.float32, device=device, requires_grad=requires_grad)
        self.rhs_flat = wp.zeros((total_constraints,), dtype=wp.float32, device=device, requires_grad=requires_grad)
        self.impulses_flat = wp.zeros(
            (total_constraints,), dtype=wp.float32, device=device, requires_grad=requires_grad
        )
        self.row_type_flat = wp.zeros((total_constraints,), dtype=wp.int32, device=device, requires_grad=requires_grad)
        self.row_beta_flat = wp.zeros(
            (total_constraints,), dtype=wp.float32, device=device, requires_grad=requires_grad
        )
        self.row_cfm_flat = wp.zeros((total_constraints,), dtype=wp.float32, device=device, requires_grad=requires_grad)
        self.target_velocity_flat = wp.zeros(
            (total_constraints,), dtype=wp.float32, device=device, requires_grad=requires_grad
        )
        self.row_parent_flat = wp.full(
            (total_constraints,), -1, dtype=wp.int32, device=device, requires_grad=requires_grad
        )
        self.row_mu_flat = wp.zeros((total_constraints,), dtype=wp.float32, device=device, requires_grad=requires_grad)
        self.C_flat = wp.zeros((total_matrix,), dtype=wp.float32, device=device, requires_grad=requires_grad)
        self.J_flat = wp.zeros((total_rows,), dtype=wp.float32, device=device, requires_grad=requires_grad)
        self.Y_flat = wp.zeros((total_rows,), dtype=wp.float32, device=device, requires_grad=requires_grad)
        self.row_parent_local_flat = wp.full(
            (total_constraints,),
            -1,
            dtype=wp.int32,
            device=device,
            requires_grad=requires_grad,
        )

    def _allocate_batched_buffers(self, model):
        if not self.size_groups:
            self.H_by_size = {}
            self.L_by_size = {}
            self.J_by_size = {}
            self.Y_by_size = {}
            self.R_by_size = {}
            self.tau_by_size = {}
            self.qdd_by_size = {}
            return

        device = model.device
        requires_grad = model.requires_grad
        max_constraints = self.pgs_max_constraints

        self.H_by_size = {}
        self.L_by_size = {}
        self.J_by_size = {}
        self.Y_by_size = {}
        self.R_by_size = {}
        self.tau_by_size = {}
        self.qdd_by_size = {}

        for size in self.size_groups:
            n_arts = self.n_arts_by_size[size]

            h_dim = size
            j_rows = max_constraints

            # Mass matrix and Cholesky factor [n_arts, h_dim, h_dim]
            self.H_by_size[size] = wp.zeros(
                (n_arts, h_dim, h_dim), dtype=wp.float32, device=device, requires_grad=requires_grad
            )
            self.L_by_size[size] = wp.zeros(
                (n_arts, h_dim, h_dim), dtype=wp.float32, device=device, requires_grad=requires_grad
            )

            # Contact Jacobian and Y = H^-1 * J^T [n_arts, j_rows, h_dim]
            self.J_by_size[size] = wp.zeros(
                (n_arts, j_rows, h_dim), dtype=wp.float32, device=device, requires_grad=requires_grad
            )
            self.Y_by_size[size] = wp.zeros(
                (n_arts, j_rows, h_dim), dtype=wp.float32, device=device, requires_grad=requires_grad
            )

            # Armature (regularization) [n_arts, h_dim] - needs to match H dimension for tile_diag_add
            self.R_by_size[size] = wp.zeros(
                (n_arts, h_dim), dtype=wp.float32, device=device, requires_grad=requires_grad
            )

            # Tau and qdd grouped buffers for tiled triangular solve [n_arts, h_dim, 1]
            self.tau_by_size[size] = wp.zeros((n_arts, h_dim, 1), dtype=wp.float32, device=device)
            self.qdd_by_size[size] = wp.zeros((n_arts, h_dim, 1), dtype=wp.float32, device=device)

        max_contacts = model.rigid_contact_max if model.rigid_contact_max > 0 else 1
        self.contact_world = wp.zeros((max_contacts,), dtype=wp.int32, device=device, requires_grad=requires_grad)
        self.contact_slot = wp.zeros((max_contacts,), dtype=wp.int32, device=device, requires_grad=requires_grad)
        self.contact_art_a = wp.zeros((max_contacts,), dtype=wp.int32, device=device, requires_grad=requires_grad)
        self.contact_art_b = wp.zeros((max_contacts,), dtype=wp.int32, device=device, requires_grad=requires_grad)
        self.slot_counter = wp.zeros((self.world_count,), dtype=wp.int32, device=device, requires_grad=requires_grad)

    def _allocate_world_buffers(self, model):
        """Allocate world-level constraint system buffers for multi-articulation support."""
        if self.world_count == 0:
            return

        device = model.device
        requires_grad = model.requires_grad
        max_constraints = self.pgs_max_constraints

        # Per-world constraint matrices and vectors
        self.C = wp.zeros(
            (self.world_count, max_constraints, max_constraints),
            dtype=wp.float32,
            device=device,
            requires_grad=requires_grad,
        )
        self.rhs = wp.zeros(
            (self.world_count, max_constraints), dtype=wp.float32, device=device, requires_grad=requires_grad
        )
        self.impulses = wp.zeros(
            (self.world_count, max_constraints), dtype=wp.float32, device=device, requires_grad=requires_grad
        )
        self.diag = wp.zeros(
            (self.world_count, max_constraints), dtype=wp.float32, device=device, requires_grad=requires_grad
        )

        # Constraint metadata (per world x constraint)
        self.row_type = wp.zeros(
            (self.world_count, max_constraints), dtype=wp.int32, device=device, requires_grad=requires_grad
        )
        self.row_parent = wp.full(
            (self.world_count, max_constraints), -1, dtype=wp.int32, device=device, requires_grad=requires_grad
        )
        self.row_mu = wp.zeros(
            (self.world_count, max_constraints), dtype=wp.float32, device=device, requires_grad=requires_grad
        )
        self.row_beta = wp.zeros(
            (self.world_count, max_constraints), dtype=wp.float32, device=device, requires_grad=requires_grad
        )
        self.row_cfm = wp.zeros(
            (self.world_count, max_constraints), dtype=wp.float32, device=device, requires_grad=requires_grad
        )
        self.phi = wp.zeros(
            (self.world_count, max_constraints), dtype=wp.float32, device=device, requires_grad=requires_grad
        )
        self.target_velocity = wp.zeros(
            (self.world_count, max_constraints), dtype=wp.float32, device=device, requires_grad=requires_grad
        )

        # Per-world constraint counts
        self.constraint_count = wp.zeros(
            (self.world_count,), dtype=wp.int32, device=device, requires_grad=requires_grad
        )

    def _scatter_armature_to_groups(self, model):
        """Copy armature from model (DOF-ordered) to size-grouped storage."""
        if not self.size_groups:
            return

        armature_np = model.joint_armature.numpy()
        art_dof_start_np = self.articulation_dof_start.numpy()
        art_H_rows_np = self.articulation_H_rows.numpy()

        # R_by_size is sized to actual DOF count (matches H_by_size allocation)
        for size in self.size_groups:
            n_arts = self.n_arts_by_size[size]
            R_np = np.zeros((n_arts, size), dtype=np.float32)

            group_to_art_np = self.group_to_art[size].numpy()
            for group_idx in range(n_arts):
                art_idx = group_to_art_np[group_idx]
                dof_start = art_dof_start_np[art_idx]
                dof_count = art_H_rows_np[art_idx]
                R_np[group_idx, :dof_count] = armature_np[dof_start : dof_start + dof_count]

            self.R_by_size[size] = wp.array(R_np, dtype=wp.float32, device=model.device)

    def _init_size_group_streams(self, model):
        """Initialize CUDA streams for parallel kernel launches across size groups.

        When multiple DOF sizes exist (heterogeneous articulations), we can launch
        tiled kernels for different sizes in parallel using separate CUDA streams.
        """
        self._size_streams: dict[int, wp.Stream | None] = {}
        self._size_events: dict[int, wp.Event | None] = {}

        if self.use_parallel_streams and model.device.is_cuda and len(self.size_groups) > 1:
            for size in self.size_groups:
                self._size_streams[size] = wp.Stream(model.device)
                self._size_events[size] = wp.Event(model.device)
        else:
            # No streams needed for CPU or single size group
            for size in self.size_groups:
                self._size_streams[size] = None
                self._size_events[size] = None

    @override
    def step(
        self,
        state_in: State,
        state_out: State,
        control: Control,
        contacts: Contacts,
        dt: float,
    ):
        if self._last_step_dt is None:
            self._last_step_dt = dt
        elif abs(self._last_step_dt - dt) > 1.0e-8:
            self._force_mass_update = True
            self._last_step_dt = dt
        else:
            self._last_step_dt = dt

        model = self.model

        if control is None:
            control = model.control(clone_variables=False)
        state_aug = self._prepare_augmented_state(state_in, state_out, control)

        self._eval_particle_forces(state_in, control, contacts)

        if not model.joint_count:
            self.integrate_particles(model, state_in, state_out, dt)
            self._step += 1
            return state_out

        if self.storage == "batched":
            self._step_batched(state_in, state_out, state_aug, control, contacts, dt)
        else:
            self._step_flat(state_in, state_out, state_aug, control, contacts, dt)

        self._step += 1
        return state_out

    def _prepare_augmented_state(
        self,
        state_in: State,
        state_out: State,
        control: Control,
    ) -> State:
        requires_grad = state_in.requires_grad
        state_aug = state_out if requires_grad else self
        model = self.model

        if not getattr(state_aug, "_featherstone_augmented", False):
            self._allocate_state_aux_vars(model, state_aug, requires_grad)

        return state_aug

    def _allocate_state_aux_vars(self, model, target, requires_grad):
        # allocate auxiliary variables that vary with state
        if model.body_count:
            # joints
            target.joint_qdd = wp.zeros_like(model.joint_qd, requires_grad=requires_grad)
            target.joint_tau = wp.empty_like(model.joint_qd, requires_grad=requires_grad)
            if requires_grad:
                # used in the custom grad implementation of trisolve_flat_loop
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

    def _eval_particle_forces(self, state_in: State, control: Control, contacts: Contacts):
        model = self.model

        particle_f = state_in.particle_f if state_in.particle_count else None
        body_f = state_in.body_f if state_in.body_count else None

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

    def _step_flat(
        self,
        state_in: State,
        state_out: State,
        state_aug: State,
        control: Control,
        contacts: Contacts,
        dt: float,
    ):
        model = self.model

        flat_tiled_allowed = (
            self._is_homogeneous
            and self.articulation_max_dofs <= int(TILE_DOF)
            and self.pgs_max_constraints <= int(TILE_CONSTRAINTS)
        )

        # ══════════════════════════════════════════════════════════════
        # STAGE 1: FK/ID + drives + CRBA
        # ══════════════════════════════════════════════════════════════
        self._stage1_fk_id(state_in, state_aug, state_out)

        if model.articulation_count:
            self._stage1_drives(state_in, state_aug, control, dt)

        self._stage1_crba_flat(state_aug)

        # ══════════════════════════════════════════════════════════════
        # STAGE 2: Cholesky
        # ══════════════════════════════════════════════════════════════
        use_tiled_cholesky = (self.cholesky_kernel != "loop") and flat_tiled_allowed
        if use_tiled_cholesky:
            self._stage2_cholesky_flat_tiled()
        else:
            self._stage2_cholesky_flat_loop()

        # ══════════════════════════════════════════════════════════════
        # STAGE 3: Triangular solve + v_hat
        # ══════════════════════════════════════════════════════════════
        self._stage3_zero_qdd(state_aug)
        use_tiled_trisolve = (self.trisolve_kernel != "loop") and flat_tiled_allowed
        if use_tiled_trisolve:
            self._stage3_trisolve_flat_tiled(state_aug)
        else:
            self._stage3_trisolve_flat_loop(state_aug)

        self._stage3_compute_v_hat(state_in, state_aug, dt)

        # ══════════════════════════════════════════════════════════════
        # STAGE 4: Build contact problem
        # ══════════════════════════════════════════════════════════════
        self._stage4_build_rows_flat(state_in, state_aug, contacts)

        use_tiled_hinv_jt = (self.hinv_jt_kernel != "par_row") and flat_tiled_allowed
        if use_tiled_hinv_jt:
            self._stage4_hinv_jt_flat_tiled()
        else:
            self._stage4_hinv_jt_flat_par_row()
            self._stage4_delassus_flat_par_row()
        self._stage4_compute_rhs_flat(dt)

        # ══════════════════════════════════════════════════════════════
        # STAGE 5: PGS solve
        # ══════════════════════════════════════════════════════════════
        self._stage5_prepare_impulses_flat()
        if self.pgs_kernel == "tiled_row":
            self._stage5_pgs_solve_flat_tiled_row()
        elif self.pgs_kernel == "tiled_contact":
            self._stage5_pgs_solve_flat_tiled_contact()
        elif self.pgs_kernel == "streaming":
            self._stage5_pgs_solve_flat_streaming()
        else:
            self._stage5_pgs_solve_flat_loop()

        # ══════════════════════════════════════════════════════════════
        # STAGE 6: Apply impulses + integrate
        # ══════════════════════════════════════════════════════════════
        self._stage6_apply_impulses_flat()
        self._stage6_update_qdd(state_in, state_aug, dt)

        self._stage6_integrate(state_in, state_aug, state_out, dt)

    @contextmanager
    def _parallel_size_region(self, enabled: bool = True):
        """Context for parallel dispatch across size groups."""
        if not enabled or not self.use_parallel_streams or not self.model.device.is_cuda or len(self.size_groups) <= 1:
            yield
            return

        main_stream = wp.get_stream(self.model.device)
        self._main_stream = main_stream
        self._init_event = main_stream.record_event()
        try:
            yield
        finally:
            for size in self.size_groups:
                stream = self._size_streams.get(size)
                if stream is not None:
                    main_stream.wait_event(stream.record_event())
            self._main_stream = None
            self._init_event = None

    @contextmanager
    def _on_size_stream(self, size: int):
        """Execute block on this size's CUDA stream."""
        stream = self._size_streams.get(size)
        init_event = getattr(self, "_init_event", None)
        if stream is not None and init_event is not None:
            stream.wait_event(init_event)
            with wp.ScopedStream(stream):
                yield
        else:
            yield

    @contextmanager
    def _size_dispatch(self, enabled: bool):
        with self._parallel_size_region(enabled=enabled):
            yield

    @contextmanager
    def _size_ctx(self, size: int):
        with self._on_size_stream(size):
            yield

    def _for_sizes(self, enabled: bool):
        # convenience generator; keeps step code tight
        with self._size_dispatch(enabled):
            for size in self.size_groups:
                yield size, self._size_ctx(size)

    def _step_batched(
        self,
        state_in: State,
        state_out: State,
        state_aug: State,
        control: Control,
        contacts: Contacts,
        dt: float,
    ):
        model = self.model

        # ══════════════════════════════════════════════════════════════
        # STAGE 1: FK/ID + drives + CRBA
        # ══════════════════════════════════════════════════════════════
        self._stage1_fk_id(state_in, state_aug, state_out)

        if model.articulation_count:
            self._stage1_drives(state_in, state_aug, control, dt)

        self._stage1_crba_batched(state_aug)

        # ══════════════════════════════════════════════════════════════
        # STAGE 2: Cholesky
        # ══════════════════════════════════════════════════════════════
        for size, ctx in self._for_sizes(enabled=self.use_parallel_streams):
            with ctx:
                use_tiled = (self.cholesky_kernel == "tiled") or (
                    self.cholesky_kernel == "auto" and size > self.small_dof_threshold
                )
                if use_tiled:
                    self._stage2_cholesky_batched_tiled(size)
                else:
                    self._stage2_cholesky_batched_loop(size)

        # ══════════════════════════════════════════════════════════════
        # STAGE 3: Triangular solve + v_hat
        # ══════════════════════════════════════════════════════════════
        self._stage3_zero_qdd(state_aug)
        for size, ctx in self._for_sizes(enabled=self.use_parallel_streams):
            with ctx:
                use_tiled = (self.trisolve_kernel == "tiled") or (
                    self.trisolve_kernel == "auto" and size > self.small_dof_threshold
                )
                if use_tiled:
                    self._stage3_trisolve_batched_tiled(size, state_aug)
                else:
                    self._stage3_trisolve_batched_loop(size, state_aug)

        self._stage3_compute_v_hat(state_in, state_aug, dt)

        # ══════════════════════════════════════════════════════════════
        # STAGE 4: Build contact problem
        # ══════════════════════════════════════════════════════════════
        self._stage4_build_rows_batched(state_in, state_aug, contacts)

        fused_ok = (
            self._is_one_art_per_world
            and self.hinv_jt_kernel != "par_row"
            and all(
                (self.hinv_jt_kernel == "tiled") or (self.hinv_jt_kernel == "auto" and size > self.small_dof_threshold)
                for size in self.size_groups
            )
        )

        if fused_ok:
            for size, ctx in self._for_sizes(enabled=self.use_parallel_streams):
                with ctx:
                    self._stage4_hinv_jt_batched_tiled_fused(size)
        else:
            self._stage4_zero_world_C()

            for size, ctx in self._for_sizes(enabled=self.use_parallel_streams):
                with ctx:
                    use_tiled = (self.hinv_jt_kernel == "tiled") or (
                        self.hinv_jt_kernel == "auto" and size > self.small_dof_threshold
                    )
                    if use_tiled:
                        self._stage4_hinv_jt_batched_tiled(size)
                    else:
                        self._stage4_hinv_jt_batched_par_row(size)

            for size in self.size_groups:
                use_tiled_delassus = self.delassus_kernel != "par_row_col"
                if use_tiled_delassus:
                    self._stage4_delassus_batched_tiled(size)
                else:
                    self._stage4_delassus_batched_par_row_col(size)

            self._stage4_finalize_world_diag_cfm()

        self._stage4_compute_rhs_world(dt)

        for size in self.size_groups:
            self._stage4_accumulate_rhs_world(size)

        # ══════════════════════════════════════════════════════════════
        # STAGE 5: PGS solve
        # ══════════════════════════════════════════════════════════════
        self._stage5_prepare_impulses_world()
        if self.pgs_kernel == "tiled_row":
            self._stage5_pgs_solve_world_tiled_row()
        elif self.pgs_kernel == "tiled_contact":
            self._stage5_pgs_solve_world_tiled_contact()
        elif self.pgs_kernel == "streaming":
            self._stage5_pgs_solve_world_streaming()
        else:
            self._stage5_pgs_solve_world_loop()

        # ══════════════════════════════════════════════════════════════
        # STAGE 6: Apply impulses + integrate
        # ══════════════════════════════════════════════════════════════
        self._stage6_prepare_world_velocity()
        for size in self.size_groups:
            self._stage6_apply_impulses_world(size)
        self._stage6_update_qdd(state_in, state_aug, dt)

        self._stage6_integrate(state_in, state_aug, state_out, dt)

    def _stage1_fk_id(self, state_in: State, state_aug: State, state_out: State):
        model = self.model

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

        if model.body_count:
            wp.launch(
                update_body_qd_from_featherstone,
                dim=model.body_count,
                inputs=[state_aug.body_v_s, state_in.body_q, model.body_com],
                outputs=[state_out.body_qd],
                device=model.device,
            )

    def _stage1_drives(self, state_in: State, state_aug: State, control: Control, dt: float):
        model = self.model

        if model.articulation_count:
            body_f = state_in.body_f if state_in.body_count else None
            # evaluate joint torques
            state_aug.body_ft_s.zero_()
            wp.launch(
                eval_rigid_tau,
                dim=model.articulation_count,
                inputs=[
                    model.articulation_start,
                    model.joint_type,
                    model.joint_parent,
                    model.joint_child,
                    model.joint_qd_start,
                    model.joint_dof_dim,
                    control.joint_f,
                    state_aug.joint_S_s,
                    state_aug.body_f_s,
                    body_f,
                    state_in.body_q,
                    model.body_com,
                ],
                outputs=[
                    state_aug.body_ft_s,
                    state_aug.joint_tau,
                ],
                device=model.device,
            )

            self.build_augmented_joint_targets(state_in, control, dt)
            self.apply_augmented_joint_tau(state_in, state_aug, dt)

            wp.launch(
                clamp_joint_tau,
                dim=model.joint_dof_count,
                inputs=[state_aug.joint_tau, model.joint_effort_limit],
                device=model.device,
            )

    def build_augmented_joint_targets(self, state_in: State, control: Control, dt: float):
        model = self.model
        if model.articulation_count == 0 or self.articulation_max_dofs == 0:
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

    def _stage1_crba_flat(self, state_aug: State):
        model = self.model
        global_flag = 1 if ((self._step % self.update_mass_matrix_interval) == 0 or self._force_mass_update) else 0

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

        wp.launch(
            compute_composite_inertia,
            dim=model.articulation_count,
            inputs=[
                model.articulation_start,
                self.mass_update_mask,
                model.joint_ancestor,
                state_aug.body_I_s,
            ],
            outputs=[self.body_I_c],
            device=model.device,
            block_dim=128,
        )

        wp.launch(
            crba_fill_flat_par_dof,
            dim=model.articulation_count * self.articulation_max_dofs,
            inputs=[
                model.articulation_start,
                self.articulation_dof_start,
                self.articulation_H_start,
                self.articulation_H_rows,
                self.mass_update_mask,
                model.joint_ancestor,
                model.joint_qd_start,
                model.joint_dof_dim,
                state_aug.joint_S_s,
                self.body_I_c,
                self.articulation_max_dofs,
            ],
            outputs=[self.H_flat],
            device=model.device,
            block_dim=128,
        )

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
                self.H_flat,
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

    def _stage1_crba_batched(self, state_aug: State):
        model = self.model
        global_flag = 1 if ((self._step % self.update_mass_matrix_interval) == 0 or self._force_mass_update) else 0

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

        wp.launch(
            compute_composite_inertia,
            dim=model.articulation_count,
            inputs=[
                model.articulation_start,
                self.mass_update_mask,
                model.joint_ancestor,
                state_aug.body_I_s,
            ],
            outputs=[self.body_I_c],
            device=model.device,
            block_dim=128,
        )

        for size in self.size_groups:
            n_arts = self.n_arts_by_size[size]
            self.H_by_size[size].zero_()
            wp.launch(
                crba_fill_batched_par_dof,
                dim=int(n_arts * size),
                inputs=[
                    model.articulation_start,
                    self.articulation_dof_start,
                    self.mass_update_mask,
                    model.joint_ancestor,
                    model.joint_qd_start,
                    model.joint_dof_dim,
                    state_aug.joint_S_s,
                    self.body_I_c,
                    self.group_to_art[size],
                    size,
                ],
                outputs=[self.H_by_size[size]],
                device=model.device,
                block_dim=128,
            )

        for size in self.size_groups:
            n_arts = self.n_arts_by_size[size]
            wp.launch(
                apply_augmented_mass_diagonal_grouped,
                dim=n_arts,
                inputs=[
                    self.group_to_art[size],
                    self.articulation_dof_start,
                    size,
                    self.articulation_max_dofs,
                    self.mass_update_mask,
                    self.aug_row_counts,
                    self.aug_row_dof_index,
                    self.aug_row_K,
                ],
                outputs=[self.H_by_size[size]],
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

    def _stage2_cholesky_flat_tiled(self):
        model = self.model
        n_dofs = self.articulation_max_dofs
        H_tiled = self.H_flat.reshape((model.articulation_count, n_dofs, n_dofs))
        R_tiled = model.joint_armature.reshape((model.articulation_count, n_dofs))
        L_tiled = self.L_flat.reshape((model.articulation_count, n_dofs, n_dofs))

        cholesky_kernel = TiledKernelFactory.get_cholesky_flat_kernel(n_dofs, model.device)
        wp.launch_tiled(
            cholesky_kernel,
            dim=[model.articulation_count],
            inputs=[H_tiled, R_tiled, self.mass_update_mask],
            outputs=[L_tiled],
            block_dim=TILE_THREADS,
            device=model.device,
        )

    def _stage2_cholesky_flat_loop(self):
        model = self.model
        wp.launch(
            cholesky_flat_loop,
            dim=model.articulation_count,
            inputs=[
                self.articulation_H_start,
                self.articulation_H_rows,
                self.H_flat,
                model.joint_armature,
                self.mass_update_mask,
            ],
            outputs=[self.L_flat],
            device=model.device,
        )

    def _stage2_cholesky_batched_tiled(self, size: int):
        model = self.model
        n_arts = self.n_arts_by_size[size]
        cholesky_kernel = TiledKernelFactory.get_cholesky_kernel(size, model.device)
        wp.launch_tiled(
            cholesky_kernel,
            dim=[n_arts],
            inputs=[
                self.H_by_size[size],
                self.R_by_size[size],
                self.group_to_art[size],
                self.mass_update_mask,
            ],
            outputs=[self.L_by_size[size]],
            block_dim=TILE_THREADS,
            device=model.device,
        )

    def _stage2_cholesky_batched_loop(self, size: int):
        model = self.model
        n_arts = self.n_arts_by_size[size]
        wp.launch(
            cholesky_batched_loop,
            dim=n_arts,
            inputs=[
                self.H_by_size[size],
                self.R_by_size[size],
                self.group_to_art[size],
                self.mass_update_mask,
                size,
            ],
            outputs=[self.L_by_size[size]],
            device=model.device,
        )

    def _stage3_zero_qdd(self, state_aug: State):
        state_aug.joint_qdd.zero_()

    def _stage3_trisolve_flat_tiled(self, state_aug: State):
        model = self.model
        n_dofs = self.articulation_max_dofs
        L_tiled = self.L_flat.reshape((model.articulation_count, n_dofs, n_dofs))
        tau_tiled = state_aug.joint_tau.reshape((model.articulation_count, n_dofs, 1))
        qdd_tiled = state_aug.joint_qdd.reshape((model.articulation_count, n_dofs, 1))

        solve_kernel = TiledKernelFactory.get_triangular_solve_flat_kernel(n_dofs, model.device)
        wp.launch_tiled(
            solve_kernel,
            dim=[model.articulation_count],
            inputs=[
                L_tiled,
                tau_tiled,
            ],
            outputs=[qdd_tiled],
            block_dim=TILE_THREADS,
            device=model.device,
        )

    def _stage3_trisolve_flat_loop(self, state_aug: State):
        model = self.model
        wp.launch(
            trisolve_flat_loop,
            dim=model.articulation_count,
            inputs=[
                self.articulation_H_start,
                self.articulation_H_rows,
                self.articulation_dof_start,
                self.H_flat,
                self.L_flat,
                state_aug.joint_tau,
            ],
            outputs=[
                state_aug.joint_qdd,
                state_aug.joint_solve_tmp,
            ],
            device=model.device,
        )

    def _stage3_trisolve_batched_tiled(self, size: int, state_aug: State):
        model = self.model
        n_arts = self.n_arts_by_size[size]

        wp.launch(
            gather_tau_to_groups,
            dim=n_arts,
            inputs=[
                state_aug.joint_tau,
                self.group_to_art[size],
                self.articulation_dof_start,
                size,
            ],
            outputs=[self.tau_by_size[size]],
            device=model.device,
        )

        solve_kernel = TiledKernelFactory.get_triangular_solve_kernel(size, model.device)
        wp.launch_tiled(
            solve_kernel,
            dim=[n_arts],
            inputs=[
                self.L_by_size[size],
                self.tau_by_size[size],
            ],
            outputs=[self.qdd_by_size[size]],
            block_dim=TILE_THREADS,
            device=model.device,
        )

        wp.launch(
            scatter_qdd_from_groups,
            dim=n_arts,
            inputs=[
                self.qdd_by_size[size],
                self.group_to_art[size],
                self.articulation_dof_start,
                size,
            ],
            outputs=[state_aug.joint_qdd],
            device=model.device,
        )

    def _stage3_trisolve_batched_loop(self, size: int, state_aug: State):
        model = self.model
        n_arts = self.n_arts_by_size[size]
        wp.launch(
            trisolve_batched_loop,
            dim=n_arts,
            inputs=[
                self.L_by_size[size],
                self.group_to_art[size],
                self.articulation_dof_start,
                size,
                state_aug.joint_tau,
            ],
            outputs=[state_aug.joint_qdd],
            device=model.device,
        )

    def _stage3_compute_v_hat(self, state_in: State, state_aug: State, dt: float):
        model = self.model
        if not model.joint_count:
            return
        wp.launch(
            compute_velocity_predictor,
            dim=model.joint_dof_count,
            inputs=[
                state_in.joint_qd,
                state_aug.joint_qdd,
                dt,
            ],
            outputs=[self.v_hat],
            device=model.device,
        )

    def _stage4_build_rows_flat(self, state_in: State, state_aug: State, contacts: Contacts):
        model = self.model

        self.constraint_count_flat.zero_()

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
                    self.articulation_H_rows,
                    self.articulation_dof_start,
                    self.body_to_joint,
                    self.body_to_articulation,
                    model.joint_ancestor,
                    model.joint_qd_start,
                    state_aug.joint_S_s,
                    self.pgs_max_constraints,
                    self.articulation_max_dofs,
                    self.pgs_beta,
                    self.pgs_cfm,
                    enable_friction_flag,
                ],
                outputs=[
                    self.constraint_count_flat,
                    self.J_flat,
                    self.phi_flat,
                    self.row_beta_flat,
                    self.row_cfm_flat,
                    self.row_type_flat,
                    self.target_velocity_flat,
                    self.row_parent_flat,
                    self.row_mu_flat,
                ],
                device=model.device,
            )

        wp.launch(
            clamp_contact_counts,
            dim=model.articulation_count,
            inputs=[self.constraint_count_flat, self.pgs_max_constraints],
            device=model.device,
        )
        wp.launch(
            localize_parent_indices,
            dim=model.articulation_count,
            inputs=[
                self.constraint_count_flat,
                self.pgs_max_constraints,
                self.row_parent_flat,  # global/flat parent indices
            ],
            outputs=[
                self.row_parent_local_flat,  # local parent indices
            ],
            device=model.device,
        )

    def _stage4_build_rows_batched(self, state_in: State, state_aug: State, contacts: Contacts):
        model = self.model
        max_constraints = self.pgs_max_constraints

        # Zero world-level buffers
        self.slot_counter.zero_()
        self.constraint_count.zero_()

        for size in self.size_groups:
            self.J_by_size[size].zero_()

        if (
            contacts is not None
            and getattr(contacts, "rigid_contact_count", None) is not None
            and contacts.rigid_contact_max > 0
        ):
            enable_friction_flag = 1 if self.enable_contact_friction else 0

            wp.launch(
                allocate_world_contact_slots,
                dim=contacts.rigid_contact_max,
                inputs=[
                    contacts.rigid_contact_count,
                    contacts.rigid_contact_shape0,
                    contacts.rigid_contact_shape1,
                    contacts.rigid_contact_point0,
                    contacts.rigid_contact_point1,
                    contacts.rigid_contact_normal,
                    contacts.rigid_contact_thickness0,
                    contacts.rigid_contact_thickness1,
                    state_in.body_q,
                    model.shape_transform,
                    model.shape_body,
                    self.body_to_articulation,
                    self.art_to_world,
                    max_constraints,
                    enable_friction_flag,
                ],
                outputs=[
                    self.contact_world,
                    self.contact_slot,
                    self.contact_art_a,
                    self.contact_art_b,
                    self.slot_counter,
                ],
                device=model.device,
            )

            for size in self.size_groups:
                wp.launch(
                    populate_world_J_for_size,
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
                        self.contact_world,
                        self.contact_slot,
                        self.contact_art_a,
                        self.contact_art_b,
                        size,  # target_size
                        self.art_size,
                        self.art_group_idx,
                        self.articulation_dof_start,
                        self.body_to_joint,
                        model.joint_ancestor,
                        model.joint_qd_start,
                        state_aug.joint_S_s,
                        model.shape_body,
                        state_in.body_q,
                        model.shape_transform,
                        self.shape_material_mu,
                        enable_friction_flag,
                        self.pgs_beta,
                        self.pgs_cfm,
                    ],
                    outputs=[
                        self.J_by_size[size],
                        self.row_type,
                        self.row_parent,
                        self.row_mu,
                        self.row_beta,
                        self.row_cfm,
                        self.phi,
                        self.target_velocity,
                    ],
                    device=model.device,
                )

        wp.launch(
            finalize_world_constraint_counts,
            dim=self.world_count,
            inputs=[self.slot_counter, max_constraints],
            outputs=[self.constraint_count],
            device=model.device,
        )

    def _stage4_hinv_jt_flat_tiled(self):
        model = self.model
        n_dofs = self.articulation_max_dofs
        max_constraints = self.pgs_max_constraints
        L_tiled = self.L_flat.reshape((model.articulation_count, n_dofs, n_dofs))
        J_tiled = self.J_flat.reshape((model.articulation_count, max_constraints, n_dofs))
        Y_tiled = self.Y_flat.reshape((model.articulation_count, max_constraints, n_dofs))
        C_tiled = self.C_flat.reshape((model.articulation_count, max_constraints, max_constraints))

        hinv_jt_kernel = TiledKernelFactory.get_hinv_jt_flat_kernel(n_dofs, max_constraints, model.device)
        wp.launch_tiled(
            hinv_jt_kernel,
            dim=[model.articulation_count],
            inputs=[
                self.constraint_count_flat,
                L_tiled,
                J_tiled,
                self.row_cfm_flat,
            ],
            outputs=[
                Y_tiled,
                C_tiled,
                self.diag_flat,
            ],
            block_dim=TILE_THREADS,
            device=model.device,
        )

    def _stage4_hinv_jt_flat_par_row(self):
        model = self.model
        wp.launch(
            hinv_jt_flat_par_row,
            dim=model.articulation_count * self.pgs_max_constraints,
            inputs=[
                self.articulation_H_start,
                self.articulation_H_rows,
                self.pgs_max_constraints,
                self.articulation_max_dofs,
                self.constraint_count_flat,
                self.L_flat,
                self.J_flat,
            ],
            outputs=[self.Y_flat],
            device=model.device,
        )

    def _stage4_delassus_flat_par_row(self):
        model = self.model
        wp.launch(
            delassus_flat_par_row,
            dim=model.articulation_count * self.pgs_max_constraints,
            inputs=[
                self.articulation_H_rows,
                self.pgs_max_constraints,
                self.articulation_max_dofs,
                self.constraint_count_flat,
                self.J_flat,
                self.Y_flat,
                self.row_cfm_flat,
            ],
            outputs=[
                self.diag_flat,
                self.C_flat,
            ],
            device=model.device,
        )

    def _stage4_zero_world_C(self):
        self.C.zero_()
        self.diag.zero_()

    def _stage4_hinv_jt_batched_tiled(self, size: int):
        model = self.model
        n_arts = self.n_arts_by_size[size]
        hinv_jt_kernel = TiledKernelFactory.get_hinv_jt_kernel(size, self.pgs_max_constraints, model.device)
        wp.launch_tiled(
            hinv_jt_kernel,
            dim=[n_arts],
            inputs=[
                self.L_by_size[size],
                self.J_by_size[size],
                self.group_to_art[size],
                self.art_to_world,
                self.constraint_count,
            ],
            outputs=[self.Y_by_size[size]],
            block_dim=TILE_THREADS,
            device=model.device,
        )

    def _stage4_hinv_jt_batched_tiled_fused(self, size: int):
        model = self.model
        n_arts = self.n_arts_by_size[size]
        hinv_jt_kernel = TiledKernelFactory.get_hinv_jt_fused_kernel(size, self.pgs_max_constraints, model.device)
        wp.launch_tiled(
            hinv_jt_kernel,
            dim=[n_arts],
            inputs=[
                self.L_by_size[size],
                self.J_by_size[size],
                self.group_to_art[size],
                self.art_to_world,
                self.constraint_count,
                self.row_cfm,
            ],
            outputs=[self.C, self.diag, self.Y_by_size[size]],
            block_dim=TILE_THREADS,
            device=model.device,
        )

    def _stage4_hinv_jt_batched_par_row(self, size: int):
        model = self.model
        n_arts = self.n_arts_by_size[size]
        wp.launch(
            hinv_jt_batched_par_row,
            dim=n_arts * self.pgs_max_constraints,
            inputs=[
                self.L_by_size[size],
                self.J_by_size[size],
                self.group_to_art[size],
                self.art_to_world,
                self.constraint_count,
                size,
                self.pgs_max_constraints,
                n_arts,
            ],
            outputs=[self.Y_by_size[size]],
            device=model.device,
        )

    def _stage4_delassus_batched_par_row_col(self, size: int):
        model = self.model
        n_arts = self.n_arts_by_size[size]
        wp.launch(
            delassus_batched_par_row_col,
            dim=n_arts * self.pgs_max_constraints * self.pgs_max_constraints,
            inputs=[
                self.J_by_size[size],
                self.Y_by_size[size],
                self.group_to_art[size],
                self.art_to_world,
                self.constraint_count,
                size,
                self.pgs_max_constraints,
                n_arts,
            ],
            outputs=[self.C, self.diag],
            device=model.device,
        )

    def _stage4_delassus_batched_tiled(self, size: int):
        model = self.model
        n_arts = self.n_arts_by_size[size]
        delassus_kernel = TiledKernelFactory.get_delassus_kernel(
            size, self.pgs_max_constraints, model.device, chunk_size=self.delassus_chunk_size
        )
        wp.launch_tiled(
            delassus_kernel,
            dim=[n_arts],
            inputs=[
                self.J_by_size[size],
                self.Y_by_size[size],
                self.group_to_art[size],
                self.art_to_world,
                self.constraint_count,
                n_arts,
            ],
            outputs=[self.C, self.diag],
            block_dim=128,
            device=model.device,
        )

    def _stage4_finalize_world_diag_cfm(self):
        model = self.model
        wp.launch(
            finalize_world_diag_cfm,
            dim=self.world_count,
            inputs=[self.constraint_count, self.row_cfm],
            outputs=[self.diag],
            device=model.device,
        )

    def _stage4_compute_rhs_flat(self, dt: float):
        model = self.model
        wp.launch(
            contact_bias_flat_par_row,
            dim=model.articulation_count * self.pgs_max_constraints,
            inputs=[
                self.articulation_dof_start,
                self.articulation_H_rows,
                self.constraint_count_flat,
                model.articulation_count,
                self.pgs_max_constraints,
                self.articulation_max_dofs,
                self.J_flat,
                self.v_hat,
                self.phi_flat,
                self.row_beta_flat,
                self.row_type_flat,
                self.target_velocity_flat,
                dt,
            ],
            outputs=[self.rhs_flat],
            device=model.device,
        )

    def _stage4_compute_rhs_world(self, dt: float):
        model = self.model
        wp.launch(
            compute_world_contact_bias,
            dim=self.world_count,
            inputs=[
                self.constraint_count,
                self.pgs_max_constraints,
                self.phi,
                self.row_beta,
                self.row_type,
                self.target_velocity,
                dt,
            ],
            outputs=[self.rhs],
            device=model.device,
        )

    def _stage4_accumulate_rhs_world(self, size: int):
        model = self.model
        n_arts = self.n_arts_by_size[size]
        wp.launch(
            rhs_accum_world_par_art,
            dim=n_arts,
            inputs=[
                self.constraint_count,
                self.pgs_max_constraints,
                self.art_to_world,
                self.art_size,
                self.art_group_idx,
                self.articulation_dof_start,
                self.v_hat,
                self.group_to_art[size],
                self.J_by_size[size],
                size,
            ],
            outputs=[self.rhs],
            device=model.device,
        )

    def _stage5_prepare_impulses_flat(self):
        model = self.model
        warmstart_flag = 1 if self.pgs_warmstart else 0
        wp.launch(
            prepare_impulses,
            dim=model.articulation_count,
            inputs=[
                self.constraint_count_flat,
                self.pgs_max_constraints,
                warmstart_flag,
                self.impulses_flat,
            ],
            device=model.device,
        )

    def _stage5_prepare_impulses_world(self):
        warmstart_flag = 1 if self.pgs_warmstart else 0
        wp.launch(
            prepare_world_impulses,
            dim=self.world_count,
            inputs=[self.constraint_count, self.pgs_max_constraints, warmstart_flag],
            outputs=[self.impulses],
            device=self.model.device,
        )

    def _stage5_pgs_solve_flat_tiled_row(self):
        model = self.model
        M = self.pgs_max_constraints
        A = model.articulation_count
        diag2 = self.diag_flat.reshape((A, M))
        rhs2 = self.rhs_flat.reshape((A, M))
        lam2 = self.impulses_flat.reshape((A, M))
        C3 = self.C_flat.reshape((A, M, M))
        rtype2 = self.row_type_flat.reshape((A, M))
        mu2 = self.row_mu_flat.reshape((A, M))
        parent2_local = self.row_parent_local_flat.reshape((A, M))

        pgs_kernel = TiledKernelFactory.get_pgs_solve_tiled_row_kernel(M, model.device)
        wp.launch_tiled(
            pgs_kernel,
            dim=[A],
            inputs=[
                self.constraint_count_flat,
                diag2,
                C3,
                rhs2,
                lam2,
                self.pgs_iterations,
                self.pgs_omega,
                rtype2,
                parent2_local,
                mu2,
            ],
            block_dim=32,
            device=model.device,
        )

    def _stage5_pgs_solve_flat_loop(self):
        model = self.model
        M = self.pgs_max_constraints
        A = model.articulation_count
        diag2 = self.diag_flat.reshape((A, M))
        rhs2 = self.rhs_flat.reshape((A, M))
        lam2 = self.impulses_flat.reshape((A, M))
        C3 = self.C_flat.reshape((A, M, M))
        rtype2 = self.row_type_flat.reshape((A, M))
        mu2 = self.row_mu_flat.reshape((A, M))
        parent2_local = self.row_parent_local_flat.reshape((A, M))

        wp.launch(
            pgs_solve_loop,
            dim=A,
            inputs=[
                self.constraint_count_flat,
                M,
                diag2,
                C3,
                rhs2,
                lam2,
                self.pgs_iterations,
                self.pgs_omega,
                rtype2,
                parent2_local,
                mu2,
            ],
            device=model.device,
        )

    def _stage5_pgs_solve_world_tiled_row(self):
        pgs_kernel = TiledKernelFactory.get_pgs_solve_tiled_row_kernel(self.pgs_max_constraints, self.model.device)
        wp.launch_tiled(
            pgs_kernel,
            dim=[self.world_count],
            inputs=[
                self.constraint_count,
                self.diag,
                self.C,
                self.rhs,
                self.impulses,
                self.pgs_iterations,
                self.pgs_omega,
                self.row_type,
                self.row_parent,
                self.row_mu,
            ],
            block_dim=32,
            device=self.model.device,
        )

    def _stage5_pgs_solve_world_loop(self):
        wp.launch(
            pgs_solve_loop,
            dim=self.world_count,
            inputs=[
                self.constraint_count,
                self.pgs_max_constraints,
                self.diag,
                self.C,
                self.rhs,
                self.impulses,
                self.pgs_iterations,
                self.pgs_omega,
                self.row_type,
                self.row_parent,
                self.row_mu,
            ],
            device=self.model.device,
        )

    def _stage5_pgs_solve_flat_tiled_contact(self):
        model = self.model
        M = self.pgs_max_constraints
        A = model.articulation_count
        rhs2 = self.rhs_flat.reshape((A, M))
        lam2 = self.impulses_flat.reshape((A, M))
        C3 = self.C_flat.reshape((A, M, M))
        mu2 = self.row_mu_flat.reshape((A, M))

        pgs_kernel = TiledKernelFactory.get_pgs_solve_tiled_contact_kernel(M, model.device)
        wp.launch_tiled(
            pgs_kernel,
            dim=[A],
            inputs=[
                self.constraint_count_flat,
                C3,
                rhs2,
                lam2,
                self.pgs_iterations,
                self.pgs_omega,
                mu2,
            ],
            block_dim=32,
            device=model.device,
        )

    def _stage5_pgs_solve_world_tiled_contact(self):
        pgs_kernel = TiledKernelFactory.get_pgs_solve_tiled_contact_kernel(
            self.pgs_max_constraints, self.model.device
        )
        wp.launch_tiled(
            pgs_kernel,
            dim=[self.world_count],
            inputs=[
                self.constraint_count,
                self.C,
                self.rhs,
                self.impulses,
                self.pgs_iterations,
                self.pgs_omega,
                self.row_mu,
            ],
            block_dim=32,
            device=self.model.device,
        )

    def _stage5_pgs_solve_flat_streaming(self):
        model = self.model
        M = self.pgs_max_constraints
        A = model.articulation_count
        rhs2 = self.rhs_flat.reshape((A, M))
        lam2 = self.impulses_flat.reshape((A, M))
        C3 = self.C_flat.reshape((A, M, M))
        mu2 = self.row_mu_flat.reshape((A, M))

        pgs_kernel = TiledKernelFactory.get_pgs_solve_streaming_kernel(M, model.device)
        wp.launch_tiled(
            pgs_kernel,
            dim=[A],
            inputs=[
                self.constraint_count_flat,
                C3,
                rhs2,
                lam2,
                self.pgs_iterations,
                self.pgs_omega,
                mu2,
            ],
            block_dim=32,
            device=model.device,
        )

    def _stage5_pgs_solve_world_streaming(self):
        pgs_kernel = TiledKernelFactory.get_pgs_solve_streaming_kernel(
            self.pgs_max_constraints, self.model.device, pgs_chunk_size=self.pgs_chunk_size
        )
        wp.launch_tiled(
            pgs_kernel,
            dim=[self.world_count],
            inputs=[
                self.constraint_count,
                self.C,
                self.rhs,
                self.impulses,
                self.pgs_iterations,
                self.pgs_omega,
                self.row_mu,
            ],
            block_dim=32,
            device=self.model.device,
        )

    def _stage6_apply_impulses_flat(self):
        model = self.model
        wp.launch(
            apply_impulses_flat_par_dof,
            dim=model.articulation_count * self.articulation_max_dofs,
            inputs=[
                self.articulation_dof_start,
                self.articulation_H_rows,
                self.constraint_count_flat,
                self.pgs_max_constraints,
                self.articulation_max_dofs,
                self.Y_flat,
                self.v_hat,
                self.impulses_flat,
            ],
            outputs=[self.v_out],
            device=model.device,
        )

    def _stage6_prepare_world_velocity(self):
        wp.copy(self.v_out, self.v_hat)

    def _stage6_apply_impulses_world(self, size: int):
        model = self.model
        n_arts = self.n_arts_by_size[size]
        wp.launch(
            apply_impulses_world_par_dof,
            dim=int(n_arts * size),
            inputs=[
                self.group_to_art[size],
                self.art_to_world,
                self.articulation_dof_start,
                size,
                n_arts,
                self.constraint_count,
                self.pgs_max_constraints,
                self.Y_by_size[size],
                self.impulses,
                self.v_hat,
            ],
            outputs=[self.v_out],
            device=model.device,
        )

    def _stage6_update_qdd(self, state_in: State, state_aug: State, dt: float):
        model = self.model
        wp.launch(
            update_qdd_from_velocity,
            dim=model.joint_dof_count,
            inputs=[state_in.joint_qd, self.v_out, 1.0 / dt],
            outputs=[state_aug.joint_qdd],
            device=model.device,
        )

    def _stage6_integrate(self, state_in: State, state_aug: State, state_out: State, dt: float):
        model = self.model

        if model.joint_count:
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


class TiledKernelFactory:
    """Factory for generating size-specialized tiled kernels for heterogeneous multi-articulation.

    This factory generates and caches tiled kernels specialized for specific DOF counts,
    enabling optimal tiled operations (Cholesky, triangular solves) for articulations
    with different numbers of degrees of freedom.

    The pattern follows ik_lbfgs_optimizer.py: kernels are generated on-demand with
    wp.constant() captured via closure, then cached by (dimensions, device.arch).
    """

    # Class-level caches: key -> compiled kernel
    _hinv_jt_flat_cache: ClassVar[dict[tuple[int, int, str], "wp.Kernel"]] = {}
    _cholesky_flat_cache: ClassVar[dict[tuple[int, str], "wp.Kernel"]] = {}
    _triangular_solve_flat_cache: ClassVar[dict[tuple[int, str], "wp.Kernel"]] = {}
    _hinv_jt_cache: ClassVar[dict[tuple[int, int, str], "wp.Kernel"]] = {}
    _hinv_jt_fused_cache: ClassVar[dict[tuple[int, int, str], "wp.Kernel"]] = {}
    _cholesky_cache: ClassVar[dict[tuple[int, str], "wp.Kernel"]] = {}
    _pgs_solve_tiled_row_cache: ClassVar[dict[tuple[int, str], "wp.Kernel"]] = {}
    _pgs_solve_tiled_contact_cache: ClassVar[dict[tuple[int, str], "wp.Kernel"]] = {}
    _pgs_solve_streaming_cache: ClassVar[dict[tuple[int, str], "wp.Kernel"]] = {}
    _triangular_solve_cache: ClassVar[dict[tuple[int, str], "wp.Kernel"]] = {}
    _delassus_cache: ClassVar[dict[tuple[int, int, str], "wp.Kernel"]] = {}

    @classmethod
    def get_hinv_jt_flat_kernel(cls, n_dofs: int, max_constraints: int, device: "wp.Device") -> "wp.Kernel":
        """Get or create a tiled flat H^-1*J^T kernel for the given dimensions."""
        key = (n_dofs, max_constraints, device.arch)
        if key not in cls._hinv_jt_flat_cache:
            cls._hinv_jt_flat_cache[key] = cls._build_hinv_jt_flat_kernel(n_dofs, max_constraints)
        return cls._hinv_jt_flat_cache[key]

    @classmethod
    def _build_hinv_jt_flat_kernel(cls, n_dofs: int, max_constraints: int) -> "wp.Kernel":
        """Build specialized flat H^-1*J^T kernel for given dimensions."""
        dofs = int(n_dofs)
        max_c = int(max_constraints)
        TILE_DOF_LOCAL = wp.constant(dofs)
        TILE_CONSTRAINTS_LOCAL = wp.constant(max_c)

        def hinv_jt_flat_tiled_template(
            constraint_counts: wp.array(dtype=int),
            L: wp.array3d(dtype=float),  # [arts, n_dofs, n_dofs]
            J_rows: wp.array3d(dtype=float),  # [arts, max_c, n_dofs]
            row_cfm: wp.array(dtype=float),
            # outputs
            Y: wp.array3d(dtype=float),  # [arts, max_c, n_dofs]
            C: wp.array3d(dtype=float),  # [arts, max_c, max_c]
            diag_out: wp.array(dtype=float),
        ):
            art, thread = wp.tid()
            constraint_count = constraint_counts[art]

            if constraint_count == 0:
                return

            L_tile = wp.tile_load(L[art], shape=(TILE_DOF_LOCAL, TILE_DOF_LOCAL), bounds_check=False)
            J_tile = wp.tile_load(J_rows[art], shape=(TILE_CONSTRAINTS_LOCAL, TILE_DOF_LOCAL), bounds_check=False)

            # Solve L * Z = J^T (forward substitution)
            Z_tile = wp.tile_lower_solve(L_tile, wp.tile_transpose(J_tile))

            # Solve L^T * X = Z (backward substitution)
            U_tile = wp.tile_transpose(L_tile)
            X_tile = wp.tile_upper_solve(U_tile, Z_tile)

            C_tile = wp.tile_zeros(shape=(TILE_CONSTRAINTS_LOCAL, TILE_CONSTRAINTS_LOCAL), dtype=wp.float32)

            # Store Y = H^-1 * J^T (transpose back to row layout)
            wp.tile_store(Y[art], wp.tile_transpose(X_tile))

            # Form C = J * H^-1 * J^T
            wp.tile_matmul(J_tile, X_tile, C_tile)
            wp.tile_store(C[art], C_tile)

            if thread == 0:
                diag_base = art * max_c
                for i in range(constraint_count):
                    diag_out[diag_base + i] = C_tile[i, i] + row_cfm[diag_base + i]

        hinv_jt_flat_tiled_template.__name__ = f"hinv_jt_flat_tiled_{dofs}_{max_c}"
        hinv_jt_flat_tiled_template.__qualname__ = f"hinv_jt_flat_tiled_{dofs}_{max_c}"
        return wp.kernel(enable_backward=False, module="unique")(hinv_jt_flat_tiled_template)

    @classmethod
    def get_cholesky_flat_kernel(cls, n_dofs: int, device: "wp.Device") -> "wp.Kernel":
        """Get or create a tiled flat Cholesky kernel for the given DOF count."""
        key = (n_dofs, device.arch)
        if key not in cls._cholesky_flat_cache:
            cls._cholesky_flat_cache[key] = cls._build_cholesky_flat_kernel(n_dofs)
        return cls._cholesky_flat_cache[key]

    @classmethod
    def _build_cholesky_flat_kernel(cls, n_dofs: int) -> "wp.Kernel":
        """Build specialized flat Cholesky kernel for given DOF count."""
        dofs = int(n_dofs)
        TILE_DOF_LOCAL = wp.constant(dofs)

        def cholesky_flat_tiled_template(
            H: wp.array3d(dtype=float),  # [arts, n_dofs, n_dofs]
            R: wp.array2d(dtype=float),  # [arts, n_dofs]
            mass_update_mask: wp.array(dtype=int),
            # output
            L: wp.array3d(dtype=float),  # [arts, n_dofs, n_dofs]
        ):
            art = wp.tid()

            if mass_update_mask[art] == 0:
                return

            H_tile = wp.tile_load(H[art], shape=(TILE_DOF_LOCAL, TILE_DOF_LOCAL), bounds_check=False)
            armature = wp.tile_load(R[art], shape=(TILE_DOF_LOCAL,), bounds_check=False)

            H_tile = wp.tile_diag_add(H_tile, armature)
            L_tile = wp.tile_cholesky(H_tile)

            wp.tile_store(L[art], L_tile)

        cholesky_flat_tiled_template.__name__ = f"cholesky_flat_tiled_{dofs}"
        cholesky_flat_tiled_template.__qualname__ = f"cholesky_flat_tiled_{dofs}"
        return wp.kernel(enable_backward=False, module="unique")(cholesky_flat_tiled_template)

    @classmethod
    def get_triangular_solve_flat_kernel(cls, n_dofs: int, device: "wp.Device") -> "wp.Kernel":
        """Get or create a tiled flat triangular solve kernel for the given DOF count."""
        key = (n_dofs, device.arch)
        if key not in cls._triangular_solve_flat_cache:
            cls._triangular_solve_flat_cache[key] = cls._build_triangular_solve_flat_kernel(n_dofs)
        return cls._triangular_solve_flat_cache[key]

    @classmethod
    def _build_triangular_solve_flat_kernel(cls, n_dofs: int) -> "wp.Kernel":
        """Build specialized flat triangular solve kernel for given DOF count."""
        dofs = int(n_dofs)
        TILE_DOF_LOCAL = wp.constant(dofs)

        def trisolve_flat_tiled_template(
            L: wp.array3d(dtype=float),  # [arts, n_dofs, n_dofs]
            tau: wp.array3d(dtype=float),  # [arts, n_dofs, 1]
            # output
            qdd: wp.array3d(dtype=float),  # [arts, n_dofs, 1]
        ):
            art = wp.tid()

            L_tile = wp.tile_load(L[art], shape=(TILE_DOF_LOCAL, TILE_DOF_LOCAL), bounds_check=False)
            tau_tile = wp.tile_load(tau[art], shape=(TILE_DOF_LOCAL, 1), bounds_check=False)

            z_tile = wp.tile_lower_solve(L_tile, tau_tile)
            Lt_tile = wp.tile_transpose(L_tile)
            qdd_tile = wp.tile_upper_solve(Lt_tile, z_tile)

            wp.tile_store(qdd[art], qdd_tile)

        trisolve_flat_tiled_template.__name__ = f"trisolve_flat_tiled_{dofs}"
        trisolve_flat_tiled_template.__qualname__ = f"trisolve_flat_tiled_{dofs}"
        return wp.kernel(enable_backward=False, module="unique")(trisolve_flat_tiled_template)

    @classmethod
    def get_hinv_jt_kernel(cls, n_dofs: int, max_constraints: int, device: "wp.Device") -> "wp.Kernel":
        """Get or create a tiled H^-1*J^T kernel for the given dimensions."""
        key = (n_dofs, max_constraints, device.arch)
        if key not in cls._hinv_jt_cache:
            cls._hinv_jt_cache[key] = cls._build_hinv_jt_kernel(n_dofs, max_constraints)
        return cls._hinv_jt_cache[key]

    @classmethod
    def get_hinv_jt_fused_kernel(cls, n_dofs: int, max_constraints: int, device: "wp.Device") -> "wp.Kernel":
        """Get or create a tiled fused H^-1*J^T + Delassus kernel for the given dimensions."""
        key = (n_dofs, max_constraints, device.arch)
        if key not in cls._hinv_jt_fused_cache:
            cls._hinv_jt_fused_cache[key] = cls._build_hinv_jt_fused_kernel(n_dofs, max_constraints)
        return cls._hinv_jt_fused_cache[key]

    @classmethod
    def _build_hinv_jt_kernel(cls, n_dofs: int, max_constraints: int) -> "wp.Kernel":
        """Build specialized H^-1*J^T kernel for given dimensions.

        Solves Y = H^-1 * J^T using tiled Cholesky solve:
          L * L^T * Y = J^T
          => L * Z = J^T (forward solve)
          => L^T * Y = Z (backward solve)
        """
        # Create compile-time constants via closure
        # Convert to Python int to ensure wp.constant() accepts them
        TILE_DOF_LOCAL = wp.constant(int(n_dofs))
        TILE_CONSTRAINTS_LOCAL = wp.constant(int(max_constraints))

        def hinv_jt_batched_tiled_template(
            L_group: wp.array3d(dtype=float),  # [n_arts, n_dofs, n_dofs]
            J_group: wp.array3d(dtype=float),  # [n_arts, max_c, n_dofs]
            group_to_art: wp.array(dtype=int),
            art_to_world: wp.array(dtype=int),
            world_constraint_count: wp.array(dtype=int),
            # output
            Y_group: wp.array3d(dtype=float),  # [n_arts, max_c, n_dofs]
        ):
            idx = wp.tid()
            art = group_to_art[idx]
            world = art_to_world[art]
            n_constraints = world_constraint_count[world]

            if n_constraints == 0:
                return

            # Load L (Cholesky factor) and J (Jacobian rows)
            L_tile = wp.tile_load(L_group[idx], shape=(TILE_DOF_LOCAL, TILE_DOF_LOCAL), bounds_check=False)
            J_tile = wp.tile_load(J_group[idx], shape=(TILE_CONSTRAINTS_LOCAL, TILE_DOF_LOCAL), bounds_check=False)

            # Solve L * Z = J^T (forward substitution)
            # J_tile is (max_c x n_dofs), J^T is (n_dofs x max_c)
            Jt_tile = wp.tile_transpose(J_tile)
            Z_tile = wp.tile_lower_solve(L_tile, Jt_tile)

            # Solve L^T * Y = Z (backward substitution)
            Lt_tile = wp.tile_transpose(L_tile)
            X_tile = wp.tile_upper_solve(Lt_tile, Z_tile)

            # Store Y = H^-1 * J^T (transpose back to row layout)
            Y_out_tile = wp.tile_transpose(X_tile)
            wp.tile_store(Y_group[idx], Y_out_tile)

        hinv_jt_batched_tiled_template.__name__ = f"hinv_jt_batched_tiled_{n_dofs}_{max_constraints}"
        hinv_jt_batched_tiled_template.__qualname__ = f"hinv_jt_batched_tiled_{n_dofs}_{max_constraints}"
        return wp.kernel(enable_backward=False, module="unique")(hinv_jt_batched_tiled_template)

    @classmethod
    def _build_hinv_jt_fused_kernel(cls, n_dofs: int, max_constraints: int) -> "wp.Kernel":
        """Build specialized fused H^-1*J^T + Delassus kernel for given dimensions."""
        TILE_DOF_LOCAL = wp.constant(int(n_dofs))
        TILE_CONSTRAINTS_LOCAL = wp.constant(int(max_constraints))

        def hinv_jt_batched_tiled_fused_template(
            L_group: wp.array3d(dtype=float),  # [n_arts, n_dofs, n_dofs]
            J_group: wp.array3d(dtype=float),  # [n_arts, max_c, n_dofs]
            group_to_art: wp.array(dtype=int),
            art_to_world: wp.array(dtype=int),
            world_constraint_count: wp.array(dtype=int),
            row_cfm: wp.array2d(dtype=float),
            # outputs
            world_C: wp.array3d(dtype=float),  # [world_count, max_c, max_c]
            world_diag: wp.array2d(dtype=float),  # [world_count, max_c]
            Y_group: wp.array3d(dtype=float),  # [n_arts, max_c, n_dofs]
        ):
            idx, thread = wp.tid()
            art = group_to_art[idx]
            world = art_to_world[art]
            n_constraints = world_constraint_count[world]

            if n_constraints == 0:
                return

            # Load L (Cholesky factor) and J (Jacobian rows)
            L_tile = wp.tile_load(L_group[idx], shape=(TILE_DOF_LOCAL, TILE_DOF_LOCAL), bounds_check=False)
            J_tile = wp.tile_load(J_group[idx], shape=(TILE_CONSTRAINTS_LOCAL, TILE_DOF_LOCAL), bounds_check=False)

            # Solve L * Z = J^T (forward substitution)
            Jt_tile = wp.tile_transpose(J_tile)
            Z_tile = wp.tile_lower_solve(L_tile, Jt_tile)

            # Solve L^T * Y = Z (backward substitution)
            Lt_tile = wp.tile_transpose(L_tile)
            X_tile = wp.tile_upper_solve(Lt_tile, Z_tile)

            # Store Y = H^-1 * J^T (transpose back to row layout)
            Y_out_tile = wp.tile_transpose(X_tile)
            wp.tile_store(Y_group[idx], Y_out_tile)

            # Form C = J * H^-1 * J^T
            C_tile = wp.tile_zeros(shape=(TILE_CONSTRAINTS_LOCAL, TILE_CONSTRAINTS_LOCAL), dtype=wp.float32)
            wp.tile_matmul(J_tile, X_tile, C_tile)
            wp.tile_store(world_C[world], C_tile)

            if thread == 0:
                for i in range(n_constraints):
                    world_diag[world, i] = C_tile[i, i] + row_cfm[world, i]

        hinv_jt_batched_tiled_fused_template.__name__ = f"hinv_jt_batched_tiled_fused_{n_dofs}_{max_constraints}"
        hinv_jt_batched_tiled_fused_template.__qualname__ = f"hinv_jt_batched_tiled_fused_{n_dofs}_{max_constraints}"
        return wp.kernel(enable_backward=False, module="unique")(hinv_jt_batched_tiled_fused_template)

    @classmethod
    def get_delassus_kernel(
        cls, n_dofs: int, max_constraints: int, device: "wp.Device", chunk_size: int | None = None
    ) -> "wp.Kernel":
        """Get or create a streaming Delassus kernel for the given dimensions."""
        key = (n_dofs, max_constraints, device.arch, chunk_size)
        if key not in cls._delassus_cache:
            cls._delassus_cache[key] = cls._build_delassus_kernel(n_dofs, max_constraints, chunk_size)
        return cls._delassus_cache[key]

    @classmethod
    def _build_delassus_kernel(
        cls, n_dofs: int, max_constraints: int, chunk_size: int | None = None
    ) -> "wp.Kernel":
        """Streaming Delassus: C += J * Y^T with shared memory."""
        TILE_D = n_dofs
        TILE_M = max_constraints
        if chunk_size is not None:
            CHUNK = chunk_size
        else:
            CHUNK = 64 if (2 * TILE_M * TILE_D * 4 > 45000) else TILE_M

        snippet = f"""
#if defined(__CUDA_ARCH__)
    const int TILE_D = {TILE_D};
    const int TILE_M = {TILE_M};
    const int CHUNK = {CHUNK};

    int lane = threadIdx.x;
    int art = group_to_art.data[idx];
    int world = art_to_world.data[art];
    int m = world_constraint_count.data[world];
    if (m == 0) return;

    __shared__ float s_J[CHUNK * TILE_D];
    __shared__ float s_Y[CHUNK * TILE_D];

    int num_chunks = (m + CHUNK - 1) / CHUNK;

    for (int ci = 0; ci < num_chunks; ci++) {{
        int i0 = ci * CHUNK, i1 = min(i0 + CHUNK, m);

        for (int t = lane; t < (i1 - i0) * TILE_D; t += blockDim.x)
            s_J[t] = J_group.data[idx * TILE_M * TILE_D + i0 * TILE_D + t];
        __syncthreads();

        for (int cj = 0; cj < num_chunks; cj++) {{
            int j0 = cj * CHUNK, j1 = min(j0 + CHUNK, m);

            for (int t = lane; t < (j1 - j0) * TILE_D; t += blockDim.x)
                s_Y[t] = Y_group.data[idx * TILE_M * TILE_D + j0 * TILE_D + t];
            __syncthreads();

            // Each thread computes multiple C elements
            for (int e = lane; e < (i1 - i0) * (j1 - j0); e += blockDim.x) {{
                int il = e / (j1 - j0), jl = e % (j1 - j0);
                float sum = 0.0f;
                for (int k = 0; k < TILE_D; k++)
                    sum += s_J[il * TILE_D + k] * s_Y[jl * TILE_D + k];
                if (sum != 0.0f) {{
                    int ig = i0 + il, jg = j0 + jl;
                    atomicAdd(&world_C.data[world * TILE_M * TILE_M + ig * TILE_M + jg], sum);
                    if (ig == jg) atomicAdd(&world_diag.data[world * TILE_M + ig], sum);
                }}
            }}
            __syncthreads();
        }}
    }}
#endif
"""

        @wp.func_native(snippet)
        def delassus_native(
            idx: int,
            J_group: wp.array3d(dtype=float),
            Y_group: wp.array3d(dtype=float),
            group_to_art: wp.array(dtype=int),
            art_to_world: wp.array(dtype=int),
            world_constraint_count: wp.array(dtype=int),
            world_C: wp.array3d(dtype=float),
            world_diag: wp.array2d(dtype=float),
        ): ...

        def delassus_template(
            J_group: wp.array3d(dtype=float),
            Y_group: wp.array3d(dtype=float),
            group_to_art: wp.array(dtype=int),
            art_to_world: wp.array(dtype=int),
            world_constraint_count: wp.array(dtype=int),
            n_arts: int,
            world_C: wp.array3d(dtype=float),
            world_diag: wp.array2d(dtype=float),
        ):
            idx, _lane = wp.tid()
            if idx < n_arts:
                delassus_native(
                    idx, J_group, Y_group, group_to_art, art_to_world, world_constraint_count, world_C, world_diag
                )

        delassus_template.__name__ = f"delassus_streaming_{n_dofs}_{max_constraints}_chunk{CHUNK}"
        delassus_template.__qualname__ = f"delassus_streaming_{n_dofs}_{max_constraints}_chunk{CHUNK}"
        return wp.kernel(enable_backward=False, module="unique")(delassus_template)

    @classmethod
    def get_cholesky_kernel(cls, n_dofs: int, device: "wp.Device") -> "wp.Kernel":
        """Get or create a tiled Cholesky kernel for the given DOF count."""
        key = (n_dofs, device.arch)
        if key not in cls._cholesky_cache:
            cls._cholesky_cache[key] = cls._build_cholesky_kernel(n_dofs)
        return cls._cholesky_cache[key]

    @classmethod
    def _build_cholesky_kernel(cls, n_dofs: int) -> "wp.Kernel":
        """Build specialized Cholesky kernel for given DOF count.

        Computes L such that H + diag(armature) = L * L^T.
        """
        # Convert to Python int to ensure wp.constant() accepts them
        TILE_DOF_LOCAL = wp.constant(int(n_dofs))

        def cholesky_batched_tiled_template(
            H_group: wp.array3d(dtype=float),  # [n_arts, n_dofs, n_dofs]
            R_group: wp.array2d(dtype=float),  # [n_arts, n_dofs] armature
            group_to_art: wp.array(dtype=int),
            mass_update_mask: wp.array(dtype=int),
            # output
            L_group: wp.array3d(dtype=float),  # [n_arts, n_dofs, n_dofs]
        ):
            idx = wp.tid()
            art = group_to_art[idx]

            if mass_update_mask[art] == 0:
                return

            # Load H and armature
            H_tile = wp.tile_load(H_group[idx], shape=(TILE_DOF_LOCAL, TILE_DOF_LOCAL), bounds_check=False)
            armature = wp.tile_load(R_group[idx], shape=(TILE_DOF_LOCAL,), bounds_check=False)

            # Add armature to diagonal
            H_tile = wp.tile_diag_add(H_tile, armature)

            # Compute Cholesky factorization
            L_tile = wp.tile_cholesky(H_tile)

            # Store result
            wp.tile_store(L_group[idx], L_tile)

        cholesky_batched_tiled_template.__name__ = f"cholesky_batched_tiled_{n_dofs}"
        cholesky_batched_tiled_template.__qualname__ = f"cholesky_batched_tiled_{n_dofs}"
        return wp.kernel(enable_backward=False, module="unique")(cholesky_batched_tiled_template)

    @classmethod
    def get_triangular_solve_kernel(cls, n_dofs: int, device: "wp.Device") -> "wp.Kernel":
        """Get or create a tiled triangular solve kernel for the given DOF count."""
        key = (n_dofs, device.arch)
        if key not in cls._triangular_solve_cache:
            cls._triangular_solve_cache[key] = cls._build_triangular_solve_kernel(n_dofs)
        return cls._triangular_solve_cache[key]

    @classmethod
    def _build_triangular_solve_kernel(cls, n_dofs: int) -> "wp.Kernel":
        """Build specialized triangular solve kernel for given DOF count.

        Solves L * L^T * x = b for x using tiled forward and backward substitution.
        """
        TILE_DOF_LOCAL = wp.constant(int(n_dofs))

        def trisolve_batched_tiled_template(
            L_group: wp.array3d(dtype=float),  # [n_arts, n_dofs, n_dofs]
            tau_group: wp.array3d(dtype=float),  # [n_arts, n_dofs, 1]
            qdd_group: wp.array3d(dtype=float),  # [n_arts, n_dofs, 1]
        ):
            idx = wp.tid()
            L_tile = wp.tile_load(L_group[idx], shape=(TILE_DOF_LOCAL, TILE_DOF_LOCAL), bounds_check=False)
            tau_tile = wp.tile_load(tau_group[idx], shape=(TILE_DOF_LOCAL, 1), bounds_check=False)

            # Forward substitution: L * z = tau
            z_tile = wp.tile_lower_solve(L_tile, tau_tile)

            # Backward substitution: L^T * qdd = z
            Lt_tile = wp.tile_transpose(L_tile)
            qdd_tile = wp.tile_upper_solve(Lt_tile, z_tile)

            wp.tile_store(qdd_group[idx], qdd_tile)

        trisolve_batched_tiled_template.__name__ = f"trisolve_batched_tiled_{n_dofs}"
        trisolve_batched_tiled_template.__qualname__ = f"trisolve_batched_tiled_{n_dofs}"
        return wp.kernel(enable_backward=False, module="unique")(trisolve_batched_tiled_template)

    @classmethod
    def get_pgs_solve_tiled_row_kernel(cls, max_constraints: int, device: "wp.Device") -> "wp.Kernel":
        """Get or create a tiled row-wise PGS world solve kernel for the given constraint count."""
        key = (max_constraints, device.arch)
        if key not in cls._pgs_solve_tiled_row_cache:
            cls._pgs_solve_tiled_row_cache[key] = cls._build_pgs_solve_tiled_row_kernel(max_constraints)
        return cls._pgs_solve_tiled_row_cache[key]

    @classmethod
    def _build_pgs_solve_tiled_row_kernel(cls, max_constraints: int) -> "wp.Kernel":
        """PGS world solve kernel that stages only the LOWER triangle of Delassus.

        Shared memory footprint drops from M*M to M*(M+1)/2 floats.
        Uses symmetry in dot: C(i,j) = L(i,j) if j<=i else L(j,i).
        """
        TILE_M = max_constraints
        TILE_M_SQ = TILE_M * TILE_M
        TILE_TRI = TILE_M * (TILE_M + 1) // 2

        ELEMS_PER_THREAD_1D = (TILE_M + 31) // 32

        def gen_load_1d(dst, src):
            return "\n".join(
                [
                    f"    {dst}[lane + {k * 32}] = {src}.data[off1 + lane + {k * 32}];"
                    for k in range(ELEMS_PER_THREAD_1D)
                    if (k * 32) < TILE_M
                ]
            )

        # Build a deterministic packed-lower-tri index order: row-major over (i, j<=i)
        # idx = i*(i+1)/2 + j
        tri_pairs = []
        for i in range(TILE_M):
            base = i * (i + 1) // 2
            for j in range(i + 1):
                tri_pairs.append((base + j, i, j))
        assert len(tri_pairs) == TILE_TRI

        load_code = "\n".join(
            [
                gen_load_1d("s_lam", "world_impulses"),
                gen_load_1d("s_rhs", "world_rhs"),
                gen_load_1d("s_diag", "world_diag"),
                gen_load_1d("s_rtype", "world_row_type"),
                gen_load_1d("s_parent", "world_row_parent"),
                gen_load_1d("s_mu", "world_row_mu"),
            ]
        )

        # Precompute lane's column indices (j_k) and their triangular bases (j_k*(j_k+1)/2)
        # so inside the dot we avoid multiply.
        precompute_j = []
        for k in range(ELEMS_PER_THREAD_1D):
            j = k * 32
            if j < TILE_M:
                precompute_j.append(
                    f"    const int j{k} = lane + {j};\n    const int jb{k} = (j{k} * (j{k} + 1)) >> 1;"
                )
        precompute_j_code = "\n".join(precompute_j)

        # Dot code: guarded on j_k < m
        dot_terms = []
        for k in range(ELEMS_PER_THREAD_1D):
            joff = k * 32
            if joff < TILE_M:
                dot_terms.append(
                    f"""    if (j{k} < m) {{
            // Use symmetry to fetch C(i, j{k}) from packed-lower shared.
            // base_i = i*(i+1)/2
            float cij = (j{k} <= i) ? s_Ctri[base_i + j{k}] : s_Ctri[jb{k} + i];
            my_sum += cij * s_lam[j{k}];
        }}"""
                )
        dot_code = "\n".join(["float my_sum = 0.0f;", "int base_i = (i * (i + 1)) >> 1;", *dot_terms])

        store_code = "\n".join(
            [
                f"    world_impulses.data[off1 + lane + {k * 32}] = s_lam[lane + {k * 32}];"
                for k in range(ELEMS_PER_THREAD_1D)
                if (k * 32) < TILE_M
            ]
        )

        snippet = f"""
    #if defined(__CUDA_ARCH__)
        const int TILE_M = {TILE_M};
        const int TILE_M_SQ = {TILE_M_SQ};
        const int TILE_TRI = {TILE_TRI};
        const unsigned MASK = 0xFFFFFFFF;

        int lane = threadIdx.x;

        int m = world_constraint_count.data[world];
        if (m == 0) return;

        // Packed LOWER triangle of C in row-major (i*(i+1)/2 + j), j<=i
        __shared__ float s_Ctri[TILE_TRI];

        __shared__ float s_lam[TILE_M];
        __shared__ float s_rhs[TILE_M];
        __shared__ float s_diag[TILE_M];
        __shared__ int   s_rtype[TILE_M];
        __shared__ int   s_parent[TILE_M];
        __shared__ float s_mu[TILE_M];

        int off1 = world * TILE_M;
        int off2 = world * TILE_M_SQ;

    {load_code}

        // Load only lower triangle from global full matrix into packed shared.
        // Work distribution: each lane walks rows; for each row i, lane loads j = lane, lane+32, lane+64...
        for (int i = 0; i < TILE_M; ++i) {{
            int base = (i * (i + 1)) >> 1; // packed base for row i
            for (int j = lane; j <= i; j += 32) {{
                s_Ctri[base + j] = world_C.data[off2 + i * TILE_M + j];
            }}
        }}
        __syncwarp();

    {precompute_j_code}

        for (int iter = 0; iter < iterations; iter++) {{
            for (int i = 0; i < m; i++) {{
                // NOTE: single-warp kernel; __syncwarp here is typically unnecessary unless divergence occurs
                // before the dot. If you want max perf, try removing it after verifying correctness.
                // __syncwarp();

                {dot_code}

                // Warp reduce my_sum
                my_sum += __shfl_down_sync(MASK, my_sum, 16);
                my_sum += __shfl_down_sync(MASK, my_sum, 8);
                my_sum += __shfl_down_sync(MASK, my_sum, 4);
                my_sum += __shfl_down_sync(MASK, my_sum, 2);
                my_sum += __shfl_down_sync(MASK, my_sum, 1);
                float dot_sum = __shfl_sync(MASK, my_sum, 0);

                float denom = s_diag[i];
                if (denom <= 0.0f) continue;

                float w_val = s_rhs[i] + dot_sum;
                float delta = -w_val / denom;
                float new_impulse = s_lam[i] + omega * delta;
                int row_type = s_rtype[i];

                if (row_type == 0) {{
                    if (new_impulse < 0.0f) new_impulse = 0.0f;
                    s_lam[i] = new_impulse;
                }} else if (row_type == 2) {{
                    int parent_idx = s_parent[i];
                    float lambda_n = s_lam[parent_idx];
                    float mu = s_mu[i];
                    float radius = fmaxf(mu * lambda_n, 0.0f);

                    if (radius <= 0.0f) {{
                        s_lam[i] = 0.0f;
                    }} else {{
                        s_lam[i] = new_impulse;
                        int sib = (i == parent_idx + 1) ? (parent_idx + 2) : (parent_idx + 1);
                        float a = s_lam[i];
                        float b = s_lam[sib];
                        float mag = sqrtf(a * a + b * b);
                        if (mag > radius) {{
                            float scale = radius / mag;
                            s_lam[i] = a * scale;
                            s_lam[sib] = b * scale;
                        }}
                    }}
                }} else {{
                    s_lam[i] = new_impulse;
                }}
            }}
        }}

    {store_code}
    #endif
    """

        @wp.func_native(snippet)
        def pgs_solve_native(
            world: int,
            world_constraint_count: wp.array(dtype=int),
            world_diag: wp.array2d(dtype=float),
            world_C: wp.array3d(dtype=float),
            world_rhs: wp.array2d(dtype=float),
            world_impulses: wp.array2d(dtype=float),
            iterations: int,
            omega: float,
            world_row_type: wp.array2d(dtype=int),
            world_row_parent: wp.array2d(dtype=int),
            world_row_mu: wp.array2d(dtype=float),
        ): ...

        def pgs_solve_tiled_template(
            world_constraint_count: wp.array(dtype=int),
            world_diag: wp.array2d(dtype=float),
            world_C: wp.array3d(dtype=float),
            world_rhs: wp.array2d(dtype=float),
            world_impulses: wp.array2d(dtype=float),
            iterations: int,
            omega: float,
            world_row_type: wp.array2d(dtype=int),
            world_row_parent: wp.array2d(dtype=int),
            world_row_mu: wp.array2d(dtype=float),
        ):
            world, _lane = wp.tid()
            pgs_solve_native(
                world,
                world_constraint_count,
                world_diag,
                world_C,
                world_rhs,
                world_impulses,
                iterations,
                omega,
                world_row_type,
                world_row_parent,
                world_row_mu,
            )

        pgs_solve_tiled_template.__name__ = f"pgs_solve_tiled_row_{max_constraints}"
        pgs_solve_tiled_template.__qualname__ = f"pgs_solve_tiled_row_{max_constraints}"
        return wp.kernel(enable_backward=False, module="unique")(pgs_solve_tiled_template)

    @classmethod
    def get_pgs_solve_tiled_contact_kernel(cls, max_constraints: int, device: "wp.Device") -> "wp.Kernel":
        """Get or create a tiled contact-wise PGS world solve kernel using 3x3 block formulation."""
        key = (max_constraints, device.arch)
        if key not in cls._pgs_solve_tiled_contact_cache:
            cls._pgs_solve_tiled_contact_cache[key] = cls._build_pgs_solve_tiled_contact_kernel(max_constraints)
        return cls._pgs_solve_tiled_contact_cache[key]

    @classmethod
    def _build_pgs_solve_tiled_contact_kernel(cls, max_constraints: int) -> "wp.Kernel":
        """PGS world solve kernel using 3x3 block formulation.

        Stores only the LOWER triangle of block Delassus matrix.
        Each contact is a 3-vector (normal, tangent1, tangent2).
        Reduces serial depth from M to M/3.

        TILE_M can be any value (power of 2 recommended for other kernels).
        Runtime m must be divisible by 3.
        """
        TILE_M = max_constraints
        # Max contacts we can handle (rounded down)
        NUM_CONTACTS_MAX = TILE_M // 3
        # Actual max constraints we'll process (may be < TILE_M)
        TILE_M_USABLE = NUM_CONTACTS_MAX * 3

        # Lower triangle of block matrix (sized for max)
        NUM_BLOCKS_TRI = NUM_CONTACTS_MAX * (NUM_CONTACTS_MAX + 1) // 2
        BLOCK_TRI_FLOATS = NUM_BLOCKS_TRI * 9

        snippet = f"""
    #if defined(__CUDA_ARCH__)
        const int TILE_M = {TILE_M};
        const int TILE_M_USABLE = {TILE_M_USABLE};
        const int NUM_CONTACTS_MAX = {NUM_CONTACTS_MAX};
        const int BLOCK_TRI_FLOATS = {BLOCK_TRI_FLOATS};
        const unsigned MASK = 0xFFFFFFFF;

        int lane = threadIdx.x;

        int m = world_constraint_count.data[world];
        if (m == 0) return;

        // Clamp m to usable range and ensure divisible by 3
        if (m > TILE_M_USABLE) m = TILE_M_USABLE;
        int num_contacts = m / 3;

        // Shared memory (sized for max)
        __shared__ float s_Dtri[BLOCK_TRI_FLOATS];
        __shared__ float s_Dinv[NUM_CONTACTS_MAX * 9];
        __shared__ float s_lam[TILE_M_USABLE];
        __shared__ float s_rhs[TILE_M_USABLE];
        __shared__ float s_mu[NUM_CONTACTS_MAX];

        int off1 = world * TILE_M;
        int off2 = world * TILE_M * TILE_M;

        // ============ LOAD PHASE ============

        // Load lambda and rhs
        for (int i = lane; i < TILE_M_USABLE; i += 32) {{
            if (i < m) {{
                s_lam[i] = world_impulses.data[off1 + i];
                s_rhs[i] = world_rhs.data[off1 + i];
            }} else {{
                s_lam[i] = 0.0f;
                s_rhs[i] = 0.0f;
            }}
        }}

        // Load mu (one per contact, stored on tangent1 row)
        for (int c = lane; c < NUM_CONTACTS_MAX; c += 32) {{
            if (c < num_contacts) {{
                s_mu[c] = world_row_mu.data[off1 + c * 3 + 1];
            }}
        }}

        // Load lower triangle of block Delassus
        for (int c = 0; c < num_contacts; c++) {{
            int base_block = (c * (c + 1)) >> 1;
            int floats_in_row = (c + 1) * 9;

            for (int f = lane; f < floats_in_row; f += 32) {{
                int j = f / 9;
                int k = f % 9;
                int lr = k / 3;
                int lc = k % 3;
                int gr = c * 3 + lr;
                int gc = j * 3 + lc;
                s_Dtri[(base_block + j) * 9 + k] = world_C.data[off2 + gr * TILE_M + gc];
            }}
        }}
        __syncwarp();

        // Compute diagonal block inverses
        for (int c = lane; c < num_contacts; c += 32) {{
            int diag_block_idx = ((c * (c + 1)) >> 1) + c;
            const float* D = &s_Dtri[diag_block_idx * 9];
            float* Dinv = &s_Dinv[c * 9];

            float det = D[0] * (D[4] * D[8] - D[5] * D[7])
                    - D[1] * (D[3] * D[8] - D[5] * D[6])
                    + D[2] * (D[3] * D[7] - D[4] * D[6]);

            float inv_det = 1.0f / det;

            Dinv[0] = (D[4] * D[8] - D[5] * D[7]) * inv_det;
            Dinv[1] = (D[2] * D[7] - D[1] * D[8]) * inv_det;
            Dinv[2] = (D[1] * D[5] - D[2] * D[4]) * inv_det;
            Dinv[3] = (D[5] * D[6] - D[3] * D[8]) * inv_det;
            Dinv[4] = (D[0] * D[8] - D[2] * D[6]) * inv_det;
            Dinv[5] = (D[2] * D[3] - D[0] * D[5]) * inv_det;
            Dinv[6] = (D[3] * D[7] - D[4] * D[6]) * inv_det;
            Dinv[7] = (D[1] * D[6] - D[0] * D[7]) * inv_det;
            Dinv[8] = (D[0] * D[4] - D[1] * D[3]) * inv_det;
        }}
        __syncwarp();

        // ============ ITERATION PHASE ============

        for (int iter = 0; iter < iterations; iter++) {{
            for (int c = 0; c < num_contacts; c++) {{
                float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f;

                for (int j = lane; j < num_contacts; j += 32) {{
                    float l0 = s_lam[j * 3 + 0];
                    float l1 = s_lam[j * 3 + 1];
                    float l2 = s_lam[j * 3 + 2];

                    int block_off;
                    bool transpose;
                    if (j <= c) {{
                        block_off = (((c * (c + 1)) >> 1) + j) * 9;
                        transpose = false;
                    }} else {{
                        block_off = (((j * (j + 1)) >> 1) + c) * 9;
                        transpose = true;
                    }}

                    const float* B = &s_Dtri[block_off];

                    if (!transpose) {{
                        sum0 += B[0] * l0 + B[1] * l1 + B[2] * l2;
                        sum1 += B[3] * l0 + B[4] * l1 + B[5] * l2;
                        sum2 += B[6] * l0 + B[7] * l1 + B[8] * l2;
                    }} else {{
                        sum0 += B[0] * l0 + B[3] * l1 + B[6] * l2;
                        sum1 += B[1] * l0 + B[4] * l1 + B[7] * l2;
                        sum2 += B[2] * l0 + B[5] * l1 + B[8] * l2;
                    }}
                }}

                // Warp reduce
                sum0 += __shfl_down_sync(MASK, sum0, 16);
                sum1 += __shfl_down_sync(MASK, sum1, 16);
                sum2 += __shfl_down_sync(MASK, sum2, 16);
                sum0 += __shfl_down_sync(MASK, sum0, 8);
                sum1 += __shfl_down_sync(MASK, sum1, 8);
                sum2 += __shfl_down_sync(MASK, sum2, 8);
                sum0 += __shfl_down_sync(MASK, sum0, 4);
                sum1 += __shfl_down_sync(MASK, sum1, 4);
                sum2 += __shfl_down_sync(MASK, sum2, 4);
                sum0 += __shfl_down_sync(MASK, sum0, 2);
                sum1 += __shfl_down_sync(MASK, sum1, 2);
                sum2 += __shfl_down_sync(MASK, sum2, 2);
                sum0 += __shfl_down_sync(MASK, sum0, 1);
                sum1 += __shfl_down_sync(MASK, sum1, 1);
                sum2 += __shfl_down_sync(MASK, sum2, 1);

                if (lane == 0) {{
                    // Corrected sign: -(rhs + D*lambda)
                    float res0 = -(s_rhs[c * 3 + 0] + sum0);
                    float res1 = -(s_rhs[c * 3 + 1] + sum1);
                    float res2 = -(s_rhs[c * 3 + 2] + sum2);

                    const float* Dinv = &s_Dinv[c * 9];
                    float d0 = Dinv[0] * res0 + Dinv[1] * res1 + Dinv[2] * res2;
                    float d1 = Dinv[3] * res0 + Dinv[4] * res1 + Dinv[5] * res2;
                    float d2 = Dinv[6] * res0 + Dinv[7] * res1 + Dinv[8] * res2;

                    float new_n  = s_lam[c * 3 + 0] + omega * d0;
                    float new_t1 = s_lam[c * 3 + 1] + omega * d1;
                    float new_t2 = s_lam[c * 3 + 2] + omega * d2;

                    // Friction cone projection
                    new_n = fmaxf(new_n, 0.0f);

                    float mu = s_mu[c];
                    float radius = mu * new_n;

                    if (radius <= 0.0f) {{
                        new_t1 = 0.0f;
                        new_t2 = 0.0f;
                    }} else {{
                        float t_mag_sq = new_t1 * new_t1 + new_t2 * new_t2;
                        if (t_mag_sq > radius * radius) {{
                            float scale = radius * rsqrtf(t_mag_sq);
                            new_t1 *= scale;
                            new_t2 *= scale;
                        }}
                    }}

                    s_lam[c * 3 + 0] = new_n;
                    s_lam[c * 3 + 1] = new_t1;
                    s_lam[c * 3 + 2] = new_t2;
                }}
                __syncwarp();
            }}
        }}

        // ============ STORE PHASE ============

        for (int i = lane; i < TILE_M_USABLE; i += 32) {{
            if (i < m) {{
                world_impulses.data[off1 + i] = s_lam[i];
            }}
        }}
    #endif
    """

        @wp.func_native(snippet)
        def pgs_solve_contact_native(
            world: int,
            world_constraint_count: wp.array(dtype=int),
            world_C: wp.array3d(dtype=float),
            world_rhs: wp.array2d(dtype=float),
            world_impulses: wp.array2d(dtype=float),
            iterations: int,
            omega: float,
            world_row_mu: wp.array2d(dtype=float),
        ): ...

        def pgs_solve_tiled_contact_template(
            world_constraint_count: wp.array(dtype=int),
            world_C: wp.array3d(dtype=float),
            world_rhs: wp.array2d(dtype=float),
            world_impulses: wp.array2d(dtype=float),
            iterations: int,
            omega: float,
            world_row_mu: wp.array2d(dtype=float),
        ):
            world, _lane = wp.tid()
            pgs_solve_contact_native(
                world,
                world_constraint_count,
                world_C,
                world_rhs,
                world_impulses,
                iterations,
                omega,
                world_row_mu,
            )

        pgs_solve_tiled_contact_template.__name__ = f"pgs_solve_tiled_contact_{max_constraints}"
        pgs_solve_tiled_contact_template.__qualname__ = f"pgs_solve_tiled_contact_{max_constraints}"
        return wp.kernel(enable_backward=False, module="unique")(pgs_solve_tiled_contact_template)

    @classmethod
    def get_pgs_solve_streaming_kernel(
        cls, max_constraints: int, device: "wp.Device", pgs_chunk_size: int = 1
    ) -> "wp.Kernel":
        """Get or create a streaming contact-wise PGS world solve kernel."""
        key = (max_constraints, device.arch, pgs_chunk_size)
        if key not in cls._pgs_solve_streaming_cache:
            cls._pgs_solve_streaming_cache[key] = cls._build_pgs_solve_streaming_kernel(
                max_constraints, pgs_chunk_size
            )
        return cls._pgs_solve_streaming_cache[key]

    @classmethod
    def _build_pgs_solve_streaming_kernel(cls, max_constraints: int, pgs_chunk_size: int = 1) -> "wp.Kernel":
        """Streaming contact-wise PGS kernel that streams block-rows from global memory.

        Unlike tiled_contact which loads the entire Delassus matrix into shared memory,
        this kernel keeps only lambda and auxiliaries in shared memory and streams
        block-rows of C on demand. This enables handling much larger constraint counts
        (hundreds of contacts) at the cost of increased global memory bandwidth.

        When pgs_chunk_size > 1, multiple block-rows are preloaded into shared memory
        at once, reducing the number of global memory round-trips per PGS iteration.

        Algorithm:
        - Load lambda, rhs, mu, and compute diagonal block inverses once
        - For each PGS iteration:
            - For each chunk of pgs_chunk_size contacts:
                - Preload pgs_chunk_size block-rows of C into shared memory
                - For each contact c in the chunk:
                    - Compute block-row dot product with lambda (warp-parallel)
                    - Update lambda[c] with friction cone projection (lane 0)
        - Store final lambda back to global memory
        """
        TILE_M = max_constraints
        NUM_CONTACTS_MAX = TILE_M // 3
        TILE_M_USABLE = NUM_CONTACTS_MAX * 3
        PGS_CHUNK = pgs_chunk_size

        snippet = f"""
    #if defined(__CUDA_ARCH__)
        const int TILE_M = {TILE_M};
        const int TILE_M_USABLE = {TILE_M_USABLE};
        const int NUM_CONTACTS_MAX = {NUM_CONTACTS_MAX};
        const int PGS_CHUNK = {PGS_CHUNK};
        const unsigned MASK = 0xFFFFFFFF;

        int lane = threadIdx.x;

        int m = world_constraint_count.data[world];
        if (m == 0) return;

        // Clamp m to usable range and ensure divisible by 3
        if (m > TILE_M_USABLE) m = TILE_M_USABLE;
        int num_contacts = m / 3;

        // ═══════════════════════════════════════════════════════════════
        // SHARED MEMORY: lambda, rhs, mu, diagonal inverses, and
        // block-row buffer for PGS_CHUNK contacts at a time
        // ═══════════════════════════════════════════════════════════════
        __shared__ float s_lam[{TILE_M_USABLE}];
        __shared__ float s_rhs[{TILE_M_USABLE}];
        __shared__ float s_mu[{NUM_CONTACTS_MAX}];
        __shared__ float s_Dinv[{NUM_CONTACTS_MAX} * 9];
        __shared__ float s_block_rows[{PGS_CHUNK} * {NUM_CONTACTS_MAX} * 9];

        int off1 = world * TILE_M;
        int off2 = world * TILE_M * TILE_M;

        // ═══════════════════════════════════════════════════════════════
        // LOAD PHASE: Load persistent data into shared memory
        // ═══════════════════════════════════════════════════════════════

        // Load lambda and rhs (coalesced)
        for (int i = lane; i < TILE_M_USABLE; i += 32) {{
            if (i < m) {{
                s_lam[i] = world_impulses.data[off1 + i];
                s_rhs[i] = world_rhs.data[off1 + i];
            }} else {{
                s_lam[i] = 0.0f;
                s_rhs[i] = 0.0f;
            }}
        }}

        // Load mu (one per contact, stored on tangent1 row)
        for (int c = lane; c < NUM_CONTACTS_MAX; c += 32) {{
            if (c < num_contacts) {{
                s_mu[c] = world_row_mu.data[off1 + c * 3 + 1];
            }}
        }}
        __syncwarp();

        // Compute diagonal block inverses (each thread handles one contact)
        for (int c = lane; c < num_contacts; c += 32) {{
            // Load diagonal block D[c,c] from global memory
            int diag_row = c * 3;
            float D[9];
            for (int k = 0; k < 9; k++) {{
                int lr = k / 3;
                int lc = k % 3;
                D[k] = world_C.data[off2 + (diag_row + lr) * TILE_M + (diag_row + lc)];
            }}

            // Compute 3x3 inverse
            float det = D[0] * (D[4] * D[8] - D[5] * D[7])
                      - D[1] * (D[3] * D[8] - D[5] * D[6])
                      + D[2] * (D[3] * D[7] - D[4] * D[6]);

            float inv_det = 1.0f / det;
            float* Dinv = &s_Dinv[c * 9];

            Dinv[0] = (D[4] * D[8] - D[5] * D[7]) * inv_det;
            Dinv[1] = (D[2] * D[7] - D[1] * D[8]) * inv_det;
            Dinv[2] = (D[1] * D[5] - D[2] * D[4]) * inv_det;
            Dinv[3] = (D[5] * D[6] - D[3] * D[8]) * inv_det;
            Dinv[4] = (D[0] * D[8] - D[2] * D[6]) * inv_det;
            Dinv[5] = (D[2] * D[3] - D[0] * D[5]) * inv_det;
            Dinv[6] = (D[3] * D[7] - D[4] * D[6]) * inv_det;
            Dinv[7] = (D[1] * D[6] - D[0] * D[7]) * inv_det;
            Dinv[8] = (D[0] * D[4] - D[1] * D[3]) * inv_det;
        }}
        __syncwarp();

        // ═══════════════════════════════════════════════════════════════
        // ITERATION PHASE: Stream block-rows in chunks and solve
        // ═══════════════════════════════════════════════════════════════

        for (int iter = 0; iter < iterations; iter++) {{
            for (int chunk_start = 0; chunk_start < num_contacts; chunk_start += PGS_CHUNK) {{
                int chunk_end = min(chunk_start + PGS_CHUNK, num_contacts);
                int chunk_len = chunk_end - chunk_start;

                // ─────────────────────────────────────────────────────────
                // STREAM: Preload chunk_len block-rows of Delassus matrix
                // ─────────────────────────────────────────────────────────
                for (int ci = 0; ci < chunk_len; ci++) {{
                    int c = chunk_start + ci;
                    int c_row = c * 3;
                    float* row_base = &s_block_rows[ci * NUM_CONTACTS_MAX * 9];
                    for (int j = lane; j < num_contacts; j += 32) {{
                        int j_col = j * 3;
                        float* dst = &row_base[j * 9];
                        for (int k = 0; k < 9; k++) {{
                            int lr = k / 3;
                            int lc = k % 3;
                            dst[k] = world_C.data[off2 + (c_row + lr) * TILE_M + (j_col + lc)];
                        }}
                    }}
                }}
                __syncwarp();

                // ─────────────────────────────────────────────────────────
                // SOLVE: Process each contact in the chunk sequentially
                // ─────────────────────────────────────────────────────────
                for (int ci = 0; ci < chunk_len; ci++) {{
                    int c = chunk_start + ci;
                    const float* row_base = &s_block_rows[ci * NUM_CONTACTS_MAX * 9];

                    // Block-row dot product sum_j C[c,j] * lambda[j]
                    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f;

                    for (int j = lane; j < num_contacts; j += 32) {{
                        float l0 = s_lam[j * 3 + 0];
                        float l1 = s_lam[j * 3 + 1];
                        float l2 = s_lam[j * 3 + 2];

                        const float* B = &row_base[j * 9];

                        sum0 += B[0] * l0 + B[1] * l1 + B[2] * l2;
                        sum1 += B[3] * l0 + B[4] * l1 + B[5] * l2;
                        sum2 += B[6] * l0 + B[7] * l1 + B[8] * l2;
                    }}

                    // Warp reduce
                    sum0 += __shfl_down_sync(MASK, sum0, 16);
                    sum1 += __shfl_down_sync(MASK, sum1, 16);
                    sum2 += __shfl_down_sync(MASK, sum2, 16);
                    sum0 += __shfl_down_sync(MASK, sum0, 8);
                    sum1 += __shfl_down_sync(MASK, sum1, 8);
                    sum2 += __shfl_down_sync(MASK, sum2, 8);
                    sum0 += __shfl_down_sync(MASK, sum0, 4);
                    sum1 += __shfl_down_sync(MASK, sum1, 4);
                    sum2 += __shfl_down_sync(MASK, sum2, 4);
                    sum0 += __shfl_down_sync(MASK, sum0, 2);
                    sum1 += __shfl_down_sync(MASK, sum1, 2);
                    sum2 += __shfl_down_sync(MASK, sum2, 2);
                    sum0 += __shfl_down_sync(MASK, sum0, 1);
                    sum1 += __shfl_down_sync(MASK, sum1, 1);
                    sum2 += __shfl_down_sync(MASK, sum2, 1);

                    // Update: Solve and project (lane 0 only)
                    if (lane == 0) {{
                        float res0 = -(s_rhs[c * 3 + 0] + sum0);
                        float res1 = -(s_rhs[c * 3 + 1] + sum1);
                        float res2 = -(s_rhs[c * 3 + 2] + sum2);

                        const float* Dinv = &s_Dinv[c * 9];
                        float d0 = Dinv[0] * res0 + Dinv[1] * res1 + Dinv[2] * res2;
                        float d1 = Dinv[3] * res0 + Dinv[4] * res1 + Dinv[5] * res2;
                        float d2 = Dinv[6] * res0 + Dinv[7] * res1 + Dinv[8] * res2;

                        float new_n  = s_lam[c * 3 + 0] + omega * d0;
                        float new_t1 = s_lam[c * 3 + 1] + omega * d1;
                        float new_t2 = s_lam[c * 3 + 2] + omega * d2;

                        // Friction cone projection
                        new_n = fmaxf(new_n, 0.0f);

                        float mu = s_mu[c];
                        float radius = mu * new_n;

                        if (radius <= 0.0f) {{
                            new_t1 = 0.0f;
                            new_t2 = 0.0f;
                        }} else {{
                            float t_mag_sq = new_t1 * new_t1 + new_t2 * new_t2;
                            if (t_mag_sq > radius * radius) {{
                                float scale = radius * rsqrtf(t_mag_sq);
                                new_t1 *= scale;
                                new_t2 *= scale;
                            }}
                        }}

                        s_lam[c * 3 + 0] = new_n;
                        s_lam[c * 3 + 1] = new_t1;
                        s_lam[c * 3 + 2] = new_t2;
                    }}
                    __syncwarp();
                }}
            }}
        }}

        // ═══════════════════════════════════════════════════════════════
        // STORE PHASE: Write final lambda back to global memory
        // ═══════════════════════════════════════════════════════════════
        for (int i = lane; i < TILE_M_USABLE; i += 32) {{
            if (i < m) {{
                world_impulses.data[off1 + i] = s_lam[i];
            }}
        }}
    #endif
    """

        @wp.func_native(snippet)
        def pgs_solve_streaming_native(
            world: int,
            world_constraint_count: wp.array(dtype=int),
            world_C: wp.array3d(dtype=float),
            world_rhs: wp.array2d(dtype=float),
            world_impulses: wp.array2d(dtype=float),
            iterations: int,
            omega: float,
            world_row_mu: wp.array2d(dtype=float),
        ): ...

        def pgs_solve_streaming_template(
            world_constraint_count: wp.array(dtype=int),
            world_C: wp.array3d(dtype=float),
            world_rhs: wp.array2d(dtype=float),
            world_impulses: wp.array2d(dtype=float),
            iterations: int,
            omega: float,
            world_row_mu: wp.array2d(dtype=float),
        ):
            world, _lane = wp.tid()
            pgs_solve_streaming_native(
                world,
                world_constraint_count,
                world_C,
                world_rhs,
                world_impulses,
                iterations,
                omega,
                world_row_mu,
            )

        pgs_solve_streaming_template.__name__ = f"pgs_solve_streaming_{max_constraints}_chunk{pgs_chunk_size}"
        pgs_solve_streaming_template.__qualname__ = f"pgs_solve_streaming_{max_constraints}_chunk{pgs_chunk_size}"
        return wp.kernel(enable_backward=False, module="unique")(pgs_solve_streaming_template)
