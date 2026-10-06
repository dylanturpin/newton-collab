# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental differentiable FeatherPGS step without contacts.

The step reuses the FeatherPGS forward law with a fixed kernel selection and
stores every intermediate in buffers owned by the output state, so a
:class:`warp.Tape` spanning many steps replays the values each step produced.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from ...sim import Contacts, Control, Model, State
from ...sim.articulation import eval_fk
from ...sim.enums import JointType
from .kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    _compute_body_net_wrench,
    _gyro_skew,
    allocate_world_contact_slots,
    apply_free_root_transport_to_predictor,
    apply_impulses_world_par_dof,
    compute_composite_inertia,
    compute_link_transform,
    compute_link_velocity,
    compute_velocity_predictor,
    compute_world_contact_bias,
    finalize_world_constraint_counts,
    finalize_world_diag_cfm,
    integrate_generalized_joints,
    prescribed_relative_contact_target,
    remove_free_root_transport_from_qdd,
    rhs_accum_world_par_art,
    update_qdd_from_velocity,
)

if TYPE_CHECKING:
    from .solver_feather_pgs import SolverFeatherPGS


def validate_differentiable_options(model: Model, options: dict) -> None:
    """Raise for constructor options the differentiable step does not reproduce."""
    beta = options["friction_anchor_beta"]
    checks = (
        (not model.requires_grad, "a model finalized with requires_grad=True"),
        (model.particle_count > 0, "no particles"),
        (options["enable_sleeping"], "enable_sleeping=False"),
        (options["pgs_warmstart"] or options["mf_warmstart"], "warm starting disabled"),
        (beta is not None and beta > 0.0, "friction_anchor_beta=0"),
        (options["contact_torsion_radius"] > 0.0 or options["contact_torsion_device"], "contact torsion disabled"),
        (options["contact_compliance"], "contact_compliance=False"),
        (options["articulated_contact_response"] != "immediate", 'articulated_contact_response="immediate"'),
        (options["update_mass_matrix_interval"] != 1, "update_mass_matrix_interval=1"),
        (options["enable_joint_limits"], "enable_joint_limits=False"),
        (options["enable_joint_velocity_limits"], "enable_joint_velocity_limits=False"),
        (options["drive_mode"] != "augmented", 'drive_mode="augmented"'),
        (options["pgs_velocity_iterations"] != 0, "pgs_velocity_iterations=0"),
        (options["pgs_debug"], "pgs_debug=False"),
        (options["parallel_tree"], "parallel_tree=False"),
    )
    for unsupported, requirement in checks:
        if unsupported:
            raise ValueError(f"differentiable=True requires {requirement}")


def validate_differentiable_model(solver: SolverFeatherPGS) -> None:
    """Raise for model features the differentiable step does not reproduce."""
    plan = solver._model_plan
    checks = (
        (solver._has_rigid_body_velocity_limits, "no rigid-body velocity limits"),
        (solver._mimic_count or solver._connect_count, "no mimic or loop-closing joints"),
        (
            not np.array_equal(plan.articulation_joint_end, solver.model.articulation_start.numpy()[1:]),
            "no mimic or loop-closing joints",
        ),
        (bool(np.any(solver._kinematic_dof_mask_host)), "no kinematic bodies"),
        (not np.array_equal(plan.response_dof_count, plan.articulation_dof_count), "every articulation DOF dynamic"),
    )
    for unsupported, requirement in checks:
        if unsupported:
            raise ValueError(f"differentiable=True requires {requirement}")


class DifferentiableStep:
    """Fixed-selection FeatherPGS step whose intermediates are owned by the output state."""

    def __init__(self, solver: SolverFeatherPGS):
        self.solver = solver
        model = solver.model
        device = model.device
        joint_parent = model.joint_parent.numpy()
        joint_child = model.joint_child.numpy()
        joint_qd_start = model.joint_qd_start.numpy()

        dof_joint = np.zeros(model.joint_dof_count, dtype=np.int32)
        for joint in range(model.joint_count):
            dof_joint[joint_qd_start[joint] : joint_qd_start[joint + 1]] = joint
        self.dof_joint = wp.array(dof_joint, dtype=wp.int32, device=device)

        body_to_joint = {int(joint_child[j]): j for j in range(model.joint_count)}

        def is_ancestor_or_self(ancestor: int, joint: int) -> bool:
            while joint >= 0:
                if joint == ancestor:
                    return True
                parent_body = joint_parent[joint]
                joint = body_to_joint.get(int(parent_body), -1) if parent_body >= 0 else -1
            return False

        # Per (group, row, col): the descendant-side DOF's composite body and the DOF pair in the
        # order crba_fill_par_dof forms dot(S_ancestor, I_c * S_descendant); -1 marks a zero entry.
        self.crba_body = {}
        self.crba_dofs = {}
        dof_start = solver._model_plan.articulation_dof_start
        for size in solver.size_groups:
            arts = np.flatnonzero(solver._model_plan.response_dof_count == size)
            body = np.full((len(arts), size, size), -1, dtype=np.int32)
            dofs = np.zeros((len(arts), size, size, 2), dtype=np.int32)
            for group, art in enumerate(arts):
                base = int(dof_start[art])
                for r in range(size):
                    for c in range(size):
                        dr, dc = base + r, base + c
                        jr, jc = int(dof_joint[dr]), int(dof_joint[dc])
                        if jr == jc:
                            ancestor, descendant = min(dr, dc), max(dr, dc)
                        elif is_ancestor_or_self(jr, jc):
                            ancestor, descendant = dr, dc
                        elif is_ancestor_or_self(jc, jr):
                            ancestor, descendant = dc, dr
                        else:
                            continue
                        body[group, r, c] = joint_child[dof_joint[descendant]]
                        dofs[group, r, c] = (ancestor, descendant)
            self.crba_body[size] = wp.array(body, dtype=wp.int32, device=device)
            self.crba_dofs[size] = wp.array(dofs.reshape(len(arts), size, size * 2), dtype=wp.int32, device=device)
        self.mass_update_mask = wp.ones(model.articulation_count, dtype=wp.int32, device=device)
        # body_dof_chain[body, d]: DOF d of the body's articulation moves the body (its joint chain).
        body_art = solver.body_to_articulation.numpy()
        max_size = max(solver.size_groups, default=1)
        chain = np.zeros((max(model.body_count, 1), max_size), dtype=np.int32)
        for body, body_joint in body_to_joint.items():
            art = int(body_art[body])
            if art < 0:
                continue
            base = int(dof_start[art])
            joint = body_joint
            while joint >= 0:
                for dof in range(joint_qd_start[joint], joint_qd_start[joint + 1]):
                    chain[body, dof - base] = 1
                parent_body = joint_parent[joint]
                joint = body_to_joint.get(int(parent_body), -1) if parent_body >= 0 else -1
        self.body_dof_chain = wp.array(chain, dtype=wp.int32, device=device)
        restitution = solver.shape_material_restitution.numpy()
        self.has_restitution = bool(solver.enable_restitution and restitution.size and np.any(restitution > 0.0))
        self.body_inertia_terms = wp.zeros((1, 12), dtype=wp.float32, device=device)

    def buffers(self, state_out: State) -> _StepBuffers:
        buffers = getattr(state_out, "_fpgs_differentiable_buffers", None)
        if buffers is None or buffers.owner is not self:
            buffers = _StepBuffers(self)
            state_out._fpgs_differentiable_buffers = buffers
        return buffers

    def step(self, state_in: State, state_out: State, control: Control | None, contacts: Contacts | None, dt: float):
        solver = self.solver
        model = solver.model
        device = model.device
        if contacts is not None:
            self.validate_contacts()
        if state_in is state_out:
            raise ValueError("differentiable=True requires distinct input and output states")
        if control is None:
            control = model.control(clone_variables=False)
        if not model.joint_count:
            solver._step += 1
            return state_out
        b = self.buffers(state_out)

        wp.launch(
            _eval_fk_id_uncached,
            dim=model.articulation_count,
            inputs=[
                model.articulation_start,
                model.joint_type,
                model.joint_parent,
                model.joint_child,
                model.joint_q_start,
                model.joint_qd_start,
                state_in.joint_q,
                state_in.joint_qd,
                model.joint_X_p,
                model.joint_X_c,
                solver.body_X_com,
                model.joint_axis,
                model.joint_dof_dim,
                model.body_com,
                model.body_mass,
                model.body_inertia,
                model.body_world,
                model.body_disable_gravity,
                model.gravity,
                self.body_inertia_terms,
            ],
            outputs=[
                b.body_q,
                b.body_q_com,
                b.articulation_origin,
                b.joint_S_s,
                b.body_I_s,
                b.body_v_s,
                b.body_f_s,
                b.body_a_s,
            ],
            device=device,
        )
        wp.launch(
            _accumulate_body_wrenches,
            dim=model.articulation_count,
            inputs=[
                model.articulation_start,
                model.joint_parent,
                model.joint_child,
                model.joint_articulation,
                b.body_f_s,
                state_in.body_f,
                model.body_flags,
                b.body_q,
                model.body_com,
                b.articulation_origin,
            ],
            outputs=[b.body_ft_s, b.body_fnet_s],
            device=device,
        )
        wp.launch(
            _project_joint_tau,
            dim=model.joint_dof_count,
            inputs=[
                self.dof_joint,
                model.joint_type,
                model.joint_child,
                model.joint_q_start,
                model.joint_qd_start,
                control.joint_f,
                state_in.joint_q,
                state_in.joint_qd,
                solver._passive_spring_stiffness,
                solver._passive_spring_ref,
                solver._passive_joint_damping,
                b.joint_S_s,
                b.body_fnet_s,
            ],
            outputs=[b.tau_rigid],
            device=device,
        )
        wp.launch(
            _augmented_drives_by_dof,
            dim=model.joint_dof_count,
            inputs=[
                self.dof_joint,
                model.joint_type,
                model.joint_q_start,
                model.joint_qd_start,
                model.joint_target_q_start,
                state_in.joint_q,
                state_in.joint_qd,
                model.joint_target_ke,
                model.joint_target_kd,
                control.joint_target_q,
                control.joint_target_qd,
                model.joint_effort_limit,
                b.tau_rigid,
                dt,
            ],
            outputs=[b.tau, b.drive_K],
            device=device,
        )
        wp.launch(
            compute_composite_inertia,
            dim=model.articulation_count,
            inputs=[
                model.articulation_start,
                solver.articulation_joint_end,
                self.mass_update_mask,
                model.joint_ancestor,
                model.joint_child,
                b.body_I_s,
            ],
            outputs=[b.body_I_c],
            device=device,
        )
        for size in solver.size_groups:
            n_arts = solver.n_arts_by_size[size]
            wp.launch(
                _crba_dense,
                dim=(n_arts, size, size),
                inputs=[
                    solver.group_to_art[size],
                    solver.articulation_dof_start,
                    self.crba_body[size],
                    self.crba_dofs[size],
                    b.joint_S_s,
                    b.body_I_c,
                    b.drive_K,
                ],
                outputs=[b.H[size]],
                device=device,
            )
            wp.launch(
                _factor_dense,
                dim=n_arts,
                inputs=[b.H[size], solver.R_by_size[size], size, b.factor_tmp[size]],
                outputs=[b.L[size]],
                device=device,
            )
            wp.launch(
                _solve_dense,
                dim=n_arts,
                inputs=[
                    b.L[size],
                    solver.group_to_art[size],
                    solver.articulation_dof_start,
                    size,
                    b.tau,
                    b.solve_tmp,
                ],
                outputs=[b.joint_qdd],
                device=device,
            )
        wp.launch(
            compute_velocity_predictor,
            dim=model.joint_dof_count,
            inputs=[state_in.joint_qd, solver._kinematic_dof_mask, dt, solver._dynamics_dof_active],
            outputs=[b.joint_qdd, b.v_hat],
            device=device,
        )
        if solver._free_root_joint_count:
            wp.launch(
                apply_free_root_transport_to_predictor,
                dim=solver._free_root_joint_count,
                inputs=[
                    solver._free_root_joint_indices,
                    model.joint_qd_start,
                    solver._kinematic_joint_mask,
                    state_in.joint_qd,
                    dt,
                    solver._dynamics_joint_active,
                ],
                outputs=[b.v_hat],
                device=device,
            )
        v_out = b.v_hat
        if solver._free_rigid_body_count:
            wp.copy(b.v_gyro, b.v_hat)
            wp.launch(
                _free_rigid_gyroscopic_update,
                dim=solver._free_root_joint_count,
                inputs=[
                    solver._free_root_joint_indices,
                    model.joint_qd_start,
                    model.joint_child,
                    solver.body_to_articulation,
                    solver.is_free_rigid,
                    solver.art_group_idx,
                    b.body_q,
                    model.body_inertia,
                    b.L[6],
                    state_in.joint_qd,
                    b.v_hat,
                    dt,
                ],
                outputs=[b.gyro_iterates, b.v_gyro],
                device=device,
            )
            v_out = b.v_gyro
        if contacts is not None:
            v_out = self._solve_dense_contacts(b, contacts, v_out, dt)
        wp.launch(
            update_qdd_from_velocity,
            dim=model.joint_dof_count,
            inputs=[state_in.joint_qd, solver._kinematic_dof_mask, 1.0 / dt, solver._dynamics_dof_active],
            outputs=[v_out, b.joint_qdd_out],
            device=device,
        )
        if solver._free_root_joint_count:
            wp.launch(
                remove_free_root_transport_from_qdd,
                dim=solver._free_root_joint_count,
                inputs=[
                    solver._free_root_joint_indices,
                    model.joint_qd_start,
                    solver._kinematic_joint_mask,
                    state_in.joint_qd,
                    solver._dynamics_joint_active,
                ],
                outputs=[b.joint_qdd_out],
                device=device,
            )
        wp.launch(
            integrate_generalized_joints,
            dim=model.joint_count,
            inputs=[
                model.joint_type,
                model.joint_parent,
                model.joint_child,
                model.joint_q_start,
                model.joint_qd_start,
                solver._kinematic_joint_mask,
                model.joint_dof_dim,
                model.body_com,
                model.joint_X_c,
                state_in.joint_q,
                state_in.joint_qd,
                b.joint_qdd_out,
                dt,
                solver.angular_damping,
                solver._dynamics_joint_active,
            ],
            outputs=[state_out.joint_q, state_out.joint_qd],
            device=device,
        )
        eval_fk(model, state_out.joint_q, state_out.joint_qd, state_out)
        solver._step += 1
        return state_out

    def validate_contacts(self) -> None:
        """Raise for contact configurations the differentiable step does not reproduce yet."""
        solver = self.solver
        checks = (
            (solver.pgs_mode != "split", 'pgs_mode="split" with contacts'),
            (solver._has_free_rigid_bodies, "no single-body free articulations with contacts"),
            (solver.enable_contact_friction, "enable_contact_friction=False"),
            (self.has_restitution, "zero contact restitution"),
            (solver._regularization_enabled, "pgs_contact_regularization=0"),
        )
        for unsupported, requirement in checks:
            if unsupported:
                raise NotImplementedError(f"differentiable=True contacts require {requirement}")

    def _solve_dense_contacts(self, b: _StepBuffers, contacts: Contacts, v_hat: wp.array, dt: float) -> wp.array:
        """Dense articulated contact rows and fixed-iteration PGS with every iterate stored."""
        # Deferred: the solver module imports this one.
        from .solver_feather_pgs import (  # noqa: PLC0415
            _CONTACT_BUILD_THREAD_CAP,
            _ROW_SLOT_UNBOUNDED,
            _clear_dense_row_state,
            _finalize_constraint_status,
        )

        solver = self.solver
        model = solver.model
        device = model.device
        max_rows = solver.dense_max_constraints
        c = b.contact_buffers(contacts)
        for src, dst in (
            (contacts.rigid_contact_count, c.count),
            (contacts.rigid_contact_point0, c.point0),
            (contacts.rigid_contact_point1, c.point1),
            (contacts.rigid_contact_normal, c.normal),
            (contacts.rigid_contact_shape0, c.shape0),
            (contacts.rigid_contact_shape1, c.shape1),
            (contacts.rigid_contact_margin0, c.margin0),
            (contacts.rigid_contact_margin1, c.margin1),
        ):
            wp.copy(dst, src)
        threads = min(contacts.rigid_contact_max, _CONTACT_BUILD_THREAD_CAP)

        # Topology: integer bookkeeping, frozen for the derivative and kept off the tape.
        wp.launch(
            _clear_dense_row_state,
            dim=solver.world_count,
            inputs=[c.slot_counter, c.dense_world_flag, solver._row_dropped_all],
            device=device,
            record_tape=False,
        )
        solver._dense_first_rejected_slot.fill_(_ROW_SLOT_UNBOUNDED)
        dummy = solver._dummy_mf_slot_counter
        wp.launch(
            allocate_world_contact_slots,
            dim=threads,
            inputs=[
                c.count,
                threads,
                c.shape0,
                c.shape1,
                c.point0,
                c.point1,
                c.normal,
                c.margin0,
                c.margin1,
                b.body_q,
                model.shape_transform,
                model.shape_body,
                solver.body_to_articulation,
                solver.art_to_world,
                solver.articulation_response_dof_count,
                model.body_flags,
                solver.body_has_response_dofs,
                solver._dummy_is_free_rigid if solver.is_free_rigid is None else solver.is_free_rigid,
                0,
                0,
                0,
                0,
                solver.contact_gap_gate,
                solver.same_articulation_contact_gap_gate,
                solver.articulation_pair_contact_gap_gate,
                max_rows,
                solver.mf_max_constraints,
                solver.propagation_max_constraints,
                0,
                solver.contact_friction_gap_threshold,
                1 if solver.contact_friction_articulation_pairs_only else 0,
                solver._friction_patches.view,
            ],
            outputs=[
                c.world,
                c.slot,
                c.art_a,
                c.art_b,
                c.slot_counter,
                c.path,
                dummy,
                dummy,
                c.dense_world_flag,
                c.slots_needed,
                solver._row_dropped_dense,
                solver._row_dropped_mf,
                solver._row_dropped_propagation,
                solver._dense_first_rejected_slot,
                dummy,
                dummy,
            ],
            device=device,
            record_tape=False,
        )
        wp.launch(
            finalize_world_constraint_counts,
            dim=solver.world_count,
            inputs=[c.slot_counter, max_rows, solver._dense_first_rejected_slot],
            outputs=[c.row_count],
            device=device,
            record_tape=False,
        )
        wp.launch(
            _finalize_constraint_status,
            dim=solver.world_count,
            inputs=[
                c.slot_counter,
                dummy,
                dummy,
                solver._row_dropped_all,
                c.count,
                contacts._reduction_overflow,
                max_rows,
                solver.mf_max_constraints,
                solver.propagation_max_constraints,
                contacts.rigid_contact_max,
                solver.constraint_overflow,
            ],
            device=device,
            record_tape=False,
        )

        # Rows: geometry recomputed from this step's FK poses; the contact normal is stop-gradient.
        for array in (c.C, c.diag, c.rhs, *c.J.values(), *c.Y.values()):
            array.zero_()
        wp.launch(
            _dense_contact_rows,
            dim=threads,
            inputs=[
                c.count,
                threads,
                c.point0,
                c.point1,
                c.normal,
                c.shape0,
                c.shape1,
                c.margin0,
                c.margin1,
                c.world,
                c.slot,
                c.art_a,
                c.art_b,
                c.path,
                model.shape_body,
                b.body_q,
                b.body_v_s,
                solver._prescribed_articulation,
                b.articulation_origin,
                solver.shape_material_mu,
                solver.shape_material_restitution,
                int(solver.contact_shared_anchor),
                solver.pgs_beta,
                solver.pgs_cfm,
            ],
            outputs=[c.row_type, c.row_parent, c.row_mu, c.row_beta, c.row_cfm, c.phi, c.target_velocity],
            device=device,
        )
        for size in solver.size_groups:
            n_arts = solver.n_arts_by_size[size]
            wp.launch(
                _dense_contact_jacobian,
                dim=(threads, size),
                inputs=[
                    c.count,
                    threads,
                    c.point0,
                    c.point1,
                    c.normal,
                    c.shape0,
                    c.shape1,
                    c.margin0,
                    c.margin1,
                    c.slot,
                    c.art_a,
                    c.art_b,
                    c.path,
                    size,
                    solver.articulation_response_dof_count,
                    solver.art_group_idx,
                    solver.articulation_dof_start,
                    b.articulation_origin,
                    self.body_dof_chain,
                    b.joint_S_s,
                    model.shape_body,
                    b.body_q,
                    int(solver.contact_shared_anchor),
                ],
                outputs=[c.J[size]],
                device=device,
            )
            wp.launch(
                _hinv_jt_dense,
                dim=n_arts * max_rows,
                inputs=[
                    b.L[size],
                    c.J[size],
                    solver.group_to_art[size],
                    solver.art_to_world,
                    c.row_count,
                    size,
                    max_rows,
                    c.Y_tmp[size],
                ],
                outputs=[c.Y[size]],
                device=device,
            )
        for size in solver.size_groups:
            n_arts = solver.n_arts_by_size[size]
            wp.launch(
                _delassus_dense,
                dim=n_arts * max_rows * max_rows,
                inputs=[
                    c.J[size],
                    c.Y[size],
                    solver.group_to_art[size],
                    solver.art_to_world,
                    c.row_count,
                    size,
                    max_rows,
                    n_arts,
                ],
                outputs=[c.C, c.diag],
                device=device,
            )
        wp.launch(
            finalize_world_diag_cfm,
            dim=solver.world_count,
            inputs=[c.row_count, c.row_cfm],
            outputs=[c.diag],
            device=device,
        )
        wp.launch(
            compute_world_contact_bias,
            dim=solver.world_count,
            inputs=[
                c.row_count,
                c.phi,
                c.row_beta,
                c.row_type,
                c.target_velocity,
                dt,
                1.0,
                solver.contact_speculative_scale,
                1.0,
                solver._contact_w,
            ],
            outputs=[c.rhs, c.row_w],
            device=device,
        )
        for size in solver.size_groups:
            wp.launch(
                rhs_accum_world_par_art,
                dim=solver.n_arts_by_size[size],
                inputs=[
                    c.row_count,
                    solver.art_to_world,
                    solver.articulation_dof_start,
                    v_hat,
                    solver.group_to_art[size],
                    c.J[size],
                    size,
                ],
                outputs=[c.rhs],
                device=device,
            )

        # Fixed-iteration PGS; iteration k reads impulses[k] and writes impulses[k + 1] once.
        c.impulses[0].zero_()
        for k in range(solver.pgs_iterations):
            wp.launch(
                _dense_pgs_sweep,
                dim=solver.world_count,
                inputs=[c.row_count, c.diag, c.C, c.rhs, c.row_type, solver.pgs_omega, c.impulses[k]],
                outputs=[c.residuals[k], c.impulses[k + 1]],
                device=device,
            )
        for size in solver.size_groups:
            n_arts = solver.n_arts_by_size[size]
            wp.launch(
                apply_impulses_world_par_dof,
                dim=n_arts * size,
                inputs=[
                    solver.group_to_art[size],
                    solver.art_to_world,
                    solver.articulation_dof_start,
                    size,
                    n_arts,
                    c.row_count,
                    c.Y[size],
                    c.impulses[solver.pgs_iterations],
                    v_hat,
                ],
                outputs=[c.v_out],
                device=device,
            )
        return c.v_out


class _StepBuffers:
    """Intermediates of one differentiable step, kept alive with its output state."""

    def __init__(self, plan: DifferentiableStep):
        self.owner = plan
        solver = plan.solver
        model = solver.model
        device = model.device

        def zeros(shape, dtype=wp.float32):
            return wp.zeros(shape, dtype=dtype, device=device, requires_grad=True)

        bodies, dofs = model.body_count, model.joint_dof_count
        self.body_q = zeros(bodies, wp.transform)
        self.body_q_com = zeros(bodies, wp.transform)
        self.articulation_origin = zeros(model.articulation_count, wp.vec3)
        self.joint_S_s = zeros(dofs, wp.spatial_vector)
        self.body_I_s = zeros(bodies, wp.spatial_matrix)
        self.body_I_c = zeros(bodies, wp.spatial_matrix)
        self.body_v_s = zeros(bodies, wp.spatial_vector)
        self.body_f_s = zeros(bodies, wp.spatial_vector)
        self.body_a_s = zeros(bodies, wp.spatial_vector)
        self.body_ft_s = zeros(bodies, wp.spatial_vector)
        self.body_fnet_s = zeros(bodies, wp.spatial_vector)
        self.tau_rigid = zeros(dofs)
        self.tau = zeros(dofs)
        self.drive_K = wp.zeros(dofs, dtype=wp.float32, device=device)
        self.joint_qdd = zeros(dofs)
        self.joint_qdd_out = zeros(dofs)
        self.v_hat = zeros(dofs)
        self.solve_tmp = zeros(dofs)
        self.v_gyro = zeros(dofs)
        self.gyro_iterates = zeros((max(solver._free_root_joint_count, 1), _GYRO_MAX_MICROSTEPS + 1), wp.vec3)
        self.contacts = None
        self.H = {}
        self.L = {}
        self.factor_tmp = {}
        for size in solver.size_groups:
            n_arts = solver.n_arts_by_size[size]
            self.H[size] = zeros((n_arts, size, size))
            self.L[size] = zeros((n_arts, size, size))
            self.factor_tmp[size] = zeros((n_arts, 2, size, size))

    def contact_buffers(self, contacts: Contacts) -> _ContactBuffers:
        if self.contacts is None or self.contacts.capacity != contacts.rigid_contact_max:
            self.contacts = _ContactBuffers(self.owner.solver, contacts.rigid_contact_max)
        return self.contacts


class _ContactBuffers:
    """One step's contact snapshot, frozen topology, dense rows and stored PGS iterates."""

    def __init__(self, solver: SolverFeatherPGS, capacity: int):
        device = solver.model.device
        self.capacity = capacity
        worlds, rows = solver.world_count, solver.dense_max_constraints

        def zeros(shape, dtype=wp.float32, requires_grad=True):
            return wp.zeros(shape, dtype=dtype, device=device, requires_grad=requires_grad)

        self.count = zeros(1, wp.int32, False)
        self.point0 = zeros(capacity, wp.vec3, False)
        self.point1 = zeros(capacity, wp.vec3, False)
        self.normal = zeros(capacity, wp.vec3, False)
        self.shape0 = zeros(capacity, wp.int32, False)
        self.shape1 = zeros(capacity, wp.int32, False)
        self.margin0 = zeros(capacity, wp.float32, False)
        self.margin1 = zeros(capacity, wp.float32, False)
        for name in ("world", "slot", "art_a", "art_b", "path", "slots_needed"):
            setattr(self, name, zeros(capacity, wp.int32, False))
        self.slot_counter = zeros(worlds, wp.int32, False)
        self.dense_world_flag = zeros(worlds, wp.int32, False)
        self.row_count = zeros(worlds, wp.int32, False)
        self.row_type = zeros((worlds, rows), wp.int32, False)
        self.row_parent = zeros((worlds, rows), wp.int32, False)
        self.row_mu = zeros((worlds, rows))
        self.row_beta = zeros((worlds, rows))
        self.row_cfm = zeros((worlds, rows))
        self.row_w = zeros((worlds, rows))
        self.phi = zeros((worlds, rows))
        self.target_velocity = zeros((worlds, rows))
        self.rhs = zeros((worlds, rows))
        self.diag = zeros((worlds, rows))
        self.C = zeros((worlds, rows, rows))
        self.impulses = [zeros((worlds, rows)) for _ in range(solver.pgs_iterations + 1)]
        self.residuals = [zeros((worlds, rows)) for _ in range(solver.pgs_iterations)]
        self.v_out = zeros(solver.model.joint_dof_count)
        self.J, self.Y, self.Y_tmp = {}, {}, {}
        for size in solver.size_groups:
            n_arts = solver.n_arts_by_size[size]
            self.J[size] = zeros((n_arts, rows, size))
            self.Y[size] = zeros((n_arts, rows, size))
            self.Y_tmp[size] = zeros((n_arts, rows, size))


@wp.kernel
def _eval_fk_id_uncached(
    articulation_start: wp.array[int],
    joint_type: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    body_X_com: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_dof_dim: wp.array2d[int],
    body_com: wp.array[wp.vec3],
    body_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    body_world: wp.array[int],
    body_disable_gravity: wp.array[bool],
    gravity: wp.array[wp.vec3],
    body_inertia_terms: wp.array2d[float],
    # outputs
    body_q: wp.array[wp.transform],
    body_q_com: wp.array[wp.transform],
    articulation_origin: wp.array[wp.vec3],
    joint_S_s: wp.array[wp.spatial_vector],
    body_I_s: wp.array[wp.spatial_matrix],
    body_v_s: wp.array[wp.spatial_vector],
    body_f_s: wp.array[wp.spatial_vector],
    body_a_s: wp.array[wp.spatial_vector],
):
    """eval_rigid_fk_id without the loop-carried parent cache, whose reverse replay is invalid."""
    index = wp.tid()
    start = articulation_start[index]
    end = articulation_start[index + 1]

    for i in range(start, end):
        compute_link_transform(
            i,
            joint_type,
            joint_parent,
            joint_child,
            joint_q_start,
            joint_qd_start,
            joint_q,
            joint_X_p,
            joint_X_c,
            body_X_com,
            joint_axis,
            joint_dof_dim,
            body_q,
            body_q_com,
        )

    origin = wp.vec3()
    if start < end:
        root_body = joint_child[start]
        if root_body >= 0:
            origin = wp.transform_point(body_q[root_body], body_com[root_body])
    articulation_origin[index] = origin

    for i in range(start, end):
        parent = joint_parent[i]
        child = joint_child[i]
        gravity_s = wp.vec3()
        if not body_disable_gravity[child]:
            gravity_s = gravity[body_world[child]]
        parent_v_s = wp.spatial_vector()
        parent_a_s = wp.spatial_vector()
        if parent >= 0:
            parent_v_s = body_v_s[parent]
            parent_a_s = body_a_s[parent]
        compute_link_velocity(
            i,
            parent,
            child,
            parent_v_s,
            parent_a_s,
            origin,
            gravity_s,
            joint_type,
            joint_qd_start,
            joint_qd,
            joint_axis,
            joint_dof_dim,
            body_mass,
            body_inertia,
            1,
            0,
            body_q,
            body_q_com,
            joint_X_p,
            joint_S_s,
            body_I_s,
            body_inertia_terms,
            body_v_s,
            body_f_s,
            body_a_s,
        )


@wp.kernel
def _accumulate_body_wrenches(
    articulation_start: wp.array[int],
    joint_parent: wp.array[int],
    joint_child: wp.array[int],
    joint_articulation: wp.array[int],
    body_fb_s: wp.array[wp.spatial_vector],
    body_f_ext: wp.array[wp.spatial_vector],
    body_flags: wp.array[wp.int32],
    body_q: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    articulation_origin: wp.array[wp.vec3],
    # outputs
    body_ft_s: wp.array[wp.spatial_vector],
    body_fnet_s: wp.array[wp.spatial_vector],
):
    """The wrench recursion of accumulate_articulation_tau, storing each joint's net wrench."""
    index = wp.tid()
    start = articulation_start[index]
    end = articulation_start[index + 1]
    for offset in range(end - start):
        i = end - offset - 1
        parent = joint_parent[i]
        child = joint_child[i]
        articulation = joint_articulation[i]
        origin = wp.vec3()
        if articulation >= 0:
            origin = articulation_origin[articulation]
        f_s = _compute_body_net_wrench(
            child, body_ft_s[child], origin, body_fb_s, body_f_ext, body_flags, body_q, body_com
        )
        body_fnet_s[child] = f_s
        if parent >= 0:
            body_ft_s[parent] = body_ft_s[parent] + f_s


@wp.kernel
def _project_joint_tau(
    dof_joint: wp.array[int],
    joint_type: wp.array[int],
    joint_child: wp.array[int],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_f: wp.array[float],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_spring_stiffness: wp.array[float],
    joint_spring_ref: wp.array[float],
    joint_damping: wp.array[float],
    joint_S_s: wp.array[wp.spatial_vector],
    body_fnet_s: wp.array[wp.spatial_vector],
    # outputs
    tau: wp.array[float],
):
    """Per-DOF form of jcalc_tau with the same arithmetic."""
    dof = wp.tid()
    joint = dof_joint[dof]
    type = joint_type[joint]
    value = -wp.dot(joint_S_s[dof], body_fnet_s[joint_child[joint]]) + joint_f[dof]
    if type == JointType.PRISMATIC or type == JointType.REVOLUTE or type == JointType.D6:
        axis = dof - joint_qd_start[joint]
        passive_f = joint_spring_stiffness[dof] * (joint_spring_ref[dof] - joint_q[joint_q_start[joint] + axis])
        passive_f -= joint_damping[dof] * joint_qd[dof]
        value = value + passive_f
    tau[dof] = value


_GYRO_MAX_MICROSTEPS = 32


@wp.kernel
def _free_rigid_gyroscopic_update(
    free_root_joint_indices: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_child: wp.array[int],
    body_to_articulation: wp.array[int],
    is_free_rigid: wp.array[int],
    art_group_index: wp.array[int],
    body_q: wp.array[wp.transform],
    body_inertia: wp.array[wp.mat33],
    cholesky: wp.array3d[float],
    joint_qd: wp.array[float],
    v_hat: wp.array[float],
    dt: float,
    # outputs
    iterates: wp.array2d[wp.vec3],
    v_out: wp.array[float],
):
    """The gyroscopic update of apply_free_root_velocity_corrections with every microstep stored once.

    The fixed-point iterations are unrolled without ``break`` and the microstep loop carries its state
    only through ``iterates``, so the reverse pass differentiates exactly the executed iterations.
    """
    root_index = wp.tid()
    joint = free_root_joint_indices[root_index]
    d = joint_qd_start[joint]
    body = joint_child[joint]
    art = body_to_articulation[body]
    if is_free_rigid[art] == 0:
        return
    w0 = wp.vec3(joint_qd[d + 3], joint_qd[d + 4], joint_qd[d + 5])
    predicted_world = wp.vec3(v_hat[d + 3], v_hat[d + 4], v_hat[d + 5])
    inertia = body_inertia[body]
    group = art_group_index[art]
    # Constructed whole: component assignment into a local matrix has no adjoint.
    lower = wp.mat33(
        cholesky[group, 3, 3],
        0.0,
        0.0,
        cholesky[group, 4, 3],
        cholesky[group, 4, 4],
        0.0,
        cholesky[group, 5, 3],
        cholesky[group, 5, 4],
        cholesky[group, 5, 5],
    )
    rotation = wp.transform_get_rotation(body_q[body])
    basis = wp.quat_to_matrix(rotation)
    effective = wp.transpose(basis) * (lower * wp.transpose(lower)) * basis
    omega = wp.quat_rotate_inv(rotation, w0)
    predicted = wp.quat_rotate_inv(rotation, predicted_world)

    scale = wp.max(effective[0, 0], wp.max(effective[1, 1], effective[2, 2]))
    a = effective / scale
    physical = inertia / scale
    inverse = wp.inverse(a)
    u0 = predicted + dt * (inverse * wp.cross(omega, physical * omega))
    speed_bound = wp.sqrt(wp.max(wp.dot(u0, a * u0) * wp.trace(inverse), 0.0))
    microsteps = wp.int32(wp.clamp(wp.ceil(2.0 * wp.abs(dt) * speed_bound), 1.0, float(_GYRO_MAX_MICROSTEPS)))
    h = dt / float(microsteps)
    iterates[root_index, 0] = u0
    for m in range(microsteps):
        u = iterates[root_index, m]
        w = u
        energy = wp.dot(u, a * u)
        active = int(1)
        for _iteration in range(8):
            if active != 0:
                s = (0.5 * h) * _gyro_skew(physical * (0.5 * (u + w)))
                candidate = wp.inverse(a - s) * ((a + s) * u)
                delta = candidate - w
                w = candidate
                if wp.dot(delta, a * delta) <= 1.0e-12 * energy:
                    active = 0
        iterates[root_index, m + 1] = w
    corrected = wp.quat_rotate(rotation, iterates[root_index, microsteps])
    for k in range(3):
        v_out[d + 3 + k] = corrected[k]


@wp.kernel
def _augmented_drives_by_dof(
    dof_joint: wp.array[int],
    joint_type: wp.array[int],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_target_q_start: wp.array[int],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_target_ke: wp.array[float],
    joint_target_kd: wp.array[float],
    joint_target_pos: wp.array[float],
    joint_target_vel: wp.array[float],
    joint_effort_limit: wp.array[float],
    tau_in: wp.array[float],
    dt: float,
    # outputs
    tau_out: wp.array[float],
    drive_K: wp.array[float],
):
    """Per-DOF form of prepare_articulation_augmented_drives with the same arithmetic."""
    dof = wp.tid()
    joint = dof_joint[dof]
    type = joint_type[joint]
    value = tau_in[dof]
    K = float(0.0)
    if type == JointType.PRISMATIC or type == JointType.REVOLUTE or type == JointType.D6:
        ke = joint_target_ke[dof]
        kd = joint_target_kd[dof]
        if ke > 0.0 or kd > 0.0:
            K_drive = ke * dt * dt + kd * dt
            if K_drive > 0.0:
                axis = dof - joint_qd_start[joint]
                q = joint_q[joint_q_start[joint] + axis]
                qd = joint_qd[dof]
                target_pos = joint_target_pos[joint_target_q_start[joint] + axis]
                u0 = -(ke * (q - target_pos + dt * qd) + kd * (qd - joint_target_vel[dof]))
                effort_limit = joint_effort_limit[dof]
                if effort_limit > 0.0:
                    u0 = wp.clamp(u0, -effort_limit, effort_limit)
                value = value + u0
                K = K_drive
    tau_out[dof] = value
    drive_K[dof] = K


@wp.kernel
def _crba_dense(
    group_to_art: wp.array[int],
    articulation_dof_start: wp.array[int],
    crba_body: wp.array3d[int],
    crba_dofs: wp.array3d[int],
    joint_S_s: wp.array[wp.spatial_vector],
    body_I_c: wp.array[wp.spatial_matrix],
    drive_K: wp.array[float],
    # outputs
    H_group: wp.array3d[float],
):
    """One generalized-mass entry per thread, as crba_fill_par_dof forms it."""
    group, row, col = wp.tid()
    body = crba_body[group, row, col]
    value = float(0.0)
    if body >= 0:
        S_ancestor = joint_S_s[crba_dofs[group, row, 2 * col]]
        S_descendant = joint_S_s[crba_dofs[group, row, 2 * col + 1]]
        value = wp.dot(S_ancestor, body_I_c[body] * S_descendant)
        if row == col:
            K = drive_K[articulation_dof_start[group_to_art[group]] + row]
            if K > 0.0:
                value += K
    H_group[group, row, col] = value


@wp.func
def _cholesky_lower(
    H: wp.array3d[float], R: wp.array2d[float], g: int, n: int, tmp: wp.array4d[float], L: wp.array3d[float]
):
    """cholesky_loop's factorization of H + diag(R) for one articulation."""
    for j in range(n):
        s = H[g, j, j] + R[g, j]
        for k in range(j):
            r = L[g, j, k]
            s -= r * r
        s = wp.sqrt(s)
        inv_s = 1.0 / s
        L[g, j, j] = s
        for i in range(j + 1, n):
            s = H[g, i, j]
            for k in range(j):
                s -= L[g, i, k] * L[g, j, k]
            L[g, i, j] = s * inv_s


@wp.func_grad(_cholesky_lower)
def _adj_cholesky_lower(
    H: wp.array3d[float], R: wp.array2d[float], g: int, n: int, tmp: wp.array4d[float], L: wp.array3d[float]
):
    # A_bar = L^-T Phi(L^T L_bar) L^-1, accumulated on the lower triangle H reads.
    if not wp.adjoint[H] or not wp.adjoint[L]:
        return
    # tmp[g, 0] = Phi(L^T L_bar): lower triangle, halved diagonal.
    for i in range(n):
        for j in range(n):
            value = float(0.0)
            if j <= i:
                for k in range(i, n):
                    value += L[g, k, i] * wp.adjoint[L][g, k, j]
                if i == j:
                    value *= 0.5
            tmp[g, 0, i, j] = value
    # tmp[g, 1] = L^-T tmp[g, 0] (back substitution down each column).
    for j in range(n):
        for ii in range(n):
            i = n - 1 - ii
            value = tmp[g, 0, i, j]
            for k in range(i + 1, n):
                value -= L[g, k, i] * tmp[g, 1, k, j]
            tmp[g, 1, i, j] = value / L[g, i, i]
    # tmp[g, 0] = tmp[g, 1] L^-1, solving S L = X row by row from the right.
    for i in range(n):
        for jj in range(n):
            j = n - 1 - jj
            value = tmp[g, 1, i, j]
            for k in range(j + 1, n):
                value -= tmp[g, 0, i, k] * L[g, k, j]
            tmp[g, 0, i, j] = value / L[g, j, j]
    for i in range(n):
        wp.adjoint[H][g, i, i] += tmp[g, 0, i, i]
        for j in range(i):
            wp.adjoint[H][g, i, j] += tmp[g, 0, i, j] + tmp[g, 0, j, i]
        for j in range(i + 1):
            wp.adjoint[L][g, i, j] = 0.0


@wp.kernel
def _factor_dense(H: wp.array3d[float], R: wp.array2d[float], n: int, tmp: wp.array4d[float], L: wp.array3d[float]):
    _cholesky_lower(H, R, wp.tid(), n, tmp, L)


@wp.func
def _cholesky_solve(
    L: wp.array3d[float],
    g: int,
    dof_start: int,
    n: int,
    tau: wp.array[float],
    tmp: wp.array[float],
    qdd: wp.array[float],
):
    """trisolve_loop's L L^T qdd = tau for one articulation."""
    for i in range(n):
        value = tau[dof_start + i]
        for k in range(i):
            value -= L[g, i, k] * qdd[dof_start + k]
        L_ii = L[g, i, i]
        if L_ii != 0.0:
            qdd[dof_start + i] = value / L_ii
        else:
            qdd[dof_start + i] = 0.0
    for i_rev in range(n):
        i = n - 1 - i_rev
        value = qdd[dof_start + i]
        for k in range(i + 1, n):
            value -= L[g, k, i] * qdd[dof_start + k]
        L_ii = L[g, i, i]
        if L_ii != 0.0:
            qdd[dof_start + i] = value / L_ii
        else:
            qdd[dof_start + i] = 0.0


@wp.func_grad(_cholesky_solve)
def _adj_cholesky_solve(
    L: wp.array3d[float],
    g: int,
    dof_start: int,
    n: int,
    tau: wp.array[float],
    tmp: wp.array[float],
    qdd: wp.array[float],
):
    # t = (L L^T)^-1 qdd_bar; tau_bar += t; L_bar -= t (L^T qdd)^T + qdd (L^T t)^T on the lower triangle.
    if not wp.adjoint[qdd]:
        return
    for i in range(n):
        value = wp.adjoint[qdd][dof_start + i]
        for k in range(i):
            value -= L[g, i, k] * tmp[dof_start + k]
        tmp[dof_start + i] = value / L[g, i, i]
    for i_rev in range(n):
        i = n - 1 - i_rev
        value = tmp[dof_start + i]
        for k in range(i + 1, n):
            value -= L[g, k, i] * tmp[dof_start + k]
        tmp[dof_start + i] = value / L[g, i, i]
    if wp.adjoint[tau]:
        for i in range(n):
            wp.adjoint[tau][dof_start + i] += tmp[dof_start + i]
    if wp.adjoint[L]:
        for j in range(n):
            y_j = float(0.0)
            u_j = float(0.0)
            for k in range(j, n):
                y_j += L[g, k, j] * qdd[dof_start + k]
                u_j += L[g, k, j] * tmp[dof_start + k]
            for i in range(j, n):
                wp.adjoint[L][g, i, j] -= tmp[dof_start + i] * y_j + qdd[dof_start + i] * u_j
    for i in range(n):
        wp.adjoint[qdd][dof_start + i] = 0.0


@wp.kernel
def _solve_dense(
    L: wp.array3d[float],
    group_to_art: wp.array[int],
    articulation_dof_start: wp.array[int],
    n: int,
    tau: wp.array[float],
    tmp: wp.array[float],
    qdd: wp.array[float],
):
    g = wp.tid()
    _cholesky_solve(L, g, articulation_dof_start[group_to_art[g]], n, tau, tmp, qdd)


@wp.func
def _contact_world_points(
    c: int,
    point0: wp.array[wp.vec3],
    point1: wp.array[wp.vec3],
    normal: wp.vec3,
    body_a: int,
    body_b: int,
    margin0: wp.array[float],
    margin1: wp.array[float],
    body_q: wp.array[wp.transform],
):
    """World witness points as _populate_world_J_for_size_contact forms them."""
    point_a = point0[c] - margin0[c] * normal
    if body_a >= 0:
        point_a = wp.transform_point(body_q[body_a], point0[c]) - margin0[c] * normal
    point_b = point1[c] + margin1[c] * normal
    if body_b >= 0:
        point_b = wp.transform_point(body_q[body_b], point1[c]) + margin1[c] * normal
    return point_a, point_b


@wp.kernel
def _dense_contact_rows(
    contact_count: wp.array[int],
    thread_count: int,
    point0: wp.array[wp.vec3],
    point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    shape0: wp.array[int],
    shape1: wp.array[int],
    margin0: wp.array[float],
    margin1: wp.array[float],
    contact_world: wp.array[int],
    contact_slot: wp.array[int],
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    contact_path: wp.array[int],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    body_v_s: wp.array[wp.spatial_vector],
    prescribed_articulation: wp.array[int],
    articulation_origin: wp.array[wp.vec3],
    shape_material_mu: wp.array[float],
    shape_material_restitution: wp.array[float],
    contact_shared_anchor: int,
    pgs_beta: float,
    pgs_cfm: float,
    # outputs
    row_type: wp.array2d[int],
    row_parent: wp.array2d[int],
    row_mu: wp.array2d[float],
    row_beta: wp.array2d[float],
    row_cfm: wp.array2d[float],
    row_phi: wp.array2d[float],
    row_target_velocity: wp.array2d[float],
):
    """Normal-row metadata of _populate_world_J_for_size_contact, one write per contact."""
    total = wp.min(contact_count[0], point0.shape[0])
    for c in range(wp.tid(), total, thread_count):
        slot = contact_slot[c]
        if contact_path[c] == 0 and slot >= 0:
            world = contact_world[c]
            normal = -contact_normal[c]
            body_a = int(-1)
            body_b = int(-1)
            mu = float(0.0)
            mat_count = int(0)
            if shape0[c] >= 0:
                body_a = shape_body[shape0[c]]
                mu += shape_material_mu[shape0[c]]
                mat_count += 1
            if shape1[c] >= 0:
                body_b = shape_body[shape1[c]]
                mu += shape_material_mu[shape1[c]]
                mat_count += 1
            if mat_count > 0:
                mu /= float(mat_count)
            point_a, point_b = _contact_world_points(
                c, point0, point1, normal, body_a, body_b, margin0, margin1, body_q
            )
            phi = wp.dot(normal, point_a - point_b)
            anchor = 0.5 * (point_a + point_b)
            target_a = point_a
            target_b = point_b
            if contact_shared_anchor != 0:
                target_a = anchor
                target_b = anchor
            row_type[world, slot] = PGS_CONSTRAINT_TYPE_CONTACT
            row_parent[world, slot] = -1
            row_mu[world, slot] = mu
            row_beta[world, slot] = pgs_beta
            row_cfm[world, slot] = pgs_cfm
            row_phi[world, slot] = phi
            row_target_velocity[world, slot] = prescribed_relative_contact_target(
                body_a,
                contact_art_a[c],
                body_b,
                contact_art_b[c],
                target_a,
                target_b,
                normal,
                prescribed_articulation,
                articulation_origin,
                body_v_s,
            )


@wp.func
def _contact_jacobian_entry(
    body: int,
    art: int,
    sign: float,
    point_world: wp.vec3,
    direction: wp.vec3,
    local_dof: int,
    articulation_dof_start: wp.array[int],
    articulation_origin: wp.array[wp.vec3],
    body_dof_chain: wp.array2d[int],
    joint_S_s: wp.array[wp.spatial_vector],
):
    """accumulate_jacobian_row_world's contribution of one DOF, from the chain table."""
    value = float(0.0)
    if body >= 0 and body_dof_chain[body, local_dof] != 0:
        S = joint_S_s[articulation_dof_start[art] + local_dof]
        lin = wp.vec3(S[0], S[1], S[2])
        ang = wp.vec3(S[3], S[4], S[5])
        v = lin + wp.cross(ang, point_world - articulation_origin[art])
        value = sign * wp.dot(direction, v)
    return value


@wp.kernel
def _dense_contact_jacobian(
    contact_count: wp.array[int],
    thread_count: int,
    point0: wp.array[wp.vec3],
    point1: wp.array[wp.vec3],
    contact_normal: wp.array[wp.vec3],
    shape0: wp.array[int],
    shape1: wp.array[int],
    margin0: wp.array[float],
    margin1: wp.array[float],
    contact_slot: wp.array[int],
    contact_art_a: wp.array[int],
    contact_art_b: wp.array[int],
    contact_path: wp.array[int],
    target_size: int,
    articulation_response_dof_count: wp.array[int],
    art_group_idx: wp.array[int],
    articulation_dof_start: wp.array[int],
    articulation_origin: wp.array[wp.vec3],
    body_dof_chain: wp.array2d[int],
    joint_S_s: wp.array[wp.spatial_vector],
    shape_body: wp.array[int],
    body_q: wp.array[wp.transform],
    contact_shared_anchor: int,
    # outputs
    J_group: wp.array3d[float],
):
    """One normal-row Jacobian entry per (contact, DOF), summed in populate_world_J_for_size's order."""
    thread, dof = wp.tid()
    total = wp.min(contact_count[0], point0.shape[0])
    for c in range(thread, total, thread_count):
        slot = contact_slot[c]
        if contact_path[c] == 0 and slot >= 0:
            normal = -contact_normal[c]
            body_a = int(-1)
            body_b = int(-1)
            if shape0[c] >= 0:
                body_a = shape_body[shape0[c]]
            if shape1[c] >= 0:
                body_b = shape_body[shape1[c]]
            point_a, point_b = _contact_world_points(
                c, point0, point1, normal, body_a, body_b, margin0, margin1, body_q
            )
            if contact_shared_anchor != 0:
                point_a = 0.5 * (point_a + point_b)
                point_b = point_a
            art_a = contact_art_a[c]
            art_b = contact_art_b[c]
            a_matches = art_a >= 0 and articulation_response_dof_count[art_a] == target_size
            b_matches = art_b >= 0 and articulation_response_dof_count[art_b] == target_size
            if a_matches:
                value = float(0.0)
                value += _contact_jacobian_entry(
                    body_a,
                    art_a,
                    1.0,
                    point_a,
                    normal,
                    dof,
                    articulation_dof_start,
                    articulation_origin,
                    body_dof_chain,
                    joint_S_s,
                )
                if b_matches and art_b == art_a:
                    value += _contact_jacobian_entry(
                        body_b,
                        art_b,
                        -1.0,
                        point_b,
                        normal,
                        dof,
                        articulation_dof_start,
                        articulation_origin,
                        body_dof_chain,
                        joint_S_s,
                    )
                J_group[art_group_idx[art_a], slot, dof] = value
            if b_matches and art_b != art_a:
                value = float(0.0)
                value += _contact_jacobian_entry(
                    body_b,
                    art_b,
                    -1.0,
                    point_b,
                    normal,
                    dof,
                    articulation_dof_start,
                    articulation_origin,
                    body_dof_chain,
                    joint_S_s,
                )
                J_group[art_group_idx[art_b], slot, dof] = value


@wp.func
def _cholesky_solve_row(
    L: wp.array3d[float],
    g: int,
    row: int,
    n: int,
    rhs: wp.array3d[float],
    tmp: wp.array3d[float],
    x: wp.array3d[float],
):
    """hinv_jt_par_row's L L^T x = rhs for one constraint row of one articulation."""
    for i in range(n):
        value = rhs[g, row, i]
        for k in range(i):
            value -= L[g, i, k] * x[g, row, k]
        L_ii = L[g, i, i]
        if L_ii != 0.0:
            x[g, row, i] = value / L_ii
        else:
            x[g, row, i] = 0.0
    for i_rev in range(n):
        i = n - 1 - i_rev
        value = x[g, row, i]
        for k in range(i + 1, n):
            value -= L[g, k, i] * x[g, row, k]
        L_ii = L[g, i, i]
        if L_ii != 0.0:
            x[g, row, i] = value / L_ii
        else:
            x[g, row, i] = 0.0


@wp.func_grad(_cholesky_solve_row)
def _adj_cholesky_solve_row(
    L: wp.array3d[float],
    g: int,
    row: int,
    n: int,
    rhs: wp.array3d[float],
    tmp: wp.array3d[float],
    x: wp.array3d[float],
):
    # Same adjoint as _adj_cholesky_solve, on one row of grouped storage.
    if not wp.adjoint[x]:
        return
    for i in range(n):
        value = wp.adjoint[x][g, row, i]
        for k in range(i):
            value -= L[g, i, k] * tmp[g, row, k]
        tmp[g, row, i] = value / L[g, i, i]
    for i_rev in range(n):
        i = n - 1 - i_rev
        value = tmp[g, row, i]
        for k in range(i + 1, n):
            value -= L[g, k, i] * tmp[g, row, k]
        tmp[g, row, i] = value / L[g, i, i]
    if wp.adjoint[rhs]:
        for i in range(n):
            wp.adjoint[rhs][g, row, i] += tmp[g, row, i]
    if wp.adjoint[L]:
        for j in range(n):
            y_j = float(0.0)
            u_j = float(0.0)
            for k in range(j, n):
                y_j += L[g, k, j] * x[g, row, k]
                u_j += L[g, k, j] * tmp[g, row, k]
            for i in range(j, n):
                wp.adjoint[L][g, i, j] -= tmp[g, row, i] * y_j + x[g, row, i] * u_j
    for i in range(n):
        wp.adjoint[x][g, row, i] = 0.0


@wp.kernel
def _hinv_jt_dense(
    L: wp.array3d[float],
    J: wp.array3d[float],
    group_to_art: wp.array[int],
    art_to_world: wp.array[int],
    row_count: wp.array[int],
    n: int,
    max_rows: int,
    tmp: wp.array3d[float],
    Y: wp.array3d[float],
):
    """Y = H^-1 J^T per (articulation, row), as hinv_jt_par_row computes it."""
    tid = wp.tid()
    row = tid % max_rows
    g = tid // max_rows
    if row < row_count[art_to_world[group_to_art[g]]]:
        _cholesky_solve_row(L, g, row, n, J, tmp, Y)


@wp.kernel
def _delassus_dense(
    J_group: wp.array3d[float],
    Y_group: wp.array3d[float],
    group_to_art: wp.array[int],
    art_to_world: wp.array[int],
    row_count: wp.array[int],
    n_dofs: int,
    max_rows: int,
    n_arts: int,
    # outputs
    world_C: wp.array3d[float],
    world_diag: wp.array2d[float],
):
    """delassus_par_row_col without its nonzero gate, which the reverse pass evaluates before the loop sum."""
    tid = wp.tid()
    j = tid % max_rows
    i = (tid // max_rows) % max_rows
    idx = tid // (max_rows * max_rows)
    if idx >= n_arts:
        return
    world = art_to_world[group_to_art[idx]]
    m = row_count[world]
    if i >= m or j >= m:
        return
    val = float(0.0)
    for k in range(n_dofs):
        val += J_group[idx, i, k] * Y_group[idx, j, k]
    wp.atomic_add(world_C, world, i, j, val)
    if i == j:
        wp.atomic_add(world_diag, world, i, val)


@wp.kernel
def _dense_pgs_sweep(
    row_count: wp.array[int],
    diag: wp.array2d[float],
    C: wp.array3d[float],
    rhs: wp.array2d[float],
    row_type: wp.array2d[int],
    omega: float,
    impulses_in: wp.array2d[float],
    # outputs
    residuals: wp.array2d[float],
    impulses_out: wp.array2d[float],
):
    """One pgs_solve_loop sweep over normal rows; earlier rows of this sweep read impulses_out.

    Each residual is stored and read back: the reverse pass replays code after a dynamic loop with
    the loop-carried sum's initial value, so the projection must branch on the stored residual.
    """
    world = wp.tid()
    m = row_count[world]
    for i in range(m):
        w = rhs[world, i]
        for j in range(m):
            impulse = impulses_in[world, j]
            if j < i:
                impulse = impulses_out[world, j]
            w += C[world, i, j] * impulse
        residuals[world, i] = w
        residual = residuals[world, i]
        new_impulse = impulses_in[world, i]
        denom = diag[world, i]
        if denom > 0.0:
            new_impulse = impulses_in[world, i] + omega * (-residual / denom)
            if row_type[world, i] == PGS_CONSTRAINT_TYPE_CONTACT and new_impulse < 0.0:
                new_impulse = 0.0
        impulses_out[world, i] = new_impulse
