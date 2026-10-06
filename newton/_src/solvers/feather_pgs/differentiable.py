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
    _compute_body_net_wrench,
    _gyro_skew,
    apply_free_root_transport_to_predictor,
    compute_composite_inertia,
    compute_link_transform,
    compute_link_velocity,
    compute_velocity_predictor,
    integrate_generalized_joints,
    remove_free_root_transport_from_qdd,
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
            raise NotImplementedError("differentiable=True does not support contacts yet; pass contacts=None")
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
        # Without constraint rows the PGS stages leave v_out == v_hat.
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
        self.H = {}
        self.L = {}
        self.factor_tmp = {}
        for size in solver.size_groups:
            n_arts = solver.n_arts_by_size[size]
            self.H[size] = zeros((n_arts, size, size))
            self.L[size] = zeros((n_arts, size, size))
            self.factor_tmp[size] = zeros((n_arts, 2, size, size))


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
