# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Paired articulation/free-body response of SolverFeatherPGS."""

import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS


def _build_mixed_response_model(
    device,
    world_count=1,
    *,
    dof_count=13,
    friction=0.0,
    restitution=0.0,
    static_plane=False,
    static_support=False,
    free_body_first=False,
    free_body_velocity_limit=None,
):
    """Build one serial articulation contacting one free rigid body.

    ``static_support`` rests the free body on a static ledge that clears the arm, so the free body also
    produces matrix-free contact rows. ``free_body_first`` builds the free body before the articulation,
    which packs the world DOFs free-body first. ``free_body_velocity_limit`` caps the free body's linear
    and angular velocity [m/s, rad/s], producing matrix-free velocity-limit rows once exceeded.
    ``static_plane`` adds a ground plane under the free body.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    if free_body_velocity_limit is not None:
        SolverFeatherPGS.register_custom_attributes(builder)
    builder.default_shape_cfg.density = 1000.0
    builder.default_shape_cfg.ke = 1.0e5
    builder.default_shape_cfg.kd = 1.0e3
    builder.default_shape_cfg.mu = friction
    builder.default_shape_cfg.restitution = restitution
    builder.default_shape_cfg.margin = 0.0
    builder.default_shape_cfg.gap = 0.0

    def add_free_body():
        custom_attributes = None
        if free_body_velocity_limit is not None:
            custom_attributes = {
                "rigid_body_max_linear_velocity": free_body_velocity_limit,
                "rigid_body_max_angular_velocity": free_body_velocity_limit,
            }
        box = builder.add_link(
            xform=wp.transform(wp.vec3(0.7, 0.0, 0.5695), wp.quat_identity()), custom_attributes=custom_attributes
        )
        builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.05)
        builder.add_articulation([builder.add_joint_free(parent=-1, child=box)])
        if static_support:
            builder.add_shape_box(
                -1,
                xform=wp.transform(wp.vec3(0.7, 0.11, 0.47), wp.quat_identity()),
                hx=0.05,
                hy=0.05,
                hz=0.05,
            )

    if free_body_first:
        add_free_body()
    arm = builder.add_link()
    builder.add_shape_box(arm, hx=0.4, hy=0.05, hz=0.02)
    joints = [
        builder.add_joint_revolute(
            parent=-1,
            child=arm,
            axis=wp.vec3(0.0, 1.0, 0.0),
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.4, 0.0, 0.0), wp.quat_identity()),
        )
    ]
    parent = arm
    for index in range(dof_count - 1):
        child = builder.add_link(
            mass=0.05,
            inertia=wp.mat33(np.eye(3, dtype=np.float32) * 1.0e-3),
            lock_inertia=True,
        )
        joints.append(
            builder.add_joint_revolute(
                parent=parent,
                child=child,
                axis=(newton.Axis.X, newton.Axis.Y, newton.Axis.Z)[index % 3],
            )
        )
        parent = child
    builder.add_articulation(joints)
    builder.set_joint_mimic(joints[1], joints[0])

    if not free_body_first:
        add_free_body()
    if static_plane:
        builder.add_shape_plane(plane=(0.0, 0.0, 1.0, -0.53))
    if world_count == 1:
        return builder.finalize(device=device)
    replicated = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    replicated.replicate(builder, world_count, spacing=(3.0, 0.0, 0.0))
    return replicated.finalize(device=device)


def _run_mixed_response(
    kernel,
    *,
    warmstart,
    preelimination,
    dof_count=13,
    dense_max_constraints=32,
    inactive_joint_limit_capacity=False,
    friction=0.0,
    restitution=0.0,
    tangential_velocity=0.0,
    static_plane=False,
    contact_regularization=0.0,
    friction_anchor_beta=None,
    model_kwargs=None,
):
    """Run a short mixed-contact trajectory with one H-inverse implementation."""
    model = _build_mixed_response_model(
        "cuda:0",
        dof_count=dof_count,
        friction=friction,
        restitution=restitution,
        static_plane=static_plane,
        **(model_kwargs or {}),
    )
    with mock.patch.object(SolverFeatherPGS, "_kernel_overrides", {"hinv_jt_kernel": kernel}):
        solver = SolverFeatherPGS(
            model,
            pgs_mode="matrix_free",
            pgs_warmstart=warmstart,
            enable_bilateral_preelimination=preelimination,
            enable_joint_limits=inactive_joint_limit_capacity,
            joint_limit_activation_gap=0.0,
            pgs_iterations=8,
            pgs_contact_regularization=contact_regularization,
            friction_anchor_beta=friction_anchor_beta,
            dense_max_constraints=dense_max_constraints,
            mf_max_constraints=32,
        )
    state_in, state_out = model.state(), model.state()
    joint_qd = state_in.joint_qd.numpy()
    free_articulation = int(np.flatnonzero(solver._model_plan.is_free_rigid)[0])
    free_dof_start = int(solver._model_plan.articulation_dof_start[free_articulation])
    joint_qd[free_dof_start] = tangential_velocity
    joint_qd[free_dof_start + 2] = -3.0
    state_in.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)

    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        reduce_contacts=False,
        contact_matching="latest",
    )
    contacts = pipeline.contacts()
    control = model.control()
    samples = []
    for _ in range(4):
        state_in.clear_forces()
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, 1.0 / 240.0)
        constraint_count = int(solver.constraint_count.numpy()[0])
        mf_count = int(solver.mf_constraint_count.numpy()[0])
        samples.append(
            (
                constraint_count,
                solver.diag.numpy()[0, :constraint_count].copy(),
                solver.impulses.numpy()[0, :constraint_count].copy(),
                state_out.joint_q.numpy().copy(),
                state_out.joint_qd.numpy().copy(),
                0,
                solver.row_type.numpy()[0, :constraint_count].copy(),
                mf_count,
                solver.mf_impulses.numpy()[0, :mf_count].copy(),
                solver.mf_row_type.numpy()[0, :mf_count].copy(),
                solver.mf_row_parent.numpy()[0, :mf_count].copy(),
                solver.mf_J_a.numpy()[0, :mf_count].copy(),
                solver.mf_J_b.numpy()[0, :mf_count].copy(),
                solver.mf_MiJt_a.numpy()[0, :mf_count].copy(),
                solver.mf_MiJt_b.numpy()[0, :mf_count].copy(),
            )
        )
        state_in, state_out = state_out, state_in
    return solver, samples


class TestFeatherPGSPairedResponse(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "paired response ownership requires CUDA")
    def test_paired_response_without_factor_coordinates_matches_general(self):
        """Match the general response with the explicit inverses that velocity limits keep in physical coordinates."""
        run_kwargs = {
            "warmstart": False,
            "preelimination": False,
            "dof_count": 23,
            "dense_max_constraints": 96,
            "inactive_joint_limit_capacity": True,
            "friction": 0.7,
            "restitution": 0.3,
            "tangential_velocity": 2.0,
            "friction_anchor_beta": 0.0,
            "contact_regularization": 0.01,
        }
        reference_solver, reference = _run_mixed_response("par_row", **run_kwargs)
        paired_solver, paired = _run_mixed_response("auto", **run_kwargs)
        self.assertIsNone(reference_solver._paired_response_primary_size)
        self.assertEqual(paired_solver._paired_response_primary_size, 23)
        self.assertFalse(paired_solver._paired_factor_coordinates)
        self.assertIsNone(paired_solver._paired_factor_solve_kernel)
        self.assertIsNotNone(paired_solver._paired_cholesky_inverse_kernel)
        for step, (expected, actual) in enumerate(zip(reference, paired, strict=True)):
            self.assertEqual(actual[0], expected[0], f"constraint count differed at step {step}")
            for label, expected_value, actual_value in zip(
                ("diagonal", "impulses", "joint_q", "joint_qd"), expected[1:5], actual[1:5], strict=True
            ):
                np.testing.assert_allclose(
                    actual_value, expected_value, rtol=5.0e-4, atol=1.0e-5, err_msg=f"{label} differed at step {step}"
                )


if __name__ == "__main__":
    unittest.main()
