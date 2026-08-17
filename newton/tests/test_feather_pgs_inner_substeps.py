# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Frozen-basis inner substeps (pgs_inner_substeps): physics equivalence to
true substeps at shared-manifold semantics, and constructor validation."""

import os
import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_test_devices

SOLVER_KW = {
    "pgs_mode": "matrix_free",
    "articulated_contact_response": "immediate",
    "pgs_iterations": 2,
    "pgs_beta": 0.08,
    "pgs_warmstart": False,
    "mf_warmstart": False,
}


def _build_stack(device, n_boxes=3):
    builder = newton.ModelBuilder()
    for i in range(n_boxes):
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.35 + 0.22 * i), wp.quat_identity()), mass=1.0)
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    builder.add_ground_plane()
    return builder.finalize(device=device)


def _run(model, solver, steps, dt, substeps=1):
    """Step with one collide per outer step (shared-manifold substep semantics)."""
    state_in, state_out = model.state(), model.state()
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    control = model.control()
    contacts = model.contacts()
    for _ in range(steps):
        model.collide(state_in, contacts)
        for _ in range(substeps):
            state_in.clear_forces()
            solver.step(state_in, state_out, control, contacts, dt / substeps)
            state_in, state_out = state_out, state_in
    return state_in


def test_inner_substeps_match_true_substeps(test, device):
    """Verify frozen inner substeps settle a box stack like true shared-manifold substeps.

    Drops a 3-box stack onto the ground and compares final poses and residual
    speeds between pgs_inner_substeps=8 (one build per step) and 8 true solver
    substeps sharing one manifold per step (the production-stack semantics).
    """
    steps, dt = 160, 0.005

    model_true = _build_stack(device)
    solver_true = SolverFeatherPGS(model_true, **SOLVER_KW)
    state_true = _run(model_true, solver_true, steps, dt, substeps=8)

    model_frozen = _build_stack(device)
    solver_frozen = SolverFeatherPGS(model_frozen, pgs_inner_substeps=8, **SOLVER_KW)
    state_frozen = _run(model_frozen, solver_frozen, steps, dt, substeps=1)

    z_true = state_true.body_q.numpy()[:, 2]
    z_frozen = state_frozen.body_q.numpy()[:, 2]
    test.assertTrue(
        np.allclose(z_frozen, z_true, atol=2.0e-3),
        f"settled heights diverged: frozen={z_frozen} true={z_true}",
    )
    speed_frozen = float(np.abs(state_frozen.body_qd.numpy()).max())
    test.assertLess(speed_frozen, 0.05, f"stack not at rest: max |body_qd| = {speed_frozen}")


def _build_bolted_arm(device):
    """Fixed-base 2-link driven arm: dense drive rows dominate the error budget.

    No contacts, so the drive geometric-error advance is the only live term in
    ``advance_frozen_row_errors`` -- deleting or sign-flipping it is measurable here,
    where a contact-dominated scene buries it in the frozen-basis contact noise floor.
    """
    builder = newton.ModelBuilder()
    parent = -1
    z = 0.5
    joints = []
    for i in range(2):
        link = builder.add_link()
        builder.add_shape_box(link, hx=0.04, hy=0.04, hz=0.12)
        joints.append(
            builder.add_joint_revolute(
                parent=parent,
                child=link,
                axis=newton.ModelBuilder.JointDofConfig(
                    axis=wp.vec3(0.0, 1.0, 0.0),
                    target_pos=0.4,
                    target_ke=40.0,
                    target_kd=2.0,
                    limit_lower=-1.0,
                    limit_upper=1.0,
                ),
                parent_xform=wp.transform(wp.vec3(0.0, 0.0, z if i == 0 else -0.24), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.12), wp.quat_identity()),
            )
        )
        parent = link
    builder.add_articulation(joints)
    return builder.finalize(device=device)


def _build_driven_arm(device):
    """Free-root articulation with a driven joint, dropped onto the ground.

    Starts clear of the ground (a penetrating initial pose ejects at any substep count,
    which would make the comparison measure an instability rather than tracking fidelity).

    Contacts on an articulated body take the dense row family, so this scene populates
    dense CONTACT rows -- the branch a free-rigid box stack (matrix-free only) and the
    bolted arm (contact-free) can never reach.
    """
    builder = newton.ModelBuilder()
    # Tilt the drop so the base lands edge-first and rocks: a flat symmetric landing
    # exerts no torque about the revolute axis, leaving the drive and limit dofs
    # motionless (jv = 0) and their error advances untestable no-ops.
    tilt = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), 0.05)
    base = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.4), tilt))
    builder.add_shape_box(base, hx=0.12, hy=0.12, hz=0.06)
    joints = [builder.add_joint_free(child=base)]
    link = builder.add_link()
    builder.add_shape_box(link, hx=0.05, hy=0.05, hz=0.12)
    joints.append(
        builder.add_joint_revolute(
            parent=base,
            child=link,
            axis=newton.ModelBuilder.JointDofConfig(
                axis=wp.vec3(0.0, 1.0, 0.0),
                target_pos=1.4,
                target_ke=40.0,
                target_kd=2.0,
                limit_lower=-1.0,
                limit_upper=1.0,
            ),
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.06), wp.quat_identity()),
            # off-axis link COM: the ground impact torques the revolute dof, so the
            # limit and drive rows see real joint velocity during the transient
            child_xform=wp.transform(wp.vec3(0.08, 0.0, -0.12), wp.quat_identity()),
        )
    )
    builder.add_articulation(joints)
    builder.add_ground_plane()
    return builder.finalize(device=device)


def _run_recording(model, solver, steps, dt, substeps=1):
    """Step like :func:`_run` but record ``joint_q`` after every outer step."""
    state_in, state_out = model.state(), model.state()
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    control = model.control()
    contacts = model.contacts()
    trajectory = []
    for _ in range(steps):
        model.collide(state_in, contacts)
        for _ in range(substeps):
            state_in.clear_forces()
            solver.step(state_in, state_out, control, contacts, dt / substeps)
            state_in, state_out = state_out, state_in
        trajectory.append(state_in.joint_q.numpy().copy())
    return np.stack(trajectory)


def test_inner_substeps_dense_contact_rows_match_true_substeps(test, device):
    """Verify frozen inner substeps track true substeps on dense CONTACT rows.

    The box-stack case exercises only the matrix-free row family; contacts on an
    articulated body take the dense family instead. The error is taken over the whole
    trajectory rather than the settled end state: the ground impact (~0.34 m free fall)
    is where the contact rows are dynamically loaded and the substep count changes the
    outcome, while at rest the error advance multiplies near-zero velocities.

    Mutation-verified for the contact-phi advance (sign flip, deletion, half scale all
    fail this assertion). The drive advance is covered by the bolted-arm case below; the
    JOINT_LIMIT clause of the position-row gate has no discriminating test -- a
    statically pinned limit row has jv = 0 (its advance is a genuine no-op), and a scene
    with dynamic limit engagement lands in the frozen basis's documented weak regime.
    """
    steps, dt = 120, 0.005
    kw = dict(SOLVER_KW, drive_mode="physx_pgs", enable_joint_limits=True, pgs_iterations=8)

    def run(solver_kwargs, substeps):
        model = _build_driven_arm(device)
        return _run_recording(model, SolverFeatherPGS(model, **solver_kwargs), steps, dt, substeps=substeps)

    q_true = run(kw, 8)
    q_frozen = run(dict(kw, pgs_inner_substeps=8), 1)
    q_one = run(kw, 1)

    test.assertTrue(np.all(np.isfinite(q_true)) and np.all(np.isfinite(q_frozen)), "arm scene diverged")
    err_frozen = float(np.abs(q_frozen - q_true).max())
    err_one = float(np.abs(q_one - q_true).max())
    test.assertLess(
        err_frozen,
        0.25 * err_one,
        f"frozen substeps should track true substeps far better than a single step "
        f"(frozen {err_frozen:.2e} vs one step {err_one:.2e})",
    )


def test_inner_substeps_dense_drive_rows_match_true_substeps(test, device):
    """Verify frozen inner substeps track true substeps on dense drive rows.

    Contact-free bolted arm, so the drive geometric-error advance dominates the error
    budget. Mutation-verified: deleting or sign-flipping the drive update fails this
    assertion (it escapes the contact scene's assertion, where the frozen-basis contact
    noise floor is larger than the whole drive contribution).
    """
    steps, dt = 120, 0.005
    kw = dict(SOLVER_KW, drive_mode="physx_pgs", enable_joint_limits=True, pgs_iterations=8)

    def run(solver_kwargs, substeps):
        model = _build_bolted_arm(device)
        state = _run(model, SolverFeatherPGS(model, **solver_kwargs), steps, dt, substeps=substeps)
        return state.joint_q.numpy().copy()

    q_true = run(kw, 8)
    q_frozen = run(dict(kw, pgs_inner_substeps=8), 1)
    q_one = run(kw, 1)

    test.assertTrue(np.all(np.isfinite(q_true)) and np.all(np.isfinite(q_frozen)), "arm scene diverged")
    err_frozen = float(np.abs(q_frozen - q_true).max())
    err_one = float(np.abs(q_one - q_true).max())
    test.assertLess(
        err_frozen,
        0.25 * err_one,
        f"frozen substeps should track true substeps far better than a single step "
        f"(frozen {err_frozen:.2e} vs one step {err_one:.2e})",
    )


def test_inner_substeps_validation(test, device):
    """Verify pgs_inner_substeps rejects unsupported solver configurations."""
    model = _build_stack(device, n_boxes=1)
    with test.assertRaises(ValueError):
        SolverFeatherPGS(model, pgs_inner_substeps=0, **SOLVER_KW)
    with test.assertRaises(ValueError):
        SolverFeatherPGS(model, pgs_inner_substeps=8, pgs_velocity_iterations=2, **SOLVER_KW)
    bad_mode = {**SOLVER_KW, "pgs_mode": "split"}
    with test.assertRaises(ValueError):
        SolverFeatherPGS(model, pgs_inner_substeps=8, **bad_mode)
    # every non-immediate response routes contacts through propagation rows
    for response in ("propagation", "propagation-fused", "propagation-colored"):
        with test.assertRaises(ValueError):
            SolverFeatherPGS(model, pgs_inner_substeps=8, **dict(SOLVER_KW, articulated_contact_response=response))
    with test.assertRaises(ValueError):
        SolverFeatherPGS(model, pgs_inner_substeps=8, **dict(SOLVER_KW, mf_warmstart=True))
    with test.assertRaises(ValueError):
        SolverFeatherPGS(model, pgs_inner_substeps=8, **dict(SOLVER_KW, pgs_debug=True))
    with test.assertRaises(ValueError):
        SolverFeatherPGS(model, pgs_inner_substeps=8, **dict(SOLVER_KW, pgs_warmstart=True))
    # the resolved warm-start flag is what matters: the env var must not bypass the guard
    with mock.patch.dict(os.environ, {"IL_NEWTON_FPGS_MF_WARMSTART": "1"}):
        with test.assertRaises(ValueError):
            SolverFeatherPGS(model, pgs_inner_substeps=8, **SOLVER_KW)
    # the supported configuration still constructs (matrix-free mode is CUDA-only)
    if device.is_cuda:
        SolverFeatherPGS(model, pgs_inner_substeps=8, **SOLVER_KW)


devices = get_test_devices(mode="basic")
# The matrix-free PGS mode is CUDA-only; the physics equivalence test cannot run on CPU.
cuda_devices = [d for d in devices if d.is_cuda]


class TestFeatherPGSInnerSubsteps(unittest.TestCase):
    pass


add_function_test(
    TestFeatherPGSInnerSubsteps,
    "test_inner_substeps_match_true_substeps",
    test_inner_substeps_match_true_substeps,
    devices=cuda_devices,
)
add_function_test(
    TestFeatherPGSInnerSubsteps,
    "test_inner_substeps_dense_contact_rows_match_true_substeps",
    test_inner_substeps_dense_contact_rows_match_true_substeps,
    devices=cuda_devices,
)
add_function_test(
    TestFeatherPGSInnerSubsteps,
    "test_inner_substeps_dense_drive_rows_match_true_substeps",
    test_inner_substeps_dense_drive_rows_match_true_substeps,
    devices=cuda_devices,
)
add_function_test(
    TestFeatherPGSInnerSubsteps,
    "test_inner_substeps_validation",
    test_inner_substeps_validation,
    devices=devices,
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
