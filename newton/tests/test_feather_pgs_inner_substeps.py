# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Frozen-basis inner substeps (pgs_inner_substeps): physics equivalence to
true substeps at shared-manifold semantics, and constructor validation."""

import unittest

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
    "test_inner_substeps_validation",
    test_inner_substeps_validation,
    devices=devices,
)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
