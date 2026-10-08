# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.kernels import crba_fill_par_dof
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_test_devices

# Free root, then two ball joints: every joint has several DOFs.
JOINT_DOF_DIMS = ((3, 3), (0, 3), (0, 3))
CRBA_ARTICULATIONS = 256
CRBA_LAUNCHES = 8


def test_crba_fill_par_dof_is_batch_invariant(test, device):
    """Identical articulations get bitwise-identical, exactly symmetric mass matrices on every launch."""
    joint_count = len(JOINT_DOF_DIMS)
    dof_count = sum(lin + ang for lin, ang in JOINT_DOF_DIMS)
    rng = np.random.default_rng(7)
    # Generic motion subspaces and inertias, so S_r.(I S_c) and S_c.(I S_r) round differently.
    motion = rng.normal(size=(dof_count, 6)).astype(np.float32)
    inertia = rng.normal(size=(joint_count, 6, 6))
    inertia = (inertia + np.swapaxes(inertia, 1, 2)).astype(np.float32)

    n = CRBA_ARTICULATIONS
    joint_qd_start = np.concatenate(([0], np.cumsum([lin + ang for lin, ang in JOINT_DOF_DIMS] * n))).astype(np.int32)
    joint_ancestor = np.array([j - 1 if j % joint_count else -1 for j in range(n * joint_count)], dtype=np.int32)
    inputs = [
        wp.array(np.arange(n + 1, dtype=np.int32) * joint_count, dtype=wp.int32, device=device),
        wp.array(np.arange(n, dtype=np.int32) * dof_count, dtype=wp.int32, device=device),
        wp.ones(n, dtype=wp.int32, device=device),
        wp.array(joint_ancestor, dtype=wp.int32, device=device),
        wp.array(np.arange(n * joint_count, dtype=np.int32), dtype=wp.int32, device=device),
        wp.array(joint_qd_start, dtype=wp.int32, device=device),
        wp.array(np.tile(np.array(JOINT_DOF_DIMS, dtype=np.int32), (n, 1)), dtype=wp.int32, device=device),
        wp.array(np.tile(motion, (n, 1)), dtype=wp.spatial_vector, device=device),
        wp.array(np.tile(inertia, (n, 1, 1)), dtype=wp.spatial_matrix, device=device),
        wp.array(np.arange(n, dtype=np.int32), dtype=wp.int32, device=device),
        dof_count,
        0,
        wp.full(n * dof_count, -1, dtype=wp.int32, device=device),
        wp.zeros(1, dtype=wp.float32, device=device),
    ]

    reference = None
    for _launch in range(CRBA_LAUNCHES):
        H = wp.zeros((n, dof_count, dof_count), dtype=wp.float32, device=device)
        wp.launch(crba_fill_par_dof, dim=n * dof_count, inputs=inputs, outputs=[H], device=device, block_dim=128)
        H = H.numpy()
        np.testing.assert_array_equal(H, np.swapaxes(H, 1, 2), err_msg="H is not exactly symmetric")
        np.testing.assert_array_equal(H, np.broadcast_to(H[0], H.shape), err_msg="identical articulations differ")
        if reference is None:
            reference = H
        np.testing.assert_array_equal(H, reference, err_msg="H differs between launches")


def _build_pendulum_model(device):
    """Three-link pendulum in two worlds: its mass matrix changes every step."""
    env = newton.ModelBuilder()
    parent = -1
    joints = []
    for link_index in range(3):
        link = env.add_link()
        env.add_shape_box(link, hx=0.2, hy=0.04, hz=0.04)
        joints.append(
            env.add_joint_revolute(
                parent,
                link,
                parent_xform=wp.transform(wp.vec3(0.2 if parent >= 0 else 0.0, 0.0, 0.0), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
                axis=newton.Axis.Y if link_index != 1 else newton.Axis.Z,
            )
        )
        parent = link
    env.add_articulation(joints)
    builder = newton.ModelBuilder()
    builder.replicate(env, 2)
    model = builder.finalize(device=device)
    model.joint_q.assign(np.tile(np.array([0.7, -0.4, 0.3], dtype=np.float32), 2))
    return model


def _simulate(model, solver, steps):
    state_0, state_1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    control = model.control()
    for _ in range(steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, 1.0 / 60.0)
        state_0, state_1 = state_1, state_0
    return state_0.body_q.numpy(), state_0.body_qd.numpy()


def test_full_reset_matches_fresh_solver(test, device):
    """After a full reset, a solver replays a fresh solver's trajectory bit for bit."""
    for interval in (1, 3):
        with test.subTest(update_mass_matrix_interval=interval):
            model = _build_pendulum_model(device)
            reused = SolverFeatherPGS(model, update_mass_matrix_interval=interval)
            _simulate(model, reused, 5)
            reused.reset(model.state())
            body_q, body_qd = _simulate(model, reused, 7)

            fresh_body_q, fresh_body_qd = _simulate(
                model, SolverFeatherPGS(model, update_mass_matrix_interval=interval), 7
            )
            np.testing.assert_array_equal(body_q, fresh_body_q)
            np.testing.assert_array_equal(body_qd, fresh_body_qd)


class TestFeatherPGSDeterminism(unittest.TestCase):
    pass


for _device in get_test_devices():
    add_function_test(
        TestFeatherPGSDeterminism,
        "test_crba_fill_par_dof_is_batch_invariant",
        test_crba_fill_par_dof_is_batch_invariant,
        devices=[_device],
    )
    add_function_test(
        TestFeatherPGSDeterminism,
        "test_full_reset_matches_fresh_solver",
        test_full_reset_matches_fresh_solver,
        devices=[_device],
    )


if __name__ == "__main__":
    unittest.main()
