# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sleeping of SolverFeatherPGS under a manipulation profile: friction patches, regularization, many iterations."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices

# A manipulation profile without contact torsion; the torsion variant is in test_feather_pgs_sleeping_torsion.
PROFILE = {
    "pgs_iterations": 32,
    "pgs_contact_regularization": 0.01,
    "friction_anchor_beta": 0.2,
    "contact_friction_gap_threshold": 0.001,
    "dense_max_constraints": 512,
    "mf_max_constraints": 512,
    "enable_sleeping": True,
    "sleep_quiet_time": 0.1,
}


def test_anchored_articulations_sleep_and_wake(test, device):
    """Drop every row of a settled scene and restore the anchored rows of a force-woken articulation."""
    model, pipeline, solver, states, control = _articulations(device)
    _advance(pipeline, solver, states, control, 5)
    awake_rows = int(solver.constraint_count.numpy()[0])
    test.assertGreater(awake_rows, 0)
    _advance(pipeline, solver, states, control, 400)
    np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [0, 0, 0, 0])
    np.testing.assert_array_equal(solver.constraint_count.numpy(), [0])
    np.testing.assert_array_equal(solver.mf_constraint_count.numpy(), [0])
    frozen = states[0].body_q.numpy().copy()
    _advance(pipeline, solver, states, control, 20)
    np.testing.assert_array_equal(states[0].body_q.numpy(), frozen)

    # Pushing the first articulation restores its rows and leaves the second asleep.
    force = np.zeros((model.body_count, 6), dtype=np.float32)
    force[0, 0] = 20.0
    for _ in range(5):
        states[0].body_f.assign(force)
        _advance(pipeline, solver, states, control, 1, clear=False)
    np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [1, 1, 0, 0])
    test.assertEqual(int(solver.constraint_count.numpy()[0]), awake_rows // 2)
    test.assertGreater(states[0].body_q.numpy()[0, 0], frozen[0, 0])
    np.testing.assert_array_equal(states[0].body_q.numpy()[2:], frozen[2:])

    _advance(pipeline, solver, states, control, 600)
    np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [0, 0, 0, 0])
    test.assertFalse(np.any(solver.constraint_overflow.numpy()))


def test_graph_replay_sleeps_and_wakes(test, device):
    """Sleep inside a captured graph and wake on an explicit wake before the next replay."""
    model, pipeline, solver, states, control = _articulations(device)
    contacts = pipeline.contacts()
    _advance(pipeline, solver, states, control, 2, contacts=contacts)
    with wp.ScopedCapture(device=model.device) as capture:
        for _ in range(2):
            states[0].clear_forces()
            pipeline.collide(states[0], contacts)
            solver.step(states[0], states[1], control, contacts, 0.005)
            states.reverse()
    for _ in range(200):
        wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [0, 0, 0, 0])
    np.testing.assert_array_equal(solver.constraint_count.numpy(), [0])
    solver.sleeping.wake()
    wp.capture_launch(capture.graph)
    test.assertGreater(int(solver.constraint_count.numpy()[0]), 0)


def test_frozen_patch_carry_copies_the_previous_history(test, device):
    """Take a sleeping pair's anchor history from the previous step, not from the current frame's storage."""
    _model, pipeline, solver, states, control = _articulations(device)
    contacts = pipeline.contacts()
    _advance(pipeline, solver, states, control, 400, contacts=contacts)
    np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [0, 0, 0, 0])
    test.assertTrue(np.all(solver.sleeping.frozen_bodies.numpy() == 1))
    current = solver._friction_patches.current
    count = int(contacts.rigid_contact_count.numpy()[0])
    names = ("valid", "displacement", "tangent_impulse", "anchor_a", "anchor_b", "owner")
    before = {name: getattr(current, name).numpy()[:count].copy() for name in names}
    test.assertGreater(int(before["valid"].sum()), 0)
    # Overwrite the current frame; the carry must restore every field from the stored history.
    for name in names:
        getattr(current, name).fill_(7)
    _advance(pipeline, solver, states, control, 1, contacts=contacts)
    for name in names:
        np.testing.assert_array_equal(getattr(current, name).numpy()[:count], before[name], err_msg=name)


def _articulations(device, tiles=False, profile=None):
    """Two undriven two-link articulations resting on the ground, which route contacts to dense rows.

    With ``tiles``, each base is a grid of small boxes so its ground pair exceeds the warp-flood threshold.
    """
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    for x in (0.0, 1.0):
        base = builder.add_link(xform=wp.transform((x, 0.0, 0.1), wp.quat_identity()))
        if tiles:
            for i in range(4):
                for j in range(3):
                    offset = wp.transform((-0.15 + 0.1 * i, -0.067 + 0.067 * j, 0.0), wp.quat_identity())
                    builder.add_shape_box(base, xform=offset, hx=0.05, hy=0.033, hz=0.1)
        else:
            builder.add_shape_box(base, hx=0.2, hy=0.1, hz=0.1)
        tip = builder.add_link(xform=wp.transform((x + 0.3, 0.0, 0.1), wp.quat_identity()))
        builder.add_shape_box(tip, hx=0.1, hy=0.1, hz=0.1)
        hinge = wp.transform((0.3, 0.0, 0.0), wp.quat_identity())
        builder.add_articulation(
            [
                builder.add_joint_free(child=base),
                builder.add_joint_revolute(parent=base, child=tip, parent_xform=hinge, axis=(0.0, 1.0, 0.0)),
            ]
        )
    model = builder.finalize(device=device)
    # Deterministic contact order, as in production, so A/B trajectories compare bitwise.
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=512, deterministic=True)
    solver = newton.solvers.SolverFeatherPGS(model, pgs_mode="matrix_free", **(profile or PROFILE))
    return model, pipeline, solver, [model.state(), model.state()], model.control()


def _advance(pipeline, solver, states, control, steps, *, clear=True, contacts=None):
    contacts = contacts or pipeline.contacts()
    for _ in range(steps):
        if clear:
            states[0].clear_forces()
        pipeline.collide(states[0], contacts)
        solver.step(states[0], states[1], control, contacts, 0.005)
        states.reverse()


devices = get_cuda_test_devices()


class TestFeatherPGSSleepingProduction(unittest.TestCase):
    pass


for _name in (
    "test_anchored_articulations_sleep_and_wake",
    "test_graph_replay_sleeps_and_wakes",
    "test_frozen_patch_carry_copies_the_previous_history",
):
    add_function_test(TestFeatherPGSSleepingProduction, _name, globals()[_name], devices=devices)


if __name__ == "__main__":
    unittest.main()
