# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverSteel, steel
from newton.tests.steel.test_differentiable import _build_chain
from newton.tests.unittest_utils import add_function_test, get_test_devices

_DT = 1.0 / 240.0
_OPTIONS = {
    "friction_anchor_beta": 0.0,
    "contact_friction_gap_max": -math.inf,
    "enable_restitution": False,
    "iterations": 16,
    "articulation_constraint_max": 48,
    "deterministic": True,
}


def _build_worlds(device, worlds=3):
    """``worlds`` copies of a hinged pair of boxes resting on a shared ground plane with friction."""
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.5, restitution=0.0)
    robot = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    base = robot.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.098), wp.quat_identity()))
    robot.add_shape_box(base, hx=0.2, hy=0.1, hz=0.1, cfg=cfg)
    link = robot.add_link(xform=wp.transform(wp.vec3(0.4, 0.0, 0.098), wp.quat_identity()))
    robot.add_shape_box(link, hx=0.2, hy=0.1, hz=0.1, cfg=cfg)
    joints = [
        robot.add_joint_free(parent=-1, child=base),
        robot.add_joint_revolute(
            base,
            link,
            parent_xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
            axis=(0.0, 1.0, 0.0),
            target_ke=40.0,
            target_kd=2.0,
            armature=0.01,
        ),
    ]
    robot.add_articulation(joints)
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    builder.replicate(robot, worlds, spacing=(2.0, 0.0, 0.0))
    builder.add_ground_plane(cfg=cfg)
    return builder.finalize(device=device, requires_grad=True)


def _initial_velocity(model, seed=0):
    rng = np.random.default_rng(seed)
    return (0.3 * rng.normal(size=model.joint_dof_count)).astype(np.float32)


def _taped_rollout(model, options, steps, *, joint_qd=None, target=0.3):
    """Step from rest with ``joint_qd``, colliding every step outside the tape; return solver, tape, states, control."""
    solver = SolverSteel(model, differentiable=True, differentiable_options=options, **_OPTIONS)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=128, deterministic=True)
    contacts = pipeline.contacts()
    states = [model.state(requires_grad=True) for _ in range(steps + 1)]
    states[0].joint_qd.assign(_initial_velocity(model) if joint_qd is None else joint_qd)
    newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
    control = model.control()
    control.joint_target_q.fill_(target)
    tape = wp.Tape()
    for k in range(steps):
        pipeline.collide(states[k], contacts)
        with tape:
            solver.step(states[k], states[k + 1], control, contacts, _DT)
    return solver, tape, states, control


def _seed(model, seed=1):
    rng = np.random.default_rng(seed)
    return (
        rng.normal(size=model.joint_coord_count).astype(np.float32),
        rng.normal(size=model.joint_dof_count).astype(np.float32),
    )


def _world_slices(model):
    """Per-world joint coordinate and DOF index arrays."""
    worlds = model.world_count
    joint_world = model.joint_world.numpy()
    q_start, qd_start = model.joint_q_start.numpy(), model.joint_qd_start.numpy()
    coords, dofs = [[] for _ in range(worlds)], [[] for _ in range(worlds)]
    for joint, world in enumerate(joint_world):
        coords[world] += list(range(q_start[joint], q_start[joint + 1]))
        dofs[world] += list(range(qd_start[joint], qd_start[joint + 1]))
    return [np.array(c) for c in coords], [np.array(d) for d in dofs]


def test_options_are_validated(test, device):
    """Invalid backward options and options without differentiable=True raise."""
    for kwargs in (
        {"pgs_adjoint": "exact"},
        {"truncated_sweeps": 0},
        {"implicit_regularization": -1.0},
        {"implicit_regularization": math.nan},
        {"zero_nonfinite_adjoints": True},
    ):
        with test.subTest(**kwargs), test.assertRaises(ValueError):
            steel.DifferentiableOptions(**kwargs)
    model = _build_chain(device)
    with test.assertRaisesRegex(ValueError, "differentiable_options requires differentiable=True"):
        SolverSteel(_build_chain(device, requires_grad=False), differentiable_options=steel.DifferentiableOptions())
    solver = SolverSteel(model, differentiable=True)
    state = model.state(requires_grad=True)
    with test.assertRaisesRegex(ValueError, "world_diagnostics=True"):
        solver.differentiable_diagnostics(state)


def test_world_diagnostics_measure_adjoint_norms(test, device):
    """The diagnostics hold each world's adjoint norm entering and leaving every step, and no flags."""
    model = _build_worlds(device)
    options = steel.DifferentiableOptions(world_diagnostics=True)
    steps = 4
    solver, tape, states, _control = _taped_rollout(model, options, steps)
    seed_q, seed_qd = _seed(model)
    tape.backward(
        grads={
            states[-1].joint_q: wp.array(seed_q, dtype=float, device=device),
            states[-1].joint_qd: wp.array(seed_qd, dtype=float, device=device),
        }
    )
    coords, dofs = _world_slices(model)
    last = solver.differentiable_diagnostics(states[-1])
    first = solver.differentiable_diagnostics(states[1])
    q_grad, qd_grad = states[0].joint_q.grad.numpy(), states[0].joint_qd.grad.numpy()
    body_f_grad = states[0].body_f.grad.numpy()
    body_world = model.body_world.numpy()
    for world in range(model.world_count):
        with test.subTest(world=world):
            expected_out = math.sqrt(float(np.sum(seed_q[coords[world]] ** 2) + np.sum(seed_qd[dofs[world]] ** 2)))
            test.assertAlmostEqual(float(last.adjoint_out.numpy()[world]), expected_out, delta=1.0e-5 * expected_out)
            expected_in = math.sqrt(
                float(
                    np.sum(q_grad[coords[world]].astype(np.float64) ** 2)
                    + np.sum(qd_grad[dofs[world]].astype(np.float64) ** 2)
                    + np.sum(body_f_grad[body_world == world].astype(np.float64) ** 2)
                )
            )
            test.assertGreater(expected_in, 0.0)
            test.assertAlmostEqual(float(first.adjoint_in.numpy()[world]), expected_in, delta=1.0e-5 * expected_in)
            for k in range(1, steps + 1):
                test.assertEqual(int(solver.differentiable_diagnostics(states[k]).flags.numpy()[world]), 0)


def test_nonfinite_world_is_flagged_and_contained(test, device):
    """A non-finite adjoint in one world is flagged; with zeroing on, the other worlds' gradients are unchanged."""
    model = _build_worlds(device)
    steps = 4
    coords, dofs = _world_slices(model)
    seed_q, seed_qd = _seed(model)
    bad = 1

    def gradients(options, poison):
        solver, tape, states, control = _taped_rollout(model, options, steps)
        q, qd = seed_q.copy(), seed_qd.copy()
        q[coords[bad]] = math.nan if poison else 0.0
        qd[dofs[bad]] = 0.0
        tape.backward(
            grads={
                states[-1].joint_q: wp.array(q, dtype=float, device=device),
                states[-1].joint_qd: wp.array(qd, dtype=float, device=device),
            }
        )
        result = {
            "joint_q": states[0].joint_q.grad.numpy().copy(),
            "joint_qd": states[0].joint_qd.grad.numpy().copy(),
            "joint_target_q": control.joint_target_q.grad.numpy().copy(),
        }
        return result, solver, states

    reference, _, _ = gradients(steel.DifferentiableOptions(world_diagnostics=True), poison=False)
    flagged, solver, states = gradients(steel.DifferentiableOptions(world_diagnostics=True), poison=True)
    test.assertTrue(np.isnan(flagged["joint_qd"][dofs[bad]]).any())
    for k in range(1, steps + 1):
        flags = solver.differentiable_diagnostics(states[k]).flags.numpy()
        with test.subTest(step=k):
            test.assertEqual(flags[bad], steel.DifferentiableDiagnostics.NONFINITE_ADJOINT)
            test.assertEqual(int(np.delete(flags, bad).max()), 0)

    contained, solver, states = gradients(
        steel.DifferentiableOptions(world_diagnostics=True, zero_nonfinite_adjoints=True), poison=True
    )
    for key, value in contained.items():
        with test.subTest(input=key):
            test.assertTrue(np.isfinite(value).all())
            np.testing.assert_array_equal(value, reference[key])
    flags = [int(solver.differentiable_diagnostics(states[k]).flags.numpy()[bad]) for k in range(1, steps + 1)]
    # Only the last step sees the poisoned adjoint; it zeroes it before the earlier steps run.
    test.assertEqual(flags, [0] * (steps - 1) + [steel.DifferentiableDiagnostics.NONFINITE_ADJOINT])


def test_nonfinite_state_is_flagged_per_world(test, device):
    """A world whose state is not finite is flagged and leaves the other worlds' forward unchanged."""
    model = _build_worlds(device)
    _coords, dofs = _world_slices(model)
    options = steel.DifferentiableOptions(world_diagnostics=True)
    velocity = _initial_velocity(model)
    _, _, reference, _ = _taped_rollout(model, options, 3, joint_qd=velocity)
    velocity[dofs[1][0]] = math.nan
    solver, _, states, _ = _taped_rollout(model, options, 3, joint_qd=velocity)
    flags = solver.differentiable_diagnostics(states[-1]).flags.numpy()
    np.testing.assert_array_equal(flags, [0, steel.DifferentiableDiagnostics.NONFINITE_STATE, 0])
    for world in (0, 2):
        with test.subTest(world=world):
            np.testing.assert_array_equal(
                states[-1].joint_qd.numpy()[dofs[world]], reference[-1].joint_qd.numpy()[dofs[world]]
            )


def test_recomputed_steps_match_one_tape(test, device):
    """Replaying each step on its own tape from saved states reproduces the one-tape gradient bitwise.

    This is the segment-recompute pattern: keep only the states of an untaped forward, then re-run and
    differentiate one step at a time from the last.
    """
    model = _build_worlds(device, worlds=2)
    steps = 5
    _, tape, states, control = _taped_rollout(model, steel.DifferentiableOptions(), steps)
    seed_q, seed_qd = _seed(model)
    tape.backward(
        grads={
            states[-1].joint_q: wp.array(seed_q, dtype=float, device=device),
            states[-1].joint_qd: wp.array(seed_qd, dtype=float, device=device),
        }
    )
    expected = (states[0].joint_q.grad.numpy(), states[0].joint_qd.grad.numpy(), control.joint_target_q.grad.numpy())

    solver = SolverSteel(model, differentiable=True, **_OPTIONS)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=128, deterministic=True)
    contacts = pipeline.contacts()
    saved = [(states[k].joint_q.numpy(), states[k].joint_qd.numpy()) for k in range(steps + 1)]
    replay_control = model.control()
    replay_control.joint_target_q.fill_(0.3)
    grad_q, grad_qd = seed_q, seed_qd
    for k in reversed(range(steps)):
        state_in, state_out = model.state(requires_grad=True), model.state(requires_grad=True)
        state_in.joint_q.assign(saved[k][0])
        state_in.joint_qd.assign(saved[k][1])
        newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
        pipeline.collide(state_in, contacts)
        step_tape = wp.Tape()
        with step_tape:
            solver.step(state_in, state_out, replay_control, contacts, _DT)
        np.testing.assert_array_equal(state_out.joint_q.numpy(), saved[k + 1][0])
        np.testing.assert_array_equal(state_out.joint_qd.numpy(), saved[k + 1][1])
        step_tape.backward(
            grads={
                state_out.joint_q: wp.array(grad_q, dtype=float, device=device),
                state_out.joint_qd: wp.array(grad_qd, dtype=float, device=device),
            }
        )
        grad_q, grad_qd = state_in.joint_q.grad.numpy(), state_in.joint_qd.grad.numpy()
    np.testing.assert_array_equal(grad_q, expected[0])
    np.testing.assert_array_equal(grad_qd, expected[1])
    np.testing.assert_array_equal(replay_control.joint_target_q.grad.numpy(), expected[2])


def test_plain_atomics_adjoint_matches_ordered(test, device):
    """deterministic_adjoint=False changes only the backward's reduction order."""
    model = _build_worlds(device, worlds=2)
    gradients = []
    for ordered in (True, False):
        _, tape, states, control = _taped_rollout(model, steel.DifferentiableOptions(deterministic_adjoint=ordered), 4)
        seed_q, seed_qd = _seed(model)
        tape.backward(
            grads={
                states[-1].joint_q: wp.array(seed_q, dtype=float, device=device),
                states[-1].joint_qd: wp.array(seed_qd, dtype=float, device=device),
            }
        )
        gradients.append(np.concatenate([states[0].joint_qd.grad.numpy(), control.joint_target_q.grad.numpy()]))
    scale = float(np.linalg.norm(gradients[0]))
    test.assertLess(float(np.linalg.norm(gradients[1] - gradients[0])), 1.0e-4 * scale)


def test_world_diagnostics_in_captured_graph(test, device):
    """A captured forward and backward with diagnostics and zeroing replays the eager diagnostics."""
    model = _build_worlds(device, worlds=2)
    options = steel.DifferentiableOptions(world_diagnostics=True, zero_nonfinite_adjoints=True)
    solver = SolverSteel(model, differentiable=True, differentiable_options=options, **_OPTIONS)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=128, deterministic=True)
    contacts = pipeline.contacts()
    steps = 3
    states = [model.state(requires_grad=True) for _ in range(steps + 1)]
    start_q, start_qd = model.state().joint_q, wp.array(_initial_velocity(model), dtype=float, device=device)
    control = model.control()
    control.joint_target_q.fill_(0.3)
    seed_q, seed_qd = (wp.array(seed, dtype=float, device=device) for seed in _seed(model))
    tape = wp.Tape()

    def run():
        wp.copy(states[0].joint_q, start_q)
        wp.copy(states[0].joint_qd, start_qd)
        newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
        tape.reset()
        for k in range(steps):
            pipeline.collide(states[k], contacts)
            with tape:
                solver.step(states[k], states[k + 1], control, contacts, _DT)
        tape.backward(grads={states[-1].joint_q: seed_q, states[-1].joint_qd: seed_qd})

    def snapshot():
        diagnostics = [solver.differentiable_diagnostics(states[k]) for k in range(1, steps + 1)]
        return [np.concatenate([d.adjoint_in.numpy(), d.adjoint_out.numpy(), d.flags.numpy()]) for d in diagnostics]

    run()
    eager = snapshot()
    with wp.ScopedCapture(device=device) as capture:
        run()
    for k in range(1, steps + 1):
        diagnostics = solver.differentiable_diagnostics(states[k])
        for array in (diagnostics.adjoint_in, diagnostics.adjoint_out, diagnostics.flags):
            array.fill_(-1)
    wp.capture_launch(capture.graph)
    for k, (expected, actual) in enumerate(zip(eager, snapshot(), strict=True)):
        with test.subTest(step=k):
            np.testing.assert_allclose(actual, expected, rtol=1.0e-5, atol=0.0)


class TestSteelDifferentiableBackward(unittest.TestCase):
    pass


for _device in get_test_devices():
    for _test in (
        test_options_are_validated,
        test_world_diagnostics_measure_adjoint_norms,
        test_nonfinite_world_is_flagged_and_contained,
        test_nonfinite_state_is_flagged_per_world,
        test_recomputed_steps_match_one_tape,
        test_plain_atomics_adjoint_matches_ordered,
    ):
        add_function_test(TestSteelDifferentiableBackward, _test.__name__, _test, devices=[_device])
    if _device.is_cuda:
        add_function_test(
            TestSteelDifferentiableBackward,
            "test_world_diagnostics_in_captured_graph",
            test_world_diagnostics_in_captured_graph,
            devices=[_device],
        )


if __name__ == "__main__":
    unittest.main()
