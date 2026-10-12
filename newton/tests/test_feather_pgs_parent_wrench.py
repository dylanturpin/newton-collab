# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The BODY_PARENT_F solver observable of SolverFeatherPGS."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS, SolverFeatherstone, SolverObservableFlags
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

DT = 1.0 / 120.0
FLAGS = {SolverObservableFlags.BODY_PARENT_F}


def _parent_wrench_scene(device):
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    for x in (0.0, 0.5):
        body = builder.add_body(xform=wp.transform(wp.vec3(x, 0.0, 0.1), wp.quat_identity()))
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=64)
    return model, pipeline


def _advance(pipeline, solver, states, control, steps, *, observables=None):
    contacts = pipeline.contacts()
    for _ in range(steps):
        states[0].clear_forces()
        pipeline.collide(states[0], contacts)
        solver.step(states[0], states[1], control, contacts, DT, observables=observables)
        states.reverse()


def _sleeping_wrench_solvers(model):
    solver = SolverFeatherPGS(
        model, pgs_mode="matrix_free", enable_sleeping=True, sleep_quiet_time=0.05, friction_anchor_beta=0.0
    )
    reference = SolverFeatherPGS(model, pgs_mode="matrix_free", friction_anchor_beta=0.0)
    return solver, reference


def _resting_wrench(model, pipeline, reference, control):
    observables = reference.observables(FLAGS)
    _advance(pipeline, reference, [model.state(), model.state()], control, 130, observables=observables)
    return observables.body_parent_f.numpy()


def test_sleeping_body_keeps_its_parent_wrench(test, device):
    """Publish the last awake joint wrench of a sleeping body instead of a skipped dynamics result."""
    model, pipeline = _parent_wrench_scene(device)
    solver, reference = _sleeping_wrench_solvers(model)
    observables = solver.observables(FLAGS)
    states, control = [model.state(), model.state()], model.control()
    _advance(pipeline, solver, states, control, 120, observables=observables)
    np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [0, 0])
    asleep = observables.body_parent_f.numpy().copy()
    _advance(pipeline, solver, states, control, 10, observables=observables)
    np.testing.assert_array_equal(observables.body_parent_f.numpy(), asleep)
    # The frozen value is the resting value an always-awake solver publishes.
    np.testing.assert_allclose(asleep, _resting_wrench(model, pipeline, reference, control), rtol=0.0, atol=1.0e-3)


def test_sleeping_body_keeps_its_legacy_parent_wrench(test, device):
    """Freeze the deprecated State.body_parent_f output of a sleeping body as well."""
    model, pipeline = _parent_wrench_scene(device)
    with test.assertWarns(DeprecationWarning):
        model.request_state_attributes("body_parent_f")
    solver, _reference = _sleeping_wrench_solvers(model)
    states, control = [model.state(), model.state()], model.control()
    test.assertIsNotNone(states[0].body_parent_f)
    _advance(pipeline, solver, states, control, 120)
    np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [0, 0])
    asleep = states[0].body_parent_f.numpy().copy()
    test.assertGreater(np.abs(asleep[:, 2]).min(), 1.0)
    _advance(pipeline, solver, states, control, 10)
    np.testing.assert_array_equal(states[0].body_parent_f.numpy(), asleep)
    np.testing.assert_array_equal(states[1].body_parent_f.numpy(), asleep)


def test_sleeping_parent_wrench_is_independent_of_the_output(test, device):
    """Report the frozen wrench in a newly allocated or cleared output, in eager and captured steps."""
    model, pipeline = _parent_wrench_scene(device)
    solver, reference = _sleeping_wrench_solvers(model)
    first = solver.observables(FLAGS)
    states, control = [model.state(), model.state()], model.control()
    _advance(pipeline, solver, states, control, 120, observables=first)
    np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [0, 0])
    asleep = first.body_parent_f.numpy().copy()
    np.testing.assert_allclose(asleep, _resting_wrench(model, pipeline, reference, control), rtol=0.0, atol=1.0e-3)
    test.assertGreater(np.abs(asleep[:, 2]).min(), 1.0)

    second = solver.observables(FLAGS)
    _advance(pipeline, solver, states, control, 1, observables=second)
    np.testing.assert_array_equal(second.body_parent_f.numpy(), asleep)
    first.body_parent_f.fill_(wp.spatial_vector(-1.0))
    _advance(pipeline, solver, states, control, 1, observables=first)
    np.testing.assert_array_equal(first.body_parent_f.numpy(), asleep)

    contacts = pipeline.contacts()

    def step(observables):
        states[0].clear_forces()
        pipeline.collide(states[0], contacts)
        solver.step(states[0], states[1], control, contacts, DT, observables=observables)
        pipeline.collide(states[1], contacts)
        solver.step(states[1], states[0], control, contacts, DT, observables=observables)

    step(second)
    third = solver.observables(FLAGS)
    with wp.ScopedCapture(device) as capture:
        step(third)
    for _ in range(3):
        wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [0, 0])
    np.testing.assert_array_equal(third.body_parent_f.numpy(), asleep)


def test_parent_wrench_first_requested_after_sleep(test, device):
    """Report the resting wrench when the first output is requested after the bodies fell asleep."""
    model, pipeline = _parent_wrench_scene(device)
    solver, reference = _sleeping_wrench_solvers(model)
    states, control = [model.state(), model.state()], model.control()
    _advance(pipeline, solver, states, control, 120)
    np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [0, 0])
    observables = solver.observables(FLAGS)
    _advance(pipeline, solver, states, control, 1, observables=observables)
    np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [0, 0])
    resting = _resting_wrench(model, pipeline, reference, control)
    np.testing.assert_allclose(observables.body_parent_f.numpy(), resting, rtol=0.0, atol=1.0e-3)
    # A reset wakes the bodies, which then publish freshly computed wrenches.
    solver.reset(states[0])
    _advance(pipeline, solver, states, control, 1, observables=observables)
    np.testing.assert_allclose(observables.body_parent_f.numpy(), resting, rtol=0.0, atol=1.0e-3)


def test_sleeping_parent_wrench_from_legacy_to_observable(test, device):
    """Report the same frozen wrench in an observable after sleeping with only the legacy output."""
    model, pipeline = _parent_wrench_scene(device)
    with test.assertWarns(DeprecationWarning):
        model.request_state_attributes("body_parent_f")
    solver, _reference = _sleeping_wrench_solvers(model)
    states, control = [model.state(), model.state()], model.control()
    _advance(pipeline, solver, states, control, 120)
    np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [0, 0])
    legacy = states[0].body_parent_f.numpy().copy()
    observables = solver.observables(FLAGS)
    _advance(pipeline, solver, states, control, 1, observables=observables)
    np.testing.assert_array_equal(observables.body_parent_f.numpy(), legacy)
    np.testing.assert_array_equal(states[0].body_parent_f.numpy(), legacy)


def _branched_chains(device, world_count=2):
    """Two-branch revolute trees on a fixed base plus a free box, with gravity, revolute joint forces and velocities."""
    template = newton.ModelBuilder()
    root = template.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.01))
    template.add_shape_box(root, hx=0.05, hy=0.05, hz=0.05)
    joints = [template.add_joint_fixed(parent=-1, child=root)]
    for side in (-1.0, 1.0):
        parent = root
        for level in range(3):
            child = template.add_link(mass=0.5 + 0.1 * level, inertia=wp.mat33(np.eye(3) * 0.005))
            template.add_shape_capsule(child, radius=0.02, half_height=0.08)
            axis = (0.0, 1.0, 0.0) if level % 2 == 0 else (1.0, 0.0, 0.0)
            joints.append(
                template.add_joint_revolute(
                    parent=parent,
                    child=child,
                    axis=axis,
                    parent_xform=wp.transform(wp.vec3(side * 0.1, 0.0, -0.05 - 0.15 * level), wp.quat_identity()),
                    child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.08), wp.quat_identity()),
                )
            )
            parent = child
    template.add_articulation(joints)
    free = template.add_body(xform=wp.transform(wp.vec3(1.0, 0.0, 2.0), wp.quat_identity()))
    template.add_shape_box(free, hx=0.1, hy=0.1, hz=0.1)
    builder = newton.ModelBuilder()
    for _ in range(world_count):
        builder.add_world(template)
    model = builder.finalize(device=device)
    state = model.state()
    rng = np.random.default_rng(3)
    joint_q = model.joint_q.numpy()
    joint_qd = model.joint_qd.numpy()
    revolute = model.joint_type.numpy() == int(newton.JointType.REVOLUTE)
    q_start = model.joint_q_start.numpy()
    qd_start = model.joint_qd_start.numpy()
    joint_f = np.zeros(model.joint_dof_count, dtype=np.float32)
    for joint in np.flatnonzero(revolute):
        joint_q[q_start[joint]] = rng.uniform(-0.6, 0.6)
        joint_qd[qd_start[joint]] = rng.uniform(-1.0, 1.0)
        # Free-joint forces enter Featherstone's and FeatherPGS's root wrench differently.
        joint_f[qd_start[joint]] = rng.uniform(-0.5, 0.5)
    state.joint_q.assign(joint_q)
    state.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    control = model.control()
    control.joint_f.assign(joint_f)
    return model, state, control


def _first_step_wrench(model, state, control, solver):
    observables = solver.observables(FLAGS)
    solver.step(state, model.state(), control, None, DT, observables=observables)
    return observables.body_parent_f.numpy()


def test_parent_wrench_matches_featherstone_on_every_inverse_dynamics_path(test, device):
    """Report Featherstone's joint wrench whichever stage-1 inverse-dynamics path a configuration selects."""
    model, state, control = _branched_chains(device)
    reference = _first_step_wrench(model, state, control, SolverFeatherstone(model, angular_damping=0.0))
    test.assertGreater(np.abs(reference).max(), 1.0)
    configurations = [{"pgs_mode": "split"}]
    if wp.get_device(device).is_cuda:
        configurations += [
            {"pgs_mode": "matrix_free"},
            {"pgs_mode": "matrix_free", "parallel_tree": True},
            {"pgs_mode": "matrix_free", "use_parallel_streams": False},
            {"pgs_mode": "matrix_free", "double_buffer": False},
        ]
    for options in configurations:
        with test.subTest(**options):
            solver = SolverFeatherPGS(model, angular_damping=0.0, **options)
            wrench = _first_step_wrench(model, state, control, solver)
            np.testing.assert_allclose(wrench, reference, rtol=1.0e-4, atol=1.0e-4)


class TestFeatherPGSParentWrench(unittest.TestCase):
    pass


for _name in (
    "test_sleeping_body_keeps_its_parent_wrench",
    "test_sleeping_body_keeps_its_legacy_parent_wrench",
    "test_sleeping_parent_wrench_is_independent_of_the_output",
    "test_parent_wrench_first_requested_after_sleep",
    "test_sleeping_parent_wrench_from_legacy_to_observable",
):
    add_function_test(TestFeatherPGSParentWrench, _name, globals()[_name], devices=get_cuda_test_devices())
add_function_test(
    TestFeatherPGSParentWrench,
    "test_parent_wrench_matches_featherstone_on_every_inverse_dynamics_path",
    test_parent_wrench_matches_featherstone_on_every_inverse_dynamics_path,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main()
