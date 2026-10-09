# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The BODY_PARENT_F solver observable of SolverFeatherPGS."""

import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS, SolverObservableFlags
from newton.tests.test_feather_pgs_connect import _build_carried_load, _build_four_bar, _prescribe_carrier
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

DT = 1.0 / 120.0
FLAGS = {SolverObservableFlags.BODY_PARENT_F}


def _parent_wrench_scene(device):
    """Two free boxes resting on the ground, each carrying a smaller box welded on top."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    for x in (0.0, 0.5):
        base = builder.add_link(xform=wp.transform(wp.vec3(x, 0.0, 0.1), wp.quat_identity()))
        builder.add_shape_box(base, hx=0.1, hy=0.1, hz=0.1)
        top = builder.add_link(xform=wp.transform(wp.vec3(x, 0.0, 0.25), wp.quat_identity()))
        builder.add_shape_box(top, hx=0.05, hy=0.05, hz=0.05)
        free = builder.add_joint_free(base)
        weld = builder.add_joint_fixed(
            base, top, parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.15), wp.quat_identity())
        )
        builder.add_articulation([free, weld])
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=64)
    return model, pipeline


def _assert_resting_wrench(test, model, wrench):
    """Free roots transmit nothing; each weld carries its top box's weight."""
    weight = model.body_mass.numpy() * 9.81
    np.testing.assert_allclose(wrench[0::2], 0.0, atol=1.0e-3)
    np.testing.assert_allclose(wrench[1::2, 2], weight[1::2], rtol=1.0e-4)
    np.testing.assert_allclose(wrench[1::2, [0, 1, 3, 4, 5]], 0.0, atol=1.0e-3)


def _advance(pipeline, solver, states, control, steps, *, observables=None):
    contacts = pipeline.contacts()
    for _ in range(steps):
        states[0].clear_forces()
        pipeline.collide(states[0], contacts)
        solver.step(states[0], states[1], control, contacts, DT, observables=observables)
        states.reverse()


def _sleeping_wrench_solvers(model):
    solver = SolverFeatherPGS(
        model,
        pgs_mode="matrix_free",
        enable_sleeping=True,
        sleep_quiet_time=0.05,
        friction_anchor_beta=0.0,
        dense_max_constraints=128,
    )
    reference = SolverFeatherPGS(model, pgs_mode="matrix_free", friction_anchor_beta=0.0, dense_max_constraints=128)
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
    np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [0, 0, 0, 0])
    asleep = observables.body_parent_f.numpy().copy()
    _advance(pipeline, solver, states, control, 10, observables=observables)
    np.testing.assert_array_equal(observables.body_parent_f.numpy(), asleep)
    # The frozen value is the resting value an always-awake solver publishes.
    np.testing.assert_allclose(asleep, _resting_wrench(model, pipeline, reference, control), rtol=0.0, atol=1.0e-3)
    _assert_resting_wrench(test, model, asleep)


def test_sleeping_body_keeps_its_legacy_parent_wrench(test, device):
    """Freeze the deprecated State.body_parent_f output of a sleeping body as well."""
    model, pipeline = _parent_wrench_scene(device)
    with test.assertWarns(DeprecationWarning):
        model.request_state_attributes("body_parent_f")
    solver, _reference = _sleeping_wrench_solvers(model)
    states, control = [model.state(), model.state()], model.control()
    test.assertIsNotNone(states[0].body_parent_f)
    _advance(pipeline, solver, states, control, 120)
    np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [0, 0, 0, 0])
    asleep = states[0].body_parent_f.numpy().copy()
    _assert_resting_wrench(test, model, asleep)
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
    np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [0, 0, 0, 0])
    asleep = first.body_parent_f.numpy().copy()
    np.testing.assert_allclose(asleep, _resting_wrench(model, pipeline, reference, control), rtol=0.0, atol=1.0e-3)
    _assert_resting_wrench(test, model, asleep)

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
    np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [0, 0, 0, 0])
    np.testing.assert_array_equal(third.body_parent_f.numpy(), asleep)


def test_parent_wrench_first_requested_after_sleep(test, device):
    """Report the resting wrench when the first output is requested after the bodies fell asleep."""
    model, pipeline = _parent_wrench_scene(device)
    solver, reference = _sleeping_wrench_solvers(model)
    states, control = [model.state(), model.state()], model.control()
    _advance(pipeline, solver, states, control, 120)
    np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [0, 0, 0, 0])
    observables = solver.observables(FLAGS)
    _advance(pipeline, solver, states, control, 1, observables=observables)
    np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [0, 0, 0, 0])
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
    np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [0, 0, 0, 0])
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


def _first_step(model, state, control, solver):
    observables = solver.observables(FLAGS)
    state_out = model.state()
    solver.step(state, state_out, control, None, DT, observables=observables)
    return observables.body_parent_f.numpy(), state_out


def _revolute_wrench_torques(model, state, wrench):
    """Project each revolute joint's published wrench onto its axis about its anchor."""
    body_q = state.body_q.numpy()
    body_com = model.body_com.numpy()
    joint_X_c = model.joint_X_c.numpy()
    joint_axis = model.joint_axis.numpy()
    torques = {}
    for joint in np.flatnonzero(model.joint_type.numpy() == int(newton.JointType.REVOLUTE)):
        child = model.joint_child.numpy()[joint]
        X_wc = wp.transform(*body_q[child])
        X_wj = X_wc * wp.transform(*joint_X_c[joint])
        anchor = np.array(wp.transform_get_translation(X_wj))
        axis = np.array(
            wp.quat_rotate(wp.transform_get_rotation(X_wj), wp.vec3(*joint_axis[model.joint_qd_start.numpy()[joint]]))
        )
        com = np.array(wp.transform_point(X_wc, wp.vec3(*body_com[child])))
        force, torque_com = wrench[child, :3], wrench[child, 3:]
        torques[joint] = float(np.dot(axis, torque_com + np.cross(com - anchor, force)))
    return torques


def test_parent_wrench_matches_inverse_dynamics_on_every_path(test, device):
    """Publish the joint wrench of the solved motion, whichever inverse-dynamics path a configuration selects.

    Every revolute wrench projects onto its axis as exactly the applied joint torque, and a free body in flight
    transmits nothing.
    """
    model, state, control = _branched_chains(device)
    configurations = [{"pgs_mode": "split"}]
    if wp.get_device(device).is_cuda:
        configurations += [
            {"pgs_mode": "matrix_free"},
            {"pgs_mode": "matrix_free", "parallel_tree": True},
            {"pgs_mode": "matrix_free", "use_parallel_streams": False},
            {"pgs_mode": "matrix_free", "double_buffer": False},
        ]
    joint_f = control.joint_f.numpy()
    qd_start = model.joint_qd_start.numpy()
    free_bodies = model.joint_child.numpy()[model.joint_type.numpy() == int(newton.JointType.FREE)]
    reference = None
    for options in configurations:
        with test.subTest(**options):
            solver = SolverFeatherPGS(model, angular_damping=0.0, **options)
            wrench, _ = _first_step(model, state, control, solver)
            test.assertGreater(np.abs(wrench).max(), 1.0)
            for joint, torque in _revolute_wrench_torques(model, state, wrench).items():
                test.assertAlmostEqual(torque, joint_f[qd_start[joint]], delta=1.0e-3)
            np.testing.assert_allclose(wrench[free_bodies], 0.0, atol=1.0e-3)
            if reference is None:
                reference = wrench
            np.testing.assert_allclose(wrench, reference, rtol=1.0e-4, atol=1.0e-4)


G = 9.81
CONTACT_DT = 1.0 / 480.0
M_CARRIAGE = 1.0
M_GRIP = 0.5
M_FINGER = 0.1
M_BOX = 0.5


def _gripper_scene(device, *, lift_z=0.5, lift_drive=True, flange_drive=True, load_box_x=None, pinch=False):
    """A held tray gripper: a fixed prismatic lift, a revolute flange about z and a 0.2 m square tray.

    Optionally a free box rests on the tray at ``load_box_x``, or two prismatic fingers pinch a free box below it.
    The tray's bottom face is 0.13 m below the lift's origin.
    """
    builder = newton.ModelBuilder()
    shape_cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=1.0)
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=1.0))
    carriage = builder.add_link(mass=M_CARRIAGE, inertia=wp.mat33(np.eye(3) * 0.01))
    ke, kd = (1.0e5, 1.0e3) if lift_drive else (0.0, 0.0)
    lift = builder.add_joint_prismatic(
        -1,
        carriage,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform((0.0, 0.0, lift_z), wp.quat_identity()),
        target_ke=ke,
        target_kd=kd,
    )
    grip = builder.add_link(
        mass=M_GRIP, com=(0.0, 0.0, -0.02), inertia=wp.mat33(np.diag([0.002, 0.002, 0.003])), lock_inertia=True
    )
    builder.add_shape_box(
        grip, xform=wp.transform((0.0, 0.0, -0.02), wp.quat_identity()), hx=0.1, hy=0.1, hz=0.01, cfg=shape_cfg
    )
    flange = builder.add_joint_revolute(
        carriage,
        grip,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform((0.0, 0.0, -0.1), wp.quat_identity()),
        target_ke=1.0e4 if flange_drive else 0.0,
        target_kd=1.0e2 if flange_drive else 0.0,
    )
    joints = [lift, flange]
    if pinch:
        for side in (-1.0, 1.0):
            finger = builder.add_link(mass=M_FINGER, inertia=wp.mat33(np.eye(3) * 1.0e-4))
            builder.add_shape_box(
                finger,
                xform=wp.transform((0.0, 0.0, -0.08), wp.quat_identity()),
                hx=0.01,
                hy=0.03,
                hz=0.05,
                cfg=shape_cfg,
            )
            joints.append(
                builder.add_joint_prismatic(
                    grip,
                    finger,
                    axis=(side, 0.0, 0.0),
                    parent_xform=wp.transform((side * 0.06, 0.0, -0.03), wp.quat_identity()),
                    target_pos=-0.03,
                    target_ke=2000.0,
                    target_kd=20.0,
                    limit_lower=-0.05,
                    limit_upper=0.05,
                )
            )
    builder.add_articulation(joints)
    box = None
    box_z = None
    if load_box_x is not None:
        box_x, box_z = load_box_x, lift_z - 0.07
    elif pinch:
        box_x, box_z = 0.0, lift_z - 0.21
    if box_z is not None:
        box = builder.add_body(
            xform=wp.transform((box_x, 0.0, box_z), wp.quat_identity()),
            mass=M_BOX,
            inertia=wp.mat33(np.eye(3) * M_BOX * 0.0032 / 6.0),
            lock_inertia=True,
        )
        builder.add_shape_box(box, hx=0.04, hy=0.04, hz=0.04, cfg=shape_cfg)
    model = builder.finalize(device=device)
    return model, {"grip": grip, "lift": lift, "flange": flange, "box": box}


_CONTACT_CONFIGURATIONS = {
    # The production profile: patch friction, shared anchors, spin rows and sleeping.
    "production": {
        "pgs_mode": "matrix_free",
        "pgs_iterations": 32,
        "pgs_contact_regularization": 0.01,
        "contact_shared_anchor": True,
        "contact_friction_shared_anchor": True,
        "friction_anchor_beta": 0.2,
        "contact_friction_gap_threshold": 0.001,
        "enable_joint_velocity_limits": True,
        "contact_torsion_radius": 0.01,
        "contact_torsion_device": True,
        "enable_sleeping": True,
        "dense_max_constraints": 256,
        "mf_max_constraints": 256,
    },
    "point_friction": {"pgs_mode": "matrix_free", "friction_anchor_beta": 0.0, "dense_max_constraints": 256},
    "split": {"pgs_mode": "split", "dense_max_constraints": 64},
    "propagation": {
        "pgs_mode": "matrix_free",
        "articulated_contact_response": "propagation",
        "dense_max_constraints": 256,
    },
    "velocity_pass": {"pgs_mode": "matrix_free", "pgs_velocity_iterations": 4, "dense_max_constraints": 256},
}


def _settled_wrench(model, ids, configuration, steps, *, joint_f=None, capture=False):
    """Average the published wrenches over the last 30 of ``steps`` steps."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        solver = SolverFeatherPGS(model, angular_damping=0.0, **_CONTACT_CONFIGURATIONS[configuration])
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=256, contact_matching="sticky")
    observables = solver.observables(FLAGS)
    states, control = [model.state(), model.state()], model.control()
    if joint_f is not None:
        values = np.zeros(model.joint_dof_count, dtype=np.float32)
        for joint, value in joint_f.items():
            values[model.joint_qd_start.numpy()[ids[joint]]] = value
        control.joint_f.assign(values)
    contacts = pipeline.contacts()

    def step():
        states[0].clear_forces()
        pipeline.collide(states[0], contacts)
        solver.step(states[0], states[1], control, contacts, CONTACT_DT, observables=observables)
        states.reverse()

    samples = []
    for i in range(steps):
        if capture and i == steps - 31:
            # Two steps per graph keep the state ping-pong in place across replays.
            solver.prepare_contact_torsion_capture(states[0], states[1])
            with wp.ScopedCapture(model.device) as recording:
                step()
                step()
            for _ in range(15):
                wp.capture_launch(recording.graph)
                solver.validate_contact_torsion()
                samples.append(observables.body_parent_f.numpy())
            break
        step()
        if i >= steps - 30:
            samples.append(observables.body_parent_f.numpy())
    return np.mean(samples, axis=0)


def test_parent_wrench_carries_a_load_resting_on_the_gripper(test, device):
    """A box resting off-centre on the held tray reaches the flange as weight plus its moment."""
    box_x = 0.03
    model, ids = _gripper_scene(device, load_box_x=box_x)
    for configuration in _CONTACT_CONFIGURATIONS:
        with test.subTest(configuration=configuration):
            wrench = _settled_wrench(model, ids, configuration, 480)
            expected = [0.0, 0.0, (M_GRIP + M_BOX) * G, 0.0, -box_x * M_BOX * G, 0.0]
            np.testing.assert_allclose(wrench[ids["grip"]], expected, atol=2.0e-3)
            np.testing.assert_allclose(wrench[ids["box"]], 0.0, atol=2.0e-3)


def test_parent_wrench_reports_a_commanded_push(test, device):
    """A lift force pressing the tray into the ground reaches the flange as that force plus the carriage weight."""
    push = 20.0
    model, ids = _gripper_scene(device, lift_z=0.1305, lift_drive=False)
    for configuration in _CONTACT_CONFIGURATIONS:
        with test.subTest(configuration=configuration):
            wrench = _settled_wrench(
                model, ids, configuration, 480, joint_f={"lift": -push}, capture=configuration == "production"
            )
            # The four tray corners share the load indeterminately, so only the force is fixed.
            np.testing.assert_allclose(wrench[ids["grip"], :3], [0.0, 0.0, -(push + M_CARRIAGE * G)], atol=2.0e-3)


def test_parent_wrench_carries_a_pinched_load(test, device):
    """Friction-held payload weight reaches the flange; the squeeze stays inside the gripper subtree."""
    model, ids = _gripper_scene(device, pinch=True)
    for configuration in _CONTACT_CONFIGURATIONS:
        with test.subTest(configuration=configuration):
            wrench = _settled_wrench(model, ids, configuration, 720)
            expected = [0.0, 0.0, (M_GRIP + 2.0 * M_FINGER + M_BOX) * G, 0.0, 0.0, 0.0]
            np.testing.assert_allclose(wrench[ids["grip"]], expected, atol=1.0e-2)
            np.testing.assert_allclose(wrench[ids["box"]], 0.0, atol=1.0e-2)


def test_parent_wrench_includes_spin_rows(test, device):
    """A box riding a spinning tray transmits nothing through its free joint once spin-row torques are counted."""
    model, ids = _gripper_scene(device, flange_drive=False, load_box_x=0.0)
    wrench = _settled_wrench(model, ids, "production", 240, joint_f={"flange": 0.3})
    np.testing.assert_allclose(wrench[ids["box"]], 0.0, atol=1.0e-3)
    test.assertAlmostEqual(wrench[ids["grip"], 5], 0.3, delta=1.0e-3)


def test_parent_wrench_excludes_velocity_limit_prescaling(test, device):
    """A disk spun past its velocity limit is rescaled before the dynamics; the rescale transmits no wrench."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    hub = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.01))
    hub_joint = builder.add_joint_revolute(-1, hub, axis=(0.0, 0.0, 1.0))
    disk = builder.add_link(mass=1.0, inertia=wp.mat33(np.diag([0.01, 0.01, 0.02])), lock_inertia=True)
    disk_joint = builder.add_joint_revolute(
        hub,
        disk,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform((0.3, 0.0, 0.0), wp.quat_identity()),
        velocity_limit=1.0,
    )
    builder.add_articulation([hub_joint, disk_joint])
    model = builder.finalize(device=device)
    solver = SolverFeatherPGS(model, pgs_mode="matrix_free", enable_joint_velocity_limits=True, angular_damping=0.0)
    observables = solver.observables(FLAGS)
    state_in, state_out = model.state(), model.state()
    state_in.joint_qd.assign(np.array([0.0, 10.0], dtype=np.float32))
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    solver.step(state_in, state_out, None, None, CONTACT_DT, observables=observables)
    test.assertAlmostEqual(float(state_out.joint_qd.numpy()[1]), 1.0, places=4)
    np.testing.assert_allclose(observables.body_parent_f.numpy(), 0.0, atol=1.0e-4)


def test_parent_wrench_includes_loop_closure_reactions(test, device):
    """Loop-closure forces reach both closure links: an undriven joint in a four-bar transmits no axis torque."""
    builder = _build_four_bar()
    model = builder.finalize(device=device)
    solver = SolverFeatherPGS(model, pgs_mode="matrix_free", pgs_iterations=16, pgs_beta=0.1, angular_damping=0.0)
    observables = solver.observables(FLAGS)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    targets = model.joint_target_q.numpy().copy()
    targets[0] = 0.6
    control.joint_target_q.assign(targets)
    for step in range(120):
        state_in = state_0
        solver.step(state_0, state_1, control, None, 1.0 / 240.0, observables=observables)
        state_0, state_1 = state_1, state_0
        if step % 20 == 19:
            wrench = observables.body_parent_f.numpy()
            test.assertGreater(np.abs(wrench).max(), 0.5)
            torques = _revolute_wrench_torques(model, state_in, wrench)
            # Joint 0 is the driven crank; the coupler and rocker joints are undriven.
            for joint in (1, 2):
                test.assertAlmostEqual(torques[joint], 0.0, delta=2.0e-3)


def test_parent_wrench_of_a_load_held_by_closures(test, device):
    """A FREE load hung from the world or a moving kinematic carrier by closures transmits nothing through its free joint."""
    for world_parent in (True, False):
        with test.subTest(world_parent=world_parent):
            builder, carrier, load, _joints = _build_carried_load(world_parent=world_parent)
            model = builder.finalize(device=device)
            solver = SolverFeatherPGS(model, pgs_mode="matrix_free", pgs_iterations=16, pgs_beta=0.2)
            observables = solver.observables(FLAGS)
            state_0, state_1 = model.state(), model.state()
            control = model.control()
            articulation = None
            if not world_parent:
                articulation = wp.array(
                    [int(solver.body_to_articulation.numpy()[carrier])], dtype=wp.int32, device=device
                )
            dt = 1.0 / 240.0
            for i in range(120):
                if articulation is not None:
                    _prescribe_carrier(
                        model,
                        solver,
                        state_0,
                        articulation,
                        i * dt,
                        np.array([0.3, 0.0, 0.1]),
                        np.array([0.0, 0.0, 1.5]),
                    )
                state_0.clear_forces()
                solver.step(state_0, state_1, control, None, dt, observables=observables)
                state_0, state_1 = state_1, state_0
            np.testing.assert_allclose(observables.body_parent_f.numpy()[load], 0.0, atol=2.0e-3)


def test_parent_wrench_warns_without_preeliminated_closures(test, device):
    """Pre-eliminated loop closures have no stored impulses, so requesting the wrench warns once."""
    model = _build_four_bar().finalize(device=device)
    solver = SolverFeatherPGS(model, pgs_mode="matrix_free", enable_bilateral_preelimination=True)
    observables = solver.observables(FLAGS)
    state_0, state_1 = model.state(), model.state()
    with test.assertWarnsRegex(UserWarning, "pre-eliminated loop-closure"):
        solver.step(state_0, state_1, None, None, 1.0 / 240.0, observables=observables)


def test_unattached_body_reports_zero(test, device):
    """A body outside every articulation overwrites its row with zero."""
    builder = newton.ModelBuilder()
    root = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.1))
    builder.add_articulation([builder.add_joint_revolute(-1, root, axis=(0.0, 1.0, 0.0))])
    builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    model = builder.finalize(device=device)
    solver = SolverFeatherPGS(model, angular_damping=0.0)
    observables = solver.observables(FLAGS)
    observables.body_parent_f.fill_(wp.spatial_vector(123.0, 123.0, 123.0, 123.0, 123.0, 123.0))
    solver.step(model.state(), model.state(), None, None, DT, observables=observables)
    np.testing.assert_array_equal(observables.body_parent_f.numpy()[1], 0.0)


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
    "test_unattached_body_reports_zero",
    test_unattached_body_reports_zero,
    devices=get_test_devices(),
)
add_function_test(
    TestFeatherPGSParentWrench,
    "test_parent_wrench_matches_inverse_dynamics_on_every_path",
    test_parent_wrench_matches_inverse_dynamics_on_every_path,
    devices=get_test_devices(),
)
for _name in (
    "test_parent_wrench_carries_a_load_resting_on_the_gripper",
    "test_parent_wrench_reports_a_commanded_push",
    "test_parent_wrench_carries_a_pinched_load",
    "test_parent_wrench_includes_spin_rows",
    "test_parent_wrench_excludes_velocity_limit_prescaling",
    "test_parent_wrench_includes_loop_closure_reactions",
    "test_parent_wrench_of_a_load_held_by_closures",
    "test_parent_wrench_warns_without_preeliminated_closures",
):
    add_function_test(TestFeatherPGSParentWrench, _name, globals()[_name], devices=get_cuda_test_devices())


if __name__ == "__main__":
    unittest.main()
