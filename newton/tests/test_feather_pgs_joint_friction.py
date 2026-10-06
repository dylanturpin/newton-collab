# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Dry joint friction (``Model.joint_friction``) in SolverFeatherPGS."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS

DT = 0.01
INERTIA = 0.5
FRICTION = 1.0


@unittest.skipUnless(wp.is_cuda_available(), "matrix-free joint friction requires CUDA")
class TestFeatherPGSJointFriction(unittest.TestCase):
    def test_sticks_below_breakaway_in_both_directions(self):
        """Hold hinge and slider DOFs at rest while the applied effort is below the friction."""
        for joint in ("revolute", "prismatic"):
            model = _single_dof_model(joint)
            for effort in (0.6 * FRICTION, -0.6 * FRICTION, 0.999 * FRICTION):
                with self.subTest(joint=joint, effort=effort):
                    qd = _run(model, effort=effort, steps=50)
                    np.testing.assert_allclose(qd, 0.0, atol=1.0e-6)

    def test_breakaway_accelerates_by_net_effort(self):
        """Accelerate at (effort - friction) / inertia once the effort exceeds the friction."""
        for joint, inertia in (("revolute", INERTIA), ("prismatic", 1.0)):
            model = _single_dof_model(joint)
            for effort in (2.5, -2.5):
                with self.subTest(joint=joint, effort=effort):
                    qd = _run(model, effort=effort, steps=20)
                    expected = np.sign(effort) * (abs(effort) - FRICTION) / inertia * DT * np.arange(1, 21)
                    np.testing.assert_allclose(qd, expected, rtol=1.0e-5, atol=1.0e-6)

    def test_sliding_decelerates_and_stops_without_chatter(self):
        """Decelerate at friction / inertia, stop at the predicted step and stay at rest."""
        model = _single_dof_model("revolute")
        for qd0 in (1.0, -1.0):
            with self.subTest(qd0=qd0):
                qd = _run(model, qd0=qd0, steps=80)
                stop_step = round(abs(qd0) * INERTIA / FRICTION / DT)
                expected = np.sign(qd0) * np.maximum(abs(qd0) - FRICTION / INERTIA * DT * np.arange(1, 81), 0.0)
                np.testing.assert_allclose(qd, expected, atol=1.0e-5)
                np.testing.assert_allclose(qd[stop_step:], 0.0, atol=1.0e-6)
                moving = qd[np.abs(qd) > 1.0e-6]
                self.assertTrue(np.all(np.sign(moving) == np.sign(qd0)))

    def test_substeps_preserve_physical_result(self):
        """Scale the impulse bound with dt so substepping keeps the same physical trajectory."""
        model = _single_dof_model("revolute")
        for substeps in (1, 4):
            with self.subTest(substeps=substeps):
                dt = DT / substeps
                breakaway = _run(model, effort=2.5, steps=20 * substeps, dt=dt)
                self.assertAlmostEqual(float(breakaway[-1]), 1.5 / INERTIA * 0.2, delta=1.0e-5)
                slide = _run(model, qd0=1.0, steps=60 * substeps, dt=dt)
                stop = int(np.argmax(np.abs(slide) < 1.0e-6)) + 1
                self.assertAlmostEqual(stop * dt, 0.5, delta=0.5 * dt)

    def test_d6_axes_are_independent_boxes(self):
        """Apply a separate box to each D6 axis, as MuJoCo does per DOF."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * INERTIA))
        cfg = newton.ModelBuilder.JointDofConfig
        joint = builder.add_joint_d6(
            -1,
            body,
            linear_axes=[cfg(axis=(1.0, 0.0, 0.0), friction=FRICTION, armature=0.0)],
            angular_axes=[cfg(axis=(0.0, 0.0, 1.0), friction=2.0 * FRICTION, armature=0.0)],
        )
        builder.add_articulation([joint])
        model = builder.finalize()
        qd = _run(model, effort=(1.5, 1.5), steps=10, dofs=2)
        np.testing.assert_allclose(qd[-1], [0.5 * DT * 10, 0.0], atol=1.0e-5)

    def test_locked_child_moves_with_composite_inertia(self):
        """Lock a stiff-friction child joint so the chain accelerates with its composite inertia."""
        length, mass = 0.4, 2.0
        model = _two_link_model(length=length, mass=mass, friction=(FRICTION, 50.0))
        qd = _run(model, effort=(3.0, 0.0), steps=10, dofs=2)
        composite = 2.0 * INERTIA + mass * length**2
        np.testing.assert_allclose(qd[:, 0], (3.0 - FRICTION) / composite * DT * np.arange(1, 11), rtol=1.0e-4)
        np.testing.assert_allclose(qd[:, 1], 0.0, atol=1.0e-5)

    def test_drive_settles_inside_friction_deadband(self):
        """Stop a PD-driven hinge where the drive effort falls inside the friction box."""
        ke, kd, target = 10.0, 2.0, 1.0
        for drive_mode in ("augmented", "physx_pgs"):
            with self.subTest(drive_mode=drive_mode):
                model = _single_dof_model("revolute", ke=ke, kd=kd, gravity=0.0)
                solver = _solver(model, drive_mode=drive_mode, pgs_iterations=32)
                states, control = [model.state(), model.state()], model.control()
                control.joint_target_q.assign([target])
                q, qd = _advance(solver, states, control, 400)
                self.assertLess(abs(float(qd[-1])), 1.0e-5)
                self.assertLessEqual(abs(float(q[-1]) - target), FRICTION / ke + 1.0e-3)
                self.assertGreater(abs(float(q[-1]) - target), 1.0e-2)
                reference = _solver(model, drive_mode=drive_mode, pgs_iterations=32, enable_joint_friction=False)
                states = [model.state(), model.state()]
                q_free, _ = _advance(reference, states, control, 400)
                self.assertAlmostEqual(float(q_free[-1]), target, delta=1.0e-3)

    def test_slider_stops_at_position_limit(self):
        """Combine friction deceleration with a position limit without violating the limit."""
        model = _single_dof_model("prismatic", limit=(-1.0, 0.1))
        solver = _solver(model, enable_joint_limits=True)
        states, control = [model.state(), model.state()], model.control()
        states[0].joint_qd.assign([2.0])
        q, qd = _advance(solver, states, control, 60)
        self.assertLess(float(np.max(q)), 0.1 + 1.0e-3)
        np.testing.assert_allclose(qd[-10:], 0.0, atol=1.0e-5)

    def test_velocity_limit_keeps_last_word(self):
        """Saturate breakaway acceleration at the joint velocity limit."""
        model = _single_dof_model("revolute", velocity_limit=1.0)
        for schedule in ("interleaved", "physx_grasp"):
            with self.subTest(schedule=schedule):
                solver = _solver(model, enable_joint_velocity_limits=True, pgs_schedule=schedule)
                states, control = [model.state(), model.state()], model.control()
                control.joint_f.assign([10.0])
                _, qd = _advance(solver, states, control, 20)
                self.assertLessEqual(float(np.max(qd)), 1.0 + 1.0e-5)
                self.assertAlmostEqual(float(qd[-1]), 1.0, delta=1.0e-5)

    def test_friction_and_ground_contact(self):
        """Hold a vertical slider against gravity, or let it fall onto the ground and rest."""
        mass, half = 1.0, 0.1
        for friction, falls in ((15.0, False), (5.0, True)):
            with self.subTest(friction=friction):
                model = _vertical_slider_model(mass=mass, half=half, friction=friction)
                pipeline = newton.CollisionPipeline(model, rigid_contact_max=16)
                contacts = pipeline.contacts()
                solver = _solver(model, friction_anchor_beta=0.0, pgs_iterations=24)
                states, control = [model.state(), model.state()], model.control()
                newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
                for _ in range(150):
                    pipeline.collide(states[0], contacts)
                    solver.step(states[0], states[1], control, contacts, DT)
                    states.reverse()
                z = float(states[0].joint_q.numpy()[0])
                if falls:
                    self.assertAlmostEqual(z, -0.4, delta=5.0e-3)
                else:
                    self.assertAlmostEqual(z, 0.0, delta=1.0e-5)
                self.assertLess(abs(float(states[0].joint_qd.numpy()[0])), 1.0e-3)

    def test_heterogeneous_worlds_match_isolated_worlds(self):
        """Solve per-world friction rows in worlds with different DOF counts and coefficients."""
        single = _single_dof_model("revolute")
        chain = _two_link_model(length=0.4, mass=2.0, friction=(FRICTION, 0.2))
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        builder.add_world(_single_dof_builder("revolute"))
        builder.add_world(_two_link_builder(length=0.4, mass=2.0, friction=(FRICTION, 0.2)))
        mixed = builder.finalize()
        effort = (2.5, 3.0, 0.1)
        qd_mixed = _run(mixed, effort=effort, steps=20, dofs=3)
        np.testing.assert_allclose(qd_mixed[:, 0], _run(single, effort=effort[0], steps=20), atol=1.0e-6)
        np.testing.assert_allclose(qd_mixed[:, 1:], _run(chain, effort=effort[1:], steps=20, dofs=2), atol=1.0e-6)

    def test_zero_friction_matches_disabled_bitwise(self):
        """Keep zero-coefficient friction bit-identical to the disabled solver on the same route."""
        model = _two_link_model(length=0.4, mass=2.0, friction=(0.0, 0.0), ke=20.0, kd=1.0)
        trajectories = []
        for enabled in (False, True):
            solver = _solver(model, enable_joint_friction=enabled, enable_joint_limits=True)
            states, control = [model.state(), model.state()], model.control()
            states[0].joint_qd.assign([1.0, -2.0])
            control.joint_target_q.assign([0.3, -0.2])
            trajectories.append(_advance(solver, states, control, 30))
        np.testing.assert_array_equal(trajectories[0][0], trajectories[1][0])
        np.testing.assert_array_equal(trajectories[0][1], trajectories[1][1])

    def test_disabled_ignores_model_friction(self):
        """Leave Model.joint_friction unused when the option is off."""
        model = _single_dof_model("revolute")
        qd = _run(model, effort=0.5, steps=10, enable_joint_friction=False)
        np.testing.assert_allclose(qd[-1], 0.5 / INERTIA * DT * 10, rtol=1.0e-5)

    def test_runtime_edits_take_effect(self):
        """Apply in-place edits and array replacement after notification."""
        model = _single_dof_model("revolute", friction=0.0)
        solver = _solver(model)
        states, control = [model.state(), model.state()], model.control()
        control.joint_f.assign([0.5])
        _, qd = _advance(solver, states, control, 10)
        self.assertAlmostEqual(float(qd[-1]), 0.5 / INERTIA * 0.1, delta=1.0e-5)

        model.joint_friction.assign([FRICTION])
        solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
        _, qd = _advance(solver, states, control, 20)
        np.testing.assert_allclose(qd[-5:], 0.0, atol=1.0e-6)

        control.joint_f.assign([1.5])
        model.joint_friction = wp.array([2.0], dtype=wp.float32, device=model.device)
        solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
        _, qd = _advance(solver, states, control, 5)
        np.testing.assert_allclose(qd, 0.0, atol=1.0e-6)

    def test_graph_replay_matches_eager_and_reads_edits(self):
        """Replay captured steps bit-identically and read in-place coefficient edits."""
        model = _single_dof_model("revolute")
        eager = _run(model, effort=2.5, steps=16)

        solver = _solver(model)
        states, control = [model.state(), model.state()], model.control()
        # Friction holds the hinge exactly at rest while warming up kernels outside capture.
        _advance(solver, states, control, 2)
        control.joint_f.assign([2.5])
        with wp.ScopedCapture(device=model.device) as capture:
            solver.seed_double_buffer_events()
            for _ in range(2):
                solver.step(states[0], states[1], control, None, DT)
                solver.step(states[1], states[0], control, None, DT)
        replayed = []
        for _ in range(4):
            wp.capture_launch(capture.graph)
            replayed.append(states[0].joint_qd.numpy()[0])
        np.testing.assert_array_equal(np.array(replayed, dtype=np.float32), eager[3::4])

        model.joint_friction.assign([3.0])
        before = float(states[0].joint_qd.numpy()[0])
        wp.capture_launch(capture.graph)
        self.assertAlmostEqual(float(states[0].joint_qd.numpy()[0]), before - 0.5 / INERTIA * 4 * DT, delta=1.0e-5)

    def test_reset_starts_from_cold_rows(self):
        """Reproduce a fresh trajectory after a full or masked reset and state rewrite."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        for _ in range(2):
            builder.add_world(_single_dof_builder("revolute"))
        model = builder.finalize()
        fresh = _run(model, qd0=(1.0, -1.0), effort=(0.2, -0.2), steps=20, dofs=2)
        for mask in (None, [True, True]):
            with self.subTest(mask=mask):
                solver = _solver(model)
                states, control = [model.state(), model.state()], model.control()
                control.joint_f.assign([0.2, -0.2])
                states[0].joint_qd.assign([0.3, 0.7])
                _advance(solver, states, control, 7)
                states[0].joint_q.zero_()
                states[0].joint_qd.assign([1.0, -1.0])
                world_mask = None if mask is None else wp.array(mask, dtype=wp.bool, device=model.device)
                solver.reset(states[0], world_mask=world_mask)
                _, qd = _advance(solver, states, control, 20)
                np.testing.assert_array_equal(qd, fresh)

    def test_dense_row_overflow_is_reported(self):
        """Flag worlds whose friction rows exceed the dense row capacity."""
        model = _two_link_model(length=0.4, mass=2.0, friction=(FRICTION, FRICTION))
        solver = _solver(model, dense_max_constraints=1, warn_constraint_overflow=False)
        states, control = [model.state(), model.state()], model.control()
        _advance(solver, states, control, 1)
        self.assertTrue(bool(solver.constraint_overflow.numpy()[0]))
        with self.assertRaises(RuntimeError):
            solver.check_constraint_capacity()

    def test_sleeping_island_wakes_on_friction_edit(self):
        """Sleep a friction-held pendulum and wake it when its friction is lowered."""
        model = _pendulum_model(friction=5.0)
        solver = _solver(model, enable_sleeping=True, sleep_quiet_time=0.05)
        states, control = [model.state(), model.state()], model.control()
        newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
        _advance(solver, states, control, 30)
        np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [0, 0])
        model.joint_friction.assign([0.0])
        solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
        _, qd = _advance(solver, states, control, 3)
        np.testing.assert_array_equal(solver.sleeping.body_awake.numpy(), [1, 1])
        self.assertGreater(abs(float(qd[-1])), 1.0e-2)

    def test_schedules_routes_and_velocity_pass_agree(self):
        """Give the same analytic result on every supported schedule and contact route."""
        model = _single_dof_model("revolute")
        expected = (2.5 - FRICTION) / INERTIA * DT * 10
        options = (
            {"pgs_schedule": "contact_then_internal"},
            {"pgs_schedule": "physx_grasp"},
            {"pgs_velocity_iterations": 2},
            {"pgs_warmstart": True},
            {"drive_mode": "physx_pgs"},
            {"articulated_contact_response": "propagation"},
            {"articulated_contact_response": "propagation-colored"},
            {"articulated_contact_response": "propagation-fused"},
        )
        for kwargs in options:
            with self.subTest(**kwargs):
                route = _solver(model, **kwargs)
                if kwargs.get("articulated_contact_response") == "propagation-fused":
                    self.assertIsNotNone(route._propagation_full_fused_size)
                if kwargs.get("articulated_contact_response") == "propagation-colored":
                    self.assertTrue(route._propagation_colored)
                stick = _run(model, effort=0.5, steps=10, **kwargs)
                np.testing.assert_allclose(stick, 0.0, atol=1.0e-6)
                slide = _run(model, effort=2.5, steps=10, **kwargs)
                self.assertAlmostEqual(float(slide[-1]), expected, delta=1.0e-5)

    def test_specialized_routes_are_not_selected(self):
        """Keep enabled friction on the generic matrix-free route instead of row-limited fast paths."""
        model = _arm_and_box_model()
        disabled = _solver(model, enable_joint_friction=False)
        enabled = _solver(model)
        self.assertTrue(disabled._local_internal_fast_path)
        for name in (
            "_local_internal_fast_path",
            "_paired_factor_coordinates",
            "_fused_diagonal_joint_limits",
            "_sparse_diagonal_contact_solve",
        ):
            self.assertFalse(getattr(enabled, name), name)
        self.assertIsNone(enabled._sparse_mass_matrix_size)

    def test_rejects_unsupported_configurations(self):
        """Raise for joint types, values and solver options the friction rows do not support."""
        with self.assertRaisesRegex(NotImplementedError, "matrix_free"):
            SolverFeatherPGS(_single_dof_model("revolute"), enable_joint_friction=True)
        with self.assertRaisesRegex(NotImplementedError, "pre-elimination"):
            _solver(_single_dof_model("revolute"), enable_bilateral_preelimination=True)
        for value in (-1.0, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "finite and non-negative"):
                _solver(_single_dof_model("revolute", friction=value))
        for joint in ("ball", "free"):
            with self.subTest(joint=joint), self.assertRaisesRegex(ValueError, "joint_friction is nonzero"):
                _solver(_unsupported_joint_model(joint, friction=FRICTION))

        model = _unsupported_joint_model("ball", friction=0.0)
        solver = _solver(model)
        model.joint_friction.fill_(FRICTION)
        with self.assertRaisesRegex(ValueError, "BALL"):
            solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)

    def test_rejects_differentiable_models(self):
        """Raise for gradient-tracking models instead of silently differentiating through the box."""
        builder = _single_dof_builder("revolute")
        model = builder.finalize(requires_grad=True)
        with self.assertRaisesRegex(NotImplementedError, "requires_grad"):
            _solver(model)


def _solver(model, **overrides):
    options = {
        "pgs_mode": "matrix_free",
        "angular_damping": 0.0,
        "enable_joint_friction": True,
        "pgs_iterations": 16,
    }
    options.update(overrides)
    return SolverFeatherPGS(model, **options)


def _advance(solver, states, control, steps, dt=DT):
    q, qd = [], []
    for _ in range(steps):
        states[0].clear_forces()
        solver.step(states[0], states[1], control, None, dt)
        states.reverse()
        q.append(states[0].joint_q.numpy().copy())
        qd.append(states[0].joint_qd.numpy().copy())
    return np.squeeze(np.array(q)), np.squeeze(np.array(qd))


def _run(model, *, effort=0.0, qd0=0.0, steps, dt=DT, dofs=1, **overrides):
    solver = _solver(model, **overrides)
    states, control = [model.state(), model.state()], model.control()
    states[0].joint_qd.assign(np.broadcast_to(np.asarray(qd0, dtype=np.float32), (dofs,)).copy())
    control.joint_f.assign(np.broadcast_to(np.asarray(effort, dtype=np.float32), (dofs,)).copy())
    return _advance(solver, states, control, steps, dt)[1]


def _single_dof_builder(joint, *, friction=FRICTION, ke=0.0, kd=0.0, limit=None, velocity_limit=None, gravity=0.0):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, gravity))
    body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * INERTIA))
    kwargs = {"friction": friction, "armature": 0.0, "target_ke": ke, "target_kd": kd}
    if limit is not None:
        kwargs.update(limit_lower=limit[0], limit_upper=limit[1])
    if velocity_limit is not None:
        kwargs["velocity_limit"] = velocity_limit
    if joint == "revolute":
        index = builder.add_joint_revolute(-1, body, axis=(0.0, 0.0, 1.0), **kwargs)
    else:
        index = builder.add_joint_prismatic(-1, body, axis=(1.0, 0.0, 0.0), **kwargs)
    builder.add_articulation([index])
    return builder


def _single_dof_model(joint, **kwargs):
    return _single_dof_builder(joint, **kwargs).finalize()


def _two_link_builder(*, length, mass, friction, ke=0.0, kd=0.0):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    inertia = wp.mat33(np.eye(3) * INERTIA)
    first = builder.add_link(mass=mass, inertia=inertia)
    second = builder.add_link(mass=mass, inertia=inertia)
    common = {"axis": (0.0, 0.0, 1.0), "armature": 0.0, "target_ke": ke, "target_kd": kd}
    j0 = builder.add_joint_revolute(-1, first, friction=friction[0], limit_lower=-2.0, limit_upper=2.0, **common)
    j1 = builder.add_joint_revolute(
        first,
        second,
        parent_xform=wp.transform((length, 0.0, 0.0), wp.quat_identity()),
        friction=friction[1],
        limit_lower=-2.0,
        limit_upper=2.0,
        **common,
    )
    builder.add_articulation([j0, j1])
    return builder


def _two_link_model(**kwargs):
    return _two_link_builder(**kwargs).finalize()


def _vertical_slider_model(*, mass, half, friction):
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    body = builder.add_link(mass=mass, inertia=wp.mat33(np.eye(3) * 0.01))
    builder.add_shape_box(body, hx=half, hy=half, hz=half, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
    joint = builder.add_joint_prismatic(
        -1,
        body,
        axis=(0.0, 0.0, 1.0),
        parent_xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()),
        friction=friction,
        armature=0.0,
    )
    builder.add_articulation([joint])
    return builder.finalize()


def _pendulum_model(*, friction):
    builder = newton.ModelBuilder()
    # A fixed root anchors the island so it may sleep.
    base = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.01))
    body = builder.add_link(mass=1.0, com=wp.vec3(0.3, 0.0, 0.0), inertia=wp.mat33(np.eye(3) * 0.01))
    root = builder.add_joint_fixed(-1, base, parent_xform=wp.transform((0.0, 0.0, 1.0), wp.quat_identity()))
    joint = builder.add_joint_revolute(base, body, axis=(0.0, 1.0, 0.0), friction=friction, armature=0.0)
    builder.add_articulation([root, joint])
    return builder.finalize()


def _unsupported_joint_model(joint, *, friction):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * INERTIA))
    index = builder.add_joint_ball(-1, body) if joint == "ball" else builder.add_joint_free(body)
    builder.add_articulation([index])
    builder.joint_friction = [friction] * len(builder.joint_friction)
    return builder.finalize()


def _arm_and_box_model():
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    link = builder.add_link(xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()))
    builder.add_shape_box(link, hx=0.2, hy=0.05, hz=0.05)
    joint = builder.add_joint_revolute(
        -1, link, axis=(0.0, 1.0, 0.0), parent_xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()), friction=0.1
    )
    builder.add_articulation([joint])
    box = builder.add_body(xform=wp.transform((0.6, 0.0, 0.1), wp.quat_identity()))
    builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.1)
    return builder.finalize()


if __name__ == "__main__":
    unittest.main(verbosity=2)
