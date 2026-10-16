# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental passive-island sleeping of SolverFeatherPGS."""

import functools
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs._testing import kernel_overrides
from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_TORSION
from newton.solvers import SolverFeatherPGS
from newton.tests.feather_pgs._fixtures import set_torsion_radius
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices


class TestSleeping(unittest.TestCase):
    def test_disabled_allocates_no_state(self):
        """Keep sleep state absent on the default solver path."""
        _model, _pipeline, solver, _states, _control = _scene(enabled=False)
        self.assertIsNone(solver._sleeping)

    def test_separated_boxes_sleep_and_force_wakes_one(self):
        """Sleep independent boxes and wake only the forced island."""
        model, pipeline, solver, states, control = _scene()
        _advance(model, pipeline, solver, states, control, 120)
        np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0])
        self.assertNotEqual(*solver._sleeping.body_island.numpy())
        before = states[0].body_q.numpy().copy()
        _advance(model, pipeline, solver, states, control, 10)
        np.testing.assert_array_equal(states[0].body_q.numpy(), before)
        force = np.zeros((2, 6), dtype=np.float32)
        force[0, 0] = 100
        states[0].body_f.assign(force)
        _advance(model, pipeline, solver, states, control, 1, clear=False)
        np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [1, 0])
        self.assertGreater(states[0].body_qd.numpy()[0, 0], 0)

    def test_nonfinite_force_wakes_and_propagates(self):
        """A NaN external force on a sleeping body wakes its island instead of publishing a frozen state."""
        model, pipeline, solver, states, control = _scene()
        _advance(model, pipeline, solver, states, control, 120)
        np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0])
        force = np.zeros((2, 6), dtype=np.float32)
        force[0, 0] = np.nan
        states[0].body_f.assign(force)
        _advance(model, pipeline, solver, states, control, 1, clear=False)
        np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [1, 0])
        self.assertFalse(np.all(np.isfinite(states[0].body_qd.numpy()[0])))

    def test_reset_and_gravity_change_wake(self):
        """Wake state after reset and model-property notifications."""
        model, pipeline, solver, states, control = _scene()
        _advance(model, pipeline, solver, states, control, 120)
        solver.reset(states[0])
        np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [1, 1])
        _advance(model, pipeline, solver, states, control, 120)
        model.set_gravity((0, 0, 9.81))
        solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
        _advance(model, pipeline, solver, states, control, 1)
        np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [1, 1])
        self.assertGreater(states[0].body_qd.numpy()[0, 2], 0)

    def test_free_flight_cannot_sleep(self):
        """Keep unsupported free bodies awake at zero speed."""
        model, pipeline, solver, states, control = _scene(ground=False)
        model.set_gravity((0, 0, 0))
        _advance(model, pipeline, solver, states, control, 120)
        np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [1, 1])

    def test_invalid_options(self):
        """Reject unsupported sleep settings explicitly."""
        model, *_ = _scene(enabled=False)
        for kwargs in (
            {"sleep_quiet_time": 0},
            {"sleep_linear_threshold": float("nan")},
            {"velocity_iterations": 1},
            {"contact_warmstart": True},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                newton.solvers.SolverFeatherPGS(model, enable_sleeping=True, friction_anchor_beta=0, **kwargs)
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        empty = builder.finalize(device=model.device)
        newton.solvers.SolverFeatherPGS(empty, friction_anchor_beta=0)
        with self.assertRaisesRegex(ValueError, "at least one articulation"):
            newton.solvers.SolverFeatherPGS(empty, enable_sleeping=True, friction_anchor_beta=0)

    def test_default_and_explicit_disabled_match(self):
        """Preserve the existing trajectory with sleeping explicitly disabled."""
        model, pipeline, explicit, states, control = _scene(enabled=False)
        default = newton.solvers.SolverFeatherPGS(model, iterations=16, solve_mode="split", friction_anchor_beta=0)
        reference = [model.state(), model.state()]
        _advance(model, pipeline, explicit, states, control, 30)
        _advance(model, pipeline, default, reference, control, 30)
        np.testing.assert_array_equal(states[0].joint_q.numpy(), reference[0].joint_q.numpy())
        np.testing.assert_array_equal(states[0].joint_qd.numpy(), reference[0].joint_qd.numpy())

    def test_stack_wakes_together(self):
        """Wake the whole contacting stack when its top box is forced."""
        model, pipeline, solver, states, control = _scene(stack=True)
        _advance(model, pipeline, solver, states, control, 200)
        np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0])
        islands = solver._sleeping.body_island.numpy()
        self.assertEqual(islands[0], islands[1])
        force = np.zeros((2, 6), dtype=np.float32)
        force[1, 0] = 100
        states[0].body_f.assign(force)
        _advance(model, pipeline, solver, states, control, 1, clear=False)
        np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [1, 1])

    def test_cuda_graph_sleep_and_wake(self):
        """Advance sleep timers and force waking on captured graph replay."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA required")
        model, pipeline, solver, states, control = _scene(device="cuda:0")
        contacts = pipeline.contacts()
        for _ in range(2):
            pipeline.collide(states[0], contacts)
            solver.step(states[0], states[1], control, contacts, 0.005)
            states.reverse()
        with wp.ScopedCapture(device=model.device) as capture:
            for _ in range(2):
                pipeline.collide(states[0], contacts)
                solver.step(states[0], states[1], control, contacts, 0.005)
                states.reverse()
        for _ in range(100):
            wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0])
        forces = np.zeros((2, 6), dtype=np.float32)
        forces[0, 0] = 100
        states[0].body_f.assign(forces)
        states[1].body_f.assign(forces)
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [1, 0])


def _scene(*, enabled=True, ground=True, device="cpu", stack=False):
    builder = newton.ModelBuilder()
    if ground:
        builder.add_ground_plane()
    for index, x in enumerate((0.0, 0.5)):
        position = wp.vec3(0, 0, 0.1 + index * 0.2) if stack else wp.vec3(x, 0, 0.1)
        body = builder.add_body(xform=wp.transform(position, wp.quat_identity()))
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=64)
    solver = newton.solvers.SolverFeatherPGS(
        model,
        enable_sleeping=enabled,
        sleep_quiet_time=0.05,
        iterations=16,
        friction_anchor_beta=0,
        solve_mode="matrix_free" if model.device.is_cuda else "split",
    )
    return model, pipeline, solver, [model.state(), model.state()], model.control()


def _advance(model, pipeline, solver, states, control, steps, *, clear=True):
    contacts = pipeline.contacts()
    for _ in range(steps):
        if clear:
            states[0].clear_forces()
        pipeline.collide(states[0], contacts)
        solver.step(states[0], states[1], control, contacts, 0.005)
        states.reverse()


DT = 0.005


SLEEP = {"enable_sleeping": True, "sleep_quiet_time": 0.05}


def _drive_scene(device, sleeping):
    """A position-driven fixed-base hinge next to an independent box on the ground."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    link = builder.add_link(xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()))
    builder.add_shape_box(link, hx=0.2, hy=0.02, hz=0.02)
    hinge = builder.add_joint_revolute(
        parent=-1,
        child=link,
        axis=newton.Axis.Z,
        parent_xform=wp.transform((0.0, 0.0, 0.5), wp.quat_identity()),
        target_ke=200.0,
        target_kd=20.0,
        actuator_mode=newton.JointTargetMode.POSITION,
    )
    builder.add_articulation([hinge])
    box = builder.add_body(xform=wp.transform((1.0, 0.0, 0.1), wp.quat_identity()))
    builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.1)
    model = builder.finalize(device=device)
    solver = SolverFeatherPGS(
        model,
        solve_mode="matrix_free",
        drive_mode="rows",
        iterations=16,
        friction_anchor_beta=0.0,
        **(SLEEP if sleeping else {}),
    )
    control = model.control()
    control.joint_target_q.assign(np.array([0.5], dtype=np.float32))
    return model, newton.CollisionPipeline(model, rigid_contact_max=64), solver, control, link, box


def _run(model, pipeline, solver, control, states, steps, *, contacts=None, force=None, stiffness=None):
    contacts = contacts or pipeline.contacts()
    for _ in range(steps):
        states[0].clear_forces()
        if force is not None:
            states[0].body_f.assign(force)
        pipeline.collide(states[0], contacts)
        if stiffness is not None:
            contacts.rigid_contact_stiffness.fill_(stiffness)
            contacts.rigid_contact_damping.fill_(20.0)
            contacts.rigid_contact_friction.fill_(1.0)
        solver.step(states[0], states[1], control, contacts, DT)
        states.reverse()


def test_drive_rows_stay_awake_while_a_box_sleeps(test, device):
    """Keep a drive-row articulation awake and tracking while an independent box sleeps and wakes."""
    model, pipeline, solver, control, link, box = _drive_scene(device, sleeping=True)
    reference_model, reference_pipeline, reference, reference_control, _, _ = _drive_scene(device, sleeping=False)
    states = [model.state(), model.state()]
    reference_states = [reference_model.state(), reference_model.state()]
    _run(model, pipeline, solver, control, states, 300)
    _run(reference_model, reference_pipeline, reference, reference_control, reference_states, 300)
    awake = solver._sleeping.body_awake.numpy()
    test.assertEqual(int(awake[link]), 1)
    test.assertEqual(int(awake[box]), 0)
    # The driven hinge reaches its target, exactly as without sleeping.
    test.assertAlmostEqual(float(states[0].joint_q.numpy()[0]), 0.5, delta=0.01)
    np.testing.assert_allclose(states[0].joint_q.numpy()[:1], reference_states[0].joint_q.numpy()[:1], atol=1.0e-5)
    frozen = states[0].body_q.numpy()[box].copy()
    _run(model, pipeline, solver, control, states, 10)
    np.testing.assert_array_equal(states[0].body_q.numpy()[box], frozen)
    # A push wakes the box, and the drive keeps holding its target.
    force = np.zeros((model.body_count, 6), dtype=np.float32)
    force[box, 0] = 300.0
    _run(model, pipeline, solver, control, states, 5, force=force)
    test.assertEqual(int(solver._sleeping.body_awake.numpy()[box]), 1)
    test.assertGreater(float(states[0].body_q.numpy()[box, 0]), float(frozen[0]))
    test.assertAlmostEqual(float(states[0].joint_q.numpy()[0]), 0.5, delta=0.01)


class TestFeatherPGSSleepingCombinations(unittest.TestCase):
    pass


for _name, _func in (("test_drive_rows_stay_awake_while_a_box_sleeps", test_drive_rows_stay_awake_while_a_box_sleeps),):
    add_function_test(TestFeatherPGSSleepingCombinations, _name, _func, devices=get_cuda_test_devices())


_FIELDS = ("body_q", "body_qd", "joint_q", "joint_qd")


@unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
class TestSleepingDynamicsSkip(unittest.TestCase):
    def test_eager_trajectories_match(self):
        """Settle, sleep, force-wake and resettle identically with and without skipped dynamics."""
        for interval in (1, 3):
            for velocity_limits in (False, True):
                with self.subTest(interval=interval, velocity_limits=velocity_limits):
                    self._compare(interval=interval, velocity_limits=velocity_limits, graph=False)

    def test_alternate_kernel_paths_match(self):
        """Cover tiled triangular solves and cooperative tree traversal."""
        for options in ({"kernel_overrides": {"trisolve_kernel": "tiled"}}, {"parallel_tree": True}):
            with self.subTest(options=options):
                self._compare(interval=3, velocity_limits=False, graph=False, **options)

    def test_graph_trajectories_match(self):
        """Replay captured steps identically with and without skipped dynamics."""
        for interval in (1, 2):
            with self.subTest(interval=interval):
                self._compare(interval=interval, velocity_limits=True, graph=True)

    def _compare(self, *, interval, velocity_limits, graph, **options):
        runs = [_Run(self, skip, interval, velocity_limits, graph, options) for skip in (False, True)]
        self.assertTrue(runs[1].solver._sleep_skips_dynamics)
        slept = woke = False
        for phase, steps, force in (("settle", 400, 0.0), ("push", 6, 20.0), ("resettle", 600, 0.0)):
            for step in range(steps):
                for run in runs:
                    run.advance(force)
                # The reference computes dynamics even while both runs omit sleeping constraint rows.
                self.assertTrue(runs[0].solver._dynamics_art_active.numpy().all())
                np.testing.assert_array_equal(
                    runs[1].solver._sleeping.art_awake.numpy(), runs[0].solver._sleeping.art_awake.numpy()
                )
                for field in _FIELDS:
                    np.testing.assert_array_equal(
                        runs[1].field(field), runs[0].field(field), err_msg=f"{phase} step {step} {field}"
                    )
                awake = runs[1].solver._sleeping.art_awake.numpy()
                slept |= phase == "settle" and not awake[:2].any()
                woke |= phase == "push" and bool(awake[0])
        self.assertTrue(slept and woke)
        # The driven mimic articulation never sleeps, and everything else settles again.
        np.testing.assert_array_equal(runs[1].solver._sleeping.art_awake.numpy(), [0, 0, 0, 1])
        self.assertFalse(runs[1].solver._dynamics_art_active.numpy()[:3].any())


class _Run:
    def __init__(self, test, skip, interval, velocity_limits, graph, options):
        self.model = _scene_sleeping_dynamics()
        self.pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=512)
        options = dict(options)
        overrides = options.pop("kernel_overrides", {})
        with kernel_overrides(overrides):
            self.solver = _solver(
                test,
                self.model,
                update_mass_matrix_interval=interval,
                enable_joint_velocity_limits=velocity_limits,
                **options,
            )
        if not skip:
            self.solver._sleeping = _SleepReference(self.solver._sleeping, compute_dynamics=True)
        self.states = [self.model.state(), self.model.state()]
        self.control = self.model.control()
        self.contacts = self.pipeline.contacts()
        self.graph = None
        if graph:
            self._step()
            self._step()
            with wp.ScopedCapture(device=self.model.device) as capture:
                self._step()
                self._step()
            self.graph = capture.graph
        self.force = np.zeros((self.model.body_count, 6), dtype=np.float32)

    def advance(self, force):
        self.force[0, 0] = force
        for state in self.states:
            state.body_f.assign(self.force)
        if self.graph is None:
            self._step()
        else:
            wp.capture_launch(self.graph)
            self.solver.check_constraint_capacity()

    def _step(self):
        self.pipeline.collide(self.states[0], self.contacts)
        self.solver.step(self.states[0], self.states[1], self.control, self.contacts, 0.005)
        self.states.reverse()

    def field(self, name):
        return getattr(self.states[0], name).numpy()


def _scene_sleeping_dynamics():
    """Two resting branching articulations, a resting box, and a driven fixed-base arm with a mimic joint."""
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    for x in (0.0, 1.0):
        base = builder.add_link(xform=wp.transform((x, 0.0, 0.1), wp.quat_identity()))
        builder.add_shape_box(base, hx=0.2, hy=0.1, hz=0.1)
        joints = [builder.add_joint_free(child=base)]
        # Two tips make the articulation a branching tree.
        for side in (-1.0, 1.0):
            tip = builder.add_link(xform=wp.transform((x + side * 0.3, 0.0, 0.1), wp.quat_identity()))
            builder.add_shape_box(tip, hx=0.1, hy=0.1, hz=0.1)
            hinge = wp.transform((side * 0.3, 0.0, 0.0), wp.quat_identity())
            joints.append(builder.add_joint_revolute(parent=base, child=tip, parent_xform=hinge, axis=(0.0, 1.0, 0.0)))
        builder.add_articulation(joints)
    box = builder.add_body(xform=wp.transform((3.0, 0.0, 0.1), wp.quat_identity()))
    builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.1)
    root = builder.add_link(xform=wp.transform((5.0, 0.0, 0.5), wp.quat_identity()))
    builder.add_shape_box(root, hx=0.05, hy=0.05, hz=0.05)
    joints = [
        builder.add_joint_fixed(parent=-1, child=root, parent_xform=wp.transform((5.0, 0.0, 0.5), wp.quat_identity()))
    ]
    parent = root
    for k in range(3):
        link = builder.add_link(xform=wp.transform((5.0, 0.0, 0.3 - 0.2 * k), wp.quat_identity()))
        builder.add_shape_box(link, hx=0.03, hy=0.03, hz=0.08)
        joints.append(
            builder.add_joint_revolute(
                parent=parent,
                child=link,
                parent_xform=wp.transform((0.0, 0.0, -0.2), wp.quat_identity()),
                axis=(0.0, 1.0, 0.0),
                target_ke=100.0,
                target_kd=10.0,
                limit_lower=-1.0,
                limit_upper=1.0,
            )
        )
        parent = link
    builder.add_articulation(joints)
    builder.set_joint_mimic(joints[3], joints[2])
    builder.joint_velocity_limit[:] = [5.0] * len(builder.joint_velocity_limit)
    return builder.finalize(device="cuda:0")


_PROFILE = {
    "solve_mode": "matrix_free",
    "row_schedule": "interleaved",
    "articulated_contact_response": "immediate",
    "iterations": 32,
    "contact_regularization": 0.01,
    "contact_shared_anchor": True,
    "contact_friction_shared_anchor": True,
    "friction_anchor_beta": 0.2,
    "contact_friction_gap_max": 0.001,
    "contact_torsion_device": True,
    "articulation_constraint_max": 512,
    "free_body_constraint_max": 512,
    "enable_sleeping": True,
    "sleep_quiet_time": 0.1,
}


@unittest.skipUnless(wp.is_cuda_available(), "CUDA required")
class TestSleepingProductionProfile(unittest.TestCase):
    def test_anchored_torsion_articulations_sleep_and_wake(self):
        """Drop every row of a settled scene and restore anchored torsion rows on a force wake."""
        model, pipeline, solver, states, control = _articulations(self)
        _advance_sleeping_production(pipeline, solver, states, control, 5)
        awake_rows = int(solver._constraint_count.numpy()[0])
        awake_torsion = _torsion_rows(solver)
        self.assertGreater(awake_torsion, 0)
        _advance_sleeping_production(pipeline, solver, states, control, 400)
        np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0, 0, 0])
        np.testing.assert_array_equal(solver._constraint_count.numpy(), [0])
        np.testing.assert_array_equal(solver._mf_constraint_count.numpy(), [0])
        frozen = states[0].body_q.numpy().copy()
        _advance_sleeping_production(pipeline, solver, states, control, 20)
        np.testing.assert_array_equal(states[0].body_q.numpy(), frozen)

        # Pushing the first articulation restores its rows and leaves the second asleep.
        force = np.zeros((model.body_count, 6), dtype=np.float32)
        force[0, 0] = 20.0
        for _ in range(5):
            states[0].body_f.assign(force)
            _advance_sleeping_production(pipeline, solver, states, control, 1, clear=False)
        np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [1, 1, 0, 0])
        self.assertEqual(int(solver._constraint_count.numpy()[0]), awake_rows // 2)
        self.assertEqual(_torsion_rows(solver), awake_torsion // 2)
        self.assertGreater(states[0].body_q.numpy()[0, 0], frozen[0, 0])
        np.testing.assert_array_equal(states[0].body_q.numpy()[2:], frozen[2:])

        _advance_sleeping_production(pipeline, solver, states, control, 600)
        np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0, 0, 0])
        self.assertFalse(np.any(solver.constraint_overflow.numpy()))

    def test_graph_replay_validates_torsion(self):
        """Sleep and wake inside a captured graph with mandatory torsion validation."""
        model, pipeline, solver, states, control = _articulations(self)
        contacts = pipeline.contacts()
        _advance_sleeping_production(pipeline, solver, states, control, 2, contacts=contacts)
        with wp.ScopedCapture(device=model.device) as capture:
            for _ in range(2):
                states[0].clear_forces()
                pipeline.collide(states[0], contacts)
                solver.step(states[0], states[1], control, contacts, 0.005)
                states.reverse()
        for _ in range(200):
            wp.capture_launch(capture.graph)
        solver.check_constraint_capacity()
        np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0, 0, 0])
        solver._sleeping.wake()
        wp.capture_launch(capture.graph)
        solver.check_constraint_capacity()
        self.assertGreater(_torsion_rows(solver), 0)

    def test_masked_reset_wakes_only_selected_worlds(self):
        """Wake only the worlds a masked reset selects, including none for an empty mask."""
        model, pipeline, solver, states = _two_world_boxes(self)
        _advance_sleeping_production(pipeline, solver, states, model.control(), 400)
        np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0])
        for mask, expected in (([False, False], [0, 0]), ([True, False], [1, 0])):
            solver.reset(states[0], world_mask=wp.array(mask, dtype=wp.bool, device="cuda:0"))
            np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), expected)
        _advance_sleeping_production(pipeline, solver, states, model.control(), 1)
        np.testing.assert_array_equal(solver._sleeping.art_awake.numpy(), [1, 0])

    def test_property_notification_wakes_only_changed_islands(self):
        """Wake the island whose mass changed, and nothing for a notification without a change."""
        model, pipeline, solver, states = _two_world_boxes(self)
        _advance_sleeping_production(pipeline, solver, states, model.control(), 400)
        np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0])
        solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
        _advance_sleeping_production(pipeline, solver, states, model.control(), 1)
        np.testing.assert_array_equal(solver._sleeping.art_awake.numpy(), [0, 0])
        mass = model.body_mass.numpy()
        mass[0] *= 2.0
        model.body_mass.assign(mass)
        solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
        _advance_sleeping_production(pipeline, solver, states, model.control(), 1)
        np.testing.assert_array_equal(solver._sleeping.art_awake.numpy(), [1, 0])

    def test_frozen_patch_carry_matches_rebuild(self):
        """Carrying frozen friction patches matches rebuilding them through sleep, wake and resettle."""
        self._check_carry(tiles=False)

    def test_frozen_patch_carry_matches_rebuild_for_flooded_pairs(self):
        """Pairs large enough for the warp flood skip it while frozen and rebuild identically on wake."""
        self._check_carry(tiles=True)

    def _check_carry(self, tiles):
        trajectories, flood_pairs = [], []
        for carry in (False, True):
            model, pipeline, solver, states, control = _articulations(self, tiles=tiles)
            if not carry:
                solver._sleeping = _SleepReference(solver._sleeping, rebuild_patches=True)
            self.assertEqual(solver._sleeping.patch_frozen_bodies is not None, carry)
            contacts = pipeline.contacts()
            frozen = 0
            trajectory = []
            for step in range(900):
                states[0].clear_forces()
                if 400 <= step < 405:
                    force = np.zeros((model.body_count, 6), dtype=np.float32)
                    force[0, 0] = 20.0
                    states[0].body_f.assign(force)
                pipeline.collide(states[0], contacts)
                solver.step(states[0], states[1], control, contacts, 0.005)
                states.reverse()
                frozen += int(solver._sleeping.frozen_bodies.numpy().sum())
                if step == 399:
                    flood_pairs.append(int(solver._friction_patches._flood_pair_count.numpy()[0]))
                trajectory.append(
                    np.concatenate(
                        [
                            getattr(states[0], field).numpy().ravel()
                            for field in ("body_q", "body_qd", "joint_q", "joint_qd")
                        ]
                    )
                )
            self.assertGreater(frozen, 0)
            np.testing.assert_array_equal(solver._sleeping.art_awake.numpy(), [0, 0])
            trajectories.append(np.array(trajectory))
        if not tiles:
            np.testing.assert_array_equal(trajectories[1][:400], trajectories[0][:400])
            # Rebuilding re-derives the carried history each sleeping step, which only differs by roundoff.
            np.testing.assert_allclose(trajectories[1], trajectories[0], rtol=0.0, atol=1.0e-6)
            return
        # Frozen pairs skip the warp flood; bodies built from many shapes are not bitwise reproducible.
        self.assertGreater(flood_pairs[0], 0)
        self.assertEqual(flood_pairs[1], 0)
        positions = [t[-1][: 7 * 4].reshape(4, 7)[:, :3] for t in trajectories]
        np.testing.assert_allclose(positions[1], positions[0], rtol=0.0, atol=2.0e-3)

    def test_successive_notifications_keep_every_wake(self):
        """A second notification before the next step keeps the first notification's wake."""
        model, pipeline, solver, states = _two_world_boxes(self)
        _advance_sleeping_production(pipeline, solver, states, model.control(), 400)
        np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0])
        mass = model.body_mass.numpy()
        mass[0] *= 2.0
        model.body_mass.assign(mass)
        solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
        mu = model.shape_material_mu.numpy()
        mu[model.shape_body.numpy() == 1] *= 0.5
        model.shape_material_mu.assign(mu)
        solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
        _advance_sleeping_production(pipeline, solver, states, model.control(), 1)
        np.testing.assert_array_equal(solver._sleeping.art_awake.numpy(), [1, 1])

    def test_replaced_property_array_wakes_its_island(self):
        """A model array replaced rather than assigned in place still wakes the island it changed."""
        model, pipeline, solver, states = _two_world_boxes(self)
        _advance_sleeping_production(pipeline, solver, states, model.control(), 400)
        np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0])
        com = model.body_com.numpy()
        com[0, 0] = 0.3
        model.body_com = wp.array(com, dtype=wp.vec3, device=model.device)
        solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
        _advance_sleeping_production(pipeline, solver, states, model.control(), 1)
        np.testing.assert_array_equal(solver._sleeping.art_awake.numpy(), [1, 0])

    def test_resized_property_array_is_rejected(self):
        """A model array replaced with a different entity count cannot be diffed and raises."""
        model, pipeline, solver, states = _two_world_boxes(self)
        _advance_sleeping_production(pipeline, solver, states, model.control(), 1)
        model.body_mass = wp.zeros(model.body_count + 1, dtype=float, device=model.device)
        with self.assertRaisesRegex(ValueError, "body_mass"):
            solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)

    def test_coincident_entity_counts_keep_property_owners(self):
        """A body array is diffed only as a body property when body and joint counts coincide."""
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        bodies = []
        for x in (0.0, 0.5):
            body = builder.add_link(xform=wp.transform((x, 0.0, 0.1), wp.quat_identity()))
            builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
            bodies.append(body)
        # Reversed joint order gives body_to_articulation [1, 0] against joint_articulation [0, 1].
        for body in reversed(bodies):
            builder.add_articulation([builder.add_joint_free(child=body)])
        model = builder.finalize(device="cuda:0")
        self.assertEqual(model.body_count, model.joint_count)
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=256)
        solver = _solver(self, model)
        states = [model.state(), model.state()]
        np.testing.assert_array_equal(solver._body_to_articulation.numpy(), [1, 0])
        _advance_sleeping_production(pipeline, solver, states, model.control(), 400)
        np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0])
        mass = model.body_mass.numpy()
        mass[0] *= 2.0
        model.body_mass.assign(mass)
        solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
        _advance_sleeping_production(pipeline, solver, states, model.control(), 1)
        np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [1, 0])
        _advance_sleeping_production(pipeline, solver, states, model.control(), 400)
        np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0])
        solver.notify_model_changed(newton.ModelFlags.JOINT_PROPERTIES)
        _advance_sleeping_production(pipeline, solver, states, model.control(), 1)
        np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0])

    def test_solver_body_attribute_notification_wakes_its_island(self):
        """A FeatherPGS custom body attribute assigned to the model is diffed like a core body property."""
        box = newton.ModelBuilder()
        body = box.add_body(xform=wp.transform((0.0, 0.0, 0.1), wp.quat_identity()))
        box.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        builder = newton.ModelBuilder()
        newton.solvers.SolverFeatherPGS.register_custom_attributes(builder)
        builder.add_ground_plane()
        builder.replicate(box, 2)
        model = builder.finalize(device="cuda:0")
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=256)
        solver = _solver(self, model)
        states = [model.state(), model.state()]
        _advance_sleeping_production(pipeline, solver, states, model.control(), 400)
        np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0])
        depenetration = model.feather_pgs.rigid_body_max_depenetration_velocity.numpy()
        depenetration[0] = 1.0
        model.feather_pgs.rigid_body_max_depenetration_velocity.assign(depenetration)
        solver.notify_model_changed(newton.ModelFlags.BODY_PROPERTIES)
        _advance_sleeping_production(pipeline, solver, states, model.control(), 1)
        np.testing.assert_array_equal(solver._sleeping.art_awake.numpy(), [1, 0])

    def test_mimic_articulation_stays_awake(self):
        """Keep an articulation with mimic rows awake while an independent box sleeps."""
        for legacy in (False, True):
            with self.subTest(legacy_constraint_mimic=legacy):
                builder = newton.ModelBuilder()
                builder.add_ground_plane()
                base = builder.add_link(xform=wp.transform((0.0, 0.0, 0.05), wp.quat_identity()))
                builder.add_shape_box(base, hx=0.2, hy=0.1, hz=0.05)
                joints = [builder.add_joint_free(child=base)]
                for side in (-1.0, 1.0):
                    finger = builder.add_link(xform=wp.transform((side * 0.15, 0.0, 0.15), wp.quat_identity()))
                    builder.add_shape_box(finger, hx=0.02, hy=0.02, hz=0.05)
                    joints.append(
                        builder.add_joint_revolute(
                            parent=base,
                            child=finger,
                            parent_xform=wp.transform((side * 0.15, 0.0, 0.1), wp.quat_identity()),
                        )
                    )
                builder.add_articulation(joints)
                if legacy:
                    _expect_one_warning(
                        self,
                        DeprecationWarning,
                        r"add_constraint_mimic\(\) is deprecated",
                        functools.partial(builder.add_constraint_mimic, joint0=joints[2], joint1=joints[1]),
                    )
                else:
                    builder.set_joint_mimic(joints[2], joints[1])
                box = builder.add_body(xform=wp.transform((2.0, 0.0, 0.1), wp.quat_identity()))
                builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.1)
                model = builder.finalize(device="cuda:0")
                solver = _solver(self, model, torsion_radius=0.0)
                self.assertGreater(solver._mimic_count, 0)
                pipeline = newton.CollisionPipeline(model, rigid_contact_max=256)
                _advance_sleeping_production(pipeline, solver, [model.state(), model.state()], model.control(), 400)
                awake = solver._sleeping.art_awake.numpy()
                self.assertEqual(int(awake[solver._body_to_articulation.numpy()[base]]), 1)
                self.assertEqual(int(awake[solver._body_to_articulation.numpy()[box]]), 0)


class _SleepReference:
    """Compute discarded sleeping dynamics or rebuild frozen patches entirely within tests."""

    def __init__(self, sleeping, *, compute_dynamics=False, rebuild_patches=False):
        self._sleeping = sleeping
        self._compute_dynamics = compute_dynamics
        self._rebuild_patches = rebuild_patches

    def __getattr__(self, name):
        return getattr(self._sleeping, name)

    @property
    def patch_frozen_bodies(self):
        return None if self._rebuild_patches else self._sleeping.patch_frozen_bodies

    def begin(self, state, control, contacts):
        self._sleeping.begin(state, control, contacts)
        if self._compute_dynamics:
            solver = self._sleeping.solver
            # These device operations are captured too, so every graph replay recomputes the reference.
            for name in (
                "_dynamics_art_active",
                "_dynamics_art_mask",
                "_dynamics_dof_active",
                "_dynamics_joint_active",
                "_dynamics_body_active",
            ):
                getattr(solver, name).fill_(1)
            solver._fk_id_cache_valid.zero_()


def _articulations(test, tiles=False):
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
    model = builder.finalize(device="cuda:0")
    # Deterministic contact order, as in production, so A/B trajectories compare bitwise.
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=512, deterministic=True)
    solver = _solver(test, model)
    return model, pipeline, solver, [model.state(), model.state()], model.control()


def _two_world_boxes(test):
    box = newton.ModelBuilder()
    body = box.add_body(xform=wp.transform((0.0, 0.0, 0.1), wp.quat_identity()))
    box.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    builder.replicate(box, 2)
    model = builder.finalize(device="cuda:0")
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=256)
    solver = _solver(test, model)
    return model, pipeline, solver, [model.state(), model.state()]


def _solver(test, model, torsion_radius=0.01, **overrides):
    """Construct the profile solver, which warns that patch friction keeps shared anchors on normal rows only."""
    set_torsion_radius(model, torsion_radius)
    return _expect_one_warning(
        test,
        UserWarning,
        "contact_shared_anchor still applies to normal rows",
        lambda: newton.solvers.SolverFeatherPGS(model, **{**_PROFILE, **overrides}),
    )


def _expect_one_warning(test, category, pattern, call):
    """Return ``call()``, requiring it to emit exactly one warning, of ``category`` and matching ``pattern``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = call()
    test.assertEqual(len(caught), 1, [f"{w.category.__name__}: {w.message}" for w in caught])
    test.assertIs(caught[0].category, category)
    test.assertRegex(str(caught[0].message), pattern)
    return result


def _torsion_rows(solver):
    row_types = solver._row_type.numpy()[0, : int(solver._constraint_count.numpy()[0])]
    return int(np.count_nonzero(row_types == PGS_CONSTRAINT_TYPE_TORSION))


def _advance_sleeping_production(pipeline, solver, states, control, steps, *, clear=True, contacts=None):
    contacts = contacts or pipeline.contacts()
    for _ in range(steps):
        if clear:
            states[0].clear_forces()
        pipeline.collide(states[0], contacts)
        solver.step(states[0], states[1], control, contacts, 0.005)
        states.reverse()


# A manipulation profile without contact torsion.
PROFILE = {
    "iterations": 32,
    "contact_regularization": 0.01,
    "friction_anchor_beta": 0.2,
    "contact_friction_gap_max": 0.001,
    "articulation_constraint_max": 512,
    "free_body_constraint_max": 512,
    "enable_sleeping": True,
    "sleep_quiet_time": 0.1,
}


def test_anchored_articulations_sleep_and_wake(test, device):
    """Drop every row of a settled scene and restore the anchored rows of a force-woken articulation."""
    model, pipeline, solver, states, control = _articulations_sleeping_production_upstream(device)
    _advance_sleeping_production_upstream(pipeline, solver, states, control, 5)
    awake_rows = int(solver._constraint_count.numpy()[0])
    test.assertGreater(awake_rows, 0)
    _advance_sleeping_production_upstream(pipeline, solver, states, control, 400)
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0, 0, 0])
    np.testing.assert_array_equal(solver._constraint_count.numpy(), [0])
    np.testing.assert_array_equal(solver._mf_constraint_count.numpy(), [0])
    frozen = states[0].body_q.numpy().copy()
    _advance_sleeping_production_upstream(pipeline, solver, states, control, 20)
    np.testing.assert_array_equal(states[0].body_q.numpy(), frozen)

    # Pushing the first articulation restores its rows and leaves the second asleep.
    force = np.zeros((model.body_count, 6), dtype=np.float32)
    force[0, 0] = 20.0
    for _ in range(5):
        states[0].body_f.assign(force)
        _advance_sleeping_production_upstream(pipeline, solver, states, control, 1, clear=False)
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [1, 1, 0, 0])
    test.assertEqual(int(solver._constraint_count.numpy()[0]), awake_rows // 2)
    test.assertGreater(states[0].body_q.numpy()[0, 0], frozen[0, 0])
    np.testing.assert_array_equal(states[0].body_q.numpy()[2:], frozen[2:])

    _advance_sleeping_production_upstream(pipeline, solver, states, control, 600)
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0, 0, 0])
    test.assertFalse(np.any(solver.constraint_overflow.numpy()))


def test_graph_replay_sleeps_and_wakes(test, device):
    """Sleep inside a captured graph and wake on an explicit wake before the next replay."""
    model, pipeline, solver, states, control = _articulations_sleeping_production_upstream(device)
    contacts = pipeline.contacts()
    _advance_sleeping_production_upstream(pipeline, solver, states, control, 2, contacts=contacts)
    with wp.ScopedCapture(device=model.device) as capture:
        for _ in range(2):
            states[0].clear_forces()
            pipeline.collide(states[0], contacts)
            solver.step(states[0], states[1], control, contacts, 0.005)
            states.reverse()
    for _ in range(200):
        wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0, 0, 0])
    np.testing.assert_array_equal(solver._constraint_count.numpy(), [0])
    solver._sleeping.wake()
    wp.capture_launch(capture.graph)
    test.assertGreater(int(solver._constraint_count.numpy()[0]), 0)


def test_frozen_patch_carry_copies_the_previous_history(test, device):
    """Take a sleeping pair's anchor history from the previous step, not from the current frame's storage."""
    _model, pipeline, solver, states, control = _articulations_sleeping_production_upstream(device)
    contacts = pipeline.contacts()
    _advance_sleeping_production_upstream(pipeline, solver, states, control, 400, contacts=contacts)
    np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0, 0, 0])
    test.assertTrue(np.all(solver._sleeping.frozen_bodies.numpy() == 1))
    current = solver._friction_patches.current
    count = int(contacts.rigid_contact_count.numpy()[0])
    names = ("valid", "displacement", "tangent_impulse", "anchor_a", "anchor_b", "owner")
    before = {name: getattr(current, name).numpy()[:count].copy() for name in names}
    test.assertGreater(int(before["valid"].sum()), 0)
    # Overwrite the current frame; the carry must restore every field from the stored history.
    for name in names:
        getattr(current, name).fill_(7)
    _advance_sleeping_production_upstream(pipeline, solver, states, control, 1, contacts=contacts)
    for name in names:
        np.testing.assert_array_equal(getattr(current, name).numpy()[:count], before[name], err_msg=name)


def _articulations_sleeping_production_upstream(device, tiles=False, profile=None):
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
    solver = newton.solvers.SolverFeatherPGS(model, solve_mode="matrix_free", **(profile or PROFILE))
    return model, pipeline, solver, [model.state(), model.state()], model.control()


def _advance_sleeping_production_upstream(pipeline, solver, states, control, steps, *, clear=True, contacts=None):
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


class TestFeatherPGSSleepingSafety(unittest.TestCase):
    def test_fixed_table_does_not_merge_dynamic_islands(self):
        """Treat a zero-response table as a support boundary between free boxes."""
        for fixed_articulation in (False, True):
            with self.subTest(fixed_articulation=fixed_articulation):
                builder = newton.ModelBuilder()
                table = -1
                if fixed_articulation:
                    table = builder.add_link()
                    root = builder.add_joint_fixed(parent=-1, child=table)
                    builder.add_articulation([root])
                builder.add_shape_box(table, hx=1.0, hy=0.5, hz=0.2)
                boxes = []
                for x in (-0.5, 0.5):
                    body = builder.add_body(xform=wp.transform(wp.vec3(x, 0.0, 0.3), wp.quat_identity()))
                    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
                    boxes.append(body)
                model = builder.finalize(device="cpu")
                pipeline, solver, state_in, _state_out, control = _runtime(model)
                contacts = pipeline.contacts()
                pipeline.collide(state_in, contacts)
                count = int(contacts.rigid_contact_count.numpy()[0])
                shape_body = model.shape_body.numpy()
                body0 = shape_body[contacts.rigid_contact_shape0.numpy()[:count]]
                body1 = shape_body[contacts.rigid_contact_shape1.numpy()[:count]]
                for body in boxes:
                    self.assertTrue(np.any(((body0 == body) & (body1 == table)) | ((body1 == body) & (body0 == table))))
                body_art = solver._body_to_articulation.numpy()
                if fixed_articulation:
                    self.assertEqual(int(solver._model_plan.response_dof_count[body_art[table]]), 0)

                solver._sleeping.begin(state_in, control, contacts)

                roots = solver._sleeping.parent.numpy()[body_art[boxes]]
                self.assertNotEqual(int(roots[0]), int(roots[1]), f"Table joined box islands: {roots}")
                np.testing.assert_array_equal(solver._sleeping.root_supported.numpy()[roots], [1, 1])
                np.testing.assert_array_equal(solver._sleeping.root_veto.numpy()[roots], [0, 0])

    def test_lost_support_wakes_below_velocity_threshold(self):
        """Release an unsupported former island member despite a small gravity kick."""
        model = _boxes(2)
        pipeline, solver, state_in, state_out, control = _runtime(model)
        contacts = pipeline.contacts()
        pipeline.collide(state_in, contacts)
        count = int(contacts.rigid_contact_count.numpy()[0])
        shape_body = model.shape_body.numpy()
        shape0 = contacts.rigid_contact_shape0.numpy()[:count]
        shape1 = contacts.rigid_contact_shape1.numpy()[:count]
        body0, body1 = shape_body[shape0], shape_body[shape1]
        keep = np.flatnonzero(((body0 == 0) & (body1 < 0)) | ((body1 == 0) & (body0 < 0)))
        self.assertGreater(len(keep), 0)
        self.assertTrue(np.any((body0 == 1) | (body1 == 1)))
        # Retain real ground contacts only for A; B belonged to A's sleeping island.
        for suffix in (
            "shape0",
            "shape1",
            "point0",
            "point1",
            "offset0",
            "offset1",
            "normal",
            "margin0",
            "margin1",
            "tids",
        ):
            array = getattr(contacts, "rigid_contact_" + suffix)
            values = array.numpy()
            values[: len(keep)] = values[keep].copy()
            array.assign(values)
        contacts.rigid_contact_count.assign(np.array([len(keep)], dtype=np.int32))
        _seed_sleep(solver, state_in, [0, 0])
        dt = 0.001
        kick = abs(float(model.gravity.numpy()[0, 2])) * dt
        self.assertGreater(kick, 0.0)
        self.assertLess(kick, solver._sleeping.linear_threshold)
        before = state_in.body_q.numpy()[1, 2]

        solver.step(state_in, state_out, control, contacts, dt)

        self.assertEqual(int(solver._sleeping.body_awake.numpy()[1]), 1)
        self.assertLess(state_out.body_q.numpy()[1, 2], before)
        self.assertLess(state_out.body_qd.numpy()[1, 2], 0.0)
        self.assertLess(abs(state_out.body_qd.numpy()[1, 2]), solver._sleeping.linear_threshold)

    def test_wake_crosses_alternating_old_and_current_islands(self):
        """Propagate a wake through alternating historical and current edges only."""
        model = _boxes(7)
        pipeline, solver, state_in, _state_out, control = _runtime(model)
        contacts = pipeline.contacts()
        pipeline.collide(state_in, contacts)
        count = int(contacts.rigid_contact_count.numpy()[0])
        self.assertLessEqual(count + 2, contacts.rigid_contact_max)
        shape_body = model.shape_body.numpy()
        shapes = [int(np.flatnonzero(shape_body == body)[0]) for body in range(7)]
        body0 = shape_body[contacts.rigid_contact_shape0.numpy()[:count]]
        body1 = shape_body[contacts.rigid_contact_shape1.numpy()[:count]]
        for body in range(7):
            self.assertTrue(np.any(((body0 == body) & (body1 < 0)) | ((body1 == body) & (body0 < 0))))
        # Old edges: 0--1, 2--3, 4--5. Current edges: 1--2, 3--4. Body 6 is isolated.
        for name, endpoints in (("shape0", [shapes[1], shapes[3]]), ("shape1", [shapes[2], shapes[4]])):
            array = getattr(contacts, "rigid_contact_" + name)
            values = array.numpy()
            values[count : count + 2] = endpoints
            array.assign(values)
        for suffix in ("point0", "point1", "normal", "margin0", "margin1"):
            array = getattr(contacts, "rigid_contact_" + suffix)
            values = array.numpy()
            values[count : count + 2] = 0
            array.assign(values)
        contacts.rigid_contact_count.assign(np.array([count + 2], dtype=np.int32))
        _seed_sleep(solver, state_in, [0, 0, 2, 2, 4, 4, 6])

        solver._sleeping.begin(state_in, control, contacts)

        np.testing.assert_array_equal(solver._sleeping.art_awake.numpy(), np.zeros(7))
        _seed_sleep(solver, state_in, [0, 0, 2, 2, 4, 4, 6])
        force = np.zeros((model.body_count, 6), dtype=np.float32)
        force[0, 0] = 1.0
        state_in.body_f.assign(force)

        solver._sleeping.begin(state_in, control, contacts)

        np.testing.assert_array_equal(solver._sleeping.art_awake.numpy(), [1, 1, 1, 1, 1, 1, 0])
        np.testing.assert_array_equal(solver._sleeping.quiet_age.numpy()[:6], np.zeros(6))

    def test_passive_spring_vetoes_sleep(self):
        """Keep a quiet spring-loaded articulation awake without motor input."""
        builder = newton.ModelBuilder()
        base = builder.add_link()
        child = builder.add_link()
        builder.add_shape_box(base, hx=0.1, hy=0.1, hz=0.1)
        builder.add_shape_box(child, hx=0.1, hy=0.1, hz=0.1)
        root = builder.add_joint_fixed(parent=-1, child=base)
        slider = builder.add_joint_prismatic(
            parent=base,
            child=child,
            axis=newton.Axis.X,
            target_ke=0.0,
            target_kd=0.0,
            spring_stiffness=0.001,
            spring_ref=0.1,
        )
        builder.add_articulation([root, slider])
        model = builder.finalize(device="cpu")
        model.set_gravity((0.0, 0.0, 0.0))
        _pipeline, solver, state_in, state_out, control = _runtime(model)
        _seed_sleep(solver, state_in, [0])
        self.assertTrue(np.any(solver._passive_spring_stiffness.numpy() > 0.0))
        np.testing.assert_array_equal(control.joint_f.numpy(), np.zeros(model.joint_dof_count))

        solver.step(state_in, state_out, control, None, 0.001)

        np.testing.assert_array_equal(solver._sleeping.art_awake.numpy(), [1])
        self.assertGreater(float(state_out.joint_qd.numpy()[0]), 0.0)
        self.assertLess(float(np.linalg.norm(state_out.body_qd.numpy()[child, :3])), solver._sleeping.linear_threshold)

    def test_nonfinite_trial_output_is_not_hidden(self):
        """Preserve evidence of invalid trial outputs instead of freezing over it."""
        for field in ("body_q", "body_qd", "joint_q", "joint_qd"):
            with self.subTest(field=field):
                model = _boxes(1)
                pipeline, solver, state_in, state_out, control = _runtime(model)
                contacts = pipeline.contacts()
                pipeline.collide(state_in, contacts)
                _seed_sleep(solver, state_in, [0])
                # Awake but quiet: only computed dynamics produce trial outputs to freeze over.
                solver._sleeping.art_awake.fill_(1)
                solver._sleeping.body_awake.fill_(1)
                integrate = solver._stage6_integrate

                def inject_nonfinite(*args, integrate=integrate, state_out=state_out, field=field):
                    integrate(*args)
                    array = getattr(state_out, field)
                    values = array.numpy()
                    values.flat[0] = np.nan
                    array.assign(values)

                with patch.object(solver, "_stage6_integrate", side_effect=inject_nonfinite):
                    solver.step(state_in, state_out, control, contacts, 0.001)

                self.assertFalse(np.isfinite(getattr(state_out, field).numpy()).all())
                np.testing.assert_array_equal(solver._sleeping.art_awake.numpy(), [1])

    def test_aliased_states_are_rejected(self):
        """Reject aliased state allocations before changing physical state."""
        for kind, field in (("same_state", ""), ("alias", "joint_q"), ("alias", "body_q")):
            with self.subTest(kind=kind, field=field):
                model = _boxes(1)
                pipeline, solver, state_in, state_out, control = _runtime(model)
                contacts = pipeline.contacts()
                pipeline.collide(state_in, contacts)
                if kind == "same_state":
                    state_out = state_in
                else:
                    setattr(state_out, field, getattr(state_in, field))
                before = state_in.joint_q.numpy().copy()

                with self.assertRaises(ValueError):
                    solver.step(state_in, state_out, control, contacts, 0.001)

                np.testing.assert_array_equal(state_in.joint_q.numpy(), before)

    def test_extended_state_outputs_are_accepted(self):
        """Sleep with requested acceleration and joint-wrench outputs."""
        model = _boxes(1)
        with self.assertWarns(DeprecationWarning):
            model.request_state_attributes("body_qdd", "body_parent_f")
        pipeline, solver, state_in, state_out, control = _runtime(model)
        self.assertIsNotNone(state_in.body_parent_f)
        contacts = pipeline.contacts()
        states = [state_in, state_out]
        for _ in range(400):
            states[0].clear_forces()
            pipeline.collide(states[0], contacts)
            solver.step(states[0], states[1], control, contacts, 0.005)
            states.reverse()
        np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0])

    def test_invalid_timestep_is_rejected(self):
        """Reject nonpositive and nonfinite timesteps before changing physical state."""
        model = _boxes(1)
        pipeline, solver, state_in, state_out, control = _runtime(model)
        contacts = pipeline.contacts()
        pipeline.collide(state_in, contacts)
        before = state_in.joint_q.numpy().copy()
        for dt in (0.0, -0.001, float("nan"), float("inf")):
            with self.subTest(dt=dt), self.assertRaises(ValueError):
                solver.step(state_in, state_out, control, contacts, dt)
            np.testing.assert_array_equal(state_in.joint_q.numpy(), before)

    def test_reduction_overflow_cannot_authorize_sleep(self):
        """Keep an island awake when its contact producer reports missing contacts."""
        model = _boxes(1)
        pipeline, solver, state_in, state_out, control = _runtime(model)
        contacts = pipeline.contacts()
        pipeline.collide(state_in, contacts)
        self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
        _seed_sleep(solver, state_in, [0])
        contacts._reduction_overflow.fill_(1)

        solver.step(state_in, state_out, control, contacts, 0.001)

        np.testing.assert_array_equal(solver._sleeping.art_awake.numpy(), [1])
        np.testing.assert_array_equal(solver._sleeping.quiet_age.numpy(), [0.0])

    def test_notifications_reset_quiet_age_and_preserve_state(self):
        """Wake sleeping state on model notification or reset without moving bodies."""
        model = _boxes(2)
        _pipeline, solver, state_in, _state_out, _control = _runtime(model)
        before = state_in.joint_q.numpy().copy()
        for event in ("model", "reset"):
            with self.subTest(event=event):
                _seed_sleep(solver, state_in, [0, 0])
                if event == "model":
                    solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
                else:
                    solver.reset(state_in)
                np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [1, 1])
                np.testing.assert_array_equal(solver._sleeping.quiet_age.numpy(), [0.0, 0.0])
                np.testing.assert_array_equal(state_in.joint_q.numpy(), before)


def _boxes(count):
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    for index in range(count):
        body = builder.add_body(xform=wp.transform(wp.vec3(index * 0.5, 0.0, 0.1), wp.quat_identity()))
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    return builder.finalize(device="cpu")


def _runtime(model):
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=128)
    solver = newton.solvers.SolverFeatherPGS(
        model,
        enable_sleeping=True,
        friction_anchor_beta=0.0,
        solve_mode="split",
        sleep_quiet_time=0.05,
        iterations=16,
    )
    state_in, state_out = model.state(), model.state()
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    state_in.clear_forces()
    return pipeline, solver, state_in, state_out, model.control()


def _seed_sleep(solver, state, previous_roots):
    sleeping = solver._sleeping
    sleeping.art_awake.zero_()
    sleeping.body_awake.zero_()
    sleeping.previous_root.assign(np.asarray(previous_roots, dtype=np.int32))
    sleeping.quiet_age.fill_(sleeping.quiet_time)
    for destination, source in (
        (sleeping.last_q, state.joint_q),
        (sleeping.last_qd, state.joint_qd),
        (sleeping.last_body_q, state.body_q),
        (sleeping.last_body_qd, state.body_qd),
    ):
        wp.copy(destination, source)
    sleeping.valid.fill_(1)


class TestSleepWorkSkipping(unittest.TestCase):
    def test_cuda_ant_joint_limits_skip_and_restore(self):
        """Skip a sleeping ant's limit rows and restore them on captured force wake."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA required")
        builder = newton.ModelBuilder()
        builder.add_mjcf(
            str(Path(newton.__file__).parent / "examples/assets/nv_ant.xml"),
            ignore_names=["floor", "ground"],
            collapse_fixed_joints=False,
            parse_sites=False,
        )
        builder.joint_target_ke[:] = [0.0] * len(builder.joint_target_ke)
        builder.joint_target_kd[:] = [0.0] * len(builder.joint_target_kd)
        builder.joint_act[:] = [0.0] * len(builder.joint_act)
        builder.add_ground_plane()
        model = builder.finalize(device="cuda:0")
        for skip in (False, True):
            with self.subTest(skip_constraints=skip):
                solver = newton.solvers.SolverFeatherPGS(
                    model,
                    enable_sleeping=True,
                    sleep_skip_constraints=skip,
                    sleep_linear_threshold=0.05,
                    sleep_angular_threshold=0.15,
                    sleep_quiet_time=0.5,
                    enable_joint_limits=True,
                    solve_mode="matrix_free",
                    iterations=8,
                    friction_anchor_beta=0.0,
                    articulation_constraint_max=256,
                    free_body_constraint_max=256,
                )
                original_indices = solver._joint_limit_q_index.numpy().copy()
                self.assertEqual(int(np.count_nonzero(original_indices >= 0)), 8)
                self.assertTrue(solver._joint_limit_warp_kernels)
                pipeline = newton.CollisionPipeline(model, rigid_contact_max=128)
                contacts = pipeline.contacts()
                states = [model.state(), model.state()]
                control = model.control()
                for _ in range(2):
                    pipeline.collide(states[0], contacts)
                    solver.step(states[0], states[1], control, contacts, 1.0 / 240.0)
                    states.reverse()
                with wp.ScopedCapture(device=model.device) as capture:
                    for _ in range(2):
                        pipeline.collide(states[0], contacts)
                        solver.step(states[0], states[1], control, contacts, 1.0 / 240.0)
                        states.reverse()
                for _ in range(1200):
                    wp.capture_launch(capture.graph)
                np.testing.assert_array_equal(solver._sleeping.art_awake.numpy(), [0])
                if skip:
                    np.testing.assert_array_equal(solver._constraint_count.numpy(), [0])
                    np.testing.assert_array_equal(solver._mf_constraint_count.numpy(), [0])
                    self.assertTrue(np.all(solver._sleeping.limit_q_index.numpy() < 0))
                else:
                    self.assertGreaterEqual(int(solver._constraint_count.numpy().sum()), 16)
                force = np.zeros((model.body_count, 6), dtype=np.float32)
                force[0, 0] = 100.0
                for state in states:
                    state.body_f.assign(force)

                wp.capture_launch(capture.graph)

                np.testing.assert_array_equal(solver._sleeping.art_awake.numpy(), [1])
                self.assertGreaterEqual(int(solver._constraint_count.numpy().sum()), 16)
                np.testing.assert_array_equal(solver._sleeping.limit_q_index.numpy(), original_indices)
                self.assertTrue(np.isfinite(states[0].body_q.numpy()).all())
                self.assertTrue(np.isfinite(states[0].body_qd.numpy()).all())
                self.assertFalse(solver.constraint_overflow.numpy().any())

    def test_settled_rows_drop_without_changing_contacts(self):
        """Remove solver rows while retaining complete collision contact inputs."""
        scene = _Scene()
        scene.settle()
        scene.pipeline.collide(scene.states[0], scene.contacts)
        before = _contact_inputs(scene.contacts)
        self.assertGreater(int(before["count"][0]), 0)
        pose = scene.states[0].body_q.numpy().copy()

        scene.step(collide=False)

        np.testing.assert_array_equal(scene.row_counts(), [0, 0])
        np.testing.assert_array_equal(scene.solver._sleeping.body_awake.numpy(), [0, 0])
        np.testing.assert_array_equal(scene.states[0].body_q.numpy(), pose)
        for name, values in before.items():
            np.testing.assert_array_equal(_contact_inputs(scene.contacts)[name], values, err_msg=name)

    def test_explicit_full_solve_retains_rows(self):
        """Compare settled row counts against explicit contact-skipping disablement."""
        full, skipped = _Scene(skip=False), _Scene()
        for scene in (full, skipped):
            scene.settle()
            scene.step()
            np.testing.assert_array_equal(scene.solver._sleeping.body_awake.numpy(), [0, 0])
            self.assertGreater(int(scene.contacts.rigid_contact_count.numpy()[0]), 0)
        self.assertGreater(int(full.row_counts().sum()), 0)
        np.testing.assert_array_equal(skipped.row_counts(), [0, 0])
        np.testing.assert_allclose(full.states[0].body_q.numpy(), skipped.states[0].body_q.numpy(), atol=1.0e-6, rtol=0)

    def test_force_restores_only_independent_ground_island_rows(self):
        """Restore forced-body rows while another ground-supported island sleeps."""
        scene = _Scene()
        scene.step()
        active_rows = int(scene.row_counts().sum())
        self.assertGreater(active_rows, 0)
        scene.settle()
        self.assertNotEqual(*scene.solver._sleeping.body_island.numpy())
        np.testing.assert_array_equal(scene.row_counts(), [0, 0])
        resting_pose = scene.states[0].body_q.numpy()[1].copy()
        force = np.zeros((2, 6), dtype=np.float32)
        force[0, 0] = 100.0
        scene.states[0].body_f.assign(force)

        scene.step(clear=False)

        rows = int(scene.row_counts().sum())
        self.assertGreater(rows, 0)
        self.assertLess(rows, active_rows)
        np.testing.assert_array_equal(scene.solver._sleeping.body_awake.numpy(), [1, 0])
        self.assertGreater(float(scene.states[0].body_qd.numpy()[0, 0]), 0.0)
        np.testing.assert_array_equal(scene.states[0].body_q.numpy()[1], resting_pose)

    def test_support_loss_wakes_before_skipped_step(self):
        """Let a newly unsupported sleeper fall during the same small timestep."""
        scene = _Scene()
        scene.settle()
        scene.pipeline.collide(scene.states[0], scene.contacts)
        count = int(scene.contacts.rigid_contact_count.numpy()[0])
        shape_body = scene.model.shape_body.numpy()
        body0 = shape_body[scene.contacts.rigid_contact_shape0.numpy()[:count]]
        body1 = shape_body[scene.contacts.rigid_contact_shape1.numpy()[:count]]
        keep = np.flatnonzero(((body0 == 0) & (body1 < 0)) | ((body1 == 0) & (body0 < 0)))
        self.assertGreater(len(keep), 0)
        self.assertTrue(np.any((body0 == 1) | (body1 == 1)))
        for suffix in _CONTACT_FIELDS:
            array = getattr(scene.contacts, "rigid_contact_" + suffix)
            values = array.numpy()
            values[: len(keep)] = values[keep].copy()
            array.assign(values)
        scene.contacts.rigid_contact_count.assign(np.array([len(keep)], dtype=np.int32))
        before = float(scene.states[0].body_q.numpy()[1, 2])

        scene.step(collide=False, dt=0.001)

        np.testing.assert_array_equal(scene.solver._sleeping.body_awake.numpy(), [0, 1])
        np.testing.assert_array_equal(scene.row_counts(), [0, 0])
        self.assertLess(float(scene.states[0].body_q.numpy()[1, 2]), before)
        velocity = float(scene.states[0].body_qd.numpy()[1, 2])
        self.assertLess(velocity, 0.0)
        self.assertLess(abs(velocity), scene.solver._sleeping.linear_threshold)

    def test_fixed_table_supports_sleep_and_independent_force_wake(self):
        """Sleep both table-supported boxes and restore only the forced box's rows."""
        scene = _Scene(fixed_table=True)
        scene.step()
        active_rows = int(scene.row_counts().sum())
        self.assertGreater(active_rows, 0)
        scene.settle()
        islands = scene.solver._sleeping.body_island.numpy()[scene.bodies]
        self.assertNotEqual(*islands)
        np.testing.assert_array_equal(scene.solver._sleeping.root_supported.numpy()[islands], [1, 1])
        np.testing.assert_array_equal(scene.row_counts(), [0, 0])
        resting_pose = scene.states[0].body_q.numpy()[scene.bodies[1]].copy()
        force = np.zeros((scene.model.body_count, 6), dtype=np.float32)
        force[scene.bodies[0], 0] = 100.0
        scene.states[0].body_f.assign(force)

        scene.step(clear=False)

        np.testing.assert_array_equal(scene.solver._sleeping.body_awake.numpy()[scene.bodies], [1, 0])
        rows = int(scene.row_counts().sum())
        self.assertGreater(rows, 0)
        self.assertLess(rows, active_rows)
        self.assertGreater(float(scene.states[0].body_qd.numpy()[scene.bodies[0], 0]), 0.0)
        np.testing.assert_array_equal(scene.states[0].body_q.numpy()[scene.bodies[1]], resting_pose)

    def test_cuda_graph_replay_restores_rows_on_force(self):
        """Restore matrix-free contact rows on a captured sleeping-to-awake transition."""
        if not wp.is_cuda_available():
            self.skipTest("CUDA required")
        scene = _Scene(device="cuda:0")
        scene.step()
        self.assertGreater(int(scene.solver._mf_constraint_count.numpy().sum()), 0)
        scene.step()
        with wp.ScopedCapture(device=scene.model.device) as capture:
            for _ in range(2):
                scene.step(clear=False)
        for _ in range(100):
            wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(scene.solver._sleeping.body_awake.numpy(), [0, 0])
        np.testing.assert_array_equal(scene.row_counts(), [0, 0])
        self.assertGreater(int(scene.contacts.rigid_contact_count.numpy()[0]), 0)
        force = np.zeros((2, 6), dtype=np.float32)
        force[0, 0] = 100.0
        for state in scene.states:
            state.body_f.assign(force)

        wp.capture_launch(capture.graph)

        np.testing.assert_array_equal(scene.solver._sleeping.body_awake.numpy(), [1, 0])
        self.assertGreater(int(scene.solver._mf_constraint_count.numpy().sum()), 0)
        self.assertGreater(float(scene.states[0].body_qd.numpy()[0, 0]), 0.0)
        self.assertTrue(np.isfinite(scene.states[0].body_q.numpy()).all())
        self.assertTrue(np.isfinite(scene.states[0].body_qd.numpy()).all())
        self.assertFalse(scene.solver.constraint_overflow.numpy().any())

    def test_nonfinite_prescribed_velocity_wakes_before_allocation(self):
        """Wake supported sleepers when their fixed table has invalid velocity."""
        scene = _Scene(fixed_table=True)
        scene.settle()
        scene.pipeline.collide(scene.states[0], scene.contacts)
        self.assertGreater(int(scene.contacts.rigid_contact_count.numpy()[0]), 0)
        pose = scene.states[0].body_q.numpy().copy()
        velocity = scene.states[0].body_qd.numpy()
        velocity[0, 0] = np.nan
        scene.states[0].body_qd.assign(velocity)

        scene.solver._sleeping.begin(scene.states[0], scene.control, scene.contacts)

        np.testing.assert_array_equal(scene.solver._sleeping.art_awake.numpy()[scene.bodies], [1, 1])
        np.testing.assert_array_equal(scene.states[0].body_q.numpy(), pose)
        self.assertTrue(np.isnan(scene.states[0].body_qd.numpy()[0, 0]))


class _Scene:
    def __init__(self, *, skip=True, device="cpu", fixed_table=False):
        builder = newton.ModelBuilder()
        if fixed_table:
            table = builder.add_link()
            root = builder.add_joint_fixed(parent=-1, child=table)
            builder.add_articulation([root])
            builder.add_shape_box(table, hx=1.0, hy=0.5, hz=0.1)
        else:
            builder.add_ground_plane()
        self.bodies = []
        for x in (0.0, 0.5):
            body = builder.add_body(
                xform=wp.transform(wp.vec3(x, 0.0, 0.2 if fixed_table else 0.1), wp.quat_identity())
            )
            builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
            self.bodies.append(body)
        self.model = builder.finalize(device=device)
        self.pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=64)
        self.contacts = self.pipeline.contacts()
        options = {} if skip else {"sleep_skip_constraints": False}
        self.solver = newton.solvers.SolverFeatherPGS(
            self.model,
            enable_sleeping=True,
            sleep_quiet_time=0.05,
            friction_anchor_beta=0.0,
            iterations=16,
            solve_mode="matrix_free" if self.model.device.is_cuda else "split",
            **options,
        )
        self.states = [self.model.state(), self.model.state()]
        self.control = self.model.control()

    def step(self, *, clear=True, collide=True, dt=0.005):
        if clear:
            self.states[0].clear_forces()
        if collide:
            self.pipeline.collide(self.states[0], self.contacts)
        self.solver.step(self.states[0], self.states[1], self.control, self.contacts, dt)
        self.states.reverse()

    def settle(self):
        for _ in range(120):
            self.step()
        np.testing.assert_array_equal(self.solver._sleeping.body_awake.numpy()[self.bodies], [0, 0])
        if self.solver.constraint_overflow.numpy().any():
            raise AssertionError("Contact capacity exceeded")

    def row_counts(self):
        return np.array([self.solver._constraint_count.numpy().sum(), self.solver._mf_constraint_count.numpy().sum()])


_CONTACT_FIELDS = ("shape0", "shape1", "point0", "point1", "offset0", "offset1", "normal", "margin0", "margin1", "tids")


def _contact_inputs(contacts):
    values = {name: getattr(contacts, "rigid_contact_" + name).numpy().copy() for name in _CONTACT_FIELDS}
    values["count"] = contacts.rigid_contact_count.numpy().copy()
    return values


def test_ant_joint_limits_skip_and_restore(test, device):
    """Skip a sleeping ant's limit rows and restore them on captured force wake."""
    builder = newton.ModelBuilder()
    builder.add_mjcf(
        str(Path(newton.__file__).parent / "examples/assets/nv_ant.xml"),
        ignore_names=["floor", "ground"],
        collapse_fixed_joints=False,
        parse_sites=False,
    )
    builder.joint_target_ke[:] = [0.0] * len(builder.joint_target_ke)
    builder.joint_target_kd[:] = [0.0] * len(builder.joint_target_kd)
    builder.joint_act[:] = [0.0] * len(builder.joint_act)
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    for skip in (False, True):
        with test.subTest(skip_constraints=skip):
            solver = newton.solvers.SolverFeatherPGS(
                model,
                solve_mode="matrix_free",
                enable_sleeping=True,
                sleep_skip_constraints=skip,
                sleep_linear_threshold=0.05,
                sleep_angular_threshold=0.15,
                sleep_quiet_time=0.5,
                enable_joint_limits=True,
                iterations=8,
                friction_anchor_beta=0.0,
                articulation_constraint_max=256,
                free_body_constraint_max=256,
            )
            original_indices = solver._joint_limit_q_index.numpy().copy()
            test.assertEqual(int(np.count_nonzero(original_indices >= 0)), 8)
            test.assertTrue(solver._joint_limit_warp_kernels)
            pipeline = newton.CollisionPipeline(model, rigid_contact_max=128)
            contacts = pipeline.contacts()
            states = [model.state(), model.state()]
            control = model.control()
            for _ in range(2):
                pipeline.collide(states[0], contacts)
                solver.step(states[0], states[1], control, contacts, 1.0 / 240.0)
                states.reverse()
            with wp.ScopedCapture(device=model.device) as capture:
                for _ in range(2):
                    pipeline.collide(states[0], contacts)
                    solver.step(states[0], states[1], control, contacts, 1.0 / 240.0)
                    states.reverse()
            for _ in range(1200):
                wp.capture_launch(capture.graph)
            np.testing.assert_array_equal(solver._sleeping.art_awake.numpy(), [0])
            if skip:
                np.testing.assert_array_equal(solver._constraint_count.numpy(), [0])
                np.testing.assert_array_equal(solver._mf_constraint_count.numpy(), [0])
                test.assertTrue(np.all(solver._sleeping.limit_q_index.numpy() < 0))
            else:
                test.assertGreaterEqual(int(solver._constraint_count.numpy().sum()), 16)
            force = np.zeros((model.body_count, 6), dtype=np.float32)
            force[0, 0] = 100.0
            for state in states:
                state.body_f.assign(force)

            wp.capture_launch(capture.graph)

            np.testing.assert_array_equal(solver._sleeping.art_awake.numpy(), [1])
            test.assertGreaterEqual(int(solver._constraint_count.numpy().sum()), 16)
            np.testing.assert_array_equal(solver._sleeping.limit_q_index.numpy(), original_indices)
            test.assertTrue(np.isfinite(states[0].body_q.numpy()).all())
            test.assertTrue(np.isfinite(states[0].body_qd.numpy()).all())
            test.assertFalse(solver.constraint_overflow.numpy().any())


def test_graph_replay_restores_rows_on_force(test, device):
    """Restore matrix-free contact rows on a captured sleeping-to-awake transition."""
    scene = _Scene_sleep_work_skipping_upstream(device)
    scene.step()
    test.assertGreater(int(scene.solver._mf_constraint_count.numpy().sum()), 0)
    scene.step()
    with wp.ScopedCapture(device=scene.model.device) as capture:
        for _ in range(2):
            scene.step(clear=False)
    for _ in range(100):
        wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(scene.solver._sleeping.body_awake.numpy(), [0, 0])
    np.testing.assert_array_equal(scene.row_counts(), [0, 0])
    test.assertGreater(int(scene.contacts.rigid_contact_count.numpy()[0]), 0)
    force = np.zeros((2, 6), dtype=np.float32)
    force[0, 0] = 100.0
    for state in scene.states:
        state.body_f.assign(force)

    wp.capture_launch(capture.graph)

    np.testing.assert_array_equal(scene.solver._sleeping.body_awake.numpy(), [1, 0])
    test.assertGreater(int(scene.solver._mf_constraint_count.numpy().sum()), 0)
    test.assertGreater(float(scene.states[0].body_qd.numpy()[0, 0]), 0.0)
    test.assertTrue(np.isfinite(scene.states[0].body_q.numpy()).all())
    test.assertTrue(np.isfinite(scene.states[0].body_qd.numpy()).all())
    test.assertFalse(scene.solver.constraint_overflow.numpy().any())


class _Scene_sleep_work_skipping_upstream:
    def __init__(self, device, *, skip=True, fixed_table=False):
        builder = newton.ModelBuilder()
        if fixed_table:
            table = builder.add_link()
            root = builder.add_joint_fixed(parent=-1, child=table)
            builder.add_articulation([root])
            builder.add_shape_box(table, hx=1.0, hy=0.5, hz=0.1)
        else:
            builder.add_ground_plane()
        self.bodies = []
        for x in (0.0, 0.5):
            body = builder.add_body(
                xform=wp.transform(wp.vec3(x, 0.0, 0.2 if fixed_table else 0.1), wp.quat_identity())
            )
            builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
            self.bodies.append(body)
        self.model = builder.finalize(device=device)
        self.pipeline = newton.CollisionPipeline(self.model, rigid_contact_max=64)
        self.contacts = self.pipeline.contacts()
        options = {} if skip else {"sleep_skip_constraints": False}
        self.solver = newton.solvers.SolverFeatherPGS(
            self.model,
            solve_mode="matrix_free",
            enable_sleeping=True,
            sleep_quiet_time=0.05,
            friction_anchor_beta=0.0,
            iterations=16,
            **options,
        )
        self.states = [self.model.state(), self.model.state()]
        self.control = self.model.control()

    def step(self, *, clear=True, collide=True, dt=0.005):
        if clear:
            self.states[0].clear_forces()
        if collide:
            self.pipeline.collide(self.states[0], self.contacts)
        self.solver.step(self.states[0], self.states[1], self.control, self.contacts, dt)
        self.states.reverse()

    def settle(self):
        for _ in range(120):
            self.step()
        np.testing.assert_array_equal(self.solver._sleeping.body_awake.numpy()[self.bodies], [0, 0])
        if self.solver.constraint_overflow.numpy().any():
            raise AssertionError("Contact capacity exceeded")

    def row_counts(self):
        return np.array([self.solver._constraint_count.numpy().sum(), self.solver._mf_constraint_count.numpy().sum()])


class TestFeatherPGSSleepWorkSkipping(unittest.TestCase):
    pass


for _name in (
    "test_ant_joint_limits_skip_and_restore",
    "test_graph_replay_restores_rows_on_force",
):
    add_function_test(TestFeatherPGSSleepWorkSkipping, _name, globals()[_name], devices=devices)


if __name__ == "__main__":
    unittest.main()
