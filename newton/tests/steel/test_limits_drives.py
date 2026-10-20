# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Joint limits, velocity limits, joint drives and joint friction of SolverSteel."""

from __future__ import annotations

import inspect
import itertools
import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton._src.core.types import MAXVAL
from newton._src.sim.enums import BodyFlags, JointType
from newton._src.solvers.steel.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    PGS_CONSTRAINT_TYPE_JOINT_TARGET,
    PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
    ROW_SLOTS_ATOMIC,
    allocate_joint_velocity_limit_slots,
    allocate_rigid_velocity_limit_slots,
    build_joint_limit_rows_for_size,
    eval_rigid_tau,
    eval_rigid_tau_add,
    eval_rigid_tau_and_augmented_drives,
    prepare_augmented_joint_drives,
    prescale_joint_velocity_limits,
)
from newton._src.solvers.steel.solver_steel import (
    _get_joint_limit_warp_kernel,
    _prepare_augmented_joint_drives_by_dof,
    _use_resident_mfgs_metadata,
)
from newton.solvers import SolverSteel
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices


def _built_rows(q: float, *, gap: float, lower: float = -1.0, upper: float = 1.0):
    device = "cpu"
    max_constraints = 8
    world_slot_counter = wp.zeros((1,), dtype=wp.int32, device=device)
    J_group = wp.zeros((1, max_constraints, 1), dtype=wp.float32, device=device)
    world_row_type = wp.zeros((1, max_constraints), dtype=wp.int32, device=device)
    world_row_parent = wp.zeros((1, max_constraints), dtype=wp.int32, device=device)
    world_row_mu = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)
    world_row_beta = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)
    world_row_cfm = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)
    world_phi = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)
    world_target_velocity = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)

    wp.launch(
        build_joint_limit_rows_for_size,
        dim=1,
        inputs=[
            wp.array([0, 1], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([int(JointType.REVOLUTE)], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([[0, 1]], dtype=wp.int32, device=device),
            wp.array([lower], dtype=wp.float32, device=device),
            wp.array([upper], dtype=wp.float32, device=device),
            wp.array([q], dtype=wp.float32, device=device),
            gap,
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            max_constraints,
            0.2,
            0.0,
        ],
        outputs=[
            world_slot_counter,
            J_group,
            world_row_type,
            world_row_parent,
            world_row_mu,
            world_row_beta,
            world_row_cfm,
            world_phi,
            world_target_velocity,
            ROW_SLOTS_ATOMIC,
            wp.zeros(1, dtype=wp.int32, device=device),
        ],
        device=device,
    )
    wp.synchronize()
    count = int(world_slot_counter.numpy()[0])
    return J_group.numpy()[0, :count, 0].tolist(), world_phi.numpy()[0, :count].tolist()


def _make_phase_layout_run(
    device,
    solve_mode,
    *,
    iterations=0,
    row_schedule="interleaved",
    response="immediate",
    enable_joint_velocity_limits=True,
):
    """Build a deterministic scene containing every phase-bounded row family."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverSteel.register_custom_attributes(builder)
    builder.default_shape_cfg.density = 1000.0
    builder.default_shape_cfg.mu = 0.5
    builder.default_shape_cfg.margin = 0.0
    builder.default_shape_cfg.gap = 0.0

    link = builder.add_link()
    builder.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1)
    joint = builder.add_joint_revolute(
        parent=-1,
        child=link,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(-1.0, 0.0, 0.05), wp.quat_identity()),
        limit_lower=-0.1,
        limit_upper=0.1,
    )
    builder.add_articulation([joint])
    builder.joint_q[0] = 0.2

    free_body = builder.add_body(
        xform=wp.transform(wp.vec3(1.0, 0.0, 0.05), wp.quat_identity()),
        custom_attributes={
            "steel:rigid_body_max_linear_velocity": 1.0,
            "steel:rigid_body_max_angular_velocity": 1.0,
        },
    )
    builder.add_shape_box(free_body, hx=0.1, hy=0.1, hz=0.1)
    builder.add_ground_plane()

    model = builder.finalize(device=device)
    velocity_limits = np.full(model.joint_dof_count, np.inf, dtype=np.float32)
    velocity_limits[0] = 0.25
    model.joint_velocity_limit.assign(velocity_limits)
    solver = SolverSteel(
        model,
        solve_mode=solve_mode,
        row_schedule=row_schedule,
        articulated_contact_response=response,
        enable_joint_limits=True,
        enable_joint_velocity_limits=enable_joint_velocity_limits,
        velocity_limit_activation_fraction=0.5,
        articulation_constraint_max=64,
        free_body_constraint_max=64,
        iterations=iterations,
        contact_warmstart=False,
    )
    state_0, state_1 = model.state(), model.state()
    joint_qd = state_0.joint_qd.numpy()
    joint_qd[0] = 0.5
    free_joint = int(np.flatnonzero(model.joint_type.numpy() == int(JointType.FREE))[0])
    free_dof = int(model.joint_qd_start.numpy()[free_joint])
    joint_qd[free_dof] = 1.5
    joint_qd[free_dof + 3] = 1.5
    state_0.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    pipeline = newton.CollisionPipeline(model)
    return model, solver, state_0, state_1, model.control(), pipeline, pipeline.contacts()


def _step_once(run):
    """Build all constraint rows for one deterministic step."""
    _model, solver, state_0, state_1, control, pipeline, contacts = run
    state_0.clear_forces()
    pipeline.collide(state_0, contacts)
    solver.step(state_0, state_1, control, contacts, 1.0 / 60.0)
    return solver


def _assert_dense_phase_layout(
    test_case: unittest.TestCase, solver: SolverSteel, *, expect_joint_velocity_limits: bool = True
):
    """Assert the three dense row-family partitions are ordered and populated."""
    dense_count = int(solver._constraint_count.numpy()[0])
    dense_types = solver._row_type.numpy()[0, :dense_count]
    phase_zero_end, phase_one_end = solver._dense_phase_bounds.numpy()[0]
    limit_rows = np.flatnonzero(dense_types == PGS_CONSTRAINT_TYPE_JOINT_LIMIT)
    velocity_limit_rows = np.flatnonzero(dense_types == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)
    contact_rows = np.flatnonzero(dense_types == PGS_CONSTRAINT_TYPE_CONTACT)
    friction_rows = np.flatnonzero(dense_types == PGS_CONSTRAINT_TYPE_FRICTION)

    test_case.assertGreater(limit_rows.size, 0, "expected an active joint position-limit row")
    if expect_joint_velocity_limits:
        test_case.assertGreater(velocity_limit_rows.size, 0, "expected active joint velocity-limit rows")
    else:
        test_case.assertEqual(velocity_limit_rows.size, 0)
    test_case.assertGreater(contact_rows.size, 0, "expected dense contact rows")
    test_case.assertGreater(friction_rows.size, 0, "expected dense friction rows")
    test_case.assertTrue(np.all(limit_rows < phase_zero_end), (limit_rows, phase_zero_end))
    if expect_joint_velocity_limits:
        test_case.assertTrue(
            np.all((phase_zero_end <= velocity_limit_rows) & (velocity_limit_rows < phase_one_end)),
            (velocity_limit_rows, phase_zero_end, phase_one_end),
        )
    test_case.assertTrue(np.all(contact_rows >= phase_one_end), (contact_rows, phase_one_end))
    test_case.assertTrue(np.all(friction_rows >= phase_one_end), (friction_rows, phase_one_end))


def _assert_mf_phase_layout(test_case: unittest.TestCase, solver: SolverSteel):
    """Assert MF contacts precede the populated rigid velocity-limit tail."""
    mf_count = int(solver._mf_constraint_count.numpy()[0])
    mf_types = solver._mf_row_type.numpy()[0, :mf_count]
    mf_contact_end = int(solver._mf_contact_rows_end.numpy()[0])
    contact_rows = np.flatnonzero(mf_types == PGS_CONSTRAINT_TYPE_CONTACT)
    friction_rows = np.flatnonzero(mf_types == PGS_CONSTRAINT_TYPE_FRICTION)
    velocity_limit_rows = np.flatnonzero(mf_types == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)

    test_case.assertGreater(contact_rows.size, 0, "expected matrix-free contact rows")
    test_case.assertGreater(friction_rows.size, 0, "expected matrix-free friction rows")
    test_case.assertGreater(velocity_limit_rows.size, 0, "expected active rigid-body velocity-limit rows")
    test_case.assertTrue(np.all(contact_rows < mf_contact_end), (contact_rows, mf_contact_end))
    test_case.assertTrue(np.all(friction_rows < mf_contact_end), (friction_rows, mf_contact_end))
    test_case.assertTrue(np.all(velocity_limit_rows >= mf_contact_end), (velocity_limit_rows, mf_contact_end))
    test_case.assertLess(mf_contact_end, mf_count)


def _run_joint_limit_trajectory(use_warp_builder: bool):
    """Run a short PhysX-grasp propagation trajectory with one row builder."""
    _model, solver, state_in, state_out, control, pipeline, contacts = _make_phase_layout_run(
        "cuda:0",
        "matrix_free",
        iterations=8,
        row_schedule="phased",
        response="propagation",
    )
    if not use_warp_builder:
        solver._joint_limit_warp_kernels.clear()

    samples = []
    for _ in range(4):
        state_in.clear_forces()
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, 1.0 / 240.0)
        dense_count = int(solver._constraint_count.numpy()[0])
        mf_count = int(solver._mf_constraint_count.numpy()[0])
        propagation_count = int(solver._propagation_constraint_count.numpy()[0])
        samples.append(
            {
                "counts": (dense_count, mf_count, propagation_count),
                "bounds": solver._dense_phase_bounds.numpy().copy(),
                "row_type": solver._row_type.numpy()[0, :dense_count].copy(),
                "mf_row_type": solver._mf_row_type.numpy()[0, :mf_count].copy(),
                "propagation_row_type": solver._propagation_row_type.numpy()[0, :propagation_count].copy(),
                "impulses": solver._impulses.numpy()[0, :dense_count].copy(),
                "mf_impulses": solver._mf_impulses.numpy()[0, :mf_count].copy(),
                "propagation_impulses": solver._propagation_impulses.numpy()[0, :propagation_count].copy(),
                "joint_q": state_out.joint_q.numpy().copy(),
                "joint_qd": state_out.joint_qd.numpy().copy(),
            }
        )
        state_in, state_out = state_out, state_in
    return solver, samples


class TestSteelJointLimitActivationGap(unittest.TestCase):
    def test_dense_row_families_respect_phase_bounds(self):
        """Keep active position limits before the CPU dense phase-0 bound."""
        solver = _step_once(_make_phase_layout_run("cpu", "split", enable_joint_velocity_limits=False))
        _assert_dense_phase_layout(self, solver, expect_joint_velocity_limits=False)

    @unittest.skipUnless(wp.is_cuda_available(), "matrix-free row-family layout requires CUDA")
    def test_combined_row_families_respect_phase_bounds(self):
        """Keep dense and matrix-free row families inside their phase ranges."""
        solver = _step_once(_make_phase_layout_run("cuda:0", "matrix_free"))
        _assert_dense_phase_layout(self, solver)
        _assert_mf_phase_layout(self, solver)

    @unittest.skipUnless(wp.is_cuda_available(), "joint-limit trajectory parity requires CUDA")
    def test_warp_builder_matches_physx_grasp_propagation_trajectory(self):
        """Match scalar joint-limit assembly under PhysX-grasp propagation."""
        reference_solver, reference = _run_joint_limit_trajectory(use_warp_builder=False)
        warp_solver, actual = _run_joint_limit_trajectory(use_warp_builder=True)

        self.assertFalse(reference_solver._joint_limit_warp_kernels)
        self.assertTrue(any(kernel is not None for kernel in warp_solver._joint_limit_warp_kernels.values()))
        self.assertEqual(reference_solver._row_schedule, "phased")
        self.assertEqual(reference_solver._articulated_contact_response, "propagation")
        self.assertEqual(len(reference), len(actual))
        for step, (expected, observed) in enumerate(zip(reference, actual, strict=True)):
            self.assertEqual(observed["counts"], expected["counts"], f"row counts differed at step {step}")
            for label in ("bounds", "row_type", "mf_row_type", "propagation_row_type"):
                np.testing.assert_array_equal(
                    observed[label], expected[label], err_msg=f"{label} differed at step {step}"
                )
            for label in ("impulses", "mf_impulses", "propagation_impulses", "joint_q", "joint_qd"):
                np.testing.assert_allclose(
                    observed[label],
                    expected[label],
                    rtol=1.0e-5,
                    atol=2.0e-6,
                    err_msg=f"{label} differed at step {step}",
                )

        limit_rows = np.flatnonzero(reference[0]["row_type"] == PGS_CONSTRAINT_TYPE_JOINT_LIMIT)
        self.assertGreater(limit_rows.size, 0)
        self.assertGreater(float(np.max(np.abs(reference[0]["impulses"][limit_rows]))), 0.0)

    def test_solver_rejects_invalid_joint_limit_activation_gap(self):
        model = newton.ModelBuilder().finalize()

        for value in (-0.1, float("nan")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "joint_limit_activation_gap"):
                    SolverSteel(model, joint_limit_activation_gap=value)

    def test_finite_gap_builds_only_near_limit_rows(self):
        self.assertEqual(_built_rows(0.0, gap=0.2), ([], []))

        jacobian, phi = _built_rows(-0.85, gap=0.2)
        self.assertEqual(jacobian, [1.0])
        self.assertAlmostEqual(phi[0], 0.15, places=6)

        jacobian, phi = _built_rows(0.85, gap=0.2)
        self.assertEqual(jacobian, [-1.0])
        self.assertAlmostEqual(phi[0], 0.15, places=6)

    def test_finite_gap_does_not_activate_unlimited_sentinel_limits(self):
        self.assertEqual(_built_rows(0.0, gap=0.2, lower=-MAXVAL, upper=MAXVAL), ([], []))

    def test_infinite_gap_preserves_historical_always_allocate_behavior(self):
        jacobian, phi = _built_rows(0.0, gap=float("inf"))

        self.assertEqual(jacobian, [1.0, -1.0])
        self.assertEqual(phi, [1.0, 1.0])


def _make_layout_scene(
    device, *, enable_joint_velocity_limits=True, solve_mode="matrix_free", response="immediate", iterations=0
):
    """Build a scene that produces every row family: limits, velocity limits and contacts."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    SolverSteel.register_custom_attributes(builder)
    builder.default_shape_cfg.density = 1000.0
    builder.default_shape_cfg.mu = 0.5
    builder.default_shape_cfg.margin = 0.0
    builder.default_shape_cfg.gap = 0.0

    link = builder.add_link()
    builder.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1)
    joint = builder.add_joint_revolute(
        parent=-1,
        child=link,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(-1.0, 0.0, 0.05), wp.quat_identity()),
        limit_lower=-0.1,
        limit_upper=0.1,
    )
    builder.add_articulation([joint])
    builder.joint_q[0] = 0.2

    free_body = builder.add_body(
        xform=wp.transform(wp.vec3(1.0, 0.0, 0.05), wp.quat_identity()),
        custom_attributes={
            "steel:rigid_body_max_linear_velocity": 1.0,
            "steel:rigid_body_max_angular_velocity": 1.0,
        },
    )
    builder.add_shape_box(free_body, hx=0.1, hy=0.1, hz=0.1)
    builder.add_ground_plane()

    model = builder.finalize(device=device)
    velocity_limits = np.full(model.joint_dof_count, np.inf, dtype=np.float32)
    velocity_limits[0] = 0.25
    model.joint_velocity_limit.assign(velocity_limits)
    solver = SolverSteel(
        model,
        solve_mode=solve_mode,
        enable_joint_limits=True,
        enable_joint_velocity_limits=enable_joint_velocity_limits,
        velocity_limit_activation_fraction=0.5,
        articulation_constraint_max=64,
        free_body_constraint_max=64,
        iterations=iterations,
        articulated_contact_response=response,
    )
    state_0, state_1 = model.state(), model.state()
    joint_qd = state_0.joint_qd.numpy()
    joint_qd[0] = 0.5
    free_dof = int(model.joint_qd_start.numpy()[1])
    joint_qd[free_dof] = 1.5
    joint_qd[free_dof + 3] = 1.5
    state_0.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    pipeline = newton.CollisionPipeline(model)
    return model, solver, state_0, state_1, pipeline


def _make_layout_run(device, *, enable_joint_velocity_limits=True, solve_mode="matrix_free"):
    """Build the layout scene and its rows for one step."""
    model, solver, state_0, state_1, pipeline = _make_layout_scene(
        device, enable_joint_velocity_limits=enable_joint_velocity_limits, solve_mode=solve_mode
    )
    contacts = pipeline.contacts()
    pipeline.collide(state_0, contacts)
    solver.step(state_0, state_1, model.control(), contacts, 1.0 / 60.0)
    return solver


def test_combined_row_families_follow_the_documented_layout(test, device):
    """Lay out dense rows as limits, velocity limits, contacts; free-body rows as contacts, velocity limits."""
    solver = _make_layout_run(device)
    dense_count = int(solver._constraint_count.numpy()[0])
    dense_types = solver._row_type.numpy()[0, :dense_count]
    limit_rows = np.flatnonzero(dense_types == PGS_CONSTRAINT_TYPE_JOINT_LIMIT)
    velocity_limit_rows = np.flatnonzero(dense_types == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)
    contact_rows = np.flatnonzero(
        (dense_types == PGS_CONSTRAINT_TYPE_CONTACT) | (dense_types == PGS_CONSTRAINT_TYPE_FRICTION)
    )
    test.assertGreater(limit_rows.size, 0)
    test.assertGreater(velocity_limit_rows.size, 0)
    test.assertGreater(contact_rows.size, 0)
    test.assertLess(limit_rows.max(), velocity_limit_rows.min())
    test.assertLess(velocity_limit_rows.max(), contact_rows.min())

    mf_count = int(solver._mf_constraint_count.numpy()[0])
    mf_types = solver._mf_row_type.numpy()[0, :mf_count]
    mf_contact_end = int(solver._mf_contact_rows_end.numpy()[0])
    mf_contacts = np.flatnonzero((mf_types == PGS_CONSTRAINT_TYPE_CONTACT) | (mf_types == PGS_CONSTRAINT_TYPE_FRICTION))
    mf_velocity_limits = np.flatnonzero(mf_types == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)
    test.assertGreater(mf_contacts.size, 0)
    test.assertGreater(mf_velocity_limits.size, 0)
    test.assertTrue(np.all(mf_contacts < mf_contact_end))
    test.assertTrue(np.all(mf_velocity_limits >= mf_contact_end))


def test_velocity_limit_rows_require_the_option(test, device):
    """Build no joint velocity-limit rows unless velocity limits are enabled."""
    solver = _make_layout_run(device, enable_joint_velocity_limits=False)
    dense_count = int(solver._constraint_count.numpy()[0])
    dense_types = solver._row_type.numpy()[0, :dense_count]
    test.assertEqual(int(np.sum(dense_types == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)), 0)
    test.assertGreater(int(np.sum(dense_types == PGS_CONSTRAINT_TYPE_JOINT_LIMIT)), 0)


def _driven_limited_joint_position(device, **solver_kwargs) -> float:
    """Drive a revolute joint limited to ``[-0.3, 0.3]`` toward 1 rad and return its final position."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    link = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.1))
    joint = builder.add_joint_revolute(
        -1, link, axis=newton.Axis.Z, limit_lower=-0.3, limit_upper=0.3, target_ke=500.0, target_kd=10.0
    )
    builder.add_articulation([joint])
    model = builder.finalize(device=device)
    solver = SolverSteel(model, iterations=8, **solver_kwargs)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    control.joint_target_q.fill_(1.0)
    for _ in range(240):
        solver.step(state_0, state_1, control, None, 1.0 / 240.0)
        state_0, state_1 = state_1, state_0
    return float(state_0.joint_q.numpy()[0])


def test_limit_holds_a_driven_joint(test, device, solve_mode="matrix_free"):
    """Hold a joint driven past its upper limit at the limit when joint limits are enabled."""
    position = _driven_limited_joint_position(device, solve_mode=solve_mode, enable_joint_limits=True)
    test.assertAlmostEqual(position, 0.3, delta=2.0e-3)


def test_disabled_joint_limits_are_not_enforced(test, device, solve_mode="matrix_free"):
    """Leave finite joint limits unenforced by default, so the drive reaches its target."""
    test.assertAlmostEqual(_driven_limited_joint_position(device, solve_mode=solve_mode), 1.0, delta=2.0e-3)
    position = _driven_limited_joint_position(device, solve_mode=solve_mode, enable_joint_limits=False)
    test.assertAlmostEqual(position, 1.0, delta=2.0e-3)


def _build_default_limit_chain(device, count: int):
    """Build ``count`` independent revolute links with the builder's default (+/-1e10) limits."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    joints = []
    for _ in range(count):
        link = builder.add_link()
        builder.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1)
        joints.append(builder.add_joint_revolute(-1, link, axis=newton.Axis.Z))
    builder.add_articulation(joints)
    return builder.finalize(device=device)


def test_disabled_joint_limits_build_no_rows(test, device, **mode_kwargs):
    """Build no limit rows and use no capacity for finite limits when joint limits are disabled."""
    model = _build_default_limit_chain(device, 17)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solvers = (
            SolverSteel(model, **mode_kwargs),
            SolverSteel(model, enable_joint_limits=False, **mode_kwargs),
        )
    # The 34 rows of the +/-1e10 defaults would exceed the default capacity of 32 if limits were enabled.
    test.assertFalse([w for w in caught if "joint position limits" in str(w.message)])
    for solver in solvers:
        test.assertFalse(solver._enable_joint_limits)
        solver.step(model.state(), model.state(), model.control(), None, 0.01)
        test.assertEqual(int(solver._constraint_count.numpy()[0]), 0)
        test.assertFalse(bool(solver.constraint_overflow.numpy()[0]))

    # A joint outside its limit gets no row either, whatever the activation gap.
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    link = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3) * 0.1))
    joint = builder.add_joint_revolute(-1, link, axis=newton.Axis.Z, limit_lower=-0.1, limit_upper=0.1)
    builder.add_articulation([joint])
    builder.joint_q[0] = 0.2
    model = builder.finalize(device=device)
    solver = SolverSteel(model, joint_limit_activation_gap=0.5, **mode_kwargs)
    state_0, state_1 = model.state(), model.state()
    solver.step(state_0, state_1, model.control(), None, 0.01)
    test.assertEqual(int(solver._constraint_count.numpy()[0]), 0)
    test.assertAlmostEqual(float(state_1.joint_q.numpy()[0]), 0.2, places=6)


def _run_layout_trajectory(device, response, iterations):
    """Step the layout scene four times; record each step's row families, impulses and joint state."""
    model, solver, state_0, state_1, pipeline = _make_layout_scene(device, response=response, iterations=iterations)
    contacts = pipeline.contacts()
    control = model.control()
    samples = []
    for _ in range(4):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
        dense_count = int(solver._constraint_count.numpy()[0])
        mf_count = int(solver._mf_constraint_count.numpy()[0])
        samples.append(
            {
                "dense_count": dense_count,
                "row_type": solver._row_type.numpy()[0, :dense_count].copy(),
                "impulses": solver._impulses.numpy()[0, :dense_count].copy(),
                "mf_row_type": solver._mf_row_type.numpy()[0, :mf_count].copy(),
                "propagation_count": int(solver._propagation_constraint_count.numpy()[0]),
                "joint_q": state_1.joint_q.numpy().copy(),
                "joint_qd": state_1.joint_qd.numpy().copy(),
            }
        )
        state_0, state_1 = state_1, state_0
    return samples


def test_fused_propagation_matches_phased_trajectory(test, device):
    """Solve every row family in the propagation schedule; the fused kernel matches the phased launches.

    Each propagation iteration solves the dense joint-limit rows, the contact rows, the
    propagation rows and their tree propagation, then the velocity-limit rows. The scene's
    articulated link (on its limit, against the ground) and free body (with velocity limits)
    do not interact, so the separately launched phases and the fused kernel, which keeps the
    free body's contacts on free-body rows, must give the same trajectory. A single iteration
    pins the order of the row families within an iteration.
    """
    for iterations in (1, 8):
        with test.subTest(iterations=iterations):
            _check_fused_matches_phased(test, device, iterations)


def _check_fused_matches_phased(test, device, iterations):
    phased = _run_layout_trajectory(device, "propagation", iterations)
    fused = _run_layout_trajectory(device, "propagation_fused", iterations)
    for step, (expected, observed) in enumerate(zip(phased, fused, strict=True)):
        for label, sample in (("propagation", expected), ("propagation_fused", observed)):
            with test.subTest(step=step, response=label):
                types = sample["row_type"]
                # Dense rows hold only the joint rows; every contact touching the link is a propagation row.
                test.assertGreater(int(np.sum(types == PGS_CONSTRAINT_TYPE_JOINT_LIMIT)), 0)
                test.assertTrue(
                    np.all(
                        (types == PGS_CONSTRAINT_TYPE_JOINT_LIMIT) | (types == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)
                    )
                )
                test.assertGreater(sample["propagation_count"], 0)
                test.assertGreater(int(np.sum(sample["mf_row_type"] == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)), 0)
        np.testing.assert_array_equal(observed["row_type"], expected["row_type"], err_msg=f"step {step}")
        # The free body's contacts are propagation rows in the phased response and free-body rows when fused.
        test.assertEqual(int(np.sum(expected["mf_row_type"] == PGS_CONSTRAINT_TYPE_CONTACT)), 0)
        test.assertGreater(int(np.sum(observed["mf_row_type"] == PGS_CONSTRAINT_TYPE_CONTACT)), 0)
        # The fused kernel solves the free body's contacts as free-body rows, which rounds differently.
        for label in ("impulses", "joint_q", "joint_qd"):
            np.testing.assert_allclose(
                observed[label], expected[label], rtol=1.0e-5, atol=1.0e-5, err_msg=f"{label} differed at step {step}"
            )
    limit_rows = np.flatnonzero(phased[0]["row_type"] == PGS_CONSTRAINT_TYPE_JOINT_LIMIT)
    test.assertGreater(float(np.max(np.abs(phased[0]["impulses"][limit_rows]))), 0.0)


class TestSteelJointLimitActivationGapUpstream(unittest.TestCase):
    pass


devices = get_cuda_test_devices()


for _name, _func in (
    (
        "test_combined_row_families_follow_the_documented_layout",
        test_combined_row_families_follow_the_documented_layout,
    ),
    ("test_velocity_limit_rows_require_the_option", test_velocity_limit_rows_require_the_option),
    ("test_limit_holds_a_driven_joint", test_limit_holds_a_driven_joint),
    ("test_disabled_joint_limits_are_not_enforced", test_disabled_joint_limits_are_not_enforced),
    ("test_disabled_joint_limits_build_no_rows", test_disabled_joint_limits_build_no_rows),
    ("test_fused_propagation_matches_phased_trajectory", test_fused_propagation_matches_phased_trajectory),
):
    add_function_test(TestSteelJointLimitActivationGapUpstream, _name, _func, devices=devices)


split_devices = get_test_devices()


for _name, _func in (
    ("test_limit_holds_a_driven_joint", test_limit_holds_a_driven_joint),
    ("test_disabled_joint_limits_are_not_enforced", test_disabled_joint_limits_are_not_enforced),
    ("test_disabled_joint_limits_build_no_rows", test_disabled_joint_limits_build_no_rows),
):
    add_function_test(
        TestSteelJointLimitActivationGapUpstream,
        f"{_name}_split",
        _func,
        devices=split_devices,
        solve_mode="split",
    )


def _outputs(articulation_count: int, max_constraints: int, size: int, device):
    return (
        wp.zeros(articulation_count, dtype=wp.int32, device=device),
        wp.zeros((articulation_count, max_constraints, size), dtype=wp.float32, device=device),
        wp.zeros((articulation_count, max_constraints), dtype=wp.int32, device=device),
        wp.zeros((articulation_count, max_constraints), dtype=wp.int32, device=device),
        wp.zeros((articulation_count, max_constraints), dtype=wp.float32, device=device),
        wp.zeros((articulation_count, max_constraints), dtype=wp.float32, device=device),
        wp.zeros((articulation_count, max_constraints), dtype=wp.float32, device=device),
        wp.zeros((articulation_count, max_constraints), dtype=wp.float32, device=device),
        wp.zeros((articulation_count, max_constraints), dtype=wp.float32, device=device),
    )


class TestSteelJointLimitWarp(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "warp-parallel joint-limit assembly requires CUDA")
    def test_warp_builder_matches_scalar_row_order_and_values(self):
        device = wp.get_cuda_device()
        articulation_count = 257
        size = 6
        max_constraints = 2 * size
        dof_count = articulation_count * size
        rng = np.random.default_rng(42)

        articulation_start = wp.array(np.arange(articulation_count + 1), dtype=wp.int32, device=device)
        articulation_dof_start = wp.array(np.arange(articulation_count) * size, dtype=wp.int32, device=device)
        joint_type = wp.array(np.full(articulation_count, JointType.D6), dtype=wp.int32, device=device)
        joint_q_start = wp.array(np.arange(articulation_count) * size, dtype=wp.int32, device=device)
        joint_qd_start = wp.array(np.arange(articulation_count) * size, dtype=wp.int32, device=device)
        joint_dof_dim = wp.array(np.tile([3, 3], (articulation_count, 1)), dtype=wp.int32, device=device)
        lower = wp.array(np.full(dof_count, -1.0, dtype=np.float32), dtype=wp.float32, device=device)
        upper = wp.array(np.full(dof_count, 1.0, dtype=np.float32), dtype=wp.float32, device=device)
        joint_q = wp.array(rng.uniform(-1.2, 1.2, dof_count).astype(np.float32), dtype=wp.float32, device=device)
        art_to_world = wp.array(np.arange(articulation_count), dtype=wp.int32, device=device)
        group_to_art = wp.array(np.arange(articulation_count), dtype=wp.int32, device=device)
        scalar = _outputs(articulation_count, max_constraints, size, device)

        wp.launch(
            build_joint_limit_rows_for_size,
            dim=articulation_count,
            inputs=[
                articulation_start,
                articulation_dof_start,
                joint_type,
                joint_q_start,
                joint_qd_start,
                joint_dof_dim,
                lower,
                upper,
                joint_q,
                0.25,
                art_to_world,
                group_to_art,
                max_constraints,
                0.2,
                1.0e-6,
            ],
            outputs=[*scalar, ROW_SLOTS_ATOMIC, wp.zeros(1, dtype=wp.int32, device=device)],
            device=device,
        )

        parallel = _outputs(articulation_count, max_constraints, size, device)
        warps_per_block = 4
        kernel = _get_joint_limit_warp_kernel(size, device.arch, warps_per_block)
        wp.launch_tiled(
            kernel,
            dim=[(articulation_count + warps_per_block - 1) // warps_per_block],
            inputs=[
                articulation_count,
                articulation_dof_start,
                art_to_world,
                group_to_art,
                wp.array(np.arange(dof_count), dtype=wp.int32, device=device),
                lower,
                upper,
                joint_q,
                0.25,
                max_constraints,
                0.2,
                1.0e-6,
            ],
            outputs=[*parallel, ROW_SLOTS_ATOMIC, wp.zeros(1, dtype=wp.int32, device=device)],
            block_dim=32 * warps_per_block,
            device=device,
        )
        wp.synchronize_device(device)

        for parallel_array, scalar_array in zip(parallel, scalar, strict=True):
            np.testing.assert_array_equal(parallel_array.numpy(), scalar_array.numpy())


def _allocated_joint_velocity_slots(qd: float, *, fraction: float, qdot_max: float = 1.0):
    device = "cpu"
    velocity_limit_slot = wp.full((2,), -1, dtype=wp.int32, device=device)
    velocity_limit_sign = wp.zeros((2,), dtype=wp.float32, device=device)
    world_slot_counter = wp.zeros((1,), dtype=wp.int32, device=device)

    wp.launch(
        allocate_joint_velocity_limit_slots,
        dim=1,
        inputs=[
            wp.array([0, 1], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([1], dtype=wp.int32, device=device),
            wp.array([int(JointType.REVOLUTE)], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([[0, 1]], dtype=wp.int32, device=device),
            wp.array([qdot_max], dtype=wp.float32, device=device),
            wp.array([qd], dtype=wp.float32, device=device),
            fraction,
            wp.array([-1], dtype=wp.int32, device=device),  # drive_slot (unused: skip_driven=0)
            0,  # skip_driven
            wp.array([0], dtype=wp.int32, device=device),
            8,
            wp.ones(1, dtype=wp.int32, device=device),
        ],
        outputs=[
            velocity_limit_slot,
            velocity_limit_sign,
            world_slot_counter,
            ROW_SLOTS_ATOMIC,
            wp.zeros(1, dtype=wp.int32, device=device),
        ],
        device=device,
    )
    return (
        velocity_limit_slot.numpy().tolist(),
        velocity_limit_sign.numpy().tolist(),
        int(world_slot_counter.numpy()[0]),
    )


def _allocated_rigid_velocity_slots(qd6, *, fraction: float, lin_limit: float = 1.0, ang_limit: float = 1.0):
    device = "cpu"
    rigid_velocity_limit_slot = wp.full((12,), -1, dtype=wp.int32, device=device)
    rigid_velocity_limit_sign = wp.zeros((12,), dtype=wp.float32, device=device)
    mf_slot_counter = wp.zeros((1,), dtype=wp.int32, device=device)

    wp.launch(
        allocate_rigid_velocity_limit_slots,
        dim=1,
        inputs=[
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([1], dtype=wp.int32, device=device),
            wp.array([int(BodyFlags.DYNAMIC)], dtype=wp.int32, device=device),
            wp.array([lin_limit], dtype=wp.float32, device=device),
            wp.array([ang_limit], dtype=wp.float32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array(list(qd6), dtype=wp.float32, device=device),
            wp.array([wp.quat_identity()], dtype=wp.quat, device=device),
            fraction,
            64,
            wp.ones(1, dtype=wp.int32, device=device),
        ],
        outputs=[
            rigid_velocity_limit_slot,
            rigid_velocity_limit_sign,
            mf_slot_counter,
            ROW_SLOTS_ATOMIC,
            wp.zeros(1, dtype=wp.int32, device=device),
        ],
        device=device,
    )
    return (
        rigid_velocity_limit_slot.numpy().tolist(),
        rigid_velocity_limit_sign.numpy().tolist(),
        int(mf_slot_counter.numpy()[0]),
    )


class TestSteelVelocityLimitActivationFraction(unittest.TestCase):
    def test_solver_rejects_invalid_velocity_limit_activation_fraction(self):
        model = newton.ModelBuilder().finalize()

        for value in (-0.1, float("nan")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "velocity_limit_activation_fraction"):
                    SolverSteel(model, velocity_limit_activation_fraction=value)

    def test_solver_accepts_and_defaults_velocity_limit_activation_fraction(self):
        model = newton.ModelBuilder().finalize()

        self.assertEqual(SolverSteel(model).velocity_limit_activation_fraction, 0.0)
        self.assertEqual(
            SolverSteel(model, velocity_limit_activation_fraction=0.7).velocity_limit_activation_fraction,
            0.7,
        )

    def test_fraction_zero_preserves_always_allocate_joint_rows(self):
        # fraction == 0.0 must match the historical behavior exactly: both
        # sides of every finitely limited DOF allocate, in the same order,
        # regardless of the current joint velocity.
        for qd in (0.0, 0.5, -2.0):
            with self.subTest(qd=qd):
                slots, signs, count = _allocated_joint_velocity_slots(qd, fraction=0.0)
                self.assertEqual(slots, [0, 1])
                self.assertEqual(signs, [1.0, -1.0])
                self.assertEqual(count, 2)

    def test_fraction_gates_static_joint_dof_to_zero_rows(self):
        slots, signs, count = _allocated_joint_velocity_slots(0.0, fraction=0.9)
        self.assertEqual(slots, [-1, -1])
        self.assertEqual(signs, [0.0, 0.0])
        self.assertEqual(count, 0)

        slots, signs, count = _allocated_joint_velocity_slots(0.85, fraction=0.9)
        self.assertEqual(slots, [-1, -1])
        self.assertEqual(count, 0)

    def test_joint_dof_past_threshold_allocates_rows_same_step(self):
        # The gate reads the same-step joint_qd input, so a DOF driven past
        # the threshold gets its row pair allocated in that very step.
        for qd in (0.95, -0.95, 1.5):
            with self.subTest(qd=qd):
                slots, signs, count = _allocated_joint_velocity_slots(qd, fraction=0.9)
                self.assertEqual(slots, [0, 1])
                self.assertEqual(signs, [1.0, -1.0])
                self.assertEqual(count, 2)

    def test_fraction_zero_preserves_always_allocate_rigid_rows(self):
        slots, signs, count = _allocated_rigid_velocity_slots([0.0] * 6, fraction=0.0)
        self.assertEqual(slots, list(range(12)))
        self.assertEqual(signs, [1.0, -1.0] * 6)
        self.assertEqual(count, 12)

    def test_fraction_gates_static_rigid_body_to_zero_rows(self):
        slots, signs, count = _allocated_rigid_velocity_slots([0.0] * 6, fraction=0.9)
        self.assertEqual(slots, [-1] * 12)
        self.assertEqual(signs, [0.0] * 12)
        self.assertEqual(count, 0)

    def test_rigid_axis_past_threshold_allocates_rows_same_step(self):
        # Only the linear-z axis (axis 2 -> slot indices 4, 5) exceeds the
        # 90% activation threshold; its pair allocates this step.
        slots, signs, count = _allocated_rigid_velocity_slots([0.0, 0.0, 0.95, 0.0, 0.0, 0.0], fraction=0.9)
        expected_slots = [-1] * 12
        expected_slots[4] = 0
        expected_slots[5] = 1
        expected_signs = [0.0] * 12
        expected_signs[4] = 1.0
        expected_signs[5] = -1.0
        self.assertEqual(slots, expected_slots)
        self.assertEqual(signs, expected_signs)
        self.assertEqual(count, 2)


QDOT_MAX = 0.5


# Slack on the limit for an under-converged sweep; the final pass leaves the DOF at the limit.
LIMIT_TOL = 1.05


def _allocated_joint_velocity_slots_velocity_limit_activation_fraction_upstream(
    device, qd: float, *, fraction: float, qdot_max: float = 1.0
):
    velocity_limit_slot = wp.full((2,), -1, dtype=wp.int32, device=device)
    velocity_limit_sign = wp.zeros((2,), dtype=wp.float32, device=device)
    world_slot_counter = wp.zeros((1,), dtype=wp.int32, device=device)
    wp.launch(
        allocate_joint_velocity_limit_slots,
        dim=1,
        inputs=[
            wp.array([0, 1], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([1], dtype=wp.int32, device=device),
            wp.array([int(JointType.REVOLUTE)], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([[0, 1]], dtype=wp.int32, device=device),
            wp.array([qdot_max], dtype=wp.float32, device=device),
            wp.array([qd], dtype=wp.float32, device=device),
            fraction,
            wp.array([-1], dtype=wp.int32, device=device),
            0,
            wp.array([0], dtype=wp.int32, device=device),
            8,
            wp.ones(1, dtype=wp.int32, device=device),
        ],
        outputs=[
            velocity_limit_slot,
            velocity_limit_sign,
            world_slot_counter,
            ROW_SLOTS_ATOMIC,
            wp.zeros(1, dtype=wp.int32, device=device),
        ],
        device=device,
    )
    return (
        velocity_limit_slot.numpy().tolist(),
        velocity_limit_sign.numpy().tolist(),
        int(world_slot_counter.numpy()[0]),
    )


def _allocated_rigid_velocity_slots_velocity_limit_activation_fraction_upstream(
    device, qd6, *, fraction: float, lin_limit: float = 1.0, ang_limit: float = 1.0
):
    rigid_velocity_limit_slot = wp.full((12,), -1, dtype=wp.int32, device=device)
    rigid_velocity_limit_sign = wp.zeros((12,), dtype=wp.float32, device=device)
    mf_slot_counter = wp.zeros((1,), dtype=wp.int32, device=device)
    wp.launch(
        allocate_rigid_velocity_limit_slots,
        dim=1,
        inputs=[
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array([1], dtype=wp.int32, device=device),
            wp.array([int(BodyFlags.DYNAMIC)], dtype=wp.int32, device=device),
            wp.array([lin_limit], dtype=wp.float32, device=device),
            wp.array([ang_limit], dtype=wp.float32, device=device),
            wp.array([0], dtype=wp.int32, device=device),
            wp.array(list(qd6), dtype=wp.float32, device=device),
            wp.array([wp.quat_identity()], dtype=wp.quat, device=device),
            fraction,
            64,
            wp.ones(1, dtype=wp.int32, device=device),
        ],
        outputs=[
            rigid_velocity_limit_slot,
            rigid_velocity_limit_sign,
            mf_slot_counter,
            ROW_SLOTS_ATOMIC,
            wp.zeros(1, dtype=wp.int32, device=device),
        ],
        device=device,
    )
    return (
        rigid_velocity_limit_slot.numpy().tolist(),
        rigid_velocity_limit_sign.numpy().tolist(),
        int(mf_slot_counter.numpy()[0]),
    )


def test_fraction_zero_allocates_every_joint_row(test, device):
    """Allocate both rows of every limited DOF regardless of its velocity when the fraction is zero."""
    for qd in (0.0, 0.5, -2.0):
        with test.subTest(qd=qd):
            slots, signs, count = _allocated_joint_velocity_slots_velocity_limit_activation_fraction_upstream(
                device, qd, fraction=0.0
            )
            test.assertEqual(slots, [0, 1])
            test.assertEqual(signs, [1.0, -1.0])
            test.assertEqual(count, 2)


def test_fraction_zero_allocates_every_rigid_row(test, device):
    """Allocate all twelve free-body velocity-limit rows when the fraction is zero."""
    slots, signs, count = _allocated_rigid_velocity_slots_velocity_limit_activation_fraction_upstream(
        device, [0.0] * 6, fraction=0.0
    )
    test.assertEqual(slots, list(range(12)))
    test.assertEqual(signs, [1.0, -1.0] * 6)
    test.assertEqual(count, 12)


def test_free_body_velocity_limits_hold(test, device, solve_mode="matrix_free"):
    """Hold a free body's authored linear and angular speed limits."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    SolverSteel.register_custom_attributes(builder)
    body = builder.add_body(
        custom_attributes={
            "steel:rigid_body_max_linear_velocity": 0.5,
            "steel:rigid_body_max_angular_velocity": 1.0,
        }
    )
    builder.add_shape_box(body, hx=0.1, hy=0.2, hz=0.3)
    model = builder.finalize(device=device)
    solver = SolverSteel(model, solve_mode=solve_mode)
    state_0, state_1 = model.state(), model.state()
    qd = state_0.joint_qd.numpy()
    qd[3:6] = (5.0, 0.0, 0.0)
    state_0.joint_qd.assign(qd)
    for _ in range(60):
        solver.step(state_0, state_1, model.control(), None, 1.0 / 60.0)
        state_0, state_1 = state_1, state_0
    final = state_0.joint_qd.numpy()
    test.assertLessEqual(float(np.max(np.abs(final[0:3]))), 0.5 * LIMIT_TOL)
    test.assertLessEqual(float(np.max(np.abs(final[3:6]))), 1.0 * LIMIT_TOL)


class TestSteelVelocityLimitActivationFractionUpstream(unittest.TestCase):
    pass


for _name, _func in (
    ("test_fraction_zero_allocates_every_joint_row", test_fraction_zero_allocates_every_joint_row),
    ("test_fraction_zero_allocates_every_rigid_row", test_fraction_zero_allocates_every_rigid_row),
    ("test_free_body_velocity_limits_hold", test_free_body_velocity_limits_hold),
):
    add_function_test(TestSteelVelocityLimitActivationFractionUpstream, _name, _func, devices=devices)


# The allocators are mode-independent kernels; free-body velocity limits are also rows in split mode.
for _name, _func in (
    ("test_fraction_zero_allocates_every_joint_row", test_fraction_zero_allocates_every_joint_row),
    ("test_fraction_zero_allocates_every_rigid_row", test_fraction_zero_allocates_every_rigid_row),
):
    add_function_test(
        TestSteelVelocityLimitActivationFractionUpstream,
        _name,
        _func,
        devices=[d for d in get_test_devices() if d.is_cpu],
    )


add_function_test(
    TestSteelVelocityLimitActivationFractionUpstream,
    "test_free_body_velocity_limits_hold_split",
    test_free_body_velocity_limits_hold,
    devices=get_test_devices(),
    solve_mode="split",
)


# GS-convergence slack on the limit; the final clamp leaves the DOF at the
# limit, later float noise stays well under 5%.
LIMIT_TOL = 1.05


def _build_arm_and_box_model(device) -> newton.Model:
    """A driven revolute arm plus a heavy free box overlapping its tip.

    Gravity is zero and the box carries a large downward velocity, so the
    contact impulse into the arm tip is the only thing fighting the velocity
    limit — fully deterministic.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.default_shape_cfg.density = 1000.0
    builder.default_shape_cfg.ke = 1.0e5
    builder.default_shape_cfg.kd = 1.0e3
    builder.default_shape_cfg.mu = 0.0
    builder.default_shape_cfg.margin = 0.0
    builder.default_shape_cfg.gap = 0.0

    # Arm spanning x in [0, 0.8] at z = 0.5, hinged at its left end about y.
    arm = builder.add_link()
    builder.add_shape_box(arm, hx=0.4, hy=0.05, hz=0.02)
    j_arm = builder.add_joint_revolute(
        parent=-1,
        child=arm,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.4, 0.0, 0.0), wp.quat_identity()),
    )
    builder.add_articulation([j_arm])

    # Heavy box, bottom face slightly penetrating the arm top near the tip.
    box = builder.add_link(xform=wp.transform(wp.vec3(0.7, 0.0, 0.5695), wp.quat_identity()))
    builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.05)
    builder.add_articulation([builder.add_joint_free(parent=-1, child=box)])

    return builder.finalize(device=device)


class TestSteelFusedVelocityLimitOrdering(unittest.TestCase):
    def _max_arm_speed_over_impact(self, fuse: bool, articulated_contact_response: str) -> float:
        model = _build_arm_and_box_model(device="cuda:0")

        n = model.joint_dof_count
        target_ke = np.zeros(n, dtype=np.float32)
        target_kd = np.zeros(n, dtype=np.float32)
        vel_limit = np.full(n, np.inf, dtype=np.float32)
        target_ke[0] = 200.0
        target_kd[0] = 5.0
        vel_limit[0] = QDOT_MAX
        model.joint_target_ke.assign(target_ke)
        model.joint_target_kd.assign(target_kd)
        model.joint_velocity_limit.assign(vel_limit)

        solver = SolverSteel(
            model,
            solve_mode="matrix_free",
            drive_mode="rows",
            articulated_contact_response=articulated_contact_response,
            enable_joint_velocity_limits=True,
            fuse_joint_velocity_limits=fuse,
            contact_friction_scale=0.0,
            # Deliberately under-converged (franka-class training configs run
            # 8): ordering only matters when the sweep does NOT converge — at
            # convergence the last contact visits apply ~zero delta and any
            # clamp placement passes. A drive-visit clamp lets this impact
            # reach 1.53x the limit at 8 iterations (7.1x at 1); dedicated
            # rows and the end-of-iteration clamp hold the limit exactly at
            # every iteration count.
            iterations=8,
            articulation_constraint_max=32,
            free_body_constraint_max=32,
        )

        state_0, state_1 = model.state(), model.state()
        control = model.control()
        joint_qd = state_0.joint_qd.numpy()
        joint_qd[3] = -3.0  # box linear z: slam into the arm tip
        state_0.joint_qd.assign(joint_qd)
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

        collision_pipeline = newton.CollisionPipeline(model)

        contacts = collision_pipeline.contacts()
        max_arm_speed = 0.0
        for _ in range(20):
            state_0.clear_forces()
            collision_pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0
            max_arm_speed = max(max_arm_speed, float(abs(state_0.joint_qd.numpy()[0])))
        wp.synchronize()
        return max_arm_speed

    @unittest.skipUnless(wp.is_cuda_available(), "matrix-free Steel requires CUDA")
    def test_dedicated_rows_hold_limit_through_impact(self):
        # Reference behavior: dedicated velocity-limit rows run in the final
        # phase of every iteration and hold the limit through the impact.
        speed = self._max_arm_speed_over_impact(fuse=False, articulated_contact_response="immediate")
        self.assertLessEqual(speed, QDOT_MAX * LIMIT_TOL)

    @unittest.skipUnless(wp.is_cuda_available(), "matrix-free Steel requires CUDA")
    def test_fused_clamp_holds_limit_through_impact(self):
        # The fused clamp must match the dedicated rows it replaces: the
        # contact impulse lands mid-iteration, so the clamp has to run after
        # contacts to keep the limit at the end of the step.
        speed = self._max_arm_speed_over_impact(fuse=True, articulated_contact_response="immediate")
        self.assertLessEqual(speed, QDOT_MAX * LIMIT_TOL)

    @unittest.skipUnless(wp.is_cuda_available(), "propagation_fused Steel requires CUDA")
    def test_fused_clamp_holds_limit_through_impact_propagation_fused(self):
        # Same invariant on the propagation full-iteration kernel, where the
        # box-arm contact is solved in the propagation family with the tree
        # response inside the same kernel.
        speed = self._max_arm_speed_over_impact(fuse=True, articulated_contact_response="propagation_fused")
        self.assertLessEqual(speed, QDOT_MAX * LIMIT_TOL)


DT = 1.0 / 120.0


MASS = 2.0


ARMATURE = 0.1


KE = 400.0


KD = 15.0


TARGET_POS = 0.3


TARGET_VEL = 0.2


def _build_slider(device, *, effort_limit=1.0e6):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(mass=MASS, inertia=wp.mat33(np.eye(3)))
    joint = builder.add_joint_prismatic(
        -1,
        body,
        axis=newton.Axis.X,
        target_ke=KE,
        target_kd=KD,
        target_pos=TARGET_POS,
        target_vel=TARGET_VEL,
        armature=ARMATURE,
        effort_limit=effort_limit,
    )
    builder.add_articulation([joint])
    return builder.finalize(device=device)


def _backward_euler_trajectory(steps):
    """Implicit PD drive: ``(m + a + dt kd + dt^2 ke) qdd = -(ke (q - q_t + dt qd) + kd (qd - qd_t))``."""
    q = 0.0
    qd = 0.0
    history = []
    effective_mass = MASS + ARMATURE + DT * KD + DT * DT * KE
    for _ in range(steps):
        u0 = -(KE * (q - TARGET_POS + DT * qd) + KD * (qd - TARGET_VEL))
        qd = qd + DT * u0 / effective_mass
        q = q + DT * qd
        history.append((q, qd))
    return np.asarray(history)


def _slider_trajectory(model, steps, **kwargs):
    solver = SolverSteel(model, solve_mode="matrix_free", drive_mode="rows", **kwargs)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    history = []
    for _ in range(steps):
        solver.step(state_0, state_1, control, None, DT)
        state_0, state_1 = state_1, state_0
        history.append((float(state_0.joint_q.numpy()[0]), float(state_0.joint_qd.numpy()[0])))
    return np.asarray(history)


def _build_driven_chain(device, num_links=3, num_worlds=2):
    chain = newton.ModelBuilder()
    joints = []
    parent = -1
    for i in range(num_links):
        link = chain.add_link()
        chain.add_shape_box(link, hx=0.2, hy=0.05, hz=0.05)
        joints.append(
            chain.add_joint_revolute(
                parent=parent,
                child=link,
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent_xform=wp.transform(wp.vec3(0.0 if i == 0 else 0.25, 0.0, 2.0 if i == 0 else 0.0)),
                child_xform=wp.transform(wp.vec3(-0.25, 0.0, 0.0)),
                target_ke=100.0,
                target_kd=5.0,
                limit_lower=-1.0,
                limit_upper=1.0,
            )
        )
        parent = link
    chain.add_articulation(joints)
    builder = newton.ModelBuilder()
    builder.replicate(chain, num_worlds, spacing=(2.0, 0.0, 0.0))
    return builder.finalize(device=device)


def _build_arm_and_box_model_pgs_drives(device) -> newton.Model:
    """A driven, velocity-limited revolute arm and a free box hitting its tip, without gravity."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.default_shape_cfg.density = 1000.0
    builder.default_shape_cfg.mu = 0.5
    builder.default_shape_cfg.margin = 0.0
    builder.default_shape_cfg.gap = 0.0
    arm = builder.add_link()
    builder.add_shape_box(arm, hx=0.4, hy=0.05, hz=0.02)
    j_arm = builder.add_joint_revolute(
        parent=-1,
        child=arm,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.4, 0.0, 0.0), wp.quat_identity()),
        target_ke=200.0,
        target_kd=5.0,
    )
    builder.add_articulation([j_arm])
    box = builder.add_link(xform=wp.transform(wp.vec3(0.7, 0.0, 0.5695), wp.quat_identity()))
    builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.05)
    builder.add_articulation([builder.add_joint_free(parent=-1, child=box)])
    model = builder.finalize(device=device)
    limits = np.full(model.joint_dof_count, np.inf, dtype=np.float32)
    limits[0] = 0.5
    model.joint_velocity_limit.assign(limits)
    return model


def _run_arm_and_box(model, solver, steps, *, capture=False):
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    joint_qd = state_0.joint_qd.numpy()
    joint_qd[3] = -3.0
    state_0.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()

    def substeps():
        # An even number of substeps keeps the state roles fixed for replay.
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
        pipeline.collide(state_1, contacts)
        solver.step(state_1, state_0, control, contacts, 1.0 / 240.0)

    if capture:
        with wp.ScopedCapture(device=model.device) as graph:
            substeps()
    history = []
    for _ in range(steps // 2):
        if capture:
            wp.capture_launch(graph.graph)
        else:
            substeps()
        history.append(state_0.joint_q.numpy().copy())
    return np.asarray(history)


def test_pgs_drive_matches_backward_euler_reference(test, device):
    """Reach the implicit PD drive solution below the effort limit: the force-drive row converges to backward Euler."""
    np.testing.assert_allclose(
        _slider_trajectory(_build_slider(device), 120), _backward_euler_trajectory(120), rtol=1.0e-4, atol=1.0e-5
    )


def test_pgs_drive_impulse_is_clamped_to_the_effort_limit(test, device):
    """Clamp the drive row's impulse to ``joint_effort_limit * dt``."""
    effort_limit = 5.0
    history = _slider_trajectory(_build_slider(device, effort_limit=effort_limit), 60)
    qd = np.concatenate(([0.0], history[:, 1]))
    drive_force = (MASS + ARMATURE) * np.diff(qd) / DT
    test.assertLessEqual(float(np.max(np.abs(drive_force))), effort_limit * (1.0 + 1.0e-3))
    # The drive saturates early in the step response.
    test.assertGreater(float(np.max(np.abs(drive_force))), 0.99 * effort_limit)


def test_saturated_pgs_drive_differs_from_the_augmented_drive(test, device):
    """Differ from the implicit drive when the effort limit saturates, at any iteration count.

    The augmented drive clamps the explicit drive force before the implicit solve, so the
    clamped force is divided by the drive-augmented mass. The drive row clamps its
    accumulated impulse at ``joint_effort_limit * dt``, which acts on the plain mass.
    """
    effort_limit = 5.0
    model = _build_slider(device, effort_limit=effort_limit)
    qd = {}
    for drive_mode in ("augmented", "rows"):
        solver = SolverSteel(model, solve_mode="matrix_free", drive_mode=drive_mode, iterations=100)
        state_0, state_1 = model.state(), model.state()
        solver.step(state_0, state_1, model.control(), None, DT)
        qd[drive_mode] = float(state_1.joint_qd.numpy()[0])
    augmented_mass = MASS + ARMATURE + DT * KD + DT * DT * KE
    test.assertAlmostEqual(qd["augmented"], DT * effort_limit / augmented_mass, delta=1.0e-6)
    test.assertAlmostEqual(qd["rows"], DT * effort_limit / (MASS + ARMATURE), delta=1.0e-6)
    test.assertGreater(qd["rows"] - qd["augmented"], 1.0e-3)


def test_pgs_drive_bounds_complete_reaction_under_external_load(test, device):
    """Bound the complete drive reaction by ``joint_effort_limit`` under a large external load.

    The augmented drive clamps only its explicit force, so its implicit reaction is unbounded in this scene.
    """
    dt = 0.01
    for effort_limit in (1.0, 2.0):
        for force in (-10000.0, 10000.0):
            with test.subTest(effort_limit=effort_limit, force=force):
                builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
                body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
                joint = builder.add_joint_prismatic(
                    -1,
                    body,
                    axis=newton.Axis.X,
                    target_ke=10000.0,
                    target_kd=0.0,
                    target_pos=0.0,
                    damping=0.0,
                    armature=0.0,
                    effort_limit=effort_limit,
                )
                builder.add_articulation([joint])
                model = builder.finalize(device=device)
                solver = SolverSteel(model, solve_mode="matrix_free", drive_mode="rows")
                state_0, state_1 = model.state(), model.state()
                newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
                state_0.body_f.assign(np.array([[force, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32))
                solver.step(state_0, state_1, model.control(), None, dt)
                reaction = float(state_1.joint_qd.numpy()[0]) / dt - force
                test.assertLessEqual(abs(reaction), 1.05 * effort_limit)
                test.assertAlmostEqual(reaction, -np.sign(force) * effort_limit, delta=0.05)


def test_drive_rows_precede_limit_rows_and_count_against_capacity(test, device):
    """Allocate drive rows first in each world and flag the world when they do not fit."""
    model = _build_driven_chain(device, num_links=3, num_worlds=2)
    solver = SolverSteel(
        model, solve_mode="matrix_free", drive_mode="rows", enable_joint_limits=True, warn_constraint_overflow=False
    )
    state_0, state_1 = model.state(), model.state()
    solver.step(state_0, state_1, model.control(), None, DT)
    counts = solver._constraint_count.numpy()
    row_type = solver._row_type.numpy()
    np.testing.assert_array_equal(counts, [9, 9])  # 3 drive rows + 2 position-limit rows per DOF
    for world in range(2):
        test.assertTrue(np.all(row_type[world, :3] == PGS_CONSTRAINT_TYPE_JOINT_TARGET))
        test.assertTrue(np.all(row_type[world, 3:9] == PGS_CONSTRAINT_TYPE_JOINT_LIMIT))
    test.assertFalse(np.any(solver.constraint_overflow.numpy()))

    small = SolverSteel(
        model,
        solve_mode="matrix_free",
        drive_mode="rows",
        enable_joint_limits=True,
        articulation_constraint_max=2,
        warn_constraint_overflow=False,
    )
    small.step(model.state(), model.state(), model.control(), None, DT)
    np.testing.assert_array_equal(small._constraint_count.numpy(), [2, 2])
    test.assertTrue(np.all(small._row_type.numpy()[:, :2] == PGS_CONSTRAINT_TYPE_JOINT_TARGET))
    np.testing.assert_array_equal(small.constraint_overflow.numpy(), [True, True])
    with test.assertRaisesRegex(RuntimeError, "overflow|capacity"):
        small.check_constraint_capacity()


def test_undriven_dofs_get_no_drive_rows(test, device):
    """Create no drive rows for DOFs with zero target stiffness and damping."""
    model = _build_driven_chain(device, num_links=3, num_worlds=1)
    ke = model.joint_target_ke.numpy()
    kd = model.joint_target_kd.numpy()
    ke[1] = 0.0
    kd[1] = 0.0
    model.joint_target_ke.assign(ke)
    model.joint_target_kd.assign(kd)
    solver = SolverSteel(model, solve_mode="matrix_free", drive_mode="rows", joint_limit_activation_gap=0.0)
    solver.step(model.state(), model.state(), model.control(), None, DT)
    test.assertEqual(int(solver._constraint_count.numpy()[0]), 2)
    np.testing.assert_array_equal(solver._drive_slot.numpy(), [0, -1, 1])


def test_prescale_skips_fused_driven_dofs(test, device):
    """Exclude driven DOFs from the pre-solve velocity scaling only when the fused clamp limits them."""

    def prescaled(skip_driven):
        joint_qd = wp.array([5.0, 0.5], dtype=wp.float32, device=device)
        wp.launch(
            prescale_joint_velocity_limits,
            dim=1,
            inputs=[
                wp.array([0, 2], dtype=wp.int32, device=device),
                wp.array([int(JointType.REVOLUTE)] * 2, dtype=wp.int32, device=device),
                wp.array([0, 1], dtype=wp.int32, device=device),
                wp.array([0, 1], dtype=wp.int32, device=device),
                wp.array([[0, 1], [0, 1]], dtype=wp.int32, device=device),
                wp.array([1.0, 1.0], dtype=wp.float32, device=device),
                wp.array([int(BodyFlags.DYNAMIC)] * 2, dtype=wp.int32, device=device),
                wp.array([0, -1], dtype=wp.int32, device=device),
                skip_driven,
                wp.ones(1, dtype=wp.int32, device=device),
            ],
            outputs=[joint_qd],
            device=device,
        )
        return joint_qd.numpy()

    # Without fusion the driven DOF's overshoot scales the whole articulation.
    np.testing.assert_allclose(prescaled(0), [1.0, 0.1], rtol=1.0e-6)
    # With fusion the driven DOF is clamped in the solve and nothing is scaled.
    np.testing.assert_allclose(prescaled(1), [5.0, 0.5], rtol=1.0e-6)


def test_drive_rows_use_the_exact_unit_response(test, device):
    """Leave CFM out of drive rows: the drive and the fused clamp divide by the exact response."""
    model = _build_driven_chain(device, num_links=3, num_worlds=1)
    diags = {}
    for cfm in (0.0, 0.5):
        solver = SolverSteel(model, solve_mode="matrix_free", drive_mode="rows", enable_joint_limits=True, cfm=cfm)
        solver.step(model.state(), model.state(), model.control(), None, DT)
        diags[cfm] = solver._diag.numpy()[0, : int(solver._constraint_count.numpy()[0])]
    test.assertEqual(diags[0.0].size, 9)  # 3 drive rows + 2 position-limit rows per DOF
    np.testing.assert_array_equal(diags[0.5][:3], diags[0.0][:3])  # drive rows
    np.testing.assert_allclose(diags[0.5][3:] - diags[0.0][3:], 0.5, rtol=1.0e-5)  # limit rows
    # With a large CFM the PGS drive still reaches the backward-Euler reference.
    np.testing.assert_allclose(
        _slider_trajectory(_build_slider(device), 120, cfm=0.5),
        _backward_euler_trajectory(120),
        rtol=1.0e-4,
        atol=1.0e-5,
    )


def test_fused_clamp_lands_on_the_limit_with_large_cfm(test, device):
    """Return a driven DOF exactly to its limit regardless of cfm."""
    model = _build_arm_and_box_model_pgs_drives(device)
    solver = SolverSteel(
        model,
        solve_mode="matrix_free",
        drive_mode="rows",
        enable_joint_velocity_limits=True,
        cfm=0.5,
        iterations=8,
    )
    state_0, state_1 = model.state(), model.state()
    joint_qd = state_0.joint_qd.numpy()
    joint_qd[3] = -3.0
    state_0.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(state_0, contacts)
    solver.step(state_0, state_1, model.control(), contacts, 1.0 / 240.0)
    # The impact drives the arm past its 0.5 rad/s limit; the clamp leaves it at the limit.
    test.assertAlmostEqual(float(abs(state_1.joint_qd.numpy()[0])), 0.5, delta=1.0e-6)


def test_captured_drive_rows_match_eager_steps(test, device):
    """Replay a captured graph of drive, velocity-limit and contact rows identically to eager steps."""
    model = _build_arm_and_box_model_pgs_drives(device)
    kwargs = {"drive_mode": "rows", "enable_joint_velocity_limits": True, "iterations": 8}
    eager = _run_arm_and_box(model, SolverSteel(model, solve_mode="matrix_free", **kwargs), 20)
    captured = _run_arm_and_box(model, SolverSteel(model, solve_mode="matrix_free", **kwargs), 20, capture=True)
    np.testing.assert_allclose(captured, eager, atol=2.0e-6)


def test_streamed_drive_metadata_matches_resident(test, device):
    """Produce the same trajectory whether the drive-row metadata is resident or streamed."""
    model = _build_arm_and_box_model_pgs_drives(device)
    max_shared = int(getattr(model.device, "max_shared_memory_per_block", 0))
    resident_rows, streamed_rows = 32, 160
    for rows, resident in ((resident_rows, True), (streamed_rows, False)):
        test.assertEqual(
            _use_resident_mfgs_metadata(rows, 32, 7, max_shared, has_drive_rows=True, fuse_vel_limits=True), resident
        )
    kwargs = {"drive_mode": "rows", "enable_joint_velocity_limits": True, "iterations": 8}
    reference = _run_arm_and_box(
        model,
        SolverSteel(model, solve_mode="matrix_free", articulation_constraint_max=resident_rows, **kwargs),
        20,
    )
    streamed = _run_arm_and_box(
        model,
        SolverSteel(model, solve_mode="matrix_free", articulation_constraint_max=streamed_rows, **kwargs),
        20,
    )
    np.testing.assert_allclose(streamed, reference, atol=1.0e-6)


class TestSteelPGSDrives(unittest.TestCase):
    pass


for _name, _func in (
    ("test_pgs_drive_matches_backward_euler_reference", test_pgs_drive_matches_backward_euler_reference),
    ("test_pgs_drive_impulse_is_clamped_to_the_effort_limit", test_pgs_drive_impulse_is_clamped_to_the_effort_limit),
    (
        "test_saturated_pgs_drive_differs_from_the_augmented_drive",
        test_saturated_pgs_drive_differs_from_the_augmented_drive,
    ),
    (
        "test_pgs_drive_bounds_complete_reaction_under_external_load",
        test_pgs_drive_bounds_complete_reaction_under_external_load,
    ),
    (
        "test_drive_rows_precede_limit_rows_and_count_against_capacity",
        test_drive_rows_precede_limit_rows_and_count_against_capacity,
    ),
    ("test_undriven_dofs_get_no_drive_rows", test_undriven_dofs_get_no_drive_rows),
    ("test_prescale_skips_fused_driven_dofs", test_prescale_skips_fused_driven_dofs),
    ("test_drive_rows_use_the_exact_unit_response", test_drive_rows_use_the_exact_unit_response),
    ("test_fused_clamp_lands_on_the_limit_with_large_cfm", test_fused_clamp_lands_on_the_limit_with_large_cfm),
    ("test_captured_drive_rows_match_eager_steps", test_captured_drive_rows_match_eager_steps),
    ("test_streamed_drive_metadata_matches_resident", test_streamed_drive_metadata_matches_resident),
):
    add_function_test(TestSteelPGSDrives, _name, _func, devices=devices)


_HAS_DRIVE_ROWS = "drive_mode" in inspect.signature(SolverSteel.__init__).parameters


_DT = 1.0 / 120.0


def _driven_lever_on_ground(device):
    """A horizontal lever held by a stiff revolute drive whose 1 kg foot rests 1 mm deep in the ground.

    Drive and contact share the weight. The contact's position bias pushes the foot out
    during the position solve; the velocity-only iterations drop that bias, so the contact
    impulse changes there and an active drive row reacts to it.
    """
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    link = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.049), wp.quat_identity()))
    # A foot at the far end of the lever touches the ground; the hinge itself has no shape.
    builder.add_shape_box(
        link,
        xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()),
        hx=0.05,
        hy=0.05,
        hz=0.05,
        cfg=newton.ModelBuilder.ShapeConfig(density=1000.0),
    )
    joint = builder.add_joint_revolute(
        -1,
        link,
        axis=wp.vec3(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.049), wp.quat_identity()),
        target_ke=1000.0,
        target_kd=50.0,
        target_pos=0.0,
    )
    builder.add_articulation([joint])
    return builder.finalize(device=device)


def _one_step(device, **solver_kwargs):
    model = _driven_lever_on_ground(device)
    solver = SolverSteel(model, solve_mode="matrix_free", drive_mode="rows", iterations=4, **solver_kwargs)
    state_0, state_1 = model.state(), model.state()
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(state_0, contacts)
    solver.step(state_0, state_1, model.control(), contacts, _DT)
    drive_slot = int(solver._drive_slot.numpy()[0])
    count = int(solver._constraint_count.numpy()[0])
    row_type = solver._row_type.numpy()[0, :count]
    impulses = solver._impulses.numpy()[0, :count]
    contact_impulse = float(impulses[row_type == PGS_CONSTRAINT_TYPE_CONTACT].sum())
    return float(impulses[drive_slot]), contact_impulse


@unittest.skipUnless(_HAS_DRIVE_ROWS, "PGS joint-drive rows (drive_mode='rows') are not available")
def test_frozen_drive_rows_keep_the_position_solve_impulse(test, device):
    """Keep the position solve's drive impulse under ``"freeze"`` and keep solving the drive rows under ``"active"``."""
    position_drive, position_contact = _one_step(device, velocity_iterations=0)
    frozen_drive, frozen_contact = _one_step(device, velocity_iterations=8, velocity_drive_mode="freeze")
    active_drive, _active_contact = _one_step(device, velocity_iterations=8, velocity_drive_mode="active")

    # The velocity-only iterations change the contact impulse (its position bias is dropped).
    test.assertGreater(abs(frozen_contact - position_contact), 1.0e-4 * abs(position_contact))
    # A frozen drive row keeps exactly its position-solve impulse; an active one reacts.
    test.assertEqual(frozen_drive, position_drive)
    test.assertGreater(abs(active_drive - position_drive), 1.0e-4 * abs(position_drive))


class TestSteelVelocityDriveMode(unittest.TestCase):
    pass


add_function_test(
    TestSteelVelocityDriveMode,
    "test_frozen_drive_rows_keep_the_position_solve_impulse",
    test_frozen_drive_rows_keep_the_position_solve_impulse,
    devices=get_cuda_test_devices(),
)


class TestSteelAugmentedDrives(unittest.TestCase):
    def test_split_augmented_drive_pipeline_matches_combined_kernel(self):
        """Match combined drive forces and clear articulations without active drives."""
        device = wp.get_device()
        articulation_start = wp.array([0, 2], dtype=wp.int32, device=device)
        articulation_joint_end = wp.array([2], dtype=wp.int32, device=device)
        articulation_dof_count = wp.array([2], dtype=wp.int32, device=device)
        joint_type = wp.array([JointType.REVOLUTE, JointType.REVOLUTE], dtype=wp.int32, device=device)
        joint_parent = wp.array([-1, 0], dtype=wp.int32, device=device)
        joint_child = wp.array([0, 1], dtype=wp.int32, device=device)
        joint_articulation = wp.zeros(2, dtype=wp.int32, device=device)
        joint_start = wp.array([0, 1], dtype=wp.int32, device=device)
        joint_dof_dim = wp.array([[0, 1], [0, 1]], dtype=wp.int32, device=device)
        joint_f = wp.array([0.4, -0.2], dtype=wp.float32, device=device)
        joint_q = wp.array([0.25, -0.35], dtype=wp.float32, device=device)
        joint_qd = wp.array([0.1, -0.2], dtype=wp.float32, device=device)
        spring_stiffness = wp.array([1.5, 0.75], dtype=wp.float32, device=device)
        spring_ref = wp.array([0.0, 0.1], dtype=wp.float32, device=device)
        damping = wp.array([0.2, 0.3], dtype=wp.float32, device=device)
        joint_S_s = wp.array(
            [wp.spatial_vector(0.0, 0.0, 1.0, 0.0, 0.0, 0.0)] * 2,
            dtype=wp.spatial_vector,
            device=device,
        )
        body_fb_s = wp.array(
            [
                wp.spatial_vector(0.1, 0.2, 0.3, 1.0, 1.5, -0.5),
                wp.spatial_vector(-0.2, 0.4, -0.1, -0.5, 0.25, 0.75),
            ],
            dtype=wp.spatial_vector,
            device=device,
        )
        body_f_ext = wp.array(
            [
                wp.spatial_vector(0.05, 0.0, -0.1, 0.0, 0.25, 0.0),
                wp.spatial_vector(0.0, -0.05, 0.1, 0.15, 0.0, -0.2),
            ],
            dtype=wp.spatial_vector,
            device=device,
        )
        body_flags = wp.zeros(2, dtype=wp.int32, device=device)
        body_q = wp.array([wp.transform_identity()] * 2, dtype=wp.transform, device=device)
        body_com = wp.array([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0]], dtype=wp.vec3, device=device)
        articulation_origin = wp.array([[0.0, 0.0, 0.0]], dtype=wp.vec3, device=device)
        target_ke = wp.array([10.0, 20.0], dtype=wp.float32, device=device)
        target_kd = wp.array([1.0, 2.0], dtype=wp.float32, device=device)
        target_q = wp.array([0.1, -0.1], dtype=wp.float32, device=device)
        target_qd = wp.array([0.0, 0.05], dtype=wp.float32, device=device)
        effort_limit = wp.array([1.25, 2.5], dtype=wp.float32, device=device)

        sequential_body_ft = wp.zeros(2, dtype=wp.spatial_vector, device=device)
        sequential_tau = wp.zeros(2, dtype=wp.float32, device=device)
        wp.launch(
            eval_rigid_tau,
            dim=1,
            inputs=[
                articulation_start,
                articulation_joint_end,
                joint_type,
                joint_parent,
                joint_child,
                joint_articulation,
                joint_start,
                joint_start,
                joint_dof_dim,
                joint_f,
                joint_q,
                joint_qd,
                spring_stiffness,
                spring_ref,
                damping,
                joint_S_s,
                body_fb_s,
                body_f_ext,
                body_flags,
                body_q,
                body_com,
                articulation_origin,
                wp.ones(1, dtype=wp.int32, device=device),
            ],
            outputs=[sequential_body_ft, sequential_tau],
            device=device,
        )

        combined_body_ft = wp.zeros_like(sequential_body_ft)
        combined_tau = wp.zeros_like(sequential_tau)
        combined_counts = wp.zeros(1, dtype=wp.int32, device=device)
        combined_indices = wp.zeros(2, dtype=wp.int32, device=device)
        combined_stiffness = wp.zeros(2, dtype=wp.float32, device=device)
        wp.launch(
            eval_rigid_tau_and_augmented_drives,
            dim=1,
            inputs=[
                articulation_start,
                articulation_joint_end,
                articulation_dof_count,
                joint_type,
                joint_parent,
                joint_child,
                joint_articulation,
                joint_start,
                joint_start,
                joint_dof_dim,
                joint_f,
                joint_q,
                joint_qd,
                spring_stiffness,
                spring_ref,
                damping,
                joint_S_s,
                body_fb_s,
                body_f_ext,
                body_flags,
                body_q,
                body_com,
                articulation_origin,
                target_ke,
                target_kd,
                target_q,
                joint_start,
                target_qd,
                effort_limit,
                2,
                0.1,
                wp.ones(1, dtype=wp.int32, device=device),
            ],
            outputs=[
                combined_body_ft,
                combined_counts,
                combined_indices,
                combined_stiffness,
                combined_tau,
            ],
            device=device,
        )
        np.testing.assert_array_equal(combined_body_ft.numpy(), sequential_body_ft.numpy())
        expected_tau = sequential_tau.numpy() + np.array([-1.25, 2.5], dtype=np.float32)
        np.testing.assert_allclose(combined_tau.numpy(), expected_tau, rtol=0.0, atol=1.0e-6)
        np.testing.assert_array_equal(combined_counts.numpy(), [2])
        np.testing.assert_array_equal(combined_indices.numpy(), [0, 1])
        np.testing.assert_allclose(combined_stiffness.numpy(), [0.2, 0.4], rtol=0.0, atol=1.0e-7)

        prepared_counts = wp.zeros_like(combined_counts)
        prepared_indices = wp.zeros_like(combined_indices)
        prepared_stiffness = wp.zeros_like(combined_stiffness)
        prepared_tau = wp.zeros_like(combined_tau)
        wp.launch(
            prepare_augmented_joint_drives,
            dim=1,
            inputs=[
                articulation_start,
                articulation_dof_count,
                joint_type,
                joint_start,
                joint_start,
                joint_dof_dim,
                joint_q,
                joint_qd,
                target_ke,
                target_kd,
                target_q,
                joint_start,
                target_qd,
                effort_limit,
                2,
                0.1,
                wp.ones(1, dtype=wp.int32, device=device),
            ],
            outputs=[prepared_counts, prepared_indices, prepared_stiffness, prepared_tau],
            device=device,
        )
        np.testing.assert_array_equal(prepared_counts.numpy(), combined_counts.numpy())
        np.testing.assert_array_equal(prepared_indices.numpy(), combined_indices.numpy())
        np.testing.assert_array_equal(prepared_stiffness.numpy(), combined_stiffness.numpy())
        np.testing.assert_allclose(prepared_tau.numpy(), [-1.25, 2.5], rtol=0.0, atol=1.0e-6)

        topology_stiffness = wp.zeros_like(combined_stiffness)
        topology_tau = wp.zeros_like(combined_tau)
        wp.launch(
            _prepare_augmented_joint_drives_by_dof,
            dim=2,
            inputs=[
                wp.array([0, 1], dtype=wp.int32, device=device),
                wp.array([0, 1], dtype=wp.int32, device=device),
                wp.array([0, 1], dtype=wp.int32, device=device),
                joint_q,
                joint_qd,
                target_ke,
                target_kd,
                target_q,
                target_qd,
                effort_limit,
                0.1,
                wp.ones(2, dtype=wp.int32, device=device),
            ],
            outputs=[topology_stiffness, topology_tau],
            device=device,
        )
        np.testing.assert_array_equal(topology_stiffness.numpy(), combined_stiffness.numpy())
        np.testing.assert_allclose(topology_tau.numpy(), [-1.25, 2.5], rtol=0.0, atol=1.0e-6)

        additive_body_ft = wp.zeros_like(combined_body_ft)
        wp.launch(
            eval_rigid_tau_add,
            dim=1,
            inputs=[
                articulation_start,
                articulation_joint_end,
                joint_type,
                joint_parent,
                joint_child,
                joint_articulation,
                joint_start,
                joint_start,
                joint_dof_dim,
                joint_f,
                joint_q,
                joint_qd,
                spring_stiffness,
                spring_ref,
                damping,
                joint_S_s,
                body_fb_s,
                body_f_ext,
                body_flags,
                body_q,
                body_com,
                articulation_origin,
                wp.ones(1, dtype=wp.int32, device=device),
            ],
            outputs=[additive_body_ft, prepared_tau],
            device=device,
        )
        np.testing.assert_array_equal(additive_body_ft.numpy(), combined_body_ft.numpy())
        np.testing.assert_allclose(prepared_tau.numpy(), combined_tau.numpy(), rtol=0.0, atol=1.0e-6)

        passive_counts = wp.array([7], dtype=wp.int32, device=device)
        passive_tau = wp.array([3.0, -2.0], dtype=wp.float32, device=device)
        zero_target_ke = wp.zeros_like(target_ke)
        zero_target_kd = wp.zeros_like(target_kd)
        wp.launch(
            prepare_augmented_joint_drives,
            dim=1,
            inputs=[
                articulation_start,
                articulation_dof_count,
                joint_type,
                joint_start,
                joint_start,
                joint_dof_dim,
                joint_q,
                joint_qd,
                zero_target_ke,
                zero_target_kd,
                target_q,
                joint_start,
                target_qd,
                effort_limit,
                2,
                0.1,
                wp.ones(1, dtype=wp.int32, device=device),
            ],
            outputs=[passive_counts, prepared_indices, prepared_stiffness, passive_tau],
            device=device,
        )

        np.testing.assert_array_equal(passive_counts.numpy(), [0])
        np.testing.assert_array_equal(passive_tau.numpy(), [0.0, 0.0])


def _build_slider_augmented_drives_upstream(device, *, effort_limit=1.0e6, damping=0.0):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(mass=MASS, inertia=wp.mat33(np.eye(3)))
    joint = builder.add_joint_prismatic(
        -1,
        body,
        axis=newton.Axis.X,
        target_ke=KE,
        target_kd=KD,
        target_pos=TARGET_POS,
        target_vel=TARGET_VEL,
        armature=ARMATURE,
        effort_limit=effort_limit,
        damping=damping,
    )
    builder.add_articulation([joint])
    return builder.finalize(device=device)


def _reference_trajectory(steps, *, effort_limit=1.0e6, damping=0.0):
    """Backward-Euler PD drive: ``(m + a + dt kd + dt^2 ke) qdd = clamp(u0) - c qd``."""
    q = 0.0
    qd = 0.0
    history = []
    effective_mass = MASS + ARMATURE + DT * KD + DT * DT * KE
    for _ in range(steps):
        u0 = -(KE * (q - TARGET_POS + DT * qd) + KD * (qd - TARGET_VEL))
        u0 = float(np.clip(u0, -effort_limit, effort_limit))
        qdd = (u0 - damping * qd) / effective_mass
        qd = qd + DT * qdd
        q = q + DT * qd
        history.append((q, qd))
    return np.asarray(history)


def _solver_trajectory(model, steps, solve_mode):
    solver = SolverSteel(model, solve_mode=solve_mode)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    history = []
    for _ in range(steps):
        solver.step(state_0, state_1, control, None, DT)
        state_0, state_1 = state_1, state_0
        history.append((float(state_0.joint_q.numpy()[0]), float(state_0.joint_qd.numpy()[0])))
    return np.asarray(history)


def test_implicit_drive_matches_backward_euler_reference(test, device, solve_mode="matrix_free"):
    """Integrate the PD drive implicitly through the augmented mass matrix."""
    np.testing.assert_allclose(
        _solver_trajectory(_build_slider_augmented_drives_upstream(device), 120, solve_mode),
        _reference_trajectory(120),
        rtol=1.0e-4,
        atol=1.0e-5,
    )


def test_drive_force_is_clamped_to_the_effort_limit(test, device, solve_mode="matrix_free"):
    """Clamp the explicit drive force to the joint effort limit before the implicit solve."""
    reference = _reference_trajectory(120, effort_limit=5.0)
    np.testing.assert_allclose(
        _solver_trajectory(_build_slider_augmented_drives_upstream(device, effort_limit=5.0), 120, solve_mode),
        reference,
        rtol=1.0e-4,
        atol=1.0e-5,
    )
    test.assertGreater(float(np.abs(reference - _reference_trajectory(120)).max()), 1.0e-2, "the clamp must be active")


def test_passive_joint_damping_is_applied(test, device, solve_mode="matrix_free"):
    """Apply Model.joint_damping as a passive joint force."""
    np.testing.assert_allclose(
        _solver_trajectory(_build_slider_augmented_drives_upstream(device, damping=3.0), 120, solve_mode),
        _reference_trajectory(120, damping=3.0),
        rtol=1.0e-4,
        atol=1.0e-5,
    )


class TestSteelAugmentedDrivesUpstream(unittest.TestCase):
    pass


for _name, _func in (
    ("test_implicit_drive_matches_backward_euler_reference", test_implicit_drive_matches_backward_euler_reference),
    ("test_drive_force_is_clamped_to_the_effort_limit", test_drive_force_is_clamped_to_the_effort_limit),
    ("test_passive_joint_damping_is_applied", test_passive_joint_damping_is_applied),
):
    add_function_test(TestSteelAugmentedDrivesUpstream, _name, _func, devices=devices)
    add_function_test(
        TestSteelAugmentedDrivesUpstream, f"{_name}_split", _func, devices=split_devices, solve_mode="split"
    )


DT_joint_friction = 0.01


INERTIA = 0.5


FRICTION = 1.0


@unittest.skipUnless(wp.is_cuda_available(), "matrix-free joint friction requires CUDA")
class TestSteelJointFriction(unittest.TestCase):
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
                    expected = (
                        np.sign(effort) * (abs(effort) - FRICTION) / inertia * DT_joint_friction * np.arange(1, 21)
                    )
                    np.testing.assert_allclose(qd, expected, rtol=1.0e-5, atol=1.0e-6)

    def test_sliding_decelerates_and_stops_without_chatter(self):
        """Decelerate at friction / inertia, stop at the predicted step and stay at rest."""
        model = _single_dof_model("revolute")
        for qd0 in (1.0, -1.0):
            with self.subTest(qd0=qd0):
                qd = _run(model, qd0=qd0, steps=80)
                stop_step = round(abs(qd0) * INERTIA / FRICTION / DT_joint_friction)
                expected = np.sign(qd0) * np.maximum(
                    abs(qd0) - FRICTION / INERTIA * DT_joint_friction * np.arange(1, 81), 0.0
                )
                np.testing.assert_allclose(qd, expected, atol=1.0e-5)
                np.testing.assert_allclose(qd[stop_step:], 0.0, atol=1.0e-6)
                moving = qd[np.abs(qd) > 1.0e-6]
                self.assertTrue(np.all(np.sign(moving) == np.sign(qd0)))

    def test_substeps_preserve_physical_result(self):
        """Scale the impulse bound with dt so substepping keeps the same physical trajectory."""
        model = _single_dof_model("revolute")
        for substeps in (1, 4):
            with self.subTest(substeps=substeps):
                dt = DT_joint_friction / substeps
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
        np.testing.assert_allclose(qd[-1], [0.5 * DT_joint_friction * 10, 0.0], atol=1.0e-5)

    def test_locked_child_moves_with_composite_inertia(self):
        """Lock a stiff-friction child joint so the chain accelerates with its composite inertia."""
        length, mass = 0.4, 2.0
        model = _two_link_model(length=length, mass=mass, friction=(FRICTION, 50.0))
        qd = _run(model, effort=(3.0, 0.0), steps=10, dofs=2)
        composite = 2.0 * INERTIA + mass * length**2
        np.testing.assert_allclose(
            qd[:, 0], (3.0 - FRICTION) / composite * DT_joint_friction * np.arange(1, 11), rtol=1.0e-4
        )
        np.testing.assert_allclose(qd[:, 1], 0.0, atol=1.0e-5)

    def test_drive_settles_inside_friction_deadband(self):
        """Stop a PD-driven hinge where the drive effort falls inside the friction box."""
        ke, kd, target = 10.0, 2.0, 1.0
        for drive_mode in ("augmented", "rows"):
            with self.subTest(drive_mode=drive_mode):
                model = _single_dof_model("revolute", ke=ke, kd=kd, gravity=0.0)
                solver = _solver(model, drive_mode=drive_mode, iterations=32)
                states, control = [model.state(), model.state()], model.control()
                control.joint_target_q.assign([target])
                q, qd = _advance(solver, states, control, 400)
                self.assertLess(abs(float(qd[-1])), 1.0e-5)
                self.assertLessEqual(abs(float(q[-1]) - target), FRICTION / ke + 1.0e-3)
                self.assertGreater(abs(float(q[-1]) - target), 1.0e-2)
                reference = _solver(model, drive_mode=drive_mode, iterations=32, enable_joint_friction=False)
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
        for schedule in ("interleaved", "phased"):
            with self.subTest(schedule=schedule):
                solver = _solver(model, enable_joint_velocity_limits=True, row_schedule=schedule)
                states, control = [model.state(), model.state()], model.control()
                control.joint_f.assign([10.0])
                _, qd = _advance(solver, states, control, 20)
                self.assertLessEqual(float(np.max(qd)), 1.0 + 1.0e-5)
                self.assertAlmostEqual(float(qd[-1]), 1.0, delta=1.0e-5)

    def test_friction_and_ground_contact(self):
        """Hold a vertical slider against gravity, or let it fall onto the ground and rest, on both contact routes."""
        mass, half = 1.0, 0.1
        for response, (friction, falls) in itertools.product(
            ("immediate", "propagation_colored"), ((15.0, False), (5.0, True))
        ):
            with self.subTest(response=response, friction=friction):
                model = _vertical_slider_model(mass=mass, half=half, friction=friction)
                pipeline = newton.CollisionPipeline(model, rigid_contact_max=16)
                contacts = pipeline.contacts()
                solver = _solver(model, friction_anchor_beta=0.0, iterations=24, articulated_contact_response=response)
                states, control = [model.state(), model.state()], model.control()
                newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
                for _ in range(150):
                    pipeline.collide(states[0], contacts)
                    solver.step(states[0], states[1], control, contacts, DT_joint_friction)
                    states.reverse()
                z = float(states[0].joint_q.numpy()[0])
                if falls:
                    self.assertAlmostEqual(z, -0.4, delta=5.0e-3)
                    count = int(contacts.rigid_contact_count.numpy()[0])
                    self.assertGreater(count, 0)
                    # Dense rows (0) on the immediate route, propagation rows (2) on the colored route.
                    expected_path = 0 if response == "immediate" else 2
                    np.testing.assert_array_equal(solver._contact_path.numpy()[:count], expected_path)
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
        np.testing.assert_allclose(qd[-1], 0.5 / INERTIA * DT_joint_friction * 10, rtol=1.0e-5)

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

    def test_strided_coefficients_are_read_by_stride(self):
        """Read strided coefficient views at construction and after notified replacement."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        for _ in range(2):
            builder.add_world(_single_dof_builder("revolute", friction=0.0))
        expected = np.array([1.5 / INERTIA * DT_joint_friction, 0.5 / INERTIA * DT_joint_friction], dtype=np.float32)
        for strided in (False, True):
            for when in ("construction", "replacement"):
                with self.subTest(strided=strided, when=when):
                    model = builder.finalize()
                    storage = wp.array([1.0, 99.0, 2.0, 99.0], dtype=wp.float32, device=model.device)
                    coefficients = (
                        storage[::2] if strided else wp.array([1.0, 2.0], dtype=wp.float32, device=model.device)
                    )
                    if when == "construction":
                        model.joint_friction = coefficients
                    solver = _solver(model)
                    if when == "replacement":
                        model.joint_friction = coefficients
                        solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
                    states, control = [model.state(), model.state()], model.control()
                    control.joint_f.fill_(2.5)
                    _, qd = _advance(solver, states, control, 1)
                    np.testing.assert_allclose(qd, expected, atol=1.0e-6)

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
            for _ in range(2):
                solver.step(states[0], states[1], control, None, DT_joint_friction)
                solver.step(states[1], states[0], control, None, DT_joint_friction)
        replayed = []
        for _ in range(4):
            wp.capture_launch(capture.graph)
            replayed.append(states[0].joint_qd.numpy()[0])
        np.testing.assert_array_equal(np.array(replayed, dtype=np.float32), eager[3::4])

        model.joint_friction.assign([3.0])
        before = float(states[0].joint_qd.numpy()[0])
        wp.capture_launch(capture.graph)
        self.assertAlmostEqual(
            float(states[0].joint_qd.numpy()[0]), before - 0.5 / INERTIA * 4 * DT_joint_friction, delta=1.0e-5
        )

    def test_invalid_coefficients_build_no_rows(self):
        """Step an invalid coefficient as frictionless."""
        frictionless = _run(_single_dof_model("revolute", friction=0.0), effort=2.5, steps=5)
        for value in (float("inf"), float("nan"), -1.0):
            with self.subTest(value=value):
                model = _single_dof_model("revolute", friction=0.0)
                solver = _solver(model)
                states, control = [model.state(), model.state()], model.control()
                control.joint_f.assign([2.5])
                model.joint_friction.assign([value])
                _, qd = _advance(solver, states, control, 5)
                np.testing.assert_array_equal(qd, frictionless)

    def test_finite_coefficient_with_unrepresentable_bound_still_sticks(self):
        """Keep the row of a finite coefficient whose impulse bound overflows to infinity."""
        model = _single_dof_model("revolute", friction=float(np.finfo(np.float32).max))
        np.testing.assert_array_equal(_run(model, effort=2.5, steps=3, dt=2.0), 0.0)

    def test_captured_notify_replay_matches_eager(self):
        """Replay per-world friction writes, notify and steps bit-identically to the eager sequence."""
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
        for _ in range(2):
            builder.add_world(_single_dof_builder("revolute", friction=0.0))
        rng = np.random.default_rng(4)
        coefficients = rng.uniform(0.0, 3.0, (4, 2)).astype(np.float32)
        runs = []
        for captured in (False, True):
            model = builder.finalize()
            solver = _solver(model)
            states, control = [model.state(), model.state()], model.control()
            control.joint_f.assign([2.0, -2.0])
            states[0].joint_qd.assign([0.5, -0.5])
            _advance(solver, states, control, 2)

            def episode(solver=solver, states=states, control=control):
                solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
                for _ in range(2):
                    solver.step(states[0], states[1], control, None, DT_joint_friction)
                    solver.step(states[1], states[0], control, None, DT_joint_friction)

            if captured:
                with wp.ScopedCapture(device=model.device) as capture:
                    episode()
            qd = []
            for row in coefficients:
                model.joint_friction.assign(row)
                if captured:
                    wp.capture_launch(capture.graph)
                else:
                    episode()
                qd.append(states[0].joint_qd.numpy().copy())
            self.assertFalse(solver.joint_friction_invalid.numpy().any())
            runs.append(np.array(qd))
        np.testing.assert_array_equal(runs[1], runs[0])

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
        solver = _solver(model, articulation_constraint_max=1, warn_constraint_overflow=False)
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
        np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [0, 0])
        model.joint_friction.assign([0.0])
        solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
        _, qd = _advance(solver, states, control, 3)
        np.testing.assert_array_equal(solver._sleeping.body_awake.numpy(), [1, 1])
        self.assertGreater(abs(float(qd[-1])), 1.0e-2)

    def test_schedules_routes_and_velocity_pass_agree(self):
        """Give the same analytic result on every supported schedule and contact route."""
        model = _single_dof_model("revolute")
        expected = (2.5 - FRICTION) / INERTIA * DT_joint_friction * 10
        options = (
            {"row_schedule": "contact_then_internal"},
            {"row_schedule": "phased"},
            {"velocity_iterations": 2},
            {"contact_warmstart": True},
            {"drive_mode": "rows"},
            {"articulated_contact_response": "propagation"},
            {"articulated_contact_response": "propagation_colored"},
            {"articulated_contact_response": "propagation_fused"},
        )
        for kwargs in options:
            with self.subTest(**kwargs):
                route = _solver(model, **kwargs)
                if kwargs.get("articulated_contact_response") == "propagation_fused":
                    self.assertIsNotNone(route._propagation_full_fused_size)
                if kwargs.get("articulated_contact_response") == "propagation_colored":
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
            SolverSteel(_single_dof_model("revolute"), enable_joint_friction=True)
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
        "solve_mode": "matrix_free",
        "enable_joint_friction": True,
        "iterations": 16,
    }
    options.update(overrides)
    solver = SolverSteel(model, **options)
    solver._rigid_body_angular_damping.zero_()
    return solver


def _advance(solver, states, control, steps, dt=DT_joint_friction):
    q, qd = [], []
    for _ in range(steps):
        states[0].clear_forces()
        solver.step(states[0], states[1], control, None, dt)
        states.reverse()
        q.append(states[0].joint_q.numpy().copy())
        qd.append(states[0].joint_qd.numpy().copy())
    return np.squeeze(np.array(q)), np.squeeze(np.array(qd))


def _run(model, *, effort=0.0, qd0=0.0, steps, dt=DT_joint_friction, dofs=1, **overrides):
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
    unittest.main()
