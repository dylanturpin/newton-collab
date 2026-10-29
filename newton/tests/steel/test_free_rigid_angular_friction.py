# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental rolling and torsional friction rows on SolverSteel matrix-free free-rigid contacts."""

import gc
import math
import unittest
from typing import NamedTuple
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton._src.geometry.contact_reduction_body_pairs import build_reduction_groups
from newton._src.solvers.steel import solver_steel
from newton._src.solvers.steel.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_CONTACT_ROLLING,
    PGS_CONSTRAINT_TYPE_CONTACT_SPIN,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PROPAGATION_UNIT_META,
    PROPAGATION_UNIT_RING,
)
from newton._src.solvers.steel.solver_steel import (
    _get_pgs_solve_propagation_colored_block_kernel,
    _get_pgs_solve_propagation_colored_warp_kernel,
    _get_pgs_solve_propagation_contact_kernel,
)
from newton.solvers import SolverBase, SolverSteel
from newton.tests.steel._fixtures import (
    COLORED_VARIANTS,
    colored_variant_switches,
    select_colored_variant,
    set_torsion_radius,
)

_DT = 1.0 / 480.0
_RADIUS = 0.03
_MASS = 0.1
_GRAVITY = 9.81
_ROLLING = {"linear": (0.0, -0.3, 0.0), "angular": (10.0, 0.0, 0.0)}
# Principal moments 0.6, 1 and 1.4 with an x-z product: a physically valid anisotropic inertia.
_ANISOTROPIC = np.array([[1.0, 0.0, 0.4], [0.0, 1.0, 0.0], [0.4, 0.0, 1.0]])
_ON = {"free_rigid_torsional_friction": 0.0}


def _coulomb_stopping_distance(speed: float, rolling: float) -> float:
    """Return a solid sphere's rolling distance [m]: v^2 (1 + I / (m r^2)) = 2 d rolling g / r."""
    return speed**2 * 1.4 / (2.0 * rolling * _GRAVITY / _RADIUS)


def _scene(body: str, tilt_deg: float = 0.0, inertia=None, count: int = 1, anchor_deg=None):
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    SolverSteel.register_custom_attributes(builder)
    table = builder.add_body(xform=wp.transform((0.0, 0.0, -0.05), wp.quat_identity()), is_kinematic=True)
    builder.add_shape_box(table, hx=2.0, hy=2.0, hz=0.05)
    rotation = wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.radians(tilt_deg))
    bodies = []
    massless = newton.ModelBuilder.ShapeConfig(density=0.0)
    for index in range(count):
        x = 0.2 * index
        if body == "sphere":
            moments = np.eye(3) * 0.4 * _MASS * _RADIUS**2 if inertia is None else np.asarray(inertia)
            z = _RADIUS
        else:
            moments = np.eye(3) * _MASS * (2.0 * _RADIUS) ** 2 / 6.0
            tilt = math.radians(tilt_deg)
            z = _RADIUS * (math.cos(tilt) + math.sin(tilt)) + 1.0e-4
        inertia_matrix = wp.mat33(*np.asarray(moments, dtype=np.float32).ravel())
        pose = wp.transform((x, 0.0, z), rotation)
        if anchor_deg is None:
            obj = builder.add_body(xform=pose, mass=_MASS, inertia=inertia_matrix, lock_inertia=True)
        else:
            # A FREE root whose parent anchor is rotated: joint_qd is expressed in that rotated frame.
            obj = builder.add_link(xform=pose, mass=_MASS, inertia=inertia_matrix, lock_inertia=True)
            anchor = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.radians(anchor_deg))
            joint = builder.add_joint_free(obj, parent_xform=wp.transform((0.0, 0.0, 0.0), anchor))
            builder.add_articulation([joint])
        if body == "sphere":
            builder.add_shape_sphere(obj, radius=_RADIUS, cfg=massless)
        else:
            builder.add_shape_box(obj, hx=_RADIUS, hy=_RADIUS, hz=_RADIUS, cfg=massless)
        bodies.append(obj)
    return builder.finalize(), bodies


def _simulate(
    body,
    seconds,
    *,
    tilt_deg=0.0,
    linear=(0.0, 0.0, 0.0),
    angular=(0.0, 0.0, 0.0),
    graph=False,
    torsion_radius=0.0,
    contact_matching="disabled",
    inertia=None,
    steps=None,
    rolling_material=None,
    torsional_material=None,
    condim=None,
    count=1,
    support_angular=None,
    observe=None,
    parent_f=False,
    anchor_deg=None,
    on_solver=None,
    on_step=None,
    **solver_kwargs,
):
    model, bodies = _scene(body, tilt_deg, inertia, count, anchor_deg)
    if torsion_radius:
        set_torsion_radius(model, torsion_radius)
    if rolling_material is not None:
        model.shape_material_mu_rolling.fill_(rolling_material)
    if torsional_material is not None:
        model.shape_material_mu_torsional.fill_(torsional_material)
    if condim is not None:
        model.steel.condim.fill_(condim)
    options = {"solve_mode": "matrix_free", "iterations": 32, "deterministic": True, **solver_kwargs}
    solver = SolverSteel(model, **options)
    if on_solver is not None:
        on_solver(solver)
    pipeline = newton.CollisionPipeline(model, deterministic=True, contact_matching=contact_matching)
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    qd = state_0.joint_qd.numpy()
    for obj in bodies:
        joint = int(np.flatnonzero(model.joint_child.numpy() == obj)[0])
        dof = int(model.joint_qd_start.numpy()[joint])
        qd[dof : dof + 3] = linear
        qd[dof + 3 : dof + 6] = angular
    if support_angular is not None:
        joint = int(np.flatnonzero(model.joint_child.numpy() == 0)[0])
        dof = int(model.joint_qd_start.numpy()[joint])
        qd[dof + 3 : dof + 6] = support_angular
    state_0.joint_qd.assign(qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    if anchor_deg is not None:
        # Initial velocities are world-frame twists; recover the joint coordinates of the rotated anchor.
        body_qd = state_0.body_qd.numpy()
        for obj in bodies:
            body_qd[obj] = (*linear, *angular)
        state_0.body_qd.assign(body_qd)
        newton.eval_ik(model, state_0, state_0.joint_q, state_0.joint_qd)
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    control = model.control()
    obj = bodies[0]
    start = state_0.body_q.numpy()[obj][:3].copy()

    observables = solver.observables({SolverBase.ObservableKind.BODY_PARENT_F}) if parent_f else None

    def step(state_in, state_out):
        state_in.clear_forces()
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, _DT, observables=observables)

    if steps is not None:
        for _ in range(steps):
            step(state_0, state_1)
            state_0, state_1 = state_1, state_0
            if on_step is not None:
                on_step(state_0.body_q.numpy()[obj], state_0.body_qd.numpy()[obj])
    elif graph:
        with wp.ScopedCapture(device=model.device) as capture:
            step(state_0, state_1)
            step(state_1, state_0)
        for _ in range(int(seconds / _DT) // 2):
            wp.capture_launch(capture.graph)
    else:
        for _ in range(int(seconds / _DT) // 2):
            step(state_0, state_1)
            step(state_1, state_0)
    if observe is not None:
        observe(solver, contacts, observables)
    pose, twist = state_0.body_q.numpy()[obj], state_0.body_qd.numpy()[obj]
    return pose, twist, float(np.linalg.norm(pose[:2] - start[:2]))


@unittest.skipUnless(wp.is_cuda_available(), "Steel matrix-free requires CUDA")
class TestFreeRigidAngularFriction(unittest.TestCase):
    def test_rolling_sphere_stops_at_the_coulomb_distance(self):
        """Stop a rolling sphere within 5% of the analytic Coulomb distance; keep it rolling when off."""
        _, twist_free, travel_free = _simulate("sphere", 1.0, **_ROLLING)
        self.assertGreater(np.linalg.norm(twist_free[3:6]), 9.0)
        self.assertGreater(travel_free, 0.25)
        for rolling in (0.001, 0.002):
            with self.subTest(rolling=rolling):
                _, twist, travel = _simulate("sphere", 1.5, free_rigid_rolling_friction=rolling, **_ON, **_ROLLING)
                self.assertLess(np.linalg.norm(twist[3:6]), 1.0e-3)
                expected = _coulomb_stopping_distance(0.3, rolling)
                self.assertAlmostEqual(travel, expected, delta=0.05 * expected)

    def test_condim_and_materials_match_the_override(self):
        """Read the shape materials where steel:condim is 6, as the solver-wide override does everywhere."""
        material, _, _ = _simulate("sphere", 0.5, condim=6, rolling_material=0.002, torsional_material=0.0, **_ROLLING)
        override, _, _ = _simulate(
            "sphere", 0.5, rolling_material=0.0, free_rigid_rolling_friction=0.002, **_ON, **_ROLLING
        )
        np.testing.assert_array_equal(material, override)

    def test_condim_selects_the_rows(self):
        """Add a spin row at condim 4 and a rolling pair at 6, from the builder's default coefficients."""
        for condim, expected in ((3, 3), (4, 4), (6, 6)):
            seen = {}

            def observe(solver, contacts, observables, seen=seen):
                seen["slots"] = int(solver._contact_slots_needed.numpy()[0])

            with self.subTest(condim=condim):
                _simulate("sphere", 0.0, steps=1, condim=condim, observe=observe)
                self.assertEqual(seen["slots"], expected)

    def test_spin_row_stops_a_spinning_sphere_at_the_coulomb_rate(self):
        """Decelerate a sphere spinning on its contact point at mu_s m g / I, and stop it."""
        torsional = 5.0e-4
        options = {
            "free_rigid_torsional_friction": torsional,
            "free_rigid_rolling_friction": 0.0,
        }
        _, free, _ = _simulate("sphere", 0.2, angular=(0.0, 0.0, 6.0))
        self.assertGreater(free[5], 5.9)
        rate = torsional * _MASS * _GRAVITY / (0.4 * _MASS * _RADIUS**2)
        _, twist, _ = _simulate("sphere", 0.2, angular=(0.0, 0.0, 6.0), **options)
        self.assertAlmostEqual(twist[5], 6.0 - rate * 0.2, delta=0.1)
        _, twist, _ = _simulate("sphere", 0.6, angular=(0.0, 0.0, 6.0), **options)
        self.assertLess(abs(twist[5]), 1.0e-3)

    def test_rows_never_add_energy(self):
        """End a contact step with no more kinetic energy than it started with, for anisotropic inertias."""
        rng = np.random.default_rng(3)
        cases = [(_ANISOTROPIC * 4.0e-5, (1.0, 0.0, -3.0), rolling) for rolling in (1.0, 1.0e-3, 1.0e-4)]
        for _ in range(6):
            axes, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            moments = np.sort(rng.uniform(0.2, 1.0, 3))
            moments[2] = min(moments[2], moments[0] + moments[1] - 1.0e-3)
            cases.append((axes @ np.diag(moments) @ axes.T * 4.0e-5, tuple(rng.normal(size=3) * 3.0), 1.0e-3))

        for index, (inertia, omega, rolling) in enumerate(cases):

            def energy(twist, inertia=inertia):
                return 0.5 * _MASS * float(twist[:3] @ twist[:3]) + 0.5 * float(twist[3:6] @ inertia @ twist[3:6])

            with self.subTest(case=index, rolling=rolling):
                # Settle at rest first so the contact carries the weight, then spin it for one step.
                kwargs = {"angular": omega, "inertia": inertia, "steps": 1}
                _, twist_free, _ = _simulate("sphere", 0.0, **kwargs)
                _, twist, _ = _simulate(
                    "sphere",
                    0.0,
                    free_rigid_rolling_friction=rolling,
                    free_rigid_torsional_friction=rolling,
                    **kwargs,
                )
                start = 0.5 * float(np.asarray(omega) @ inertia @ np.asarray(omega))
                # Gravity does at most m g |v_z| dt of work in one step.
                self.assertLessEqual(energy(twist), start + _MASS * _GRAVITY * 1.0e-3 * _DT)
                self.assertLessEqual(energy(twist), energy(twist_free) * (1.0 + 1.0e-4))

    def test_spin_is_relative_to_a_turning_support(self):
        """Spin a resting sphere up to a kinematic turntable's rate at mu_s m g / I; leave it still when off."""
        torsional = 5.0e-4
        table = (0.0, 0.0, 2.0)
        _, free, _ = _simulate("sphere", 0.5, support_angular=table)
        self.assertLess(abs(free[5]), 1.0e-3)
        options = {"free_rigid_torsional_friction": torsional, "free_rigid_rolling_friction": 0.0}
        rate = torsional * _GRAVITY / (0.4 * _RADIUS**2)
        _, twist, _ = _simulate("sphere", 0.1, support_angular=table, **options)
        self.assertAlmostEqual(twist[5], rate * 0.1, delta=0.05)
        _, twist, _ = _simulate("sphere", 0.5, support_angular=table, **options)
        self.assertAlmostEqual(twist[5], 2.0, delta=1.0e-3)

    def test_rows_act_in_world_frame_under_a_rotated_free_root_anchor(self):
        """Match the identity-anchor result with a FREE root parent anchor rotated by 45 and 90 degrees about y.

        A sphere spinning about the contact normal keeps its spin under rolling rows, rolls about the same world axis
        to the same stop, and slows its spin at the same rate under the spin row, eagerly and in a captured graph.
        """
        cases = {
            "normal spin, rolling rows": {
                "seconds": 0.5,
                "angular": (0.0, 0.0, 6.0),
                "free_rigid_rolling_friction": 0.01,
            },
            "rolling stop": {"seconds": 1.5, "free_rigid_rolling_friction": 0.002, **_ROLLING},
            "spin row": {"seconds": 0.2, "angular": (0.0, 0.0, 6.0), "free_rigid_torsional_friction": 5.0e-4},
        }
        for name, case in cases.items():
            kwargs = {"free_rigid_torsional_friction": 0.0, "free_rigid_rolling_friction": 0.0, **case}
            _, reference, travel = _simulate("sphere", anchor_deg=0.0, **kwargs)
            if name == "normal spin, rolling rows":
                np.testing.assert_allclose(reference[3:6], (0.0, 0.0, 6.0 * (1.0 - 0.05 * _DT) ** 240), atol=1.0e-3)
            for anchor in (45.0, 90.0):
                for graph in (False, True):
                    with self.subTest(case=name, anchor=anchor, graph=graph):
                        _, twist, rotated_travel = _simulate("sphere", anchor_deg=anchor, graph=graph, **kwargs)
                        np.testing.assert_allclose(twist, reference, atol=2.0e-4)
                        self.assertAlmostEqual(rotated_travel, travel, delta=1.0e-4)

    def test_patch_angular_rows_do_not_depend_on_contact_order(self):
        """Keep each material's angular rows when two shapes of one body with different angular coefficients rest
        side by side, whichever shape's contacts come first."""
        results = []
        for order in ((3, 6), (6, 3)):
            builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
            SolverSteel.register_custom_attributes(builder)
            table = builder.add_body(xform=wp.transform((0.0, 0.0, -0.05), wp.quat_identity()), is_kinematic=True)
            builder.add_shape_box(table, hx=1.0, hy=1.0, hz=0.05)
            inertia = wp.mat33(*(np.eye(3) * 1.0e-4).astype(np.float32).ravel())
            body = builder.add_body(
                xform=wp.transform((0.0, 0.0, 0.0201), wp.quat_identity()),
                mass=_MASS,
                inertia=inertia,
                lock_inertia=True,
            )
            shapes = {}
            for condim in order:
                x = -0.02 if condim == 3 else 0.02
                shapes[condim] = builder.add_shape_box(
                    body,
                    xform=wp.transform((x, 0.0, 0.0), wp.quat_identity()),
                    hx=0.02,
                    hy=0.02,
                    hz=0.02,
                    cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu_torsional=0.01, mu_rolling=0.002),
                    custom_attributes={"steel:condim": condim},
                )
            model = builder.finalize()
            solver = SolverSteel(model, solve_mode="matrix_free", iterations=32, friction_anchor_beta=0.2)
            pipeline = newton.CollisionPipeline(model)
            contacts = pipeline.contacts()
            state_0, state_1, control = model.state(), model.state(), model.control()
            for _ in range(3):
                state_0.clear_forces()
                pipeline.collide(state_0, contacts)
                solver.step(state_0, state_1, control, contacts, _DT)
                state_0, state_1 = state_1, state_0
            count = int(contacts.rigid_contact_count.numpy()[0])
            shape0 = contacts.rigid_contact_shape0.numpy()[:count]
            shape1 = contacts.rigid_contact_shape1.numpy()[:count]
            slots = solver._contact_slots_needed.numpy()[:count]
            angular = {}
            for condim, shape in shapes.items():
                mask = (shape0 == shape) | (shape1 == shape)
                angular[condim] = int(np.count_nonzero(slots[mask] == 6))
            results.append(angular)
        for angular in results:
            self.assertEqual(angular[3], 0)
            self.assertEqual(angular[6], 1)

    def test_articulated_contacts_group_patches_regardless_of_condim(self):
        """Group an articulated body's patches as without angular rows: mixed condim matches the condim 3 control."""
        seen = {}
        for mixed in (False, True):
            builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
            SolverSteel.register_custom_attributes(builder)
            table = builder.add_body(xform=wp.transform((0.0, 0.0, -0.05), wp.quat_identity()), is_kinematic=True)
            builder.add_shape_box(table, hx=1.0, hy=1.0, hz=0.05)
            # A free sphere elsewhere keeps the matrix-free route, and with it the angular rows, in use.
            sphere_inertia = wp.mat33(*(np.eye(3) * 0.4 * _MASS * _RADIUS**2).astype(np.float32).ravel())
            free = builder.add_body(
                xform=wp.transform((0.5, 0.5, _RADIUS), wp.quat_identity()),
                mass=_MASS,
                inertia=sphere_inertia,
                lock_inertia=True,
            )
            builder.add_shape_sphere(free, radius=_RADIUS, cfg=newton.ModelBuilder.ShapeConfig(density=0.0))
            inertia = wp.mat33(*(np.eye(3) * 1.0e-4).astype(np.float32).ravel())
            link = builder.add_link(mass=_MASS, inertia=inertia, lock_inertia=True)
            joint = builder.add_joint_prismatic(
                -1, link, parent_xform=wp.transform((0.0, 0.0, 0.0201), wp.quat_identity()), axis=newton.Axis.X
            )
            builder.add_articulation([joint])
            for x, condim in ((-0.02, 3), (0.02, 6 if mixed else 3)):
                builder.add_shape_box(
                    link,
                    xform=wp.transform((x, 0.0, 0.0), wp.quat_identity()),
                    hx=0.02,
                    hy=0.02,
                    hz=0.02,
                    cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu_torsional=0.01, mu_rolling=0.002),
                    custom_attributes={"steel:condim": condim},
                )
            model = builder.finalize()
            solver = SolverSteel(
                model,
                solve_mode="matrix_free",
                iterations=32,
                friction_anchor_beta=0.2,
                free_rigid_rolling_friction=0.001 if mixed else None,
            )
            pipeline = newton.CollisionPipeline(model, deterministic=True)
            contacts = pipeline.contacts()
            state_0, state_1, control = model.state(), model.state(), model.control()
            for _ in range(3):
                state_0.clear_forces()
                pipeline.collide(state_0, contacts)
                solver.step(state_0, state_1, control, contacts, _DT)
                state_0, state_1 = state_1, state_0
            count = int(contacts.rigid_contact_count.numpy()[0])
            dense = solver._contact_path.numpy()[:count] == 0
            seen[mixed] = (
                solver._contact_slots_needed.numpy()[:count][dense].tolist(),
                solver._friction_patches.view.weight.numpy()[:count][dense].tolist(),
                solver._constraint_count.numpy().tolist(),
            )
        self.assertTrue(len(seen[False][0]) > 0)
        self.assertEqual(seen[True], seen[False])

    def test_toppling_and_sliding_cube_are_unchanged(self):
        """Leave a toppling or sliding cube within 2 mm of the option-off pose; the spinning cube slows at most."""
        for kwargs in ({"tilt_deg": 40.0}, {"linear": (0.5, 0.0, 0.0)}, {"angular": (0.0, 0.0, 6.0)}):
            with self.subTest(**kwargs):
                pose_free, _, _ = _simulate("box", 1.0, **kwargs)
                pose, _, _ = _simulate("box", 1.0, free_rigid_rolling_friction=0.002, **_ON, **kwargs)
                np.testing.assert_allclose(pose[:3], pose_free[:3], atol=2.0e-3)
                self.assertGreater(abs(float(np.dot(pose[3:7], pose_free[3:7]))), 0.999)

    def test_composes_with_sleeping_anchors_torsion_warm_start_and_velocity_iterations(self):
        """Keep the Coulomb stopping distance with each production-style option."""
        expected = _coulomb_stopping_distance(0.3, 0.002)
        for options in (
            {"enable_sleeping": True},
            {"friction_anchor_beta": 0.2},
            {"torsion_radius": 0.01},
            {"enable_sleeping": True, "friction_anchor_beta": 0.2, "torsion_radius": 0.01},
            {"contact_warmstart": True, "contact_matching": "sticky"},
            {"velocity_iterations": 1},
        ):
            with self.subTest(**options):
                _, twist, travel = _simulate(
                    "sphere", 1.5, free_rigid_rolling_friction=0.002, **_ON, **_ROLLING, **options
                )
                self.assertLess(np.linalg.norm(twist[3:6]), 1.0e-3)
                self.assertAlmostEqual(travel, expected, delta=0.05 * expected)

    def test_patch_carries_one_set_of_angular_rows(self):
        """Give a friction patch's first anchor the angular rows with the whole coefficient; the other keeps sliding only."""
        seen = {}

        def observe(solver, contacts, observables):
            count = int(contacts.rigid_contact_count.numpy()[0])
            slots = solver._contact_slots_needed.numpy()[:count]
            slot = solver._contact_slot.numpy()[:count]
            types = solver._mf_row_type.numpy()[0]
            mu = solver._mf_row_mu.numpy()[0]
            anchors = [int(s) for s, n in zip(slot, slots, strict=True) if n >= 3]
            angular = [int(s) for s, n in zip(slot, slots, strict=True) if n == 6]
            seen["anchors"], seen["angular"] = len(anchors), len(angular)
            for s in angular:
                self.assertEqual(types[s], PGS_CONSTRAINT_TYPE_CONTACT)
                self.assertEqual(types[s + 1], PGS_CONSTRAINT_TYPE_FRICTION)
                self.assertEqual(types[s + 3], PGS_CONSTRAINT_TYPE_CONTACT_ROLLING)
                self.assertEqual(types[s + 4], PGS_CONSTRAINT_TYPE_CONTACT_ROLLING)
                self.assertEqual(types[s + 5], PGS_CONSTRAINT_TYPE_CONTACT_SPIN)
                self.assertAlmostEqual(mu[s + 3], 0.002, places=7)
                self.assertAlmostEqual(mu[s + 5], 0.001, places=7)
                self.assertAlmostEqual(mu[s + 1] * len(anchors), 1.0, places=6)

        _simulate(
            "box",
            0.0,
            steps=3,
            friction_anchor_beta=0.2,
            free_rigid_rolling_friction=0.002,
            free_rigid_torsional_friction=0.001,
            observe=observe,
        )
        self.assertEqual((seen["anchors"], seen["angular"]), (2, 1))

    def test_repeated_and_captured_runs_are_bitwise_identical(self):
        """Match two eager runs and a CUDA-graph replay bit for bit, with several worlds of bodies."""
        kwargs = {
            "free_rigid_rolling_friction": 0.001,
            "free_rigid_torsional_friction": 0.001,
            "friction_anchor_beta": 0.2,
            "enable_sleeping": True,
            "count": 3,
            **_ON,
            **_ROLLING,
        }
        kwargs["free_rigid_torsional_friction"] = 0.001
        first, _, _ = _simulate("sphere", 0.3, **kwargs)
        second, _, _ = _simulate("sphere", 0.3, **kwargs)
        captured, _, _ = _simulate("sphere", 0.3, graph=True, **kwargs)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(first, captured)

    def test_condim_rows_keep_the_larger_coefficient_under_every_friction_combine(self):
        """Combine sliding friction by friction_combine while condim rows keep the larger contributed coefficient."""
        mu = np.array([0.5, 0.8], dtype=np.float32)
        torsional = np.array([0.002, 0.004], dtype=np.float32)
        rolling = np.array([0.003, 0.001], dtype=np.float32)
        sliding = {"average": (mu[0] + mu[1]) / np.float32(2.0), "min": mu[0], "multiply": mu[0] * mu[1], "max": mu[1]}
        for mode, expected_mu in sliding.items():
            seen = {}

            def on_solver(solver):
                solver.model.shape_material_mu.assign(mu)
                solver.model.shape_material_mu_torsional.assign(torsional)
                solver.model.shape_material_mu_rolling.assign(rolling)
                solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)

            def observe(solver, contacts, observables, seen=seen):
                count = int(contacts.rigid_contact_count.numpy()[0])
                needed = solver._contact_slots_needed.numpy()[:count]
                slots = solver._contact_slot.numpy()[:count]
                row = int(slots[int(np.flatnonzero(needed == 6)[0])])
                types = solver._mf_row_type.numpy()[0]
                self.assertEqual(types[row + 3], PGS_CONSTRAINT_TYPE_CONTACT_ROLLING)
                self.assertEqual(types[row + 5], PGS_CONSTRAINT_TYPE_CONTACT_SPIN)
                seen["mu"] = solver._mf_row_mu.numpy()[0][[row, row + 3, row + 5]]

            with self.subTest(mode=mode):
                _simulate("sphere", 0.0, steps=1, condim=6, friction_combine=mode, on_solver=on_solver, observe=observe)
                np.testing.assert_array_equal(seen["mu"], [expected_mu, rolling.max(), torsional.max()])

    def test_one_override_keeps_the_other_coefficient_on_condim(self):
        """Override only the rolling coefficient: shapes at condim 3 get the rolling pair and no spin row."""
        seen = {}

        def observe(solver, contacts, observables):
            seen["slots"] = int(solver._contact_slots_needed.numpy()[0])

        _simulate("sphere", 0.0, steps=1, free_rigid_rolling_friction=0.001, observe=observe)
        self.assertEqual(seen["slots"], 5)
        _simulate("sphere", 0.0, steps=1, condim=4, free_rigid_rolling_friction=0.001, observe=observe)
        self.assertEqual(seen["slots"], 6)

    def test_shape_notify_inside_graph_capture_refreshes_coefficients(self):
        """Capture a shape-property notify that changes mu_rolling, and replay it into the rows' bound."""
        model, _ = _scene("sphere")
        model.steel.condim.fill_(6)
        model.shape_material_mu_rolling.fill_(0.001)
        solver = SolverSteel(model, solve_mode="matrix_free")
        with wp.ScopedCapture(device=model.device) as capture:
            solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
        model.shape_material_mu_rolling.fill_(0.003)
        wp.capture_launch(capture.graph)
        np.testing.assert_allclose(solver._free_rigid_mu_rolling.numpy()[: model.shape_count], 0.003)

    def test_default_condim_and_zero_coefficients_match_the_base(self):
        """Match the default bit for bit at condim 3, and add no rows for zero coefficients at condim 6."""
        default, _, _ = _simulate("sphere", 0.3, **_ROLLING)
        condim3, _, _ = _simulate("sphere", 0.3, condim=3, **_ROLLING)
        zero, _, _ = _simulate("sphere", 0.3, condim=6, rolling_material=0.0, torsional_material=0.0, **_ROLLING)
        np.testing.assert_array_equal(default, condim3)
        np.testing.assert_array_equal(default, zero)

    def test_inactive_friction_zeroes_both_rows_of_each_pair(self):
        """Hold both rows of a sliding and a rolling pair at zero until friction activates, then solve both."""
        model, _ = _scene("sphere")
        solver = SolverSteel(
            model,
            solve_mode="matrix_free",
            iterations=32,
            deterministic=True,
            contact_warmstart=True,
            free_rigid_rolling_friction=0.002,
            **_ON,
        )
        pipeline = newton.CollisionPipeline(model, deterministic=True, contact_matching="sticky")
        contacts = pipeline.contacts()
        states = [model.state(), model.state()]
        qd = states[0].joint_qd.numpy()
        qd[:3] = (0.4, 0.25, 0.0)
        states[0].joint_qd.assign(qd)
        newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
        control = model.control()

        def pair_impulses():
            states[0].clear_forces()
            pipeline.collide(states[0], contacts)
            solver.step(states[0], states[1], control, contacts, _DT)
            states.reverse()
            count = int(solver._mf_constraint_count.numpy()[0])
            row_type = solver._mf_row_type.numpy()[0, :count]
            impulses = solver._mf_impulses.numpy()[0, :count]
            return impulses[row_type == PGS_CONSTRAINT_TYPE_FRICTION], impulses[
                row_type == PGS_CONSTRAINT_TYPE_CONTACT_ROLLING
            ]

        for _ in range(4):
            sliding, rolling = pair_impulses()
        self.assertEqual((sliding.size, rolling.size), (2, 2))
        self.assertTrue(np.all(sliding != 0.0))
        # The constructor rejects delayed friction with warm starting; set it directly so the second tangent row
        # starts the solve with a carried impulse that only the zeroing of inactive pairs clears.
        solver._contact_friction_position_iterations = 0
        sliding, rolling = pair_impulses()
        np.testing.assert_array_equal(sliding, 0.0)
        np.testing.assert_array_equal(rolling, 0.0)
        solver._contact_friction_position_iterations = 8
        sliding, rolling = pair_impulses()
        self.assertTrue(np.all(sliding != 0.0))
        self.assertTrue(np.all(rolling != 0.0))

    def test_body_parent_f_balances_the_rolling_torque(self):
        """Keep a rolling free sphere's parent wrench at zero: its contact wrench carries the rolling torque."""
        seen = {}

        def observe(solver, contacts, observables):
            slot = int(solver._contact_slot.numpy()[0])
            seen["rolling"] = float(np.linalg.norm(solver._mf_impulses.numpy()[0, slot + 3 : slot + 5])) / _DT
            seen["parent"] = observables.body_parent_f.numpy()[1]

        _simulate(
            "sphere", 0.0, steps=4, parent_f=True, free_rigid_rolling_friction=0.002, **_ON, observe=observe, **_ROLLING
        )
        torque = 0.002 * _MASS * _GRAVITY
        self.assertAlmostEqual(seen["rolling"], torque, delta=0.05 * torque)
        self.assertLess(float(np.linalg.norm(seen["parent"][3:])), 0.02 * torque)

    def test_invalid_options_raise(self):
        """Reject bad coefficients and condim, unsupported solve modes, and a condim raise without rebuilding."""
        model, _ = _scene("sphere")
        for value in (-1.0e-3, float("nan"), float("inf")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                SolverSteel(model, free_rigid_rolling_friction=value)
        with self.assertRaises(ValueError):
            SolverSteel(model, solve_mode="split", free_rigid_rolling_friction=1.0e-3)
        with self.assertRaises(ValueError):
            # Point friction: bisection with persistent patches would warn before the rejection under test.
            SolverSteel(
                model,
                solve_mode="matrix_free",
                friction_mode="bisection",
                friction_anchor_beta=0.0,
                free_rigid_rolling_friction=1.0e-3,
            )
        model.steel.condim.fill_(7)
        with self.assertRaises(ValueError):
            SolverSteel(model, solve_mode="matrix_free")
        model.steel.condim.fill_(3)
        solver = SolverSteel(model, solve_mode="matrix_free")
        model.steel.condim.fill_(6)
        with self.assertRaises(ValueError):
            solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)


_PROPAGATION = {"articulated_contact_response": "propagation", "deterministic": False}
_RESPONSES = {"immediate": {}, "propagation": _PROPAGATION}
# Matrix-free (1) on the immediate response; free-free routing moves the same rows to propagation (2).
_ROUTE = {"immediate": 1, "propagation": 2}


def _row_arrays(solver, response):
    """Return the row family arrays that carry free-rigid contacts on ``response``."""
    if response == "immediate":
        return solver._mf_row_type, solver._mf_row_parent, solver._mf_row_mu, solver._mf_impulses
    return (
        solver._propagation_row_type,
        solver._propagation_row_parent,
        solver._propagation_row_mu,
        solver._propagation_impulses,
    )


@unittest.skipUnless(wp.is_cuda_available(), "Steel matrix-free requires CUDA")
class TestFreeRigidAngularFrictionPropagation(unittest.TestCase):
    """The rows on the plain propagation response, which routes free-free contacts to propagation rows."""

    def test_rows_take_each_response_route(self):
        """Lay out normal, sliding pair, rolling pair and spin on the route each response gives the contact."""
        for response, options in _RESPONSES.items():
            for torsional, rolling, slots in ((0.001, 0.0, 4), (0.0, 0.002, 5), (0.001, 0.002, 6)):
                seen = {}

                def observe(solver, contacts, observables, seen=seen, response=response, slots=slots):
                    count = int(contacts.rigid_contact_count.numpy()[0])
                    seen["path"] = solver._contact_path.numpy()[:count].tolist()
                    seen["slots"] = solver._contact_slots_needed.numpy()[:count].tolist()
                    row_type, row_parent, row_mu, _ = _row_arrays(solver, response)
                    slot = int(solver._contact_slot.numpy()[0])
                    seen["types"] = row_type.numpy()[0, slot : slot + slots].tolist()
                    seen["parents"] = row_parent.numpy()[0, slot + 1 : slot + slots].tolist()
                    seen["mu"] = row_mu.numpy()[0, slot + 3 : slot + slots].tolist()
                    seen["slot"] = slot

                with self.subTest(response=response, torsional=torsional, rolling=rolling):
                    _simulate(
                        "sphere",
                        0.0,
                        steps=2,
                        free_rigid_torsional_friction=torsional,
                        free_rigid_rolling_friction=rolling,
                        observe=observe,
                        **options,
                    )
                    self.assertEqual(seen["path"], [_ROUTE[response]])
                    self.assertEqual(seen["slots"], [slots])
                    angular = [PGS_CONSTRAINT_TYPE_CONTACT_ROLLING] * (2 if rolling else 0)
                    angular += [PGS_CONSTRAINT_TYPE_CONTACT_SPIN] if torsional else []
                    expected = [PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION, PGS_CONSTRAINT_TYPE_FRICTION]
                    self.assertEqual(seen["types"], expected + angular)
                    self.assertEqual(seen["parents"], [seen["slot"]] * (slots - 1))
                    coefficients = [rolling] * (2 if rolling else 0) + ([torsional] if torsional else [])
                    np.testing.assert_allclose(seen["mu"], coefficients, rtol=1.0e-6)

    def test_rolling_sphere_stops_at_the_coulomb_distance(self):
        """Stop a rolling sphere within 5% of the analytic Coulomb distance on the propagation rows."""
        for rolling in (0.001, 0.002):
            with self.subTest(rolling=rolling):
                _, twist, travel = _simulate(
                    "sphere", 1.5, free_rigid_rolling_friction=rolling, **_ON, **_ROLLING, **_PROPAGATION
                )
                self.assertLess(np.linalg.norm(twist[3:6]), 1.0e-3)
                expected = _coulomb_stopping_distance(0.3, rolling)
                self.assertAlmostEqual(travel, expected, delta=0.05 * expected)

    def test_spin_row_stops_a_spinning_sphere_at_the_coulomb_rate(self):
        """Decelerate a spinning sphere at mu_s m g / I on the propagation rows, and stop it."""
        torsional = 5.0e-4
        options = {"free_rigid_torsional_friction": torsional, "free_rigid_rolling_friction": 0.0, **_PROPAGATION}
        rate = torsional * _MASS * _GRAVITY / (0.4 * _MASS * _RADIUS**2)
        _, twist, _ = _simulate("sphere", 0.2, angular=(0.0, 0.0, 6.0), **options)
        self.assertAlmostEqual(twist[5], 6.0 - rate * 0.2, delta=0.1)
        _, twist, _ = _simulate("sphere", 0.6, angular=(0.0, 0.0, 6.0), **options)
        self.assertLess(abs(twist[5]), 1.0e-3)

    def test_spin_is_relative_to_a_turning_support(self):
        """Spin a resting sphere up to a kinematic turntable's rate at mu_s m g / I on the propagation rows."""
        torsional = 5.0e-4
        options = {"free_rigid_torsional_friction": torsional, "free_rigid_rolling_friction": 0.0, **_PROPAGATION}
        rate = torsional * _GRAVITY / (0.4 * _RADIUS**2)
        _, twist, _ = _simulate("sphere", 0.1, support_angular=(0.0, 0.0, 2.0), **options)
        self.assertAlmostEqual(twist[5], rate * 0.1, delta=0.05)
        _, twist, _ = _simulate("sphere", 0.5, support_angular=(0.0, 0.0, 2.0), **options)
        self.assertAlmostEqual(twist[5], 2.0, delta=1.0e-3)

    def test_rows_stay_within_their_bounds(self):
        """Bound the rolling pair by mu_r and the spin row by mu_s times the contact's normal impulse."""
        seen = {}

        def observe(solver, contacts, observables):
            slot = int(solver._contact_slot.numpy()[0])
            impulses = solver._propagation_impulses.numpy()[0]
            seen["normal"], seen["rolling"] = impulses[slot], float(np.linalg.norm(impulses[slot + 3 : slot + 5]))
            seen["spin"] = abs(float(impulses[slot + 5]))

        for steps in (1, 5, 40):
            with self.subTest(steps=steps):
                _simulate(
                    "sphere",
                    0.0,
                    steps=steps,
                    angular=(10.0, 0.0, 6.0),
                    free_rigid_rolling_friction=0.002,
                    free_rigid_torsional_friction=0.001,
                    observe=observe,
                    **_PROPAGATION,
                )
                self.assertGreater(seen["normal"], 0.0)
                self.assertLessEqual(seen["rolling"], 0.002 * seen["normal"] * (1.0 + 1.0e-5))
                self.assertLessEqual(seen["spin"], 0.001 * seen["normal"] * (1.0 + 1.0e-5))

    def test_rows_stay_within_the_pooled_load_after_every_sweep(self):
        """Hold every rolling and spin impulse at zero before friction starts and within the pooled patch load after."""
        sweeps = _sweep_box()
        self.assertTrue(any(sweep.offset < sweep.friction_start for sweep in sweeps))
        _assert_cold_before_friction_start(self, sweeps)
        _assert_within_the_pooled_load(self, sweeps)
        # A four-anchor patch saturates far above any one anchor's load, so the pooled bound is the one in force.
        self.assertGreater(_own_load_ratio(sweeps), 2.0)

    def test_sweep_checks_catch_an_unguarded_or_unpooled_kernel(self):
        """Trip the sweep checks with the rows-off kernel, which neither holds nor bounds them, and an unpooled load."""

        def unguarded(solver):
            solver._pgs_solve_propagation_kernel = _get_pgs_solve_propagation_contact_kernel(
                solver._propagation_max_constraints,
                str(solver.model.device.arch),
                worlds_per_block=solver._propagation_gs_worlds_per_block,
            )

        with self.assertRaisesRegex(AssertionError, "before friction start"):
            _assert_cold_before_friction_start(self, _sweep_box(on_solver=unguarded))

        def unpooled(solver):
            # Cut each normal row's patch ring before the sweep, after the recorder has read the true ring.
            row_type = solver._propagation_row_type.numpy()
            row_parent = solver._propagation_row_parent.numpy()
            row_parent[row_type == PGS_CONSTRAINT_TYPE_CONTACT] = -1
            solver._propagation_row_parent.assign(row_parent)

        sweeps = _sweep_box(before_sweep=unpooled)
        _assert_within_the_pooled_load(self, sweeps)
        self.assertLessEqual(_own_load_ratio(sweeps), 1.0 + 1.0e-5)

    def test_rows_never_add_energy(self):
        """End a contact step with no more kinetic energy than the rows-off step, for an anisotropic inertia."""
        inertia = _ANISOTROPIC * 4.0e-5
        omega = (1.0, 0.0, -3.0)

        def energy(twist):
            return 0.5 * _MASS * float(twist[:3] @ twist[:3]) + 0.5 * float(twist[3:6] @ inertia @ twist[3:6])

        for rolling in (1.0, 1.0e-3, 1.0e-4):
            with self.subTest(rolling=rolling):
                kwargs = {"angular": omega, "inertia": inertia, "steps": 1, **_PROPAGATION}
                _, twist_free, _ = _simulate("sphere", 0.0, **kwargs)
                _, twist, _ = _simulate(
                    "sphere", 0.0, free_rigid_rolling_friction=rolling, free_rigid_torsional_friction=rolling, **kwargs
                )
                self.assertLessEqual(energy(twist), energy(twist_free) * (1.0 + 1.0e-4))

    def test_trajectory_matches_the_immediate_response(self):
        """Track the immediate response's free-sphere trajectory at every step with rolling and spin rows on.

        The bound is empirical for this fixture: the two routes round differently, so their twists drift apart.
        """
        kwargs = {
            "steps": 60,
            "linear": (0.0, -0.3, 0.0),
            "angular": (10.0, 0.0, 6.0),
            "free_rigid_rolling_friction": 0.002,
            "free_rigid_torsional_friction": 0.001,
        }
        runs = {}
        for name, options in (("immediate", kwargs), ("propagation", {**kwargs, **_PROPAGATION})):
            poses, twists = [], []
            _simulate(
                "sphere",
                0.0,
                on_step=lambda pose, twist, p=poses, t=twists: (p.append(pose), t.append(twist)),
                **options,
            )
            runs[name] = np.array(poses), np.array(twists)
        _, twist_off, _ = _simulate("sphere", 0.0, steps=60, linear=kwargs["linear"], angular=kwargs["angular"])
        (pose_immediate, twist_immediate), (pose, twist) = runs["immediate"], runs["propagation"]
        np.testing.assert_allclose(twist, twist_immediate, rtol=0.0, atol=5.0e-5)
        np.testing.assert_allclose(pose[:, :3], pose_immediate[:, :3], rtol=0.0, atol=1.0e-6)
        # The rows change the trajectory far beyond that gap, so this is no silent drop.
        self.assertGreater(float(np.max(np.abs(twist_immediate[-1] - twist_off))), 1.0)

    def test_rows_start_cold_under_warm_start(self):
        """Carry the normal and sliding impulses but start the angular rows at zero, as the immediate route does."""
        for response, options in _RESPONSES.items():
            with self.subTest(response=response):
                model, _ = _scene("sphere")
                solver = SolverSteel(
                    model,
                    solve_mode="matrix_free",
                    iterations=32,
                    contact_warmstart=True,
                    free_rigid_rolling_friction=0.002,
                    free_rigid_torsional_friction=0.001,
                    **{"deterministic": True, **options},
                )
                pipeline = newton.CollisionPipeline(model, deterministic=True, contact_matching="sticky")
                contacts = pipeline.contacts()
                states = [model.state(), model.state()]
                qd = states[0].joint_qd.numpy()
                qd[3:6] = (10.0, 0.0, 6.0)
                states[0].joint_qd.assign(qd)
                newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
                control = model.control()
                for steps in (20, 1):
                    if steps == 1:
                        # A solve-free step leaves exactly the gathered warm start in the impulses.
                        solver._iterations = 0
                    for _ in range(steps):
                        states[0].clear_forces()
                        pipeline.collide(states[0], contacts)
                        solver.step(states[0], states[1], control, contacts, _DT)
                        states.reverse()
                    _, _, _, impulses = _row_arrays(solver, response)
                    slot = int(solver._contact_slot.numpy()[0])
                    if steps == 20:
                        self.assertEqual(int(solver._contact_path.numpy()[0]), _ROUTE[response])
                        solved = impulses.numpy()[0, slot : slot + 6].copy()
                gathered = impulses.numpy()[0, slot : slot + 6]
                self.assertTrue(np.all(solved[[0, 3, 4, 5]] != 0.0))
                self.assertGreater(gathered[0], 0.0)
                np.testing.assert_array_equal(gathered[3:6], 0.0)

    def test_body_parent_f_balances_the_rolling_torque(self):
        """Keep a rolling free sphere's parent wrench at zero on the propagation route, rows asserted first."""
        seen = {}

        def observe(solver, contacts, observables):
            slot = int(solver._contact_slot.numpy()[0])
            seen["path"] = int(solver._contact_path.numpy()[0])
            seen["types"] = solver._propagation_row_type.numpy()[0, slot + 3 : slot + 5].tolist()
            seen["rolling"] = float(np.linalg.norm(solver._propagation_impulses.numpy()[0, slot + 3 : slot + 5])) / _DT
            seen["parent"] = observables.body_parent_f.numpy()[1]

        _simulate(
            "sphere",
            0.0,
            steps=4,
            parent_f=True,
            free_rigid_rolling_friction=0.002,
            observe=observe,
            **_ON,
            **_ROLLING,
            **_PROPAGATION,
        )
        self.assertEqual(seen["path"], 2)
        self.assertEqual(seen["types"], [PGS_CONSTRAINT_TYPE_CONTACT_ROLLING] * 2)
        torque = 0.002 * _MASS * _GRAVITY
        self.assertAlmostEqual(seen["rolling"], torque, delta=0.05 * torque)
        self.assertLess(float(np.linalg.norm(seen["parent"][3:])), 0.02 * torque)

    def test_composes_with_anchors_warm_start_and_velocity_iterations(self):
        """Keep the Coulomb stopping distance with each production-style option the propagation response takes."""
        expected = _coulomb_stopping_distance(0.3, 0.002)
        for options in (
            {"friction_anchor_beta": 0.2},
            {"contact_warmstart": True, "contact_matching": "sticky"},
            {"velocity_iterations": 1},
        ):
            with self.subTest(**options):
                _, twist, travel = _simulate(
                    "sphere", 1.5, free_rigid_rolling_friction=0.002, **_ON, **_ROLLING, **options, **_PROPAGATION
                )
                self.assertLess(np.linalg.norm(twist[3:6]), 1.0e-3)
                self.assertAlmostEqual(travel, expected, delta=0.05 * expected)

    def test_captured_run_matches_the_eager_run(self):
        """Replay a captured step pair bit for bit against eager steps, several bodies in one world."""
        kwargs = {
            "free_rigid_rolling_friction": 0.001,
            "free_rigid_torsional_friction": 0.001,
            "friction_anchor_beta": 0.2,
            "count": 3,
            **_ROLLING,
            **_PROPAGATION,
        }
        eager, _, _ = _simulate("sphere", 0.3, **kwargs)
        captured, _, _ = _simulate("sphere", 0.3, graph=True, **kwargs)
        np.testing.assert_array_equal(eager, captured)


_COLORED = {"articulated_contact_response": "propagation_colored"}


def _simulate_colored(variant, *args, on_solver=None, **kwargs):
    """Run :func:`_simulate` on the colored response built as colored variant ``variant``."""

    def hook(solver):
        select_colored_variant(solver, variant)
        if on_solver is not None:
            on_solver(solver)

    with colored_variant_switches(variant):
        return _simulate(*args, on_solver=hook, **_COLORED, **kwargs)


def _sweep_colored(variant, on_solver=None):
    """Record :func:`_sweep_box` on deterministic colored variant ``variant``."""

    def hook(solver):
        select_colored_variant(solver, variant)
        if on_solver is not None:
            on_solver(solver)

    with colored_variant_switches(variant):
        return _sweep_box(on_solver=hook, response={**_COLORED, "deterministic": True})


def _trajectory(*args, simulate=_simulate, **kwargs):
    """Return the body's poses and twists after every step."""
    poses, twists = [], []
    simulate(*args, on_step=lambda pose, twist: (poses.append(pose), twists.append(twist)), **kwargs)
    return np.array(poses), np.array(twists)


def _pile(spins):
    """Return one world per entry of ``spins`` of three touching boxes, spinning at those rates, on a kinematic table."""
    scene = newton.ModelBuilder(up_axis=newton.Axis.Z)
    SolverSteel.register_custom_attributes(scene)
    for _ in spins:
        world = newton.ModelBuilder(up_axis=newton.Axis.Z)
        SolverSteel.register_custom_attributes(world)
        table = world.add_body(xform=wp.transform((0.0, 0.0, -0.05), wp.quat_identity()), is_kinematic=True)
        world.add_shape_box(table, hx=0.5, hy=0.5, hz=0.05)
        for index in range(3):
            pose = wp.transform((0.0601 * index, 0.0, _RADIUS + 1.0e-4), wp.quat_identity())
            box = world.add_body(xform=pose, mass=_MASS)
            world.add_shape_box(box, hx=_RADIUS, hy=_RADIUS, hz=_RADIUS)
        scene.add_world(world)
    model = scene.finalize()
    qd = np.zeros(model.joint_dof_count, dtype=np.float32)
    starts = model.joint_qd_start.numpy()
    joints_per_world = model.joint_count // len(spins)
    for joint in range(model.joint_count):
        world, index = divmod(joint, joints_per_world)
        if index > 0:
            qd[starts[joint] : starts[joint] + 6] = (0.0, 0.1 * index, 0.0, 0.0, 0.0, spins[world] * index)
    return model, qd


def _pile_rollout(spins, steps, graph_at=None, **solver_kwargs):
    """Step a pile from :func:`_pile` with every angular row on, optionally replaying a captured step from ``graph_at``."""
    model, qd = _pile(spins)
    options = {
        "solve_mode": "matrix_free",
        "iterations": 16,
        "deterministic": True,
        "free_rigid_rolling_friction": 0.002,
        "free_rigid_torsional_friction": 0.001,
        "friction_anchor_beta": 0.2,
        **solver_kwargs,
    }
    solver = SolverSteel(model, **options)
    pipeline = newton.CollisionPipeline(model, deterministic=True)
    contacts = pipeline.contacts()
    state_in, state_out = model.state(), model.state()
    state_in.joint_qd.assign(qd)
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    control = model.control()

    def step():
        state_in.clear_forces()
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, _DT)
        wp.copy(state_in.body_q, state_out.body_q)
        wp.copy(state_in.body_qd, state_out.body_qd)
        wp.copy(state_in.joint_q, state_out.joint_q)
        wp.copy(state_in.joint_qd, state_out.joint_qd)

    history, graph = [], None
    for k in range(steps):
        if k == graph_at:
            with wp.ScopedCapture(device=model.device) as capture:
                step()
            graph = capture.graph
        if graph is not None:
            wp.capture_launch(graph)
        else:
            step()
        history.append(state_in.body_qd.numpy().copy())
    return np.array(history), solver


@unittest.skipUnless(wp.is_cuda_available(), "Steel matrix-free requires CUDA")
class TestFreeRigidAngularFrictionColored(unittest.TestCase):
    """The rows on the colored propagation response, solved inside their contact's colored unit."""

    def test_units_carry_the_angular_rows(self):
        """Lay out four-, five- and six-row units that keep the straight-line kind for their contact row and sliding pair."""
        for torsional, rolling, slots in ((0.001, 0.0, 4), (0.0, 0.002, 5), (0.001, 0.002, 6)):
            seen = {}

            def observe(solver, contacts, observables, seen=seen, slots=slots):
                count = int(contacts.rigid_contact_count.numpy()[0])
                seen["path"] = solver._contact_path.numpy()[:count].tolist()
                seen["unit_len"] = solver._color_unit_len.numpy()[:count].tolist()
                slot = int(solver._contact_slot.numpy()[0])
                seen["types"] = solver._propagation_row_type.numpy()[0, slot : slot + slots].tolist()
                units = int(solver._color_world_offsets.numpy()[-1])
                meta = solver._color_unit_meta.numpy().reshape(-1, PROPAGATION_UNIT_META)[:units]
                seen["rows"] = meta[:, 5 + PROPAGATION_UNIT_RING].tolist()
                seen["kinds"] = meta[:, 6 + PROPAGATION_UNIT_RING].tolist()

            with self.subTest(torsional=torsional, rolling=rolling):
                _simulate_colored(
                    "packed",
                    "sphere",
                    0.0,
                    steps=2,
                    free_rigid_torsional_friction=torsional,
                    free_rigid_rolling_friction=rolling,
                    observe=observe,
                )
                self.assertEqual(seen["path"], [2])
                self.assertEqual(seen["unit_len"], [slots])
                angular = [PGS_CONSTRAINT_TYPE_CONTACT_ROLLING] * (2 if rolling else 0)
                angular += [PGS_CONSTRAINT_TYPE_CONTACT_SPIN] if torsional else []
                self.assertEqual(seen["types"][3:], angular)
                self.assertEqual(seen["rows"], [slots])
                self.assertEqual(seen["kinds"], [1])

    def test_every_variant_tracks_plain_and_immediate(self):
        """Track the plain and immediate twists of a rolling, spinning sphere at every step on each colored body.

        Four-, five- and six-row units; the bounds are plain's empirical gap to immediate on this fixture.
        """
        rows = {"spin": (0.001, 0.0), "rolling": (0.0, 0.002), "both": (0.001, 0.002)}
        motion = {"steps": 60, "linear": (0.0, -0.3, 0.0), "angular": (10.0, 0.0, 6.0)}
        _, twist_off = _trajectory("sphere", 0.0, **motion)
        for label, (torsional, rolling) in rows.items():
            kwargs = {**motion, "free_rigid_torsional_friction": torsional, "free_rigid_rolling_friction": rolling}
            references = {
                "immediate": _trajectory("sphere", 0.0, **kwargs),
                "plain": _trajectory("sphere", 0.0, **kwargs, **_PROPAGATION),
            }
            effect = float(np.abs(references["immediate"][1][-1] - twist_off[-1]).max())
            for variant in COLORED_VARIANTS:
                with self.subTest(rows=label, variant=variant):
                    pose, twist = _trajectory(
                        "sphere", 0.0, simulate=lambda *a, v=variant, **k: _simulate_colored(v, *a, **k), **kwargs
                    )
                    for name, (pose_ref, twist_ref) in references.items():
                        np.testing.assert_allclose(twist, twist_ref, rtol=0.0, atol=5.0e-5, err_msg=name)
                        np.testing.assert_allclose(pose[:, :3], pose_ref[:, :3], rtol=0.0, atol=1.0e-6, err_msg=name)
                    gap = float(np.abs(twist[-1] - references["immediate"][1][-1]).max())
                    colored_effect = float(np.abs(twist[-1] - twist_off[-1]).max())
                    self.assertGreaterEqual(colored_effect, 0.5 * effect)
                    self.assertGreaterEqual(colored_effect, 10.0 * gap)

    def test_spin_is_relative_to_a_turning_support(self):
        """Spin a resting sphere up to a kinematic turntable's rate, as plain does, with the turntable's side folded."""
        options = {"free_rigid_torsional_friction": 5.0e-4, "free_rigid_rolling_friction": 0.0}
        kwargs = {"steps": 120, "support_angular": (0.0, 0.0, 2.0), **options}
        _, plain = _trajectory("sphere", 0.0, **kwargs, **_PROPAGATION)
        self.assertGreater(float(plain[-1][5]), 1.0)
        for variant in ("packed", "reference", "two lanes", "block", "block global"):
            with self.subTest(variant=variant):

                def folded(solver, test=self):
                    test.assertTrue(np.any(solver._body_prescribed.numpy()))

                _, twist = _trajectory(
                    "sphere",
                    0.0,
                    simulate=lambda *a, v=variant, **k: _simulate_colored(v, *a, on_solver=folded, **k),
                    **kwargs,
                )
                np.testing.assert_allclose(twist, plain, rtol=0.0, atol=5.0e-5)

    def test_rows_stay_within_the_pooled_load_after_every_sweep(self):
        """Hold the rows at zero before friction starts and within the pooled patch load after, on each colored body."""
        for variant in COLORED_VARIANTS:
            with self.subTest(variant=variant):
                sweeps = _sweep_colored(variant)
                self.assertTrue(any(sweep.offset < sweep.friction_start for sweep in sweeps))
                _assert_cold_before_friction_start(self, sweeps)
                _assert_within_the_pooled_load(self, sweeps)
                self.assertGreater(_own_load_ratio(sweeps), 2.0)

    def test_bodies_built_without_the_rows_leave_them_unsolved(self):
        """Leave every angular impulse at zero with colored bodies built without the rows, which end units early."""
        for variant in ("packed", "block"):
            with self.subTest(variant=variant):

                def rows_off(solver):
                    arch = str(solver.model.device.arch)
                    entries = solver._color_world_offsets.shape[0] // solver._world_count
                    warp_kernel = solver._pgs_solve_propagation_colored_warp_kernel
                    if warp_kernel is None:
                        solver._pgs_solve_propagation_colored_block_kernel = (
                            _get_pgs_solve_propagation_colored_block_kernel(
                                solver._propagation_max_constraints,
                                entries,
                                solver._propagation_colored_block_dim,
                                int(solver._max_propagation_bodies),
                                arch,
                            )
                        )
                        return
                    solver._pgs_solve_propagation_colored_warp_kernel = _get_pgs_solve_propagation_colored_warp_kernel(
                        solver._propagation_max_constraints,
                        entries,
                        int(solver._max_propagation_bodies),
                        1,
                        solver._propagation_colored_worlds_per_block,
                        arch,
                        prefetch=True,
                        packed="_pk1" in warp_kernel.key,
                    )

                moved = {}
                for name, on_solver in (("rows", None), ("rows off", rows_off)):
                    sweeps = _sweep_colored(variant, on_solver=on_solver)
                    blocks = [
                        sweep.after[row : row + width] for sweep in sweeps for row, width in _angular_blocks(sweep)
                    ]
                    self.assertTrue(blocks)
                    moved[name] = any(np.any(block != 0.0) for block in blocks)
                self.assertEqual(moved, {"rows": True, "rows off": False})

    def test_rows_start_cold_under_warm_start(self):
        """Carry the normal and sliding impulses but start the angular rows at zero, as the other responses do."""
        model, _ = _scene("sphere")
        solver = SolverSteel(
            model,
            solve_mode="matrix_free",
            iterations=32,
            contact_warmstart=True,
            free_rigid_rolling_friction=0.002,
            free_rigid_torsional_friction=0.001,
            **_COLORED,
        )
        pipeline = newton.CollisionPipeline(model, deterministic=True, contact_matching="sticky")
        contacts = pipeline.contacts()
        states = [model.state(), model.state()]
        qd = states[0].joint_qd.numpy()
        qd[3:6] = (10.0, 0.0, 6.0)
        states[0].joint_qd.assign(qd)
        newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
        control = model.control()
        for steps in (20, 1):
            if steps == 1:
                # A solve-free step leaves exactly the gathered warm start in the impulses.
                solver._iterations = 0
            for _ in range(steps):
                states[0].clear_forces()
                pipeline.collide(states[0], contacts)
                solver.step(states[0], states[1], control, contacts, _DT)
                states.reverse()
            slot = int(solver._contact_slot.numpy()[0])
            impulses = solver._propagation_impulses.numpy()[0, slot : slot + 6].copy()
            if steps == 20:
                self.assertEqual(int(solver._contact_path.numpy()[0]), 2)
                solved = impulses
        self.assertTrue(np.all(solved[[0, 3, 4, 5]] != 0.0))
        self.assertGreater(impulses[0], 0.0)
        np.testing.assert_array_equal(impulses[3:6], 0.0)

    def test_deterministic_runs_repeat_bitwise_and_ignore_world_order(self):
        """Repeat a three-world pile bitwise, and give each world the same twists when the worlds are permuted."""
        spins = (4.0, 1.0, 7.0)
        first, solver = _pile_rollout(spins, 40, **_COLORED)
        # Box-box contacts give each world more units than bodies touching the table.
        self.assertGreater(int(solver._color_world_unit_cursor.numpy()[0]), 12)
        second, _ = _pile_rollout(spins, 40, **_COLORED)
        np.testing.assert_array_equal(first, second)
        order = (2, 0, 1)
        permuted, _ = _pile_rollout(tuple(spins[w] for w in order), 40, **_COLORED)
        bodies = first.shape[1] // len(spins)
        for slot, world in enumerate(order):
            np.testing.assert_array_equal(
                permuted[:, slot * bodies : (slot + 1) * bodies], first[:, world * bodies : (world + 1) * bodies]
            )
        # The rows act on this pile: switching them off changes the twists.
        off, _ = _pile_rollout(
            spins, 40, **_COLORED, free_rigid_rolling_friction=0.0, free_rigid_torsional_friction=0.0
        )
        self.assertGreater(float(np.abs(off - first).max()), 1.0e-2)

    def test_serial_tail_solves_the_rows(self):
        """Solve the rows of units that overflow a two-color cap in the serial tail, as with the full cap.

        The row-loop bodies sweep the tail in the same order and arithmetic, so they match bitwise. The straight-line
        bodies sum in a different order in the tail; they are compared over one step, as the straight-line check in
        test_propagation does, since a pile amplifies that rounding over later steps.
        """
        for variant, steps in (("row loop", 20), ("block", 20), ("packed", 1), ("reference", 1)):
            with self.subTest(variant=variant):
                with colored_variant_switches(variant):
                    uncapped, _ = _pile_rollout((4.0, 1.0), steps, **_COLORED)
                    with patch.object(solver_steel, "PROPAGATION_COLOR_TAIL", 2):
                        capped, solver = _pile_rollout((4.0, 1.0), steps, **_COLORED)
                        entries = solver._color_world_offsets.shape[0] // solver._world_count
                        offsets = solver._color_world_offsets.numpy().reshape(solver._world_count, entries)
                self.assertTrue(np.all(offsets[:, -1] - offsets[:, -2] > 0), "no unit reached the serial tail")
                if steps > 1:
                    np.testing.assert_array_equal(capped, uncapped)
                else:
                    np.testing.assert_allclose(capped, uncapped, rtol=1e-4, atol=1e-6)

    def test_captured_steps_match_eager_steps(self):
        """Replay captured colored steps bit for bit against eager steps, on each colored body."""
        for variant in ("packed", "staged", "reference", "two lanes", "block"):
            with self.subTest(variant=variant):
                with colored_variant_switches(variant):
                    eager, _ = _pile_rollout((4.0, 1.0), 30, **_COLORED)
                    captured, _ = _pile_rollout((4.0, 1.0), 30, graph_at=3, **_COLORED)
                np.testing.assert_array_equal(captured, eager)


class _Sweep(NamedTuple):
    offset: int
    friction_start: int
    before: np.ndarray
    after: np.ndarray
    row_type: np.ndarray
    row_parent: np.ndarray
    row_mu: np.ndarray


def _sweep_box(on_solver=None, before_sweep=None, response=None):
    """Record world 0's propagation rows around every sweep of a spinning box on a persistent patch.

    ``response`` holds the solver's response options, plain propagation by default.
    """
    sweeps = []

    def record(solver):
        if on_solver is not None:
            on_solver(solver)
        sweep_once = solver._propagation_pgs_solve_one_iteration

        def sweep(**kwargs):
            count = int(solver._propagation_constraint_count.numpy()[0])
            rows = [array.numpy()[0, :count].copy() for array in _row_arrays(solver, "propagation")]
            if before_sweep is not None:
                before_sweep(solver)
            sweep_once(**kwargs)
            after = solver._propagation_impulses.numpy()[0, :count].copy()
            sweeps.append(
                _Sweep(kwargs["iteration_offset"], kwargs["friction_start_iteration"], rows[3], after, *rows[:3])
            )

        solver._propagation_pgs_solve_one_iteration = sweep

    _simulate(
        "box",
        0.0,
        steps=3,
        angular=(0.0, 0.0, 6.0),
        free_rigid_rolling_friction=0.002,
        free_rigid_torsional_friction=0.001,
        friction_anchor_beta=0.2,
        contact_friction_position_iterations=8,
        on_solver=record,
        **(_PROPAGATION if response is None else response),
    )
    return sweeps


def _angular_blocks(sweep):
    """Yield each rolling pair's and spin row's first row and width."""
    for row in np.flatnonzero(sweep.row_type == PGS_CONSTRAINT_TYPE_CONTACT_SPIN):
        yield int(row), 1
    rolling = sweep.row_type == PGS_CONSTRAINT_TYPE_CONTACT_ROLLING
    for row in np.flatnonzero(rolling & ~np.roll(rolling, 1)):
        yield int(row), 2


def _projection_load(sweep, row):
    """Return the pooled patch normal load the serial sweep projects ``row`` with.

    Patch normals before ``row`` hold this sweep's values and those after it the previous sweep's.
    """
    normal = int(sweep.row_parent[row])
    load, patch_row = 0.0, normal
    while True:
        load += float(sweep.after[patch_row] if patch_row < row else sweep.before[patch_row])
        patch_row = int(sweep.row_parent[patch_row])
        if patch_row < 0 or patch_row == normal:
            return load


def _assert_cold_before_friction_start(test, sweeps):
    for sweep in sweeps:
        if sweep.offset < sweep.friction_start:
            for row, width in _angular_blocks(sweep):
                test.assertTrue(np.all(sweep.after[row : row + width] == 0.0), f"row {row} moved before friction start")


def _assert_within_the_pooled_load(test, sweeps):
    for sweep in sweeps:
        for row, width in _angular_blocks(sweep):
            impulse = float(np.linalg.norm(sweep.after[row : row + width]))
            bound = float(sweep.row_mu[row]) * _projection_load(sweep, row)
            test.assertLessEqual(impulse, bound * (1.0 + 1.0e-5) + 1.0e-12, f"row {row} exceeds the pooled load")


def _own_load_ratio(sweeps):
    """Return the largest angular impulse over its own anchor's normal-load bound."""
    ratio = 0.0
    for sweep in sweeps:
        for row, width in _angular_blocks(sweep):
            normal = int(sweep.row_parent[row])
            load = float(sweep.after[normal] if normal < row else sweep.before[normal])
            impulse = float(np.linalg.norm(sweep.after[row : row + width]))
            if load > 0.0:
                ratio = max(ratio, impulse / (float(sweep.row_mu[row]) * load))
    return ratio


def _padded_body_scene(pad_condim: int, body: str = "free"):
    """Build a body whose wide condim-3 box carries an inset pad, 2 um shallower, with the same raw materials.

    ``body`` is ``"free"``, ``"articulated"`` (a prismatic link beside a free sphere) or ``"kinematic"`` (a kinematic
    free body beside a prismatic link, so no dynamic free body keeps the free-rigid route).
    """
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    SolverSteel.register_custom_attributes(builder)
    ground = newton.ModelBuilder.ShapeConfig(mu=0.0, mu_torsional=0.0, mu_rolling=0.0)
    builder.add_shape_box(
        -1, xform=wp.transform((0.0, 0.0, -0.05), wp.quat_identity()), hx=1.0, hy=1.0, hz=0.05, cfg=ground
    )
    inertia = wp.mat33(*(np.eye(3) * 1.0e-3).astype(np.float32).ravel())
    pose = wp.transform((0.0, 0.0, 0.02), wp.quat_identity())
    if body == "free":
        padded = builder.add_body(xform=pose, mass=2.0, inertia=inertia, lock_inertia=True)
    else:
        link_pose = wp.transform((0.5, 0.5, 0.02), wp.quat_identity())
        link = builder.add_link(xform=link_pose, mass=2.0, inertia=inertia, lock_inertia=True)
        joint = builder.add_joint_prismatic(-1, link, parent_xform=link_pose, axis=newton.Axis.X)
        builder.add_articulation([joint])
        builder.add_shape_box(link, hx=0.02, hy=0.02, hz=0.02)
        if body == "articulated":
            free = builder.add_body(xform=wp.transform((-0.5, -0.5, _RADIUS), wp.quat_identity()), mass=_MASS)
            builder.add_shape_sphere(free, radius=_RADIUS)
            padded = builder.add_link(xform=pose, mass=2.0, inertia=inertia, lock_inertia=True)
            joint = builder.add_joint_prismatic(-1, padded, parent_xform=pose, axis=newton.Axis.Y)
            builder.add_articulation([joint])
        else:
            padded = builder.add_body(xform=pose, mass=2.0, inertia=inertia, lock_inertia=True, is_kinematic=True)
    material = newton.ModelBuilder.ShapeConfig(density=0.0, mu=0.0, mu_torsional=0.01, mu_rolling=0.0)
    box = builder.add_shape_box(padded, hx=0.05, hy=0.05, hz=0.02, cfg=material, custom_attributes={"steel:condim": 3})
    pad = builder.add_shape_box(
        padded,
        xform=wp.transform((0.0, 0.0, -0.015 + 2.0e-6), wp.quat_identity()),
        hx=0.02,
        hy=0.02,
        hz=0.005,
        cfg=material,
        custom_attributes={"steel:condim": pad_condim},
    )
    return builder.finalize(), box, pad


def _spin_solver(model):
    return SolverSteel(model, solve_mode="matrix_free", iterations=16, deterministic=True, friction_anchor_beta=0.0)


@unittest.skipUnless(wp.is_cuda_available(), "Steel matrix-free requires CUDA")
class TestFreeRigidAngularFrictionBodyPairReduction(unittest.TestCase):
    def _spin(self, pad_condim: int, body_pairs: bool, solver_first: bool = True):
        model, _, pad = _padded_body_scene(pad_condim)
        solver = _spin_solver(model) if solver_first else None
        reduction = newton.CollisionPipeline.ContactReductionConfig(body_pairs=body_pairs)
        pipeline = newton.CollisionPipeline(model, deterministic=True, reduce_contacts=reduction)
        if solver is None:
            solver = _spin_solver(model)
        contacts = pipeline.contacts()
        state_0, state_1, control = model.state(), model.state(), model.control()
        qd = state_0.joint_qd.numpy()
        qd[5] = 6.0
        state_0.joint_qd.assign(qd)
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
        pad_spin_rows = 0
        for _ in range(100):
            state_0.clear_forces()
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, _DT)
            state_0, state_1 = state_1, state_0
            count = int(contacts.rigid_contact_count.numpy()[0])
            on_pad = (contacts.rigid_contact_shape0.numpy()[:count] == pad) | (
                contacts.rigid_contact_shape1.numpy()[:count] == pad
            )
            pad_spin_rows += int(np.count_nonzero(solver._contact_slots_needed.numpy()[:count][on_pad] >= 4))
        return float(state_0.body_qd.numpy()[0][5]), pad_spin_rows

    def test_reduction_keeps_a_condim_pad_of_matching_material(self):
        """Keep every spin row of a condim-4 pad under body-pair reduction, though it shares the box's raw materials,
        whether the solver or the pipeline is built first."""
        spin_full, rows_full = self._spin(4, body_pairs=False)
        spin_control, _ = self._spin(3, body_pairs=True)
        self.assertGreater(rows_full, 0)
        self.assertLess(abs(spin_full), 1.0)
        self.assertGreater(spin_control, 5.9)
        for solver_first in (True, False):
            with self.subTest(solver_first=solver_first):
                spin, rows = self._spin(4, body_pairs=True, solver_first=solver_first)
                self.assertEqual(rows, rows_full)
                self.assertAlmostEqual(spin, spin_full, delta=1.0e-4)

    def test_reduction_groups_split_only_where_the_solver_applies_rows(self):
        """Split a free body's condim-4 pad into its own group once Steel is built; keep the core grouping
        without the solver, on an articulated body, and on a kinematic free body with no free-rigid route."""
        model, box, pad = _padded_body_scene(4)
        merged, _ = build_reduction_groups(model)
        self.assertEqual(merged[box], merged[pad])
        solver = _spin_solver(model)
        groups, _ = build_reduction_groups(model)
        self.assertNotEqual(groups[box], groups[pad])
        for body in ("articulated", "kinematic"):
            with self.subTest(body=body):
                model, box, pad = _padded_body_scene(4, body)
                core, _ = build_reduction_groups(model)
                solver = _spin_solver(model)
                self.assertTrue(solver._free_rigid_angular_rows)
                groups, _ = build_reduction_groups(model)
                np.testing.assert_array_equal(groups, core)
                self.assertEqual(groups[box], groups[pad])

    def test_reduction_keeps_every_live_solvers_groups(self):
        """Keep the pad's group while its solver lives, whatever other Steel solvers are built on the model."""
        model, box, pad = _padded_body_scene(4)
        first = _spin_solver(model)
        pipeline = newton.CollisionPipeline(
            model, reduce_contacts=newton.CollisionPipeline.ContactReductionConfig(body_pairs=True)
        )
        contacts = pipeline.contacts()
        state = model.state()
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)

        def pad_contacts():
            pipeline.collide(state, contacts)
            count = int(contacts.rigid_contact_count.numpy()[0])
            shapes = contacts.rigid_contact_shape0.numpy()[:count], contacts.rigid_contact_shape1.numpy()[:count]
            return int(np.count_nonzero((shapes[0] == pad) | (shapes[1] == pad)))

        expected = pad_contacts()
        self.assertGreater(expected, 0)
        zero = SolverSteel(
            model, solve_mode="matrix_free", free_rigid_torsional_friction=0.0, free_rigid_rolling_friction=0.0
        )
        self.assertEqual(pad_contacts(), expected)
        model.steel.condim.fill_(3)
        rows_off = SolverSteel(model, solve_mode="matrix_free")
        self.assertFalse(rows_off._free_rigid_angular_rows)
        pipeline.refresh_body_pair_reduction_groups()
        self.assertEqual(pad_contacts(), expected)
        groups, _ = build_reduction_groups(model)
        self.assertNotEqual(groups[box], groups[pad])
        del first, zero
        gc.collect()
        groups, _ = build_reduction_groups(model)
        self.assertEqual(groups[box], groups[pad])


class TestFreeRigidAngularFrictionImport(unittest.TestCase):
    """MJCF ``condim`` stays with MuJoCo: registering the Steel attributes never enables angular rows."""

    def test_mjcf_condim_does_not_enable_angular_rows(self):
        """Construct the same split solver for MJCF condim 3/4/6, direct or inherited, with and without registration."""
        for registered in (False, True):
            for condim in (3, 4, 6):
                for inherited in (False, True):
                    with self.subTest(registered=registered, condim=condim, inherited=inherited):
                        default = f'<default><geom condim="{condim}"/></default>' if inherited else ""
                        geom = "" if inherited else f' condim="{condim}"'
                        mjcf = (
                            f"<mujoco>{default}<worldbody>"
                            '<geom type="plane" size="1 1 0.1"/>'
                            f'<body pos="0 0 0.1"><freejoint/><geom type="sphere" size="0.03"{geom}/></body>'
                            "</worldbody></mujoco>"
                        )
                        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
                        if registered:
                            SolverSteel.register_custom_attributes(builder)
                        builder.add_mjcf(mjcf)
                        model = builder.finalize(device="cpu")
                        if registered:
                            np.testing.assert_array_equal(model.steel.condim.numpy(), 3)
                        solver = SolverSteel(model, solve_mode="split")
                        self.assertFalse(solver._free_rigid_angular_rows)


if __name__ == "__main__":
    unittest.main()
