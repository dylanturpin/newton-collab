# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Experimental rolling and torsional friction rows on SolverFeatherPGS matrix-free free-rigid contacts."""

import math
import unittest

import numpy as np
import warp as wp

import newton
from newton._src.geometry.contact_reduction_body_pairs import build_reduction_groups
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_CONTACT_ROLLING,
    PGS_CONSTRAINT_TYPE_CONTACT_SPIN,
    PGS_CONSTRAINT_TYPE_FRICTION,
)
from newton.solvers import SolverFeatherPGS, SolverObservableFlags
from newton.tests.feather_pgs._fixtures import set_torsion_radius

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
    SolverFeatherPGS.register_custom_attributes(builder)
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
        model.feather_pgs.condim.fill_(condim)
    options = {"solve_mode": "matrix_free", "iterations": 32, "deterministic": True, **solver_kwargs}
    solver = SolverFeatherPGS(model, **options)
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

    observables = solver.observables({SolverObservableFlags.BODY_PARENT_F}) if parent_f else None

    def step(state_in, state_out):
        state_in.clear_forces()
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, _DT, observables=observables)

    if steps is not None:
        for _ in range(steps):
            step(state_0, state_1)
            state_0, state_1 = state_1, state_0
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


@unittest.skipUnless(wp.is_cuda_available(), "FeatherPGS matrix-free requires CUDA")
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
        """Read the shape materials where feather_pgs:condim is 6, as the solver-wide override does everywhere."""
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

    def test_rows_reject_propagation_responses(self):
        """Reject the angular rows when articulated or free-free contacts take the propagation response."""
        model, _ = _scene("sphere", anchor_deg=45.0)
        for options in (
            {"articulated_contact_response": "propagation"},
            {"articulated_contact_response": "propagation_colored"},
        ):
            with self.subTest(**options), self.assertRaises(ValueError):
                SolverFeatherPGS(model, solve_mode="matrix_free", free_rigid_rolling_friction=1.0e-3, **options)

    def test_patch_angular_rows_do_not_depend_on_contact_order(self):
        """Keep each material's angular rows when two shapes of one body with different angular coefficients rest
        side by side, whichever shape's contacts come first."""
        results = []
        for order in ((3, 6), (6, 3)):
            builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
            SolverFeatherPGS.register_custom_attributes(builder)
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
                    custom_attributes={"feather_pgs:condim": condim},
                )
            model = builder.finalize()
            solver = SolverFeatherPGS(model, solve_mode="matrix_free", iterations=32, friction_anchor_beta=0.2)
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
            SolverFeatherPGS.register_custom_attributes(builder)
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
                    custom_attributes={"feather_pgs:condim": condim},
                )
            model = builder.finalize()
            solver = SolverFeatherPGS(
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
        model.feather_pgs.condim.fill_(6)
        model.shape_material_mu_rolling.fill_(0.001)
        solver = SolverFeatherPGS(model, solve_mode="matrix_free")
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
                SolverFeatherPGS(model, free_rigid_rolling_friction=value)
        with self.assertRaises(ValueError):
            SolverFeatherPGS(model, solve_mode="split", free_rigid_rolling_friction=1.0e-3)
        with self.assertRaises(ValueError):
            # Point friction: bisection with persistent patches would warn before the rejection under test.
            SolverFeatherPGS(
                model,
                solve_mode="matrix_free",
                friction_mode="bisection",
                friction_anchor_beta=0.0,
                free_rigid_rolling_friction=1.0e-3,
            )
        model.feather_pgs.condim.fill_(7)
        with self.assertRaises(ValueError):
            SolverFeatherPGS(model, solve_mode="matrix_free")
        model.feather_pgs.condim.fill_(3)
        solver = SolverFeatherPGS(model, solve_mode="matrix_free")
        model.feather_pgs.condim.fill_(6)
        with self.assertRaises(ValueError):
            solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)


def _padded_body_scene(pad_condim: int, body: str = "free"):
    """Build a body whose wide condim-3 box carries an inset pad, 2 um shallower, with the same raw materials.

    ``body`` is ``"free"``, ``"articulated"`` (a prismatic link beside a free sphere) or ``"kinematic"`` (a kinematic
    free body beside a prismatic link, so no dynamic free body keeps the free-rigid route).
    """
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    SolverFeatherPGS.register_custom_attributes(builder)
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
    box = builder.add_shape_box(
        padded, hx=0.05, hy=0.05, hz=0.02, cfg=material, custom_attributes={"feather_pgs:condim": 3}
    )
    pad = builder.add_shape_box(
        padded,
        xform=wp.transform((0.0, 0.0, -0.015 + 2.0e-6), wp.quat_identity()),
        hx=0.02,
        hy=0.02,
        hz=0.005,
        cfg=material,
        custom_attributes={"feather_pgs:condim": pad_condim},
    )
    return builder.finalize(), box, pad


def _spin_solver(model):
    return SolverFeatherPGS(
        model, solve_mode="matrix_free", iterations=16, deterministic=True, friction_anchor_beta=0.0
    )


@unittest.skipUnless(wp.is_cuda_available(), "FeatherPGS matrix-free requires CUDA")
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
        """Split a free body's condim-4 pad into its own group once FeatherPGS is built; keep the core grouping
        without the solver, on an articulated body, and on a kinematic free body with no free-rigid route."""
        model, box, pad = _padded_body_scene(4)
        merged, _ = build_reduction_groups(model)
        self.assertEqual(merged[box], merged[pad])
        _spin_solver(model)
        groups, _ = build_reduction_groups(model)
        self.assertNotEqual(groups[box], groups[pad])
        for body in ("articulated", "kinematic"):
            with self.subTest(body=body):
                model, box, pad = _padded_body_scene(4, body)
                core, _ = build_reduction_groups(model)
                _spin_solver(model)
                groups, _ = build_reduction_groups(model)
                np.testing.assert_array_equal(groups, core)
                self.assertEqual(groups[box], groups[pad])


class TestFreeRigidAngularFrictionImport(unittest.TestCase):
    """MJCF ``condim`` stays with MuJoCo: registering the FeatherPGS attributes never enables angular rows."""

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
                            SolverFeatherPGS.register_custom_attributes(builder)
                        builder.add_mjcf(mjcf)
                        model = builder.finalize(device="cpu")
                        if registered:
                            np.testing.assert_array_equal(model.feather_pgs.condim.numpy(), 3)
                        solver = SolverFeatherPGS(model, solve_mode="split")
                        self.assertFalse(solver._free_rigid_angular_rows)


if __name__ == "__main__":
    unittest.main()
