# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Contact compliance and contact regularization of SolverFeatherPGS."""

import gc
import inspect
import os
import unittest
import weakref
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs._testing import kernel_overrides
from newton._src.solvers.feather_pgs.contact_compliance import (
    begin_rows,
    material_coefficients,
    normal_coefficients,
    prepare_rows,
    start_step,
)
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    compute_world_contact_bias,
    pgs_solve_mf_loop,
)
from newton.geometry import HydroelasticSDF
from newton.solvers import SolverFeatherPGS
from newton.tests.feather_pgs._compliance_reference import solve_reference, support_trajectory
from newton.tests.feather_pgs._fixtures import (
    _build_mixed_response_model,
    _build_scissor_model,
    _contact_rows,
    make_fixture,
    run_fixture,
)
from newton.tests.unittest_utils import (
    add_function_test,
    get_cuda_test_devices,
    get_selected_cuda_test_devices,
    get_test_devices,
)


def run_native_hydro_fixture(*, articulated=True):
    """Consume emitted SDF hydro coefficients unchanged on dense or MF sphere rows."""
    with wp.ScopedDevice(os.environ.get("HYDRO_TEST_DEVICE", "cuda:0")):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.default_shape_cfg = newton.ModelBuilder.ShapeConfig(
            mu=0.5,
            is_hydroelastic=True,
            sdf_max_resolution=32,
            sdf_narrow_band_range=(-0.01, 0.01),
            sdf_padding=0.006,
            gap=0.005,
            kh=1e7,
        )
        builder.add_shape_box(
            body=-1, hx=0.1, hy=0.1, hz=0.025, xform=wp.transform(wp.vec3(0, 0, -0.025), wp.quat_identity())
        )
        transform = wp.transform(wp.vec3(0, 0, 0.048), wp.quat_identity())
        if articulated:
            body = builder.add_link(xform=transform)
            joint = builder.add_joint_prismatic(parent=-1, child=body, axis=newton.Axis.Z, parent_xform=transform)
            builder.add_articulation([joint])
        else:
            body = builder.add_body(xform=transform)
        cfg = builder.default_shape_cfg.copy()
        cfg.density = 0.3 / (4 / 3 * np.pi * 0.05**3)
        builder.add_shape_sphere(body, radius=0.05, cfg=cfg)
        model = builder.finalize()
        pipeline = newton.CollisionPipeline(
            model,
            rigid_contact_max=512,
            deterministic=True,
            sdf_hydroelastic_config=HydroelasticSDF.Config(
                reduce_contacts=True, moment_matching=True, anchor_contact=True, buffer_fraction=1.0
            ),
        )
        model.rigid_contact_max = 512
        inertia = tuple(x.numpy().copy() for x in (model.body_mass, model.body_com, model.body_inertia))
        initial = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, initial)
        contacts = pipeline.contacts()
        pipeline.collide(initial, contacts)
        input_contract = {
            name: getattr(initial, name).numpy().copy() for name in ("joint_q", "joint_qd", "body_q", "body_qd")
        }
        count = int(contacts.rigid_contact_count.numpy()[0])
        stiffness = contacts.rigid_contact_stiffness.numpy()[:count].copy()
        damping = contacts.rigid_contact_damping.numpy()[:count].copy()
        friction = contacts.rigid_contact_friction.numpy()[:count].copy()
        if count <= 0 or count > 512 or not (stiffness > 0).any():
            raise AssertionError("Expected valid native hydro contacts with emitted positive stiffness")
        results = {}
        for enabled in (False, True):
            solver = SolverFeatherPGS(
                model,
                contact_compliance=enabled,
                friction_anchor_beta=0.0,
                solve_mode="matrix_free",
                row_schedule="interleaved",
                articulated_contact_response="immediate",
                enable_restitution=False,
                iterations=128,
                contact_regularization=0.0,
                articulation_constraint_max=2048,
                free_body_constraint_max=32 if articulated else 2048,
                baumgarte=0.05,
            )
            output = model.state()
            step_input = model.state()
            for name, value in input_contract.items():
                getattr(step_input, name).assign(value)
                np.testing.assert_array_equal(getattr(step_input, name).numpy(), value)
            step_input.clear_forces()
            solver.step(step_input, output, model.control(), contacts, 0.0025)
            paths = solver._contact_path.numpy()[:count]
            if not np.all(paths == (0 if articulated else 1)):
                raise AssertionError("Native hydro fixture did not use the requested contact route")
            impulses = solver._impulses if articulated else solver._mf_impulses
            row_types = solver._row_type if articulated else solver._mf_row_type
            results["compliant" if enabled else "stock_law"] = {
                "body_z_m": float(output.body_q.numpy()[body, 2]),
                "joint_velocity_m_s": float(output.joint_qd.numpy()[0]),
                "normal_impulse_N_s": float(impulses.numpy()[0, row_types.numpy()[0] == 0].sum()),
                "consumed_compliant_contacts": solver.compliance_contact_count,
            }
        for before, after in zip(inertia, (model.body_mass, model.body_com, model.body_inertia), strict=True):
            np.testing.assert_array_equal(before, after.numpy())
        np.testing.assert_array_equal(stiffness, contacts.rigid_contact_stiffness.numpy()[:count])
        np.testing.assert_array_equal(damping, contacts.rigid_contact_damping.numpy()[:count])
        np.testing.assert_array_equal(friction, contacts.rigid_contact_friction.numpy()[:count])
        results["native_material"] = {
            "contacts": count,
            "positive_stiffness_contacts": int((stiffness > 0).sum()),
            "stiffness_min_N_per_m": float(stiffness.min()),
            "stiffness_max_N_per_m": float(stiffness.max()),
            "damping_max_N_s_per_m": float(damping.max()),
            "friction_weight_min": float(friction.min()),
            "friction_weight_max": float(friction.max()),
            "mass_kg": float(inertia[0][body]),
        }
        return results


def run_paired_fixture(*, hinv_jt_kernel, friction, steps=40):
    """Drop a free box onto a 23-DOF arm (one robot/free-body pair per world) with compliant contacts."""
    device = os.environ.get("HYDRO_TEST_DEVICE", "cuda:0")
    with wp.ScopedDevice(device):
        model = _build_mixed_response_model(device, dof_count=23, friction=friction)
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=64, broad_phase="nxn", reduce_contacts=False)
        model.rigid_contact_max = 64
        contacts = pipeline.contacts()
        for name in ("rigid_contact_stiffness", "rigid_contact_damping", "rigid_contact_friction"):
            setattr(contacts, name, wp.zeros(64, dtype=float))
        with kernel_overrides({"hinv_jt_kernel": hinv_jt_kernel}):
            solver = SolverFeatherPGS(
                model,
                contact_compliance=True,
                friction_anchor_beta=0.0,
                solve_mode="matrix_free",
                enable_restitution=False,
                contact_friction_scale=1.0 if friction > 0.0 else 0.0,
                joint_limit_activation_gap=0.0,
                iterations=8,
                articulation_constraint_max=96,
                free_body_constraint_max=32,
            )
        state, output = model.state(), model.state()
        free = int(np.flatnonzero(solver._model_plan.is_free_rigid)[0])
        start = int(solver._model_plan.articulation_dof_start[free])
        qd = state.joint_qd.numpy()
        qd[start] = 0.5
        qd[start + 2] = -3.0
        state.joint_qd.assign(qd)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        control = model.control()
        trace = []
        compliant = 0
        for _ in range(steps):
            state.clear_forces()
            pipeline.collide(state, contacts)
            contacts.rigid_contact_stiffness.fill_(2.0e4)
            contacts.rigid_contact_damping.fill_(10.0)
            contacts.rigid_contact_friction.fill_(1.0)
            solver.step(state, output, control, contacts, 1.0 / 240.0)
            state, output = output, state
            trace.append(state.joint_q.numpy().copy())
            compliant = max(compliant, solver.compliance_contact_count)
        return solver, np.asarray(trace), compliant


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestContactComplianceIntegration(unittest.TestCase):
    """Gate the experimental adapter against actual dense and MF contact paths."""

    def test_dense_articulated_support(self):
        """Recover finite-stiffness indentation on real dense articulated rows."""
        trace, paths, solver, mass = run_fixture(articulated=True, enabled=True)
        self.assertEqual(paths, {0})
        self.assertGreater(solver.compliance_contact_count, 0)
        self.assertLess(abs(trace[-1, 2] - (0.05 - mass * 9.81 / 3000)), 2e-6)

    def test_mf_free_body_support(self):
        """Recover the same indentation on real matrix-free free-body rows."""
        trace, paths, solver, mass = run_fixture(articulated=False, enabled=True)
        self.assertEqual(paths, {1})
        self.assertGreater(solver.compliance_contact_count, 0)
        self.assertLess(abs(trace[-1, 2] - (0.05 - mass * 9.81 / 3000)), 2e-6)

    def test_default_off_exact_state(self):
        """Match an omitted option to explicit OFF, not to a separate pristine solver."""
        for articulated in (False, True):
            original, _, _, _ = run_fixture(articulated=articulated, enabled=False, stock=True, steps=30)
            disabled, _, _, _ = run_fixture(articulated=articulated, enabled=False, steps=30)
            np.testing.assert_array_equal(original, disabled)

    def test_enabled_zero_stiffness_preserves_hard_contacts(self):
        """Preserve hard-contact motion with zero stiffness even when compliance is ON."""
        for articulated in (False, True):
            hard, paths, _, _ = run_fixture(articulated=articulated, enabled=False, stiffness=0, steps=60)
            noop, actual_paths, solver, _ = run_fixture(articulated=articulated, enabled=True, stiffness=0, steps=60)
            self.assertEqual(actual_paths, paths)
            self.assertEqual(solver.compliance_contact_count, 0)
            np.testing.assert_array_equal(hard, noop)

    def test_friction_uses_compliant_load_and_reducer_weight(self):
        """Hold below mu*mg and slip after reducing the exported friction weight."""
        held, _, solver, mass = run_fixture(articulated=True, enabled=True, lateral_force=0.5)
        slipped, _, _, _ = run_fixture(articulated=True, enabled=True, lateral_force=0.5, friction_scale=0.25)
        self.assertLess(abs(held[-1, 0]), 1e-5)
        self.assertGreater(slipped[-1, 0], 0.005)
        types = solver._row_type.numpy()[0]
        normal = solver._impulses.numpy()[0, types == 0].sum() / 0.005
        tangent = np.linalg.norm(solver._impulses.numpy()[0, types == 2]) / 0.005
        self.assertAlmostEqual(float(normal), mass * 9.81, delta=1e-4)
        self.assertAlmostEqual(float(tangent), 0.5, delta=1e-4)

    def test_paired_factor_owner_matches_general_owner(self):
        """Apply the compliant residual in the paired factor-coordinate owner as in the general owner."""
        # Frictional contacts use the contact-triple branch; frictionless normal-only rows use the general branch.
        for friction in (0.7, 0.0):
            with self.subTest(friction=friction):
                general_solver, general, _ = run_paired_fixture(hinv_jt_kernel="par_row", friction=friction)
                paired_solver, paired, compliant = run_paired_fixture(hinv_jt_kernel="auto", friction=friction)
                self.assertFalse(general_solver._paired_factor_coordinates)
                self.assertTrue(paired_solver._paired_factor_coordinates)
                self.assertGreater(compliant, 0)
                np.testing.assert_allclose(paired, general, rtol=5.0e-4, atol=1.0e-5)

    def test_native_hydro_material_pipeline(self):
        """Apply native pressure-weighted hydro coefficients without overwriting fields."""
        result = run_native_hydro_fixture()
        self.assertEqual(
            result["compliant"]["consumed_compliant_contacts"], result["native_material"]["positive_stiffness_contacts"]
        )
        self.assertGreater(
            abs(result["compliant"]["normal_impulse_N_s"] - result["stock_law"]["normal_impulse_N_s"]), 1e-5
        )
        self.assertTrue(
            all(np.isfinite(list(result[key].values())).all() for key in ("compliant", "stock_law", "native_material"))
        )

    def test_native_hydro_free_body_manifold(self):
        """Handle multiple MF hydro normals without indexing a dummy prescribed-target buffer."""
        result = run_native_hydro_fixture(articulated=False)
        self.assertGreater(result["native_material"]["positive_stiffness_contacts"], 1)
        self.assertEqual(
            result["compliant"]["consumed_compliant_contacts"], result["native_material"]["positive_stiffness_contacts"]
        )
        self.assertTrue(np.isfinite(list(result["compliant"].values())).all())


def run_fixture_contact_compliance_upstream(
    *,
    device,
    articulated,
    enabled,
    steps=200,
    dt=0.005,
    iterations=8,
    stock=False,
    lateral_force=0.0,
    friction_scale=1.0,
    stiffness=3000.0,
    height=0.05,
    kinematic=False,
    restitution=0.0,
    solver_options=None,
):
    """Run sphere/plane contacts with prescribed material arrays, one physical step per tick."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.rigid_gap = 0.005
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
        xform = wp.transform(wp.vec3(0, 0, height), wp.quat_identity())
        if articulated:
            body = builder.add_link(xform=xform)
            if lateral_force:
                joint = builder.add_joint_d6(
                    parent=-1,
                    child=body,
                    parent_xform=xform,
                    linear_axes=[
                        newton.ModelBuilder.JointDofConfig(axis=newton.Axis.X),
                        newton.ModelBuilder.JointDofConfig(axis=newton.Axis.Z),
                    ],
                )
            else:
                joint = builder.add_joint_prismatic(parent=-1, child=body, axis=newton.Axis.Z, parent_xform=xform)
            builder.add_articulation([joint])
        else:
            body = builder.add_body(xform=xform)
        builder.add_shape_sphere(
            body,
            radius=0.05,
            cfg=newton.ModelBuilder.ShapeConfig(
                density=0.3 / (4 / 3 * np.pi * 0.05**3), mu=0.5, restitution=restitution
            ),
        )
        model = builder.finalize()
        if kinematic:
            flags = model.body_flags.numpy()
            flags[body] |= int(newton.BodyFlags.KINEMATIC)
            model.body_flags.assign(flags)
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=32)
        model.rigid_contact_max = 32
        contacts = pipeline.contacts()
        for name in ("rigid_contact_stiffness", "rigid_contact_damping", "rigid_contact_friction"):
            setattr(contacts, name, wp.zeros(32, dtype=float))
        extra = {} if stock else {"contact_compliance": enabled}
        options = dict(
            # Isolate the normal material law from positional patch friction.
            friction_anchor_beta=0.0,
            iterations=iterations,
            velocity_iterations=0,
            articulation_constraint_max=32,
            free_body_constraint_max=32,
            baumgarte=0.05,
            **extra,
        )
        options.update(solver_options or {})
        solver = SolverFeatherPGS(model, solve_mode="matrix_free", **options)
        state, next_state = model.state(), model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        trace = []
        seen_paths = set()
        control = model.control()
        for step in range(steps):
            state.clear_forces()
            pipeline.collide(state, contacts)
            count = int(contacts.rigid_contact_count.numpy()[0])
            contacts.rigid_contact_stiffness.fill_(stiffness)
            contacts.rigid_contact_damping.fill_(20.0)
            contacts.rigid_contact_friction.fill_(friction_scale)
            if lateral_force and step >= steps // 2:
                control.joint_f.assign(np.array([lateral_force, 0.0], dtype=np.float32))
            solver.step(state, next_state, control, contacts, dt)
            state, next_state = next_state, state
            seen_paths.update(solver._contact_path.numpy()[:count].tolist())
            trace.append(state.body_q.numpy()[body].copy())
        return np.asarray(trace), seen_paths, solver, float(model.body_mass.numpy()[body])


def run_native_hydro_fixture_contact_compliance_upstream(*, device, articulated=True):
    """Consume the emitted SDF hydroelastic coefficients unchanged on dense or free-body sphere rows."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.default_shape_cfg = newton.ModelBuilder.ShapeConfig(
            mu=0.5,
            is_hydroelastic=True,
            sdf_max_resolution=32,
            sdf_narrow_band_range=(-0.01, 0.01),
            sdf_padding=0.006,
            gap=0.005,
            kh=1e7,
        )
        builder.add_shape_box(
            body=-1, hx=0.1, hy=0.1, hz=0.025, xform=wp.transform(wp.vec3(0, 0, -0.025), wp.quat_identity())
        )
        transform = wp.transform(wp.vec3(0, 0, 0.048), wp.quat_identity())
        if articulated:
            body = builder.add_link(xform=transform)
            joint = builder.add_joint_prismatic(parent=-1, child=body, axis=newton.Axis.Z, parent_xform=transform)
            builder.add_articulation([joint])
        else:
            body = builder.add_body(xform=transform)
        cfg = builder.default_shape_cfg.copy()
        cfg.density = 0.3 / (4 / 3 * np.pi * 0.05**3)
        builder.add_shape_sphere(body, radius=0.05, cfg=cfg)
        model = builder.finalize()
        pipeline = newton.CollisionPipeline(
            model,
            rigid_contact_max=512,
            deterministic=True,
            sdf_hydroelastic_config=HydroelasticSDF.Config(
                reduce_contacts=True, moment_matching=True, anchor_contact=True, buffer_fraction=1.0
            ),
        )
        model.rigid_contact_max = 512
        inertia = tuple(x.numpy().copy() for x in (model.body_mass, model.body_com, model.body_inertia))
        initial = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, initial)
        contacts = pipeline.contacts()
        pipeline.collide(initial, contacts)
        input_contract = {
            name: getattr(initial, name).numpy().copy() for name in ("joint_q", "joint_qd", "body_q", "body_qd")
        }
        count = int(contacts.rigid_contact_count.numpy()[0])
        stiffness = contacts.rigid_contact_stiffness.numpy()[:count].copy()
        damping = contacts.rigid_contact_damping.numpy()[:count].copy()
        friction = contacts.rigid_contact_friction.numpy()[:count].copy()
        if count <= 0 or count > 512 or not (stiffness > 0).any():
            raise AssertionError("Expected valid hydroelastic contacts with positive stiffness")
        results = {}
        for enabled in (False, True):
            solver = SolverFeatherPGS(
                model,
                solve_mode="matrix_free",
                contact_compliance=enabled,
                friction_anchor_beta=0.0,
                iterations=128,
                contact_regularization=0.0,
                articulation_constraint_max=2048,
                free_body_constraint_max=32 if articulated else 2048,
                baumgarte=0.05,
            )
            output = model.state()
            step_input = model.state()
            for name, value in input_contract.items():
                getattr(step_input, name).assign(value)
                np.testing.assert_array_equal(getattr(step_input, name).numpy(), value)
            step_input.clear_forces()
            solver.step(step_input, output, model.control(), contacts, 0.0025)
            paths = solver._contact_path.numpy()[:count]
            if not np.all(paths == (0 if articulated else 1)):
                raise AssertionError("Hydroelastic fixture did not use the requested contact route")
            impulses = solver._impulses if articulated else solver._mf_impulses
            row_types = solver._row_type if articulated else solver._mf_row_type
            results["compliant" if enabled else "stock_law"] = {
                "body_z_m": float(output.body_q.numpy()[body, 2]),
                "joint_velocity_m_s": float(output.joint_qd.numpy()[0]),
                "normal_impulse_N_s": float(impulses.numpy()[0, row_types.numpy()[0] == 0].sum()),
                "consumed_compliant_contacts": solver.compliance_contact_count,
            }
        for before, after in zip(inertia, (model.body_mass, model.body_com, model.body_inertia), strict=True):
            np.testing.assert_array_equal(before, after.numpy())
        np.testing.assert_array_equal(stiffness, contacts.rigid_contact_stiffness.numpy()[:count])
        np.testing.assert_array_equal(damping, contacts.rigid_contact_damping.numpy()[:count])
        np.testing.assert_array_equal(friction, contacts.rigid_contact_friction.numpy()[:count])
        results["native_material"] = {
            "contacts": count,
            "positive_stiffness_contacts": int((stiffness > 0).sum()),
            "stiffness_min_N_per_m": float(stiffness.min()),
            "stiffness_max_N_per_m": float(stiffness.max()),
            "damping_max_N_s_per_m": float(damping.max()),
            "friction_weight_min": float(friction.min()),
            "friction_weight_max": float(friction.max()),
            "mass_kg": float(inertia[0][body]),
        }
        return results


def test_omitted_friction_anchor_beta_keeps_patches_without_compliance(test, device):
    """Keep the default patch friction when friction_anchor_beta is omitted and compliance is off."""
    builder = newton.ModelBuilder()
    body = builder.add_body()
    builder.add_shape_sphere(body, radius=0.05)
    model = builder.finalize(device=device)
    for kwargs in ({}, {"friction_anchor_beta": None}, {"contact_compliance": False}):
        with test.subTest(**kwargs):
            solver = SolverFeatherPGS(model, solve_mode="matrix_free", **kwargs)
            test.assertEqual(solver._friction_anchor_beta, 0.2)
            test.assertTrue(solver._friction_anchors_enabled)


def test_shape_restitution_is_rejected(test, device):
    """Reject a positive shape restitution, which has no defined composition with the compliant law."""
    with test.assertRaisesRegex(ValueError, "restitution"):
        run_fixture_contact_compliance_upstream(device=device, articulated=True, enabled=True, steps=1, restitution=0.5)
    # Restitution keeps working without compliance.
    trace, _, _, _ = run_fixture_contact_compliance_upstream(
        device=device, articulated=True, enabled=False, steps=1, restitution=0.5
    )
    test.assertTrue(np.isfinite(trace).all())


class TestContactComplianceIntegrationUpstream(unittest.TestCase):
    """Gate the experimental compliance against the dense and free-body contact rows."""


devices = get_cuda_test_devices()


for _fn in (
    test_omitted_friction_anchor_beta_keeps_patches_without_compliance,
    test_shape_restitution_is_rejected,
):
    add_function_test(TestContactComplianceIntegrationUpstream, _fn.__name__, _fn, devices=devices)


class TestContactComplianceLaw(unittest.TestCase):
    """Test isolated compliant contact equations independently of integration."""

    device = os.environ.get("HYDRO_TEST_DEVICE", "cpu")

    def trajectory(self, *, dt=0.0025, iterations=16, damping=20.0, x=0.0, v=0.0, compliant=True):
        """Return an eight-second support trace for a 0.3 kg, 3000 N/m pad."""
        with wp.ScopedDevice(self.device):
            result = wp.zeros((round(8.0 / dt), 3), dtype=float)
            wp.launch(
                support_trajectory, 1, inputs=[0.3, 9.81, 3000.0, damping, dt, x, v, iterations, int(compliant), result]
            )
            return result.numpy()

    def test_material_fallbacks_and_units(self):
        """Preserve weighted stiffness and apply friction weighting only once."""
        self.assertEqual(material_coefficients(100, 0, 0, shape_friction=2), (100, 0, 2))
        self.assertEqual(material_coefficients(100, 3, 0.25, shape_friction=2), (100, 3, 0.5))
        with self.assertRaises(ValueError):
            material_coefficients(0, 0, 0)

    def test_unsupported_solver_modes_rejected(self):
        """Reject settings whose update law has not been wired or validated."""
        model = SimpleNamespace(device=SimpleNamespace(is_cuda=True))
        for key, value in (
            ("articulated_contact_response", "propagation_colored"),
            ("velocity_iterations", 1),
            ("contact_regularization", 0.01),
            ("contact_warmstart", True),
            ("enable_restitution", True),
            ("friction_mode", "coulomb_newton"),
            ("contact_friction_shared_anchor", True),
        ):
            kwargs = {"solve_mode": "matrix_free", "enable_restitution": False, key: value}
            with self.assertRaisesRegex(ValueError, key):
                SolverFeatherPGS(model, contact_compliance=True, **kwargs)

    def test_static_indentation_and_baseline_failure(self):
        """Recover mg/k indentation that the hard-normal baseline cannot supply."""
        target = -0.3 * 9.81 / 3000.0
        compliant = self.trajectory()
        hard = self.trajectory(compliant=False)
        self.assertLess(abs(compliant[-1, 0] - target), 2e-7)
        self.assertGreater(abs(hard[-1, 0] - target), 0.0009)
        self.assertLess(abs(compliant[-1, 2] - 0.3 * 9.81), 2e-4)

    def test_timestep_and_iteration_convergence(self):
        """Keep static stiffness invariant across substeps and iteration counts."""
        target = -0.3 * 9.81 / 3000.0
        for dt in (0.005, 0.0025, 0.00125):
            for iterations in (1, 8, 16, 128):
                trace = self.trajectory(dt=dt, iterations=iterations)
                self.assertLess(abs(trace[-1, 0] - target), 2e-7)

    def test_no_attraction(self):
        """Return zero tensile impulse even for fast separation while indented."""
        for phi, velocity in ((0.01, 0.0), (0.0, 1.0), (-0.001, 10.0)):
            impulse, residual = solve_reference(
                [[1 / 0.3]], [velocity], [phi], [3000.0], [20.0], dt=0.0025, device=self.device
            )
            self.assertEqual(float(impulse[0]), 0.0)
            self.assertGreaterEqual(float(residual[0]), 0.0)

    def test_damping_reduces_oscillation(self):
        """Reduce early normal oscillation without changing static indentation."""
        undamped = self.trajectory(damping=0, x=-0.002)
        damped = self.trajectory(damping=20, x=-0.002)
        self.assertLess(np.sqrt(np.mean(damped[:80, 1] ** 2)), np.sqrt(np.mean(undamped[:80, 1] ** 2)))
        self.assertLess(abs(damped[-1, 0] - undamped[-1, 0]), 2e-7)

    def test_no_dashpot_force_across_open_gap(self):
        """Avoid a damping force before the predicted trajectory closes the gap."""
        result, _ = solve_reference([[1.0]], [-0.1], [0.01], [3000.0], [10000.0], dt=0.0025, device=self.device)
        self.assertEqual(float(result[0]), 0.0)

    def test_coupled_rows_match_independent_linear_solve(self):
        """Match an independent exact solve on an all-active coupled fixture."""
        matrix = np.array([[4.0, 1.0], [1.0, 3.0]])
        phi = [-0.002, -0.001]
        stiffness, damping = [3000.0, 5000.0], [20.0, 10.0]
        gamma, bias = np.array(
            [normal_coefficients(k, c, p, dt=0.0025) for k, c, p in zip(stiffness, damping, phi, strict=True)]
        ).T
        velocity = np.array([-0.1, -0.2])
        expected = np.linalg.solve(matrix + np.diag(gamma), -(velocity + bias))
        self.assertTrue((expected > 0).all())
        result, residual = solve_reference(matrix, velocity, phi, stiffness, damping, dt=0.0025, device=self.device)
        np.testing.assert_allclose(result, expected, rtol=2e-6, atol=1e-7)
        np.testing.assert_allclose(residual, 0, atol=1e-6)

    def test_area_partition_preserves_normal_load(self):
        """Preserve compliance when one patch is split into equal quadrature rows."""
        for count in (1, 4, 16):
            result, residual = solve_reference(
                np.full((count, count), 1 / 0.3),
                np.full(count, -0.1),
                np.full(count, -0.001),
                np.full(count, 3000 / count),
                np.full(count, 20 / count),
                dt=0.0025,
                iterations=128,
                device=self.device,
            )
            gamma, bias = normal_coefficients(3000, 20, -0.001, dt=0.0025)
            expected = (0.1 - bias) / (1 / 0.3 + gamma)
            self.assertAlmostEqual(float(result.sum()), expected, places=6)
            np.testing.assert_allclose(residual, 0, atol=1e-6)


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestContactComplianceSafety(unittest.TestCase):
    """Reject invalid routing and preserve physical reset and release behavior."""

    @classmethod
    def setUpClass(cls):
        """Select the assigned CUDA device without changing global solver defaults."""
        cls.device = wp.get_device(os.environ.get("HYDRO_TEST_DEVICE", "cuda:0"))

    def contacts(self, *, count=0):
        """Create bounded material buffers for guard-only tests."""
        return SimpleNamespace(
            rigid_contact_count=wp.array([count], dtype=int, device=self.device),
            rigid_contact_max=4,
            rigid_contact_stiffness=wp.zeros(4, device=self.device),
            rigid_contact_damping=wp.zeros(4, device=self.device),
            rigid_contact_friction=wp.zeros(4, device=self.device),
        )

    def test_input_overflow_and_reduction_rejected(self):
        """Fail explicitly for lost candidates or unqualified body-pair reduction."""
        fixture = make_fixture(articulated=True, enabled=True)
        model, contacts = fixture.model, fixture.contacts
        state, output = model.state(), model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        fixture.pipeline.collide(state, contacts)
        contacts.rigid_contact_stiffness.fill_(3000.0)
        contacts.rigid_contact_count.fill_(contacts.rigid_contact_max + 1)
        with self.assertRaisesRegex(RuntimeError, "overflowing contact"):
            fixture.solver.step(state, output, model.control(), contacts, 0.005)
        fixture.pipeline.collide(state, contacts)
        contacts.rigid_contacts_body_pair_reduced = True
        with self.assertRaisesRegex(ValueError, "contact_compliance=True is not validated for body-pair"):
            fixture.solver.step(state, output, model.control(), contacts, 0.005)

    def test_nonfinite_inputs_rejected(self):
        """Reject invalid time steps instead of emitting infinite row coefficients."""
        solver = SimpleNamespace(model=SimpleNamespace(device=self.device))
        contacts = self.contacts()
        for dt in (0, -1, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "dt"):
                start_step(solver, contacts, dt)

    def test_missing_material_arrays_rejected(self):
        """Require the per-contact material arrays that the compliant law reads."""
        solver = SimpleNamespace(model=SimpleNamespace(device=self.device))
        for name in ("rigid_contact_stiffness", "rigid_contact_damping", "rigid_contact_friction"):
            contacts = self.contacts()
            setattr(contacts, name, None)
            with self.subTest(missing=name), self.assertRaisesRegex(ValueError, "per-contact material arrays"):
                start_step(solver, contacts, 0.005)
        with self.assertRaisesRegex(ValueError, "per-contact material arrays"):
            start_step(solver, None, 0.005)

    def test_invalid_contact_stiffness_rejected(self):
        """Reject a negative or non-finite contact stiffness before writing compliant rows."""
        for stiffness in (-1.0, float("nan")):
            with (
                self.subTest(stiffness=stiffness),
                self.assertRaisesRegex(ValueError, "Invalid exported contact stiffness"),
            ):
                run_fixture(articulated=True, enabled=True, steps=1, stiffness=stiffness)

    def test_persistent_patch_guard(self):
        """Reject positive persistent-anchor bias but permit a disabled dummy patch buffer."""
        solver = SimpleNamespace(
            model=SimpleNamespace(device=self.device),
            _friction_anchor_beta=0.0,
            _friction_patches=object(),
            _contact_torsion_enabled=False,
            _sparse_mass_matrix_size=None,
        )
        start_step(solver, self.contacts(), 0.005)
        solver._friction_anchor_beta = 0.2
        with self.assertRaisesRegex(ValueError, "friction_anchor_beta"):
            start_step(solver, self.contacts(), 0.005)
        solver._contact_compliance = True
        with self.assertRaisesRegex(ValueError, "friction_anchor_beta"):
            SolverFeatherPGS._run_step(solver, None, None, None, None, 0.005, None, None)

    def test_filtered_speculative_contacts_are_skipped(self):
        """Honor the normal allocator's positive-gap exclusion for compliant contacts."""
        for articulated in (False, True):
            _, paths, solver, _ = run_fixture(
                articulated=articulated,
                enabled=True,
                steps=1,
                height=0.052,
                solver_options={"contact_gap_gate": 0.001},
            )
            self.assertEqual(paths, {-1})
            self.assertEqual(solver.compliance_contact_count, 0)
            self.assertEqual(solver.compliance_skipped_contact_count, 1)

    def test_nonresponding_contact_is_skipped(self):
        """Ignore a kinematic-ground pair without treating it as lost capacity."""
        _, paths, solver, _ = run_fixture(articulated=True, enabled=True, steps=1, kinematic=True)
        self.assertEqual(paths, {-1})
        self.assertEqual(solver.compliance_contact_count, 0)
        self.assertEqual(solver.compliance_skipped_contact_count, 1)

    def test_rolled_back_capacity_fails_without_warnings(self):
        """Reject an actual failed reservation even when its counter was rolled back."""
        for articulated in (False, True):
            with self.assertRaisesRegex(RuntimeError, "overflowing solver rows"):
                run_fixture(
                    articulated=articulated,
                    enabled=True,
                    steps=1,
                    solver_options={
                        "articulation_constraint_max": 1,
                        "free_body_constraint_max": 1,
                        "warn_constraint_overflow": False,
                    },
                )

    def test_solver_overflow_and_dropped_contact_rejected(self):
        """Reject overflowing rows and missing mappings for positive stiffness."""
        fixture = make_fixture(articulated=True, enabled=True)
        model, contacts, solver = fixture.model, fixture.contacts, fixture.solver
        state, output = model.state(), model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        fixture.pipeline.collide(state, contacts)
        contacts.rigid_contact_stiffness.fill_(3000.0)
        solver.step(state, output, model.control(), contacts, 0.005)
        solver._constraint_count.fill_(solver._articulation_constraint_max + 1)
        with self.assertRaisesRegex(RuntimeError, "overflowing solver rows"):
            prepare_rows(solver, contacts, 0.005)
        solver._constraint_count.fill_(1)
        solver._contact_slot.fill_(-1)
        with self.assertRaisesRegex(RuntimeError, "dropped"):
            prepare_rows(solver, contacts, 0.005)

    def stepped_fixture(self, *, articulated, height, world_count=1, damping=20.0):
        """Take one compliant step and return the fixture with its contacts still loaded."""
        fixture = make_fixture(articulated=articulated, enabled=True, height=height, world_count=world_count)
        model, contacts = fixture.model, fixture.contacts
        state, output = model.state(), model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        fixture.pipeline.collide(state, contacts)
        contacts.rigid_contact_stiffness.fill_(3000.0)
        contacts.rigid_contact_damping.fill_(damping)
        contacts.rigid_contact_friction.fill_(1.0)
        fixture.solver.step(state, output, model.control(), contacts, 0.005)
        return fixture

    def test_device_coefficients_match_host_law(self):
        """Prepare open-gap and penetrating rows exactly as the float64 host law, damping only when closed."""
        for articulated in (True, False):
            for height in (0.052, 0.0495):
                with self.subTest(articulated=articulated, height=height):
                    solver = self.stepped_fixture(articulated=articulated, height=height, damping=200.0).solver
                    count = solver.compliance_contact_count
                    self.assertEqual(count, 1)
                    path, slot, world = (
                        int(a.numpy()[0]) for a in (solver._contact_path, solver._contact_slot, solver._contact_world)
                    )
                    phi = (solver._phi if path == 0 else solver._mf_phi).numpy()[world, slot]
                    self.assertEqual(phi > 0.0, height > 0.05)
                    gamma, bias = normal_coefficients(3000.0, 200.0, float(phi), dt=0.005)
                    buffers = solver._compliance
                    stored_gamma = (buffers.dense_gamma if path == 0 else buffers.mf_gamma).numpy()[world, slot]
                    stored_bias = (solver._rhs if path == 0 else solver._mf_rhs).numpy()[world, slot]
                    self.assertEqual(stored_gamma, np.float32(gamma))
                    self.assertEqual(stored_bias, np.float32(bias))

    def test_duplicate_row_mapping_rejected(self):
        """Two compliant contacts must never share one normal row."""
        for articulated in (True, False):
            with self.subTest(articulated=articulated):
                fixture = self.stepped_fixture(articulated=articulated, height=0.05, world_count=2)
                solver, contacts = fixture.solver, fixture.contacts
                self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 2)
                for name in ("_contact_path", "_contact_slot", "_contact_world"):
                    values = getattr(solver, name).numpy()
                    values[1] = values[0]
                    getattr(solver, name).assign(values)
                begin_rows(solver)
                with self.assertRaisesRegex(RuntimeError, "one-to-one"):
                    prepare_rows(solver, contacts, 0.005)

    def test_reset_and_open_gap_release(self):
        """Clear only step-local compliance data and exert no force across an open gap."""
        _, _, solver, _ = run_fixture(articulated=True, enabled=True, steps=5)
        model = solver.model
        state = model.state()
        state.joint_q.assign(np.array([0.01], dtype=np.float32))
        state.joint_qd.assign(np.array([0.2], dtype=np.float32))
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        before = {name: getattr(state, name).numpy().copy() for name in ("joint_q", "joint_qd", "body_q", "body_qd")}
        solver.reset(state)
        self.assertEqual(solver.compliance_contact_count, 0)
        self.assertEqual(solver.compliance_skipped_contact_count, 0)
        for name, value in before.items():
            np.testing.assert_array_equal(getattr(state, name).numpy(), value)
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=32)
        contacts = pipeline.contacts()
        for field in ("rigid_contact_stiffness", "rigid_contact_damping", "rigid_contact_friction"):
            setattr(contacts, field, wp.zeros(32, device=self.device))
        pipeline.collide(state, contacts)
        self.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 0)
        output = model.state()
        solver.step(state, output, model.control(), contacts, 0.005)
        self.assertEqual(solver.compliance_contact_count, 0)
        self.assertAlmostEqual(float(output.joint_qd.numpy()[0]), 0.2 - 9.81 * 0.005, delta=1e-6)
        self.assertTrue(np.isfinite(output.body_q.numpy()).all())

    def test_actual_timestep_and_iterations(self):
        """Keep native row stiffness invariant under smaller timesteps and more iterations."""
        for dt, iterations in ((0.005, 1), (0.0025, 16)):
            trace, _, _, mass = run_fixture(
                articulated=True, enabled=True, dt=dt, iterations=iterations, steps=round(1 / dt)
            )
            self.assertAlmostEqual(float(trace[-1, 2]), 0.05 - mass * 9.81 / 3000, delta=2e-6)

    def test_public_defaults_remain_hard(self):
        """Expose an explicit default-off public constructor option."""
        parameter = inspect.signature(SolverFeatherPGS.__init__).parameters["contact_compliance"]
        self.assertIs(parameter.default, False)
        self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)


def _materials(contacts, stiffness):
    contacts.rigid_contact_stiffness.fill_(stiffness)
    contacts.rigid_contact_damping.fill_(20.0)
    contacts.rigid_contact_friction.fill_(1.0)


def _run_free_spheres(*, dense_rows, graph, steps=20):
    """Four free spheres in one world: their matrix-free slots exceed a one-row dense capacity."""
    device = os.environ.get("HYDRO_TEST_DEVICE", "cuda:0")
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.add_ground_plane()
        for i in range(4):
            body = builder.add_body(xform=wp.transform((i * 0.3, 0.0, 0.049), wp.quat_identity()))
            builder.add_shape_sphere(body, radius=0.05)
        model = builder.finalize()
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=32)
        model.rigid_contact_max = 32
        contacts = pipeline.contacts()
        for name in ("rigid_contact_stiffness", "rigid_contact_damping", "rigid_contact_friction"):
            setattr(contacts, name, wp.zeros(32, dtype=float))
        solver = newton.solvers.SolverFeatherPGS(
            model,
            solve_mode="matrix_free",
            contact_compliance=True,
            friction_anchor_beta=0.0,
            enable_restitution=False,
            articulation_constraint_max=dense_rows,
            free_body_constraint_max=32,
            iterations=8,
        )
        states = (model.state(), model.state())
        newton.eval_fk(model, model.joint_q, model.joint_qd, states[0])
        control = model.control()

        def two_steps():
            for source, target in (states, states[::-1]):
                source.clear_forces()
                pipeline.collide(source, contacts)
                _materials(contacts, 3000.0)
                solver.step(source, target, control, contacts, 0.005)

        trace = []
        two_steps()
        trace.append(states[0].body_q.numpy().copy())
        slots = solver._contact_slot.numpy()[: int(contacts.rigid_contact_count.numpy()[0])].copy()
        captured = None
        if graph:
            with wp.ScopedCapture() as capture:
                two_steps()
            captured = capture.graph
        for _ in range(steps - 1):
            if graph:
                wp.capture_launch(captured)
            else:
                two_steps()
            solver.validate_contact_compliance()
            trace.append(states[0].body_q.numpy().copy())
        return solver, np.asarray(trace), slots


class _Episode:
    """Two fixed state buffers stepped A->B->A, so one captured graph covers two steps."""

    def __init__(self, *, stiffness=3000.0, dt=0.005, **fixture_options):
        self.fixture = make_fixture(**fixture_options)
        self.model = self.fixture.model
        self.stiffness = stiffness
        self.dt = dt
        self.states = (self.model.state(), self.model.state())
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.states[0])
        self.control = self.model.control()

    def two_steps(self):
        fixture = self.fixture
        for source, target in (self.states, self.states[::-1]):
            source.clear_forces()
            fixture.pipeline.collide(source, fixture.contacts)
            _materials(fixture.contacts, self.stiffness)
            fixture.solver.step(source, target, self.control, fixture.contacts, self.dt)

    def snapshot(self):
        state = self.states[0]
        return {name: getattr(state, name).numpy().copy() for name in ("body_q", "body_qd", "joint_q", "joint_qd")}

    def run(self, pairs, *, graph=False, between=None):
        """Advance ``2 * pairs`` steps eagerly or by replaying one captured two-step graph."""
        trace = []
        captured = None
        with wp.ScopedDevice(self.fixture.device):
            if graph:
                # Kernel modules load outside capture.
                self.two_steps()
                trace.append(self.snapshot())
                with wp.ScopedCapture() as capture:
                    self.two_steps()
                captured = capture.graph
            for pair in range(len(trace), pairs):
                if between is not None:
                    between(self, pair)
                if graph:
                    wp.capture_launch(captured)
                    self.fixture.solver.validate_contact_compliance()
                else:
                    self.two_steps()
                trace.append(self.snapshot())
        return trace


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestContactComplianceCapture(unittest.TestCase):
    """Replay must reproduce eager compliant steps and keep their error contract."""

    def assert_traces_equal(self, eager, replay):
        self.assertEqual(len(eager), len(replay))
        for step, (a, b) in enumerate(zip(eager, replay, strict=True)):
            for name in a:
                np.testing.assert_array_equal(a[name], b[name], err_msg=f"{name} at pair {step}")

    def test_replay_matches_eager(self):
        """Dense and free-body routes, one and four worlds, with contacts appearing during replay."""
        for articulated in (True, False):
            for world_count in (1, 4):
                with self.subTest(articulated=articulated, worlds=world_count):
                    # Starting 2 cm above the gap gives zero contacts for the first steps.
                    options = {"articulated": articulated, "enabled": True, "world_count": world_count, "height": 0.08}
                    eager = _Episode(**options).run(60)
                    replay_episode = _Episode(**options)
                    replay = replay_episode.run(60, graph=True)
                    self.assert_traces_equal(eager, replay)
                    self.assertGreater(replay_episode.fixture.solver.compliance_contact_count, 0)
                    target = 0.05 - 0.3 * 9.81 / 3000.0
                    z = replay[-1]["body_q"][replay_episode.fixture.bodies, 2]
                    self.assertLess(np.max(np.abs(z - target)), 5e-4)

    def test_replay_zero_stiffness_matches_hard_contacts(self):
        """Compliance ON with zero stiffness stays the hard law under replay."""
        for articulated in (True, False):
            with self.subTest(articulated=articulated):
                hard = _Episode(articulated=articulated, enabled=False, stiffness=0.0).run(30, graph=True)
                noop_episode = _Episode(articulated=articulated, enabled=True, stiffness=0.0)
                noop = noop_episode.run(30, graph=True)
                self.assert_traces_equal(hard, noop)
                self.assertEqual(noop_episode.fixture.solver.compliance_contact_count, 0)

    def test_reset_between_replays_matches_eager(self):
        """A solver reset between replays acts as it does between eager steps."""

        def reset(episode, pair):
            if pair == 10:
                episode.fixture.solver.reset(episode.states[0])

        eager = _Episode(articulated=True, enabled=True).run(20, between=reset)
        replay = _Episode(articulated=True, enabled=True).run(20, graph=True, between=reset)
        self.assert_traces_equal(eager, replay)

    def test_replay_latches_input_overflow(self):
        """Replay never raises by itself; validation reports an overflowing step and then clears."""
        episode = _Episode(articulated=True, enabled=True)
        fixture = episode.fixture
        solver, contacts = fixture.solver, fixture.contacts
        with wp.ScopedDevice(fixture.device):
            state, output = episode.states
            fixture.pipeline.collide(state, contacts)
            _materials(contacts, 3000.0)
            solver.step(state, output, episode.control, contacts, episode.dt)
            with wp.ScopedCapture() as capture:
                solver.step(state, output, episode.control, contacts, episode.dt)
            contacts.rigid_contact_count.fill_(contacts.rigid_contact_max + 1)
            wp.capture_launch(capture.graph)
            with self.assertRaisesRegex(RuntimeError, "overflowing contact"):
                solver.validate_contact_compliance()
            fixture.pipeline.collide(state, contacts)
            _materials(contacts, 3000.0)
            wp.capture_launch(capture.graph)
            solver.validate_contact_compliance()
            self.assertGreater(solver.compliance_contact_count, 0)

    def test_replay_latches_row_capacity_loss(self):
        """A contact that first appears during replay and cannot get rows invalidates that replay."""
        for articulated in (True, False):
            with self.subTest(articulated=articulated):
                episode = _Episode(
                    articulated=articulated,
                    enabled=True,
                    height=0.08,
                    solver_options={
                        "articulation_constraint_max": 1,
                        "free_body_constraint_max": 1,
                        "warn_constraint_overflow": False,
                    },
                )
                fixture = episode.fixture
                with wp.ScopedDevice(fixture.device):
                    # Airborne: no contacts, so the warm-up and capture are valid.
                    episode.two_steps()
                    fixture.solver.validate_contact_compliance()
                    with wp.ScopedCapture() as capture:
                        episode.two_steps()
                    with self.assertRaisesRegex(RuntimeError, "overflowing solver rows"):
                        for _ in range(40):
                            wp.capture_launch(capture.graph)
                            fixture.solver.validate_contact_compliance()

    def test_unequal_row_capacities(self):
        """Free-body rows beyond the dense capacity prepare as with a large dense capacity, eager and replayed."""
        traces, counts = {}, {}
        for dense_rows in (1, 32):
            for graph in (False, True):
                solver, trace, slots = _run_free_spheres(dense_rows=dense_rows, graph=graph)
                counts[dense_rows, graph] = solver.compliance_contact_count
                if dense_rows == 1:
                    self.assertGreaterEqual(int(slots.max()), dense_rows)
                traces[dense_rows, graph] = trace
        self.assertGreaterEqual(counts[32, False], 4)
        for key, trace in traces.items():
            with self.subTest(dense_rows=key[0], graph=key[1]):
                self.assertEqual(counts[key], counts[32, False])
                np.testing.assert_array_equal(trace, traces[32, False])

    def test_validate_rejects_capture(self):
        """Status readback is refused inside a capture."""
        episode = _Episode(articulated=True, enabled=True)
        with wp.ScopedDevice(episode.fixture.device):
            with self.assertRaisesRegex(RuntimeError, "outside CUDA graph capture"):
                with wp.ScopedCapture():
                    episode.fixture.solver.validate_contact_compliance()


def _reducer(fixture):
    """Body-pair reducer writing the fixture's Contacts buffer."""
    return newton.CollisionPipeline(
        fixture.model,
        broad_phase="nxn",
        deterministic=True,
        rigid_contact_max=fixture.contacts.rigid_contact_max,
        reduce_contacts=newton.CollisionPipeline.ContactReductionConfig(body_pairs=True),
    )


def _contact_snapshot(contacts):
    count = int(contacts.rigid_contact_count.numpy()[0])
    return (
        count,
        contacts.rigid_contact_normal.numpy()[:count].copy(),
        contacts.rigid_contact_point0.numpy()[:count].copy(),
    )


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestContactComplianceContactsLease(unittest.TestCase):
    """A captured compliant step must keep body-pair reduction off its Contacts buffer."""

    def assert_reducer_rejected(self, episode, reducer):
        """Eager and captured reducer writes both fail before touching the buffer."""
        contacts = episode.fixture.contacts
        state = episode.states[0]
        before = _contact_snapshot(contacts)
        with self.assertRaisesRegex(RuntimeError, "unreduced-only solver configuration"):
            reducer.collide(state, contacts)
        capture = wp.ScopedCapture()
        capture.__enter__()
        try:
            with self.assertRaisesRegex(RuntimeError, "unreduced-only solver configuration"):
                reducer.collide(state, contacts)
        finally:
            capture.__exit__(None, None, None)
        self.assertFalse(contacts.rigid_contacts_body_pair_reduced)
        self.assertFalse(contacts.rigid_contacts_body_pair_reduced_capture)
        after = _contact_snapshot(contacts)
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        np.testing.assert_array_equal(before[2], after[2])

    def test_captured_step_blocks_reducer_and_replays(self):
        """Consumer first: reducer writes are refused while the replayed graph still matches eager."""
        options = {"articulated": True, "enabled": True}
        eager = _Episode(**options).run(12)
        episode = _Episode(**options)
        reducer = _reducer(episode.fixture)
        attempted = []

        def write_reduced(episode, pair):
            if pair in (3, 7):
                self.assert_reducer_rejected(episode, reducer)
                attempted.append(pair)

        replay = episode.run(12, graph=True, between=write_reduced)
        self.assertEqual(attempted, [3, 7])
        self.assertEqual(len(eager), len(replay))
        for pair, (a, b) in enumerate(zip(eager, replay, strict=True)):
            for name in a:
                np.testing.assert_array_equal(a[name], b[name], err_msg=f"{name} at pair {pair}")
        self.assertGreater(episode.fixture.solver.compliance_contact_count, 0)

    def test_reader_lease_lives_with_graph(self):
        """Lifetime: the lease outlives the capture object and ends with the last graph reference."""
        episode = _Episode(articulated=False, enabled=True)
        fixture = episode.fixture
        reducer = _reducer(fixture)
        with wp.ScopedDevice(fixture.device):
            episode.two_steps()
            with wp.ScopedCapture() as capture:
                episode.two_steps()
            graph = capture.graph
            graph_ref = weakref.ref(graph)
            del capture
            gc.collect()
            wp.capture_launch(graph)
            fixture.solver.validate_contact_compliance()
            self.assert_reducer_rejected(episode, reducer)
            del graph
            gc.collect()
            self.assertIsNone(graph_ref())
            state, output = episode.states
            reducer.collide(state, fixture.contacts)
            self.assertTrue(fixture.contacts.rigid_contacts_body_pair_reduced)
            _materials(fixture.contacts, 3000.0)
            # Already-reduced eager input still rejects with the configuration named.
            with self.assertRaisesRegex(ValueError, "contact_compliance=True is not validated for body-pair"):
                fixture.solver.step(state, output, episode.control, fixture.contacts, episode.dt)

    def test_live_reducer_graph_rejects_step(self):
        """Producer first: a live reducer graph rejects compliant steps after an ordinary refill."""
        episode = _Episode(articulated=True, enabled=True)
        fixture = episode.fixture
        contacts = fixture.contacts
        reducer = _reducer(fixture)
        state, output = episode.states
        with wp.ScopedDevice(fixture.device):
            episode.two_steps()
            reducer.collide(state, contacts)
            with wp.ScopedCapture() as reducer_capture:
                reducer.collide(state, contacts)
            wp.capture_launch(reducer_capture.graph)
            fixture.pipeline.collide(state, contacts)
            _materials(contacts, 3000.0)
            self.assertFalse(contacts.rigid_contacts_body_pair_reduced)
            self.assertTrue(contacts.rigid_contacts_body_pair_reduced_capture)
            body_q = output.body_q.numpy().copy()
            with self.assertRaisesRegex(ValueError, "contact_compliance=True is not validated for body-pair"):
                fixture.solver.step(state, output, episode.control, contacts, episode.dt)
            np.testing.assert_array_equal(output.body_q.numpy(), body_q)
            solver_capture = wp.ScopedCapture()
            solver_capture.__enter__()
            try:
                with self.assertRaisesRegex(ValueError, "contact_compliance=True is not validated for body-pair"):
                    fixture.solver.step(state, output, episode.control, contacts, episode.dt)
            finally:
                solver_capture.__exit__(None, None, None)
            del reducer_capture, solver_capture
            gc.collect()
            reducer.release_body_pair_reduction_capture()
            self.assertFalse(contacts.rigid_contacts_body_pair_reduced_capture)
            fixture.pipeline.collide(state, contacts)
            _materials(contacts, 3000.0)
            fixture.solver.step(state, output, episode.control, contacts, episode.dt)
            fixture.solver.validate_contact_compliance()

    def test_noncompliant_capture_keeps_reduction(self):
        """Control: without compliance, FeatherPGS steps reduced contacts and its graph leaves reduction available."""
        episode = _Episode(articulated=True, enabled=False)
        fixture = episode.fixture
        contacts = fixture.contacts
        reducer = _reducer(fixture)
        state, output = episode.states
        with wp.ScopedDevice(fixture.device):
            episode.two_steps()
            reducer.collide(state, contacts)
            self.assertTrue(contacts.rigid_contacts_body_pair_reduced)
            fixture.solver.step(state, output, episode.control, contacts, episode.dt)
            self.assertTrue(np.isfinite(output.body_q.numpy()).all())
            with wp.ScopedCapture() as capture:
                episode.two_steps()
            wp.capture_launch(capture.graph)
            reducer.collide(state, contacts)
            self.assertTrue(contacts.rigid_contacts_body_pair_reduced)
            self.assertFalse(contacts.rigid_contacts_body_pair_reduced_capture)
            wp.capture_launch(capture.graph)
            self.assertTrue(np.isfinite(episode.states[0].body_q.numpy()).all())


DT = 1.0 / 60.0


PATH_DENSE = 0  # contact routing id of the dense Delassus path


def _solve_rows(device, n_rows, g, iterations=300):
    """Solve ``n_rows`` identical unit contact rows on one 6-DOF body with unit
    mass and rhs = -1 through the reference MF loop kernel."""
    m = 8
    J = np.zeros((1, m, 6), dtype=np.float32)
    for r in range(n_rows):
        J[0, r, 2] = 1.0
    row_type = np.zeros((1, m), dtype=np.int32)
    row_type[0, :n_rows] = PGS_CONSTRAINT_TYPE_CONTACT
    eff_inv = np.zeros((1, m), dtype=np.float32)
    eff_inv[0, :n_rows] = 1.0
    rhs = np.zeros((1, m), dtype=np.float32)
    rhs[0, :n_rows] = -1.0
    body_a = np.full((1, m), -1, dtype=np.int32)
    body_a[0, :n_rows] = 0

    mf_impulses = wp.zeros((1, m), dtype=wp.float32)
    v_out = wp.zeros(6, dtype=wp.float32)
    wp.launch(
        pgs_solve_mf_loop,
        dim=1,
        inputs=[
            wp.array([n_rows], dtype=wp.int32),
            wp.array(body_a, dtype=wp.int32),
            wp.full((1, m), -1, dtype=wp.int32),  # body_b
            wp.array(J, dtype=wp.float32),  # MiJt_a (unit mass)
            wp.zeros((1, m, 6), dtype=wp.float32),  # MiJt_b
            wp.array(J, dtype=wp.float32),  # J_a
            wp.zeros((1, m, 6), dtype=wp.float32),  # J_b
            wp.array(eff_inv, dtype=wp.float32),
            wp.array(rhs, dtype=wp.float32),
            wp.array(row_type, dtype=wp.int32),
            wp.full((1, m), -1, dtype=wp.int32),  # row_parent
            wp.zeros((1, m), dtype=wp.float32),  # row_mu
            wp.full((1, m), 1.0 / (1.0 + g), dtype=wp.float32),  # mf_row_w
            wp.array([0], dtype=wp.int32),  # body_to_articulation
            wp.array([0], dtype=wp.int32),  # art_dof_start
            iterations,
            1.0,  # omega
            1,  # regularize
            iterations,  # friction_start_iteration (no friction rows)
            0,  # iteration_offset
            mf_impulses,
            v_out,
        ],
        device=device,
    )
    return mf_impulses.numpy()[0, :n_rows]


def test_regularization_fixed_points(test: unittest.TestCase, device):
    """Analytic contracts of the regularized update on unit rows:

    - one determined row converges to ``1/(1+g)`` of the rigid impulse (the
      documented load deficit);
    - two redundant identical rows converge to the unique symmetric split
      ``lambda_1 = lambda_2 = 1/(2+g)``;
    - ``g = 0`` reproduces the exact rigid law (total impulse 1).
    """
    with wp.ScopedDevice(device):
        g = 0.5
        lam = _solve_rows(device, 1, g)
        test.assertAlmostEqual(float(lam[0]), 1.0 / (1.0 + g), places=5)
        lam = _solve_rows(device, 2, g)
        test.assertAlmostEqual(float(lam[0]), 1.0 / (2.0 + g), places=5)
        test.assertAlmostEqual(float(lam[1]), 1.0 / (2.0 + g), places=5)
        lam = _solve_rows(device, 2, 0.0)
        test.assertAlmostEqual(float(lam.sum()), 1.0, places=5)


def _stack_scene(device, g, solve_mode="matrix_free", response="immediate", velocity_iterations=2):
    builder = newton.ModelBuilder()
    builder.rigid_gap = 0.003
    cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.7)
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.7))
    bodies = []
    for k in range(3):
        b = builder.add_body(xform=wp.transform(wp.vec3(0.002 * k, 0.0, 0.0505 + 0.101 * k), wp.quat_identity()))
        builder.add_shape_box(b, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
        bodies.append(b)
    model = builder.finalize()
    pipeline = newton.CollisionPipeline(
        model,
        reduce_contacts=True,
        rigid_contact_max=128,
        broad_phase="nxn",
        deterministic=True,
        contact_matching="latest",
    )
    solver = newton.solvers.SolverFeatherPGS(
        model,
        solve_mode=solve_mode,
        iterations=8,
        velocity_iterations=velocity_iterations,
        articulated_contact_response=response,
        contact_regularization=g,
        contact_warmstart=True,
    )
    return model, pipeline, solver, bodies


def _run(model, pipeline, solver, frames, dt=DT):
    contacts = pipeline.contacts()
    s0, s1 = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    zs = []
    for _k in range(frames):
        pipeline.collide(s0, contacts)
        s0.clear_forces()
        solver.step(s0, s1, control, contacts, dt)
        s0, s1 = s1, s0
        zs.append(float(s0.body_q.numpy()[-1][2]))
    return s0.body_q.numpy(), zs


def test_regularization_route_agreement(test: unittest.TestCase, device):
    """The regularized update must agree across the matrix-free execution
    routes: the fused MF-GS kernel (matrix_free/immediate), the standalone
    MF kernel (split/immediate), and the propagation_fused full-iteration
    kernel: same scene, same g, matching resting poses."""
    with wp.ScopedDevice(device):
        results = {}
        for label, mode, response, velit in (
            ("fused", "matrix_free", "immediate", 2),
            # velocity iterations are matrix_free-only; split runs without
            ("standalone", "split", "immediate", 0),
            ("propagation_fused", "matrix_free", "propagation_fused", 2),
        ):
            model, pipeline, solver, _bodies = _stack_scene(
                device, g=0.05, solve_mode=mode, response=response, velocity_iterations=velit
            )
            body_q, _ = _run(model, pipeline, solver, 120)
            results[label] = body_q
        ref = results["fused"]
        for label, body_q in results.items():
            np.testing.assert_allclose(
                body_q[:, :3], ref[:, :3], atol=2.0e-3, err_msg=f"route {label} diverged from fused"
            )


def _resting_box(device, g, solve_mode, rate, frames, **solver_kwargs):
    builder = newton.ModelBuilder()
    builder.rigid_gap = 0.01
    cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.7)
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.7))
    b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.05), wp.quat_identity()))
    builder.add_shape_box(b, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
    model = builder.finalize()
    pipeline = newton.CollisionPipeline(
        model,
        reduce_contacts=True,
        rigid_contact_max=32,
        broad_phase="nxn",
        deterministic=True,
        contact_matching="latest",
    )
    solver = newton.solvers.SolverFeatherPGS(
        model,
        solve_mode=solve_mode,
        iterations=12,
        contact_regularization=g,
        **solver_kwargs,
    )
    _, zs = _run(model, pipeline, solver, frames, dt=1.0 / rate)
    return 0.05 - zs[-1]


def _sag_formula(g, rate):
    """Resting sag of a body under gravity: ``g * a * dt^2 / baumgarte`` with the default beta."""
    return g * 9.81 / (rate * rate) / 0.2


def test_regularization_documented_sag(test: unittest.TestCase, device):
    """The regularizer is a numerical damped compliance: a resting box sags by
    ``g * a * dt^2 / beta`` (6.8 mm at g = 0.5 and 60 Hz, 0.43 mm at 240 Hz), on the
    matrix-free route."""
    with wp.ScopedDevice(device):
        for rate in (60, 240):
            sag = _resting_box(device, 0.5, "matrix_free", rate, 3 * rate, contact_warmstart=True)
            expected = _sag_formula(0.5, rate)
            test.assertAlmostEqual(sag, expected, delta=0.15 * expected, msg=f"{rate} Hz: sag {sag * 1000:.2f} mm")


def test_regularization_split_route_documented_sag(test: unittest.TestCase, device):
    """The split route's standalone matrix-free kernel carries the same law as the
    fused kernel, so its resting box sags by the same formula (CPU-runnable)."""
    with wp.ScopedDevice(device):
        sag = _resting_box(device, 0.5, "split", 60, 180, contact_warmstart=True)
        expected = _sag_formula(0.5, 60)
        test.assertAlmostEqual(sag, expected, delta=0.15 * expected, msg=f"split sag {sag * 1000:.2f} mm")


def test_regularization_velocity_pass_exempt(test: unittest.TestCase, device):
    """The velocity-only pass solves the exact rigid law: a settled stack holds
    its height to well under a millimetre over the last second."""
    with wp.ScopedDevice(device):
        for response in ("immediate", "propagation_fused"):
            model, pipeline, solver, _bodies = _stack_scene(device, g=0.05, response=response, velocity_iterations=4)
            _, zs = _run(model, pipeline, solver, 240)
            drift = abs(zs[-1] - zs[179])
            test.assertLess(
                drift, 5.0e-4, f"{response}: top box kept sinking ({drift * 1000:.2f} mm over the last second)"
            )


def test_velocity_pass_rigid_on_propagation_schedules(test: unittest.TestCase, device):
    """Same-articulation link-link contacts stay dense under the ``propagation`` and
    ``propagation_colored`` schedules, and their velocity-only pass runs through
    nested fused-GS launches inside the propagation-tree launcher. Every one of
    those launches must be rigid (``soft_relax=True``), and the converged velocity
    after the pass must be the ``g = 0`` solution, since the rigid law does not
    depend on ``g``."""

    def stepped(response, g):
        model = _build_scissor_model(device)
        solver = newton.solvers.SolverFeatherPGS(
            model,
            solve_mode="matrix_free",
            articulated_contact_response=response,
            iterations=8,
            velocity_iterations=8,
            contact_regularization=g,
            contact_friction_scale=0.0,
            articulation_constraint_max=64,
            free_body_constraint_max=16,
        )
        calls = []
        in_velocity_pass = {"on": False}
        launch = solver._launch_matrix_free_gs_solve
        post = solver._run_matrix_free_velocity_post_solve

        def recording_launch(*args, **kwargs):
            calls.append((in_velocity_pass["on"], bool(kwargs.get("soft_relax", False))))
            return launch(*args, **kwargs)

        def flagged_post():
            in_velocity_pass["on"] = True
            try:
                return post()
            finally:
                in_velocity_pass["on"] = False

        solver._launch_matrix_free_gs_solve = recording_launch
        solver._run_matrix_free_velocity_post_solve = flagged_post
        state_in, state_out = model.state(), model.state()
        newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()
        state_in.clear_forces()
        collision_pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, model.control(), contacts, 1.0 / 200.0)
        wp.synchronize()
        test.assertTrue(
            _contact_rows(solver, PATH_DENSE, int(contacts.rigid_contact_count.numpy()[0])),
            f"{response}: scene produced no dense self-contact row",
        )
        return calls, state_out.joint_qd.numpy().copy()

    with wp.ScopedDevice(device):
        for response in ("propagation", "propagation_colored"):
            calls, qd_soft = stepped(response, 0.5)
            velocity_calls = [soft for on, soft in calls if on]
            position_calls = [soft for on, soft in calls if not on]
            test.assertTrue(velocity_calls, f"{response}: velocity pass issued no fused-GS launch")
            test.assertTrue(all(velocity_calls), f"{response}: velocity-pass launches were regularized: {calls}")
            test.assertFalse(any(position_calls), f"{response}: position-pass launches were rigid: {calls}")
            _, qd_rigid = stepped(response, 0.0)
            np.testing.assert_allclose(
                qd_soft, qd_rigid, rtol=0.0, atol=1.0e-4, err_msg=f"{response}: velocity pass depends on g"
            )


def test_regularization_indeterminate_split(test: unittest.TestCase, device):
    """A plank on three identical supports has no unique rigid force split;
    the regularizer must select the symmetric one (outer supports equal)."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder()
        builder.rigid_gap = 0.003
        cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.7)
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.7))
        supports = []
        for k in range(3):
            b = builder.add_body(xform=wp.transform(wp.vec3(0.15 * (k - 1), 0.0, 0.0255), wp.quat_identity()))
            s = builder.add_shape_box(b, hx=0.025, hy=0.025, hz=0.025, cfg=cfg)
            supports.append(s)
        plank = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0605), wp.quat_identity()))
        builder.add_shape_box(plank, hx=0.2, hy=0.03, hz=0.01, cfg=cfg)
        model = builder.finalize()
        pipeline = newton.CollisionPipeline(
            model,
            reduce_contacts=True,
            rigid_contact_max=256,
            broad_phase="nxn",
            deterministic=True,
            contact_matching="latest",
        )
        solver = newton.solvers.SolverFeatherPGS(
            model,
            solve_mode="matrix_free",
            iterations=8,
            velocity_iterations=2,
            contact_regularization=0.02,
            contact_warmstart=True,
        )
        observables = solver.observables({newton.solvers.SolverObservableFlags.CONTACT_F})
        contacts = pipeline.contacts()
        s0, s1 = model.state(), model.state()
        control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
        for _k in range(120):
            pipeline.collide(s0, contacts)
            s0.clear_forces()
            solver.step(s0, s1, control, contacts, DT, observables=observables)
            s0, s1 = s1, s0

        count = int(contacts.rigid_contact_count.numpy()[0])
        shape0 = contacts.rigid_contact_shape0.numpy()[:count]
        shape1 = contacts.rigid_contact_shape1.numpy()[:count]
        force = observables.contact_f.numpy()[:count, :3]
        loads = []
        for s in supports:
            fz = 0.0
            for i in range(count):
                # plank-support pairs only (exclude support-ground)
                if s in (shape0[i], shape1[i]) and 0 not in (shape0[i], shape1[i]):
                    fz += abs(float(force[i][2]))
            loads.append(fz)
        test.assertGreater(min(loads), 0.0, f"a support carries no load: {loads}")
        test.assertAlmostEqual(
            loads[0],
            loads[2],
            delta=0.1 * max(loads),
            msg=f"outer supports asymmetric: {loads}",
        )


def test_regularization_validation(test: unittest.TestCase, device):
    """Parameter contract: finite non-negative values only."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
        builder.add_shape_box(b, hx=0.05, hy=0.05, hz=0.05)
        model = builder.finalize()
        for bad in (-0.1, float("nan"), float("inf"), 1.0e7):
            with test.assertRaises(ValueError):
                newton.solvers.SolverFeatherPGS(model, contact_regularization=bad)
        newton.solvers.SolverFeatherPGS(model, contact_regularization=0.0)
        newton.solvers.SolverFeatherPGS(model, contact_regularization=1.0e6)


def test_zero_regularization_shares_one_weight_slot(test: unittest.TestCase, device):
    """The exact-rigid path carries no capacity-sized per-row weight buffers."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
        builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05)
        model = builder.finalize()
        solve_mode = "split" if wp.get_device(device).is_cpu else "matrix_free"
        solver = newton.solvers.SolverFeatherPGS(model, solve_mode=solve_mode)
        test.assertFalse(solver._regularization_enabled)
        test.assertIs(solver._row_w, solver._contact_row_w_dummy)
        test.assertIs(solver._mf_row_w, solver._contact_row_w_dummy)
        test.assertEqual(solver._row_w.shape, (1, 1))


def test_zero_regularization_split_runtime_uses_no_weights(test: unittest.TestCase, device):
    """The standalone MF route must not read its one-element inactive weight argument."""
    with wp.ScopedDevice(device):
        model, pipeline, solver, _bodies = _stack_scene(
            device, g=0.0, solve_mode="split", response="immediate", velocity_iterations=0
        )
        body_q, _zs = _run(model, pipeline, solver, 1)
        test.assertGreater(int(solver._mf_constraint_count.numpy().sum()), 1)
        test.assertTrue(np.isfinite(body_q).all())
        test.assertIs(solver._mf_row_w, solver._contact_row_w_dummy)
        test.assertEqual(float(solver._mf_row_w.numpy()[0, 0]), 1.0)


def test_exact_surface_contact_is_regularized(test: unittest.TestCase, device):
    """A zero-gap contact is active; only a strictly positive gap is speculative."""
    with wp.ScopedDevice(device):
        row_w = wp.zeros((1, 1), dtype=wp.float32, device=device)
        wp.launch(
            compute_world_contact_bias,
            dim=1,
            inputs=[
                wp.array([1], dtype=wp.int32, device=device),
                wp.zeros((1, 1), dtype=wp.float32, device=device),
                wp.full((1, 1), 0.2, dtype=wp.float32, device=device),
                wp.full((1, 1), PGS_CONSTRAINT_TYPE_CONTACT, dtype=wp.int32, device=device),
                wp.zeros((1, 1), dtype=wp.float32, device=device),
                DT,
                1.0,
                1.0,
                1.0,
                0.5,
            ],
            outputs=[wp.zeros((1, 1), dtype=wp.float32, device=device), row_w],
            device=device,
        )
        test.assertEqual(float(row_w.numpy()[0, 0]), 0.5)


def test_assembled_dense_regularization_scales_diagonal(test: unittest.TestCase, device):
    """An actual articulated dense contact receives the same g*d diagonal term."""
    with wp.ScopedDevice(device):
        model = _build_scissor_model(device)
        solver = newton.solvers.SolverFeatherPGS(
            model,
            solve_mode="split",
            iterations=0,
            velocity_iterations=0,
            contact_regularization=0.5,
            contact_friction_scale=0.0,
            articulation_constraint_max=64,
            free_body_constraint_max=16,
        )
        unregularized_diag = {}
        apply_regularization = solver._stage6_apply_contact_regularization

        def recording_apply_regularization():
            unregularized_diag["value"] = solver._diag.numpy()[0].copy()
            apply_regularization()

        solver._stage6_apply_contact_regularization = recording_apply_regularization
        state_in = model.state()
        state_out = model.state()
        newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()
        state_in.clear_forces()
        collision_pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, model.control(), contacts, 1.0 / 200.0)
        wp.synchronize()
        rows = _contact_rows(solver, PATH_DENSE, int(contacts.rigid_contact_count.numpy()[0]))
        test.assertGreater(len(rows), 0)
        base_diag = unregularized_diag["value"]
        regularized_diag = solver._diag.numpy()[0]
        weights = solver._row_w.numpy()[0]
        for slot in rows.values():
            test.assertAlmostEqual(float(weights[slot]), 2.0 / 3.0, places=6)
            test.assertAlmostEqual(
                float(regularized_diag[slot]),
                1.5 * float(base_diag[slot]),
                delta=1.0e-5 * max(float(base_diag[slot]), 1.0),
            )


def test_regularized_propagation_rows_execute(test: unittest.TestCase, device):
    """Serial and colored propagation solve real articulated rows with per-row weights."""
    with wp.ScopedDevice(device):
        for response in ("propagation", "propagation_colored"):
            model = _build_scissor_model(device)
            solver = newton.solvers.SolverFeatherPGS(
                model,
                solve_mode="matrix_free",
                articulated_contact_response=response,
                propagation_same_articulation_rows=True,
                iterations=1,
                velocity_iterations=0,
                contact_regularization=0.5,
                contact_friction_scale=0.0,
                articulation_constraint_max=64,
                free_body_constraint_max=16,
            )
            state_in, state_out = model.state(), model.state()
            newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
            collision_pipeline = newton.CollisionPipeline(model)
            contacts = collision_pipeline.contacts()
            state_in.clear_forces()
            collision_pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, model.control(), contacts, 1.0 / 200.0)
            wp.synchronize()
            rows = _contact_rows(solver, 2, int(contacts.rigid_contact_count.numpy()[0]))
            test.assertGreater(len(rows), 0, f"{response}: no propagation rows")
            weights = solver._propagation_row_w.numpy()[0]
            for slot in rows.values():
                test.assertAlmostEqual(float(weights[slot]), 2.0 / 3.0, places=6)
            test.assertTrue(np.isfinite(state_out.joint_qd.numpy()).all())


def test_compliance_parameters_removed(test: unittest.TestCase, device):
    """The no-op dense compliance knobs are gone; passing them fails loudly."""
    names = tuple(inspect.signature(newton.solvers.SolverFeatherPGS).parameters)
    test.assertNotIn("dense_contact_compliance", names)
    test.assertNotIn("speculative_dense_contact_compliance", names)
    start = names.index("baumgarte")
    test.assertEqual(names[start : start + 4], ("baumgarte", "cfm", "relaxation", "contact_regularization"))
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
        builder.add_shape_box(b, hx=0.05, hy=0.05, hz=0.05)
        model = builder.finalize()
        with test.assertRaises(TypeError):
            newton.solvers.SolverFeatherPGS(model, dense_contact_compliance=1.0e-4)


def test_restitution_rows_stay_rigid(test: unittest.TestCase, device):
    """A row whose rebound target fires is solved rigid, so the rebound is
    e * v_in whatever the regularizer. Without the exemption the regularized
    fixed point would be (e - g)/(1 + g) * v_in and vanish at g = e."""

    def rebound(g):
        builder = newton.ModelBuilder()
        builder.rigid_gap = 0.003
        cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.7, restitution=0.8)
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.7, restitution=0.8))
        b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.25), wp.quat_identity()))
        builder.add_shape_sphere(b, radius=0.05, cfg=cfg)
        model = builder.finalize()
        pipeline = newton.CollisionPipeline(
            model,
            reduce_contacts=True,
            rigid_contact_max=32,
            broad_phase="nxn",
            deterministic=True,
            contact_matching="latest",
        )
        contacts = pipeline.contacts()
        solver = newton.solvers.SolverFeatherPGS(
            model,
            solve_mode="matrix_free",
            iterations=16,
            velocity_iterations=0,
            contact_regularization=g,
            contact_warmstart=True,
        )
        s0, s1 = model.state(), model.state()
        control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
        max_up = 0.0
        for _k in range(90):
            pipeline.collide(s0, contacts)
            s0.clear_forces()
            solver.step(s0, s1, control, contacts, DT)
            s0, s1 = s1, s0
            max_up = max(max_up, float(s0.body_qd.numpy()[b][2]))
        return max_up

    with wp.ScopedDevice(device):
        v_ref = rebound(0.0)
        test.assertGreater(v_ref, 1.0, "no bounce measured")
        for g in (0.02, 0.5):
            ratio = rebound(g) / v_ref
            test.assertAlmostEqual(ratio, 1.0, delta=0.05, msg=f"g={g}: rebound ratio {ratio:.3f}")


class TestFeatherPGSRegularization(unittest.TestCase):
    pass


add_function_test(
    TestFeatherPGSRegularization,
    "test_regularization_fixed_points",
    test_regularization_fixed_points,
    devices=get_test_devices(),
)


add_function_test(
    TestFeatherPGSRegularization,
    "test_regularization_route_agreement",
    test_regularization_route_agreement,
    devices=get_selected_cuda_test_devices(),
)


add_function_test(
    TestFeatherPGSRegularization,
    "test_regularization_documented_sag",
    test_regularization_documented_sag,
    devices=get_selected_cuda_test_devices(),
)


add_function_test(
    TestFeatherPGSRegularization,
    "test_regularization_split_route_documented_sag",
    test_regularization_split_route_documented_sag,
    devices=get_test_devices(),
)


add_function_test(
    TestFeatherPGSRegularization,
    "test_regularization_velocity_pass_exempt",
    test_regularization_velocity_pass_exempt,
    devices=get_selected_cuda_test_devices(),
)


add_function_test(
    TestFeatherPGSRegularization,
    "test_velocity_pass_rigid_on_propagation_schedules",
    test_velocity_pass_rigid_on_propagation_schedules,
    devices=get_selected_cuda_test_devices(),
)


add_function_test(
    TestFeatherPGSRegularization,
    "test_regularization_indeterminate_split",
    test_regularization_indeterminate_split,
    devices=get_selected_cuda_test_devices(),
)


add_function_test(
    TestFeatherPGSRegularization,
    "test_regularization_validation",
    test_regularization_validation,
    devices=get_test_devices(),
)


add_function_test(
    TestFeatherPGSRegularization,
    "test_zero_regularization_shares_one_weight_slot",
    test_zero_regularization_shares_one_weight_slot,
    devices=get_test_devices(),
)


add_function_test(
    TestFeatherPGSRegularization,
    "test_zero_regularization_split_runtime_uses_no_weights",
    test_zero_regularization_split_runtime_uses_no_weights,
    devices=get_test_devices(),
)


add_function_test(
    TestFeatherPGSRegularization,
    "test_exact_surface_contact_is_regularized",
    test_exact_surface_contact_is_regularized,
    devices=get_test_devices(),
)


add_function_test(
    TestFeatherPGSRegularization,
    "test_assembled_dense_regularization_scales_diagonal",
    test_assembled_dense_regularization_scales_diagonal,
    devices=get_test_devices(),
)


add_function_test(
    TestFeatherPGSRegularization,
    "test_regularized_propagation_rows_execute",
    test_regularized_propagation_rows_execute,
    devices=get_selected_cuda_test_devices(),
)


add_function_test(
    TestFeatherPGSRegularization,
    "test_compliance_parameters_removed",
    test_compliance_parameters_removed,
    devices=get_test_devices(),
)


add_function_test(
    TestFeatherPGSRegularization,
    "test_restitution_rows_stay_rigid",
    test_restitution_rows_stay_rigid,
    devices=get_selected_cuda_test_devices(),
)


PATH_DENSE = 0  # contact routing id of the dense rows


def _contact_rows_regularization_upstream(solver, path_id: int, contact_count: int) -> dict[int, int]:
    """Map each contact of world 0 routed to ``path_id`` to its first row."""
    contact_path = solver._contact_path.numpy()
    contact_slot = solver._contact_slot.numpy()
    contact_world = solver._contact_world.numpy()
    return {
        c: int(contact_slot[c])
        for c in range(contact_count)
        if int(contact_world[c]) == 0 and int(contact_path[c]) == path_id and int(contact_slot[c]) >= 0
    }


def _resting_box_regularization_upstream(device, g, rate, frames, articulated=False, **solver_kwargs):
    """Rest a box on the ground; ``articulated`` mounts it on a vertical prismatic joint (dense rows)."""
    builder = newton.ModelBuilder()
    builder.rigid_gap = 0.01
    cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.7)
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.7))
    xform = wp.transform(wp.vec3(0.0, 0.0, 0.05), wp.quat_identity())
    if articulated:
        b = builder.add_link(xform=xform)
        joint = builder.add_joint_prismatic(-1, b, axis=wp.vec3(0.0, 0.0, 1.0), parent_xform=xform)
        builder.add_articulation([joint])
    else:
        b = builder.add_body(xform=xform)
    builder.add_shape_box(b, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(
        model,
        reduce_contacts=True,
        rigid_contact_max=32,
        broad_phase="nxn",
        deterministic=True,
        contact_matching="latest",
    )
    solver = newton.solvers.SolverFeatherPGS(
        model, solve_mode="matrix_free", iterations=12, contact_regularization=g, **solver_kwargs
    )
    _, zs = _run(model, pipeline, solver, frames, dt=1.0 / rate)
    return 0.05 - zs[-1]


def _scissor_step(device, g, velocity_iterations):
    """Step the same-articulation scissor scene once; return the solver, contact count and joint velocity."""
    model = _build_scissor_model(device)
    solver = newton.solvers.SolverFeatherPGS(
        model,
        solve_mode="matrix_free",
        iterations=8,
        velocity_iterations=velocity_iterations,
        contact_regularization=g,
        articulation_constraint_max=64,
        free_body_constraint_max=16,
    )
    state_in, state_out = model.state(), model.state()
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_in.clear_forces()
    pipeline.collide(state_in, contacts)
    solver.step(state_in, state_out, model.control(), contacts, 1.0 / 200.0)
    return solver, int(contacts.rigid_contact_count.numpy()[0]), state_out.joint_qd.numpy().copy()


def test_regularization_documented_sag_on_dense_rows(test: unittest.TestCase, device):
    """Apply the same sag law to dense (articulated) contact rows.

    At rest each row satisfies ``beta * phi / dt = -g * d * lambda``. A box on a vertical
    prismatic joint has ``d = 1 / m`` per contact and its four contacts share the weight,
    ``lambda = m * a * dt / 4``, so it sags by ``g * a * dt^2 / (4 * beta)``.
    """
    for rate in (60, 240):
        sag = _resting_box_regularization_upstream(
            device, 0.5, rate, 3 * rate, articulated=True, contact_warmstart=True
        )
        expected = 0.25 * _sag_formula(0.5, rate)
        test.assertAlmostEqual(sag, expected, delta=0.15 * expected, msg=f"{rate} Hz: sag {sag * 1000:.2f} mm")
        rigid = _resting_box_regularization_upstream(
            device, 0.0, rate, 3 * rate, articulated=True, contact_warmstart=True
        )
        test.assertLess(abs(rigid), 0.1 * expected, msg=f"{rate} Hz: rigid sag {rigid * 1000:.3f} mm")


def test_dense_contact_rows_carry_the_regularization_weight(test: unittest.TestCase, device):
    """Give penetrating articulated (dense) contact rows the weight ``1 / (1 + g)``."""
    solver, count, _ = _scissor_step(device, 0.5, 0)
    rows = _contact_rows_regularization_upstream(solver, PATH_DENSE, count)
    test.assertGreater(len(rows), 0, "scene produced no dense self-contact row")
    phi = solver._phi.numpy()[0]
    weights = solver._row_w.numpy()[0]
    for slot in rows.values():
        test.assertLess(float(phi[slot]), 0.0)
        test.assertAlmostEqual(float(weights[slot]), 2.0 / 3.0, places=6)


def test_velocity_pass_is_rigid_on_dense_rows(test: unittest.TestCase, device):
    """Ignore the regularizer on dense rows in the velocity-only pass.

    After the pass the joint velocity is the same as with ``g = 0``, since the rigid law does not depend on ``g``.
    """
    solver, count, qd_soft = _scissor_step(device, 0.5, 8)
    test.assertGreater(len(_contact_rows_regularization_upstream(solver, PATH_DENSE, count)), 0)
    _, _, qd_rigid = _scissor_step(device, 0.0, 8)
    _, _, qd_position_soft = _scissor_step(device, 0.5, 0)
    np.testing.assert_allclose(qd_soft, qd_rigid, rtol=0.0, atol=1.0e-4, err_msg="velocity pass depends on g")
    # Positive control: the regularized position solve alone does depend on g.
    test.assertGreater(float(np.max(np.abs(qd_position_soft - qd_rigid))), 1.0e-3)


for _fn in (
    test_regularization_documented_sag_on_dense_rows,
    test_dense_contact_rows_carry_the_regularization_weight,
    test_velocity_pass_is_rigid_on_dense_rows,
):
    add_function_test(TestFeatherPGSRegularization, _fn.__name__, _fn, devices=devices)


if __name__ == "__main__":
    unittest.main()
