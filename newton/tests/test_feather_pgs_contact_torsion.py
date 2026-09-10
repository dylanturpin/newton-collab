# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Validate opt-in, load-bounded spin resistance on articulated contacts."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS


def fixture(
    radius=None,
    *,
    mu=0.5,
    closing=0.1,
    spin=1.0,
    sliding=0.0,
    sat=False,
    separation=0.0,
    center_only=False,
    center_count=1,
    dt=0.0025,
    row_limit=None,
    **solver_overrides,
):
    """Solve two equal articulated rectangular pads touching face to face."""
    b = newton.ModelBuilder(gravity=(0, 0, 0))
    for side in (-1, 1):
        pose = wp.transform(wp.vec3(0, 0, side * (0.025 + separation / 2)), wp.quat_identity())
        body = b.add_link(
            xform=pose, mass=0.3, inertia=wp.mat33(np.diag([0.00012, 0.00012, 0.00008]).astype(np.float32))
        )
        axes = [newton.ModelBuilder.JointDofConfig(axis=wp.vec3(*a)) for a in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
        joint = b.add_joint_d6(-1, body, parent_xform=pose, linear_axes=axes, angular_axes=axes)
        b.add_articulation([joint])
        b.add_shape_box(
            body, hx=0.02, hy=0.015, hz=0.025, cfg=newton.ModelBuilder.ShapeConfig(density=0, mu=mu, restitution=0)
        )
    model = b.finalize(device="cuda:0")
    a, z = model.state(), model.state()
    a.joint_qd.assign(np.array([sliding, 0, closing, 0, 0, spin, -sliding, 0, -closing, 0, 0, -spin], np.float32))
    newton.eval_fk(model, a.joint_q, a.joint_qd, a)
    pipeline = newton.CollisionPipeline(
        model,
        contact_matching="latest",
        reduce_contacts=False,
        broad_phase="nxn",
        rigid_contact_max=64,
        box_box_sat=sat,
    )
    contacts = pipeline.contacts()
    pipeline.collide(a, contacts)
    if center_only and int(contacts.rigid_contact_count.numpy()[0]) > 0:
        # Deliberately represent the known planar disk by one central witness:
        # this isolates the new material spin law from native point lever arms.
        shapes = [contacts.rigid_contact_shape0.numpy()[0], contacts.rigid_contact_shape1.numpy()[0]]
        poses = a.body_q.numpy()
        for side, name in enumerate(("rigid_contact_point0", "rigid_contact_point1")):
            values = getattr(contacts, name).numpy()
            body = model.shape_body.numpy()[shapes[side]]
            values[:center_count] = -poses[body, :3]
            getattr(contacts, name).assign(values)
        contacts.rigid_contact_count.assign(np.array([center_count], np.int32))
    kwargs = {
        "pgs_mode": "matrix_free",
        "articulated_contact_response": "immediate",
        "pgs_iterations": 128,
        "pgs_velocity_iterations": 0,
        "pgs_beta": 0.05,
        "pgs_cfm": 0,
        "pgs_contact_regularization": 0,
        "contact_friction_position_iterations": -1,
        "contact_shared_anchor": True,
        "contact_friction_shared_anchor": True,
        "enable_joint_velocity_limits": True,
        "pgs_warmstart": False,
        "angular_damping": 0,
        "dense_max_constraints": 64,
        "mf_max_constraints": 16,
        "row_watermark": True,
    }
    if radius is not None:
        kwargs["contact_torsion_radius"] = radius
    kwargs.update(solver_overrides)
    solver = SolverFeatherPGS(model, **kwargs)
    if row_limit is not None:
        solver.dense_max_constraints = row_limit
    solver.step(a, z, model.control(), contacts, dt)
    result = {name: getattr(z, name).numpy() for name in ("body_q", "body_qd", "joint_q", "joint_qd")}
    result.update(
        {
            name: getattr(solver, name).numpy()
            for name in ("impulses", "row_type", "row_parent", "row_mu", "J_world", "Y_world", "v_hat", "v_out")
        }
    )
    result["count"] = solver.constraint_count.numpy()
    return result, solver, model, a, contacts


@unittest.skipUnless(wp.get_device().is_cuda, "Contact torsion currently requires CUDA")
class TestContactTorsion(unittest.TestCase):
    """Exercise an explicitly assumed uniform-disk effective spin radius."""

    def test_default_off_equivalent(self):
        """Keep the zero-radius option exactly equivalent to prefeature output."""
        reference, *_ = fixture(0.0)
        actual, *_ = fixture(0.01, contact_torsion_shape_indices=())
        for key in reference:
            np.testing.assert_array_equal(actual[key], reference[key], err_msg=key)

    def test_spin_stops_below_bound(self):
        """Remove spin below the prescribed disk's Coulomb torque capacity."""
        baseline, *_ = fixture(0.0, spin=1.0, center_only=True)
        self.assertGreater(np.max(np.abs(baseline["body_qd"][:, 5])), 0.9)
        result, solver, *_ = fixture(0.01, spin=1.0, center_only=True)
        self.assertLess(np.max(np.abs(result["body_qd"][:, 5])), 1e-4)
        self.assertGreater(solver._torsion_stats["rows"], 0)

    def test_zero_friction_and_zero_load(self):
        """Apply no torsion when friction or compressive normal load vanishes."""
        for options in ({"mu": 0.0}, {"closing": 0.0}, {"separation": 0.004}):
            actual, solver, *_ = fixture(0.01, **options)
            if "closing" in options:
                self.assertGreater(solver._torsion_stats["rows"], 0)
            else:
                self.assertEqual(solver._torsion_stats["rows"], 0)
            active = actual["row_type"] == 7
            self.assertLess(np.abs(actual["impulses"][active]).max(initial=0), 1e-9)

    def test_joint_response_and_coupled_budget(self):
        """Respect Coulomb sharing, articulated response and kinetic-energy bounds."""
        for sat in (False, True):
            for spin, sliding in ((1.0, 0.0), (100.0, 0.0), (10.0, 1.0), (100.0, 10.0)):
                actual, solver, _model, initial, _ = fixture(
                    0.01, spin=spin, sliding=sliding, sat=sat, center_only=not sat
                )
                # Evaluate contact response at its input pose, before D6
                # integration changes the motion-basis readback. The existing
                # predictor already changes lateral momentum at extreme mixed
                # velocity; this gate isolates the contact solve from that predictor.
                v0 = actual["v_hat"].reshape(2, 6)
                v1 = actual["v_out"].reshape(2, 6)
                inertia = np.array([0.00012, 0.00012, 0.00008])

                def energy(v, inertia=inertia):
                    return float(0.5 * 0.3 * np.sum(v[:, :3] ** 2) + 0.5 * np.sum(v[:, 3:] ** 2 * inertia))

                self.assertLessEqual(energy(v1), energy(v0) + 2e-6)
                np.testing.assert_allclose(np.sum(v1[:, :3], axis=0), np.sum(v0[:, :3], axis=0), atol=1e-5)
                positions = initial.body_q.numpy()[:, :3]

                def angular(v, inertia=inertia, positions=positions):
                    return np.sum(v[:, 3:] * inertia + np.cross(positions, 0.3 * v[:, :3]), axis=0)

                np.testing.assert_allclose(angular(v1), angular(v0), atol=2e-6)
                count = int(actual["count"][0])
                impulse = actual["impulses"][0, :count]
                response = actual["Y_world"][0, :count].T @ impulse
                np.testing.assert_allclose(actual["v_out"] - actual["v_hat"], response, atol=2e-5)
                for group in solver._torsion_stats["groups"]:
                    rows = group["normal_rows"]
                    budget = 0.5 * sum(max(float(impulse[r]), 0) for r in rows)
                    used = sum(float(np.linalg.norm(impulse[r + 1 : r + 3])) for r in rows)
                    used += abs(float(impulse[group["row"]])) / 0.01
                    self.assertLessEqual(used, budget + 2e-6)

    def test_witness_count_does_not_multiply_torque(self):
        """Share one footprint budget across repeated normal quadrature witnesses."""
        outputs = []
        for count in (1, 2, 4):
            result, solver, *_ = fixture(0.01, spin=100.0, center_only=True, center_count=count)
            self.assertEqual(solver._torsion_stats["rows"], 1)
            outputs.append(result["body_qd"])
        for output in outputs[1:]:
            np.testing.assert_allclose(output, outputs[0], atol=2e-5)

    def test_release_and_reset_carry_no_torque(self):
        """Forget spin impulse on contact loss and reproduce cold state after reset."""
        expected, solver, model, initial, contacts = fixture(0.01, spin=100.0)
        output = model.state()
        count = contacts.rigid_contact_count.numpy()
        contacts.rigid_contact_count.zero_()
        solver.step(initial, output, model.control(), contacts, 0.0025)
        self.assertEqual(solver._torsion_stats["rows"], 0)
        contacts.rigid_contact_count.assign(count)
        solver.reset(initial)
        solver.step(initial, output, model.control(), contacts, 0.0025)
        np.testing.assert_allclose(output.body_qd.numpy(), expected["body_qd"], atol=2e-5)

    def test_timestep_and_capacity(self):
        """Keep impulse-level overload behavior across dt and reject insufficient rows."""
        velocities = []
        for dt in (0.00125, 0.0025, 0.005):
            result, solver, *_ = fixture(0.01, spin=100.0, center_only=True, dt=dt)
            velocities.append(result["v_out"])
            self.assertEqual(int(solver._row_dropped_dense_high_water.numpy()[0]), 0)
            self.assertLessEqual(int(solver.constraint_count.numpy().max()), solver.dense_max_constraints)
        for velocity in velocities[1:]:
            np.testing.assert_allclose(velocity, velocities[0], atol=2e-5)
        baseline, *_ = fixture(0.0, center_only=True)
        with self.assertRaisesRegex(RuntimeError, "capacity exceeded"):
            fixture(0.01, center_only=True, row_limit=int(baseline["count"][0]))

    def test_public_shape_selection(self):
        """Resolve public index and regex scopes without private model patches."""
        baseline, *_ = fixture(0.0, center_only=True)
        excluded, *_ = fixture(0.01, center_only=True, contact_torsion_shape_indices=())
        np.testing.assert_array_equal(baseline["body_qd"], excluded["body_qd"])
        for selection in ({"contact_torsion_shape_indices": (0,)}, {"contact_torsion_shape_patterns": (".*",)}):
            result, solver, *_ = fixture(0.01, center_only=True, **selection)
            self.assertLess(np.max(np.abs(result["body_qd"][:, 5])), 1e-4)
            self.assertEqual(solver._torsion_stats["rows"], 1)

    def test_invalid_input_and_unsupported_modes(self):
        """Reject invalid scopes and modes rather than silently ignoring spin friction."""
        for radius in (-1.0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                fixture(radius)
        for options in (
            {"contact_torsion_shape_indices": (-1,)},
            {"contact_torsion_shape_indices": (True,)},
            {"contact_torsion_shape_patterns": ("[",)},
            {"contact_torsion_shape_patterns": ("missing-label",)},
            {"contact_torsion_shape_indices": (), "contact_torsion_shape_patterns": ()},
            {"pgs_warmstart": True},
            {"pgs_velocity_iterations": 1},
            {"pgs_contact_regularization": 0.01},
            {"articulated_contact_response": "propagation"},
            {"friction_mode": "bisection"},
            {"pgs_debug": True},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                fixture(0.01, **options)
        with patch.dict("os.environ", {"IL_NEWTON_FPGS_MF_WARMSTART": "1"}):
            with self.assertRaises(ValueError):
                fixture(0.01)

    def test_capture_is_explicitly_rejected(self):
        """Reject capture before host contact grouping is attempted."""
        _, solver, model, initial, contacts = fixture(0.01)
        output = model.state()
        with self.assertRaisesRegex(RuntimeError, "graph capture"):
            with wp.ScopedCapture():
                solver.step(initial, output, model.control(), contacts, 0.0025)

    def test_hydro_combination_is_rejected(self):
        """Reject actual hydro contact stiffness before combining contact mechanisms."""
        _, solver, model, initial, contacts = fixture(0.01)
        contacts.rigid_contact_stiffness = wp.ones(contacts.rigid_contact_max, device=model.device)
        with self.assertRaisesRegex(ValueError, "hydroelastic"):
            solver.step(initial, model.state(), model.control(), contacts, 0.0025)


if __name__ == "__main__":
    unittest.main()
