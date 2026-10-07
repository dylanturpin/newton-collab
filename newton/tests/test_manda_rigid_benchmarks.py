# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate reconstructed Manda rigid fixtures and independent physical controls."""

import unittest

import numpy as np

from scripts.benchmarks.manda_rigid import FPGSRunner, MuJoCoRunner, build_scene, hinge_reference, make_tape


class TestMandaRigidBenchmarks(unittest.TestCase):
    def test_imported_fixture_properties(self):
        """Preserve authored mass, COM, full inertia, joint state and collision masks."""
        for scene in ("slide", "drop", "hinge", "collision", "panda_effort", "grasp", "stack", "push"):
            with self.subTest(scene=scene):
                fixture = build_scene(scene)
                runner = FPGSRunner(fixture, device="cpu", iterations=8)
                self.assertLess(runner.audit["mass_max_error_kg"], 1.0e-6)
                self.assertLess(runner.audit["inertia_max_error_kg_m2"], 1.0e-6)
                self.assertLess(runner.audit["fk_max_error_m"], 1.0e-5)
                self.assertFalse(np.any(runner.model.joint_target_ke.numpy()))
                self.assertFalse(np.any(runner.model.joint_target_kd.numpy()))

    def test_drop_before_contact(self):
        """Follow semi-implicit free fall before the cube reaches the ground."""
        fixture = build_scene("drop")
        runner = FPGSRunner(fixture, device="cpu", iterations=8)
        for _ in range(20):
            runner.step(np.empty(0), 0.001)
        position = runner.observe()[0][0]
        self.assertAlmostEqual(position[2], 0.35 - 9.81 * 0.001**2 * 20 * 21 / 2, places=6)

    def test_slide_friction_changes_stopping_distance(self):
        """Distinguish frictional stopping from a frictionless translating cube."""
        endpoints = []
        for friction in (0.0, 0.4):
            fixture = build_scene("slide", friction=friction)
            runner = FPGSRunner(fixture, device="cpu", iterations=16)
            for _ in range(160):
                runner.step(np.empty(0), 0.002)
            endpoints.append(runner.observe()[0][0, 0])
        self.assertGreater(endpoints[0], 0.30)
        self.assertLess(abs(endpoints[1] - 1.0 / (2 * 0.4 * 9.81)), 0.015)

    def test_hinge_timestep_refinement(self):
        """Reduce hinge angle error against an independent RK4 reference at half timestep."""
        errors = []
        for dt in (0.001, 0.0005):
            fixture = build_scene("hinge")
            runner = FPGSRunner(fixture, device="cpu", iterations=8)
            for step in range(round(0.3 / dt)):
                torque = 0.2 if 0.1 <= step * dt < 0.2 else 0.0
                runner.step(np.array([torque]), dt)
            errors.append(abs(runner.joints()[0][0] - hinge_reference(0.3)[0]))
        self.assertLess(errors[0], 0.003)
        self.assertLess(errors[1], 0.6 * errors[0])

    def test_reference_is_frozen_and_fingers_independent(self):
        """Generate identical unloaded tapes with bounded independent finger commands."""
        fixture = build_scene("grasp")
        first, second = make_tape(fixture), make_tape(fixture)
        np.testing.assert_array_equal(first.q, second.q)
        np.testing.assert_array_equal(first.feedforward, second.feedforward)
        self.assertEqual(first.q.shape[1], 9)
        self.assertEqual(first.sha256, second.sha256)
        self.assertAlmostEqual(first.q[800, -1], 0.012)
        self.assertEqual(fixture.native_model.neq, 0)
        self.assertEqual(fixture.native_model.ntendon, 0)

    def test_contact_force_sign_and_step_alignment(self):
        """Balance one slider step's native contact impulse and COM velocity change."""
        fixture = build_scene("slide")
        for runner in (FPGSRunner(fixture, device="cpu", iterations=32), MuJoCoRunner(fixture)):
            with self.subTest(solver=type(runner).__name__):
                before = runner.observe()[2][0].copy()
                runner.step(np.empty(0), 0.002)
                after = runner.observe()[2][0]
                impulse = (runner.forces()[0] + [0, 0, -0.981]) * 0.002
                np.testing.assert_allclose(0.1 * (after - before), impulse, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
