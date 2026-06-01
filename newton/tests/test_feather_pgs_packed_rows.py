# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton


def _sync_np(array):
    wp.synchronize()
    return array.numpy().copy()


class TestFeatherPGSPackedRowsConfig(unittest.TestCase):
    def test_packed_rows_requires_matrix_free_mode(self):
        builder = newton.ModelBuilder(gravity=0.0)
        body = builder.add_link()
        joint = builder.add_joint_prismatic(parent=-1, child=body, axis=newton.Axis.X)
        builder.add_articulation([joint])
        model = builder.finalize()

        with self.assertRaisesRegex(NotImplementedError, "requires pgs_mode='matrix_free'"):
            newton.solvers.SolverFeatherPGS(model, pgs_mode="split", packed_constraint_rows=True)

    def test_packed_rows_rejects_non_current_friction_mode(self):
        builder = newton.ModelBuilder(gravity=0.0)
        body = builder.add_link()
        joint = builder.add_joint_prismatic(parent=-1, child=body, axis=newton.Axis.X)
        builder.add_articulation([joint])
        model = builder.finalize()

        with self.assertRaisesRegex(NotImplementedError, "friction_mode='current' only"):
            newton.solvers.SolverFeatherPGS(
                model,
                pgs_mode="matrix_free",
                packed_constraint_rows=True,
                friction_mode="bisection",
            )


@unittest.skipUnless(wp.is_cuda_available(), "packed FeatherPGS row stream uses CUDA native kernels")
class TestFeatherPGSPackedRowsCUDA(unittest.TestCase):
    def setUp(self):
        self.device = wp.get_device("cuda:0")

    def _free_contact_case(self, packed: bool):
        builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Z)
        builder.default_shape_cfg.ke = 1.0e4
        builder.default_shape_cfg.kd = 1.0e3
        builder.default_shape_cfg.kf = 0.6
        builder.default_shape_cfg.mu = 0.8
        builder.default_shape_cfg.gap = 0.02
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.08), wp.quat_identity()))
        builder.add_shape_sphere(body=body, radius=0.1)
        builder.add_ground_plane()
        model = builder.finalize(device=self.device)

        state_in = model.state()
        state_out = model.state()
        control = model.control()
        contacts = model.contacts()
        model.collide(state_in, contacts)

        solver = newton.solvers.SolverFeatherPGS(
            model,
            pgs_mode="matrix_free",
            dense_max_constraints=8,
            mf_max_constraints=32,
            pgs_iterations=4,
            enable_contact_friction=True,
            packed_constraint_rows=packed,
        )
        solver.step(state_in, state_out, control, contacts, 1.0 / 120.0)
        wp.synchronize()
        return solver, _sync_np(state_out.body_qd), _sync_np(state_out.body_q)

    def _articulated_one_dof_case(self, packed: bool):
        builder = newton.ModelBuilder(gravity=0.0, up_axis=newton.Axis.Z)
        body = builder.add_link(mass=2.0, armature=0.0)
        joint = builder.add_joint_prismatic(
            parent=-1,
            child=body,
            axis=newton.Axis.X,
            limit_lower=-0.05,
            limit_upper=0.05,
            target_pos=0.0,
            target_vel=0.0,
            target_ke=100.0,
            target_kd=10.0,
            effort_limit=1000.0,
            velocity_limit=0.1,
            actuator_mode=newton.JointTargetMode.POSITION_VELOCITY,
        )
        builder.add_articulation([joint])
        builder.joint_q[0] = 0.12
        builder.joint_qd[0] = 0.45
        model = builder.finalize(device=self.device)

        state_in = model.state()
        state_out = model.state()
        control = model.control()
        contacts = model.contacts()
        state_in.joint_q.assign([0.12])
        state_in.joint_qd.assign([0.45])
        control.joint_target_pos.assign([0.0])
        control.joint_target_vel.assign([0.0])
        newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)

        solver = newton.solvers.SolverFeatherPGS(
            model,
            pgs_mode="matrix_free",
            drive_mode="physx_pgs",
            dense_max_constraints=16,
            mf_max_constraints=8,
            pgs_iterations=4,
            pgs_velocity_iterations=1,
            enable_joint_limits=True,
            enable_joint_velocity_limits=True,
            packed_constraint_rows=packed,
        )
        solver.step(state_in, state_out, control, contacts, 1.0 / 120.0)
        wp.synchronize()
        return solver, _sync_np(state_out.joint_qd), _sync_np(state_out.joint_q)

    def test_free_contact_friction_matches_legacy_and_uses_mf_packed_stream(self):
        legacy_solver, legacy_qd, legacy_q = self._free_contact_case(False)
        packed_solver, packed_qd, packed_q = self._free_contact_case(True)

        self.assertFalse(legacy_solver._last_matrix_free_packed_launch)
        self.assertTrue(packed_solver._last_matrix_free_packed_launch)
        np.testing.assert_allclose(packed_qd, legacy_qd, atol=1.0e-5, rtol=1.0e-5)
        np.testing.assert_allclose(packed_q, legacy_q, atol=1.0e-5, rtol=1.0e-5)

        count = int(packed_solver.mf_constraint_count.numpy()[0])
        meta = packed_solver.mf_meta_packed.numpy()[0, : count * 4].reshape(count, 4)
        scalars = packed_solver.mf_scalar_packed.numpy()[0, : count * 4].reshape(count, 4)
        row_types = meta[:, 3] & 0xFFFF
        self.assertIn(0, row_types)
        self.assertIn(2, row_types)
        self.assertGreater(float(np.max(scalars[:, 0])), 0.0)

    def test_one_dof_joint_rows_match_legacy_and_pack_selectors(self):
        legacy_solver, legacy_qd, legacy_q = self._articulated_one_dof_case(False)
        packed_solver, packed_qd, packed_q = self._articulated_one_dof_case(True)

        self.assertFalse(legacy_solver._last_matrix_free_packed_launch)
        self.assertTrue(packed_solver._last_matrix_free_packed_launch)
        np.testing.assert_allclose(packed_qd, legacy_qd, atol=1.0e-5, rtol=1.0e-5)
        np.testing.assert_allclose(packed_q, legacy_q, atol=1.0e-5, rtol=1.0e-5)

        count = int(packed_solver.constraint_count.numpy()[0])
        meta = packed_solver.dense_meta_packed.numpy()[0, : count * 4].reshape(count, 4)
        row_types = meta[:, 0] & 0xFFFF
        selectors = meta[:, 1]
        for row_type in (1, 3, 4):
            matches = np.where(row_types == row_type)[0]
            self.assertGreater(matches.size, 0, f"missing packed row type {row_type}")
            self.assertTrue(np.all(selectors[matches] >= 0), f"missing selector for row type {row_type}")


if __name__ == "__main__":
    unittest.main(verbosity=2)

