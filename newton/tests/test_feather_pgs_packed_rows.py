# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import os
import unittest

import numpy as np

try:
    import warp as wp

    import newton
    from newton._src.solvers.feather_pgs.kernels import (
        PGS_CONSTRAINT_TYPE_CONTACT,
        PGS_CONSTRAINT_TYPE_FRICTION,
        PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
        PGS_CONSTRAINT_TYPE_JOINT_TARGET,
        PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
        pack_dense_constraint_row_streams,
        pack_mf_constraint_row_streams,
    )
    from newton._src.solvers.feather_pgs.solver_feather_pgs import SolverFeatherPGS, TiledKernelFactory
    from newton.tests.unittest_utils import get_cuda_test_devices
except ModuleNotFoundError as exc:
    wp = None
    NEWTON_IMPORT_ERROR = exc
else:
    NEWTON_IMPORT_ERROR = None


@unittest.skipIf(NEWTON_IMPORT_ERROR is not None, f"Newton/Warp unavailable: {NEWTON_IMPORT_ERROR}")
class TestFeatherPgsPackedRows(unittest.TestCase):
    def _cuda_device(self):
        devices = get_cuda_test_devices("basic")
        if not devices:
            self.skipTest("packed FeatherPGS fused kernel is CUDA-only")
        return devices[0]

    def _solver_kwargs(self, *, packed: bool, **overrides):
        kwargs = {
            "pgs_mode": "matrix_free",
            "packed_constraint_rows": packed,
            "pgs_iterations": 4,
            "pgs_warmstart": False,
            "dense_max_constraints": 16,
            "mf_max_constraints": 32,
            "enable_contact_friction": True,
            "contact_friction_gap_threshold": float("inf"),
            "enable_joint_limits": True,
            "enable_joint_velocity_limits": True,
            "drive_mode": "physx_pgs",
            "double_buffer": False,
        }
        kwargs.update(overrides)
        return kwargs

    def _assert_step_outputs_close(self, legacy, packed):
        for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
            np.testing.assert_allclose(packed[name], legacy[name], rtol=2.0e-5, atol=2.0e-5, err_msg=name)
        np.testing.assert_allclose(
            packed["dense_impulses"], legacy["dense_impulses"], rtol=2.0e-5, atol=2.0e-5
        )
        np.testing.assert_allclose(packed["mf_impulses"], legacy["mf_impulses"], rtol=2.0e-5, atol=2.0e-5)

    def _build_joint_row_model(self, device):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
        inertia = wp.mat33((0.2, 0.0, 0.0), (0.0, 0.2, 0.0), (0.0, 0.0, 0.2))
        body = builder.add_link(mass=1.0, inertia=inertia, label="hinge_link")
        builder.add_shape_box(body=body, hx=0.12, hy=0.08, hz=0.08)
        joint = builder.add_joint_revolute(
            parent=-1,
            child=body,
            axis=newton.Axis.Z,
            target_pos=-0.05,
            target_vel=-0.15,
            target_ke=24.0,
            target_kd=2.0,
            limit_lower=-0.20,
            limit_upper=0.20,
            velocity_limit=0.10,
            effort_limit=8.0,
            armature=0.0,
            actuator_mode=newton.JointTargetMode.POSITION_VELOCITY,
            label="packed_row_hinge",
        )
        builder.add_articulation([joint])
        return builder.finalize(device=device)

    def _run_joint_row_step(self, device, *, packed: bool):
        model = self._build_joint_row_model(device)
        solver = SolverFeatherPGS(model, **self._solver_kwargs(packed=packed))
        state_in, state_out = model.state(), model.state()
        control = model.control()

        state_in.joint_q.assign(np.array([0.26], dtype=np.float32))
        state_in.joint_qd.assign(np.array([0.35], dtype=np.float32))
        control.joint_target_pos.assign(np.array([-0.05], dtype=np.float32))
        control.joint_target_vel.assign(np.array([-0.15], dtype=np.float32))
        newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)

        solver.step(state_in, state_out, control, None, 1.0 / 120.0)
        wp.synchronize_device(device)

        dense_count = int(solver.constraint_count.numpy()[0])
        dense_types = solver.row_type.numpy()[0, :dense_count].astype(np.int32, copy=True)
        self.assertIn(PGS_CONSTRAINT_TYPE_JOINT_TARGET, dense_types.tolist())
        self.assertIn(PGS_CONSTRAINT_TYPE_JOINT_LIMIT, dense_types.tolist())
        self.assertIn(PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT, dense_types.tolist())
        self.assertEqual(bool(packed), bool(solver._last_packed_constraint_rows_selected))

        return {
            "joint_q": state_out.joint_q.numpy().copy(),
            "joint_qd": state_out.joint_qd.numpy().copy(),
            "body_q": state_out.body_q.numpy().copy(),
            "body_qd": state_out.body_qd.numpy().copy(),
            "dense_impulses": solver.impulses.numpy()[0, :dense_count].copy(),
            "mf_impulses": np.zeros((0,), dtype=np.float32),
            "dense_types": dense_types,
        }

    def test_solver_step_packed_matches_legacy_for_dense_joint_rows(self):
        device = self._cuda_device()

        legacy = self._run_joint_row_step(device, packed=False)
        packed = self._run_joint_row_step(device, packed=True)

        self._assert_step_outputs_close(legacy, packed)
        np.testing.assert_array_equal(packed["dense_types"], legacy["dense_types"])

    def _build_free_contact_model(self, device):
        cfg = newton.ModelBuilder.ShapeConfig()
        cfg.mu = 0.85
        cfg.ke = 2.0e4
        cfg.kd = 500.0
        cfg.kf = 0.0
        cfg.gap = 0.05

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=0.0)
        builder.add_ground_plane(cfg=cfg)
        body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.085), wp.quat_identity()),
            mass=1.0,
            label="free_contact_box",
        )
        builder.add_shape_box(body=body, hx=0.10, hy=0.10, hz=0.10, cfg=cfg)
        joint = builder.add_joint_free(body)
        builder.add_articulation([joint])
        return builder.finalize(device=device)

    def _run_contact_friction_step(self, device, *, packed: bool):
        model = self._build_free_contact_model(device)
        solver = SolverFeatherPGS(
            model,
            **self._solver_kwargs(
                packed=packed,
                dense_max_constraints=4,
                mf_max_constraints=32,
                enable_joint_limits=False,
                enable_joint_velocity_limits=False,
                drive_mode="augmented",
            ),
        )
        state_in, state_out = model.state(), model.state()
        control = model.control()
        state_in.joint_qd.assign(np.array([0.35, -0.20, -0.05, 0.0, 0.0, 0.0], dtype=np.float32))
        newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
        contacts = model.contacts()
        model.collide(state_in, contacts)
        self.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)

        solver.step(state_in, state_out, control, contacts, 1.0 / 120.0)
        wp.synchronize_device(device)

        mf_count = int(solver.mf_constraint_count.numpy()[0])
        mf_types = solver.mf_row_type.numpy()[0, :mf_count].astype(np.int32, copy=True)
        self.assertIn(PGS_CONSTRAINT_TYPE_CONTACT, mf_types.tolist())
        self.assertIn(PGS_CONSTRAINT_TYPE_FRICTION, mf_types.tolist())
        self.assertEqual(bool(packed), bool(solver._last_packed_constraint_rows_selected))

        return {
            "joint_q": state_out.joint_q.numpy().copy(),
            "joint_qd": state_out.joint_qd.numpy().copy(),
            "body_q": state_out.body_q.numpy().copy(),
            "body_qd": state_out.body_qd.numpy().copy(),
            "dense_impulses": np.zeros((0,), dtype=np.float32),
            "mf_impulses": solver.mf_impulses.numpy()[0, :mf_count].copy(),
            "mf_types": mf_types,
        }

    def test_solver_step_packed_matches_legacy_for_mf_contact_friction_rows(self):
        device = self._cuda_device()

        legacy = self._run_contact_friction_step(device, packed=False)
        packed = self._run_contact_friction_step(device, packed=True)

        self._assert_step_outputs_close(legacy, packed)
        np.testing.assert_array_equal(packed["mf_types"], legacy["mf_types"])

    def test_solver_option_and_env_select_packed_rows(self):
        builder = newton.ModelBuilder()
        model = builder.finalize()

        solver = SolverFeatherPGS(model, pgs_mode="matrix_free", packed_constraint_rows=True)
        self.assertTrue(solver.packed_constraint_rows)

        old = os.environ.get("FEATHER_PGS_PACKED_CONSTRAINT_ROWS")
        os.environ["FEATHER_PGS_PACKED_CONSTRAINT_ROWS"] = "true"
        try:
            solver_from_env = SolverFeatherPGS(model, pgs_mode="matrix_free", packed_constraint_rows=False)
        finally:
            if old is None:
                os.environ.pop("FEATHER_PGS_PACKED_CONSTRAINT_ROWS", None)
            else:
                os.environ["FEATHER_PGS_PACKED_CONSTRAINT_ROWS"] = old
        self.assertTrue(solver_from_env.packed_constraint_rows)

    def test_packed_kernel_matches_legacy_and_uses_dense_selectors(self):
        device = self._cuda_device()

        worlds = 1
        dense_rows = 6
        mf_rows = 3
        dofs = 8

        constraint_count = wp.array(np.array([dense_rows], dtype=np.int32), dtype=wp.int32, device=device)
        mf_constraint_count = wp.array(np.array([mf_rows], dtype=np.int32), dtype=wp.int32, device=device)
        world_dof_start = wp.array(np.array([0], dtype=np.int32), dtype=wp.int32, device=device)

        row_type_np = np.array(
            [
                [
                    PGS_CONSTRAINT_TYPE_CONTACT,
                    PGS_CONSTRAINT_TYPE_FRICTION,
                    PGS_CONSTRAINT_TYPE_FRICTION,
                    PGS_CONSTRAINT_TYPE_JOINT_TARGET,
                    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
                    PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
                ]
            ],
            dtype=np.int32,
        )
        row_parent_np = np.array([[-1, 0, 0, -1, -1, -1]], dtype=np.int32)
        row_mu_np = np.array([[0.7, 0.7, 0.7, 0.0, 0.0, 0.0]], dtype=np.float32)
        row_dof_np = np.array([[-1, -1, -1, 3, 4, 5]], dtype=np.int32)
        row_sign_np = np.array([[0.0, 0.0, 0.0, 1.0, -1.0, 1.0]], dtype=np.float32)
        dense_diag_np = np.array([[0.5, 0.4, 0.45, 0.3, 0.4, 0.2]], dtype=np.float32)
        dense_rhs_np = np.array([[-0.35, 0.0, 0.0, 0.0, -0.15, 0.45]], dtype=np.float32)

        j_legacy = np.zeros((worlds, dense_rows, dofs), dtype=np.float32)
        y_world = np.zeros((worlds, dense_rows, dofs), dtype=np.float32)
        j_legacy[0, 0, 0] = 1.0
        j_legacy[0, 1, 1] = 1.0
        j_legacy[0, 2, 2] = 1.0
        j_legacy[0, 3, 3] = 1.0
        j_legacy[0, 4, 4] = -1.0
        j_legacy[0, 5, 5] = 1.0
        y_world[0, 0, 0] = 0.5
        y_world[0, 1, 1] = 0.4
        y_world[0, 2, 2] = 0.45
        y_world[0, 3, 3] = 0.3
        y_world[0, 4, 4] = -0.4
        y_world[0, 5, 5] = 0.2

        # Poison selector-row dense J in the packed input. Packed mode should
        # match legacy anyway because drive/limit/velocity-limit J*v comes
        # from row_dof/row_sign metadata.
        j_packed = j_legacy.copy()
        j_packed[0, 3:, :] = 0.0

        drive_target_np = np.zeros((worlds, dense_rows), dtype=np.float32)
        drive_vel_mul_np = np.zeros((worlds, dense_rows), dtype=np.float32)
        drive_imp_mul_np = np.ones((worlds, dense_rows), dtype=np.float32)
        drive_max_imp_np = np.full((worlds, dense_rows), 1.0e6, dtype=np.float32)
        drive_target_np[0, 3] = 0.08
        drive_vel_mul_np[0, 3] = -0.2
        drive_imp_mul_np[0, 3] = 0.85
        drive_max_imp_np[0, 3] = 2.0

        mf_row_type_np = np.array([[PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION, PGS_CONSTRAINT_TYPE_FRICTION]], dtype=np.int32)
        mf_row_parent_np = np.array([[-1, 0, 0]], dtype=np.int32)
        mf_row_mu_np = np.array([[0.6, 0.6, 0.6]], dtype=np.float32)
        mf_dof_a_np = np.array([[0, 0, 0]], dtype=np.int32)
        mf_dof_b_np = np.array([[-1, -1, -1]], dtype=np.int32)
        mf_eff_inv_np = np.array([[0.5, 0.4, 0.45]], dtype=np.float32)
        mf_rhs_np = np.array([[-0.1, 0.0, 0.0]], dtype=np.float32)
        mf_phi_np = np.array([[-0.01, 0.0, 0.0]], dtype=np.float32)
        mf_J_a_np = np.zeros((worlds, mf_rows, 6), dtype=np.float32)
        mf_J_b_np = np.zeros_like(mf_J_a_np)
        mf_MiJt_a_np = np.zeros_like(mf_J_a_np)
        mf_MiJt_b_np = np.zeros_like(mf_J_a_np)
        mf_J_a_np[0, 0, 0] = 0.5
        mf_J_a_np[0, 1, 1] = 0.4
        mf_J_a_np[0, 2, 2] = 0.3
        mf_MiJt_a_np[0, 0, 0] = 0.25
        mf_MiJt_a_np[0, 1, 1] = 0.2
        mf_MiJt_a_np[0, 2, 2] = 0.15

        v0_np = np.array([-0.4, 0.2, -0.3, 0.5, 0.8, -0.7, 0.0, 0.0], dtype=np.float32)
        dense_imp0_np = np.array([[0.2, 0.05, -0.03, 0.1, 0.0, 0.0]], dtype=np.float32)
        mf_imp0_np = np.array([[0.1, 0.02, -0.01]], dtype=np.float32)

        def arr(value, dtype):
            return wp.array(value, dtype=dtype, device=device)

        row_type = arr(row_type_np, wp.int32)
        row_parent = arr(row_parent_np, wp.int32)
        row_mu = arr(row_mu_np, wp.float32)
        row_dof = arr(row_dof_np, wp.int32)
        row_sign = arr(row_sign_np, wp.float32)
        dense_diag = arr(dense_diag_np, wp.float32)
        dense_rhs = arr(dense_rhs_np, wp.float32)
        y = arr(y_world, wp.float32)
        drive_target = arr(drive_target_np, wp.float32)
        drive_vel_mul = arr(drive_vel_mul_np, wp.float32)
        drive_imp_mul = arr(drive_imp_mul_np, wp.float32)
        drive_max_imp = arr(drive_max_imp_np, wp.float32)

        mf_row_type = arr(mf_row_type_np, wp.int32)
        mf_row_parent = arr(mf_row_parent_np, wp.int32)
        mf_row_mu = arr(mf_row_mu_np, wp.float32)
        mf_dof_a = arr(mf_dof_a_np, wp.int32)
        mf_dof_b = arr(mf_dof_b_np, wp.int32)
        mf_eff_inv = arr(mf_eff_inv_np, wp.float32)
        mf_rhs = arr(mf_rhs_np, wp.float32)
        mf_phi = arr(mf_phi_np, wp.float32)
        mf_J_a = arr(mf_J_a_np, wp.float32)
        mf_J_b = arr(mf_J_b_np, wp.float32)
        mf_MiJt_a = arr(mf_MiJt_a_np, wp.float32)
        mf_MiJt_b = arr(mf_MiJt_b_np, wp.float32)

        legacy_mf_meta = wp.zeros((worlds, mf_rows * 4), dtype=wp.int32, device=device)
        legacy_pack = TiledKernelFactory.get_pack_mf_meta_kernel(mf_rows, device)
        wp.launch_tiled(
            legacy_pack,
            dim=[worlds],
            inputs=[mf_constraint_count, mf_dof_a, mf_dof_b, mf_eff_inv, mf_rhs, mf_row_type, mf_row_parent],
            outputs=[legacy_mf_meta],
            block_dim=32,
            device=device,
        )

        packed_dense_meta = wp.zeros((worlds, dense_rows * 4), dtype=wp.int32, device=device)
        packed_dense_scalar = wp.zeros((worlds, dense_rows * 8), dtype=wp.float32, device=device)
        packed_mf_meta = wp.zeros((worlds, mf_rows * 4), dtype=wp.int32, device=device)
        packed_mf_scalar = wp.zeros((worlds, mf_rows * 4), dtype=wp.float32, device=device)
        wp.launch(
            pack_dense_constraint_row_streams,
            dim=worlds * dense_rows,
            inputs=[
                constraint_count,
                row_type,
                row_parent,
                row_mu,
                row_dof,
                row_sign,
                dense_diag,
                dense_rhs,
                drive_target,
                drive_vel_mul,
                drive_imp_mul,
                drive_max_imp,
                dense_rows,
            ],
            outputs=[packed_dense_meta, packed_dense_scalar],
            device=device,
        )
        wp.launch(
            pack_mf_constraint_row_streams,
            dim=worlds * mf_rows,
            inputs=[
                mf_constraint_count,
                mf_dof_a,
                mf_dof_b,
                mf_eff_inv,
                mf_rhs,
                mf_row_type,
                mf_row_parent,
                mf_row_mu,
                mf_phi,
                mf_rows,
            ],
            outputs=[packed_mf_meta, packed_mf_scalar],
            device=device,
        )

        legacy_impulses = arr(dense_imp0_np.copy(), wp.float32)
        legacy_mf_impulses = arr(mf_imp0_np.copy(), wp.float32)
        legacy_v = arr(v0_np.copy(), wp.float32)
        legacy_kernel = TiledKernelFactory.get_pgs_solve_mf_gs_kernel(
            dense_rows,
            mf_rows,
            dofs,
            device,
            friction_mode="current",
        )
        wp.launch_tiled(
            legacy_kernel,
            dim=[worlds],
            inputs=[
                constraint_count,
                world_dof_start,
                dense_rhs,
                dense_diag,
                legacy_impulses,
                arr(j_legacy, wp.float32),
                y,
                row_type,
                row_parent,
                row_mu,
                drive_target,
                drive_vel_mul,
                drive_imp_mul,
                drive_max_imp,
                mf_constraint_count,
                legacy_mf_meta,
                legacy_mf_impulses,
                mf_J_a,
                mf_J_b,
                mf_MiJt_a,
                mf_MiJt_b,
                mf_row_mu,
                3,
                1.0,
                0,
                0,
                0,
                0,
            ],
            outputs=[legacy_v],
            block_dim=32,
            device=device,
        )

        packed_impulses = arr(dense_imp0_np.copy(), wp.float32)
        packed_mf_impulses = arr(mf_imp0_np.copy(), wp.float32)
        packed_v = arr(v0_np.copy(), wp.float32)
        packed_kernel = TiledKernelFactory.get_pgs_solve_packed_row_stream_kernel(dense_rows, mf_rows, dofs, device)
        wp.launch_tiled(
            packed_kernel,
            dim=[worlds],
            inputs=[
                constraint_count,
                world_dof_start,
                packed_impulses,
                arr(j_packed, wp.float32),
                y,
                packed_dense_meta,
                packed_dense_scalar,
                mf_constraint_count,
                packed_mf_meta,
                packed_mf_scalar,
                packed_mf_impulses,
                mf_J_a,
                mf_J_b,
                mf_MiJt_a,
                mf_MiJt_b,
                3,
                1.0,
                0,
                0,
                0,
                0,
            ],
            outputs=[packed_v],
            block_dim=32,
            device=device,
        )

        wp.synchronize_device(device)

        np.testing.assert_allclose(packed_v.numpy(), legacy_v.numpy(), rtol=1.0e-5, atol=1.0e-5)
        np.testing.assert_allclose(packed_impulses.numpy(), legacy_impulses.numpy(), rtol=1.0e-5, atol=1.0e-5)
        np.testing.assert_allclose(
            packed_mf_impulses.numpy(), legacy_mf_impulses.numpy(), rtol=1.0e-5, atol=1.0e-5
        )


if __name__ == "__main__":
    unittest.main()
