# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Verify physical invariants needed by persistent FeatherPGS contacts."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS


class TestFeatherPGSBoundedDrives(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "requires CUDA")
    def test_finite_drive_bounds_cuda_tiled_split_path(self):
        """The tiled row solver must handle bounded drives without serial fallback."""
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        joint = builder.add_joint_prismatic(
            -1,
            body,
            axis=newton.Axis.X,
            target_ke=10000.0,
            target_kd=0.0,
            damping=0.0,
            armature=0.0,
            effort_limit=1.0,
        )
        builder.add_articulation([joint])
        model = builder.finalize(device=wp.get_cuda_device())
        solver = SolverFeatherPGS(model, pgs_mode="split", pgs_kernel="tiled_row", dense_max_constraints=32)
        state_in, state_out = model.state(), model.state()
        control = model.control()
        newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
        state_in.body_f.assign(np.array([[10000.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32))
        with patch.object(
            solver, "_stage5_pgs_solve_world_tiled_row", wraps=solver._stage5_pgs_solve_world_tiled_row
        ) as tiled:
            solver.step(state_in, state_out, control, None, 0.01)
            tiled.assert_called_once()
        reaction = float(state_out.joint_qd.numpy()[0]) / 0.01 - 10000.0
        self.assertAlmostEqual(reaction, -1.0, delta=0.05)
        with wp.ScopedCapture(device=model.device) as capture:
            solver.seed_double_buffer_events()
            solver.step(state_in, state_out, control, None, 0.01)
            solver.step(state_in, state_out, control, None, 0.01)
        for bound in (2.0, 1.0):
            model.joint_effort_limit.fill_(bound)
            wp.capture_launch(capture.graph)
            reaction = float(state_out.joint_qd.numpy()[0]) / 0.01 - 10000.0
            self.assertAlmostEqual(reaction, -bound, delta=0.05)

    @unittest.skipUnless(wp.is_cuda_available(), "requires CUDA")
    def test_finite_drive_bounds_local_articulation_fast_path(self):
        """The contact-free fused path must preserve the same actuator bound."""
        builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
        body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        joint = builder.add_joint_prismatic(
            -1,
            body,
            axis=newton.Axis.X,
            target_ke=10000.0,
            target_kd=0.0,
            damping=0.0,
            armature=0.0,
            effort_limit=1.0,
        )
        builder.add_articulation([joint])
        builder.add_body(mass=1.0, inertia=wp.mat33(np.eye(3)))
        model = builder.finalize(device=wp.get_cuda_device())
        solver = SolverFeatherPGS(model, pgs_mode="matrix_free", angular_damping=0.0)
        self.assertTrue(solver._local_internal_fast_path)
        state_in, state_out = model.state(), model.state()
        newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
        forces = np.zeros((2, 6), dtype=np.float32)
        forces[body, 0] = 10000.0
        state_in.body_f.assign(forces)
        solver.step(state_in, state_out, model.control(), None, 0.01)
        reaction = float(state_out.joint_qd.numpy()[0]) / 0.01 - 10000.0
        self.assertAlmostEqual(reaction, -1.0, delta=0.05)

    def test_finite_drive_bounds_complete_reaction_under_external_load(self):
        """Bound the complete implicit actuator reaction under external loading."""
        for force in (-10000.0, 10000.0):
            with self.subTest(force=force):
                builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
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
                    effort_limit=1.0,
                )
                builder.add_articulation([joint])
                model = builder.finalize(device="cpu")
                solver = SolverFeatherPGS(model, angular_damping=0.0)
                state_in, state_out = model.state(), model.state()
                newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
                state_in.body_f.assign(np.array([[force, 0, 0, 0, 0, 0]], dtype=np.float32))
                solver.step(state_in, state_out, model.control(), None, 0.01)
                reaction = float(state_out.joint_qd.numpy()[0]) / 0.01 - force
                self.assertLessEqual(abs(reaction), 1.05)
                self.assertAlmostEqual(reaction, -np.sign(force), delta=0.05)


if __name__ == "__main__":
    unittest.main()
