# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Integration checks for the sparse dynamics and contact path."""

import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_JOINT_LIMIT
from newton.solvers import SolverFeatherPGS
from newton.tests.test_feather_pgs_propagation_free_root_warp import _build_model as _build_chain

DT = 1.0 / 400.0
STATE_FIELDS = ("joint_q", "joint_qd", "body_q", "body_qd")


def _build_model():
    """Build two free-root tripods with driven, limited, contacting branches."""
    env = newton.ModelBuilder()
    root = env.add_link(
        xform=wp.transform(wp.vec3(0, 0, 0.16), wp.quat_identity()), mass=2.0, inertia=wp.mat33(np.eye(3) * 0.03)
    )
    joints = [env.add_joint_free(root)]
    for x, y in ((0.16, 0.0), (-0.08, 0.14), (-0.08, -0.14)):
        child = env.add_link(mass=0.5, inertia=wp.mat33(np.eye(3) * 0.002))
        cfg = env.default_shape_cfg.copy()
        cfg.density, cfg.mu = 0.0, 0.6
        env.add_shape_sphere(child, radius=0.055, cfg=cfg)
        joints.append(
            env.add_joint_revolute(
                root,
                child,
                axis=newton.Axis.Y,
                parent_xform=wp.transform(wp.vec3(x, y, -0.11), wp.quat_identity()),
                target_pos=0.05,
                target_ke=15.0,
                target_kd=1.0,
                armature=0.05,
                limit_lower=-0.2,
                limit_upper=0.2,
                effort_limit=10.0,
            )
        )
    env.add_articulation(joints)
    env.joint_q[7] = 0.23  # Exercise an actual unilateral limit on the first step.
    builder = newton.ModelBuilder()
    builder.replicate(env, 2)
    builder.add_ground_plane()
    return builder.finalize(device="cuda:0")


def _make_solver(model, sparse, **overrides):
    """Change only the representation selection between comparison solvers."""
    options = {
        "pgs_mode": "matrix_free",
        "articulated_contact_response": "immediate",
        "drive_mode": "augmented",
        "friction_mode": "current",
        "pgs_iterations": 32,
        "enable_joint_limits": True,
        "enable_joint_velocity_limits": False,
        "pgs_warmstart": False,
        "mf_warmstart": False,
        "dense_max_constraints": 32,
        "use_parallel_streams": True,
        "double_buffer": False,
        "update_mass_matrix_interval": 4,
    }
    options.update(overrides)
    with mock.patch.dict(SolverFeatherPGS._kernel_overrides, {"sparse_mass_matrix": sparse}):
        return SolverFeatherPGS(model, **options)


def _case(sparse):
    """Allocate a stable input state for eager and captured cache reuse."""
    model = _build_model()
    solver = _make_solver(model, sparse)
    state, out = model.state(), model.state()
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=64)
    return model, solver, state, out, model.control(), pipeline, pipeline.contacts()


def _step(case):
    """Advance without swapping the input object used by the dynamics cache."""
    _, solver, state, out, control, pipeline, contacts = case
    state.clear_forces()
    pipeline.collide(state, contacts)
    solver.step(state, out, control, contacts, DT)
    for name in STATE_FIELDS:
        wp.copy(getattr(state, name), getattr(out, name))


@unittest.skipUnless(wp.is_cuda_available(), "Sparse integrated solve requires CUDA")
class TestFeatherPGSSparseSolver(unittest.TestCase):
    def assert_states_close(self, first, second):
        """Allow float32 factor-order roundoff, not a changed physics recipe."""
        for name in STATE_FIELDS:
            # Independent elimination/reduction orders accumulate over 32 contact steps.
            tolerance = 2.0e-3 if name.endswith("qd") else 2.0e-4
            np.testing.assert_allclose(
                getattr(first, name).numpy(), getattr(second, name).numpy(), rtol=2.0e-4, atol=tolerance, err_msg=name
            )

    def assert_sparse_state(self, case):
        """Check storage elimination, row support, impulses, and residual coordinates."""
        _, solver, state, *_ = case
        size = solver._sparse_mass_matrix_size
        self.assertEqual(size, 9)
        for array in (
            solver.J_world,
            solver.Y_world,
            solver.J_by_size[size],
            solver.Y_by_size[size],
            solver.H_by_size[size],
            solver.L_by_size[size],
        ):
            self.assertEqual(array.shape, (1, 1, 1))  # Only argument stand-ins remain.
        np.testing.assert_array_equal(solver._sparse_mass_matrix_status.numpy(), 0)
        self.assertFalse(solver.constraint_overflow.numpy().any())
        for name in STATE_FIELDS:
            self.assertTrue(np.isfinite(getattr(state, name).numpy()).all(), name)
        plan = solver._sparse_mass_matrix_plan
        counts, impulses = solver.constraint_count.numpy(), solver.impulses.numpy()
        row_dof, values = solver._sparse_row_dof.numpy(), solver._sparse_row_factor.numpy()
        incidence, rhs = solver._sparse_row_incident.numpy(), solver.rhs.numpy()
        v_hat, v_out = solver.v_hat.numpy(), solver.v_out.numpy()
        for group, art in enumerate(solver.group_to_art[size].numpy()):
            world = int(solver.art_to_world.numpy()[art])
            count = int(counts[world])
            self.assertLessEqual(count, solver.dense_max_constraints)
            self.assertTrue(np.isfinite(impulses[world, :count]).all())
            kinds = solver.row_type.numpy()[world, :count]
            unilateral = (kinds == PGS_CONSTRAINT_TYPE_CONTACT) | (kinds == PGS_CONSTRAINT_TYPE_JOINT_LIMIT)
            self.assertTrue(np.all(impulses[world, :count][unilateral] >= -1.0e-7))
            z = np.zeros((count, size))
            for row in range(count):
                support = row_dof[world, row]
                valid = support >= 0
                self.assertTrue(np.all(support[valid] < size))
                self.assertEqual(len(set(support[valid])), int(valid.sum()))
                z[row, support[valid]] = values[world, row, valid]
            lower = np.zeros((size, size))
            lower[plan.entry_rows, plan.columns] = solver._sparse_L.numpy()[group]
            physical_j = np.zeros_like(z)
            physical_j[:, plan.permutation] = z @ lower.T
            start = int(solver.articulation_dof_start.numpy()[art])
            velocity = slice(start, start + size)
            np.testing.assert_allclose(physical_j @ v_hat[velocity], incidence[world, :count], atol=2.0e-5, rtol=2.0e-5)
            factor_residual = (
                z @ solver._sparse_factor_velocity_delta.numpy()[world] + incidence[world, :count] + rhs[world, :count]
            )
            np.testing.assert_allclose(
                physical_j @ v_out[velocity] + rhs[world, :count], factor_residual, atol=3.0e-5, rtol=3.0e-5
            )

    def test_trajectory_masked_reset_and_inertial_notify(self):
        """Match 32 contact steps and refresh only the requested cached factors."""
        dense, sparse = _case(False), _case(True)
        self.assertIsNone(dense[1]._sparse_mass_matrix_size)
        self.assertEqual(sparse[1]._sparse_mass_matrix_size, 9)
        contact_seen = limit_seen = False
        for _ in range(32):
            for case in (dense, sparse):
                _step(case)
            contact_seen |= bool(sparse[-1].rigid_contact_count.numpy()[0])
            limit_seen |= bool(np.any(sparse[1].row_type.numpy() == PGS_CONSTRAINT_TYPE_JOINT_LIMIT))
            self.assert_states_close(dense[2], sparse[2])
        self.assertTrue(contact_seen and limit_seen, "Fixture must exercise contacts and joint limits")
        self.assert_sparse_state(sparse)
        for case in (dense, sparse):
            _step(case)  # Step32 refreshes globally; step33 is a held-mass step.
        held = sparse[1]._sparse_Linv.numpy().copy()
        for model, solver, state, *_ in (dense, sparse):
            q = state.joint_q.numpy()
            q[2] += 0.02
            state.joint_q.assign(q)
            newton.eval_fk(model, state.joint_q, state.joint_qd, state)
            solver.reset(state, wp.array([True, False], dtype=bool, device=model.device))
        for case in (dense, sparse):
            _step(case)
        np.testing.assert_array_equal(sparse[1].mass_update_mask.numpy(), [1, 0])
        np.testing.assert_array_equal(sparse[1]._sparse_Linv.numpy()[1], held[1])
        self.assert_states_close(dense[2], sparse[2])
        for model, solver, *_ in (dense, sparse):
            mass = model.body_mass.numpy()
            mass[0] *= 1.1
            model.body_mass.assign(mass)
            solver.notify_model_changed(newton.ModelFlags.BODY_INERTIAL_PROPERTIES)
        for case in (dense, sparse):
            _step(case)
        np.testing.assert_array_equal(sparse[1].mass_update_mask.numpy(), [1, 1])
        self.assertFalse(np.array_equal(sparse[1]._sparse_Linv.numpy()[0], held[0]))
        self.assert_states_close(dense[2], sparse[2])
        self.assert_sparse_state(sparse)

    def test_graph_replay_preserves_sparse_problem(self):
        """Replay a complete four-step mass-refresh cadence without dense scratch."""
        eager, graph = _case(True), _case(True)
        for case in (eager, graph):
            _step(case)  # Finish allocation/compilation outside capture.
        with wp.ScopedCapture("cuda:0") as capture:
            for _ in range(4):
                _step(graph)
        for _ in range(8):
            wp.capture_launch(capture.graph)
            for _ in range(4):
                _step(eager)
        self.assert_states_close(eager[2], graph[2])
        self.assert_sparse_state(graph)

    def test_unsupported_configurations_keep_existing_path(self):
        """Reject full-support chains, torsion, velocity limits, and warm starts."""
        chain = _build_chain("cuda:0", base_z=1.2, contact_sphere="deep", ground=False)
        self.assertIsNone(_make_solver(chain, True)._sparse_mass_matrix_size)
        model = _build_model()
        for options in (
            {"contact_torsion_radius": 0.01},
            {"enable_joint_velocity_limits": True},
            {"pgs_warmstart": True},
            {"mf_warmstart": True},
            {"dense_max_constraints": 2048},
        ):
            with self.subTest(options=options):
                solver = _make_solver(model, True, **options)
                self.assertIsNone(solver._sparse_mass_matrix_size)
                self.assertGreater(solver.J_by_size[9].size, 1)


if __name__ == "__main__":
    unittest.main()
