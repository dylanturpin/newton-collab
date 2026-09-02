# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for mimic constraint rows in SolverFeatherPGS (matrix-free mode)."""

import unittest

import warp as wp

import newton
from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_MIMIC


def _build_two_revolute_chain(coef0: float, coef1: float):
    """Build a fixed-base chain of two revolute Z-joints with a mimic between them.

    The leader joint is position-driven; the follower joint has no drive and no spring,
    so any tracking it does comes from the mimic constraint alone.
    """
    b = newton.ModelBuilder(up_axis=newton.Axis.Z)
    # add_link (not add_body): add_body eagerly wraps each body in its own
    # single-body free-joint articulation, which would split this chain.
    link_a = b.add_link(xform=wp.transform(wp.vec3(0.2, 0.0, 0.5), wp.quat_identity()))
    b.add_shape_box(link_a, hx=0.1, hy=0.02, hz=0.02)
    j_leader = b.add_joint_revolute(
        parent=-1,
        child=link_a,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
    )
    link_b = b.add_link(xform=wp.transform(wp.vec3(0.6, 0.0, 0.5), wp.quat_identity()))
    b.add_shape_box(link_b, hx=0.1, hy=0.02, hz=0.02)
    j_follower = b.add_joint_revolute(
        parent=link_a,
        child=link_b,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
    )
    b.add_articulation([j_leader, j_follower], label="mimic_chain")
    # Drive the leader only.
    b.joint_target_ke[0] = 50.0
    b.joint_target_kd[0] = 5.0
    b.joint_target_mode[0] = int(newton.JointTargetMode.POSITION)
    # follower: q_follower = coef0 + coef1 * q_leader
    b.add_constraint_mimic(joint0=j_follower, joint1=j_leader, coef0=coef0, coef1=coef1)
    return b, j_leader, j_follower


def _run_chain(coef0: float, coef1: float, leader_target: float, steps: int = 600, **solver_kwargs):
    builder, _, _ = _build_two_revolute_chain(coef0, coef1)
    model = builder.finalize()
    solver = newton.solvers.SolverFeatherPGS(
        model, pgs_mode="matrix_free", pgs_iterations=16, pgs_beta=0.1, **solver_kwargs
    )
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    targets = model.joint_target_q.numpy().copy()
    targets[0] = leader_target
    control.joint_target_q.assign(targets)
    dt = 1.0 / 240.0
    for _ in range(steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, dt)
        state_0, state_1 = state_1, state_0
    return solver, state_0.joint_q.numpy()


@unittest.skipUnless(wp.get_device().is_cuda, "SolverFeatherPGS matrix-free mode requires CUDA")
class TestFeatherPGSMimic(unittest.TestCase):
    def test_identity_mimic_tracks_leader(self):
        """Verify a 1:1 mimic makes the undriven follower joint track the driven leader."""
        _, q = _run_chain(coef0=0.0, coef1=1.0, leader_target=0.5)
        self.assertAlmostEqual(q[0], 0.5, delta=0.05)
        self.assertAlmostEqual(q[1], q[0], delta=0.02)

    def test_scaled_offset_mimic(self):
        """Verify q_follower converges to coef0 + coef1 * q_leader for a scaled, offset mimic."""
        _, q = _run_chain(coef0=0.1, coef1=-0.5, leader_target=0.6)
        self.assertAlmostEqual(q[1], 0.1 - 0.5 * q[0], delta=0.02)

    def test_mimic_row_is_assembled(self):
        """Verify the solver assembles a MIMIC constraint row that carries the coupling."""
        solver, _ = _run_chain(coef0=0.0, coef1=1.0, leader_target=0.5, steps=10)
        row_types = solver.row_type.numpy()
        counts = solver.constraint_count.numpy()
        rows = [int(t) for w in range(row_types.shape[0]) for t in row_types[w, : counts[w]]]
        self.assertIn(PGS_CONSTRAINT_TYPE_MIMIC, rows)

    def test_replicated_mimics_have_articulation_local_ranges(self):
        """Verify replicated mimic rows use one compact lookup range per articulation."""
        template, _, _ = _build_two_revolute_chain(coef0=0.0, coef1=1.0)
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.replicate(template, world_count=4)
        model = builder.finalize()
        solver = newton.solvers.SolverFeatherPGS(model, pgs_mode="matrix_free")

        self.assertEqual(solver._mimic_art_start.numpy().tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(solver._mimic_art_list.numpy().tolist(), [0, 1, 2, 3])

    def test_disabled_mimic_is_ignored(self):
        """Verify a disabled mimic constraint leaves the follower joint uncoupled."""
        builder, _, _ = _build_two_revolute_chain(0.0, 1.0)
        builder.constraint_mimic_enabled[0] = False
        model = builder.finalize()
        solver = newton.solvers.SolverFeatherPGS(model, pgs_mode="matrix_free", pgs_iterations=16)
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        targets = model.joint_target_q.numpy().copy()
        targets[0] = 0.5
        control.joint_target_q.assign(targets)
        dt = 1.0 / 240.0
        for _ in range(600):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, None, dt)
            state_0, state_1 = state_1, state_0
        q = state_0.joint_q.numpy()
        self.assertAlmostEqual(q[0], 0.5, delta=0.05)
        # The undriven follower swings free under gravity; it must NOT sit at the leader angle.
        self.assertNotAlmostEqual(q[1], q[0], delta=0.02)


if __name__ == "__main__":
    unittest.main()
