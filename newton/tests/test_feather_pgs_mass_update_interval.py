# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS

DT = 1.0 / 60.0
# Per-articulation initial pose/velocity: [base, left branch, right branch].
INITIAL_JOINT_Q = (0.7, 0.3, -0.4)
INITIAL_JOINT_QD = (0.5, -0.2, 0.3)


def _build_model(device, num_worlds=2, ground=True):
    """Two-branch pendulum: base revolute joint plus two sibling branch links.

    The sibling branches are not ancestor-related, so H has structural zeros
    between their DOFs; with a ground plane the branch tips swing into contact.
    """
    env = newton.ModelBuilder()
    env.default_shape_cfg.density = 1000.0

    base = env.add_link()
    env.add_shape_box(base, hx=0.08, hy=0.08, hz=0.08)
    left = env.add_link()
    env.add_shape_box(left, hx=0.25, hy=0.05, hz=0.05)
    right = env.add_link()
    env.add_shape_box(right, hx=0.25, hy=0.05, hz=0.05)

    j_base = env.add_joint_revolute(
        -1,
        base,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.55), wp.quat_identity()),
        axis=newton.Axis.Y,
        target_ke=15.0,
        target_kd=1.5,
    )
    j_left = env.add_joint_revolute(
        base,
        left,
        parent_xform=wp.transform(wp.vec3(0.08, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.25, 0.0, 0.0), wp.quat_identity()),
        axis=newton.Axis.Y,
        target_ke=15.0,
        target_kd=1.5,
    )
    j_right = env.add_joint_revolute(
        base,
        right,
        parent_xform=wp.transform(wp.vec3(-0.08, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.25, 0.0, 0.0), wp.quat_identity()),
        axis=newton.Axis.Y,
        target_ke=15.0,
        target_kd=1.5,
    )
    env.add_articulation([j_base, j_left, j_right])

    builder = newton.ModelBuilder()
    builder.replicate(env, num_worlds)
    if ground:
        builder.add_ground_plane()
    return builder.finalize(device=device)


def _make_initial_state(model):
    state = model.state()
    num_arts = model.articulation_count
    state.joint_q.assign(np.tile(np.asarray(INITIAL_JOINT_Q, dtype=np.float32), num_arts))
    state.joint_qd.assign(np.tile(np.asarray(INITIAL_JOINT_QD, dtype=np.float32), num_arts))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return state


def _run_trajectory(model, solver, num_steps):
    state_0 = _make_initial_state(model)
    state_1 = model.state()
    contacts = model.contacts()
    control = model.control()
    joint_q_history = []
    for _ in range(num_steps):
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0
        joint_q_history.append(state_0.joint_q.numpy().copy())
    return np.stack(joint_q_history)


class TestFeatherPGSMassUpdateInterval(unittest.TestCase):
    def test_interval_two_contact_trajectory_stays_close_to_reference(self):
        device = wp.get_device()
        history = {}
        for interval in (1, 2):
            model = _build_model(device)
            solver = SolverFeatherPGS(model, update_mass_matrix_interval=interval)
            history[interval] = _run_trajectory(model, solver, num_steps=60)

        self.assertTrue(np.isfinite(history[2]).all(), "interval=2 trajectory diverged to non-finite values")
        # The trajectories are intentionally NOT bit-identical (stale H between
        # mass updates is the optimization); they must stay close on a short
        # contact-rich horizon.
        drift = np.abs(history[2] - history[1]).max()
        self.assertLess(drift, 0.05, f"interval=2 drifted {drift} rad from the interval=1 reference")
        # Sanity: the scene actually moved, so the comparison is not vacuous.
        moved = np.abs(history[1][-1] - np.tile(np.asarray(INITIAL_JOINT_Q, dtype=np.float32), 2)).max()
        self.assertGreater(moved, 1.0e-3)

    def test_global_flag_cadence_bakes_one_zero_pattern(self):
        device = wp.get_device()
        model = _build_model(device)
        solver = SolverFeatherPGS(model, update_mass_matrix_interval=2)
        state_0 = _make_initial_state(model)
        state_1 = model.state()
        contacts = model.contacts()
        control = model.control()

        expected_masks = ([1, 1], [0, 0], [1, 1], [0, 0])
        for step_index, expected in enumerate(expected_masks):
            model.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
            self.assertEqual(
                solver.mass_update_mask.numpy().tolist(),
                expected,
                f"unexpected mass_update_mask after step {step_index}",
            )

    def test_limit_change_refresh_on_skipped_step_matches_full_rebuild(self):
        """A limit-count change on a global_flag=0 step must refresh that articulation's mass factor.

        At this base revision ``build_augmented_joint_rows`` always writes
        ``aug_limit_counts = 0`` (the device-side producer is inert), so a joint
        crossing the limit activation gap cannot change the count yet. We emulate
        the crossing by perturbing ``aug_prev_limit_counts``, which is exactly the
        signal ``detect_limit_count_changes`` consumes. The refreshed Cholesky
        factor must match an interval=1 reference bitwise-closely even though the
        masked rebuild writes into a stale (non-zeroed) H buffer.
        """
        device = wp.get_device()
        model_a = _build_model(device, ground=False)
        model_b = _build_model(device, ground=False)
        solver_a = SolverFeatherPGS(
            model_a, update_mass_matrix_interval=2, double_buffer=False, use_parallel_streams=False
        )
        solver_b = SolverFeatherPGS(
            model_b, update_mass_matrix_interval=1, double_buffer=False, use_parallel_streams=False
        )

        runs = []
        for model, solver in ((model_a, solver_a), (model_b, solver_b)):
            state_0 = _make_initial_state(model)
            state_1 = model.state()
            contacts = model.contacts()
            control = model.control()
            runs.append((model, solver, state_0, state_1, contacts, control))

        def step_once(run):
            model, solver, state_0, state_1, contacts, control = run
            model.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            return (model, solver, state_1, state_0, contacts, control)

        # Step 0: global flag is 1 for both solvers; identical full updates.
        runs = [step_once(run) for run in runs]
        np.testing.assert_array_equal(runs[0][2].joint_q.numpy(), runs[1][2].joint_q.numpy())

        # Emulate articulation 0 crossing a joint-limit activation gap before
        # the interval=2 solver's flag-0 step.
        prev_counts = solver_a.aug_prev_limit_counts.numpy()
        prev_counts[0] = 1
        solver_a.aug_prev_limit_counts.assign(prev_counts)

        # Step 1: solver_a global flag is 0; only articulation 0 must refresh.
        runs = [step_once(run) for run in runs]
        self.assertEqual(solver_a.mass_update_mask.numpy().tolist(), [1, 0])
        self.assertEqual(solver_b.mass_update_mask.numpy().tolist(), [1, 1])

        size = next(iter(solver_a.size_groups))
        L_a = solver_a.L_by_size[size].numpy()
        L_b = solver_b.L_by_size[size].numpy()
        # Refreshed articulation: masked rebuild into the stale H buffer must
        # reproduce the full-rebuild factorization.
        np.testing.assert_allclose(L_a[0], L_b[0], rtol=0.0, atol=1.0e-6)
        # Skipped articulation: still carries the step-0 factorization.
        self.assertFalse(np.array_equal(L_a[1], L_b[1]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
