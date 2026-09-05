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


def _run_state_trajectory(model, solver, num_steps, *, reset_each_step):
    state_0 = _make_initial_state(model)
    state_1 = model.state()
    contacts = model.contacts()
    control = model.control()
    history = {name: [] for name in ("joint_q", "joint_qd", "body_q", "body_qd")}
    for _ in range(num_steps):
        if reset_each_step:
            solver.reset(state_0)
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0
        for name, values in history.items():
            values.append(getattr(state_0, name).numpy().copy())
    return {name: np.stack(values) for name, values in history.items()}


class TestFeatherPGSMassUpdateInterval(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "FK/ID reuse requires CUDA")
    def test_cached_fk_id_matches_forced_recomputation(self):
        trajectories = []
        for reset_each_step in (False, True):
            model = _build_model("cuda:0", ground=False)
            solver = SolverFeatherPGS(model, update_mass_matrix_interval=2)
            trajectories.append(_run_state_trajectory(model, solver, num_steps=20, reset_each_step=reset_each_step))

        for name in trajectories[0]:
            np.testing.assert_allclose(
                trajectories[0][name], trajectories[1][name], rtol=0.0, atol=2.0e-6, err_msg=name
            )

    @unittest.skipUnless(wp.is_cuda_available(), "parallel compact inertia refresh requires CUDA")
    def test_parallel_compact_refresh_matches_serial_trajectory(self):
        trajectories = []
        for use_parallel_streams in (False, True):
            model = _build_model("cuda:0")
            solver = SolverFeatherPGS(
                model,
                update_mass_matrix_interval=1,
                double_buffer=False,
                use_parallel_streams=use_parallel_streams,
            )
            trajectories.append(_run_trajectory(model, solver, num_steps=20))

        np.testing.assert_allclose(trajectories[1], trajectories[0], rtol=0.0, atol=2.0e-6)

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

    def test_model_change_request_refreshes_a_reuse_step(self):
        model = _build_model(wp.get_device(), ground=False)
        solver = SolverFeatherPGS(model, update_mass_matrix_interval=2)
        state_0 = _make_initial_state(model)
        state_1 = model.state()
        contacts = model.contacts()
        control = model.control()

        model.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0
        solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
        model.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT)

        self.assertEqual(solver.mass_update_mask.numpy().tolist(), [1, 1])
        self.assertEqual(solver._mass_update_requested.numpy().tolist(), [0])

    def test_mass_refresh_has_no_obsolete_limit_count_state(self):
        solver = SolverFeatherPGS(_build_model(wp.get_device(), ground=False))
        for attribute in ("aug_limit_counts", "aug_prev_limit_counts", "limit_change_mask"):
            self.assertFalse(hasattr(solver, attribute), f"obsolete mass-refresh state {attribute!r} was restored")


if __name__ == "__main__":
    unittest.main(verbosity=2)
