# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for world-subset features on heterogeneous multi-world models.

Covers:
- ArticulationView over a pattern that matches only a subset of worlds
  (``world_indices`` mapping, subset get/set round-trips).
- ``SolverMuJoCo(model, worlds=[...])`` simulating a subset of worlds of a
  model, matching a full-model solver trajectory exactly.
- ``SolverMuJoCoGroup`` auto-partitioning a structurally heterogeneous model
  into structural groups and stepping all of them.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton.selection import ArticulationView
from newton.solvers import SolverMuJoCo, SolverMuJoCoGroup


def _make_chain(n_links: int, label: str) -> newton.ModelBuilder:
    """Build an ``n_links`` serial revolute chain articulation labeled ``label``."""
    builder = newton.ModelBuilder()
    hx = 0.5
    joints = []
    prev = -1
    for i in range(n_links):
        link = builder.add_link()
        builder.add_shape_box(link, hx=hx, hy=0.05, hz=0.05)
        if prev == -1:
            parent_xform = wp.transform(p=wp.vec3(0.0, 0.0, 3.0), q=wp.quat_identity())
        else:
            parent_xform = wp.transform(p=wp.vec3(hx, 0.0, 0.0), q=wp.quat_identity())
        joints.append(
            builder.add_joint_revolute(
                parent=prev,
                child=link,
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent_xform=parent_xform,
                child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
                label=f"{label}_joint_{i}",
            )
        )
    builder.add_articulation(joints, label=label)
    return builder


def _make_alternating_model() -> newton.Model:
    """4 worlds alternating structure: worlds 0/2 = pendulum, worlds 1/3 = 3-link chain."""
    pendulum = _make_chain(1, label="pendulum")
    chain = _make_chain(3, label="chain")
    scene = newton.ModelBuilder()
    for world in range(4):
        scene.add_world(pendulum if world % 2 == 0 else chain)
    return scene.finalize()


def _make_homogeneous_pendulum_model(num_worlds: int = 4) -> newton.Model:
    """``num_worlds`` structurally identical single-pendulum worlds."""
    pendulum = _make_chain(1, label="pendulum")
    scene = newton.ModelBuilder()
    for _ in range(num_worlds):
        scene.add_world(pendulum)
    return scene.finalize()


# flat joint_q layout of the alternating model:
# world 0 -> [0:1], world 1 -> [1:4], world 2 -> [4:5], world 3 -> [5:8]
_PENDULUM_Q_SLICES = (slice(0, 1), slice(4, 5))
_CHAIN_Q_SLICES = (slice(1, 4), slice(5, 8))


class TestArticulationViewWorldSubset(unittest.TestCase):
    def test_articulation_view_subset_of_worlds(self):
        """A pattern matching only worlds {0, 2} yields a 2-world view mapped via world_indices."""
        model = _make_alternating_model()
        self.assertEqual(model.world_count, 4)
        self.assertEqual(model.joint_coord_count, 8)

        view = ArticulationView(model, pattern="pendulum")
        self.assertEqual(view.world_count, 2)
        self.assertEqual(list(view.world_indices), [0, 2])
        self.assertEqual(view.model_world_count, 4)
        self.assertEqual(view.count_per_world, 1)
        self.assertEqual(view.count, 2)
        self.assertEqual(view.joint_dof_count, 1)
        self.assertEqual(view.joint_coord_count, 1)

        chain_view = ArticulationView(model, pattern="chain")
        self.assertEqual(chain_view.world_count, 2)
        self.assertEqual(list(chain_view.world_indices), [1, 3])
        self.assertEqual(chain_view.joint_dof_count, 3)

        # set_dof_positions must land in the pendulum segments of the flat array only
        state = model.state()
        values = np.array([[[0.25]], [[-0.5]]], dtype=np.float32)  # (world, arti, coord)
        view.set_dof_positions(state, values)

        joint_q = state.joint_q.numpy()
        np.testing.assert_allclose(joint_q[_PENDULUM_Q_SLICES[0]], [0.25])
        np.testing.assert_allclose(joint_q[_PENDULUM_Q_SLICES[1]], [-0.5])
        for chain_slice in _CHAIN_Q_SLICES:
            np.testing.assert_allclose(joint_q[chain_slice], 0.0)

        # get_dof_positions must round-trip the same values
        got = view.get_dof_positions(state).numpy()
        self.assertEqual(got.shape, (2, 1, 1))
        np.testing.assert_allclose(got, values)

        # writes through the chain view land in the complementary segments
        chain_values = np.array([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]], dtype=np.float32)
        chain_view.set_dof_positions(state, chain_values)
        joint_q = state.joint_q.numpy()
        np.testing.assert_allclose(joint_q[_CHAIN_Q_SLICES[0]], [1.0, 2.0, 3.0])
        np.testing.assert_allclose(joint_q[_CHAIN_Q_SLICES[1]], [4.0, 5.0, 6.0])
        np.testing.assert_allclose(joint_q[_PENDULUM_Q_SLICES[0]], [0.25])
        np.testing.assert_allclose(joint_q[_PENDULUM_Q_SLICES[1]], [-0.5])


@unittest.skipUnless(wp.is_cuda_available(), "world-subset MuJoCo solvers require the GPU (mujoco_warp) backend")
class TestSolverMuJoCoWorldSubset(unittest.TestCase):
    SIM_DT = 1.0 / 240.0
    NUM_STEPS = 100

    @staticmethod
    def _rollout(model, solvers, num_steps, sim_dt):
        """Step ``model`` with one or more solvers sharing its state; return joint_q history."""
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
        history = np.zeros((num_steps, model.joint_coord_count))
        for step in range(num_steps):
            state_0.clear_forces()
            for solver in solvers:
                solver.step(state_0, state_1, control, None, sim_dt)
            state_0, state_1 = state_1, state_0
            history[step] = state_0.joint_q.numpy()
        return history

    def test_solver_mujoco_world_subset_matches_full(self):
        """Two subset solvers covering worlds {0,2} and {1,3} reproduce the full-model solve."""
        initial_q = np.array([0.3, -0.6, 0.9, -1.2], dtype=np.float32)

        model_full = _make_homogeneous_pendulum_model(4)
        model_full.joint_q.assign(initial_q)
        solver_full = SolverMuJoCo(model_full)
        history_full = self._rollout(model_full, [solver_full], self.NUM_STEPS, self.SIM_DT)

        model_split = _make_homogeneous_pendulum_model(4)
        model_split.joint_q.assign(initial_q)
        solver_even = SolverMuJoCo(model_split, worlds=[0, 2])
        solver_odd = SolverMuJoCo(model_split, worlds=[1, 3])
        history_split = self._rollout(model_split, [solver_even, solver_odd], self.NUM_STEPS, self.SIM_DT)

        self.assertTrue(np.all(np.isfinite(history_full)))
        self.assertTrue(np.all(np.isfinite(history_split)))
        # sanity: the pendulums actually moved
        self.assertGreater(np.max(np.abs(history_full[-1] - initial_q)), 1e-3)
        np.testing.assert_allclose(history_split, history_full, rtol=0.0, atol=1e-5)

    def test_solver_mujoco_group_heterogeneous(self):
        """SolverMuJoCoGroup groups alternating structures into 2 solvers and steps them stably."""
        model = _make_alternating_model()

        joint_q = model.joint_q.numpy()
        joint_q[_PENDULUM_Q_SLICES[0]] = 0.4
        joint_q[_PENDULUM_Q_SLICES[1]] = 0.4  # world 2 starts identical to world 0
        joint_q[_CHAIN_Q_SLICES[0]] = [0.2, -0.1, 0.3]
        joint_q[_CHAIN_Q_SLICES[1]] = [-0.5, 0.25, -0.75]
        model.joint_q.assign(joint_q)

        solver = SolverMuJoCoGroup(model)
        self.assertEqual(solver.world_groups, [[0, 2], [1, 3]])
        self.assertEqual(len(solver.solvers), 2)

        history = self._rollout(model, [solver], self.NUM_STEPS, self.SIM_DT)

        self.assertTrue(np.all(np.isfinite(history)), "heterogeneous grouped rollout produced non-finite joint_q")
        # worlds moved under gravity
        self.assertGreater(np.max(np.abs(history[-1] - joint_q)), 1e-3)
        # worlds 0 and 2 share structure, initial state, and solver rows -> identical trajectories
        np.testing.assert_allclose(
            history[:, _PENDULUM_Q_SLICES[0]],
            history[:, _PENDULUM_Q_SLICES[1]],
            rtol=0.0,
            atol=1e-6,
        )
        # worlds 1 and 3 share structure but start differently -> trajectories must differ
        self.assertGreater(
            np.max(np.abs(history[:, _CHAIN_Q_SLICES[0]] - history[:, _CHAIN_Q_SLICES[1]])),
            1e-3,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
