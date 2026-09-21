# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check complete level-synchronous FPGS trees against serial dynamics."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.solver_feather_pgs import _FeatherPGSTreePlan
from newton.solvers import SolverFeatherPGS
from newton.tests.test_feather_pgs_leaf_publication import _build_branched_model, _build_model, _fields, _solver


def _build_fingers(device="cpu", *, articulations=3, include_chain=False):
    """Build four four-joint fingers per moving palm, plus an optional serial tree."""
    builder = newton.ModelBuilder(gravity=(0.7, -1.2, -9.1))
    inertia = wp.mat33(0.3, 0.02, 0.01, 0.02, 0.4, 0.03, 0.01, 0.03, 0.5)
    for articulation in range(articulations + int(include_chain)):
        root = builder.add_link(mass=1.0, inertia=inertia, com=wp.vec3(0.03, -0.02, 0.01))
        joints = [
            builder.add_joint_revolute(
                parent=-1,
                child=root,
                axis=newton.Axis.Z,
                parent_xform=wp.transform(wp.vec3(2.0 * articulation, -0.3, 0.8), wp.quat_rpy(0.2, -0.1, 0.3)),
            )
        ]
        fingers = 1 if articulation == articulations else 4
        for finger in range(fingers):
            parent = root
            for level in range(4):
                child = builder.add_link(mass=0.5, inertia=inertia, com=wp.vec3(0.02, 0.01, -0.03))
                joints.append(
                    builder.add_joint_revolute(
                        parent=parent,
                        child=child,
                        axis=newton.Axis.Y if level % 2 else newton.Axis.X,
                        parent_xform=wp.transform(wp.vec3(0.05, 0.04 * finger, 0.08), wp.quat_rpy(0.1, -0.2, 0.05)),
                        child_xform=wp.transform(wp.vec3(-0.01, 0.02, 0.03), wp.quat_identity()),
                    )
                )
                parent = child
        builder.add_articulation(joints)
    model = builder.finalize(device=device)
    model.joint_q.assign(np.linspace(-0.3, 0.4, model.joint_coord_count, dtype=np.float32))
    model.joint_qd.assign(np.linspace(-0.7, 0.8, model.joint_dof_count, dtype=np.float32))
    return model


class TestFeatherPGSTreePlan(unittest.TestCase):
    def test_complete_finger_levels_and_serial_group(self):
        """Schedule every finger segment and retain serial trees as one-lane groups."""
        model = _build_fingers(include_chain=True)
        solver = SolverFeatherPGS(model, pgs_mode="split", use_parallel_streams=False)
        plan = _FeatherPGSTreePlan.build(model, solver.articulation_joint_end)
        self.assertIsNotNone(plan)
        self.assertTrue(any(group.lanes > 1 for group in plan.groups))
        self.assertTrue(any(group.lanes == 1 for group in plan.groups))
        seen = []
        parents = model.joint_parent.numpy()
        children = model.joint_child.numpy()
        starts = model.articulation_start.numpy()
        for group in plan.groups:
            offsets = group.level_offsets.numpy()
            joints = group.level_joints.numpy()
            for row, articulation in enumerate(group.articulations.numpy()):
                levels = {}
                for level in range(group.max_levels):
                    for joint in joints[offsets[row, level] : offsets[row, level + 1]]:
                        levels[int(children[joint])] = level
                        if parents[joint] >= 0:
                            self.assertLess(levels[int(parents[joint])], level)
                        seen.append(int(joint))
                if articulation < 3:
                    self.assertEqual(
                        [int(offsets[row, level + 1] - offsets[row, level]) for level in range(5)], [1, 4, 4, 4, 4]
                    )
                    self.assertEqual(len(levels), int(starts[articulation + 1] - starts[articulation]))
        self.assertEqual(sorted(seen), list(range(model.joint_count)))

    def test_serial_and_unsafe_topologies_fall_back(self):
        """Retain wholly serial, aliased, cross-articulation and loop topologies."""
        model, _, _ = _build_model(chain=True)
        solver = SolverFeatherPGS(model, pgs_mode="split", use_parallel_streams=False)
        self.assertIsNone(_FeatherPGSTreePlan.build(model, solver.articulation_joint_end))
        for unsafe in ("alias", "foreign_parent", "loop"):
            with self.subTest(unsafe=unsafe):
                model = _build_fingers(articulations=2)
                solver = SolverFeatherPGS(model, pgs_mode="split", use_parallel_streams=False)
                ends = solver.articulation_joint_end.numpy().copy()
                if unsafe == "alias":
                    children = model.joint_child.numpy().copy()
                    children[2] = children[1]
                    model.joint_child.assign(children)
                elif unsafe == "foreign_parent":
                    parents = model.joint_parent.numpy().copy()
                    parents[1] = model.joint_child.numpy()[model.articulation_start.numpy()[1]]
                    model.joint_parent.assign(parents)
                else:
                    ends[0] -= 1
                plan = _FeatherPGSTreePlan.build(model, wp.array(ends, dtype=wp.int32, device=model.device))
                self.assertIsNone(plan)


class TestFeatherPGSTreeExecution(unittest.TestCase):
    def test_cuda_stage1_cold_mixed_cache_and_velocity_prescale(self):
        """Match every canonical dynamics field across partial warps and active prescaling."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("Parallel tree execution requires CUDA")
        for device in devices:
            for velocity_limits in (False, True):
                with self.subTest(device=str(device), velocity_limits=velocity_limits):
                    model = _build_fingers(device)
                    model.joint_velocity_limit.fill_(0.05)
                    solvers = []
                    states = []
                    outputs = []
                    for parallel in (False, True):
                        solver = SolverFeatherPGS(
                            model,
                            pgs_mode="matrix_free",
                            pgs_iterations=8,
                            use_parallel_streams=False,
                            enable_joint_velocity_limits=velocity_limits,
                        )
                        if parallel:
                            self.assertIsNotNone(solver._tree_plan)
                        else:
                            solver._tree_plan = None
                        state, following = model.state(), model.state()
                        solver._prepare_augmented_state(state, following, model.control())
                        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
                        for name, array in _fields(solver, state).items():
                            if name not in ("body_q", "body_qd", "valid"):
                                array.fill_(-123.25)
                        solver._fk_id_cache_valid.zero_()
                        solvers.append(solver)
                        states.append(state)
                        outputs.append(following)
                    original_qd = model.joint_qd.numpy().copy()
                    for phase in ("cold", "mixed", "warm"):
                        for solver, state, following in zip(solvers, states, outputs, strict=True):
                            if phase == "mixed":
                                q = state.joint_q.numpy().copy()
                                start, end = model.articulation_start.numpy()[1:3]
                                q0, q1 = model.joint_q_start.numpy()[[start, end]]
                                q[q0:q1] += 0.1
                                state.joint_q.assign(q)
                                solver._fk_id_cache_valid.assign(np.array([1, 0, 1], dtype=np.int32))
                            elif phase == "warm":
                                solver._fk_id_cache_valid.fill_(1)
                            _, predictor_qd = solver._stage1_fk_id(state, solver, following)
                            np.testing.assert_array_equal(state.joint_qd.numpy(), original_qd)
                            if velocity_limits:
                                self.assertGreater(float(np.max(np.abs(predictor_qd.numpy() - original_qd))), 0.01)
                        for name, actual in _fields(solvers[1], states[1]).items():
                            np.testing.assert_allclose(
                                actual.numpy(),
                                _fields(solvers[0], states[0])[name].numpy(),
                                rtol=3e-6,
                                atol=3e-6,
                                err_msg=f"{phase}: {name}",
                            )
                        if velocity_limits:
                            np.testing.assert_array_equal(solvers[1].qd_work.numpy(), solvers[0].qd_work.numpy())

    def test_cuda_uncached_publication_matches_public_fk(self):
        """Publish uncached FREE and mixed D6 states using the public FK convention."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("Parallel tree publication requires CUDA")
        for device in devices:
            model, _, _ = _build_branched_model(device, floating_root=True)
            solver = _solver(model, enabled=True, mode="matrix_free")
            solver._fk_id_cache_enabled = False
            state = model.state()
            state.body_q.fill_(-123.25)
            state.body_qd.fill_(-123.25)
            solver._stage7_update_kinematics(state, solver)
            reference = model.state()
            newton.eval_fk(model, reference.joint_q, reference.joint_qd, reference)
            for name in ("body_q", "body_qd"):
                np.testing.assert_allclose(
                    getattr(state, name).numpy(), getattr(reference, name).numpy(), rtol=3e-6, atol=3e-6, err_msg=name
                )


if __name__ == "__main__":
    unittest.main()
