# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check body-parallel publication against the existing articulation traversal."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.prismatic_publication import PrismaticPublicationPlan
from newton.solvers import SolverFeatherPGS


def _build_model(device="cpu", *, locked_d6=False, chain=False, leaves=7):
    """Use real joints, nontrivial anchors/COMs, and unrelated moving bodies."""
    builder = newton.ModelBuilder(gravity=(0.7, -1.2, -9.1))

    def body(index):
        return builder.add_link(
            mass=0.5 + 0.1 * index,
            com=wp.vec3(0.07, -0.03, 0.02),
            inertia=wp.mat33(0.3, 0.02, 0.01, 0.02, 0.4, 0.03, 0.01, 0.03, 0.5),
        )

    base = body(0)
    root_kwargs = {
        "parent": -1,
        "child": base,
        "parent_xform": wp.transform(wp.vec3(2.0, -1.0, 0.6), wp.quat_rpy(0.3, -0.2, 0.4)),
        "child_xform": wp.transform(wp.vec3(0.2, -0.1, 0.05), wp.quat_rpy(-0.1, 0.2, 0.0)),
    }
    root = builder.add_joint_d6(**root_kwargs) if locked_d6 else builder.add_joint_fixed(**root_kwargs)
    joints = [root]
    leaf_bodies = []
    for index in range(leaves):
        child = body(index + 1)
        leaf_bodies.append(child)
        joints.append(
            builder.add_joint_prismatic(
                parent=leaf_bodies[0] if chain and index == 1 else base,
                child=child,
                axis=wp.vec3(0.2 + 0.1 * index, 0.5, 0.9),
                parent_xform=wp.transform(wp.vec3(0.1 * index, -0.05, 0.13), wp.quat_rpy(0.2, -0.1 * index, 0.3)),
                child_xform=wp.transform(wp.vec3(-0.09, 0.04, 0.08), wp.quat_rpy(0.4, 0.2, -0.1)),
                target_ke=3.0,
                target_kd=0.2,
                limit_lower=-0.5,
                limit_upper=0.5,
            )
        )
    builder.add_articulation(joints)
    arm = body(leaves + 1)
    hinge = builder.add_joint_revolute(
        parent=-1,
        child=arm,
        axis=newton.Axis.Y,
        parent_xform=wp.transform(wp.vec3(-0.8, 0.1, 1.0), wp.quat_identity()),
    )
    builder.add_articulation([hinge])
    free = body(leaves + 2)
    builder.add_articulation([builder.add_joint_free(free)])
    model = builder.finalize(device=device)
    model.rigid_contact_max = 1
    q = model.joint_q.numpy()
    q[: leaves + 1] = np.linspace(-0.31, 0.27, leaves + 1)
    model.joint_q.assign(q)
    model.joint_qd.assign(np.linspace(-0.7, 0.8, model.joint_dof_count, dtype=np.float32))
    # The specialized producer must not assume a unit axis.
    axes = model.joint_axis.numpy()
    axes[0] *= 1.4
    model.joint_axis.assign(axes)
    return model, joints, leaf_bodies


def _solver(model, *, enabled, mode="split"):
    solver = SolverFeatherPGS(
        model,
        pgs_mode=mode,
        pgs_kernel="loop",
        pgs_iterations=8,
        update_mass_matrix_interval=2,
        use_parallel_streams=False,
    )
    if enabled and model.device.is_cuda:
        assert solver._prismatic_publication is not None
    solver._prismatic_publication = (
        PrismaticPublicationPlan.build(model, solver.articulation_joint_end) if enabled else None
    )
    # Exercise the production cached Stage 7 on CPU, where it is normally off.
    solver._fk_id_cache_enabled = True
    # CPU does not normally allocate the optional parallel-refresh terms.
    if model.device.is_cpu:
        solver._body_inertia_terms = wp.empty((model.body_count, 12), dtype=float, device=model.device)
    solver._prepare_augmented_state(model.state(), model.state(), model.control())
    return solver


def _fields(solver, state):
    cache = solver._fk_id_cache
    dynamics = solver if cache is None else cache
    return {
        "body_q": state.body_q,
        "body_qd": state.body_qd,
        "body_q_com": dynamics.body_q_com,
        "origin": dynamics.articulation_origin,
        "S": dynamics.joint_S_s,
        "v": dynamics.body_v_s,
        "a": dynamics.body_a_s,
        "I": dynamics.body_I_s,
        "terms": solver._body_inertia_terms if cache is None else cache.body_inertia_terms,
        "f": dynamics.body_f_s,
        "valid": solver._fk_id_cache_valid,
    }


class TestPrismaticPublication(unittest.TestCase):
    def test_cuda_actual_graph_publication(self):
        """Check the actual matrix-free Stage 7 eager/two-graph outputs and inputs."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("CUDA graph replay requires CUDA")
        for device in devices:
            model, _, _ = _build_model(device, locked_d6=True, leaves=108)
            solvers = [_solver(model, enabled=value, mode="matrix_free") for value in (False, True)]
            inputs = {
                name: getattr(model, name)
                for name in (
                    "joint_X_p",
                    "joint_X_c",
                    "joint_axis",
                    "body_com",
                    "body_mass",
                    "body_inertia",
                    "gravity",
                    "joint_type",
                    "joint_parent",
                    "joint_child",
                    "joint_q_start",
                    "joint_qd_start",
                )
            }
            before = {name: array.numpy().copy() for name, array in inputs.items()}
            for step in (0, 1):
                eager = []
                for solver in solvers:
                    state = model.state()
                    solver._step = step
                    fields = _fields(solver, state)
                    q, qd = state.joint_q.numpy().copy(), state.joint_qd.numpy().copy()

                    def poison(outputs=fields):
                        for name, array in outputs.items():
                            array.fill_(-123 if name == "valid" else -123.25)

                    poison()
                    solver._stage7_update_kinematics(state, solver)
                    expected = {name: array.numpy().copy() for name, array in fields.items()}
                    self.assertTrue(all(np.isfinite(value).all() for value in expected.values()))
                    with wp.ScopedCapture(device=device) as capture:
                        solver._stage7_update_kinematics(state, solver)
                    for _ in range(2):
                        poison()
                        wp.capture_launch(capture.graph)
                        for name, array in fields.items():
                            np.testing.assert_array_equal(array.numpy(), expected[name], err_msg=name)
                    np.testing.assert_array_equal(state.joint_q.numpy(), q)
                    np.testing.assert_array_equal(state.joint_qd.numpy(), qd)
                    eager.append(expected)
                for name in eager[0]:
                    np.testing.assert_allclose(eager[1][name], eager[0][name], rtol=3e-6, atol=3e-6, err_msg=name)
            for name, array in inputs.items():
                np.testing.assert_array_equal(array.numpy(), before[name], err_msg=name)

    def test_exact_topology_and_fallback(self):
        """Admit zero-DOF fixed/D6 roots and reject a non-star without task names."""
        for locked in (False, True):
            model, joints, leaves = _build_model(locked_d6=locked)
            baseline = _solver(model, enabled=False)
            self.assertIsNone(baseline._prismatic_publication)
            plan = PrismaticPublicationPlan.build(model, baseline.articulation_joint_end)
            self.assertEqual(plan.articulation_count, 1)
            self.assertEqual(plan.body_count, len(leaves))
            np.testing.assert_array_equal(plan.body_joint.numpy()[leaves], joints[1:])
            self.assertEqual(int(plan.joint_end.numpy()[0]), joints[0] + 1)
            np.testing.assert_array_equal(plan.joint_end.numpy()[1:], baseline.articulation_joint_end.numpy()[1:])
        model, _, _ = _build_model(chain=True)
        baseline = _solver(model, enabled=False)
        self.assertIsNone(PrismaticPublicationPlan.build(model, baseline.articulation_joint_end))

    def test_complete_canonical_publication_and_held_inertia(self):
        """Match all canonical fields, preserving held I/terms on each cadence arm."""
        model, _, leaves = _build_model(locked_d6=True, leaves=108)
        baseline = _solver(model, enabled=False)
        candidate = _solver(model, enabled=True)
        self.assertIsNotNone(candidate._prismatic_publication)
        for step, compact in ((0, False), (1, False), (1, True)):
            snapshots = []
            for solver in (baseline, candidate):
                state = model.state()
                solver._step = step
                # Only the identity check is consumed by Stage 7, no stream is used.
                solver._global_inertia_stream = object() if compact else None
                for name, array in _fields(solver, state).items():
                    array.fill_(-123 if name == "valid" else -123.25)
                solver._stage7_update_kinematics(state, solver)
                snapshots.append({name: array.numpy().copy() for name, array in _fields(solver, state).items()})
                solver._global_inertia_stream = None
            for name in snapshots[0]:
                np.testing.assert_allclose(snapshots[1][name], snapshots[0][name], rtol=3e-6, atol=3e-6, err_msg=name)
            np.testing.assert_array_equal(snapshots[1]["valid"], np.ones(model.articulation_count))
            np.testing.assert_array_equal(snapshots[1]["a"][leaves], 0.0)
            if step == 0 or compact:
                np.testing.assert_array_equal(snapshots[1]["I"][leaves], -123.25)
            if not compact:
                np.testing.assert_array_equal(snapshots[1]["terms"][leaves], -123.25)
            # Independent public FK computes COM velocities from q/qd, not our cache.
            reference = model.state()
            newton.eval_fk(model, reference.joint_q, reference.joint_qd, reference)
            np.testing.assert_allclose(snapshots[1]["body_q"], reference.body_q.numpy(), rtol=3e-6, atol=3e-6)
            np.testing.assert_allclose(snapshots[1]["body_qd"], reference.body_qd.numpy(), rtol=3e-6, atol=3e-6)

    def test_current_root_frames_and_inertial_notifications(self):
        """Use notified current root/leaf frames, axes, COMs, mass, inertia, and gravity."""
        model, _, _ = _build_model()
        solvers = [_solver(model, enabled=value) for value in (False, True)]
        before = []
        for solver in solvers:
            state = model.state()
            solver._step = 1
            solver._stage7_update_kinematics(state, solver)
            before.append(state.body_q.numpy().copy())
        frames = model.joint_X_p.numpy()
        frames[0, :3] += (0.3, -0.7, 0.5)
        frames[1, :3] += (0.2, 0.4, -0.1)
        model.joint_X_p.assign(frames)
        child_frames = model.joint_X_c.numpy()
        child_frames[1, :3] += (-0.1, 0.07, 0.09)
        model.joint_X_c.assign(child_frames)
        axes = model.joint_axis.numpy()
        axes[0] = (0.4, -0.7, 1.2)
        model.joint_axis.assign(axes)
        com = model.body_com.numpy()
        com += (0.01, 0.02, -0.03)
        model.body_com.assign(com)
        model.body_mass.assign(model.body_mass.numpy() * 1.3)
        model.body_inertia.assign(model.body_inertia.numpy() * 1.2)
        model.gravity.assign(np.array([[1.0, -2.0, -8.0]], dtype=np.float32))
        after = []
        for solver in solvers:
            solver.notify_model_changed(newton.ModelFlags.ALL)
            np.testing.assert_array_equal(solver._fk_id_cache_valid.numpy(), 0)
            state = model.state()
            solver._stage7_update_kinematics(state, solver)
            after.append(
                {name: array.numpy().copy() for name, array in _fields(solver, state).items() if name != "terms"}
            )
            self.assertIs(solver._fk_id_cache_source_state, state)
            solver.reset(state)
            np.testing.assert_array_equal(solver._fk_id_cache_valid.numpy(), 0)
        self.assertFalse(np.array_equal(before[1], after[1]["body_q"]))
        for name in after[0]:
            np.testing.assert_allclose(after[1][name], after[0][name], rtol=3e-6, atol=3e-6, err_msg=name)

    def test_cached_next_step_dynamics_and_reset(self):
        """Feed publication back into original dynamics across refresh/reuse and reset."""
        self._check_cached_next_step_dynamics_and_reset("cpu", "split")

    def test_cuda_cached_next_step_dynamics_and_reset(self):
        """Preserve complete CUDA steps with matrix-free and snapshot-backed publication."""
        devices = wp.get_cuda_devices()
        if not devices:
            self.skipTest("CUDA full-step coverage requires CUDA")
        for device in devices:
            for mode in ("matrix_free", "split"):
                with self.subTest(device=str(device), mode=mode):
                    self._check_cached_next_step_dynamics_and_reset(device, mode)

    def _check_cached_next_step_dynamics_and_reset(self, device, mode):
        model, _, _ = _build_model(device, leaves=4)
        solvers = [_solver(model, enabled=value, mode=mode) for value in (False, True)]
        states = [[model.state(), model.state()] for _ in solvers]
        controls = [model.control() for _ in solvers]
        contacts = [model.contacts() for _ in solvers]
        for pair in states:
            newton.eval_fk(model, pair[0].joint_q, pair[0].joint_qd, pair[0])
        for step in range(6):
            for index, solver in enumerate(solvers):
                current, following = states[index]
                if step == 3:
                    solver.reset(current)
                forces = np.zeros((model.body_count, 6), dtype=np.float32)
                forces[1, :3] = (0.2 * step, -0.1, 0.3)
                current.body_f.assign(forces)
                controls[index].joint_f.assign(np.linspace(-0.1, 0.2, model.joint_dof_count, dtype=np.float32))
                solver.step(current, following, controls[index], contacts[index], 1.0 / 240.0)
                states[index] = [following, current]
            for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
                np.testing.assert_allclose(
                    getattr(states[1][0], name).numpy(),
                    getattr(states[0][0], name).numpy(),
                    rtol=1e-5,
                    atol=2e-6,
                    err_msg=f"step {step} {name}",
                )
            np.testing.assert_allclose(solvers[1].v_hat.numpy(), solvers[0].v_hat.numpy(), rtol=1e-5, atol=2e-6)
            for size in solvers[0].L_by_size:
                np.testing.assert_allclose(
                    solvers[1].L_by_size[size].numpy(),
                    solvers[0].L_by_size[size].numpy(),
                    rtol=1e-5,
                    atol=2e-6,
                )


if __name__ == "__main__":
    unittest.main()
