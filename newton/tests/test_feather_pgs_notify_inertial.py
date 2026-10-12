# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import gc
import unittest
import weakref

import numpy as np
import warp as wp

import newton
from newton import ModelFlags
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

DT = 1.0 / 60.0
INITIAL_JOINT_Q = 0.3
NEW_COM = (0.15, 0.0, 0.05)


def _build_model(device, com=None):
    """Single-link pendulum on a Y-axis revolute joint, box COM at the origin.

    With the COM at the body origin (the pivot) gravity exerts no torque, so
    any post-randomization swing is attributable to the COM offset alone.
    """
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.density = 1000.0

    link = builder.add_link()
    builder.add_shape_box(link, hx=0.25, hy=0.05, hz=0.05)
    joint = builder.add_joint_revolute(
        -1,
        link,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.8), wp.quat_identity()),
        axis=newton.Axis.Y,
    )
    builder.add_articulation([joint])
    builder.joint_q[0] = INITIAL_JOINT_Q
    model = builder.finalize(device=device)
    if com is not None:
        body_com = model.body_com.numpy()
        body_com[0] = com
        model.body_com.assign(body_com)
    return model


def _run_trajectory(model, solver, num_steps):
    state_0 = model.state()
    state_1 = model.state()
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()
    control = model.control()
    joint_q_history = []
    for _ in range(num_steps):
        collision_pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0
        joint_q_history.append(state_0.joint_q.numpy().copy())
    return np.stack(joint_q_history)


class TestFeatherPGSNotifyInertial(unittest.TestCase):
    def test_stepped_solver_releases_resources_without_cyclic_gc(self):
        """Release a stepped solver before cyclic GC can finalize its streams first."""
        gc_enabled = gc.isenabled()
        gc.disable()
        try:
            model = _build_model(wp.get_device())
            solver = SolverFeatherPGS(model)
            reference = weakref.ref(solver)
            solver.step(model.state(), model.state(), model.control(), None, DT)
            del solver
            self.assertIsNone(reference(), "Stepping must not create a solver ownership cycle")
        finally:
            if gc_enabled:
                gc.enable()

    def test_step_refreshes_body_pose_after_generalized_coordinate_update(self):
        """A direct ``joint_q`` update must not require a caller-side FK pass.

        ``SolverFeatherPGS.step`` historically derives body poses from the
        generalized coordinates before inverse dynamics.  Reset and direct
        generalized-coordinate callers rely on that public step behavior.
        """
        device = wp.get_device()
        outputs = {}

        for caller_refreshes_fk in (False, True):
            model = _build_model(device, com=NEW_COM)
            state_0 = model.state()
            state_1 = model.state()

            joint_q = state_0.joint_q.numpy()
            joint_q[0] = 1.1
            state_0.joint_q.assign(joint_q)
            stale_body_q = state_0.body_q.numpy().copy()

            if caller_refreshes_fk:
                newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

            SolverFeatherPGS(model, pgs_mode="split").step(
                state_0,
                state_1,
                model.control(),
                None,
                DT,
            )
            outputs[caller_refreshes_fk] = {
                "joint_q": state_1.joint_q.numpy().copy(),
                "joint_qd": state_1.joint_qd.numpy().copy(),
                "body_q": state_0.body_q.numpy().copy(),
                "stale_body_q": stale_body_q,
            }

        self.assertFalse(np.allclose(outputs[False]["stale_body_q"], outputs[True]["body_q"]))
        np.testing.assert_allclose(outputs[False]["body_q"], outputs[True]["body_q"], rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(outputs[False]["joint_q"], outputs[True]["joint_q"], rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(outputs[False]["joint_qd"], outputs[True]["joint_qd"], rtol=0.0, atol=1.0e-6)

    def test_notify_refreshes_baked_com_and_inertia_buffers(self):
        """Verify BODY_INERTIAL_PROPERTIES re-derives body_X_com and body_I_m.

        Both buffers are baked from the model at construction time; writing
        model.body_com/body_mass alone must not change them until the solver
        is notified.
        """
        device = wp.get_device()
        model = _build_model(device)
        solver = SolverFeatherPGS(model)
        stale_X_com = solver.body_X_com.numpy().copy()
        stale_I_m = solver.body_I_m.numpy().copy()

        body_com = model.body_com.numpy()
        body_com[0] = NEW_COM
        model.body_com.assign(body_com)
        body_mass = model.body_mass.numpy()
        body_mass[0] *= 2.0
        model.body_mass.assign(body_mass)

        np.testing.assert_array_equal(solver.body_X_com.numpy(), stale_X_com)
        solver.notify_model_changed(ModelFlags.BODY_INERTIAL_PROPERTIES)

        np.testing.assert_allclose(
            solver.body_X_com.numpy()[0][:3], np.asarray(NEW_COM, dtype=np.float32), rtol=0.0, atol=0.0
        )
        self.assertFalse(np.allclose(solver.body_I_m.numpy(), stale_I_m))
        self.assertEqual(solver._mass_update_requested.numpy()[0], 1)

    def test_com_change_with_notify_matches_freshly_built_solver(self):
        """Verify a mid-run COM change plus notify reproduces baked-COM dynamics.

        The trajectory after the change must match a solver constructed with
        the new COM already in the model and stepped from the same state; a
        stale solver (COM written without notify) must diverge, proving the
        comparison is not vacuous.
        """
        device = wp.get_device()
        pre_steps, post_steps = 30, 60

        # Reference: solver built with the new COM, run from the pre-change
        # state (which is static: zero torque while the COM sits at the pivot).
        reference_model = _build_model(device, com=NEW_COM)
        reference_solver = SolverFeatherPGS(reference_model)
        reference = _run_trajectory(reference_model, reference_solver, post_steps)

        histories = {}
        for notify in (True, False):
            model = _build_model(device)
            solver = SolverFeatherPGS(model)
            _run_trajectory(model, solver, pre_steps)
            body_com = model.body_com.numpy()
            body_com[0] = NEW_COM
            model.body_com.assign(body_com)
            if notify:
                solver.notify_model_changed(ModelFlags.BODY_INERTIAL_PROPERTIES)
            histories[notify] = _run_trajectory(model, solver, post_steps)

        np.testing.assert_allclose(histories[True], reference, rtol=0.0, atol=1.0e-5)
        stale_drift = np.abs(histories[False] - reference).max()
        self.assertGreater(stale_drift, 1.0e-2, "stale-solver trajectory should diverge without notify")

    def test_joint_frame_change_with_notify_matches_freshly_built_solver(self):
        """Refresh cached mass factors on JOINT_PROPERTIES, eagerly and under graph replay."""
        device = wp.get_device()
        interval = 100
        modes = ["split", "matrix_free"] if device.is_cuda else ["split"]

        def shift_child_frame(model):
            joint_X_c = model.joint_X_c.numpy()
            joint_X_c[0, 0] = 0.5
            model.joint_X_c.assign(joint_X_c)

        for mode in modes:
            reference_model = _build_model(device)
            shift_child_frame(reference_model)
            reference_solver = SolverFeatherPGS(reference_model, pgs_mode=mode, update_mass_matrix_interval=interval)
            reference = _run_trajectory(reference_model, reference_solver, 5)
            for capture in [False, True] if device.is_cuda else [False]:
                with self.subTest(mode=mode, capture=capture):
                    model = _build_model(device)
                    solver = SolverFeatherPGS(model, pgs_mode=mode, update_mass_matrix_interval=interval)
                    state_0, state_1 = model.state(), model.state()
                    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
                    control = model.control()

                    def substep(solver=solver, state_0=state_0, state_1=state_1, control=control):
                        solver.step(state_0, state_1, control, None, DT)
                        wp.copy(state_0.joint_q, state_1.joint_q)
                        wp.copy(state_0.joint_qd, state_1.joint_qd)

                    substep()
                    initial_q = model.joint_q.numpy().copy()
                    graph = None
                    if capture:
                        with wp.ScopedCapture(device=device) as graph:
                            substep()
                    state_0.joint_q.assign(initial_q)
                    state_0.joint_qd.zero_()
                    shift_child_frame(model)
                    solver.notify_model_changed(ModelFlags.JOINT_PROPERTIES)
                    history = []
                    for _ in range(5):
                        if graph is None:
                            substep()
                        else:
                            wp.capture_launch(graph.graph)
                        history.append(state_0.joint_q.numpy().copy())
                    np.testing.assert_allclose(np.asarray(history), reference, rtol=0.0, atol=1.0e-5)


def _check_joint_dof_edit_matches_fresh_solver(test, device, name: str, value: float, flag):
    """Assign one joint DOF property after a step, notify ``flag``, and compare the next step to a fresh solver."""
    pgs_mode = "matrix_free" if wp.get_device(device).is_cuda else "split"
    # Graph capture needs a CUDA device.
    for capture in (False, True) if wp.get_device(device).is_cuda else (False,):
        with test.subTest(capture=capture):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            link = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
            joint = builder.add_joint_prismatic(-1, link, axis=newton.Axis.X, target_kd=10.0, armature=0.0)
            builder.add_articulation([joint])
            model = builder.finalize(device=device)
            solver = SolverFeatherPGS(model, pgs_mode=pgs_mode, update_mass_matrix_interval=100)
            state_0, state_1 = model.state(), model.state()
            state_0.joint_qd.fill_(1.0)
            control = model.control()
            solver.step(state_0, state_1, control, None, 0.01)
            if capture:
                with wp.ScopedCapture(device=device) as graph:
                    solver.step(state_0, state_1, control, None, 0.01)
            stale_qd = state_1.joint_qd.numpy().copy()
            getattr(model, name).assign([value])
            solver.notify_model_changed(flag)
            if capture:
                wp.capture_launch(graph.graph)
            else:
                solver.step(state_0, state_1, control, None, 0.01)

            fresh = SolverFeatherPGS(model, pgs_mode=pgs_mode, update_mass_matrix_interval=100)
            fresh_out = model.state()
            fresh.step(state_0, fresh_out, control, None, 0.01)
            test.assertFalse(np.allclose(fresh_out.joint_qd.numpy(), stale_qd, atol=1.0e-4))
            np.testing.assert_allclose(state_1.joint_qd.numpy(), fresh_out.joint_qd.numpy(), atol=1.0e-6)


def test_dof_force_flag_refreshes_drive_gains(test, device):
    """Fold a drive damping edit into the mass matrix on a JOINT_DOF_FORCE_PROPERTIES notify."""
    _check_joint_dof_edit_matches_fresh_solver(
        test, device, "joint_target_kd", 40.0, ModelFlags.JOINT_DOF_FORCE_PROPERTIES
    )


def test_dof_force_flag_refreshes_joint_damping(test, device):
    """Read joint damping at the next JOINT_DOF_FORCE_PROPERTIES notify."""
    _check_joint_dof_edit_matches_fresh_solver(
        test, device, "joint_damping", 20.0, ModelFlags.JOINT_DOF_FORCE_PROPERTIES
    )


def test_dof_inertial_flag_refreshes_armature(test, device):
    """Fold an armature edit into the mass matrix on a JOINT_DOF_INERTIAL_PROPERTIES notify."""
    _check_joint_dof_edit_matches_fresh_solver(
        test, device, "joint_armature", 1.0, ModelFlags.JOINT_DOF_INERTIAL_PROPERTIES
    )


def _check_damping_edit_matches_fresh_solver(test, device, replace: bool, flag, pgs_mode: str):
    """Edit joint damping after a step, notify, and compare the next eager or captured step to a fresh solver."""
    # Graph capture needs a CUDA device.
    for capture in (False, True) if wp.get_device(device).is_cuda else (False,):
        with test.subTest(capture=capture):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            link = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
            joint = builder.add_joint_prismatic(-1, link, axis=newton.Axis.X, damping=1.0, armature=0.0)
            builder.add_articulation([joint])
            model = builder.finalize(device=device)
            solver = SolverFeatherPGS(model, pgs_mode=pgs_mode, update_mass_matrix_interval=100)
            state_0, state_1 = model.state(), model.state()
            state_0.joint_qd.fill_(1.0)
            control = model.control()
            solver.step(state_0, state_1, control, None, 0.01)
            if capture:
                with wp.ScopedCapture(device=device) as graph:
                    solver.step(state_0, state_1, control, None, 0.01)
            stale_qd = state_1.joint_qd.numpy().copy()
            if replace:
                model.joint_damping = wp.array([3.0], dtype=wp.float32, device=device)
            else:
                model.joint_damping.assign([3.0])
            solver.notify_model_changed(flag)
            if capture:
                wp.capture_launch(graph.graph)
            else:
                solver.step(state_0, state_1, control, None, 0.01)

            fresh = SolverFeatherPGS(model, pgs_mode=pgs_mode, update_mass_matrix_interval=100)
            fresh_out = model.state()
            fresh.step(state_0, fresh_out, control, None, 0.01)
            test.assertFalse(np.allclose(fresh_out.joint_qd.numpy(), stale_qd, atol=1.0e-4))
            np.testing.assert_allclose(state_1.joint_qd.numpy(), fresh_out.joint_qd.numpy(), atol=1.0e-6)


def _check_friction_edit_matches_fresh_solver(test, device, replace: bool, pgs_mode: str):
    """Edit shape friction after a step, notify, and compare the next eager or captured step to a fresh solver."""
    # Graph capture needs a CUDA device.
    for capture in (False, True) if wp.get_device(device).is_cuda else (False,):
        with test.subTest(capture=capture):
            builder = newton.ModelBuilder()
            builder.default_shape_cfg.mu = 0.0
            builder.add_ground_plane()
            body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
            builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
            model = builder.finalize(device=device)
            solver = SolverFeatherPGS(model, pgs_mode=pgs_mode)
            state_0, state_1 = model.state(), model.state()
            state_0.joint_qd.assign([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
            pipeline = newton.CollisionPipeline(model)
            contacts = pipeline.contacts()
            pipeline.collide(state_0, contacts)
            control = model.control()
            solver.step(state_0, state_1, control, contacts, DT)
            if capture:
                with wp.ScopedCapture(device=device) as graph:
                    solver.step(state_0, state_1, control, contacts, DT)
            mu = np.full(model.shape_count, 1.0, dtype=np.float32)
            if replace:
                model.shape_material_mu = wp.array(mu, dtype=wp.float32, device=device)
            else:
                model.shape_material_mu.assign(mu)
            solver.notify_model_changed(ModelFlags.SHAPE_PROPERTIES)
            if capture:
                wp.capture_launch(graph.graph)
            else:
                solver.step(state_0, state_1, control, contacts, DT)

            fresh = SolverFeatherPGS(model, pgs_mode=pgs_mode)
            fresh_out = model.state()
            fresh.step(state_0, fresh_out, control, contacts, DT)
            test.assertLess(float(fresh_out.joint_qd.numpy()[0]), 0.95)
            np.testing.assert_allclose(state_1.joint_qd.numpy(), fresh_out.joint_qd.numpy(), atol=1.0e-5)


def test_replaced_joint_damping_matches_fresh_solver(test, device, pgs_mode="matrix_free"):
    """Read a joint damping array replaced on the model at the next JOINT_DOF_PROPERTIES notify."""
    _check_damping_edit_matches_fresh_solver(test, device, True, ModelFlags.JOINT_DOF_PROPERTIES, pgs_mode)


def test_assigned_joint_damping_matches_fresh_solver(test, device, pgs_mode="matrix_free"):
    """Read joint damping modified in place at the next JOINT_DOF_PROPERTIES notify."""
    _check_damping_edit_matches_fresh_solver(test, device, False, ModelFlags.JOINT_DOF_PROPERTIES, pgs_mode)


def test_dof_force_flag_reads_replaced_joint_damping(test, device, pgs_mode="matrix_free"):
    """Read a joint damping array replaced on the model at the next JOINT_DOF_FORCE_PROPERTIES notify."""
    _check_damping_edit_matches_fresh_solver(test, device, True, ModelFlags.JOINT_DOF_FORCE_PROPERTIES, pgs_mode)


def test_replaced_shape_friction_matches_fresh_solver(test, device, pgs_mode="matrix_free"):
    """Read a friction array replaced on the model at the next SHAPE_PROPERTIES notify."""
    _check_friction_edit_matches_fresh_solver(test, device, replace=True, pgs_mode=pgs_mode)


def test_assigned_shape_friction_matches_fresh_solver(test, device, pgs_mode="matrix_free"):
    """Read friction modified in place at the next SHAPE_PROPERTIES notify."""
    _check_friction_edit_matches_fresh_solver(test, device, replace=False, pgs_mode=pgs_mode)


class TestFeatherPGSNarrowJointDofFlags(unittest.TestCase):
    pass


for _name in (
    "test_dof_force_flag_refreshes_drive_gains",
    "test_dof_force_flag_refreshes_joint_damping",
    "test_dof_inertial_flag_refreshes_armature",
):
    add_function_test(TestFeatherPGSNarrowJointDofFlags, _name, globals()[_name], devices=get_test_devices())


class TestFeatherPGSNotifyReplacedArrays(unittest.TestCase):
    pass


for _name in (
    "test_replaced_joint_damping_matches_fresh_solver",
    "test_assigned_joint_damping_matches_fresh_solver",
    "test_dof_force_flag_reads_replaced_joint_damping",
    "test_replaced_shape_friction_matches_fresh_solver",
    "test_assigned_shape_friction_matches_fresh_solver",
):
    add_function_test(TestFeatherPGSNotifyReplacedArrays, _name, globals()[_name], devices=get_cuda_test_devices())
    add_function_test(
        TestFeatherPGSNotifyReplacedArrays,
        f"{_name}_split",
        globals()[_name],
        devices=get_test_devices(),
        pgs_mode="split",
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
