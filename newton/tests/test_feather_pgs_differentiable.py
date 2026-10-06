# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_test_devices

_DT = 1.0 / 120.0


def _build_chain(device, *, drives=False, gravity=-9.81, requires_grad=True):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, gravity))
    parent = -1
    joints = []
    for i in range(3):
        body = builder.add_link(
            mass=1.0 + 0.3 * i,
            inertia=wp.mat33(np.diag([0.02 + 0.01 * i, 0.03, 0.015 + 0.005 * i])),
            com=wp.vec3(0.25, 0.0, 0.0),
        )
        joints.append(
            builder.add_joint_revolute(
                parent,
                body,
                parent_xform=wp.transform(wp.vec3(0.5 if parent >= 0 else 0.0, 0.0, 0.0), wp.quat_identity()),
                axis=(0.0, 1.0, 0.0) if i % 2 == 0 else (0.0, 0.3, 0.95),
                target_ke=20.0 if drives else 0.0,
                target_kd=1.0 if drives else 0.0,
                armature=0.01,
            )
        )
        parent = body
    builder.add_articulation(joints)
    return builder.finalize(device=device, requires_grad=requires_grad)


def _build_floating_chain(device, requires_grad=True):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    base = builder.add_link(mass=3.0, inertia=wp.mat33(np.diag([0.1, 0.2, 0.15])))
    joints = [builder.add_joint_free(parent=-1, child=base)]
    parent = base
    for axis in ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)):
        body = builder.add_link(mass=0.7, inertia=wp.mat33(np.diag([0.01, 0.02, 0.015])), com=wp.vec3(0.2, 0.0, 0.0))
        joints.append(
            builder.add_joint_revolute(
                parent,
                body,
                parent_xform=wp.transform(wp.vec3(0.3, 0.0, 0.0), wp.quat_identity()),
                axis=axis,
                armature=0.01,
            )
        )
        parent = body
    builder.add_articulation(joints)
    return builder.finalize(device=device, requires_grad=requires_grad)


def _build_free_body(device, requires_grad=True):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    body = builder.add_link(mass=2.0, inertia=wp.mat33(np.diag([0.08, 0.12, 0.16])))
    builder.add_articulation([builder.add_joint_free(parent=-1, child=body)])
    return builder.finalize(device=device, requires_grad=requires_grad)


_MODELS = {
    "chain_drives": lambda device, requires_grad=True: _build_chain(device, drives=True, requires_grad=requires_grad),
    "floating_chain": _build_floating_chain,
    "free_body": _build_free_body,
}


def _initial_state(model, seed):
    rng = np.random.default_rng(seed)
    q = model.joint_q.numpy().copy()
    joint_type = model.joint_type.numpy()
    q_start = model.joint_q_start.numpy()
    for joint, kind in enumerate(joint_type):
        if kind == newton.JointType.REVOLUTE:
            q[q_start[joint]] += 0.3 * rng.normal()
        elif kind == newton.JointType.FREE:
            q[q_start[joint] : q_start[joint] + 3] += 0.3 * rng.normal(size=3)
            quat = rng.normal(size=4)
            q[q_start[joint] + 3 : q_start[joint] + 7] = quat / np.linalg.norm(quat)
    qd = rng.normal(size=model.joint_dof_count)
    return q.astype(np.float32), qd.astype(np.float32)


def _rollout(model, solver, inputs, steps, *, rotate=False, tape=None):
    """Run ``steps`` steps from ``inputs`` and return the final state and the step inputs."""
    state_count = 2 if rotate else steps + 1
    states = [model.state(requires_grad=model.requires_grad) for _ in range(state_count)]
    states[0].joint_q.assign(inputs["joint_q"])
    states[0].joint_qd.assign(inputs["joint_qd"])
    newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
    control = model.control()
    control.joint_f.assign(inputs["joint_f"])
    control.joint_target_q.assign(inputs["joint_target_q"])
    if tape is not None:
        tape.__enter__()
    try:
        for k in range(steps):
            if rotate:
                solver.step(states[0], states[1], control, None, _DT)
                states[0], states[1] = states[1], states[0]
            else:
                solver.step(states[k], states[k + 1], control, None, _DT)
    finally:
        if tape is not None:
            tape.__exit__(None, None, None)
    return states[0] if rotate else states[-1], states[0], control


def _random_inputs(model, seed):
    q, qd = _initial_state(model, seed)
    rng = np.random.default_rng(seed + 1)
    return {
        "joint_q": q,
        "joint_qd": qd,
        "joint_f": rng.normal(size=model.joint_dof_count).astype(np.float32),
        "joint_target_q": (0.3 * rng.normal(size=model.control().joint_target_q.shape[0])).astype(np.float32),
    }


@wp.kernel
def _weighted_loss(
    q: wp.array[float], wq: wp.array[float], qd: wp.array[float], wqd: wp.array[float], loss: wp.array[float]
):
    i = wp.tid()
    if i < q.shape[0]:
        wp.atomic_add(loss, 0, q[i] * wq[i])
    if i < qd.shape[0]:
        wp.atomic_add(loss, 0, qd[i] * wqd[i])


def test_rejects_unsupported_configs(test, device):
    model = _build_chain(device)
    for kwargs in (
        {"enable_sleeping": True},
        {"pgs_warmstart": True},
        {"friction_anchor_beta": 0.2},
        {"contact_torsion_radius": 0.01},
        {"articulated_contact_response": "propagation"},
        {"articulated_contact_response": "propagation-colored"},
        {"update_mass_matrix_interval": 2},
        {"enable_joint_limits": True},
        {"enable_joint_velocity_limits": True},
        {"drive_mode": "physx_pgs"},
        {"pgs_velocity_iterations": 2},
        {"pgs_debug": True},
    ):
        with test.subTest(**kwargs), test.assertRaisesRegex(ValueError, "differentiable=True requires"):
            SolverFeatherPGS(model, differentiable=True, **kwargs)
    with test.assertRaisesRegex(ValueError, "requires_grad=True"):
        SolverFeatherPGS(_build_chain(device, requires_grad=False), differentiable=True)

    solver = SolverFeatherPGS(model, differentiable=True)
    state = model.state(requires_grad=True)
    with test.assertRaisesRegex(ValueError, "distinct input and output states"):
        solver.step(state, state, None, None, _DT)
    with test.assertRaisesRegex(NotImplementedError, "contacts"):
        contacts = newton.CollisionPipeline(model, rigid_contact_max=8).contacts()
        solver.step(state, model.state(requires_grad=True), None, contacts, _DT)


def test_forward_matches_default(test, device):
    steps = 30
    for name, build in _MODELS.items():
        inputs = _random_inputs(build(device), seed=0)
        for requires_grad in (True, False):
            model = build(device, requires_grad=requires_grad)
            reference, _, _ = _rollout(
                model, SolverFeatherPGS(model, friction_anchor_beta=0.0), inputs, steps, rotate=True
            )
            model = build(device)
            result, _, _ = _rollout(model, SolverFeatherPGS(model, differentiable=True), inputs, steps)
            for attribute in ("joint_q", "joint_qd", "body_q", "body_qd"):
                expected = getattr(reference, attribute).numpy()
                actual = getattr(result, attribute).numpy()
                with test.subTest(model=name, requires_grad=requires_grad, attribute=attribute):
                    if wp.get_device(device).is_cpu:
                        # Identical kernels and operation order on the serial CPU path.
                        np.testing.assert_array_equal(actual, expected)
                    else:
                        # CUDA defaults select tiled/fused kernels that round differently.
                        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-4)


def test_gradient_matches_finite_difference(test, device):
    steps = 3
    eps = 1.0e-2
    for name, build in _MODELS.items():
        model = build(device)
        solver = SolverFeatherPGS(model, differentiable=True)
        inputs = _random_inputs(model, seed=1)
        rng = np.random.default_rng(2)
        wq = wp.array(rng.normal(size=model.joint_coord_count).astype(np.float32), device=device)
        wqd = wp.array(rng.normal(size=model.joint_dof_count).astype(np.float32), device=device)

        def loss_of(values, tape=None, solver=solver, model=model, wq=wq, wqd=wqd):
            final, initial, control = _rollout(model, solver, values, steps, tape=tape)
            loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
            if tape is not None:
                tape.__enter__()
            wp.launch(
                _weighted_loss,
                dim=max(model.joint_coord_count, model.joint_dof_count),
                inputs=[final.joint_q, wq, final.joint_qd, wqd],
                outputs=[loss],
                device=device,
            )
            if tape is not None:
                tape.__exit__(None, None, None)
            return loss, initial, control

        tape = wp.Tape()
        loss, initial, control = loss_of(inputs, tape)
        tape.backward(loss)
        gradients = {
            "joint_q": initial.joint_q.grad.numpy(),
            "joint_qd": initial.joint_qd.grad.numpy(),
            "joint_f": control.joint_f.grad.numpy(),
            "joint_target_q": control.joint_target_q.grad.numpy(),
        }
        joint_type = model.joint_type.numpy()
        q_start = model.joint_q_start.numpy()
        for key, gradient in gradients.items():
            if key == "joint_target_q" and name != "chain_drives":
                continue
            direction = rng.normal(size=gradient.shape[0]).astype(np.float32)
            if key == "joint_q":
                for joint in np.flatnonzero(joint_type == newton.JointType.FREE):
                    direction[q_start[joint] + 3 : q_start[joint] + 7] = 0.0
            direction /= np.linalg.norm(direction)
            plus = dict(inputs, **{key: inputs[key] + eps * direction})
            minus = dict(inputs, **{key: inputs[key] - eps * direction})
            fd = (loss_of(plus)[0].numpy()[0] - loss_of(minus)[0].numpy()[0]) / (2.0 * eps)
            analytic = float(gradient @ direction)
            with test.subTest(model=name, input=key):
                test.assertAlmostEqual(analytic, fd, delta=2.0e-3 * max(1.0, abs(fd)))


def test_gradient_matches_analytic_pendulum(test, device):
    # Without gravity a single hinge about its COM has constant generalized inertia.
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(mass=1.0, inertia=wp.mat33(np.diag([0.02, 0.03, 0.04])))
    builder.add_articulation([builder.add_joint_revolute(-1, body, axis=(0.0, 1.0, 0.0), armature=0.01)])
    model = builder.finalize(device=device, requires_grad=True)
    solver = SolverFeatherPGS(model, differentiable=True, angular_damping=0.0)
    steps = 10
    inputs = {"joint_q": np.zeros(1, np.float32), "joint_qd": np.zeros(1, np.float32)}
    inputs.update(joint_f=np.array([0.5], np.float32), joint_target_q=np.zeros(1, np.float32))
    tape = wp.Tape()
    final, initial, control = _rollout(model, solver, inputs, steps, tape=tape)
    tape.backward(grads={final.joint_q: wp.ones(1, dtype=float, device=device)})
    # Semi-implicit Euler: q_T = q_0 + T dt qd_0 + dt^2 T(T+1)/2 f / I.
    inertia = 0.03 + 0.01
    np.testing.assert_allclose(initial.joint_qd.grad.numpy(), [steps * _DT], rtol=1.0e-5)
    np.testing.assert_allclose(control.joint_f.grad.numpy(), [_DT**2 * steps * (steps + 1) / 2 / inertia], rtol=1.0e-5)
    np.testing.assert_allclose(initial.joint_q.grad.numpy(), [1.0], rtol=1.0e-6)


def test_repeated_backward_is_stable(test, device):
    model = _build_floating_chain(device)
    solver = SolverFeatherPGS(model, differentiable=True)
    inputs = _random_inputs(model, seed=3)
    tape = wp.Tape()
    final, initial, _ = _rollout(model, solver, inputs, 5, tape=tape)
    seed_grads = {final.joint_qd: wp.ones(model.joint_dof_count, dtype=float, device=device)}
    tape.backward(grads=seed_grads)
    first = initial.joint_q.grad.numpy().copy()
    tape.zero()
    tape.backward(grads=seed_grads)
    np.testing.assert_array_equal(initial.joint_q.grad.numpy(), first)
    test.assertTrue(np.all(np.isfinite(first)))


class TestFeatherPGSDifferentiable(unittest.TestCase):
    pass


for _device in get_test_devices():
    for _test in (
        test_rejects_unsupported_configs,
        test_forward_matches_default,
        test_gradient_matches_finite_difference,
        test_gradient_matches_analytic_pendulum,
        test_repeated_backward_is_stable,
    ):
        add_function_test(TestFeatherPGSDifferentiable, _test.__name__, _test, devices=[_device])


if __name__ == "__main__":
    unittest.main()
