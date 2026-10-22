# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import contextlib
import math
import unittest
import warnings

import numpy as np
import warp as wp

import newton
from newton._src.solvers.steel._testing import kernel_overrides
from newton._src.solvers.steel.differentiable import (
    _dense_pgs_sweep,
    _friction_pair,
    _smooth_contact_impulse,
    _taped_compute_world_contact_bias,
)
from newton._src.solvers.steel.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_TORSION,
)
from newton.solvers import SolverSteel, steel
from newton.tests.steel._fixtures import set_torsion_radius
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


def _build_free_body(device, requires_grad=True, anchor=None):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    body = builder.add_link(mass=2.0, inertia=wp.mat33(np.diag([0.08, 0.12, 0.16])))
    parent_xform = None if anchor is None else wp.transform(wp.vec3(0.1, -0.2, 0.3), anchor)
    builder.add_articulation([builder.add_joint_free(parent=-1, child=body, parent_xform=parent_xform)])
    return builder.finalize(device=device, requires_grad=requires_grad)


def _build_ball_chain(device, requires_grad=True):
    """A damped ball joint carrying a hinged link: exercises the ball joint's passive damping."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    upper = builder.add_link(mass=1.2, inertia=wp.mat33(np.diag([0.03, 0.02, 0.025])), com=wp.vec3(0.0, 0.0, -0.25))
    lower = builder.add_link(mass=0.6, inertia=wp.mat33(np.diag([0.01, 0.015, 0.012])), com=wp.vec3(0.2, 0.0, 0.0))
    joints = [
        builder.add_joint_ball(-1, upper, armature=0.01, damping=0.4),
        builder.add_joint_revolute(
            upper,
            lower,
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, -0.5), wp.quat_identity()),
            axis=(0.0, 1.0, 0.0),
            armature=0.01,
        ),
    ]
    builder.add_articulation(joints)
    return builder.finalize(device=device, requires_grad=requires_grad)


_MODELS = {
    "chain_drives": lambda device, requires_grad=True: _build_chain(device, drives=True, requires_grad=requires_grad),
    "floating_chain": _build_floating_chain,
    "free_body": _build_free_body,
    # A rotated FREE anchor: the gyroscopic update runs in the anchor frame of the joint velocity.
    "free_body_anchor": lambda device, requires_grad=True: _build_free_body(
        device, requires_grad, wp.quat_from_axis_angle(wp.normalize(wp.vec3(1.0, 2.0, 0.5)), 0.7)
    ),
    "ball_chain": _build_ball_chain,
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
    control.joint_target_qd.assign(inputs["joint_target_qd"])
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
        "joint_target_qd": rng.normal(size=model.joint_dof_count).astype(np.float32),
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
        {"contact_warmstart": True},
        {"enable_torsional_rolling_friction": True},
        {"articulated_contact_response": "propagation"},
        {"articulated_contact_response": "propagation_colored"},
        {"update_mass_matrix_interval": 2},
        {"enable_joint_friction": True},
        {"contact_compliance": True},
        {"drive_mode": "rows"},
        {"friction_mode": "bisection"},
        {"velocity_iterations": 2},
    ):
        with test.subTest(**kwargs), test.assertRaisesRegex(ValueError, "differentiable=True requires"):
            SolverSteel(model, differentiable=True, **kwargs)
    with test.assertRaisesRegex(ValueError, "requires_grad=True"):
        SolverSteel(_build_chain(device, requires_grad=False), differentiable=True)

    solver = SolverSteel(model, differentiable=True)
    state = model.state(requires_grad=True)
    with test.assertRaisesRegex(ValueError, "distinct input and output states"):
        solver.step(state, state, None, None, _DT)


def test_forward_matches_default(test, device):
    steps = 30
    for name, build in _MODELS.items():
        inputs = _random_inputs(build(device), seed=0)
        model = build(device, requires_grad=False)
        reference, _, _ = _rollout(model, SolverSteel(model, friction_anchor_beta=0.0), inputs, steps, rotate=True)
        model = build(device)
        result, _, _ = _rollout(model, SolverSteel(model, differentiable=True), inputs, steps)
        for attribute in ("joint_q", "joint_qd", "body_q", "body_qd"):
            expected = getattr(reference, attribute).numpy()
            actual = getattr(result, attribute).numpy()
            with test.subTest(model=name, attribute=attribute):
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
        solver = SolverSteel(model, differentiable=True)
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
            "joint_target_qd": control.joint_target_qd.grad.numpy(),
        }
        joint_type = model.joint_type.numpy()
        q_start = model.joint_q_start.numpy()
        for key, gradient in gradients.items():
            if key.startswith("joint_target") and name != "chain_drives":
                continue
            direction = rng.normal(size=gradient.shape[0]).astype(np.float32)
            if key == "joint_q":
                for joint in np.flatnonzero(joint_type == newton.JointType.FREE):
                    direction[q_start[joint] + 3 : q_start[joint] + 7] = 0.0
                for joint in np.flatnonzero(joint_type == newton.JointType.BALL):
                    direction[q_start[joint] : q_start[joint] + 4] = 0.0
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
    solver = SolverSteel(model, differentiable=True)
    solver._rigid_body_angular_damping.zero_()
    steps = 10
    inputs = {"joint_q": np.zeros(1, np.float32), "joint_qd": np.zeros(1, np.float32)}
    inputs.update(
        joint_f=np.array([0.5], np.float32),
        joint_target_q=np.zeros(1, np.float32),
        joint_target_qd=np.zeros(1, np.float32),
    )
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
    solver = SolverSteel(model, differentiable=True)
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


# A friction gap limit no contact reaches gives normal rows only.
_CONTACT_OPTIONS = {"friction_anchor_beta": 0.0, "contact_friction_gap_max": -math.inf, "enable_restitution": False}


def _build_box_chain_on_plane(device, *, restitution=0.0, free_box=False, extra_free_box=False, requires_grad=True):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.0, restitution=restitution)
    base = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.098), wp.quat_identity()))
    builder.add_shape_box(base, hx=0.2, hy=0.1, hz=0.1, cfg=cfg)
    joints = [builder.add_joint_free(parent=-1, child=base)]
    if not free_box:
        link = builder.add_link(xform=wp.transform(wp.vec3(0.4, 0.0, 0.098), wp.quat_identity()))
        builder.add_shape_box(link, hx=0.2, hy=0.1, hz=0.1, cfg=cfg)
        joints.append(
            builder.add_joint_revolute(
                base,
                link,
                parent_xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
                axis=(0.0, 1.0, 0.0),
                armature=0.01,
            )
        )
    builder.add_articulation(joints)
    if extra_free_box:
        box = builder.add_link(xform=wp.transform(wp.vec3(1.5, 0.0, 0.098), wp.quat_identity()))
        builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
        builder.add_articulation([builder.add_joint_free(parent=-1, child=box)])
    builder.add_ground_plane(cfg=cfg)
    return builder.finalize(device=device, requires_grad=requires_grad)


def _contact_rollout(model, pipeline, inputs, steps, *, differentiable, use_contacts=True, tape=None):
    """Collide every step outside the tape; return the loss inputs and per-step contact counts."""
    solver = SolverSteel(model, differentiable=differentiable, iterations=8, **_CONTACT_OPTIONS)
    states = [model.state(requires_grad=True) for _ in range(steps + 1)]
    states[0].joint_q.assign(inputs["joint_q"])
    states[0].joint_qd.assign(inputs["joint_qd"])
    newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
    control = model.control()
    control.joint_f.assign(inputs["joint_f"])
    contacts = pipeline.contacts()
    counts = []
    for k in range(steps):
        pipeline.collide(states[k], contacts)
        counts.append(int(contacts.rigid_contact_count.numpy()[0]))
        if tape is not None:
            tape.__enter__()
        solver.step(states[k], states[k + 1], control, contacts if use_contacts else None, _DT)
        if tape is not None:
            tape.__exit__(None, None, None)
    return states, control, counts


def _contact_inputs(model):
    rng = np.random.default_rng(0)
    return {
        "joint_q": model.joint_q.numpy().copy(),
        "joint_qd": (0.05 * rng.normal(size=model.joint_dof_count)).astype(np.float32),
        "joint_f": (0.5 * rng.normal(size=model.joint_dof_count)).astype(np.float32),
    }


def test_contact_forward_matches_default(test, device):
    model = _build_box_chain_on_plane(device)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=64)
    inputs = _contact_inputs(model)
    reference_model = _build_box_chain_on_plane(device, requires_grad=False)
    reference, _, reference_counts = _contact_rollout(reference_model, pipeline, inputs, 20, differentiable=False)
    result, _, counts = _contact_rollout(model, pipeline, inputs, 20, differentiable=True)
    test.assertEqual(counts, reference_counts)
    test.assertGreater(min(counts), 0)
    for attribute in ("joint_q", "joint_qd", "body_q", "body_qd"):
        expected = getattr(reference[-1], attribute).numpy()
        actual = getattr(result[-1], attribute).numpy()
        with test.subTest(attribute=attribute):
            if wp.get_device(device).is_cpu:
                np.testing.assert_array_equal(actual, expected)
            else:
                np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-5)


def test_contact_gradient_matches_finite_difference(test, device):
    model = _build_box_chain_on_plane(device)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=64)
    inputs = _contact_inputs(model)
    steps = 5
    eps = 1.0e-3
    rng = np.random.default_rng(6)
    wq = wp.array(rng.normal(size=model.joint_coord_count).astype(np.float32), device=device)
    wqd = wp.array(rng.normal(size=model.joint_dof_count).astype(np.float32), device=device)

    def loss_of(values, *, tape=None, use_contacts=True):
        states, control, counts = _contact_rollout(
            model, pipeline, values, steps, differentiable=True, use_contacts=use_contacts, tape=tape
        )
        loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
        if tape is not None:
            tape.__enter__()
        wp.launch(
            _weighted_loss,
            dim=max(model.joint_coord_count, model.joint_dof_count),
            inputs=[states[-1].joint_q, wq, states[-1].joint_qd, wqd],
            outputs=[loss],
            device=device,
        )
        if tape is not None:
            tape.__exit__(None, None, None)
        return loss, states, control, counts

    gradients = {}
    for use_contacts in (True, False):
        tape = wp.Tape()
        loss, states, control, counts = loss_of(inputs, tape=tape, use_contacts=use_contacts)
        tape.backward(loss)
        gradients[use_contacts] = {
            "joint_q": states[0].joint_q.grad.numpy(),
            "joint_qd": states[0].joint_qd.grad.numpy(),
            "joint_f": control.joint_f.grad.numpy(),
        }
        if use_contacts:
            nominal_counts = counts
            # Strict complementarity keeps the FD interval off active-set kinks.
            for state in states[1:]:
                rows = state._steel_differentiable_buffers.contacts
                count = int(rows.row_count.numpy()[0])
                impulse = rows.impulses[-1].numpy()[0, :count]
                residual = rows.residuals[-1].numpy()[0, :count]
                test.assertGreater(float(np.min(np.maximum(impulse, np.abs(residual)))), 1.0e-5)
    for key, gradient in gradients[True].items():
        direction = rng.normal(size=gradient.shape[0]).astype(np.float32)
        if key == "joint_q":
            direction[3:7] = 0.0
        direction /= np.linalg.norm(direction)
        plus_loss, _, _, plus_counts = loss_of(dict(inputs, **{key: inputs[key] + eps * direction}))
        minus_loss, _, _, minus_counts = loss_of(dict(inputs, **{key: inputs[key] - eps * direction}))
        fd = (plus_loss.numpy()[0] - minus_loss.numpy()[0]) / (2.0 * eps)
        analytic = float(gradient @ direction)
        with test.subTest(input=key):
            # The local derivative holds the contact set fixed.
            test.assertEqual(plus_counts, nominal_counts)
            test.assertEqual(minus_counts, nominal_counts)
            test.assertAlmostEqual(analytic, fd, delta=3.0e-3 * max(1.0, abs(fd)))

    # The contact solve changes the gradient.
    with_contacts = np.concatenate(list(gradients[True].values()))
    without_contacts = np.concatenate(list(gradients[False].values()))
    test.assertGreater(np.linalg.norm(with_contacts - without_contacts), 0.1 * np.linalg.norm(with_contacts))


def test_contact_rejects_unsupported_configs(test, device):
    model = _build_box_chain_on_plane(device)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=64)
    state = model.state(requires_grad=True)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    unsupported = [
        {**_CONTACT_OPTIONS, "contact_regularization": 0.1},
    ]
    for options in unsupported:
        solver = SolverSteel(model, differentiable=True, **options)
        with test.subTest(**options), test.assertRaisesRegex(NotImplementedError, "contacts require"):
            solver.step(state, model.state(requires_grad=True), None, contacts, _DT)
    for model, options in ((_build_box_chain_on_plane(device, extra_free_box=True), _CONTACT_OPTIONS),):
        solver = SolverSteel(model, differentiable=True, **options)
        state = model.state(requires_grad=True)
        contacts = newton.CollisionPipeline(model, rigid_contact_max=64).contacts()
        with test.assertRaisesRegex(NotImplementedError, "contacts require"):
            solver.step(state, model.state(requires_grad=True), None, contacts, _DT)


def _build_slider_on_plane(device, requires_grad=True):
    """Sphere on a prismatic x/y/z chain resting on a plane: one contact with one friction pair."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.5, restitution=0.0)
    inertia = wp.mat33(np.eye(3) * 0.01)
    link_x = builder.add_link(mass=1.0, inertia=inertia)
    link_y = builder.add_link(mass=0.5, inertia=inertia)
    link_z = builder.add_link(mass=0.3, xform=wp.transform(wp.vec3(0.0, 0.0, 0.098), wp.quat_identity()))
    builder.add_shape_sphere(link_z, radius=0.1, cfg=cfg)
    joints = [
        builder.add_joint_prismatic(-1, link_x, axis=(1.0, 0.0, 0.0)),
        builder.add_joint_prismatic(link_x, link_y, axis=(0.0, 1.0, 0.0)),
        builder.add_joint_prismatic(
            link_y,
            link_z,
            axis=(0.0, 0.0, 1.0),
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.098), wp.quat_identity()),
        ),
    ]
    builder.add_articulation(joints)
    builder.add_ground_plane(cfg=cfg)
    return builder.finalize(device=device, requires_grad=requires_grad)


def test_friction_gradient_matches_finite_difference(test, device):
    model = _build_slider_on_plane(device)
    reference_model = _build_slider_on_plane(device, requires_grad=False)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=16)
    steps = 5
    eps = 1.0e-3
    rng = np.random.default_rng(7)
    wq = wp.array(rng.normal(size=3).astype(np.float32), device=device)
    wqd = wp.array(rng.normal(size=3).astype(np.float32), device=device)
    options = {**_CONTACT_OPTIONS, "contact_friction_gap_max": None}

    def rollout(values, *, differentiable, tape=None):
        stepped = model if differentiable else reference_model
        solver = SolverSteel(stepped, differentiable=differentiable, iterations=8, **options)
        states = [stepped.state(requires_grad=True) for _ in range(steps + 1)]
        states[0].joint_qd.assign(values["joint_qd"])
        states[0].joint_q.assign(values["joint_q"])
        newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
        control = model.control()
        control.joint_f.assign(values["joint_f"])
        contacts = pipeline.contacts()
        for k in range(steps):
            pipeline.collide(states[k], contacts)
            if tape is not None:
                tape.__enter__()
            solver.step(states[k], states[k + 1], control, contacts, _DT)
            if tape is not None:
                tape.__exit__(None, None, None)
        loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
        if tape is not None:
            tape.__enter__()
        wp.launch(
            _weighted_loss,
            dim=3,
            inputs=[states[-1].joint_q, wq, states[-1].joint_qd, wqd],
            outputs=[loss],
            device=device,
        )
        if tape is not None:
            tape.__exit__(None, None, None)
        return loss, states, control

    # Coulomb budget mu * m_sphere * g is about 22 N: 2 N sticks, 60 N slides.
    for regime, push in (("stick", 2.0), ("slide", 60.0)):
        inputs = {
            "joint_q": np.zeros(3, np.float32),
            "joint_qd": np.array([0.02, -0.01, 0.0], np.float32),
            "joint_f": np.array([push, 0.4 * push, 0.0], np.float32),
        }
        _, reference, _ = rollout(inputs, differentiable=False)
        tape = wp.Tape()
        loss, states, control = rollout(inputs, differentiable=True, tape=tape)
        tape.backward(loss)
        with test.subTest(regime=regime, check="forward"):
            if wp.get_device(device).is_cpu:
                np.testing.assert_array_equal(states[-1].joint_qd.numpy(), reference[-1].joint_qd.numpy())
            else:
                np.testing.assert_allclose(states[-1].joint_qd.numpy(), reference[-1].joint_qd.numpy(), atol=1.0e-6)
        for state in states[1:]:
            rows = state._steel_differentiable_buffers.contacts
            impulse = rows.impulses[-1].numpy()[0, :3]
            budget = 0.5 * impulse[0]
            tangential = np.hypot(impulse[1], impulse[2])
            with test.subTest(regime=regime, check="regime"):
                if regime == "stick":
                    test.assertLess(tangential, 0.9 * budget)
                else:
                    test.assertAlmostEqual(tangential, budget, delta=1.0e-5 * budget)
        gradients = {
            "joint_q": states[0].joint_q.grad.numpy(),
            "joint_qd": states[0].joint_qd.grad.numpy(),
            "joint_f": control.joint_f.grad.numpy(),
        }
        for key, gradient in gradients.items():
            direction = rng.normal(size=3).astype(np.float32)
            direction /= np.linalg.norm(direction)
            plus = rollout(dict(inputs, **{key: inputs[key] + eps * direction}), differentiable=True)[0]
            minus = rollout(dict(inputs, **{key: inputs[key] - eps * direction}), differentiable=True)[0]
            fd = (plus.numpy()[0] - minus.numpy()[0]) / (2.0 * eps)
            with test.subTest(regime=regime, input=key):
                test.assertAlmostEqual(float(gradient @ direction), fd, delta=2.0e-3 * max(1.0, abs(fd)))


@wp.kernel
def _body_height_loss(body_q: wp.array[wp.transform], target: wp.vec3, loss: wp.array[float]):
    delta = wp.transform_get_translation(body_q[0]) - target
    loss[0] = wp.dot(delta, delta)


def test_sphere_multistep_gradient_flow(test, device):
    """Steel counterpart of test_differentiable_contacts.test_multistep_gradient_flow."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 2.0)))
    builder.add_shape_sphere(body=body, radius=0.5)
    builder.add_ground_plane()
    model = builder.finalize(device=device, requires_grad=True)
    pipeline = newton.CollisionPipeline(model, broad_phase="explicit", soft_contact_gap=10.0, requires_grad=True)
    substeps = 4
    dt = 1.0 / 60.0 / substeps
    target = wp.vec3(0.0, 0.0, 5.0)

    def loss_of(height, tape=None):
        solver = SolverSteel(model, differentiable=True, **_CONTACT_OPTIONS)
        states = [model.state(requires_grad=True) for _ in range(substeps + 1)]
        q = states[0].joint_q.numpy()
        q[2] = height
        states[0].joint_q.assign(q)
        newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
        contacts = pipeline.contacts()
        loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
        counts = []
        for k in range(substeps):
            pipeline.collide(states[k], contacts)
            counts.append(int(contacts.rigid_contact_count.numpy()[0]))
            if tape is not None:
                tape.__enter__()
            solver.step(states[k], states[k + 1], None, contacts, dt)
            if tape is not None:
                tape.__exit__(None, None, None)
        if tape is not None:
            tape.__enter__()
        wp.launch(_body_height_loss, dim=1, inputs=[states[-1].body_q, target], outputs=[loss], device=device)
        if tape is not None:
            tape.__exit__(None, None, None)
        return loss, states, counts

    tape = wp.Tape()
    loss, states, counts = loss_of(2.0, tape)
    tape.backward(loss)
    analytic_dz = float(states[0].joint_q.grad.numpy()[2])
    eps = 1.0e-3
    plus, _, plus_counts = loss_of(2.0 + eps)
    minus, _, minus_counts = loss_of(2.0 - eps)
    fd_dz = (plus.numpy()[0] - minus.numpy()[0]) / (2.0 * eps)
    # As in the SemiImplicit reference, the sphere starts well above the plane.
    test.assertEqual(plus_counts, counts)
    test.assertEqual(minus_counts, counts)
    test.assertLess(analytic_dz, 0.0)
    test.assertAlmostEqual(analytic_dz, fd_dz, delta=1.0e-3 * abs(fd_dz))


def test_sphere_friction_matches_default_and_finite_difference(test, device):
    """Resting sphere sliding to a stop: free rigid-body rows, which the default solves matrix-free."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.5, restitution=0.0)
    body = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.098), wp.quat_identity()))
    builder.add_shape_sphere(body, radius=0.1, cfg=cfg)
    builder.add_articulation([builder.add_joint_free(parent=-1, child=body)])
    builder.add_ground_plane(cfg=cfg)
    model = builder.finalize(device=device, requires_grad=True)
    reference_model = builder.finalize(device=device, requires_grad=False)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=16)
    options = {**_CONTACT_OPTIONS, "contact_friction_gap_max": None}
    steps = 5
    rng = np.random.default_rng(8)
    wq = wp.array(rng.normal(size=model.joint_coord_count).astype(np.float32), device=device)
    wqd = wp.array(rng.normal(size=model.joint_dof_count).astype(np.float32), device=device)
    inputs = {
        "joint_q": model.joint_q.numpy().copy(),
        "joint_qd": np.array([0.5, 0.2, 0.0, 0.0, 0.0, 0.3], np.float32),
        "joint_f": np.zeros(6, np.float32),
    }

    def rollout(values, *, differentiable, tape=None):
        stepped = model if differentiable else reference_model
        solver = SolverSteel(stepped, differentiable=differentiable, iterations=8, **options)
        states = [stepped.state(requires_grad=True) for _ in range(steps + 1)]
        states[0].joint_q.assign(values["joint_q"])
        states[0].joint_qd.assign(values["joint_qd"])
        newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
        control = model.control()
        control.joint_f.assign(values["joint_f"])
        contacts = pipeline.contacts()
        counts = []
        for k in range(steps):
            pipeline.collide(states[k], contacts)
            counts.append(int(contacts.rigid_contact_count.numpy()[0]))
            if tape is not None:
                tape.__enter__()
            solver.step(states[k], states[k + 1], control, contacts, _DT)
            if tape is not None:
                tape.__exit__(None, None, None)
        loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
        if tape is not None:
            tape.__enter__()
        wp.launch(
            _weighted_loss,
            dim=7,
            inputs=[states[-1].joint_q, wq, states[-1].joint_qd, wqd],
            outputs=[loss],
            device=device,
        )
        if tape is not None:
            tape.__exit__(None, None, None)
        return loss, states, control, counts

    _, reference, _, reference_counts = rollout(inputs, differentiable=False)
    tape = wp.Tape()
    loss, states, control, counts = rollout(inputs, differentiable=True, tape=tape)
    tape.backward(loss)
    test.assertEqual(counts, reference_counts)
    # The default solves free-body rows matrix-free: same Gauss-Seidel law, different rounding.
    np.testing.assert_allclose(states[-1].joint_qd.numpy(), reference[-1].joint_qd.numpy(), rtol=0.0, atol=1.0e-5)
    np.testing.assert_allclose(states[-1].joint_q.numpy(), reference[-1].joint_q.numpy(), rtol=0.0, atol=1.0e-6)
    gradients = {
        "joint_q": states[0].joint_q.grad.numpy(),
        "joint_qd": states[0].joint_qd.grad.numpy(),
        "joint_f": control.joint_f.grad.numpy(),
    }
    test.assertTrue(all(np.all(np.isfinite(gradient)) for gradient in gradients.values()))
    eps = 1.0e-3
    for key, gradient in gradients.items():
        direction = rng.normal(size=gradient.shape[0]).astype(np.float32)
        if key == "joint_q":
            direction[3:7] = 0.0
        direction /= np.linalg.norm(direction)
        plus = rollout(dict(inputs, **{key: inputs[key] + eps * direction}), differentiable=True)
        minus = rollout(dict(inputs, **{key: inputs[key] - eps * direction}), differentiable=True)
        fd = (plus[0].numpy()[0] - minus[0].numpy()[0]) / (2.0 * eps)
        with test.subTest(input=key):
            test.assertEqual(plus[3], counts)
            test.assertEqual(minus[3], counts)
            test.assertAlmostEqual(float(gradient @ direction), fd, delta=3.0e-3 * max(1.0, abs(fd)))


def test_repeated_rollouts_reuse_buffers(test, device):
    """One solver and one State list reused across identical rollouts reproduce values and gradients."""
    scenes = {
        "chain_drives": (_MODELS["chain_drives"](device), None, {}),
        "floating_chain": (_MODELS["floating_chain"](device), None, {}),
        "free_body": (_MODELS["free_body"](device), None, {}),
        "box_chain_contacts": (_build_box_chain_on_plane(device), 64, _CONTACT_OPTIONS),
        "slider_friction": (_build_slider_on_plane(device), 16, {**_CONTACT_OPTIONS, "contact_friction_gap_max": None}),
    }
    steps = 6
    for name, (model, contact_max, options) in scenes.items():
        solver = SolverSteel(model, differentiable=True, **options)
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=contact_max) if contact_max else None
        contacts = pipeline.contacts() if pipeline else None
        states = [model.state(requires_grad=True) for _ in range(steps + 1)]
        control = model.control()
        q0, qd0 = _initial_state(model, seed=9) if contact_max is None else (model.joint_q.numpy(), None)
        rng = np.random.default_rng(10)
        qd0 = (0.05 * rng.normal(size=model.joint_dof_count)).astype(np.float32) if qd0 is None else qd0
        control.joint_f.assign((rng.normal(size=model.joint_dof_count)).astype(np.float32))
        tape = wp.Tape()
        results = []
        for _rollout in range(3):
            states[0].joint_q.assign(q0)
            states[0].joint_qd.assign(qd0)
            newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
            tape.reset()
            for k in range(steps):
                if pipeline:
                    pipeline.collide(states[k], contacts)
                with tape:
                    solver.step(states[k], states[k + 1], control, contacts, _DT)
            tape.backward(grads={states[-1].joint_qd: wp.ones(model.joint_dof_count, dtype=float, device=device)})
            results.append(
                (
                    states[-1].joint_q.numpy().copy(),
                    states[0].joint_q.grad.numpy().copy(),
                    control.joint_f.grad.numpy().copy(),
                )
            )
            tape.zero()
        for rollout in results[1:]:
            for index, (expected, actual) in enumerate(zip(results[0], rollout, strict=True)):
                with test.subTest(scene=name, output=index):
                    test.assertTrue(np.all(np.isfinite(actual)))
                    if index == 0 or wp.get_device(device).is_cpu:
                        np.testing.assert_array_equal(actual, expected)
                    else:
                        # CUDA adjoint atomics accumulate in a nondeterministic order.
                        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-6 * np.max(np.abs(expected)))


def test_restitution_bounce_matches_default_and_finite_difference(test, device):
    """A sphere bounces (e = 0.5) at the same step in every FD sample; the firing step is the nonsmooth event."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.3, restitution=0.5)
    body = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.15), wp.quat_identity()))
    builder.add_shape_sphere(body, radius=0.1, cfg=cfg)
    builder.add_articulation([builder.add_joint_free(parent=-1, child=body)])
    builder.add_ground_plane(cfg=cfg)
    model = builder.finalize(device=device, requires_grad=True)
    reference_model = builder.finalize(device=device, requires_grad=False)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=8)
    options = {"friction_anchor_beta": 0.0, "enable_restitution": True, "iterations": 8}
    steps = 20
    dt = 1.0 / 240.0
    wq = wp.array(np.array([1.0, 0.5, 2.0, 0.0, 0.0, 0.0, 0.0], np.float32), device=device)
    wqd = wp.array(np.array([0.3, 0.2, 1.0, 0.1, 0.1, 0.1], np.float32), device=device)
    qd0 = np.array([0.3, 0.0, -2.0, 0.0, 0.0, 0.0], np.float32)

    def rollout(qd, *, differentiable, tape=None):
        stepped = model if differentiable else reference_model
        solver = SolverSteel(stepped, differentiable=differentiable, **options)
        states = [stepped.state(requires_grad=True) for _ in range(steps + 1)]
        states[0].joint_qd.assign(qd)
        newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
        contacts = pipeline.contacts()
        fired = []
        for k in range(steps):
            pipeline.collide(states[k], contacts)
            if tape is not None:
                tape.__enter__()
            solver.step(states[k], states[k + 1], None, contacts, dt)
            if tape is not None:
                tape.__exit__(None, None, None)
            if differentiable:
                rows = states[k + 1]._steel_differentiable_buffers.contacts
                count = int(rows.row_count.numpy()[0])
                if not np.array_equal(rows.rhs.numpy()[0, :count], rows.rhs_restituted.numpy()[0, :count]):
                    fired.append(k)
        loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
        if tape is not None:
            tape.__enter__()
        wp.launch(
            _weighted_loss,
            dim=7,
            inputs=[states[-1].joint_q, wq, states[-1].joint_qd, wqd],
            outputs=[loss],
            device=device,
        )
        if tape is not None:
            tape.__exit__(None, None, None)
        return loss, states, fired

    _, reference, _ = rollout(qd0, differentiable=False)
    tape = wp.Tape()
    loss, states, fired = rollout(qd0, differentiable=True, tape=tape)
    tape.backward(loss)
    test.assertEqual(len(fired), 1)
    test.assertGreater(states[-1].joint_qd.numpy()[2], -1.0)
    np.testing.assert_allclose(states[-1].joint_qd.numpy(), reference[-1].joint_qd.numpy(), rtol=0.0, atol=1.0e-5)
    gradient = states[0].joint_qd.grad.numpy()
    rng = np.random.default_rng(11)
    eps = 1.0e-2
    for _trial in range(2):
        direction = rng.normal(size=6).astype(np.float32)
        direction /= np.linalg.norm(direction)
        plus, _, plus_fired = rollout(qd0 + eps * direction, differentiable=True)
        minus, _, minus_fired = rollout(qd0 - eps * direction, differentiable=True)
        test.assertEqual(plus_fired, fired)
        test.assertEqual(minus_fired, fired)
        fd = (plus.numpy()[0] - minus.numpy()[0]) / (2.0 * eps)
        test.assertAlmostEqual(float(gradient @ direction), fd, delta=2.0e-3 * max(1.0, abs(fd)))


def _build_free_shape(device, shape, *, restitution=0.0, requires_grad=True):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.5, density=1000.0, restitution=restitution)
    builder.add_ground_plane(cfg=cfg)
    body = builder.add_link(mass=0.0)
    if shape == "box":
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
    else:
        builder.add_shape_sphere(body, radius=0.1, cfg=cfg)
    builder.add_articulation([builder.add_joint_free(parent=-1, child=body)])
    return builder.finalize(device=device, requires_grad=requires_grad)


def _free_body_rollout(model, solver, q0, qd0, steps, dt, contacts=None, tape=None):
    """Collide every step unless contacts are given (then they are reused, frozen)."""
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=16)
    states = [model.state(requires_grad=True) for _ in range(steps + 1)]
    states[0].joint_q.assign(np.asarray(q0, np.float32))
    states[0].joint_qd.assign(np.asarray(qd0, np.float32))
    newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
    frozen = contacts if contacts is not None else [pipeline.contacts() for _ in range(steps)]
    for k in range(steps):
        if contacts is None:
            pipeline.collide(states[k], frozen[k])
        if tape is not None:
            tape.__enter__()
        solver.step(states[k], states[k + 1], None, frozen[k], dt)
        if tape is not None:
            tape.__exit__(None, None, None)
    return states, frozen


def _velocity_jacobians(model, options, q0, qd0, dt):
    """Tape and central-FD Jacobians of joint_qd after one step w.r.t. initial joint_qd (contacts frozen)."""
    device = model.device
    solver = SolverSteel(model, differentiable=True, **options)
    _, contacts = _free_body_rollout(model, solver, q0, qd0, 1, dt)
    tape_jacobian = np.zeros((6, 6))
    for row in range(6):
        tape = wp.Tape()
        states, _ = _free_body_rollout(model, solver, q0, qd0, 1, dt, contacts, tape)
        seed = np.zeros(6, np.float32)
        seed[row] = 1.0
        tape.backward(grads={states[1].joint_qd: wp.array(seed, dtype=float, device=device)})
        tape_jacobian[row] = states[0].joint_qd.grad.numpy()
    fd_jacobian = np.zeros((6, 6))
    for column in range(6):
        step = np.zeros(6, np.float32)
        step[column] = 3.0e-3
        plus = _free_body_rollout(model, solver, q0, qd0 + step, 1, dt, contacts)[0][1].joint_qd.numpy()
        minus = _free_body_rollout(model, solver, q0, qd0 - step, 1, dt, contacts)[0][1].joint_qd.numpy()
        fd_jacobian[:, column] = (plus.astype(np.float64) - minus) / 6.0e-3
    return tape_jacobian, fd_jacobian


def test_sliding_box_velocity_jacobian(test, device):
    """Four sliding friction pairs: every entry, including rotational couplings, matches FD."""
    model = _build_free_shape(device, "box")
    q0 = [0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0]
    qd0 = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], np.float32)
    options = {"friction_anchor_beta": 0.0, "iterations": 12}
    tape_jacobian, fd_jacobian = _velocity_jacobians(model, options, q0, qd0, 1.0 / 240.0)
    np.testing.assert_allclose(tape_jacobian, fd_jacobian, rtol=0.0, atol=2.0e-3)
    # Rolling-free sliding: Coulomb gives d(vx)/d(vz0) = mu and no induced spin.
    test.assertAlmostEqual(fd_jacobian[0, 2], 0.5, delta=1.0e-3)


def test_frictional_impact_matches_default(test, device):
    """A sphere hitting the plane with tangential velocity rebounds as in the default matrix-free solve."""
    q0 = [0.0, 0.0, 0.15, 0.0, 0.0, 0.0, 1.0]
    qd0 = np.array([0.3, 0.0, -1.0, 0.0, 0.0, 0.0], np.float32)
    dt = 1.0 / 240.0
    results = {}
    for differentiable in (False, True):
        model = _build_free_shape(device, "sphere", restitution=0.5, requires_grad=differentiable)
        solver = SolverSteel(model, differentiable=differentiable, friction_anchor_beta=0.0)
        results[differentiable] = _free_body_rollout(model, solver, q0, qd0, 12, dt)[0][-1].joint_qd.numpy()
    test.assertGreater(results[True][2], 0.0)
    test.assertGreater(abs(results[True][4]), 1.0)
    np.testing.assert_allclose(results[True], results[False], rtol=0.0, atol=1.0e-6)
    model = _build_free_shape(device, "sphere", restitution=0.5)
    tape_jacobian, fd_jacobian = _velocity_jacobians(
        model, {"friction_anchor_beta": 0.0}, [0.0, 0.0, 0.102, 0, 0, 0, 1], qd0, dt
    )
    np.testing.assert_allclose(tape_jacobian, fd_jacobian, rtol=0.0, atol=2.0e-3)


def test_contact_law_options(test, device):
    """Shared anchors, the default patch friction and runtime depenetration clamps follow the default."""
    q0 = [0.0, 0.0, 0.09, 0.0, 0.0, 0.0, 1.0]
    qd0 = np.array([0.4, 0.1, 0.0, 0.0, 0.0, 0.2], np.float32)
    cases = (
        ({"friction_anchor_beta": 0.0, "contact_shared_anchor": True}, None),
        ({"friction_anchor_beta": 0.0, "contact_friction_shared_anchor": True}, None),
        ({"friction_anchor_beta": 0.0}, 0.05),
        # An omitted friction_anchor_beta selects the default persistent patch friction.
        ({}, None),
    )
    for options, clamp in cases:
        results = {}
        for differentiable in (False, True):
            model = _build_free_shape(device, "box", requires_grad=differentiable)
            solver = SolverSteel(model, differentiable=differentiable, **options)
            if clamp is not None:
                solver._rigid_body_max_depenetration_velocity.fill_(clamp)
            results[differentiable] = _free_body_rollout(model, solver, q0, qd0, 4, _DT)[0][-1]
        with test.subTest(options=options, clamp=clamp):
            np.testing.assert_allclose(
                results[True].joint_qd.numpy(), results[False].joint_qd.numpy(), rtol=0.0, atol=1.0e-5
            )
            if clamp is not None:
                # The 1 cm penetration recovers at the clamped speed.
                test.assertAlmostEqual(float(results[True].joint_qd.numpy()[2]), clamp, delta=1.0e-3)


def test_isotropic_friction_orientation_gradient(test, device):
    """An offset sphere sticking under an isotropic tangent block: the tangent-direction pose derivative matches FD."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    body = builder.add_link(
        mass=1.0,
        inertia=wp.mat33(np.diag([2.0, 1.0, 2.0])),
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.99), wp.quat_identity()),
    )
    cfg = newton.ModelBuilder.ShapeConfig(mu=2.0, density=0.0, restitution=0.0)
    builder.add_shape_sphere(body, radius=1.0, xform=wp.transform(wp.vec3(1.0, 0.0, 0.0), wp.quat_identity()), cfg=cfg)
    builder.add_articulation([builder.add_joint_free(parent=-1, child=body)])
    builder.add_ground_plane(cfg=cfg)
    model = builder.finalize(device=device, requires_grad=True)
    solver = SolverSteel(
        model,
        differentiable=True,
        friction_anchor_beta=0.0,
        iterations=1,
        baumgarte=0.0,
        cfm=0.0,
        enable_restitution=False,
    )
    solver._rigid_body_angular_damping.zero_()
    q0 = model.joint_q.numpy().copy()
    qd0 = np.array([0.2, 0.3, -0.5, 0.0, 0.0, 0.0], np.float32)
    initial = model.state()
    initial.joint_q.assign(q0)
    newton.eval_fk(model, initial.joint_q, initial.joint_qd, initial)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=8)
    contacts = pipeline.contacts()
    pipeline.collide(initial, contacts)
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 1)

    def run(theta, seed=None):
        state, output = model.state(requires_grad=True), model.state(requires_grad=True)
        q = q0.copy()
        q[3:7] = [np.sin(theta / 2), 0.0, 0.0, np.cos(theta / 2)]
        state.joint_q.assign(q)
        state.joint_qd.assign(qd0)
        tape = wp.Tape()
        with tape:
            solver.step(state, output, None, contacts, 0.01)
        if seed is None:
            return output.joint_qd.numpy().astype(np.float64)
        tape.backward(grads={output.joint_qd: wp.array(seed, dtype=float, device=device)})
        direction = np.array([0, 0, 0, 0.5 * np.cos(theta / 2), 0, 0, -0.5 * np.sin(theta / 2)])
        return float(state.joint_q.grad.numpy() @ direction)

    # At theta = 0 the pair sticks strictly inside its cone with a 2I tangent block, so FD is smooth.
    for theta in (0.0, 0.05):
        tape_derivative = np.array([run(theta, np.eye(6, dtype=np.float32)[j]) for j in range(6)])
        fd = (run(theta + 3.0e-3) - run(theta - 3.0e-3)) / 6.0e-3
        with test.subTest(theta=theta):
            test.assertGreater(abs(fd[1]), 5.0e-3)
            np.testing.assert_allclose(tape_derivative, fd, rtol=0.0, atol=1.0e-4)


def _pair_sweep(device, a, c, d, rhs, seed=(0.0, 1.0, 0.0)):
    """One sweep over a normal row and its friction pair; return impulses and the tangent-block gradients."""

    def array(values, dtype=float):
        return wp.array(np.asarray(values), dtype=dtype, device=device, requires_grad=True)

    diag = array([[1.0, a, d]])
    matrix = array([[[1.0, 0.0, 0.0], [0.0, a, c], [0.0, c, d]]])
    bias = array([rhs])
    residuals = wp.zeros((1, 3), device=device, requires_grad=True)
    impulses = wp.zeros((1, 3), device=device, requires_grad=True)
    row_type = [[PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION, PGS_CONSTRAINT_TYPE_FRICTION]]
    tape = wp.Tape()
    with tape:
        wp.launch(
            _dense_pgs_sweep,
            dim=1,
            inputs=[
                array([3], int),
                diag,
                matrix,
                bias,
                array(row_type, int),
                array([[-1, 0, 0]], int),
                array([[1.0, 1.0, 1.0]]),
                array([[1.0, 1.0, 1.0]]),
                0,
                1.0,
                0,
                0,
                array([[0.0, 0.0, 0.0]]),
                0,
                array([[0, 0, 0]], int),
                array([[0.0, 0.0, 0.0]]),
            ],
            outputs=[residuals, wp.zeros((1, 3), device=device), impulses],
            device=device,
        )
    tape.backward(grads={impulses: array([seed])})
    matrix_grad, diag_grad = matrix.grad.numpy()[0], diag.grad.numpy()[0]
    block_grad = [diag_grad[1], matrix_grad[1, 2] + matrix_grad[2, 1], diag_grad[2]]
    return impulses.numpy()[0], np.array([*block_grad, *bias.grad.numpy()[0, 1:]], np.float64)


def test_friction_pair_off_diagonal_gradient(test, device):
    """One sweep over a normal row and its friction pair: the tangent off-diagonal derivative matches FD."""
    eps = 1.0e-3
    for a, c, d in ((1.0, 0.0, 1.0), (1.0, 0.2, 2.0)):
        for rhs in ((-1.0, 0.2, 0.3), (-1.0, 2.0, 3.0)):
            with test.subTest(block=(a, c, d), rhs=rhs):
                plus = _pair_sweep(device, a, c + eps, d, rhs)[0][1]
                minus = _pair_sweep(device, a, c - eps, d, rhs)[0][1]
                fd = (float(plus) - float(minus)) / (2.0 * eps)
                test.assertGreater(abs(fd), 0.03)
                test.assertAlmostEqual(_pair_sweep(device, a, c, d, rhs)[1][1], fd, delta=2.0e-4)


def test_friction_pair_gradient_scale_invariance(test, device):
    """Scaling the tangent block and its bias by s keeps the impulse and scales every block derivative by 1/s."""
    cases = (
        ((1.0, 0.2, 2.0), (0.2, 0.3)),  # sticking
        ((1.0, 0.2, 2.0), (2.0, 3.0)),  # sliding
        ((0.0, 0.0, 2.25), (0.0, 0.3)),  # sticking on a rank-one block
    )
    for (a, c, d), tangent_rhs in cases:
        reference_impulses, reference_grad = _pair_sweep(device, a, c, d, (-1.0, *tangent_rhs), seed=(0.0, 1.0, 1.0))
        test.assertGreater(np.max(np.abs(reference_grad)), 0.01)
        for scale in (1.0e-22, 1.0e22):
            with test.subTest(block=(a, c, d), rhs=tangent_rhs, scale=scale):
                impulses, grad = _pair_sweep(
                    device,
                    a * scale,
                    c * scale,
                    d * scale,
                    (-1.0, tangent_rhs[0] * scale, tangent_rhs[1] * scale),
                    seed=(0.0, 1.0, 1.0),
                )
                np.testing.assert_allclose(impulses, reference_impulses, rtol=1.0e-5, atol=1.0e-6)
                np.testing.assert_allclose(grad * scale, reference_grad, rtol=1.0e-4, atol=1.0e-6)


@wp.kernel
def _friction_pair_kernel(rows: wp.array2d[float], out: wp.array[wp.vec2]):
    i = wp.tid()
    out[i] = _friction_pair(
        rows[i, 0],
        rows[i, 1],
        rows[i, 2],
        wp.vec2(rows[i, 3], rows[i, 4]),
        wp.vec2(rows[i, 5], rows[i, 6]),
        rows[i, 7],
        rows[i, 8],
    )


def test_friction_pair_shift_dominated_slide(test, device):
    """Sliding on a vanishing tangent block, where the root's shift dwarfs the block: x = -R r / |r| exactly."""
    residual = np.array([0.01, 0.005])
    radius = 0.01
    seed = np.array([1.0, 0.3])
    unit = residual / np.linalg.norm(residual)
    expected = -radius / np.linalg.norm(residual) * (np.eye(2) - np.outer(unit, unit)) @ seed
    for block in (1.0e-20, 0.0):
        with test.subTest(block=block):
            rows = wp.array(
                [[block, 0.0, block, *residual, 0.0, 0.0, radius, 1.0]], dtype=float, device=device, requires_grad=True
            )
            out = wp.zeros(1, dtype=wp.vec2, device=device, requires_grad=True)
            tape = wp.Tape()
            with tape:
                wp.launch(_friction_pair_kernel, dim=1, inputs=[rows], outputs=[out], device=device)
            tape.backward(grads={out: wp.array([seed], dtype=wp.vec2, device=device)})
            np.testing.assert_allclose(out.numpy()[0], -radius * unit, rtol=1.0e-5)
            np.testing.assert_allclose(rows.grad.numpy()[0, 3:5], expected, rtol=1.0e-4)


def test_rank_one_friction_pose_gradient(test, device):
    """A prismatic-hinge chain whose contact has a rank-one tangent block: the hinge-angle derivative matches FD."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    slider = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    body = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    prismatic = builder.add_joint_prismatic(
        -1, slider, axis=(0.0, 0.0, 1.0), parent_xform=wp.transform(wp.vec3(0.0, 0.0, 1.49), wp.quat_identity())
    )
    hinge = builder.add_joint_revolute(slider, body, axis=(0.0, 1.0, 0.0))
    builder.add_articulation([prismatic, hinge])
    cfg = newton.ModelBuilder.ShapeConfig(mu=10.0, density=0.0, restitution=0.0)
    builder.add_shape_sphere(body, radius=1.0, xform=wp.transform(wp.vec3(1.0, 0.0, -0.5), wp.quat_identity()), cfg=cfg)
    builder.add_ground_plane(cfg=cfg)
    model = builder.finalize(device=device, requires_grad=True)
    initial = model.state()
    newton.eval_fk(model, initial.joint_q, initial.joint_qd, initial)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=8)
    contacts = pipeline.contacts()
    pipeline.collide(initial, contacts)
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 1)
    solver = SolverSteel(
        model,
        differentiable=True,
        friction_anchor_beta=0.0,
        iterations=1,
        baumgarte=0.0,
        cfm=0.0,
        enable_restitution=False,
    )
    solver._rigid_body_angular_damping.zero_()

    def run(angle, grad=False):
        state, output = model.state(requires_grad=True), model.state(requires_grad=True)
        state.joint_q.assign(np.array([0.0, angle], np.float32))
        state.joint_qd.assign(np.array([-1.0, 0.4], np.float32))
        tape = wp.Tape()
        with tape:
            solver.step(state, output, None, contacts, 0.01)
        if not grad:
            return float(output.joint_qd.numpy()[1])
        # The y-tangent row has no Jacobian for these two DOFs, so the tangent block stays rank one.
        rows = output._steel_differentiable_buffers.contacts
        diag = rows.diag.numpy()[0]
        test.assertEqual(float(diag[1]), 0.0)
        test.assertGreater(float(diag[2]), 0.1)
        # The pair sticks strictly inside the friction disk, away from the stick/slip switch.
        impulses = rows.impulses[-1].numpy()[0]
        test.assertLess(float(np.hypot(impulses[1], impulses[2])), 0.5 * 10.0 * float(impulses[0]))
        tape.backward(grads={output.joint_qd: wp.array([0.0, 1.0], dtype=float, device=device)})
        return float(state.joint_q.grad.numpy()[1])

    for angle in (0.0, 0.1, 0.3):
        with test.subTest(angle=angle):
            fd = (run(angle + 3.0e-3) - run(angle - 3.0e-3)) / 6.0e-3
            test.assertAlmostEqual(run(angle, grad=True), fd, delta=2.0e-3)


@wp.func_native(
    snippet="""
#if defined(__CUDA_ARCH__)
unsigned long long start = clock64();
while (clock64() - start < (unsigned long long) ticks) {}
#endif
"""
)
def _spin(ticks: wp.int64): ...


@wp.kernel
def _publish_after_delay(count: wp.array[int], value: int, ticks: wp.int64):
    _spin(ticks)
    count[0] = value


def test_collide_done_event(test, device):
    """Contacts published on another stream are read after collide_done_event, eagerly and in a graph."""
    device = wp.get_device(device)
    model = _build_free_shape(device, "sphere")
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=16)
    producer = wp.Stream(device)
    solver = SolverSteel(model, differentiable=True, friction_anchor_beta=0.0, iterations=8)
    initial, output = model.state(), model.state()
    initial.joint_q.assign(np.array([0.0, 0.0, 0.095, 0.0, 0.0, 0.0, 1.0], np.float32))
    newton.eval_fk(model, initial.joint_q, initial.joint_qd, initial)
    contacts = pipeline.contacts()
    pipeline.collide(initial, contacts)
    count = int(contacts.rigid_contact_count.numpy()[0])
    test.assertGreater(count, 0)
    solver.step(initial, output, None, contacts, _DT)
    reference = output.joint_qd.numpy().copy()
    test.assertGreater(reference[2], 0.0)

    def publish_on_producer():
        producer.wait_stream(wp.get_stream(device))
        with wp.ScopedStream(producer):
            wp.launch(
                _publish_after_delay, dim=1, inputs=[contacts.rigid_contact_count, count, 200_000_000], device=device
            )
            return producer.record_event()

    wp.launch(_publish_after_delay, dim=1, inputs=[contacts.rigid_contact_count, count, 0], device=device)
    contacts.rigid_contact_count.zero_()
    solver.step(initial, output, None, contacts, _DT, collide_done_event=publish_on_producer())
    np.testing.assert_array_equal(output.joint_qd.numpy(), reference)

    with wp.ScopedCapture(device) as capture:
        contacts.rigid_contact_count.zero_()
        solver.step(initial, output, None, contacts, _DT, collide_done_event=publish_on_producer())
    output.joint_qd.zero_()
    wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(output.joint_qd.numpy(), reference)


def test_pgs_kernel_selection(test, device):
    """Accepted dense PGS kernels match the default step; CUDA rejects the contact-block kernels."""
    reference_model = _build_box_chain_on_plane(device, requires_grad=False)
    model = _build_box_chain_on_plane(device)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=16)
    initial = model.state()
    initial.joint_qd.assign(np.full(model.joint_dof_count, 0.5, np.float32))
    newton.eval_fk(model, initial.joint_q, initial.joint_qd, initial)
    contacts = pipeline.contacts()
    pipeline.collide(initial, contacts)
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 8)
    block_kernels = ("tiled_contact", "streaming") if wp.get_device(device).is_cuda else ()
    options = {"friction_anchor_beta": 0.0, "iterations": 1}
    for kernel in ("loop", "tiled_row", "tiled_contact", "streaming"):
        with test.subTest(pgs_kernel=kernel), kernel_overrides({"pgs_kernel": kernel}):
            if kernel in block_kernels:
                with test.assertRaisesRegex(ValueError, 'the "loop" or "tiled_row" dense PGS kernel'):
                    SolverSteel(model, differentiable=True, **options)
                continue
            results = []
            for differentiable in (False, True):
                output = model.state(requires_grad=True)
                SolverSteel(
                    model if differentiable else reference_model, differentiable=differentiable, **options
                ).step(initial, output, None, contacts, 0.01)
                results.append(output.joint_qd.numpy())
            np.testing.assert_allclose(results[1], results[0], rtol=0.0, atol=1.0e-5)


def test_free_body_rows_use_dense_capacity(test, device):
    """Free-body contact rows take articulation_constraint_max slots; free_body_constraint_max does not cover them."""
    q0 = [0.0, 0.0, 0.095, 0.0, 0.0, 0.0, 1.0]
    qd0 = np.array([0.2, 0.3, -0.5, 0.0, 0.0, 0.0], np.float32)
    for capacity in (3, 32):
        results = {}
        for differentiable in (False, True):
            model = _build_free_shape(device, "box", requires_grad=differentiable)
            solver = SolverSteel(
                model,
                differentiable=differentiable,
                friction_anchor_beta=0.0,
                articulation_constraint_max=capacity,
                free_body_constraint_max=32,
                iterations=1,
            )
            states, contacts = _free_body_rollout(model, solver, q0, qd0, 1, 0.01)
            test.assertEqual(int(contacts[0].rigid_contact_count.numpy()[0]), 4)
            results[differentiable] = states[1].joint_qd.numpy()
            with test.subTest(capacity=capacity, differentiable=differentiable):
                if differentiable and capacity == 3:
                    with test.assertRaisesRegex(RuntimeError, "capacity exceeded"):
                        solver.check_constraint_capacity()
                else:
                    solver.check_constraint_capacity()
        if capacity == 32:
            np.testing.assert_allclose(results[True], results[False], rtol=0.0, atol=1.0e-5)


def test_smooth_contact_rejects_unsupported_configs(test, device):
    model = _build_free_body(device)
    law = steel.SmoothContactLaw(stiffness=1.0e4)
    with test.assertRaises(ValueError):
        SolverSteel(model, smooth_contact=law, enable_restitution=False)
    with test.assertRaises(ValueError):
        SolverSteel(model, differentiable=True, smooth_contact=law)
    for bad in ({"softness": 0.0}, {"stiffness": math.inf}, {"damping": math.inf}, {"softness": math.inf}):
        with test.assertRaises(ValueError):
            steel.SmoothContactLaw(**{"stiffness": 1.0e4, **bad})


@wp.kernel
def _add_separating_velocity(bias: wp.array2d[float], velocity: float, rhs: wp.array2d[float]):
    rhs[0, 0] = bias[0, 0] + velocity


def _smooth_zero_gap_impulse(device, gap, velocity, damping, beta, grad=False):
    """One contact row through the contact bias and the smooth impulse; return the impulse or d/dgap."""

    def array(values, dtype=float):
        return wp.array(values, dtype=dtype, device=device, requires_grad=dtype is float)

    phi, row_beta, target = array([[gap]]), array([[beta]]), array([[0.0]])
    count, kind = array([1], int), array([[PGS_CONSTRAINT_TYPE_CONTACT]], int)
    bias, row_w, rhs, impulse = (wp.zeros((1, 1), device=device, requires_grad=True) for _ in range(4))
    dt, speculative = 0.01, 1.0
    tape = wp.Tape()
    with tape:
        wp.launch(
            _taped_compute_world_contact_bias,
            1,
            inputs=[count, phi, row_beta, kind, target, dt, 1.0, speculative, 1.0, 1.0],
            outputs=[bias, row_w],
            device=device,
        )
        wp.launch(_add_separating_velocity, 1, inputs=[bias, velocity], outputs=[rhs], device=device)
        wp.launch(
            _smooth_contact_impulse,
            (1, 1),
            inputs=[count, phi, row_beta, kind, rhs, dt, speculative, 1000.0, damping, 1.0e-3],
            outputs=[impulse],
            device=device,
        )
    if not grad:
        return float(impulse.numpy()[0, 0])
    tape.backward(grads={impulse: wp.ones_like(impulse)})
    return float(phi.grad.numpy()[0, 0])


def test_smooth_contact_zero_gap_activation_gradient(test, device):
    """At phi == 0 with no damping the impulse slope is -dt * k * sigmoid(0), on the activation branch point."""
    for gap in (-1.0e-6, 0.0, 1.0e-6):
        slope = _smooth_zero_gap_impulse(device, gap, 0.0, damping=0.0, beta=0.2, grad=True)
        test.assertAlmostEqual(slope, -5.0, delta=1.0e-2)


def test_smooth_contact_damped_zero_gap_gradient_matches_finite_difference(test, device):
    """Damped zero-gap row with beta != speculative scale: the bias builder and the smooth law cancel exactly."""
    eps = 1.0e-5
    tape_slope = _smooth_zero_gap_impulse(device, 0.0, 0.1, damping=10.0, beta=0.2, grad=True)
    fd_slope = (
        _smooth_zero_gap_impulse(device, eps, 0.1, damping=10.0, beta=0.2)
        - _smooth_zero_gap_impulse(device, -eps, 0.1, damping=10.0, beta=0.2)
    ) / (2.0 * eps)
    test.assertAlmostEqual(tape_slope, fd_slope, delta=1.0e-2 * abs(fd_slope))


def test_smooth_contact_bounce_gradient_converges_across_eps(test, device):
    """An elastic SmoothContactLaw bounce: central FD agrees with the tape at every eps, unlike hard restitution."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.3)
    body = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.15), wp.quat_identity()))
    builder.add_shape_sphere(body, radius=0.1, cfg=cfg)
    builder.add_articulation([builder.add_joint_free(parent=-1, child=body)])
    builder.add_ground_plane(cfg=cfg)
    model = builder.finalize(device=device, requires_grad=True)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=8)
    law = steel.SmoothContactLaw(stiffness=1.0e4, softness=1.0e-2)
    options = {"friction_anchor_beta": 0.0, "enable_restitution": False, "iterations": 8, "smooth_contact": law}
    steps = 40
    dt = 1.0 / 240.0
    wq = wp.array(np.array([1.0, 0.5, 2.0, 0.0, 0.0, 0.0, 0.0], np.float32), device=device)
    wqd = wp.array(np.array([0.3, 0.2, 1.0, 0.1, 0.1, 0.1], np.float32), device=device)
    qd0 = np.array([0.3, 0.0, -2.0, 0.0, 0.0, 0.0], np.float32)
    solver = SolverSteel(model, differentiable=True, **options)
    states = [model.state(requires_grad=True) for _ in range(steps + 1)]
    contacts = pipeline.contacts()

    def rollout(qd, tape=None):
        states[0].joint_q.assign(model.joint_q)
        states[0].joint_qd.assign(qd)
        newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
        for k in range(steps):
            pipeline.collide(states[k], contacts)
            if tape is not None:
                tape.__enter__()
            solver.step(states[k], states[k + 1], None, contacts, dt)
            if tape is not None:
                tape.__exit__(None, None, None)
        loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
        if tape is not None:
            tape.__enter__()
        wp.launch(
            _weighted_loss,
            dim=7,
            inputs=[states[-1].joint_q, wq, states[-1].joint_qd, wqd],
            outputs=[loss],
            device=device,
        )
        if tape is not None:
            tape.__exit__(None, None, None)
        return loss

    tape = wp.Tape()
    loss = rollout(qd0, tape)
    tape.backward(loss)
    gradient = states[0].joint_qd.grad.numpy().copy()
    test.assertGreater(states[-1].joint_qd.numpy()[2], 1.0)
    tape.zero()
    direction = np.random.default_rng(3).normal(size=6).astype(np.float32)
    direction /= np.linalg.norm(direction)
    for eps in (1.0e-1, 1.0e-2):
        fd = (rollout(qd0 + eps * direction).numpy()[0] - rollout(qd0 - eps * direction).numpy()[0]) / (2.0 * eps)
        with test.subTest(eps=eps):
            test.assertAlmostEqual(float(gradient @ direction), fd, delta=1.0e-2 * max(1.0, abs(fd)))


def test_runtime_disabled_tree_joint_is_reported(test, device):
    """A tree joint disabled after construction fails the status check, eagerly and in a replayed graph."""
    model = _build_chain(device)
    solver = SolverSteel(model, differentiable=True)
    state, output = model.state(requires_grad=True), model.state(requires_grad=True)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    capture = None
    if wp.get_device(device).is_cuda:
        solver.step(state, output, None, None, _DT)
        with wp.ScopedCapture(device) as capture:
            solver.step(state, output, None, None, _DT)
    enabled = model.joint_enabled.numpy()

    def step():
        if capture is None:
            solver.step(state, output, None, None, _DT)
        else:
            wp.capture_launch(capture.graph)

    step()
    solver.check_constraint_capacity()
    enabled[1] = False
    model.joint_enabled.assign(enabled)
    step()
    with test.assertRaisesRegex(RuntimeError, "disables an articulation tree joint"):
        solver.check_constraint_capacity()


def test_unarticulated_body_contact_matches_default(test, device):
    """A raised body outside every articulation supports the slider: its pose reaches the contact rows."""

    def build(requires_grad):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        cfg = newton.ModelBuilder.ShapeConfig(mu=0.5, restitution=0.0)
        inertia = wp.mat33(np.eye(3) * 0.01)
        link_x = builder.add_link(mass=1.0, inertia=inertia)
        link_y = builder.add_link(mass=0.5, inertia=inertia)
        link_z = builder.add_link(mass=0.3, xform=wp.transform(wp.vec3(0.0, 0.0, 0.598), wp.quat_identity()))
        builder.add_shape_sphere(link_z, radius=0.1, cfg=cfg)
        builder.add_articulation(
            [
                builder.add_joint_prismatic(-1, link_x, axis=(1.0, 0.0, 0.0)),
                builder.add_joint_prismatic(link_x, link_y, axis=(0.0, 1.0, 0.0)),
                builder.add_joint_prismatic(
                    link_y,
                    link_z,
                    axis=(0.0, 0.0, 1.0),
                    parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.598), wp.quat_identity()),
                ),
            ]
        )
        support = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.25), wp.quat_identity()))
        builder.add_shape_box(support, hx=1.0, hy=1.0, hz=0.25, cfg=cfg)
        return builder.finalize(device=device, requires_grad=requires_grad), support

    options = {**_CONTACT_OPTIONS, "contact_friction_gap_max": None, "iterations": 8}
    qd0 = np.array([0.3, -0.2, -0.5], np.float32)
    results = {}
    for differentiable in (False, True):
        model, support = build(differentiable)
        solver = SolverSteel(model, differentiable=differentiable, **options)
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=16)
        contacts = pipeline.contacts()
        states = [model.state(requires_grad=True) for _ in range(9)]
        states[0].joint_qd.assign(qd0)
        newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
        for k in range(8):
            pipeline.collide(states[k], contacts)
            test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
            solver.step(states[k], states[k + 1], None, contacts, _DT)
        results[differentiable] = states[-1]
        np.testing.assert_array_equal(states[-1].body_q.numpy()[support], states[0].body_q.numpy()[support])
    # The sphere rests on the support, half a metre above the ground plane it never reaches.
    test.assertGreater(float(results[True].joint_q.numpy()[2]), -0.05)
    for attribute in ("joint_q", "joint_qd"):
        expected = getattr(results[False], attribute).numpy()
        actual = getattr(results[True], attribute).numpy()
        with test.subTest(attribute=attribute):
            if wp.get_device(device).is_cpu:
                np.testing.assert_array_equal(actual, expected)
            else:
                np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1.0e-5)


def _contact_scene_rollout(model, options, inputs, steps, *, differentiable, dt=_DT):
    """Step a contact scene from ``inputs``, colliding every step; return the final state."""
    solver = SolverSteel(model, differentiable=differentiable, **options)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=64)
    contacts = pipeline.contacts()
    states = [model.state(requires_grad=True) for _ in range(steps + 1)]
    states[0].joint_q.assign(inputs["joint_q"])
    states[0].joint_qd.assign(inputs["joint_qd"])
    newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
    control = model.control()
    control.joint_f.assign(inputs["joint_f"])
    for k in range(steps):
        pipeline.collide(states[k], contacts)
        solver.step(states[k], states[k + 1], control, contacts, dt)
    return states[-1], solver


_MATRIX_FREE_SCENES = {
    "box_chain": _build_box_chain_on_plane,
    "slider": lambda device, requires_grad=True: _build_slider_on_plane(device, requires_grad=requires_grad),
    "free_box": lambda device, requires_grad=True: _build_free_shape(device, "box", requires_grad=requires_grad),
}


def test_matrix_free_contacts_match_default(test, device):
    """With solve_mode="matrix_free" the dense sweep follows the fused matrix-free Gauss-Seidel solve."""
    options = {"friction_anchor_beta": 0.0, "solve_mode": "matrix_free", "iterations": 8}
    for name, build in _MATRIX_FREE_SCENES.items():
        model = build(device, requires_grad=False)
        rng = np.random.default_rng(0)
        inputs = {
            "joint_q": model.joint_q.numpy().copy(),
            "joint_qd": (0.3 * rng.normal(size=model.joint_dof_count)).astype(np.float32),
            "joint_f": (0.5 * rng.normal(size=model.joint_dof_count)).astype(np.float32),
        }
        reference, _ = _contact_scene_rollout(model, options, inputs, 20, differentiable=False)
        result, _ = _contact_scene_rollout(build(device), options, inputs, 20, differentiable=True)
        for attribute in ("joint_q", "joint_qd"):
            with test.subTest(scene=name, attribute=attribute):
                np.testing.assert_allclose(
                    getattr(result, attribute).numpy(), getattr(reference, attribute).numpy(), rtol=0.0, atol=1.0e-5
                )


def _build_torsion_scene(device, shape, *, requires_grad=True, radius=0.05):
    """A sphere (one contact) or box (four) floating base carrying a hinged arm, with contact torsion."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.5, restitution=0.0)
    base = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.099), wp.quat_identity()))
    if shape == "sphere":
        builder.add_shape_sphere(base, radius=0.1, cfg=cfg)
    else:
        builder.add_shape_box(base, hx=0.2, hy=0.15, hz=0.1, cfg=cfg)
    arm = builder.add_link(
        mass=0.2, inertia=wp.mat33(np.eye(3) * 1.0e-3), xform=wp.transform(wp.vec3(0.0, 0.0, 0.3), wp.quat_identity())
    )
    joints = [
        builder.add_joint_free(parent=-1, child=base),
        builder.add_joint_revolute(
            base,
            arm,
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.2), wp.quat_identity()),
            axis=(1.0, 0.0, 0.0),
            armature=0.01,
        ),
    ]
    builder.add_articulation(joints)
    builder.add_ground_plane(cfg=cfg)
    model = builder.finalize(device=device, requires_grad=requires_grad)
    set_torsion_radius(model, radius)
    return model


_TORSION_OPTIONS = {"friction_anchor_beta": 0.0, "solve_mode": "matrix_free", "iterations": 8}


def _torsion_inputs(model, spin):
    return {
        "joint_q": model.joint_q.numpy().copy(),
        "joint_qd": np.array([0.2, 0.1, 0.0, 0.0, 0.0, spin, 0.5], np.float32),
        "joint_f": np.zeros(model.joint_dof_count, np.float32),
    }


def test_contact_torsion_matches_default(test, device):
    """Spin rows of sliding (saturated) and sticking contact groups follow the default matrix-free solve."""
    # The device preparation builds the host's groups; the differentiable step groups on the host for both.
    cases = (
        ("sphere", 20.0, 4, False),
        ("sphere", 3.0, 10, False),
        ("box", 20.0, 4, False),
        ("sphere", 20.0, 4, True),
    )
    for shape, spin, steps, device_preparation in cases:
        options = {**_TORSION_OPTIONS, "contact_torsion_device": device_preparation}
        inputs = _torsion_inputs(_build_torsion_scene(device, shape), spin)
        reference_model = _build_torsion_scene(device, shape, requires_grad=False)
        reference, _ = _contact_scene_rollout(reference_model, options, inputs, steps, differentiable=False)
        result, _ = _contact_scene_rollout(
            _build_torsion_scene(device, shape), options, inputs, steps, differentiable=True
        )
        untwisted, _ = _contact_scene_rollout(
            _build_torsion_scene(device, shape, requires_grad=False, radius=0.0),
            options,
            inputs,
            steps,
            differentiable=False,
        )
        with test.subTest(shape=shape, spin=spin, contact_torsion_device=device_preparation):
            rows = result._steel_differentiable_buffers.contacts
            spin_rows = rows.row_type.numpy()[0, : int(rows.row_count.numpy()[0])] == PGS_CONSTRAINT_TYPE_TORSION
            test.assertEqual(int(np.count_nonzero(spin_rows)), 1)
            if shape == "sphere":
                # One witness has no spin resistance without its spin row; the sliding box corners use the whole
                # shared budget, which leaves the spin row a zero bound.
                test.assertGreater(abs(float(untwisted.joint_qd.numpy()[5] - reference.joint_qd.numpy()[5])), 0.5)
            for attribute in ("joint_q", "joint_qd"):
                np.testing.assert_allclose(
                    getattr(result, attribute).numpy(), getattr(reference, attribute).numpy(), rtol=1.0e-6, atol=1.0e-5
                )


def test_contact_torsion_gradient_matches_finite_difference(test, device):
    """Tape against central differences for spin rows at their bound and inside it."""
    for shape, spin in (("sphere", 20.0), ("sphere", 0.3), ("box", 0.3)):
        model = _build_torsion_scene(device, shape)
        solver = SolverSteel(model, differentiable=True, **_TORSION_OPTIONS)
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=64)
        rng = np.random.default_rng(3)
        wq = wp.array(rng.normal(size=model.joint_coord_count).astype(np.float32), device=device)
        wqd = wp.array(rng.normal(size=model.joint_dof_count).astype(np.float32), device=device)
        qd0 = _torsion_inputs(model, spin)["joint_qd"]

        def loss_of(qd, tape=None, model=model, solver=solver, pipeline=pipeline, wq=wq, wqd=wqd):
            states = [model.state(requires_grad=True) for _ in range(5)]
            states[0].joint_qd.assign(qd)
            newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
            counts = []
            for k in range(4):
                contacts = pipeline.contacts()
                pipeline.collide(states[k], contacts)
                counts.append(int(contacts.rigid_contact_count.numpy()[0]))
                if tape is not None:
                    tape.__enter__()
                solver.step(states[k], states[k + 1], None, contacts, _DT)
                if tape is not None:
                    tape.__exit__(None, None, None)
            loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
            if tape is not None:
                tape.__enter__()
            wp.launch(
                _weighted_loss,
                dim=max(model.joint_coord_count, model.joint_dof_count),
                inputs=[states[-1].joint_q, wq, states[-1].joint_qd, wqd],
                outputs=[loss],
                device=device,
            )
            if tape is not None:
                tape.__exit__(None, None, None)
            return float(loss.numpy()[0]), states, counts

        tape = wp.Tape()
        _, states, counts = loss_of(qd0, tape)
        tape.backward(grads={tape.launches[-1][4][0]: wp.ones(1, dtype=float, device=device)})
        gradient = states[0].joint_qd.grad.numpy().astype(np.float64)
        eps = 3.0e-3
        for trial in range(3):
            direction = rng.normal(size=model.joint_dof_count)
            direction /= np.linalg.norm(direction)
            plus = loss_of((qd0 + eps * direction).astype(np.float32))
            minus = loss_of((qd0 - eps * direction).astype(np.float32))
            fd = (plus[0] - minus[0]) / (2.0 * eps)
            with test.subTest(shape=shape, spin=spin, trial=trial):
                test.assertEqual(plus[2], counts)
                test.assertEqual(minus[2], counts)
                test.assertAlmostEqual(float(gradient @ direction), fd, delta=2.0e-3 * max(1.0, abs(fd)))


def _build_arm_on_box_stack(device, *, requires_grad=True, stack=True):
    """A two-link arm pressing a stack of two free boxes, or one box, on the ground, all in one world."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.5, restitution=0.0)
    builder.add_ground_plane(cfg=cfg)
    lower = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0995), wp.quat_identity()))
    builder.add_shape_box(lower, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
    if stack:
        upper = builder.add_body(xform=wp.transform(wp.vec3(0.02, 0.01, 0.2985), wp.quat_identity()))
        builder.add_shape_box(upper, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
    height = 0.6 if stack else 0.4
    inertia = wp.mat33(np.eye(3) * 0.01)
    link0 = builder.add_link(
        mass=1.0, inertia=inertia, xform=wp.transform(wp.vec3(-0.3, 0.0, height), wp.quat_identity())
    )
    link1 = builder.add_link(
        mass=0.5, inertia=inertia, xform=wp.transform(wp.vec3(0.0, 0.0, height), wp.quat_identity())
    )
    builder.add_shape_capsule(link1, radius=0.03, half_height=0.12, cfg=cfg)
    joints = [
        builder.add_joint_revolute(
            -1,
            link0,
            parent_xform=wp.transform(wp.vec3(-0.3, 0.0, height), wp.quat_identity()),
            axis=(0.0, 1.0, 0.0),
            armature=0.01,
        ),
        builder.add_joint_revolute(
            link0,
            link1,
            parent_xform=wp.transform(wp.vec3(0.3, 0.0, 0.0), wp.quat_identity()),
            axis=(0.0, 1.0, 0.0),
            armature=0.01,
        ),
    ]
    builder.add_articulation(joints)
    return builder.finalize(device=device, requires_grad=requires_grad)


# Every contact row takes an articulated slot in the differentiable step, free-body rows included.
_ARM_ON_BOX_OPTIONS = {
    "friction_anchor_beta": 0.0,
    "solve_mode": "matrix_free",
    "iterations": 16,
    "articulation_constraint_max": 64,
}


def _arm_on_box_inputs(model):
    joint_f = np.zeros(model.joint_dof_count, np.float32)
    joint_f[-2:] = (3.0, 2.0)
    return {
        "joint_q": model.joint_q.numpy().copy(),
        "joint_qd": np.zeros(model.joint_dof_count, np.float32),
        "joint_f": joint_f,
    }


def test_mixed_world_contacts_match_default(test, device):
    """A world with articulated and free-body contacts follows the matrix-free solve, which sweeps them in turn."""
    for extra in ({}, {"deterministic": True}, {"contact_regularization": 0.01}, {"contact_regularization": 0.5}):
        options = {**_ARM_ON_BOX_OPTIONS, **extra}
        inputs = _arm_on_box_inputs(_build_arm_on_box_stack(device))
        reference, solver = _contact_scene_rollout(
            _build_arm_on_box_stack(device, requires_grad=False), options, inputs, 20, differentiable=False
        )
        test.assertGreater(int(solver._constraint_count.numpy()[0]), 0)
        test.assertGreater(int(solver._mf_constraint_count.numpy()[0]), 0)
        result, solver = _contact_scene_rollout(
            _build_arm_on_box_stack(device), options, inputs, 20, differentiable=True
        )
        solver.check_constraint_capacity()
        with test.subTest(**extra):
            for attribute in ("joint_q", "joint_qd"):
                np.testing.assert_allclose(
                    getattr(result, attribute).numpy(), getattr(reference, attribute).numpy(), rtol=1.0e-5, atol=2.0e-5
                )


def test_deterministic_rollouts_repeat(test, device):
    """deterministic=True reserves rows in contact order and sums articulations in order: repeats are bitwise."""
    options = {**_ARM_ON_BOX_OPTIONS, "deterministic": True}
    inputs = _arm_on_box_inputs(_build_arm_on_box_stack(device))
    results = [
        _contact_scene_rollout(_build_arm_on_box_stack(device), options, inputs, 20, differentiable=True)[0]
        for _ in range(3)
    ]
    for result in results[1:]:
        np.testing.assert_array_equal(result.joint_qd.numpy(), results[0].joint_qd.numpy())
        np.testing.assert_array_equal(result.body_q.numpy(), results[0].body_q.numpy())


def test_mixed_world_gradient_matches_finite_difference(test, device):
    """Tape against central differences through the arm's contact with a free box stack, with and without
    contact regularization."""
    for extra in ({}, {"contact_regularization": 0.01}):
        with test.subTest(**extra):
            _check_arm_on_box_gradient(test, device, {**_ARM_ON_BOX_OPTIONS, **extra})


def _check_arm_on_box_gradient(test, device, options):
    model = _build_arm_on_box_stack(device)
    solver = SolverSteel(model, differentiable=True, **options)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=64)
    rng = np.random.default_rng(5)
    wq = wp.array(rng.normal(size=model.joint_coord_count).astype(np.float32), device=device)
    wqd = wp.array(rng.normal(size=model.joint_dof_count).astype(np.float32), device=device)
    inputs = _arm_on_box_inputs(model)
    steps = 4

    def loss_of(joint_f, tape=None):
        states = [model.state(requires_grad=True) for _ in range(steps + 1)]
        newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
        control = model.control()
        control.joint_f.assign(joint_f)
        counts = []
        for k in range(steps):
            contacts = pipeline.contacts()
            pipeline.collide(states[k], contacts)
            counts.append(int(contacts.rigid_contact_count.numpy()[0]))
            if tape is not None:
                tape.__enter__()
            solver.step(states[k], states[k + 1], control, contacts, _DT)
            if tape is not None:
                tape.__exit__(None, None, None)
        loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
        if tape is not None:
            tape.__enter__()
        wp.launch(
            _weighted_loss,
            dim=max(model.joint_coord_count, model.joint_dof_count),
            inputs=[states[-1].joint_q, wq, states[-1].joint_qd, wqd],
            outputs=[loss],
            device=device,
        )
        if tape is not None:
            tape.__exit__(None, None, None)
        return loss, control, counts

    tape = wp.Tape()
    loss, control, counts = loss_of(inputs["joint_f"], tape)
    tape.backward(loss)
    gradient = control.joint_f.grad.numpy().astype(np.float64)
    eps = 1.0e-2
    for trial in range(3):
        direction = rng.normal(size=model.joint_dof_count)
        direction /= np.linalg.norm(direction)
        plus = loss_of((inputs["joint_f"] + eps * direction).astype(np.float32))
        minus = loss_of((inputs["joint_f"] - eps * direction).astype(np.float32))
        fd = (float(plus[0].numpy()[0]) - float(minus[0].numpy()[0])) / (2.0 * eps)
        with test.subTest(trial=trial):
            test.assertEqual(plus[2], counts)
            test.assertEqual(minus[2], counts)
            test.assertAlmostEqual(float(gradient @ direction), fd, delta=2.0e-3 * max(1.0, abs(fd)))


def test_matrix_free_restitution_matches_default(test, device):
    """Rebound targets of articulated and free-body rows follow the matrix-free route, regularized or not."""
    scenes = {
        "box_chain": lambda requires_grad: _build_box_chain_on_plane(
            device, restitution=0.5, requires_grad=requires_grad
        ),
        "sphere": lambda requires_grad: _build_free_shape(
            device, "sphere", restitution=0.5, requires_grad=requires_grad
        ),
    }
    for name, build in scenes.items():
        for regularization in (0.0, 0.01):
            options = {
                "friction_anchor_beta": 0.0,
                "solve_mode": "matrix_free",
                "iterations": 8,
                "contact_regularization": regularization,
            }
            model = build(False)
            inputs = {
                "joint_q": model.joint_q.numpy().copy(),
                "joint_qd": np.zeros(model.joint_dof_count, np.float32),
                "joint_f": np.zeros(model.joint_dof_count, np.float32),
            }
            # Raised and dropped, so the rows fire their rebound target on impact.
            inputs["joint_q"][2] += 0.05
            inputs["joint_qd"][2] = -1.5
            reference, _ = _contact_scene_rollout(model, options, inputs, 24, differentiable=False, dt=1.0 / 240.0)
            result, _ = _contact_scene_rollout(build(True), options, inputs, 24, differentiable=True, dt=1.0 / 240.0)
            with test.subTest(scene=name, contact_regularization=regularization):
                test.assertGreater(float(reference.joint_qd.numpy()[2]), 0.1)
                np.testing.assert_allclose(
                    result.joint_qd.numpy(), reference.joint_qd.numpy(), rtol=1.0e-5, atol=1.0e-5
                )


def _build_limited_chain(device, *, requires_grad=True, velocity_limit=1.0e6):
    """Three hinges with position limits [-0.3, 0.4] rad and a joint velocity limit [rad/s]."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    parent = -1
    joints = []
    for i in range(3):
        body = builder.add_link(
            mass=1.0 + 0.3 * i, inertia=wp.mat33(np.diag([0.02, 0.03, 0.015])), com=wp.vec3(0.25, 0.0, 0.0)
        )
        joints.append(
            builder.add_joint_revolute(
                parent,
                body,
                parent_xform=wp.transform(wp.vec3(0.5 if parent >= 0 else 0.0, 0.0, 0.0), wp.quat_identity()),
                axis=(0.0, 1.0, 0.0),
                armature=0.01,
                limit_lower=-0.3,
                limit_upper=0.4,
                velocity_limit=velocity_limit,
            )
        )
        parent = body
    builder.add_articulation(joints)
    return builder.finalize(device=device, requires_grad=requires_grad)


_LIMIT_FORCE = np.array([5.0, -3.0, 2.0], np.float32)


def _limit_cases(device):
    cases = [("split", {"enable_joint_limits": True, "joint_limit_activation_gap": 0.2}, 1.0e6)]
    if wp.get_device(device).is_cuda:
        cases += [
            ("matrix_free", {"enable_joint_limits": True, "joint_limit_activation_gap": 0.2}, 1.0e6),
            ("matrix_free", {"enable_joint_velocity_limits": True, "velocity_limit_activation_fraction": 0.7}, 1.5),
            (
                "matrix_free",
                {
                    "enable_joint_limits": True,
                    "joint_limit_activation_gap": 0.2,
                    "enable_joint_velocity_limits": True,
                    "velocity_limit_activation_fraction": 0.7,
                    "deterministic": True,
                },
                1.5,
            ),
        ]
    return cases


def _limited_rollout(model, solver, joint_f, steps, joint_qd=None, tape=None):
    states = [model.state(requires_grad=True) for _ in range(steps + 1)]
    if joint_qd is not None:
        states[0].joint_qd.assign(joint_qd)
    newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
    control = model.control()
    control.joint_f.assign(joint_f)
    if tape is not None:
        tape.__enter__()
    for k in range(steps):
        solver.step(states[k], states[k + 1], control, None, _DT)
    if tape is not None:
        tape.__exit__(None, None, None)
    return states, control


def test_joint_limits_match_default(test, device):
    """Position-limit rows, velocity-limit rows and the velocity prescale follow the default step."""
    for solve_mode, limit_options, velocity_limit in _limit_cases(device):
        options = {"solve_mode": solve_mode, "iterations": 12, **limit_options}
        model = _build_limited_chain(device, requires_grad=False, velocity_limit=velocity_limit)
        reference = _limited_rollout(model, SolverSteel(model, **options), _LIMIT_FORCE, 40)[0][-1]
        model = _build_limited_chain(device, requires_grad=False, velocity_limit=velocity_limit)
        free = _limited_rollout(model, SolverSteel(model, solve_mode=solve_mode, iterations=12), _LIMIT_FORCE, 40)[0][
            -1
        ]
        model = _build_limited_chain(device, velocity_limit=velocity_limit)
        result = _limited_rollout(model, SolverSteel(model, differentiable=True, **options), _LIMIT_FORCE, 40)[0][-1]
        with test.subTest(**options):
            test.assertGreater(float(np.abs(free.joint_q.numpy() - reference.joint_q.numpy()).max()), 0.5)
            for attribute in ("joint_q", "joint_qd"):
                np.testing.assert_allclose(
                    getattr(result, attribute).numpy(), getattr(reference, attribute).numpy(), rtol=0.0, atol=1.0e-5
                )


def test_joint_limits_gradient_matches_finite_difference(test, device):
    """Tape against central differences with active position- and velocity-limit rows."""
    for solve_mode, limit_options, velocity_limit in _limit_cases(device):
        options = {"solve_mode": solve_mode, "iterations": 12, **limit_options}
        model = _build_limited_chain(device, velocity_limit=velocity_limit)
        solver = SolverSteel(model, differentiable=True, **options)
        # Start inside the activation band and push toward the limits.
        qd0 = np.array([1.2, -1.3, 1.1], np.float32)
        rng = np.random.default_rng(13)
        wq = rng.normal(size=3)

        def loss_of(joint_f, joint_qd, tape=None, model=model, solver=solver, wq=wq):
            states, control = _limited_rollout(model, solver, joint_f, 10, joint_qd, tape)
            final = states[-1]
            return float(wq @ final.joint_q.numpy() + wq @ final.joint_qd.numpy()), states, control

        tape = wp.Tape()
        _, states, control = loss_of(_LIMIT_FORCE, qd0, tape)
        seed = wp.array(wq.astype(np.float32), dtype=float, device=device)
        tape.backward(grads={states[-1].joint_q: seed, states[-1].joint_qd: seed})
        gradients = {"joint_f": control.joint_f.grad.numpy(), "joint_qd": states[0].joint_qd.grad.numpy()}
        rows = [int(state._steel_differentiable_buffers.contacts.row_count.numpy()[0]) for state in states[1:]]
        test.assertGreater(max(rows), 0)
        eps = 1.0e-3
        for name, gradient in gradients.items():
            for trial in range(2):
                direction = rng.normal(size=3)
                direction /= np.linalg.norm(direction)
                values = []
                for sign in (1.0, -1.0):
                    joint_f, joint_qd = _LIMIT_FORCE, qd0
                    if name == "joint_f":
                        joint_f = (_LIMIT_FORCE + sign * eps * direction).astype(np.float32)
                    else:
                        joint_qd = (qd0 + sign * eps * direction).astype(np.float32)
                    loss, perturbed, _ = loss_of(joint_f, joint_qd)
                    test.assertEqual(
                        [
                            int(state._steel_differentiable_buffers.contacts.row_count.numpy()[0])
                            for state in perturbed[1:]
                        ],
                        rows,
                    )
                    values.append(loss)
                fd = (values[0] - values[1]) / (2.0 * eps)
                with test.subTest(**options, input=name, trial=trial):
                    test.assertAlmostEqual(float(gradient @ direction), fd, delta=3.0e-3 * max(1.0, abs(fd)))


def _free_rigid_angular_options(rolling, torsional):
    return {
        "friction_anchor_beta": 0.0,
        "solve_mode": "matrix_free",
        "iterations": 8,
        "free_rigid_rolling_friction": rolling,
        "free_rigid_torsional_friction": torsional,
    }


def _rolling_inputs(shape):
    return {
        "joint_q": np.array([0.0, 0.0, 0.099 if shape == "sphere" else 0.0995, 0.0, 0.0, 0.0, 1.0], np.float32),
        "joint_qd": np.array([1.0, 0.3, 0.0, -3.0, 10.0, 4.0], np.float32),
        "joint_f": np.zeros(6, np.float32),
    }


def test_free_rigid_angular_friction_matches_default(test, device):
    """Free-body rolling pairs and spin rows follow the matrix-free route's angular rows."""
    cases = (
        ("sphere", 0.001, 0.0, 30, 1.0e-5),
        ("sphere", 0.001, 0.01, 30, 1.0e-5),
        ("sphere", 0.02, 0.0, 30, 1.0e-5),
        # A tumbling box rounds chaotically.
        ("box", 0.001, 0.0, 20, 1.0e-4),
    )
    for shape, rolling, torsional, steps, tolerance in cases:
        options = _free_rigid_angular_options(rolling, torsional)
        inputs = _rolling_inputs(shape)
        reference, _ = _contact_scene_rollout(
            _build_free_shape(device, shape, requires_grad=False), options, inputs, steps, differentiable=False
        )
        plain, _ = _contact_scene_rollout(
            _build_free_shape(device, shape, requires_grad=False),
            _free_rigid_angular_options(None, None),
            inputs,
            steps,
            differentiable=False,
        )
        result, _ = _contact_scene_rollout(
            _build_free_shape(device, shape), options, inputs, steps, differentiable=True
        )
        with test.subTest(shape=shape, rolling=rolling, torsional=torsional):
            test.assertGreater(float(np.abs(plain.joint_qd.numpy() - reference.joint_qd.numpy()).max()), 0.1)
            np.testing.assert_allclose(result.joint_qd.numpy(), reference.joint_qd.numpy(), rtol=0.0, atol=tolerance)


def test_free_rigid_angular_friction_gradient_matches_finite_difference(test, device):
    """Tape against central differences for a sphere rolling against its rolling and spin bounds."""
    for rolling, torsional in ((0.02, 0.0), (0.001, 0.01)):
        model = _build_free_shape(device, "sphere")
        solver = SolverSteel(model, differentiable=True, **_free_rigid_angular_options(rolling, torsional))
        inputs = _rolling_inputs("sphere")
        rng = np.random.default_rng(17)
        weights = rng.normal(size=6)

        def loss_of(joint_qd, tape=None, model=model, solver=solver, inputs=inputs, weights=weights):
            states, _ = _free_body_rollout(model, solver, inputs["joint_q"], joint_qd, 5, _DT, tape=tape)
            return float(weights @ states[-1].joint_qd.numpy()), states

        tape = wp.Tape()
        _, states = loss_of(inputs["joint_qd"], tape)
        tape.backward(grads={states[-1].joint_qd: wp.array(weights.astype(np.float32), dtype=float, device=device)})
        gradient = states[0].joint_qd.grad.numpy().astype(np.float64)
        eps = 1.0e-3
        for trial in range(3):
            direction = rng.normal(size=6)
            direction /= np.linalg.norm(direction)
            plus = loss_of((inputs["joint_qd"] + eps * direction).astype(np.float32))[0]
            minus = loss_of((inputs["joint_qd"] - eps * direction).astype(np.float32))[0]
            fd = (plus - minus) / (2.0 * eps)
            with test.subTest(rolling=rolling, torsional=torsional, trial=trial):
                test.assertAlmostEqual(float(gradient @ direction), fd, delta=2.0e-3 * max(1.0, abs(fd)))


def test_sleeping_option_keeps_islands_awake(test, device):
    """enable_sleeping=True is accepted and has no effect: while no island of the default step sleeps, they match."""
    options = {**_ARM_ON_BOX_OPTIONS, "enable_sleeping": True}
    inputs = _arm_on_box_inputs(_build_arm_on_box_stack(device))
    reference, _ = _contact_scene_rollout(
        _build_arm_on_box_stack(device, requires_grad=False), options, inputs, 20, differentiable=False
    )
    result, solver = _contact_scene_rollout(_build_arm_on_box_stack(device), options, inputs, 20, differentiable=True)
    test.assertIsNone(solver._sleeping)
    np.testing.assert_allclose(result.joint_qd.numpy(), reference.joint_qd.numpy(), rtol=1.0e-5, atol=2.0e-5)


_PATCH_OPTIONS = {**_ARM_ON_BOX_OPTIONS, "friction_anchor_beta": 0.2}


def test_friction_patches_match_default(test, device):
    """Persistent patch friction, with and without shared anchors, follows the default step's patches."""
    for shared in (False, True):
        options = {**_PATCH_OPTIONS, "contact_shared_anchor": shared, "contact_friction_shared_anchor": shared}
        inputs = _arm_on_box_inputs(_build_arm_on_box_stack(device, stack=False))
        with warnings.catch_warnings():
            # Patch friction warns that it picks its own friction locations under shared anchors.
            warnings.simplefilter("ignore", UserWarning)
            reference, _ = _contact_scene_rollout(
                _build_arm_on_box_stack(device, requires_grad=False, stack=False),
                options,
                inputs,
                40,
                differentiable=False,
            )
            point, _ = _contact_scene_rollout(
                _build_arm_on_box_stack(device, requires_grad=False, stack=False),
                {**options, "friction_anchor_beta": 0.0},
                inputs,
                40,
                differentiable=False,
            )
            result, _ = _contact_scene_rollout(
                _build_arm_on_box_stack(device, stack=False), options, inputs, 40, differentiable=True
            )
        with test.subTest(shared=shared):
            test.assertGreater(float(np.abs(point.joint_q.numpy() - reference.joint_q.numpy()).max()), 1.0e-3)
            for attribute in ("joint_q", "joint_qd"):
                np.testing.assert_allclose(
                    getattr(result, attribute).numpy(), getattr(reference, attribute).numpy(), rtol=0.0, atol=1.0e-4
                )


def test_friction_patches_gradient_matches_finite_difference(test, device):
    """Tape against central differences over steps whose patches carry history from earlier steps."""
    for shared in (False, True):
        options = {**_PATCH_OPTIONS, "contact_shared_anchor": shared, "contact_friction_shared_anchor": shared}
        model = _build_arm_on_box_stack(device, stack=False)
        rng = np.random.default_rng(5)
        weights_q = rng.normal(size=model.joint_coord_count)
        weights_qd = rng.normal(size=model.joint_dof_count)
        settle, steps = 30, 6
        joint_f = _arm_on_box_inputs(model)["joint_f"]

        def loss_of(
            force,
            tape=None,
            model=model,
            options=options,
            weights_q=weights_q,
            weights_qd=weights_qd,
            settle=settle,
            steps=steps,
            joint_f=joint_f,
        ):
            # A fresh solver repeats the same settling steps, so every evaluation starts from the same history.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                solver = SolverSteel(model, differentiable=True, **options)
            pipeline = newton.CollisionPipeline(model, rigid_contact_max=64)
            states = [model.state(requires_grad=True) for _ in range(settle + steps + 1)]
            newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
            settling, control = model.control(), model.control()
            settling.joint_f.assign(joint_f)
            control.joint_f.assign(force)
            rows = []
            for k in range(settle + steps):
                contacts = pipeline.contacts()
                pipeline.collide(states[k], contacts)
                live = k >= settle
                if tape is not None and live:
                    tape.__enter__()
                solver.step(states[k], states[k + 1], control if live else settling, contacts, _DT)
                if tape is not None and live:
                    tape.__exit__(None, None, None)
                if live:
                    rows.append(int(states[k + 1]._steel_differentiable_buffers.contacts.row_count.numpy()[0]))
            carried = int(np.count_nonzero(solver._friction_patches.current.source.numpy() >= 0))
            final = states[-1]
            loss = float(weights_q @ final.joint_q.numpy() + weights_qd @ final.joint_qd.numpy())
            return loss, final, control, rows, carried

        tape = wp.Tape()
        _, final, control, rows, carried = loss_of(joint_f, tape)
        test.assertGreater(carried, 0)
        tape.backward(
            grads={
                final.joint_q: wp.array(weights_q.astype(np.float32), dtype=float, device=device),
                final.joint_qd: wp.array(weights_qd.astype(np.float32), dtype=float, device=device),
            }
        )
        gradient = control.joint_f.grad.numpy().astype(np.float64)
        eps = 1.0e-2
        for trial in range(3):
            direction = rng.normal(size=model.joint_dof_count)
            direction /= np.linalg.norm(direction)
            plus = loss_of((joint_f + eps * direction).astype(np.float32))
            minus = loss_of((joint_f - eps * direction).astype(np.float32))
            fd = (plus[0] - minus[0]) / (2.0 * eps)
            with test.subTest(shared=shared, trial=trial):
                test.assertEqual(plus[3], rows)
                test.assertEqual(minus[3], rows)
                test.assertAlmostEqual(float(gradient @ direction), fd, delta=2.0e-3 * max(1.0, abs(fd)))


# Every option a production Steel profile sets, so each is either accepted here or listed as rejected.
_PRODUCTION_PROFILE = {
    "iterations": 32,
    "velocity_iterations": 0,
    "contact_regularization": 0.01,
    "articulation_constraint_max": 1024,
    "free_body_constraint_max": 4096,
    "solve_mode": "matrix_free",
    "row_schedule": "interleaved",
    "articulated_contact_response": "immediate",
    "drive_mode": "augmented",
    "baumgarte": 0.2,
    "cfm": 1.0e-6,
    "contact_warmstart": False,
    "update_mass_matrix_interval": 1,
    "contact_speculative_scale": 1.0,
    "contact_friction_gap_max": 0.001,
    "contact_friction_position_iterations": -1,
    "contact_shared_anchor": True,
    "contact_friction_shared_anchor": True,
    "friction_anchor_beta": 0.2,
    "deterministic": True,
    "enable_joint_limits": True,
    "joint_limit_activation_gap": 0.2,
    "enable_joint_velocity_limits": True,
    "velocity_limit_activation_fraction": 0.7,
    "enable_bilateral_preelimination": False,
    "contact_torsion_device": True,
    "contact_compliance": False,
    "enable_restitution": True,
    "warn_constraint_overflow": True,
    "row_watermark": True,
    "enable_sleeping": True,
    "sleep_linear_threshold": 0.05,
    "sleep_angular_threshold": 0.15,
    "sleep_quiet_time": 0.5,
    "sleep_skip_constraints": True,
    "free_rigid_rolling_friction": 0.001,
    "free_rigid_torsional_friction": 0.0,
}


def _build_gripper(device, *, requires_grad=True, round_fingers=False):
    """A lift and two prismatic fingers squeezing a free box on the ground, with contact torsion on every shape.

    The box carries the speed and depenetration bounds of production task objects.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    SolverSteel.register_custom_attributes(builder)
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.8, restitution=0.0)
    builder.add_ground_plane(cfg=cfg)
    # The rigid-body bounds production task objects carry.
    box = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0495), wp.quat_identity()),
        custom_attributes={
            "steel:rigid_body_max_linear_velocity": 2.0,
            "steel:rigid_body_max_angular_velocity": 2.0 * math.pi,
            "steel:rigid_body_max_depenetration_velocity": 0.5,
        },
    )
    builder.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05, cfg=newton.ModelBuilder.ShapeConfig(mu=0.8, density=200.0))
    lift = builder.add_link(
        mass=1.0, inertia=wp.mat33(np.eye(3) * 0.002), xform=wp.transform(wp.vec3(0.0, 0.0, 0.05), wp.quat_identity())
    )
    fingers = []
    for side in (-1.0, 1.0):
        finger = builder.add_link(
            mass=0.2,
            inertia=wp.mat33(np.eye(3) * 2.0e-4),
            xform=wp.transform(wp.vec3(0.075 * side, 0.0, 0.05), wp.quat_identity()),
        )
        if round_fingers:
            builder.add_shape_sphere(finger, radius=0.02, cfg=cfg)
        else:
            builder.add_shape_box(finger, hx=0.015, hy=0.03, hz=0.03, cfg=cfg)
        fingers.append(finger)
    joints = [
        builder.add_joint_prismatic(
            -1,
            lift,
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.05), wp.quat_identity()),
            axis=(0.0, 0.0, 1.0),
            target_ke=400.0,
            target_kd=40.0,
            limit_lower=-0.02,
            limit_upper=0.3,
            velocity_limit=0.5,
            armature=0.01,
        )
    ]
    for side, finger in zip((-1.0, 1.0), fingers, strict=True):
        joints.append(
            builder.add_joint_prismatic(
                lift,
                finger,
                parent_xform=wp.transform(wp.vec3(0.075 * side, 0.0, 0.0), wp.quat_identity()),
                axis=(-side, 0.0, 0.0),
                target_ke=300.0,
                target_kd=20.0,
                limit_lower=-0.02,
                limit_upper=0.03,
                velocity_limit=0.3,
                armature=0.01,
            )
        )
    builder.add_articulation(joints)
    model = builder.finalize(device=device, requires_grad=requires_grad)
    set_torsion_radius(model, 0.01)
    return model


def _production_pipeline(model):
    return newton.CollisionPipeline(
        model,
        rigid_contact_max=128,
        deterministic=True,
        convex_face_clip=True,
        contact_matching="sticky",
        reduce_contacts=False,
    )


def _gripper_targets(control, lift):
    targets = control.joint_target_q.numpy()
    targets[-3:] = (lift, 0.03, 0.03)
    control.joint_target_q.assign(targets)


def test_production_profile_matches_default(test, device):
    """The production option set: each differentiable step matches the default step from the same state.

    The default's grasp is sensitive to rounding when the fingers land, so the steps are compared one at a time
    along the default's trajectory, a squeeze and a lift.
    """
    model = _build_gripper(device, requires_grad=False)
    reference_solver = SolverSteel(model, **_PRODUCTION_PROFILE)
    twin = _build_gripper(device)
    solver = SolverSteel(twin, differentiable=True, **_PRODUCTION_PROFILE)
    pipeline = _production_pipeline(model)
    states = [model.state() for _ in range(161)]
    newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
    control, twin_control = model.control(), twin.control()
    errors = []
    for k in range(160):
        for each in (control, twin_control):
            _gripper_targets(each, 0.0 if k < 40 else 0.1)
        contacts = pipeline.contacts()
        pipeline.collide(states[k], contacts)
        state = twin.state(requires_grad=True)
        state.joint_q.assign(states[k].joint_q)
        state.joint_qd.assign(states[k].joint_qd)
        newton.eval_fk(twin, state.joint_q, state.joint_qd, state)
        output = twin.state(requires_grad=True)
        reference_solver.step(states[k], states[k + 1], control, contacts, 1.0 / 240.0)
        solver.step(state, output, twin_control, contacts, 1.0 / 240.0)
        errors.append(float(np.abs(output.joint_qd.numpy() - states[k + 1].joint_qd.numpy()).max()))
    # The fingers lift the box off the ground.
    test.assertGreater(float(states[-1].body_q.numpy()[0][2]), 0.09)
    test.assertLess(max(errors), 2.0e-5)


def test_production_profile_gradient_matches_finite_difference(test, device):
    """Tape against central differences of the lift and finger targets, after a grasp, with the production options.

    Round fingers keep the grasp to one contact per finger; flat fingers make the force split indeterminate.
    """
    model = _build_gripper(device, round_fingers=True)
    rng = np.random.default_rng(2)
    weights_q = rng.normal(size=model.joint_coord_count)
    weights_qd = rng.normal(size=model.joint_dof_count)
    settle, steps = 30, 4
    base = np.array([0.1, 0.03, 0.03], np.float32)

    def loss_of(target, tape=None):
        solver = SolverSteel(model, differentiable=True, **_PRODUCTION_PROFILE)
        pipeline = _production_pipeline(model)
        states = [model.state(requires_grad=True) for _ in range(settle + steps + 1)]
        newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
        settling, control = model.control(), model.control()
        _gripper_targets(settling, 0.0)
        targets = control.joint_target_q.numpy()
        targets[-3:] = target
        control.joint_target_q.assign(targets)
        rows = []
        for k in range(settle + steps):
            contacts = pipeline.contacts()
            pipeline.collide(states[k], contacts)
            live = k >= settle
            if tape is not None and live:
                tape.__enter__()
            solver.step(states[k], states[k + 1], control if live else settling, contacts, 1.0 / 240.0)
            if tape is not None and live:
                tape.__exit__(None, None, None)
            if live:
                rows.append(int(states[k + 1]._steel_differentiable_buffers.contacts.row_count.numpy()[0]))
        final = states[-1]
        return float(weights_q @ final.joint_q.numpy() + weights_qd @ final.joint_qd.numpy()), final, control, rows

    tape = wp.Tape()
    _, final, control, rows = loss_of(base, tape)
    tape.backward(
        grads={
            final.joint_q: wp.array(weights_q.astype(np.float32), dtype=float, device=device),
            final.joint_qd: wp.array(weights_qd.astype(np.float32), dtype=float, device=device),
        }
    )
    gradient = control.joint_target_q.grad.numpy()[-3:].astype(np.float64)
    test.assertGreater(min(rows), 0)
    for trial in range(2):
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        # Contact and stick/slip events make isolated step sizes see a different regime: take the median.
        differences = []
        for eps in (3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2):
            plus = loss_of((base + eps * direction).astype(np.float32))[0]
            minus = loss_of((base - eps * direction).astype(np.float32))[0]
            differences.append((plus - minus) / (2.0 * eps))
        fd = float(np.median(differences))
        with test.subTest(trial=trial):
            test.assertAlmostEqual(float(gradient @ direction), fd, delta=2.0e-3 * max(1.0, abs(fd)))


def _build_speed_limited_boxes(device, *, requires_grad=True):
    """Two free boxes with a 1 m/s and 2 rad/s speed limit, one on the ground and one in the air."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    SolverSteel.register_custom_attributes(builder)
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.3, restitution=0.0)
    builder.add_ground_plane(cfg=cfg)
    for height in (0.0995, 1.0):
        body = builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 2.0 * height, height), wp.quat_identity()),
            custom_attributes={
                "steel:rigid_body_max_linear_velocity": 1.0,
                "steel:rigid_body_max_angular_velocity": 2.0,
            },
        )
        builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=cfg)
    return builder.finalize(device=device, requires_grad=requires_grad)


def _speed_limited_inputs(model):
    joint_qd = np.tile(np.array([3.0, -0.4, 0.0, 0.5, -1.0, 4.0], np.float32), 2)
    return {"joint_q": model.joint_q.numpy().copy(), "joint_qd": joint_qd, "joint_f": np.zeros(12, np.float32)}


def test_rigid_velocity_limits_match_default(test, device):
    """Free bodies' linear and angular speed limits are solved as the matrix-free route's velocity-limit rows."""
    for fraction in (0.0, 0.7):
        options = {
            "friction_anchor_beta": 0.0,
            "solve_mode": "matrix_free",
            "iterations": 8,
            "velocity_limit_activation_fraction": fraction,
            "articulation_constraint_max": 64,
        }
        inputs = _speed_limited_inputs(_build_speed_limited_boxes(device))
        reference, solver = _contact_scene_rollout(
            _build_speed_limited_boxes(device, requires_grad=False), options, inputs, 10, differentiable=False
        )
        test.assertTrue(solver._has_rigid_body_velocity_limits)
        result, _ = _contact_scene_rollout(_build_speed_limited_boxes(device), options, inputs, 10, differentiable=True)
        with test.subTest(velocity_limit_activation_fraction=fraction):
            # Both boxes end within their limits, from three times the linear one.
            test.assertLess(float(np.abs(reference.joint_qd.numpy()[:3]).max()), 1.0 + 1.0e-4)
            np.testing.assert_allclose(result.joint_qd.numpy(), reference.joint_qd.numpy(), rtol=0.0, atol=1.0e-5)


def test_rigid_velocity_limits_gradient_matches_finite_difference(test, device):
    """Tape against central differences with active free-body speed-limit rows."""
    model = _build_speed_limited_boxes(device)
    options = {
        "friction_anchor_beta": 0.0,
        "solve_mode": "matrix_free",
        "iterations": 8,
        "articulation_constraint_max": 64,
    }
    solver = SolverSteel(model, differentiable=True, **options)
    inputs = _speed_limited_inputs(model)
    rng = np.random.default_rng(23)
    weights = rng.normal(size=12)

    def loss_of(joint_qd, tape=None):
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=64)
        states = [model.state(requires_grad=True) for _ in range(5)]
        states[0].joint_qd.assign(joint_qd)
        newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
        for k in range(4):
            contacts = pipeline.contacts()
            pipeline.collide(states[k], contacts)
            if tape is not None:
                tape.__enter__()
            solver.step(states[k], states[k + 1], None, contacts, _DT)
            if tape is not None:
                tape.__exit__(None, None, None)
        return float(weights @ states[-1].joint_qd.numpy()), states

    tape = wp.Tape()
    _, states = loss_of(inputs["joint_qd"], tape)
    tape.backward(grads={states[-1].joint_qd: wp.array(weights.astype(np.float32), dtype=float, device=device)})
    gradient = states[0].joint_qd.grad.numpy().astype(np.float64)
    eps = 1.0e-3
    for trial in range(3):
        direction = rng.normal(size=12)
        direction /= np.linalg.norm(direction)
        plus = loss_of((inputs["joint_qd"] + eps * direction).astype(np.float32))[0]
        minus = loss_of((inputs["joint_qd"] - eps * direction).astype(np.float32))[0]
        fd = (plus - minus) / (2.0 * eps)
        with test.subTest(trial=trial):
            test.assertAlmostEqual(float(gradient @ direction), fd, delta=2.0e-3 * max(1.0, abs(fd)))


def _build_kinematic_belt(device, *, requires_grad=True):
    """A free box on a kinematic belt."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.5, gap=0.0, restitution=0.0, density=500.0)
    belt = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, -0.05), wp.quat_identity()))
    builder.add_shape_box(belt, hx=2.0, hy=2.0, hz=0.05, cfg=cfg)
    box = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.05), wp.quat_identity()))
    builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.05, cfg=cfg)
    builder.body_flags[belt] = int(newton.BodyFlags.KINEMATIC)
    return builder.finalize(device=device, requires_grad=requires_grad)


def _kinematic_inputs(model):
    joint_qd = np.zeros(model.joint_dof_count, np.float32)
    # The belt moves at 0.5 m/s along x.
    joint_qd[0] = 0.5
    return {"joint_q": model.joint_q.numpy().copy(), "joint_qd": joint_qd, "joint_f": np.zeros_like(joint_qd)}


def test_kinematic_bodies_match_default(test, device):
    """Kinematic bodies drive contact rows through their prescribed velocity, as in the default step."""
    cases = [("split", 0.0), ("split", 0.2)]
    if wp.get_device(device).is_cuda:
        cases += [("matrix_free", 0.0), ("matrix_free", 0.2)]
    for solve_mode, beta in cases:
        options = {
            "friction_anchor_beta": beta,
            "solve_mode": solve_mode,
            "iterations": 32,
            "articulation_constraint_max": 64,
        }
        inputs = _kinematic_inputs(_build_kinematic_belt(device))
        reference, _ = _contact_scene_rollout(
            _build_kinematic_belt(device, requires_grad=False),
            options,
            inputs,
            60,
            differentiable=False,
            dt=1.0 / 240.0,
        )
        result, _ = _contact_scene_rollout(
            _build_kinematic_belt(device), options, inputs, 60, differentiable=True, dt=1.0 / 240.0
        )
        with test.subTest(solve_mode=solve_mode, friction_anchor_beta=beta):
            # The belt drags the box to its speed.
            test.assertAlmostEqual(float(reference.joint_qd.numpy()[6]), 0.5, delta=1.0e-3)
            np.testing.assert_allclose(result.joint_qd.numpy(), reference.joint_qd.numpy(), rtol=0.0, atol=1.0e-5)


def test_kinematic_velocity_gradient_matches_finite_difference(test, device):
    """Tape against central differences of the box's and the kinematic belt's initial velocities.

    The contacts are frozen: a turning belt also turns the contact normals, which are stop-gradient.
    """
    model = _build_kinematic_belt(device)
    solver = SolverSteel(model, differentiable=True, friction_anchor_beta=0.0, iterations=32)
    inputs = _kinematic_inputs(model)
    rng = np.random.default_rng(29)
    weights = rng.normal(size=model.joint_dof_count)

    _, contacts = _free_body_rollout(model, solver, inputs["joint_q"], inputs["joint_qd"], 5, 1.0 / 240.0)

    def loss_of(joint_qd, tape=None):
        states, _ = _free_body_rollout(
            model, solver, inputs["joint_q"], joint_qd, 5, 1.0 / 240.0, contacts=contacts, tape=tape
        )
        return float(weights @ states[-1].joint_qd.numpy()), states

    tape = wp.Tape()
    _, states = loss_of(inputs["joint_qd"], tape)
    tape.backward(grads={states[-1].joint_qd: wp.array(weights.astype(np.float32), dtype=float, device=device)})
    gradient = states[0].joint_qd.grad.numpy().astype(np.float64)
    # The belt's velocity reaches the box only through the friction rows' target.
    test.assertGreater(abs(float(gradient[0])), 0.1)
    eps = 1.0e-3
    for trial in range(3):
        direction = rng.normal(size=model.joint_dof_count)
        direction /= np.linalg.norm(direction)
        plus = loss_of((inputs["joint_qd"] + eps * direction).astype(np.float32))[0]
        minus = loss_of((inputs["joint_qd"] - eps * direction).astype(np.float32))[0]
        fd = (plus - minus) / (2.0 * eps)
        with test.subTest(trial=trial):
            test.assertAlmostEqual(float(gradient @ direction), fd, delta=2.0e-3 * max(1.0, abs(fd)))


def test_deterministic_gradients_repeat(test, device):
    """With deterministic=True, repeated rollouts from the same state give bitwise-identical gradients."""
    model = _build_gripper(device)
    gradients = []
    for _ in range(3):
        solver = SolverSteel(model, differentiable=True, **_PRODUCTION_PROFILE)
        pipeline = _production_pipeline(model)
        states = [model.state(requires_grad=True) for _ in range(21)]
        newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
        control = model.control()
        _gripper_targets(control, 0.0)
        tape = wp.Tape()
        for k in range(20):
            contacts = pipeline.contacts()
            pipeline.collide(states[k], contacts)
            with tape:
                solver.step(states[k], states[k + 1], control, contacts, 1.0 / 240.0)
        tape.backward(grads={states[-1].joint_qd: wp.ones(model.joint_dof_count, dtype=float, device=device)})
        gradients.append(
            np.concatenate(
                [states[0].joint_q.grad.numpy(), states[0].joint_qd.grad.numpy(), control.joint_target_q.grad.numpy()]
            )
        )
    test.assertTrue(np.all(np.isfinite(gradients[0])))
    test.assertGreater(float(np.abs(gradients[0]).max()), 0.0)
    for repeat in gradients[1:]:
        np.testing.assert_array_equal(repeat, gradients[0])


_OBSERVABLE_KINDS = {
    newton.solvers.SolverBase.ObservableKind.CONTACT_F,
    newton.solvers.SolverBase.ObservableKind.BODY_PARENT_F,
}


def _observed_rollout(model, options, inputs, steps, *, differentiable, tape=None, dt=_DT):
    """Step a contact scene and return each step's state, CONTACT_F and BODY_PARENT_F observables."""
    solver = SolverSteel(model, differentiable=differentiable, **options)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=64)
    contacts = pipeline.contacts()
    states = [model.state(requires_grad=True) for _ in range(steps + 1)]
    states[0].joint_q.assign(inputs["joint_q"])
    states[0].joint_qd.assign(inputs["joint_qd"])
    newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
    control = model.control()
    control.joint_f.assign(inputs["joint_f"])
    records = []
    for k in range(steps):
        pipeline.collide(states[k], contacts)
        # Fresh buffers per step so the tape seeds only the last step's observables.
        observables = solver.observables(_OBSERVABLE_KINDS, contacts=contacts)
        with tape if tape is not None else contextlib.nullcontext():
            solver.step(states[k], states[k + 1], control, contacts, dt, observables=observables)
        records.append((observables.contact_f.numpy().copy(), observables.body_parent_f.numpy().copy()))
    return states, control, observables, records


def test_observables_match_default(test, device):
    """CONTACT_F and BODY_PARENT_F of the differentiable step follow the default step's."""
    scenes = [("box_chain", lambda requires_grad: _build_box_chain_on_plane(device, requires_grad=requires_grad), {})]
    if wp.get_device(device).is_cuda:
        scenes.append(
            (
                "arm_on_boxes",
                lambda requires_grad: _build_arm_on_box_stack(device, requires_grad=requires_grad),
                {"solve_mode": "matrix_free", "articulation_constraint_max": 64, "contact_regularization": 0.01},
            )
        )
    for name, build, extra in scenes:
        options = {"friction_anchor_beta": 0.0, "iterations": 16, **extra}
        model = build(False)
        inputs = _arm_on_box_inputs(model) if name == "arm_on_boxes" else _contact_inputs(model)
        _, _, _, reference = _observed_rollout(model, options, inputs, 6, differentiable=False)
        _, _, _, result = _observed_rollout(build(True), options, inputs, 6, differentiable=True)
        for k, ((contact_ref, parent_ref), (contact, parent)) in enumerate(zip(reference, result, strict=True)):
            with test.subTest(scene=name, step=k):
                test.assertGreater(float(np.abs(contact_ref).max()), 0.0)
                np.testing.assert_allclose(contact, contact_ref, rtol=1.0e-4, atol=1.0e-3)
                np.testing.assert_allclose(parent, parent_ref, rtol=1.0e-4, atol=1.0e-3)


def test_observables_gradient_matches_finite_difference(test, device):
    """Tape against central differences of a loss on CONTACT_F and BODY_PARENT_F."""
    model = _build_box_chain_on_plane(device)
    inputs = _contact_inputs(model)
    options = {"friction_anchor_beta": 0.0, "iterations": 16}
    rng = np.random.default_rng(31)
    contact_weights = rng.normal(size=(64, 6)).astype(np.float32)
    parent_weights = rng.normal(size=(model.body_count, 6)).astype(np.float32)

    def loss_of(joint_f, tape=None):
        values = {**inputs, "joint_f": joint_f}
        _, control, observables, records = _observed_rollout(model, options, values, 3, differentiable=True, tape=tape)
        contact, parent = records[-1]
        return float(np.sum(contact_weights * contact) + np.sum(parent_weights * parent)), control, observables

    tape = wp.Tape()
    _, control, observables = loss_of(inputs["joint_f"], tape)
    tape.backward(
        grads={
            observables.contact_f: wp.array(contact_weights, dtype=wp.spatial_vector, device=device),
            observables.body_parent_f: wp.array(parent_weights, dtype=wp.spatial_vector, device=device),
        }
    )
    gradient = control.joint_f.grad.numpy().astype(np.float64)
    test.assertGreater(float(np.abs(gradient).max()), 0.0)
    # Large step: the float32 loss is too noisy for central differences below this.
    eps = 1.0e-1
    for trial in range(3):
        direction = rng.normal(size=model.joint_dof_count)
        direction /= np.linalg.norm(direction)
        plus = loss_of((inputs["joint_f"] + eps * direction).astype(np.float32))[0]
        minus = loss_of((inputs["joint_f"] - eps * direction).astype(np.float32))[0]
        fd = (plus - minus) / (2.0 * eps)
        with test.subTest(trial=trial):
            test.assertAlmostEqual(float(gradient @ direction), fd, delta=5.0e-3 * max(1.0, abs(fd)))


def test_taped_kernels_keep_adjoints(test, device):
    """Every launch the step records has an adjoint, although the Steel kernel modules disable them."""
    for model in (_build_box_chain_on_plane(device), _build_free_shape(device, "box")):
        pipeline = newton.CollisionPipeline(model, rigid_contact_max=16)
        state, output = model.state(requires_grad=True), model.state(requires_grad=True)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
        solver = SolverSteel(model, differentiable=True, friction_anchor_beta=0.0)
        tape = wp.Tape()
        with tape:
            solver.step(state, output, None, contacts, _DT)
        kernels = [launch[0] for launch in tape.launches if not callable(launch)]
        test.assertGreater(len(kernels), 0)
        for kernel in kernels:
            enabled = kernel.options.get("enable_backward", kernel.module.options.get("enable_backward"))
            with test.subTest(kernel=kernel.key):
                test.assertTrue(enabled)


class TestSteelDifferentiable(unittest.TestCase):
    pass


for _device in get_test_devices():
    for _test in (
        test_rejects_unsupported_configs,
        test_forward_matches_default,
        test_gradient_matches_finite_difference,
        test_gradient_matches_analytic_pendulum,
        test_repeated_backward_is_stable,
        test_contact_forward_matches_default,
        test_contact_gradient_matches_finite_difference,
        test_contact_rejects_unsupported_configs,
        test_friction_gradient_matches_finite_difference,
        test_sphere_multistep_gradient_flow,
        test_sphere_friction_matches_default_and_finite_difference,
        test_repeated_rollouts_reuse_buffers,
        test_restitution_bounce_matches_default_and_finite_difference,
        test_sliding_box_velocity_jacobian,
        test_frictional_impact_matches_default,
        test_contact_law_options,
        test_isotropic_friction_orientation_gradient,
        test_friction_pair_off_diagonal_gradient,
        test_friction_pair_gradient_scale_invariance,
        test_rank_one_friction_pose_gradient,
        test_friction_pair_shift_dominated_slide,
        test_pgs_kernel_selection,
        test_free_body_rows_use_dense_capacity,
        test_smooth_contact_rejects_unsupported_configs,
        test_smooth_contact_zero_gap_activation_gradient,
        test_smooth_contact_damped_zero_gap_gradient_matches_finite_difference,
        test_smooth_contact_bounce_gradient_converges_across_eps,
        test_taped_kernels_keep_adjoints,
        test_runtime_disabled_tree_joint_is_reported,
        test_unarticulated_body_contact_matches_default,
        test_joint_limits_match_default,
        test_joint_limits_gradient_matches_finite_difference,
        test_kinematic_bodies_match_default,
        test_kinematic_velocity_gradient_matches_finite_difference,
        test_observables_match_default,
        test_observables_gradient_matches_finite_difference,
    ):
        add_function_test(TestSteelDifferentiable, _test.__name__, _test, devices=[_device])
    if _device.is_cuda:
        # solve_mode="matrix_free" and its features run on CUDA only.
        for _test in (
            test_collide_done_event,
            test_matrix_free_contacts_match_default,
            test_contact_torsion_matches_default,
            test_contact_torsion_gradient_matches_finite_difference,
            test_mixed_world_contacts_match_default,
            test_deterministic_rollouts_repeat,
            test_mixed_world_gradient_matches_finite_difference,
            test_matrix_free_restitution_matches_default,
            test_free_rigid_angular_friction_matches_default,
            test_free_rigid_angular_friction_gradient_matches_finite_difference,
            test_sleeping_option_keeps_islands_awake,
            test_friction_patches_match_default,
            test_friction_patches_gradient_matches_finite_difference,
            test_production_profile_matches_default,
            test_production_profile_gradient_matches_finite_difference,
            test_rigid_velocity_limits_match_default,
            test_rigid_velocity_limits_gradient_matches_finite_difference,
            test_deterministic_gradients_repeat,
        ):
            add_function_test(TestSteelDifferentiable, _test.__name__, _test, devices=[_device])


if __name__ == "__main__":
    unittest.main()
