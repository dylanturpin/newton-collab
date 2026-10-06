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


_CONTACT_OPTIONS = {"friction_anchor_beta": 0.0, "enable_contact_friction": False, "enable_restitution": False}


def _build_box_chain_on_plane(device, *, restitution=0.0, free_box=False, extra_free_box=False):
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
    return builder.finalize(device=device, requires_grad=True)


def _contact_rollout(model, pipeline, inputs, steps, *, differentiable, use_contacts=True, tape=None):
    """Collide every step outside the tape; return the loss inputs and per-step contact counts."""
    solver = SolverFeatherPGS(model, differentiable=differentiable, pgs_iterations=8, **_CONTACT_OPTIONS)
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
    reference, _, reference_counts = _contact_rollout(model, pipeline, inputs, 20, differentiable=False)
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
                rows = state._fpgs_differentiable_buffers.contacts
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
        {**_CONTACT_OPTIONS, "enable_contact_friction": True, "contact_friction_shared_anchor": True},
        {**_CONTACT_OPTIONS, "pgs_contact_regularization": 0.1},
    ]
    if wp.get_device(device).is_cuda:
        unsupported.append({**_CONTACT_OPTIONS, "pgs_mode": "matrix_free"})
    for options in unsupported:
        solver = SolverFeatherPGS(model, differentiable=True, **options)
        with test.subTest(**options), test.assertRaisesRegex(NotImplementedError, "contacts require"):
            solver.step(state, model.state(requires_grad=True), None, contacts, _DT)
    for model, options in ((_build_box_chain_on_plane(device, extra_free_box=True), _CONTACT_OPTIONS),):
        solver = SolverFeatherPGS(model, differentiable=True, **options)
        state = model.state(requires_grad=True)
        contacts = newton.CollisionPipeline(model, rigid_contact_max=64).contacts()
        with test.assertRaisesRegex(NotImplementedError, "contacts require"):
            solver.step(state, model.state(requires_grad=True), None, contacts, _DT)


def _build_slider_on_plane(device):
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
    return builder.finalize(device=device, requires_grad=True)


def test_friction_gradient_matches_finite_difference(test, device):
    model = _build_slider_on_plane(device)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=16)
    steps = 5
    eps = 1.0e-3
    rng = np.random.default_rng(7)
    wq = wp.array(rng.normal(size=3).astype(np.float32), device=device)
    wqd = wp.array(rng.normal(size=3).astype(np.float32), device=device)
    options = {**_CONTACT_OPTIONS, "enable_contact_friction": True}

    def rollout(values, *, differentiable, tape=None):
        solver = SolverFeatherPGS(model, differentiable=differentiable, pgs_iterations=8, **options)
        states = [model.state(requires_grad=True) for _ in range(steps + 1)]
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
            rows = state._fpgs_differentiable_buffers.contacts
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
    """FeatherPGS counterpart of test_differentiable_contacts.test_multistep_gradient_flow."""
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
        solver = SolverFeatherPGS(model, differentiable=True, **_CONTACT_OPTIONS)
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
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=16)
    options = {**_CONTACT_OPTIONS, "enable_contact_friction": True}
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
        solver = SolverFeatherPGS(model, differentiable=differentiable, pgs_iterations=8, **options)
        states = [model.state(requires_grad=True) for _ in range(steps + 1)]
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
        "slider_friction": (_build_slider_on_plane(device), 16, {**_CONTACT_OPTIONS, "enable_contact_friction": True}),
    }
    steps = 6
    for name, (model, contact_max, options) in scenes.items():
        solver = SolverFeatherPGS(model, differentiable=True, **options)
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
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=8)
    options = {"friction_anchor_beta": 0.0, "enable_restitution": True, "pgs_iterations": 8}
    steps = 20
    dt = 1.0 / 240.0
    wq = wp.array(np.array([1.0, 0.5, 2.0, 0.0, 0.0, 0.0, 0.0], np.float32), device=device)
    wqd = wp.array(np.array([0.3, 0.2, 1.0, 0.1, 0.1, 0.1], np.float32), device=device)
    qd0 = np.array([0.3, 0.0, -2.0, 0.0, 0.0, 0.0], np.float32)

    def rollout(qd, *, differentiable, tape=None):
        solver = SolverFeatherPGS(model, differentiable=differentiable, **options)
        states = [model.state(requires_grad=True) for _ in range(steps + 1)]
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
                rows = states[k + 1]._fpgs_differentiable_buffers.contacts
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


class TestFeatherPGSDifferentiable(unittest.TestCase):
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
    ):
        add_function_test(TestFeatherPGSDifferentiable, _test.__name__, _test, devices=[_device])


if __name__ == "__main__":
    unittest.main()
