# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverSteel, steel
from newton.tests.steel.test_differentiable import (
    _CONTACT_OPTIONS,
    _DT,
    _build_box_chain_on_plane,
    _build_chain,
    _build_slider_on_plane,
    _random_inputs,
    _rollout,
    _weighted_loss,
)
from newton.tests.unittest_utils import add_function_test, get_test_devices

_FRICTION_OPTIONS = {**_CONTACT_OPTIONS, "contact_friction_gap_max": None}


def _contact_gradients(
    model, inputs, steps, *, iterations, adjoint, options, perturb=None, tiled_sweeps=True, regularization=1.0e-6
):
    """Loss and input gradients of a contact rollout with the PGS solve differentiated by ``adjoint``."""
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=64)
    solver = SolverSteel(
        model,
        differentiable=True,
        iterations=iterations,
        differentiable_options=steel.DifferentiableOptions(
            pgs_adjoint=adjoint, truncated_sweeps=iterations, implicit_regularization=regularization
        ),
        **options,
    )
    plan = solver._differentiable_step
    plan._tiled_sweeps = tiled_sweeps
    values = dict(inputs)
    if perturb is not None:
        key, delta = perturb
        values[key] = values[key] + delta
    states = [model.state(requires_grad=True) for _ in range(steps + 1)]
    states[0].joint_q.assign(values["joint_q"])
    states[0].joint_qd.assign(values["joint_qd"])
    newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])
    control = model.control()
    control.joint_f.assign(values["joint_f"])
    contacts = pipeline.contacts()
    rng = np.random.default_rng(3)
    device = model.device
    wq = wp.array(rng.normal(size=model.joint_coord_count).astype(np.float32), device=device)
    wqd = wp.array(rng.normal(size=model.joint_dof_count).astype(np.float32), device=device)
    tape = wp.Tape()
    for k in range(steps):
        pipeline.collide(states[k], contacts)
        with tape:
            solver.step(states[k], states[k + 1], control, contacts, _DT)
    loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
    with tape:
        wp.launch(
            _weighted_loss,
            dim=max(model.joint_coord_count, model.joint_dof_count),
            inputs=[states[-1].joint_q, wq, states[-1].joint_qd, wqd],
            outputs=[loss],
            device=device,
        )
    tape.backward(loss)
    gradients = {
        "joint_q": states[0].joint_q.grad.numpy().copy(),
        "joint_qd": states[0].joint_qd.grad.numpy().copy(),
        "joint_f": control.joint_f.grad.numpy().copy(),
    }
    return float(loss.numpy()[0]), gradients, plan


def _slider_inputs(push):
    return {
        "joint_q": np.zeros(3, np.float32),
        "joint_qd": np.array([0.02, -0.01, 0.0], np.float32),
        "joint_f": np.array([push, 0.4 * push, 0.0], np.float32),
    }


def _assert_gradients_close(test, actual, expected, rtol):
    # One scale for all inputs: sticking friction leaves some input gradients at rounding level.
    scale = max(*(float(np.linalg.norm(value)) for value in expected.values()), 1.0e-6)
    for key, value in expected.items():
        with test.subTest(input=key):
            test.assertLess(float(np.linalg.norm(actual[key] - value)) / scale, rtol)


def test_implicit_matches_converged_unrolled(test, device):
    cases = [
        ("box chain", _build_box_chain_on_plane, None, _CONTACT_OPTIONS),
        ("stick", _build_slider_on_plane, _slider_inputs(2.0), _FRICTION_OPTIONS),
        ("slide", _build_slider_on_plane, _slider_inputs(60.0), _FRICTION_OPTIONS),
    ]
    for name, build, case_inputs, options in cases:
        model = build(device)
        inputs = case_inputs
        if inputs is None:
            rng = np.random.default_rng(0)
            inputs = {
                "joint_q": model.joint_q.numpy().copy(),
                "joint_qd": (0.05 * rng.normal(size=model.joint_dof_count)).astype(np.float32),
                "joint_f": (0.5 * rng.normal(size=model.joint_dof_count)).astype(np.float32),
            }
        _, unrolled, _ = _contact_gradients(model, inputs, 4, iterations=96, adjoint="unrolled", options=options)
        _, implicit, plan = _contact_gradients(model, inputs, 4, iterations=96, adjoint="implicit", options=options)
        with test.subTest(case=name):
            test.assertEqual(plan.implicit_status.numpy().tolist(), [0, 0])
            _assert_gradients_close(test, implicit, unrolled, 2.0e-2)


def test_implicit_sliding_gradient_matches_finite_difference(test, device):
    model = _build_slider_on_plane(device)
    inputs = _slider_inputs(60.0)
    eps = 1.0e-3
    rng = np.random.default_rng(11)
    _, gradients, _ = _contact_gradients(model, inputs, 4, iterations=48, adjoint="implicit", options=_FRICTION_OPTIONS)
    for key, gradient in gradients.items():
        direction = rng.normal(size=gradient.shape[0]).astype(np.float32)
        direction /= np.linalg.norm(direction)
        plus, _, _ = _contact_gradients(
            model,
            inputs,
            4,
            iterations=48,
            adjoint="implicit",
            options=_FRICTION_OPTIONS,
            perturb=(key, eps * direction),
        )
        minus, _, _ = _contact_gradients(
            model,
            inputs,
            4,
            iterations=48,
            adjoint="implicit",
            options=_FRICTION_OPTIONS,
            perturb=(key, -eps * direction),
        )
        fd = (plus - minus) / (2.0 * eps)
        with test.subTest(input=key):
            test.assertAlmostEqual(float(gradient @ direction), fd, delta=3.0e-3 * max(1.0, abs(fd)))


def test_implicit_rejects_unlinearized_laws(test, device):
    model = _build_slider_on_plane(device)
    for name, options in (
        ("friction patches", {**_FRICTION_OPTIONS, "friction_anchor_beta": None}),
        ("joint velocity limits", {**_FRICTION_OPTIONS, "enable_joint_velocity_limits": True}),
    ):
        with test.subTest(case=name), test.assertRaises(NotImplementedError):
            SolverSteel(
                model,
                differentiable=True,
                differentiable_options=steel.DifferentiableOptions(pgs_adjoint="implicit"),
                **options,
            )


def test_tiled_implicit_matches_serial(test, device):
    model = _build_box_chain_on_plane(device)
    rng = np.random.default_rng(1)
    inputs = {
        "joint_q": model.joint_q.numpy().copy(),
        "joint_qd": (0.05 * rng.normal(size=model.joint_dof_count)).astype(np.float32),
        "joint_f": (0.5 * rng.normal(size=model.joint_dof_count)).astype(np.float32),
    }
    _, serial, _ = _contact_gradients(
        model, inputs, 4, iterations=48, adjoint="implicit", options=_CONTACT_OPTIONS, tiled_sweeps=False
    )
    _, tiled, _ = _contact_gradients(
        model, inputs, 4, iterations=48, adjoint="implicit", options=_CONTACT_OPTIONS, tiled_sweeps=True
    )
    _assert_gradients_close(test, tiled, serial, 1.0e-3)


def test_truncated_over_every_sweep_matches_unrolled(test, device):
    """Truncating at the iteration count differentiates every sweep, exactly as the unrolled adjoint."""
    model = _build_box_chain_on_plane(device)
    rng = np.random.default_rng(2)
    inputs = {
        "joint_q": model.joint_q.numpy().copy(),
        "joint_qd": (0.05 * rng.normal(size=model.joint_dof_count)).astype(np.float32),
        "joint_f": (0.5 * rng.normal(size=model.joint_dof_count)).astype(np.float32),
    }
    _, unrolled, _ = _contact_gradients(model, inputs, 3, iterations=16, adjoint="unrolled", options=_CONTACT_OPTIONS)
    _, truncated, _ = _contact_gradients(model, inputs, 3, iterations=16, adjoint="truncated", options=_CONTACT_OPTIONS)
    # Equal up to the run-to-run rounding of the backward's atomic adds.
    _assert_gradients_close(test, truncated, unrolled, 1.0e-5)


def test_parallel_cholesky_adjoint_matches_serial(test, device):
    model = _build_chain(device, drives=True)
    inputs = _random_inputs(model, 5)
    gradients = {}
    for parallel in (False, True):
        solver = SolverSteel(model, differentiable=True)
        solver._differentiable_step._parallel_cholesky_adjoint = parallel
        tape = wp.Tape()
        final, initial, control = _rollout(model, solver, inputs, 6, tape=tape)
        loss = wp.zeros(1, dtype=float, device=device, requires_grad=True)
        weights = wp.array(np.linspace(-1.0, 1.0, model.joint_dof_count).astype(np.float32), device=device)
        with tape:
            wp.launch(
                _weighted_loss,
                dim=model.joint_dof_count,
                inputs=[final.joint_q, weights, final.joint_qd, weights],
                outputs=[loss],
                device=device,
            )
        tape.backward(loss)
        gradients[parallel] = {
            "joint_q": initial.joint_q.grad.numpy().copy(),
            "joint_qd": initial.joint_qd.grad.numpy().copy(),
            "joint_target_q": control.joint_target_q.grad.numpy().copy(),
        }
    _assert_gradients_close(test, gradients[True], gradients[False], 1.0e-4)


class TestSteelDifferentiableImplicit(unittest.TestCase):
    pass


for _device in get_test_devices():
    for _test in (
        test_implicit_matches_converged_unrolled,
        test_implicit_sliding_gradient_matches_finite_difference,
        test_implicit_rejects_unlinearized_laws,
        test_tiled_implicit_matches_serial,
        test_truncated_over_every_sweep_matches_unrolled,
        test_parallel_cholesky_adjoint_matches_serial,
    ):
        add_function_test(TestSteelDifferentiableImplicit, _test.__name__, _test, devices=[_device])


if __name__ == "__main__":
    unittest.main()
