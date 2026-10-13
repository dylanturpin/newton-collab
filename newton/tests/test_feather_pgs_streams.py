# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Parallel size-group streams and double buffering of SolverFeatherPGS."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.test_feather_pgs_connect import _build_four_bar
from newton.tests.test_feather_pgs_contact_torsion import set_torsion_radius
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices


def _scene(device, worlds=2):
    """Two articulation sizes (a 2-link and a 3-link chain) and a box, falling onto the ground."""
    template = newton.ModelBuilder()
    for x, links in ((0.0, 2), (0.8, 3)):
        parent = -1
        joints = []
        for i in range(links):
            link = template.add_link(xform=wp.transform(wp.vec3(x + 0.25 * i, 0.0, 0.3), wp.quat_identity()))
            template.add_shape_capsule(
                link,
                radius=0.04,
                half_height=0.1,
                xform=wp.transform(wp.vec3(), wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.5 * np.pi)),
            )
            joints.append(
                template.add_joint_revolute(
                    parent,
                    link,
                    axis=wp.vec3(0.0, 1.0, 0.0),
                    parent_xform=wp.transform(
                        wp.vec3(x, 0.0, 0.3) if parent < 0 else wp.vec3(0.25, 0.0, 0.0), wp.quat_identity()
                    ),
                    child_xform=wp.transform_identity(),
                    limit_lower=-1.0,
                    limit_upper=1.0,
                )
            )
            parent = link
        template.add_articulation(joints)
    box = template.add_body(xform=wp.transform(wp.vec3(0.4, 0.4, 0.3), wp.quat_rpy(0.2, 0.1, 0.0)))
    template.add_shape_box(box, hx=0.06, hy=0.06, hz=0.06)
    builder = newton.ModelBuilder()
    builder.replicate(template, worlds)
    builder.add_ground_plane()
    return builder.finalize(device=device)


def _run(model, steps=60, capture=False, **solver_kwargs):
    solver = SolverFeatherPGS(
        model, enable_joint_limits=True, friction_anchor_beta=0.0, articulation_constraint_max=96, **solver_kwargs
    )
    pipeline = newton.CollisionPipeline(model, deterministic=True)
    contacts = pipeline.contacts()
    state_0, state_1 = model.state(), model.state()
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    control = model.control()

    def substep():
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
        for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
            wp.copy(getattr(state_0, name), getattr(state_1, name))

    if capture:
        # A graph of two steps cycles both buffer sets.
        with wp.ScopedCapture(model.device) as scope:
            substep()
            substep()
        for _ in range(steps // 2):
            wp.capture_launch(scope.graph)
    else:
        for _ in range(steps):
            substep()
    solver.check_constraint_capacity()
    return solver, state_0.joint_q.numpy().copy()


def test_captured_double_buffer_matches_eager(test, device, solve_mode="matrix_free"):
    """Replay a two-step graph of the double-buffered solve with the eager result."""
    model = _scene(device)
    _, eager = _run(model, solve_mode=solve_mode, use_parallel_streams=True, double_buffer=True)
    _, captured = _run(model, solve_mode=solve_mode, use_parallel_streams=True, double_buffer=True, capture=True)
    np.testing.assert_array_equal(captured, eager)


def test_reset_after_an_eager_error_steps_like_a_fresh_solver(test, device):
    """Recover from an eager step that raised mid-step: after reset, step bitwise like a fresh solver.

    The step raises after the closure rows were written into the active buffer set, and the
    four-bar starts with a closure error so those rows are nonzero.
    """
    results = []
    for fail_first in (False, True):
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        builder.replicate(_build_four_bar(), 2)
        model = builder.finalize(device=device)
        set_torsion_radius(model, 0.01)
        solver = SolverFeatherPGS(
            model,
            solve_mode="matrix_free",
            iterations=16,
            baumgarte=0.1,
            double_buffer=True,
            contact_torsion_device=True,
        )
        test.assertIsNotNone(solver._memset_stream)
        state, output = model.state(), model.state()
        joint_q = state.joint_q.numpy()
        joint_q[0] = 0.2
        state.joint_q.assign(joint_q)
        newton.eval_fk(model, state.joint_q, state.joint_qd, state)
        if fail_first:
            # A latched torsion error makes the eager step raise during torsion preparation.
            solver._device_torsion.work.status.assign(np.array([16], dtype=np.int32))
            with test.assertRaisesRegex(RuntimeError, "load ring"):
                solver.step(state, output, model.control(), None, 1.0 / 240.0)
            solver.reset(state)
            solver.check_constraint_capacity()
        for _ in range(3):
            solver.step(state, output, model.control(), None, 1.0 / 240.0)
            for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
                wp.copy(getattr(state, name), getattr(output, name))
        results.append((state.joint_q.numpy(), state.joint_qd.numpy()))
    for got, want in zip(results[1], results[0], strict=True):
        np.testing.assert_array_equal(got, want)


class TestFeatherPGSStreams(unittest.TestCase):
    pass


cuda_devices = get_cuda_test_devices()
for _fn in (test_captured_double_buffer_matches_eager,):
    add_function_test(TestFeatherPGSStreams, _fn.__name__, _fn, devices=cuda_devices)
    add_function_test(TestFeatherPGSStreams, f"{_fn.__name__}_split", _fn, devices=cuda_devices, solve_mode="split")
add_function_test(
    TestFeatherPGSStreams,
    test_reset_after_an_eager_error_steps_like_a_fresh_solver.__name__,
    test_reset_after_an_eager_error_steps_like_a_fresh_solver,
    devices=cuda_devices,
)


if __name__ == "__main__":
    unittest.main()
