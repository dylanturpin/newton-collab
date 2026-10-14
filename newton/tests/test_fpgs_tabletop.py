# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check state-buffer reuse in the interactive tabletop examples."""

import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

import newton
from newton.examples.robot._tabletop_demo import TabletopDemo
from newton.examples.robot.example_fpgs_wrecking_ball import Example
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_selected_cuda_test_devices
from newton.viewer import ViewerNull


def test_capture_substeps(test, device, substeps):
    """Match eager integration after repeatedly replaying odd/even frame graphs."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder()
        body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 5), wp.quat_identity()))
        builder.add_shape_sphere(body, radius=0.1)
        example = TabletopDemo.__new__(TabletopDemo)
        example.model = builder.finalize()
        example.state_0, example.state_1 = example.model.state(), example.model.state()
        example.control = example.model.control()
        newton.eval_fk(example.model, example.model.joint_q, example.model.joint_qd, example.state_0)
        initial = example.model.state()
        initial.assign(example.state_0)
        example.viewer = ViewerNull()
        example.pipeline = newton.CollisionPipeline(example.model)
        example.contacts = example.pipeline.contacts()
        example.solver = SolverFeatherPGS(example.model, solve_mode="matrix_free", iterations=4)
        example.sim_substeps = substeps
        example.sim_dt = 1 / (60 * substeps)
        example.simulate()
        example.state_0.assign(initial)
        example.state_1.assign(initial)
        example.solver.reset(example.state_0)
        for _ in range(30):
            example.simulate()
        expected = example.state_0.body_q.numpy()
        example.state_0.assign(initial)
        example.state_1.assign(initial)
        example.solver.reset(example.state_0)
        with wp.ScopedCapture() as capture:
            example.simulate()
        example.state_0.assign(initial)
        example.state_1.assign(initial)
        example.solver.reset(example.state_0)
        for _ in range(30):
            wp.capture_launch(capture.graph)
        np.testing.assert_allclose(example.state_0.body_q.numpy(), expected, atol=1e-5, rtol=0)
        test.assertLess(float(expected[0, 2]), 4.0)


class TestTabletopCapture(unittest.TestCase):
    pass


for count in (1, 2, 3, 4):
    add_function_test(
        TestTabletopCapture,
        f"test_capture_{count}_substeps",
        test_capture_substeps,
        devices=get_selected_cuda_test_devices(),
        substeps=count,
    )


def test_wrecking_ball_idle(test, device):
    """The unstruck building must stand for the entire demonstration."""
    with wp.ScopedDevice(device):
        example = Example(
            ViewerNull(), SimpleNamespace(substeps=4, iterations=12, contact_response="immediate", test=True)
        )
        example.autoplay = False
        for _ in range(900):
            example.step()
            example.test_post_step()
        test.assertFalse(example.ball_struck_building)
        example.check_capacity()


class TestWreckingBallStability(unittest.TestCase):
    pass


add_function_test(
    TestWreckingBallStability,
    "test_idle_building",
    test_wrecking_ball_idle,
    devices=get_selected_cuda_test_devices(),
)


if __name__ == "__main__":
    unittest.main()
