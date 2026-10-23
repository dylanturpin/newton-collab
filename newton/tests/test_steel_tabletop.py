# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check state-buffer reuse in the interactive tabletop examples."""

import unittest

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.robot._tabletop_demo import TabletopDemo
from newton.examples.robot.example_steel_padlock import Example as Padlock
from newton.examples.robot.example_steel_plate_rack import Example as PlateRack
from newton.examples.robot.example_steel_wrecking_ball import Example
from newton.solvers import SolverSteel
from newton.tests.unittest_utils import USD_AVAILABLE, add_function_test, get_selected_cuda_test_devices
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
        example.solver = SolverSteel(example.model, solve_mode="matrix_free", iterations=4)
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


def test_padlock_startup(test, device):
    """Warm one substep and preserve captured frame/reset equivalence."""
    startup_substeps = []

    class StartupPadlock(Padlock):
        def simulate(self, *args, **kwargs):
            startup_substeps.append(kwargs.get("substeps", self.sim_substeps))
            return super().simulate(*args, **kwargs)

    with wp.ScopedDevice(device):
        args = newton.examples.default_args(Padlock.create_parser())
        test.assertEqual(args.substeps, 4)
        example = StartupPadlock(ViewerNull(), args)
        test.assertEqual(startup_substeps, [1, args.substeps])
        np.testing.assert_array_equal(example.state_0.body_q.numpy(), example.initial.body_q.numpy())
        test.assertEqual(float(example.clock.numpy()[0]), 0.0)
        example.step()
        captured_pose = example.state_0.body_q.numpy()
        captured_velocity = example.state_0.body_qd.numpy()
        captured_coordinates = example.state_0.joint_q.numpy()
        captured_clock = example.clock.numpy()
        example.reset()
        example.simulate()
        np.testing.assert_allclose(example.state_0.body_q.numpy(), captured_pose, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(example.state_0.body_qd.numpy(), captured_velocity, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(example.state_0.joint_q.numpy(), captured_coordinates, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(example.clock.numpy(), captured_clock, atol=1e-8, rtol=0)
        np.testing.assert_allclose(captured_clock, [1 / 60], atol=1e-8, rtol=0)
        example.reset()
        example.clock.fill_(1.8)
        example.step()
        test.assertLess(example.state_0.joint_q.numpy()[example.coordinates["key_drive"] + 2], -0.2)
        test.assertAlmostEqual(float(example.clock.numpy()[0]), 1.8, places=6)


add_function_test(
    TestTabletopCapture,
    "test_padlock_startup",
    test_padlock_startup,
    devices=get_selected_cuda_test_devices(),
)


for count in (1, 2, 3, 4):
    add_function_test(
        TestTabletopCapture,
        f"test_capture_{count}_substeps",
        test_capture_substeps,
        devices=get_selected_cuda_test_devices(),
        substeps=count,
    )


def test_wrecking_ball_idle(test, device):
    """Keep the unstruck building standing for the entire demonstration."""
    with wp.ScopedDevice(device):
        args = newton.examples.default_args(Example.create_parser())
        args.test = True
        example = Example(ViewerNull(), args)
        example.autoplay = False
        for _ in range(900):
            example.step()
            example.test_post_step()
        test.assertFalse(example.ball_struck_building)
        example.check_capacity()


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestWreckingBallStability(unittest.TestCase):
    pass


add_function_test(
    TestWreckingBallStability,
    "test_idle_building",
    test_wrecking_ball_idle,
    devices=get_selected_cuda_test_devices(),
)


def test_complete_demo(test, device, example_type):
    """Finish the demonstration and validate placement or post-impact stability."""
    with wp.ScopedDevice(device):
        args = newton.examples.default_args(example_type.create_parser())
        args.test = True
        example = example_type(ViewerNull(), args)
        for _ in range(args.num_frames):
            example.step()
            example.test_post_step()
        example.test_final()


@unittest.skipUnless(USD_AVAILABLE, "Requires usd-core")
class TestTabletopCompletion(unittest.TestCase):
    pass


for name, example_type in (("plate_rack", PlateRack), ("wrecking_ball", Example)):
    add_function_test(
        TestTabletopCompletion,
        f"test_{name}",
        test_complete_demo,
        devices=get_selected_cuda_test_devices(),
        example_type=example_type,
    )


if __name__ == "__main__":
    unittest.main()
