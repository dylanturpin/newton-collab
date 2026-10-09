# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Row high-water marks of SolverFeatherPGS (``row_watermark``)."""

import unittest

import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices


def _scene(device):
    """Two worlds: a box on the ground next to a fixed-base slider resting on it, one world without contacts."""
    template = newton.ModelBuilder()
    box = template.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
    template.add_shape_box(box, hx=0.1, hy=0.1, hz=0.1)
    link = template.add_link(xform=wp.transform(wp.vec3(0.5, 0.0, 0.1), wp.quat_identity()))
    template.add_shape_box(link, hx=0.1, hy=0.1, hz=0.1)
    template.add_articulation(
        [
            template.add_joint_prismatic(
                -1, link, axis=newton.Axis.Z, parent_xform=wp.transform(wp.vec3(0.5, 0.0, 0.1), wp.quat_identity())
            )
        ]
    )
    lifted = newton.ModelBuilder()
    lifted.add_shape_box(
        lifted.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 3.0), wp.quat_identity())), hx=0.1, hy=0.1, hz=0.1
    )
    builder = newton.ModelBuilder()
    builder.add_world(template)
    builder.add_world(lifted)
    builder.add_ground_plane()
    return builder.finalize(device=device)


def test_watermarks_accumulate_in_captured_steps(test, device):
    """Accumulate the same marks under CUDA graph replay as in eager steps."""
    results = []
    for capture in (False, True):
        model = _scene(device)
        solver = SolverFeatherPGS(model, row_watermark=True, friction_anchor_beta=0.0)
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        state_0, state_1 = model.state(), model.state()
        newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
        control = model.control()

        def substep(
            state_0=state_0, state_1=state_1, contacts=contacts, solver=solver, control=control, pipeline=pipeline
        ):
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
            wp.copy(state_0.joint_q, state_1.joint_q)
            wp.copy(state_0.joint_qd, state_1.joint_qd)
            wp.copy(state_0.body_q, state_1.body_q)
            wp.copy(state_0.body_qd, state_1.body_qd)

        if capture:
            substep()
            with wp.ScopedCapture(device) as capture_scope:
                substep()
            for _ in range(9):
                wp.capture_launch(capture_scope.graph)
        else:
            for _ in range(11):
                substep()
        results.append(solver.constraint_row_watermarks())
    test.assertEqual(results[0], results[1])
    test.assertGreater(results[1]["dense_high_water"], 0)


class TestFeatherPGSRowWatermark(unittest.TestCase):
    pass


cuda_devices = get_cuda_test_devices()
add_function_test(
    TestFeatherPGSRowWatermark,
    "test_watermarks_accumulate_in_captured_steps",
    test_watermarks_accumulate_in_captured_steps,
    devices=cuda_devices,
)


if __name__ == "__main__":
    unittest.main()
