# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS

DT = 1.0 / 240.0


def _build_model(device):
    builder = newton.ModelBuilder()
    builder.add_ground_plane()
    link0 = builder.add_link(xform=wp.transform((0.0, 0.0, 0.3), wp.quat_identity()))
    builder.add_shape_box(link0, hx=0.12, hy=0.05, hz=0.05)
    link1 = builder.add_link(xform=wp.transform((0.25, 0.0, 0.3), wp.quat_identity()))
    builder.add_shape_box(link1, hx=0.12, hy=0.05, hz=0.05)
    j0 = builder.add_joint_revolute(
        -1, link0, axis=(0.0, 1.0, 0.0), parent_xform=wp.transform((0.0, 0.0, 0.3), wp.quat_identity())
    )
    j1 = builder.add_joint_revolute(
        link0, link1, axis=(0.0, 1.0, 0.0), parent_xform=wp.transform((0.25, 0.0, 0.0), wp.quat_identity())
    )
    builder.add_articulation([j0, j1])
    box = builder.add_body(xform=wp.transform((0.6, 0.0, 0.15), wp.quat_identity()))
    builder.add_shape_box(box, hx=0.1, hy=0.1, hz=0.1)
    return builder.finalize(device=device)


def _run(device, double_buffer, pause_between_steps, replays):
    """Capture two FPGS steps, optionally with a capture-pausing conditional node between them."""
    model = _build_model(device)
    solver = SolverFeatherPGS(model, pgs_mode="matrix_free", double_buffer=double_buffer)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()
    flag = wp.ones(1, dtype=wp.int32, device=device)

    def simulate():
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT)
        if pause_between_steps:
            # A conditional node pauses the capture, as Kamino and implicit MPM do inside their steps.
            wp.capture_if(flag, on_true=lambda: flag.fill_(1))
        pipeline.collide(state_1, contacts)
        solver.step(state_1, state_0, control, contacts, DT)

    with wp.ScopedCapture(device) as capture:
        simulate()
    for _ in range(replays):
        wp.capture_launch(capture.graph)
    return solver, state_0.body_q.numpy(), state_0.body_qd.numpy()


@unittest.skipUnless(wp.is_cuda_available(), "requires CUDA")
class TestFeatherPGSDoubleBufferCapture(unittest.TestCase):
    def setUp(self):
        self.device = wp.get_cuda_device()
        if not wp.is_conditional_graph_supported():
            self.skipTest("requires conditional graph nodes")

    def test_capture_pause_after_double_buffered_step(self):
        """A capture pause between double-buffered steps succeeds and matches single buffering."""
        solver, body_q, body_qd = _run(self.device, True, True, replays=20)
        self.assertIsNotNone(solver._memset_stream)
        _, ref_q, ref_qd = _run(self.device, False, True, replays=20)
        np.testing.assert_array_equal(body_q, ref_q)
        np.testing.assert_array_equal(body_qd, ref_qd)

    def test_double_buffer_matches_single_buffer_without_pause(self):
        """Captured double-buffered steps match single buffering bitwise."""
        _, body_q, body_qd = _run(self.device, True, False, replays=20)
        _, ref_q, ref_qd = _run(self.device, False, False, replays=20)
        self.assertTrue(np.isfinite(body_q).all())
        np.testing.assert_array_equal(body_q, ref_q)
        np.testing.assert_array_equal(body_qd, ref_qd)


if __name__ == "__main__":
    unittest.main(verbosity=2)
