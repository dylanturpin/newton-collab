# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for persistent patch friction in FeatherPGS."""

import unittest

import numpy as np
import warp as wp

import newton

# FeatherPGS and its patch kernels are exercised on CUDA, the solver's only device.
_DEVICE = "cuda:0"


@unittest.skipUnless(wp.is_cuda_available(), "SolverFeatherPGS requires CUDA")
class TestFeatherPGSFrictionPatches(unittest.TestCase):
    def test_notified_friction_edits_reach_the_patch_builder(self):
        """Build patches from notified friction edits, in place or replaced, in eager and captured steps."""
        for replace in (False, True):
            for capture in (False, True):
                with self.subTest(replace=replace, capture=capture), wp.ScopedDevice(_DEVICE):
                    _check_notified_friction_edit(self, _DEVICE, replace, capture)


def _check_notified_friction_edit(test, device, replace, capture):
    """Edit friction after the anchors formed, notify, and require the next patch build to read it."""
    # A box held on a 0.3 rad incline by static friction carries patch anchors.
    builder = newton.ModelBuilder(gravity=(9.81 * np.sin(0.3), 0.0, -9.81 * np.cos(0.3)))
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
    body = builder.add_body(xform=wp.transform(wp.vec3(0, 0, 0.1), wp.quat_identity()))
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1, cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverFeatherPGS(model, pgs_mode="matrix_free", pgs_iterations=32)
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=64)
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()

    def advance():
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 0.005)
        state_0.assign(state_1)

    def patch_mu():
        count = int(contacts.rigid_contact_count.numpy()[0])
        test.assertGreater(count, 0)
        return solver._friction_patches.current.mu.numpy()[:count]

    for _ in range(20):
        advance()
    test.assertGreater(int(solver._friction_patches.previous.valid.numpy().sum()), 0)
    np.testing.assert_array_equal(patch_mu(), np.float32(0.5))
    if capture:
        with wp.ScopedCapture(device=device) as graph:
            advance()

    def run():
        if capture:
            wp.capture_launch(graph.graph)
        else:
            advance()

    mu = np.full(model.shape_count, 0.8, dtype=np.float32)
    if replace:
        model.shape_material_mu = wp.array(mu, dtype=wp.float32, device=device)
    else:
        model.shape_material_mu.assign(mu)
    solver.notify_model_changed(newton.ModelFlags.SHAPE_PROPERTIES)
    run()
    np.testing.assert_array_equal(patch_mu(), np.float32(0.8))
    # Like the contact rows, the builder sees friction edits only through a notification.
    model.shape_material_mu.fill_(0.3)
    run()
    np.testing.assert_array_equal(patch_mu(), np.float32(0.8))


if __name__ == "__main__":
    unittest.main()
