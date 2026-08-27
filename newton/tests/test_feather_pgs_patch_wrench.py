# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for FeatherPGS patch-wrench contact blocks."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_selected_cuda_test_devices


def test_patch_wrench_tilted_box_falls_flat(test: unittest.TestCase, device):
    """Verify a box dropped on an edge or vertex tips over and settles flat.

    Runs in both matrix_free and split modes: the negative-mu moment clamp
    is inlined separately in the fused MF-GS kernel and the standalone MF
    kernel, so both code paths need the physics check.

    The patch-wrench moment clamp must not grant tipping capacity a
    degenerate (line/point) support does not have: the wrench center must
    be the support centroid (a body-anchored center zeroes the gravity
    torque in the constraint frame, so unbalanced boxes hang mid-tilt) and
    the clamp basis must align with the support's principal axes for
    anisotropic supports (an arbitrary basis circumscribes a diagonal edge,
    creating cross-moment capacity out of nothing).
    """
    with wp.ScopedDevice(device):
        cases = [
            (wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), float(np.radians(25.0))), "edge"),
            (
                wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), float(np.radians(25.0)))
                * wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(np.radians(45.0))),
                "diagonal edge",
            ),
            (wp.quat_from_axis_angle(wp.vec3(0.7071, 0.7071, 0.0), float(np.radians(30.0))), "vertex"),
        ]
        for q0, label in [(q, lb + f" ({mode})") for q, lb in cases for mode in ("matrix_free", "split")]:
            mode = label.rsplit("(", 1)[1].rstrip(")")
            builder = newton.ModelBuilder()
            builder.rigid_gap = 0.003
            cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.7)
            builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.7))
            body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.09), q0))
            builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
            model = builder.finalize()
            pipeline = newton.CollisionPipeline(
                model,
                reduce_contacts=True,
                rigid_contact_max=256,
                broad_phase="nxn",
                deterministic=True,
                contact_matching="latest",
            )
            contacts = pipeline.contacts()
            solver = newton.solvers.SolverFeatherPGS(
                model,
                pgs_mode=mode,
                pgs_iterations=8,
                contact_patch_wrench=True,
                mf_warmstart=True,
                # velocity iterations are matrix_free-only; split exercises the
                # standalone MF kernel's moment-clamp path instead
                pgs_velocity_iterations=2 if mode == "matrix_free" else 0,
            )
            s0, s1 = model.state(), model.state()
            control = model.control()
            newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
            for _k in range(300):
                pipeline.collide(s0, contacts)
                s0.clear_forces()
                solver.step(s0, s1, control, contacts, 1.0 / 60.0)
                s0, s1 = s1, s0
            z = float(s0.body_q.numpy()[body][2])
            test.assertLess(abs(z - 0.05), 0.01, f"{label}: box stuck tilted at z={z:.3f}, expected flat at 0.05")


class TestFeatherPGSPatchWrench(unittest.TestCase):
    pass


add_function_test(
    TestFeatherPGSPatchWrench,
    "test_patch_wrench_tilted_box_falls_flat",
    test_patch_wrench_tilted_box_falls_flat,
    devices=get_selected_cuda_test_devices(),
)

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
