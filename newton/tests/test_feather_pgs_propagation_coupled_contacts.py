# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Propagation contact rows on several links of one articulation converge instead of oscillating."""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS

DT = 1.0 / 240.0
PROPAGATION_RESPONSES = ("propagation", "propagation-fused")


def _quadruped_model(device):
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.mu = 0.6
    root = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.25), wp.quat_identity()))
    builder.add_shape_box(root, hx=0.2, hy=0.12, hz=0.04)
    joints = [builder.add_joint_free(root)]
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            leg = builder.add_link(xform=wp.transform(wp.vec3(0.18 * sx, 0.1 * sy, 0.08), wp.quat_identity()))
            builder.add_shape_capsule(leg, radius=0.03, half_height=0.1)
            joints.append(
                builder.add_joint_revolute(
                    root,
                    leg,
                    axis=wp.vec3(0.0, 1.0, 0.0),
                    parent_xform=wp.transform(wp.vec3(0.18 * sx, 0.1 * sy, -0.03), wp.quat_identity()),
                    child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.13), wp.quat_identity()),
                )
            )
    builder.add_articulation(joints)
    builder.add_ground_plane()
    return builder.finalize(device=device)


def _three_foot_slider_model(device):
    """One prismatic-Z DOF carrying three spheres on fixed-linked bodies above a plane."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    builder.default_shape_cfg.mu = 0.0
    slider = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
    builder.add_shape_box(
        slider, hx=0.3, hy=0.05, hz=0.02, cfg=newton.ModelBuilder.ShapeConfig(has_shape_collision=False)
    )
    joints = [builder.add_joint_prismatic(-1, slider, axis=wp.vec3(0.0, 0.0, 1.0))]
    for x in (-0.2, 0.0, 0.2):
        foot = builder.add_link(xform=wp.transform(wp.vec3(x, 0.0, 0.05), wp.quat_identity()))
        builder.add_shape_sphere(foot, radius=0.05)
        joints.append(
            builder.add_joint_fixed(
                slider,
                foot,
                parent_xform=wp.transform(wp.vec3(x, 0.0, -0.05), wp.quat_identity()),
                child_xform=wp.transform_identity(),
            )
        )
    builder.add_articulation(joints)
    builder.add_ground_plane()
    return builder.finalize(device=device)


def _solver(model, response, iterations, **kwargs):
    return SolverFeatherPGS(
        model,
        pgs_mode="matrix_free",
        articulated_contact_response=response,
        propagation_cached_response=False,
        friction_anchor_beta=0.0,
        dense_max_constraints=96,
        mf_max_constraints=96,
        pgs_iterations=iterations,
        **kwargs,
    )


def _rollout(model, solver, joint_qd, steps):
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    state_0.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    for _ in range(steps):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0
        yield state_0.joint_qd.numpy()


@unittest.skipUnless(wp.get_cuda_device_count() > 0, "requires CUDA")
class TestPropagationCoupledContacts(unittest.TestCase):
    def test_shared_dof_feet_reach_rest(self):
        """Three feet on one prismatic DOF stop at the plane for every sweep count."""
        model = _three_foot_slider_model("cuda:0")
        for response in PROPAGATION_RESPONSES:
            for iterations in (1, 2, 12, 13):
                with self.subTest(response=response, iterations=iterations):
                    solver = _solver(model, response, iterations, pgs_beta=0.0, pgs_cfm=0.0)
                    (qd,) = _rollout(model, solver, np.array([-1.0], dtype=np.float32), 1)
                    self.assertLess(abs(float(qd[0])), 1.0e-3)

    def test_floating_quadruped_with_joint_velocities_stays_bounded(self):
        """Four feet coupled through a light free root keep their roll rate physical."""
        model = _quadruped_model("cuda:0")
        joint_qd = np.linspace(-1.5, 1.5, model.joint_dof_count).astype(np.float32)
        for response in PROPAGATION_RESPONSES:
            with self.subTest(response=response):
                solver = _solver(model, response, 12)
                trajectory = list(_rollout(model, solver, joint_qd, 120))
                self.assertLess(float(np.abs(trajectory[0][3:6]).max()), 10.0)
                self.assertTrue(np.isfinite(trajectory[-1]).all())
                self.assertLess(float(np.abs(np.stack(trajectory)).max()), 50.0)


if __name__ == "__main__":
    unittest.main()
