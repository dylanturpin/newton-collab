# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for a prismatic FeatherPGS pusher loading a cube contact."""

from __future__ import annotations

from dataclasses import dataclass
import math
import unittest

import numpy as np
import warp as wp

import newton
from newton import JointTargetMode
from newton._src.solvers import SolverFeatherPGS


CUBE_HALF = 0.05
CUBE_MASS = 1.0
SPHERE_RADIUS = 0.035
SPHERE_MASS = 0.25
TABLE_HALF_XY = 0.35
TABLE_HALF_Z = 0.025
CONTACT_GAP = 0.004
DRIVE_STIFFNESS = 1.0e3
TARGET_OVERLAP = 0.025
DT = 0.005


@dataclass(frozen=True)
class SceneIds:
    cube_body: int
    pusher_body: int
    table_shape: int
    cube_shape: int
    pusher_shape: int
    pusher_joint: int


def _shape_cfg(mu: float) -> newton.ModelBuilder.ShapeConfig:
    cfg = newton.ModelBuilder.ShapeConfig(mu=mu, density=0.0)
    cfg.ke = 5.0e4
    cfg.kd = 5.0e2
    cfg.kf = 1.0e3
    cfg.margin = 0.0
    cfg.gap = CONTACT_GAP
    return cfg


def _make_solver(model: newton.Model) -> SolverFeatherPGS:
    return SolverFeatherPGS(
        model,
        update_mass_matrix_interval=1,
        pgs_mode="matrix_free",
        pgs_kernel="tiled_row",
        pgs_iterations=16,
        pgs_beta=0.05,
        pgs_cfm=1.0e-6,
        pgs_omega=1.0,
        dense_max_constraints=64,
        mf_max_constraints=256,
        pgs_warmstart=False,
        enable_contact_friction=True,
        contact_friction_gap_threshold=0.04,
        enable_joint_limits=True,
        enable_joint_velocity_limits=True,
        drive_mode="physx_pgs",
        friction_mode="bisection",
        pgs_schedule="contact_then_internal",
        double_buffer=False,
        use_parallel_streams=False,
    )


def _build_scene(device: wp.context.Device) -> tuple[newton.Model, SceneIds]:
    builder = newton.ModelBuilder(gravity=-9.81)
    builder.rigid_gap = CONTACT_GAP

    table_shape = builder.add_shape_box(
        body=-1,
        xform=wp.transform(wp.vec3(0.0, 0.0, -TABLE_HALF_Z), wp.quat_identity()),
        hx=TABLE_HALF_XY,
        hy=TABLE_HALF_XY,
        hz=TABLE_HALF_Z,
        cfg=_shape_cfg(0.8),
        label="table",
    )

    cube_inertia = (1.0 / 6.0) * CUBE_MASS * (2.0 * CUBE_HALF) ** 2
    cube_body = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, CUBE_HALF), wp.quat_identity()),
        mass=CUBE_MASS,
        inertia=wp.mat33(cube_inertia, 0.0, 0.0, 0.0, cube_inertia, 0.0, 0.0, 0.0, cube_inertia),
        lock_inertia=True,
        label="cube",
    )
    cube_shape = builder.add_shape_box(
        body=cube_body,
        hx=CUBE_HALF,
        hy=CUBE_HALF,
        hz=CUBE_HALF,
        cfg=_shape_cfg(0.8),
        label="cube_shape",
    )

    pusher_start = wp.vec3(0.0, 0.0, 2.0 * CUBE_HALF + SPHERE_RADIUS)
    sphere_inertia = (2.0 / 5.0) * SPHERE_MASS * SPHERE_RADIUS**2
    pusher_body = builder.add_link(
        xform=wp.transform(pusher_start, wp.quat_identity()),
        mass=SPHERE_MASS,
        inertia=wp.mat33(
            sphere_inertia,
            0.0,
            0.0,
            0.0,
            sphere_inertia,
            0.0,
            0.0,
            0.0,
            sphere_inertia,
        ),
        lock_inertia=True,
        label="driven_pusher",
    )
    pusher_shape = builder.add_shape_sphere(
        body=pusher_body,
        radius=SPHERE_RADIUS,
        cfg=_shape_cfg(0.8),
        label="driven_pusher_shape",
    )
    pusher_joint = builder.add_joint_prismatic(
        parent=-1,
        child=pusher_body,
        parent_xform=wp.transform(pusher_start, wp.quat_identity()),
        child_xform=wp.transform_identity(),
        axis=wp.vec3(0.0, 0.0, -1.0),
        target_pos=TARGET_OVERLAP,
        target_vel=0.0,
        target_ke=DRIVE_STIFFNESS,
        target_kd=2.0 * math.sqrt(DRIVE_STIFFNESS),
        limit_lower=0.0,
        limit_upper=0.08,
        effort_limit=60.0,
        velocity_limit=1.0,
        actuator_mode=JointTargetMode.POSITION,
        label="pusher_axis",
    )
    builder.add_articulation([pusher_joint], label="driven_pusher_articulation")

    model = builder.finalize(device=device)
    model.request_contact_attributes("force")
    return model, SceneIds(cube_body, pusher_body, table_shape, cube_shape, pusher_shape, pusher_joint)


def _contact_normals_from_a_to_b(contacts, shape_a: int, shape_b: int) -> np.ndarray:
    count = int(contacts.rigid_contact_count.numpy()[0])
    shape0 = contacts.rigid_contact_shape0.numpy()[:count]
    shape1 = contacts.rigid_contact_shape1.numpy()[:count]
    normals = contacts.rigid_contact_normal.numpy()[:count]
    out: list[np.ndarray] = []
    for i in range(count):
        s0 = int(shape0[i])
        s1 = int(shape1[i])
        if s0 == shape_a and s1 == shape_b:
            out.append(np.array(normals[i], dtype=np.float64))
        elif s0 == shape_b and s1 == shape_a:
            out.append(-np.array(normals[i], dtype=np.float64))
    if not out:
        return np.zeros((0, 3), dtype=np.float64)
    return np.stack(out, axis=0)


def _contact_force_sum_between(contacts, shape_a: int, shape_b: int) -> float:
    count = int(contacts.rigid_contact_count.numpy()[0])
    shape0 = contacts.rigid_contact_shape0.numpy()[:count]
    shape1 = contacts.rigid_contact_shape1.numpy()[:count]
    forces = contacts.rigid_contact_force.numpy()[:count]
    total = 0.0
    for i in range(count):
        s0 = int(shape0[i])
        s1 = int(shape1[i])
        if (s0 == shape_a and s1 == shape_b) or (s0 == shape_b and s1 == shape_a):
            total += float(np.linalg.norm(forces[i]))
    return total


def _collision_pipeline(model: newton.Model) -> newton.CollisionPipeline:
    return newton.CollisionPipeline(model, broad_phase="nxn", contact_matching="sticky", reduce_contacts=False)


class TestFeatherPGSPrismaticSphereCube(unittest.TestCase):
    def setUp(self):
        if not wp.is_cuda_available():
            self.skipTest("FeatherPGS matrix-free contact probe is exercised on CUDA only")
        self.device = wp.get_device("cuda:0")

    def test_initial_contacts_include_table_support_and_prismatic_pusher(self):
        model, ids = _build_scene(self.device)
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        pipeline = _collision_pipeline(model)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)

        table_to_cube = _contact_normals_from_a_to_b(contacts, ids.table_shape, ids.cube_shape)
        cube_to_pusher = _contact_normals_from_a_to_b(contacts, ids.cube_shape, ids.pusher_shape)

        self.assertGreaterEqual(len(table_to_cube), 3)
        self.assertGreaterEqual(float(np.min(table_to_cube[:, 2])), 0.90)
        self.assertEqual(len(cube_to_pusher), 1)
        self.assertGreater(float(cube_to_pusher[0, 2]), 0.95)
        self.assertLess(float(np.linalg.norm(cube_to_pusher[0, :2])), 0.20)

    def test_matrix_free_step_loads_contact_without_unbounded_cube_motion(self):
        model, ids = _build_scene(self.device)
        solver = _make_solver(model)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        pipeline = _collision_pipeline(model)
        contacts = pipeline.contacts()

        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
        dof_start = int(model.joint_qd_start.numpy()[ids.pusher_joint])
        control_targets = control.joint_target_pos.numpy()
        control_targets[dof_start] = TARGET_OVERLAP
        control.joint_target_pos.assign(control_targets)

        peak_pusher_force = 0.0
        peak_cube_speed = 0.0
        peak_cube_omega = 0.0
        for _step_idx in range(80):
            state_0.clear_forces()
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            solver.update_contacts(contacts)
            state_0, state_1 = state_1, state_0

            cube_qd = state_0.body_qd.numpy()[ids.cube_body]
            peak_cube_speed = max(peak_cube_speed, float(np.linalg.norm(cube_qd[:3])))
            peak_cube_omega = max(peak_cube_omega, float(np.linalg.norm(cube_qd[3:6])))
            peak_pusher_force = max(
                peak_pusher_force,
                _contact_force_sum_between(contacts, ids.cube_shape, ids.pusher_shape),
            )

        pipeline.collide(state_0, contacts)
        cube_q = state_0.body_q.numpy()[ids.cube_body]
        cube_qd = state_0.body_qd.numpy()[ids.cube_body]

        self.assertGreater(peak_pusher_force, 1.0)
        self.assertLess(peak_cube_speed, 10.0)
        self.assertLess(peak_cube_omega, 3.0)
        self.assertTrue(np.all(np.isfinite(cube_q)))
        self.assertTrue(np.all(np.isfinite(cube_qd)))
        self.assertLess(abs(float(cube_q[0])), 0.01)
        self.assertLess(abs(float(cube_q[1])), 0.01)
        self.assertGreater(float(cube_q[2]), 0.035)
        self.assertLess(float(cube_q[2]), 0.065)


if __name__ == "__main__":
    unittest.main(verbosity=2)
