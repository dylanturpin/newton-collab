# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example FeatherPGS Mesh Grasp
#
# Five independently driven two-link arms with parallel jaws pinch five
# free payloads built from closed convex pieces: a bottle, a globe statue,
# a hollow cup, a hexagonal collar, and a fluted vase. Nothing constrains a
# payload to its gripper: lifting depends entirely on contact and friction
# against force-limited finger drives. The sequence is settle, descend,
# grip, lift, hold; Reset replays it.
#
# Command: python -m newton.examples fpgs_mesh_grasp
#
###########################################################################

from __future__ import annotations

import math
import warnings
from itertools import pairwise

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.feather_pgs._showreel import Stepper, assert_finite

SOLVERS = {
    "feather_pgs": {
        "pgs_iterations": 4,
        "dense_max_constraints": 2048,
        "mf_max_constraints": 4096,
        # Finger travel is bounded by joint limits, and contacts between links of one
        # arm (finger against finger or palm) ride the colored family with the rest.
        "enable_joint_limits": True,
        "propagation_same_articulation_rows": True,
        "substeps": 4,
    },
    "mujoco": {},
}
STATIONS = 5
SPACING = 0.40
BASE_Y, BASE_Z = 0.31, 0.64
LINK = 0.26
HIGH_PALM = 0.64
TABLE_TOP = 0.22
GRIP_FORCE = 6.0
NAMES = ("Medical bottle", "Globe statue", "Hollow cup", "Hexagonal collar", "Fluted vase")
COLORS = (
    wp.vec3(0.90, 0.45, 0.16),
    wp.vec3(0.12, 0.55, 0.80),
    wp.vec3(0.77, 0.22, 0.36),
    wp.vec3(0.68, 0.57, 0.26),
    wp.vec3(0.28, 0.66, 0.46),
)
DARK = wp.vec3(0.14, 0.18, 0.23)
SILVER = wp.vec3(0.60, 0.66, 0.72)


# --- closed convex pieces -------------------------------------------------------


def _prism(bottom, top):
    """Closed prism between two equal-count polygons: (vertices, flat triangle indices)."""
    n = len(bottom)
    faces = []
    for i in range(1, n - 1):
        faces += [(0, i + 1, i), (n, n + i, n + i + 1)]
    for i in range(n):
        j = (i + 1) % n
        faces += [(i, j, n + j), (i, n + j, n + i)]
    return np.asarray(bottom + top, dtype=np.float32), np.asarray(faces, dtype=np.int32).flatten()


def _ring_points(radius, z, sides=16):
    return [
        (radius * math.cos(2 * math.pi * i / sides), radius * math.sin(2 * math.pi * i / sides), z)
        for i in range(sides)
    ]


def _lathe(profile):
    """Stack of frustums through (radius, z) pairs."""
    return [_prism(_ring_points(r0, z0), _ring_points(r1, z1)) for (r0, z0), (r1, z1) in pairwise(profile)]


def _ring(outer0, outer1, inner0, inner1, z0, z1, sides):
    pieces = []
    for i in range(sides):
        a, b = 2 * math.pi * i / sides, 2 * math.pi * (i + 1) / sides

        def face(outer, inner, z, a=a, b=b):
            return [
                (inner * math.cos(a), inner * math.sin(a), z),
                (outer * math.cos(a), outer * math.sin(a), z),
                (outer * math.cos(b), outer * math.sin(b), z),
                (inner * math.cos(b), inner * math.sin(b), z),
            ]

        pieces.append(_prism(face(outer0, inner0, z0), face(outer1, inner1, z1)))
    return pieces


def _fluted(r0, r1, z0, z1):
    """Twenty-four convex sectors with alternating radii: real longitudinal flutes."""
    pieces = []
    for i in range(24):
        a, b = 2 * math.pi * i / 24, 2 * math.pi * (i + 1) / 24

        def face(radius, z, a=a, b=b, i=i):
            ra = radius * (1.0 if i % 2 == 0 else 0.86)
            rb = radius * (0.86 if i % 2 == 0 else 1.0)
            return [(0.0, 0.0, z), (ra * math.cos(a), ra * math.sin(a), z), (rb * math.cos(b), rb * math.sin(b), z)]

        pieces.append(_prism(face(r0, z0), face(r1, z1)))
    return pieces


def _payload_pieces(index):
    if index == 0:
        return _lathe(
            [(0.028, 0), (0.032, 0.008), (0.032, 0.095), (0.020, 0.112), (0.020, 0.124), (0.023, 0.124), (0.023, 0.142)]
        )
    if index == 1:
        globe = [
            (max(0.001, 0.05 * math.sin(math.pi * i / 10)), 0.105 - 0.05 * math.cos(math.pi * i / 10))
            for i in range(11)
        ]
        return _lathe([(0.034, 0), (0.034, 0.012), (0.013, 0.022), (0.013, 0.054)]) + _lathe(globe)
    if index == 2:
        return _lathe([(0.034, 0), (0.0347, 0.008)]) + _ring(0.0347, 0.046, 0.0287, 0.040, 0.008, 0.13, 16)
    if index == 3:
        return _ring(0.047, 0.047, 0.023, 0.023, 0.0, 0.10, 6)
    return [
        p
        for z0, z1, r0, r1 in ((0.0, 0.012, 0.032, 0.040), (0.012, 0.09, 0.040, 0.027), (0.09, 0.14, 0.027, 0.043))
        for p in _fluted(r0, r1, z0, z1)
    ]


def _arm_angles(height):
    reach, dz = 0.31, height - BASE_Z
    elbow = -math.acos((reach * reach + dz * dz - 2 * LINK * LINK) / (2 * LINK * LINK))
    shoulder = math.atan2(dz, reach) - math.atan2(math.sin(elbow), 1 + math.cos(elbow))
    return [shoulder, elbow, -shoulder - elbow]


def _ramp(t, start, end):
    u = min(1.0, max(0.0, (t - start) / (end - start)))
    return u * u * (3 - 2 * u)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.friction = 0.7

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.rigid_gap = 0.001
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=self.friction))
        static = newton.ModelBuilder.ShapeConfig(density=0.0, mu=self.friction)
        payload_cfg = newton.ModelBuilder.ShapeConfig(density=600.0, mu=self.friction, restitution=0.0)
        axis = wp.vec3(-1.0, 0.0, 0.0)
        self.payloads, self.joints, self.closed_travel = [], [], []
        self.initial_angles = _arm_angles(HIGH_PALM)
        for i in range(STATIONS):
            x = (i - 2) * SPACING
            builder.add_shape_box(
                -1,
                xform=wp.transform(wp.vec3(x, 0.10, 0.20), wp.quat_identity()),
                hx=0.17,
                hy=0.24,
                hz=0.02,
                cfg=static,
                color=DARK,
            )
            builder.add_shape_box(
                -1,
                xform=wp.transform(wp.vec3(x, BASE_Y, 0.42), wp.quat_identity()),
                hx=0.035,
                hy=0.035,
                hz=0.20,
                cfg=static,
                color=DARK,
            )

            # Payload: one free body made of closed convex pieces resting on the table.
            pieces = _payload_pieces(i)
            payload = builder.add_body(xform=wp.transform(wp.vec3(x, 0.0, TABLE_TOP + 0.002), wp.quat_identity()))
            half_width = 0.0
            for verts, faces in pieces:
                builder.add_shape_convex_hull(payload, mesh=newton.Mesh(verts, faces), cfg=payload_cfg, color=COLORS[i])
                half_width = max(half_width, float(np.abs(verts[:, 0]).max()))
            self.payloads.append(payload)
            self.closed_travel.append(0.072 - 0.006 - half_width + min(GRIP_FORCE * 0.8 / 1800.0, 0.003))

            # Arm: shoulder, elbow, wrist about -x, built so zero joint coordinates are this pose.
            s, e, w = self.initial_angles
            q1 = wp.quat_from_axis_angle(axis, s)
            q2 = wp.quat_from_axis_angle(axis, s + e)
            base = wp.vec3(x, BASE_Y, BASE_Z)
            elbow = base + wp.quat_rotate(q1, wp.vec3(0.0, -LINK, 0.0))
            palm_point = elbow + wp.quat_rotate(q2, wp.vec3(0.0, -LINK, 0.0))
            arm_cfg = newton.ModelBuilder.ShapeConfig(density=940.0, mu=self.friction)
            palm_cfg = newton.ModelBuilder.ShapeConfig(density=1040.0, mu=self.friction)
            finger_cfg = newton.ModelBuilder.ShapeConfig(density=600.0, mu=self.friction)
            upper = builder.add_link(xform=wp.transform(base + wp.quat_rotate(q1, wp.vec3(0.0, -LINK / 2, 0.0)), q1))
            builder.add_shape_box(upper, hx=0.016, hy=LINK / 2, hz=0.016, cfg=arm_cfg, color=SILVER)
            lower = builder.add_link(xform=wp.transform(elbow + wp.quat_rotate(q2, wp.vec3(0.0, -LINK / 2, 0.0)), q2))
            builder.add_shape_box(lower, hx=0.015, hy=LINK / 2, hz=0.015, cfg=arm_cfg, color=SILVER)
            palm = builder.add_link(xform=wp.transform(palm_point, wp.quat_identity()))
            builder.add_shape_box(palm, hx=0.08, hy=0.03, hz=0.0125, cfg=palm_cfg, color=DARK)
            proximal = wp.transform(wp.vec3(0.0, LINK / 2, 0.0), wp.quat_identity())
            station_joints = [
                builder.add_joint_revolute(
                    -1,
                    upper,
                    parent_xform=wp.transform(base, q1),
                    child_xform=proximal,
                    axis=axis,
                    target_pos=0.0,
                    target_ke=1800.0,
                    target_kd=40.0,
                    effort_limit=12.0,
                ),
                builder.add_joint_revolute(
                    upper,
                    lower,
                    parent_xform=wp.transform(wp.vec3(0.0, -LINK / 2, 0.0), wp.quat_from_axis_angle(axis, e)),
                    child_xform=proximal,
                    axis=axis,
                    target_pos=0.0,
                    target_ke=1400.0,
                    target_kd=30.0,
                    effort_limit=8.0,
                ),
                builder.add_joint_revolute(
                    lower,
                    palm,
                    parent_xform=wp.transform(wp.vec3(0.0, -LINK / 2, 0.0), wp.quat_from_axis_angle(axis, w)),
                    child_xform=wp.transform_identity(),
                    axis=axis,
                    target_pos=0.0,
                    target_ke=800.0,
                    target_kd=18.0,
                    effort_limit=4.0,
                ),
            ]
            for sign in (-1.0, 1.0):
                rail = wp.vec3(sign * 0.072, 0.0, -0.008)
                finger = builder.add_link(
                    xform=wp.transform(palm_point + rail + wp.vec3(0.0, 0.0, -0.07), wp.quat_identity())
                )
                builder.add_shape_box(finger, hx=0.006, hy=0.03, hz=0.07, cfg=finger_cfg, color=DARK)
                station_joints.append(
                    builder.add_joint_prismatic(
                        palm,
                        finger,
                        parent_xform=wp.transform(rail, wp.quat_identity()),
                        child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.07), wp.quat_identity()),
                        axis=wp.vec3(-sign, 0.0, 0.0),
                        limit_lower=0.0,
                        limit_upper=0.055,
                        target_pos=0.0,
                        target_ke=1800.0,
                        target_kd=12.0,
                        effort_limit=GRIP_FORCE,
                    )
                )
            builder.add_articulation(station_joints, label=f"arm_{i}")
            self.joints.append(station_joints)

        with warnings.catch_warnings():
            # FeatherPGS reads joint_target_q in the legacy DOF layout, so keep it.
            warnings.simplefilter("ignore", DeprecationWarning)
            self.model = builder.finalize()
        self.model.rigid_contact_max = 4096
        self.collision_pipeline = newton.examples.create_collision_pipeline(
            self.model, args, broad_phase="sap", rigid_contact_max=self.model.rigid_contact_max
        )
        self.contacts = self.collision_pipeline.contacts()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.stepper = Stepper(self, solver_overrides=SOLVERS, solver=str(getattr(args, "solver", "feather_pgs")))
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        # FeatherPGS reads drive targets in DOF layout (indexed by joint_qd_start).
        self.target_index = self.model.joint_qd_start.numpy()
        self.targets = self.control.joint_target_q.numpy().copy()
        self.initial_heights = self.state_0.body_q.numpy()[self.payloads, 2].copy()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(-1.5, -1.75, 1.2), pitch=-22.0, yaw=48.0)

    @staticmethod
    def phase(t):
        if t < 1.0:
            return "Settle"
        if t < 3.0:
            return "Descend"
        if t < 4.5:
            return "Grip"
        if t < 7.0:
            return "Lift"
        return "Hold - reset to replay"

    def _command(self):
        t = self.sim_time
        height = HIGH_PALM - 0.22 * _ramp(t, 1.0, 3.0) + 0.22 * _ramp(t, 4.5, 7.0)
        arm = [a - a0 for a, a0 in zip(_arm_angles(height), self.initial_angles, strict=True)]
        for joints, travel in zip(self.joints, self.closed_travel, strict=True):
            close = travel * _ramp(t, 3.0, 4.3)
            for joint, value in zip(joints, [*arm, close, close], strict=True):
                self.targets[self.target_index[joint]] = value
        self.control.joint_target_q.assign(self.targets)

    def step(self):
        self._command()
        self.stepper.step()

    def substep(self):
        self.state_0.clear_forces()
        self.viewer.apply_forces(self.state_0)
        self.stepper.collide()
        self.stepper.solve()

    def lifts(self):
        return self.state_0.body_q.numpy()[self.payloads, 2] - self.initial_heights

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        for name, lift in zip(NAMES, self.lifts(), strict=True):
            self.viewer.log_scalar(f"{name} lift [m]", float(lift))
        self.viewer.end_frame()

    def gui(self, ui):
        ui.text(f"Phase: {self.phase(self.sim_time)}   t = {self.sim_time:4.1f} s")
        changed, mu = ui.slider_float("Friction mu", self.friction, 0.1, 1.5)
        if changed:
            self.friction = mu
            self.model.shape_material_mu.fill_(mu)
        for name, lift in zip(NAMES, self.lifts(), strict=True):
            ui.text(f"{name:18s} lift {1e3 * lift:6.1f} mm")
        self.stepper.gui(ui)

    def test_final(self):
        assert_finite(self.state_0.body_q, self.state_0.body_qd)
        lifts = self.lifts()
        for name, lift in zip(NAMES, lifts, strict=True):
            if lift < 0.15:
                raise ValueError(f"{name} was not lifted: {lift:.3f} m")
        speeds = np.linalg.norm(self.state_0.body_qd.numpy()[self.payloads, :3], axis=1)
        if speeds.max() > 0.2:
            raise ValueError("a payload is slipping in the grasp")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--solver", default="feather_pgs", choices=list(SOLVERS), help="Rigid-body solver.")
        parser.set_defaults(num_frames=480)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
