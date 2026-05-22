# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Minimal prismatic-drive sphere pressing a cube on a table.

This diagnostic isolates the drive/contact/table-support pattern behind the
Franka cube-lift instability without IsaacLab, policy actions, or a gripper.
It deliberately asserts contact assumptions before checking rollout behavior.

Command:
    python -m newton.examples.diagnostics.example_prismatic_sphere_cube_press --assert
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
import warp as wp

import newton
from newton import JointTargetMode
from newton._src.solvers import SolverFeatherPGS
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    PGS_CONSTRAINT_TYPE_JOINT_TARGET,
    PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
)

DEFAULT_CUBE_HALF = 0.05
CUBE_MASS = 1.0
DEFAULT_SPHERE_RADIUS = 0.035
SPHERE_MASS = 0.25
TABLE_HALF_XY = 0.35
TABLE_HALF_Z = 0.025
TABLE_TOP_Z = 0.0
CONTACT_GAP = 0.004
DEFAULT_TARGET_OVERLAP = 0.025
DEFAULT_EFFORT_LIMIT = 15.0
DEFAULT_JOINT_LIMIT_UPPER = 0.08
DRIVE_STIFFNESS = 1.0e3
DEFAULT_DT = 0.005
DEFAULT_STEPS = 200
BAD_FRAME_CUBE_HALF = 0.024
BAD_FRAME_TABLE_TOP_Z = -0.003
BAD_FRAME_CUBE_TO_FINGER_NORMAL = (-0.720389, 0.153977, 0.676262)
LATEST_BAD_FRAME_CUBE_TO_LEFTFINGER_NORMAL = (-0.999726, 0.000014, 0.023398)
LATEST_BAD_FRAME_CUBE_TO_LEFTFINGER_POINTS = (
    (-0.02399966, 0.01117754, 0.01322154),
    (-0.02399966, 0.01038351, 0.02399999),
    (-0.02399963, 0.01039103, 0.02400350),
)


@dataclass(frozen=True)
class PressScenario:
    name: str
    sphere_xy: tuple[float, float]
    start_z: float
    contact_kind: Literal["top", "corner"]
    max_final_xy_m: float
    max_final_tilt_deg: float
    max_peak_omega_radps: float
    max_final_omega_radps: float
    max_final_z_m: float
    min_final_z_m: float
    cube_half: float = DEFAULT_CUBE_HALF
    sphere_radius: float = DEFAULT_SPHERE_RADIUS
    table_top_z: float = TABLE_TOP_Z
    expected_cube_to_sphere_normal: tuple[float, float, float] | None = None
    max_peak_speed_mps: float = 10.0
    pusher_shape: Literal["sphere", "box"] = "sphere"
    pusher_joint_kind: Literal["prismatic", "d6"] = "prismatic"
    pusher_mass: float = SPHERE_MASS
    pusher_quat: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    pusher_axis: tuple[float, float, float] = (0.0, 0.0, -1.0)
    pusher_d6_target_scales: tuple[float, float, float, float, float, float] = (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    pusher_d6_angular_axes: bool = False
    pusher_box_half_extents: tuple[float, float, float] = (0.0091, 0.0060, 0.0054)
    pusher_box_local_pos: tuple[float, float, float] = (0.0, 0.0, 0.0448)


def _cube_center_z(scenario: PressScenario) -> float:
    return scenario.table_top_z + scenario.cube_half


def _cube_top_z(cube_half: float = DEFAULT_CUBE_HALF, table_top_z: float = TABLE_TOP_Z) -> float:
    return table_top_z + 2.0 * cube_half


def _corner_sphere_center(offset_xy: float) -> tuple[tuple[float, float], float]:
    z_offset = math.sqrt(max(DEFAULT_SPHERE_RADIUS**2 - 2.0 * offset_xy**2, 0.0))
    return (DEFAULT_CUBE_HALF + offset_xy, DEFAULT_CUBE_HALF + offset_xy), _cube_top_z() + z_offset


def _sphere_center_from_corner_normal(
    corner: tuple[float, float, float],
    normal: tuple[float, float, float],
    radius: float,
) -> tuple[tuple[float, float], float]:
    n = np.array(normal, dtype=np.float64)
    n /= float(np.linalg.norm(n))
    center = np.array(corner, dtype=np.float64) + radius * n
    return (float(center[0]), float(center[1])), float(center[2])


CORNER_XY, CORNER_Z = _corner_sphere_center(0.014)
BAD_FRAME_XY, BAD_FRAME_Z = _sphere_center_from_corner_normal(
    (-BAD_FRAME_CUBE_HALF, BAD_FRAME_CUBE_HALF, _cube_top_z(BAD_FRAME_CUBE_HALF, BAD_FRAME_TABLE_TOP_Z)),
    BAD_FRAME_CUBE_TO_FINGER_NORMAL,
    DEFAULT_SPHERE_RADIUS,
)
LEFTFINGER_FRAME_XY = (-0.02923153, 0.01091796)
LEFTFINGER_FRAME_Z = 0.08423754

SCENARIOS: tuple[PressScenario, ...] = (
    PressScenario(
        name="top_center",
        sphere_xy=(0.0, 0.0),
        start_z=_cube_top_z() + DEFAULT_SPHERE_RADIUS,
        contact_kind="top",
        max_final_xy_m=0.010,
        max_final_tilt_deg=2.0,
        max_peak_omega_radps=3.0,
        max_final_omega_radps=0.30,
        max_final_z_m=0.060,
        min_final_z_m=0.040,
    ),
    PressScenario(
        name="top_offset",
        sphere_xy=(0.030, 0.0),
        start_z=_cube_top_z() + DEFAULT_SPHERE_RADIUS,
        contact_kind="top",
        max_final_xy_m=0.030,
        max_final_tilt_deg=6.0,
        max_peak_omega_radps=6.0,
        max_final_omega_radps=0.75,
        max_final_z_m=0.065,
        min_final_z_m=0.035,
    ),
    PressScenario(
        name="top_corner",
        sphere_xy=CORNER_XY,
        start_z=CORNER_Z,
        contact_kind="corner",
        max_final_xy_m=0.120,
        max_final_tilt_deg=45.0,
        max_peak_omega_radps=20.0,
        max_final_omega_radps=8.0,
        max_final_z_m=0.120,
        min_final_z_m=0.020,
    ),
    PressScenario(
        name="bad_frame_corner",
        sphere_xy=BAD_FRAME_XY,
        start_z=BAD_FRAME_Z,
        contact_kind="corner",
        max_final_xy_m=0.120,
        max_final_tilt_deg=45.0,
        max_peak_omega_radps=20.0,
        max_final_omega_radps=8.0,
        max_final_z_m=0.080,
        min_final_z_m=0.000,
        cube_half=BAD_FRAME_CUBE_HALF,
        sphere_radius=DEFAULT_SPHERE_RADIUS,
        table_top_z=BAD_FRAME_TABLE_TOP_Z,
        expected_cube_to_sphere_normal=BAD_FRAME_CUBE_TO_FINGER_NORMAL,
    ),
)

CONTACT_PROXY_SCENARIOS: tuple[PressScenario, ...] = (
    PressScenario(
        name="latest_leftfinger_face",
        sphere_xy=LEFTFINGER_FRAME_XY,
        start_z=LEFTFINGER_FRAME_Z,
        contact_kind="corner",
        max_final_xy_m=0.120,
        max_final_tilt_deg=45.0,
        max_peak_omega_radps=20.0,
        max_final_omega_radps=8.0,
        max_final_z_m=0.080,
        min_final_z_m=0.000,
        cube_half=BAD_FRAME_CUBE_HALF,
        sphere_radius=DEFAULT_SPHERE_RADIUS,
        table_top_z=BAD_FRAME_TABLE_TOP_Z,
        expected_cube_to_sphere_normal=LATEST_BAD_FRAME_CUBE_TO_LEFTFINGER_NORMAL,
        pusher_shape="box",
        pusher_quat=(0.9991463, -0.04031042, 0.00328010, -0.00848851),
        pusher_axis=(0.999726, -0.000014, -0.023398),
        pusher_box_half_extents=(0.0091, 0.0060, 0.0054),
        pusher_box_local_pos=(0.0, 0.0060, 0.0448),
    ),
    PressScenario(
        name="latest_leftfinger_cartesian",
        sphere_xy=LEFTFINGER_FRAME_XY,
        start_z=LEFTFINGER_FRAME_Z,
        contact_kind="corner",
        max_final_xy_m=0.120,
        max_final_tilt_deg=45.0,
        max_peak_omega_radps=20.0,
        max_final_omega_radps=8.0,
        max_final_z_m=0.080,
        min_final_z_m=0.000,
        cube_half=BAD_FRAME_CUBE_HALF,
        sphere_radius=DEFAULT_SPHERE_RADIUS,
        table_top_z=BAD_FRAME_TABLE_TOP_Z,
        expected_cube_to_sphere_normal=LATEST_BAD_FRAME_CUBE_TO_LEFTFINGER_NORMAL,
        pusher_shape="box",
        pusher_joint_kind="d6",
        pusher_mass=0.10,
        pusher_quat=(0.9991463, -0.04031042, 0.00328010, -0.00848851),
        pusher_axis=(0.999726, -0.000014, -0.023398),
        pusher_d6_target_scales=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        pusher_box_half_extents=(0.0091, 0.0060, 0.0054),
        pusher_box_local_pos=(0.0, 0.0060, 0.0448),
    ),
)
ALL_SCENARIOS: tuple[PressScenario, ...] = SCENARIOS + CONTACT_PROXY_SCENARIOS
ROW_TYPE_NAMES = {
    int(PGS_CONSTRAINT_TYPE_CONTACT): "contact",
    int(PGS_CONSTRAINT_TYPE_JOINT_TARGET): "joint_target",
    int(PGS_CONSTRAINT_TYPE_FRICTION): "friction",
    int(PGS_CONSTRAINT_TYPE_JOINT_LIMIT): "joint_limit",
    int(PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT): "joint_velocity_limit",
}


@dataclass(frozen=True)
class SceneIds:
    cube_body: int
    pusher_body: int
    table_shape: int
    cube_shape: int
    sphere_shape: int
    pusher_joint: int


def _shape_cfg(mu: float, *, density: float = 0.0) -> newton.ModelBuilder.ShapeConfig:
    cfg = newton.ModelBuilder.ShapeConfig(mu=mu, density=density)
    cfg.ke = 5.0e4
    cfg.kd = 5.0e2
    cfg.kf = 1.0e3
    cfg.margin = 0.0
    cfg.gap = CONTACT_GAP
    return cfg


def _make_solver(model: newton.Model, *, drive_mode: str) -> SolverFeatherPGS:
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
        drive_mode=drive_mode,
        friction_mode="bisection",
        pgs_schedule="contact_then_internal",
        double_buffer=False,
        use_parallel_streams=False,
    )


def build_scene(
    scenario: PressScenario,
    *,
    device: wp.context.Device | None = None,
    target_overlap: float = DEFAULT_TARGET_OVERLAP,
    effort_limit: float = DEFAULT_EFFORT_LIMIT,
    joint_limit_upper: float = DEFAULT_JOINT_LIMIT_UPPER,
    scene_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[newton.Model, SceneIds]:
    builder = newton.ModelBuilder(gravity=-9.81)
    builder.rigid_gap = CONTACT_GAP
    table_cfg = _shape_cfg(0.8)
    cube_cfg = _shape_cfg(0.8)
    sphere_cfg = _shape_cfg(0.8)
    offset = wp.vec3(*scene_offset)
    cube_inertia = (1.0 / 6.0) * CUBE_MASS * (2.0 * scenario.cube_half) ** 2
    pusher_mass = scenario.pusher_mass
    sphere_inertia = (2.0 / 5.0) * pusher_mass * scenario.sphere_radius**2
    if scenario.pusher_shape == "box":
        hx, hy, hz = scenario.pusher_box_half_extents
        pusher_inertia = wp.mat33(
            (1.0 / 12.0) * pusher_mass * ((2.0 * hy) ** 2 + (2.0 * hz) ** 2),
            0.0,
            0.0,
            0.0,
            (1.0 / 12.0) * pusher_mass * ((2.0 * hx) ** 2 + (2.0 * hz) ** 2),
            0.0,
            0.0,
            0.0,
            (1.0 / 12.0) * pusher_mass * ((2.0 * hx) ** 2 + (2.0 * hy) ** 2),
        )
    else:
        pusher_inertia = wp.mat33(
            sphere_inertia,
            0.0,
            0.0,
            0.0,
            sphere_inertia,
            0.0,
            0.0,
            0.0,
            sphere_inertia,
        )

    table_shape = builder.add_shape_box(
        body=-1,
        xform=wp.transform(
            offset + wp.vec3(0.0, 0.0, scenario.table_top_z - TABLE_HALF_Z),
            wp.quat_identity(),
        ),
        hx=TABLE_HALF_XY,
        hy=TABLE_HALF_XY,
        hz=TABLE_HALF_Z,
        cfg=table_cfg,
        label="table",
    )

    cube_body = builder.add_body(
        xform=wp.transform(offset + wp.vec3(0.0, 0.0, _cube_center_z(scenario)), wp.quat_identity()),
        mass=CUBE_MASS,
        inertia=wp.mat33(cube_inertia, 0.0, 0.0, 0.0, cube_inertia, 0.0, 0.0, 0.0, cube_inertia),
        lock_inertia=True,
        label="cube",
    )
    cube_shape = builder.add_shape_box(
        body=cube_body,
        hx=scenario.cube_half,
        hy=scenario.cube_half,
        hz=scenario.cube_half,
        cfg=cube_cfg,
        label="cube_shape",
    )

    start_pos = offset + wp.vec3(scenario.sphere_xy[0], scenario.sphere_xy[1], scenario.start_z)
    pusher_quat = wp.quat(*scenario.pusher_quat)
    pusher_body = builder.add_link(
        xform=wp.transform(start_pos, pusher_quat),
        mass=pusher_mass,
        inertia=pusher_inertia,
        lock_inertia=True,
        label="driven_pusher",
    )
    if scenario.pusher_shape == "box":
        hx, hy, hz = scenario.pusher_box_half_extents
        sphere_shape = builder.add_shape_box(
            body=pusher_body,
            xform=wp.transform(wp.vec3(*scenario.pusher_box_local_pos), wp.quat_identity()),
            hx=hx,
            hy=hy,
            hz=hz,
            cfg=sphere_cfg,
            label="driven_pusher_shape",
        )
    else:
        sphere_shape = builder.add_shape_sphere(
            body=pusher_body,
            radius=scenario.sphere_radius,
            cfg=sphere_cfg,
            label="driven_pusher_shape",
        )
    if scenario.pusher_joint_kind == "d6":
        drive_kd = 2.0 * math.sqrt(DRIVE_STIFFNESS)
        axes = [
            newton.ModelBuilder.JointDofConfig(
                axis=axis,
                target_pos=target_overlap * scenario.pusher_d6_target_scales[i],
                target_vel=0.0,
                target_ke=DRIVE_STIFFNESS,
                target_kd=drive_kd,
                effort_limit=effort_limit,
                velocity_limit=1.0,
                actuator_mode=JointTargetMode.POSITION,
            )
            for i, axis in enumerate((wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 1.0, 0.0), wp.vec3(0.0, 0.0, 1.0)))
        ]
        angular_axes = []
        if scenario.pusher_d6_angular_axes:
            angular_axes = [
                newton.ModelBuilder.JointDofConfig(
                    axis=axis,
                    target_pos=target_overlap * scenario.pusher_d6_target_scales[i + 3],
                    target_vel=0.0,
                    target_ke=DRIVE_STIFFNESS,
                    target_kd=drive_kd,
                    effort_limit=effort_limit,
                    velocity_limit=1.0,
                    actuator_mode=JointTargetMode.POSITION,
                )
                for i, axis in enumerate(
                    (wp.vec3(1.0, 0.0, 0.0), wp.vec3(0.0, 1.0, 0.0), wp.vec3(0.0, 0.0, 1.0))
                )
            ]
        pusher_joint = builder.add_joint_d6(
            parent=-1,
            child=pusher_body,
            parent_xform=wp.transform(start_pos, pusher_quat),
            child_xform=wp.transform_identity(),
            linear_axes=axes,
            angular_axes=angular_axes,
            label="pusher_d6",
        )
    else:
        pusher_joint = builder.add_joint_prismatic(
            parent=-1,
            child=pusher_body,
            parent_xform=wp.transform(start_pos, pusher_quat),
            child_xform=wp.transform_identity(),
            axis=wp.vec3(*scenario.pusher_axis),
            target_pos=target_overlap,
            target_vel=0.0,
            target_ke=DRIVE_STIFFNESS,
            target_kd=2.0 * math.sqrt(DRIVE_STIFFNESS),
            limit_lower=0.0,
            limit_upper=joint_limit_upper,
            effort_limit=effort_limit,
            velocity_limit=1.0,
            actuator_mode=JointTargetMode.POSITION,
            label="pusher_axis",
        )
    builder.add_articulation([pusher_joint], label="driven_pusher_articulation")

    model = builder.finalize(device=device)
    model.request_contact_attributes("force")
    return model, SceneIds(cube_body, pusher_body, table_shape, cube_shape, sphere_shape, pusher_joint)


def _box_tilt_deg(body_q_row: np.ndarray) -> float:
    qx, qy, _qz, _qw = body_q_row[3:7]
    local_up_dot_world_up = 1.0 - 2.0 * (qx * qx + qy * qy)
    local_up_dot_world_up = float(np.clip(local_up_dot_world_up, -1.0, 1.0))
    return math.degrees(math.acos(local_up_dot_world_up))


def _normals_from_a_to_b(contacts, shape_a: int, shape_b: int) -> np.ndarray:
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


def _contact_records_from_a_to_b(contacts, shape_a: int, shape_b: int) -> list[dict[str, object]]:
    """Return contact records in shape_a -> shape_b convention.

    For dynamic shapes the returned points are in that body's local frame; for
    static shapes Newton stores the point in world coordinates. This is useful
    for comparing the local cube-side contacts independent of scene origin.
    """

    count = int(contacts.rigid_contact_count.numpy()[0])
    shape0 = contacts.rigid_contact_shape0.numpy()[:count]
    shape1 = contacts.rigid_contact_shape1.numpy()[:count]
    normals = contacts.rigid_contact_normal.numpy()[:count]
    points0 = contacts.rigid_contact_point0.numpy()[:count]
    points1 = contacts.rigid_contact_point1.numpy()[:count]
    records: list[dict[str, object]] = []
    for i in range(count):
        s0 = int(shape0[i])
        s1 = int(shape1[i])
        if s0 == shape_a and s1 == shape_b:
            normal = np.array(normals[i], dtype=np.float64)
            point_a = np.array(points0[i], dtype=np.float64)
            point_b = np.array(points1[i], dtype=np.float64)
        elif s0 == shape_b and s1 == shape_a:
            normal = -np.array(normals[i], dtype=np.float64)
            point_a = np.array(points1[i], dtype=np.float64)
            point_b = np.array(points0[i], dtype=np.float64)
        else:
            continue
        records.append(
            {
                "normal": [float(x) for x in normal],
                "point_a": [float(x) for x in point_a],
                "point_b": [float(x) for x in point_b],
            }
        )
    return records


def _contact_pair_count(contacts, shape_a: int, shape_b: int) -> int:
    return int(_normals_from_a_to_b(contacts, shape_a, shape_b).shape[0])


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


def _pusher_dof_count(model: newton.Model, ids: SceneIds) -> int:
    starts = model.joint_qd_start.numpy()
    return int(starts[ids.pusher_joint + 1] - starts[ids.pusher_joint])


def _pusher_target_vector(scenario: PressScenario, target_overlap: float, dof_count: int) -> np.ndarray:
    if scenario.pusher_joint_kind == "d6":
        scales = np.array(scenario.pusher_d6_target_scales[:dof_count], dtype=np.float32)
        if len(scales) < dof_count:
            scales = np.pad(scales, (0, dof_count - len(scales)))
        return target_overlap * scales
    return np.array([target_overlap], dtype=np.float32)


def _apply_pusher_control_target(
    model: newton.Model,
    control: newton.Control,
    ids: SceneIds,
    scenario: PressScenario,
    target_overlap: float,
) -> tuple[int, int, np.ndarray]:
    dof_start = int(model.joint_qd_start.numpy()[ids.pusher_joint])
    dof_count = _pusher_dof_count(model, ids)
    target = _pusher_target_vector(scenario, target_overlap, dof_count)
    control_targets = control.joint_target_pos.numpy()
    control_targets[dof_start : dof_start + dof_count] = target
    control.joint_target_pos.assign(control_targets)
    return dof_start, dof_count, target


def _assert_initial_contacts(scenario: PressScenario, ids: SceneIds, contacts) -> dict[str, object]:
    total_contacts = int(contacts.rigid_contact_count.numpy()[0])
    table_to_cube = _normals_from_a_to_b(contacts, ids.table_shape, ids.cube_shape)
    cube_to_sphere = _normals_from_a_to_b(contacts, ids.cube_shape, ids.sphere_shape)

    if len(table_to_cube) < 3:
        raise AssertionError(
            f"{scenario.name}: expected at least 3 cube/table support contacts, got {len(table_to_cube)} "
            f"(total_contacts={total_contacts})"
        )
    table_z = table_to_cube[:, 2]
    if float(np.min(table_z)) < 0.90:
        raise AssertionError(f"{scenario.name}: cube/table normals should point upward; normals={table_to_cube}")

    if scenario.contact_kind == "top":
        if len(cube_to_sphere) != 1:
            raise AssertionError(
                f"{scenario.name}: expected exactly 1 sphere/cube top contact, got {len(cube_to_sphere)} "
                f"(total_contacts={total_contacts})"
            )
        n = cube_to_sphere[0]
        horizontal = float(np.linalg.norm(n[:2]))
        if n[2] < 0.95 or horizontal > 0.20:
            raise AssertionError(f"{scenario.name}: expected top normal near +Z, got {n}")
    else:
        if len(cube_to_sphere) < 1:
            raise AssertionError(f"{scenario.name}: expected at least 1 sphere/cube corner contact")
        if scenario.expected_cube_to_sphere_normal is not None:
            expected = np.array(scenario.expected_cube_to_sphere_normal, dtype=np.float64)
            expected /= float(np.linalg.norm(expected))
            dots = cube_to_sphere @ expected
            if float(np.max(dots)) < 0.97:
                raise AssertionError(
                    f"{scenario.name}: expected a cube-to-sphere normal close to {expected}, got {cube_to_sphere}"
                )
        else:
            best = cube_to_sphere[int(np.argmax(cube_to_sphere[:, 2]))]
            horizontal = float(np.linalg.norm(best[:2]))
            if best[2] < 0.45 or horizontal < 0.25 or best[0] < 0.15 or best[1] < 0.15:
                raise AssertionError(
                    f"{scenario.name}: expected cube-to-sphere corner normal with +X/+Y/+Z components, got {best}"
                )

    return {
        "initial_total_contacts": total_contacts,
        "initial_cube_table_contacts": len(table_to_cube),
        "initial_sphere_cube_contacts": len(cube_to_sphere),
        "initial_cube_table_min_nz": float(np.min(table_to_cube[:, 2])),
        "initial_cube_sphere_mean_normal": [float(x) for x in np.mean(cube_to_sphere, axis=0)],
    }


def initial_contact_report(
    scenario: PressScenario,
    *,
    device: wp.context.Device | None = None,
    contact_matching: Literal["disabled", "latest", "sticky"] = "sticky",
    target_overlap: float = DEFAULT_TARGET_OVERLAP,
    effort_limit: float = DEFAULT_EFFORT_LIMIT,
    joint_limit_upper: float = DEFAULT_JOINT_LIMIT_UPPER,
    scene_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict[str, object]:
    model, ids = build_scene(
        scenario,
        device=device,
        target_overlap=target_overlap,
        effort_limit=effort_limit,
        joint_limit_upper=joint_limit_upper,
        scene_offset=scene_offset,
    )
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    collision_pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        contact_matching=contact_matching,
        reduce_contacts=False,
    )
    contacts = collision_pipeline.contacts()
    collision_pipeline.collide(state, contacts)
    cube_to_table = _contact_records_from_a_to_b(contacts, ids.cube_shape, ids.table_shape)
    cube_to_pusher = _contact_records_from_a_to_b(contacts, ids.cube_shape, ids.sphere_shape)
    return {
        "scenario": scenario.name,
        "contact_matching": contact_matching,
        "total_contacts": int(contacts.rigid_contact_count.numpy()[0]),
        "cube_to_table_contacts": cube_to_table,
        "cube_to_pusher_contacts": cube_to_pusher,
        "initial_cube_table_contacts": len(cube_to_table),
        "initial_cube_pusher_contacts": len(cube_to_pusher),
    }


def _cube_dof_slice(model: newton.Model, ids: SceneIds, world_dof_start: int) -> slice | None:
    joint_child = model.joint_child.numpy()
    joint_qd_start = model.joint_qd_start.numpy()
    matches = np.nonzero(joint_child == ids.cube_body)[0]
    if len(matches) == 0:
        return None
    start = int(joint_qd_start[int(matches[0])]) - world_dof_start
    return slice(start, start + 6)


def _dense_row_summary(
    row: int,
    row_type: np.ndarray,
    row_parent: np.ndarray,
    phi: np.ndarray,
    rhs: np.ndarray,
    diag: np.ndarray,
    impulses: np.ndarray,
    row_mu: np.ndarray,
    j_rows: np.ndarray,
    y_rows: np.ndarray,
    cube_slice: slice | None,
) -> dict[str, object]:
    j = np.asarray(j_rows[row], dtype=np.float64)
    y = np.asarray(y_rows[row], dtype=np.float64)
    if cube_slice is None:
        j_cube = np.zeros((0,), dtype=np.float64)
        y_cube = np.zeros((0,), dtype=np.float64)
        j_other = j
        y_other = y
    else:
        mask = np.ones(j.shape[0], dtype=bool)
        mask[cube_slice] = False
        j_cube = j[cube_slice]
        y_cube = y[cube_slice]
        j_other = j[mask]
        y_other = y[mask]
    kind = int(row_type[row])
    return {
        "row": row,
        "type": ROW_TYPE_NAMES.get(kind, str(kind)),
        "type_id": kind,
        "parent": int(row_parent[row]),
        "phi": float(phi[row]),
        "rhs": float(rhs[row]),
        "diag": float(diag[row]),
        "impulse": float(impulses[row]),
        "mu": float(row_mu[row]),
        "j_norm": float(np.linalg.norm(j)),
        "y_norm": float(np.linalg.norm(y)),
        "j_cube": [float(x) for x in j_cube],
        "j_cube_norm": float(np.linalg.norm(j_cube)),
        "y_cube_norm": float(np.linalg.norm(y_cube)),
        "j_other_norm": float(np.linalg.norm(j_other)),
        "y_other_norm": float(np.linalg.norm(y_other)),
    }


def row_problem_report(
    scenario: PressScenario,
    *,
    device: wp.context.Device | None = None,
    drive_mode: str = "physx_pgs",
    contact_matching: Literal["disabled", "latest", "sticky"] = "sticky",
    target_overlap: float = DEFAULT_TARGET_OVERLAP,
    effort_limit: float = DEFAULT_EFFORT_LIMIT,
    joint_limit_upper: float = DEFAULT_JOINT_LIMIT_UPPER,
    dt: float = DEFAULT_DT,
) -> dict[str, object]:
    """Build one FeatherPGS step and summarize its row-level constraint problem."""

    model, ids = build_scene(
        scenario,
        device=device,
        target_overlap=target_overlap,
        effort_limit=effort_limit,
        joint_limit_upper=joint_limit_upper,
    )
    solver = _make_solver(model, drive_mode=drive_mode)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    collision_pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        contact_matching=contact_matching,
        reduce_contacts=False,
    )
    contacts = collision_pipeline.contacts()
    collision_pipeline.collide(state_0, contacts)

    _apply_pusher_control_target(model, control, ids, scenario, target_overlap)

    solver.step(state_0, state_1, control, contacts, dt)

    world = 0
    dense_count = int(solver.constraint_count.numpy()[world])
    mf_count = int(solver.mf_constraint_count.numpy()[world])
    world_dof_start = int(solver.world_dof_start.numpy()[world])
    cube_slice = _cube_dof_slice(model, ids, world_dof_start)

    row_type = solver.row_type.numpy()[world, :dense_count]
    row_parent = solver.row_parent.numpy()[world, :dense_count]
    phi = solver.phi.numpy()[world, :dense_count]
    rhs = solver.rhs.numpy()[world, :dense_count]
    diag = solver.diag.numpy()[world, :dense_count]
    impulses = solver.impulses.numpy()[world, :dense_count]
    row_mu = solver.row_mu.numpy()[world, :dense_count]
    j_rows = solver.J_world.numpy()[world, :dense_count]
    y_rows = solver.Y_world.numpy()[world, :dense_count]

    type_counts: dict[str, int] = {}
    for kind in row_type:
        name = ROW_TYPE_NAMES.get(int(kind), str(int(kind)))
        type_counts[name] = type_counts.get(name, 0) + 1

    dense_rows = [
        _dense_row_summary(row, row_type, row_parent, phi, rhs, diag, impulses, row_mu, j_rows, y_rows, cube_slice)
        for row in range(dense_count)
    ]
    dense_contact_rows = [row for row in dense_rows if row["type"] == "contact"]
    dense_drive_rows = [row for row in dense_rows if row["type"] == "joint_target"]
    contact_indices = [int(row["row"]) for row in dense_contact_rows]
    drive_indices = [int(row["row"]) for row in dense_drive_rows]
    a_matrix = np.asarray(j_rows, dtype=np.float64) @ np.asarray(y_rows, dtype=np.float64).T
    a_contact_contact = a_matrix[np.ix_(contact_indices, contact_indices)] if contact_indices else np.zeros((0, 0))
    a_contact_drive = a_matrix[np.ix_(contact_indices, drive_indices)] if contact_indices and drive_indices else np.zeros((0, 0))

    return {
        "scenario": scenario.name,
        "drive_mode": drive_mode,
        "contact_matching": contact_matching,
        "dense_count": dense_count,
        "mf_count": mf_count,
        "dof_count": int(solver.max_world_dofs),
        "world_dof_start": world_dof_start,
        "cube_dof_start": None if cube_slice is None else int(cube_slice.start),
        "cube_dof_count": 0 if cube_slice is None else int(cube_slice.stop - cube_slice.start),
        "dense_type_counts": type_counts,
        "dense_contact_rows": dense_contact_rows,
        "dense_drive_rows": dense_drive_rows,
        "a_contact_contact": [[float(x) for x in row] for row in a_contact_contact],
        "a_contact_drive": [[float(x) for x in row] for row in a_contact_drive],
        "a_contact_contact_diag": [float(x) for x in np.diag(a_contact_contact)],
        "a_contact_drive_max_abs": float(np.max(np.abs(a_contact_drive))) if a_contact_drive.size else 0.0,
    }


def _assert_bounded_motion(scenario: PressScenario, metrics: dict[str, object]) -> None:
    for key in ("final_x_m", "final_y_m", "final_z_m", "final_tilt_deg", "peak_cube_omega_radps"):
        value = float(metrics[key])
        if not math.isfinite(value):
            raise AssertionError(f"{scenario.name}: non-finite metric {key}={value}")

    final_xy = math.hypot(float(metrics["final_x_m"]), float(metrics["final_y_m"]))
    if final_xy > scenario.max_final_xy_m:
        raise AssertionError(
            f"{scenario.name}: final cube xy drift {final_xy:.6f} m exceeds {scenario.max_final_xy_m:.6f} m"
        )
    final_z = float(metrics["final_z_m"])
    if final_z < scenario.min_final_z_m or final_z > scenario.max_final_z_m:
        raise AssertionError(
            f"{scenario.name}: final cube z {final_z:.6f} outside "
            f"[{scenario.min_final_z_m:.6f}, {scenario.max_final_z_m:.6f}]"
        )
    final_tilt = float(metrics["final_tilt_deg"])
    if final_tilt > scenario.max_final_tilt_deg:
        raise AssertionError(
            f"{scenario.name}: final tilt {final_tilt:.3f} deg exceeds {scenario.max_final_tilt_deg:.3f} deg"
        )
    peak_omega = float(metrics["peak_cube_omega_radps"])
    if peak_omega > scenario.max_peak_omega_radps:
        raise AssertionError(
            f"{scenario.name}: peak angular speed {peak_omega:.6f} rad/s exceeds "
            f"{scenario.max_peak_omega_radps:.6f} rad/s"
        )
    peak_speed = float(metrics["peak_cube_speed_mps"])
    if peak_speed > scenario.max_peak_speed_mps:
        raise AssertionError(
            f"{scenario.name}: peak linear speed {peak_speed:.6f} m/s exceeds "
            f"{scenario.max_peak_speed_mps:.6f} m/s"
        )
    final_omega = float(metrics["final_cube_omega_radps"])
    if final_omega > scenario.max_final_omega_radps:
        raise AssertionError(
            f"{scenario.name}: final angular speed {final_omega:.6f} rad/s exceeds "
            f"{scenario.max_final_omega_radps:.6f} rad/s"
        )
    drive_error = float(metrics["final_drive_error_m"])
    target_overlap = float(metrics["target_overlap_m"])
    if metrics["drive_mode"] == "augmented" and drive_error < 0.5 * target_overlap:
        raise AssertionError(
            f"{scenario.name}: drive target error {drive_error:.6f} m is too small; "
            "the pusher may have passed through the cube instead of loading the contact"
        )
    sphere_cube_force = float(metrics["peak_sphere_cube_force_n"])
    if sphere_cube_force < 1.0:
        raise AssertionError(
            f"{scenario.name}: peak sphere/cube contact force {sphere_cube_force:.6f} N is too small; "
            "the drive/contact stack is not loaded"
        )


def run_scenario(
    scenario: PressScenario,
    *,
    device: wp.context.Device | None = None,
    steps: int = DEFAULT_STEPS,
    dt: float = DEFAULT_DT,
    assert_assumptions: bool = False,
    drive_mode: str = "augmented",
    contact_matching: Literal["disabled", "latest", "sticky"] = "sticky",
    target_overlap: float = DEFAULT_TARGET_OVERLAP,
    effort_limit: float = DEFAULT_EFFORT_LIMIT,
    joint_limit_upper: float = DEFAULT_JOINT_LIMIT_UPPER,
    scene_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict[str, object]:
    model, ids = build_scene(
        scenario,
        device=device,
        target_overlap=target_overlap,
        effort_limit=effort_limit,
        joint_limit_upper=joint_limit_upper,
        scene_offset=scene_offset,
    )
    solver = _make_solver(model, drive_mode=drive_mode)
    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    collision_pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        contact_matching=contact_matching,
        reduce_contacts=False,
    )
    contacts = collision_pipeline.contacts()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)

    q_starts = model.joint_q_start.numpy()
    q_start = int(q_starts[ids.pusher_joint])
    q_count = int(q_starts[ids.pusher_joint + 1] - q_starts[ids.pusher_joint])
    _dof_start, dof_count, target_vector = _apply_pusher_control_target(
        model, control, ids, scenario, target_overlap
    )

    collision_pipeline.collide(state_0, contacts)
    contact_metrics = _assert_initial_contacts(scenario, ids, contacts) if assert_assumptions else {}

    peak_cube_speed = 0.0
    peak_cube_omega = 0.0
    peak_cube_speed_step = -1
    peak_cube_omega_step = -1
    peak_cube_speed_qd = np.zeros(6, dtype=np.float64)
    peak_cube_omega_qd = np.zeros(6, dtype=np.float64)
    peak_total_contacts = int(contacts.rigid_contact_count.numpy()[0])
    peak_cube_table_contacts = _contact_pair_count(contacts, ids.table_shape, ids.cube_shape)
    peak_sphere_cube_contacts = _contact_pair_count(contacts, ids.cube_shape, ids.sphere_shape)
    peak_cube_table_force = 0.0
    peak_sphere_cube_force = 0.0

    for step_idx in range(steps):
        state_0.clear_forces()
        collision_pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, dt)
        solver.update_contacts(contacts)
        state_0, state_1 = state_1, state_0

        body_qd = state_0.body_qd.numpy()
        cube_qd = body_qd[ids.cube_body]
        cube_speed = float(np.linalg.norm(cube_qd[:3]))
        cube_omega = float(np.linalg.norm(cube_qd[3:6]))
        if cube_speed > peak_cube_speed:
            peak_cube_speed = cube_speed
            peak_cube_speed_step = step_idx
            peak_cube_speed_qd = np.array(cube_qd, dtype=np.float64)
        if cube_omega > peak_cube_omega:
            peak_cube_omega = cube_omega
            peak_cube_omega_step = step_idx
            peak_cube_omega_qd = np.array(cube_qd, dtype=np.float64)
        peak_total_contacts = max(peak_total_contacts, int(contacts.rigid_contact_count.numpy()[0]))
        peak_cube_table_contacts = max(peak_cube_table_contacts, _contact_pair_count(contacts, ids.table_shape, ids.cube_shape))
        peak_sphere_cube_contacts = max(peak_sphere_cube_contacts, _contact_pair_count(contacts, ids.cube_shape, ids.sphere_shape))
        peak_cube_table_force = max(
            peak_cube_table_force, _contact_force_sum_between(contacts, ids.table_shape, ids.cube_shape)
        )
        peak_sphere_cube_force = max(
            peak_sphere_cube_force, _contact_force_sum_between(contacts, ids.cube_shape, ids.sphere_shape)
        )

    collision_pipeline.collide(state_0, contacts)
    body_q = state_0.body_q.numpy()
    body_qd = state_0.body_qd.numpy()
    joint_q = state_0.joint_q.numpy()
    cube_q = body_q[ids.cube_body]
    cube_qd = body_qd[ids.cube_body]
    sphere_q = body_q[ids.pusher_body]
    scene_offset_np = np.array(scene_offset, dtype=np.float64)
    cube_local_q = np.array(cube_q[:3], dtype=np.float64) - scene_offset_np
    sphere_local_q = np.array(sphere_q[:3], dtype=np.float64) - scene_offset_np
    joint_q_slice = joint_q[q_start : q_start + min(q_count, dof_count)]
    target_slice = target_vector[: len(joint_q_slice)]
    drive_error = float(np.linalg.norm(target_slice - joint_q_slice)) if len(joint_q_slice) else 0.0

    metrics: dict[str, object] = {
        "scenario": scenario.name,
        "drive_mode": drive_mode,
        "contact_matching": contact_matching,
        "steps": steps,
        "dt_s": dt,
        "scene_offset_m": [float(x) for x in scene_offset],
        "cube_half_m": scenario.cube_half,
        "sphere_radius_m": scenario.sphere_radius,
        "drive_effort_limit_n": effort_limit,
        "drive_saturation_error_m": effort_limit / DRIVE_STIFFNESS,
        "joint_limit_upper_m": joint_limit_upper,
        "target_overlap_m": target_overlap,
        "target_dof_pos": [float(x) for x in target_vector],
        "final_drive_error_m": drive_error,
        "final_x_m": float(cube_local_q[0]),
        "final_y_m": float(cube_local_q[1]),
        "final_z_m": float(cube_local_q[2]),
        "final_world_x_m": float(cube_q[0]),
        "final_world_y_m": float(cube_q[1]),
        "final_world_z_m": float(cube_q[2]),
        "final_tilt_deg": _box_tilt_deg(cube_q),
        "final_cube_speed_mps": float(np.linalg.norm(cube_qd[:3])),
        "final_cube_omega_radps": float(np.linalg.norm(cube_qd[3:6])),
        "peak_cube_speed_mps": peak_cube_speed,
        "peak_cube_omega_radps": peak_cube_omega,
        "peak_cube_speed_step": peak_cube_speed_step,
        "peak_cube_omega_step": peak_cube_omega_step,
        "peak_cube_speed_body_qd": [float(x) for x in peak_cube_speed_qd],
        "peak_cube_omega_body_qd": [float(x) for x in peak_cube_omega_qd],
        "final_sphere_z_m": float(sphere_local_q[2]),
        "final_sphere_world_z_m": float(sphere_q[2]),
        "final_joint_q_m": float(joint_q[q_start]) if q_count else 0.0,
        "final_joint_q": [float(x) for x in joint_q[q_start : q_start + q_count]],
        "peak_total_contacts": peak_total_contacts,
        "peak_cube_table_contacts": peak_cube_table_contacts,
        "peak_sphere_cube_contacts": peak_sphere_cube_contacts,
        "peak_cube_table_force_n": peak_cube_table_force,
        "peak_sphere_cube_force_n": peak_sphere_cube_force,
        "final_total_contacts": int(contacts.rigid_contact_count.numpy()[0]),
        "final_cube_table_contacts": _contact_pair_count(contacts, ids.table_shape, ids.cube_shape),
        "final_sphere_cube_contacts": _contact_pair_count(contacts, ids.cube_shape, ids.sphere_shape),
    }
    metrics.update(contact_metrics)

    if assert_assumptions:
        _assert_bounded_motion(scenario, metrics)

    return metrics


def run_all(
    *,
    device: wp.context.Device | None = None,
    steps: int = DEFAULT_STEPS,
    dt: float = DEFAULT_DT,
    assert_assumptions: bool = False,
    drive_mode: str = "augmented",
    contact_matching: Literal["disabled", "latest", "sticky"] = "sticky",
    target_overlap: float = DEFAULT_TARGET_OVERLAP,
    effort_limit: float = DEFAULT_EFFORT_LIMIT,
    joint_limit_upper: float = DEFAULT_JOINT_LIMIT_UPPER,
    scene_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> list[dict[str, object]]:
    return [
        run_scenario(
            scenario,
            device=device,
            steps=steps,
            dt=dt,
            assert_assumptions=assert_assumptions,
            drive_mode=drive_mode,
            contact_matching=contact_matching,
            target_overlap=target_overlap,
            effort_limit=effort_limit,
            joint_limit_upper=joint_limit_upper,
            scene_offset=scene_offset,
        )
        for scenario in SCENARIOS
    ]


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--scenario", choices=[s.name for s in ALL_SCENARIOS] + ["all"], default="all")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT)
    parser.add_argument("--device", default=None)
    parser.add_argument("--drive-mode", choices=("augmented", "physx_pgs"), default="augmented")
    parser.add_argument("--contact-matching", choices=("disabled", "latest", "sticky"), default="sticky")
    parser.add_argument("--target-overlap", type=float, default=DEFAULT_TARGET_OVERLAP)
    parser.add_argument("--effort-limit", type=float, default=DEFAULT_EFFORT_LIMIT)
    parser.add_argument("--joint-limit-upper", type=float, default=DEFAULT_JOINT_LIMIT_UPPER)
    parser.add_argument(
        "--scene-offset",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "Z"),
        help="Translate the entire scene to test frame/large-coordinate sensitivity.",
    )
    parser.add_argument("--assert", dest="do_assert", action="store_true", help="Enable staged assertions.")
    parser.add_argument("--json", action="store_true", help="Print full JSON metrics.")
    parser.add_argument("--row-report", action="store_true", help="Print one-step FeatherPGS row summary JSON.")
    return parser


def main() -> None:
    args = create_parser().parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.target_overlap <= 0.0:
        raise ValueError("--target-overlap must be positive")
    if args.effort_limit <= 0.0:
        raise ValueError("--effort-limit must be positive")
    if args.joint_limit_upper <= 0.0:
        raise ValueError("--joint-limit-upper must be positive")
    device = wp.get_device(args.device) if args.device is not None else None
    scenarios = SCENARIOS if args.scenario == "all" else tuple(s for s in ALL_SCENARIOS if s.name == args.scenario)
    if args.row_report:
        rows = [
            row_problem_report(
                scenario,
                device=device,
                drive_mode=args.drive_mode,
                contact_matching=args.contact_matching,
                target_overlap=args.target_overlap,
                effort_limit=args.effort_limit,
                joint_limit_upper=args.joint_limit_upper,
                dt=args.dt,
            )
            for scenario in scenarios
        ]
        print(json.dumps(rows, indent=2, sort_keys=True))
        return

    rows = [
        run_scenario(
            scenario,
            device=device,
            steps=args.steps,
            dt=args.dt,
            assert_assumptions=args.do_assert,
            drive_mode=args.drive_mode,
            contact_matching=args.contact_matching,
            target_overlap=args.target_overlap,
            effort_limit=args.effort_limit,
            joint_limit_upper=args.joint_limit_upper,
            scene_offset=tuple(args.scene_offset),
        )
        for scenario in scenarios
    ]

    if args.json:
        print(json.dumps(rows, indent=2, sort_keys=True))
        return

    for row in rows:
        print(
            f"{row['scenario']}: "
            f"initial(table={row.get('initial_cube_table_contacts', '')}, sphere={row.get('initial_sphere_cube_contacts', '')}) "
            f"final_pos=({float(row['final_x_m']):+.4f}, {float(row['final_y_m']):+.4f}, {float(row['final_z_m']):+.4f}) "
            f"final_tilt={float(row['final_tilt_deg']):.2f}deg "
            f"peak_omega={float(row['peak_cube_omega_radps']):.3f}rad/s "
            f"final_omega={float(row['final_cube_omega_radps']):.3f}rad/s "
            f"drive_err={float(row['final_drive_error_m']):.4f}m "
            f"peak_forces(table={float(row['peak_cube_table_force_n']):.1f}N, "
            f"sphere={float(row['peak_sphere_cube_force_n']):.1f}N) "
            f"final_contacts(table={row['final_cube_table_contacts']}, sphere={row['final_sphere_cube_contacts']})"
        )


if __name__ == "__main__":
    main()
