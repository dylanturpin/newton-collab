# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""A contact-released bottle flip with a free-flight, upright landing."""

import itertools
import json
import math
from pathlib import Path

import numpy as np
import warp as wp
from robots import make_arm

import newton


def populate(b, info, variant, assets, device):
    from scene import plan_ik, tf  # noqa: PLC0415

    ee, _, fingers = make_arm(b, assets, device, robot="ur10", root=(-0.48, -0.25, 0), instrument=False)
    plan = json.loads((Path(__file__).parent / "assets" / "bottle_flip_throw.json").read_text())
    start = np.array([*plan["pickup_xy"], 0.160])
    # Use the shared robot setup to filter overlapping arm/hand import meshes.
    # The bottle is added afterwards, so all hand-bottle contacts stay enabled.
    plan_ik(
        b,
        ee,
        (0, 0, 0.1034),
        [(0.0, start, 0.04)],
        device,
        desired_rotation=wp.quat_rpy(math.pi / 2, 0.0, 0.0) * wp.quat_rpy(0.0, 0.0, math.pi / 2),
    )
    pickup_q = np.asarray(b.joint_q).copy()
    q0 = np.asarray(plan["q0"])
    q1 = np.asarray(plan["q1"])
    qd = np.asarray(plan["qd"]).copy()
    pickup_q[0], pickup_q[4] = math.pi, -math.pi
    q0[0], q0[4] = math.pi, -math.pi
    q1[0], q1[4] = math.pi, -math.pi
    qd[0] = qd[4] = 0.0
    braking = 0.120
    release_offset = 0.0305
    duration = plan["duration"]
    release = 1.5 + duration
    b.joint_q[:] = pickup_q.tolist()
    b.joint_q[-2:] = [0.04, 0.04]
    b.joint_target_q[:] = b.joint_q[:]
    waypoints = []
    for t in np.unique(
        np.r_[0.0, np.linspace(0.7, 1.5, 41), np.linspace(release - duration, release + 0.3, 86), 3.0, 7.0]
    ):
        u = np.clip((t - (release - duration)) / duration, 0, 1)
        q = q0.copy()
        q[:6] = (2 * u**3 - 3 * u**2 + 1) * q0[:6] + (-2 * u**3 + 3 * u**2) * q1 + (u**3 - u**2) * duration * qd
        if t < 1.5:
            lift = np.clip((t - 0.7) / 0.8, 0, 1)
            lift = lift * lift * (3 - 2 * lift)
            q[:6] = pickup_q[:6] * (1 - lift) + q0[:6] * lift
        if t > release:
            dt = min(t - release, braking)
            q[:6] = q1 + qd * (dt - dt * dt / (2 * braking))
        aperture = 0.04 - 0.025 * np.clip(t / 0.7, 0, 1)
        if t >= release + release_offset - 0.02:
            aperture = min(0.04, 0.015 + (t - release - release_offset + 0.02) * 0.9)
        q[-2:] = aperture
        waypoints.append((float(t), q))
    limits = np.minimum(np.asarray(b.joint_velocity_limit[:6]), 6.0)
    for (ta, qa), (tb, qb) in itertools.pairwise(waypoints):
        if np.any(np.abs(qb[:6] - qa[:6]) / (tb - ta) > limits + 1e-4):
            raise ValueError("Bottle throw exceeds the robot joint-speed limits")
    info["waypoints"] = waypoints
    for i in range(6):
        b.joint_target_ke[i], b.joint_target_kd[i], b.joint_effort_limit[i] = 16000, 220, 300
    for i in fingers:
        b.joint_target_ke[i], b.joint_target_kd[i], b.joint_effort_limit[i] = 3000, 40, 35
    for i, owner in enumerate(b.shape_body):
        if owner >= 0 and b.body_label[owner].endswith(("/fr3_leftfinger", "/fr3_rightfinger")):
            b.shape_material_mu[i] = 1.2
    # Rigid partial-fill approximation: a 25 g shell and 160 g water load.
    # No liquid/slosh solver is claimed; there is no release impulse or pose reset.
    body = b.add_body(
        xform=tf((start[0], start[1], 0)),
        mass=0.185,
        com=(0, 0, 0.030),
        inertia=wp.mat33(0.0004, 0, 0, 0, 0.0004, 0, 0, 0, 0.00015),
        lock_inertia=True,
        label="bottle/body",
    )
    cfg = b.default_shape_cfg.copy()
    cfg.density, cfg.mu, cfg.restitution, cfg.gap = 0, 0.65, 0.02, 0.00015
    # Convex lathed collision pieces follow the body, tapered shoulder and neck.
    from scipy.spatial import ConvexHull

    for rings in (
        ((0, 0.038), (0.005, 0.040), (0.075, 0.040), (0.105, 0.025), (0.175, 0.025)),
        ((0.175, 0.025), (0.184, 0.024), (0.210, 0.020), (0.215, 0.017)),
        ((0.215, 0.017), (0.25, 0.017)),
    ):
        vertices = np.array(
            [(r * math.cos(a), r * math.sin(a), z) for z, r in rings for a in np.linspace(0, math.tau, 49)[:-1]],
            dtype=np.float32,
        )
        hull = ConvexHull(vertices)
        faces = hull.simplices.copy()
        for i, face in enumerate(faces):
            v = vertices[face]
            if np.dot(np.cross(v[1] - v[0], v[2] - v[0]), hull.equations[i, :3]) < 0:
                faces[i] = face[::-1]
        shape = b.add_shape_mesh(
            body, mesh=newton.Mesh(vertices, faces.ravel(), compute_inertia=False), cfg=cfg, label="bottle/collision"
        )
        b.shape_flags[shape] &= ~int(newton.ShapeFlags.VISIBLE)
        if rings[0][0] >= 0.215:
            b.shape_material_mu[shape] = 1.2
    b.approximate_meshes(
        "convex_hull",
        keep_visual_shapes=False,
        shape_indices=[i for i, label in enumerate(b.shape_label) if label == "bottle/collision"],
    )
    # A smooth tapered shell with a colored sleeve and cap.
    visual = cfg.copy()
    visual.has_shape_collision = visual.has_particle_collision = False
    profile = [(0, 0), (0.038, 0), (0.040, 0.005), (0.040, 0.075)]
    # Recess the neck inside the cap to avoid coincident exterior surfaces.
    profile.extend([(0.025, 0.175), (0.024, 0.184), (0.020, 0.210), (0.017, 0.215), (0.016, 0.249), (0, 0.249)])
    vertices, normals, indices = [], [], []
    for a, c in itertools.pairwise(profile):
        tangent = np.subtract(c, a)
        n = np.array([tangent[1], -tangent[0]]) / np.linalg.norm(tangent)
        base = len(vertices)
        for radius, z in (a, c):
            for j in range(97):
                theta = j * math.tau / 96
                vertices.append((radius * math.cos(theta), radius * math.sin(theta), z))
                normals.append((n[0] * math.cos(theta), n[0] * math.sin(theta), n[1]))
        for j in range(96):
            k = base + j
            indices.extend((k, k + 1, k + 98, k, k + 98, k + 97))
    mesh = newton.Mesh(
        np.array(vertices, np.float32),
        np.array(indices, np.int32),
        normals=np.array(normals, np.float32),
        compute_inertia=False,
        roughness=0.26,
    )
    b.add_shape_mesh(body, mesh=mesh, cfg=visual, color=(0.72, 0.86, 0.91), label="bottle/molded-shell")
    b.add_shape_cylinder(
        body,
        xform=tf((0, 0, 0.045)),
        radius=0.0404,
        half_height=0.022,
        cfg=visual,
        color=(0.05, 0.43, 0.69),
        label="bottle/sleeve",
    )
    b.add_shape_cylinder(
        body,
        xform=tf((0, 0, 0.242)),
        radius=0.017,
        half_height=0.008,
        cfg=visual,
        color=(0.06, 0.57, 0.43),
        label="bottle/cap",
    )
    info.update(
        robot="ur10",
        bottle_hand=ee,
        tracked_body=body,
        dense_waypoints=True,
        phase_delay=0,
        bottle_com=[0, 0, 0.030],
        task="Flip a free bottle once and land upright",
        bottle_release_time=release + release_offset,
        water_model="Rigid partial-fill mass/inertia approximation; no liquid slosh simulation",
    )


def check(world, poses, fps):
    from scipy.spatial.transform import Rotation

    p = poses[:, world["body_start"] + world["tracked_body"]]
    rotation = Rotation.from_quat(p[:, 3:])
    up = rotation.apply([0, 0, 1])
    tilt = np.degrees(np.arccos(np.clip(up[:, 2], -1, 1)))
    com = p[:, :3] + rotation.apply(world["bottle_com"])
    spin = np.unwrap(np.arctan2(up[:, 0], up[:, 2]))
    result = {
        "peak_com_height_m": float(com[:, 2].max()),
        "rotation_turns": float(spin[-1] / math.tau),
        "final_tilt_deg": float(tilt[-1]),
        "resting_tilt_max_deg": float(tilt[-2 * fps :].max()),
        "resting_motion_m": float(np.linalg.norm(np.ptp(p[-2 * fps :, :3], axis=0))),
        "final_position": p[-1, :3].tolist(),
    }
    radial_z = np.sqrt(np.maximum(0, 1 - up[:, 2] ** 2))
    lowest = np.minimum.reduce(
        [
            p[:, 2] + z * up[:, 2] - radius * radial_z
            for z, radius in ((0, 0.038), (0.075, 0.040), (0.175, 0.025), (0.25, 0.017))
        ]
    )
    flying = np.flatnonzero((np.arange(len(p)) / fps > world["bottle_release_time"] + 0.12) & (lowest > 0.025))
    groups = np.split(flying, np.flatnonzero(np.diff(flying) > 1) + 1)
    flight = max(groups, key=len)
    gravity = 0.0
    if len(flight) >= 8:
        gravity = float(-2 * np.polyfit(flight / fps, com[flight, 2], 2)[0])
    result["free_flight_gravity_m_s2"] = gravity
    result["free_flight_duration_s"] = len(flight) / fps
    result["pass"] = bool(
        abs(gravity - 9.81) < 0.2
        and len(flight) / fps > 0.25
        and result["peak_com_height_m"] > 0.35
        and 0.9 < abs(result["rotation_turns"]) < 1.1
        and result["resting_tilt_max_deg"] < 3
        and result["resting_motion_m"] < 0.002
        and abs(p[-1, 2]) < 0.004
    )
    return result
