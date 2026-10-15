# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Two physical grippers turn a threaded nut while supporting its bolt."""

import math
import tempfile
from pathlib import Path

import numpy as np
import warp as wp
from robots import make_arm

import newton
import newton.examples


def populate(b, info, variant, assets, device):
    import trimesh
    from scene import COLORS, plan_ik, tf  # noqa: PLC0415 -- shared task helpers

    origin = np.array([0.10, 0.03, 0.15])
    builders, paths = [], []
    for index in range(2):
        arm = newton.ModelBuilder()
        arm.default_shape_cfg.gap = 0.0002
        robot = "franka" if index == 0 else "ur10"
        root = (-0.48, -0.40, 0) if index == 0 else (-0.48, 0.55, 0)
        ee, _, fingers = make_arm(arm, assets, device, robot=robot, root=root)
        if index == 0:
            rotation = wp.quat_rpy(-math.pi / 2, 0.0, 0.0) * wp.quat_rpy(0.0, 0.0, math.pi / 2)
            target = origin + np.array([0, 0, 0.012])
            poses = [(0, target, 0.022), (10, target, 0.022)]
        else:
            arm.joint_q[5] = -4.0
            rotation = wp.quat_rpy(0.0, 0.0, math.pi / 2) * wp.quat(1, 0, 0, 0)
            target = origin + np.array([0, 0, 0.138])
            poses = [(0, target, 0.029), (1, target, 0.029)]
            for start, end, a, c, grip in ((1.0, 7.0, 0.0, -math.tau, 0.029),):
                for t in np.linspace(start, end, round((end - start) * 10) + 1)[1:]:
                    u = (t - start) / (end - start)
                    angle = float(a + (c - a) * u * u * (3 - 2 * u))
                    poses.append((float(t), target, grip, wp.quat_rpy(0.0, 0.0, angle) * rotation))
            final_rotation = wp.quat_rpy(0.0, 0.0, -math.tau) * rotation
            poses.append((10, target, 0.029, final_rotation))
        for i in range(arm._hero_driven_dofs - 2):
            arm.joint_target_ke[i], arm.joint_target_kd[i] = 8000, 180
            arm.joint_effort_limit[i] = 180
        for i in fingers:
            arm.joint_target_ke[i], arm.joint_target_kd[i] = (2500 if index == 0 else 1400), 20
            arm.joint_effort_limit[i] = 35
        for si, owner in enumerate(arm.shape_body):
            if owner >= 0 and arm.body_label[owner].endswith(("/fr3_leftfinger", "/fr3_rightfinger")):
                arm.shape_material_mu[si] = 1.2
        path = plan_ik(arm, ee, (0, 0, 0.1034), poses, device, desired_rotation=rotation)
        builders.append(arm)
        paths.append(path)
    # Build each robot's IK problem separately, then combine their physical trees.
    times = sorted({t for path in paths for t, _ in path})
    values = []
    for path in paths:
        knots = np.array([t for t, _ in path])
        joints = np.stack([q for _, q in path])
        values.append(np.stack([np.interp(times, knots, joints[:, i]) for i in range(joints.shape[1])], axis=1))
    info["waypoints"] = [(t, q) for t, q in zip(times, np.concatenate(values, axis=1), strict=True)]
    for i, arm in enumerate(builders):
        b.add_builder(arm, label_prefix=f"assembly_arm_{i}")
    directory = newton.examples.download_external_git_folder(
        "https://github.com/isaac-sim/IsaacGymEnvs.git", "assets/factory/mesh/factory_nut_bolt"
    )
    bodies = []
    for name, filename, position, rotation, color in (
        ("bolt", "factory_bolt_m20_loose.obj", origin, wp.quat_identity(), COLORS[0]),
        (
            "nut",
            "factory_nut_m20_loose_subdiv_3x.obj",
            origin + np.array([0, 0, 0.082]),
            wp.quat_rpy(0.0, 0.0, math.pi / 8),
            COLORS[2],
        ),
    ):
        source = trimesh.load(directory / filename, force="mesh")
        mesh = newton.Mesh(
            np.asarray(source.vertices * 2, dtype=np.float32),
            np.asarray(source.faces.ravel(), dtype=np.int32),
            normals=np.asarray(source.vertex_normals, dtype=np.float32),
            roughness=0.3,
        )
        mesh.build_sdf(
            max_resolution=256,
            narrow_band_range=(-0.003, 0.003),
            margin=0.0003,
            cache_dir=Path(tempfile.gettempdir()) / "hero_thread_sdf",
        )
        cfg = b.default_shape_cfg.copy()
        cfg.mu, cfg.gap, cfg.restitution, cfg.density = 0.01, 0.0002, 0, 1000
        body = b.add_body(xform=tf(position, rotation), label=f"threaded/{name}")
        b.add_shape_mesh(body, mesh=mesh, color=color, cfg=cfg, label=f"threaded/{name}")
        bodies.append(body)
    info.update(
        robot="franka+ur10",
        tracked_body=bodies[1],
        bolt_body=bodies[0],
        dense_waypoints=True,
        phase_delay=0,
        thread_pitch_m=0.005,
        task="Hold a free threaded bolt with one arm and screw its nut on with the other gripper",
    )


def check(world, poses, fps):
    from scipy.spatial.transform import Rotation

    bolt = poses[:, world["body_start"] + world["bolt_body"]]
    nut = poses[:, world["body_start"] + world["tracked_body"]]
    rb, rn = Rotation.from_quat(bolt[:, 3:]), Rotation.from_quat(nut[:, 3:])
    relative = rb.inv().apply(nut[:, :3] - bolt[:, :3])
    angles = np.unwrap((rb.inv() * rn).as_euler("xyz")[:, 2])
    turns = float((angles[8 * fps] - angles[fps]) / math.tau)
    descent = float(relative[fps, 2] - relative[8 * fps, 2])
    drift = float(np.linalg.norm(bolt[:, :3] - bolt[0, :3], axis=1).max())
    alignment = (rb.inv() * rn).apply(np.tile([0, 0, 1], (len(nut), 1)))[:, 2]
    lateral = float(np.linalg.norm(relative[:, :2], axis=1).max())
    result = {
        "nut_turns": turns,
        "nut_descent_m": descent,
        "bolt_max_drift_m": drift,
        "minimum_thread_axis_alignment": float(alignment.min()),
        "maximum_thread_offset_m": lateral,
    }
    result["pass"] = bool(
        turns < -0.8 and 0.003 < descent < 0.008 and drift < 0.015 and alignment.min() > 0.98 and lateral < 0.003
    )
    return result
