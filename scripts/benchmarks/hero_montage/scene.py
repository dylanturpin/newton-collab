# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Nine procedural task templates for the FPGS blog hero prototype."""

import math
from itertools import pairwise
from pathlib import Path

import numpy as np
import warp as wp
import yaml
from furnishings import dress
from robots import make_arm

import newton
import newton.ik as ik

COLORS = [(0.035, 0.43, 0.78), (0.02, 0.65, 0.47), (0.96, 0.61, 0.065)]
TEMPLATES = ("lift", "stack", "sort", "hand", "spill", "insert", "g1", "drawer", "shadow")


def tf(p=(0, 0, 0), q=None):
    return wp.transform(wp.vec3(*p), wp.quat_identity() if q is None else q)


def box(b, p, size, color, body=-1, label=""):
    return b.add_shape_box(body, xform=tf(p), hx=size[0], hy=size[1], hz=size[2], color=color, label=label)


def prop(b, p, size, color, kind="box", label="prop"):
    body = b.add_body(xform=tf(p), label=label)
    if kind == "sphere":
        b.add_shape_sphere(body, radius=size[0], color=color)
    elif kind == "cylinder":
        b.add_shape_cylinder(body, radius=size[0], half_height=size[2], color=color)
    else:
        box(b, (0, 0, 0), size, color, body)
    return body


def gains(b, count, stiffness=700, damping=45):
    for i in range(count):
        b.joint_target_ke[i] = stiffness
        b.joint_target_kd[i] = damping
        b.joint_armature[i] = 0.08
        b.joint_target_mode[i] = int(newton.JointTargetMode.POSITION)
        b.joint_effort_limit[i] = 100
    b.joint_target_q[:] = b.joint_q[:]


def plan_ik(b, ee, offset, poses, device, desired_rotation=None, cartesian_step=None):
    """Solve a sparse waypoint path; the simulated drives track its interpolation."""
    robot_shapes = [i for i, body in enumerate(b.shape_body) if body >= 0]
    for i, a in enumerate(robot_shapes):
        for c in robot_shapes[i + 1 :]:
            b.add_shape_collision_filter_pair(a, c)
    model = b.finalize(device=device)
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    rotation = wp.quat(1.0, 0.0, 0.0, 0.0) if desired_rotation is None else desired_rotation
    if cartesian_step is not None:
        dense = [poses[0]]
        for a, c in pairwise(poses):
            n = max(1, math.ceil((c[0] - a[0]) / cartesian_step))
            qa = rotation if len(a) == 3 else a[3]
            qc = rotation if len(c) == 3 else c[3]
            for j in range(1, n + 1):
                u = j / n
                ease = u * u * u * (10 + u * (-15 + 6 * u))
                position = np.asarray(a[1]) * (1 - ease) + np.asarray(c[1]) * ease
                grip = None if a[2] is None else a[2] * (1 - ease) + c[2] * ease
                dense.append((a[0] + (c[0] - a[0]) * u, position, grip, wp.quat_slerp(qa, qc, ease)))
        poses = dense
    pos = ik.IKObjectivePosition(
        link_index=ee,
        link_offset=wp.vec3(*offset),
        target_positions=wp.array([wp.vec3(*poses[0][1])], dtype=wp.vec3, device=device),
    )
    rot = ik.IKObjectiveRotation(
        link_index=ee,
        link_offset_rotation=wp.quat_identity(),
        target_rotations=wp.array([wp.vec4(*rotation)], dtype=wp.vec4, device=device),
    )
    lower, upper = model.joint_limit_lower, model.joint_limit_upper
    if getattr(b, "_hero_robot", None) in ("ur5", "ur10"):
        # Keep the upper arm above its work surface. The native +/-2pi
        # shoulder range also permits equivalent poses through the plinth.
        lower_values, upper_values = lower.numpy(), upper.numpy()
        lower_values[1], upper_values[1] = -math.pi, -0.05
        lower, upper = wp.array(lower_values, device=device), wp.array(upper_values, device=device)
        model.joint_limit_lower.assign(lower_values)
        model.joint_limit_upper.assign(upper_values)
    limits = ik.IKObjectiveJointLimit(joint_limit_lower=lower, joint_limit_upper=upper, weight=5)
    solver = ik.IKSolver(
        model=model,
        n_problems=1,
        objectives=[pos, rot, limits],
        lambda_initial=0.05,
        jacobian_mode=ik.IKJacobianType.ANALYTIC,
    )
    q = wp.array(np.asarray(b.joint_q, dtype=np.float32)[None], device=device)
    fallback = None
    local_fallback = None
    candidate_solver = None
    candidate_positions = None
    candidate_rotations = None
    seed_rng = np.random.default_rng(147)
    driven = getattr(b, "_hero_driven_dofs", 9 if any(pose[2] is not None for pose in poses) else 6)

    def path_jump(values, previous, t):
        if not result:
            return False
        values = values.copy()
        for joint, joint_type in enumerate(b.joint_type):
            if joint_type == newton.JointType.REVOLUTE:
                qi, vi = b.joint_q_start[joint], b.joint_qd_start[joint]
                choices = [values[qi] + turns * math.tau for turns in (-1, 0, 1)]
                choices = [v for v in choices if b.joint_limit_lower[vi] <= v <= b.joint_limit_upper[vi]]
                if choices:
                    values[qi] = min(choices, key=lambda value: abs(value - previous[qi]))
        return np.max(np.abs(values[:driven] - previous[:driven])) > 6 * (t - result[-1][0])

    # The first solve can stop in a local minimum from the asset's rest pose.
    # Continue around the complete path, then retry from a solved configuration.
    for _attempt in range(3):
        result, errors = [], []
        for pose in poses:
            t, xyz, grip = pose[:3]
            target_rotation = rotation if len(pose) == 3 else pose[3]
            rot.set_target_rotation(0, wp.vec4(*target_rotation))
            pos.set_target_position(0, wp.vec3(*xyz))
            previous = q.numpy()[0].copy()
            solver.step(q, q, iterations=96)
            target = q.numpy()[0].copy()
            newton.eval_fk(model, q.flatten(), model.joint_qd, state)
            actual = wp.transform(*state.body_q.numpy()[ee])
            position_error = np.linalg.norm(np.asarray(wp.transform_point(actual, wp.vec3(*offset))) - xyz)
            delta = wp.quat_inverse(target_rotation) * wp.transform_get_rotation(actual)
            angle_error = 2 * math.acos(min(1.0, abs(float(delta[3]))))
            if position_error >= 0.003 or angle_error >= 0.02 or path_jump(target, previous, t):
                if result:
                    if local_fallback is None:
                        local_fallback = ik.IKSolver(
                            model=model,
                            n_problems=1,
                            objectives=[pos, rot, limits],
                            sampler=ik.IKSampler.GAUSS,
                            noise_std=0.12,
                            n_seeds=48,
                            rng_seed=147,
                            lambda_initial=0.05,
                            jacobian_mode=ik.IKJacobianType.ANALYTIC,
                        )
                    recovery = local_fallback
                    q.assign(previous[None])
                else:
                    if fallback is None:
                        fallback = ik.IKSolver(
                            model=model,
                            n_problems=1,
                            objectives=[pos, rot, limits],
                            sampler=ik.IKSampler.UNIFORM,
                            n_seeds=48,
                            rng_seed=147,
                            lambda_initial=0.05,
                            jacobian_mode=ik.IKJacobianType.ANALYTIC,
                        )
                    recovery = fallback
                recovery.step(q, q, iterations=192)
                target = q.numpy()[0].copy()
                newton.eval_fk(model, q.flatten(), model.joint_qd, state)
                actual = wp.transform(*state.body_q.numpy()[ee])
                position_error = np.linalg.norm(np.asarray(wp.transform_point(actual, wp.vec3(*offset))) - xyz)
                delta = wp.quat_inverse(target_rotation) * wp.transform_get_rotation(actual)
                angle_error = 2 * math.acos(min(1.0, abs(float(delta[3]))))
                if result and (position_error >= 0.003 or angle_error >= 0.02 or path_jump(target, previous, t)):
                    # Keep every uniformly seeded solution so residual ties
                    # cannot arbitrarily select a distant arm branch.
                    if candidate_solver is None:
                        candidate_positions = wp.empty(48, dtype=wp.vec3, device=device)
                        candidate_rotations = wp.empty(48, dtype=wp.vec4, device=device)
                        candidate_solver = ik.IKSolver(
                            model=model,
                            n_problems=48,
                            objectives=[
                                ik.IKObjectivePosition(ee, wp.vec3(*offset), candidate_positions),
                                ik.IKObjectiveRotation(ee, wp.quat_identity(), candidate_rotations),
                                limits,
                            ],
                            lambda_initial=0.05,
                            jacobian_mode=ik.IKJacobianType.ANALYTIC,
                        )
                    candidate_positions.assign(np.tile(xyz, (48, 1)).astype(np.float32))
                    candidate_rotations.assign(np.tile(target_rotation, (48, 1)).astype(np.float32))
                    seeds = seed_rng.uniform(lower.numpy(), upper.numpy(), size=(48, len(previous))).astype(np.float32)
                    seeds[:16] = np.clip(
                        previous + seed_rng.normal(0, 0.2, size=(16, len(previous))), lower.numpy(), upper.numpy()
                    )
                    seeds[0] = previous
                    candidates = wp.array(seeds, device=device)
                    candidate_solver.step(candidates, candidates, iterations=192)
                    feasible = []
                    for candidate in candidates.numpy():
                        for joint, joint_type in enumerate(b.joint_type):
                            if joint_type != newton.JointType.REVOLUTE:
                                continue
                            qi, vi = b.joint_q_start[joint], b.joint_qd_start[joint]
                            choices = [candidate[qi] + turns * math.tau for turns in (-1, 0, 1)]
                            choices = [v for v in choices if b.joint_limit_lower[vi] <= v <= b.joint_limit_upper[vi]]
                            if choices:
                                candidate[qi] = min(choices, key=lambda value: abs(value - previous[qi]))
                        q.assign(candidate[None])
                        newton.eval_fk(model, q.flatten(), model.joint_qd, state)
                        actual = wp.transform(*state.body_q.numpy()[ee])
                        pe = np.linalg.norm(np.asarray(wp.transform_point(actual, wp.vec3(*offset))) - xyz)
                        delta = wp.quat_inverse(target_rotation) * wp.transform_get_rotation(actual)
                        ae = 2 * math.acos(min(1.0, abs(float(delta[3]))))
                        if pe < 0.003 and ae < 0.02:
                            feasible.append((float(np.linalg.norm(candidate - previous)), candidate.copy(), pe, ae))
                    if feasible:
                        _, target, position_error, angle_error = min(feasible, key=lambda item: item[0])
                        q.assign(target[None])
            errors.append((position_error, angle_error))
            if grip is not None:
                target[getattr(b, "_hero_grip_dofs", [7, 8])] = grip
            # Preserve the nearest equivalent coordinate for continuous or
            # +/-2pi joints instead of commanding a full-turn branch jump.
            for joint, joint_type in enumerate(b.joint_type):
                if joint_type != newton.JointType.REVOLUTE:
                    continue
                qi, vi = b.joint_q_start[joint], b.joint_qd_start[joint]
                choices = [target[qi] + turns * math.tau for turns in (-1, 0, 1)]
                choices = [v for v in choices if b.joint_limit_lower[vi] <= v <= b.joint_limit_upper[vi]]
                if choices:
                    target[qi] = min(choices, key=lambda value: abs(value - previous[qi]))
            q.assign(target[None])
            result.append((t, target))
        if max(e[0] for e in errors) < 0.003 and max(e[1] for e in errors) < 0.02:
            break
    else:
        worst = max(range(len(errors)), key=lambda i: errors[i][0])
        raise ValueError(f"Unreachable IK waypoint at {poses[worst][0]}s: {errors[worst]}")
    b.joint_q[:] = result[0][1].tolist()
    b.joint_target_q[:] = b.joint_q[:]
    b._hero_ik_model = model
    for (ta, qa), (tb, qb) in pairwise(result):
        speed = np.max(np.abs(qb[:driven] - qa[:driven])) / (tb - ta)
        if speed > 6.0:
            raise ValueError(f"Discontinuous IK path at {ta:.2f}-{tb:.2f}s: {speed:.2f} rad/s")
    return [(t, q[:driven]) for t, q in result]


def fk_pose(b, body, device):
    model = b.finalize(device=device)
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    return wp.transform(*state.body_q.numpy()[body])


def ur_arm(b, assets, device, variant=0):
    if variant:
        robot = ("ur5", "kuka", "ur10", "kinova")[variant % 4]
        root = (-0.45, 0, 0.27) if robot == "ur10" else (-0.42, 0, 0)
        ee, rotation, _ = make_arm(b, assets, device, robot=robot, root=root, gripper=False)
        if robot == "ur10":
            box(b, (-0.45, 0, 0.135), (0.09, 0.09, 0.135), (0.22, 0.27, 0.33), label="robot_pedestal")
        return ee, rotation
    b.add_mjcf(
        str(Path(assets["ur5e_menagerie"]) / "ur5e.xml"),
        xform=tf((-0.45, 0, 0.27)),
        floating=False,
        enable_self_collisions=False,
    )
    box(b, (-0.45, 0, 0.135), (0.09, 0.09, 0.135), (0.22, 0.27, 0.33), label="robot_pedestal")
    b.joint_q[:6] = [0, -1.2, 1.6, -1.95, -1.57, 0]
    gains(b, 6, 1000, 60)
    ee = next((i for i, x in enumerate(b.body_label) if x.endswith("wrist_3_link")), b.body_count - 1)
    b.approximate_meshes("convex_hull", keep_visual_shapes=True)
    b._hero_robot = "ur5"
    return ee, wp.quat(1.0, 0.0, 0.0, 0.0)


def goal(b, center, half_width, color, direction):
    x, y, z = center
    for side in (-1, 1):
        box(b, (x + side * half_width, y + direction * 0.08, z + 0.065), (0.009, 0.09, 0.065), color)
    box(b, (x, y + direction * 0.17, z + 0.065), (half_width, 0.009, 0.065), color)
    box(b, (x, y + direction * 0.08, z - 0.015), (half_width, 0.09, 0.005), (0.22, 0.27, 0.33))
    # White net bars are visual only so they cannot change the ball's dynamics.
    cfg = b.default_shape_cfg.copy()
    cfg.has_shape_collision = False
    for j in range(7):
        b.add_shape_box(
            -1,
            xform=tf((x - half_width + j * half_width / 3, y + direction * 0.178, z + 0.065)),
            hx=0.002,
            hy=0.002,
            hz=0.065,
            color=(0.94, 0.97, 1.0),
            cfg=cfg,
        )


def markings(b, center, width, length, z, football):
    cfg = b.default_shape_cfg.copy()
    cfg.has_shape_collision = False
    white = (0.88, 0.94, 0.97)

    def line(p, half, q=None):
        b.add_shape_box(-1, xform=tf(p, q), hx=half[0], hy=half[1], hz=0.0003, color=white, cfg=cfg)

    x, y = center
    line((x, y, z), (width / 2, 0.003))
    for j in range(48):
        a = j * math.tau / 48
        line(
            (x + 0.13 * math.cos(a), y + 0.13 * math.sin(a), z), (0.009, 0.002), wp.quat_rpy(0.0, 0.0, a + math.pi / 2)
        )
    if football:
        for sign in (-1, 1):
            yy = y + sign * (length / 2 - 0.14)
            line((x, yy, z), (0.22, 0.002))
            for xx in (x - 0.22, x + 0.22):
                line((xx, yy + sign * 0.07, z), (0.002, 0.07))


def prism_mesh(points, half_height, top_points=None):
    """A closed convex prism; used as one piece of a concave star decomposition."""
    count = len(points)
    top_points = points if top_points is None else top_points
    vertices = [(x, y, -half_height) for x, y in points] + [(x, y, half_height) for x, y in top_points]
    faces = []
    for i in range(1, count - 1):
        faces.extend([0, i + 1, i, count, count + i, count + i + 1])
    for i in range(count):
        j = (i + 1) % count
        faces.extend([i, j, count + j, i, count + j, count + i])
    # Split vertices at every face so mechanical edges retain flat normals.
    vertices = np.asarray(vertices, dtype=np.float32)[np.asarray(faces, dtype=np.int32)]
    triangles = vertices.reshape(-1, 3, 3)
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    return newton.Mesh(
        vertices=vertices,
        indices=np.arange(len(vertices), dtype=np.int32),
        normals=np.repeat(normals, 3, axis=0),
        roughness=0.40,
    )


def star_profile(points, outer, inner, angle=0.0):
    return [
        (
            (outer if i % 2 == 0 else inner) * math.cos(angle + math.pi * i / points),
            (outer if i % 2 == 0 else inner) * math.sin(angle + math.pi * i / points),
        )
        for i in range(points * 2)
    ]


def utensil(b, p, color, style="spoon", angle=0.0, label="utensil"):
    """Import the stock proc-gen mesh and its original convex cells."""
    from task_assets import import_object, load  # noqa: PLC0415 -- optional procedural tasks

    name = f"cutlery_{style}"
    meta, _ = load(name)
    mapping, _ = import_object(
        b,
        name,
        tf(p, wp.quat_rpy(0.0, 0.0, angle)),
        free_root=True,
        variant=int(np.argmin(np.linalg.norm(np.asarray(COLORS) - color, axis=1))),
    )
    body = mapping[meta["root"]]
    b.body_label[body] = label
    return body


def build_template(kind, variant, assets, device, reference=False):
    b = newton.ModelBuilder()
    b.default_shape_cfg.mu = 0.75
    b.default_shape_cfg.gap = 0.001
    b.default_shape_cfg.restitution = 0.02
    b.default_shape_cfg.density = 600
    info = {"kind": kind, "variant": variant, "waypoints": [], "policy": None, "kinematic": []}
    color = COLORS[variant % 3]
    # Separate, low plinths make each world read as an individual exhibit.
    box(b, (0, 0, -0.065), (1.15, 1.15, 0.065), (0.76, 0.81, 0.86), label="plinth")
    box(b, (0, -1.11, 0.002), (0.95, 0.014, 0.003), color, label="accent")
    if kind in ("lift", "insert"):
        from catalog_tasks import populate  # noqa: PLC0415 -- shared scene helpers

        populate(b, info, kind, variant, assets, device)
    elif kind in ("kit", "gear", "serve", "toy", "puzzle", "interlock", "pile"):
        from proc_tasks import populate  # noqa: PLC0415 -- mutually shared scene helpers

        populate(b, info, kind, variant, assets, device)
    elif kind in ("lift", "stack"):
        robot = ("franka", "kuka", "kinova", "xarm")[variant % 4]
        ee, _, fingers = make_arm(b, assets, device, robot=robot)
        info["robot"] = robot
        y = -0.28
        dest = 0.23
        half = (0.16, 0.028, 0.025) if kind == "lift" else (0.09, 0.026, 0.017)
        goal_z = 0.195 if kind == "lift" else (6 + variant % 3) * 0.035 + 0.028
        grip_height = 0.055 if kind == "lift" else 0.0
        poses = [
            (0, (0.11, y, 0.40), 0.04),
            (1.1, (0.11, y, 0.16), 0.04),
            (2, (0.11, y, half[2] + grip_height + 0.008), 0.04),
            (2.8, (0.11, y, half[2] + grip_height + 0.008), 0.015),
            (4.2, (0.11, y, 0.48), 0.015),
            (5.5, (0.16, dest, 0.48), 0.015),
            (7, (0.16, dest, goal_z + grip_height + 0.005), 0.015),
            (7.8, (0.16, dest, goal_z + grip_height + 0.005), 0.04),
            (9, (0.16, dest, 0.48), 0.04),
            (12, (0.11, y, 0.40), 0.04),
        ]
        info["waypoints"] = plan_ik(b, ee, (0, 0, 0.1034), poses, device)
        info["arm_dofs"] = len(info["waypoints"][0][1])
        if kind == "lift":
            # Assemble the missing span of a procedural toy suspension bridge.
            height = 0.17
            half = (0.16, 0.028, 0.025)
            goal_z = height + half[2]
            for x in (0.025, 0.295):
                for side in (-1, 1):
                    box(
                        b,
                        (x, dest + side * 0.078, height / 2),
                        (0.025, 0.025, height / 2),
                        COLORS[(variant + 1) % 3],
                        label="bridge_pier",
                    )
                box(b, (x, dest, height - 0.012), (0.060, 0.105, 0.012), COLORS[(variant + 1) % 3])
                for side in (-1, 1):
                    box(b, (x, dest + side * 0.105, height + 0.10), (0.015, 0.015, 0.10), color)
            for x in (-0.15, 0.47):
                box(b, (x, dest, height - 0.012), (0.115, 0.095, 0.012), (0.23, 0.28, 0.35))
                for side in (-1, 1):
                    box(b, (x, dest + side * 0.095, height + 0.035), (0.115, 0.010, 0.035), color)
            # Slender stringers support the removable road deck between the piers.
            for side in (-1, 1):
                box(
                    b,
                    (0.16, dest + side * 0.034, height - 0.012),
                    (0.18, 0.012, 0.012),
                    COLORS[(variant + 1) % 3],
                    label="bridge_stringer",
                )
            # Water, islands and a wheeled toy give the assembly a readable purpose.
            box(b, (0.16, dest, 0.006), (0.23, 0.21, 0.006), (0.035, 0.43, 0.78))
            for x in (-0.16, 0.48):
                box(b, (x, dest, 0.012), (0.11, 0.15, 0.012), COLORS[(variant + 1) % 3])
            car = prop(
                b, (0.47, dest, height + 0.025), (0.058, 0.026, 0.022), COLORS[(variant + 2) % 3], label="toy_car"
            )
            box(b, (0.006, 0, 0.025), (0.027, 0.024, 0.013), color, car)
            for x in (-0.035, 0.035):
                for yy in (-0.032, 0.032):
                    b.add_shape_sphere(car, xform=tf((x, yy, -0.014)), radius=0.017, color=(0.10, 0.13, 0.18))
            info["task"] = "Place the missing bridge span between two abutments"
        else:
            # A fully dynamic, alternating Jenga tower receives one final crosspiece.
            layers = 6 + variant % 3
            half = (0.09, 0.026, 0.017)
            goal_z = layers * 0.035 + 0.018
            box(b, (0.16, dest, 0.005), (0.16, 0.16, 0.005), (0.23, 0.28, 0.35))
            for level in range(layers):
                for j in range(3):
                    across = (j - 1) * 0.054
                    pos = (
                        0.16 + (across if level % 2 else 0),
                        dest + (0 if level % 2 else across),
                        0.028 + level * 0.035,
                    )
                    brick = b.add_body(
                        xform=tf(pos, wp.quat_rpy(0.0, 0.0, math.pi / 2 if level % 2 else 0.0)),
                        label=f"jenga_{level}_{j}",
                    )
                    box(b, (0, 0, 0), half, COLORS[(level + variant) % 3], brick)
                    # Small face inlays identify individual pieces without fake constraints.
                    box(b, (0.0903, 0, 0), (0.0003, 0.012, 0.007), (0.93, 0.95, 0.97), brick)
            goal_z += 0.010
            for j in range(4):
                prop(b, (0.51, -0.28 + j * 0.10, 0.018), half, COLORS[(j + variant) % 3], label="spare_jenga_piece")
            info["task"] = "Lift a loose Jenga block and extend the alternating tower"
        info["tracked_body"] = prop(b, (0.11, y, half[2] + 0.002), half, color, label="manipulated_object")
        if kind == "lift":
            box(
                b,
                (0, 0, 0.055),
                (0.027, 0.026, 0.025),
                COLORS[(variant + 2) % 3],
                info["tracked_body"],
                "bridge_grasping_key",
            )
        # A shaped source cradle stages the part for the gripper.
        for x in (-0.10, 0.32):
            box(b, (x, y, 0.022), (0.014, 0.075, 0.022), (0.23, 0.28, 0.35))
        box(b, (0.11, y - 0.13, 0.005), (0.23, 0.012, 0.005), COLORS[(variant + 2) % 3])
    elif kind == "insert":
        robot = ("franka", "ur5", "xarm", "kinova")[variant % 4]
        ee, _, fingers = make_arm(b, assets, device, robot=robot)
        info["robot"] = robot
        initial_rotation = wp.quat(1.0, 0.0, 0.0, 0.0)
        for i in range(b.joint_dof_count):
            b.joint_target_ke[i] = 1000 if i in fingers else 3000
            b.joint_target_kd[i] = 18 if i in fingers else 100
        twist = 0.30 + variant * 0.07
        aligned = wp.quat_rpy(0.0, 0.0, twist) * initial_rotation
        poses = [
            (0, (0.10, -0.29, 0.38), 0.04),
            (1.2, (0.10, -0.29, 0.24), 0.04),
            (2, (0.10, -0.29, 0.181), 0.04),
            (2.8, (0.10, -0.29, 0.181), 0.019),
            (4, (0.10, -0.29, 0.45), 0.019),
            (5.5, (0.16, 0.23, 0.40), 0.019, aligned),
            (6.4, (0.16, 0.23, 0.31), 0.019, aligned),
            (8, (0.16, 0.23, 0.181), 0.019, aligned),
            (10.8, (0.16, 0.23, 0.181), 0.019, aligned),
            (11.4, (0.16, 0.23, 0.181), 0.04, aligned),
            (12.6, (0.16, 0.23, 0.43), 0.04, aligned),
            (14, (0.10, -0.29, 0.38), 0.04),
        ]
        info["waypoints"] = plan_ik(b, ee, (0, 0, 0.1034), poses, device, cartesian_step=0.12)
        # Cartesian waypoints already sample an eased path. Easing every small
        # joint-space segment again introduces stop/start acceleration pulses.
        info["dense_waypoints"] = True
        info["arm_dofs"] = len(info["waypoints"][0][1])
        points = 5 + variant % 2
        profile = star_profile(points, 0.043, 0.026, angle=-twist)
        inner = star_profile(points, 0.0442, 0.0272)
        mouth = star_profile(points, 0.048, 0.031)
        outer = star_profile(points, 0.105, 0.105)
        b.default_shape_cfg.gap = 0.0001
        box(b, (0.10, -0.29, 0.006), (0.075, 0.075, 0.006), (0.22, 0.27, 0.33), label="key_pickup_stand")
        box(b, (0.16, 0.23, 0.006), (0.14, 0.14, 0.006), (0.22, 0.27, 0.33), label="socket_base")
        # Individual convex sectors preserve the star-shaped hole all the way down.
        for i in range(len(inner)):
            j = (i + 1) % len(inner)
            b.add_shape_mesh(
                -1,
                xform=tf((0.16, 0.23, 0.058)),
                mesh=prism_mesh([inner[i], outer[i], outer[j], inner[j]], 0.046),
                color=COLORS[(variant + 1) % 3],
                label=f"star_socket_sector_{i}",
            )
            b.add_shape_mesh(
                -1,
                xform=tf((0.16, 0.23, 0.121)),
                mesh=prism_mesh(
                    [inner[i], outer[i], outer[j], inner[j]], 0.017, [mouth[i], outer[i], outer[j], mouth[j]]
                ),
                color=COLORS[(variant + 1) % 3],
                label=f"star_socket_leadin_{i}",
            )
        # Effective density of a printed polymer part.
        b.default_shape_cfg.density = 600
        key = b.add_body(xform=tf((0.10, -0.29, 0.067)), label="star_insertion_key")
        for i in range(len(profile)):
            j = (i + 1) % len(profile)
            b.add_shape_mesh(
                key, mesh=prism_mesh([(0, 0), profile[i], profile[j]], 0.055), color=color, label=f"star_key_sector_{i}"
            )
        box(b, (0, 0, 0.114), (0.025, 0.024, 0.030), COLORS[(variant + 2) % 3], key, "insertion_grip_collar")
        b.add_shape_cylinder(key, xform=tf((0, 0, 0.073)), radius=0.014, half_height=0.020, color=color)
        # A clocking fin makes the initial angular mismatch visible.
        box(b, (0.025, 0, 0.134), (0.008, 0.008, 0.008), color, key)
        info["tracked_body"] = key
        info["key_twist"] = twist
        info["_servo_model"] = b._hero_ik_model
        info["servo_ee"] = ee
        info["profile_points"] = points
        info["profile_radial_clearance_m"] = 0.0012
        info["task"] = f"Rotate a {points}-point star key and insert it into a matching deep socket"
        for j in range(3):
            box(b, (0.50, -0.29 + 0.19 * j, 0.006), (0.070, 0.070, 0.006), (0.22, 0.27, 0.33))
            b.add_shape_cylinder(
                -1,
                xform=tf((0.50, -0.29 + 0.19 * j, 0.035)),
                radius=0.047,
                half_height=0.023,
                color=COLORS[(j + variant) % 3],
            )
    elif kind == "drawer":
        robot = ("franka", "kinova", "ur5", "kuka")[variant % 4]
        ee, _, fingers = make_arm(b, assets, device, robot=robot, root=(-0.30, -0.16, 0))
        for finger in fingers:
            b.joint_target_ke[finger] = 2000
            b.joint_target_kd[finger] = 18
            b.joint_effort_limit[finger] = 25
        info["robot"] = robot
        poses = [
            (0, (0.20, -0.17, 0.190), 0.04),
            (0.9, (0.20, 0.035, 0.190), 0.04),
            (1.6, (0.20, 0.035, 0.190), 0.0),
            (4.0, (0.11, -0.445, 0.190), 0.0),
            (4.5, (0.11, -0.445, 0.190), 0.04),
            (4.9, (0.05, -0.50, 0.35), 0.04),
            (5.5, (0.10, -0.50, 0.43), 0.04),
            (5.8, (0.29, -0.56, 0.40), 0.04),
            (6.5, (0.29, -0.56, 0.094), 0.04),
            (7.1, (0.29, -0.56, 0.094), 0.0),
            (8.0, (0.29, -0.56, 0.46), 0.0),
            (9.0, (0.15, -0.19, 0.46), 0.0),
            (10.0, (0.15, -0.19, 0.153), 0.0),
            (10.6, (0.15, -0.19, 0.153), 0.018),
            (11.5, (0.15, -0.19, 0.43), 0.018),
            (12.0, (0.15, -0.19, 0.43), 0.04),
            (15, (0.25, -0.56, 0.40), 0.04),
        ]
        handle_rotation = wp.quat_rpy(-math.pi / 2, 0.0, 0.0)
        down = wp.quat(1.0, 0.0, 0.0, 0.0)
        poses = [(*pose, handle_rotation if pose[0] <= 4.9 else down) for pose in poses]
        info["waypoints"] = plan_ik(
            b, ee, (0, 0, 0.1034), poses, device, desired_rotation=handle_rotation, cartesian_step=0.10
        )
        info["arm_dofs"] = len(info["waypoints"][0][1])
        info["dense_waypoints"] = True
        # Fixed cabinet with an actual bounded prismatic drawer articulation.
        box(b, (0.20, 0.40, 0.045), (0.32, 0.26, 0.045), (0.21, 0.26, 0.32), label="cabinet_base")
        for side in (-1, 1):
            box(b, (0.20 + side * 0.32, 0.40, 0.205), (0.012, 0.26, 0.095), color, label="cabinet_side")
        box(b, (0.20, 0.665, 0.205), (0.33, 0.012, 0.095), color, label="cabinet_back")
        box(b, (0.20, 0.40, 0.306), (0.33, 0.275, 0.012), (0.83, 0.86, 0.88), label="cabinet_worktop")
        cabinet = b.add_link(xform=tf(), label="cabinet_fixed_root")
        root_joint = b.add_joint_fixed(-1, cabinet)
        drawer = b.add_link(xform=tf((0.20, 0.40, 0.12)), label="sliding_utensil_drawer")
        joint = b.add_joint_prismatic(
            cabinet,
            drawer,
            parent_xform=tf((0.20, 0.40, 0.12)),
            axis=(0, 1, 0),
            limit_lower=-0.48,
            limit_upper=0.0,
            target_ke=0.0,
            target_kd=0.0,
            damping=20.0,
            armature=0.015,
            label="drawer_slide",
        )
        b.add_articulation([root_joint, joint])
        q_index = b.joint_q_start[joint]
        info["drawer_q"] = q_index
        info["drawer_ee"] = ee
        info["drawer_handle_local"] = [0, -0.365, 0.070]
        info["drawer_actuator_stiffness"] = 0.0
        info["drawer_body"] = drawer
        box(b, (0, 0, 0), (0.30, 0.235, 0.010), (0.23, 0.29, 0.34), drawer, "drawer_floor")
        for side in (-1, 1):
            box(b, (side * 0.29, 0, 0.046), (0.008, 0.235, 0.036), color, drawer)
            box(b, (0, side * 0.23, 0.046), (0.29, 0.008, 0.036), color, drawer)
        box(b, (0, -0.244, 0.055), (0.30, 0.010, 0.065), color, drawer, "drawer_front")
        box(b, (0, -0.365, 0.070), (0.10, 0.010, 0.014), (0.83, 0.86, 0.88), drawer, "drawer_handle")
        for xx in (-0.09, 0.09):
            box(b, (xx, -0.304, 0.070), (0.009, 0.056, 0.009), (0.58, 0.64, 0.68), drawer, "handle_standoff")
        # Two long cutlery channels plus a small miscellaneous compartment.
        box(b, (0.16, 0, 0.028), (0.008, 0.215, 0.018), (0.83, 0.86, 0.88), drawer)
        box(b, (-0.06, -0.05, 0.028), (0.215, 0.008, 0.018), (0.83, 0.86, 0.88), drawer)
        for j, yy in enumerate((-0.130, 0.035)):
            box(b, (-0.075, yy, 0.013), (0.19, 0.070, 0.003), COLORS[(j + variant + 1) % 3], drawer)
        for x in (0.18, 0.32):
            b.add_shape_cylinder(
                -1,
                xform=tf((x, -0.56, 0.040)),
                radius=0.012,
                half_height=0.040,
                color=(0.24, 0.29, 0.34),
                label="cutlery_pick_rest",
            )
        first = utensil(b, (0.25, -0.56, 0.074), COLORS[(variant + 1) % 3], "fork", label="sorted_fork")
        info["tracked_body"] = first
        info["placed_bodies"] = [first]
        from task_assets import load  # noqa: PLC0415 -- procedural contact geometry

        fork_meta, fork_arrays = load("cutlery_fork")
        info["fork_contact_vertices"] = np.concatenate(
            [fork_arrays[visual["prefix"] + "_v"] for visual in fork_meta["bodies"][0]["visuals"]]
        ).tolist()
        for j in range(3):
            utensil(
                b,
                (0.43, -0.52 + j * 0.070, 0.018),
                COLORS[(j + variant) % 3],
                ("spoon", "knife", "fork")[j],
                angle=(-1) ** j * 0.10,
                label=f"loose_cutlery_{j}",
            )
        info["task"] = "Grasp the handle, pull open an unpowered drawer, then place a proc-gen fork in its tray"
    elif kind == "shadow":
        from lighter_task import populate as populate_lighter  # noqa: PLC0415 -- optional hand task

        populate_lighter(b, info, assets, device)
    elif kind == "sort":
        ee, rotation = ur_arm(b, assets, device, variant)
        info["robot"] = ("ur5", "kuka", "ur10", "kinova")[variant % 4]
        # A real cylindrical striker is bolted to the robot flange.
        b.add_shape_cylinder(
            ee,
            xform=tf((0, 0, 0.13), wp.quat_inverse(rotation)),
            radius=0.060,
            half_height=0.027,
            color=color,
            label="air_hockey_striker",
        )
        b.add_shape_cylinder(
            ee,
            xform=tf((0, 0, 0.085), wp.quat_inverse(rotation)),
            radius=0.025,
            half_height=0.035,
            color=(0.16, 0.19, 0.24),
        )
        # Low-friction field with rebounding side cushions and two open goals.
        field = box(b, (0.18, 0.08, 0.014), (0.44, 0.65, 0.014), (0.045, 0.39, 0.43), label="hockey_field")
        b.shape_material_mu[field] = 0.02
        b.shape_material_restitution[field] = 0.5
        for x in (-0.27, 0.63):
            box(b, (x, 0.08, 0.055), (0.017, 0.67, 0.055), (0.18, 0.23, 0.29))
        for yy in (-0.58, 0.74):
            for x in (-0.11, 0.47):
                box(b, (x, yy, 0.055), (0.145, 0.017, 0.055), color)
            goal(b, (0.18, yy, 0.03), 0.14, color, direction=1 if yy > 0 else -1)
        markings(b, center=(0.18, 0.08), width=0.84, length=1.28, z=0.0285, football=False)
        poses = [
            (0, (0.18, -0.40, 0.16), None),
            (1, (0.18, -0.40, 0.066), None),
            (2.2, (0.18, -0.25, 0.066), None),
            (2.65, (0.18, 0.28, 0.066), None),
            (3.8, (0.18, 0.28, 0.25), None),
            (5, (0.24, -0.36, 0.25), None),
            (6, (0.24, -0.36, 0.066), None),
            (8, (0.08, 0.36, 0.066), None),
            (9, (0.08, 0.36, 0.25), None),
            (12, (0.18, -0.40, 0.16), None),
        ]
        info["waypoints"] = plan_ik(b, ee, (0, 0, 0.13), poses, device, desired_rotation=rotation)
        info["arm_dofs"] = len(info["waypoints"][0][1])
        puck_cfg = b.default_shape_cfg.copy()
        puck_cfg.mu = 0.04
        puck_cfg.restitution = 0.75
        puck = b.add_body(xform=tf((0.18, -0.15, 0.045)), label="hockey_puck")
        b.add_shape_cylinder(puck, radius=0.041, half_height=0.014, cfg=puck_cfg, color=COLORS[(variant + 2) % 3])
        info["tracked_body"] = puck
        info["task"] = "Strike a free puck through the far air-hockey goal"
    elif kind == "hand":
        b.add_mjcf(
            str(Path(assets["kuka_iiwa_14"]) / "iiwa14.xml"),
            xform=tf((-0.25, 0, 0)),
            floating=False,
            enable_self_collisions=False,
        )
        b.joint_q[:7] = [0, 0.5, 0, -1.3, 0, 0.8, 0]
        ee = b.body_count - 1
        arm_model = b.finalize(device=device)
        arm_state = arm_model.state()
        newton.eval_fk(arm_model, arm_model.joint_q, arm_model.joint_qd, arm_state)
        wrist_pose = wp.transform(*arm_state.body_q.numpy()[ee])
        wrist_q = wp.transform_get_rotation(wrist_pose)
        trained_rotation = wp.quat_rpy(0.0, -math.pi / 2, 0.0) * wp.quat_rpy(math.pi / 2, 0.0, 0.0)
        mount = wp.quat_inverse(wrist_q) * trained_rotation
        # Route the bracket away from the authored link7 attachment site
        # before rising: a vertical shaft from the wrist origin intersects
        # the link6 housing even when internal collision pairs are filtered.
        flange = wp.transform_point(wrist_pose, wp.vec3(0, 0, 0.045))
        elbow = flange + wp.vec3(0.15, 0, 0)
        hand_origin = elbow + wp.vec3(0, 0, 0.18)
        rear_mount = hand_origin + wp.quat_rotate(trained_rotation, wp.vec3(-0.0093, 0, -0.095))
        stand_foot = elbow + wp.vec3(0, 0.13, 0)
        stand_top = wp.vec3(rear_mount[0], stand_foot[1], rear_mount[2])
        mount_position = wp.transform_point(wp.transform_inverse(wrist_pose), hand_origin)
        palm = b.body_count
        b.add_urdf(
            str(Path(__file__).parent / "assets/hora/hand.urdf"),
            parent_body=ee,
            xform=tf(mount_position, mount),
            floating=False,
            enable_self_collisions=False,
            joint_ordering="dfs",
        )
        route = (flange, elbow, stand_foot, stand_top, rear_mount)
        for index, (start, end) in enumerate(pairwise(route)):
            direction = wp.quat_rotate_inv(wrist_q, end - start)
            midpoint = wp.transform_point(wp.transform_inverse(wrist_pose), (start + end) * 0.5)
            b.add_shape_cylinder(
                ee,
                xform=tf(midpoint, wp.quat_between_vectors(wp.vec3(0, 0, 1), direction)),
                radius=0.018,
                half_height=float(wp.length(direction)) * 0.5,
                color=(0.16, 0.19, 0.23),
                label=f"allegro_mount_adapter_{index}",
            )
        gains(b, b.joint_dof_count, 12000, 220)
        robot_shapes = [i for i, body in enumerate(b.shape_body) if body >= 0]
        for i, a in enumerate(robot_shapes):
            for c in robot_shapes[i + 1 :]:
                b.add_shape_collision_filter_pair(a, c)
        cache = np.load(Path(__file__).parent / "assets/hora/grasps-projected.npy")
        grasp_index = variant
        grasp = cache[grasp_index]
        order = [0, 1, 2, 3, 12, 13, 14, 15, 4, 5, 6, 7, 8, 9, 10, 11]
        indices = []
        lower, upper = [], []
        for k, joint_number in enumerate(order):
            j = next(j for j, label in enumerate(b.joint_label) if label.endswith(f"joint_{joint_number}.0"))
            qi, vi = b.joint_q_start[j], b.joint_qd_start[j]
            indices.append(qi)
            lower.append(b.joint_limit_lower[vi])
            upper.append(b.joint_limit_upper[vi])
            b.joint_q[qi] = float(grasp[k])
            b.joint_target_ke[vi] = 3.0
            b.joint_target_kd[vi] = 0.1
            b.joint_armature[vi] = 0.001
            b.joint_effort_limit[vi] = 0.5
        b.joint_target_q[:] = b.joint_q[:]
        for i in range(7):
            b.joint_target_ke[i] = 12000
            b.joint_target_kd[i] = 220
        palm_pose = fk_pose(b, palm, device)
        training_frame = tf((0, 0, 0.5), trained_rotation)
        relocation = palm_pose * wp.transform_inverse(training_frame)
        object_pose = relocation * wp.transform(*grasp[16:])
        piece = b.add_body(
            xform=object_pose,
            mass=0.05,
            inertia=wp.mat33(np.eye(3) * 0.05 * 0.064**2 / 6),
            lock_inertia=True,
            label="hora_free_rotation_cube",
        )
        cfg = b.default_shape_cfg.copy()
        cfg.density = 0
        cfg.gap = 0.0002
        cfg.mu = 0.8
        b.add_shape_box(piece, hx=0.032, hy=0.032, hz=0.032, color=color, cfg=cfg)
        visual = cfg.copy()
        visual.has_shape_collision = False
        for axis in range(3):
            for sign in (-1, 1):
                center = [0.0, 0.0, 0.0]
                center[axis] = sign * 0.03205
                half = [0.026, 0.026, 0.026]
                half[axis] = 0.00008
                b.add_shape_box(
                    piece,
                    xform=tf(center),
                    hx=half[0],
                    hy=half[1],
                    hz=half[2],
                    color=COLORS[(axis + (sign > 0)) % 3],
                    cfg=visual,
                    label="rotation_cube_face",
                )
        info["tracked_body"] = piece
        info["palm_body"] = palm
        info["waypoints"] = [
            (0, np.asarray(b.joint_q[:7], dtype=np.float32)),
            (15, np.asarray(b.joint_q[:7], dtype=np.float32)),
        ]
        info["hora_policy"] = {
            "q_indices": indices,
            "lower": lower,
            "upper": upper,
            "grasp_index": grasp_index,
            "controller_hz": 20,
        }
        info["task"] = (
            "Continuously reorient a free cube with the public HORA Allegro policy, on a stationary Kuka wrist"
        )
        # Display sockets and alternate keyed parts turn the hand scene into a puzzle station.
        for j in range(3):
            x, y = 0.63, -0.25 + j * 0.22
            box(b, (x, y, 0.01), (0.10, 0.08, 0.01), (0.22, 0.27, 0.33))
            for sign in (-1, 1):
                box(b, (x + sign * 0.065, y, 0.04), (0.01, 0.07, 0.03), COLORS[(j + variant) % 3])
            prop(
                b,
                (x, y, 0.055),
                (0.032, 0.032, 0.033),
                COLORS[(j + variant) % 3],
                "cylinder" if j % 2 else "box",
                label="puzzle_insert",
            )
    elif kind in ("g1", "go2"):
        key = "unitree_" + kind
        cfgname = "g1_29dof" if kind == "g1" else "go2"
        cfg = yaml.safe_load((Path(assets[key]) / "rl_policies" / f"{cfgname}.yaml").read_text())
        usd = "g1_isaac.usd" if kind == "g1" else "go2.usda"
        b.add_usd(
            str(Path(assets[key]) / "usd" / usd),
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            joint_ordering="dfs",
            hide_collision_shapes=True,
        )
        original_shape_count = len(b.shape_flags)
        b.approximate_meshes("convex_hull", keep_visual_shapes=True)
        # USD already supplied render meshes. The approximation helper also
        # creates visual copies of hidden collision meshes; keep those hidden.
        for i in range(original_shape_count, len(b.shape_flags)):
            b.shape_flags[i] &= ~int(newton.ShapeFlags.VISIBLE)
        b.joint_q[:3] = [0, -0.4 if variant % 2 == 0 else 0.4, 0.76 if kind == "g1" else 0.36]
        for i, body in enumerate(b.shape_body):
            if body >= 0:
                b.shape_opacity[i] = 1.0
                label = b.body_label[body].rsplit("/", 1)[-1].lower()
                if kind == "g1":
                    dark = any(
                        part in label for part in ("pelvis", "waist", "hand_", "ankle", "shoulder_pitch", "hip_pitch")
                    )
                    b.shape_color[i] = (
                        (0.09, 0.105, 0.12) if dark else ((0.50, 0.53, 0.57) if reference else (0.73, 0.76, 0.79))
                    )
                else:
                    dark = any(part in label for part in ("calf", "foot", "head", "hip"))
                    b.shape_color[i] = (0.09, 0.105, 0.12) if dark else (0.58, 0.61, 0.64)
                mesh = b.shape_source[i]
                if isinstance(mesh, newton.Mesh):
                    mesh.roughness = 0.55
                    mesh.metallic = 0.05
        b.joint_q[3:7] = [0, 0, 0, 1]
        b.joint_q[7:] = cfg["mjw_joint_pos"]
        for i in range(cfg["num_dofs"]):
            b.joint_target_ke[i + 6] = cfg["mjw_joint_stiffness"][i]
            b.joint_target_kd[i + 6] = cfg["mjw_joint_damping"][i]
            b.joint_armature[i + 6] = cfg["mjw_joint_armature"][i]
            b.joint_target_mode[i + 6] = int(newton.JointTargetMode.POSITION)
        b.joint_target_q[:] = b.joint_q[:]
        floor_cfg = b.default_shape_cfg.copy()
        floor_cfg.is_visible = False
        b.add_ground_plane(cfg=floor_cfg, label="locomotion_safety_floor")
        if reference:
            b.joint_q[1] = 0.0
            b.shape_flags[0] = 0
            b.shape_flags[1] = 0
            box(b, (1.0, 0, -0.025), (4.0, 2.0, 0.025), (0.37, 0.39, 0.43), label="demo_room_floor")
            box(b, (1.0, -0.86, 1.30), (4.0, 0.025, 1.30), (0.29, 0.33, 0.42), label="demo_room_backdrop")
            for j in range(72):
                b.add_shape_cylinder(
                    -1,
                    xform=tf((-2.8 + j * 0.11, -0.81, 1.30)),
                    radius=0.052,
                    half_height=1.30,
                    color=(0.34, 0.38, 0.47),
                    label="curtain_fold",
                )
        info["policy"] = {
            "path": str(Path(assets[key]) / "rl_policies" / ("mjw_g1_29DOF.onnx" if kind == "g1" else "mjw_go2.onnx")),
            "config": cfg,
        }
        info["tracked_body"] = 0
        # Low visual course marks retain a readable silhouette.
        for j in range(0 if reference else 8):
            a = j * math.pi / 4
            box(b, (0.87 * math.cos(a), 0.87 * math.sin(a), 0.004), (0.045, 0.025, 0.004), color)
    elif kind == "spill":
        if variant % 4 == 1:
            ee, rotation = ur_arm(b, assets, device, 2)
            info["robot"] = "ur10"
        else:
            ee, rotation = ur_arm(b, assets, device, variant)
            info["robot"] = ("ur5", "kuka", "ur10", "kinova")[variant % 4]
        mount = wp.quat_inverse(rotation)
        cup_offset = wp.vec3(0, 0, 0.27)
        poses = [
            (0, (0.10, -0.32, 0.30), None),
            (2, (0.10, -0.32, 0.46), None),
            (4, (0.20, 0.12, 0.48), None),
            (6, (0.20, 0.12, 0.48), None),
            (8, (0.20, 0.12, 0.48), None),
            (10, (0.20, 0.12, 0.48), None),
            (15, (0.20, 0.12, 0.48), None),
        ]
        # Tilt the cup about world X while IK holds its bottom above the run.
        tilted = wp.quat_rpy(-1.8, 0.0, 0.0) * rotation
        for j in (3, 4):
            poses[j] = (*poses[j], tilted)
        info["waypoints"] = plan_ik(b, ee, (0, 0, 0.27), poses, device, desired_rotation=rotation, cartesian_step=0.10)
        info["dense_waypoints"] = True
        info["arm_dofs"] = len(info["waypoints"][0][1])
        initial = fk_pose(b, ee, device)
        center = wp.transform_point(initial, cup_offset)
        from task_assets import collision_bounds, import_object, load  # noqa: PLC0415 -- optional procedural tasks

        # A stock uncapped jar is physically mounted to the tool flange.
        # Its fixed joint belongs to the robot tree; every poured brick is free.
        joint_count = b.joint_count
        mapping, _ = import_object(
            b, "pour_jar", tf(cup_offset, mount), scale=1.85, parent_body=ee, register_articulations=False
        )
        jar = mapping[load("pour_jar")[0]["root"]]
        b.articulation_end[-1] = b.joint_count
        for j in range(joint_count, b.joint_count):
            b.joint_articulation[j] = b.articulation_count - 1
        for si, owner in enumerate(b.shape_body):
            if owner == jar:
                b.shape_material_mu[si] = 0.08
        info["poured_bodies"] = []
        packed_bounds = []
        brick_names = ("brick_1x1", "brick_1x2", "brick_2x2", "brick_3x1", "brick_3x2", "brick_4x1")
        for i in range(18):
            p = np.asarray(center) + np.array(
                [((i % 2) - 0.5) * 0.060, (((i // 2) % 3) - 1) * 0.050, 0.014 + (i // 6) * 0.050]
            )
            name = brick_names[i % len(brick_names)]
            orientation = wp.quat_rpy(0.0, 0.0, 0.06 * (i % 2))
            low, high = collision_bounds(name, 1.6, orientation)
            bounds = (p + low, p + high)
            assert all(
                np.any(bounds[0] > other[1] + 0.001) or np.any(other[0] > bounds[1] + 0.001) for other in packed_bounds
            )
            packed_bounds.append(bounds)
            mapping, _ = import_object(
                b,
                name,
                tf(p, orientation),
                scale=1.6,
                free_root=True,
                finish_color=COLORS[(i + variant) % 3],
            )
            brick_body = mapping[load(name)[0]["root"]]
            info["poured_bodies"].append(brick_body)
            for si, owner in enumerate(b.shape_body):
                if owner == brick_body:
                    b.shape_material_mu[si] = 0.12
            if i == 0:
                info["tracked_body"] = brick_body
        info["pour_jar"] = jar
        info["contents_kind"] = "six shapes of hollow studded toy bricks"
        # Three chutes descend toward individual colored collection pockets.
        angle = -0.30
        chute_cfg = b.default_shape_cfg.copy()
        chute_cfg.mu = 0.08
        chute_cfg.gap = 0.0002
        for lane in range(3):
            x = 0.04 + lane * 0.19
            b.add_shape_box(
                -1,
                xform=tf((x, 0.34, 0.19), wp.quat_rpy(angle, 0.0, 0.0)),
                hx=0.089,
                hy=0.35,
                hz=0.012,
                color=COLORS[(lane + variant) % 3],
                cfg=chute_cfg,
            )
            for sign in (-1, 1):
                b.add_shape_box(
                    -1,
                    xform=tf((x + sign * 0.089, 0.34, 0.225), wp.quat_rpy(angle, 0.0, 0.0)),
                    hx=0.007,
                    hy=0.35,
                    hz=0.025,
                    color=(0.24, 0.29, 0.34),
                    cfg=chute_cfg,
                )
            box(b, (x, 0.81, 0.018), (0.087, 0.12, 0.018), COLORS[(lane + variant) % 3])
            for sign in (-1, 1):
                box(b, (x + sign * 0.089, 0.81, 0.06), (0.007, 0.12, 0.06), (0.24, 0.29, 0.34))
            box(b, (x, 0.93, 0.06), (0.096, 0.007, 0.06), (0.24, 0.29, 0.34))
            box(b, (x, 0.05, 0.13), (0.022, 0.025, 0.13), (0.24, 0.29, 0.34))
            # Fit each post below the tilted deck; the former 150 mm post
            # protruded through its low end and arrested passing bolts.
            support_top = 0.19 + (0.63 + 0.025 - 0.34) * math.tan(angle) - 0.012 / math.cos(angle) - 0.002
            box(b, (x, 0.63, support_top / 2), (0.022, 0.025, support_top / 2), (0.24, 0.29, 0.34))
        info["task"] = (
            "Pour six shapes of loose studded toy bricks from a proc-gen threaded jar down three collection chutes"
        )
    if not reference:
        dress(b, kind, variant, info, color)
    return b, info
