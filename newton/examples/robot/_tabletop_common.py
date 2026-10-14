# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Shared geometry and robot initialization for the tabletop examples."""

import math
from itertools import pairwise

import numpy as np
import warp as wp

import newton
import newton.ik as ik
import newton.utils


def tf(p=(0, 0, 0), q=None):
    return wp.transform(wp.vec3(*p), wp.quat_identity() if q is None else q)


def box(b, p, size, color, body=-1, label=""):
    return b.add_shape_box(body, xform=tf(p), hx=size[0], hy=size[1], hz=size[2], color=color, label=label)


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
    if getattr(b, "_tabletop_robot", None) in ("ur5", "ur10"):
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
    driven = getattr(b, "_tabletop_driven_dofs", 9 if any(pose[2] is not None for pose in poses) else 6)

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
                target[getattr(b, "_tabletop_grip_dofs", [7, 8])] = grip
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
    b._tabletop_ik_model = model
    for (ta, qa), (tb, qb) in pairwise(result):
        speed = np.max(np.abs(qb[:driven] - qa[:driven])) / (tb - ta)
        if speed > 6.0:
            raise ValueError(f"Discontinuous IK path at {ta:.2f}-{tb:.2f}s: {speed:.2f} rad/s")
    return [(t, q[:driven]) for t, q in result]
