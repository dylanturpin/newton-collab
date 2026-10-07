# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Simulate one heterogeneous CUDA FPGS batch and render it with Newton."""

import argparse
import hashlib
import json
import math
import os
import time
from itertools import pairwise, product
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import warp as wp
from audit import audit, catalog_check, demolition_check, drawer_placement, knife_slot_fit, pile_collection
from contact_geometry import surface_contacts
from hora_policy import HoraPolicy
from scene import TEMPLATES, build_template
from warp_nn.runtime import OnnxRuntime

import newton
import newton.ik as ik
from newton.solvers import SolverFeatherPGS
from newton.viewer import ViewerGL


@wp.kernel
def targets_at_frame(targets: wp.array2d[float], frame: int, out: wp.array[float]):
    i = wp.tid()
    out[i] = targets[frame, i]


@wp.kernel
def lighter_cam(
    q_indices: wp.array[int],
    v_indices: wp.array[int],
    closed_detents: wp.array[float],
    q: wp.array[float],
    qd: wp.array[float],
    forces: wp.array[float],
):
    i = wp.tid()
    angle = q[q_indices[i]]
    velocity = qd[v_indices[i]]
    # A preloaded torsion spring and localized closed-cap detent. Both
    # torques derive from an angle-only potential; there is no clock,
    # commanded cap pose or actuator. Thumb contact releases the detent.
    scaled = angle / 0.08
    forces[v_indices[i]] = 0.01 * (2.4 - angle) - closed_detents[i] * wp.exp(-scaled * scaled) - 0.001 * velocity


@wp.kernel
def compensate_drive_error(
    indices: wp.array[int], q: wp.array[float], integral: wp.array[float], targets: wp.array[float], dt: float
):
    i = wp.tid()
    j = indices[i]
    # Integral joint control compensates gravity and sustained contact loads.
    # It changes actuator commands only; bodies remain freely simulated.
    error = wp.clamp(targets[j] - q[j], -0.08, 0.08)
    integral[i] = wp.clamp(integral[i] + 3.0 * dt * error, -0.12, 0.12)
    targets[j] += integral[i]


@wp.kernel
def set_gripper_aperture(targets: wp.array[float], left: int, right: int, aperture: float):
    targets[left] = aperture
    targets[right] = aperture


@wp.kernel
def observation(
    q: wp.array[float],
    qd: wp.array[float],
    q0: int,
    v0: int,
    default: wp.array[float],
    previous: wp.array2d[float],
    n: int,
    command: wp.vec3,
    out: wp.array2d[float],
):
    rotation = wp.quat(q[q0 + 3], q[q0 + 4], q[q0 + 5], q[q0 + 6])
    v = wp.quat_rotate_inv(rotation, wp.vec3(qd[v0], qd[v0 + 1], qd[v0 + 2]))
    w = wp.quat_rotate_inv(rotation, wp.vec3(qd[v0 + 3], qd[v0 + 4], qd[v0 + 5]))
    g = wp.quat_rotate_inv(rotation, wp.vec3(0.0, 0.0, -1.0))
    for k in range(3):
        out[0, k] = v[k]
        out[0, k + 3] = w[k]
        out[0, k + 6] = g[k]
        out[0, k + 9] = command[k]
    for k in range(n):
        out[0, 12 + k] = q[q0 + 7 + k] - default[k]
        out[0, 12 + n + k] = qd[v0 + 6 + k]
        out[0, 12 + 2 * n + k] = previous[0, k]


@wp.kernel
def policy_target(action: wp.array2d[float], default: wp.array[float], start: int, scale: float, out: wp.array[float]):
    i = wp.tid()
    out[start + 7 + i] = default[i] + scale * action[0, i]


def interpolate(waypoints, t, smooth=True):
    if t <= waypoints[0][0]:
        return waypoints[0][1]
    for (ta, a), (tb, b) in pairwise(waypoints):
        if t <= tb:
            u = (t - ta) / (tb - ta)
            if smooth:
                u = u * u * u * (10 + u * (-15 + 6 * u))
            return a + (b - a) * u
    return waypoints[-1][1]


class Example:
    def __init__(self, args):
        self.args = args
        self.fps = 50
        self.dt = 1 / (self.fps * args.substeps)
        self.frames = round(args.duration * self.fps)
        assets = json.loads(Path(args.assets).read_text())
        self.worlds = []
        builder = newton.ModelBuilder()
        kinds = args.templates.split(",")
        cases = (
            [(item.split(":")[0], int(item.split(":")[1])) for item in args.cases.split(",")]
            if args.cases
            else [
                (kinds[(k + variant * 3) % len(kinds)], variant)
                for variant in range(args.copies)
                for k in range(len(kinds))
            ]
        )
        if args.roster:
            cases = [(w["kind"], w["variant"]) for w in json.loads(args.roster.read_text())["stations"]]
        for kind, variant in cases:
            print(f"Building {kind} / {variant}", flush=True)
            b, info = build_template(kind, variant, assets, "cuda:0", reference=args.straight_walk)
            info.update(
                body_start=builder.body_count,
                q_start=builder.joint_coord_count,
                v_start=builder.joint_dof_count,
                body_count=b.body_count,
                coord_count=b.joint_coord_count,
                dof_count=b.joint_dof_count,
                id=f"{kind}-{variant}",
            )
            builder.add_world(b, label_prefix=f"{len(self.worlds):02d}_{kind}")
            self.worlds.append(info)
        builder.add_ground_plane(height=-0.80, color=(0.78, 0.79, 0.78))
        self.model = builder.finalize(device="cuda:0")
        self.model.set_gravity((0, 0, -9.81))
        summary = {
            "worlds": [
                {k: v for k, v in w.items() if k not in ("waypoints", "policy") and not k.startswith("_")}
                for w in self.worlds
            ],
            "body_labels": self.model.body_label,
        }
        (args.output / "model-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.lighter_q = wp.array(
            [w["q_start"] + w["lighter_lid_q"] for w in self.worlds if w.get("lighter")], dtype=int
        )
        self.lighter_v = wp.array(
            [w["v_start"] + w["lighter_lid_v"] for w in self.worlds if w.get("lighter")], dtype=int
        )
        self.lighter_detents = wp.array(
            [w["passive_cam"]["closed_detent_torque"] for w in self.worlds if w.get("lighter")], dtype=float
        )
        self.pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="sap",
            rigid_contact_max=32768,
            reduce_contacts=newton.CollisionPipeline.ContactReductionConfig(
                body_pairs=True, body_pair_cell_size=0.06, body_pair_hysteresis=0.0
            ),
            contact_matching="disabled",
        )
        self.contacts = self.pipeline.contacts()
        self.solver = SolverFeatherPGS(
            self.model,
            pgs_mode="matrix_free",
            articulated_contact_response="immediate",
            enable_joint_limits=True,
            enable_joint_velocity_limits=True,
            pgs_iterations=args.iterations,
            mf_max_constraints=8192,
            dense_max_constraints=2048,
            angular_damping=0.02,
            friction_anchor_beta=0,
            pgs_warmstart=False,
            row_watermark=True,
            use_parallel_streams=False,
        )
        targets = np.tile(np.asarray(builder.joint_target_q, dtype=np.float32), (self.frames + 1, 1))
        self.policies = []
        for w in self.worlds:
            if w["waypoints"]:
                start = w["q_start"]
                n = len(w["waypoints"][0][1])
                for f in range(self.frames + 1):
                    targets[f, start : start + n] = interpolate(
                        w["waypoints"],
                        max(0, f / self.fps - w.get("phase_delay", w["variant"] * 0.25)),
                        smooth=not w.get("dense_waypoints", False),
                    )
            for local_index, points in w.get("extra_drives", []):
                for f in range(self.frames + 1):
                    targets[f, w["q_start"] + local_index] = interpolate(
                        points, max(0, f / self.fps - w.get("phase_delay", w["variant"] * 0.25))
                    )
            if w["policy"]:
                c = w["policy"]["config"]
                n = c["num_dofs"]
                runtime = OnnxRuntime(w["policy"]["path"], device="cuda:0")
                self.policies.append(
                    {
                        "world": w,
                        "runtime": runtime,
                        "n": n,
                        "default": wp.array(np.asarray(c["mjw_joint_pos"], dtype=np.float32)),
                        "obs": wp.zeros((1, 12 + 3 * n), dtype=float),
                        "prev": wp.zeros((1, n), dtype=float),
                    }
                )
        replay = os.environ.get("HERO_REPLAY_JOINT_TARGETS")
        if replay:
            targets = np.load(replay)["targets"]
            assert targets.shape == (self.frames + 1, self.model.joint_coord_count)
        self.targets = wp.array(targets)
        self.thumb_servos = []
        for w in self.worlds:
            if "_thumb_ik_model" not in w or w.get("thumb_frozen") or replay:
                continue
            model = w["_thumb_ik_model"]
            position = ik.IKObjectivePosition(
                w["thumb_body"], wp.vec3(*w["thumb_tip_point"]), wp.zeros(1, dtype=wp.vec3)
            )
            limits = ik.IKObjectiveJointLimit(model.joint_limit_lower, model.joint_limit_upper, weight=5)
            mask = np.zeros(model.joint_dof_count, dtype=bool)
            mask[w["thumb_command_coordinates"]] = True
            solver = ik.IKSolver(
                model,
                n_problems=1,
                objectives=[position, limits],
                joint_dof_mask=wp.array(mask, dtype=wp.bool),
                lambda_initial=0.01,
                jacobian_mode=ik.IKJacobianType.ANALYTIC,
            )
            self.thumb_servos.append((w, position, solver, wp.array(model.joint_q.numpy()[None])))
        integral_indices = [w["q_start"] + i for w in self.worlds for i in range(w.get("integral_drive_count", 0))]
        self.integral_indices = wp.array(integral_indices, dtype=int)
        self.drive_integral = wp.zeros(len(integral_indices), dtype=float)
        np.savez_compressed(args.output / "joint-targets.npz", targets=targets, fps=self.fps)
        self.hora_policies = [HoraPolicy(w, self.model) for w in self.worlds if "hora_policy" in w]
        self.insertion_servos = []
        for w in self.worlds:
            if "_servo_model" not in w:
                continue
            model = w["_servo_model"]
            position = ik.IKObjectivePosition(w["servo_ee"], wp.vec3(), wp.zeros(1, dtype=wp.vec3))
            rotation = ik.IKObjectiveRotation(w["servo_ee"], wp.quat_identity(), wp.zeros(1, dtype=wp.vec4))
            limits = ik.IKObjectiveJointLimit(model.joint_limit_lower, model.joint_limit_upper, weight=5)
            solver = ik.IKSolver(
                model,
                n_problems=1,
                objectives=[position, rotation, limits],
                lambda_initial=0.05,
                jacobian_mode=ik.IKJacobianType.ANALYTIC,
            )
            self.insertion_servos.append(
                (
                    w,
                    position,
                    rotation,
                    solver,
                    wp.array(np.asarray(model.joint_q.numpy())[None]),
                    {"position_i": np.zeros(3), "rotation_i": np.zeros(3), "depth": 0.194},
                )
            )
        self.reset()
        self.pipeline.collide(self.state_0, self.contacts)
        np.savez_compressed(
            args.output / "initial-state.npz",
            poses=self.state_0.body_q.numpy(),
            joint_q=self.state_0.joint_q.numpy(),
        )
        initial_contacts = surface_contacts(builder, self.state_0, self.contacts)
        (args.output / "initial-contacts.json").write_text(json.dumps(initial_contacts, indent=2) + "\n")
        # Check the actual CUDA IK branch, rather than a separately rebuilt
        # CPU configuration: redundant arm/wrist solutions can differ.
        robot_names = ("/ur", "/fr3/", "/franka_hand/", "/right_shadow_hand/", "/iiwa14/", "/instrument/")
        fixture_intersections = [
            row
            for row in initial_contacts
            if row["surface_gap_m"] < -0.002
            and (
                (row["body0"] == -1 and any(name in row["body_label1"] for name in robot_names))
                or (row["body1"] == -1 and any(name in row["body_label0"] for name in robot_names))
            )
        ]
        if fixture_intersections:
            raise ValueError(f"Initial robot/fixture intersections: {fixture_intersections}")
        self.graph = None
        print(
            f"Batch ready: {self.model.world_count} worlds, {self.model.body_count} bodies, "
            f"{self.model.joint_dof_count} DOFs",
            flush=True,
        )

    def reset(self):
        self.drive_integral.zero_()
        for world in self.worlds:
            world.pop("_thumb_released", None)
        for policy in self.hora_policies:
            policy.reset()
        for *_, servo in self.insertion_servos:
            servo["position_i"][:] = 0
            servo["rotation_i"][:] = 0
            servo["depth"] = 0.194
            servo.pop("grasp_relative", None)
            servo.pop("grasp_lost", None)
            servo.pop("previous_key_position", None)
            servo.pop("release_time", None)
            servo.pop("seat_time", None)
            servo.pop("seated_pose", None)
            servo.pop("arm_target", None)
            servo["seating_frames"] = 0
        for s in (self.state_0, self.state_1):
            wp.copy(s.joint_q, self.model.joint_q)
            wp.copy(s.joint_qd, self.model.joint_qd)
            newton.eval_fk(self.model, s.joint_q, s.joint_qd, s)
            s.clear_forces()
        self.solver.reset(self.state_0)
        self.pipeline.reset_contact_matching()
        self.pipeline.clear_body_pair_reduction_stats()

    def physics(self):
        for _ in range(self.args.substeps):
            self.state_0.clear_forces()
            if len(self.lighter_q):
                wp.launch(
                    lighter_cam,
                    dim=len(self.lighter_q),
                    inputs=[
                        self.lighter_q,
                        self.lighter_v,
                        self.lighter_detents,
                        self.state_0.joint_q,
                        self.state_0.joint_qd,
                        self.control.joint_f,
                    ],
                )
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self, f):
        wp.launch(
            targets_at_frame, dim=self.model.joint_coord_count, inputs=[self.targets, f, self.control.joint_target_q]
        )
        if len(self.integral_indices):
            wp.launch(
                compensate_drive_error,
                dim=len(self.integral_indices),
                inputs=[
                    self.integral_indices,
                    self.state_0.joint_q,
                    self.drive_integral,
                    self.control.joint_target_q,
                    1 / self.fps,
                ],
            )
        if self.thumb_servos:
            body_q, joint_q = self.state_0.body_q.numpy(), self.state_0.joint_q.numpy()
            commanded = self.control.joint_target_q.numpy()
            for w, position, solver, q in self.thumb_servos:
                t = f / self.fps
                if t < 1.8:
                    continue
                case = wp.transform(*body_q[w["body_start"] + w["tracked_body"]])
                lid = wp.transform(*body_q[w["body_start"] + w["lighter_lid"]])
                config = w["thumb_feedback"]
                point = wp.transform_point(lid, wp.vec3(*config.get("contact_point", [0.021, -0.012, 0.060])))
                hinge = wp.transform_point(case, wp.vec3(*w["lighter_hinge_anchor"]))
                axis = wp.transform_vector(case, wp.vec3(*w["lighter_hinge_axis"]))
                tangent = wp.normalize(wp.cross(axis, point - hinge))
                normal = wp.transform_vector(
                    lid, wp.normalize(wp.vec3(*config.get("contact_normal", [1.0, -0.25, 0.0])))
                )
                angle = joint_q[w["q_start"] + w["lighter_lid_q"]]
                if angle > config.get("release_angle", math.inf):
                    w["_thumb_released"] = True
                if w.get("_thumb_released"):
                    release = config.get("release_joints", w["thumb_command_rest"])
                    for local, value in zip(w["thumb_command_coordinates"], release, strict=True):
                        commanded[w["q_start"] + local] = value
                    continue
                approach = np.clip((t - 1.8) / 1.0, 0.0, 1.0)
                if angle > 1.5:
                    goal = point + 0.025 * normal - wp.transform_vector(case, wp.vec3(0.0, 0.025, 0.0))
                else:
                    goal = point + float(approach * config["lead"]) * tangent
                    goal += float(0.010 * (1 - approach) - config["depth"] * approach) * normal
                position.set_target_position(0, goal)
                count = w["arm_dofs"]
                seed = joint_q[w["q_start"] : w["q_start"] + count].copy()
                if "ik_seed" in config:
                    seed[w["thumb_command_coordinates"]] = config["ik_seed"]
                q.assign(seed[None])
                solver.reset()
                solver.step(q, q, iterations=20)
                result = q.numpy()[0]
                for local in w["thumb_command_coordinates"]:
                    commanded[w["q_start"] + local] = result[local]
            self.control.joint_target_q.assign(commanded)
        active = [
            item
            for item in self.insertion_servos
            if (5.4 if item[0].get("catalog_task") == "knife" else 5.7)
            <= f / self.fps - item[0]["variant"] * 0.25
            <= (self.args.duration if item[0].get("catalog_task") == "knife" else 11.4)
        ]
        if active:
            body_q, joint_q = self.state_0.body_q.numpy(), self.state_0.joint_q.numpy()
            for w, position, rotation, solver, q, servo in active:
                t = f / self.fps - w.get("phase_delay", w["variant"] * 0.25)
                key = wp.transform(*body_q[w["body_start"] + w["tracked_body"]])
                hand = wp.transform(*body_q[w["body_start"] + w["servo_ee"]])
                relative = wp.transform_inverse(hand) * key
                key_position = np.asarray(wp.transform_get_translation(key))
                knife = w.get("catalog_task") == "knife"
                if knife:
                    # Calibrate the grasp once while clear of the fixture.
                    # Re-estimating it under contact would let the controller
                    # wind the gripper around a knife sliding in its jaws.
                    if "grasp_relative" not in servo:
                        servo["grasp_relative"] = relative
                    grasp_point = wp.vec3(*w["knife_grasp_point"])
                    grasp_error = np.linalg.norm(
                        np.asarray(wp.transform_point(relative, grasp_point))
                        - np.asarray(wp.transform_point(servo["grasp_relative"], grasp_point))
                    )
                    grasp_delta = wp.quat_inverse(
                        wp.transform_get_rotation(servo["grasp_relative"])
                    ) * wp.transform_get_rotation(relative)
                    grasp_angle = 2 * math.acos(min(1.0, abs(float(grasp_delta[3]))))
                    if "release_time" not in servo and (grasp_error > 0.006 or grasp_angle > math.radians(5)):
                        servo["grasp_lost"] = True
                    relative = servo["grasp_relative"]
                if knife:
                    goal_rotation = (
                        wp.transform_get_rotation(servo["seated_pose"])
                        if "seated_pose" in servo
                        else wp.quat_identity()
                    )
                else:
                    goal_rotation = wp.quat_rpy(0.0, 0.0, float(w["key_twist"]))
                error_q = goal_rotation * wp.quat_inverse(wp.transform_get_rotation(key))
                angle_error = np.asarray(error_q)[:3] * (2 if error_q[3] >= 0 else -2)
                goal_xy = np.asarray(w["knife_goal_pose"][:2]) if knife else np.array([0.16, 0.23])
                if knife and "seated_pose" in servo:
                    goal_xy = np.asarray(wp.transform_get_translation(servo["seated_pose"]))[:2]
                position_error = np.r_[goal_xy, key_position[2]] - key_position
                if not knife or "seat_time" not in servo:
                    position_gain = 2.0 if knife else 5.0
                    servo["position_i"] = np.clip(
                        servo["position_i"] + position_error * (position_gain / self.fps), -0.006, 0.006
                    )
                rotation_gain, rotation_limit = (0.5, 0.02) if knife else (2.0, 0.10)
                if not knife or "seat_time" not in servo:
                    servo["rotation_i"] = np.clip(
                        servo["rotation_i"] + angle_error * (rotation_gain / self.fps), -rotation_limit, rotation_limit
                    )
                if knife and t <= 7.0:
                    z = float(interpolate([(5.4, 0.214), (6.2, 0.184), (7.0, 0.158)], t))
                    servo["depth"] = z
                elif knife and "seat_time" in servo:
                    z = float(wp.transform_get_translation(servo["seated_pose"])[2] + 0.001)
                    servo["depth"] = z
                    if "release_time" in servo and t > servo["release_time"] + 0.8:
                        lift = np.clip((t - servo["release_time"] - 0.8) / 1.2, 0.0, 1.0)
                        z += 0.205 * lift**3 * (10 + lift * (-15 + 6 * lift))
                elif knife:
                    predicted_pose = body_q[w["body_start"] + w["tracked_body"]].copy()
                    predicted_pose[2] -= 0.001
                    fit = knife_slot_fit(w, predicted_pose)
                    aligned = not servo.get("grasp_lost", False) and (
                        np.linalg.norm(position_error[:2]) < 0.0008
                        if fit is None
                        else fit[0] > -0.05 and fit[1] > -0.05 and fit[2] > 3.0
                    )
                    if aligned:
                        speed = 0.0005 if key_position[2] < -0.0042 else 0.004
                        servo["depth"] = max(-0.0242, min(servo["depth"], key_position[2]) - speed)
                    else:
                        servo["depth"] = min(0.165, max(servo["depth"], key_position[2] + 0.0002))
                    z = float(servo["depth"])
                elif t <= 7.2:
                    z = float(interpolate([(5.7, 0.28), (6.4, 0.23), (7.2, 0.194)], t))
                else:
                    aligned = np.linalg.norm(position_error[:2]) < 0.0004 and np.linalg.norm(angle_error) < 0.0044
                    if aligned:
                        servo["depth"] = max(0.067, min(servo["depth"], key_position[2]) - 0.001)
                    else:
                        servo["depth"] = min(0.215, max(servo["depth"], key_position[2] + 0.0003))
                    z = float(servo["depth"])
                if knife:
                    previous = servo.get("previous_key_position", key_position)
                    speed = np.linalg.norm(key_position - previous) * self.fps
                    servo["previous_key_position"] = key_position.copy()
                    neck = np.asarray(wp.transform_point(key, wp.vec3(-0.0594, 0.0, w["knife_neck_seating_z"])))
                    seated = (
                        abs(neck[2] - w["knife_goal_pose"][2] - 0.19) < 0.003
                        and speed < 0.030
                        and not servo.get("grasp_lost", False)
                    )
                    servo["seating_frames"] = servo["seating_frames"] + 1 if seated else 0
                    # Stop driving downward at the first supporting contact.
                    # Holding the measured pose prevents integral windup from
                    # rocking the knife against the supporting slot lip.
                    if abs(neck[2] - w["knife_goal_pose"][2] - 0.19) < 0.002 and "seat_time" not in servo:
                        servo["seat_time"] = t
                        servo["seated_pose"] = key
                        servo["position_i"][:] = 0
                        servo["rotation_i"][:] = 0
                        w["measured_knife_seat_time"] = t
                    if (
                        t >= 9.5
                        and "seat_time" in servo
                        and t > servo["seat_time"] + 1.0
                        and servo["seating_frames"] >= 4
                        and "release_time" not in servo
                    ):
                        servo["release_time"] = t
                        w["measured_knife_release_time"] = t
                    release = np.clip((t - servo.get("release_time", t)) / 0.8, 0.0, 1.0)
                    closed = w["knife_closed_aperture"]
                    aperture = closed + (0.022 - closed) * release**3 * (10 + release * (-15 + 6 * release))
                    if "release_time" in servo and t > servo["release_time"] + 2.0:
                        aperture = 0.04
                    left, right = w["knife_grip_coords"]
                    wp.launch(
                        set_gripper_aperture,
                        dim=1,
                        inputs=[
                            self.control.joint_target_q,
                            w["q_start"] + left,
                            w["q_start"] + right,
                            float(aperture),
                        ],
                    )
                correction = wp.quat_rpy(*(float(v) for v in servo["rotation_i"]))
                goal = wp.transform(wp.vec3(*(np.r_[goal_xy, z] + servo["position_i"])), correction * goal_rotation)
                desired = goal * wp.transform_inverse(relative)
                position.set_target_position(0, wp.transform_get_translation(desired))
                rotation.set_target_rotation(0, wp.vec4(*wp.transform_get_rotation(desired)))
                count = w["arm_dofs"]
                q.assign(joint_q[w["q_start"] : w["q_start"] + count][None])
                solver.step(q, q, iterations=24)
                if knife:
                    target = q.numpy()[0]
                    previous = servo.get("arm_target", joint_q[w["q_start"] : w["q_start"] + count])
                    target[: count - 2] = previous[: count - 2] + np.clip(
                        target[: count - 2] - previous[: count - 2], -0.02, 0.02
                    )
                    servo["arm_target"] = target.copy()
                    q.assign(target[None])
                # Only arm drives are corrected; the gripper keeps its scheduled force.
                wp.copy(self.control.joint_target_q, q.flatten(), dest_offset=w["q_start"], count=count - 2)
        for p in self.policies:
            w = p["world"]
            # Small turning commands keep the policy-driven robots within their exhibits.
            speed = 0.20 if w["kind"] == "g1" else 0.28
            command = wp.vec3(speed, 0, 0.85 if w["variant"] % 2 == 0 else -0.85)
            if self.args.straight_walk:
                command = wp.vec3(0.40, 0, 0)
            if f < 50:
                command = wp.vec3()
            wp.launch(
                observation,
                dim=1,
                inputs=[
                    self.state_0.joint_q,
                    self.state_0.joint_qd,
                    w["q_start"],
                    w["v_start"],
                    p["default"],
                    p["prev"],
                    p["n"],
                    command,
                    p["obs"],
                ],
            )
            rt = p["runtime"]
            act = rt({rt.input_names[0]: p["obs"]})[rt.output_names[0]]
            wp.launch(
                policy_target,
                dim=p["n"],
                inputs=[
                    act,
                    p["default"],
                    w["q_start"],
                    float(w["policy"]["config"]["action_scale"]),
                    self.control.joint_target_q,
                ],
            )
            wp.copy(p["prev"], act)
        for policy in self.hora_policies:
            policy.step(f, self.fps, self.state_0, self.control)
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.physics()

    def simulate(self):
        print("Compiling physics and policy kernels", flush=True)
        self.step(0)
        self.reset()
        for p in self.policies:
            p["prev"].zero_()
        if self.args.use_graph:
            wp.synchronize()
            with wp.ScopedCapture() as capture:
                self.solver.seed_double_buffer_events()
                self.physics()
            self.graph = capture.graph
        self.reset()
        poses = [self.state_0.body_q.numpy()]
        joint_positions = [self.state_0.joint_q.numpy()]
        commands = []
        started = time.perf_counter()
        for f in range(self.frames):
            self.step(f)
            poses.append(self.state_0.body_q.numpy())
            joint_positions.append(self.state_0.joint_q.numpy())
            commands.append(self.control.joint_target_q.numpy())
            if f % 100 == 0:
                print(f"Simulated {f / self.fps:.1f}s/{self.args.duration}s", flush=True)
        commands.append(commands[-1].copy())
        np.savez_compressed(self.args.output / "executed-joint-targets.npz", targets=np.asarray(commands), fps=self.fps)
        self.poses = np.asarray(poses, dtype=np.float32)
        self.wall = time.perf_counter() - started
        summary = {
            "worlds": [
                {k: v for k, v in w.items() if k not in ("waypoints", "policy") and not k.startswith("_")}
                for w in self.worlds
            ],
            "body_labels": self.model.body_label,
        }
        (self.args.output / "model-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        np.savez_compressed(
            self.args.output / "trace.npz", poses=self.poses, joint_positions=np.asarray(joint_positions), fps=self.fps
        )
        self.test_final()

    def test_final(self):
        invalid = np.argwhere(~np.isfinite(self.poses).all(axis=2))
        assert not len(invalid), f"Non-finite physics state first at frame/body {invalid[0].tolist()}"
        self.watermarks = self.solver.constraint_row_watermarks()
        self.metrics = []
        for w in self.worlds:
            d = {
                "kind": w["kind"],
                "id": w["id"],
                "robot": w.get("robot", w["kind"]),
                "task": w.get("task", "Pretrained locomotion policy"),
                "variant": w["variant"],
                "dofs": w["dof_count"],
                "bodies": w["body_count"],
            }
            if "tracked_body" in w:
                p = self.poses[:, w["body_start"] + w["tracked_body"], :3]
                d.update(min_z=float(p[:, 2].min()), max_z=float(p[:, 2].max()), final_xyz=p[-1].tolist())
                if w.get("demolition"):
                    d.update(demolition_check(w, self.poses, self.fps))
                elif "catalog_task" in w:
                    result = catalog_check(w, self.poses, self.fps)
                    d.update(result)
                elif "placement_target" in w:
                    d["placement_error_m"] = float(np.linalg.norm(p[-1] - w["placement_target"]))
                    tolerance = 0.006 if w["kind"] in ("kit", "puzzle", "interlock") else 0.035
                    d["placed"] = d["placement_error_m"] < tolerance
                elif w["kind"] == "toy":
                    d["travel_m"] = float(p[-1, 1] - p[0, 1])
                    q = self.poses[-1, w["body_start"] + w["tracked_body"], 3:]
                    d["upright"] = bool(1 - 2 * (q[0] ** 2 + q[1] ** 2) > 0.95)
                    d["parked"] = bool(0.35 < p[-1, 1] < 0.76 and abs(p[-1, 0] - 0.18) < 0.15 and d["upright"])
                elif w["kind"] == "gear":
                    d["load_lift_m"] = float(np.max(p[:, 2]) - p[0, 2])
                    d["load_horizontal_travel_m"] = float(np.linalg.norm(p[-1, :2] - p[0, :2]))
                    d["load_transferred"] = bool(d["load_lift_m"] > 0.06 and d["load_horizontal_travel_m"] > 0.25)
                elif w["kind"] in ("lift", "stack"):
                    goal_z = 0.195 if w["kind"] == "lift" else (6 + w["variant"] % 3) * 0.035 + 0.028
                    d["placement_error_m"] = float(np.linalg.norm(p[-1] - [0.16, 0.23, goal_z]))
                    d["placed"] = d["placement_error_m"] < 0.06
                elif w["kind"] == "insert":
                    from scipy.spatial.transform import Rotation

                    angles = Rotation.from_quat(self.poses[-1, w["body_start"] + w["tracked_body"], 3:]).as_euler("xyz")
                    d["socket_xy_error_m"] = float(np.linalg.norm(p[-1, :2] - [0.16, 0.23]))
                    d["socket_tilt_degrees"] = float(np.rad2deg(np.linalg.norm(angles[:2])))
                    period = 2 * np.pi / w["profile_points"]
                    clocking = (angles[2] - w["key_twist"] + period / 2) % period - period / 2
                    d["socket_clocking_degrees"] = float(np.rad2deg(abs(clocking)))
                    d["seated"] = bool(
                        abs(p[-1, 2] - 0.067) < 0.0015
                        and d["socket_xy_error_m"] < 0.0012
                        and d["socket_tilt_degrees"] < 1.0
                        and d["socket_clocking_degrees"] < 1.0
                    )
                elif w["kind"] in ("sort", "domino"):
                    crossed = (p[:, 1] > 0.75) & (np.abs(p[:, 0] - 0.18) < 0.14) & (p[:, 2] < 0.16)
                    d["goal_crossed"] = bool(np.any(crossed))
                    d["travel_m"] = float(np.linalg.norm(p[-1] - p[0]))
                elif w["kind"] in ("hand", "shadow"):
                    palm = self.poses[:, w["body_start"] + w["palm_body"]]
                    distance = np.linalg.norm(p - palm[:, :3], axis=1)
                    d["max_object_palm_distance_m"] = float(distance.max())
                    d["retained"] = bool(np.all(distance < 0.18))
                elif w["kind"] in ("g1", "go2"):
                    d["upright_height"] = bool(p[:, 2].min() > (0.60 if w["kind"] == "g1" else 0.22))
                    d["inside_plinth"] = bool(np.abs(p[:, :2]).max() < 1.08)
            if w["kind"] == "drawer":
                drawer = self.poses[:, w["body_start"] + w["drawer_body"], :3]
                relative = self.poses[-1, [w["body_start"] + i for i in w["placed_bodies"]], :3] - drawer[-1]
                d["drawer_travel_m"] = float(np.ptp(drawer[:, 1]))
                d["drawer_opened"] = bool(drawer[0, 1] - drawer[-1, 1] > 0.35)
                d["utensil_relative_xyz"] = relative.tolist()
                d["utensils_sorted"] = drawer_placement(w, self.poses)["fork_in_tray"]
            if w["kind"] == "spill":
                ids = [w["body_start"] + i for i in w.get("poured_bodies", w.get("hardware_bodies", []))]
                positions = self.poses[-1, ids, :3]
                lane_offset = np.min(np.abs(positions[:, 0, None] - np.array([0.04, 0.23, 0.42])), axis=1)
                d["terminal_pocket_arrivals"] = int(
                    np.sum(
                        (lane_offset < 0.080)
                        & (positions[:, 1] > 0.69)
                        & (positions[:, 1] < 0.923)
                        & (positions[:, 2] > 0.035)
                        & (positions[:, 2] < 0.12)
                    )
                )
                d["poured_object_count"] = len(ids)
                d["contents"] = w.get("contents_kind", "hardware")
            if w["kind"] == "pile":
                collection = pile_collection(w, self.poses, self.fps)
                d["pile_displacements_m"] = collection["displacements_m"]
                d["pile_gathered_count"] = collection["dropped_into_bin_count"]
                d["pile_gathered"] = bool(d["pile_gathered_count"] >= w.get("pile_required_count", 10))

            self.metrics.append(d)
        print(json.dumps(self.metrics, indent=2), flush=True)

    def render(self):
        from PIL import Image

        args = self.args
        viewer = ViewerGL(
            width=args.width, height=args.height, headless=True, enable_cuda_interop=ViewerGL.CudaInterop.NONE
        )
        viewer.set_model(self.model)
        viewer.set_world_offsets((2.6, 2.6, 0))
        if self.model.world_count >= 16:
            columns = 9 if self.model.world_count == 36 else math.ceil(math.sqrt(self.model.world_count))
            rows = math.ceil(self.model.world_count / columns)
            order = [w["id"] for w in self.worlds]
            if args.roster:
                order = json.loads(args.roster.read_text()).get("display_order", order)
            assert sorted(order) == sorted(w["id"] for w in self.worlds)
            layout = np.asarray(
                [
                    (
                        (order.index(w["id"]) % columns - (columns - 1) / 2) * 2.6,
                        (order.index(w["id"]) // columns - (rows - 1) / 2) * 2.6,
                        0,
                    )
                    for w in self.worlds
                ]
            )
            viewer.world_offsets.assign(layout.astype(np.float32))
        viewer.renderer.msaa_samples = 8
        # The renderer allocates its initial FBO before this override.
        viewer.renderer._setup_frame_buffer()
        viewer.renderer.draw_sky = False
        viewer.renderer.background_color = (0.90, 0.93, 0.96)
        viewer.renderer.draw_fps = False
        viewer.renderer.spotlight_enabled = False
        viewer.renderer.shadow_extents = 18
        viewer.renderer._exposure = 1.15
        if args.straight_walk:
            sun = np.array([0.25, 0.60, 0.80])
            viewer.renderer._sun_direction = sun / np.linalg.norm(sun)
            viewer.renderer._exposure = 1.15
            viewer.renderer.shadow_extents = 5.0
        viewer.camera.fov = 43
        offsets = viewer.world_offsets.numpy()
        state = self.model.state()

        def frame(sample, eye, target):
            delta = np.asarray(target) - np.asarray(eye)
            viewer.set_camera(
                wp.vec3(*eye),
                pitch=math.degrees(math.atan2(delta[2], np.linalg.norm(delta[:2]))),
                yaw=math.degrees(math.atan2(delta[1], delta[0])),
            )
            sample = min(sample, len(self.poses) - 1)
            lo = int(sample)
            hi = min(lo + 1, len(self.poses) - 1)
            blend = sample - lo
            a, b = self.poses[lo], self.poses[hi].copy()
            b[:, 3:] *= np.where(np.sum(a[:, 3:] * b[:, 3:], axis=1) < 0, -1, 1)[:, None]
            pose = a * (1 - blend) + b * blend
            pose[:, 3:] /= np.linalg.norm(pose[:, 3:], axis=1)[:, None]
            state.body_q.assign(pose)
            viewer.begin_frame(sample / self.fps)
            viewer.log_state(state)
            viewer.end_frame()
            return viewer.get_frame().numpy()

        center = offsets.mean(axis=0) + np.array([0, 0, -0.10])
        extent = max(np.ptp(offsets[:, 0]) + 2.3, np.ptp(offsets[:, 1]) + 2.3)
        far = center + np.array([0.30, -0.74, 0.84]) * extent * 1.08
        if args.roster:
            # Fit the perspective projection, including the front plinths and
            # table legs; centering only the world origins clips the near row.
            corners = np.array(list(product((-1.1, 1.1), (-1.1, 1.1), (-0.8, 1.35))))
            bounds = (offsets[:, None, :] + corners[None, :, :]).reshape(-1, 3)
            distance = np.linalg.norm(far - center)
            forward = (center - far) / distance
            right = np.cross(forward, [0, 0, 1])
            right /= np.linalg.norm(right)
            up = np.cross(right, forward)
            tangent = math.tan(math.radians(viewer.camera.fov / 2))
            aspect = args.width / args.height
            for _ in range(16):
                relative = bounds - center
                depth = distance + relative @ forward
                x = relative @ right / depth / tangent / aspect
                y = relative @ up / depth / tangent
                center += right * (x.max() + x.min()) / 2 * tangent * aspect * distance
                center += up * (y.max() + y.min()) / 2 * tangent * distance
                distance *= max(np.abs(x).max() / 0.93, np.abs(y).max() / 0.90)
            far = center - forward * distance
        initial = offsets[0] + np.array([2.6, -3.8, 2.8])
        near_target = offsets[0] + np.array([0.1, 0, 0.4])
        if args.roster:
            near_target = offsets[0] + np.array([0.1, 0, 0.25])
            initial = near_target + np.array([1.15, -1.65, 1.25])
        if args.shots in ("all", "overview"):
            writer = imageio_ffmpeg.write_frames(
                str(args.output / "overview.mp4"),
                (args.width, args.height),
                fps=30,
                codec="libx264",
                quality=8,
                macro_block_size=1,
                pix_fmt_in="rgb24",
                pix_fmt_out="yuv420p",
                output_params=["-movflags", "+faststart"],
            )
            writer.send(None)
            for f in range(round(args.duration * 30)):
                t = f / 30
                u = np.clip((t - 0.75) / (4.5 if args.roster else 6), 0, 1)
                u = u * u * u * (10 + u * (-15 + 6 * u))
                if args.straight_walk:
                    root = self.poses[min(round(t * self.fps), len(self.poses) - 1), 0, :3]
                    focus = root + np.array([0, 0, -0.060])
                    rgb = frame(t * self.fps, focus + np.array([0, 2.4, 0.16]), focus)
                else:
                    sample = min(t + (2 if args.roster else 0), args.duration) * self.fps
                    rgb = frame(sample, initial * (1 - u) + far * u, near_target * (1 - u) + center * u)
                writer.send(rgb)
                if f in [0, 90, 240, round(args.duration * 30) - 1]:
                    Image.fromarray(rgb).save(args.output / f"overview-{f:04d}.png")
                if f % 120 == 0:
                    print(f"Rendered {f}/ {round(args.duration * 30)}", flush=True)
            writer.close()

        def select_worlds(ids):
            viewer.set_visible_worlds(ids)
            # Visibility repacks batch names in this checkout. Recreate GL buffers
            # so retained names cannot reuse another mesh or instance capacity.
            viewer._rebuild_shape_batches_for_opacity_groups()

        # Close views make contact outcomes visible before montage editing.
        viewer.renderer.shadow_extents = 3.0
        for world, w in enumerate(self.worlds):
            if args.shots == "extras" and w["kind"] != "insert":
                continue
            if args.shots not in ("all", "extras") and w["kind"] != args.shots:
                continue
            select_worlds([world])
            viewer.world_offsets.zero_()
            station_target = np.array([0, 0, -0.08])
            station_rgb = frame(
                min(5, args.duration) * self.fps, station_target + np.array([2.2, -2.9, 2.0]), station_target
            )
            Image.fromarray(station_rgb).save(args.output / f"station-{w['id']}.png")
            target = np.array([0.1, 0.03, 0.40 if w["kind"] in ("g1", "hand", "shadow") else 0.22])
            eye = target + np.array([1.15, -1.65, 1.25])
            if w["kind"] == "g1":
                target = np.array([0, 0, 0.65])
                eye = target + np.array([1.45, -2.20, 1.05])
            if w["kind"] in ("kit", "serve", "puzzle", "interlock", "toy", "pile"):
                target = np.array([0.14, 0.02, 0.18])
                eye = target + np.array([0.68, -0.96, 0.88])
            if w["kind"] == "gear":
                target = np.array([0.16, 0.20, 0.10])
                eye = target + np.array([0.42, -0.58, 0.66])
            if w["kind"] == "drawer":
                target = np.array([0.07, -0.08, 0.21])
                eye = target + np.array([0.95, -1.35, 1.1])
            if w["kind"] == "shadow":
                eye = target + np.array([0.65, -0.90, 0.72])
            if w["kind"] == "hand":
                palm = self.poses[100, w["body_start"] + w["palm_body"], :3]
                target = palm + np.array([0.035, 0, 0.025])
                eye = target + np.array([0.34, -0.46, 0.38])
            for t in [0, 3, 5, 8, min(11, args.duration)]:
                rgb = frame(t * self.fps, eye, target)
                Image.fromarray(rgb).save(args.output / f"template-{w['id']}-{t:02.0f}.png")
            if w["kind"] == "drawer":
                focus = np.array([0.22, -0.53, 0.06])
                rgb = frame(0, focus + np.array([0.28, -0.34, 0.43]), focus)
                Image.fromarray(rgb).save(args.output / "cutlery-detail.png")
            if args.closeups and (not args.clip_cases or w["id"] in args.clip_cases.split(",")):
                movie = imageio_ffmpeg.write_frames(
                    str(args.output / f"task-{w['id']}.mp4"),
                    (args.width, args.height),
                    fps=30,
                    codec="libx264",
                    quality=8,
                    macro_block_size=1,
                    output_params=["-movflags", "+faststart"],
                )
                movie.send(None)
                for f in range(round(args.duration * 30)):
                    movie.send(frame(f / 30 * self.fps, eye, target))
                movie.close()
                if w["kind"] == "insert":
                    detail = imageio_ffmpeg.write_frames(
                        str(args.output / "insertion-detail.mp4"),
                        (args.width, args.height),
                        fps=30,
                        codec="libx264",
                        quality=8,
                        macro_block_size=1,
                        output_params=["-movflags", "+faststart"],
                    )
                    detail.send(None)
                    focus = np.array([0.16, 0.23, 0.22])
                    close_eye = focus + np.array([0.42, -0.58, 0.55])
                    for f in range(225):
                        rgb = frame((5.5 + f / 30) * self.fps, close_eye, focus)
                        detail.send(rgb)
                        if f in (0, 90, 179, 224):
                            Image.fromarray(rgb).save(args.output / f"insertion-detail-{f:03d}.png")
                    detail.close()
                if w["kind"] in ("hand", "shadow"):
                    name = "allegro" if w["kind"] == "hand" else "shadow"
                    movie = imageio_ffmpeg.write_frames(
                        str(args.output / f"{name}-detail.mp4"),
                        (args.width, args.height),
                        fps=30,
                        codec="libx264",
                        quality=8,
                        macro_block_size=1,
                    )
                    movie.send(None)
                    for f in range(round(args.duration * 30)):
                        sample = f / 30 * self.fps
                        palm = self.poses[min(round(sample), len(self.poses) - 1), w["body_start"] + w["palm_body"], :3]
                        if w["kind"] == "shadow":
                            focus = palm + np.array([0, 0.075, 0.03])
                            eye_offset = np.array([0.26, 0.24, 0.25])
                        else:
                            focus = palm + np.array([0.04, 0, 0.03])
                            eye_offset = np.array([0.32, -0.46, 0.38])
                        movie.send(frame(sample, focus + eye_offset, focus))
                    movie.close()
        if args.closeups and self.model.world_count >= 16 and args.shots in ("all", "extras"):
            # Recompose the Franka subset from this same recorded heterogeneous batch.
            ids = [i for i, w in enumerate(self.worlds) if w.get("robot") == "franka"]
            select_worlds(ids)
            layout = np.zeros((self.model.world_count, 3), dtype=np.float32)
            cols = 2 if len(ids) <= 4 else 4
            for j, i in enumerate(ids):
                layout[i] = (
                    (j % cols - (cols - 1) / 2) * 2.6,
                    (j // cols - (math.ceil(len(ids) / cols) - 1) / 2) * 2.6,
                    0,
                )
            viewer.world_offsets.assign(layout)
            viewer.renderer.shadow_extents = 12
            movie = imageio_ffmpeg.write_frames(
                str(args.output / "franka-at-scale.mp4"),
                (args.width, args.height),
                fps=30,
                codec="libx264",
                quality=8,
                macro_block_size=1,
            )
            movie.send(None)
            for f in range(round(args.duration * 30)):
                u = f / (args.duration * 30)
                focus = np.array([0, 0, 0.20])
                eye = focus + np.array([3.0 - u, -6.5, 7.2])
                movie.send(frame(f / 30 * self.fps, eye, focus))
            movie.close()
        viewer.close()


def main():
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import pyglet

    pyglet.options["headless"] = True
    p = argparse.ArgumentParser()
    p.add_argument("--assets", default="assets.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--templates", default=",".join(TEMPLATES))
    p.add_argument("--cases", default="", help="Curated kind:variant list, overriding templates and copies")
    p.add_argument("--roster", type=Path, help="JSON station roster, overriding cases")
    p.add_argument("--clip-cases", default="", help="Comma-separated station IDs to render as close-up videos")
    p.add_argument("--copies", type=int, default=1)
    p.add_argument("--duration", type=float, default=14)
    p.add_argument("--substeps", type=int, default=8)
    p.add_argument("--iterations", type=int, default=16)
    p.add_argument("--width", type=int, default=1600)
    p.add_argument("--height", type=int, default=900)
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--replay", action="store_true")
    p.add_argument("--closeups", action="store_true")
    p.add_argument("--straight-walk", action="store_true")
    p.add_argument("--shots", choices=["all", "overview", "extras", "drawer", "shadow", "g1"], default="all")
    p.add_argument("--use-graph", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    wp.init()
    wp.set_device("cuda:0")
    example = Example(args)
    if args.replay:
        example.poses = np.load(args.output / "trace.npz")["poses"]
        example.wall = 0.0
        example.render()
        return
    example.simulate()
    report = {
        "solver": "SolverFeatherPGS",
        "mode": "matrix_free",
        "contact_response": "immediate",
        "worlds": example.model.world_count,
        "simultaneous_heterogeneous_batch": len({w["kind"] for w in example.worlds}) > 1,
        "device": wp.get_device().name,
        "duration": args.duration,
        "simulation_wall_s": example.wall,
        "substeps": args.substeps,
        "iterations": args.iterations,
        "metrics": example.metrics,
        "constraint_row_watermarks": example.watermarks,
        "contact_reduction": example.pipeline.body_pair_reduction_stats(),
        "trace_sha256": hashlib.sha256((args.output / "trace.npz").read_bytes()).hexdigest(),
    }
    quality = audit(args.output)
    outcome_flags = [value for metric in example.metrics for value in metric.values() if isinstance(value, bool)]
    pours_pass = all(
        m.get("terminal_pocket_arrivals", m.get("hardware_terminal_pockets", 18)) >= 12 for m in example.metrics
    )
    quality["summary_checks_pass"] = all(outcome_flags) and pours_pass
    quality["constraint_capacity_pass"] = not any(
        value for key, value in example.watermarks.items() if "overflow_world_steps" in key
    )
    quality["contact_reduction_pass"] = not any(
        report["contact_reduction"][key]
        for key in ("invariant_violations", "probe_failures", "input_overflow_frames", "fallback_frames")
    )
    quality["pass"] = all(
        quality[key] for key in ("pass", "summary_checks_pass", "constraint_capacity_pass", "contact_reduction_pass")
    )
    (args.output / "strict-audit.json").write_text(json.dumps(quality, indent=2) + "\n")
    report["quality_gate_passed"] = quality["pass"]
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    assert quality["pass"], "Task quality gates failed; inspect strict-audit.json and report.json"
    if not args.no_render:
        example.render()


if __name__ == "__main__":
    main()
