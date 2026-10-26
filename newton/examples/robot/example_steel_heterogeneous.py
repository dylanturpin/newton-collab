# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Many scene types in one SolverSteel model, each world resetting to its own scene type.

Command: python -m newton.examples steel_heterogeneous

The worlds of one batch are drawn from seven different scenes instead of copies
of one scene: four walking robots under their pretrained locomotion policies
(ANYmal C, the Unitree G1 with 23 actuated joints, the Unitree Go2 and the G1
with 29 actuated joints, the ONNX policies of the robot_policy example run
through warp-nn), a fixed-base seven-link arm sweeping over three boxes, a
passive three-link pendulum, and a pyramid of boxes dropped onto the ground.
World i holds scene type i modulo the number of scene types, so the degrees of
freedom, bodies and contacts differ from world to world and, with the default
square world count, each column of the viewer grid is one scene type.

Every world runs episodes. A walking robot draws a command (forward speed,
sideways speed, yaw rate) for the episode and follows it under its policy, one
batched inference per robot type per control step; the arm follows scripted
sinusoidal targets with phases drawn per episode; the pendulum and the pile
are passive. When a world's episode ends it is reset to its own scene type's
start state while the other worlds keep running. Resets never change a world's
scene type, so the model, the solver's per-world buffers and the captured CUDA
graph stay fixed: the episode bookkeeping, the masked state restore, the
solver's masked reset, the policies and the physics substeps are all recorded
in one graph that is replayed every control step.

On CUDA the solver runs in its matrix-free mode; on CPU it runs in split mode.
"""

import argparse
import os
from dataclasses import dataclass

import numpy as np
import warp as wp
import yaml
from warp_nn.runtime import OnnxRuntime

import newton
import newton.examples
import newton.utils
from newton import JointTargetMode
from newton.examples.robot.onnx_policy_utils import validate_policy_io_shapes
from newton.solvers import SolverSteel

CONTROL_HZ = 50  # the policies were trained at 50 Hz with physics at 200 Hz
WORLD_SPACING = 4.0  # viewer spacing between worlds [m]
BOX_HALF = 0.1  # half extent of every loose box [m]
ARTICULATION_ROWS = 256  # dense (articulation) constraint rows per world
FREE_BODY_ROWS = 512  # matrix-free (free body) constraint rows per world
ARM_POSE = [0.6, 0.0, 0.7, 0.0, 0.6, 0.0, 0.4]  # the arm arcs forward over its boxes [rad]
FORWARD_MIN = 0.2  # per-episode forward speed command of a walker [m/s]
FORWARD_MAX = 0.6
SIDEWAYS_MAX = 0.3  # per-episode sideways speed command magnitude [m/s]
TURN_MAX = 0.5  # per-episode yaw-rate command magnitude [rad/s]
RANDOM_HOLD_FRAMES = 10  # random joint targets are redrawn every this many control steps
LANDING_FRAMES = 50  # a walker is judged upright only this long after its episode started [control steps]
UPRIGHT_COS = 0.85  # a walker counts as upright when its up axis is within about 32 degrees of vertical
HEIGHT_MIN = 0.5  # and its base is within this band of its walking height (a lying robot is far below)
HEIGHT_MAX = 1.5
ARROW_HEIGHT = 0.6  # the command arrows float this far above each walker's base [m]


@dataclass(frozen=True)
class RobotSpec:
    """Where a walking robot's USD, policy and policy configuration live in the asset registry."""

    asset_dir: str
    usd_path: str
    policy_path: str
    config_path: str
    base_height: float  # base height while walking [m], for the upright test


ROBOTS = {
    "anymal": RobotSpec(
        "anybotics_anymal_c", "usd/anymal_c.usda", "rl_policies/mjw_anymal.onnx", "rl_policies/anymal.yaml", 0.60
    ),
    "g1_23dof": RobotSpec(
        "unitree_g1", "usd/g1_minimal.usd", "rl_policies/mjw_g1_23DOF.onnx", "rl_policies/g1_23dof.yaml", 0.76
    ),
    "go2": RobotSpec("unitree_go2", "usd/go2.usda", "rl_policies/mjw_go2.onnx", "rl_policies/go2.yaml", 0.38),
    "g1_29dof": RobotSpec(
        "unitree_g1", "usd/g1_isaac.usd", "rl_policies/mjw_g1_29DOF.onnx", "rl_policies/g1_29dof.yaml", 0.79
    ),
}


@wp.kernel(enable_backward=False)
def advance_episodes(
    base_length: wp.array[wp.int32],
    jitter_length: wp.array[wp.int32],
    seed: wp.int32,
    joint_q: wp.array[wp.float32],
    joint_coord_start: wp.array[wp.int32],
    base_height: wp.array[wp.float32],
    landing_frames: wp.int32,
    # in/out
    episode_frame: wp.array[wp.int32],
    episode_length: wp.array[wp.int32],
    episode_count: wp.array[wp.int32],
    reset_count: wp.array[wp.int32],
    fall_count: wp.array[wp.int32],
    # output
    reset_mask: wp.array[wp.bool],
):
    """Advance every world's episode clock by one control step and flag the worlds whose episode ended.

    A walking robot's episode also ends early when it has fallen: once it has landed, its base must stay within
    half to one and a half of its walking height and its up axis near vertical, as in the training environments
    of its policy.
    """
    world = wp.tid()
    frame = episode_frame[world] + 1
    ended = frame >= episode_length[world]
    fell = False
    if not ended and base_height[world] > 0.0 and frame >= landing_frames:
        qs = joint_coord_start[world]  # the walker's free joint comes first in its world
        rot = wp.quat(joint_q[qs + 3], joint_q[qs + 4], joint_q[qs + 5], joint_q[qs + 6])
        up = wp.quat_rotate(rot, wp.vec3(0.0, 0.0, 1.0))
        height = joint_q[qs + 2] / base_height[world]
        fell = height < HEIGHT_MIN or height > HEIGHT_MAX or up[2] < UPRIGHT_COS
    if ended or fell:
        reset_mask[world] = True
        episode_frame[world] = 0
        episode_count[world] = episode_count[world] + 1
        reset_count[world] = reset_count[world] + 1
        if fell:
            fall_count[world] = fall_count[world] + 1
        # the next episode length carries a per-episode jitter so the resets of one scene type spread out
        state = wp.rand_init(seed, world * 65536 + episode_count[world])
        episode_length[world] = base_length[world] + wp.int32(wp.randf(state) * wp.float32(jitter_length[world]))
    else:
        reset_mask[world] = False
        episode_frame[world] = frame


@wp.kernel(enable_backward=False)
def restore_joint_state(
    reset_mask: wp.array[wp.bool],
    joint_world: wp.array[wp.int32],
    joint_q_start: wp.array[wp.int32],
    joint_qd_start: wp.array[wp.int32],
    initial_q: wp.array[wp.float32],
    initial_qd: wp.array[wp.float32],
    initial_target_q: wp.array[wp.float32],
    # output
    joint_q: wp.array[wp.float32],
    joint_qd: wp.array[wp.float32],
    joint_target_q: wp.array[wp.float32],
):
    """Copy the start state of every joint that belongs to a reset world."""
    joint = wp.tid()
    world = joint_world[joint]
    if world < 0 or not reset_mask[world]:
        return
    for coord in range(joint_q_start[joint], joint_q_start[joint + 1]):
        joint_q[coord] = initial_q[coord]
        joint_target_q[coord] = initial_target_q[coord]
    for dof in range(joint_qd_start[joint], joint_qd_start[joint + 1]):
        joint_qd[dof] = initial_qd[dof]


@wp.kernel(enable_backward=False)
def restore_body_state(
    reset_mask: wp.array[wp.bool],
    body_world: wp.array[wp.int32],
    initial_body_q: wp.array[wp.transform],
    initial_body_qd: wp.array[wp.spatial_vector],
    # output
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
):
    """Copy the start pose and velocity of every body that belongs to a reset world."""
    body = wp.tid()
    world = body_world[body]
    if world < 0 or not reset_mask[world]:
        return
    body_q[body] = initial_body_q[body]
    body_qd[body] = initial_body_qd[body]


@wp.kernel(enable_backward=False)
def draw_commands(
    reset_mask: wp.array[wp.bool],
    world_is_walker: wp.array[wp.bool],
    episode_count: wp.array[wp.int32],
    seed: wp.int32,
    # output
    command: wp.array2d[wp.float32],
):
    """Draw the episode command (forward, sideways, yaw rate) of every walking robot whose world was reset."""
    world = wp.tid()
    if not reset_mask[world]:
        return
    if not world_is_walker[world]:
        command[world, 0] = 0.0
        command[world, 1] = 0.0
        command[world, 2] = 0.0
        return
    state = wp.rand_init(seed + 104729 * episode_count[world], world)
    command[world, 0] = FORWARD_MIN + (FORWARD_MAX - FORWARD_MIN) * wp.randf(state)
    command[world, 1] = SIDEWAYS_MAX * (2.0 * wp.randf(state) - 1.0)
    command[world, 2] = TURN_MAX * (2.0 * wp.randf(state) - 1.0)


@wp.kernel(enable_backward=False)
def reset_prev_actions(
    reset_mask: wp.array[wp.bool],
    row_world: wp.array[wp.int32],
    # output
    prev_action: wp.array2d[wp.float32],
):
    """Forget the previous policy action of every reset world of a robot type."""
    row, k = wp.tid()
    if reset_mask[row_world[row]]:
        prev_action[row, k] = 0.0


@wp.kernel(enable_backward=False)
def draw_drive_phases(
    reset_mask: wp.array[wp.bool],
    drive_world: wp.array[wp.int32],
    episode_count: wp.array[wp.int32],
    seed: wp.int32,
    # output
    drive_phase: wp.array[wp.float32],
    drive_scale: wp.array[wp.float32],
):
    """Draw a new phase and amplitude scale for every sinusoidally driven joint of a reset world."""
    drive = wp.tid()
    world = drive_world[drive]
    if not reset_mask[world]:
        return
    state = wp.rand_init(seed + episode_count[world], drive)
    drive_phase[drive] = 2.0 * wp.pi * wp.randf(state)
    drive_scale[drive] = 0.5 + 0.5 * wp.randf(state)


@wp.kernel(enable_backward=False)
def apply_sine_drives(
    drive_coord: wp.array[wp.int32],
    drive_world: wp.array[wp.int32],
    drive_rest: wp.array[wp.float32],
    drive_amplitude: wp.array[wp.float32],
    drive_frequency: wp.array[wp.float32],
    drive_phase: wp.array[wp.float32],
    drive_scale: wp.array[wp.float32],
    episode_frame: wp.array[wp.int32],
    frame_dt: wp.float32,
    # output
    joint_target_q: wp.array[wp.float32],
):
    """Write the scripted sinusoidal target of every driven joint for its world's episode time."""
    drive = wp.tid()
    time = wp.float32(episode_frame[drive_world[drive]]) * frame_dt
    angle = 2.0 * wp.pi * drive_frequency[drive] * time + drive_phase[drive]
    joint_target_q[drive_coord[drive]] = drive_rest[drive] + drive_scale[drive] * drive_amplitude[drive] * wp.sin(angle)


@wp.kernel(enable_backward=False)
def apply_random_targets(
    drive_coord: wp.array[wp.int32],
    drive_world: wp.array[wp.int32],
    drive_rest: wp.array[wp.float32],
    drive_amplitude: wp.array[wp.float32],
    episode_frame: wp.array[wp.int32],
    episode_count: wp.array[wp.int32],
    seed: wp.int32,
    hold_frames: wp.int32,
    # output
    joint_target_q: wp.array[wp.float32],
):
    """Redraw a uniformly random target around the rest pose every hold period of the world's episode."""
    drive = wp.tid()
    world = drive_world[drive]
    frame = episode_frame[world]
    if frame % hold_frames != 0:
        return
    state = wp.rand_init(seed + 7919 * episode_count[world] + frame / hold_frames, drive)
    joint_target_q[drive_coord[drive]] = drive_rest[drive] + drive_amplitude[drive] * (2.0 * wp.randf(state) - 1.0)


@wp.kernel(enable_backward=False)
def compute_observations(
    joint_q: wp.array[wp.float32],
    joint_qd: wp.array[wp.float32],
    joint_coord_start: wp.array[wp.int32],
    joint_dof_start: wp.array[wp.int32],
    row_world: wp.array[wp.int32],
    joint_pos_initial: wp.array[wp.float32],
    gravity_dir: wp.vec3,
    command: wp.array2d[wp.float32],
    prev_action: wp.array2d[wp.float32],
    num_dofs: wp.int32,
    # output
    obs: wp.array2d[wp.float32],
):
    """One observation per world of a robot type, in the layout the robot_policy policies expect.

    Base linear and angular velocity in the base frame, gravity in the base frame, the world's command (forward,
    sideways, yaw rate), joint positions relative to the initial pose, joint velocities and the previous action.
    """
    row = wp.tid()
    world = row_world[row]
    qs = joint_coord_start[world]  # the robot's free joint comes first in its world: 7 coordinates, 6 DOFs
    ds = joint_dof_start[world]
    rot = wp.quat(joint_q[qs + 3], joint_q[qs + 4], joint_q[qs + 5], joint_q[qs + 6])
    lin_b = wp.quat_rotate_inv(rot, wp.vec3(joint_qd[ds + 0], joint_qd[ds + 1], joint_qd[ds + 2]))
    ang_b = wp.quat_rotate_inv(rot, wp.vec3(joint_qd[ds + 3], joint_qd[ds + 4], joint_qd[ds + 5]))
    grav_b = wp.quat_rotate_inv(rot, gravity_dir)
    for i in range(3):
        obs[row, i] = lin_b[i]
        obs[row, 3 + i] = ang_b[i]
        obs[row, 6 + i] = grav_b[i]
        obs[row, 9 + i] = command[world, i]
    for k in range(num_dofs):
        obs[row, 12 + k] = joint_q[qs + 7 + k] - joint_pos_initial[k]
        obs[row, 12 + num_dofs + k] = joint_qd[ds + 6 + k]
        obs[row, 12 + 2 * num_dofs + k] = prev_action[row, k]


@wp.kernel(enable_backward=False)
def apply_actions(
    action: wp.array2d[wp.float32],
    joint_coord_start: wp.array[wp.int32],
    row_world: wp.array[wp.int32],
    joint_pos_initial: wp.array[wp.float32],
    action_scale: wp.float32,
    # outputs
    joint_target_q: wp.array[wp.float32],
    prev_action: wp.array2d[wp.float32],
):
    """Turn a policy action into joint position targets and keep it for the next observation."""
    row, k = wp.tid()
    a = action[row, k]
    joint_target_q[joint_coord_start[row_world[row]] + 7 + k] = joint_pos_initial[k] + action_scale * a
    prev_action[row, k] = a


@wp.kernel(enable_backward=False)
def command_arrows(
    joint_q: wp.array[wp.float32],
    joint_qd: wp.array[wp.float32],
    joint_coord_start: wp.array[wp.int32],
    joint_dof_start: wp.array[wp.int32],
    walker_world: wp.array[wp.int32],
    world_offsets: wp.array[wp.vec3],
    command: wp.array2d[wp.float32],
    # outputs
    starts: wp.array[wp.vec3],
    commanded_ends: wp.array[wp.vec3],
    achieved_ends: wp.array[wp.vec3],
):
    """Two arrows above each walker, one metre per m/s: the commanded planar velocity and the one the base has."""
    i = wp.tid()
    world = walker_world[i]
    qs = joint_coord_start[world]
    ds = joint_dof_start[world]
    pos = wp.vec3(joint_q[qs + 0], joint_q[qs + 1], joint_q[qs + 2] + ARROW_HEIGHT) + world_offsets[world]
    rot = wp.quat(joint_q[qs + 3], joint_q[qs + 4], joint_q[qs + 5], joint_q[qs + 6])
    commanded = wp.quat_rotate(rot, wp.vec3(command[world, 0], command[world, 1], 0.0))
    starts[i] = pos
    commanded_ends[i] = pos + wp.vec3(commanded[0], commanded[1], 0.0)
    achieved_ends[i] = pos + wp.vec3(joint_qd[ds + 0], joint_qd[ds + 1], 0.0)


def _add_box(builder: newton.ModelBuilder, pos, color) -> int:
    body = builder.add_body(xform=wp.transform(wp.vec3(*pos), wp.quat_identity()))
    builder.add_shape_box(body, hx=BOX_HALF, hy=BOX_HALF, hz=BOX_HALF, color=color)
    return body


def _add_chain(builder: newton.ModelBuilder, link_count: int, link_length: float, radius: float, base, hang: bool):
    """Add a fixed-base chain of capsule links on revolute joints, standing up from or hanging below its base."""
    direction = -1.0 if hang else 1.0
    joints = []
    parent = -1
    for i in range(link_count):
        link = builder.add_link()
        builder.add_shape_capsule(
            link,
            xform=wp.transform(wp.vec3(0.0, 0.0, direction * 0.5 * link_length), wp.quat_identity()),
            radius=radius,
            half_height=0.5 * link_length - radius,
        )
        if parent == -1:
            parent_xform = wp.transform(wp.vec3(*base), wp.quat_identity())
        else:
            parent_xform = wp.transform(wp.vec3(0.0, 0.0, direction * link_length), wp.quat_identity())
        # alternate pitch and roll axes so the chain can bend in both directions
        axis = wp.vec3(0.0, 1.0, 0.0) if i % 2 == 0 else wp.vec3(1.0, 0.0, 0.0)
        joints.append(
            builder.add_joint_revolute(
                parent=parent,
                child=link,
                axis=axis,
                parent_xform=parent_xform,
                child_xform=wp.transform_identity(),
                limit_lower=-2.5,
                limit_upper=2.5,
            )
        )
        parent = link
    builder.add_articulation(joints)


def _build_arm() -> newton.ModelBuilder:
    """A fixed-base seven-link arm holding a bent pose over three boxes (7 + 18 DOFs)."""
    builder = newton.ModelBuilder()
    builder.default_joint_cfg.armature = 0.05
    builder.default_joint_cfg.target_ke = 1000.0
    builder.default_joint_cfg.target_kd = 40.0
    builder.default_shape_cfg.mu = 1.0
    _add_chain(builder, link_count=7, link_length=0.22, radius=0.04, base=(0.0, 0.0, 0.12), hang=False)
    builder.joint_q[-7:] = ARM_POSE
    builder.joint_target_q[-7:] = ARM_POSE
    for i in range(3):
        _add_box(builder, (0.45 + 0.35 * i, 0.0, BOX_HALF), (0.3, 0.8, 0.3))
    return builder


def _build_pendulum() -> newton.ModelBuilder:
    """A passive three-link pendulum released from the side, hanging from a fixed base (3 DOFs)."""
    builder = newton.ModelBuilder()
    builder.default_joint_cfg.armature = 0.01
    builder.default_shape_cfg.mu = 1.0
    _add_chain(builder, link_count=3, link_length=0.4, radius=0.04, base=(0.0, 0.0, 1.8), hang=True)
    builder.joint_q[0] = 1.4
    return builder


def _build_pile() -> newton.ModelBuilder:
    """A pyramid of eight loose boxes, four by three by one, dropped from just above the ground (48 DOFs)."""
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.mu = 1.0
    pitch = 2.0 * BOX_HALF + 0.02
    for layer, count in enumerate((4, 3, 1)):
        x0 = -0.5 * (count - 1) * pitch
        for i in range(count):
            z = 0.3 + BOX_HALF + layer * (2.0 * BOX_HALF + 0.02)
            _add_box(builder, (x0 + i * pitch, 0.0, z), (0.8, 0.3, 0.3))
    return builder


@dataclass
class RobotType:
    """A walking robot of the batch: its builder, its policy file and the policy's configuration."""

    name: str
    spec: RobotSpec
    config: dict
    builder: newton.ModelBuilder
    policy_file: str

    @property
    def num_dofs(self) -> int:
        return int(self.config["num_dofs"])

    @property
    def obs_width(self) -> int:
        return 12 + 3 * self.num_dofs


def load_robot(name: str) -> RobotType:
    """Build one walking robot the way the robot_policy example does, from the registry's USD and policy files."""
    spec = ROBOTS[name]
    asset_dir = str(newton.utils.download_asset(spec.asset_dir))
    with open(f"{asset_dir}/{spec.config_path}", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    num_dofs = int(config["num_dofs"])

    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(armature=0.1, limit_ke=1.0e2, limit_kd=1.0)
    builder.default_shape_cfg.ke = 5.0e4
    builder.default_shape_cfg.kd = 5.0e2
    builder.default_shape_cfg.kf = 1.0e3
    builder.default_shape_cfg.mu = 0.75
    builder.rigid_gap = 0.0
    builder.add_usd(
        newton.examples.get_asset(f"{asset_dir}/{spec.usd_path}"),
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.8)),
        collapse_fixed_joints=False,
        enable_self_collisions=False,
        joint_ordering="dfs",
        hide_collision_shapes=True,
    )
    builder.approximate_meshes("convex_hull")
    # some of the USD files carry a physics scene with zero gravity (g1_minimal.usd does); every world of the batch
    # gets standard gravity
    builder.gravity = (0.0, 0.0, -9.81)
    if builder.articulation_count != 1 or builder.joint_dof_count != 6 + num_dofs:
        raise ValueError(
            f"{name}: expected one articulation with a free joint and {num_dofs} actuated DOFs, got "
            f"{builder.articulation_count} articulations with {builder.joint_dof_count} DOFs"
        )

    # the free joint first (7 coordinates, 6 DOFs), then the actuated joints in the policy's order; the robots
    # start facing +y and drop onto the ground from 0.76 m like the robot_policy example
    builder.joint_q[:3] = [0.0, 0.0, 0.76]
    builder.joint_q[3:7] = [0.0, 0.0, 0.7071, 0.7071]
    builder.joint_q[7:] = config["mjw_joint_pos"]
    builder.joint_target_q[7:] = config["mjw_joint_pos"]
    for i in range(num_dofs):
        builder.joint_target_ke[6 + i] = config["mjw_joint_stiffness"][i]
        builder.joint_target_kd[6 + i] = config["mjw_joint_damping"][i]
        builder.joint_armature[6 + i] = config["mjw_joint_armature"][i]
        builder.joint_target_mode[6 + i] = int(JointTargetMode.POSITION)
    return RobotType(name, spec, config, builder, f"{asset_dir}/{spec.policy_path}")


class PolicyBatch:
    """The worlds of one robot type, their observation and action buffers and the type's batched policy."""

    def __init__(self, robot: RobotType, worlds: list[int], device):
        self.robot = robot
        self.worlds = worlds
        self.row_world = wp.array(worlds, dtype=wp.int32, device=device)
        self.joint_pos_initial = wp.array(np.asarray(robot.config["mjw_joint_pos"], dtype=np.float32), device=device)
        self.action_scale = float(robot.config["action_scale"])
        self.obs = wp.zeros((len(worlds), robot.obs_width), dtype=wp.float32, device=device)
        self.prev_action = wp.zeros((len(worlds), robot.num_dofs), dtype=wp.float32, device=device)
        # one warp-nn runtime per robot type: inference is one batched call per type, and the runtime
        # pre-allocates its buffers for the batch so the call can be graph captured
        self.policy = OnnxRuntime(robot.policy_file, device=device)
        self.input_name = self.policy.inputs[0].name
        self.output_name = self.policy.outputs[0].name
        validate_policy_io_shapes(
            robot.policy_file,
            self.input_name,
            self.output_name,
            obs_width=robot.obs_width,
            action_width=robot.num_dofs,
            context="steel_heterogeneous",
        )
        self.policy.prepare({self.input_name: self.obs})

    def observe(self, model: newton.Model, state: newton.State, command: wp.array):
        wp.launch(
            compute_observations,
            dim=len(self.worlds),
            inputs=[
                state.joint_q,
                state.joint_qd,
                model.joint_coord_world_start,
                model.joint_dof_world_start,
                self.row_world,
                self.joint_pos_initial,
                wp.vec3(0.0, 0.0, -1.0),
                command,
                self.prev_action,
                self.robot.num_dofs,
            ],
            outputs=[self.obs],
            device=model.device,
        )

    def act(self, model: newton.Model, control: newton.Control):
        action = self.policy({self.input_name: self.obs})[self.output_name]
        wp.launch(
            apply_actions,
            dim=(len(self.worlds), self.robot.num_dofs),
            inputs=[action, model.joint_coord_world_start, self.row_world, self.joint_pos_initial, self.action_scale],
            outputs=[control.joint_target_q, self.prev_action],
            device=model.device,
        )

    def run(self, model: newton.Model, state: newton.State, control: newton.Control, command: wp.array):
        """Observations for every world of the type, one policy evaluation, joint targets."""
        self.observe(model, state, command)
        self.act(model, control)


@dataclass
class SceneType:
    """How one scene type is built and how its episodes run."""

    build: object  # a callable returning the scene's ModelBuilder, or None for a walking robot loaded by name
    episode_seconds: float
    jitter_seconds: float
    drive: str = "none"  # "policy", "sine", "random" or "none"
    amplitude: float = 0.0  # sine amplitude or random target half range around the rest pose [rad]
    frequency: float = 0.0  # sine frequency [Hz]

    @property
    def is_walker(self) -> bool:
        return self.build is None


# the default order alternates quadrupeds and humanoids, then the three scenes without a policy
SCENE_TYPES = {
    "anymal": SceneType(None, 3.0, 1.0, "policy", 0.3),
    "g1_23dof": SceneType(None, 3.0, 1.0, "policy", 0.3),
    "go2": SceneType(None, 3.0, 1.0, "policy", 0.3),
    "g1_29dof": SceneType(None, 3.0, 1.0, "policy", 0.3),
    "arm": SceneType(_build_arm, 4.0, 1.0, "sine", 0.6, 0.3),
    "pendulum": SceneType(_build_pendulum, 4.0, 1.0),
    "pile": SceneType(_build_pile, 4.0, 1.0),
}
WALKERS = [name for name, scene_type in SCENE_TYPES.items() if scene_type.is_walker]


def _articulation_dofs(builder: newton.ModelBuilder) -> list[int]:
    """DOF count of each articulation in a builder, in articulation order."""
    starts = [*builder.articulation_start, builder.joint_count]
    qd_start = [*builder.joint_qd_start, builder.joint_dof_count]
    return [qd_start[starts[a + 1]] - qd_start[starts[a]] for a in range(len(starts) - 1)]


class Example:
    def __init__(self, viewer, args):
        self.fps = CONTROL_HZ
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = args.substeps
        if args.substeps < 1 or args.iterations < 1:
            raise ValueError("Substeps and iterations must be positive")
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.frame = 0
        self.viewer = viewer
        self.args = args
        self.test_mode = args.test
        self.seed = args.seed

        types = list(dict.fromkeys(args.scene_types))
        self.world_count = args.world_count if args.world_count is not None else len(types) ** 2
        if self.world_count < 1:
            raise ValueError("The world count must be positive")
        self.world_scene = [types[i % len(types)] for i in range(self.world_count)]
        self.scene_types = [name for name in types if name in self.world_scene]
        # --max-worlds renders the first worlds only; the command arrows follow the rendered subset
        self.visible_worlds = self.world_count if args.max_worlds is None else min(args.max_worlds, self.world_count)
        # a walker runs on random joint targets instead of its policy when asked to
        self.drive_mode = {
            name: "random" if name in args.random_actions and SCENE_TYPES[name].is_walker else SCENE_TYPES[name].drive
            for name in self.scene_types
        }

        # one builder per scene type, added as a separate world wherever the round robin places it
        self.robots = {name: load_robot(name) for name in self.scene_types if SCENE_TYPES[name].is_walker}
        self.builders = {
            name: self.robots[name].builder if name in self.robots else SCENE_TYPES[name].build()
            for name in self.scene_types
        }
        scene = newton.ModelBuilder()
        scene.add_ground_plane()
        for name in self.world_scene:
            scene.add_world(self.builders[name])
        self.model = scene.finalize()
        self.device = self.model.device

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.initial = self.model.state()
        self.initial.assign(self.state_0)

        # the pipeline publishes the contact capacity on the model, so it is built before the solver
        self.pipeline = newton.CollisionPipeline(self.model, deterministic=args.deterministic)
        self.contacts = self.pipeline.contacts()
        # matrix_free is the CUDA path; split is the only mode on CPU
        self.solve_mode = "matrix_free" if self.device.is_cuda else "split"
        self.solver = SolverSteel(
            self.model,
            solve_mode=self.solve_mode,
            iterations=args.iterations,
            enable_joint_limits=True,
            articulation_constraint_max=ARTICULATION_ROWS,
            free_body_constraint_max=FREE_BODY_ROWS,
            row_watermark=True,
            use_parallel_streams=False,
            deterministic=args.deterministic,
        )

        self.setup_episodes()
        self.setup_drives()
        self.setup_policies()
        self.setup_test_stats()
        self.print_summary()

        self.viewer.set_model(self.model)
        if args.max_worlds is not None:
            self.viewer.set_visible_worlds(range(args.max_worlds))
        self.viewer.set_world_offsets((WORLD_SPACING, WORLD_SPACING, 0.0))
        # the rendered worlds sit on a grid centred on the origin; a low three-quarter view from the front-left
        # corner with a narrow lens, so the robots read as robots (with --max-worlds 14 the near ones fill the frame)
        half = max(1.5, 0.5 * WORLD_SPACING * (int(np.ceil(np.sqrt(self.visible_worlds))) - 1))
        pos = wp.vec3(-1.9 * half, -2.9 * half, 0.9 * half)
        target = wp.vec3(0.08 * half, -0.08 * half, 0.3)
        dx, dy, dz = target[0] - pos[0], target[1] - pos[1], target[2] - pos[2]
        pitch = float(np.degrees(np.arctan2(dz, np.hypot(dx, dy))))
        yaw = float(np.degrees(np.arctan2(dy, dx)))
        if hasattr(self.viewer, "camera") and hasattr(self.viewer.camera, "fov"):
            self.viewer.camera.fov = 34.0
        self.viewer.set_camera(pos=pos, pitch=pitch, yaw=yaw)

        self.capture()

    def setup_episodes(self):
        """Allocate the per-world episode clocks and commands; the first episodes are staggered so resets spread."""
        device = self.device
        count = self.world_count
        rows = max(1, int(np.ceil(count / len(self.scene_types))))
        base = np.zeros(count, dtype=np.int32)
        jitter = np.zeros(count, dtype=np.int32)
        first_length = np.zeros(count, dtype=np.int32)
        shortest = LANDING_FRAMES + self.fps // 2  # the shortest first episode still lands and walks half a second
        for world, name in enumerate(self.world_scene):
            scene_type = SCENE_TYPES[name]
            base[world] = max(1, round(scene_type.episode_seconds * self.fps))
            jitter[world] = round(scene_type.jitter_seconds * self.fps)
            # the worlds of one scene type get first episodes of different lengths, so their resets spread out
            # without any world skipping the landing window of its clock
            row = world // len(self.scene_types)
            first_length[world] = max(1, shortest + int((base[world] - shortest) * (1.0 - row / rows)))
        self.base_length = wp.array(base, dtype=wp.int32, device=device)
        self.jitter_length = wp.array(jitter, dtype=wp.int32, device=device)
        self.first_length = first_length
        self.episode_frame = wp.zeros(count, dtype=wp.int32, device=device)
        self.episode_length = wp.array(first_length, dtype=wp.int32, device=device)
        self.episode_count = wp.zeros(count, dtype=wp.int32, device=device)
        self.reset_count = wp.zeros(count, dtype=wp.int32, device=device)
        self.fall_count = wp.zeros(count, dtype=wp.int32, device=device)
        # walking robots end their episode early when they fall; the other scenes carry no base height
        heights = [ROBOTS[name].base_height if SCENE_TYPES[name].is_walker else 0.0 for name in self.world_scene]
        self.world_base_height = wp.array(heights, dtype=wp.float32, device=device)
        # the reset mask uses the (world_count + 1) layout of the pipeline; the global entry stays False
        self.reset_mask = wp.zeros(count + 1, dtype=wp.bool, device=device)
        self.all_worlds = wp.ones(count + 1, dtype=wp.bool, device=device)
        # the per-world command of the walkers (forward, sideways, yaw rate); zero for every other world
        walkers = [self.drive_mode[name] == "policy" for name in self.world_scene]
        self.world_is_walker = wp.array(walkers, dtype=wp.bool, device=device)
        visible_walkers = [w for w, walker in enumerate(walkers) if walker and w < self.visible_worlds]
        self.walker_world = wp.array(visible_walkers, dtype=wp.int32, device=device)
        self.command = wp.zeros((count, 3), dtype=wp.float32, device=device)
        self.draw_commands(self.all_worlds)

    def draw_commands(self, mask: wp.array):
        wp.launch(
            draw_commands,
            dim=self.world_count,
            inputs=[mask, self.world_is_walker, self.episode_count, self.seed],
            outputs=[self.command],
            device=self.device,
        )

    def setup_drives(self):
        """List the driven single-DOF joints of the scripted scene types with their rest targets and motion."""
        joint_type = self.model.joint_type.numpy()
        joint_world = self.model.joint_world.numpy()
        q_start = self.model.joint_q_start.numpy()
        qd_start = self.model.joint_qd_start.numpy()
        target_ke = self.model.joint_target_ke.numpy()
        target_kd = self.model.joint_target_kd.numpy()
        rest = self.model.joint_target_q.numpy()
        drives = {"sine": [], "random": []}
        for joint in range(self.model.joint_count):
            world = int(joint_world[joint])
            if world < 0 or joint_type[joint] not in (newton.JointType.REVOLUTE, newton.JointType.PRISMATIC):
                continue
            dof = int(qd_start[joint])
            if target_ke[dof] <= 0.0 and target_kd[dof] <= 0.0:
                continue
            name = self.world_scene[world]
            mode = self.drive_mode[name]
            if mode not in drives:
                continue
            scene_type = SCENE_TYPES[name]
            coord = int(q_start[joint])
            drives[mode].append((coord, world, float(rest[coord]), scene_type.amplitude, scene_type.frequency))
        self.drives = {}
        for mode, entries in drives.items():
            columns = list(zip(*entries, strict=True)) if entries else [[] for _ in range(5)]
            self.drives[mode] = {
                "count": len(entries),
                "coord": wp.array(columns[0], dtype=wp.int32, device=self.device),
                "world": wp.array(columns[1], dtype=wp.int32, device=self.device),
                "rest": wp.array(columns[2], dtype=wp.float32, device=self.device),
                "amplitude": wp.array(columns[3], dtype=wp.float32, device=self.device),
                "frequency": wp.array(columns[4], dtype=wp.float32, device=self.device),
                "phase": wp.zeros(len(entries), dtype=wp.float32, device=self.device),
                "scale": wp.ones(len(entries), dtype=wp.float32, device=self.device),
            }
        # the first episode of every world draws its phases as a reset would
        self.draw_drive_phases(self.all_worlds)

    def draw_drive_phases(self, mask: wp.array):
        sine = self.drives["sine"]
        if sine["count"]:
            wp.launch(
                draw_drive_phases,
                dim=sine["count"],
                inputs=[mask, sine["world"], self.episode_count, self.seed],
                outputs=[sine["phase"], sine["scale"]],
                device=self.device,
            )

    def setup_policies(self):
        """One batched policy per walking robot type that runs under its policy."""
        self.batches = [
            PolicyBatch(self.robots[name], [w for w, kind in enumerate(self.world_scene) if kind == name], self.device)
            for name in self.scene_types
            if self.drive_mode[name] == "policy"
        ]
        self.arrow_starts = wp.zeros(self.walker_world.shape[0], dtype=wp.vec3, device=self.device)
        self.commanded_ends = wp.zeros_like(self.arrow_starts)
        self.achieved_ends = wp.zeros_like(self.arrow_starts)

    def drive_text(self, name: str) -> str:
        scene_type = SCENE_TYPES[name]
        mode = self.drive_mode[name]
        if mode == "policy":
            robot = self.robots[name]
            return f"policy {os.path.basename(robot.policy_file)} ({robot.obs_width} -> {robot.num_dofs})"
        if mode == "random":
            hold = RANDOM_HOLD_FRAMES / self.fps
            return f"random targets within {scene_type.amplitude:.2f} rad, redrawn every {hold:.1f} s"
        if mode == "sine":
            return f"sine {scene_type.amplitude:.2f} rad at {scene_type.frequency:.1f} Hz"
        return "none"

    def print_summary(self):
        """Print what the batch is made of: the scene types, their sizes, their episodes and their drives."""
        dof_per_world = np.diff(self.model.joint_dof_world_start.numpy()[: self.world_count + 1])
        graph = " inside one CUDA graph" if self.device.is_cuda else ""
        print(
            f"Steel heterogeneous worlds: {self.world_count} worlds, {len(self.scene_types)} scene types "
            f"(round robin), solve_mode={self.solve_mode}, control {self.fps} Hz, physics "
            f"{self.fps * self.sim_substeps} Hz, rows per world: articulation {ARTICULATION_ROWS}, "
            f"free body {FREE_BODY_ROWS}, {len(self.batches)} policy evaluations per control step{graph}, "
            f"seed {self.seed}"
        )
        print(
            f"  {'scene type':<12}{'worlds':>8}{'dofs/world':>12}  {'articulations (dofs)':<24}"
            f"{'bodies':>8}{'shapes':>8}{'episode [s]':>13}  drive"
        )
        for name in self.scene_types:
            builder = self.builders[name]
            scene_type = SCENE_TYPES[name]
            first = self.world_scene.index(name)
            arts = ", ".join(str(d) for d in _articulation_dofs(builder))
            episode = (
                f"{scene_type.episode_seconds:.1f} to {scene_type.episode_seconds + scene_type.jitter_seconds:.1f}"
            )
            print(
                f"  {name:<12}{self.world_scene.count(name):>8}{int(dof_per_world[first]):>12}  {arts:<24}"
                f"{builder.body_count:>8}{builder.shape_count:>8}{episode:>13}  {self.drive_text(name)}"
            )

    def resets_per_scene_type(self) -> dict[str, int]:
        counts = self.reset_count.numpy()
        kinds = np.array(self.world_scene)
        return {name: int(counts[kinds == name].sum()) for name in self.scene_types}

    def falls_per_scene_type(self) -> dict[str, int]:
        counts = self.fall_count.numpy()
        kinds = np.array(self.world_scene)
        return {name: int(counts[kinds == name].sum()) for name in self.scene_types if SCENE_TYPES[name].is_walker}

    def walker_status(self) -> dict[str, np.ndarray]:
        """Per world, from the current state on the host: upright, forward speed in the base frame, command."""
        joint_q = self.state_0.joint_q.numpy()
        joint_qd = self.state_0.joint_qd.numpy()
        coord_start = self.model.joint_coord_world_start.numpy()
        dof_start = self.model.joint_dof_world_start.numpy()
        command = self.command.numpy()
        status = {key: np.zeros(self.world_count) for key in ("height", "tilt_cos", "forward", "commanded")}
        status["upright"] = np.zeros(self.world_count, dtype=bool)
        for world, name in enumerate(self.world_scene):
            if self.drive_mode[name] != "policy":
                continue
            qs, ds = coord_start[world], dof_start[world]
            rot = wp.quat(*joint_q[qs + 3 : qs + 7])
            height = joint_q[qs + 2] / self.robots[name].spec.base_height
            tilt_cos = wp.quat_rotate(rot, wp.vec3(0.0, 0.0, 1.0))[2]
            status["height"][world] = height
            status["tilt_cos"][world] = tilt_cos
            status["forward"][world] = wp.quat_rotate_inv(rot, wp.vec3(*joint_qd[ds : ds + 3]))[0]
            status["commanded"][world] = command[world, 0]
            status["upright"][world] = HEIGHT_MIN < height < HEIGHT_MAX and tilt_cos > UPRIGHT_COS
        return status

    def print_resets(self):
        falls = self.falls_per_scene_type()
        resets = ", ".join(
            f"{name} {count}" + (f" ({falls[name]} after a fall)" if falls.get(name) else "")
            for name, count in self.resets_per_scene_type().items()
        )
        marks = self.solver.constraint_row_watermarks()
        upright = ""
        if self.batches:
            status = self.walker_status()
            kinds = np.array(self.world_scene)
            upright = "; upright: " + ", ".join(
                f"{batch.robot.name} {int(status['upright'][kinds == batch.robot.name].sum())}/{len(batch.worlds)}"
                for batch in self.batches
            )
        print(
            f"  after {self.sim_time:.1f} s: resets by scene type: {resets}{upright}; row high-water per world: "
            f"articulation {marks['articulation_high_water']}/{ARTICULATION_ROWS}, "
            f"free body {marks['free_body_high_water']}/{FREE_BODY_ROWS}"
        )

    def reset(self):
        """Reset every world to its start state and restart the episode clocks (outside the graph)."""
        self.state_0.assign(self.initial)
        self.state_1.assign(self.initial)
        self.control.joint_target_q.assign(self.model.joint_target_q)
        self.control.joint_target_qd.assign(self.model.joint_target_qd)
        self.episode_frame.zero_()
        self.episode_length.assign(self.first_length)
        self.episode_count.zero_()
        self.reset_count.zero_()
        self.fall_count.zero_()
        self.reset_mask.zero_()
        self.draw_commands(self.all_worlds)
        self.draw_drive_phases(self.all_worlds)
        for batch in self.batches:
            batch.prev_action.zero_()
        self.solver.reset(self.state_0)
        self.pipeline.reset_contact_matching()
        self.sim_time = 0.0
        self.frame = 0
        self.setup_test_stats()

    def capture(self):
        self.graph = None
        if self.device.is_cuda and not wp.config.verify_cuda:
            # run one control step eagerly so first-step allocations and the policies' module loads happen outside
            # the capture, then start over
            self.simulate()
            self.reset()
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
            self.reset()

    def reset_finished_episodes(self):
        """Flag the worlds whose episode ended and restore them to their scene type's start state.

        Everything here is a device launch on preallocated arrays, so the whole sequence is captured with the
        policies and the physics substeps: the episode clocks, the masked copy of the start state, the solver's
        masked reset of its per-world history, the new commands, drive phases and cleared policy actions of the
        reset worlds, and the scripted targets of the control step.
        """
        device = self.device
        wp.launch(
            advance_episodes,
            dim=self.world_count,
            inputs=[
                self.base_length,
                self.jitter_length,
                self.seed,
                self.state_0.joint_q,
                self.model.joint_coord_world_start,
                self.world_base_height,
                LANDING_FRAMES,
            ],
            outputs=[
                self.episode_frame,
                self.episode_length,
                self.episode_count,
                self.reset_count,
                self.fall_count,
                self.reset_mask,
            ],
            device=device,
        )
        wp.launch(
            restore_joint_state,
            dim=self.model.joint_count,
            inputs=[
                self.reset_mask,
                self.model.joint_world,
                self.model.joint_q_start,
                self.model.joint_qd_start,
                self.initial.joint_q,
                self.initial.joint_qd,
                self.model.joint_target_q,
            ],
            outputs=[self.state_0.joint_q, self.state_0.joint_qd, self.control.joint_target_q],
            device=device,
        )
        wp.launch(
            restore_body_state,
            dim=self.model.body_count,
            inputs=[self.reset_mask, self.model.body_world, self.initial.body_q, self.initial.body_qd],
            outputs=[self.state_0.body_q, self.state_0.body_qd],
            device=device,
        )
        # the solver drops the reset worlds' cached dynamics, warm starts and friction history
        self.solver.reset(self.state_0, self.reset_mask)
        self.pipeline.reset_contact_matching(self.reset_mask)
        self.draw_commands(self.reset_mask)
        self.draw_drive_phases(self.reset_mask)
        for batch in self.batches:
            wp.launch(
                reset_prev_actions,
                dim=(len(batch.worlds), batch.robot.num_dofs),
                inputs=[self.reset_mask, batch.row_world],
                outputs=[batch.prev_action],
                device=device,
            )
        sine = self.drives["sine"]
        if sine["count"]:
            wp.launch(
                apply_sine_drives,
                dim=sine["count"],
                inputs=[
                    sine["coord"],
                    sine["world"],
                    sine["rest"],
                    sine["amplitude"],
                    sine["frequency"],
                    sine["phase"],
                    sine["scale"],
                    self.episode_frame,
                    self.frame_dt,
                ],
                outputs=[self.control.joint_target_q],
                device=device,
            )
        random = self.drives["random"]
        if random["count"]:
            wp.launch(
                apply_random_targets,
                dim=random["count"],
                inputs=[
                    random["coord"],
                    random["world"],
                    random["rest"],
                    random["amplitude"],
                    self.episode_frame,
                    self.episode_count,
                    self.seed,
                    RANDOM_HOLD_FRAMES,
                ],
                outputs=[self.control.joint_target_q],
                device=device,
            )

    def run_policies(self):
        """One batched policy evaluation per robot type: observations, inference, joint targets."""
        for batch in self.batches:
            batch.run(self.model, self.state_0, self.control, self.command)

    def simulate(self):
        """One control step: the episode resets and scripted targets, the policies, then the physics substeps."""
        self.reset_finished_episodes()
        self.run_policies()
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
        if self.sim_substeps % 2:
            # keep the captured input and output addresses stable for odd substep counts
            self.state_1.assign(self.state_0)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        self.frame += 1
        if self.frame % (10 * self.fps) == 0:
            self.print_resets()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        offsets = self.viewer.world_offsets
        if self.walker_world.shape[0] and offsets is not None and hasattr(self.viewer, "log_arrows"):
            wp.launch(
                command_arrows,
                dim=self.walker_world.shape[0],
                inputs=[
                    self.state_0.joint_q,
                    self.state_0.joint_qd,
                    self.model.joint_coord_world_start,
                    self.model.joint_dof_world_start,
                    self.walker_world,
                    offsets,
                    self.command,
                ],
                outputs=[self.arrow_starts, self.commanded_ends, self.achieved_ends],
                device=self.device,
            )
            self.viewer.log_arrows("command/commanded", self.arrow_starts, self.commanded_ends, (1.0, 0.75, 0.1))
            self.viewer.log_arrows("command/achieved", self.arrow_starts, self.achieved_ends, (0.2, 0.9, 1.0))
        self.viewer.end_frame()

    def setup_test_stats(self):
        """Per-world tallies of the walkers' upright frames and forward speed, kept in test mode."""
        self.judged_frames = np.zeros(self.world_count, dtype=np.int64)
        self.upright_frames = np.zeros(self.world_count, dtype=np.int64)
        self.forward_sum = np.zeros(self.world_count)
        self.commanded_sum = np.zeros(self.world_count)

    def test_post_step(self):
        """Reject nonfinite states after every control step and tally how the walkers do under their policies."""
        assert np.isfinite(self.state_0.body_q.numpy()).all(), "Nonfinite rigid body pose"
        assert np.isfinite(self.state_0.joint_qd.numpy()).all(), "Nonfinite joint velocity"
        if not self.batches:
            return
        # a walker is judged once it has landed and taken its first steps after a reset
        judged = self.episode_frame.numpy() >= LANDING_FRAMES
        status = self.walker_status()
        for batch in self.batches:
            for world in batch.worlds:
                if judged[world]:
                    self.judged_frames[world] += 1
                    self.upright_frames[world] += int(status["upright"][world])
                    self.forward_sum[world] += status["forward"][world]
                    self.commanded_sum[world] += status["commanded"][world]

    def test_final(self):
        """Check that every world reset at least once, no row budget overflowed, nothing exploded and the policies
        kept their robots upright and walking on command."""
        # the batch really differs per world: every world carries its scene type's DOF count
        dof_per_world = np.diff(self.model.joint_dof_world_start.numpy()[: self.world_count + 1])
        expected = [self.builders[name].joint_dof_count for name in self.world_scene]
        if dof_per_world.tolist() != expected:
            raise ValueError(f"per-world DOF counts {dof_per_world.tolist()} do not match the scene types {expected}")

        # every world has standard gravity (the robot USD files do not all agree; the last row is the global world)
        gravity = self.model.gravity.numpy()[: self.world_count]
        if not np.allclose(gravity, (0.0, 0.0, -9.81), atol=1.0e-3):
            raise ValueError(f"per-world gravity is not standard: {np.unique(gravity, axis=0).tolist()}")

        # every world finished at least one episode and was reset to its own scene type
        resets = self.reset_count.numpy()
        never = [f"{world} ({self.world_scene[world]})" for world in range(self.world_count) if resets[world] < 1]
        if never:
            raise ValueError(f"worlds that never reset in {self.frame} frames: {', '.join(never)}")

        # no world ran out of constraint rows and no contact rows were dropped
        self.solver.check_constraint_capacity()
        marks = self.solver.constraint_row_watermarks()
        bad = {key: value for key, value in marks.items() if ("overflow" in key or "dropped" in key) and value}
        if bad:
            raise ValueError(f"constraint rows overflowed or were dropped: {bad}")

        # nothing fell through the ground or flew away; Warp reads each lambda from its own source line
        model, state = self.model, self.state_0
        newton.examples.test_body_state(model, state, "bodies above the ground", lambda q, qd: q[2] > -0.05)
        newton.examples.test_body_state(model, state, "bodies not flying away", lambda q, qd: wp.length(qd) < 50.0)

        # the policies keep their robots upright after landing and move them the way they were commanded
        kinds = np.array(self.world_scene)
        for batch in self.batches:
            name = batch.robot.name
            worlds = kinds == name
            judged = int(self.judged_frames[worlds].sum())
            if judged == 0:
                continue
            upright = self.upright_frames[worlds].sum() / judged
            forward = self.forward_sum[worlds].sum() / judged
            commanded = self.commanded_sum[worlds].sum() / judged
            falls = int(self.fall_count.numpy()[worlds].sum())
            print(
                f"  {name}: upright in {100.0 * upright:.0f}% of {judged} judged world-steps, forward speed "
                f"{forward:.2f} m/s against {commanded:.2f} m/s commanded, {falls} episodes ended by a fall"
            )
            if upright < 0.9:
                raise ValueError(f"{name} robots were upright in only {100.0 * upright:.0f}% of the judged steps")
            if forward < 0.3 * commanded:
                raise ValueError(f"{name} robots walked at {forward:.2f} m/s against {commanded:.2f} m/s commanded")
        self.print_resets()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_world_count_arg(parser)
        newton.examples.add_max_worlds_arg(parser)
        parser.set_defaults(num_frames=500, world_count=None)
        parser.add_argument(
            "--scene-types",
            nargs="+",
            choices=list(SCENE_TYPES),
            default=list(SCENE_TYPES),
            metavar="TYPE",
            help=f"Scene types in the batch, assigned round robin over the worlds. The default world count is "
            f"the square of the number of scene types, so each viewer column holds one scene type. "
            f"Choices: {', '.join(SCENE_TYPES)}.",
        )
        parser.add_argument(
            "--random-actions",
            nargs="*",
            choices=WALKERS,
            default=[],
            metavar="ROBOT",
            help=f"Walking robots that follow uniformly random joint targets instead of their policy. "
            f"Choices: {', '.join(WALKERS)}.",
        )
        parser.add_argument("--substeps", type=int, default=4, help="Physics substeps per 50 Hz control step.")
        parser.add_argument("--iterations", type=int, default=12, help="Steel sweeps per substep.")
        parser.add_argument("--seed", type=int, default=0, help="Seed of the episode commands, phases and lengths.")
        parser.add_argument(
            "--deterministic",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Deterministic contact ordering and row reservation, so repeated runs agree bitwise.",
        )
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    example = Example(viewer, args)
    newton.examples.run(example, args)
