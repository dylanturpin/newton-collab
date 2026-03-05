# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Robot control via keyboard
#
# Shows how to control robot pretrained in IsaacLab with RL.
# The policy is loaded from a file and the robot is controlled via keyboard.
#
# Press "p" to reset the robot.
# Press "i", "j", "k", "l", "u", "o" to move the robot.
# Run this example with:
# python -m newton.examples robot_policy --robot g1_29dof
# python -m newton.examples robot_policy --robot g1_23dof
# python -m newton.examples robot_policy --robot go2
# python -m newton.examples robot_policy --robot anymal
# python -m newton.examples robot_policy --robot anymal --physx
# to run the example with a PhysX-trained policy run with --physx
###########################################################################

from dataclasses import dataclass
from typing import Any

import torch
import warp as wp
import yaml

import newton
import newton.examples
import newton.utils
from newton import JointTargetMode, State


@dataclass
class RobotConfig:
    """Configuration for a robot including asset paths and policy paths."""

    asset_dir: str
    policy_path: dict[str, str]
    asset_path: str
    yaml_path: str  # Path within the asset directory to the configuration YAML


# Robot configurations pointing to newton-assets repository
ROBOT_CONFIGS = {
    "anymal": RobotConfig(
        asset_dir="anybotics_anymal_c",
        policy_path={"mjw": "rl_policies/mjw_anymal.pt", "physx": "rl_policies/physx_anymal.pt"},
        asset_path="usd/anymal_c.usda",
        yaml_path="rl_policies/anymal.yaml",
    ),
    "go2": RobotConfig(
        asset_dir="unitree_go2",
        policy_path={"mjw": "rl_policies/mjw_go2.pt", "physx": "rl_policies/physx_go2.pt"},
        asset_path="usd/go2.usda",
        yaml_path="rl_policies/go2.yaml",
    ),
    "g1_29dof": RobotConfig(
        asset_dir="unitree_g1",
        policy_path={"mjw": "rl_policies/mjw_g1_29DOF.pt"},
        asset_path="usd/g1_isaac.usd",
        yaml_path="rl_policies/g1_29dof.yaml",
    ),
    "g1_23dof": RobotConfig(
        asset_dir="unitree_g1",
        policy_path={"mjw": "rl_policies/mjw_g1_23DOF.pt", "physx": "rl_policies/physx_g1_23DOF.pt"},
        asset_path="usd/g1_minimal.usd",
        yaml_path="rl_policies/g1_23dof.yaml",
    ),
}


@torch.jit.script
def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by the inverse of a quaternion.

    Args:
        q: The quaternion in (x, y, z, w). Shape is (..., 4).
        v: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    q_w = q[..., 3]  # w component is at index 3 for XYZW format
    q_vec = q[..., :3]  # xyz components are at indices 0, 1, 2
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    # for two-dimensional tensors, bmm is faster than einsum
    if q_vec.dim() == 2:
        c = q_vec * torch.bmm(q_vec.view(q.shape[0], 1, 3), v.view(q.shape[0], 3, 1)).squeeze(-1) * 2.0
    else:
        c = q_vec * torch.einsum("...i,...i->...", q_vec, v).unsqueeze(-1) * 2.0
    return a - b + c


def compute_obs(
    actions: torch.Tensor,
    state: State,
    joint_pos_initial_full: torch.Tensor,
    device: str,
    indices: torch.Tensor,
    gravity_vec: torch.Tensor,
    command: torch.Tensor,
    subset_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute observation for robot policy.

    Args:
        actions: Previous actions tensor
        state: Current simulation state
        joint_pos_initial: Initial joint positions
        device: PyTorch device string
        indices: Index mapping for joint reordering
        gravity_vec: Gravity vector in world frame
        command: Command vector

    Returns:
        Observation tensor for policy input
    """
    # Extract state information with proper handling
    joint_q = state.joint_q if state.joint_q is not None else []
    joint_qd = state.joint_qd if state.joint_qd is not None else []

    root_pos_w = torch.tensor(joint_q[0:3], device=device, dtype=torch.float32).unsqueeze(0)
    root_quat_w = torch.tensor(joint_q[3:7], device=device, dtype=torch.float32).unsqueeze(0)
    root_lin_vel_origin_w = torch.tensor(joint_qd[:3], device=device, dtype=torch.float32).unsqueeze(0)
    root_ang_vel_w = torch.tensor(joint_qd[3:6], device=device, dtype=torch.float32).unsqueeze(0)
    # For free roots, qd[:3] stores linear velocity at world origin. Convert to
    # root-frame linear velocity at the root position to remove origin dependence.
    root_lin_vel_w = root_lin_vel_origin_w + torch.cross(root_ang_vel_w, root_pos_w, dim=1)
    joint_pos_current = torch.tensor(joint_q[7:], device=device, dtype=torch.float32).unsqueeze(0)
    joint_vel_current = torch.tensor(joint_qd[6:], device=device, dtype=torch.float32).unsqueeze(0)

    vel_b = quat_rotate_inverse(root_quat_w, root_lin_vel_w)
    a_vel_b = quat_rotate_inverse(root_quat_w, root_ang_vel_w)
    grav = quat_rotate_inverse(root_quat_w, gravity_vec)
    joint_pos_rel = joint_pos_current - joint_pos_initial_full
    joint_vel_rel = joint_vel_current
    rearranged_joint_pos_rel = torch.index_select(joint_pos_rel, 1, indices)
    rearranged_joint_vel_rel = torch.index_select(joint_vel_rel, 1, indices)
    if subset_idx is not None:
        rearranged_joint_pos_rel = torch.index_select(rearranged_joint_pos_rel, 1, subset_idx)
        rearranged_joint_vel_rel = torch.index_select(rearranged_joint_vel_rel, 1, subset_idx)
        if actions.shape[1] != subset_idx.shape[0]:
            actions = torch.index_select(actions, 1, subset_idx)
    obs = torch.cat([vel_b, a_vel_b, grav, command, rearranged_joint_pos_rel, rearranged_joint_vel_rel, actions], dim=1)

    return obs


def load_policy_and_setup_tensors(
    example: Any,
    policy_path: str,
    policy_dofs: int,
    joint_pos_slice: slice,
    subset_idx_mjc: torch.Tensor | None,
):
    """Load policy and setup initial tensors for robot control.

    Args:
        example: Robot example instance
        policy_path: Path to the policy file
        num_dofs: Number of degrees of freedom
        joint_pos_slice: Slice for extracting joint positions from state
    """
    device = example.torch_device
    print("[INFO] Loading policy from:", policy_path)
    example.policy = torch.jit.load(policy_path, map_location=device)

    # Handle potential None state
    joint_q = example.state_0.joint_q if example.state_0.joint_q is not None else []
    full_joint_pos = torch.tensor(joint_q[joint_pos_slice], device=device, dtype=torch.float32).unsqueeze(0)
    example.joint_pos_initial_full = full_joint_pos
    example.joint_pos_initial = (
        torch.index_select(full_joint_pos, 1, subset_idx_mjc) if subset_idx_mjc is not None else full_joint_pos
    )
    example.act = torch.zeros(1, policy_dofs, device=device, dtype=torch.float32)
    example.rearranged_act = torch.zeros(1, policy_dofs, device=device, dtype=torch.float32)


def find_physx_mjwarp_mapping(mjwarp_joint_names, physx_joint_names):
    """
    Finds the mapping between PhysX and MJWarp joint names.
    Returns a tuple of two lists: (mjc_to_physx, physx_to_mjc).
    """
    mjc_to_physx = []
    physx_to_mjc = []
    for j in mjwarp_joint_names:
        if j in physx_joint_names:
            mjc_to_physx.append(physx_joint_names.index(j))

    for j in physx_joint_names:
        if j in mjwarp_joint_names:
            physx_to_mjc.append(mjwarp_joint_names.index(j))

    return mjc_to_physx, physx_to_mjc


class Example:
    def __init__(
        self,
        viewer,
        robot_config: RobotConfig,
        config,
        asset_directory: str,
        mjc_to_physx: list[int],
        physx_to_mjc: list[int],
        subset_idx_mjc: list[int] | None,
        policy_dofs: int,
        solver_type: str = "feather_pgs",
    ):
        # Setup simulation parameters first
        fps = 200
        self.frame_dt = 1.0e0 / fps
        self.decimation = 4
        self.cycle_time = 1 / fps * self.decimation

        # Group related attributes by prefix
        self.sim_time = 0.0
        self.sim_step = 0
        self.sim_substeps = 1
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.solver_type = solver_type

        # Save a reference to the viewer
        self.viewer = viewer

        # Store configuration
        self.use_mujoco = False
        self.config = config
        self.robot_config = robot_config
        self.total_dofs = config["num_dofs"]
        self.policy_dofs = policy_dofs
        self.policy_subset_idx_mjc = None

        # Device setup
        self.device = wp.get_device()
        self.torch_device = "cuda" if self.device.is_cuda else "cpu"

        # Build the model
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
            armature=0.1,
            limit_ke=1.0e2,
            limit_kd=1.0e0,
        )
        builder.default_shape_cfg.ke = 5.0e4
        builder.default_shape_cfg.kd = 5.0e2
        builder.default_shape_cfg.kf = 1.0e3
        builder.default_shape_cfg.mu = 0.75

        builder.add_usd(
            newton.examples.get_asset(asset_directory + "/" + robot_config.asset_path),
            xform=wp.transform(wp.vec3(0, 0, 0.8)),
            collapse_fixed_joints=False,
            enable_self_collisions=False,
            joint_ordering="dfs",
            hide_collision_shapes=True,
        )
        # builder.approximate_meshes("convex_hull")
        # builder.approximate_meshes("bounding_box")

        builder.add_ground_plane()
        # builder's gravity isn't a vec3. use model.set_gravity()
        # builder.gravity = wp.vec3(0.0, 0.0, -9.81)

        builder.joint_q[:3] = [0.0, 0.0, 0.76]
        builder.joint_q[3:7] = [0.0, 0.0, 0.7071, 0.7071]
        builder.joint_q[7:] = config["mjw_joint_pos"]

        # config["action_scale"] = 0.0
        # stiffness_scale = 1.5
        # damping_scale = 1.2
        stiffness_scale = 1.0
        damping_scale = 1.0
        for i in range(len(config["mjw_joint_stiffness"])):
            builder.joint_target_ke[i + 6] = config["mjw_joint_stiffness"][i] * stiffness_scale
            builder.joint_target_kd[i + 6] = config["mjw_joint_damping"][i] * damping_scale
            builder.joint_armature[i + 6] = config["mjw_joint_armature"][i]
            builder.joint_target_mode[i + 6] = int(JointTargetMode.POSITION)

        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, -9.81))

        # Create solver based on solver_type
        if solver_type == "feather_pgs":
            from newton._src.solvers import SolverFeatherPGS

            self.solver = SolverFeatherPGS(
                self.model,
                update_mass_matrix_interval=1,
                pgs_iterations=32,
                dense_max_constraints=64,
                pgs_warmstart=False,
                pgs_omega=1.0,
                pgs_beta=0.1,
                storage="batched",
            )
        elif solver_type == "mujoco":
            self.solver = newton.solvers.SolverMuJoCo(
                self.model,
                njmax=210,
                nconmax=35,
                ls_iterations=10,
                ls_parallel=True,
                cone="pyramidal",
                impratio=1,
                integrator="implicit",
            )
        else:
            raise ValueError(f"Unknown solver type: {solver_type}")

        print(f"[INFO] Using solver: {solver_type}")

        # Initialize state objects
        self.state_temp = self.model.state()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = newton.Contacts(self.solver.get_max_contact_count(), 0)

        # Set model in viewer
        self.viewer.set_model(self.model)
        self.viewer.vsync = True

        # Ensure FK evaluation (for non-MuJoCo solvers)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

        # Store initial joint state for fast reset
        self._initial_joint_q = wp.clone(self.state_0.joint_q)
        self._initial_joint_qd = wp.clone(self.state_0.joint_qd)

        # Pre-compute tensors that don't change during simulation
        self.physx_to_mjc_indices = torch.tensor(physx_to_mjc, device=self.torch_device, dtype=torch.long)
        self.mjc_to_physx_indices = torch.tensor(mjc_to_physx, device=self.torch_device, dtype=torch.long)
        self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.torch_device, dtype=torch.float32).unsqueeze(0)
        self.command = torch.zeros((1, 3), device=self.torch_device, dtype=torch.float32)
        self._reset_key_prev = False
        if subset_idx_mjc:
            self.policy_subset_idx_mjc = torch.tensor(
                subset_idx_mjc, device=self.torch_device, dtype=torch.long, requires_grad=False
            )

        # Initialize policy-related attributes
        # (will be set by load_policy_and_setup_tensors)
        self.policy = None
        self.joint_pos_initial = None
        self.joint_pos_initial_full = None
        self.act = None
        self.rearranged_act = None

        # Call capture at the end
        self.capture()

    def capture(self):
        """Put graph capture into it's own method."""
        self.graph = None
        self.use_cuda_graph = False
        if wp.get_device().is_cuda and wp.is_mempool_enabled(wp.get_device()):
            print("[INFO] Using CUDA graph")
            self.use_cuda_graph = True
            torch_tensor = torch.zeros(self.config["num_dofs"] + 6, device=self.torch_device, dtype=torch.float32)
            self.control.joint_target_pos = wp.from_torch(torch_tensor, dtype=wp.float32, requires_grad=False)
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        """Simulate performs one frame's worth of updates."""
        need_state_copy = self.use_cuda_graph and self.sim_substeps % 2 == 1

        for i in range(self.sim_substeps):
            self.state_0.clear_forces()

            # Apply forces to the model for picking, wind, etc
            self.viewer.apply_forces(self.state_0)

            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)

            # Swap states - handle CUDA graph case specially
            if need_state_copy and i == self.sim_substeps - 1:
                # Swap states by copying the state arrays for graph capture
                self.state_0.assign(self.state_1)
            else:
                # We can just swap the state references
                self.state_0, self.state_1 = self.state_1, self.state_0

        self.solver.update_contacts(self.contacts, self.state_0)

    def reset(self):
        print("[INFO] Resetting example")
        # Restore initial joint positions and velocities in-place.
        wp.copy(self.state_0.joint_q, self._initial_joint_q)
        wp.copy(self.state_0.joint_qd, self._initial_joint_qd)
        wp.copy(self.state_1.joint_q, self._initial_joint_q)
        wp.copy(self.state_1.joint_qd, self._initial_joint_qd)
        # Recompute forward kinematics to refresh derived state.
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.state_1.joint_q, self.state_1.joint_qd, self.state_1)

    def step(self):
        # # --- FLOATING ORIGIN FIX ---
        # # Check if we have drifted far from the origin
        # root_pos = self.state_0.joint_q.numpy()[:3]
        # dist_sq = root_pos[0]**2 + root_pos[1]**2

        # # If we are more than 2 meters away, teleport back
        # if dist_sq > 0.05:
        #     # Calculate the offset
        #     offset_x = root_pos[0]
        #     offset_y = root_pos[1]

        #     # Teleport state_0
        #     # Note: We do NOT touch Z (index 2), orientation, or velocities
        #     current_q_0 = self.state_0.joint_q.numpy()
        #     current_q_0[0] -= offset_x
        #     current_q_0[1] -= offset_y
        #     self.state_0.joint_q.assign(current_q_0)

        #     # Teleport state_1 (to keep integration consistent)
        #     current_q_1 = self.state_1.joint_q.numpy()
        #     current_q_1[0] -= offset_x
        #     current_q_1[1] -= offset_y
        #     self.state_1.joint_q.assign(current_q_1)

        #     # Teleport the camera so the viewer doesn't see the snap
        #     # (Optional, depends on how your viewer camera is implemented)
        #     # if hasattr(self.viewer, "camera_pos"):
        #     #      cam_pos = list(self.viewer.camera_pos)
        #     #      cam_pos[0] -= offset_x
        #     #      cam_pos[1] -= offset_y
        #     #      self.viewer.camera_pos = tuple(cam_pos)

        #     #      tgt_pos = list(self.viewer.camera_look_at)
        #     #      tgt_pos[0] -= offset_x
        #     #      tgt_pos[1] -= offset_y
        #     #      self.viewer.camera_look_at = tuple(tgt_pos)

        #     # Important: Re-evaluate FK immediately so body transforms
        #     # match the new joint positions before the physics step
        #     newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        # ---------------------------
        # Build command from viewer keyboard
        if hasattr(self.viewer, "is_key_down"):
            fwd = 1.0 if self.viewer.is_key_down("i") else (-1.0 if self.viewer.is_key_down("k") else 0.0)
            lat = 0.5 if self.viewer.is_key_down("j") else (-0.5 if self.viewer.is_key_down("l") else 0.0)
            rot = 1.0 if self.viewer.is_key_down("u") else (-1.0 if self.viewer.is_key_down("o") else 0.0)
            self.command[0, 0] = float(fwd)
            self.command[0, 1] = float(lat)
            self.command[0, 2] = float(rot)
            # Reset when 'P' is pressed (edge-triggered)
            reset_down = bool(self.viewer.is_key_down("p"))
            if reset_down and not self._reset_key_prev:
                self.reset()
            self._reset_key_prev = reset_down

        obs = compute_obs(
            self.act,
            self.state_0,
            self.joint_pos_initial_full,
            self.torch_device,
            self.physx_to_mjc_indices,
            self.gravity_vec,
            self.command,
            self.policy_subset_idx_mjc,
        )
        with torch.no_grad():
            self.act = self.policy(obs)
            if self.policy_subset_idx_mjc is not None:
                full_act_mjc = torch.zeros(
                    1, self.total_dofs, device=self.torch_device, dtype=torch.float32, requires_grad=False
                )
                full_act_mjc[0, self.policy_subset_idx_mjc] = self.act[0]
            else:
                full_act_mjc = self.act
            self.rearranged_act = torch.index_select(full_act_mjc, 1, self.mjc_to_physx_indices)
            a = self.joint_pos_initial_full + self.config["action_scale"] * self.rearranged_act
            a_with_zeros = torch.cat([torch.zeros(6, device=self.torch_device, dtype=torch.float32), a.squeeze(0)])
            a_wp = wp.from_torch(a_with_zeros, dtype=wp.float32, requires_grad=False)
            wp.copy(self.control.joint_target_pos, a_wp)

        for _ in range(self.decimation):
            if self.graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()

        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        newton.examples.test_body_state(
            self.model,
            self.state_0,
            "all bodies are above the ground",
            lambda q, qd: q[2] > 0.0,
        )


if __name__ == "__main__":
    # Create parser that inherits common arguments and adds
    # example-specific ones
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--robot", type=str, default="g1_29dof", choices=list(ROBOT_CONFIGS.keys()), help="Robot name to load"
    )
    parser.add_argument("--physx", action="store_true", help="Run physX policy instead of MJWarp.")
    parser.add_argument(
        "--solver",
        type=str,
        choices=["mujoco", "feather_pgs"],
        default="feather_pgs",
        help="Which articulated solver to use.",
    )
    parser.add_argument(
        "--lower-body-only",
        action="store_true",
        help="Use lower-body-only obs/action layout to match IL v1 (G1 only).",
    )

    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init(parser)

    # Get robot configuration
    if args.robot not in ROBOT_CONFIGS:
        print(f"[ERROR] Unknown robot: {args.robot}")
        print(f"[INFO] Available robots: {list(ROBOT_CONFIGS.keys())}")
        exit(1)

    robot_config = ROBOT_CONFIGS[args.robot]
    print(f"[INFO] Selected robot: {args.robot}")

    # Download assets from newton-assets repository
    asset_directory = str(newton.utils.download_asset(robot_config.asset_dir))
    print(f"[INFO] Asset directory: {asset_directory}")

    # Load robot configuration from YAML file in the downloaded assets
    yaml_file_path = f"{asset_directory}/{robot_config.yaml_path}"
    try:
        with open(yaml_file_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"[ERROR] Robot config file not found: {yaml_file_path}")
        exit(1)
    except yaml.YAMLError as e:
        print(f"[ERROR] Error parsing YAML file: {e}")
        exit(1)

    print(f"[INFO] Loaded config with {config['num_dofs']} DOFs")

    lower_body_order = [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
    ]
    subset_idx_mjc = None
    policy_dofs = config["num_dofs"]

    if args.lower_body_only:
        if args.robot not in ("g1_29dof",):
            raise ValueError("--lower-body-only is currently supported only for g1_29dof.")
        if "mjw_joint_names" not in config:
            raise ValueError("mjw_joint_names missing from config; cannot build lower-body mapping.")
        name_to_idx = {name: idx for idx, name in enumerate(config["mjw_joint_names"])}
        missing = [name for name in lower_body_order if name not in name_to_idx]
        if missing:
            raise ValueError(f"Lower-body joints not found in MJWarp joint list: {missing}")
        subset_idx_mjc = [name_to_idx[name] for name in lower_body_order]
        policy_dofs = len(subset_idx_mjc)
        print(f"[INFO] Using lower-body-only mode: policy DOFs {policy_dofs} of {config['num_dofs']} total.")

    mjc_to_physx = list(range(config["num_dofs"]))
    physx_to_mjc = list(range(config["num_dofs"]))

    if args.physx:
        if "physx" not in robot_config.policy_path or "physx_joint_names" not in config:
            physx_robots = [name for name, cfg in ROBOT_CONFIGS.items() if "physx" in cfg.policy_path]
            print(f"[ERROR] PhysX policy not available for robot '{args.robot}'.")
            print(f"[INFO] Robots with PhysX support: {physx_robots}")
            exit(1)
        policy_path = f"{asset_directory}/{robot_config.policy_path['physx']}"
        mjc_to_physx, physx_to_mjc = find_physx_mjwarp_mapping(config["mjw_joint_names"], config["physx_joint_names"])
    else:
        policy_path = f"{asset_directory}/{robot_config.policy_path['mjw']}"

    example = Example(
        viewer,
        robot_config,
        config,
        asset_directory,
        mjc_to_physx,
        physx_to_mjc,
        subset_idx_mjc,
        policy_dofs,
        solver_type=args.solver,
    )

    # Use utility function to load policy and setup tensors
    subset_idx_tensor = example.policy_subset_idx_mjc
    load_policy_and_setup_tensors(example, policy_path, policy_dofs, slice(7, None), subset_idx_tensor)

    # Run using standard example loop
    newton.examples.run(example, args)
