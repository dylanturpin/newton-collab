# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Franka Shirt Fold and Stack (proxy coupling)
#
# The T-shirt and Franka of the cloth Franka example, at meter scale and
# with two-way coupling. A fixed-base Franka, simulated by MuJoCo or
# FeatherPGS, folds the VBD T-shirt with scripted IK keyframes, then picks
# two rigid blocks off a pedestal and stacks them on the folded shirt.
# SolverCoupledProxy exposes the gripper and the blocks to VBD as proxies,
# so the cloth both resists the fingers and carries the block stack. Cloth
# grasps are not deterministic: task_success() reports each run's outcome.
#
# Command: python -m newton.examples franka_shirt_fold_stack
#          python -m newton.examples franka_shirt_fold_stack --rigid-solver featherpgs
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledProxy
from pxr import Usd

import newton
import newton.examples
import newton.ik as ik
import newton.usd
import newton.utils
from newton.solvers import SolverFeatherPGS, SolverMuJoCo, SolverVBD

FRANKA_BASE = (-0.5, -0.5, 0.0)
FRANKA_Q = [0.0, 0.0, 0.0, -1.59695, 0.0, 2.5307, 0.785, 0.032, 0.032]

TABLE_CENTER = (0.0, -0.5, 0.1)
TABLE_HALF = (0.4, 0.4, 0.1)
TABLE_TOP = TABLE_CENTER[2] + TABLE_HALF[2]

PEDESTAL_CENTER = (-0.28, -0.05, 0.1)
PEDESTAL_HALF = (0.1, 0.04, 0.1)
BLOCK_HALF = 0.025
BLOCK_MASS = 0.1
BLOCK_XS = (-0.33, -0.23)

# Occupancy-grid cell [m] for the shirt footprint metric.
FOOTPRINT_CELL = 0.02
# The stack goes on the flattest block-sized patch within this square around the center [m].
STACK_SEARCH_CENTER = (0.0, -0.6)
STACK_SEARCH_RADIUS = 0.08
# Blocks travel above the folded pile.
CARRY_Z = TABLE_TOP + 0.3
# Success thresholds: folded footprint fraction of the flat shirt, and stack drift after settling [m].
FOLD_FOOTPRINT_RATIO = 0.75
STACK_DRIFT_LIMIT = 0.01

GRIP_OPEN = 0.032
GRIP_PINCH = 0.0
GRIP_BLOCK = 0.0

CLOTH_RADIUS = 0.003
CLOTH_CONTACT_GAP = 0.005
# Cloth-side contact material: the self-contact spring, averaged with each shape's ke/kd for cloth-body contacts.
# The cloth Franka example's cloth-robot contact in SI at this cloth density; stiffer self-contact makes the cloth bounce.
CLOTH_SELF_KE = 30.0
CLOTH_SELF_KD = 0.03
# Finger and block material, stiff so the averaged contact pinches the cloth and carries the stack.
# MuJoCo uses their raw solref instead, so these reach only VBD.
CLOTH_KE = 2.0e4
CLOTH_KD = 2.0e1
CLOTH_MU = 5.0
# Gripper friction outside cloth pinches, so released cloth slides off the hand and fingers.
FINGER_RELEASE_MU = 0.5
GRASP_KE = 1.0e4
GRASP_KD = 1.0e1
SOLREF_MODE_RAW = 1

# Top-down hand orientations (x, y, z, w): fingers close along world y, or along world x.
HAND_DOWN = (1.0, 0.0, 0.0, 0.0)
HAND_DOWN_YAW = (0.70710678, 0.70710678, 0.0, 0.0)
CLOTH_GRASP_Z = TABLE_TOP + 0.004
FOLD_LIFT_Z = TABLE_TOP + 0.10
HEM_LIFT_Z = TABLE_TOP + 0.12
FOLD_LIFT_TIME = 1.5
FOLD_MOVE_TIME = 3.0
RELEASE_SLIDE = 0.08


def fold_keys(grasp, release, hand=HAND_DOWN, lift=FOLD_LIFT_Z):
    """Keyframes [duration s, x, y, z, hand quat xyzw, finger opening] for one pinch-and-fold move."""
    (gx, gy), (rx, ry) = grasp, release
    # After letting go, slide on past the fold so the fingers leave the flap instead of lifting it.
    direction = np.asarray((rx - gx, ry - gy)) / np.hypot(rx - gx, ry - gy)
    sx, sy = np.asarray((rx, ry)) + RELEASE_SLIDE * direction
    return [
        [1.5, gx, gy, lift, *hand, GRIP_OPEN],
        [1.0, gx, gy, CLOTH_GRASP_Z, *hand, GRIP_OPEN],
        [1.0, gx, gy, CLOTH_GRASP_Z, *hand, GRIP_PINCH],
        [FOLD_LIFT_TIME, gx, gy, lift, *hand, GRIP_PINCH],
        [FOLD_MOVE_TIME, rx, ry, lift, *hand, GRIP_PINCH],
        [1.0, rx, ry, CLOTH_GRASP_Z + 0.03, *hand, GRIP_PINCH],
        [0.75, rx, ry, CLOTH_GRASP_Z + 0.03, *hand, GRIP_OPEN],
        [1.0, sx, sy, CLOTH_GRASP_Z + 0.03, *hand, GRIP_OPEN],
        [0.75, sx, sy, lift, *hand, GRIP_OPEN],
    ]


# Sleeves in, then each half of the hem up over the collar.
FOLD_KEYS = [
    *fold_keys((0.27, -0.6), (0.05, -0.6), HAND_DOWN),
    *fold_keys((-0.27, -0.6), (-0.05, -0.6), HAND_DOWN),
    *fold_keys((0.1, -0.18), (0.1, -0.72), HAND_DOWN_YAW, lift=HEM_LIFT_Z),
    *fold_keys((-0.1, -0.18), (-0.1, -0.72), HAND_DOWN_YAW, lift=HEM_LIFT_Z),
]


@wp.kernel
def set_gripper_q(joint_q: wp.array2d[float], finger_pos: wp.array[float], idx0: int, idx1: int):
    world_idx = wp.tid()
    joint_q[world_idx, idx0] = finger_pos[world_idx]
    joint_q[world_idx, idx1] = finger_pos[world_idx]


@wp.kernel
def set_shape_mu(shape_ids: wp.array[int], mu: float, shape_mu: wp.array[float]):
    shape_mu[shape_ids[wp.tid()]] = mu


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = max(1, int(args.substeps))
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.use_graph = bool(args.graph_capture)
        self.rigid_solver = args.rigid_solver
        self.world_count = max(1, int(args.world_count))

        self._build_scene()
        self.use_graph = self.use_graph and self.device.is_cuda
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(
            self.model, broad_phase="explicit", soft_contact_gap=CLOTH_CONTACT_GAP
        )
        self.contacts = self.collision_pipeline.contacts()
        if self.rigid_solver == "featherpgs":
            # FeatherPGS sizes its contact scratch from the model before it sees a contact buffer.
            self.model.rigid_contact_max = self.contacts.rigid_contact_max
        self._build_solver(args)
        self._build_ik()

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.solver.prepare_contacts(self.contacts)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_1)
        self.flat_extent = None

        newton.examples.configure_coupled_view(self, args)
        if self.world_count > 1:
            self.viewer.set_world_offsets((2.0, 2.0, 0.0))
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer.set_camera(pos=wp.vec3(0.75, -1.35, 1.05), pitch=-38.0, yaw=125.0)
            self.viewer.camera.look_at(wp.vec3(-0.05, -0.4, 0.2))

        self.capture()

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------
    @staticmethod
    def _add_franka(builder):
        builder.add_urdf(
            newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
            xform=wp.transform(wp.vec3(*FRANKA_BASE), wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
            parse_visuals_as_colliders=False,
            force_show_colliders=False,
        )
        builder.joint_q[: len(FRANKA_Q)] = FRANKA_Q
        builder.joint_target_q[: len(FRANKA_Q)] = FRANKA_Q

    def _emit_template(self, builder):
        franka_body_start = builder.body_count
        franka_joint_start = builder.joint_count
        self._add_franka(builder)
        builder.joint_target_ke[:7] = [600.0] * 7
        builder.joint_target_kd[:7] = [80.0] * 7
        builder.joint_target_ke[7:9] = [4000.0, 4000.0]
        builder.joint_target_kd[7:9] = [100.0, 100.0]
        builder.joint_effort_limit[:4] = [87.0] * 4
        builder.joint_effort_limit[4:7] = [12.0] * 3
        builder.joint_effort_limit[7:9] = [200.0, 200.0]
        builder.joint_armature[:7] = [1.0e-3] * 7
        self.franka_bodies = list(range(franka_body_start, builder.body_count))
        self.franka_joints = list(range(franka_joint_start, builder.joint_count))

        gravcomp = builder.custom_attributes["mujoco:gravcomp"]
        if gravcomp.values is None:
            gravcomp.values = {}
        for body in self.franka_bodies:
            gravcomp.values[body] = 1.0
            if self.rigid_solver == "featherpgs":
                builder.body_disable_gravity[body] = True

        block_cfg = newton.ModelBuilder.ShapeConfig(
            density=BLOCK_MASS / (2.0 * BLOCK_HALF) ** 3, ke=CLOTH_KE, kd=CLOTH_KD, mu=1.0, gap=0.005
        )
        pedestal_top = PEDESTAL_CENTER[2] + PEDESTAL_HALF[2]
        self.block_bodies, self.block_joints = [], []
        for x in BLOCK_XS:
            body = builder.add_body(
                xform=wp.transform((x, PEDESTAL_CENTER[1], pedestal_top + BLOCK_HALF + 0.002), wp.quat_identity()),
                label=f"block_{len(self.block_bodies)}",
            )
            builder.add_shape_box(body, hx=BLOCK_HALF, hy=BLOCK_HALF, hz=BLOCK_HALF, cfg=block_cfg)
            self.block_bodies.append(body)
            self.block_joints.append(builder.joint_count - 1)

        self.gripper_bodies = [
            body
            for body in self.franka_bodies
            if "hand" in builder.body_label[body] or "finger" in builder.body_label[body]
        ]
        # MuJoCo turns ke/kd into a soft penalty; a raw solref stiffens only its grasp contacts.
        solref = builder.custom_attributes["mujoco:solref"]
        solref_mode = builder.custom_attributes["mujoco:solref_mode"]
        for attribute in (solref, solref_mode):
            if attribute.values is None:
                attribute.values = {}
        for shape, body in enumerate(builder.shape_body):
            if body in (*self.gripper_bodies, *self.block_bodies):
                solref.values[shape] = wp.vec2(0.004, 1.0)
                solref_mode.values[shape] = SOLREF_MODE_RAW
            if body in self.gripper_bodies:
                builder.shape_material_ke[shape] = CLOTH_KE
                builder.shape_material_kd[shape] = CLOTH_KD
                builder.shape_material_mu[shape] = CLOTH_MU

        stage = Usd.Stage.Open(newton.examples.get_asset("unisex_shirt.usd"))
        shirt = newton.usd.get_mesh(stage.GetPrimAtPath("/root/shirt"))
        self.particle_start = builder.particle_count
        builder.add_cloth_mesh(
            vertices=[wp.vec3(v) for v in shirt.vertices],
            indices=shirt.indices,
            rot=wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), np.pi),
            pos=wp.vec3(0.0, 0.7, 0.3),
            vel=wp.vec3(0.0),
            density=0.2,
            scale=0.01,
            tri_ke=1.0e3,
            tri_ka=1.0e3,
            tri_kd=1.0e-3,
            edge_ke=1.0e-3,
            edge_kd=1.0e-4,
            particle_radius=CLOTH_RADIUS,
        )
        self.particles_per_world = builder.particle_count - self.particle_start
        self.finger_shapes = [shape for shape, body in enumerate(builder.shape_body) if body in self.gripper_bodies]

    def _build_scene(self):
        template = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        # A tight speculative gap keeps the arm's resting contacts with the tables out of the solvers' row budgets.
        template.rigid_gap = 0.005
        SolverMuJoCo.register_custom_attributes(template)
        self._emit_template(template)
        bodies_per_world = template.body_count
        joints_per_world = template.joint_count

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.rigid_gap = template.rigid_gap
        builder.replicate(template, world_count=self.world_count)
        builder.add_ground_plane()
        # Tables are global like the ground so every entry's view keeps them.
        table_cfg = newton.ModelBuilder.ShapeConfig(ke=GRASP_KE, kd=GRASP_KD, mu=1.0)
        for center, half, label in ((TABLE_CENTER, TABLE_HALF, "table"), (PEDESTAL_CENTER, PEDESTAL_HALF, "pedestal")):
            builder.add_shape_box(
                -1,
                xform=wp.transform(center, wp.quat_identity()),
                hx=half[0],
                hy=half[1],
                hz=half[2],
                cfg=table_cfg,
                label=label,
            )
        builder.color()
        self.model = builder.finalize()
        self.device = self.model.device
        self.model.soft_contact_ke = CLOTH_SELF_KE
        self.model.soft_contact_kd = CLOTH_SELF_KD
        self.model.soft_contact_mu = 0.25
        self.model.edge_rest_angle.zero_()

        def expand(ids, stride):
            return [world * stride + i for world in range(self.world_count) for i in ids]

        self.franka_bodies = expand(self.franka_bodies, bodies_per_world)
        self.franka_joints = expand(self.franka_joints, joints_per_world)
        self.gripper_bodies = expand(self.gripper_bodies, bodies_per_world)
        self.block_bodies = expand(self.block_bodies, bodies_per_world)
        self.block_joints = expand(self.block_joints, joints_per_world)
        shapes_per_world = template.shape_count
        self.finger_shape_ids = wp.array(expand(self.finger_shapes, shapes_per_world), dtype=int, device=self.device)
        self.finger_mu = None

    # ------------------------------------------------------------------
    # Solver
    # ------------------------------------------------------------------
    def _build_solver(self, args):
        if self.rigid_solver == "featherpgs":
            rigid_name = "fpgs"

            def rigid_solver(view):
                return SolverFeatherPGS(
                    view,
                    pgs_mode="matrix_free",
                    pgs_iterations=24,
                    enable_joint_limits=True,
                    dense_max_constraints=256,
                )

        else:
            rigid_name = "mjc"

            def rigid_solver(view):
                return SolverMuJoCo(
                    model=view,
                    solver="newton",
                    integrator="implicitfast",
                    cone="elliptic",
                    iterations=100,
                    ls_iterations=50,
                    use_mujoco_contacts=False,
                    njmax=max(512, 128 * self.world_count),
                    nconmax=max(256, 64 * self.world_count),
                )

        proxy_bodies = self.gripper_bodies + self.block_bodies
        vbd_iterations = int(args.vbd_iterations)
        self.solver = SolverCoupledProxy(
            model=self.model,
            entries=[
                SolverCoupled.Entry(
                    name=rigid_name,
                    solver=rigid_solver,
                    bodies=self.franka_bodies + self.block_bodies,
                    joints=self.franka_joints + self.block_joints,
                ),
                SolverCoupled.Entry(
                    name="vbd",
                    solver=lambda view: SolverVBD(
                        model=view,
                        iterations=vbd_iterations,
                        rigid_compliant_alm=True,
                        # Low slip threshold keeps the stack from creeping on the cloth.
                        friction_epsilon=1.0e-3,
                        particle_enable_self_contact=True,
                        particle_self_contact_margin=0.002,
                        particle_self_contact_gap=0.0,
                        particle_topological_contact_filter_threshold=1,
                        particle_rest_shape_contact_exclusion_radius=0.005,
                        particle_vertex_contact_buffer_size=16,
                        particle_edge_contact_buffer_size=20,
                    ),
                    particles=list(range(self.model.particle_count)),
                ),
            ],
            coupling=SolverCoupledProxy.Config(
                proxies=[
                    SolverCoupledProxy.Proxy(
                        source=rigid_name,
                        destination="vbd",
                        bodies=proxy_bodies,
                        mode="lagged",
                        collision_pipeline=lambda view: newton.examples.create_collision_pipeline(
                            view, broad_phase="explicit", soft_contact_gap=CLOTH_CONTACT_GAP
                        ),
                        collide_interval=1,
                    )
                ],
                iterations=int(args.proxy_iterations),
            ),
        )

    # ------------------------------------------------------------------
    # IK and keyframes
    # ------------------------------------------------------------------
    def _build_ik(self):
        ik_builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        self._add_franka(ik_builder)
        self.ik_model = ik_builder.finalize(device=self.device)
        self.n_coords = self.ik_model.joint_coord_count
        coords_per_world = self.model.joint_coord_count // self.world_count
        self.ik_joint_q = wp.clone(self.model.joint_q.reshape((self.world_count, -1))[:, : self.n_coords])
        self.control_joint_target_q = self.control.joint_target_q.reshape((self.world_count, coords_per_world))
        self.finger_idx0 = self.n_coords - 2
        self.finger_idx1 = self.n_coords - 1
        self.finger_pos_buf = wp.full(self.world_count, GRIP_OPEN, dtype=float, device=self.device)
        hand_body = next(i for i, label in enumerate(self.ik_model.body_label) if label.endswith("fr3_hand"))

        self._build_keyframes()
        self.ik_target_positions = wp.array(self.targets[:, 0, :3], dtype=wp.vec3, device=self.device)
        self.ik_target_rotations = wp.array(self.targets[:, 0, 3:7], dtype=wp.vec4, device=self.device)
        pos_obj = ik.IKObjectivePosition(
            link_index=hand_body,
            link_offset=wp.vec3(0.0, 0.0, 0.107),
            target_positions=self.ik_target_positions,
        )
        rot_obj = ik.IKObjectiveRotation(
            link_index=hand_body,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=self.ik_target_rotations,
        )
        lower = wp.clone(self.model.joint_limit_lower.reshape((self.world_count, -1))[:, : self.n_coords])
        upper = wp.clone(self.model.joint_limit_upper.reshape((self.world_count, -1))[:, : self.n_coords])
        limit_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=lower.flatten(), joint_limit_upper=upper.flatten(), weight=10.0
        )
        self.ik_solver = ik.IKSolver(
            model=self.ik_model,
            n_problems=self.world_count,
            objectives=[pos_obj, rot_obj, limit_obj],
            lambda_initial=0.05,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )
        self.ik_iters = 24

    def _build_keyframes(self):
        down = HAND_DOWN
        pedestal_top = PEDESTAL_CENTER[2] + PEDESTAL_HALF[2]
        block_grasp_z = pedestal_top + BLOCK_HALF
        keys = [[1.5, -0.1, -0.5, 0.45, *down, GRIP_OPEN]]
        keys += FOLD_KEYS
        self.fold_end_time = float(np.sum([k[0] for k in keys]))
        # Stack site and heights are placeholders until the fold ends, when each world measures its own.
        self.stack_sites = np.tile(np.asarray(STACK_SEARCH_CENTER), (self.world_count, 1))
        self.stack_rows, self.stack_levels = [], []
        for level, x in enumerate(BLOCK_XS):
            place_z = TABLE_TOP + 0.03 + (2 * level + 1) * BLOCK_HALF
            start = len(keys)
            keys += [
                [1.5, x, PEDESTAL_CENTER[1], CARRY_Z, *down, GRIP_OPEN],
                [1.0, x, PEDESTAL_CENTER[1], block_grasp_z, *down, GRIP_OPEN],
                [0.75, x, PEDESTAL_CENTER[1], block_grasp_z, *down, GRIP_BLOCK],
                [1.0, x, PEDESTAL_CENTER[1], CARRY_Z, *down, GRIP_BLOCK],
                [1.5, *STACK_SEARCH_CENTER, CARRY_Z, *down, GRIP_BLOCK],
                [1.5, *STACK_SEARCH_CENTER, place_z, *down, GRIP_BLOCK],
                [0.5, *STACK_SEARCH_CENTER, place_z, *down, GRIP_BLOCK],
                [0.75, *STACK_SEARCH_CENTER, place_z, *down, GRIP_OPEN],
                [1.5, *STACK_SEARCH_CENTER, CARRY_Z, *down, GRIP_OPEN],
            ]
            self.stack_rows += list(range(start + 4, start + 9))
            self.stack_levels += [None, level, level, level, None]
            if level == 1:
                # The upper block is aimed at wherever the lower block came to rest.
                self.upper_aim_time = float(np.sum([k[0] for k in keys[: start + 4]]))
                self.upper_rows = list(range(start + 4, start + 9))
        self.stack_settled_time = float(np.sum([k[0] for k in keys]))
        keys.append([1.0, *STACK_SEARCH_CENTER, 0.5, *down, GRIP_OPEN])
        self.stack_rows.append(len(keys) - 1)
        self.stack_levels.append(None)
        poses = np.asarray(keys, dtype=np.float32)
        self.targets = np.tile(poses[None, :, 1:], (self.world_count, 1, 1))
        self.key_times = np.cumsum(poses[:, 0])
        self.stack_done_time = float(self.key_times[-1])
        self.fold_measured = False
        self.upper_aimed = False
        self.settled_blocks = None

    def _choose_stack_sites(self):
        """Pick the flattest block-sized patch of each folded shirt and rewrite that world's stack keyframes."""
        particle_q = self.state_0.particle_q.numpy().reshape(self.world_count, -1, 3)
        half = BLOCK_HALF + 0.005
        offsets = np.arange(-STACK_SEARCH_RADIUS, STACK_SEARCH_RADIUS + 1.0e-6, 0.01)
        for world in range(self.world_count):
            cloth = particle_q[world]
            best = None
            for dx in offsets:
                for dy in offsets:
                    center = np.asarray(STACK_SEARCH_CENTER) + np.asarray((dx, dy))
                    under = np.all(np.abs(cloth[:, :2] - center) < half, axis=1)
                    if under.sum() < 20:
                        continue
                    heights = cloth[under, 2]
                    score = heights.max() - heights.min() + 0.2 * np.hypot(dx, dy)
                    if best is None or score < best[0]:
                        best = (score, center, heights.max() + CLOTH_RADIUS)
            if best is None:
                best = (0.0, np.asarray(STACK_SEARCH_CENTER), TABLE_TOP + CLOTH_RADIUS)
            _, center, surface = best
            self.stack_sites[world] = center
            for row, level in zip(self.stack_rows, self.stack_levels, strict=True):
                self.targets[world, row, :2] = center
                if level is not None:
                    self.targets[world, row, 2] = surface + (2 * level + 1) * BLOCK_HALF + 0.002

    def update_ik_targets(self):
        if not self.fold_measured and self.sim_time >= self.fold_end_time:
            self._choose_stack_sites()
            self.fold_measured = True
        if not self.upper_aimed and self.sim_time >= self.upper_aim_time:
            lower = self.state_0.body_q.numpy()[self.block_bodies].reshape(self.world_count, -1, 7)[:, 0]
            for world in range(self.world_count):
                top = lower[world, 2] + BLOCK_HALF
                for row in self.upper_rows:
                    self.targets[world, row, :2] = lower[world, :2]
                for row in self.upper_rows[1:4]:
                    self.targets[world, row, 2] = top + BLOCK_HALF + 0.002
            self.upper_aimed = True
        if self.settled_blocks is None and self.sim_time >= self.stack_settled_time:
            self.settled_blocks = self.state_0.body_q.numpy()[self.block_bodies].copy()
        t = min(self.sim_time, float(self.key_times[-1]) - 1e-6)
        interval = int(np.searchsorted(self.key_times, t))
        t_start = self.key_times[interval - 1] if interval > 0 else 0.0
        alpha = float(np.clip((t - t_start) / max(self.key_times[interval] - t_start, 1e-6), 0.0, 1.0))
        cur = self.targets[:, interval]
        prev = self.targets[:, interval - 1] if interval > 0 else cur
        interp = (1.0 - alpha) * prev + alpha * cur
        rot = interp[:, 3:7] / np.linalg.norm(interp[:, 3:7], axis=1, keepdims=True)
        # Fingers grip cloth only while pinching it during the fold.
        pinching = self.sim_time < self.fold_end_time and float(interp[0, 7]) < 0.5 * GRIP_OPEN
        finger_mu = CLOTH_MU if pinching else FINGER_RELEASE_MU
        if finger_mu != self.finger_mu:
            wp.launch(
                set_shape_mu,
                dim=self.finger_shape_ids.shape[0],
                inputs=[self.finger_shape_ids, finger_mu],
                outputs=[self.model.shape_material_mu],
                device=self.device,
            )
            self.finger_mu = finger_mu
        self.ik_target_positions.assign(interp[:, :3])
        self.ik_target_rotations.assign(rot)
        self.finger_pos_buf.assign(interp[:, 7])

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    def capture(self):
        self.graph = None
        if self.use_graph:
            with wp.ScopedDevice(self.device), wp.ScopedCapture() as capture:
                self.simulate()
            if capture.graph is None:
                raise RuntimeError(f"Graph capture failed on device {self.device}")
            self.graph = capture.graph

    def simulate(self):
        self.ik_solver.step(self.ik_joint_q, self.ik_joint_q, iterations=self.ik_iters)
        wp.launch(
            set_gripper_q,
            dim=self.world_count,
            inputs=[self.ik_joint_q, self.finger_pos_buf, self.finger_idx0, self.finger_idx1],
            device=self.device,
        )
        wp.copy(dest=self.control_joint_target_q[:, : self.n_coords], src=self.ik_joint_q)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            newton.examples.apply_coupled_viewer_forces(self, self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.flat_extent is None and self.sim_time >= 1.5:
            self.flat_extent = [(r["xy_area"], r["footprint_area"]) for r in self.shirt_report()]
        self.update_ik_targets()
        if self.graph is not None:
            with wp.ScopedDevice(self.device):
                wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        newton.examples.log_coupled_view(self, self.contacts)
        self.viewer.end_frame()

    def shirt_report(self) -> list[dict]:
        """Return per-world shirt footprint, block poses and cloth clearance under the lower block."""
        particle_q = self.state_0.particle_q.numpy().reshape(self.world_count, -1, 3)
        body_q = self.state_0.body_q.numpy()
        blocks = body_q[self.block_bodies].reshape(self.world_count, -1, 7)
        report = []
        for world in range(self.world_count):
            cloth = particle_q[world]
            low, high = cloth.min(axis=0), cloth.max(axis=0)
            cells = np.unique(np.floor(cloth[:, :2] / FOOTPRINT_CELL).astype(np.int64), axis=0)
            lower = min(blocks[world], key=lambda pose: pose[2])
            # Measure in the block frame, from the face nearest the bottom: a block resting on a crease tilts.
            qv, qw = -lower[3:6], lower[6]
            offset = np.vstack([cloth - lower[:3], (0.0, 0.0, 1.0)])
            t = 2.0 * np.cross(qv, offset)
            local = offset + qw * t + np.cross(qv, t)
            local, up = local[:-1], local[-1]
            axis = int(np.argmax(np.abs(up)))
            height = np.sign(up[axis]) * local[:, axis]
            # The inner footprint excludes cloth folded up against the block's sides.
            under = np.all(np.abs(np.delete(local, axis, axis=1)) < 0.7 * BLOCK_HALF, axis=1)
            clearance = float(-BLOCK_HALF - (height[under].max() + CLOTH_RADIUS)) if under.any() else None
            report.append(
                {
                    "xy_area": float((high[0] - low[0]) * (high[1] - low[1])),
                    "footprint_area": float(cells.shape[0] * FOOTPRINT_CELL**2),
                    "blocks": blocks[world].copy(),
                    "cloth_clearance_under_lower_block": clearance,
                }
            )
        return report

    def task_success(self) -> list[dict]:
        """Return per-world fold and stack outcomes once the stack has had time to settle."""
        settled = self.settled_blocks.reshape(self.world_count, -1, 7)
        outcomes = []
        for world, entry in enumerate(self.shirt_report()):
            low, high = sorted(entry["blocks"], key=lambda pose: pose[2])
            clearance = entry["cloth_clearance_under_lower_block"]
            checks = {
                "folded": entry["footprint_area"] / self.flat_extent[world][1] < FOLD_FOOTPRINT_RATIO,
                "lower_on_site": np.linalg.norm(low[:2] - self.stack_sites[world]) < 0.05,
                "upper_on_lower": np.linalg.norm(high[:2] - low[:2]) < 0.025
                and abs(high[2] - low[2] - 2.0 * BLOCK_HALF) < 0.006,
                "lower_on_cloth": low[2] - BLOCK_HALF > TABLE_TOP + 0.003 and clearance is not None,
                "stable": np.abs(entry["blocks"][:, :3] - settled[world][:, :3]).max() < STACK_DRIFT_LIMIT,
            }
            outcomes.append(checks)
        return outcomes

    def test_final(self):
        if self.use_graph:
            assert self.graph is not None, "Graph capture was requested but no graph was captured"
        assert np.all(np.isfinite(self.state_0.body_q.numpy())), "Body state is not finite"
        assert np.all(np.isfinite(self.state_0.particle_q.numpy())), "Cloth state is not finite"
        for world, entry in enumerate(self.shirt_report()):
            # Blocks stay near the table: nothing is launched by the coupled contacts.
            assert np.all(np.abs(entry["blocks"][:, :2] - (0.0, -0.45)) < 1.0), f"World {world} block left the scene"
            clearance = entry["cloth_clearance_under_lower_block"]
            assert clearance is None or clearance > -0.003, f"World {world} cloth penetrates the lower block"
        # The scripted fold and stack are not deterministic, so task_success() reports them instead.

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        newton.examples.add_coupled_view_args(parser)
        newton.examples.add_world_count_arg(parser)
        parser.add_argument(
            "--rigid-solver",
            type=str,
            choices=["mujoco", "featherpgs"],
            default="mujoco",
            help="Solver that owns the Franka arm and the blocks.",
        )
        parser.add_argument("--substeps", type=int, default=20, help="Coupled substeps per rendered frame.")
        parser.add_argument("--vbd-iterations", type=int, default=5, help="VBD iterations per coupled substep.")
        parser.add_argument("--proxy-iterations", type=int, default=1, help="Proxy relaxation passes per substep.")
        parser.add_argument(
            "--no-graph-capture",
            action="store_false",
            dest="graph_capture",
            default=True,
            help="Disable graph capture.",
        )
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    parser.set_defaults(num_frames=4500)
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
