# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example Franka Cloth Block (proxy coupling)
#
# A rigid block rests on a VBD cloth hammock pinned along two edges, as in
# the cloth Franka example but at meter scale and with a block between the
# robot and the cloth. A fixed-base Franka, simulated by MuJoCo or
# FeatherPGS, picks the block up off the sagging cloth with GPU IK and
# places it on a table beside the hammock. SolverCoupledProxy exposes the
# block and the gripper to VBD as proxies: the cloth carries the block's
# weight and sags under it, and springs back once the robot lifts the
# block off.
#
# Command: python -m newton.examples franka_cloth_block
#          python -m newton.examples franka_cloth_block --rigid-solver featherpgs
#
###########################################################################

from __future__ import annotations

import numpy as np
import warp as wp
from newton.solvers.experimental.coupled import SolverCoupled, SolverCoupledProxy

import newton
import newton.examples
import newton.ik as ik
import newton.utils
from newton.solvers import SolverFeatherPGS, SolverMuJoCo, SolverVBD

# Initial Franka joint configuration (7 arm + 2 finger).
FRANKA_Q = [0.0, -0.3, 0.0, -2.2, 0.0, 1.9, 0.785, 0.04, 0.04]

# Hammock: pinned along its two x edges, centered under the block.
CLOTH_CENTER = (0.5, 0.0, 0.25)
CLOTH_CELLS = 20
CLOTH_CELL = 0.02
CLOTH_PARTICLE_MASS = 0.0005

BLOCK_HALF = 0.025
BLOCK_MASS = 0.25
TABLE_CENTER = (0.5, 0.33, 0.12)
TABLE_HALF = (0.15, 0.08, 0.12)

# Top-down gripper orientation: 180 deg about world x flips the hand z-axis to -z.
GRIPPER_DOWN = (1.0, 0.0, 0.0, 0.0)
GRIP_OPEN = 0.04
GRIP_CLOSE = 0.0
CONTACT_KE = 2.0e3
CONTACT_KD = 1.0e-1
GRASP_KE = 1.0e4
SOLREF_MODE_RAW = 1
GRASP_KD = 1.0e1


@wp.kernel
def set_gripper_q(joint_q: wp.array2d[float], finger_pos: wp.array[float], idx0: int, idx1: int):
    world_idx = wp.tid()
    joint_q[world_idx, idx0] = finger_pos[world_idx]
    joint_q[world_idx, idx1] = finger_pos[world_idx]


@wp.kernel
def set_task_targets(
    target_positions: wp.array[wp.vec3],
    target_rotations: wp.array[wp.vec4],
    finger_pos: wp.array[float],
    pos: wp.vec3,
    rot: wp.vec4,
    grip_width: float,
):
    world_idx = wp.tid()
    target_positions[world_idx] = pos
    target_rotations[world_idx] = rot
    finger_pos[world_idx] = grip_width


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
        self.collision_pipeline = newton.CollisionPipeline(self.model, broad_phase="explicit")
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

        newton.examples.configure_coupled_view(self, args)
        if self.world_count > 1:
            self.viewer.set_world_offsets((1.2, 1.2, 0.0))
        if isinstance(self.viewer, newton.viewer.ViewerGL):
            self.viewer.set_camera(pos=wp.vec3(1.25, -0.6, 0.65), pitch=-20.0, yaw=150.0)
            self.viewer.camera.look_at(wp.vec3(0.45, 0.1, 0.25))

        self.capture()

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------
    @staticmethod
    def _add_franka(builder):
        builder.add_urdf(
            newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
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
        builder.joint_target_ke[:7] = [400.0] * 7
        builder.joint_target_kd[:7] = [80.0] * 7
        builder.joint_target_ke[7:9] = [4000.0, 4000.0]
        builder.joint_target_kd[7:9] = [100.0, 100.0]
        builder.joint_effort_limit[:4] = [87.0] * 4
        builder.joint_effort_limit[4:7] = [12.0] * 3
        builder.joint_effort_limit[7:9] = [200.0, 200.0]
        builder.joint_armature[:7] = [1.0e-3] * 7
        self.franka_bodies = list(range(franka_body_start, builder.body_count))
        self.franka_joints = list(range(franka_joint_start, builder.joint_count))

        # Gravity compensation on the arm so the PD targets track without sag.
        gravcomp = builder.custom_attributes["mujoco:gravcomp"]
        if gravcomp.values is None:
            gravcomp.values = {}
        for body in self.franka_bodies:
            gravcomp.values[body] = 1.0
            if self.rigid_solver == "featherpgs":
                builder.body_disable_gravity[body] = True

        cx, cy, cz = CLOTH_CENTER
        block_cfg = newton.ModelBuilder.ShapeConfig(
            density=BLOCK_MASS / (2.0 * BLOCK_HALF) ** 3, ke=GRASP_KE, kd=GRASP_KD, mu=1.0, gap=0.005
        )
        self.block_body = builder.add_body(
            xform=wp.transform((cx, cy, cz + BLOCK_HALF + 0.01), wp.quat_identity()), label="block"
        )
        builder.add_shape_box(self.block_body, hx=BLOCK_HALF, hy=BLOCK_HALF, hz=BLOCK_HALF, cfg=block_cfg)
        self.block_joint = builder.joint_count - 1

        half = 0.5 * CLOTH_CELLS * CLOTH_CELL
        self.particle_start = builder.particle_count
        builder.add_cloth_grid(
            pos=wp.vec3(cx - half, cy - half, cz),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=CLOTH_CELLS,
            dim_y=CLOTH_CELLS,
            cell_x=CLOTH_CELL,
            cell_y=CLOTH_CELL,
            mass=CLOTH_PARTICLE_MASS,
            fix_left=True,
            fix_right=True,
            tri_ke=1.0e3,
            tri_ka=1.0e3,
            tri_kd=1.0,
            edge_ke=1.0e-2,
            edge_kd=1.0e-4,
            particle_radius=0.005,
        )
        self.particles_per_world = builder.particle_count - self.particle_start

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
            if body in (*self.gripper_bodies, self.block_body):
                solref.values[shape] = wp.vec2(0.004, 1.0)
                solref_mode.values[shape] = SOLREF_MODE_RAW
            if body in self.gripper_bodies:
                builder.shape_material_ke[shape] = GRASP_KE
                builder.shape_material_kd[shape] = GRASP_KD
                builder.shape_material_mu[shape] = 1.0

    def _build_scene(self):
        template = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        SolverMuJoCo.register_custom_attributes(template)
        self._emit_template(template)
        bodies_per_world = template.body_count
        joints_per_world = template.joint_count

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.replicate(template, world_count=self.world_count)
        builder.add_ground_plane()
        # The table is global like the ground so every entry's view keeps it.
        builder.add_shape_box(
            -1,
            xform=wp.transform(TABLE_CENTER, wp.quat_identity()),
            hx=TABLE_HALF[0],
            hy=TABLE_HALF[1],
            hz=TABLE_HALF[2],
            cfg=newton.ModelBuilder.ShapeConfig(ke=GRASP_KE, kd=GRASP_KD, mu=1.0),
            label="table",
        )
        builder.color()
        self.model = builder.finalize()
        self.device = self.model.device
        self.model.soft_contact_ke = CONTACT_KE
        self.model.soft_contact_kd = CONTACT_KD
        self.model.soft_contact_mu = 1.0

        def expand(ids, stride):
            return [world * stride + i for world in range(self.world_count) for i in ids]

        self.bodies_per_world = bodies_per_world
        self.franka_bodies = expand(self.franka_bodies, bodies_per_world)
        self.franka_joints = expand(self.franka_joints, joints_per_world)
        self.gripper_bodies = expand(self.gripper_bodies, bodies_per_world)
        self.block_bodies = expand([self.block_body], bodies_per_world)
        self.block_joints = expand([self.block_joint], joints_per_world)

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
                    impratio=10.0,
                    iterations=100,
                    ls_iterations=50,
                    use_mujoco_contacts=False,
                    njmax=max(256, 64 * self.world_count),
                    nconmax=max(256, 64 * self.world_count),
                )

        self.rigid_name = rigid_name
        proxy_bodies = self.gripper_bodies + self.block_bodies
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
                        model=view, iterations=int(args.vbd_iterations), rigid_compliant_alm=True
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
                        mode=args.coupling_mode,
                        collision_pipeline=lambda view: newton.examples.create_collision_pipeline(
                            view, broad_phase="explicit"
                        ),
                        collide_interval=1,
                    )
                ],
                iterations=int(args.proxy_iterations),
            ),
        )
        self.block_proxy_rows = [proxy_bodies.index(b) for b in self.block_bodies]

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

        self._build_keyframes(CLOTH_CENTER[:2], grasp_z=CLOTH_CENTER[2] + BLOCK_HALF)
        self.grasp_measured = False
        self.ik_target_positions = wp.array(
            [wp.vec3(*self.targets[0][:3].tolist())] * self.world_count, dtype=wp.vec3, device=self.device
        )
        self.ik_target_rotations = wp.array(
            [wp.vec4(*self.targets[0][3:7].tolist())] * self.world_count, dtype=wp.vec4, device=self.device
        )
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

    def _build_keyframes(self, pick_xy, grasp_z):
        above = CLOTH_CENTER[2] + 0.2
        lift = grasp_z + 0.12
        # Release just above the table so the fingertips clear its top.
        place_z = TABLE_CENTER[2] + TABLE_HALF[2] + BLOCK_HALF + 0.012
        qx, qy, qz, qw = GRIPPER_DOWN
        px, py = float(pick_xy[0]), float(pick_xy[1])
        self.place_xy = (TABLE_CENTER[0], TABLE_CENTER[1])
        tx, ty = self.place_xy
        # [duration, px, py, pz, qx, qy, qz, qw, finger_width]; the block settles on the cloth first.
        poses = np.array(
            [
                [1.0, px, py, above, qx, qy, qz, qw, GRIP_OPEN],
                [0.5, px, py, above, qx, qy, qz, qw, GRIP_OPEN],
                [0.75, px, py, grasp_z, qx, qy, qz, qw, GRIP_OPEN],
                [0.75, px, py, grasp_z, qx, qy, qz, qw, GRIP_CLOSE],
                [1.0, px, py, lift, qx, qy, qz, qw, GRIP_CLOSE],
                [1.5, tx, ty, lift, qx, qy, qz, qw, GRIP_CLOSE],
                [0.75, tx, ty, place_z, qx, qy, qz, qw, GRIP_CLOSE],
                [0.75, tx, ty, place_z, qx, qy, qz, qw, GRIP_OPEN],
                [1.0, tx, ty, place_z + 0.15, qx, qy, qz, qw, GRIP_OPEN],
            ],
            dtype=np.float32,
        )
        self.targets = poses[:, 1:]
        self.key_times = np.cumsum(poses[:, 0])
        self.settle_time = float(self.key_times[0])
        self.lift_time = float(self.key_times[4])

    def update_ik_targets(self):
        if not self.grasp_measured and self.sim_time >= self.settle_time - 1.0e-6:
            # Grasp where the block came to rest on the sagging cloth.
            block = self.state_0.body_q.numpy()[self.block_bodies[0], :3]
            self.loaded_cloth_min_z = [entry["cloth_min_z"] for entry in self.block_cloth_report()]
            # Grip the upper half so the fingertips stay clear of the cloth.
            self._build_keyframes(block[:2], grasp_z=float(block[2]) + 0.2 * BLOCK_HALF)
            self.grasp_measured = True
        t = min(self.sim_time, float(self.key_times[-1]) - 1e-6)
        interval = int(np.searchsorted(self.key_times, t))
        t_start = self.key_times[interval - 1] if interval > 0 else 0.0
        alpha = float(np.clip((t - t_start) / max(self.key_times[interval] - t_start, 1e-6), 0.0, 1.0))
        cur = self.targets[interval]
        prev = self.targets[interval - 1] if interval > 0 else cur
        interp = (1.0 - alpha) * prev + alpha * cur
        wp.launch(
            set_task_targets,
            dim=self.world_count,
            inputs=[
                self.ik_target_positions,
                self.ik_target_rotations,
                self.finger_pos_buf,
                wp.vec3(*interp[:3].tolist()),
                wp.vec4(*interp[3:7].tolist()),
                float(interp[-1]),
            ],
            device=self.device,
        )

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

    def block_cloth_report(self) -> list[dict]:
        """Return per-world block position, cloth low point and block-to-cloth gap."""
        body_q = self.state_0.body_q.numpy()
        particle_q = self.state_0.particle_q.numpy().reshape(self.world_count, -1, 3)
        report = []
        for world, block in enumerate(self.block_bodies):
            position = body_q[block, :3]
            cloth = particle_q[world]
            under = np.linalg.norm(cloth[:, :2] - position[:2], axis=1) < BLOCK_HALF
            report.append(
                {
                    "block": position.copy(),
                    "cloth_min_z": float(cloth[:, 2].min()),
                    "gap_under_block": float(position[2] - BLOCK_HALF - cloth[under, 2].max()) if under.any() else None,
                }
            )
        return report

    def test_final(self):
        if self.use_graph:
            assert self.graph is not None, "Graph capture was requested but no graph was captured"
        assert np.all(np.isfinite(self.state_0.body_q.numpy())), "Body state is not finite"
        assert np.all(np.isfinite(self.state_0.particle_q.numpy())), "Cloth state is not finite"
        target = np.asarray(self.place_xy)
        table_top = TABLE_CENTER[2] + TABLE_HALF[2]
        for world, entry in enumerate(self.block_cloth_report()):
            error = float(np.linalg.norm(entry["block"][:2] - target))
            assert error < 0.02, f"World {world} block placed {error * 100:.1f} cm from target"
            height = float(entry["block"][2] - BLOCK_HALF - table_top)
            assert abs(height) < 0.005, f"World {world} block is {height * 1000:.1f} mm off the table top"
            # Unloaded, the pinned cloth springs back from the sag the block caused.
            rebound = entry["cloth_min_z"] - self.loaded_cloth_min_z[world]
            assert rebound > 0.01, f"World {world} cloth rose only {rebound * 1000:.1f} mm after the block was lifted"

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
            help="Solver that owns the Franka arm and the block.",
        )
        parser.add_argument("--substeps", type=int, default=10, help="Coupled substeps per rendered frame.")
        parser.add_argument("--proxy-iterations", type=int, default=1, help="Proxy relaxation passes per substep.")
        parser.add_argument(
            "--coupling-mode", type=str, choices=["lagged", "staggered"], default="lagged", help="Proxy transfer mode."
        )
        parser.add_argument("--vbd-iterations", type=int, default=10, help="VBD iterations per coupled substep.")
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
    parser.set_defaults(num_frames=510)
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
