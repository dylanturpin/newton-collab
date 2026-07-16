# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example IK Waiter (dynamics-aware trajectory IK with simulation in the loop)
#
# A Franka FR3 carries a plate with a free ball resting on it. Trajectory
# IK plans the carry toward a draggable goal with IKObjectiveApparentGravity,
# which keeps the apparent gravity felt by the plate aligned with its
# normal: the solver banks the plate into every acceleration like a waiter
# with a loaded tray. The IK output drives a PD-controlled MuJoCo forward
# simulation (gravity-compensated actuators, velocity feedforward), so the
# ball's fate is decided by contact physics, not by the objective.
# - Drag the gizmo to move the carry goal; the trajectory re-solves and the
#   arm dashes over while the plate banks
# - Toggle "waiter objective" in the UI to compare with a level-plate
#   orientation objective: fast dashes then throw the ball off
#
# Command: python -m newton.examples ik_waiter
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
import newton.utils

HAND = 10  # fr3_hand
PLATE_LOCAL = wp.vec3(0.0, 0.0, 0.118)  # plate center in the hand frame
PLATE_HALF = 0.13
PLATE_THICK = 0.004
BALL_RADIUS = 0.03
ARM_COORDS = 9
NEUTRAL_Q = [0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854]
CARRY_POS = wp.vec3(0.45, 0.0, 0.45)


def _quat_rotate_np(q, v):
    xyz = q[:3]
    uv = 2.0 * np.cross(xyz, v)
    return v + q[3] * uv + np.cross(xyz, uv)


class Example:
    def __init__(self, viewer, args):
        self.fps = 30
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.viewer = viewer

        self.n_frames = 76  # ~2.5 s planning horizon
        # the stencil cannot cover the last frames' orientation, and smoothness
        # propagates that boundary freedom ~10 frames back as a slow tilt
        # drift; keep a padded tail that is solved but never executed
        self.play_end = self.n_frames - 16
        self.control_substeps = 4  # control rate 120 Hz
        self.sim_substeps = 12  # physics 1440 Hz
        self.waiter_enabled = True
        self._replan_needed = True

        # ------------------------------------------------------------------
        # models: IK sees the arm + plate; the simulation adds the free ball
        # ------------------------------------------------------------------
        self.ik_model = self._build_arm().finalize()

        sim_builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(sim_builder)
        arm = self._build_arm()
        arm.joint_target_ke = [1000.0] * 7 + [200.0, 200.0]
        arm.joint_target_kd = [80.0] * 7 + [20.0, 20.0]
        arm.joint_effort_limit = [87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0, 100.0, 100.0]
        arm.joint_target_mode = [int(newton.JointTargetMode.POSITION_VELOCITY)] * 9
        arm.joint_armature = [0.3] * 4 + [0.11] * 3 + [0.15] * 2
        sim_builder.add_builder(arm, xform=wp.transform_identity())
        # gravity-compensated actuators (like a real FR3): the arm holds pose
        # without PD droop while the ball keeps its gravity
        sim_builder.custom_attributes["mujoco:jnt_actgravcomp"].values = dict.fromkeys(range(7), True)
        sim_builder.custom_attributes["mujoco:gravcomp"].values = dict.fromkeys(range(1, 14), 1.0)
        self.ball_body = sim_builder.add_body(
            xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()), label="ball"
        )
        ball_shape = sim_builder.add_shape_sphere(
            self.ball_body,
            radius=BALL_RADIUS,
            cfg=newton.ModelBuilder.ShapeConfig(density=600.0, mu=0.5, mu_rolling=5e-4, mu_torsional=2e-4),
        )
        # condim 6 activates rolling/torsional friction so residual ball drift
        # decays instead of persisting forever (a real ball has rolling
        # resistance; contact condim is the max over the geom pair)
        sim_builder.custom_attributes["mujoco:condim"].values = {ball_shape: 6}
        sim_builder.add_ground_plane()
        self.model = sim_builder.finalize()

        self.solver = newton.solvers.SolverMuJoCo(
            self.model,
            solver="newton",
            integrator="implicitfast",
            cone="elliptic",
            impratio=100.0,
            iterations=20,
            nconmax=256,
        )
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.contacts = self.model.contacts()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(1.9, -1.3, 1.1), pitch=-18.0, yaw=145.0)

        # ------------------------------------------------------------------
        # carry home: gripper-up "waiter" pose found by per-frame IK
        # ------------------------------------------------------------------
        self.home_q = self._find_carry_home()

        # ------------------------------------------------------------------
        # trajectory solver; both plate objectives stay in the residual
        # layout, the UI toggle just swaps their weights
        # ------------------------------------------------------------------
        n = self.n_frames
        self.targets_wp = wp.zeros(n, dtype=wp.vec3)
        rot_targets = wp.array(np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)).astype(np.float32), dtype=wp.vec4)
        reference_q = wp.array(np.tile(self.home_q, (n, 1)).astype(np.float32), dtype=wp.float32)
        self.rot_obj = ik.IKObjectiveRotation(HAND, wp.quat_identity(), rot_targets, weight=0.02)
        self.waiter_obj = ik.IKObjectiveApparentGravity(
            self.ik_model, HAND, PLATE_LOCAL, plate_axis=wp.vec3(0.0, 0.0, 1.0), dt=self.frame_dt, weight=0.2
        )
        objectives = [
            ik.IKObjectivePosition(HAND, PLATE_LOCAL, self.targets_wp, weight=2.0),
            ik.IKObjectiveJointLimit(self.ik_model.joint_limit_lower, self.ik_model.joint_limit_upper, weight=10.0),
            ik.IKObjectiveJointReference(self.ik_model, reference_q, weight=1e-3),
            ik.IKObjectiveSmoothness(self.ik_model, derivative=1, dt=self.frame_dt, weight=0.01),
            ik.IKObjectiveSmoothness(self.ik_model, derivative=2, dt=self.frame_dt, weight=0.0005),
            self.rot_obj,
            self.waiter_obj,
        ]
        self.traj_solver = ik.IKSolverTrajectory(
            self.ik_model,
            n,
            objectives,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
            linear_solver="direct",
            fixed_frames=[0],
            lambda_initial=0.1,
        )
        self.plan_q = wp.array(np.tile(self.home_q, (n, 1)).astype(np.float32), dtype=wp.float32)
        self._apply_mode()

        # goal gizmo + scripted goal sequence (used until the user grabs it)
        self.goal_tf = wp.transform(CARRY_POS, wp.quat_identity())
        self._last_goal = np.array([*CARRY_POS])
        self._user_moved_goal = False
        self._script = [(1.0, [0.45, 0.42, 0.45]), (4.0, [0.55, 0.05, 0.55]), (7.0, [0.45, -0.25, 0.45])]

        # fk scratch for planning
        self._ik_state = self.ik_model.state()

        # start the simulation at the home pose with the ball on the plate
        q0 = np.zeros(self.model.joint_coord_count, dtype=np.float32)
        q0[:ARM_COORDS] = self.home_q
        self._ik_state.joint_q.assign(self.home_q)
        newton.eval_fk(self.ik_model, self._ik_state.joint_q, self.ik_model.joint_qd, self._ik_state)
        plate_center = self._plate_center(self._ik_state.body_q.numpy()[HAND])
        q0[ARM_COORDS : ARM_COORDS + 3] = plate_center + np.array([0.0, 0.0, PLATE_THICK + BALL_RADIUS + 1e-3])
        q0[ARM_COORDS + 6] = 1.0
        self.state_0.joint_q.assign(q0)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)
        # joint targets use the DoF layout; the FR3's arm coords equal its DoFs
        self._target_q_np = np.zeros(self.model.joint_dof_count, dtype=np.float32)
        self._target_qd_np = np.zeros(self.model.joint_dof_count, dtype=np.float32)
        self._target_q_np[:ARM_COORDS] = q0[:ARM_COORDS]
        self.control.joint_target_q.assign(self._target_q_np)

        self.play_frame = 0
        self._replan()

    # ----------------------------------------------------------------------
    def _build_arm(self) -> newton.ModelBuilder:
        builder = newton.ModelBuilder()
        builder.add_urdf(
            newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
            floating=False,
        )
        builder.joint_q[:7] = NEUTRAL_Q
        builder.add_shape_box(
            HAND,
            xform=wp.transform(PLATE_LOCAL, wp.quat_identity()),
            hx=PLATE_HALF,
            hy=PLATE_HALF,
            hz=PLATE_THICK,
            cfg=newton.ModelBuilder.ShapeConfig(density=300.0, mu=0.8),
        )
        return builder

    def _plate_center(self, hand_tf: np.ndarray) -> np.ndarray:
        return hand_tf[:3] + _quat_rotate_np(hand_tf[3:7], np.array([PLATE_LOCAL[0], PLATE_LOCAL[1], PLATE_LOCAL[2]]))

    def _find_carry_home(self) -> np.ndarray:
        pos_obj = ik.IKObjectivePosition(
            HAND, PLATE_LOCAL, wp.array(np.array([[*CARRY_POS]], dtype=np.float32), dtype=wp.vec3), weight=1.0
        )
        rot_obj = ik.IKObjectiveRotation(
            HAND,
            wp.quat_identity(),
            wp.array(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), dtype=wp.vec4),
            weight=1.0,
        )
        lim_obj = ik.IKObjectiveJointLimit(
            self.ik_model.joint_limit_lower, self.ik_model.joint_limit_upper, weight=10.0
        )
        solver = ik.IKSolver(self.ik_model, 1, [pos_obj, rot_obj, lim_obj], jacobian_mode=ik.IKJacobianType.ANALYTIC)
        seed = np.zeros((1, self.ik_model.joint_coord_count), dtype=np.float32)
        seed[0, :7] = NEUTRAL_Q
        seed[0, 5] = 3.0  # bend the wrist toward gripper-up so LM starts in the right basin
        joint_q = wp.array(seed, dtype=wp.float32)
        solver.step(joint_q, joint_q, iterations=100)
        return joint_q.numpy()[0]

    def _apply_mode(self):
        # weights are read at launch time, so toggling needs no recompilation
        if self.waiter_enabled:
            self.waiter_obj.weight = 0.2
            self.rot_obj.weight = 0.02
        else:
            self.waiter_obj.weight = 0.0
            self.rot_obj.weight = 0.5

    def _replan(self):
        """Plan a smooth carry from the current plan frame to the goal."""
        n = self.n_frames
        q_now = self.plan_q.numpy()[min(self.play_frame, self.play_end)]
        self._ik_state.joint_q.assign(q_now)
        newton.eval_fk(self.ik_model, self._ik_state.joint_q, self.ik_model.joint_qd, self._ik_state)
        p0 = self._plate_center(self._ik_state.body_q.numpy()[HAND])
        goal = np.array([*wp.transform_get_translation(self.goal_tf)])
        # min-jerk-style ease over 3/4 of the horizon, then hold
        s = np.clip(np.linspace(0.0, 4.0 / 3.0, n), 0.0, 1.0)
        s = s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)
        targets = p0[None, :] + s[:, None] * (goal - p0)[None, :]
        self.targets_wp.assign(targets.astype(np.float32))

        seed = np.tile(q_now, (n, 1)).astype(np.float32)
        self.plan_q.assign(seed)
        self.traj_solver.step(self.plan_q, self.plan_q, iterations=96)
        self._plan_np = self.plan_q.numpy()
        self._plan_qd = np.zeros_like(self._plan_np)
        self._plan_qd[1:] = (self._plan_np[1:] - self._plan_np[:-1]) / self.frame_dt
        self.play_frame = 0

    # ----------------------------------------------------------------------
    def gui(self, imgui):
        changed, value = imgui.checkbox("waiter objective", self.waiter_enabled)
        if changed:
            self.waiter_enabled = value
            self._apply_mode()
            self._replan_needed = True
        if imgui.button("reset ball"):
            self._reset_ball()

    def _reset_ball(self):
        q = self.state_0.joint_q.numpy().copy()
        self._ik_state.joint_q.assign(q[:ARM_COORDS])
        newton.eval_fk(self.ik_model, self._ik_state.joint_q, self.ik_model.joint_qd, self._ik_state)
        plate_center = self._plate_center(self._ik_state.body_q.numpy()[HAND])
        q[ARM_COORDS : ARM_COORDS + 3] = plate_center + np.array([0.0, 0.0, PLATE_THICK + BALL_RADIUS + 5e-3])
        q[ARM_COORDS + 3 : ARM_COORDS + 7] = [0.0, 0.0, 0.0, 1.0]
        self.state_0.joint_q.assign(q)
        qd = self.state_0.joint_qd.numpy()
        qd[ARM_COORDS - 2 : ARM_COORDS + 4] = 0.0
        self.state_0.joint_qd.assign(qd)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

    def step(self):
        # scripted goal motion until the user grabs the gizmo
        if not self._user_moved_goal and self._script and self.sim_time >= self._script[0][0]:
            _, goal = self._script.pop(0)
            self.goal_tf = wp.transform(wp.vec3(*goal), wp.transform_get_rotation(self.goal_tf))
            self._last_goal = np.array(goal)
            self._replan_needed = True

        goal_now = np.array([*wp.transform_get_translation(self.goal_tf)])
        if np.linalg.norm(goal_now - self._last_goal) > 5e-3:
            self._user_moved_goal = True
            self._last_goal = goal_now
            self._replan_needed = True
        if self._replan_needed:
            self._replan_needed = False
            self._replan()

        # follow the current plan through the PD-controlled simulation
        f = min(self.play_frame, self.play_end)
        f_next = min(f + 1, self.play_end)
        sim_dt = self.frame_dt / (self.control_substeps * self.sim_substeps)
        for s in range(self.control_substeps):
            alpha = (s + 1) / self.control_substeps
            self._target_q_np[:ARM_COORDS] = (1 - alpha) * self._plan_np[f] + alpha * self._plan_np[f_next]
            self._target_qd_np[:ARM_COORDS] = self._plan_qd[f_next]
            self.control.joint_target_q.assign(self._target_q_np)
            self.control.joint_target_qd.assign(self._target_qd_np)
            for _ in range(self.sim_substeps):
                self.state_0.clear_forces()
                self.model.collide(self.state_0, self.contacts)
                self.solver.step(self.state_0, self.state_1, self.control, self.contacts, sim_dt)
                self.state_0, self.state_1 = self.state_1, self.state_0

        self.play_frame = min(self.play_frame + 1, self.play_end)
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_gizmo("carry_goal", self.goal_tf, rotate=())
        self.viewer.end_frame()

    def ball_offset(self) -> float:
        """Ball distance from the plate center, in the plate plane [m]."""
        bq = self.state_0.body_q.numpy()
        hand_tf = bq[HAND]
        d = bq[self.ball_body][:3] - self._plate_center(hand_tf)
        q = hand_tf[3:7]
        conj = np.array([-q[0], -q[1], -q[2], q[3]])
        local = _quat_rotate_np(conj, d)
        return float(np.linalg.norm(local[:2]))

    def test_post_step(self):
        if not np.isfinite(self.state_0.joint_q.numpy()).all():
            raise ValueError("simulation state is not finite")

    def test_final(self):
        # the scripted dashes must finish with the ball still on the plate
        if self.ball_offset() > PLATE_HALF:
            raise ValueError(f"ball left the plate (offset {self.ball_offset():.3f} m)")
        bq = self.state_0.body_q.numpy()
        if bq[self.ball_body][2] < 0.2:
            raise ValueError("ball fell to the ground")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.set_defaults(num_frames=300)
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
