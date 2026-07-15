# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example IK Trajectory (whole-trajectory optimization)
#
# Trajectory inverse kinematics on a Franka FR3: all frames of a
# pick-and-place end-effector path are optimized jointly with
# IKSolverTrajectory, using temporal smoothness objectives to couple
# consecutive frames.
# - The viewer loops the solved trajectory while a gizmo moves the pick
#   waypoint; the whole trajectory re-solves every frame at interactive
#   rates (warm-started, CUDA-graph captured)
# - At startup the same targets are also solved frame by frame with the
#   per-frame IKSolver and a comparison table (solve time, velocity,
#   acceleration, jerk) is printed to the console
#
# Command: python -m newton.examples ik_trajectory
###########################################################################

import time

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.ik as ik
import newton.utils


class Example:
    def __init__(self, viewer, args):
        # frame timing (trajectory frames play back in real time)
        self.fps = 30
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0

        self.viewer = viewer

        # trajectory horizon
        self.n_frames = 90
        self.ik_iters_init = 32
        self.ik_iters = 8

        # ------------------------------------------------------------------
        # Build a single FR3 (fixed base) + ground, posed at a ready config
        # ------------------------------------------------------------------
        franka = newton.ModelBuilder()
        franka.add_urdf(
            newton.utils.download_asset("franka_emika_panda") / "urdf/fr3_franka_hand.urdf",
            floating=False,
        )
        franka.joint_q[:9] = [0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854, 0.04, 0.04]
        franka.add_ground_plane()

        self.graph = None
        self.model = franka.finalize()
        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(1.6, -1.6, 1.2), pitch=-25.0, yaw=135.0)

        # state used for rendering the current playback frame
        self.state = self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)

        # ------------------------------------------------------------------
        # Task: pick-and-place EE path through waypoints, anchored at home
        # ------------------------------------------------------------------
        self.ee_index = 10  # fr3_hand_tcp

        body_q_np = self.state.body_q.numpy()
        home_tf = wp.transform(*body_q_np[self.ee_index])
        home_pos = wp.transform_get_translation(home_tf)
        home_rot = wp.transform_get_rotation(home_tf)

        # home -> above pick -> pick -> lift -> above place -> place
        self.waypoints = np.array(
            [
                [home_pos[0], home_pos[1], home_pos[2]],
                [0.45, -0.30, 0.35],
                [0.45, -0.30, 0.15],
                [0.45, 0.00, 0.45],
                [0.45, 0.30, 0.35],
                [0.45, 0.30, 0.15],
            ],
            dtype=np.float32,
        )
        self.pick_waypoint = 2  # gizmo-controlled

        targets_np = self._interp_targets()
        self.targets_wp = wp.array(targets_np, dtype=wp.vec3)

        # persistent gizmo transform (mutated in place by the viewer)
        self.pick_tf = wp.transform(wp.vec3(*self.waypoints[self.pick_waypoint]), home_rot)

        # path visualization buffers
        self.path_starts = wp.array(targets_np[:-1], dtype=wp.vec3)
        self.path_ends = wp.array(targets_np[1:], dtype=wp.vec3)
        self.path_colors = wp.full(self.n_frames - 1, wp.vec3(0.95, 0.65, 0.15), dtype=wp.vec3)
        self.waypoint_points = wp.array(self.waypoints, dtype=wp.vec3)
        self.waypoint_radii = wp.full(len(self.waypoints), 0.02, dtype=wp.float32)
        self.waypoint_colors = wp.full(len(self.waypoints), wp.vec3(0.2, 0.5, 0.9), dtype=wp.vec3)

        # ------------------------------------------------------------------
        # Trajectory IK: per-frame tracking + temporal smoothness objectives
        # ------------------------------------------------------------------
        def _q2v4(q):
            return wp.vec4(q[0], q[1], q[2], q[3])

        # keep the home (downward-facing) hand orientation on every frame
        rot_targets = wp.array([_q2v4(home_rot)] * self.n_frames, dtype=wp.vec4)

        home_q_np = self.model.joint_q.numpy()
        reference_q = wp.array(np.tile(home_q_np, (self.n_frames, 1)), dtype=wp.float32)

        objectives = [
            ik.IKObjectivePosition(
                link_index=self.ee_index,
                link_offset=wp.vec3(0.0, 0.0, 0.0),
                target_positions=self.targets_wp,
                weight=1.0,
            ),
            ik.IKObjectiveRotation(
                link_index=self.ee_index,
                link_offset_rotation=wp.quat_identity(),
                target_rotations=rot_targets,
                weight=0.5,
            ),
            ik.IKObjectiveJointLimit(
                joint_limit_lower=self.model.joint_limit_lower,
                joint_limit_upper=self.model.joint_limit_upper,
                weight=10.0,
            ),
            ik.IKObjectiveSmoothness(self.model, derivative=1, dt=self.frame_dt, weight=0.01),
            ik.IKObjectiveSmoothness(self.model, derivative=2, dt=self.frame_dt, weight=0.0005),
            ik.IKObjectiveJointReference(self.model, reference_q, weight=1e-3),
        ]

        self.solver = ik.IKSolverTrajectory(
            self.model,
            self.n_frames,
            objectives,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
            linear_solver="direct",
            fixed_frames=[0],  # frame 0 stays at the home configuration
            lambda_initial=0.1,
        )

        # joint trajectory, frame-major [n_frames, joint_coord_count]
        seed_np = np.tile(home_q_np, (self.n_frames, 1)).astype(np.float32)
        self.joint_q = wp.array(seed_np, dtype=wp.float32)

        # warm up (module load / tile-kernel compile), then time the cold solve
        self.solver.step(self.joint_q, self.joint_q, iterations=2)
        self.joint_q.assign(seed_np)
        wp.synchronize()
        t0 = time.perf_counter()
        self.solver.step(self.joint_q, self.joint_q, iterations=self.ik_iters_init)
        wp.synchronize()
        traj_ms = 1e3 * (time.perf_counter() - t0)

        # ------------------------------------------------------------------
        # Baseline: solve the same targets frame by frame with IKSolver
        # ------------------------------------------------------------------
        fb_q_np, fb_ms = self._solve_frame_by_frame(targets_np, rot_targets, seed_np)
        self._print_comparison(fb_q_np, fb_ms, self.joint_q.numpy(), traj_ms, targets_np)

        # subsequent re-solves are warm-started from the current trajectory
        self.capture()

        self.play_frame = 0

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------
    def capture(self):
        self.graph = None
        with wp.ScopedCapture() as capture:
            self.simulate()
        self.graph = capture.graph

    def simulate(self):
        self.solver.step(self.joint_q, self.joint_q, iterations=self.ik_iters)

    def _interp_targets(self):
        """Piecewise-linear EE targets through the waypoints, one per frame."""
        knots = np.linspace(0.0, 1.0, len(self.waypoints))
        s = np.linspace(0.0, 1.0, self.n_frames)
        return np.stack([np.interp(s, knots, self.waypoints[:, k]) for k in range(3)], axis=1).astype(np.float32)

    def _push_targets_from_gizmo(self):
        """Rebuild the target path from the gizmo-controlled pick waypoint."""
        pos = wp.transform_get_translation(self.pick_tf)
        pos = wp.vec3(pos[0], pos[1], max(pos[2], 0.05))
        self.waypoints[self.pick_waypoint] = [pos[0], pos[1], pos[2]]
        # the approach waypoint hovers above the pick point
        self.waypoints[self.pick_waypoint - 1] = [pos[0], pos[1], pos[2] + 0.2]

        targets_np = self._interp_targets()
        wp.copy(self.targets_wp, wp.array(targets_np, dtype=wp.vec3, device="cpu"))
        wp.copy(self.path_starts, wp.array(targets_np[:-1], dtype=wp.vec3, device="cpu"))
        wp.copy(self.path_ends, wp.array(targets_np[1:], dtype=wp.vec3, device="cpu"))
        wp.copy(self.waypoint_points, wp.array(self.waypoints, dtype=wp.vec3, device="cpu"))

    def _ee_positions(self, q_np):
        """World EE positions for each frame of a joint trajectory."""
        positions = np.zeros((self.n_frames, 3), dtype=np.float32)
        q_row = wp.zeros(self.model.joint_coord_count, dtype=wp.float32)
        for t in range(self.n_frames):
            q_row.assign(q_np[t])
            newton.eval_fk(self.model, q_row, self.model.joint_qd, self.state)
            positions[t] = self.state.body_q.numpy()[self.ee_index][:3]
        return positions

    def _solve_frame_by_frame(self, targets_np, rot_targets, seed_np):
        """Solve each frame independently (warm-started) with the per-frame IKSolver."""
        pos_obj = ik.IKObjectivePosition(
            link_index=self.ee_index,
            link_offset=wp.vec3(0.0, 0.0, 0.0),
            target_positions=wp.array(targets_np[:1], dtype=wp.vec3),
            weight=1.0,
        )
        rot_obj = ik.IKObjectiveRotation(
            link_index=self.ee_index,
            link_offset_rotation=wp.quat_identity(),
            target_rotations=wp.array(rot_targets.numpy()[:1], dtype=wp.vec4),
            weight=0.5,
        )
        lim_obj = ik.IKObjectiveJointLimit(
            joint_limit_lower=self.model.joint_limit_lower,
            joint_limit_upper=self.model.joint_limit_upper,
            weight=10.0,
        )
        solver = ik.IKSolver(
            model=self.model,
            n_problems=1,
            objectives=[pos_obj, rot_obj, lim_obj],
            lambda_initial=0.1,
            jacobian_mode=ik.IKJacobianType.ANALYTIC,
        )

        joint_q = wp.array(seed_np[:1], dtype=wp.float32)
        solver.step(joint_q, joint_q, iterations=2)  # warm up compile
        joint_q.assign(seed_np[:1])

        out = np.empty_like(seed_np)
        wp.synchronize()
        t0 = time.perf_counter()
        for t in range(self.n_frames):
            pos_obj.set_target_position(0, wp.vec3(*targets_np[t]))
            solver.step(joint_q, joint_q, iterations=16)
            out[t] = joint_q.numpy()[0]
        wp.synchronize()
        elapsed_ms = 1e3 * (time.perf_counter() - t0)
        return out, elapsed_ms

    def _print_comparison(self, fb_q_np, fb_ms, traj_q_np, traj_ms, targets_np):
        def metrics(q_np):
            vel = np.diff(q_np, axis=0) / self.frame_dt
            acc = np.diff(vel, axis=0) / self.frame_dt
            jerk = np.diff(acc, axis=0) / self.frame_dt
            return np.abs(vel).max(), np.abs(acc).max(), np.abs(jerk).mean()

        print(f"Trajectory IK vs per-frame IK ({self.n_frames} frames @ {self.fps} fps):")
        print("  method          solve time   ee err max   max |vel|   max |acc|   mean |jerk|")
        for name, q_np, ms in (
            ("per-frame IK", fb_q_np, fb_ms),
            ("trajectory IK", traj_q_np, traj_ms),
        ):
            err = np.linalg.norm(self._ee_positions(q_np) - targets_np, axis=1).max()
            vmax, amax, jmean = metrics(q_np)
            print(f"  {name:14s}  {ms:7.1f} ms   {err:7.4f} m   {vmax:8.2f}   {amax:9.1f}   {jmean:10.1f}")

    # ----------------------------------------------------------------------
    # Template API
    # ----------------------------------------------------------------------
    def step(self):
        self._push_targets_from_gizmo()

        # jointly re-solve the whole trajectory, warm-started
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.play_frame = (self.play_frame + 1) % self.n_frames
        self.sim_time += self.frame_dt

    def test_final(self):
        q_np = self.joint_q.numpy()
        if not np.isfinite(q_np).all():
            raise ValueError("joint trajectory contains non-finite values")

        targets_np = self._interp_targets()
        err = np.linalg.norm(self._ee_positions(q_np) - targets_np, axis=1)
        if err[1:].max() > 0.02:
            raise ValueError(f"EE tracking error too large: {err[1:].max():.4f} m")

    def render(self):
        self.viewer.begin_frame(self.sim_time)

        # visualize the current playback frame of the solved trajectory
        newton.eval_fk(self.model, self.joint_q[self.play_frame], self.model.joint_qd, self.state)
        self.viewer.log_state(self.state)

        self.viewer.log_lines("/target_path", self.path_starts, self.path_ends, self.path_colors)
        self.viewer.log_points("/waypoints", self.waypoint_points, self.waypoint_radii, self.waypoint_colors)
        self.viewer.log_gizmo("pick_waypoint", self.pick_tf, rotate=())

        self.viewer.end_frame()
        wp.synchronize()


if __name__ == "__main__":
    # Parse arguments and initialize viewer
    viewer, args = newton.examples.init()
    newton.examples.run(Example(viewer, args), args)
