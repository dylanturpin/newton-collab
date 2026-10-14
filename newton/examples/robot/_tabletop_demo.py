# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Interactive, force-driven FPGS tabletop demonstrations."""

import numpy as np
import warp as wp

import newton
import newton.examples
import newton.utils
from newton.solvers import SolverFeatherPGS

from ._tabletop_common import box
from ._tabletop_robot import make_arm


@wp.kernel
def command_path(
    path: wp.array2d[float],
    time: float,
    q: wp.array[float],
    integral: wp.array[float],
    dt: float,
    integral_count: int,
    target: wp.array[float],
):
    i = wp.tid()
    frame = wp.clamp(time * 60.0, 0.0, float(path.shape[0] - 1))
    a = int(frame)
    b = wp.min(a + 1, path.shape[0] - 1)
    value = wp.lerp(path[a, i], path[b, i], frame - float(a))
    if i < integral_count:
        error = wp.clamp(value - q[i], -0.08, 0.08)
        integral[i] = wp.clamp(integral[i] + 3.0 * dt * error, -0.12, 0.12)
        value += integral[i]
    target[i] = value


class TabletopDemo:
    """Share simulation plumbing without coupling the three scene builders."""

    task = ""

    def __init__(self, viewer, args):
        newton.use_coord_layout_targets = True
        if not wp.get_device().is_cuda:
            raise RuntimeError("These FPGS matrix-free examples require a CUDA device")
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = args.substeps
        if args.substeps < 1 or args.iterations < 1:
            raise ValueError("Substeps and iterations must be positive")
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = self.phase = 0.0
        self.autoplay = True
        self.test_mode = args.test
        self.info = {}
        b = newton.ModelBuilder()
        b.default_shape_cfg.mu = 0.75
        b.default_shape_cfg.gap = 0.001
        b.default_shape_cfg.restitution = 0.02
        b.default_shape_cfg.density = 600
        assets = {
            "franka_emika_panda": str(
                newton.utils.download_asset("franka_emika_panda", ref="f8fb7abcbeba2318814a74f3eeb02780ad7925d6")
            )
        }
        robot = "franka" if self.task == "plates" else "ur10"
        if robot == "ur10":
            assets["ur10_menagerie"] = str(
                newton.examples.download_external_git_folder(
                    "https://github.com/google-deepmind/mujoco_menagerie.git",
                    "universal_robots_ur10e",
                    ref="f054586a8e90465d49ee5be15335c4a0c7f57caf",
                )
            )
        ee, down, fingers = make_arm(
            b,
            assets,
            wp.get_device(),
            robot=robot,
            root=(-0.30, 0, 0) if robot == "franka" else (-0.42, 0, 0),
            gripper=self.task != "sweep",
        )
        self.populate(b, self.info, ee, down, fingers)
        self.driven_count = b._tabletop_driven_dofs
        self.integral_count = self.info.get("integral_drive_count", 0)
        waypoints = self.info.pop("waypoints")
        times = np.array([p[0] for p in waypoints])
        values = np.array([p[1] for p in waypoints])
        timeline = np.arange(901) / 60
        targets = np.column_stack([np.interp(timeline, times, values[:, i]) for i in range(self.driven_count)])
        self.targets = wp.array(targets, dtype=float)
        self.integral = wp.zeros(self.driven_count, dtype=float)
        box(b, (0.05, 0, -0.035), (0.95, 0.86, 0.035), (0.52, 0.61, 0.67), label="table")
        for x in (-0.72, 0.82):
            for y in (-0.68, 0.68):
                box(b, (x, y, -0.39), (0.025, 0.025, 0.32), (0.16, 0.20, 0.23), label="table_leg")
        b.add_ground_plane(height=-0.75)
        self.model = b.finalize()
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.initial = self.model.state()
        self.initial.assign(self.state_0)
        self.pipeline = newton.CollisionPipeline(
            self.model,
            broad_phase="sap",
            rigid_contact_max=16384,
            reduce_contacts=newton.CollisionPipeline.ContactReductionConfig(
                body_pairs=True, body_pair_cell_size=0.06, body_pair_hysteresis=0.0
            ),
            contact_matching="disabled",
        )
        self.contacts = self.pipeline.contacts()
        self.solver = SolverFeatherPGS(
            self.model,
            solve_mode="matrix_free",
            iterations=args.iterations,
            enable_joint_limits=True,
            enable_joint_velocity_limits=True,
            friction_anchor_beta=0.0,
            articulation_constraint_max=512,
            free_body_constraint_max=4096,
            row_watermark=True,
            use_parallel_streams=False,
        )
        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(1.65, -2.1, 1.6), pitch=-28, yaw=128)
        self.graph = None
        self.peak_height = float(self.initial.body_q.numpy()[self.info["tracked_body"], 2])
        if self.model.device.is_cuda:
            self.simulate()
            self.reset()
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
            self.reset()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=900)
        parser.add_argument("--substeps", type=int, default=4, help="Physics substeps per 60 Hz frame")
        parser.add_argument("--iterations", type=int, default=12, help="FPGS sweeps per substep")
        return parser

    def reset(self):
        """Restore the initial physical state and clear controller/solver history."""
        self.state_0.assign(self.initial)
        self.state_1.assign(self.initial)
        self.control.joint_target_q.assign(self.model.joint_target_q)
        self.integral.zero_()
        self.solver.reset(self.state_0)
        self.sim_time = self.phase = 0.0
        self.peak_height = float(self.initial.body_q.numpy()[self.info["tracked_body"], 2])

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        if self.sim_substeps % 2:
            # Keep captured input/output addresses stable for odd substep counts.
            self.state_1.assign(self.state_0)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        wp.launch(
            command_path,
            dim=self.driven_count,
            inputs=[
                self.targets,
                self.phase,
                self.state_0.joint_q,
                self.integral,
                self.frame_dt,
                self.integral_count,
                self.control.joint_target_q,
            ],
        )
        if self.graph is None:
            self.simulate()
        else:
            wp.capture_launch(self.graph)
        if self.autoplay:
            self.phase = min(15.0, self.phase + self.frame_dt)
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.contacts, self.state_0)
        self.viewer.end_frame()

    def gui(self, imgui):
        imgui.text("Drag bodies to apply forces. Targets drive only the robot.")
        _, self.autoplay = imgui.checkbox("Run demonstration", self.autoplay)
        changed, phase = imgui.slider_float("Robot path [s]", self.phase, 0.0, 15.0)
        if changed:
            self.autoplay = False
            self.phase = phase
            self.integral.zero_()
        if imgui.button("Reset scene"):
            self.reset()
            self.autoplay = True

    def test_post_step(self):
        """Reject nonfinite states and track the physical object lift."""
        poses = self.state_0.body_q.numpy()
        assert np.isfinite(poses).all(), "Nonfinite rigid body pose"
        assert np.isfinite(self.state_0.joint_qd.numpy()).all(), "Nonfinite joint velocity"
        self.peak_height = max(self.peak_height, float(poses[self.info["tracked_body"], 2]))

    def check_capacity(self):
        """Reject solver rows lost to insufficient contact capacity."""
        stats = self.solver.constraint_row_watermarks()
        assert all(v == 0 for k, v in stats.items() if "overflow" in k or "dropped" in k), stats
