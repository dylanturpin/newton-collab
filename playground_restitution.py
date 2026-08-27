# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Restitution Playground (FeatherPGS)
#
# Drops a row of balls, one per restitution coefficient, so different
# bounce parameters can be compared side by side in one view. The scene
# resets on a fixed cycle and prints the measured rebound ratio of each
# ball against its authored coefficient after every cycle.
#
# Run:            python playground_restitution.py
# Coefficients:   python playground_restitution.py --restitution 0.1,0.5,0.9
# Threshold:      python playground_restitution.py --threshold 1.0
# Solver mode:    python playground_restitution.py --mode split
# Drop height:    python playground_restitution.py --drop-height 2.0
#
# The console prints which lane (front to back) carries which coefficient.
###########################################################################

import numpy as np
import warp as wp

import newton
import newton.examples

RADIUS = 0.25


class Example:
    def __init__(self, viewer, args):
        self.fps = 120
        self.frame_dt = 1.0 / self.fps
        # The captured graph ping-pongs state_0/state_1, so the substep
        # count must be even for each replay to hand its result to the next.
        self.sim_substeps = max(int(args.substeps), 2)
        if self.sim_substeps % 2:
            self.sim_substeps += 1
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.viewer = viewer

        self.coefficients = [float(v) for v in args.restitution.split(",")]
        self.drop_height = float(args.drop_height)
        self.reset_frames = max(int(args.reset_seconds * self.fps), 1)
        self.frame_count = 0
        self.cycle = 0

        pad_hz = 0.05
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -args.gravity))
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=args.mu, restitution=0.0))
        self.bodies = []
        for i, e in enumerate(self.coefficients):
            # Contacts average the two shapes' coefficients, so each lane gets
            # its own static pad carrying the same value as its ball -- the
            # effective pair coefficient is then exactly the authored one.
            builder.add_shape_box(
                -1,
                xform=wp.transform(wp.vec3(0.0, i * 1.0, pad_hz), wp.quat_identity()),
                hx=0.45,
                hy=0.45,
                hz=pad_hz,
                cfg=newton.ModelBuilder.ShapeConfig(mu=args.mu, restitution=e),
                label=f"pad_e{e:g}",
            )
            body = builder.add_body(
                xform=wp.transform(
                    wp.vec3(0.0, i * 1.0, 2.0 * pad_hz + RADIUS + self.drop_height), wp.quat_identity()
                ),
                mass=1.0,
                label=f"ball_e{e:g}",
            )
            builder.add_shape_sphere(
                body,
                radius=RADIUS,
                cfg=newton.ModelBuilder.ShapeConfig(density=0.0, mu=args.mu, restitution=e),
            )
            self.bodies.append(body)

        self.model = builder.finalize()
        self.solver = newton.solvers.SolverFeatherPGS(
            self.model,
            pgs_mode=args.mode,
            enable_contact_friction=args.mu > 0.0,
            restitution_velocity_threshold=args.threshold,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()

        # Rebound bookkeeping: fastest approach and fastest rebound per cycle.
        self.min_vz = np.zeros(len(self.bodies))
        self.max_vz = np.zeros(len(self.bodies))

        self.viewer.set_model(self.model)
        mid_y = 0.5 * (len(self.coefficients) - 1)
        self.viewer.set_camera(
            pos=wp.vec3(4.0 + 0.6 * len(self.coefficients), mid_y, 1.5),
            pitch=-10.0,
            yaw=-180.0,
        )

        print(f"[playground] mode={args.mode} threshold={args.threshold} m/s "
              f"drop={self.drop_height} m dt=1/{self.fps}/{self.sim_substeps}")
        for i, e in enumerate(self.coefficients):
            print(f"[playground] lane {i} (y={i * 1.0:.0f}): restitution={e:g}")

        self.capture()

    def capture(self):
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def _report_and_reset(self):
        self.cycle += 1
        print(f"[playground] cycle {self.cycle}: measured rebound / impact speed")
        for i, e in enumerate(self.coefficients):
            down = -self.min_vz[i]
            up = self.max_vz[i]
            measured = up / down if down > 1.0e-6 else 0.0
            print(f"  lane {i}: e={e:g}  impact={down:5.2f} m/s  rebound={up:5.2f} m/s  "
                  f"measured e={measured:.3f}")
        self.min_vz[:] = 0.0
        self.max_vz[:] = 0.0

        joint_q = self.model.joint_q.numpy().copy()
        joint_qd = np.zeros(self.model.joint_dof_count, dtype=np.float32)
        self.state_0.joint_q.assign(joint_q)
        self.state_0.joint_qd.assign(joint_qd)
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

    def step(self):
        if self.graph:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        self.frame_count += 1

        vz = self.state_0.body_qd.numpy()[self.bodies, 2]
        self.min_vz = np.minimum(self.min_vz, vz)
        self.max_vz = np.maximum(self.max_vz, vz)

        if self.frame_count % self.reset_frames == 0:
            self._report_and_reset()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--restitution",
        type=str,
        default="0.0,0.2,0.4,0.6,0.8,0.95",
        help="Comma-separated ball restitution coefficients, one lane each.",
    )
    parser.add_argument("--threshold", type=float, default=0.5, help="restitution_velocity_threshold [m/s].")
    parser.add_argument("--drop-height", type=float, default=1.5, help="Initial gap under each ball [m].")
    parser.add_argument("--gravity", type=float, default=9.81, help="Gravity magnitude [m/s^2].")
    parser.add_argument("--mu", type=float, default=0.0, help="Friction coefficient (0 = frictionless).")
    parser.add_argument(
        "--mode", type=str, default="matrix_free", choices=["matrix_free", "split", "dense"], help="FPGS pgs_mode."
    )
    parser.add_argument("--substeps", type=int, default=2, help="Physics substeps per frame (rounded up to even).")
    parser.add_argument("--reset-seconds", type=float, default=6.0, help="Seconds between automatic scene resets.")

    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
