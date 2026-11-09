# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Example Steel Magnetic Door

A cabinet door on a hinge has a magnet strip that faces a strip on the cabinet frame,
after MuJoCo's ``adhesion/fridge_door.xml``. Both strips have ``steel:adhesion`` and a
3 mm contact gap, so their contacts pull within the gap. The door starts ajar and
snaps shut. A light tug on the handle does not open it; a hard tug pops it open, and
the hinge spring swings it back until the magnets latch it again.

Success criteria:
- The ajar door closes on its own within half a second.
- The light tug leaves the door latched.
- The hard tug opens the door by more than 20 degrees.
- The hinge spring brings the door back and the magnets hold it shut at the end.

Runs:
- `python -m newton.examples steel_magnetic_door`
"""

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.solvers import SolverSteel

MAGNET_ADHESION = 1.5  # [N] per contact point and strip
MAGNET_GAP = 0.003  # [m]
AJAR_ANGLE = 0.006  # [rad]
HANDLE_LEVER = 0.46  # [m] from the hinge
# (start time [s], duration [s], handle pull [N]); the light tug stays below the latch's pull-off.
TUGS = ((1.0, 0.3, 4.0), (2.0, 0.15, 25.0))
LATCHED_ANGLE = 0.002  # [rad]


@wp.kernel(enable_backward=False)
def apply_hinge_torque(time: wp.array[wp.float32], dof: int, joint_f: wp.array[wp.float32]):
    """Write the handle tugs of the current time as an opening hinge torque."""
    t = time[0]
    torque = float(0.0)
    if t >= wp.static(TUGS[0][0]) and t < wp.static(TUGS[0][0] + TUGS[0][1]):
        torque = wp.static(TUGS[0][2] * HANDLE_LEVER)
    if t >= wp.static(TUGS[1][0]) and t < wp.static(TUGS[1][0] + TUGS[1][1]):
        torque = wp.static(TUGS[1][2] * HANDLE_LEVER)
    joint_f[dof] = torque


@wp.kernel(enable_backward=False)
def advance_time(dt: float, time: wp.array[wp.float32]):
    time[0] = time[0] + dt


class Example:
    def __init__(self, viewer, args):
        """Build the cabinet, the hinged door with its magnet strips and the Steel solver."""
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 8
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0

        builder = newton.ModelBuilder()
        SolverSteel.register_custom_attributes(builder)
        builder.add_ground_plane()
        body_cfg = newton.ModelBuilder.ShapeConfig(mu=0.5)
        magnet_cfg = newton.ModelBuilder.ShapeConfig(mu=0.5, gap=MAGNET_GAP)
        magnet = {"steel:adhesion": MAGNET_ADHESION}
        white = (0.92, 0.92, 0.94)
        dark = (0.2, 0.2, 0.22)
        # Cabinet interior x [-0.35, 0], y [-0.25, 0.25], z [0, 0.8].
        for pos, half in (
            ((-0.175, -0.235, 0.4), (0.175, 0.015, 0.4)),
            ((-0.175, 0.235, 0.4), (0.175, 0.015, 0.4)),
            ((-0.175, 0.0, 0.785), (0.175, 0.25, 0.015)),
            ((-0.175, 0.0, 0.03), (0.175, 0.25, 0.015)),
            ((-0.335, 0.0, 0.4), (0.015, 0.25, 0.4)),
        ):
            builder.add_shape_box(
                -1,
                xform=wp.transform(pos, wp.quat_identity()),
                hx=half[0],
                hy=half[1],
                hz=half[2],
                cfg=body_cfg,
                color=white,
            )
        builder.add_shape_box(
            -1,
            xform=wp.transform((-0.015, 0.19, 0.4), wp.quat_identity()),
            hx=0.005,
            hy=0.01,
            hz=0.3,
            cfg=magnet_cfg,
            color=dark,
            custom_attributes=magnet,
        )

        hinge = wp.transform((0.0, -0.25, 0.42), wp.quat_identity())
        door = builder.add_link(xform=hinge, label="door")
        builder.add_shape_box(
            door,
            xform=wp.transform((0.015, 0.25, 0.0), wp.quat_identity()),
            hx=0.015,
            hy=0.24,
            hz=0.37,
            cfg=newton.ModelBuilder.ShapeConfig(mu=0.5, density=1.5 / (0.03 * 0.48 * 0.74)),
            color=white,
        )
        builder.add_shape_capsule(
            door,
            xform=wp.transform((0.06, HANDLE_LEVER, 0.0), wp.quat_identity()),
            radius=0.012,
            half_height=0.12,
            cfg=newton.ModelBuilder.ShapeConfig(has_shape_collision=False, density=500.0),
            color=(0.4, 0.4, 0.42),
        )
        builder.add_shape_box(
            door,
            xform=wp.transform((-0.005, 0.44, 0.0), wp.quat_identity()),
            hx=0.005,
            hy=0.01,
            hz=0.3,
            cfg=magnet_cfg,
            color=dark,
            custom_attributes=magnet,
        )
        joint = builder.add_joint_revolute(
            -1,
            door,
            parent_xform=hinge,
            axis=(0.0, 0.0, -1.0),
            limit_lower=0.0,
            limit_upper=math.radians(135.0),
            damping=0.08,
            stiffness=0.05,
        )
        builder.add_articulation([joint])
        builder.joint_q[builder.joint_q_start[joint]] = AJAR_ANGLE
        self.door_dof = builder.joint_qd_start[joint]
        self.model = builder.finalize()

        self.solver = SolverSteel(self.model, enable_joint_limits=True, iterations=16, articulation_constraint_max=256)
        self.collision_pipeline = newton.CollisionPipeline(self.model)
        self.contacts = self.collision_pipeline.contacts()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.time = wp.zeros(1, dtype=wp.float32)

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(1.1, 0.8, 0.9), pitch=-19.0, yaw=-146.0)

        self.angles = []
        self.capture()

    def capture(self):
        """Capture one frame of substeps in a CUDA graph when the device supports it."""
        self.graph = None
        if wp.get_device().is_cuda:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        """Advance one rendered frame of substeps, pulling the handle during the tugs."""
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            wp.launch(apply_hinge_torque, dim=1, inputs=[self.time, self.door_dof], outputs=[self.control.joint_f])
            self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            wp.launch(advance_time, dim=1, inputs=[self.sim_dt], outputs=[self.time])

    def step(self):
        """Advance the door by one rendered frame."""
        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_post_step(self):
        """Record the door angle of every frame."""
        angle = float(self.state_0.joint_q.numpy()[0])
        if not math.isfinite(angle):
            raise AssertionError(f"The door angle is not finite at t={self.sim_time:.3f} s.")
        self.angles.append((self.sim_time, angle))

    def test_final(self):
        """Check that the door closes, survives the light tug, opens on the hard tug and latches again."""
        times, angles = (np.asarray(v) for v in zip(*self.angles, strict=True))
        if times[-1] < 6.9:
            raise AssertionError("Run at least 420 frames.")
        light_start = TUGS[0][0]
        hard_start = TUGS[1][0]
        closed = angles[(times >= 0.5) & (times < light_start)]
        if np.any(closed > LATCHED_ANGLE):
            raise AssertionError(f"The ajar door did not close: max angle {closed.max():.4f} rad.")
        light = angles[(times >= light_start) & (times < hard_start)]
        if np.any(light > LATCHED_ANGLE):
            raise AssertionError(f"The light tug opened the door to {light.max():.4f} rad.")
        hard = angles[times >= hard_start]
        if hard.max() < math.radians(20.0):
            raise AssertionError(f"The hard tug opened the door only to {math.degrees(hard.max()):.1f} degrees.")
        if np.any(angles[times >= times[-1] - 0.5] > LATCHED_ANGLE):
            raise AssertionError(f"The door is not latched at the end: angle {angles[-1]:.4f} rad.")


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
