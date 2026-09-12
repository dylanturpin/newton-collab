# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example FeatherPGS Crane
#
# A tower crane slews a quarter turn and back with steel tubes hanging from
# its jib on rope chains. Nothing about the load is scripted: the slew is
# the only thing driven, and what the tubes do afterwards is the pendulum
# they make with the rope, the lag as the jib takes up speed, and the way
# each one settles at a different rate for its own rope length.
#
# Command: python -m newton.examples fpgs_crane
#
###########################################################################

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.feather_pgs._showreel import Stepper, assert_finite

SOLVERS = {
    "feather_pgs": {
        "pgs_iterations": 8,
        "dense_max_constraints": 2048,
        "mf_max_constraints": 4096,
        "propagation_same_articulation_rows": True,
        "substeps": 4,
    },
    "mujoco": {"njmax": 4096, "nconmax": 2048},
}

TOWER_H = 17.0
TOWER_W = 1.5
JIB_LEN, JIB_BACK = 14.0, 5.0
JIB_H = TOWER_H + 1.2
# One rope per hook, at these radii along the jib, each this long. Ropes, jib and
# tubes are one articulation, and its dense factorization is what limits the link
# count: fine chains push past a hundred and fifty coordinates and the tiled
# Cholesky gives up, so the links are chunky, as on the wrecking ball's chain.
HOOKS = ((6.0, 7.2), (9.0, 8.4), (12.0, 6.0))
LINK_HALF, LINK_R = 0.6, 0.12
# A rope link to tube mass ratio near ten keeps the chain free of velocity spikes
# when the slew snatches the load; a thread-thin rope under a heavy tube blows up.
ROPE_DENSITY = 7800.0
TUBE_R, TUBE_HALF = 0.25, 1.5
TUBE_DENSITY = 2400.0

# The slew is the whole programme: out a quarter turn, hold, back, hold.
SLEW_ANGLE = math.radians(90.0)
# Six seconds each way, not four. A rope this long is a pendulum of about five
# seconds, so a four second slew drives it near resonance and every cycle pumps
# the load higher; by twenty seconds the tubes swing ten metres above their rest.
SLEW_OUT, SLEW_HOLD, SLEW_BACK, SLEW_REST = 6.0, 3.0, 6.0, 3.0
CYCLE_TIME = SLEW_OUT + SLEW_HOLD + SLEW_BACK + SLEW_REST

STEEL = wp.vec3(0.88, 0.74, 0.16)
DARK = wp.vec3(0.24, 0.25, 0.27)
ROPE = wp.vec3(0.35, 0.36, 0.38)
TUBE_COLORS = (wp.vec3(0.55, 0.2, 0.18), wp.vec3(0.25, 0.45, 0.6), wp.vec3(0.5, 0.5, 0.52))


def _smoothstep(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.slewing = True

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.rigid_gap = 0.004
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.9))
        steel = newton.ModelBuilder.ShapeConfig(density=1500.0, mu=0.8, restitution=0.0)
        self._build_tower(builder, newton.ModelBuilder.ShapeConfig(mu=0.9))
        joints = self._build_jib(builder, steel)
        self.tubes = []
        for index, (radius, rope_len) in enumerate(HOOKS):
            joints += self._build_rope(builder, radius, rope_len, index)
        builder.add_articulation(joints, label="crane")

        self.model = builder.finalize()
        self.model.rigid_contact_max = 4096
        self.collision_pipeline = newton.examples.create_collision_pipeline(
            self.model, args, broad_phase="sap", rigid_contact_max=self.model.rigid_contact_max
        )
        self.contacts = self.collision_pipeline.contacts()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.stepper = Stepper(self, solver_overrides=SOLVERS, solver=str(getattr(args, "solver", "feather_pgs")))
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        # FeatherPGS reads drive targets in DOF layout (indexed by joint_qd_start).
        self.target_index = self.model.joint_qd_start.numpy()
        self.targets = self.control.joint_target_q.numpy().copy()
        self.tube_start = self.state_0.body_q.numpy()[self.tubes, :3].copy()
        self.carried = 0.0

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(-26.0, -30.0, 20.0), pitch=-20.0, yaw=49.0)

    def _build_tower(self, builder, cfg):
        """Static mast and pad: the crane slews, it does not travel."""
        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.4), wp.quat_identity()),
            hx=3.2,
            hy=3.2,
            hz=0.4,
            cfg=cfg,
            color=DARK,
        )
        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(0.0, 0.0, TOWER_H / 2), wp.quat_identity()),
            hx=TOWER_W / 2,
            hy=TOWER_W / 2,
            hz=TOWER_H / 2,
            cfg=cfg,
            color=STEEL,
        )

    def _build_jib(self, builder, cfg):
        """The slewing part: jib, counter-jib and counterweight on one driven axis."""
        self.jib = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, JIB_H), wp.quat_identity()))
        builder.add_shape_box(
            self.jib,
            xform=wp.transform(wp.vec3(JIB_LEN / 2, 0.0, 0.0), wp.quat_identity()),
            hx=JIB_LEN / 2,
            hy=0.3,
            hz=0.4,
            cfg=cfg,
            color=STEEL,
        )
        builder.add_shape_box(
            self.jib,
            xform=wp.transform(wp.vec3(-JIB_BACK / 2, 0.0, 0.0), wp.quat_identity()),
            hx=JIB_BACK / 2,
            hy=0.3,
            hz=0.4,
            cfg=cfg,
            color=STEEL,
        )
        builder.add_shape_box(
            self.jib,
            xform=wp.transform(wp.vec3(-JIB_BACK, 0.0, -0.3), wp.quat_identity()),
            hx=0.9,
            hy=0.9,
            hz=0.7,
            cfg=cfg,
            color=DARK,
        )
        builder.add_shape_box(
            self.jib,
            xform=wp.transform(wp.vec3(0.0, 0.0, 1.6), wp.quat_identity()),
            hx=0.5,
            hy=0.5,
            hz=1.2,
            cfg=cfg,
            color=STEEL,
        )
        self.slew_joint = builder.add_joint_revolute(
            -1,
            self.jib,
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, JIB_H), wp.quat_identity()),
            child_xform=wp.transform_identity(),
            axis=wp.vec3(0.0, 0.0, 1.0),
            # A stiff slew snatches the load: at ten times this the ropes act like
            # springs, throw the tubes above the jib and the chain goes non-finite.
            target_ke=1.0e6,
            target_kd=3.0e5,
            effort_limit=6.0e5,
        )
        return [self.slew_joint]

    def _build_rope(self, builder, radius: float, length: float, index: int):
        """A chain of ball-jointed links from the jib down to one tube."""
        # The rope carries load, it does not need to collide: links two apart are not
        # filtered against each other, and a chain that folds during the slew reversal
        # drives those pairs into each other hard enough to go non-finite.
        rope_cfg = newton.ModelBuilder.ShapeConfig(
            density=ROPE_DENSITY, mu=0.5, restitution=0.0, has_shape_collision=False
        )
        tube_cfg = newton.ModelBuilder.ShapeConfig(density=TUBE_DENSITY, mu=0.8, restitution=0.0)
        links = max(3, round(length / (2 * LINK_HALF)))
        link_len = length / links
        joints, parent = [], self.jib
        anchor = wp.transform(wp.vec3(radius, 0.0, -0.4), wp.quat_identity())
        for k in range(links):
            z = JIB_H - 0.4 - (k + 0.5) * link_len
            link = builder.add_link(xform=wp.transform(wp.vec3(radius, 0.0, z), wp.quat_identity()))
            builder.add_shape_capsule(link, radius=LINK_R, half_height=link_len / 2 - LINK_R, cfg=rope_cfg, color=ROPE)
            top = wp.transform(wp.vec3(0.0, 0.0, link_len / 2), wp.quat_identity())
            joints.append(
                builder.add_joint_ball(
                    parent,
                    link,
                    parent_xform=anchor
                    if k == 0
                    else wp.transform(wp.vec3(0.0, 0.0, -link_len / 2), wp.quat_identity()),
                    child_xform=top,
                )
            )
            parent = link
        # The tube hangs across the rope, so the slew swings it broadside.
        tube_z = JIB_H - 0.4 - length - TUBE_R
        tube = builder.add_link(xform=wp.transform(wp.vec3(radius, 0.0, tube_z), wp.quat_identity()))
        builder.add_shape_capsule(
            tube,
            xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi / 2)),
            radius=TUBE_R,
            half_height=TUBE_HALF,
            cfg=tube_cfg,
            color=TUBE_COLORS[index % len(TUBE_COLORS)],
        )
        joints.append(
            builder.add_joint_ball(
                parent,
                tube,
                parent_xform=wp.transform(wp.vec3(0.0, 0.0, -link_len / 2), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(0.0, 0.0, TUBE_R), wp.quat_identity()),
            )
        )
        self.tubes.append(tube)
        return joints

    # ------------------------------------------------------------------- cycle
    def slew_target(self, t: float) -> float:
        """Out a quarter turn, hold, back, hold; smoothed so the load is not snatched."""
        local = t % CYCLE_TIME
        if local < SLEW_OUT:
            return SLEW_ANGLE * _smoothstep(local / SLEW_OUT)
        if local < SLEW_OUT + SLEW_HOLD:
            return SLEW_ANGLE
        if local < SLEW_OUT + SLEW_HOLD + SLEW_BACK:
            return SLEW_ANGLE * (1.0 - _smoothstep((local - SLEW_OUT - SLEW_HOLD) / SLEW_BACK))
        return 0.0

    def _command(self):
        self.targets[self.target_index[self.slew_joint]] = self.slew_target(self.sim_time)
        self.control.joint_target_q.assign(self.targets)

    def swing(self) -> float:
        """How far the tubes trail the jib, in degrees: the lag that makes the shot."""
        q = self.state_0.body_q.numpy()
        jib = float(
            math.atan2(*reversed(tuple(wp.quat_rotate(wp.quat(*q[self.jib, 3:7]), wp.vec3(1.0, 0.0, 0.0))[:2])))
        )
        lag = [abs(math.degrees(math.atan2(q[t, 1], q[t, 0]) - jib)) for t in self.tubes]
        return max(lag)

    def on_reset(self):
        self.slewing = True
        self.carried = 0.0

    def step(self):
        if self.slewing:
            self._command()
        self.stepper.step()
        # How far the load has been taken at any point; the cycle brings it back, so
        # the final position says nothing about whether the crane did its job.
        travelled = np.linalg.norm(self.state_0.body_q.numpy()[self.tubes, :2] - self.tube_start[:, :2], axis=1)
        self.carried = max(self.carried, float(travelled.max()))

    def substep(self):
        self.state_0.clear_forces()
        self.viewer.apply_forces(self.state_0)
        self.stepper.collide()
        self.stepper.solve()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_scalar("slew [deg]", math.degrees(self.slew_target(self.sim_time)))
        self.viewer.log_scalar("load lag [deg]", self.swing())
        self.viewer.end_frame()

    def gui(self, ui):
        ui.text(f"{len(self.tubes)} tubes on rope chains, {self.model.body_count} bodies")
        ui.text(
            f"slew {math.degrees(self.slew_target(self.sim_time)):5.1f} deg, lag {self.swing():4.1f} deg, carried {self.carried:4.1f} m"
        )
        changed, slewing = ui.checkbox("Run the slew cycle", self.slewing)
        if changed:
            self.slewing = slewing
        self.stepper.gui(ui)

    def test_final(self):
        assert_finite(self.state_0.body_q, self.state_0.body_qd)
        q = self.state_0.body_q.numpy()
        if np.any(q[:, 2] < -0.05):
            raise ValueError("a body fell through the ground")
        hung = q[self.tubes, 2]
        if np.any(hung < 1.0):
            raise ValueError("a tube came off its rope")
        if self.carried < 8.0:
            raise ValueError(f"the slew did not carry the tubes, furthest carried {self.carried:.1f} m")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--solver", default="feather_pgs", choices=list(SOLVERS), help="Rigid-body solver.")
        parser.set_defaults(num_frames=1140)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
