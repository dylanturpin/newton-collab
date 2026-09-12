# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example FeatherPGS Wrecking Ball
#
# A crane gantry hoists a giant steel ball away from an open-frame furnished
# building, six storeys of columns and slabs with no walls, and lets it go.
# The ball swings through the east face at mid-height and ploughs into the
# floors. Nothing is glued: every column, slab and piece of furniture is a
# free body held in place by friction. The chain is a Featherstone
# articulation of ball joints. The layout follows the avbd-metal wrecking
# ball demo.
#
# Command: python -m newton.examples fpgs_wrecking_ball
#
###########################################################################

from __future__ import annotations

import math

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.feather_pgs._showreel import Stepper, assert_finite, make_solver

FLOORS = 6
WIDTH, DEPTH = 14.0, 10.0
FLOOR_H, SLAB_T, COLUMN = 2.7, 0.3, 0.55
SLAB_OVERHANG = 0.4
# Slab pieces per storey, split along x. One slab per storey is a contact hub the
# graph-coloured solver must sweep colour by colour; halves rest on six columns
# each and stay just as stable.
SLAB_PIECES = 2
# Crane: jib tip just outside the east face, jib above the roof, chain reaching
# down to the building's mid-height. The ball is hoisted THETA from vertical
# away from the building and released from rest.
JIB_TIP_X = 8.5
JIB_Z = FLOORS * FLOOR_H + 4.6
BALL_RADIUS = 1.9
CHAIN_LEN = JIB_Z - FLOORS * FLOOR_H / 2 - BALL_RADIUS
THETA = 1.1
MAST_X = JIB_TIP_X + math.sin(THETA) * (CHAIN_LEN + BALL_RADIUS) + BALL_RADIUS + 1.2
LINK_HALF, LINK_R = 0.5, 0.3
LINK_DENSITY = 7800.0
LINKS = max(3, round(CHAIN_LEN / (2 * LINK_HALF)))
BALL_DENSITY = 7800.0
SOLVER_OVERRIDES = {
    "pgs_iterations": 4,
    "pgs_contact_regularization": 0.01,
    "dense_max_constraints": 4096,
    "mf_max_constraints": 8192,
}

CONCRETE = wp.vec3(0.72, 0.70, 0.66)
SLAB = wp.vec3(0.58, 0.58, 0.6)
STEEL = wp.vec3(0.25, 0.26, 0.28)
YELLOW = wp.vec3(0.9, 0.7, 0.15)


class Example:
    def __init__(self, viewer, args):
        newton.use_coord_layout_targets = True
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        # Four substeps at four iterations with a small proximal regularization keep
        # the two-hundred-tonne ball finite in the debris pile.
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.release_angle = THETA

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.rigid_gap = 0.005
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.9))
        self.concrete = newton.ModelBuilder.ShapeConfig(density=1600.0, mu=0.9, restitution=0.0)
        self.slab_cfg = newton.ModelBuilder.ShapeConfig(density=550.0, mu=0.9, restitution=0.0)
        self.building = []
        self.slabs = []
        self._build_building(builder)
        self._build_crane(builder)

        self.model = builder.finalize()
        self.model.rigid_contact_max = 64 * (len(self.building) + LINKS + 1)
        self.solver = make_solver(self.model, **SOLVER_OVERRIDES)
        self.collision_pipeline = newton.examples.create_collision_pipeline(
            self.model, args, broad_phase="sap", rigid_contact_max=self.model.rigid_contact_max
        )
        self.contacts = self.collision_pipeline.contacts()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.stepper = Stepper(self, solver_overrides=SOLVER_OVERRIDES)
        self.release()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(-2.0, -46.0, 15.0), pitch=-9.0, yaw=76.0)

    # ------------------------------------------------------------------ pieces
    def _body(self, builder, center, yaw=0.0):
        body = builder.add_body(
            xform=wp.transform(wp.vec3(*center), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw))
        )
        self.building.append(body)
        return body

    def _part(self, builder, body, size, local, density, color):
        """Box shape on ``body``; ``size`` is the full extent and ``local`` the box centre."""
        builder.add_shape_box(
            body,
            xform=wp.transform(wp.vec3(*local), wp.quat_identity()),
            hx=size[0] / 2,
            hy=size[1] / 2,
            hz=size[2] / 2,
            cfg=newton.ModelBuilder.ShapeConfig(density=density, mu=0.7, restitution=0.0),
            color=color,
        )

    def _box(self, builder, size, at, density=450.0, yaw=0.0, color=None):
        """Single free box whose centre is ``at``."""
        body = self._body(builder, at, yaw)
        self._part(builder, body, size, (0.0, 0.0, 0.0), density, color or wp.vec3(0.6, 0.55, 0.5))
        return body

    def _sofa(self, builder, p, yaw=0.0, color=None):
        fabric = color or wp.vec3(0.2, 0.35, 0.65)
        body = self._body(builder, p, yaw)
        self._part(builder, body, (2.0, 0.85, 0.4), (0.0, 0.0, 0.2), 450.0, fabric)
        self._part(builder, body, (2.0, 0.25, 0.55), (0.0, -0.32, 0.65), 450.0, fabric)
        self._part(builder, body, (0.25, 0.85, 0.3), (-0.93, 0.0, 0.55), 450.0, fabric)
        self._part(builder, body, (0.25, 0.85, 0.3), (0.93, 0.0, 0.55), 450.0, fabric)

    def _table(self, builder, p, top=(1.7, 1.0, 0.08), h=0.74, yaw=0.0):
        oak = wp.vec3(0.55, 0.36, 0.2)
        body = self._body(builder, p, yaw)
        self._part(builder, body, top, (0.0, 0.0, h - top[2] / 2), 600.0, oak)
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                self._part(
                    builder,
                    body,
                    (0.09, 0.09, h - top[2]),
                    (sx * (top[0] / 2 - 0.1), sy * (top[1] / 2 - 0.1), (h - top[2]) / 2),
                    600.0,
                    oak,
                )

    def _chair(self, builder, p, yaw=0.0):
        seat = wp.vec3(0.15, 0.15, 0.17)
        body = self._body(builder, p, yaw)
        self._part(builder, body, (0.45, 0.45, 0.08), (0.0, 0.0, 0.45), 450.0, seat)
        self._part(builder, body, (0.38, 0.38, 0.41), (0.0, 0.0, 0.205), 250.0, seat)
        self._part(builder, body, (0.45, 0.07, 0.5), (0.0, -0.19, 0.74), 450.0, seat)

    def _bed(self, builder, p, yaw=0.0):
        frame, linen = wp.vec3(0.35, 0.22, 0.14), wp.vec3(0.9, 0.9, 0.86)
        body = self._body(builder, p, yaw)
        self._part(builder, body, (2.0, 1.5, 0.35), (0.0, 0.0, 0.175), 500.0, frame)
        self._part(builder, body, (1.9, 1.4, 0.2), (0.0, 0.0, 0.45), 200.0, linen)
        self._part(builder, body, (0.12, 1.5, 0.75), (-1.0, 0.0, 0.375), 500.0, frame)

    def _toilet(self, builder, p, yaw=0.0):
        white = wp.vec3(0.92, 0.92, 0.9)
        body = self._body(builder, p, yaw)
        self._part(builder, body, (0.45, 0.55, 0.42), (0.0, 0.0, 0.21), 800.0, white)
        self._part(builder, body, (0.45, 0.18, 0.5), (0.0, -0.32, 0.62), 800.0, white)

    def _fridge(self, builder, p):
        self._box(builder, (0.75, 0.75, 1.85), (p[0], p[1], p[2] + 0.925), 800.0, color=wp.vec3(0.8, 0.82, 0.84))

    def _counter(self, builder, p, length, yaw=0.0):
        self._box(builder, (length, 0.65, 0.95), (p[0], p[1], p[2] + 0.475), 700.0, yaw, wp.vec3(0.75, 0.72, 0.65))

    def _shelf(self, builder, p, yaw=0.0):
        self._box(builder, (1.2, 0.35, 1.7), (p[0], p[1], p[2] + 0.85), 500.0, yaw, wp.vec3(0.4, 0.25, 0.15))

    def _tv(self, builder, p, yaw=0.0):
        body = self._body(builder, p, yaw)
        self._part(builder, body, (1.5, 0.45, 0.5), (0.0, 0.0, 0.25), 600.0, wp.vec3(0.3, 0.3, 0.32))
        self._part(builder, body, (1.3, 0.08, 0.75), (0.0, 0.0, 0.9), 300.0, wp.vec3(0.05, 0.05, 0.06))

    def _dining_set(self, builder, p):
        self._table(builder, p)
        self._chair(builder, (p[0], p[1] + 0.85, p[2]), yaw=math.pi)
        self._chair(builder, (p[0], p[1] - 0.85, p[2]), yaw=0.0)
        self._chair(builder, (p[0] + 1.15, p[1], p[2]), yaw=math.pi / 2)
        self._chair(builder, (p[0] - 1.15, p[1], p[2]), yaw=-math.pi / 2)

    # --------------------------------------------------------------- building
    def _furnish(self, builder, floor, z):
        """Rooms per floor, cycling through three programmes."""
        program = floor % 3
        if program == 0:  # kitchen + dining + living
            self._counter(builder, (-5.8, -3.5, z), 3.4)
            self._fridge(builder, (-3.6, -4.2, z))
            self._counter(builder, (-6.3, -1.2, z), 2.0, yaw=math.pi / 2)
            self._dining_set(builder, (-1.5, -2.2, z))
            self._sofa(builder, (3.5, 3.2, z), yaw=math.pi)
            self._tv(builder, (3.5, -0.5, z))
            self._shelf(builder, (6.2, 2.0, z), yaw=math.pi / 2)
        elif program == 1:  # guest room + bathroom
            self._bed(builder, (-4.5, 2.5, z))
            self._shelf(builder, (-6.4, -1.0, z), yaw=math.pi / 2)
            self._sofa(builder, (0.5, -3.3, z), color=wp.vec3(0.6, 0.25, 0.2))
            self._table(builder, (0.5, -1.2, z), top=(0.9, 0.9, 0.07), h=0.45)
            self._toilet(builder, (5.8, -3.8, z), yaw=math.pi / 2)
            self._counter(builder, (5.8, -1.8, z), 1.4, yaw=math.pi / 2)
            self._bed(builder, (4.5, 2.8, z), yaw=math.pi / 2)
        else:  # office + lounge
            self._table(builder, (-4.5, -3.0, z), top=(1.6, 0.8, 0.07))
            self._chair(builder, (-4.5, -1.9, z), yaw=math.pi)
            self._shelf(builder, (-6.4, 0.5, z), yaw=math.pi / 2)
            self._shelf(builder, (-6.4, 2.5, z), yaw=math.pi / 2)
            self._sofa(builder, (1.0, 2.8, z), yaw=math.pi, color=wp.vec3(0.25, 0.45, 0.3))
            self._sofa(builder, (-1.5, 0.0, z), yaw=-math.pi / 2, color=wp.vec3(0.7, 0.55, 0.2))
            self._tv(builder, (3.0, -3.5, z))
            self._dining_set(builder, (4.5, 1.5, z))

    def _build_building(self, builder):
        """Ten perimeter columns and one slab per storey, all free bodies, plus furniture."""
        col_h = FLOOR_H - SLAB_T
        for floor in range(FLOORS):
            z0 = floor * FLOOR_H
            xs = [-WIDTH / 2, -WIDTH / 4, 0.0, WIDTH / 4, WIDTH / 2]
            for x in xs:
                for y in (-DEPTH / 2, DEPTH / 2):
                    self._box(builder, (COLUMN, COLUMN, col_h - 0.004), (x, y, z0 + col_h / 2), 1600.0, color=CONCRETE)
            self._furnish(builder, floor, z0)
            piece = (WIDTH + 2 * SLAB_OVERHANG) / SLAB_PIECES
            for k in range(SLAB_PIECES):
                slab = self._box(
                    builder,
                    (piece - 0.02, DEPTH + 2 * SLAB_OVERHANG, SLAB_T),
                    (-WIDTH / 2 - SLAB_OVERHANG + (k + 0.5) * piece, 0.0, z0 + FLOOR_H - SLAB_T / 2),
                    550.0,
                    color=SLAB,
                )
                self.slabs.append(slab)
        self.height = FLOORS * FLOOR_H

    # ------------------------------------------------------------------ crane
    def _build_crane(self, builder):
        # Static gantry: base, mast, jib and counterweight.
        gantry = newton.ModelBuilder.ShapeConfig(mu=0.5)
        for size, at, color in (
            ((5.5, 5.5, 0.8), (MAST_X + 1.0, 0.0, 0.4), STEEL),
            ((1.1, 1.1, JIB_Z), (MAST_X, 0.0, JIB_Z / 2), YELLOW),
            ((MAST_X - JIB_TIP_X + 2.4, 0.9, 0.9), ((MAST_X + JIB_TIP_X) / 2, 0.0, JIB_Z + 0.45), YELLOW),
            ((2.0, 2.2, 1.6), (MAST_X + 2.6, 0.0, JIB_Z - 0.5), STEEL),
        ):
            builder.add_shape_box(
                -1,
                xform=wp.transform(wp.vec3(*at), wp.quat_identity()),
                hx=size[0] / 2,
                hy=size[1] / 2,
                hz=size[2] / 2,
                cfg=gantry,
                color=color,
            )
        # Chain of ball joints hanging straight down from the jib tip; release()
        # then swings the root joint out to the hoisted angle.
        steel = newton.ModelBuilder.ShapeConfig(density=LINK_DENSITY, mu=0.5, restitution=0.0)
        # Hang from just below the jib so no link capsule starts inside the static box.
        tip = wp.vec3(JIB_TIP_X, 0.0, JIB_Z - LINK_R - 0.05)
        link_len = CHAIN_LEN / LINKS
        joints, parent = [], -1
        self.links = []
        for i in range(LINKS):
            z = tip[2] - (i + 0.5) * link_len
            link = builder.add_link(xform=wp.transform(wp.vec3(JIB_TIP_X, 0.0, z), wp.quat_identity()))
            # Capsule ends stop short of the joints so neighbouring links never overlap.
            builder.add_shape_capsule(
                link, radius=LINK_R, half_height=link_len / 2 - LINK_R - 0.02, cfg=steel, color=STEEL
            )
            top = wp.transform(wp.vec3(0.0, 0.0, link_len / 2), wp.quat_identity())
            if parent < 0:
                joint = builder.add_joint_ball(
                    -1, link, parent_xform=wp.transform(tip, wp.quat_identity()), child_xform=top
                )
            else:
                joint = builder.add_joint_ball(
                    parent,
                    link,
                    parent_xform=wp.transform(wp.vec3(0.0, 0.0, -link_len / 2), wp.quat_identity()),
                    child_xform=top,
                )
            joints.append(joint)
            self.links.append(link)
            parent = link
        ball_cfg = newton.ModelBuilder.ShapeConfig(density=BALL_DENSITY, mu=0.4, restitution=0.0)
        ball_z = tip[2] - CHAIN_LEN - BALL_RADIUS
        self.ball = builder.add_link(xform=wp.transform(wp.vec3(JIB_TIP_X, 0.0, ball_z), wp.quat_identity()))
        builder.add_shape_sphere(self.ball, radius=BALL_RADIUS, cfg=ball_cfg, color=wp.vec3(0.1, 0.1, 0.11))
        joints.append(
            builder.add_joint_ball(
                parent,
                self.ball,
                parent_xform=wp.transform(wp.vec3(0.0, 0.0, -link_len / 2), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(0.0, 0.0, BALL_RADIUS), wp.quat_identity()),
            )
        )
        builder.add_articulation(joints, label="crane")
        self.root_joint = joints[0]

    def release(self):
        """Hoist the chain away from the building by the release angle and let go from rest."""
        q = self.model.joint_q.numpy().copy()
        start = int(self.model.joint_q_start.numpy()[self.root_joint])
        # Positive rotation about +y swings the hanging chain toward -x; the crane is
        # east of the building, so hoist toward +x.
        swing = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), -float(self.release_angle))
        q[start : start + 4] = [swing[0], swing[1], swing[2], swing[3]]
        self.state_0.joint_q.assign(q)
        self.state_0.joint_qd.zero_()
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

    # ------------------------------------------------------------------- loop
    def step(self):
        self.stepper.step()

    def substep(self):
        self.state_0.clear_forces()
        self.viewer.apply_forces(self.state_0)
        self.collision_pipeline.collide(self.state_0, self.contacts)
        self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
        self.state_0, self.state_1 = self.state_1, self.state_0

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        qd = self.state_0.body_qd.numpy()
        self.viewer.log_scalar("ball speed [m/s]", float(np.linalg.norm(qd[self.ball, :3])))
        self.viewer.log_scalar("contacts", int(self.contacts.rigid_contact_count.numpy()[0]))
        self.viewer.end_frame()

    def gui(self, ui):
        mass = 4.0 / 3.0 * math.pi * BALL_RADIUS**3 * BALL_DENSITY
        ui.text(
            f"{len(self.building)} building parts, {LINKS}-link chain, {BALL_RADIUS:.1f} m / {mass / 1000:.0f} t ball"
        )
        _, deg = ui.slider_float("Hoist angle [deg]", math.degrees(self.release_angle), 10.0, 85.0)
        self.release_angle = math.radians(deg)
        q = self.state_0.body_q.numpy()
        ui.text(f"ball at x={q[self.ball, 0]:+.2f} z={q[self.ball, 2]:.2f}")
        ui.text("Reset the example to swing again from the chosen angle")
        self.stepper.gui(ui)

    def test_final(self):
        assert_finite(self.state_0.body_q, self.state_0.body_qd)
        q = self.state_0.body_q.numpy()
        if np.any(q[:, 2] < -0.05):
            raise ValueError("a body fell through the ground")
        if q[self.slabs[-SLAB_PIECES:], 2].min() > self.height - 0.5:
            raise ValueError("the wrecking ball did not bring the roof down")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=600)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
