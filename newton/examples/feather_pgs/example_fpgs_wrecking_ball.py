# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example FeatherPGS Wrecking Ball
#
# A seventeen-tonne ball hangs from a crane on a twelve-link ball-jointed chain
# and swings into a three-storey building: concrete columns and slabs,
# brick facade panels with glass strips, and furnished floors. Nothing is
# glued: every column, panel, pane and piece of furniture is a free body
# held in place by friction. The chain is a Featherstone articulation with
# a seventeen-to-one mass ratio between the bob and its links.
#
# Command: python -m newton.examples fpgs_wrecking_ball
#
###########################################################################

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.feather_pgs._showreel import Stepper, assert_finite, make_solver

FLOORS = 3
WIDTH, DEPTH = 14.0, 10.0
FLOOR_H, SLAB_T, COLUMN = 2.7, 0.3, 0.55
EAVE = 0.4  # slab overhang past the outer column faces
MU_CONCRETE, MU_BRICK = 0.6, 0.6
CLAD_GAP = 0.06  # clearance between cladding tops and the slab above
SLAB_DENSITY = 2400.0
BAYS = 4
COLUMN_SEGMENTS = 3  # stacked blocks per storey column
LINKS = 12
# A 17:1 ball-to-link mass ratio keeps the articulated chain free of velocity
# spikes on secondary impacts; thinner links or a heavier ball spike past 60 m/s.
LINK_HALF, LINK_R = 0.5, 0.18
BALL_RADIUS = 0.8
# The bottom of the swing puts the ball centre mid-height on the first floor, clear
# of the slabs above and below it, so the impact takes out columns rather than
# wedging the ball under a slab edge.
ANCHOR = (-WIDTH / 2 - 0.3, 0.0, FLOOR_H * 1.5 - 0.15 + 2 * LINKS * LINK_HALF + BALL_RADIUS)
RELEASE_ANGLE = math.radians(60.0)
SOLVER_OVERRIDES = {
    "pgs_iterations": 8,
    "pgs_contact_regularization": 0.01,
    "dense_max_constraints": 4096,
    "mf_max_constraints": 8192,
}

CONCRETE = wp.vec3(0.72, 0.70, 0.66)
SLAB = wp.vec3(0.58, 0.58, 0.6)
BRICK = wp.vec3(0.62, 0.26, 0.18)
GLASS = wp.vec3(0.55, 0.8, 0.95)
STEEL = wp.vec3(0.25, 0.26, 0.28)


class Example:
    def __init__(self, viewer, args):
        newton.use_coord_layout_targets = True
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        # Two substeps at eight iterations with a small proximal regularization keep
        # the seventeen-tonne impact finite; six iterations blow up in the debris pile.
        self.sim_substeps = 2
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.release_angle = RELEASE_ANGLE

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.rigid_gap = 0.005
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.9))
        # Real material densities keep the ball-to-debris mass ratio near 100:1,
        # which velocity-level PGS handles; lighter debris blows up on impact.
        self.concrete = newton.ModelBuilder.ShapeConfig(density=2400.0, mu=MU_CONCRETE, restitution=0.0)
        self.slab_cfg = newton.ModelBuilder.ShapeConfig(density=SLAB_DENSITY, mu=MU_CONCRETE, restitution=0.0)
        self.brick = newton.ModelBuilder.ShapeConfig(density=1800.0, mu=MU_BRICK, restitution=0.0)
        self.glass = newton.ModelBuilder.ShapeConfig(density=2500.0, mu=0.4, restitution=0.0)
        self.wood = newton.ModelBuilder.ShapeConfig(density=700.0, mu=0.7, restitution=0.0)
        self.building = []
        self.slabs = []
        self._build_building(builder)
        self._build_crane(builder)

        self.model = builder.finalize()
        self.model.rigid_contact_max = 48 * (len(self.building) + LINKS + 1)
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
        self.viewer.set_camera(pos=wp.vec3(-21.0, -27.0, 11.0), pitch=-15.0, yaw=52.0)

    def _box(self, builder, cfg, color, center, half, yaw=0.0):
        body = builder.add_body(
            xform=wp.transform(wp.vec3(*center), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw))
        )
        builder.add_shape_box(body, hx=half[0], hy=half[1], hz=half[2], cfg=cfg, color=color)
        self.building.append(body)
        return body

    def _composite(self, builder, center, parts, yaw=0.0):
        """One free body made of several boxes: (local center, half extents, cfg, color)."""
        body = builder.add_body(
            xform=wp.transform(wp.vec3(*center), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw))
        )
        for local, half, cfg, color in parts:
            builder.add_shape_box(
                body,
                xform=wp.transform(wp.vec3(*local), wp.quat_identity()),
                hx=half[0],
                hy=half[1],
                hz=half[2],
                cfg=cfg,
                color=color,
            )
        self.building.append(body)
        return body

    def _sofa(self, builder, center, yaw=0.0):
        fabric = wp.vec3(0.2, 0.35, 0.65)
        self._composite(
            builder,
            center,
            [
                ((0.0, 0.0, 0.2), (1.0, 0.42, 0.2), self.wood, fabric),
                ((0.0, -0.32, 0.65), (1.0, 0.12, 0.27), self.wood, fabric),
                ((-0.93, 0.0, 0.55), (0.12, 0.42, 0.15), self.wood, fabric),
                ((0.93, 0.0, 0.55), (0.12, 0.42, 0.15), self.wood, fabric),
            ],
            yaw,
        )

    def _table(self, builder, center, yaw=0.0):
        oak = wp.vec3(0.55, 0.36, 0.2)
        legs = [((sx * 0.75, sy * 0.4, 0.35), (0.05, 0.05, 0.35), self.wood, oak) for sx in (-1, 1) for sy in (-1, 1)]
        self._composite(builder, center, [((0.0, 0.0, 0.74), (0.85, 0.5, 0.04), self.wood, oak), *legs], yaw)

    def _shelf(self, builder, center, yaw=0.0):
        walnut = wp.vec3(0.4, 0.25, 0.15)
        boards = [((0.0, 0.0, 0.3 + 0.45 * k), (0.6, 0.18, 0.02), self.wood, walnut) for k in range(4)]
        self._composite(
            builder,
            center,
            [
                ((-0.6, 0.0, 0.95), (0.02, 0.18, 0.95), self.wood, walnut),
                ((0.6, 0.0, 0.95), (0.02, 0.18, 0.95), self.wood, walnut),
                *boards,
            ],
            yaw,
        )

    def _build_building(self, builder):
        col_h = FLOOR_H - SLAB_T
        bay = WIDTH / BAYS
        for floor in range(FLOORS):
            z0 = floor * FLOOR_H
            # Columns: corners, mid-edges, and interior bays, each a stack of short
            # blocks. A monolithic column wedged between two slabs can only slide,
            # and sliding under the building's weight is a friction brake that eats
            # the ball's energy without dropping anything. A knocked-out block leaves
            # the blocks above it unsupported instead.
            seg_h = col_h / COLUMN_SEGMENTS
            for gx in range(BAYS + 1):
                for gy in (-DEPTH / 2, 0.0, DEPTH / 2):
                    x = -WIDTH / 2 + gx * bay
                    for k in range(COLUMN_SEGMENTS):
                        self._box(
                            builder,
                            self.concrete,
                            CONCRETE,
                            (x, gy, z0 + (k + 0.5) * seg_h),
                            (COLUMN / 2, COLUMN / 2, seg_h / 2 - 0.002),
                        )
            # Facade: brick spandrel, glass strip and parapet stacked per bay on the front
            # and back. The stack stops CLAD_GAP short of the slab above so the columns
            # alone carry the floors and a knocked-out column actually drops something.
            for gy, sign in ((-DEPTH / 2, 1.0), (DEPTH / 2, -1.0)):
                for gx in range(BAYS):
                    x = -WIDTH / 2 + (gx + 0.5) * bay
                    y = gy + sign * 0.15
                    self._box(builder, self.brick, BRICK, (x, y, z0 + 0.45), (bay / 2 - COLUMN / 2 - 0.02, 0.12, 0.45))
                    self._box(builder, self.brick, BRICK, (x, y, z0 + 2.07), (bay / 2 - COLUMN / 2 - 0.02, 0.12, 0.27))
                    self._box(builder, self.glass, GLASS, (x, y, z0 + 1.35), (bay / 2 - COLUMN / 2 - 0.04, 0.03, 0.44))
            # Side walls: brick panels between the end columns.
            for gx in (-WIDTH / 2, WIDTH / 2):
                for gy in (-DEPTH / 4, DEPTH / 4):
                    self._box(
                        builder,
                        self.brick,
                        BRICK,
                        (gx, gy, z0 + (col_h - CLAD_GAP) / 2),
                        (0.12, DEPTH / 4 - COLUMN / 2 - 0.02, (col_h - CLAD_GAP) / 2 - 0.005),
                    )
            # Furniture on every floor.
            self._sofa(builder, (-3.5, -1.5, z0), yaw=0.4)
            self._table(builder, (-3.5, 1.8, z0), yaw=0.2)
            self._sofa(builder, (3.0, 2.0, z0), yaw=math.pi + 0.3)
            self._shelf(builder, (4.5, -3.0, z0), yaw=-0.5)
            # Slab, as one segment per bay and half-depth. A monolithic slab touches
            # every column and panel on the floor, and the graph-colored solver can
            # place only one row per body per colour, so a single hub body would
            # serialise the sweep. Segments rest on four columns each and
            # overhang the outer column faces so upper columns stand fully on slab.
            xs = [-WIDTH / 2 - EAVE, *(-WIDTH / 2 + gx * bay for gx in range(1, BAYS)), WIDTH / 2 + EAVE]
            ys = [-DEPTH / 2 - EAVE, 0.0, DEPTH / 2 + EAVE]
            for x_lo, x_hi in pairwise(xs):
                for y_lo, y_hi in pairwise(ys):
                    slab = self._box(
                        builder,
                        self.slab_cfg,
                        SLAB,
                        ((x_lo + x_hi) / 2, (y_lo + y_hi) / 2, z0 + FLOOR_H - SLAB_T / 2),
                        ((x_hi - x_lo) / 2 - 0.01, (y_hi - y_lo) / 2 - 0.01, SLAB_T / 2),
                    )
                    self.slabs.append(slab)

    def _build_crane(self, builder):
        steel = newton.ModelBuilder.ShapeConfig(density=7800.0, mu=0.5, restitution=0.0)
        anchor = wp.vec3(*ANCHOR)
        joints, parent = [], -1
        self.links = []
        for i in range(LINKS):
            z = ANCHOR[2] - (2 * i + 1) * LINK_HALF
            link = builder.add_link(xform=wp.transform(wp.vec3(ANCHOR[0], ANCHOR[1], z), wp.quat_identity()))
            builder.add_shape_capsule(link, radius=LINK_R, half_height=LINK_HALF, cfg=steel, color=STEEL)
            top = wp.transform(wp.vec3(0.0, 0.0, LINK_HALF), wp.quat_identity())
            if parent < 0:
                joint = builder.add_joint_ball(
                    -1, link, parent_xform=wp.transform(anchor, wp.quat_identity()), child_xform=top
                )
            else:
                joint = builder.add_joint_ball(
                    parent,
                    link,
                    parent_xform=wp.transform(wp.vec3(0.0, 0.0, -LINK_HALF), wp.quat_identity()),
                    child_xform=top,
                )
            joints.append(joint)
            self.links.append(link)
            parent = link
        ball_z = ANCHOR[2] - 2 * LINKS * LINK_HALF - BALL_RADIUS
        self.ball = builder.add_link(xform=wp.transform(wp.vec3(ANCHOR[0], ANCHOR[1], ball_z), wp.quat_identity()))
        builder.add_shape_sphere(self.ball, radius=BALL_RADIUS, cfg=steel, color=wp.vec3(0.1, 0.1, 0.11))
        joints.append(
            builder.add_joint_ball(
                parent,
                self.ball,
                parent_xform=wp.transform(wp.vec3(0.0, 0.0, -LINK_HALF), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(0.0, 0.0, BALL_RADIUS), wp.quat_identity()),
            )
        )
        builder.add_articulation(joints, label="crane")
        self.root_joint = joints[0]

    def release(self):
        """Hang the chain swung back by the release angle and let go from rest."""
        q = self.model.joint_q.numpy().copy()
        start = int(self.model.joint_q_start.numpy()[self.root_joint])
        swing = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), float(self.release_angle))
        q[start : start + 4] = [swing[0], swing[1], swing[2], swing[3]]
        self.state_0.joint_q.assign(q)
        self.state_0.joint_qd.zero_()
        newton.eval_fk(self.model, self.state_0.joint_q, self.state_0.joint_qd, self.state_0)

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
        ui.text(f"{len(self.building)} building parts, {LINKS}-link chain, {BALL_RADIUS:.1f} m ball")
        _, deg = ui.slider_float("Release angle [deg]", math.degrees(self.release_angle), 10.0, 85.0)
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
        roof = self.slabs[-2 * BAYS :]
        if q[roof, 2].min() > FLOORS * FLOOR_H - 0.5:
            raise ValueError("the wrecking ball did not bring any roof segment down")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=600)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
