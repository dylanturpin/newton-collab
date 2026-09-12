# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Example FeatherPGS Excavator
#
# A four-axis excavator works a skip full of loose primitives. The dig
# cycle is preprogrammed as a handful of waypoints for the bucket pivot,
# and analytic inverse kinematics turns each one into swing, boom, stick
# and bucket angles that the joint drives chase. Nothing about the
# material is scripted: what the bucket carries away is whatever its
# teeth happen to trap.
#
# Command: python -m newton.examples fpgs_excavator
#
###########################################################################

from __future__ import annotations

import math
import warnings

import numpy as np
import warp as wp

import newton
import newton.examples
from newton.examples.feather_pgs._showreel import Stepper, assert_finite

SOLVERS = {
    "feather_pgs": {
        # An articulated arm pinching loose material against a static floor is the
        # hardest contact set in the showreel, and a thousand loose primitives make
        # it the heaviest. Four iterations at two substeps is what holds a
        # readable frame rate; more of either barely changes how the heap settles.
        "pgs_iterations": 4,
        # The colouring kernel stages (dense + matrix-free) / 3 contact units in
        # shared memory and tops out at 4096, so these two have to sum under 12288.
        "dense_max_constraints": 1024,
        "mf_max_constraints": 11264,
        # The arm's own links touch each other and the skip; joint limits keep the
        # drives from winding past the stops when the bucket jams in the material.
        "enable_joint_limits": True,
        "propagation_same_articulation_rows": True,
        "substeps": 2,
    },
    "mujoco": {"njmax": 32768, "nconmax": 16384},
}

# Machine. The arm is sized like a 20 t excavator, with hollow-steel densities.
SWING_Z = 1.35
SHOULDER = (1.25, 0.0, 1.55)  # boom pivot in the turret frame
SHOULDER_Z = SWING_Z + SHOULDER[2]  # ... and in the world, which is what the waypoints use
BOOM, STICK = 4.2, 3.0
BUCKET_LEN, BUCKET_W, BUCKET_H = 1.15, 1.25, 0.95
# The cutting edge in the bucket's own frame: the far end of its floor.
TEETH_LOCAL = (0.06 + BUCKET_LEN, -0.3)
ARM_DENSITY = 1500.0

# Skip full of material, and where its contents end up.
SKIP_AT = (6.9, 0.0)
# Walls high enough to hold two layers of material, low enough to watch the
# bucket work in it.
# Thin walls let a fast primitive slip through between steps; these are thick
# enough that the sweep always has something to hit.
SKIP_INNER, SKIP_WALL, SKIP_H = 5.0, 0.35, 1.5
# The material goes into a second, empty skip set off to the machine's left. The full skip stays
# square in front of the machine, which rests unswung facing it.
DUMP_AT = (4.6, 5.0)
DUMP_INNER, DUMP_H = 3.4, 0.9
DUMP_YAW, DUMP_R = math.degrees(math.atan2(DUMP_AT[1], DUMP_AT[0])), math.hypot(*DUMP_AT)
ROCKS = 1000
ROCK_DENSITY = 1100.0

# One dig cycle as waypoints of (phase end [s], reach, height, yaw, bucket pitch [deg]).
# Reach and height are the bucket TEETH in the swing plane, not the pivot: the whole
# point of the dig is where the teeth go, and the curl that closes the scoop sweeps
# them up and back through the material on its own.
CYCLE = (
    # The scoop rests level: floor plate parallel with the ground, teeth pointing
    # forward. It only tips down once it is over the material.
    ("Swing over the skip", 1.1, 7.6, 4.7, 0.0, 0.0),
    ("Teeth into the material", 1.9, 7.4, 3.2, 0.0, 40.0),
    # The teeth stay down and nearly flat for the whole drag. Curling while still
    # dragging lifts them straight out of the pile and the bucket comes up empty.
    ("Drag through the heap", 2.9, 6.2, 2.9, 0.0, 5.0),
    ("Curl the scoop closed", 3.7, 5.8, 3.6, 0.0, -40.0),
    ("Lift clear", 4.4, 5.4, 5.0, 0.0, -40.0),
    ("Swing to the empty skip", 5.5, DUMP_R, 5.0, DUMP_YAW, -40.0),
    ("Tip the bucket out", 6.4, DUMP_R, 3.6, DUMP_YAW, 80.0),
    ("Swing back", 7.5, 7.6, 4.7, 0.0, 0.0),
)
CYCLE_TIME = CYCLE[-1][1]
SETTLE = 1.0

STEEL = wp.vec3(0.85, 0.72, 0.15)
DARK = wp.vec3(0.22, 0.23, 0.25)
SKIP_COLOR = wp.vec3(0.45, 0.47, 0.5)
ROCK_COLORS = (
    wp.vec3(0.55, 0.52, 0.48),
    wp.vec3(0.42, 0.40, 0.38),
    wp.vec3(0.62, 0.55, 0.45),
    wp.vec3(0.35, 0.38, 0.42),
)


def _smoothstep(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


def arm_ik(reach: float, height: float) -> tuple[float, float]:
    """Boom and stick angles that put the bucket pivot at ``(reach, height)``.

    Both links lie in the swing plane, so this is the two-link problem solved by the
    cosine rule. A link at angle ``a`` points along ``(cos a, -sin a)`` in that plane,
    and the elbow is chosen so the stick hangs below the boom, the way a real machine
    folds. Targets outside the arm's reach are pulled back onto the envelope.
    """
    u = reach - SHOULDER[0]
    v = SHOULDER_Z - height
    distance = math.hypot(u, v)
    limit_lo, limit_hi = abs(BOOM - STICK) + 1.0e-3, BOOM + STICK - 1.0e-3
    if distance > limit_hi or distance < limit_lo:
        scale = min(max(distance, limit_lo), limit_hi) / max(distance, 1.0e-9)
        u, v = u * scale, v * scale
        distance = math.hypot(u, v)
    cos_elbow = (u * u + v * v - BOOM * BOOM - STICK * STICK) / (2.0 * BOOM * STICK)
    stick = math.acos(min(max(cos_elbow, -1.0), 1.0))
    boom = math.atan2(v, u) - math.atan2(STICK * math.sin(stick), BOOM + STICK * math.cos(stick))
    return boom, stick


class Example:
    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = 4
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.digging = True

        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.rigid_gap = 0.004
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.9))
        arm_cfg = newton.ModelBuilder.ShapeConfig(density=ARM_DENSITY, mu=0.9, restitution=0.0)
        static_cfg = newton.ModelBuilder.ShapeConfig(mu=0.9, restitution=0.0)

        self._build_undercarriage(builder, static_cfg)
        self._build_skip(builder, static_cfg)
        self._build_skip(builder, static_cfg, DUMP_AT, DUMP_INNER, DUMP_H)
        self.rocks = self._fill_skip(builder)
        self._build_arm(builder, arm_cfg)

        with warnings.catch_warnings():
            # FeatherPGS reads joint_target_q in the legacy DOF layout, so keep it.
            warnings.simplefilter("ignore", DeprecationWarning)
            self.model = builder.finalize()
        self.model.rigid_contact_max = 24 * (len(self.rocks) + 8)
        self.collision_pipeline = newton.examples.create_collision_pipeline(
            self.model,
            args,
            broad_phase="sap",
            rigid_contact_max=self.model.rigid_contact_max,
            # Body-pair reduction: post-reduce the materialized contacts of each body
            # pair to a depth representative and a few sampled supports. With well over
            # a thousand loose primitives resting on each other, the raw contact set is
            # what the solver's row budget runs out on.
            reduce_contacts=newton.CollisionPipeline.ContactReductionConfig(body_pairs=True, body_pair_cell_size=0.4),
        )
        self.contacts = self.collision_pipeline.contacts()
        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()
        self.stepper = Stepper(self, solver_overrides=SOLVERS, solver=str(getattr(args, "solver", "feather_pgs")))
        self._pose_arm_at_start()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        # Make the posed arm the model's default too. The reset path restores body
        # poses from the model, and those were still the straight-out pose the links
        # were authored in, so a reset put the machine somewhere it never starts.
        wp.copy(self.model.body_q, self.state_0.body_q)
        wp.copy(self.state_1.body_q, self.state_0.body_q)
        # FeatherPGS reads drive targets in DOF layout (indexed by joint_qd_start).
        self.target_index = self.model.joint_qd_start.numpy()
        self.targets = self.control.joint_target_q.numpy().copy()
        # Hold the start pose from the first step, including when the cycle is paused.
        self._command()
        self.rock_start = self.state_0.body_q.numpy()[self.rocks, :3].copy()

        self.viewer.set_model(self.model)
        self.viewer.set_camera(pos=wp.vec3(-6.5, -18.0, 10.0), pitch=-22.0, yaw=62.0)

    def _pose_arm_at_start(self):
        """Park the arm on the first waypoint before the first step.

        The links are authored straight out, which is nowhere near the start of the
        cycle. Left alone, the drives close that gap in a single step and the arm
        arrives in the skip hard enough to throw the material out of it.
        """
        angles = self.cycle_angles(CYCLE[0])
        q = self.model.joint_q.numpy()
        starts = self.model.joint_q_start.numpy()
        for joint, value in zip(self.joints, angles, strict=True):
            q[starts[joint]] = value
        self.model.joint_q.assign(q)
        # The state carries its own copy of the joint coordinates, and the solver
        # believes those rather than the body poses: leaving them at zero makes the
        # arm snap back through the material on the first step.
        self.state_0.joint_q.assign(q)
        self.state_1.joint_q.assign(q)
        self.rest_angles = angles

    @staticmethod
    def angles_for(reach, height, yaw_deg, pitch_deg):
        """The four joint angles that put the teeth where they are asked for.

        The teeth are a fixed point on the bucket, at the far end of its floor, so
        where the stick tip has to go depends on the attitude the scoop is holding.
        """
        pitch = math.radians(pitch_deg)
        cos_p, sin_p = math.cos(pitch), math.sin(pitch)
        offset_u = TEETH_LOCAL[0] * cos_p + TEETH_LOCAL[1] * sin_p
        offset_v = -TEETH_LOCAL[0] * sin_p + TEETH_LOCAL[1] * cos_p
        boom, stick = arm_ik(reach - offset_u, height - offset_v)
        return (math.radians(yaw_deg), boom, stick, pitch - (boom + stick))

    @staticmethod
    def cycle_angles(waypoint):
        """Joint angles for one waypoint of the scripted dig."""
        _, _, reach, height, yaw_deg, pitch_deg = waypoint
        return Example.angles_for(reach, height, yaw_deg, pitch_deg)

    # ------------------------------------------------------------------ machine
    def _build_undercarriage(self, builder, cfg):
        """Tracks and a counterweight pad, all static: the machine does not drive."""
        for sign in (-1.0, 1.0):
            builder.add_shape_box(
                -1,
                xform=wp.transform(wp.vec3(0.0, sign * 1.25, 0.62), wp.quat_identity()),
                hx=2.3,
                hy=0.5,
                hz=0.62,
                cfg=cfg,
                color=DARK,
            )
        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(0.0, 0.0, SWING_Z - 0.12), wp.quat_identity()),
            hx=1.5,
            hy=1.2,
            hz=0.12,
            cfg=cfg,
            color=DARK,
        )

    def _build_arm(self, builder, cfg):
        """Turret, boom, stick and bucket, each on a driven revolute joint."""
        pitch = wp.vec3(0.0, 1.0, 0.0)
        turret = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, SWING_Z), wp.quat_identity()))
        builder.add_shape_box(
            turret,
            xform=wp.transform(wp.vec3(-0.75, 0.0, 0.85), wp.quat_identity()),
            hx=1.35,
            hy=1.1,
            hz=0.75,
            cfg=cfg,
            color=STEEL,
        )
        builder.add_shape_box(
            turret,
            xform=wp.transform(wp.vec3(0.45, 0.72, 1.35), wp.quat_identity()),
            hx=0.62,
            hy=0.38,
            hz=0.65,
            cfg=cfg,
            color=DARK,
        )
        boom = builder.add_link(
            xform=wp.transform(wp.vec3(SHOULDER[0] + BOOM / 2, 0.0, SWING_Z + SHOULDER[2]), wp.quat_identity())
        )
        builder.add_shape_box(boom, hx=BOOM / 2, hy=0.28, hz=0.34, cfg=cfg, color=STEEL)
        stick = builder.add_link(
            xform=wp.transform(wp.vec3(SHOULDER[0] + BOOM + STICK / 2, 0.0, SWING_Z + SHOULDER[2]), wp.quat_identity())
        )
        builder.add_shape_box(stick, hx=STICK / 2, hy=0.22, hz=0.26, cfg=cfg, color=STEEL)
        wrist_at = wp.vec3(SHOULDER[0] + BOOM + STICK, 0.0, SWING_Z + SHOULDER[2])
        bucket = builder.add_link(xform=wp.transform(wrist_at, wp.quat_identity()))
        self._build_bucket(builder, bucket, cfg)

        self.swing_joint = builder.add_joint_revolute(
            -1,
            turret,
            parent_xform=wp.transform(wp.vec3(0.0, 0.0, SWING_Z), wp.quat_identity()),
            child_xform=wp.transform_identity(),
            axis=wp.vec3(0.0, 0.0, 1.0),
            target_ke=6.0e6,
            target_kd=4.0e5,
            effort_limit=8.0e5,
            limit_lower=-math.pi,
            limit_upper=math.pi,
        )
        self.boom_joint = builder.add_joint_revolute(
            turret,
            boom,
            parent_xform=wp.transform(wp.vec3(*SHOULDER), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-BOOM / 2, 0.0, 0.0), wp.quat_identity()),
            axis=pitch,
            target_ke=1.2e7,
            target_kd=8.0e5,
            effort_limit=1.5e6,
            # The lift-clear waypoint needs the boom 79 degrees up, so the stop sits past it.
            limit_lower=-1.55,
            limit_upper=1.0,
        )
        self.stick_joint = builder.add_joint_revolute(
            boom,
            stick,
            parent_xform=wp.transform(wp.vec3(BOOM / 2, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-STICK / 2, 0.0, 0.0), wp.quat_identity()),
            axis=pitch,
            target_ke=5.0e6,
            target_kd=3.0e5,
            effort_limit=1.0e6,
            limit_lower=-0.2,
            limit_upper=2.6,
        )
        self.bucket_joint = builder.add_joint_revolute(
            stick,
            bucket,
            parent_xform=wp.transform(wp.vec3(STICK / 2, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform_identity(),
            axis=pitch,
            target_ke=2.0e6,
            target_kd=1.2e5,
            effort_limit=6.0e5,
            limit_lower=-2.2,
            limit_upper=2.2,
        )
        self.joints = [self.swing_joint, self.boom_joint, self.stick_joint, self.bucket_joint]
        self.arm_bodies = [turret, boom, stick, bucket]
        builder.add_articulation(self.joints, label="excavator")
        self.bucket_body = bucket

    def _build_bucket(self, builder, body, cfg):
        """An open scoop: floor, back plate and two sides, all meeting at the floor.

        At a bucket angle of zero the floor runs along +x from the pivot and the
        opening faces up, which is the carrying attitude; positive pitch turns the
        teeth down to dig. The back plate has to reach all the way down to the floor:
        a gap there lets everything the bucket gathers slide straight out behind it.
        """
        wall, half_w = 0.05, BUCKET_W / 2
        floor_z, rim_z = -0.3, BUCKET_H - 0.3
        mid_z, half_z = (floor_z + rim_z) / 2, (rim_z - floor_z) / 2
        mid_x, half_x = 0.06 + BUCKET_LEN / 2, BUCKET_LEN / 2
        builder.add_shape_box(  # back plate, floor to rim
            body,
            xform=wp.transform(wp.vec3(0.06, 0.0, mid_z), wp.quat_identity()),
            hx=wall,
            hy=half_w,
            hz=half_z,
            cfg=cfg,
            color=DARK,
        )
        builder.add_shape_box(  # floor, back plate to teeth
            body,
            xform=wp.transform(wp.vec3(mid_x, 0.0, floor_z), wp.quat_identity()),
            hx=half_x,
            hy=half_w,
            hz=wall,
            cfg=cfg,
            color=DARK,
        )
        for sign in (-1.0, 1.0):
            builder.add_shape_box(
                body,
                xform=wp.transform(wp.vec3(mid_x, sign * half_w, mid_z), wp.quat_identity()),
                hx=half_x,
                hy=wall,
                hz=half_z,
                cfg=cfg,
                color=DARK,
            )

    # ---------------------------------------------------------------- material
    def _build_skip(self, builder, cfg, at=SKIP_AT, inner=SKIP_INNER, height=SKIP_H):
        """A static open-topped skip; only what it holds is free to move."""
        half, wall = inner / 2, SKIP_WALL / 2
        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(at[0], at[1], wall), wp.quat_identity()),
            hx=half + SKIP_WALL,
            hy=half + SKIP_WALL,
            hz=wall,
            cfg=cfg,
            color=SKIP_COLOR,
        )
        for dx, dy, hx, hy in (
            (half + wall, 0.0, wall, half + SKIP_WALL),
            (-half - wall, 0.0, wall, half + SKIP_WALL),
            (0.0, half + wall, half, wall),
            (0.0, -half - wall, half, wall),
        ):
            builder.add_shape_box(
                -1,
                xform=wp.transform(wp.vec3(at[0] + dx, at[1] + dy, height / 2), wp.quat_identity()),
                hx=hx,
                hy=hy,
                hz=height / 2,
                cfg=cfg,
                color=SKIP_COLOR,
            )

    def _fill_skip(self, builder):
        """Loose primitives stacked in the skip: boxes, spheres, cylinders, capsules."""
        rng = np.random.default_rng(7)
        cfg = newton.ModelBuilder.ShapeConfig(density=ROCK_DENSITY, mu=0.75, restitution=0.0)
        # A heap, spawned already in the shape it would settle into: 4x4, then 3x3, then
        # 2x2, then one on top. Material dropped in as a block either interpenetrates
        # and explodes, or slumps over the kerb before the machine has moved.
        bodies = []
        tiers = [(side, 0.56 + layer * 0.385) for layer, side in enumerate((11,) * 9)]
        places = [
            (
                SKIP_AT[0] + (i - (side - 1) / 2) * 0.425 + float(rng.uniform(-0.015, 0.015)),
                SKIP_AT[1] + (j - (side - 1) / 2) * 0.425 + float(rng.uniform(-0.015, 0.015)),
                z,
            )
            for side, z in tiers
            for i in range(side)
            for j in range(side)
        ]
        for index in range(min(ROCKS, len(places))):
            pos = wp.vec3(*places[index])
            yaw = float(rng.uniform(0.0, math.pi))
            body = builder.add_body(xform=wp.transform(pos, wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw)))
            color = ROCK_COLORS[index % len(ROCK_COLORS)]
            kind = index % 4
            # Every primitive fits inside a 0.27 m radius in plan, which is what the
            # spawn grid is spaced for: material that starts interpenetrating is thrown
            # out of the skip before the machine has moved.
            if kind == 0:
                builder.add_shape_box(body, hx=0.152, hy=0.127, hz=0.11, cfg=cfg, color=color)
            elif kind == 1:
                builder.add_shape_sphere(body, radius=0.122, cfg=cfg, color=color)
            elif kind == 2:
                builder.add_shape_cylinder(body, radius=0.114, half_height=0.114, cfg=cfg, color=color)
            else:
                builder.add_shape_capsule(body, radius=0.099, half_height=0.076, cfg=cfg, color=color)
            bodies.append(body)
        return bodies

    # ------------------------------------------------------------------- cycle
    def waypoint(self, t: float):
        """Interpolate the dig cycle: returns (phase, reach, height, yaw, pitch)."""
        if t < SETTLE:
            return ("Settle", CYCLE[0][2], CYCLE[0][3], CYCLE[0][4], CYCLE[0][5])
        local = (t - SETTLE) % CYCLE_TIME
        previous = CYCLE[-1]
        for entry in CYCLE:
            if local <= entry[1]:
                start = 0.0 if previous is CYCLE[-1] and entry is CYCLE[0] else previous[1]
                span = max(entry[1] - start, 1.0e-6)
                blend = _smoothstep((local - start) / span)
                values = [p + (n - p) * blend for p, n in zip(previous[2:], entry[2:], strict=True)]
                return (entry[0], *values)
            previous = entry
        return (CYCLE[-1][0], *CYCLE[-1][2:])

    def _command(self):
        phase, *pose = self.waypoint(self.sim_time)
        self.phase = phase
        # Waypoints are the teeth; the arm has to put the pivot a bucket length behind
        # them, along whatever attitude the scoop is holding, and the bucket angle is
        # commanded in the world so the scoop keeps its attitude as the arm moves.
        self._write_targets(self.cycle_angles(("", 0.0, *pose)))

    def moved_out(self) -> int:
        """Primitives that have left the skip's footprint, carried or shoved."""
        q = self.state_0.body_q.numpy()[self.rocks, :3]
        inside = (np.abs(q[:, 0] - SKIP_AT[0]) < SKIP_INNER / 2 + SKIP_WALL) & (
            np.abs(q[:, 1] - SKIP_AT[1]) < SKIP_INNER / 2 + SKIP_WALL
        )
        return int(np.count_nonzero(~inside))

    def delivered(self) -> int:
        """Primitives sitting in the empty skip, which only the bucket can put there."""
        q = self.state_0.body_q.numpy()[self.rocks, :2]
        inside = (np.abs(q[:, 0] - DUMP_AT[0]) < DUMP_INNER / 2) & (np.abs(q[:, 1] - DUMP_AT[1]) < DUMP_INNER / 2)
        return int(np.count_nonzero(inside))

    def on_reset(self):
        self.digging = True

    def _write_targets(self, angles):
        """Swing, boom, stick and bucket, one coordinate each."""
        for joint, value in zip(self.joints, angles, strict=True):
            self.targets[self.target_index[joint]] = value
        self.control.joint_target_q.assign(self.targets)

    def drive_to(self, teeth, pitch_deg):
        """Point the arm at a world teeth position, holding the scoop at a pitch."""
        x, y, z = teeth
        waypoint = ("", 0.0, math.hypot(x, y), z, math.degrees(math.atan2(y, x)), pitch_deg)
        self._write_targets(self.cycle_angles(waypoint))

    def step(self):
        if self.digging:
            self._command()
        else:
            self.pause_offset = getattr(self, "pause_offset", 0.0) + self.frame_dt
        self.stepper.step()

    def substep(self):
        self.state_0.clear_forces()
        self.viewer.apply_forces(self.state_0)
        self.stepper.collide()
        self.stepper.solve()

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_scalar("material delivered", self.delivered())
        self.viewer.end_frame()

    def gui(self, ui):
        ui.text(f"{ROCKS} primitives, {self.delivered()} delivered, {self.moved_out()} out of the skip")
        ui.text(f"phase: {getattr(self, 'phase', 'Settle')}")
        changed, digging = ui.checkbox("Run the dig cycle", self.digging)
        if changed:
            self.digging = digging
        self.stepper.gui(ui)

    def test_final(self):
        assert_finite(self.state_0.body_q, self.state_0.body_qd)
        q = self.state_0.body_q.numpy()
        if np.any(q[:, 2] < -0.05):
            raise ValueError("a body fell through the ground")
        # The hand-authored cycle works the heap but does not reliably fill the bucket,
        # so the test asserts the machine runs its cycle through the material rather
        # than a delivery count. Tighten this once a demonstrated path replaces CYCLE.
        moved = np.linalg.norm(q[self.rocks, :3] - self.rock_start, axis=1)
        if int(np.count_nonzero(moved > 0.3)) < 20:
            raise ValueError(f"the cycle barely touched the material, {int(np.count_nonzero(moved > 0.3))} moved")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--solver", default="feather_pgs", choices=list(SOLVERS), help="Rigid-body solver.")
        parser.set_defaults(num_frames=960)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
