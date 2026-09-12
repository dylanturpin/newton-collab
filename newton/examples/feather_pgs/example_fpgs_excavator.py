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

import atexit
import math
import os
import pathlib
import sys
import time
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

# Teleop. The kit resolves operator intent into this host's world frame and latches
# its reference from the pose the host reports, so what arrives is the phone's
# displacement from the machine's rest pose, in world axes. A phone works in a box a
# few tens of centimetres across and this machine reaches seven metres, so that
# displacement is amplified about the rest point; the axes are the kit's, only the
# gain is ours. Tilting the phone drives the one end-effector freedom worth having,
# the bucket's pitch, and nothing is bound to the volume buttons.
TELEOP_GAIN = 6.0
# One pose tick has to fit a 1400-byte datagram, so only this many bodies can stream
# their motion to the phone. The machine always does; the rest of the budget goes to
# material, and everything past it is mirrored as static scenery rather than silently
# overflowing the packet and leaving the phone with nothing to draw.
TELEOP_TRACKED_BODIES = 0  # 0 streams every body
TELEOP_PATH = os.environ.get(
    "TELEOPKIT_PYTHON", str(pathlib.Path.home() / "Documents/github/teleopkit/bindings/python")
)

STEEL = wp.vec3(0.85, 0.72, 0.15)
DARK = wp.vec3(0.22, 0.23, 0.25)
SKIP_COLOR = wp.vec3(0.45, 0.47, 0.5)
ROCK_COLORS = (
    wp.vec3(0.55, 0.52, 0.48),
    wp.vec3(0.42, 0.40, 0.38),
    wp.vec3(0.62, 0.55, 0.45),
    wp.vec3(0.35, 0.38, 0.42),
)


def _matrix(quat) -> np.ndarray:
    """Rotation matrix from an xyzw quaternion."""
    x, y, z, w = (float(v) for v in quat)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
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


class _TeleopDriver:
    """Drive the bucket from a phone through TeleopKit, following its examples.

    The kit resolves operator intent into this host's own world frame, so
    ``frame.target`` is a point in metres where the bucket's cutting edge should
    go: no scaling and no frame conversion. The clutch reference is latched from
    the pose reported by :meth:`teleop.Host.set_current_pose`, which is why the
    teeth pose is published every step and why recenter moves home before
    returning. The volume buttons curl the scoop.

    Everything the operator drives is recorded, so a demonstration can be read
    back as waypoints for :data:`CYCLE`.
    """

    def __init__(self, address=None):
        if TELEOP_PATH not in sys.path:
            sys.path.insert(0, TELEOP_PATH)
        try:
            import teleop  # noqa: PLC0415
        except ImportError as err:  # pragma: no cover - depends on a local checkout
            raise SystemExit(
                f"TeleopKit's Python binding was not importable from {TELEOP_PATH}.\n"
                "Point TELEOPKIT_PYTHON at <teleopkit>/bindings/python."
            ) from err
        self.teleop = teleop
        self.example = None
        self.home = (0.0, 0.0, 0.0)
        self.home_quat = (0.0, 0.0, 0.0, 1.0)
        self.home_pitch = 0.0
        # Latched from the phone the moment control is taken, not at launch: the
        # operator's zero is wherever they are standing when they engage.
        self.ref_target = None
        self.ref_quat = None
        self.teeth = self.home
        self.orientation = self.home_quat
        self.pitch = 0.0
        self.gain = TELEOP_GAIN
        self.track: list[tuple[float, float, float, float, float]] = []
        self.closed = False
        self.reported_at = 0.0
        self.peer_was_live = False
        self.mirror = None
        self._streamed: list[int] = []

        # The host comes up before the model is built, so the connect code is on
        # screen while Warp is still compiling rather than a minute later.
        self.host = teleop.Host(_TeleopFeedback(self), name="Newton Excavator")
        self.host.set_profile_preset("floating-gripper")
        self.host.on_recenter = self._go_home
        self._banner(address)
        atexit.register(self.close)

    def attach(self, example):
        """Bind to the built scene: latch its rest pose and mirror it to the phone."""
        self.example = example
        self.home, self.home_quat = example.teeth_pose()
        self.home_pitch = example.bucket_pitch()
        self.teeth = self.home
        self.orientation = self.home_quat
        self.pitch = self.home_pitch
        self._publish_scene()

    # ------------------------------------------------------------------ scene
    def _publish_scene(self):
        """Mirror the Newton model to the phone, so it renders without a window."""
        teleop, model = self.teleop, self.example.model
        body = model.shape_body.numpy()
        kind = model.shape_type.numpy()
        scale = model.shape_scale.numpy()
        local = model.shape_transform.numpy()
        colors = getattr(model, "shape_color", None)
        colors = colors.numpy() if colors is not None else None
        # The machine first, then as much material as the pose budget allows.
        machine = list(dict.fromkeys([*self.example.arm_bodies, self.example.bucket_body]))
        budget = [b for b in machine if b >= 0]
        for rock in self.example.rocks:
            if TELEOP_TRACKED_BODIES and len(budget) >= TELEOP_TRACKED_BODIES:
                break
            budget.append(rock)
        streamed = set(budget)
        self._streamed = budget
        objects = []
        for shape in range(len(body)):
            geometry, size = self._primitive(kind[shape], scale[shape])
            if geometry is None:
                continue
            rgba = (*(float(c) for c in colors[shape]), 1.0) if colors is not None else (0.7, 0.7, 0.72, 1.0)
            xform = local[shape]
            owner = int(body[shape])
            if owner in streamed:
                attach, local_xform = f"body{owner}", teleop.transform(tuple(xform[:3]), tuple(xform[3:7]), size)
            else:
                # Untracked: bake the body's current world pose into the item so it
                # still appears, just frozen where the scene started.
                attach = None
                local_xform = teleop.transform(*self._world_of(owner, xform), size)
            objects.append(teleop.SceneObject(geometry=geometry, body=attach, local=local_xform, color=rgba))
        eye, look = (-4.0, 0.0, 2.5), (2.0, 0.0, -1.0)
        self.mirror = teleop.SceneMirror(self.host)
        poses = self._poses()
        installed = self.mirror.publish(
            objects,
            tracked=poses,
            palm=f"body{self.example.bucket_body}",
            fpv_local=teleop.look_at(eye, look),
        )
        print(
            f"[teleop] scene mirror: {len(objects)} objects, {len(poses)} tracked bodies, "
            f"{'published' if installed else 'REJECTED (over the phone render limits)'}",
            flush=True,
        )

    def _primitive(self, kind, scale):
        """Newton shape to a TeleopKit primitive; box scale is the full size."""
        teleop = self.teleop
        geo = newton.GeoType
        if kind == geo.BOX:
            return teleop.ScenePrimitive(teleop.MESH_BOX), tuple(float(v) * 2.0 for v in scale[:3])
        if kind == geo.SPHERE:
            d = float(scale[0]) * 2.0
            return teleop.ScenePrimitive(teleop.MESH_SPHERE), (d, d, d)
        if kind == geo.CAPSULE:
            return teleop.ScenePrimitive(teleop.MESH_CAPSULE, float(scale[1]) * 2.0, float(scale[0])), (1.0, 1.0, 1.0)
        if kind == geo.CYLINDER:
            return teleop.ScenePrimitive(teleop.MESH_CYLINDER, float(scale[0]), float(scale[1]) * 2.0), (1.0, 1.0, 1.0)
        return None, None  # the ground plane and anything exotic stay off the wire

    def _world_of(self, owner, local):
        """Compose a shape's local transform with its body pose, for static mirroring."""
        if owner < 0:
            return tuple(local[:3]), tuple(local[3:7])
        bq = self.example.state_0.body_q.numpy()[owner]
        frame = wp.transform(wp.vec3(*bq[:3]), wp.quat(*bq[3:7]))
        composed = frame * wp.transform(wp.vec3(*local[:3]), wp.quat(*local[3:7]))
        return tuple(float(v) for v in composed.p), tuple(float(v) for v in composed.q)

    def _poses(self):
        q = self.example.state_0.body_q.numpy()
        return {
            f"body{index}": self.teleop.BodyPose(
                tuple(float(v) for v in q[index][:3]), tuple(float(v) for v in q[index][3:7])
            )
            for index in self._streamed
        }

    def _banner(self, prefer):
        codes = self.host.connect_codes()
        print("\n  TeleopKit host 'Newton Excavator' is live.", flush=True)
        if not codes:
            print("  No reachable IPv4 address; check the network.", flush=True)
            return
        if prefer:
            wanted = [c for c in codes if c.rsplit("/", 1)[-1].rsplit(":", 1)[0] == prefer]
            codes = wanted + [c for c in codes if c not in wanted]
        print(
            "  "
            + (
                "Bonjour is advertising it; scan this if the phone does not list it:"
                if self.host.bonjour_available
                else "Bonjour is unavailable here, so scan this to connect:"
            ),
            flush=True,
        )
        print(self.teleop.qr_terminal(codes[0]), flush=True)
        for code in codes:
            print(f"  {code}", flush=True)
        if len(codes) > 1:
            print("  Several addresses: pass --teleop-address to encode another.", flush=True)
        print("  Lay the phone flat, tap Set forward, then Start.\n", flush=True)

    # ------------------------------------------------------------------- loop
    def _go_home(self):
        """Recenter: put the whole scene back and be there before returning.

        The kit latches its clutch reference from the pose this returns, so the
        machine has to be standing in it already. Resetting the material too means
        the operator re-establishes the frame against a known state rather than
        against whatever the last attempt left behind.
        """
        self.ref_target = None
        self.ref_quat = None
        example = self.example
        if example is not None:
            example.stepper.reset_scene()
            self.home, self.home_quat = example.teeth_pose()
            self.home_pitch = example.bucket_pitch()
            self.teeth, self.orientation = self.home, self.home_quat
            example.drive_pose(self.teeth, self.orientation)
        return self.home, self.home_quat

    def service(self):
        """Pump the host and stream the scene, every rendered frame.

        This cannot live in the step path. The example loop only steps when the
        viewer says to, so a paused or throttled viewer would stop servicing the
        socket and the phone would sit there with no reply and nothing to draw.
        """
        if self.example is None:
            self.host.pump(timeout_ms=0)
            return
        pose, orientation = self.example.teeth_pose()
        self.host.set_current_pose(pose, orientation)
        self.host.pump(timeout_ms=0)
        if self.mirror is not None:
            self.mirror.update(self._poses())
        self._report(pose)

    def update(self):
        ex = self.example
        frame = self.host.latest()
        if not (frame and frame.engaged):
            # Control released: drop both references so the next engage starts from
            # the machine's rest pose and from wherever the phone happens to be.
            self.ref_target = None
            self.ref_quat = None
        else:
            if frame.target:
                if self.ref_target is None:
                    self.ref_target = tuple(float(v) for v in frame.target)
                    self.teeth = self.home
                else:
                    self.teeth = tuple(
                        h + self.gain * (float(t) - r)
                        for h, t, r in zip(self.home, frame.target, self.ref_target, strict=True)
                    )
            if frame.hand_target:
                # The wrist takes the phone's whole attitude, measured from the one it
                # had at engage, so the scoop turns and rolls as well as pitches.
                if self.ref_quat is None:
                    self.ref_quat = tuple(float(v) for v in frame.hand_target)
                delta = wp.quat(*frame.hand_target) * wp.quat_inverse(wp.quat(*self.ref_quat))
                target = delta * wp.quat(*self.home_quat)
                self.orientation = tuple(float(v) for v in target)
        for kind, _body, _point in self.host.take_events():
            if kind in ("reset", "new_round"):
                self._go_home()
        ex.phase = f"Teleop  bucket {ex.bucket_pitch():+.0f} deg"
        ex.drive_pose(self.teeth, self.orientation)
        x, y, z = self.teeth
        self.track.append((ex.sim_time, math.hypot(x, y), z, math.degrees(math.atan2(y, x)), self.pitch))

    def _report(self, tool):
        """Say whether the phone reaches us at all; silence hides a blocked port."""
        now = time.monotonic()
        if now - self.reported_at < 2.0:
            return
        self.reported_at = now
        status = self.host.status()
        if status["peer_live"] != self.peer_was_live:
            self.peer_was_live = status["peer_live"]
            print(f"[teleop] phone {'connected' if self.peer_was_live else 'gone'}", flush=True)
        if self.peer_was_live:
            print(
                f"[teleop] {status['rate_hz']} Hz engaged={status['engaged']} "
                f"teeth=({tool[0]:+.2f},{tool[1]:+.2f},{tool[2]:+.2f}) bucket={self.pitch:+.0f}",
                flush=True,
            )
        else:
            # The packet count separates "the phone cannot reach us" - a blocked UDP
            # port, or client isolation on the network - from "packets arrive but the
            # session has not come up".
            print(f"[teleop] waiting for a phone; {status['packets_received']} packets received", flush=True)

    def waypoints(self, count=8):
        """Read the recorded drive back as a CYCLE-shaped waypoint list."""
        if not self.track:
            return []
        step = max(1, len(self.track) // count)
        return [
            (f"Recorded {i}", round(row[0], 2), round(row[1], 2), round(row[2], 2), round(row[3], 1), round(row[4], 1))
            for i, row in enumerate(self.track[::step])
        ]

    def close(self):
        if self.closed:
            return
        self.closed = True
        lines = self.waypoints()
        if lines:
            out = pathlib.Path("excavator_demo_waypoints.py")
            out.write_text("CYCLE = (\n" + "".join(f"    {row!r},\n" for row in lines) + ")\n")
            print(f"wrote {len(lines)} demonstrated waypoints to {out}", flush=True)
        self.host.close()


class _TeleopFeedback:
    """What the phone shows and feels: grip state and the tool point."""

    def __init__(self, driver):
        self.driver = driver

    def current_pose(self):
        example = self.driver.example
        return example.teeth_pose() if example is not None else (self.driver.home, self.driver.home_quat)

    def control(self, frame):
        """The step loop polls ``latest()``; this hook only has to exist."""

    def event(self, kind, body, point):
        """Recenter is handled through ``Host.on_recenter``."""

    def feedback(self):
        return {"clutch_engaged": True, "tcp": self.driver.teeth}


class Example:
    def __init__(self, viewer, args):
        # Before anything slow, so the connect code is on screen immediately.
        self.teleop = _TeleopDriver(getattr(args, "teleop_address", None)) if getattr(args, "teleop", False) else None
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

        if self.teleop is not None:
            self.teleop.attach(self)

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
    def pose_angles(teeth, orientation):
        """Swing, boom, stick and bucket angles for a teeth position and attitude.

        The bucket has one freedom, so only the attitude's pitch is reachable: the
        scoop's forward axis is tipped to match, and the swing follows the target's
        own bearing.
        """
        forward = _matrix(orientation) @ np.array([1.0, 0.0, 0.0])
        pitch = math.degrees(math.atan2(-forward[2], math.hypot(forward[0], forward[1])))
        x, y, z = teeth
        return Example.angles_for(math.hypot(x, y), z, math.degrees(math.atan2(y, x)), pitch)

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

    def teeth_pose(self):
        """World position of the cutting edge, and the bucket's orientation."""
        bq = self.state_0.body_q.numpy()[self.bucket_body]
        frame = wp.transform(wp.vec3(*bq[:3]), wp.quat(*bq[3:7]))
        teeth = wp.transform_point(frame, wp.vec3(TEETH_LOCAL[0], 0.0, TEETH_LOCAL[1]))
        return (float(teeth[0]), float(teeth[1]), float(teeth[2])), tuple(float(v) for v in bq[3:7])

    def bucket_pitch(self) -> float:
        """The scoop's absolute attitude in degrees, positive teeth-down."""
        bq = self.state_0.body_q.numpy()[self.bucket_body]
        forward = wp.quat_rotate(wp.quat(*bq[3:7]), wp.vec3(1.0, 0.0, 0.0))
        return math.degrees(math.atan2(-float(forward[2]), math.hypot(float(forward[0]), float(forward[1]))))

    def _write_targets(self, angles):
        """Swing, boom, stick and bucket, one coordinate each."""
        for joint, value in zip(self.joints, angles, strict=True):
            self.targets[self.target_index[joint]] = value
        self.control.joint_target_q.assign(self.targets)

    def drive_pose(self, teeth, orientation):
        """Point the arm at a world teeth position with a full bucket orientation."""
        self._write_targets(self.pose_angles(teeth, orientation))

    def drive_to(self, teeth, pitch_deg):
        """Point the arm at a world teeth position, holding the scoop at a pitch."""
        x, y, z = teeth
        waypoint = ("", 0.0, math.hypot(x, y), z, math.degrees(math.atan2(y, x)), pitch_deg)
        self._write_targets(self.cycle_angles(waypoint))

    def step(self):
        if self.teleop is not None:
            self.teleop.update()
        elif self.digging:
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
        if self.teleop is not None:
            self.teleop.service()
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
        delivered = self.delivered()
        if delivered < 4:
            raise ValueError(f"the excavator delivered only {delivered} of {ROCKS} primitives to the empty skip")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--solver", default="feather_pgs", choices=list(SOLVERS), help="Rigid-body solver.")
        parser.add_argument("--teleop", action="store_true", help="Drive the bucket from a phone through TeleopKit.")
        parser.add_argument(
            "--teleop-address", default=None, help="Host IP the QR should encode, on a multi-homed machine."
        )
        parser.set_defaults(num_frames=960)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)
