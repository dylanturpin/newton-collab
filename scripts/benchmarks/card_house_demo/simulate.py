# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Record a friction-supported card house struck by a dynamic ball on CUDA."""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS


@wp.kernel
def prepare_ball(
    tick: wp.array[int],
    release: int,
    ball: int,
    speed: float,
    rolling: bool,
    q: wp.array[wp.transform],
    v: wp.array[wp.spatial_vector],
):
    if tick[0] < release:
        q[ball] = wp.transform(wp.vec3(-0.55, 0.0, 0.035), wp.quat_identity())
        v[ball] = wp.spatial_vector()
    elif tick[0] == release:
        q[ball] = wp.transform(wp.vec3(-0.55, 0.0, 0.035), wp.quat_identity())
        spin = float(0.0)
        if rolling:
            spin = speed / 0.035
        v[ball] = wp.spatial_vector(wp.vec3(speed, 0.0, 0.0), wp.vec3(0.0, spin, 0.0))


@wp.kernel
def advance(tick: wp.array[int], count: wp.array[int], peak: wp.array[int]):
    tick[0] += 1
    peak[0] = wp.max(peak[0], count[0])


@wp.kernel
def record(tick: wp.array[int], substeps: int, q: wp.array[wp.transform], poses: wp.array2d[wp.transform]):
    i = wp.tid()
    poses[tick[0] // substeps, i] = q[i]


class Example:
    def __init__(self, args):
        self.args = args
        self.dt = 1.0 / (60 * args.substeps)
        self.frames = round(60 * args.duration)
        self.bodies = []
        builder = newton.ModelBuilder()
        builder.default_shape_cfg.gap = args.gap
        builder.default_shape_cfg.mu = args.friction
        builder.default_shape_cfg.restitution = 0.0
        floor = builder.default_shape_cfg.copy()
        floor.density = 0.0
        builder.add_shape_box(
            -1, xform=wp.transform(wp.vec3(0, 0, -0.02), wp.quat_identity()), hx=1.0, hy=0.7, hz=0.02, cfg=floor
        )
        length, width, thick = 0.1, 0.065, args.thickness
        angle = math.radians(args.angle)
        pitch = 0.08
        height = length * math.cos(angle) + thick * math.sin(angle)
        colors = [(0.88, 0.25, 0.28), (0.21, 0.48, 0.83), (0.20, 0.65, 0.49)]

        def card(pos, theta, size, tier):
            rotation = wp.quat_from_axis_angle(wp.vec3(0, 1, 0), float(theta))
            body = builder.add_body(xform=wp.transform(wp.vec3(*pos), rotation), label=f"card_{len(self.bodies)}")
            cfg = builder.default_shape_cfg.copy()
            cfg.density = 650.0
            builder.add_shape_box(body, hx=size[0], hy=size[1], hz=size[2], cfg=cfg, color=colors[tier])
            self.bodies.append({"kind": "card", "size": size, "color": colors[tier], "tier": tier})

        def world_point(base, theta, x, z):
            return (
                base[0] + math.cos(theta) * x + math.sin(theta) * z,
                0.0,
                base[2] - math.sin(theta) * x + math.cos(theta) * z,
            )

        bases = [((x, 0.0, 0.0), 0.0) for x in (-pitch, 0.0, pitch)]
        for tier in range(3):
            peaks = []
            for base, base_angle in bases:
                offset = 0.5 * (length * math.sin(angle) + thick * math.cos(angle))
                z = 0.5 * height + 0.00002
                for sign in (-1, 1):
                    card(
                        world_point(base, base_angle, sign * offset, z),
                        base_angle - sign * angle,
                        (thick / 2, width / 2, length / 2),
                        tier,
                    )
                corners = [
                    world_point(base, base_angle, sign * thick * math.cos(angle), height + 0.00002) for sign in (-1, 1)
                ]
                peaks.append(max(corners, key=lambda p: p[2]))
            next_bases = []
            previous_deck = None
            for deck in range(len(bases) - 1):
                left, right = peaks[deck], peaks[deck + 1]
                if previous_deck is not None:
                    # The overlapping roof rests on the preceding card's edge.
                    # Account for its thickness instead of interpenetrating it.
                    left = world_point(previous_deck[0], previous_deck[1], length / 2, thick / 2)
                theta = -math.atan2(right[2] - left[2], right[0] - left[0])
                x = (peaks[deck][0] + peaks[deck + 1][0]) / 2
                z = left[2] - math.tan(theta) * (x - left[0]) + (thick / 2 + 0.00004) / math.cos(theta)
                center = (x, 0.0, z)
                card(center, theta, (length / 2, width / 2, thick / 2), tier)
                previous_deck = (center, theta)
                next_bases.append((world_point(center, theta, 0, thick / 2 + 0.00002), theta))
            bases = next_bases
        self.card_count = len(self.bodies)
        self.ball = builder.add_body(xform=wp.transform(wp.vec3(-0.55, 0, 0.035), wp.quat_identity()), label="ball")
        cfg = builder.default_shape_cfg.copy()
        cfg.mu = 0.18
        cfg.density = 350.0
        builder.add_shape_sphere(self.ball, radius=0.035, cfg=cfg, color=(0.96, 0.68, 0.13))
        self.bodies.append({"kind": "ball", "radius": 0.035, "color": (0.96, 0.68, 0.13)})
        self.model = builder.finalize(device=args.device)
        self.model.rigid_contact_max = 1024
        self.pipeline = newton.CollisionPipeline(
            self.model,
            rigid_contact_max=1024,
            broad_phase="sap",
            reduce_contacts=True,
            contact_matching="disabled",
            include_static_kinematic_pairs=False,
        )
        self.contacts = self.pipeline.contacts()
        self.solver = SolverFeatherPGS(
            self.model,
            pgs_mode="matrix_free",
            articulated_contact_response="propagation-colored",
            pgs_iterations=args.iterations,
            dense_max_constraints=3072,
            mf_max_constraints=64,
            angular_damping=0.02,
            friction_anchor_beta=0.2,
            contact_torsion_radius=0.0,
            row_watermark=True,
            pgs_warmstart=False,
            use_parallel_streams=False,
        )
        self.a, self.b = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.tick = wp.zeros(1, dtype=int, device=args.device)
        self.peak = wp.zeros(1, dtype=int, device=args.device)
        self.poses = wp.zeros((self.frames + 1, self.model.body_count), dtype=wp.transform, device=args.device)
        self.reset()

    def reset(self):
        for state in (self.a, self.b):
            wp.copy(state.joint_q, self.model.joint_q)
            wp.copy(state.joint_qd, self.model.joint_qd)
            newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)
            state.clear_forces()
        self.tick.zero_()
        self.peak.zero_()
        self.solver.reset(self.a)
        self.pipeline.reset_contact_matching()
        self.record()

    def record(self):
        wp.launch(
            record,
            dim=self.model.body_count,
            inputs=[self.tick, self.args.substeps, self.a.body_q, self.poses],
            device=self.model.device,
        )

    def simulate(self):
        for _ in range(self.args.substeps):
            wp.launch(
                prepare_ball,
                dim=1,
                inputs=[
                    self.tick,
                    round(self.args.impact / self.dt),
                    self.ball,
                    self.args.ball_speed,
                    self.args.rolling_ball,
                    self.a.body_q,
                    self.a.body_qd,
                ],
                device=self.model.device,
            )
            newton.eval_ik(self.model, self.a, self.a.joint_q, self.a.joint_qd)
            self.a.clear_forces()
            self.pipeline.collide(self.a, self.contacts, dt=self.dt)
            self.solver.step(self.a, self.b, self.control, self.contacts, self.dt)
            self.a, self.b = self.b, self.a
            wp.launch(
                advance,
                dim=1,
                inputs=[self.tick, self.contacts.rigid_contact_count, self.peak],
                device=self.model.device,
            )
        self.record()

    def test_final(self, poses):
        self.solver.check_constraint_capacity()
        before = poses[round((self.args.impact - 0.1) * 60), : self.card_count]
        initial = poses[0, : self.card_count]
        displacement = np.linalg.norm(before[:, :3] - initial[:, :3], axis=1)
        tilt = 2 * np.arccos(np.clip(abs(np.sum(before[:, 3:] * initial[:, 3:], axis=1)), 0, 1))
        after = poses[-1, : self.card_count]
        fallen = abs(after[:, 2] - before[:, 2]) > 0.02
        changed = np.linalg.norm(after[:, :3] - before[:, :3], axis=1) > 0.03
        result = {
            "finite": bool(np.isfinite(poses).all()),
            "max_preimpact_displacement_m": float(displacement.max()),
            "max_preimpact_rotation_deg": float(np.degrees(tilt.max())),
            "cards_moved_after_impact": int(changed.sum()),
            "cards_changed_height_after_impact": int(fallen.sum()),
            "max_contacts": int(self.peak.numpy()[0]),
            "min_card_center_height_m": float(poses[:, : self.card_count, 2].min()),
        }
        result["standing_before_impact"] = bool(displacement.max() < 0.015 and tilt.max() < math.radians(15))
        result["destroyed_after_impact"] = bool(changed.sum() >= self.card_count // 2 and fallen.sum() >= 4)
        return result

    def run(self):
        print("Compiling CUDA frame", flush=True)
        self.simulate()
        self.simulate()
        wp.synchronize_device(self.model.device)
        self.reset()
        with wp.ScopedCapture(device=self.model.device) as capture:
            self.solver.seed_double_buffer_events()
            self.simulate()
        self.reset()
        wp.synchronize_device(self.model.device)
        start = time.perf_counter()
        for _ in range(self.frames):
            wp.capture_launch(capture.graph)
        wp.synchronize_device(self.model.device)
        elapsed = time.perf_counter() - start
        poses = self.poses.numpy()
        self.args.output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(self.args.output / "trace.npz", poses=poses, time_s=np.arange(self.frames + 1) / 60)
        result = {
            "solver": "FPGS",
            "response": "propagation-colored",
            "friction_anchor_beta": 0.2,
            "gpu": self.model.device.name,
            "substeps": self.args.substeps,
            "iterations": self.args.iterations,
            "cards": self.card_count,
            "duration_s": self.frames / 60,
            "launch_s": self.args.impact,
            "ball_speed_m_s": self.args.ball_speed,
            "rolling_ball": self.args.rolling_ball,
            "wall_s": elapsed,
            "ms_per_frame": 1000 * elapsed / self.frames,
            "realtime_factor": self.frames / 60 / elapsed,
            "timing_scope": "collision, solve, ball launch control, GPU recording and graph launches; excludes compilation and rendering",
            "thickness_m": self.args.thickness,
            "contact_gap_m": self.args.gap,
            "lean_angle_deg": self.args.angle,
            "validation": self.test_final(poses),
            "bodies": self.bodies,
        }
        (self.args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        if not all(result["validation"][key] for key in ("finite", "standing_before_impact", "destroyed_after_impact")):
            raise RuntimeError("Card-house validation failed; inspect recorded poses before presenting this run")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--impact", type=float, default=3.0)
    parser.add_argument("--ball-speed", type=float, default=2.2)
    parser.add_argument("--rolling-ball", action="store_true")
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--thickness", type=float, default=0.001)
    parser.add_argument("--friction", type=float, default=0.8)
    parser.add_argument("--gap", type=float, default=0.0005)
    parser.add_argument("--angle", type=float, default=20.0)
    args = parser.parse_args()
    if args.substeps < 2 or args.substeps % 2:
        parser.error("CUDA graph replay requires an even number of substeps")
    Example(args).run()
