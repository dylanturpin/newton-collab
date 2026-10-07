# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run a sixty-object kitchen pour and scripted bin spill on CUDA FPGS.

The complete evolving rollout is timed after kernel compilation and graph
capture. All released objects are dynamic. Only the bin follows a prescribed
motion; object poses are recorded from FPGS and never taken from the reference.
"""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS


@wp.func
def _smooth(t: float, start: float, end: float):
    u = wp.clamp((t - start) / (end - start), 0.0, 1.0)
    return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))


@wp.func
def _bin_pose(t: float, support_height: float):
    angle = 2.15 * _smooth(t, 14.0, 16.0) - 2.15 * _smooth(t, 20.0, 21.5)
    q = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), angle)
    pivot = wp.vec3(0.17, 0.0, 0.001 + support_height)
    p = pivot - wp.quat_rotate(q, wp.vec3(0.17, 0.0, 0.0))
    p += wp.vec3(
        -0.55 * _smooth(t, 17.0, 20.0),
        0.08 * _smooth(t, 17.0, 20.0),
        wp.max(0.0, -0.65 * wp.cos(angle)) + 0.42 * _smooth(t, 17.0, 18.5) - 0.42 * _smooth(t, 21.5, 22.5),
    )
    return wp.transform(p, q)


@wp.kernel
def _prepare(
    tick: wp.array[int],
    dt: float,
    support_height: float,
    release_ticks: wp.array[int],
    release_poses: wp.array[wp.transform],
    radii: wp.array[float],
    com: wp.array[wp.vec3],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    release_height: wp.array[float],
):
    body = wp.tid()
    step = tick[0]
    if body == 0:
        t = float(step) * dt
        pose, next_pose = _bin_pose(t, support_height), _bin_pose(t + dt, support_height)
        p, q = wp.transform_get_translation(pose), wp.transform_get_rotation(pose)
        pn, qn = wp.transform_get_translation(next_pose), wp.transform_get_rotation(next_pose)
        v = (pn + wp.quat_rotate(qn, com[body]) - p - wp.quat_rotate(q, com[body])) / dt
        delta = qn * wp.quat_inverse(q)
        omega = 2.0 * wp.vec3(delta[0], delta[1], delta[2]) / dt
        body_q[body] = pose
        body_qd[body] = wp.spatial_vector(v, omega)
    elif step <= release_ticks[body]:
        pose = wp.transform(wp.vec3(4.0 + float(body) * 0.5, 4.0, 0.5), wp.quat_identity())
        if step == release_ticks[body]:
            pose = release_poses[body]
            p, q = wp.transform_get_translation(pose), wp.transform_get_rotation(pose)
            center = p + wp.quat_rotate(q, com[body])
            height = center[2]
            # Raise a birth above the live pile using conservative COM bounds.
            # The fixed release order and times are preserved, without overlaps.
            for other in range(1, body_q.shape[0]):
                if other != body and release_ticks[other] < step:
                    other_center = wp.transform_point(body_q[other], com[other])
                    delta_xy = wp.vec2(center[0] - other_center[0], center[1] - other_center[1])
                    if wp.length(delta_xy) < radii[body] + radii[other] + 0.005:
                        height = wp.max(height, other_center[2] + radii[other] + radii[body] + 0.005)
            p[2] += height - center[2]
            pose = wp.transform(p, q)
            release_height[body] = p[2]
        body_q[body] = pose
        body_qd[body] = wp.spatial_vector()


@wp.kernel
def _advance(tick: wp.array[int]):
    tick[0] += 1


@wp.kernel
def _contact_peak(contacts: wp.array[int], peak: wp.array[int]):
    peak[0] = wp.max(peak[0], contacts[0])


@wp.kernel
def _record(
    tick: wp.array[int],
    substeps: int,
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    contacts: wp.array[int],
    poses: wp.array2d[wp.transform],
    velocities: wp.array2d[wp.spatial_vector],
    contact_counts: wp.array[int],
):
    body = wp.tid()
    frame = tick[0] // substeps
    if frame < poses.shape[0]:
        poses[frame, body] = body_q[body]
        velocities[frame, body] = body_qd[body]
        if body == 0:
            contact_counts[frame] = contacts[0]


def _build_model(asset, document, device, voxel, cache, support, support_height):
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.gap = 0.0005
    builder.default_shape_cfg.mu = 0.5
    builder.default_shape_cfg.restitution = 0.02
    floor = builder.default_shape_cfg.copy()
    floor.density = 0.0
    if support == "floor":
        builder.add_ground_plane(height=support_height, cfg=floor, label="checkerboard_floor")
    else:
        builder.add_shape_box(
            -1,
            xform=wp.transform(wp.vec3(0.2, 0.0, -0.02), wp.quat_identity()),
            hx=1.1,
            hy=0.8,
            hz=0.02,
            cfg=floor,
            color=(0.27, 0.32, 0.36),
            label="table",
        )
    box = builder.add_body(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.001 + support_height), wp.quat_identity()),
        is_kinematic=True,
        label="bin",
    )
    box_cfg = builder.default_shape_cfg.copy()
    box_cfg.density, box_cfg.mu = 600.0, 0.35
    specs = [
        ((0, 0, 0.006), (0.17, 0.17, 0.006)),
        ((-0.164, 0, 0.331), (0.006, 0.17, 0.319)),
        ((0.164, 0, 0.331), (0.006, 0.17, 0.319)),
        ((0, -0.164, 0.331), (0.158, 0.006, 0.319)),
        ((0, 0.164, 0.331), (0.158, 0.006, 0.319)),
    ]
    for index, (position, size) in enumerate(specs):
        builder.add_shape_box(
            box,
            xform=wp.transform(wp.vec3(*position), wp.quat_identity()),
            hx=size[0],
            hy=size[1],
            hz=size[2],
            cfg=box_cfg,
            color=(0.58, 0.74, 0.91),
            opacity=0.2,
            label=f"bin_panel_{index}",
        )
    meshes = {}
    for kind, item in document["bank"].items():
        part = item["parts"][0]
        if len(item["parts"]) == 1 and part["op"] in ("ellipsoid", "sdf_capsule"):
            continue
        with np.load(asset / f"{item['mesh']}.npz") as data:
            mesh = newton.Mesh(data["vertices"], data["faces"].ravel(), compute_inertia=False)
        print(f"Preparing collision SDF: {kind}", flush=True)
        mesh.build_sdf(
            device=device,
            target_voxel_size=voxel,
            narrow_band_range=(-0.003, 0.003),
            margin=0.005,
            sign_method="winding",
            cache_dir=cache,
        )
        meshes[kind] = mesh
    for index, item in enumerate(document["bodies"]):
        kind = document["bank"][item["kind"]]
        body = builder.add_body(
            xform=wp.transform(wp.vec3(4.0 + (index + 1) * 0.5, 4.0, 0.5), wp.quat_identity()),
            mass=kind["mass"],
            com=wp.vec3(kind["com"]),
            inertia=wp.mat33(np.asarray(kind["inertia"])),
            lock_inertia=True,
            label=f"{index:02d}_{item['kind'].replace('/', '_')}",
        )
        cfg = builder.default_shape_cfg.copy()
        cfg.density = 0.0
        part = kind["parts"][0]
        if len(kind["parts"]) == 1 and part["op"] == "ellipsoid":
            builder.add_shape_sphere(body, radius=part["radius"], cfg=cfg, color=kind["color"])
        elif len(kind["parts"]) == 1 and part["op"] == "sdf_capsule":
            builder.add_shape_capsule(
                body,
                radius=part["radius"],
                half_height=float(np.linalg.norm(np.asarray(part["b"]) - part["a"])) / 2,
                cfg=cfg,
                color=kind["color"],
            )
        else:
            builder.add_shape_mesh(body, mesh=meshes[item["kind"]], cfg=cfg, color=kind["color"])
    print("Finalizing rigid-body model", flush=True)
    return builder.finalize(device=device)


class Example:
    def __init__(self, args):
        self.args = args
        self.document = json.loads((args.asset / "scene.json").read_text())
        self.frames = math.ceil(args.duration * 60)
        self.dt = 1.0 / (60 * args.substeps)
        self.support_height = 0.005 if args.support == "floor" else 0.0
        self.model = _build_model(
            args.asset, self.document, args.device, args.voxel, args.cache / "sdf", args.support, self.support_height
        )
        self.model.rigid_contact_max = args.contacts
        print("Creating collision pipeline and colored FPGS solver", flush=True)
        self.pipeline = newton.CollisionPipeline(
            self.model,
            rigid_contact_max=args.contacts,
            broad_phase="sap",
            reduce_contacts=True,
            contact_matching="disabled",
            max_triangle_pairs=1_000_000,
            include_static_kinematic_pairs=False,
        )
        self.contacts = self.pipeline.contacts()
        self.solver = SolverFeatherPGS(
            self.model,
            pgs_mode="matrix_free",
            articulated_contact_response="propagation-colored",
            pgs_iterations=args.iterations,
            # Colored propagation absorbs both requested row budgets. This
            # all-free-body scene needs no legacy matrix-free contact rows;
            # keep that shared-memory allocation small and reserve contacts
            # through the dense budget (internally reduced to 16 dense rows).
            dense_max_constraints=3 * args.contacts,
            mf_max_constraints=64,
            angular_damping=0.02,
            friction_anchor_beta=0.0,
            contact_torsion_radius=0.0,
            row_watermark=True,
            pgs_warmstart=False,
            use_parallel_streams=False,
        )
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self.tick = wp.zeros(1, dtype=wp.int32, device=args.device)
        releases = [-1] + [round(b["release_s"] / self.dt) for b in self.document["bodies"]]
        self.release_ticks = wp.array(releases, dtype=wp.int32, device=args.device)
        initial = [wp.transform_identity()] + [
            wp.transform(wp.vec3(b["position"]) + wp.vec3(0.0, 0.0, self.support_height), wp.quat(b["rotation_xyzw"]))
            for b in self.document["bodies"]
        ]
        self.release_poses = wp.array(initial, dtype=wp.transform, device=args.device)
        radii = [0.0] + [self.document["bank"][b["kind"]]["radius"] for b in self.document["bodies"]]
        self.radii = wp.array(radii, dtype=float, device=args.device)
        self.release_height = wp.zeros(self.model.body_count, dtype=float, device=args.device)
        self.poses = wp.zeros((self.frames + 1, self.model.body_count), dtype=wp.transform, device=args.device)
        self.velocities = wp.zeros(
            (self.frames + 1, self.model.body_count), dtype=wp.spatial_vector, device=args.device
        )
        self.contact_counts = wp.zeros(self.frames + 1, dtype=int, device=args.device)
        self.contact_peak = wp.zeros(1, dtype=int, device=args.device)
        self.reset()

    def reset(self):
        for state in (self.state_0, self.state_1):
            wp.copy(state.joint_q, self.model.joint_q)
            wp.copy(state.joint_qd, self.model.joint_qd)
            newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)
            state.clear_forces()
        self.tick.zero_()
        self.contact_peak.zero_()
        self.solver.reset(self.state_0)
        self.pipeline.reset_contact_matching()
        self.prepare()
        self.record()

    def prepare(self):
        wp.launch(
            _prepare,
            dim=self.model.body_count,
            inputs=[
                self.tick,
                self.dt,
                self.support_height,
                self.release_ticks,
                self.release_poses,
                self.radii,
                self.model.body_com,
            ],
            outputs=[self.state_0.body_q, self.state_0.body_qd, self.release_height],
            device=self.model.device,
        )
        newton.eval_ik(self.model, self.state_0, self.state_0.joint_q, self.state_0.joint_qd)

    def record(self):
        wp.launch(
            _record,
            dim=self.model.body_count,
            inputs=[
                self.tick,
                self.args.substeps,
                self.state_0.body_q,
                self.state_0.body_qd,
                self.contacts.rigid_contact_count,
            ],
            outputs=[self.poses, self.velocities, self.contact_counts],
            device=self.model.device,
        )

    def simulate(self):
        for _ in range(self.args.substeps):
            self.prepare()
            self.state_0.clear_forces()
            self.pipeline.collide(self.state_0, self.contacts, dt=self.dt)
            wp.launch(
                _contact_peak,
                dim=1,
                inputs=[self.contacts.rigid_contact_count],
                outputs=[self.contact_peak],
                device=self.model.device,
            )
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            wp.launch(_advance, dim=1, inputs=[self.tick], device=self.model.device)
        self.record()

    def capture(self):
        # Compile both state-buffer phases before capture, then restore births.
        self.simulate()
        self.simulate()
        wp.synchronize_device(self.model.device)
        self.reset()
        with wp.ScopedCapture(device=self.model.device) as capture:
            self.solver.seed_double_buffer_events()
            self.simulate()
        self.graph = capture.graph
        self.reset()
        wp.synchronize_device(self.model.device)

    def test_final(self, poses, velocities):
        if not np.isfinite(poses).all() or not np.isfinite(velocities).all():
            raise AssertionError("Nonfinite body state in recorded rollout")
        self.solver.check_constraint_capacity()
        contact_peak = int(self.contact_peak.numpy()[0])
        if contact_peak > self.args.contacts:
            raise AssertionError(f"Contact buffer overflow: {contact_peak} > {self.args.contacts}")
        norm_error = float(np.max(abs(np.linalg.norm(poses[:, :, 3:], axis=2) - 1)))
        if norm_error > 1e-3:
            raise AssertionError(f"Invalid rotation quaternion: {norm_error}")
        q, com = poses[:, :, 3:], self.model.body_com.numpy()
        centers = poses[:, :, :3] + com + 2 * np.cross(q[:, :, :3], np.cross(q[:, :, :3], com) + q[:, :, 3:] * com)
        interior = (centers[:, 1:, 0] > -0.8) & (centers[:, 1:, 0] < 1.2) & (abs(centers[:, 1:, 1]) < 0.7)
        support_mask = np.ones_like(interior) if self.args.support == "floor" else interior
        below_support = support_mask & (centers[:, 1:, 2] < self.support_height - 0.06)
        if below_support.any():
            raise AssertionError(f"A body passed below the {self.args.support} support surface")
        result = {
            "finite": True,
            "max_quaternion_norm_error": norm_error,
            "max_contacts_over_all_substeps": contact_peak,
            "bodies_below_support": 0,
            "support": self.args.support,
            "support_height_m": self.support_height,
            "capacity": self.solver.constraint_row_watermarks(),
        }
        for name, t in (("before_tip", 13.5), ("after_spill", 23.0)):
            if t > self.args.duration:
                continue
            frame = round(t * 60)
            delta = centers[frame, 1:] - poses[frame, 0, :3]
            inverse = -q[frame, 0, :3]
            local = delta + 2 * np.cross(inverse, np.cross(inverse, delta) + q[frame, 0, 3] * delta)
            inside = (abs(local[:, 0]) < 0.17) & (abs(local[:, 1]) < 0.17) & (local[:, 2] > 0) & (local[:, 2] < 0.75)
            result[f"objects_in_bin_{name}"] = int(inside.sum())
        return result

    def run(self):
        print("Compiling and capturing simulation frame", flush=True)
        self.capture()
        print("Starting timed rollout", flush=True)
        started = time.perf_counter()
        chunks = []
        for frame in range(self.frames):
            wp.capture_launch(self.graph)
            if (frame + 1) % 120 == 0 or frame + 1 == self.frames:
                wp.synchronize_device(self.model.device)
                elapsed = time.perf_counter() - started
                chunks.append({"sim_time_s": (frame + 1) / 60, "wall_s": elapsed})
                print(
                    json.dumps(
                        {
                            "sim_s": (frame + 1) / 60,
                            "wall_s": round(elapsed, 3),
                            "realtime_factor": round((frame + 1) / 60 / elapsed, 3),
                        }
                    ),
                    flush=True,
                )
        elapsed = time.perf_counter() - started
        poses, velocities = self.poses.numpy(), self.velocities.numpy()
        np.savez_compressed(
            self.args.output / "trace.npz",
            time_s=np.arange(self.frames + 1) / 60,
            poses=poses,
            velocities=velocities,
            contact_counts=self.contact_counts.numpy(),
            release_height=self.release_height.numpy(),
        )
        validation = self.test_final(poses, velocities)
        result = {
            "device": str(self.model.device),
            "gpu": self.model.device.name,
            "warp": wp.__version__,
            "bodies": self.model.body_count,
            "objects": len(self.document["bodies"]),
            "shapes": self.model.shape_count,
            "solver": "FPGS",
            "response": self.solver.articulated_contact_response,
            "iterations": self.args.iterations,
            "substeps": self.args.substeps,
            "dt_s": self.dt,
            "voxel_m": self.args.voxel,
            "contact_capacity": self.args.contacts,
            "propagation_row_capacity": self.solver.propagation_max_constraints,
            "support": self.args.support,
            "support_height_m": self.support_height,
            "duration_s": self.frames / 60,
            "wall_s": elapsed,
            "realtime_factor": self.frames / 60 / elapsed,
            "frames_per_wall_second": self.frames / elapsed,
            "ms_per_simulated_frame": elapsed * 1000 / self.frames,
            "timing_scope": "evolving full collision + solver + release/bin control + GPU pose recording + graph launch; compilation, initial SDF cook, output copy and video encoding excluded",
            "chunks": chunks,
            "validation": validation,
            "asset_sha256": hashlib.sha256((self.args.asset / "scene.json").read_bytes()).hexdigest(),
            "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        (self.args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--support", choices=("table", "floor"), default="table")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--duration", default=23.0, type=float)
    parser.add_argument("--iterations", default=10, type=int)
    parser.add_argument("--substeps", default=4, type=int)
    parser.add_argument("--voxel", default=0.00075, type=float)
    parser.add_argument("--contacts", default=8192, type=int)
    args = parser.parse_args()
    if args.duration <= 0 or args.iterations < 1 or args.substeps < 2 or args.substeps % 2:
        parser.error("Use a positive duration/iteration count and an even substep count >= 2")
    args.output.mkdir(parents=True, exist_ok=False)
    args.cache.mkdir(parents=True, exist_ok=True)
    wp.config.kernel_cache_dir = str(args.cache / "warp")
    wp.init()
    if not wp.get_device(args.device).is_cuda:
        parser.error("This runner requires an NVIDIA CUDA device")
    with wp.ScopedDevice(args.device):
        Example(args).run()


if __name__ == "__main__":
    main()
