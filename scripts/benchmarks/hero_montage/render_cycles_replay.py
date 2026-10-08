# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Render packed recorded poses on OptiX; no physics is executed here."""

import itertools
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Quaternion, Vector


def camera_pose(camera, t):
    keys = camera.get("keyframes")
    if not keys:
        keys = [dict(camera, time=0)]
        if all(k in camera for k in ("endPosition", "endTarget", "moveStart", "moveEnd")):
            keys = [
                dict(camera, time=camera["moveStart"]),
                {
                    "position": camera["endPosition"],
                    "target": camera["endTarget"],
                    "fov": camera.get("endFov", camera["fov"]),
                    "time": camera["moveEnd"],
                },
            ]
    if t <= keys[0]["time"]:
        return keys[0]
    for a, b in itertools.pairwise(keys):
        if t <= b["time"]:
            u = (t - a["time"]) / (b["time"] - a["time"])
            u = u**3 * (10 + u * (-15 + 6 * u))
            return {k: (1 - u) * np.asarray(a[k]) + u * np.asarray(b[k]) for k in ("position", "target", "fov")}
    return keys[-1]


def render(source, output, ffmpeg):
    started = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=str(source / "scene.blend"))
    scene = bpy.context.scene
    camera = json.loads((source / "camera.json").read_text())
    meta = json.loads((source / "metadata.json").read_text())
    motion = np.load(source / "motion.npz")
    positions, rotations = motion["positions"], motion["rotations"]
    controllers = [(obj, int(obj["recorded_body"])) for obj in bpy.data.objects if "recorded_body" in obj]
    assert len(controllers) == meta["body_controllers"]
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.get_devices()
    devices = [d for d in prefs.devices if d.type == "OPTIX"]
    device_index = int(os.environ.get("CYCLES_DEVICE_INDEX", "0"))
    assert device_index < len(devices), [(d.name, d.type) for d in prefs.devices]
    for device in prefs.devices:
        device.use = device == devices[device_index]
    scene.cycles.device = "GPU"
    scene.cycles.samples = meta["samples"]
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = True
    scene.cycles.denoising_use_gpu = True
    scene.cycles.max_bounces = 6
    scene.cycles.diffuse_bounces = 2
    scene.cycles.glossy_bounces = 3
    scene.cycles.transmission_bounces = 6
    scene.render.use_persistent_data = True
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.compression = 10
    frames = round(camera["duration"] * 30)
    folder = output / "frames"
    folder.mkdir(exist_ok=True)
    setup = time.monotonic() - started
    times = []
    for frame in range(frames):
        tick = time.monotonic()
        t = frame / 30
        sample = (camera.get("startTime", 0) + t * camera.get("playbackSpeed", 1)) * meta["recording_fps"]
        a = min(int(sample), len(positions) - 1)
        b = min(a + 1, len(positions) - 1)
        u = sample - a
        pos = (1 - u) * positions[a, :, :3] + u * positions[b, :, :3]
        qa, qb = rotations[a], rotations[b].copy()
        dot = np.sum(qa * qb, axis=1)
        qb[dot < 0] *= -1
        dot = np.abs(dot).clip(0, 1)
        theta = np.arccos(dot)
        small = theta < 1e-4
        denom = np.where(small, 1, np.sin(theta))
        wa = np.where(small, 1 - u, np.sin((1 - u) * theta) / denom)
        wb = np.where(small, u, np.sin(u * theta) / denom)
        quat = wa[:, None] * qa + wb[:, None] * qb
        quat /= np.linalg.norm(quat, axis=1)[:, None]
        for obj, body in controllers:
            obj.location = pos[body]
            x, y, z, w = quat[body]
            obj.rotation_quaternion = Quaternion((float(w), float(x), float(y), float(z)))
        cam = camera_pose(camera, t)
        scene.camera.location = cam["position"]
        scene.camera.rotation_euler = (
            (Vector(cam["target"]) - scene.camera.location).to_track_quat("-Z", "Y").to_euler()
        )
        scene.camera.data.lens = 36 / (2 * math.tan(math.radians(float(cam["fov"])) / 2))
        scene.render.filepath = str(folder / f"{frame:04d}.png")
        bpy.ops.render.render(write_still=True)
        times.append(time.monotonic() - tick)
        if frame % 30 == 0:
            print(f"PROGRESS {source.name} {frame + 1}/{frames} {times[-1]:.3f}s", flush=True)
    destination = output / (source.name + ".mp4")
    subprocess.run(
        [
            str(ffmpeg),
            "-v",
            "error",
            "-y",
            "-framerate",
            "30",
            "-i",
            str(folder / "%04d.png"),
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "17",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(destination),
        ],
        check=True,
    )
    subprocess.run([str(ffmpeg), "-v", "error", "-xerror", "-i", str(destination), "-f", "null", "-"], check=True)
    report = dict(
        meta,
        frames=frames,
        fps=30,
        width=1920,
        height=1080,
        renderer="Blender Cycles OptiX",
        device=devices[device_index].name,
        setup_seconds=setup,
        wall_seconds=time.monotonic() - started,
        mean_frame_seconds=float(np.mean(times)),
        steady_frame_seconds=float(np.median(times[1:])),
        simulation_steps_executed=0,
        camera=camera,
        decode_validated=True,
    )
    (output / "render-report.json").write_text(json.dumps(report, indent=2))
    print("COMPLETE", source.name, report["steady_frame_seconds"], flush=True)


if __name__ == "__main__":
    render(*map(Path, sys.argv[sys.argv.index("--") + 1 :]))
