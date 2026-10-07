# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render saved rigid-benchmark poses to MP4 without rerunning dynamics.

Run with ``uv run --extra dev --with pillow --with imageio-ffmpeg python -m
scripts.benchmarks.render_manda_rigid --input /path/to/manda-run``.
MuJoCo supplies rendering and forward kinematics only; poses come from the
selected backend's trace. Display geometry is simplified and noncolliding.
"""

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio_ffmpeg
import numpy as np

SCENES = ("grasp", "stack", "push", "collision", "slide", "drop", "hinge", "panda_effort")
TITLES = {
    "slide": "Sliding cube",
    "drop": "Drop onto a rigid floor",
    "hinge": "Driven pendulum",
    "collision": "Sphere / cube / prism collisions",
    "panda_effort": "Panda effort replay",
    "grasp": "Grasp, lift, hold, release",
    "stack": "Pick and stack",
    "push": "Push three blocks through a channel",
}


def _display_xml(xml, width, height):
    """Add lights, materials and noncolliding gripper visuals to a render copy."""
    root = ET.fromstring(xml)
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth=str(width), offheight=str(height))
    ET.SubElement(visual, "headlight", ambient="0.2 0.2 0.2", diffuse="0.55 0.55 0.55", specular="0.08 0.08 0.08")
    ET.SubElement(visual, "quality", shadowsize="4096")
    ET.SubElement(visual, "map", znear="0.0001", zfar="20")
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        type="skybox",
        builtin="gradient",
        rgb1="0.90 0.94 0.98",
        rgb2="0.97 0.98 1",
        width="512",
        height="512",
    )
    ET.SubElement(
        asset,
        "texture",
        name="render_floor_texture",
        type="2d",
        builtin="checker",
        rgb1="0.84 0.88 0.93",
        rgb2="0.93 0.95 0.98",
        width="512",
        height="512",
    )
    ET.SubElement(asset, "material", name="render_floor", texture="render_floor_texture", texrepeat="20 20")
    world = root.find("worldbody")
    ET.SubElement(
        world,
        "light",
        pos="1 -1 2",
        dir="-0.3 0.3 -1",
        directional="true",
        diffuse="0.4 0.4 0.4",
        ambient="0.06 0.06 0.06",
        castshadow="true",
    )
    for body in world.iter("body"):
        name = body.get("name")
        if name == "hand":
            ET.SubElement(
                body,
                "geom",
                name="render_palm",
                type="box",
                pos="0 0 0.032",
                size="0.04 0.035 0.025",
                rgba="0.25 0.31 0.39 1",
                contype="0",
                conaffinity="0",
            )
        if name in ("left_finger", "right_finger"):
            display = body.find(f"geom[@name='{name}_display']")
            if display is not None:
                body.remove(display)
            ET.SubElement(
                body,
                "geom",
                name=f"{name}_stem",
                type="capsule",
                fromto="0 0 0 0 0 0.029",
                size="0.009",
                rgba="0.2 0.26 0.34 1",
                contype="0",
                conaffinity="0",
            )
    colors = {
        "table": "0.60 0.66 0.74 1",
        "support_geom": "0.25 0.49 0.80 1",
        "front_left_geom": "0.19 0.63 0.77 1",
        "front_right_geom": "0.36 0.48 0.87 1",
        "sphere_geom": "0.85 0.43 0.22 1",
        "prism_geom": "0.45 0.38 0.77 1",
    }
    for geom in world.iter("geom"):
        name = geom.get("name", "")
        if name == "floor":
            geom.set("material", "render_floor")
            geom.set("rgba", "1 1 1 1")
        elif name in colors:
            geom.set("rgba", colors[name])
        elif name.startswith("wall"):
            geom.set("rgba", "0.49 0.58 0.68 1")
        elif name.endswith("_pad"):
            geom.set("rgba", "0.08 0.12 0.19 1")
        elif name.endswith("_geom"):
            geom.set("rgba", "0.22 0.62 0.40 1")
        elif name.endswith("_display"):
            geom.set("rgba", "0.58 0.66 0.75 1")
        if root.get("model", "").endswith("_push") and (name.endswith("_display") or name == "render_palm"):
            # Fade noncolliding arm visuals so all three physical blocks remain visible.
            geom.set("rgba", " ".join([*geom.get("rgba").split()[:3], "0.16"]))
    return ET.tostring(root, encoding="unicode")


def _camera(mj, name, metadata):
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    settings = {
        "slide": ([0.065, 0, 0.025], 0.40, 90, -24),
        "drop": ([0, 0, 0.17], 0.88, 110, -15),
        "hinge": ([0, 0, 0.29], 0.85, 90, -8),
        "collision": ([0.04, -0.065, 0.03], 0.88, 100, -55),
        "panda_effort": ([0.33, 0, 0.635], 0.66, 120, -23),
        "grasp": ([0.307, 0, 0.56], 0.48, 120, -23),
        "stack": ([0.36, 0, 0.57], 0.62, 120, -27),
        "push": ([0.465, 0, 0.505], 0.59, 90, -65),
    }
    lookat, distance, azimuth, elevation = settings[name]
    if name in ("grasp", "stack", "push"):
        home = np.asarray(metadata["home_hand_position_m"])
        lookat = np.asarray(lookat) + home - [0.30689056659294095, 0, 0.5902820523028393]
    camera.lookat[:] = lookat
    camera.distance, camera.azimuth, camera.elevation = distance, azimuth, elevation
    return camera


def _phase(name, time_s):
    schedules = {
        "grasp": (
            (0.3, "Ready"),
            (0.8, "Closing"),
            (1.0, "Gripped"),
            (1.8, "Lifting"),
            (2.3, "Holding"),
            (2.7, "Releasing"),
        ),
        "stack": (
            (0.3, "Ready"),
            (0.8, "Closing"),
            (1.0, "Gripped"),
            (1.8, "Lifting"),
            (2.6, "Transferring"),
            (3.4, "Lowering"),
            (3.6, "Positioned"),
            (4.0, "Releasing"),
            (4.2, "Settling"),
            (4.8, "Withdrawing"),
        ),
        "push": ((0.5, "Ready"), (3.5, "Pushing")),
    }
    for end, label in schedules.get(name, ()):
        if time_s < end:
            return label
    return {"grasp": "Released", "stack": "Settling", "push": "Finished"}.get(name, "Recorded trajectory")


def _caption(name, result):
    metrics, metadata = result["metrics"], result["metadata"]
    if name == "slide":
        distance = metrics.get("travel_m", metrics["final_positions_m"]["cube"][0])
        return f"Friction {metadata['friction']:.1f}  |  Travel {distance * 1000:.2f} mm"
    if name == "drop":
        return "40 mm cube  |  Initial center height 350 mm"
    if name == "hinge":
        return "Rigid hinge  |  0.2 Nm pulse from 0.1 to 0.2 seconds"
    if name == "collision":
        return "Rolling sphere strikes a cube and triangular prism"
    if name == "panda_effort":
        return "Seven Panda arm joints  |  Saved effort-driven motion"
    if name == "grasp":
        hold = "Held through lift" if metrics.get("held_during_hold") else "Hold criterion failed"
        return f"40 mm cube  |  {hold}  |  Peak lift {metrics['max_cube_lift_m'] * 1000:.1f} mm"
    if name == "stack":
        outcome = "Stack survives" if metrics.get("stack_survives_final_half_second") else "Stack criterion failed"
        offset = metadata["stack_offset_m"] * 1000
        return f"40 mm cubes  |  Stack offset {offset:g} mm  |  {outcome}"
    return f"{metadata['channel_width_m'] * 1000:.0f} mm channel  |  {metrics['blocks_past_goal']} of 3 blocks pass the goal"


def _font(size):
    from PIL import ImageFont

    for path in (
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _encoder(path, width, height, fps):
    encoder = imageio_ffmpeg.write_frames(
        str(path),
        size=(width, height),
        fps=fps,
        codec="libx264",
        pix_fmt_out="yuv420p",
        macro_block_size=1,
        ffmpeg_log_level="error",
        output_params=["-crf", "20", "-preset", "fast", "-movflags", "+faststart"],
    )
    encoder.send(None)
    return encoder


def _render_scene(directory, name, solver, output, width, height, fps, reel):
    import mujoco as mj
    from PIL import Image, ImageDraw

    folder = directory / name / f"{solver}-0"
    report = folder / "analysis.json"
    if not report.exists():
        report = folder / "result.json"
    result = json.loads(report.read_text())
    if result["status"] != "completed":
        raise ValueError(f"Cannot render incomplete run: {folder}")
    tracked = tuple(result["metrics"]["final_positions_m"])
    trace = np.load(folder / "trace.npz")
    viewport_height = height - 166
    model = mj.MjModel.from_xml_string(
        _display_xml((directory / name / "scene.xml").read_text(), width, viewport_height)
    )
    data = mj.MjData(model)
    controlled = [j for j in range(model.njnt) if model.jnt_type[j] != mj.mjtJoint.mjJNT_FREE]
    if len(controlled) != trace["controlled_joint_q"].shape[1]:
        raise ValueError("Recorded joint count does not match the saved scene")
    bodies = [model.body(body_name).id for body_name in tracked]
    free_joints = []
    for index, body in enumerate(bodies):
        joint = model.body_jntadr[body]
        if joint >= 0 and model.jnt_type[joint] == mj.mjtJoint.mjJNT_FREE:
            free_joints.append((index, model.jnt_qposadr[joint]))
    camera = _camera(mj, name, result["metadata"])
    renderer = mj.Renderer(model, height=viewport_height, width=width)
    encoder = _encoder(output / f"{name}-{solver}.mp4", width, height, fps)
    fonts = [_font(size) for size in (14, 30, 22, 18, 16)]
    speed = 0.5 if name in ("grasp", "stack", "push") else 0.2 if name in ("slide", "panda_effort") else 0.25
    duration = float(trace["time_s"][-1])
    initial_hold, final_hold = 0.4, 0.8
    frame_count = math.ceil((initial_hold + duration / speed + final_hold) * fps)
    caption = _caption(name, result)
    max_error = 0.0
    max_rotation_error = 0.0
    last_index, pixels = -1, None
    poster_index = int((initial_hold + duration * 0.55 / speed) * fps)
    try:
        for frame in range(frame_count):
            time_s = float(np.clip((frame / fps - initial_hold) * speed, 0, duration))
            index = min(int(np.searchsorted(trace["time_s"], time_s)), len(trace["time_s"]) - 1)
            if index and abs(trace["time_s"][index - 1] - time_s) < abs(trace["time_s"][index] - time_s):
                index -= 1
            if index != last_index:
                for joint, value in zip(controlled, trace["controlled_joint_q"][index], strict=True):
                    data.qpos[model.jnt_qposadr[joint]] = value
                for body_index, start in free_joints:
                    data.qpos[start : start + 3] = trace["positions_m"][index, body_index]
                    quat = trace["rotations_xyzw"][index, body_index]
                    data.qpos[start + 3 : start + 7] = quat[[3, 0, 1, 2]]
                mj.mj_forward(model, data)
                error = float(np.max(np.linalg.norm(data.xpos[bodies] - trace["positions_m"][index], axis=1)))
                max_error = max(error, max_error)
                if error > 1e-5:
                    raise ValueError(f"Replay differs from recorded body positions by {error} m")
                for body_index, body in enumerate(bodies):
                    rotation = np.empty(9)
                    quat = trace["rotations_xyzw"][index, body_index].astype(float)[[3, 0, 1, 2]]
                    quat /= np.linalg.norm(quat)
                    mj.mju_quat2Mat(rotation, quat)
                    rotation_error = float(np.max(np.abs(data.xmat[body] - rotation)))
                    max_rotation_error = max(rotation_error, max_rotation_error)
                    if rotation_error > 1e-4:
                        raise ValueError(f"Replay rotation differs from the saved body orientation by {rotation_error}")
                renderer.update_scene(data, camera=camera)
                pixels = renderer.render()
                last_index = index
            canvas = Image.new("RGB", (width, height), (240, 246, 253))
            canvas.paste(Image.fromarray(pixels), (0, 86))
            draw = ImageDraw.Draw(canvas)
            draw.rectangle((0, 0, width, 85), fill=(18, 34, 53))
            draw.text((28, 11), "MANDA  /  RIGID SCENE RECONSTRUCTION", font=fonts[0], fill=(159, 181, 204))
            draw.text((26, 35), TITLES[name], font=fonts[1], fill=(247, 250, 255))
            badge = "FPGS  |  CPU" if solver == "fpgs" else "MuJoCo  |  CPU"
            draw.rounded_rectangle((width - 212, 26, width - 28, 62), radius=8, fill=(35, 91, 91))
            draw.text((width - 198, 33), badge, font=fonts[3], fill=(224, 255, 244))
            draw.text((28, height - 72), _phase(name, time_s), font=fonts[2], fill=(23, 44, 64))
            draw.text((28, height - 39), caption, font=fonts[4], fill=(55, 79, 101))
            clock = f"t = {trace['time_s'][index]:.3f} s  /  {duration:.1f} s"
            draw.text((width - 290, height - 70), clock, font=fonts[3], fill=(23, 44, 64))
            iterations = f"{result['iterations']} iterations  /  " if solver == "fpgs" else ""
            draw.text((width - 290, height - 40), f"{iterations}{speed:g}x playback", font=fonts[4], fill=(55, 79, 101))
            if duration:
                draw.rectangle((0, height - 4, int(width * time_s / duration), height), fill=(30, 155, 122))
            frame_array = np.asarray(canvas)
            encoder.send(frame_array)
            reel.send(frame_array)
            if frame == poster_index:
                canvas.save(output / f"{name}-{solver}.png")
    finally:
        renderer.close()
        encoder.close()
        trace.close()
    info = {
        "scene": name,
        "solver": solver,
        "file": f"{name}-{solver}.mp4",
        "frames": frame_count,
        "fps": fps,
        "playback_speed": speed,
        "simulation_duration_s": duration,
        "max_replay_body_position_error_m": max_error,
        "max_replay_rotation_matrix_error": max_rotation_error,
        "source_scene_sha256": result.get("scene_sha256"),
        "dynamics_rerun": False,
    }
    print(f"Rendered {name}: {frame_count} frames; replay error {max_error:.3g} m", flush=True)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Directory written by manda_rigid")
    parser.add_argument("--output", type=Path, help="New video directory; defaults to INPUT/videos")
    parser.add_argument("--scene", choices=("all", *SCENES), default="all")
    parser.add_argument("--solver", choices=("fpgs", "mujoco"), default="fpgs")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    if args.fps <= 0 or args.width < 1280 or args.height < 720 or args.width % 2 or args.height % 2:
        parser.error("Use positive FPS and even dimensions of at least 1280 x 720")
    output = args.output or args.input / "videos"
    output.mkdir(parents=True, exist_ok=False)
    scenes = SCENES if args.scene == "all" else (args.scene,)
    reel = _encoder(output / f"{args.solver}-rigid-scenes.mp4", args.width, args.height, args.fps)
    manifest = []
    try:
        for scene in scenes:
            manifest.append(
                _render_scene(args.input, scene, args.solver, output, args.width, args.height, args.fps, reel)
            )
    finally:
        reel.close()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Videos saved to {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
