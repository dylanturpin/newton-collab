# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render the recorded GPU kitchen pour at normal playback speed."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio_ffmpeg
import numpy as np


def _numbers(values):
    return " ".join(str(float(v)) for v in values)


def _scene(asset, document):
    root = ET.Element("mujoco", model="FPGS kitchen replay")
    ET.SubElement(root, "compiler", angle="radian", inertiafromgeom="false", meshdir=str(asset.resolve()))
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="590")
    ET.SubElement(visual, "headlight", ambient="0.22 0.22 0.22", diffuse="0.5 0.5 0.5", specular="0.12 0.12 0.12")
    ET.SubElement(visual, "quality", shadowsize="4096")
    ET.SubElement(visual, "map", znear="0.01", zfar="10")
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "geom", contype="0", conaffinity="0")
    bank = ET.SubElement(root, "asset")
    ET.SubElement(
        bank,
        "texture",
        type="skybox",
        builtin="gradient",
        rgb1="0.84 0.88 0.92",
        rgb2="0.96 0.97 0.99",
        width="512",
        height="512",
    )
    for item in document["bank"].values():
        ET.SubElement(bank, "mesh", name=item["mesh"], file=f"{item['mesh']}.obj", inertia="shell")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(
        world, "light", pos="-0.7 -1 2", dir="0.2 0.3 -1", directional="true", diffuse="0.5 0.5 0.5", castshadow="false"
    )
    ET.SubElement(
        world, "geom", name="table", type="box", pos="0.2 0 -0.02", size="1.1 0.8 0.02", rgba="0.32 0.38 0.42 1"
    )
    container = ET.SubElement(world, "body", name="bin")
    ET.SubElement(container, "freejoint")
    ET.SubElement(container, "inertial", pos="0 0 0", mass="1", diaginertia="1 1 1")
    specs = [
        ((0, 0, 0.006), (0.17, 0.17, 0.006)),
        ((-0.164, 0, 0.331), (0.006, 0.17, 0.319)),
        ((0.164, 0, 0.331), (0.006, 0.17, 0.319)),
        ((0, -0.164, 0.331), (0.158, 0.006, 0.319)),
        ((0, 0.164, 0.331), (0.158, 0.006, 0.319)),
    ]
    for position, size in specs:
        ET.SubElement(
            container, "geom", type="box", pos=_numbers(position), size=_numbers(size), rgba="0.60 0.76 0.94 0.17"
        )
    for index, item in enumerate(document["bodies"]):
        kind = document["bank"][item["kind"]]
        body = ET.SubElement(world, "body", name=f"object_{index:02d}")
        ET.SubElement(body, "freejoint")
        ET.SubElement(body, "inertial", pos="0 0 0", mass="1", diaginertia="1 1 1")
        ET.SubElement(body, "geom", type="mesh", mesh=kind["mesh"], rgba=_numbers([*kind["color"], 1]))
    return ET.tostring(root, encoding="unicode")


def main():
    import mujoco as mj
    from PIL import Image, ImageDraw, ImageFont

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", required=True, type=Path)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output video path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    document = json.loads((args.asset / "scene.json").read_text())
    result = json.loads((args.run / "result.json").read_text())
    trace = np.load(args.run / "trace.npz")
    model = mj.MjModel.from_xml_string(_scene(args.asset, document))
    data = mj.MjData(model)
    renderer = mj.Renderer(model, height=590, width=1280)
    camera = mj.MjvCamera()
    camera.azimuth, camera.elevation = 125, -25
    font_path = "/System/Library/Fonts/Helvetica.ttc"
    if not Path(font_path).exists():
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    fonts = [ImageFont.truetype(font_path, size) for size in (30, 18, 16)]
    writer = imageio_ffmpeg.write_frames(
        str(args.output),
        (1280, 720),
        fps=args.fps,
        codec="libx264",
        pix_fmt_out="yuv420p",
        macro_block_size=1,
        ffmpeg_log_level="error",
        output_params=["-crf", "18", "-preset", "fast", "-movflags", "+faststart"],
    )
    writer.send(None)
    duration = trace["time_s"][-1]
    posters = {round(t * args.fps): t for t in (5, 12, 15, 17, 20, 23) if t <= duration}
    frame_count = int(round(duration * args.fps)) + 1
    try:
        for frame in range(frame_count):
            time_s = min(frame / args.fps, duration)
            pullback = np.clip((time_s - 13) / 4, 0, 1)
            pullback = pullback * pullback * (3 - 2 * pullback)
            camera.lookat[:] = [0.04 + 0.16 * pullback, 0, 0.4 - 0.09 * pullback]
            camera.distance = 1.7 + 0.38 * pullback
            index = int(np.argmin(abs(trace["time_s"] - time_s)))
            poses = trace["poses"][index]
            q = data.qpos.reshape((-1, 7))
            q[:, :3] = poses[:, :3]
            q[:, 3:] = poses[:, [6, 3, 4, 5]]
            mj.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            image = Image.new("RGB", (1280, 720), (238, 243, 249))
            image.paste(Image.fromarray(renderer.render()), (0, 76))
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, 1280, 76), fill=(18, 34, 53))
            draw.text((25, 18), "Kitchen pour + spill / 60 rigid objects", font=fonts[0], fill="white")
            draw.text((845, 14), "FPGS / NVIDIA RTX A6000", font=fonts[1], fill=(215, 246, 236))
            draw.text(
                (845, 42),
                f"{result['realtime_factor']:.2f}x real time / colored propagation",
                font=fonts[2],
                fill=(181, 210, 220),
            )
            phase = (
                "Pouring"
                if time_s < 11.8
                else "Settling"
                if time_s < 14
                else "Tipping"
                if time_s < 16
                else "Spilling"
                if time_s < 18
                else "Lifting the bin"
                if time_s < 22.5
                else "Settling"
            )
            live = sum(b["release_s"] <= time_s + 1e-8 for b in document["bodies"])
            draw.text((25, 680), f"{phase}  /  {live} objects released", font=fonts[1], fill=(25, 47, 66))
            draw.text(
                (710, 680),
                f"t = {time_s:05.2f} s   /   1x playback   /   {result['iterations']} solver iterations",
                font=fonts[1],
                fill=(25, 47, 66),
            )
            draw.rectangle((0, 716, int(1280 * time_s / duration), 720), fill=(37, 158, 130))
            writer.send(np.asarray(image))
            if frame in posters:
                image.save(args.output.parent / f"frame-{posters[frame]:02d}.png")
    finally:
        writer.close()
        renderer.close()
        trace.close()
    print(f"Rendered {frame_count} frames: {args.output}")


if __name__ == "__main__":
    main()
