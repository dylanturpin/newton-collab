# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Render FPGS card poses on EGL; MuJoCo is used only for drawing."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio_ffmpeg
import numpy as np


def numbers(values):
    return " ".join(str(float(v)) for v in values)


def main():
    import mujoco as mj
    from PIL import Image, ImageDraw, ImageFont

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    result = json.loads((args.run / "result.json").read_text())
    trace = np.load(args.run / "trace.npz")
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 26)
    small = ImageFont.truetype(font_path, 20)
    root = ET.Element("mujoco", model="FPGS house of cards replay")
    ET.SubElement(root, "compiler", inertiafromgeom="false")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="650")
    ET.SubElement(visual, "quality", shadowsize="4096", offsamples="4")
    ET.SubElement(visual, "map", znear="0.002", zfar="10")
    ET.SubElement(visual, "headlight", ambient="0.4 0.4 0.4", diffuse="0.5 0.5 0.5", specular="0.1 0.1 0.1")
    assets = ET.SubElement(root, "asset")
    ET.SubElement(
        assets,
        "texture",
        name="sky",
        type="skybox",
        builtin="gradient",
        rgb1="0.18 0.23 0.30",
        rgb2="0.35 0.41 0.48",
        width="512",
        height="512",
    )
    for tier, suit in enumerate(("♦", "♠", "♣")):
        card = Image.new("RGB", (256, 384), (247, 244, 233))
        draw = ImageDraw.Draw(card)
        color = (185, 40, 45) if tier == 0 else (32, 42, 54)
        draw.rounded_rectangle((6, 6, 249, 377), radius=14, outline=color, width=5)
        draw.text((20, 13), "A", font=ImageFont.truetype(font_path, 42), fill=color)
        draw.text((19, 53), suit, font=ImageFont.truetype(font_path, 40), fill=color)
        draw.text((128, 190), suit, anchor="mm", font=ImageFont.truetype(font_path, 134), fill=color)
        draw.text((224, 348), "A", anchor="mm", font=ImageFont.truetype(font_path, 42), fill=color)
        path = args.run / f"card-{tier}.png"
        card.save(path)
        ET.SubElement(assets, "texture", name=f"face{tier}", type="2d", file=str(path.resolve()))
        ET.SubElement(
            assets,
            "material",
            name=f"card{tier}",
            texture=f"face{tier}",
            texuniform="true",
            reflectance="0",
            specular="0.1",
            shininess="0.1",
        )
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="-0.5 -0.5 1.5", dir="0.3 0.3 -1", directional="true", castshadow="true")
    ET.SubElement(world, "geom", type="box", pos="0 0 -0.02", size="1 0.7 0.02", rgba="0.15 0.25 0.23 1")
    for i, spec in enumerate(result["bodies"]):
        body = ET.SubElement(world, "body", name=f"body{i}")
        ET.SubElement(body, "freejoint")
        ET.SubElement(body, "inertial", pos="0 0 0", mass="1", diaginertia="1 1 1")
        if spec["kind"] == "ball":
            ET.SubElement(body, "geom", type="sphere", size=str(spec["radius"]), rgba=numbers([*spec["color"], 1]))
        else:
            size = spec["size"]
            attrs = {"type": "box", "material": f"card{spec['tier']}"}
            if size[0] < size[2]:
                attrs["quat"] = "0.707106781 0 0.707106781 0"
                size = [size[2], size[1], size[0]]
            ET.SubElement(body, "geom", size=numbers(size), **attrs)
    model = mj.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    data = mj.MjData(model)
    renderer = mj.Renderer(model, height=650, width=1280)
    camera = mj.MjvCamera()
    camera.azimuth, camera.elevation, camera.distance = 105, -18, 0.83
    camera.lookat[:] = [-0.045, 0, 0.12]
    writer = imageio_ffmpeg.write_frames(
        str(args.run / "card-house.mp4"),
        (1280, 720),
        fps=30,
        codec="libx264",
        pix_fmt_out="yuv420p",
        macro_block_size=1,
        ffmpeg_log_level="error",
        output_params=["-crf", "18", "-preset", "fast", "-movflags", "+faststart"],
    )
    writer.send(None)
    try:
        for frame in range(round(result["duration_s"] * 30) + 1):
            t = frame / 30
            poses = trace["poses"][min(frame * 2, len(trace["poses"]) - 1)]
            q = data.qpos.reshape((-1, 7))
            q[:, :3], q[:, 3:] = poses[:, :3], poses[:, [6, 3, 4, 5]]
            mj.mj_forward(model, data)
            renderer.update_scene(data, camera)
            image = Image.new("RGB", (1280, 720), (22, 29, 38))
            image.paste(Image.fromarray(renderer.render()), (0, 70))
            draw = ImageDraw.Draw(image)
            draw.text((22, 9), "FPGS / HOUSE OF CARDS", font=font, fill=(242, 242, 237))
            phase = "Friction-supported / no glue" if t < result["launch_s"] else "Ball launched / dynamic collapse"
            draw.text((22, 42), phase, font=small, fill=(189, 202, 211))
            draw.text(
                (810, 12),
                f"CUDA | {result['substeps']} substeps x {result['iterations']} iterations",
                font=small,
                fill=(230, 230, 230),
            )
            draw.text(
                (810, 42),
                f"t = {t:.2f}s | simulation {result['ms_per_frame']:.2f} ms/frame",
                font=small,
                fill=(189, 202, 211),
            )
            writer.send(np.asarray(image))
            if frame in (0, 87, 105, 150, 240):
                image.save(args.run / f"frame-{t:.2f}.png")
    finally:
        writer.close()
        renderer.close()
    print(args.run / "card-house.mp4", flush=True)


if __name__ == "__main__":
    main()
