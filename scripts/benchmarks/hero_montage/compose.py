# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Assemble native Newton renders and an attributed public hardware reference."""

import argparse
import json
import math
import subprocess
from pathlib import Path

import imageio_ffmpeg

LABELS = {
    "lift": "BRIDGE ASSEMBLY",
    "stack": "JENGA BUILD",
    "sort": "AIR HOCKEY",
    "hand": "KUKA + ALLEGRO",
    "spill": "MARBLE RUN",
    "insert": "STAR KEY INSERTION",
    "g1": "G1 / TRAINED POLICY",
    "drawer": "UTENSIL DRAWER",
    "shadow": "FIVE-FINGER PEN GRASP",
    "kit": "VIADUCT KIT",
    "gear": "GEAR CRANK",
    "serve": "SERVING",
    "toy": "ROLLING TOYS",
    "puzzle": "SHAPE PUZZLE",
    "interlock": "CROSS PUZZLE",
    "pile": "CUTLERY PILE",
    "go2": "GO2 / TRAINED POLICY",
}


def main():
    from PIL import Image, ImageDraw, ImageFont

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--g1", type=Path, required=True)
    parser.add_argument("--real", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    font = "/System/Library/Fonts/Supplemental/Arial.ttf"

    def encode(arguments):
        subprocess.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error", *arguments], check=True)

    codec = [
        "-an",
        "-r",
        "30",
        "-fps_mode",
        "cfr",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]

    def label(text):
        return (
            "drawbox=x=35:y=34:w=850:h=62:color=0x17212d@0.78:t=fill,"
            f"drawtext=fontfile={font}:text='{text}':x=55:y=51:fontsize=26:fontcolor=white"
        )

    report = json.loads((args.batch / "report.json").read_text())
    duration = report["duration"]
    worlds = report["worlds"]
    parts = []
    sequence = [
        (args.batch / "overview.mp4", 0, 10, f"{worlds} DISTINCT STATIONS / ONE CUDA BATCH"),
        (args.batch / "allegro-detail.mp4", 2, 4, "KUKA + ALLEGRO / IN-HAND MOTION"),
        (args.batch / "franka-at-scale.mp4", 2, 4, "FRANKA / FOUR DIFFERENT TASKS"),
        (args.batch / "task-kit-0.mp4", 4.5, 5, "KINOVA / KEYED VIADUCT ASSEMBLY"),
        (args.batch / "insertion-detail.mp4", 1.5, 6, "FRANKA / TIGHT STAR INSERTION"),
        (args.batch / "task-pile-0.mp4", 4, 4.5, "UR5 / CUTLERY PILE HANDLING"),
        (args.batch / "task-gear-0.mp4", 2.5, 4.5, "SOCKET TOOL / CONTACT-DRIVEN GEARS"),
        (args.batch / "task-toy-1.mp4", 2.5, 4.5, "UR5 / ARTICULATED TOY TRUCK"),
        (args.batch / "task-serve-1.mp4", 5, 4.5, "KUKA / CERAMIC MUG PLACEMENT"),
    ]
    sequence = [shot for shot in sequence if shot[0].exists()]
    for i, (source, start, shot_duration, title) in enumerate(sequence):
        dest = args.output / f"segment-{i}.mp4"
        encode(
            [
                "-ss",
                str(start),
                "-i",
                str(source),
                "-t",
                str(shot_duration),
                "-vf",
                f"fps=30,scale=1920:1080,setsar=1,{label(title)}",
                *codec,
                str(dest),
            ]
        )
        parts.append(dest)
    pair = args.output / "g1-real-sim-pair.mp4"
    filters = (
        "[0:v]fps=30,crop=960:1080:480:0,setsar=1,setpts=PTS-STARTPTS[real];"
        "[1:v]fps=30,crop=960:1080:480:0,setsar=1,setpts=PTS-STARTPTS[sim];"
        "[real][sim]hstack=inputs=2,"
        "drawbox=x=0:y=0:w=1920:h=75:color=0x17212d@0.8:t=fill,"
        f"drawtext=fontfile={font}:text='UNITREE / REAL ROBOT':x=38:y=24:fontsize=28:fontcolor=white:enable='gte(t,3)',"
        f"drawtext=fontfile={font}:text='FPGS / SIMULATION':x=1000:y=24:fontsize=28:fontcolor=white:enable='gte(t,3)',"
        f"drawtext=fontfile={font}:text='WHICH ONE IS REAL?':x=(w-tw)/2:y=24:fontsize=30:fontcolor=white:enable='lt(t,3)',"
        "drawbox=x=0:y=1034:w=1920:h=46:color=0x17212d@0.85:t=fill,"
        f"drawtext=fontfile={font}:text='Hardware footage - Unitree Robotics. Illustrative comparison; different policy checkpoints.':"
        "x=35:y=1048:fontsize=18:fontcolor=white[out]"
    )
    encode(
        [
            "-i",
            str(args.real),
            "-ss",
            "1",
            "-i",
            str(args.g1 / "overview.mp4"),
            "-filter_complex",
            filters,
            "-map",
            "[out]",
            "-t",
            "5.3",
            *codec,
            str(pair),
        ]
    )
    parts.append(pair)
    listing = args.output / "concat.txt"
    listing.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
    encode(
        [
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(listing),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(args.output / "hero-prototype.mp4"),
        ]
    )

    summary = json.loads((args.batch / "model-summary.json").read_text())
    stations = summary["worlds"]

    def caption(w):
        special = {
            "hand": "KUKA + ALLEGRO / IN-HAND",
            "shadow": "SHADOW HAND / PEN GRASP",
            "g1": "G1 / TRAINED WALK",
            "go2": "GO2 / TRAINED WALK",
        }
        if w["kind"] in special:
            return special[w["kind"]]
        robot = w.get("robot", {"hand": "KUKA + ALLEGRO", "shadow": "SHADOW"}.get(w["kind"], w["kind"]))
        title = LABELS[w["kind"]]
        if w["kind"] == "toy":
            title = "TOY TRAIN" if w["variant"] % 2 == 0 else "DUMP TRUCK"
        return f"{robot.upper()} / {title}"

    # Four-view panels expose each selected interaction for task review.
    clips = [w for w in stations if (args.batch / f"task-{w['id']}.mp4").exists()]
    review_parts = []
    for page in range(math.ceil(len(clips) / 4)):
        group = clips[page * 4 : (page + 1) * 4]
        inputs, filters = [], []
        for i, w in enumerate(group):
            inputs += ["-i", str(args.batch / f"task-{w['id']}.mp4")]
            filters.append(f"[{i}:v]fps=30,scale=960:540,setsar=1,{label(caption(w))}[v{i}]")
        for i in range(len(group), 4):
            filters.append(f"color=c=0x17212d:s=960x540:r=30:d={duration}[v{i}]")
        filters.append("[v0][v1][v2][v3]xstack=inputs=4:layout=0_0|960_0|0_540|960_540[out]")
        dest = args.output / f"review-{page}.mp4"
        encode([*inputs, "-filter_complex", ";".join(filters), "-map", "[out]", "-t", str(duration), *codec, str(dest)])
        review_parts.append(dest)
    listing.write_text("".join(f"file '{p.resolve()}'\n" for p in review_parts))
    encode(
        [
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(listing),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(args.output / "all-tasks-review.mp4"),
        ]
    )

    sheet = Image.new("RGB", (1920, math.ceil(len(stations) / 4) * 306), "#17212d")
    draw = ImageDraw.Draw(sheet)
    title_font = ImageFont.truetype(font, 20)
    for i, w in enumerate(stations):
        x, y = (i % 4) * 480, (i // 4) * 306
        im = (
            Image.open(args.batch / f"station-{w['id']}.png")
            .convert("RGB")
            .resize((480, 270), Image.Resampling.LANCZOS)
        )
        sheet.paste(im, (x, y))
        draw.text((x + 10, y + 278), caption(w), font=title_font, fill="white")
    sheet.save(args.output / "all-stations.jpg", quality=95)

    chosen = ("kit-0", "toy-1", "gear-0", "drawer-0", "pile-0", "insert-0", "hand-0", "shadow-0")
    preview = Image.new("RGB", (1920, 612), "#17212d")
    draw = ImageDraw.Draw(preview)
    for i, key in enumerate(chosen):
        w = next(w for w in stations if w["id"] == key)
        x, y = (i % 4) * 480, (i // 4) * 306
        im = (
            Image.open(args.batch / f"template-{key}-05.png")
            .convert("RGB")
            .resize((480, 270), Image.Resampling.LANCZOS)
        )
        preview.paste(im, (x, y))
        draw.text((x + 10, y + 278), caption(w), font=title_font, fill="white")
    preview.save(args.output / "preview.jpg", quality=95)
    print(args.output / "hero-prototype.mp4")


if __name__ == "__main__":
    main()
