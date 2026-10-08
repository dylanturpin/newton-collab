# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Assemble a slow recorded pullback beside three staggered editorial rows."""

import argparse
import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from prepare_metal_paper_view import prepare
from texture_knife_holders import apply as texture_knife_holders

# Each row lasts 24 seconds. All 15 task types appear, with drawer opening
# and placement joined into one shot. Cuts never align across the three rows.
ROWS = [
    [
        ("07-lighter-contact", 0, 4.5),
        ("drawer", 0.5, 5),
        ("06-knife-insertion", 0.6, 4.7),
        ("13-toy-assembly", 0.5, 5),
        ("15-quadruped", 0.6, 4.8),
    ],
    [
        ("04-hardware-bin", 0.3, 4),
        ("05-wrecking-ball", 2.5, 5),
        ("10-hockey", 0.4, 4.8),
        ("02-balance-scale", 0.5, 5),
        ("14-humanoid", 0.4, 5.2),
    ],
    [
        ("08-plate-rack", 0.3, 5.1),
        ("01-allegro", 0.4, 5.2),
        ("12-brick-chutes", 0.8, 4.5),
        ("11-jenga", 0.7, 4.6),
        ("09-toy-truck", 0.7, 4.6),
    ],
]


def prepare_left(root, output):
    camera = json.loads((root / "hq-clips-no-drills/16-final-zoom-out/camera.json").read_text())[0]
    camera.update(name="teaser-left", duration=24, playbackSpeed=0.25)
    end = json.loads((root / "paper-teaser/overview.json").read_text())[0]
    for keyframe, time_value in zip(camera["keyframes"], (0, 1.2, 22, 24), strict=True):
        keyframe["time"] = time_value
        if time_value >= 22:
            for key in ("position", "target", "fov"):
                keyframe[key] = end[key]
    (output / "left-camera.json").write_text(json.dumps([camera], indent=2) + "\n")
    first, last = camera["keyframes"][0], camera["keyframes"][-1]
    views = [
        {
            **{
                key: ((1 - u) * np.array(first[key]) + u * np.array(last[key])).tolist()
                for key in ("position", "target")
            },
            "fov": (1 - u) * first["fov"] + u * last["fov"],
            "aspect": 1.5,
        }
        for u in np.linspace(0, 1, 25)
    ]
    culling = output / "left-culling.json"
    culling.write_text(json.dumps(views, indent=2) + "\n")
    prepare(root / "finale-data", root / "hq-clips-no-drills/16-final-zoom-out/snapshot", culling, output / "left-data")
    texture_knife_holders(output / "left-data")


def assemble(root, renderer, *, wait=False):
    output = root / "video-teaser"
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "left-data/scene.json").exists():
        prepare_left(root, output)
    clips = root / "hq-clips-no-drills/editing-clips"
    manifest_path = clips / "manifest.json"
    schedule = []
    for row in ROWS:
        elapsed = 0.0
        entries = []
        for name, start, duration in row:
            entries.append(
                {"clip": name, "source_start": start, "duration": duration, "montage_start": round(elapsed, 3)}
            )
            elapsed += duration
        assert abs(elapsed - 24) < 1e-6
        schedule.append(entries)
    (output / "timeline.json").write_text(json.dumps(schedule, indent=2) + "\n")
    while True:
        try:
            manifest = json.loads(manifest_path.read_text())
            if len(manifest["clips"]) == 17:
                break
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        if not wait:
            raise RuntimeError("The 17 validated source clips are not ready")
        print("Waiting for the 17 validated HQ clips", flush=True)
        time.sleep(30)
    for item in manifest["clips"]:
        assert hashlib.sha256((clips / item["file"]).read_bytes()).hexdigest() == item["sha256"]
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    def encode(arguments):
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *map(str, arguments)], check=True)

    codec = [
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "17",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
    ]
    drawer = output / "drawer.mp4"
    encode(
        [
            "-i",
            clips / "03a-drawer-open.mp4",
            "-i",
            clips / "03b-drawer-place.mp4",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0[v]",
            "-map",
            "[v]",
            *codec,
            drawer,
        ]
    )
    for index, row in enumerate(ROWS):
        arguments, filters, inputs = [], [], []
        for j, (name, start, duration) in enumerate(row):
            source = drawer if name == "drawer" else clips / (name + ".mp4")
            arguments += ["-i", source]
            filters.append(
                f"[{j}:v]trim=start={start}:duration={duration},setpts=PTS-STARTPTS,"
                f"scale=624:390:force_original_aspect_ratio=increase,crop=624:390,setsar=1[v{j}]"
            )
            inputs.append(f"[v{j}]")
        filters.append("".join(inputs) + f"concat=n={len(row)}:v=1:a=0[row]")
        encode(
            [
                *arguments,
                "-filter_complex",
                ";".join(filters),
                "-map",
                "[row]",
                "-r",
                "30",
                *codec,
                output / f"row-{index}.mp4",
            ]
        )
        print(f"Encoded staggered row {index + 1}/3", flush=True)
    print("Rendering the dedicated 24-second pullback", flush=True)
    subprocess.run(
        [
            str(renderer),
            "--data",
            str(output / "left-data"),
            "--cameras",
            str(output / "left-camera.json"),
            "--output",
            str(output / "native-render"),
            "--width",
            "1800",
            "--height",
            "1200",
            "--quality",
            "high",
            "--accumulation",
            "4",
            "--samples-per-frame",
            "4",
            "--diffuse-samples",
            "8",
            "--lighting",
            "soft-studio",
            "--reset-history-per-frame",
            "--direct-video",
        ],
        check=True,
    )
    left = output / "native-render/teaser-left/video.mp4"
    render_report = json.loads((left.parent / "render-report.json").read_text())
    assert render_report["frames"] == 720 and render_report["playback_speed"] == 0.25
    destination = output / "hero-teaser.mp4"
    encode(
        [
            "-i",
            left,
            *[argument for i in range(3) for argument in ("-i", output / f"row-{i}.mp4")],
            "-filter_complex",
            "[0:v]pad=2460:1224:12:12:color=white[bg];"
            "[bg][1:v]overlay=1824:12[a];[a][2:v]overlay=1824:417[b];[b][3:v]overlay=1824:822[out]",
            "-map",
            "[out]",
            "-frames:v",
            "720",
            "-r",
            "30",
            *codec,
            destination,
        ]
    )
    result = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-xerror",
            "-i",
            str(destination),
            "-progress",
            "pipe:1",
            "-nostats",
            "-f",
            "null",
            "-",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    frames = int(re.findall(r"^frame=(\d+)$", result.stdout, re.MULTILINE)[-1])
    reader = imageio_ffmpeg.read_frames(str(destination))
    metadata = next(reader)
    reader.close()
    assert frames == 720 and metadata["size"] == (2460, 1224) and metadata["fps"] == 30
    report = {
        "file": str(destination),
        "duration": 24,
        "fps": 30,
        "width": 2460,
        "height": 1224,
        "decoded_frames": frames,
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "rows": schedule,
        "left_recording_speed": 0.25,
        "right_recording_speed": 1,
        "simultaneous_heterogeneous_batch": False,
        "left_render": render_report,
    }
    (output / "teaser-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("TEASER COMPLETE AND DECODE-VALIDATED", destination, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("renderer", type=Path)
    parser.add_argument("--wait-for-clips", action="store_true")
    args = parser.parse_args()
    assemble(args.root.resolve(), args.renderer.resolve(), wait=args.wait_for_clips)
