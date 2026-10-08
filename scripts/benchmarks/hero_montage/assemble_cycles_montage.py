# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Assemble the CUDA Cycles editorial clips without upscaling source images."""

import json
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

# First demonstrations share a six-second window; later cuts show variety.
ROWS = [
    [
        ("07-lighter-contact", 0, 4.5, 6),
        ("drawer", 2.5, 2, 2),
        ("06-knife-insertion", 1, 2, 2),
        ("13-toy-assembly", 1, 2, 2),
        ("15-quadruped", 1, 2, 2),
    ],
    [
        ("04-hardware-bin", 0, 6, 6),
        ("05-wrecking-ball", 3, 2, 2),
        ("10-hockey", 1, 2, 2),
        ("02-balance-scale", 1, 2, 2),
        ("14-humanoid", 1, 2, 2),
    ],
    [
        ("08-plate-rack", 0, 6, 6),
        ("01-allegro", 1, 2, 2),
        ("12-brick-chutes", 1.5, 2, 2),
        ("11-jenga", 2, 2, 2),
        ("09-toy-truck", 1, 2, 2),
    ],
]


def assemble(root):
    output = root / "montage"
    output.mkdir(exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    codec = [
        "-an",
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
    ]

    def encode(args):
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", *map(str, args)], check=True)

    def clip(name):
        return root / "clips" / f"{name}.mp4"

    drawer = output / "drawer.mp4"
    encode(
        [
            "-i",
            clip("03a-drawer-open"),
            "-i",
            clip("03b-drawer-place"),
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0[v]",
            "-map",
            "[v]",
            *codec,
            drawer,
        ]
    )
    for i, row in enumerate(ROWS):
        inputs, filters = [], []
        for j, (name, start, duration, screen_duration) in enumerate(row):
            inputs.extend(["-i", drawer if name == "drawer" else clip(name)])
            filters.append(
                f"[{j}:v]trim=start={start}:duration={duration},setpts=(PTS-STARTPTS)*{screen_duration / duration},fps=30,scale=486:304:force_original_aspect_ratio=increase,crop=486:304,setsar=1[v{j}]"
            )
        filters.append("".join(f"[v{j}]" for j in range(len(row))) + f"concat=n={len(row)}:v=1:a=0[row]")
        encode(
            [
                *inputs,
                "-filter_complex",
                ";".join(filters),
                "-map",
                "[row]",
                "-r",
                "30",
                *codec,
                output / f"row-{i}.mp4",
            ]
        )
    destination = output / "hero-teaser-cycles-384.mp4"
    encode(
        [
            "-i",
            clip("16-final-zoom-out"),
            *[arg for i in range(3) for arg in ("-i", output / f"row-{i}.mp4")],
            "-filter_complex",
            "[0:v]crop=1620:1080,scale=1404:936,setsar=1,pad=1920:956:10:10:color=white[bg];[bg][1:v]overlay=1424:10[a];[a][2:v]overlay=1424:326[b];[b][3:v]overlay=1424:642[out]",
            "-map",
            "[out]",
            "-frames:v",
            "420",
            "-r",
            "30",
            *codec,
            destination,
        ]
    )
    subprocess.run([ffmpeg, "-v", "error", "-xerror", "-i", str(destination), "-f", "null", "-"], check=True)
    reader = imageio_ffmpeg.read_frames(str(destination))
    metadata = next(reader)
    frames = sum(1 for _ in reader)
    assert frames == 420 and metadata["size"] == (1920, 956)
    (output / "report.json").write_text(
        json.dumps(
            {
                "frames": frames,
                "width": 1920,
                "height": 956,
                "fps": 30,
                "rows": ROWS,
                "renderer": "Cycles OptiX",
                "scenes": 384,
                "decoded": True,
            },
            indent=2,
        )
    )
    print("MONTAGE COMPLETE", destination, flush=True)


if __name__ == "__main__":
    assemble(Path(sys.argv[1]))
