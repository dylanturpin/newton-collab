# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Decode the hardware benchmark and check timing, channels, and frame ordinals."""

import argparse
import json
import math
import re
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    report = json.loads((args.directory / "encoding-benchmark.json").read_text())
    movie = args.directory / "encoding-benchmark.mp4"
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    assert report["encoder"] == "VideoToolbox hardware H.264 (required)"
    assert report["cpu_pixel_readbacks"] == report["png_frames"] == 0
    reader = imageio_ffmpeg.read_frames(str(movie))
    metadata = next(reader)
    reader.close()
    width, height = metadata["source_size"]
    assert (width, height) == (report["width"], report["height"])
    assert metadata["codec"] == "h264" and metadata["fps"] == report["fps"] == 30
    decoded = subprocess.run(
        [ffmpeg, "-v", "error", "-xerror", "-i", str(movie), "-progress", "pipe:1", "-nostats", "-f", "null", "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    frames = int(re.findall(r"^frame=(\d+)$", decoded.stdout, re.MULTILINE)[-1])
    duration = int(re.findall(r"^out_time_us=(\d+)$", decoded.stdout, re.MULTILINE)[-1]) / 1e6
    assert not decoded.stderr and frames == report["frames"] == report["video_duration_s"] * 30
    assert abs(duration - report["video_duration_s"]) < 1 / 30
    ordinals = sorted({0, frames // 2, frames - 1})
    selection = "+".join(f"eq(n\\,{index})" for index in ordinals)
    sampled = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-xerror",
            "-i",
            str(movie),
            "-vf",
            f"select={selection}",
            "-fps_mode",
            "vfr",
            "-pix_fmt",
            "rgb24",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        check=True,
        capture_output=True,
    )
    assert not sampled.stderr
    images = np.frombuffer(sampled.stdout, dtype=np.uint8).reshape(len(ordinals), height, width, 3)
    patches = [(0.1, 0.1, (255, 0, 0)), (0.9, 0.1, (0, 0, 255)), (0.1, 0.9, (0, 255, 0)), (0.9, 0.9, (255, 255, 255))]
    for image, ordinal in zip(images, ordinals, strict=True):
        for x, y, color in patches:
            actual = image[int(y * height), int(x * width)].astype(int)
            assert np.max(np.abs(actual - color)) < 20, (ordinal, x, y, actual)
        decoded_ordinal = sum(
            (int(image[int(0.22 * height), int((bit + 0.5) / 11 * width), 0]) > 127) << bit for bit in range(11)
        )
        assert decoded_ordinal == ordinal, (decoded_ordinal, ordinal)
        for bit in range(11):
            gray = image[int(0.22 * height), int((bit + 0.5) / 11 * width)].astype(int)
            expected_gray = 230 if (ordinal >> bit) & 1 else 26
            assert np.max(np.abs(gray - expected_gray)) < 12, (ordinal, bit, gray)
        x, y = int(width * 0.47), int(height * 0.61)
        checker = ((x + ordinal * 3) // 32 + y // 32) & 1
        background = 0.8 if checker else 0.15
        expected = np.array([x / width, y / height, 0.25 + 0.2 * math.sin(x / width * 40 + ordinal / 30)])
        expected = (expected * 0.7 + background * 0.3) * 255
        assert np.max(np.abs(image[y, x].astype(int) - expected)) < 12, (ordinal, image[y, x], expected)
    validation = {
        "decoded_frames": frames,
        "decoded_duration_s": duration,
        "sampled_frame_ordinals": ordinals,
        "channels_and_orientation_passed": True,
        "brightness_passed": True,
        "width": width,
        "height": height,
    }
    (args.directory / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
