# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Encode reviewed native frames with straight cuts and an auditable edit list."""

import argparse
import hashlib
import json
import math
import re
import subprocess
import time
from pathlib import Path

import imageio_ffmpeg


def main():
    from PIL import Image

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="Accepted replay data used for every shot")
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--shots", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--full-decode", action="store_true", help="Decode every streamed frame as an additional offline check"
    )
    args = parser.parse_args()
    scene = json.loads((args.data / "scene.json").read_text())
    assert scene["quality_gate_passed"] and not scene.get("diagnostic", False), (
        "Only accepted simulations may be assembled"
    )
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    shots = json.loads(args.shots.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    clips = args.output / "clips"
    clips.mkdir(exist_ok=True)
    edit_list = []
    total_frames = 0
    trace_hash = scene["trace_sha256"]
    encode_started = time.perf_counter()
    for shot in shots:
        folder = args.frames / shot["name"]
        report = json.loads((folder / "render-report.json").read_text())
        count = round(shot["duration"] * 30)
        assert report["frames"] == count and report.get("first_frame", 0) == 0
        assert math.isclose(report["source_start_s"], shot["startTime"], abs_tol=1.0e-9)
        assert math.isclose(report["duration_s"], shot["duration"], abs_tol=1.0e-9)
        assert report["playback_speed"] == 1 and report["simulation_steps_executed"] == 0
        assert report["reconstruction_scale"] == 1
        direct_video = report.get("direct_video", False)
        if not direct_video:
            assert report["reset_history_per_frame"]
        assert (report["width"], report["height"], report["output_fps"]) == (2560, 1440, 30)
        trace_hash = trace_hash or report["trace_sha256"]
        assert trace_hash == report["trace_sha256"]
        output = clips / f"{shot['name']}.mp4"
        if direct_video:
            encoder = report["encoder_metrics"]
            assert encoder["frames"] == count and encoder["cpu_pixel_readbacks"] == encoder["png_frames"] == 0
            assert encoder["encoder"] == "VideoToolbox hardware H.264 (required)"
            source = folder / report["video_file"]
            assert source.is_file()
            command = [
                ffmpeg,
                "-v",
                "error",
                "-xerror",
                "-y",
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-an",
                "-c:v",
                "copy",
                "-movflags",
                "+faststart",
                str(output),
            ]
        else:
            for frame in range(count):
                path = folder / f"frame_{frame:04d}.png"
                assert path.is_file(), path
                with Image.open(path) as image:
                    assert image.size == (2560, 1440), path
            command = [
                ffmpeg,
                "-v",
                "error",
                "-xerror",
                "-y",
                "-framerate",
                "30",
                "-start_number",
                "0",
                "-i",
                str(folder / "frame_%04d.png"),
                "-frames:v",
                str(count),
                "-c:v",
                "libx264",
                "-crf",
                "17",
                "-preset",
                "medium",
                "-pix_fmt",
                "yuv420p",
                "-vf",
                "scale=out_color_matrix=bt709",
                "-color_primaries",
                "bt709",
                "-color_trc",
                "bt709",
                "-colorspace",
                "bt709",
                "-movflags",
                "+faststart",
                str(output),
            ]
        subprocess.run(
            command,
            check=True,
        )
        edit_list.append(
            {
                "shot": shot["name"],
                "start_frame": total_frames,
                "frames": count,
                "source_start_s": shot["startTime"],
                "worlds": shot["worlds"],
                "render": report,
            }
        )
        total_frames += count
        operation = "Remuxed" if direct_video else "Encoded"
        print(f"{operation} {shot['name']}: {count} frames", flush=True)
    concat = args.output / "concat.txt"
    encode_seconds = time.perf_counter() - encode_started
    concat.write_text("".join(f"file 'clips/{shot['name']}.mp4'\n" for shot in shots))
    movie = args.output / "hero-teaser-1440p.mp4"
    assembly_started = time.perf_counter()
    subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-xerror",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(movie),
        ],
        check=True,
    )
    assembly_seconds = time.perf_counter() - assembly_started
    all_direct = all(item["render"].get("direct_video", False) for item in edit_list)
    full_decode = args.full_decode or not all_direct
    validation_started = time.perf_counter()
    validation_command = [ffmpeg, "-v", "error", "-xerror", "-i", str(movie), "-map", "0:v:0"]
    if not full_decode:
        validation_command += ["-c:v", "copy"]
    validation_command += ["-progress", "pipe:1", "-nostats", "-f", "null", "-"]
    validation = subprocess.run(
        validation_command,
        check=True,
        capture_output=True,
        text=True,
    )
    decoded_frames = int(re.findall(r"^frame=(\d+)$", validation.stdout, flags=re.MULTILINE)[-1])
    decoded_seconds = int(re.findall(r"^out_time_us=(\d+)$", validation.stdout, flags=re.MULTILINE)[-1]) / 1.0e6
    assert decoded_frames == total_frames and not validation.stderr
    assert math.isclose(decoded_seconds, total_frames / 30, abs_tol=1 / 30 + 1e-6)
    manifest = {
        "movie": movie.name,
        "width": 2560,
        "height": 1440,
        "fps": 30,
        "frames": total_frames,
        "duration_s": total_frames / 30,
        "validated_frames": decoded_frames,
        "validated_duration_s": decoded_seconds,
        "full_decode_validated": full_decode,
        "decoded_frames": decoded_frames if full_decode else None,
        "decoded_duration_s": decoded_seconds if full_decode else None,
        "encode_or_remux_s": encode_seconds,
        "assembly_s": assembly_seconds,
        "validation_s": time.perf_counter() - validation_started,
        "trace_sha256": trace_hash,
        "movie_sha256": hashlib.sha256(movie.read_bytes()).hexdigest(),
        "playback_speed": 1,
        "simulation_steps_executed_during_render": 0,
        "overlays": False,
        "audio": False,
        "upscaling": False,
        "transitions": "straight cuts",
        "direct_video": all_direct,
        "edit_list": edit_list,
    }
    (args.output / "teaser-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Validated {total_frames} frames, {total_frames / 30:.1f}s: {movie}", flush=True)


if __name__ == "__main__":
    main()
