# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Replace reviewed cuts while preserving each accepted batch's provenance."""

import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path

import imageio_ffmpeg


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def accepted_recording(run):
    report = json.loads((run / "report.json").read_text())
    audit = json.loads((run / "strict-audit.json").read_text())
    assert report["quality_gate_passed"] and audit["pass"] and audit["finite"], "Recording failed physics acceptance"
    scene = json.loads((run / "metal-data/scene.json").read_text())
    assert scene["quality_gate_passed"] and not scene["diagnostic"]
    assert report["simultaneous_heterogeneous_batch"] and report["worlds"] == len(scene["worlds"]) == 20
    assert digest(run / "trace.npz") == report["trace_sha256"] == scene["trace_sha256"]
    return {
        "run": str(run.resolve()),
        "trace_sha256": report["trace_sha256"],
        "worlds": report["worlds"],
        "substeps": report["substeps"],
        "iterations": report["iterations"],
        "device": report["device"],
        "quality_gate_passed": True,
    }


def checked_delivery(folder, recording):
    manifest = json.loads((folder / "teaser-manifest.json").read_text())
    assert manifest["decoded_frames"] == manifest["frames"]
    assert not manifest["upscaling"] and not manifest["overlays"]
    assert manifest["playback_speed"] == 1 and manifest["simulation_steps_executed_during_render"] == 0
    assert (manifest["width"], manifest["height"], manifest["fps"]) == (2560, 1440, 30)
    assert manifest["trace_sha256"] == recording["trace_sha256"]
    assert digest(folder / manifest["movie"]) == manifest["movie_sha256"]
    for cut in manifest["edit_list"]:
        render = cut["render"]
        assert render["trace_sha256"] == recording["trace_sha256"]
        assert render["reconstruction_scale"] == 1 and render["reset_history_per_frame"]
        assert render["quality"] == "high" and render["samples_per_frame"] >= 4
        assert render["frames"] == cut["frames"] == round(render["duration_s"] * 30)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-run", type=Path, required=True)
    parser.add_argument("--base-delivery", type=Path, required=True)
    parser.add_argument(
        "--retakes", type=Path, required=True, help="JSON list of replace, shot, run and delivery paths"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base_recording = accepted_recording(args.base_run)
    base = checked_delivery(args.base_delivery, base_recording)
    replacements = json.loads(args.retakes.read_text())
    requested = [item["replace"] for item in replacements]
    assert len(requested) == len(set(requested))
    assert set(requested) <= {cut["shot"] for cut in base["edit_list"]}
    selected = {}
    recordings = {base_recording["trace_sha256"]: base_recording}
    for item in replacements:
        run, folder = Path(item["run"]), Path(item["delivery"])
        recording = accepted_recording(run)
        manifest = checked_delivery(folder, recording)
        cut = next(cut for cut in manifest["edit_list"] if cut["shot"] == item["shot"])
        selected[item["replace"]] = (cut, folder, recording)
        recordings[recording["trace_sha256"]] = recording
    args.output.mkdir(parents=True, exist_ok=True)
    edit_list, paths, frame = [], [], 0
    for original in base["edit_list"]:
        cut, folder, recording = selected.get(original["shot"], (original, args.base_delivery, base_recording))
        assert cut["frames"] == original["frames"], "Retakes must preserve each task's time allocation"
        path = folder / "clips" / f"{cut['shot']}.mp4"
        assert path.is_file()
        paths.append(path.resolve())
        edit_list.append(
            {
                **cut,
                "start_frame": frame,
                "replaces": original["shot"] if original["shot"] in selected else None,
                "recording": recording,
                "encoded_clip": str(path.resolve()),
                "encoded_clip_sha256": digest(path),
            }
        )
        frame += cut["frames"]
    for index, cut in enumerate(edit_list):
        if cut["shot"] == "16-batch-finale":
            previous = edit_list[index - 1]
            assert previous["shot"] == "15-quadruped"
            assert previous["recording"]["trace_sha256"] == cut["recording"]["trace_sha256"]
            assert math.isclose(
                previous["source_start_s"] + previous["frames"] / 30,
                cut["source_start_s"],
                abs_tol=1e-9,
            ), "The final pullback must continue its preceding simulation time"
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    # Different MP4 muxers choose different track timebases. The concat
    # demuxer assumes one timebase, so rescale containers before joining
    # old and new hardware-encoded cuts without touching their frames.
    # Give low-latency and B-frame tracks the same two-frame decode lead
    # so DTS stays ordered across cuts while presentation order is retained.
    normalized = args.output / "normalized-clips"
    normalized.mkdir(exist_ok=True)
    mux_paths = []
    for index, path in enumerate(paths):
        mux_path = normalized / f"{index:02d}-{edit_list[index]['shot']}.mp4"
        subprocess.run(
            [
                ffmpeg,
                "-v",
                "error",
                "-xerror",
                "-y",
                "-i",
                str(path),
                "-map",
                "0:v:0",
                "-c",
                "copy",
                "-bsf:v",
                "setts=dts=(N-2)/(30*TB):pts=PTS-STARTPTS:duration=1/(30*TB)",
                "-video_track_timescale",
                "90000",
                "-movflags",
                "+faststart",
                str(mux_path),
            ],
            check=True,
        )
        mux_paths.append(mux_path.resolve())
        edit_list[index]["mux_clip"] = str(mux_path.resolve())
        edit_list[index]["mux_clip_sha256"] = digest(mux_path)
    concat = args.output / "concat.txt"
    concat.write_text("".join("file '" + str(path).replace("'", "'\\''") + "'\n" for path in mux_paths))
    movie = args.output / "hero-teaser-1440p.mp4"
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
    result = subprocess.run(
        [ffmpeg, "-v", "error", "-xerror", "-i", str(movie), "-progress", "pipe:1", "-nostats", "-f", "null", "-"],
        check=True,
        capture_output=True,
        text=True,
    )
    decoded = int(re.findall(r"^frame=(\d+)$", result.stdout, re.MULTILINE)[-1])
    seconds = int(re.findall(r"^out_time_us=(\d+)$", result.stdout, re.MULTILINE)[-1]) / 1e6
    assert decoded == frame and not result.stderr
    assert math.isclose(seconds, frame / 30, abs_tol=1 / 30)
    manifest = {
        "movie": movie.name,
        "movie_sha256": digest(movie),
        "width": 2560,
        "height": 1440,
        "fps": 30,
        "frames": frame,
        "duration_s": frame / 30,
        "decoded_frames": decoded,
        "decoded_duration_s": seconds,
        "full_decode_validated": True,
        "playback_speed": 1,
        "upscaling": False,
        "overlays": False,
        "audio": False,
        "transitions": "straight cuts",
        "simulation_steps_executed_during_render": 0,
        "video_track_timescale": 90000,
        "uniform_decode_delay_frames": 2,
        "lossless_container_remux": True,
        "multiple_accepted_recordings": len(recordings) > 1,
        "recordings": list(recordings.values()),
        "edit_list": edit_list,
    }
    (args.output / "teaser-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Validated {decoded} frames, {seconds:.1f}s with {len(selected)} retakes: {movie}")


if __name__ == "__main__":
    main()
