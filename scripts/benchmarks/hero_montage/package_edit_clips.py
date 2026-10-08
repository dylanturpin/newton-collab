# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Package separate editorial clips with provenance and a complete decode check."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

import imageio_ffmpeg


def package(spec, output):
    output.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    records = []
    manifest_path = output / "manifest.json"
    previous = (
        {item["name"]: item for item in json.loads(manifest_path.read_text())["clips"]}
        if manifest_path.exists()
        else {}
    )
    for item in spec:
        source = Path(item["source"])
        destination = output / (item["name"] + ".mp4")
        prior = previous.get(item["name"])
        if (
            prior
            and destination.exists()
            and prior["decoded_frames"] == item["frames"]
            and hashlib.sha256(source.read_bytes()).hexdigest()
            == hashlib.sha256(destination.read_bytes()).hexdigest()
            == prior["sha256"]
        ):
            records.append({**prior, **item})
            continue
        shutil.copy2(source, destination)
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
            capture_output=True,
            text=True,
            check=True,
        )
        frames = int(re.findall(r"^frame=(\d+)$", result.stdout, re.MULTILINE)[-1])
        assert not result.stderr and frames == item["frames"], (item["name"], frames, result.stderr)
        reader = imageio_ffmpeg.read_frames(str(destination))
        metadata = next(reader)
        reader.close()
        assert tuple(metadata["size"]) == (2560, 1440) and metadata["fps"] == 30
        records.append(
            {
                **item,
                "file": destination.name,
                "decoded_frames": frames,
                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            }
        )
        print(f"Validated {destination.name}: {frames} frames", flush=True)
    manifest = {
        "width": 2560,
        "height": 1440,
        "fps": 30,
        "playback_speed": "per_clip_render_report",
        "upscaling": False,
        "overlays": False,
        "simulation_steps_executed_during_render": 0,
        "clips": records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    lines = [
        "# Separate montage clips",
        "",
        "Native 2560x1440, 30 fps. Action clips play at real time; the 24-second overview stretches the six-second recording. No titles or overlays.",
        "",
        "The final overview combines recorded worlds from several CUDA runs; it is not a new simultaneous batch run.",
        "The lighter is the specifically accepted contact-only take (about 65° opening); it does not pass the full-open/repeatability gate.",
        "",
    ]
    lines.extend(f"- {r['file']}: {r['frames'] / 30:g} seconds" for r in records)
    (output / "README.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package(json.loads(args.spec.read_text()), args.output)
