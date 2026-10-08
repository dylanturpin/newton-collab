# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Apply approved paper decor to complete recorded editorial takes."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from prepare_metal_paper_view import prepare


def read(path):
    return json.loads(path.read_text())


def shift_camera(camera, delta):
    result = copy.deepcopy(camera)
    for key in ("position", "target", "endPosition", "endTarget"):
        if key in result:
            result[key] = (np.asarray(result[key]) + delta).tolist()
    for keyframe in result.get("keyframes", []):
        for key in ("position", "target"):
            keyframe[key] = (np.asarray(keyframe[key]) + delta).tolist()
    return result


def link_geometry(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("vertices.bin", "indices.bin", "textures"):
        target = destination / name
        if not target.exists():
            target.symlink_to(source / name)


def write_motion(meta, destination, positions, rotations):
    positions = positions.astype("<f4")
    rotations = rotations.astype("<f4")
    assert np.isfinite(positions).all() and np.isfinite(rotations).all()
    np.testing.assert_allclose(np.linalg.norm(rotations, axis=-1), 1, atol=1e-4)
    positions.tofile(destination / "positions.bin")
    rotations.tofile(destination / "rotations.bin")
    meta["sample_count"] = len(positions)
    meta["trace_sha256"] = hashlib.sha256(positions.tobytes() + rotations.tobytes()).hexdigest()
    (destination / "scene.json").write_text(json.dumps(meta, indent=2) + "\n")


def prepare_clips(root, dressed, output):
    root, dressed, output = (p.resolve() for p in (root, dressed, output))
    source = root / "finale-data"
    original, arrangement = read(source / "scene.json"), read(dressed / "scene.json")
    package = read(root / "clip-package.json")
    output.mkdir(parents=True, exist_ok=True)
    overrides = {"shadow-0": 88, "pile-0": 6, "serve-1": 38}
    plan = []
    for item in package[:-1]:
        name = item["name"]
        recording = item["recording"]
        run = Path(recording["run"] if isinstance(recording, dict) else recording)
        raw_path = run / "metal-data"
        raw = read(raw_path / "scene.json")
        selected = item.get("render", {}).get("worlds", [])
        source_id = (
            selected[0]
            if selected
            else {
                "07-lighter-contact": "shadow-3",
                "05-wrecking-ball": "gear-0",
                "15-quadruped": "go2-0",
            }[name]
        )
        target_id = "shadow-0" if name == "07-lighter-contact" else source_id
        raw_wi, raw_world = next((i, w) for i, w in enumerate(raw["worlds"]) if w["id"] == source_id)
        source_world = next(w for w in original["worlds"] if w["id"] == target_id)
        tile_index = overrides.get(target_id)
        if tile_index is None:
            tile_index = next(
                i for i, p in enumerate(arrangement["replica_provenance"]) if p["source_world"] == target_id
            )
        tile = arrangement["worlds"][tile_index]
        provenance = arrangement["replica_provenance"][tile_index]
        translation = np.asarray(provenance["task_centering_translation"])
        count = raw_world["body_count"]
        assert count == tile["body_count"] == source_world["body_count"]
        start = item.get("source_start_s", item.get("render", {}).get("source_start_s"))
        duration = item["frames"] / 30
        fps = raw["recording_fps"]
        first, samples = round(start * fps), round(duration * fps) + 1
        assert first + samples <= raw["sample_count"]
        new_offset = np.array([0, 0, 0.805])
        snapshot = output / name / "snapshot"
        link_geometry(dressed, snapshot)
        meta = copy.deepcopy(arrangement)
        meta.update(
            worlds=[{**source_world, "body_start": 0, "display_offset": new_offset.tolist()}],
            body_count=count + 1,
            recorded_body_count=count,
            recording_fps=fps,
            source_recordings=[
                {"source": str(raw_path), "world": source_id, "start": start, "trace_sha256": raw["trace_sha256"]}
            ],
            replica_provenance=[provenance],
            meshes=[
                {
                    **m,
                    "world": 0,
                    "body": m["body"] - tile["body_start"] if m["body"] < arrangement["recorded_body_count"] else count,
                }
                for m in arrangement["meshes"]
                if m["world"] == tile_index
            ],
        )
        buffers = []
        for filename in ("positions.bin", "rotations.bin"):
            recording_array = np.memmap(raw_path / filename, dtype="<f4", mode="r").reshape(
                raw["sample_count"], raw["body_count"], 4
            )
            bodies = [
                *range(raw_world["body_start"], raw_world["body_start"] + count),
                raw["recorded_body_count"] + raw_wi,
            ]
            values = recording_array[first : first + samples, bodies].copy()
            if filename == "positions.bin":
                delta = new_offset - raw_world["display_offset"]
                values[:, :, :3] += delta
                values[:, :count, :3] += translation
                np.testing.assert_allclose(
                    values[:, :count, :3] - delta - translation,
                    recording_array[first : first + samples, bodies[:count], :3],
                    atol=2e-6,
                )
            else:
                np.testing.assert_array_equal(values, recording_array[first : first + samples, bodies])
            buffers.append(values)
        write_motion(meta, snapshot, *buffers)
        if target_id in overrides:
            camera = read(root / "paper-teaser/no-drills-closeups" / name / "camera.json")[0]
            camera = shift_camera(camera, new_offset - source_world["display_offset"])
        else:
            candidates = [run / "clip.json", run / "shots.json", run / "shots-updated.json", run / "shots-all.json"]
            camera = next(c for p in candidates if p.exists() for c in read(p) if c["name"] == name)
            camera = shift_camera(camera, new_offset - raw_world["display_offset"] + translation)
        camera.update(name=name, worlds=[target_id], startTime=0, duration=duration)
        camera_path = snapshot.parent / "camera.json"
        camera_path.write_text(json.dumps([camera], indent=2) + "\n")
        data = snapshot.parent / "data"
        prepare(source, snapshot, camera_path, data)
        assert not any("drill" in m["name"].lower() for m in read(data / "scene.json")["meshes"])
        plan.append(
            {
                "name": name,
                "data": str(data),
                "camera": str(camera_path),
                "frames": item["frames"],
                "recording": meta["source_recordings"][0],
            }
        )
    plan.append(prepare_finale(source, dressed, output, original, arrangement))
    (output / "render-plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def prepare_finale(source, dressed, output, original, arrangement):
    """Replicate continuous recorded motion onto the accepted 96-table layout."""
    name = "16-final-zoom-out"
    snapshot = output / name / "snapshot"
    link_geometry(dressed, snapshot)
    meta = copy.deepcopy(arrangement)
    buffers = []
    for filename in ("positions.bin", "rotations.bin"):
        recorded = np.memmap(source / filename, dtype="<f4", mode="r").reshape(
            original["sample_count"], original["body_count"], 4
        )
        still = np.fromfile(dressed / filename, dtype="<f4").reshape(meta["body_count"], 4)
        values = np.broadcast_to(still, (original["sample_count"], *still.shape)).copy()
        for wi, world in enumerate(meta["worlds"]):
            provenance = meta["replica_provenance"][wi]
            source_world = next(w for w in original["worlds"] if w["id"] == provenance["source_world"])
            target_slice = slice(world["body_start"], world["body_start"] + world["body_count"])
            source_slice = slice(source_world["body_start"], source_world["body_start"] + source_world["body_count"])
            values[:, target_slice] = recorded[:, source_slice]
            if filename == "positions.bin":
                delta = (
                    np.asarray(world["display_offset"])
                    - source_world["display_offset"]
                    + provenance["task_centering_translation"]
                )
                values[:, target_slice, :3] += delta
                np.testing.assert_allclose(
                    values[:, target_slice, :3] - delta, recorded[:, source_slice, :3], atol=3e-6
                )
            else:
                np.testing.assert_array_equal(values[:, target_slice], recorded[:, source_slice])

        buffers.append(values)
    meta.update(
        recording_composite=True,
        simultaneous_heterogeneous_batch=False,
        source_recordings=original["source_recordings"],
    )
    write_motion(meta, snapshot, *buffers)
    end = read(source.parent / "paper-teaser/overview.json")[0]
    # Preserve the still's horizontal coverage in a 16:9 video frame.
    end["fov"] = 12
    first = read(output / "08-plate-rack/camera.json")[0]
    first = shift_camera(first, np.asarray(arrangement["worlds"][38]["display_offset"]) - [0, 0, 0.805])
    # Keep focal length fixed so the visible footprint grows monotonically
    # rather than overshooting the final frame during the pullback.
    direction = np.asarray(first["position"]) - first["target"]
    scale = np.tan(np.deg2rad(first["fov"] / 2)) / np.tan(np.deg2rad(end["fov"] / 2))
    first["position"] = (np.asarray(first["target"]) + direction * scale).tolist()
    first["fov"] = end["fov"]
    camera = {
        "name": name,
        "worlds": [],
        "startTime": 0,
        "duration": 6,
        "position": first["position"],
        "target": first["target"],
        "fov": first["fov"],
        "keyframes": [
            {"time": time, **{key: pose[key] for key in ("position", "target", "fov")}}
            for time, pose in ((0, first), (0.6, first), (5.4, end), (6, end))
        ],
    }
    camera_path = snapshot.parent / "camera.json"
    camera_path.write_text(json.dumps([camera], indent=2) + "\n")
    end_path = snapshot.parent / "culling-camera.json"
    culling_views = [
        {
            "position": ((1 - u) * np.asarray(first["position"]) + u * np.asarray(end["position"])).tolist(),
            "target": ((1 - u) * np.asarray(first["target"]) + u * np.asarray(end["target"])).tolist(),
            "fov": end["fov"],
            "aspect": 16 / 9,
        }
        for u in np.linspace(0, 1, 25)
    ]
    end_path.write_text(json.dumps(culling_views, indent=2) + "\n")
    data = snapshot.parent / "data"
    prepare(source, snapshot, end_path, data)
    exported = read(data / "scene.json")
    assert any(m["world"] == 38 for m in exported["meshes"])
    assert not any("drill" in m["name"].lower() for m in exported["meshes"])
    return {
        "name": name,
        "data": str(data),
        "camera": str(camera_path),
        "frames": 180,
        "recording": "Continuous accepted composite replay replicated across 96 tiles; not a simultaneous simulation",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "dressed", "output"):
        parser.add_argument(key, type=Path)
    args = parser.parse_args()
    prepare_clips(args.root, args.dressed, args.output)
