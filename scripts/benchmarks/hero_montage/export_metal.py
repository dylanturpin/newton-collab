# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Export the accepted Newton model and unchanged poses for local Metal replay."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import warp as wp
from scene import build_template

import newton


def primitive(kind, scale):
    import trimesh

    if kind == newton.GeoType.BOX:
        mesh = trimesh.creation.box(extents=2 * scale)
        flat = True
    elif kind == newton.GeoType.SPHERE:
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=scale[0])
        flat = False
    elif kind == newton.GeoType.CYLINDER:
        assert scale[2] == 0 or abs(scale[2] - scale[0]) < 1.0e-6, scale
        mesh = trimesh.creation.cylinder(radius=scale[0], height=2 * scale[1], sections=96)
        flat = True
    elif kind == newton.GeoType.CAPSULE:
        mesh = trimesh.creation.capsule(radius=scale[0], height=2 * scale[1], count=(24, 48))
        mesh.vertices -= mesh.bounds.mean(axis=0)
        flat = False
    else:
        raise ValueError(f"Unsupported visible geometry {kind}: {scale}")
    if flat:
        vertices = mesh.vertices[mesh.faces].reshape(-1, 3)
        normals = np.repeat(mesh.face_normals, 3, axis=0)
        if kind == newton.GeoType.CYLINDER:
            side = np.abs(normals[:, 2]) < 0.5
            normals[side, :2] = vertices[side, :2] / scale[0]
        indices = np.arange(len(vertices), dtype=np.uint32)
    else:
        vertices, normals, indices = mesh.vertices, mesh.vertex_normals, mesh.faces.ravel()
    return np.asarray(vertices), np.asarray(normals), np.asarray(indices), np.zeros((len(vertices), 2))


def main():
    import trimesh
    from PIL import Image
    from scipy.spatial.transform import Rotation

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--assets", type=Path, default=Path("assets.json"))
    parser.add_argument("--roster", type=Path, default=Path("assets/hero_roster.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0", help="Geometry/IK construction only; no simulation is stepped")
    parser.add_argument(
        "--diagnostic", action="store_true", help="Export a failed probe for inspection; never a deliverable"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "textures").mkdir(exist_ok=True)
    report = json.loads((args.run / "report.json").read_text())
    summary = json.loads((args.run / "model-summary.json").read_text())
    assert args.diagnostic or not any(w.get("lighter_force_reference") for w in summary["worlds"]), (
        "Force-authoring reference recordings are diagnostic only"
    )
    assert report["quality_gate_passed"] or args.diagnostic
    trace_path = args.run / "trace.npz"
    assert hashlib.sha256(trace_path.read_bytes()).hexdigest() == report["trace_sha256"]
    recorded = np.load(trace_path)
    poses = recorded["poses"]
    assert np.isfinite(poses).all()
    wp.init()
    wp.set_device(args.device)
    assets = json.loads(args.assets.read_text())
    builder = newton.ModelBuilder()
    for i, world in enumerate(summary["worlds"]):
        print(f"Exporting geometry {world['id']}", flush=True)
        template, _ = build_template(world["kind"], world["variant"], assets, args.device)
        assert template.body_count == world["body_count"]
        builder.add_world(template, label_prefix=f"{i:02d}_{world['kind']}")
    model = builder.finalize(device=args.device)
    assert list(model.body_label) == summary["body_labels"]
    assert poses.shape[1] == model.body_count
    order = json.loads(args.roster.read_text())["display_order"]
    if args.diagnostic:
        order = [world["id"] for world in summary["worlds"]]
    columns = math.ceil(math.sqrt(len(order)))
    offsets = []
    worlds = []
    for world in summary["worlds"]:
        cell = order.index(world["id"])
        # Only display translations change. The original collision floor at
        # -0.8 m coincides with the Metal renderer's built-in floor at 0.005 m.
        offset = np.array([(cell % columns - 2) * 2.6, (cell // columns - 1.5) * 2.6, 0.805])
        offsets.append(offset)
        worlds.append({**world, "display_offset": offset.tolist()})
    bodies = model.body_count + len(worlds)
    positions = np.ones((len(poses), bodies, 4), dtype="<f4")
    rotations = np.zeros_like(positions)
    positions[:, : model.body_count, :3] = poses[:, :, :3]
    rotations[:, : model.body_count] = poses[:, :, 3:]
    for i, world in enumerate(worlds):
        start, count = world["body_start"], world["body_count"]
        positions[:, start : start + count, :3] += offsets[i]
        positions[:, model.body_count + i, :3] = offsets[i]
        rotations[:, model.body_count + i, 3] = 1
        np.testing.assert_allclose(
            positions[:, start : start + count, :3] - offsets[i], poses[:, start : start + count, :3], atol=5.0e-7
        )
    assert np.array_equal(rotations[:, : model.body_count], poses[:, :, 3:])
    positions.tofile(args.output / "positions.bin")
    rotations.tofile(args.output / "rotations.bin")

    types = model.shape_type.numpy()
    flags = model.shape_flags.numpy()
    parents = model.shape_body.numpy()
    shape_worlds = model.shape_world.numpy()
    scales = model.shape_scale.numpy()
    transforms = model.shape_transform.numpy()
    colors = model.shape_color.numpy()
    vertices, indices, records, materials = [], [], [], []
    material_ids, texture_ids = {}, {}
    vertex_count, index_count = 0, 0
    skipped_labels = []
    hidden_mounts = []
    world_kinds = [w["kind"] for w in summary["worlds"]]
    for i, kind in enumerate(types):
        if not flags[i] & int(newton.ShapeFlags.VISIBLE):
            continue
        label = model.shape_label[i]
        world_kind = world_kinds[int(shape_worlds[i])]
        body_label = model.body_label[parents[i]] if parents[i] >= 0 else ""
        if (world_kind == "hand" and "/iiwa14/" in body_label) or (
            world_kind == "shadow"
            and body_label
            and (
                "/ur10e/" in body_label
                or "/instrument/" in body_label
                or body_label.endswith(("rh_forearm", "rh_wrist"))
            )
        ):
            # The user requested floating hands in the teaser. Keep every
            # recorded body pose and collision; omit only mounting visuals.
            hidden_mounts.append(label)
            continue
        if "furnishing/label" in label:
            skipped_labels.append(label)
            continue
        source = model.shape_source[i]
        roughness, metallic, texture_name = 0.42, 0.0, None
        if kind in (newton.GeoType.MESH, newton.GeoType.CONVEX_MESH):
            xyz = np.asarray(source.vertices, dtype=np.float64) * scales[i]
            tri = np.asarray(source.indices, dtype=np.uint32).reshape(-1, 3).copy()
            if np.prod(scales[i]) < 0:
                tri = tri[:, [0, 2, 1]]
            normals = getattr(source, "normals", None)
            if normals is None:
                normals = trimesh.Trimesh(xyz, tri, process=False).vertex_normals
            else:
                normals = np.asarray(normals, dtype=np.float64) / scales[i]
            uv = getattr(source, "uvs", None)
            uv = np.zeros((len(xyz), 2)) if uv is None else np.asarray(uv).copy()
            uv[:, 1] = 1 - uv[:, 1]
            tri = tri.ravel()
            roughness = float(source.roughness if source.roughness is not None else 0.42)
            metallic = float(source.metallic if source.metallic is not None else 0)
            texture = getattr(source, "texture", None)
            if texture is not None:
                pixels = np.asarray(Image.open(texture) if isinstance(texture, str | Path) else texture)
                if pixels.dtype != np.uint8:
                    pixels = np.clip(pixels * 255, 0, 255).astype(np.uint8)
                digest = hashlib.sha256(pixels.tobytes()).hexdigest()
                if digest not in texture_ids:
                    texture_ids[digest] = f"textures/{digest[:16]}.png"
                    Image.fromarray(pixels).save(args.output / texture_ids[digest])
                texture_name = texture_ids[digest]
        else:
            xyz, normals, tri, uv = primitive(kind, scales[i])
            if "wrecking_chain_link" in label or label.endswith("wrecking_ball"):
                roughness, metallic = 0.24, 0.85
        rotation = Rotation.from_quat(transforms[i, 3:])
        xyz = rotation.apply(xyz) + transforms[i, :3]
        normals = rotation.apply(np.array(normals, dtype=np.float64, copy=True))
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-12)
        material = {
            "color": colors[i].tolist(),
            "roughness": max(0.08, roughness),
            "metallic": metallic,
            "texture": texture_name,
        }
        if label.endswith("procgen_visual/pour_jar"):
            material.update(color=[0.98, 0.995, 1.0], roughness=0.015, transmission=0.98, ior=1.47)
        key = json.dumps(material, sort_keys=True)
        if key not in material_ids:
            material_ids[key] = len(materials) + 1
            materials.append(material)
        world = int(shape_worlds[i])
        body = int(parents[i]) if parents[i] >= 0 else model.body_count + world
        data = np.zeros((len(xyz), 16), dtype="<f4")
        data[:, :3] = xyz
        data[:, 3].view("<u4")[:] = body
        data[:, 4:7] = normals
        data[:, 7] = material["roughness"]
        data[:, 8:11] = 1
        data[:, 11] = metallic
        data[:, 12:14] = uv
        data[:, 14] = material_ids[key]
        vertices.append(data)
        indices.append(tri.astype("<u4") + vertex_count)
        records.append(
            {"name": label, "world": world, "body": body, "first_index": index_count, "index_count": len(tri)}
        )
        vertex_count += len(xyz)
        index_count += len(tri)
    np.concatenate(vertices).tofile(args.output / "vertices.bin")
    np.concatenate(indices).tofile(args.output / "indices.bin")
    metadata = {
        "body_count": bodies,
        "recorded_body_count": model.body_count,
        "sample_count": len(poses),
        "recording_fps": int(recorded["fps"]),
        "substeps": report["substeps"],
        "iterations": report["iterations"],
        "vertex_count": vertex_count,
        "vertex_stride": 64,
        "index_count": index_count,
        "materials": materials,
        "meshes": records,
        "worlds": worlds,
        "trace_sha256": report["trace_sha256"],
        "builtin_ground": True,
        "omitted_visual_placards": skipped_labels,
        "omitted_hand_mount_visuals": hidden_mounts,
        "simulation_steps_executed": 0,
        "diagnostic": args.diagnostic,
        "quality_gate_passed": report["quality_gate_passed"],
    }
    (args.output / "scene.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        json.dumps(
            {
                "vertices": vertex_count,
                "triangles": index_count // 3,
                "materials": len(materials),
                "textures": len(texture_ids),
                "worlds": len(worlds),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
