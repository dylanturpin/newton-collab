# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import argparse
import hashlib
import json
import sys
from pathlib import Path

import bpy
import numpy as np

parser = argparse.ArgumentParser(description="Export the kitchen Blender replay to AVBD Metal render buffers")
parser.add_argument("--run", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--builtin-floor", action="store_true")
parser.add_argument("--studio-floor-width", type=float, default=6.0)
parser.add_argument("--glass-wall-thickness", type=float, default=0.012)
args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
if not 0 < args.glass_wall_thickness < 0.17:
    parser.error("glass wall thickness must be between 0 and 0.17 metres")
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=True)
result = json.loads((args.run / "result.json").read_text())
assert result["substeps"] == 8 and result["iterations"] == 10
trace = np.load(args.run / "trace.npz")
poses = trace["poses"]
assert poses.shape == (1381, 61, 7)
objects = [(0, bpy.data.objects["Glass bin | 12 mm walls"])]
objects += sorted((int(o["fpgs_body_index"]), o) for o in bpy.context.scene.objects if "fpgs_body_index" in o)
if not args.builtin_floor:
    objects += [(61, bpy.data.objects["Table"]), (62, bpy.data.objects["Studio floor"])]
body_count = 61 if args.builtin_floor else 63
if args.builtin_floor:
    assert result["support"] == "floor" and result["support_height_m"] == 0.005
materials = []
material_ids = {}
vertices = []
indices = []
mesh_records = []
for body, obj in objects:
    mesh = obj.data
    mat = obj.active_material
    if mat.name not in material_ids:
        material_ids[mat.name] = len(materials) + 1
        if body == 0:
            materials.append(
                {
                    "name": mat.name,
                    "color": [1, 1, 1],
                    "roughness": 0.015,
                    "transmission": 1,
                    "ior": 1.5,
                    "clearcoat": 0,
                }
            )
        else:
            shader = mat.node_tree.nodes.get("Principled BSDF")
            materials.append(
                {
                    "name": mat.name,
                    "color": list(shader.inputs["Base Color"].default_value)[:3],
                    "roughness": shader.inputs["Roughness"].default_value,
                    "transmission": 0,
                    "ior": 1.5,
                    "clearcoat": shader.inputs["Coat Weight"].default_value,
                }
            )
    mesh.calc_loop_triangles()
    start = len(vertices)
    for tri in mesh.loop_triangles:
        for loop in tri.loops:
            p = mesh.vertices[mesh.loops[loop].vertex_index].co.copy()
            if body == 0 and abs(p.x) < 0.169 and abs(p.y) < 0.169:
                # Move the inner shell only; preserve the exterior dimensions
                # and recorded poses. Its axis-aligned corner normals stay valid.
                inner_radius = 0.17 - args.glass_wall_thickness
                p.x = inner_radius if p.x > 0 else -inner_radius
                p.y = inner_radius if p.y > 0 else -inner_radius
                if p.z < 0.64:
                    p.z = args.glass_wall_thickness
            if body == 62:
                # Keep the render-only studio plane near the SI-scale scene
                # instead of tracing the original 200 m Blender backdrop.
                p.x *= args.studio_floor_width / obj.dimensions.x
                p.y *= args.studio_floor_width / obj.dimensions.y
            n = mesh.corner_normals[loop].vector
            vertices.append(
                [
                    *p,
                    0,
                    *n,
                    max(materials[material_ids[mat.name] - 1]["roughness"], 0.001),
                    1,
                    1,
                    1,
                    0,
                    0,
                    0,
                    material_ids[mat.name],
                    0,
                ]
            )
            indices.append(len(indices))
    mesh_records.append({"name": obj.name, "body": body, "first_vertex": start, "vertex_count": len(vertices) - start})
a = np.asarray(vertices, dtype="<f4")
for item in mesh_records:
    start = item["first_vertex"]
    a[start : start + item["vertex_count"], 3].view("<u4")[:] = item["body"]
a.tofile(out / "vertices.bin")
np.asarray(indices, dtype="<u4").tofile(out / "indices.bin")
positions = np.ones((len(poses), body_count, 4), dtype="<f4")
rotations = np.zeros_like(positions)
positions[:, :61, :3] = poses[:, :, :3]
rotations[:, :61, :] = poses[:, :, 3:7]
for body, obj in objects[61:]:
    positions[:, body, :3] = obj.location[:]
    q = obj.rotation_euler.to_quaternion()
    rotations[:, body] = [q.x, q.y, q.z, q.w]
assert np.array_equal(positions[:, :61, :3], poses[:, :, :3])
assert np.array_equal(rotations[:, :61, :], poses[:, :, 3:7])
positions.tofile(out / "positions.bin")
rotations.tofile(out / "rotations.bin")
metadata = {
    "body_count": body_count,
    "builtin_ground": args.builtin_floor,
    "recorded_body_count": 61,
    "sample_count": len(poses),
    "recording_fps": 60,
    "substeps": 8,
    "iterations": 10,
    "index_count": len(indices),
    "vertex_count": len(vertices),
    "materials": materials,
    "meshes": mesh_records,
    "trace_sha256": hashlib.sha256((args.run / "trace.npz").read_bytes()).hexdigest(),
    "vertex_stride": 64,
    "sharp_angle_degrees": 30,
    "source_blend": bpy.data.filepath,
    "studio_floor_width_m": args.studio_floor_width,
    "glass_wall_thickness_m": args.glass_wall_thickness,
}
(out / "scene.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(json.dumps({k: v for k, v in metadata.items() if k not in ["meshes", "materials"]}, indent=2))
