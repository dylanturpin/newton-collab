# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Pack decorated recorded replays into linked Blender scenes for CUDA rendering."""

import hashlib
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Quaternion, Vector


def linear(rgb):
    return [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in rgb]


def studio(scene):
    floor = bpy.data.materials.new("checkerboard")
    floor.use_nodes = True
    nodes, links = floor.node_tree.nodes, floor.node_tree.links
    checker = nodes.new("ShaderNodeTexChecker")
    for name, rgb in (("Color1", (0.62, 0.67, 0.74)), ("Color2", (0.88, 0.90, 0.92))):
        checker.inputs[name].default_value = (*linear(rgb), 1)
    checker.inputs["Scale"].default_value = 1
    coords = nodes.new("ShaderNodeTexCoord")
    links.new(coords.outputs["Object"], checker.inputs["Vector"])
    links.new(checker.outputs["Color"], nodes.get("Principled BSDF").inputs["Base Color"])
    nodes.get("Principled BSDF").inputs["Roughness"].default_value = 0.72
    bpy.ops.mesh.primitive_plane_add(size=300, location=(0, 0, 0.005))
    bpy.context.object.data.materials.append(floor)
    world = bpy.data.worlds.new("studio")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.72, 0.76, 0.82, 1)
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.5
    scene.world = world
    for name, location, power, size, color in (
        ("large-key", (-8, -12, 22), 5500, 18, (1, 0.95, 0.87)),
        ("large-fill", (12, 5, 18), 4500, 20, (0.84, 0.91, 1)),
    ):
        light = bpy.data.lights.new(name, "AREA")
        light.energy, light.shape, light.size, light.color = power, "DISK", size, color
        obj = bpy.data.objects.new(name, light)
        scene.collection.objects.link(obj)
        obj.location = location
        obj.rotation_euler = (-obj.location).to_track_quat("-Z", "Y").to_euler()
    light = bpy.data.lights.new("soft-sun", "SUN")
    light.energy, light.angle = 1.5, 0.12
    obj = bpy.data.objects.new("soft-sun", light)
    scene.collection.objects.link(obj)
    obj.rotation_euler = (0.25, -0.35, -0.2)
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = 0
    scene.view_settings.gamma = 1


def export(data, camera, output, layout=None):
    output.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    meta = json.loads((data / "scene.json").read_text())
    vertices = np.memmap(data / "vertices.bin", "<f4", "r").reshape(-1, 16)
    indices = np.memmap(data / "indices.bin", "<u4", "r")
    shape = (meta["sample_count"], meta["body_count"], 4)
    positions = np.memmap(data / "positions.bin", "<f4", "r").reshape(shape)
    rotations = np.memmap(data / "rotations.bin", "<f4", "r").reshape(shape)
    np.savez_compressed(output / "motion.npz", positions=positions, rotations=rotations)
    collections = {}
    for wi in range(len(meta["worlds"])):
        col = bpy.data.collections.new(f"recorded-world-{wi}")
        collections[wi] = col
        if layout is None:
            scene.collection.children.link(col)
    materials, geometry, bodies = {}, {}, {}
    for record in meta["meshes"]:
        body, wi = record["body"], record["world"]
        if body not in bodies:
            obj = bpy.data.objects.new(f"body-{body}", None)
            obj["recorded_body"] = body
            obj.location = positions[0, body, :3]
            x, y, z, w = rotations[0, body]
            obj.rotation_mode = "QUATERNION"
            obj.rotation_quaternion = Quaternion((float(w), float(x), float(y), float(z)))
            collections[wi].objects.link(obj)
            bodies[body] = obj
        ix = indices[record["first_index"] : record["first_index"] + record["index_count"]]
        attrs = np.asarray(vertices[ix])
        if record["name"].endswith("/oak-finish"):
            # The baked Blender atlas was flipped for Metal's texture origin.
            # Restore its original UVs when importing it back into Blender.
            attrs = attrs.copy()
            attrs[:, 13] = 1 - attrs[:, 13]
        points = attrs[:, :3].reshape(-1, 3, 3)
        valid = np.linalg.norm(np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]), axis=1) > 1e-12
        attrs = attrs.reshape(-1, 3, 16)[valid].reshape(-1, 16)
        if not len(attrs):
            continue
        faces = np.arange(len(attrs), dtype=np.int32)
        spec = meta["materials"][int(attrs[0, 14]) - 1]
        material_key = json.dumps(spec, sort_keys=True)
        if material_key not in materials:
            mat = bpy.data.materials.new("surface")
            mat.use_nodes = True
            shader = mat.node_tree.nodes.get("Principled BSDF")
            shader.inputs["Base Color"].default_value = (*linear(spec["color"]), 1)
            shader.inputs["Roughness"].default_value = spec["roughness"]
            shader.inputs["Metallic"].default_value = spec["metallic"]
            shader.inputs["Transmission Weight"].default_value = spec.get("transmission", 0)
            shader.inputs["IOR"].default_value = spec.get("ior", 1.47)
            if spec.get("texture"):
                tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
                tex.image = bpy.data.images.load(str(data / spec["texture"]), check_existing=True)
                tint = mat.node_tree.nodes.new("ShaderNodeMixRGB")
                tint.blend_type = "MULTIPLY"
                tint.inputs[0].default_value = 1
                tint.inputs[2].default_value = (*linear(spec["color"]), 1)
                mat.node_tree.links.new(tex.outputs["Color"], tint.inputs[1])
                mat.node_tree.links.new(tint.outputs[0], shader.inputs["Base Color"])
            materials[material_key] = mat
        mat = materials[material_key]
        digest = hashlib.sha256(attrs[:, [0, 1, 2, 4, 5, 6, 12, 13]].tobytes() + faces.tobytes()).hexdigest()
        if digest not in geometry:
            mesh = bpy.data.meshes.new(record["name"])
            mesh.from_pydata(attrs[:, :3].tolist(), [], faces.reshape(-1, 3).tolist())
            mesh.polygons.foreach_set("use_smooth", np.ones(len(faces) // 3, dtype=bool))
            mesh.update()
            mesh.normals_split_custom_set_from_vertices(attrs[:, 4:7].tolist())
            uv = mesh.uv_layers.new(name="UVMap")
            uv.data.foreach_set("uv", attrs[faces, 12:14].ravel())
            mesh.materials.append(mat)
            geometry[digest] = mesh
        obj = bpy.data.objects.new(record["name"], geometry[digest])
        collections[wi].objects.link(obj)
        obj.parent = bodies[body]
        obj.material_slots[0].link = "OBJECT"
        obj.material_slots[0].material = mat
    if layout:
        for tile in layout["tiles"]:
            wi = tile["template"]
            obj = bpy.data.objects.new(f"replica-{tile['cell']:03d}", None)
            obj.instance_type = "COLLECTION"
            obj.instance_collection = collections[wi]
            obj.location = Vector(tile["position"]) - Vector(meta["worlds"][wi]["display_offset"])
            scene.collection.objects.link(obj)
    studio(scene)
    cam = bpy.data.cameras.new("replay-camera")
    scene.camera = bpy.data.objects.new("replay-camera", cam)
    scene.collection.objects.link(scene.camera)
    cam.sensor_fit, cam.sensor_height, cam.clip_end = "VERTICAL", 36, 400
    scene.camera.location = camera["position"]
    scene.camera.rotation_euler = (Vector(camera["target"]) - scene.camera.location).to_track_quat("-Z", "Y").to_euler()
    cam.lens = 36 / (2 * math.tan(math.radians(camera["fov"] / 2)))
    scene.render.engine = "CYCLES"
    scene.render.resolution_x, scene.render.resolution_y = 1920, 1080
    scene.render.resolution_percentage = 100
    scene.render.fps = 30
    scene.render.use_persistent_data = True
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "scene.blend"), compress=True)
    (output / "camera.json").write_text(json.dumps(camera, indent=2))
    (output / "metadata.json").write_text(
        json.dumps(
            {
                "recording_fps": meta["recording_fps"],
                "sample_count": meta["sample_count"],
                "body_count": meta["body_count"],
                "trace_sha256": meta["trace_sha256"],
                "rendered_scenes": 384 if layout else len(meta["worlds"]),
                "samples": 8 if layout else 16,
                "source": str(data),
                "unique_meshes": len(geometry),
                "body_controllers": len(bodies),
            },
            indent=2,
        )
    )
    print("EXPORTED", output.name, len(geometry), "shared meshes", flush=True)


if __name__ == "__main__":
    arguments = sys.argv[sys.argv.index("--") + 1 :]
    root, output = map(Path, arguments[:2])
    spec = json.loads((root / "hq-clips-no-drills/package-spec.json").read_text())
    for item in spec:
        if len(arguments) > 2 and item["name"] != arguments[2]:
            continue
        dest = output / item["name"]
        if (dest / "metadata.json").exists():
            continue
        camera = json.loads(Path(item["camera"]).read_text())[0]
        layout = None
        data = Path(item["data"])
        if item["name"] == "16-final-zoom-out":
            data = root / "realtime-iteration/data"
            layout = json.loads((root / "cycles-384-preview/layout.json").read_text())
            end = json.loads((root / "cycles-384-preview/frame-filling-camera.json").read_text())
            tile = min(
                (t for t in layout["tiles"] if t["kind"] == "gear"), key=lambda t: sum(v * v for v in t["position"][:2])
            )
            source_camera = json.loads((root / "hq-clips-no-drills/05-wrecking-ball/camera.json").read_text())[0]
            target = np.array(source_camera["target"]) + np.array(tile["position"]) - [0, 0, 0.805]
            eye = target + (np.array(source_camera["position"]) - source_camera["target"]) * math.tan(
                math.radians(source_camera["fov"] / 2)
            ) / math.tan(math.radians(6))
            for key in camera["keyframes"][:2]:
                key.update(position=eye.tolist(), target=target.tolist(), fov=12)
            for key in camera["keyframes"][-2:]:
                key.update(position=end["position"], target=end["target"], fov=end["fov"])
            duration_scale = 14 / camera["duration"]
            for key in camera["keyframes"]:
                key["time"] *= duration_scale
            camera["playbackSpeed"] /= duration_scale
            camera["duration"] = 14
            camera.update(position=eye.tolist(), target=target.tolist(), fov=12)
        export(data, camera, dest, layout)
