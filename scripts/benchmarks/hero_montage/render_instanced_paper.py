# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Blender still export using linked full-resolution meshes across recorded tiles.

Run with Blender --background --python SCRIPT -- SOURCE REPLICAS CAMERA OUTPUT.
"""

import argparse
import collections
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Quaternion, Vector


def lighter_material(spec):
    """Restyle lacquer in teal and metallic trim in the warm scene palette."""
    spec = dict(spec)
    spec.pop("texture", None)
    if spec["metallic"] > 0.5:
        spec.update(color=[0.96, 0.66, 0.16], metallic=0.7, roughness=0.25)
    else:
        spec.update(color=[0.03, 0.66, 0.51], metallic=0.12, roughness=0.25)
    return spec


def configure_cycles(scene):
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 128
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.025
    scene.cycles.use_denoising = True
    scene.cycles.denoising_use_gpu = True
    scene.cycles.max_bounces = 10
    preferences = bpy.context.preferences.addons["cycles"].preferences
    try:
        preferences.compute_device_type = "METAL"
        preferences.get_devices()
        for device in preferences.devices:
            device.use = device.type == "METAL"
        scene.cycles.device = "GPU" if any(d.use for d in preferences.devices) else "CPU"
    except TypeError:
        scene.cycles.device = "CPU"


def render(source, replicas, camera_path, output, single_scene=False):
    source, replicas, camera_path, output = (path.resolve() for path in (source, replicas, camera_path, output))
    started = time.monotonic()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    original = json.loads((source / "scene.json").read_text())
    arrangement = json.loads((replicas / "scene.json").read_text())
    camera_spec = json.loads(camera_path.read_text())[0]
    if not single_scene:
        assert len(arrangement["worlds"]) == 96
    vertices = np.memmap(source / "vertices.bin", dtype="<f4", mode="r").reshape(-1, 16)
    indices = np.memmap(source / "indices.bin", dtype="<u4", mode="r")
    poses = np.fromfile(replicas / "positions.bin", "<f4").reshape(-1, 4)
    quats = np.fromfile(replicas / "rotations.bin", "<f4").reshape(-1, 4)
    replica_vertices = np.memmap(replicas / "vertices.bin", dtype="<f4", mode="r").reshape(-1, 16)
    replica_indices = np.memmap(replicas / "indices.bin", dtype="<u4", mode="r")
    material_cache = {}

    def material(spec):
        key = json.dumps(spec, sort_keys=True)
        if key in material_cache:
            return material_cache[key]
        mat = bpy.data.materials.new("surface-" + hashlib.sha256(key.encode()).hexdigest()[:12])
        mat.use_nodes = True
        bsdf = mat.node_tree.nodes.get("Principled BSDF")
        color = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in spec["color"]] + [1]
        bsdf.inputs["Base Color"].default_value = color
        bsdf.inputs["Roughness"].default_value = spec["roughness"]
        bsdf.inputs["Metallic"].default_value = spec["metallic"]
        bsdf.inputs["Transmission Weight"].default_value = spec.get("transmission", 0)
        bsdf.inputs["IOR"].default_value = spec.get("ior", 1.47)
        if spec.get("texture"):
            tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
            texture_path = replicas / spec["texture"]
            if not texture_path.exists():
                texture_path = source / spec["texture"]
            tex.image = bpy.data.images.load(str(texture_path), check_existing=True)
            multiply = mat.node_tree.nodes.new("ShaderNodeMixRGB")
            multiply.blend_type = "MULTIPLY"
            multiply.inputs[0].default_value = 1
            multiply.inputs[2].default_value = color
            mat.node_tree.links.new(tex.outputs["Color"], multiply.inputs[1])
            mat.node_tree.links.new(multiply.outputs[0], bsdf.inputs["Base Color"])
        material_cache[key] = mat
        return mat

    # Match surviving records to full-resolution originals. Iterating the dressed
    # arrangement also preserves explicit deletions and appended set dressing.
    source_lookup = collections.defaultdict(list)
    for record in original["meshes"]:
        source_lookup[(record["world"], record["body"], record["name"])].append(record)
    worlds = {w["id"]: (i, w) for i, w in enumerate(original["worlds"])}
    counters = collections.Counter()
    geometry = {}
    object_count = 0
    source_geometry = set()
    decor_object_count = 0
    for record in arrangement["meshes"]:
        cell = record["world"]
        tile = arrangement["worlds"][cell]
        if single_scene and tile["id"] not in camera_spec["worlds"]:
            continue
        source_wi, source_world = worlds[arrangement["replica_provenance"][cell]["source_world"]]
        body = record["body"]
        source_body = (
            body - tile["body_start"] + source_world["body_start"]
            if body < arrangement["recorded_body_count"]
            else original["recorded_body_count"] + source_wi
        )
        key = (source_wi, source_body, record["name"])
        replica_tri = replica_indices[record["first_index"] : record["first_index"] + record["index_count"]]
        spec = arrangement["materials"][int(replica_vertices[replica_tri[0], 14]) - 1]
        if "/retro_lighter" in record["name"]:
            spec = lighter_material(spec)
        if source_lookup.get(key):
            candidates = source_lookup[key]
            source_record = candidates[counters[(cell, key)] % len(candidates)]
            counters[(cell, key)] += 1
            geometry_key = ("source", source_record["first_index"])
            source_geometry.add(geometry_key)
            tri = indices[source_record["first_index"] : source_record["first_index"] + source_record["index_count"]]
            vertex_data = vertices
        else:
            geometry_key = ("decor", record["first_index"])
            tri, vertex_data = replica_tri, replica_vertices
        if geometry_key not in geometry:
            attrs = vertex_data[tri]
            triangle_points = attrs[:, :3].reshape(-1, 3, 3)
            valid = (
                np.linalg.norm(
                    np.cross(
                        triangle_points[:, 1] - triangle_points[:, 0], triangle_points[:, 2] - triangle_points[:, 0]
                    ),
                    axis=1,
                )
                > 1e-12
            )
            attrs = attrs.reshape(-1, 3, 16)[valid].reshape(-1, 16)
            faces = np.arange(len(attrs), dtype="i4")
            mesh = bpy.data.meshes.new(record["name"])
            mesh.from_pydata(attrs[:, :3].tolist(), [], faces.reshape(-1, 3).tolist())
            mesh.polygons.foreach_set("use_smooth", np.ones(len(faces) // 3, dtype=bool))
            mesh.update(calc_edges=True)
            mesh.normals_split_custom_set_from_vertices(attrs[:, 4:7].tolist())
            uv = mesh.uv_layers.new(name="UVMap")
            uv.data.foreach_set("uv", attrs[faces, 12:14].ravel())
            mesh.materials.append(material(spec))
            geometry[geometry_key] = mesh
        mesh = geometry[geometry_key]
        obj = bpy.data.objects.new(f"{tile['id']}/{record['name']}", mesh)
        bpy.context.scene.collection.objects.link(obj)
        obj.location = poses[body, :3]
        obj.rotation_mode = "QUATERNION"
        x, y, z, w = quats[body]
        obj.rotation_quaternion = Quaternion((float(w), float(x), float(y), float(z)))
        obj.material_slots[0].link = "OBJECT"
        obj.material_slots[0].material = material(spec)
        object_count += 1
        decor_object_count += record["name"].startswith("paper-decor/")
        if object_count % 1000 == 0:
            print(f"Imported {object_count} objects, {len(geometry)} shared meshes", flush=True)
    source_mesh_count = len(source_geometry)
    bpy.ops.mesh.primitive_plane_add(size=180, location=(0, 0, 0.005))
    ground = bpy.context.object
    floor = bpy.data.materials.new("checkerboard")
    floor.use_nodes = True
    nodes = floor.node_tree.nodes
    checker = nodes.new("ShaderNodeTexChecker")
    for name, rgb in (("Color1", (0.62, 0.67, 0.74)), ("Color2", (0.88, 0.90, 0.92))):
        checker.inputs[name].default_value = [
            c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb
        ] + [1]
    checker.inputs["Scale"].default_value = 1
    coordinates = nodes.new("ShaderNodeTexCoord")
    floor.node_tree.links.new(coordinates.outputs["Object"], checker.inputs["Vector"])
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Roughness"].default_value = 0.72
    floor.node_tree.links.new(checker.outputs["Color"], bsdf.inputs["Base Color"])
    ground.data.materials.append(floor)
    world = bpy.data.worlds.new("studio")
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs[0].default_value = (0.72, 0.76, 0.82, 1)
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.5
    scene = bpy.context.scene
    scene.world = world

    def area_light(name, location, energy, size, color):
        light = bpy.data.lights.new(name, "AREA")
        light.energy, light.shape, light.size, light.color = energy, "DISK", size, color
        obj = bpy.data.objects.new(name, light)
        scene.collection.objects.link(obj)
        obj.location = location
        obj.rotation_euler = (Vector((0, 0, 0)) - obj.location).to_track_quat("-Z", "Y").to_euler()

    area_light("large-key", (-8, -12, 22), 5500, 18, (1, 0.95, 0.87))
    area_light("large-fill", (12, 5, 18), 4500, 20, (0.84, 0.91, 1))
    sun = bpy.data.lights.new("soft-sun", "SUN")
    sun.energy, sun.angle = 1.5, 0.12
    sun_obj = bpy.data.objects.new("soft-sun", sun)
    scene.collection.objects.link(sun_obj)
    sun_obj.rotation_euler = (0.25, -0.35, -0.2)
    cam = bpy.data.cameras.new("teaser-camera")
    obj = bpy.data.objects.new("teaser-camera", cam)
    scene.collection.objects.link(obj)
    obj.location = camera_spec["position"]
    obj.rotation_euler = (Vector(camera_spec["target"]) - obj.location).to_track_quat("-Z", "Y").to_euler()
    cam.sensor_fit = "VERTICAL"
    cam.sensor_height = 36
    cam.lens = 36 / (2 * math.tan(math.radians(camera_spec["fov"] / 2)))
    cam.clip_end = 250
    scene.camera = obj
    configure_cycles(scene)
    scene.render.resolution_x, scene.render.resolution_y = (1600, 1000) if single_scene else (3600, 2400)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.filepath = str(output)
    scene.view_settings.view_transform = "AgX"
    try:
        scene.view_settings.look = "AgX - Medium High Contrast"
    except TypeError:
        pass
    scene.view_settings.gamma = 1
    scene.view_settings.exposure = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    setup_seconds = time.monotonic() - started
    print(
        f"{object_count} linked objects share {source_mesh_count} original meshes; setup {setup_seconds:.1f}s",
        flush=True,
    )
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(
        filepath=str(output.parent / ("scene.blend" if single_scene else "instanced-96.blend")), compress=True
    )
    render_started = time.monotonic()
    bpy.ops.render.render(write_still=True)
    (output.parent / "blender-render-report.json").write_text(
        json.dumps(
            {
                "renderer": "Blender Cycles",
                "tiles": 1 if single_scene else 96,
                "linked_objects": object_count,
                "unique_source_meshes": source_mesh_count,
                "unique_geometry_meshes": len(geometry),
                "decoration_objects": decor_object_count,
                "device": scene.cycles.device,
                "gpu_denoising_requested": scene.cycles.denoising_use_gpu,
                "arrangement_export": str(replicas),
                "mesh_simplification": False,
                "setup_seconds": setup_seconds,
                "render_seconds": time.monotonic() - render_started,
                "width": scene.render.resolution_x,
                "height": scene.render.resolution_y,
                "samples": scene.cycles.samples,
                "physics_steps_during_render": 0,
                "packed_textures": True,
                "view_transform": scene.view_settings.view_transform,
                "exposure": scene.view_settings.exposure,
                "gamma": scene.view_settings.gamma,
                "material_input_colors": "sRGB converted to scene-linear",
                "source_export": str(source.resolve()),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("replicas", type=Path)
    parser.add_argument("camera", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--single-scene", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    render(args.source, args.replicas, args.camera, args.output, args.single_scene)
