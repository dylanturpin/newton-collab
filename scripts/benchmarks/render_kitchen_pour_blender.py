# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build a self-contained Blender replay of the recorded kitchen-pour poses.

Run with Blender's Python interpreter, passing arguments after ``--``. The
saved animation contains every 60 Hz recorded pose, with no rigid-body world,
simulation modifiers, or external assets needed for playback.
"""

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import bpy
import numpy as np
from mathutils import Vector


def _material(name, color, *, roughness=0.25, metallic=0.0):
    color = tuple(v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in color)
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1)
    shader.inputs["Roughness"].default_value = roughness
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Coat Weight"].default_value = 0.2
    shader.inputs["Coat Roughness"].default_value = 0.22
    material.diffuse_color = (*color, 1)
    return material


def _glass():
    material = bpy.data.materials.new("Clear glass | IOR 1.5")
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    nodes.remove(nodes.get("Principled BSDF"))
    shader = nodes.new("ShaderNodeBsdfGlass")
    shader.inputs["Color"].default_value = (1, 1, 1, 1)
    shader.inputs["Roughness"].default_value = 0.015
    shader.inputs["IOR"].default_value = 1.5
    links.new(shader.outputs[0], nodes.get("Material Output").inputs["Surface"])
    material.use_raytrace_refraction = True
    return material


def _glass_bin(parent):
    # The exterior matches the five collision panels. A continuous shell
    # avoids artificial internal glass/air interfaces at panel joins.
    vertices = []
    for radius, z in ((0.17, 0), (0.17, 0.65), (0.158, 0.65), (0.158, 0.012)):
        vertices.extend(
            (x, y, z) for x, y in ((-radius, -radius), (radius, -radius), (radius, radius), (-radius, radius))
        )
    faces = [(3, 2, 1, 0), (12, 13, 14, 15)]
    for i in range(4):
        j = (i + 1) % 4
        faces.extend(((i, j, j + 4, i + 4), (i + 4, j + 4, j + 8, i + 8), (i + 8, j + 8, j + 12, i + 12)))
    mesh = bpy.data.meshes.new("Continuous 12 mm glass bin")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new("Glass bin | 12 mm walls", mesh)
    bpy.context.collection.objects.link(obj)
    obj.parent = parent
    mesh.materials.append(_glass())


def _box(name, position, half_size, material, *, parent=None):
    bpy.ops.mesh.primitive_cube_add(size=2, location=position)
    obj = bpy.context.object
    obj.name = name
    obj.scale = half_size
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material)
    if parent is not None:
        obj.parent = parent
    return obj


def _light(name, position, target, energy, size):
    data = bpy.data.lights.new(name, "AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    obj = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(obj)
    obj.location = position
    obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()


def _key_poses(obj, poses):
    obj.rotation_mode = "QUATERNION"
    # Blender stores wxyz, while the trace stores xyzw. Keep signs continuous
    # so interpolation cannot take an unnecessary turn between samples.
    rotations = poses[:, [6, 3, 4, 5]].copy()
    for frame in range(1, len(rotations)):
        if np.dot(rotations[frame - 1], rotations[frame]) < 0:
            rotations[frame] *= -1
    for sample, (pose, rotation) in enumerate(zip(poses, rotations, strict=True)):
        frame = 1 + sample / 2
        obj.location = pose[:3]
        obj.rotation_quaternion = rotation
        obj.keyframe_insert(data_path="location", frame=frame, group="Recorded position")
        obj.keyframe_insert(data_path="rotation_quaternion", frame=frame, group="Recorded rotation")
    action = obj.animation_data.action
    for layer in action.layers:
        for strip in layer.strips:
            bag = strip.channelbag(obj.animation_data.action_slot)
            for curve in bag.fcurves:
                for key in curve.keyframe_points:
                    key.interpolation = "LINEAR"


def _configure_render(scene, args):
    scene.render.engine = args.engine
    scene.render.resolution_x = args.width
    scene.render.resolution_y = args.height
    scene.render.resolution_percentage = 100
    scene.render.fps = 30
    scene.render.fps_base = 1
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.image_settings.compression = 15
    scene.render.filepath = str(args.output / "frames" / "frame_")
    scene.render.film_transparent = False
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.exposure = -0.5
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs[0].default_value = (0.68, 0.74, 0.84, 1)
    scene.world.node_tree.nodes["Background"].inputs[1].default_value = 0.3
    if args.engine == "CYCLES":
        preferences = bpy.context.preferences.addons["cycles"].preferences
        preferences.compute_device_type = "METAL"
        preferences.get_devices()
        devices = []
        for device in preferences.devices:
            device.use = device.type == "METAL"
            if device.use:
                devices.append(device.name)
        if not devices:
            raise RuntimeError("No Metal GPU available for this local render")
        scene.cycles.device = "GPU"
        scene.render.use_persistent_data = True
        scene.cycles.samples = args.samples
        scene.cycles.use_denoising = True
        scene.cycles.denoising_use_gpu = True
        scene.cycles.denoising_prefilter = "FAST"
        scene.cycles.denoising_quality = "BALANCED"
        scene.cycles.max_bounces = 16
        scene.cycles.transmission_bounces = 12
        scene.cycles.glossy_bounces = 8
        scene.cycles.transparent_max_bounces = 16
        scene.cycles.use_adaptive_sampling = True
        scene.cycles.adaptive_threshold = 0.025
        print(f"Rendering on Metal: {devices}", flush=True)
    else:
        scene.eevee.taa_render_samples = args.samples
        scene.eevee.use_raytracing = False
        scene.eevee.shadow_ray_count = 1
        scene.eevee.shadow_step_count = 6
        # Denoise stochastic shadows while retaining the recorded geometry.
        group = bpy.data.node_groups.new("Replay denoising", "CompositorNodeTree")
        group.interface.new_socket(name="Image", in_out="OUTPUT", socket_type="NodeSocketColor")
        layers = group.nodes.new("CompositorNodeRLayers")
        denoise = group.nodes.new("CompositorNodeDenoise")
        output = group.nodes.new("NodeGroupOutput")
        group.links.new(layers.outputs["Image"], denoise.inputs["Image"])
        group.links.new(denoise.outputs["Image"], output.inputs["Image"])
        scene.compositing_node_group = group
        scene.render.use_compositing = True
        scene.render.compositor_device = "GPU"
        scene.render.compositor_denoise_device = "GPU"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine", choices=("CYCLES", "BLENDER_EEVEE"), default="CYCLES")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--preview-time", type=float, default=12.0)
    parser.add_argument("--render-animation", action="store_true", help="Render all replay frames after the preview")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    args.asset, args.run, args.output = args.asset.resolve(), args.run.resolve(), args.output.resolve()
    if (args.output / "kitchen-pour-8ss-10iter.blend").exists():
        raise FileExistsError("Choose a new output directory to preserve the existing Blender project")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "frames").mkdir(exist_ok=True)
    document = json.loads((args.asset / "scene.json").read_text())
    result = json.loads((args.run / "result.json").read_text())
    trace = np.load(args.run / "trace.npz")
    poses, times = trace["poses"], trace["time_s"]
    if result["substeps"] != 8 or result["iterations"] != 10:
        raise ValueError("This requested Blender replay must use the 8-substep, 10-iteration recording")
    if not np.allclose(np.diff(times), 1 / 60) or poses.shape[1] != 61:
        raise ValueError("Expected a 60 Hz trace with the bin and 60 objects")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.world = bpy.data.worlds.new("Studio world")
    _configure_render(scene, args)
    scene.frame_start, scene.frame_end, scene.frame_step = 1, (len(poses) + 1) // 2, 1
    scene.unit_settings.system = "METRIC"
    scene["simulation"] = "Recorded FPGS poses only; no Blender physics"
    scene["physics_substeps"] = result["substeps"]
    scene["physics_iterations"] = result["iterations"]
    scene["physics_device"] = result["gpu"]
    scene["trace_sha256"] = hashlib.sha256((args.run / "trace.npz").read_bytes()).hexdigest()
    meshes, materials = {}, {}
    for kind, item in document["bank"].items():
        data = np.load(args.asset / f"{item['mesh']}.npz")
        mesh = bpy.data.meshes.new(kind)
        mesh.from_pydata(data["vertices"].tolist(), [], data["faces"].reshape(-1, 3).tolist())
        mesh.update()
        for polygon in mesh.polygons:
            polygon.use_smooth = True
        # Smooth curved walls without blending normals across rims, flat
        # bottoms, or other hard profile transitions. Geometry is unchanged.
        mesh.set_sharp_from_angle(angle=math.radians(30))
        meshes[kind] = mesh
        materials[kind] = _material(kind, item["color"], roughness=0.3 if "pin" in kind or "stick" in kind else 0.24)
        mesh.materials.append(materials[kind])
    bin_root = bpy.data.objects.new("00 Bin | recorded FPGS body", None)
    bpy.context.collection.objects.link(bin_root)
    bin_root.empty_display_size = 0.08
    _glass_bin(bin_root)
    animated = [bin_root]
    for index, item in enumerate(document["bodies"], 1):
        obj = bpy.data.objects.new(f"{index:02d} {item['kind']}", meshes[item["kind"]])
        bpy.context.collection.objects.link(obj)
        obj["fpgs_body_index"] = index
        obj["release_time_s"] = item["release_s"]
        animated.append(obj)
    for index, obj in enumerate(animated):
        _key_poses(obj, poses[:, index])
    _box("Table", (0.2, 0, -0.02), (1.1, 0.8, 0.02), _material("Slate tabletop", (0.09, 0.13, 0.17), roughness=0.6))
    _box("Studio floor", (0, 0, -0.12), (100, 100, 0.02), _material("Studio floor", (0.7, 0.75, 0.81), roughness=0.8))
    _light("Large soft key", (-2, -3, 4), (0, 0, 0.3), 650, 3)
    _light("Soft fill", (3, -1, 2.5), (0.1, 0, 0.3), 250, 3)
    _light("Rim light", (0.5, 2, 3), (0, 0, 0.3), 450, 2)
    camera_data = bpy.data.cameras.new("Camera")
    camera = bpy.data.objects.new("Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera
    camera_data.lens = 46
    camera_data.clip_start, camera_data.clip_end = 0.02, 100
    direction = Vector((-0.60, -0.72, 0.45)).normalized()
    for sample, time_s in enumerate(times):
        frame = 1 + sample / 2
        u = float(np.clip((time_s - 13) / 4, 0, 1))
        u = u * u * (3 - 2 * u)
        target = Vector((0.04 + 0.16 * u, 0, 0.42 - 0.09 * u))
        camera.location = target + (2.65 + 0.60 * u) * direction
        camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
        camera.keyframe_insert(data_path="location", frame=frame)
        camera.keyframe_insert(data_path="rotation_euler", frame=frame)
    # Confirm animation at the recorded frames before saving the portable file.
    position_error, rotation_error = 0.0, 0.0
    for index in (0, 300, 301, 720, 721, 1020, len(poses) - 1):
        scene.frame_set(index // 2 + 1, subframe=(index % 2) / 2)
        for body, obj in enumerate(animated):
            position_error = max(position_error, float(np.max(abs(np.asarray(obj.location) - poses[index, body, :3]))))
            q = np.asarray(obj.rotation_quaternion)
            expected = poses[index, body, [6, 3, 4, 5]]
            rotation_error = max(rotation_error, float(min(np.max(abs(q - expected)), np.max(abs(q + expected)))))
    if position_error > 1e-5 or rotation_error > 1e-5 or scene.rigidbody_world is not None:
        raise AssertionError("Recorded-pose replay validation failed")
    report = {
        "blender_version": bpy.app.version_string,
        "engine": args.engine,
        "samples": args.samples,
        "gpu_denoising": args.engine == "CYCLES",
        "resolution": [args.width, args.height],
        "trace_sha256": scene["trace_sha256"],
        "substeps": result["substeps"],
        "iterations": result["iterations"],
        "pose_samples": len(poses),
        "bodies": len(animated),
        "recording_fps": 60,
        "timeline_fps": 30,
        "render_frame_step": 1,
        "video_fps": 30,
        "max_checked_position_error_m": position_error,
        "max_checked_quaternion_component_error": rotation_error,
        "blender_physics_enabled": False,
        "smooth_normals_max_angle_degrees": 30,
        "bin_material": {"shader": "Glass BSDF", "ior": 1.5, "roughness": 0.015, "wall_thickness_m": 0.012},
        "sharp_edges_per_mesh": {
            kind: sum(edge.use_edge_sharp for edge in mesh.edges) for kind, mesh in meshes.items()
        },
    }
    text = bpy.data.texts.new("REPLAY_INFO.json")
    text.write(json.dumps(report, indent=2))
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                area.spaces.active.region_3d.view_perspective = "CAMERA"
                area.spaces.active.shading.type = "MATERIAL"
    scene.frame_set(round(args.preview_time * 30) + 1)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output / "kitchen-pour-8ss-10iter.blend"), compress=True)
    scene.render.filepath = str(args.output / "preview.png")
    started = time.perf_counter()
    bpy.ops.render.render(write_still=True)
    report["preview_render_wall_s"] = time.perf_counter() - started
    if args.render_animation:
        scene.render.filepath = str(args.output / "frames" / "frame_")
        started = time.perf_counter()
        bpy.ops.render.render(animation=True)
        report["animation_render_wall_s"] = time.perf_counter() - started
    (args.output / "render-info.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
