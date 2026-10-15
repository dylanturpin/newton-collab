# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Apply the shared proc-gen dressing to saved replays without resimulating."""

import json
import math
import shutil
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_cycles_replays import linear
from mathutils import Quaternion, Vector

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets" / "furnishings"


def restyle(kind, source, data, output, time):
    output.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.open_mainfile(filepath=str(source / "scene.blend"))
    scene = bpy.context.scene
    camera = json.loads((source / "camera.json").read_text())
    scene_meta = json.loads((data / "scene.json").read_text())
    offset = np.array(scene_meta["worlds"][0]["display_offset"])
    layouts = json.loads((ROOT / "assets/task_decor.json").read_text())
    table = json.loads((ROOT / "assets/task_tables.json").read_text())[kind]
    manifest = json.loads((ASSETS / "manifest.json").read_text())["assets"]
    for obj in list(bpy.data.objects):
        if "/furnishing/" in obj.name or obj.name.startswith("task-decor/"):
            bpy.data.objects.remove(obj, do_unlink=True)
    props = [dict(table, position=[0, 0, -0.001], angle=0, scale=1)] + layouts[kind]
    for index, prop in enumerate(props):
        name = prop["asset"]
        with np.load(ASSETS / f"{name}.npz") as arrays:
            for region_index, region in enumerate(manifest[name]["regions"]):
                vertices, faces = arrays[f"v{region_index}"], arrays[f"f{region_index}"].reshape(-1, 3)
                mesh = bpy.data.meshes.new(f"task-decor/{name}/{index}/{region_index}")
                mesh.from_pydata(vertices.tolist(), [], faces.tolist())
                mesh.update()
                mesh.polygons.foreach_set("use_smooth", np.ones(len(faces), dtype=bool))
                mesh.normals_split_custom_set_from_vertices(arrays[f"n{region_index}"].tolist())
                uv = mesh.uv_layers.new(name="UVMap")
                uv.data.foreach_set("uv", arrays[f"uv{region_index}"][faces.ravel()].ravel())
                mat = bpy.data.materials.new(f"decor-{name}-{region_index}")
                mat.use_nodes = True
                shader = mat.node_tree.nodes.get("Principled BSDF")
                material = region["material"]
                wood = material in ("oak", "ash", "walnut")
                rgb = prop.get("top_color", region["color"]) if material == "plastic" else region["color"]
                shader.inputs["Base Color"].default_value = (*linear(rgb), 1)
                shader.inputs["Roughness"].default_value = 0.46 if wood else 0.32
                shader.inputs["Metallic"].default_value = 0.75 if material in ("chrome", "aluminum", "brass") else 0
                if wood:
                    texture = mat.node_tree.nodes.new("ShaderNodeTexImage")
                    texture.image = bpy.data.images.load(str(ASSETS / f"{material}.png"), check_existing=True)
                    mat.node_tree.links.new(texture.outputs["Color"], shader.inputs["Base Color"])
                mesh.materials.append(mat)
                obj = bpy.data.objects.new(mesh.name, mesh)
                scene.collection.objects.link(obj)
                obj.location = offset + np.array(prop["position"])
                obj.rotation_euler.z = prop["angle"]
                obj.scale = (prop["scale"],) * 3
    if kind == "bird_automaton":
        target = np.add(offset, [0.12, 0.12, 0.40])
        camera.update(target=target.tolist(), position=(np.add(target, [1.35, -1.1, 0.70])).tolist(), fov=38)
    elif kind == "pelican":
        target = np.add(offset, [0.02, 0.02, 0.40])
        camera.update(target=target.tolist(), position=(np.add(target, [0.80, -1.90, 0.80])).tolist(), fov=39)
    scene.camera.location = camera["position"]
    scene.camera.rotation_euler = (Vector(camera["target"]) - scene.camera.location).to_track_quat("-Z", "Y").to_euler()
    scene.camera.data.lens = 36 / (2 * math.tan(math.radians(camera["fov"] / 2)))
    meta = json.loads((source / "metadata.json").read_text())
    with np.load(source / "motion.npz") as motion:
        sample = min(round(time * meta["recording_fps"]), len(motion["positions"]) - 1)
        for obj in bpy.data.objects:
            if "recorded_body" in obj:
                body = obj["recorded_body"]
                obj.location = motion["positions"][sample, body, :3]
                x, y, z, w = motion["rotations"][sample, body]
                obj.rotation_mode = "QUATERNION"
                obj.rotation_quaternion = Quaternion((float(w), float(x), float(y), float(z)))
    shutil.copy2(source / "motion.npz", output / "motion.npz")
    shutil.copy2(source / "metadata.json", output / "metadata.json")
    (output / "camera.json").write_text(json.dumps(camera, indent=2))
    (output / "table.json").write_text(json.dumps(table, indent=2))
    (output / "dressing.json").write_text(json.dumps(layouts[kind], indent=2))
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(output / "scene.blend"), compress=True)
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "METAL"
    prefs.get_devices()
    for device in prefs.devices:
        device.use = device.type == "METAL"
    scene.cycles.device = "GPU"
    scene.cycles.samples = 32
    scene.cycles.use_denoising = True
    scene.render.resolution_x, scene.render.resolution_y = 1280, 720
    scene.render.resolution_percentage = 100
    scene.render.filepath = str(output / "preview.png")
    bpy.ops.render.render(write_still=True)


def main():
    config = json.loads(Path(sys.argv[sys.argv.index("--") + 1]).read_text())
    for record in config:
        restyle(record["kind"], Path(record["source"]), Path(record["data"]), Path(record["output"]), record["time"])


if __name__ == "__main__":
    main()
