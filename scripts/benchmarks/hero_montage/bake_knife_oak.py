# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Run in Blender with -- SOURCE_PROC_GEN_CHECKOUT."""

import argparse
import json
import sys
from pathlib import Path

import bpy
import numpy as np

parser = argparse.ArgumentParser(description="Bake original proc-gen oak onto the existing holder visuals")
parser.add_argument("source", type=Path)
args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
sys.path.insert(0, str(args.source / "scripts"))
from render_detailed import material  # noqa: E402 -- selected catalog checkout

out = Path(__file__).parent / "assets/knife_oak"
out.mkdir(exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
s = bpy.context.scene
s.render.engine = "CYCLES"
s.cycles.device = "CPU"
s.cycles.samples = 1
s.render.threads_mode = "FIXED"
s.render.threads = 4
for name in ["knife_block_task", "knife_source_task"]:
    src = Path(__file__).parent / "assets/tasks"
    meta = json.loads((src / (name + ".json")).read_text())
    arrays = np.load(src / (name + ".npz"))
    body = next(b for b in meta["bodies"] if any(v["material"] == "oak" for v in b["visuals"]))
    v = body["visuals"][0]
    key = v["prefix"]
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(arrays[key + "_v"].tolist(), [], arrays[key + "_f"].tolist())
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    mat = material("oak", v["color"] + [1])
    # Fine longitudinal grain and restrained neutral ash tones suit the
    # montage better than the catalog's broad, high-contrast oak fields.
    for shader in mat.node_tree.nodes:
        if shader.bl_idname == "ShaderNodeVectorMath" and shader.operation == "MULTIPLY":
            shader.inputs[1].default_value = (190, 190, 2.5)
        elif shader.bl_idname == "ShaderNodeValToRGB":
            shader.color_ramp.elements[0].color = (0.37, 0.295, 0.215, 1)
            shader.color_ramp.elements[1].color = (0.43, 0.35, 0.26, 1)
    mesh.materials.append(mat)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(island_margin=0.025)
    bpy.ops.object.mode_set(mode="OBJECT")
    im = bpy.data.images.new(name, width=1024, height=1024)
    im.colorspace_settings.name = "sRGB"
    node = mat.node_tree.nodes.new("ShaderNodeTexImage")
    node.image = im
    mat.node_tree.nodes.active = node
    bpy.ops.object.bake(type="DIFFUSE", pass_filter={"COLOR"}, margin=12)
    im.filepath_raw = str(out / (name + ".png"))
    im.file_format = "PNG"
    im.save()
    uv = np.array([x.uv[:] for x in mesh.uv_layers.active.data])
    np.savez(out / (name + ".npz"), vertices=arrays[key + "_v"], faces=arrays[key + "_f"], uv=uv.reshape(-1, 3, 2))
    bpy.data.objects.remove(obj, do_unlink=True)
print("BAKED BOTH OAK HOLDERS", flush=True)
