# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Apply baked proc-gen oak to holder visuals without changing replay poses."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).parent / "assets/knife_oak"
TASKS = Path(__file__).parent / "assets/tasks"


def apply(folder):
    meta = json.loads((folder / "scene.json").read_text())
    vertices = np.memmap(folder / "vertices.bin", dtype="<f4", mode="r").reshape(-1, 16)
    indices = np.memmap(folder / "indices.bin", dtype="<u4", mode="r")
    changes = []
    for mesh in meta["meshes"]:
        if mesh["name"].endswith("/oak-finish"):
            name = mesh["name"].split("/")[-2]
            shutil.copy2(ASSETS / (name + ".png"), folder / "textures" / (name + "-oak.png"))
            continue
        name = mesh["name"].split("/")[-1]
        if name not in ("knife_block_task", "knife_source_task"):
            continue
        used = indices[mesh["first_index"] : mesh["first_index"] + mesh["index_count"]]
        material = meta["materials"][int(vertices[used[0], 14]) - 1]
        if not np.allclose(material["color"], [0.53, 0.32, 0.16], atol=1e-5):
            continue
        baked = np.load(ASSETS / (name + ".npz"))
        source = np.load(TASKS / (name + ".npz"))
        faces = baked["faces"].ravel()
        xyz = baked["vertices"][faces]
        np.testing.assert_allclose(xyz.min(0), vertices[used, :3].min(0), atol=1e-5)
        np.testing.assert_allclose(xyz.max(0), vertices[used, :3].max(0), atol=1e-5)
        attrs = np.zeros((len(faces), 16), dtype="<f4")
        attrs[:, :3] = xyz
        attrs[:, 3].view("<u4")[:] = mesh["body"]
        attrs[:, 4:7] = source["b0_v0_n"][faces]
        attrs[:, 7] = 0.38
        attrs[:, 8:11] = 1
        attrs[:, 12:14] = baked["uv"].reshape(-1, 2)
        attrs[:, 13] = 1 - attrs[:, 13]
        texture = "textures/" + name + "-oak.png"
        (folder / "textures").mkdir(exist_ok=True)
        shutil.copy2(ASSETS / (name + ".png"), folder / texture)
        meta["materials"].append({"color": [1, 1, 1], "roughness": 0.38, "metallic": 0, "texture": texture})
        attrs[:, 14] = len(meta["materials"])
        changes.append((mesh, attrs))
    del vertices, indices
    if not changes:
        print(folder, "already textured or no holders")
        return
    for filename in ("vertices.bin", "indices.bin"):
        path = folder / filename
        if path.is_symlink():
            target = path.resolve()
            path.unlink()
            shutil.copy2(target, path)
    nv, ni = meta["vertex_count"], meta["index_count"]
    with (folder / "vertices.bin").open("ab") as vf, (folder / "indices.bin").open("ab") as inf:
        for mesh, attrs in changes:
            attrs.tofile(vf)
            (np.arange(len(attrs), dtype="<u4") + nv).tofile(inf)
            mesh.update(name=mesh["name"] + "/oak-finish", first_index=ni, index_count=len(attrs))
            nv += len(attrs)
            ni += len(attrs)
    meta.update(
        vertex_count=nv, index_count=ni, knife_holder_finish="Baked original proc-gen oak; colored handles preserved"
    )
    (folder / "scene.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(folder, len(changes), "holders textured")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folders", type=Path, nargs="+")
    for folder in parser.parse_args().folders:
        apply(folder)
