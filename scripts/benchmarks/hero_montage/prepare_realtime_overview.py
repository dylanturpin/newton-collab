# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Keep the opening tile detailed and use validated overview meshes elsewhere."""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


def prepare(full, draft, out, *, opening_tile=78):
    out.mkdir(parents=True, exist_ok=True)
    m = json.loads((full / "scene.json").read_text())
    d = json.loads((draft / "scene.json").read_text())
    mats = {json.dumps(s, sort_keys=True): i + 1 for i, s in enumerate(m["materials"])}
    nv = ni = 0
    meshes = []
    for name in ["positions.bin", "rotations.bin"]:
        shutil.copy2(full / name, out / name)
    shutil.copytree(full / "textures", out / "textures", dirs_exist_ok=True)
    with (out / "vertices.bin").open("wb") as vf, (out / "indices.bin").open("wb") as inf:
        for root, meta, keep in [(full, m, lambda w: w == opening_tile), (draft, d, lambda w: w != opening_tile)]:
            v = np.memmap(root / "vertices.bin", "<f4", "r").reshape(-1, 16)
            idx = np.memmap(root / "indices.bin", "<u4", "r")
            for mesh in meta["meshes"]:
                if not keep(mesh["world"]):
                    continue
                used, faces = np.unique(
                    idx[mesh["first_index"] : mesh["first_index"] + mesh["index_count"]], return_inverse=True
                )
                a = v[used].copy()
                mat = meta["materials"][int(a[0, 14]) - 1]
                key = json.dumps(mat, sort_keys=True)
                if key not in mats:
                    m["materials"].append(mat)
                    mats[key] = len(m["materials"])
                a[:, 14] = mats[key]
                a.tofile(vf)
                (faces.astype("<u4") + nv).tofile(inf)
                meshes.append({**mesh, "first_index": ni, "index_count": len(faces)})
                nv += len(a)
                ni += len(faces)
    m.update(
        meshes=meshes,
        vertex_count=nv,
        index_count=ni,
        overview_geometry_lod=True,
        full_detail_tiles=[opening_tile],
        visible_tile_count=96,
    )
    (out / "scene.json").write_text(json.dumps(m, indent=2) + "\n")
    print(nv, "vertices", ni // 3, "triangles")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("full", "draft", "output"):
        parser.add_argument(name, type=Path)
    args = parser.parse_args()
    prepare(args.full, args.draft, args.output)
