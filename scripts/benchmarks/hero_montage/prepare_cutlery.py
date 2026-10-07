# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Author watertight cutlery surfaces and decompose their union with CoACD."""

import argparse
import json
from pathlib import Path

import numpy as np


def sweep(stations, sides=32, samples=100):
    """Loft rounded elliptical sections along a shaped handle or tine."""
    import trimesh
    from scipy.interpolate import PchipInterpolator

    stations = np.asarray(stations)
    x = np.linspace(stations[0, 0], stations[-1, 0], samples)
    y, z, width, depth = PchipInterpolator(stations[:, 0], stations[:, 1:], axis=0)(x).T
    angles = np.arange(sides) * 2 * np.pi / sides
    vertices = np.stack(
        [
            np.broadcast_to(x[:, None], (samples, sides)),
            y[:, None] + width[:, None] * np.cos(angles),
            z[:, None] + depth[:, None] * np.sin(angles),
        ],
        axis=-1,
    ).reshape(-1, 3)
    faces = []
    for ring in range(samples - 1):
        for j in range(sides):
            a, b = ring * sides + j, ring * sides + (j + 1) % sides
            faces.extend([[a, b, b + sides], [a, b + sides, a + sides]])
    vertices = np.vstack([vertices, [x[0], y[0], z[0]], [x[-1], y[-1], z[-1]]])
    for j in range(sides):
        faces.append([len(vertices) - 2, (j + 1) % sides, j])
        a = (samples - 1) * sides
        faces.append([len(vertices) - 1, a + j, a + (j + 1) % sides])
    mesh = trimesh.Trimesh(vertices, faces, process=True)
    mesh.fix_normals()
    return mesh


def bowl():
    """A thin oval spoon shell, with a concave top and curved underside."""
    import trimesh

    n, rings = 96, 16
    vertices, faces = [], []
    for lower in (False, True):
        offset = len(vertices)
        vertices.append([0.109, 0, -0.004 - (0.0025 if lower else 0)])
        for r in np.linspace(1 / rings, 1, rings):
            for a in np.arange(n) * 2 * np.pi / n:
                vertices.append(
                    [
                        0.109 + 0.046 * r * np.cos(a),
                        0.030 * r * np.sin(a),
                        -0.004 + 0.016 * r * r - (0.0025 if lower else 0),
                    ]
                )
        for j in range(n):
            faces.append([offset, offset + 1 + j, offset + 1 + (j + 1) % n])
        for ring in range(rings - 1):
            for j in range(n):
                a, b = offset + 1 + ring * n + j, offset + 1 + ring * n + (j + 1) % n
                faces.extend([[a, a + n, b + n], [a, b + n, b]])
    stride = 1 + rings * n
    for j in range(n):
        a, b = 1 + (rings - 1) * n + j, 1 + (rings - 1) * n + (j + 1) % n
        faces.extend([[a, b, b + stride], [a, b + stride, a + stride]])
    mesh = trimesh.Trimesh(vertices, faces, process=True)
    mesh.fix_normals()
    return mesh


def make(style):
    import trimesh

    handle = sweep(
        [
            [-0.100, 0, 0, 0.0008, 0.0008],
            [-0.096, 0, 0, 0.007, 0.003],
            [-0.080, 0, 0.0015, 0.0105, 0.0038],
            [-0.030, 0, 0.0015, 0.0105, 0.0038],
            [0.025, 0, 0.0015, 0.0105, 0.0038],
            [0.045, 0, 0.004, 0.006, 0.0028],
            [0.065, 0, 0.007, 0.008, 0.0024],
            [0.080, 0, 0.007, 0.006, 0.0020],
        ]
    )
    if style == "spoon":
        head = bowl()
    elif style == "fork":
        pieces = [
            sweep(
                [
                    [0.053, 0, 0.005, 0.005, 0.002],
                    [0.071, 0, 0.006, 0.015, 0.002],
                    [0.089, 0, 0.006, 0.022, 0.002],
                    [0.109, 0, 0.005, 0.022, 0.0018],
                ],
                samples=35,
            )
        ]
        for j in range(4):
            y = (j - 1.5) * 0.013
            pieces.append(
                sweep(
                    [
                        [0.101, y, 0.005, 0.0037, 0.0018],
                        [0.122, y, 0.006, 0.0032, 0.0016],
                        [0.148, y, 0.012, 0.0024, 0.0013],
                        [0.163 - abs(j - 1.5) * 0.001, y, 0.014, 0.0007, 0.0007],
                    ],
                    samples=32,
                    sides=20,
                )
            )
        head = trimesh.boolean.union(pieces, engine="manifold")
    else:
        head = sweep(
            [
                [0.055, -0.003, 0.005, 0.007, 0.0021],
                [0.075, -0.005, 0.005, 0.013, 0.0019],
                [0.108, -0.006, 0.005, 0.015, 0.0017],
                [0.145, -0.004, 0.005, 0.013, 0.0014],
                [0.165, 0.002, 0.005, 0.006, 0.0010],
                [0.171, 0.003, 0.005, 0.0007, 0.0006],
            ],
            samples=90,
        )
    united = trimesh.boolean.union([handle, head], engine="manifold")
    assert united.is_watertight and united.volume > 0
    assert len(united.split()) == 1, f"Disconnected {style}"
    return handle, head, united


def main():
    import coacd

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "assets/cutlery")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    coacd.set_log_level("warn")
    report = {}
    for style in ("spoon", "fork", "knife"):
        handle, head, mesh = make(style)
        mesh.export(args.output / f"{style}.obj")
        payload = {}
        for name, part in (("handle", handle), ("head", head)):
            payload[f"{name}_vertices"] = np.asarray(part.vertices, dtype=np.float32)
            payload[f"{name}_faces"] = np.asarray(part.faces, dtype=np.int32)
            payload[f"{name}_normals"] = np.asarray(part.vertex_normals, dtype=np.float32)
        # Work in millimetres so thin shells are well conditioned for decomposition.
        hulls = coacd.run_coacd(
            coacd.Mesh(np.asarray(mesh.vertices) * 1000, np.asarray(mesh.faces)),
            threshold=0.025,
            max_convex_hull=32,
            preprocess_mode="off",
            resolution=1500,
            mcts_iterations=100,
            mcts_max_depth=4,
            mcts_nodes=20,
            merge=True,
            seed=17,
        )
        for i, (vertices, faces) in enumerate(hulls):
            payload[f"hull_{i}_vertices"] = np.asarray(vertices / 1000, dtype=np.float32)
            payload[f"hull_{i}_faces"] = np.asarray(faces, dtype=np.int32)
        mesh.density = 1800
        payload.update(mass=mesh.mass, com=mesh.center_mass, inertia=mesh.moment_inertia)
        np.savez_compressed(args.output / f"{style}.npz", **payload)
        report[style] = {
            "watertight": bool(mesh.is_watertight),
            "triangles": len(mesh.faces),
            "convex_pieces": len(hulls),
            "mass_kg": float(mesh.mass),
            "bounds_m": mesh.bounds.tolist(),
        }
        print(style, report[style], flush=True)
    (args.output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
