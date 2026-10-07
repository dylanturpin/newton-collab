# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Prepare portable kitchen-pour geometry from the user's numeric authoring source."""

import argparse
import hashlib
import importlib.util
import json
import math
import random
import struct
from pathlib import Path

import numpy as np


def _tube(spans):
    import trimesh

    points, radii, tangents = [], [], []
    for span_index, span in enumerate(spans):
        controls = np.asarray(span["control"])
        for t in np.linspace(0, 1, 9)[:-1] if span_index < len(spans) - 1 else np.linspace(0, 1, 9):
            points.append(
                (1 - t) ** 3 * controls[0]
                + 3 * (1 - t) ** 2 * t * controls[1]
                + 3 * (1 - t) * t**2 * controls[2]
                + t**3 * controls[3]
            )
            tangents.append(
                3 * (1 - t) ** 2 * (controls[1] - controls[0])
                + 6 * (1 - t) * t * (controls[2] - controls[1])
                + 3 * t**2 * (controls[3] - controls[2])
            )
            radii.append((1 - t) * span["radii"][0] + t * span["radii"][1])
    vertices = []
    sides = 16
    for point, tangent, radius in zip(points, tangents, radii, strict=True):
        direction = tangent / np.linalg.norm(tangent)
        u = np.cross(direction, [0, 1, 0])
        u /= np.linalg.norm(u)
        v = np.cross(direction, u)
        vertices.extend(point + radius * (np.cos(a) * u + np.sin(a) * v) for a in np.arange(sides) * 2 * np.pi / sides)
    faces = []
    for ring in range(len(points) - 1):
        for j in range(sides):
            a, b = ring * sides + j, ring * sides + (j + 1) % sides
            faces.extend([[a, b, b + sides], [a, b + sides, a + sides]])
    vertices.extend([points[0], points[-1]])
    for j in range(sides):
        faces.append([len(vertices) - 2, (j + 1) % sides, j])
        a = (len(points) - 1) * sides
        faces.append([len(vertices) - 1, a + j, a + (j + 1) % sides])
    mesh = trimesh.Trimesh(vertices, faces, process=True)
    mesh.fix_normals()
    return mesh


def _mesh(parts):
    import trimesh

    meshes = []
    for part in parts:
        op = part["op"]
        if op == "lathe_sectors":
            profile = np.asarray(part["profile"])
            mesh = trimesh.creation.revolve(np.vstack([profile, profile[0]]), sections=96)
        elif op == "curve_sweep":
            mesh = _tube(part["spans"])
        elif op == "sdf_capsule":
            a, b = np.asarray(part["a"]), np.asarray(part["b"])
            mesh = trimesh.creation.capsule(height=float(np.linalg.norm(b - a)), radius=part["radius"], count=[16, 32])
            mesh.apply_translation((a + b) / 2)
        elif op == "ellipsoid":
            mesh = trimesh.creation.icosphere(subdivisions=3, radius=part["radius"])
        elif op == "disc":
            r, h, e = part["radius"], part["height"] / 2, part["edge_radius"]
            profile = [[0, -h], [r - e, -h]]
            profile += [
                [r - e + e * math.cos(a), -h + e + e * math.sin(a)] for a in np.linspace(-math.pi / 2, 0, 7)[1:]
            ]
            profile += [[r - e + e * math.cos(a), h - e + e * math.sin(a)] for a in np.linspace(0, math.pi / 2, 7)]
            profile += [[0, h], [0, -h]]
            mesh = trimesh.creation.revolve(profile, sections=48)
        else:
            raise ValueError(f"Unsupported authoring operation: {op}")
        mesh.fix_normals()
        if not mesh.is_watertight or mesh.volume <= 0:
            raise ValueError(f"Invalid closed mesh: {op}")
        meshes.append(mesh)
    return trimesh.util.concatenate(meshes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kitchen-source", required=True, type=Path)
    parser.add_argument("--cmg-pack", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", default=1, type=int)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("kitchen_authoring", args.kitchen_source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = module.kitchen()
    by_name = {item["id"]: item for item in source["objects"]}
    kinds = [
        "cup",
        "plate",
        "bowl/0",
        "mug/0",
        "jar",
        "vase",
        "chopstick",
        "rolling-pin",
        "sweet/sphere",
        "sweet/disc",
        "cup",
        "plate",
    ]
    colors = {
        "plate": [0.16, 0.66, 0.48],
        "bowl/0": [0.18, 0.65, 0.58],
        "jar": [0.88, 0.61, 0.27],
        "vase": [0.27, 0.49, 0.72],
        "rolling-pin": [0.82, 0.64, 0.39],
        "chopstick": [0.86, 0.64, 0.20],
        "sweet/sphere": [0.93, 0.65, 0.19],
        "sweet/disc": [0.92, 0.72, 0.23],
    }
    bank = {}
    for name in dict.fromkeys(kinds):
        authored = by_name[name]
        mesh = _mesh(authored["parts"])
        mesh.density = authored["density"]
        mass, com, inertia = mesh.mass, mesh.center_mass, mesh.moment_inertia
        mass_source = "closed source mesh, authored density"
        collider = args.cmg_pack / "colliders" / (name.replace("/", "_") + ".cmgc")
        if collider.exists():
            data = collider.read_bytes()
            offset, length = struct.unpack_from("<QQ", data, 16)
            properties = json.loads(data[offset : offset + length])["mass"]
            mass, com, inertia = (
                properties["mass_kg"],
                properties["centre_of_mass"],
                np.asarray(properties["inertia_about_com"]).reshape(3, 3),
            )
            mass_source = "CMG asset pack supplied mass, COM and full inertia"
        if name == "sweet/sphere":
            radius = authored["parts"][0]["radius"]
            mass = 4 * np.pi * radius**3 * authored["density"] / 3
            com, inertia = np.zeros(3), np.eye(3) * (0.4 * mass * radius**2)
            mass_source = "analytic sphere, authored radius and density"
        key = name.replace("/", "_")
        mesh.export(args.output / f"{key}.obj")
        np.savez_compressed(
            args.output / f"{key}.npz",
            vertices=np.asarray(mesh.vertices, np.float32),
            faces=np.asarray(mesh.faces, np.int32),
        )
        bank[name] = {
            "mesh": key,
            "parts": authored["parts"],
            "mass": float(mass),
            "com": np.asarray(com).tolist(),
            "inertia": np.asarray(inertia).tolist(),
            "radius": float(np.linalg.norm(np.maximum(abs(mesh.bounds[0] - com), abs(mesh.bounds[1] - com)))),
            "color": colors.get(name, [0.29, 0.53, 0.77]),
            "mass_source": mass_source,
        }
    rng = random.Random(args.seed)
    order = kinds * 5
    rng.shuffle(order)
    bodies = []
    for index, name in enumerate(order):
        q = np.asarray([rng.gauss(0, 1) for _ in range(4)])
        q /= np.linalg.norm(q)
        radius = bank[name]["radius"]
        bodies.append(
            {
                "kind": name,
                "release_s": index * 0.2,
                "position": [
                    rng.uniform(-0.02, 0.02),
                    rng.uniform(-0.02, 0.02),
                    0.85 + radius + np.linalg.norm(bank[name]["com"]),
                ],
                "rotation_xyzw": q.tolist(),
            }
        )
    document = {
        "name": "kitchen pour / 60 objects",
        "units": "m, kg, s",
        "seed": args.seed,
        "bank": bank,
        "bodies": bodies,
        "source_sha256": hashlib.sha256(args.kitchen_source.read_bytes()).hexdigest(),
        "provenance": {
            "kitchen_source": str(args.kitchen_source),
            "cmg_pack": str(args.cmg_pack),
            "reference": "Screen Recording 2026-09-28 at 1.52.41 PM.MOV",
            "reconstructed": "seeded original authoring mix; bin motion reconstructed from video; no source trajectories replayed",
        },
    }
    (args.output / "scene.json").write_text(json.dumps(document, indent=2) + "\n")
    print(json.dumps({"bodies": len(bodies), "kinds": len(bank), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
