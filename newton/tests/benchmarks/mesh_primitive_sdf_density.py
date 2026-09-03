# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Mesh vs analytic-primitive collision: legacy per-triangle route against the SDF route.

For each primitive family and each mesh density a grid-tessellated box mesh rests on the
primitive with a fixed penetration. The collision pipeline runs once with
``mesh_primitive_sdf=False`` (BVH overlap + one GJK/MPR test per overlapping triangle) and
once with ``mesh_primitive_sdf=True`` (edge and face sampling of the primitive's closed-form
signed distance). The report lists per case:

* triangles in the mesh,
* triangle pairs the legacy midphase produced (its work),
* mesh-vs-SDF pairs the SDF route produced (always one per mesh/primitive pair),
* exported contacts after reduction,
* median wall time of ``CollisionPipeline.collide`` (CUDA-synchronised).

Timing is reported, never asserted: absolute numbers depend on the GPU and driver.

Example::

    python newton/tests/benchmarks/mesh_primitive_sdf_density.py --worlds 64 --densities 4,16,64
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import warp as wp

_NEWTON_REPO = Path(__file__).resolve().parents[3]
if str(_NEWTON_REPO) not in sys.path:
    sys.path.insert(0, str(_NEWTON_REPO))

import newton  # noqa: E402

PRIMITIVES = ("box", "sphere", "capsule", "cylinder", "cone", "ellipsoid")
MESH_HALF = 0.25
PENETRATION = 0.01


@dataclass
class CaseResult:
    primitive: str
    density: int
    triangles: int
    worlds: int
    route: str
    triangle_pairs: int
    mesh_sdf_pairs: int
    contacts: int
    median_ms: float
    p95_ms: float


def grid_box_mesh(half: float, n: int) -> newton.Mesh:
    """Closed box of half-extent ``half`` whose six faces are ``n x n`` quads (12 n^2 triangles)."""
    verts: list[tuple[float, float, float]] = []
    tris: list[int] = []
    axes = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
    for u_axis, v_axis, w_axis in axes:
        for sign in (-1.0, 1.0):
            base = len(verts)
            for i in range(n + 1):
                for j in range(n + 1):
                    p = [0.0, 0.0, 0.0]
                    p[u_axis] = -half + 2.0 * half * i / n
                    p[v_axis] = -half + 2.0 * half * j / n
                    p[w_axis] = sign * half
                    verts.append((p[0], p[1], p[2]))
            for i in range(n):
                for j in range(n):
                    a = base + i * (n + 1) + j
                    b = a + 1
                    c = a + (n + 1)
                    d = c + 1
                    # Orient the face outward: flip winding for the negative side.
                    if sign > 0.0:
                        tris += [a, c, d, a, d, b]
                    else:
                        tris += [a, d, c, a, b, d]
    return newton.Mesh(np.asarray(verts, dtype=np.float32), np.asarray(tris, dtype=np.int32), compute_inertia=False)


def add_primitive(builder: newton.ModelBuilder, body: int, name: str, top_z: float, x_off: float) -> int:
    """Add a primitive whose topmost point sits at ``(x_off, 0, top_z)``."""
    q = wp.quat_identity()
    if name == "box":
        return builder.add_shape_box(
            body, xform=wp.transform(wp.vec3(x_off, 0.0, top_z - 0.25), q), hx=2.0, hy=2.0, hz=0.25
        )
    if name == "sphere":
        return builder.add_shape_sphere(body, xform=wp.transform(wp.vec3(x_off, 0.0, top_z - 0.5), q), radius=0.5)
    if name == "capsule":
        return builder.add_shape_capsule(
            body, xform=wp.transform(wp.vec3(x_off, 0.0, top_z - 0.75), q), radius=0.25, half_height=0.5
        )
    if name == "cylinder":
        return builder.add_shape_cylinder(
            body, xform=wp.transform(wp.vec3(x_off, 0.0, top_z - 0.4), q), radius=0.6, half_height=0.4
        )
    if name == "cone":
        return builder.add_shape_cone(
            body, xform=wp.transform(wp.vec3(x_off, 0.0, top_z - 0.4), q), radius=0.5, half_height=0.4
        )
    if name == "ellipsoid":
        return builder.add_shape_ellipsoid(
            body, xform=wp.transform(wp.vec3(x_off, 0.0, top_z - 0.2), q), rx=0.5, ry=0.3, rz=0.2
        )
    raise ValueError(name)


def build_model(primitive: str, density: int, worlds: int, device) -> newton.Model:
    mesh = grid_box_mesh(MESH_HALF, density)
    mesh.build_sdf(max_resolution=32, device=device)
    builder = newton.ModelBuilder()
    for _ in range(worlds):
        world = newton.ModelBuilder()
        world.add_shape_mesh(
            body=-1,
            xform=wp.transform(wp.vec3(0.0, 0.0, MESH_HALF - PENETRATION), wp.quat_identity()),
            mesh=mesh,
        )
        body = world.add_body(xform=wp.transform_identity())
        # Slightly off-centre so curved apexes fall inside a face, not on an edge.
        add_primitive(world, body, primitive, top_z=0.0, x_off=0.03)
        builder.add_world(world)
    return builder.finalize(device=device)


def time_route(
    model: newton.Model, mesh_primitive_sdf: bool, iters: int, warmup: int
) -> tuple[list[float], int, int, int]:
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        rigid_contact_max=max(4096, 512 * model.world_count),
        max_triangle_pairs=4_000_000,
        mesh_primitive_sdf=mesh_primitive_sdf,
    )
    state = model.state()
    contacts = pipeline.contacts()
    for _ in range(warmup):
        pipeline.collide(state, contacts)
    wp.synchronize_device(model.device)
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        pipeline.collide(state, contacts)
        wp.synchronize_device(model.device)
        samples.append((time.perf_counter() - t0) * 1e3)
    np_ = pipeline.narrow_phase
    tri_pairs = (
        int(np_.triangle_pairs_count.numpy()[0]) if getattr(np_, "triangle_pairs_count", None) is not None else 0
    )
    sdf_pairs = int(np_.shape_pairs_mesh_sdf_count.numpy()[0]) if np_.shape_pairs_mesh_sdf_count is not None else 0
    n_contacts = int(contacts.rigid_contact_count.numpy()[0])
    return samples, tri_pairs, sdf_pairs, n_contacts


def run(args) -> list[CaseResult]:
    device = wp.get_device(args.device)
    results: list[CaseResult] = []
    for primitive in args.primitives:
        for density in args.densities:
            model = build_model(primitive, density, args.worlds, device)
            triangles = 12 * density * density
            for route, flag in (("legacy", False), ("sdf", True)):
                samples, tri_pairs, sdf_pairs, n_contacts = time_route(model, flag, args.iters, args.warmup)
                results.append(
                    CaseResult(
                        primitive=primitive,
                        density=density,
                        triangles=triangles,
                        worlds=args.worlds,
                        route=route,
                        triangle_pairs=tri_pairs,
                        mesh_sdf_pairs=sdf_pairs,
                        contacts=n_contacts,
                        median_ms=statistics.median(samples),
                        p95_ms=float(np.percentile(samples, 95)),
                    )
                )
                r = results[-1]
                print(
                    f"{primitive:9s} n={density:3d} tris={triangles:6d} {route:6s} "
                    f"tri_pairs={tri_pairs:8d} sdf_pairs={sdf_pairs:5d} contacts={n_contacts:6d} "
                    f"median={r.median_ms:8.3f} ms p95={r.p95_ms:8.3f} ms",
                    flush=True,
                )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default=None)
    parser.add_argument("--worlds", type=int, default=16)
    parser.add_argument("--densities", default="2,4,8,16,32,64", help="grid subdivisions per box edge")
    parser.add_argument("--primitives", default=",".join(PRIMITIVES))
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    args.densities = [int(d) for d in args.densities.split(",")]
    args.primitives = [p.strip() for p in args.primitives.split(",")]
    for p in args.primitives:
        if p not in PRIMITIVES:
            parser.error(f"unknown primitive {p!r}; choose from {PRIMITIVES}")
    wp.init()
    results = run(args)
    if args.csv:
        with args.csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(asdict(results[0]).keys()))
            writer.writeheader()
            for r in results:
                writer.writerow(asdict(r))
    if args.json:
        args.json.write_text(json.dumps([asdict(r) for r in results], indent=2))


if __name__ == "__main__":
    main()
