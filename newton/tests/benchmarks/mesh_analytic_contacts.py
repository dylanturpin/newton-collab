# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Mesh vs analytic primitive: legacy GJK/MPR, band cull, and exact feature contacts.

Sweeps mesh density and primitive type, reporting candidates, contacts and ``collide`` time::

    python newton/tests/benchmarks/mesh_analytic_contacts.py --worlds 32 --densities 4,16,64
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

MODES = {"legacy": 0, "band": 1, "exact": 2}
PRIMITIVES = ("box", "sphere", "capsule", "cylinder", "cone", "ellipsoid")
MESH_HALF = 0.25
PENETRATION = 0.01


@dataclass
class CaseResult:
    primitive: str
    density: int
    triangles: int
    worlds: int
    mode: str
    candidates: int
    contacts: int
    median_ms: float
    p95_ms: float


def grid_box_mesh(half: float, n: int) -> newton.Mesh:
    """Closed box of half-extent ``half`` whose six faces are ``n x n`` quads."""
    verts: list[list[float]] = []
    tris: list[int] = []
    for u_axis, v_axis, w_axis in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        for sign in (-1.0, 1.0):
            base = len(verts)
            for i in range(n + 1):
                for j in range(n + 1):
                    p = [0.0, 0.0, 0.0]
                    p[u_axis] = -half + 2.0 * half * i / n
                    p[v_axis] = -half + 2.0 * half * j / n
                    p[w_axis] = sign * half
                    verts.append(p)
            for i in range(n):
                for j in range(n):
                    a = base + i * (n + 1) + j
                    b = a + 1
                    c = a + (n + 1)
                    d = c + 1
                    tris += [a, c, d, a, d, b] if sign > 0.0 else [a, d, c, a, b, d]
    return newton.Mesh(np.asarray(verts, dtype=np.float32), np.asarray(tris, dtype=np.int32), compute_inertia=False)


def add_primitive(builder: newton.ModelBuilder, body: int, name: str, cfg) -> float:
    """Add the primitive at the body's origin and return the height of its topmost point."""
    ident = wp.transform_identity()
    if name == "box":
        builder.add_shape_box(body, xform=ident, hx=2.0, hy=2.0, hz=0.25, cfg=cfg)
        return 0.25
    if name == "sphere":
        builder.add_shape_sphere(body, xform=ident, radius=0.5, cfg=cfg)
        return 0.5
    if name == "capsule":
        builder.add_shape_capsule(body, xform=ident, radius=0.25, half_height=0.5, cfg=cfg)
        return 0.75
    if name == "cylinder":
        builder.add_shape_cylinder(body, xform=ident, radius=0.6, half_height=0.4, cfg=cfg)
        return 0.4
    if name == "cone":
        builder.add_shape_cone(body, xform=ident, radius=0.5, half_height=0.4, cfg=cfg)
        return 0.4
    if name == "ellipsoid":
        builder.add_shape_ellipsoid(body, xform=ident, rx=0.5, ry=0.3, rz=0.2, cfg=cfg)
        return 0.2
    raise ValueError(name)


def build_model(primitive: str, density: int, worlds: int, device) -> newton.Model:
    mesh = grid_box_mesh(MESH_HALF, density)
    builder = newton.ModelBuilder()
    for _ in range(worlds):
        world = newton.ModelBuilder()
        cfg = newton.ModelBuilder.ShapeConfig(gap=1.0e-3)
        body = world.add_body(xform=wp.transform_identity())
        top = add_primitive(world, body, primitive, cfg)
        world.add_shape_mesh(
            body=-1,
            xform=wp.transform(wp.vec3(0.03, 0.0, top + MESH_HALF - PENETRATION), wp.quat_identity()),
            mesh=mesh,
            cfg=cfg,
        )
        builder.add_world(world)
    return builder.finalize(device=device)


def time_mode(model: newton.Model, mode: str, iters: int, warmup: int):
    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        rigid_contact_max=max(4096, 512 * model.world_count),
        max_triangle_pairs=4_000_000,
        _analytic_mesh_features=MODES[mode],
    )
    state = model.state()
    contacts = pipeline.contacts()
    for _ in range(warmup):
        pipeline.collide(state, contacts)
    wp.synchronize_device(model.device)
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        pipeline.collide(state, contacts)
        wp.synchronize_device(model.device)
        samples.append((time.perf_counter() - start) * 1e3)
    candidates = int(pipeline.narrow_phase.triangle_pairs_count.numpy()[0])
    return samples, candidates, int(contacts.rigid_contact_count.numpy()[0])


def run(args) -> list[CaseResult]:
    device = wp.get_device(args.device)
    results: list[CaseResult] = []
    for primitive in args.primitives:
        for density in args.densities:
            model = build_model(primitive, density, args.worlds, device)
            triangles = 12 * density * density
            for mode in args.modes:
                samples, candidates, contacts = time_mode(model, mode, args.iters, args.warmup)
                results.append(
                    CaseResult(
                        primitive=primitive,
                        density=density,
                        triangles=triangles,
                        worlds=args.worlds,
                        mode=mode,
                        candidates=candidates,
                        contacts=contacts,
                        median_ms=statistics.median(samples),
                        p95_ms=float(np.percentile(samples, 95)),
                    )
                )
                r = results[-1]
                print(
                    f"{primitive:9s} n={density:3d} tris={triangles:6d} {mode:6s} "
                    f"candidates={candidates:8d} contacts={contacts:6d} "
                    f"median={r.median_ms:8.3f} ms p95={r.p95_ms:8.3f} ms",
                    flush=True,
                )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default=None)
    parser.add_argument("--worlds", type=int, default=32)
    parser.add_argument("--densities", default="2,4,8,16,32,64", help="grid subdivisions per box edge")
    parser.add_argument("--primitives", default=",".join(PRIMITIVES))
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    args.densities = [int(d) for d in args.densities.split(",")]
    args.primitives = [p.strip() for p in args.primitives.split(",")]
    args.modes = [m.strip() for m in args.modes.split(",")]
    for p in args.primitives:
        if p not in PRIMITIVES:
            parser.error(f"unknown primitive {p!r}; choose from {PRIMITIVES}")
    for m in args.modes:
        if m not in MODES:
            parser.error(f"unknown mode {m!r}; choose from {tuple(MODES)}")
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
