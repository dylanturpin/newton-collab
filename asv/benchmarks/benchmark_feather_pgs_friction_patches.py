# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare captured FeatherPGS collision and solve cost across friction implementations.

Run this same script against the PR's base and feature revision::

    uv run python asv/benchmarks/benchmark_feather_pgs_friction_patches.py --worlds 64 --tiles 4

Each one-kilogram body is assembled from ``tiles ** 2`` touching boxes. Its
footprint, total mass, and applied load stay fixed as contact density increases.
Both revisions use contact matching for a like-for-like comparison; persistent
patch anchors themselves do not require it. Compilation and warmup are excluded.
"""

import argparse
import json
import os
import time

import numpy as np
import warp as wp

if "FPGS_PATCH_CACHE" in os.environ:
    wp.config.kernel_cache_dir = os.environ["FPGS_PATCH_CACHE"]
import newton


@wp.kernel
def _apply_load(body_f: wp.array[wp.spatial_vector]):
    i = wp.tid()
    body_f[i] = wp.spatial_vector(1.0, 0.0, 0.0, 0.0, 0.0, 0.03)


def run(worlds: int, tiles: int, beta: float, steps: int):
    """Measure an even number of captured steps and report row counts and motion."""
    if worlds < 1 or tiles < 1 or tiles > 4 or steps < 2 or steps % 2:
        raise ValueError("Require worlds >= 1, 1 <= tiles <= 4, and an even steps >= 2")
    template = newton.ModelBuilder()
    template.add_ground_plane()
    body = template.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0999), wp.quat_identity()))
    h = 0.1 / tiles
    for x in range(tiles):
        for y in range(tiles):
            template.add_shape_box(
                body,
                hx=h,
                hy=h,
                hz=0.1,
                xform=wp.transform(wp.vec3(-0.1 + h + 2 * h * x, -0.1 + h + 2 * h * y, 0), wp.quat_identity()),
                cfg=newton.ModelBuilder.ShapeConfig(density=125.0, mu=0.5),
            )
    builder = newton.ModelBuilder()
    builder.replicate(template, worlds)
    model = builder.finalize(device="cuda:0")
    model.rigid_contact_max = worlds * tiles * tiles * 4
    solver = newton.solvers.SolverFeatherPGS(
        model,
        pgs_mode="matrix_free",
        pgs_iterations=32,
        pgs_beta=0.05,
        friction_anchor_beta=beta,
        dense_max_constraints=32,
        mf_max_constraints=256,
        warn_constraint_overflow=True,
    )
    pipeline = newton.CollisionPipeline(
        model, rigid_contact_max=model.rigid_contact_max, broad_phase="nxn", contact_matching="latest"
    )
    contacts = pipeline.contacts()
    s0, s1 = model.state(), model.state()
    control = model.control()

    def step(source, dest):
        source.clear_forces()
        wp.launch(_apply_load, dim=model.body_count, inputs=[source.body_f], device=model.device)
        pipeline.collide(source, contacts)
        solver.step(source, dest, control, contacts, 0.005)

    for _ in range(10):
        step(s0, s1)
        step(s1, s0)
    with wp.ScopedCapture(device=model.device) as capture:
        solver.seed_double_buffer_events()
        step(s0, s1)
        step(s1, s0)
    wp.synchronize()
    before = s0.body_q.numpy().copy()
    start = time.perf_counter()
    for _ in range(steps // 2):
        wp.capture_launch(capture.graph)
    wp.synchronize()
    elapsed = time.perf_counter() - start
    after = s0.body_q.numpy()
    types = solver.mf_row_type.numpy()
    counts = solver.mf_constraint_count.numpy()
    normal_rows = sum(int(np.count_nonzero(types[w, : counts[w]] == 0)) for w in range(worlds))
    friction_rows = sum(int(np.count_nonzero(types[w, : counts[w]] == 2)) for w in range(worlds))
    result = {
        "newton": newton.__file__,
        "warp": wp.__version__,
        "worlds": worlds,
        "tiles": tiles,
        "beta": beta,
        "steps": steps,
        "milliseconds_per_step": elapsed * 1000 / steps,
        "contacts": int(contacts.rigid_contact_count.numpy()[0]),
        "normal_rows": normal_rows,
        "friction_rows": friction_rows,
        "max_translation": float(np.linalg.norm(after[:, :3] - before[:, :3], axis=1).max()),
        "max_quaternion_change": float(np.linalg.norm(after[:, 3:] - before[:, 3:], axis=1).max()),
        "finite": bool(np.isfinite(after).all()),
    }
    print("RESULT " + json.dumps(result), flush=True)
    assert result["finite"]
    assert normal_rows > 0
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", type=int, default=64)
    parser.add_argument("--tiles", type=int, default=4)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--steps", type=int, default=400)
    args = parser.parse_args()
    run(args.worlds, args.tiles, args.beta, args.steps)
