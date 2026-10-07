# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Load original procedural convex pieces and articulated topology into Newton."""

import json
from functools import cache
from pathlib import Path

import numpy as np
import warp as wp

import newton

ROOT = Path(__file__).parent / "assets" / "tasks"
COLORS = [(0.035, 0.43, 0.78), (0.02, 0.65, 0.47), (0.96, 0.61, 0.065)]


@cache
def load(name):
    meta = json.loads((ROOT / f"{name}.json").read_text())
    with np.load(ROOT / f"{name}.npz") as src:
        arrays = dict(src)
    return meta, arrays


def transform(record, scale):
    w, x, y, z = record["quaternion"]
    return wp.transform(wp.vec3(*(np.asarray(record["translation"]) * scale)), wp.quat(x, y, z, w))


def collision_bounds(name, scale, orientation):
    """Bound a single-body convex asset after its intended spawn rotation."""
    meta, arrays = load(name)
    assert len(meta["bodies"]) == 1
    points = []
    rotation = wp.transform(wp.vec3(), orientation)
    for cell in meta["bodies"][0]["collisions"]:
        assert cell["kind"] == "convex"
        local = transform({"translation": cell["center"], "quaternion": cell["quaternion"]}, scale)
        for vertex in arrays[cell["prefix"] + "_v"]:
            points.append(np.asarray(wp.transform_point(rotation * local, wp.vec3(*(vertex * scale)))))
    points = np.asarray(points)
    return points.min(axis=0), points.max(axis=0)


def import_object(
    b,
    name,
    pose,
    *,
    scale=1.0,
    free_root=False,
    moving=None,
    detached=(),
    body_poses=None,
    variant=0,
    register_articulations=True,
    finish_color=None,
    parent_body=-1,
):
    """Keep authored joints; optionally fix all except a selected removable part.

    `moving=None` preserves every authored joint. A set selects which authored
    free/scalar bodies remain movable, while the rest form a fixed fixture.
    A caller extending the tree may defer registration, then register the
    imported joints together with its additional links.
    """
    meta, arrays = load(name)
    body_poses = body_poses or {}
    records = {r["id"]: r for r in meta["bodies"]}
    incoming = {j["child"]: j for j in meta["joints"]}
    children = {name: [] for name in records}
    for joint in meta["joints"]:
        children[joint["parent"]].append(joint["child"])
    mapping, qmap, groups, group_for = {}, {}, [], {}
    pending = []

    def joint_kind(body_id):
        original = incoming.get(body_id)
        kind = "free" if original is None and free_root else "fixed" if original is None else original["kind"]
        if moving is not None and body_id not in moving and original is not None:
            kind = "fixed"
        return "free" if body_id in detached else kind

    def add(body_id):
        record = records[body_id]
        carrier = None
        if name == "truck" and body_id.startswith("wheel.1."):
            original = incoming[body_id]
            carrier = b.add_link(
                mass=0.005, inertia=wp.mat33(np.eye(3, dtype=np.float32) * 1e-6), label=f"truck/{body_id}/suspension"
            )
            suspension = b.add_joint_prismatic(
                mapping[original["parent"]],
                carrier,
                parent_xform=transform(original["parent_frame"], scale),
                axis=(0, 0, 1),
                limit_lower=-0.002,
                limit_upper=0.002,
                spring_stiffness=100,
                spring_ref=-0.001,
                damping=0.2,
                armature=0.005,
                label=f"truck/{body_id}/suspension_slide",
            )
            groups[group_for[original["parent"]]].append(suspension)
        body = b.add_link(
            xform=body_poses.get(body_id, pose),
            mass=record["mass"] * scale**3,
            com=wp.vec3(*(np.asarray(record["com"]) * scale)),
            inertia=wp.mat33(*(np.asarray(record["inertia"]) * scale**5).ravel()),
            lock_inertia=True,
            label=f"procgen/{name}/{body_id}",
        )
        mapping[body_id] = body
        original = incoming.get(body_id)
        kind = joint_kind(body_id)
        if original is None or kind == "free":
            groups.append([])
            group_for[body_id] = len(groups) - 1
            if kind == "free":
                joint = b.add_joint_free(body, label=f"{name}/{body_id}/free")
            else:
                joint = b.add_joint_fixed(parent_body, body, parent_xform=pose, label=f"{name}/{body_id}/fixed")
        else:
            parent = mapping[original["parent"]] if carrier is None else carrier
            group_for[body_id] = group_for[original["parent"]]
            common = {
                "parent_xform": transform(original["parent_frame"], scale)
                if carrier is None
                else wp.transform_identity(),
                "child_xform": transform(original["child_frame"], scale),
                "label": f"{name}/{original['id']}",
            }
            if kind == "fixed":
                joint = b.add_joint_fixed(parent, body, **common)
            else:
                options = dict(
                    axis=wp.vec3(*original["axis"]),
                    target_ke=0.0,
                    target_kd=0.0,
                    armature=1.0e-5,
                    friction=0.0,
                    **common,
                )
                if original["limits"] is not None:
                    factor = scale if kind == "prismatic" else 1
                    options.update(
                        limit_lower=original["limits"][0] * factor, limit_upper=original["limits"][1] * factor
                    )
                if kind in ("continuous", "revolute"):
                    joint = b.add_joint_revolute(parent, body, **options)
                elif kind == "prismatic":
                    joint = b.add_joint_prismatic(parent, body, **options)
                else:
                    raise ValueError(kind)
                qmap[original["id"]] = b.joint_q_start[joint]
        groups[group_for[body_id]].append(joint)
        cfg = b.default_shape_cfg.copy()
        cfg.density = 0
        cfg.is_visible = False
        cfg.gap = 0.00025
        if name in ("hex_bolt", "socket_bolt", "hex_nut", "washer"):
            cfg.mu = 0.18
        collisions = record["collisions"]
        if name in ("train", "truck") and body_id.startswith("wheel."):
            bounds = np.asarray(record["bounds"]) * scale
            center = bounds.mean(axis=0)
            half = (bounds[1] - bounds[0]) / 2
            # Circular rolling contact avoids the polygonal proxy's facet rocking.
            b.add_shape_cylinder(
                body,
                xform=wp.transform(wp.vec3(*center), wp.quat(0.0, 0.70710678, 0.0, 0.70710678)),
                radius=float(half[2]),
                half_height=float(half[0]),
                cfg=cfg,
            )
            collisions = []
        for c in collisions:
            prefix = c["prefix"]
            local = transform({"translation": c["center"], "quaternion": c["quaternion"]}, scale)
            if c["kind"] == "convex":
                mesh = newton.Mesh(arrays[prefix + "_v"] * scale, arrays[prefix + "_f"].ravel(), compute_inertia=False)
                b.add_shape_mesh(body, xform=local, mesh=mesh, cfg=cfg, label=f"procgen_collision/{name}")
                b.shape_type[-1] = newton.GeoType.CONVEX_MESH
            elif c["kind"] == "box":
                half = np.asarray(c["size"]) * scale / 2
                b.add_shape_box(body, xform=local, hx=half[0], hy=half[1], hz=half[2], cfg=cfg)
            else:
                raise ValueError(f"Unsupported procedural collision {c['kind']}")
        visual = cfg.copy()
        visual.is_visible = True
        visual.has_shape_collision = False
        visual.has_particle_collision = False
        for v in record["visuals"]:
            prefix = v["prefix"]
            material = v["material"]
            color = (
                finish_color
                if finish_color is not None
                else (COLORS[(list(records).index(body_id) + variant) % 3] if material == "plastic" else v["color"])
            )
            mesh = newton.Mesh(
                arrays[prefix + "_v"] * scale,
                arrays[prefix + "_f"].ravel(),
                normals=arrays[prefix + "_n"],
                compute_inertia=False,
                roughness=0.24 if material == "brass" else 0.36,
                metallic=0.85 if material == "brass" else 0.65 if material in ("chrome", "aluminum") else 0.0,
            )
            b.add_shape_mesh(body, mesh=mesh, color=color, cfg=visual, label=f"procgen_visual/{name}")
        for child in children[body_id]:
            if joint_kind(child) == "free":
                pending.append(child)
            else:
                add(child)

    add(meta["root"])
    while pending:
        add(pending.pop(0))
    if register_articulations:
        for group in groups:
            b.add_articulation(group)
    # A hinge already constrains its bearing. Disable contact against the
    # parent's welded parts as well as the immediate parent: imported axles
    # often live in a separate fixed chrome body. Keep gear-to-gear, wheel-to-
    # ground and every detached assembly-piece contact enabled.
    weld = {key: key for key in mapping}

    def root(key):
        while weld[key] != key:
            key = weld[key]
        return key

    for joint in meta["joints"]:
        if joint_kind(joint["child"]) == "fixed":
            weld[root(joint["child"])] = root(joint["parent"])
    adjacent = set()
    for joint in meta["joints"]:
        if joint_kind(joint["child"]) in ("continuous", "revolute", "prismatic"):
            adjacent.add(frozenset((root(joint["parent"]), root(joint["child"]))))
    shapes = {key: [i for i, body in enumerate(b.shape_body) if body == value] for key, value in mapping.items()}
    ids = list(mapping)
    for i, a in enumerate(ids):
        for c in ids[i + 1 :]:
            if root(a) == root(c) or frozenset((root(a), root(c))) in adjacent:
                for sa in shapes[a]:
                    for sc in shapes[c]:
                        b.add_shape_collision_filter_pair(sa, sc)
    return mapping, qmap
