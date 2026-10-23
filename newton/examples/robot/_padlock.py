# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Frozen padlock geometry and bounded six-axis key drives."""

import gzip
import json

import numpy as np
import warp as wp

import newton
import newton.examples

# Use mm internally to resolve micrometre keyway clearances; mass stays in kg.
LENGTH = wp.constant(1000.0)


@wp.kernel
def drive(
    clock: wp.array[float],
    target: wp.array[float],
    key_q: int,
    joint_targets: wp.array[float],
    turn: float,
):
    t = clock[0]
    blend = wp.clamp(t / 1.5, 0.0, 1.0)
    joint_targets[key_q] = target[0]
    joint_targets[key_q + 1] = target[1]
    joint_targets[key_q + 2] = target[3] * (1.0 - blend) + target[2] * blend
    joint_targets[key_q + 3] = 0.0
    joint_targets[key_q + 4] = 0.0
    joint_targets[key_q + 5] = turn * wp.clamp((t - 1.8) / 1.5, 0.0, 1.0)


@wp.kernel
def tick(clock: wp.array[float], playing: wp.array[int], q: wp.array[float], key_q: int, dt: float):
    next_time = wp.min(3.6, clock[0] + float(playing[0]) * dt)
    if clock[0] <= 1.8 and q[key_q + 2] < -0.2:
        next_time = wp.min(next_time, 1.8)
    clock[0] = next_time


@wp.kernel
def monitor(
    q: wp.array[float],
    qd: wp.array[float],
    overflow: wp.array[wp.bool],
    contacts: wp.array[int],
    metrics: wp.array[int],
    capacity: int,
):
    i = wp.tid()
    if i < q.shape[0] and not wp.isfinite(q[i]):
        wp.atomic_max(metrics, 2, 1)
    if i < qd.shape[0] and not wp.isfinite(qd[i]):
        wp.atomic_max(metrics, 2, 1)
    if i < overflow.shape[0] and overflow[i]:
        wp.atomic_max(metrics, 1, 1)
    if i == 0:
        wp.atomic_max(metrics, 0, contacts[0])
        if contacts[0] > capacity:
            wp.atomic_max(metrics, 3, 1)


def build(key, dt):
    """Build the held housing, passive mechanism and independently moving key."""
    with gzip.open(newton.examples.get_asset("steel_tabletop/padlock.json.gz"), "rt") as stream:
        data = json.load(stream)
    b = newton.ModelBuilder(gravity=(0, 0, 0))
    b.rigid_gap = data["collision_gap"] * LENGTH
    insertion_speed = (data["design"]["seated_target_z"] - data["design"]["start_z"]) * LENGTH / 1.5
    cfg = newton.ModelBuilder.ShapeConfig(
        density=0,
        mu=data["friction"],
        margin=0,
        gap=max(b.rigid_gap, insertion_speed * dt),
        restitution=0,
        is_visible=False,
    )
    shackle_mass = data["bodies"]["guide"]["mass"] + data["bodies"]["shackle"]["mass"]
    shackle_spring = next(joint for joint in data["joints"] if joint["name"] == "shackle_lift")
    shackle_acceleration = shackle_spring["stiffness"] * shackle_spring["rest"] * LENGTH / shackle_mass
    latch_cfg = cfg.copy()
    latch_cfg.gap = max(cfg.gap, 0.5 * shackle_acceleration * dt * dt)
    ids, shapes, coordinates = {"housing": -1}, {}, {}

    def add_cells(name, body, source):
        result = []
        for cell in source["cells"]:
            vertices = np.asarray(cell["vertices"])
            center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2
            mesh = newton.Mesh((vertices - center) * LENGTH, np.asarray(cell["faces"]).flatten(), compute_inertia=False)
            result.append(
                b.add_shape_convex_hull(
                    body,
                    mesh=mesh,
                    cfg=latch_cfg if name in ("shackle", "left_ball", "right_ball") else cfg,
                    xform=wp.transform(wp.vec3(*(center * LENGTH)), wp.quat_identity()),
                    label=f"{name}/collision_{len(result)}",
                )
            )
        return result

    shapes["housing"] = add_cells("housing", -1, data["bodies"]["housing"])
    for root in (j for j in data["joints"] if j["parent"] == "housing"):
        pending, articulation = [root], []
        while pending:
            joint = pending.pop(0)
            name = joint["child"]
            part = data["bodies"][name]
            child = b.add_link(
                label=name,
                mass=part["mass"],
                com=np.asarray(part["com"]) * LENGTH,
                inertia=np.asarray(part["inertia"]) * LENGTH**2,
                lock_inertia=True,
            )
            ids[name] = child
            frames = []
            for frame in (joint["parent_frame"], joint["child_frame"]):
                w, x, y, z = frame["wxyz"]
                frames.append(wp.transform(wp.vec3(*(np.asarray(frame["xyz"]) * LENGTH)), wp.quat(x, y, z, w)))
            angular = joint["kind"] == "revolute"
            length, effort = (1.0, LENGTH**2) if angular else (LENGTH, 1.0)
            constructor = b.add_joint_revolute if angular else b.add_joint_prismatic
            index = constructor(
                ids[joint["parent"]],
                child,
                axis=joint["axis"],
                parent_xform=frames[0],
                child_xform=frames[1],
                limit_lower=joint["limits"][0] * length,
                limit_upper=joint["limits"][1] * length,
                armature=max(0.0, shackle_mass / 16.0 - part["mass"]) if not angular else 0.0,
                target_ke=joint["stiffness"] * effort,
                target_kd=joint["damping"] * effort,
                target_pos=joint["rest"] * length,
                actuator_mode=newton.JointTargetMode.POSITION
                if joint["stiffness"]
                else newton.JointTargetMode.VELOCITY,
                collision_filter_parent=not joint["collision_enabled"],
                label=joint["name"],
            )
            b.joint_target_q[b.joint_q_start[index]] = joint["rest"] * length
            articulation.append(index)
            coordinates[joint["name"]] = b.joint_q_start[index]
            shapes[name] = add_cells(name, child, part)
            if joint["parent"] == "housing" and not joint["collision_enabled"]:
                b.shape_collision_filter_pairs.extend((a, c) for a in shapes["housing"] for c in shapes[name])
            pending[0:0] = [j for j in data["joints"] if j["parent"] == name]
        b.add_articulation(articulation)
    part = data["bodies"]["key" if key == "matched" else "wrong_bitting"]
    data["bodies"]["key"] = part
    ids["key"] = b.add_link(
        label="key",
        mass=part["mass"],
        com=np.asarray(part["com"]) * LENGTH,
        inertia=np.asarray(part["inertia"]) * LENGTH**2,
        lock_inertia=True,
        xform=wp.transform(wp.vec3(0, 0, data["design"]["start_z"] * LENGTH), wp.quat_identity()),
    )
    linear_axes = [
        newton.ModelBuilder.JointDofConfig(
            axis=axis,
            target_ke=600.0 if index == 2 else 6000.0,
            target_kd=2.0,
            target_pos=data["design"]["start_z"] * LENGTH if index == 2 else 0.0,
            effort_limit=3.0 * LENGTH / np.sqrt(3.0),
            velocity_limit=2.0 * insertion_speed,
        )
        for index, axis in enumerate(np.eye(3))
    ]
    angular_axes = [
        newton.ModelBuilder.JointDofConfig(
            axis=axis,
            target_ke=stiffness * LENGTH**2,
            target_kd=damping * LENGTH**2,
            effort_limit=cap * LENGTH**2,
            velocity_limit=2.0 * data["design"]["turn"] / 1.5,
        )
        for axis, stiffness, damping, cap in zip(
            np.eye(3), (100.0, 100.0, 0.03), (0.005, 0.005, 0.0005), (0.1, 0.1, 0.008), strict=True
        )
    ]
    key_joint = b.add_joint_d6(-1, ids["key"], linear_axes=linear_axes, angular_axes=angular_axes, label="key_drive")
    coordinates["key_drive"] = b.joint_q_start[key_joint]
    b.joint_q[coordinates["key_drive"] + 2] = data["design"]["start_z"] * LENGTH
    b.add_articulation([key_joint])
    add_cells("key", ids["key"], part)

    # Display meshes have neither collision flags nor mass. Both views share all 791 collision cells.
    visual_cfg = newton.ModelBuilder.ShapeConfig(density=0, has_shape_collision=False, has_particle_collision=False)
    views = {"full": [], "cutaway": []}
    for name, body in ids.items():
        part = data["bodies"][name]
        for i, visual in enumerate(part["visuals"]):
            for view in ("full", "cutaway") if i == 0 and "cutaway" in part else ("both",):
                mesh_data = part["cutaway"] if view == "cutaway" else visual
                vertices = np.asarray(mesh_data["vertices"])[np.asarray(mesh_data["faces"])].reshape(-1, 3)
                mesh = newton.Mesh(
                    vertices * LENGTH,
                    np.arange(len(vertices), dtype=np.int32),
                    normals=np.asarray(mesh_data["normals"]).reshape(-1, 3),
                    compute_inertia=False,
                    color=mesh_data["color"],
                    roughness=0.28,
                    metallic=0.85,
                )
                shape = b.add_shape_mesh(body, mesh=mesh, cfg=visual_cfg, label=f"{name}/{view}_{i}")
                if view != "both":
                    views[view].append(shape)
    return b, data["design"], ids["key"], coordinates, views
