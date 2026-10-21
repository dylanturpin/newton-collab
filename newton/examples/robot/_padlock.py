# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Frozen padlock geometry and a bounded force controller for the free key."""

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
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_f: wp.array[wp.spatial_vector],
    key_ids: wp.array[int],
    com: wp.array[wp.vec3],
    clock: wp.array[float],
    target: wp.array[float],
    rotations: wp.array[wp.quat],
    masses: wp.array[float],
    inertias: wp.array[wp.mat33],
    command: wp.array[wp.spatial_vector],
    dt: float,
    turn: float,
):
    i = wp.tid()
    key = key_ids[i]
    t = clock[0]
    blend = wp.clamp(t / 1.5, 0.0, 1.0)
    position = wp.transform_get_translation(body_q[key])
    rotation = wp.transform_get_rotation(body_q[key])
    omega = wp.spatial_bottom(body_qd[key])
    velocity = wp.spatial_top(body_qd[key]) - wp.cross(omega, wp.quat_rotate(rotation, com[key]))
    z = target[4 * i + 3] * (1.0 - blend) + target[4 * i + 2] * blend
    goal = wp.vec3(target[4 * i], target[4 * i + 1], z)
    force = 600.0 * (goal - position) - (2.0 + 600.0 * dt) * velocity
    force /= 1.0 + (2.0 * dt + 600.0 * dt * dt) / masses[i]
    force *= wp.min(1.0, 3.0 * LENGTH / wp.max(wp.length(force), 1.0e-12))
    angle = turn * wp.clamp((t - 1.8) / 1.5, 0.0, 1.0)
    desired = rotations[i] * wp.quat_from_axis_angle(wp.vec3(0, 0, 1), angle)
    error_q = desired * wp.quat_inverse(rotation)
    error = 2.0 * wp.vec3(error_q[0], error_q[1], error_q[2])
    if error_q[3] < 0.0:
        error = -error
    stiffness = wp.vec3(1.0, 1.0, 0.03)
    damping = wp.vec3(0.005, 0.005, 0.0005)
    rhs, diagonal = wp.vec3(), wp.vec3()
    for axis in range(3):
        rhs[axis] = (
            (stiffness[axis] * error[axis] - (damping[axis] + stiffness[axis] * dt) * omega[axis]) * LENGTH * LENGTH
        )
        diagonal[axis] = (damping[axis] * dt + stiffness[axis] * dt * dt) * LENGTH * LENGTH
    rotation_matrix = wp.quat_to_matrix(rotation)
    inertia = rotation_matrix * inertias[i] * wp.transpose(rotation_matrix)
    regularizer = wp.mat33(diagonal[0], 0.0, 0.0, 0.0, diagonal[1], 0.0, 0.0, 0.0, diagonal[2])
    torque = inertia * wp.inverse(inertia + regularizer) * rhs
    cap = wp.vec3(0.1, 0.1, 0.008) * LENGTH * LENGTH
    for axis in range(3):
        torque[axis] = wp.clamp(torque[axis], -cap[axis], cap[axis])
    body_f[key] = wp.spatial_vector(force, torque)
    command[i] = wp.spatial_vector(force / LENGTH, torque / (LENGTH * LENGTH))


@wp.kernel
def tick(clock: wp.array[float], playing: wp.array[int], dt: float):
    clock[0] = wp.min(3.6, clock[0] + float(playing[0]) * dt)


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


def build(key):
    """Build the held housing, passive mechanism and independently moving key."""
    with gzip.open(newton.examples.get_asset("steel_tabletop/padlock.json.gz"), "rt") as stream:
        data = json.load(stream)
    b = newton.ModelBuilder(gravity=(0, 0, 0))
    b.rigid_gap = data["collision_gap"] * LENGTH
    cfg = newton.ModelBuilder.ShapeConfig(
        density=0, mu=data["friction"], margin=0, gap=b.rigid_gap, restitution=0, is_visible=False
    )
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
                    cfg=cfg,
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
                damping=joint["damping"] * effort,
                spring_stiffness=joint["stiffness"] * effort,
                spring_ref=joint["rest"] * length,
                collision_filter_parent=not joint["collision_enabled"],
                label=joint["name"],
            )
            articulation.append(index)
            coordinates[joint["name"]] = b.joint_q_start[index]
            shapes[name] = add_cells(name, child, part)
            if joint["parent"] == "housing" and not joint["collision_enabled"]:
                b.shape_collision_filter_pairs.extend((a, c) for a in shapes["housing"] for c in shapes[name])
            pending[0:0] = [j for j in data["joints"] if j["parent"] == name]
        b.add_articulation(articulation)
    part = data["bodies"]["key" if key == "matched" else "wrong_bitting"]
    data["bodies"]["key"] = part
    ids["key"] = b.add_body(
        label="key",
        mass=part["mass"],
        com=np.asarray(part["com"]) * LENGTH,
        inertia=np.asarray(part["inertia"]) * LENGTH**2,
        lock_inertia=True,
        xform=wp.transform(wp.vec3(0, 0, data["design"]["start_z"] * LENGTH), wp.quat_identity()),
    )
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
