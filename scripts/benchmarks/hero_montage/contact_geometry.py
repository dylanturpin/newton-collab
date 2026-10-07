# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Read-only narrow-phase geometry inspection; never steps a physics solver."""

import numpy as np
import warp as wp

import newton


def surface_contacts(builder, state, contacts):
    """Return geometric surface gaps in metres; negative means penetration.

    Contact normals point from shape 0 to shape 1 in world coordinates.
    Contact points are in their respective body frames (world if body == -1).
    Margin arrays include each primitive's effective radius and shape margin.
    """
    count = int(contacts.rigid_contact_count.numpy()[0])
    a = contacts.rigid_contact_shape0.numpy()
    d = contacts.rigid_contact_shape1.numpy()
    if count > len(a):
        raise RuntimeError(f"Contact capacity exceeded: {count} > {len(a)}")
    p0 = contacts.rigid_contact_point0.numpy()
    p1 = contacts.rigid_contact_point1.numpy()
    normals = contacts.rigid_contact_normal.numpy()
    margin0 = contacts.rigid_contact_margin0.numpy()
    margin1 = contacts.rigid_contact_margin1.numpy()
    poses = state.body_q.numpy()

    def world_point(body, point):
        if body < 0:
            return point
        return np.asarray(wp.transform_point(wp.transform(*poses[body]), wp.vec3(*point)))

    rows = []
    for k in range(count):
        i, j = int(a[k]), int(d[k])
        if i < 0 or j < 0:
            continue
        bi, bj = builder.shape_body[i], builder.shape_body[j]
        w0, w1 = world_point(bi, p0[k]), world_point(bj, p1[k])
        gap = float(np.dot(w1 - w0, normals[k]) - margin0[k] - margin1[k])
        rows.append(
            {
                "shape0": i,
                "shape1": j,
                "body0": bi,
                "body1": bj,
                "body_label0": builder.body_label[bi] if bi >= 0 else "world",
                "body_label1": builder.body_label[bj] if bj >= 0 else "world",
                "shape_label0": builder.shape_label[i],
                "shape_label1": builder.shape_label[j],
                "surface_gap_m": gap,
                "world_point0": w0.tolist(),
                "world_point1": w1.tolist(),
                "normal": normals[k].tolist(),
            }
        )
    return rows


def query_geometry(builder, model, body_q=None):
    """Run FK or install recorded body poses, then query collisions only.

    model may be on CUDA. The returned rows are CPU diagnostic values.
    Existing collision filters are respected; separately inspect custom mount
    geometry where robot/self-collision filters could hide intersections.
    """
    state = model.state()
    if body_q is None:
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    else:
        state.body_q.assign(np.asarray(body_q, dtype=np.float32))
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=8192, reduce_contacts=True)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    return surface_contacts(builder, state, contacts)
