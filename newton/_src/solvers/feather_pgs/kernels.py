# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warp as wp

from ...core import transform_twist
from ...sim import JointType
from ...sim.articulation import (
    compute_2d_rotational_dofs,
    compute_3d_rotational_dofs,
)

PGS_CONSTRAINT_TYPE_CONTACT = 0
PGS_CONSTRAINT_TYPE_JOINT_TARGET = 1
PGS_CONSTRAINT_TYPE_FRICTION = 2


@wp.kernel
def copy_int_array_masked(
    src: wp.array(dtype=int),
    mask: wp.array(dtype=int),
    # outputs
    dst: wp.array(dtype=int),
):
    tid = wp.tid()
    if mask[tid] != 0:
        dst[tid] = src[tid]


@wp.kernel
def compute_spatial_inertia(
    body_inertia: wp.array(dtype=wp.mat33),
    body_mass: wp.array(dtype=float),
    # outputs
    body_I_m: wp.array(dtype=wp.spatial_matrix),
):
    tid = wp.tid()
    I = body_inertia[tid]
    m = body_mass[tid]
    # fmt: off
    body_I_m[tid] = wp.spatial_matrix(
        m,   0.0, 0.0, 0.0,     0.0,     0.0,
        0.0, m,   0.0, 0.0,     0.0,     0.0,
        0.0, 0.0, m,   0.0,     0.0,     0.0,
        0.0, 0.0, 0.0, I[0, 0], I[0, 1], I[0, 2],
        0.0, 0.0, 0.0, I[1, 0], I[1, 1], I[1, 2],
        0.0, 0.0, 0.0, I[2, 0], I[2, 1], I[2, 2],
    )
    # fmt: on


@wp.kernel
def compute_com_transforms(
    body_com: wp.array(dtype=wp.vec3),
    # outputs
    body_X_com: wp.array(dtype=wp.transform),
):
    tid = wp.tid()
    com = body_com[tid]
    body_X_com[tid] = wp.transform(com, wp.quat_identity())


@wp.func
def transform_spatial_inertia(t: wp.transform, I: wp.spatial_matrix):
    """
    Transform a spatial inertia tensor to a new coordinate frame.

    This computes the change of coordinates for a spatial inertia tensor under a rigid-body
    transformation `t`. The result is mathematically equivalent to:

        adj_t^-T * I * adj_t^-1

    where `adj_t` is the adjoint transformation matrix of `t`, and `I` is the spatial inertia
    tensor in the original frame. This operation is described in Frank & Park, "Modern Robotics",
    Section 8.2.3 (pg. 290).

    Args:
        t (wp.transform): The rigid-body transform (destination ← source).
        I (wp.spatial_matrix): The spatial inertia tensor in the source frame.

    Returns:
        wp.spatial_matrix: The spatial inertia tensor expressed in the destination frame.
    """
    t_inv = wp.transform_inverse(t)

    q = wp.transform_get_rotation(t_inv)
    p = wp.transform_get_translation(t_inv)

    r1 = wp.quat_rotate(q, wp.vec3(1.0, 0.0, 0.0))
    r2 = wp.quat_rotate(q, wp.vec3(0.0, 1.0, 0.0))
    r3 = wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0))

    R = wp.matrix_from_cols(r1, r2, r3)
    S = wp.skew(p) @ R

    T = wp.spatial_matrix(
        R[0, 0],
        R[0, 1],
        R[0, 2],
        S[0, 0],
        S[0, 1],
        S[0, 2],
        R[1, 0],
        R[1, 1],
        R[1, 2],
        S[1, 0],
        S[1, 1],
        S[1, 2],
        R[2, 0],
        R[2, 1],
        R[2, 2],
        S[2, 0],
        S[2, 1],
        S[2, 2],
        0.0,
        0.0,
        0.0,
        R[0, 0],
        R[0, 1],
        R[0, 2],
        0.0,
        0.0,
        0.0,
        R[1, 0],
        R[1, 1],
        R[1, 2],
        0.0,
        0.0,
        0.0,
        R[2, 0],
        R[2, 1],
        R[2, 2],
    )

    return wp.mul(wp.mul(wp.transpose(T), I), T)


# compute transform across a joint
@wp.func
def jcalc_transform(
    type: int,
    joint_axis: wp.array(dtype=wp.vec3),
    axis_start: int,
    lin_axis_count: int,
    ang_axis_count: int,
    joint_q: wp.array(dtype=float),
    q_start: int,
):
    if type == JointType.PRISMATIC:
        q = joint_q[q_start]
        axis = joint_axis[axis_start]
        X_jc = wp.transform(axis * q, wp.quat_identity())
        return X_jc

    if type == JointType.REVOLUTE:
        q = joint_q[q_start]
        axis = joint_axis[axis_start]
        X_jc = wp.transform(wp.vec3(), wp.quat_from_axis_angle(axis, q))
        return X_jc

    if type == JointType.BALL:
        qx = joint_q[q_start + 0]
        qy = joint_q[q_start + 1]
        qz = joint_q[q_start + 2]
        qw = joint_q[q_start + 3]

        X_jc = wp.transform(wp.vec3(), wp.quat(qx, qy, qz, qw))
        return X_jc

    if type == JointType.FIXED:
        X_jc = wp.transform_identity()
        return X_jc

    if type == JointType.FREE or type == JointType.DISTANCE:
        px = joint_q[q_start + 0]
        py = joint_q[q_start + 1]
        pz = joint_q[q_start + 2]

        qx = joint_q[q_start + 3]
        qy = joint_q[q_start + 4]
        qz = joint_q[q_start + 5]
        qw = joint_q[q_start + 6]

        X_jc = wp.transform(wp.vec3(px, py, pz), wp.quat(qx, qy, qz, qw))
        return X_jc

    if type == JointType.D6:
        pos = wp.vec3(0.0)
        rot = wp.quat_identity()

        # unroll for loop to ensure joint actions remain differentiable
        # (since differentiating through a for loop that updates a local variable is not supported)

        if lin_axis_count > 0:
            axis = joint_axis[axis_start + 0]
            pos += axis * joint_q[q_start + 0]
        if lin_axis_count > 1:
            axis = joint_axis[axis_start + 1]
            pos += axis * joint_q[q_start + 1]
        if lin_axis_count > 2:
            axis = joint_axis[axis_start + 2]
            pos += axis * joint_q[q_start + 2]

        ia = axis_start + lin_axis_count
        iq = q_start + lin_axis_count
        if ang_axis_count == 1:
            axis = joint_axis[ia]
            rot = wp.quat_from_axis_angle(axis, joint_q[iq])
        if ang_axis_count == 2:
            rot, _ = compute_2d_rotational_dofs(
                joint_axis[ia + 0],
                joint_axis[ia + 1],
                joint_q[iq + 0],
                joint_q[iq + 1],
                0.0,
                0.0,
            )
        if ang_axis_count == 3:
            rot, _ = compute_3d_rotational_dofs(
                joint_axis[ia + 0],
                joint_axis[ia + 1],
                joint_axis[ia + 2],
                joint_q[iq + 0],
                joint_q[iq + 1],
                joint_q[iq + 2],
                0.0,
                0.0,
                0.0,
            )

        X_jc = wp.transform(pos, rot)
        return X_jc

    # default case
    return wp.transform_identity()


# compute motion subspace and velocity for a joint
@wp.func
def jcalc_motion(
    type: int,
    joint_axis: wp.array(dtype=wp.vec3),
    lin_axis_count: int,
    ang_axis_count: int,
    X_sc: wp.transform,
    joint_qd: wp.array(dtype=float),
    qd_start: int,
    # outputs
    joint_S_s: wp.array(dtype=wp.spatial_vector),
):
    if type == JointType.PRISMATIC:
        axis = joint_axis[qd_start]
        S_s = transform_twist(X_sc, wp.spatial_vector(axis, wp.vec3()))
        v_j_s = S_s * joint_qd[qd_start]
        joint_S_s[qd_start] = S_s
        return v_j_s

    if type == JointType.REVOLUTE:
        axis = joint_axis[qd_start]
        S_s = transform_twist(X_sc, wp.spatial_vector(wp.vec3(), axis))
        v_j_s = S_s * joint_qd[qd_start]
        joint_S_s[qd_start] = S_s
        return v_j_s

    if type == JointType.D6:
        v_j_s = wp.spatial_vector()
        if lin_axis_count > 0:
            axis = joint_axis[qd_start + 0]
            S_s = transform_twist(X_sc, wp.spatial_vector(axis, wp.vec3()))
            v_j_s += S_s * joint_qd[qd_start + 0]
            joint_S_s[qd_start + 0] = S_s
        if lin_axis_count > 1:
            axis = joint_axis[qd_start + 1]
            S_s = transform_twist(X_sc, wp.spatial_vector(axis, wp.vec3()))
            v_j_s += S_s * joint_qd[qd_start + 1]
            joint_S_s[qd_start + 1] = S_s
        if lin_axis_count > 2:
            axis = joint_axis[qd_start + 2]
            S_s = transform_twist(X_sc, wp.spatial_vector(axis, wp.vec3()))
            v_j_s += S_s * joint_qd[qd_start + 2]
            joint_S_s[qd_start + 2] = S_s
        if ang_axis_count > 0:
            axis = joint_axis[qd_start + lin_axis_count + 0]
            S_s = transform_twist(X_sc, wp.spatial_vector(wp.vec3(), axis))
            v_j_s += S_s * joint_qd[qd_start + lin_axis_count + 0]
            joint_S_s[qd_start + lin_axis_count + 0] = S_s
        if ang_axis_count > 1:
            axis = joint_axis[qd_start + lin_axis_count + 1]
            S_s = transform_twist(X_sc, wp.spatial_vector(wp.vec3(), axis))
            v_j_s += S_s * joint_qd[qd_start + lin_axis_count + 1]
            joint_S_s[qd_start + lin_axis_count + 1] = S_s
        if ang_axis_count > 2:
            axis = joint_axis[qd_start + lin_axis_count + 2]
            S_s = transform_twist(X_sc, wp.spatial_vector(wp.vec3(), axis))
            v_j_s += S_s * joint_qd[qd_start + lin_axis_count + 2]
            joint_S_s[qd_start + lin_axis_count + 2] = S_s

        return v_j_s

    if type == JointType.BALL:
        S_0 = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 0.0, 1.0, 0.0, 0.0))
        S_1 = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 1.0, 0.0))
        S_2 = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 1.0))

        joint_S_s[qd_start + 0] = S_0
        joint_S_s[qd_start + 1] = S_1
        joint_S_s[qd_start + 2] = S_2

        return S_0 * joint_qd[qd_start + 0] + S_1 * joint_qd[qd_start + 1] + S_2 * joint_qd[qd_start + 2]

    if type == JointType.FIXED:
        return wp.spatial_vector()

    if type == JointType.FREE or type == JointType.DISTANCE:
        v_j_s = transform_twist(
            X_sc,
            wp.spatial_vector(
                joint_qd[qd_start + 0],
                joint_qd[qd_start + 1],
                joint_qd[qd_start + 2],
                joint_qd[qd_start + 3],
                joint_qd[qd_start + 4],
                joint_qd[qd_start + 5],
            ),
        )

        joint_S_s[qd_start + 0] = transform_twist(X_sc, wp.spatial_vector(1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        joint_S_s[qd_start + 1] = transform_twist(X_sc, wp.spatial_vector(0.0, 1.0, 0.0, 0.0, 0.0, 0.0))
        joint_S_s[qd_start + 2] = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
        joint_S_s[qd_start + 3] = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 0.0, 1.0, 0.0, 0.0))
        joint_S_s[qd_start + 4] = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 1.0, 0.0))
        joint_S_s[qd_start + 5] = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 1.0))

        return v_j_s

    wp.printf("jcalc_motion not implemented for joint type %d\n", type)

    # default case
    return wp.spatial_vector()


# computes joint space forces/torques in tau
@wp.func
def jcalc_tau(
    type: int,
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    joint_f: wp.array(dtype=float),
    dof_start: int,
    lin_axis_count: int,
    ang_axis_count: int,
    body_f_s: wp.spatial_vector,
    # outputs
    tau: wp.array(dtype=float),
):
    if type == JointType.BALL:
        # target_ke = joint_target_ke[dof_start]
        # target_kd = joint_target_kd[dof_start]

        for i in range(3):
            S_s = joint_S_s[dof_start + i]

            # w = joint_qd[dof_start + i]
            # r = joint_q[coord_start + i]

            tau[dof_start + i] = -wp.dot(S_s, body_f_s) + joint_f[dof_start + i]
            # tau -= w * target_kd - r * target_ke

        return

    if type == JointType.FREE or type == JointType.DISTANCE:
        for i in range(6):
            S_s = joint_S_s[dof_start + i]
            tau[dof_start + i] = -wp.dot(S_s, body_f_s) + joint_f[dof_start + i]

        return

    if type == JointType.PRISMATIC or type == JointType.REVOLUTE or type == JointType.D6:
        axis_count = lin_axis_count + ang_axis_count

        for i in range(axis_count):
            j = dof_start + i
            S_s = joint_S_s[j]
            # total torque / force on the joint (drive forces handled via augmented mass)
            tau[j] = -wp.dot(S_s, body_f_s) + joint_f[j]

        return


@wp.func
def jcalc_integrate(
    type: int,
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_qdd: wp.array(dtype=float),
    coord_start: int,
    dof_start: int,
    lin_axis_count: int,
    ang_axis_count: int,
    dt: float,
    # outputs
    joint_q_new: wp.array(dtype=float),
    joint_qd_new: wp.array(dtype=float),
):
    if type == JointType.FIXED:
        return

    # prismatic / revolute
    if type == JointType.PRISMATIC or type == JointType.REVOLUTE:
        qdd = joint_qdd[dof_start]
        qd = joint_qd[dof_start]
        q = joint_q[coord_start]

        qd_new = qd + qdd * dt
        q_new = q + qd_new * dt

        joint_qd_new[dof_start] = qd_new
        joint_q_new[coord_start] = q_new

        return

    # ball
    if type == JointType.BALL:
        m_j = wp.vec3(joint_qdd[dof_start + 0], joint_qdd[dof_start + 1], joint_qdd[dof_start + 2])
        w_j = wp.vec3(joint_qd[dof_start + 0], joint_qd[dof_start + 1], joint_qd[dof_start + 2])

        r_j = wp.quat(
            joint_q[coord_start + 0], joint_q[coord_start + 1], joint_q[coord_start + 2], joint_q[coord_start + 3]
        )

        # symplectic Euler
        w_j_new = w_j + m_j * dt

        drdt_j = wp.quat(w_j_new, 0.0) * r_j * 0.5

        # new orientation (normalized)
        r_j_new = wp.normalize(r_j + drdt_j * dt)

        # update joint coords
        joint_q_new[coord_start + 0] = r_j_new[0]
        joint_q_new[coord_start + 1] = r_j_new[1]
        joint_q_new[coord_start + 2] = r_j_new[2]
        joint_q_new[coord_start + 3] = r_j_new[3]

        # update joint vel
        joint_qd_new[dof_start + 0] = w_j_new[0]
        joint_qd_new[dof_start + 1] = w_j_new[1]
        joint_qd_new[dof_start + 2] = w_j_new[2]

        return

    if type == JointType.FREE or type == JointType.DISTANCE:
        a_s = wp.vec3(joint_qdd[dof_start + 0], joint_qdd[dof_start + 1], joint_qdd[dof_start + 2])
        m_s = wp.vec3(joint_qdd[dof_start + 3], joint_qdd[dof_start + 4], joint_qdd[dof_start + 5])

        v_s = wp.vec3(joint_qd[dof_start + 0], joint_qd[dof_start + 1], joint_qd[dof_start + 2])
        w_s = wp.vec3(joint_qd[dof_start + 3], joint_qd[dof_start + 4], joint_qd[dof_start + 5])

        # symplectic Euler
        w_s = w_s + m_s * dt
        v_s = v_s + a_s * dt

        p_s = wp.vec3(joint_q[coord_start + 0], joint_q[coord_start + 1], joint_q[coord_start + 2])

        dpdt_s = v_s + wp.cross(w_s, p_s)
        r_s = wp.quat(
            joint_q[coord_start + 3], joint_q[coord_start + 4], joint_q[coord_start + 5], joint_q[coord_start + 6]
        )

        drdt_s = wp.quat(w_s, 0.0) * r_s * 0.5

        # new orientation (normalized)
        p_s_new = p_s + dpdt_s * dt
        r_s_new = wp.normalize(r_s + drdt_s * dt)

        # update transform
        joint_q_new[coord_start + 0] = p_s_new[0]
        joint_q_new[coord_start + 1] = p_s_new[1]
        joint_q_new[coord_start + 2] = p_s_new[2]

        joint_q_new[coord_start + 3] = r_s_new[0]
        joint_q_new[coord_start + 4] = r_s_new[1]
        joint_q_new[coord_start + 5] = r_s_new[2]
        joint_q_new[coord_start + 6] = r_s_new[3]

        joint_qd_new[dof_start + 0] = v_s[0]
        joint_qd_new[dof_start + 1] = v_s[1]
        joint_qd_new[dof_start + 2] = v_s[2]
        joint_qd_new[dof_start + 3] = w_s[0]
        joint_qd_new[dof_start + 4] = w_s[1]
        joint_qd_new[dof_start + 5] = w_s[2]

        return

    # other joint types (compound, universal, D6)
    if type == JointType.D6:
        axis_count = lin_axis_count + ang_axis_count

        for i in range(axis_count):
            qdd = joint_qdd[dof_start + i]
            qd = joint_qd[dof_start + i]
            q = joint_q[coord_start + i]

            qd_new = qd + qdd * dt
            q_new = q + qd_new * dt

            joint_qd_new[dof_start + i] = qd_new
            joint_q_new[coord_start + i] = q_new

        return


@wp.func
def compute_link_transform(
    i: int,
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_q: wp.array(dtype=float),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    body_X_com: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_dof_dim: wp.array(dtype=int, ndim=2),
    # outputs
    body_q: wp.array(dtype=wp.transform),
    body_q_com: wp.array(dtype=wp.transform),
):
    # parent transform
    parent = joint_parent[i]
    child = joint_child[i]

    # parent transform in spatial coordinates
    X_pj = joint_X_p[i]
    X_cj = joint_X_c[i]
    # parent anchor frame in world space
    X_wpj = X_pj
    if parent >= 0:
        X_wp = body_q[parent]
        X_wpj = X_wp * X_wpj

    type = joint_type[i]
    qd_start = joint_qd_start[i]
    lin_axis_count = joint_dof_dim[i, 0]
    ang_axis_count = joint_dof_dim[i, 1]
    coord_start = joint_q_start[i]

    # compute transform across joint
    X_j = jcalc_transform(type, joint_axis, qd_start, lin_axis_count, ang_axis_count, joint_q, coord_start)

    # transform from world to joint anchor frame at child body
    X_wcj = X_wpj * X_j
    # transform from world to child body frame
    X_wc = X_wcj * wp.transform_inverse(X_cj)

    # compute transform of center of mass
    X_cm = body_X_com[child]
    X_sm = X_wc * X_cm

    # store geometry transforms
    body_q[child] = X_wc
    body_q_com[child] = X_sm


@wp.kernel
def eval_rigid_fk(
    articulation_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_q: wp.array(dtype=float),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    body_X_com: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_dof_dim: wp.array(dtype=int, ndim=2),
    # outputs
    body_q: wp.array(dtype=wp.transform),
    body_q_com: wp.array(dtype=wp.transform),
):
    # one thread per-articulation
    index = wp.tid()

    start = articulation_start[index]
    end = articulation_start[index + 1]

    for i in range(start, end):
        compute_link_transform(
            i,
            joint_type,
            joint_parent,
            joint_child,
            joint_q_start,
            joint_qd_start,
            joint_q,
            joint_X_p,
            joint_X_c,
            body_X_com,
            joint_axis,
            joint_dof_dim,
            body_q,
            body_q_com,
        )


@wp.func
def spatial_cross(a: wp.spatial_vector, b: wp.spatial_vector):
    w_a = wp.spatial_bottom(a)
    v_a = wp.spatial_top(a)

    w_b = wp.spatial_bottom(b)
    v_b = wp.spatial_top(b)

    w = wp.cross(w_a, w_b)
    v = wp.cross(w_a, v_b) + wp.cross(v_a, w_b)

    return wp.spatial_vector(v, w)


@wp.func
def spatial_cross_dual(a: wp.spatial_vector, b: wp.spatial_vector):
    w_a = wp.spatial_bottom(a)
    v_a = wp.spatial_top(a)

    w_b = wp.spatial_bottom(b)
    v_b = wp.spatial_top(b)

    w = wp.cross(w_a, w_b) + wp.cross(v_a, v_b)
    v = wp.cross(w_a, v_b)

    return wp.spatial_vector(v, w)


@wp.func
def dense_index(stride: int, i: int, j: int):
    return i * stride + j


@wp.func
def compute_link_velocity(
    i: int,
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_qd: wp.array(dtype=float),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_dof_dim: wp.array(dtype=int, ndim=2),
    body_I_m: wp.array(dtype=wp.spatial_matrix),
    body_q: wp.array(dtype=wp.transform),
    body_q_com: wp.array(dtype=wp.transform),
    joint_X_p: wp.array(dtype=wp.transform),
    gravity: wp.array(dtype=wp.vec3),
    # outputs
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    body_I_s: wp.array(dtype=wp.spatial_matrix),
    body_v_s: wp.array(dtype=wp.spatial_vector),
    body_f_s: wp.array(dtype=wp.spatial_vector),
    body_a_s: wp.array(dtype=wp.spatial_vector),
):
    type = joint_type[i]
    child = joint_child[i]
    parent = joint_parent[i]
    qd_start = joint_qd_start[i]

    X_pj = joint_X_p[i]
    # X_cj = joint_X_c[i]

    # parent anchor frame in world space
    X_wpj = X_pj
    if parent >= 0:
        X_wp = body_q[parent]
        X_wpj = X_wp * X_wpj

    # compute motion subspace and velocity across the joint (also stores S_s to global memory)
    lin_axis_count = joint_dof_dim[i, 0]
    ang_axis_count = joint_dof_dim[i, 1]
    v_j_s = jcalc_motion(
        type,
        joint_axis,
        lin_axis_count,
        ang_axis_count,
        X_wpj,
        joint_qd,
        qd_start,
        joint_S_s,
    )

    # parent velocity
    v_parent_s = wp.spatial_vector()
    a_parent_s = wp.spatial_vector()

    if parent >= 0:
        v_parent_s = body_v_s[parent]
        a_parent_s = body_a_s[parent]

    # body velocity, acceleration
    v_s = v_parent_s + v_j_s
    a_s = a_parent_s + spatial_cross(v_s, v_j_s)

    # compute body forces
    X_sm = body_q_com[child]
    I_m = body_I_m[child]

    # gravity and external forces (expressed in frame aligned with s but centered at body mass)
    m = I_m[0, 0]

    f_g = m * gravity[0]
    r_com = wp.transform_get_translation(X_sm)
    f_g_s = wp.spatial_vector(f_g, wp.cross(r_com, f_g))

    # body forces
    I_s = transform_spatial_inertia(X_sm, I_m)

    f_b_s = I_s * a_s + spatial_cross_dual(v_s, I_s * v_s)

    body_v_s[child] = v_s
    body_a_s[child] = a_s
    body_f_s[child] = f_b_s - f_g_s
    body_I_s[child] = I_s


# Inverse dynamics via Recursive Newton-Euler algorithm (Featherstone Table 5.1)
@wp.kernel
def eval_rigid_id(
    articulation_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_qd: wp.array(dtype=float),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_dof_dim: wp.array(dtype=int, ndim=2),
    body_I_m: wp.array(dtype=wp.spatial_matrix),
    body_q: wp.array(dtype=wp.transform),
    body_q_com: wp.array(dtype=wp.transform),
    joint_X_p: wp.array(dtype=wp.transform),
    gravity: wp.array(dtype=wp.vec3),
    # outputs
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    body_I_s: wp.array(dtype=wp.spatial_matrix),
    body_v_s: wp.array(dtype=wp.spatial_vector),
    body_f_s: wp.array(dtype=wp.spatial_vector),
    body_a_s: wp.array(dtype=wp.spatial_vector),
):
    # one thread per-articulation
    index = wp.tid()

    start = articulation_start[index]
    end = articulation_start[index + 1]

    # compute link velocities and coriolis forces
    for i in range(start, end):
        compute_link_velocity(
            i,
            joint_type,
            joint_parent,
            joint_child,
            joint_qd_start,
            joint_qd,
            joint_axis,
            joint_dof_dim,
            body_I_m,
            body_q,
            body_q_com,
            joint_X_p,
            gravity,
            joint_S_s,
            body_I_s,
            body_v_s,
            body_f_s,
            body_a_s,
        )


@wp.kernel
def eval_rigid_tau(
    articulation_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_dof_dim: wp.array(dtype=int, ndim=2),
    joint_f: wp.array(dtype=float),
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    body_fb_s: wp.array(dtype=wp.spatial_vector),
    body_f_ext: wp.array(dtype=wp.spatial_vector),
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    # outputs
    body_ft_s: wp.array(dtype=wp.spatial_vector),
    tau: wp.array(dtype=float),
):
    # one thread per-articulation
    index = wp.tid()

    start = articulation_start[index]
    end = articulation_start[index + 1]
    count = end - start

    # compute joint forces
    for offset in range(count):
        # for backwards traversal
        i = end - offset - 1

        type = joint_type[i]
        parent = joint_parent[i]
        child = joint_child[i]
        dof_start = joint_qd_start[i]
        lin_axis_count = joint_dof_dim[i, 0]
        ang_axis_count = joint_dof_dim[i, 1]

        # body forces in Featherstone frame (origin)
        f_b_s = body_fb_s[child]
        f_t_s = body_ft_s[child]

        # external wrench is provided at COM in world frame; shift torque to origin
        f_ext_com = body_f_ext[child]
        f_ext_f = wp.spatial_bottom(f_ext_com)
        f_ext_t = wp.spatial_top(f_ext_com)

        X_wb = body_q[child]
        com_local = body_com[child]
        com_world = wp.transform_point(X_wb, com_local)
        tau_origin = f_ext_f + wp.cross(com_world, f_ext_t)
        f_ext_origin = wp.spatial_vector(f_ext_t, tau_origin)

        # subtract external wrench to get net wrench on body
        f_s = f_b_s + f_t_s - f_ext_origin

        # compute joint-space forces, writes out tau
        jcalc_tau(
            type,
            joint_S_s,
            joint_f,
            dof_start,
            lin_axis_count,
            ang_axis_count,
            f_s,
            tau,
        )

        # update parent forces, todo: check that this is valid for the backwards pass
        if parent >= 0:
            wp.atomic_add(body_ft_s, parent, f_s)


@wp.kernel
def eval_rigid_mass(
    articulation_start: wp.array(dtype=int),
    articulation_M_start: wp.array(dtype=int),
    mass_update_mask: wp.array(dtype=int),
    body_I_s: wp.array(dtype=wp.spatial_matrix),
    # outputs
    M_blocks: wp.array(dtype=float),
):
    # one thread per-articulation
    index = wp.tid()

    if mass_update_mask[index] == 0:
        return

    joint_start = articulation_start[index]
    joint_end = articulation_start[index + 1]
    joint_count = joint_end - joint_start

    M_offset = articulation_M_start[index]

    for l in range(joint_count):
        I = body_I_s[joint_start + l]
        block = M_offset + l * 36
        for row in range(6):
            for col in range(6):
                M_blocks[block + row * 6 + col] = I[row, col]


@wp.kernel
def compute_composite_inertia(
    articulation_start: wp.array(dtype=int),
    mass_update_mask: wp.array(dtype=int),
    joint_ancestor: wp.array(dtype=int),
    body_I_s: wp.array(dtype=wp.spatial_matrix),
    # outputs
    body_I_c: wp.array(dtype=wp.spatial_matrix),
):
    art_idx = wp.tid()

    if mass_update_mask[art_idx] == 0:
        return

    start = articulation_start[art_idx]
    end = articulation_start[art_idx + 1]
    count = end - start

    for i in range(count):
        idx = start + i
        body_I_c[idx] = body_I_s[idx]

    for i in range(count - 1, -1, -1):
        child_idx = start + i
        parent_idx = joint_ancestor[child_idx]

        if parent_idx >= start:
            body_I_c[parent_idx] += body_I_c[child_idx]


@wp.kernel
def crba_fill_flat_par_dof(
    articulation_start: wp.array(dtype=int),
    articulation_dof_start: wp.array(dtype=int),
    articulation_H_start: wp.array(dtype=int),
    articulation_H_rows: wp.array(dtype=int),
    mass_update_mask: wp.array(dtype=int),
    joint_ancestor: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_dof_dim: wp.array(dtype=int, ndim=2),
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    body_I_c: wp.array(dtype=wp.spatial_matrix),
    max_dofs: int,
    # outputs
    H: wp.array(dtype=float),
):
    tid = wp.tid()

    art_idx = tid // max_dofs
    col_idx = tid % max_dofs

    if mass_update_mask[art_idx] == 0:
        return

    num_dofs = articulation_H_rows[art_idx]
    if col_idx >= num_dofs:
        return

    global_dof_start = articulation_dof_start[art_idx]
    target_dof_global = global_dof_start + col_idx

    joint_start = articulation_start[art_idx]
    joint_end = articulation_start[art_idx + 1]

    pivot_joint = int(-1)

    if pivot_joint == -1:
        for j in range(joint_start, joint_end):
            q_start = joint_qd_start[j]
            q_end = joint_qd_start[j + 1]
            if target_dof_global >= q_start and target_dof_global < q_end:
                pivot_joint = j
                break

    if pivot_joint == -1:
        return

    # Compute Force F = I_c[pivot] * S[column]
    S_col = joint_S_s[target_dof_global]
    I_comp = body_I_c[pivot_joint]
    F = I_comp * S_col

    # Walk up the tree and project F onto ancestors
    # H[row, col] = S[row] * F

    curr = pivot_joint
    H_base = articulation_H_start[art_idx]
    stride = num_dofs

    while curr != -1:
        if curr < joint_start:
            break

        q_start = joint_qd_start[curr]
        q_dim = joint_dof_dim[curr]
        count = q_dim[0] + q_dim[1]

        dof_offset_local = q_start - global_dof_start

        for k in range(count):
            row_idx = dof_offset_local + k

            S_row = joint_S_s[q_start + k]
            val = wp.dot(S_row, F)

            H[H_base + row_idx * stride + col_idx] = val
            H[H_base + col_idx * stride + row_idx] = val

        curr = joint_ancestor[curr]


@wp.func
def dense_cholesky(
    n: int,
    A: wp.array(dtype=float),
    R: wp.array(dtype=float),
    A_start: int,
    R_start: int,
    # outputs
    L: wp.array(dtype=float),
):
    # compute the Cholesky factorization of A = L L^T with diagonal regularization R
    for j in range(n):
        s = A[A_start + dense_index(n, j, j)] + R[R_start + j]

        for k in range(j):
            r = L[A_start + dense_index(n, j, k)]
            s -= r * r

        s = wp.sqrt(s)
        invS = 1.0 / s

        L[A_start + dense_index(n, j, j)] = s

        for i in range(j + 1, n):
            s = A[A_start + dense_index(n, i, j)]

            for k in range(j):
                s -= L[A_start + dense_index(n, i, k)] * L[A_start + dense_index(n, j, k)]

            L[A_start + dense_index(n, i, j)] = s * invS


@wp.kernel
def cholesky_flat_loop(
    A_starts: wp.array(dtype=int),
    A_dim: wp.array(dtype=int),
    A: wp.array(dtype=float),
    R: wp.array(dtype=float),
    mass_update_mask: wp.array(dtype=int),
    L: wp.array(dtype=float),
):
    """Flat layout, loop-based Cholesky (one thread per articulation)."""
    batch = wp.tid()

    if mass_update_mask[batch] == 0:
        return

    n = A_dim[batch]
    A_start = A_starts[batch]
    R_start = n * batch

    dense_cholesky(n, A, R, A_start, R_start, L)


@wp.kernel
def cholesky_batched_loop(
    H_group: wp.array3d(dtype=float),  # [n_arts, n_dofs, n_dofs]
    R_group: wp.array2d(dtype=float),  # [n_arts, n_dofs]
    group_to_art: wp.array(dtype=int),
    mass_update_mask: wp.array(dtype=int),
    n_dofs: int,
    # output
    L_group: wp.array3d(dtype=float),  # [n_arts, n_dofs, n_dofs]
):
    """Non-tiled Cholesky for grouped articulation storage.

    One thread per articulation, loop-based Cholesky decomposition.
    Efficient for small articulations where tile overhead dominates.
    """
    group_idx = wp.tid()
    art_idx = group_to_art[group_idx]

    if mass_update_mask[art_idx] == 0:
        return

    # Cholesky decomposition with regularization: L L^T = H + diag(R)
    for j in range(n_dofs):
        # Compute diagonal element L[j,j]
        s = H_group[group_idx, j, j] + R_group[group_idx, j]

        for k in range(j):
            r = L_group[group_idx, j, k]
            s -= r * r

        s = wp.sqrt(s)
        inv_s = 1.0 / s
        L_group[group_idx, j, j] = s

        # Compute off-diagonal elements L[i,j] for i > j
        for i in range(j + 1, n_dofs):
            s = H_group[group_idx, i, j]

            for k in range(j):
                s -= L_group[group_idx, i, k] * L_group[group_idx, j, k]

            L_group[group_idx, i, j] = s * inv_s


@wp.func
def dense_subs(
    n: int,
    L_start: int,
    b_start: int,
    L: wp.array(dtype=float),
    b: wp.array(dtype=float),
    # outputs
    x: wp.array(dtype=float),
):
    # Solves (L L^T) x = b for x given the Cholesky factor L
    # forward substitution solves the lower triangular system L y = b for y
    for i in range(n):
        s = b[b_start + i]

        for j in range(i):
            s -= L[L_start + dense_index(n, i, j)] * x[b_start + j]

        x[b_start + i] = s / L[L_start + dense_index(n, i, i)]

    # backward substitution solves the upper triangular system L^T x = y for x
    for i in range(n - 1, -1, -1):
        s = x[b_start + i]

        for j in range(i + 1, n):
            s -= L[L_start + dense_index(n, j, i)] * x[b_start + j]

        x[b_start + i] = s / L[L_start + dense_index(n, i, i)]


@wp.func
def dense_solve(
    n: int,
    L_start: int,
    b_start: int,
    A: wp.array(dtype=float),
    L: wp.array(dtype=float),
    b: wp.array(dtype=float),
    # outputs
    x: wp.array(dtype=float),
    tmp: wp.array(dtype=float),
):
    # helper function to include tmp argument for backward pass
    dense_subs(n, L_start, b_start, L, b, x)


@wp.kernel
def trisolve_flat_loop(
    L_start: wp.array(dtype=int),
    L_dim: wp.array(dtype=int),
    b_start: wp.array(dtype=int),
    A: wp.array(dtype=float),
    L: wp.array(dtype=float),
    b: wp.array(dtype=float),
    # outputs
    x: wp.array(dtype=float),
    tmp: wp.array(dtype=float),
):
    """Flat layout, loop-based triangular solve (one thread per articulation)."""
    batch = wp.tid()

    dense_solve(L_dim[batch], L_start[batch], b_start[batch], A, L, b, x, tmp)


@wp.kernel
def integrate_generalized_joints(
    joint_type: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_dof_dim: wp.array(dtype=int, ndim=2),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_qdd: wp.array(dtype=float),
    dt: float,
    # outputs
    joint_q_new: wp.array(dtype=float),
    joint_qd_new: wp.array(dtype=float),
):
    # one thread per-articulation
    index = wp.tid()

    type = joint_type[index]
    coord_start = joint_q_start[index]
    dof_start = joint_qd_start[index]
    lin_axis_count = joint_dof_dim[index, 0]
    ang_axis_count = joint_dof_dim[index, 1]

    jcalc_integrate(
        type,
        joint_q,
        joint_qd,
        joint_qdd,
        coord_start,
        dof_start,
        lin_axis_count,
        ang_axis_count,
        dt,
        joint_q_new,
        joint_qd_new,
    )


@wp.kernel
def compute_velocity_predictor(
    joint_qd: wp.array(dtype=float),
    joint_qdd: wp.array(dtype=float),
    dt: float,
    # outputs
    v_hat: wp.array(dtype=float),
):
    tid = wp.tid()
    v_hat[tid] = joint_qd[tid] + joint_qdd[tid] * dt


@wp.kernel
def update_qdd_from_velocity(
    joint_qd: wp.array(dtype=float),
    v_new: wp.array(dtype=float),
    inv_dt: float,
    # outputs
    joint_qdd: wp.array(dtype=float),
):
    tid = wp.tid()
    joint_qdd[tid] = (v_new[tid] - joint_qd[tid]) * inv_dt


@wp.func
def contact_tangent_basis(n: wp.vec3):
    # pick an arbitrary perpendicular vector and orthonormalize
    tangent0 = wp.cross(n, wp.vec3(1.0, 0.0, 0.0))
    if wp.length_sq(tangent0) < 1.0e-12:
        tangent0 = wp.cross(n, wp.vec3(0.0, 1.0, 0.0))
    tangent0 = wp.normalize(tangent0)
    tangent1 = wp.normalize(wp.cross(n, tangent0))
    return tangent0, tangent1


# Computes J*v contribution on the fly by walking the tree
# This keeps the S vectors in L2 cache and avoids reading the large J matrix.
@wp.func
def accumulate_contact_jacobian_matrix_free(
    articulation: int,
    body_index: int,
    weight: float,
    point_world: wp.vec3,
    n_vec: wp.vec3,
    body_to_joint: wp.array(dtype=int),
    body_to_articulation: wp.array(dtype=int),
    joint_ancestor: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    articulation_dof_start: int,
    # Outputs
    row_base_index: int,
    Jc_out: wp.array(dtype=float),
):
    if body_index < 0:
        return

    curr_joint = body_to_joint[body_index]

    while curr_joint != -1:
        dof_start = joint_qd_start[curr_joint]
        dof_end = joint_qd_start[curr_joint + 1]
        dof_count = dof_end - dof_start

        for k in range(dof_count):
            global_dof = dof_start + k

            S = joint_S_s[global_dof]

            linear = wp.vec3(S[0], S[1], S[2])
            angular = wp.vec3(S[3], S[4], S[5])

            lin_vel_at_point = linear + wp.cross(angular, point_world)
            proj = wp.dot(n_vec, lin_vel_at_point)

            local_dof = global_dof - articulation_dof_start

            Jc_out[row_base_index + local_dof] += weight * proj

        curr_joint = joint_ancestor[curr_joint]


@wp.kernel
def build_contact_rows_normal(
    contact_count: wp.array(dtype=int),
    contact_point0: wp.array(dtype=wp.vec3),
    contact_point1: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_shape0: wp.array(dtype=int),
    contact_shape1: wp.array(dtype=int),
    contact_thickness0: wp.array(dtype=float),
    contact_thickness1: wp.array(dtype=float),
    shape_body: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    shape_transform: wp.array(dtype=wp.transform),
    shape_material_mu: wp.array(dtype=float),
    articulation_start: wp.array(dtype=int),
    articulation_H_rows: wp.array(dtype=int),
    articulation_dof_start: wp.array(dtype=int),
    body_to_joint: wp.array(dtype=int),
    body_to_articulation: wp.array(dtype=int),
    joint_ancestor: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    max_constraints: int,
    max_dofs: int,
    contact_beta: float,
    contact_cfm: float,
    enable_friction: int,
    # Outputs
    constraint_counts: wp.array(dtype=int),
    Jc_out: wp.array(dtype=float),
    phi_out: wp.array(dtype=float),
    row_beta: wp.array(dtype=float),
    row_cfm: wp.array(dtype=float),
    row_types: wp.array(dtype=int),
    target_velocity: wp.array(dtype=float),
    row_parent: wp.array(dtype=int),
    row_mu: wp.array(dtype=float),
):
    tid = wp.tid()
    total_contacts = contact_count[0]
    if tid >= total_contacts:
        return

    n = contact_normal[tid]
    shape_a = contact_shape0[tid]
    shape_b = contact_shape1[tid]

    body_a = -1
    body_b = -1
    if shape_a >= 0:
        body_a = shape_body[shape_a]
    if shape_b >= 0:
        body_b = shape_body[shape_b]

    articulation_a = -1
    articulation_b = -1
    if body_a >= 0:
        articulation_a = body_to_articulation[body_a]
    if body_b >= 0:
        articulation_b = body_to_articulation[body_b]

    articulation = articulation_a
    if articulation < 0:
        articulation = articulation_b
    elif articulation_b >= 0 and articulation_b != articulation:
        return
    if articulation < 0:
        return

    thickness_a = contact_thickness0[tid]
    thickness_b = contact_thickness1[tid]
    mu = 0.0
    mat_count = 0
    if shape_a >= 0:
        mu += shape_material_mu[shape_a]
        mat_count += 1
    if shape_b >= 0:
        mu += shape_material_mu[shape_b]
        mat_count += 1
    if mat_count > 0:
        mu /= float(mat_count)

    point_a_local = contact_point0[tid]
    point_b_local = contact_point1[tid]
    point_a_world = wp.vec3(0.0)
    point_b_world = wp.vec3(0.0)

    if body_a >= 0:
        X_wb_a = body_q[body_a]  # World-from-Body transform
        X_bs_a = shape_transform[shape_a]  # Body-from-Shape transform
        X_ws_a = wp.transform_multiply(X_wb_a, X_bs_a)  # World-from-Shape

        point_a_world = wp.transform_point(X_ws_a, point_a_local) - thickness_a * n
    else:
        point_a_world = point_a_local - thickness_a * n

    if body_b >= 0:
        X_wb_b = body_q[body_b]  # World-from-Body transform
        X_bs_b = shape_transform[shape_b]  # Body-from-Shape transform
        X_ws_b = wp.transform_multiply(X_wb_b, X_bs_b)  # World-from-Shape

        point_b_world = wp.transform_point(X_ws_b, point_b_local) + thickness_b * n
    else:
        point_b_world = point_b_local + thickness_b * n

    phi = wp.dot(n, point_a_world - point_b_world)

    # Determine upfront if we'll add friction rows (needed for atomic slot allocation)
    dof_count = articulation_H_rows[articulation]
    will_add_friction = enable_friction != 0 and mu > 0.0 and dof_count > 0

    # Allocate all slots (normal + 2 friction) in a single atomic operation
    # This guarantees contiguous layout: [normal, friction1, friction2]
    slots_needed = 3 if will_add_friction else 1
    base_slot = wp.atomic_add(constraint_counts, articulation, slots_needed)

    # Check for overflow (all slots must fit)
    if base_slot + slots_needed > max_constraints:
        return

    art_dof_start = articulation_dof_start[articulation]

    # --- Normal contact row at base_slot ---
    phi_index = articulation * max_constraints + base_slot
    phi_out[phi_index] = phi
    row_beta[phi_index] = contact_beta
    row_cfm[phi_index] = contact_cfm
    row_types[phi_index] = PGS_CONSTRAINT_TYPE_CONTACT
    target_velocity[phi_index] = 0.0
    row_parent[phi_index] = -1
    row_mu[phi_index] = mu

    row_base = phi_index * max_dofs
    for col in range(max_dofs):
        Jc_out[row_base + col] = 0.0

    accumulate_contact_jacobian_matrix_free(
        articulation,
        body_a,
        1.0,
        point_a_world,
        n,
        body_to_joint,
        body_to_articulation,
        joint_ancestor,
        joint_qd_start,
        joint_S_s,
        art_dof_start,
        row_base,
        Jc_out,
    )

    accumulate_contact_jacobian_matrix_free(
        articulation,
        body_b,
        -1.0,
        point_b_world,
        n,
        body_to_joint,
        body_to_articulation,
        joint_ancestor,
        joint_qd_start,
        joint_S_s,
        art_dof_start,
        row_base,
        Jc_out,
    )

    # --- Friction rows at base_slot + 1 and base_slot + 2 ---
    if will_add_friction:
        t0, t1 = contact_tangent_basis(n)

        # Friction row 1 at base_slot + 1
        row_index_1 = articulation * max_constraints + base_slot + 1
        tangent_base_1 = row_index_1 * max_dofs

        for col in range(max_dofs):
            Jc_out[tangent_base_1 + col] = 0.0

        row_beta[row_index_1] = 0.0
        row_cfm[row_index_1] = contact_cfm
        row_types[row_index_1] = PGS_CONSTRAINT_TYPE_FRICTION
        target_velocity[row_index_1] = 0.0
        phi_out[row_index_1] = 0.0
        row_parent[row_index_1] = phi_index
        row_mu[row_index_1] = mu

        accumulate_contact_jacobian_matrix_free(
            articulation,
            body_a,
            1.0,
            point_a_world,
            t0,
            body_to_joint,
            body_to_articulation,
            joint_ancestor,
            joint_qd_start,
            joint_S_s,
            art_dof_start,
            tangent_base_1,
            Jc_out,
        )

        accumulate_contact_jacobian_matrix_free(
            articulation,
            body_b,
            -1.0,
            point_b_world,
            t0,
            body_to_joint,
            body_to_articulation,
            joint_ancestor,
            joint_qd_start,
            joint_S_s,
            art_dof_start,
            tangent_base_1,
            Jc_out,
        )

        # Friction row 2 at base_slot + 2
        row_index_2 = articulation * max_constraints + base_slot + 2
        tangent_base_2 = row_index_2 * max_dofs

        for col in range(max_dofs):
            Jc_out[tangent_base_2 + col] = 0.0

        row_beta[row_index_2] = 0.0
        row_cfm[row_index_2] = contact_cfm
        row_types[row_index_2] = PGS_CONSTRAINT_TYPE_FRICTION
        target_velocity[row_index_2] = 0.0
        phi_out[row_index_2] = 0.0
        row_parent[row_index_2] = phi_index
        row_mu[row_index_2] = mu

        accumulate_contact_jacobian_matrix_free(
            articulation,
            body_a,
            1.0,
            point_a_world,
            t1,
            body_to_joint,
            body_to_articulation,
            joint_ancestor,
            joint_qd_start,
            joint_S_s,
            art_dof_start,
            tangent_base_2,
            Jc_out,
        )

        accumulate_contact_jacobian_matrix_free(
            articulation,
            body_b,
            -1.0,
            point_b_world,
            t1,
            body_to_joint,
            body_to_articulation,
            joint_ancestor,
            joint_qd_start,
            joint_S_s,
            art_dof_start,
            tangent_base_2,
            Jc_out,
        )


@wp.kernel
def build_augmented_joint_rows(
    articulation_start: wp.array(dtype=int),
    articulation_dof_start: wp.array(dtype=int),
    articulation_H_rows: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_dof_dim: wp.array(dtype=int, ndim=2),
    joint_target_ke: wp.array(dtype=float),
    joint_target_kd: wp.array(dtype=float),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_target_pos: wp.array(dtype=float),
    joint_target_vel: wp.array(dtype=float),
    max_dofs: int,
    dt: float,
    # outputs
    row_counts: wp.array(dtype=int),
    row_dof_index: wp.array(dtype=int),
    row_K: wp.array(dtype=float),
    row_u0: wp.array(dtype=float),
    limit_counts: wp.array(dtype=int),
):
    articulation = wp.tid()
    if max_dofs == 0:
        row_counts[articulation] = 0
        limit_counts[articulation] = 0
        return

    dof_count = articulation_H_rows[articulation]
    if dof_count == 0:
        row_counts[articulation] = 0
        limit_counts[articulation] = 0
        return

    joint_start = articulation_start[articulation]
    joint_end = articulation_start[articulation + 1]

    slot = int(0)
    limit_counts[articulation] = 0

    for joint_index in range(joint_start, joint_end):
        type = joint_type[joint_index]
        if type != JointType.PRISMATIC and type != JointType.REVOLUTE and type != JointType.D6:
            continue

        lin_axis_count = joint_dof_dim[joint_index, 0]
        ang_axis_count = joint_dof_dim[joint_index, 1]
        axis_count = lin_axis_count + ang_axis_count

        qd_start = joint_qd_start[joint_index]
        coord_start = joint_q_start[joint_index]

        for axis in range(axis_count):
            if slot >= max_dofs:
                break
            dof_index = qd_start + axis
            coord_index = coord_start + axis

            ke = joint_target_ke[dof_index]
            kd = joint_target_kd[dof_index]
            if ke <= 0.0 and kd <= 0.0:
                continue

            K = ke * dt * dt + kd * dt
            if K <= 0.0:
                continue

            row_index = articulation * max_dofs + slot
            row_dof_index[row_index] = dof_index
            q = joint_q[coord_index]
            qd_val = joint_qd[dof_index]
            target_pos = joint_target_pos[dof_index]
            target_vel = joint_target_vel[dof_index]
            u0 = -(ke * (q - target_pos + dt * qd_val) + kd * (qd_val - target_vel))
            row_K[row_index] = K
            row_u0[row_index] = u0

            slot += 1
            if slot >= max_dofs:
                break

    row_counts[articulation] = slot
    limit_counts[articulation] = 0


@wp.kernel
def detect_limit_count_changes(
    limit_counts: wp.array(dtype=int),
    prev_limit_counts: wp.array(dtype=int),
    # outputs
    limit_change_mask: wp.array(dtype=int),
):
    tid = wp.tid()
    change = 1 if limit_counts[tid] != prev_limit_counts[tid] else 0
    limit_change_mask[tid] = change


@wp.kernel
def build_mass_update_mask(
    global_flag: int,
    limit_change_mask: wp.array(dtype=int),
    # outputs
    mass_update_mask: wp.array(dtype=int),
):
    tid = wp.tid()
    flag = 1 if global_flag != 0 else 0
    if limit_change_mask[tid] != 0:
        flag = 1
    mass_update_mask[tid] = flag


# =============================================================================
# Multi-Articulation Contact Building Kernels
# =============================================================================
# These kernels enable contacts between multiple articulations within the same
# world. The constraint system becomes world-level instead of per-articulation.


@wp.kernel
def allocate_world_contact_slots(
    contact_count: wp.array(dtype=int),
    contact_shape0: wp.array(dtype=int),
    contact_shape1: wp.array(dtype=int),
    shape_body: wp.array(dtype=int),
    body_to_articulation: wp.array(dtype=int),
    art_to_world: wp.array(dtype=int),
    max_constraints: int,
    enable_friction: int,
    # outputs
    contact_world: wp.array(dtype=int),
    contact_slot: wp.array(dtype=int),
    contact_art_a: wp.array(dtype=int),
    contact_art_b: wp.array(dtype=int),
    world_slot_counter: wp.array(dtype=int),
):
    """
    Phase 1 of multi-articulation contact building.

    Allocates world-level constraint slots for each contact and records
    which articulations are involved.

    Each contact reserves 3 slots (normal + 2 friction) in its world's constraint buffer.
    """
    c = wp.tid()
    total_contacts = contact_count[0]
    if c >= total_contacts:
        contact_slot[c] = -1
        return

    shape_a = contact_shape0[c]
    shape_b = contact_shape1[c]

    # Get bodies and articulations
    body_a = -1
    body_b = -1
    if shape_a >= 0:
        body_a = shape_body[shape_a]
    if shape_b >= 0:
        body_b = shape_body[shape_b]

    art_a = -1
    art_b = -1
    if body_a >= 0:
        art_a = body_to_articulation[body_a]
    if body_b >= 0:
        art_b = body_to_articulation[body_b]

    # Determine world (both bodies must be in same world, or one is ground)
    world = -1
    if art_a >= 0:
        world = art_to_world[art_a]
    if art_b >= 0:
        world_b = art_to_world[art_b]
        if world >= 0 and world_b != world:
            # Cross-world contact - shouldn't happen, skip
            contact_slot[c] = -1
            return
        world = world_b

    if world < 0:
        # No articulation involved (ground-ground?)
        contact_slot[c] = -1
        return

    # Allocate slots (1 normal + 2 friction)
    slots_needed = 1
    if enable_friction != 0:
        slots_needed = 3

    slot = wp.atomic_add(world_slot_counter, world, slots_needed)

    if slot + slots_needed > max_constraints:
        # Overflow - skip this contact
        contact_slot[c] = -1
        return

    contact_world[c] = world
    contact_slot[c] = slot
    contact_art_a[c] = art_a
    contact_art_b[c] = art_b


@wp.func
def accumulate_jacobian_row_world(
    body_index: int,
    sign: float,
    point_world: wp.vec3,
    direction: wp.vec3,
    body_to_joint: wp.array(dtype=int),
    joint_ancestor: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    art_dof_start: int,
    n_dofs: int,
    group_idx: int,
    row: int,
    J_group: wp.array3d(dtype=float),
):
    """Accumulate Jacobian contributions by walking up the kinematic tree."""
    if body_index < 0:
        return

    curr_joint = body_to_joint[body_index]

    while curr_joint >= 0:
        dof_start = joint_qd_start[curr_joint]
        dof_end = joint_qd_start[curr_joint + 1]

        for global_dof in range(dof_start, dof_end):
            S = joint_S_s[global_dof]
            lin = wp.vec3(S[0], S[1], S[2])
            ang = wp.vec3(S[3], S[4], S[5])

            # Velocity at contact point from this joint
            v = lin + wp.cross(ang, point_world)
            proj = wp.dot(direction, v)

            local_dof = global_dof - art_dof_start
            if local_dof >= 0 and local_dof < n_dofs:
                J_group[group_idx, row, local_dof] += sign * proj

        curr_joint = joint_ancestor[curr_joint]


@wp.kernel
def populate_world_J_for_size(
    contact_count: wp.array(dtype=int),
    contact_point0: wp.array(dtype=wp.vec3),
    contact_point1: wp.array(dtype=wp.vec3),
    contact_normal: wp.array(dtype=wp.vec3),
    contact_shape0: wp.array(dtype=int),
    contact_shape1: wp.array(dtype=int),
    contact_thickness0: wp.array(dtype=float),
    contact_thickness1: wp.array(dtype=float),
    contact_world: wp.array(dtype=int),
    contact_slot: wp.array(dtype=int),
    contact_art_a: wp.array(dtype=int),
    contact_art_b: wp.array(dtype=int),
    target_size: int,
    art_size: wp.array(dtype=int),
    art_group_idx: wp.array(dtype=int),
    art_dof_start: wp.array(dtype=int),
    body_to_joint: wp.array(dtype=int),
    joint_ancestor: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    shape_body: wp.array(dtype=int),
    body_q: wp.array(dtype=wp.transform),
    shape_transform: wp.array(dtype=wp.transform),
    shape_material_mu: wp.array(dtype=float),
    enable_friction: int,
    pgs_beta: float,
    pgs_cfm: float,
    # outputs
    J_group: wp.array3d(dtype=float),
    world_row_type: wp.array2d(dtype=int),
    world_row_parent: wp.array2d(dtype=int),
    world_row_mu: wp.array2d(dtype=float),
    world_row_beta: wp.array2d(dtype=float),
    world_row_cfm: wp.array2d(dtype=float),
    world_phi: wp.array2d(dtype=float),
    world_target_velocity: wp.array2d(dtype=float),
):
    """
    Phase 2 of multi-articulation contact building (per size group).

    Populates the Jacobian matrix for articulations of a specific DOF size.
    Each contact may contribute to multiple articulations' J matrices.
    """
    c = wp.tid()
    total_contacts = contact_count[0]
    if c >= total_contacts:
        return

    slot = contact_slot[c]
    if slot < 0:
        return

    world = contact_world[c]
    art_a = contact_art_a[c]
    art_b = contact_art_b[c]

    # Get contact geometry
    normal = contact_normal[c]
    shape_a = contact_shape0[c]
    shape_b = contact_shape1[c]

    body_a = -1
    body_b = -1
    if shape_a >= 0:
        body_a = shape_body[shape_a]
    if shape_b >= 0:
        body_b = shape_body[shape_b]

    thickness_a = contact_thickness0[c]
    thickness_b = contact_thickness1[c]

    # Compute contact points in world frame
    point_a_local = contact_point0[c]
    point_b_local = contact_point1[c]
    point_a_world = wp.vec3(0.0)
    point_b_world = wp.vec3(0.0)

    if body_a >= 0:
        X_wb_a = body_q[body_a]
        X_bs_a = shape_transform[shape_a]
        X_ws_a = wp.transform_multiply(X_wb_a, X_bs_a)
        point_a_world = wp.transform_point(X_ws_a, point_a_local) - thickness_a * normal
    else:
        point_a_world = point_a_local - thickness_a * normal

    if body_b >= 0:
        X_wb_b = body_q[body_b]
        X_bs_b = shape_transform[shape_b]
        X_ws_b = wp.transform_multiply(X_wb_b, X_bs_b)
        point_b_world = wp.transform_point(X_ws_b, point_b_local) + thickness_b * normal
    else:
        point_b_world = point_b_local + thickness_b * normal

    # Compute penetration depth
    phi = wp.dot(normal, point_a_world - point_b_world)

    # Compute friction coefficient
    mu = 0.0
    mat_count = 0
    if shape_a >= 0:
        mu += shape_material_mu[shape_a]
        mat_count += 1
    if shape_b >= 0:
        mu += shape_material_mu[shape_b]
        mat_count += 1
    if mat_count > 0:
        mu /= float(mat_count)

    # Compute tangent basis for friction
    t0, t1 = contact_tangent_basis(normal)

    # Handle articulation A if it matches target size
    if art_a >= 0 and art_size[art_a] == target_size:
        group_idx_a = art_group_idx[art_a]
        dof_start_a = art_dof_start[art_a]

        # Normal row (slot + 0)
        accumulate_jacobian_row_world(
            body_a,
            1.0,
            point_a_world,
            normal,
            body_to_joint,
            joint_ancestor,
            joint_qd_start,
            joint_S_s,
            dof_start_a,
            target_size,
            group_idx_a,
            slot,
            J_group,
        )

        if enable_friction != 0:
            # Friction row 1 (slot + 1)
            accumulate_jacobian_row_world(
                body_a,
                1.0,
                point_a_world,
                t0,
                body_to_joint,
                joint_ancestor,
                joint_qd_start,
                joint_S_s,
                dof_start_a,
                target_size,
                group_idx_a,
                slot + 1,
                J_group,
            )
            # Friction row 2 (slot + 2)
            accumulate_jacobian_row_world(
                body_a,
                1.0,
                point_a_world,
                t1,
                body_to_joint,
                joint_ancestor,
                joint_qd_start,
                joint_S_s,
                dof_start_a,
                target_size,
                group_idx_a,
                slot + 2,
                J_group,
            )

    # Handle articulation B if it matches target size
    if art_b >= 0 and art_size[art_b] == target_size:
        group_idx_b = art_group_idx[art_b]
        dof_start_b = art_dof_start[art_b]

        # Opposite sign for body B
        accumulate_jacobian_row_world(
            body_b,
            -1.0,
            point_b_world,
            normal,
            body_to_joint,
            joint_ancestor,
            joint_qd_start,
            joint_S_s,
            dof_start_b,
            target_size,
            group_idx_b,
            slot,
            J_group,
        )

        if enable_friction != 0:
            accumulate_jacobian_row_world(
                body_b,
                -1.0,
                point_b_world,
                t0,
                body_to_joint,
                joint_ancestor,
                joint_qd_start,
                joint_S_s,
                dof_start_b,
                target_size,
                group_idx_b,
                slot + 1,
                J_group,
            )
            accumulate_jacobian_row_world(
                body_b,
                -1.0,
                point_b_world,
                t1,
                body_to_joint,
                joint_ancestor,
                joint_qd_start,
                joint_S_s,
                dof_start_b,
                target_size,
                group_idx_b,
                slot + 2,
                J_group,
            )

    # Set row metadata (only once per contact, from whichever articulation runs first)
    # Use art_a preferentially to avoid double-writes
    if art_a >= 0 and art_size[art_a] == target_size:
        # Normal contact row
        world_row_type[world, slot] = PGS_CONSTRAINT_TYPE_CONTACT
        world_row_parent[world, slot] = -1
        world_row_mu[world, slot] = mu
        world_row_beta[world, slot] = pgs_beta
        world_row_cfm[world, slot] = pgs_cfm
        world_phi[world, slot] = phi
        world_target_velocity[world, slot] = 0.0

        if enable_friction != 0:
            # Friction row 1
            world_row_type[world, slot + 1] = PGS_CONSTRAINT_TYPE_FRICTION
            world_row_parent[world, slot + 1] = slot
            world_row_mu[world, slot + 1] = mu
            world_row_beta[world, slot + 1] = 0.0
            world_row_cfm[world, slot + 1] = pgs_cfm
            world_phi[world, slot + 1] = 0.0
            world_target_velocity[world, slot + 1] = 0.0

            # Friction row 2
            world_row_type[world, slot + 2] = PGS_CONSTRAINT_TYPE_FRICTION
            world_row_parent[world, slot + 2] = slot
            world_row_mu[world, slot + 2] = mu
            world_row_beta[world, slot + 2] = 0.0
            world_row_cfm[world, slot + 2] = pgs_cfm
            world_phi[world, slot + 2] = 0.0
            world_target_velocity[world, slot + 2] = 0.0

    elif art_b >= 0 and art_size[art_b] == target_size:
        # Only write metadata from art_b if art_a didn't match this size
        world_row_type[world, slot] = PGS_CONSTRAINT_TYPE_CONTACT
        world_row_parent[world, slot] = -1
        world_row_mu[world, slot] = mu
        world_row_beta[world, slot] = pgs_beta
        world_row_cfm[world, slot] = pgs_cfm
        world_phi[world, slot] = phi
        world_target_velocity[world, slot] = 0.0

        if enable_friction != 0:
            world_row_type[world, slot + 1] = PGS_CONSTRAINT_TYPE_FRICTION
            world_row_parent[world, slot + 1] = slot
            world_row_mu[world, slot + 1] = mu
            world_row_beta[world, slot + 1] = 0.0
            world_row_cfm[world, slot + 1] = pgs_cfm
            world_phi[world, slot + 1] = 0.0
            world_target_velocity[world, slot + 1] = 0.0

            world_row_type[world, slot + 2] = PGS_CONSTRAINT_TYPE_FRICTION
            world_row_parent[world, slot + 2] = slot
            world_row_mu[world, slot + 2] = mu
            world_row_beta[world, slot + 2] = 0.0
            world_row_cfm[world, slot + 2] = pgs_cfm
            world_phi[world, slot + 2] = 0.0
            world_target_velocity[world, slot + 2] = 0.0


@wp.kernel
def finalize_world_constraint_counts(
    world_slot_counter: wp.array(dtype=int),
    max_constraints: int,
    # outputs
    world_constraint_count: wp.array(dtype=int),
):
    """Copy and clamp the slot counter to constraint counts."""
    world = wp.tid()
    count = world_slot_counter[world]
    if count > max_constraints:
        count = max_constraints
    world_constraint_count[world] = count


@wp.kernel
def clamp_contact_counts(
    constraint_counts: wp.array(dtype=int),
    max_constraints: int,
):
    articulation = wp.tid()
    count = constraint_counts[articulation]
    if count > max_constraints:
        constraint_counts[articulation] = max_constraints


@wp.kernel
def apply_augmented_mass_diagonal(
    articulation_H_start: wp.array(dtype=int),
    articulation_H_rows: wp.array(dtype=int),
    articulation_dof_start: wp.array(dtype=int),
    max_dofs: int,
    mass_update_mask: wp.array(dtype=int),
    row_counts: wp.array(dtype=int),
    row_dof_index: wp.array(dtype=int),
    row_K: wp.array(dtype=float),
    # outputs
    H: wp.array(dtype=float),
):
    articulation = wp.tid()
    if mass_update_mask[articulation] == 0:
        return

    n = articulation_H_rows[articulation]
    if n == 0 or max_dofs == 0:
        return

    count = row_counts[articulation]
    if count == 0:
        return

    H_start = articulation_H_start[articulation]
    dof_start = articulation_dof_start[articulation]

    for i in range(count):
        row_index = articulation * max_dofs + i
        dof = row_dof_index[row_index]
        local = dof - dof_start
        if local < 0 or local >= n:
            continue

        K = row_K[row_index]
        if K <= 0.0:
            continue

        diag_index = H_start + dense_index(n, local, local)
        H[diag_index] += K


@wp.kernel
def apply_augmented_mass_diagonal_grouped(
    group_to_art: wp.array(dtype=int),
    articulation_dof_start: wp.array(dtype=int),
    n_dofs: int,
    max_dofs: int,
    mass_update_mask: wp.array(dtype=int),
    row_counts: wp.array(dtype=int),
    row_dof_index: wp.array(dtype=int),
    row_K: wp.array(dtype=float),
    # outputs
    H_group: wp.array3d(dtype=float),  # [n_arts, n_dofs, n_dofs]
):
    """Apply augmented mass diagonal for grouped H storage."""
    idx = wp.tid()
    articulation = group_to_art[idx]

    if mass_update_mask[articulation] == 0:
        return

    count = row_counts[articulation]
    if count == 0:
        return

    dof_start = articulation_dof_start[articulation]

    for i in range(count):
        row_index = articulation * max_dofs + i
        dof = row_dof_index[row_index]
        local = dof - dof_start
        if local < 0 or local >= n_dofs:
            continue

        K = row_K[row_index]
        if K <= 0.0:
            continue

        H_group[idx, local, local] += K


@wp.kernel
def apply_augmented_joint_tau(
    max_dofs: int,
    row_counts: wp.array(dtype=int),
    row_dof_index: wp.array(dtype=int),
    row_u0: wp.array(dtype=float),
    # outputs
    joint_tau: wp.array(dtype=float),
):
    articulation = wp.tid()
    if max_dofs == 0:
        return

    count = row_counts[articulation]
    if count == 0:
        return

    for i in range(count):
        row_index = articulation * max_dofs + i
        dof = row_dof_index[row_index]
        u0 = row_u0[row_index]
        if u0 == 0.0:
            continue

        wp.atomic_add(joint_tau, dof, u0)


@wp.kernel
def hinv_jt_flat_par_row(
    articulation_H_start: wp.array(dtype=int),
    articulation_H_rows: wp.array(dtype=int),
    max_constraints: int,
    max_dofs: int,
    constraint_counts: wp.array(dtype=int),
    L: wp.array(dtype=float),
    J_rows: wp.array(dtype=float),
    # outputs
    Y_rows: wp.array(dtype=float),
):
    """Flat layout, par-row H^-1 J^T (one thread per (articulation, constraint))."""
    # One thread per (articulation, local_constraint)
    tid = wp.tid()

    articulation = tid // max_constraints
    local_constraint = tid - articulation * max_constraints  # tid % max_constraints

    m = constraint_counts[articulation]
    n = articulation_H_rows[articulation]

    # Nothing to do if no dofs or this constraint index is unused
    if n == 0 or local_constraint >= m:
        return

    L_start = articulation_H_start[articulation]
    stride = n

    # Base index into the flattened [articulation, constraint, dof] buffer
    row_base = (articulation * max_constraints + local_constraint) * max_dofs

    # ----------------------------------------------------------------------
    # Forward substitution: solve L * y = J_row
    # ----------------------------------------------------------------------
    for row in range(n):
        s = J_rows[row_base + row]

        # subtract contributions from previously solved entries
        for k in range(row):
            s -= L[L_start + dense_index(stride, row, k)] * Y_rows[row_base + k]

        diag = L[L_start + dense_index(stride, row, row)]
        if diag != 0.0:
            Y_rows[row_base + row] = s / diag
        else:
            Y_rows[row_base + row] = 0.0

    # ----------------------------------------------------------------------
    # Backward substitution: solve L^T * x = y
    # ----------------------------------------------------------------------
    for row in range(n - 1, -1, -1):
        s = Y_rows[row_base + row]

        for k in range(row + 1, n):
            s -= L[L_start + dense_index(stride, k, row)] * Y_rows[row_base + k]

        diag = L[L_start + dense_index(stride, row, row)]
        if diag != 0.0:
            Y_rows[row_base + row] = s / diag
        else:
            Y_rows[row_base + row] = 0.0

    # Zero-pad up to max_dofs for safety / consistency
    for row in range(n, max_dofs):
        Y_rows[row_base + row] = 0.0


@wp.kernel
def delassus_flat_par_row(
    articulation_H_rows: wp.array(dtype=int),
    max_constraints: int,
    max_dofs: int,
    constraint_counts: wp.array(dtype=int),
    J_rows: wp.array(dtype=float),
    Y_rows: wp.array(dtype=float),
    row_cfm: wp.array(dtype=float),
    # outputs
    diag_out: wp.array(dtype=float),
    matrix_out: wp.array(dtype=float),
):
    """Flat layout, par-row Delassus accumulation (one thread per row)."""
    idx = wp.tid()
    articulation = idx // max_constraints
    i_row = idx % max_constraints

    m = constraint_counts[articulation]
    n = articulation_H_rows[articulation]
    if i_row >= m or n == 0:
        return

    diag_base = articulation * max_constraints
    mat_base = articulation * max_constraints * max_constraints

    row_i = (articulation * max_constraints + i_row) * max_dofs

    # diag
    diag_val = float(0.0)
    for k in range(n):
        diag_val += J_rows[row_i + k] * Y_rows[row_i + k]
    cfm = row_cfm[diag_base + i_row]
    diag_out[diag_base + i_row] = diag_val + cfm

    # off-diagonals
    for j in range(m):
        row_j = (articulation * max_constraints + j) * max_dofs
        s = float(0.0)
        for k in range(n):
            s += J_rows[row_i + k] * Y_rows[row_j + k]
        matrix_out[mat_base + i_row * max_constraints + j] = s


@wp.kernel
def contact_bias_flat_par_row(
    articulation_dof_start: wp.array(dtype=int),
    articulation_H_rows: wp.array(dtype=int),
    constraint_counts: wp.array(dtype=int),
    articulation_count: int,
    max_constraints: int,
    max_dofs: int,
    J_rows: wp.array(dtype=float),
    v_hat: wp.array(dtype=float),
    phi: wp.array(dtype=float),
    row_beta: wp.array(dtype=float),
    row_types: wp.array(dtype=int),
    target_velocity: wp.array(dtype=float),
    dt: float,
    # outputs
    rhs_out: wp.array(dtype=float),
):
    """Parallelized contact bias computation - one thread per constraint slot.

    Launched with dim = articulation_count * max_constraints.
    Each thread computes J*v_hat for one constraint and the corresponding RHS.
    """
    tid = wp.tid()
    articulation = tid // max_constraints
    constraint_idx = tid % max_constraints

    # Bounds check
    if articulation >= articulation_count:
        return

    m = constraint_counts[articulation]
    if constraint_idx >= m:
        return

    n = articulation_H_rows[articulation]
    if n == 0:
        return

    dof_start = articulation_dof_start[articulation]
    row_base = tid * max_dofs  # = (articulation * max_constraints + constraint_idx) * max_dofs
    rhs_idx = articulation * max_constraints + constraint_idx

    # Compute J * v_hat (dot product)
    rel_vel = float(0.0)
    for k in range(n):
        rel_vel += J_rows[row_base + k] * v_hat[dof_start + k]

    # Load constraint parameters
    gap = phi[rhs_idx]
    beta_val = row_beta[rhs_idx]
    constraint_type = row_types[rhs_idx]
    target_vel = target_velocity[rhs_idx]

    # Compute RHS based on constraint type
    if constraint_type == PGS_CONSTRAINT_TYPE_CONTACT:
        rhs = rel_vel - target_vel
        if gap < 0.0:
            rhs += beta_val * gap / dt
        rhs_out[rhs_idx] = rhs
    elif constraint_type == PGS_CONSTRAINT_TYPE_FRICTION:
        rhs_out[rhs_idx] = rel_vel - target_vel
    elif constraint_type == PGS_CONSTRAINT_TYPE_JOINT_TARGET:
        correction = beta_val * gap / dt
        rhs_out[rhs_idx] = rel_vel - target_vel + correction
    else:
        rhs_out[rhs_idx] = rel_vel


@wp.kernel
def prepare_impulses(
    constraint_counts: wp.array(dtype=int),
    max_constraints: int,
    warmstart: int,
    # outputs
    impulses: wp.array(dtype=float),
):
    articulation = wp.tid()
    m = constraint_counts[articulation]
    base = articulation * max_constraints

    for i in range(max_constraints):
        if warmstart == 0 or i >= m:
            impulses[base + i] = 0.0


@wp.kernel
def apply_impulses_flat_par_dof(
    articulation_dof_start: wp.array(dtype=int),
    articulation_H_rows: wp.array(dtype=int),
    constraint_counts: wp.array(dtype=int),
    max_constraints: int,
    max_dofs: int,
    Y_rows: wp.array(dtype=float),
    v_hat: wp.array(dtype=float),
    impulses: wp.array(dtype=float),
    # outputs
    v_out: wp.array(dtype=float),
):
    tid = wp.tid()

    articulation = tid // max_dofs
    local_dof = tid % max_dofs

    num_dofs = articulation_H_rows[articulation]
    if local_dof >= num_dofs:
        return

    num_constraints = constraint_counts[articulation]
    dof_start = articulation_dof_start[articulation]

    delta_v = float(0.0)

    for i in range(num_constraints):
        row_start = (articulation * max_constraints + i) * max_dofs
        y_val = Y_rows[row_start + local_dof]

        impulse_val = impulses[articulation * max_constraints + i]

        delta_v += y_val * impulse_val

    v_out[dof_start + local_dof] = v_hat[dof_start + local_dof] + delta_v


@wp.kernel
def clamp_joint_tau(
    joint_tau: wp.array(dtype=float),
    joint_effort_limit: wp.array(dtype=float),
):
    tid = wp.tid()

    # Per-DoF effort limit (same convention as MuJoCo actuators)
    limit = joint_effort_limit[tid]

    # If limit <= 0, treat as unlimited
    if limit <= 0.0:
        return

    t = joint_tau[tid]

    if t > limit:
        t = limit
    elif t < -limit:
        t = -limit

    joint_tau[tid] = t


# --- Tile configuration for contact system build ---
# Kernel naming: {op}_{layout}_{parallelism}
# layout: flat | batched, parallelism: tiled | loop | par_row | par_row_col | par_dof

# Max generalized dofs per articulation we support in the tiled path.
# joint_dof_count per articulation must be <= TILE_DOF or we use fall back
TILE_DOF = wp.constant(49)

# Max constraints per articulation we support in the tiled path.
# pgs_max_constraints must be <= TILE_CONSTRAINTS or we use fall back
TILE_CONSTRAINTS = wp.constant(128)

# Threads per tile/block for tile kernels
TILE_THREADS = 64


@wp.kernel
def update_body_qd_from_featherstone(
    body_v_s: wp.array(dtype=wp.spatial_vector),
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    body_qd_out: wp.array(dtype=wp.spatial_vector),
):
    tid = wp.tid()

    twist = body_v_s[tid]  # spatial twist about origin
    v0 = wp.spatial_top(twist)
    w = wp.spatial_bottom(twist)

    X_wb = body_q[tid]
    com_local = body_com[tid]
    com_world = wp.transform_point(X_wb, com_local)

    v_com = v0 + wp.cross(w, com_world)

    body_qd_out[tid] = wp.spatial_vector(v_com, w)


# =============================================================================
# World-Level PGS and Velocity Kernels for Multi-Articulation
# =============================================================================


@wp.kernel
def compute_world_contact_bias(
    world_constraint_count: wp.array(dtype=int),
    max_constraints: int,
    world_phi: wp.array2d(dtype=float),
    world_row_beta: wp.array2d(dtype=float),
    world_row_type: wp.array2d(dtype=int),
    world_target_velocity: wp.array2d(dtype=float),
    dt: float,
    # outputs
    world_rhs: wp.array2d(dtype=float),
):
    """Compute the RHS bias term for world-level PGS solve.

    The RHS follows the convention: rhs = J*v + stabilization
    For contacts with penetration (phi < 0): rhs = J*v + beta * phi / dt (negative)
    This leads to positive impulses when resolved by PGS.
    """
    world = wp.tid()
    m = world_constraint_count[world]

    inv_dt = 1.0 / dt

    for i in range(m):
        phi = world_phi[world, i]
        beta = world_row_beta[world, i]
        row_type = world_row_type[world, i]
        target_vel = world_target_velocity[world, i]

        # Initialize with -target_velocity (will add J*v later)
        rhs = -target_vel

        # For contacts: add Baumgarte stabilization when penetrating
        if row_type == PGS_CONSTRAINT_TYPE_CONTACT:
            if phi < 0.0:
                rhs += beta * phi * inv_dt  # Negative for penetration
        elif row_type == PGS_CONSTRAINT_TYPE_JOINT_TARGET:
            rhs += beta * phi * inv_dt

        world_rhs[world, i] = rhs


@wp.kernel
def rhs_accum_world_par_art(
    world_constraint_count: wp.array(dtype=int),
    max_constraints: int,
    art_to_world: wp.array(dtype=int),
    art_size: wp.array(dtype=int),
    art_group_idx: wp.array(dtype=int),
    art_dof_start: wp.array(dtype=int),
    v_hat: wp.array(dtype=float),
    group_to_art: wp.array(dtype=int),
    J_group: wp.array3d(dtype=float),
    n_dofs: int,
    # outputs
    world_rhs: wp.array2d(dtype=float),
):
    """
    Accumulate J*v_hat into world RHS for a single size group.

    RHS = J*v + stabilization (already includes stabilization from compute_world_contact_bias)
    This kernel is launched once per size group to accumulate velocity contributions.
    """
    idx = wp.tid()
    art = group_to_art[idx]
    world = art_to_world[art]
    n_constraints = world_constraint_count[world]

    if n_constraints == 0:
        return

    dof_start = art_dof_start[art]

    for c in range(n_constraints):
        jv = float(0.0)
        for d in range(n_dofs):
            jv += J_group[idx, c, d] * v_hat[dof_start + d]
        wp.atomic_add(world_rhs, world, c, jv)  # Add J*v (positive)


@wp.kernel
def prepare_world_impulses(
    world_constraint_count: wp.array(dtype=int),
    max_constraints: int,
    warmstart: int,
    # in/out
    world_impulses: wp.array2d(dtype=float),
):
    """Initialize world impulses (zero or warmstart)."""
    world = wp.tid()
    m = world_constraint_count[world]

    for i in range(max_constraints):
        if warmstart == 0 or i >= m:
            world_impulses[world, i] = 0.0


@wp.kernel
def pgs_solve_loop(
    world_constraint_count: wp.array(dtype=int),
    max_constraints: int,
    world_diag: wp.array2d(dtype=float),
    world_C: wp.array3d(dtype=float),
    world_rhs: wp.array2d(dtype=float),
    world_impulses: wp.array2d(dtype=float),
    iterations: int,
    omega: float,
    world_row_type: wp.array2d(dtype=int),
    world_row_parent: wp.array2d(dtype=int),
    world_row_mu: wp.array2d(dtype=float),
):
    """
    World-level Projected Gauss-Seidel solver.

    Similar to pgs_solve_contacts but operates on 2D world-indexed arrays.
    """
    world = wp.tid()
    m = world_constraint_count[world]

    if m == 0:
        return

    for _ in range(iterations):
        for i in range(m):
            # Compute residual: w = rhs_i + sum_j C_ij * lambda_j
            w = world_rhs[world, i]
            for j in range(m):
                w += world_C[world, i, j] * world_impulses[world, j]

            denom = world_diag[world, i]
            if denom <= 0.0:
                continue

            delta = -w / denom
            new_impulse = world_impulses[world, i] + omega * delta
            row_type = world_row_type[world, i]

            # --- Normal contact: lambda_n >= 0 ---
            if row_type == PGS_CONSTRAINT_TYPE_CONTACT:
                if new_impulse < 0.0:
                    new_impulse = 0.0
                world_impulses[world, i] = new_impulse

            # --- Friction: isotropic Coulomb ---
            elif row_type == PGS_CONSTRAINT_TYPE_FRICTION:
                parent_idx = world_row_parent[world, i]
                lambda_n = world_impulses[world, parent_idx]
                mu = world_row_mu[world, i]
                radius = wp.max(mu * lambda_n, 0.0)

                if radius <= 0.0:
                    world_impulses[world, i] = 0.0
                    continue

                world_impulses[world, i] = new_impulse

                # Sibling friction row: constraints are laid out as [normal, friction1, friction2]
                # so friction rows are at parent_idx+1 and parent_idx+2
                if i == parent_idx + 1:
                    sib = parent_idx + 2
                else:
                    sib = parent_idx + 1

                # Project tangent impulses onto friction disk
                a = world_impulses[world, i]
                b = world_impulses[world, sib]

                mag = wp.sqrt(a * a + b * b)
                if mag > radius:
                    scale = radius / mag
                    world_impulses[world, i] = a * scale
                    world_impulses[world, sib] = b * scale

            else:
                world_impulses[world, i] = new_impulse


@wp.kernel
def apply_impulses_world_par_dof(
    group_to_art: wp.array(dtype=int),
    art_to_world: wp.array(dtype=int),
    art_dof_start: wp.array(dtype=int),
    n_dofs: int,
    n_arts: int,
    world_constraint_count: wp.array(dtype=int),
    max_constraints: int,
    Y_group: wp.array3d(dtype=float),
    world_impulses: wp.array2d(dtype=float),
    v_hat: wp.array(dtype=float),
    # outputs
    v_out: wp.array(dtype=float),
):
    """
    Accumulate velocity changes from world impulses for a single size group.
    DOF-parallelized: each thread handles one (articulation, DOF) pair.

    v_out = v_hat + Y * impulses
    """
    tid = wp.tid()

    # Decode thread index
    local_dof = tid % n_dofs
    idx = tid // n_dofs  # group index

    if idx >= n_arts:
        return

    art = group_to_art[idx]
    world = art_to_world[art]
    n_constraints = world_constraint_count[world]
    dof_start = art_dof_start[art]

    # Inner loop only over constraints
    delta_v = float(0.0)
    for c in range(n_constraints):
        delta_v += Y_group[idx, c, local_dof] * world_impulses[world, c]

    global_dof = dof_start + local_dof
    v_out[global_dof] = v_hat[global_dof] + delta_v


@wp.kernel
def finalize_world_diag_cfm(
    world_constraint_count: wp.array(dtype=int),
    world_row_cfm: wp.array2d(dtype=float),
    # in/out
    world_diag: wp.array2d(dtype=float),
):
    """Add CFM to world diagonal after Delassus accumulation."""
    world = wp.tid()
    m = world_constraint_count[world]

    for i in range(m):
        world_diag[world, i] += world_row_cfm[world, i]


@wp.kernel
def zero_world_C_and_diag(
    world_count: int,
    max_constraints: int,
    # outputs
    world_C: wp.array3d(dtype=float),
    world_diag: wp.array2d(dtype=float),
):
    """Zero world Delassus matrices before accumulation."""
    world = wp.tid()
    for i in range(max_constraints):
        world_diag[world, i] = 0.0
        for j in range(max_constraints):
            world_C[world, i, j] = 0.0


# =============================================================================
# Parallelized Non-Tiled Kernels for Heterogeneous Multi-Articulation
# =============================================================================
# These kernels parallelize across constraints (and constraint pairs) to achieve
# much better GPU utilization than the single-thread-per-articulation versions.


@wp.kernel
def hinv_jt_batched_par_row(
    # Grouped Cholesky factor storage [n_arts, n_dofs, n_dofs]
    L_group: wp.array3d(dtype=float),
    # Size-grouped Jacobian [n_arts_of_size, max_constraints, n_dofs]
    J_group: wp.array3d(dtype=float),
    # Indirection arrays
    group_to_art: wp.array(dtype=int),
    art_to_world: wp.array(dtype=int),
    world_constraint_count: wp.array(dtype=int),
    # Size parameters
    n_dofs: int,
    max_constraints: int,
    n_arts: int,
    # Output: Y = H^-1 * J^T [n_arts_of_size, max_constraints, n_dofs]
    Y_group: wp.array3d(dtype=float),
):
    """
    Compute Y = H^-1 * J^T for one size group using forward/backward substitution.

    GROUPED STORAGE VERSION: Uses L_group (3D array) instead of flat L.
    Efficient for small articulations where tile overhead dominates.

    Each thread handles one (articulation, constraint) pair.

    For each articulation in the group, solves:
        L * L^T * Y = J^T
    Using:
        1. Forward substitution: L * Z = J^T
        2. Backward substitution: L^T * Y = Z

    Thread dimension: n_arts_of_size * max_constraints
    """
    tid = wp.tid()

    # Decode thread index
    c = tid % max_constraints  # constraint index
    idx = tid // max_constraints  # group index (articulation within size group)

    # Bounds check for articulation
    if idx >= n_arts:
        return

    art = group_to_art[idx]
    world = art_to_world[art]
    n_constraints = world_constraint_count[world]

    # Early exit if this constraint is beyond the actual count
    if c >= n_constraints:
        return

    # ----------------------------------------------------------------
    # Forward substitution: L * z = j
    # L is lower triangular, so solve from top to bottom
    # ----------------------------------------------------------------
    for i in range(n_dofs):
        # z[i] = (j[i] - sum_{k<i} L[i,k] * z[k]) / L[i,i]
        val = J_group[idx, c, i]

        for k in range(i):
            # z[k] is stored in Y_group temporarily
            val -= L_group[idx, i, k] * Y_group[idx, c, k]

        L_ii = L_group[idx, i, i]
        if L_ii != 0.0:
            Y_group[idx, c, i] = val / L_ii
        else:
            Y_group[idx, c, i] = 0.0

    # ----------------------------------------------------------------
    # Backward substitution: L^T * y = z
    # L^T is upper triangular, so solve from bottom to top
    # z is currently stored in Y_group, we overwrite with y
    # ----------------------------------------------------------------
    for i_rev in range(n_dofs):
        i = n_dofs - 1 - i_rev

        # y[i] = (z[i] - sum_{k>i} L[k,i] * y[k]) / L[i,i]
        # Note: L^T[i,k] = L[k,i], so we read L[k,i] for k > i
        val = Y_group[idx, c, i]  # This is z[i] from forward pass

        for k in range(i + 1, n_dofs):
            val -= L_group[idx, k, i] * Y_group[idx, c, k]

        L_ii = L_group[idx, i, i]
        if L_ii != 0.0:
            Y_group[idx, c, i] = val / L_ii
        else:
            Y_group[idx, c, i] = 0.0


@wp.kernel
def delassus_batched_par_row_col(
    # Size-grouped arrays
    J_group: wp.array3d(dtype=float),  # [n_arts_of_size, max_constraints, n_dofs]
    Y_group: wp.array3d(dtype=float),  # [n_arts_of_size, max_constraints, n_dofs]
    # Indirection arrays
    group_to_art: wp.array(dtype=int),
    art_to_world: wp.array(dtype=int),
    world_constraint_count: wp.array(dtype=int),
    # Size parameters
    n_dofs: int,
    max_constraints: int,
    n_arts: int,
    # Output: Delassus matrix C and diagonal (accumulated via atomics)
    world_C: wp.array3d(dtype=float),  # [world_count, max_constraints, max_constraints]
    world_diag: wp.array2d(dtype=float),  # [world_count, max_constraints]
):
    """
    Accumulate Delassus matrix contribution C += J * Y^T from one size group.

    PARALLELIZED VERSION: Each thread handles one (articulation, i, j) triplet.

    The Delassus matrix is: C = sum_art J_art * H_art^-1 * J_art^T = sum_art J_art * Y_art^T

    Since Y is stored as [constraint, dof], we compute:
        C[i,j] = sum_k J[i,k] * Y[j,k]

    Thread dimension: n_arts_of_size * max_constraints * max_constraints
    """
    tid = wp.tid()

    # Decode thread index
    j = tid % max_constraints
    i = (tid // max_constraints) % max_constraints
    idx = tid // (max_constraints * max_constraints)

    # Bounds check for articulation
    if idx >= n_arts:
        return

    art = group_to_art[idx]
    world = art_to_world[art]
    n_constraints = world_constraint_count[world]

    # Early exit if this (i, j) is beyond the actual constraint count
    if i >= n_constraints or j >= n_constraints:
        return

    # Compute C[i,j] = sum_k J[i,k] * Y[j,k]
    val = float(0.0)
    for k in range(n_dofs):
        val += J_group[idx, i, k] * Y_group[idx, j, k]

    if val != 0.0:
        wp.atomic_add(world_C, world, i, j, val)

    # Also accumulate diagonal separately (only when i == j)
    if i == j and val != 0.0:
        wp.atomic_add(world_diag, world, i, val)


# =============================================================================
# Tiled kernels for homogenous multi-articulation support
# =============================================================================


@wp.kernel
def crba_fill_batched_par_dof(
    articulation_start: wp.array(dtype=int),
    articulation_dof_start: wp.array(dtype=int),
    mass_update_mask: wp.array(dtype=int),
    joint_ancestor: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_dof_dim: wp.array(dtype=int, ndim=2),
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    body_I_c: wp.array(dtype=wp.spatial_matrix),
    # Size-group parameters
    group_to_art: wp.array(dtype=int),
    n_dofs: int,  # = TILE_DOF for tiled path
    # outputs
    H_group: wp.array3d(dtype=float),  # [n_arts_of_size, n_dofs, n_dofs]
):
    """
    CRBA fill kernel that writes directly to size-grouped H storage.

    Thread dimension: n_arts_of_size * n_dofs (one thread per articulation-column pair)

    This version is for homogenous multi-articulation where all articulations have
    the same DOF count equal to TILE_DOF.
    """
    tid = wp.tid()

    group_idx = tid // n_dofs
    col_idx = tid % n_dofs

    art_idx = group_to_art[group_idx]

    if mass_update_mask[art_idx] == 0:
        return

    # All articulations in this group have exactly n_dofs DOFs
    if col_idx >= n_dofs:
        return

    global_dof_start = articulation_dof_start[art_idx]
    target_dof_global = global_dof_start + col_idx

    joint_start = articulation_start[art_idx]
    joint_end = articulation_start[art_idx + 1]

    # Find the joint that owns this DOF
    pivot_joint = int(-1)
    for j in range(joint_start, joint_end):
        q_start = joint_qd_start[j]
        q_end = joint_qd_start[j + 1]
        if target_dof_global >= q_start and target_dof_global < q_end:
            pivot_joint = j
            break

    if pivot_joint == -1:
        return

    # Compute Force F = I_c[pivot] * S[column]
    S_col = joint_S_s[target_dof_global]
    I_comp = body_I_c[pivot_joint]
    F = I_comp * S_col

    # Walk up the tree and project F onto ancestors
    # H[row, col] = S[row] * F
    curr = pivot_joint

    while curr != -1:
        if curr < joint_start:
            break

        q_start = joint_qd_start[curr]
        q_dim = joint_dof_dim[curr]
        count = q_dim[0] + q_dim[1]

        dof_offset_local = q_start - global_dof_start

        for k in range(count):
            row_idx = dof_offset_local + k

            S_row = joint_S_s[q_start + k]
            val = wp.dot(S_row, F)

            # Write to grouped 3D array
            H_group[group_idx, row_idx, col_idx] = val
            H_group[group_idx, col_idx, row_idx] = val

        curr = joint_ancestor[curr]


@wp.kernel
def trisolve_batched_loop(
    L_group: wp.array3d(dtype=float),  # [n_arts_of_size, n_dofs, n_dofs]
    group_to_art: wp.array(dtype=int),
    articulation_dof_start: wp.array(dtype=int),
    n_dofs: int,
    joint_tau: wp.array(dtype=float),  # [total_dofs]
    # output
    joint_qdd: wp.array(dtype=float),  # [total_dofs]
):
    """
    Solve L * L^T * qdd = tau for grouped articulations using forward/backward substitution.

    Thread dimension: n_arts_of_size (one thread per articulation in this size group)
    """
    idx = wp.tid()
    art = group_to_art[idx]
    dof_start = articulation_dof_start[art]

    # Forward substitution: L * z = tau
    # z is stored temporarily in joint_qdd
    for i in range(n_dofs):
        val = joint_tau[dof_start + i]
        for k in range(i):
            L_ik = L_group[idx, i, k]
            val -= L_ik * joint_qdd[dof_start + k]

        L_ii = L_group[idx, i, i]
        if L_ii != 0.0:
            joint_qdd[dof_start + i] = val / L_ii
        else:
            joint_qdd[dof_start + i] = 0.0

    # Backward substitution: L^T * qdd = z
    for i_rev in range(n_dofs):
        i = n_dofs - 1 - i_rev

        val = joint_qdd[dof_start + i]
        for k in range(i + 1, n_dofs):
            L_ki = L_group[idx, k, i]
            val -= L_ki * joint_qdd[dof_start + k]

        L_ii = L_group[idx, i, i]
        if L_ii != 0.0:
            joint_qdd[dof_start + i] = val / L_ii
        else:
            joint_qdd[dof_start + i] = 0.0


@wp.kernel
def gather_tau_to_groups(
    joint_tau: wp.array(dtype=float),  # [total_dofs]
    group_to_art: wp.array(dtype=int),
    articulation_dof_start: wp.array(dtype=int),
    n_dofs: int,
    tau_group: wp.array3d(dtype=float),  # [n_arts, n_dofs, 1]
):
    """Gather joint_tau from flat array into grouped 3D buffer for tiled solve.

    Thread dimension: n_arts_of_size (one thread per articulation in this size group)
    """
    idx = wp.tid()
    art = group_to_art[idx]
    dof_start = articulation_dof_start[art]
    for i in range(n_dofs):
        tau_group[idx, i, 0] = joint_tau[dof_start + i]


@wp.kernel
def scatter_qdd_from_groups(
    qdd_group: wp.array3d(dtype=float),  # [n_arts, n_dofs, 1]
    group_to_art: wp.array(dtype=int),
    articulation_dof_start: wp.array(dtype=int),
    n_dofs: int,
    joint_qdd: wp.array(dtype=float),  # [total_dofs]
):
    """Scatter qdd from grouped 3D buffer back to flat array after tiled solve.

    Thread dimension: n_arts_of_size (one thread per articulation in this size group)
    """
    idx = wp.tid()
    art = group_to_art[idx]
    dof_start = articulation_dof_start[art]
    for i in range(n_dofs):
        joint_qdd[dof_start + i] = qdd_group[idx, i, 0]
