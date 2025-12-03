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
from ..semi_implicit.kernels_body import joint_force

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
    joint_target_ke: wp.array(dtype=float),
    joint_target_kd: wp.array(dtype=float),
    joint_limit_ke: wp.array(dtype=float),
    joint_limit_kd: wp.array(dtype=float),
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_f: wp.array(dtype=float),
    joint_target_pos: wp.array(dtype=float),
    joint_target_vel: wp.array(dtype=float),
    joint_limit_lower: wp.array(dtype=float),
    joint_limit_upper: wp.array(dtype=float),
    coord_start: int,
    dof_start: int,
    lin_axis_count: int,
    ang_axis_count: int,
    body_f_s: wp.spatial_vector,
    use_joint_targets: int,
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

            q = joint_q[coord_start + i]
            qd = joint_qd[j]

            lower = joint_limit_lower[j]
            upper = joint_limit_upper[j]
            limit_ke = joint_limit_ke[j]
            limit_kd = joint_limit_kd[j]
            target_ke = joint_target_ke[j]
            target_kd = joint_target_kd[j]
            target_pos = joint_target_pos[j]
            target_vel = joint_target_vel[j]

            drive_f = 0.0
            if use_joint_targets:
                drive_f = joint_force(
                    q,
                    qd,
                    target_pos,
                    target_vel,
                    target_ke,
                    target_kd,
                    lower,
                    upper,
                    limit_ke,
                    limit_kd,
                )

            # total torque / force on the joint
            t = -wp.dot(S_s, body_f_s) + drive_f + joint_f[j]

            tau[j] = t

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
    a_s = a_parent_s + spatial_cross(v_s, v_j_s)  # + joint_S_s[i]*self.joint_qdd[i]

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
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_dof_dim: wp.array(dtype=int, ndim=2),
    joint_target_pos: wp.array(dtype=float),
    joint_target_vel: wp.array(dtype=float),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_f: wp.array(dtype=float),
    joint_target_ke: wp.array(dtype=float),
    joint_target_kd: wp.array(dtype=float),
    joint_limit_lower: wp.array(dtype=float),
    joint_limit_upper: wp.array(dtype=float),
    joint_limit_ke: wp.array(dtype=float),
    joint_limit_kd: wp.array(dtype=float),
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    body_fb_s: wp.array(dtype=wp.spatial_vector),
    body_f_ext: wp.array(dtype=wp.spatial_vector),
    body_q: wp.array(dtype=wp.transform),
    body_com: wp.array(dtype=wp.vec3),
    use_joint_targets: int,
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
        coord_start = joint_q_start[i]
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
            joint_target_ke,
            joint_target_kd,
            joint_limit_ke,
            joint_limit_kd,
            joint_S_s,
            joint_q,
            joint_qd,
            joint_f,
            joint_target_pos,
            joint_target_vel,
            joint_limit_lower,
            joint_limit_upper,
            coord_start,
            dof_start,
            lin_axis_count,
            ang_axis_count,
            f_s,
            use_joint_targets,
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
def eval_crba(
    articulation_start: wp.array(dtype=int),
    mass_update_mask: wp.array(dtype=int),
    joint_ancestor: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    articulation_H_start: wp.array(dtype=int),
    articulation_H_rows: wp.array(dtype=int),
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    body_I_s: wp.array(dtype=wp.spatial_matrix),
    # outputs
    H: wp.array(dtype=float),
):
    articulation = wp.tid()

    if mass_update_mask[articulation] == 0:
        return

    joint_start = articulation_start[articulation]
    joint_end = articulation_start[articulation + 1]
    joint_count = joint_end - joint_start

    H_offset = articulation_H_start[articulation]
    dof_count = articulation_H_rows[articulation]
    articulation_dof_start = joint_qd_start[joint_start]

    for local_index in range(joint_count - 1, -1, -1):
        joint_index = joint_start + local_index
        I = body_I_s[joint_index]
        parent = joint_ancestor[joint_index]

        if parent != -1:
            parent_I = body_I_s[parent]
            parent_I += I
            body_I_s[parent] = parent_I

        joint_dof_start_0 = joint_qd_start[joint_index]
        joint_dof_end_0 = joint_qd_start[joint_index + 1]
        joint_dof_count_0 = joint_dof_end_0 - joint_dof_start_0
        joint_dof_base_0 = joint_dof_start_0 - articulation_dof_start

        j = joint_index
        while j != -1:
            joint_dof_start_1 = joint_qd_start[j]
            joint_dof_end_1 = joint_qd_start[j + 1]
            joint_dof_count_1 = joint_dof_end_1 - joint_dof_start_1
            joint_dof_base_1 = joint_dof_start_1 - articulation_dof_start

            for dof_0 in range(joint_dof_count_0):
                row = joint_dof_base_0 + dof_0
                S_0 = joint_S_s[joint_dof_start_0 + dof_0]
                F = I * S_0

                for dof_1 in range(joint_dof_count_1):
                    col = joint_dof_base_1 + dof_1
                    S_1 = joint_S_s[joint_dof_start_1 + dof_1]
                    r = wp.dot(F, S_1)

                    H[H_offset + dense_index(dof_count, row, col)] = r
                    H[H_offset + dense_index(dof_count, col, row)] = r

            j = joint_ancestor[j]


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


@wp.func_grad(dense_cholesky)
def adj_dense_cholesky(
    n: int,
    A: wp.array(dtype=float),
    R: wp.array(dtype=float),
    A_start: int,
    R_start: int,
    # outputs
    L: wp.array(dtype=float),
):
    # nop, use dense_solve to differentiate through (A^-1)b = x
    pass


@wp.kernel
def eval_dense_cholesky_batched(
    A_starts: wp.array(dtype=int),
    A_dim: wp.array(dtype=int),
    A: wp.array(dtype=float),
    R: wp.array(dtype=float),
    mass_update_mask: wp.array(dtype=int),
    L: wp.array(dtype=float),
):
    batch = wp.tid()

    if mass_update_mask[batch] == 0:
        return

    n = A_dim[batch]
    A_start = A_starts[batch]
    R_start = n * batch

    dense_cholesky(n, A, R, A_start, R_start, L)


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


@wp.func_grad(dense_solve)
def adj_dense_solve(
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
    if not tmp or not wp.adjoint[x] or not wp.adjoint[A] or not wp.adjoint[L]:
        return
    for i in range(n):
        tmp[b_start + i] = 0.0

    dense_subs(n, L_start, b_start, L, wp.adjoint[x], tmp)

    for i in range(n):
        wp.adjoint[b][b_start + i] += tmp[b_start + i]

    # A* = -adj_b*x^T
    for i in range(n):
        for j in range(n):
            wp.adjoint[L][L_start + dense_index(n, i, j)] += -tmp[b_start + i] * x[b_start + j]

    for i in range(n):
        for j in range(n):
            wp.adjoint[A][L_start + dense_index(n, i, j)] += -tmp[b_start + i] * x[b_start + j]


@wp.kernel
def eval_dense_solve_batched(
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

    slot = wp.atomic_add(constraint_counts, articulation, 1)
    if slot >= max_constraints:
        return

    phi_index = articulation * max_constraints + slot
    phi_out[phi_index] = phi
    row_beta[phi_index] = contact_beta
    row_cfm[phi_index] = contact_cfm
    row_types[phi_index] = PGS_CONSTRAINT_TYPE_CONTACT
    target_velocity[phi_index] = 0.0
    row_parent[phi_index] = -1
    row_mu[phi_index] = mu
    row_base = (articulation * max_constraints + slot) * max_dofs
    for col in range(max_dofs):
        Jc_out[row_base + col] = 0.0

    art_dof_start = articulation_dof_start[articulation]

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

    dof_count = articulation_H_rows[articulation]
    if enable_friction == 0 or mu <= 0.0 or dof_count == 0:
        return

    t0, t1 = contact_tangent_basis(n)

    for tangent_index in range(2):
        tangent = t0
        if tangent_index == 1:
            tangent = t1

        tangent_slot = wp.atomic_add(constraint_counts, articulation, 1)
        if tangent_slot >= max_constraints:
            return

        row_index = articulation * max_constraints + tangent_slot
        tangent_base = row_index * max_dofs

        for col in range(max_dofs):
            Jc_out[tangent_base + col] = 0.0

        row_beta[row_index] = 0.0
        row_cfm[row_index] = contact_cfm
        row_types[row_index] = PGS_CONSTRAINT_TYPE_FRICTION
        target_velocity[row_index] = 0.0
        phi_out[row_index] = 0.0
        row_parent[row_index] = phi_index
        row_mu[row_index] = mu

        accumulate_contact_jacobian_matrix_free(
            articulation,
            body_a,
            1.0,
            point_a_world,
            tangent,
            body_to_joint,
            body_to_articulation,
            joint_ancestor,
            joint_qd_start,
            joint_S_s,
            art_dof_start,
            tangent_base,
            Jc_out,
        )

        accumulate_contact_jacobian_matrix_free(
            articulation,
            body_b,
            -1.0,
            point_b_world,
            tangent,
            body_to_joint,
            body_to_articulation,
            joint_ancestor,
            joint_qd_start,
            joint_S_s,
            art_dof_start,
            tangent_base,
            Jc_out,
        )


@wp.kernel
def build_joint_target_rows(
    articulation_start: wp.array(dtype=int),
    articulation_H_rows: wp.array(dtype=int),
    articulation_dof_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_dof_dim: wp.array(dtype=int, ndim=2),
    joint_target_ke: wp.array(dtype=float),
    joint_target_kd: wp.array(dtype=float),
    joint_q: wp.array(dtype=float),
    joint_target_pos: wp.array(dtype=float),
    joint_target_vel: wp.array(dtype=float),
    # in-out
    constraint_counts: wp.array(dtype=int),
    max_constraints: int,
    max_dofs: int,
    row_beta: wp.array(dtype=float),
    row_cfm: wp.array(dtype=float),
    row_types: wp.array(dtype=int),
    target_velocity: wp.array(dtype=float),
    phi_out: wp.array(dtype=float),
    J_rows: wp.array(dtype=float),
    row_parent: wp.array(dtype=int),
    row_mu: wp.array(dtype=float),
    dt: float,
    default_beta: float,
    default_cfm: float,
    beta_override: float,
    cfm_override: float,
):
    articulation = wp.tid()
    dof_count = articulation_H_rows[articulation]
    if dof_count == 0:
        return

    joint_start = articulation_start[articulation]
    joint_end = articulation_start[articulation + 1]
    slot = constraint_counts[articulation]
    dof_start = articulation_dof_start[articulation]

    for joint_index in range(joint_start, joint_end):
        if slot >= max_dofs:
            break
        type = joint_type[joint_index]
        if type != JointType.PRISMATIC and type != JointType.REVOLUTE and type != JointType.D6:
            continue

        lin_axis_count = joint_dof_dim[joint_index, 0]
        ang_axis_count = joint_dof_dim[joint_index, 1]
        axis_count = lin_axis_count + ang_axis_count

        qd_start = joint_qd_start[joint_index]
        coord_start = joint_q_start[joint_index]

        for axis in range(axis_count):
            if slot >= max_constraints:
                break

            dof_index = qd_start + axis
            coord_index = coord_start + axis

            ke = joint_target_ke[dof_index]
            kd = joint_target_kd[dof_index]

            if ke <= 0.0 and kd <= 0.0:
                continue

            local_dof = dof_index - dof_start
            if local_dof < 0 or local_dof >= max_dofs:
                continue

            row_index = articulation * max_constraints + slot
            row_base = row_index * max_dofs

            for col in range(max_dofs):
                J_rows[row_base + col] = 0.0

            J_rows[row_base + local_dof] = 1.0

            phi = joint_q[coord_index] - joint_target_pos[dof_index]
            phi_out[row_index] = phi
            target_velocity[row_index] = joint_target_vel[dof_index]
            row_types[row_index] = PGS_CONSTRAINT_TYPE_JOINT_TARGET
            row_parent[row_index] = -1
            row_mu[row_index] = 0.0

            denom = kd + dt * ke
            beta_drive = default_beta
            cfm_drive = default_cfm

            if beta_override >= 0.0:
                beta_drive = beta_override
            elif denom > 0.0:
                beta_drive = dt * ke / denom

            if cfm_override >= 0.0:
                cfm_drive = cfm_override
            elif denom > 0.0:
                cfm_drive = 1.0 / denom

            row_beta[row_index] = beta_drive
            row_cfm[row_index] = cfm_drive

            slot += 1

            if slot >= max_constraints:
                break

    constraint_counts[articulation] = slot


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
    if max_dofs == 0 or dt <= 0.0:
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
def apply_hinv_Jt_multi_rhs(
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
def form_contact_matrix(
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


# @wp.kernel
# def form_contact_matrix(
#     articulation_H_rows: wp.array(dtype=int),
#     max_constraints: int,
#     max_dofs: int,
#     constraint_counts: wp.array(dtype=int),
#     J_rows: wp.array(dtype=float),
#     Y_rows: wp.array(dtype=float),
#     row_cfm: wp.array(dtype=float),
#     # outputs
#     diag_out: wp.array(dtype=float),
#     matrix_out: wp.array(dtype=float),
# ):
#     articulation = wp.tid()
#     m = constraint_counts[articulation]
#     n = articulation_H_rows[articulation]

#     if m == 0 or n == 0:
#         return

#     diag_base = articulation * max_constraints
#     mat_base = articulation * max_constraints * max_constraints

#     for i in range(m):
#         row_i = (articulation * max_constraints + i) * max_dofs

#         diag_val = float(0.0)
#         for k in range(n):
#             diag_val += J_rows[row_i + k] * Y_rows[row_i + k]

#         diag_out[diag_base + i] = diag_val + row_cfm[diag_base + i]

#         for j in range(m):
#             row_j = (articulation * max_constraints + j) * max_dofs

#             s = float(0.0)
#             for k in range(n):
#                 s += J_rows[row_i + k] * Y_rows[row_j + k]

#             matrix_out[mat_base + i * max_constraints + j] = s


@wp.kernel
def compute_contact_bias(
    articulation_dof_start: wp.array(dtype=int),
    articulation_H_rows: wp.array(dtype=int),
    constraint_counts: wp.array(dtype=int),
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
    articulation = wp.tid()
    m = constraint_counts[articulation]
    n = articulation_H_rows[articulation]

    if m == 0 or n == 0 or dt <= 0.0:
        return

    dof_start = articulation_dof_start[articulation]
    rhs_base = articulation * max_constraints

    for i in range(m):
        row_base = (articulation * max_constraints + i) * max_dofs

        rel_vel = float(0.0)
        for k in range(n):
            rel_vel += J_rows[row_base + k] * v_hat[dof_start + k]

        gap = phi[rhs_base + i]
        beta_val = row_beta[rhs_base + i]
        constraint_type = row_types[rhs_base + i]
        target_vel = target_velocity[rhs_base + i]

        if constraint_type == PGS_CONSTRAINT_TYPE_CONTACT:
            rhs = rel_vel - target_vel
            if gap < 0.0 and dt > 0.0:
                rhs += beta_val * gap / dt
            rhs_out[rhs_base + i] = rhs
        elif constraint_type == PGS_CONSTRAINT_TYPE_FRICTION:
            rhs_out[rhs_base + i] = rel_vel - target_vel
        elif constraint_type == PGS_CONSTRAINT_TYPE_JOINT_TARGET:
            correction = 0.0
            if dt > 0.0:
                correction = beta_val * gap / dt
            rhs_out[rhs_base + i] = rel_vel - target_vel + correction
        else:
            rhs_out[rhs_base + i] = rel_vel


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
def pgs_solve_contacts(
    constraint_counts: wp.array(dtype=int),
    max_constraints: int,
    diag: wp.array(dtype=float),
    matrix: wp.array(dtype=float),
    rhs: wp.array(dtype=float),
    impulses: wp.array(dtype=float),
    iterations: int,
    omega: float,
    row_types: wp.array(dtype=int),
    row_parent: wp.array(dtype=int),
    row_mu: wp.array(dtype=float),
):
    articulation = wp.tid()
    m = constraint_counts[articulation]

    if m == 0:
        return

    base = articulation * max_constraints
    mat_base = articulation * max_constraints * max_constraints

    for _ in range(iterations):
        for i in range(m):
            idx = base + i
            row_offset = mat_base + i * max_constraints

            w = rhs[idx]
            for j in range(m):
                w += matrix[row_offset + j] * impulses[base + j]

            denom = diag[idx]
            if denom <= 0.0:
                continue

            delta = -w / denom
            new_impulse = impulses[idx] + omega * delta
            row_type = row_types[idx]

            if row_type == PGS_CONSTRAINT_TYPE_CONTACT:
                if new_impulse < 0.0:
                    new_impulse = 0.0
            elif row_type == PGS_CONSTRAINT_TYPE_FRICTION:
                parent_idx = row_parent[idx]
                limit = 0.0
                if parent_idx >= 0:
                    limit = wp.max(row_mu[idx] * impulses[parent_idx], 0.0)
                if limit <= 0.0:
                    new_impulse = 0.0
                else:
                    if new_impulse > limit:
                        new_impulse = limit
                    elif new_impulse < -limit:
                        new_impulse = -limit

            impulses[idx] = new_impulse


@wp.kernel
def accumulate_contact_velocity(
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

    # Per-DoF effort limit (same convention as you use for MuJoCo actuators)
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

# Max generalized dofs per articulation we support in the tiled path.
# joint_dof_count per articulation must be <= TILE_DOF or we use fall back
TILE_DOF = wp.constant(49)

# Max constraints per articulation we support in the tiled path.
# pgs_max_constraints must be <= TILE_CONSTRAINTS or we use fall back
TILE_CONSTRAINTS = wp.constant(32)

# Threads per tile/block for tile kernels
TILE_THREADS = 64


@wp.kernel
def apply_hinv_Jt_multi_rhs_tiled(
    articulation_H_start: wp.array(dtype=int),
    articulation_H_rows: wp.array(dtype=int),
    max_constraints: int,
    max_dofs: int,
    constraint_counts: wp.array(dtype=int),
    L: wp.array3d(dtype=float),
    J_rows: wp.array3d(dtype=float),
    row_cfm: wp.array(dtype=float),
    # outputs
    Y: wp.array3d(dtype=float),
    C: wp.array3d(dtype=float),
    diag_out: wp.array(dtype=float),
):
    """
    Tiled version of H^{-1} J^T application.

    For each articulation 'a', we:
      - load the lower Cholesky factor L_a (n x n) into a tile (padded to TILE_DOF)
      - load J_a rows (m x n) into a RHS tile as columns
      - solve L_a L_a^T X = J^T for X using tile_lower_solve + tile_upper_solve
      - write X back into Y_rows in the same layout as before

    Assumes:
      articulation_H_rows[a] <= TILE_DOF
      constraint_counts[a]   <= TILE_CONSTRAINTS
    """
    articulation, thread = wp.tid()

    # --- 1. Load L for this articulation into L_tile (padded) ---
    L_tile = wp.tile_load(L[articulation], shape=(TILE_DOF, TILE_DOF), bounds_check=False)

    # --- 2. Load J rows into RHS_tile as columns: RHS[:, ci] = J_row(ci)^T ---
    J_tile = wp.tile_load(J_rows[articulation], shape=(TILE_CONSTRAINTS, TILE_DOF), bounds_check=False)

    # --- 3. Solve L * Z = RHS (forward) ---
    Z_tile = wp.tile_lower_solve(L_tile, wp.tile_transpose(J_tile))

    # --- 4. Solve L^T * X = Z (backward) ---
    U_tile = wp.tile_transpose(L_tile)  # U is upper-triangular
    X_tile = wp.tile_upper_solve(U_tile, Z_tile)

    C_tile = wp.tile_zeros(shape=(TILE_CONSTRAINTS, TILE_CONSTRAINTS), dtype=wp.float32)

    # store Y = H^-1 * J^T (will be re-used during impulse application)
    wp.tile_store(Y[articulation], wp.tile_transpose(X_tile))

    # form C = J * H^-1 * J^T
    wp.tile_matmul(J_tile, X_tile, C_tile)
    wp.tile_store(C[articulation], C_tile)

    if thread == 0:
        constraint_count = constraint_counts[articulation]
        diag_base = articulation * max_constraints

        for i in range(constraint_count):
            # write diagonal of constraint matrix
            # todo: remove this since we should already
            # have it during the PGS solve
            diag_out[diag_base + i] = C_tile[i, i] + row_cfm[diag_base + i]


@wp.kernel
def form_contact_matrix_tiled(
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
    """
    Tiled version of Delassus matrix build:
        W = J H^{-1} J^T  (plus CFM on diagonal)

    We treat J and Y as (m x n) blocks per articulation, pad them into tiles of
    size (TILE_CONSTRAINTS x TILE_DOF), and then compute:

        W = J * Y^T

    with tile_matmul.
    """
    articulation = wp.tid()

    m = constraint_counts[articulation]
    n = articulation_H_rows[articulation]

    # constraints (rows) x dofs (cols)
    J_tile = wp.tile_zeros(shape=(TILE_CONSTRAINTS, TILE_DOF), dtype=wp.float32)
    Y_tile = wp.tile_zeros(shape=(TILE_CONSTRAINTS, TILE_DOF), dtype=wp.float32)

    # --- 1. Load J and Y into tiles  ---
    # (not using tile_load since stored flat to support varied sizes)

    for i in range(m):
        row_i = (articulation * max_constraints + i) * max_dofs
        for k in range(n):
            J_tile[i, k] = J_rows[row_i + k]
            Y_tile[i, k] = Y_rows[row_i + k]

    # --- 2. Compute W = J * Y^T ---

    Y_T = wp.tile_transpose(Y_tile)  # (TILE_DOF x TILE_CONSTRAINTS)

    W_tile = wp.tile_zeros(
        shape=(TILE_CONSTRAINTS, TILE_CONSTRAINTS),
        dtype=wp.float32,
    )

    wp.tile_matmul(J_tile, Y_T, W_tile)

    # --- 3. Write back to diag_out and matrix_out (only m x m) ---

    diag_base = articulation * max_constraints
    mat_base = articulation * max_constraints * max_constraints

    for i in range(m):
        # diag
        diag_val = W_tile[i, i] + row_cfm[diag_base + i]
        diag_out[diag_base + i] = diag_val

        # full row
        for j in range(m):
            matrix_out[mat_base + i * max_constraints + j] = W_tile[i, j]


@wp.kernel
def eval_dense_cholesky_batched_tiled(
    H_starts: wp.array(dtype=int),
    H_dim: wp.array(dtype=int),
    H: wp.array3d(dtype=float),
    R: wp.array2d(dtype=float),
    mass_update_mask: wp.array(dtype=int),
    L: wp.array3d(dtype=float),
):
    # articulation index
    batch = wp.tid()

    if mass_update_mask[batch] == 0:
        return
    # Load H
    H_tile = wp.tile_load(H[batch], shape=(TILE_DOF, TILE_DOF), bounds_check=False)
    armature = wp.tile_load(R[batch], shape=TILE_DOF, bounds_check=False)

    # add armature to the diagonal of H
    H_tile = wp.tile_diag_add(H_tile, armature)

    # decomposition
    L_tile = wp.tile_cholesky(H_tile)

    wp.tile_store(L[batch], L_tile)


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
