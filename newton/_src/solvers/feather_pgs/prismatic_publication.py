# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Body-parallel Stage 7 publication for fixed-root prismatic stars.

The plan partitions only immutable topology, never joint frames or world poses.
The ordinary FK launch publishes each admitted root before the body launch reads
it. Unsupported articulations retain their complete ordinary FK traversal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from ...sim import JointType, Model
from .kernels import finalize_body_dynamics_body


@dataclass
class PrismaticPublicationPlan:
    """Persistent Stage 7-only partition; no factor or physical state is cached."""

    joint_end: wp.array[int]
    body_joint: wp.array[int]
    articulation_count: int
    body_count: int

    @classmethod
    def build(cls, model: Model, articulation_joint_end: wp.array[int]) -> PrismaticPublicationPlan | None:
        """Admit only a world-fixed root followed by independent prismatic leaves.

        As with the solver's other topology schedules, topology changes require
        reconstructing the solver. Frame, axis, inertial and state values are
        read from the current canonical buffers at every publication.
        """
        if not model.articulation_count or not model.joint_count:
            return None
        starts = model.articulation_start.numpy()
        ends = articulation_joint_end.numpy().copy()
        types = model.joint_type.numpy()
        parents = model.joint_parent.numpy()
        children = model.joint_child.numpy()
        dims = model.joint_dof_dim.numpy()
        q_start = model.joint_q_start.numpy()
        qd_start = model.joint_qd_start.numpy()
        owners = np.bincount(children[children >= 0], minlength=model.body_count)
        descendants = np.bincount(parents[parents >= 0], minlength=model.body_count)
        body_joint = np.full(model.body_count, -1, dtype=np.int32)
        admitted = 0
        for art in range(model.articulation_count):
            start, end = int(starts[art]), int(ends[art])
            # Do not partition loop-closing or empty articulations.
            if end != int(starts[art + 1]) or end - start < 2:
                continue
            root = int(children[start])
            if (
                types[start] not in (JointType.FIXED, JointType.D6)
                or parents[start] != -1
                or root < 0
                or owners[root] != 1
                or np.any(dims[start] != 0)
                or q_start[start + 1] != q_start[start]
                or qd_start[start + 1] != qd_start[start]
            ):
                continue
            joints = np.arange(start + 1, end, dtype=np.int32)
            leaves = children[joints]
            if (
                np.any(leaves < 0)
                or np.any(leaves == root)
                or np.any(types[joints] != JointType.PRISMATIC)
                or np.any(parents[joints] != root)
                or np.any(dims[joints, 0] != 1)
                or np.any(dims[joints, 1] != 0)
                or np.any(q_start[joints + 1] - q_start[joints] != 1)
                or np.any(qd_start[joints + 1] - qd_start[joints] != 1)
                or np.any(owners[leaves] != 1)
                or np.any(descendants[leaves] != 0)
            ):
                continue
            ends[art] = start + 1
            body_joint[leaves] = joints
            admitted += 1
        if not admitted:
            return None
        return cls(
            joint_end=wp.array(ends, dtype=wp.int32, device=model.device),
            body_joint=wp.array(body_joint, dtype=wp.int32, device=model.device),
            articulation_count=admitted,
            body_count=int(np.count_nonzero(body_joint >= 0)),
        )


@wp.kernel
def finalize_prismatic_body_dynamics(
    body_joint: wp.array[int],
    joint_parent: wp.array[int],
    joint_q_start: wp.array[int],
    joint_qd_start: wp.array[int],
    joint_q: wp.array[float],
    joint_qd: wp.array[float],
    joint_X_p: wp.array[wp.transform],
    joint_X_c: wp.array[wp.transform],
    body_X_com: wp.array[wp.transform],
    joint_axis: wp.array[wp.vec3],
    joint_S_s: wp.array[wp.spatial_vector],
    body_to_articulation: wp.array[int],
    body_q: wp.array[wp.transform],
    body_q_com: wp.array[wp.transform],
    body_com: wp.array[wp.vec3],
    body_mass: wp.array[float],
    body_inertia: wp.array[wp.mat33],
    is_free_rigid: wp.array[int],
    articulation_origin: wp.array[wp.vec3],
    materialize_all_body_inertia: int,
    materialize_body_inertia_terms: int,
    gravity: wp.array[wp.vec3],
    body_v_s: wp.array[wp.spatial_vector],
    body_a_s: wp.array[wp.spatial_vector],
    body_I_s: wp.array[wp.spatial_matrix],
    body_inertia_terms: wp.array2d[float],
    body_f_s: wp.array[wp.spatial_vector],
    body_qd: wp.array[wp.spatial_vector],
):
    """Publish each leaf once, then use the unchanged full body-dynamics law."""
    body = wp.tid()
    joint = body_joint[body]
    if joint >= 0:
        # The preceding FK launch has already written the current fixed root.
        # Its frames may change through notify_model_changed; nothing is baked.
        dof = joint_qd_start[joint]
        X_wpj = body_q[joint_parent[joint]] * joint_X_p[joint]
        axis = joint_axis[dof]
        X_j = wp.transform(axis * joint_q[joint_q_start[joint]], wp.quat_identity())
        X_wc = (X_wpj * X_j) * wp.transform_inverse(joint_X_c[joint])
        body_q[body] = X_wc
        body_q_com[body] = X_wc * body_X_com[body]
        # A translation-only motion has no angular component or origin shift.
        # The parent's generalized velocity is exactly zero by admission.
        S = wp.spatial_vector(wp.quat_rotate(wp.transform_get_rotation(X_wpj), axis), wp.vec3())
        joint_S_s[dof] = S
        body_v_s[body] = S * joint_qd[dof]
        body_a_s[body] = wp.spatial_vector()
    finalize_body_dynamics_body(
        body,
        body_to_articulation,
        body_q,
        body_q_com,
        body_com,
        body_mass,
        body_inertia,
        is_free_rigid,
        articulation_origin,
        materialize_all_body_inertia,
        materialize_body_inertia_terms,
        gravity,
        body_v_s,
        body_a_s,
        body_I_s,
        body_inertia_terms,
        body_f_s,
        body_qd,
    )
