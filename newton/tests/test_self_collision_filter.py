# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import math
import unittest

import warp as wp

import newton
from newton.utils import filter_static_self_collision_pairs, find_static_self_collision_filters


def _build_test_builder() -> tuple[newton.ModelBuilder, int, int, int]:
    """Build an articulation with a permanently-adjacent pair and a reachable pair.

    Links A and B are parallel capsules tilted 45 degrees in the xz-plane and
    offset 6 cm along the tilt's perpendicular: their world AABBs overlap in
    every configuration (both sit on +-0.02 rad joints) while their surfaces
    stay ~3 cm apart -- the "permanent broad-phase candidate that never
    touches" class. Link C swings on a wide-range joint whose sweep passes
    through link A, so the (A, C) pair genuinely touches in some sampled
    configurations and must never be filtered.
    """
    builder = newton.ModelBuilder(gravity=0.0)
    builder.rigid_gap = 0.001
    builder.default_shape_cfg.gap = 0.001
    builder.default_shape_cfg.margin = 0.0

    qy45 = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.pi / 4.0)

    base = builder.add_link()
    builder.add_shape_box(base, hx=0.02, hy=0.02, hz=0.02)
    j_base = builder.add_joint_revolute(
        parent=-1,
        child=base,
        axis=wp.vec3(0.0, 0.0, 1.0),
        parent_xform=wp.transform_identity(),
        child_xform=wp.transform_identity(),
        limit_lower=-0.01,
        limit_upper=0.01,
    )
    joints = [j_base]

    def add_capsule_link(parent_xform, limit_lower, limit_upper):
        link = builder.add_link()
        shape = builder.add_shape_capsule(link, radius=0.015, half_height=0.35)
        joints.append(
            builder.add_joint_revolute(
                parent=base,
                child=link,
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent_xform=parent_xform,
                child_xform=wp.transform_identity(),
                limit_lower=limit_lower,
                limit_upper=limit_upper,
            )
        )
        return shape

    # A and B: parallel diagonal capsules, perpendicular offset 0.06 m.
    perp = 0.06 / math.sqrt(2.0)
    shape_a = add_capsule_link(wp.transform(wp.vec3(0.0, 0.0, 0.5), qy45), -0.02, 0.02)
    shape_b = add_capsule_link(wp.transform(wp.vec3(perp, 0.0, 0.5 - perp), qy45), -0.02, 0.02)

    # C: pivots right next to A's band with a wide range, so a large fraction
    # of its sweep intersects A.
    shape_c = add_capsule_link(wp.transform(wp.vec3(0.0, 0.0, 0.85), wp.quat_identity()), -1.6, 1.6)

    builder.add_articulation(joints)
    return builder, shape_a, shape_b, shape_c


class TestSelfCollisionFilter(unittest.TestCase):
    def test_permanent_pair_filtered_reachable_pair_kept(self):
        """Filter the always-overlapping never-touching pair but keep the reachable pair."""
        builder, shape_a, shape_b, shape_c = _build_test_builder()
        filters = find_static_self_collision_filters(builder, num_samples=64, seed=7)

        pair_ab = (min(shape_a, shape_b), max(shape_a, shape_b))
        pair_ac = (min(shape_a, shape_c), max(shape_a, shape_c))
        self.assertIn(pair_ab, filters)
        self.assertNotIn(pair_ac, filters)

    def test_apply_removes_pair_from_broad_phase(self):
        """Applying the filters removes the permanent pair from broad-phase candidates."""
        builder, shape_a, shape_b, _ = _build_test_builder()
        pairs = filter_static_self_collision_pairs(builder, num_samples=64, seed=7)
        pair_ab = (min(shape_a, shape_b), max(shape_a, shape_b))
        self.assertIn(pair_ab, pairs)

        model = builder.finalize()
        state = model.state()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state)
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        pipeline.collide(state, contacts)
        count = int(pipeline.broad_phase_pair_count.numpy()[0])
        cands = {(int(min(p0, p1)), int(max(p0, p1))) for p0, p1 in pipeline.broad_phase_shape_pairs.numpy()[:count]}
        self.assertNotIn(pair_ab, cands)

    def test_deterministic(self):
        """Return identical filters for identical seeds."""
        builder1, *_ = _build_test_builder()
        builder2, *_ = _build_test_builder()
        self.assertEqual(
            find_static_self_collision_filters(builder1, num_samples=16, seed=3),
            find_static_self_collision_filters(builder2, num_samples=16, seed=3),
        )


if __name__ == "__main__":
    unittest.main()
