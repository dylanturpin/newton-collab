# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Sparse mass factors of SolverFeatherPGS against the dense factors in complete contact steps."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS

DT = 1.0 / 400.0
STATE_FIELDS = ("joint_q", "joint_qd", "body_q", "body_qd")


def _build_y_tree(device, *, mimic=False, legacy_mimic=False, closure=False, closure_enabled=True):
    """Build a fixed-base Y tree (a root link with two child links), optionally coupled.

    The tree's mass factor has structural zeros, so it selects sparse factors on its own.
    The couplings are a joint-owned or legacy mimic between the two children, or a BALL
    loop-closing joint between their tips.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    inertia = wp.mat33(np.eye(3) * 0.01)
    root = builder.add_link(mass=1.0, inertia=inertia)
    joints = [builder.add_joint_revolute(-1, root, axis=newton.Axis.Z)]
    children = []
    for side in (-1.0, 1.0):
        child = builder.add_link(mass=0.5, inertia=inertia)
        joints.append(
            builder.add_joint_revolute(
                root,
                child,
                axis=newton.Axis.Z,
                parent_xform=wp.transform(wp.vec3(side * 0.2, 0.0, 0.0), wp.quat_identity()),
            )
        )
        children.append(child)
    builder.add_articulation(joints)
    if mimic:
        builder.set_joint_mimic(joints[2], joints[1], coeffs=(0.0, 1.0))
    if legacy_mimic:
        builder.add_constraint_mimic(joints[2], joints[1], coef1=1.0)
    if closure:
        builder.add_joint_ball(
            children[0],
            children[1],
            parent_xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
            enabled=closure_enabled,
        )
    return builder.finalize(device=device)


class TestFeatherPGSSparseSelectionGuard(unittest.TestCase):
    """Keep dense factors wherever the solver builds dense bilateral rows."""

    @unittest.skipUnless(wp.is_cuda_available(), "Sparse integrated solve requires CUDA")
    def test_y_tree_selects_sparse_factors(self):
        """Select sparse factors for the uncoupled Y tree used by the bilateral guard."""
        self.assertEqual(
            SolverFeatherPGS(
                _build_y_tree("cuda:0"), pgs_mode="matrix_free", friction_anchor_beta=0.0
            )._sparse_mass_matrix_size,
            3,
        )

    @unittest.skipUnless(wp.is_cuda_available(), "Sparse integrated solve requires CUDA")
    @unittest.skipUnless(
        hasattr(SolverFeatherPGS, "set_loop_joint_enabled"), "requires the FeatherPGS mimic and connect rows"
    )
    def test_bilateral_rows_keep_dense_factors(self):
        """Keep dense factors, and their response storage, when the solver builds bilateral rows."""
        for label, kwargs in (
            ("mimic", {"mimic": True}),
            ("closure", {"closure": True}),
            ("disabled closure", {"closure": True, "closure_enabled": False}),
        ):
            with self.subTest(label):
                solver = SolverFeatherPGS(_build_y_tree("cuda:0", **kwargs), pgs_mode="matrix_free")
                self.assertIsNone(solver._sparse_mass_matrix_size)
                self.assertGreater(solver.J_by_size[3].size, 1)


if __name__ == "__main__":
    unittest.main()
