# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Sparse-diagonal contact solve of SolverFeatherPGS: one diagonal-mass and one small dense articulation per world."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices

_PAIR_SOLVER = {
    "pgs_mode": "matrix_free",
    "dense_max_constraints": 64,
    "mf_max_constraints": 32,
    "pgs_iterations": 8,
    "enable_joint_limits": True,
}


def _build_sparse_diagonal_pair_model(
    num_branches=16, num_worlds=2, *, device=None, revolute_branches=False, with_free_body=False
):
    """Build one independent fixed-base star and one compact serial chain per world.

    ``revolute_branches`` hinges the star branches about Y instead of sliding them along Z, so the branch
    response depends on the link inertia and center of mass rather than on the mass alone.
    """
    scene = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    base = scene.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    star_joints = [scene.add_joint_fixed(parent=-1, child=base)]
    add_branch_joint = scene.add_joint_revolute if revolute_branches else scene.add_joint_prismatic
    for branch in range(num_branches):
        child = scene.add_link(mass=1.0 + 0.01 * branch, inertia=wp.mat33(np.eye(3)))
        star_joints.append(
            add_branch_joint(
                parent=base,
                child=child,
                axis=newton.Axis.Y if revolute_branches else newton.Axis.Z,
                parent_xform=wp.transform(wp.vec3(0.03 * branch, 0.0, 0.0), wp.quat_identity()),
                limit_lower=-0.1,
                limit_upper=0.1,
            )
        )
    scene.add_articulation(star_joints)

    chain_joints = []
    parent = -1
    for _link_index in range(3):
        child = scene.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        chain_joints.append(
            scene.add_joint_revolute(
                parent=parent,
                child=child,
                axis=newton.Axis.Y,
                parent_xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
                limit_lower=-0.2,
                limit_upper=0.2,
            )
        )
        parent = child
    scene.add_articulation(chain_joints)
    if with_free_body:
        body = scene.add_body(xform=wp.transform(wp.vec3(2.0, 0.0, 0.0), wp.quat_identity()))
        scene.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)

    replicated = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    replicated.replicate(scene, num_worlds, spacing=(3.0, 3.0, 0.0))
    return replicated.finalize(device=device)


def test_sparse_diagonal_is_selected_only_where_supported(test, device):
    """Keep the general sweep without streams, on CPU, with a free body or with options it does not solve."""
    model = _build_sparse_diagonal_pair_model(device=device)
    if not wp.get_device(device).is_cuda:
        solver = SolverFeatherPGS(
            model, pgs_mode="split", enable_joint_limits=True, dense_max_constraints=64, use_parallel_streams=True
        )
        test.assertFalse(solver._sparse_diagonal_contact_solve)
        return
    options = dict(_PAIR_SOLVER, use_parallel_streams=True)
    test.assertTrue(SolverFeatherPGS(model, **options)._sparse_diagonal_contact_solve)
    for label, overrides in (
        ("no streams", {"use_parallel_streams": False}),
        ("joint limits off", {"enable_joint_limits": False}),
        ("torsion", {"contact_torsion_radius": 0.01}),
        ("regularization", {"pgs_contact_regularization": 0.02}),
        ("velocity iterations", {"pgs_velocity_iterations": 1}),
        ("warm start", {"pgs_warmstart": True}),
        ("schedule", {"pgs_schedule": "physx_grasp"}),
        ("drive rows", {"drive_mode": "physx_pgs"}),
        ("friction mode", {"friction_mode": "bisection", "friction_anchor_beta": 0.0}),
    ):
        with test.subTest(label):
            solver = SolverFeatherPGS(model, **dict(options, **overrides))
            test.assertFalse(solver._sparse_diagonal_contact_solve)
            test.assertFalse(solver._sparse_diagonal_contact_triples)
    with_free_body = _build_sparse_diagonal_pair_model(device=device, with_free_body=True)
    test.assertFalse(SolverFeatherPGS(with_free_body, **options)._sparse_diagonal_contact_solve)


class TestFeatherPGSSparseDiagonal(unittest.TestCase):
    pass


cuda_devices = get_cuda_test_devices()
add_function_test(
    TestFeatherPGSSparseDiagonal,
    "test_sparse_diagonal_is_selected_only_where_supported",
    test_sparse_diagonal_is_selected_only_where_supported,
    devices=get_test_devices(),
)


if __name__ == "__main__":
    unittest.main()
