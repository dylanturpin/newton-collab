# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Construction, option validation and kernel selection of SolverFeatherPGS."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices


def _build_chain_model(device, num_links=3, num_worlds=2, *, with_free_body=False):
    chain = newton.ModelBuilder()
    hx = 0.3
    joints = []
    parent = -1
    root_rot = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), 0.45 * wp.pi)
    for _ in range(num_links):
        link = chain.add_link()
        chain.add_shape_box(link, hx=hx - 0.08, hy=0.05, hz=0.05)
        if parent == -1:
            parent_xform = wp.transform(p=wp.vec3(0.0, 0.0, 2.5), q=root_rot)
        else:
            parent_xform = wp.transform(p=wp.vec3(hx, 0.0, 0.0), q=wp.quat_identity())
        joints.append(
            chain.add_joint_revolute(
                parent=parent,
                child=link,
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent_xform=parent_xform,
                child_xform=wp.transform(p=wp.vec3(-hx, 0.0, 0.0), q=wp.quat_identity()),
            )
        )
        parent = link
    chain.add_articulation(joints)
    if with_free_body:
        body = chain.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
        chain.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
        chain.add_articulation([chain.add_joint_free(parent=-1, child=body)])
    main = newton.ModelBuilder()
    main.replicate(chain, num_worlds, spacing=(3.0, 3.0, 0.0))
    return main.finalize(device=device)


def _build_limited_chain_on_ground(device, num_links, num_worlds=2, *, with_free_body=False):
    """A hanging chain of driven, limited capsule links above the ground, optionally with a falling box."""
    world = newton.ModelBuilder()
    world.add_ground_plane()
    joints = []
    parent = -1
    top = 0.3 * num_links + 0.2
    for i in range(num_links):
        link = world.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, top - 0.3 * i), wp.quat_identity()))
        world.add_shape_capsule(link, radius=0.04, half_height=0.12)
        joints.append(
            world.add_joint_revolute(
                parent,
                link,
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent_xform=wp.transform(
                    wp.vec3(0.0, 0.0, top + 0.15) if parent < 0 else wp.vec3(0.0, 0.0, -0.15), wp.quat_identity()
                ),
                child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.15), wp.quat_identity()),
                limit_lower=-0.6,
                limit_upper=0.6,
                target_ke=200.0,
                target_kd=5.0,
                target_pos=0.3,
            )
        )
        parent = link
    world.add_articulation(joints)
    if with_free_body:
        body = world.add_body(xform=wp.transform(wp.vec3(0.3, 0.0, 0.3), wp.quat_rpy(0.2, 0.1, 0.3)))
        world.add_shape_box(body, hx=0.1, hy=0.08, hz=0.06)
    builder = newton.ModelBuilder()
    builder.replicate(world, num_worlds)
    return builder.finalize(device=device)


def test_default_kernel_selection_is_cached(test, device):
    """Resolve identical solver shapes to the same cached kernel objects."""
    model = _build_chain_model(device)
    first = SolverFeatherPGS(model, pgs_mode="matrix_free")
    second = SolverFeatherPGS(model, pgs_mode="matrix_free")
    for attr in ("_cholesky_kernels_by_size", "_triangular_solve_kernels_by_size", "_hinv_jt_kernels_by_size"):
        first_kernels = getattr(first, attr)
        second_kernels = getattr(second, attr)
        test.assertEqual(set(first_kernels), set(second_kernels))
        for size, kernel in first_kernels.items():
            test.assertIs(kernel, second_kernels[size], f"{attr}[{size}]")
    test.assertIs(first._pgs_solve_mf_gs_kernel, second._pgs_solve_mf_gs_kernel)


def test_diagonal_fusion_requires_nonaliased_world_response(test, device):
    """Compute the row diagonal in H^-1 J^T only when it writes separate world storage."""
    aliased = SolverFeatherPGS(
        _build_chain_model(device, num_links=23, num_worlds=1), pgs_mode="matrix_free", dense_max_constraints=192
    )
    direct = SolverFeatherPGS(
        _build_chain_model(device, num_links=23, num_worlds=1, with_free_body=True),
        pgs_mode="matrix_free",
        dense_max_constraints=192,
    )
    test.assertTrue(aliased._jy_world_aliased)
    test.assertFalse(aliased._hinv_jt_writes_world)
    test.assertEqual(aliased._hinv_jt_diag_sizes, frozenset())
    test.assertFalse(direct._jy_world_aliased)
    test.assertTrue(direct._hinv_jt_writes_world)
    test.assertEqual(direct._hinv_jt_diag_sizes, frozenset((23,)))


def test_tiled_and_loop_kernels_step_identically(test, device):
    """Match the tiled and loop factorizations on a chain that selects tiled kernels by default."""
    trajectories = []
    try:
        for overrides in ({}, {"cholesky_kernel": "loop", "trisolve_kernel": "loop", "hinv_jt_kernel": "par_row"}):
            SolverFeatherPGS._kernel_overrides = overrides
            model = _build_chain_model(device, num_links=14, num_worlds=2)
            solver = SolverFeatherPGS(model, pgs_mode="matrix_free", dense_max_constraints=64)
            if not overrides:
                test.assertTrue(solver._execution_plan.use_tiled_cholesky(14))
                test.assertTrue(solver._execution_plan.use_tiled_hinv_jt(14))
            state_0, state_1 = model.state(), model.state()
            control = model.control()
            for _ in range(10):
                solver.step(state_0, state_1, control, None, 1.0 / 120.0)
                state_0, state_1 = state_1, state_0
            trajectories.append(state_0.joint_q.numpy().copy())
    finally:
        SolverFeatherPGS._kernel_overrides = {}
    np.testing.assert_allclose(trajectories[0], trajectories[1], rtol=0.0, atol=1.0e-4)


def test_split_defaults(test, device):
    """Keep the documented defaults in split mode and allocate its Delassus storage."""
    solver = SolverFeatherPGS(_build_chain_model(device, num_links=2, num_worlds=2), pgs_mode="split")
    test.assertEqual(solver.pgs_mode, "split")
    test.assertEqual(solver.pgs_iterations, 12)
    test.assertEqual(solver.dense_max_constraints, 32)
    test.assertEqual(solver.C.shape, (2, 32, 32))
    test.assertFalse(solver._jy_world_aliased)
    test.assertFalse(solver._hinv_jt_writes_world)


def test_split_kernel_selection_is_cached(test, device):
    """Resolve identical split-mode solver shapes to the same cached kernel objects."""
    model = _build_chain_model(device, num_links=14, num_worlds=2, with_free_body=True)
    first = SolverFeatherPGS(model, pgs_mode="split")
    second = SolverFeatherPGS(model, pgs_mode="split")
    test.assertEqual(first._delassus_kernels_by_size.keys(), second._delassus_kernels_by_size.keys())
    for size, kernel in first._delassus_kernels_by_size.items():
        test.assertIs(kernel, second._delassus_kernels_by_size[size])
    test.assertIs(first._pgs_solve_tiled_row_kernel, second._pgs_solve_tiled_row_kernel)
    test.assertIs(first._pgs_solve_mf_kernel, second._pgs_solve_mf_kernel)


def test_split_native_and_scalar_kernels_step_identically(test, device):
    """Match the native split-mode kernels and the scalar Warp kernels of the CPU path on CUDA."""
    scalar = {
        "cholesky_kernel": "loop",
        "trisolve_kernel": "loop",
        "hinv_jt_kernel": "par_row",
        "delassus_kernel": "par_row_col",
        "pgs_kernel": "loop",
    }
    for with_free_body in (False, True):
        trajectories = []
        for overrides in ({}, scalar):
            SolverFeatherPGS._kernel_overrides = overrides
            try:
                model = _build_limited_chain_on_ground(device, 14, with_free_body=with_free_body)
                solver = SolverFeatherPGS(model, pgs_mode="split", enable_joint_limits=True, dense_max_constraints=64)
            finally:
                SolverFeatherPGS._kernel_overrides = {}
            if not overrides:
                test.assertIsNotNone(solver._pgs_solve_tiled_row_kernel)
            pipeline = newton.CollisionPipeline(model)
            contacts = pipeline.contacts()
            state_0, state_1 = model.state(), model.state()
            control = model.control()
            for _ in range(120):
                pipeline.collide(state_0, contacts)
                solver.step(state_0, state_1, control, contacts, 1.0 / 240.0)
                state_0, state_1 = state_1, state_0
            solver.check_constraint_capacity()
            test.assertGreater(int(solver.constraint_count.numpy().max()), 0)
            trajectories.append(state_0.joint_q.numpy().copy())
        with test.subTest(with_free_body=with_free_body):
            np.testing.assert_allclose(trajectories[0], trajectories[1], rtol=0.0, atol=1.0e-4)


def test_drive_mode_validation(test, device):
    """Default to the implicit drive and reject unknown drive formulations."""
    model = _build_chain_model(device, num_links=2, num_worlds=1)
    solver = SolverFeatherPGS(model)
    test.assertEqual(solver.drive_mode, "augmented")
    test.assertFalse(solver.fuse_joint_velocity_limits)
    test.assertEqual(SolverFeatherPGS(model, pgs_mode="matrix_free", drive_mode="physx_pgs").drive_mode, "physx_pgs")
    with test.assertRaisesRegex(NotImplementedError, "requires pgs_mode='matrix_free'"):
        SolverFeatherPGS(model, pgs_mode="split", drive_mode="physx_pgs")
    with test.assertRaisesRegex(ValueError, "drive_mode"):
        SolverFeatherPGS(model, drive_mode="implicit")


class TestFeatherPGSLaunchConfig(unittest.TestCase):
    pass


devices = get_cuda_test_devices()
for _name, _func in (
    ("test_default_kernel_selection_is_cached", test_default_kernel_selection_is_cached),
    (
        "test_diagonal_fusion_requires_nonaliased_world_response",
        test_diagonal_fusion_requires_nonaliased_world_response,
    ),
    ("test_tiled_and_loop_kernels_step_identically", test_tiled_and_loop_kernels_step_identically),
    ("test_drive_mode_validation", test_drive_mode_validation),
):
    add_function_test(TestFeatherPGSLaunchConfig, _name, _func, devices=devices)
split_devices = get_test_devices()
for _name, _func, _devices in (
    ("test_split_defaults", test_split_defaults, split_devices),
    ("test_split_kernel_selection_is_cached", test_split_kernel_selection_is_cached, split_devices),
    (
        "test_split_native_and_scalar_kernels_step_identically",
        test_split_native_and_scalar_kernels_step_identically,
        devices,
    ),
):
    add_function_test(TestFeatherPGSLaunchConfig, _name, _func, devices=_devices)


if __name__ == "__main__":
    unittest.main()
