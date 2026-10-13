# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Mass-matrix assembly, diagonal mass, refresh cadence and model notifications of SolverFeatherPGS."""

import gc
import subprocess
import sys
import unittest
import weakref
from types import SimpleNamespace
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton import BodyFlags, ModelFlags
from newton._src.solvers.feather_pgs import solver_feather_pgs as solver_feather_pgs_module
from newton._src.solvers.feather_pgs._testing import kernel_overrides
from newton._src.solvers.feather_pgs.friction import friction_pair_candidate
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT,
    PGS_LOCAL_SOLVE_OWNER_GENERAL,
    PGS_LOCAL_SOLVE_OWNER_PAIR,
    PGS_LOCAL_SOLVE_OWNER_PAIR_RESIDUAL,
    PGS_LOCAL_SOLVE_OWNER_SINGLE,
    accumulate_group_diag_worlds,
    compute_composite_inertia,
)
from newton._src.solvers.feather_pgs.solver_feather_pgs import (
    _FeatherPGSExecutionPlan,
    _get_build_independent_sparse_contact_groups_kernel,
    _get_composite_inertia_warp_kernel,
    _get_direct_diagonal_inverse_mass_kernel,
    _get_hinv_jt_kernel,
    _get_mark_independent_sparse_contact_candidates_kernel,
    _get_partitioned_inverse_dynamics_kernels,
    _get_pgs_solve_sparse_diagonal_kernel,
)
from newton.solvers import SolverFeatherPGS
from newton.tests.feather_pgs._fixtures import _build_mixed_response_model, set_torsion_radius
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices


class TestFeatherPGSCompositeInertia(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "warp-parallel composite inertia requires CUDA")
    def test_warp_reduction_matches_scalar_branched_trees(self):
        device = wp.get_cuda_device()
        articulation_start = wp.array([0, 4, 7], dtype=wp.int32, device=device)
        articulation_joint_end = wp.array([4, 7], dtype=wp.int32, device=device)
        joint_ancestor = wp.array([-1, 0, 0, 2, -1, 4, 5], dtype=wp.int32, device=device)
        joint_child = wp.array(np.arange(7, dtype=np.int32), dtype=wp.int32, device=device)
        body_inertia = wp.array(
            np.random.default_rng(42).normal(size=(7, 6, 6)).astype(np.float32),
            dtype=wp.spatial_matrix,
            device=device,
        )

        scalar = wp.empty_like(body_inertia)
        wp.launch(
            compute_composite_inertia,
            dim=2,
            inputs=[
                articulation_start,
                articulation_joint_end,
                wp.ones(2, dtype=wp.int32, device=device),
                joint_ancestor,
                joint_child,
                body_inertia,
            ],
            outputs=[scalar],
            device=device,
        )

        parallel = wp.empty_like(body_inertia)
        warps_per_block = 4
        kernel = _get_composite_inertia_warp_kernel(device.arch, warps_per_block)
        wp.launch_tiled(
            kernel,
            dim=[1],
            inputs=[
                2,
                wp.array([0, 1], dtype=wp.int32, device=device),
                articulation_start,
                articulation_joint_end,
                joint_ancestor,
                joint_child,
                body_inertia,
                wp.ones(2, dtype=wp.int32, device=device),
            ],
            outputs=[parallel],
            block_dim=32 * warps_per_block,
            device=device,
        )
        wp.synchronize_device(device)

        np.testing.assert_array_equal(parallel.numpy(), scalar.numpy())

    @unittest.skipUnless(wp.is_cuda_available(), "warp-parallel composite inertia requires CUDA")
    def test_compact_reduction_matches_assembled_inertia(self):
        device = wp.get_cuda_device()
        articulation_start = wp.array([0, 4, 7], dtype=wp.int32, device=device)
        articulation_joint_end = wp.array([4, 7], dtype=wp.int32, device=device)
        joint_ancestor = wp.array([-1, 0, 0, 2, -1, 4, 5], dtype=wp.int32, device=device)
        joint_child = wp.array(np.arange(7, dtype=np.int32), dtype=wp.int32, device=device)

        rng = np.random.default_rng(43)
        mass = rng.uniform(0.1, 10.0, size=7).astype(np.float32)
        com = rng.normal(size=(7, 3)).astype(np.float32)
        inertia_origin = rng.normal(size=(7, 3, 3)).astype(np.float32)
        terms = np.concatenate((com, inertia_origin.reshape(7, 9)), axis=1)
        assembled = np.zeros((7, 6, 6), dtype=np.float32)
        for body in range(7):
            com_cross = np.array(
                [
                    [0.0, -com[body, 2], com[body, 1]],
                    [com[body, 2], 0.0, -com[body, 0]],
                    [-com[body, 1], com[body, 0], 0.0],
                ],
                dtype=np.float32,
            )
            assembled[body, :3, :3] = mass[body] * np.eye(3, dtype=np.float32)
            assembled[body, :3, 3:] = -mass[body] * com_cross
            assembled[body, 3:, :3] = mass[body] * com_cross
            assembled[body, 3:, 3:] = inertia_origin[body]

        scalar = wp.empty((7,), dtype=wp.spatial_matrix, device=device)
        wp.launch(
            compute_composite_inertia,
            dim=2,
            inputs=[
                articulation_start,
                articulation_joint_end,
                wp.ones(2, dtype=wp.int32, device=device),
                joint_ancestor,
                joint_child,
                wp.array(assembled, dtype=wp.spatial_matrix, device=device),
            ],
            outputs=[scalar],
            device=device,
        )

        parallel = wp.empty_like(scalar)
        warps_per_block = 4
        kernel = _get_composite_inertia_warp_kernel(device.arch, warps_per_block, compact_terms=True)
        wp.launch_tiled(
            kernel,
            dim=[1],
            inputs=[
                2,
                wp.array([0, 1], dtype=wp.int32, device=device),
                articulation_start,
                articulation_joint_end,
                joint_ancestor,
                joint_child,
                wp.array(mass, dtype=wp.float32, device=device),
                wp.array(terms, dtype=wp.float32, device=device),
                wp.ones(2, dtype=wp.int32, device=device),
            ],
            outputs=[parallel],
            block_dim=32 * warps_per_block,
            device=device,
        )
        wp.synchronize_device(device)

        np.testing.assert_allclose(parallel.numpy(), scalar.numpy(), rtol=0.0, atol=2.0e-6)


def _build_fixed_base_star_model(
    device, num_branches=16, num_worlds=2, *, ground=False, armature=0.0, with_chain=False, ball_branch=False
):
    """Build fixed-base stars of independent prismatic branches (a structurally diagonal mass matrix).

    ``with_chain`` adds a two-link serial chain per world, so each world holds two response
    groups; ``ball_branch`` replaces the last branch by a BALL joint, whose DOFs are coupled.
    """
    star = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81 if ground else 0.0))
    base = star.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    joints = [
        star.add_joint_fixed(parent=-1, child=base, parent_xform=wp.transform((0.0, 0.0, 0.3), wp.quat_identity()))
    ]
    for branch in range(num_branches):
        child = star.add_link(mass=1.0 + 0.01 * branch, inertia=wp.mat33(np.eye(3)))
        if ground:
            star.add_shape_sphere(child, radius=0.05)
        parent_xform = wp.transform(wp.vec3(0.12 * branch, 0.0, 0.0), wp.quat_identity())
        if ball_branch and branch == num_branches - 1:
            joints.append(star.add_joint_ball(parent=base, child=child, parent_xform=parent_xform))
            continue
        joints.append(
            star.add_joint_prismatic(
                parent=base,
                child=child,
                axis=newton.Axis.Z,
                parent_xform=parent_xform,
                limit_lower=-0.4,
                limit_upper=0.05,
                armature=armature,
            )
        )
    star.add_articulation(joints)
    if with_chain:
        links = [star.add_link(mass=1.0, inertia=wp.mat33(np.eye(3))) for _ in range(2)]
        star.add_shape_sphere(links[1], radius=0.05)
        root = star.add_joint_revolute(
            -1, links[0], axis=newton.Axis.Y, parent_xform=wp.transform((-0.5, 0.0, 0.3), wp.quat_identity())
        )
        elbow = star.add_joint_revolute(
            links[0], links[1], axis=newton.Axis.Y, parent_xform=wp.transform((0.0, 0.0, -0.15), wp.quat_identity())
        )
        star.add_articulation([root, elbow])
    main = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81 if ground else 0.0))
    main.replicate(star, num_worlds, spacing=(3.0, 3.0, 0.0))
    if ground:
        main.add_ground_plane()
    return main.finalize(device=device)


def test_sleeping_diagonal_dynamics_skip_is_exact(test, device):
    """Skip the diagonal-mass dynamics of sleeping stars without changing any published state.

    Two solvers differ only in whether sleeping islands skip their dynamics; over a sleep, a
    force wake and a resettle, every state array stays bitwise equal.
    """
    runs = []
    for skip in (False, True):
        model = _build_fixed_base_star_model(device, num_branches=3, ground=True)
        solver = SolverFeatherPGS(
            model, solve_mode="matrix_free", enable_sleeping=True, sleep_quiet_time=0.05, articulation_constraint_max=64
        )
        test.assertTrue(solver._execution_plan.use_diagonal_mass(3))
        solver._sleeping.skip_dynamics = skip
        pipeline = newton.CollisionPipeline(model)
        runs.append([solver, pipeline, pipeline.contacts(), model.state(), model.state(), model.control()])
    slept = woke = False
    for step in range(500):
        for run in runs:
            solver, pipeline, contacts, state, out, control = run
            state.clear_forces()
            if 300 <= step < 306:
                forces = state.body_f.numpy()
                forces[1, 2] = 50.0
                state.body_f.assign(forces)
            pipeline.collide(state, contacts)
            solver.step(state, out, control, contacts, 1.0 / 120.0)
            run[3], run[4] = out, state
        for field in ("body_q", "body_qd", "joint_q", "joint_qd"):
            np.testing.assert_array_equal(
                getattr(runs[0][3], field).numpy(), getattr(runs[1][3], field).numpy(), err_msg=f"{step}: {field}"
            )
        awake = runs[1][0]._sleeping.art_awake.numpy()
        slept |= step < 300 and not awake.any()
        woke |= 300 <= step < 306 and bool(awake[0])
    test.assertTrue(slept, "the diagonal-mass articulations never slept")
    test.assertTrue(woke, "the external force did not wake the first articulation")
    for run in runs:
        run[0].check_constraint_capacity()


class TestFeatherPGSDiagonalMass(unittest.TestCase):
    pass


devices = get_cuda_test_devices()


for _name, _func in (("test_sleeping_diagonal_dynamics_skip_is_exact", test_sleeping_diagonal_dynamics_skip_is_exact),):
    add_function_test(TestFeatherPGSDiagonalMass, _name, _func, devices=devices)


DT = 1.0 / 60.0


# Per-articulation initial pose/velocity: [base, left branch, right branch].
INITIAL_JOINT_Q = (0.7, 0.3, -0.4)


INITIAL_JOINT_QD = (0.5, -0.2, 0.3)


def _build_model(device, num_worlds=2, ground=True):
    """Two-branch pendulum: base revolute joint plus two sibling branch links.

    The sibling branches are not ancestor-related, so H has structural zeros
    between their DOFs; with a ground plane the branch tips swing into contact.
    """
    env = newton.ModelBuilder()
    env.default_shape_cfg.density = 1000.0

    base = env.add_link()
    env.add_shape_box(base, hx=0.08, hy=0.08, hz=0.08)
    left = env.add_link()
    env.add_shape_box(left, hx=0.25, hy=0.05, hz=0.05)
    right = env.add_link()
    env.add_shape_box(right, hx=0.25, hy=0.05, hz=0.05)

    j_base = env.add_joint_revolute(
        -1,
        base,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.55), wp.quat_identity()),
        axis=newton.Axis.Y,
        target_ke=15.0,
        target_kd=1.5,
    )
    j_left = env.add_joint_revolute(
        base,
        left,
        parent_xform=wp.transform(wp.vec3(0.08, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.25, 0.0, 0.0), wp.quat_identity()),
        axis=newton.Axis.Y,
        target_ke=15.0,
        target_kd=1.5,
    )
    j_right = env.add_joint_revolute(
        base,
        right,
        parent_xform=wp.transform(wp.vec3(-0.08, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.25, 0.0, 0.0), wp.quat_identity()),
        axis=newton.Axis.Y,
        target_ke=15.0,
        target_kd=1.5,
    )
    env.add_articulation([j_base, j_left, j_right])

    builder = newton.ModelBuilder()
    builder.replicate(env, num_worlds)
    if ground:
        builder.add_ground_plane()
    return builder.finalize(device=device)


def _build_floating_model(device, num_worlds=2):
    """Floating two-link articulation with nonzero root and joint motion."""
    env = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    root = env.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
        mass=2.0,
        com=wp.vec3(0.07, -0.03, 0.02),
        inertia=wp.mat33(0.03, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 0.05),
    )
    child = env.add_link(
        xform=wp.transform(wp.vec3(0.3, 0.0, 1.0), wp.quat_identity()),
        mass=1.0,
        com=wp.vec3(0.04, 0.01, -0.02),
        inertia=wp.mat33(0.01, 0.0, 0.0, 0.0, 0.012, 0.0, 0.0, 0.0, 0.014),
    )
    free = env.add_joint_free(root)
    hinge = env.add_joint_revolute(
        root,
        child,
        parent_xform=wp.transform(wp.vec3(0.2, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.1, 0.0, 0.0), wp.quat_identity()),
        axis=newton.Axis.Y,
    )
    env.add_articulation([free, hinge])
    builder = newton.ModelBuilder()
    builder.replicate(env, num_worlds)
    return builder.finalize(device=device)


def _make_floating_state(model):
    state = model.state()
    qd = state.joint_qd.numpy()
    for articulation in range(model.articulation_count):
        start = articulation * 7
        qd[start : start + 7] = (0.4, -0.2, 0.1, 0.5, -0.7, 0.3, 0.8)
    state.joint_qd.assign(qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return state


def _run_floating_trajectory(model, solver, num_steps, *, invalidate_fk_id_each_step):
    state_0 = _make_floating_state(model)
    state_1 = model.state()
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()
    control = model.control()
    history = {name: [] for name in ("joint_q", "joint_qd", "body_q", "body_qd")}
    for _ in range(num_steps):
        if invalidate_fk_id_each_step:
            # Reset also forces a mass refresh, changing the comparison's cadence.
            solver.notify_state_changed()
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0
        for name, values in history.items():
            values.append(getattr(state_0, name).numpy().copy())
    return {name: np.stack(values) for name, values in history.items()}


def _make_initial_state(model):
    state = model.state()
    num_arts = model.articulation_count
    state.joint_q.assign(np.tile(np.asarray(INITIAL_JOINT_Q, dtype=np.float32), num_arts))
    state.joint_qd.assign(np.tile(np.asarray(INITIAL_JOINT_QD, dtype=np.float32), num_arts))
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    return state


def _run_trajectory(model, solver, num_steps):
    state_0 = _make_initial_state(model)
    state_1 = model.state()
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()
    control = model.control()
    joint_q_history = []
    for _ in range(num_steps):
        collision_pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0
        joint_q_history.append(state_0.joint_q.numpy().copy())
    return np.stack(joint_q_history)


def _run_state_trajectory(model, solver, num_steps, *, invalidate_fk_id_each_step):
    state_0 = _make_initial_state(model)
    state_1 = model.state()
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()
    control = model.control()
    history = {name: [] for name in ("joint_q", "joint_qd", "body_q", "body_qd")}
    for _ in range(num_steps):
        if invalidate_fk_id_each_step:
            # Isolate FK/ID reuse while preserving mass factors and warm starts.
            solver.notify_state_changed()
        collision_pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0
        for name, values in history.items():
            values.append(getattr(state_0, name).numpy().copy())
    return {name: np.stack(values) for name, values in history.items()}


class TestFeatherPGSMassUpdateInterval(unittest.TestCase):
    @unittest.skipUnless(wp.is_cuda_available(), "FK/ID reuse requires CUDA")
    def test_split_fk_id_cache_does_not_alias_active_step_state(self):
        """Keep split-mode cached dynamics isolated from active solve scratch."""
        model = _build_floating_model("cuda:0")
        solver = SolverFeatherPGS(model, update_mass_matrix_interval=2, solve_mode="split")
        state_in = _make_floating_state(model)
        state_aug = solver._prepare_augmented_state(state_in, model.state(), model.control())
        cache = solver._fk_id_cache

        active_cache_pairs = (
            (state_aug._body_q_com, cache.body_q_com),
            (state_aug._joint_S_s, cache.joint_S_s),
            (state_aug._body_I_s, cache.body_I_s),
            (solver._body_inertia_terms, cache.body_inertia_terms),
            (state_aug._body_v_s, cache.body_v_s),
            (state_aug._body_f_s, cache.body_f_s),
            (state_aug._body_a_s, cache.body_a_s),
            (solver._articulation_origin, cache.articulation_origin),
        )
        for active, cached in active_cache_pairs:
            self.assertNotEqual(active.ptr, cached.ptr)

    @unittest.skipUnless(wp.is_cuda_available(), "matrix-free PGS requires CUDA")
    def test_matrix_free_fk_id_cache_keeps_zero_copy_path(self):
        """Keep matrix-free dynamics publication on its validated zero-copy path."""
        model = _build_floating_model("cuda:0")
        solver = SolverFeatherPGS(model, update_mass_matrix_interval=2, solve_mode="matrix_free")

        self.assertTrue(solver._fk_id_cache_enabled)
        self.assertFalse(solver._fk_id_cache_uses_snapshot)
        self.assertIsNone(solver._fk_id_cache)

    @unittest.skipUnless(wp.is_cuda_available(), "FK/ID launch requires CUDA")
    def test_fk_id_launch_storage_exists_when_reuse_is_disabled(self):
        """Keep the unconditional FK/ID launch output valid without reuse."""
        model = _build_model("cuda:0", num_worlds=1, ground=False)
        solver = SolverFeatherPGS(model, solve_mode="matrix_free", enable_joint_velocity_limits=True)
        self.assertFalse(solver._fk_id_cache_enabled)
        self.assertEqual(solver._fk_id_cache_valid.shape[0], model.articulation_count)

        state_in = _make_initial_state(model)
        state_out = model.state()
        solver.step(state_in, state_out, model.control(), None, DT)
        wp.synchronize()

    @unittest.skipUnless(wp.is_cuda_available(), "FK/ID reuse requires CUDA")
    def test_cached_fk_id_matches_forced_recomputation(self):
        """Match forced recomputation across cached fixed-base steps."""
        trajectories = []
        for invalidate_fk_id_each_step in (False, True):
            model = _build_model("cuda:0", ground=False)
            solver = SolverFeatherPGS(model, update_mass_matrix_interval=2)
            trajectories.append(
                _run_state_trajectory(
                    model, solver, num_steps=20, invalidate_fk_id_each_step=invalidate_fk_id_each_step
                )
            )

        for name in trajectories[0]:
            np.testing.assert_allclose(
                trajectories[0][name], trajectories[1][name], rtol=0.0, atol=2.0e-6, err_msg=name
            )

    @unittest.skipUnless(wp.is_cuda_available(), "FK/ID reuse requires CUDA")
    def test_cached_fk_id_matches_forced_recomputation_for_floating_base(self):
        """Match forced recomputation across cached floating-base steps."""
        trajectories = []
        for invalidate_fk_id_each_step in (False, True):
            model = _build_floating_model("cuda:0")
            solver = SolverFeatherPGS(model, update_mass_matrix_interval=2)
            trajectories.append(
                _run_floating_trajectory(
                    model, solver, num_steps=20, invalidate_fk_id_each_step=invalidate_fk_id_each_step
                )
            )

        for name in trajectories[0]:
            np.testing.assert_allclose(
                trajectories[0][name], trajectories[1][name], rtol=0.0, atol=2.0e-6, err_msg=name
            )

    @unittest.skipUnless(wp.is_cuda_available(), "parallel compact inertia refresh requires CUDA")
    def test_parallel_compact_refresh_matches_serial_trajectory(self):
        trajectories = []
        for use_parallel_streams in (False, True):
            model = _build_model("cuda:0")
            solver = SolverFeatherPGS(
                model,
                update_mass_matrix_interval=1,
                double_buffer=False,
                use_parallel_streams=use_parallel_streams,
            )
            trajectories.append(_run_trajectory(model, solver, num_steps=20))

        np.testing.assert_allclose(trajectories[1], trajectories[0], rtol=0.0, atol=2.0e-6)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA graph capture requires CUDA")
    def test_captured_parallel_dynamics_matches_fused_trajectory(self):
        """Match captured parallel dynamics against eager fused execution."""
        graph_model = _build_model("cuda:0", ground=False)
        eager_model = _build_model("cuda:0", ground=False)
        graph_solver = SolverFeatherPGS(
            graph_model,
            update_mass_matrix_interval=1,
            double_buffer=False,
            use_parallel_streams=True,
        )
        eager_solver = SolverFeatherPGS(
            eager_model,
            update_mass_matrix_interval=1,
            double_buffer=False,
            use_parallel_streams=False,
        )
        graph_state, graph_out = _make_initial_state(graph_model), graph_model.state()
        eager_state, eager_out = _make_initial_state(eager_model), eager_model.state()
        graph_control, eager_control = graph_model.control(), eager_model.control()
        graph_pipeline = newton.CollisionPipeline(graph_model)
        eager_pipeline = newton.CollisionPipeline(eager_model)
        graph_contacts, eager_contacts = graph_pipeline.contacts(), eager_pipeline.contacts()

        def one_step(solver, state_in, state_out, control, contacts):
            solver.step(state_in, state_out, control, contacts, DT)
            wp.copy(state_in.body_q, state_out.body_q)
            wp.copy(state_in.body_qd, state_out.body_qd)
            wp.copy(state_in.joint_q, state_out.joint_q)
            wp.copy(state_in.joint_qd, state_out.joint_qd)

        one_step(graph_solver, graph_state, graph_out, graph_control, graph_contacts)
        one_step(eager_solver, eager_state, eager_out, eager_control, eager_contacts)
        with wp.ScopedCapture("cuda:0") as capture:
            one_step(graph_solver, graph_state, graph_out, graph_control, graph_contacts)

        for _ in range(20):
            wp.capture_launch(capture.graph)
            one_step(eager_solver, eager_state, eager_out, eager_control, eager_contacts)

        for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
            np.testing.assert_allclose(
                getattr(graph_state, name).numpy(),
                getattr(eager_state, name).numpy(),
                rtol=0.0,
                atol=2.0e-6,
                err_msg=name,
            )

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA graph capture requires CUDA")
    def test_captured_two_substep_fk_id_reuse_matches_eager(self):
        """Replay two captured substeps without erasing the cached body forces.

        The solver allocates its augmented dynamics buffers on the first step. When that first step runs inside a
        CUDA graph capture, a zero-initialised allocation records a memset that replays before every frame; the
        FK/ID reuse path then consumes cleared body forces while its cache still reads valid.
        """
        warm_model = _build_model("cuda:0", ground=False)
        warm_solver = SolverFeatherPGS(warm_model, update_mass_matrix_interval=2, solve_mode="matrix_free")
        warm_solver.step(_make_initial_state(warm_model), warm_model.state(), warm_model.control(), None, DT)
        wp.synchronize_device("cuda:0")

        graph_model = _build_model("cuda:0", ground=False)
        eager_model = _build_model("cuda:0", ground=False)
        # matrix_free keeps the zero-copy FK/ID reuse path; split restores its snapshot and would mask the defect.
        graph_solver = SolverFeatherPGS(graph_model, update_mass_matrix_interval=2, solve_mode="matrix_free")
        eager_solver = SolverFeatherPGS(eager_model, update_mass_matrix_interval=2, solve_mode="matrix_free")
        self.assertTrue(graph_solver._fk_id_cache_enabled)
        self.assertFalse(graph_solver._fk_id_cache_uses_snapshot)
        graph_a, graph_b = _make_initial_state(graph_model), graph_model.state()
        eager_a, eager_b = _make_initial_state(eager_model), eager_model.state()
        graph_control, eager_control = graph_model.control(), eager_model.control()

        def two_substeps(solver, state_in, state_out, control):
            # Keep one input state object so the FK/ID cache stays valid across substeps and frames.
            for _ in range(2):
                solver.step(state_in, state_out, control, None, DT)
                for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
                    wp.copy(getattr(state_in, name), getattr(state_out, name))

        # The very first launches of graph_solver happen inside the capture.
        with wp.ScopedCapture("cuda:0") as capture:
            two_substeps(graph_solver, graph_a, graph_b, graph_control)
        for _ in range(64):
            wp.capture_launch(capture.graph)
            two_substeps(eager_solver, eager_a, eager_b, eager_control)
        wp.synchronize_device("cuda:0")

        for name in ("joint_q", "joint_qd"):
            captured = getattr(graph_a, name).numpy()
            self.assertTrue(np.isfinite(captured).all(), f"{name} became non-finite under graph replay")
            np.testing.assert_allclose(captured, getattr(eager_a, name).numpy(), rtol=0.0, atol=2.0e-6, err_msg=name)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA graph capture requires CUDA")
    def test_first_step_capture_keeps_fk_id_cache_between_graph_launches(self):
        """Keep the FK/ID cache across launches of a graph that captures the solver's first step.

        Graph allocations are released and reacquired on every launch, so buffers that carry data between steps
        must exist before capture. Scratch released earlier in the frame makes the reacquired memory alias it.
        """
        graph_model = _build_model("cuda:0", ground=False)
        eager_model = _build_model("cuda:0", ground=False)
        graph_solver = SolverFeatherPGS(graph_model, solve_mode="matrix_free")
        eager_solver = SolverFeatherPGS(eager_model, solve_mode="matrix_free")
        self.assertTrue(graph_solver._fk_id_cache_enabled)
        graph_a, graph_b = _make_initial_state(graph_model), graph_model.state()
        eager_a, eager_b = _make_initial_state(eager_model), eager_model.state()
        graph_control, eager_control = graph_model.control(), eager_model.control()

        def two_substeps(solver, state_in, state_out, control):
            for _ in range(2):
                solver.step(state_in, state_out, control, None, DT)
                for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
                    wp.copy(getattr(state_in, name), getattr(state_out, name))

        # The very first launches of graph_solver happen inside the capture.
        with wp.ScopedCapture("cuda:0") as capture:
            scratch = wp.full(1000, float("nan"), dtype=wp.float32, device="cuda:0")
            del scratch
            two_substeps(graph_solver, graph_a, graph_b, graph_control)
        for _ in range(16):
            wp.capture_launch(capture.graph)
            two_substeps(eager_solver, eager_a, eager_b, eager_control)

        for name in ("joint_q", "joint_qd"):
            captured = getattr(graph_a, name).numpy()
            self.assertTrue(np.isfinite(captured).all(), f"{name} became non-finite under graph replay")
            np.testing.assert_allclose(captured, getattr(eager_a, name).numpy(), rtol=0.0, atol=2.0e-6, err_msg=name)

    def test_interval_two_contact_trajectory_stays_close_to_reference(self):
        device = wp.get_device()
        history = {}
        for interval in (1, 2):
            model = _build_model(device)
            solver = SolverFeatherPGS(model, update_mass_matrix_interval=interval)
            history[interval] = _run_trajectory(model, solver, num_steps=60)

        self.assertTrue(np.isfinite(history[2]).all(), "interval=2 trajectory diverged to non-finite values")
        # The trajectories are intentionally NOT bit-identical (stale H between
        # mass updates is the optimization); they must stay close on a short
        # contact-rich horizon.
        drift = np.abs(history[2] - history[1]).max()
        self.assertLess(drift, 0.05, f"interval=2 drifted {drift} rad from the interval=1 reference")
        # Sanity: the scene actually moved, so the comparison is not vacuous.
        moved = np.abs(history[1][-1] - np.tile(np.asarray(INITIAL_JOINT_Q, dtype=np.float32), 2)).max()
        self.assertGreater(moved, 1.0e-3)

    def test_global_flag_cadence_bakes_one_zero_pattern(self):
        device = wp.get_device()
        model = _build_model(device)
        solver = SolverFeatherPGS(model, update_mass_matrix_interval=2)
        state_0 = _make_initial_state(model)
        state_1 = model.state()
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()
        control = model.control()

        expected_masks = ([1, 1], [0, 0], [1, 1], [0, 0])
        for step_index, expected in enumerate(expected_masks):
            collision_pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, DT)
            state_0, state_1 = state_1, state_0
            self.assertEqual(
                solver._mass_update_mask.numpy().tolist(),
                expected,
                f"unexpected mass_update_mask after step {step_index}",
            )

    def test_model_change_request_refreshes_a_reuse_step(self):
        """Refresh every articulation after notification and consume each request."""
        model = _build_model(wp.get_device(), ground=False)
        solver = SolverFeatherPGS(model, update_mass_matrix_interval=2)
        state_0 = _make_initial_state(model)
        state_1 = model.state()
        collision_pipeline = newton.CollisionPipeline(model)
        contacts = collision_pipeline.contacts()
        control = model.control()

        collision_pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0
        solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
        collision_pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT)

        self.assertEqual(solver._mass_update_mask.numpy().tolist(), [1, 1])
        self.assertEqual(solver._mass_update_requested.numpy().tolist(), [0] * model.articulation_count)

    def test_teardown_waits_once_for_each_owned_stream(self):
        """Synchronize current stream owners and tolerate CUDA shutdown."""
        solver = object.__new__(SolverFeatherPGS)
        local = object()
        residual = object()
        dynamics = object()
        memset = object()
        size = object()
        solver._local_internal_stream = local
        solver._local_residual_stream = residual
        solver._local_pair_stream = local
        solver._global_inertia_stream = None
        solver._articulation_dynamics_stream = dynamics
        solver._memset_stream = memset
        solver._size_streams = {3: local, 6: size}

        def synchronize(stream):
            if stream is residual:
                raise RuntimeError("CUDA is shutting down")

        with mock.patch.object(wp, "synchronize_stream", side_effect=synchronize) as synchronize_stream:
            solver.__del__()
        self.assertEqual(
            synchronize_stream.call_args_list,
            [mock.call(local), mock.call(residual), mock.call(dynamics), mock.call(memset), mock.call(size)],
        )

        # Prevent a second meaningful synchronization when Python collects the
        # deliberately uninitialized instance after this regression returns.
        solver._local_internal_stream = None
        solver._local_residual_stream = None
        solver._local_pair_stream = None
        solver._articulation_dynamics_stream = None
        solver._memset_stream = None
        solver._size_streams = {}

    def test_teardown_waits_once_for_finalized_stream_device(self):
        """Synchronize each finalized-stream device while retaining live-stream waits."""
        device = wp.get_device()
        first = SimpleNamespace(device=device)
        second = SimpleNamespace(device=device)
        live = object()
        solver = object.__new__(SolverFeatherPGS)
        solver._local_internal_stream = first
        solver._local_residual_stream = second
        solver._local_pair_stream = live
        solver._size_streams = {3: first}
        try:
            with (
                mock.patch("gc.is_finalized", side_effect=lambda stream: stream is not live),
                mock.patch.object(wp, "synchronize_device") as synchronize_device,
                mock.patch.object(wp, "synchronize_stream") as synchronize_stream,
            ):
                solver.__del__()
            synchronize_device.assert_called_once_with(device)
            synchronize_stream.assert_called_once_with(live)
        finally:
            solver._local_internal_stream = None
            solver._local_residual_stream = None
            solver._local_pair_stream = None
            solver._size_streams = {}

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA stream cleanup requires a GPU")
    def test_teardown_handles_already_finalized_stream_in_cycle(self):
        """Cyclic GC can finalize a stream before its solver without crashing cleanup."""
        # Isolate the native crash so a failure cannot abort the enclosing suite.
        code = """
import gc
import sys
import weakref
import warp as wp
from newton.solvers import SolverFeatherPGS
wp.init()
unraisable = []
sys.unraisablehook = unraisable.append
stream = wp.Stream("cuda:0")
solver = object.__new__(SolverFeatherPGS)
solver._local_internal_stream = stream
stream.solver_owner = solver
reference = weakref.ref(solver)
del stream, solver
gc.collect()
assert not unraisable, [str(error.exc_value) for error in unraisable]
assert reference() is None
print("CYCLE_CLEANUP_COMPLETE", flush=True)
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=False)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("CYCLE_CLEANUP_COMPLETE", result.stdout)
        self.assertNotIn("Warp CUDA error", result.stderr)

    def test_mass_refresh_has_no_obsolete_limit_count_state(self):
        solver = SolverFeatherPGS(_build_model(wp.get_device(), ground=False))
        for attribute in ("aug_limit_counts", "aug_prev_limit_counts", "limit_change_mask"):
            self.assertFalse(hasattr(solver, attribute), f"obsolete mass-refresh state {attribute!r} was restored")


def _build_model_mass_update_interval_upstream(device, num_worlds=2, ground=True):
    """Two-branch pendulum: base revolute joint plus two sibling branch links.

    The sibling branches are not ancestor-related, so the mass matrix has structural zeros
    between their DOFs; with a ground plane the branch tips swing into contact.
    """
    env = newton.ModelBuilder()
    env.default_shape_cfg.density = 1000.0
    base = env.add_link()
    env.add_shape_box(base, hx=0.08, hy=0.08, hz=0.08)
    left = env.add_link()
    env.add_shape_box(left, hx=0.25, hy=0.05, hz=0.05)
    right = env.add_link()
    env.add_shape_box(right, hx=0.25, hy=0.05, hz=0.05)
    drive = {"axis": newton.Axis.Y, "target_ke": 15.0, "target_kd": 1.5}
    j_base = env.add_joint_revolute(
        -1, base, parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.55), wp.quat_identity()), **drive
    )
    j_left = env.add_joint_revolute(
        base,
        left,
        parent_xform=wp.transform(wp.vec3(0.08, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(-0.25, 0.0, 0.0), wp.quat_identity()),
        **drive,
    )
    j_right = env.add_joint_revolute(
        base,
        right,
        parent_xform=wp.transform(wp.vec3(-0.08, 0.0, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.25, 0.0, 0.0), wp.quat_identity()),
        **drive,
    )
    env.add_articulation([j_base, j_left, j_right])
    builder = newton.ModelBuilder()
    builder.replicate(env, num_worlds)
    if ground:
        builder.add_ground_plane()
    return builder.finalize(device=device)


def _run_trajectory_mass_update_interval_upstream(model, solver, num_steps):
    state_0 = _make_initial_state(model)
    state_1 = model.state()
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    control = model.control()
    history = []
    for _ in range(num_steps):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0
        history.append(state_0.joint_q.numpy().copy())
    return np.stack(history)


def test_mass_refresh_cadence_follows_interval(test, device, solve_mode="matrix_free"):
    """Refresh every articulation's factorization on every ``interval``-th step only."""
    model = _build_model_mass_update_interval_upstream(device)
    solver = SolverFeatherPGS(model, solve_mode=solve_mode, update_mass_matrix_interval=2)
    state_0 = _make_initial_state(model)
    state_1 = model.state()
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    control = model.control()
    for step_index, expected in enumerate(([1, 1], [0, 0], [1, 1], [0, 0])):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0
        test.assertEqual(solver._mass_update_mask.numpy().tolist(), expected, f"step {step_index}")


def test_stale_factorization_stays_close_to_interval_one(test, device, solve_mode="matrix_free"):
    """Keep an interval-2 contact trajectory close to the every-step refresh."""
    history = {}
    for interval in (1, 2):
        model = _build_model_mass_update_interval_upstream(device)
        solver = SolverFeatherPGS(model, solve_mode=solve_mode, update_mass_matrix_interval=interval)
        history[interval] = _run_trajectory_mass_update_interval_upstream(model, solver, num_steps=60)
    test.assertTrue(np.isfinite(history[2]).all())
    # A stale factorization between refreshes is the intended trade-off, so the runs differ.
    test.assertLess(float(np.abs(history[2] - history[1]).max()), 0.05)
    moved = np.abs(history[1][-1] - np.tile(np.asarray(INITIAL_JOINT_Q, dtype=np.float32), 2)).max()
    test.assertGreater(moved, 1.0e-3)


def test_timestep_change_forces_a_refresh(test, device, solve_mode="matrix_free"):
    """Refresh the augmented mass matrix when the time step changes on a reuse step."""
    model = _build_model_mass_update_interval_upstream(device, ground=False)
    solver = SolverFeatherPGS(model, solve_mode=solve_mode, update_mass_matrix_interval=4)
    state_0 = _make_initial_state(model)
    state_1 = model.state()
    control = model.control()
    solver.step(state_0, state_1, control, None, DT)
    solver.step(state_1, state_0, control, None, DT)
    test.assertEqual(solver._mass_update_mask.numpy().tolist(), [0, 0])
    solver.step(state_0, state_1, control, None, 0.5 * DT)
    test.assertEqual(solver._mass_update_mask.numpy().tolist(), [1, 1])


def test_captured_two_substep_replay_matches_eager(test, device, solve_mode="matrix_free"):
    """Replay two captured substeps, including the first-step allocations, like eager stepping."""
    graph_model = _build_model_mass_update_interval_upstream(device, ground=False)
    eager_model = _build_model_mass_update_interval_upstream(device, ground=False)
    graph_solver = SolverFeatherPGS(graph_model, solve_mode=solve_mode, update_mass_matrix_interval=2)
    eager_solver = SolverFeatherPGS(eager_model, solve_mode=solve_mode, update_mass_matrix_interval=2)
    graph_a, graph_b = _make_initial_state(graph_model), graph_model.state()
    eager_a, eager_b = _make_initial_state(eager_model), eager_model.state()
    graph_control, eager_control = graph_model.control(), eager_model.control()

    def two_substeps(solver, state_in, state_out, control):
        for _ in range(2):
            solver.step(state_in, state_out, control, None, DT)
            for name in ("joint_q", "joint_qd", "body_q", "body_qd"):
                wp.copy(getattr(state_in, name), getattr(state_out, name))

    # The very first launches of graph_solver happen inside the capture.
    with wp.ScopedCapture(device) as capture:
        two_substeps(graph_solver, graph_a, graph_b, graph_control)
    for _ in range(64):
        wp.capture_launch(capture.graph)
        two_substeps(eager_solver, eager_a, eager_b, eager_control)
    for name in ("joint_q", "joint_qd"):
        captured = getattr(graph_a, name).numpy()
        test.assertTrue(np.isfinite(captured).all(), f"{name} became non-finite under graph replay")
        np.testing.assert_allclose(captured, getattr(eager_a, name).numpy(), rtol=0.0, atol=2.0e-6, err_msg=name)


class TestFeatherPGSMassUpdateIntervalUpstream(unittest.TestCase):
    pass


for _name in (
    "test_mass_refresh_cadence_follows_interval",
    "test_stale_factorization_stays_close_to_interval_one",
    "test_timestep_change_forces_a_refresh",
    "test_captured_two_substep_replay_matches_eager",
):
    add_function_test(TestFeatherPGSMassUpdateIntervalUpstream, _name, globals()[_name], devices=devices)
    split_devices = devices if _name == "test_captured_two_substep_replay_matches_eager" else get_test_devices()
    add_function_test(
        TestFeatherPGSMassUpdateIntervalUpstream,
        f"{_name}_split",
        globals()[_name],
        devices=split_devices,
        solve_mode="split",
    )


INITIAL_JOINT_Q_notify_inertial = 0.3


NEW_COM = (0.15, 0.0, 0.05)


def _build_model_notify_inertial(device, com=None):
    """Single-link pendulum on a Y-axis revolute joint, box COM at the origin.

    With the COM at the body origin (the pivot) gravity exerts no torque, so
    any post-randomization swing is attributable to the COM offset alone.
    """
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.density = 1000.0

    link = builder.add_link()
    builder.add_shape_box(link, hx=0.25, hy=0.05, hz=0.05)
    joint = builder.add_joint_revolute(
        -1,
        link,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.8), wp.quat_identity()),
        axis=newton.Axis.Y,
    )
    builder.add_articulation([joint])
    builder.joint_q[0] = INITIAL_JOINT_Q_notify_inertial
    model = builder.finalize(device=device)
    if com is not None:
        body_com = model.body_com.numpy()
        body_com[0] = com
        model.body_com.assign(body_com)
    return model


def _run_trajectory_notify_inertial(model, solver, num_steps):
    state_0 = model.state()
    state_1 = model.state()
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    collision_pipeline = newton.CollisionPipeline(model)
    contacts = collision_pipeline.contacts()
    control = model.control()
    joint_q_history = []
    for _ in range(num_steps):
        collision_pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, DT)
        state_0, state_1 = state_1, state_0
        joint_q_history.append(state_0.joint_q.numpy().copy())
    return np.stack(joint_q_history)


class TestFeatherPGSNotifyInertial(unittest.TestCase):
    def test_stepped_solver_releases_resources_without_cyclic_gc(self):
        """Release a stepped solver before cyclic GC can finalize its streams first."""
        gc_enabled = gc.isenabled()
        gc.disable()
        try:
            model = _build_model_notify_inertial(wp.get_device())
            solver = SolverFeatherPGS(model)
            reference = weakref.ref(solver)
            solver.step(model.state(), model.state(), model.control(), None, DT)
            del solver
            self.assertIsNone(reference(), "Stepping must not create a solver ownership cycle")
        finally:
            if gc_enabled:
                gc.enable()

    def test_step_refreshes_body_pose_after_generalized_coordinate_update(self):
        """A direct ``joint_q`` update must not require a caller-side FK pass.

        ``SolverFeatherPGS.step`` historically derives body poses from the
        generalized coordinates before inverse dynamics.  Reset and direct
        generalized-coordinate callers rely on that public step behavior.
        """
        device = wp.get_device()
        outputs = {}

        for caller_refreshes_fk in (False, True):
            model = _build_model_notify_inertial(device, com=NEW_COM)
            state_0 = model.state()
            state_1 = model.state()

            joint_q = state_0.joint_q.numpy()
            joint_q[0] = 1.1
            state_0.joint_q.assign(joint_q)
            stale_body_q = state_0.body_q.numpy().copy()

            if caller_refreshes_fk:
                newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)

            SolverFeatherPGS(model, solve_mode="split").step(
                state_0,
                state_1,
                model.control(),
                None,
                DT,
            )
            outputs[caller_refreshes_fk] = {
                "joint_q": state_1.joint_q.numpy().copy(),
                "joint_qd": state_1.joint_qd.numpy().copy(),
                "body_q": state_0.body_q.numpy().copy(),
                "stale_body_q": stale_body_q,
            }

        self.assertFalse(np.allclose(outputs[False]["stale_body_q"], outputs[True]["body_q"]))
        np.testing.assert_allclose(outputs[False]["body_q"], outputs[True]["body_q"], rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(outputs[False]["joint_q"], outputs[True]["joint_q"], rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(outputs[False]["joint_qd"], outputs[True]["joint_qd"], rtol=0.0, atol=1.0e-6)

    def test_notify_refreshes_baked_com_and_inertia_buffers(self):
        """Verify BODY_INERTIAL_PROPERTIES re-derives body_X_com and body_I_m.

        Both buffers are baked from the model at construction time; writing
        model.body_com/body_mass alone must not change them until the solver
        is notified.
        """
        device = wp.get_device()
        model = _build_model_notify_inertial(device)
        solver = SolverFeatherPGS(model)
        stale_X_com = solver._body_X_com.numpy().copy()
        stale_I_m = solver._body_I_m.numpy().copy()

        body_com = model.body_com.numpy()
        body_com[0] = NEW_COM
        model.body_com.assign(body_com)
        body_mass = model.body_mass.numpy()
        body_mass[0] *= 2.0
        model.body_mass.assign(body_mass)

        np.testing.assert_array_equal(solver._body_X_com.numpy(), stale_X_com)
        solver.notify_model_changed(ModelFlags.BODY_INERTIAL_PROPERTIES)

        np.testing.assert_allclose(
            solver._body_X_com.numpy()[0][:3], np.asarray(NEW_COM, dtype=np.float32), rtol=0.0, atol=0.0
        )
        self.assertFalse(np.allclose(solver._body_I_m.numpy(), stale_I_m))
        self.assertEqual(solver._mass_update_requested.numpy()[0], 1)

    def test_com_change_with_notify_matches_freshly_built_solver(self):
        """Verify a mid-run COM change plus notify reproduces baked-COM dynamics.

        The trajectory after the change must match a solver constructed with
        the new COM already in the model and stepped from the same state; a
        stale solver (COM written without notify) must diverge, proving the
        comparison is not vacuous.
        """
        device = wp.get_device()
        pre_steps, post_steps = 30, 60

        # Reference: solver built with the new COM, run from the pre-change
        # state (which is static: zero torque while the COM sits at the pivot).
        reference_model = _build_model_notify_inertial(device, com=NEW_COM)
        reference_solver = SolverFeatherPGS(reference_model)
        reference = _run_trajectory_notify_inertial(reference_model, reference_solver, post_steps)

        histories = {}
        for notify in (True, False):
            model = _build_model_notify_inertial(device)
            solver = SolverFeatherPGS(model)
            _run_trajectory_notify_inertial(model, solver, pre_steps)
            body_com = model.body_com.numpy()
            body_com[0] = NEW_COM
            model.body_com.assign(body_com)
            if notify:
                solver.notify_model_changed(ModelFlags.BODY_INERTIAL_PROPERTIES)
            histories[notify] = _run_trajectory_notify_inertial(model, solver, post_steps)

        np.testing.assert_allclose(histories[True], reference, rtol=0.0, atol=1.0e-5)
        stale_drift = np.abs(histories[False] - reference).max()
        self.assertGreater(stale_drift, 1.0e-2, "stale-solver trajectory should diverge without notify")

    def test_joint_frame_change_with_notify_matches_freshly_built_solver(self):
        """Refresh cached mass factors on JOINT_PROPERTIES, eagerly and under graph replay."""
        device = wp.get_device()
        interval = 100
        modes = ["split", "matrix_free"] if device.is_cuda else ["split"]

        def shift_child_frame(model):
            joint_X_c = model.joint_X_c.numpy()
            joint_X_c[0, 0] = 0.5
            model.joint_X_c.assign(joint_X_c)

        for mode in modes:
            reference_model = _build_model_notify_inertial(device)
            shift_child_frame(reference_model)
            reference_solver = SolverFeatherPGS(reference_model, solve_mode=mode, update_mass_matrix_interval=interval)
            reference = _run_trajectory_notify_inertial(reference_model, reference_solver, 5)
            for capture in [False, True] if device.is_cuda else [False]:
                with self.subTest(mode=mode, capture=capture):
                    model = _build_model_notify_inertial(device)
                    solver = SolverFeatherPGS(model, solve_mode=mode, update_mass_matrix_interval=interval)
                    state_0, state_1 = model.state(), model.state()
                    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
                    control = model.control()

                    def substep(solver=solver, state_0=state_0, state_1=state_1, control=control):
                        solver.step(state_0, state_1, control, None, DT)
                        wp.copy(state_0.joint_q, state_1.joint_q)
                        wp.copy(state_0.joint_qd, state_1.joint_qd)

                    substep()
                    initial_q = model.joint_q.numpy().copy()
                    graph = None
                    if capture:
                        with wp.ScopedCapture(device=device) as graph:
                            substep()
                    state_0.joint_q.assign(initial_q)
                    state_0.joint_qd.zero_()
                    shift_child_frame(model)
                    solver.notify_model_changed(ModelFlags.JOINT_PROPERTIES)
                    history = []
                    for _ in range(5):
                        if graph is None:
                            substep()
                        else:
                            wp.capture_launch(graph.graph)
                        history.append(state_0.joint_q.numpy().copy())
                    np.testing.assert_allclose(np.asarray(history), reference, rtol=0.0, atol=1.0e-5)


def _check_joint_dof_edit_matches_fresh_solver(test, device, name: str, value: float, flag):
    """Assign one joint DOF property after a step, notify ``flag``, and compare the next step to a fresh solver."""
    solve_mode = "matrix_free" if wp.get_device(device).is_cuda else "split"
    # Graph capture needs a CUDA device.
    for capture in (False, True) if wp.get_device(device).is_cuda else (False,):
        with test.subTest(capture=capture):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            link = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
            joint = builder.add_joint_prismatic(-1, link, axis=newton.Axis.X, target_kd=10.0, armature=0.0)
            builder.add_articulation([joint])
            model = builder.finalize(device=device)
            solver = SolverFeatherPGS(model, solve_mode=solve_mode, update_mass_matrix_interval=100)
            state_0, state_1 = model.state(), model.state()
            state_0.joint_qd.fill_(1.0)
            control = model.control()
            solver.step(state_0, state_1, control, None, 0.01)
            if capture:
                with wp.ScopedCapture(device=device) as graph:
                    solver.step(state_0, state_1, control, None, 0.01)
            stale_qd = state_1.joint_qd.numpy().copy()
            getattr(model, name).assign([value])
            solver.notify_model_changed(flag)
            if capture:
                wp.capture_launch(graph.graph)
            else:
                solver.step(state_0, state_1, control, None, 0.01)

            fresh = SolverFeatherPGS(model, solve_mode=solve_mode, update_mass_matrix_interval=100)
            fresh_out = model.state()
            fresh.step(state_0, fresh_out, control, None, 0.01)
            test.assertFalse(np.allclose(fresh_out.joint_qd.numpy(), stale_qd, atol=1.0e-4))
            np.testing.assert_allclose(state_1.joint_qd.numpy(), fresh_out.joint_qd.numpy(), atol=1.0e-6)


def test_dof_force_flag_refreshes_drive_gains(test, device):
    """Fold a drive damping edit into the mass matrix on a JOINT_DOF_FORCE_PROPERTIES notify."""
    _check_joint_dof_edit_matches_fresh_solver(
        test, device, "joint_target_kd", 40.0, ModelFlags.JOINT_DOF_FORCE_PROPERTIES
    )


def test_dof_force_flag_refreshes_joint_damping(test, device):
    """Read joint damping at the next JOINT_DOF_FORCE_PROPERTIES notify."""
    _check_joint_dof_edit_matches_fresh_solver(
        test, device, "joint_damping", 20.0, ModelFlags.JOINT_DOF_FORCE_PROPERTIES
    )


def test_dof_inertial_flag_refreshes_armature(test, device):
    """Fold an armature edit into the mass matrix on a JOINT_DOF_INERTIAL_PROPERTIES notify."""
    _check_joint_dof_edit_matches_fresh_solver(
        test, device, "joint_armature", 1.0, ModelFlags.JOINT_DOF_INERTIAL_PROPERTIES
    )


def _check_damping_edit_matches_fresh_solver(test, device, replace: bool, flag, solve_mode: str):
    """Edit joint damping after a step, notify, and compare the next eager or captured step to a fresh solver."""
    # Graph capture needs a CUDA device.
    for capture in (False, True) if wp.get_device(device).is_cuda else (False,):
        with test.subTest(capture=capture):
            builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
            link = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
            joint = builder.add_joint_prismatic(-1, link, axis=newton.Axis.X, damping=1.0, armature=0.0)
            builder.add_articulation([joint])
            model = builder.finalize(device=device)
            solver = SolverFeatherPGS(model, solve_mode=solve_mode, update_mass_matrix_interval=100)
            state_0, state_1 = model.state(), model.state()
            state_0.joint_qd.fill_(1.0)
            control = model.control()
            solver.step(state_0, state_1, control, None, 0.01)
            if capture:
                with wp.ScopedCapture(device=device) as graph:
                    solver.step(state_0, state_1, control, None, 0.01)
            stale_qd = state_1.joint_qd.numpy().copy()
            if replace:
                model.joint_damping = wp.array([3.0], dtype=wp.float32, device=device)
            else:
                model.joint_damping.assign([3.0])
            solver.notify_model_changed(flag)
            if capture:
                wp.capture_launch(graph.graph)
            else:
                solver.step(state_0, state_1, control, None, 0.01)

            fresh = SolverFeatherPGS(model, solve_mode=solve_mode, update_mass_matrix_interval=100)
            fresh_out = model.state()
            fresh.step(state_0, fresh_out, control, None, 0.01)
            test.assertFalse(np.allclose(fresh_out.joint_qd.numpy(), stale_qd, atol=1.0e-4))
            np.testing.assert_allclose(state_1.joint_qd.numpy(), fresh_out.joint_qd.numpy(), atol=1.0e-6)


def _check_friction_edit_matches_fresh_solver(test, device, replace: bool, solve_mode: str):
    """Edit shape friction after a step, notify, and compare the next eager or captured step to a fresh solver."""
    # Graph capture needs a CUDA device.
    for capture in (False, True) if wp.get_device(device).is_cuda else (False,):
        with test.subTest(capture=capture):
            builder = newton.ModelBuilder()
            builder.default_shape_cfg.mu = 0.0
            builder.add_ground_plane()
            body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
            builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
            model = builder.finalize(device=device)
            solver = SolverFeatherPGS(model, solve_mode=solve_mode)
            state_0, state_1 = model.state(), model.state()
            state_0.joint_qd.assign([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
            pipeline = newton.CollisionPipeline(model)
            contacts = pipeline.contacts()
            pipeline.collide(state_0, contacts)
            control = model.control()
            solver.step(state_0, state_1, control, contacts, DT)
            if capture:
                with wp.ScopedCapture(device=device) as graph:
                    solver.step(state_0, state_1, control, contacts, DT)
            mu = np.full(model.shape_count, 1.0, dtype=np.float32)
            if replace:
                model.shape_material_mu = wp.array(mu, dtype=wp.float32, device=device)
            else:
                model.shape_material_mu.assign(mu)
            solver.notify_model_changed(ModelFlags.SHAPE_PROPERTIES)
            if capture:
                wp.capture_launch(graph.graph)
            else:
                solver.step(state_0, state_1, control, contacts, DT)

            fresh = SolverFeatherPGS(model, solve_mode=solve_mode)
            fresh_out = model.state()
            fresh.step(state_0, fresh_out, control, contacts, DT)
            test.assertLess(float(fresh_out.joint_qd.numpy()[0]), 0.95)
            np.testing.assert_allclose(state_1.joint_qd.numpy(), fresh_out.joint_qd.numpy(), atol=1.0e-5)


def test_replaced_joint_damping_matches_fresh_solver(test, device, solve_mode="matrix_free"):
    """Read a joint damping array replaced on the model at the next JOINT_DOF_PROPERTIES notify."""
    _check_damping_edit_matches_fresh_solver(test, device, True, ModelFlags.JOINT_DOF_PROPERTIES, solve_mode)


def test_assigned_joint_damping_matches_fresh_solver(test, device, solve_mode="matrix_free"):
    """Read joint damping modified in place at the next JOINT_DOF_PROPERTIES notify."""
    _check_damping_edit_matches_fresh_solver(test, device, False, ModelFlags.JOINT_DOF_PROPERTIES, solve_mode)


def test_dof_force_flag_reads_replaced_joint_damping(test, device, solve_mode="matrix_free"):
    """Read a joint damping array replaced on the model at the next JOINT_DOF_FORCE_PROPERTIES notify."""
    _check_damping_edit_matches_fresh_solver(test, device, True, ModelFlags.JOINT_DOF_FORCE_PROPERTIES, solve_mode)


def test_replaced_shape_friction_matches_fresh_solver(test, device, solve_mode="matrix_free"):
    """Read a friction array replaced on the model at the next SHAPE_PROPERTIES notify."""
    _check_friction_edit_matches_fresh_solver(test, device, replace=True, solve_mode=solve_mode)


def test_assigned_shape_friction_matches_fresh_solver(test, device, solve_mode="matrix_free"):
    """Read friction modified in place at the next SHAPE_PROPERTIES notify."""
    _check_friction_edit_matches_fresh_solver(test, device, replace=False, solve_mode=solve_mode)


class TestFeatherPGSNarrowJointDofFlags(unittest.TestCase):
    pass


for _name in (
    "test_dof_force_flag_refreshes_drive_gains",
    "test_dof_force_flag_refreshes_joint_damping",
    "test_dof_inertial_flag_refreshes_armature",
):
    add_function_test(TestFeatherPGSNarrowJointDofFlags, _name, globals()[_name], devices=get_test_devices())


class TestFeatherPGSNotifyReplacedArrays(unittest.TestCase):
    pass


for _name in (
    "test_replaced_joint_damping_matches_fresh_solver",
    "test_assigned_joint_damping_matches_fresh_solver",
    "test_dof_force_flag_reads_replaced_joint_damping",
    "test_replaced_shape_friction_matches_fresh_solver",
    "test_assigned_shape_friction_matches_fresh_solver",
):
    add_function_test(TestFeatherPGSNotifyReplacedArrays, _name, globals()[_name], devices=get_cuda_test_devices())
    add_function_test(
        TestFeatherPGSNotifyReplacedArrays,
        f"{_name}_split",
        globals()[_name],
        devices=get_test_devices(),
        solve_mode="split",
    )


INITIAL_JOINT_Q_notify_inertial_upstream = 0.3


def _build_model_notify_inertial_upstream(device, com=None):
    """Single-link pendulum on a Y-axis revolute joint, box COM at the pivot.

    With the COM at the pivot gravity exerts no torque, so any swing after a COM change is
    attributable to the change alone.
    """
    builder = newton.ModelBuilder()
    builder.default_shape_cfg.density = 1000.0
    link = builder.add_link()
    builder.add_shape_box(link, hx=0.25, hy=0.05, hz=0.05)
    joint = builder.add_joint_revolute(
        -1,
        link,
        parent_xform=wp.transform(wp.vec3(0.0, 0.0, 0.8), wp.quat_identity()),
        axis=newton.Axis.Y,
    )
    builder.add_articulation([joint])
    builder.joint_q[0] = INITIAL_JOINT_Q_notify_inertial_upstream
    model = builder.finalize(device=device)
    if com is not None:
        body_com = model.body_com.numpy()
        body_com[0] = com
        model.body_com.assign(body_com)
    return model


def test_step_refreshes_reused_state_after_joint_q_write(test, device, solve_mode="matrix_free"):
    """Pick up a joint_q write into a state the solver produced on the previous step."""
    model = _build_model_notify_inertial_upstream(device)
    solver = SolverFeatherPGS(model, solve_mode=solve_mode)
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    solver.step(state_0, state_1, control, None, DT)
    joint_q = state_1.joint_q.numpy()
    joint_q[0] = -0.9
    state_1.joint_q.assign(joint_q)
    solver.step(state_1, state_0, control, None, DT)

    reference_model = _build_model_notify_inertial_upstream(device)
    reference_state = reference_model.state()
    reference_state.joint_q.assign(state_1.joint_q)
    reference_state.joint_qd.assign(state_1.joint_qd)
    reference_out = reference_model.state()
    SolverFeatherPGS(reference_model, solve_mode=solve_mode).step(
        reference_state, reference_out, reference_model.control(), None, DT
    )
    np.testing.assert_allclose(state_0.joint_q.numpy(), reference_out.joint_q.numpy(), rtol=0.0, atol=1.0e-6)
    np.testing.assert_allclose(state_0.body_q.numpy(), reference_out.body_q.numpy(), rtol=0.0, atol=1.0e-6)


def test_kinematic_flag_change_is_picked_up(test, device, solve_mode="matrix_free"):
    """Hold a body still once it is flagged kinematic and the solver is notified."""
    model = _build_model_notify_inertial_upstream(device, com=NEW_COM)
    solver = SolverFeatherPGS(model, solve_mode=solve_mode)
    state_0, state_1 = model.state(), model.state()
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    control = model.control()
    flags = model.body_flags.numpy()
    flags[0] = int(newton.BodyFlags.KINEMATIC)
    model.body_flags.assign(flags)
    solver.notify_model_changed(ModelFlags.BODY_PROPERTIES)
    for _ in range(30):
        solver.step(state_0, state_1, control, None, DT)
        state_0, state_1 = state_1, state_0
    test.assertAlmostEqual(float(state_0.joint_q.numpy()[0]), INITIAL_JOINT_Q_notify_inertial_upstream, places=5)


class TestFeatherPGSNotifyInertialUpstream(unittest.TestCase):
    pass


for _name in (
    "test_step_refreshes_reused_state_after_joint_q_write",
    "test_kinematic_flag_change_is_picked_up",
):
    add_function_test(TestFeatherPGSNotifyInertialUpstream, _name, globals()[_name], devices=devices)
    add_function_test(
        TestFeatherPGSNotifyInertialUpstream,
        f"{_name}_split",
        globals()[_name],
        devices=get_test_devices(),
        solve_mode="split",
    )


WORLD_COUNT = 3


GEOMETRY_FIELDS = (
    "shape_body",
    "shape_transform",
    "shape_scale",
    "shape_type",
    "shape_source_ptr",
    "shape_margin",
    "shape_is_solid",
)


class TestFeatherPGSNotifyDevice(unittest.TestCase):
    def test_armature_notify_matches_host_reference_and_fresh_solver(self):
        """Refresh the effective armature and its size groups exactly as a host rebuild would."""
        device = wp.get_device()
        model = _build_model_notify_device(device)
        options = {"solve_mode": "matrix_free", "enable_joint_friction": True} if device.is_cuda else {}
        solver = SolverFeatherPGS(model, **options)
        rng = np.random.default_rng(0)
        model.joint_armature.assign(rng.uniform(0.0, 0.5, model.joint_dof_count).astype(np.float32))
        solver.notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)
        _assert_armature_matches(self, solver, SolverFeatherPGS(model, **options))

    def test_kinematic_notify_matches_host_reference_and_fresh_solver(self):
        """Give newly kinematic joints the large armature and drop it when they become dynamic again."""
        model = _build_model_notify_device(wp.get_device())
        solver = SolverFeatherPGS(model)
        model.joint_armature.assign(np.full(model.joint_dof_count, 0.25, dtype=np.float32))
        chain_link = int(model.joint_child.numpy()[1])
        for flag in (BodyFlags.KINEMATIC, BodyFlags.DYNAMIC):
            body_flags = model.body_flags.numpy()
            body_flags[chain_link] = int(flag)
            model.body_flags.assign(body_flags)
            solver.notify_model_changed(ModelFlags.BODY_PROPERTIES)
            _assert_armature_matches(self, solver, SolverFeatherPGS(model))

    def test_shape_notify_matches_host_reference(self):
        """Refresh body radii and retire exactly the anchors whose geometry changed."""
        model = _build_model_notify_device(wp.get_device())
        solver = SolverFeatherPGS(model, friction_anchor_beta=0.2)
        patches = solver._friction_patches
        rng = np.random.default_rng(1)
        seen = {name: getattr(model, name).numpy().copy() for name in GEOMETRY_FIELDS}
        for edit in ("shape_transform", "shape_scale", "shape_margin", "shape_body", "shape_collision_radius", None):
            _randomize_history(patches, model, rng)
            valid_before = patches.previous.valid.numpy()
            shape = int(rng.integers(model.shape_count))
            if edit == "shape_body":
                values = model.shape_body.numpy()
                values[shape] = (values[shape] + 1) % model.body_count
                model.shape_body.assign(values)
            elif edit is not None:
                values = getattr(model, edit).numpy()
                values[shape] = values[shape] * 1.5 + 0.01
                getattr(model, edit).assign(values)
            solver.notify_model_changed(ModelFlags.SHAPE_PROPERTIES)
            expected_radius, expected_valid, seen = _host_update_geometry(model, patches, seen, valid_before)
            np.testing.assert_array_equal(patches.body_radius.numpy(), expected_radius)
            np.testing.assert_array_equal(patches.previous.valid.numpy(), expected_valid)
        self.assertGreater(np.count_nonzero(patches.previous.valid.numpy()), 0)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA graph capture requires a CUDA device")
    def test_armature_and_shape_notify_capture_into_a_graph(self):
        """Replay a captured notify and pick up armature and geometry written after the capture."""
        device = wp.get_device("cuda:0")
        model = _build_model_notify_device(device)
        solver = SolverFeatherPGS(model, solve_mode="matrix_free", enable_joint_friction=True, friction_anchor_beta=0.2)
        patches = solver._friction_patches
        flags = (
            ModelFlags.JOINT_PROPERTIES
            | ModelFlags.JOINT_DOF_PROPERTIES
            | ModelFlags.BODY_INERTIAL_PROPERTIES
            | ModelFlags.SHAPE_PROPERTIES
        )
        with wp.ScopedCapture(device=device) as capture:
            solver.notify_model_changed(flags)
        seen = {name: getattr(model, name).numpy().copy() for name in GEOMETRY_FIELDS}
        rng = np.random.default_rng(2)
        _randomize_history(patches, model, rng)
        valid_before = patches.previous.valid.numpy()
        model.joint_armature.assign(rng.uniform(0.0, 0.5, model.joint_dof_count).astype(np.float32))
        model.shape_collision_radius.assign(model.shape_collision_radius.numpy() * 2.0)
        scale = model.shape_scale.numpy()
        scale[2] *= 1.25
        model.shape_scale.assign(scale)
        wp.capture_launch(capture.graph)
        _assert_armature_matches(
            self, solver, SolverFeatherPGS(model, solve_mode="matrix_free", enable_joint_friction=True)
        )
        expected_radius, expected_valid, _ = _host_update_geometry(model, patches, seen, valid_before)
        np.testing.assert_array_equal(patches.body_radius.numpy(), expected_radius)
        np.testing.assert_array_equal(patches.previous.valid.numpy(), expected_valid)
        self.assertLess(np.count_nonzero(expected_valid), np.count_nonzero(valid_before))

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA graph capture requires a CUDA device")
    def test_narrow_dof_flags_and_material_copies_capture_into_a_graph(self):
        """Replay a captured narrow-flag notify and pick up armature, damping, friction and materials edited in place."""
        device = wp.get_device("cuda:0")
        model = _build_model_notify_device(device)
        solver = SolverFeatherPGS(model, solve_mode="matrix_free", enable_joint_friction=True)
        flags = (
            ModelFlags.JOINT_DOF_INERTIAL_PROPERTIES
            | ModelFlags.JOINT_DOF_FORCE_PROPERTIES
            | ModelFlags.SHAPE_PROPERTIES
        )
        with wp.ScopedCapture(device=device) as capture:
            solver.notify_model_changed(flags)
        rng = np.random.default_rng(3)
        model.joint_armature.assign(rng.uniform(0.0, 0.5, model.joint_dof_count).astype(np.float32))
        damping = rng.uniform(0.1, 1.0, model.joint_dof_count).astype(np.float32)
        model.joint_damping.assign(damping)
        materials = {
            "shape_material_mu": solver._shape_mu,
            "shape_material_restitution": solver._shape_restitution,
            "shape_material_mu_torsional": solver._shape_mu_torsional,
            "shape_material_mu_rolling": solver._shape_mu_rolling,
        }
        edited = {}
        for name in materials:
            edited[name] = rng.uniform(0.01, 0.9, model.shape_count).astype(np.float32)
            getattr(model, name).assign(edited[name])
        fresh = SolverFeatherPGS(model, solve_mode="matrix_free", enable_joint_friction=True)
        revolute, _ = _friction_dofs(model)
        friction = np.zeros(model.joint_dof_count, dtype=np.float32)
        friction[revolute[2]] = -1.0
        model.joint_friction.assign(friction)
        wp.capture_launch(capture.graph)
        _assert_armature_matches(self, solver, fresh)
        np.testing.assert_array_equal(solver._passive_joint_damping.numpy()[: model.joint_dof_count], damping)
        for name, buffer in materials.items():
            np.testing.assert_array_equal(buffer.numpy()[: model.shape_count], edited[name], err_msg=name)
        np.testing.assert_array_equal(solver.joint_friction_invalid.numpy(), np.arange(WORLD_COUNT) == 2)
        flags = solver.joint_friction_invalid
        with self.assertRaises(AttributeError):
            solver.joint_friction_invalid = flags
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            solver.check_joint_friction()
        self.assertIs(solver.joint_friction_invalid, flags)
        self.assertFalse(flags.numpy().any())

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA graph capture requires a CUDA device")
    def test_captured_notify_flags_invalid_joint_friction(self):
        """Latch the world of each invalid coefficient on replay and report it outside capture."""
        device = wp.get_device("cuda:0")
        model = _build_model_notify_device(device)
        solver = SolverFeatherPGS(model, solve_mode="matrix_free", enable_joint_friction=True)
        with wp.ScopedCapture(device=device) as capture:
            solver.notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)
        revolute, free = _friction_dofs(model)
        cases = (
            (revolute[1], -1.0, "finite and non-negative"),
            (revolute[2], float("nan"), "finite and non-negative"),
            (revolute[0], float("inf"), "finite and non-negative"),
            (free[1], 0.5, "FREE"),
        )
        for dof, value, message in cases:
            with self.subTest(value=value, unsupported=dof in free.values()):
                friction = np.zeros(model.joint_dof_count, dtype=np.float32)
                friction[dof] = value
                model.joint_friction.assign(friction)
                wp.capture_launch(capture.graph)
                expected = _dof_world(model)[dof] == np.arange(WORLD_COUNT)
                np.testing.assert_array_equal(solver.joint_friction_invalid.numpy(), expected)
                with self.assertRaisesRegex(ValueError, message):
                    solver.check_joint_friction()
                self.assertFalse(solver.joint_friction_invalid.numpy().any())

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA graph capture requires a CUDA device")
    def test_invalid_joint_friction_stays_latched_until_reported(self):
        """Keep the flag through later valid replays and resets until the host reports it once."""
        device = wp.get_device("cuda:0")
        model = _build_model_notify_device(device)
        solver = SolverFeatherPGS(model, solve_mode="matrix_free", enable_joint_friction=True)
        with wp.ScopedCapture(device=device) as capture:
            solver.notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)
        revolute, _ = _friction_dofs(model)
        friction = np.zeros(model.joint_dof_count, dtype=np.float32)
        friction[revolute[1]] = -1.0
        model.joint_friction.assign(friction)
        wp.capture_launch(capture.graph)
        model.joint_friction.zero_()
        wp.capture_launch(capture.graph)
        solver.reset(model.state())
        np.testing.assert_array_equal(solver.joint_friction_invalid.numpy(), [False, True, False])
        with self.assertRaisesRegex(ValueError, r"captured notify in worlds \[1\]"):
            solver.check_joint_friction()
        solver.check_joint_friction()
        solver.notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)

    @unittest.skipUnless(wp.is_cuda_available(), "CUDA graph capture requires a CUDA device")
    def test_valid_joint_friction_replays_without_flags(self):
        """Leave the flag clear for valid coefficients, including zero and supported nonzero values."""
        device = wp.get_device("cuda:0")
        model = _build_model_notify_device(device)
        solver = SolverFeatherPGS(model, solve_mode="matrix_free", enable_joint_friction=True)
        with wp.ScopedCapture(device=device) as capture:
            solver.notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)
        revolute, _ = _friction_dofs(model)
        rng = np.random.default_rng(3)
        for _ in range(3):
            friction = np.zeros(model.joint_dof_count, dtype=np.float32)
            dofs = np.flatnonzero(_revolute_dof_mask(model))
            friction[dofs] = rng.uniform(0.0, 2.0, dofs.size).astype(np.float32)
            friction[revolute[0]] = 0.0
            model.joint_friction.assign(friction)
            wp.capture_launch(capture.graph)
            self.assertFalse(solver.joint_friction_invalid.numpy().any())
        solver.check_joint_friction()

    @unittest.skipUnless(wp.is_cuda_available(), "matrix-free joint friction requires CUDA")
    def test_eager_notify_raises_for_invalid_joint_friction(self):
        """Raise from an eager notify after applying it, and accept the next valid notify."""
        model = _build_model_notify_device(wp.get_device("cuda:0"))
        solver = SolverFeatherPGS(model, solve_mode="matrix_free", enable_joint_friction=True)
        revolute, _ = _friction_dofs(model)
        model.joint_armature.fill_(0.3)
        friction = np.zeros(model.joint_dof_count, dtype=np.float32)
        friction[revolute[2]] = -1.0
        model.joint_friction.assign(friction)
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            solver.notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)
        self.assertFalse(solver.joint_friction_invalid.numpy().any())
        self.assertTrue(np.all(solver._joint_armature_device.numpy() == np.float32(0.3)))
        model.joint_friction.zero_()
        solver.notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)

    @unittest.skipUnless(wp.is_cuda_available(), "matrix-free joint friction requires CUDA")
    def test_invalid_joint_friction_flags_the_owning_world(self):
        """Attribute loop-closing and global DOFs to their solver world in a sparse layout."""
        device = wp.get_device("cuda:0")
        model, dofs = _build_loop_model(device)
        solver = SolverFeatherPGS(model, solve_mode="matrix_free", enable_joint_friction=True)
        self.assertEqual(solver._world_count, 3)
        with wp.ScopedCapture(device=device) as capture:
            solver.notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)
        cases = (
            (dofs["loop"], 0.5, 2, "loop joint"),
            (dofs["chain"], -1.0, 2, "finite and non-negative"),
            (dofs["global"], -1.0, 0, "finite and non-negative"),
        )
        for dof, value, world, message in cases:
            with self.subTest(dof=dof, value=value):
                friction = np.zeros(model.joint_dof_count, dtype=np.float32)
                friction[dof] = value
                model.joint_friction.assign(friction)
                wp.capture_launch(capture.graph)
                np.testing.assert_array_equal(solver.joint_friction_invalid.numpy(), np.arange(3) == world)
                with self.assertRaisesRegex(ValueError, message):
                    solver.check_joint_friction()
                model.joint_friction.zero_()
                solver.notify_model_changed(ModelFlags.JOINT_DOF_PROPERTIES)


def _build_loop_model(device):
    """Loop-closed chains of different sizes in worlds 0 and 2, static geometry in world 1, and a global chain."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    for link_count in (3, 0, 2):
        builder.begin_world()
        if link_count == 0:
            builder.add_shape_box(-1, hx=0.1, hy=0.1, hz=0.1)
        else:
            links = [builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3))) for _ in range(link_count)]
            joints = [builder.add_joint_revolute(-1, link, axis=newton.Axis.Y) for link in links]
            builder.add_articulation(joints)
            loop = builder.add_joint_ball(links[0], links[-1])
        builder.end_world()
    global_link = builder.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    builder.add_articulation([builder.add_joint_revolute(-1, global_link, axis=newton.Axis.Y)])
    model = builder.finalize(device=device)
    joint_qd_start = model.joint_qd_start.numpy()
    joint_dofs = {"loop": loop, "chain": joints[1], "global": model.joint_count - 1}
    return model, {name: int(joint_qd_start[joint]) for name, joint in joint_dofs.items()}


def _build_model_notify_device(device):
    """Worlds with a free sphere, a 3-link and a 2-link revolute chain: three response sizes."""
    builder = newton.ModelBuilder()
    for world in range(WORLD_COUNT):
        builder.begin_world()
        sphere = builder.add_body(xform=wp.transform(wp.vec3(1.0, 0.0, 0.5), wp.quat_identity()))
        builder.add_shape_sphere(sphere, radius=0.1)
        for link_count, x in ((3, 0.0), (2, -1.0)):
            parent, joints = -1, []
            for link_index in range(link_count):
                link = builder.add_link()
                builder.add_shape_capsule(
                    link, radius=0.05, half_height=0.1, xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity())
                )
                joint_z = 1.0 if parent < 0 else 0.2
                joints.append(
                    builder.add_joint_revolute(
                        parent,
                        link,
                        parent_xform=wp.transform(wp.vec3(x if parent < 0 else 0.0, 0.0, joint_z), wp.quat_identity()),
                        axis=newton.Axis.Y,
                        armature=0.01 * (world + link_index + 1),
                    )
                )
                parent = link
            builder.add_articulation(joints)
        builder.end_world()
    return builder.finalize(device=device)


def _dof_world(model):
    joint_world = model.joint_world.numpy()
    return np.repeat(joint_world, np.diff(model.joint_qd_start.numpy()))


def _revolute_dof_mask(model):
    joint_type = model.joint_type.numpy()
    return np.repeat(joint_type == int(newton.JointType.REVOLUTE), np.diff(model.joint_qd_start.numpy()))


def _friction_dofs(model):
    """First revolute DOF and first free-joint DOF of each world."""
    world = _dof_world(model)
    free = np.repeat(model.joint_type.numpy() == int(newton.JointType.FREE), np.diff(model.joint_qd_start.numpy()))
    revolute = _revolute_dof_mask(model)
    return tuple(
        {w: int(np.flatnonzero(mask & (world == w))[0]) for w in range(WORLD_COUNT)} for mask in (revolute, free)
    )


def _assert_armature_matches(test, solver, fresh):
    """Compare the effective armature and grouped copies with the host rebuild and a fresh solver."""
    model = solver.model
    armature = model.joint_armature.numpy().copy()
    kinematic = (model.body_flags.numpy() & int(BodyFlags.KINEMATIC)) != 0
    joint_qd_start = model.joint_qd_start.numpy()
    for joint, child in enumerate(model.joint_child.numpy()):
        if kinematic[child]:
            armature[joint_qd_start[joint] : joint_qd_start[joint + 1]] = 1.0e10
    np.testing.assert_array_equal(solver._joint_armature_device.numpy(), armature)
    np.testing.assert_array_equal(solver._joint_armature_device.numpy(), fresh._joint_armature_device.numpy())
    plan = solver._model_plan
    test.assertEqual(len(solver._size_groups), 3)
    for size in solver._size_groups:
        expected = np.zeros((solver._n_arts_by_size[size], size), dtype=np.float32)
        for group, art in enumerate(np.flatnonzero(plan.response_dof_count == size)):
            start, count = plan.articulation_dof_start[art], plan.articulation_dof_count[art]
            expected[group, :count] = armature[start : start + count]
        np.testing.assert_array_equal(solver._R_by_size[size].numpy(), expected)
        np.testing.assert_array_equal(solver._R_by_size[size].numpy(), fresh._R_by_size[size].numpy())


def _randomize_history(patches, model, rng):
    """Fill the previous patch frame with valid anchors on random shapes and bodies, including -1 sentinels."""
    n = patches.previous.valid.shape[0]
    patches.previous.valid.fill_(1)
    for field, count in (("shape", model.shape_count), ("body", model.body_count)):
        for side in ("a", "b"):
            getattr(patches.previous, f"{field}_{side}").assign(rng.integers(-1, count, n).astype(np.int32))


def _host_update_geometry(model, patches, seen, valid):
    """Host reference for the friction-anchor geometry refresh: body radii and retired history."""
    geometry = {name: getattr(model, name).numpy().copy() for name in GEOMETRY_FIELDS}
    radii = np.zeros(model.body_count, dtype=np.float32)
    for body, radius, transform in zip(
        geometry["shape_body"], model.shape_collision_radius.numpy(), geometry["shape_transform"], strict=True
    ):
        if body >= 0:
            radii[body] = max(radii[body], radius + np.linalg.norm(transform[:3]))
    changed = np.zeros(model.shape_count, dtype=bool)
    for name, values in geometry.items():
        difference = values != seen[name]
        if difference.ndim > 1:
            difference = np.any(difference, axis=tuple(range(1, difference.ndim)))
        changed |= difference
    bodies = np.zeros(model.body_count, dtype=bool)
    for shape_body in (seen["shape_body"], geometry["shape_body"]):
        affected = shape_body[changed]
        bodies[affected[affected >= 0]] = True
    valid = valid.copy()
    previous = patches.previous
    for ids, mask in (
        (previous.shape_a.numpy(), changed),
        (previous.shape_b.numpy(), changed),
        (previous.body_a.numpy(), bodies),
        (previous.body_b.numpy(), bodies),
    ):
        hit = ids >= 0
        valid[hit & mask[np.maximum(ids, 0)]] = 0
    return radii, valid, geometry


def _run_mixed_response(
    kernel,
    *,
    warmstart,
    preelimination,
    dof_count=13,
    articulation_constraint_max=32,
    inactive_joint_limit_capacity=False,
    friction=0.0,
    restitution=0.0,
    tangential_velocity=0.0,
    static_plane=False,
    contact_regularization=0.0,
    friction_anchor_beta=None,
    model_kwargs=None,
):
    """Run a short mixed-contact trajectory with one H-inverse implementation."""
    model = _build_mixed_response_model(
        "cuda:0",
        dof_count=dof_count,
        friction=friction,
        restitution=restitution,
        static_plane=static_plane,
        **(model_kwargs or {}),
    )
    with kernel_overrides({"hinv_jt_kernel": kernel}):
        solver = SolverFeatherPGS(
            model,
            solve_mode="matrix_free",
            contact_warmstart=warmstart,
            enable_bilateral_preelimination=preelimination,
            contact_friction_scale=1.0 if friction > 0.0 else 0.0,
            enable_joint_limits=inactive_joint_limit_capacity,
            joint_limit_activation_gap=0.0,
            iterations=8,
            contact_regularization=contact_regularization,
            friction_anchor_beta=friction_anchor_beta,
            articulation_constraint_max=articulation_constraint_max,
            free_body_constraint_max=32,
        )
    state_in, state_out = model.state(), model.state()
    joint_qd = state_in.joint_qd.numpy()
    free_articulation = int(np.flatnonzero(solver._model_plan.is_free_rigid)[0])
    free_dof_start = int(solver._model_plan.articulation_dof_start[free_articulation])
    joint_qd[free_dof_start] = tangential_velocity
    joint_qd[free_dof_start + 2] = -3.0
    state_in.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)

    pipeline = newton.CollisionPipeline(
        model,
        broad_phase="nxn",
        reduce_contacts=False,
        contact_matching="latest",
    )
    contacts = pipeline.contacts()
    control = model.control()
    samples = []
    for _ in range(4):
        state_in.clear_forces()
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, 1.0 / 240.0)
        constraint_count = int(solver._constraint_count.numpy()[0])
        mf_count = int(solver._mf_constraint_count.numpy()[0])
        samples.append(
            (
                constraint_count,
                solver._diag.numpy()[0, :constraint_count].copy(),
                solver._impulses.numpy()[0, :constraint_count].copy(),
                state_out.joint_q.numpy().copy(),
                state_out.joint_qd.numpy().copy(),
                int(solver._local_solve_owner.numpy()[0]),
                solver._row_type.numpy()[0, :constraint_count].copy(),
                mf_count,
                solver._mf_impulses.numpy()[0, :mf_count].copy(),
                solver._mf_row_type.numpy()[0, :mf_count].copy(),
                solver._mf_row_parent.numpy()[0, :mf_count].copy(),
                solver._mf_J_a.numpy()[0, :mf_count].copy(),
                solver._mf_J_b.numpy()[0, :mf_count].copy(),
                solver._mf_MiJt_a.numpy()[0, :mf_count].copy(),
                solver._mf_MiJt_b.numpy()[0, :mf_count].copy(),
            )
        )
        state_in, state_out = state_out, state_in
    return solver, samples


def _assert_owner_parity(test, general, local):
    """Assert the local owner trajectory matches the general owner row for row."""
    test.assertEqual(len(general), len(local))
    test.assertGreater(general[0][0], 0, "mixed scene generated no dense constraint rows")
    for step, (expected, actual) in enumerate(zip(general, local, strict=True)):
        test.assertEqual(actual[0], expected[0], f"constraint count differed at step {step}")
        test.assertEqual(actual[7], expected[7], f"matrix-free row count differed at step {step}")
        np.testing.assert_array_equal(actual[6], expected[6], err_msg=f"row types differed at step {step}")
        np.testing.assert_array_equal(actual[9], expected[9], err_msg=f"matrix-free row types differed at step {step}")
        for label, expected_value, actual_value in zip(
            ("diagonal", "impulses", "joint_q", "joint_qd", "mf_impulses"),
            (*expected[1:5], expected[8]),
            (*actual[1:5], actual[8]),
            strict=True,
        ):
            np.testing.assert_allclose(
                actual_value,
                expected_value,
                rtol=5.0e-4,
                atol=2.0e-6,
                err_msg=f"{label} differed at step {step}",
            )


def _build_diagonal_mass_local_model(device):
    """Build two local-owner worlds whose only articulation has a diagonal mass matrix.

    World 0 rests a prismatic sphere on the ground beside a distant free box (paired owner); world 1 holds a
    contactless hinge against its upper limit (single owner).
    """
    world_builders = []
    sphere = newton.ModelBuilder()
    sphere_xform = wp.transform(wp.vec3(0.0, 0.0, 0.05), wp.quat_identity())
    sphere_body = sphere.add_link(xform=sphere_xform)
    sphere.add_shape_sphere(
        sphere_body, radius=0.05, cfg=newton.ModelBuilder.ShapeConfig(density=0.3 / (4.0 / 3.0 * np.pi * 0.05**3))
    )
    sphere.add_articulation(
        [sphere.add_joint_prismatic(parent=-1, child=sphere_body, axis=newton.Axis.Z, parent_xform=sphere_xform)]
    )
    box = sphere.add_body(xform=wp.transform(wp.vec3(1.0, 0.0, 0.5), wp.quat_identity()))
    sphere.add_shape_box(box, hx=0.05, hy=0.05, hz=0.05)
    world_builders.append(sphere)

    hinge = newton.ModelBuilder()
    hinge_body = hinge.add_link(
        com=wp.vec3(0.2, 0.0, 0.0), mass=0.5, inertia=wp.mat33(np.eye(3, dtype=np.float32) * 1.0e-3), lock_inertia=True
    )
    hinge.add_articulation(
        [
            hinge.add_joint_revolute(
                parent=-1,
                child=hinge_body,
                axis=newton.Axis.Y,
                parent_xform=wp.transform(wp.vec3(0.0, 2.0, 1.0), wp.quat_identity()),
                limit_lower=-1.0,
                limit_upper=0.0,
            )
        ]
    )
    world_builders.append(hinge)

    scene = newton.ModelBuilder()
    scene.rigid_gap = 0.005
    for world_builder in world_builders:
        scene.add_world(world_builder)
    scene.add_ground_plane()
    return scene.finalize(device=device)


def _run_diagonal_mass_local(kernel, *, iterations, graph, steps=24):
    """Record dense rows and joint coordinates per step for one H-inverse implementation."""
    model = _build_diagonal_mass_local_model("cuda:0")
    overrides = {} if kernel is None else {"hinv_jt_kernel": kernel}
    with kernel_overrides(overrides):
        solver = SolverFeatherPGS(
            model,
            solve_mode="matrix_free",
            iterations=iterations,
            enable_joint_limits=True,
            joint_limit_activation_gap=0.0,
            friction_anchor_beta=0.0,
            articulation_constraint_max=32,
            free_body_constraint_max=32,
        )
    model.rigid_contact_max = 32
    pipeline = newton.CollisionPipeline(model, rigid_contact_max=32)
    contacts = pipeline.contacts()
    control = model.control()
    states = [model.state(), model.state()]
    newton.eval_fk(model, model.joint_q, model.joint_qd, states[0])

    def step():
        for _ in range(2):
            states[0].clear_forces()
            pipeline.collide(states[0], contacts)
            solver.step(states[0], states[1], control, contacts, 0.005)
            states.reverse()

    def sample():
        count = solver._constraint_count.numpy()
        return (
            count.copy(),
            [solver._diag.numpy()[world, : count[world]].copy() for world in range(model.world_count)],
            [solver._row_type.numpy()[world, : count[world]].copy() for world in range(model.world_count)],
            solver._local_solve_owner.numpy().copy(),
            states[0].joint_q.numpy().copy(),
        )

    step()
    samples = [sample()]
    captured = None
    if graph:
        with wp.ScopedCapture(device=model.device) as capture:
            step()
        captured = capture.graph
    for _ in range(steps // 2 - 1):
        if captured is None:
            step()
        else:
            wp.capture_launch(captured)
        samples.append(sample())
    return solver, samples


_ARMATURE = (9.0, 4.0, 2.0, 0.5, 1.5, 3.0)


def _armature_sphere_on_ground(device, armature, dense, com=(0.0, 0.0, 0.0), mu=0.0):
    """A 1 kg sphere on a free joint, optionally with a massless fixed child that routes it to dense rows."""
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    body = builder.add_link(
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0999), wp.quat_identity()),
        mass=1.0,
        com=wp.vec3(com),
        inertia=wp.mat33(np.eye(3) * 0.01),
    )
    joints = [builder.add_joint_free(body)]
    if dense:
        child = builder.add_link(mass=0.0, inertia=wp.mat33(0.0))
        joints.append(builder.add_joint_fixed(body, child))
    builder.add_articulation(joints)
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, mu=mu)
    builder.add_shape_sphere(body, radius=0.1, cfg=cfg)
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=mu))
    model = builder.finalize(device=device)
    joint_armature = model.joint_armature.numpy()
    joint_armature[:6] = armature
    model.joint_armature.assign(joint_armature)
    return model


def _armature_impact(model, solver, joint_qd):
    state_in, state_out = model.state(), model.state()
    qd = state_in.joint_qd.numpy()
    qd[:6] = joint_qd
    state_in.joint_qd.assign(qd)
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(state_in, contacts)
    observables = solver.observables({newton.solvers.SolverObservableFlags.CONTACT_F})
    solver.step(state_in, state_out, model.control(), contacts, 0.01, observables=observables)
    count = int(contacts.rigid_contact_count.numpy()[0])
    return state_out.joint_qd.numpy()[:6], observables.contact_f.numpy()[:count, :3].sum(axis=0)


def _armature_modes(device):
    return ["matrix_free", "split"] if device.is_cuda else ["split"]


class TestFeatherPGSResponseDiagonal(unittest.TestCase):
    def test_free_body_contact_response_includes_armature(self):
        """Free-body rows respond with the same armature-augmented inertia as articulated rows."""
        device = wp.get_device()
        stop = (0.0, 0.0, -1.0, 0.0, 0.0, 0.0)
        oblique = (0.3, -0.2, -1.0, 0.4, 0.5, -0.3)
        for mode in _armature_modes(device):
            kwargs = {"solve_mode": mode, "cfm": 0.0, "friction_anchor_beta": 0.0}
            for armature, expected in ((0.0, 100.0), (9.0, 1000.0)):
                with self.subTest(mode=mode, armature=armature):
                    model = _armature_sphere_on_ground(device, armature, dense=False)
                    qd, force = _armature_impact(model, SolverFeatherPGS(model, **kwargs), stop)
                    # The sphere starts 0.1 mm deep, so the depenetration bias adds 0.2 %.
                    self.assertAlmostEqual(abs(float(force[2])), expected, delta=5.0e-3 * expected)
                    self.assertAlmostEqual(float(qd[2]), 0.0, delta=5.0e-3)
            with self.subTest(mode=mode, case="notified"):
                # An armature change notified after construction reaches the free-body response.
                model = _armature_sphere_on_ground(device, 0.0, dense=False)
                solver = SolverFeatherPGS(model, **kwargs)
                joint_armature = model.joint_armature.numpy()
                joint_armature[:6] = 9.0
                model.joint_armature.assign(joint_armature)
                solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
                _, force = _armature_impact(model, solver, stop)
                self.assertAlmostEqual(abs(float(force[2])), 1000.0, delta=5.0)
            with self.subTest(mode=mode, case="oblique"):
                results = {}
                for dense in (False, True):
                    model = _armature_sphere_on_ground(device, _ARMATURE, dense=dense, com=(0.03, -0.02, 0.01), mu=0.5)
                    solver = SolverFeatherPGS(model, iterations=64, **kwargs)
                    results[dense] = _armature_impact(model, solver, oblique)
                np.testing.assert_allclose(results[False][0], results[True][0], rtol=0.0, atol=1.0e-4)
                np.testing.assert_allclose(results[False][1], results[True][1], rtol=1.0e-3, atol=1.0e-2)

    @unittest.skipUnless(wp.is_cuda_available(), "sparse diagonal GS requires CUDA")
    def test_sparse_diagonal_gs_matches_scalar_reference(self):
        """Match a scalar PGS reference with coupled limits and friction."""
        device = wp.get_device("cuda:0")
        max_constraints, world_dofs, dense_dofs = 8, 4, 2
        cfm, omega, iterations = 1.0e-6, 1.0, 8
        inverse_mass = np.array((0.5, 0.25, 0.75, 1.0), dtype=np.float32)
        initial_velocity = np.array((-0.4, 0.3, -0.2, 0.1), dtype=np.float32)
        jacobian = np.array(
            ((0.6, -0.4, 0.2, 0.0), (0.1, 0.5, -0.3, 0.4), (-0.2, 0.25, 0.1, -0.35)),
            dtype=np.float32,
        )
        response = jacobian * inverse_mass
        rhs = np.zeros((1, max_constraints), dtype=np.float32)
        rhs[0, :3] = (-0.1, 0.0, 0.0)
        diag = np.zeros_like(rhs)
        diag[0, :3] = np.sum(jacobian * response, axis=1) + cfm
        row_type = np.zeros((1, max_constraints), dtype=np.int32)
        row_type[0, :3] = (PGS_CONSTRAINT_TYPE_CONTACT, PGS_CONSTRAINT_TYPE_FRICTION, PGS_CONSTRAINT_TYPE_FRICTION)
        row_parent = np.full((1, max_constraints), -1, dtype=np.int32)
        row_parent[0, 1:3] = 0
        row_mu = np.zeros((1, max_constraints), dtype=np.float32)
        row_mu[0, 1:3] = 0.7
        dense_j = np.zeros((1, max_constraints, dense_dofs), dtype=np.float32)
        dense_y = np.zeros_like(dense_j)
        dense_j[0, :3] = jacobian[:, :dense_dofs]
        dense_y[0, :3] = response[:, :dense_dofs]
        sparse_dof = np.full((1, max_constraints, 2), -1, dtype=np.int32)
        sparse_jy = np.zeros((1, max_constraints, 4), dtype=np.float32)
        sparse_dof[0, :3] = (2, 3)
        sparse_jy[0, :3, 0] = jacobian[:, 2]
        sparse_jy[0, :3, 1] = response[:, 2]
        sparse_jy[0, :3, 2] = jacobian[:, 3]
        sparse_jy[0, :3, 3] = response[:, 3]
        limit_active = np.zeros((1, world_dofs), dtype=np.int32)
        limit_active[0, 2] = 1
        limit_lower_rhs = np.zeros((1, world_dofs), dtype=np.float32)
        limit_lower_rhs[0, 2] = -0.15

        expected_velocity = initial_velocity.copy()
        expected_impulses = np.zeros(3, dtype=np.float32)
        expected_limit_lambda = np.float32(0.0)
        for _ in range(iterations):
            old_limit = expected_limit_lambda
            residual = expected_velocity[2] + limit_lower_rhs[0, 2]
            expected_limit_lambda = np.maximum(0.0, old_limit - omega * residual / (inverse_mass[2] + cfm))
            expected_velocity[2] += inverse_mass[2] * (expected_limit_lambda - old_limit)
            old_impulse = expected_impulses[0]
            new_impulse = old_impulse - omega * (jacobian[0] @ expected_velocity + rhs[0, 0]) / diag[0, 0]
            new_impulse = max(new_impulse, 0.0)
            expected_impulses[0] = new_impulse
            expected_velocity += response[0] * (new_impulse - old_impulse)
            # The general owner solves both tangent rows together on the friction disk at the current normal
            # load (friction_pair_candidate); the second tangent visit is a no-op.
            radius = max(row_mu[0, 1] * expected_impulses[0], 0.0)
            pair = friction_pair_candidate(
                float(diag[0, 1]),
                float(jacobian[1] @ response[2]),
                float(diag[0, 2]),
                wp.vec2(
                    float(jacobian[1] @ expected_velocity + rhs[0, 1]),
                    float(jacobian[2] @ expected_velocity + rhs[0, 2]),
                ),
                wp.vec2(float(expected_impulses[1]), float(expected_impulses[2])),
                float(radius),
                omega,
            )
            magnitude = np.sqrt(pair[0] * pair[0] + pair[1] * pair[1])
            scale = radius / magnitude if magnitude > radius else 1.0
            for tangent, value in ((1, pair[0] * scale), (2, pair[1] * scale)):
                expected_velocity += response[tangent] * (value - expected_impulses[tangent])
                expected_impulses[tangent] = np.float32(value)

        impulses = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)
        lower_lambda = wp.zeros((1, world_dofs), dtype=wp.float32, device=device)
        upper_lambda = wp.zeros_like(lower_lambda)
        velocity = wp.array(initial_velocity, dtype=wp.float32, device=device)
        kernel = _get_pgs_solve_sparse_diagonal_kernel(
            max_constraints, world_dofs, dense_dofs, str(device.arch), contact_triples=True
        )
        wp.launch_tiled(
            kernel,
            dim=[1],
            inputs=[
                wp.array((3,), dtype=wp.int32, device=device),
                wp.array(np.arange(world_dofs, dtype=np.int32)[None, :], device=device),
                wp.array(rhs, device=device),
                wp.array(diag, device=device),
                impulses,
                wp.array(row_type, device=device),
                wp.array(row_parent, device=device),
                wp.array(row_mu, device=device),
                wp.zeros((1, 2), dtype=wp.int32, device=device),
                wp.zeros(1, dtype=wp.int32, device=device),
                wp.empty((1, world_dofs), dtype=wp.int32, device=device),
                wp.zeros(1, dtype=wp.int32, device=device),
                wp.empty((1, (max_constraints + 2) // 3), dtype=wp.int32, device=device),
                wp.array((0,), dtype=wp.int32, device=device),
                wp.array((0,), dtype=wp.int32, device=device),
                wp.array(dense_j, device=device),
                wp.array(dense_y, device=device),
                wp.array(sparse_dof, device=device),
                wp.array(sparse_jy, device=device),
                wp.array(limit_active, device=device),
                wp.array(limit_lower_rhs, device=device),
                wp.zeros((1, world_dofs), dtype=wp.float32, device=device),
                wp.array(inverse_mass, device=device),
                cfm,
                iterations,
                omega,
                0,
                0,
            ],
            outputs=[lower_lambda, upper_lambda, velocity],
            block_dim=32,
            device=device,
        )

        np.testing.assert_allclose(velocity.numpy(), expected_velocity, rtol=2.0e-5, atol=2.0e-6)
        np.testing.assert_allclose(impulses.numpy()[0, :3], expected_impulses, rtol=2.0e-5, atol=2.0e-6)
        np.testing.assert_allclose(lower_lambda.numpy()[0, 2], expected_limit_lambda, rtol=2.0e-5, atol=2.0e-6)
        self.assertEqual(float(upper_lambda.numpy()[0, 2]), 0.0)

    @unittest.skipUnless(wp.is_cuda_available(), "speculative contact batches require CUDA")
    def test_speculative_sparse_contact_batches_match_serial_reference(self):
        """Skip exact no-op prefixes without changing later serial contact updates."""
        device = wp.get_device("cuda:0")
        max_constraints, world_dofs, dense_dofs = 12, 8, 6
        cfm, omega, iterations = 1.0e-6, 1.0, 2
        inverse_mass = np.array((0.5, 0.25, 0.75, 1.0, 0.4, 0.6, 0.75, 0.8), dtype=np.float32)
        initial_velocity = np.array((0.3, -0.2, 0.1, 0.4, -0.1, 0.2, 0.5, 0.25), dtype=np.float32)
        jacobian = np.zeros((max_constraints, world_dofs), dtype=np.float32)
        jacobian[0, 6] = 1.0
        jacobian[3, 7] = 1.0
        jacobian[6, (0, 6)] = (-0.2, -1.0)
        jacobian[9, (0, 6)] = (0.3, 1.0)
        response = jacobian * inverse_mass
        rhs = np.zeros((1, max_constraints), dtype=np.float32)
        rhs[0, 6] = -0.1
        rhs[0, 9] = -0.2
        diag = np.zeros_like(rhs)
        normal_rows = np.array((0, 3, 6, 9), dtype=np.int32)
        diag[0, normal_rows] = np.sum(jacobian[normal_rows] * response[normal_rows], axis=1) + cfm
        row_type = np.full((1, max_constraints), PGS_CONSTRAINT_TYPE_FRICTION, dtype=np.int32)
        row_type[0, normal_rows] = PGS_CONSTRAINT_TYPE_CONTACT
        row_parent = np.full((1, max_constraints), -1, dtype=np.int32)
        for normal in normal_rows:
            row_parent[0, normal + 1 : normal + 3] = normal
        row_mu = np.zeros((1, max_constraints), dtype=np.float32)
        dense_j = jacobian[:, :dense_dofs][None, ...]
        dense_y = response[:, :dense_dofs][None, ...]
        sparse_dof = np.empty((1, max_constraints, 2), dtype=np.int32)
        sparse_dof[:, :, 0] = 6
        sparse_dof[:, :, 1] = 7
        sparse_jy = np.zeros((1, max_constraints, 4), dtype=np.float32)
        sparse_jy[0, :, 0] = jacobian[:, 6]
        sparse_jy[0, :, 1] = response[:, 6]
        sparse_jy[0, :, 2] = jacobian[:, 7]
        sparse_jy[0, :, 3] = response[:, 7]

        expected_velocity = initial_velocity.copy()
        expected_impulses = np.zeros(max_constraints, dtype=np.float32)
        for _ in range(iterations):
            for row in normal_rows:
                old_impulse = expected_impulses[row]
                residual = jacobian[row] @ expected_velocity + rhs[0, row]
                new_impulse = max(old_impulse - omega * residual / diag[0, row], 0.0)
                expected_impulses[row] = new_impulse
                expected_velocity += response[row] * (new_impulse - old_impulse)

        impulses = wp.zeros((1, max_constraints), dtype=wp.float32, device=device)
        velocity = wp.array(initial_velocity, dtype=wp.float32, device=device)
        kernel = _get_pgs_solve_sparse_diagonal_kernel(
            max_constraints,
            world_dofs,
            dense_dofs,
            str(device.arch),
            contact_triples=True,
            speculative_contact_batches=True,
        )
        wp.launch_tiled(
            kernel,
            dim=[1],
            inputs=[
                wp.array((max_constraints,), dtype=wp.int32, device=device),
                wp.array(np.arange(world_dofs, dtype=np.int32)[None, :], device=device),
                wp.array(rhs, device=device),
                wp.array(diag, device=device),
                impulses,
                wp.array(row_type, device=device),
                wp.array(row_parent, device=device),
                wp.array(row_mu, device=device),
                wp.zeros((1, 2), dtype=wp.int32, device=device),
                wp.zeros(1, dtype=wp.int32, device=device),
                wp.empty((1, world_dofs), dtype=wp.int32, device=device),
                wp.array((len(normal_rows),), dtype=wp.int32, device=device),
                wp.array(normal_rows[None, :], device=device),
                wp.array((0,), dtype=wp.int32, device=device),
                wp.array((0,), dtype=wp.int32, device=device),
                wp.array(dense_j, device=device),
                wp.array(dense_y, device=device),
                wp.array(sparse_dof, device=device),
                wp.array(sparse_jy, device=device),
                wp.zeros((1, world_dofs), dtype=wp.int32, device=device),
                wp.zeros((1, world_dofs), dtype=wp.float32, device=device),
                wp.zeros((1, world_dofs), dtype=wp.float32, device=device),
                wp.array(inverse_mass, device=device),
                cfm,
                iterations,
                omega,
                0,
                0,
            ],
            outputs=[
                wp.zeros((1, world_dofs), dtype=wp.float32, device=device),
                wp.zeros((1, world_dofs), dtype=wp.float32, device=device),
                velocity,
            ],
            block_dim=32,
            device=device,
        )

        np.testing.assert_allclose(velocity.numpy(), expected_velocity, rtol=2.0e-5, atol=2.0e-6)
        np.testing.assert_allclose(impulses.numpy()[0], expected_impulses, rtol=2.0e-5, atol=2.0e-6)
        np.testing.assert_array_equal(impulses.numpy()[0, :6], np.zeros(6, dtype=np.float32))

    @unittest.skipUnless(wp.is_cuda_available(), "direct diagonal projection requires CUDA")
    def test_direct_diagonal_inverse_mass_matches_compact_inertia_terms(self):
        """Project a one-body branch directly from its compact inertia representation."""
        device = wp.get_device("cuda:0")
        mass = np.float32(2.5)
        com = np.array((0.2, -0.1, 0.3), dtype=np.float32)
        inertia = np.array(((0.8, 0.1, -0.05), (0.1, 1.1, 0.02), (-0.05, 0.02, 1.4)), dtype=np.float32)
        motion = np.array((0.3, -0.2, 0.4, 0.5, 0.1, -0.6), dtype=np.float32)
        armature = np.float32(0.25)
        drive_stiffness = np.float32(0.75)
        terms = np.concatenate((com, inertia.reshape(-1)))[None, :]
        inverse_mass = wp.zeros(1, dtype=wp.float32, device=device)
        kernel = _get_direct_diagonal_inverse_mass_kernel(1, str(device.arch))
        wp.launch(
            kernel,
            dim=1,
            inputs=[
                wp.array((0, 1), dtype=wp.int32, device=device),
                wp.array((0,), dtype=wp.int32, device=device),
                wp.array((1,), dtype=wp.int32, device=device),
                wp.array((0,), dtype=wp.int32, device=device),
                wp.array((wp.spatial_vector(*motion),), dtype=wp.spatial_vector, device=device),
                wp.array((mass,), dtype=wp.float32, device=device),
                wp.array(terms, dtype=wp.float32, device=device),
                wp.array((0,), dtype=wp.int32, device=device),
                wp.array((0,), dtype=wp.int32, device=device),
                wp.array(((armature,),), dtype=wp.float32, device=device),
                wp.array((0,), dtype=wp.int32, device=device),
                wp.array((drive_stiffness,), dtype=wp.float32, device=device),
            ],
            outputs=[inverse_mass],
            device=device,
        )
        wp.synchronize_device(device)

        linear = motion[:3]
        angular = motion[3:]
        response = np.concatenate(
            (
                mass * (linear - np.cross(com, angular)),
                mass * np.cross(com, linear) + inertia @ angular,
            )
        )
        expected = np.float32(1.0) / (np.dot(motion, response) + armature + drive_stiffness)
        np.testing.assert_allclose(inverse_mass.numpy()[0], expected, rtol=2.0e-6, atol=0.0)

    @unittest.skipUnless(wp.is_cuda_available(), "partitioned inverse dynamics requires CUDA")
    def test_direct_branch_inverse_dynamics_matches_articulation_path(self):
        """A one-body branch must produce the same torque through either schedule."""
        device = wp.get_device("cuda:0")
        spatial = wp.spatial_vector(0.4, -0.2, 0.6, 0.1, 0.3, -0.5)
        external = wp.spatial_vector(-0.3, 0.5, 0.2, -0.4, 0.1, 0.7)
        articulation_start = wp.array((0, 1), dtype=wp.int32, device=device)
        articulation_end = wp.array((1,), dtype=wp.int32, device=device)
        joint_type = wp.array((int(newton.JointType.PRISMATIC),), dtype=wp.int32, device=device)
        joint_parent = wp.array((-1,), dtype=wp.int32, device=device)
        joint_child = wp.array((0,), dtype=wp.int32, device=device)
        joint_articulation = wp.array((0,), dtype=wp.int32, device=device)
        starts = wp.array((0, 1), dtype=wp.int32, device=device)
        dof_dim = wp.array(((1, 0),), dtype=wp.int32, device=device)
        joint_f = wp.array((0.2,), dtype=wp.float32, device=device)
        joint_q = wp.array((0.15,), dtype=wp.float32, device=device)
        joint_qd = wp.array((-0.25,), dtype=wp.float32, device=device)
        stiffness = wp.array((1.4,), dtype=wp.float32, device=device)
        spring_ref = wp.array((0.05,), dtype=wp.float32, device=device)
        damping = wp.array((0.3,), dtype=wp.float32, device=device)
        joint_S_s = wp.array((spatial,), dtype=wp.spatial_vector, device=device)
        body_force = wp.array((spatial,), dtype=wp.spatial_vector, device=device)
        external_force = wp.array((external,), dtype=wp.spatial_vector, device=device)
        body_flags = wp.zeros(1, dtype=wp.int32, device=device)
        body_q = wp.array((wp.transform_identity(),), dtype=wp.transform, device=device)
        body_com = wp.array((wp.vec3(0.2, -0.1, 0.3),), dtype=wp.vec3, device=device)
        origin = wp.array((wp.vec3(),), dtype=wp.vec3, device=device)
        selected_tau = wp.array((0.7,), dtype=wp.float32, device=device)
        direct_tau = wp.clone(selected_tau)
        body_ft = wp.zeros(1, dtype=wp.spatial_vector, device=device)
        selected_kernel, direct_kernel = _get_partitioned_inverse_dynamics_kernels(1, str(device.arch))
        common = [
            articulation_start,
            articulation_end,
            joint_type,
            joint_parent,
            joint_child,
            joint_articulation,
            starts,
            starts,
            dof_dim,
            joint_f,
            joint_q,
            joint_qd,
            stiffness,
            spring_ref,
            damping,
            joint_S_s,
            body_force,
            external_force,
            body_flags,
            body_q,
            body_com,
            origin,
        ]
        wp.launch(
            selected_kernel,
            dim=1,
            inputs=[
                wp.array((0,), dtype=wp.int32, device=device),
                *common,
                1,
                wp.ones(1, dtype=wp.int32, device=device),
            ],
            outputs=[body_ft, selected_tau],
            device=device,
        )
        wp.launch(
            direct_kernel,
            dim=1,
            inputs=[
                wp.array((0,), dtype=wp.int32, device=device),
                wp.array((0,), dtype=wp.int32, device=device),
                articulation_start,
                joint_type,
                joint_child,
                starts,
                starts,
                dof_dim,
                joint_f,
                joint_q,
                joint_qd,
                stiffness,
                spring_ref,
                damping,
                joint_S_s,
                body_force,
                external_force,
                body_flags,
                body_q,
                body_com,
                origin,
                1,
                wp.ones(1, dtype=wp.int32, device=device),
            ],
            outputs=[direct_tau],
            device=device,
        )
        wp.synchronize_device(device)
        np.testing.assert_array_equal(direct_tau.numpy(), selected_tau.numpy())

    @unittest.skipUnless(wp.is_cuda_available(), "independent contact groups require CUDA")
    def test_independent_sparse_contact_groups_exclude_coupled_coordinates(self):
        """Keep scalar contacts serial when a coupled contact shares their coordinate."""
        device = wp.get_device("cuda:0")
        max_constraints, max_world_dofs, contact_count = 12, 4, 4
        sparse_response_dofs = 108
        sparse_row_dof_np = np.full((1, max_constraints, 2), -1, dtype=np.int32)
        sparse_row_dof_np[0, 0:3, 0] = 2
        sparse_row_dof_np[0, 3:6, 0] = 3
        sparse_row_dof_np[0, 6:9, 0] = 3
        sparse_row_dof_np[0, 9:12, 0] = 2
        sparse_row_dof = wp.array(sparse_row_dof_np, device=device)
        group_count = wp.zeros(1, dtype=wp.int32, device=device)
        group_heads = wp.empty((1, max_world_dofs), dtype=wp.int32, device=device)
        serial_count = wp.zeros(1, dtype=wp.int32, device=device)
        serial_normals = wp.empty((1, (max_constraints + 2) // 3), dtype=wp.int32, device=device)

        marker = _get_mark_independent_sparse_contact_candidates_kernel(sparse_response_dofs, str(device.arch))
        wp.launch(
            marker,
            dim=contact_count,
            inputs=[
                wp.array((contact_count,), dtype=wp.int32, device=device),
                contact_count,
                wp.zeros(contact_count, dtype=wp.int32, device=device),
                wp.array((0, 3, 6, 9), dtype=wp.int32, device=device),
                wp.zeros(contact_count, dtype=wp.int32, device=device),
                wp.array((-1, 1, -1, -1), dtype=wp.int32, device=device),
                wp.zeros(contact_count, dtype=wp.int32, device=device),
                wp.full(contact_count, 3, dtype=wp.int32, device=device),
                wp.array((sparse_response_dofs, 6), dtype=wp.int32, device=device),
            ],
            outputs=[sparse_row_dof],
            device=device,
        )
        schedule = _get_build_independent_sparse_contact_groups_kernel(
            max_constraints, max_world_dofs, str(device.arch), build_serial_contacts=True
        )
        wp.launch(
            schedule,
            dim=32,
            inputs=[
                wp.array((max_constraints,), dtype=wp.int32, device=device),
                wp.zeros((1, 2), dtype=wp.int32, device=device),
            ],
            outputs=[sparse_row_dof, group_count, group_heads, serial_count, serial_normals],
            device=device,
        )
        wp.synchronize_device(device)

        self.assertEqual(int(group_count.numpy()[0]), 1)
        self.assertEqual(int(group_heads.numpy()[0, 0]), 0)
        self.assertEqual(int(serial_count.numpy()[0]), 2)
        np.testing.assert_array_equal(serial_normals.numpy()[0, :2], np.array((3, 6), dtype=np.int32))
        normal_links = sparse_row_dof.numpy()[0, ::3, 1]
        np.testing.assert_array_equal(normal_links, np.array((-12, -1, -1, -2), dtype=np.int32))

    def test_mixed_world_diagonal_map_includes_free_rigid_response(self):
        """Include free rigid articulations in every forced-tiled diagonal group."""
        solver = SolverFeatherPGS(_build_mixed_response_model("cpu", world_count=2), articulation_constraint_max=32)
        self.assertEqual(set(solver._size_groups), {6, 13})

        response_dof_count = solver._model_plan.response_dof_count
        articulation_world = solver._model_plan.articulation_world
        free_mask = solver._model_plan.is_free_rigid != 0
        for size in solver._size_groups:
            size_arts = np.flatnonzero(response_dof_count == size)
            response_order = np.asarray(
                sorted(size_arts, key=lambda art: (int(articulation_world[art]), int(art))), dtype=np.int32
            )
            propagation_order = response_order[~free_mask[response_order]]
            response_counts = np.bincount(articulation_world[response_order], minlength=solver._world_count)
            propagation_counts = np.bincount(articulation_world[propagation_order], minlength=solver._world_count)
            response_starts = np.pad(np.cumsum(response_counts), (1, 0)).astype(np.int32)
            propagation_starts = np.pad(np.cumsum(propagation_counts), (1, 0)).astype(np.int32)

            np.testing.assert_array_equal(solver._world_response_group_art_start[size].numpy(), response_starts)
            np.testing.assert_array_equal(solver._world_response_group_to_art[size].numpy(), response_order)
            np.testing.assert_array_equal(solver._world_group_art_start[size].numpy(), propagation_starts)
            np.testing.assert_array_equal(solver._world_group_to_art[size].numpy(), propagation_order)

        free_articulations = np.flatnonzero(free_mask).astype(np.int32)
        np.testing.assert_array_equal(solver._world_group_art_start[6].numpy(), np.array((0, 0, 0), dtype=np.int32))
        np.testing.assert_array_equal(solver._world_group_to_art[6].numpy(), np.empty(0, dtype=np.int32))
        np.testing.assert_array_equal(
            solver._world_response_group_art_start[6].numpy(), np.array((0, 1, 2), dtype=np.int32)
        )
        np.testing.assert_array_equal(solver._world_response_group_to_art[6].numpy(), free_articulations)

        forced_tiled = _FeatherPGSExecutionPlan.build(
            solver._size_groups,
            max_constraints=solver._articulation_constraint_max,
            max_shared_memory=101376,
            cholesky_kernel="auto",
            hinv_jt_kernel="tiled",
            small_dof_threshold=12,
            tile_threads=64,
        )
        self.assertEqual(forced_tiled.hinv_jt_tiled_sizes, frozenset((6, 13)))

    @unittest.skipUnless(wp.is_cuda_available(), "matrix-free response diagonal parity requires CUDA")
    def test_forced_tiled_mixed_world_matches_par_row_trajectory(self):
        """Match diagonal, impulses, and state when every response group is tiled."""
        for warmstart in (False, True):
            for preelimination in (False, True):
                with self.subTest(warmstart=warmstart, preelimination=preelimination):
                    reference_solver, reference = _run_mixed_response(
                        "par_row", warmstart=warmstart, preelimination=preelimination
                    )
                    tiled_solver, tiled = _run_mixed_response(
                        "tiled", warmstart=warmstart, preelimination=preelimination
                    )

                    self.assertEqual(reference_solver._hinv_jt_diag_sizes, frozenset())
                    self.assertEqual(reference_solver._preelim_active, preelimination)
                    self.assertEqual(tiled_solver._preelim_active, preelimination)
                    expected_diag_sizes = frozenset() if preelimination else frozenset((6, 13))
                    self.assertEqual(tiled_solver._hinv_jt_diag_sizes, expected_diag_sizes)
                    self.assertEqual(len(reference), len(tiled))
                    self.assertGreater(reference[0][0], 0, "mixed scene generated no dense constraint rows")
                    for step, (expected, actual) in enumerate(zip(reference, tiled, strict=True)):
                        self.assertEqual(actual[0], expected[0], f"constraint count differed at step {step}")
                        for label, expected_value, actual_value in zip(
                            ("diagonal", "impulses", "joint_q", "joint_qd"),
                            expected[1:5],
                            actual[1:5],
                            strict=True,
                        ):
                            np.testing.assert_allclose(
                                actual_value,
                                expected_value,
                                rtol=5.0e-4,
                                atol=2.0e-6,
                                err_msg=f"{label} differed at step {step}",
                            )

    @unittest.skipUnless(wp.is_cuda_available(), "paired response ownership requires CUDA")
    def test_paired_response_matches_general_23_dof_trajectory(self):
        """Match the general response when one warp owns a robot/free-body pair, with patch and point friction."""
        run_kwargs = {
            "warmstart": False,
            "preelimination": False,
            "dof_count": 23,
            "articulation_constraint_max": 96,
            "inactive_joint_limit_capacity": True,
            "friction": 0.7,
            "restitution": 0.3,
            "tangential_velocity": 2.0,
        }
        # Patch friction rows are not uniform triples, so the generic factor path pools the patch load; point
        # friction keeps the contact-triple layout.
        for label, friction_anchor_beta, triples in (("patch", None, False), ("point", 0.0, True)):
            with self.subTest(friction=label):
                reference_solver, reference = _run_mixed_response(
                    "par_row", friction_anchor_beta=friction_anchor_beta, **run_kwargs
                )
                paired_solver, paired = _run_mixed_response(
                    "auto", friction_anchor_beta=friction_anchor_beta, **run_kwargs
                )

                self.assertIsNone(reference_solver._paired_response_primary_size)
                self.assertEqual(paired_solver._friction_anchors_enabled, label == "patch")
                self.assertEqual(paired_solver._paired_response_primary_size, 23)
                self.assertEqual(paired_solver._paired_response_secondary_size, 6)
                self.assertIsNotNone(paired_solver._paired_response_kernel)
                self.assertIsNotNone(paired_solver._paired_factor_solve_kernel)
                self.assertTrue(paired_solver._paired_factor_coordinates)
                self.assertEqual(paired_solver._factor_coordinate_contact_triples, triples)
                self.assertGreater(reference[0][0], 0, "mixed scene generated no dense constraint rows")
                self.assertTrue(
                    any(np.any(sample[6] == PGS_CONSTRAINT_TYPE_FRICTION) for sample in paired),
                    "mixed scene generated no friction rows",
                )
                for step, (expected, actual) in enumerate(zip(reference, paired, strict=True)):
                    self.assertEqual(actual[0], expected[0], f"constraint count differed at step {step}")
                    np.testing.assert_array_equal(actual[6], expected[6], err_msg=f"row types differed at step {step}")
                    for label_value, expected_value, actual_value in zip(
                        ("diagonal", "impulses", "joint_q", "joint_qd"), expected[1:5], actual[1:5], strict=True
                    ):
                        np.testing.assert_allclose(
                            actual_value,
                            expected_value,
                            rtol=5.0e-4,
                            atol=1.0e-5,
                            err_msg=f"{label_value} differed at step {step}",
                        )

    @unittest.skipUnless(wp.is_cuda_available(), "paired response fallback requires CUDA")
    def test_paired_response_falls_back_for_matrix_free_rows(self):
        """Keep mixed dense/matrix-free worlds in physical coordinates."""
        run_kwargs = {
            "warmstart": False,
            "preelimination": False,
            "dof_count": 23,
            "articulation_constraint_max": 96,
            "friction": 0.7,
            "restitution": 0.3,
            "tangential_velocity": 2.0,
            "static_plane": True,
        }
        reference_solver, reference = _run_mixed_response("par_row", **run_kwargs)
        paired_solver, paired = _run_mixed_response("auto", **run_kwargs)

        self.assertIsNone(reference_solver._paired_response_primary_size)
        self.assertTrue(paired_solver._paired_factor_coordinates)
        self.assertTrue(any(sample[7] > 0 for sample in paired), "static contact generated no matrix-free rows")
        for step, (expected, actual) in enumerate(zip(reference, paired, strict=True)):
            self.assertEqual(actual[0], expected[0], f"constraint count differed at step {step}")
            np.testing.assert_array_equal(actual[6], expected[6], err_msg=f"row types differed at step {step}")
            for label, expected_value, actual_value in zip(
                ("diagonal", "impulses", "joint_q", "joint_qd"), expected[1:5], actual[1:5], strict=True
            ):
                np.testing.assert_allclose(
                    actual_value,
                    expected_value,
                    rtol=5.0e-4,
                    atol=1.0e-5,
                    err_msg=f"{label} differed at step {step}",
                )

    @unittest.skipUnless(wp.is_cuda_available(), "paired response ownership requires CUDA")
    def test_paired_factor_coordinates_follow_patch_friction_and_torsion(self):
        """Keep factor coordinates with default patch friction and yield them to the general owner for torsion."""
        model = _build_mixed_response_model("cuda:0", dof_count=23, friction=0.7)
        solvers = {}
        with kernel_overrides({"hinv_jt_kernel": "auto"}):
            for radius in (0.0, 0.01):
                set_torsion_radius(model, radius)
                solvers[radius] = SolverFeatherPGS(
                    model,
                    solve_mode="matrix_free",
                    enable_joint_limits=True,
                    joint_limit_activation_gap=0.0,
                    iterations=8,
                    articulation_constraint_max=96,
                    free_body_constraint_max=32,
                )
            set_torsion_radius(model, 0.0)
            point = SolverFeatherPGS(
                model,
                solve_mode="matrix_free",
                enable_joint_limits=True,
                joint_limit_activation_gap=0.0,
                iterations=8,
                articulation_constraint_max=96,
                free_body_constraint_max=32,
                friction_anchor_beta=0.0,
            )
        self.assertTrue(solvers[0.0]._friction_anchors_enabled)
        self.assertTrue(solvers[0.0]._paired_factor_coordinates)
        # Patch rows are not uniform triples; the generic factor path pools the patch load instead.
        self.assertFalse(solvers[0.0]._factor_coordinate_contact_triples)
        self.assertFalse(point._friction_anchors_enabled)
        self.assertTrue(point._paired_factor_coordinates)
        self.assertTrue(point._factor_coordinate_contact_triples)
        self.assertTrue(solvers[0.01]._contact_torsion_enabled)
        self.assertIsNotNone(solvers[0.01]._paired_response_primary_size)
        self.assertFalse(solvers[0.01]._paired_factor_coordinates)
        self.assertFalse(solvers[0.01]._factor_coordinate_contact_triples)

    @unittest.skipUnless(wp.is_cuda_available(), "articulation-local mixed-world parity requires CUDA")
    def test_local_internal_mixed_world_matches_general_response(self):
        """Match the general response when an articulation contacts a free body."""
        general_solver, general = _run_mixed_response(
            "tiled", warmstart=False, preelimination=False, inactive_joint_limit_capacity=True
        )
        local_solver, local = _run_mixed_response(
            "par_row", warmstart=False, preelimination=False, inactive_joint_limit_capacity=True
        )

        self.assertFalse(general_solver._local_internal_fast_path)
        self.assertTrue(local_solver._local_internal_fast_path)
        self.assertEqual(len(general), len(local))
        self.assertGreater(general[0][0], 0, "mixed scene generated no dense constraint rows")
        self.assertTrue(
            any(sample[5] == PGS_LOCAL_SOLVE_OWNER_PAIR for sample in local),
            "mixed contact never selected the paired local owner",
        )

        free_articulation = int(np.flatnonzero(general_solver._model_plan.is_free_rigid)[0])
        free_dof_start = int(general_solver._model_plan.articulation_dof_start[free_articulation])
        self.assertGreater(
            abs(float(general[0][4][free_dof_start + 2]) + 3.0),
            1.0e-4,
            "general response did not couple the dense contact to the free body",
        )

        for step, (expected, actual) in enumerate(zip(general, local, strict=True)):
            self.assertEqual(actual[0], expected[0], f"constraint count differed at step {step}")
            for label, expected_value, actual_value in zip(
                ("diagonal", "impulses", "joint_q", "joint_qd"), expected[1:5], actual[1:5], strict=True
            ):
                np.testing.assert_allclose(
                    actual_value,
                    expected_value,
                    rtol=5.0e-4,
                    atol=2.0e-6,
                    err_msg=f"{label} differed at step {step}",
                )

    @unittest.skipUnless(wp.is_cuda_available(), "articulation-local mixed-world parity requires CUDA")
    def test_local_pair_matches_general_friction_and_restitution(self):
        """Preserve paired friction projection and restitution response."""
        run_kwargs = {
            "warmstart": False,
            "preelimination": False,
            "inactive_joint_limit_capacity": True,
            "friction": 0.7,
            "restitution": 0.3,
            "tangential_velocity": 2.0,
        }
        general_solver, general = _run_mixed_response("tiled", **run_kwargs)
        local_solver, local = _run_mixed_response("par_row", **run_kwargs)

        self.assertFalse(general_solver._local_internal_fast_path)
        self.assertTrue(local_solver._local_internal_fast_path)
        pair_samples = [sample for sample in local if sample[5] == PGS_LOCAL_SOLVE_OWNER_PAIR]
        self.assertTrue(pair_samples, "mixed friction contact never selected the paired local owner")
        self.assertTrue(
            any(
                np.any(np.abs(sample[2][sample[6] == PGS_CONSTRAINT_TYPE_FRICTION]) > 1.0e-5) for sample in pair_samples
            ),
            "paired solve produced no nonzero friction impulse",
        )

        for step, (expected, actual) in enumerate(zip(general, local, strict=True)):
            self.assertEqual(actual[0], expected[0], f"constraint count differed at step {step}")
            np.testing.assert_array_equal(actual[6], expected[6], err_msg=f"row types differed at step {step}")
            for label, expected_value, actual_value in zip(
                ("diagonal", "impulses", "joint_q", "joint_qd"), expected[1:5], actual[1:5], strict=True
            ):
                np.testing.assert_allclose(
                    actual_value,
                    expected_value,
                    rtol=5.0e-4,
                    atol=2.0e-6,
                    err_msg=f"{label} differed at step {step}",
                )

    @unittest.skipUnless(wp.is_cuda_available(), "articulation-local mixed-world parity requires CUDA")
    def test_local_owners_match_general_contact_regularization(self):
        """Apply the general owner's per-row regularization weights in the local owners."""
        run_kwargs = {
            "warmstart": False,
            "preelimination": False,
            "inactive_joint_limit_capacity": True,
            "friction": 0.7,
            "restitution": 0.3,
            "tangential_velocity": 2.0,
            "contact_regularization": 1.0,
            "model_kwargs": {"static_support": True},
        }
        general_solver, general = _run_mixed_response("tiled", **run_kwargs)
        local_solver, local = _run_mixed_response("par_row", **run_kwargs)

        self.assertTrue(general_solver._regularization_enabled)
        self.assertFalse(general_solver._local_internal_fast_path)
        self.assertTrue(local_solver._local_internal_fast_path)
        owners = [sample[5] for sample in local]
        self.assertIn(
            PGS_LOCAL_SOLVE_OWNER_PAIR_RESIDUAL, owners, "regularized mixed contact never selected the residual owner"
        )
        self.assertTrue(
            any(sample[7] > 0 and sample[5] != PGS_LOCAL_SOLVE_OWNER_GENERAL for sample in local),
            "no local owner solved matrix-free rows",
        )
        _assert_owner_parity(self, general, local)

    @unittest.skipUnless(wp.is_cuda_available(), "articulation-local mixed-world parity requires CUDA")
    def test_local_owner_matches_general_persistent_patch_friction(self):
        """Preserve pooled normal load and coupled tangent response under local ownership."""
        run_kwargs = {
            "warmstart": False,
            "preelimination": False,
            "inactive_joint_limit_capacity": True,
            "friction": 0.7,
            "restitution": 0.3,
            "tangential_velocity": 2.0,
            "model_kwargs": {"static_support": True},
        }
        general_solver, general = _run_mixed_response("tiled", **run_kwargs)
        local_solver, local = _run_mixed_response("par_row", **run_kwargs)

        self.assertTrue(general_solver._friction_anchors_enabled)
        self.assertTrue(local_solver._friction_anchors_enabled)
        self.assertFalse(general_solver._local_internal_fast_path)
        self.assertTrue(local_solver._local_internal_fast_path)
        self.assertIn(PGS_LOCAL_SOLVE_OWNER_PAIR_RESIDUAL, [sample[5] for sample in local])

        self.assertTrue(
            any(
                sample[10][row] >= 0 and sample[10][row] != row
                for sample in local
                for row in np.flatnonzero(sample[9] == PGS_CONSTRAINT_TYPE_CONTACT)
            ),
            "scene generated no linked multi-normal patch",
        )

        tangent_cross_terms = []
        for sample in local:
            row_type, row_parent = sample[9], sample[10]
            jacobian_a, jacobian_b = sample[11], sample[12]
            response_a, response_b = sample[13], sample[14]
            for row in np.flatnonzero(row_type == PGS_CONSTRAINT_TYPE_FRICTION):
                parent = row_parent[row]
                if row != parent + 1:
                    continue
                sibling = parent + 2
                tangent_cross_terms.append(
                    float(jacobian_a[row] @ response_a[sibling] + jacobian_b[row] @ response_b[sibling])
                )
        self.assertTrue(tangent_cross_terms, "scene generated no tangent pair")
        self.assertGreater(max(abs(value) for value in tangent_cross_terms), 1.0e-3)
        _assert_owner_parity(self, general, local)

    @unittest.skipUnless(wp.is_cuda_available(), "articulation-local mixed-world parity requires CUDA")
    def test_free_body_velocity_limit_rows_stay_on_general_owner(self):
        """Keep worlds with free-body velocity-limit rows on the general owner."""
        run_kwargs = {
            "warmstart": False,
            "preelimination": False,
            "inactive_joint_limit_capacity": True,
            "model_kwargs": {"free_body_velocity_limit": 1.0},
        }
        general_solver, general = _run_mixed_response("tiled", **run_kwargs)
        local_solver, local = _run_mixed_response("par_row", **run_kwargs)

        self.assertFalse(general_solver._local_internal_fast_path)
        self.assertTrue(local_solver._local_internal_fast_path)
        limited = [sample for sample in local if np.any(sample[9] == PGS_CONSTRAINT_TYPE_JOINT_VELOCITY_LIMIT)]
        self.assertTrue(limited, "free body never produced a velocity-limit row")
        # The residual owner is eligible for this world; only the velocity-limit rows keep it general.
        self.assertGreaterEqual(int(local_solver._local_residual_pair_articulation.numpy()[0]), 0)
        for sample in limited:
            self.assertEqual(sample[5], PGS_LOCAL_SOLVE_OWNER_GENERAL, "velocity-limit rows left the general owner")
        _assert_owner_parity(self, general, local)

    @unittest.skipUnless(wp.is_cuda_available(), "articulation-local mixed-world parity requires CUDA")
    def test_free_body_first_world_layout_matches_general(self):
        """Reject the residual owner when the world packs the free body before the articulation."""
        run_kwargs = {
            "warmstart": False,
            "preelimination": False,
            "inactive_joint_limit_capacity": True,
            "friction": 0.7,
            "restitution": 0.3,
            "tangential_velocity": 2.0,
            "model_kwargs": {"static_support": True, "free_body_first": True},
        }
        general_solver, general = _run_mixed_response("tiled", **run_kwargs)
        local_solver, local = _run_mixed_response("par_row", **run_kwargs)

        free_articulation = int(np.flatnonzero(local_solver._model_plan.is_free_rigid)[0])
        self.assertEqual(int(local_solver._model_plan.articulation_dof_start[free_articulation]), 0)
        self.assertFalse(general_solver._local_internal_fast_path)
        self.assertTrue(local_solver._local_internal_fast_path)
        np.testing.assert_array_equal(local_solver._local_residual_pair_articulation.numpy(), [-1])
        self.assertGreaterEqual(int(local_solver._local_pair_articulation.numpy()[0]), 0)
        self.assertTrue(any(sample[7] > 0 for sample in local), "free body generated no matrix-free rows")
        for sample in local:
            if sample[7] > 0:
                self.assertEqual(sample[5], PGS_LOCAL_SOLVE_OWNER_GENERAL, "matrix-free rows left the general owner")
        _assert_owner_parity(self, general, local)

    @unittest.skipUnless(wp.is_cuda_available(), "articulation-local mixed-world parity requires CUDA")
    def test_local_owners_stay_general_with_contact_torsion(self):
        """Keep torsion-eligible worlds on the general owner, which solves the appended torsion row."""
        model = _build_mixed_response_model("cuda:0", friction=0.7)
        solvers = {}
        with kernel_overrides({"hinv_jt_kernel": "par_row"}):
            for radius in (0.0, 0.01):
                set_torsion_radius(model, radius)
                solvers[radius] = SolverFeatherPGS(
                    model,
                    solve_mode="matrix_free",
                    enable_joint_limits=True,
                    joint_limit_activation_gap=0.0,
                    iterations=8,
                    articulation_constraint_max=32,
                    free_body_constraint_max=32,
                )
        self.assertTrue(solvers[0.0]._local_internal_fast_path)
        self.assertTrue(solvers[0.01]._contact_torsion_enabled)
        self.assertFalse(solvers[0.01]._local_internal_fast_path)

    @unittest.skipUnless(wp.is_cuda_available(), "articulation-local owners require CUDA")
    def test_local_owners_are_limited_to_small_world_counts(self):
        """Select local owners only up to one world per SM; larger models solve exactly as the general owner."""
        limit = wp.get_device("cuda:0").sm_count
        kwargs = {"solve_mode": "matrix_free", "articulation_constraint_max": 32, "free_body_constraint_max": 32}
        with kernel_overrides({"hinv_jt_kernel": "par_row"}):
            for world_count, expected in ((limit, True), (limit + 1, False)):
                model = _build_mixed_response_model("cuda:0", world_count=world_count)
                solver = SolverFeatherPGS(model, **kwargs)
                self.assertEqual(solver._local_internal_fast_path, expected, f"world_count={world_count}")

        # The two-world diagonal-mass scene is local by default, gated with no worlds per SM, ineligible without DOFs.
        trajectories = {}
        for label, attribute, value in (
            ("local", "_LOCAL_SOLVE_WORLDS_PER_SM", 1),
            ("gated", "_LOCAL_SOLVE_WORLDS_PER_SM", 0),
            ("ineligible", "_LOCAL_INTERNAL_MAX_DOF", 0),
        ):
            with mock.patch.object(solver_feather_pgs_module, attribute, value):
                solver, samples = _run_diagonal_mass_local(None, iterations=4, graph=False, steps=8)
            self.assertEqual(solver._local_internal_fast_path, label == "local", label)
            trajectories[label] = np.stack([sample[4] for sample in samples])
        np.testing.assert_array_equal(trajectories["gated"], trajectories["ineligible"])
        np.testing.assert_allclose(trajectories["local"], trajectories["gated"], rtol=0.0, atol=1.0e-6)

    @unittest.skipUnless(wp.is_cuda_available(), "articulation-local owners require CUDA")
    def test_local_owners_match_general_for_diagonal_mass_articulations(self):
        """Count the response once in local owners whose articulation uses the diagonal-mass H-inverse."""
        for iterations in (1, 8):
            for graph in (False, True):
                with self.subTest(iterations=iterations, graph=graph):
                    general_solver, general = _run_diagonal_mass_local("tiled", iterations=iterations, graph=graph)
                    local_solver, local = _run_diagonal_mass_local(None, iterations=iterations, graph=graph)
                    self.assertFalse(general_solver._local_internal_fast_path)
                    self.assertTrue(local_solver._local_internal_fast_path)
                    self.assertTrue(local_solver._hinv_jt_writes_world)
                    self.assertIn(1, local_solver._hinv_jt_diag_sizes)
                    owners = np.stack([sample[3] for sample in local])
                    counts = np.stack([sample[0] for sample in local])
                    self.assertTrue((counts > 0).all(axis=1).sum() >= len(local) - 1, "worlds generated no rows")
                    self.assertTrue(
                        np.isin(
                            owners[counts[:, 0] > 0, 0],
                            (PGS_LOCAL_SOLVE_OWNER_PAIR, PGS_LOCAL_SOLVE_OWNER_PAIR_RESIDUAL),
                        ).all()
                    )
                    np.testing.assert_array_equal(owners[counts[:, 1] > 0, 1], PGS_LOCAL_SOLVE_OWNER_SINGLE)
                    for step, (expected, actual) in enumerate(zip(general, local, strict=True)):
                        np.testing.assert_array_equal(actual[0], expected[0], err_msg=f"row counts at step {step}")
                        for world in range(2):
                            np.testing.assert_array_equal(actual[2][world], expected[2][world])
                            np.testing.assert_allclose(
                                actual[1][world],
                                expected[1][world],
                                rtol=1.0e-5,
                                err_msg=f"world {world} diagonal at step {step}",
                            )
                        np.testing.assert_allclose(
                            actual[4], expected[4], rtol=0.0, atol=1.0e-6, err_msg=f"joint_q at step {step}"
                        )

    @unittest.skipUnless(wp.is_cuda_available(), "tiled H-inverse response requires CUDA")
    def test_tiled_response_diagonal_matches_dense_reference(self):
        device = wp.get_device("cuda:0")
        rng = np.random.default_rng(41)
        num_dofs = 23
        max_constraints = 16
        num_articulations = 4
        world_constraint_count_np = np.array((13, 7), dtype=np.int32)
        articulation_world_np = np.array((0, 0, 1, 1), dtype=np.int32)
        articulation_world_dof_offset_np = np.array((0, num_dofs, 0, num_dofs), dtype=np.int32)

        factors = rng.normal(size=(num_articulations, num_dofs, num_dofs)).astype(np.float32)
        mass = factors @ np.transpose(factors, (0, 2, 1))
        mass += 2.0 * np.eye(num_dofs, dtype=np.float32)[None, :, :]
        cholesky = wp.array(np.linalg.cholesky(mass).astype(np.float32), device=device)

        jacobian_np = np.zeros((num_articulations, max_constraints, num_dofs), dtype=np.float32)
        for articulation, world in enumerate(articulation_world_np):
            count = int(world_constraint_count_np[world])
            jacobian_np[articulation, :count] = rng.normal(size=(count, num_dofs))
        jacobian = wp.array(jacobian_np, device=device)

        group_to_art = wp.array(np.arange(num_articulations, dtype=np.int32), device=device)
        art_to_world = wp.array(articulation_world_np, device=device)
        articulation_world_dof_offset = wp.array(articulation_world_dof_offset_np, device=device)
        constraint_count = wp.array(world_constraint_count_np, device=device)
        response = wp.zeros_like(jacobian)
        world_jacobian = wp.zeros((2, max_constraints, 2 * num_dofs), dtype=wp.float32, device=device)
        world_response = wp.zeros_like(world_jacobian)
        group_diag = wp.zeros((num_articulations, max_constraints), dtype=wp.float32, device=device)

        kernel = _get_hinv_jt_kernel(
            num_dofs,
            max_constraints,
            str(device.arch),
            64,
            constraint_chunk_size=8,
            write_world=True,
            write_group=False,
        )
        wp.launch_tiled(
            kernel,
            dim=(num_articulations, 2),
            inputs=[
                cholesky,
                jacobian,
                group_to_art,
                art_to_world,
                articulation_world_dof_offset,
                constraint_count,
            ],
            outputs=[response, world_jacobian, world_response, group_diag],
            block_dim=64,
            device=device,
        )

        world_diag = wp.zeros((2, max_constraints), dtype=wp.float32, device=device)
        wp.launch(
            accumulate_group_diag_worlds,
            dim=2 * max_constraints,
            inputs=[
                group_diag,
                wp.array((0, 2, 4), dtype=wp.int32, device=device),
                group_to_art,
                group_to_art,
                constraint_count,
                max_constraints,
            ],
            outputs=[world_diag],
            device=device,
        )
        wp.synchronize_device(device)

        response_ref = np.zeros_like(jacobian_np)
        group_diag_ref = np.zeros((num_articulations, max_constraints), dtype=np.float32)
        world_jacobian_ref = np.zeros((2, max_constraints, 2 * num_dofs), dtype=np.float32)
        world_response_ref = np.zeros_like(world_jacobian_ref)
        for articulation, world in enumerate(articulation_world_np):
            count = int(world_constraint_count_np[world])
            response_ref[articulation, :count] = np.linalg.solve(
                mass[articulation], jacobian_np[articulation, :count].T
            ).T
            group_diag_ref[articulation, :count] = np.sum(
                jacobian_np[articulation, :count] * response_ref[articulation, :count], axis=1
            )
            dof_offset = articulation_world_dof_offset_np[articulation]
            world_jacobian_ref[world, :count, dof_offset : dof_offset + num_dofs] = jacobian_np[articulation, :count]
            world_response_ref[world, :count, dof_offset : dof_offset + num_dofs] = response_ref[articulation, :count]
        world_diag_ref = np.stack((group_diag_ref[:2].sum(axis=0), group_diag_ref[2:].sum(axis=0)))

        np.testing.assert_array_equal(response.numpy(), np.zeros_like(response_ref))
        np.testing.assert_allclose(world_jacobian.numpy(), world_jacobian_ref, rtol=0.0, atol=0.0)
        np.testing.assert_allclose(world_response.numpy(), world_response_ref, rtol=2.0e-5, atol=2.0e-6)
        np.testing.assert_allclose(group_diag.numpy(), group_diag_ref, rtol=2.0e-5, atol=2.0e-6)
        np.testing.assert_allclose(world_diag.numpy(), world_diag_ref, rtol=2.0e-5, atol=2.0e-6)


if __name__ == "__main__":
    unittest.main()
