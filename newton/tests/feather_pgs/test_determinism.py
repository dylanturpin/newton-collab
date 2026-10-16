# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Deterministic stepping of SolverFeatherPGS."""

import hashlib
import re
import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.feather_pgs import solver_feather_pgs
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT_ANGULAR_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    crba_fill_par_dof,
)
from newton.solvers import SolverBase, SolverFeatherPGS
from newton.tests.unittest_utils import StdOutCapture, add_function_test, get_test_devices

# Free root, then two ball joints: every joint has several DOFs.
JOINT_DOF_DIMS = ((3, 3), (0, 3), (0, 3))


CRBA_ARTICULATIONS = 256


CRBA_LAUNCHES = 8


def test_crba_fill_par_dof_is_batch_invariant(test, device):
    """Identical articulations get bitwise-identical, exactly symmetric mass matrices on every launch."""
    joint_count = len(JOINT_DOF_DIMS)
    dof_count = sum(lin + ang for lin, ang in JOINT_DOF_DIMS)
    rng = np.random.default_rng(7)
    # Generic motion subspaces and inertias, so S_r.(I S_c) and S_c.(I S_r) round differently.
    motion = rng.normal(size=(dof_count, 6)).astype(np.float32)
    inertia = rng.normal(size=(joint_count, 6, 6))
    inertia = (inertia + np.swapaxes(inertia, 1, 2)).astype(np.float32)

    n = CRBA_ARTICULATIONS
    joint_qd_start = np.concatenate(([0], np.cumsum([lin + ang for lin, ang in JOINT_DOF_DIMS] * n))).astype(np.int32)
    joint_ancestor = np.array([j - 1 if j % joint_count else -1 for j in range(n * joint_count)], dtype=np.int32)
    inputs = [
        wp.array(np.arange(n + 1, dtype=np.int32) * joint_count, dtype=wp.int32, device=device),
        wp.array(np.arange(n, dtype=np.int32) * dof_count, dtype=wp.int32, device=device),
        wp.ones(n, dtype=wp.int32, device=device),
        wp.array(joint_ancestor, dtype=wp.int32, device=device),
        wp.array(np.arange(n * joint_count, dtype=np.int32), dtype=wp.int32, device=device),
        wp.array(joint_qd_start, dtype=wp.int32, device=device),
        wp.array(np.tile(np.array(JOINT_DOF_DIMS, dtype=np.int32), (n, 1)), dtype=wp.int32, device=device),
        wp.array(np.tile(motion, (n, 1)), dtype=wp.spatial_vector, device=device),
        wp.array(np.tile(inertia, (n, 1, 1)), dtype=wp.spatial_matrix, device=device),
        wp.array(np.arange(n, dtype=np.int32), dtype=wp.int32, device=device),
        dof_count,
        0,
        wp.full(n * dof_count, -1, dtype=wp.int32, device=device),
        wp.zeros(1, dtype=wp.float32, device=device),
    ]

    reference = None
    for _launch in range(CRBA_LAUNCHES):
        H = wp.zeros((n, dof_count, dof_count), dtype=wp.float32, device=device)
        wp.launch(crba_fill_par_dof, dim=n * dof_count, inputs=inputs, outputs=[H], device=device, block_dim=128)
        H = H.numpy()
        np.testing.assert_array_equal(H, np.swapaxes(H, 1, 2), err_msg="H is not exactly symmetric")
        np.testing.assert_array_equal(H, np.broadcast_to(H[0], H.shape), err_msg="identical articulations differ")
        if reference is None:
            reference = H
        np.testing.assert_array_equal(H, reference, err_msg="H differs between launches")


def _build_pendulum_model(device):
    """Three-link pendulum in two worlds: its mass matrix changes every step."""
    env = newton.ModelBuilder()
    parent = -1
    joints = []
    for link_index in range(3):
        link = env.add_link()
        env.add_shape_box(link, hx=0.2, hy=0.04, hz=0.04)
        joints.append(
            env.add_joint_revolute(
                parent,
                link,
                parent_xform=wp.transform(wp.vec3(0.2 if parent >= 0 else 0.0, 0.0, 0.0), wp.quat_identity()),
                child_xform=wp.transform(wp.vec3(-0.2, 0.0, 0.0), wp.quat_identity()),
                axis=newton.Axis.Y if link_index != 1 else newton.Axis.Z,
            )
        )
        parent = link
    env.add_articulation(joints)
    builder = newton.ModelBuilder()
    builder.replicate(env, 2)
    model = builder.finalize(device=device)
    model.joint_q.assign(np.tile(np.array([0.7, -0.4, 0.3], dtype=np.float32), 2))
    return model


def _simulate(model, solver, steps):
    state_0, state_1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    control = model.control()
    for _ in range(steps):
        state_0.clear_forces()
        solver.step(state_0, state_1, control, None, 1.0 / 60.0)
        state_0, state_1 = state_1, state_0
    return state_0.body_q.numpy(), state_0.body_qd.numpy()


def test_full_reset_matches_fresh_solver(test, device):
    """After a full reset, eager steps replay a fresh solver's trajectory bit for bit."""
    for interval in (1, 3):
        with test.subTest(update_mass_matrix_interval=interval):
            model = _build_pendulum_model(device)
            reused = SolverFeatherPGS(model, update_mass_matrix_interval=interval)
            _simulate(model, reused, 5)
            reused.reset(model.state())
            body_q, body_qd = _simulate(model, reused, 7)

            fresh_body_q, fresh_body_qd = _simulate(
                model, SolverFeatherPGS(model, update_mass_matrix_interval=interval), 7
            )
            np.testing.assert_array_equal(body_q, fresh_body_q)
            np.testing.assert_array_equal(body_qd, fresh_body_qd)


# Two floating ants dropped onto each other plus a stack of boxes in every world: articulation-articulation,
# articulation-box and box-box contacts, joint limits, PhysX drives, joint velocity limits and joint friction.
ROBOT_SOLVER_OPTIONS = {
    "solve_mode": "matrix_free",
    "drive_mode": "rows",
    "enable_joint_limits": True,
    "enable_joint_velocity_limits": True,
    "enable_joint_friction": True,
    "iterations": 8,
    "articulation_constraint_max": 256,
    "free_body_constraint_max": 256,
    # Rows that do not fit are dropped deterministically too, so overflow stays in scope.
    "warn_constraint_overflow": False,
}


# Torsional and rolling rows ride on the robot options; their capacity fits, so no world overflows.
ANGULAR_FRICTION_SOLVER_OPTIONS = {
    **ROBOT_SOLVER_OPTIONS,
    "enable_torsional_rolling_friction": True,
    "friction_anchor_beta": 0.0,
    "articulation_constraint_max": 512,
}


SPLIT_SOLVER_OPTIONS = {
    "solve_mode": "split",
    "enable_joint_limits": True,
    "articulation_constraint_max": 128,
    "free_body_constraint_max": 256,
    # Rows that do not fit are dropped deterministically too, so overflow stays in scope.
    "warn_constraint_overflow": False,
}


def _build_contact_scene(device, worlds=4, boxes=6, ants=2):
    env = newton.ModelBuilder()
    for ant in range(ants):
        env.add_mjcf(
            newton.examples.get_asset("nv_ant.xml"),
            xform=wp.transform(wp.vec3(0.1 * ant, 0.05 * ant, 0.6 + 0.5 * ant), wp.quat_identity()),
            floating=True,
            ignore_names=["floor"],
        )
    rng = np.random.default_rng(3)
    for box in range(boxes):
        body = env.add_body(
            xform=wp.transform(
                wp.vec3(
                    0.13 * (box % 3) + 0.01 * rng.standard_normal(), 0.13 * ((box // 3) % 2), 0.07 + 0.13 * (box // 6)
                ),
                wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(rng.uniform(0.0, 3.0))),
            )
        )
        env.add_shape_box(body, hx=0.06, hy=0.05, hz=0.06, cfg=newton.ModelBuilder.ShapeConfig(density=300.0, mu=0.7))
    builder = newton.ModelBuilder()
    builder.replicate(env, worlds)
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    model.joint_velocity_limit.fill_(8.0)
    revolute_dofs = model.joint_qd_start.numpy()[:-1][model.joint_type.numpy() == newton.JointType.REVOLUTE]
    friction = np.zeros(model.joint_dof_count, dtype=np.float32)
    friction[revolute_dofs] = 0.01
    model.joint_friction.assign(friction)
    return model


def _build_angular_friction_scene(device, worlds=4):
    """Three floating ants per world with torsional and rolling friction on every shape."""
    model = _build_contact_scene(device, worlds=worlds, boxes=0, ants=3)
    model.shape_material_mu_torsional.fill_(0.02)
    model.shape_material_mu_rolling.fill_(0.01)
    return model


def _actuate(model, control):
    torques = np.zeros(model.joint_dof_count, dtype=np.float32)
    torques[np.arange(model.joint_dof_count) % 14 >= 6] = 30.0
    control.joint_f.assign(torques)


def _sample(state):
    """Hash of the state and whether every value in it is finite."""
    digest = hashlib.sha256()
    finite = True
    for array in (state.body_q, state.body_qd, state.joint_q, state.joint_qd):
        values = np.ascontiguousarray(array.numpy())
        digest.update(values.tobytes())
        finite = finite and bool(np.isfinite(values).all())
    return digest.hexdigest(), finite


def _run_hashes(model, options, steps, graph=False, collision_options=None):
    """Simulate with deterministic collision and solver; return (hash, finite) per step or per graph replay."""
    solver = SolverFeatherPGS(model, deterministic=True, **options)
    pipeline = newton.CollisionPipeline(model, deterministic=True, **(collision_options or {}))
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    _actuate(model, control)

    def step(state_in, state_out):
        state_in.clear_forces()
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, 0.005)

    hashes = []
    if graph:
        with wp.ScopedCapture(device=model.device) as capture:
            step(state_0, state_1)
            step(state_1, state_0)
        for _ in range(steps // 2):
            wp.capture_launch(capture.graph)
            hashes.append(_sample(state_0))
        return hashes
    for _ in range(steps):
        step(state_0, state_1)
        state_0, state_1 = state_1, state_0
        hashes.append(_sample(state_0))
    return hashes


def test_warm_started_full_reset_matches_fresh_solver(test, device):
    """After a full reset, warm-started deterministic steps replay a fresh solver's states and contact forces."""
    model = _build_contact_scene(device, worlds=2, boxes=3, ants=1)
    configurations = {"split": SPLIT_SOLVER_OPTIONS}
    if wp.get_device(device).is_cuda:
        configurations["matrix_free"] = ROBOT_SOLVER_OPTIONS

    def simulate(solver, steps):
        pipeline = newton.CollisionPipeline(model, deterministic=True, contact_matching="latest")
        contacts = pipeline.contacts()
        observables = solver.observables({SolverBase.ObservableKind.CONTACT_F}, contacts=contacts)
        state_0, state_1, control = model.state(), model.state(), model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
        _actuate(model, control)
        for _ in range(steps):
            state_0.clear_forces()
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, 0.005, observables=observables)
            state_0, state_1 = state_1, state_0
        return _sample(state_0), observables.contact_f.numpy()

    for name, base_options in configurations.items():
        with test.subTest(name):
            options = {**base_options, "contact_warmstart": True}
            reused = SolverFeatherPGS(model, deterministic=True, **options)
            simulate(reused, 5)
            reused.reset(model.state())
            (state_hash, finite), contact_f = simulate(reused, 7)
            (fresh_hash, _), fresh_contact_f = simulate(SolverFeatherPGS(model, deterministic=True, **options), 7)
            test.assertTrue(finite)
            test.assertEqual(state_hash, fresh_hash)
            test.assertTrue(np.abs(contact_f).max() > 0.0)
            np.testing.assert_array_equal(contact_f, fresh_contact_f)


def _struct_arrays(value):
    """Arrays passed directly or as fields of a Warp struct."""
    if isinstance(value, wp.array):
        return [value]
    fields = getattr(getattr(value, "_cls", None), "vars", {})
    return [getattr(value, name) for name in fields if isinstance(getattr(value, name), wp.array)]


def _replaying(original, replays, varied):
    """Wrap a launch function: rerun every launch from a snapshot of its arrays and record kernels whose outputs vary."""

    def launch(kernel, dim, inputs=(), outputs=(), **kwargs):
        arrays = [a for v in (*(inputs or ()), *(outputs or ())) for a in _struct_arrays(v) if a.size > 0 and a.ptr]
        wp.synchronize()
        snapshot = [wp.clone(a) for a in arrays]
        reference = None
        for replay in range(replays):
            if replay:
                for array, saved in zip(arrays, snapshot, strict=True):
                    wp.copy(array, saved)
            result = original(kernel, dim=dim, inputs=inputs, outputs=outputs, **kwargs)
            wp.synchronize()
            values = [a.numpy().tobytes() for a in arrays]
            if reference is None:
                reference = values
            elif values != reference:
                varied.add(kernel.key)
        return result

    return launch


def test_deterministic_launches_replay_identically(test, device):
    """Every solver ``wp.launch``/``wp.launch_tiled`` reproduces its outputs bitwise when rerun from the same inputs.

    The native contact radix sort is outside this interception."""
    robot_model = _build_contact_scene(device, worlds=8)
    for name, model, options in (
        ("matrix_free", robot_model, ROBOT_SOLVER_OPTIONS),
        ("split", robot_model, SPLIT_SOLVER_OPTIONS),
        ("angular_friction", _build_angular_friction_scene(device, worlds=8), ANGULAR_FRICTION_SOLVER_OPTIONS),
    ):
        with test.subTest(name):
            solver = SolverFeatherPGS(model, deterministic=True, **options)
            pipeline = newton.CollisionPipeline(model, deterministic=True)
            contacts = pipeline.contacts()
            state_0, state_1, control = model.state(), model.state(), model.control()
            newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
            _actuate(model, control)
            varied = set()
            for _ in range(12):
                state_0.clear_forces()
                pipeline.collide(state_0, contacts)
                with (
                    mock.patch.object(wp, "launch", _replaying(wp.launch, 3, varied)),
                    mock.patch.object(wp, "launch_tiled", _replaying(wp.launch_tiled, 3, varied)),
                ):
                    solver.step(state_0, state_1, control, contacts, 0.005)
                state_0, state_1 = state_1, state_0
            count = int(contacts.rigid_contact_count.numpy()[0])
            test.assertGreater(count, 0)
            if options.get("enable_torsional_rolling_friction"):
                test.assertTrue((solver._contact_slots_needed.numpy()[:count] == 6).any())
            test.assertEqual(varied, set())


def test_deterministic_runs_are_bitwise_identical(test, device):
    """Two runs produce identical finite states at every eager step and every two-step CUDA graph boundary."""
    robot_model = _build_contact_scene(device)
    configurations = [("split", robot_model, SPLIT_SOLVER_OPTIONS, False)]
    if wp.get_device(device).is_cuda:
        angular_model = _build_angular_friction_scene(device)
        configurations += [
            ("matrix_free", robot_model, ROBOT_SOLVER_OPTIONS, False),
            ("matrix_free_graph", robot_model, ROBOT_SOLVER_OPTIONS, True),
            ("angular_friction", angular_model, ANGULAR_FRICTION_SOLVER_OPTIONS, False),
            ("angular_friction_graph", angular_model, ANGULAR_FRICTION_SOLVER_OPTIONS, True),
        ]
    for name, model, options, graph in configurations:
        with test.subTest(name):
            first = _run_hashes(model, options, 60, graph=graph)
            second = _run_hashes(model, options, 60, graph=graph)
            test.assertEqual(first, second)
            test.assertTrue(all(finite for _, finite in first))


def _build_scaled_sdf_mesh_scene(device, scale, worlds=4):
    """Texture-SDF mesh cubes dropped on a heightfield in every world, at mesh ``scale``."""
    cube = newton.Mesh.create_box(0.05, 0.05, 0.05, compute_inertia=False)
    cube.build_sdf(max_resolution=32, device=device)
    n = 17
    xs = np.linspace(-1.0, 1.0, n)
    elevation = (0.5 + 0.5 * np.sin(3.0 * xs)[None, :] * np.cos(2.0 * xs)[:, None]).astype(np.float32)
    env = newton.ModelBuilder()
    rotation = wp.quat_from_axis_angle(wp.normalize(wp.vec3(0.3, 0.2, 1.0)), 0.4)
    for i in range(6):
        body = env.add_body(xform=wp.transform((-0.4 + 0.16 * i, 0.1 * (i % 2), 0.12 + 0.03 * i), rotation))
        env.add_shape_mesh(body=body, mesh=cube, scale=scale, cfg=newton.ModelBuilder.ShapeConfig(density=300.0))
    builder = newton.ModelBuilder()
    builder.replicate(env, worlds)
    builder.add_shape_heightfield(
        heightfield=newton.Heightfield(data=elevation, nrow=n, ncol=n, hx=1.0, hy=1.0, min_z=0.0, max_z=0.04)
    )
    return builder.finalize(device=device)


def test_deterministic_runs_with_dynamic_shape_scale_are_bitwise_identical(test, device):
    """Texture-SDF meshes rescaled after construction under ``dynamic_shape_scale`` repeat bitwise."""
    model = _build_scaled_sdf_mesh_scene(device, (1.0, 1.0, 1.0))
    rescaled = _build_scaled_sdf_mesh_scene(device, (1.6, 1.4, 1.2))
    for name in ("shape_scale", "shape_collision_radius", "shape_collision_aabb_lower", "shape_collision_aabb_upper"):
        getattr(model, name).assign(getattr(rescaled, name))
    collision_options = {"dynamic_shape_scale": True}
    options = {**SPLIT_SOLVER_OPTIONS, "free_body_constraint_max": 2048}
    pipeline = newton.CollisionPipeline(model, deterministic=True, **collision_options)
    test.assertFalse(pipeline.narrow_phase.mesh_sdf_identity_scale_only)
    for graph in (False, True):
        with test.subTest(graph=graph):
            first = _run_hashes(model, options, 60, graph=graph, collision_options=collision_options)
            second = _run_hashes(model, options, 60, graph=graph, collision_options=collision_options)
            test.assertEqual(first, second)
            test.assertTrue(all(finite for _, finite in first))


def test_deterministic_overflow_keeps_earliest_contacts(test, device):
    """On overflow a world keeps a contact-order prefix of its rows and reports the rest as dropped."""
    model = _build_contact_scene(device, worlds=4, boxes=12, ants=0)
    pipeline = newton.CollisionPipeline(model, deterministic=True)
    contacts = pipeline.contacts()
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    pipeline.collide(state, contacts)
    count = int(contacts.rigid_contact_count.numpy()[0])

    def route(capacity):
        solver = SolverFeatherPGS(
            model,
            deterministic=True,
            solve_mode="split",
            free_body_constraint_max=capacity,
            warn_constraint_overflow=False,
        )
        solver.step(state, model.state(), model.control(), contacts, 0.005)
        return (
            solver._contact_world.numpy()[:count],
            solver._contact_slot.numpy()[:count],
            solver._contact_path.numpy()[:count],
            solver._contact_slots_needed.numpy()[:count],
            solver._row_dropped_mf.numpy(),
            solver.constraint_overflow.numpy(),
        )

    world, _, path, needed, _, _ = route(4096)
    small_world, slot, small_path, _, dropped, overflow = route(24)
    for w in range(model.world_count):
        routed = np.nonzero((path == 1) & (world == w))[0]
        kept = np.nonzero((small_path == 1) & (small_world == w))[0]
        test.assertGreater(len(routed), len(kept))
        np.testing.assert_array_equal(kept, routed[: len(kept)])
        np.testing.assert_array_equal(slot[kept], np.concatenate(([0], np.cumsum(needed[kept])[:-1])))
        test.assertEqual(dropped[w], int(needed[routed[len(kept) :]].sum()))
        test.assertTrue(overflow[w])


def test_overflow_warning_names_deterministic_mode(test, device):
    """The row-overflow warning is the same in both modes, plus the drop rule in deterministic mode, which holds."""
    model = _build_contact_scene(device, worlds=2, boxes=12, ants=0)
    pipeline = newton.CollisionPipeline(model, deterministic=True)
    contacts = pipeline.contacts()
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    pipeline.collide(state, contacts)
    warning = r"Warning: FeatherPGS matrix-free constraint-row overflow in world \d+: requested \d+ rows, limit 24; "
    warning += r"dropped \d+ contact/friction rows\. Increase free_body_constraint_max\.\n"
    note = (
        "FeatherPGS deterministic mode: each world keeps the rows of its earliest contacts in contact-buffer order. "
        "While its contacts and joint rows are unchanged it drops the same contacts every step; otherwise it drops "
        "the tail of the new contact order.\n"
    )
    for deterministic in (False, True):
        with test.subTest(deterministic=deterministic):
            solver = SolverFeatherPGS(
                model, deterministic=deterministic, solve_mode="split", free_body_constraint_max=24
            )
            wp.synchronize_device(device)
            capture = StdOutCapture()
            capture.begin()
            try:
                paths = []
                for _ in range(2):
                    solver.step(state, model.state(), model.control(), contacts, 0.005)
                    paths.append(solver._contact_path.numpy())
                wp.synchronize_device(device)
            finally:
                output = capture.end()
            test.assertEqual(len(re.findall(warning, output)), 1, output)
            test.assertEqual(output.count(note), int(deterministic), output)
            test.assertTrue(solver.constraint_overflow.numpy().all())
            if deterministic:
                np.testing.assert_array_equal(paths[0], paths[1], err_msg="same contacts dropped differently")


def test_deterministic_requires_immediate_or_plain_propagation(test, device):
    model = _build_contact_scene(device, worlds=1, boxes=1, ants=1)
    for response in ("propagation_fused", "propagation_colored"):
        with test.subTest(response), test.assertRaisesRegex(ValueError, "deterministic=True requires"):
            SolverFeatherPGS(model, deterministic=True, solve_mode="matrix_free", articulated_contact_response=response)


# A fixed-base three-link arm pressed into a 5x5x2 heap of boxes and spheres: every link and every heap body has
# contacts, about 1.9k per world. Plain propagation carries all of them.
PROPAGATION_SOLVER_OPTIONS = {
    "solve_mode": "matrix_free",
    "articulated_contact_response": "propagation",
    "iterations": 16,
    "articulation_constraint_max": 256,
    "free_body_constraint_max": 6144,
    "friction_anchor_beta": 0.2,
}


def _build_heap_arm_world(seed):
    env = newton.ModelBuilder()
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.6)
    joints, parent = [], -1
    for link_index in range(3):
        link = env.add_link(xform=wp.transform(wp.vec3(-0.12 + 0.1 * link_index, 0.0, 0.1025), wp.quat_identity()))
        env.add_shape_box(link, hx=0.05, hy=0.03, hz=0.02, cfg=cfg)
        parent_xform = wp.transform(wp.vec3(-0.17, 0.0, 0.1025) if parent < 0 else wp.vec3(0.05, 0.0, 0.0))
        joints.append(
            env.add_joint_revolute(
                parent,
                link,
                axis=newton.Axis.Y,
                parent_xform=parent_xform,
                child_xform=wp.transform(wp.vec3(-0.05, 0.0, 0.0)),
                target_kd=0.2,
            )
        )
        parent = link
    env.add_articulation(joints)
    rng = np.random.default_rng(seed)
    for ix in range(5):
        for iy in range(5):
            for iz in range(2):
                pose = wp.transform(
                    wp.vec3((ix - 2) * 0.042 + 0.0008 * rng.uniform(-1.0, 1.0), (iy - 2) * 0.042, 0.021 + 0.041 * iz),
                    wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), float(rng.uniform(-0.03, 0.03))),
                )
                body = env.add_body(xform=pose)
                if (ix + iy + iz) % 2:
                    env.add_shape_sphere(body, radius=0.02, cfg=cfg)
                else:
                    env.add_shape_box(body, hx=0.02, hy=0.02, hz=0.02, cfg=cfg)
    return env


def _build_heap_arm_scene(device, seeds, global_floor=False):
    """One heap-and-arm world per seed, in the given order, so world contents follow their seed, not their slot.

    ``global_floor`` replaces the ground plane with a kinematic box outside every world, which all worlds touch.
    """
    builder = newton.ModelBuilder()
    for seed in seeds:
        builder.add_world(_build_heap_arm_world(seed))
    if global_floor:
        floor = builder.add_link(is_kinematic=True)
        builder.add_shape_box(floor, hx=1.0, hy=1.0, hz=0.02, xform=wp.transform(wp.vec3(0.0, 0.0, -0.02)))
    else:
        builder.add_ground_plane()
    return builder.finalize(device=device)


def _run_heap_arm(model, steps, *, cached, warmstart, graph=False, on_step=None):
    """Return each world's body poses after every second step, and the final state hash."""
    solver = SolverFeatherPGS(
        model,
        deterministic=True,
        propagation_cached_response=cached,
        contact_warmstart=warmstart,
        **PROPAGATION_SOLVER_OPTIONS,
    )
    pipeline = newton.CollisionPipeline(
        model, deterministic=True, contact_matching="sticky" if warmstart else "disabled"
    )
    contacts = pipeline.contacts()
    state_0, state_1, control = model.state(), model.state(), model.control()
    torques = np.zeros(model.joint_dof_count, dtype=np.float32)
    torques[model.joint_qd_start.numpy()[:-1][model.joint_type.numpy() == newton.JointType.REVOLUTE]] = -0.3
    control.joint_f.assign(torques)

    def step(state_in, state_out):
        state_in.clear_forces()
        pipeline.collide(state_in, contacts)
        solver.step(state_in, state_out, control, contacts, 1.0 / 240.0)

    poses = []
    if graph:
        with wp.ScopedCapture(device=model.device) as capture:
            step(state_0, state_1)
            step(state_1, state_0)
        for _ in range(steps // 2):
            wp.capture_launch(capture.graph)
            poses.append(state_0.body_q.numpy())
    else:
        for index in range(steps):
            step(state_0, state_1)
            state_0, state_1 = state_1, state_0
            if on_step is not None:
                on_step(index, solver, contacts)
            if index % 2:
                poses.append(state_0.body_q.numpy())
    per_world = np.stack(poses).reshape(len(poses), model.world_count, -1, 7)
    return per_world, _sample(state_0)


def test_deterministic_propagation_runs_are_bitwise_identical(test, device):
    """Repeat plain propagation bitwise, eager and captured, with and without the cached response and warm start."""
    model = _build_heap_arm_scene(device, (0, 1, 2))
    arm_bodies = np.arange(3)
    seen = {}

    def observe(index, solver, contacts):
        if index != 0:
            return
        count = int(contacts.rigid_contact_count.numpy()[0])
        body_0 = model.shape_body.numpy()[contacts.rigid_contact_shape0.numpy()[:count]]
        body_1 = model.shape_body.numpy()[contacts.rigid_contact_shape1.numpy()[:count]]
        bodies = np.concatenate([body_0, body_1])
        seen["contacts_per_world"] = np.bincount(
            model.body_world.numpy()[np.where(body_0 >= 0, body_0, body_1)], minlength=3
        )
        seen["active_links"] = np.intersect1d(bodies, arm_bodies).size
        seen["paths"] = np.bincount(solver._contact_path.numpy()[:count] + 1, minlength=4)

    _run_heap_arm(model, 2, cached=True, warmstart=False, on_step=observe)
    test.assertGreater(seen["contacts_per_world"].min(), 1000)
    test.assertEqual(seen["active_links"], 3)
    # Contacts between the arm's own links stay dense; every other contact takes the propagation route.
    test.assertEqual(seen["paths"][0] + seen["paths"][2], 0)
    test.assertGreater(seen["paths"][3], 20 * seen["paths"][1])
    first = {}
    for cached in (True, False):
        for warmstart in (False, True):
            for graph in (False, True):
                with test.subTest(cached=cached, warmstart=warmstart, graph=graph):
                    runs = [_run_heap_arm(model, 40, cached=cached, warmstart=warmstart, graph=graph) for _ in range(3)]
                    for poses, sample in runs[1:]:
                        np.testing.assert_array_equal(poses, runs[0][0])
                        test.assertEqual(sample, runs[0][1])
                    test.assertTrue(runs[0][1][1])
                    first[cached, warmstart, graph] = runs[0]
            with test.subTest(cached=cached, warmstart=warmstart, graph="eager vs captured"):
                np.testing.assert_array_equal(first[cached, warmstart, True][0], first[cached, warmstart, False][0])
                test.assertEqual(first[cached, warmstart, True][1], first[cached, warmstart, False][1])


def test_deterministic_propagation_is_world_permutation_invariant(test, device):
    """Give each world the same trajectory, bitwise, whichever slot it occupies."""
    order = (2, 0, 1)
    in_order = _build_heap_arm_scene(device, (0, 1, 2))
    permuted = _build_heap_arm_scene(device, order)
    for cached in (True, False):
        for warmstart in (False, True):
            with test.subTest(cached=cached, warmstart=warmstart):
                reference, _ = _run_heap_arm(in_order, 40, cached=cached, warmstart=warmstart)
                moved, _ = _run_heap_arm(permuted, 40, cached=cached, warmstart=warmstart)
                for slot, seed in enumerate(order):
                    np.testing.assert_array_equal(moved[:, slot], reference[:, seed])


@wp.kernel
def _accumulate_warmstart_in_reverse_row_order(
    constraint_count: wp.array[int],
    body_a: wp.array2d[int],
    body_b: wp.array2d[int],
    J_a: wp.array3d[float],
    J_b: wp.array3d[float],
    MiJt_a: wp.array3d[float],
    MiJt_b: wp.array3d[float],
    impulses: wp.array2d[float],
    max_constraints: int,
    body_qd: wp.array2d[float],
    body_impulses: wp.array2d[float],
):
    """Seed the warm start in reverse row order: one order an atomic scatter may take."""
    tid = wp.tid()
    world = tid // 6
    k = tid - world * 6
    count = wp.min(constraint_count[world], max_constraints)
    for step in range(count):
        row = count - 1 - step
        impulse = impulses[world, row]
        if impulse != 0.0:
            ba = body_a[world, row]
            bb = body_b[world, row]
            if ba >= 0:
                body_qd[ba, k] = body_qd[ba, k] + MiJt_a[world, row, k] * impulse
                body_impulses[ba, k] = body_impulses[ba, k] + J_a[world, row, k] * impulse
            if bb >= 0:
                body_qd[bb, k] = body_qd[bb, k] + MiJt_b[world, row, k] * impulse
                body_impulses[bb, k] = body_impulses[bb, k] + J_b[world, row, k] * impulse


def test_deterministic_propagation_warm_start_sums_in_row_order(test, device):
    """Seed warm-started impulses in row order, which a reordered sum over a shared body does not reproduce."""
    model = _build_heap_arm_scene(device, (1,))
    seen = {"shared": 0, "arm_links": 0}
    seed = SolverFeatherPGS._apply_propagation_warmstart_velocity

    def recording_seed(solver):
        count = int(solver._propagation_constraint_count.numpy()[0])
        seeded = solver._propagation_impulses.numpy()[0, :count] != 0.0
        bodies = np.concatenate(
            [
                solver._propagation_body_a.numpy()[0, :count][seeded],
                solver._propagation_body_b.numpy()[0, :count][seeded],
            ]
        )
        bodies = bodies[bodies >= 0]
        if bodies.size:
            seen["shared"] = max(seen["shared"], int(np.bincount(bodies).max()))
            seen["arm_links"] = max(seen["arm_links"], np.intersect1d(bodies, np.arange(3)).size)
        seed(solver)

    with mock.patch.object(SolverFeatherPGS, "_apply_propagation_warmstart_velocity", recording_seed):
        ordered, ordered_sample = _run_heap_arm(model, 40, cached=False, warmstart=True)
    test.assertGreater(seen["shared"], 2)
    test.assertGreaterEqual(seen["arm_links"], 2)
    repeated, repeated_sample = _run_heap_arm(model, 40, cached=False, warmstart=True)
    np.testing.assert_array_equal(repeated, ordered)
    test.assertEqual(repeated_sample, ordered_sample)
    with mock.patch.object(
        solver_feather_pgs,
        "accumulate_propagation_warmstart_body_impulses_ordered",
        _accumulate_warmstart_in_reverse_row_order,
    ):
        reversed_order, _ = _run_heap_arm(model, 40, cached=False, warmstart=True)
    test.assertFalse(np.array_equal(reversed_order, ordered))


def test_deterministic_propagation_rejects_bodies_shared_by_worlds(test, device):
    """Reject a shaped body whose contacts can reach several worlds; accept it without the flag or on one world."""
    shared = {"outside every world": _build_heap_arm_scene(device, (0, 1), global_floor=True)}
    # A world-local floor whose shape is added outside every world collides with both worlds.
    builder = newton.ModelBuilder()
    first = _build_heap_arm_world(0)
    floor = first.add_link(is_kinematic=True)
    builder.add_world(first)
    builder.add_world(_build_heap_arm_world(1))
    builder.add_shape_box(floor, hx=1.0, hy=1.0, hz=0.02, xform=wp.transform(wp.vec3(0.0, 0.0, -0.02)))
    shared["global shape on a world body"] = builder.finalize(device=device)
    for name, model in shared.items():
        with test.subTest(name):
            with test.assertRaisesRegex(ValueError, "each shaped body and its shapes to share one world"):
                SolverFeatherPGS(model, deterministic=True, **PROPAGATION_SOLVER_OPTIONS)
            SolverFeatherPGS(model, **PROPAGATION_SOLVER_OPTIONS)
    single = _build_heap_arm_scene(device, (0,), global_floor=True)
    SolverFeatherPGS(single, deterministic=True, **PROPAGATION_SOLVER_OPTIONS)


def test_deterministic_propagation_drops_explicit_cross_world_pairs(test, device):
    """Keep a world's prescribed body in that world's rows when explicit pairs join it to another world's shapes."""
    builder = newton.ModelBuilder()
    first = _build_heap_arm_world(0)
    floor = first.add_link(is_kinematic=True)
    floor_shape = first.add_shape_box(floor, hx=1.0, hy=1.0, hz=0.02, xform=wp.transform(wp.vec3(0.0, 0.0, -0.02)))
    builder.add_world(first)
    builder.add_world(_build_heap_arm_world(1))
    model = builder.finalize(device=device)
    pairs = np.array(
        [[floor_shape, shape] for shape in range(model.shape_count) if shape != floor_shape], dtype=np.int32
    )
    rows_by_world = {}
    for deterministic in (False, True):
        solver = SolverFeatherPGS(
            model, deterministic=deterministic, warn_constraint_overflow=False, **PROPAGATION_SOLVER_OPTIONS
        )
        pipeline = newton.CollisionPipeline(
            model,
            deterministic=True,
            broad_phase="explicit",
            shape_pairs_filtered=wp.array(pairs, dtype=wp.vec2i, device=device),
        )
        contacts = pipeline.contacts()
        state_0, state_1 = model.state(), model.state()
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, model.control(), contacts, 1.0 / 240.0)
        counts = solver._propagation_constraint_count.numpy()
        sides = [solver._propagation_body_a.numpy(), solver._propagation_body_b.numpy()]
        rows_by_world[deterministic] = [
            int(sum(np.count_nonzero(side[world, :count] == floor) for side in sides))
            for world, count in enumerate(counts)
        ]
        overflow = solver.constraint_overflow.numpy()
    # Without the flag the pairs reach both worlds' rows; with it only the floor's own world keeps them.
    test.assertGreater(min(rows_by_world[False]), 0)
    test.assertGreater(rows_by_world[True][0], 0)
    test.assertEqual(rows_by_world[True][1], 0)
    test.assertTrue(overflow[1])


def test_deterministic_joint_rows_follow_articulation_order(test, device):
    """Drive, joint-limit and velocity-limit rows of a world take their slots in articulation order."""
    arts_per_world = 40
    env = newton.ModelBuilder()
    for art in range(arts_per_world):
        link = env.add_link(xform=wp.transform(wp.vec3(0.5 * art, 0.0, 1.0), wp.quat_identity()))
        env.add_shape_box(link, hx=0.1, hy=0.02, hz=0.02)
        joint = env.add_joint_revolute(
            -1,
            link,
            parent_xform=wp.transform(wp.vec3(0.5 * art, 0.0, 1.0), wp.quat_identity()),
            axis=newton.Axis.Y,
            target_ke=10.0,
            limit_lower=-0.1,
            limit_upper=0.1,
            velocity_limit=5.0,
        )
        env.add_articulation([joint])
    builder = newton.ModelBuilder()
    builder.replicate(env, 2)
    model = builder.finalize(device=device)
    # Every joint violates its upper limit by a different amount, so limit rows are told apart by phi.
    q = 0.2 + 0.001 * np.arange(model.joint_coord_count, dtype=np.float32)
    model.joint_q.assign(q)
    solver = SolverFeatherPGS(
        model,
        deterministic=True,
        solve_mode="matrix_free",
        drive_mode="rows",
        enable_joint_limits=True,
        joint_limit_activation_gap=0.0,
        enable_joint_velocity_limits=True,
        fuse_joint_velocity_limits=False,
        articulation_constraint_max=4 * arts_per_world,
    )
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    solver.step(state, model.state(), model.control(), None, 0.005)

    order = np.arange(arts_per_world)
    drive_slot = solver._drive_slot.numpy().reshape(2, arts_per_world)
    velocity_limit_slot = solver._velocity_limit_slot.numpy().reshape(2, arts_per_world, 2)
    row_type = solver._row_type.numpy()
    phi = solver._phi.numpy()
    for world in range(2):
        np.testing.assert_array_equal(drive_slot[world], order)
        limit_rows = np.nonzero(row_type[world] == PGS_CONSTRAINT_TYPE_JOINT_LIMIT)[0]
        np.testing.assert_array_equal(limit_rows, arts_per_world + order)
        np.testing.assert_allclose(phi[world, limit_rows], 0.1 - q.reshape(2, arts_per_world)[world], atol=1.0e-6)
        np.testing.assert_array_equal(
            velocity_limit_slot[world].reshape(-1), 2 * arts_per_world + np.arange(2 * arts_per_world)
        )


def test_deterministic_angular_friction_rows_follow_their_contact(test, device):
    """A contact's torsional and rolling rows follow its sliding pair, and on overflow it keeps or drops all six."""
    model = _build_angular_friction_scene(device)
    pipeline = newton.CollisionPipeline(model, deterministic=True)
    contacts = pipeline.contacts()
    state, state_next = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    settle = SolverFeatherPGS(model, **ANGULAR_FRICTION_SOLVER_OPTIONS)
    # Settle the ants onto the ground so every world has dense contacts.
    for _ in range(40):
        pipeline.collide(state, contacts)
        settle.step(state, state_next, model.control(), contacts, 0.005)
        state, state_next = state_next, state
    pipeline.collide(state, contacts)
    count = int(contacts.rigid_contact_count.numpy()[0])

    def route(capacity):
        options = {**ANGULAR_FRICTION_SOLVER_OPTIONS, "articulation_constraint_max": capacity}
        solver = SolverFeatherPGS(model, deterministic=True, **options)
        solver.step(state, model.state(), model.control(), contacts, 0.005)
        return (
            solver._contact_world.numpy()[:count],
            solver._contact_slot.numpy()[:count],
            solver._contact_path.numpy()[:count],
            solver._contact_slots_needed.numpy()[:count],
            solver._row_type.numpy(),
            solver._row_parent.numpy(),
            solver._row_dropped_dense.numpy(),
            solver.constraint_overflow.numpy(),
        )

    world, slot, path, needed, row_type, row_parent, _, overflow = route(512)
    test.assertFalse(overflow.any())
    small_world, small_slot, small_path, _, _, _, dropped, small_overflow = route(192)
    angular = []
    for w in range(model.world_count):
        routed = np.nonzero((path == 0) & (world == w))[0]
        test.assertTrue((needed[routed] == 6).any())
        np.testing.assert_array_equal(
            slot[routed], slot[routed[0]] + np.concatenate(([0], np.cumsum(needed[routed])[:-1]))
        )
        for c in routed[needed[routed] == 6]:
            rows = slot[c] + np.arange(1, 6)
            angular.append(row_type[w, rows])
            np.testing.assert_array_equal(row_parent[w, rows], slot[c])
        kept = np.nonzero((small_path == 0) & (small_world == w))[0]
        test.assertGreater(len(routed), len(kept))
        np.testing.assert_array_equal(kept, routed[: len(kept)])
        np.testing.assert_array_equal(small_slot[kept], slot[kept])
        test.assertEqual(dropped[w], int(needed[routed[len(kept) :]].sum()))
        test.assertTrue(small_overflow[w])
    np.testing.assert_array_equal(np.array(angular), PGS_CONSTRAINT_TYPE_CONTACT_ANGULAR_FRICTION)


class TestFeatherPGSDeterminism(unittest.TestCase):
    pass


for _device in get_test_devices():
    add_function_test(
        TestFeatherPGSDeterminism,
        "test_crba_fill_par_dof_is_batch_invariant",
        test_crba_fill_par_dof_is_batch_invariant,
        devices=[_device],
    )
    add_function_test(
        TestFeatherPGSDeterminism,
        "test_full_reset_matches_fresh_solver",
        test_full_reset_matches_fresh_solver,
        devices=[_device],
    )
    add_function_test(
        TestFeatherPGSDeterminism,
        "test_warm_started_full_reset_matches_fresh_solver",
        test_warm_started_full_reset_matches_fresh_solver,
        devices=[_device],
    )
    add_function_test(
        TestFeatherPGSDeterminism,
        "test_deterministic_runs_are_bitwise_identical",
        test_deterministic_runs_are_bitwise_identical,
        devices=[_device],
    )
    add_function_test(
        TestFeatherPGSDeterminism,
        "test_deterministic_overflow_keeps_earliest_contacts",
        test_deterministic_overflow_keeps_earliest_contacts,
        devices=[_device],
    )
    add_function_test(
        TestFeatherPGSDeterminism,
        "test_overflow_warning_names_deterministic_mode",
        test_overflow_warning_names_deterministic_mode,
        devices=[_device],
        check_output=False,
    )
    add_function_test(
        TestFeatherPGSDeterminism,
        "test_deterministic_requires_immediate_or_plain_propagation",
        test_deterministic_requires_immediate_or_plain_propagation,
        devices=[_device],
    )


for _device in get_test_devices():
    if wp.get_device(_device).is_cuda:
        add_function_test(
            TestFeatherPGSDeterminism,
            "test_deterministic_launches_replay_identically",
            test_deterministic_launches_replay_identically,
            devices=[_device],
        )
        add_function_test(
            TestFeatherPGSDeterminism,
            "test_deterministic_runs_with_dynamic_shape_scale_are_bitwise_identical",
            test_deterministic_runs_with_dynamic_shape_scale_are_bitwise_identical,
            devices=[_device],
        )
        add_function_test(
            TestFeatherPGSDeterminism,
            "test_deterministic_angular_friction_rows_follow_their_contact",
            test_deterministic_angular_friction_rows_follow_their_contact,
            devices=[_device],
        )
        add_function_test(
            TestFeatherPGSDeterminism,
            "test_deterministic_joint_rows_follow_articulation_order",
            test_deterministic_joint_rows_follow_articulation_order,
            devices=[_device],
        )
        add_function_test(
            TestFeatherPGSDeterminism,
            "test_deterministic_propagation_runs_are_bitwise_identical",
            test_deterministic_propagation_runs_are_bitwise_identical,
            devices=[_device],
        )
        add_function_test(
            TestFeatherPGSDeterminism,
            "test_deterministic_propagation_is_world_permutation_invariant",
            test_deterministic_propagation_is_world_permutation_invariant,
            devices=[_device],
        )
        add_function_test(
            TestFeatherPGSDeterminism,
            "test_deterministic_propagation_warm_start_sums_in_row_order",
            test_deterministic_propagation_warm_start_sums_in_row_order,
            devices=[_device],
        )
        add_function_test(
            TestFeatherPGSDeterminism,
            "test_deterministic_propagation_rejects_bodies_shared_by_worlds",
            test_deterministic_propagation_rejects_bodies_shared_by_worlds,
            devices=[_device],
        )
        add_function_test(
            TestFeatherPGSDeterminism,
            "test_deterministic_propagation_drops_explicit_cross_world_pairs",
            test_deterministic_propagation_drops_explicit_cross_world_pairs,
            devices=[_device],
        )


if __name__ == "__main__":
    unittest.main()
