# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import hashlib
import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
import newton.examples
from newton._src.solvers.feather_pgs.kernels import PGS_CONSTRAINT_TYPE_JOINT_LIMIT, crba_fill_par_dof
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_test_devices

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
    "pgs_mode": "matrix_free",
    "drive_mode": "physx_pgs",
    "enable_joint_limits": True,
    "enable_joint_velocity_limits": True,
    "enable_joint_friction": True,
    "pgs_iterations": 8,
    "dense_max_constraints": 256,
    "mf_max_constraints": 256,
    # Rows that do not fit are dropped deterministically too, so overflow stays in scope.
    "warn_constraint_overflow": False,
}

SPLIT_SOLVER_OPTIONS = {
    "pgs_mode": "split",
    "enable_joint_limits": True,
    "dense_max_constraints": 128,
    "mf_max_constraints": 256,
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


def _actuate(model, control):
    torques = np.zeros(model.joint_dof_count, dtype=np.float32)
    torques[np.arange(model.joint_dof_count) % 14 >= 6] = 30.0
    control.joint_f.assign(torques)


def _hash(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array.numpy()).tobytes())
    return digest.hexdigest()


def _run_hashes(model, options, steps, graph=False):
    """Simulate with deterministic collision and solver; return one state hash per step."""
    solver = SolverFeatherPGS(model, deterministic=True, **options)
    pipeline = newton.CollisionPipeline(model, deterministic=True)
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
            hashes.append(_hash(state_0.body_q, state_0.body_qd, state_0.joint_q, state_0.joint_qd))
        return hashes
    for _ in range(steps):
        step(state_0, state_1)
        state_0, state_1 = state_1, state_0
        hashes.append(_hash(state_0.body_q, state_0.body_qd, state_0.joint_q, state_0.joint_qd))
    return hashes


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
    """Every solver launch reproduces its outputs bitwise when rerun from the same inputs."""
    model = _build_contact_scene(device, worlds=8)
    for options in (
        ROBOT_SOLVER_OPTIONS,
        SPLIT_SOLVER_OPTIONS,
    ):
        with test.subTest(pgs_mode=options["pgs_mode"]):
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
            test.assertGreater(int(contacts.rigid_contact_count.numpy()[0]), 0)
            test.assertEqual(varied, set())


def test_deterministic_runs_are_bitwise_identical(test, device):
    """Two runs from the same model produce identical states at every step, eagerly and in a CUDA graph."""
    model = _build_contact_scene(device)
    configurations = [("split", SPLIT_SOLVER_OPTIONS, False)]
    if wp.get_device(device).is_cuda:
        configurations += [
            ("matrix_free", ROBOT_SOLVER_OPTIONS, False),
            ("matrix_free_graph", ROBOT_SOLVER_OPTIONS, True),
        ]
    for name, options, graph in configurations:
        with test.subTest(name):
            first = _run_hashes(model, options, 60, graph=graph)
            second = _run_hashes(model, options, 60, graph=graph)
            test.assertEqual(first, second)


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
            pgs_mode="split",
            mf_max_constraints=capacity,
            warn_constraint_overflow=False,
        )
        solver.step(state, model.state(), model.control(), contacts, 0.005)
        return (
            solver.contact_world.numpy()[:count],
            solver.contact_slot.numpy()[:count],
            solver.contact_path.numpy()[:count],
            solver.contact_slots_needed.numpy()[:count],
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


def test_deterministic_requires_immediate_contact_response(test, device):
    model = _build_contact_scene(device, worlds=1, boxes=1, ants=1)
    with test.assertRaisesRegex(ValueError, "deterministic=True requires"):
        SolverFeatherPGS(model, deterministic=True, pgs_mode="matrix_free", articulated_contact_response="propagation")


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
        pgs_mode="matrix_free",
        drive_mode="physx_pgs",
        enable_joint_limits=True,
        joint_limit_activation_gap=0.0,
        enable_joint_velocity_limits=True,
        fuse_joint_velocity_limits=False,
        dense_max_constraints=4 * arts_per_world,
    )
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    solver.step(state, model.state(), model.control(), None, 0.005)

    order = np.arange(arts_per_world)
    drive_slot = solver.drive_slot.numpy().reshape(2, arts_per_world)
    velocity_limit_slot = solver.velocity_limit_slot.numpy().reshape(2, arts_per_world, 2)
    row_type = solver.row_type.numpy()
    phi = solver.phi.numpy()
    for world in range(2):
        np.testing.assert_array_equal(drive_slot[world], order)
        limit_rows = np.nonzero(row_type[world] == PGS_CONSTRAINT_TYPE_JOINT_LIMIT)[0]
        np.testing.assert_array_equal(limit_rows, arts_per_world + order)
        np.testing.assert_allclose(phi[world, limit_rows], 0.1 - q.reshape(2, arts_per_world)[world], atol=1.0e-6)
        np.testing.assert_array_equal(
            velocity_limit_slot[world].reshape(-1), 2 * arts_per_world + np.arange(2 * arts_per_world)
        )


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
        "test_deterministic_requires_immediate_contact_response",
        test_deterministic_requires_immediate_contact_response,
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
            "test_deterministic_joint_rows_follow_articulation_order",
            test_deterministic_joint_rows_follow_articulation_order,
            devices=[_device],
        )


if __name__ == "__main__":
    unittest.main()
