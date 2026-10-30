# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverSteel
from newton.tests.unittest_utils import add_function_test, get_selected_cuda_test_devices


def _chain(builder, x, num_links, top):
    joints = []
    parent = -1
    for i in range(num_links):
        link = builder.add_link(xform=wp.transform(wp.vec3(x, 0.0, top - 0.3 * i), wp.quat_identity()))
        builder.add_shape_capsule(link, radius=0.04, half_height=0.12)
        joints.append(
            builder.add_joint_revolute(
                parent,
                link,
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent_xform=wp.transform(
                    wp.vec3(x, 0.0, top + 0.15) if parent < 0 else wp.vec3(0.0, 0.0, -0.15), wp.quat_identity()
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
    builder.add_articulation(joints)


def _free_box(builder, x, z):
    body = builder.add_link(xform=wp.transform(wp.vec3(x, 0.0, z), wp.quat_identity()))
    builder.add_shape_box(body, hx=0.08, hy=0.08, hz=0.08)
    builder.add_articulation([builder.add_joint_free(parent=-1, child=body)])


def _build_two_island_model(device, num_worlds=2):
    """Per world: a chain resting on a box, a separate chain, and a lone box, so rows form three islands."""
    world = newton.ModelBuilder()
    _chain(world, 0.0, 3, 0.92)
    _free_box(world, 0.0, 0.09)
    _chain(world, 1.5, 3, 1.05)
    _free_box(world, 3.0, 0.2)
    scene = newton.ModelBuilder()
    scene.add_ground_plane()
    scene.replicate(world, num_worlds, spacing=(0.0, 6.0, 0.0))
    return scene.finalize(device=device)


def _build_crossed_slider_model(device):
    """A box sliding on z resting on a plate sliding on x: the normal row's J misses the plate, its friction rows' J the box."""
    builder = newton.ModelBuilder()
    shape = newton.ModelBuilder.ShapeConfig(mu=0.8)
    for z, axis, half in ((0.1, newton.Axis.Z, (0.1, 0.1, 0.1)), (-0.05, newton.Axis.X, (1.0, 1.0, 0.05))):
        pose = wp.transform(wp.vec3(0.0, 0.0, z), wp.quat_identity())
        body = builder.add_link(xform=pose, mass=1.0, inertia=wp.mat33(np.eye(3) * 0.01), lock_inertia=True)
        builder.add_shape_box(body, hx=half[0], hy=half[1], hz=half[2], cfg=shape)
        builder.add_articulation([builder.add_joint_prismatic(-1, body, axis=axis, parent_xform=pose)])
    model = builder.finalize(device=device)
    model.joint_qd.assign(np.array([-0.2, 1.0], dtype=np.float32))
    return model


def _rollout(model, steps, **options):
    solver = SolverSteel(
        model,
        solve_mode="matrix_free",
        iterations=12,
        enable_joint_limits=True,
        articulation_constraint_max=64,
        free_body_constraint_max=64,
        **options,
    )
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state, next_state, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    trace = []
    coupled_steps = 0
    for _ in range(steps):
        state.clear_forces()
        pipeline.collide(state, contacts)
        solver.step(state, next_state, control, contacts, 1.0 / 240.0)
        state, next_state = next_state, state
        trace.append(np.concatenate([state.body_q.numpy().ravel(), state.body_qd.numpy().ravel()]))
        coupled_steps += _has_cross_articulation_row(solver)
    return solver, np.asarray(trace), coupled_steps


def _has_cross_articulation_row(solver):
    """Whether world 0 has a dense row whose Jacobian spans two articulations."""
    if solver._island_dof_component is None:
        return False
    components = solver._island_dof_component.numpy()[0]
    rows = int(solver._constraint_count.numpy()[0])
    jacobian = solver._J_world.numpy()[0, :rows]
    return any(len(set(components[np.flatnonzero(row)])) > 1 for row in jacobian)


def test_islands_match_one_warp_bitwise(test, device, **options):
    """The island sweep reproduces the one-warp sweep bit for bit across contact, limit and free-body rows."""
    model = _build_two_island_model(device)
    reference_solver, reference, _ = _rollout(model, 150, world_parallel="warp", **options)
    solver, trace, coupled_steps = _rollout(model, 150, world_parallel="islands", **options)
    test.assertFalse(reference_solver._pgs_solve_mf_gs_islands_kernels)
    test.assertGreater(solver._island_warps, 1)
    test.assertGreater(coupled_steps, 0)
    test.assertTrue(np.isfinite(reference).all())
    np.testing.assert_array_equal(trace.view(np.uint32), reference.view(np.uint32))


def test_islands_join_rows_of_one_contact(test, device):
    """A contact whose normal and friction rows touch different articulations stays on one warp."""
    model = _build_crossed_slider_model(device)
    _, reference, _ = _rollout(model, 40, world_parallel="warp")
    for world_parallel in ("islands", "auto"):
        solver, trace, _ = _rollout(model, 40, world_parallel=world_parallel)
        test.assertGreater(solver._island_warps, 1)
        np.testing.assert_array_equal(trace.view(np.uint32), reference.view(np.uint32))


def test_islands_refuse_articulated_angular_friction(test, device):
    """The torsional/rolling block writes velocities through a helper, so islands refuse it and auto keeps one warp."""
    builder = newton.ModelBuilder()
    material = newton.ModelBuilder.ShapeConfig(mu=0.5, mu_rolling=0.01, mu_torsional=0.01, density=0.0)
    builder.add_ground_plane(cfg=material)
    axes = [newton.ModelBuilder.JointDofConfig(axis=axis) for axis in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
    for x in (0.0, 1.0):
        pose = wp.transform(wp.vec3(x, 0.0, 0.1), wp.quat_identity())
        body = builder.add_link(xform=pose, mass=1.0, inertia=wp.mat33(np.eye(3) * 0.004), lock_inertia=True)
        builder.add_articulation(
            [builder.add_joint_d6(-1, body, parent_xform=pose, linear_axes=axes, angular_axes=axes)]
        )
        builder.add_shape_sphere(body, radius=0.1, cfg=material)
    model = builder.finalize(device=device)
    options = {"solve_mode": "matrix_free", "enable_torsional_rolling_friction": True, "friction_anchor_beta": 0.0}
    with test.assertRaises(ValueError):
        SolverSteel(model, world_parallel="islands", **options)
    test.assertFalse(SolverSteel(model, world_parallel="auto", **options)._pgs_solve_mf_gs_islands_kernels)


def test_auto_selects_islands_for_few_worlds(test, device):
    model = _build_two_island_model(device, num_worlds=1)
    solver = SolverSteel(model, solve_mode="matrix_free", world_parallel="auto")
    test.assertTrue(solver._pgs_solve_mf_gs_islands_kernels)
    test.assertEqual(solver.world_parallel, "auto")


def test_islands_refuse_oversized_shared_memory(test, device):
    """A row capacity whose island lists would pass the static shared-memory limit is refused, and auto keeps one warp."""
    model = _build_two_island_model(device, num_worlds=1)
    options = {"solve_mode": "matrix_free", "articulation_constraint_max": 1024, "free_body_constraint_max": 10240}
    with test.assertRaises(ValueError):
        SolverSteel(model, world_parallel="islands", **options)
    test.assertFalse(SolverSteel(model, world_parallel="auto", **options)._pgs_solve_mf_gs_islands_kernels)


class TestSteelWorldParallel(unittest.TestCase):
    def test_rejects_unknown_world_parallel(self):
        model = _build_two_island_model("cpu", num_worlds=1)
        with self.assertRaises(ValueError):
            SolverSteel(model, world_parallel="block")


_devices = get_selected_cuda_test_devices()
for _name, _options in (
    ("default", {}),
    ("phased", {"row_schedule": "phased"}),
    ("contact_then_internal", {"row_schedule": "contact_then_internal"}),
    ("bisection", {"friction_mode": "bisection"}),
    ("velocity_limits", {"enable_joint_velocity_limits": True, "velocity_limit_activation_fraction": 0.0}),
    ("sleeping", {"enable_sleeping": True}),
    ("drive_rows", {"drive_mode": "rows"}),
):
    add_function_test(
        TestSteelWorldParallel,
        f"test_islands_match_one_warp_bitwise_{_name}",
        test_islands_match_one_warp_bitwise,
        devices=_devices,
        **_options,
    )
add_function_test(
    TestSteelWorldParallel,
    "test_islands_join_rows_of_one_contact",
    test_islands_join_rows_of_one_contact,
    devices=_devices,
)
add_function_test(
    TestSteelWorldParallel,
    "test_islands_refuse_oversized_shared_memory",
    test_islands_refuse_oversized_shared_memory,
    devices=_devices,
)
add_function_test(
    TestSteelWorldParallel,
    "test_islands_refuse_articulated_angular_friction",
    test_islands_refuse_articulated_angular_friction,
    devices=_devices,
)
add_function_test(
    TestSteelWorldParallel,
    "test_auto_selects_islands_for_few_worlds",
    test_auto_selects_islands_for_few_worlds,
    devices=_devices,
)


if __name__ == "__main__":
    unittest.main(verbosity=2)
