# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverSteel
from newton.tests.unittest_utils import add_function_test, get_selected_cuda_test_devices


def _build_chains_and_boxes(device):
    """Two limited chains, one resting on a free box, and a lone box, in two worlds."""
    world = newton.ModelBuilder()
    for x, top in ((0.0, 0.92), (1.5, 1.05)):
        joints, parent = [], -1
        for i in range(3):
            link = world.add_link(xform=wp.transform(wp.vec3(x, 0.0, top - 0.3 * i), wp.quat_identity()))
            world.add_shape_capsule(link, radius=0.04, half_height=0.12)
            joints.append(
                world.add_joint_revolute(
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
        world.add_articulation(joints)
    for x, z in ((0.0, 0.09), (3.0, 0.2)):
        body = world.add_link(xform=wp.transform(wp.vec3(x, 0.0, z), wp.quat_identity()))
        world.add_shape_box(body, hx=0.08, hy=0.08, hz=0.08)
        world.add_articulation([world.add_joint_free(parent=-1, child=body)])
    scene = newton.ModelBuilder()
    scene.add_ground_plane()
    scene.replicate(world, 2, spacing=(0.0, 6.0, 0.0))
    return scene.finalize(device=device)


def _solver(model, **options):
    return SolverSteel(
        model,
        solve_mode="matrix_free",
        iterations=12,
        enable_joint_limits=True,
        articulation_constraint_max=64,
        free_body_constraint_max=64,
        **options,
    )


def _rollout(model, solver, captured, step_pairs=4):
    """Step in pairs a->b->a, eagerly or by replaying a capture of the fresh solver's first pair; read back once."""
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_a, state_b, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_a)
    body_count = model.body_count
    trace_q = wp.empty(2 * step_pairs * body_count, dtype=wp.transform, device=model.device)
    trace_qd = wp.empty(2 * step_pairs * body_count, dtype=wp.spatial_vector, device=model.device)

    def step_pair():
        for state_in, state_out in ((state_a, state_b), (state_b, state_a)):
            state_in.clear_forces()
            pipeline.collide(state_in, contacts)
            solver.step(state_in, state_out, control, contacts, 1.0 / 240.0)

    if captured:
        with wp.ScopedCapture(model.device) as capture:
            step_pair()
    for pair in range(step_pairs):
        if captured:
            wp.capture_launch(capture.graph)
        else:
            step_pair()
        for i, state in enumerate((state_b, state_a)):
            offset = (2 * pair + i) * body_count
            wp.copy(trace_q, state.body_q, dest_offset=offset, count=body_count)
            wp.copy(trace_qd, state.body_qd, dest_offset=offset, count=body_count)
    return np.concatenate([trace_q.numpy().ravel(), trace_qd.numpy().ravel()])


def test_compiled_row_phase_matches_runtime_phase(test, device, **options):
    """A captured rollout with the row phases compiled in matches an eager runtime-phase rollout bit for bit."""
    model = _build_chains_and_boxes(device)
    solver = _solver(model, **options)
    prebuilt = set(solver._pgs_solve_mf_gs_kernels)
    compiled = _rollout(model, solver, captured=True)
    test.assertTrue(all("_phase" in kernel.key for kernel in solver._pgs_solve_mf_gs_kernels.values()))
    # The constructor prebuilds exactly the phases the captured first step launches.
    test.assertEqual(set(solver._pgs_solve_mf_gs_kernels), prebuilt)

    solver = _solver(model, **options)
    # Every launch takes the kernel that reads the phase argument.
    runtime_kernel = solver._pgs_solve_mf_gs_kernel_for_phase(row_phase=None)
    solver._pgs_solve_mf_gs_kernels = dict.fromkeys(range(6), runtime_kernel)
    runtime = _rollout(model, solver, captured=False)
    test.assertTrue(np.isfinite(runtime).all())
    np.testing.assert_array_equal(compiled.view(np.uint32), runtime.view(np.uint32))


class TestSteelMatrixFreeRowPhase(unittest.TestCase):
    pass


for _name, _options in (
    ("interleaved", {}),
    ("phased", {"row_schedule": "phased"}),
    ("contact_then_internal", {"row_schedule": "contact_then_internal"}),
    ("velocity_limits", {"enable_joint_velocity_limits": True, "velocity_limit_activation_fraction": 0.0}),
    ("propagation", {"articulated_contact_response": "propagation"}),
    ("propagation_colored", {"articulated_contact_response": "propagation_colored"}),
    (
        "propagation_velocity_limits",
        {
            "articulated_contact_response": "propagation",
            "enable_joint_velocity_limits": True,
            "velocity_limit_activation_fraction": 0.0,
        },
    ),
):
    add_function_test(
        TestSteelMatrixFreeRowPhase,
        f"test_compiled_row_phase_matches_runtime_phase_{_name}",
        test_compiled_row_phase_matches_runtime_phase,
        devices=get_selected_cuda_test_devices(),
        **_options,
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
