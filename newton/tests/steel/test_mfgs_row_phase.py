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


def _rollout(model, runtime_phase, **options):
    solver = _solver(model, **options)
    prebuilt = set(solver._pgs_solve_mf_gs_kernels)
    if runtime_phase:
        # Every launch takes the kernel that reads the phase argument.
        runtime_kernel = solver._pgs_solve_mf_gs_kernel_for_phase(row_phase=None)
        solver._pgs_solve_mf_gs_kernels = dict.fromkeys(range(6), runtime_kernel)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state, next_state, control = model.state(), model.state(), model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    trace = []
    for _ in range(120):
        state.clear_forces()
        pipeline.collide(state, contacts)
        solver.step(state, next_state, control, contacts, 1.0 / 240.0)
        state, next_state = next_state, state
        trace.append(np.concatenate([state.body_q.numpy().ravel(), state.body_qd.numpy().ravel()]))
    return solver, prebuilt, np.asarray(trace)


def test_compiled_row_phase_matches_runtime_phase(test, device, **options):
    """Kernels with the row phase compiled in reproduce the runtime-phase kernel bit for bit."""
    model = _build_chains_and_boxes(device)
    solver, prebuilt, compiled = _rollout(model, False, **options)
    _, _, runtime = _rollout(model, True, **options)
    test.assertTrue(all("_phase" in kernel.key for kernel in solver._pgs_solve_mf_gs_kernels.values()))
    # The constructor prebuilds exactly the phases the step launches.
    test.assertEqual(set(solver._pgs_solve_mf_gs_kernels), prebuilt)
    test.assertTrue(np.isfinite(runtime).all())
    np.testing.assert_array_equal(compiled.view(np.uint32), runtime.view(np.uint32))


def test_capture_first_step_matches_eager(test, device, **options):
    """Capturing the first steps of a fresh solver replays like eager steps and builds no kernel inside the capture."""
    model = _build_chains_and_boxes(device)
    traces = []
    for captured in (True, False):
        solver = _solver(model, **options)
        prebuilt = set(solver._pgs_solve_mf_gs_kernels)
        pipeline = newton.CollisionPipeline(model)
        contacts = pipeline.contacts()
        state_a, state_b, control = model.state(), model.state(), model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, state_a)

        def two_steps(solver=solver, pipeline=pipeline, contacts=contacts, a=state_a, b=state_b, control=control):
            for state_in, state_out in ((a, b), (b, a)):
                state_in.clear_forces()
                pipeline.collide(state_in, contacts)
                solver.step(state_in, state_out, control, contacts, 1.0 / 240.0)

        if captured:
            with wp.ScopedCapture() as capture:
                two_steps()
            test.assertEqual(set(solver._pgs_solve_mf_gs_kernels), prebuilt)
        trace = []
        for _ in range(30):
            if captured:
                wp.capture_launch(capture.graph)
            else:
                two_steps()
            trace.append(np.concatenate([state_a.body_q.numpy().ravel(), state_a.body_qd.numpy().ravel()]))
        traces.append(np.asarray(trace))
    test.assertTrue(np.isfinite(traces[1]).all())
    np.testing.assert_array_equal(traces[0].view(np.uint32), traces[1].view(np.uint32))


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

for _name, _options in (
    ("interleaved", {}),
    ("propagation", {"articulated_contact_response": "propagation"}),
    ("propagation_colored", {"articulated_contact_response": "propagation_colored"}),
):
    add_function_test(
        TestSteelMatrixFreeRowPhase,
        f"test_capture_first_step_matches_eager_{_name}",
        test_capture_first_step_matches_eager,
        devices=get_selected_cuda_test_devices(),
        **_options,
    )


if __name__ == "__main__":
    unittest.main(verbosity=2)
