# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the FeatherPGS proximal contact regularizer."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    pgs_solve_mf_loop,
)
from newton.tests.unittest_utils import add_function_test, get_selected_cuda_test_devices, get_test_devices


def _solve_rows(device, n_rows, g, iterations=300):
    """Solve ``n_rows`` identical unit contact rows on one 6-DOF body with
    unit mass and rhs = -1 through the reference MF loop kernel."""
    m = 8
    J = np.zeros((1, m, 6), dtype=np.float32)
    for r in range(n_rows):
        J[0, r, 2] = 1.0
    row_type = np.zeros((1, m), dtype=np.int32)
    row_type[0, :n_rows] = PGS_CONSTRAINT_TYPE_CONTACT
    eff_inv = np.zeros((1, m), dtype=np.float32)
    eff_inv[0, :n_rows] = 1.0
    rhs = np.zeros((1, m), dtype=np.float32)
    rhs[0, :n_rows] = -1.0
    body_a = np.full((1, m), -1, dtype=np.int32)
    body_a[0, :n_rows] = 0

    mf_impulses = wp.zeros((1, m), dtype=wp.float32)
    v_out = wp.zeros(6, dtype=wp.float32)
    wp.launch(
        pgs_solve_mf_loop,
        dim=1,
        inputs=[
            wp.array([n_rows], dtype=wp.int32),
            wp.array(body_a, dtype=wp.int32),
            wp.full((1, m), -1, dtype=wp.int32),  # body_b
            wp.array(J, dtype=wp.float32),  # MiJt_a (unit mass)
            wp.zeros((1, m, 6), dtype=wp.float32),  # MiJt_b
            wp.array(J, dtype=wp.float32),  # J_a
            wp.zeros((1, m, 6), dtype=wp.float32),  # J_b
            wp.array(eff_inv, dtype=wp.float32),
            wp.array(rhs, dtype=wp.float32),
            wp.array(row_type, dtype=wp.int32),
            wp.full((1, m), -1, dtype=wp.int32),  # row_parent
            wp.zeros((1, m), dtype=wp.float32),  # row_mu
            1.0 / (1.0 + g),  # contact_regularization_w
            wp.array([0], dtype=wp.int32),  # body_to_articulation
            wp.array([0], dtype=wp.int32),  # art_dof_start
            iterations,
            1.0,  # omega
            0,  # friction_mode
            iterations,  # friction_start_iteration (no friction rows)
            0,  # iteration_offset
            mf_impulses,
            v_out,
        ],
        device=device,
    )
    return mf_impulses.numpy()[0, :n_rows]


def test_regularization_fixed_points(test: unittest.TestCase, device):
    """Analytic contracts of the proximal update on unit rows:

    - one determined row converges to ``1/(1+g)`` of the rigid impulse (the
      documented direct load error);
    - two redundant identical rows converge to the unique symmetric split
      ``lambda_1 = lambda_2 = 1/(2+g)``;
    - ``g = 0`` reproduces the exact rigid law (total impulse 1).
    """
    with wp.ScopedDevice(device):
        g = 0.5
        lam = _solve_rows(device, 1, g)
        test.assertAlmostEqual(float(lam[0]), 1.0 / (1.0 + g), places=5)

        lam = _solve_rows(device, 2, g)
        test.assertAlmostEqual(float(lam[0]), 1.0 / (2.0 + g), places=5)
        test.assertAlmostEqual(float(lam[1]), 1.0 / (2.0 + g), places=5)

        lam = _solve_rows(device, 2, 0.0)
        test.assertAlmostEqual(float(lam.sum()), 1.0, places=5)

        lam = _solve_rows(device, 1, 0.0)
        test.assertAlmostEqual(float(lam[0]), 1.0, places=6)


def _stack_scene(device, g, pgs_mode="matrix_free", response="immediate", velocity_iterations=2):
    builder = newton.ModelBuilder()
    builder.rigid_gap = 0.003
    cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.7)
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.7))
    bodies = []
    for k in range(3):
        b = builder.add_body(xform=wp.transform(wp.vec3(0.002 * k, 0.0, 0.0505 + 0.101 * k), wp.quat_identity()))
        builder.add_shape_box(b, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
        bodies.append(b)
    model = builder.finalize()
    pipeline = newton.CollisionPipeline(
        model,
        reduce_contacts=True,
        rigid_contact_max=128,
        broad_phase="nxn",
        deterministic=True,
        contact_matching="latest",
    )
    solver = newton.solvers.SolverFeatherPGS(
        model,
        pgs_mode=pgs_mode,
        pgs_iterations=8,
        pgs_velocity_iterations=velocity_iterations,
        articulated_contact_response=response,
        pgs_contact_regularization=g,
        mf_warmstart=True,
    )
    return model, pipeline, solver, bodies


def _run(model, pipeline, solver, frames):
    contacts = pipeline.contacts()
    s0, s1 = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    zs = []
    for _k in range(frames):
        pipeline.collide(s0, contacts)
        s0.clear_forces()
        solver.step(s0, s1, control, contacts, 1.0 / 60.0)
        s0, s1 = s1, s0
        zs.append(float(s0.body_q.numpy()[-1][2]))
    return s0.body_q.numpy(), zs


def test_regularization_route_agreement(test: unittest.TestCase, device):
    """The regularized update must agree across the matrix-free execution
    routes: the fused MF-GS kernel (matrix_free/immediate), the standalone
    MF kernel (split/immediate), and the propagation-fused full-iteration
    kernel — same scene, same g, matching resting poses."""
    with wp.ScopedDevice(device):
        results = {}
        for label, mode, response, velit in (
            ("fused", "matrix_free", "immediate", 2),
            # velocity iterations are matrix_free-only; split runs without
            ("standalone", "split", "immediate", 0),
            ("propagation-fused", "matrix_free", "propagation-fused", 2),
        ):
            model, pipeline, solver, _bodies = _stack_scene(
                device, g=0.05, pgs_mode=mode, response=response, velocity_iterations=velit
            )
            body_q, _ = _run(model, pipeline, solver, 120)
            results[label] = body_q
        ref = results["fused"]
        for label, body_q in results.items():
            np.testing.assert_allclose(
                body_q[:, :3], ref[:, :3], atol=2.0e-3, err_msg=f"route {label} diverged from fused"
            )


def test_regularization_velocity_pass_exempt(test: unittest.TestCase, device):
    """The velocity-only pass solves the exact rigid law. If it were
    regularized, its converged state keeps a residual velocity of roughly
    g * lambda * dt per step (~8 mm/s of steady sinking at g = 0.05), while
    the exempt pass holds a settled stack to well under a millimeter."""
    with wp.ScopedDevice(device):
        for response in ("immediate", "propagation-fused"):
            model, pipeline, solver, _bodies = _stack_scene(device, g=0.05, response=response, velocity_iterations=4)
            _, zs = _run(model, pipeline, solver, 240)
            drift = abs(zs[-1] - zs[179])
            test.assertLess(
                drift, 5.0e-4, f"{response}: top box kept sinking ({drift * 1000:.2f} mm over the last second)"
            )
            # sanity bound only: equilibrium sag at default beta with g=0.05
            # is mm-scale; a regularized velocity pass never reaches one
            sag = 0.2525 - zs[-1]
            test.assertLess(sag, 1.5e-2, f"{response}: sag {sag * 1000:.1f} mm out of range")


def test_regularization_indeterminate_split(test: unittest.TestCase, device):
    """A plank on three identical supports has no unique rigid force split;
    the regularizer must select the symmetric one (outer supports equal)."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder()
        builder.rigid_gap = 0.003
        cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.7)
        builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.7))
        supports = []
        for k in range(3):
            b = builder.add_body(xform=wp.transform(wp.vec3(0.15 * (k - 1), 0.0, 0.0255), wp.quat_identity()))
            s = builder.add_shape_box(b, hx=0.025, hy=0.025, hz=0.025, cfg=cfg)
            supports.append(s)
        plank = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.0605), wp.quat_identity()))
        builder.add_shape_box(plank, hx=0.2, hy=0.03, hz=0.01, cfg=cfg)
        model = builder.finalize()
        pipeline = newton.CollisionPipeline(
            model,
            reduce_contacts=True,
            rigid_contact_max=256,
            broad_phase="nxn",
            deterministic=True,
            contact_matching="latest",
        )
        solver = newton.solvers.SolverFeatherPGS(
            model,
            pgs_mode="matrix_free",
            pgs_iterations=8,
            pgs_velocity_iterations=2,
            pgs_contact_regularization=0.02,
            mf_warmstart=True,
        )
        contacts = pipeline.contacts()
        s0, s1 = model.state(), model.state()
        control = model.control()
        newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
        for _k in range(120):
            pipeline.collide(s0, contacts)
            s0.clear_forces()
            solver.step(s0, s1, control, contacts, 1.0 / 60.0)
            s0, s1 = s1, s0
        solver.update_contacts(contacts)

        count = int(contacts.rigid_contact_count.numpy()[0])
        shape0 = contacts.rigid_contact_shape0.numpy()[:count]
        shape1 = contacts.rigid_contact_shape1.numpy()[:count]
        force = contacts.rigid_contact_force.numpy()[:count]
        loads = []
        for s in supports:
            fz = 0.0
            for i in range(count):
                # plank-support pairs only (exclude support-ground)
                if s in (shape0[i], shape1[i]) and 0 not in (shape0[i], shape1[i]):
                    fz += abs(float(force[i][2]))
            loads.append(fz)
        test.assertGreater(min(loads), 0.0, f"a support carries no load: {loads}")
        test.assertAlmostEqual(
            loads[0],
            loads[2],
            delta=0.1 * max(loads),
            msg=f"outer supports asymmetric: {loads}",
        )


def test_regularization_validation(test: unittest.TestCase, device):
    """Parameter contract: finite non-negative values only, and pure
    propagation routings (which move contacts off the matrix-free family)
    reject a nonzero value."""
    with wp.ScopedDevice(device):
        builder = newton.ModelBuilder()
        builder.add_ground_plane()
        b = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
        builder.add_shape_box(b, hx=0.05, hy=0.05, hz=0.05)
        model = builder.finalize()
        for bad in (-0.1, float("nan"), float("inf")):
            with test.assertRaises(ValueError):
                newton.solvers.SolverFeatherPGS(model, pgs_contact_regularization=bad)
        for response in ("propagation", "propagation-colored"):
            with test.assertRaises(NotImplementedError):
                newton.solvers.SolverFeatherPGS(
                    model, pgs_contact_regularization=0.02, articulated_contact_response=response
                )
        if wp.get_device().is_cuda:  # matrix_free requires CUDA
            newton.solvers.SolverFeatherPGS(
                model,
                pgs_mode="matrix_free",
                pgs_contact_regularization=0.02,
                articulated_contact_response="propagation-fused",
            )


class TestFeatherPGSRegularization(unittest.TestCase):
    pass


add_function_test(
    TestFeatherPGSRegularization,
    "test_regularization_fixed_points",
    test_regularization_fixed_points,
    devices=get_test_devices(),
)
add_function_test(
    TestFeatherPGSRegularization,
    "test_regularization_route_agreement",
    test_regularization_route_agreement,
    devices=get_selected_cuda_test_devices(),
)
add_function_test(
    TestFeatherPGSRegularization,
    "test_regularization_velocity_pass_exempt",
    test_regularization_velocity_pass_exempt,
    devices=get_selected_cuda_test_devices(),
)
add_function_test(
    TestFeatherPGSRegularization,
    "test_regularization_indeterminate_split",
    test_regularization_indeterminate_split,
    devices=get_selected_cuda_test_devices(),
)
add_function_test(
    TestFeatherPGSRegularization,
    "test_regularization_validation",
    test_regularization_validation,
    devices=get_test_devices(),
)

if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
