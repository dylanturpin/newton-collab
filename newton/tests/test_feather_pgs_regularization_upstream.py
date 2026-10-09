# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for the FeatherPGS contact regularizer (``pgs_contact_regularization``)."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.test_feather_pgs_propagation_same_articulation import _build_scissor_model
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices

DT = 1.0 / 60.0
PATH_DENSE = 0  # contact routing id of the dense rows


def _contact_rows(solver, path_id: int, contact_count: int) -> dict[int, int]:
    """Map each contact of world 0 routed to ``path_id`` to its first row."""
    contact_path = solver.contact_path.numpy()
    contact_slot = solver.contact_slot.numpy()
    contact_world = solver.contact_world.numpy()
    return {
        c: int(contact_slot[c])
        for c in range(contact_count)
        if int(contact_world[c]) == 0 and int(contact_path[c]) == path_id and int(contact_slot[c]) >= 0
    }


def _run(model, pipeline, solver, frames, dt=DT):
    contacts = pipeline.contacts()
    s0, s1 = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    zs = []
    for _k in range(frames):
        pipeline.collide(s0, contacts)
        s0.clear_forces()
        solver.step(s0, s1, control, contacts, dt)
        s0, s1 = s1, s0
        zs.append(float(s0.body_q.numpy()[-1][2]))
    return s0.body_q.numpy(), zs


def _resting_box(device, g, rate, frames, articulated=False, **solver_kwargs):
    """Rest a box on the ground; ``articulated`` mounts it on a vertical prismatic joint (dense rows)."""
    builder = newton.ModelBuilder()
    builder.rigid_gap = 0.01
    cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.7)
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.7))
    xform = wp.transform(wp.vec3(0.0, 0.0, 0.05), wp.quat_identity())
    if articulated:
        b = builder.add_link(xform=xform)
        joint = builder.add_joint_prismatic(-1, b, axis=wp.vec3(0.0, 0.0, 1.0), parent_xform=xform)
        builder.add_articulation([joint])
    else:
        b = builder.add_body(xform=xform)
    builder.add_shape_box(b, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(
        model,
        reduce_contacts=True,
        rigid_contact_max=32,
        broad_phase="nxn",
        deterministic=True,
        contact_matching="latest",
    )
    solver = newton.solvers.SolverFeatherPGS(
        model, pgs_mode="matrix_free", pgs_iterations=12, pgs_contact_regularization=g, **solver_kwargs
    )
    _, zs = _run(model, pipeline, solver, frames, dt=1.0 / rate)
    return 0.05 - zs[-1]


def _sag_formula(g, rate):
    """Resting sag of a body under gravity: ``g * a * dt^2 / pgs_beta`` with the default beta."""
    return g * 9.81 / (rate * rate) / 0.2


def _scissor_step(device, g, velocity_iterations):
    """Step the same-articulation scissor scene once; return the solver, contact count and joint velocity."""
    model = _build_scissor_model(device)
    solver = newton.solvers.SolverFeatherPGS(
        model,
        pgs_mode="matrix_free",
        pgs_iterations=8,
        pgs_velocity_iterations=velocity_iterations,
        pgs_contact_regularization=g,
        dense_max_constraints=64,
        mf_max_constraints=16,
    )
    state_in, state_out = model.state(), model.state()
    newton.eval_fk(model, state_in.joint_q, state_in.joint_qd, state_in)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_in.clear_forces()
    pipeline.collide(state_in, contacts)
    solver.step(state_in, state_out, model.control(), contacts, 1.0 / 200.0)
    return solver, int(contacts.rigid_contact_count.numpy()[0]), state_out.joint_qd.numpy().copy()


def test_regularization_documented_sag_on_dense_rows(test: unittest.TestCase, device):
    """Apply the same sag law to dense (articulated) contact rows.

    At rest each row satisfies ``beta * phi / dt = -g * d * lambda``. A box on a vertical
    prismatic joint has ``d = 1 / m`` per contact and its four contacts share the weight,
    ``lambda = m * a * dt / 4``, so it sags by ``g * a * dt^2 / (4 * beta)``.
    """
    for rate in (60, 240):
        sag = _resting_box(device, 0.5, rate, 3 * rate, articulated=True, pgs_warmstart=True)
        expected = 0.25 * _sag_formula(0.5, rate)
        test.assertAlmostEqual(sag, expected, delta=0.15 * expected, msg=f"{rate} Hz: sag {sag * 1000:.2f} mm")
        rigid = _resting_box(device, 0.0, rate, 3 * rate, articulated=True, pgs_warmstart=True)
        test.assertLess(abs(rigid), 0.1 * expected, msg=f"{rate} Hz: rigid sag {rigid * 1000:.3f} mm")


def test_dense_contact_rows_carry_the_regularization_weight(test: unittest.TestCase, device):
    """Give penetrating articulated (dense) contact rows the weight ``1 / (1 + g)``."""
    solver, count, _ = _scissor_step(device, 0.5, 0)
    rows = _contact_rows(solver, PATH_DENSE, count)
    test.assertGreater(len(rows), 0, "scene produced no dense self-contact row")
    phi = solver.phi.numpy()[0]
    weights = solver.row_w.numpy()[0]
    for slot in rows.values():
        test.assertLess(float(phi[slot]), 0.0)
        test.assertAlmostEqual(float(weights[slot]), 2.0 / 3.0, places=6)


def test_velocity_pass_is_rigid_on_dense_rows(test: unittest.TestCase, device):
    """Ignore the regularizer on dense rows in the velocity-only pass.

    After the pass the joint velocity is the same as with ``g = 0``, since the rigid law does not depend on ``g``.
    """
    solver, count, qd_soft = _scissor_step(device, 0.5, 8)
    test.assertGreater(len(_contact_rows(solver, PATH_DENSE, count)), 0)
    _, _, qd_rigid = _scissor_step(device, 0.0, 8)
    _, _, qd_position_soft = _scissor_step(device, 0.5, 0)
    np.testing.assert_allclose(qd_soft, qd_rigid, rtol=0.0, atol=1.0e-4, err_msg="velocity pass depends on g")
    # Positive control: the regularized position solve alone does depend on g.
    test.assertGreater(float(np.max(np.abs(qd_position_soft - qd_rigid))), 1.0e-3)


class TestFeatherPGSRegularization(unittest.TestCase):
    pass


devices = get_cuda_test_devices()
for _fn in (
    test_regularization_documented_sag_on_dense_rows,
    test_dense_contact_rows_carry_the_regularization_weight,
    test_velocity_pass_is_rigid_on_dense_rows,
):
    add_function_test(TestFeatherPGSRegularization, _fn.__name__, _fn, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
