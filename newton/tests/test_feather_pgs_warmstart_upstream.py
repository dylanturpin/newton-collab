# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for identity-matched FeatherPGS contact warm start."""

import unittest

import numpy as np
import warp as wp

import newton
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
)
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices


def test_substeps_reuse_contacts_with_their_own_history(test, device):
    """Seed each contact from its own last solve across solver substeps on one contact set.

    Match indices refer to the contact set before the last collision pass, while the
    history is saved every solver step. Inserting and deleting a contact moves the
    persistent contact's index, so reading the match index on a substep would seed it
    from another contact or from nothing.
    """
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    body_a = builder.add_body(xform=wp.transform(wp.vec3(-0.5, 0.0, 1.0), wp.quat_identity()))
    shape_a = builder.add_shape_sphere(body_a, radius=0.1)
    body_b = builder.add_body(xform=wp.transform(wp.vec3(0.5, 0.0, 0.1), wp.quat_identity()))
    shape_b = builder.add_shape_sphere(body_b, radius=0.1)
    builder.add_ground_plane()
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, broad_phase="nxn", contact_matching="sticky")
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverFeatherPGS(model, pgs_mode="matrix_free", pgs_iterations=8, pgs_warmstart=True)
    states = [model.state(), model.state()]
    control = model.control()
    dt = 1.0 / 240.0

    def step(iterations):
        solver.pgs_iterations = iterations
        solver.step(states[0], states[1], control, contacts, dt)
        states.reverse()

    def contact_impulses():
        count = int(contacts.rigid_contact_count.numpy()[0])
        shape0 = contacts.rigid_contact_shape0.numpy()[:count]
        shape1 = contacts.rigid_contact_shape1.numpy()[:count]
        slots = solver.contact_slot.numpy()[:count]
        impulses = solver.mf_impulses.numpy()[0]
        result = {}
        for name, shape in (("a", shape_a), ("b", shape_b)):
            index = np.flatnonzero((shape0 == shape) | (shape1 == shape))
            if len(index):
                result[name] = float(impulses[int(slots[int(index[0])])])
        return result

    def check_substeps(phase):
        # Solve a substep, then seed the next one without a sweep: its impulses are the seeds.
        for substep in range(2):
            step(8)
            solved = contact_impulses()
            step(0)
            seeded = contact_impulses()
            with test.subTest(phase=phase, substep=substep):
                test.assertEqual(seeded.keys(), solved.keys())
                test.assertGreater(solved["b"], 0.0)
                for name, value in solved.items():
                    test.assertAlmostEqual(seeded[name], value, delta=1.0e-6 * max(1.0, abs(value)))

    pipeline.collide(states[0], contacts)
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 1)
    check_substeps("B alone")

    # Insert the lower shape-id sphere: sorting puts its contact before B's.
    q = states[0].body_q.numpy()
    q[body_a][2] = 0.1
    states[0].body_q.assign(q)
    qd = states[0].body_qd.numpy()
    qd[body_a] = 0.0
    states[0].body_qd.assign(qd)
    pipeline.collide(states[0], contacts)
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 2)
    test.assertGreaterEqual(int(contacts.rigid_contact_match_index.numpy()[1]), 0)
    check_substeps("insertion")

    # Delete A's contact again: B moves back to index 0.
    q = states[0].body_q.numpy()
    q[body_a][2] = 1.0
    states[0].body_q.assign(q)
    pipeline.collide(states[0], contacts)
    test.assertEqual(int(contacts.rigid_contact_count.numpy()[0]), 1)
    test.assertEqual(int(contacts.rigid_contact_match_index.numpy()[0]), 1)
    check_substeps("deletion")


def _resting_box_rows(device, articulated: bool):
    """Settle a box on the ground with warm start; return the model, pipeline, contacts, solver and states.

    ``articulated`` mounts the box on a vertical prismatic joint, so its contacts are dense
    rows; otherwise it is a free body on the free-body rows.
    """
    builder = newton.ModelBuilder()
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.7))
    cfg = newton.ModelBuilder.ShapeConfig(density=1000.0, mu=0.7)
    xform = wp.transform(wp.vec3(0.0, 0.0, 0.05), wp.quat_identity())
    if articulated:
        body = builder.add_link(xform=xform)
        joint = builder.add_joint_prismatic(-1, body, axis=wp.vec3(0.0, 0.0, 1.0), parent_xform=xform)
        builder.add_articulation([joint])
    else:
        body = builder.add_body(xform=xform)
    builder.add_shape_box(body, hx=0.05, hy=0.05, hz=0.05, cfg=cfg)
    model = builder.finalize(device=device)
    pipeline = newton.CollisionPipeline(model, contact_matching="latest", deterministic=True)
    contacts = pipeline.contacts()
    solver = newton.solvers.SolverFeatherPGS(model, pgs_mode="matrix_free", pgs_warmstart=True, pgs_iterations=12)
    state_0, state_1 = model.state(), model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    control = model.control()
    for _ in range(60):
        pipeline.collide(state_0, contacts)
        solver.step(state_0, state_1, control, contacts, 1.0 / 120.0)
        state_0, state_1 = state_1, state_0
    return model, pipeline, contacts, solver, state_0, state_1, control


def test_both_row_families_seed_impulses_scaled_by_the_step_ratio(test, device):
    """Seed the carried normal impulses of dense and free-body rows, scaled by ``dt / dt_previous``.

    With no sweep (``pgs_iterations = 0``) the solved impulses are exactly the seeds.
    """
    for articulated in (True, False):
        with test.subTest(articulated=articulated):
            _model, pipeline, contacts, solver, state_0, state_1, control = _resting_box_rows(device, articulated)
            if articulated:
                impulses, row_type, count = solver.impulses, solver.row_type, solver.constraint_count
            else:
                impulses, row_type, count = solver.mf_impulses, solver.mf_row_type, solver.mf_constraint_count
            n = int(count.numpy()[0])
            normal = row_type.numpy()[0, :n] == PGS_CONSTRAINT_TYPE_CONTACT
            previous = impulses.numpy()[0, :n][normal]
            test.assertGreater(float(previous.sum()), 0.0)

            solver.pgs_iterations = 0
            pipeline.collide(state_0, contacts)
            solver.step(state_0, state_1, control, contacts, 0.5 / 120.0)
            test.assertEqual(int(count.numpy()[0]), n)
            seeded = impulses.numpy()[0, :n][row_type.numpy()[0, :n] == PGS_CONSTRAINT_TYPE_CONTACT]
            np.testing.assert_allclose(seeded, 0.5 * previous, rtol=1.0e-6, atol=1.0e-9)


class TestFeatherPGSIdentityWarmstart(unittest.TestCase):
    pass


devices = get_cuda_test_devices()
for _fn in (
    test_both_row_families_seed_impulses_scaled_by_the_step_ratio,
    test_substeps_reuse_contacts_with_their_own_history,
):
    add_function_test(TestFeatherPGSIdentityWarmstart, _fn.__name__, _fn, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
