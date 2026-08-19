# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the matrix-free velocity pass and speculative contacts.

The final velocity pass rebuilds the contact RHS with the speculative term
scaled to zero. For a row that participated in the position solve that is
correct -- the geometric bias belongs there. For a row that merely exists
inside the collision margin and took no impulse, it rewrites the constraint
from "you may close by the remaining gap" (``u + phi/h >= 0``) into "you may
not approach at all" (``u >= 0``), stopping a falling body at the edge of the
margin.

No restitution is involved anywhere here: this is ordinary free fall with
``pgs_velocity_iterations`` enabled.
"""

import unittest
from unittest import mock

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_selected_cuda_test_devices

SIM_DT = 1.0 / 2000.0
RADIUS = 0.05
DROP = 0.12

RESPONSE_MODES = ("immediate", "propagation", "propagation-fused", "propagation-colored")

# Values written into ``SolverFeatherPGS.contact_path``.
PATH_DENSE = 0
PATH_MATRIX_FREE = 1
PATH_PROPAGATION = 2

# The route each scene and response mode takes, measured rather than assumed.
# A lone free body never reaches dense rows, and "propagation-fused" only uses
# propagation rows once a real articulation exists -- with a free body it
# degrades to the matrix-free path -- so a single-sphere scene cannot claim to
# cover either. Asserting the exact set keeps the suite from silently claiming
# coverage of a path it never reaches.
EXPECTED_PATHS = {
    ("free", "immediate"): {PATH_MATRIX_FREE},
    ("free", "propagation"): {PATH_PROPAGATION},
    ("free", "propagation-fused"): {PATH_MATRIX_FREE},
    ("free", "propagation-colored"): {PATH_PROPAGATION},
    ("articulated", "immediate"): {PATH_DENSE},
    ("articulated", "propagation"): {PATH_PROPAGATION},
    ("articulated", "propagation-fused"): {PATH_PROPAGATION},
    ("articulated", "propagation-colored"): {PATH_PROPAGATION},
}


def _build(device, scene, mass=1.0):
    """Free sphere, or a two-link articulation whose root carries the contact."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.5)
    if scene == "free":
        body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, RADIUS + DROP), wp.quat_identity()))
        builder.add_shape_sphere(body, radius=RADIUS, cfg=cfg)
    else:
        # add_link + explicit joints + add_articulation: add_body would create a
        # standalone free articulation per call, so appending a revolute to two
        # of them yields two articulations and an unowned loop joint rather than
        # the single two-link chain this fixture is meant to exercise.
        body = builder.add_link(xform=wp.transform(wp.vec3(0.0, 0.0, RADIUS + DROP), wp.quat_identity()), mass=mass)
        builder.add_shape_sphere(body, radius=RADIUS, cfg=cfg)
        link = builder.add_link(xform=wp.transform(wp.vec3(0.12, 0.0, RADIUS + DROP), wp.quat_identity()), mass=mass)
        builder.add_shape_sphere(link, radius=RADIUS, cfg=cfg)
        root_joint = builder.add_joint_free(child=body)
        hinge = builder.add_joint_revolute(
            parent=body,
            child=link,
            axis=(0.0, 1.0, 0.0),
            parent_xform=wp.transform(wp.vec3(0.12, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        )
        builder.add_articulation([root_joint, hinge])
    builder.add_ground_plane(cfg=cfg)
    return builder.finalize(device=device), body


def _drop(device, velocity_iterations, response="immediate", steps=620, scene="free"):
    """Drop a body onto a plane; return heights, velocities, contact counts, routes."""
    model, body = _build(device, scene)
    solver = newton.solvers.SolverFeatherPGS(
        model,
        angular_damping=0.0,
        pgs_mode="matrix_free",
        pgs_velocity_iterations=velocity_iterations,
        articulated_contact_response=response,
    )
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    heights, velocities, counts, routes = [], [], [], set()
    for _ in range(steps):
        contacts = model.collide(state_0)
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts, SIM_DT)
        state_0, state_1 = state_1, state_0
        heights.append(float(state_0.body_q.numpy()[body][2]) - RADIUS)
        velocities.append(float(state_0.body_qd.numpy()[body][2]))
        n = int(contacts.rigid_contact_count.numpy()[0])
        counts.append(n)
        if n:
            routes.update(int(v) for v in solver.contact_path.numpy()[:n] if v >= 0)
    return np.asarray(heights), np.asarray(velocities), np.asarray(counts), routes


def _assert_landed(test, heights, velocities, counts, label):
    """Assert the body came to rest ON the plane, not through it and not above it."""
    final_h, final_v = float(heights[-1]), float(velocities[-1])
    # Bounded on BOTH sides: a tunnelling body has negative heights and would
    # satisfy any upper bound by itself.
    test.assertLess(abs(final_h), 0.005, f"{label}: rested at {final_h:+.4f} m instead of on the surface")
    test.assertLess(abs(final_v), 0.02, f"{label}: still moving at {final_v:+.4f} m/s")
    test.assertGreater(float(heights[:80].min()), 0.05, f"{label}: body was not still falling early in the run")
    test.assertGreater(int(counts.max()), 0, f"{label}: no contact was ever generated")
    settled = heights[-150:]
    test.assertLess(float(settled.max() - settled.min()), 0.002, f"{label}: resting height was not stable")


def _single_step(device, gap, approach_speed, velocity_iterations=4, mass=1.0):
    """One step of a body ``gap`` above the plane closing at ``approach_speed``.

    Returns the velocity before and after, the contact count, the position-pass
    impulse on positive-gap rows, and the largest such gap, so a test can
    confirm it exercised a genuine speculative contact rather than passing
    because nothing was generated.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))  # no gravity: isolate the constraint
    # density=0 so the body's mass is exactly the requested value: the point of
    # the light-body case is that the position impulse scales with it.
    cfg = newton.ModelBuilder.ShapeConfig(mu=0.0, density=0.0)
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, RADIUS + gap), wp.quat_identity()), mass=mass)
    builder.add_shape_sphere(body, radius=RADIUS, cfg=cfg)
    builder.add_ground_plane(cfg=cfg)
    model = builder.finalize(device=device)
    solver = newton.solvers.SolverFeatherPGS(
        model, angular_damping=0.0, pgs_mode="matrix_free", pgs_velocity_iterations=velocity_iterations
    )
    state_0, state_1 = model.state(), model.state()
    # A free body is driven through joint_qd; writing body_qd alone is discarded
    # by the forward kinematics the solver runs from joint state.
    joint_qd = state_0.joint_qd.numpy()
    joint_qd[2] = -abs(approach_speed)
    state_0.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    before = float(state_0.body_qd.numpy()[body][2])
    contacts = model.collide(state_0)
    state_0.clear_forces()
    solver.step(state_0, state_1, model.control(), contacts, SIM_DT)
    n = int(contacts.rigid_contact_count.numpy()[0])
    # The classification reads the impulse the POSITION solve produced;
    # solver.mf_impulses has since been overwritten by the velocity pass.
    rows = int(solver.mf_constraint_count.numpy()[0])
    phi = solver._debug_position_mf_phi.numpy()[0][:rows]
    lam = solver._debug_position_mf_impulses.numpy()[0][:rows]
    contact_rows = np.flatnonzero(phi > 0.0)
    position_impulse = float(lam[contact_rows].max()) if contact_rows.size else 0.0
    max_phi = float(phi[contact_rows].max()) if contact_rows.size else 0.0
    return before, float(state_1.body_qd.numpy()[body][2]), n, position_impulse, max_phi


def test_velocity_iterations_do_not_halt_a_falling_body(test, device):
    """Land a falling body on the plane with the velocity pass enabled."""
    for iterations in (2, 8):
        h, v, n, _r = _drop(device, iterations)
        _assert_landed(test, h, v, n, f"pgs_velocity_iterations={iterations}")


def test_velocity_iterations_match_the_position_only_landing(test, device):
    """Land at the same height with and without velocity iterations."""
    baseline, base_v, base_n, _r0 = _drop(device, 0)
    with_velocity, vel_v, vel_n, _r4 = _drop(device, 4)
    _assert_landed(test, baseline, base_v, base_n, "velocity_iterations=0")
    _assert_landed(test, with_velocity, vel_v, vel_n, "velocity_iterations=4")
    test.assertAlmostEqual(
        float(with_velocity[-1]),
        float(baseline[-1]),
        delta=0.002,
        msg=(
            f"resting height moved from {float(baseline[-1]):.4f} m to "
            f"{float(with_velocity[-1]):.4f} m when velocity iterations were enabled"
        ),
    )


def test_every_contact_response_route_lands(test, device):
    """Land correctly on every internal contact-response route.

    Asserts the exact ``contact_path`` each combination took, so the suite
    cannot claim coverage of a route the scene never reaches.
    """
    for scene in ("free", "articulated"):
        for response in RESPONSE_MODES:
            label = f"{scene}/{response}"
            h, v, n, routes = _drop(device, 4, response=response, scene=scene)
            _assert_landed(test, h, v, n, label)
            test.assertEqual(
                routes,
                EXPECTED_PATHS[(scene, response)],
                f"{label}: exercised contact_path {sorted(routes)}, "
                f"expected {sorted(EXPECTED_PATHS[(scene, response)])}",
            )


def test_non_crossing_row_keeps_its_speculative_allowance(test, device):
    """Leave the velocity untouched when the body cannot reach the surface.

    With ``phi + h*u > 0`` the row did not participate in the position solve. If
    the velocity pass dropped its ``phi/h`` allowance it would forbid any
    approach and brake a body still far from contact.
    """
    gap = 0.02
    before, after, n, impulse, phi = _single_step(device, gap, 0.5 * gap / SIM_DT)
    test.assertLess(before, -1.0, "setup failed to give the body an approach velocity")
    test.assertGreater(n, 0, "no speculative contact was generated, so the branch was never reached")
    test.assertGreater(phi, 0.0, "the row under test was not a positive-gap speculative contact")
    test.assertEqual(impulse, 0.0, f"a non-crossing row took a position impulse of {impulse:.3e}")
    test.assertAlmostEqual(
        after, before, delta=0.02 * abs(before), msg=f"free approach was braked: {before:.3f} -> {after:.3f} m/s"
    )


def test_crossing_row_loses_its_speculative_allowance(test, device):
    """Arrest a body that would cross the surface within the step.

    A blanket "always retain phi/h" implementation leaves it closing at the gap
    rate instead, which this bound catches.
    """
    gap = 0.002
    before, after, n, impulse, phi = _single_step(device, gap, 5.0 * gap / SIM_DT)
    test.assertLess(before, -1.0, "setup failed to give the body an approach velocity")
    test.assertGreater(n, 0, "no contact was generated")
    test.assertGreater(phi, 0.0, "the row under test was not a positive-gap speculative contact")
    test.assertGreater(impulse, 0.0, "a crossing row took no position impulse, so it was never classified loaded")
    # Magnitude, not a signed bound: "after <= 0.05*|before|" is satisfied by any
    # negative value, so a row that merely slowed from -20 to -4 m/s would pass.
    test.assertLess(
        abs(after),
        0.05 * abs(before),
        f"crossing contact kept approaching: {before:.3f} -> {after:.3f} m/s",
    )


def test_light_body_crossing_is_still_classified_loaded(test, device):
    """Arrest a very light crossing body, whose position impulse is tiny.

    The impulse is effective mass times the velocity change, so it can be made
    arbitrarily small without changing the physics of the impact. At this mass it
    is ~2e-11, below the 1e-9 absolute threshold this implementation once used;
    that cutoff would call the row inactive, hand back its speculative allowance
    and let the body keep closing at the gap rate. The mass is deliberately
    unphysical: the point is the classification boundary, not the scenario.
    """
    gap = 0.002
    before, after, n, impulse, phi = _single_step(device, gap, 5.0 * gap / SIM_DT, mass=1.0e-12)
    test.assertGreater(n, 0, "no contact was generated")
    test.assertGreater(phi, 0.0, "the row under test was not a positive-gap speculative contact")
    # Straddles the 1e-9 cutoff this implementation once used: positive, so the
    # row genuinely loaded, yet small enough that an absolute threshold misses it.
    test.assertGreater(impulse, 0.0, "light crossing row took no position impulse")
    test.assertLess(impulse, 1.0e-9, f"position impulse {impulse:.3e} does not straddle the old 1e-9 cutoff")
    test.assertLess(
        abs(after),
        0.05 * abs(before),
        f"light crossing body kept approaching: {before:.3f} -> {after:.3f} m/s (position impulse {impulse:.3e})",
    )


def test_warm_start_with_velocity_iterations_is_rejected(test, device):
    """Reject every warm-start entry point while velocity iterations are on.

    Warm start seeds the impulse array the velocity pass reads to decide which
    rows participated, so residue would revive the stopping behaviour.
    """
    model, _body = _build(device, "free")

    def make(**kwargs):
        return newton.solvers.SolverFeatherPGS(model, pgs_mode="matrix_free", pgs_velocity_iterations=4, **kwargs)

    with test.assertRaises(NotImplementedError):
        make(pgs_warmstart=True)
    with test.assertRaises(NotImplementedError):
        make(mf_warmstart=True)
    with mock.patch.dict("os.environ", {"IL_NEWTON_FPGS_MF_WARMSTART": "1"}):
        with test.assertRaises(NotImplementedError):
            make()
    # Warm start remains available when no velocity pass runs.
    newton.solvers.SolverFeatherPGS(model, pgs_mode="matrix_free", pgs_velocity_iterations=0, pgs_warmstart=True)


devices = get_selected_cuda_test_devices()


class TestFeatherPGSVelocityPassSpeculative(unittest.TestCase):
    pass


for _fn in (
    test_velocity_iterations_do_not_halt_a_falling_body,
    test_velocity_iterations_match_the_position_only_landing,
    test_every_contact_response_route_lands,
    test_non_crossing_row_keeps_its_speculative_allowance,
    test_crossing_row_loses_its_speculative_allowance,
    test_light_body_crossing_is_still_classified_loaded,
    test_warm_start_with_velocity_iterations_is_rejected,
):
    add_function_test(TestFeatherPGSVelocityPassSpeculative, _fn.__name__, _fn, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
