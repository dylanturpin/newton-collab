# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the matrix-free velocity pass and speculative contacts.

The final velocity pass rebuilds the contact RHS with the speculative term
scaled to zero. For a row that is genuinely in contact that is correct -- the
geometric bias belongs to the position solve. For a row that merely exists
inside the collision margin and carried no impulse, it rewrites the constraint
from "you may close by the remaining gap" (``u + phi/h >= 0``) into "you may not
approach at all" (``u >= 0``), which stops a falling body dead at the edge of
the margin.

These tests use no restitution: the behaviour under test is ordinary free fall
with ``pgs_velocity_iterations`` enabled.
"""

import unittest

import numpy as np
import warp as wp

import newton
from newton.tests.unittest_utils import add_function_test, get_selected_cuda_test_devices

SIM_DT = 1.0 / 2000.0
RADIUS = 0.05
DROP = 0.30


RESPONSE_MODES = ("immediate", "propagation", "propagation-fused", "propagation-colored")

# Values written into ``SolverFeatherPGS.contact_path``.
PATH_DENSE = 0
PATH_MATRIX_FREE = 1
PATH_PROPAGATION = 2

# Routes each response mode must exercise, measured rather than assumed. The
# characteristic path is REQUIRED; ALLOWED bounds what else may appear, since
# which bodies are touching on a given step varies run to run (an articulated
# scene under "immediate" may or may not have produced a dense row yet).
# "propagation-fused" needs an articulation before it uses propagation rows at
# all -- with a lone free body it degrades to the matrix-free path, so a
# single-sphere scene cannot claim to cover it.
REQUIRED_PATHS = {
    ("free", "immediate"): {PATH_MATRIX_FREE},
    ("free", "propagation"): {PATH_PROPAGATION},
    ("free", "propagation-fused"): {PATH_MATRIX_FREE},
    ("free", "propagation-colored"): {PATH_PROPAGATION},
    ("articulated", "immediate"): {PATH_MATRIX_FREE},
    ("articulated", "propagation"): {PATH_PROPAGATION},
    ("articulated", "propagation-fused"): {PATH_PROPAGATION},
    ("articulated", "propagation-colored"): {PATH_PROPAGATION},
}
ALLOWED_PATHS = {PATH_DENSE, PATH_MATRIX_FREE, PATH_PROPAGATION}


def _drop(device, velocity_iterations, response="immediate", steps=1400, scene="free"):
    """Drop a body onto a plane; return heights, velocities, contact counts, routes used.

    ``scene="articulated"`` adds a revolute-jointed second link, which is what
    makes ``propagation-fused`` use propagation rows rather than degrading to
    the matrix-free path.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, RADIUS + DROP), wp.quat_identity()))
    builder.add_shape_sphere(body, radius=RADIUS, cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
    if scene == "articulated":
        link = builder.add_body(xform=wp.transform(wp.vec3(0.12, 0.0, RADIUS + DROP), wp.quat_identity()))
        builder.add_shape_sphere(link, radius=RADIUS, cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
        builder.add_joint_revolute(
            parent=body,
            child=link,
            axis=(0.0, 1.0, 0.0),
            parent_xform=wp.transform(wp.vec3(0.12, 0.0, 0.0), wp.quat_identity()),
            child_xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        )
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
    model = builder.finalize(device=device)
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
    heights, velocities, contact_counts = [], [], []
    routes = set()
    for _ in range(steps):
        contacts = model.collide(state_0)
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts, SIM_DT)
        state_0, state_1 = state_1, state_0
        heights.append(float(state_0.body_q.numpy()[body][2]) - RADIUS)
        velocities.append(float(state_0.body_qd.numpy()[body][2]))
        n = int(contacts.rigid_contact_count.numpy()[0])
        contact_counts.append(n)
        if n:
            routes.update(int(v) for v in solver.contact_path.numpy()[:n] if v >= 0)
    return np.asarray(heights), np.asarray(velocities), np.asarray(contact_counts), routes


def _assert_landed(test, heights, velocities, contact_counts, label):
    """Assert the sphere came to rest ON the plane, not through it and not above it."""
    final_h, final_v = float(heights[-1]), float(velocities[-1])
    # Bounded on BOTH sides: a tunnelling body has negative heights and would
    # satisfy any upper bound on its own.
    test.assertLess(abs(final_h), 0.005, f"{label}: rested at {final_h:+.4f} m instead of on the surface")
    test.assertLess(abs(final_v), 0.02, f"{label}: still moving at {final_v:+.4f} m/s")
    test.assertGreater(float(heights[:200].min()), 0.05, f"{label}: sphere was not still falling early in the run")
    test.assertGreater(int(contact_counts.max()), 0, f"{label}: no contact was ever generated")
    settled = heights[-400:]
    test.assertLess(float(settled.max() - settled.min()), 0.002, f"{label}: resting height was not stable")


def test_velocity_iterations_do_not_halt_a_falling_body(test, device):
    """Land a falling sphere on the plane with the velocity pass enabled.

    Without the fix the sphere stops inside the collision margin and hovers
    there for the rest of the run instead of reaching the surface.
    """
    for iterations in (2, 4, 8):
        h, v, n, _r = _drop(device, iterations)
        _assert_landed(test, h, v, n, f"pgs_velocity_iterations={iterations}")


def test_velocity_iterations_match_the_position_only_landing(test, device):
    """Land at the same height with and without velocity iterations.

    The velocity pass refines velocities; it must not change where a simple
    unconstrained drop comes to rest.
    """
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
    """Land correctly on all four internal contact-response routes.

    ``propagation`` and ``propagation-colored`` route even free/ground contacts
    through propagation rows, so a fix applied only to the matrix-free route
    leaves them hovering.
    """
    for scene in ("free", "articulated"):
        for response in RESPONSE_MODES:
            label = f"{scene}/{response}"
            h, v, n, routes = _drop(device, 4, response=response, scene=scene)
            _assert_landed(test, h, v, n, label)
            # Assert the route actually taken, so the test cannot silently claim
            # coverage of a path the scene never reaches.
            required = REQUIRED_PATHS[(scene, response)]
            test.assertTrue(
                required <= routes,
                f"{label}: exercised contact_path {sorted(routes)}, missing required {sorted(required - routes)}",
            )
            test.assertTrue(
                routes <= ALLOWED_PATHS,
                f"{label}: exercised unexpected contact_path {sorted(routes - ALLOWED_PATHS)}",
            )


def test_separated_body_still_falls_through_the_margin(test, device):
    """Keep a body accelerating while it is inside the margin but not touching.

    The speculative row exists well before the surface is reached; it must not
    apply an impulse until the body would actually cross it.
    """
    heights, _v, _n, _r = _drop(device, 4, steps=500)
    descending = np.diff(heights[:450])
    test.assertLess(
        float(descending.max()),
        1e-6,
        "sphere moved upward during free fall inside the collision margin",
    )
    test.assertGreater(
        float(heights[0] - heights[449]),
        0.20,
        "sphere did not fall freely through the collision margin",
    )


def _single_step_velocity(device, gap, approach_speed, velocity_iterations=4):
    """Place a sphere ``gap`` above the plane moving down at ``approach_speed``; step once.

    Returns (velocity before the step, velocity after one step). One step keeps
    the branch under test isolated: nothing else has had a chance to act.
    """
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))  # no gravity: isolate the constraint
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, RADIUS + gap), wp.quat_identity()))
    builder.add_shape_sphere(body, radius=RADIUS, cfg=newton.ModelBuilder.ShapeConfig(mu=0.0))
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(mu=0.0))
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
    return before, float(state_1.body_qd.numpy()[body][2])


def test_non_crossing_row_keeps_its_speculative_allowance(test, device):
    """Leave the velocity untouched when the body cannot reach the surface this step.

    With ``phi + h*u > 0`` the row is positively slack. If the velocity pass
    dropped its ``phi/h`` allowance it would forbid any approach and brake a
    body that is still far from contact.
    """
    gap = 0.02
    speed = 0.5 * gap / SIM_DT  # crosses only half the gap this step
    before, after = _single_step_velocity(device, gap, speed)
    test.assertLess(before, -1.0, "test setup failed to give the body an approach velocity")
    test.assertAlmostEqual(
        after, before, delta=0.02 * abs(before), msg=f"free approach was braked: {before:.3f} -> {after:.3f} m/s"
    )


def test_crossing_row_loses_its_speculative_allowance(test, device):
    """Stop the body at the surface when it would cross within this step.

    With ``phi + h*u <= 0`` the row loads, so the allowance must be removed and
    the outgoing velocity must not still be approaching at the full rate. This
    is the case a blanket "always retain phi/h" implementation would fail.
    """
    gap = 0.002
    speed = 5.0 * gap / SIM_DT  # overshoots the gap fivefold
    before, after = _single_step_velocity(device, gap, speed)
    test.assertLess(before, -1.0, "test setup failed to give the body an approach velocity")
    # Magnitude, not a signed bound: "after <= 0.05*|before|" is satisfied by any
    # negative value, so a row that merely slowed from -20 to -4 m/s would pass.
    test.assertLess(
        abs(after),
        0.05 * abs(before),
        f"crossing contact kept approaching: {before:.3f} -> {after:.3f} m/s "
        "(a retained phi/h allowance leaves it closing at the gap rate)",
    )


devices = get_selected_cuda_test_devices()


class TestFeatherPGSVelocityPassSpeculative(unittest.TestCase):
    pass


for _fn in (
    test_velocity_iterations_do_not_halt_a_falling_body,
    test_every_contact_response_route_lands,
    test_non_crossing_row_keeps_its_speculative_allowance,
    test_crossing_row_loses_its_speculative_allowance,
    test_velocity_iterations_match_the_position_only_landing,
    test_separated_body_still_falls_through_the_margin,
):
    add_function_test(TestFeatherPGSVelocityPassSpeculative, _fn.__name__, _fn, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
