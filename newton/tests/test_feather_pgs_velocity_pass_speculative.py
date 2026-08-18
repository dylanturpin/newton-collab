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


def _drop(device, velocity_iterations, response="immediate", steps=1400):
    """Drop a sphere onto a plane; return per-step surface height, velocity, contact count."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, RADIUS + DROP), wp.quat_identity()))
    builder.add_shape_sphere(body, radius=RADIUS, cfg=newton.ModelBuilder.ShapeConfig(mu=0.5))
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
    for _ in range(steps):
        contacts = model.collide(state_0)
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts, SIM_DT)
        state_0, state_1 = state_1, state_0
        heights.append(float(state_0.body_q.numpy()[body][2]) - RADIUS)
        velocities.append(float(state_0.body_qd.numpy()[body][2]))
        contact_counts.append(int(contacts.rigid_contact_count.numpy()[0]))
    return np.asarray(heights), np.asarray(velocities), np.asarray(contact_counts)


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
        h, v, n = _drop(device, iterations)
        _assert_landed(test, h, v, n, f"pgs_velocity_iterations={iterations}")


def test_velocity_iterations_match_the_position_only_landing(test, device):
    """Land at the same height with and without velocity iterations.

    The velocity pass refines velocities; it must not change where a simple
    unconstrained drop comes to rest.
    """
    baseline, base_v, base_n = _drop(device, 0)
    with_velocity, vel_v, vel_n = _drop(device, 4)
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
    for response in RESPONSE_MODES:
        h, v, n = _drop(device, 4, response=response)
        _assert_landed(test, h, v, n, f"articulated_contact_response={response!r}")


def test_separated_body_still_falls_through_the_margin(test, device):
    """Keep a body accelerating while it is inside the margin but not touching.

    The speculative row exists well before the surface is reached; it must not
    apply an impulse until the body would actually cross it.
    """
    heights, _v, _n = _drop(device, 4, steps=500)
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


devices = get_selected_cuda_test_devices()


class TestFeatherPGSVelocityPassSpeculative(unittest.TestCase):
    pass


for _fn in (
    test_velocity_iterations_do_not_halt_a_falling_body,
    test_every_contact_response_route_lands,
    test_velocity_iterations_match_the_position_only_landing,
    test_separated_body_still_falls_through_the_margin,
):
    add_function_test(TestFeatherPGSVelocityPassSpeculative, _fn.__name__, _fn, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
