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


def _drop_heights(device, velocity_iterations, steps=1400):
    """Drop a sphere onto a plane and return its surface height each step."""
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
    )
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, state_0.joint_q, state_0.joint_qd, state_0)
    heights = []
    for _ in range(steps):
        contacts = model.collide(state_0)
        state_0.clear_forces()
        solver.step(state_0, state_1, control, contacts, SIM_DT)
        state_0, state_1 = state_1, state_0
        heights.append(float(state_0.body_q.numpy()[body][2]) - RADIUS)
    return np.asarray(heights)


def test_velocity_iterations_do_not_halt_a_falling_body(test, device):
    """Land a falling sphere on the plane with the velocity pass enabled.

    Without the fix the sphere stops inside the collision margin and hovers
    there for the rest of the run instead of reaching the surface.
    """
    for iterations in (2, 4, 8):
        heights = _drop_heights(device, iterations)
        settled = float(heights[-400:].max())
        test.assertLess(
            settled,
            0.005,
            f"pgs_velocity_iterations={iterations}: sphere halted at {settled:.4f} m instead of landing",
        )


def test_velocity_iterations_match_the_position_only_landing(test, device):
    """Land at the same height with and without velocity iterations.

    The velocity pass refines velocities; it must not change where a simple
    unconstrained drop comes to rest.
    """
    baseline = _drop_heights(device, 0)
    with_velocity = _drop_heights(device, 4)
    test.assertAlmostEqual(
        float(with_velocity[-1]),
        float(baseline[-1]),
        delta=0.002,
        msg=(
            f"resting height moved from {float(baseline[-1]):.4f} m to "
            f"{float(with_velocity[-1]):.4f} m when velocity iterations were enabled"
        ),
    )


def test_separated_body_still_falls_through_the_margin(test, device):
    """Keep a body accelerating while it is inside the margin but not touching.

    The speculative row exists well before the surface is reached; it must not
    apply an impulse until the body would actually cross it.
    """
    heights = _drop_heights(device, 4, steps=500)
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
    test_velocity_iterations_match_the_position_only_landing,
    test_separated_body_still_falls_through_the_margin,
):
    add_function_test(TestFeatherPGSVelocityPassSpeculative, _fn.__name__, _fn, devices=devices)


if __name__ == "__main__":
    wp.clear_kernel_cache()
    unittest.main(verbosity=2)
