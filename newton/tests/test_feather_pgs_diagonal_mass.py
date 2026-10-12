# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Diagonal mass matrices of SolverFeatherPGS: detection and agreement with the factor paths."""

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices


def _build_fixed_base_star_model(
    device, num_branches=16, num_worlds=2, *, ground=False, armature=0.0, with_chain=False, ball_branch=False
):
    """Build fixed-base stars of independent prismatic branches (a structurally diagonal mass matrix).

    ``with_chain`` adds a two-link serial chain per world, so each world holds two response
    groups; ``ball_branch`` replaces the last branch by a BALL joint, whose DOFs are coupled.
    """
    star = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81 if ground else 0.0))
    base = star.add_link(mass=1.0, inertia=wp.mat33(np.eye(3)))
    joints = [
        star.add_joint_fixed(parent=-1, child=base, parent_xform=wp.transform((0.0, 0.0, 0.3), wp.quat_identity()))
    ]
    for branch in range(num_branches):
        child = star.add_link(mass=1.0 + 0.01 * branch, inertia=wp.mat33(np.eye(3)))
        if ground:
            star.add_shape_sphere(child, radius=0.05)
        parent_xform = wp.transform(wp.vec3(0.12 * branch, 0.0, 0.0), wp.quat_identity())
        if ball_branch and branch == num_branches - 1:
            joints.append(star.add_joint_ball(parent=base, child=child, parent_xform=parent_xform))
            continue
        joints.append(
            star.add_joint_prismatic(
                parent=base,
                child=child,
                axis=newton.Axis.Z,
                parent_xform=parent_xform,
                limit_lower=-0.4,
                limit_upper=0.05,
                armature=armature,
            )
        )
    star.add_articulation(joints)
    if with_chain:
        links = [star.add_link(mass=1.0, inertia=wp.mat33(np.eye(3))) for _ in range(2)]
        star.add_shape_sphere(links[1], radius=0.05)
        root = star.add_joint_revolute(
            -1, links[0], axis=newton.Axis.Y, parent_xform=wp.transform((-0.5, 0.0, 0.3), wp.quat_identity())
        )
        elbow = star.add_joint_revolute(
            links[0], links[1], axis=newton.Axis.Y, parent_xform=wp.transform((0.0, 0.0, -0.15), wp.quat_identity())
        )
        star.add_articulation([root, elbow])
    main = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81 if ground else 0.0))
    main.replicate(star, num_worlds, spacing=(3.0, 3.0, 0.0))
    if ground:
        main.add_ground_plane()
    return main.finalize(device=device)


def test_sleeping_diagonal_dynamics_skip_is_exact(test, device):
    """Skip the diagonal-mass dynamics of sleeping stars without changing any published state.

    Two solvers differ only in whether sleeping islands skip their dynamics; over a sleep, a
    force wake and a resettle, every state array stays bitwise equal.
    """
    runs = []
    for skip in (False, True):
        model = _build_fixed_base_star_model(device, num_branches=3, ground=True)
        solver = SolverFeatherPGS(
            model, pgs_mode="matrix_free", enable_sleeping=True, sleep_quiet_time=0.05, dense_max_constraints=64
        )
        test.assertTrue(solver._execution_plan.use_diagonal_mass(3))
        solver.sleeping.skip_dynamics = skip
        pipeline = newton.CollisionPipeline(model)
        runs.append([solver, pipeline, pipeline.contacts(), model.state(), model.state(), model.control()])
    slept = woke = False
    for step in range(500):
        for run in runs:
            solver, pipeline, contacts, state, out, control = run
            state.clear_forces()
            if 300 <= step < 306:
                forces = state.body_f.numpy()
                forces[1, 2] = 50.0
                state.body_f.assign(forces)
            pipeline.collide(state, contacts)
            solver.step(state, out, control, contacts, 1.0 / 120.0)
            run[3], run[4] = out, state
        for field in ("body_q", "body_qd", "joint_q", "joint_qd"):
            np.testing.assert_array_equal(
                getattr(runs[0][3], field).numpy(), getattr(runs[1][3], field).numpy(), err_msg=f"{step}: {field}"
            )
        awake = runs[1][0].sleeping.art_awake.numpy()
        slept |= step < 300 and not awake.any()
        woke |= 300 <= step < 306 and bool(awake[0])
    test.assertTrue(slept, "the diagonal-mass articulations never slept")
    test.assertTrue(woke, "the external force did not wake the first articulation")
    for run in runs:
        run[0].check_constraint_capacity()


class TestFeatherPGSDiagonalMass(unittest.TestCase):
    pass


devices = get_cuda_test_devices()
for _name, _func in (("test_sleeping_diagonal_dynamics_skip_is_exact", test_sleeping_diagonal_dynamics_skip_is_exact),):
    add_function_test(TestFeatherPGSDiagonalMass, _name, _func, devices=devices)


if __name__ == "__main__":
    unittest.main()
