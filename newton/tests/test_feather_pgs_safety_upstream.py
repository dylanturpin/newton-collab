# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Constraint-capacity status of SolverFeatherPGS."""

import unittest

import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices, get_test_devices


def _box_on_ground(device, worlds=1):
    template = newton.ModelBuilder()
    body = template.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 0.1), wp.quat_identity()))
    template.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    builder = newton.ModelBuilder()
    builder.replicate(template, worlds, spacing=(1.0, 0.0, 0.0))
    builder.add_ground_plane()
    return builder.finalize(device=device)


def test_capacity_failure_is_observable(test, device, pgs_mode="matrix_free"):
    """Flag dropped free-body contact rows until reset, without optional telemetry."""
    model = _box_on_ground(device)
    solver = SolverFeatherPGS(model, pgs_mode=pgs_mode, mf_max_constraints=1, warn_constraint_overflow=False)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    state_in, state_out = model.state(), model.state()
    pipeline.collide(state_in, contacts)
    solver.step(state_in, state_out, model.control(), contacts, 1.0 / 240.0)
    with test.assertRaisesRegex(RuntimeError, "capacity"):
        solver.check_constraint_capacity()
    test.assertTrue(solver.constraint_overflow.numpy()[0])
    test.assertGreater(int(solver._row_dropped_mf.numpy()[0]), 0)
    solver.reset(state_out)
    solver.check_constraint_capacity()


class TestFeatherPGSCapacityStatus(unittest.TestCase):
    pass


devices = get_cuda_test_devices()
add_function_test(
    TestFeatherPGSCapacityStatus,
    "test_capacity_failure_is_observable",
    test_capacity_failure_is_observable,
    devices=devices,
)
for _name in ("test_capacity_failure_is_observable",):
    add_function_test(
        TestFeatherPGSCapacityStatus,
        f"{_name}_split",
        globals()[_name],
        devices=get_test_devices(),
        check_output=_name != "test_overflow_warning_is_printed_once",
        pgs_mode="split",
    )


if __name__ == "__main__":
    unittest.main()
