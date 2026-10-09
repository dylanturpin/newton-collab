# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for FeatherPGS speculative-contact controls."""

import unittest

import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_cuda_test_devices

PATH_DENSE = 0
PATH_MATRIX_FREE = 1


# Body-frame witness points and thicknesses of the dense same-articulation fixture contact.
# World positions of the fixture's two contact bodies (shape 0 on body 2, shape 1 on body 1).


def _free_body_model(device):
    builder = newton.ModelBuilder()
    body = builder.add_body(xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()))
    builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
    return builder.finalize(device=device)


def test_shared_anchors_warn_with_patch_friction(test, device):
    """Explain that patch anchors keep their friction points when shared anchors are requested."""
    model = _free_body_model(device)
    for flag in ("contact_shared_anchor", "contact_friction_shared_anchor"):
        with test.subTest(flag=flag), test.assertWarnsRegex(UserWarning, "friction_anchor_beta=0"):
            SolverFeatherPGS(model, **{flag: True})
        solver = SolverFeatherPGS(model, friction_anchor_beta=0.0, **{flag: True})
        test.assertTrue(getattr(solver, flag))


class TestFeatherPGSContactControls(unittest.TestCase):
    pass


devices = get_cuda_test_devices()
for _fn in (test_shared_anchors_warn_with_patch_friction,):
    add_function_test(TestFeatherPGSContactControls, _fn.__name__, _fn, devices=devices)


if __name__ == "__main__":
    unittest.main(verbosity=2)
