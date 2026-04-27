# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Parameter-plumbing tests for the FeatherPGS ``friction_mode`` config axis.

``FPGS Friction Modes 4/13`` factored the per-row Coulomb friction step
out of the matrix-free PGS kernel body into a Warp ``@wp.func`` seam so
later issues in the series could plug in alternate strategies without
rewriting the kernel body.  ``FPGS Friction Modes 5/13`` then wired
``friction_mode="bisection"`` (RAISim-style bisection on λ_n) through
that seam, and ``FPGS Friction Modes 6/13`` wired
``friction_mode="bisection_desaxce"`` (same bisection with the de Saxce
``μ · ‖c_T‖`` bias correction) on top of it.  Only
``"coulomb_newton"`` remains tracked as a follow-up issue.

The tests in this file cover:

1. Defaulting: ``friction_mode`` defaults to ``"current"``.
2. Validation: unknown values raise :class:`ValueError`.
3. ``friction_mode="bisection"`` constructs successfully under
   ``pgs_mode="matrix_free"`` and stores the string on the solver.
4. Reserved-but-unimplemented modes under ``pgs_mode="matrix_free"``
   raise :class:`NotImplementedError` whose message names the follow-up
   issue.
5. Any non-``"current"`` mode with ``pgs_mode`` other than
   ``"matrix_free"`` raises :class:`ValueError`.
"""

from __future__ import annotations

import unittest

import warp as wp

import newton
from newton._src.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_test_devices


def _build_one_dof_pendulum_model(device: str) -> newton.Model:
    """Build a minimal 1-DOF revolute pendulum for FPGS-plumbing tests."""
    builder = newton.ModelBuilder()
    link = builder.add_link()
    joint = builder.add_joint_revolute(
        parent=-1,
        child=link,
        axis=wp.vec3(0.0, 0.0, 1.0),
        target_ke=0.0,
        target_kd=0.0,
    )
    builder.add_articulation([joint])
    model = builder.finalize(device=device)
    return model


class TestFeatherPGSFrictionModeDefault(unittest.TestCase):
    """Default and round-trip behavior for the ``friction_mode`` kwarg."""


def run_default_mode_is_current(test: TestFeatherPGSFrictionModeDefault, device):
    """With no kwarg, ``friction_mode`` defaults to the baseline ``"current"``."""
    model = _build_one_dof_pendulum_model(device)
    solver = SolverFeatherPGS(model)
    test.assertEqual(solver.friction_mode, "current")


def run_current_mode_explicit_round_trip(test: TestFeatherPGSFrictionModeDefault, device):
    """Explicitly requesting ``"current"`` is accepted under any ``pgs_mode``."""
    model = _build_one_dof_pendulum_model(device)
    for pgs_mode in ("dense", "split", "matrix_free"):
        solver = SolverFeatherPGS(model, pgs_mode=pgs_mode, friction_mode="current")
        test.assertEqual(solver.friction_mode, "current")
        test.assertEqual(solver.pgs_mode, pgs_mode)


def run_invalid_mode_raises(test: TestFeatherPGSFrictionModeDefault, device):
    """An unknown ``friction_mode`` value must raise :class:`ValueError`."""
    model = _build_one_dof_pendulum_model(device)
    with test.assertRaises(ValueError):
        SolverFeatherPGS(model, friction_mode="bogus")


class TestFeatherPGSFrictionModeBisection(unittest.TestCase):
    """``friction_mode="bisection"`` is wired on ``pgs_mode="matrix_free"``."""


def run_bisection_construct_matrix_free(test: TestFeatherPGSFrictionModeBisection, device):
    """``friction_mode="bisection"`` now constructs under matrix_free (5/13)."""
    model = _build_one_dof_pendulum_model(device)
    solver = SolverFeatherPGS(model, pgs_mode="matrix_free", friction_mode="bisection")
    test.assertEqual(solver.friction_mode, "bisection")
    test.assertEqual(solver.pgs_mode, "matrix_free")


class TestFeatherPGSFrictionModeBisectionDeSaxce(unittest.TestCase):
    """``friction_mode="bisection_desaxce"`` is wired on ``pgs_mode="matrix_free"`` (6/13)."""


def run_bisection_desaxce_construct_matrix_free(test: TestFeatherPGSFrictionModeBisectionDeSaxce, device):
    """``friction_mode="bisection_desaxce"`` constructs under matrix_free (6/13)."""
    model = _build_one_dof_pendulum_model(device)
    solver = SolverFeatherPGS(model, pgs_mode="matrix_free", friction_mode="bisection_desaxce")
    test.assertEqual(solver.friction_mode, "bisection_desaxce")
    test.assertEqual(solver.pgs_mode, "matrix_free")


class TestFeatherPGSFrictionModeCoulombNewton(unittest.TestCase):
    """``friction_mode="coulomb_newton"`` is wired on ``pgs_mode="matrix_free"`` (7/13)."""


def run_coulomb_newton_construct_matrix_free(test: TestFeatherPGSFrictionModeCoulombNewton, device):
    """``friction_mode="coulomb_newton"`` constructs under matrix_free (7/13).

    Previously this constructor raised :class:`NotImplementedError`
    while Gilles Daviet's 1D Coulomb Newton was still reserved.  FPGS
    Friction Modes 7/13 wires it in; the acceptance check is now that
    the constructor succeeds and the mode round-trips on the solver.
    """
    model = _build_one_dof_pendulum_model(device)
    solver = SolverFeatherPGS(model, pgs_mode="matrix_free", friction_mode="coulomb_newton")
    test.assertEqual(solver.friction_mode, "coulomb_newton")
    test.assertEqual(solver.pgs_mode, "matrix_free")


class TestFeatherPGSFrictionModeRejectsNonMatrixFree(unittest.TestCase):
    """Non-``"current"`` is matrix-free only; other ``pgs_mode`` values reject it."""


def run_dense_rejects_bisection(test: TestFeatherPGSFrictionModeRejectsNonMatrixFree, device):
    """``pgs_mode="dense"`` + ``friction_mode="bisection"`` is rejected loudly."""
    model = _build_one_dof_pendulum_model(device)
    with test.assertRaises(ValueError) as cm:
        SolverFeatherPGS(model, pgs_mode="dense", friction_mode="bisection")
    msg = str(cm.exception)
    test.assertIn("matrix_free", msg)
    test.assertIn("bisection", msg)


def run_split_rejects_bisection(test: TestFeatherPGSFrictionModeRejectsNonMatrixFree, device):
    """``pgs_mode="split"`` + ``friction_mode="bisection"`` is rejected loudly."""
    model = _build_one_dof_pendulum_model(device)
    with test.assertRaises(ValueError) as cm:
        SolverFeatherPGS(model, pgs_mode="split", friction_mode="bisection")
    msg = str(cm.exception)
    test.assertIn("matrix_free", msg)
    test.assertIn("bisection", msg)


devices = get_test_devices()

for device in devices:
    add_function_test(
        TestFeatherPGSFrictionModeDefault,
        f"test_default_mode_is_current_{device}",
        run_default_mode_is_current,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSFrictionModeDefault,
        f"test_current_mode_explicit_round_trip_{device}",
        run_current_mode_explicit_round_trip,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSFrictionModeDefault,
        f"test_invalid_mode_raises_{device}",
        run_invalid_mode_raises,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSFrictionModeBisection,
        f"test_bisection_construct_matrix_free_{device}",
        run_bisection_construct_matrix_free,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSFrictionModeBisectionDeSaxce,
        f"test_bisection_desaxce_construct_matrix_free_{device}",
        run_bisection_desaxce_construct_matrix_free,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSFrictionModeCoulombNewton,
        f"test_coulomb_newton_construct_matrix_free_{device}",
        run_coulomb_newton_construct_matrix_free,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSFrictionModeRejectsNonMatrixFree,
        f"test_dense_rejects_bisection_{device}",
        run_dense_rejects_bisection,
        devices=[device],
    )
    add_function_test(
        TestFeatherPGSFrictionModeRejectsNonMatrixFree,
        f"test_split_rejects_bisection_{device}",
        run_split_rejects_bisection,
        devices=[device],
    )


if __name__ == "__main__":
    unittest.main()
