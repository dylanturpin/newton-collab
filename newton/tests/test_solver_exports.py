# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest
import sys

from newton import solvers
from newton._src import solvers as private_solvers
from newton._src.solvers import SolverFeatherPGS


class TestSolverExports(unittest.TestCase):
    def test_feather_pgs_public_export(self):
        self.assertIs(solvers.SolverFeatherPGS, SolverFeatherPGS)
        self.assertIn("SolverFeatherPGS", solvers.__all__)

    def test_feather_pgs_export_does_not_eagerly_import_implicit_mpm(self):
        self.assertIs(private_solvers.SolverFeatherPGS, SolverFeatherPGS)
        self.assertIn("SolverImplicitMPM", private_solvers.__all__)
        self.assertIn("SolverImplicitMPM", solvers.__all__)
        self.assertNotIn("SolverImplicitMPM", private_solvers.__dict__)
        self.assertNotIn("SolverImplicitMPM", solvers.__dict__)
        self.assertNotIn("newton._src.solvers.implicit_mpm", sys.modules)
        self.assertNotIn("newton._src.solvers.implicit_mpm.solver_implicit_mpm", sys.modules)


if __name__ == "__main__":
    unittest.main()
