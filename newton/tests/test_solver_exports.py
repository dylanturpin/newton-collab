# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import unittest

from newton import solvers
from newton._src.solvers import SolverFeatherPGS


class TestSolverExports(unittest.TestCase):
    def test_feather_pgs_public_export(self):
        self.assertIs(solvers.SolverFeatherPGS, SolverFeatherPGS)
        self.assertIn("SolverFeatherPGS", solvers.__all__)


if __name__ == "__main__":
    unittest.main()
