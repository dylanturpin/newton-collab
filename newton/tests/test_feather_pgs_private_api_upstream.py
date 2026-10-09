# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Static checks of the FeatherPGS public surface and its private kernel factories."""

import ast
import unittest
from pathlib import Path

import newton
from newton.solvers import SolverFeatherPGS

_PACKAGE_DIR = Path(__file__).parents[1] / "_src" / "solvers" / "feather_pgs"


class TestFeatherPGSPrivateApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        solver_path = _PACKAGE_DIR / "solver_feather_pgs.py"
        cls.solver_module = ast.parse(solver_path.read_text(encoding="utf-8"))
        cls.top_level_functions = {
            node.name: node for node in cls.solver_module.body if isinstance(node, ast.FunctionDef)
        }
        cls.solver_class = next(
            node
            for node in cls.solver_module.body
            if isinstance(node, ast.ClassDef) and node.name == "SolverFeatherPGS"
        )
        cls.solver_methods = {node.name: node for node in cls.solver_class.body if isinstance(node, ast.FunctionDef)}

    def test_solver_is_exported_once_from_newton_solvers(self):
        """Expose the solver through newton.solvers only."""
        self.assertIn("SolverFeatherPGS", newton.solvers.__all__)
        self.assertIs(newton.solvers.SolverFeatherPGS, SolverFeatherPGS)
        self.assertNotIn("SolverFeatherPGS", newton.__all__)

    def test_cholesky_and_triangular_kernels_are_built_for_every_group(self):
        """Build the factorization kernels independently of the dense row capacity."""
        init_method = self.solver_methods["_init_tiled_kernels"]
        size_group_loop = next(
            node
            for node in ast.walk(init_method)
            if isinstance(node, ast.For) and isinstance(node.iter, ast.Attribute) and node.iter.attr == "size_groups"
        )
        calls = {
            child.func.id: child.lineno
            for child in ast.walk(size_group_loop)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
        }
        first_continue = min(
            (child.lineno for child in ast.walk(size_group_loop) if isinstance(child, ast.Continue)), default=None
        )
        for helper_name in ("_get_cholesky_kernel", "_get_triangular_solve_kernel"):
            with self.subTest(helper_name=helper_name):
                self.assertIn(helper_name, calls)
                if first_continue is not None:
                    self.assertLess(calls[helper_name], first_continue)


if __name__ == "__main__":
    unittest.main()
