# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
import unittest


class TestTopLevelLazyImports(unittest.TestCase):
    def test_viewer_import_does_not_eager_import_solvers(self):
        for module_name in list(sys.modules):
            if module_name == "newton" or module_name.startswith("newton."):
                del sys.modules[module_name]

        viewer = importlib.import_module("newton.viewer")

        self.assertTrue(hasattr(viewer, "ViewerRerun"))
        self.assertNotIn("newton.solvers", sys.modules)

        import newton

        self.assertIs(importlib.import_module("newton.solvers"), newton.solvers)


if __name__ == "__main__":
    unittest.main(verbosity=2)
