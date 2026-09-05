# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import ast
import unittest
from pathlib import Path

_TILED_KERNEL_HELPERS = (
    "_get_prepare_mf_solve_metadata_kernel",
    "_get_pgs_solve_mf_gs_kernel",
    "_get_crba_cholesky_warp_kernel",
    "_get_crba_cholesky_kernel",
    "_get_cholesky_kernel",
    "_get_triangular_solve_kernel",
    "_get_hinv_jt_kernel",
    "_get_hinv_jt_plain_kernel",
    "_get_hinv_jt_persistent_kernel",
    "_get_hinv_jt_persistent_plain_kernel",
    "_get_hinv_jt_fused_kernel",
    "_get_delassus_kernel",
    "_get_pgs_solve_tiled_row_kernel",
    "_get_pgs_solve_tiled_contact_kernel",
    "_get_pgs_solve_streaming_kernel",
    "_get_pgs_solve_mf_kernel",
)

_DYNAMICS_COMPILE_DOMAINS = {
    "eval_rigid_fk_id": "_KINEMATICS_KERNEL_MODULE",
    "eval_rigid_fk_kinematics": "_KINEMATICS_KERNEL_MODULE",
    "eval_rigid_tau_add": "_INVERSE_DYNAMICS_KERNEL_MODULE",
    "compute_composite_inertia": "_MASS_DYNAMICS_KERNEL_MODULE",
    "crba_fill_par_dof": "_MASS_DYNAMICS_KERNEL_MODULE",
    "finalize_body_dynamics": "_MASS_DYNAMICS_KERNEL_MODULE",
}


class TestFeatherPGSPrivateApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        solver_path = Path(__file__).parents[1] / "_src" / "solvers" / "feather_pgs" / "solver_feather_pgs.py"
        cls.solver_module = ast.parse(solver_path.read_text())
        kernels_path = solver_path.with_name("kernels.py")
        cls.kernels_module = ast.parse(kernels_path.read_text())
        cls.kernel_functions = {
            node.name: node for node in cls.kernels_module.body if isinstance(node, ast.FunctionDef)
        }
        cls.top_level_functions = {
            node.name: node for node in cls.solver_module.body if isinstance(node, ast.FunctionDef)
        }
        cls.solver_class = next(
            node
            for node in cls.solver_module.body
            if isinstance(node, ast.ClassDef) and node.name == "SolverFeatherPGS"
        )
        cls.solver_methods = {node.name: node for node in cls.solver_class.body if isinstance(node, ast.FunctionDef)}

    def test_prescribed_response_is_not_a_public_execution_knob(self):
        init_method = self.solver_methods["__init__"]
        parameters = {
            argument.arg
            for argument in [*init_method.args.posonlyargs, *init_method.args.args, *init_method.args.kwonlyargs]
        }
        self.assertNotIn("exclude_fully_kinematic_free_articulations", parameters)

    def test_tiled_kernel_factory_is_not_exported(self):
        class_names = {node.name for node in self.solver_module.body if isinstance(node, ast.ClassDef)}
        self.assertNotIn("TiledKernelFactory", class_names)

    def test_tiled_kernel_helpers_are_private_cached_functions(self):
        for helper_name in _TILED_KERNEL_HELPERS:
            with self.subTest(helper_name=helper_name):
                self.assertIn(helper_name, self.top_level_functions)
                helper = self.top_level_functions[helper_name]
                decorator_names = {
                    decorator.id for decorator in helper.decorator_list if isinstance(decorator, ast.Name)
                }
                parameters = {arg.arg for arg in [*helper.args.posonlyargs, *helper.args.args, *helper.args.kwonlyargs]}
                self.assertIn("cache", decorator_names)
                self.assertIn("device_arch", parameters)

    def test_mf_solve_metadata_has_one_preparation_owner(self):
        """Do not restore the redundant DOF-offset handoff kernel."""
        self.assertNotIn("compute_mf_world_dof_offsets", self.kernel_functions)
        self.assertIn("_prepare_mf_solve_metadata", self.solver_methods)

    def test_mf_setup_does_not_clear_active_row_outputs(self):
        """Keep active-row ownership in the setup kernel instead of clearing full capacities."""
        setup = self.solver_methods["_mf_pgs_setup"]
        cleared = {
            call.func.value.attr
            for call in ast.walk(setup)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "zero_"
            and isinstance(call.func.value, ast.Attribute)
            and isinstance(call.func.value.value, ast.Name)
            and call.func.value.value.id == "self"
        }
        self.assertTrue({"mf_rhs", "mf_eff_mass_inv"}.isdisjoint(cleared))

    def test_mf_row_writers_own_active_initialization(self):
        """Initialize solve state at active row writers instead of sweeping MF capacity."""
        for kernel_name in ("_build_mf_contact_row", "build_mf_contact_rows", "populate_rigid_velocity_limit_rows"):
            with self.subTest(kernel_name=kernel_name):
                parameters = {argument.arg for argument in self.kernel_functions[kernel_name].args.args}
                self.assertIn("mf_impulses", parameters)
        velocity_limit_parameters = {
            argument.arg for argument in self.kernel_functions["populate_rigid_velocity_limit_rows"].args.args
        }
        self.assertIn("mf_target_velocity", velocity_limit_parameters)

        build = self.solver_methods["_stage4_build_rows"]
        cleared = {
            call.func.value.attr
            for call in ast.walk(build)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "zero_"
            and isinstance(call.func.value, ast.Attribute)
            and isinstance(call.func.value.value, ast.Name)
            and call.func.value.value.id == "self"
        }
        self.assertTrue({"mf_impulses", "mf_target_velocity"}.isdisjoint(cleared))

    def test_world_diag_writer_owns_active_initialization(self):
        """Let the active-row producer initialize world diagonals without a capacity clear."""
        method = self.solver_methods["_stage4_compute_matrix_free_diag"]
        top_level_clears = [
            call
            for statement in method.body
            for call in ast.walk(statement)
            if isinstance(statement, ast.Expr)
            and isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "zero_"
            and isinstance(call.func.value, ast.Attribute)
            and call.func.value.attr == "diag"
        ]
        self.assertFalse(top_level_clears)

        kernel = self.kernel_functions["diag_from_JY_world"]
        owner_returns = [
            node
            for node in ast.walk(kernel)
            if isinstance(node, ast.If)
            and any(isinstance(child, ast.Name) and child.id == "local_solve_owner" for child in ast.walk(node.test))
            and any(isinstance(child, ast.Return) for child in ast.walk(ast.Module(body=node.body, type_ignores=[])))
        ]
        self.assertFalse(owner_returns)

    def test_local_solve_owners_build_their_effective_diagonal(self):
        """Keep local effective diagonals inside the solver that consumes them."""
        self.assertNotIn("finalize_local_owner_world_diag", self.kernel_functions)
        for factory_name in ("_get_pgs_solve_local_owned_kernel", "_get_pgs_solve_mf_gs_kernel"):
            with self.subTest(factory_name=factory_name):
                nested_functions = [
                    node
                    for node in ast.walk(self.top_level_functions[factory_name])
                    if isinstance(node, ast.FunctionDef)
                ]
                parameter_sets = [{argument.arg for argument in node.args.args} for node in nested_functions]
                self.assertTrue(any("world_row_cfm" in parameters for parameters in parameter_sets))

    def test_matrix_free_rhs_has_one_active_row_owner(self):
        """Build matrix-free bias and restitution in one active-row pass."""
        self.assertIn("compute_world_contact_bias_restitution_matrix_free", self.kernel_functions)
        self.assertNotIn("apply_world_contact_restitution_matrix_free", self.kernel_functions)

    def test_stage7_kinematics_uses_full_warp_blocks(self):
        """Keep one articulation worker in every lane of a CUDA warp."""
        method = self.solver_methods["_stage7_update_kinematics"]
        launch = next(
            call
            for call in ast.walk(method)
            if isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "launch"
            and call.args
            and isinstance(call.args[0], ast.Name)
            and call.args[0].id == "eval_rigid_fk_kinematics"
        )
        block_dim = next(keyword.value for keyword in launch.keywords if keyword.arg == "block_dim")
        self.assertIsInstance(block_dim, ast.Constant)
        self.assertEqual(block_dim.value, 32)

    def test_mimic_row_population_is_constraint_parallel(self):
        """Keep independent mimic rows out of articulation-serial loops."""
        kernel = self.kernel_functions["populate_mimic_J_for_size"]
        parameters = {argument.arg for argument in kernel.args.args}
        self.assertIn("mimic_indices", parameters)
        self.assertIn("mimic_articulation", parameters)
        self.assertIn("articulation_group_index", parameters)
        self.assertNotIn("mimic_art_start", parameters)
        self.assertNotIn("mimic_art_list", parameters)

    def test_local_owner_classification_owns_dispatch(self):
        """Do not restore post-classification candidate rescans."""
        self.assertIn("classify_and_dispatch_local_solve_worlds", self.kernel_functions)
        self.assertNotIn("classify_local_solve_worlds", self.kernel_functions)
        self.assertNotIn("compact_local_pair_candidates", self.kernel_functions)
        self.assertNotIn("compact_local_residual_candidates", self.kernel_functions)

    def test_launch_specific_dynamics_kernels_have_separate_compile_domains(self):
        for kernel_name, expected_module in _DYNAMICS_COMPILE_DOMAINS.items():
            with self.subTest(kernel_name=kernel_name):
                kernel = self.kernel_functions[kernel_name]
                decorator = next(
                    decorator
                    for decorator in kernel.decorator_list
                    if isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Attribute)
                    and decorator.func.attr == "kernel"
                )
                module_keyword = next(keyword for keyword in decorator.keywords if keyword.arg == "module")
                self.assertIsInstance(module_keyword.value, ast.Name)
                self.assertEqual(module_keyword.value.id, expected_module)

    def test_plain_hinv_jt_kernel_omits_diagonal_output(self):
        plain_factory = self.top_level_functions["_get_hinv_jt_plain_kernel"]
        diagonal_factory = self.top_level_functions["_get_hinv_jt_kernel"]
        plain_template = next(node for node in plain_factory.body if isinstance(node, ast.FunctionDef))
        diagonal_template = next(node for node in diagonal_factory.body if isinstance(node, ast.FunctionDef))

        diagonal_factory_parameters = {argument.arg for argument in diagonal_factory.args.args}
        plain_parameters = {argument.arg for argument in plain_template.args.args}
        diagonal_parameters = {argument.arg for argument in diagonal_template.args.args}
        plain_names = {node.id for node in ast.walk(plain_template) if isinstance(node, ast.Name)}
        diagonal_names = {node.id for node in ast.walk(diagonal_factory) if isinstance(node, ast.Name)}

        self.assertNotIn("diag_group", plain_parameters)
        self.assertNotIn("diag_tile", plain_names)
        self.assertIn("compute_diag", diagonal_factory_parameters)
        self.assertIn("COMPUTE_DIAG", diagonal_names)
        self.assertIn("diag_group", diagonal_parameters)
        self.assertIn("diag_tile", diagonal_names)

    def test_persistent_plain_hinv_jt_kernel_omits_diagonal_output(self):
        plain_factory = self.top_level_functions["_get_hinv_jt_persistent_plain_kernel"]
        diagonal_factory = self.top_level_functions["_get_hinv_jt_persistent_kernel"]
        plain_template = next(node for node in plain_factory.body if isinstance(node, ast.FunctionDef))
        diagonal_template = next(node for node in diagonal_factory.body if isinstance(node, ast.FunctionDef))

        plain_parameters = {argument.arg for argument in plain_template.args.args}
        diagonal_parameters = {argument.arg for argument in diagonal_template.args.args}
        plain_names = {node.id for node in ast.walk(plain_template) if isinstance(node, ast.Name)}
        diagonal_names = {node.id for node in ast.walk(diagonal_template) if isinstance(node, ast.Name)}

        self.assertNotIn("diag_group", plain_parameters)
        self.assertNotIn("diag_tile", plain_names)
        self.assertIn("diag_group", diagonal_parameters)
        self.assertIn("diag_tile", diagonal_names)

    def test_cholesky_and_triangular_kernels_are_not_dense_constraint_gated(self):
        init_method = self.solver_methods["_init_tiled_kernels"]
        size_group_loop = self._find_size_group_loop(init_method)
        first_dense_continue = self._first_dense_guard_continue_line(size_group_loop)

        for helper_name in ("_get_cholesky_kernel", "_get_triangular_solve_kernel"):
            with self.subTest(helper_name=helper_name):
                call_line = self._first_call_line(size_group_loop, helper_name)
                if first_dense_continue is not None:
                    self.assertLess(call_line, first_dense_continue)

        dense_guarded_none_assignments = self._dense_guarded_none_assignments(size_group_loop)
        self.assertNotIn("_cholesky_kernels_by_size", dense_guarded_none_assignments)
        self.assertNotIn("_triangular_solve_kernels_by_size", dense_guarded_none_assignments)

    def test_tiled_triangular_solve_has_no_grouped_vector_buffers(self):
        """Keep the tiled solve boundary in canonical joint storage."""
        attribute_names = {node.attr for node in ast.walk(self.solver_class) if isinstance(node, ast.Attribute)}
        self.assertNotIn("tau_by_size", attribute_names)
        self.assertNotIn("qdd_by_size", attribute_names)

    @staticmethod
    def _find_size_group_loop(function_node):
        for node in ast.walk(function_node):
            if isinstance(node, ast.For) and isinstance(node.iter, ast.Attribute):
                if node.iter.attr == "size_groups":
                    return node
        raise AssertionError("_init_tiled_kernels does not iterate over self.size_groups")

    @staticmethod
    def _references_dense_max_constraints(node):
        return any(
            isinstance(child, ast.Attribute) and child.attr == "dense_max_constraints" for child in ast.walk(node)
        )

    @classmethod
    def _first_dense_guard_continue_line(cls, node):
        continue_lines = []
        for child in ast.walk(node):
            if isinstance(child, ast.If) and cls._references_dense_max_constraints(child.test):
                for body_node in child.body:
                    continue_lines.extend(
                        descendant.lineno for descendant in ast.walk(body_node) if isinstance(descendant, ast.Continue)
                    )
        return min(continue_lines) if continue_lines else None

    @staticmethod
    def _first_call_line(node, function_name):
        call_lines = [
            child.lineno
            for child in ast.walk(node)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name) and child.func.id == function_name
        ]
        if not call_lines:
            raise AssertionError(f"{function_name} is not called while initializing tiled kernels")
        return min(call_lines)

    @classmethod
    def _dense_guarded_none_assignments(cls, node):
        assigned_attrs = set()
        for child in ast.walk(node):
            if not isinstance(child, ast.If) or not cls._references_dense_max_constraints(child.test):
                continue
            for body_node in child.body:
                for descendant in ast.walk(body_node):
                    if not isinstance(descendant, ast.Assign):
                        continue
                    if not isinstance(descendant.value, ast.Constant) or descendant.value.value is not None:
                        continue
                    for target in descendant.targets:
                        attr_name = cls._assigned_self_collection_attr(target)
                        if attr_name is not None:
                            assigned_attrs.add(attr_name)
        return assigned_attrs

    @staticmethod
    def _assigned_self_collection_attr(target):
        if isinstance(target, ast.Subscript):
            target = target.value
        if isinstance(target, ast.Attribute):
            return target.attr
        return None


if __name__ == "__main__":
    unittest.main()
