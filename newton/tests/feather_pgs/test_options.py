# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Global worlds, target layouts and the private API surface of SolverFeatherPGS."""

import ast
import unittest
import warnings
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS
from newton.tests.unittest_utils import add_function_test, get_selected_cuda_test_devices, get_test_devices

_DT = 0.01


def _floor_model(device, floor: str, kinematic_world0=False):
    """Two worlds with one unit box each on a floor box whose top is at z = 0.5.

    ``floor`` is ``"static"`` (world geometry), ``"kinematic"`` (a global kinematic body)
    or ``"dynamic"`` (a global dynamic body).
    """
    builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
    for x in (-2.0, 2.0):
        world = newton.ModelBuilder(gravity=wp.vec3(0.0))
        kinematic = kinematic_world0 and builder.world_count == 0
        body = world.add_body(
            xform=wp.transform(wp.vec3(x, 0.0, 0.999), wp.quat_identity()), mass=1.0, is_kinematic=kinematic
        )
        world.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5, cfg=newton.ModelBuilder.ShapeConfig(mu=0.0))
        builder.add_world(world)
    floor_body = -1 if floor == "static" else builder.add_body(is_kinematic=floor == "kinematic", mass=1.0)
    builder.add_shape_box(floor_body, hx=5.0, hy=5.0, hz=0.5, cfg=newton.ModelBuilder.ShapeConfig(mu=0.0))
    return builder.finalize(device=device)


def _step_once(model, solver, box_velocity=(-1.0, -1.0), floor_velocity=0.0):
    state, output = model.state(), model.state()
    joint_qd = state.joint_qd.numpy()
    joint_qd[2], joint_qd[8] = box_velocity
    if joint_qd.size > 12:
        joint_qd[14] = floor_velocity
    state.joint_qd.assign(joint_qd)
    newton.eval_fk(model, state.joint_q, state.joint_qd, state)
    pipeline = newton.CollisionPipeline(model)
    contacts = pipeline.contacts()
    pipeline.collide(state, contacts)
    solver.step(state, output, model.control(), contacts, _DT)
    return output.body_qd.numpy()[:, 2], contacts


def _construct(test, model, expect_warning, **kwargs):
    """Construct the solver and check whether it warns about unsolvable global contacts."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver = SolverFeatherPGS(model, **kwargs)
    messages = [str(w.message) for w in caught if "cannot be solved" in str(w.message)]
    test.assertEqual(len(messages), 1 if expect_warning else 0, msg=messages)
    if expect_warning:
        test.assertIn("global (world -1) articulations", messages[0])
    return solver


def _modes(device):
    return ["matrix_free", "split"] if device.is_cuda else ["split"]


class TestFeatherPGSGlobalWorld(unittest.TestCase):
    def test_global_kinematic_floor_supports_every_world(self):
        """A global kinematic floor stops the boxes of every world exactly like world geometry does."""
        device = wp.get_device()
        for mode in ["matrix_free"] if device.is_cuda else []:
            with self.subTest(mode=mode):
                static_model = _floor_model(device, "static")
                static_v = _step_once(static_model, SolverFeatherPGS(static_model, solve_mode=mode))[0][:2]
                self.assertLess(np.max(np.abs(static_v)), 0.05)

                model = _floor_model(device, "kinematic")
                solver = SolverFeatherPGS(model, solve_mode=mode, warn_constraint_overflow=False)
                v, contacts = _step_once(model, solver)
                count = int(contacts.rigid_contact_count.numpy()[0])
                self.assertGreater(count, 0)
                np.testing.assert_array_equal(solver._contact_path.numpy()[:count] >= 0, True)
                np.testing.assert_allclose(v[:2], static_v, atol=1.0e-5)
                np.testing.assert_array_equal(solver.constraint_overflow.numpy(), [False, False])

                moving_v = _step_once(model, SolverFeatherPGS(model, solve_mode=mode), floor_velocity=0.5)[0][:2]
                np.testing.assert_allclose(moving_v, static_v + 0.5, atol=1.0e-4)

                # The floor is elided even when world 0, where it is stored, has no dynamic body.
                model = _floor_model(device, "kinematic", kinematic_world0=True)
                solver = SolverFeatherPGS(model, solve_mode=mode, warn_constraint_overflow=False)
                v, _ = _step_once(model, solver, box_velocity=(0.0, -1.0))
                np.testing.assert_allclose(v[1], static_v[1], atol=1.0e-5)
                np.testing.assert_array_equal(solver.constraint_overflow.numpy(), [False, False])

    def test_split_global_kinematic_floor_flags_other_worlds(self):
        """Without prescribed-response elision, other worlds' contacts with a global kinematic body are flagged."""
        device = wp.get_device()
        model = _floor_model(device, "kinematic")
        solver = _construct(self, model, True, solve_mode="split", warn_constraint_overflow=False)
        _, contacts = _step_once(model, solver)
        count = int(contacts.rigid_contact_count.numpy()[0])
        paths = solver._contact_path.numpy()[:count]
        shape0 = contacts.rigid_contact_shape0.numpy()[:count]
        shape1 = contacts.rigid_contact_shape1.numpy()[:count]
        world1 = (shape0 == 1) | (shape1 == 1)
        self.assertTrue(np.any(world1))
        np.testing.assert_array_equal(paths[~world1] >= 0, True)
        np.testing.assert_array_equal(paths[world1], -1)
        np.testing.assert_array_equal(solver.constraint_overflow.numpy(), [True, True])

    def test_dynamic_global_body_flags_other_world_contacts(self):
        """Contacts between a dynamic global body and another world are dropped and flagged, not ignored."""
        device = wp.get_device()
        for mode in _modes(device):
            with self.subTest(mode=mode):
                model = _floor_model(device, "dynamic")
                joint_q = model.joint_q.numpy()
                joint_q[2] += 3.0  # lift world 0's box out of contact
                model.joint_q.assign(joint_q)
                solver = _construct(self, model, True, solve_mode=mode, warn_constraint_overflow=False)
                _, contacts = _step_once(model, solver)
                count = int(contacts.rigid_contact_count.numpy()[0])
                self.assertGreater(count, 0)
                np.testing.assert_array_equal(solver._contact_path.numpy()[:count], -1)
                # The global body is solved in world 0, so both worlds are flagged.
                np.testing.assert_array_equal(solver.constraint_overflow.numpy(), [True, True])

    def test_construction_warns_about_unsolvable_global_contacts(self):
        """The constructor warns once when shapes allow contacts that couple a global articulation with another world."""
        device = wp.get_device()
        for mode in _modes(device):
            with self.subTest(mode=mode):
                # A dynamic global body always keeps response DOFs in world 0.
                _construct(self, _floor_model(device, "dynamic"), True, solve_mode=mode)
                # A global kinematic free body is prescribed only on the matrix_free/immediate path.
                _construct(self, _floor_model(device, "kinematic"), mode != "matrix_free", solve_mode=mode)
                # World geometry is solvable in every world.
                _construct(self, _floor_model(device, "static"), False, solve_mode=mode)

                # One world, or global shapes that cannot collide with the worlds' bodies, never warn.
                builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
                body = builder.add_body(mass=1.0)
                builder.add_shape_box(body, hx=0.1, hy=0.1, hz=0.1)
                builder.add_ground_plane()
                _construct(self, builder.finalize(device=device), False, solve_mode=mode)
                model = _floor_model(device, "dynamic")
                groups = model.shape_collision_group.numpy()
                groups[model.shape_body.numpy() == 2] = 7
                model.shape_collision_group.assign(groups)
                _construct(self, model, False, solve_mode=mode)

        if device.is_cuda:
            # A kinematic global articulation with joints keeps response DOFs on every path.
            builder = newton.ModelBuilder(gravity=wp.vec3(0.0))
            for x in (-2.0, 2.0):
                world = newton.ModelBuilder(gravity=wp.vec3(0.0))
                body = world.add_body(xform=wp.transform(wp.vec3(x, 0.0, 0.999), wp.quat_identity()), mass=1.0)
                world.add_shape_box(body, hx=0.5, hy=0.5, hz=0.5)
                builder.add_world(world)
            floor = builder.add_link(mass=1.0, is_kinematic=True)
            builder.add_articulation([builder.add_joint_prismatic(-1, floor, axis=newton.Axis.Z)])
            builder.add_shape_box(floor, hx=5.0, hy=5.0, hz=0.5)
            _construct(self, builder.finalize(device=device), True, solve_mode="matrix_free")


def _floating_base_arm():
    """A free-floating base with one position-driven revolute joint, no gravity."""
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, 0.0))
    base = builder.add_link(mass=5.0, inertia=wp.mat33(np.eye(3) * 0.1))
    builder.add_shape_box(base, hx=0.1, hy=0.1, hz=0.1)
    free = builder.add_joint_free(child=base)
    arm = builder.add_link(mass=0.5, inertia=wp.mat33(np.eye(3) * 0.01))
    builder.add_shape_box(arm, hx=0.1, hy=0.02, hz=0.02)
    hinge = builder.add_joint_revolute(
        parent=base,
        child=arm,
        axis=newton.Axis.Z,
        target_ke=200.0,
        target_kd=20.0,
        actuator_mode=newton.JointTargetMode.POSITION,
    )
    builder.add_articulation([free, hinge])
    return builder.finalize(), hinge


def test_floating_base_drive_reads_its_own_target(test, device, drive_mode):
    """A revolute drive after a free joint must read its target at joint_target_q_start, not its DOF index."""
    with wp.ScopedDevice(device):
        model, hinge = _floating_base_arm()
        solve_mode = "matrix_free" if wp.get_device(device).is_cuda else "split"
        solver = newton.solvers.SolverFeatherPGS(model, solve_mode=solve_mode, iterations=16, drive_mode=drive_mode)
        state_0, state_1 = model.state(), model.state()
        control = model.control()
        targets = control.joint_target_q.numpy()
        targets[int(model.joint_target_q_start.numpy()[hinge])] = 0.4
        control.joint_target_q.assign(targets)
        for _ in range(240):
            solver.step(state_0, state_1, control, None, 1.0 / 240.0)
            state_0, state_1 = state_1, state_0
        q = state_0.joint_q.numpy()[int(model.joint_q_start.numpy()[hinge])]
        test.assertAlmostEqual(float(q), 0.4, delta=0.03)


class TestFeatherPGSTargetLayout(unittest.TestCase):
    pass


# rows drives require the CUDA-only matrix-free mode.
for _mode, _devices in (("augmented", get_test_devices()), ("rows", get_selected_cuda_test_devices())):
    add_function_test(
        TestFeatherPGSTargetLayout,
        f"test_floating_base_drive_reads_its_own_target_{_mode}",
        test_floating_base_drive_reads_its_own_target,
        devices=_devices,
        drive_mode=_mode,
    )


_TILED_KERNEL_HELPERS = (
    "_get_pack_mf_meta_kernel",
    "_get_pgs_solve_mf_gs_kernel",
    "_get_cholesky_kernel",
    "_get_triangular_solve_kernel",
    "_get_hinv_jt_kernel",
    "_get_hinv_jt_plain_kernel",
    "_get_hinv_jt_fused_kernel",
    "_get_delassus_kernel",
    "_get_pgs_solve_tiled_row_kernel",
    "_get_pgs_solve_tiled_contact_kernel",
    "_get_pgs_solve_streaming_kernel",
    "_get_pgs_solve_mf_kernel",
)


class TestFeatherPGSPrivateApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        solver_path = Path(__file__).parents[2] / "_src" / "solvers" / "feather_pgs" / "solver_feather_pgs.py"
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

    def test_plain_hinv_jt_kernel_omits_diagonal_output(self):
        plain_factory = self.top_level_functions["_get_hinv_jt_plain_kernel"]
        diagonal_factory = self.top_level_functions["_get_hinv_jt_kernel"]
        plain_template = next(node for node in plain_factory.body if isinstance(node, ast.FunctionDef))
        diagonal_template = next(node for node in diagonal_factory.body if isinstance(node, ast.FunctionDef))

        plain_parameters = {argument.arg for argument in plain_template.args.args}
        diagonal_parameters = {argument.arg for argument in diagonal_template.args.args}
        plain_names = {node.id for node in ast.walk(plain_template) if isinstance(node, ast.Name)}
        diagonal_names = {node.id for node in ast.walk(diagonal_factory) if isinstance(node, ast.Name)}

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

    @staticmethod
    def _find_size_group_loop(function_node):
        for node in ast.walk(function_node):
            if isinstance(node, ast.For) and isinstance(node.iter, ast.Attribute):
                if node.iter.attr == "_size_groups":
                    return node
        raise AssertionError("_init_tiled_kernels does not iterate over self._size_groups")

    @staticmethod
    def _references_dense_max_constraints(node):
        return any(
            isinstance(child, ast.Attribute) and child.attr == "_articulation_constraint_max"
            for child in ast.walk(node)
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


_PACKAGE_DIR = Path(__file__).parents[2] / "_src" / "solvers" / "feather_pgs"


class TestFeatherPGSPrivateApiUpstream(unittest.TestCase):
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
            if isinstance(node, ast.For) and isinstance(node.iter, ast.Attribute) and node.iter.attr == "_size_groups"
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
