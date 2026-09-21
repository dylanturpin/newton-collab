# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exercise direct contact ownership without changing the sparse response law."""

import ast
import inspect
import textwrap
import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import kernels
from newton._src.solvers.feather_pgs import solver_feather_pgs as implementation
from newton._src.solvers.feather_pgs.friction_patches import FrictionPatches


def _launch_workers(capacity):
    """Evaluate the actual sparse launch's host-only worker selection."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(implementation.SolverFeatherPGS._stage4_build_rows)))
    call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and node.args
        and isinstance(node.args[0], ast.Name)
        and node.args[0].id == "populate_sparse_diagonal_contact_response"
    )
    dimension = next(item.value for item in call.keywords if item.arg == "dim")
    arguments = next(item.value for item in call.keywords if item.arg == "inputs")
    if ast.dump(dimension) != ast.dump(arguments.elts[1]):
        raise AssertionError("launch dimension and device stride must agree")
    values = {
        "contacts": SimpleNamespace(rigid_contact_max=capacity),
        "contact_jacobian_workers": min(capacity, implementation._CONTACT_JACOBIAN_WORKER_CAP),
    }
    return eval(compile(ast.Expression(dimension), "<actual sparse worker selection>", "eval"), {}, values)


def _fixture(device, capacity=4353, *, shared_anchor=0, friction_shared_anchor=1, patches_enabled=0):
    """Create unique row slots and mixed endpoint ownership above 4096 contacts."""
    indices = np.arange(capacity, dtype=np.int32)
    world_count = (capacity + 4) // 5
    world = indices // 5
    slot = 3 * (indices % 5)
    path = np.where(indices % 11 == 0, -1, 0).astype(np.int32)
    slot[indices % 13 == 0] = -1
    shape1 = (indices % 5 - 1).astype(np.int32)
    art1 = np.where(shape1 < 0, -1, np.where(shape1 == 2, 1, 0)).astype(np.int32)
    body_q = np.array(
        (
            (0.1, 0.2, 0.3, 0, 0, 0, 1),
            (0.2, -0.1, 0.4, 0, 0, 0, 1),
            (0, 0, 0, 0, 0, 0, 1),
            (0, 0, 0, 0, 0, 0, 1),
        ),
        dtype=np.float32,
    )
    motion = np.zeros((114, 6), dtype=np.float32)
    motion[0] = (0.2, 0.3, 1.0, 0.1, -0.2, 0.3)
    motion[1] = (-0.4, 0.5, 0.7, -0.2, 0.1, 0.4)
    normal = np.tile(np.array((0.0, 0.0, -1.0), dtype=np.float32), (capacity, 1))
    point0 = np.tile(np.array((0.1, -0.2, 0.3), dtype=np.float32), (capacity, 1))
    point1 = np.tile(np.array((-0.2, 0.1, 0.4), dtype=np.float32), (capacity, 1))

    def array(value, dtype):
        return wp.array(value, dtype=dtype, device=device)

    patches = FrictionPatches()
    patches.enabled = patches_enabled
    patches.weight = wp.ones(capacity, dtype=float, device=device)
    patches.next_contact = wp.full(capacity, -1, dtype=int, device=device)
    # Distinct world-space patch anchors must override the ordinary contact midpoint.
    patches.point_a = array(np.tile((0.7, -0.4, 0.2), (capacity, 1)), wp.vec3)
    patches.point_b = array(np.tile((-0.3, 0.6, 0.9), (capacity, 1)), wp.vec3)
    patches.phi = wp.zeros(capacity, dtype=wp.vec2, device=device)
    return {
        "contact_count": array([capacity], int),
        "contact_point0": array(point0, wp.vec3),
        "contact_point1": array(point1, wp.vec3),
        "contact_normal": array(normal, wp.vec3),
        "contact_shape0": array(np.zeros(capacity, dtype=np.int32), int),
        "contact_shape1": array(shape1, int),
        "contact_thickness0": array(np.full(capacity, 0.01, dtype=np.float32), float),
        "contact_thickness1": array(np.full(capacity, 0.02, dtype=np.float32), float),
        "contact_world": array(world, int),
        "contact_slot": array(slot, int),
        "contact_art_a": array(np.zeros(capacity, dtype=np.int32), int),
        "contact_art_b": array(art1, int),
        "contact_path": array(path, int),
        "contact_slots_needed": array(1 + indices % 3, int),
        "target_size": 108,
        "articulation_response_dof_count": array([108, 6], int),
        "articulation_dof_start": array([0, 108], int),
        "articulation_world_dof_offset": array([6, 0], int),
        "articulation_origin": array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], wp.vec3),
        "body_single_response_dof": array([0, 1, -1, -1], int),
        "diagonal_inverse_mass": array(np.linspace(0.25, 1.0, 114, dtype=np.float32), float),
        "joint_S_s": array(motion, wp.spatial_vector),
        "shape_body": array([0, 1, 2, 3], int),
        "body_q": array(body_q, wp.transform),
        "contact_friction_shared_anchor": friction_shared_anchor,
        "friction_patches": patches,
        "contact_shared_anchor": shared_anchor,
        "sparse_row_dof": wp.full((world_count, 15, 2), -777, dtype=int, device=device),
        "sparse_row_jy": wp.full((world_count, 15, 4), np.nan, dtype=float, device=device),
    }


def _launch(values, workers, device):
    """Launch the unchanged kernel with a capped or complete-prefix owner set."""
    values = {**values, "total_num_workers": workers}
    arguments = [
        values[name] for name in inspect.signature(kernels.populate_sparse_diagonal_contact_response.func).parameters
    ]
    wp.launch(kernels.populate_sparse_diagonal_contact_response, dim=workers, inputs=arguments, device=device)


def _expected(values, active_count):
    """Evaluate the sparse physical response independently in FP64."""
    host = {key: value.numpy() if isinstance(value, wp.array) else value for key, value in values.items()}
    patches = host["friction_patches"]
    patch_points = (patches.point_a.numpy(), patches.point_b.numpy())
    dofs = np.full(host["sparse_row_dof"].shape, -777, dtype=np.int32)
    response = np.full(host["sparse_row_jy"].shape, np.nan, dtype=np.float32)
    count = min(active_count, len(host["contact_slot"]))
    for contact in range(count):
        slot = host["contact_slot"][contact]
        if slot < 0 or host["contact_path"][contact] != 0:
            continue
        normal = -host["contact_normal"][contact].astype(np.float64)
        tangent0 = np.cross(normal, (1.0, 0.0, 0.0))
        tangent0 /= np.linalg.norm(tangent0)
        tangent1 = np.cross(normal, tangent0)
        tangent1 /= np.linalg.norm(tangent1)
        shapes = (host["contact_shape0"][contact], host["contact_shape1"][contact])
        bodies = tuple(host["shape_body"][shape] if shape >= 0 else -1 for shape in shapes)
        points = [host[f"contact_point{side}"][contact].astype(np.float64).copy() for side in range(2)]
        for side in range(2):
            if bodies[side] >= 0:
                points[side] += host["body_q"][bodies[side], :3]
            points[side] += (-1.0 if side == 0 else 1.0) * host[f"contact_thickness{side}"][contact] * normal
        anchor = 0.5 * (points[0] + points[1])
        for row, direction in enumerate((normal, tangent0, tangent1)):
            if row >= host["contact_slots_needed"][contact]:
                continue
            coordinates = [-1, -1]
            coefficients = [0.0, 0.0]
            actions = [0.0, 0.0]
            for side in range(2):
                body = bodies[side]
                art = host["contact_art_a" if side == 0 else "contact_art_b"][contact]
                if body < 0 or art < 0 or host["articulation_response_dof_count"][art] != host["target_size"]:
                    continue
                dof = host["body_single_response_dof"][body]
                if dof < 0:
                    continue
                point = points[side]
                if host["contact_shared_anchor"] or (row > 0 and host["contact_friction_shared_anchor"]):
                    point = anchor
                if row > 0 and patches.enabled:
                    point = patch_points[side][contact].astype(np.float64)
                motion = host["joint_S_s"][dof].astype(np.float64)
                velocity = motion[:3] + np.cross(motion[3:], point - host["articulation_origin"][art])
                coefficients[side] = (1.0 if side == 0 else -1.0) * np.dot(direction, velocity)
                actions[side] = coefficients[side] * host["diagonal_inverse_mass"][dof]
                coordinates[side] = (
                    host["articulation_world_dof_offset"][art] + dof - host["articulation_dof_start"][art]
                )
            if coordinates[0] >= 0 and coordinates[0] == coordinates[1]:
                coefficients[0] += coefficients[1]
                actions[0] += actions[1]
                coordinates[1], coefficients[1], actions[1] = -1, 0.0, 0.0
            world = host["contact_world"][contact]
            dofs[world, slot + row] = coordinates
            response[world, slot + row] = (coefficients[0], actions[0], coefficients[1], actions[1])
    return dofs, response


def _assert_response(test, values, expected):
    """Check the physical response and untouched sentinel rows."""
    actual = (values["sparse_row_dof"].numpy(), values["sparse_row_jy"].numpy())
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_allclose(actual[1], expected[1], rtol=2e-6, atol=2e-7)
    test.assertTrue(np.isfinite(actual[1][actual[0][:, :, 0] >= 0]).all())
    return actual


class TestSparseResponseOwner(unittest.TestCase):
    def test_actual_host_launch_selection(self):
        """Give every scalar contact an owner instead of inheriting the warp cap."""
        for capacity in (0, 1, 4096, 4353, 147456):
            with self.subTest(capacity=capacity):
                self.assertEqual(_launch_workers(capacity), capacity)

    def test_cpu_mixed_complete_prefix(self):
        """Preserve both anchor laws and all mixed responses above the old cap."""
        for patches_enabled in (0, 1):
            for shared_anchor in (0, 1):
                for friction_shared_anchor in (0, 1):
                    with self.subTest(
                        patches=patches_enabled, shared=shared_anchor, friction_shared=friction_shared_anchor
                    ):
                        values = _fixture(
                            "cpu",
                            patches_enabled=patches_enabled,
                            shared_anchor=shared_anchor,
                            friction_shared_anchor=friction_shared_anchor,
                        )
                        count = len(values["contact_slot"])
                        expected = _expected(values, count)
                        capped = None
                        for workers in (min(count, 4096), count):
                            values["sparse_row_dof"].fill_(-777)
                            values["sparse_row_jy"].fill_(np.nan)
                            _launch(values, workers, "cpu")
                            actual = _assert_response(self, values, expected)
                            if capped is not None:
                                np.testing.assert_array_equal(actual[0], capped[0])
                                np.testing.assert_array_equal(actual[1], capped[1])
                            capped = actual

    def test_cpu_empty_and_overflow_prefix(self):
        """Retain empty and bounded-overflow behavior without exceeding storage."""
        values = _fixture("cpu", capacity=17, patches_enabled=1)
        for raw_count in (0, 18, 17):
            values["contact_count"].assign(np.array([raw_count], dtype=np.int32))
            values["sparse_row_dof"].fill_(-777)
            values["sparse_row_jy"].fill_(np.nan)
            _launch(values, 17, "cpu")
            _assert_response(self, values, _expected(values, raw_count))
        empty = _fixture("cpu", capacity=0, patches_enabled=1)
        _launch(empty, 0, "cpu")
        self.assertEqual(empty["sparse_row_dof"].numpy().size, 0)

    @unittest.skipUnless(wp.is_cuda_available(), "actual CUDA graph replay requires CUDA")
    def test_cuda_graph_actual_prefix_transitions(self):
        """Replay both owner graphs over full, empty, truncated and overflow prefixes."""
        device = wp.get_device("cuda:0")
        values = _fixture(device, patches_enabled=1, shared_anchor=1)
        capacity = len(values["contact_slot"])
        owners = (values["sparse_row_dof"].ptr, values["sparse_row_jy"].ptr)
        for workers in (min(capacity, 4096), _launch_workers(capacity)):
            _launch(values, workers, device)
            with wp.ScopedCapture(device=device) as capture:
                _launch(values, workers, device)
            for raw_count in (capacity, 0, 4097, capacity + 1, capacity):
                values["contact_count"].assign(np.array([raw_count], dtype=np.int32))
                expected = _expected(values, raw_count)
                for graph in (False, True, True):
                    values["sparse_row_dof"].fill_(-777)
                    values["sparse_row_jy"].fill_(np.nan)
                    if graph:
                        wp.capture_launch(capture.graph)
                    else:
                        _launch(values, workers, device)
                    _assert_response(self, values, expected)
                self.assertEqual(owners, (values["sparse_row_dof"].ptr, values["sparse_row_jy"].ptr))


if __name__ == "__main__":
    unittest.main()
