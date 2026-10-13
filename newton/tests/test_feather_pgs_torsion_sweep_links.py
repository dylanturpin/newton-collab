# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Check the linked contact-torsion sweep against the full-scan sweep it accelerates."""

import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from newton._src.solvers.feather_pgs import solver_feather_pgs
from newton._src.solvers.feather_pgs.contact_torsion import _TORSION_SPIN_RETIRED
from newton._src.solvers.feather_pgs.kernels import (
    PGS_CONSTRAINT_TYPE_CONTACT,
    PGS_CONSTRAINT_TYPE_FRICTION,
    PGS_CONSTRAINT_TYPE_JOINT_LIMIT,
    PGS_CONSTRAINT_TYPE_TORSION,
)
from newton._src.solvers.feather_pgs.solver_feather_pgs import (
    _estimate_mfgs_shared_memory,
    _get_pgs_solve_mf_gs_kernel,
    _use_mfgs_torsion_links,
)
from newton.tests.test_feather_pgs_contact_torsion import fixture

ROWS = 40
DOFS = 12
RADIUS = 0.01
# Tangent rows allow more traction than the group budget, so the sweep must rescale them.
TANGENT_MU = 1.0
GROUP_MU = 0.2


class TestTorsionLinkSharedMemory(unittest.TestCase):
    """Admit the shared row lists only within the kernel's static shared-memory limit."""

    def test_estimator_counts_lists(self):
        """Count two short or int list entries per dense row."""
        for rows, entry_bytes in ((1024, 2), (8192, 2), (32767, 2), (32768, 4)):
            with self.subTest(rows=rows):
                scan = _estimate_mfgs_shared_memory(rows, 512, 32, metadata_bytes=0, torsion_links=False)
                linked = _estimate_mfgs_shared_memory(rows, 512, 32, metadata_bytes=0, torsion_links=True)
                self.assertEqual(linked - scan, 2 * entry_bytes * rows)
        # Static arrays of the shape reported to exceed sm_75's limit once the lists were added.
        self.assertEqual(_estimate_mfgs_shared_memory(8192, 512, 32, metadata_bytes=0, torsion_links=True), 67840)

    def test_selection(self):
        """Fall back to the scan when the lists, plus resident metadata, exceed the limit."""
        options = {"has_drive_rows": False, "fuse_vel_limits": False}
        device_limit = 101376
        self.assertTrue(_use_mfgs_torsion_links(1024, 1024, 30, device_limit, shared_metadata=False, **options))
        self.assertFalse(_use_mfgs_torsion_links(8192, 512, 32, device_limit, shared_metadata=False, **options))
        self.assertFalse(_use_mfgs_torsion_links(64, 16, 12, 0, shared_metadata=False, **options))
        # 128 + 4 * (16 + 8 + rows) + 4 * rows plus the 2-KiB Warp reserve is 48 KiB at rows = 5860.
        self.assertTrue(_use_mfgs_torsion_links(5860, 8, 16, device_limit, shared_metadata=False, **options))
        self.assertFalse(_use_mfgs_torsion_links(5861, 8, 16, device_limit, shared_metadata=False, **options))
        self.assertFalse(_use_mfgs_torsion_links(5860, 8, 16, 48 * 1024 - 1, shared_metadata=False, **options))
        self.assertFalse(_use_mfgs_torsion_links(2048, 0, 16, device_limit, shared_metadata=True, **options))
        self.assertTrue(_use_mfgs_torsion_links(2048, 0, 16, device_limit, shared_metadata=False, **options))


@unittest.skipUnless(wp.is_cuda_available(), "Requires CUDA")
class TestTorsionSweepLinks(unittest.TestCase):
    """Run the linked and scanning sweeps on identical inputs and require bitwise equal outputs."""

    def test_boundary_layouts_match_scan(self):
        """Match the scan bitwise for empty, first-row, last-row, over-budget, invalid and retired groups."""
        layouts = _boundary_layouts()
        inputs = _kernel_inputs(layouts)
        for shared_metadata in (True, False):
            for friction_start in (0, 3):
                results = {}
                for linked in (False, True):
                    kernel = _get_pgs_solve_mf_gs_kernel(
                        ROWS,
                        1,
                        DOFS,
                        wp.get_device("cuda:0").arch,
                        shared_metadata=shared_metadata,
                        has_drive_rows=False,
                        has_dense_velocity_limit_rows=False,
                        contact_torsion=True,
                        torsion_links=linked,
                    )
                    self.assertEqual(kernel.key.endswith("_torsion_links"), linked)
                    results[linked] = _launch(kernel, inputs, friction_start)
                with self.subTest(shared_metadata=shared_metadata, friction_start=friction_start):
                    for name in ("impulses", "v_out"):
                        np.testing.assert_array_equal(
                            results[True][name].view(np.uint32), results[False][name].view(np.uint32)
                        )
                    if friction_start == 0:
                        _assert_layouts_exercised(self, layouts, results[True]["impulses"])

    def test_solver_selects_links_and_matches_fallback(self):
        """Select the lists at a compact capacity and match the forced fallback bitwise."""
        options = {
            "friction_anchor_beta": 0.2,
            "contact_shared_anchor": False,
            "contact_friction_shared_anchor": False,
            "pgs_iterations": 8,
            "spin": 10.0,
            "sliding": 1.0,
        }
        linked, solver, *_ = fixture(0.0075, **options)
        self.assertTrue(solver._mfgs_torsion_links)
        self.assertTrue(solver._pgs_solve_mf_gs_kernel.key.endswith("_torsion_links"))
        with patch.object(solver_feather_pgs, "_use_mfgs_torsion_links", return_value=False):
            scanned, solver, *_ = fixture(0.0075, **options)
        self.assertFalse(solver._mfgs_torsion_links)
        self.assertTrue(solver._pgs_solve_mf_gs_kernel.key.endswith("_torsion"))
        self.assertTrue(np.any(linked["row_type"] == PGS_CONSTRAINT_TYPE_TORSION))
        for name, value in linked.items():
            np.testing.assert_array_equal(np.asarray(value), np.asarray(scanned[name]), err_msg=name)

    def test_largest_admitted_capacity_compiles(self):
        """Compile the linked kernel at the largest capacity the selection admits."""
        options = {"has_drive_rows": False, "fuse_vel_limits": False}
        device = wp.get_device("cuda:0")
        limit = device.max_shared_memory_per_block
        self.assertTrue(_use_mfgs_torsion_links(5860, 8, 16, limit, shared_metadata=False, **options))
        kernel = _get_pgs_solve_mf_gs_kernel(
            5860,
            8,
            16,
            device.arch,
            shared_metadata=False,
            has_drive_rows=False,
            has_dense_velocity_limit_rows=False,
            contact_torsion=True,
            torsion_links=True,
        )
        wp.load_module(kernel.module, device=device)

    def test_large_capacity_falls_back(self):
        """Fall back to the scan when the lists exceed the static limit, and still solve torsion."""
        result, solver, *_ = fixture(
            0.0075, dense_max_constraints=8192, mf_max_constraints=512, spin=10.0, pgs_iterations=8
        )
        self.assertFalse(solver._mfgs_torsion_links)
        self.assertTrue(solver._pgs_solve_mf_gs_kernel.key.endswith("_torsion"))
        count = int(result["count"][0])
        torsion_rows = result["row_type"][0, :count] == PGS_CONSTRAINT_TYPE_TORSION
        self.assertTrue(np.any(torsion_rows))
        self.assertTrue(np.any(result["impulses"][0, :count][torsion_rows] != 0.0))
        for name in ("body_q", "body_qd", "impulses"):
            self.assertTrue(np.all(np.isfinite(result[name])), name)


def _boundary_layouts():
    """Return per-world (row types, parents, torsion groups, active rows, loaded torsion rows)."""

    def world(active):
        return (
            np.full(ROWS, PGS_CONSTRAINT_TYPE_JOINT_LIMIT, np.int32),
            np.full(ROWS, -1, np.int32),
            np.full(ROWS, -1, np.int32),
            active,
        )

    def contact(types, parents, groups, row, spin, tangents=True):
        types[row] = PGS_CONSTRAINT_TYPE_CONTACT
        groups[row] = spin
        if tangents:
            types[row + 1 : row + 3] = PGS_CONSTRAINT_TYPE_FRICTION
            parents[row + 1 : row + 3] = row

    def torsion(types, groups, row, retired=False):
        types[row] = PGS_CONSTRAINT_TYPE_TORSION
        if retired:
            groups[row] = _TORSION_SPIN_RETIRED

    layouts = []
    # No torsion rows.
    types, parents, groups, active = world(ROWS)
    for row in range(0, 36, 3):
        contact(types, parents, groups, row, -1)
    layouts.append((types, parents, groups, active, []))
    # Torsion row first, every member after it.
    types, parents, groups, active = world(ROWS)
    torsion(types, groups, 0)
    for row in range(1, 37, 3):
        contact(types, parents, groups, row, 0)
    layouts.append((types, parents, groups, active, [0]))
    # Torsion row last at full capacity, after a normal-only member.
    types, parents, groups, active = world(ROWS)
    for row in range(0, 36, 3):
        contact(types, parents, groups, row, ROWS - 1)
    contact(types, parents, groups, 37, ROWS - 1, tangents=False)
    torsion(types, groups, ROWS - 1)
    layouts.append((types, parents, groups, active, [ROWS - 1]))
    # Interleaved groups, empty and retired groups, invalid targets, and normal-only members on the
    # last active rows; inactive rows hold a torsion group that both sweeps must ignore.
    types, parents, groups, active = world(31)
    torsion(types, groups, 0)
    torsion(types, groups, 1)
    torsion(types, groups, 2)
    torsion(types, groups, 3, retired=True)
    targets = [0, 1, 0, 1, -1, 3, 4, 35]
    for row, spin in zip(range(4, 28, 3), targets, strict=True):
        contact(types, parents, groups, row, spin)
    contact(types, parents, groups, 28, 1, tangents=False)
    contact(types, parents, groups, 29, 0, tangents=False)
    contact(types, parents, groups, 30, 1, tangents=False)
    torsion(types, groups, 34)
    contact(types, parents, groups, 35, 34)
    layouts.append((types, parents, groups, active, [0, 1]))
    return layouts


def _kernel_inputs(layouts):
    rng = np.random.default_rng(7)
    worlds = len(layouts)
    types = np.stack([layout[0] for layout in layouts])
    parents = np.stack([layout[1] for layout in layouts])
    groups = np.stack([layout[2] for layout in layouts])
    counts = np.array([layout[3] for layout in layouts], np.int32)
    J = rng.normal(size=(worlds, ROWS, DOFS)).astype(np.float32)
    Y = (0.5 * J).astype(np.float32)
    diag = np.einsum("wrd,wrd->wr", J, Y).astype(np.float32) + np.float32(1.0e-3)
    rhs = rng.normal(scale=0.5, size=(worlds, ROWS)).astype(np.float32)
    # Approaching normals load the groups; strong tangential slip saturates their tangent rows.
    rhs[types == PGS_CONSTRAINT_TYPE_CONTACT] = -np.abs(rhs[types == PGS_CONSTRAINT_TYPE_CONTACT]) - 0.5
    rhs[types == PGS_CONSTRAINT_TYPE_FRICTION] *= 8.0
    mu = np.zeros((worlds, ROWS), np.float32)
    mu[types == PGS_CONSTRAINT_TYPE_FRICTION] = TANGENT_MU
    mu[types == PGS_CONSTRAINT_TYPE_TORSION] = GROUP_MU
    return {
        "worlds": worlds,
        "counts": counts,
        "types": types,
        "parents": parents,
        "groups": groups,
        "J": J,
        "Y": Y,
        "diag": diag,
        "rhs": rhs,
        "mu": mu,
        "v": rng.normal(size=worlds * DOFS).astype(np.float32),
    }


def _launch(kernel, inputs, friction_start):
    worlds = inputs["worlds"]

    def arr(value, dtype=float):
        return wp.array(value, dtype=dtype, device="cuda:0")

    def zeros(shape, dtype=float):
        return wp.zeros(shape, dtype=dtype, device="cuda:0")

    impulses = zeros((worlds, ROWS))
    v_out = arr(inputs["v"])
    dense = (worlds, ROWS)
    wp.launch_tiled(
        kernel,
        dim=[worlds],
        inputs=[
            zeros(1, int),
            zeros(1, int),
            worlds,
            0,
            arr(inputs["counts"], int),
            zeros((worlds, 2), int),
            zeros(worlds, int),
            arr(np.arange(worlds * DOFS, dtype=np.int32).reshape(worlds, DOFS), int),
            arr(inputs["rhs"]),
            arr(inputs["diag"]),
            arr(np.ones(dense, np.float32)),
            impulses,
            arr(inputs["J"]),
            arr(inputs["Y"]),
            arr(inputs["types"], int),
            arr(inputs["parents"], int),
            arr(inputs["mu"]),
            arr(inputs["groups"], int),
            RADIUS,
            zeros(dense),
            zeros(dense),
            zeros(dense),
            zeros(dense),
            zeros((1, 1)),
            zeros(worlds, int),
            zeros(worlds, int),
            zeros((worlds, 4), int),
            zeros((worlds, 1)),
            zeros((worlds, 1, 6)),
            zeros((worlds, 1, 6)),
            zeros((worlds, 1, 6)),
            zeros((worlds, 1, 6)),
            zeros((worlds, 1)),
            zeros((worlds, 1)),
            zeros((1, 1)),
            zeros((1, 1)),
            6,
            1.0,
            0,
            0,
            friction_start,
            0,
            0,
        ],
        outputs=[v_out],
        block_dim=32,
        device="cuda:0",
    )
    return {"impulses": impulses.numpy(), "v_out": v_out.numpy()}


def _assert_layouts_exercised(test, layouts, impulses):
    """Require loaded groups within budget, a rescaled group, a spin impulse and idle retired rows."""
    rescaled = spun = False
    for world, (types, parents, groups, active, spins) in enumerate(layouts):
        lam = impulses[world]
        torsion_rows = [row for row in range(active) if types[row] == PGS_CONSTRAINT_TYPE_TORSION]
        if not spins:
            test.assertFalse(np.any(lam[torsion_rows]))
        for spin in spins:
            members = [n for n in range(active) if types[n] == PGS_CONSTRAINT_TYPE_CONTACT and groups[n] == spin]
            load = sum(max(float(lam[n]), 0.0) for n in members)
            sliding = sum(
                float(np.hypot(lam[n + 1], lam[n + 2]))
                for n in members
                if n + 2 < active and parents[n + 1] == n and parents[n + 2] == n
            )
            budget = GROUP_MU * load
            test.assertGreater(load, 0.0)
            test.assertLessEqual(sliding + abs(float(lam[spin])) / RADIUS, budget * (1.0 + 1.0e-5) + 1.0e-6)
            rescaled |= sliding > 0.99 * budget
            spun |= abs(float(lam[spin])) / RADIUS > 0.5 * budget
        retired = [row for row in torsion_rows if groups[row] == _TORSION_SPIN_RETIRED]
        test.assertFalse(np.any(lam[retired]))
    test.assertTrue(rescaled, "No group exceeded its friction budget")
    test.assertTrue(spun, "No group carried a spin impulse")


if __name__ == "__main__":
    unittest.main()
