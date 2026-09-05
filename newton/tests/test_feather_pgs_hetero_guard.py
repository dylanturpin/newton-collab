# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Heterogeneous multi-world support in SolverFeatherPGS.

Heterogeneous multi-world models (worlds whose per-world DOF counts differ)
are safe on matrix-free immediate, serial propagation, colored propagation,
and split paths.  The fused propagation path still uses a fixed-width
per-world velocity window and must reject heterogeneous worlds.  These tests
exercise both the constructor contract and cross-world isolation under active
contacts.
"""

from __future__ import annotations

import unittest

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherPGS

SUPPORTED_PROPAGATION_RESPONSES = ("propagation", "propagation-colored")


def _make_chain_world(n_links: int, *, base_z: float = 1.0) -> newton.ModelBuilder:
    """A fixed-base serial chain of ``n_links`` revolute links (n_links DOFs)."""
    builder = newton.ModelBuilder()
    joints = []
    prev = -1
    for i in range(n_links):
        link = builder.add_link()
        builder.add_shape_box(link, hx=0.15, hy=0.03, hz=0.03)
        if prev == -1:
            parent_xform = wp.transform(p=wp.vec3(0.0, 0.0, base_z), q=wp.quat_identity())
        else:
            parent_xform = wp.transform(p=wp.vec3(0.15, 0.0, 0.0), q=wp.quat_identity())
        joints.append(
            builder.add_joint_revolute(
                parent=prev,
                child=link,
                axis=wp.vec3(0.0, 1.0, 0.0),
                parent_xform=parent_xform,
                child_xform=wp.transform(p=wp.vec3(-0.15, 0.0, 0.0), q=wp.quat_identity()),
                label=f"chain_joint_{i}",
            )
        )
        prev = link
    builder.add_articulation(joints, label="chain")
    return builder


def _build_model(world_link_counts: list[int], device: str, *, base_z: float = 1.0) -> newton.Model:
    """One chain world per entry in ``world_link_counts``, plus a ground plane."""
    scene = newton.ModelBuilder()
    for n_links in world_link_counts:
        scene.add_world(_make_chain_world(n_links, base_z=base_z))
    scene.add_ground_plane()
    return scene.finalize(device=device)


class TestFeatherPGSHeteroGuard(unittest.TestCase):
    """Gate supported and unsupported heterogeneous-world solver paths."""

    @classmethod
    def setUpClass(cls):
        if wp.get_cuda_device_count() == 0:
            raise unittest.SkipTest("SolverFeatherPGS construction tests require cuda:0")
        cls.device = "cuda:0"
        # Worlds alternate a 1-link pendulum and a 3-link chain: per-world DOF
        # counts [1, 3, 1, 3] -> heterogeneous.
        cls.hetero_model = _build_model([1, 3, 1, 3], cls.device)
        # The first box in every chain starts 10 mm inside the ground plane,
        # ensuring the propagation row path is exercised by the isolation gate.
        cls.hetero_contact_model = _build_model([1, 3, 1, 3], cls.device, base_z=0.02)
        # All 3-link chains: per-world DOF counts uniform -> homogeneous.
        cls.homogeneous_model = _build_model([3, 3, 3, 3], cls.device)

    def test_hetero_propagation_modes_isolate_world_velocities(self):
        """Keep untouched worlds invariant when another world's velocity changes."""
        for response in SUPPORTED_PROPAGATION_RESPONSES:
            with self.subTest(articulated_contact_response=response):
                outputs = []
                propagation_rows = []
                first_world_indices = None
                other_world_indices = None
                for first_world_speed in (0.0, 3.0):
                    solver = SolverFeatherPGS(
                        self.hetero_contact_model,
                        pgs_mode="matrix_free",
                        articulated_contact_response=response,
                        pgs_iterations=4,
                        dense_max_constraints=32,
                        mf_max_constraints=32,
                    )
                    counts = solver.world_dof_count.numpy()
                    indices = solver.world_dof_indices.numpy()
                    first_world_indices = indices[0, : counts[0]]
                    other_world_indices = np.concatenate(
                        [indices[world, : counts[world]] for world in range(1, len(counts))]
                    )

                    state_in = self.hetero_contact_model.state()
                    state_out = self.hetero_contact_model.state()
                    joint_qd = state_in.joint_qd.numpy()
                    joint_qd[first_world_indices] = first_world_speed
                    state_in.joint_qd.assign(joint_qd)
                    newton.eval_fk(
                        self.hetero_contact_model,
                        state_in.joint_q,
                        state_in.joint_qd,
                        state_in,
                    )
                    pipeline = newton.CollisionPipeline(self.hetero_contact_model, broad_phase="nxn")
                    contacts = pipeline.contacts()
                    state_in.clear_forces()
                    pipeline.collide(state_in, contacts)
                    solver.step(
                        state_in,
                        state_out,
                        self.hetero_contact_model.control(),
                        contacts,
                        1.0 / 240.0,
                    )
                    outputs.append(state_out.joint_qd.numpy())
                    propagation_rows.append(int(solver.propagation_constraint_count.numpy().sum()))

                self.assertGreater(min(propagation_rows), 0)
                self.assertGreater(
                    float(np.max(np.abs(outputs[0][first_world_indices] - outputs[1][first_world_indices]))),
                    1.0e-4,
                )
                np.testing.assert_allclose(
                    outputs[0][other_world_indices],
                    outputs[1][other_world_indices],
                    rtol=0.0,
                    atol=1.0e-6,
                )

    def test_hetero_propagation_fused_raises(self):
        """Reject the fused propagation mode's fixed-width velocity window."""
        with self.assertRaises(ValueError) as ctx:
            SolverFeatherPGS(
                self.hetero_model,
                pgs_mode="matrix_free",
                articulated_contact_response="propagation-fused",
            )
        message = str(ctx.exception)
        self.assertIn("propagation-fused", message)
        self.assertIn("heterogeneous", message)
        self.assertIn("1, 3", message)

    def test_hetero_matrix_free_constructs(self):
        solver = SolverFeatherPGS(self.hetero_model, pgs_mode="matrix_free")
        self.assertEqual(solver.pgs_mode, "matrix_free")

    def test_hetero_split_constructs(self):
        solver = SolverFeatherPGS(self.hetero_model, pgs_mode="split")
        self.assertEqual(solver.pgs_mode, "split")

    def test_homogeneous_all_modes_construct(self):
        for pgs_mode, response in (
            ("split", "immediate"),
            ("matrix_free", "immediate"),
            ("matrix_free", "propagation"),
            ("matrix_free", "propagation-fused"),
            ("matrix_free", "propagation-colored"),
        ):
            with self.subTest(pgs_mode=pgs_mode, articulated_contact_response=response):
                solver = SolverFeatherPGS(
                    self.homogeneous_model,
                    pgs_mode=pgs_mode,
                    articulated_contact_response=response,
                )
                self.assertEqual(solver.pgs_mode, pgs_mode)
                self.assertEqual(solver.articulated_contact_response, response)


if __name__ == "__main__":
    unittest.main()
