# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression wrapper for the prismatic sphere/cube/table contact probe."""

import unittest

import numpy as np
import warp as wp

from newton.examples.diagnostics.example_prismatic_sphere_cube_press import (
    BAD_FRAME_CUBE_HALF,
    BAD_FRAME_CUBE_TO_FINGER_NORMAL,
    CONTACT_PROXY_SCENARIOS,
    LATEST_BAD_FRAME_CUBE_TO_LEFTFINGER_NORMAL,
    LATEST_BAD_FRAME_CUBE_TO_LEFTFINGER_POINTS,
    SCENARIOS,
    initial_contact_report,
    row_problem_report,
    run_all,
    run_scenario,
)


LATEST_CAPTURED_LEFTFINGER_ROW_SUMMARY = {
    "dof_count": 15,
    "dense_count": 66,
    "drive_rows": 9,
    "contact_diag": np.array([9.85766887664795, 14.213494300842285, 14.217225074768066]),
    "contact_j_other_norm": np.array([0.6812865622006786, 0.6644039640376553, 0.664398035878512]),
    "contact_j_cube": np.array(
        [
            [
                0.9997262358665466,
                -1.4075219041842502e-05,
                -0.023397525772452354,
                -0.0002613405813463032,
                0.012656403705477715,
                -0.011174137704074383,
            ],
            [
                0.9997262358665466,
                -1.4075219041842502e-05,
                -0.023397525772452354,
                -0.0002426105347694829,
                0.023431912064552307,
                -0.01038032490760088,
            ],
            [
                0.9997262358665466,
                -1.4075219041842502e-05,
                -0.023397525772452354,
                -0.00024278639466501772,
                0.02343541942536831,
                -0.010387841612100601,
            ],
        ],
        dtype=np.float64,
    ),
}


def _scenario(name: str):
    return next(s for s in SCENARIOS if s.name == name)


def _proxy_scenario(name: str):
    return next(s for s in CONTACT_PROXY_SCENARIOS if s.name == name)


class TestFeatherPGSPrismaticSphereCube(unittest.TestCase):
    def test_augmented_drive_contact_probe(self):
        if not wp.is_cuda_available():
            self.skipTest("FeatherPGS matrix-free contact probe is exercised on CUDA only")
        rows = run_all(
            device=wp.get_device("cuda:0"),
            assert_assumptions=True,
            drive_mode="augmented",
            contact_matching="sticky",
        )
        self.assertEqual(len(rows), len(SCENARIOS))

    def test_bad_frame_corner_contact_reproduction(self):
        if not wp.is_cuda_available():
            self.skipTest("FeatherPGS matrix-free contact probe is exercised on CUDA only")

        report = initial_contact_report(
            _scenario("bad_frame_corner"),
            device=wp.get_device("cuda:0"),
            contact_matching="sticky",
        )
        self.assertEqual(report["initial_cube_table_contacts"], 4)
        self.assertEqual(report["initial_cube_pusher_contacts"], 1)

        contact = report["cube_to_pusher_contacts"][0]
        normal = np.array(contact["normal"], dtype=np.float64)
        expected = np.array(BAD_FRAME_CUBE_TO_FINGER_NORMAL, dtype=np.float64)
        expected /= np.linalg.norm(expected)
        self.assertGreater(float(normal @ expected), 0.9999)

        cube_point = np.array(contact["point_a"], dtype=np.float64)
        expected_corner = np.array([-BAD_FRAME_CUBE_HALF, BAD_FRAME_CUBE_HALF, BAD_FRAME_CUBE_HALF])
        np.testing.assert_allclose(cube_point, expected_corner, atol=2.0e-3, rtol=0.0)

    def test_latest_leftfinger_face_contact_proxy(self):
        if not wp.is_cuda_available():
            self.skipTest("FeatherPGS matrix-free contact probe is exercised on CUDA only")

        report = initial_contact_report(
            _proxy_scenario("latest_leftfinger_face"),
            device=wp.get_device("cuda:0"),
            contact_matching="sticky",
        )
        self.assertEqual(report["initial_cube_table_contacts"], 4)
        self.assertEqual(report["initial_cube_pusher_contacts"], 3)

        contacts = report["cube_to_pusher_contacts"]
        normals = np.array([c["normal"] for c in contacts], dtype=np.float64)
        expected_normal = np.array(LATEST_BAD_FRAME_CUBE_TO_LEFTFINGER_NORMAL, dtype=np.float64)
        expected_normal /= np.linalg.norm(expected_normal)
        self.assertGreater(float(np.min(normals @ expected_normal)), 0.9995)

        cube_points = np.array([c["point_a"] for c in contacts], dtype=np.float64)
        self.assertLess(float(np.max(np.abs(cube_points[:, 0] + BAD_FRAME_CUBE_HALF))), 1.0e-4)
        target_points = np.array(LATEST_BAD_FRAME_CUBE_TO_LEFTFINGER_POINTS, dtype=np.float64)
        low_side_error = float(np.min(np.linalg.norm(cube_points - target_points[0], axis=1)))
        top_edge_error = float(np.min(np.linalg.norm(cube_points - target_points[1], axis=1)))
        self.assertLess(low_side_error, 7.5e-4)
        self.assertLess(top_edge_error, 7.5e-4)

    def test_latest_leftfinger_face_row_problem_summary(self):
        if not wp.is_cuda_available():
            self.skipTest("FeatherPGS matrix-free contact probe is exercised on CUDA only")

        report = row_problem_report(
            _proxy_scenario("latest_leftfinger_face"),
            device=wp.get_device("cuda:0"),
            drive_mode="physx_pgs",
            contact_matching="sticky",
        )
        self.assertEqual(report["dense_type_counts"]["contact"], 3)
        self.assertEqual(report["dense_type_counts"]["friction"], 6)

        # The cube-side Jacobian block is close because the proxy matches the
        # local cube anchors. The rest of the row problem is deliberately not
        # equivalent to the captured Franka frame.
        proxy_j_cube = np.array([row["j_cube"] for row in report["dense_contact_rows"]], dtype=np.float64)
        cube_block_errors = [
            float(np.min(np.linalg.norm(proxy_j_cube - target[None, :], axis=1)))
            for target in LATEST_CAPTURED_LEFTFINGER_ROW_SUMMARY["contact_j_cube"]
        ]
        self.assertLess(max(cube_block_errors), 3.0e-2)

        proxy_diag = np.array([row["diag"] for row in report["dense_contact_rows"]], dtype=np.float64)
        ref_diag = LATEST_CAPTURED_LEFTFINGER_ROW_SUMMARY["contact_diag"]
        self.assertGreater(float(np.min(np.abs(proxy_diag[:, None] - ref_diag[None, :]))), 3.0)
        self.assertNotEqual(report["dof_count"], LATEST_CAPTURED_LEFTFINGER_ROW_SUMMARY["dof_count"])
        self.assertNotEqual(len(report["dense_drive_rows"]), LATEST_CAPTURED_LEFTFINGER_ROW_SUMMARY["drive_rows"])

    def test_bad_frame_corner_drive_force_proxy(self):
        if not wp.is_cuda_available():
            self.skipTest("FeatherPGS matrix-free contact probe is exercised on CUDA only")

        metrics = run_scenario(
            _scenario("bad_frame_corner"),
            device=wp.get_device("cuda:0"),
            assert_assumptions=True,
            drive_mode="physx_pgs",
            contact_matching="sticky",
            target_overlap=0.08,
            effort_limit=60.0,
            joint_limit_upper=0.16,
        )
        self.assertGreater(metrics["peak_sphere_cube_force_n"], 35.0)
        self.assertLess(metrics["peak_sphere_cube_force_n"], 80.0)
        self.assertLess(metrics["peak_cube_omega_radps"], 2.5)
        self.assertLess(metrics["final_drive_error_m"], 0.04)


if __name__ == "__main__":
    unittest.main(verbosity=2)
