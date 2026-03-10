# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmarks.nightly.nsys_trace import summarize_kernels


class TestNightlyProfileSummary(unittest.TestCase):
    def test_summarize_kernels_trims_to_measured_graph_launches(self):
        with TemporaryDirectory() as tmp_dir:
            sqlite_path = Path(tmp_dir) / "profile.sqlite"
            with sqlite3.connect(sqlite_path) as conn:
                conn.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
                conn.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME (start INTEGER, end INTEGER, nameId INTEGER)")
                conn.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INTEGER, end INTEGER, shortName INTEGER)")
                conn.executemany(
                    "INSERT INTO StringIds (id, value) VALUES (?, ?)",
                    [
                        (1, "cudaGraphLaunch_v10000"),
                        (2, "build_contact_row_kernel"),
                        (3, "pgs_solve_kernel"),
                    ],
                )
                conn.executemany(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME (start, end, nameId) VALUES (?, ?, ?)",
                    [
                        (0, 10_000_000, 1),
                        (20_000_000, 30_000_000, 1),
                        (40_000_000, 50_000_000, 1),
                    ],
                )
                conn.executemany(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL (start, end, shortName) VALUES (?, ?, ?)",
                    [
                        (2_000_000, 8_000_000, 2),
                        (22_000_000, 28_000_000, 2),
                        (24_000_000, 29_000_000, 3),
                        (44_000_000, 47_000_000, 2),
                        (45_000_000, 52_000_000, 3),
                    ],
                )

            summary = summarize_kernels(sqlite_path, measure_frames=2)

            self.assertEqual(summary["measure_frames"], 2)
            self.assertEqual(summary["window_start_ns"], 20_000_000)
            self.assertEqual(summary["window_end_ns"], 52_000_000)
            self.assertEqual(summary["kernel_event_count"], 4)
            self.assertEqual(summary["kernel_count"], 2)
            self.assertAlmostEqual(summary["kernels"]["build_contact_row_kernel"], 9.0)
            self.assertAlmostEqual(summary["kernels"]["pgs_solve_kernel"], 12.0)
            self.assertAlmostEqual(summary["total_kernel_ms"], 21.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
