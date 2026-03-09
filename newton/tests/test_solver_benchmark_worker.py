# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import contextlib
import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from newton.tools import solver_benchmark


def _make_args(**overrides):
    values = {
        "scenario": "g1_flat",
        "solver": "fpgs_tiled",
        "num_worlds": 1024,
        "substeps": 2,
        "warmup_frames": 4,
        "measure_frames": 8,
        "render_frames": 120,
        "viewer": "null",
        "pgs_iterations": 4,
        "dense_max_constraints": 32,
        "pgs_beta": 0.1,
        "pgs_cfm": 1.0e-6,
        "pgs_omega": 1.0,
        "pgs_warmstart": False,
        "pgs_mode": "hybrid",
        "delassus_chunk_size": None,
        "pgs_chunk_size": None,
        "double_buffer": True,
        "pipeline_collide": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TestSolverBenchmarkWorker(unittest.TestCase):
    def test_parser_rejects_legacy_orchestration_flags(self):
        parser = solver_benchmark._build_arg_parser()
        legacy_argv_cases = [
            ["--sweep"],
            ["--ablation"],
            ["--plot"],
            ["--replot", "results"],
            ["--compare", "a.json", "b.json"],
            ["--solvers", "mujoco"],
            ["--min-log2-worlds", "10"],
            ["--max-log2-worlds", "12"],
            ["--substeps-list", "2,4"],
            ["--ablation-pgs", "streaming"],
            ["--timing-out", "timings.jsonl"],
            ["--summary-timer"],
        ]

        for argv in legacy_argv_cases:
            with self.subTest(argv=argv):
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parser.parse_args(argv)

    def test_build_run_command_emits_concrete_worker_invocation(self):
        args = _make_args()

        command = solver_benchmark.build_run_command(args, solver_benchmark.SOLVER_PRESETS["fpgs_tiled"], 2048)

        self.assertEqual(command[1:3], ["-m", "newton.tools.solver_benchmark"])
        self.assertIn("--benchmark", command)
        self.assertIn("--num-worlds", command)
        self.assertIn("2048", command)
        self.assertIn("--double-buffer", command)
        self.assertIn("--pipeline-collide", command)
        self.assertNotIn("--sweep", command)
        self.assertNotIn("--ablation", command)
        self.assertNotIn("--plot", command)

    def test_write_benchmark_artifacts_writes_measurement_and_metadata(self):
        args = _make_args()
        measurement = solver_benchmark.build_measurement_row(
            args=args,
            elapsed_s=1.25,
            env_fps=8192.0,
            gpu_used_gb=3.5,
            gpu_total_gb=24.0,
            kernels={"pgs_solve": 4.2},
        )
        metadata = {"scenario": "g1_flat", "mode": "benchmark"}

        with TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            solver_benchmark.write_benchmark_artifacts(out_dir, measurement, metadata)

            rows = (out_dir / "measurements.jsonl").read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(rows), 1)
            self.assertEqual(json.loads(rows[0])["env_fps"], 8192.0)
            self.assertEqual(
                json.loads((out_dir / "metadata.json").read_text(encoding="utf-8")),
                metadata,
            )

    def test_write_render_artifacts_writes_metadata_and_render_manifest(self):
        args = _make_args(render_frames=300)
        render_metadata = solver_benchmark.build_render_metadata(
            args,
            render_fps=60.0,
            width=1920,
            height=1080,
            video_name="g1_flat.mp4",
        )
        metadata = {"scenario": "g1_flat", "mode": "render", "num_worlds": 1}

        with TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            solver_benchmark.write_render_artifacts(out_dir, metadata, render_metadata)

            self.assertEqual(
                json.loads((out_dir / "metadata.json").read_text(encoding="utf-8")),
                metadata,
            )
            saved_render_metadata = json.loads((out_dir / "render_meta.json").read_text(encoding="utf-8"))
            self.assertEqual(saved_render_metadata["video"], "g1_flat.mp4")
            self.assertEqual(saved_render_metadata["num_worlds"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
