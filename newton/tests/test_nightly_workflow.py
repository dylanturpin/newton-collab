# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmarks.nightly.local import run_local_nightly
from benchmarks.nightly.plan import DEFAULT_PLAN_PATH


class FakeWorkerRunner:
    """Small fake subprocess runner for end-to-end workflow tests."""

    def __call__(self, command: list[str], cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
        out_dir = Path(command[command.index("--out") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)

        if "--benchmark" in command:
            measurement = {
                "solver": command[command.index("--solver") + 1],
                "substeps": int(command[command.index("--substeps") + 1]),
                "num_worlds": int(command[command.index("--num-worlds") + 1]),
                "env_fps": 512.0,
                "gpu_used_gb": 1.5,
                "gpu_total_gb": 24.0,
                "elapsed_s": 0.5,
                "ok": True,
                "kernels": {},
            }
            metadata = {
                "scenario": command[command.index("--scenario") + 1],
                "timestamp": "2026-03-08T12:00:00Z",
                "gpu": "Workflow GPU",
                "gpu_memory_total_gb": 24.0,
                "platform": "Linux",
                "python_version": "3.12.0",
            }
            (out_dir / "measurements.jsonl").write_text(json.dumps(measurement) + "\n", encoding="utf-8")
            (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout="benchmark ok\n", stderr="")

        scenario = command[command.index("--scenario") + 1]
        metadata = {
            "scenario": scenario,
            "timestamp": "2026-03-08T12:00:00Z",
            "gpu": "Workflow GPU",
            "gpu_memory_total_gb": 24.0,
            "platform": "Linux",
            "python_version": "3.12.0",
        }
        render_meta = {
            "scenario": scenario,
            "video": f"{scenario}.mp4",
            "num_worlds": 1,
            "label": "Workflow render",
            "fps": 60,
        }
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        (out_dir / "render_meta.json").write_text(json.dumps(render_meta, indent=2) + "\n", encoding="utf-8")
        (out_dir / f"{scenario}.mp4").write_bytes(b"fake mp4")
        return subprocess.CompletedProcess(command, 0, stdout="render ok\n", stderr="")


class TestNightlyWorkflow(unittest.TestCase):
    def test_validation_run_creates_expected_artifact_tree_without_publish(self):
        with TemporaryDirectory() as tmp_dir:
            summary = run_local_nightly(
                plan_path=DEFAULT_PLAN_PATH,
                run_mode="validation",
                run_id="workflow-validation",
                shared_state_dir=str(Path(tmp_dir) / "shared"),
                work_base_dir=str(Path(tmp_dir) / "work"),
                cache_env_overrides={
                    "TMPDIR": str(Path(tmp_dir) / "tmp"),
                    "UV_CACHE_DIR": str(Path(tmp_dir) / "uv-cache"),
                    "UV_PROJECT_ENVIRONMENT": str(Path(tmp_dir) / "uv-env"),
                    "WARP_CACHE_PATH": str(Path(tmp_dir) / "warp-cache"),
                },
                publish=False,
                working_dir=Path(tmp_dir),
                runner=FakeWorkerRunner(),
            )

            run_dir = Path(summary["run_dir"])
            self.assertEqual(summary["publish"]["status"], "skipped")
            self.assertTrue(summary["artifacts_verified"])
            self.assertEqual(summary["selected_task_ids"], ["validation_g1_flat_sweep", "validation_g1_flat_render"])

            self.assertTrue((run_dir / "run.json").is_file())
            self.assertTrue((run_dir / "plan.lock.yaml").is_file())
            self.assertTrue((run_dir / "publish" / "summary.json").is_file())

            benchmark_task_dir = run_dir / "tasks" / "validation_g1_flat_sweep"
            render_task_dir = run_dir / "tasks" / "validation_g1_flat_render"
            for task_dir in (benchmark_task_dir, render_task_dir):
                self.assertTrue((task_dir / "task.json").is_file())
                self.assertTrue((task_dir / "status.json").is_file())
                self.assertTrue((task_dir / "stdout.log").is_file())

            benchmark_jobs = sorted((benchmark_task_dir / "jobs").iterdir())
            self.assertEqual(len(benchmark_jobs), 4)
            for job_dir in benchmark_jobs:
                self.assertTrue((job_dir / "job.json").is_file())
                self.assertTrue((job_dir / "status.json").is_file())
                self.assertTrue((job_dir / "stdout.log").is_file())
                self.assertTrue((job_dir / "stderr.log").is_file())
                self.assertTrue((job_dir / "results" / "measurements.jsonl").is_file())
                self.assertTrue((job_dir / "results" / "metadata.json").is_file())

            render_job_dir = render_task_dir / "jobs" / "validation_g1_flat_render__0001"
            self.assertTrue((render_job_dir / "job.json").is_file())
            self.assertTrue((render_job_dir / "status.json").is_file())
            self.assertTrue((render_job_dir / "results" / "render_meta.json").is_file())
            self.assertTrue((render_job_dir / "results" / "g1_flat.mp4").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
