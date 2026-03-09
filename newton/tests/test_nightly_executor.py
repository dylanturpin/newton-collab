# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmarks.nightly.common import resolve_run_paths, validate_run_environment
from benchmarks.nightly.executor import run_task
from benchmarks.nightly.plan import DEFAULT_PLAN_PATH, expand_plan, load_plan


class FakeWorkerRunner:
    """Small fake subprocess runner for executor tests."""

    def __init__(self, *, failed_job_ids: set[str] | None = None):
        self.failed_job_ids = set(failed_job_ids or set())
        self.commands: list[list[str]] = []
        self.envs: list[dict[str, str]] = []

    def __call__(self, command: list[str], cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
        self.commands.append(command)
        self.envs.append(dict(env))

        out_dir = Path(command[command.index("--out") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        job_id = out_dir.parent.name

        if job_id in self.failed_job_ids:
            return subprocess.CompletedProcess(command, 17, stdout=f"{job_id} failed\n", stderr="worker failure\n")

        if "--benchmark" in command:
            measurement = {
                "solver": command[command.index("--solver") + 1],
                "substeps": int(command[command.index("--substeps") + 1]),
                "num_worlds": int(command[command.index("--num-worlds") + 1]),
                "env_fps": 4096.0,
                "gpu_used_gb": 2.5,
                "gpu_total_gb": 24.0,
                "elapsed_s": 1.0,
                "ok": True,
                "kernels": {},
            }
            metadata = {"scenario": command[command.index("--scenario") + 1], "mode": "benchmark"}
            (out_dir / "measurements.jsonl").write_text(json.dumps(measurement) + "\n", encoding="utf-8")
            (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            return subprocess.CompletedProcess(command, 0, stdout=f"{job_id} completed\n", stderr="")

        if "--render" in command:
            scenario = command[command.index("--scenario") + 1]
            metadata = {"scenario": scenario, "mode": "render", "num_worlds": 1}
            render_meta = {"scenario": scenario, "video": f"{scenario}.mp4", "num_worlds": 1}
            (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            (out_dir / "render_meta.json").write_text(json.dumps(render_meta, indent=2) + "\n", encoding="utf-8")
            (out_dir / f"{scenario}.mp4").write_bytes(b"fake mp4")
            return subprocess.CompletedProcess(command, 0, stdout=f"{job_id} rendered\n", stderr="")

        raise AssertionError(f"Unexpected command: {command}")


class TestNightlyExecutor(unittest.TestCase):
    def _make_run_paths(self, tmp_dir: str):
        return resolve_run_paths(
            run_id="2026-03-08T20-30-15Z",
            shared_state_dir=Path(tmp_dir) / "shared",
            work_base_dir=Path(tmp_dir) / "work",
            cache_env={
                "TMPDIR": str(Path(tmp_dir) / "tmp"),
                "UV_CACHE_DIR": str(Path(tmp_dir) / "uv-cache"),
                "UV_PROJECT_ENVIRONMENT": str(Path(tmp_dir) / "uv-env"),
                "WARP_CACHE_PATH": str(Path(tmp_dir) / "warp-cache"),
            },
        )

    def _load_validation_tasks(self):
        loaded = load_plan(DEFAULT_PLAN_PATH, env={"USER": "executor-test"})
        expanded = expand_plan(loaded, run_mode="validation")
        benchmark_task = next(task for task in expanded["tasks"] if task["kind"] == "benchmark")
        render_task = next(task for task in expanded["tasks"] if task["kind"] == "render")
        return benchmark_task, render_task

    def test_run_task_writes_benchmark_manifests_statuses_and_logs(self):
        benchmark_task, _ = self._load_validation_tasks()

        with TemporaryDirectory() as tmp_dir:
            run_paths = self._make_run_paths(tmp_dir)
            validate_run_environment(run_paths)
            runner = FakeWorkerRunner()

            result = run_task(run_paths, benchmark_task, runner=runner, working_dir=Path(tmp_dir))

            self.assertEqual(result.status.state, "completed")
            self.assertEqual(len(result.job_results), benchmark_task["job_count"])
            self.assertTrue(runner.envs[0]["UV_CACHE_DIR"].endswith("uv-cache"))

            task_log_records = [
                json.loads(line)
                for line in run_paths.task_log_path(benchmark_task["id"]).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(task_log_records[0]["event"], "task_started")
            self.assertEqual(task_log_records[-1]["event"], "task_completed")

            for job_result in result.job_results:
                self.assertEqual(job_result.status.state, "completed")
                output_dir = Path(job_result.job_manifest.output_dir)
                self.assertTrue((output_dir / "measurements.jsonl").is_file())
                self.assertTrue((output_dir / "metadata.json").is_file())

    def test_run_task_preserves_partial_success_when_one_job_fails(self):
        benchmark_task, _ = self._load_validation_tasks()
        failed_job_id = benchmark_task["jobs"][1]["id"]

        with TemporaryDirectory() as tmp_dir:
            run_paths = self._make_run_paths(tmp_dir)
            validate_run_environment(run_paths)
            runner = FakeWorkerRunner(failed_job_ids={failed_job_id})

            result = run_task(run_paths, benchmark_task, runner=runner, working_dir=Path(tmp_dir))

            self.assertEqual(result.status.state, "failed")
            failed_result = next(
                job_result for job_result in result.job_results if job_result.job_manifest.job_id == failed_job_id
            )
            self.assertEqual(failed_result.status.state, "failed")
            self.assertEqual(failed_result.status.failure_phase, "execution")

            successful_results = [
                job_result for job_result in result.job_results if job_result.job_manifest.job_id != failed_job_id
            ]
            self.assertTrue(successful_results)
            for job_result in successful_results:
                self.assertTrue((Path(job_result.job_manifest.output_dir) / "measurements.jsonl").is_file())

    def test_run_task_writes_render_outputs(self):
        _, render_task = self._load_validation_tasks()

        with TemporaryDirectory() as tmp_dir:
            run_paths = self._make_run_paths(tmp_dir)
            validate_run_environment(run_paths)
            runner = FakeWorkerRunner()

            result = run_task(run_paths, render_task, runner=runner, working_dir=Path(tmp_dir))

            self.assertEqual(result.status.state, "completed")
            self.assertEqual(len(result.job_results), 1)
            output_dir = Path(result.job_results[0].job_manifest.output_dir)
            self.assertTrue((output_dir / "metadata.json").is_file())
            self.assertTrue((output_dir / "render_meta.json").is_file())
            self.assertTrue((output_dir / "g1_flat.mp4").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
