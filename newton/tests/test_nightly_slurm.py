# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmarks.nightly.common import resolve_run_paths, validate_run_environment
from benchmarks.nightly.plan import DEFAULT_PLAN_PATH, expand_plan, load_plan
from benchmarks.nightly.slurm import build_task_sbatch_command, submit_slurm_nightly


class FakeSbatchRunner:
    """Capture sbatch commands and return deterministic submission ids."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str]) -> str:
        self.commands.append(command)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class TestNightlySlurm(unittest.TestCase):
    def _load_validation_plan(self):
        loaded = load_plan(DEFAULT_PLAN_PATH, env={"USER": "slurm-test"})
        return expand_plan(loaded, run_mode="validation")

    def _make_run_paths(self, tmp_dir: str):
        run_paths = resolve_run_paths(
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
        validate_run_environment(run_paths)
        return run_paths

    def test_build_task_sbatch_command_maps_resources_dependencies_and_cache_env(self):
        expanded = self._load_validation_plan()
        benchmark_task = next(task for task in expanded["tasks"] if task["kind"] == "benchmark")

        with TemporaryDirectory() as tmp_dir:
            run_paths = self._make_run_paths(tmp_dir)
            command = build_task_sbatch_command(
                run_paths=run_paths,
                task=benchmark_task,
                slurm_settings=expanded["hardware_profiles"][benchmark_task["profile"]]["slurm"],
                dependency_ids=["12345"],
            )

            joined = " ".join(command)
            self.assertIn("--account all-users", joined)
            self.assertIn("--partition rtx-5090@ts2/b650d4u/1gpu-16cpu-128gb", joined)
            self.assertIn("--gres gpu:1", joined)
            self.assertIn("--time 01:00:00", joined)
            self.assertIn("--exclude 2u1g-b650-0839", joined)
            self.assertIn("--dependency afterany:12345", joined)
            self.assertIn("UV_CACHE_DIR=", joined)
            self.assertIn("benchmarks.nightly.slurm execute-task", joined)
            self.assertNotIn("ssh", joined)

    def test_submit_slurm_nightly_parallel_mode_submits_independent_tasks(self):
        runner = FakeSbatchRunner(["4101", "4102"])

        with TemporaryDirectory() as tmp_dir:
            summary = submit_slurm_nightly(
                plan_path=DEFAULT_PLAN_PATH,
                run_mode="validation",
                run_id="parallel-run",
                shared_state_dir=str(Path(tmp_dir) / "shared"),
                work_base_dir=str(Path(tmp_dir) / "work"),
                cache_env_overrides={
                    "TMPDIR": str(Path(tmp_dir) / "tmp"),
                    "UV_CACHE_DIR": str(Path(tmp_dir) / "uv-cache"),
                    "UV_PROJECT_ENVIRONMENT": str(Path(tmp_dir) / "uv-env"),
                    "WARP_CACHE_PATH": str(Path(tmp_dir) / "warp-cache"),
                },
                publish=False,
                submission_mode="parallel",
                sbatch_runner=runner,
            )

            self.assertEqual(summary["submitted_tasks"], 2)
            self.assertEqual(summary["failed_task_submissions"], 0)
            self.assertEqual(summary["publish"]["status"], "skipped")
            self.assertEqual(len(runner.commands), 2)
            self.assertTrue(all("--dependency" not in command for command in runner.commands))

            run_manifest = json.loads(
                (Path(summary["run_dir"]) / "run.json").read_text(encoding="utf-8")
            )
            self.assertEqual(run_manifest["mode"], "slurm")
            self.assertEqual(run_manifest["submission_mode"], "parallel")

    def test_submit_slurm_nightly_serial_mode_chains_tasks_and_publish_job(self):
        runner = FakeSbatchRunner(["5101", "5102", "6101"])

        with TemporaryDirectory() as tmp_dir:
            summary = submit_slurm_nightly(
                plan_path=DEFAULT_PLAN_PATH,
                run_mode="validation",
                run_id="serial-run",
                shared_state_dir=str(Path(tmp_dir) / "shared"),
                work_base_dir=str(Path(tmp_dir) / "work"),
                cache_env_overrides={
                    "TMPDIR": str(Path(tmp_dir) / "tmp"),
                    "UV_CACHE_DIR": str(Path(tmp_dir) / "uv-cache"),
                    "UV_PROJECT_ENVIRONMENT": str(Path(tmp_dir) / "uv-env"),
                    "WARP_CACHE_PATH": str(Path(tmp_dir) / "warp-cache"),
                },
                publish=True,
                submission_mode="serial",
                sbatch_runner=runner,
            )

            self.assertEqual(summary["publish"]["status"], "scheduled")
            self.assertEqual(summary["publish"]["submission_id"], "6101")
            self.assertEqual(len(runner.commands), 3)
            self.assertNotIn("--dependency", runner.commands[0])
            self.assertIn("--dependency", runner.commands[1])
            self.assertIn("afterany:5101", " ".join(runner.commands[1]))
            publish_command = " ".join(runner.commands[2])
            self.assertIn("afterany:5101:5102", publish_command)
            self.assertNotIn("--gres", publish_command)

    def test_submit_slurm_nightly_continues_after_task_submission_failure(self):
        runner = FakeSbatchRunner([RuntimeError("first submit failed"), "7102", "8101"])

        with TemporaryDirectory() as tmp_dir:
            summary = submit_slurm_nightly(
                plan_path=DEFAULT_PLAN_PATH,
                run_mode="validation",
                run_id="partial-submit-run",
                shared_state_dir=str(Path(tmp_dir) / "shared"),
                work_base_dir=str(Path(tmp_dir) / "work"),
                cache_env_overrides={
                    "TMPDIR": str(Path(tmp_dir) / "tmp"),
                    "UV_CACHE_DIR": str(Path(tmp_dir) / "uv-cache"),
                    "UV_PROJECT_ENVIRONMENT": str(Path(tmp_dir) / "uv-env"),
                    "WARP_CACHE_PATH": str(Path(tmp_dir) / "warp-cache"),
                },
                publish=True,
                submission_mode="parallel",
                sbatch_runner=runner,
            )

            self.assertEqual(summary["failed_task_submissions"], 1)
            self.assertEqual(summary["submitted_tasks"], 1)
            self.assertEqual(summary["publish"]["submission_id"], "8101")
            publish_command = " ".join(runner.commands[-1])
            self.assertIn("afterany:7102", publish_command)
            self.assertNotIn("afterany:7101", publish_command)

            failed_task = next(task for task in summary["tasks"] if task["status"] == "failed")
            failed_status = json.loads(
                (Path(summary["run_dir"]) / "tasks" / failed_task["task_id"] / "status.json").read_text(encoding="utf-8")
            )
            self.assertEqual(failed_status["failure_phase"], "submission")


if __name__ == "__main__":
    unittest.main(verbosity=2)
