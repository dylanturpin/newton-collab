# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmarks.nightly.local import run_local_nightly
from benchmarks.nightly.plan import DEFAULT_PLAN_PATH
from benchmarks.nightly.publish import publish_run


class FakeWorkerRunner:
    """Small fake subprocess runner for publisher tests."""

    def __init__(self, *, failed_job_ids: set[str] | None = None):
        self.failed_job_ids = set(failed_job_ids or set())

    def __call__(self, command: list[str], cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
        command, out_dir, profiled = _unwrap_test_command(command)
        out_dir.mkdir(parents=True, exist_ok=True)
        job_id = out_dir.parent.name
        gpu_name = "RTX PRO 6000 Blackwell Server Edition" if "--render" in command else "Synthetic GPU"

        if job_id in self.failed_job_ids:
            return subprocess.CompletedProcess(command, 9, stdout="", stderr="synthetic failure\n")

        scenario = command[command.index("--scenario") + 1]
        metadata = {
            "scenario": scenario,
            "timestamp": "2026-03-08T12:00:00Z",
            "gpu": gpu_name,
            "gpu_memory_total_gb": 48.0,
            "platform": "Linux",
            "python_version": "3.12.0",
            "pgs_iterations": 8,
            "measure_frames": 8,
            "warmup_frames": 4,
        }

        if "--benchmark" in command:
            measurement = {
                "solver": command[command.index("--solver") + 1],
                "substeps": int(command[command.index("--substeps") + 1]),
                "num_worlds": int(command[command.index("--num-worlds") + 1]),
                "env_fps": 1024.0,
                "gpu_used_gb": 2.0,
                "gpu_total_gb": 48.0,
                "elapsed_s": 1.0,
                "ok": True,
                "kernels": {"pgs_solve_kernel": 8.7},
            }
            (out_dir / "measurements.jsonl").write_text(json.dumps(measurement) + "\n", encoding="utf-8")
            (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            if profiled:
                _write_profile_artifacts(out_dir)
            return subprocess.CompletedProcess(command, 0, stdout="benchmark ok\n", stderr="")

        render_meta = {
            "scenario": scenario,
            "solver": command[command.index("--solver") + 1],
            "substeps": int(command[command.index("--substeps") + 1]),
            "fps": 60,
            "frames": 32,
            "width": 640,
            "height": 360,
            "video": f"{scenario}.mp4",
            "label": "Synthetic render",
        }
        (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        (out_dir / "render_meta.json").write_text(json.dumps(render_meta, indent=2) + "\n", encoding="utf-8")
        (out_dir / f"{scenario}.mp4").write_bytes(b"fake mp4")
        return subprocess.CompletedProcess(command, 0, stdout="render ok\n", stderr="")


def _unwrap_test_command(command: list[str]) -> tuple[list[str], Path, bool]:
    if "benchmarks.nightly.profiled_worker" not in command:
        return command, Path(command[command.index("--out") + 1]), False
    results_dir = Path(command[command.index("--results-dir") + 1])
    inner_command = command[command.index("--") + 1 :]
    return inner_command, results_dir, True


def _write_profile_artifacts(out_dir: Path) -> None:
    (out_dir / "profile.nsys-rep").write_text("synthetic nsys report\n", encoding="utf-8")
    (out_dir / "profile.trace.json").write_text('{"traceEvents":[]}\n', encoding="utf-8")
    (out_dir / "profile_meta.json").write_text(
        json.dumps(
            {
                "tool": "nsys",
                "cuda_graph_trace": "node",
                "measure_frames": 8,
                "report_name": "profile.nsys-rep",
                "trace_name": "profile.trace.json",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


class TestNightlyPublish(unittest.TestCase):
    def _run_validation(self, tmp_dir: str, *, run_id: str, failed_job_ids: set[str] | None = None) -> Path:
        summary = run_local_nightly(
            plan_path=DEFAULT_PLAN_PATH,
            run_mode="validation",
            run_id=run_id,
            shared_state_dir=str(Path(tmp_dir) / "shared"),
            work_base_dir=str(Path(tmp_dir) / "work"),
            cache_env_overrides={
                "TMPDIR": str(Path(tmp_dir) / "tmp"),
                "UV_CACHE_DIR": str(Path(tmp_dir) / "uv-cache"),
                "UV_PROJECT_ENVIRONMENT": str(Path(tmp_dir) / "uv-env"),
                "WARP_CACHE_PATH": str(Path(tmp_dir) / "warp-cache"),
                "NEWTON_CACHE_PATH": str(Path(tmp_dir) / "newton-cache"),
                "CUDA_CACHE_PATH": str(Path(tmp_dir) / "cuda-cache"),
            },
            publish=False,
            working_dir=Path(tmp_dir),
            runner=FakeWorkerRunner(failed_job_ids=failed_job_ids),
        )
        return Path(summary["run_dir"])

    def test_publish_run_writes_site_and_replaces_rows_for_same_run(self):
        with TemporaryDirectory() as tmp_dir:
            run_dir = self._run_validation(tmp_dir, run_id="publish-test")
            site_root = Path(tmp_dir) / "site" / "nightly"

            publish_run(run_dir, publish_root=site_root)
            run_rows = [row for row in _jsonl_rows(site_root / "runs.jsonl") if row["run_id"] == "publish-test"]
            self.assertEqual(len(run_rows), 2)
            benchmark_row = next(row for row in run_rows if row["gpu"] == "Synthetic GPU")
            render_row = next(row for row in run_rows if row["gpu"] == "RTX PRO 6000 Blackwell Server Edition")
            self.assertEqual(benchmark_row["point_count"], 4)
            self.assertEqual(benchmark_row["render_count"], 0)
            self.assertEqual(render_row["point_count"], 0)
            self.assertEqual(render_row["render_count"], 1)

            initial_points = [row for row in _jsonl_rows(site_root / "points.jsonl") if row["run_id"] == "publish-test"]
            self.assertEqual(len(initial_points), 4)
            self.assertTrue(all(row["mode"] == "sweep" for row in initial_points))
            self.assertTrue((site_root / "runs" / "publish-test" / "renders.json").is_file())

            measurement_path = (
                run_dir
                / "tasks"
                / "validation_g1_flat_sweep"
                / "jobs"
                / "validation_g1_flat_sweep__0001"
                / "results"
                / "measurements.jsonl"
            )
            measurement = _jsonl_rows(measurement_path)[0]
            measurement["env_fps"] = 2048.0
            measurement_path.write_text(json.dumps(measurement) + "\n", encoding="utf-8")

            publish_run(run_dir, publish_root=site_root)
            updated_points = [row for row in _jsonl_rows(site_root / "points.jsonl") if row["run_id"] == "publish-test"]
            self.assertEqual(len(updated_points), 4)
            updated_first = next(row for row in updated_points if row["job_id"] == "validation_g1_flat_sweep__0001")
            self.assertEqual(updated_first["env_fps"], 2048.0)
            self.assertEqual(
                len([row for row in _jsonl_rows(site_root / "runs.jsonl") if row["run_id"] == "publish-test"]), 2
            )

    def test_publish_run_keeps_partial_success_and_failed_placeholder_rows(self):
        with TemporaryDirectory() as tmp_dir:
            run_dir = self._run_validation(
                tmp_dir,
                run_id="publish-failure",
                failed_job_ids={"validation_g1_flat_sweep__0002"},
            )
            site_root = Path(tmp_dir) / "site" / "nightly"

            publish_run(run_dir, publish_root=site_root)

            run_row = next(row for row in _jsonl_rows(site_root / "runs.jsonl") if row["run_id"] == "publish-failure")
            self.assertEqual(run_row["status"], "failed")
            self.assertEqual(run_row["failed_jobs"], 1)

            run_points = [row for row in _jsonl_rows(site_root / "points.jsonl") if row["run_id"] == "publish-failure"]
            self.assertEqual(len(run_points), 4)
            failed_rows = [row for row in run_points if row.get("ok") is False]
            self.assertEqual(len(failed_rows), 1)
            self.assertEqual(failed_rows[0]["job_id"], "validation_g1_flat_sweep__0002")
            self.assertEqual(failed_rows[0]["failure_phase"], "execution")

            summary = json.loads((site_root / "runs" / "publish-failure" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["failed_jobs"], 1)
            self.assertEqual(summary["render_count"], 1)

    def test_failed_render_without_metadata_stays_in_one_gpu_group(self):
        with TemporaryDirectory() as tmp_dir:
            run_dir = self._run_validation(
                tmp_dir,
                run_id="publish-render-failure",
                failed_job_ids={"validation_g1_flat_render__0001"},
            )
            site_root = Path(tmp_dir) / "site" / "nightly"

            publish_run(run_dir, publish_root=site_root)

            run_rows = [
                row for row in _jsonl_rows(site_root / "runs.jsonl") if row["run_id"] == "publish-render-failure"
            ]
            self.assertEqual(len(run_rows), 2)
            benchmark_row = next(row for row in run_rows if row["gpu"] == "Synthetic GPU")
            failed_render_row = next(row for row in run_rows if row["gpu"] == "rtx-pro-6000-blackwell-server-edition")
            self.assertEqual(benchmark_row["failed_jobs"], 0)
            self.assertEqual(benchmark_row["point_count"], 4)
            self.assertEqual(failed_render_row["failed_jobs"], 1)
            self.assertEqual(failed_render_row["point_count"], 0)

    def test_publish_run_emits_one_run_row_per_gpu_for_mixed_profile_run(self):
        with TemporaryDirectory() as tmp_dir:
            run_dir = self._run_validation(tmp_dir, run_id="publish-multi-gpu")
            render_task_dir = run_dir / "tasks" / "validation_g1_flat_render"
            render_job_dir = render_task_dir / "jobs" / "validation_g1_flat_render__0001"

            task_manifest = json.loads((render_task_dir / "task.json").read_text(encoding="utf-8"))
            task_manifest["hardware_label"] = "rtx-pro-6000-blackwell-server-edition"
            (render_task_dir / "task.json").write_text(
                json.dumps(task_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

            job_manifest = json.loads((render_job_dir / "job.json").read_text(encoding="utf-8"))
            job_manifest["hardware_label"] = "rtx-pro-6000-blackwell-server-edition"
            (render_job_dir / "job.json").write_text(
                json.dumps(job_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )

            metadata = json.loads((render_job_dir / "results" / "metadata.json").read_text(encoding="utf-8"))
            metadata["gpu"] = "RTX PRO 6000 Blackwell Server Edition"
            (render_job_dir / "results" / "metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            site_root = Path(tmp_dir) / "site" / "nightly"
            publish_run(run_dir, publish_root=site_root)

            run_rows = [row for row in _jsonl_rows(site_root / "runs.jsonl") if row["run_id"] == "publish-multi-gpu"]
            self.assertEqual(len(run_rows), 2)
            rows_by_gpu = {row["gpu"]: row for row in run_rows}
            self.assertEqual(rows_by_gpu["Synthetic GPU"]["point_count"], 4)
            self.assertEqual(rows_by_gpu["Synthetic GPU"]["render_count"], 0)
            self.assertEqual(rows_by_gpu["RTX PRO 6000 Blackwell Server Edition"]["point_count"], 0)
            self.assertEqual(rows_by_gpu["RTX PRO 6000 Blackwell Server Edition"]["render_count"], 1)

    def test_publish_run_copies_profile_artifacts_and_links_them_from_points(self):
        with TemporaryDirectory() as tmp_dir:
            run_dir = self._run_validation(tmp_dir, run_id="publish-profiled")
            profile_results_dir = (
                run_dir / "tasks" / "validation_g1_flat_sweep" / "jobs" / "validation_g1_flat_sweep__0001" / "results"
            )
            (profile_results_dir / "profile.nsys-rep").write_text("synthetic nsys report\n", encoding="utf-8")
            (profile_results_dir / "profile.trace.json").write_text('{"traceEvents":[]}\n', encoding="utf-8")
            (profile_results_dir / "profile_meta.json").write_text(
                json.dumps(
                    {
                        "tool": "nsys",
                        "cuda_graph_trace": "node",
                        "measure_frames": 8,
                        "report_name": "profile.nsys-rep",
                        "trace_name": "profile.trace.json",
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            site_root = Path(tmp_dir) / "site" / "nightly"
            publish_run(run_dir, publish_root=site_root)

            run_points = [row for row in _jsonl_rows(site_root / "points.jsonl") if row["run_id"] == "publish-profiled"]
            profiled_row = next(row for row in run_points if row["job_id"] == "validation_g1_flat_sweep__0001")
            self.assertEqual(
                profiled_row["nsys_report_path"], "profiles/validation_g1_flat_sweep__0001/profile.nsys-rep"
            )
            self.assertEqual(
                profiled_row["nsys_trace_path"], "profiles/validation_g1_flat_sweep__0001/profile.trace.json"
            )
            self.assertTrue(
                (
                    site_root
                    / "runs"
                    / "publish-profiled"
                    / "profiles"
                    / "validation_g1_flat_sweep__0001"
                    / "profile.nsys-rep"
                ).is_file()
            )
            self.assertTrue(
                (
                    site_root
                    / "runs"
                    / "publish-profiled"
                    / "profiles"
                    / "validation_g1_flat_sweep__0001"
                    / "profile.trace.json"
                ).is_file()
            )


def _jsonl_rows(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if raw:
            rows.append(json.loads(raw))
    return rows


if __name__ == "__main__":
    unittest.main(verbosity=2)
