# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmarks.nightly.common import PreparedRepo
from benchmarks.nightly.local import run_local_nightly
from benchmarks.nightly.plan import DEFAULT_PLAN_PATH


class FakeWorkerRunner:
    """Small fake subprocess runner for local nightly tests."""

    def __call__(self, command: list[str], cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
        command, out_dir, profiled = _unwrap_test_command(command)
        out_dir.mkdir(parents=True, exist_ok=True)

        if "--benchmark" in command:
            measurement = {
                "solver": command[command.index("--solver") + 1],
                "substeps": int(command[command.index("--substeps") + 1]),
                "num_worlds": int(command[command.index("--num-worlds") + 1]),
                "env_fps": 1024.0,
                "gpu_used_gb": 2.0,
                "gpu_total_gb": 24.0,
                "elapsed_s": 1.0,
                "ok": True,
                "kernels": {},
            }
            metadata = {
                "scenario": command[command.index("--scenario") + 1],
                "mode": "benchmark",
                "timestamp": "2026-03-08T12:00:00Z",
                "gpu": "Test GPU",
                "gpu_memory_total_gb": 24.0,
                "platform": "Linux",
                "python_version": "3.12.0",
                "pgs_iterations": 8,
                "measure_frames": 8,
                "warmup_frames": 4,
            }
            (out_dir / "measurements.jsonl").write_text(json.dumps(measurement) + "\n", encoding="utf-8")
            (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            if profiled:
                _write_profile_artifacts(out_dir)
            return subprocess.CompletedProcess(command, 0, stdout="benchmark ok\n", stderr="")

        scenario = command[command.index("--scenario") + 1]
        metadata = {
            "scenario": scenario,
            "mode": "render",
            "num_worlds": 1,
            "timestamp": "2026-03-08T12:00:00Z",
            "gpu": "Test GPU",
            "gpu_memory_total_gb": 24.0,
            "platform": "Linux",
            "python_version": "3.12.0",
        }
        render_meta = {"scenario": scenario, "video": f"{scenario}.mp4", "num_worlds": 1}
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


class RecordingWorkerRunner(FakeWorkerRunner):
    """Capture worker cwd values while reusing the fake artifact behavior."""

    def __init__(self):
        self.cwds: list[Path] = []

    def __call__(self, command: list[str], cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
        self.cwds.append(cwd)
        return super().__call__(command, cwd, env)


class TestNightlyLocal(unittest.TestCase):
    def test_run_local_nightly_validation_mode_writes_run_artifacts(self):
        with TemporaryDirectory() as tmp_dir:
            summary = run_local_nightly(
                plan_path=DEFAULT_PLAN_PATH,
                run_mode="validation",
                run_id="validation-run",
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
                runner=FakeWorkerRunner(),
            )

            run_dir = Path(summary["run_dir"])
            self.assertTrue((run_dir / "run.json").is_file())
            self.assertTrue((run_dir / "plan.lock.yaml").is_file())
            self.assertTrue((run_dir / "publish" / "summary.json").is_file())
            self.assertTrue(summary["artifacts_verified"])
            self.assertEqual(summary["publish"]["status"], "skipped")
            self.assertEqual(summary["selected_task_ids"], ["validation_g1_flat_sweep", "validation_g1_flat_render"])

            run_manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(run_manifest["plan_run_mode"], "validation")
            self.assertFalse(run_manifest["publish_requested"])
            self.assertEqual(run_manifest["publish_status"], "skipped")
            self.assertEqual(run_manifest["selected_profiles"], ["rtx_5090", "rtx_pro_6000_server"])

    def test_run_local_nightly_can_select_one_task(self):
        with TemporaryDirectory() as tmp_dir:
            summary = run_local_nightly(
                plan_path=DEFAULT_PLAN_PATH,
                run_mode="validation",
                task_ids=["validation_g1_flat_render"],
                run_id="render-only",
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
                runner=FakeWorkerRunner(),
            )

            self.assertEqual(summary["selected_task_ids"], ["validation_g1_flat_render"])
            self.assertEqual(summary["total_tasks"], 1)
            self.assertEqual(summary["total_jobs"], 1)

    def test_run_local_nightly_can_publish_to_a_local_site_root(self):
        with TemporaryDirectory() as tmp_dir:
            site_root = Path(tmp_dir) / "published" / "nightly"
            summary = run_local_nightly(
                plan_path=DEFAULT_PLAN_PATH,
                run_mode="validation",
                run_id="publish-local",
                shared_state_dir=str(Path(tmp_dir) / "shared"),
                work_base_dir=str(Path(tmp_dir) / "work"),
                publish_root=site_root,
                cache_env_overrides={
                    "TMPDIR": str(Path(tmp_dir) / "tmp"),
                    "UV_CACHE_DIR": str(Path(tmp_dir) / "uv-cache"),
                    "UV_PROJECT_ENVIRONMENT": str(Path(tmp_dir) / "uv-env"),
                    "WARP_CACHE_PATH": str(Path(tmp_dir) / "warp-cache"),
                    "NEWTON_CACHE_PATH": str(Path(tmp_dir) / "newton-cache"),
                    "CUDA_CACHE_PATH": str(Path(tmp_dir) / "cuda-cache"),
                },
                publish=True,
                working_dir=Path(tmp_dir),
                runner=FakeWorkerRunner(),
            )

            self.assertEqual(summary["publish"]["status"], "completed")
            self.assertTrue((site_root / "runs.jsonl").is_file())
            self.assertTrue((site_root / "points.jsonl").is_file())
            self.assertTrue((site_root / "runs" / "publish-local" / "renders.json").is_file())

    def test_run_local_nightly_can_prepare_a_cherry_picked_checkout(self):
        with TemporaryDirectory() as tmp_dir:
            prepared_repo_root = Path(tmp_dir) / "prepared-repo"
            prepared_repo_root.mkdir(parents=True)
            worker_runner = RecordingWorkerRunner()

            def fake_repo_preparer(*, source_repo_root, run_dir, cherry_pick_refs):
                return PreparedRepo(
                    source_repo_root=Path(source_repo_root),
                    repo_root=prepared_repo_root,
                    base_revision="base123",
                    revision="exec456",
                    cherry_pick_refs=list(cherry_pick_refs or []),
                    resolved_cherry_pick_refs=["origin/fast-bulk-replicate"],
                )

            summary = run_local_nightly(
                plan_path=DEFAULT_PLAN_PATH,
                run_mode="validation",
                run_id="prepared-repo-run",
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
                cherry_pick_refs=["fast-bulk-replicate"],
                working_dir=Path(tmp_dir) / "ignored-working-dir",
                repo_preparer=fake_repo_preparer,
                runner=worker_runner,
            )

            self.assertTrue(worker_runner.cwds)
            self.assertTrue(all(cwd == prepared_repo_root for cwd in worker_runner.cwds))
            run_manifest = json.loads((Path(summary["run_dir"]) / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(run_manifest["revision"], "exec456")
            self.assertEqual(run_manifest["base_revision"], "base123")
            self.assertEqual(run_manifest["execution_repo_root"], str(prepared_repo_root))
            self.assertEqual(run_manifest["cherry_pick_refs"], ["fast-bulk-replicate"])
            self.assertEqual(run_manifest["resolved_cherry_pick_refs"], ["origin/fast-bulk-replicate"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
