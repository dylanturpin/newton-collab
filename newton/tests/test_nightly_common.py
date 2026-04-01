# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import datetime as dt
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmarks.nightly.common import (
    CACHE_ENV_KEYS,
    JobManifest,
    RunManifest,
    StatusRecord,
    TaskManifest,
    expand_env_path,
    make_log_record,
    make_run_id,
    resolve_cache_env,
    resolve_run_paths,
    validate_run_environment,
)


class TestNightlyCommon(unittest.TestCase):
    def test_make_run_id_uses_utc_timestamp_format(self):
        run_id = make_run_id(dt.datetime(2026, 3, 8, 20, 30, 15, tzinfo=dt.UTC))
        self.assertEqual(run_id, "2026-03-08T20-30-15Z")

    def test_resolve_cache_env_expands_values(self):
        resolved = resolve_cache_env(
            {
                "TMPDIR": "/tmp",
                "UV_CACHE_DIR": "/tmp/$USER/cache",
                "UV_PROJECT_ENVIRONMENT": "/tmp/$USER/env",
                "WARP_CACHE_PATH": "/tmp/$USER/warp",
                "NEWTON_CACHE_PATH": "/tmp/$USER/newton",
                "CUDA_CACHE_PATH": "/tmp/$USER/cuda",
            },
            env={"USER": "nightly"},
        )
        self.assertEqual(resolved["UV_CACHE_DIR"], "/tmp/nightly/cache")
        self.assertEqual(resolved["NEWTON_CACHE_PATH"], "/tmp/nightly/newton")
        self.assertEqual(resolved["CUDA_CACHE_PATH"], "/tmp/nightly/cuda")
        self.assertEqual(tuple(resolved), CACHE_ENV_KEYS)

    def test_resolve_run_paths_and_validate_environment(self):
        with TemporaryDirectory() as tmp_dir:
            shared_state_dir = Path(tmp_dir) / "shared"
            work_base_dir = Path(tmp_dir) / "work"
            run_paths = resolve_run_paths(
                run_id="2026-03-08T20-30-15Z",
                shared_state_dir=shared_state_dir,
                work_base_dir=work_base_dir,
                cache_env={
                    "TMPDIR": str(Path(tmp_dir) / "tmp"),
                    "UV_CACHE_DIR": str(Path(tmp_dir) / "uv-cache"),
                    "UV_PROJECT_ENVIRONMENT": str(Path(tmp_dir) / "uv-env"),
                    "WARP_CACHE_PATH": str(Path(tmp_dir) / "warp-cache"),
                    "NEWTON_CACHE_PATH": str(Path(tmp_dir) / "newton-cache"),
                    "CUDA_CACHE_PATH": str(Path(tmp_dir) / "cuda-cache"),
                },
            )

            validate_run_environment(run_paths)

            self.assertTrue(run_paths.shared_state_dir.is_dir())
            self.assertTrue(run_paths.work_base_dir.is_dir())
            self.assertEqual(run_paths.run_dir, shared_state_dir / "runs" / run_paths.run_id)

    def test_make_log_record_emits_parseable_json(self):
        line = make_log_record(
            "task_submitted",
            run_id="run-001",
            task_id="g1_flat_sweep",
            hardware_label="rtx-5090",
            submission_id="12345",
        )
        record = json.loads(line)
        self.assertEqual(record["event"], "task_submitted")
        self.assertEqual(record["run_id"], "run-001")
        self.assertEqual(record["submission_id"], "12345")

    def test_manifest_records_are_json_serializable(self):
        run_manifest = RunManifest(
            run_id="run-001",
            mode="slurm",
            revision="abcdef1",
            created_at="2026-03-08T20:30:15Z",
            plan_path="benchmarks/nightly/nightly.yaml",
            plan_run_mode="validation",
            selected_task_ids=["validation_g1_flat_sweep"],
            selected_profiles=["rtx_5090"],
            cache_env={"TMPDIR": "/tmp"},
            shared_state_dir="/tmp/state",
            work_base_dir="/tmp/work",
            publish_requested=False,
            publish_status="skipped",
        )
        task_manifest = TaskManifest(
            task_id="g1_flat_sweep",
            kind="benchmark",
            profile="rtx_5090",
            hardware_label="rtx-5090",
            job_ids=["g1_flat_sweep__0001"],
            dependency_ids=[],
            submission_id="12345",
            output_dir="/tmp/state/runs/run-001/tasks/g1_flat_sweep",
        )
        job_manifest = JobManifest(
            job_id="g1_flat_sweep__0001",
            task_id="g1_flat_sweep",
            kind="benchmark",
            scenario="g1_flat",
            series="g1_flat_env_fps",
            profile="rtx_5090",
            hardware_label="rtx-5090",
            params={"solver": "fpgs_split", "num_worlds": 1024},
            output_dir="/tmp/state/runs/run-001/tasks/g1_flat_sweep/jobs/g1_flat_sweep__0001",
        )
        status_record = StatusRecord(state="completed", exit_code=0, artifact_paths=["measurements.jsonl"])

        for record in (
            run_manifest.to_record(),
            task_manifest.to_record(),
            job_manifest.to_record(),
            status_record.to_record(),
        ):
            self.assertIsInstance(json.dumps(record), str)

    def test_expand_env_path_returns_path(self):
        path = expand_env_path("/tmp/$USER/nightly", env={"USER": "codex"})
        self.assertEqual(path, Path("/tmp/codex/nightly"))

    def test_run_paths_include_run_level_artifact_locations(self):
        with TemporaryDirectory() as tmp_dir:
            run_paths = resolve_run_paths(
                run_id="run-001",
                shared_state_dir=Path(tmp_dir) / "shared",
                work_base_dir=Path(tmp_dir) / "work",
                cache_env={
                    "TMPDIR": str(Path(tmp_dir) / "tmp"),
                    "UV_CACHE_DIR": str(Path(tmp_dir) / "uv-cache"),
                    "UV_PROJECT_ENVIRONMENT": str(Path(tmp_dir) / "uv-env"),
                    "WARP_CACHE_PATH": str(Path(tmp_dir) / "warp-cache"),
                    "NEWTON_CACHE_PATH": str(Path(tmp_dir) / "newton-cache"),
                    "CUDA_CACHE_PATH": str(Path(tmp_dir) / "cuda-cache"),
                },
            )

            self.assertEqual(run_paths.run_manifest_path(), run_paths.run_dir / "run.json")
            self.assertEqual(run_paths.plan_lock_path(), run_paths.run_dir / "plan.lock.yaml")
            self.assertEqual(run_paths.publish_summary_path(), run_paths.publish_dir / "summary.json")


if __name__ == "__main__":
    unittest.main(verbosity=2)
