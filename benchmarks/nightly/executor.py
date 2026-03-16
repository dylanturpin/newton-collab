# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared nightly task executor for local and Slurm-owned submissions."""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from benchmarks.nightly.common import (
    JobManifest,
    RunPaths,
    StatusRecord,
    TaskManifest,
    make_log_record,
    utc_now_iso,
    validate_run_environment,
    write_json,
)
from newton.tools import solver_benchmark

CompletedProcessRunner = Callable[[list[str], Path, Mapping[str, str]], subprocess.CompletedProcess[str]]

REPO_ROOT = Path(__file__).resolve().parents[2]

BENCHMARK_REQUIRED_ARTIFACTS = ("measurements.jsonl", "metadata.json")
PROFILE_REQUIRED_ARTIFACTS = ("profile.nsys-rep", "profile.trace.json", "profile_meta.json")
RENDER_REQUIRED_ARTIFACTS = ("metadata.json", "render_meta.json")

JOB_METADATA_KEYS = {
    "id",
    "task_id",
    "kind",
    "profile",
    "hardware_label",
    "series",
    "scenario",
}


@dataclasses.dataclass(slots=True)
class JobExecutionResult:
    """Durable result summary for one concrete job."""

    job_manifest: JobManifest
    status: StatusRecord


@dataclasses.dataclass(slots=True)
class TaskExecutionResult:
    """Durable result summary for one task submission."""

    task_manifest: TaskManifest
    status: StatusRecord
    job_results: list[JobExecutionResult]


def run_task(
    run_paths: RunPaths,
    task: Mapping[str, Any],
    *,
    submission_id: str | None = None,
    dependency_ids: Sequence[str] = (),
    scheduler_context: Mapping[str, Any] | None = None,
    extra_env: Mapping[str, str] | None = None,
    working_dir: Path | str | None = None,
    runner: CompletedProcessRunner | None = None,
) -> TaskExecutionResult:
    """Execute one expanded task and write manifests, statuses, and logs."""
    validate_run_environment(run_paths)

    task_id = str(task["id"])
    task_dir = run_paths.task_dir(task_id)
    task_dir.mkdir(parents=True, exist_ok=True)
    task_log_path = run_paths.task_log_path(task_id)

    task_manifest = TaskManifest(
        task_id=task_id,
        kind=str(task["kind"]),
        profile=str(task["profile"]),
        hardware_label=str(task["hardware_label"]),
        job_ids=[str(job["id"]) for job in task["jobs"]],
        dependency_ids=[str(item) for item in dependency_ids],
        submission_id=submission_id,
        output_dir=str(task_dir),
    )
    task_payload = task_manifest.to_record()
    task_payload["series"] = str(task["series"])
    task_payload["scenario"] = str(task["scenario"])
    task_payload["definition"] = json.loads(json.dumps(dict(task)))
    write_json(run_paths.task_manifest_path(task_id), task_payload)

    started_at = utc_now_iso()
    running_task_status = StatusRecord(
        state="running",
        started_at=started_at,
        artifact_paths=[str(run_paths.task_manifest_path(task_id)), str(task_log_path)],
    )
    write_json(run_paths.task_status_path(task_id), running_task_status.to_record())
    _append_task_log(
        task_log_path,
        make_log_record(
            "task_started",
            run_id=run_paths.run_id,
            task_id=task_id,
            hardware_label=task_manifest.hardware_label,
            submission_id=submission_id,
            dependency_ids=list(dependency_ids),
            job_count=len(task_manifest.job_ids),
        ),
    )

    job_results: list[JobExecutionResult] = []
    failed_jobs = 0
    start_monotonic = time.monotonic()

    try:
        for job in task["jobs"]:
            job_result = run_job(
                run_paths,
                task,
                job,
                submission_id=submission_id,
                scheduler_context=scheduler_context,
                extra_env=extra_env,
                working_dir=working_dir,
                runner=runner,
            )
            job_results.append(job_result)
            if job_result.status.state == "failed":
                failed_jobs += 1
    except Exception as exc:
        task_status = StatusRecord(
            state="failed",
            started_at=started_at,
            finished_at=utc_now_iso(),
            duration_s=round(time.monotonic() - start_monotonic, 6),
            exit_code=1,
            error_summary=str(exc),
            failure_phase="setup",
            artifact_paths=_existing_paths(
                [
                    run_paths.task_manifest_path(task_id),
                    task_log_path,
                    *(result.job_manifest.output_dir for result in job_results),
                ]
            ),
        )
        write_json(run_paths.task_status_path(task_id), task_status.to_record())
        _append_task_log(
            task_log_path,
            make_log_record(
                "task_failed",
                run_id=run_paths.run_id,
                task_id=task_id,
                hardware_label=task_manifest.hardware_label,
                submission_id=submission_id,
                state=task_status.state,
                failure_phase=task_status.failure_phase,
                error_summary=task_status.error_summary,
            ),
        )
        return TaskExecutionResult(task_manifest=task_manifest, status=task_status, job_results=job_results)

    finished_at = utc_now_iso()
    duration_s = round(time.monotonic() - start_monotonic, 6)
    if failed_jobs:
        task_status = StatusRecord(
            state="failed",
            started_at=started_at,
            finished_at=finished_at,
            duration_s=duration_s,
            exit_code=1,
            error_summary=f"{failed_jobs}/{len(job_results)} jobs failed",
            failure_phase="execution",
            artifact_paths=_existing_paths(
                [
                    run_paths.task_manifest_path(task_id),
                    task_log_path,
                    *(result.job_manifest.output_dir for result in job_results),
                ]
            ),
        )
        event_name = "task_failed"
    else:
        task_status = StatusRecord(
            state="completed",
            started_at=started_at,
            finished_at=finished_at,
            duration_s=duration_s,
            exit_code=0,
            artifact_paths=_existing_paths(
                [
                    run_paths.task_manifest_path(task_id),
                    task_log_path,
                    *(result.job_manifest.output_dir for result in job_results),
                ]
            ),
        )
        event_name = "task_completed"

    write_json(run_paths.task_status_path(task_id), task_status.to_record())
    _append_task_log(
        task_log_path,
        make_log_record(
            event_name,
            run_id=run_paths.run_id,
            task_id=task_id,
            hardware_label=task_manifest.hardware_label,
            submission_id=submission_id,
            failed_jobs=failed_jobs,
            total_jobs=len(job_results),
            duration_s=duration_s,
            state=task_status.state,
        ),
    )
    return TaskExecutionResult(task_manifest=task_manifest, status=task_status, job_results=job_results)


def run_job(
    run_paths: RunPaths,
    task: Mapping[str, Any],
    job: Mapping[str, Any],
    *,
    submission_id: str | None = None,
    scheduler_context: Mapping[str, Any] | None = None,
    extra_env: Mapping[str, str] | None = None,
    working_dir: Path | str | None = None,
    runner: CompletedProcessRunner | None = None,
) -> JobExecutionResult:
    """Execute one concrete job and write job-owned artifacts."""
    task_id = str(task["id"])
    job_id = str(job["id"])
    job_dir = run_paths.job_dir(task_id, job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    results_dir = run_paths.job_results_dir(task_id, job_id)
    results_dir.mkdir(parents=True, exist_ok=True)

    scheduler_payload = dict(scheduler_context or {})
    if submission_id is not None:
        scheduler_payload.setdefault("submission_id", submission_id)
    if not scheduler_payload:
        scheduler_payload = None

    job_manifest = JobManifest(
        job_id=job_id,
        task_id=task_id,
        kind=str(job["kind"]),
        scenario=str(job["scenario"]),
        series=str(job["series"]),
        profile=str(job["profile"]),
        hardware_label=str(job["hardware_label"]),
        params=_resolved_job_params(job),
        output_dir=str(results_dir),
        scheduler=scheduler_payload,
    )
    write_json(run_paths.job_manifest_path(task_id, job_id), job_manifest.to_record())

    started_at = utc_now_iso()
    running_status = StatusRecord(
        state="running",
        started_at=started_at,
        artifact_paths=_existing_paths([run_paths.job_manifest_path(task_id, job_id), results_dir]),
    )
    write_json(run_paths.job_status_path(task_id, job_id), running_status.to_record())
    _append_task_log(
        run_paths.task_log_path(task_id),
        make_log_record(
            "job_started",
            run_id=run_paths.run_id,
            task_id=task_id,
            job_id=job_id,
            hardware_label=str(job["hardware_label"]),
            submission_id=submission_id,
            kind=str(job["kind"]),
        ),
    )

    start_monotonic = time.monotonic()
    try:
        command = build_job_command(job, results_dir)
    except Exception as exc:
        _write_text(run_paths.job_stderr_path(task_id, job_id), f"{type(exc).__name__}: {exc}\n")
        status = StatusRecord(
            state="failed",
            started_at=started_at,
            finished_at=utc_now_iso(),
            duration_s=round(time.monotonic() - start_monotonic, 6),
            exit_code=1,
            error_summary=str(exc),
            failure_phase="setup",
            artifact_paths=_existing_paths(
                [
                    run_paths.job_manifest_path(task_id, job_id),
                    run_paths.job_stderr_path(task_id, job_id),
                    results_dir,
                ]
            ),
        )
        write_json(run_paths.job_status_path(task_id, job_id), status.to_record())
        _append_task_log(
            run_paths.task_log_path(task_id),
            make_log_record(
                "job_failed",
                run_id=run_paths.run_id,
                task_id=task_id,
                job_id=job_id,
                hardware_label=str(job["hardware_label"]),
                exit_code=status.exit_code,
                failure_phase=status.failure_phase,
                error_summary=status.error_summary,
            ),
        )
        return JobExecutionResult(job_manifest=job_manifest, status=status)

    env = _worker_env(run_paths, task_id, extra_env)
    cwd = Path(working_dir) if working_dir is not None else REPO_ROOT
    job_runner = runner or _default_runner

    try:
        completed = job_runner(command, cwd, env)
    except Exception as exc:
        _write_text(run_paths.job_stderr_path(task_id, job_id), f"{type(exc).__name__}: {exc}\n")
        status = StatusRecord(
            state="failed",
            started_at=started_at,
            finished_at=utc_now_iso(),
            duration_s=round(time.monotonic() - start_monotonic, 6),
            exit_code=1,
            error_summary=str(exc),
            failure_phase="setup",
            artifact_paths=_existing_paths(
                [
                    run_paths.job_manifest_path(task_id, job_id),
                    run_paths.job_stderr_path(task_id, job_id),
                    results_dir,
                ]
            ),
        )
        write_json(run_paths.job_status_path(task_id, job_id), status.to_record())
        _append_task_log(
            run_paths.task_log_path(task_id),
            make_log_record(
                "job_failed",
                run_id=run_paths.run_id,
                task_id=task_id,
                job_id=job_id,
                hardware_label=str(job["hardware_label"]),
                exit_code=status.exit_code,
                failure_phase=status.failure_phase,
                error_summary=status.error_summary,
            ),
        )
        return JobExecutionResult(job_manifest=job_manifest, status=status)

    _write_text(run_paths.job_stdout_path(task_id, job_id), completed.stdout or "")
    _write_text(run_paths.job_stderr_path(task_id, job_id), completed.stderr or "")

    missing_artifacts = _missing_required_artifacts(job, results_dir)
    duration_s = round(time.monotonic() - start_monotonic, 6)
    if completed.returncode != 0:
        status = StatusRecord(
            state="failed",
            started_at=started_at,
            finished_at=utc_now_iso(),
            duration_s=duration_s,
            exit_code=completed.returncode,
            error_summary=f"worker exited with code {completed.returncode}",
            failure_phase="execution",
            artifact_paths=_existing_paths(
                [
                    run_paths.job_manifest_path(task_id, job_id),
                    run_paths.job_stdout_path(task_id, job_id),
                    run_paths.job_stderr_path(task_id, job_id),
                    results_dir,
                ]
            ),
        )
        event_name = "job_failed"
    elif missing_artifacts:
        status = StatusRecord(
            state="failed",
            started_at=started_at,
            finished_at=utc_now_iso(),
            duration_s=duration_s,
            exit_code=1,
            error_summary=f"missing expected artifacts: {', '.join(missing_artifacts)}",
            failure_phase="execution",
            artifact_paths=_existing_paths(
                [
                    run_paths.job_manifest_path(task_id, job_id),
                    run_paths.job_stdout_path(task_id, job_id),
                    run_paths.job_stderr_path(task_id, job_id),
                    results_dir,
                ]
            ),
        )
        event_name = "job_failed"
    else:
        status = StatusRecord(
            state="completed",
            started_at=started_at,
            finished_at=utc_now_iso(),
            duration_s=duration_s,
            exit_code=0,
            artifact_paths=_existing_paths(
                [
                    run_paths.job_manifest_path(task_id, job_id),
                    run_paths.job_stdout_path(task_id, job_id),
                    run_paths.job_stderr_path(task_id, job_id),
                    *required_artifact_paths(job, results_dir),
                ]
            ),
        )
        event_name = "job_completed"

    write_json(run_paths.job_status_path(task_id, job_id), status.to_record())
    _append_task_log(
        run_paths.task_log_path(task_id),
        make_log_record(
            event_name,
            run_id=run_paths.run_id,
            task_id=task_id,
            job_id=job_id,
            hardware_label=str(job["hardware_label"]),
            exit_code=status.exit_code,
            duration_s=status.duration_s,
            failure_phase=status.failure_phase,
            error_summary=status.error_summary,
            artifacts=status.artifact_paths,
        ),
    )
    return JobExecutionResult(job_manifest=job_manifest, status=status)


def build_worker_command(job: Mapping[str, Any], results_dir: Path) -> list[str]:
    """Build the concrete worker command for one expanded job."""
    if job["kind"] == "benchmark":
        args = SimpleNamespace(**job)
        command = solver_benchmark.build_run_command(
            args,
            dict(job["solver_config"]),
            int(job["num_worlds"]),
            substeps=int(job["substeps"]),
        )
        command.extend(["--out", str(results_dir)])
        return command
    if job["kind"] == "render":
        return _build_render_command(job, results_dir)
    raise ValueError(f"Unsupported job kind: {job['kind']}")


def build_job_command(job: Mapping[str, Any], results_dir: Path) -> list[str]:
    """Build the concrete job command, optionally wrapped with Nsight Systems."""
    worker_command = build_worker_command(job, results_dir)
    if not _job_uses_nsys(job):
        return worker_command
    return [
        sys.executable,
        "-m",
        "benchmarks.nightly.profiled_worker",
        "--results-dir",
        str(results_dir),
        "--measure-frames",
        str(job["measure_frames"]),
        "--cuda-graph-trace",
        str(job.get("nsys_cuda_graph_trace", "node")),
        "--",
        *worker_command,
    ]


def required_artifact_paths(job: Mapping[str, Any], results_dir: Path) -> list[Path]:
    """Return the artifact paths expected for a successful job."""
    if job["kind"] == "benchmark":
        artifact_names = list(BENCHMARK_REQUIRED_ARTIFACTS)
        if _job_uses_nsys(job):
            artifact_names.extend(PROFILE_REQUIRED_ARTIFACTS)
        return [results_dir / name for name in artifact_names]
    if job["kind"] == "render":
        return [results_dir / name for name in RENDER_REQUIRED_ARTIFACTS] + [results_dir / f"{job['scenario']}.mp4"]
    raise ValueError(f"Unsupported job kind: {job['kind']}")


def _build_render_command(job: Mapping[str, Any], results_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "newton.tools.solver_benchmark",
        "--render",
        "--scenario",
        str(job["scenario"]),
        "--solver",
        str(job["solver"]),
        "--num-worlds",
        str(job.get("num_worlds", 1)),
        "--substeps",
        str(job["substeps"]),
        "--render-frames",
        str(job["render_frames"]),
        "--render-width",
        str(job["render_width"]),
        "--render-height",
        str(job["render_height"]),
        "--out",
        str(results_dir),
    ]

    if job.get("camera_pos") is not None:
        command.extend(["--camera-pos", _camera_pos_value(job["camera_pos"])])
    if job.get("camera_pitch") is not None:
        command.extend(["--camera-pitch", str(job["camera_pitch"])])
    if job.get("camera_yaw") is not None:
        command.extend(["--camera-yaw", str(job["camera_yaw"])])

    _append_solver_override_flags(command, job)
    return command


def _append_solver_override_flags(command: list[str], job: Mapping[str, Any]) -> None:
    single_value_flags = (
        ("cholesky_kernel", "--cholesky-kernel"),
        ("trisolve_kernel", "--trisolve-kernel"),
        ("hinv_jt_kernel", "--hinv-jt-kernel"),
        ("delassus_kernel", "--delassus-kernel"),
        ("pgs_kernel", "--pgs-kernel"),
        ("pgs_iterations", "--pgs-iterations"),
        ("dense_max_constraints", "--dense-max-constraints"),
        ("pgs_beta", "--pgs-beta"),
        ("pgs_cfm", "--pgs-cfm"),
        ("pgs_omega", "--pgs-omega"),
        ("pgs_mode", "--pgs-mode"),
        ("delassus_chunk_size", "--delassus-chunk-size"),
        ("pgs_chunk_size", "--pgs-chunk-size"),
        ("mj_solver", "--mj-solver"),
        ("mj_integrator", "--mj-integrator"),
        ("mj_njmax", "--mj-njmax"),
        ("mj_nconmax", "--mj-nconmax"),
    )
    for field_name, flag in single_value_flags:
        value = job.get(field_name)
        if value is not None:
            command.extend([flag, str(value)])

    if job.get("pgs_warmstart"):
        command.append("--pgs-warmstart")

    if job.get("use_parallel_streams") is True:
        command.append("--use-parallel-streams")
    elif job.get("use_parallel_streams") is False:
        command.append("--no-parallel-streams")

    if job.get("double_buffer") is True:
        command.append("--double-buffer")
    elif job.get("double_buffer") is False:
        command.append("--no-double-buffer")

    if job.get("override_scenario_defaults"):
        command.append("--override-scenario-defaults")


def _default_runner(command: list[str], cwd: Path, env: Mapping[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, env=dict(env), capture_output=True, text=True, check=False)


def _append_task_log(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _task_cache_env(run_paths: RunPaths, task_id: str) -> dict[str, str]:
    cache_env: dict[str, str] = {}
    for key, base_path in run_paths.cache_env.items():
        task_path = base_path / task_id
        task_path.mkdir(parents=True, exist_ok=True)
        cache_env[key] = str(task_path)
    return cache_env


def _worker_env(run_paths: RunPaths, task_id: str, extra_env: Mapping[str, str] | None) -> dict[str, str]:
    env = dict(os.environ)
    env.update(_task_cache_env(run_paths, task_id))
    env["PYTHONUNBUFFERED"] = "1"
    if extra_env is not None:
        env.update(dict(extra_env))
    return env


def _resolved_job_params(job: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in job.items() if key not in JOB_METADATA_KEYS}


def _camera_pos_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence):
        return ",".join(str(item) for item in value)
    raise TypeError(f"Unsupported camera_pos value: {value!r}")


def _missing_required_artifacts(job: Mapping[str, Any], results_dir: Path) -> list[str]:
    return [str(path.name) for path in required_artifact_paths(job, results_dir) if not path.exists()]


def _job_uses_nsys(job: Mapping[str, Any]) -> bool:
    return job["kind"] == "benchmark" and bool(job.get("nsys_profile"))


def _existing_paths(paths: Sequence[str | Path]) -> list[str]:
    existing: list[str] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.exists():
            existing.append(str(path))
    return existing
