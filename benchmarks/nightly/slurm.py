# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Head-node Slurm submission and compute-node task execution helpers."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shlex
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from benchmarks.nightly.common import (
    RunManifest,
    RunPaths,
    StatusRecord,
    TaskManifest,
    make_log_record,
    make_run_id,
    prepare_execution_repo,
    resolve_run_paths,
    utc_now_iso,
    validate_run_environment,
    write_json,
)
from benchmarks.nightly.executor import required_artifact_paths, run_task
from benchmarks.nightly.local import _apply_plan_overrides, _prepare_run_directories
from benchmarks.nightly.plan import DEFAULT_PLAN_PATH, RUN_MODES, expand_plan, load_plan, write_plan_lock

DEFAULT_UV_EXTRAS = ("--extra", "examples", "--extra", "torch-cu12")
SBATCHRunner = Callable[[list[str]], str]

REPO_ROOT = Path(__file__).resolve().parents[2]


def submit_slurm_nightly(
    *,
    plan_path: Path | str = DEFAULT_PLAN_PATH,
    run_mode: str = "full",
    task_ids: Sequence[str] | None = None,
    run_id: str | None = None,
    shared_state_dir: str | None = None,
    work_base_dir: str | None = None,
    publish_root: Path | str | None = None,
    cache_env_overrides: Mapping[str, str] | None = None,
    cherry_pick_refs: Sequence[str] | None = None,
    publish: bool = False,
    submission_mode: str = "parallel",
    sbatch_runner: SBATCHRunner | None = None,
    repo_root: Path | str = REPO_ROOT,
    repo_preparer=prepare_execution_repo,
) -> dict[str, Any]:
    """Submit the expanded nightly plan to Slurm from the head node."""
    if submission_mode not in {"parallel", "serial"}:
        raise ValueError(f"Unsupported submission mode: {submission_mode}")

    loaded_plan = load_plan(plan_path)
    _apply_plan_overrides(
        loaded_plan,
        shared_state_dir=shared_state_dir,
        work_base_dir=work_base_dir,
        cache_env_overrides=cache_env_overrides,
    )
    expanded_plan = expand_plan(loaded_plan, run_mode=run_mode, selected_task_ids=task_ids)

    effective_run_id = run_id or make_run_id()
    run_paths = resolve_run_paths(
        run_id=effective_run_id,
        shared_state_dir=loaded_plan["defaults"]["shared_state_dir"],
        work_base_dir=loaded_plan["defaults"]["work_base_dir"],
        cache_env=loaded_plan["defaults"]["cache_env"],
    )
    validate_run_environment(run_paths)
    _prepare_run_directories(run_paths)
    prepared_repo = repo_preparer(
        source_repo_root=repo_root,
        run_dir=run_paths.run_dir,
        cherry_pick_refs=cherry_pick_refs,
    )

    run_manifest = RunManifest(
        run_id=effective_run_id,
        mode="slurm",
        revision=prepared_repo.revision,
        created_at=utc_now_iso(),
        plan_path=str(Path(plan_path)),
        plan_run_mode=run_mode,
        selected_task_ids=[task["id"] for task in expanded_plan["tasks"]],
        selected_profiles=sorted({task["profile"] for task in expanded_plan["tasks"]}),
        cache_env={key: str(value) for key, value in run_paths.cache_env.items()},
        shared_state_dir=str(run_paths.shared_state_dir),
        work_base_dir=str(run_paths.work_base_dir),
        publish_root=str(publish_root) if publish_root is not None else None,
        publish_requested=publish,
        publish_status="pending" if publish else "skipped",
        submission_mode=submission_mode,
        base_revision=prepared_repo.base_revision,
        execution_repo_root=str(prepared_repo.repo_root),
        cherry_pick_refs=list(prepared_repo.cherry_pick_refs),
        resolved_cherry_pick_refs=list(prepared_repo.resolved_cherry_pick_refs),
    )
    write_json(run_paths.run_manifest_path(), run_manifest.to_record())
    write_plan_lock(expanded_plan, run_paths.plan_lock_path())

    sbatch = sbatch_runner or _default_sbatch_runner
    previous_submission_by_profile: dict[str, str] = {}
    successful_submission_ids: list[str] = []
    task_summaries: list[dict[str, Any]] = []

    for task in expanded_plan["tasks"]:
        task_id = str(task["id"])
        profile_name = str(task["profile"])
        dependency_ids = []
        if submission_mode == "serial" and profile_name in previous_submission_by_profile:
            dependency_ids = [previous_submission_by_profile[profile_name]]

        _write_task_submission_artifacts(run_paths, task, dependency_ids=dependency_ids)

        task_command = build_task_sbatch_command(
            run_paths=run_paths,
            task=task,
            slurm_settings=expanded_plan["hardware_profiles"][profile_name]["slurm"],
            dependency_ids=dependency_ids,
            repo_root=prepared_repo.repo_root,
        )
        try:
            submission_id = sbatch(task_command)
        except Exception as exc:
            _mark_task_submission_failed(run_paths, task, dependency_ids=dependency_ids, error_summary=str(exc))
            task_summaries.append(
                {
                    "task_id": task_id,
                    "profile": profile_name,
                    "hardware_label": task["hardware_label"],
                    "dependency_ids": dependency_ids,
                    "submission_id": None,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            continue

        _record_task_submission(
            run_paths,
            task,
            submission_id=submission_id,
            dependency_ids=dependency_ids,
        )
        successful_submission_ids.append(submission_id)
        previous_submission_by_profile[profile_name] = submission_id
        task_summaries.append(
            {
                "task_id": task_id,
                "profile": profile_name,
                "hardware_label": task["hardware_label"],
                "dependency_ids": dependency_ids,
                "submission_id": submission_id,
                "status": "submitted",
            }
        )

    publish_submission_id = None
    publish_error = None
    if publish:
        publish_command = build_publish_sbatch_command(
            run_paths=run_paths,
            dependency_ids=successful_submission_ids,
            slurm_settings=_publish_slurm_settings(expanded_plan),
            repo_root=prepared_repo.repo_root,
        )
        try:
            publish_submission_id = sbatch(publish_command)
            run_manifest.publish_status = "scheduled"
            run_manifest.publish_submission_id = publish_submission_id
        except Exception as exc:
            publish_error = str(exc)
            run_manifest.publish_status = "failed"
            _append_publish_log(
                run_paths,
                make_log_record(
                    "publish_submission_failed",
                    run_id=run_paths.run_id,
                    error_summary=publish_error,
                    dependency_ids=successful_submission_ids,
                ),
            )
    else:
        run_manifest.publish_status = "skipped"

    write_json(run_paths.run_manifest_path(), run_manifest.to_record())
    return {
        "run_id": run_paths.run_id,
        "run_dir": str(run_paths.run_dir),
        "submission_mode": submission_mode,
        "selected_task_ids": [task["id"] for task in expanded_plan["tasks"]],
        "submitted_tasks": len(successful_submission_ids),
        "failed_task_submissions": sum(1 for item in task_summaries if item["status"] == "failed"),
        "tasks": task_summaries,
        "publish": {
            "requested": publish,
            "submission_id": publish_submission_id,
            "status": run_manifest.publish_status,
            "error": publish_error,
            "dependency_ids": successful_submission_ids,
        },
    }


def build_task_sbatch_command(
    *,
    run_paths: RunPaths,
    task: Mapping[str, Any],
    slurm_settings: Mapping[str, Any],
    dependency_ids: Sequence[str],
    repo_root: Path | str = REPO_ROOT,
) -> list[str]:
    """Build the `sbatch` command for one authored task submission."""
    task_dir = run_paths.task_dir(str(task["id"]))
    task_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "sbatch",
        "--parsable",
        "--chdir",
        str(Path(repo_root)),
        "--job-name",
        _job_name(run_paths.run_id, str(task["id"])),
        "--output",
        str(task_dir / "slurm-%j.out"),
        "--error",
        str(task_dir / "slurm-%j.err"),
        "--export",
        _export_spec(run_paths),
    ]
    command.extend(_slurm_resource_args(slurm_settings))
    if dependency_ids:
        command.extend(["--dependency", _afterany_dependency(dependency_ids)])

    wrapped = shlex.join(build_task_execution_command(run_paths.run_dir, str(task["id"])))
    command.extend(["--wrap", wrapped])
    return command


def build_publish_sbatch_command(
    *,
    run_paths: RunPaths,
    dependency_ids: Sequence[str],
    slurm_settings: Mapping[str, Any],
    repo_root: Path | str = REPO_ROOT,
) -> list[str]:
    """Build the final publish submission command."""
    run_paths.publish_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "sbatch",
        "--parsable",
        "--chdir",
        str(Path(repo_root)),
        "--job-name",
        _job_name(run_paths.run_id, "publish"),
        "--output",
        str(run_paths.publish_dir / "slurm-%j.out"),
        "--error",
        str(run_paths.publish_dir / "slurm-%j.err"),
        "--export",
        _export_spec(run_paths),
    ]
    command.extend(_slurm_resource_args(slurm_settings))
    if dependency_ids:
        command.extend(["--dependency", _afterany_dependency(dependency_ids)])

    wrapped = shlex.join(build_publish_execution_command(run_paths.run_dir))
    command.extend(["--wrap", wrapped])
    return command


def build_task_execution_command(run_dir: Path | str, task_id: str) -> list[str]:
    """Build the compute-node command that executes one task."""
    return [
        "uv",
        "run",
        *DEFAULT_UV_EXTRAS,
        "-m",
        "benchmarks.nightly.slurm",
        "execute-task",
        "--run-dir",
        str(run_dir),
        "--task-id",
        task_id,
    ]


def build_publish_execution_command(run_dir: Path | str) -> list[str]:
    """Build the final publish command."""
    return [
        "uv",
        "run",
        *DEFAULT_UV_EXTRAS,
        "-m",
        "benchmarks.nightly.slurm",
        "run-publish",
        "--run-dir",
        str(run_dir),
    ]


def execute_task(run_dir: Path | str, task_id: str) -> int:
    """Execute one locked task from within a Slurm allocation."""
    run_paths = _load_run_paths_from_run_dir(run_dir)
    expanded_plan = _load_locked_plan(run_paths.plan_lock_path())
    task = _task_by_id(expanded_plan, task_id)
    dependency_ids = _existing_task_dependency_ids(run_paths, task_id)
    scheduler_context = _scheduler_context_from_env()

    result = run_task(
        run_paths,
        task,
        submission_id=os.environ.get("SLURM_JOB_ID"),
        dependency_ids=dependency_ids,
        scheduler_context=scheduler_context,
        working_dir=REPO_ROOT,
    )
    return 0 if result.status.state == "completed" else 1


def run_publish(run_dir: Path | str) -> int:
    """Run the final publish entrypoint after upstream Slurm tasks resolve."""
    run_paths = _load_run_paths_from_run_dir(run_dir)
    expanded_plan = _load_locked_plan(run_paths.plan_lock_path())
    run_manifest = json.loads(run_paths.run_manifest_path().read_text(encoding="utf-8"))

    publish_func, publish_error = _resolve_publish_callable()
    if publish_func is None:
        summary = _summarize_run_from_disk(
            run_paths,
            expanded_plan,
            publish_requested=bool(run_manifest.get("publish_requested")),
            publish_status="failed",
            publish_error=publish_error,
        )
        write_json(run_paths.publish_summary_path(), summary)
        _append_publish_log(
            run_paths,
            make_log_record(
                "publish_failed",
                run_id=run_paths.run_id,
                error_summary=publish_error,
            ),
        )
        run_manifest["publish_status"] = "failed"
        write_json(run_paths.run_manifest_path(), run_manifest)
        return 1

    try:
        publish_func(run_dir=run_paths.run_dir, publish_root=run_manifest.get("publish_root"))
    except Exception as exc:  # pragma: no cover - exercised once publish exists
        summary = _summarize_run_from_disk(
            run_paths,
            expanded_plan,
            publish_requested=bool(run_manifest.get("publish_requested")),
            publish_status="failed",
            publish_error=str(exc),
        )
        write_json(run_paths.publish_summary_path(), summary)
        _append_publish_log(
            run_paths,
            make_log_record(
                "publish_failed",
                run_id=run_paths.run_id,
                error_summary=str(exc),
            ),
        )
        run_manifest["publish_status"] = "failed"
        write_json(run_paths.run_manifest_path(), run_manifest)
        return 1

    summary = _summarize_run_from_disk(
        run_paths,
        expanded_plan,
        publish_requested=bool(run_manifest.get("publish_requested")),
        publish_status="completed",
    )
    write_json(run_paths.publish_summary_path(), summary)
    _append_publish_log(run_paths, make_log_record("publish_completed", run_id=run_paths.run_id))
    run_manifest["publish_status"] = "completed"
    write_json(run_paths.run_manifest_path(), run_manifest)
    return 0


def _write_task_submission_artifacts(
    run_paths: RunPaths,
    task: Mapping[str, Any],
    *,
    dependency_ids: Sequence[str],
) -> None:
    task_id = str(task["id"])
    task_dir = run_paths.task_dir(task_id)
    task_dir.mkdir(parents=True, exist_ok=True)

    manifest = TaskManifest(
        task_id=task_id,
        kind=str(task["kind"]),
        profile=str(task["profile"]),
        hardware_label=str(task["hardware_label"]),
        job_ids=[str(job["id"]) for job in task["jobs"]],
        dependency_ids=list(dependency_ids),
        output_dir=str(task_dir),
    )
    payload = manifest.to_record()
    payload["series"] = str(task["series"])
    payload["scenario"] = str(task["scenario"])
    payload["definition"] = json.loads(json.dumps(dict(task)))
    write_json(run_paths.task_manifest_path(task_id), payload)
    write_json(
        run_paths.task_status_path(task_id),
        StatusRecord(
            state="pending",
            artifact_paths=[str(run_paths.task_manifest_path(task_id)), str(run_paths.task_log_path(task_id))],
        ).to_record(),
    )


def _record_task_submission(
    run_paths: RunPaths,
    task: Mapping[str, Any],
    *,
    submission_id: str,
    dependency_ids: Sequence[str],
) -> None:
    task_id = str(task["id"])
    payload = json.loads(run_paths.task_manifest_path(task_id).read_text(encoding="utf-8"))
    payload["submission_id"] = submission_id
    payload["dependency_ids"] = list(dependency_ids)
    write_json(run_paths.task_manifest_path(task_id), payload)
    write_json(
        run_paths.task_status_path(task_id),
        StatusRecord(
            state="pending",
            artifact_paths=[str(run_paths.task_manifest_path(task_id)), str(run_paths.task_log_path(task_id))],
        ).to_record(),
    )
    _append_task_log(
        run_paths,
        task_id,
        make_log_record(
            "task_submitted",
            run_id=run_paths.run_id,
            task_id=task_id,
            hardware_label=str(task["hardware_label"]),
            submission_id=submission_id,
            dependency_ids=list(dependency_ids),
        ),
    )


def _mark_task_submission_failed(
    run_paths: RunPaths,
    task: Mapping[str, Any],
    *,
    dependency_ids: Sequence[str],
    error_summary: str,
) -> None:
    task_id = str(task["id"])
    write_json(
        run_paths.task_status_path(task_id),
        StatusRecord(
            state="failed",
            finished_at=utc_now_iso(),
            exit_code=1,
            error_summary=error_summary,
            failure_phase="submission",
            artifact_paths=[str(run_paths.task_manifest_path(task_id)), str(run_paths.task_log_path(task_id))],
        ).to_record(),
    )
    _append_task_log(
        run_paths,
        task_id,
        make_log_record(
            "task_submission_failed",
            run_id=run_paths.run_id,
            task_id=task_id,
            hardware_label=str(task["hardware_label"]),
            dependency_ids=list(dependency_ids),
            error_summary=error_summary,
        ),
    )


def _default_sbatch_runner(command: list[str]) -> str:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip() or "sbatch failed"
        raise RuntimeError(stderr)
    return _parse_submission_id(completed.stdout)


def _parse_submission_id(stdout: str) -> str:
    line = stdout.strip().splitlines()[0]
    return line.split(";", 1)[0]


def _load_locked_plan(plan_lock_path: Path) -> dict[str, Any]:
    with plan_lock_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _load_run_paths_from_run_dir(run_dir: Path | str) -> RunPaths:
    resolved_run_dir = Path(run_dir)
    run_manifest = json.loads((resolved_run_dir / "run.json").read_text(encoding="utf-8"))
    run_paths = resolve_run_paths(
        run_id=str(run_manifest["run_id"]),
        shared_state_dir=str(run_manifest["shared_state_dir"]),
        work_base_dir=str(run_manifest["work_base_dir"]),
        cache_env=dict(run_manifest["cache_env"]),
    )
    if run_paths.run_dir != resolved_run_dir:
        raise ValueError(f"Run directory does not match run manifest: {resolved_run_dir}")
    return run_paths


def _task_by_id(expanded_plan: Mapping[str, Any], task_id: str) -> dict[str, Any]:
    for task in expanded_plan["tasks"]:
        if task["id"] == task_id:
            return dict(task)
    raise ValueError(f"Unknown task id in locked plan: {task_id}")


def _existing_task_dependency_ids(run_paths: RunPaths, task_id: str) -> list[str]:
    manifest_path = run_paths.task_manifest_path(task_id)
    if not manifest_path.exists():
        return []
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return [str(item) for item in payload.get("dependency_ids", [])]


def _scheduler_context_from_env() -> dict[str, str]:
    field_map = {
        "job_id": "SLURM_JOB_ID",
        "job_name": "SLURM_JOB_NAME",
        "partition": "SLURM_JOB_PARTITION",
        "nodelist": "SLURMD_NODENAME",
        "cluster": "SLURM_CLUSTER_NAME",
        "submit_host": "SLURM_SUBMIT_HOST",
    }
    return {key: os.environ[value] for key, value in field_map.items() if os.environ.get(value)}


def _publish_slurm_settings(expanded_plan: Mapping[str, Any]) -> dict[str, Any]:
    task_profiles = [task["profile"] for task in expanded_plan["tasks"]]
    if not task_profiles:
        return {"time_limit": "00:15:00"}
    first_profile = expanded_plan["hardware_profiles"][task_profiles[0]].get("slurm", {})
    settings = {
        "account": first_profile.get("account"),
        "partition": first_profile.get("partition"),
        "time_limit": "00:15:00",
    }
    return {key: value for key, value in settings.items() if value}


def _slurm_resource_args(slurm_settings: Mapping[str, Any]) -> list[str]:
    command: list[str] = []
    mapping = (
        ("account", "--account"),
        ("partition", "--partition"),
        ("gres", "--gres"),
        ("time_limit", "--time"),
    )
    for field_name, flag in mapping:
        value = slurm_settings.get(field_name)
        if value:
            command.extend([flag, str(value)])
    exclude = slurm_settings.get("exclude")
    if exclude:
        if isinstance(exclude, Sequence) and not isinstance(exclude, str):
            exclude_value = ",".join(str(item) for item in exclude)
        else:
            exclude_value = str(exclude)
        command.extend(["--exclude", exclude_value])
    return command


def _export_spec(run_paths: RunPaths) -> str:
    items = ["ALL", "PYTHONUNBUFFERED=1", "UV_NO_CONFIG=1"]
    items.extend(f"{key}={value}" for key, value in run_paths.cache_env.items())
    return ",".join(items)


def _afterany_dependency(dependency_ids: Sequence[str]) -> str:
    return "afterany:" + ":".join(dependency_ids)


def _job_name(run_id: str, suffix: str) -> str:
    return f"nightly-{run_id[:19]}-{suffix}".replace("_", "-")


def _append_task_log(run_paths: RunPaths, task_id: str, line: str) -> None:
    log_path = run_paths.task_log_path(task_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _append_publish_log(run_paths: RunPaths, line: str) -> None:
    log_path = run_paths.publish_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _resolve_publish_callable() -> tuple[Callable[..., Any] | None, str | None]:
    try:
        publish_module = importlib.import_module("benchmarks.nightly.publish")
    except ModuleNotFoundError as exc:
        return None, f"benchmarks.nightly.publish is not implemented yet: {exc}"
    publish_func = getattr(publish_module, "publish_run", None)
    if publish_func is None:
        return None, "benchmarks.nightly.publish.publish_run is missing."
    return publish_func, None


def _summarize_run_from_disk(
    run_paths: RunPaths,
    expanded_plan: Mapping[str, Any],
    *,
    publish_requested: bool,
    publish_status: str,
    publish_error: str | None = None,
) -> dict[str, Any]:
    completed_tasks = 0
    failed_tasks = 0
    completed_jobs = 0
    failed_jobs = 0
    missing_artifacts: list[str] = []
    task_summaries: list[dict[str, Any]] = []

    jobs_by_id = {job["id"]: job for task in expanded_plan["tasks"] for job in task["jobs"]}
    for task in expanded_plan["tasks"]:
        task_id = str(task["id"])
        task_status_payload = json.loads(run_paths.task_status_path(task_id).read_text(encoding="utf-8"))
        task_state = str(task_status_payload["state"])
        if task_state == "completed":
            completed_tasks += 1
        elif task_state == "failed":
            failed_tasks += 1

        task_expected = [
            run_paths.task_manifest_path(task_id),
            run_paths.task_status_path(task_id),
            run_paths.task_log_path(task_id),
        ]
        missing_artifacts.extend(str(path) for path in task_expected if not path.exists())

        failed_task_jobs = 0
        for job in task["jobs"]:
            job_id = str(job["id"])
            job_status_path = run_paths.job_status_path(task_id, job_id)
            if job_status_path.exists():
                job_status_payload = json.loads(job_status_path.read_text(encoding="utf-8"))
                job_state = str(job_status_payload["state"])
            else:
                job_state = "pending"
            if job_state == "completed":
                completed_jobs += 1
            elif job_state == "failed":
                failed_jobs += 1
                failed_task_jobs += 1
            else:
                continue

            job_expected = [
                run_paths.job_manifest_path(task_id, job_id),
                job_status_path,
                run_paths.job_stderr_path(task_id, job_id),
            ]
            if run_paths.job_stdout_path(task_id, job_id).exists():
                job_expected.append(run_paths.job_stdout_path(task_id, job_id))
            if job_state == "completed":
                job_expected.extend(
                    required_artifact_paths(jobs_by_id[job_id], run_paths.job_results_dir(task_id, job_id))
                )
            missing_artifacts.extend(str(path) for path in job_expected if not path.exists())

        task_summaries.append(
            {
                "task_id": task_id,
                "kind": str(task["kind"]),
                "status": task_state,
                "failed_jobs": failed_task_jobs,
                "job_ids": [str(job["id"]) for job in task["jobs"]],
                "series": str(task["series"]),
            }
        )

    return {
        "run_id": run_paths.run_id,
        "run_dir": str(run_paths.run_dir),
        "plan_path": str(expanded_plan["plan_path"]),
        "plan_run_mode": expanded_plan["run_mode"],
        "selected_task_ids": [task["id"] for task in expanded_plan["tasks"]],
        "total_tasks": len(expanded_plan["tasks"]),
        "completed_tasks": completed_tasks,
        "failed_tasks": failed_tasks,
        "total_jobs": len(jobs_by_id),
        "completed_jobs": completed_jobs,
        "failed_jobs": failed_jobs,
        "artifacts_verified": not missing_artifacts,
        "missing_artifacts": sorted(set(missing_artifacts)),
        "publish": {
            "requested": publish_requested,
            "status": publish_status,
            "error": publish_error,
        },
        "tasks": task_summaries,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Submit or execute nightly Slurm tasks.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit_parser = subparsers.add_parser("submit", help="Submit the nightly plan to Slurm from the head node")
    submit_parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN_PATH, help="Path to nightly.yaml")
    submit_parser.add_argument("--run-mode", choices=RUN_MODES, default="full", help="Expanded task set to submit")
    submit_parser.add_argument(
        "--task-id", action="append", default=None, help="Restrict submission to specific task ids"
    )
    submit_parser.add_argument("--run-id", type=str, default=None, help="Override the generated run identifier")
    submit_parser.add_argument("--shared-state-dir", type=str, default=None, help="Override defaults.shared_state_dir")
    submit_parser.add_argument("--work-base-dir", type=str, default=None, help="Override defaults.work_base_dir")
    submit_parser.add_argument(
        "--publish-root",
        type=Path,
        default=None,
        help="Optional local nightly site root; skips git branch publication when set",
    )
    submit_parser.add_argument("--tmpdir", type=str, default=None, help="Override defaults.cache_env.TMPDIR")
    submit_parser.add_argument(
        "--uv-cache-dir", type=str, default=None, help="Override defaults.cache_env.UV_CACHE_DIR"
    )
    submit_parser.add_argument(
        "--uv-project-environment",
        type=str,
        default=None,
        help="Override defaults.cache_env.UV_PROJECT_ENVIRONMENT",
    )
    submit_parser.add_argument(
        "--warp-cache-path", type=str, default=None, help="Override defaults.cache_env.WARP_CACHE_PATH"
    )
    submit_parser.add_argument(
        "--newton-cache-path", type=str, default=None, help="Override defaults.cache_env.NEWTON_CACHE_PATH"
    )
    submit_parser.add_argument(
        "--cuda-cache-path", type=str, default=None, help="Override defaults.cache_env.CUDA_CACHE_PATH"
    )
    submit_parser.add_argument(
        "--publish", action="store_true", help="Submit the final publish job after task submissions"
    )
    submit_parser.add_argument("--skip-publish", action="store_true", help="Explicitly skip the final publish job")
    submit_parser.add_argument(
        "--submission-mode",
        choices=["parallel", "serial"],
        default="parallel",
        help="Submit tasks independently or chain them per hardware profile with afterany dependencies",
    )
    submit_parser.add_argument(
        "--cherry-pick-ref",
        action="append",
        default=None,
        help="Prepare a run-specific checkout and cherry-pick the given ref before task submission",
    )

    execute_parser = subparsers.add_parser(
        "execute-task", help="Internal compute-node entrypoint for one submitted task"
    )
    execute_parser.add_argument("--run-dir", type=Path, required=True, help="Path to the shared run directory")
    execute_parser.add_argument("--task-id", type=str, required=True, help="Task id from plan.lock.yaml")

    publish_parser = subparsers.add_parser("run-publish", help="Internal final publish entrypoint")
    publish_parser.add_argument("--run-dir", type=Path, required=True, help="Path to the shared run directory")
    return parser


def main() -> None:
    """CLI entrypoint for Slurm submission and execution."""
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "submit":
        if args.publish and args.skip_publish:
            parser.error("--publish and --skip-publish are mutually exclusive")
        summary = submit_slurm_nightly(
            plan_path=args.plan,
            run_mode=args.run_mode,
            task_ids=args.task_id,
            run_id=args.run_id,
            shared_state_dir=args.shared_state_dir,
            work_base_dir=args.work_base_dir,
            publish_root=args.publish_root,
            cache_env_overrides={
                "TMPDIR": args.tmpdir,
                "UV_CACHE_DIR": args.uv_cache_dir,
                "UV_PROJECT_ENVIRONMENT": args.uv_project_environment,
                "WARP_CACHE_PATH": args.warp_cache_path,
                "NEWTON_CACHE_PATH": args.newton_cache_path,
                "CUDA_CACHE_PATH": args.cuda_cache_path,
            },
            publish=args.publish and not args.skip_publish,
            submission_mode=args.submission_mode,
            cherry_pick_refs=args.cherry_pick_ref,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        if summary["failed_task_submissions"] or summary["publish"]["status"] == "failed":
            raise SystemExit(1)
        return

    if args.command == "execute-task":
        raise SystemExit(execute_task(args.run_dir, args.task_id))

    if args.command == "run-publish":
        raise SystemExit(run_publish(args.run_dir))

    parser.error(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
