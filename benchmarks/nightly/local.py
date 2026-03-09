# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Local nightly driver built on the shared plan and executor layers."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmarks.nightly.common import (
    RunManifest,
    make_run_id,
    prepare_execution_repo,
    resolve_run_paths,
    utc_now_iso,
    validate_run_environment,
    write_json,
)
from benchmarks.nightly.executor import TaskExecutionResult, required_artifact_paths, run_task
from benchmarks.nightly.plan import DEFAULT_PLAN_PATH, RUN_MODES, expand_plan, load_plan, write_plan_lock

REPO_ROOT = Path(__file__).resolve().parents[2]


def run_local_nightly(
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
    working_dir: Path | str | None = None,
    source_repo_root: Path | str = REPO_ROOT,
    repo_preparer=prepare_execution_repo,
    runner=None,
) -> dict[str, Any]:
    """Run the local nightly flow and return a JSON-serializable summary."""
    loaded_plan = load_plan(plan_path)
    _apply_plan_overrides(
        loaded_plan,
        shared_state_dir=shared_state_dir,
        work_base_dir=work_base_dir,
        cache_env_overrides=cache_env_overrides,
    )
    expanded_plan = expand_plan(loaded_plan, run_mode=run_mode, selected_task_ids=task_ids)
    publish_func = _resolve_publish_callable(publish)

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
        source_repo_root=source_repo_root,
        run_dir=run_paths.run_dir,
        cherry_pick_refs=cherry_pick_refs,
    )

    run_manifest = RunManifest(
        run_id=effective_run_id,
        mode="local",
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
        base_revision=prepared_repo.base_revision,
        execution_repo_root=str(prepared_repo.repo_root),
        cherry_pick_refs=list(prepared_repo.cherry_pick_refs),
        resolved_cherry_pick_refs=list(prepared_repo.resolved_cherry_pick_refs),
    )
    write_json(run_paths.run_manifest_path(), run_manifest.to_record())
    write_plan_lock(expanded_plan, run_paths.plan_lock_path())

    task_results = []
    execution_working_dir = (
        prepared_repo.repo_root if cherry_pick_refs else Path(working_dir) if working_dir is not None else REPO_ROOT
    )
    for task in expanded_plan["tasks"]:
        task_results.append(
            run_task(
                run_paths,
                task,
                working_dir=execution_working_dir,
                runner=runner,
            )
        )

    publish_status = "skipped"
    if publish_func is not None:
        try:
            publish_func(run_dir=run_paths.run_dir, publish_root=publish_root)
            publish_status = "completed"
        except Exception as exc:  # pragma: no cover - exercised when publish exists
            publish_status = "failed"
            summary = _build_run_summary(
                run_paths,
                expanded_plan,
                task_results,
                publish_requested=True,
                publish_status=publish_status,
                publish_error=str(exc),
            )
            write_json(run_paths.publish_summary_path(), summary)
            _update_run_manifest_publish_status(run_paths, publish_status)
            return summary

    summary = _build_run_summary(
        run_paths,
        expanded_plan,
        task_results,
        publish_requested=publish,
        publish_status=publish_status,
    )
    write_json(run_paths.publish_summary_path(), summary)
    _update_run_manifest_publish_status(run_paths, publish_status)
    return summary


def _prepare_run_directories(run_paths: RunPaths) -> None:
    run_paths.run_dir.mkdir(parents=True, exist_ok=True)
    run_paths.publish_dir.mkdir(parents=True, exist_ok=True)
    run_paths.tasks_dir.mkdir(parents=True, exist_ok=True)
    run_paths.run_work_dir().mkdir(parents=True, exist_ok=True)


def _apply_plan_overrides(
    plan: dict[str, Any],
    *,
    shared_state_dir: str | None,
    work_base_dir: str | None,
    cache_env_overrides: Mapping[str, str] | None,
) -> None:
    if shared_state_dir is not None:
        plan["defaults"]["shared_state_dir"] = shared_state_dir
    if work_base_dir is not None:
        plan["defaults"]["work_base_dir"] = work_base_dir
    if cache_env_overrides is not None:
        for key, value in cache_env_overrides.items():
            if value is not None:
                plan["defaults"]["cache_env"][key] = value


def _resolve_publish_callable(publish_requested: bool):
    if not publish_requested:
        return None
    try:
        publish_module = importlib.import_module("benchmarks.nightly.publish")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Local publish was requested, but benchmarks.nightly.publish is not implemented yet."
        ) from exc

    publish_func = getattr(publish_module, "publish_run", None)
    if publish_func is None:
        raise RuntimeError("Local publish was requested, but benchmarks.nightly.publish.publish_run is missing.")
    return publish_func


def _build_run_summary(
    run_paths: RunPaths,
    expanded_plan: Mapping[str, Any],
    task_results: Sequence[TaskExecutionResult],
    *,
    publish_requested: bool,
    publish_status: str,
    publish_error: str | None = None,
) -> dict[str, Any]:
    tasks_by_id = {task["id"]: task for task in expanded_plan["tasks"]}
    jobs_by_id = {job["id"]: job for task in expanded_plan["tasks"] for job in task["jobs"]}
    missing_artifacts: list[str] = []
    completed_tasks = 0
    failed_tasks = 0
    completed_jobs = 0
    failed_jobs = 0

    for task_result in task_results:
        task_id = task_result.task_manifest.task_id
        if task_result.status.state == "completed":
            completed_tasks += 1
        else:
            failed_tasks += 1

        task_expected = (
            run_paths.task_manifest_path(task_id),
            run_paths.task_status_path(task_id),
            run_paths.task_log_path(task_id),
        )
        missing_artifacts.extend(str(path) for path in task_expected if not path.exists())

        for job_result in task_result.job_results:
            job_id = job_result.job_manifest.job_id
            job_expected = [
                run_paths.job_manifest_path(task_id, job_id),
                run_paths.job_status_path(task_id, job_id),
                run_paths.job_stderr_path(task_id, job_id),
            ]
            if run_paths.job_stdout_path(task_id, job_id).exists():
                job_expected.append(run_paths.job_stdout_path(task_id, job_id))

            if job_result.status.state == "completed":
                completed_jobs += 1
                job_expected.extend(
                    required_artifact_paths(jobs_by_id[job_id], Path(job_result.job_manifest.output_dir))
                )
            else:
                failed_jobs += 1

            missing_artifacts.extend(str(path) for path in job_expected if not path.exists())

    return {
        "run_id": run_paths.run_id,
        "run_dir": str(run_paths.run_dir),
        "plan_path": str(expanded_plan["plan_path"]),
        "plan_run_mode": expanded_plan["run_mode"],
        "selected_task_ids": [task["id"] for task in expanded_plan["tasks"]],
        "total_tasks": len(task_results),
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
        "tasks": [
            {
                "task_id": task_result.task_manifest.task_id,
                "kind": task_result.task_manifest.kind,
                "status": task_result.status.state,
                "failed_jobs": sum(1 for job in task_result.job_results if job.status.state == "failed"),
                "job_ids": [job.job_manifest.job_id for job in task_result.job_results],
                "series": tasks_by_id[task_result.task_manifest.task_id]["series"],
            }
            for task_result in task_results
        ],
    }


def _update_run_manifest_publish_status(run_paths: RunPaths, publish_status: str) -> None:
    payload = json.loads(run_paths.run_manifest_path().read_text(encoding="utf-8"))
    payload["publish_status"] = publish_status
    write_json(run_paths.run_manifest_path(), payload)


def _git_revision() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the nightly plan locally via the shared planner and executor.")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN_PATH, help="Path to nightly.yaml")
    parser.add_argument("--run-mode", choices=RUN_MODES, default="full", help="Expanded task set to run")
    parser.add_argument("--task-id", action="append", default=None, help="Restrict execution to specific task ids")
    parser.add_argument("--run-id", type=str, default=None, help="Override the generated run identifier")
    parser.add_argument("--shared-state-dir", type=str, default=None, help="Override defaults.shared_state_dir")
    parser.add_argument("--work-base-dir", type=str, default=None, help="Override defaults.work_base_dir")
    parser.add_argument(
        "--publish-root",
        type=Path,
        default=None,
        help="Optional local nightly site root; skips git branch publication when set",
    )
    parser.add_argument("--tmpdir", type=str, default=None, help="Override defaults.cache_env.TMPDIR")
    parser.add_argument("--uv-cache-dir", type=str, default=None, help="Override defaults.cache_env.UV_CACHE_DIR")
    parser.add_argument(
        "--uv-project-environment",
        type=str,
        default=None,
        help="Override defaults.cache_env.UV_PROJECT_ENVIRONMENT",
    )
    parser.add_argument("--warp-cache-path", type=str, default=None, help="Override defaults.cache_env.WARP_CACHE_PATH")
    parser.add_argument(
        "--cherry-pick-ref",
        action="append",
        default=None,
        help="Prepare a run-specific checkout and cherry-pick the given ref before executing jobs",
    )
    parser.add_argument("--publish", action="store_true", help="Run gather-and-publish after local execution")
    parser.add_argument("--skip-publish", action="store_true", help="Explicitly skip gather-and-publish")
    return parser


def main() -> None:
    """CLI entrypoint for local nightly execution."""
    parser = _build_arg_parser()
    args = parser.parse_args()

    if args.publish and args.skip_publish:
        parser.error("--publish and --skip-publish are mutually exclusive")

    publish = args.publish and not args.skip_publish
    summary = run_local_nightly(
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
        },
        cherry_pick_refs=args.cherry_pick_ref,
        publish=publish,
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["failed_tasks"] or not summary["artifacts_verified"] or summary["publish"]["status"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
