# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Gather durable nightly artifacts into published dashboard data."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from benchmarks.nightly.common import make_log_record, resolve_run_paths

REPO_ROOT = Path(__file__).resolve().parents[2]

GitRunner = Callable[[Sequence[str], Path], subprocess.CompletedProcess[str]]


def publish_run(
    run_dir: Path | str,
    *,
    publish_root: Path | str | None = None,
    repo_root: Path | str = REPO_ROOT,
    git_runner: GitRunner | None = None,
    push_retries: int = 2,
) -> dict[str, Any]:
    """Publish one run into dashboard-facing JSONL artifacts.

    When ``publish_root`` is provided, the published files are written directly
    into that directory and no git synchronization is attempted. Otherwise the
    publisher stages and updates ``origin/<results_branch>`` from ``repo_root``.
    """
    context = _load_context(run_dir)
    _append_publish_log(
        context,
        make_log_record(
            "publish_started",
            run_id=context["run_id"],
            results_branch=context["results_branch"],
            publish_root=str(publish_root) if publish_root is not None else None,
        ),
    )

    publication = _build_publication(context)
    try:
        if publish_root is not None:
            site_root = Path(publish_root)
            _write_site(site_root, publication, repo_root=Path(repo_root))
            checkout_dir = None
        else:
            checkout_dir = _prepare_site_checkout(
                repo_root=Path(repo_root),
                results_branch=context["results_branch"],
                checkout_dir=context["publish_dir"] / "site-checkout",
                git_runner=git_runner,
            )
            site_root = checkout_dir / "nightly"
            _write_site(site_root, publication, repo_root=Path(repo_root))
            _commit_and_push_checkout(
                checkout_dir=checkout_dir,
                results_branch=context["results_branch"],
                run_id=context["run_id"],
                git_runner=git_runner,
                push_retries=push_retries,
            )
    except Exception as exc:
        _append_publish_log(
            context,
            make_log_record(
                "publish_failed",
                run_id=context["run_id"],
                error_summary=str(exc),
                publish_root=str(publish_root) if publish_root is not None else None,
            ),
        )
        raise

    _append_publish_log(
        context,
        make_log_record(
            "publish_materialized",
            run_id=context["run_id"],
            publish_root=str(site_root),
            run_rows=publication["run_rows"],
            point_count=len(publication["point_rows"]),
            render_count=len(publication["render_entries"]),
            profile_count=publication["profile_count"],
        ),
    )
    return {
        "run_id": context["run_id"],
        "publish_root": str(site_root),
        "site_checkout": str(checkout_dir) if checkout_dir is not None else None,
        "results_branch": context["results_branch"],
        "run_rows": publication["run_rows"],
        "run_row": publication["run_row"],
        "point_count": len(publication["point_rows"]),
        "render_count": len(publication["render_entries"]),
        "profile_count": publication["profile_count"],
    }


def build_points_rows(run_dir: Path | str) -> list[dict[str, Any]]:
    """Return published point rows for ``run_dir`` without writing them."""
    return _build_publication(_load_context(run_dir))["point_rows"]


def write_run_summary(run_dir: Path | str) -> dict[str, Any]:
    """Return the public per-run summary without writing it to a site root."""
    return _build_publication(_load_context(run_dir))["run_summary"]


def _build_publication(context: dict[str, Any]) -> dict[str, Any]:
    point_rows: list[dict[str, Any]] = []
    render_entries: list[dict[str, Any]] = []
    profile_artifacts: list[dict[str, Any]] = []
    scenarios: set[str] = set()
    series_names: set[str] = set()
    task_summaries: list[dict[str, Any]] = []
    sample_metadata: dict[str, Any] | None = None
    fallback_hardware_label = None
    gpu_groups: dict[str, dict[str, Any]] = {}

    completed_tasks = 0
    failed_tasks = 0
    completed_jobs = 0
    failed_jobs = 0

    for task in context["tasks"]:
        task_status = task["status"]
        task_definition = task["definition"]
        task_mode = _task_mode(task["manifest"], task_definition, task["jobs"])
        task_summary = {
            "task_id": task["task_id"],
            "kind": task["manifest"]["kind"],
            "scenario": task["manifest"].get("scenario"),
            "series": task["manifest"].get("series"),
            "status": task_status.get("state", "failed"),
            "job_ids": [],
            "completed_jobs": 0,
            "failed_jobs": 0,
            "submission_id": task["manifest"].get("submission_id"),
            "dependency_ids": task["manifest"].get("dependency_ids", []),
            "gpu": None,
            "hardware_label": task["manifest"].get("hardware_label"),
        }
        if fallback_hardware_label is None:
            fallback_hardware_label = task["manifest"].get("hardware_label")
        if task_summary["status"] == "completed":
            completed_tasks += 1
        elif task_summary["status"] == "failed":
            failed_tasks += 1

        task_rows: list[dict[str, Any]] = []
        task_renders: list[dict[str, Any]] = []
        for job in task["jobs"]:
            task_summary["job_ids"].append(job["job_id"])
            scenarios.add(str(job["manifest"]["scenario"]))
            series_names.add(str(job["manifest"]["series"]))
            metadata = job.get("metadata")
            task_summary["gpu"] = task_summary["gpu"] or _resolved_job_gpu(job)
            if sample_metadata is None and metadata and metadata.get("gpu"):
                sample_metadata = metadata

            job_state = job["status"].get("state")
            if job_state == "completed":
                completed_jobs += 1
                task_summary["completed_jobs"] += 1
            elif job_state == "failed":
                failed_jobs += 1
                task_summary["failed_jobs"] += 1
            else:
                continue

            if job["manifest"]["kind"] == "benchmark":
                task_rows.extend(
                    _build_job_point_rows(
                        context=context,
                        task=task,
                        job=job,
                        task_mode=task_mode,
                    )
                )
                profile_artifacts.extend(_build_profile_artifacts(job))
            elif job["manifest"]["kind"] == "render":
                render_entry = _build_render_entry(context=context, task=task, job=job)
                if render_entry is not None:
                    task_renders.append(render_entry)

        point_rows.extend(task_rows)
        render_entries.extend(task_renders)
        task_summaries.append(task_summary)
        gpu_key = task_summary["hardware_label"] or task_summary["gpu"] or "unknown"
        gpu_group = gpu_groups.setdefault(
            gpu_key,
            {
                "hardware_label": task_summary["hardware_label"],
                "display_gpu": task_summary["gpu"],
                "sample_metadata": None,
                "scenarios": set(),
                "series": set(),
                "completed_tasks": 0,
                "failed_tasks": 0,
                "completed_jobs": 0,
                "failed_jobs": 0,
            },
        )
        if gpu_group["sample_metadata"] is None:
            for job in task["jobs"]:
                metadata = job.get("metadata")
                if metadata and metadata.get("gpu"):
                    gpu_group["sample_metadata"] = metadata
                    gpu_group["display_gpu"] = metadata.get("gpu")
                    break
        gpu_group["scenarios"].add(task_summary["scenario"])
        gpu_group["series"].add(task_summary["series"])
        gpu_group["completed_jobs"] += task_summary["completed_jobs"]
        gpu_group["failed_jobs"] += task_summary["failed_jobs"]
        if task_summary["status"] == "completed":
            gpu_group["completed_tasks"] += 1
        elif task_summary["status"] == "failed":
            gpu_group["failed_tasks"] += 1

    run_summary = {
        "run_id": context["run_id"],
        "run_dir": f"runs/{context['run_id']}",
        "timestamp": context["run_timestamp"],
        "commit": context["commit"],
        "commit_short": context["commit_short"],
        "execution_mode": context["run_manifest"].get("mode"),
        "plan_run_mode": context["run_manifest"].get("plan_run_mode"),
        "status": "failed" if failed_tasks or failed_jobs else "completed",
        "selected_task_ids": list(context["run_manifest"].get("selected_task_ids", [])),
        "scenarios": sorted(scenarios),
        "series": sorted(series_names),
        "total_tasks": len(task_summaries),
        "completed_tasks": completed_tasks,
        "failed_tasks": failed_tasks,
        "total_jobs": completed_jobs + failed_jobs,
        "completed_jobs": completed_jobs,
        "failed_jobs": failed_jobs,
        "point_count": len(point_rows),
        "successful_point_count": sum(1 for row in point_rows if row.get("ok") is not False),
        "failed_point_count": sum(1 for row in point_rows if row.get("ok") is False),
        "render_count": len(render_entries),
        "profile_count": len({artifact["job_id"] for artifact in profile_artifacts}),
        "tasks": task_summaries,
    }
    if sample_metadata is not None:
        for key in ("gpu", "gpu_memory_total_gb", "platform", "python_version"):
            if sample_metadata.get(key) is not None:
                run_summary[key] = sample_metadata[key]
    elif fallback_hardware_label:
        run_summary["gpu"] = fallback_hardware_label

    run_rows = []
    for gpu_key, gpu_group in sorted(gpu_groups.items()):
        successful_points = [
            row for row in point_rows if row.get("hardware_label") == gpu_key and row.get("ok") is not False
        ]
        failed_point_rows = [
            row for row in point_rows if row.get("hardware_label") == gpu_key and row.get("ok") is False
        ]
        gpu_renders = [row for row in render_entries if row.get("gpu_tag") == gpu_key]
        run_row = {
            "run_id": context["run_id"],
            "timestamp": context["run_timestamp"],
            "commit": context["commit"],
            "commit_short": context["commit_short"],
            "run_dir": f"runs/{context['run_id']}",
            "execution_mode": context["run_manifest"].get("mode"),
            "plan_run_mode": context["run_manifest"].get("plan_run_mode"),
            "status": "failed" if gpu_group["failed_tasks"] or gpu_group["failed_jobs"] else "completed",
            "failed_tasks": gpu_group["failed_tasks"],
            "failed_jobs": gpu_group["failed_jobs"],
            "point_count": len(successful_points),
            "failed_point_count": len(failed_point_rows),
            "render_count": len(gpu_renders),
            "scenarios": sorted(gpu_group["scenarios"]),
            "gpu": gpu_group["display_gpu"] or gpu_key,
            "gpu_tag": gpu_group["hardware_label"],
        }
        sample = gpu_group["sample_metadata"]
        if sample is not None:
            for key in ("gpu_memory_total_gb", "platform", "python_version"):
                if sample.get(key) is not None:
                    run_row[key] = sample[key]
        run_rows.append(run_row)

    if not run_rows:
        run_rows.append(
            {
                "run_id": context["run_id"],
                "timestamp": context["run_timestamp"],
                "commit": context["commit"],
                "commit_short": context["commit_short"],
                "run_dir": f"runs/{context['run_id']}",
                "execution_mode": context["run_manifest"].get("mode"),
                "plan_run_mode": context["run_manifest"].get("plan_run_mode"),
                "status": run_summary["status"],
                "failed_tasks": failed_tasks,
                "failed_jobs": failed_jobs,
                "point_count": run_summary["successful_point_count"],
                "failed_point_count": run_summary["failed_point_count"],
                "render_count": len(render_entries),
                "scenarios": run_summary["scenarios"],
                "gpu": sample_metadata.get("gpu") if sample_metadata else fallback_hardware_label,
            }
        )
    run_row = run_rows[0]

    point_rows.sort(
        key=lambda row: (
            str(row.get("series", "")),
            str(row.get("scenario", "")),
            str(row.get("mode", "")),
            str(row.get("solver", "")),
            int(row.get("substeps") or 0),
            int(row.get("num_worlds") or 0),
            str(row.get("task_id", "")),
            str(row.get("job_id", "")),
            int(row.get("step_index") or -1),
            0 if row.get("ok") is not False else 1,
        )
    )
    render_entries.sort(
        key=lambda row: (
            str(row.get("scenario", "")),
            str(row.get("series", "")),
            str(row.get("solver", "")),
            int(row.get("substeps") or 0),
            str(row.get("task_id", "")),
            str(row.get("job_id", "")),
        )
    )

    return {
        "run_rows": run_rows,
        "run_row": run_row,
        "run_summary": run_summary,
        "point_rows": point_rows,
        "render_entries": render_entries,
        "profile_artifacts": profile_artifacts,
        "profile_count": len({artifact["job_id"] for artifact in profile_artifacts}),
    }


def _build_job_point_rows(
    *,
    context: dict[str, Any],
    task: Mapping[str, Any],
    job: Mapping[str, Any],
    task_mode: str,
) -> list[dict[str, Any]]:
    rows = list(job.get("measurements", []))
    if not rows:
        return [
            _base_point_row(
                context=context,
                task=task,
                job=job,
                task_mode=task_mode,
                ok=False,
                error_summary=job["status"].get("error_summary") or "missing measurements.jsonl",
                failure_phase=job["status"].get("failure_phase"),
            )
        ]

    point_rows = []
    for measurement in rows:
        row = _base_point_row(context=context, task=task, job=job, task_mode=task_mode, ok=measurement.get("ok", True))
        row.update(measurement)
        profile_summary = job.get("profile_summary")
        if isinstance(profile_summary, Mapping) and not row.get("kernels"):
            kernels = profile_summary.get("kernels")
            if isinstance(kernels, Mapping):
                row["kernels"] = {str(name): float(value) for name, value in kernels.items()}
        point_rows.append(row)
    return point_rows


def _build_render_entry(
    *,
    context: dict[str, Any],
    task: Mapping[str, Any],
    job: Mapping[str, Any],
) -> dict[str, Any] | None:
    render_meta = job.get("render_meta")
    if render_meta is None:
        return None
    video_name = render_meta.get("video")
    if not isinstance(video_name, str) or not video_name:
        return None

    public_path = Path("renders") / str(job["job_id"]) / video_name
    entry = {
        "run_id": context["run_id"],
        "run_timestamp": context["run_timestamp"],
        "task_id": task["task_id"],
        "job_id": job["job_id"],
        "series": task["manifest"].get("series"),
        "scenario": job["manifest"]["scenario"],
        "solver": job["manifest"]["params"].get("solver"),
        "substeps": job["manifest"]["params"].get("substeps"),
        "label": render_meta.get("label"),
        "fps": render_meta.get("fps"),
        "frames": render_meta.get("frames"),
        "width": render_meta.get("width"),
        "height": render_meta.get("height"),
        "path": str(public_path),
        "gpu": _resolved_job_gpu(job),
        "gpu_tag": job["manifest"].get("hardware_label"),
    }
    entry["_source_path"] = str(job["results_dir"] / video_name)
    return entry


def _base_point_row(
    *,
    context: dict[str, Any],
    task: Mapping[str, Any],
    job: Mapping[str, Any],
    task_mode: str,
    ok: bool,
    error_summary: str | None = None,
    failure_phase: str | None = None,
) -> dict[str, Any]:
    params = job["manifest"]["params"]
    metadata = job.get("metadata") or {}
    row = {
        "run_id": context["run_id"],
        "run_timestamp": context["run_timestamp"],
        "timestamp": metadata.get("timestamp", context["run_timestamp"]),
        "commit": context["commit"],
        "commit_short": context["commit_short"],
        "task_id": task["task_id"],
        "job_id": job["job_id"],
        "series": task["manifest"].get("series"),
        "mode": task_mode,
        "scenario": job["manifest"]["scenario"],
        "solver": params.get("solver"),
        "substeps": params.get("substeps"),
        "num_worlds": params.get("num_worlds"),
        "label": params.get("ablation_label"),
        "step_index": params.get("ablation_step_index"),
        "gpu": metadata.get("gpu"),
        "gpu_tag": job["manifest"].get("hardware_label"),
        "gpu_memory_total_gb": metadata.get("gpu_memory_total_gb"),
        "platform": metadata.get("platform"),
        "python_version": metadata.get("python_version"),
        "hardware_label": job["manifest"].get("hardware_label"),
        "job_status": job["status"].get("state"),
        "ok": ok,
    }
    profile_meta = job.get("profile_meta")
    if profile_meta is not None:
        report_name = profile_meta.get("report_name")
        trace_name = profile_meta.get("trace_name")
        if isinstance(report_name, str) and report_name:
            row["nsys_report_path"] = str(_public_profile_path(job, report_name))
        if isinstance(trace_name, str) and trace_name:
            row["nsys_trace_path"] = str(_public_profile_path(job, trace_name))
    profile_summary = job.get("profile_summary")
    if isinstance(profile_summary, Mapping):
        kernels = profile_summary.get("kernels")
        if isinstance(kernels, Mapping):
            row["kernels"] = {str(name): float(value) for name, value in kernels.items()}
    if metadata.get("pgs_iterations") is not None:
        row["pgs_iterations"] = metadata["pgs_iterations"]
    if metadata.get("measure_frames") is not None:
        row["measure_frames"] = metadata["measure_frames"]
    if metadata.get("warmup_frames") is not None:
        row["warmup_frames"] = metadata["warmup_frames"]
    if error_summary is not None:
        row["error_summary"] = error_summary
    if failure_phase is not None:
        row["failure_phase"] = failure_phase
    if row["gpu"] is None:
        row["gpu"] = job["manifest"].get("hardware_label")
    return row


def _resolved_job_gpu(job: Mapping[str, Any]) -> str:
    metadata = job.get("metadata") or {}
    return str(metadata.get("gpu") or job["manifest"].get("hardware_label") or "unknown")


def _build_profile_artifacts(job: Mapping[str, Any]) -> list[dict[str, Any]]:
    profile_meta = job.get("profile_meta")
    if profile_meta is None:
        return []

    artifacts = []
    for key in ("report_name", "trace_name"):
        file_name = profile_meta.get(key)
        if not isinstance(file_name, str) or not file_name:
            continue
        source_path = job["results_dir"] / file_name
        if not source_path.is_file():
            continue
        artifacts.append(
            {
                "job_id": job["job_id"],
                "path": str(_public_profile_path(job, file_name)),
                "_source_path": str(source_path),
            }
        )
    return artifacts


def _public_profile_path(job: Mapping[str, Any], file_name: str | None) -> Path:
    if not isinstance(file_name, str) or not file_name:
        raise ValueError("Profile artifact file name must be a non-empty string.")
    params = job["manifest"]["params"]
    label = params.get("ablation_label") or params.get("solver") or "profile"
    scenario = job["manifest"].get("scenario") or "scenario"
    num_worlds = params.get("num_worlds")
    substeps = params.get("substeps")
    extension = _profile_extension(file_name)
    public_name = (
        "--".join(
            [
                str(job["job_id"]),
                _slug_fragment(str(scenario)),
                _slug_fragment(str(label)),
                f"s{substeps}",
                f"n{num_worlds}",
            ]
        )
        + extension
    )
    return Path("profiles") / str(job["job_id"]) / public_name


def _profile_extension(file_name: str) -> str:
    if file_name.endswith(".trace.json"):
        return ".trace.json"
    if file_name.endswith(".nsys-rep"):
        return ".nsys-rep"
    return Path(file_name).suffix or ""


def _slug_fragment(value: str) -> str:
    cleaned = []
    previous_dash = False
    for char in value.lower():
        if char.isalnum():
            cleaned.append(char)
            previous_dash = False
            continue
        if not previous_dash:
            cleaned.append("-")
            previous_dash = True
    slug = "".join(cleaned).strip("-")
    return slug or "item"


def _task_mode(
    task_manifest: Mapping[str, Any],
    task_definition: Mapping[str, Any],
    jobs: Sequence[Mapping[str, Any]],
) -> str:
    if task_manifest["kind"] != "benchmark":
        return "render"
    if task_definition.get("ablation_sequence"):
        return "ablation"
    for job in jobs:
        params = job.get("manifest", {}).get("params", {})
        if params.get("ablation_step_index") is not None or params.get("ablation_label"):
            return "ablation"
    return "sweep"


def _write_site(site_root: Path, publication: Mapping[str, Any], *, repo_root: Path) -> None:
    site_root.mkdir(parents=True, exist_ok=True)
    runs_dir = site_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    existing_runs = _read_jsonl(site_root / "runs.jsonl")
    updated_runs = [row for row in existing_runs if row.get("run_id") != publication["run_row"]["run_id"]]
    updated_runs.extend(dict(row) for row in publication["run_rows"])
    updated_runs.sort(key=lambda row: str(row.get("timestamp", "")), reverse=True)
    _write_jsonl(site_root / "runs.jsonl", updated_runs)

    existing_points = _read_jsonl(site_root / "points.jsonl")
    updated_points = [row for row in existing_points if row.get("run_id") != publication["run_row"]["run_id"]]
    updated_points.extend(dict(row) for row in publication["point_rows"])
    updated_points.sort(
        key=lambda row: (str(row.get("run_timestamp", "")), str(row.get("task_id", "")), str(row.get("job_id", "")))
    )
    _write_jsonl(site_root / "points.jsonl", updated_points)

    public_run_dir = runs_dir / str(publication["run_row"]["run_id"])
    if public_run_dir.exists():
        shutil.rmtree(public_run_dir)
    public_run_dir.mkdir(parents=True, exist_ok=True)

    _write_json(public_run_dir / "meta.json", publication["run_row"])
    _write_json(public_run_dir / "summary.json", publication["run_summary"])
    render_entries = [dict(entry) for entry in publication["render_entries"]]
    for entry in render_entries:
        source_path = Path(entry.pop("_source_path"))
        target_path = public_run_dir / entry["path"]
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
    _write_json_value(public_run_dir / "renders.json", render_entries)

    for artifact in publication["profile_artifacts"]:
        source_path = Path(artifact["_source_path"])
        target_path = public_run_dir / artifact["path"]
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)

    dashboard_source = repo_root / "benchmarks" / "nightly" / "index.html"
    if dashboard_source.is_file():
        shutil.copy2(dashboard_source, site_root / "index.html")

    nojekyll_path = site_root.parent / ".nojekyll"
    nojekyll_path.write_text("", encoding="utf-8")


def _prepare_site_checkout(
    *,
    repo_root: Path,
    results_branch: str,
    checkout_dir: Path,
    git_runner: GitRunner | None,
) -> Path:
    runner = git_runner or _default_git_runner
    if checkout_dir.exists():
        shutil.rmtree(checkout_dir)
    checkout_dir.parent.mkdir(parents=True, exist_ok=True)

    remote_url = _git_stdout(runner, ["git", "remote", "get-url", "origin"], repo_root)
    branch_exists = _git_remote_branch_exists(runner, repo_root, results_branch)
    if branch_exists:
        _run_git(
            runner,
            ["git", "clone", "--branch", results_branch, "--single-branch", remote_url, str(checkout_dir)],
            repo_root,
        )
        return checkout_dir

    _run_git(runner, ["git", "clone", remote_url, str(checkout_dir)], repo_root)
    _run_git(runner, ["git", "checkout", "--orphan", results_branch], checkout_dir)
    for child in checkout_dir.iterdir():
        if child.name == ".git":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    return checkout_dir


def _commit_and_push_checkout(
    *,
    checkout_dir: Path,
    results_branch: str,
    run_id: str,
    git_runner: GitRunner | None,
    push_retries: int,
) -> None:
    runner = git_runner or _default_git_runner
    _run_git(runner, ["git", "add", "nightly", ".nojekyll"], checkout_dir)
    if _git_has_no_staged_changes(runner, checkout_dir):
        return

    _run_git(
        runner,
        ["git", "commit", "-m", f"Update nightly benchmarks for {run_id}"],
        checkout_dir,
    )
    for attempt in range(push_retries + 1):
        try:
            if _git_remote_branch_exists(runner, checkout_dir, results_branch):
                _run_git(runner, ["git", "pull", "--rebase", "origin", results_branch], checkout_dir)
            _run_git(runner, ["git", "push", "-u", "origin", results_branch], checkout_dir)
            return
        except RuntimeError:
            if attempt >= push_retries:
                raise


def _load_context(run_dir: Path | str) -> dict[str, Any]:
    resolved_run_dir = Path(run_dir)
    run_manifest = _read_json(resolved_run_dir / "run.json")
    run_paths = resolve_run_paths(
        run_id=str(run_manifest["run_id"]),
        shared_state_dir=str(run_manifest["shared_state_dir"]),
        work_base_dir=str(run_manifest["work_base_dir"]),
        cache_env=dict(run_manifest["cache_env"]),
    )
    if run_paths.run_dir != resolved_run_dir:
        raise ValueError(f"Run directory does not match run manifest: {resolved_run_dir}")

    plan_lock = _read_yaml(run_paths.plan_lock_path())
    tasks = []
    for task in plan_lock.get("tasks", []):
        task_id = str(task["id"])
        task_manifest = _read_json(run_paths.task_manifest_path(task_id))
        task_status = _read_json_if_exists(run_paths.task_status_path(task_id)) or {"state": "pending"}
        jobs = []
        for job in task.get("jobs", []):
            job_id = str(job["id"])
            job_manifest = _read_json_if_exists(run_paths.job_manifest_path(task_id, job_id)) or _planned_job_manifest(
                task, job
            )
            job_status = _read_json_if_exists(run_paths.job_status_path(task_id, job_id)) or {"state": "pending"}
            results_dir = run_paths.job_results_dir(task_id, job_id)
            jobs.append(
                {
                    "job_id": job_id,
                    "manifest": job_manifest,
                    "status": job_status,
                    "results_dir": results_dir,
                    "metadata": _read_json_if_exists(results_dir / "metadata.json"),
                    "measurements": _read_jsonl_if_exists(results_dir / "measurements.jsonl"),
                    "render_meta": _read_json_if_exists(results_dir / "render_meta.json"),
                    "profile_meta": _read_json_if_exists(results_dir / "profile_meta.json"),
                    "profile_summary": _read_json_if_exists(results_dir / "profile_summary.json"),
                }
            )
        tasks.append(
            {
                "task_id": task_id,
                "definition": dict(task),
                "manifest": task_manifest,
                "status": task_status,
                "jobs": jobs,
            }
        )

    run_timestamp = str(run_manifest.get("created_at") or run_manifest.get("timestamp") or run_manifest["run_id"])
    commit = str(run_manifest.get("revision") or "unknown")
    return {
        "run_id": str(run_manifest["run_id"]),
        "run_timestamp": run_timestamp,
        "commit": commit,
        "commit_short": commit[:7],
        "results_branch": str(plan_lock.get("defaults", {}).get("results_branch", "gh-pages")),
        "run_manifest": run_manifest,
        "run_paths": run_paths,
        "publish_dir": run_paths.publish_dir,
        "tasks": tasks,
    }


def _planned_job_manifest(task: Mapping[str, Any], job: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "job_id": str(job["id"]),
        "task_id": str(task["id"]),
        "kind": str(job["kind"]),
        "scenario": str(job["scenario"]),
        "series": str(job["series"]),
        "profile": str(job["profile"]),
        "hardware_label": str(job["hardware_label"]),
        "params": {
            key: value
            for key, value in dict(job).items()
            if key not in {"id", "task_id", "kind", "profile", "hardware_label", "series", "scenario"}
        },
        "output_dir": "",
        "scheduler": None,
    }


def _append_publish_log(context: Mapping[str, Any], line: str) -> None:
    log_path = Path(context["publish_dir"]) / "publish.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _read_json(path)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return _parse_jsonl(path.read_text(encoding="utf-8"))


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return _parse_jsonl(path.read_text(encoding="utf-8"))


def _parse_jsonl(text: str) -> list[dict[str, Any]]:
    rows = []
    for line in text.splitlines():
        raw = line.strip()
        if not raw:
            continue
        rows.append(json.loads(raw))
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _write_json_value(path, dict(payload))


def _write_json_value(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(dict(row), sort_keys=True) for row in rows]
    suffix = "\n" if lines else ""
    path.write_text("\n".join(lines) + suffix, encoding="utf-8")


def _default_git_runner(command: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(command), cwd=cwd, capture_output=True, text=True, check=False)


def _run_git(runner: GitRunner, command: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    completed = runner(command, cwd)
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip() or "git command failed"
        raise RuntimeError(f"{' '.join(command)}: {stderr}")
    return completed


def _git_stdout(runner: GitRunner, command: Sequence[str], cwd: Path) -> str:
    return _run_git(runner, command, cwd).stdout.strip()


def _git_remote_branch_exists(runner: GitRunner, cwd: Path, branch: str) -> bool:
    completed = runner(["git", "ls-remote", "--exit-code", "--heads", "origin", branch], cwd)
    return completed.returncode == 0


def _git_has_no_staged_changes(runner: GitRunner, checkout_dir: Path) -> bool:
    completed = runner(["git", "diff", "--staged", "--quiet"], checkout_dir)
    return completed.returncode == 0
