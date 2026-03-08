# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Shared nightly artifact contracts, path helpers, and parseable log records.

Nightly orchestration uses the following terms consistently:

- Task: an authored unit in the nightly plan that may expand to multiple jobs.
- Job: one concrete benchmark or render invocation with fully resolved parameters.
- Submission: the local process or Slurm allocation that executes a task's jobs.
- Measurement: a raw benchmark row emitted by one completed job.
- Series: a stable published grouping used by dashboard charts.
- Display: a chart or comparison definition owned by the dashboard, not by the plan.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import os
import string
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

CACHE_ENV_KEYS = ("TMPDIR", "UV_CACHE_DIR", "UV_PROJECT_ENVIRONMENT", "WARP_CACHE_PATH")
StatusState = Literal["pending", "running", "completed", "failed"]


def expand_env_string(value: str, env: Mapping[str, str] | None = None) -> str:
    """Expand ``$VARS`` and ``~`` in a plan-managed string."""
    env_map = os.environ if env is None else env
    return os.path.expanduser(string.Template(value).safe_substitute(env_map))


def expand_env_path(value: str | Path, env: Mapping[str, str] | None = None) -> Path:
    """Expand environment-backed path values into :class:`Path` objects."""
    if isinstance(value, Path):
        return value.expanduser()
    return Path(expand_env_string(value, env))


def resolve_cache_env(cache_env: Mapping[str, str], env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Resolve cache and workspace environment variables into concrete strings."""
    resolved = {}
    for key in CACHE_ENV_KEYS:
        value = cache_env.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"cache_env.{key} must be a non-empty string.")
        resolved[key] = str(expand_env_path(value, env))
    return resolved


def make_run_id(now: dt.datetime | None = None) -> str:
    """Create the canonical UTC run identifier used by nightly runs."""
    current = now or dt.datetime.now(dt.UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=dt.UTC)
    return current.astimezone(dt.UTC).strftime("%Y-%m-%dT%H-%M-%SZ")


def utc_now_iso(now: dt.datetime | None = None) -> str:
    """Return an ISO-8601 UTC timestamp."""
    current = now or dt.datetime.now(dt.UTC)
    if current.tzinfo is None:
        current = current.replace(tzinfo=dt.UTC)
    return current.astimezone(dt.UTC).isoformat().replace("+00:00", "Z")


@dataclasses.dataclass(frozen=True, slots=True)
class RunPaths:
    """Resolved run-level paths for durable state and node-local scratch."""

    run_id: str
    shared_state_dir: Path
    run_dir: Path
    publish_dir: Path
    tasks_dir: Path
    work_base_dir: Path
    cache_env: dict[str, Path]

    def task_dir(self, task_id: str) -> Path:
        """Return the durable task directory for ``task_id``."""
        return self.tasks_dir / task_id

    def run_manifest_path(self) -> Path:
        """Return the run manifest path."""
        return self.run_dir / "run.json"

    def plan_lock_path(self) -> Path:
        """Return the expanded plan lock path."""
        return self.run_dir / "plan.lock.yaml"

    def publish_summary_path(self) -> Path:
        """Return the local or publish summary path."""
        return self.publish_dir / "summary.json"

    def publish_log_path(self) -> Path:
        """Return the gather-and-publish log path."""
        return self.publish_dir / "publish.log"

    def task_manifest_path(self, task_id: str) -> Path:
        """Return the task manifest path for ``task_id``."""
        return self.task_dir(task_id) / "task.json"

    def task_log_path(self, task_id: str) -> Path:
        """Return the structured task log path for ``task_id``."""
        return self.task_dir(task_id) / "stdout.log"

    def task_status_path(self, task_id: str) -> Path:
        """Return the task status artifact path for ``task_id``."""
        return self.task_dir(task_id) / "status.json"

    def job_dir(self, task_id: str, job_id: str) -> Path:
        """Return the durable job directory for ``job_id``."""
        return self.task_dir(task_id) / "jobs" / job_id

    def job_manifest_path(self, task_id: str, job_id: str) -> Path:
        """Return the job manifest path for ``job_id``."""
        return self.job_dir(task_id, job_id) / "job.json"

    def job_status_path(self, task_id: str, job_id: str) -> Path:
        """Return the job status artifact path for ``job_id``."""
        return self.job_dir(task_id, job_id) / "status.json"

    def job_results_dir(self, task_id: str, job_id: str) -> Path:
        """Return the durable results directory for ``job_id``."""
        return self.job_dir(task_id, job_id) / "results"

    def job_stdout_path(self, task_id: str, job_id: str) -> Path:
        """Return the raw stdout log path for ``job_id``."""
        return self.job_dir(task_id, job_id) / "stdout.log"

    def job_stderr_path(self, task_id: str, job_id: str) -> Path:
        """Return the raw stderr log path for ``job_id``."""
        return self.job_dir(task_id, job_id) / "stderr.log"

    def run_work_dir(self) -> Path:
        """Return the node-local work directory for this run."""
        return self.work_base_dir / self.run_id

    def task_work_dir(self, task_id: str) -> Path:
        """Return the node-local task work directory for ``task_id``."""
        return self.run_work_dir() / task_id

    def job_work_dir(self, task_id: str, job_id: str) -> Path:
        """Return the node-local job work directory for ``job_id``."""
        return self.task_work_dir(task_id) / job_id


@dataclasses.dataclass(slots=True)
class RunManifest:
    """Durable run-level metadata written before concrete jobs start."""

    run_id: str
    mode: str
    revision: str
    created_at: str
    plan_path: str
    plan_run_mode: str
    selected_task_ids: list[str]
    selected_profiles: list[str]
    cache_env: dict[str, str]
    shared_state_dir: str
    work_base_dir: str
    publish_requested: bool
    publish_status: str
    publish_root: str | None = None
    submission_mode: str | None = None
    publish_submission_id: str | None = None

    def to_record(self) -> dict[str, Any]:
        """Convert the manifest into JSON-serializable data."""
        return _jsonable(dataclasses.asdict(self))


@dataclasses.dataclass(slots=True)
class TaskManifest:
    """Durable task-level metadata, including scheduler submission lineage."""

    task_id: str
    kind: str
    profile: str
    hardware_label: str
    job_ids: list[str]
    dependency_ids: list[str]
    submission_id: str | None = None
    output_dir: str | None = None

    def to_record(self) -> dict[str, Any]:
        """Convert the manifest into JSON-serializable data."""
        return _jsonable(dataclasses.asdict(self))


@dataclasses.dataclass(slots=True)
class JobManifest:
    """Durable job metadata with resolved parameters and artifact ownership."""

    job_id: str
    task_id: str
    kind: str
    scenario: str
    series: str
    profile: str
    hardware_label: str
    params: dict[str, Any]
    output_dir: str
    scheduler: dict[str, Any] | None = None

    def to_record(self) -> dict[str, Any]:
        """Convert the manifest into JSON-serializable data."""
        return _jsonable(dataclasses.asdict(self))


@dataclasses.dataclass(slots=True)
class StatusRecord:
    """Execution status for a run, task, or job artifact owner."""

    state: StatusState
    started_at: str | None = None
    finished_at: str | None = None
    duration_s: float | None = None
    exit_code: int | None = None
    error_summary: str | None = None
    failure_phase: str | None = None
    artifact_paths: list[str] = dataclasses.field(default_factory=list)

    def to_record(self) -> dict[str, Any]:
        """Convert the status into JSON-serializable data."""
        return _jsonable(dataclasses.asdict(self))


def resolve_run_paths(
    *,
    run_id: str,
    shared_state_dir: str | Path,
    work_base_dir: str | Path,
    cache_env: Mapping[str, str],
    env: Mapping[str, str] | None = None,
) -> RunPaths:
    """Resolve durable and node-local paths for a nightly run."""
    resolved_shared_state_dir = expand_env_path(shared_state_dir, env)
    resolved_work_base_dir = expand_env_path(work_base_dir, env)
    resolved_cache_env = {key: expand_env_path(value, env) for key, value in resolve_cache_env(cache_env, env).items()}
    run_dir = resolved_shared_state_dir / "runs" / run_id
    return RunPaths(
        run_id=run_id,
        shared_state_dir=resolved_shared_state_dir,
        run_dir=run_dir,
        publish_dir=run_dir / "publish",
        tasks_dir=run_dir / "tasks",
        work_base_dir=resolved_work_base_dir,
        cache_env=resolved_cache_env,
    )


def validate_run_environment(run_paths: RunPaths) -> None:
    """Fail early if shared state, node-local scratch, or cache paths are unusable."""
    _ensure_writable_directory(run_paths.shared_state_dir, create=True, label="shared_state_dir")
    _ensure_writable_directory(run_paths.work_base_dir, create=True, label="work_base_dir")
    for key, path in run_paths.cache_env.items():
        _ensure_writable_directory(path, create=True, label=key)


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write a JSON artifact with stable formatting."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_jsonable(dict(payload)), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def make_log_record(
    event: str,
    *,
    run_id: str,
    task_id: str | None = None,
    job_id: str | None = None,
    hardware_label: str | None = None,
    timestamp: dt.datetime | None = None,
    **fields: Any,
) -> str:
    """Build a parseable JSON log line for nightly orchestration events."""
    record: dict[str, Any] = {
        "timestamp": utc_now_iso(timestamp),
        "event": event,
        "run_id": run_id,
        "task_id": task_id,
        "job_id": job_id,
        "hardware_label": hardware_label,
        **fields,
    }
    compact = {key: value for key, value in record.items() if value is not None}
    return json.dumps(_jsonable(compact), sort_keys=True)


def _ensure_writable_directory(path: Path, *, create: bool, label: str) -> None:
    if path.exists() and not path.is_dir():
        raise ValueError(f"{label} is not a directory: {path}")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        raise ValueError(f"{label} does not exist: {path}")
    try:
        with tempfile.NamedTemporaryFile(dir=path, prefix=".nightly-write-check-", delete=True):
            pass
    except OSError as exc:  # pragma: no cover - exercised indirectly in environments with path issues
        raise ValueError(f"{label} is not writable: {path}") from exc


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_jsonable(item) for item in value]
    return value
