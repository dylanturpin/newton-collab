# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Nightly plan loading, validation, and deterministic job expansion."""

from __future__ import annotations

import argparse
import copy
import itertools
import os
import string
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from newton.tools.solver_benchmark import SCENARIOS, SOLVER_PRESETS

DEFAULT_PLAN_PATH = Path(__file__).with_name("nightly.yaml")

RUN_MODES = ("full", "validation")
TASK_KINDS = ("benchmark", "render")
CACHE_ENV_KEYS = (
    "TMPDIR",
    "UV_CACHE_DIR",
    "UV_PROJECT_ENVIRONMENT",
    "WARP_CACHE_PATH",
    "NEWTON_CACHE_PATH",
    "CUDA_CACHE_PATH",
)
EXPANSION_FIELD_ORDER = (
    "solver",
    "substeps",
    "num_worlds",
    "render_frames",
    "camera_pos",
    "camera_pitch",
    "camera_yaw",
)
SOLVER_OVERRIDE_FIELDS = (
    "cholesky_kernel",
    "trisolve_kernel",
    "hinv_jt_kernel",
    "delassus_kernel",
    "pgs_kernel",
    "pgs_mode",
    "dense_max_constraints",
    "use_parallel_streams",
    "double_buffer",
    "pipeline_collide",
    "delassus_chunk_size",
    "pgs_chunk_size",
)
TASK_METADATA_FIELDS = {
    "id",
    "kind",
    "profile",
    "targets",
    "series",
    "tags",
    "scenario",
    "run_modes",
    "jobs",
}
JOB_METADATA_FIELDS = {
    "id",
    "label",
    "tags",
}
BENCHMARK_OPTION_FIELDS = {
    "solver",
    "substeps",
    "num_worlds",
    "warmup_frames",
    "measure_frames",
    "viewer",
    "cholesky_kernel",
    "trisolve_kernel",
    "hinv_jt_kernel",
    "delassus_kernel",
    "pgs_kernel",
    "pgs_iterations",
    "dense_max_constraints",
    "pgs_beta",
    "pgs_cfm",
    "pgs_omega",
    "pgs_warmstart",
    "pgs_mode",
    "use_parallel_streams",
    "double_buffer",
    "pipeline_collide",
    "delassus_chunk_size",
    "pgs_chunk_size",
    "mj_solver",
    "mj_integrator",
    "mj_njmax",
    "mj_nconmax",
    "override_scenario_defaults",
    "nsys_profile",
    "nsys_cuda_graph_trace",
}
RENDER_OPTION_FIELDS = {
    "solver",
    "substeps",
    "num_worlds",
    "render_frames",
    "render_width",
    "render_height",
    "camera_pos",
    "camera_pitch",
    "camera_yaw",
    "cholesky_kernel",
    "trisolve_kernel",
    "hinv_jt_kernel",
    "delassus_kernel",
    "pgs_kernel",
    "pgs_iterations",
    "dense_max_constraints",
    "pgs_beta",
    "pgs_cfm",
    "pgs_omega",
    "pgs_warmstart",
    "pgs_mode",
    "use_parallel_streams",
    "double_buffer",
    "pipeline_collide",
    "delassus_chunk_size",
    "pgs_chunk_size",
    "mj_solver",
    "mj_integrator",
    "mj_njmax",
    "mj_nconmax",
    "override_scenario_defaults",
}


class PlanValidationError(ValueError):
    """Raised when the nightly plan is invalid."""


def load_plan(path: Path | str = DEFAULT_PLAN_PATH, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Load a nightly plan YAML file and expand environment-backed strings."""
    env_map = dict(os.environ if env is None else env)
    plan_path = Path(path)
    with plan_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise PlanValidationError(f"Nightly plan root must be a mapping: {plan_path}")
    loaded["__plan_path__"] = str(plan_path)
    return _expand_env_values(loaded, env_map)


def validate_plan(plan: Mapping[str, Any]) -> None:
    """Validate nightly plan structure and task definitions."""
    if not isinstance(plan, Mapping):
        raise PlanValidationError("Nightly plan must be a mapping.")
    if plan.get("version") != 1:
        raise PlanValidationError("Nightly plan version must be 1.")

    defaults = _require_mapping(plan, "defaults")
    _validate_defaults(defaults)

    hardware_profiles = _require_mapping(plan, "hardware_profiles")
    if not hardware_profiles:
        raise PlanValidationError("Nightly plan must define at least one hardware profile.")
    for profile_name, profile in hardware_profiles.items():
        if not isinstance(profile, Mapping):
            raise PlanValidationError(f"Hardware profile '{profile_name}' must be a mapping.")
        if not profile.get("label"):
            raise PlanValidationError(f"Hardware profile '{profile_name}' must define a non-empty label.")

    tasks = plan.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise PlanValidationError("Nightly plan must define at least one task.")

    seen_task_ids: set[str] = set()
    for task in tasks:
        _validate_task(task, hardware_profiles, seen_task_ids)

    validation = plan.get("validation", {})
    if validation:
        if not isinstance(validation, Mapping):
            raise PlanValidationError("validation must be a mapping when present.")
        task_ids = validation.get("task_ids", [])
        if not isinstance(task_ids, list) or not task_ids:
            raise PlanValidationError("validation.task_ids must be a non-empty list when validation is configured.")
        known_task_ids = {task["id"] for task in tasks}
        missing = sorted(set(task_ids) - known_task_ids)
        if missing:
            raise PlanValidationError(f"validation.task_ids references unknown tasks: {', '.join(missing)}")


def expand_plan(
    plan: Mapping[str, Any],
    *,
    run_mode: str = "full",
    selected_task_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Expand authored tasks into deterministic concrete jobs."""
    validate_plan(plan)
    if run_mode not in RUN_MODES:
        raise PlanValidationError(f"Unsupported run mode: {run_mode}")

    effective_task_ids = list(selected_task_ids or [])
    if not effective_task_ids and run_mode == "validation":
        validation = plan.get("validation", {})
        effective_task_ids = list(validation.get("task_ids", []))

    selected_set = set(effective_task_ids)
    matched_selected: set[str] = set()
    expanded_tasks = []
    seen_expanded_ids: set[str] = set()
    for task in plan["tasks"]:
        run_modes = _normalized_run_modes(task)
        if run_mode not in run_modes:
            continue
        concrete_tasks = _expand_task(task, plan["defaults"], plan["hardware_profiles"])
        for concrete_task in concrete_tasks:
            include = not selected_set or task["id"] in selected_set or concrete_task["id"] in selected_set
            if not include:
                continue
            if concrete_task["id"] in seen_expanded_ids:
                raise PlanValidationError(f"Expanded task id collision: {concrete_task['id']}")
            seen_expanded_ids.add(concrete_task["id"])
            expanded_tasks.append(concrete_task)
            if task["id"] in selected_set:
                matched_selected.add(task["id"])
            if concrete_task["id"] in selected_set:
                matched_selected.add(concrete_task["id"])

    if selected_set:
        missing = sorted(selected_set - matched_selected)
        if missing:
            raise PlanValidationError(
                f"Selected task ids were not available in mode '{run_mode}': {', '.join(missing)}"
            )

    if not expanded_tasks:
        raise PlanValidationError(f"No tasks selected for run mode '{run_mode}'.")

    return {
        "version": plan["version"],
        "plan_path": plan.get("__plan_path__", str(DEFAULT_PLAN_PATH)),
        "run_mode": run_mode,
        "defaults": copy.deepcopy(plan["defaults"]),
        "hardware_profiles": copy.deepcopy(plan["hardware_profiles"]),
        "validation": copy.deepcopy(plan.get("validation", {})),
        "tasks": expanded_tasks,
    }


def write_plan_lock(expanded_plan: Mapping[str, Any], destination: Path | str) -> Path:
    """Write an expanded plan lock file."""
    lock_path = Path(destination)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(expanded_plan), handle, sort_keys=False, allow_unicode=False)
    return lock_path


def _validate_defaults(defaults: Mapping[str, Any]) -> None:
    cache_env = _require_mapping(defaults, "cache_env")
    for key in CACHE_ENV_KEYS:
        value = cache_env.get(key)
        if not isinstance(value, str) or not value.strip():
            raise PlanValidationError(f"defaults.cache_env.{key} must be a non-empty string.")
    for path_key in ("shared_state_dir", "work_base_dir"):
        value = defaults.get(path_key)
        if not isinstance(value, str) or not value.strip():
            raise PlanValidationError(f"defaults.{path_key} must be a non-empty string.")


def _validate_task(task: Any, hardware_profiles: Mapping[str, Any], seen_task_ids: set[str]) -> None:
    if not isinstance(task, Mapping):
        raise PlanValidationError("Each task must be a mapping.")

    task_id = task.get("id")
    if not isinstance(task_id, str) or not task_id:
        raise PlanValidationError("Each task must define a non-empty id.")
    if task_id in seen_task_ids:
        raise PlanValidationError(f"Duplicate task id: {task_id}")
    seen_task_ids.add(task_id)

    kind = task.get("kind")
    if kind not in TASK_KINDS:
        raise PlanValidationError(f"Task '{task_id}' must have kind in {TASK_KINDS}.")

    if task.get("scenario") not in SCENARIOS:
        raise PlanValidationError(f"Task '{task_id}' references unknown scenario '{task.get('scenario')}'.")
    if not isinstance(task.get("series"), str) or not task["series"]:
        raise PlanValidationError(f"Task '{task_id}' must define a non-empty series.")
    _validate_tags(task.get("tags"), task_id, "task")
    _normalized_targets(task, hardware_profiles)

    run_modes = _normalized_run_modes(task)
    if not run_modes:
        raise PlanValidationError(f"Task '{task_id}' must include at least one run mode.")

    option_fields = RENDER_OPTION_FIELDS if kind == "render" else BENCHMARK_OPTION_FIELDS
    allowed_fields = TASK_METADATA_FIELDS | option_fields
    unknown_fields = sorted(set(task) - allowed_fields)
    if unknown_fields:
        raise PlanValidationError(f"Task '{task_id}' contains unknown fields: {', '.join(unknown_fields)}")

    jobs = task.get("jobs")
    if jobs is not None:
        _validate_explicit_job_task(task, option_fields)
        return

    list_fields = [
        key for key, value in task.items() if isinstance(value, list) and key not in {"run_modes", "targets", "tags"}
    ]
    invalid_list_fields = sorted(field for field in list_fields if field not in EXPANSION_FIELD_ORDER)
    if invalid_list_fields:
        raise PlanValidationError(
            f"Task '{task_id}' uses list-valued fields that are not supported for deterministic expansion: "
            f"{', '.join(invalid_list_fields)}"
        )

    solver_values = _as_list(task.get("solver", "feather_pgs"))
    for solver_name in solver_values:
        if solver_name not in SOLVER_PRESETS:
            raise PlanValidationError(f"Task '{task_id}' references unknown solver preset '{solver_name}'.")

    if kind == "benchmark" and "num_worlds" not in task:
        raise PlanValidationError(f"Benchmark task '{task_id}' must define num_worlds.")

    _validate_target_overrides(task, option_fields)


def _validate_explicit_job_task(task: Mapping[str, Any], option_fields: set[str]) -> None:
    task_id = task["id"]
    jobs = task.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise PlanValidationError(f"Task '{task_id}' jobs must be a non-empty list.")

    task_list_fields = [
        key
        for key, value in task.items()
        if isinstance(value, list) and key not in {"run_modes", "targets", "tags", "jobs"}
    ]
    if task_list_fields:
        raise PlanValidationError(
            f"Explicit-job task '{task_id}' must not use list-valued runtime fields: {', '.join(sorted(task_list_fields))}"
        )

    seen_job_ids: set[str] = set()
    for job in jobs:
        if not isinstance(job, Mapping):
            raise PlanValidationError(f"Explicit job entries for task '{task_id}' must be mappings.")
        job_id = job.get("id")
        if not isinstance(job_id, str) or not job_id:
            raise PlanValidationError(f"Explicit jobs for task '{task_id}' must define a non-empty id.")
        if not _is_valid_job_fragment(job_id):
            raise PlanValidationError(
                f"Explicit job id '{job_id}' in task '{task_id}' must use only letters, numbers, '-' or '_'."
            )
        if job_id in seen_job_ids:
            raise PlanValidationError(f"Duplicate explicit job id '{job_id}' in task '{task_id}'.")
        seen_job_ids.add(job_id)

        label = job.get("label")
        if label is not None and (not isinstance(label, str) or not label):
            raise PlanValidationError(f"Explicit job '{job_id}' in task '{task_id}' must use a non-empty label.")
        _validate_tags(job.get("tags"), task_id, f"job '{job_id}'")

        allowed_job_fields = JOB_METADATA_FIELDS | option_fields
        unknown_job_fields = sorted(set(job) - allowed_job_fields)
        if unknown_job_fields:
            raise PlanValidationError(
                f"Explicit job '{job_id}' in task '{task_id}' contains unknown fields: {', '.join(unknown_job_fields)}"
            )

        job_list_fields = [key for key, value in job.items() if isinstance(value, list) and key != "tags"]
        if job_list_fields:
            raise PlanValidationError(
                f"Explicit job '{job_id}' in task '{task_id}' must not use list-valued runtime fields: "
                f"{', '.join(sorted(job_list_fields))}"
            )

        solver_name = job.get("solver", task.get("solver", "feather_pgs"))
        if solver_name not in SOLVER_PRESETS:
            raise PlanValidationError(
                f"Explicit job '{job_id}' in task '{task_id}' references unknown solver preset '{solver_name}'."
            )

        num_worlds = job.get("num_worlds", task.get("num_worlds"))
        if task["kind"] == "benchmark" and not isinstance(num_worlds, int):
            raise PlanValidationError(
                f"Explicit benchmark job '{job_id}' in task '{task_id}' must resolve to an integer num_worlds."
            )

    _validate_target_overrides(task, option_fields, explicit_jobs=True)


def _validate_target_overrides(
    task: Mapping[str, Any],
    option_fields: set[str],
    *,
    explicit_jobs: bool = False,
) -> None:
    targets = task.get("targets")
    if not isinstance(targets, Mapping):
        return

    for profile_name, overrides in targets.items():
        if overrides in (None, {}):
            continue
        if not isinstance(overrides, Mapping):
            raise PlanValidationError(f"Target '{profile_name}' overrides for task '{task['id']}' must be a mapping.")
        unknown_fields = sorted(set(overrides) - option_fields)
        if unknown_fields:
            raise PlanValidationError(
                f"Target '{profile_name}' overrides for task '{task['id']}' contain unknown fields: "
                f"{', '.join(unknown_fields)}"
            )
        if explicit_jobs:
            list_fields = [key for key, value in overrides.items() if isinstance(value, list)]
            if list_fields:
                raise PlanValidationError(
                    f"Explicit-job task '{task['id']}' target '{profile_name}' must not use list-valued overrides: "
                    f"{', '.join(sorted(list_fields))}"
                )
        else:
            list_fields = [key for key, value in overrides.items() if isinstance(value, list)]
            invalid_list_fields = sorted(field for field in list_fields if field not in EXPANSION_FIELD_ORDER)
            if invalid_list_fields:
                raise PlanValidationError(
                    f"Task '{task['id']}' target '{profile_name}' uses list-valued overrides that are not "
                    f"supported for deterministic expansion: {', '.join(invalid_list_fields)}"
                )
        solver_name = overrides.get("solver")
        if solver_name is not None and solver_name not in SOLVER_PRESETS:
            raise PlanValidationError(
                f"Target '{profile_name}' overrides for task '{task['id']}' reference unknown solver '{solver_name}'."
            )


def _expand_task(
    task: Mapping[str, Any],
    defaults: Mapping[str, Any],
    hardware_profiles: Mapping[str, Any],
) -> list[dict[str, Any]]:
    targets = _normalized_targets(task, hardware_profiles)
    multi_target = len(targets) > 1
    expanded_tasks: list[dict[str, Any]] = []
    for profile_name, target_overrides in targets:
        profile = copy.deepcopy(hardware_profiles[profile_name])
        concrete_task = copy.deepcopy(task)
        concrete_task["source_task_id"] = task["id"]
        concrete_task["id"] = _expanded_task_id(task["id"], profile_name, multi_target)
        concrete_task["profile"] = profile_name
        concrete_task.pop("targets", None)

        if task.get("jobs") is not None:
            jobs = _expand_explicit_jobs(concrete_task, defaults, profile, target_overrides)
            job_style = "explicit"
        else:
            matrix_task = _apply_overrides(concrete_task, target_overrides)
            jobs = _expand_matrix_jobs(matrix_task, defaults, profile)
            job_style = "matrix"

        expanded_tasks.append(
            {
                "id": concrete_task["id"],
                "source_task_id": task["id"],
                "kind": concrete_task["kind"],
                "profile": profile_name,
                "hardware_label": profile["label"],
                "series": concrete_task["series"],
                "tags": list(concrete_task.get("tags", [])),
                "scenario": concrete_task["scenario"],
                "run_modes": _normalized_run_modes(concrete_task),
                "job_style": job_style,
                "job_count": len(jobs),
                "jobs": jobs,
            }
        )
    return expanded_tasks


def _expand_matrix_jobs(
    task: Mapping[str, Any],
    defaults: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> list[dict[str, Any]]:
    matrix_fields = [field for field in EXPANSION_FIELD_ORDER if isinstance(task.get(field), list)]
    matrix_values = [_as_list(task[field]) for field in matrix_fields]
    combinations = list(itertools.product(*matrix_values)) if matrix_fields else [()]

    jobs = []
    for job_index, combination in enumerate(combinations, start=1):
        job = _base_job(task, defaults, profile)
        for field_name, field_value in zip(matrix_fields, combination, strict=False):
            job[field_name] = field_value
        job["solver_config"] = _resolve_solver_config(job)
        job["id"] = _format_matrix_job_id(task["id"], job_index)
        jobs.append(job)
    return jobs


def _expand_explicit_jobs(
    task: Mapping[str, Any],
    defaults: Mapping[str, Any],
    profile: Mapping[str, Any],
    target_overrides: Mapping[str, Any],
) -> list[dict[str, Any]]:
    jobs = []
    task_tags = task.get("tags", [])
    for step_index, job_spec in enumerate(task["jobs"], start=0):
        job = _base_job(task, defaults, profile)
        for field, value in job_spec.items():
            if field in JOB_METADATA_FIELDS:
                continue
            job[field] = copy.deepcopy(value)
        job = _apply_overrides(job, target_overrides)
        job["solver_config"] = _resolve_solver_config(job)
        job["variant_id"] = job_spec["id"]
        job["label"] = job_spec.get("label", job_spec["id"])
        job["step_index"] = step_index
        job["tags"] = _combine_tags(task_tags, job_spec.get("tags", []))
        job["id"] = _format_explicit_job_id(task["id"], job_spec["id"])
        jobs.append(job)
    return jobs


def _base_job(task: Mapping[str, Any], defaults: Mapping[str, Any], profile: Mapping[str, Any]) -> dict[str, Any]:
    scenario_cfg = SCENARIOS[task["scenario"]]
    benchmark_defaults = defaults.get("benchmark", {})
    render_defaults = defaults.get("render", {})
    kind = task["kind"]

    job = {
        "task_id": task["id"],
        "kind": kind,
        "profile": task["profile"],
        "hardware_label": profile["label"],
        "series": task["series"],
        "tags": list(task.get("tags", [])),
        "scenario": task["scenario"],
        "solver": task.get("solver", "feather_pgs"),
        "substeps": task.get("substeps", scenario_cfg.get("default_substeps")),
        "num_worlds": task.get("num_worlds", 1),
    }

    if kind == "benchmark":
        for field in BENCHMARK_OPTION_FIELDS - {"solver", "substeps", "num_worlds"}:
            if field in task:
                job[field] = task[field]
            elif field in benchmark_defaults:
                job[field] = benchmark_defaults[field]
    else:
        for field in RENDER_OPTION_FIELDS - {"solver", "substeps", "num_worlds"}:
            if field in task:
                job[field] = task[field]
            elif field in render_defaults:
                job[field] = render_defaults[field]
        job["num_worlds"] = task.get("num_worlds", 1)

    return job


def _resolve_solver_config(job: Mapping[str, Any]) -> dict[str, Any]:
    solver_name = job["solver"]
    solver_config = copy.deepcopy(SOLVER_PRESETS[solver_name])
    for field in SOLVER_OVERRIDE_FIELDS:
        if field in job and job[field] is not None:
            solver_config[field] = job[field]
    return solver_config


def _normalized_run_modes(task: Mapping[str, Any]) -> list[str]:
    run_modes = task.get("run_modes", ["full"])
    if not isinstance(run_modes, list):
        raise PlanValidationError(f"Task '{task.get('id', '<unknown>')}' run_modes must be a list.")
    for run_mode in run_modes:
        if run_mode not in RUN_MODES:
            raise PlanValidationError(
                f"Task '{task.get('id', '<unknown>')}' references unsupported run mode '{run_mode}'."
            )
    return list(run_modes)


def _normalized_targets(
    task: Mapping[str, Any],
    hardware_profiles: Mapping[str, Any],
) -> list[tuple[str, Mapping[str, Any]]]:
    profile = task.get("profile")
    targets = task.get("targets")
    task_id = task.get("id", "<unknown>")

    if profile is not None and targets is not None:
        raise PlanValidationError(f"Task '{task_id}' must use either profile or targets, not both.")
    if profile is None and targets is None:
        raise PlanValidationError(f"Task '{task_id}' must define profile or targets.")

    if profile is not None:
        if profile not in hardware_profiles:
            raise PlanValidationError(f"Task '{task_id}' references unknown hardware profile '{profile}'.")
        return [(profile, {})]

    if isinstance(targets, list):
        normalized = []
        seen_profiles: set[str] = set()
        for target in targets:
            if not isinstance(target, str) or not target:
                raise PlanValidationError(f"Task '{task_id}' targets must be non-empty profile names.")
            if target not in hardware_profiles:
                raise PlanValidationError(f"Task '{task_id}' references unknown hardware profile '{target}'.")
            if target in seen_profiles:
                raise PlanValidationError(f"Task '{task_id}' targets include duplicate profile '{target}'.")
            seen_profiles.add(target)
            normalized.append((target, {}))
        if not normalized:
            raise PlanValidationError(f"Task '{task_id}' targets must not be empty.")
        return normalized

    if isinstance(targets, Mapping):
        normalized = []
        for target, overrides in targets.items():
            if not isinstance(target, str) or not target:
                raise PlanValidationError(f"Task '{task_id}' targets must use non-empty profile names.")
            if target not in hardware_profiles:
                raise PlanValidationError(f"Task '{task_id}' references unknown hardware profile '{target}'.")
            if overrides is None:
                normalized.append((target, {}))
            elif isinstance(overrides, Mapping):
                normalized.append((target, dict(overrides)))
            else:
                raise PlanValidationError(f"Task '{task_id}' target '{target}' overrides must be a mapping or null.")
        if not normalized:
            raise PlanValidationError(f"Task '{task_id}' targets must not be empty.")
        return normalized

    raise PlanValidationError(f"Task '{task_id}' targets must be a list or mapping.")


def _expand_env_values(value: Any, env: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(string.Template(value).safe_substitute(env))
    if isinstance(value, list):
        return [_expand_env_values(item, env) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env_values(item, env) for key, item in value.items()}
    return value


def _require_mapping(parent: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    value = parent.get(field_name)
    if not isinstance(value, Mapping):
        raise PlanValidationError(f"{field_name} must be a mapping.")
    return value


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else [value]


def _combine_tags(task_tags: Sequence[str], job_tags: Sequence[str]) -> list[str]:
    combined: list[str] = []
    for tag in itertools.chain(task_tags, job_tags):
        if tag not in combined:
            combined.append(tag)
    return combined


def _validate_tags(tags: Any, task_id: str, owner: str) -> None:
    if tags is None:
        return
    if not isinstance(tags, list) or not tags:
        raise PlanValidationError(f"{owner.capitalize()} for task '{task_id}' must use a non-empty tags list.")
    for tag in tags:
        if not isinstance(tag, str) or not tag:
            raise PlanValidationError(f"{owner.capitalize()} for task '{task_id}' uses an invalid tag value.")


def _apply_overrides(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in overrides.items():
        merged[key] = copy.deepcopy(value)
    return merged


def _expanded_task_id(task_id: str, profile_name: str, multi_target: bool) -> str:
    return f"{task_id}_{profile_name}" if multi_target else task_id


def _format_matrix_job_id(task_id: str, job_index: int) -> str:
    return f"{task_id}__{job_index:04d}"


def _format_explicit_job_id(task_id: str, job_id: str) -> str:
    return f"{task_id}__{job_id}"


def _is_valid_job_fragment(value: str) -> bool:
    return bool(value) and all(char.isalnum() or char in {"-", "_"} for char in value)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate and expand the nightly execution plan.")
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN_PATH, help="Path to nightly.yaml")
    parser.add_argument("--run-mode", choices=RUN_MODES, default="full", help="Task set to expand")
    parser.add_argument("--task-id", action="append", default=None, help="Restrict expansion to specific task ids")
    parser.add_argument("--write-lock", type=Path, default=None, help="Destination plan.lock.yaml path")
    return parser


def main() -> None:
    """CLI entrypoint for validating and writing expanded nightly plans."""
    parser = _build_arg_parser()
    args = parser.parse_args()
    plan = load_plan(args.plan)
    expanded = expand_plan(plan, run_mode=args.run_mode, selected_task_ids=args.task_id)
    if args.write_lock is not None:
        write_plan_lock(expanded, args.write_lock)
    else:
        yaml.safe_dump(expanded, stream=sys.stdout, sort_keys=False, allow_unicode=False)


if __name__ == "__main__":
    main()
