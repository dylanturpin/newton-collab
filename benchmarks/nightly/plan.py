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

from newton.tools.solver_benchmark import ABLATION_SEQUENCES, SCENARIOS, SOLVER_PRESETS

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
    "storage",
    "cholesky_kernel",
    "trisolve_kernel",
    "hinv_jt_kernel",
    "delassus_kernel",
    "pgs_kernel",
    "pgs_mode",
    "dense_max_constraints",
    "use_parallel_streams",
    "delassus_chunk_size",
    "pgs_chunk_size",
)
TASK_METADATA_FIELDS = {
    "id",
    "kind",
    "profile",
    "series",
    "scenario",
    "run_modes",
    "ablation_sequence",
    "ablation_pgs",
}
BENCHMARK_OPTION_FIELDS = {
    "solver",
    "substeps",
    "num_worlds",
    "warmup_frames",
    "measure_frames",
    "viewer",
    "storage",
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
    "storage",
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
    expanded_tasks = []
    for task in plan["tasks"]:
        run_modes = _normalized_run_modes(task)
        if run_mode not in run_modes:
            continue
        if selected_set and task["id"] not in selected_set:
            continue
        expanded_tasks.append(_expand_task(task, plan["defaults"], plan["hardware_profiles"]))

    if selected_set:
        expanded_ids = {task["id"] for task in expanded_tasks}
        missing = sorted(selected_set - expanded_ids)
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

    if task.get("profile") not in hardware_profiles:
        raise PlanValidationError(f"Task '{task_id}' references unknown hardware profile '{task.get('profile')}'.")
    if task.get("scenario") not in SCENARIOS:
        raise PlanValidationError(f"Task '{task_id}' references unknown scenario '{task.get('scenario')}'.")
    if not isinstance(task.get("series"), str) or not task["series"]:
        raise PlanValidationError(f"Task '{task_id}' must define a non-empty series.")

    run_modes = _normalized_run_modes(task)
    if not run_modes:
        raise PlanValidationError(f"Task '{task_id}' must include at least one run mode.")

    allowed_fields = TASK_METADATA_FIELDS | (RENDER_OPTION_FIELDS if kind == "render" else BENCHMARK_OPTION_FIELDS)
    unknown_fields = sorted(set(task) - allowed_fields)
    if unknown_fields:
        raise PlanValidationError(f"Task '{task_id}' contains unknown fields: {', '.join(unknown_fields)}")

    list_fields = [key for key, value in task.items() if isinstance(value, list) and key not in {"run_modes"}]
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

    if task.get("ablation_sequence"):
        _validate_ablation_task(task)
    elif kind == "benchmark" and "num_worlds" not in task:
        raise PlanValidationError(f"Benchmark task '{task_id}' must define num_worlds.")


def _validate_ablation_task(task: Mapping[str, Any]) -> None:
    task_id = task["id"]
    sequence_name = task.get("ablation_sequence") or SCENARIOS[task["scenario"]].get("ablation_sequence", "default")
    if sequence_name not in ABLATION_SEQUENCES:
        raise PlanValidationError(f"Task '{task_id}' references unknown ablation sequence '{sequence_name}'.")
    solver_name = task.get("solver", "feather_pgs")
    if isinstance(solver_name, list):
        raise PlanValidationError(f"Ablation task '{task_id}' must not use a solver list.")
    if solver_name != "feather_pgs":
        raise PlanValidationError(
            f"Ablation task '{task_id}' must use the concrete FeatherPGS worker path, not '{solver_name}'."
        )
    num_worlds = task.get("num_worlds")
    if not isinstance(num_worlds, int):
        raise PlanValidationError(f"Ablation task '{task_id}' must define an integer num_worlds.")


def _expand_task(
    task: Mapping[str, Any],
    defaults: Mapping[str, Any],
    hardware_profiles: Mapping[str, Any],
) -> dict[str, Any]:
    profile = copy.deepcopy(hardware_profiles[task["profile"]])
    sequence_name = task.get("ablation_sequence")
    jobs = (
        _expand_ablation_jobs(task, defaults, profile)
        if sequence_name
        else _expand_matrix_jobs(task, defaults, profile)
    )
    return {
        "id": task["id"],
        "kind": task["kind"],
        "profile": task["profile"],
        "hardware_label": profile["label"],
        "series": task["series"],
        "scenario": task["scenario"],
        "run_modes": _normalized_run_modes(task),
        "ablation_sequence": task.get("ablation_sequence"),
        "job_count": len(jobs),
        "jobs": jobs,
    }


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
        job["id"] = _format_job_id(task["id"], job_index)
        jobs.append(job)
    return jobs


def _expand_ablation_jobs(
    task: Mapping[str, Any],
    defaults: Mapping[str, Any],
    profile: Mapping[str, Any],
) -> list[dict[str, Any]]:
    scenario_cfg = SCENARIOS[task["scenario"]]
    sequence_name = task.get("ablation_sequence") or scenario_cfg.get("ablation_sequence", "default")
    ablation_sequence = copy.deepcopy(ABLATION_SEQUENCES[sequence_name])
    ablation_pgs = task.get("ablation_pgs", "auto")

    ablation_steps = []
    for step in ablation_sequence:
        step_config = copy.deepcopy(step)
        step_config["type"] = "feather_pgs"
        if ablation_pgs != "auto" and (
            "PGS" in step_config["label"]
            or "parallel streams" in step_config["label"]
            or "double buffer" in step_config["label"]
            or "pipeline collide" in step_config["label"]
        ):
            step_config["pgs_kernel"] = ablation_pgs
        ablation_steps.append(step_config)
    ablation_steps.append({"label": "MuJoCo baseline", "type": "mujoco"})

    jobs = []
    for step_index, step_config in enumerate(ablation_steps, start=1):
        job = _base_job(task, defaults, profile)
        job["solver"] = "mujoco" if step_config["type"] == "mujoco" else "feather_pgs"
        job["solver_config"] = copy.deepcopy(step_config)
        job["ablation_label"] = step_config["label"]
        job["ablation_step_index"] = step_index - 1
        if "double_buffer" in step_config:
            job["double_buffer"] = step_config["double_buffer"]
        if "pipeline_collide" in step_config:
            job["pipeline_collide"] = step_config["pipeline_collide"]
        job["id"] = _format_job_id(task["id"], step_index)
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


def _format_job_id(task_id: str, job_index: int) -> str:
    return f"{task_id}__{job_index:04d}"


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
