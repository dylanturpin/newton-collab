# Nightly Implementation Notes

This file records extra implementation decisions made while finishing the
nightly Slurm path without waiting for review between tasks.

## 2026-03-08

- Keep the published surface intentionally small: top-level `runs.jsonl`,
  `points.jsonl`, and per-run `renders.json`, plus copied per-run artifacts
  under `runs/<run_id>/`.
- Treat the publisher as a pure gather step over durable task/job artifacts.
  It does not re-run benchmarks or derive missing worker outputs.
- Preserve stable chart grouping through the authored `series` field, but keep
  chart layout and scene/display ownership in `index.html`.
- Add an optional `publish_root` override for local-safe publication and tests.
  When set, publication writes directly to a filesystem directory and skips git
  branch synchronization. The default path still targets `origin/<results_branch>`.
- Keep `mode`, `label`, and `step_index` in published point rows as dashboard
  convenience fields even though the display routing is owned by stable
  `series`. This keeps the page logic simple while removing execution packing
  from the planner-to-dashboard contract.
- For mixed-profile runs, emit one `runs.jsonl` row per GPU while preserving a
  single `run_id`, shared run directory, and shared `summary.json`. This keeps
  one launch/finalize path across GPUs without collapsing the dashboard onto a
  fake single-GPU summary.
- Route render tasks to the RTX PRO 6000 Blackwell Server Edition profile for
  now. The 5090 nodes benchmark correctly but do not have a reliable GL/EGL
  stack for headless rendering, so the dashboard falls back to any same-run
  render and labels the source GPU explicitly.
