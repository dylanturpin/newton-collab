# FeatherPGS Dense vs Matrix-Free Explainer Handoff

## Current State

- Completed M1: source inventory and page-placement decision.
- Completed M2: schema/bootstrap tooling for the explainer raw-data lane.
- Advanced the first M3 slice: runtime-backed scenario sizing artifacts now exist for `g1_flat` and `h1_tabletop`.
- Advanced the first M4 slice: checked-in kernel-work and memory-layout artifacts now exist for the dense tiled-row PGS kernel and the fused matrix-free GS kernel.
- Completed M5: the docs draft now exists as a sibling concepts page with navigation wiring.
- Completed M6: the docs build and required `pre-commit` validation now pass locally.
- Completed M7: the finalized source/docs state has now been published to `origin/feather_pgs`.
- Chosen docs structure: keep `docs/concepts/feather_pgs.md` as the overview page and add a sibling concepts page for the deep dense-vs-matrix-free comparison.
- Recorded the initial inventory in `.agent/data/fpgs-matrix-free-dense-explainer/m1-source-inventory.md`.
- Added checked-in schemas in `.agent/data/fpgs-matrix-free-dense-explainer/schema/`.
- Added the generated manifest `.agent/data/fpgs-matrix-free-dense-explainer/m2-capture-manifest.json`.
- Added the runtime capture helper `scripts/analysis/capture_fpgs_scenario_sizing.py`.
- Added the kernel artifact helper `scripts/analysis/capture_fpgs_kernel_memory_artifacts.py`.
- Added scenario artifacts:
  - `.agent/data/fpgs-matrix-free-dense-explainer/scenarios/g1_flat/fpgs_dense_row.json`
  - `.agent/data/fpgs-matrix-free-dense-explainer/scenarios/g1_flat/fpgs_matrix_free.json`
  - `.agent/data/fpgs-matrix-free-dense-explainer/scenarios/h1_tabletop/fpgs_dense_row.json`
  - `.agent/data/fpgs-matrix-free-dense-explainer/scenarios/h1_tabletop/fpgs_matrix_free.json`
- Added kernel artifacts:
  - `.agent/data/fpgs-matrix-free-dense-explainer/kernels/dense-row.json`
  - `.agent/data/fpgs-matrix-free-dense-explainer/kernels/matrix-free-gs.json`
- Added the doc-ready M4 summary:
  - `.agent/data/fpgs-matrix-free-dense-explainer/m4-kernel-memory-summary.md`
- Added the explainer docs page and links:
  - `docs/concepts/feather_pgs_dense_vs_matrix_free.md`
  - `docs/index.rst`
  - `docs/concepts/feather_pgs.md`
- Refreshed the kernel artifacts so their provenance matches the source revision used for the docs draft:
  - `.agent/data/fpgs-matrix-free-dense-explainer/kernels/dense-row.json`
  - `.agent/data/fpgs-matrix-free-dense-explainer/kernels/matrix-free-gs.json`
- Accepted formatting-only `ruff` rewrites in two helper scripts:
  - `scripts/analysis/capture_fpgs_kernel_memory_artifacts.py`
  - `scripts/analysis/capture_fpgs_scenario_sizing.py`

## Key Findings From This Pass

- The source/docs publication gate is now cleared: `origin/feather_pgs` was updated from `79f10ef3` to the explainer branch tip.
- The explainer deliverable published cleanly without reopening docs-content work or touching `origin/gh-pages`.
- This pass only updates publication bookkeeping in the living ExecPlan and handoff after the successful source-branch push.

## Recommended Next Pass

- M8 remains explicitly gated and is now the only remaining incomplete milestone.
- When publication of the rendered site is in scope, update only `origin/gh-pages` and preserve nightly benchmark assets while refreshing the docs payload.
- Keep the `gh-pages` work docs-only and record exactly which paths changed and which nightly artifacts were intentionally left untouched.

## Publication Record

- Pre-publication remote state:
  - `origin/feather_pgs` -> `79f10ef391c3931d135db76c6b9fe572c09895b3`
  - `origin/gh-pages` -> `cfd01f5f69909e9c90f7d637c6772d6f09df31e2`
- Source publication command:
  - `git push origin HEAD:feather_pgs`
  - result: fast-forward update from `79f10ef3` to `ace07904`
- `origin/gh-pages` was intentionally left untouched in this pass.

## Validation Evidence

- `uv run python scripts/analysis/bootstrap_fpgs_explainer_artifacts.py`
- `python -m json.tool .agent/data/fpgs-matrix-free-dense-explainer/schema/scenario-sizing.schema.json`
- `python -m json.tool .agent/data/fpgs-matrix-free-dense-explainer/schema/kernel-memory-analysis.schema.json`
- `uv run --with GitPython --with usd-core --with scipy python scripts/analysis/capture_fpgs_scenario_sizing.py`
- `python -m json.tool .agent/data/fpgs-matrix-free-dense-explainer/scenarios/g1_flat/fpgs_matrix_free.json`
- `python -m json.tool .agent/data/fpgs-matrix-free-dense-explainer/scenarios/h1_tabletop/fpgs_dense_row.json`
- `uv run python scripts/analysis/capture_fpgs_kernel_memory_artifacts.py`
- `python -m json.tool .agent/data/fpgs-matrix-free-dense-explainer/kernels/dense-row.json`
- `python -m json.tool .agent/data/fpgs-matrix-free-dense-explainer/kernels/matrix-free-gs.json`
- `git diff -- .agent/data/fpgs-matrix-free-dense-explainer/kernels`
- `uv run --extra docs --extra sim sphinx-build -j auto -b html docs docs/_build/html`
  - result: passed
  - note: emitted existing multiple-toctree consistency notices for generated `docs/api/newton_*.rst` pages
- `uvx pre-commit run -a`
  - first result: failed because `ruff` and `ruff format` rewrote two helper scripts
  - second result: passed
