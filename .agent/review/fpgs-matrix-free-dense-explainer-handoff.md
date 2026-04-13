# FeatherPGS Dense vs Matrix-Free Explainer Handoff

## Current State

- Completed M1: source inventory and page-placement decision.
- Completed M2: schema/bootstrap tooling for the explainer raw-data lane.
- Advanced the first M3 slice: runtime-backed scenario sizing artifacts now exist for `g1_flat` and `h1_tabletop`.
- Chosen docs structure: keep `docs/concepts/feather_pgs.md` as the overview page and add a sibling concepts page for the deep dense-vs-matrix-free comparison.
- Recorded the initial inventory in `.agent/data/fpgs-matrix-free-dense-explainer/m1-source-inventory.md`.
- Added checked-in schemas in `.agent/data/fpgs-matrix-free-dense-explainer/schema/`.
- Added the generated manifest `.agent/data/fpgs-matrix-free-dense-explainer/m2-capture-manifest.json`.
- Added the runtime capture helper `scripts/analysis/capture_fpgs_scenario_sizing.py`.
- Added scenario artifacts:
  - `.agent/data/fpgs-matrix-free-dense-explainer/scenarios/g1_flat/fpgs_dense_row.json`
  - `.agent/data/fpgs-matrix-free-dense-explainer/scenarios/g1_flat/fpgs_matrix_free.json`
  - `.agent/data/fpgs-matrix-free-dense-explainer/scenarios/h1_tabletop/fpgs_dense_row.json`
  - `.agent/data/fpgs-matrix-free-dense-explainer/scenarios/h1_tabletop/fpgs_matrix_free.json`

## Key Findings From This Pass

- The capture helper can build real benchmark scenarios from the in-tree benchmark definitions, but the local environment needed transient `uv --with` dependencies:
  - `GitPython`
  - `usd-core`
  - `scipy`
- The first reviewable sizing slice is now data-backed:
  - `g1_flat` dense-row and matrix-free both warm to an 8-contact / 24-dense-row state with 49 articulation/world DOFs and no matrix-free free-rigid rows
  - `h1_tabletop` dense-row warms to 117 dense rows at 133 world DOFs
  - `h1_tabletop` matrix-free warms to 30 dense articulated rows plus 102 matrix-free free-rigid rows at the same 133 world DOFs
- The checked-in artifacts include actual solver allocation sizes for:
  - grouped `H_by_size`, `L_by_size`, `J_by_size`, `Y_by_size`
  - world dense buffers (`C`, `rhs`, `diag`, `impulses`) where present
  - articulated matrix-free world buffers (`J_world`, `Y_world`) and free-rigid MF buffers (`mf_*`) where present

## Recommended Next Pass

- Continue M3 or start M4 depending on doc drafting needs.
- If M3 continues, extend the same capture helper to the remaining comparison presets from the manifest:
  - `fpgs_dense_loop`
  - `fpgs_dense_streaming`
  - `fpgs_split`
- If M4 starts next, use the new scenario data to anchor the kernel-memory artifacts and explain why `h1_tabletop` matrix-free keeps only 30 dense articulated rows while routing 102 rows through the free-rigid MF path.

## Validation Evidence

- `uv run python scripts/analysis/bootstrap_fpgs_explainer_artifacts.py`
- `python -m json.tool .agent/data/fpgs-matrix-free-dense-explainer/schema/scenario-sizing.schema.json`
- `python -m json.tool .agent/data/fpgs-matrix-free-dense-explainer/schema/kernel-memory-analysis.schema.json`
- `uv run --with GitPython --with usd-core --with scipy python scripts/analysis/capture_fpgs_scenario_sizing.py`
- `python -m json.tool .agent/data/fpgs-matrix-free-dense-explainer/scenarios/g1_flat/fpgs_matrix_free.json`
- `python -m json.tool .agent/data/fpgs-matrix-free-dense-explainer/scenarios/h1_tabletop/fpgs_dense_row.json`
