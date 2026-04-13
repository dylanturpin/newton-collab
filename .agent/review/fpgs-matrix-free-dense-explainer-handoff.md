# FeatherPGS Dense vs Matrix-Free Explainer Handoff

## Current State

- Completed M1: source inventory and page-placement decision.
- Completed M2: schema/bootstrap tooling for the explainer raw-data lane.
- Chosen docs structure: keep `docs/concepts/feather_pgs.md` as the overview page and add a sibling concepts page for the deep dense-vs-matrix-free comparison.
- Recorded the initial inventory in `.agent/data/fpgs-matrix-free-dense-explainer/m1-source-inventory.md`.
- Added checked-in schemas in `.agent/data/fpgs-matrix-free-dense-explainer/schema/`.
- Added the generated manifest `.agent/data/fpgs-matrix-free-dense-explainer/m2-capture-manifest.json`.

## Key Findings From This Pass

- The raw-data lane now has a fixed contract before any scenario capture starts:
  - scenario sizing artifact schema
  - kernel memory-analysis schema
  - generated manifest enumerating `g1_flat`, `h1_tabletop`, and the five comparison presets
- `newton/tools/solver_benchmark.py` is import-safe for this bootstrap use case, so the manifest can stay derived from the in-tree benchmark source instead of duplicating scenario/preset metadata.
- The manifest is already useful for M3/M4 because it predeclares the expected scenario and kernel output paths, reducing naming churn across later passes.

## Recommended Next Pass

- Advance M3.
- Add runtime-backed extraction that writes real scenario artifacts under:
  - `.agent/data/fpgs-matrix-free-dense-explainer/scenarios/g1_flat/`
  - `.agent/data/fpgs-matrix-free-dense-explainer/scenarios/h1_tabletop/`
- Start with one reviewable capture slice if needed:
  - dense vs matrix-free logical buffer sizing for `g1_flat`
  - then extend to `h1_tabletop`

## Validation Evidence

- `uv run python scripts/analysis/bootstrap_fpgs_explainer_artifacts.py`
- `python -m json.tool .agent/data/fpgs-matrix-free-dense-explainer/schema/scenario-sizing.schema.json`
- `python -m json.tool .agent/data/fpgs-matrix-free-dense-explainer/schema/kernel-memory-analysis.schema.json`
