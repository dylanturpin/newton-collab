# FeatherPGS Dense vs Matrix-Free Explainer Handoff

## Current State

- Completed M1: source inventory and page-placement decision.
- Chosen docs structure: keep `docs/concepts/feather_pgs.md` as the overview page and add a sibling concepts page for the deep dense-vs-matrix-free comparison.
- Recorded the initial inventory in `.agent/data/fpgs-matrix-free-dense-explainer/m1-source-inventory.md`.

## Key Findings From This Pass

- The existing FeatherPGS page already covers the shared formulation, mode taxonomy, and math-to-code mapping.
- The dense-vs-matrix-free explainer needs scenario-backed tables and kernel-memory-layout analysis substantial enough to warrant a separate page.
- `newton/tools/solver_benchmark.py` already contains the right scenario and preset surface for later sizing/data-capture work, including `g1_flat`, `h1_tabletop`, and explicit dense/streaming/matrix-free preset names.
- Nightly publication already has a durable data model in `benchmarks/nightly/publish.py`; later `gh-pages` work must preserve `nightly/runs.jsonl`, `nightly/points.jsonl`, and per-run assets under `nightly/runs/`.
- Existing docs deployment workflows are not safe templates for the final publication step because they replace docs directories on `gh-pages` without any nightly-specific preservation logic.

## Recommended Next Pass

- Advance M2.
- Add a small extraction script and raw artifact schema under `.agent/data/fpgs-matrix-free-dense-explainer/`.
- Make the first artifact describe scenario sizing and buffer/layout fields needed for `g1_flat` and `h1_tabletop`.
