# FeatherPGS Dense vs Matrix-Free Explainer Handoff

## Current State

- Completed M1: source inventory and page-placement decision.
- Completed M2: schema/bootstrap tooling for the explainer raw-data lane.
- Advanced the first M3 slice: runtime-backed scenario sizing artifacts now exist for `g1_flat` and `h1_tabletop`.
- Advanced the first M4 slice: checked-in kernel-work and memory-layout artifacts now exist for the dense tiled-row PGS kernel and the fused matrix-free GS kernel.
- Completed M5: the docs draft now exists as a sibling concepts page with navigation wiring.
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

## Key Findings From This Pass

- The docs deliverable now has a concrete landing page rather than only `.agent` artifacts.
- The first reviewable sizing slice remains the factual baseline:
  - `g1_flat` dense-row and matrix-free both warm to an 8-contact / 24-dense-row state with 49 articulation/world DOFs and no matrix-free free-rigid rows
  - `h1_tabletop` dense-row warms to 117 dense rows at 133 world DOFs
  - `h1_tabletop` matrix-free warms to 30 dense articulated rows plus 102 matrix-free free-rigid rows at the same 133 world DOFs
- The new docs page carries the central explanation into reviewable prose and tables:
  - it states the exact dense update `w_i = rhs_i + \sum_j C_{ij}\lambda_j`
  - it states the articulated matrix-free update `w_i = rhs_i + J_i v` with `v += Y_i \Delta\lambda_i`
  - it explains that matrix-free still materializes `J` and `Y`, but does not materialize the dense `C`
- The M4 slice now ties those scenario counts to the actual stage-5/stage-6 kernels:
  - dense tiled-row PGS stages a packed lower triangle of the explicit Delassus matrix `C` in shared memory and recomputes `sum_j C_ij lambda_j` from that staged tile on every sweep
  - the fused matrix-free GS kernel stages the live velocity vector `v_out` in shared memory, streams `J_world` and `Y_world`, and recomputes `J_i v` rather than reading a dense row of `C`
  - the same fused kernel streams free-rigid `mf_J_*` and `mf_MiJt_*` data for the second phase while keeping free-rigid impulses in shared memory
- The first memory numbers now appear in the docs draft rather than only the artifact summary:
  - `g1_flat` dense-row explicit `C`: 4096 B
  - `h1_tabletop` dense-row explicit `C`: 65536 B
  - `h1_tabletop` matrix-free articulated `J_world` + `Y_world`: 136192 B
  - `h1_tabletop` matrix-free free-rigid `mf_*` working set captured in the artifact summary: 63488 B
- The stale kernel provenance called out by the judge is fixed: both kernel artifacts now record `git_commit: "54bb7277"` instead of `93f67400`.

## Recommended Next Pass

- Advance M6 and run the real docs validation path:
  - `uv run --extra docs --extra sim sphinx-build -j auto -b html docs docs/_build/html`
  - `uvx pre-commit run -a`
- Fix any rendering or lint issues from the new page before considering M6 complete.
- Only circle back to the remaining M4 planned artifacts if the docs build or review shows the draft still needs equal-depth treatment for:
  - `fpgs_dense_streaming`
  - `fpgs_dense_loop`
  - the split/tiled-contact fused dense kernel

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
