# FeatherPGS Dense vs Matrix-Free Explainer Handoff

## Current State

- Completed M1: source inventory and page-placement decision.
- Completed M2: schema/bootstrap tooling for the explainer raw-data lane.
- Advanced the first M3 slice: runtime-backed scenario sizing artifacts now exist for `g1_flat` and `h1_tabletop`.
- Advanced the first M4 slice: checked-in kernel-work and memory-layout artifacts now exist for the dense tiled-row PGS kernel and the fused matrix-free GS kernel.
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

## Key Findings From This Pass

- The first reviewable sizing slice remains the factual baseline:
  - `g1_flat` dense-row and matrix-free both warm to an 8-contact / 24-dense-row state with 49 articulation/world DOFs and no matrix-free free-rigid rows
  - `h1_tabletop` dense-row warms to 117 dense rows at 133 world DOFs
  - `h1_tabletop` matrix-free warms to 30 dense articulated rows plus 102 matrix-free free-rigid rows at the same 133 world DOFs
- The new M4 slice now ties those scenario counts to the actual stage-5/stage-6 kernels:
  - dense tiled-row PGS stages a packed lower triangle of the explicit Delassus matrix `C` in shared memory and recomputes `sum_j C_ij lambda_j` from that staged tile on every sweep
  - the fused matrix-free GS kernel stages the live velocity vector `v_out` in shared memory, streams `J_world` and `Y_world`, and recomputes `J_i v` rather than reading a dense row of `C`
  - the same fused kernel streams free-rigid `mf_J_*` and `mf_MiJt_*` data for the second phase while keeping free-rigid impulses in shared memory
- The first memory numbers worth carrying into the docs are now checked in rather than inferred:
  - `g1_flat` dense-row explicit `C`: 4096 B
  - `h1_tabletop` dense-row explicit `C`: 65536 B
  - `h1_tabletop` matrix-free articulated `J_world` + `Y_world`: 136192 B
  - `h1_tabletop` matrix-free free-rigid `mf_*` working set captured in the artifact summary: 63488 B

## Recommended Next Pass

- Start M5 and land the actual docs page.
- Use `.agent/data/fpgs-matrix-free-dense-explainer/m4-kernel-memory-summary.md` as the initial table/outline source and pull the deeper claims from:
  - `.agent/data/fpgs-matrix-free-dense-explainer/kernels/dense-row.json`
  - `.agent/data/fpgs-matrix-free-dense-explainer/kernels/matrix-free-gs.json`
- Only circle back to the remaining M4 planned artifacts if the docs draft clearly needs equal-depth treatment for:
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
