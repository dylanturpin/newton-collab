# M1 Source Inventory

## Placement Decision

Use a dedicated sibling page under `docs/concepts/` for the dense-vs-matrix-free explainer.

Why:

- `docs/concepts/feather_pgs.md` is already the main theory and architecture page.
- The requested deliverable needs large scenario-backed tables and kernel-memory-layout analysis.
- Keeping the empirical comparison separate is additive, easier to review, and avoids overloading the overview page.

## Docs Entry Points

- `docs/concepts/feather_pgs.md`: current FeatherPGS overview and math/code mapping
- `docs/index.rst`: Concepts toctree entry for FeatherPGS

## Solver / Benchmark Sources

- `newton/_src/solvers/feather_pgs/solver_feather_pgs.py`: solver mode split, stage orchestration, dense kernels, matrix-free fused kernel factory
- `newton/_src/solvers/feather_pgs/kernels.py`: row-build and free-rigid matrix-free helper kernels
- `newton/tools/solver_benchmark.py`: scenario presets, solver presets, benchmark worker, `g1_flat`, `h1_tabletop`

## Dense Path Locations

- `_stage4_hinv_jt_tiled`, `_stage4_hinv_jt_par_row`
- `_stage4_delassus_tiled`, `_stage4_delassus_par_row_col`
- `_stage4_compute_rhs_world`, `_stage4_accumulate_rhs_world`
- `_dispatch_dense_pgs_solve`
- `_stage5_pgs_solve_world_loop`
- `_stage5_pgs_solve_world_tiled_row`
- `_stage5_pgs_solve_world_tiled_contact`
- `_stage5_pgs_solve_world_streaming`
- `_stage6_apply_impulses_world`

## Matrix-Free / Mixed Path Locations

- `step()` branch for `pgs_mode == "matrix_free"` including `S5_GatherJY`
- `_stage4_diag_from_JY`
- `_mf_pgs_setup`, `_mf_pgs_solve`, `_stage6b_mf_pgs`
- `TiledKernelFactory.get_pgs_solve_mf_gs_kernel`
- `TiledKernelFactory._build_pgs_solve_mf_gs_kernel`

## Existing Scenarios Relevant To The Explainer

- `g1_flat`
- `h1_tabletop`

Relevant preset labels already in `newton/tools/solver_benchmark.py`:

- `fpgs_dense_loop`
- `fpgs_dense_row`
- `fpgs_dense_streaming`
- `fpgs_split`
- `fpgs_matrix_free`

## Nightly / Publication Sources

- `benchmarks/nightly/nightly.yaml`: benchmark task inventory for `g1_flat` and `h1_tabletop`
- `benchmarks/nightly/publish.py`: durable site data update logic
- `benchmarks/nightly/index.html`: dashboard that consumes `runs.jsonl` and `points.jsonl`
- `.github/workflows/docs-release.yml`: release docs deploy pattern
- `.github/workflows/docs-dev.yml`: dev docs deploy pattern

## `gh-pages` Artifacts To Preserve Later

- `nightly/runs.jsonl`
- `nightly/points.jsonl`
- `nightly/runs/<run_id>/meta.json`
- `nightly/runs/<run_id>/summary.json`
- `nightly/runs/<run_id>/...` copied render/profile artifacts
- `nightly/index.html`
- `.nojekyll`

## Notes For M2+

- Do not reuse the existing docs deployment workflows as-is for the final `gh-pages` step; both workflows replace docs directories directly and do not reason about preserving `nightly/`.
- Prefer additive docs publication mechanics that touch only the docs-specific subtree chosen for this explainer.
