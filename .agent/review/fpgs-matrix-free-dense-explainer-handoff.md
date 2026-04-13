# FeatherPGS Dense vs Matrix-Free Explainer Handoff

## Current State

- Completed M1: source inventory and page-placement decision.
- Completed M2: schema/bootstrap tooling for the explainer raw-data lane.
- Advanced the first M3 slice: runtime-backed scenario sizing artifacts now exist for `g1_flat` and `h1_tabletop`.
- Advanced the first M4 slice: checked-in kernel-work and memory-layout artifacts now exist for the dense tiled-row PGS kernel and the fused matrix-free GS kernel.
- Completed M5: the docs draft now exists as a sibling concepts page with navigation wiring.
- Completed M6: the docs build and required `pre-commit` validation now pass locally.
- Completed M7: the finalized source/docs state has now been published to `origin/feather_pgs`.
- Completed M8: the rendered docs have now been published safely to `origin/gh-pages` without touching the nightly benchmark payload.
- Completed M9: the docs now include an investigations journal and a decision-facing code-path ablation recommendation layer on top of the dense-vs-matrix-free explainer.
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
- Added the investigations journal and navigation wiring:
  - `docs/investigations/index.md`
  - `docs/index.rst`
- Added the code-path ablation and recommendation section to:
  - `docs/concepts/feather_pgs_dense_vs_matrix_free.md`
- Refreshed the kernel artifacts so their provenance matches the source revision used for the docs draft:
  - `.agent/data/fpgs-matrix-free-dense-explainer/kernels/dense-row.json`
  - `.agent/data/fpgs-matrix-free-dense-explainer/kernels/matrix-free-gs.json`
- Accepted formatting-only `ruff` rewrites in two helper scripts:
  - `scripts/analysis/capture_fpgs_kernel_memory_artifacts.py`
  - `scripts/analysis/capture_fpgs_scenario_sizing.py`
- Added `gh-pages` publication audit artifacts:
  - `.agent/data/fpgs-matrix-free-dense-explainer/publication/m8-gh-pages-name-status.txt`
  - `.agent/data/fpgs-matrix-free-dense-explainer/publication/m8-gh-pages-nightly-diff.txt`
  - `.agent/data/fpgs-matrix-free-dense-explainer/publication/m8-gh-pages-summary.env`

## Key Findings From This Pass

- The explainer now includes a recommendation table covering `dense_loop`, `dense_row`, `dense_streaming`, `split`, and `matrix_free`, with explicit keep/deprecate/remove guidance tied back to the checked-in scenario and kernel artifacts.
- The new `docs/investigations/index.md` page establishes a dated journal structure and cross-references parallel solver workstreams so the dense-vs-matrix-free report sits in a broader decision log rather than as an isolated page.
- This pass is intentionally validation-only. It does not republish `origin/feather_pgs` or `origin/gh-pages`; the next pass, if needed, is the new final publication milestone.
- The final publication milestone is complete: `origin/gh-pages` advanced from `cfd01f5f69909e9c90f7d637c6772d6f09df31e2` to `d7f564af85b2e4b298f1fe246149e6addc1a8565`.
- The safe publication path worked as intended: only branch-root `404.html`, branch-root `.nojekyll`, and the `latest/` dev-docs payload were refreshed.
- `nightly/` was preserved exactly. The checked-in `m8-gh-pages-nightly-diff.txt` artifact is empty, which confirms that the publication did not touch the benchmark dashboard data.
- The new explainer page is now live in the published site as `latest/concepts/feather_pgs_dense_vs_matrix_free.html`.

## Recommended Next Pass

- Activate M10 only if these M9 docs changes should be published.
- Reuse the established M7/M8 publication procedure: push source to `origin/feather_pgs`, rebuild into a clean temp tree, and refresh only `gh-pages/latest/` plus the small branch-root routing files.

## Publication Record

- Pre-M8 remote state:
  - `origin/feather_pgs` -> `8eac16b5b4cf564f60d704beeb7fe11f9ae89ed9`
  - `origin/gh-pages` -> `cfd01f5f69909e9c90f7d637c6772d6f09df31e2`
- Source publication command already completed before this pass:
  - `git push origin HEAD:feather_pgs`
  - result: `origin/feather_pgs` remained at `8eac16b5b4cf564f60d704beeb7fe11f9ae89ed9` during the M8 site publish
- M8 clean docs rebuild:
  - `uv run --extra docs --extra sim sphinx-build -j auto -b html docs /tmp/fpgs-docs-html`
  - result: passed
- M8 safe `gh-pages` publication commands:
  - `git worktree add --detach /tmp/newton-gh-pages-fpgs-explainer origin/gh-pages`
  - `git -C /tmp/newton-gh-pages-fpgs-explainer checkout -b fpgs-matrix-free-dense-explainer-gh-pages`
  - `git -C /tmp/newton-gh-pages-fpgs-explainer add 404.html .nojekyll latest`
  - `git -C /tmp/newton-gh-pages-fpgs-explainer commit -m "Publish FeatherPGS explainer docs"`
  - `git -C /tmp/newton-gh-pages-fpgs-explainer push origin HEAD:gh-pages`
  - result: fast-forward update from `cfd01f5f69909e9c90f7d637c6772d6f09df31e2` to `d7f564af85b2e4b298f1fe246149e6addc1a8565`
- Post-M8 remote state:
  - `origin/feather_pgs` -> `8eac16b5b4cf564f60d704beeb7fe11f9ae89ed9`
  - `origin/gh-pages` -> `d7f564af85b2e4b298f1fe246149e6addc1a8565`
- Exact changed-path inventory:
  - `.agent/data/fpgs-matrix-free-dense-explainer/publication/m8-gh-pages-name-status.txt`
- Exact nightly-preservation proof:
  - `.agent/data/fpgs-matrix-free-dense-explainer/publication/m8-gh-pages-nightly-diff.txt`
  - file size: `0` bytes

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
- `uv run --extra docs --extra sim sphinx-build -j auto -b html docs /tmp/fpgs-docs-html`
  - result: passed
  - note: used for the final M8 `gh-pages` publication so the published payload came from a clean output tree
- `git -C /tmp/newton-gh-pages-fpgs-explainer diff --name-only -- nightly`
  - result: no output
- `uvx pre-commit run -a`
  - first result: failed because `ruff` and `ruff format` rewrote two helper scripts
  - second result: passed
