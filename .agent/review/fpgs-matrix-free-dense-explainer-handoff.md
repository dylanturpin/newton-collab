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
- Completed M10: the final published source/docs state now includes the added "Warp kernel migration (Zach Corse)" journal note, and both `origin/feather_pgs` and `origin/gh-pages` have been refreshed to that corrected state.
- Advanced M11 with a first reviewable clarity slice: the explainer page now has notation alignment with the main FeatherPGS concepts page, a clearer scenario-sizing table, and a profiler-backed explanation of why `split` is not equivalent to the fused `matrix_free` path.
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
- Added the journal follow-up note:
  - `docs/investigations/index.md` now includes "Warp kernel migration (Zach Corse)"
- Updated the investigations journal ordering:
  - `docs/investigations/index.md` now moves the TGS note to the bottom and marks it `[In-flight]`
- Updated the explainer page clarity and split-vs-fused analysis:
  - `docs/concepts/feather_pgs_dense_vs_matrix_free.md`
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
- Added corrected M10 publication audit artifacts:
  - `.agent/data/fpgs-matrix-free-dense-explainer/publication/m10-gh-pages-name-status.txt`
  - `.agent/data/fpgs-matrix-free-dense-explainer/publication/m10-gh-pages-nightly-diff.txt`
  - `.agent/data/fpgs-matrix-free-dense-explainer/publication/m10-publication-summary.env`
- Added a `typos` allowlist entry for the surname `Corse`:
  - `pyproject.toml`

## Key Findings From This Pass

- The explainer now includes a recommendation table covering `dense_loop`, `dense_row`, `dense_streaming`, `split`, and `matrix_free`, with explicit keep/deprecate/remove guidance tied back to the checked-in scenario and kernel artifacts.
- The new `docs/investigations/index.md` page establishes a dated journal structure and cross-references parallel solver workstreams so the dense-vs-matrix-free report sits in a broader decision log rather than as an isolated page.
- The final published source commit is now `1999330f643c5914e3d417675d1c8fd0967976f0`, not the earlier `ac1d83a0` draft publication. That corrected source commit includes the human-authored Warp migration journal note.
- The final published site commit is now `ff38559855cec148208e38f8df0f47dd30f89f1a`, superseding the earlier in-session `gh-pages` update so the rendered docs match the corrected source branch.
- The safe publication path still worked as intended: only the `latest/` dev-docs payload and `.nojekyll` were refreshed in the final push, while branch-root `404.html`, `nightly/`, `index.html`, `stable/`, versioned release docs, and `switcher.json` were preserved.
- `nightly/` was preserved exactly. The checked-in `m10-gh-pages-nightly-diff.txt` artifact is empty, which confirms that the final publication did not touch the benchmark dashboard data.
- The investigations journal now includes the Zach Warp migration note in the published site in addition to the dense-vs-matrix-free explainer and related solver investigation links.
- The explainer page now uses the same `\hat{\dot{q}}` / `p` notation as the main FeatherPGS concepts page instead of mixing that notation with `\lambda`-centric phrasing.
- The scenario table is now easier to read because it distinguishes articulated rows from free-rigid matrix-free rows, adds warm-contact totals, and uses human-readable sizes instead of raw byte counts.
- The checked-in `h1_tabletop` captures do not represent exactly the same warm state: the dense snapshot has 229 warm contacts, while the matrix-free snapshot has 235. The revised docs now say that explicitly instead of implying the row-total difference is purely routing.
- The fetched `origin/gh-pages` `h1_tabletop` RTX 5090 trace JSONs now provide a concrete argument against treating `split` as equivalent to the fused `matrix_free` path:
  - `split` spends about 7.33 s of measured solver-kernel time and repeatedly relaunches `pgs_solve_tiled_contact_*`, `rhs_accum_world_*`, `apply_impulses_world_*`, and `pgs_solve_mf_*`
  - `matrix_free` spends about 3.46 s and concentrates its hot loop in `pgs_solve_mf_gs_*`
  - this supports the docs claim that the current win is tied to fused kernel structure, not only to rigid-row routing

## Recommended Next Pass

- Finish M11 by rerunning the local docs build and `uvx pre-commit run -a` on top of the clarity edits, then commit the slice locally if validation passes.
- If a later worker starts M12, treat it as a fresh publication milestone. Rebuild from the committed M11 source state and repeat the same nightly-preserving `origin/gh-pages` procedure used in M8/M10.

## Validation Update

- `uv run --extra docs --extra sim sphinx-build -j auto -b html docs docs/_build/html`
  - result: passed
  - note: Sphinx still reports the pre-existing multiple-toctree consistency notices for generated `docs/api/newton_*.rst` pages
- `uvx pre-commit run -a`
  - result: passed

## Publication Record

- Final source publication:
  - before: `origin/feather_pgs` -> `ac1d83a000eb8eb531f2ef5c72f00760b84c2e5a`
  - command: `git push origin 1999330f:feather_pgs`
  - after: `origin/feather_pgs` -> `1999330f643c5914e3d417675d1c8fd0967976f0`
- Final docs rebuilds:
  - detached-source proof build:
    - `uv run --extra docs --extra sim sphinx-build -j auto -b html docs /tmp/fpgs-docs-html-m10-proof`
    - workdir: `/tmp/newton-fpgs-m10-src-final`
    - result: passed
  - publication payload build:
    - `uv run --extra docs --extra sim sphinx-build -j auto -b html docs /tmp/fpgs-docs-html-m10-final`
    - result: passed
- Final `gh-pages` publication:
  - before: `origin/gh-pages` -> `e682d645d84159b67a71e241696a304a5c4f4bd4`
  - commands:
    - `git worktree add --detach /tmp/newton-gh-pages-fpgs-explainer-m10-final origin/gh-pages`
    - `git -C /tmp/newton-gh-pages-fpgs-explainer-m10-final checkout -b fpgs-matrix-free-dense-explainer-gh-pages-m10-final`
    - replace `latest/` from `/tmp/fpgs-docs-html-m10-final/`
    - `git -C /tmp/newton-gh-pages-fpgs-explainer-m10-final add .nojekyll latest`
    - `git -C /tmp/newton-gh-pages-fpgs-explainer-m10-final commit -m "Publish FeatherPGS investigation docs"`
    - `git -C /tmp/newton-gh-pages-fpgs-explainer-m10-final push origin HEAD:gh-pages`
  - after: `origin/gh-pages` -> `ff38559855cec148208e38f8df0f47dd30f89f1a`
- Exact changed-path inventory:
  - `.agent/data/fpgs-matrix-free-dense-explainer/publication/m10-gh-pages-name-status.txt`
- Exact nightly-preservation proof:
  - `.agent/data/fpgs-matrix-free-dense-explainer/publication/m10-gh-pages-nightly-diff.txt`
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
- `uv run --extra docs --extra sim sphinx-build -j auto -b html docs /tmp/fpgs-docs-html-m10-proof`
  - result: passed
  - note: executed from detached worktree `/tmp/newton-fpgs-m10-src-final` to prove the final published source revision builds cleanly
- `uv run --extra docs --extra sim sphinx-build -j auto -b html docs /tmp/fpgs-docs-html-m10-final`
  - result: passed
  - note: used for the final corrected `gh-pages` publication payload
- `git -C /tmp/newton-gh-pages-fpgs-explainer-m10-final diff --name-only -- nightly`
  - result: no output
- `uvx pre-commit run -a`
  - first result: failed because `typos` flagged the surname `Corse`
  - second result: passed after allowlisting `Corse` in `pyproject.toml`
