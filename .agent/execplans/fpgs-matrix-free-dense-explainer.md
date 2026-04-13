# ExecPlan: FeatherPGS Dense vs Matrix-Free Explainer

## Objective

Produce a deep technical documentation deliverable for the FeatherPGS line that compares the dense and matrix-free formulations with concrete, scenario-backed data. The final deliverable may be either:

- a substantial new section in `docs/concepts/feather_pgs.md`, or
- a separate page linked from the FeatherPGS docs,

whichever yields the cleaner final documentation structure.

This lane is for documentation, data capture, and safe publication mechanics. It is not a general solver refactor lane unless limited code changes are needed to extract, validate, or present the required data.

## Workspace Facts

- Repo: `newton-collab` fork at `/home/dturpin/repos/newton-collab-fpgs-explainer`
- Working branch: `dturpin/fpgs-matrix-free-dense-explainer`
- Base line of work: `feather_pgs`
- Fork remote: `origin = git@github.com:dylanturpin/newton-collab.git`
- Upstream remote: `upstream = https://github.com/newton-physics/newton.git`
- Existing FeatherPGS doc page: `docs/concepts/feather_pgs.md`
- Nightly benchmark plan: `benchmarks/nightly/nightly.yaml`
- Nightly benchmark site/artifacts source branch: `origin/gh-pages`
- Optional representative Isaac Lab repo: `~/repos/il-newton-dev`

## Non-Negotiable Guardrails

- Do not create a GitHub issue.
- Do not open a PR.
- Never push to `upstream`.
- Do not publish to `gh-pages` until the docs work is actually complete.
- Preserve nightly benchmark data on `gh-pages`; any eventual docs update must be docs-only and must not clobber JSONL history, plots, or benchmark assets.
- Treat raw data as a first-class deliverable, not just prose support.
- Main comparison is dense vs matrix-free. Hybrid may be mentioned briefly with caveats.
- Keep the plan and workflow self-contained so a new contributor can continue from the workspace.

## Push Policy

There is an explicit constraint conflict:

- requested workflow behavior: commit and push at milestones
- repository safety rule from the user: only the final worker, when the work is actually complete, may push to fork branches `feather_pgs` and `gh-pages`

This ExecPlan resolves that conflict conservatively:

- milestone work may be committed locally on `dturpin/fpgs-matrix-free-dense-explainer`
- remote pushes are forbidden until the relevant final milestone is complete
- the only allowed remote pushes are:
  - source/docs changes to `origin/feather_pgs`
  - docs site changes to `origin/gh-pages`
- no other remote branches may be pushed from this workflow unless the user changes policy explicitly

## Required Deliverables

1. A living technical explainer in the docs comparing dense and matrix-free FeatherPGS.
2. Raw data artifacts under version control for representative scenarios, including logical array shapes and realistic sizing.
3. Clear discussion of the actual work performed by each path:
   - matrix/vector objects
   - multiplies and accumulations
   - what is materialized vs recomputed
4. Memory-layout discussion for the main PGS solve kernels:
   - registers
   - shared memory
   - global memory
   - streamed vs preloaded data
5. Real scenario sizing for at least:
   - `g1_flat`
   - `h1_tabletop`
   - ideally one or more Isaac Lab tasks from `~/repos/il-newton-dev` if useful
6. The motivating observations:
   - matrix-free performs surprisingly well without shared memory
   - dense path has issues
   - streaming dense kernel does not fully resolve them
7. A small open-questions section if unexplained observations remain.
8. Local docs build validation with the real toolchain.
9. A safe, manual `gh-pages` update procedure or helper script that preserves nightly data.

## Evidence and Artifact Locations

- Living plan: `.agent/execplans/fpgs-matrix-free-dense-explainer.md`
- Workflow definition: `.agent/workflows/fpgs-matrix-free-dense-explainer.yaml`
- Handoff notes / human summary draft: `.agent/review/fpgs-matrix-free-dense-explainer-handoff.md`
- Raw data and derived tables: `.agent/data/fpgs-matrix-free-dense-explainer/`
- Optional helper scripts for extraction / sizing / gh-pages safety:
  - `scripts/analysis/`
  - `scripts/docs/`

If a different path is materially cleaner, update this section before moving files.

## Pass Update (2026-04-13, pass 10)

M10 is the active milestone for this pass. This pass will:

- commit the pending `docs/investigations/index.md` addition for the "Warp kernel migration (Zach Corse)" journal note before any new publication push
- publish that new source/docs commit to `origin/feather_pgs`
- rebuild the docs from that exact commit in a clean tree
- safely refresh `origin/gh-pages` while preserving `nightly/`
- record exact remote/build/publication results in the ExecPlan and handoff

## Milestones

### M0. Orchestration Setup

Status: complete

Definition of done:

- create this living ExecPlan
- create a milestone-driven workflow definition in the workspace
- start a workflow run attached to this workspace
- record any setup caveats for the next worker

### M1. Source Inventory and Page Placement Decision

Status: complete

Definition of done:

- inventory the exact solver kernels, docs sections, nightly configs, and `gh-pages` artifacts needed
- decide whether the deliverable belongs inside `docs/concepts/feather_pgs.md` or in a separate page
- document the decision and rationale here
- identify which existing branch edits in `docs/concepts/feather_pgs.md` should be retained, revised, or moved

Expected outputs:

- updated page-placement note in this ExecPlan
- initial artifact inventory under `.agent/data/...` or `.agent/review/...`

### M2. Data Extraction Tooling and Schema

Status: complete

Definition of done:

- add any helper scripts needed to extract logical shapes, storage sizes, and scenario-backed counts
- define the raw artifact schema plainly enough that later passes stay consistent
- prefer checked-in machine-readable artifacts such as JSON or CSV plus lightweight markdown summaries

Expected outputs:

- helper scripts if needed
- initial raw artifact schema and output location

Completed outputs:

- checked-in schema files under `.agent/data/fpgs-matrix-free-dense-explainer/schema/`:
  - `scenario-sizing.schema.json`
  - `kernel-memory-analysis.schema.json`
- bootstrap helper script:
  - `scripts/analysis/bootstrap_fpgs_explainer_artifacts.py`
- generated capture manifest:
  - `.agent/data/fpgs-matrix-free-dense-explainer/m2-capture-manifest.json`

Validation for this milestone:

- `uv run python scripts/analysis/bootstrap_fpgs_explainer_artifacts.py`
- `python -m json.tool .agent/data/fpgs-matrix-free-dense-explainer/schema/scenario-sizing.schema.json`
- `python -m json.tool .agent/data/fpgs-matrix-free-dense-explainer/schema/kernel-memory-analysis.schema.json`

### M3. Scenario-Backed Dense vs Matrix-Free Data Capture

Status: in progress

Definition of done:

- gather concrete sizing and storage data for `g1_flat` and `h1_tabletop`
- include array/tensor names, shapes, storage class, and size implications
- include at least one Isaac Lab task if it materially improves the comparison
- verify the captured data is tied to real scenarios, not just theoretical formulas

Expected outputs:

- versioned raw data artifacts
- concise notes explaining provenance and scenario assumptions

Completed slice (2026-04-13):

- added runtime-backed capture helper:
  - `scripts/analysis/capture_fpgs_scenario_sizing.py`
- captured the first reviewable dense-vs-matrix-free artifact set under:
  - `.agent/data/fpgs-matrix-free-dense-explainer/scenarios/g1_flat/fpgs_dense_row.json`
  - `.agent/data/fpgs-matrix-free-dense-explainer/scenarios/g1_flat/fpgs_matrix_free.json`
  - `.agent/data/fpgs-matrix-free-dense-explainer/scenarios/h1_tabletop/fpgs_dense_row.json`
  - `.agent/data/fpgs-matrix-free-dense-explainer/scenarios/h1_tabletop/fpgs_matrix_free.json`
- first scenario-backed counts now checked in:
  - `g1_flat`: 49 articulation DOFs, 24 dense rows in the warmed contact state, no matrix-free rows in the current single-articulation ground-contact setup
  - `h1_tabletop` dense-row: 133 world DOFs, 117 dense rows in the warmed contact state
  - `h1_tabletop` matrix-free: 133 world DOFs, 30 dense articulated rows plus 102 matrix-free free-rigid rows in the warmed contact state

Validation for this slice:

- `uv run --with GitPython --with usd-core --with scipy python scripts/analysis/capture_fpgs_scenario_sizing.py`
- `python -m json.tool .agent/data/fpgs-matrix-free-dense-explainer/scenarios/g1_flat/fpgs_matrix_free.json`
- `python -m json.tool .agent/data/fpgs-matrix-free-dense-explainer/scenarios/h1_tabletop/fpgs_dense_row.json`

Remaining M3 work:

- extend capture coverage to the remaining comparison presets from the manifest if they are still needed in the final explainer tables
- decide whether to add a larger-world sizing slice in addition to the current per-world captures
- fold the checked-in scenario data into doc-ready summary tables

### M4. Kernel Work / Memory Layout Analysis

Status: in progress

Definition of done:

- document the main dense and matrix-free PGS solve kernels used by the current FeatherPGS line
- explain which multiply / accumulation each path performs
- explain what resides in registers, shared memory, and global memory as implemented today
- call out what is streamed, what is preloaded, and what is recomputed
- explicitly address the surprising observations motivating the explainer

Expected outputs:

- artifact notes and/or tables that can be pulled into docs directly

Completed slice (2026-04-13):

- added a kernel-artifact generator:
  - `scripts/analysis/capture_fpgs_kernel_memory_artifacts.py`
- generated the first reviewable M4 kernel artifacts:
  - `.agent/data/fpgs-matrix-free-dense-explainer/kernels/dense-row.json`
  - `.agent/data/fpgs-matrix-free-dense-explainer/kernels/matrix-free-gs.json`
- generated a doc-ready summary note tying kernel behavior back to the scenario artifacts:
  - `.agent/data/fpgs-matrix-free-dense-explainer/m4-kernel-memory-summary.md`
- recorded the first code-backed memory-layout findings:
  - dense tiled-row stages the lower triangle of the explicit Delassus matrix `C` in shared memory and recomputes `sum_j C_ij lambda_j` from that staged tile on every sweep
  - articulated matrix-free stages the live world velocity vector `v` in shared memory, streams `J_world` and `Y_world`, and recomputes `J_i v` instead of reading a stored dense row of `C`
  - the fused matrix-free kernel also streams free-rigid `mf_J_*` and `mf_MiJt_*` data while keeping free-rigid impulses in shared memory
  - `h1_tabletop` is the key mixed-world case so far: 117 dense rows in `fpgs_dense_row` versus 30 articulated dense rows plus 102 free-rigid matrix-free rows in `fpgs_matrix_free`

Validation for this slice:

- `uv run python scripts/analysis/capture_fpgs_kernel_memory_artifacts.py`
- `python -m json.tool .agent/data/fpgs-matrix-free-dense-explainer/kernels/dense-row.json`
- `python -m json.tool .agent/data/fpgs-matrix-free-dense-explainer/kernels/matrix-free-gs.json`

Remaining M4 work:

- add the remaining planned kernel artifacts if the final docs draft still needs them:
  - `kernels/dense-loop.json`
  - `kernels/dense-streaming.json`
  - `kernels/tiled-contact-fused.json`
- decide in the docs draft whether the existing M4 slice is already sufficient to explain the main performance observation, or whether the dense-streaming kernel needs equal depth
- fold the kernel-memory notes into the docs page as tables and short code-path callouts

### M5. Draft the Explainer Content

Status: complete

Definition of done:

- land a coherent docs draft in the chosen page location
- make the dense vs matrix-free comparison the center of the page
- include raw-data-backed tables / figures / summaries as first-class content
- mention hybrid briefly with caveats only if it helps
- include open questions only where genuinely unresolved

Expected outputs:

- docs content in repo
- updated handoff summary draft

Completed outputs (2026-04-13):

- added the sibling explainer page:
  - `docs/concepts/feather_pgs_dense_vs_matrix_free.md`
- wired the new page into concepts navigation:
  - `docs/index.rst`
- added an overview-page pointer to the new explainer:
  - `docs/concepts/feather_pgs.md`
- refreshed the checked-in kernel artifacts so their provenance matches the source revision used for the docs draft:
  - `.agent/data/fpgs-matrix-free-dense-explainer/kernels/dense-row.json`
  - `.agent/data/fpgs-matrix-free-dense-explainer/kernels/matrix-free-gs.json`

### M6. Validate Docs Build and Tighten Presentation

Status: complete

Definition of done:

- build docs locally with the real docs toolchain:
  - `uv run --extra docs --extra sim sphinx-build -j auto -b html docs docs/_build/html`
- fix doc build issues caused by the explainer work
- run `uvx pre-commit run -a`
- update this ExecPlan and handoff notes with exact validation evidence

Expected outputs:

- passing local docs build
- lint / formatting evidence

Completed outputs (2026-04-13):

- ran the required local docs build successfully:
  - `uv run --extra docs --extra sim sphinx-build -j auto -b html docs docs/_build/html`
- ran `uvx pre-commit run -a`; the first pass reformatted two analysis helpers without changing behavior:
  - `scripts/analysis/capture_fpgs_kernel_memory_artifacts.py`
  - `scripts/analysis/capture_fpgs_scenario_sizing.py`
- reran `uvx pre-commit run -a` successfully on the formatted tree
- confirmed the explainer page builds into `docs/_build/html/concepts/feather_pgs_dense_vs_matrix_free.html`

Validation evidence:

- `uv run --extra docs --extra sim sphinx-build -j auto -b html docs docs/_build/html`
  - result: passed
  - notable output: Sphinx reported existing multiple-toctree consistency notices for several generated `docs/api/newton_*.rst` pages, but the build succeeded and the new explainer page emitted cleanly
- `uvx pre-commit run -a`
  - first result: failed because `ruff` / `ruff format` rewrote two helper scripts
  - second result: passed after accepting those formatting-only edits

### M7. Final Source-Branch Publication

Status: complete

Definition of done:

- docs work is actually complete
- source changes are committed and ready to publish from the working branch
- push only to `origin/feather_pgs`
- record exact push command/result in the handoff summary

This milestone must not start early.

Completed outputs (2026-04-13):

- confirmed the docs/content work was already complete on local branch `dturpin/fpgs-matrix-free-dense-explainer` at commit `ace07904`
- published that finalized source/docs state to `origin/feather_pgs`:
  - `git push origin HEAD:feather_pgs`
  - result: fast-forward update from `79f10ef3` to `ace07904`
- recorded the publication pass metadata in the living ExecPlan and handoff, then published the metadata commit to `origin/feather_pgs`:
  - `git push origin HEAD:feather_pgs`
  - result: fast-forward update from `ace07904` to the final M7 metadata commit on this branch

Validation evidence:

- `git status --short --branch`
  - result before publication: clean worktree on `dturpin/fpgs-matrix-free-dense-explainer`
- `git ls-remote --heads origin feather_pgs gh-pages dturpin/fpgs-matrix-free-dense-explainer`
  - result before publication: `origin/feather_pgs` at `79f10ef391c3931d135db76c6b9fe572c09895b3`, `origin/gh-pages` at `cfd01f5f69909e9c90f7d637c6772d6f09df31e2`
- `git push origin HEAD:feather_pgs`
  - first result: published the explainer deliverable commit `ace07904`
  - second result: published the follow-up metadata commit that marks M7 complete in-tree

### M8. Safe `gh-pages` Update

Status: complete

Definition of done:

- docs work is already complete and source branch publication is done
- update `origin/gh-pages` manually and safely without damaging nightly benchmark content
- preserve benchmark JSONL rows, plots, and unrelated site assets
- record exactly what was updated and what was intentionally left untouched

This milestone must not start early.

Completed outputs (2026-04-13):

- rebuilt the docs into a clean temporary output tree instead of publishing from the polluted in-repo build directory:
  - `uv run --extra docs --extra sim sphinx-build -j auto -b html docs /tmp/fpgs-docs-html`
- created a detached `gh-pages` worktree and refreshed only the dev-docs payload:
  - `git worktree add --detach /tmp/newton-gh-pages-fpgs-explainer origin/gh-pages`
  - replaced `latest/` from `/tmp/fpgs-docs-html/`
  - refreshed branch-root `404.html`
  - refreshed branch-root `.nojekyll`
- preserved the nightly benchmark payload intentionally:
  - `nightly/` was left untouched
  - branch-root `index.html`, `stable/`, versioned release docs, and `switcher.json` were left untouched
- published the safe site update:
  - `git -C /tmp/newton-gh-pages-fpgs-explainer checkout -b fpgs-matrix-free-dense-explainer-gh-pages`
  - `git -C /tmp/newton-gh-pages-fpgs-explainer add 404.html .nojekyll latest`
  - `git -C /tmp/newton-gh-pages-fpgs-explainer commit -m "Publish FeatherPGS explainer docs"`
  - `git -C /tmp/newton-gh-pages-fpgs-explainer push origin HEAD:gh-pages`
  - result: fast-forward update from `cfd01f5f69909e9c90f7d637c6772d6f09df31e2` to `d7f564af85b2e4b298f1fe246149e6addc1a8565`
- checked in publication artifacts for auditability:
  - `.agent/data/fpgs-matrix-free-dense-explainer/publication/m8-gh-pages-name-status.txt`
  - `.agent/data/fpgs-matrix-free-dense-explainer/publication/m8-gh-pages-nightly-diff.txt`
  - `.agent/data/fpgs-matrix-free-dense-explainer/publication/m8-gh-pages-summary.env`

Validation evidence:

- `git ls-remote --heads origin feather_pgs gh-pages`
  - result before M8 push: `origin/feather_pgs` at `8eac16b5b4cf564f60d704beeb7fe11f9ae89ed9`, `origin/gh-pages` at `cfd01f5f69909e9c90f7d637c6772d6f09df31e2`
- `uv run --extra docs --extra sim sphinx-build -j auto -b html docs /tmp/fpgs-docs-html`
  - result: passed
  - notable output: the same existing multiple-toctree consistency notices for generated `docs/api/newton_*.rst` pages appeared, but the build succeeded cleanly
- `git -C /tmp/newton-gh-pages-fpgs-explainer diff --name-only -- nightly`
  - result: no output
- `git -C /tmp/newton-gh-pages-fpgs-explainer diff --name-status cfd01f5f69909e9c90f7d637c6772d6f09df31e2 d7f564af85b2e4b298f1fe246149e6addc1a8565 > .agent/data/fpgs-matrix-free-dense-explainer/publication/m8-gh-pages-name-status.txt`
- `git -C /tmp/newton-gh-pages-fpgs-explainer diff --name-only cfd01f5f69909e9c90f7d637c6772d6f09df31e2 d7f564af85b2e4b298f1fe246149e6addc1a8565 -- nightly > .agent/data/fpgs-matrix-free-dense-explainer/publication/m8-gh-pages-nightly-diff.txt`
  - result: empty file, confirming no nightly-path edits
- `git ls-remote --heads origin feather_pgs gh-pages`
  - result after M8 push: `origin/feather_pgs` still at `8eac16b5b4cf564f60d704beeb7fe11f9ae89ed9`, `origin/gh-pages` at `d7f564af85b2e4b298f1fe246149e6addc1a8565`

## Current Placement Hypothesis

Decision after M1: keep `docs/concepts/feather_pgs.md` as the theory / architecture page and draft the dense-vs-matrix-free explainer as a dedicated sibling page under `docs/concepts/`.

Rationale:

- the current `docs/concepts/feather_pgs.md` already covers formulation, mode taxonomy, and code mapping at substantial length
- the requested deliverable needs scenario-backed tables, kernel-memory-layout discussion, and raw-data-driven comparisons that would make the existing page too long and harder to scan
- a sibling page keeps the current page stable as the conceptual entry point while allowing the new explainer to be explicitly data-heavy and benchmark-oriented
- the sibling-page path is additive and minimizes churn in already-useful narrative material on the current branch

Planned page structure:

- retain `docs/concepts/feather_pgs.md` as the main overview page
- add a new sibling page for the deep dense-vs-matrix-free comparison and link it from `docs/index.rst` and/or from the FeatherPGS overview page
- reuse short background snippets from `docs/concepts/feather_pgs.md` only where they improve local readability; do not duplicate the full theory section

Existing branch edits in `docs/concepts/feather_pgs.md`:

- retain: motivation/background, dense-vs-matrix-free math equivalence, mode taxonomy, mapping-from-math-to-code, and frame-convention notes
- revise later only as needed to add a short pointer to the new explainer page and tighten any wording that becomes redundant
- move: none verbatim for now; prefer leaving the existing narrative in place and building the empirical comparison as a new document

## M1 Source Inventory

Documentation entry points:

- current FeatherPGS overview page: `docs/concepts/feather_pgs.md`
- concepts navigation: `docs/index.rst`

Primary solver implementation locations:

- solver orchestration and mode split: `newton/_src/solvers/feather_pgs/solver_feather_pgs.py`
- row builders / supporting kernels: `newton/_src/solvers/feather_pgs/kernels.py`
- benchmark entry point and scenario presets: `newton/tools/solver_benchmark.py`

Dense articulated path locations to analyze in later milestones:

- stage-4 articulated operator build: `_stage4_hinv_jt_*`, `_stage4_delassus_*`, `_stage4_compute_rhs_world`, `_stage4_accumulate_rhs_world`
- dense PGS dispatch / kernels: `_dispatch_dense_pgs_solve`, `_stage5_pgs_solve_world_loop`, `_stage5_pgs_solve_world_tiled_row`, `_stage5_pgs_solve_world_tiled_contact`, `_stage5_pgs_solve_world_streaming`
- dense impulse application: `_stage6_apply_impulses_world`

Matrix-free / mixed-path locations to analyze in later milestones:

- articulated matrix-free gather and solve setup in `step()`: `S5_GatherJY`, `self._stage6_prepare_world_velocity()`, and the `pgs_mode == "matrix_free"` branch
- diagonal-only articulated setup: `_stage4_diag_from_JY`
- free-rigid matrix-free setup / solve: `_mf_pgs_setup`, `_mf_pgs_solve`, `_stage6b_mf_pgs`
- fused two-phase dense+MF kernel factory: `TiledKernelFactory.get_pgs_solve_mf_gs_kernel()` and `_build_pgs_solve_mf_gs_kernel()`

Scenarios and benchmark presets already present:

- `g1_flat` and `h1_tabletop` are already first-class scenarios in `newton/tools/solver_benchmark.py`
- solver presets already distinguish `fpgs_dense_loop`, `fpgs_dense_row`, `fpgs_dense_streaming`, `fpgs_split`, and `fpgs_matrix_free`
- scenario defaults already encode different dense-constraint budgets and note that `h1_tabletop` routes rigid-body contacts to the matrix-free path

Nightly benchmark planning and publication touchpoints:

- benchmark plan: `benchmarks/nightly/nightly.yaml`
- nightly dashboard source: `benchmarks/nightly/index.html`
- nightly publication logic: `benchmarks/nightly/publish.py`
- local/slurm wrappers: `benchmarks/nightly.sh`, `benchmarks/nightly_slurm.sh`

Concrete nightly artifacts that must be preserved when `gh-pages` is eventually updated:

- `nightly/runs.jsonl`
- `nightly/points.jsonl`
- `nightly/runs/<run_id>/meta.json`
- `nightly/runs/<run_id>/summary.json`
- `nightly/runs/<run_id>/...` render/profile artifacts copied by `benchmarks/nightly/publish.py`
- `nightly/index.html`
- branch-root `.nojekyll`

Existing docs deployment workflows that currently touch `gh-pages` and therefore must not be reused blindly for the nightly-preserving update:

- `.github/workflows/docs-release.yml`
- `.github/workflows/docs-dev.yml`

These workflows replace `stable/`, versioned docs, or `latest/` directly on `gh-pages`; they are useful references for docs build mechanics but not yet safe publication procedures for this explainer lane.

### M9. Add Investigations Journal + Code-Path Ablation Recommendation

Status: complete

This milestone addresses a standing team action item from a prior meeting:

> Create a report with ablations / reasoning for choice of code paths.
> Then we can finally clean them out from our code base.
> Maybe have one branch that maintains old paths.
> Already pretty easy to collapse the small kernel choice branching.
> Harder one is the top-level: dense-delassus vs. matrix-free.
> Gather enough info to make the decision.
> Can think of main having a "distilled" version.

The existing M1-M8 explainer page (`docs/concepts/feather_pgs_dense_vs_matrix_free.md`) already has the technical sizing data and kernel analysis. This milestone adds the *decision-facing* layer on top.

Definition of done:

1. Add a `docs/investigations/` directory with an `index.md` journal page.
   - The journal page has dated section headers with brief summaries and links to individual investigation pages.
   - The first entry (2026-04-13) should link to the existing explainer page and reference results from parallel workstreams.

2. Add a **code-path ablation and recommendation** section to the existing explainer page (or a new companion page linked from it). This section should:
   - Present a clear recommendation table: for each code path (`dense_loop`, `dense_row`, `dense_streaming`, `split`, `matrix_free`), state whether to KEEP, DEPRECATE, or REMOVE, with one-line reasoning.
   - Reference the scenario sizing data already checked in under `.agent/data/` (link to the JSON artifacts).
   - Reference the kernel memory analysis from M4.
   - Note that the private API cleanup on branch `dturpin/fpgs-private-api-matrix-free` has already collapsed the public surface to matrix-free-only, providing implementation evidence that the collapse is feasible.
   - Note the velocity spike investigation finding that the dominant spike class is unconstrained v_hat (not PGS divergence), which is solver-path-independent — this means the choice between dense and matrix-free does not affect the spike behavior.
   - Include a brief "what to keep on a legacy branch" note for any deprecated paths.

3. Wire `docs/investigations/index.md` into `docs/index.rst` as a new toctree section called "Investigations".

4. The journal index entry for 2026-04-13 should include brief summaries (2-3 lines each) for:
   - Dense vs matrix-free explainer (this page) — link to `concepts/feather_pgs_dense_vs_matrix_free`
   - Velocity spike root-cause analysis — summarize: dominant class is unconstrained v_hat, PGS converges in 8 iterations, post-solve clamp + compliance recommended. Branch: `dt/velocity-spike-claude`.
   - TGS feasibility study — summarize: equivalent physics at ~2x cost, velocity guard is dominant training blocker. Branch: `dt/tgs-feather-pgs-study`. (Results still in progress.)
   - Private API matrix-free cleanup — summarize: collapsed public FeatherPGS surface to matrix-free-only, workflow accepted. Branch: `dturpin/fpgs-private-api-matrix-free`.

5. Run docs build and pre-commit. Do NOT re-publish to `gh-pages` yet — just validate the build.

Expected outputs:
- `docs/investigations/index.md`
- Updated `docs/concepts/feather_pgs_dense_vs_matrix_free.md` with ablation/recommendation section
- Updated `docs/index.rst` with investigations toctree
- Passing docs build

Completed outputs (2026-04-13):

- added the investigations journal landing page:
  - `docs/investigations/index.md`
- added a code-path ablation and recommendation section to the explainer page:
  - `docs/concepts/feather_pgs_dense_vs_matrix_free.md`
- wired the investigations journal into the docs navigation:
  - `docs/index.rst`
- updated the plan and handoff to record the new milestone state and follow-up publication gate

Validation evidence:

- `uv run --extra docs --extra sim sphinx-build -j auto -b html docs docs/_build/html`
- `uvx pre-commit run -a`

### M10. Final Publication (source branch + gh-pages)

Status: in progress

Same publication procedure as M7+M8: push source to `origin/feather_pgs`, rebuild docs, safely update `origin/gh-pages` preserving nightly data.

## Immediate Next Action

Commit the pending investigations journal update, then republish M10 from that new commit so both `origin/feather_pgs` and `origin/gh-pages` include the Zach Warp migration note.

## Change Log

- 2026-04-13: Created ExecPlan, recorded branch/remotes/guardrails, and set up workflow-driven milestone execution for this workspace.
- 2026-04-13: Completed M1 source inventory, chose a dedicated sibling docs page for the dense-vs-matrix-free explainer, and recorded the nightly artifacts that later publication steps must preserve.
- 2026-04-13: Completed M5 by landing the sibling dense-vs-matrix-free explainer page, wiring docs navigation, and refreshing kernel-artifact provenance for the checked-in evidence bundle.
- 2026-04-13: Completed M7 by publishing the finalized source/docs state to `origin/feather_pgs` and leaving `origin/gh-pages` untouched for the later gated publication milestone.
- 2026-04-13: Completed M8 by rebuilding the docs into `/tmp/fpgs-docs-html`, publishing the refreshed `latest/` payload to `origin/gh-pages`, and checking in exact changed-path and nightly-preservation artifacts.
