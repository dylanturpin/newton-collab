# Newton Development Guidelines

- `newton/_src/` is internal. Examples and docs must not import from `newton._src`. Expose user-facing symbols via public modules (`newton/geometry.py`, `newton/solvers.py`, etc.).
- Breaking changes require a deprecation first. Do not remove or rename public API symbols without deprecating them in a prior release.
- Prefix-first naming for autocomplete: `ActuatorPD` (not `PDActuator`), `add_shape_sphere()` (not `add_sphere_shape()`).
- Prefer nested classes for self-contained helper types/enums.
- PEP 604 unions (`x | None`, not `Optional[x]`).
- Annotate Warp arrays with bracket syntax (`wp.array[wp.vec3]`, `wp.array2d[float]`, `wp.array[Any]`), not the parenthesized form (`wp.array(dtype=...)`). Use `wp.array[X]` for 1-D arrays, not `wp.array1d[X]`.
- Follow Google-style docstrings. Types in annotations, not docstrings. `Args:` use `name: description`.
  - Sphinx cross-refs (`:class:`, `:meth:`) with shortest possible targets. Prefer public API paths; never use `newton._src`.
  - SI units for physical quantities in public API docstrings: `"""Particle positions [m], shape [particle_count, 3]."""`. Joint-dependent: `[m or rad]`. Spatial vectors: `[N, N·m]`. Compound arrays: per-component. Skip non-physical fields.
- Code comments: brief, and only for non-obvious code. Explain *why* (intent, constraints, edge cases), not *what* the code already shows. Prefer a cross-reference (doc, `:class:`/`:meth:`) over re-explaining context.
- Run `docs/generate_api.py` when adding public API symbols.
- Before relying on or changing a documented claim, open the relevant internal cross-references and external primary-source links. Verify Newton-specific behavior against the current code; if a linked source is unavailable, state that limitation instead of assuming it supports the claim.
- Avoid new required dependencies. Strongly prefer not adding optional ones — use Warp, NumPy, or stdlib.
- Create a feature branch before committing — never commit directly to `main`. Use `<username>/feature-desc`.
- Imperative mood in commit messages ("Fix X", not "Fixed X"), ~50 char subject, body wraps at 72 chars explaining _what_ and _why_.
- Verify regression tests fail without the fix before committing.
- Pin GitHub Actions by SHA: `action@<sha>  # vX.Y.Z`. Check `.github/workflows/` for allowlisted hashes.
- In SPDX copyright lines, use the year the file was first created. Do not create date ranges or update the year when modifying a file.

Run `uvx pre-commit run -a` to lint/format before committing. Use `uv` for all commands; fall back to `venv`/`conda` if unavailable.

```bash
# Examples
uv sync --extra examples
uv run -m newton.examples basic_pendulum
```

## Collab-fork solver branches (skild workstream)

This fork's solver work is consumed as a **git submodule of `skild-ai/skild-IL-solver`**,
whose production branch (`fpgs-main`) pins exact commits of the branches below. That
changes the branch rules:

- Lineage (full map: `reports/vishal/robotiq/branches.md` in skild-IL-solver):
  base `3ee2ff96` → `vishal/fpgs-mimic-rows` (FPGS mimic/connect rows, passive joint
  springs, CRBA loop-joint indexing fix) → `vishal/fpgs-preelim` (bilateral
  pre-elimination). `vishal/kamino-robotiq` is a pinned workspace at the mimic-rows tip.
  New solver features branch off the current validated tip (`<username>/<feature-desc>`
  per the rule above).
- **Push before pointer.** Push the newton-collab branch to origin *before* committing
  the parent-repo submodule pointer that references it — otherwise fresh parent
  checkouts fail at submodule init.
- **Never force-push or rewrite a branch whose commits a parent pointer references.**
  Remote GC can prune orphaned SHAs and break every parent checkout pinned to them.
  When splitting history for upstream PRs (`git rebase --onto origin/main 3ee2ff96`),
  create *new* branches and leave the referenced ones intact.
- **Validation gates before a tip may become a parent pointer**: the FPGS test modules
  pass (`uv run --extra dev python -m unittest newton.tests.test_feather_pgs_mimic
  newton.tests.test_feather_pgs_connect newton.tests.test_feather_pgs_springs
  newton.tests.test_feather_pgs_preelim`), regression-first discipline holds (tests
  verified failing without the change), and solver changes that affect grasp behavior
  pass the parent repo's P25 sustained-squeeze gate
  (`P25_FPGS_PREELIM=1 P25_FPGS_ITERS=64 ... p25_curling_grasp.py --solver feather_pgs
  --object pinch --headless --steps 300 --close-at 60 --rigid-gap 0.005` must hold the
  cube).
- **Warp kernel cache when bisecting generated-source edits** (e.g. the FPGS sweep-phase
  filters): stale caches in `~/.cache/warp` silently run the old kernels — clear the
  cache between bisection steps.

## Tests

Always use `unittest`, not pytest.

```bash
uv run --extra dev -m newton.tests
uv run --extra dev -m newton.tests -k test_viewer_log_shapes           # specific test
uv run --extra dev -m newton.tests -k test_basic.example_basic_shapes  # example test
uv run --extra dev --extra torch-cu12 -m newton.tests                  # with PyTorch
```

### Testing guidelines

- Give every test function or method a docstring using triple double quotes (`"""..."""`). Start with a concise one-line summary in imperative mood that states what the test verifies. For a particularly complex test, add a body that elaborates on the tested behavior, separated from the summary by a blank line following Google-style docstring conventions.
- Never call `wp.synchronize()` or `wp.synchronize_device()` right before `.numpy()` on a Warp array. This is redundant as `.numpy()` performs a synchronous device-to-host copy that completes all outstanding work.

```bash
# Benchmarks
uvx --with virtualenv asv run --launch-method spawn main^!
```

## PR Instructions

- If opening a pull request on GitHub, use the template in `.github/PULL_REQUEST_TEMPLATE.md`.
- If a change modifies user-facing behavior, insert an entry at a random position within the correct category (`Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`) in `CHANGELOG.md`'s `[Unreleased]` section. Use imperative present tense ("Add X") and avoid internal implementation details.
- For `Deprecated`, `Changed`, and `Removed` entries, include migration guidance: "Deprecate `Model.geo_meshes` in favor of `Model.shapes`".

## Examples

- Follow the `Example` class format.
  - Implement `test_final()` — runs after the example completes to verify simulation state is valid.
  - Optionally implement `test_post_step()` — runs after every `step()` for per-step validation.
- Register in `README.md` with `python -m newton.examples <name>` command and a 320x320 jpg screenshot.
