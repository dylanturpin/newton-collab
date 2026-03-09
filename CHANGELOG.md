# Changelog

## [Unreleased]

### Added

- Add optional Nsight profiling artifacts to nightly benchmark publication, including `.nsys-rep` and Perfetto JSON links in the nightly dashboard.
- Add a post-run nightly Nsight kernel-summary pass that derives per-kernel timing JSON from published `.nsys-rep` artifacts for dashboard enrichment.

### Changed

- Default nightly benchmark workers to CUDA-graph execution with pipeline collision enabled, extend H1 FeatherPGS ablations with double-buffer and pipeline-collide stages, add a G1 pipeline-collide ablation step after parallel streams, and default the nightly shell wrappers to cherry-pick `fast-bulk-replicate`.
- Present nightly profiling artifacts as a compact ablation table with Env-FPS and direct artifact links so profiled runs are easier to scan in the dashboard.
- Default the nightly dashboard to H1 Tabletop on first load, narrow the kernel-detail view to the final Newton ablation plus MuJoCo baseline, and hide low-signal kernel rows below a 3% cutoff.

### Deprecated

### Removed

### Fixed

- Keep nightly MuJoCo benchmark jobs off the FeatherPGS-only pipeline-collision path, and route nightly renders through the known-good RTX PRO 6000 render lane so nightly runs produce stable videos and benchmark results.
## [1.0.0] - YYYY-MM-DD

Initial public release.
