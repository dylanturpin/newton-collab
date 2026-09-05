# FPGS persistent contacts: PR validation

Base: `e452daa3a67e008f9dcef06fe1b809b36cfd386a` (`fpgs-main`).

This change bounds finite-effort actuator reactions, combines body-pair reduction
with identity matching and warm starting, transports cached friction in the
solver's tangent frame, invalidates episode/model history, reports capacity
failures, and fixes reducer table exhaustion when replicating environments.
Iteration counts, warm-start defaults and friction defaults are unchanged.

## Regression attribution

The original stack outlier was **not sufficient evidence of a new regression**.
A matched repeat experiment on the unchanged base reproduced the same severe
failure: 1.377 m sink, 0.757 m drift and 7.358 m/s peak linear speed, with finite
state. This occurred in repeat 10 (zero-based) of 20 runs. The other baseline
runs show the smaller, preexisting warm-start convergence problem.

Inspection also found a separate bug in this branch's new tangent transport:
collision normals point A-to-B, whereas solver Jacobians construct tangents
from B-to-A normals. Using the former reverses the off-diagonal terms of the
basis change. The implementation now negates both cached and current normals
before constructing tangent bases. The regression reconstructs world impulse
from the actual Jacobian rows, independently projects it into the new contact
plane, and tests CPU and CUDA. The previous seam-only test did not exercise
mixing between the two tangent coordinates.

The strengthened regression fails against the immutable pre-fix snapshot with
0.0287 N·s maximum component error for a 0.5 N·s cached tangent impulse. It
passes after the fix. See [pre-fix evidence](evidence/normal_convention_jacobian_before.log).
The normal-convention bug is fixed; it is not established as the cause of the
historical collapse, which is independently reproducible on the baseline.

## Repeated stack results

| Source | Runs | Runs with >0.1 m sink | Median peak drift | Median peak speed |
|---|---:|---:|---:|---:|
| Unchanged base | 20 | 1 | 39.88 mm | 0.3749 m/s |
| Pre-fix branch | 20 | 1 | 40.22 mm | 0.3729 m/s |
| Corrected branch | 20 | 2 | 38.76 mm | 0.3743 m/s |

[All 60 runs and source hashes](evidence/regression_comparison.json) are included.
`current` identifies the pre-fix v8 snapshot; `fixed` identifies corrected v9.
The [source verification](evidence/source_verification_v9.json) confirms all 17
changed Python files in the final CUDA snapshot match the PR workspace.
After the repeated runs, only the matcher method docstring was clarified;
this explains its different whole-file hash in the final source verification.

The typical drift and speed are comparable. These samples establish that the
collapse predates this PR; they do **not** establish equal rare-failure rates
or prove that the branch can never worsen an existing failure. The stack remains
an explicit release-level defect. The branch's demonstrable tangent-transport
error was corrected before publication.

## Matched experiment protocol

The same harness runs against separate source roots and fresh Warp caches.
Each case contains 128 environments with eight 0.2 m boxes per stack, friction
0.6, gravity plus one body weight of downward loading and 0.1 body weight of
horizontal loading. All versions use latest matching, no reduction, matrix-free
PGS, warm starting, 64 iterations and a 1/240 s timestep.

Each run advances 100 eager steps, then 400 steps through one two-step CUDA
graph. Metrics span the latter 400 steps relative to the initial pose; sink is
pose displacement, not a penetration measurement. Each source version runs 20
freshly constructed scenes. This is a native controlled scene, not Isaac Lab
certification or proof of stability for arbitrary workloads.

Reproduce on a CUDA machine from a Newton environment:

```sh
FPGS_SOURCE_ROOT=/absolute/path/to/source \
FPGS_IMPL_CACHE=/absolute/path/to/fresh/cache \
FPGS_ROLLING_OUT=/absolute/path/to/results.json \
uv run --extra dev python reports/fpgs_contact_implementation/evidence/regression_stack.py
```

## Verification

- CUDA host: RTX A6000, Python 3.11.15, Warp 1.16.0.
- Local: Apple CPU, Python 3.12, Warp 1.17.0.dev20260807.
- All **223 FPGS tests pass** on the CUDA host, including CPU and CUDA checks
  of the corrected tangent convention.
- All **128 collision tests pass** on that host. See the
  [validation log](evidence/cuda_validation_v9.log) (module-load noise removed;
  [complete compressed output](evidence/cuda_validation_v9.log.gz)).
- P25 retained the cube for **300 frames / 5 s** at 64 iterations,
  eight substeps and pre-elimination enabled. Final height was 132.34 mm;
  recorded body/object/pad poses and joints were finite. The probe uses
  gravity compensation and damping; it is not an unassisted grasp assertion.
  See [summary](evidence/p25_v9_summary.json) and [raw trajectory](evidence/p25_v9.npz).
- [All 30 rolling configurations](evidence/rolling_v9_compact.json) completed
  without observed nonfinite state or capacity overflow; every reduced case
  had zero fallback frames. All five warm-stack configurations still fail
  the provisional 20 mm drift / 0.1 m/s speed quality targets. Those targets
  were chosen after initial baseline measurements, not preregistered release
  acceptance criteria. Passing finite-state checks does not clear them.
- Local suite: **211 tests, no failures, 110 CUDA-dependent skips**.
- Pre-commit checks and Towncrier draft rendering pass.

Regression-first counterexamples were recorded before the corresponding fixes:
complete finite drive reaction (roughly 5,000 N against a 1 N limit), cached
friction direction and material cone, matching/reduction incompatibility,
model-change invalidation, missing capacity status, replicated reducer-table
fallback, and the intermediate tiled-drive serial fallback. The final fixtures
retain those contracts. The normal-convention counterexample above additionally
uses the actual solver Jacobian to avoid mirroring a basis-sign mistake in the test.

## Remaining release work

The stack convergence/collapse issue remains a release-level problem on the
baseline. A seeded CPU pile also diverges on both the base and this branch;
its cause is not attributed to this change. Neither issue is hidden by changing
iteration defaults, velocity caps, physical thresholds or test tolerances.

Earlier matched frozen-state throughput measurements showed roughly 3–11%
overhead on articulated scenes, largely associated with supporting live finite
drive rows. The reducer capacity fix avoids keep-all fallback at 1,024 simple
environments; reduction changes contact count and therefore must be assessed
with task-specific support and friction quality, not timing alone. Warm/reduced
support tests do not certify stack convergence, twisting friction, unassisted
grasp fidelity or real Isaac Lab workloads.
