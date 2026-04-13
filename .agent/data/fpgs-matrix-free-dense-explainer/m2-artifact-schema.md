# M2 Artifact Schema

This milestone fixes the machine-readable artifact layout before runtime-backed
capture begins.

## Artifact Root

- Root: `.agent/data/fpgs-matrix-free-dense-explainer/`
- Schemas: `.agent/data/fpgs-matrix-free-dense-explainer/schema/`
- Generated manifest: `.agent/data/fpgs-matrix-free-dense-explainer/m2-capture-manifest.json`
- Scenario-backed raw captures planned for M3:
  - `.agent/data/fpgs-matrix-free-dense-explainer/scenarios/<scenario>/<preset>.json`
- Kernel/layout notes planned for M4:
  - `.agent/data/fpgs-matrix-free-dense-explainer/kernels/<kernel-id>.json`

## Scenario Sizing Artifact

Schema file: `schema/scenario-sizing.schema.json`

Purpose:

- record one real scenario + solver preset capture
- hold both logical sizing counts and named solver buffers
- keep dense and matrix-free artifacts comparable without forcing every field to
  be populated in every mode

Key top-level sections:

- `provenance`: generator script, git commit, source files, capture timestamp
- `scenario`: benchmark scenario metadata copied from `newton/tools/solver_benchmark.py`
- `solver_preset`: selected preset metadata and effective mode
- `capture`: runtime knobs such as `num_worlds`
- `world_counts`: realized counts from the capture
- `logical_buffers`: named arrays/buffers with shapes, dtype, storage class,
  per-world counts, and byte estimates
- `notes`: concise scenario assumptions or caveats

## Kernel Memory Analysis Artifact

Schema file: `schema/kernel-memory-analysis.schema.json`

Purpose:

- record one kernel's dense or matrix-free work decomposition in a machine-readable
  way that can be turned into docs tables later
- separate code-path facts from prose conclusions

Key top-level sections:

- `kernel`: name, source location, phase, and mode coverage
- `launch`: launch geometry and sizing symbols
- `memory_layout`: global/shared/register-resident data
- `operations`: streamed inputs, preloaded inputs, recomputed values, and
  dominant multiply/accumulate work
- `observations`: data-backed notes or open questions

## Manifest Contract

The generated manifest is intentionally lightweight. It enumerates:

- required scenarios: `g1_flat`, `h1_tabletop`
- comparison presets: dense loop, dense row, dense streaming, split,
  matrix-free
- planned output files for M3 and M4
- schema version and provenance

Later passes may add fields, but should not rename existing top-level keys
without updating the schema version.
