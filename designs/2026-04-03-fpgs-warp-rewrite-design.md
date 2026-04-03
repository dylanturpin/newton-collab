# FPGS Velocity-Mode Pure Warp Rewrite — Design Spec (First Pass)

**Date:** 2026-04-03
**Branch:** `zcorse/fpgs-tile-conversion`
**Status:** First pass — expect iteration on performance

## Goal

Replace the two native CUDA string-template kernels in the velocity-mode solve
pipeline with pure Warp code. Primary value is maintainability; performance
target is functional correctness with no more than ~10% regression on the first
pass.

## Scope

**In scope:**
- `_build_pack_mf_meta_kernel` — eliminated, replaced by struct array
- `_build_pgs_solve_mf_gs_kernel` — rewritten in pure Warp

**Out of scope:**
- Non-velocity-mode native kernels (delassus, streaming PGS, tiled row PGS,
  tiled contact PGS, standalone MF PGS)
- All kernels in `kernels.py` (already pure Warp)
- Tiled Cholesky, triangular solve, hinv_jt kernels (already pure Warp)

## Benchmarking

Before/after comparison using the established benchmark:

```bash
uv run python -m newton.tools.solver_benchmark \
    --scenario h1_tabletop --benchmark --num-worlds 4096 \
    --solver fpgs_mf --summary-timer
```

Current baseline: ~98,920 FPS (CUDA graphs, 8 substeps).

Visual verification:

```bash
uv run python -m newton.tools.solver_benchmark \
    --scenario h1_tabletop --num-worlds 1 --solver fpgs_mf
```

## Design

### 1. MF Metadata Struct (eliminates pack kernel)

Define a 16-byte Warp struct replacing the `int4` bit-packing scheme:

```python
@wp.struct
class MfConstraintMeta:
    dof_a: wp.int16      # 2B — DOF offset for body_a into velocity vector
    dof_b: wp.int16      # 2B — DOF offset for body_b
    diag: wp.float32     # 4B — effective mass inverse (1/C_ii)
    rhs: wp.float32      # 4B — constraint RHS bias
    row_type: wp.int16   # 2B — 0=normal, 2=friction, 3=joint limit
    parent: wp.int16     # 2B — parent constraint index (for friction rows)
```

**Total: 16 bytes** — fits in a single cache line access from global memory.

**Changes to upstream kernels:** `build_mf_contact_rows` and
`compute_mf_effective_mass_and_rhs` (in `kernels.py`) will write directly to a
`wp.array(dtype=MfConstraintMeta)` instead of 6 separate arrays. The separate
`_build_pack_mf_meta_kernel` and `mf_meta_packed` int array are removed.

**Rationale:** The current approach loads metadata from 6 separate arrays (up to
6 cache lines) and packs them into an `int4` via a separate kernel with manual
bit-twiddling. A 16-byte struct achieves the same cache locality (one cache line)
without the pack kernel or the unpack ALU. Even if the compiler emits multiple
load instructions for the struct, they all hit the same cache line.

### 2. PGS Kernel: Overall Structure

```
wp.launch_tiled(kernel, dim=[world_count], block_dim=32)
```

One tile (one warp, 32 threads) per world. `world, thread = wp.tid()`.

Kernel phases:
1. **Load phase** — cooperative tile loads from global to shared
2. **Solve phase** — sequential GS loop with Phase 1 (dense) + Phase 2 (MF)
3. **Store phase** — cooperative tile stores from shared to global

### 3. Shared Tiles (persistent across all iterations)

| Tile | Shape | Dtype | Access pattern |
|------|-------|-------|----------------|
| `s_v` | `(D,)` | float32 | Read/write every constraint (both phases) |
| `s_lam_dense` | `(M_D,)` | float32 | Random read/write (projection, sibling) |
| `s_lam_mf` | `(M_MF,)` | float32 | Random read/write (projection, sibling) |
| `s_rhs_dense` | `(M_D,)` | float32 | Read-only by constraint index |
| `s_diag_dense` | `(M_D,)` | float32 | Read-only by constraint index |
| `s_rtype_dense` | `(M_D,)` | int16 | Read-only by constraint index |
| `s_parent_dense` | `(M_D,)` | int16 | Read-only by constraint index |
| `s_mu_dense` | `(M_D,)` | float32 | Read-only by constraint index |

**Load phase:** Eight `wp.tile_load()` calls into shared tiles.

### 4. Phase 1: Dense Constraints

Sequential GS loop over `m_dense` constraints per PGS iteration.

#### 4a. J/Y register-tile pipelining

Uses the double-buffer pattern from Warp tiles.rst. J and Y rows (D=133 floats
each) are loaded as register tiles one iteration ahead:

```python
# Prefetch first constraint
J_next = wp.tile_load(J_world[world], shape=(D,), offset=(0, 0), storage="register")
Y_next = wp.tile_load(Y_world[world], shape=(D,), offset=(0, 0), storage="register")

for i in range(m_dense):
    J_row = J_next
    Y_row = Y_next
    if i + 1 < m_dense:
        J_next = wp.tile_load(J_world[world], shape=(D,), offset=(i+1, 0), storage="register")
        Y_next = wp.tile_load(Y_world[world], shape=(D,), offset=(i+1, 0), storage="register")
    # ... process constraint i ...
```

GPU instruction scheduler overlaps the i+1 loads with constraint i's compute.

#### 4b. Dot product

```python
jv_tile = wp.tile_sum(wp.tile_map(wp.mul, J_row, s_v))
```

`J_row` is register, `s_v` is shared — mixed-storage `tile_map` should work.
`tile_sum` returns a single-element tile; all threads hold the result.

#### 4c. PGS projection

Scalar Warp code executed redundantly by all 32 threads (same as current — the
scalar ALU is "free" since all threads compute the identical result).

- **Reads:** `wp.tile_extract()` on `s_lam_dense`, `s_rhs_dense`, `s_diag_dense`,
  `s_rtype_dense`, `s_parent_dense`, `s_mu_dense` at constraint index `i` or
  data-dependent indices (`parent_idx`, `sib`).
- **Writes:** `wp.tile_write_thread()` on `s_lam_dense` at index `i` and `sib`.
  All threads agree on the value; one thread writes
  (`has_value = (thread == 0)`).

Friction cone clamping: identical logic to current code — branches, sibling
lookups, sqrt, scaling. All within scalar Warp.

#### 4d. Velocity update

```python
s_v += Y_row * delta_impulse
```

Warp tile arithmetic: register tile scaled by scalar, added to shared tile.

#### 4e. Sibling velocity update (friction rare-path)

When friction cone clamping scales the sibling, load sibling's Y row from global
(not pipelined — random index) and update `s_v`:

```python
Y_sib = wp.tile_load(Y_world[world], shape=(D,), offset=(sib, 0), storage="register")
s_v += Y_sib * sib_delta
```

### 5. Phase 2: Matrix-Free Constraints

Sequential GS loop over `m_mf` constraints per PGS iteration. Same kernel,
continues after Phase 1. Uses `s_v` and `s_lam_mf` shared tiles.

Thread assignment: threads 0-5 handle body_a (6 DOFs), threads 6-11 handle
body_b (6 DOFs), threads 12-31 idle (contribute zeros).

#### 5a. Metadata load

Scalar struct load from `wp.array(dtype=MfConstraintMeta)`. All threads load
the same constraint's struct — broadcast read, one cache line.

#### 5b. J/MiJt load

Scalar per-thread loads from global arrays:
- Threads 0-5: load `mf_J_a[world, i, thread]` and `mf_MiJt_a[world, i, thread]`
- Threads 6-11: load `mf_J_b[world, i, thread-6]` and `mf_MiJt_b[world, i, thread-6]`
- Threads 12-31: no load

Software pipelining: same consume/prefetch pattern as Phase 1, but with scalar
variables rather than register tiles (data is too small for tiles — 6 floats per
body).

#### 5c. Dot product

Each thread computes one multiply (J element * s_v element) or zero:

```python
my_val = 0.0
if thread < 6 and dof_a >= 0:
    my_val = cur_Ja * wp.tile_extract(s_v, dof_a + thread)
elif thread >= 6 and thread < 12 and dof_b >= 0:
    my_val = cur_Jb * wp.tile_extract(s_v, dof_b + thread - 6)

product_tile = wp.tile_zeros(shape=(32,), dtype=float, storage="shared")
wp.tile_write_thread(product_tile, thread, my_val, True)
jv_tile = wp.tile_sum(product_tile)
```

#### 5d. PGS projection

Same pattern as Phase 1: scalar Warp code, all threads redundant.
`wp.tile_extract` for `s_lam_mf` reads, `wp.tile_write_thread` for writes.

#### 5e. Velocity update

```python
idx = -1
val = 0.0
if thread < 6 and dof_a >= 0:
    idx = dof_a + thread
    val = cur_MiJta * delta_impulse
elif thread >= 6 and thread < 12 and dof_b >= 0:
    idx = dof_b + thread - 6
    val = cur_MiJtb * delta_impulse

has_value = (idx >= 0) and (delta_impulse != 0.0)
wp.tile_scatter_add(s_v, idx, val, has_value)
```

**Future optimization:** Where scatter-add indices are known non-overlapping
(body_a and body_b have distinct DOF ranges), `tile_thread_write` with manual
read-add-write could replace `tile_scatter_add` to avoid shared memory atomics.

#### 5f. Sibling velocity update (friction rare-path)

Same `tile_scatter_add` pattern as 5e, using sibling's DOF offsets and MiJt
values loaded from global.

### 6. Store Phase

Three `wp.tile_store()` calls:
- `s_v` → `v_out`
- `s_lam_dense` → `world_impulses`
- `s_lam_mf` → `mf_impulses`

### 7. Upstream Changes

Kernels in `kernels.py` that currently write to separate MF metadata arrays
(`mf_dof_a`, `mf_dof_b`, `mf_eff_mass_inv`, `mf_rhs`, `mf_row_type`,
`mf_row_parent`) will be modified to write to a single
`wp.array(dtype=MfConstraintMeta)`. Affected kernels:
- `build_mf_contact_rows`
- `compute_mf_effective_mass_and_rhs`

The separate metadata arrays and the pack kernel launch in the solver pipeline
are removed.

### 8. What This Does NOT Change

- Kernel launch pattern (one `wp.launch_tiled` per PGS solve, same as today)
- Block size (32 threads = one warp per world)
- Shared memory layout semantics (same tiles, just expressed via Warp API)
- GS iteration structure (sequential constraints, same ordering)
- Friction cone projection logic (identical math)
- CUDA graph compatibility (pure Warp kernels are graph-capturable)

### 9. Risk and Iteration Plan

This is a first pass. Known performance risks:

1. **Phase 2 dot product via shared memory** — `tile_write_thread` + `tile_sum`
   goes through shared memory instead of warp shuffles (~35 cycles). May add
   latency per MF constraint.
2. **`tile_extract` overhead** — scalar reads from shared tiles for projection
   logic. Should be similar to raw shared memory indexing but needs verification.
3. **`tile_scatter_add` atomics** — unnecessary for non-overlapping writes.
   Can switch to `tile_thread_write` with read-add-write if profiling shows cost.
4. **Register-tile pipelining effectiveness** — Warp compiler may or may not
   schedule register-tile loads to overlap with compute as effectively as the
   hand-written CUDA pipeline. Needs nsys profiling.

**Iteration approach:** benchmark after each change, profile with nsys if
regression >5%, address hotspots one at a time.

## Key Files

| File | Changes |
|------|---------|
| `newton/_src/solvers/feather_pgs/solver_feather_pgs.py` | Remove `_build_pack_mf_meta_kernel`, `_build_pgs_solve_mf_gs_kernel`; add pure Warp PGS kernel; update pipeline launch code |
| `newton/_src/solvers/feather_pgs/kernels.py` | Add `MfConstraintMeta` struct; modify `build_mf_contact_rows`, `compute_mf_effective_mass_and_rhs` to write struct array |
| `newton/tools/solver_benchmark.py` | No changes expected |
