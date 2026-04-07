# Migrating a Hand-Optimized CUDA Kernel to Warp Tiles: Performance Analysis

**Date:** 2026-04-07
**Authors:** Zach Corse, Claude (Anthropic)
**GPU:** NVIDIA GeForce RTX 5090 (sm_120)
**Warp version:** 1.13.0.dev0

---

## 1. Overview

We are migrating the velocity-mode PGS contact solver kernel from hand-written CUDA to the Warp tile API. The native CUDA baseline is highly optimized (software-pipelined loads, warp shuffle reductions, compile-time constants, direct shared memory access). This provides a rare opportunity to directly compare Warp tile code against a known-optimal native implementation and identify where Warp's abstractions introduce overhead.

**Current state:** ~40% performance regression (68k vs 114k FPS, 168 vs 76 registers) with both phases fully tiled. Phase 1 alone achieves 80 registers with ~20% regression. Adding tiled Phase 2 doubles the register count to 168 — notably, an equivalent native CUDA snippet for Phase 2 uses only 39 registers and adds negligible overhead when fused. The Phase 1 gap persists despite eliminating sync barriers and adding fused tile primitives — indicating possible compute-bound overhead.

---

## 2. The Kernel

One warp (32 threads) per world. Two constraint phases share a velocity vector `s_v[D]` in shared memory (D=133):

- **Dense constraints:** D-wide dot products and velocity updates. 32 threads cooperate across 133 DOFs. This is where tiles naturally fit — `tile_load`, `tile_dot`, `tile_axpy` operate on the full velocity vector.

- **Matrix-free (MF) constraints:** 6-DOF per body. Small scatter-gather operations at runtime offsets into `s_v`. The parallelism structure doesn't align with tiles — all threads compute the same 12-element dot product redundantly, and velocity updates are lane-parallel (threads 0-5 for body A, 6-11 for body B).

Both phases execute sequentially within a Gauss-Seidel iteration (constraint `i` must see constraint `i-1`'s velocity update). The inner loop alternates between cooperative tile operations (dot product, velocity update) and scalar broadcast operations (PGS projection, friction cone clamping).

---

## 3. Performance Summary

| Metric | Native CUDA | Full Warp (tiled) | Phase 1 only (tiled) |
|--------|------------|-------------------|---------------------|
| Env-FPS | 113,575 | 67,942 | ~90,000 |
| Registers/thread | 76 | 168 | 80 |
| Max blocks/SM | 26 | 12 | 25 |
| Kernel duration (8 iters) | ~2.1 ms | ~6.2 ms | ~3.7 ms |

Phase 1 alone (with Phase 2 temporarily native) achieves 80 registers and ~20% regression. Fusing Phase 2 as tile code doubles the register count to 168 and regresses to ~40%. PTX comparison (Phase 1 tiled, Phase 2 native — closest to apples-to-apples for the tile overhead analysis):

| PTX Metric | Native | Warp (Phase 1 tiled) | Ratio |
|------------|--------|---------------------|-------|
| PTX lines | 986 | 2,491 | 2.53x |
| MUL instructions | 54 | 198 | 3.67x |
| ADD instructions | 112 | 346 | 3.09x |
| Barriers | 3 | 19 | 6.33x |

The Phase 1 tile kernel runs **1.76x slower** with near-identical registers (80 vs 76). The full tiled kernel runs **2.95x slower** with 2.2x the registers. The PTX shows 3-4x more ALU operations — primarily from runtime array coordinate computation (see Section 5).

---

## 4. What We Tested and Ruled Out

| Hypothesis | Test | Result | Conclusion |
|-----------|------|--------|------------|
| **Register pressure / occupancy** | Scoped `@wp.func` reduced registers 113→80 | FPS unchanged | Occupancy not the bottleneck at these levels |
| **Tile sync barriers** | Fused tile_dot/tile_axpy reduced barriers 37→19 | FPS unchanged | Barriers not the dominant cost |
| **Tile sync barriers (aggressive)** | Native Phase 1 snippet eliminated most barriers | +2.7% only | Partial improvement confirms barriers are minor |
| **SSA zero-init overhead** | Removed `tile_register_t` default zero-init | Registers unchanged (113→113) | NVRTC already optimizes dead stores |
| **copy_to_register overhead** | Optimized tile_dot/tile_axpy to avoid unnecessary copies | PTX improved, FPS unchanged | NVRTC already optimizes copies in SASS |
| **PTX instruction count** | Reduced PTX via fused ops and copy elimination | FPS unchanged despite fewer PTX instructions | PTX overstates runtime impact; NVRTC backend optimizes significantly |
| **Register tile double-buffering** | Removed software pipelining (eliminated copy pattern) | Registers 120→113, FPS unchanged | Copy of `float data[5]` arrays may not fully optimize; simpler code is equivalent |

**Key finding:** Reducing registers, sync barriers, and PTX instruction counts individually or in combination did NOT improve performance. The remaining gap is compute-bound, not resource-bound.

---

## 5. Leading Hypothesis: Runtime Array Coordinate Computation

### The observation

The PTX shows **3.7x more MUL** and **3.1x more ADD** instructions than native. After ruling out sync barriers, register pressure, and intermediate tile creation as causes, a significant fraction of this ALU overhead must come from array indexing.

### How Warp indexes arrays

Every Warp array access goes through a generic stride-based coordinate path:

```cpp
// Warp: J_world[world, row, d]
byte_offset = offset[0] * strides[0] + offset[1] * strides[1] + offset[2] * strides[2];
value = *(T*)((char*)data + byte_offset);
```

Strides and shapes are runtime values in `wp::array_t`. Each array access requires integer multiplies to compute the byte offset — even when the array dimensions never change for a given kernel configuration.

### How native code indexes arrays

```c
// Native: all dimensions are compile-time literals
int base = world * 128 * 133;  // computed once, with literal constants
float val = J_world.data[base + lane];  // one integer add
```

All array dimensions (D=133, M_D=128, M_MF=512) are baked into the code as literals via string template generation. The compiler folds constant multiplications, strength-reduces where possible, and encodes offsets into addressing modes. **Zero per-access stride computation.**

### Why NVRTC can't optimize this away

Unlike dead stores or unnecessary copies (which NVRTC handles well), stride computations are **semantically required**. The compiler cannot prove the strides are constant because they come from runtime struct fields (`array_t.strides[d]`). Even if every invocation uses identical strides, each access must emit the multiply.

### Estimated impact

With ~500+ array accesses per kernel invocation (tile_load, global array reads, tile operations), each requiring 2-3 stride multiplies, this adds an estimated ~1,000-1,500 extra ALU instructions — consistent with the observed 3-4x MUL/ADD bloat and the ~1.76x runtime overhead.

---

## 6. Additional Architectural Findings

### 6.1 Scoped functions reduce register pressure but not runtime

Wrapping tile operations in `@wp.func` creates scope boundaries that help the register allocator reuse storage across calls (113→80 registers). However, this doesn't reduce instruction count — the same coordinate computations execute, just with fewer registers live simultaneously.

### 6.2 Tile double-buffer pipelining can be counterproductive

The native kernel's software pipeline (`pre_dJ = load; cur_dJ = pre_dJ; pre_dJ = next_load`) relies on the compiler aliasing scalar variables across loop iterations. With Warp's SSA codegen, `J_row = J_next` copies a `float data[5]` array into a new variable rather than aliasing. Whether the compiler fully optimizes this copy for aggregate types (as opposed to scalars, where it trivially does) is uncertain. Removing pipelining was neutral-to-positive.

### 6.3 Fusing both phases in Warp increases register pressure significantly

Warp-generated code for both phases, when fused in one kernel, results in ~168-189 registers. For comparison, an equivalent native CUDA snippet for Phase 2 uses only 39 registers and adds negligible overhead to the fused kernel (80 total registers). The cause of this disparity between Warp-generated and native Phase 2 register usage is not fully understood but is consistent with the general compute overhead pattern observed throughout this analysis.

---

## 7. Proposed Improvements

### 7.1 Compile-time array stride specialization (highest expected impact)

When array shapes/strides are known at compile time, emit specialized indexing code with literal constants instead of runtime stride reads. This addresses the dominant overhead source and benefits **all** Warp GPU kernels.

**Approach:** Propagate compile-time-known array shapes through codegen. When shapes are `wp.constant` or can be inferred, emit `base + i * LITERAL_STRIDE` instead of `base + i * array.strides[d]`.

### 7.2 Preload J/Y into shared memory

Instead of loading J/Y rows from global inside the constraint loop (one `tile_load` per row, each with stride computation), load all active rows into a shared 2D tile upfront. The inner loop then reads from shared memory with simple offset arithmetic.

**Tradeoff:** Shared memory capacity. M_D × D × 4 bytes = ~68KB per tile. Can be managed by loading only active rows or processing in chunks. Pipelining shared tile loads (load next chunk while processing current) is not yet supported in Warp but could be added.

### 7.3 `wp.shared_array()` + `wp.syncthreads()`

Non-tile shared memory with direct indexing. Would enable SIMT Warp code structurally identical to native CUDA — no tile overhead, no coordinate abstraction.

### 7.4 `tile_write_thread(sync=False)`

Skip the sync barrier when the caller knows subsequent operations will sync. Removes ~24 unnecessary barriers per kernel invocation. Minor impact individually but contributes to the instruction count gap.

---

## 8. Optimization History

| # | Change | FPS | vs Native | Registers | Key Insight |
|---|--------|-----|-----------|-----------|-------------|
| 0 | Native CUDA baseline | 113,575 | — | 76 | Compile-time constants, warp shuffles, direct shared memory |
| 1 | Direct tile translation (both phases) | 62,624 | -44.9% | 205 | Both phases as tiles — register explosion |
| 2 | Split kernels | 78,897 | -30.5% | 86+39 | Isolating phases confirmed register coexistence was the issue |
| 3 | Global reads for metadata | 80,340 | -29.3% | 96 | L1 cache competitive with shared tiles for read-only data |
| 4 | Vec3 impulse triplets | 86,453 | -23.9% | 118 | Restructured data to match algorithm's contact-triplet structure |
| 5 | Fused kernel | 88,360 | -22.2% | 120 | func_native tile-by-reference enables shared s_v across phases |
| 6 | Remove pipelining | 90,358 | -20.4% | 113 | Simpler code = fewer tile intermediates |
| 7 | Scoped functions (Phase 1) | ~90,000 | -20.8% | **80** | Register reuse via scope boundaries; FPS unchanged |
| 8 | Fused tile_dot/tile_axpy | ~90,000 | -20.8% | 80 | Barriers halved (37→19); FPS unchanged |
| 9 | Optimized copy_to_register | ~90,000 | -20.8% | 80 | NVRTC already handles this in SASS |
| 10 | **Full tiled (Phase 2 tile_gather)** | **67,942** | **-40.2%** | **168** | Phase 2 tile code doubles registers when fused with Phase 1 |

Steps 7-9 reduced Phase 1 registers to near-native (80 vs 76) without improving FPS — indicating the Phase 1 gap is compute-bound (runtime coordinate overhead). Step 10 shows that extending tiles to Phase 2 increases the fused kernel to 168 registers (vs 80 with native Phase 2, which itself uses only 39 registers). Closing the full gap requires addressing both the per-access compute overhead and the register cost of Warp-generated code for both phases.
