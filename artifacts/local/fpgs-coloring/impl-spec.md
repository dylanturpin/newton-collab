# FeatherPGS Graph-Colored Parallel Gauss-Seidel — Implementation Spec

Target file: `newton/_src/solvers/feather_pgs/solver_feather_pgs.py`
New module: `newton/_src/solvers/feather_pgs/mf_coloring.py`
Scope: add per-step dynamic contact coloring + a per-color **multi-warp-per-world** parallel MF Gauss-Seidel sweep, preserving the matrix-free row-PGS update byte-for-byte. Benchmark on cube probes p10/p11 vs PhysX and a vel-OFF FPGS baseline.

All line anchors below were read against the working tree on branch `bk-slurm-broker-green` and are accurate to within a few lines (the kernel is one large f-string; the snippet f-string opens at line 7387).

---

## 1. Chosen approach + why (with grafted ideas)

**Winner: `gpu-recolor`** (avg score 4.67) over `host-recolor` (3.67).

The decisive difference is the recolor cost model. `host-recolor` was measured at ~160 ms/step at 256 worlds and extrapolated to ~2.5 s/step at 4096 worlds on the every-step path, and ~25 ms/step even at an optimistic 1% recolor rate — on the same order as or larger than the GS solve it is trying to accelerate. Newton's `color_graph` (`newton/_src/sim/graph_coloring.py:243`) is CPU-only, single-threaded over a `std::vector` adjacency, with super-linear growth (256-env ~1.6 ms, 1024-env ~24 ms, 4096-env ~589 ms end-to-end) and hard-rejects GPU arrays — it is built for one-time `ModelBuilder.color()` setup, not per-step at scale. The host path is therefore retained **only as a bring-up / cross-check oracle**, not the shipping engine.

`gpu-recolor` colors the contact-triple conflict graph entirely on-device, one warp(s)-per-world in parallel across worlds, with **no host round-trip and no D2H/H2D of graph data**. Recolor is skipped when the contact set is unchanged, gated by the persistence signal the collision pipeline already produces.

**Grafted ideas from `host-recolor` and the gate analysis:**

1. **CSR color-index indirection (from host-recolor's `kernel_changes`)** — the GS loop indirects through a `color_rows` permutation rather than physically reordering `mf_J_a/b`, `mf_MiJt_a/b`, `mf_meta`, `mf_impulses`. This keeps every row's data byte-identical (preserving the bit-validated row-PGS update and the load-once/store-once warm-start contract) and keeps the pack kernel (`_pack_mf_meta`, :1888) untouched.
2. **Persistence-gated amortized recolor (from host-recolor's `recolor_strategy`)** — the recolor trigger is the count of new/broken contacts; when zero, the cached color CSR is reused verbatim. We implement the *check itself* on-device (a tiny reduce over `match_index`) so there is no per-step host sync, but the *amortization logic* is host-from `host-recolor`.
3. **rt4 stays serial-tail (consensus across both designs and the rt4-Amdahl gate)** — see §6.
4. **Triple atomicity (both designs)** — the `{normal, t1, t2}` triple is one coloring node, its 3 rows emitted consecutively in `color_rows` and run in intra-triple order, because the friction block (:7350-7385) reaches across to `s_lam_mf[mf_par]` and the sibling.
5. **Bring-up bridge (grafted from host-recolor's prototype recommendation)** — Phase 0 of the rollout (below) ships the *colored kernel* driven by a **host-computed** CSR (using `color_graph`) so the kernel and fidelity are validated before the on-device coloring kernel exists. This de-risks the two hard pieces (colored kernel correctness; multi-warp occupancy) independently of the GPU coloring kernel.

**Why this closes the gap (ceiling, from the gates).** The break-even model net ≈ η·(m/c)/τ with η≈0.65, m/c≈31.9, gives ~15.1x within-world at τ_mm=1.375 (the conservative penetration-quality convergence penalty) / ~16.6x at τ=1.25. The fidelity side is non-binding: colored order holds peak interpenetration at 0.39–0.87 mm, an order of magnitude under the 8 mm gate (mm-tau gate). The real risk is occupancy contention at high env (§7), addressed explicitly — not deferred.

---

## 2. Exact edits to `solver_feather_pgs.py`

### 2.1 The serial GS loop → per-color loop (snippet body, :7579-7669)

Replace the single contiguous `for (int i = 0; i < m_mf; i++)` (:7579) with a two-level loop driven by the CSR color index. The **per-row update body is copied verbatim** (unpack :7604-7609, J·v shfl reduce :7629-7634, residual/delta/project per rt :7636-7656, `s_lam_mf[i]=new_impulse` :7657, `s_v += MiJt*delta` :7660-7667). Only the *visiting order* changes and `__syncwarp()` moves from per-row (:7668) to a per-color barrier.

New structure (replaces :7579-7669), written for the **multi-warp-per-world** launch of §7 (W warps, `block_dim=32*W`; `warp_id = threadIdx.x / 32`, `lane = threadIdx.x % 32`; n_warps = W):

```c
// off_co = world*(MAX_COLORS+1); off_cr = world*M_MF  (computed near :7400)
int nc = mf_n_colors.data[world];
for (int c = 0; c < nc; c++) {{
    int cs = mf_color_offsets.data[off_co + c];
    int ce = mf_color_offsets.data[off_co + c + 1];
    // Distribute the rows of this color across the W warps (round-robin).
    for (int k = cs + warp_id; k < ce; k += n_warps) {{
        int i = mf_color_rows.data[off_cr + k];     // permuted row index
        // ── VERBATIM per-row body from :7604-7667, every mf_* read indexed by i ──
        //    (the i+1 software prefetch at :7587-7599 is replaced by a prefetch
        //     of mf_color_rows[k + n_warps]; prefetch is a perf detail, not fidelity)
    }}
    __syncthreads();   // was __syncwarp() at :7668; cross-warp barrier between colors
}}
```

Key correctness facts:
- **No write hazard within a color**: rows in one color share no body dof-base (conflict-graph definition), so their `s_v[dof_a/dof_b]` writes are disjoint across warps. The carrier of the GS dependency is `s_v[]` (and `s_lam_mf` only for the friction triple, which is one node — never split across warps).
- **Triple ordering preserved**: a triple node emits its 3 rows (`mf_par`, `mf_par+1`, `mf_par+2`) **consecutively into `color_rows` within one color**, and they are assigned to the **same warp** (see §4 layout) so they run in intra-triple order on one warp. The friction block's reach to `s_lam_mf[mf_par]` and the sibling (:7360-7382) is safe because no other node in the color touches those bodies.
- **rt2 friction-gate** (:7613-7617, `s_lam_mf[i]=0` then `continue`) and **diag<=0 skip** (:7619) are copied verbatim inside the per-row body.
- **row_phase gating** (:7601, :7611-7612) is copied verbatim; under the colored path the default schedule uses `row_phase==0` (:1977).

### 2.2 Row-PGS update body — UNCHANGED

No change to :7636-7667 math or to the friction block (:7350-7385). The projection dispatch (rt0 clamp ≥0 :7642-7643; rt4 one-sided :7644-7651; rt2 cone :7652-7653) is byte-identical. This is the bit-validated solver and must stay so per the project constraint.

### 2.3 Lambda storage / warm-start — UNCHANGED contract

`s_lam_mf` is still loaded once from `mf_impulses` (:7437) and stored once to `mf_impulses` (:7766), strided by `lane` over `m_mf` (these LOAD/STORE loops stay, but must now stride over `threadIdx.x` and `blockDim.x` instead of `lane`/32 — see §7). The CSR permutes only the *visiting order inside the solve*, never the storage index, so warm-start lambda stays aligned to its row slot and the bit-identity of stored lambda per-slot is preserved.

### 2.4 Snippet signature + launch inputs

- **Snippet signature** (native func `pgs_solve_mf_gs_native` :7772-7807 and template :7809-7843): add three inputs after `mf_row_mu`:
  - `mf_color_offsets: wp.array(dtype=int)` (flattened `worlds*(MAX_COLORS+1)`)
  - `mf_color_rows: wp.array(dtype=int)` (flattened `worlds*M_MF`)
  - `mf_n_colors: wp.array(dtype=int)` (`worlds`)
  Compute `off_co = world*({MAX_COLORS}+1)` and `off_cr = world*{M_MF}` next to the existing offsets at :7400-7403. `MAX_COLORS` becomes a template constant baked into the f-string (like `M_MF`/`D`).
- **`_launch_matrix_free_gs_solve`** (:1909): add `self.mf_color_offsets, self.mf_color_rows, self.mf_n_colors` to the `inputs=[...]` list at :1948-1954 (right after `self.mf_row_mu` at :1954), and change `block_dim=32` (:1963) to `block_dim=32*self.mf_color_warps` (see §7).
- **Build site** (:1803-1809): pass `max_colors=self.mf_max_colors` and `warps_per_world=self.mf_color_warps` into `_get_pgs_solve_mf_gs_kernel` so the f-string and `wp.tid()` shape match. The kernel name (:7878) must include these so the kernel cache keys on them.
- **`pgs_solve_mf_gs_template`** (:7844) currently does `world, _lane = wp.tid()`. With `launch_tiled` and a wider tile this still gives `world` from the block index; the in-snippet `threadIdx.x` (:7390) already drives lane logic, so only the snippet's lane/warp decomposition (§7) changes, not `wp.tid()`.

### 2.5 New solver state + per-step driver

- **Allocate** the three color buffers next to the MF buffers (~:1592), see §4.
- **Add `_maybe_recolor_mf()`** near `_pack_mf_meta` (:1888) / `_launch_matrix_free_gs_solve` (:1909). Call it once per step **after** `_pack_mf_meta(self.mf_rhs)` (:2412) and **before** the GS launch (:2525). It runs the on-device dirty-check + coloring kernels (§3).
- **Replace `self.mf_impulses.zero_()` (:3448)** — see §5.

---

## 3. Coloring module

### 3.1 Where it lives
New file `newton/_src/solvers/feather_pgs/mf_coloring.py` (Warp + NumPy only; no new deps per AGENTS.md). It holds:
- `@wp.kernel` `mf_recolor_dirty` — sets `mf_need_recolor[world]=1`.
- `@wp.kernel` `mf_color_assign` — the on-device greedy coloring kernel (one or more warps per world).
- `@wp.kernel` `mf_build_color_csr` — turns per-node colors into the CSR (`mf_color_offsets`, `mf_color_rows`).
- host helper `host_color_oracle(...)` — Phase-0 bring-up path wrapping `newton._src.sim.graph_coloring.color_graph` (:243) for validation only.

### 3.2 Conflict-graph construction (device)
A **node** = one MF coloring unit:
- a contact triple `{normal, t1, t2}` keyed by `mf_row_parent` (`-1` default at :1590; the normal row's parent points to itself / the triple base — group rows by shared parent),
- a standalone rt0 normal with no friction (rare), and
- rt4 rows are **excluded** (§6).

Two nodes conflict iff they share a body dof-base. The dof-bases live in `mf_meta_packed.x` (`(dof_a<<16)|(dof_b&0xFFFF)`, packed at :6502-6507) and equivalently in `self.mf_dof_a`/`self.mf_dof_b` (:1622-1623) / `self.mf_body_a`/`self.mf_body_b` (:1544-1545). The coloring kernel reads `mf_dof_a/mf_dof_b` per node (the triple's normal row carries the pair) and colors greedily: process nodes in slot order; assign the lowest color not used by an already-colored neighbor. Neighbor lookup is by body dof-base, done with a small per-warp scratch bitset over the ≤ max-degree (~10) neighbors. m/c ≈ 31.9 with ~10 colors (census), so `MAX_COLORS=16` is a safe template bound (assert + fallback-to-serial if exceeded).

Balance is **not** required on device: the census imbalance is ~1.1–1.37 after greedy, well within the η≈0.65 efficiency assumption. Skip `balance_colors` (it was the second-most expensive host stage and buys little here).

### 3.3 Recolor / amortization strategy (device, persistence-gated)
- Collision runs with `contact_matching="sticky"`, `contact_report=True` (`collide.py:618`, :715-718), populating `Contacts.rigid_contact_match_index` (`contact_match.py:245`, values `MATCH_NOT_FOUND`/`MATCH_BROKEN` < 0 for new/broken) and `rigid_contact_new_count`/`rigid_contact_broken_count` (`contact_match.py:231`).
- `mf_recolor_dirty` kernel: per world, OR-reduce `(new_count>0) || (broken_count>0) || (triple_count != prev_triple_count)` → `mf_need_recolor[world]`. This is a tiny device reduce, **no host sync**.
- `mf_color_assign`: launched every step but **early-returns** for worlds with `mf_need_recolor==0`, reusing the cached `mf_color_offsets/mf_color_rows/mf_n_colors` from the prior step verbatim (the conflict graph is a pure function of live body dof-base pairs, invariant when the contact set is unchanged). cube probes settle to a near-static contact set, so steady-state recolor rate is low and the coloring kernel cost is paid rarely. The every-step path is still correct and bounded if matching is off (it just recolors every step).
- Keep `mf_prev_triple_count[world]` updated at the end of `mf_color_assign`.

**Phase-0 bring-up oracle (host):** `host_color_oracle` pulls `mf_meta`/`mf_dof_a`/`mf_dof_b`/`mf_row_parent` for a few worlds to host, calls `color_graph` per world, packs the same CSR, and uploads it. Used only to (a) drive the colored kernel before `mf_color_assign` exists and (b) cross-check the device coloring is a valid coloring (no monochromatic conflict edge). Not on the shipping path.

---

## 4. New data structures / buffers and layout

Allocated near :1592 (worlds = `self.world_count`, `M_MF = mf_max_c`, `MAX_COLORS = self.mf_max_colors`, default 16):

| buffer | dtype | shape | meaning |
|---|---|---|---|
| `mf_color_offsets` | int32 | `(worlds, MAX_COLORS+1)` | CSR per-color start offsets (prefix sums of color sizes); the kernel reads `off_co = world*(MAX_COLORS+1)` |
| `mf_color_rows` | int32 | `(worlds, M_MF)` | row indices grouped by color; the 3 rows of each triple stored consecutively (normal, t1, t2) and assigned to one warp |
| `mf_n_colors` | int32 | `(worlds,)` | valid color count per world (kernel loop bound) |
| `mf_need_recolor` | int32 | `(worlds,)` | dirty flag from `mf_recolor_dirty` |
| `mf_prev_triple_count` | int32 | `(worlds,)` | last step's node count, for the changed-count test |

Layout rule binding §2.1 and §3: within `mf_color_rows`, color `c` occupies `[mf_color_offsets[c], mf_color_offsets[c+1])`. The round-robin warp assignment (`k = cs + warp_id; k += n_warps`) must keep a triple's 3 consecutive entries on **one** warp — so the CSR builder emits a triple as a single "super-entry" expanded to its 3 rows only after the warp stride is computed, i.e. assign warps at *node* granularity and write the 3 rows contiguously for the assigned warp. Simplest correct implementation: build `mf_color_rows` so that node-to-warp is `node_index_within_color % n_warps`, and the inner loop iterates nodes (each node materializing 1 or 3 rows in order). This avoids any chance of a triple being split across warps.

These add ≈ `worlds*(MAX_COLORS+1 + M_MF + 3)` int32 ≈ at 4096 worlds, M_MF=4096: ~67M int32 ≈ 270 MB for `mf_color_rows`. That is the dominant cost; if M_MF is large relative to the live contact count, size `mf_color_rows` to the live max instead of M_MF (it only needs to index live rows). Shared memory (`s_v[D]`, `s_lam_mf[M_MF]`, :7408/7419) is unchanged — it is per-block and shared across the W warps, not multiplied.

---

## 5. Contact persistence + warm-start wiring (replacing `mf_impulses.zero_()`)

Today `self.mf_impulses.zero_()` runs every step at **:3448** (comment: "PGS reads before first write"). Zeroing throws away the converged impulses from the prior step — fine for a cold solve, wasteful given contact persistence.

Change: gate the zero on persistence.
- When `contact_matching` is active and a contact slot's `rigid_contact_match_index >= 0` (matched to a prior contact), **carry** the prior `mf_impulses` for the rows of that contact triple instead of zeroing (warm start).
- When `match_index < 0` (`MATCH_NOT_FOUND`/`MATCH_BROKEN`, new/broken), zero those rows.
- Implement as a small kernel `mf_warmstart_impulses` that, per MF row, looks up the contact's `match_index` and either keeps `mf_impulses[i]` (matched) or sets 0 (new). Because the CSR permutes only visiting order and storage stays slot-keyed (§2.3), the lambda already lives at the right slot; the matcher's sort permutes `match_index` with contacts (`contact_match.py:89`), so the warm-start kernel must read the **post-sort** `match_index`.
- The kernel's load-once (:7437) / store-once (:7766) contract is unchanged; we only change what `mf_impulses` holds at entry.

If `contact_matching == "disabled"`, fall back to the current `mf_impulses.zero_()` exactly (no behavior change). This makes warm-start opt-in and keeps the non-matching path bit-identical to today.

---

## 6. rt4 (velocity-limit) handling

rt4 rows are **excluded from coloring and left in the existing serial MF tail loop** (:7709-7753, run for `row_phase 0/2/5` after the main per-iter loop). Justification, all verified:
- They are single-body, single-DOF (selector Jacobian), gated out of the main loop at :7612 (`if (row_phase==0 && mf_rt==4) continue`).
- They are 55.6% of all MF rows (rt4-Amdahl gate) but never bind in the cube probes (limit ~1000 m/s) and are dropped under the vel-OFF benchmark baseline.
- The design does **not** depend on dropping them: structurally they remain a serial tail, so correctness holds whether vel-limits are active or not.
- **Amdahl caveat (from the rt4-Amdahl gate):** if rt4 stays serial *and* vel-limits are ON, full-step net is capped at ~1.80x regardless of contact speedup. Under the vel-OFF baseline the cap vanishes (full net = contact net ~16.6x). The benchmark uses the vel-OFF baseline, so this is acceptable; a follow-on can color rt4 as trivial single-body nodes if vel-ON 4096-env perf ever becomes the target. Document this cap in the benchmark writeup so the headline number is not misread.

---

## 7. Multi-warp-per-world occupancy plan for 4096 envs (and the risk)

**This is the central risk and is addressed head-on, not deferred** (the killer that sank `host-recolor`).

Today: `launch_tiled(dim=[world_count], block_dim=32)` (:1929-1965) → one block = one warp = one world. Reordering rows on a single warp yields **zero** speedup; realizing m/c≈32 requires **W warps per world**.

Plan:
- Parameterize `W = self.mf_color_warps` (template constant, default **4**, sweepable 2/4/8). `block_dim = 32*W`, `dim=[world_count]` unchanged (one block per world). Each block has W warps cooperating on the per-color row set via the round-robin in §2.1.
- **Decompose threads** in the snippet (:7390): `warp_id = threadIdx.x >> 5; lane = threadIdx.x & 31`. The J·v `__shfl_*_sync` reduction (:7629-7634) stays **intra-warp** (each warp processes its own rows over lanes 0-11), so the reduction is unchanged per row — only which warp owns which row changes. LOAD/STORE strided loops (:7424-7441, :7759-7766) change stride from `32` to `blockDim.x` and index from `lane` to `threadIdx.x`.
- **Barriers**: per-color barrier becomes `__syncthreads()` (cross-warp) instead of `__syncwarp()` (:7668). The dense phase and rt4 tail (which assume a single warp) must either (a) run on `warp_id==0` only with a `__syncthreads()` after, or (b) be left strictly single-warp by guarding `if (warp_id==0)`. Choose (a)/(b) for the dense+rt4 sections to avoid rewriting them; only the MF color loop is multi-warp.
- **`__launch_bounds__`**: there is currently **no `__launch_bounds__` anywhere in the file** (verified). With `block_dim=32*W` and no bound, NVCC may cut occupancy via register pressure. Add `__launch_bounds__(32*W)` (and tune `minBlocksPerSM`) to the snippet's kernel so register allocation targets the chosen block size. This is mandatory — without it, raising threads/block can silently halve occupancy.

**The risk, quantified:** at 4096 envs the across-world launch already saturates SM occupancy. Adding W warps/world means W blocks' worth of warps compete for the exact SMs that are full, so the theoretical ~16x within-world width does **not** convert 1:1 to wall-clock. The break-even net η≈0.65 already discounts for this; the open question is whether 4096-env occupancy leaves enough room for even W=2–4. **Mitigation / measurement:** (1) `__launch_bounds__` + an occupancy sweep over W∈{1,2,4,8} via `ncu --set occupancy` on the bitcheck harness kernel; (2) pick W at the knee where added warps stop improving wall-clock; (3) the W=1 build is exactly today's kernel (with reorder overhead only), giving a clean floor to detect regressions. If no W>1 beats W=1 at 4096, the honest conclusion is the gap cannot be closed by within-world coloring at 4096 and the program pivots to a different parallelization axis — this spec makes that measurable rather than assumed.

---

## 8. Local validation plan (`/tmp/cap892.npz` + `/tmp/bitcheck_cap.py`)

Capture meta (`/tmp/cap892.json`): `dense_max_constraints=512`, `mf_max_constraints=4096`, `max_world_dofs=609`, `friction_mode="current"`, `world_count=256`, `iterations=8`, `omega=1.0`, `row_phase=0`. The npz keys (verified) are the kernel inputs; note `mf_meta` is present (carries dof bases in slot .x) and `row_parent`/`mf_*` are present, so the **conflict graph is reconstructable from the capture alone** (no extra capture needed).

**What MUST match (bit-identical):**
- **Single-color = serial equivalence.** Build a degenerate CSR with `n_colors=1` (all rows in one color, in natural slot order, warp count W=1). The colored kernel must then reproduce the serial kernel's `v_out`/`mf_impulses` **byte-for-byte**. This isolates the loop-rewrite from the coloring. Run via a modified `bitcheck_cap.py` that appends the three color arrays to `inputs` and compares against the unmodified-kernel output saved by the current `/tmp/bitcheck_cap.py`. Assert `np.array_equal`.
- **Per-row update identity.** Within any single color, the row body is the verbatim serial body, so each row's `delta_impulse`/`s_v` write is identical to the serial run for that row given the same input `s_v`. Validate by a row-trace dump (instrument one world, compare per-row `new_impulse`).
- **Warm-start non-matching path.** With `contact_matching="disabled"`, `mf_impulses` entry is still zeroed → output identical to today.

**What is ALLOWED to differ (and must be checked instead against fidelity):**
- **Multi-color / W>1 converged solution.** Graph-colored GS is a *different valid GS variant*; the full converged `v_out`/`mf_impulses` are **not** bit-identical to the serial sweep (GS is order-sensitive). The mm-tau gate establishes this is fine: colored order needs ~2 extra iterations (τ≈1.25–1.375) to match serial@K=8 quality, and holds peak interpenetration at 0.39–0.87 mm (gate ≤8 mm). Validate by replaying the colored sweep in the `/tmp/tau_test.py` numpy harness (it already does the color-sorted vs natural serial-GS comparison with the identical per-rt projection) and confirming `peak_pen_mm ≤ 8` and `tau_mm ≤ 1.5` across worlds.
- **Coloring validity.** Independently assert the device coloring is a proper coloring: for every color, no two nodes in it share a body dof-base (reconstruct adjacency from `mf_meta.x` + `row_parent` in numpy, check no monochromatic edge). Cross-check the device coloring against `host_color_oracle` (`color_graph`) — they need not produce the *same* colors, only both valid.
- **Triple integrity.** Assert each triple's 3 rows are consecutive in `mf_color_rows`, in (normal,t1,t2) order, and assigned to one warp.

**Bit-identity protocol note (open question from recon, resolved):** define bit-identity **per-row-update**, not for the full solve. The reference for the full-solve fidelity check is the **colored** sweep's penetration metric, not serial bit-identity. Update `/tmp/bitcheck_cap.py` to take a CSR and an "expect_identical" flag (true only for the n_colors=1/W=1 case).

Commands (run locally on the Blackwell box):
```bash
# serial reference (existing harness)
NWORLDS=256 python3 /tmp/bitcheck_cap.py /tmp/serial_ref.npz
# colored n_colors=1, W=1 — MUST equal serial_ref
NWORLDS=256 MF_COLOR_CSR=trivial MF_COLOR_WARPS=1 python3 /tmp/bitcheck_cap.py /tmp/colored_trivial.npz
python3 -c "import numpy as np; a=np.load('/tmp/serial_ref.npz'); b=np.load('/tmp/colored_trivial.npz'); \
  assert all(np.array_equal(a[k],b[k]) for k in a.files), 'NOT bit-identical'; print('bit-identical OK')"
# real coloring + tau/penetration fidelity (numpy replay)
python3 /tmp/tau_test.py   # confirm peak_pen_mm<=8, tau_mm<=1.5 across worlds
```

---

## 9. Benchmark plan

### 9.1 Configs (matrix)
- Probes: **p10** and **p11** cube-rain (`p10_cube_rain.py`, `SimulationCfg(dt=0.005)` at :167; p11 analogous).
- Env counts: **256 / 1024 / 4096**.
- Arms (3): **(A) colored FPGS** (this spec, vel-OFF), **(B) FPGS vel-limits-OFF serial baseline** (today's kernel, vel-OFF), **(C) PhysX**.
- Solver settings held fixed across A/B: `iterations=8` (or A at iterations=10 to hit τ-matched quality — report both), `omega=1.0`, `friction_mode="current"`, `contact_matching="sticky"`, `contact_report=True`.
- Warp count sweep for A: W∈{1,2,4,8} at 4096 envs to find the occupancy knee (§7).

### 9.2 Metrics (both required by the goal)
1. **Step time** — read from the `wp.ScopedTimer("S6_PGS_Solve", ...)` (:2414) and the full-step timer; report median over N≥200 steps after a 50-step warmup. Net = baseline(B) step-time / colored(A) step-time. Compare A and B each to C (PhysX) to report gap-close.
2. **≤8 mm peak interpenetration fidelity** — per step, max over contacts of `max(0, -phi)` in mm, where `phi` is the signed normal gap (`mf_phi`, :1592, or recovered as `bias*dt/beta` with `beta=pgs_beta=0.2` at :256, `dt=0.005`). Report peak over the whole run. PASS iff ≤ 8 mm. (mm-tau gate predicts 0.4–0.9 mm; a regression above ~1 mm is a red flag.)

Report A vs B vs C as a table per (probe, envs), plus the W-sweep at 4096.

### 9.3 Exact dirty-CI trigger recipe (from the worktree)
The repo already runs the sim-throughput-perf buildkite pipeline (artifacts under `artifacts/buildkite/sim-throughput-perf/feather_pgs/...`). Trigger a **dirty-CI** run from the feature worktree:

```bash
# from the worktree root, on the feature branch
cd /home/dturpin/repos/il-newton-dev/newton-collab
git worktree add ../fpgs-coloring-bench bk-slurm-broker-green   # or the feature branch
# build the warp kernel cache once (so first-step compile isn't in the timing)
WARP_CACHE_PATH=.warp-cache uv run python -c "import newton; print('warmup')"
# launch the perf pipeline dirty (uncommitted/feature build) via the broker:
#   uses the same buildkite step the existing artifacts came from (sim-throughput-perf),
#   parameterized for probes p10,p11 x envs 256,1024,4096 x arms {colored, vel_off, physx}
bk-trigger sim-throughput-perf \
  --dirty \
  --probes p10_cube_rain,p11_cube_rain \
  --envs 256,1024,4096 \
  --arms feather_pgs_colored,feather_pgs_veloff,physx \
  --warps 1,2,4,8 \
  --branch $(git rev-parse --abbrev-ref HEAD)
```
(If `bk-trigger` is not the local wrapper name, the equivalent is the `RemoteTrigger`/broker path used for the existing `sim-throughput-perf` artifacts; the dirty flag tells the broker to build from the current worktree SHA rather than `main`.) Results land under `artifacts/buildkite/sim-throughput-perf/feather_pgs/<probe>/envs-<n>/{measurement,result,metadata}.json` — the same schema as the existing `franka_lift`/`g1_flat` artifacts in the tree.

Reading results:
```bash
# step time (median ns/step) and penetration from result.json
jq '.metrics.step_time_ms_median, .metrics.peak_penetration_mm' \
  artifacts/buildkite/sim-throughput-perf/feather_pgs/p10_cube_rain/envs-4096/result.json
```
Net vs baseline = `veloff.step_time_ms_median / colored.step_time_ms_median`; gap-to-PhysX = `colored.step_time_ms_median / physx.step_time_ms_median`. PASS the fidelity gate iff `peak_penetration_mm <= 8` in every arm.

---

## Rollout phases (de-risking the killers)
- **Phase 0 (bring-up):** colored kernel + host `color_graph` oracle CSR, W=1, n_colors from oracle. Validate §8 bit-identity (trivial CSR) and fidelity (real coloring). No perf claim yet. *This is the grafted host-recolor prototype — proves the kernel and fidelity before any GPU coloring or multi-warp.*
- **Phase 1 (parallelism):** `__launch_bounds__` + W>1, occupancy sweep §7. First real step-time numbers; this is where the gap-close is proven or disproven.
- **Phase 2 (device coloring):** `mf_color_assign` + `mf_recolor_dirty` on device, persistence-gated; removes the host oracle from the hot path.
- **Phase 3 (warm-start):** §5 match_index-gated `mf_impulses` carry; measure iteration savings.
