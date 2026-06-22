# F2 design: eliminate the per-step live mf6 relayout (kpw / K=1 path)

**Verdict: GO-with-caveats.** Candidate **A** (producer inner-emit dual-write +
skip the 4 mf6 transpose launches). Ship it as a *staged* change behind
`FEATHER_PGS_TPW_K1_F2`, gated on a bit-identity proof and a p10/p11 nsys
ablation that must beat F1c-alone before the flag is allowed to default-influence
anything.

This reverses the prior `F2_DEFERRED.md` deferral on ONE specific, verified
ground the prior analysis missed (the re-entry multiplier, below). The other
three candidates (B strided in-place read, C launch-fusion, D in-kernel
recompute) are confirmed dead-ends — see §6.

---

## 1. The fact that flips the economics: the transpose runs N×/step, the producers run 1×/step

`_build_tpw_k1_inputs` (solver_feather_pgs.py:2183) — which launches the 4
`_tpw_k1_transpose_mf6_to_inner` kernels (:2263-2266) — is called from *inside*
`_launch_matrix_free_gs_solve` (:2475). And `_launch_matrix_free_gs_solve` is
**re-entered 2-3× per step** on the product path:

- velocity solve (the `pgs_iterations` solve, :3218),
- velocity post-solve (`_run_matrix_free_velocity_post_solve`, :2664),
- (debug path adds a per-iteration re-entry at :3121).

So when K=1 is on, the 4 mf6 transposes execute **once per re-entry = 2-3×/step**.

Meanwhile the *producers* of the world-outer mf6 buffers run **once per step**,
in the contact-assembly phase, NOT inside `_launch_matrix_free_gs_solve`:

- `build_mf_contact_rows` @ :4477 (writes `mf_J_a/_b`),
- `compute_mf_effective_mass_and_rhs` @ :5628 (writes `mf_MiJt_a/_b`).

**Verified step-constancy across re-entries:** between the velocity solve and the
velocity post-solve, `_conclude_matrix_free_position_problem` (:2675) recomputes
only the **RHS** (`_stage4_compute_rhs_world` → `rhs_unbiased`,
`_compute_mf_rhs_bias` → `mf_rhs_unbiased`). It does **NOT** rebuild J or MiJt.
The re-entry re-packs `mf_meta` (which carries the changed RHS — that is why
`mf_meta` has its OWN separate transpose at :2268 that F2 KEEPS) but reads the
**identical** `mf_J_a/_b`, `mf_MiJt_a/_b`. The 4 mf6 transposes therefore copy
**byte-identical bytes** on the 2nd and 3rd re-entry — pure redundant work.

**Consequence.** `F2_DEFERRED.md`'s economic objection ("payoff well under the
8.4% headline because dual-write keeps the world-outer stores AND adds strided
ones") undercounts the win. The dual-write pays its added strided store **once**
(in the once-per-step producers) and eliminates the mf6 transpose on **every**
re-entry. Realistic gross saving ≈ **N × 8.4%** of mf-relayout (N≈2-3), traded
for a **1×** strided store. That asymmetry is the whole case for GO.

nsys anchor (artifacts/local/nsys-kernel-breakdown/ANALYSIS.md, build #1177-1180):
on p10-ON the kpw solve saves ~8900 ms but the transpose adds **~4600 ms back**;
`_tpw_k1_transpose_mf6_to_inner` alone is **8.4% = 1815 ms** of one re-entry's
worth as summed in the CSV. Killing its redundant re-executions is the target.

---

## 2. Why A and not B/C/D (one-line each, full kill in §6)

| Cand | Idea | Byte-id | Blast radius | NET payoff vs 8.4% | Regress risk | Verdict |
|------|------|---------|--------------|--------------------|--------------|---------|
| **A** | producer writes inner flats; skip mf6 transpose | YES (dual-write mirrors store-vs-skip) | 3 shared kernels (gated, dead-code-elim off-path) | **+ (N× saved, 1× added store)** | Moderate (managed) | **GO** |
| B | kpw reads world-outer in place, strided | YES trivially | NONE outside K=1 | **NEGATIVE** — moves un-coalescing INTO the 12-24×/step hot read loop | HIGH | dead |
| C | fuse 4 mf6 launches → 1 | YES trivially | NONE | **~0** — bandwidth-bound, only ~3 dispatch saved | low | dead |
| D | recompute J/MiJt in kpw | NO (friction sorted-contact lookback unreproducible per-row) | NEW inputs/maps | NEGATIVE (more traffic + reg pressure on occupancy-bound kernel) | HIGH | dead |

The deciding axis is **NET payoff under coalescing**. The transpose exists
*specifically* to give the kpw kernel a coalesced read: kpw reads
`mf_J_a.data[(i*6+k)*W + world]` with `world` (the warp lane) unit-stride
innermost (solver:9188/9193/9230/9235/9247/9252 contact phase; :9302/9307/9318/
9323 rt4 phase) — a warp of 32 worlds reads 32 consecutive floats. Candidate B
would make those reads `[(world*M_MF+i)*6+k]` → 32-way scatter (stride
`M_MF*6 ≈ 12288` floats ≈ 48 KB at M_MF=2048) inside the loop the kpw re-reads
~12× per sweep × (contact+rt4). That is the wrong direction on a kernel ncu
already calls latency/occupancy-bound (build-887: SoL 3.7%, long-scoreboard
stalls). A keeps the kpw read coalesced (the inner flats it fills have the
identical world-inner layout the transpose produced) and only changes WHERE the
write happens (folded into the producer), so the hot read loop is byte- and
bandwidth-identical.

---

## 3. Candidate A — exact design

### 3.1 What changes
Producers additionally fill the K=1 world-inner flats `_k1_mf_J_a/_b`,
`_k1_mf_MiJt_a/_b` (alloc solver:2171-2174, layout `[(i*6+k)*W + world]`) at the
SAME conditional branch as their world-outer store. Then the 4 mf6 transpose
launches are SKIPPED under F2. Everything else (world-outer stores, all 8
non-K1 consumer sites, the `mf_meta` / `mf_lam` / `mf_row_mu` / dense / J/Y / v
transposes) is untouched.

### 3.2 Byte-identity contract (the correctness crux)
The transpose blindly copies all 6 components of every row `i < mf_cnt[world]`
from the world-outer buffer — including bytes the producers **skipped** (left
stale). To be byte-identical the inner emit must reproduce the EXACT write-vs-skip
pattern AND the stale bytes:

- `build_mf_contact_rows` writes `mf_J_a` only if `body_a >= 0` (kernels.py:3727),
  `mf_J_b` only if `body_b >= 0` (:3743). For friction rows it `break`s out of
  the `row_offset` loop when `not will_add_friction` (:3713-3714), so rows
  2..M_MF beyond the live count are never written **by this contact** — but the
  transpose is gated by `mf_cnt[world]` (the finalized live row count), so it
  only copies rows the producers DID populate. Within a live row, the
  conditional `body_a/body_b` skip is the hazard: if `body_b < 0`, the transpose
  copies whatever stale `mf_J_b` bytes sit in the world-outer buffer.
- `compute_mf_effective_mass_and_rhs` writes `mf_MiJt_a` only if `ba >= 0`
  (:4209), `mf_MiJt_b` only if `bb >= 0` (:4221). For a vel-limit row (`bb==-1`)
  `mf_MiJt_b` is NEVER written → transpose copies stale `mf_MiJt_b`.
- `populate_rigid_velocity_limit_rows` zeros all 6 of `mf_J_a/_b` then sets one
  axis (kernels.py:4001-4003); it never touches `mf_MiJt`.

**The clean guarantee** (per F2_DEFERRED §"If revived"): the inner store goes in
the **same conditional branch** as the world-outer store, writing the **same
value just stored**. Then write==write and skip==skip by construction, AND the
skipped inner slots must match what the transpose would have copied from the
world-outer buffer in that slot. The world-outer buffer is NOT re-zeroed per step
(buffers persist), so a producer that skips a slot leaves the OLD world-outer
value there, which the transpose copies. The inner K=1 scratch is ALSO not
re-zeroed per step. So for byte-identity the inner skipped slot must equal the
world-outer skipped slot. **These are only guaranteed equal if the inner scratch
also carries the same history** — which it does NOT in general (different write
sequence). **Therefore the safe rule is: in every skip branch, STILL copy the
world-outer value into the inner flat** (an unconditional `dst_inner[...] =
src_outer[...]` after the conditional world-outer store, OR mirror the exact
conditional and additionally copy-through the stale world-outer in the else).
Simplest correct form: at the END of each producer's per-row work, do an
unconditional 6-wide copy of the **final world-outer row** into the inner flat
(read-after-write of the buffer the producer just touched), so the inner flat is
always an exact image of the world-outer row regardless of which sub-stores
fired. That makes the inner emit a true mirror of "world-outer row → inner row,"
i.e. exactly what the transpose did — minus the separate launch and minus the
extra world-outer re-read the transpose incurred.

> **Recommended concrete form (avoids the stale-slot trap entirely):** do NOT
> try to interleave the inner store into each `body_a>=0` branch. Instead, after
> the producer finishes writing row `i`'s world-outer slots, emit a single
> 6-iteration `for k: inner[(i*6+k)*W+world] = outer[world,i,k]` copy of that row
> (guarded by the compile-constant gate). This reads the row the producer just
> wrote (hot in L2/registers) and copies all 6 — identical to the transpose's
> per-row behavior, including stale slots, because it copies the SAME world-outer
> bytes the transpose would have. Byte-identity is then a tautology: same source,
> same destination layout, same gate (`i < mf_cnt`). It costs one extra 6-wide
> read+write per live row in the producer vs the transpose's read+write in a
> separate launch — but saves the (N-1) redundant re-launches per step.

This "copy-through after write" framing is strictly simpler and safer than the
DEFERRED dual-write's "mirror each conditional branch" framing, and it makes the
NaN-poison gate (§4) a belt-and-suspenders check rather than the sole guarantor.

### 3.3 Producer write-set must cover the kpw read-set
`mf_cnt = self.mf_constraint_count` gates BOTH the transpose (:2247) and the kpw
`m_mf_g` bound (:2481). The copy-through must run for exactly `i < mf_cnt[world]`.
- `build_mf_contact_rows` runs per-contact and writes rows `slot..slot+2`; the
  copy-through for `mf_J` rows must cover every J row the kpw reads. Since
  `mf_cnt` is finalized AFTER build (`finalize_mf_constraint_counts` @ :4519+),
  the build kernel does not know the final `mf_cnt`. **Resolution:** the
  copy-through for J is keyed on the rows build itself writes (`row_idx = slot +
  row_offset` for the rows it populates), which is a SUPERSET-or-equal of the
  rows < mf_cnt (build writes exactly the live rows). This is sound because the
  transpose's `i < mf_cnt` gate and build's `row_idx` write-set coincide on the
  live rows; rows build never writes are also rows kpw never reads. (Verify in
  the bit-id harness, §4, including the vel-limit rt4 rows which build does NOT
  produce — those come from `populate_rigid_velocity_limit_rows`, which must do
  its own J copy-through.)
- `populate_rigid_velocity_limit_rows` writes its rt4 J rows → it must
  copy-through `mf_J_a/_b` for those slots. It writes NO `mf_MiJt`, and for rt4
  `bb==-1` so `compute` never writes `mf_MiJt_b` → the transpose copies stale
  `mf_MiJt_b`. The copy-through in `compute` must therefore copy the stale
  world-outer `mf_MiJt_b` for those rows (the "copy-through after write" form
  does this automatically: it copies the world-outer row whether or not the
  `bb>=0` store fired).
- `compute_mf_effective_mass_and_rhs` writes `mf_MiJt_a/_b` per row `i < mf_cnt`
  (its grid is `W*M_MF`, early-returns `i >= mf_cnt`). Its copy-through covers
  exactly the kpw `mf_MiJt` read-set. 

### 3.4 Signature / launch-site edits (blast radius)
Three shared `@wp.kernel`s gain: one `int` compile-constant gate
(`emit_k1_inner`), `W` (the per-row producers don't currently know W — add it),
and the relevant inner-flat output array(s):

1. **`build_mf_contact_rows`** (kernels.py:~3608; launch solver:4477) → add
   `emit_k1_inner: int`, `W: int`, outputs `k1_mf_J_a`, `k1_mf_J_b`
   (`wp.array[float]`). Copy-through J rows.
2. **`populate_rigid_velocity_limit_rows`** (kernels.py:~3965; launch
   solver:4598) → add `emit_k1_inner: int`, `W: int`, outputs `k1_mf_J_a`,
   `k1_mf_J_b`. Copy-through its rt4 J rows.
3. **`compute_mf_effective_mass_and_rhs`** (kernels.py:~4152; launch
   solver:5628) → add `emit_k1_inner: int`, `W: int`, outputs `k1_mf_MiJt_a`,
   `k1_mf_MiJt_b`. Copy-through MiJt rows.

At each launch site, when `self._tpw_k1 and self._k1_f2`: pass `emit_k1_inner=1`,
`W=self.world_count`, and the real `self._k1_mf_*` flats. Otherwise pass
`emit_k1_inner=0`, `W` (harmless), and a **cached 1-element dummy** `wp.zeros`
array (allocate once in `__init__`, e.g. `self._k1_dummy_f32 = wp.zeros(1,
dtype=f32)`), bound at every non-F2/non-K1 launch. The `if emit_k1_inner != 0:`
guard is a compile-constant literal `0` on the off path → Warp/ptxas dead-code
eliminates the copy block → **default-path PTX byte-unchanged** (the same
mechanism F1c uses; verified pattern). `enable_backward=False` is already implied
by these being non-grad kernels — confirm each `@wp.kernel` carries it (the
transpose kernels at :9489/:9503/:9532 already do); add where missing on the 3
producers if backward is not otherwise disabled.

> **Bind-error guard:** a missed/mis-typed dummy at ANY of the 3 sites is a launch
> error or silent wrong-array bind, and `compute` @ :5628 feeds EVERY scene
> (non-K1 product path) so a bind error there breaks all scenes. Mitigation: a
> unit test that constructs the solver with K=1 OFF and runs one step (must not
> raise / must be byte-identical to pre-F2), PLUS the dead-code-elim PTX check.

### 3.5 Skip the mf6 transposes
In `_build_tpw_k1_inputs`, wrap the 4-launch mf6 loop (:2257-2266) in
`if not (self._k1_f2):` so F2 SKIPS them (producers already filled the flats).
**KEEP** every other transpose: `mf_meta` (:2268, RHS changes per re-entry →
MUST re-transpose), `mf_lam`/`mf_row_mu` (:2248-2255, lam evolves), dense scalars
(:2205-2218), J/Y (:2224-2241), v-gather (:2274-2277). F2 touches ONLY the mf6
quartet.

### 3.6 Enablement
Replace the `raise ValueError` at solver:747-750 with `self._k1_f2 = True` (still
require `self._tpw_k1`; raise if `_k1_f2 and not _tpw_k1` — F2 is meaningless
without the K=1 path).

---

## 4. Bit-identity test (extend f1c_transpose_bit_identity.py)

Reuse the harness pattern at
`artifacts/local/fpgs-k1-dense/f1c_transpose_bit_identity.py` (the one that
proved F1c `max|J_world_current - f1c| = 0.000e+00` on the captured cap892 J).
Add an `f2_mf6_bit_identity` mode on cap892 (`/tmp/cap892.npz`, N=256, RTX 3090,
`.venv` py3.12 warp 1.14.0.dev20260518, `WARP_CACHE_PATH=/tmp/warp_cache_f2`):

1. Run the K=1 path with F2 **OFF** (transpose path). Snapshot
   `_k1_mf_J_a/_b`, `_k1_mf_MiJt_a/_b` (the inner flats) AND the final `v_out`
   and `peak_pen`.
2. **NaN-poison** the 4 inner scratch buffers (`_k1_mf_*` = NaN) to catch any
   slot the F2 emit fails to cover.
3. Run the K=1 path with F2 **ON** (producer copy-through, transpose skipped).
   Snapshot the same 4 inner flats + `v_out` + `peak_pen`.
4. **Assert** `max|F2 - OFF| == 0.000e+00` for all four inner flats, INCLUDING:
   - vel-limit rt4 rows where `bb==-1` (stale `mf_MiJt_b` slot — the prime
     stale-slot hazard),
   - single-body contact rows (`body_a>=0, body_b<0` → stale `mf_J_b`/`mf_MiJt_b`),
   - friction rows (`row_offset>0`) including the `friction_anchor_scale=0.5`
     shared-anchor branch.
5. **Assert** `max|v_out_F2 - v_out_OFF| == 0` and `peak_pen` byte-identical (the
   end-to-end seal).
6. Re-run 3× for determinism (the kpw colored order is a valid GS; F2 does not
   change the kpw read order, so this must be exactly 0, not within-tolerance).

A NaN surviving into `v_out` (NaN-poison leak) = an uncovered slot = FAIL. This
is the gate that the prior deferral flagged as mandatory ("invisible to
penetration-only metrics").

Also add a **default-path PTX-unchanged check**: build `build_mf_contact_rows` /
`populate_rigid_velocity_limit_rows` / `compute_mf_effective_mass_and_rhs` with
`emit_k1_inner=0` and confirm the generated module hash / PTX matches the
pre-F2 kernel (dead-code-elim proof). If Warp does not expose a stable hash,
diff the lowered PTX for the off-path launch.

---

## 5. CI ablation

Run all three dense scenes (p10, p11) AND a robot scene (franka or h1, where
`mf_cnt << M_MF` so the inner-emit covers fewer rows) under the existing
dense-contact / kpw harness
(`.claude/worktrees/kpw/tpw_blackwell/kpw_node_bench.py` for cap892;
`il-newton-dev` pipeline build for full-step):

| Config | Flags | Expect |
|--------|-------|--------|
| baseline | K1 off | reference throughput + fidelity |
| F1+F1b | K1 on, F1c off, F2 off | current shipped kpw |
| F1c | K1 on, F1c on, F2 off | JY idle-launch elim (already shipped) |
| **F1c+F2** | K1 on, F1c on, **F2 on** | **must beat F1c-alone on p10/p11 net kernel time** |

Gating rule (same as F1c's): **keep F2 only if the F1c+F2 nsys kernel-time on
p10 AND p11 is lower than F1c-alone by more than run-to-run variance** (mf6
stddev ~250 us/launch per ANALYSIS.md → require ≥ ~1 ms net improvement, well
inside reach if N≥2 re-entries each drop a 1815 ms transpose). Fidelity
(`peak_pen`, tunneling count) must be byte-identical to baseline on all scenes.
If F2 does NOT beat F1c on p10/p11 (e.g. producer becomes store-bound and the
once-per-step added store eats the N× transpose saving), **keep the flag wired
but default-off and re-affirm the deferral** — do not ship a regression on the
headline.

---

## 6. Why B, C, D are dead (kept for the record)

**B (kpw reads world-outer in place, strided):** byte-identical and zero non-K1
blast radius, BUT trades a once-per-step coalesced relayout for **uncoalesced
32-way-scattered reads inside the kpw's hottest loop** (jv-accumulate + v-apply,
re-read ~12× per sweep × contact+rt4 phases). On a latency/occupancy-bound kernel
(build-887 ncu) this is the wrong direction; expected NET **negative-to-zero** on
exactly the p10/p11 headline scenes (dense MF, large M_MF → widest scatter). The
transpose moves each float once/step; the kpw reads it ~12-24×/step — moving the
un-coalescing into the 12-24× loop loses. DEAD.

**C (fuse 4 mf6 launches → 1):** byte-identical, zero blast radius, but the mf6
transpose is **bandwidth-bound** (~888 GB/s ≈ 49% of peak per the p10-ON CSV),
NOT dispatch-bound. Fusing 4→1 removes ~3 dispatches ≈ 0.007-0.035% of step ≈
<0.05% of the 8.4% headline; below the 250 us/launch nsys stddev (unmeasurable).
Cannot improve coalescing (the scattered write to world-inner is intrinsic to the
target layout). DEAD.

**D (recompute J/MiJt in kpw):** byte-identity NOT cheaply achievable — the J
build's `friction_anchor_scale` (0.5 vs 1.0) depends on a sorted-contact lookback
over preceding contacts (kernels.py:3700-3708) that the per-world-row kpw kernel
cannot reproduce, and the `MiJt = Hinv·J` 6×6 spatial matmul must match the FMA
order bit-for-bit. Feeding a recompute needs MORE world-inner traffic (~64+ f32:
raw contact geom + 36-f32 Hinv) than the 24 f32 it avoids storing, PLUS register
pressure on the occupancy-bound kpw (where the K=1 win is itself fragile — only
2.52× realized on p10 vs 3.2× cap). NET negative + silent friction divergence.
DEAD.

---

## 7. Residual risk on A (honest)

1. **Producer store-bound flip.** `compute_mf_effective_mass_and_rhs` runs on a
   `W*M_MF` grid; adding a 6-wide strided inner store per live row could make it
   store-bound and erase the per-step saving. This is the one way A regresses;
   it is exactly what the §5 ablation measures. Because the added store is
   ONCE/step and the saved transposes are N/step, the margin is favorable, but
   **it is unproven until the p10/p11 nsys runs** — hence GO-**with-caveats**, not
   unconditional GO.
2. **Stale-slot divergence** — fully neutralized by the "copy-through after
   write" form (§3.2) + NaN-poison gate (§4). The hazard exists only if someone
   implements the riskier "mirror each conditional branch" form; the design
   mandates copy-through.
3. **Bind error at `compute` launch** breaks all scenes → guarded by the K1-off
   byte-identity unit test + PTX-unchanged check (§4).
4. **Graph-capture:** store count is static (literal-0 compile-constant gate, no
   dynamic launch dims); launch dims of the 3 producers are unchanged. Safe.

---

## 8. Implementation checklist

1. `solver:__init__` — allocate `self._k1_dummy_f32 = wp.zeros(1, dtype=f32,
   device=device)`. Replace `raise ValueError` (:747-750) with `self._k1_f2 =
   True`; raise only if `_k1_f2 and not _tpw_k1`.
2. `kernels.py` — add `emit_k1_inner: int`, `W: int`, + inner-flat output args to
   `build_mf_contact_rows`, `populate_rigid_velocity_limit_rows` (J only),
   `compute_mf_effective_mass_and_rhs` (MiJt only). Per-row copy-through guarded
   by `if emit_k1_inner != 0:`, writing `inner[(i*6+k)*W+world] = outer[world,i,k]`
   for the row just produced. Confirm `enable_backward=False`.
3. `solver:4477 / :4598 / :5628` — bind `emit_k1_inner`, `W`, real `_k1_mf_*`
   flats when `_tpw_k1 and _k1_f2`, else `0` + dummy.
4. `solver:_build_tpw_k1_inputs :2257-2266` — wrap the 4 mf6 transposes in
   `if not self._k1_f2:`.
5. Extend `artifacts/local/fpgs-k1-dense/f1c_transpose_bit_identity.py` with the
   `f2_mf6` mode (§4): NaN-poison + 0.0 max-abs-diff on the 4 inner flats + v_out
   + peak_pen, 3× determinism.
6. Add a K1-OFF byte-identity unit test (one step, F2-off path unchanged) +
   default-path PTX-unchanged check.
7. Run §5 CI ablation. KEEP F2 only if F1c+F2 beats F1c-alone on p10 AND p11 net
   kernel time by > variance, fidelity byte-identical. Otherwise re-defer (flag
   stays wired, default off).
