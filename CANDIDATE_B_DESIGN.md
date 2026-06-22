# Candidate B (recompute MiJt) — design verdict: NO-GO

**Decision: NO-GO.** Do not build Candidate B on the shipping default warp-per-world
MF-GS solve path. Both variants the model asked us to weigh lose on the term that
actually binds the kernel. No flag was added; no kernel was shipped. The model
(`SOLVE_KERNEL_MODEL.md` §5c) already flagged Candidate B as the likely-backfire
lever (lower confidence than Candidate A); this analysis confirms the backfire
quantitatively from the real lane layout and the real ncu occupancy anchor.

---

## 1. What Candidate B proposed

Drop the per-contact `mf_MiJt_a/_b` storage (6 f32 each) and its per-sweep re-read;
instead store only the per-body inverse spatial inertia `mf_body_Hinv` and recompute
`MiJt = Hinv · Jᵀ` inside the solve kernel each Gauss-Seidel sweep, spending the
~200x-idle FP32 pipe to cut the ~92%-of-traffic matrix-free re-read and raise
arithmetic intensity (AI ~0.21 FLOP/B today).

The crux the design had to settle: the binding ceiling everywhere is
**occupancy / latency-hiding** (ncu SoL 3.67%, ~2 blk/SM, smem-bound), NOT bandwidth
(95x headroom) or compute. So the AI raise must outweigh any occupancy/latency cost.
Sub-question: can `Hinv` be **read from global** each sweep (amortized per body,
coalesced, AI raise at ~no register/occupancy cost), or does it force **caching**
(occupancy cost)?

## 2. The decisive layout fact (settles the crux)

The default `friction_mode='current'`, `MFGS_COLORED=False` solve is **one lane per
DOF**, not a 6-vector per row (`solver_feather_pgs.py:8285-8333`):

- Lane `l ∈ [0,5]` owns body-a DOF `l`; lanes 6-11 own body-b. Per row, lane `l`
  prefetches a **single float** `cur_Ja = mf_J_a[mf6+l]` (for `J·v`) and a **single
  float** `cur_MiJta = mf_MiJt_a[mf6+l]` (for the v-update), then applies
  `s_v[dof_a+l] += cur_MiJta * delta_impulse` (solver:8288, 8311, 8333).
- So `mf_MiJt` is exactly **1 of ~2 float global loads per active lane per row** =
  48 B of the 112 B/contact re-read (~43%).

To recompute `MiJt_a[l] = Σ_k Hinv[ba][l,k]·Ja[k]`, lane `l` needs **row `l` of
Hinv (6 floats)** plus **all 6 `Ja[k]`**. `Ja` is free via 5 `__shfl` across the 6
sibling lanes (already resident). `Hinv` is the problem.

The GS rows are walked in **sorted-contact slot order** (`for i = 0..m_mf-1`,
solver:8319), NOT body-major (contacts arrive in collision-sorted slot order;
confirmed at solver:1750-1761). Consecutive rows reference **different bodies**, so a
per-row `Hinv` load **cannot be amortized within a sweep** without caching the whole
per-world `Hinv` set.

## 3. Both variants net negative — the numbers

### B-glob (read Hinv from global per row)

Per row, the 6 active A-lanes must each load row `l` of `Hinv[ba]` = 6 floats ⇒ 36
floats = 144 B for side A; +144 B side B = **288 B/contact, every row, every sweep**,
because slot order gives no body reuse.

| | re-read B/contact/sweep | per-lane dependent loads/row | AI vs today |
|---|---|---|---|
| today (store+read MiJt) | 112 (J 48 + MiJt 48 + meta 16) | 1 (MiJt float) | 1.00x |
| **B-glob** | **352** (J 48 + meta 16 + Hinv 288) | **6** (Hinv row) | **~0.32x (≈3.1x WORSE)** |

The model's "amortized per body, coalesced, AI raise with no occupancy cost" premise
is **FALSE on the default slot-ordered path**. B-glob trades a 48 B MiJt re-read for a
288 B Hinv re-read and grows the per-lane dependent-load chain 1→6 on a kernel that is
**long-scoreboard-stall bound** (40.3% of issue, ncu build-887). Strictly negative.

### B-cache (cache Hinv in smem to actually amortize per body)

`mf_body_Hinv` is `wp.spatial_matrix` = 36 f32 = **144 B/body** (solver:1816). The
default kernel is **smem-bound at 2 blk/SM** (ncu build-887: 39.43 KB static smem,
Block-Limit-Shared-Mem = 2, 102.4 KB smem/SM, achieved occ 2.82%). Registers have 4x
slack (Block-Limit-Registers = 8), so smem is the binding term.

Caching ≥128 bodies/world ⇒ ≥18 KB ⇒ block smem 39.43 → >51.2 KB ⇒ resident blocks
**2 → 1 (occupancy HALVED)**. Dense piles — the only scenes where re-reading MiJt
costs anything — have the **most bodies**, so the cache is largest exactly where it
hurts most. ~1.4x traffic/AI win against a ~2x occupancy loss = **net ~0.7x**.

Caching Hinv in registers across rows is **infeasible**: the referenced body changes
every row (slot order), so a lane can't hold "its body's Hinv" without a body-major
reorder — which is precisely the K=1 transpose tax already measured-dead on p11/h1.

## 4. Why occupancy, not bandwidth, decides (anchored)

ncu build-887 (real Blackwell sm_120): resident blk/SM = min(reg-limit 8, **smem-limit
2**, warp-limit 48) = 2, SMEM-BOUND; achieved occ 2.82%, 1.00 active warp/scheduler vs
12 max, SoL 3.67%, DRAM 1.18%. The kernel is 95x below its bandwidth roofline. The
recompute's register growth (~+15-40 regs, 170→~185-210) keeps the reg limit ≥8 ≫ smem
limit 2, so registers never bind — B-glob is occupancy-neutral but loses on
traffic/latency; B-cache attacks the binding smem term directly and halves it. **There
is no variant that nets positive.**

## 5. Byte-identity (achievable but a real, wasted cost)

Were it built: recompute must consume the SAME `mf_body_Hinv[b]` that
`compute_mf_effective_mass_and_rhs` reads (`kernels.py:4245-4246`:
`Hinv_a = mf_body_Hinv[ba]; MiJt_a = Hinv_a * Ja`) and hand-match warp's
`spatial_matrix * spatial_vector` FMA lowering per output row (`MiJt[l] = Σ_{k=0..5}
Hinv[l,k]·Ja[k]`, fp32, k ascending). Same array + same k-order ⇒ bit-exact is
replicable, but it demands an F1c/F2-style bit-identity harness plus passing the global
body id (`mf_body_a/b`) into the solve kernel (currently only `mf_local_body_a/b` +
packed dofs are passed) — non-trivial CI/engineering burden against a negative payoff.
Whether warp's mat-vec codegen matches a literal `Σ_k` loop bit-for-bit is **unverified**
and is itself a blocker for any future attempt.

## 6. What to do instead (the genuinely positive levers, from the model §5b)

1. **smem streaming** (40324→15748 B): lifts the SAME 2-blk/SM occupancy ceiling
   Candidate B would damage; helps every scene; 1.6x measured (sm_86, needs sm_120
   confirm P1). This is the inverse of B-cache's failure mode.
2. **rt4 row drop** — shipped (lazy vel-limit alloc, ACTIVATION_FRACTION=0.5).
3. **Candidate A (F2 world-innermost geometry build)** — byte-identical, layout-only,
   converts the measured 3.2x K=1 kernel win into full-step throughput.

If anything, the correct direction is the **inverse** of Candidate B (model lever #4:
keep MiJt cached, recompute the *cheap* per-lane `J·v`) — but that is not Candidate B
and is 2nd-order behind occupancy.

## 7. Falsifier (what would overturn NO-GO)

The NO-GO rests on two pure-arithmetic facts, either of which, if wrong, flips the call:

- **F-glob:** if a target dense scene's contacts were body-grouped in slot order (or a
  cheap fused body-major emit existed at geometry-build that did NOT re-incur the K=1
  transpose tax), B-glob's Hinv would amortize 288→~48 B/contact and the AI raise could
  go positive. **Test:** instrument the default solve at p11/16384 — measure the
  fraction of consecutive GS rows sharing `ba`/`bb`. If >~50% share a body in slot
  order, re-open B-glob. (Expected from collision-sorted slot order: near 0%.)
- **F-cache:** if a target dense scene has **<~64 free rigid bodies/world**, the smem
  Hinv cache fits within 2 blk/SM (no occupancy halving). **Test:** read `body_count`
  per world from the p11 capture. But those are exactly the scenes where the MiJt
  re-read is cheapest, so even then the win is marginal; the catastrophic-halving
  framing weakens but the "no worthwhile win" conclusion holds.

If neither falsifier fires (the expected case), Candidate B is dead.

## 8. Status

- No code shipped. Flag `FEATHER_PGS_MFGS_RECOMPUTE_MIJT` **not** added.
  `ci_plan.py:88 FEATHER_PGS_TUNING_ENV` **not** modified. With no flag, the default
  path is unchanged / byte-identical to today by construction.
- Key files for any future revisit: `kernels.py:4156` (compute_mf_body_Hinv),
  `kernels.py:4179-4265` (compute_mf_effective_mass_and_rhs — the MiJt writer + exact
  arithmetic to replicate), `solver_feather_pgs.py:8285-8333` (the default per-lane
  MiJt re-read), `solver_feather_pgs.py:1816` (mf_body_Hinv alloc, spatial_matrix).
