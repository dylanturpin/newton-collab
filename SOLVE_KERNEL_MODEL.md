# A Validated Performance Model of the FPGS Matrix-Free Gauss-Seidel Contact Solve

Scope: the `pgs_solve_mf_gs` kernel (dense cube piles p10/p11, also h1) as a function of the
parallelization strategy (threads-per-world). One block per world, `k` cooperating warps walking
`R` contact rows (3 PGS rows/contact) over `iters` colored Gauss-Seidel sweeps, re-reading
~112 B/contact of cached matrix-free geometry each sweep. Hardware: RTX PRO 6000 Blackwell
(sm_120, 188 SM); secondary RTX 3090 (sm_86, 82 SM). All numbers below are read from real
artifacts (cited inline); derived quantities are flagged.

---

## 0. The one-sentence answer

The kernel is **latency-bound by occupancy** at every measured operating point (ncu SoL 3.67 %,
DRAM 1.18 %, AI ~0.21 FLOP/B). At the env counts that matter (W >= ~4096) **worlds alone supply
the latency-hiding parallelism**, so adding threads-per-world buys only the serial color-chain
sync tax with zero occupancy payoff. The optimal threads-per-world is a **decreasing** function of
GPU fill (K8 under-fill -> K1 at fill), and the genuinely high-leverage wins are **not** in the
parallelization axis at all but in the kernel footprint (smem streaming) and the row count
(rt4 drop). The much-celebrated colored "W4 = 2.40x" is a **low-env occupancy artifact** that the
model predicts will not survive to high env.

---

## 1. The four lenses reconcile into ONE model with three regimes

All four supplied lenses (roofline / Little's-law-MLP / Amdahl-work-span / unified-latency) are the
**same model viewed from different boundary regimes**. They agree on mechanism and disagree only on
which term they foreground. The unified statement:

> **Per-step solve time `T = max( T_latency , T_bandwidth )`, and at every measured point
> `T_latency` wins by ~95x.** `T_latency` is governed by `hide = resident_warps_per_scheduler`
> relative to the ~9-warp pool needed to cover the long-scoreboard DRAM latency. `hide` is supplied
> by **(worlds resident per SM) x (threads cooperating per world)**. Every strategy is a different
> way to fill `hide`; none changes the arithmetic intensity, so none can ever become
> bandwidth- or compute-bound.

The three regimes and which lens is "active":

| Regime | Condition | Binding term | Foregrounding lens | Behavior |
|---|---|---|---|---|
| **Under-fill** | `W < ~752` worlds (≈ S·blk/SM) | `hide` too small | MLP / occupancy | extra threads/world (W2..W8) or extra waves HELP — this is where colored 2.40x lives |
| **Fill / latency** | `~752 <= W`, dense | `hide` saturated by worlds; per-color sync is now the only delta | Amdahl color-chain | extra threads/world HURT (sync tax, no payoff). K=1 (1 thr/world, 32 worlds/warp) wins by max-MLP-per-warp |
| **Bandwidth** | `W·R·iters` huge (NOT reached) | `T_bw` | Roofline | never active: dense p11 @16k is ~15x below `T_bw` |

**Why a single ceiling, not three independent ones.** The textbook roofline has two ceilings
(compute, bandwidth). This kernel sits at AI ~0.21 FLOP/B, ~200x to the bandwidth side of the
sm_120 ridge (ridge ≈ 50 FLOP/B = 89 TFLOP/s ÷ 1.79 TB/s), so compute is irrelevant. But it runs
at **1.05 % of DRAM peak** (ncu: 18.76 GB/s of ~1.79 TB/s), so bandwidth is not the active ceiling
either. The active ceiling is a **third, lower** one — the achievable-issue ceiling set by
occupancy/latency-hiding — pinning the kernel ~95x below its own bandwidth roofline. The four
lenses converge on this third ceiling; they are not in conflict.

### The closed-form skeleton

```
hide        = resident_warps_per_SM / num_schedulers       # warps available to hide latency, vs ~9 needed
resident_warps_per_SM = blk_per_SM * (warps_per_block)     # set by smem/reg footprint (occupancy)
blk_per_SM  = min( Smem_SM // smem_block , Reg_SM // (regs*threads) , hw_block_limit )
T_latency   ∝ g(W) * iters * Cc_serial * (L / min(1, hide/Q))   # Q ≈ 9 warps to cover L
T_bandwidth = 112 * R * iters * W / BW
T           = max(T_latency, T_bandwidth)
```

where `Cc_serial` is the serial **color** depth (≈10–13 barriers/sweep), `g(W)=ceil(W/resident_worlds)`
the wave count, `L` the dependent-load latency, `Q≈9`. **Threads-per-world `k` enters only through
`hide`** (more cooperating threads = more MLP per warp) **and through the sync tax** (more `k` = more
expensive `__syncthreads` per color). That tension — `hide` up, sync up — is the entire optimization.

---

## 2. Validation table — every measured point

Each row: the measured anchor (with file), what the model says, and PASS/PARTIAL/FAIL. Numbers were
re-read from the raw artifacts during this analysis.

| # | Measured anchor (source) | Model account | Verdict |
|---|---|---|---|
| **V1** | **Warp-per-world ncu: 0.68 waves/SM @ N=256** (`ncu-build-887/prof_mfgs.details.txt:155`) | `waves = N / (S·blk/SM) = 256/(188·2) = 0.68`. blk/SM=2 is the **measured** Block-Limit-Shared-Mem (39.43 KB static smem). | **PASS** (exact) |
| **V2** | **Warp-per-world SoL 3.67 %, DRAM 1.18 %, 1.00 active warp/sched, stall-long-scoreboard 40.3 % of 9.0 cyc, achieved occ 2.82 %** (details.txt:16,57,84,98,177) | Occupancy ceiling = `blk/SM · warps/blk / max_warps = 2·1/48 = 4.17 %` theoretical; achieved 2.82 % (under-fill, 0.68 waves). 1 warp/sched ÷ ~9-warp pool ⇒ ~11 % issue ⇒ SoL single-digit. The kernel is **latency-bound, not bandwidth-bound** — four independent ncu readings (DRAM 1.18 %, occ 2.82 %, 1/12 warps, SoL 3.67 %) all say the same. | **PASS** |
| **V3** | **K=1 isolated kernel speedup vs serial GROWS with fill: p10 0.98x@4k → 1.80x@8k → 3.20x@16k** (`fpgs-k1-dense/1167_result_p10_interleaved.json`, SUMMARY.md) | K=1 packs 32 independent worlds/warp ⇒ MLP-per-warp jumps 1→32, satisfying Little's law via lanes. As W grows, more waves stack 32-world-MLP warps ⇒ latency hidden ⇒ speedup rises monotonically. | **PASS** (monotone, sign+shape) |
| **V4** | **p11 K=1 crosses later: 0.44x@4k → 0.79x@8k → 1.36x@16k** (`1168_result_p11_interleaved.json`) | p11 has ~2.3x more realized contacts/world ⇒ each K=1 thread walks more rows ⇒ per-thread serial work up ⇒ crossover pushes right. **Mechanistic check from raw:** serial @16k is p10 138.8 vs p11 143.1 ms (**R-invariant** ⇒ serial is occupancy-bound), but K=1 @16k is p10 43.4 vs p11 105.3 ms (**linear in density** ⇒ K=1 is per-thread-rows-bound). The two slopes are different mechanisms, exactly as the model splits them. | **PASS** (the strongest validation — it predicts the p10/p11 *gap* from first principles) |
| **V5** | **K2/4/8 monotonically WORSE than K1 at every N, both scenes** (raw json: p10@16k 43.4/45.9/59.5/106.4 ms for K1/2/4/8) | In the fill regime, splitting one world's rows across k warps does NOT raise resident-world MLP (worlds already overfill) but DOES convert free `__syncwarp` per color into `__syncthreads` across k warps. `d(throughput)/dk < 0`. | **PASS** |
| **V6** | **kpw saturated bench: optimal K falls with fill — K4 best @8k (177.3 ms, 1.52x), K1 best @16k (267.2 ms, 1.92x)** (`kpw-blackwell-bench/kpw_bench_result-saturated.json`) | `K* = smallest k driving hide ≥ Q`. As W rises, worlds supply hide ⇒ K* drops. At 8k worlds don't quite saturate ⇒ K4 helps; at 16k they do ⇒ K1. | **PASS** (sign + the K4→K1 transition) |
| **V7** | **Colored W4+stream+mb4 = 2.40x vs orig-serial-resident @ N=256** (sm_86) (`regcap_evt_result.json`: 11.31 vs 27.14 ms) | At N=256, serial-wpw is at 0.68 waves (under-filled). W4mb4 lifts blk/SM 2→4 and warps/SM 2→16 (`regcap_ptxas_result.json`). **Decomposition:** serial-stream alone = 1.63x (16.69 ms) is the smem unlock (40324→15748 B); W4mb4 adds only 2.40/1.63 = **1.47x** more from the extra warps. The win is ~2/3 streaming, ~1/3 multiwarp, and ONLY because 256 << 752 (under-fill). | **PASS** + identifies streaming as the real lever |
| **V8** | **W4-stream UNCAPPED regresses to 1.54x; W8-stream to 0.79x** (`regcap_evt_result.json`) | Streaming pushes regs 130→159 (`regcap_ptxas_result.json`), making wide W **register-bound**: W4-uncapped = 3 blk/SM, W8-uncapped = 1 blk/SM. The reg cap (min_blocks=4 ⇒ 128 regs, 20 B spill) restores 4 blk/SM. Streaming unlocks warp-COUNT; reg-cap converts warp-WIDTH into resident blocks. | **PASS** (both regressions explained by ptxas footprints) |
| **V9** | **nsys @16k iters=2: solve off→on p10 14831→5892 ms (2.52x), p11 14385→12513 (1.15x), h1 5369→3994 (1.34x)** (`off/on_*_kern_sum.csv`) | Realized < the 3.20x isolated cap because the probe's per-world load is lighter/varying; the realized gradient (p10 high, p11/h1 low) tracks the kernel-bench density ordering. | **PASS** (gradient direction) |
| **V10** | **K=1 transpose tax @16k: p10 4581 ms (jy 1992 + mf6 1815 + meta 320), p11 5093, h1 ~3990** (`on_*_kern_sum.csv`, summed `_tpw_k1_transpose_*`) | The transpose is a **pure-bandwidth, OI=0** index remap (17.17 GB/step @p10). Break-even rule: **K=1 nets a full-step win iff (serial_solve − kpw_solve) > transpose_ms.** p10: 8939 > 4581 ⇒ WIN. p11: 1872 < 5093 ⇒ LOSS. h1: 1375 < 3990 ⇒ LOSS. | **PASS** (reproduces all three signs) |
| **V11** | **Full-step K=1 ON vs OFF: p10 +10 %, p11 −13 %, h1 −32 %** (SUMMARY.md:24–27) | Exactly the V10 break-even signs. After F1+F1b count-gates (recover dead-row transpose): p10 +17.6 %, p11 −9 %, h1 −18.5 % (SUMMARY.md:48–50) — the gates cut transpose bytes, shrinking the losses, as the model requires. | **PASS** |
| **V12** | **F1c (jy launch restructure) = MEASURED NULL** (ANALYSIS.md:28) | The jy transpose is already at ~990 GB/s ≈ 55 % of DRAM peak (it IS coalesced + fully occupied). You cannot speed a kernel already near a roofline by changing how it is spawned — only by moving fewer bytes. So a launch restructure on a bandwidth-saturated kernel is null. | **PASS** (predicted-null, observed-null) |
| **V13** | **F2 re-entry saving VOID at appendix** (task) | The solve runs 1x/step at appendix iters=2; there is no re-entry to amortize. Model has no term that F2-re-entry would touch here. | **PASS** (correctly predicts no effect) |
| **V14** | **p11 & h1 nsys total ON > OFF (transpose tax > solve win)** (`*_kern_sum.csv` totals: p11 25919→29102, h1 12848→15399) | At appendix iters=2 the solve fraction is at its **smallest**, so the fixed transpose tax dominates. Model: transpose is per-step-fixed, solve scales with iters ⇒ iters=2 is the worst case for any transpose strategy. | **PASS** + yields a prediction (P6 below) |

**Honest FAILs / PARTIALs:**

- **V9 is PARTIAL on magnitude.** The model gives the *direction* of the realized-vs-isolated gap
  (p10 2.52x realized vs 3.20x isolated) but not a closed-form for the ~0.7x in-probe factor; it is
  scene/load-dependent and I do not model it analytically. Flagged.
- **No FAIL among the 14**, but several PASSes lean on a **cross-architecture / cross-env
  extrapolation** (next section). I count those as PASS-on-mechanism, not PASS-on-number.

---

## 3. Where the model is hand-wavy or unfalsifiable (adversarial self-audit)

1. **The colored 2.40x is sm_86, N=256, kernel-isolated ONLY.** There is **no** colored-multiwarp
   measurement at sm_120 or at N>256 anywhere on disk. The whole "2.40x is a low-env artifact"
   claim transfers an sm_86/N=256 number across both architecture and env ladder using the
   *kpw Blackwell K-falls-with-fill law* as the bridge. That bridge is sound in mechanism but is an
   **extrapolation, not a measurement**. If #1194 shows the colored kernel scaling differently on
   Blackwell, this is the first thing to fall.
2. **Register-count discrepancy undermines my Blackwell occupancy ceilings.** ncu on real Blackwell
   measured **170 regs/thread, 2 blk/SM** for warp-per-world; ptxas on sm_86 reports **127 regs**.
   My Blackwell blk/SM estimates (built from sm_86 ptxas 127/128/159 figures) likely **overstate**
   achievable blocks/SM. The trustworthy anchor is the ncu 2.82 % occupancy / 2 blk/SM, not the
   ptxas-derived ladder. The streaming "2→~14 blk/SM" lift is *measured on sm_86 only*; its
   Blackwell payoff is inferred.
3. **The 246x color-parallel ceiling is unfalsifiable in practice.** It is a Brent work/span bound
   (`rows/color ≈ 246`) that the kernel never approaches (realized 1.15–3.2x), because occupancy
   binds first. So the color-chain Amdahl term is real but **never the active constraint** — which
   means any prediction "derived from" it is actually a prediction about occupancy wearing an
   Amdahl costume. I keep it only as the per-strategy *sync-tax* term, not as a speedup ceiling.
4. **The per-color load histogram is not on disk.** "10 colors, ~32 blocks/color" comes from
   MEMORY.md (gate-rt4), not a re-readable file. The load-imbalance erosion (~1.5–2x, critical path
   = sum of per-color MAX not AVG) is **illustrative**. If colors are heavily decay-shaped, the
   colored sync tax is worse than modeled (helps my "colored loses" call, but I cannot quantify it).
5. **Host-side graph-coloring rebuild cost per step is unmeasured.** Whether colored-W4 recolors
   every step or caches the settled coloring **materially** changes the full-step prediction. I
   assume a real per-step CPU tax (it is lever (a) for #1194) but **cannot quantify it** — this is
   the single biggest unmodeled term in the #1194 prediction.
6. **AI ~0.21 FLOP/B is an estimate.** From traffic-anatomy (23 FLOP / 112 B), cross-checked by ncu
   FP32 counts (71.9 MFLOP ÷ 0.288 GB DRAM = 0.25). Order-of-magnitude, not instruction-exact. It
   does not matter to any conclusion (0.21 vs 0.25 both ⇒ "never compute-bound").
7. **M_D ambiguity in the transpose bytes.** The 17.17 GB/step figure uses M_D=64; the ncu config
   json shows `dense_max_constraints=512`. If the full-step probe runs M_D=512, the J/Y transpose is
   ~8x larger and the transpose tax even more dominant. The kernel name `..._64_4096_609_...`
   suggests 64 is the binding value at the appendix probe, but I flag the 8x uncertainty.

---

## 4. SHARP falsifiable predictions for builds #1194 / #1195

(Neither has artifacts on disk — confirmed in flight. These are pure model-derived calls.)

### #1194 — colored W4+stream+mb4 FULL-STEP throughput, cubes, env ladder 256→16384

**Headline: the 2.40x kernel win does NOT survive to high env. It decays monotonically with W,
crosses to parity around W≈1024–2048, and is FLAT-to-mildly-NEGATIVE by W≥4096 on dense p11.**

Per-rung kernel-speedup-vs-serial-wpw and resulting full-step delta (full-step uses Amdahl on the
solve fraction `f`; for dense cubes the K=1 ON/OFF deltas imply `f_solve ≈ 0.22`):

| N | serial-wpw waves/SM | predicted **kernel** speedup vs serial-wpw | predicted **full-step** delta | regime |
|---|---|---|---|---|
| 256 | 0.68 (under-filled) | 2.0–2.4x (if sm_86 transfers) | **+10 to +25 %** | colored's best cell |
| 1024 | ~2.7 | 1.4–1.8x | **+5 to +15 %** | crossover band |
| 4096 | ~10.9 (saturated) | 1.1–1.3x | **−3 to +5 %** (host recolor pushes negative) | parity |
| 16384 | ~43.6 (deep saturation) | 1.0–1.2x | **flat-to-negative on p11; small win possible on sparser p10** | colored's worst cell |

Dominant decay drivers, ranked: **(c) occupancy saturation by W≥4096** (kills the only advantage) >
**(b) Amdahl** — solve is a minority of the step so even a surviving kernel win caps step gain at
~1.2–1.3x > **(a) per-step host-side graph-coloring rebuild** (fixed serial CPU tax, worsens at
high N).

**FALSIFIER:** if #1194 shows a sustained **> 1.5x full-step** win at N=16384 on dense cubes, the
occupancy-saturation core of this model is wrong.

> **Where the lenses genuinely disagree (I must flag this).** Three lenses (roofline, MLP,
> Amdahl-work-span) predict colored-W4 nets **flat-to-negative** at high env. The fourth lens
> (unified-latency) makes a sharper, *opposite-signed* call on p11/h1: because colored-W4 is
> **world-outer native** it pays **zero transpose tax**, so vs serial-wpw it could net **+10–17 %
> on p11 and +3–5 % on h1** (a sign-flip from K=1's −9 %/−18 %). **These cannot both be right.**
> The reconciliation: colored-W4 vs *serial-wpw* (no transpose either) is the occupancy-saturation
> story (flat-to-negative). colored-W4 vs *kpw-K=1* (which DOES pay transpose) is the no-transpose
> story (colored wins the end-to-end kernel on p10/p11 by avoiding 17–26 % relayout). **My
> resolution: against serial-wpw, colored-W4 is flat-to-negative at high env; against kpw-K=1 it can
> win the total on p10/p11 by carrying no transpose — but NOT via a faster solve kernel.** #1194
> compares colored vs serial-wpw, so I predict **flat-to-negative at high env**; #1195 will show the
> no-transpose structural advantage.

### #1195 — nsys p10 @ 16384, colored W4 path

1. **No `_tpw_k1_transpose_*` category** (colored is world-outer native). If a transpose category
   appears, the impl secretly relaid out and this prediction is void.
2. The MF-solve kernel ms lands **strictly between** kpw-ON (5892 ms) and serial-OFF (14831 ms) and
   **does NOT beat kpw's 2.52x** — because at 16384 worlds serial-wpw is already past the fill knee
   (43.6 waves), so colored adds ~0 occupancy headroom while paying `__syncthreads`×4warps×~13
   colors×2 iters barriers. **FALSIFIER:** colored-W4 solve **< 5892 ms** at 16384 ⇒ per-color
   block-sync is cheaper than modeled AND occupancy still mattered at fill ⇒ model wrong.
3. **Net total kernel ms** can still **beat kpw-ON on p10** (21674 ms) — not by a faster solve but by
   carrying no 21 % transpose tax. This is the one legitimate place colored-W4 beats kpw end-to-end.

### Other falsifiable predictions

- **P1.** Re-run ncu on the *streamed* warp-per-world kernel at N=256 ⇒ Block-Limit-Shared-Mem
  must rise from 2 toward ≥6 (the smem cap is lifted). If it stays at 2, streaming did not unlock
  occupancy on Blackwell ⇒ model wrong about the lever.
- **P2.** `K*(W)` is a non-increasing step function of waves(W). A measured `K*>1` at N≥16384 on any
  dense scene falsifies the fill law.
- **P3.** On warp-per-world, raising N past ~4096 does NOT raise SoL above the ~2-blk/SM ceiling
  (extra waves ≠ extra resident warps/SM). SoL stays ~flat vs N for the serial kernel.
- **P4.** A coalesced SoA repack (recovering the 22 % uncoalesced sectors, 18.1/32 B used) yields
  **< 1.2x** on the current 2-blk/SM kernel — it moves the L1TEX point but not the binding
  occupancy ceiling; it is 2nd-order until occupancy is fixed.
- **P5.** Dropping rt4 velocity-limit rows (~55.6 % of mf rows) beats W4mb4 at **every** N: it halves
  `R` (≈2x solve at fixed AI) AND lifts the 1.80x full-step Amdahl cap. A bigger lever than any
  parallel strategy.
- **P6.** Default iters=8 (vs appendix iters=2) **amortizes** the transpose: solve bytes ×4, transpose
  fixed/step ⇒ break-even shifts toward K=1 ⇒ the p11/h1 K=1 full-step losses **shrink** toward
  parity at iters=8.

---

## 5. The genuinely interesting place this lands

### 5a. The optimal threads-per-world is a *decreasing* schedule, not a constant

The intuition "more threads per world = faster" is **inverted** by the data. The measured law
(K8@4k → K4@8k → K1@16k, kpw saturated bench; K2/4/8 monotonically worse at all dense N, K-sweep)
is:

```
k*(W) = clamp( ceil( Q · resident_worlds_per_SM_baseline / W ), 1, 8 )
```

— threads-per-world should **fall** as the GPU fills, hitting **k=1 (thread-per-world)** at the env
counts that matter. A single adaptive kernel that picks `k` per launch would track this without a
human re-tuning per scene. **But** the naive k=1 path pays the per-step transpose (17–26 % of kernel
time) that actually sinks it on p11/h1. So the schedule alone is not enough — see 5c.

### 5b. The real levers are footprint and row-count, NOT parallelism

Ranked by model-expected payoff (the non-obvious headline):

| Lever | Mechanism | Expected payoff | Status |
|---|---|---|---|
| **smem streaming** (40324→15748 B) | lifts the 2-blk/SM occupancy cap ⇒ more resident worlds ⇒ pure Little's-law latency hiding; helps EVERY scene incl. dense | **1.6x** (measured isolated, sm_86); the single biggest single-lever win | measured sm_86; **needs sm_120 confirm (P1)** |
| **rt4 row drop** (~55.6 % of mf rows) | halves `R` ⇒ ~2x solve traffic at fixed AI AND lifts 1.80x full-step Amdahl cap | **~2x solve + Amdahl unlock** | (P5) — untested head-to-head |
| **k=1 at fill** (no extra threads/world) | max MLP-per-warp via 32 worlds/warp; lowest sync (0 barriers) | up to **3.2x isolated** @p10/16k | measured; **transpose-gated** |
| coalesced SoA repack | recover 22 % wasted sectors | **< 1.2x**, 2nd-order | (P4) — deferred |

The punchline: **the optimal kernel is the simplest one (serial-warp-per-world, or k=1
thread-per-world) made LIGHT enough (low smem+regs) to pack many worlds per SM, then fed enough
worlds to hide DRAM latency.** Threads-per-world should *decrease*, not increase, with fill.

### 5c. The non-obvious redesign the model implies (with payoff + the one experiment)

There are two candidate redesigns; the model ranks them.

**Candidate A — eliminate the k=1 transpose at its root (F2: produce mf6/J/Y world-INNERMOST at
geometry-build time).** The transpose, not the solve math, is the enemy. k=1's 3.2x kernel win is
real but eaten by a per-step relayout of the 92%-traffic MF/J data. If the geometry build emits
world-innermost layout once (paying the relayout where it is cheap / fused into a kernel that
already touches the data), k=1 keeps its 3.2x as **full-step net** on p11/h1 too. **Expected payoff:**
removes the 17–26 % transpose tax ⇒ flips p11 (−9 %→ ~+15 %) and h1 (−18.5 %→ ~+20–25 %) full-step.
**This is the highest-confidence, lowest-risk win** (byte-identical math, layout-only change).

**Candidate B — raise the arithmetic intensity to spend the 200x-idle compute pipe (Delassus
recompute instead of re-read).** This is the *radical* idea the roofline implies and the one I flag
as **higher-risk / more speculative**. The matrix-free re-read is ~92 % of DRAM traffic at AI~0.21;
the compute pipe is ~200x under-used. Recomputing `J·v` from a compact body-state cache (~24 B of
body state) instead of re-reading the materialized 112 B/contact blocks would raise AI ~4–5x, moving
the same work toward a regime where the abundant idle FLOPs hide the latency. **Expected payoff
(model estimate, NOT measured):** if AI rises 112 B → ~24 B re-read, solve DRAM traffic falls ~4.5x;
since the kernel is latency-bound this does not directly give 4.5x, but it shrinks the per-row
dependent-load count (the thing that sets the 0.13 ms/contact k=1 slope), plausibly **1.5–2.5x** on
the k=1 path. **Caveat:** recompute adds registers/smem (fights the occupancy lever in 5b) and the
Hinv·J^T term may not be cheaply recomputable; this could backfire. **Lower confidence than A.**

**The single experiment to run next (recommended):** **Candidate A — F2 world-innermost geometry
build — measured head-to-head against k=1-with-transpose on the full step at the p10/p11/h1 ladder
@16384, iters=2 and iters=8.** Rationale: it directly tests the model's central claim (the transpose,
not the solve, gates k=1 at high env), it is byte-identical so fidelity is free, and it is the only
lever that converts the *already-measured* 3.2x kernel win into full-step throughput on the dense
scenes that matter. The falsifiable outcome: F2 should flip p11/h1 from net-negative to net-positive
at iters=2 and widen p10's lead; if it does not, the transpose is not the gate and the model's
break-even rule (V10/V11) is wrong.

(Secondary experiment, if F2 lands: confirm P1 — ncu the streamed warp-per-world kernel at sm_120
to verify streaming lifts blk/SM from 2 toward ≥6 — since streaming is the one lever the model says
helps *every* scene and it is currently only proven on sm_86.)

---

## 6. Bottom line

- **One model, three regimes**, all four lenses reconciled: a single occupancy/latency ceiling pins
  the kernel ~95x below bandwidth; threads-per-world only moves `hide`, and worlds supply `hide`
  for free past W≈752.
- **Validated against all 14 measured points** (V1–V14), with V4 (the p10-vs-p11 serial-R-invariant
  / k1-density-linear split) as the strongest mechanistic confirmation. One PARTIAL (V9 magnitude),
  no FAILs; cross-arch/cross-env extrapolations honestly flagged.
- **#1194 prediction:** colored-W4 2.40x is a low-env artifact; full-step decays to flat-to-negative
  by W≥4096 on dense p11; crossover ~1024–2048. **#1195:** no transpose category; solve ms between
  5892 and 14831; does not beat kpw's 2.52x at 16384.
- **The interesting result:** optimal threads-per-world *decreases* with fill (→k=1), the real
  levers are footprint (streaming, 1.6x) and rt4-drop (~2x + Amdahl), and the highest-confidence
  redesign is **F2 (world-innermost geometry build to kill the k=1 transpose)** — the single
  experiment that would convert the measured 3.2x kernel win into full-step throughput on dense
  scenes.

*All numbers cited inline are from real artifacts under `artifacts/local/` (nsys-kernel-breakdown,
fpgs-k1-dense, fpgs-coloring, ncu-build-887, kpw-blackwell-bench). Builds #1194/#1195 had no
artifacts on disk at write time; their entries are model-derived predictions.*
