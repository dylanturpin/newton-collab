# F2 deferred: `_tpw_k1_transpose_mf6_to_inner` elimination

**Decision:** F2 is NOT implemented. The validated design returned
`recommend_implement: false`. F1c (the sibling JY-transpose idle-launch
elimination, ~9.2% of K=1 kernel time) ships; F2 (~8.4%) is deferred.

`FEATHER_PGS_TPW_K1_F2` is still read at solver init and forwarded by CI, but
enabling it raises `ValueError` (not implemented).

## Why the clean "redirect the producer output" version is impossible

The goal sketched "have the mf6 producers write the K=1 world-inner flats
directly instead of going through the transpose." That is not safe because the
world-outer mf6 buffers are shared and the producer chain is coupled:

1. **Blast radius.** `self.mf_J_a/_b` and `self.mf_MiJt_a/_b` (world-outer
   `(W, M_MF, 6)`) are consumed by 6+ NON-K1 paths: the colored/serial MF-GS
   solve, the standalone MF PGS kernel, the CPU loop fallback, the two
   position-solve passes, a debug copy, and a `.numpy()` dump that hard-assumes
   the `(W, M_MF, 6)` shape. None are gated by `FEATHER_PGS_TPW_K1`. A producer
   writing world-inner *instead of* world-outer silently corrupts the default
   product path.

2. **Producer #3 is also a consumer.** `compute_mf_effective_mass_and_rhs`
   READS world-outer `mf_J_a/_b` and WRITES world-outer `mf_MiJt_a/_b` in the
   same kernel. So even within K=1, `build_mf_contact_rows` must still leave a
   world-outer J for `compute` to read. The three producers
   (`build_mf_contact_rows` -> `populate_rigid_velocity_limit_rows` ->
   `compute_mf_effective_mass_and_rhs`) are a coupled chain; F2 cannot touch one
   cleanly.

## The only byte-identical-default option is DUAL-WRITE, and its payoff is poor

The sole safe F2 is a dual-write: producers keep every world-outer store AND
additionally write the world-inner flats behind a compile-constant `int` flag
(literal `0` on the default path so Warp/ptxas dead-code-eliminates the extra
store block -> default PTX unchanged). It only removes the 4 mf6 transpose
*launches*; it does NOT remove the world-outer stores. It ADDS a second
(strided, possibly worse-coalesced) store of the same `4 * 6 * live` floats
inside the producers plus a duplicated MiJt store in `compute`. Realistic upside
is well under the 8.4% nsys headline and could regress if the producer flat
store is worse-coalesced than the dedicated transpose. Unmeasured.

It also forces signature changes on 3 shared `@wp.kernel`s, touching ALL launch
sites (K1 and non-K1); a missed dummy-arg anywhere is a launch error or a silent
wrong-array bind. And correctness hinges on the producer write-set EXACTLY
covering the kpw read-set (friction-row break, vel-limit unallocated slots) —
any slot the transpose copied but a producer early-returns past reads stale K1
scratch, invisible to penetration-only metrics. That requires a NaN-poison gate
before it can be trusted.

## Contrast with F1c (shipped)

F1c is a pure launch-gating change in `_build_tpw_k1_inputs` plus one new
`enable_backward=False` kernel: zero `kernels.py` edits, zero non-K1 blast
radius, bigger headline (~9.2% > 8.4%), and verified byte-identical on the real
captured `cap892` J_world (max|current - f1c| = 0.000e+00, 512x fewer threads).

## If F2 is revived

Do the safest partial-F2 first: dual-write ONLY `build_mf_contact_rows` (the
dominant mf6 producer), keep the transpose for `mf_MiJt` + vel-limit rows
(skip only the `mf_J_a/_b` transpose pair), leave `compute_mf_effective_mass`
untouched. Gate it, measure it separately on the p10 nsys, and keep it only if
it beats F1c-only.
