# Trajectory IK: design space, prototype, and Warp assessment

*Dylan Turpin, July 2026 — companion note for the `dylanturpin/trajectory-ik` branch.
Prototype: `newton.ik.IKSolverTrajectory` + temporal objectives; demo `ik_trajectory`; tests `test_ik_trajectory`.*

## The problem and the design space

Trajectory IK solves all frames of a joint trajectory jointly:
min over q₁..q_T of per-frame task costs (EE pose, limits) plus temporal costs
(velocity/accel/jerk smoothness, velocity limits, anchors). The Gauss-Newton
Hessian of this problem is **block-banded** — block-tridiagonal for
first-difference costs, penta/hepta-diagonal for accel/jerk (bandwidth
`2(k+1)n−1` for k-order costs; Toussaint's KOMO tutorial proves the banded
Newton step is O(T·k²n³), i.e. linear in T, and identical in structure to
Riccati/DDP and factor-graph message passing). Everyone in the space sits at
one of four points on how they solve that system:

| Approach | Who | Linear algebra | Parallel over T? | Batch? |
|---|---|---|---|---|
| **Direct banded / Riccati** | KOMO, GPMP2 (GTSAM sparse Cholesky), Crocoddyl/Aligator, OCS2 | banded/block-tridiag Cholesky, exact | no (sequential sweep) | rare (CPU) |
| **Iterative CG on normal eqs** | pyroki/jaxls (default: CG + block-Jacobi + Eisenstat–Walker), MPCGPU (PCG + "symmetric stair" preconditioner, 3.6–10× vs CPU direct) | matrix-free or BSR SpMV | **yes** | yes |
| **Quasi-Newton, no linear solve** | cuRobo (L-BFGS on B-spline knots, fused cost+grad kernels, 4-candidate parallel line search, MPPI warmup, CUDA graphs) | two-loop recursion only | yes | **excellent** |
| **Sampling** | STOMP, MPPI, MJPC predictive sampling | none | rollouts parallel | excellent |

Notable details from the two closest references:

- **pyroki/jaxls**: trajectory = one batched variable (`JointVar(arange(T))`);
  stencil costs are pure index arithmetic; LM with frozen column scaling and a
  cancellation-safe predicted-reduction formula; hard constraints via augmented
  Lagrangian (too slow for their online demo, which falls back to soft weights).
  **jaxls has no banded direct solver** — its only sparse direct option is
  CHOLMOD via a CPU host-callback. The block-banded structure of trajectories
  is exactly the case its own docs leave on the table.
- **cuRobo**: never forms a Jacobian or Hessian. Temporal structure lives in the
  B-spline transition kernel (16 knots × 7 dof = 112 variables — small enough
  that sparse vs dense stops mattering) and dt-retimed cost weights; time
  optimality via an outer "compress dt by 0.55 and re-solve" loop. Strength:
  graph-capture-friendly fixed control flow and seed multiplicity. Weakness:
  linear-ish convergence, 100 fixed iterations, 1e6 penalty weights, no
  constraint semantics.
- Parallel-in-time direct methods exist (Särkkä's associative-scan LQR,
  Aligator's block-elimination parallel LQR, cyclic reduction) but are only
  worth it for single very-long trajectories; batch parallelism is the cheaper
  GPU win (VAMP, cuRobo, GATO all agree).

## What the prototype does

`IKSolverTrajectory` keeps Newton's existing per-frame machinery intact and
adds the trajectory layer:

- **Rows = (trajectory, frame).** Existing per-frame objectives
  (`IKObjectivePosition/Rotation/JointLimit`), both FK paths, all three
  Jacobian modes (autodiff/analytic/mixed), and the tangent-space retraction
  (`jcalc_integrate`) are reused **unchanged** — targets are simply sized
  `n_problems × n_frames`.
- **Temporal objectives** (`IKObjectiveSmoothness` k=1/2/3,
  `IKObjectiveVelocityLimit`, `IKObjectiveJointReference`) report bounded
  per-frame-offset coefficient blocks instead of a global Jacobian; their
  Gauss-Newton contribution is accumulated analytically into a banded store.
  Most joints populate only the diagonal; free-joint linear rows carry the
  exact `[p]×` lever-arm coupling to the angular tangent coordinates
  (Newton's free-joint tangent is a world-origin spatial velocity, so the
  retraction moves `p` by `δ_lin + δ_ang×p` — dropping that coupling made
  floating-base solves degrade measurably even a couple of meters from the
  origin, caught by adversarial review; the temporal gradient is now
  finite-difference-verified at 58 m offset). Ball/free rotations use
  log-map differences consistent with the solver's retraction. Residual
  values are origin-invariant; very far from the origin (tens of meters)
  convergence still slows because the tangent convention pivots base
  rotations about the world origin — an upstream conditioning wart shared by
  all of `newton.ik`, worth revisiting via body-centered free-joint tangents.
- **Two linear backends**, selected per solve, agreeing to ~1e-6:
  - `direct` — batched block-tridiagonal Cholesky (block Thomas): one CUDA
    block per trajectory, sequential in-kernel loop over frames using
    `wp.tile_cholesky`/`tile_lower_solve`/`tile_matmul`. Higher-order stencils
    are reblocked into k·n superblocks, so accel/jerk costs use the same
    tridiagonal kernel. Frames pinned via `fixed_frames` are eliminated
    exactly (identity row/col).
  - `cg` — BSR block-banded Hessian (`warp.sparse`, fixed topology, values
    rewritten in place) + `warp.optim.linear.cg` with per-trajectory
    `batch_offsets` and a custom **block-Jacobi** preconditioner
    (per-frame n×n diagonal blocks inverted with `tile_cholesky_solve`),
    warm-started across LM iterations.
- **LM globalization per trajectory**: one damping λ and one accept/reject
  decision per trajectory (all frames atomically), fixed iteration counts, no
  host syncs — the whole `step()` is CUDA-graph capturable (verified for both
  backends, including CG's conditional-graph loop).

**Results** (RTX PRO 4500, Franka fr3, 9 dof, T=120, 16 LM iterations,
position + limits + velocity + accel smoothness):

| method | wall time | max \|q̈\| | mean \|q⃛\| |
|---|---|---|---|
| trajectory IK, direct, 4 trajectories | **16.2 ms** (8.6 ms graph replay) | 4.5 | 2.9 |
| trajectory IK, CG, 4 trajectories | 46.1 ms (20.9 ms replay) | 4.5 | 2.9 |
| frame-by-frame `IKSolver`, 1 trajectory, warm-started | 720 ms | 16.3 | 7.6 |

Direct wins at these scales, as predicted (sequential-in-T cost hidden by
batch parallelism; CG's iteration count pays the conditioning tax). The gap to
frame-by-frame is structural: sequential solves can't batch over frames, and
greedy per-frame solutions are 3–4× rougher at equal iteration budgets.

## Warp: what worked, what's missing, opportunities

**Out of the box (pinned 1.15.0.dev):** `tile_cholesky`/`tile_lower_solve`/
`tile_matmul` compose into a batched banded factorization inside a single
kernel with a runtime-length sequential loop — this "just worked", including
CUDA graph capture (~0.9 ms for a batch of 256 T=240 n=8 tridiagonal solves).
BSR construction (`bsr_from_triplets`, in-place value rewrites via
kernel-side `bsr_block_index`), batched CG (`batch_offsets`), and the
capturable `check_every=0` loop all did what the docs promise. No fork of
warp was needed to build trajik.

**Gaps & bugs hit:**
1. `warp.optim.linear.preconditioner(A, "diag")` is *point*-Jacobi even for
   matrix-block BSR — no block-Jacobi. We hand-roll one (~60 lines). jaxls
   defaults to block-Jacobi for good reason; MPCGPU's stair preconditioner
   would be better still.
2. **Bug (pinned version, fixed on warp main):** the iterative solvers'
   internal reduction launches don't pass `device=`, so a CPU solve on a
   machine whose default device is CUDA returns NaNs. Worked around with
   `wp.ScopedDevice` at solver construction and call time.
3. No direct sparse factorization of any kind in warp (confirmed) — the tile
   API is the only route, which is fine for banded but rules out general
   sparsity patterns.
4. Tile shapes are compile-time: one kernel specialization per
   (n_dofs, n_residuals, bandwidth); first-use JIT of the mathdx tile kernels
   is 30–60 s per combination. Fine for solvers, annoying for iteration.
5. Shared-memory bounds the superblock size (k·n ≲ 64–96 fp32) — jerk-order
   smoothing on high-dof humanoids would need the banded (non-reblocked)
   variant or the CG path.

**Small warp extensions worth proposing** (each general-purpose, in rough
order of value): `preconditioner(A, "block_diag")` for BSR;
`block_tridiag_cholesky(D, L, b, x)` on `(B,T,n,n)` arrays — the canonical
trajopt/Kalman/spline kernel, which this branch effectively contains;
`bsr_diag_add(A, λ)` for LM damping; a deterministic `bsr_mv(transpose=True)`;
per-batch early exit in the batched iterative solvers. The device= bug fix is
already on main.

## Where this goes next

- **Constraint phases on top** (Justin's task-authoring layer): the natural
  compile target is per-frame objective target arrays + `frame_weights` +
  `fixed_frames`, which already vary per frame. Missing pieces for hard
  constraint semantics: an augmented-Lagrangian outer loop (jaxls-style
  `sqrt(ρ)·relu(g+λ/ρ)` folds into the existing residual machinery cleanly).
- **Collision costs** as per-frame objectives (sphere-approximated, cuRobo
  style) — slots into the existing `IKObjective` contract; swept (t,t+1)
  variants would be the first *dense* off-diagonal coupling, which the band
  store already supports structurally.
- **Retargeting at scale** (soma-retargeter): the batch axis is already there;
  T≈500+ humanoid clips would motivate the banded (non-superblock) kernel and
  a stair/SSOR preconditioner for the CG path.
- **Task-space temporal costs** (EE-velocity smoothing) need dense off-diagonal
  blocks — the assembly supports it; only the temporal-objective contract
  would grow a `compute_jacobian_band` variant.
- **Time parametrization**: keep it decoupled (TOPP-RA/ruckig style retiming
  after the geometric solve), per the field's consensus.
