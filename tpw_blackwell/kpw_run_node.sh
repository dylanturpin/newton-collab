#!/usr/bin/env bash
# On-node orchestrator: CUDA-event-time the kpw colored MF-GS solve vs the
# serial warp-per-world FPGS contact-solve, tiled to saturated world counts on
# a single Blackwell GPU (sm_120). No ncu -- plain python timing.
# Runs INSIDE the slurm-broker Pyxis container.
set -uo pipefail

echo "######## NODE: $(hostname) ########"
echo "== nvidia-smi =="
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv || nvidia-smi -L || true
echo "== gpu count (single-GPU check) =="
nvidia-smi -L | wc -l
PY="$(command -v python || command -v python3)"
echo "python = $PY"

: "${TPW_REPO:?TPW_REPO must point to the cloned newton-collab checkout}"
echo "TPW_REPO=$TPW_REPO"
BDIR="$TPW_REPO/tpw_blackwell"
ls -la "$BDIR" || { echo "NO $BDIR"; exit 2; }

export TPW_CAP_NPZ="${TPW_CAP_NPZ:-$BDIR/cap892_c.npz}"
export TPW_CAP_JSON="${TPW_CAP_JSON:-$BDIR/cap892.json}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-/tmp/warp-cache-$$}"
mkdir -p "$WARP_CACHE_PATH"
echo "TPW_CAP_NPZ=$TPW_CAP_NPZ ($(du -h "$TPW_CAP_NPZ" 2>/dev/null | cut -f1))"

export KPW_NS="${KPW_NS:-4096,8192,16384}"
export KPW_KS="${KPW_KS:-1,4,8}"
export KPW_REPS="${KPW_REPS:-20}"
echo "KPW_NS=$KPW_NS KPW_KS=$KPW_KS KPW_REPS=$KPW_REPS"

echo
echo "######## STEP 0: SMOKE (N=4096 only, K=1,8) -- prove launch + int64 fix ########"
KPW_NS="4096" KPW_KS="1,8" KPW_REPS="3" "$PY" "$BDIR/kpw_node_bench.py" 2>&1 | tail -40
SMOKE_RC=${PIPESTATUS[0]}
echo "smoke rc=$SMOKE_RC"
if [ "$SMOKE_RC" -ne 0 ]; then echo "SMOKE FAILED -- aborting before full sweep"; exit 5; fi

echo
echo "######## STEP 1: INT64 FIX PROOF -- N=16384 must launch without CUDA-700 ########"
KPW_NS="16384" KPW_KS="8" KPW_REPS="3" "$PY" "$BDIR/kpw_node_bench.py" 2>&1 | tail -30
FIX_RC=${PIPESTATUS[0]}
echo "int64-fix probe rc=$FIX_RC"

echo
echo "######## STEP 2: FULL SWEEP (N in $KPW_NS, K in $KPW_KS) ########"
RUN_DIR="${RUN_DIR:-/workspace/slurm-run}" "$PY" "$BDIR/kpw_node_bench.py"
BENCH_RC=$?
echo "bench rc=$BENCH_RC"

echo
echo "######## STEP 3: COPY ARTIFACTS TO RUN_DIR ########"
RUN_DIR="${RUN_DIR:-/workspace/slurm-run}"
if [ -d "$RUN_DIR" ]; then
  cp -v "$BDIR/kpw_bench_result.json" "$RUN_DIR/" 2>/dev/null || echo "(result json already in RUN_DIR or absent)"
fi

echo
echo "######## ALL DONE  bench_rc=$BENCH_RC ########"
exit "$BENCH_RC"
