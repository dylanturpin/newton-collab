#!/usr/bin/env bash
# On-node orchestrator: ncu-profile the SERIAL FPGS contact-solve kernel
# (pgs_solve_mf_gs) at a SATURATED tiled world count on Blackwell.
# Runs INSIDE the slurm-broker Pyxis container (NCU image variant).
set -uo pipefail

echo "######## NODE: $(hostname) ########"
echo "== nvidia-smi =="
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv || nvidia-smi -L || true
echo "== ncu =="
which ncu && ncu --version | head -3 || { echo "NO ncu in image"; exit 3; }
PY="$(command -v python || command -v python3)"
echo "python = $PY"

: "${TPW_REPO:?TPW_REPO must point to the cloned il-newton-dev checkout}"
echo "TPW_REPO=$TPW_REPO"
BDIR="$TPW_REPO/tpw_blackwell"
ls -la "$BDIR" || { echo "NO $BDIR"; exit 2; }

# Decompress the capture on-node (savez_compressed -> usable npz; numpy reads
# compressed npz directly, so we just point at it).
export TPW_CAP_NPZ="${TPW_CAP_NPZ:-$BDIR/cap892_c.npz}"
export TPW_CAP_JSON="${TPW_CAP_JSON:-$BDIR/cap892.json}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-/tmp/warp-cache-$$}"
mkdir -p "$WARP_CACHE_PATH"
echo "TPW_CAP_NPZ=$TPW_CAP_NPZ ($(du -h "$TPW_CAP_NPZ" | cut -f1))"

# Profile config: saturated world count + warmup so the profiled launch is warm.
export NCU_N="${NCU_N:-8192}"
export NCU_NWARM="${NCU_NWARM:-6}"
export NCU_NPROF="${NCU_NPROF:-2}"
# ncu --launch-skip skips the warmup launches; capture exactly 1 warm steady launch.
SKIP="${NCU_SKIP:-$NCU_NWARM}"

OUT="$BDIR/prof_blackwell"
DETAILS="$BDIR/details.txt"
RAW="$BDIR/raw.txt"

echo
echo "######## STEP 0: WARP-CACHE PREWARM (compile serial kernel, no ncu) ########"
# Compile the serial kernel once outside ncu so the very first ncu-seen launch
# isn't a JIT-compile event. We run with NWARM=2 NPROF=0-ish just to populate cache.
NCU_N="$NCU_N" NCU_NWARM=2 NCU_NPROF=0 "$PY" "$BDIR/ncu_serial_replay.py" || { echo "prewarm failed"; exit 4; }

echo
echo "######## STEP 1: NCU PROFILE (set=basic, kernel=pgs_solve_mf_gs, 1 warm launch) ########"
echo "ncu N=$NCU_N NWARM=$NCU_NWARM launch-skip=$SKIP launch-count=1"
set -x
ncu --set basic \
    --kernel-name "regex:pgs_solve_mf_gs" \
    --launch-count 1 \
    --launch-skip "$SKIP" \
    --target-processes all \
    --force-overwrite \
    -o "$OUT" \
    "$PY" "$BDIR/ncu_serial_replay.py"
NCU_RC=$?
set +x
echo "ncu rc=$NCU_RC"
ls -la "$OUT".ncu-rep 2>/dev/null || echo "WARN: no .ncu-rep produced"

echo
echo "######## STEP 2: EXPORT HUMAN-READABLE SUMMARY ########"
if [ -f "$OUT".ncu-rep ]; then
  echo "---- ncu --page details ----"
  ncu --import "$OUT".ncu-rep --page details | tee "$DETAILS"
  echo
  echo "---- ncu --page raw (key metrics grep) ----"
  ncu --import "$OUT".ncu-rep --page raw > "$RAW" 2>/dev/null || true
  echo "(raw page written to $RAW; $(wc -l < "$RAW" 2>/dev/null || echo 0) lines)"
fi

echo
echo "######## STEP 3: COPY ARTIFACTS TO RUN_DIR (/workspace/slurm-run) ########"
RUN_DIR="${RUN_DIR:-/workspace/slurm-run}"
if [ -d "$RUN_DIR" ]; then
  cp -v "$OUT".ncu-rep "$RUN_DIR/" 2>/dev/null || echo "no ncu-rep to copy"
  cp -v "$DETAILS" "$RUN_DIR/" 2>/dev/null || echo "no details.txt to copy"
  cp -v "$RAW" "$RUN_DIR/" 2>/dev/null || echo "no raw.txt to copy"
fi

echo
echo "######## ALL DONE  ncu_rc=$NCU_RC ########"
exit "$NCU_RC"
