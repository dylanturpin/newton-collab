#!/usr/bin/env bash
# On-node orchestrator for the Blackwell tpw-vs-warp FPGS bench.
# Runs INSIDE the slurm-broker Pyxis container. Clones the pushed branch (done by
# the broker payload before calling this, OR clones here if TPW_REPO unset), then:
#   1. prints GPU identity (nvidia-smi)
#   2. runs the standalone tpw-vs-warp solve-kernel bench (the CORRECT tpw path)
#   3. runs the ptxas occupancy probe for warp AND tpw on this arch
set -uo pipefail

echo "######## NODE: $(hostname) ########"
echo "== nvidia-smi =="
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv || nvidia-smi -L || true
echo "== nvcc =="
which nvcc && nvcc --version | tail -2 || echo "no nvcc"
echo "== python =="
which python || which python3

PY="$(command -v python || command -v python3)"

# Find the repo: either TPW_REPO is set (broker clone) or we are launched from within it.
: "${TPW_REPO:?TPW_REPO must point to the cloned il-newton-dev checkout}"
echo "TPW_REPO=$TPW_REPO"
ls -la "$TPW_REPO/newton" | head -3 || { echo "NO newton/ in TPW_REPO"; exit 2; }

BDIR="$TPW_REPO/tpw_blackwell"
export TPW_CAP_NPZ="${TPW_CAP_NPZ:-$BDIR/cap892_c.npz}"
export TPW_CAP_JSON="${TPW_CAP_JSON:-$BDIR/cap892.json}"
export TPW_NS="${TPW_NS:-256,1024,4096,8192,16384}"
export TPW_BD="${TPW_BD:-256}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-/tmp/warp-cache-$$}"
mkdir -p "$WARP_CACHE_PATH"

echo "TPW_CAP_NPZ=$TPW_CAP_NPZ ($(du -h "$TPW_CAP_NPZ" | cut -f1))"
echo "TPW_NS=$TPW_NS  TPW_BD=$TPW_BD  WARP_CACHE_PATH=$WARP_CACHE_PATH"

echo
echo "######## PART 1: STANDALONE TPW-VS-WARP SOLVE-KERNEL BENCH ########"
"$PY" "$BDIR/tpw_node_bench.py"
BENCH_RC=$?
echo "bench rc=$BENCH_RC"

echo
echo "######## PART 2: PTXAS OCCUPANCY (warp + tpw, this arch) ########"
"$PY" "$BDIR/tpw_ptxas_occ.py"
PTX_RC=$?
echo "ptxas rc=$PTX_RC"

echo
echo "######## ALL DONE  bench_rc=$BENCH_RC ptxas_rc=$PTX_RC ########"
exit $(( BENCH_RC | PTX_RC ))
