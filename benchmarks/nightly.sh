#!/bin/bash

# Nightly benchmark runner for Newton.
#
# This script:
#   1. Clones your fork and rebases a source branch on top of upstream.
#   2. Runs solver_benchmark sweeps + ablations for g1_flat and h1_tabletop.
#   3. Appends results to JSONL files in the gh-pages branch.
#   4. Updates a static Chart.js dashboard under nightly/index.html.
#
# Required environment variables:
#   REPO_URL     SSH URL for your fork (e.g., git@github.com:yourname/newton.git)
#   UPSTREAM_URL SSH URL for upstream (e.g., git@github.com:Newton-Physics/newton.git)
#
# Optional environment variables:
#   SOURCE_BRANCH                   Source branch to rebase (default: feather_pgs)
#   AUTO_BRANCH                     Target branch to push (default: feather_pgs_auto_rebase)
#   UPSTREAM_REF                    Upstream ref to rebase onto (default: upstream/main)
#   RESULTS_BRANCH                  Results branch (default: gh-pages)
#   SWEEP_SOLVERS_G1_FLAT           Solver list for g1_flat sweeps (default: fpgs_tiled,mujoco)
#   SWEEP_SOLVERS_H1_TABLETOP       Solver list for h1_tabletop sweeps (default: fpgs_mf,mujoco)
#   SWEEP_MIN_LOG2_G1_FLAT          Min log2 worlds for g1_flat sweep (default: 10)
#   SWEEP_MAX_LOG2_G1_FLAT          Max log2 worlds for g1_flat sweep (default: 15)
#   SWEEP_MIN_LOG2_H1_TABLETOP      Min log2 worlds for h1_tabletop sweep (default: 10)
#   SWEEP_MAX_LOG2_H1_TABLETOP      Max log2 worlds for h1_tabletop sweep (default: 11)
#   ABLATION_WORLDS_G1_FLAT          Num worlds for g1_flat ablation (default: 8192)
#   ABLATION_WORLDS_H1_TABLETOP      Num worlds for h1_tabletop ablation (default: 2048)
#   RUN_G1_FLAT                     Set to 0 to skip g1_flat sweeps/ablations (default: 1)
#   RUN_H1_TABLETOP                 Set to 0 to skip h1_tabletop sweeps/ablations (default: 1)
#   SKIP_ABLATIONS                  Set to 1 to skip ablations (default: 0)
#   ENABLE_PLOTS                    Set to 0 to skip matplotlib plots (default: 1)
#   CHERRY_PICK_BRANCHES            Comma-separated branches to cherry-pick after rebase (default: "")
#   UV_EXTRAS                       Extras for uv run (default: "--extra examples --extra torch-cu12")

set -euo pipefail

REPO_URL="${REPO_URL:-}"
UPSTREAM_URL="${UPSTREAM_URL:-}"
SOURCE_BRANCH="${SOURCE_BRANCH:-feather_pgs}"
AUTO_BRANCH="${AUTO_BRANCH:-feather_pgs_auto_rebase}"
UPSTREAM_REF="${UPSTREAM_REF:-upstream/main}"
RESULTS_BRANCH="${RESULTS_BRANCH:-gh-pages}"
SWEEP_SOLVERS_G1_FLAT="${SWEEP_SOLVERS_G1_FLAT:-fpgs_tiled,mujoco}"
SWEEP_SOLVERS_H1_TABLETOP="${SWEEP_SOLVERS_H1_TABLETOP:-fpgs_mf,mujoco}"
SWEEP_MIN_LOG2_G1_FLAT="${SWEEP_MIN_LOG2_G1_FLAT:-10}"
SWEEP_MAX_LOG2_G1_FLAT="${SWEEP_MAX_LOG2_G1_FLAT:-17}"
SWEEP_MIN_LOG2_H1_TABLETOP="${SWEEP_MIN_LOG2_H1_TABLETOP:-10}"
SWEEP_MAX_LOG2_H1_TABLETOP="${SWEEP_MAX_LOG2_H1_TABLETOP:-15}"
ABLATION_WORLDS_G1_FLAT="${ABLATION_WORLDS_G1_FLAT:-16384}"
ABLATION_WORLDS_H1_TABLETOP="${ABLATION_WORLDS_H1_TABLETOP:-8192}"
ENABLE_PLOTS="${ENABLE_PLOTS:-1}"
UV_EXTRAS="${UV_EXTRAS:---extra examples --extra torch-cu12}"
RUN_G1_FLAT="${RUN_G1_FLAT:-1}"
RUN_H1_TABLETOP="${RUN_H1_TABLETOP:-1}"
SKIP_ABLATIONS="${SKIP_ABLATIONS:-0}"
CHERRY_PICK_BRANCHES="${CHERRY_PICK_BRANCHES:-}"

export AUTO_BRANCH

if [[ -z "$REPO_URL" || -z "$UPSTREAM_URL" ]]; then
  echo "ERROR: Set REPO_URL and UPSTREAM_URL before running." >&2
  exit 1
fi

RUN_ID="$(date -u '+%Y-%m-%dT%H-%M-%SZ')"
WORK_DIR="/tmp/newton-nightly-$$"
RESULTS_DIR="/tmp/newton-results-$$"
RESULTS_DIR_PATH="$RESULTS_DIR/nightly"

export UV_CACHE_DIR="/tmp/uv-cache-nightly-$$"

log() {
  ( [ -n "${1:-}" ] && echo "$@" || cat ) | while read -r line; do
    printf "[%(%Y-%m-%d %H:%M:%S)T] %s\n" -1 "$line"
  done
}

cleanup() {
  [[ -d "$WORK_DIR" ]] && rm -rf "$WORK_DIR"
  [[ -d "$RESULTS_DIR" ]] && rm -rf "$RESULTS_DIR"
}
trap cleanup EXIT

log "Cloning fork: $REPO_URL"
git clone "$REPO_URL" "$WORK_DIR"
cd "$WORK_DIR"

log "Configuring upstream: $UPSTREAM_URL"
git remote add upstream "$UPSTREAM_URL"
git fetch upstream
log "Fetching origin branches"
git fetch origin

if ! git show-ref --verify --quiet "refs/remotes/origin/$SOURCE_BRANCH"; then
  echo "ERROR: origin/$SOURCE_BRANCH not found" >&2
  exit 1
fi

log "Rebasing $SOURCE_BRANCH onto $UPSTREAM_REF"
git checkout -B "$AUTO_BRANCH" "origin/$SOURCE_BRANCH"
git rebase "$UPSTREAM_REF"

if [[ -n "$CHERRY_PICK_BRANCHES" ]]; then
  IFS=',' read -ra CP_BRANCHES <<< "$CHERRY_PICK_BRANCHES"
  for branch in "${CP_BRANCHES[@]}"; do
    branch="$(echo "$branch" | xargs)"  # trim whitespace
    log "Cherry-picking origin/$branch"
    git fetch origin "$branch"
    git cherry-pick "origin/$branch"
  done
fi

log "Pushing $AUTO_BRANCH"
git push origin "$AUTO_BRANCH" --force-with-lease

RUN_COMMIT="$(git rev-parse HEAD)"
RUN_COMMIT_SHORT="$(git rev-parse --short HEAD)"
RUN_TIMESTAMP="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

log "Cloning results branch: $RESULTS_BRANCH"
git clone --branch "$RESULTS_BRANCH" --depth 1 "$REPO_URL" "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR_PATH/runs/$RUN_ID"

# Ensure dashboard assets are present
cp "$WORK_DIR/benchmarks/nightly/index.html" "$RESULTS_DIR_PATH/index.html"

touch "$RESULTS_DIR/.nojekyll"

# Ensure JSONL files exist
mkdir -p "$RESULTS_DIR_PATH"
[[ -f "$RESULTS_DIR_PATH/runs.jsonl" ]] || touch "$RESULTS_DIR_PATH/runs.jsonl"
[[ -f "$RESULTS_DIR_PATH/points.jsonl" ]] || touch "$RESULTS_DIR_PATH/points.jsonl"

RUN_DIR="$RESULTS_DIR_PATH/runs/$RUN_ID"

run_sweep() {
  local scenario="$1"
  local min_log2="$2"
  local max_log2="$3"
  local solvers="$4"
  local out_dir="$RUN_DIR/${scenario}_sweep"

  local cmd=(uv run $UV_EXTRAS -m newton.tools.solver_benchmark \
    --scenario "$scenario" \
    --sweep \
    --solvers "$solvers" \
    --min-log2-worlds "$min_log2" \
    --max-log2-worlds "$max_log2" \
    --out "$out_dir")

  if [[ "$ENABLE_PLOTS" == "1" ]]; then
    cmd+=(--plot)
  fi

  log "Running sweep: $scenario"
  UV_NO_CONFIG=1 PYTHONUNBUFFERED=1 "${cmd[@]}" 2>&1 | log
}

run_ablation() {
  local scenario="$1"
  local worlds="$2"
  local out_dir="$RUN_DIR/${scenario}_ablation"

  local cmd=(uv run $UV_EXTRAS -m newton.tools.solver_benchmark \
    --scenario "$scenario" \
    --ablation \
    --num-worlds "$worlds" \
    --out "$out_dir")

  if [[ "$ENABLE_PLOTS" == "1" ]]; then
    cmd+=(--plot)
  fi

  log "Running ablation: $scenario"
  UV_NO_CONFIG=1 PYTHONUNBUFFERED=1 "${cmd[@]}" 2>&1 | log
}

run_render() {
  local scenario="$1"
  local solver="$2"
  local substeps="$3"
  local tag="${solver}_${substeps}sub"
  local out_dir="$RUN_DIR/${scenario}_render/$tag"

  log "Rendering video: $scenario with $solver ($substeps substeps)"
  UV_NO_CONFIG=1 PYTHONUNBUFFERED=1 uv run $UV_EXTRAS -m newton.tools.solver_benchmark \
    --scenario "$scenario" \
    --solver "$solver" \
    --substeps "$substeps" \
    --render \
    --render-frames 300 \
    --out "$out_dir" 2>&1 | log
}

build_renders_manifest() {
  # Collect all render_meta.json files into a single renders.json for this run
  python - "$RUN_DIR" <<'PY'
import json, sys
from pathlib import Path

run_dir = Path(sys.argv[1])
renders = []
for meta_file in sorted(run_dir.rglob("render_meta.json")):
    meta = json.loads(meta_file.read_text())
    # Store path relative to run dir
    video_rel = str(meta_file.parent / meta["video"])
    video_rel = str(Path(video_rel).relative_to(run_dir))
    meta["path"] = video_rel
    renders.append(meta)

manifest = run_dir / "renders.json"
manifest.write_text(json.dumps(renders, indent=2))
print(f"Wrote {len(renders)} render entries to {manifest}")
PY
}

append_points() {
  local mode="$1"
  local scenario="$2"
  local data_file="$3"
  local meta_file="$4"

  RUN_ID="$RUN_ID" RUN_TIMESTAMP="$RUN_TIMESTAMP" RUN_COMMIT="$RUN_COMMIT" RUN_COMMIT_SHORT="$RUN_COMMIT_SHORT" \
  RUN_MODE="$mode" RUN_SCENARIO="$scenario" DATA_FILE="$data_file" META_FILE="$meta_file" \
  python - <<'PY' >> "$RESULTS_DIR_PATH/points.jsonl"
import json
import os

run_id = os.environ['RUN_ID']
run_timestamp = os.environ['RUN_TIMESTAMP']
commit = os.environ['RUN_COMMIT']
commit_short = os.environ['RUN_COMMIT_SHORT']
mode = os.environ['RUN_MODE']
scenario = os.environ['RUN_SCENARIO']

with open(os.environ['META_FILE']) as f:
    meta = json.load(f)

with open(os.environ['DATA_FILE']) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        out = {
            **row,
            "run_id": run_id,
            "run_timestamp": run_timestamp,
            "commit": commit,
            "commit_short": commit_short,
            "mode": mode,
            "scenario": scenario,
            "gpu": meta.get("gpu"),
            "gpu_memory_total_gb": meta.get("gpu_memory_total_gb"),
            "platform": meta.get("platform"),
            "python_version": meta.get("python_version"),
            "pgs_iterations": meta.get("pgs_iterations"),
            "measure_frames": meta.get("measure_frames"),
            "warmup_frames": meta.get("warmup_frames"),
        }
        out.setdefault("substeps", row.get("substeps", meta.get("substeps")))
        print(json.dumps(out))
PY
}

log "Benchmark run id: $RUN_ID"

if [[ "$RUN_G1_FLAT" == "1" ]]; then
  run_sweep "g1_flat" "$SWEEP_MIN_LOG2_G1_FLAT" "$SWEEP_MAX_LOG2_G1_FLAT" "$SWEEP_SOLVERS_G1_FLAT"
  append_points "sweep" "g1_flat" "$RUN_DIR/g1_flat_sweep/sweep.jsonl" "$RUN_DIR/g1_flat_sweep/metadata.json"

  if [[ "$SKIP_ABLATIONS" != "1" ]]; then
    run_ablation "g1_flat" "$ABLATION_WORLDS_G1_FLAT"
    append_points "ablation" "g1_flat" "$RUN_DIR/g1_flat_ablation/ablation.jsonl" "$RUN_DIR/g1_flat_ablation/metadata.json"
  fi

  IFS=',' read -ra G1_SOLVERS <<< "$SWEEP_SOLVERS_G1_FLAT"
  G1_SUBSTEPS="${SCENARIOS_SUBSTEPS_G1_FLAT:-2}"
  for solver in "${G1_SOLVERS[@]}"; do
    run_render "g1_flat" "$solver" "$G1_SUBSTEPS"
  done
fi

if [[ "$RUN_H1_TABLETOP" == "1" ]]; then
  run_sweep "h1_tabletop" "$SWEEP_MIN_LOG2_H1_TABLETOP" "$SWEEP_MAX_LOG2_H1_TABLETOP" "$SWEEP_SOLVERS_H1_TABLETOP"
  append_points "sweep" "h1_tabletop" "$RUN_DIR/h1_tabletop_sweep/sweep.jsonl" "$RUN_DIR/h1_tabletop_sweep/metadata.json"

  if [[ "$SKIP_ABLATIONS" != "1" ]]; then
    run_ablation "h1_tabletop" "$ABLATION_WORLDS_H1_TABLETOP"
    append_points "ablation" "h1_tabletop" "$RUN_DIR/h1_tabletop_ablation/ablation.jsonl" "$RUN_DIR/h1_tabletop_ablation/metadata.json"
  fi

  IFS=',' read -ra H1_SOLVERS <<< "$SWEEP_SOLVERS_H1_TABLETOP"
  H1_SUBSTEPS="${SCENARIOS_SUBSTEPS_H1_TABLETOP:-8}"
  for solver in "${H1_SOLVERS[@]}"; do
    run_render "h1_tabletop" "$solver" "$H1_SUBSTEPS"
  done
fi

build_renders_manifest

# Create run-level metadata
RUN_META_JSON=$(RUN_DIR="$RUN_DIR" RUN_ID="$RUN_ID" RUN_TIMESTAMP="$RUN_TIMESTAMP" RUN_COMMIT="$RUN_COMMIT" RUN_COMMIT_SHORT="$RUN_COMMIT_SHORT" AUTO_BRANCH="$AUTO_BRANCH" python - <<'PY'
import json
import os
from pathlib import Path

run_dir = Path(os.environ['RUN_DIR'])
meta_files = list(run_dir.rglob('metadata.json'))
metas = [json.loads(p.read_text()) for p in meta_files]

run_meta = {
    "run_id": os.environ['RUN_ID'],
    "timestamp": os.environ['RUN_TIMESTAMP'],
    "commit": os.environ['RUN_COMMIT'],
    "commit_short": os.environ['RUN_COMMIT_SHORT'],
    "branch": os.environ['AUTO_BRANCH'],
    "scenarios": sorted({m.get('scenario') for m in metas if m.get('scenario')}),
}

if metas:
    sample = metas[0]
    for key in ["gpu", "gpu_memory_total_gb", "platform", "python_version"]:
        if sample.get(key) is not None:
            run_meta[key] = sample.get(key)

(run_dir / 'meta.json').write_text(json.dumps(run_meta, indent=2))
print(json.dumps({**run_meta, "run_dir": f"runs/{os.environ['RUN_ID']}"}))
PY
)

printf "%s\n" "$RUN_META_JSON" >> "$RESULTS_DIR_PATH/runs.jsonl"

log "Committing results to $RESULTS_BRANCH"
cd "$RESULTS_DIR"

# Stage new run data

git add nightly/runs nightly/points.jsonl nightly/runs.jsonl nightly/index.html .nojekyll

if git diff --staged --quiet; then
  log "No changes to commit"
else
  COMMIT_MSG="Update nightly benchmarks - $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  git commit -m "$COMMIT_MSG"
  git push origin "$RESULTS_BRANCH"
  log "Pushed nightly results"
fi

log "Nightly benchmarking complete"
