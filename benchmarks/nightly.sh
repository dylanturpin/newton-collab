#!/bin/bash

# Thin local wrapper for the shared nightly planner and executor.
#
# Examples:
#   benchmarks/nightly.sh --run-mode validation --skip-publish
#   benchmarks/nightly.sh --task-id validation_g1_flat_render --skip-publish

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
USER_NAME="${USER:-$(id -un)}"

export TMPDIR="${TMPDIR:-/tmp}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/$USER_NAME/uv-cache}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-/tmp/$USER_NAME/newton-venv}"
export WARP_CACHE_PATH="${WARP_CACHE_PATH:-/tmp/$USER_NAME/warp-cache}"
export NEWTON_CACHE_PATH="${NEWTON_CACHE_PATH:-/tmp/$USER_NAME/newton-cache}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-/tmp/$USER_NAME/cuda-compute-cache}"
DEFAULT_CHERRY_PICK_REF="${NIGHTLY_CHERRY_PICK_REF-fast-bulk-replicate}"

cd "$REPO_ROOT"

CMD=(uv run --extra examples --extra torch-cu12 -m benchmarks.nightly.local)
if [[ -n "$DEFAULT_CHERRY_PICK_REF" ]]; then
  CMD+=(--cherry-pick-ref "$DEFAULT_CHERRY_PICK_REF")
fi
CMD+=("$@")

exec "${CMD[@]}"
