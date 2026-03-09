#!/bin/bash

# Thin head-node wrapper for the shared nightly Slurm submission path.
#
# Examples:
#   benchmarks/nightly_slurm.sh --run-mode validation --submission-mode parallel --skip-publish
#   benchmarks/nightly_slurm.sh --task-id validation_g1_flat_sweep --submission-mode serial --skip-publish

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

cd "$REPO_ROOT"

exec uv run --extra examples --extra torch-cu12 -m benchmarks.nightly.slurm submit "$@"
