#!/bin/bash

# Thin head-node wrapper for the shared nightly Slurm submission path.
#
# Examples:
#   benchmarks/nightly_slurm.sh --run-mode validation --submission-mode parallel --skip-publish
#   benchmarks/nightly_slurm.sh --task-id validation_g1_flat_sweep --submission-mode serial --skip-publish

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

exec uv run --extra examples --extra torch-cu12 -m benchmarks.nightly.slurm submit "$@"
