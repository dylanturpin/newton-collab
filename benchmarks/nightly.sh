#!/bin/bash

# Thin local wrapper for the shared nightly planner and executor.
#
# Examples:
#   benchmarks/nightly.sh --run-mode validation --skip-publish
#   benchmarks/nightly.sh --task-id validation_g1_flat_render --skip-publish

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

exec uv run --extra examples --extra torch-cu12 -m benchmarks.nightly.local "$@"
