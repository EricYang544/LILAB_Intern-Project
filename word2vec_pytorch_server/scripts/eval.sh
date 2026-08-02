#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
run_dir="${1:?Usage: $0 RUN_DIR [CHECKPOINT]}"
checkpoint="${2:-best.pt}"
exec "${PYTHON:-python3}" sgns_server.py evaluate --run-dir "$run_dir" --checkpoint "$checkpoint"

