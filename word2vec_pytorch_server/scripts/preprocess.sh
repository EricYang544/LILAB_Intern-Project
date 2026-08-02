#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
config="${1:-configs/sgns_server.yaml}"
exec "${PYTHON:-python3}" sgns_server.py preprocess --config "$config"

