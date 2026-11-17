#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m elmo_project.prepare --config configs/smoke.yaml
python -m elmo_project.pilot --config configs/smoke.yaml
python -m elmo_project.run --plan runs/smoke_pilot/resolved_plan.json
python -m elmo_project.report --run-dir runs/smoke
