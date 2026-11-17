#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m elmo_project.prepare --config configs/ner_24h.yaml
python -m elmo_project.pilot --config configs/ner_24h.yaml
python -m elmo_project.run --plan runs/pilot/resolved_plan.json
python -m elmo_project.report --run-dir runs/ner_24h
