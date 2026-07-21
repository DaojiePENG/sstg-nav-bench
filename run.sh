#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
if python -c 'import habitat_sim' >/dev/null 2>&1; then
  HABITAT_PYTHON="${HABITAT_PYTHON:-python}"
else
  HABITAT_PYTHON="${HABITAT_PYTHON:-/home/daojie/anaconda3/envs/habitat/bin/python}"
fi
"$HABITAT_PYTHON" -m sstg_bench.benchmark --config configs/minival.yaml
