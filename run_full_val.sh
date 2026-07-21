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

# Habitat-side oracle/coverage upper bound on all 1,000 official v2 episodes.
"$HABITAT_PYTHON" -m sstg_bench.experiments \
  --split val --scene-dir val --skip-vlm --stress-seeds 0 \
  --output outputs/hm3d_val_oracle_analysis

# After semantic maps have been constructed by either PeterAI or local Qwen,
# evaluate them by replacing --vlm-maps with the desired map directory.
if [[ -d outputs/hm3d_val_qwen/maps/4ok3usBNeis ]]; then
  "$HABITAT_PYTHON" -m sstg_bench.experiments \
    --split val --scene-dir val --stress-seeds 0 \
    --vlm-maps outputs/hm3d_val_qwen/maps \
    --output outputs/hm3d_val_qwen_analysis
fi
