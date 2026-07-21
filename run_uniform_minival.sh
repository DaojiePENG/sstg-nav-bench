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

# This sampling pass is independent of ObjectNav goal annotations.
"$HABITAT_PYTHON" -m sstg_bench.uniform_map \
  --split val_mini --scene-dir minival \
  --output outputs/hm3d_minival_uniform/source

echo "Now activate the sstg-nav-vlm environment and run:"
echo "  python -m sstg_bench.local_vlm_map --source outputs/hm3d_minival_uniform/source --output outputs/hm3d_minival_uniform/qwen_maps --max-new-tokens 300"
echo "Then return to the Habitat environment and run:"
echo "  $HABITAT_PYTHON -m sstg_bench.experiments --vlm-maps outputs/hm3d_minival_uniform/qwen_maps --output outputs/hm3d_minival_uniform/analysis --stress-seeds 0"
