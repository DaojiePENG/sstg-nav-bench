#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

HABITAT_PYTHON="${HABITAT_PYTHON:-/home/daojie/anaconda3/envs/habitat/bin/python}"
VLM_PYTHON="${VLM_PYTHON:-python}"
SEMANTICS="outputs/hm3d_minival_uniform/qwen_rgbd_semantics_90"
FUSION="outputs/hm3d_minival_uniform/qwen_rgbd_fusion_90"

"$VLM_PYTHON" -m sstg_bench.local_rgbd_semantics \
  --source outputs/hm3d_minival_uniform/rgbd_capture \
  --output "$SEMANTICS" --max-new-tokens 350 --max-visual-tokens 768

"$VLM_PYTHON" -m sstg_bench.camera_node_map \
  --source "$SEMANTICS" \
  --output outputs/hm3d_minival_uniform/qwen_camera_nodes_90

"$HABITAT_PYTHON" -m sstg_bench.rgbd_fusion \
  --source "$SEMANTICS" --output "$FUSION" --min-support 1

for representation in camera raw fused; do
  case "$representation" in
    camera) maps="outputs/hm3d_minival_uniform/qwen_camera_nodes_90" ;;
    raw) maps="$FUSION/raw_maps" ;;
    fused) maps="$FUSION/clustered_maps" ;;
  esac
  "$HABITAT_PYTHON" -m sstg_bench.experiments \
    --vlm-maps "$maps" \
    --output "outputs/hm3d_minival_uniform/qwen_rgbd_90_analysis_${representation}" \
    --stress-seeds 0
done

echo "Qwen same-response camera/raw/fused summaries are complete."
