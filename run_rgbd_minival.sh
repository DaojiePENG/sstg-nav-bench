#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

BACKEND="${1:-peterai}"
SENSOR="${2:-wide120}"
HABITAT_PYTHON="${HABITAT_PYTHON:-/home/daojie/anaconda3/envs/habitat/bin/python}"
VLM_PYTHON="${VLM_PYTHON:-python}"

case "$SENSOR" in
  standard)
    CAPTURE="outputs/hm3d_minival_uniform/rgbd_capture"
    SENSOR_ARGS=(--width 640 --height 360 --hfov 90)
    ;;
  wide120)
    CAPTURE="outputs/hm3d_minival_uniform/rgbd_capture_wide120"
    SENSOR_ARGS=(--width 640 --height 640 --hfov 120)
    ;;
  *)
    echo "sensor must be standard or wide120" >&2
    exit 2
    ;;
esac

"$HABITAT_PYTHON" -m sstg_bench.rgbd_capture \
  --source outputs/hm3d_minival_uniform/source --output "$CAPTURE" "${SENSOR_ARGS[@]}"

case "$BACKEND" in
  peterai)
    SEMANTICS="outputs/hm3d_minival_uniform/peterai_rgbd_semantics_${SENSOR}"
    "$VLM_PYTHON" -m sstg_bench.peterai_rgbd_semantics \
      --source "$CAPTURE" --output "$SEMANTICS" \
      --model "${PETERAI_MODEL:-gpt-5.4}" --wire-api "${PETERAI_WIRE_API:-chat}" \
      --workers "${VLM_WORKERS:-6}"
    ;;
  mimo)
    SEMANTICS="outputs/hm3d_minival_uniform/mimo_rgbd_semantics_${SENSOR}"
    "$VLM_PYTHON" -m sstg_bench.mimo_rgbd_semantics \
      --source "$CAPTURE" --output "$SEMANTICS" --workers "${VLM_WORKERS:-4}"
    ;;
  *)
    echo "backend must be peterai or mimo" >&2
    exit 2
    ;;
esac

PREFIX="outputs/hm3d_minival_uniform/${BACKEND}_rgbd_${SENSOR}"
ANALYSIS="${PREFIX}_analysis_soft"
"$HABITAT_PYTHON" -m sstg_bench.rgbd_fusion \
  --source "$SEMANTICS" --output "${PREFIX}_fusion" --min-support 1
"$HABITAT_PYTHON" -m sstg_bench.experiments \
  --vlm-maps "${PREFIX}_fusion/clustered_maps" --output "$ANALYSIS" --stress-seeds 0
"$HABITAT_PYTHON" -m sstg_bench.rgbd_visuals \
  --semantics "$SEMANTICS" --fusion "${PREFIX}_fusion" --output "${PREFIX}_visuals"

echo "Results: ${ANALYSIS}/summary_vlm_all_confidence.json"
echo "Visuals: ${PREFIX}_visuals/"
