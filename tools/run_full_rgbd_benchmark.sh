#!/usr/bin/env bash
set -euo pipefail

# Reproduce the full 36-scene, goal-independent GPT-5.4 RGB-D experiment.
# Successful VLM responses are cached, so the semantic stage is resumable.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=.

HABITAT_PYTHON="${HABITAT_PYTHON:-/home/daojie/anaconda3/envs/habitat/bin/python}"
GENERAL_PYTHON="${GENERAL_PYTHON:-python}"
WORKERS="${WORKERS:-24}"

"$HABITAT_PYTHON" -m sstg_bench.uniform_map \
  --split val --scene-dir val \
  --output outputs/hm3d_val_uniform/source \
  --pool-size 12000 --cover-radius 0.8 --max-nodes 2500 --skip-panorama

"$HABITAT_PYTHON" -m sstg_bench.rgbd_capture \
  --split val --scene-dir val \
  --source outputs/hm3d_val_uniform/source \
  --output outputs/hm3d_val_uniform/rgbd_capture_wide120 \
  --width 640 --height 640 --hfov 120

"$GENERAL_PYTHON" -m sstg_bench.peterai_rgbd_semantics \
  --source outputs/hm3d_val_uniform/rgbd_capture_wide120 \
  --output outputs/hm3d_val_uniform/gpt54_rgbd_semantics_wide120 \
  --model gpt-5.4 --wire-api chat --workers "$WORKERS" --checkpoint-every 25

"$GENERAL_PYTHON" -m sstg_bench.camera_node_map \
  --source outputs/hm3d_val_uniform/gpt54_rgbd_semantics_wide120 \
  --output outputs/hm3d_val_uniform/gpt54_camera_nodes_wide120

"$HABITAT_PYTHON" -m sstg_bench.rgbd_fusion \
  --split val --scene-dir val \
  --source outputs/hm3d_val_uniform/gpt54_rgbd_semantics_wide120 \
  --output outputs/hm3d_val_uniform/gpt54_rgbd_wide120_fusion

"$HABITAT_PYTHON" -m sstg_bench.experiments \
  --split val --scene-dir val --stress-seeds 0 \
  --vlm-maps outputs/hm3d_val_uniform/gpt54_camera_nodes_wide120 \
  --output outputs/hm3d_val_uniform/gpt54_camera_node_analysis

"$HABITAT_PYTHON" -m sstg_bench.experiments \
  --split val --scene-dir val --stress-seeds 0 \
  --vlm-maps outputs/hm3d_val_uniform/gpt54_rgbd_wide120_fusion/raw_maps \
  --output outputs/hm3d_val_uniform/gpt54_rgbd_wide120_analysis_raw

"$HABITAT_PYTHON" -m sstg_bench.experiments \
  --split val --scene-dir val --stress-seeds 0 \
  --vlm-maps outputs/hm3d_val_uniform/gpt54_rgbd_wide120_fusion/clustered_maps \
  --output outputs/hm3d_val_uniform/gpt54_rgbd_wide120_analysis_fused

"$HABITAT_PYTHON" -m sstg_bench.topk \
  --split val --scene-dir val --max-k 3 --min-separation 0 \
  --maps outputs/hm3d_val_uniform/gpt54_rgbd_wide120_fusion/raw_maps \
  --output outputs/hm3d_val_uniform/gpt54_rgbd_wide120_raw_topk_0m

"$HABITAT_PYTHON" -m sstg_bench.topk \
  --split val --scene-dir val --max-k 3 --min-separation 0 \
  --maps outputs/hm3d_val_uniform/gpt54_rgbd_wide120_fusion/clustered_maps \
  --output outputs/hm3d_val_uniform/gpt54_rgbd_wide120_topk_0m

# Matched 2 m controls used by the main-paper recovery-policy ablation.
"$HABITAT_PYTHON" -m sstg_bench.topk \
  --split val --scene-dir val --max-k 3 --min-separation 2.0 \
  --maps outputs/hm3d_val_uniform/gpt54_rgbd_wide120_fusion/raw_maps \
  --output outputs/hm3d_val_uniform/gpt54_rgbd_wide120_raw_topk_2p0m

"$HABITAT_PYTHON" -m sstg_bench.topk \
  --split val --scene-dir val --max-k 3 --min-separation 2.0 \
  --maps outputs/hm3d_val_uniform/gpt54_rgbd_wide120_fusion/clustered_maps \
  --output outputs/hm3d_val_uniform/gpt54_rgbd_wide120_fused_topk_2p0m

"$HABITAT_PYTHON" -m sstg_bench.topk \
  --split val --scene-dir val --max-k 3 --min-separation 2.0 \
  --ranking-strategy confidence_support \
  --primary-maps outputs/hm3d_val_uniform/gpt54_rgbd_wide120_fusion/clustered_maps --primary-k 1 \
  --maps outputs/hm3d_val_uniform/gpt54_rgbd_wide120_fusion/raw_maps \
  --output outputs/hm3d_val_uniform/gpt54_rgbd_wide120_fusion_first_raw_topk_2p0m

"$HABITAT_PYTHON" -m sstg_bench.arrival_capture \
  --split val --scene-dir val --max-k 3 --min-separation 2.0 \
  --ranking-strategy confidence_support \
  --primary-maps outputs/hm3d_val_uniform/gpt54_rgbd_wide120_fusion/clustered_maps --primary-k 1 \
  --maps outputs/hm3d_val_uniform/gpt54_rgbd_wide120_fusion/raw_maps \
  --output outputs/hm3d_val_uniform/gpt54_arrival_capture_fusion_aware_top3 \
  --width 640 --height 640 --hfov 120

# Retain the 3 m raw diagnostic and the strict fresh-arrival calibration as
# auxiliary audits.  They no longer define the headline Top-3 row.
"$HABITAT_PYTHON" -m sstg_bench.topk \
  --split val --scene-dir val --max-k 3 --min-separation 3.0 \
  --maps outputs/hm3d_val_uniform/gpt54_rgbd_wide120_fusion/raw_maps \
  --output outputs/hm3d_val_uniform/gpt54_rgbd_wide120_raw_topk_3m

"$HABITAT_PYTHON" -m sstg_bench.arrival_capture \
  --split val --scene-dir val --max-k 3 --min-separation 3.0 \
  --maps outputs/hm3d_val_uniform/gpt54_rgbd_wide120_fusion/raw_maps \
  --output outputs/hm3d_val_uniform/gpt54_arrival_capture_top3

"$GENERAL_PYTHON" -m sstg_bench.arrival_vlm \
  --source outputs/hm3d_val_uniform/gpt54_arrival_capture_top3 \
  --output outputs/hm3d_val_uniform/gpt54_arrival_verifier_top3_strict \
  --model gpt-5.4 --workers "$WORKERS"

"$HABITAT_PYTHON" -m sstg_bench.arrival_evaluate \
  --split val --scene-dir val --max-k 3 --min-separation 3.0 \
  --source outputs/hm3d_val_uniform/gpt54_arrival_capture_top3 \
  --verifier outputs/hm3d_val_uniform/gpt54_arrival_verifier_top3_strict \
  --output outputs/hm3d_val_uniform/gpt54_arrival_verified_top3_strict

"$HABITAT_PYTHON" -m sstg_bench.arrival_visuals \
  --split val --scene-dir val \
  --source outputs/hm3d_val_uniform/gpt54_arrival_capture_top3 \
  --verifier outputs/hm3d_val_uniform/gpt54_arrival_verifier_top3_strict \
  --evaluation outputs/hm3d_val_uniform/gpt54_arrival_verified_top3_strict \
  --output outputs/hm3d_val_uniform/gpt54_arrival_verified_visuals

"$HABITAT_PYTHON" -m sstg_bench.uniform_density \
  --split val --scene-dir val --fractions 1.0 \
  --maps outputs/hm3d_val_uniform/source \
  --output outputs/hm3d_val_uniform/oracle_geometry
