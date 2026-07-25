# Claim-to-file index

All paths are relative to this `supplementary/` directory.

| Paper evidence | Machine-readable source |
|---|---|
| Protocol boundary and sampling counts | `evidence/outputs/release_audit.json`; `reproduction/sstg_bench/uniform_map.py` |
| Main Table 3, target-view and independent geometry | `evidence/outputs/hm3d_val_oracle_analysis/`; `evidence/outputs/hm3d_val_uniform/oracle_geometry/` |
| Main Table 4, camera/raw/fused full validation | `evidence/outputs/hm3d_val_uniform/gpt54_camera_node_analysis/`; `evidence/outputs/hm3d_val_uniform/gpt54_rgbd_wide120_analysis_raw/`; `evidence/outputs/hm3d_val_uniform/gpt54_rgbd_wide120_analysis_fused/` |
| Main Table 5, backend/FoV representation controls | GPT-5.4, MiMo, and Qwen summaries under `evidence/outputs/hm3d_minival_uniform/` |
| Main Table 6, fusion-aware sequential recovery | `evidence/outputs/hm3d_val_uniform/gpt54_rgbd_wide120_fusion_first_raw_topk_2p0m/` |
| Fused-only and raw Top-3 controls | `evidence/outputs/hm3d_val_uniform/gpt54_rgbd_wide120_fused_topk_2p0m/`; `evidence/outputs/hm3d_val_uniform/gpt54_rgbd_wide120_raw_topk_2p0m/` |
| Arrival-verifier calibration | `evidence/outputs/hm3d_val_uniform/gpt54_arrival_verified_top3_strict/`; `evidence/outputs/hm3d_val_uniform/gpt54_arrival_verifier_top3_strict/` |
| Mapping and arrival prompts | `reproduction/sstg_bench/peterai_rgbd_semantics.py`; `reproduction/sstg_bench/arrival_vlm.py`; `reproduction/sstg_bench/mimo_rgbd_semantics.py`; `reproduction/sstg_bench/local_rgbd_semantics.py` |
| Fusion and reachability rules | `reproduction/sstg_bench/rgbd_fusion.py` |
| Density and corruption studies | `evidence/outputs/hm3d_minival_uniform/density_analysis/`; `evidence/outputs/analysis/stress_aggregate.csv`; `evidence/outputs/analysis/stress_runs.csv` |
| GPT-5.5 semantic-isolation pilot | `evidence/outputs/analysis/`; `evidence/outputs/hm3d_minival_vlm/maps/mapping_report.json`; `evidence/outputs/hm3d_val_vlm/maps/mapping_report.json` |
| Cross-file consistency audit | `evidence/outputs/release_audit.json` |

The PDF includes representative figures. Raw RGB-D, image sequences, MP4 files,
scene datasets, and the real-robot video are intentionally excluded and belong
to the benchmark storage or separate multimedia submission.
