#!/usr/bin/env python3
"""Build compact machine-readable tables from completed full-validation artifacts."""
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/hm3d_val_uniform"


def load(relative: str):
    return json.loads((ROOT / relative).read_text())


def metric_row(name: str, protocol: str, summary: dict) -> dict:
    return {
        "method": name,
        "protocol": protocol,
        "episodes": summary["episodes"],
        "successes": summary["successes"],
        "sr": summary["sr"],
        "sr_ci_low": summary["sr_95ci"][0],
        "sr_ci_high": summary["sr_95ci"][1],
        "spl": summary["spl"],
        "spl_ci_low": summary["spl_95ci"][0],
        "spl_ci_high": summary["spl_95ci"][1],
        "dtg": summary.get("dtg", ""),
    }


def main() -> None:
    geometry = load("outputs/hm3d_val_uniform/oracle_geometry/density_ablation.json")["summaries"]["1.0"]
    camera = load("outputs/hm3d_val_uniform/gpt54_camera_node_analysis/summary_vlm_all_confidence.json")
    raw = load("outputs/hm3d_val_uniform/gpt54_rgbd_wide120_analysis_raw/summary_vlm_all_confidence.json")
    fused = load("outputs/hm3d_val_uniform/gpt54_rgbd_wide120_analysis_fused/summary_vlm_all_confidence.json")
    raw_topk = load("outputs/hm3d_val_uniform/gpt54_rgbd_wide120_raw_topk_2p0m/summary_topk.json")
    fused_topk = load("outputs/hm3d_val_uniform/gpt54_rgbd_wide120_fused_topk_2p0m/summary_topk.json")
    fusion_aware_topk = load("outputs/hm3d_val_uniform/gpt54_rgbd_wide120_fusion_first_raw_topk_2p0m/summary_topk.json")
    topk = load("outputs/hm3d_val_uniform/gpt54_rgbd_wide120_raw_topk_3m/summary_topk.json")
    arrival = load("outputs/hm3d_val_uniform/gpt54_arrival_verified_top3_strict/summary_arrival_verified.json")

    rows = [
        metric_row("Independent geometry ceiling", "evaluator semantics; goal-independent topology", geometry),
        metric_row("GPT-5.4 camera node", "autonomous single candidate", camera),
        metric_row("GPT-5.4 raw RGB-D", "autonomous single candidate", raw),
        metric_row("GPT-5.4 soft fusion", "autonomous single candidate", fused),
    ]
    for family, summary, protocol in (
        ("GPT-5.4 raw", raw_topk, "non-fused ranking; 2 m spatial diversity"),
        ("GPT-5.4 fused representatives", fused_topk, "collapsed fused representatives; 2 m spatial diversity"),
        ("GPT-5.4 fusion-aware residual", fusion_aware_topk, "fused primary; confidence-ranked 2 m-diverse residuals"),
    ):
        for k in (1, 2, 3):
            metric = summary["metrics"][str(k)]
            rows.append({
                "method": f"{family} Top-{k}",
                "protocol": protocol,
                "episodes": summary["episodes"],
                "successes": metric["successes"],
                "sr": metric["sr"],
                "sr_ci_low": metric["sr_95ci"][0],
                "sr_ci_high": metric["sr_95ci"][1],
                "spl": metric["spl"],
                "spl_ci_low": metric["spl_95ci"][0],
                "spl_ci_high": metric["spl_95ci"][1],
                "dtg": "",
            })
    for k in (1, 2, 3):
        metric = arrival["metrics"][str(k)]
        rows.append({
            "method": f"GPT-5.4 fresh-arrival verified Top-{k}",
            "protocol": "auxiliary arrival-classifier calibration; fresh RGB-D, target VLM, strict dual geometry",
            "episodes": arrival["episodes"],
            "successes": metric["successes"],
            "sr": metric["sr"],
            "sr_ci_low": metric["sr_95ci"][0],
            "sr_ci_high": metric["sr_95ci"][1],
            "spl": metric["spl"],
            "spl_ci_low": metric["spl_95ci"][0],
            "spl_ci_high": metric["spl_95ci"][1],
            "dtg": arrival["dtg"] if k == 3 else "",
        })
    for k in (1, 2, 3):
        metric = topk["metrics"][str(k)]
        rows.append({
            "method": f"GPT-5.4 raw diverse Top-{k}",
            "protocol": "oracle success feedback after each visit; 3 m candidate separation",
            "episodes": topk["episodes"],
            "successes": metric["successes"],
            "sr": metric["sr"],
            "sr_ci_low": metric["sr_95ci"][0],
            "sr_ci_high": metric["sr_95ci"][1],
            "spl": metric["spl"],
            "spl_ci_low": metric["spl_95ci"][0],
            "spl_ci_high": metric["spl_95ci"][1],
            "dtg": "",
        })

    paired = []
    for name, filename in (
        ("camera_to_raw", "paired_camera_to_raw.json"),
        ("raw_to_fused", "paired_raw_to_fused.json"),
        ("camera_to_fused", "paired_camera_to_fused.json"),
    ):
        item = load(f"outputs/hm3d_val_uniform/{filename}")
        paired.append({
            "comparison": name,
            "episodes": item["episodes"],
            "gains": item["gains"],
            "losses": item["losses"],
            "delta_sr": item["delta_sr"],
            "mcnemar_exact_two_sided_p": item["mcnemar_exact_two_sided_p"],
            "delta_spl": item["delta_spl"],
            "delta_spl_ci_low": item["delta_spl_95ci"][0],
            "delta_spl_ci_high": item["delta_spl_95ci"][1],
        })

    payload = {
        "dataset": "HM3D ObjectNav-v2 validation",
        "episodes": 1000,
        "scenes": 36,
        "topology": {"nodes": 6642, "edges": 21845, "empirical_cover_radius_m": 0.7999346},
        "sensor": {"views_per_node": 4, "width": 640, "height": 640, "hfov_deg": 120, "vfov_deg": 120},
        "model": "gpt-5.4",
        "metrics": rows,
        "paired_comparisons": paired,
    }
    (OUTPUT / "benchmark_summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (OUTPUT / "benchmark_summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (OUTPUT / "paired_summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=paired[0].keys())
        writer.writeheader()
        writer.writerows(paired)
    print(json.dumps({"metrics": len(rows), "paired_comparisons": len(paired), "output": str(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
