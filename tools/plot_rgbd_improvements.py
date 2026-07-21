"""Render the independent-topology improvement and VLM ablation table."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/hm3d_minival_uniform/improvement_summary"
OUT.mkdir(parents=True, exist_ok=True)


def summary(path):
    return json.loads((ROOT / path).read_text())


def conventional(label, path, protocol, backend):
    data = summary(path)
    return {"method": label, "protocol": protocol, "backend": backend, "sr": data["sr"], "spl": data["spl"]}


def topk(label, path, k, separation, backend="Qwen2.5-VL-3B"):
    data = summary(path)["metrics"][str(k)]
    return {"method": label, "protocol": "oracle-feedback diagnostic", "backend": backend,
            "sr": data["sr"], "spl": data["spl"], "k": k, "separation_m": separation}


density = summary("outputs/hm3d_minival_uniform/density_analysis/density_ablation.json")["summaries"]["1.0"]
rows = [
    {"method": "Independent topology, oracle semantics", "protocol": "coverage ceiling", "backend": "oracle",
     "sr": density["sr"], "spl": density["spl"]},
    conventional("Qwen RGB panorama, confidence", "outputs/hm3d_minival_uniform/analysis_fixed/summary_vlm_all_confidence.json",
                 "single candidate", "Qwen2.5-VL-3B"),
    topk("Qwen Top-3", "outputs/hm3d_minival_uniform/topk_analysis/summary_topk.json", 3, 0),
    topk("Qwen diverse Top-3 (5 m)", "outputs/hm3d_minival_uniform/topk_diverse_5p0m/summary_topk.json", 3, 5),
    conventional("Peter RGB-D 90° raw", "outputs/hm3d_minival_uniform/peterai_rgbd_analysis_raw/summary_vlm_all_confidence.json",
                 "depth-projected confidence", "PeterAI gpt-5.4"),
    conventional("Peter RGB-D 90° fused", "outputs/hm3d_minival_uniform/peterai_rgbd_analysis_soft/summary_vlm_all_confidence.json",
                 "soft 3D fusion", "PeterAI gpt-5.4"),
    conventional("Peter RGB-D 120° raw", "outputs/hm3d_minival_uniform/peterai_rgbd_wide120_analysis_raw/summary_vlm_all_confidence.json",
                 "depth-projected confidence", "PeterAI gpt-5.4"),
    conventional("Peter RGB-D 120° fused", "outputs/hm3d_minival_uniform/peterai_rgbd_wide120_analysis_soft/summary_vlm_all_confidence.json",
                 "soft 3D fusion", "PeterAI gpt-5.4"),
    topk("Peter RGB-D 120° Top-2", "outputs/hm3d_minival_uniform/peterai_rgbd_wide120_topk_0m/summary_topk.json", 2, 0, "PeterAI gpt-5.4"),
    conventional("MiMo RGB-D 90° fused", "outputs/hm3d_minival_uniform/mimo_rgbd_analysis_reparsed_soft/summary_vlm_all_confidence.json",
                 "soft 3D fusion", "mimo-v2.5"),
    conventional("MiMo RGB-D 120° fused", "outputs/hm3d_minival_uniform/mimo_rgbd_wide120_analysis_soft/summary_vlm_all_confidence.json",
                 "soft 3D fusion", "mimo-v2.5"),
]

fields = sorted({key for row in rows for key in row})
with (OUT / "benchmark_comparison.csv").open("w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
(OUT / "benchmark_comparison.json").write_text(json.dumps(rows, indent=2))

labels = [row["method"] for row in rows]
x = np.arange(len(rows)); colors = ["#455a64", "#9e9e9e", "#ffb74d", "#fb8c00", "#90caf9", "#64b5f6", "#1e88e5", "#1565c0", "#ff8f00", "#66bb6a", "#2e7d32"]
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
for ax, metric, title in zip(axes, ("sr", "spl"), ("Success rate", "SPL")):
    values = [row[metric] for row in rows]
    bars = ax.bar(x, values, color=colors)
    for index in (2, 3, 8):
        bars[index].set_hatch("//")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, value + .012, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 1.08); ax.set_ylabel(title); ax.grid(axis="y", alpha=.25)
axes[0].set_title("Independent 0.8 m topology: semantic localization improvements (30 episodes / 2 scenes)")
axes[1].set_xticks(x, labels, rotation=24, ha="right")
fig.text(.5, .005, "Hatched bars use evaluator oracle feedback after each candidate and are diagnostic upper bounds.", ha="center", fontsize=9)
fig.tight_layout(rect=(0, .035, 1, 1))
fig.savefig(OUT / "rgbd_improvements.png", dpi=220)
fig.savefig(OUT / "rgbd_improvements.pdf")
plt.close(fig)

tex = [r"\begin{table*}[t]", r"\centering", r"\caption{Improvements on the annotation-independent 0.8 m topology (HM3D-v2 minival, 30 episodes). Top-$K$ uses oracle success feedback and is a diagnostic upper bound.}",
       r"\label{tab:rgbd_improvements}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{llcc}", "\\toprule Method & Protocol & SR $\\uparrow$ & SPL $\\uparrow$ \\\\", r"\midrule"]
for row in rows:
    method = row["method"].replace("°", r"$^\circ$")
    tex.append(f"{method} & {row['protocol']} & {row['sr']:.3f} & {row['spl']:.3f}" + r" \\")
tex += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table*}", ""]
(ROOT / "paper/tables/rgbd_improvements.tex").write_text("\n".join(tex))
print(json.dumps({"rows": len(rows), "output": str(OUT.relative_to(ROOT))}, indent=2))
