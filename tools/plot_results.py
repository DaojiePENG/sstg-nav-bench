"""Generate paper-ready full-validation category and failure plots."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parents[1];out=root/"outputs/analysis_figures";out.mkdir(exist_ok=True)
oracle=json.loads((root/"outputs/hm3d_val_oracle_analysis/summary_oracle.json").read_text())
qwen=json.loads((root/"outputs/hm3d_val_qwen_analysis/summary_vlm_all_confidence.json").read_text())
cats=list(oracle["per_category"]);x=np.arange(len(cats));width=.35
fig,axes=plt.subplots(1,2,figsize=(11,4))
for ax,metric,title in zip(axes,("sr","spl"),("Success rate","SPL")):
    ax.bar(x-width/2,[oracle["per_category"][c][metric] for c in cats],width,label="Oracle",color="#455a64")
    ax.bar(x+width/2,[qwen["per_category"][c][metric] for c in cats],width,label="Qwen confidence",color="#26a69a")
    ax.set_xticks(x,cats,rotation=25);ax.set_ylim(0,1.05);ax.set_title(title);ax.grid(axis="y",alpha=.25);ax.legend()
fig.tight_layout();fig.savefig(out/"full_val_per_category.png",dpi=220);fig.savefig(out/"full_val_per_category.pdf");plt.close(fig)

nearest=json.loads((root/"outputs/hm3d_val_qwen_analysis/summary_vlm_all_nearest.json").read_text())
primary=json.loads((root/"outputs/hm3d_val_qwen_analysis/summary_vlm_primary.json").read_text())
variants=[nearest,primary,qwen];labels=["Nearest","Primary","Confidence"]
wrong=[v["failure_counts"].get("wrong_semantic_candidate",0) for v in variants]
missing=[v["failure_counts"].get("no_semantic_candidate",0) for v in variants]
fig,ax=plt.subplots(figsize=(7,4));ax.bar(labels,wrong,label="Wrong candidate",color="#ef5350")
ax.bar(labels,missing,bottom=wrong,label="No candidate",color="#ffa726")
ax.set(ylabel="Failed episodes",title="Full HM3D-v2 failure decomposition");ax.legend();ax.grid(axis="y",alpha=.25)
fig.tight_layout();fig.savefig(out/"full_val_failure_modes.png",dpi=220);fig.savefig(out/"full_val_failure_modes.pdf");plt.close(fig)
