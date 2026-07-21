"""Render the audited baseline CSV as a compact LaTeX longtable."""
import csv
from pathlib import Path

def v(x):
    x=(x or "").strip()
    return "--" if not x else x

def esc(x):
    return x.replace("_","\\_").replace("%","\\%")

rows=list(csv.DictReader(Path("paper/baselines.csv").open()))
out=[r"\begin{table*}[t]",r"\centering",r"\scriptsize",
     r"\caption{Audited ObjectNav comparison. Results from different HM3D versions or custom protocols are not directly ranked.}",
     r"\label{tab:objectnav_full}",r"\resizebox{\textwidth}{!}{%",r"\begin{tabular}{llcccccc}",
     r"\toprule Method & Setting & HM3D & SR & SPL & MP3D SR & MP3D SPL & Ref. \\",r"\midrule"]
last=None
for r in rows:
    group="Pre-mapped" if r["paradigm"].startswith("pre-explored") else "Online/unknown"
    if group!=last:
        if last is not None:out.append(r"\midrule")
        out.append(r"\multicolumn{8}{l}{\textit{"+group+r"}} \\")
        last=group
    ref="--" if r["cite_key"]=="none" else r"\cite{"+r["cite_key"]+"}"
    out.append("{} & {} & {} & {} & {} & {} & {} & {} \\\\".format(
        esc(r["method"]),esc(r["training"]),esc(v(r["hm3d_version"])),v(r["hm3d_sr"]),v(r["hm3d_spl"]),v(r["mp3d_sr"]),v(r["mp3d_spl"]),ref))
out += [r"\bottomrule",r"\end{tabular}%",r"}",r"\end{table*}",""]
Path("paper/tables/baselines_full.tex").write_text("\n".join(out))
print("wrote paper/tables/baselines_full.tex",len(rows),"rows")
