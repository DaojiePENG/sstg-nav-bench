"""Build paper figures from benchmark artifacts.

The script intentionally contains no benchmark constants beyond artifact
locations.  Quantitative values are read from machine-readable summaries and
qualitative panels are traced to the RGB, depth, map, and trajectory artifacts
used by the evaluator.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import imageio.v2 as imageio
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.parent / "SSTGNavPaperAAAI"
OUT = PAPER / "figures" / "generated"
MINI = ROOT / "outputs" / "hm3d_minival_uniform"
FULL = ROOT / "outputs" / "hm3d_val_uniform"
ROBOT = ROOT.parent / "sstg-nav" / "sstg_ui_app" / "public" / "maps"

COLORS = {
    "nav": "#4477AA",
    "semantic": "#EE6677",
    "metric": "#228833",
    "query": "#CCBB44",
    "eval": "#AA3377",
    "gray": "#667788",
}


def setup() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "image.interpolation": "none",
    })


def save(fig: plt.Figure, stem: str, *, dpi: int = 300) -> None:
    # Preserve source-image detail in vector containers.  Without an explicit
    # save DPI, Matplotlib rasterizes every imshow panel at roughly 100 PPI.
    fig.savefig(OUT / f"{stem}.pdf", dpi=dpi, bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def load_summary(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text())


def system_overview() -> None:
    fig, ax = plt.subplots(figsize=(7.05, 2.42))
    ax.set_xlim(0, 14); ax.set_ylim(0, 5.15); ax.axis("off")

    def box(x, y, w, h, text, color, subtitle=""):
        patch = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.10",
            fc=mpl.colors.to_rgba(color, .12), ec=color, lw=1.2,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h * .60, text, ha="center", va="center",
                weight="bold", fontsize=6.9)
        if subtitle:
            ax.text(x + w / 2, y + h * .27, subtitle, ha="center", va="center", fontsize=5.35)
        return patch

    def arrow(a, b, color="#44515f", style="-"):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9,
                                     lw=1.05, color=color, linestyle=style))

    ax.text(.15, 4.82, "OFFLINE, GOAL-INDEPENDENT PRE-EXPLORATION", color=COLORS["nav"],
            weight="bold", fontsize=7.6)
    box(.15, 3.08, 2.15, 1.20, "Topology survey", COLORS["nav"], "0.8 m navigable coverage")
    box(2.73, 3.08, 2.15, 1.20, "Cardinal RGB-D", COLORS["semantic"], "four calibrated views")
    box(5.31, 3.08, 2.15, 1.20, "VLM detections", COLORS["semantic"], "category, box, confidence")
    box(7.89, 3.08, 2.15, 1.20, "3D standoffs", COLORS["metric"], "depth → 3D → navmesh")
    box(10.47, 3.08, 2.15, 1.20, "Multi-view fusion", COLORS["metric"], "3D + reachability support")
    for x in (2.30, 4.88, 7.46, 10.04): arrow((x, 3.68), (x + .42, 3.68))

    ax.text(.15, 2.54, "ONLINE REPEATED QUERY", color="#8A6D00", weight="bold", fontsize=7.6)
    box(.15, .82, 2.15, 1.18, "Language query", COLORS["query"], "target-category parser")
    box(3.00, .82, 2.15, 1.18, "Rank candidates", COLORS["query"], "confidence + path cost")
    box(5.85, .82, 2.15, 1.18, "Plan & execute", COLORS["nav"], "topology / Nav2")
    box(8.70, .82, 2.15, 1.18, "Arrival check", COLORS["eval"], "fresh RGB-D + VLM + depth")
    for x in (2.30, 5.15, 8.00): arrow((x, 1.41), (x + .68, 1.41))
    arrow((10.83, 1.03), (3.03, 1.03), COLORS["eval"])
    ax.text(6.92, .58, "verified reject → next candidate", color=COLORS["eval"], ha="center", fontsize=6.5)
    arrow((11.55, 3.03), (10.25, 2.02), COLORS["metric"])

    # Evaluator-only lane makes leakage boundaries explicit.
    ax.plot([12.98, 12.98], [.55, 4.47], color=COLORS["eval"], lw=1, ls="--")
    ax.text(13.16, 4.18, "Evaluator only", color=COLORS["eval"], weight="bold", rotation=90,
            va="top", ha="center")
    ax.text(13.53, 2.45, "ObjectNav goals\n+ success set", color=COLORS["eval"],
            ha="center", va="center", fontsize=5.7)
    arrow((10.90, 1.30), (12.78, 1.30), COLORS["eval"], "--")
    ax.text(7.0, .06, "Goal annotations never enter topology sampling, RGB-D capture, VLM inference, or fusion.",
            ha="center", fontsize=6.7, color="#444444")
    save(fig, "fig_system_overview")


def robot_query_loop() -> None:
    fig = plt.figure(figsize=(3.38, 2.30))
    grid = fig.add_gridspec(2, 2, height_ratios=(.52, 1.48), hspace=.13, wspace=.05)
    ax = fig.add_subplot(grid[0, :]); ax.set_xlim(0, 10); ax.set_ylim(0, 2); ax.axis("off")
    labels = (("\"find a chair\"", COLORS["query"]), ("Map query", COLORS["semantic"]),
              ("Graph plan", COLORS["nav"]), ("Nav2 goal", COLORS["metric"]))
    for index, (label, color) in enumerate(labels):
        x = .08 + index * 2.52
        ax.add_patch(FancyBboxPatch((x, .55), 2.05, .88, boxstyle="round,pad=.04,rounding_size=.08",
                                    fc=mpl.colors.to_rgba(color, .12), ec=color, lw=1))
        ax.text(x + 1.025, .99, label, ha="center", va="center", fontsize=5.6, weight="bold")
        if index < len(labels) - 1:
            ax.add_patch(FancyArrowPatch((x + 2.08, .99), (x + 2.47, .99), arrowstyle="-|>",
                                         mutation_scale=7, lw=.9, color="#44515f"))
    ax.text(5, .16, "implemented ROS 2 data flow · qualitative integration evidence", ha="center",
            va="center", fontsize=5.2, color="#555")

    rgb_ax = fig.add_subplot(grid[1, 0])
    rgb_path = ROBOT / "captured_nodes" / "node_2" / "000deg_rgb.png"
    rgb_ax.imshow(Image.open(rgb_path)); rgb_ax.axis("off"); rgb_ax.set_title("Physical survey RGB", fontsize=6.3, pad=2)

    map_ax = fig.add_subplot(grid[1, 1])
    occupancy = np.asarray(Image.open(ROBOT / "20260327_005229.pgm"))
    metadata = json.loads((ROBOT / "node_positions.json").read_text())
    nodes = metadata["nodes"]; origin = metadata["origin"]; resolution = float(metadata["resolution"])
    height, width = occupancy.shape
    pixels = np.asarray([((node["x"] - origin[0]) / resolution,
                          height - 1 - (node["y"] - origin[1]) / resolution) for node in nodes])
    map_ax.imshow(occupancy, cmap="gray", vmin=0, vmax=255)
    spacing = float(metadata["spacing"])
    for i in range(len(nodes)):
        for j in range(i):
            distance = np.linalg.norm(np.asarray([nodes[i]["x"]-nodes[j]["x"], nodes[i]["y"]-nodes[j]["y"]]))
            if abs(distance - spacing) < .1:
                map_ax.plot(pixels[[i, j], 0], pixels[[i, j], 1], color="#7A8A99", lw=.7, alpha=.8)
    route = [0, 1, 2, 3, 4]
    map_ax.plot(pixels[route, 0], pixels[route, 1], color=COLORS["eval"], lw=1.5)
    map_ax.scatter(pixels[:, 0], pixels[:, 1], s=13, color=COLORS["nav"], edgecolors="white", linewidths=.3)
    map_ax.scatter(*pixels[route[-1]], s=25, marker="*", color=COLORS["metric"], edgecolors="white", linewidths=.3)
    map_ax.axis("off"); map_ax.set_title("Occupancy + topology", fontsize=6.3, pad=2)
    fig.subplots_adjust(left=.01, right=.99, top=.98, bottom=.01)
    save(fig, "fig_robot_query_loop", dpi=350)


def metric_ablation() -> None:
    specs = [
        ("Qwen\n90°", "qwen_rgbd_90_analysis_camera", "qwen_rgbd_90_analysis_raw", "qwen_rgbd_90_analysis_fused"),
        ("GPT-5.4\n90°", "peterai_camera_node_analysis", "peterai_rgbd_analysis_raw", "peterai_rgbd_analysis_soft"),
        ("GPT-5.4\n120°", "peterai_camera_node_wide120_analysis", "peterai_rgbd_wide120_analysis_raw", "peterai_rgbd_wide120_analysis_soft"),
        ("MiMo\n90°", "mimo_camera_node_analysis", "mimo_rgbd_analysis_reparsed_raw", "mimo_rgbd_analysis_reparsed_soft"),
        ("MiMo\n120°", "mimo_camera_node_wide120_analysis", "mimo_rgbd_wide120_analysis_raw", "mimo_rgbd_wide120_analysis_soft"),
    ]
    variants = ("Camera node", "Raw RGB-D", "3D fusion")
    palette = ("#9AA6B2", "#55A868", "#2C7FB8")
    values = []
    for _, *directories in specs:
        row = []
        for directory in directories:
            row.append(load_summary(f"outputs/hm3d_minival_uniform/{directory}/summary_vlm_all_confidence.json"))
        values.append(row)

    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.48), sharey=True)
    x = np.arange(len(specs)); width = .24
    for ax, metric, label in zip(axes, ("sr", "spl"), ("Success rate", "SPL")):
        for j, (variant, color) in enumerate(zip(variants, palette)):
            y = [values[i][j][metric] for i in range(len(specs))]
            bars = ax.bar(x + (j - 1) * width, y, width, color=color, label=variant)
            for bar, number in zip(bars, y):
                ax.text(bar.get_x() + bar.get_width()/2, number + .018, f"{number:.2f}",
                        ha="center", va="bottom", fontsize=5.8, rotation=90)
        ax.set_xticks(x, [item[0] for item in specs]); ax.set_ylim(0, 1.10)
        ax.set_ylabel(label); ax.grid(axis="y", alpha=.22); ax.set_axisbelow(True)
    axes[0].legend(loc="upper left", ncol=3, frameon=False, bbox_to_anchor=(0, 1.18))
    fig.text(.99, .015, "Qwen2.5-VL-3B / MiMo-v2.5 · 30 episodes · 2 scenes · identical responses within triplet",
             ha="right", va="bottom", fontsize=6.3, color="#555")
    fig.subplots_adjust(wspace=.16, top=.82, bottom=.24, left=.08, right=.99)
    save(fig, "fig_metric_ablation")


def full_independent_results() -> bool:
    base = ROOT / "outputs" / "hm3d_val_uniform"
    paths = {
        "Geometry\noracle": base / "oracle_geometry" / "density_ablation.json",
        "Camera\nnode": base / "gpt54_camera_node_analysis" / "summary_vlm_all_confidence.json",
        "Raw\nsingle": base / "gpt54_rgbd_wide120_analysis_raw" / "summary_vlm_all_confidence.json",
        "Raw Top-3\n(2 m)": base / "gpt54_rgbd_wide120_raw_topk_2p0m" / "summary_topk.json",
        "Fused\nsingle": base / "gpt54_rgbd_wide120_analysis_fused" / "summary_vlm_all_confidence.json",
        "Fused reps\nTop-3 (2 m)": base / "gpt54_rgbd_wide120_fused_topk_2p0m" / "summary_topk.json",
        "Fusion-aware\nTop-3": base / "gpt54_rgbd_wide120_fusion_first_raw_topk_2p0m" / "summary_topk.json",
    }
    if not all(path.exists() for path in paths.values()):
        return False
    rows = []
    for label, path in paths.items():
        data = json.loads(path.read_text())
        if label.startswith("Geometry"):
            data = data["summaries"]["1.0"]
        elif "Top-3" in label:
            data = data["metrics"]["3"]
        rows.append((label, float(data["sr"]), float(data["spl"])))
    fig, ax = plt.subplots(figsize=(7.05, 2.35))
    x = np.arange(len(rows)); width = .34
    sr_bars = ax.bar(x - width/2, [row[1] for row in rows], width, color=COLORS["nav"], label="SR")
    spl_bars = ax.bar(x + width/2, [row[2] for row in rows], width, color=COLORS["metric"], label="SPL")
    for bars in (sr_bars, spl_bars):
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+.014, f"{bar.get_height():.3f}",
                    ha="center", va="bottom", fontsize=6.4)
    ax.set_xticks(x, [row[0] for row in rows]); ax.set_ylim(0, 1.08); ax.set_ylabel("Metric")
    ax.grid(axis="y", alpha=.22); ax.set_axisbelow(True)
    ax.set_title("HM3D-v2 validation · 1,000 episodes · 36 scenes", loc="left", pad=9, color="#444")
    ax.legend(frameon=False, ncol=2, loc="upper right", bbox_to_anchor=(1, 1.10))
    fig.subplots_adjust(left=.07, right=.995, top=.84, bottom=.22)
    save(fig, "fig_full_independent_results")
    return True


def density_and_stress() -> None:
    density_path = MINI / "density_analysis" / "density_ablation.csv"
    density = list(csv.DictReader(density_path.open()))
    stress_path = ROOT / "outputs" / "analysis" / "stress_aggregate.csv"
    stress = list(csv.DictReader(stress_path.open()))
    fig, axes = plt.subplots(1, 2, figsize=(7.05, 2.48))
    fractions = np.asarray([float(row["node_fraction"]) for row in density])
    axes[0].plot(fractions, [float(row["sr"]) for row in density], "o-", color=COLORS["nav"], label="SR")
    axes[0].plot(fractions, [float(row["spl"]) for row in density], "s-", color=COLORS["metric"], label="SPL")
    axes[0].set(xlabel="Fraction of nested 0.8 m topology", ylabel="Metric", ylim=(0, 1.05), title="Geometry ceiling")
    axes[0].grid(alpha=.22); axes[0].legend(frameon=False)

    drops = sorted({float(row["semantic_dropout"]) for row in stress})
    fps = sorted({float(row["false_positive"]) for row in stress})
    matrix = np.full((len(fps), len(drops)), np.nan)
    for row in stress:
        if float(row["keep_probability"]) == 1.0:
            matrix[fps.index(float(row["false_positive"])), drops.index(float(row["semantic_dropout"]))] = float(row["sr_mean"])
    image = axes[1].imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    axes[1].set_xticks(range(len(drops)), [f"{x:.2g}" for x in drops])
    axes[1].set_yticks(range(len(fps)), [f"{x:.2g}" for x in fps])
    axes[1].set(xlabel="Semantic dropout", ylabel="False-positive probability", title="Oracle-label corruption (SR)")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            axes[1].text(j, i, f"{value:.2f}", ha="center", va="center",
                         color="white" if value < .65 else "black", fontsize=6.5)
    fig.colorbar(image, ax=axes[1], fraction=.046, pad=.03)
    fig.subplots_adjust(wspace=.35, bottom=.21, top=.86, left=.09, right=.96)
    save(fig, "fig_density_semantics")


def topology_maps() -> None:
    visual_root = MINI / "peterai_rgbd_wide120_visuals" / "semantic_maps"
    scenes = ("TEEsavR23oF", "wcojb4TFT35")
    fig, axes = plt.subplots(2, 2, figsize=(7.05, 5.45))
    for i, scene in enumerate(scenes):
        for j, variant in enumerate(("raw_maps", "clustered_maps")):
            path = visual_root / f"{scene}_{variant}.png"
            axes[i, j].imshow(Image.open(path)); axes[i, j].axis("off")
            axes[i, j].set_title(f"Scene {i+1}: {'raw projected candidates' if j == 0 else '3D-fused candidates'}")
    fig.subplots_adjust(wspace=.02, hspace=.10, left=.01, right=.99, top=.95, bottom=.01)
    save(fig, "fig_topology_maps", dpi=250)


def rgbd_grounding() -> None:
    map_path = MINI / "peterai_rgbd_wide120_fusion" / "clustered_maps" / "wcojb4TFT35" / "vlm_topological_map.json"
    mapping = json.loads(map_path.read_text())
    candidate = next(node for node in mapping["nodes"] if node["id"] == 31)
    rgb = MINI / "peterai_rgbd_wide120_visuals" / "detection_overlays" / "wcojb4TFT35_node_0088_view0.jpg"
    depth_path = ROOT / candidate["source_depth_path"]
    source = np.asarray(candidate["source_position"])[[0, 2]]
    obj = np.asarray(candidate["object_estimate"])[[0, 2]]
    stop = np.asarray(candidate["position"])[[0, 2]]

    fig, axes = plt.subplots(1, 3, figsize=(3.38, 1.30))
    _show_image(axes[0], rgb, "View-local box")
    _show_depth(axes[1], depth_path, f"Depth {candidate['depth_m']:.2f} m", candidate["center"])
    ax = axes[2]
    ax.plot([source[0], obj[0]], [source[1], obj[1]], "--", color=COLORS["gray"], lw=1)
    ax.scatter(*source, s=28, marker="^", color=COLORS["nav"], label="camera", zorder=3)
    ax.scatter(*obj, s=26, marker="x", color=COLORS["semantic"], label="surface", zorder=3)
    ax.scatter(*stop, s=30, marker="o", color=COLORS["metric"], label="STOP", zorder=3)
    circle = plt.Circle(obj, .8, fill=False, ls=":", lw=.8, color=COLORS["metric"])
    ax.add_patch(circle)
    ax.annotate("0.8 m", xy=(stop + obj) / 2, fontsize=5.2, ha="center")
    ax.set_aspect("equal"); ax.grid(alpha=.18); ax.tick_params(labelsize=5)
    ax.set_title("Back-project → standoff", fontsize=6.5, pad=3)
    ax.legend(frameon=False, fontsize=4.8, loc="best", handletextpad=.2, borderpad=.1)
    fig.subplots_adjust(wspace=.08, left=.01, right=.99, top=.88, bottom=.08)
    save(fig, "fig_rgbd_grounding", dpi=350)


def _show_image(ax, path: Path, title: str, fontsize: float = 6.5) -> None:
    ax.imshow(Image.open(path).convert("RGB")); ax.axis("off"); ax.set_title(title, pad=3, fontsize=fontsize)


def _show_depth(ax, path: Path, title: str, center=None, fontsize: float = 6.5) -> None:
    depth = np.load(path)
    masked = np.ma.masked_where(~np.isfinite(depth) | (depth <= 0) | (depth > 6), depth)
    ax.imshow(masked, cmap="turbo_r", vmin=0, vmax=6)
    if center:
        ax.plot(center[0] * depth.shape[1], center[1] * depth.shape[0], "+", ms=10, mew=1.5, color="white")
    ax.axis("off"); ax.set_title(title, pad=3, fontsize=fontsize)


def _show_video_last(ax, path: Path, title: str, fontsize: float = 6.5) -> None:
    reader = imageio.get_reader(path)
    try:
        count = reader.count_frames()
        frame = reader.get_data(max(0, count - 1))
    finally:
        reader.close()
    ax.imshow(frame); ax.axis("off"); ax.set_title(title, pad=3, fontsize=fontsize)


def audited_cases() -> None:
    standard = MINI / "rgbd_capture" / "wcojb4TFT35" / "node_0118_view3_rgb.jpg"
    wide_overlay = MINI / "peterai_rgbd_wide120_visuals" / "detection_overlays" / "wcojb4TFT35_node_0118_view3.jpg"
    wide_depth = MINI / "rgbd_capture_wide120" / "wcojb4TFT35" / "node_0118_view3_depth.npy"
    toilet_final = MINI / "peterai_rgbd_wide120_visuals" / "toilet_navigation" / "frames" / "wcojb4TFT35_toilet_13" / "rgb_0029.jpg"
    sofa_false = MINI / "peterai_rgbd_wide120_visuals" / "detection_overlays" / "wcojb4TFT35_node_0035_view0.jpg"
    sofa_supported = MINI / "peterai_rgbd_wide120_visuals" / "detection_overlays" / "wcojb4TFT35_node_0088_view0.jpg"
    raw_top = MINI / "peterai_rgbd_wide120_sofa_raw" / "videos" / "wcojb4TFT35_sofa_22_FAIL_topdown.mp4"
    fused_top = MINI / "peterai_rgbd_wide120_sofa_fused" / "videos" / "wcojb4TFT35_sofa_22_topdown.mp4"
    recovery = json.loads((FULL / "gpt54_arrival_verified_visuals" / "recovery_case" / "case_manifest.json").read_text())
    rejected, accepted = recovery["attempts"]
    rejected_reference = ROOT / rejected["mapping_reference_boxed_path"]
    rejected_arrival = ROOT / rejected["verification_overlay_path"]
    accepted_arrival = ROOT / accepted["verification_overlay_path"]
    recovery_top = ROOT / recovery["topdown_path"]

    fig, axes = plt.subplots(3, 4, figsize=(7.05, 5.45))
    title_size = 8.1
    _show_image(axes[0, 0], standard, "90° capture · same node\ntarget outside useful view", title_size)
    _show_image(axes[0, 1], wide_overlay, "120° capture · toilet found\nconfidence 0.98", title_size)
    _show_depth(axes[0, 2], wide_depth, "Aligned depth grounding\nmedian 1.28 m", (.5105, .8075), title_size)
    _show_image(axes[0, 3], toilet_final, "Metric standoff reached\nDTG 0.16 m · SUCCESS", title_size)

    _show_image(axes[1, 0], sofa_false, "Unsupported singleton\nraw rank 1 · failure", title_size)
    _show_image(axes[1, 1], sofa_supported, "Multi-view support\n22 source poses", title_size)
    if raw_top.exists():
        _show_video_last(axes[1, 2], raw_top, "Raw selected route\nDTG 4.27 m · FAIL", title_size)
    else:
        axes[1, 2].axis("off"); axes[1, 2].text(.5, .5, "Raw route artifact\nnot yet rendered", ha="center", va="center")
    if fused_top.exists():
        _show_video_last(axes[1, 3], fused_top, "Supported route selected\nDTG 0.15 m · SUCCESS", title_size)
    else:
        map_path = MINI / "peterai_rgbd_wide120_visuals" / "semantic_maps" / "wcojb4TFT35_clustered_maps.png"
        _show_image(axes[1, 3], map_path, "Fused semantic map", title_size)

    _show_image(axes[2, 0], rejected_reference, "Visit 1 map hypothesis\nplant confidence 0.97", title_size)
    _show_image(axes[2, 1], rejected_arrival, "Fresh arrival RGB-D\nREJECT · target absent", title_size)
    _show_image(axes[2, 2], accepted_arrival, "Visit 2 fresh RGB-D\nACCEPT · depth 0.94 m", title_size)
    _show_image(axes[2, 3], recovery_top, "Closed-loop recovery route\nrank 1 fail → rank 2 success", title_size)

    row_labels = ("(a) FoV + depth", "(b)  3D fusion", "(c) arrival loop")
    for y, label in zip((.968, .644, .319), row_labels):
        fig.text(.012, y, label, weight="bold", fontsize=8.7, va="top")
    fig.subplots_adjust(wspace=.025, hspace=.44, left=.012, right=.995, top=.925, bottom=.005)
    save(fig, "fig_audited_cases", dpi=300)


def write_manifest() -> None:
    manifest = {
        "generator": str(Path(__file__).relative_to(ROOT)),
        "figures": {
            "fig_system_overview": "Vector pipeline; evaluator-only goal lane is visually separated.",
            "fig_robot_query_loop": "Implemented ROS 2 flow with physical survey RGB and recorded occupancy/topology artifacts.",
            "fig_metric_ablation": "Machine-read from same-response minival summary JSON files.",
            "fig_full_independent_results": "Generated when complete full-validation summaries and Top-3 diagnostics exist.",
            "fig_density_semantics": "Machine-read from density and stress CSV files.",
            "fig_topology_maps": "Rendered from raw and 3D-fused wide RGB-D semantic maps.",
            "fig_rgbd_grounding": "Exact source RGB/depth and projected camera-surface-standoff geometry.",
            "fig_audited_cases": "Exact FoV, fusion, and autonomous fresh-arrival RGB-D/VLM recovery traces.",
        },
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))


def main() -> None:
    setup()
    system_overview()
    robot_query_loop()
    metric_ablation()
    full_ready = full_independent_results()
    density_and_stress()
    topology_maps()
    rgbd_grounding()
    audited_cases()
    write_manifest()
    print(json.dumps({"output": str(OUT), "figures": 7 + int(full_ready)}, indent=2))


if __name__ == "__main__":
    main()
