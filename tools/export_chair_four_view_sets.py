"""Render clear four-view RGB/depth PNG sets for figure construction.

Each entry in ``GROUPS`` uses one fixed topology-node position and four
cardinal camera offsets.  RGB is written directly from Habitat observations as
PNG; depth is retained as meters in ``.npy`` and exported as a consistent
turbo-r color visualization for drawing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sstg_bench.benchmark import set_pose
from sstg_bench.rgbd_capture import make_rgbd_sim
from sstg_bench.vlm_map import yaw_quaternion


ROOT = Path(__file__).resolve().parents[1]
WIDTH = 640
HEIGHT = 640
HFOV = 120.0
CAMERA_HEIGHT_M = 1.25
YAW_OFFSETS = (0, 90, 180, 270)
DEPTH_MAX_M = 6.0

# These are deliberately diverse and were visually checked before rendering.
GROUPS = (
    {
        "name": "set_01_blue_armchair_lounge",
        "split": "val",
        "scene": "LT9Jq6dN3Ea",
        "node": 246,
        "description": "single blue armchair visible in a bright lounge",
    },
    {
        "name": "set_02_single_upholstered_chair",
        "split": "val",
        "scene": "5cdEh9F2hJL",
        "node": 24,
        "description": "single upholstered chair in a blue bedroom",
    },
    {
        "name": "set_03_brown_armchair_bedroom",
        "split": "val",
        "scene": "bCPU9suPUw9",
        "node": 76,
        "description": "single brown armchair beside the bedroom beds",
    },
    {
        "name": "set_04_pattern_armchair_lounge",
        "split": "val",
        "scene": "cvZr5TUy5C5",
        "node": 283,
        "description": "single patterned armchair in a sunlit lounge",
    },
    {
        "name": "set_05_wingback_armchair",
        "split": "val",
        "scene": "k1cupFYWXJ6",
        "node": 302,
        "description": "single wingback armchair in a red sitting room",
    },
    {
        "name": "set_06_office_chair_minival",
        "split": "minival",
        "scene": "TEEsavR23oF",
        "node": 30,
        "description": "single office chair in the minival living area",
    },
)


def _scene_files(scene_dir: Path, scene: str) -> tuple[Path, Path]:
    matches = sorted(scene_dir.glob(f"*-{scene}"))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one scene directory for {scene}, found {matches}")
    directory = matches[0]
    return directory / f"{scene}.basis.glb", directory / f"{scene}.basis.navmesh"


def _source_map(root: Path, group: dict) -> Path:
    if group["split"] == "minival":
        return root / "outputs/hm3d_minival_uniform/rgbd_capture_wide120" / group["scene"] / "rgbd_topological_map.json"
    return root / "outputs/hm3d_val_uniform/rgbd_capture_wide120" / group["scene"] / "rgbd_topological_map.json"


def _candidate_map(root: Path, group: dict) -> Path:
    if group["split"] == "minival":
        return root / "outputs/hm3d_minival_uniform/peterai_rgbd_wide120_fusion/clustered_maps" / group["scene"] / "vlm_topological_map.json"
    return root / "outputs/hm3d_val_uniform/gpt54_rgbd_wide120_fusion/clustered_maps" / group["scene"] / "vlm_topological_map.json"


def _font(size: int, bold: bool = False):
    names = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    path = names[0 if bold else 1]
    return ImageFont.truetype(path, size=size) if Path(path).exists() else ImageFont.load_default()


def _depth_color(depth: np.ndarray) -> Image.Image:
    valid = np.isfinite(depth) & (depth > 0) & (depth <= DEPTH_MAX_M)
    normalized = np.clip(depth / DEPTH_MAX_M, 0.0, 1.0)
    rgb = (matplotlib.colormaps["turbo_r"](normalized)[..., :3] * 255).astype(np.uint8)
    rgb[~valid] = 0
    return Image.fromarray(rgb, mode="RGB")


def _strip(images: list[Image.Image], labels: list[str], *, title: str | None = None) -> Image.Image:
    label_h = 46
    title_h = 44 if title else 0
    canvas = Image.new("RGB", (WIDTH * 4, HEIGHT + label_h + title_h), "white")
    draw = ImageDraw.Draw(canvas)
    if title:
        draw.text((18, 10), title, fill="black", font=_font(26, bold=True))
    for index, image in enumerate(images):
        x = index * WIDTH
        canvas.paste(image.convert("RGB"), (x, title_h + label_h))
        draw.text((x + 12, title_h + 11), labels[index], fill="black", font=_font(22, bold=True))
    return canvas


def _grid(rgb: list[Image.Image], depth: list[Image.Image], *, title: str) -> Image.Image:
    label_h = 42
    title_h = 46
    row_h = HEIGHT + label_h
    canvas = Image.new("RGB", (WIDTH * 4, title_h + row_h * 2), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 8), title, fill="black", font=_font(26, bold=True))
    for row, images, row_name in ((0, rgb, "RGB"), (1, depth, "Colorized depth")):
        y = title_h + row * row_h
        for index, image in enumerate(images):
            x = index * WIDTH
            canvas.paste(image.convert("RGB"), (x, y + label_h))
            draw.text(
                (x + 12, y + 10),
                f"{row_name} · view {index} · yaw {YAW_OFFSETS[index]}°",
                fill="black",
                font=_font(21, bold=True),
            )
    return canvas


def _write_depth_legend(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(1.0, 4.0), dpi=180)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=DEPTH_MAX_M)
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=matplotlib.colormaps["turbo_r"]), cax=ax, label="Depth (m)")
    fig.savefig(out / "depth_color_legend.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _write_selection_overview(output: Path, records: list[dict]) -> None:
    rows = []
    for record in records:
        strip = Image.open(output / record["name"] / "rgb_four_view_strip.png").convert("RGB")
        rows.append(strip.resize((strip.width // 2, strip.height // 2), Image.Resampling.LANCZOS))
    canvas = Image.new("RGB", (max(row.width for row in rows), sum(row.height for row in rows)), "white")
    y = 0
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height
    canvas.save(output / "selection_overview_rgb.png", format="PNG", optimize=True)


def render_group(root: Path, output_root: Path, group: dict) -> dict:
    scene = group["scene"]
    split = group["split"]
    source_map_path = _source_map(root, group)
    source_map = json.loads(source_map_path.read_text())
    node = next(item for item in source_map["nodes"] if int(item["id"]) == int(group["node"]))
    scene_glb, navmesh = _scene_files(root / "data/hm3d" / split, scene)
    out = output_root / group["name"]
    out.mkdir(parents=True, exist_ok=True)

    candidate_records = []
    candidate_path = _candidate_map(root, group)
    if candidate_path.exists():
        candidates = json.loads(candidate_path.read_text()).get("nodes", [])
        candidate_records = [
            item for item in candidates
            if item.get("category") == "chair" and int(item.get("source_topology_node", -1)) == int(group["node"])
        ]

    sim = make_rgbd_sim(scene_glb, navmesh, WIDTH, HEIGHT, HFOV)
    rgb_images: list[Image.Image] = []
    depth_images: list[Image.Image] = []
    views = []
    try:
        for view_index, yaw in enumerate(YAW_OFFSETS):
            rotation = yaw_quaternion(node.get("rotation", [0, 0, 0, 1]), yaw)
            set_pose(sim, node["position"], rotation)
            observations = sim.get_sensor_observations()
            rgb = np.asarray(observations["rgb"])[..., :3].astype(np.uint8)
            depth = np.asarray(observations["depth"], dtype=np.float32)
            rgb_image = Image.fromarray(rgb, mode="RGB")
            depth_image = _depth_color(depth)
            rgb_path = out / f"rgb_view{view_index}_yaw{yaw:03d}.png"
            depth_path = out / f"depth_view{view_index}_yaw{yaw:03d}.png"
            raw_depth_path = out / f"depth_view{view_index}_yaw{yaw:03d}_meters.npy"
            rgb_image.save(rgb_path, format="PNG", optimize=True)
            depth_image.save(depth_path, format="PNG", optimize=True)
            np.save(raw_depth_path, depth)
            rgb_images.append(rgb_image)
            depth_images.append(depth_image)
            valid = depth[np.isfinite(depth) & (depth > 0)]
            views.append(
                {
                    "view_index": view_index,
                    "yaw_offset_deg": yaw,
                    "rotation": rotation,
                    "rgb_png": str(rgb_path.relative_to(root)),
                    "depth_color_png": str(depth_path.relative_to(root)),
                    "depth_meters_npy": str(raw_depth_path.relative_to(root)),
                    "valid_depth_min_m": float(np.min(valid)) if valid.size else None,
                    "valid_depth_max_m": float(np.max(valid)) if valid.size else None,
                }
            )
    finally:
        sim.close()

    labels = [f"view {i} · yaw {yaw}°" for i, yaw in enumerate(YAW_OFFSETS)]
    _strip(rgb_images, labels, title=f"{group['name']} · RGB").save(out / "rgb_four_view_strip.png", format="PNG", optimize=True)
    _strip(depth_images, labels, title=f"{group['name']} · colorized depth").save(out / "depth_four_view_strip.png", format="PNG", optimize=True)
    _grid(rgb_images, depth_images, title=f"{group['description']} · fixed position, four cardinal views").save(out / "rgb_depth_four_view_grid.png", format="PNG", optimize=True)

    metadata = {
        "group": group,
        "scene": scene,
        "split": split,
        "node_id": int(node["id"]),
        "position_world": node["position"],
        "base_rotation": node.get("rotation", [0, 0, 0, 1]),
        "camera": {"width": WIDTH, "height": HEIGHT, "hfov_deg": HFOV, "height_m": CAMERA_HEIGHT_M},
        "views": views,
        "candidate_evidence": candidate_records,
        "source_topology_map": str(source_map_path.relative_to(root)),
        "scene_glb": str(scene_glb.relative_to(root)),
        "navmesh": str(navmesh.relative_to(root)),
        "depth_visualization": {"colormap": "turbo_r", "range_m": [0.0, DEPTH_MAX_M], "invalid_pixels": "black"},
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    return {"name": group["name"], "scene": scene, "node": node["id"], "path": str(out.relative_to(root)), "candidate_evidence": len(candidate_records)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/chair_four_view_sets")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    output.mkdir(parents=True, exist_ok=True)
    _write_depth_legend(output)
    records = [render_group(root, output, group) for group in GROUPS]
    _write_selection_overview(output, records)
    manifest = {
        "description": "Habitat renders from one fixed topology position at four cardinal yaw offsets.",
        "camera": {"width": WIDTH, "height": HEIGHT, "hfov_deg": HFOV, "height_m": CAMERA_HEIGHT_M},
        "yaw_offsets_deg": list(YAW_OFFSETS),
        "depth_colormap": {"name": "turbo_r", "range_m": [0.0, DEPTH_MAX_M], "invalid": "black"},
        "groups": records,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
