"""Audit visualizations for view-local RGB-D semantic mapping."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from .benchmark import draw_map, load_scene_episodes, make_sim, scene_paths
from .map_artifacts import contact_sheet


COLORS = {
    "chair": "#ef5350", "bed": "#5c6bc0", "plant": "#66bb6a",
    "toilet": "#26c6da", "tv_monitor": "#ffa726", "sofa": "#ab47bc",
}


def draw_detections(image_path, detections, output_path):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for detection in detections:
        bbox_norm = detection.get("bbox_norm")
        if bbox_norm:
            bbox = [
                bbox_norm[0] * width, bbox_norm[1] * height,
                bbox_norm[2] * width, bbox_norm[3] * height,
            ]
        else:
            bbox = detection["bbox_2d"]
        color = COLORS.get(detection["category"], "white")
        draw.rectangle(bbox, outline=color, width=4)
        label = f"{detection['category']} {detection['confidence']:.2f}"
        text_box = draw.textbbox((bbox[0], bbox[1]), label)
        draw.rectangle((text_box[0] - 2, text_box[1] - 2, text_box[2] + 2, text_box[3] + 2), fill="black")
        draw.text((bbox[0], bbox[1]), label, fill=color)
    image.save(output_path, quality=92)


def depth_pair(rgb_path, depth_path, detections, output_path, max_depth=6.0):
    rgb = Image.open(rgb_path).convert("RGB")
    depth = np.load(depth_path)
    valid = np.isfinite(depth) & (depth > 0)
    scaled = np.zeros_like(depth, dtype=np.float32)
    scaled[valid] = 1 - np.clip(depth[valid], 0, max_depth) / max_depth
    # A compact blue-to-yellow map without adding another plotting dependency.
    color = np.stack((scaled, np.sqrt(scaled), 1 - scaled), axis=-1)
    depth_image = Image.fromarray(np.uint8(np.clip(color, 0, 1) * 255))
    canvas = Image.new("RGB", (rgb.width * 2, rgb.height))
    canvas.paste(rgb, (0, 0)); canvas.paste(depth_image, (rgb.width, 0))
    draw = ImageDraw.Draw(canvas)
    for detection in detections:
        center = detection["center"]
        x, y = center[0] * rgb.width, center[1] * rgb.height
        category = detection["category"]; color_name = COLORS.get(category, "white")
        for offset in (0, rgb.width):
            draw.ellipse((x + offset - 6, y - 6, x + offset + 6, y + 6), outline=color_name, width=3)
        value = depth[int(np.clip(round(y), 0, depth.shape[0]-1)), int(np.clip(round(x), 0, depth.shape[1]-1))]
        draw.text((x + rgb.width + 9, y), f"{category} {value:.2f}m", fill=color_name, stroke_width=2, stroke_fill="black")
    canvas.save(output_path, quality=92)


def run(args):
    root = Path(args.root).resolve()
    semantics = root / args.semantics
    fusion = root / args.fusion
    output = root / args.output
    overlays = output / "detection_overlays"
    depth_previews = output / "rgb_depth_pairs"
    map_previews = output / "semantic_maps"
    for directory in (output, overlays, depth_previews, map_previews):
        directory.mkdir(parents=True, exist_ok=True)

    detection_rows, overlay_gallery, depth_gallery = [], [], []
    for semantic_path in sorted(semantics.glob("*/rgbd_semantic_map.json")):
        scene = semantic_path.parent.name
        mapping = json.loads(semantic_path.read_text())
        for node in mapping["nodes"]:
            grouped = {}
            for detection in node.get("localized_vlm", {}).get("detections", []):
                grouped.setdefault(int(detection["view_index"]), []).append(detection)
            for view_index, detections in grouped.items():
                view = node["rgbd_views"][view_index]
                rgb_path = root / view["rgb_path"]
                overlay_path = overlays / f"{scene}_node_{int(node['id']):04d}_view{view_index}.jpg"
                draw_detections(rgb_path, detections, overlay_path)
                confidence = max(item["confidence"] for item in detections)
                overlay_gallery.append((confidence, f"{scene} n{node['id']} v{view_index}", overlay_path))
                if len(depth_gallery) < args.depth_examples:
                    pair_path = depth_previews / overlay_path.name
                    depth_pair(rgb_path, root / view["depth_path"], detections, pair_path)
                    depth_gallery.append((f"{scene} n{node['id']} v{view_index}", pair_path))
                for detection in detections:
                    p, q = node["position"], view["rotation"]
                    detection_rows.append({
                        "scene": scene, "node_id": node["id"], "view_index": view_index,
                        "category": detection["category"], "confidence": detection["confidence"],
                        "bbox_norm": json.dumps(detection.get("bbox_norm")), "center": json.dumps(detection["center"]),
                        "x": p[0], "y": p[1], "z": p[2], "qx": q[0], "qy": q[1], "qz": q[2], "qw": q[3],
                        "rgb_path": view["rgb_path"], "depth_path": view["depth_path"],
                        "overlay_path": str(overlay_path.relative_to(root)),
                    })
    if detection_rows:
        with (output / "detection_pose_manifest.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=detection_rows[0].keys())
            writer.writeheader(); writer.writerows(detection_rows)
    overlay_gallery.sort(reverse=True, key=lambda item: item[0])
    contact_sheet([(label, path) for _, label, path in overlay_gallery[:args.gallery_size]], output / "detection_contact_sheet.jpg", columns=4, size=(400, 270))
    contact_sheet(depth_gallery, output / "rgb_depth_contact_sheet.jpg", columns=3, size=(660, 240))

    candidate_rows, map_gallery = [], []
    dataset = root / "data/datasets/objectnav_hm3d_v2" / args.split
    scenes = root / "data/hm3d" / args.scene_dir
    scene_data = {}
    for _, data in load_scene_episodes(dataset):
        _, scene, base, nav = scene_paths(scenes, data["episodes"][0]["scene_id"])
        scene_data[scene] = (base, nav)
    for variant in ("raw_maps", "clustered_maps"):
        for map_path in sorted((fusion / variant).glob("*/vlm_topological_map.json")):
            scene = map_path.parent.name
            mapping = json.loads(map_path.read_text())
            sim = make_sim(*scene_data[scene])
            preview = draw_map(sim, mapping["nodes"], title=f"{scene}: RGB-D {variant.replace('_maps','')}")
            sim.close()
            preview_path = map_previews / f"{scene}_{variant}.png"
            imageio.imwrite(preview_path, preview); map_gallery.append((f"{scene} {variant}", preview_path))
            for node in mapping["nodes"]:
                if variant != "raw_maps":
                    continue
                candidate_rows.append({
                    "scene": scene, "candidate_id": node["id"], "category": node["category"],
                    "confidence": node["confidence"], "source_topology_node": node["source_topology_node"],
                    "view_index": node["view_index"], "depth_m": node["depth_m"],
                    "source_position": json.dumps(node["source_position"]),
                    "object_estimate": json.dumps(node["object_estimate"]),
                    "navigation_stop": json.dumps(node["position"]),
                    "rgb_path": node.get("source_rgb_path", ""), "depth_path": node.get("source_depth_path", ""),
                })
    if candidate_rows:
        with (output / "projected_candidates.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=candidate_rows[0].keys())
            writer.writeheader(); writer.writerows(candidate_rows)
    contact_sheet(map_gallery, output / "semantic_map_contact_sheet.jpg", columns=2, size=(500, 440))
    report = {
        "detections": len(detection_rows), "overlay_images": len(overlay_gallery),
        "depth_examples": len(depth_gallery), "projected_candidates": len(candidate_rows),
        "semantic_maps": len(map_gallery),
    }
    (output / "visual_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--split", default="val_mini")
    parser.add_argument("--scene-dir", default="minival")
    parser.add_argument("--semantics", default="outputs/hm3d_minival_uniform/peterai_rgbd_semantics")
    parser.add_argument("--fusion", default="outputs/hm3d_minival_uniform/peterai_rgbd_fusion")
    parser.add_argument("--output", default="outputs/hm3d_minival_uniform/peterai_rgbd_visuals")
    parser.add_argument("--gallery-size", type=int, default=48)
    parser.add_argument("--depth-examples", type=int, default=18)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
