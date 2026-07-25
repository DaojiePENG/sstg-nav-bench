"""Capture fresh target-oriented RGB-D observations for Top-K candidates.

The candidate list is formed from the semantic map and episode query only.
ObjectNav goal annotations are never inspected.  Each unique candidate is
captured once and can be reused by episodes that visit the same hypothesis.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from .benchmark import load_scene_episodes, scene_paths, set_pose, shortest, yaw_quat
from .experiments import load_nodes
from .rgbd_capture import make_rgbd_sim
from .topk import RANKING_STRATEGIES, hierarchical_candidates, rank_nodes
from .vlm_map import yaw_quaternion


def ranked_candidates(
    sim,
    start,
    nodes,
    category,
    max_k,
    min_separation,
    ranking_strategy,
    primary_nodes=None,
    primary_k=1,
):
    ranked = rank_nodes(sim, start, nodes, category, ranking_strategy)
    if primary_nodes is None:
        return hierarchical_candidates([], ranked, max_k, min_separation, 0), len(ranked)
    primary_ranked = rank_nodes(sim, start, primary_nodes, category, "category_score")
    selected = hierarchical_candidates(
        primary_ranked,
        ranked,
        max_k,
        min_separation,
        primary_k,
    )
    return selected, len(primary_ranked) + len(ranked)


def draw_reference(root: Path, node: dict, output: Path) -> None:
    source = root / node["source_rgb_path"]
    image = Image.open(source).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    x1, y1, x2, y2 = node["bbox_norm"]
    box = (x1 * width, y1 * height, x2 * width, y2 * height)
    draw.rectangle(box, outline="#ff2d95", width=6)
    label = f"MAP HYPOTHESIS: {node['category']} {float(node['confidence']):.2f}"
    text_box = draw.textbbox((box[0], box[1]), label)
    draw.rectangle((text_box[0] - 3, text_box[1] - 3, text_box[2] + 3, text_box[3] + 3), fill="black")
    draw.text((box[0], box[1]), label, fill="#ff60b5")
    image.save(output, quality=94)


def capture_candidate(root: Path, sim, scene_output: Path, node: dict, args) -> dict:
    category = node["category"]
    candidate_id = int(node["id"])
    directory = scene_output / f"candidate_{candidate_id:05d}_{category}"
    directory.mkdir(parents=True, exist_ok=True)
    reference_path = directory / "mapping_reference_boxed.jpg"
    if not reference_path.exists():
        draw_reference(root, node, reference_path)

    position = np.asarray(node["position"], dtype=float)
    object_estimate = np.asarray(node["object_estimate"], dtype=float)
    base_rotation = yaw_quat(position, object_estimate)
    views = []
    canvas = Image.new("RGB", (args.width * 2, args.height * 2))
    for view_index, yaw in enumerate((0, 90, 180, 270)):
        rotation = yaw_quaternion(base_rotation, yaw)
        set_pose(sim, position, rotation)
        observations = sim.get_sensor_observations()
        rgb = observations["rgb"][:, :, :3]
        depth = np.asarray(observations["depth"], dtype=np.float32)
        rgb_path = directory / f"arrival_view{view_index}_rgb.jpg"
        depth_path = directory / f"arrival_view{view_index}_depth.npy"
        if not rgb_path.exists():
            imageio.imwrite(rgb_path, rgb, quality=94)
        if not depth_path.exists():
            np.save(depth_path, depth)
        canvas.paste(Image.fromarray(rgb), ((view_index % 2) * args.width, (view_index // 2) * args.height))
        views.append({
            "view_index": view_index,
            "yaw_from_object_deg": yaw,
            "rotation": rotation,
            "resolution": [args.width, args.height],
            "hfov_deg": args.hfov,
            "rgb_path": str(rgb_path.relative_to(root)),
            "depth_path": str(depth_path.relative_to(root)),
        })
    panorama_path = directory / "arrival_four_view.jpg"
    if not panorama_path.exists():
        canvas.save(panorama_path, quality=92)
    return {
        "candidate_id": candidate_id,
        "category": category,
        "confidence": float(node.get("confidence", node.get("category_scores", {}).get(category, 0.0))),
        "category_score": float(node.get("category_scores", {}).get(category, 0.0)),
        "cluster_support": int(node.get("cluster_support", 1)),
        "cluster_detections": int(node.get("cluster_detections", 1)),
        "fusion_cluster_id": node.get("fusion_cluster_id"),
        "is_cluster_representative": bool(node.get("is_cluster_representative", False)),
        "source_candidate_id": node.get("source_candidate_id", candidate_id),
        "position": node["position"],
        "object_estimate": node["object_estimate"],
        "source_topology_node": node.get("source_topology_node"),
        "source_rgb_path": node.get("source_rgb_path"),
        "source_bbox_norm": node.get("bbox_norm"),
        "mapping_reference_boxed_path": str(reference_path.relative_to(root)),
        "arrival_panorama_path": str(panorama_path.relative_to(root)),
        "views": views,
    }


def run(args) -> None:
    root = Path(args.root).resolve()
    maps = root / args.maps
    dataset = root / "data/datasets/objectnav_hm3d_v2" / args.split
    scenes = root / "data/hm3d" / args.scene_dir
    output = root / args.output
    output.mkdir(parents=True, exist_ok=True)
    episode_sequences = {}
    report = {
        "protocol": "query-ranked candidates; no ObjectNav goals used in selection or capture",
        "split": args.split,
        "max_k": args.max_k,
        "candidate_min_separation_m": args.min_separation,
        "ranking_strategy": args.ranking_strategy,
        "primary_maps": args.primary_maps,
        "primary_k": args.primary_k if args.primary_maps else 0,
        "sensor": {"width": args.width, "height": args.height, "hfov": args.hfov, "height_m": 1.25},
        "scenes": {},
    }

    for _, data in load_scene_episodes(dataset):
        _, scene, base, nav = scene_paths(scenes, data["episodes"][0]["scene_id"])
        map_path = maps / scene / "vlm_topological_map.json"
        if not map_path.exists():
            continue
        sim = make_rgbd_sim(base, nav, args.width, args.height, args.hfov)
        nodes = [dict(node) for node in load_nodes(root, scene, "vlm_all", vlm_maps=args.maps)]
        # Combined representative + multi-standoff maps use independent local
        # IDs.  Offset fallback IDs so captures and verifier caches remain
        # unambiguous while preserving the source map files unchanged.
        if args.primary_maps:
            for node in nodes:
                node["source_candidate_id"] = int(node["id"])
                node["id"] = 100000 + int(node["id"])
            primary_nodes = [dict(node) for node in load_nodes(root, scene, "vlm_all", vlm_maps=args.primary_maps)]
            for node in primary_nodes:
                node["source_candidate_id"] = int(node["id"])
        else:
            primary_nodes = None
        unique = {}
        for episode in data["episodes"]:
            category = episode["object_category"]
            candidates, candidate_count = ranked_candidates(
                sim,
                episode["start_position"],
                nodes,
                category,
                args.max_k,
                args.min_separation,
                args.ranking_strategy,
                primary_nodes,
                args.primary_k,
            )
            key = f"{scene}_{category}_{episode['episode_id']}"
            ids = []
            for node in candidates:
                candidate_id = int(node["id"])
                unique[candidate_id] = node
                ids.append(candidate_id)
            episode_sequences[key] = {
                "episode": key,
                "scene": scene,
                "category": category,
                "candidate_count": candidate_count,
                "candidate_ids": ids,
            }
        scene_output = output / scene
        scene_output.mkdir(exist_ok=True)
        candidates = {
            str(candidate_id): capture_candidate(root, sim, scene_output, node, args)
            for candidate_id, node in sorted(unique.items())
        }
        (scene_output / "arrival_candidates.json").write_text(json.dumps({"scene": scene, "candidates": candidates}, indent=2))
        report["scenes"][scene] = {"episodes": len(data["episodes"]), "unique_candidates": len(candidates)}
        sim.close()
        print(f"{scene}: {len(data['episodes'])} episodes, {len(candidates)} unique arrival candidates", flush=True)

    (output / "episode_candidates.json").write_text(json.dumps(episode_sequences, indent=2))
    report["episodes"] = len(episode_sequences)
    report["unique_candidates"] = sum(item["unique_candidates"] for item in report["scenes"].values())
    report["views"] = report["unique_candidates"] * 4
    report["goal_annotations_used"] = False
    (output / "capture_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--split", default="val")
    parser.add_argument("--scene-dir", default="val")
    parser.add_argument("--maps", default="outputs/hm3d_val_uniform/gpt54_rgbd_wide120_fusion/raw_maps")
    parser.add_argument("--output", default="outputs/hm3d_val_uniform/gpt54_arrival_capture_top3")
    parser.add_argument("--max-k", type=int, default=3)
    parser.add_argument("--min-separation", type=float, default=3.0)
    parser.add_argument("--ranking-strategy", choices=RANKING_STRATEGIES, default="category_score")
    parser.add_argument("--primary-maps", default=None)
    parser.add_argument("--primary-k", type=int, default=1)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--hfov", type=float, default=120.0)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
