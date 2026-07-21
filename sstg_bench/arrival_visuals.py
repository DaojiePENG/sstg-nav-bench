"""Traceable overlays and route audit for the arrival-verification loop."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

from .benchmark import load_scene_episodes, make_sim, map_pixel, scene_paths, topdown_base
from .map_artifacts import contact_sheet


def load_candidates(source: Path):
    result = {}
    for path in source.glob("*/arrival_candidates.json"):
        data = json.loads(path.read_text())
        scene = data["scene"]
        for candidate_id, candidate in data["candidates"].items():
            result[f"{scene}/candidate_{int(candidate_id):05d}"] = candidate
    return result


def overlay_candidate(root: Path, cache_key: str, candidate: dict, response: dict, output: Path) -> Path:
    decision = response.get("decision", {}) or {}
    accepted = bool(response.get("status") == "ok" and decision.get("accept"))
    view_index = decision.get("view_index") if accepted else 0
    if view_index is None:
        view_index = 0
    image = Image.open(root / candidate["views"][int(view_index)]["rgb_path"]).convert("RGB")
    draw = ImageDraw.Draw(image)
    color = "#22aa55" if accepted else "#e53935"
    bbox = decision.get("bbox_norm") if accepted else None
    if bbox:
        x1, y1, x2, y2 = bbox
        draw.rectangle((x1 * image.width, y1 * image.height, x2 * image.width, y2 * image.height), outline=color, width=7)
    if accepted:
        label = (
            f"ACCEPT {candidate['category']} q={float(decision.get('confidence', 0)):.2f} "
            f"depth={float(decision.get('depth_m', 0)):.2f}m"
        )
    else:
        label = f"REJECT {candidate['category']}: {decision.get('reason_code', response.get('status', 'missing'))}"
    text_box = draw.textbbox((10, 10), label)
    draw.rectangle((5, 5, text_box[2] + 8, text_box[3] + 8), fill="black")
    draw.text((10, 10), label, fill=color)
    image.save(output, quality=94)
    return output


def render_recovery_route(root: Path, scene: str, attempts: list[dict], output: Path, split: str, scene_dir: str):
    dataset = root / "data/datasets/objectnav_hm3d_v2" / split
    scenes = root / "data/hm3d" / scene_dir
    scene_data = next(data for _, data in load_scene_episodes(dataset) if scene in data["episodes"][0]["scene_id"])
    _, _, base, nav = scene_paths(scenes, scene_data["episodes"][0]["scene_id"])
    sim = make_sim(base, nav)
    base_map = topdown_base(sim)
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.imshow(base_map, cmap="gray", origin="upper")
    colors = ("#e53935", "#1e88e5", "#22aa55")
    for index, attempt in enumerate(attempts):
        route = np.asarray(attempt["route"], dtype=float)
        xy = np.asarray([map_pixel(sim, point, base_map.shape) for point in route])
        label = f"visit {attempt['rank']}: {'ACCEPT' if attempt['verifier_accept'] else 'REJECT'}"
        ax.plot(xy[:, 0], xy[:, 1], color=colors[index % len(colors)], lw=3.5, label=label)
        ax.scatter(xy[-1, 0], xy[-1, 1], s=65, color=colors[index % len(colors)], edgecolor="white", linewidth=.8)
        ax.text(xy[-1, 0] + 4, xy[-1, 1] - 4, str(attempt["rank"]), color=colors[index % len(colors)], weight="bold")
    ax.set_title("Fresh RGB-D verification controls STOP / continue", fontsize=11)
    ax.axis("off")
    ax.legend(loc="lower left", frameon=True, fontsize=8)
    fig.tight_layout(pad=.2)
    fig.savefig(output, dpi=240, bbox_inches="tight")
    plt.close(fig)
    sim.close()


def run(args) -> None:
    root = Path(args.root).resolve()
    source = root / args.source
    verifier = root / args.verifier
    evaluation = root / args.evaluation
    output = root / args.output
    overlays = output / "verification_overlays"
    case_dir = output / "recovery_case"
    overlays.mkdir(parents=True, exist_ok=True)
    case_dir.mkdir(parents=True, exist_ok=True)
    candidates = load_candidates(source)
    responses = json.loads((verifier / "verifier_decisions.json").read_text())
    accepted_gallery, rejected_gallery = [], []
    overlay_paths = {}
    for cache_key, candidate in candidates.items():
        scene, candidate_name = cache_key.split("/")
        response = responses.get(cache_key, {}) or {}
        accepted = bool(response.get("status") == "ok" and response.get("decision", {}).get("accept"))
        path = overlays / f"{scene}_{candidate_name}_{candidate['category']}_{'accept' if accepted else 'reject'}.jpg"
        overlay_candidate(root, cache_key, candidate, response, path)
        overlay_paths[cache_key] = path
        label = f"{scene} {candidate['category']} {'accept' if accepted else 'reject'}"
        (accepted_gallery if accepted else rejected_gallery).append((label, path))
    contact_sheet(accepted_gallery[:24], output / "accepted_contact_sheet.jpg", columns=4, size=(420, 300))
    contact_sheet(rejected_gallery[:24], output / "rejected_contact_sheet.jpg", columns=4, size=(420, 300))

    rows = list(csv.DictReader((evaluation / "episodes_arrival_verified.csv").open()))
    recoveries = []
    for row in rows:
        attempts = json.loads(row["attempts_json"])
        accepted_rank = int(row["accepted_rank"])
        if (
            float(row["success"]) > .5
            and accepted_rank > 1
            and all(not attempt["official_success_at_candidate"] for attempt in attempts if attempt["rank"] < accepted_rank)
        ):
            recoveries.append(row)
    if not recoveries:
        raise RuntimeError("no verifier-recovered episode available for audit")
    case = recoveries[0]
    attempts = json.loads(case["attempts_json"])
    scene = case["scene"]
    for attempt in attempts:
        candidate_id = int(attempt["candidate_id"])
        cache_key = f"{scene}/candidate_{candidate_id:05d}"
        attempt["mapping_reference_boxed_path"] = candidates[cache_key]["mapping_reference_boxed_path"]
        attempt["verification_overlay_path"] = str(overlay_paths[cache_key].relative_to(root))
    route_path = case_dir / "sequential_recovery_topdown.png"
    render_recovery_route(root, scene, attempts, route_path, args.split, args.scene_dir)
    manifest = {
        "episode": case["episode"],
        "category": case["category"],
        "accepted_rank": int(case["accepted_rank"]),
        "spl": float(case["spl"]),
        "attempts": attempts,
        "topdown_path": str(route_path.relative_to(root)),
    }
    (case_dir / "case_manifest.json").write_text(json.dumps(manifest, indent=2))
    report = {
        "candidate_overlays": len(overlay_paths),
        "accepted_overlays": len(accepted_gallery),
        "rejected_overlays": len(rejected_gallery),
        "recovery_episode": case["episode"],
    }
    (output / "visual_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--split", default="val")
    parser.add_argument("--scene-dir", default="val")
    parser.add_argument("--source", default="outputs/hm3d_val_uniform/gpt54_arrival_capture_top3")
    parser.add_argument("--verifier", default="outputs/hm3d_val_uniform/gpt54_arrival_verifier_top3_strict")
    parser.add_argument("--evaluation", default="outputs/hm3d_val_uniform/gpt54_arrival_verified_top3_strict")
    parser.add_argument("--output", default="outputs/hm3d_val_uniform/gpt54_arrival_verified_visuals")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
