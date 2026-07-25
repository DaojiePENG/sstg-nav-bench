"""Build a camera-node baseline from cached view-local VLM detections.

This deliberately discards boxes and depth: every category observed from a
topology pose is attached to that pose. Evaluating it beside RGB-D candidates
isolates representation while holding the semantic responses fixed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def attach_detections_to_camera_nodes(mapping):
    nodes = []
    for source in mapping["nodes"]:
        node = dict(source)
        scores = {}
        for detection in source.get("localized_vlm", {}).get("detections", []):
            category = detection["category"]
            scores[category] = max(scores.get(category, 0.0), float(detection["confidence"]))
        categories = sorted(scores)
        node.update(
            categories=categories,
            categories_all=categories,
            categories_primary=[max(scores, key=scores.get)] if scores else [],
            category_scores=scores,
            representation="camera_node_no_depth",
        )
        nodes.append(node)
    result = dict(mapping)
    result["nodes"] = nodes
    result["baseline"] = {
        "representation": "camera_node_no_depth",
        "semantic_responses_reused": True,
        "boxes_used": False,
        "depth_used": False,
        "goal_annotations_used": False,
    }
    return result


def run(root, source, output):
    root = Path(root).resolve()
    source_root = root / source
    output_root = root / output
    report = {"source": source, "scenes": {}, "nodes": 0, "labeled_nodes": 0}
    for path in sorted(source_root.glob("*/rgbd_semantic_map.json")):
        mapping = attach_detections_to_camera_nodes(json.loads(path.read_text()))
        scene = path.parent.name
        destination = output_root / scene
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "vlm_topological_map.json").write_text(json.dumps(mapping, indent=2))
        labeled = sum(bool(node["categories_all"]) for node in mapping["nodes"])
        report["scenes"][scene] = {"nodes": len(mapping["nodes"]), "labeled_nodes": labeled}
        report["nodes"] += len(mapping["nodes"])
        report["labeled_nodes"] += labeled
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "camera_node_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.root, args.source, args.output)


if __name__ == "__main__":
    main()
