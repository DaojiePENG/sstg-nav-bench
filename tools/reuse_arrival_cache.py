#!/usr/bin/env python3
"""Reuse arrival-VLM replies when two candidate lists share an exact pose/view."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def candidates(root: Path):
    result = {}
    for path in root.glob("*/arrival_candidates.json"):
        data = json.loads(path.read_text())
        scene = data["scene"]
        for candidate_id, item in data["candidates"].items():
            result[f"{scene}/candidate_{int(candidate_id):05d}"] = item
    return result


def signature(scene_key: str, item: dict):
    scene = scene_key.split("/")[0]
    return (
        scene,
        item["category"],
        tuple(round(float(value), 5) for value in item["position"]),
        tuple(round(float(value), 5) for value in item["object_estimate"]),
        item.get("source_rgb_path"),
        tuple(round(float(value), 5) for value in item.get("source_bbox_norm", [])),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-capture", required=True)
    parser.add_argument("--source-cache", required=True)
    parser.add_argument("--target-capture", required=True)
    parser.add_argument("--output-cache", required=True)
    args = parser.parse_args()
    source_candidates = candidates(Path(args.source_capture))
    target_candidates = candidates(Path(args.target_capture))
    source_cache = json.loads(Path(args.source_cache).read_text())
    index = {
        signature(key, item): source_cache[key]
        for key, item in source_candidates.items()
        if key in source_cache and source_cache[key].get("status") == "ok"
    }
    reused = {
        key: index[signature(key, item)]
        for key, item in target_candidates.items()
        if signature(key, item) in index
    }
    output = Path(args.output_cache)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(reused, indent=2))
    print(json.dumps({
        "source_cached": len(source_cache),
        "target_candidates": len(target_candidates),
        "reused": len(reused),
        "remaining": len(target_candidates) - len(reused),
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
