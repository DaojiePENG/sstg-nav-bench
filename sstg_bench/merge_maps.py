"""Merge independently constructed semantic candidate maps without goal data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def run(args):
    root = Path(args.root).resolve()
    inputs = [root / value for value in args.maps]
    output = root / args.output
    output.mkdir(parents=True, exist_ok=True)
    scenes = sorted({path.parent.name for directory in inputs for path in directory.glob("*/vlm_topological_map.json")})
    report = {"input_maps": args.maps, "scenes": {}}
    for scene in scenes:
        nodes = []
        scene_id = None
        counts = {}
        for directory in inputs:
            path = directory / scene / "vlm_topological_map.json"
            if not path.exists():
                continue
            mapping = json.loads(path.read_text())
            scene_id = scene_id or mapping["scene"]
            if mapping["scene"] != scene_id:
                raise ValueError(f"scene id mismatch for {scene}: {path}")
            source_name = str(path.relative_to(root))
            counts[source_name] = len(mapping["nodes"])
            for source_node in mapping["nodes"]:
                node = dict(source_node)
                node["source_map"] = source_name
                node["source_map_node_id"] = source_node["id"]
                node["id"] = len(nodes)
                nodes.append(node)
        scene_output = output / scene
        scene_output.mkdir(exist_ok=True)
        result = {
            "scene": scene_id,
            "nodes": nodes,
            "edges": [],
            "merge": {"goal_annotations_used": False, "source_counts": counts},
        }
        (scene_output / "vlm_topological_map.json").write_text(json.dumps(result, indent=2))
        report["scenes"][scene] = {"nodes": len(nodes), "source_counts": counts}
    report["nodes"] = sum(value["nodes"] for value in report["scenes"].values())
    (output / "merge_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--maps", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
