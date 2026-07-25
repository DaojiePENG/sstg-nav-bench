"""Target-conditioned VLM verification from fresh arrival RGB-D capture."""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import time

import numpy as np
import requests

from .peterai_rgbd_semantics import load_key, write_json_atomic
from .vlm import CATEGORIES


PROMPT_VERSION = "gpt54-arrival-verifier-reference-rgbd-v1"
PROMPT = """You are the arrival verifier of an ObjectNav robot.
The robot was asked to navigate to TARGET_CATEGORY={category}.

REFERENCE is a mapping-time RGB image. Its magenta box marks the semantic
hypothesis that generated this navigation candidate. The reference may be a
false detection, reflection, screen image, or a real object.

ARRIVAL VIEW 0..3 are fresh 120-degree RGB observations captured after the
robot physically reached the candidate. VIEW 0 faces the stored 3D object
estimate; the other views rotate by 90, 180, and 270 degrees.

Accept STOP only when a real physical instance of TARGET_CATEGORY is clearly
visible in an ARRIVAL view and the robot appears to be on an unobstructed,
reasonable stopping side of it. Reject a category visible only in REFERENCE,
on a TV/poster/mirror, behind a wall/glass/railing, or too ambiguous to ground.
Do not accept from room context. Depth is checked separately by the robot, so
do not guess metric distance.

Return JSON only:
{{"target_category":"{category}","target_visible":true|false,
  "stop_geometry":"valid"|"invalid"|"uncertain",
  "view_index":0|1|2|3|null,
  "bbox_norm":[x1,y1,x2,y2]|null,
  "confidence":0..1,
  "reason_code":"confirmed"|"not_visible"|"wrong_category"|
                "reference_false_positive"|"barrier"|"ambiguous"}}
Coordinates are normalized within the selected ARRIVAL view. A false decision
must use null view_index and bbox_norm. Do not add Markdown or prose."""

PROMPT_VERSION_V2 = "arrival-verifier-fused-prior-rgbd-v2"
PROMPT_V2 = """You verify arrival for an ObjectNav robot whose requested
category is TARGET_CATEGORY={category}.

The candidate pose was produced before this query by multi-view 3D fusion,
shifted to a reachable object-facing standoff, and snapped to the navmesh.
REFERENCE shows the mapping observation and its magenta detection box.
ARRIVAL VIEW 0..3 are fresh 120-degree RGB images captured at the reached pose;
together they cover the full surroundings.  View 0 faces the stored object
estimate.

Find a real physical instance of TARGET_CATEGORY in any ARRIVAL view.  A
partially cropped, oblique, or low-camera view is sufficient when the category
is visually unambiguous.  Do not reject merely because the object is not
centered or because another stopping side might be nicer: reachability and
metric range are checked independently from aligned depth.  Reject only when
no real instance is visible, the apparent target is clearly a screen/poster/
reflection, the category is wrong, or a physical barrier makes the observed
instance unreachable from this side.  REFERENCE is supporting context, never
enough by itself.

Return the most clearly visible arrival instance as JSON only:
{{"target_category":"{category}","target_visible":true|false,
  "stop_geometry":"valid"|"invalid"|"uncertain",
  "view_index":0|1|2|3|null,
  "bbox_norm":[x1,y1,x2,y2]|null,
  "confidence":0..1,
  "reason_code":"confirmed"|"not_visible"|"wrong_category"|
                "reference_false_positive"|"barrier"|"ambiguous"}}
Coordinates are normalized within the selected ARRIVAL view.  When no target
is visible, use null view_index and bbox_norm.  Do not add Markdown or prose."""

PROMPT_PRESETS = {
    "strict_v1": (PROMPT_VERSION, PROMPT),
    "fused_prior_v2": (PROMPT_VERSION_V2, PROMPT_V2),
}


def parse_output(raw: str, expected_category: str) -> dict:
    cleaned = raw.strip().replace("```json", "").replace("```", "")
    match = re.search(r"\{.*\}", cleaned, re.S)
    if not match:
        raise ValueError("no JSON object in verifier output: " + raw[:500])
    data = json.loads(match.group(0))
    visible = bool(data.get("target_visible", False))
    geometry = str(data.get("stop_geometry", "uncertain")).lower()
    confidence = float(data.get("confidence", 0.0) or 0.0)
    view_index = data.get("view_index")
    bbox = data.get("bbox_norm")
    if data.get("target_category") not in (None, expected_category):
        visible = False
    valid_box = False
    normalized = None
    try:
        view_index = int(view_index)
        x1, y1, x2, y2 = map(float, bbox)
        if max(x1, y1, x2, y2) > 1 and all(0 <= value <= 1000 for value in (x1, y1, x2, y2)):
            x1, y1, x2, y2 = (value / 1000 for value in (x1, y1, x2, y2))
        valid_box = view_index in range(4) and 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1
        normalized = [x1, y1, x2, y2] if valid_box else None
    except (TypeError, ValueError):
        view_index = None
    return {
        "target_category": expected_category,
        "target_visible": visible,
        "stop_geometry": geometry if geometry in {"valid", "invalid", "uncertain"} else "uncertain",
        "view_index": view_index if valid_box else None,
        "bbox_norm": normalized,
        "confidence": max(0.0, min(1.0, confidence)),
        "reason_code": str(data.get("reason_code", "ambiguous")),
    }


def depth_median(depth_path: Path, bbox: list[float]) -> float | None:
    depth = np.load(depth_path)
    height, width = depth.shape[:2]
    x1, y1, x2, y2 = bbox
    # Use the central 50% of the returned box to reduce background leakage.
    px1 = int(np.clip((0.75 * x1 + 0.25 * x2) * width, 0, width - 1))
    px2 = int(np.clip((0.25 * x1 + 0.75 * x2) * width, px1 + 1, width))
    py1 = int(np.clip((0.75 * y1 + 0.25 * y2) * height, 0, height - 1))
    py2 = int(np.clip((0.25 * y1 + 0.75 * y2) * height, py1 + 1, height))
    patch = depth[py1:py2, px1:px2]
    valid = patch[np.isfinite(patch) & (patch > 0.05)]
    return float(np.median(valid)) if valid.size else None


def finalize_decision(parsed: dict, candidate: dict, root: Path, args) -> dict:
    result = dict(parsed)
    view_index = parsed.get("view_index")
    bbox = parsed.get("bbox_norm")
    depth = None
    if view_index is not None and bbox is not None:
        view = candidate["views"][view_index]
        depth = depth_median(root / view["depth_path"], bbox)
    result["depth_m"] = depth
    result["checks"] = {
        "visible": parsed["target_visible"],
        "confidence": parsed["confidence"] >= args.min_confidence,
        "depth_valid": depth is not None and args.min_depth <= depth <= args.max_depth,
    }
    result["checks"]["vlm_stop_geometry_valid"] = parsed["stop_geometry"] == "valid"
    required = ("visible", "confidence", "depth_valid")
    if args.require_vlm_geometry:
        required += ("vlm_stop_geometry_valid",)
    result["accept"] = all(result["checks"][key] for key in required)
    result["decision_policy"] = "vlm_category_plus_rgbd_geometry" if not args.require_vlm_geometry else "strict_dual_geometry"
    result["thresholds"] = {
        "min_confidence": args.min_confidence,
        "min_depth_m": args.min_depth,
        "max_depth_m": args.max_depth,
    }
    return result


def request_candidate(candidate: dict, root: Path, key: str, args) -> tuple[dict, str, str | None, dict]:
    category = candidate["category"]
    _, prompt = PROMPT_PRESETS[args.prompt_preset]
    content = [{"type": "text", "text": prompt.format(category=category)}]
    images = [("REFERENCE", root / candidate["mapping_reference_boxed_path"])]
    images.extend((f"ARRIVAL VIEW {view['view_index']}", root / view["rgb_path"]) for view in candidate["views"])
    for label, path in images:
        encoded = base64.b64encode(path.read_bytes()).decode()
        content.append({"type": "text", "text": label + ":"})
        content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encoded, "detail": "high"}})
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 900,
    }
    endpoint = args.base_url.rstrip("/") + "/chat/completions"
    last = None
    for attempt in range(args.retries):
        try:
            response = requests.post(
                endpoint,
                headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
                json=body,
                timeout=240,
            )
            response.raise_for_status()
            payload = response.json()
            raw = payload.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
            if not raw:
                raise ValueError("empty verifier response")
            return parse_output(raw, category), raw, payload.get("id"), payload.get("usage", {})
        except Exception as error:
            last = error
            if attempt + 1 < args.retries:
                time.sleep(min(30, 2 ** attempt))
    raise last


def infer_job(cache_key: str, candidate: dict, root: Path, key: str, args):
    started = time.time()
    try:
        parsed, raw, response_id, usage = request_candidate(candidate, root, key, args)
        decision = finalize_decision(parsed, candidate, root, args)
        return cache_key, {
            "status": "ok",
            "prompt_version": PROMPT_PRESETS[args.prompt_preset][0],
            "model": args.model,
            "wire_api": "chat",
            "candidate_category": candidate["category"],
            "decision": decision,
            "raw": raw,
            "response_id": response_id,
            "usage": usage,
            "latency_s": round(time.time() - started, 3),
        }
    except Exception as error:
        return cache_key, {
            "status": "error",
            "prompt_version": PROMPT_PRESETS[args.prompt_preset][0],
            "model": args.model,
            "candidate_category": candidate["category"],
            "decision": {"accept": False},
            "error": repr(error),
            "latency_s": round(time.time() - started, 3),
        }


def run(args) -> None:
    root = Path(args.root).resolve()
    source = root / args.source
    output = root / args.output
    output.mkdir(parents=True, exist_ok=True)
    key = load_key(root, args.env)
    cache_path = output / "gpt_arrival_responses.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    expected_prompt_version = PROMPT_PRESETS[args.prompt_preset][0]
    candidates = {}
    for path in sorted(source.glob("*/arrival_candidates.json")):
        scene_data = json.loads(path.read_text())
        scene = scene_data["scene"]
        for candidate_id, candidate in scene_data["candidates"].items():
            candidates[f"{scene}/candidate_{int(candidate_id):05d}"] = candidate

    # Reparse immutable raw responses when thresholds or parser code change.
    for cache_key, item in cache.items():
        if item.get("status") == "ok" and item.get("raw") and cache_key in candidates:
            try:
                parsed = parse_output(item["raw"], candidates[cache_key]["category"])
                item["decision"] = finalize_decision(parsed, candidates[cache_key], root, args)
            except Exception:
                pass
    if cache:
        write_json_atomic(cache_path, cache)

    todo = [
        (cache_key, candidate) for cache_key, candidate in candidates.items()
        if cache.get(cache_key, {}).get("status") != "ok"
        or cache.get(cache_key, {}).get("prompt_version") != expected_prompt_version
        or cache.get(cache_key, {}).get("model") != args.model
    ]
    if args.cache_only:
        todo = []
    if args.limit is not None:
        todo = todo[:args.limit]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(infer_job, cache_key, candidate, root, key, args): cache_key
            for cache_key, candidate in todo
        }
        for index, future in enumerate(as_completed(futures), 1):
            cache_key, result = future.result()
            cache[cache_key] = result
            if index % args.checkpoint_every == 0 or index == len(todo):
                write_json_atomic(cache_path, cache)
            print(
                f"[{index}/{len(todo)}] {cache_key}: {result['status']}, "
                f"accept={result.get('decision', {}).get('accept')}, {result['latency_s']} s",
                flush=True,
            )
    if args.limit is not None:
        return

    complete = [cache[key] for key in candidates if cache.get(key, {}).get("status") == "ok"]
    report = {
        "protocol": "fresh target-oriented RGB-D; target-conditioned VLM plus depth gate",
        "prompt_version": expected_prompt_version,
        "prompt_preset": args.prompt_preset,
        "model": args.model,
        "candidates": len(candidates),
        "api_successes": len(complete),
        "accepted": sum(bool(item.get("decision", {}).get("accept")) for item in complete),
        "rejected": sum(not bool(item.get("decision", {}).get("accept")) for item in complete),
        "thresholds": {"min_confidence": args.min_confidence, "min_depth_m": args.min_depth, "max_depth_m": args.max_depth},
        "decision_policy": "strict_dual_geometry" if args.require_vlm_geometry else "vlm_category_plus_rgbd_geometry",
        "usage": {
            usage_key: sum(int(item.get("usage", {}).get(usage_key, 0) or 0) for item in complete)
            for usage_key in ("prompt_tokens", "completion_tokens", "total_tokens")
        },
        "latency_s": {
            "mean": float(np.mean([item.get("latency_s", 0.0) for item in complete])) if complete else None,
            "median": float(np.median([item.get("latency_s", 0.0) for item in complete])) if complete else None,
            "p95": float(np.percentile([item.get("latency_s", 0.0) for item in complete], 95)) if complete else None,
        },
    }
    (output / "verifier_report.json").write_text(json.dumps(report, indent=2))
    (output / "verifier_decisions.json").write_text(json.dumps({key: cache.get(key) for key in sorted(candidates)}, indent=2))
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--source", default="outputs/hm3d_val_uniform/gpt54_arrival_capture_top3")
    parser.add_argument("--output", default="outputs/hm3d_val_uniform/gpt54_arrival_verifier_top3_strict")
    parser.add_argument("--env", default="../.env")
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--prompt-preset", choices=tuple(PROMPT_PRESETS), default="strict_v1")
    parser.add_argument("--base-url", default="https://api.peterai.cc.cd/v1")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--min-confidence", type=float, default=0.75)
    parser.add_argument("--min-depth", type=float, default=0.25)
    parser.add_argument("--max-depth", type=float, default=2.5)
    geometry = parser.add_mutually_exclusive_group()
    geometry.add_argument(
        "--require-vlm-geometry", dest="require_vlm_geometry", action="store_true",
        help="Require both the VLM stopping-side judgment and the measured RGB-D range gate (default).",
    )
    geometry.add_argument(
        "--rgbd-only-geometry", dest="require_vlm_geometry", action="store_false",
        help="Ablation: ignore the VLM stopping-side judgment and use category visibility plus RGB-D range.",
    )
    parser.set_defaults(require_vlm_geometry=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cache-only", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
