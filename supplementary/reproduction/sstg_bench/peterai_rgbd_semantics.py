"""GPT RGB-D grounding for independently sampled topology nodes.

Each request receives four *separate* full-resolution cardinal RGB views.  GPT
returns view-local boxes; the corresponding depth pixels are consumed later by
``rgbd_fusion.py``.  Successful responses are immutable and the cache is
written after every completed request so expensive runs are safely resumable.
"""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import time

import requests

from .vlm import CATEGORIES


PROMPT_VERSION = "peterai-gpt-rgbd-grounding-v2-variable-resolution"
PARSER_VERSION = "bbox-norm-or-1000-v2"
PROMPT = """You are constructing a metric semantic map for an ObjectNav robot.
The four images below are separate RGB views captured at exactly the
same robot position. They are ordered VIEW 0, VIEW 1, VIEW 2, VIEW 3 and face
yaw offsets 0, 90, 180, and 270 degrees respectively. Inspect every view.

Detect only clearly visible instances of these allowed categories:
chair, bed, plant, toilet, tv_monitor, sofa.
Map any television or computer monitor to tv_monitor. Do not infer an object
only from room context. A partly visible object is valid only if its category
is visually unambiguous. Empty output is valid and preferred over guessing.

Return JSON only with this exact top-level shape:
{"detections": [...]}
Each detection must contain:
- "view_index": integer 0..3
- "category": one allowed category
- "bbox_norm": [x1,y1,x2,y2], tight coordinates normalized independently to
  that view, where the top-left is [0,0] and bottom-right is [1,1]
- "confidence": number 0..1 reflecting visual certainty

Do not duplicate one physical object within the same view. Include only
confidence >= 0.60. Do not add prose or Markdown."""


def write_json_atomic(path: Path, data) -> None:
    """Replace a JSON cache atomically so interruption cannot truncate it."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def load_key(root: Path, env_path: str) -> str:
    """Load PeterAI_KEY without parsing unrelated TOML-like .env content."""
    key = os.getenv("PeterAI_KEY", "").strip()
    path = root / env_path
    if not key and path.exists():
        for line in path.read_text().splitlines():
            match = re.match(r"\s*PeterAI_KEY\s*=\s*(.*?)\s*$", line)
            if match:
                key = match.group(1).strip().strip('"\'')
                break
    if not key:
        raise SystemExit("PeterAI_KEY is not set and was not found in --env")
    return key


def parse_output(raw: str):
    cleaned = raw.strip().replace("```json", "").replace("```", "")
    match = re.search(r"\{.*\}", cleaned, re.S)
    if not match:
        raise ValueError("no JSON object in model output: " + raw[:500])
    data = json.loads(match.group(0))
    detections = []
    for item in data.get("detections", []):
        category = item.get("category", item.get("label"))
        bbox = item.get("bbox_norm")
        try:
            view = int(item.get("view_index"))
            confidence = float(item.get("confidence"))
            x1, y1, x2, y2 = map(float, bbox)
        except (TypeError, ValueError):
            continue
        # Some compatible multimodal models use the established 0..1000
        # grounding coordinate convention even when the requested key is
        # bbox_norm. Preserve their detections by normalizing that convention.
        if max(x1, y1, x2, y2) > 1 and all(0 <= value <= 1000 for value in (x1, y1, x2, y2)):
            x1, y1, x2, y2 = (value / 1000 for value in (x1, y1, x2, y2))
        if category not in CATEGORIES or view not in range(4):
            continue
        if confidence < 0.60 or not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
            continue
        detections.append({
            "category": category,
            "confidence": confidence,
            "view_index": view,
            "center": [(x1 + x2) / 2, (y1 + y2) / 2],
            "bbox_norm": [x1, y1, x2, y2],
        })
    return detections


def response_text(payload):
    chunks = []
    for item in payload.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                chunks.append(content.get("text", ""))
    return "\n".join(chunks)


def annotate_node(images, key, base_url, model, wire_api, retries):
    responses_content = [{"type": "input_text", "text": PROMPT}]
    chat_content = [{"type": "text", "text": PROMPT}]
    for index, image in enumerate(images):
        encoded = base64.b64encode(image.read_bytes()).decode()
        data_url = "data:image/jpeg;base64," + encoded
        responses_content.append({"type": "input_text", "text": f"VIEW {index}:"})
        responses_content.append({
            "type": "input_image",
            "image_url": data_url,
            "detail": "high",
        })
        chat_content.append({"type": "text", "text": f"VIEW {index}:"})
        chat_content.append({"type": "image_url", "image_url": {"url": data_url, "detail": "high"}})
    if wire_api == "responses":
        endpoint = base_url.rstrip("/") + "/responses"
        body = {
            "model": model,
            "input": [{"role": "user", "content": responses_content}],
            "reasoning": {"effort": "low"},
            "max_output_tokens": 1600,
        }
    else:
        endpoint = base_url.rstrip("/") + "/chat/completions"
        body = {
            "model": model,
            "messages": [{"role": "user", "content": chat_content}],
            "max_tokens": 1600,
        }
    last = None
    for attempt in range(retries):
        try:
            response = requests.post(
                endpoint,
                headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
                json=body,
                timeout=240,
            )
            response.raise_for_status()
            payload = response.json()
            if wire_api == "responses":
                raw = response_text(payload)
            else:
                raw = payload.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
            if not raw:
                raise ValueError("empty Responses output: " + json.dumps(payload)[:1000])
            detections = parse_output(raw)
            return detections, raw, payload.get("id"), payload.get("usage", {})
        except Exception as error:  # transport and malformed model replies are retryable
            last = error
            if attempt + 1 < retries:
                time.sleep(min(30, 2 ** attempt))
    raise last


def annotate_job(job, key, args):
    scene, node, images, cache_key = job
    started = time.time()
    try:
        detections, raw, response_id, usage = annotate_node(
            images, key, args.base_url, args.model, args.wire_api, args.retries
        )
        return cache_key, {
            "status": "ok",
            "prompt_version": PROMPT_VERSION,
            "parser_version": PARSER_VERSION,
            "model": args.model,
            "wire_api": args.wire_api,
            "detections": detections,
            "response_id": response_id,
            "usage": usage,
            "raw": raw,
            "latency_s": round(time.time() - started, 3),
        }
    except Exception as error:
        return cache_key, {
            "status": "error",
            "prompt_version": PROMPT_VERSION,
            "model": args.model,
            "wire_api": args.wire_api,
            "detections": [],
            "error": repr(error),
            "latency_s": round(time.time() - started, 3),
        }


def run(args):
    root = Path(args.root).resolve()
    source = root / args.source
    output = root / args.output
    output.mkdir(parents=True, exist_ok=True)
    key = load_key(root, args.env)
    # Keep new experiment artifacts provider-neutral: the hosted service is a
    # transport detail, while ``model`` is the scientific variable.  Continue
    # to read legacy caches so earlier completed experiments remain reusable.
    cache_path = output / "gpt_rgbd_responses.json"
    legacy_cache_path = output / "peterai_rgbd_responses.json"
    if not cache_path.exists() and legacy_cache_path.exists():
        cache_path = legacy_cache_path
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    for item in cache.values():
        if item.get("status") == "ok" and item.get("raw"):
            try:
                item["detections"] = parse_output(item["raw"])
                item["parser_version"] = PARSER_VERSION
            except Exception:
                pass
    if cache:
        write_json_atomic(cache_path, cache)

    maps = []
    jobs = []
    for path in sorted(source.glob("*/rgbd_topological_map.json")):
        mapping = json.loads(path.read_text())
        scene = path.parent.name
        maps.append((scene, mapping))
        for node in mapping["nodes"]:
            images = [root / view["rgb_path"] for view in node["rgbd_views"]]
            cache_key = f"{scene}/node_{int(node['id']):04d}"
            jobs.append((scene, node, images, cache_key))

    todo = [
        job for job in jobs
        if cache.get(job[3], {}).get("status") != "ok"
        or cache.get(job[3], {}).get("prompt_version") != PROMPT_VERSION
        or cache.get(job[3], {}).get("model") != args.model
        or cache.get(job[3], {}).get("wire_api") != args.wire_api
    ]
    if args.cache_only:
        todo = []
    if args.limit is not None:
        todo = todo[:args.limit]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(annotate_job, job, key, args): job for job in todo}
        for index, future in enumerate(as_completed(futures), 1):
            cache_key, result = future.result()
            cache[cache_key] = result
            if index % args.checkpoint_every == 0 or index == len(todo):
                write_json_atomic(cache_path, cache)
            print(
                f"[{index}/{len(todo)}] {cache_key}: {result['status']}, "
                f"{len(result['detections'])} detections, {result['latency_s']} s",
                flush=True,
            )

    # A limited smoke test only updates the cache; a complete run materializes
    # evaluation-ready scene maps after every job is available.
    if args.limit is not None:
        return
    for scene, mapping in maps:
        for node in mapping["nodes"]:
            cache_key = f"{scene}/node_{int(node['id']):04d}"
            node["localized_vlm"] = cache.get(cache_key, {
                "status": "missing", "detections": [], "prompt_version": PROMPT_VERSION
            })
        scene_output = output / scene
        scene_output.mkdir(exist_ok=True)
        (scene_output / "rgbd_semantic_map.json").write_text(json.dumps(mapping, indent=2))

    results = [cache[f"{scene}/node_{int(node['id']):04d}"] for scene, mapping in maps for node in mapping["nodes"]]
    ok = [item for item in results if item["status"] == "ok"]
    report = {
        "model": args.model,
        "wire_api": args.wire_api,
        "prompt_version": PROMPT_VERSION,
        "parser_version": PARSER_VERSION,
        "cached_prompt_versions": {version: sum(item.get("prompt_version") == version for item in results)
                                   for version in sorted({item.get("prompt_version", "missing") for item in results})},
        "nodes": len(results),
        "api_successes": len(ok),
        "api_success_rate": len(ok) / len(results),
        "detections": sum(len(item.get("detections", [])) for item in results),
        "empty_successful_nodes": sum(not item.get("detections") for item in ok),
        "usage": {
            key: sum(int(item.get("usage", {}).get(key, 0) or 0) for item in ok)
            for key in ("input_tokens", "output_tokens", "prompt_tokens", "completion_tokens", "total_tokens")
        },
    }
    (output / "semantic_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--source", default="outputs/hm3d_minival_uniform/rgbd_capture")
    parser.add_argument("--output", default="outputs/hm3d_minival_uniform/gpt54_rgbd_semantics")
    parser.add_argument("--env", default="../.env")
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--wire-api", choices=("responses", "chat"), default="responses")
    parser.add_argument("--base-url", default="https://api.peterai.cc.cd/v1")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--checkpoint-every", type=int, default=20,
                        help="Persist the resumable response cache every N completed requests.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cache-only", action="store_true", help="Reparse cached raw responses and rebuild maps without API calls.")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
