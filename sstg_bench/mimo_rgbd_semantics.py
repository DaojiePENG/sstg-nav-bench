"""MiMo-v2.5 multi-view grounding for the independent RGB-D topology."""
from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import re
import time

from openai import OpenAI

from .peterai_rgbd_semantics import PARSER_VERSION, PROMPT, parse_output


PROMPT_VERSION = "mimo-v2.5-rgbd-grounding-v2-variable-resolution"


def load_key(root: Path, env_path: str) -> str:
    key = os.getenv("MIMO_API_KEY", "").strip()
    path = root / env_path
    if not key and path.exists():
        for line in path.read_text().splitlines():
            match = re.match(r"\s*MIMO_API_KEY\s*=\s*(.*?)\s*$", line)
            if match:
                key = match.group(1).strip().strip('"\'')
                break
    if not key:
        raise SystemExit("MIMO_API_KEY is not set and was not found in --env")
    return key


def annotate_node(client, images, model, retries):
    image_content = []
    for index, image in enumerate(images):
        encoded = base64.b64encode(image.read_bytes()).decode()
        image_content.extend([
            {"type": "text", "text": f"VIEW {index}:"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + encoded}},
        ])
    messages = [
        {"role": "system", "content": "You are MiMo, an AI assistant developed by Xiaomi."},
        {"role": "user", "content": PROMPT},
        {"role": "user", "content": image_content},
    ]
    last = None
    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                max_completion_tokens=1600,
                stream=False,
                extra_body={"thinking": {"type": "disabled"}},
            )
            raw = completion.choices[0].message.content or ""
            if not raw:
                raise ValueError("empty MiMo assistant content")
            usage = completion.usage.model_dump() if completion.usage else {}
            return parse_output(raw), raw, completion.id, usage
        except Exception as error:
            last = error
            if attempt + 1 < retries:
                time.sleep(min(30, 2 ** attempt))
    raise last


def annotate_job(job, client, args):
    _, _, images, cache_key = job
    started = time.time()
    try:
        detections, raw, response_id, usage = annotate_node(client, images, args.model, args.retries)
        return cache_key, {
            "status": "ok",
            "prompt_version": PROMPT_VERSION,
            "parser_version": PARSER_VERSION,
            "model": args.model,
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
            "detections": [],
            "error": repr(error),
            "latency_s": round(time.time() - started, 3),
        }


def run(args):
    root = Path(args.root).resolve()
    source = root / args.source
    output = root / args.output
    output.mkdir(parents=True, exist_ok=True)
    client = OpenAI(
        api_key=load_key(root, args.env),
        base_url=args.base_url,
        timeout=args.timeout,
        max_retries=0,
    )
    cache_path = output / "mimo_rgbd_responses.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    for item in cache.values():
        if item.get("status") == "ok" and item.get("raw"):
            try:
                item["detections"] = parse_output(item["raw"])
                item["parser_version"] = PARSER_VERSION
            except Exception:
                pass
    if cache:
        cache_path.write_text(json.dumps(cache, indent=2))

    maps, jobs = [], []
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
    ]
    if args.cache_only:
        todo = []
    if args.limit is not None:
        todo = todo[:args.limit]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(annotate_job, job, client, args): job for job in todo}
        for index, future in enumerate(as_completed(futures), 1):
            cache_key, result = future.result()
            cache[cache_key] = result
            if index % args.checkpoint_every == 0 or index == len(todo):
                cache_path.write_text(json.dumps(cache, indent=2))
            print(
                f"[{index}/{len(todo)}] {cache_key}: {result['status']}, "
                f"{len(result['detections'])} detections, {result['latency_s']} s",
                flush=True,
            )

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

    results = [
        cache[f"{scene}/node_{int(node['id']):04d}"]
        for scene, mapping in maps for node in mapping["nodes"]
    ]
    ok = [item for item in results if item["status"] == "ok"]
    usage_keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    report = {
        "model": args.model,
        "base_url": args.base_url,
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
            for key in usage_keys
        },
    }
    (output / "semantic_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--source", default="outputs/hm3d_minival_uniform/rgbd_capture")
    parser.add_argument("--output", default="outputs/hm3d_minival_uniform/mimo_rgbd_semantics")
    parser.add_argument("--env", default="../.env")
    parser.add_argument("--model", default="mimo-v2.5")
    parser.add_argument("--base-url", default="https://token-plan-cn.xiaomimimo.com/v1")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--checkpoint-every", type=int, default=20,
                        help="Persist the resumable response cache every N completed requests.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--cache-only", action="store_true", help="Reparse cached raw responses and rebuild maps without API calls.")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
