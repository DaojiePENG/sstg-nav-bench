"""Evaluate autonomous sequential stopping with cached arrival verification."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from .benchmark import load_scene_episodes, make_sim, scene_paths, shortest
from .experiments import bootstrap_ci, cached_goal_distance, prepare_scene, wilson


def load_candidate_maps(source: Path) -> dict[str, dict[int, dict]]:
    result = {}
    for path in source.glob("*/arrival_candidates.json"):
        data = json.loads(path.read_text())
        result[data["scene"]] = {int(key): value for key, value in data["candidates"].items()}
    return result


def verifier_accept(response: dict, policy: str, min_depth: float, max_depth: float) -> bool:
    """Re-evaluate a cached arrival response under an explicit STOP policy.

    ``cached`` preserves the original VLM-side decision exactly.  The other
    policies expose reproducible arrival-signal ablations without issuing a
    new API request or reading ObjectNav goals.
    """
    if response.get("status") != "ok":
        return False
    decision = response.get("decision", {}) or {}
    if policy == "cached":
        return bool(decision.get("accept"))
    visible = bool(decision.get("target_visible"))
    if policy == "visible":
        return visible
    depth = decision.get("depth_m")
    depth_valid = depth is not None and min_depth <= float(depth) <= max_depth
    if policy == "visible_depth":
        return visible and depth_valid
    raise ValueError(f"unknown decision policy: {policy}")


def prefix_metrics(attempts: list[dict], optimal: float, max_k: int) -> dict:
    output = {}
    for k in range(1, max_k + 1):
        prefix = [attempt for attempt in attempts if attempt["rank"] <= k]
        accepted = next((attempt for attempt in prefix if attempt["verifier_accept"]), None)
        success = float(bool(accepted and accepted["official_success_at_candidate"]))
        distance = accepted["cumulative_distance"] if accepted else math.inf
        output[f"success_at_{k}"] = success
        output[f"spl_at_{k}"] = optimal / max(optimal, distance) if success else 0.0
    return output


def run(args) -> None:
    root = Path(args.root).resolve()
    source = root / args.source
    verifier = root / args.verifier
    output = root / args.output
    output.mkdir(parents=True, exist_ok=True)
    dataset = root / "data/datasets/objectnav_hm3d_v2" / args.split
    scenes = root / "data/hm3d" / args.scene_dir
    sequences = json.loads((source / "episode_candidates.json").read_text())
    candidate_maps = load_candidate_maps(source)
    decisions = json.loads((verifier / "verifier_decisions.json").read_text())
    rows = []
    all_attempts = []

    for _, data in load_scene_episodes(dataset):
        _, scene, base, nav = scene_paths(scenes, data["episodes"][0]["scene_id"])
        if scene not in candidate_maps:
            continue
        sim = make_sim(base, nav)
        nodes = list(candidate_maps[scene].values())
        _, goal_cache = prepare_scene(sim, data, nodes)
        for episode in data["episodes"]:
            category = episode["object_category"]
            key = f"{scene}_{category}_{episode['episode_id']}"
            sequence = sequences[key]["candidate_ids"]
            current = episode["start_position"]
            cumulative = 0.0
            attempts = []
            for rank, candidate_id in enumerate(sequence, 1):
                candidate = candidate_maps[scene][int(candidate_id)]
                ok, leg, route = shortest(sim, current, candidate["position"])
                if not ok or not math.isfinite(leg):
                    continue
                cumulative += leg
                cache_key = f"{scene}/candidate_{int(candidate_id):05d}"
                response = decisions.get(cache_key, {}) or {}
                decision = response.get("decision", {}) or {}
                accept = verifier_accept(
                    response,
                    args.decision_policy,
                    args.min_verifier_depth,
                    args.max_verifier_depth,
                )
                dtg = cached_goal_distance(goal_cache, candidate, category)
                official_success = bool(dtg <= args.success_distance)
                attempt = {
                    "episode": key,
                    "rank": rank,
                    "candidate_id": int(candidate_id),
                    "category": category,
                    "leg_distance": leg,
                    "cumulative_distance": cumulative,
                    "verifier_accept": accept,
                    "verifier_confidence": decision.get("confidence"),
                    "verifier_depth_m": decision.get("depth_m"),
                    "verifier_reason": decision.get("reason_code"),
                    "official_dtg": dtg,
                    "official_success_at_candidate": official_success,
                    "route": [point.tolist() for point in route],
                }
                attempts.append(attempt)
                all_attempts.append(attempt)
                current = candidate["position"]
                if accept:
                    break
            accepted = next((attempt for attempt in attempts if attempt["verifier_accept"]), None)
            success = float(bool(accepted and accepted["official_success_at_candidate"]))
            optimal = float(episode["info"]["geodesic_distance"])
            planned = accepted["cumulative_distance"] if accepted else cumulative
            row = {
                "episode": key,
                "scene": scene,
                "category": category,
                "candidate_count": sequences[key]["candidate_count"],
                "candidate_sequence_count": len(sequence),
                "attempted": len(attempts),
                "accepted_rank": accepted["rank"] if accepted else 0,
                "success": success,
                "spl": optimal / max(optimal, planned) if success else 0.0,
                "dtg": accepted["official_dtg"] if accepted else (attempts[-1]["official_dtg"] if attempts else math.inf),
                "optimal_distance": optimal,
                "planned_distance": planned,
                "failure": "" if success else ("false_accept" if accepted else "no_verifier_accept"),
                "attempts_json": json.dumps(attempts, separators=(",", ":")),
            }
            row.update(prefix_metrics(attempts, optimal, args.max_k))
            rows.append(row)
        sim.close()

    with (output / "episodes_arrival_verified.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    with (output / "arrival_attempts.csv").open("w", newline="") as file:
        attempt_rows = [{key: value for key, value in attempt.items() if key != "route"} for attempt in all_attempts]
        writer = csv.DictWriter(file, fieldnames=attempt_rows[0].keys())
        writer.writeheader()
        writer.writerows(attempt_rows)

    true_accept = sum(a["verifier_accept"] and a["official_success_at_candidate"] for a in all_attempts)
    false_accept = sum(a["verifier_accept"] and not a["official_success_at_candidate"] for a in all_attempts)
    false_reject = sum(not a["verifier_accept"] and a["official_success_at_candidate"] for a in all_attempts)
    true_reject = sum(not a["verifier_accept"] and not a["official_success_at_candidate"] for a in all_attempts)
    summary = {
        "protocol": "fresh arrival RGB-D and GPT-5.4 verifier control STOP/continue; official goals used only for post-hoc scoring",
        "decision_policy": args.decision_policy,
        "verifier_depth_range_m": [args.min_verifier_depth, args.max_verifier_depth],
        "episodes": len(rows),
        "max_k": args.max_k,
        "candidate_min_separation_m": args.min_separation,
        "successes": int(sum(row["success"] for row in rows)),
        "sr": float(np.mean([row["success"] for row in rows])),
        "sr_95ci": wilson(sum(row["success"] for row in rows), len(rows)),
        "spl": float(np.mean([row["spl"] for row in rows])),
        "spl_95ci": bootstrap_ci([row["spl"] for row in rows], seed=20260720),
        "dtg": float(np.mean([row["dtg"] for row in rows if math.isfinite(row["dtg"])])),
        "failure_counts": {
            "success": sum(not row["failure"] for row in rows),
            "false_accept": sum(row["failure"] == "false_accept" for row in rows),
            "no_verifier_accept": sum(row["failure"] == "no_verifier_accept" for row in rows),
        },
        "execution": {
            "mean_attempts": float(np.mean([row["attempted"] for row in rows])),
            "accepted_rank_1": sum(row["accepted_rank"] == 1 for row in rows),
            "accepted_rank_2": sum(row["accepted_rank"] == 2 for row in rows),
            "accepted_rank_3": sum(row["accepted_rank"] == 3 for row in rows),
            "no_accept": sum(row["accepted_rank"] == 0 for row in rows),
        },
        "verifier_attempt_confusion": {
            "true_accept": true_accept,
            "false_accept": false_accept,
            "false_reject": false_reject,
            "true_reject": true_reject,
            "precision": true_accept / max(1, true_accept + false_accept),
            "recall": true_accept / max(1, true_accept + false_reject),
        },
        "metrics": {},
        "per_category": {},
    }
    for k in range(1, args.max_k + 1):
        successes = sum(row[f"success_at_{k}"] for row in rows)
        spl = [row[f"spl_at_{k}"] for row in rows]
        summary["metrics"][str(k)] = {
            "successes": int(successes),
            "sr": successes / len(rows),
            "sr_95ci": wilson(successes, len(rows)),
            "spl": float(np.mean(spl)),
            "spl_95ci": bootstrap_ci(spl, seed=20260720 + k),
        }
    for category in sorted({row["category"] for row in rows}):
        category_rows = [row for row in rows if row["category"] == category]
        summary["per_category"][category] = {
            "n": len(category_rows),
            "sr": float(np.mean([row["success"] for row in category_rows])),
            "spl": float(np.mean([row["spl"] for row in category_rows])),
            "mean_attempts": float(np.mean([row["attempted"] for row in category_rows])),
        }
    (output / "summary_arrival_verified.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--split", default="val")
    parser.add_argument("--scene-dir", default="val")
    parser.add_argument("--source", default="outputs/hm3d_val_uniform/gpt54_arrival_capture_top3")
    parser.add_argument("--verifier", default="outputs/hm3d_val_uniform/gpt54_arrival_verifier_top3_strict")
    parser.add_argument("--output", default="outputs/hm3d_val_uniform/gpt54_arrival_verified_top3_strict")
    parser.add_argument("--max-k", type=int, default=3)
    parser.add_argument("--min-separation", type=float, default=3.0)
    parser.add_argument("--success-distance", type=float, default=1.0)
    parser.add_argument(
        "--decision-policy",
        choices=("cached", "visible", "visible_depth"),
        default="cached",
        help="STOP rule applied to the immutable cached verifier response.",
    )
    parser.add_argument("--min-verifier-depth", type=float, default=0.25)
    parser.add_argument("--max-verifier-depth", type=float, default=2.5)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
