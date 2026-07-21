"""Paired comparison of two ObjectNav episode tables."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def exact_mcnemar(gains: int, losses: int) -> float:
    discordant = gains + losses
    if discordant == 0:
        return 1.0
    tail = min(gains, losses)
    probability = sum(math.comb(discordant, i) for i in range(tail + 1)) / (2 ** discordant)
    return min(1.0, 2 * probability)


def bootstrap_delta(values: np.ndarray, seed: int, iterations: int) -> list[float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(iterations, len(values)))
    means = values[indices].mean(axis=1)
    return np.quantile(means, [.025, .975]).tolist()


def load(path: Path) -> dict[str, dict]:
    rows = list(csv.DictReader(path.open()))
    result = {row["episode"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"duplicate episode keys in {path}")
    return result


def compare(a_path: Path, b_path: Path, seed: int, iterations: int) -> dict:
    a = load(a_path); b = load(b_path)
    if a.keys() != b.keys():
        raise ValueError("episode sets differ")
    keys = sorted(a)
    success_a = np.asarray([float(a[key]["success"]) for key in keys])
    success_b = np.asarray([float(b[key]["success"]) for key in keys])
    spl_delta = np.asarray([float(b[key]["spl"]) - float(a[key]["spl"]) for key in keys])
    gains = int(np.sum((success_a == 0) & (success_b == 1)))
    losses = int(np.sum((success_a == 1) & (success_b == 0)))
    by_category = defaultdict(lambda: {"n": 0, "gains": 0, "losses": 0, "delta_spl": []})
    changed = []
    for index, key in enumerate(keys):
        category = a[key]["category"]
        group = by_category[category]; group["n"] += 1; group["delta_spl"].append(float(spl_delta[index]))
        if success_a[index] != success_b[index]:
            direction = "gain" if success_b[index] > success_a[index] else "loss"
            group["gains" if direction == "gain" else "losses"] += 1
            changed.append({"episode": key, "category": category, "direction": direction,
                            "a_failure": a[key].get("failure", ""), "b_failure": b[key].get("failure", "")})
    category_summary = {
        category: {"n": values["n"], "gains": values["gains"], "losses": values["losses"],
                   "delta_spl": float(np.mean(values["delta_spl"]))}
        for category, values in sorted(by_category.items())
    }
    return {
        "a": str(a_path), "b": str(b_path), "episodes": len(keys),
        "sr_a": float(success_a.mean()), "sr_b": float(success_b.mean()),
        "delta_sr": float((success_b - success_a).mean()),
        "gains": gains, "losses": losses, "mcnemar_exact_two_sided_p": exact_mcnemar(gains, losses),
        "spl_a": float(np.mean([float(a[key]["spl"]) for key in keys])),
        "spl_b": float(np.mean([float(b[key]["spl"]) for key in keys])),
        "delta_spl": float(spl_delta.mean()),
        "delta_spl_95ci": bootstrap_delta(spl_delta, seed, iterations),
        "per_category": category_summary, "changed_episodes": changed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--iterations", type=int, default=10000)
    args = parser.parse_args()
    result = compare(args.a, args.b, args.seed, args.iterations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
