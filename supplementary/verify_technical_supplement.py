"""Verify checksums, size limits, exclusions, and headline supplement metrics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile


TEN_MIB = 10 * 1024 * 1024
FORBIDDEN_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".npy", ".jpg", ".jpeg", ".png"}
FORBIDDEN_NAMES = {".env"}


REGULAR_RESULTS = {
    "camera": (
        "evidence/outputs/hm3d_val_uniform/gpt54_camera_node_analysis/episodes_vlm_all_confidence.csv",
        "evidence/outputs/hm3d_val_uniform/gpt54_camera_node_analysis/summary_vlm_all_confidence.json",
        0.835,
        0.5601654038897288,
    ),
    "raw": (
        "evidence/outputs/hm3d_val_uniform/gpt54_rgbd_wide120_analysis_raw/episodes_vlm_all_confidence.csv",
        "evidence/outputs/hm3d_val_uniform/gpt54_rgbd_wide120_analysis_raw/summary_vlm_all_confidence.json",
        0.920,
        0.6026456464576543,
    ),
    "fused": (
        "evidence/outputs/hm3d_val_uniform/gpt54_rgbd_wide120_analysis_fused/episodes_vlm_all_confidence.csv",
        "evidence/outputs/hm3d_val_uniform/gpt54_rgbd_wide120_analysis_fused/summary_vlm_all_confidence.json",
        0.926,
        0.585766799584227,
    ),
}


TOPK_RESULTS = {
    "raw_top3": (
        "evidence/outputs/hm3d_val_uniform/gpt54_rgbd_wide120_raw_topk_2p0m",
        (0.920, 0.963, 0.975),
        0.6157510076724466,
    ),
    "fused_top3": (
        "evidence/outputs/hm3d_val_uniform/gpt54_rgbd_wide120_fused_topk_2p0m",
        (0.926, 0.952, 0.964),
        0.5963826701058228,
    ),
    "fusion_aware_top3": (
        "evidence/outputs/hm3d_val_uniform/gpt54_rgbd_wide120_fusion_first_raw_topk_2p0m",
        (0.928, 0.965, 0.975),
        0.6007234714721443,
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(actual: float, expected: float) -> bool:
    return abs(actual - expected) < 1e-9


def regular_check(root: Path, specification: tuple) -> dict:
    csv_relative, summary_relative, expected_sr, expected_spl = specification
    rows = list(csv.DictReader((root / csv_relative).open()))
    summary = json.loads((root / summary_relative).read_text())
    sr = sum(float(row["success"]) for row in rows) / len(rows)
    spl = sum(float(row["spl"]) for row in rows) / len(rows)
    passed = (
        len(rows) == 1000
        and len({row["episode"] for row in rows}) == 1000
        and close(sr, summary["sr"])
        and close(spl, summary["spl"])
        and close(sr, expected_sr)
        and close(spl, expected_spl)
    )
    return {"rows": len(rows), "sr": sr, "spl": spl, "passed": passed}


def topk_check(root: Path, specification: tuple) -> dict:
    directory_relative, expected_sr, expected_spl3 = specification
    directory = root / directory_relative
    rows = list(csv.DictReader((directory / "episodes_topk.csv").open()))
    summary = json.loads((directory / "summary_topk.json").read_text())
    sr = tuple(sum(float(row[f"success_at_{k}"]) for row in rows) / len(rows) for k in range(1, 4))
    spl3 = sum(float(row["spl_at_3"]) for row in rows) / len(rows)
    passed = (
        len(rows) == 1000
        and len({row["episode"] for row in rows}) == 1000
        and all(close(sr[k - 1], summary["metrics"][str(k)]["sr"]) for k in range(1, 4))
        and close(spl3, summary["metrics"]["3"]["spl"])
        and all(close(actual, expected) for actual, expected in zip(sr, expected_sr))
        and close(spl3, expected_spl3)
    )
    return {"rows": len(rows), "success_at_1_2_3": sr, "spl_at_3": spl3, "passed": passed}


def verify(root: Path) -> dict:
    manifest = json.loads((root / "MANIFEST.json").read_text())
    checksum_failures = []
    for entry in manifest["files"]:
        path = root / entry["path"]
        if not path.is_file():
            checksum_failures.append({"path": entry["path"], "reason": "missing"})
        elif path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            checksum_failures.append({"path": entry["path"], "reason": "size or SHA-256 mismatch"})

    forbidden = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name in FORBIDDEN_NAMES or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            forbidden.append(str(path.relative_to(root)))

    pdf = root / "SupplementaryMaterial2027.pdf"
    archive = root / "SSTGNav_TechnicalSupplement.zip"
    size_check = {
        "pdf_bytes": pdf.stat().st_size if pdf.is_file() else None,
        "archive_bytes": archive.stat().st_size if archive.is_file() else None,
        "pdf_under_10_mib": pdf.is_file() and pdf.stat().st_size < TEN_MIB,
        "archive_under_10_mib": archive.is_file() and archive.stat().st_size < TEN_MIB,
    }
    archive_media = []
    if archive.is_file():
        with zipfile.ZipFile(archive) as handle:
            for name in handle.namelist():
                suffix = Path(name).suffix.lower()
                if Path(name).name in FORBIDDEN_NAMES or suffix in FORBIDDEN_SUFFIXES:
                    archive_media.append(name)

    regular = {name: regular_check(root, spec) for name, spec in REGULAR_RESULTS.items()}
    topk = {name: topk_check(root, spec) for name, spec in TOPK_RESULTS.items()}
    passed = (
        not checksum_failures
        and not forbidden
        and not archive_media
        and all(size_check[key] for key in ("pdf_under_10_mib", "archive_under_10_mib"))
        and all(item["passed"] for item in (*regular.values(), *topk.values()))
    )
    return {
        "passed": passed,
        "manifest_files": manifest["file_count"],
        "checksum_failures": checksum_failures,
        "forbidden_files": forbidden,
        "forbidden_archive_entries": archive_media,
        "size_check": size_check,
        "regular_results": regular,
        "topk_results": topk,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", nargs="?", default=".", type=Path)
    args = parser.parse_args()
    result = verify(args.bundle.resolve())
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
