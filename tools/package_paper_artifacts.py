"""Build a compact, auditable bundle of the artifacts used by the paper.

The full output tree intentionally keeps RGB-D frames, videos, and diagnostic
runs.  This tool materializes only manuscript-facing numeric evidence and map
metadata under ``outputs/paper_core``.  Files are hard-linked when possible so
the organized view does not duplicate storage; the optional tarball is a
portable copy suitable for a supplementary upload.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
DESTINATION = OUTPUTS / "paper_core"
ARCHIVE = OUTPUTS / "sstg_nav_paper_core.tar.gz"
CLEANUP_REPORT = OUTPUTS / "output_cleanup_report.json"

# These are the exact full-validation artifacts behind the manuscript tables.
# Large RGB/depth arrays and videos remain in their original directories and
# are indexed separately below.
CORE_PATTERNS = (
    "outputs/release_audit.json",
    "outputs/output_cleanup_report.json",
    "outputs/hm3d_val_uniform/VISUAL_INDEX.md",
    "outputs/hm3d_val_uniform/benchmark_summary.csv",
    "outputs/hm3d_val_uniform/benchmark_summary.json",
    "outputs/hm3d_val_uniform/paired_summary.csv",
    "outputs/hm3d_val_uniform/paired_*.json",
    "outputs/hm3d_val_uniform/model_identity_audit.json",
    "outputs/hm3d_val_uniform/source/sampling_report.json",
    "outputs/hm3d_val_uniform/source/*/vlm_topological_map.json",
    "outputs/hm3d_val_uniform/rgbd_capture_wide120/capture_report.json",
    "outputs/hm3d_val_uniform/rgbd_capture_wide120/*/rgbd_topological_map.json",
    "outputs/hm3d_val_uniform/gpt54_rgbd_semantics_wide120/semantic_report.json",
    "outputs/hm3d_val_uniform/gpt54_rgbd_semantics_wide120/gpt_rgbd_responses.json",
    "outputs/hm3d_val_uniform/gpt54_rgbd_semantics_wide120/*/rgbd_semantic_map.json",
    "outputs/hm3d_val_uniform/gpt54_rgbd_wide120_fusion/fusion_report.json",
    "outputs/hm3d_val_uniform/gpt54_rgbd_wide120_fusion/raw_maps/*/vlm_topological_map.json",
    "outputs/hm3d_val_uniform/gpt54_rgbd_wide120_fusion/clustered_maps/*/vlm_topological_map.json",
    "outputs/hm3d_val_uniform/gpt54_rgbd_wide120_fusion/multi_standoff_maps/*/vlm_topological_map.json",
    "outputs/hm3d_val_uniform/oracle_geometry/*.json",
    "outputs/hm3d_val_uniform/oracle_geometry/*.csv",
    "outputs/hm3d_val_uniform/gpt54_camera_node_analysis/*.json",
    "outputs/hm3d_val_uniform/gpt54_camera_node_analysis/*.csv",
    "outputs/hm3d_val_uniform/gpt54_rgbd_wide120_analysis_raw/*.json",
    "outputs/hm3d_val_uniform/gpt54_rgbd_wide120_analysis_raw/*.csv",
    "outputs/hm3d_val_uniform/gpt54_rgbd_wide120_analysis_fused/*.json",
    "outputs/hm3d_val_uniform/gpt54_rgbd_wide120_analysis_fused/*.csv",
    "outputs/hm3d_val_uniform/gpt54_rgbd_wide120_raw_topk_0m/*.json",
    "outputs/hm3d_val_uniform/gpt54_rgbd_wide120_raw_topk_0m/*.csv",
    "outputs/hm3d_val_uniform/gpt54_rgbd_wide120_raw_topk_2p0m/*.json",
    "outputs/hm3d_val_uniform/gpt54_rgbd_wide120_raw_topk_2p0m/*.csv",
    "outputs/hm3d_val_uniform/gpt54_rgbd_wide120_topk_0m/*.json",
    "outputs/hm3d_val_uniform/gpt54_rgbd_wide120_topk_0m/*.csv",
    "outputs/hm3d_val_uniform/gpt54_rgbd_wide120_fused_topk_2p0m/*.json",
    "outputs/hm3d_val_uniform/gpt54_rgbd_wide120_fused_topk_2p0m/*.csv",
    "outputs/hm3d_val_uniform/gpt54_rgbd_wide120_fusion_first_raw_topk_2p0m/*.json",
    "outputs/hm3d_val_uniform/gpt54_rgbd_wide120_fusion_first_raw_topk_2p0m/*.csv",
    "outputs/hm3d_val_uniform/gpt54_arrival_capture_fusion_aware_top3/capture_report.json",
    "outputs/hm3d_val_uniform/gpt54_arrival_capture_fusion_aware_top3/episode_candidates.json",
    "outputs/hm3d_val_uniform/gpt54_arrival_capture_fusion_aware_top3/*/arrival_candidates.json",
)

DOCUMENTS = {
    ROOT / "README.md": Path("docs/BENCHMARK_README.md"),
    ROOT.parent / "SSTGNavPaperAAAI" / "ARTIFACT_INDEX.md": Path("docs/ARTIFACT_INDEX.md"),
    ROOT / "tools" / "run_full_rgbd_benchmark.sh": Path("code/run_full_rgbd_benchmark.sh"),
    ROOT / "tools" / "validate_release.py": Path("code/validate_release.py"),
    ROOT / "tools" / "package_paper_artifacts.py": Path("code/package_paper_artifacts.py"),
    ROOT / "tools" / "verify_paper_bundle.py": Path("verify_paper_bundle.py"),
}

MEDIA_ROOTS = (
    "outputs/hm3d_val_uniform/rgbd_capture_wide120",
    "outputs/hm3d_val_uniform/gpt54_arrival_capture_fusion_aware_top3",
    "outputs/hm3d_val_uniform/gpt54_rgbd_wide120_visuals",
    "outputs/hm3d_val_uniform/gpt54_arrival_verified_visuals",
    "outputs/chair_four_view_sets",
)
MEDIA_SUFFIXES = {".jpg", ".jpeg", ".png", ".mp4", ".npy"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def resolve_core_files() -> list[Path]:
    paths: set[Path] = set()
    for pattern in CORE_PATTERNS:
        matches = [path for path in ROOT.glob(pattern) if path.is_file()]
        if not matches:
            raise FileNotFoundError(f"Required paper artifact pattern matched nothing: {pattern}")
        paths.update(matches)
    return sorted(paths)


def clean_obsolete_tables() -> dict:
    """Remove tiny, unreferenced RAL-era table fragments, never media/results."""
    removed = []
    total_bytes = 0
    for path in sorted(OUTPUTS.rglob("ral_results_table.tex")):
        total_bytes += path.stat().st_size
        removed.append(str(path.relative_to(ROOT)))
        path.unlink()
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "derived, unreferenced RAL-era LaTeX table fragments only",
        "removed_files": len(removed),
        "removed_bytes": total_bytes,
        "paths": removed,
        "preserved_media": True,
        "preserved_evaluation_json_csv": True,
        "preserved_diagnostic_runs": True,
    }
    CLEANUP_REPORT.write_text(json.dumps(report, indent=2) + "\n")
    return report


def media_inventory() -> dict:
    inventory = {}
    for relative in MEDIA_ROOTS:
        root = ROOT / relative
        suffix_counts: Counter[str] = Counter()
        total_bytes = 0
        samples = []
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in MEDIA_SUFFIXES:
                continue
            suffix_counts[path.suffix.lower()] += 1
            total_bytes += path.stat().st_size
            if len(samples) < 12:
                samples.append(str(path.relative_to(ROOT)))
        inventory[relative] = {
            "files": sum(suffix_counts.values()),
            "bytes": total_bytes,
            "counts_by_suffix": dict(sorted(suffix_counts.items())),
            "sample_paths": samples,
            "status": "preserved in full output tree; intentionally not duplicated in compact archive",
        }
    return inventory


def render_readme(manifest: dict, media: dict) -> str:
    return f"""# SSTG-Nav paper-core artifact bundle

This directory is the curated, manuscript-facing subset of the full benchmark
outputs. It preserves the exact 1,000-episode evidence behind the paper while
the original RGB-D frames, images, depth arrays, and videos remain untouched in
the full `outputs/` tree.

## Headline full-validation results

| Policy | S@1 | S@2 | S@3 | SPL@end |
|---|---:|---:|---:|---:|
| GPT-5.4 raw single | 0.920 | -- | -- | 0.603 |
| GPT-5.4 raw Top-3 (2 m) | 0.920 | 0.963 | 0.975 | 0.616 |
| GPT-5.4 fused single | 0.926 | -- | -- | 0.586 |
| Fused representatives Top-3 (2 m) | 0.926 | 0.952 | 0.964 | 0.596 |
| Fusion-aware residual Top-3 | **0.928** | **0.965** | **0.975** | **0.601** |

The final policy visits a fused representative first, then at most two
confidence-ranked raw metric residual standoffs with 2 m spatial diversity.
This is the corrected fused+Top-3 comparison; Qwen is only an auxiliary
small-split control.

## Contents

- `data/outputs/hm3d_val_uniform/`: exact summaries, per-episode CSVs,
  topology/capture metadata, GPT-5.4 response cache, raw/fused maps, and final
  arrival-candidate metadata using their original repository-relative paths.
- `manifest.json`: SHA-256, byte size, and provenance for every bundled file.
- `media_manifest.json`: counts and locations of preserved large visual media.
- `docs/`: benchmark instructions and the paper claim-to-artifact index.
- `verify_paper_bundle.py`: dependency-free checksum and metric verifier.

## Verify after extraction

```bash
python verify_paper_bundle.py .
```

The bundle contains {manifest['file_count']} hashed source artifacts
({manifest['source_bytes'] / (1024 ** 2):.1f} MiB apparent size). The media
index covers {sum(item['files'] for item in media.values()):,} preserved image,
depth, and video files without copying those multi-gigabyte assets into the
supplementary archive.
"""


def build_bundle(create_archive: bool) -> dict:
    temporary = DESTINATION.with_name(DESTINATION.name + ".building")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    entries = []
    link_modes: Counter[str] = Counter()
    for source in resolve_core_files():
        relative = source.relative_to(ROOT)
        destination_relative = Path("data") / relative
        mode = link_or_copy(source, temporary / destination_relative)
        link_modes[mode] += 1
        entries.append(
            {
                "path": str(destination_relative),
                "source": str(relative),
                "bytes": source.stat().st_size,
                "sha256": sha256(source),
            }
        )

    for source, destination_relative in DOCUMENTS.items():
        if not source.is_file():
            raise FileNotFoundError(f"Missing bundle documentation/code file: {source}")
        mode = link_or_copy(source, temporary / destination_relative)
        link_modes[mode] += 1
        entries.append(
            {
                "path": str(destination_relative),
                "source": str(source.relative_to(ROOT.parent)),
                "bytes": source.stat().st_size,
                "sha256": sha256(source),
            }
        )

    entries.sort(key=lambda item: item["path"])
    manifest = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "curated evidence for the SSTG-Nav paper and supplementary material",
        "file_count": len(entries),
        "source_bytes": sum(item["bytes"] for item in entries),
        "materialization": dict(link_modes),
        "files": entries,
    }
    media = media_inventory()
    (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (temporary / "media_manifest.json").write_text(json.dumps(media, indent=2) + "\n")
    (temporary / "README.md").write_text(render_readme(manifest, media))

    if DESTINATION.exists():
        shutil.rmtree(DESTINATION)
    temporary.rename(DESTINATION)

    archive_sha256 = None
    if create_archive:
        if ARCHIVE.exists():
            ARCHIVE.unlink()
        with tarfile.open(ARCHIVE, "w:gz", compresslevel=9) as handle:
            handle.add(DESTINATION, arcname=DESTINATION.name)
        archive_sha256 = sha256(ARCHIVE)

    return {
        "destination": str(DESTINATION.relative_to(ROOT)),
        "archive": str(ARCHIVE.relative_to(ROOT)) if create_archive else None,
        "archive_sha256": archive_sha256,
        "file_count": manifest["file_count"],
        "source_bytes": manifest["source_bytes"],
        "media_files_indexed": sum(item["files"] for item in media.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cleanup-derived-tables",
        action="store_true",
        help="remove only unreferenced outputs/**/ral_results_table.tex fragments",
    )
    parser.add_argument("--no-archive", action="store_true", help="skip portable .tar.gz creation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cleanup_derived_tables:
        clean_obsolete_tables()
    elif not CLEANUP_REPORT.exists():
        CLEANUP_REPORT.write_text(
            json.dumps(
                {
                    "generated_utc": datetime.now(timezone.utc).isoformat(),
                    "scope": "no cleanup requested",
                    "removed_files": 0,
                    "removed_bytes": 0,
                    "paths": [],
                    "preserved_media": True,
                    "preserved_evaluation_json_csv": True,
                    "preserved_diagnostic_runs": True,
                },
                indent=2,
            )
            + "\n"
        )
    print(json.dumps(build_bundle(create_archive=not args.no_archive), indent=2))


if __name__ == "__main__":
    main()
