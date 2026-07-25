"""Build the submission-ready SSTG-Nav technical-supplement package.

The package keeps the PDF, compact numeric evidence, prompts, environment
files, and the algorithm modules needed to interpret the reported results.
Raw RGB-D, images, videos, model caches larger than the evidence budget, scene
assets, and API credentials are deliberately excluded.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = ROOT.parent / "SSTGNavPaperAAAI"
DESTINATION = ROOT / "supplementary"
ARCHIVE_NAME = "SSTGNav_TechnicalSupplement.zip"
TEN_MIB = 10 * 1024 * 1024


EVIDENCE_FILES = (
    # Cross-file audit and compact headline tables.
    "outputs/release_audit.json",
    "outputs/hm3d_val_uniform/benchmark_summary.json",
    "outputs/hm3d_val_uniform/benchmark_summary.csv",
    "outputs/hm3d_val_uniform/paired_summary.csv",
    "outputs/hm3d_val_uniform/model_identity_audit.json",
    "outputs/hm3d_val_uniform/paired_camera_to_raw.json",
    "outputs/hm3d_val_uniform/paired_raw_to_fused.json",
    "outputs/hm3d_val_uniform/paired_camera_to_fused.json",
    # Full goal-independent geometry and camera/raw/fused evaluations.
    "outputs/hm3d_val_uniform/oracle_geometry/density_ablation.json",
    "outputs/hm3d_val_uniform/oracle_geometry/density_ablation.csv",
    "outputs/hm3d_val_uniform/oracle_geometry/episodes_uniform_density_1.csv",
    "outputs/hm3d_val_uniform/gpt54_camera_node_analysis/summary_vlm_all_confidence.json",
    "outputs/hm3d_val_uniform/gpt54_camera_node_analysis/episodes_vlm_all_confidence.csv",
    "outputs/hm3d_val_uniform/gpt54_rgbd_wide120_analysis_raw/summary_vlm_all_confidence.json",
    "outputs/hm3d_val_uniform/gpt54_rgbd_wide120_analysis_raw/episodes_vlm_all_confidence.csv",
    "outputs/hm3d_val_uniform/gpt54_rgbd_wide120_analysis_fused/summary_vlm_all_confidence.json",
    "outputs/hm3d_val_uniform/gpt54_rgbd_wide120_analysis_fused/episodes_vlm_all_confidence.csv",
    "outputs/hm3d_val_uniform/gpt54_rgbd_semantics_wide120/semantic_report.json",
    "outputs/hm3d_val_uniform/gpt54_rgbd_wide120_fusion/fusion_report.json",
    # Full candidate recovery and sensitivity controls.
    "outputs/hm3d_val_uniform/gpt54_rgbd_wide120_raw_topk_0m/summary_topk.json",
    "outputs/hm3d_val_uniform/gpt54_rgbd_wide120_raw_topk_2p0m/summary_topk.json",
    "outputs/hm3d_val_uniform/gpt54_rgbd_wide120_raw_topk_2p0m/episodes_topk.csv",
    "outputs/hm3d_val_uniform/gpt54_rgbd_wide120_raw_topk_3m/summary_topk.json",
    "outputs/hm3d_val_uniform/gpt54_rgbd_wide120_topk_0m/summary_topk.json",
    "outputs/hm3d_val_uniform/gpt54_rgbd_wide120_fused_topk_2p0m/summary_topk.json",
    "outputs/hm3d_val_uniform/gpt54_rgbd_wide120_fused_topk_2p0m/episodes_topk.csv",
    "outputs/hm3d_val_uniform/gpt54_rgbd_wide120_fused_topk_3m/summary_topk.json",
    "outputs/hm3d_val_uniform/gpt54_rgbd_wide120_fusion_first_raw_topk_2p0m/summary_topk.json",
    "outputs/hm3d_val_uniform/gpt54_rgbd_wide120_fusion_first_raw_topk_2p0m/episodes_topk.csv",
    # Fresh-arrival capture and verifier calibration, without RGB-D media.
    "outputs/hm3d_val_uniform/gpt54_arrival_capture_fusion_aware_top3/capture_report.json",
    "outputs/hm3d_val_uniform/gpt54_arrival_capture_fusion_aware_top3/episode_candidates.json",
    "outputs/hm3d_val_uniform/gpt54_arrival_capture_top3/capture_report.json",
    "outputs/hm3d_val_uniform/gpt54_arrival_capture_top3/episode_candidates.json",
    "outputs/hm3d_val_uniform/gpt54_arrival_verifier_top3_strict/verifier_report.json",
    "outputs/hm3d_val_uniform/gpt54_arrival_verifier_top3_strict/gpt_arrival_responses.json",
    "outputs/hm3d_val_uniform/gpt54_arrival_verified_top3_strict/summary_arrival_verified.json",
    "outputs/hm3d_val_uniform/gpt54_arrival_verified_top3_strict/episodes_arrival_verified.csv",
    "outputs/hm3d_val_uniform/gpt54_arrival_verified_top3_strict/arrival_attempts.csv",
    # Target-view controls.
    "outputs/hm3d_val_oracle_analysis/summary_oracle.json",
    "outputs/hm3d_val_oracle_analysis/episodes_oracle.csv",
    "outputs/hm3d_val_qwen_analysis/all_summaries.json",
    "outputs/hm3d_val_qwen_analysis/summary_vlm_all_confidence.json",
    "outputs/hm3d_val_qwen_analysis/episodes_vlm_all_confidence.csv",
    # GPT-5.5 semantic-isolation pilot and corruption grid.
    "outputs/hm3d_minival_vlm/maps/mapping_report.json",
    "outputs/hm3d_val_vlm/maps/mapping_report.json",
    "outputs/analysis/all_summaries.json",
    "outputs/analysis/episodes_vlm_all_confidence.csv",
    "outputs/analysis/stress_aggregate.csv",
    "outputs/analysis/stress_runs.csv",
    # Goal-independent mini controls for GPT-5.4, MiMo, and Qwen.
    "outputs/hm3d_minival_uniform/analysis_fixed/summary_vlm_all_confidence.json",
    "outputs/hm3d_minival_uniform/peterai_camera_node_analysis/summary_vlm_all_confidence.json",
    "outputs/hm3d_minival_uniform/peterai_rgbd_analysis_raw/summary_vlm_all_confidence.json",
    "outputs/hm3d_minival_uniform/peterai_rgbd_analysis_soft/summary_vlm_all_confidence.json",
    "outputs/hm3d_minival_uniform/peterai_camera_node_wide120_analysis/summary_vlm_all_confidence.json",
    "outputs/hm3d_minival_uniform/peterai_rgbd_wide120_analysis_raw/summary_vlm_all_confidence.json",
    "outputs/hm3d_minival_uniform/peterai_rgbd_wide120_analysis_soft/summary_vlm_all_confidence.json",
    "outputs/hm3d_minival_uniform/paired_gpt90_camera_to_fusion.json",
    "outputs/hm3d_minival_uniform/mimo_camera_node_analysis/summary_vlm_all_confidence.json",
    "outputs/hm3d_minival_uniform/mimo_rgbd_analysis_reparsed_raw/summary_vlm_all_confidence.json",
    "outputs/hm3d_minival_uniform/mimo_rgbd_analysis_reparsed_soft/summary_vlm_all_confidence.json",
    "outputs/hm3d_minival_uniform/mimo_camera_node_wide120_analysis/summary_vlm_all_confidence.json",
    "outputs/hm3d_minival_uniform/mimo_rgbd_wide120_analysis_raw/summary_vlm_all_confidence.json",
    "outputs/hm3d_minival_uniform/mimo_rgbd_wide120_analysis_soft/summary_vlm_all_confidence.json",
    "outputs/hm3d_minival_uniform/qwen_rgbd_90_analysis_camera/summary_vlm_all_confidence.json",
    "outputs/hm3d_minival_uniform/qwen_rgbd_90_analysis_raw/summary_vlm_all_confidence.json",
    "outputs/hm3d_minival_uniform/qwen_rgbd_90_analysis_fused/summary_vlm_all_confidence.json",
    "outputs/hm3d_minival_uniform/qwen_camera_to_raw.json",
    "outputs/hm3d_minival_uniform/qwen_raw_to_fused.json",
    "outputs/hm3d_minival_uniform/qwen_camera_to_fused.json",
    "outputs/hm3d_minival_uniform/density_analysis/density_ablation.json",
    "outputs/hm3d_minival_uniform/density_analysis/density_ablation.csv",
    "outputs/hm3d_minival_uniform/peterai_rgbd_wide120_topk_0m/summary_topk.json",
    "outputs/hm3d_minival_uniform/peterai_rgbd_wide120_topk_5m/summary_topk.json",
    "outputs/hm3d_minival_uniform/topk_analysis/summary_topk.json",
    "outputs/hm3d_minival_uniform/topk_diverse_5p0m/summary_topk.json",
)


REPRODUCTION_FILES = {
    "environment.yml": "reproduction/environment.yml",
    "environment-vlm.yml": "reproduction/environment-vlm.yml",
    "configs/minival.yaml": "reproduction/configs/minival.yaml",
    "README.md": "reproduction/BENCHMARK_README.md",
    "tools/run_full_rgbd_benchmark.sh": "reproduction/tools/run_full_rgbd_benchmark.sh",
    "tools/validate_release.py": "reproduction/tools/validate_release.py",
    "tools/package_technical_supplement.py": "reproduction/tools/package_technical_supplement.py",
    "tools/verify_technical_supplement.py": "verify_technical_supplement.py",
    "sstg_bench/uniform_map.py": "reproduction/sstg_bench/uniform_map.py",
    "sstg_bench/rgbd_capture.py": "reproduction/sstg_bench/rgbd_capture.py",
    "sstg_bench/peterai_rgbd_semantics.py": "reproduction/sstg_bench/peterai_rgbd_semantics.py",
    "sstg_bench/mimo_rgbd_semantics.py": "reproduction/sstg_bench/mimo_rgbd_semantics.py",
    "sstg_bench/local_rgbd_semantics.py": "reproduction/sstg_bench/local_rgbd_semantics.py",
    "sstg_bench/rgbd_fusion.py": "reproduction/sstg_bench/rgbd_fusion.py",
    "sstg_bench/camera_node_map.py": "reproduction/sstg_bench/camera_node_map.py",
    "sstg_bench/benchmark.py": "reproduction/sstg_bench/benchmark.py",
    "sstg_bench/topk.py": "reproduction/sstg_bench/topk.py",
    "sstg_bench/arrival_capture.py": "reproduction/sstg_bench/arrival_capture.py",
    "sstg_bench/arrival_vlm.py": "reproduction/sstg_bench/arrival_vlm.py",
    "sstg_bench/arrival_evaluate.py": "reproduction/sstg_bench/arrival_evaluate.py",
}


README = """# SSTG-Nav Technical Supplement Submission Directory

This directory is the submission-ready technical supplement for the anonymous
SSTG-Nav paper. It is deliberately separate from the multimedia submission.

## What to submit

- `SupplementaryMaterial2027.pdf` is the primary Technical Supplement. Use this
  file when the portal accepts only a supplementary PDF.
- `SSTGNav_TechnicalSupplement.zip` contains the same PDF plus compact numeric
  evidence, prompts, selected implementation modules, checksums, and a verifier.
  Upload it only when the portal permits an additional technical archive.
- Real-robot, first-person, and top-down videos are not present here. Submit
  those through the separate multimedia-material field.

Both standalone submission files are checked against the 10 MiB limit. The
main paper remains self-contained; this package only provides optional detail
and machine-readable support.

## Directory layout

- `SupplementaryMaterial2027.pdf`: compiled anonymous supplement.
- `CLAIM_TO_FILE.md`: paper/table claim to evidence mapping.
- `evidence/outputs/`: compact result files under their original repository
  paths, including per-episode rows for all headline full-validation results.
- `reproduction/`: Conda environments, prompts embedded in source modules,
  selected algorithm modules, and ordered benchmark commands.
- `MANIFEST.json` and `SHA256SUMS`: byte-level provenance.
- `verify_technical_supplement.py`: dependency-free integrity and metric check.

## Verify

From this directory, run:

```bash
python verify_technical_supplement.py .
```

The verifier checks every checksum, recomputes the headline SR/SPL and Top-3
metrics from per-episode CSV files, rejects multimedia or secret files, and
checks the standalone PDF and ZIP size limits.
"""


CLAIM_TO_FILE = """# Claim-to-file index

All paths are relative to this `supplementary/` directory.

| Paper evidence | Machine-readable source |
|---|---|
| Protocol boundary and sampling counts | `evidence/outputs/release_audit.json`; `reproduction/sstg_bench/uniform_map.py` |
| Main Table 3, target-view and independent geometry | `evidence/outputs/hm3d_val_oracle_analysis/`; `evidence/outputs/hm3d_val_uniform/oracle_geometry/` |
| Main Table 4, camera/raw/fused full validation | `evidence/outputs/hm3d_val_uniform/gpt54_camera_node_analysis/`; `evidence/outputs/hm3d_val_uniform/gpt54_rgbd_wide120_analysis_raw/`; `evidence/outputs/hm3d_val_uniform/gpt54_rgbd_wide120_analysis_fused/` |
| Main Table 5, backend/FoV representation controls | GPT-5.4, MiMo, and Qwen summaries under `evidence/outputs/hm3d_minival_uniform/` |
| Main Table 6, fusion-aware sequential recovery | `evidence/outputs/hm3d_val_uniform/gpt54_rgbd_wide120_fusion_first_raw_topk_2p0m/` |
| Fused-only and raw Top-3 controls | `evidence/outputs/hm3d_val_uniform/gpt54_rgbd_wide120_fused_topk_2p0m/`; `evidence/outputs/hm3d_val_uniform/gpt54_rgbd_wide120_raw_topk_2p0m/` |
| Arrival-verifier calibration | `evidence/outputs/hm3d_val_uniform/gpt54_arrival_verified_top3_strict/`; `evidence/outputs/hm3d_val_uniform/gpt54_arrival_verifier_top3_strict/` |
| Mapping and arrival prompts | `reproduction/sstg_bench/peterai_rgbd_semantics.py`; `reproduction/sstg_bench/arrival_vlm.py`; `reproduction/sstg_bench/mimo_rgbd_semantics.py`; `reproduction/sstg_bench/local_rgbd_semantics.py` |
| Fusion and reachability rules | `reproduction/sstg_bench/rgbd_fusion.py` |
| Density and corruption studies | `evidence/outputs/hm3d_minival_uniform/density_analysis/`; `evidence/outputs/analysis/stress_aggregate.csv`; `evidence/outputs/analysis/stress_runs.csv` |
| GPT-5.5 semantic-isolation pilot | `evidence/outputs/analysis/`; `evidence/outputs/hm3d_minival_vlm/maps/mapping_report.json`; `evidence/outputs/hm3d_val_vlm/maps/mapping_report.json` |
| Cross-file consistency audit | `evidence/outputs/release_audit.json` |

The PDF includes representative figures. Raw RGB-D, image sequences, MP4 files,
scene datasets, and the real-robot video are intentionally excluded and belong
to the benchmark storage or separate multimedia submission.
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_required(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"required technical-supplement file is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def manifest_entries(root: Path) -> list[dict]:
    excluded = {"MANIFEST.json", "SHA256SUMS", ARCHIVE_NAME}
    entries = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root)
        if str(relative) in excluded:
            continue
        entries.append(
            {
                "path": str(relative),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return entries


def build() -> dict:
    temporary = DESTINATION.with_name(DESTINATION.name + ".building")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    copy_required(PAPER_ROOT / "SupplementaryMaterial2027.pdf", temporary / "SupplementaryMaterial2027.pdf")
    for relative in EVIDENCE_FILES:
        copy_required(ROOT / relative, temporary / "evidence" / relative)
    for source, destination in REPRODUCTION_FILES.items():
        copy_required(ROOT / source, temporary / destination)

    (temporary / "README.md").write_text(README, encoding="utf-8")
    (temporary / "CLAIM_TO_FILE.md").write_text(CLAIM_TO_FILE, encoding="utf-8")

    entries = manifest_entries(temporary)
    manifest = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "anonymous SSTG-Nav technical supplement; multimedia excluded",
        "file_count": len(entries),
        "total_uncompressed_bytes": sum(item["bytes"] for item in entries),
        "excluded_content": [
            "API credentials and .env files",
            "raw RGB-D and scene datasets",
            "image sequences and MP4 videos",
            "real-robot multimedia",
        ],
        "files": entries,
    }
    (temporary / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    checksum_lines = [f'{item["sha256"]}  {item["path"]}' for item in entries]
    (temporary / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")

    if DESTINATION.exists():
        shutil.rmtree(DESTINATION)
    temporary.rename(DESTINATION)

    archive = DESTINATION / ARCHIVE_NAME
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as handle:
        for path in sorted(item for item in DESTINATION.rglob("*") if item.is_file() and item != archive):
            relative = path.relative_to(DESTINATION)
            handle.write(path, Path("sstg_nav_technical_supplement") / relative)

    pdf = DESTINATION / "SupplementaryMaterial2027.pdf"
    if pdf.stat().st_size >= TEN_MIB:
        raise RuntimeError(f"technical supplement PDF exceeds 10 MiB: {pdf.stat().st_size} bytes")
    if archive.stat().st_size >= TEN_MIB:
        raise RuntimeError(f"technical supplement archive exceeds 10 MiB: {archive.stat().st_size} bytes")

    report = {
        "destination": str(DESTINATION.relative_to(ROOT)),
        "files": manifest["file_count"],
        "uncompressed_bytes": manifest["total_uncompressed_bytes"],
        "pdf_bytes": pdf.stat().st_size,
        "archive_bytes": archive.stat().st_size,
        "pdf_sha256": sha256(pdf),
        "archive_sha256": sha256(archive),
        "multimedia_included": False,
        "under_10_mib": True,
    }
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    build()
