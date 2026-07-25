# SSTG-Nav Technical Supplement Submission Directory

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
