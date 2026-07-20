# Model weights

Weights are not included in this archive. The archive also does not currently record a verified OSF URL or DOI, so it must not be described as independently downloadable until that locator is added and tested.

## Locally audited reference artifacts

The following SHA-256 values were computed on 2026-07-11 from the three local checkpoint files used by the manuscript rerun. They are recorded as `reference_sha256` in `config/ensemble.json`.

| ensemble member | audited local source directory name | `model.safetensors` SHA-256 |
|---|---|---|
| `pubmedbert_stratified` | `pubmedbert_stratified_v2_best` | `755711147f1779dfbeb6a7f96dce9a26378e19a15b702e689522b4938287f783` |
| `biolinkbert_stratified` | `biolinkbert_stratified_v2_best` | `30cb90ebb3a96cb8fdd51f5dc106d41f19c69410d06ed70b5768b59ff34afc42` |
| `pubmedbert_allsource` | `pubmedbert_E3b_raw_with_trackb_best` | `b69eb00096a33c6d02f2d25631caf96aafa8155546a4e679d101d23782cc32c6` |

These hashes identify the audited local files; they do not prove that an external deposit exists or contains the same bytes. Before release, upload/locate the artifacts, download them into a clean environment, verify these hashes, and add the real persistent locator here.

## Expected inference layout

Rename or link the obtained checkpoint directories to the ensemble member names:

```text
checkpoints/
  pubmedbert_stratified/model.safetensors
  biolinkbert_stratified/model.safetensors
  pubmedbert_allsource/model.safetensors
```

Verify and score with:

```bash
python src/inference.py \
  --config config/ensemble.json \
  --models checkpoints \
  --verify-reference-hashes \
  --input data/materialized/test.csv \
  --out scores.json
```

The reference ensemble weights are 0.50, 0.25, and 0.25 respectively. The two-model sensitivity variant uses the first two members at 0.50/0.50 and is selected with `--variant sensitivity_two_model_50_50`.

Newly retrained checkpoints are not expected to match these hashes because the documented training recipe is stochastic. The exact base-model repository revisions used for the audited checkpoints have not yet been established and remain a reproducibility blocker.
