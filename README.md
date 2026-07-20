# TITAN-SR — reproducibility package

Code and configuration to reproduce **TITAN-SR**, a biomedical BERT ensemble for systematic-review
**title and abstract** screening (manuscript JCEPI-D-26-00715, *Journal of Clinical Epidemiology*).

> **This repository deliberately does *not* include the trained model weights.** It provides everything
> needed to (a) understand and re-run the method, (b) re-train from the released data manifests, and
> (c) reproduce the paper's evaluation and sensitivity analyses. The fine-tuned weights are available
> from the corresponding author on reasonable request for research use, and will be deposited (with SHA-256
> integrity hashes) on OSF at publication. See `WEIGHTS.md`.

## What TITAN-SR does
For each citation, the model reads the review's **eligibility criteria** together with the citation's
**title + abstract** (`[CLS] criteria [SEP] title. abstract [SEP]`, truncated to 512 tokens) and outputs a
single inclusion-relevance score. Scores rank citations for screening; a stopping-rule engine flags
high-priority, borderline, and stopping-eligible records. The reported results use a **weighted
three-checkpoint ensemble** (`config/ensemble.json`).

## Repository layout
```
src/
  train.py            fine-tune one biomedical BERT checkpoint (the TITAN-SR recipe)
  inference.py        score citations with the weighted-ensemble (loads local checkpoints)
  evaluate.py         metrics: AUC, specificity@95/99% recall, WSS@95, review-level bootstrap CIs
  fetch_abstracts.py  rebuild training/eval CSVs from the manifests (PubMed E-utilities)
  llm_benchmark.py    stateless commercial-chatbot screening benchmark (API keys via env vars)
  analyses/
    leakage.py        citation-level overlap + within-review positive control
    lock_threshold.py lock a deployable threshold on validation, evaluate recall on a holdout
config/
  ensemble.json       3-checkpoint ensemble (members, weights, input format); + 2-model sensitivity
  thresholds.json     example deployable thresholds (validation-derived; ensemble/data-specific)
data/
  README.md           data-manifest schema + how to obtain the manifests
  example_criteria.json
  example_dataset.csv (tiny synthetic sample showing the column schema)
requirements.txt  environment.yml  LICENSE  DATA_LICENSE.md  WEIGHTS.md  CITATION.cff
```

## Quick start
```bash
pip install -r requirements.txt           # or: conda env create -f environment.yml

# 0) Rebuild a training CSV from the manifests (download them first — see "Data"; fetches abstracts from PubMed)
python src/fetch_abstracts.py --labels data/manifests/pmc_train_labels.csv.gz --criteria data/manifests/criteria.csv.gz --out data/pmc_train.csv

# 1) Re-train a checkpoint from a manifest CSV (criteria,title,abstract,label,review_id)
python src/train.py --base microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
                    --train data/train.csv --val data/val.csv --out checkpoints/pubmedbert

# 2) Score an evaluation set with the weighted ensemble
python src/inference.py --config config/ensemble.json --models checkpoints \
                        --input data/test.csv --out scores.json

# 3) Compute metrics (per-review medians + bootstrap CIs)
python src/evaluate.py --scores scores.json --out metrics.json
```

## Data
Training/evaluation **manifests are distributed separately** from this repository (PMID identifiers +
screening labels, per-review eligibility criteria, and the review-level split): they will be deposited on OSF
at publication and are available from the corresponding author in the interim — see `data/README.md`. Place
them under `data/manifests/` and `data/splits/`. Abstracts are reconstructed from the identifiers with
`src/fetch_abstracts.py` (PubMed). No model weights and no API keys are included. The Chan gold-standard and
Chan-2025 external sets are obtained from their original public releases (refs 13, 14).

## Reproducing the paper's analyses
- External validation (Tables 2–3): `src/inference.py` + `src/evaluate.py` with `config/ensemble.json`.
- Ensemble-choice sensitivity (3-checkpoint vs 2-model 50/50): add `--block sensitivity_two_model_50_50`
  to the `src/inference.py` call (selects the alternative member block in `config/ensemble.json`).
- Encoder ablation (R2-c5/c21): re-run `src/train.py` with `--base dmis-lab/biobert-base-cased-v1.1` (etc.).
- Citation-level leakage + positive control (R3-maj1): `src/analyses/leakage.py`. Run inference with
  `--emit_titles`; `--train_titles` takes a JSON list of titles extracted from the rebuilt training CSV.
- Registered primary outcome (recall at a validation-locked threshold): `src/analyses/lock_threshold.py`.

## Determinism
Inference is deterministic (eval mode, no shuffling). Training is **not** fully deterministic (class-balanced
sampler + mixed precision); released weights carry SHA-256 hashes so a deposited checkpoint can be verified.

## License
Code: MIT (`LICENSE`). Data manifests: CC-BY 4.0 (`DATA_LICENSE.md`).

## Citation
See `CITATION.cff`.
