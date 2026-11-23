# TITAN-SR — reproducibility package

Code and configuration for TITAN-SR, a biomedical BERT ensemble for systematic-review title and abstract screening (manuscript JCEPI-D-26-00715, *Journal of Clinical Epidemiology*).

## Reproducibility status

This archive contains the PMC and Track B label manifests, eligibility criteria for the PMC reviews whose articles are openly licensed (CC BY or CC0), a review-level split, metric and analysis code, and reference hashes for the three locally audited checkpoints. It does **not** contain:

- eligibility-criteria text for the 3,525 Track B reviews or for the 897 PMC reviews whose article licence does not permit open redistribution. These are listed in `data/manifests/criteria_withheld.csv.gz` with a SHA-256 of the passage used, so criteria you extract from your own copies can be checked against the originals;
- eligibility-criteria text for the 22 external-holdout reviews, which `src/build_external_criteria.py` rebuilds from the Chan et al. archive;
- model weights or a confirmed public weight-deposit URL;
- a normalized Chan citation/criteria manifest matching the all-source training schema (the verified source locator and checksums are included);
- PubMed abstracts, which must be reconstructed from PMIDs;
- abstracts/PMIDs for the title-only temporal-holdout manifest; or
- a complete environment lock or container.

The two-model workflow below is runnable on the PMC reviews with released criteria after fetching PubMed records; Track B rows are used only for reviews whose criteria you supply yourself. Exact reproduction of the manuscript's three-checkpoint experiment remains blocked until the documented Chan source is normalized into the released training schema, the reference weights are deposited, and the base-model revisions are documented. See [data/README.md](data/README.md) and [WEIGHTS.md](WEIGHTS.md).

## Method

For each citation, the model reads the review's eligibility criteria together with the citation's title and abstract (`[CLS] criteria [SEP] title. abstract [SEP]`, truncated to 512 tokens) and emits an inclusion-relevance score. The primary manuscript configuration is a weighted three-checkpoint ensemble. A selectable two-model 50/50 sensitivity configuration is included for the reproducible two-model scaffold.

## Repository layout

```text
src/
  fetch_abstracts.py    resumable PubMed reconstruction with cache + audit
  build_external_criteria.py  rebuild external-holdout criteria from the Chan archive
  materialize_splits.py build leakage-clean train/val/test CSVs
  train.py              fine-tune one biomedical BERT checkpoint
  inference.py          score citations with a selected local ensemble
  evaluate.py           review metrics and review-level bootstrap CIs
  llm_benchmark.py      stateless commercial-chatbot benchmark scaffold
  analyses/
    leakage.py          title overlap + within-review positive control
    lock_threshold.py   lock a validation threshold, evaluate a holdout
config/
  ensemble.json         primary ensemble + selectable two-model variant
  thresholds.json       example validation-derived thresholds
data/
  manifests/            released labels, openly licensed criteria, withheld-criteria hashes
  splits/               released review-level split assignment
tests/                   metric and data-workflow regression tests
```

## Quick start: two-model scaffold

Python 3.12 is the documented environment. NCBI asks API users to identify themselves; set an email and, optionally, an API key before a large reconstruction.

```bash
conda env create -f environment.yml
conda activate titan-sr
export NCBI_EMAIL='your-email@example.org'
# export NCBI_API_KEY='...'  # optional; raises the permitted request rate

python src/fetch_abstracts.py \
  --labels data/manifests/pmc_train_labels.csv.gz \
  --criteria data/manifests/criteria.csv.gz \
  --out data/reconstructed/pmc.csv \
  --cache data/cache/pubmed.sqlite3 \
  --audit data/reconstructed/pmc.audit.json

python src/materialize_splits.py \
  --input data/reconstructed/pmc.csv \
  --splits data/splits/stratified_split_v2.json \
  --out-dir data/materialized

python src/train.py \
  --base microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext \
  --train data/materialized/train.csv \
  --val data/materialized/val.csv \
  --out checkpoints/pubmedbert_stratified

python src/train.py \
  --base michiyasunaga/BioLinkBERT-base \
  --train data/materialized/train.csv \
  --val data/materialized/val.csv \
  --out checkpoints/biolinkbert_stratified

python src/inference.py \
  --config config/ensemble.json \
  --variant sensitivity_two_model_50_50 \
  --models checkpoints \
  --input data/materialized/test.csv \
  --out scores.json

python src/evaluate.py --scores scores.json --out metrics.json
```

To add Track B, or the withheld PMC reviews, extract their eligibility criteria from your own copies of the articles into a CSV with `review_id,eligibility_criteria` (review IDs as in `criteria_withheld.csv.gz`), pass it as a second `--criteria` file, and add the resulting CSV as another `--input` to `materialize_splits.py`:

```bash
python src/fetch_abstracts.py \
  --labels data/manifests/trackb_train_labels.csv.gz \
  --criteria data/manifests/criteria.csv.gz \
  --criteria my_trackb_criteria.csv \
  --out data/reconstructed/trackb.csv \
  --cache data/cache/pubmed.sqlite3 \
  --audit data/reconstructed/trackb.audit.json
```

The audit reports how many supplied passages match the recorded hashes. Rows whose review has no criteria are excluded by default (`--missing-criteria exclude`), never scored against an empty criteria string.

`fetch_abstracts.py` resumes from its transactional SQLite cache. It writes an audit for exclusions and fails without producing a partial CSV when a network batch fails; use `--allow-partial` only for an explicitly incomplete exploratory run. Records successfully returned by PubMed but lacking abstracts are audited and excluded by default.

`materialize_splits.py` resolves the source prefixes omitted from the label manifests, removes duplicate review/citation rows, excludes conflicting within-review labels by default, and prevents one PMID from crossing split boundaries. Test is retained over validation, and validation over training. Its audit verifies the split metadata against the assignments; the incorrect original header counts in the included split have been recomputed from those assignments.

The commands above retrain nondeterministic checkpoints on the released data; they do not recreate the manuscript weights, its withheld Track B criteria, or its missing Chan component.

## Reference-weight inference

If the three reference checkpoints have been obtained, place them under the names below:

```text
checkpoints/
  pubmedbert_stratified/
  biolinkbert_stratified/
  pubmedbert_allsource/
```

Then use the primary configuration. `--verify-reference-hashes` checks the single `model.safetensors` in each directory against `config/ensemble.json` before loading it.

```bash
python src/inference.py \
  --config config/ensemble.json \
  --variant primary \
  --models checkpoints \
  --verify-reference-hashes \
  --input data/materialized/test.csv \
  --out scores.json
```

Do not use reference-hash verification with newly retrained weights; stochastic retraining will produce different hashes.

## Metric semantics

Fixed-recall metrics include every record tied at the cutoff score, making results conservative and independent of row order. The evaluator reports both:

- `work_saved_at_95_recall`: the raw fraction left unscreened; and
- `wss_at_95_recall`: Cohen's work saved over random sampling, `work_saved - 0.05`.

Inference output includes titles and can be passed to `src/analyses/leakage.py`. Equal prediction scores receive equal average percentiles in that analysis.

Historical multi-variant score JSONs can be replayed explicitly with
`--score-key 2model` or `--score-key 3model`; newly generated inference output
uses the default `score` field.

## Threshold locking

```bash
python src/analyses/lock_threshold.py \
  --val_scores validation_scores.json \
  --holdout_scores holdout_scores.json \
  --target_recall 0.99
```

The threshold is selected from validation positives and uses an inclusive `>=` comparator. All cutoff ties are retained, and the command prints the achieved validation recall rather than assuming it equals the requested target.

## Tests

```bash
python -m unittest discover -s tests -v
```

## License and citation

Code is MIT licensed. Data terms are in `DATA_LICENSE.md` and third-party attributions in `NOTICE.md`: each released criteria passage stays under its source article's licence, with attribution in its row. See `CITATION.cff` for citation metadata.
