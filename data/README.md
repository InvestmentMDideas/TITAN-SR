# Data

The data manifests are **not stored in the GitHub repository**. They will be deposited on **OSF** at
publication and are available from the corresponding author in the interim. Two further deliberate
exceptions: the fine-tuned **model weights** (see `WEIGHTS.md`) and the raw **abstracts**, which are
**reconstructed from identifiers** with `src/fetch_abstracts.py` to respect copyright.

## Manifests (place under `data/` after download)
| path | contents |
|---|---|
| `manifests/pmc_train_labels.csv.gz` | PMC / Track A training labels — `review_id, candidate_pmid, label, source` (346,847 rows; 139,109 includes) |
| `manifests/trackb_train_labels.csv.gz` | Track B training labels (67,141 rows; 17,931 includes) |
| `manifests/criteria.csv.gz` | per-review eligibility criteria — `review_id, source, eligibility_criteria` (11,248 reviews) |
| `manifests/external_holdout_labels.csv.gz` | 22-review temporal holdout — `review_id, title, label, publication_year` (142,504 rows; 768 includes) |
| `manifests/external_criteria.csv.gz` | external-review eligibility criteria |
| `splits/stratified_split_v2.json` | review-level train/val/test assignment (seed 42; 19,787 reviews, 762,934 items) |
| `example_dataset.csv`, `example_criteria.json` | tiny synthetic illustration of the column schema |

## Third-party data (cite the original release; not redistributed here)
- The **Chan gold-standard** training subset and the **Chan-2025 temporal-holdout** external set are from
  Chan et al. (manuscript refs 13 and 14; publicly released datasets). This repository includes our derived
  screening **labels** and review identifiers; obtain the citation records from the original Chan releases.
- **Abstracts** are not stored. Reconstruct them from the PMIDs with `src/fetch_abstracts.py` (PubMed E-utilities;
  set `NCBI_API_KEY` for a higher rate limit).

## Rebuild a training / evaluation CSV
```bash
python src/fetch_abstracts.py \
    --labels   data/manifests/pmc_train_labels.csv.gz \
    --criteria data/manifests/criteria.csv.gz \
    --out      data/pmc_train.csv
# -> columns: review_id, criteria, title, abstract, label   (ready for src/train.py and src/inference.py)
```
