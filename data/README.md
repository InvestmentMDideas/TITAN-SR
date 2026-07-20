# Data

The archive contains derived labels and criteria for PMC and Track B, a title-only external holdout label manifest, a review-level split, and a verified source/checksum manifest for Chan et al. It does not bundle raw abstracts, the 466 MB Chan archive, or a normalized Chan citation/criteria table matching this package's training schema.

## Included files

| path | contents |
|---|---|
| `manifests/pmc_train_labels.csv.gz` | PMC labels: `review_id, candidate_pmid, label, source` |
| `manifests/trackb_train_labels.csv.gz` | Track B labels with the same schema |
| `manifests/criteria.csv.gz` | PMC/Track B eligibility criteria |
| `manifests/external_holdout_labels.csv.gz` | title-only 22-review temporal holdout labels; no PMID or abstract |
| `manifests/external_criteria.csv.gz` | external-review eligibility criteria |
| `splits/stratified_split_v2.json` | source-prefixed review-level assignments |
| `sources/chan_mendeley_v1.json` | authoritative Chan V1 DOI, download URL, CC BY 4.0 terms, and archive/member SHA-256 checksums |
| `example_dataset.csv`, `example_criteria.json` | tiny synthetic schema examples |

## Important limitations

- The split contains 8,539 `chan_` assignments whose normalized citation and criteria rows are not included. `sources/chan_mendeley_v1.json` records the authoritative Mendeley V1 source and hashes, but a documented source-to-schema materializer is still required before those assignments can be recreated.
- The originally supplied split header disagreed with its assignment contents. Its review/item counts have been recomputed from the assignments, and `src/materialize_splits.py` independently checks them on each run.
- Manifest review IDs are unprefixed, while split IDs use `pmc_`, `tb_`, and `chan_`. The materializer resolves this mapping explicitly.
- The external holdout supplies titles but not PMIDs/abstracts, so `src/fetch_abstracts.py` cannot reconstruct it. A separately licensed citation source or title-to-PMID linkage is required.
- Abstract availability is outcome-sensitive in the external data. Report excluded rows and positives from fetch/inference audits rather than describing the full title manifest as scored.

## Reconstruct PubMed text

```bash
export NCBI_EMAIL='your-email@example.org'
python src/fetch_abstracts.py \
  --labels data/manifests/pmc_train_labels.csv.gz \
  --criteria data/manifests/criteria.csv.gz \
  --out data/reconstructed/pmc.csv \
  --cache data/cache/pubmed.sqlite3 \
  --audit data/reconstructed/pmc.audit.json
```

The output columns are:

```text
review_id,candidate_pmid,criteria,title,abstract,label,source
```

Successful results, including confirmed no-abstract records, are committed transactionally to SQLite after every batch. Failed network batches cause a nonzero exit and no output CSV unless `--allow-partial` is explicitly supplied. The audit distinguishes invalid PMIDs, PubMed non-returns, genuine no-abstract records, and unavailable rows after request failure.

## Materialize clean splits

```bash
python src/materialize_splits.py \
  --input data/reconstructed/pmc.csv \
  --input data/reconstructed/trackb.csv \
  --splits data/splits/stratified_split_v2.json \
  --out-dir data/materialized
```

This writes `train.csv`, `val.csv`, `test.csv`, and `materialization.audit.json`. The default rules are:

1. map manifest IDs to source-prefixed split IDs;
2. collapse repeated same-label review/PMID rows;
3. exclude every row for a review/PMID key with conflicting labels rather than invent an adjudication;
4. when a PMID appears across splits, keep all occurrences only in the highest-priority split (`test`, then `val`, then `train`); and
5. add deterministic `row_id`, `split_assignment_id`, and `split` fields.

These defaults prevent cross-split citation overlap but do not supply the missing Chan records. When Chan data are acquired, normalize them to the same CSV schema (including `source=chan` and a stable citation identifier), add another `--input`, and review the generated audit before training.
