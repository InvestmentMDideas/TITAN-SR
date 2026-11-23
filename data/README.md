# Data

The archive contains derived labels for PMC and Track B, eligibility criteria for the PMC reviews whose articles are openly licensed, a hash manifest for the criteria it does not redistribute, a title-only external holdout label manifest, a review-level split, and a verified source/checksum record for Chan et al. It does not bundle raw abstracts, the Chan archives, or a normalized Chan citation/criteria table matching this package's training schema.

## Included files

| path | contents |
|---|---|
| `manifests/pmc_train_labels.csv.gz` | PMC labels: `review_id, candidate_pmid, label, source` |
| `manifests/trackb_train_labels.csv.gz` | Track B labels with the same schema |
| `manifests/criteria.csv.gz` | eligibility criteria for 6,826 PMC reviews whose article is CC BY or CC0, with attribution: `review_id, source, eligibility_criteria, license, license_url, copyright, authors, citation, doi, pmid, source_url` |
| `manifests/criteria_withheld.csv.gz` | 4,444 criteria passages that are **not** redistributed: `review_id, source, reason, license, doi, pmid, text_sha256, text_chars` |
| `manifests/external_holdout_labels.csv.gz` | title-only 22-review temporal holdout labels from Chan et al. (CC BY 4.0); no PMID or abstract |
| `splits/stratified_split_v2.json` | source-prefixed review-level assignments |
| `sources/chan_mendeley_v1.json` | authoritative Chan V1 DOI, download URLs, CC BY 4.0 terms, and archive/member SHA-256 checksums for both Chan archives |
| `example_dataset.csv`, `example_criteria.json` | tiny synthetic schema examples |

## Criteria: released and withheld

Each released passage is an excerpt from the review article named in its row and stays under that article's licence (`license`, `license_url`); see `DATA_LICENSE.md`. The licence of every PMC article was taken from PMC's own article metadata (`license_code`) and agrees with the licence in the article XML.

`criteria_withheld.csv.gz` lists the passages used in the manuscript that this archive does not redistribute, by `reason`:

| reason | rows | what it means |
|---|---|---|
| `publisher_full_text` | 3,525 | Track B: text extracted from publisher PDFs |
| `license_not_open` | 891 | PMC article under CC BY-NC or CC BY-NC-ND |
| `license_unconfirmed` | 6 | PMC metadata records no licence for the article |
| `rebuild_from_chan_archive` | 22 | Cochrane selection criteria for the external holdout; rebuild with `src/build_external_criteria.py` |

`text_sha256` is the SHA-256 of the UTF-8 passage exactly as used. If you extract a withheld passage from your own copy of the article and pass it to `fetch_abstracts.py` as an extra `--criteria` file, the audit reports whether it matches.

## Important limitations

- The split contains 8,539 `chan_` assignments whose normalized citation and criteria rows are not included. `sources/chan_mendeley_v1.json` records the authoritative Mendeley V1 source and hashes, but a documented source-to-schema materializer is still required before those assignments can be recreated.
- Track B and the withheld PMC reviews contribute rows only when you supply their criteria. Without them, `fetch_abstracts.py` excludes their rows (`--missing-criteria exclude`) and counts them in the audit.
- The originally supplied split header disagreed with its assignment contents. Its review/item counts have been recomputed from the assignments, and `src/materialize_splits.py` independently checks them on each run.
- Manifest review IDs are unprefixed, while split IDs use `pmc_`, `tb_`, and `chan_`. The materializer resolves this mapping explicitly.
- The external holdout supplies titles but not PMIDs/abstracts, so `src/fetch_abstracts.py` cannot reconstruct it. A separately licensed citation source or title-to-PMID linkage is required.
- Abstract availability is outcome-sensitive in the external data. Report excluded rows and positives from fetch/inference audits rather than describing the full title manifest as scored.

## Rebuild the external-holdout criteria

Download `20240827_Systematic_review_manual_search_data.zip` from Chan et al. (Mendeley Data, doi:10.17632/7sgmg89zb6.1; URL in `sources/chan_mendeley_v1.json`), then:

```bash
python src/build_external_criteria.py \
  --archive 20240827_Systematic_review_manual_search_data.zip \
  --out data/reconstructed/external_criteria.csv
```

The script verifies the archive checksum and every rebuilt passage against `criteria_withheld.csv.gz`.

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

Successful results, including confirmed no-abstract records, are committed transactionally to SQLite after every batch. Failed network batches cause a nonzero exit and no output CSV unless `--allow-partial` is explicitly supplied. The audit distinguishes invalid PMIDs, PubMed non-returns, genuine no-abstract records, rows without criteria, and unavailable rows after request failure.

## Materialize clean splits

```bash
python src/materialize_splits.py \
  --input data/reconstructed/pmc.csv \
  --splits data/splits/stratified_split_v2.json \
  --out-dir data/materialized
```

Add `--input data/reconstructed/trackb.csv` when you have built Track B rows from your own criteria.

This writes `train.csv`, `val.csv`, `test.csv`, and `materialization.audit.json`. The default rules are:

1. map manifest IDs to source-prefixed split IDs;
2. collapse repeated same-label review/PMID rows;
3. exclude every row for a review/PMID key with conflicting labels rather than invent an adjudication;
4. when a PMID appears across splits, keep all occurrences only in the highest-priority split (`test`, then `val`, then `train`); and
5. add deterministic `row_id`, `split_assignment_id`, and `split` fields.

These defaults prevent cross-split citation overlap but do not supply the missing Chan records. When Chan data are acquired, normalize them to the same CSV schema (including `source=chan` and a stable citation identifier), add another `--input`, and review the generated audit before training.
