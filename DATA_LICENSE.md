# Data license

## Project-authored data: CC BY 4.0

The project authors license the data they created under the **Creative Commons Attribution 4.0 International** licence (CC BY 4.0, https://creativecommons.org/licenses/by/4.0/): the screening labels in `pmc_train_labels.csv.gz` and `trackb_train_labels.csv.gz`, the split assignments in `splits/stratified_split_v2.json`, `criteria_withheld.csv.gz` (identifiers and hashes only), and the source/checksum records. Identifiers (PMIDs, PMCIDs, DOIs) are facts and carry no licence.

## Third-party text: under its own licence, not the project's

- **`manifests/criteria.csv.gz`.** Each `eligibility_criteria` passage is an excerpt from the review article identified in the same row. It is reproduced under that article's licence, given in the `license` and `license_url` columns (Creative Commons Attribution 2.0, 3.0, 3.0 IGO or 4.0, or CC0 1.0). Attribution for each passage is in its row: `authors`, `copyright`, `citation`, `doi`, `source_url`. Changes: the passage was excerpted from the article's methods, whitespace may have been normalised, and passages longer than 3,000 characters were truncated. The project's CC BY grant does not cover these passages, and their authors and publishers do not endorse this project.
- **`manifests/external_holdout_labels.csv.gz`** and the `chan_` entries in `splits/stratified_split_v2.json` are derived from: Chan G, He E, Leung J, Verspoor K. *Title and Abstract Screening Data for Systematic Review*. Mendeley Data, V1, 2024. doi:10.17632/7sgmg89zb6.1, licensed CC BY 4.0. Changes: columns were selected and reformatted; split assignments were added by the project. Chan et al. do not endorse this project.

## Not redistributed

Text whose licence does not permit open redistribution is not in this archive: Track B criteria (extracted from publisher PDFs), criteria from PMC articles under non-commercial licences or without a recorded licence, the Cochrane selection criteria for the external holdout, PubMed titles and abstracts, and the Chan archives. `criteria_withheld.csv.gz` identifies the withheld passages by hash only. Users who obtain any of this material are responsible for complying with its terms.

## Code

The source code is licensed MIT (see `LICENSE`). Third-party attributions are collected in `NOTICE.md`.
