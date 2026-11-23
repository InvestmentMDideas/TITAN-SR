# Third-party notices

## Eligibility-criteria excerpts

`data/manifests/criteria.csv.gz` contains excerpts from 6,826 openly licensed systematic reviews in PubMed Central. Each row carries the excerpt's licence, licence URL, authors, copyright statement, citation, DOI and a link to the source article. Those rows are the attribution for the excerpts. The excerpts remain under their original licences (see `DATA_LICENSE.md`).

## Chan et al. screening dataset

Labels and titles for the 22-review external holdout, and the `chan_` split assignments, are derived from:

> Chan G, He E, Leung J, Verspoor K. Title and Abstract Screening Data for Systematic Review. Mendeley Data, V1, 2024. doi:10.17632/7sgmg89zb6.1. Licensed CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).

Associated article: doi:10.1017/rsm.2025.1.

## Base models (for any deposited weights)

The fine-tuned checkpoints described in `WEIGHTS.md` derive from:

- `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` (formerly `...-PubMedBERT-...`): MIT License, Copyright (c) Microsoft Corporation.
- `michiyasunaga/BioLinkBERT-base`: Apache License 2.0.

Any deposit of the weights must include these licences alongside the checkpoints.
