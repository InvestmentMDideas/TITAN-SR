# Model weights (not included)

The fine-tuned TITAN-SR checkpoints are **intentionally excluded** from this repository and from the public package.

- They are available from the corresponding author **on reasonable request** for non-commercial research use.
- They will also be deposited on OSF with **SHA-256 integrity hashes** so a received checkpoint can be verified.
- To reproduce from scratch, run `src/train.py` on the released manifests (see `data/README.md`).

## Ensemble (config/ensemble.json)
| member | base architecture | weight |
|---|---|---|
| pubmedbert_stratified | PubMedBERT (BiomedNLP) | 0.50 |
| biolinkbert_stratified | BioLinkBERT-base | 0.25 |
| pubmedbert_allsource | PubMedBERT (trained on all three sources) | 0.25 |

## SHA-256 checksums (fill on deposit)
```
pubmedbert_stratified/model.safetensors   <sha256>
biolinkbert_stratified/model.safetensors  <sha256>
pubmedbert_allsource/model.safetensors    <sha256>
```
