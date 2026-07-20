"""Score citations with the TITAN-SR weighted ensemble. Loads local checkpoints (weights NOT shipped)."""
import argparse, csv, json, glob
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from safetensors import safe_open
csv.field_size_limit(10**9)

def num_labels(path):
    sf = glob.glob(f"{path}/*.safetensors")
    if not sf: return 1
    with safe_open(sf[0], framework="pt") as f:
        for k in f.keys():
            if k.endswith("classifier.weight"): return f.get_slice(k).get_shape()[0]
    return 1

@torch.no_grad()
def score(tok, model, criteria, texts, maxlen, dev, bs=64):
    out = []
    for i in range(0, len(texts), bs):
        chunk = texts[i:i+bs]
        enc = tok([criteria]*len(chunk), chunk, truncation=True, max_length=maxlen,
                  padding="max_length", return_tensors="pt").to(dev)
        lo = model(**enc).logits; lo = lo[:, 1] if lo.shape[-1] > 1 else lo.squeeze(-1)
        out += torch.sigmoid(lo).float().cpu().tolist()
    return np.array(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--models", required=True,
        help="directory holding one sub-folder per ensemble member (named by member 'name')")
    ap.add_argument("--input", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--cols", default="review_id,criteria,title,abstract,label")
    ap.add_argument("--block", default=None, help="alternative member block inside the config "
                    "(e.g. 'sensitivity_two_model_50_50' in config/ensemble.json)")
    ap.add_argument("--emit_titles", action="store_true",
                    help="include raw titles in the output (needed by analyses/leakage.py)")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    if a.block: cfg = {**cfg, **cfg[a.block]}
    maxlen = cfg.get("max_length", 512)
    w = np.array([m["weight"] for m in cfg["members"]]); w = w / w.sum()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    members = []
    for m in cfg["members"]:
        p = f'{a.models}/{m["name"]}'
        tok = AutoTokenizer.from_pretrained(p)
        mdl = AutoModelForSequenceClassification.from_pretrained(p, num_labels=num_labels(p)).to(dev).eval()
        members.append((tok, mdl))
    rid, crit, title, abs, lab = a.cols.split(",")
    reviews = {}
    with open(a.input, encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            d = reviews.setdefault(r[rid], {"criteria": r.get(crit, "") or "", "texts": [], "labels": [], "titles": []})
            d["texts"].append(f'{r.get(title,"") or ""}. {r.get(abs,"") or ""}'.strip())
            d["titles"].append(r.get(title, "") or "")
            d["labels"].append(int(float(r.get(lab, 0) or 0)))
    res = {}
    for k, d in reviews.items():
        preds = [score(t, m, d["criteria"], d["texts"], maxlen, dev) for t, m in members]
        res[k] = {"labels": d["labels"], "score": np.average(np.stack(preds), axis=0, weights=w).tolist()}
        if a.emit_titles: res[k]["titles"] = d["titles"]
    json.dump(res, open(a.out, "w"))
    print(f"scored {len(res)} reviews -> {a.out}")

if __name__ == "__main__":
    main()
