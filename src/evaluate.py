"""Metrics from a scores.json (review_id -> {labels, score}): AUC, S@95R, S@99R, WSS@95,
per-review medians and review-level cluster-bootstrap 95% CIs."""
import argparse, json
import numpy as np
from sklearn.metrics import roc_auc_score

def spec_at_recall(lab, sc, rt):
    lab = np.asarray(lab); sc = np.asarray(sc); npos = int(lab.sum()); nneg = len(lab) - npos
    if npos == 0 or nneg == 0: return None
    order = np.argsort(-sc); sl = lab[order]; tgt = int(np.ceil(rt * npos)); found = 0
    for i, x in enumerate(sl):
        found += x
        if found >= tgt: return (nneg - ((i + 1) - found)) / nneg
    return 0.0

def wss(lab, sc, rt=0.95):
    lab = np.asarray(lab); sc = np.asarray(sc); npos = int(lab.sum())
    if npos == 0: return None
    sl = lab[np.argsort(-sc)]; tgt = int(np.ceil(rt * npos)); found = 0
    for i, x in enumerate(sl):
        found += x
        if found >= tgt: return 1.0 - (i + 1) / len(lab)
    return 0.0

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scores", required=True)
    ap.add_argument("--out", default=None); ap.add_argument("--bootstrap", type=int, default=10000)
    a = ap.parse_args()
    data = json.load(open(a.scores)); rng = np.random.default_rng(42)
    rows = []
    for k, d in data.items():
        lab, sc = d["labels"], d["score"]
        if sum(lab) < 2 or sum(lab) == len(lab): continue
        try: auc = roc_auc_score(lab, sc)
        except Exception: auc = float("nan")
        rows.append(dict(review=k, auc=auc, s95r=spec_at_recall(lab, sc, 0.95),
                         s99r=spec_at_recall(lab, sc, 0.99), wss95=wss(lab, sc)))
    def boot_median(key):
        v = np.array([r[key] for r in rows if r[key] == r[key]], float); n = len(v)
        bs = [np.median(v[rng.integers(0, n, n)]) for _ in range(a.bootstrap)]
        return float(np.median(v)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    summary = {"n_reviews": len(rows)}
    for k in ("auc", "s95r", "s99r", "wss95"):
        m, lo, hi = boot_median(k); summary[k] = {"median": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)]}
    print(json.dumps(summary, indent=2))
    if a.out: json.dump({"summary": summary, "per_review": rows}, open(a.out, "w"), indent=2)

if __name__ == "__main__":
    main()
