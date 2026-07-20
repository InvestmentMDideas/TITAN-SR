"""Citation-level overlap (potential leakage) + within-review positive control.
Inputs: scores.json (review_id->{labels,score,titles}) and a newline/JSON list of normalised training titles."""
import argparse, json, re
import numpy as np
def norm(t):
    t = re.sub(r"[^a-z0-9]+", " ", (t or "").lower()).strip(); return t if len(t) >= 10 else None
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scores", required=True)
    ap.add_argument("--train_titles", required=True, help="JSON list of training titles"); a = ap.parse_args()
    pc = json.load(open(a.scores)); train = set(norm(x) for x in json.load(open(a.train_titles)))
    seen_pct, unseen_pct, per_rev = [], [], []; n_ov = n_tot = n_ov_pos = n_pos = 0
    for d in pc.values():
        lab = np.asarray(d["labels"]); sc = np.asarray(d["score"]); titles = d.get("titles", [""]*len(lab))
        ov = np.array([norm(t) in train for t in titles]); n = len(lab)
        n_tot += n; n_ov += int(ov.sum()); n_pos += int(lab.sum()); n_ov_pos += int(((lab == 1) & ov).sum())
        order = np.argsort(-sc); pct = np.empty(n); pct[order] = 1.0 - np.arange(n) / max(n - 1, 1)
        inc = lab == 1; s = inc & ov; u = inc & ~ov
        if s.sum() and u.sum(): per_rev.append(np.median(pct[s]) - np.median(pct[u]))
        seen_pct += pct[s].tolist(); unseen_pct += pct[u].tolist()
    print(f"overlap records {n_ov}/{n_tot} ({100*n_ov/n_tot:.1f}%); overlap-includes {n_ov_pos}/{n_pos} ({100*n_ov_pos/max(n_pos,1):.1f}%)")
    print(f"positive control: seen include median pctile {np.median(seen_pct):.3f} vs unseen {np.median(unseen_pct):.3f}; "
          f"review-level median diff {np.median(per_rev):+.3f}  (>0 and significant would indicate memorization)")
if __name__ == "__main__":
    main()
