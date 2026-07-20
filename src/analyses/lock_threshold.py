"""Registered primary outcome: lock a single deployable threshold on validation (to hit a target recall),
then report recall on a holdout with a review-level cluster-bootstrap CI. Inputs are scores.json files."""
import argparse, json
import numpy as np
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--val_scores", required=True)
    ap.add_argument("--holdout_scores", required=True); ap.add_argument("--target_recall", type=float, default=0.99)
    ap.add_argument("--bootstrap", type=int, default=10000); a = ap.parse_args()
    val = json.load(open(a.val_scores)); hold = json.load(open(a.holdout_scores)); rng = np.random.default_rng(42)
    val_inc = np.concatenate([np.asarray(d["score"])[np.asarray(d["labels"]) == 1] for d in val.values()])
    tau = float(np.percentile(val_inc, (1 - a.target_recall) * 100))
    rp = np.array([(int(np.asarray(d["labels"]).sum()),
                    int((np.asarray(d["score"])[np.asarray(d["labels"]) == 1] >= tau).sum()))
                   for d in hold.values() if np.asarray(d["labels"]).sum() > 0])
    pooled = rp[:, 1].sum() / rp[:, 0].sum(); per_rev = rp[:, 1] / rp[:, 0]
    bs = [(rp[idx := rng.integers(0, len(rp), len(rp)), 1].sum() / rp[idx, 0].sum()) for _ in range(a.bootstrap)]
    print(f"locked tau={tau:.4f} (val recall {a.target_recall})")
    print(f"holdout pooled recall {pooled:.4f} [95% CI {np.percentile(bs,2.5):.4f}, {np.percentile(bs,97.5):.4f}]")
    print(f"per-review recall median {np.median(per_rev):.3f}, min {per_rev.min():.3f}, "
          f">= {a.target_recall}: {int((per_rev>=a.target_recall).sum())}/{len(per_rev)}")
if __name__ == "__main__":
    main()
