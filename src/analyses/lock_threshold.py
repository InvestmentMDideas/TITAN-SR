"""Lock a validation threshold and evaluate it unchanged on a holdout.

The locked threshold is the score of the required validation positive and the
decision comparator is inclusive (``score >= threshold``).  Consequently all
ties at the cutoff are included and validation recall can conservatively
exceed the requested target.
"""

import argparse
import json

import numpy as np


def positive_scores(score_data):
    values = []
    for review in score_data.values():
        labels = np.asarray(review["labels"])
        scores = np.asarray(review["score"], dtype=float)
        if labels.shape != scores.shape:
            raise ValueError("every review must have equally sized labels and scores")
        if not np.isin(labels, (0, 1)).all() or not np.isfinite(scores).all():
            raise ValueError("labels must be binary and scores must be finite")
        values.extend(scores[labels == 1].tolist())
    if not values:
        raise ValueError("validation scores contain no positive records")
    return np.asarray(values, dtype=float)


def threshold_at_recall(scores, target_recall):
    if not 0 < target_recall <= 1:
        raise ValueError("target_recall must be in (0, 1]")
    required = int(np.ceil(target_recall * len(scores)))
    return float(np.sort(scores)[::-1][required - 1])


def holdout_counts(score_data, threshold):
    counts = []
    for review_id, review in score_data.items():
        labels = np.asarray(review["labels"])
        scores = np.asarray(review["score"], dtype=float)
        if labels.shape != scores.shape:
            raise ValueError(f"review {review_id!r} has mismatched labels and scores")
        n_positive = int(labels.sum())
        if n_positive:
            recalled = int((scores[labels == 1] >= threshold).sum())
            counts.append((review_id, n_positive, recalled))
    if not counts:
        raise ValueError("holdout scores contain no positive records")
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_scores", required=True)
    parser.add_argument("--holdout_scores", required=True)
    parser.add_argument("--target_recall", type=float, default=0.99)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.bootstrap <= 0:
        parser.error("--bootstrap must be positive")

    with open(args.val_scores, encoding="utf-8") as handle:
        validation = json.load(handle)
    with open(args.holdout_scores, encoding="utf-8") as handle:
        holdout = json.load(handle)

    rng = np.random.default_rng(args.seed)
    validation_positive = positive_scores(validation)
    threshold = threshold_at_recall(validation_positive, args.target_recall)
    validation_recall = float((validation_positive >= threshold).mean())
    counts = holdout_counts(holdout, threshold)
    positives = np.asarray([row[1] for row in counts], dtype=int)
    recalled = np.asarray([row[2] for row in counts], dtype=int)
    pooled = recalled.sum() / positives.sum()
    per_review = recalled / positives

    bootstrap_recall = np.empty(args.bootstrap, dtype=float)
    for start in range(0, args.bootstrap, 256):
        stop = min(start + 256, args.bootstrap)
        indices = rng.integers(0, len(counts), size=(stop - start, len(counts)))
        bootstrap_recall[start:stop] = (
            recalled[indices].sum(axis=1) / positives[indices].sum(axis=1)
        )
    print(
        f"locked tau={threshold:.6g}; validation recall={validation_recall:.6f} "
        f"(target={args.target_recall:.6f}, comparator=>=, cutoff ties included)"
    )
    print(
        f"holdout pooled recall {pooled:.4f} "
        f"[95% CI {np.percentile(bootstrap_recall, 2.5):.4f}, "
        f"{np.percentile(bootstrap_recall, 97.5):.4f}]"
    )
    print(
        f"per-review recall median {np.median(per_review):.3f}, min {per_review.min():.3f}, "
        f">= {args.target_recall}: {int((per_review >= args.target_recall).sum())}/{len(per_review)}"
    )


if __name__ == "__main__":
    main()
