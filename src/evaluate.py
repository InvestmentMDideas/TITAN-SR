"""Evaluate review-level TITAN-SR ranking scores.

The fixed-recall operating point screens every record whose score is greater
than or equal to the score of the required positive.  Including the complete
cutoff tie is conservative and makes the result independent of input order.

``work_saved_at_recall`` is the unscreened fraction.  Cohen's WSS@R is that
fraction minus the work a random ordering would save at the same target
recall: ``work_saved_at_recall - (1 - target_recall)``.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


def _validated_arrays(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or len(labels) != len(scores):
        raise ValueError("labels and scores must be one-dimensional and equally sized")
    if len(labels) == 0:
        raise ValueError("labels and scores must not be empty")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must contain only 0 and 1")
    if not np.isfinite(scores).all():
        raise ValueError("scores must all be finite")
    return labels.astype(np.int8), scores


def operating_point_at_recall(labels, scores, target_recall):
    """Return a conservative, deterministic fixed-recall operating point.

    The threshold is the score of the ``ceil(target_recall * n_positive)``-th
    highest positive.  All records tied at that score are screened.
    """

    if not 0 < target_recall <= 1:
        raise ValueError("target_recall must be in (0, 1]")
    labels, scores = _validated_arrays(labels, scores)
    positive_scores = scores[labels == 1]
    if len(positive_scores) == 0:
        return None

    required = int(np.ceil(target_recall * len(positive_scores)))
    threshold = float(np.sort(positive_scores)[::-1][required - 1])
    screened = scores >= threshold
    true_positive = int(((labels == 1) & screened).sum())
    true_negative = int(((labels == 0) & ~screened).sum())
    n_negative = int((labels == 0).sum())
    work_saved = float((~screened).mean())

    return {
        "target_recall": float(target_recall),
        "threshold": threshold,
        "screened": int(screened.sum()),
        "achieved_recall": true_positive / len(positive_scores),
        "specificity": true_negative / n_negative if n_negative else None,
        "work_saved_at_recall": work_saved,
        "wss_at_recall": work_saved - (1.0 - target_recall),
    }


def spec_at_recall(labels, scores, target_recall):
    point = operating_point_at_recall(labels, scores, target_recall)
    return None if point is None else point["specificity"]


def work_saved_at_recall(labels, scores, target_recall=0.95):
    point = operating_point_at_recall(labels, scores, target_recall)
    return None if point is None else point["work_saved_at_recall"]


def wss_at_recall(labels, scores, target_recall=0.95):
    point = operating_point_at_recall(labels, scores, target_recall)
    return None if point is None else point["wss_at_recall"]


def bootstrap_median(rows, key, bootstrap, rng):
    values = np.asarray([row[key] for row in rows if row[key] is not None], dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None
    if bootstrap <= 0:
        raise ValueError("bootstrap must be positive")
    bootstrapped = np.empty(bootstrap, dtype=float)
    # Bound memory for evaluations containing thousands of reviews.
    for start in range(0, bootstrap, 256):
        stop = min(start + 256, bootstrap)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        bootstrapped[start:stop] = np.median(values[indices], axis=1)
    return {
        "median": round(float(np.median(values)), 4),
        "ci95": [
            round(float(np.percentile(bootstrapped, 2.5)), 4),
            round(float(np.percentile(bootstrapped, 97.5)), 4),
        ],
    }


def evaluate(data, bootstrap=10_000, seed=42, score_key="score"):
    rng = np.random.default_rng(seed)
    rows = []
    skipped = {}
    for review_id, review in data.items():
        if score_key not in review:
            raise KeyError(
                f"review {review_id!r} has no score field {score_key!r}; "
                f"available fields: {sorted(review)}"
            )
        labels, scores = _validated_arrays(review["labels"], review[score_key])
        n_positive = int(labels.sum())
        if n_positive < 2:
            skipped[review_id] = "fewer_than_two_positives"
            continue
        if n_positive == len(labels):
            skipped[review_id] = "no_negatives"
            continue

        point95 = operating_point_at_recall(labels, scores, 0.95)
        point99 = operating_point_at_recall(labels, scores, 0.99)
        rows.append(
            {
                "review": review_id,
                "n_records": len(labels),
                "n_positive": n_positive,
                "auc": float(roc_auc_score(labels, scores)),
                "specificity_at_95_recall": point95["specificity"],
                "specificity_at_99_recall": point99["specificity"],
                "work_saved_at_95_recall": point95["work_saved_at_recall"],
                "wss_at_95_recall": point95["wss_at_recall"],
            }
        )

    summary = {"n_reviews": len(rows), "n_skipped_reviews": len(skipped)}
    for key in (
        "auc",
        "specificity_at_95_recall",
        "specificity_at_99_recall",
        "work_saved_at_95_recall",
        "wss_at_95_recall",
    ):
        summary[key] = bootstrap_median(rows, key, bootstrap, rng)
    return {"summary": summary, "per_review": rows, "skipped_reviews": skipped}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--score-key",
        default="score",
        help="score field inside each review (for example score, 2model, or 3model)",
    )
    args = parser.parse_args()

    with open(args.scores, encoding="utf-8") as handle:
        data = json.load(handle)
    result = evaluate(
        data,
        bootstrap=args.bootstrap,
        seed=args.seed,
        score_key=args.score_key,
    )
    print(json.dumps(result["summary"], indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")


if __name__ == "__main__":
    main()
