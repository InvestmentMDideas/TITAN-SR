"""Citation-title overlap and a within-review positive-control analysis.

Inputs are an inference ``scores.json`` (including ``titles``) and a JSON list
of training titles. Equal scores receive the same average percentile so the
analysis is deterministic under row reordering.
"""

import argparse
import json
import re

import numpy as np


def normalize_title(title):
    value = re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()
    return value if len(value) >= 10 else None


def average_percentiles(scores):
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    if len(scores) <= 1:
        return np.ones(len(scores), dtype=float)
    unique, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    starts = np.cumsum(counts) - counts
    average_zero_based_rank = starts + (counts - 1) / 2
    return average_zero_based_rank[inverse] / (len(scores) - 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", required=True)
    parser.add_argument("--train_titles", required=True, help="JSON list of training titles")
    args = parser.parse_args()

    with open(args.scores, encoding="utf-8") as handle:
        predictions = json.load(handle)
    with open(args.train_titles, encoding="utf-8") as handle:
        normalized = [normalize_title(value) for value in json.load(handle)]
    training_titles = {value for value in normalized if value is not None}

    seen_percentiles = []
    unseen_percentiles = []
    per_review = []
    n_overlap = n_total = n_overlap_positive = n_positive = 0
    for review_id, review in predictions.items():
        labels = np.asarray(review["labels"])
        scores = np.asarray(review["score"], dtype=float)
        if "titles" not in review:
            raise ValueError(
                f"review {review_id!r} has no titles; regenerate scores with src/inference.py"
            )
        titles = review["titles"]
        if not (len(labels) == len(scores) == len(titles)):
            raise ValueError(f"review {review_id!r} has mismatched labels, scores, and titles")
        overlap = np.asarray(
            [
                (value := normalize_title(title)) is not None and value in training_titles
                for title in titles
            ]
        )
        n_total += len(labels)
        n_overlap += int(overlap.sum())
        n_positive += int(labels.sum())
        n_overlap_positive += int(((labels == 1) & overlap).sum())

        percentile = average_percentiles(scores)
        included = labels == 1
        seen = included & overlap
        unseen = included & ~overlap
        if seen.sum() and unseen.sum():
            per_review.append(float(np.median(percentile[seen]) - np.median(percentile[unseen])))
        seen_percentiles.extend(percentile[seen].tolist())
        unseen_percentiles.extend(percentile[unseen].tolist())

    print(
        f"overlap records {n_overlap}/{n_total} ({100 * n_overlap / max(n_total, 1):.1f}%); "
        f"overlap-includes {n_overlap_positive}/{n_positive} "
        f"({100 * n_overlap_positive / max(n_positive, 1):.1f}%)"
    )
    if seen_percentiles and unseen_percentiles and per_review:
        print(
            f"positive control: seen include median pctile {np.median(seen_percentiles):.3f} "
            f"vs unseen {np.median(unseen_percentiles):.3f}; review-level median diff "
            f"{np.median(per_review):+.3f} (>0 would be consistent with memorization)"
        )
    else:
        print("positive control unavailable: no review contains both seen and unseen includes")


if __name__ == "__main__":
    main()
