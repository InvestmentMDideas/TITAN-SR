"""Score citations with a local TITAN-SR weighted ensemble."""

import argparse
import csv
import glob
import hashlib
import json
import os

import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoModelForSequenceClassification, AutoTokenizer

csv.field_size_limit(10**9)


def num_labels(path):
    safetensor_files = glob.glob(os.path.join(path, "*.safetensors"))
    if not safetensor_files:
        return 1
    with safe_open(safetensor_files[0], framework="pt") as handle:
        for key in handle.keys():
            if key.endswith("classifier.weight"):
                return handle.get_slice(key).get_shape()[0]
    return 1


def verify_reference_hash(path, expected):
    model_files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    if len(model_files) != 1:
        raise ValueError(
            f"hash verification expects exactly one .safetensors file in {path}; found {len(model_files)}"
        )
    digest = hashlib.sha256()
    with open(model_files[0], "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual.lower() != expected.lower():
        raise ValueError(
            f"reference weight hash mismatch for {model_files[0]}: expected {expected}, got {actual}"
        )


def select_ensemble(config, variant="primary"):
    """Select the primary ensemble or a named nested sensitivity variant."""

    if variant in (None, "", "primary"):
        selected = dict(config)
    else:
        if variant not in config or not isinstance(config[variant], dict):
            available = ["primary"] + [
                key for key, value in config.items() if isinstance(value, dict) and "members" in value
            ]
            raise ValueError(f"unknown ensemble variant {variant!r}; choose from {available}")
        selected = dict(config)
        selected.update(config[variant])
        selected["variant"] = variant

    members = selected.get("members")
    if not isinstance(members, list) or not members:
        raise ValueError("selected ensemble must contain a non-empty members list")
    names = [member.get("name") for member in members]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("ensemble member names must be present and unique")
    weights = np.asarray([member.get("weight") for member in members], dtype=float)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("ensemble weights must be finite and positive")
    selected["weights"] = (weights / weights.sum()).tolist()
    return selected


def load_reviews(path, columns):
    try:
        review_col, criteria_col, title_col, abstract_col, label_col = columns.split(",")
    except ValueError as exc:
        raise ValueError("--cols must contain review_id,criteria,title,abstract,label") from exc

    reviews = {}
    with open(path, encoding="utf-8", errors="replace", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {
            col
            for col in (review_col, criteria_col, title_col, abstract_col, label_col)
            if col not in (reader.fieldnames or [])
        }
        if missing:
            raise ValueError(f"input is missing required columns: {sorted(missing)}")
        for row_number, row in enumerate(reader, start=2):
            review_id = (row.get(review_col) or "").strip()
            if not review_id:
                raise ValueError(f"empty review id at row {row_number}")
            label_text = (row.get(label_col) or "").strip()
            if label_text not in ("0", "1", "0.0", "1.0"):
                raise ValueError(f"invalid binary label {label_text!r} at row {row_number}")
            criteria = row.get(criteria_col, "") or ""
            title = row.get(title_col, "") or ""
            abstract = row.get(abstract_col, "") or ""
            review = reviews.setdefault(
                review_id,
                {"criteria": criteria, "texts": [], "titles": [], "labels": []},
            )
            if review["criteria"] != criteria:
                raise ValueError(f"review {review_id!r} has inconsistent eligibility criteria")
            review["texts"].append(f"{title}. {abstract}".strip())
            review["titles"].append(title)
            review["labels"].append(int(float(label_text)))
    return reviews


@torch.no_grad()
def score(tokenizer, model, criteria, texts, max_length, device, batch_size=64):
    output = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start : start + batch_size]
        encoded = tokenizer(
            [criteria] * len(chunk),
            chunk,
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        ).to(device)
        logits = model(**encoded).logits
        if logits.shape[-1] > 1:
            probabilities = torch.softmax(logits, dim=-1)[:, 1]
        else:
            probabilities = torch.sigmoid(logits.squeeze(-1))
        output.extend(probabilities.float().cpu().tolist())
    return np.asarray(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--variant",
        default="primary",
        help="primary, or a named nested ensemble block such as sensitivity_two_model_50_50",
    )
    parser.add_argument(
        "--models",
        required=True,
        help="directory containing one subdirectory per selected member, named by member 'name'",
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cols", default="review_id,criteria,title,abstract,label")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--verify-reference-hashes",
        action="store_true",
        help="verify model.safetensors against reference_sha256 values in the selected config",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        config = select_ensemble(json.load(handle), args.variant)
    max_length = config.get("max_length", 512)
    weights = np.asarray(config["weights"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    members = []
    for member in config["members"]:
        path = os.path.join(args.models, member["name"])
        if not os.path.isdir(path):
            raise FileNotFoundError(f"missing ensemble member directory: {path}")
        if args.verify_reference_hashes:
            expected = member.get("reference_sha256")
            if not expected:
                raise ValueError(f"member {member['name']!r} has no reference_sha256")
            verify_reference_hash(path, expected)
        tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForSequenceClassification.from_pretrained(
            path, num_labels=num_labels(path)
        ).to(device).eval()
        members.append((tokenizer, model))

    reviews = load_reviews(args.input, args.cols)
    result = {}
    for review_id, review in reviews.items():
        predictions = [
            score(
                tokenizer,
                model,
                review["criteria"],
                review["texts"],
                max_length,
                device,
                args.batch_size,
            )
            for tokenizer, model in members
        ]
        result[review_id] = {
            "labels": review["labels"],
            "score": np.average(np.stack(predictions), axis=0, weights=weights).tolist(),
            "titles": review["titles"],
        }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle)
        handle.write("\n")
    print(f"scored {len(result)} reviews with variant {args.variant!r} -> {args.out}")


if __name__ == "__main__":
    main()
