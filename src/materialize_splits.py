"""Materialize review-level train/validation/test CSVs from reconstructed data.

The released split uses source-prefixed assignment IDs (``pmc_``, ``tb_``,
and ``chan_``), while the manifests contain unprefixed review IDs. This CLI
resolves those IDs, removes duplicate review/citation rows, handles conflicting
labels explicitly, and prevents the same citation identifier from crossing
split boundaries. By default, a citation found in several splits is retained
only in the highest-priority split (test, then validation, then training).
"""

import argparse
import csv
import gzip
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone

csv.field_size_limit(10**9)
SOURCE_PREFIX = {"pmc": "pmc_", "trackb": "tb_", "tb": "tb_", "chan": "chan_"}


def open_text(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="")
    return open(path, encoding="utf-8", errors="replace", newline="")


def read_inputs(paths):
    rows = []
    fieldnames = []
    for path in paths:
        with open_text(path) as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError(f"input has no header: {path}")
            for field in reader.fieldnames:
                if field not in fieldnames:
                    fieldnames.append(field)
            for input_row, row in enumerate(reader, start=2):
                row["_input_path"] = path
                row["_input_row"] = input_row
                rows.append(row)
    return fieldnames, rows


def resolve_assignment_id(row, assignments):
    review_id = (row.get("review_id") or "").strip()
    source = (row.get("source") or "").strip().lower()
    candidates = [review_id]
    if source in SOURCE_PREFIX:
        candidates.append(SOURCE_PREFIX[source] + review_id)
    if review_id.startswith("PMC"):
        candidates.append("pmc_" + review_id)
    candidates.extend(("pmc_" + review_id, "tb_" + review_id, "chan_" + review_id))
    matches = list(dict.fromkeys(candidate for candidate in candidates if candidate in assignments))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"ambiguous split assignment for review {review_id!r}: {matches}")
    return None


def normalized_label(row):
    value = (row.get("label") or "").strip()
    if value not in ("0", "1", "0.0", "1.0"):
        raise ValueError(
            f"invalid label {value!r} in {row.get('_input_path')} row {row.get('_input_row')}"
        )
    return str(int(float(value)))


def stable_row_id(row, key_column):
    components = (
        row.get("source", ""),
        row.get("review_id", ""),
        row.get(key_column, ""),
        row.get("title", ""),
    )
    payload = "\0".join(value.strip().lower() for value in components)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def deduplicate_review_keys(rows, key_column, conflict_policy):
    grouped = defaultdict(list)
    without_key = []
    for row in rows:
        citation_id = (row.get(key_column) or "").strip()
        if not citation_id:
            without_key.append(row)
        else:
            grouped[(row["review_id"], citation_id)].append(row)

    kept = list(without_key)
    stats = Counter()
    conflict_examples = []
    for key, group in grouped.items():
        labels = {row["label"] for row in group}
        if len(labels) > 1:
            stats["conflicting_review_citation_keys"] += 1
            stats["conflicting_rows"] += len(group)
            if len(conflict_examples) < 20:
                conflict_examples.append({"review_id": key[0], key_column: key[1], "labels": sorted(labels)})
            if conflict_policy == "error":
                raise ValueError(f"conflicting labels for review/citation key {key}")
            if conflict_policy == "keep":
                kept.extend(group)
            continue
        kept.append(group[0])
        stats["duplicate_same_label_rows_removed"] += len(group) - 1
    return kept, stats, conflict_examples


def remove_cross_split_overlap(rows, key_column, priority):
    priority_index = {name: index for index, name in enumerate(priority)}
    unknown = sorted({row["split"] for row in rows}.difference(priority_index))
    if unknown:
        raise ValueError(f"--split-priority is missing split names: {unknown}")

    highest_priority = {}
    crossed = defaultdict(set)
    for row in rows:
        key = (row.get(key_column) or "").strip()
        if not key:
            continue
        crossed[key].add(row["split"])
        if key not in highest_priority or priority_index[row["split"]] < priority_index[highest_priority[key]]:
            highest_priority[key] = row["split"]

    kept = []
    removed = Counter()
    for row in rows:
        key = (row.get(key_column) or "").strip()
        if key and row["split"] != highest_priority[key]:
            removed[row["split"]] += 1
        else:
            kept.append(row)
    overlapping_keys = sum(len(splits) > 1 for splits in crossed.values())
    return kept, removed, overlapping_keys


def write_csv_atomic(path, fieldnames, rows):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_json_atomic(path, payload):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True, help="reconstructed CSV; repeatable")
    parser.add_argument("--splits", required=True, help="review-level split JSON")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--key-column", default="candidate_pmid")
    parser.add_argument(
        "--conflicts",
        choices=("exclude", "error", "keep"),
        default="exclude",
        help="handling for conflicting labels on one review/citation key",
    )
    parser.add_argument(
        "--unassigned",
        choices=("error", "exclude"),
        default="error",
        help="handling for review IDs absent from the split file",
    )
    parser.add_argument(
        "--split-priority",
        default="test,val,train",
        help="retention priority when a citation identifier crosses splits",
    )
    parser.add_argument("--audit", default=None)
    args = parser.parse_args()

    with open(args.splits, encoding="utf-8") as handle:
        split_payload = json.load(handle)
    assignments = split_payload.get("assignments")
    if not isinstance(assignments, dict):
        parser.error("split JSON must contain an assignments object")

    fieldnames, input_rows = read_inputs(args.input)
    required = {"review_id", "label", args.key_column}
    if missing := required.difference(fieldnames):
        parser.error(f"input CSVs are missing columns: {sorted(missing)}")

    assigned_rows = []
    unassigned = Counter()
    for row in input_rows:
        assignment_id = resolve_assignment_id(row, assignments)
        if assignment_id is None:
            unassigned[row.get("review_id", "")] += 1
            continue
        row["label"] = normalized_label(row)
        row["split_assignment_id"] = assignment_id
        row["split"] = assignments[assignment_id]["split"]
        row["row_id"] = stable_row_id(row, args.key_column)
        assigned_rows.append(row)
    if unassigned and args.unassigned == "error":
        examples = list(unassigned)[:10]
        raise ValueError(
            f"{len(unassigned)} review IDs ({sum(unassigned.values())} rows) have no split assignment; "
            f"examples: {examples}"
        )

    deduplicated, duplicate_stats, conflict_examples = deduplicate_review_keys(
        assigned_rows, args.key_column, args.conflicts
    )
    clean_rows, cross_split_removed, overlapping_keys = remove_cross_split_overlap(
        deduplicated,
        args.key_column,
        [value.strip() for value in args.split_priority.split(",") if value.strip()],
    )

    output_fields = list(fieldnames)
    for field in ("row_id", "split_assignment_id", "split"):
        if field not in output_fields:
            output_fields.append(field)
    os.makedirs(args.out_dir, exist_ok=True)
    output_counts = Counter(row["split"] for row in clean_rows)
    split_names = sorted({value["split"] for value in assignments.values()})
    for split_name in split_names:
        write_csv_atomic(
            os.path.join(args.out_dir, f"{split_name}.csv"),
            output_fields,
            [row for row in clean_rows if row["split"] == split_name],
        )

    assignment_counts = Counter(value["split"] for value in assignments.values())
    assignment_item_counts = Counter()
    for value in assignments.values():
        assignment_item_counts[value["split"]] += int(value.get("n_items", 0))
    declared_metadata = split_payload.get("metadata", {})
    metadata_discrepancies = {}
    for split_name in split_names:
        declared = declared_metadata.get(f"{split_name}_reviews")
        actual = assignment_counts[split_name]
        if declared is not None and declared != actual:
            metadata_discrepancies[f"{split_name}_reviews"] = {"declared": declared, "actual": actual}
    declared_total = declared_metadata.get("total_reviews")
    if declared_total is not None and declared_total != len(assignments):
        metadata_discrepancies["total_reviews"] = {
            "declared": declared_total,
            "actual": len(assignments),
        }
    declared_items = declared_metadata.get("total_items")
    actual_items = sum(assignment_item_counts.values())
    if declared_items is not None and declared_items != actual_items:
        metadata_discrepancies["total_items"] = {
            "declared": declared_items,
            "actual": actual_items,
        }

    audit_path = args.audit or os.path.join(args.out_dir, "materialization.audit.json")
    audit = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": args.input,
        "split_file": args.splits,
        "key_column": args.key_column,
        "conflict_policy": args.conflicts,
        "split_priority": args.split_priority,
        "input_rows": len(input_rows),
        "assigned_rows": len(assigned_rows),
        "unassigned_reviews": len(unassigned),
        "unassigned_rows": sum(unassigned.values()),
        "unassigned_review_examples": list(unassigned)[:20],
        "duplicate_handling": dict(duplicate_stats),
        "conflict_examples": conflict_examples,
        "cross_split_overlapping_keys": overlapping_keys,
        "cross_split_rows_removed": dict(cross_split_removed),
        "output_rows_by_split": dict(output_counts),
        "assignment_reviews_by_split": dict(assignment_counts),
        "assignment_items_by_split": dict(assignment_item_counts),
        "split_metadata_discrepancies": metadata_discrepancies,
    }
    write_json_atomic(audit_path, audit)
    print(f"materialized {sum(output_counts.values())} rows into {args.out_dir}")
    print(f"split counts: {dict(output_counts)}")
    print(f"audit: {audit_path}")
    if metadata_discrepancies:
        print(f"warning: split metadata discrepancies recorded: {metadata_discrepancies}")


if __name__ == "__main__":
    main()
