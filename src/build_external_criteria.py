"""Rebuild the external-holdout eligibility criteria from the Chan et al. archive.

The 22 temporal-holdout reviews are Cochrane reviews distributed by Chan et al.
(Mendeley Data, doi:10.17632/7sgmg89zb6.1, CC BY 4.0) in
``20240827_Systematic_review_manual_search_data.zip``. This package does not
redistribute the review text; download the archive from the publisher, then run:

    python src/build_external_criteria.py \
      --archive 20240827_Systematic_review_manual_search_data.zip \
      --out data/reconstructed/external_criteria.csv

The archive is checked against the SHA-256 recorded in
``data/sources/chan_mendeley_v1.json`` and every rebuilt passage is checked
against the hash in ``data/manifests/criteria_withheld.csv.gz``.
"""

import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import re
import zipfile
from pathlib import Path

csv.field_size_limit(10**9)
ROOT = Path(__file__).resolve().parents[1]
SECTION = re.compile(r"\[SEL\]\s*(.*?)(?=\s*\[[A-Z]{2,5}\]|$)", re.S)
MEMBER = re.compile(r"(?:^|/)(CD\d+)_data_cleaned\.csv$")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_hashes(path, source="external_cochrane"):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return {
            row["review_id"]: row["text_sha256"]
            for row in csv.DictReader(handle)
            if row["source"] == source
        }


def selection_criteria(review_data):
    match = SECTION.search(review_data or "")
    return match.group(1).strip() if match else ""


def rebuild(archive):
    criteria = {}
    with zipfile.ZipFile(archive) as bundle:
        for name in sorted(bundle.namelist()):
            match = MEMBER.search(name)
            if not match:
                continue
            with bundle.open(name) as raw:
                reader = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", newline=""))
                first = next(reader, None)
            if first is not None:
                criteria[match.group(1)] = selection_criteria(first.get("Review_data"))
    return criteria


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--archive", required=True, help="Chan manual-search ZIP from Mendeley Data")
    parser.add_argument("--out", required=True)
    parser.add_argument("--source-record", default=str(ROOT / "data" / "sources" / "chan_mendeley_v1.json"))
    parser.add_argument("--hashes", default=str(ROOT / "data" / "manifests" / "criteria_withheld.csv.gz"))
    parser.add_argument("--skip-archive-check", action="store_true", help="do not verify the archive SHA-256")
    args = parser.parse_args()

    if not args.skip_archive_check:
        record = json.loads(Path(args.source_record).read_text(encoding="utf-8"))
        expected = record["manual_search_archive"]["sha256"]
        actual = sha256_file(args.archive)
        if actual != expected:
            raise SystemExit(f"archive SHA-256 {actual} does not match the recorded {expected}")

    wanted = expected_hashes(args.hashes)
    rebuilt = rebuild(args.archive)
    missing = sorted(set(wanted) - set(rebuilt))
    mismatched = sorted(
        review_id
        for review_id, digest in wanted.items()
        if review_id in rebuilt and hashlib.sha256(rebuilt[review_id].encode("utf-8")).hexdigest() != digest
    )
    if missing or mismatched:
        raise SystemExit(f"rebuild failed: missing={missing} hash_mismatch={mismatched}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["review_id", "selection_criteria"])
        writer.writeheader()
        for review_id in sorted(wanted):
            writer.writerow({"review_id": review_id, "selection_criteria": rebuilt[review_id]})
    print(f"wrote {len(wanted)} reviews -> {args.out}; all passages match the recorded hashes")


if __name__ == "__main__":
    main()
