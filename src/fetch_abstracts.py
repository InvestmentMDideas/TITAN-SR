"""Reconstruct citation text from PubMed with a persistent cache and audit log.

The input label manifest is never modified. Successful batches are committed
to the cache immediately, so an interrupted run can resume without re-fetching
them. Network failures are distinct from PubMed records that genuinely have no
abstract; network failures fail closed unless ``--allow-partial`` is supplied.
"""

import argparse
import csv
import gzip
import json
import os
import sqlite3
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone

csv.field_size_limit(10**9)
CACHE_SCHEMA_VERSION = 1


def open_text(path):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace", newline="")
    return open(path, encoding="utf-8", errors="replace", newline="")


def fetch_batch(pmids, api_key=None, email=None, retries=5, timeout=90):
    parameters = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
        "tool": "TITAN-SR-reproducibility",
    }
    if api_key:
        parameters["api_key"] = api_key
    if email:
        parameters["email"] = email
    request = urllib.request.Request(
        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
        data=urllib.parse.urlencode(parameters).encode(),
        headers={"User-Agent": "TITAN-SR-reproducibility/1.0"},
    )
    errors = []
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except Exception as exc:  # urllib exposes several unrelated exception classes
            errors.append(f"{type(exc).__name__}: {exc}")
            if attempt + 1 < retries:
                time.sleep(min(8.0, 0.5 * (2**attempt)))
    raise RuntimeError(f"PubMed efetch failed after {retries} attempts: {errors[-1]}")


def parse_pubmed_xml(xml_bytes):
    root = ET.fromstring(xml_bytes)
    records = {}
    for article in root.iter("PubmedArticle"):
        pmid_node = article.find(".//MedlineCitation/PMID")
        if pmid_node is None or not (pmid_node.text or "").strip():
            continue
        pmid = pmid_node.text.strip()
        title_node = article.find(".//Article/ArticleTitle")
        title = "".join(title_node.itertext()).strip() if title_node is not None else ""
        abstract_parts = []
        for abstract_node in article.findall(".//Article/Abstract/AbstractText"):
            text = "".join(abstract_node.itertext()).strip()
            label = (abstract_node.get("Label") or "").strip()
            if text:
                abstract_parts.append(f"{label}: {text}" if label else text)
        abstract = " ".join(abstract_parts)
        records[pmid] = {
            "title": title,
            "abstract": abstract,
            "status": "ok" if abstract else "no_abstract",
        }
    return records


def connect_cache(path):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    connection = sqlite3.connect(path)
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if version not in (0, CACHE_SCHEMA_VERSION):
        connection.close()
        raise ValueError(f"unsupported PubMed cache schema version {version} in {path}")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS pubmed_records (
            pmid TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            abstract TEXT NOT NULL,
            status TEXT NOT NULL,
            fetched_utc TEXT NOT NULL
        )
        """
    )
    connection.execute(f"PRAGMA user_version = {CACHE_SCHEMA_VERSION}")
    connection.commit()
    return connection


def cache_statuses(connection, pmids):
    statuses = {}
    for start in range(0, len(pmids), 800):
        batch = pmids[start : start + 800]
        placeholders = ",".join("?" for _ in batch)
        query = f"SELECT pmid, status FROM pubmed_records WHERE pmid IN ({placeholders})"
        statuses.update(connection.execute(query, batch).fetchall())
    return statuses


def cached_record(connection, pmid):
    row = connection.execute(
        "SELECT title, abstract, status FROM pubmed_records WHERE pmid = ?", (pmid,)
    ).fetchone()
    if row is None:
        return None
    return {"title": row[0], "abstract": row[1], "status": row[2]}


def upsert_cache(connection, records):
    rows = [
        (
            pmid,
            record.get("title", ""),
            record.get("abstract", ""),
            record["status"],
            record.get("fetched_utc") or datetime.now(timezone.utc).isoformat(),
        )
        for pmid, record in records.items()
    ]
    with connection:
        connection.executemany(
            """
            INSERT INTO pubmed_records (pmid, title, abstract, status, fetched_utc)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(pmid) DO UPDATE SET
                title = excluded.title,
                abstract = excluded.abstract,
                status = excluded.status,
                fetched_utc = excluded.fetched_utc
            """,
            rows,
        )


def save_cache(path, records):
    """Small convenience API used by tests and one-off cache seeding."""

    connection = connect_cache(path)
    try:
        upsert_cache(connection, records)
    finally:
        connection.close()


def save_json_atomic(path, payload):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def read_rows(path):
    with open_text(path) as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames or [], list(reader)


def criteria_by_review(path):
    fields, rows = read_rows(path)
    if "review_id" not in fields:
        raise ValueError("criteria manifest must contain review_id")
    candidates = [name for name in ("eligibility_criteria", "selection_criteria", "criteria") if name in fields]
    if not candidates:
        raise ValueError(
            "criteria manifest must contain eligibility_criteria, selection_criteria, or criteria"
        )
    criteria_column = candidates[0]
    return {row["review_id"]: row.get(criteria_column) or "" for row in rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--criteria", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--cache",
        default=None,
        help="persistent SQLite cache (default: OUT.pubmed-cache.sqlite3)",
    )
    parser.add_argument("--audit", default=None, help="audit JSON (default: OUT.audit.json)")
    parser.add_argument("--batch", type=int, default=200)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument(
        "--missing-abstract",
        choices=("exclude", "keep", "error"),
        default="exclude",
        help="handling for successfully fetched PubMed records with no abstract",
    )
    parser.add_argument(
        "--refresh-missing",
        action="store_true",
        help="retry cached PubMed IDs whose prior status was not_returned",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="write rows available from cache/successful batches despite failed network batches",
    )
    args = parser.parse_args()
    if not 1 <= args.batch <= 200:
        parser.error("--batch must be between 1 and 200")
    if args.retries < 1:
        parser.error("--retries must be positive")

    cache_path = args.cache or f"{args.out}.pubmed-cache.sqlite3"
    audit_path = args.audit or f"{args.out}.audit.json"
    fields, label_rows = read_rows(args.labels)
    required = {"review_id", "candidate_pmid", "label"}
    if missing := required.difference(fields):
        parser.error(f"labels manifest is missing columns: {sorted(missing)}")
    criteria = criteria_by_review(args.criteria)
    cache_connection = connect_cache(cache_path)

    pmids = sorted(
        {
            (row.get("candidate_pmid") or "").strip()
            for row in label_rows
            if (row.get("candidate_pmid") or "").strip().isdigit()
        },
        key=int,
    )
    statuses = cache_statuses(cache_connection, pmids)
    initially_cached = {
        pmid
        for pmid in pmids
        if pmid in statuses and not (args.refresh_missing and statuses[pmid] == "not_returned")
    }
    pending = [pmid for pmid in pmids if pmid not in initially_cached]
    failed_batches = []
    api_key = os.environ.get("NCBI_API_KEY")
    email = os.environ.get("NCBI_EMAIL")
    request_interval = 0.11 if api_key else 0.35

    for start in range(0, len(pending), args.batch):
        batch = pending[start : start + args.batch]
        try:
            parsed = parse_pubmed_xml(
                fetch_batch(
                    batch,
                    api_key=api_key,
                    email=email,
                    retries=args.retries,
                    timeout=args.timeout,
                )
            )
            unexpected = set(parsed).difference(batch)
            if unexpected:
                raise ValueError(f"PubMed returned unexpected PMIDs: {sorted(unexpected)[:5]}")
            fetched_at = datetime.now(timezone.utc).isoformat()
            batch_records = {}
            for pmid in batch:
                record = parsed.get(
                    pmid,
                    {"title": "", "abstract": "", "status": "not_returned"},
                )
                record["fetched_utc"] = fetched_at
                batch_records[pmid] = record
            upsert_cache(cache_connection, batch_records)
            statuses.update({pmid: record["status"] for pmid, record in batch_records.items()})
        except Exception as exc:
            failed_batches.append(
                {
                    "first_pmid": batch[0],
                    "last_pmid": batch[-1],
                    "n_pmids": len(batch),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        completed = min(start + len(batch), len(pending))
        print(f"fetched/resolved {completed}/{len(pending)} uncached PMIDs", flush=True)
        if start + args.batch < len(pending):
            time.sleep(request_interval)

    output_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temporary_output = f"{output_path}.tmp.{os.getpid()}"
    output_fields = [
        "review_id",
        "candidate_pmid",
        "criteria",
        "title",
        "abstract",
        "label",
        "source",
    ]
    output_count = 0
    excluded = Counter()
    missing_criteria_reviews = set()
    with open(temporary_output, "w", encoding="utf-8", newline="") as output_handle:
        writer = csv.DictWriter(output_handle, fieldnames=output_fields)
        writer.writeheader()
        for row in label_rows:
            pmid = (row.get("candidate_pmid") or "").strip()
            if not pmid.isdigit():
                excluded["invalid_or_missing_pmid"] += 1
                continue
            record = cached_record(cache_connection, pmid)
            if record is None:
                excluded["unavailable_after_fetch_failure"] += 1
                continue
            if record.get("status") == "not_returned":
                excluded["not_returned_by_pubmed"] += 1
                continue
            if not record.get("abstract"):
                if args.missing_abstract == "error":
                    excluded["missing_abstract_error"] += 1
                    continue
                if args.missing_abstract == "exclude":
                    excluded["no_abstract"] += 1
                    continue
            label = (row.get("label") or "").strip()
            if label not in ("0", "1", "0.0", "1.0"):
                raise ValueError(f"invalid binary label {label!r} for PMID {pmid}")
            review_id = row["review_id"]
            if review_id not in criteria:
                missing_criteria_reviews.add(review_id)
            writer.writerow(
                {
                    "review_id": review_id,
                    "candidate_pmid": pmid,
                    "criteria": criteria.get(review_id, ""),
                    "title": record.get("title", ""),
                    "abstract": record.get("abstract", ""),
                    "label": str(int(float(label))),
                    "source": row.get("source", ""),
                }
            )
            output_count += 1

    requested_status = Counter(statuses[pmid] for pmid in pmids if pmid in statuses)
    fatal_reasons = []
    if failed_batches and not args.allow_partial:
        fatal_reasons.append(f"{len(failed_batches)} PubMed batch(es) failed")
    if excluded["missing_abstract_error"]:
        fatal_reasons.append(
            f"{excluded['missing_abstract_error']} rows have no abstract under --missing-abstract error"
        )
    audit = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {"labels": args.labels, "criteria": args.criteria},
        "outputs": {"csv": args.out, "cache": cache_path, "audit": audit_path},
        "input_rows": len(label_rows),
        "unique_valid_pmids": len(pmids),
        "cache_hits_at_start": len(initially_cached),
        "pmids_requested_this_run": len(pending),
        "requested_pmid_status": dict(sorted(requested_status.items())),
        "output_rows": output_count,
        "excluded_rows_by_reason": dict(sorted(excluded.items())),
        "reviews_missing_criteria": len(missing_criteria_reviews),
        "reviews_missing_criteria_examples": sorted(missing_criteria_reviews)[:20],
        "failed_batches": failed_batches,
        "allow_partial": args.allow_partial,
        "missing_abstract_policy": args.missing_abstract,
        "output_committed": not fatal_reasons,
        "fatal_reasons": fatal_reasons,
    }
    save_json_atomic(audit_path, audit)
    cache_connection.close()

    if fatal_reasons:
        os.remove(temporary_output)
        raise SystemExit("; ".join(fatal_reasons) + f"; see audit: {audit_path}")

    os.replace(temporary_output, output_path)
    print(
        f"wrote {output_count} rows -> {args.out}; cache={cache_path}; audit={audit_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
