import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import fetch_abstracts  # noqa: E402


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class FetchAbstractTests(unittest.TestCase):
    def test_structured_abstract_parser_preserves_labels_and_inline_text(self):
        xml = b"""<PubmedArticleSet><PubmedArticle><MedlineCitation>
        <PMID>123</PMID><Article><ArticleTitle>A <i>nested</i> title</ArticleTitle>
        <Abstract><AbstractText Label="METHODS">Did <b>work</b>.</AbstractText>
        <AbstractText Label="RESULTS">It worked.</AbstractText></Abstract>
        </Article></MedlineCitation></PubmedArticle></PubmedArticleSet>"""
        record = fetch_abstracts.parse_pubmed_xml(xml)["123"]
        self.assertEqual(record["title"], "A nested title")
        self.assertEqual(record["abstract"], "METHODS: Did work. RESULTS: It worked.")
        self.assertEqual(record["status"], "ok")

    def test_cached_run_writes_pmid_and_audits_exclusions(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            labels = directory / "labels.csv"
            criteria = directory / "criteria.csv"
            output = directory / "out.csv"
            cache = directory / "cache.sqlite3"
            audit = directory / "audit.json"
            write_csv(
                labels,
                ["review_id", "candidate_pmid", "label", "source"],
                [
                    {"review_id": "r", "candidate_pmid": "123", "label": "1", "source": "pmc"},
                    {"review_id": "r", "candidate_pmid": "124", "label": "0", "source": "pmc"},
                    {"review_id": "r", "candidate_pmid": "bad", "label": "0", "source": "pmc"},
                ],
            )
            write_csv(criteria, ["review_id", "eligibility_criteria"], [{"review_id": "r", "eligibility_criteria": "RCT"}])
            fetch_abstracts.save_cache(
                cache,
                {
                    "123": {"title": "Title", "abstract": "Abstract", "status": "ok"},
                    "124": {"title": "No abstract", "abstract": "", "status": "no_abstract"},
                },
            )
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "src" / "fetch_abstracts.py"),
                    "--labels",
                    str(labels),
                    "--criteria",
                    str(criteria),
                    "--out",
                    str(output),
                    "--cache",
                    str(cache),
                    "--audit",
                    str(audit),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            with output.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            report = json.loads(audit.read_text())
            self.assertEqual(rows[0]["candidate_pmid"], "123")
            self.assertEqual(report["output_rows"], 1)
            self.assertEqual(report["excluded_rows_by_reason"]["no_abstract"], 1)
            self.assertEqual(report["excluded_rows_by_reason"]["invalid_or_missing_pmid"], 1)

    def test_network_failure_fails_closed_but_leaves_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            labels = directory / "labels.csv"
            criteria = directory / "criteria.csv"
            output = directory / "out.csv"
            audit = directory / "audit.json"
            write_csv(
                labels,
                ["review_id", "candidate_pmid", "label"],
                [{"review_id": "r", "candidate_pmid": "999", "label": "1"}],
            )
            write_csv(criteria, ["review_id", "criteria"], [{"review_id": "r", "criteria": "RCT"}])
            argv = [
                "fetch_abstracts.py",
                "--labels",
                str(labels),
                "--criteria",
                str(criteria),
                "--out",
                str(output),
                "--audit",
                str(audit),
                "--retries",
                "1",
            ]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                fetch_abstracts, "fetch_batch", side_effect=RuntimeError("offline")
            ), self.assertRaises(SystemExit):
                fetch_abstracts.main()
            self.assertFalse(output.exists())
            self.assertEqual(len(json.loads(audit.read_text())["failed_batches"]), 1)


class MaterializeSplitTests(unittest.TestCase):
    def test_prefix_resolution_conflict_exclusion_and_cross_split_dedup(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "input.csv"
            split_file = directory / "splits.json"
            output = directory / "split"
            write_csv(
                source,
                ["review_id", "candidate_pmid", "criteria", "title", "abstract", "label", "source"],
                [
                    {"review_id": "PMC1", "candidate_pmid": "100", "label": "0", "source": "pmc"},
                    {"review_id": "PMC2", "candidate_pmid": "100", "label": "1", "source": "pmc"},
                    {"review_id": "PMC2", "candidate_pmid": "100", "label": "1", "source": "pmc"},
                    {"review_id": "PMC3", "candidate_pmid": "200", "label": "0", "source": "pmc"},
                    {"review_id": "PMC3", "candidate_pmid": "200", "label": "1", "source": "pmc"},
                    {"review_id": "PMC1", "candidate_pmid": "300", "label": "1", "source": "pmc"},
                ],
            )
            split_file.write_text(
                json.dumps(
                    {
                        "metadata": {"total_reviews": 99, "total_items": 99},
                        "assignments": {
                            "pmc_PMC1": {"split": "train", "n_items": 2},
                            "pmc_PMC2": {"split": "test", "n_items": 2},
                            "pmc_PMC3": {"split": "val", "n_items": 2},
                        },
                    }
                )
            )
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "src" / "materialize_splits.py"),
                    "--input",
                    str(source),
                    "--splits",
                    str(split_file),
                    "--out-dir",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            def ids(split):
                with (output / f"{split}.csv").open(encoding="utf-8") as handle:
                    return [row["candidate_pmid"] for row in csv.DictReader(handle)]

            self.assertEqual(ids("test"), ["100"])
            self.assertEqual(ids("train"), ["300"])
            self.assertEqual(ids("val"), [])
            report = json.loads((output / "materialization.audit.json").read_text())
            self.assertEqual(report["duplicate_handling"]["duplicate_same_label_rows_removed"], 1)
            self.assertEqual(report["duplicate_handling"]["conflicting_review_citation_keys"], 1)
            self.assertEqual(report["cross_split_overlapping_keys"], 1)
            self.assertIn("total_reviews", report["split_metadata_discrepancies"])


if __name__ == "__main__":
    unittest.main()
