"""Guards on what this package redistributes.

Only eligibility-criteria passages from articles whose PMC licence is CC BY or
CC0 may be shipped, each with its attribution fields. Everything else is listed
in criteria_withheld.csv.gz as a hash, never as text.
"""

import csv
import gzip
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

csv.field_size_limit(10**9)
ROOT = Path(__file__).resolve().parents[1]
MANIFESTS = ROOT / "data" / "manifests"
OPEN_LICENSES = {"CC BY", "CC0"}
ATTRIBUTION_FIELDS = ("license", "license_url", "authors", "citation", "doi", "source_url")


def read_gz(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class ReleasedManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.criteria = read_gz(MANIFESTS / "criteria.csv.gz")
        cls.withheld = read_gz(MANIFESTS / "criteria_withheld.csv.gz")

    def test_released_criteria_are_openly_licensed_and_attributed(self):
        self.assertTrue(self.criteria)
        for row in self.criteria:
            self.assertEqual(row["source"], "pmc", row["review_id"])
            self.assertIn(row["license"], OPEN_LICENSES, row["review_id"])
            self.assertTrue(
                row["license_url"].startswith(
                    ("https://creativecommons.org/licenses/by/", "https://creativecommons.org/publicdomain/zero/")
                ),
                row["review_id"],
            )
            for field in ATTRIBUTION_FIELDS:
                self.assertTrue(row[field], f"{row['review_id']} lacks {field}")

    def test_withheld_reviews_ship_hashes_not_text(self):
        released = {row["review_id"] for row in self.criteria}
        self.assertFalse(released & {row["review_id"] for row in self.withheld})
        self.assertEqual(
            set(self.withheld[0]),
            {"review_id", "source", "reason", "license", "doi", "pmid", "text_sha256", "text_chars"},
        )
        for row in self.withheld:
            self.assertRegex(row["text_sha256"], r"^[0-9a-f]{64}$")

    def test_no_track_b_or_external_text_is_shipped(self):
        self.assertFalse((MANIFESTS / "external_criteria.csv.gz").exists())
        self.assertFalse(any(row["source"] == "trackb" for row in self.criteria))


class ExternalCriteriaBuilderTests(unittest.TestCase):
    def test_rebuild_checks_every_passage_against_the_recorded_hash(self):
        text = "We included randomised controlled trials of adults."
        review_data = f"[RTI] A review [BG] Background. [OBJ] Objective. [SEL] {text}"
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            archive = directory / "manual.zip"
            member = io.StringIO()
            writer = csv.DictWriter(member, fieldnames=["", "Review_data", "Title_Abstract", "included"])
            writer.writeheader()
            writer.writerow({"": "0", "Review_data": review_data, "Title_Abstract": "[TIT] t [ABS] a", "included": "0"})
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("CD000001_data_cleaned.csv", member.getvalue())
            hashes = directory / "hashes.csv.gz"
            with gzip.open(hashes, "wt", encoding="utf-8", newline="") as handle:
                out = csv.DictWriter(handle, fieldnames=["review_id", "source", "text_sha256"])
                out.writeheader()
                out.writerow(
                    {
                        "review_id": "CD000001",
                        "source": "external_cochrane",
                        "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                    }
                )
            output = directory / "external_criteria.csv"
            command = [
                sys.executable,
                str(ROOT / "src" / "build_external_criteria.py"),
                "--archive",
                str(archive),
                "--out",
                str(output),
                "--hashes",
                str(hashes),
                "--skip-archive-check",
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            with output.open(encoding="utf-8") as handle:
                self.assertEqual(list(csv.DictReader(handle)), [{"review_id": "CD000001", "selection_criteria": text}])

            with gzip.open(hashes, "wt", encoding="utf-8", newline="") as handle:
                out = csv.DictWriter(handle, fieldnames=["review_id", "source", "text_sha256"])
                out.writeheader()
                out.writerow({"review_id": "CD000001", "source": "external_cochrane", "text_sha256": "0" * 64})
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)

    def test_source_record_lists_the_manual_search_archive(self):
        record = json.loads((ROOT / "data" / "sources" / "chan_mendeley_v1.json").read_text(encoding="utf-8"))
        archive = record["manual_search_archive"]
        self.assertRegex(archive["sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(len(archive["members"]), 22)


if __name__ == "__main__":
    unittest.main()
