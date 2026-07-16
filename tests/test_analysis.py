"""Unit tests for shared analysis utilities."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from change_impact.analysis import cluster_bootstrap, extract_json_object, read_jsonl, write_csv


class AnalysisUtilitiesTest(unittest.TestCase):
    def test_extract_json_object_accepts_fences_and_surrounding_text(self) -> None:
        self.assertEqual(extract_json_object('```json\n{"score": 2}\n```'), {"score": 2})
        self.assertEqual(extract_json_object('result: {"score": 1} done'), {"score": 1})

    def test_jsonl_and_csv_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jsonl = root / "records.jsonl"
            jsonl.write_text('{"task_id": "a", "score": 1}\n', encoding="utf-8")
            rows = read_jsonl(jsonl)
            self.assertEqual(rows[0]["task_id"], "a")
            output = root / "records.csv"
            write_csv(output, rows)
            self.assertEqual(output.read_text(encoding="utf-8").splitlines()[0], "task_id,score")

    def test_cluster_bootstrap_is_deterministic(self) -> None:
        rows = [
            {"repo": "a", "score": 0.0},
            {"repo": "a", "score": 1.0},
            {"repo": "b", "score": 1.0},
        ]
        first = cluster_bootstrap(rows, "score", cluster_key="repo", replicates=100, seed=7)
        second = cluster_bootstrap(rows, "score", cluster_key="repo", replicates=100, seed=7)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first[0], 2 / 3)


if __name__ == "__main__":
    unittest.main()
