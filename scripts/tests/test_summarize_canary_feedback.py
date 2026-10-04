from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "summarize-canary-feedback.py"
SPEC = importlib.util.spec_from_file_location("canary_feedback_summary", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


class FeedbackSummaryTests(unittest.TestCase):
    def test_groups_failed_lanes_and_exposes_first_trace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "feedback.json").write_text(json.dumps({
                "candidate_failures": ["first", "second"],
            }))
            for family, note in (("first", "VIEW geometry exceeds source storage"),
                                 ("second", "RESHAPE changes element count")):
                family_dir = root / family
                family_dir.mkdir()
                (family_dir / "results.jsonl").write_text(json.dumps({
                    "family": family,
                    "outcomes": [{"name": "stage-replay", "status": "fail", "note": note}],
                }))
            rendered = SUMMARY.summary(root)
            self.assertIn("stage-replay (2)", rendered)
            self.assertIn("first: VIEW geometry exceeds source storage", rendered)
            self.assertIn("second: RESHAPE changes element count", rendered)


if __name__ == "__main__":
    unittest.main()
