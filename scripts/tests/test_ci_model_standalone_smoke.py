"""Contract checks for executable standalone dense pilot results."""

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "skippy/scripts/ci-model-standalone-smoke.py"
SPEC = importlib.util.spec_from_file_location("ci_model_standalone_smoke", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CompletionEvidenceTests(unittest.TestCase):
    def test_accepts_positive_prefill_and_decode(self):
        response = {"model": "pinned", "choices": [{"message": {"content": "Hello"}}],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 2}}
        self.assertEqual(MODULE.completion_evidence(response, "pinned"),
                         {"prompt_tokens": 7, "completion_tokens": 2})

    def test_rejects_empty_or_foreign_completion(self):
        response = {"model": "pinned", "choices": [{"message": {"content": "Hello"}}],
                    "usage": {"prompt_tokens": 7, "completion_tokens": 0}}
        with self.assertRaisesRegex(ValueError, "positive prefill and decode"):
            MODULE.completion_evidence(response, "pinned")
        response["usage"]["completion_tokens"] = 2
        with self.assertRaisesRegex(ValueError, "model differs"):
            MODULE.completion_evidence(response, "other")


if __name__ == "__main__":
    unittest.main()
