"""Contract checks for executable standalone model pilot results."""

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

    def test_windows_cpu_pilots_use_pinned_models_and_composed_product(self):
        workflow = (SCRIPT.parents[2] / ".github/workflows/ci-skippy-product-slice.yml").read_text()
        windows = workflow.split("  windows_product:\n", 1)[1]
        self.assertIn("--product-dir skippy-product-input", windows)
        self.assertIn("model_artifact_id: smollm2-q8-inference", windows)
        self.assertIn("model_artifact_id: family-granite-hybrid", windows)
        self.assertIn("model_artifact_id: family-granite-moe", windows)
        self.assertIn("--suite dense-pilot", windows)
        self.assertIn("--suite recurrent-pilot", windows)
        self.assertIn("--suite moe-pilot", windows)
        self.assertIn("name: ci-skippy-model-pilot-windows-${{ matrix.runtime.architecture }}", windows)
        self.assertEqual(windows.count("if: ${{ matrix.runtime.backend == 'cpu' }}"), 10)


if __name__ == "__main__":
    unittest.main()
