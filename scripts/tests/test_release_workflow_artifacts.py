from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
RELEASE_WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"


class ReleaseWorkflowArtifactTests(unittest.TestCase):
    def test_inference_smoke_consumes_composed_product(self) -> None:
        workflow = RELEASE_WORKFLOW.read_text(encoding="utf-8")

        self.assertEqual(
            workflow.count("release-linux-inference-product"),
            2,
        )
        self.assertNotIn("release-linux-inference-binary", workflow)
        for required_path in (
            "smoke-input/mesh-llm",
            "smoke-input/host-imports.json",
            'smoke-input/native-runtimes/$runtime_name',
            "smoke-input/product-manifest.json",
        ):
            self.assertIn(required_path, workflow)

        self.assertIn(
            "python3 scripts/compose-product-bundle.py",
            workflow,
        )


if __name__ == "__main__":
    unittest.main()
