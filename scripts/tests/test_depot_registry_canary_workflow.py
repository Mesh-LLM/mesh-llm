import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "depot-registry-canary.yml"


class DepotRegistryCanaryWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = WORKFLOW.read_text(encoding="utf-8")

    def test_canary_is_manual_and_main_only(self) -> None:
        self.assertIn("workflow_dispatch:", self.workflow)
        self.assertNotIn("pull_request:", self.workflow)
        self.assertNotIn("push:", self.workflow)
        self.assertIn('"refs/heads/main"', self.workflow)

    def test_registry_auth_is_native_and_job_scoped(self) -> None:
        self.assertIn("runs-on: depot-ubuntu-24.04", self.workflow)
        self.assertIn('"${DEPOT_ORG_ID:-}" != "1ntz5vlngn"', self.workflow)
        self.assertIn(
            "registry canaries require a pre-authenticated Mesh-LLM Depot runner",
            self.workflow,
        )
        self.assertNotIn("id-token: write", self.workflow)
        self.assertNotIn("depot pull-token", self.workflow)
        self.assertNotIn("docker login", self.workflow)
        self.assertNotIn("DEPOT_PROJECT_ID", self.workflow)
        self.assertNotIn("secrets.", self.workflow)
        self.assertNotIn("DEPOT_REGISTRY_PULL_TOKEN", self.workflow)
        self.assertNotIn("printenv", self.workflow)

    def test_canary_uses_fresh_runner_samples_and_exact_digest(self) -> None:
        self.assertIn("source: [upstream, depot]", self.workflow)
        self.assertIn("sample: [1, 2, 3, 4, 5]", self.workflow)
        self.assertEqual(self.workflow.count("runs-on: depot-ubuntu-24.04"), 1)
        self.assertIn("upstream_image must be pinned by sha256 digest", self.workflow)
        self.assertIn("digest mismatch", self.workflow)

    def test_canary_uses_pinned_artifact_actions(self) -> None:
        self.assertIn("actions/upload-artifact@b7c566a772e6b6bfb58ed0dc250532a479d7789f", self.workflow)
        self.assertIn("actions/download-artifact@37930b1c2abaa49bbe596cd826c3c89aef350131", self.workflow)


if __name__ == "__main__":
    unittest.main()
