import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "depot-canary.yml"


class DepotCanaryWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = WORKFLOW.read_text(encoding="utf-8")

    def test_canary_has_no_code_or_credential_access(self) -> None:
        self.assertIn("permissions: {}", self.workflow)
        self.assertNotIn("actions/checkout", self.workflow)
        self.assertNotIn("secrets.", self.workflow)
        self.assertNotIn("pull_request", self.workflow)
        self.assertNotIn("push:", self.workflow)

    def test_canary_covers_measured_depot_sizes(self) -> None:
        for runner in (
            "depot-ubuntu-24.04",
            "depot-ubuntu-24.04-4",
            "depot-ubuntu-24.04-8",
            "depot-ubuntu-24.04-16",
            "depot-ubuntu-24.04-arm",
            "depot-ubuntu-24.04-arm-8",
        ):
            with self.subTest(runner=runner):
                self.assertIn(f"- {runner}", self.workflow)
        self.assertIn("expected_arch=aarch64", self.workflow)
        self.assertIn('actual_arch="$(uname -m)"', self.workflow)

    def test_canary_uses_a_pinned_cache_action_without_printing_tokens(
        self,
    ) -> None:
        self.assertIn(
            "actions/cache@caa296126883cff596d87d8935842f9db880ef25 "
            "# v5.1.0",
            self.workflow,
        )
        self.assertIn("${DEPOT_CACHE_TOKEN:-}", self.workflow)
        self.assertIn("${SCCACHE_WEBDAV_TOKEN:-}", self.workflow)
        self.assertIn("${SCCACHE_WEBDAV_ENDPOINT:-}", self.workflow)
        self.assertIn("Depot Cache authentication was not injected", self.workflow)
        self.assertIn("Depot runner image identity was not injected", self.workflow)
        self.assertIn('"$ImageOS" "$ImageVersion"', self.workflow)
        self.assertIn("mozilla-actions/sccache-action@", self.workflow)
        self.assertIn("sccache cc", self.workflow)
        self.assertIn(
            "sccache did not select the Depot WebDAV backend",
            self.workflow,
        )
        self.assertIn("expected a warm Depot sccache hit", self.workflow)
        self.assertIn("expect_cache_hit:", self.workflow)
        self.assertIn(
            '[[ "$EXPECT_CACHE_HIT" == "true" && "$CACHE_HIT" != "true" ]]',
            self.workflow,
        )
        self.assertNotIn("echo \"$DEPOT_CACHE_TOKEN\"", self.workflow)
        self.assertNotIn("printenv", self.workflow)


if __name__ == "__main__":
    unittest.main()
