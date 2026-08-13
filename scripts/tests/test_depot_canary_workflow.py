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

    def test_canary_fails_closed_on_cache_and_registry_injection(self) -> None:
        self.assertIn("forbidden_names=", self.workflow)
        for name in (
            "DEPOT_CACHE_TOKEN",
            "DEPOT_CACHE_URL",
            "DEPOT_CACHE_API_URL",
            "DEPOT_TOKEN",
            "DEPOT_REGISTRY_HOST",
            "DEPOT_REGISTRY_URL",
            "DEPOT_REGISTRY_PULL_TOKEN",
            "DEPOT_REGISTRY_TOKEN",
            "SCCACHE_WEBDAV_ENDPOINT",
            "SCCACHE_WEBDAV_TOKEN",
            "SCCACHE_WEBDAV_USERNAME",
            "SCCACHE_WEBDAV_PASSWORD",
        ):
            with self.subTest(name=name):
                self.assertIn(name, self.workflow)
        self.assertIn('actions_cache_url="${ACTIONS_CACHE_URL:-}"', self.workflow)
        self.assertIn('actions_results_url="${ACTIONS_RESULTS_URL:-}"', self.workflow)
        self.assertIn("${actions_cache_url,,}", self.workflow)
        self.assertIn("${actions_results_url,,}", self.workflow)
        self.assertIn("transparently redirected to Depot", self.workflow)
        self.assertIn("ACTIONS_RESULTS_URL", self.workflow)
        self.assertIn("actions\\.githubusercontent\\.com", self.workflow)
        self.assertIn('docker_auth_config="${DOCKER_AUTH_CONFIG:-}"', self.workflow)
        self.assertIn("${docker_auth_config,,}", self.workflow)
        self.assertIn("Docker config contains Depot registry authentication", self.workflow)
        self.assertIn(
            "Depot cache credentials/endpoints injected | no",
            self.workflow,
        )
        self.assertIn(
            "Depot registry credentials/config injected | no",
            self.workflow,
        )
        self.assertNotIn("expect_cache_hit", self.workflow)
        self.assertNotIn("actions/cache@", self.workflow)
        self.assertNotIn("sccache cc", self.workflow)
        self.assertNotIn("mozilla-actions/sccache-action@", self.workflow)
        self.assertNotIn("Exercise authenticated Depot sccache", self.workflow)
        self.assertIn('"$ImageOS" "$ImageVersion"', self.workflow)
        self.assertNotIn("printenv", self.workflow)


if __name__ == "__main__":
    unittest.main()
