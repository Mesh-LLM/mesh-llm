import unittest
import os
from pathlib import Path
import subprocess
import tempfile
from textwrap import dedent


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

    def test_pr_audit_uses_portable_endpoint_and_docker_config_checks(self) -> None:
        action = (
            ROOT
            / ".github"
            / "actions"
            / "audit-depot-pr-isolation"
            / "action.yml"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'endpoint_lower="$(printf \'%s\' "$endpoint" | tr '
            "'[:upper:]' '[:lower:]')\"",
            action,
        )
        self.assertIn(
            'docker_config="${DOCKER_CONFIG:-${HOME:-}/.docker}/config.json"',
            action,
        )
        self.assertIn("grep -Eiq 'depot\\.dev'", action)

    def test_endpoint_and_docker_auth_probes_fail_closed(self) -> None:
        action = (
            ROOT
            / ".github"
            / "actions"
            / "audit-depot-pr-isolation"
            / "action.yml"
        ).read_text(encoding="utf-8")
        action_run = action.split("      run: |\n", maxsplit=1)[1]
        action_script = dedent(action_run)
        canary_start = self.workflow.index(
            "          for endpoint_name in ACTIONS_CACHE_URL "
            "ACTIONS_RESULTS_URL; do",
        )
        canary_end = self.workflow.index(
            "          {\n",
            canary_start,
        )
        canary_script = "set -euo pipefail\n" + dedent(
            self.workflow[canary_start:canary_end],
        )

        def run_probe(
            script: str,
            cache_url: str,
            results_url: str,
            docker_auth_config: str | None = None,
            docker_config_content: str | None = None,
        ) -> subprocess.CompletedProcess[str]:
            with (
                tempfile.TemporaryDirectory() as home,
                tempfile.TemporaryDirectory() as docker_config,
            ):
                if docker_config_content is not None:
                    (Path(docker_config) / "config.json").write_text(
                        docker_config_content,
                        encoding="utf-8",
                    )
                environment = {
                    "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                    "HOME": home,
                    "DOCKER_CONFIG": docker_config,
                    "GITHUB_EVENT_NAME": "pull_request",
                    "INPUT_ORIGINAL_EVENT_NAME": "pull_request",
                    "ACTIONS_CACHE_URL": cache_url,
                    "ACTIONS_RESULTS_URL": results_url,
                }
                if docker_auth_config is not None:
                    environment["DOCKER_AUTH_CONFIG"] = docker_auth_config
                return subprocess.run(
                    ["bash", "-c", script],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=environment,
                )

        valid_endpoints = (
            (
                "https://actions.githubusercontent.com/cache",
                "https://results-receiver.actions.githubusercontent.com/results",
            ),
            (
                "https://actions.githubusercontent.com:443/cache",
                "https://results-receiver.actions.githubusercontent.com:8443/results?x=1#ok",
            ),
            (
                "HTTPS://ACTIONS.GITHUBUSERCONTENT.COM?cache=1",
                "https://results-receiver.actions.githubusercontent.com#results",
            ),
        )
        invalid_endpoints = (
            (
                "https://actions.githubusercontent.com:443@attacker.example/",
                valid_endpoints[0][1],
            ),
            (
                "http://actions.githubusercontent.com/cache",
                valid_endpoints[0][1],
            ),
            (
                "https://actions.githubusercontent.com.evil/cache",
                valid_endpoints[0][1],
            ),
            ("", valid_endpoints[0][1]),
        )
        for script_name, script in (
            ("audit action", action_script),
            ("Depot canary", canary_script),
        ):
            for endpoints in valid_endpoints:
                with self.subTest(script=script_name, endpoints=endpoints):
                    result = run_probe(script, *endpoints)
                    self.assertEqual(result.returncode, 0, result.stderr)
            for endpoints in invalid_endpoints:
                with self.subTest(script=script_name, endpoints=endpoints):
                    result = run_probe(script, *endpoints)
                    self.assertNotEqual(result.returncode, 0)

        depot_auth = '{"auths":{"REGISTRY.DEPOT.DEV":{"auth":"secret"}}}'
        depot_config = '{"auths":{"registry.depot.dev":{"auth":"secret"}}}'
        safe_config = '{"auths":{"ghcr.io":{"auth":"secret"}}}'
        for script_name, script in (
            ("audit action", action_script),
            ("Depot canary", canary_script),
        ):
            with self.subTest(script=script_name, auth="DOCKER_AUTH_CONFIG"):
                result = run_probe(
                    script,
                    *valid_endpoints[0],
                    docker_auth_config=depot_auth,
                )
                self.assertNotEqual(result.returncode, 0)
            with self.subTest(script=script_name, auth="config.json"):
                result = run_probe(
                    script,
                    *valid_endpoints[0],
                    docker_config_content=depot_config,
                )
                self.assertNotEqual(result.returncode, 0)
            with self.subTest(script=script_name, auth="safe config.json"):
                result = run_probe(
                    script,
                    *valid_endpoints[0],
                    docker_config_content=safe_config,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
            with self.subTest(script=script_name, auth="unset"):
                result = run_probe(script, *valid_endpoints[0])
                self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
