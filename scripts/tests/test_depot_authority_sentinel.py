from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import tempfile
import unittest
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
SELECTOR = ROOT / ".github" / "actions" / "select-ci-runners" / "action.yml"


class DepotAuthoritySentinelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = (WORKFLOWS / "ci-quality-slice.yml").read_text(
            encoding="utf-8"
        )
        self.selector = SELECTOR.read_text(encoding="utf-8")
        self.sentinel = self.workflow.split("  authority_sentinel:\n", 1)[1]

    @staticmethod
    def _job_block(workflow: str, job_name: str) -> str:
        jobs = workflow.split("\njobs:\n", 1)[1]
        match = re.search(
            rf"^  {re.escape(job_name)}:\n(?P<body>.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)",
            jobs,
            flags=re.MULTILINE | re.DOTALL,
        )
        if match is None:
            raise AssertionError(f"missing job {job_name}")
        return match.group("body")

    def _run_selector(
        self,
        *,
        event_name: str = "pull_request",
        ref: str = "refs/pull/42/merge",
        repository: str = "Mesh-LLM/mesh-llm",
        head_repository: str = "Mesh-LLM/mesh-llm",
        sentinel_ref: str = "refs/pull/42/merge",
        force_hosted: str = "false",
        pr_enabled: str = "false",
    ) -> tuple[subprocess.CompletedProcess[str], dict[str, str]]:
        run_block = self.selector.split("      run: |\n", 1)[1]
        script = "\n".join(
            line[8:] if line.startswith("        ") else line
            for line in run_block.splitlines()
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "github-output"
            result = subprocess.run(
                ["bash", "-c", script],
                cwd=ROOT,
                env={
                    **os.environ,
                    "GITHUB_OUTPUT": str(output),
                    "GITHUB_EVENT_NAME": event_name,
                    "GITHUB_REPOSITORY": repository,
                    "GITHUB_REF": ref,
                    "INPUT_EVENT_NAME": event_name,
                    "INPUT_ORIGINAL_EVENT_NAME": event_name,
                    "INPUT_REPOSITORY": repository,
                    "INPUT_HEAD_REPOSITORY": head_repository,
                    "INPUT_REF": ref,
                    "INPUT_DEPOT_MAIN_ENABLED": "false",
                    "INPUT_DEPOT_PR_ENABLED": pr_enabled,
                    "INPUT_PR_CANARY_REF": sentinel_ref,
                    "INPUT_FORCE_HOSTED": force_hosted,
                    "INPUT_MANUAL_USE_DEPOT": "false",
                    "DISPATCH_ORIGINAL_EVENT_NAME": "",
                },
                check=False,
                capture_output=True,
                text=True,
            )
            outputs = {}
            if output.exists():
                outputs = dict(
                    line.split("=", maxsplit=1)
                    for line in output.read_text(encoding="utf-8").splitlines()
                )
            return result, outputs

    def _validation_script(self) -> str:
        block = self.sentinel.split(
            "        run: |\n", 1
        )[1].split("\n      - name:", 1)[0]
        return "set -euo pipefail\n" + dedent(block)

    def _attestation_script(self) -> str:
        start = self.sentinel.index(
            "      - name: Attest provider-injected cache backend"
        )
        block = self.sentinel[start:].split("        run: |\n", 1)[1]
        block = block.split("\n      - name:", 1)[0]
        return "set -euo pipefail\n" + dedent(block)

    def _run_validation(
        self,
        sentinel_id: str,
        pr_number: str,
        configured_ref: str = "refs/pull/42/merge",
    ) -> tuple[subprocess.CompletedProcess[str], str]:
        with tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8") as output:
            result = subprocess.run(
                ["bash", "-c", self._validation_script()],
                cwd=ROOT,
                env={
                    "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
                    "SENTINEL_ID": sentinel_id,
                    "PR_NUMBER": pr_number,
                    "CONFIGURED_SENTINEL_REF": configured_ref,
                    "GITHUB_OUTPUT": output.name,
                },
                check=False,
                capture_output=True,
                text=True,
            )
            output.seek(0)
            return result, output.read()

    def _run_attestation(
        self,
        cache_url: str,
        results_url: str,
        *,
        runtime_token: str = "non-secret-runtime-token",
    ) -> subprocess.CompletedProcess[str]:
        environment = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "ACTIONS_CACHE_URL": cache_url,
            "ACTIONS_RESULTS_URL": results_url,
            "ACTIONS_RUNTIME_TOKEN": runtime_token,
        }
        if runtime_token == "":
            environment.pop("ACTIONS_RUNTIME_TOKEN")
        return subprocess.run(
            ["bash", "-c", self._attestation_script()],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_selector_truth_table_is_separate_from_normal_policy(self) -> None:
        cases = (
            ("exact same-repository PR ref", {}, True),
            (
                "missing sentinel ref",
                {"sentinel_ref": ""},
                False,
            ),
            (
                "different PR ref",
                {"sentinel_ref": "refs/pull/43/merge"},
                False,
            ),
            (
                "fork head",
                {"head_repository": "attacker/mesh-llm"},
                False,
            ),
            (
                "forced hosted",
                {"force_hosted": "true"},
                False,
            ),
            (
                "pull request target",
                {"event_name": "pull_request_target"},
                False,
            ),
            (
                "dispatch with PR source",
                {
                    "event_name": "workflow_dispatch",
                    "ref": "refs/heads/main",
                },
                False,
            ),
            (
                "non-merge ref",
                {"ref": "refs/pull/42/head"},
                False,
            ),
        )
        for name, kwargs, expected_depot in cases:
            with self.subTest(case=name):
                result, outputs = self._run_selector(**kwargs)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(outputs["depot_enabled"], str(expected_depot).lower())
                self.assertEqual(
                    outputs["runner"],
                    "depot-ubuntu-24.04" if expected_depot else "ubuntu-24.04",
                )

        malformed, outputs = self._run_selector(sentinel_ref="refs/heads/main")
        self.assertNotEqual(malformed.returncode, 0)
        self.assertIn("exact pull-request merge ref", malformed.stderr)
        self.assertEqual(outputs, {})

        global_gate, global_outputs = self._run_selector(
            sentinel_ref="", pr_enabled="true"
        )
        self.assertEqual(global_gate.returncode, 0, global_gate.stderr)
        self.assertEqual(global_outputs["depot_enabled"], "true")
        self.assertIn("github.ref == vars.DEPOT_PR_SENTINEL_REF", self.sentinel)

    def test_normal_quality_jobs_keep_the_existing_provider_output(self) -> None:
        policy = self.workflow.split("  runner_policy:\n", 1)[1].split(
            "\n  quality_contracts:", 1
        )[0]
        self.assertIn(
            "pr_canary_ref: ${{ vars.DEPOT_PR_CANARY_REF }}",
            policy,
        )
        self.assertIn(
            "pr_canary_ref: ${{ vars.DEPOT_PR_SENTINEL_REF }}",
            policy,
        )
        self.assertIn("depot_main_enabled: 'false'", policy)
        self.assertIn("depot_pr_enabled: 'false'", policy)
        self.assertIn("manual_use_depot: 'false'", policy)
        self.assertIn(
            "authority_sentinel_runner: ${{ steps.sentinel_policy.outputs.runner }}",
            self.workflow,
        )
        self.assertIn(
            "authority_sentinel_depot_enabled: ${{ steps.sentinel_policy.outputs.depot_enabled }}",
            self.workflow,
        )

        for job_name in (
            "quality_contracts",
            "rust_fmt",
            "rust_clippy",
            "cli_docs_sync",
        ):
            job = self._job_block(self.workflow, job_name)
            self.assertRegex(
                job,
                r"runs-on: \$\{\{ needs\.runner_policy\.outputs\.runner_(4|8) \}\}",
            )
            self.assertNotIn("sentinel_policy", job)

    def test_authority_job_is_protected_and_has_no_pr_code_or_audit(self) -> None:
        self.assertIn(
            "needs.runner_policy.outputs.authority_sentinel_depot_enabled == 'true'",
            self.sentinel,
        )
        self.assertIn("inputs.original_event_name == 'pull_request'", self.sentinel)
        self.assertIn("github.event_name == 'pull_request'", self.sentinel)
        self.assertIn("github.ref == vars.DEPOT_PR_SENTINEL_REF", self.sentinel)
        self.assertIn(
            "runs-on: ${{ needs.runner_policy.outputs.authority_sentinel_runner }}",
            self.sentinel,
        )
        self.assertIn("permissions: {}", self.sentinel)
        for forbidden in (
            "actions/checkout@",
            "source_sha",
            "secrets.",
            "audit-depot-pr-isolation@",
        ):
            self.assertNotIn(forbidden, self.sentinel)
        self.assertIn("SENTINEL_ID: ${{ vars.DEPOT_PR_SENTINEL_ID }}", self.sentinel)
        self.assertIn(
            "PR_NUMBER: ${{ github.event.pull_request.number }}",
            self.sentinel,
        )

    def test_sentinel_identity_and_key_grammar_are_bounded(self) -> None:
        self.assertIn(
            "if [[ ! \"$SENTINEL_ID\" =~ ^[0-9a-f]{32}$ ]]",
            self.sentinel,
        )
        self.assertIn(
            "if [[ ! \"$PR_NUMBER\" =~ ^[1-9][0-9]{0,8}$ ]]",
            self.sentinel,
        )
        self.assertIn(
            "seed_key=\"mesh-llm-depot-authority-seed-v1-${SENTINEL_ID}\"",
            self.sentinel,
        )
        self.assertIn(
            "poison_key=\"mesh-llm-depot-authority-pr-v1-${SENTINEL_ID}-pr-${PR_NUMBER}\"",
            self.sentinel,
        )
        self.assertNotIn("key: ${{ vars.", self.sentinel)
        self.assertNotIn("key: ${{ github.", self.sentinel)

        valid_id = "0123456789abcdef0123456789abcdef"
        valid_pr = "42"
        result, output = self._run_validation(valid_id, valid_pr)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            output,
            "sentinel_id="
            f"{valid_id}\n"
            "pr_number=42\n"
            "seed_key=mesh-llm-depot-authority-seed-v1-"
            f"{valid_id}\n"
            "poison_key=mesh-llm-depot-authority-pr-v1-"
            f"{valid_id}-pr-42\n",
        )
        for invalid_id in (
            "",
            "ABCDEF0123456789abcdef0123456789",
            "0" * 31,
            "0" * 33,
        ):
            with self.subTest(invalid_id=invalid_id):
                result, _ = self._run_validation(invalid_id, valid_pr)
                self.assertNotEqual(result.returncode, 0)
        for invalid_pr in ("", "0", "01", "+1", " 1", "1 ", "1" * 10):
            with self.subTest(invalid_pr=invalid_pr):
                result, _ = self._run_validation(valid_id, invalid_pr)
                self.assertNotEqual(result.returncode, 0)
        mismatch, _ = self._run_validation(
            valid_id,
            valid_pr,
            configured_ref="refs/pull/43/merge",
        )
        self.assertNotEqual(mismatch.returncode, 0)
        number_mismatch, _ = self._run_validation(
            valid_id,
            "43",
            configured_ref="refs/pull/42/merge",
        )
        self.assertNotEqual(number_mismatch.returncode, 0)

    def test_cache_probe_restores_then_publishes_before_gate(self) -> None:
        restore_index = self.sentinel.index("id: restore_seed")
        replace_index = self.sentinel.index("Replace with deterministic PR poison marker")
        save_index = self.sentinel.index("Save PR poison marker")
        gate_index = self.sentinel.index("Require trusted seed isolation after poison publication")
        self.assertLess(restore_index, replace_index)
        self.assertLess(replace_index, save_index)
        self.assertLess(save_index, gate_index)
        restore = self.sentinel[restore_index:replace_index]
        self.assertNotIn("lookup-only", restore)
        self.assertIn("uses: actions/cache/restore@caa296126883cff596d87d8935842f9db880ef25", restore)
        self.assertIn("uses: actions/cache/save@caa296126883cff596d87d8935842f9db880ef25", self.sentinel[save_index:gate_index])
        self.assertIn("rm -rf -- .depot-authority-sentinel", self.sentinel)
        self.assertIn(
            "Trusted seed was not readable; pending trusted-main verify-pr-write.",
            self.sentinel,
        )
        self.assertIn('if [[ "$CACHE_HIT" == "true" ]]', self.sentinel)
        self.assertNotIn("${endpoint,,}", self.sentinel)

    def test_authority_backend_attestation_is_value_free_and_fail_closed(self) -> None:
        valid_cache = "http://cache.example.invalid:1234/cache"
        valid_results = "http://results.example.invalid:5678/results"
        result = self._run_attestation(valid_cache, valid_results)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertEqual(result.stderr, "")

        invalid_cases = (
            ("https://cache.example.invalid:1234/cache", "scheme"),
            ("http://actions.githubusercontent.com:1234/cache", "github"),
            ("http://localhost:1234/cache", "loopback"),
            ("http://127.0.0.1:1234/cache", "loopback"),
            ("http://[::1]:1234/cache", "loopback"),
            ("http://user@cache.example.invalid:1234/cache", "userinfo"),
            ("http://cache.example.invalid/cache", "port"),
            ("http://cache.example.invalid:0/cache", "port"),
            ("http://cache.example.invalid:65536/cache", "port"),
            ("http://cache.example.invalid:1234", "path"),
            ("http://cache.example.invalid:1234/cache with-space", "whitespace"),
        )
        for endpoint, reason in invalid_cases:
            with self.subTest(endpoint=endpoint, reason=reason):
                result = self._run_attestation(endpoint, valid_results)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(
                    "cache backend attestation failed (variable=ACTIONS_CACHE_URL "
                    f"reason={reason})",
                    result.stderr,
                )
                self.assertNotIn(endpoint, result.stderr)
                self.assertNotIn("cache.example.invalid", result.stderr)
                self.assertNotIn("actions.githubusercontent.com", result.stderr)
                self.assertNotIn("non-secret-runtime-token", result.stderr)

        missing_token = self._run_attestation(
            valid_cache,
            valid_results,
            runtime_token="",
        )
        self.assertNotEqual(missing_token.returncode, 0)
        self.assertIn(
            "cache backend attestation failed (variable=ACTIONS_RUNTIME_TOKEN reason=missing)",
            missing_token.stderr,
        )
        self.assertNotIn("non-secret-runtime-token", missing_token.stderr)

    def test_five_pr_entrypoints_and_existing_build_shape_are_unchanged(self) -> None:
        expected = {
            "pr_quality.yml",
            "pr_website.yml",
            "pr_linux.yml",
            "pr_macos.yml",
            "pr_windows.yml",
        }
        validation_entrypoints = {
            path.name
            for path in WORKFLOWS.glob("pr_*.yml")
            if "  pull_request:" in path.read_text(encoding="utf-8")
        }
        self.assertEqual(validation_entrypoints, expected)

        for name in ("ci-quality-lane.yml", *sorted(expected)):
            current = (WORKFLOWS / name).read_text(encoding="utf-8")
            baseline_entry = subprocess.check_output(
                ["git", "show", f"HEAD:.github/workflows/{name}"],
                cwd=ROOT,
                text=True,
            )
            self.assertEqual(current, baseline_entry, name)

        baseline = subprocess.check_output(
            ["git", "show", "HEAD:.github/workflows/ci-quality-slice.yml"],
            cwd=ROOT,
            text=True,
        )
        current_jobs = self.workflow.split("\njobs:\n", 1)[1]
        baseline_jobs = baseline.split("\njobs:\n", 1)[1]
        self.assertEqual(
            {
                line.split(":", 1)[0].strip()
                for line in current_jobs.splitlines()
                if line.startswith("  ") and not line.startswith("    ")
            },
            {
                line.split(":", 1)[0].strip()
                for line in baseline_jobs.splitlines()
                if line.startswith("  ") and not line.startswith("    ")
            }
            | {"authority_sentinel"},
        )
        for job_name in (
            "quality_contracts",
            "rust_fmt",
            "rust_clippy",
            "cli_docs_sync",
        ):
            current_job = self._job_block(self.workflow, job_name)
            baseline_job = self._job_block(baseline, job_name)
            self.assertEqual(current_job.rstrip(), baseline_job.rstrip(), job_name)


if __name__ == "__main__":
    unittest.main()
