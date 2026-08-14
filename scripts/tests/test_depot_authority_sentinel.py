from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
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
        self.canary = (WORKFLOWS / "depot-canary.yml").read_text(
            encoding="utf-8"
        )
        self.selector = SELECTOR.read_text(encoding="utf-8")
        self.sentinel = self._job_block(self.workflow, "authority_sentinel")

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

    @staticmethod
    def _job_names(workflow: str) -> set[str]:
        jobs = workflow.split("\njobs:\n", 1)[1]
        return set(re.findall(r"^  ([A-Za-z0-9_-]+):", jobs, re.MULTILINE))

    @staticmethod
    def _step_names(job: str) -> list[str]:
        return re.findall(r"^      - name: (.+)$", job, re.MULTILINE)

    @staticmethod
    def _step_block(job: str, step_name: str) -> str:
        match = re.search(
            rf"^      - name: {re.escape(step_name)}\n(?P<body>.*?)(?=^      - name:|\Z)",
            job,
            flags=re.MULTILINE | re.DOTALL,
        )
        if match is None:
            raise AssertionError(f"missing step {step_name}")
        return match.group("body")

    @classmethod
    def _step_script(cls, job: str, step_name: str) -> str:
        run = re.search(
            r"^        run: \|\n(?P<script>.*)",
            cls._step_block(job, step_name),
            flags=re.MULTILINE | re.DOTALL,
        )
        if run is None:
            raise AssertionError(f"step {step_name} has no run script")
        return dedent(run.group("script"))

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
        return "set -euo pipefail\n" + self._step_script(
            self.sentinel, "Validate protected sentinel identity"
        )

    def _attestation_script(self) -> str:
        return "set -euo pipefail\n" + self._step_script(
            self.sentinel, "Attest provider-injected cache backend"
        )

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
        path: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        environment = {
            "PATH": path or os.environ.get("PATH", "/usr/bin:/bin"),
            "ACTIONS_CACHE_URL": cache_url,
            "ACTIONS_RESULTS_URL": results_url,
            "ACTIONS_RUNTIME_TOKEN": runtime_token,
        }
        if runtime_token == "":
            environment.pop("ACTIONS_RUNTIME_TOKEN")
        return subprocess.run(
            ["/bin/bash", "-c", self._attestation_script()],
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
        step_names = self._step_names(self.sentinel)
        positions = {name: index for index, name in enumerate(step_names)}
        self.assertLess(
            positions["Restore trusted authority marker"],
            positions["Replace with deterministic PR poison marker"],
        )
        self.assertLess(
            positions["Replace with deterministic PR poison marker"],
            positions["Save PR poison marker"],
        )
        self.assertLess(
            positions["Save PR poison marker"],
            positions["Require trusted seed isolation after poison publication"],
        )
        restore = self._step_block(self.sentinel, "Restore trusted authority marker")
        self.assertNotIn("lookup-only", restore)
        self.assertIn("uses: actions/cache/restore@caa296126883cff596d87d8935842f9db880ef25", restore)
        self.assertIn(
            "uses: actions/cache/save@caa296126883cff596d87d8935842f9db880ef25",
            self._step_block(self.sentinel, "Save PR poison marker"),
        )
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

        for endpoint in (
            "http://126.255.255.255:1234/cache",
            "http://128.0.0.0:1234/cache",
            "http://[::ffff:126.255.255.255]:1234/cache",
            "http://[::ffff:7e00:1]:1234/cache",
            "http://[2001:db8::1]:1234/cache",
        ):
            with self.subTest(non_loopback=endpoint):
                result = self._run_attestation(endpoint, valid_results)
                self.assertEqual(result.returncode, 0, result.stderr)

        invalid_cases = (
            ("https://cache.example.invalid:1234/cache", "scheme"),
            ("http://actions.githubusercontent.com:1234/cache", "github"),
            ("http://localhost:1234/cache", "loopback"),
            ("http://127.0.0.1:1234/cache", "loopback"),
            ("http://127.0.0.0:1234/cache", "loopback"),
            ("http://127.1.2.3:1234/cache", "loopback"),
            ("http://127.255.255.255:1234/cache", "loopback"),
            ("http://[::1]:1234/cache", "loopback"),
            ("http://[::ffff:127.0.0.1]:1234/cache", "loopback"),
            ("http://[::ffff:127.255.255.255]:1234/cache", "loopback"),
            ("http://[::ffff:7f00:1]:1234/cache", "loopback"),
            ("http://[0:0:0:0:0:ffff:127.0.0.1]:1234/cache", "loopback"),
            ("http://[0:0:0:0:0:ffff:7f00:1]:1234/cache", "loopback"),
            ("http://[0:0:0:0:0:0:0:1]:1234/cache", "loopback"),
            ("http://[0::1]:1234/cache", "loopback"),
            ("http://[0:0:0:0::1]:1234/cache", "loopback"),
            ("http://[0::ffff:127.0.0.1]:1234/cache", "loopback"),
            ("http://[0:0:0:0::ffff:127.0.0.1]:1234/cache", "loopback"),
            ("http://[not-an-ip]:1234/cache", "parser"),
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

        with tempfile.TemporaryDirectory() as bin_dir:
            tr_path = shutil.which("tr")
            self.assertIsNotNone(tr_path)
            os.symlink(tr_path, Path(bin_dir) / "tr")
            parser_missing = self._run_attestation(
                "http://[2001:db8::1]:1234/cache",
                valid_results,
                path=bin_dir,
            )
        self.assertNotEqual(parser_missing.returncode, 0)
        self.assertIn(
            "cache backend attestation failed (variable=ACTIONS_CACHE_URL reason=parser)",
            parser_missing.stderr,
        )
        self.assertNotIn("2001:db8::1", parser_missing.stderr)
        self.assertIn("sys.version_info < (3, 8)", self._attestation_script())

        with tempfile.TemporaryDirectory() as bin_dir:
            tr_path = shutil.which("tr")
            self.assertIsNotNone(tr_path)
            os.symlink(tr_path, Path(bin_dir) / "tr")
            fake_python = Path(bin_dir) / "python3"
            fake_python.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
            fake_python.chmod(0o755)
            classifier_failed = self._run_attestation(
                "http://[2001:db8::1]:1234/cache",
                valid_results,
                path=bin_dir,
            )
        self.assertNotEqual(classifier_failed.returncode, 0)
        self.assertIn(
            "cache backend attestation failed (variable=ACTIONS_CACHE_URL reason=parser)",
            classifier_failed.stderr,
        )
        self.assertNotIn("2001:db8::1", classifier_failed.stderr)

    def test_authority_attestation_copies_are_identical(self) -> None:
        quality_script = self._attestation_script()
        seed_script = "set -euo pipefail\n" + self._step_script(
            self._job_block(self.canary, "seed_authority_marker"),
            "Attest provider-injected cache backend",
        )
        verify_script = "set -euo pipefail\n" + self._step_script(
            self._job_block(self.canary, "verify_pr_write"),
            "Attest provider-injected cache backend",
        )
        self.assertEqual(quality_script, seed_script)
        self.assertEqual(quality_script, verify_script)
        for required in (
            "is_loopback_authority()",
            "127\\.[0-9]{1,3}",
            "command -v python3",
            "import ipaddress",
            "sys.version_info < (3, 8)",
            "ipv4_mapped",
            "else 3",
            "python_status == 3",
            "ACTIONS_CACHE_URL",
            "ACTIONS_RESULTS_URL",
            "ACTIONS_RUNTIME_TOKEN",
        ):
            self.assertIn(required, quality_script)

    def test_five_pr_entrypoints_and_existing_build_shape_are_unchanged(self) -> None:
        expected = {
            "pr_quality.yml": ("quality", "Quality", "quality"),
            "pr_website.yml": ("website", "Website", "website"),
            "pr_linux.yml": ("Linux", "Linux", "linux"),
            "pr_macos.yml": ("macOS", "macOS", "macos"),
            "pr_windows.yml": ("Windows", "Windows", "windows"),
        }
        validation_entrypoints = {
            path.name
            for path in WORKFLOWS.glob("pr_*.yml")
            if "  pull_request:" in path.read_text(encoding="utf-8")
        }
        self.assertEqual(validation_entrypoints, set(expected))

        quality_lane = (WORKFLOWS / "ci-quality-lane.yml").read_text(
            encoding="utf-8"
        )
        self.assertEqual(
            self._job_names(quality_lane),
            {"quality", "runner_contract", "summary"},
        )
        self.assertIn(
            "uses: ./.github/workflows/ci-quality-slice.yml",
            self._job_block(quality_lane, "quality"),
        )
        self.assertIn(
            "uses: ./.github/workflows/ci-runner-contract-slice.yml",
            self._job_block(quality_lane, "runner_contract"),
        )
        summary = self._job_block(quality_lane, "summary")
        self.assertIn("needs: [quality, runner_contract]", summary)
        self.assertIn("runs-on: ubuntu-24.04", summary)

        for filename, (plan_name, lane_name, lane_id) in expected.items():
            workflow = (WORKFLOWS / filename).read_text(encoding="utf-8")
            with self.subTest(workflow=filename):
                self.assertEqual(
                    self._job_names(workflow),
                    {"plan", "lane", "required"},
                )
                plan = self._job_block(workflow, "plan")
                lane = self._job_block(workflow, "lane")
                required = self._job_block(workflow, "required")
                self.assertIn(f"name: Plan {plan_name}", plan)
                self.assertIn("runs-on: ubuntu-24.04", plan)
                self.assertIn(
                    f"uses: Mesh-LLM/mesh-llm/.github/workflows/ci-{lane_id}-lane.yml@main",
                    lane,
                )
                self.assertIn(f"name: {lane_name}", lane)
                self.assertIn(f"name: PR / {lane_name}", required)
                self.assertIn("needs: [plan, lane]", required)
                self.assertIn("runs-on: ubuntu-24.04", required)
                self.assertIn("permissions: {}", required)

        expected_slice_jobs = {
            "runner_policy": "runs-on: ubuntu-24.04",
            "quality_contracts": "runs-on: ${{ needs.runner_policy.outputs.runner_4 }}",
            "rust_fmt": "runs-on: ${{ needs.runner_policy.outputs.runner_4 }}",
            "rust_clippy": "runs-on: ${{ needs.runner_policy.outputs.runner_8 }}",
            "cli_docs_sync": "runs-on: ${{ needs.runner_policy.outputs.runner_4 }}",
            "authority_sentinel": "runs-on: ${{ needs.runner_policy.outputs.authority_sentinel_runner }}",
        }
        self.assertEqual(self._job_names(self.workflow), set(expected_slice_jobs))
        for job_name, runner_expression in expected_slice_jobs.items():
            with self.subTest(job=job_name):
                self.assertIn(runner_expression, self._job_block(self.workflow, job_name))

        for job_name in (
            "quality_contracts",
            "rust_fmt",
            "rust_clippy",
            "cli_docs_sync",
        ):
            self.assertNotIn("authority_sentinel", self._job_block(self.workflow, job_name))


if __name__ == "__main__":
    unittest.main()
