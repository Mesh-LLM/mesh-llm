from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]
ACTION_DIR = ROOT / ".github" / "actions" / "capture-sccache-stats"
CAPTURE = ACTION_DIR / "capture.py"
SUMMARY = ROOT / "scripts" / "summarize-sccache-stats.py"
CONFIGURE_ACTION = (
    ROOT / ".github" / "actions" / "configure-sccache-gha" / "action.yml"
)
WORKFLOWS = {
    "quality": ROOT / ".github" / "workflows" / "ci-quality-slice.yml",
    "rust-tests": ROOT / ".github" / "workflows" / "ci-rust-tests-slice.yml",
    "host": ROOT / ".github" / "workflows" / "ci-linux-host-slice.yml",
    "runtime": ROOT / ".github" / "workflows" / "ci-linux-runtime-slice.yml",
    "static-abi": ROOT / ".github" / "workflows" / "static-abi-artifact.yml",
    "main": ROOT / ".github" / "workflows" / "ci.yml",
    "swift-sdk": ROOT / ".github" / "workflows" / "swift-sdk-artifact.yml",
}
INSTRUMENTED = WORKFLOWS.keys() - {"main"}
HF_WORKFLOW = ROOT / ".github" / "workflows" / "hf-download-smoke.yml"
NATIVE_SDK_WORKFLOW = (
    ROOT / ".github" / "workflows" / "native-sdk-artifact.yml"
)


def valid_payload(*, compile_requests: int = 12) -> dict[str, object]:
    return {
        "stats": {
            "compile_requests": compile_requests,
            "requests_executed": 10,
            "compilations": 4,
            "cache_writes": 3,
            "cache_read_errors": 0,
            "cache_write_errors": 0,
            "cache_hits": {
                "counts": {"Rust": 6},
                "adv_counts": {},
            },
            "cache_misses": {
                "counts": {"Rust": 4},
                "adv_counts": {},
            },
            "cache_errors": {
                "counts": {},
                "adv_counts": {},
            },
        },
        "version": "test",
    }


class SccacheEvidenceTests(unittest.TestCase):
    def test_configure_callers_declare_native_cache_policy(self) -> None:
        policy = "${{ needs.runner_policy.outputs.allow_native_github_cache }}"
        effective_release = (
            "${{ ((matrix.target == 'x86_64-unknown-linux-gnu' && "
            "startsWith(needs.metadata.outputs.runner_8, 'depot-')) || "
            "(matrix.target == 'aarch64-unknown-linux-gnu' && "
            "startsWith(needs.metadata.outputs.runner_arm_8, 'depot-'))) "
            "&& 'false' || 'true' }}"
        )
        effective_release_runner_16 = (
            "${{ startsWith(needs.metadata.outputs.runner_16, 'depot-') "
            "&& 'false' || 'true' }}"
        )
        expected = {
            ("ci-linux-host-slice.yml", "linux_host"): policy,
            ("ci-linux-runtime-slice.yml", "linux_runtime"): policy,
            ("ci-quality-slice.yml", "rust_clippy"): policy,
            ("ci-rust-tests-slice.yml", "rust_tests"): policy,
            ("ci-windows-host-slice.yml", "windows_host"): policy,
            ("ci-windows-runtime-slice.yml", "windows_runtime"): policy,
            ("hf-download-smoke.yml", "hf_download_smoke"): "true",
            ("native-sdk-artifact.yml", "linux_native_sdk_artifact"): policy,
            ("native-sdk-artifact.yml", "macos_native_sdk_artifact"): policy,
            ("node-sdk-addon-artifact.yml", "linux_addon"): "true",
            ("node-sdk-addon-artifact.yml", "macos_addon"): "true",
            ("node-sdk-addon-artifact.yml", "windows_addon"): "true",
            ("release.yml", "build"): "true",
            ("release.yml", "build_native_runtime"): effective_release,
            ("release.yml", "build_native_runtime_linux_aarch64_cuda"): "true",
            ("release.yml", "build_native_runtime_linux_x86_64_cuda"): "true",
            ("release.yml", "build_native_runtime_linux_x86_64_rocm"): effective_release_runner_16,
            ("release.yml", "build_native_runtime_linux_x86_64_vulkan"): effective_release_runner_16,
            ("static-abi-artifact.yml", "static_abi_artifact"): policy,
            ("swift-sdk-artifact.yml", "swift_sdk_artifact"): policy,
        }
        actual: dict[tuple[str, str], str] = {}
        for workflow_path in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
            workflow = yaml.safe_load(workflow_path.read_text(encoding="utf-8")) or {}
            for job_name, job in (workflow.get("jobs") or {}).items():
                for step in job.get("steps") or []:
                    if step.get("uses") != "./.github/actions/configure-sccache-gha":
                        continue
                    with_values = step.get("with") or {}
                    self.assertIn(
                        "allow_native_github_cache",
                        with_values,
                        f"{workflow_path.name}:{job_name} must set cache policy",
                    )
                    actual[(workflow_path.name, job_name)] = str(
                        with_values["allow_native_github_cache"],
                    )
        self.assertEqual(set(expected), set(actual))
        self.assertEqual(expected, actual)

    def run_capture(
        self,
        payload: dict[str, object],
        *,
        artifact_name: str = "sccache-test-1",
        sccache_error: str = "",
    ) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        fake_sccache = root / "sccache"
        fake_sccache.write_text(
            "#!/bin/sh\n"
            "if [ \"$#\" -eq 1 ] && [ \"$1\" = \"--show-stats\" ]; then\n"
            "  printf '%s\\n' "
            "'Cache location: https://human-stats-secret.example/private'\n"
            "elif [ \"$#\" -eq 3 ] && [ \"$1\" = \"--show-stats\" ] "
            "&& [ \"$2\" = \"--stats-format\" ] && [ \"$3\" = \"json\" ]; then\n"
            "  if [ -n \"$FAKE_SCCACHE_ERROR\" ]; then\n"
            "    printf '%s\\n' \"$FAKE_SCCACHE_ERROR\" >&2\n"
            "    exit 23\n"
            "  fi\n"
            "  printf '%s\\n' \"$FAKE_SCCACHE_JSON\"\n"
            "else\n"
            "  exit 2\n"
            "fi\n",
            encoding="utf-8",
        )
        fake_sccache.chmod(
            fake_sccache.stat().st_mode | stat.S_IXUSR,
        )
        stats_file = root / "evidence" / "sccache-stats.json"
        github_output = root / "github-output"
        result = subprocess.run(
            [
                sys.executable,
                str(CAPTURE),
                "--artifact-name",
                artifact_name,
                "--output",
                str(stats_file),
                "--github-output",
                str(github_output),
            ],
            env={
                **os.environ,
                "PATH": f"{root}{os.pathsep}{os.environ['PATH']}",
                "FAKE_SCCACHE_JSON": json.dumps(payload),
                "FAKE_SCCACHE_ERROR": sccache_error,
            },
            check=False,
            capture_output=True,
            text=True,
        )
        return result, stats_file, github_output

    def test_capture_writes_only_sanitized_machine_readable_counters(self) -> None:
        payload = valid_payload()
        result, stats_file, github_output = self.run_capture(payload)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn("Human-readable sccache statistics", result.stdout)
        self.assertNotIn("human-stats-secret", result.stdout)
        self.assertEqual(
            json.loads(stats_file.read_text()),
            {
                "schema": "mesh-llm.sccache-stats",
                "schema_version": 1,
                "stats": {
                    "compile_requests": 12,
                    "requests_executed": 10,
                    "compilations": 4,
                    "cache_writes": 3,
                    "cache_read_errors": 0,
                    "cache_write_errors": 0,
                    "cache_hits": {"counts": {"total": 6}},
                    "cache_misses": {"counts": {"total": 4}},
                    "cache_errors": {"counts": {"total": 0}},
                },
            },
        )
        outputs = github_output.read_text(encoding="utf-8")
        self.assertIn("compile_requests=12", outputs)
        self.assertIn("requests_executed=10", outputs)
        self.assertIn("cache_hits=6", outputs)
        self.assertIn("cache_misses=4", outputs)

    def test_raw_secrets_urls_and_paths_cannot_reach_logs_or_evidence(self) -> None:
        payload = valid_payload()
        stats = payload["stats"]
        self.assertIsInstance(stats, dict)
        cache_hits = stats["cache_hits"]
        self.assertIsInstance(cache_hits, dict)
        cache_hits["counts"] = {
            "/Users/private/cache/path?token=count-key-secret": 6,
        }
        stats["not_cached"] = {
            "/home/runner/private-source": 1,
            "https://stats-secret.example/cache": 2,
        }
        payload.update(
            {
                "cache_location": (
                    "WebDAV: https://cache-user:location-secret@cache.example"
                ),
                "basedirs": ["/home/runner/work/private-repository"],
                "version": "raw-version-secret",
                "url": "https://payload-secret.example/cache",
                "absolute_path": "/Users/private/sccache",
            },
        )

        result, stats_file, github_output = self.run_capture(payload)

        self.assertEqual(result.returncode, 0, result.stderr)
        evidence = json.loads(stats_file.read_text(encoding="utf-8"))
        self.assertEqual(evidence["stats"]["cache_hits"], {"counts": {"total": 6}})
        exposed_surface = "\n".join(
            (
                result.stdout,
                result.stderr,
                stats_file.read_text(encoding="utf-8"),
                github_output.read_text(encoding="utf-8"),
            ),
        )
        for forbidden in (
            "human-stats-secret",
            "count-key-secret",
            "private-source",
            "stats-secret",
            "cache_location",
            "basedirs",
            "location-secret",
            "raw-version-secret",
            "payload-secret",
            "/Users/private",
            "/home/runner/work/private-repository",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, exposed_surface)

    def test_sccache_error_output_cannot_leak_into_the_job_log(self) -> None:
        secret = "https://cache-user:stderr-secret@cache.example/private"
        result, stats_file, _ = self.run_capture(
            valid_payload(),
            sccache_error=secret,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(stats_file.exists())
        self.assertNotIn(secret, result.stdout)
        self.assertNotIn(secret, result.stderr)
        self.assertIn("exit code 23", result.stderr)

    def test_zero_compile_requests_warns_but_remains_valid_evidence(self) -> None:
        result, stats_file, _ = self.run_capture(
            valid_payload(compile_requests=0),
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(stats_file.is_file())
        self.assertIn("::warning title=sccache reported zero compile requests", result.stdout)

    def test_missing_or_invalid_counter_rejects_evidence(self) -> None:
        payload = valid_payload()
        stats = payload["stats"]
        self.assertIsInstance(stats, dict)
        del stats["cache_misses"]

        result, stats_file, _ = self.run_capture(payload)

        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(stats_file.exists())
        self.assertIn("stats.cache_misses must be an object", result.stderr)

    def test_artifact_name_cannot_escape_the_evidence_namespace(self) -> None:
        result, stats_file, _ = self.run_capture(
            valid_payload(),
            artifact_name="../sccache-test",
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(stats_file.exists())
        self.assertIn("artifact name must contain only", result.stderr)

    def test_composite_action_uploads_fourteen_day_json_evidence(self) -> None:
        action = (ACTION_DIR / "action.yml").read_text(encoding="utf-8")
        capture = CAPTURE.read_text(encoding="utf-8")

        self.assertIn("artifact_name:", action)
        self.assertIn("actions/upload-artifact@b7c566a772e6b6bfb58ed0dc250532a479d7789f", action)
        self.assertIn("retention-days: 14", action)
        self.assertIn("if-no-files-found: error", action)
        self.assertIn('"--show-stats", "--stats-format", "json"', capture)
        self.assertIn("REQUIRED_COUNTERS", capture)
        self.assertIn("REQUIRED_COUNT_MAPS", capture)

    def test_configure_action_resets_each_successful_server_route(self) -> None:
        configure = CONFIGURE_ACTION.read_text(encoding="utf-8")

        self.assertIn("['--zero-stats']", configure)
        self.assertEqual(configure.count("await resetStatistics("), 7)

    def test_remote_multilevel_writes_finish_before_ephemeral_job_exit(self) -> None:
        configure = CONFIGURE_ACTION.read_text(encoding="utf-8")

        self.assertEqual(
            configure.count(
                "core.exportVariable("
                "'SCCACHE_MULTILEVEL_WRITE_ERROR_POLICY', 'all'"
                ")",
            ),
            2,
        )
        self.assertNotIn(
            "'SCCACHE_MULTILEVEL_WRITE_ERROR_POLICY', 'ignore'",
            configure,
        )

    def test_pull_request_sccache_is_disk_only(self) -> None:
        configure = CONFIGURE_ACTION.read_text(encoding="utf-8")

        self.assertIn("DISPATCH_ORIGINAL_EVENT_NAME", configure)
        self.assertIn("effectiveEventName === 'pull_request'", configure)
        self.assertIn("effectiveEventName === 'pull_request_target'", configure)
        self.assertIn(
            "core.exportVariable('SCCACHE_GHA_RW_MODE', ghaRemoteMode)",
            configure,
        )
        self.assertIn("if (ghaRemoteMode === 'READ_ONLY')", configure)
        self.assertIn(
            "Pull-request trust context detected; using baked sccache "
            "with job-local disk only.",
            configure,
        )
        for path in (*WORKFLOWS.values(), HF_WORKFLOW, NATIVE_SDK_WORKFLOW):
            workflow = path.read_text(encoding="utf-8")
            if "uses: ./.github/actions/configure-sccache-gha" in workflow:
                self.assertNotIn("SCCACHE_WEBDAV_RW_MODE", workflow)
        self.assertNotIn("SCCACHE_WEBDAV_RW_MODE", configure)

    def test_pull_request_direct_sccache_users_are_reconfigured(self) -> None:
        workflows = (*WORKFLOWS.values(), HF_WORKFLOW, NATIVE_SDK_WORKFLOW)

        for path in workflows:
            lines = path.read_text(encoding="utf-8").splitlines()
            direct_users = [
                index
                for index, line in enumerate(lines)
                if "uses: mozilla-actions/sccache-action@" in line
            ]
            for index in direct_users:
                with self.subTest(workflow=path.name, line=index + 1):
                    next_step = next(
                        line.strip()
                        for line in lines[index + 1 :]
                        if line.strip()
                    )
                    self.assertEqual(
                        next_step,
                        "- uses: ./.github/actions/configure-sccache-gha",
                    )

    def test_swift_uses_trusted_main_seeded_dependency_cache(self) -> None:
        swift = WORKFLOWS["swift-sdk"].read_text(encoding="utf-8")

        self.assertIn(
            "uses: Swatinem/rust-cache@"
            "e18b497796c12c097a38f9edb9d0641fb99eee32",
            swift,
        )
        self.assertIn("shared-key: swift-sdk", swift)
        self.assertIn("key: ${{ steps.native_toolchain.outputs.epoch }}", swift)
        self.assertIn('add-job-id-key: "false"', swift)
        self.assertIn(
            "save-if: ${{ github.event_name == 'push' "
            "&& github.ref == 'refs/heads/main' }}",
            swift,
        )

    def test_instrumented_workflows_use_unique_evidence_artifacts(self) -> None:
        for workflow_name in INSTRUMENTED:
            path = WORKFLOWS[workflow_name]
            workflow = path.read_text(encoding="utf-8")
            with self.subTest(workflow=workflow_name):
                self.assertNotIn("Show sccache stats", workflow)
                self.assertIn(
                    "uses: ./.github/actions/capture-sccache-stats",
                    workflow,
                )
                names = [
                    line.split("artifact_name:", 1)[1].strip()
                    for line in workflow.splitlines()
                    if "artifact_name:" in line
                    and "sccache-" in line
                ]
                self.assertEqual(
                    workflow.count("uses: ./.github/actions/capture-sccache-stats"),
                    len(names),
                )
                self.assertEqual(len(names), len(set(names)))
                for artifact_name in names:
                    self.assertIn("${{ github.run_attempt }}", artifact_name)


class SccacheStatsSummaryTests(unittest.TestCase):
    def write_evidence(
        self,
        path: Path,
        *,
        hits: int,
        misses: int,
        advanced_hits: int = 0,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "stats": {
                        "cache_hits": {
                            "counts": {"Rust": hits},
                            "adv_counts": {"Rust": advanced_hits},
                        },
                        "cache_misses": {
                            "counts": {"Rust": misses},
                            "adv_counts": {},
                        },
                    },
                },
            ),
            encoding="utf-8",
        )

    def run_summary(
        self,
        evidence: Path,
        *,
        minimum: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        command = [
            sys.executable,
            str(SUMMARY),
            "--format",
            "json",
        ]
        if minimum is not None:
            command.extend(["--minimum-hit-rate", minimum])
        command.append(str(evidence))
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )

    def test_offline_summary_aggregates_counts_without_advanced_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary)
            self.write_evidence(
                evidence / "job-a" / "sccache-stats.json",
                hits=60,
                misses=10,
                advanced_hits=60,
            )
            self.write_evidence(
                evidence / "job-b" / "sccache-stats-warm.json",
                hits=20,
                misses=10,
                advanced_hits=20,
            )

            result = self.run_summary(evidence, minimum="0.80")

        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads(result.stdout)
        self.assertEqual(summary["file_count"], 2)
        self.assertEqual(summary["cache_hits"], 80)
        self.assertEqual(summary["cache_misses"], 20)
        self.assertEqual(summary["hit_rate"], 0.8)
        self.assertTrue(summary["passed"])

    def test_offline_summary_fails_a_missed_hit_rate_gate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary) / "sccache-stats.json"
            self.write_evidence(evidence, hits=79, misses=21)

            result = self.run_summary(evidence, minimum="0.80")

        self.assertEqual(result.returncode, 1)
        summary = json.loads(result.stdout)
        self.assertEqual(summary["hit_rate"], 0.79)
        self.assertFalse(summary["passed"])

    def test_offline_summary_rejects_invalid_count_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary) / "sccache-stats.json"
            self.write_evidence(evidence, hits=-1, misses=1)

            result = self.run_summary(evidence)

        self.assertEqual(result.returncode, 1)
        self.assertIn("negative counter", result.stderr)


if __name__ == "__main__":
    unittest.main()
