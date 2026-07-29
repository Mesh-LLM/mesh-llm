from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
ACTION_DIR = ROOT / ".github" / "actions" / "capture-sccache-stats"
CAPTURE = ACTION_DIR / "capture.py"
SUMMARY = ROOT / "scripts" / "summarize-sccache-stats.py"
CONFIGURE_ACTION = (
    ROOT / ".github" / "actions" / "configure-sccache-gha" / "action.yml"
)
WORKFLOWS = {
    "pr-builds": ROOT / ".github" / "workflows" / "pr_builds.yml",
    "pr-quality": ROOT / ".github" / "workflows" / "pr_quality.yml",
    "main": ROOT / ".github" / "workflows" / "ci.yml",
}


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
    def run_capture(
        self,
        payload: dict[str, object],
        *,
        artifact_name: str = "sccache-test-1",
    ) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        fake_sccache = root / "sccache"
        fake_sccache.write_text(
            "#!/bin/sh\n"
            "if [ \"$#\" -eq 1 ] && [ \"$1\" = \"--show-stats\" ]; then\n"
            "  printf '%s\\n' 'Compile requests                     12'\n"
            "elif [ \"$#\" -eq 3 ] && [ \"$1\" = \"--show-stats\" ] "
            "&& [ \"$2\" = \"--stats-format\" ] && [ \"$3\" = \"json\" ]; then\n"
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
            },
            check=False,
            capture_output=True,
            text=True,
        )
        return result, stats_file, github_output

    def test_capture_emits_human_stats_and_machine_readable_counters(self) -> None:
        payload = valid_payload()
        result, stats_file, github_output = self.run_capture(payload)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Human-readable sccache statistics", result.stdout)
        self.assertIn("Compile requests                     12", result.stdout)
        self.assertEqual(json.loads(stats_file.read_text()), payload)
        outputs = github_output.read_text(encoding="utf-8")
        self.assertIn("compile_requests=12", outputs)
        self.assertIn("requests_executed=10", outputs)
        self.assertIn("cache_hits=6", outputs)
        self.assertIn("cache_misses=4", outputs)

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
        self.assertEqual(configure.count("await resetStatistics("), 6)

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

    def test_instrumented_workflows_use_unique_evidence_artifacts(self) -> None:
        expected_names = {
            "pr-builds": (
                "sccache-pr-linux-host-${{ github.run_attempt }}",
                "sccache-pr-linux-cpu-runtime-${{ github.run_attempt }}",
                "sccache-pr-rust-crate-tests-${{ matrix.batch.idx }}-${{ github.run_attempt }}",
                "sccache-pr-linux-tests-${{ matrix.group }}-${{ github.run_attempt }}",
            ),
            "pr-quality": (
                "sccache-pr-quality-clippy-${{ matrix.batch.idx }}-${{ github.run_attempt }}",
            ),
            "main": (
                "sccache-main-linux-host-${{ github.run_attempt }}",
                "sccache-main-linux-cpu-runtime-${{ github.run_attempt }}",
                "sccache-main-rust-crate-tests-${{ matrix.batch.idx }}-${{ github.run_attempt }}",
                "sccache-main-linux-tests-${{ matrix.group }}-${{ github.run_attempt }}",
            ),
        }

        for workflow_name, path in WORKFLOWS.items():
            workflow = path.read_text(encoding="utf-8")
            with self.subTest(workflow=workflow_name):
                self.assertNotIn("Show sccache stats", workflow)
                self.assertEqual(
                    workflow.count("uses: ./.github/actions/capture-sccache-stats"),
                    len(expected_names[workflow_name]),
                )
                for artifact_name in expected_names[workflow_name]:
                    self.assertIn(f"artifact_name: {artifact_name}", workflow)


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
