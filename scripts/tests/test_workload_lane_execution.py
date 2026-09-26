"""Exercise the real shell lane functions without loading models or building."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]


def shell_function(script: str, name: str) -> str:
    """Load one top-level function verbatim, excluding the script's entrypoint."""
    source = (ROOT / "scripts" / script).read_text(encoding="utf-8")
    start = source.index(f"{name}() {{\n")
    end = source.index("\n}\n", start) + 3
    return source[start:end]


class WorkloadLaneExecutionTests(unittest.TestCase):
    def test_summary_preserves_preflight_classes_without_claiming_certification(self) -> None:
        """Use planned classes for preflight; environment checks have no model class."""
        classes = ["causal_generation", "embedding", "rerank", "encoder_decoder",
                   "ocr", "speech_synthesis", "speech_recognition"]
        models = [{"family": f"family-{index}", "class": value}
                  for index, value in enumerate(classes)]
        outcome = {"name": "model-preflight", "status": "pass", "outcome": "pass", "exit_code": 0}
        rows = [{"family": model["family"], "outcomes": [outcome]} for model in models]
        rows.extend([
            {"family": "battery", "outcomes": [{**outcome, "name": "environment-preflight"}]},
            {"family": "family-0", "split_layer": 2, "outcomes": [{**outcome, "name": "chain"}]},
            {"family": "explicit", "workload_class": "embedding",
             "outcomes": [{**outcome, "name": "embedding-oracle"}]},
        ])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy, results, summary = (root / name for name in ("plan.json", "results.jsonl", "summary.tsv"))
            policy.write_text(json.dumps({"selected_models": models}))
            source = "\n".join(json.dumps(row) for row in rows)
            results.write_text(source)
            env = {**os.environ, "POLICY_PLAN_COPY": str(policy),
                   "RESULTS_JSONL": str(results), "SUMMARY_TSV": str(summary)}
            script = "set -euo pipefail\n" + shell_function("skippy-family-battery.sh", "write_lane_summary")
            result = subprocess.run(["bash", "-c", script + "\nwrite_lane_summary"], env=env,
                                    capture_output=True, text=True, check=False, timeout=15)
            self.assertEqual(0, result.returncode, result.stderr)
            actual = [line.split("\t") for line in summary.read_text().splitlines()]
            self.assertEqual(["family", "class", "split_layer", "lane", "status", "outcome", "exit_code"], actual[0])
            self.assertEqual(classes + ["", "causal_generation", "embedding"], [row[1] for row in actual[1:]])
            self.assertEqual(source, results.read_text(), "summary must not promote preflight rows to certifications")

    def run_lane(self, dry_run: bool) -> tuple[subprocess.CompletedProcess[str], list[dict]]:
        """Run an isolated battery function with deterministic producer and log fixtures."""
        with tempfile.TemporaryDirectory() as directory:
            env = {key: value for key, value in os.environ.items()
                   if not key.startswith("SKIPPY_WORKLOAD_ORACLE_")}
            env.update({"ROOT": str(ROOT), "CERT_DIR": directory,
                        "RESULTS_JSONL": str(Path(directory) / "results.jsonl"),
                        "DRY_RUN": str(int(dry_run))})
            script = "\n".join([
                "set -euo pipefail", "TOTAL=0; CERT_FAILURE_COUNT=0; FAILURES=()",
                shell_function("skippy-family-battery.sh", "slugify"),
                "cert_timeout_for_startup() { printf 1800; }",
                shell_function("skippy-family-battery.sh", "run_workload_certify"),
                'run_workload_certify first embedding /unused/model.gguf fixture rev 600 1024 embedding-smoke,embedding-oracle ""',
                'run_workload_certify second rerank /unused/model.gguf fixture rev 900 1024 rerank-smoke,rerank-oracle ""',
                'printf "counts=%s,%s\\n" "$TOTAL" "$CERT_FAILURE_COUNT"',
            ])
            result = subprocess.run(["bash", "-c", script], env=env, cwd=ROOT,
                                    capture_output=True, text=True, check=False, timeout=15)
            path = Path(env["RESULTS_JSONL"])
            # jq emits pretty-printed objects separated by whitespace.
            rows = []
            remaining = path.read_text() if path.exists() else ""
            decoder = json.JSONDecoder()
            while remaining.strip():
                row, end = decoder.raw_decode(remaining.lstrip())
                rows.append(row)
                remaining = remaining.lstrip()[end:]
            return result, rows

    def test_dry_run_prints_both_certified_rows_without_oracles(self) -> None:
        """Planning must remain usable before oracle executables are provisioned."""
        result, rows = self.run_lane(True)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("--startup-timeout-secs 600", result.stdout)
        self.assertIn("--startup-timeout-secs 900", result.stdout)
        self.assertEqual(2, result.stdout.count("--require-oracle"))
        self.assertIn("counts=2,0", result.stdout)
        self.assertEqual([], rows)

    def test_missing_oracle_records_both_failures_and_continues(self) -> None:
        """A missing prerequisite must retain failures for every selected family."""
        result, rows = self.run_lane(False)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertIn("counts=2,2", result.stdout)
        self.assertEqual(["first", "second"], [row["family"] for row in rows])
        for row in rows:
            self.assertEqual(1, row["exit_code"])
            self.assertEqual(2, len(row["outcomes"]))
            self.assertTrue(all(lane["status"] == "fail" for lane in row["outcomes"]))

    def test_readiness_uses_deadline_for_each_server_and_rejects_dead_process(self) -> None:
        """Give each server its planned budget while detecting an exited child promptly."""
        function = shell_function("skippy-workload-certify.sh", "wait_for_workload_server")
        for label in ("OpenAI server", "monolithic oracle server"):
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                log = Path(directory) / "server.log"
                log.write_text("fixture startup log\n")
                env = {**os.environ, "STARTUP_TIMEOUT_SECS": "1", "MODEL_CLASS": "embedding",
                       "MODEL_ID": "fixture", "SERVER_LOG": str(log), "LABEL": label}
                prefix = "set -euo pipefail\n" + function + "\n"
                start = time.monotonic()
                result = subprocess.run(["bash", "-c", prefix +
                    'curl() { return 1; }; wait_for_workload_server $$ 1 "$SERVER_LOG" "$LABEL"'],
                    env=env, capture_output=True, text=True, check=False, timeout=5)
                self.assertEqual(1, result.returncode)
                self.assertLess(time.monotonic() - start, 3)
                self.assertIn(f"{label} was not ready within 1 seconds", result.stderr)
                self.assertIn("fixture startup log", result.stderr)
                result = subprocess.run(["bash", "-c", prefix +
                    'wait_for_workload_server 99999999 1 "$SERVER_LOG" "$LABEL"'],
                    env=env, capture_output=True, text=True, check=False, timeout=5)
                self.assertEqual(1, result.returncode)
                self.assertIn(f"{label} exited early", result.stderr)
                result = subprocess.run(["bash", "-c", prefix +
                    'curl() { printf \'{"data":[{"id":"fixture"}]}\'; }; wait_for_workload_server $$ 1 "$SERVER_LOG" "$LABEL"'],
                    env=env, capture_output=True, text=True, check=False, timeout=5)
                self.assertEqual(0, result.returncode, result.stderr)

    def test_address_in_use_detection_is_narrow(self) -> None:
        function = shell_function("skippy-workload-certify.sh", "address_in_use_log")
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "server.log"
            prefix = "set -euo pipefail\n" + function + "\n"
            for message in ("Address already in use (os error 48)", "AddrInUse", "EADDRINUSE"):
                with self.subTest(message=message):
                    log.write_text(message + "\n")
                    result = subprocess.run(
                        ["bash", "-c", prefix + 'address_in_use_log "$LOG"'],
                        env={**os.environ, "LOG": str(log)}, capture_output=True,
                        text=True, check=False, timeout=5,
                    )
                    self.assertEqual(0, result.returncode, result.stderr)
            log.write_text("model initialization failed\n")
            result = subprocess.run(
                ["bash", "-c", prefix + 'address_in_use_log "$LOG"'],
                env={**os.environ, "LOG": str(log)}, capture_output=True,
                text=True, check=False, timeout=5,
            )
            self.assertEqual(1, result.returncode)

    def test_candidate_start_retries_only_address_in_use_with_fresh_port(self) -> None:
        address_check = shell_function("skippy-workload-certify.sh", "address_in_use_log")
        start_server = shell_function("skippy-workload-certify.sh", "start_candidate_server")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary_dir = root / "bin"
            binary_dir.mkdir()
            calls = root / "calls"
            server = binary_dir / "skippy-server"
            server.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\n' \"$*\" >> \"$CALLS\"\n"
                "if [[ $(wc -l < \"$CALLS\") -eq 1 ]]; then\n"
                "  echo 'Address already in use (os error 48)' >&2\n"
                "  exit 1\n"
                "fi\n"
            )
            server.chmod(0o755)
            env = {
                **os.environ,
                "BACKEND": "cpu",
                "CALLS": str(calls),
                "CANDIDATE_BIN_DIR": str(binary_dir),
                "CONFIG_PATH": str(root / "config.json"),
                "SERVER_LOG": str(root / "server.log"),
                "PORT": "",
                "PORT_START_ATTEMPTS": "3",
                "ROOT": str(ROOT),
            }
            script = "\n".join([
                "set -euo pipefail",
                address_check,
                "python3() { if [[ -f \"$CALLS\" ]]; then printf 41002; else printf 41001; fi; }",
                "wait_for_workload_server() { wait \"$1\"; }",
                start_server,
                "start_candidate_server",
            ])
            result = subprocess.run(
                ["bash", "-c", script], env=env, capture_output=True,
                text=True, check=False, timeout=5,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            invocations = calls.read_text().splitlines()
            self.assertEqual(2, len(invocations))
            self.assertIn("127.0.0.1:41001", invocations[0])
            self.assertIn("127.0.0.1:41002", invocations[1])
            self.assertTrue((root / "server.log.attempt-1").exists())
            self.assertTrue((root / "server.log").exists())

    def test_candidate_start_does_not_retry_other_startup_failures(self) -> None:
        address_check = shell_function("skippy-workload-certify.sh", "address_in_use_log")
        start_server = shell_function("skippy-workload-certify.sh", "start_candidate_server")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary_dir = root / "bin"
            binary_dir.mkdir()
            calls = root / "calls"
            server = binary_dir / "skippy-server"
            server.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\n' \"$*\" >> \"$CALLS\"\n"
                "echo 'model initialization failed' >&2\n"
                "exit 1\n"
            )
            server.chmod(0o755)
            env = {
                **os.environ,
                "BACKEND": "cpu",
                "CALLS": str(calls),
                "CANDIDATE_BIN_DIR": str(binary_dir),
                "CONFIG_PATH": str(root / "config.json"),
                "SERVER_LOG": str(root / "server.log"),
                "PORT": "",
                "PORT_START_ATTEMPTS": "3",
                "ROOT": str(ROOT),
            }
            script = "\n".join([
                "set -euo pipefail",
                address_check,
                "python3() { printf 41001; }",
                "wait_for_workload_server() { wait \"$1\"; }",
                start_server,
                "start_candidate_server",
            ])
            result = subprocess.run(
                ["bash", "-c", script], env=env, capture_output=True,
                text=True, check=False, timeout=5,
            )
            self.assertEqual(1, result.returncode)
            self.assertEqual(1, len(calls.read_text().splitlines()))
            self.assertIn("model initialization failed", (root / "server.log").read_text())


if __name__ == "__main__":
    unittest.main()
