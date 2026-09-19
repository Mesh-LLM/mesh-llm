"""Admission, runner preflight, and failure-path contracts for nightly replay."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/agentic-replay-nightly.yml"


class NightlyWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.workflow = yaml.safe_load(WORKFLOW.read_text())
        self.job = self.workflow["jobs"]["replay"]
        self.steps = self.job["steps"]

    def step(self, name):
        return next(step for step in self.steps if step.get("name") == name)

    def test_daily_and_manual_admission_stays_on_canonical_main(self):
        # Evaluate the small closed admission expression against its event matrix.
        expression = self.job["if"]
        for repo in ("Mesh-LLM/mesh-llm", "fork/mesh-llm"):
            for ref in ("refs/heads/main", "refs/heads/feature"):
                for event in ("schedule", "workflow_dispatch", "pull_request", "push"):
                    with self.subTest(repo=repo, ref=ref, event=event):
                        expr = expression
                        for key, value in (("repository", repo), ("ref", ref), ("event_name", event)):
                            expr = expr.replace(f"github.{key}", repr(value))
                        expr = " ".join(expr.replace("&&", "and").replace("||", "or").split())
                        actual = eval(expr, {"__builtins__": {}}, {})
                        expected = repo == "Mesh-LLM/mesh-llm" and ref == "refs/heads/main" and event in ("schedule", "workflow_dispatch")
                        self.assertEqual(actual, expected)
        self.assertNotIn("MESH_AGENTIC_REPLAY_NIGHTLY_ENABLED", WORKFLOW.read_text())
        self.assertFalse(self.workflow["concurrency"]["cancel-in-progress"])

    def test_native_runner_guard_executes_before_checkout(self):
        preflight = self.steps[0]
        self.assertEqual(preflight["name"], "Verify pinned replay runner")
        self.assertTrue(self.steps[1]["uses"].startswith("actions/checkout@"))
        for arch, runner, expected in (("arm64", "micstudio", 0), ("x86_64", "micstudio", 1), ("arm64", "studio54", 1)):
            with self.subTest(arch=arch, runner=runner), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                for name, body in (("uname", f"echo {arch}"), ("git", "exit 0"), ("xcrun", "exit 0")):
                    command = root / name
                    command.write_text("#!/bin/sh\n" + body + "\n")
                    command.chmod(0o755)
                env = dict(os.environ, PATH=f"{root}:{os.environ['PATH']}", RUNNER_NAME=runner, EXPECTED_REPLAY_RUNNER_NAME="micstudio")
                result = subprocess.run(["bash", "-c", preflight["run"]], env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, expected, result.stderr)

    def test_online_download_does_not_write_shared_model_cache(self):
        inputs = self.step("Verify pinned replay inputs")["run"]
        self.assertIn('--cache-dir "$RUNNER_TEMP/agentic-replay-dataset-cache"', inputs)
        self.assertEqual(self.job["env"]["HF_HUB_OFFLINE"], "1")
        self.assertEqual(self.job["env"]["HF_HUB_DISABLE_IMPLICIT_TOKEN"], "1")
        self.assertNotIn("env", self.step("Download cohort-matched history"))

    def test_repair_requires_explicit_regression_and_preserves_evidence(self):
        repair = self.step("Prepare repair PR artifact on regression (opencode loop)")
        self.assertIn("steps.history.outcome == 'failure'", repair["if"])
        self.assertIn("steps.history.outputs.repair_required == 'true'", repair["if"])
        self.assertIn("!cancelled()", repair["if"])
        history = self.step("Normalize history and gate on regression")
        self.assertIn("steps.benchmark.outcome == 'success'", history["if"])
        self.assertIn("steps.baseline.outcome == 'success'", history["if"])
        artifact = next(step for step in self.steps if step.get("uses", "").startswith("actions/upload-artifact@"))
        self.assertIn("cancelled()", artifact["if"])
        self.assertNotIn("!cancelled()", artifact["if"])
        self.assertIn("repair.log", repair["run"])
        replay = self.step("Replay each pinned model")
        self.assertGreater(self.job["timeout-minutes"], replay["timeout-minutes"] + repair["timeout-minutes"])


if __name__ == "__main__":
    unittest.main()
