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

    def test_shared_cache_allows_pinned_model_and_trajectory_downloads(self):
        inputs = self.step("Verify pinned replay inputs")["run"]
        self.assertNotIn("--cache-dir", inputs)
        self.assertIn('hf download "$repo" "$file" --revision "$revision"', inputs)
        self.assertIn('--repo-type dataset --revision "$dataset_revision"', inputs)
        self.assertEqual(self.job["env"]["HF_HUB_OFFLINE"], "0")
        toolchain = self.step("Verify runner toolchain")["run"]
        self.assertIn('export HF_HOME="$HF_CACHE"', toolchain)
        self.assertIn('export HF_HUB_CACHE="$HF_CACHE/hub"', toolchain)
        self.assertIn('! -w "$HF_CACHE/hub"', toolchain)
        self.assertEqual(self.job["env"]["HF_HUB_DISABLE_IMPLICIT_TOKEN"], "1")
        self.assertNotIn("env", self.step("Download cohort-matched history"))

    def test_repair_requires_explicit_regression_and_preserves_evidence(self):
        repair = self.step("Prepare repair PR artifact on regression (Goose)")
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
        self.assertIn("vars.LLAMA_CANARY_GOOSE_PROVIDER", repair["env"]["REPLAY_AGENT_PROVIDER"])
        self.assertIn("vars.LLAMA_CANARY_GOOSE_MODEL", repair["env"]["REPLAY_AGENT_MODEL"])

    def test_step_timeouts_respect_github_limit_and_leave_job_headroom(self):
        """Reject invalid step budgets independently of the larger job limit."""
        for job_id, job in self.workflow["jobs"].items():
            for step in job.get("steps", []):
                if "timeout-minutes" not in step:
                    continue
                with self.subTest(job=job_id, step=step.get("name", step.get("uses"))):
                    timeout = step["timeout-minutes"]
                    self.assertIs(type(timeout), int)
                    self.assertGreater(timeout, 0)
                    self.assertLessEqual(timeout, 360)
        repair = self.step("Prepare repair PR artifact on regression (Goose)")
        replay = self.step("Replay each pinned model")
        self.assertGreater(self.job["timeout-minutes"], replay["timeout-minutes"] + repair["timeout-minutes"])


if __name__ == "__main__":
    unittest.main()
