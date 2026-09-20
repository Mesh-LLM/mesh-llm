"""Admission, runner preflight, and failure-path contracts for nightly replay."""

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import textwrap
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
        self.assertNotIn("hf download", inputs)
        self.assertIn('repo_type="dataset", token=False', inputs)
        self.assertEqual(self.job["env"]["HF_HUB_OFFLINE"], "0")
        toolchain = self.step("Verify runner toolchain")["run"]
        self.assertIn('export HF_HOME="$HF_CACHE"', toolchain)
        self.assertIn('export HF_HUB_CACHE="$HF_CACHE/hub"', toolchain)
        self.assertIn('! -w "$HF_CACHE/hub"', toolchain)
        self.assertEqual(self.job["env"]["HF_HUB_DISABLE_IMPLICIT_TOKEN"], "1")
        self.assertNotIn("env", self.step("Download cohort-matched history"))

    def test_input_verification_uses_api_paths_and_fails_closed(self):
        # Execute the actual workflow step with small pinned files. The fake
        # CLI reproduces the runner's decorated stdout, while the API returns
        # a real path and may emit unrelated status text of its own.
        cases = ("success", "model-corrupt", "dataset-corrupt", "model-download", "dataset-download")
        for failure in cases:
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "evals").mkdir()
                (root / "scripts").mkdir()
                shutil.copy(ROOT / "scripts/agentic-replay-params.py", root / "scripts")
                cache = root / "shared cache"
                cache.mkdir()
                matrix = json.loads((ROOT / "ci/agentic-replay-nightly/matrix.json").read_text())
                for model in matrix["models"]:
                    data = model["family"].encode()
                    (cache / model["file"]).write_bytes(data)
                    model["sha256"] = hashlib.sha256(data).hexdigest()
                replay = matrix["replay"]
                data = b"trajectory fixture"
                dataset = cache / replay["dataset_file"]
                dataset.write_bytes(data)
                replay["dataset_sha256"] = hashlib.sha256(data).hexdigest()
                (root / "matrix.json").write_text(json.dumps(matrix))
                canonical = {
                    "repo": replay["dataset"], "revision": replay["dataset_revision"],
                    "filename": replay["dataset_file"], "sha256": replay["dataset_sha256"],
                }
                (root / "evals/skippy-competitive-benchmark.json").write_text(
                    json.dumps({"thoughtworks": {"dataset": canonical}})
                )
                (root / "huggingface_hub.py").write_text(textwrap.dedent(r'''
                    import json, os
                    from pathlib import Path

                    def hf_hub_download(*, repo_id, filename, revision, token, repo_type="model"):
                        assert token is False
                        assert os.environ["HF_HUB_OFFLINE"] == "0"
                        matrix = json.loads(Path("matrix.json").read_text())
                        expected = matrix["models"] if repo_type == "model" else [{
                            "repo": matrix["replay"]["dataset"],
                            "file": matrix["replay"]["dataset_file"],
                            "revision": matrix["replay"]["dataset_revision"],
                        }]
                        assert any((row["repo"], row["file"], row["revision"]) ==
                                   (repo_id, filename, revision) for row in expected)
                        if os.environ["FAILURE"] == repo_type + "-download":
                            raise RuntimeError("fixture download failure")
                        path = Path(os.environ["HF_HUB_CACHE"]) / filename
                        print("\x1b[32m✓ Downloaded\x1b[0m\n  path: " + str(path))
                        return str(path)
                '''))
                hf = root / "hf"
                hf.write_text("#!/bin/sh\nprintf '\\033[32m✓ Downloaded\\033[0m\\n  path: /not/a/path\\n'\n")
                hf.chmod(0o755)
                if failure == "model-corrupt":
                    (cache / matrix["models"][0]["file"]).write_bytes(b"bad model")
                elif failure == "dataset-corrupt":
                    dataset.write_bytes(b"bad dataset")
                env_file = root / "github-env"
                env = dict(os.environ, PATH=f"{root}:{os.environ['PATH']}",
                           PYTHONPATH=str(root), MATRIX_FILE=str(root / "matrix.json"),
                           RUNNER_TEMP=str(root), GITHUB_ENV=str(env_file),
                           HF_HUB_CACHE=str(cache), HF_HUB_OFFLINE="0", FAILURE=failure)
                result = subprocess.run(
                    ["bash", "-c", self.step("Verify pinned replay inputs")["run"]],
                    cwd=root, env=env, capture_output=True, text=True,
                )
                if failure == "success":
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn(f"DATASET_FILE={dataset}\n", env_file.read_text())
                    self.assertEqual(result.stdout.count("verified "), len(matrix["models"]) + 1)
                else:
                    self.assertNotEqual(result.returncode, 0)
                    self.assertNotIn("DATASET_FILE=", env_file.read_text())
                    expected = "SHA-256 mismatch" if failure.endswith("corrupt") else "fixture download failure"
                    self.assertIn(expected, result.stderr)

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
