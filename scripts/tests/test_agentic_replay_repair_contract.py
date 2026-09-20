from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPAIR = ROOT / "scripts" / "agentic-replay-repair.sh"
PARAMS = ROOT / "scripts" / "agentic-replay-params.py"
WORKFLOW = ROOT / ".github" / "workflows" / "agentic-replay-nightly.yml"
MATRIX = ROOT / "ci" / "agentic-replay-nightly" / "matrix.json"


class AgenticReplayRepairContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repair = REPAIR.read_text(encoding="utf-8")
        self.params = PARAMS.read_text(encoding="utf-8")
        self.workflow = WORKFLOW.read_text(encoding="utf-8")
        self.matrix = json.loads(MATRIX.read_text(encoding="utf-8"))

    def test_failed_unchanged_or_policy_modifying_agent_cannot_publish(self) -> None:
        for mode in ("failure", "unchanged", "protected", "preflight"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                repo = root / "repo"
                repo.mkdir()
                (repo / "scripts").mkdir()
                (repo / "ci").mkdir()
                (repo / "ci/policy.json").write_text("{}\n")
                shutil.copy(REPAIR, repo / "scripts/agentic-replay-repair.sh")
                shutil.copy(ROOT / "scripts/run-command-with-timeout.py", repo / "scripts/run-command-with-timeout.py")
                bin_dir = root / "bin"
                bin_dir.mkdir()
                stub = bin_dir / "goose"
                stub.write_text('''#!/bin/sh
test -z "${GH_TOKEN:-}${GITHUB_TOKEN:-}${HF_TOKEN:-}${CANARY_REPAIR_TOKEN:-}" || exit 98
if [ "$1" = "--version" ]; then echo test-goose; exit 0; fi
if [ "$1 $2" = "info --check" ]; then
  test "$GOOSE_PROVIDER/$GOOSE_MODEL" = "test-provider/test-model" || exit 97
  test "$TEST_AGENT_MODE" != "preflight" || exit 43
  exit 0
fi
test "$1 $2 $3 $4 $5" = "run --provider test-provider --model test-model" || exit 99
test "$GOOSE_MODE/$GOOSE_DISABLE_SESSION_NAMING" = "auto/true" || exit 96
case "$TEST_AGENT_MODE" in
  failure) exit 42 ;;
  protected) echo '{"weaken":true}' > ci/policy.json ;;
esac
''')
                stub.chmod(0o755)
                env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", TEST_AGENT_MODE=mode,
                           GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1", GITHUB_RUN_ID="123",
                           GITHUB_RUN_ATTEMPT="1", GH_TOKEN="test", GITHUB_TOKEN="test",
                           HF_TOKEN="test", CANARY_REPAIR_TOKEN="test",
                           REPLAY_AGENT_PROVIDER="test-provider", REPLAY_AGENT_MODEL="test-model")
                def git(*args):
                    return subprocess.run(["git", *args], cwd=repo, env=env, check=True, capture_output=True)
                git("init", "-b", "main")
                git("config", "user.name", "Test Fixture")
                git("config", "user.email", "fixture@example.invalid")
                git("add", ".")
                git("-c", "commit.gpgsign=false", "commit", "-m", "test: fixture")
                output = root / "artifacts"
                output.mkdir()
                result = subprocess.run(["bash", "scripts/agentic-replay-repair.sh", str(output)],
                                        cwd=repo, env=env, capture_output=True, text=True)
                expected_status = {"failure": 42, "preflight": 43}.get(mode, 1)
                self.assertEqual(result.returncode, expected_status, result.stderr)
                self.assertFalse((output / "repair-publication").exists())
                self.assertEqual(git("rev-list", "--count", "HEAD").stdout.strip(), b"1")

    def test_persistent_repair_only_emits_publication_data(self) -> None:
        self.assertIn("unset CANARY_REPAIR_TOKEN GH_TOKEN GITHUB_TOKEN", self.repair)
        self.assertIn("run_untrusted goose", self.repair)
        self.assertIn("run_untrusted python3 evals/agentic-replay.py", self.repair)
        self.assertIn("run_untrusted python3 scripts/agentic-replay-history.py", self.repair)
        self.assertIn('PUBLICATION_DIR="$OUTPUT_DIR/repair-publication"', self.repair)
        self.assertIn("git format-patch -1 --binary --stdout HEAD", self.repair)
        self.assertIn('"patch_sha256"', self.repair)
        self.assertIn('"body_sha256"', self.repair)
        self.assertNotIn('REPAIR_TOKEN="', self.repair)
        self.assertNotIn("GIT_ASKPASS", self.repair)
        self.assertNotIn("gh_repair", self.repair)
        self.assertNotIn("gh pr create", self.repair)
        self.assertNotIn("git push", self.repair)
        self.assertIn("git -c core.hooksPath=/dev/null commit", self.repair)
        self.assertIn('BASE_SHA=$(git rev-parse HEAD)', self.repair)
        self.assertIn('git diff --cached --quiet "$BASE_SHA" -- .github .agents scripts evals ci', self.repair)
        self.assertNotIn('--mode agent', self.repair)
        self.assertIn('goose run --provider', self.repair)
        self.assertIn('goose info --check', self.repair)
        self.assertIn('--with-builtin developer --no-profile', self.repair)
        self.assertIn('--output-format text', self.repair)
        self.assertNotIn('opencode', self.repair)
        self.assertIn('--seconds 3600 --label agentic-replay-agent', self.repair)
        self.assertNotIn('--allow-empty', self.repair)
        self.assertIn("git -c core.hooksPath=/dev/null commit --no-gpg-sign", self.repair)
        self.assertIn('RUN_ATTEMPT="${GITHUB_RUN_ATTEMPT:-1}"', self.repair)
        self.assertIn('repair-${RUN_ID}-${RUN_ATTEMPT}', self.repair)
        self.assertIn('s|{{SOURCE_SHA}}|${BASE_SHA}|g', self.repair)
        self.assertIn('if [[ ! -d "$OUTPUT_DIR" || -L "$OUTPUT_DIR" ]]; then', self.repair)
        self.assertIn('rm -rf -- "$PUBLICATION_DIR"', self.repair)
        self.assertIn('install -d -m 700 -- "$PUBLICATION_DIR"', self.repair)
        self.assertNotIn('"repair_commit_sha"', self.repair)

    def test_hosted_publication_job_owns_secret_and_publication(self) -> None:
        repair_step = self.workflow.split(
            "      - name: Prepare repair PR artifact on regression (Goose)", 1
        )[1].split("      - uses: actions/upload-artifact@", 1)[0]
        publication = self.workflow.split("  publish-repair:", 1)[1]
        self.assertNotIn("CANARY_REPAIR_TOKEN: ${{ secrets.CANARY_REPAIR_TOKEN }}", repair_step)
        self.assertNotIn("git push", repair_step)
        self.assertNotIn("gh pr create", repair_step)
        self.assertIn("needs: replay", publication)
        self.assertIn("!cancelled()", publication)
        self.assertNotIn("always()", publication)
        self.assertIn("needs.replay.result == 'failure'", publication)
        self.assertIn("needs.replay.outputs.repair_prepared == 'true'", publication)
        self.assertIn("repair_prepared: ${{ steps.repair.outputs.prepared }}", self.workflow)
        self.assertIn('echo "prepared=true" >> "$GITHUB_OUTPUT"', repair_step)
        self.assertIn('echo "prepared=false" >> "$GITHUB_OUTPUT"', repair_step)
        self.assertIn("runs-on: ubuntu-latest", publication)
        self.assertIn("actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c", publication)
        self.assertIn("ref: ${{ github.sha }}", publication)
        replay_workflow = self.workflow.split("  publish-repair:", 1)[0]
        self.assertIn(
            'with:\n          ref: ${{ github.sha }}\n          persist-credentials: false',
            replay_workflow,
        )
        self.assertIn("CANARY_REPAIR_TOKEN: ${{ secrets.CANARY_REPAIR_TOKEN }}", publication)
        self.assertIn("git -c core.hooksPath=/dev/null", publication)
        self.assertIn("push \"https://github.com/${GITHUB_REPOSITORY}.git\"", publication)
        self.assertIn("gh pr create", publication)
        self.assertIn("--body-file \"$PUBLICATION_DIR/pr-body.md\"", publication)
        self.assertIn('"${{ github.run_attempt }}"', publication)
        self.assertIn('"run_attempt"', publication)
        self.assertIn('echo "run_attempt=$RUN_ATTEMPT"', publication)
        self.assertIn("git -c core.hooksPath=/dev/null am --no-verify --no-gpg-sign --empty=keep", publication)
        self.assertIn("git diff --quiet HEAD^ HEAD -- .github .agents scripts evals ci", publication)
        self.assertNotIn('"repair_commit_sha"', publication)
        self.assertIn('repair-${RUN_ID}-${RUN_ATTEMPT}', publication)
        self.assertIn('run ${RUN_ID}-${RUN_ATTEMPT}', publication)

    def test_hosted_validator_binds_publication_to_workflow_run(self) -> None:
        self.assertIn(
            '"${{ github.sha }}" "${{ github.run_id }}" "${{ github.run_attempt }}"',
            self.workflow,
        )
        self.assertIn('expected_run_id = sys.argv[6]', self.workflow)
        self.assertIn('expected_attempt = sys.argv[7]', self.workflow)
        self.assertIn('or status["run_id"] != expected_run_id', self.workflow)

    def test_repair_rerun_uses_all_matrix_replay_parameters(self) -> None:
        for argument, variable in (
            ("--replay-mode", '"$REPLAY_MODE"'),
            ("--trajectories-per-framework", '"$TRAJECTORIES_PER_FRAMEWORK"'),
            ("--passes", '"$PASSES"'),
            ("--warmup-turns", '"$WARMUP_TURNS"'),
            ("--max-output-tokens", '"$MAX_OUTPUT_TOKENS"'),
        ):
            self.assertIn(f"{argument} {variable}", self.repair)
        self.assertIn('MATRIX_FILE="${MATRIX_FILE:-ci/agentic-replay-nightly/matrix.json}"', self.repair)
        self.assertIn("python3 scripts/agentic-replay-params.py", self.repair)
        self.assertIn('--json-output "$REPLAY_PARAMS_FILE" --print-shell', self.repair)
        self.assertIn("MAX_OUTPUT_TOKENS", self.repair)
        self.assertNotIn("--trajectories-per-framework 8", self.repair)
        self.assertNotIn("--warmup-turns 4", self.repair)

    def test_repair_compares_against_the_immutable_workflow_base(self) -> None:
        self.assertIn("immutable base $BASE_SHA", self.repair)
        self.assertIn('--ref "base=$BASE_SHA"', self.repair)
        self.assertNotIn("origin/main", self.repair)

    def test_history_and_publication_explicitly_allow_network_access(self) -> None:
        self.assertIn('HF_HUB_OFFLINE=0 hf download "$DATASET_REPO"', self.workflow)
        self.assertIn('HF_HUB_OFFLINE=0 python3 - "$DATASET_REPO"', self.workflow)
        self.assertEqual(self.workflow.count('HF_HUB_OFFLINE=0 hf upload "$DATASET_REPO"'), 2)

    def test_workflow_validates_and_passes_mode_and_max_output(self) -> None:
        self.assertIn("python3 scripts/agentic-replay-params.py", self.workflow)
        self.assertIn('--github-env "$GITHUB_ENV"', self.workflow)
        self.assertIn('MODE_MAP = {"checkpoint": "checkpoints", "final": "final", "all": "all"}', self.params)
        self.assertIn('raise SystemExit(f"{key} must be a positive integer")', self.params)
        self.assertIn("AGENTIC_REPLAY_MAX_OUTPUT_TOKENS", self.workflow)
        self.assertIn('--replay-mode "$REPLAY_MODE"', self.workflow)
        self.assertIn('--max-output-tokens "$MAX_OUTPUT"', self.workflow)
        self.assertIn('--trajectories-per-framework "$TPFS"', self.workflow)
        self.assertIn('--passes "$PASSES"', self.workflow)
        self.assertIn('--warmup-turns "$WARMUP"', self.workflow)
        self.assertIn('LEVEL_ARGS+=(--concurrency "$level")', self.workflow)
        self.assertIn('json.dumps(replay, sort_keys=True)', self.params)
        self.assertIn('--replay "$RUNNER_TEMP/agentic-replay-params.json"', self.workflow)
        self.assertEqual(self.matrix["replay"]["mode"], "checkpoint")
        self.assertEqual(
            {"checkpoint": "checkpoints", "final": "final", "all": "all"}[self.matrix["replay"]["mode"]],
            "checkpoints",
        )
        self.assertGreater(self.matrix["replay"]["max_output_tokens"], 0)

    def test_replay_parameter_contract_has_one_implementation(self) -> None:
        self.assertNotIn("mode_map =", self.workflow)
        self.assertNotIn("mode_map =", self.repair)
        self.assertNotIn("concurrency must be a non-empty list", self.workflow)
        self.assertNotIn("concurrency must be a non-empty list", self.repair)
        self.assertIn("concurrency must be a non-empty list", self.params)


if __name__ == "__main__":
    unittest.main()
