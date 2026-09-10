from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPAIR = ROOT / "scripts" / "agentic-replay-repair.sh"
WORKFLOW = ROOT / ".github" / "workflows" / "agentic-replay-nightly.yml"
MATRIX = ROOT / "ci" / "agentic-replay-nightly" / "matrix.json"


class AgenticReplayRepairContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.repair = REPAIR.read_text(encoding="utf-8")
        self.workflow = WORKFLOW.read_text(encoding="utf-8")
        self.matrix = json.loads(MATRIX.read_text(encoding="utf-8"))

    def test_credentials_are_captured_then_removed_before_untrusted_execution(self) -> None:
        capture = self.repair.index('REPAIR_TOKEN="${CANARY_REPAIR_TOKEN:-}"')
        clear = self.repair.index("unset CANARY_REPAIR_TOKEN GH_TOKEN GITHUB_TOKEN")
        opencode = self.repair.index("run_untrusted opencode")
        self.assertLess(capture, clear)
        self.assertLess(clear, opencode)
        self.assertIn("export -n REPAIR_TOKEN", self.repair)
        self.assertIn("env -u CANARY_REPAIR_TOKEN -u GH_TOKEN -u GITHUB_TOKEN -u REPAIR_TOKEN", self.repair)
        self.assertIn("run_untrusted python3 evals/agentic-replay.py", self.repair)
        self.assertIn("run_untrusted python3 scripts/agentic-replay-history.py", self.repair)
        self.assertNotIn("${GH_TOKEN:-}", self.repair)

    def test_push_and_github_api_use_narrow_credential_scopes(self) -> None:
        self.assertIn('CANARY_REPAIR_TOKEN="$REPAIR_TOKEN"', self.repair)
        self.assertIn('GIT_ASKPASS="$ASKPASS_SCRIPT"', self.repair)
        self.assertIn('GIT_TERMINAL_PROMPT=0', self.repair)
        self.assertIn('git -c core.hooksPath=/dev/null -c credential.helper=', self.repair)
        self.assertIn('push "https://github.com/${GITHUB_REPOSITORY:-Mesh-LLM/mesh-llm}.git"', self.repair)
        self.assertNotIn("x-access-token:${", self.repair)
        self.assertNotIn("PUSH_REMOTE=", self.repair)
        self.assertIn('GH_TOKEN="$REPAIR_TOKEN" gh "$@"', self.repair)
        self.assertIn("gh_repair pr create", self.repair)
        self.assertIn('REDACTION_TOKEN="$REPAIR_TOKEN"', self.repair)
        self.assertIn('.replace(os.environ["REDACTION_TOKEN"]', self.repair)
        self.assertIn('trap cleanup EXIT', self.repair)

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
        self.assertIn("unsupported replay mode", self.repair)
        self.assertIn('mode_map = {"checkpoint": "checkpoints", "final": "final", "all": "all"}', self.repair)
        self.assertIn("max_output_tokens", self.repair)
        self.assertNotIn("--trajectories-per-framework 8", self.repair)
        self.assertNotIn("--warmup-turns 4", self.repair)

    def test_workflow_validates_and_passes_mode_and_max_output(self) -> None:
        self.assertIn('mode_map = {"checkpoint": "checkpoints", "final": "final", "all": "all"}', self.workflow)
        self.assertIn('replay_mode = mode_map[mode]', self.workflow)
        self.assertIn('f"{key} must be a positive integer"', self.workflow)
        self.assertIn('"max_output_tokens"', self.workflow)
        self.assertIn('--replay-mode "$REPLAY_MODE"', self.workflow)
        self.assertIn('--max-output-tokens "$MAX_OUTPUT"', self.workflow)
        self.assertIn('--trajectories-per-framework "$TPFS"', self.workflow)
        self.assertIn('--passes "$PASSES"', self.workflow)
        self.assertIn('--warmup-turns "$WARMUP"', self.workflow)
        self.assertIn('LEVEL_ARGS+=(--concurrency "$level")', self.workflow)
        self.assertIn("CANARY_REPAIR_TOKEN: ${{ secrets.CANARY_REPAIR_TOKEN }}", self.workflow)
        self.assertNotIn("GH_TOKEN: ${{ secrets.CANARY_REPAIR_TOKEN }}", self.workflow)
        self.assertIn('json.dumps(replay, sort_keys=True)', self.workflow)
        self.assertIn('--replay "$RUNNER_TEMP/agentic-replay-params.json"', self.workflow)
        self.assertEqual(self.matrix["replay"]["mode"], "checkpoint")
        self.assertEqual(
            {"checkpoint": "checkpoints", "final": "final", "all": "all"}[self.matrix["replay"]["mode"]],
            "checkpoints",
        )
        self.assertGreater(self.matrix["replay"]["max_output_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
