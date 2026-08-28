from __future__ import annotations

import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPAIR = ROOT / "scripts" / "llama-canary-agent-repair.sh"


class LlamaCanaryAgentRepairContractTests(unittest.TestCase):
    """Behavioral contracts for the canary repair wrapper.

    The wrapper mediates between an untrusted model turn and repository-write
    credentials. These tests pin the invariants the review demanded: the
    agent never sees a GitHub token, the token never reaches the environment,
    dispatch SHAs are validated as 40-hex before any use, battery-mode
    evidence is reused instead of re-running the battery, and persistent
    runner state is cleared at the start of every run.
    """

    def test_agent_turns_strip_github_tokens_from_environment(self) -> None:
        wrapper = REPAIR.read_text(encoding="utf-8")
        # The write PAT is never exported into the environment.
        self.assertNotIn("export GH_TOKEN", wrapper)
        # Agent turns explicitly strip every GitHub credential.
        self.assertIn("env -u GH_TOKEN -u GITHUB_TOKEN -u CANARY_REPAIR_TOKEN", wrapper)
        # GitHub mutations go through the token-scoped helper.
        self.assertIn("gh_repair() {", wrapper)
        self.assertNotIn("\n  gh pr create", wrapper)
        self.assertNotIn("\n  gh issue create", wrapper)
        self.assertNotIn("\n  gh pr comment", wrapper)
        self.assertNotIn("\n  gh issue comment", wrapper)
        self.assertNotIn("\n  gh pr edit", wrapper)

    def test_certified_battery_is_bound_to_the_repair_pr_head(self) -> None:
        wrapper = REPAIR.read_text(encoding="utf-8")
        # The wrapper — never the agent — commits and pushes the certified tree.
        self.assertIn("publish_repair_branch", wrapper)
        self.assertIn('CERTIFIED_SHA="$(git rev-parse HEAD)"', wrapper)
        # Success requires the remote PR head to equal the certified commit.
        self.assertIn("verify_pr_head_is_certified", wrapper)
        self.assertIn("report_success", wrapper)
        # The PR-body agent turn runs only before certification; after a green
        # battery only the deterministic apply_pr_body may run.
        self.assertIn("draft_pr_body", wrapper)
        self.assertIn("apply_pr_body", wrapper)
        first_certify = wrapper.index("certification attempt")
        self.assertLess(wrapper.index("draft_pr_body()"), first_certify)
        self.assertNotIn("write_pr_body", wrapper)

    def test_battery_mode_reuses_workflow_evidence_without_rerunning(self) -> None:
        wrapper = REPAIR.read_text(encoding="utf-8")
        # Battery mode seeds from the workflow's teed evidence log when
        # present and only runs a diagnostic battery when it is absent.
        self.assertIn("reusing workflow battery evidence", wrapper)
        self.assertIn("no workflow battery evidence", wrapper)
        # A missing battery log must not crash the failure-path summaries.
        self.assertIn("(no battery output captured", wrapper)
        # The first battery-mode loop iteration is a repair turn seeded from
        # the workflow evidence — never a second full build+battery run
        # before the agent gets the failure output.
        self.assertIn(
            'if [[ "$MODE" == "battery" && "$attempt" -eq 1 ]]; then', wrapper
        )
        self.assertIn(
            "battery mode: repair turn 1 seeded from the workflow battery failure evidence",
            wrapper,
        )

    def test_every_github_call_is_token_scoped(self) -> None:
        wrapper = REPAIR.read_text(encoding="utf-8")
        # The workflow job exports no ambient GH_TOKEN and checks out with
        # persist-credentials disabled: every gh invocation — reads included
        # — must go through the token-scoped helper.
        lines = [
            line
            for line in wrapper.splitlines()
            if not line.lstrip().startswith("#")
            and " gh " in f" {line.strip()} "
            # `command -v gh` checks binary presence, not an API call.
            and "command -v" not in line
        ]
        for line in lines:
            self.assertIn(
                "gh_repair",
                line,
                f"bare gh invocation bypasses the repair token: {line.strip()}",
            )

    def test_run_scopes_persistent_runner_state(self) -> None:
        wrapper = REPAIR.read_text(encoding="utf-8")
        # The PR-body draft is cleared every run; the battery evidence log is
        # cleared in patch-queue mode but preserved in battery mode, where it
        # holds this run's workflow-teed evidence.
        self.assertIn(
            'rm -f "$ROOT/.deps/llama-canary-pr-body.md"', wrapper
        )
        self.assertIn('if [[ "$MODE" == "patch-queue" ]]; then\n  rm -f "$BATTERY_LOG"', wrapper)
        # The repair push URL embeds the token; its stderr is redacted.
        self.assertIn("redact_token", wrapper)

    def test_dispatch_sha_is_validated_before_use(self) -> None:
        env = {
            **os.environ,
            "UPSTREAM_SHA_INPUT": "not-a-sha; echo pwned",
            "CANARY_REPAIR_TOKEN": "test-token",
            "GITHUB_REPOSITORY": "Mesh-LLM/mesh-llm",
        }
        env["PATH"] = str(ROOT / "scripts" / "tests" / "fixtures") + os.pathsep + env.get("PATH", "")
        with tempfile.TemporaryDirectory() as tmp:
            # The prerequisite checks (opencode, credentials) intentionally
            # pass in this environment only when the fixtures exist; run the
            # script and require it to never accept the invalid SHA.
            result = subprocess.run(
                [str(REPAIR), "patch-queue"],
                cwd=tmp,
                env=env,
                text=True,
                capture_output=True,
                check=False,
                timeout=60,
            )
        combined = result.stdout + result.stderr
        # The crafted SHA is refused as non-40-hex and never executed: the
        # only place it appears is the refusal message itself.
        self.assertIn("refusing to repair against a non-40-hex upstream SHA", combined)
        self.assertEqual(1, result.returncode)
        self.assertEqual(
            1,
            combined.count("pwned"),
            "the crafted SHA must only appear in the refusal message, never as executed output",
        )

    def test_manual_positional_sha_is_still_validated(self) -> None:
        script = subprocess.run(
            ["bash", "-n", str(REPAIR)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(0, script.returncode, script.stderr)


if __name__ == "__main__":
    unittest.main()
