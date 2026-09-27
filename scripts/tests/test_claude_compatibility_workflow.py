from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"


class ClaudeCompatibilityWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pr_source = (WORKFLOWS / "pr_linux.yml").read_text(encoding="utf-8")
        self.pr = yaml.safe_load(self.pr_source)
        self.live_source = (WORKFLOWS / "claude-live-model-gate.yml").read_text(
            encoding="utf-8"
        )

    def test_real_client_test_is_enabled_and_required_for_affected_prs(self) -> None:
        job = self.pr["jobs"]["plan"]
        self.assertEqual(job["timeout-minutes"], 45)
        commands = "\n".join(
            step.get("run", "") for step in job["steps"] if isinstance(step, dict)
        )
        self.assertIn("@anthropic-ai/claude-code@2.1.273", commands)
        self.assertIn("--features claude-code-integration", commands)
        self.assertIn("claude_cli_executes_read_tool_through_host_ingress", commands)
        self.assertNotIn("--ignored", commands)
        integration_step = next(
            step
            for step in job["steps"]
            if step.get("name") == "Run Claude Code protocol and real-client integration"
        )
        self.assertEqual(integration_step["env"]["CARGO_INCREMENTAL"], "0")
        self.assertEqual(integration_step["env"]["CARGO_PROFILE_TEST_DEBUG"], "0")

    def test_manual_live_gate_can_build_the_repository(self) -> None:
        live = yaml.safe_load(self.live_source)
        job = live["jobs"]["live_claude"]
        self.assertEqual(job["environment"], "claude-live-model")
        self.assertEqual(
            job["env"]["ANTHROPIC_API_KEY"], "${{ secrets.ANTHROPIC_API_KEY }}"
        )
        self.assertIn("CARGO_PROFILE_TEST_DEBUG: '0'", self.live_source)
        self.assertIn("mozilla-actions/sccache-action@", self.live_source)
        self.assertIn("--features claude-live-model-integration", self.live_source)


if __name__ == "__main__":
    unittest.main()
