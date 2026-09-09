from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WRAPPER = ROOT / "scripts" / "llama-canary-agent-repair.sh"


class LlamaCanaryStateMachineContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.wrapper = WRAPPER.read_text(encoding="utf-8")

    def test_wrapper_has_one_ordered_state_machine(self) -> None:
        main = self.wrapper[self.wrapper.index('phase="prepare"\nwhile true; do') :]
        self.assertIn('phase="build"; continue', main)
        self.assertIn('phase="certify"; continue', main)
        self.assertIn("report_terminal certified", main)
        self.assertIn('phase="prepare"\ndone', main)
        self.assertNotIn("post_green", self.wrapper)
        self.assertNotIn("patch-queue | battery", self.wrapper)

    def test_every_agent_edit_restarts_prepare_and_full_build(self) -> None:
        main = self.wrapper[self.wrapper.index('phase="prepare"\nwhile true; do') :]
        self.assertLess(main.index("run_prepare"), main.index("run_full_build"))
        self.assertLess(main.index("run_full_build"), main.index("run_certification"))
        self.assertLess(main.index("agent_turn"), main.rindex('phase="prepare"'))
        prompt = self.wrapper[
            self.wrapper.index("repair_prompt() {") : self.wrapper.index("current_pr() {")
        ]
        self.assertIn("restart at prepare", prompt)
        self.assertIn("complete build", prompt)
        self.assertIn("full supported-family certification", prompt)

    def test_prepare_owns_pin_and_exact_prepared_upstream(self) -> None:
        prepare = self.wrapper[
            self.wrapper.index("run_prepare() {") : self.wrapper.index("run_full_build() {")
        ]
        self.assertIn('scripts/update-llama-pin.sh "$UPSTREAM_SHA"', self.wrapper)
        self.assertIn("verify_repair_pin", prepare)
        self.assertIn("scripts/prepare-llama.sh pinned", prepare)
        self.assertIn(".mesh-llm-upstream-sha", prepare)
        self.assertNotIn("PIN_MIRROR_FILE", self.wrapper)

    def test_build_gate_is_complete_and_precedes_certification(self) -> None:
        build = self.wrapper[
            self.wrapper.index("run_full_build() {") : self.wrapper.index("run_certification() {")
        ]
        self.assertIn("LLAMA_STAGE_UPSTREAM_TESTS=ON", build)
        self.assertIn("arch -arm64 bash scripts/build-llama.sh", build)
        self.assertIn("candidate native archive must be arm64", build)
        self.assertIn("scripts/check-skippy-generated-family-patch.sh", build)
        for package in (
            "skippy-runtime",
            "skippy-server",
            "skippy-model-package",
            "skippy-correctness",
        ):
            self.assertIn(f"-p {package}", build)
        self.assertIn("scripts/skippy-ci-smoke.sh", build)

    def test_certification_is_full_and_uses_prebuilt_candidate(self) -> None:
        certify = self.wrapper[
            self.wrapper.index("run_certification() {") : self.wrapper.index("phase_log() {")
        ]
        self.assertIn("skippy-llama-parity.py --llama-src .deps/llama.cpp validate", certify)
        self.assertIn("--cadence llama-bump", certify)
        self.assertNotIn("--families", certify)
        self.assertIn("scripts/skippy-canary-live-matrix.sh --prepare", certify)
        self.assertIn("scripts/skippy-family-battery.sh --skip-build --plan", certify)

    def test_internal_deadline_reserves_terminal_publication_time(self) -> None:
        self.assertIn('REPAIR_BUDGET_SECONDS="${CANARY_REPAIR_BUDGET_SECONDS:-41400}"', self.wrapper)
        self.assertIn('PUBLISH_RESERVE_SECONDS="${CANARY_PUBLISH_RESERVE_SECONDS:-1800}"', self.wrapper)
        self.assertIn("DEADLINE_AT - $(date +%s) - PUBLISH_RESERVE_SECONDS", self.wrapper)
        self.assertIn("scripts/run-command-with-timeout.py", self.wrapper)
        for label in (
            "apply llama.cpp patch queue",
            "complete patched llama.cpp build",
            "full supported-family certification",
            "agent repair turn",
        ):
            self.assertIn(label, self.wrapper)
        self.assertIn("publication reserve is active", self.wrapper)
        self.assertIn("report_terminal failed", self.wrapper)

    def test_repair_turn_limit_is_per_phase(self) -> None:
        for counter in (
            "PREPARE_REPAIR_TURNS",
            "BUILD_REPAIR_TURNS",
            "CERTIFY_REPAIR_TURNS",
        ):
            self.assertIn(counter, self.wrapper)
        self.assertIn('phase_turns "$phase"', self.wrapper)
        self.assertIn('increment_phase_turns "$phase"', self.wrapper)

    def test_terminal_publication_uses_unique_branch_without_force_push(self) -> None:
        self.assertIn('BRANCH="llama-canary/repair-${RUN_KEY}-${UPSTREAM_SHA:0:10}"', self.wrapper)
        publish = self.wrapper[
            self.wrapper.index("publish_terminal_branch() {") : self.wrapper.index("write_pr_body() {")
        ]
        self.assertIn('"HEAD:refs/heads/${BRANCH}"', publish)
        self.assertNotIn("+HEAD", publish)
        self.assertNotIn("--force", publish)
        main = self.wrapper[self.wrapper.index('phase="prepare"\nwhile true; do') :]
        self.assertNotIn("publish_terminal_branch", main)
        self.assertNotIn("gh pr create", main)
        self.assertIn("report_terminal certified", main)
        self.assertIn("report_terminal failed", main)

    def test_failed_terminal_state_is_draft_and_green_is_exact_head(self) -> None:
        ensure = self.wrapper[
            self.wrapper.index("ensure_pr() {") : self.wrapper.index("verify_pr_head() {")
        ]
        self.assertIn("create_args=(--draft)", ensure)
        self.assertIn("create_args=()", ensure)
        self.assertIn("not certified and is not eligible to merge", self.wrapper)
        report = self.wrapper[
            self.wrapper.index("report_terminal() {") : self.wrapper.index('phase="prepare"')
        ]
        self.assertLess(report.index("publish_terminal_branch"), report.index("ensure_pr"))
        self.assertLess(report.index("ensure_pr"), report.index("verify_pr_head"))
        self.assertIn('CERTIFIED_SHA="$PUBLISHED_SHA"', self.wrapper)

    def test_agent_has_no_github_credentials_or_publication_authority(self) -> None:
        self.assertNotIn("export GH_TOKEN", self.wrapper)
        agent = self.wrapper[
            self.wrapper.index("agent_turn() {") : self.wrapper.index("write_repair_pin() {")
        ]
        self.assertIn("-u GH_TOKEN -u GITHUB_TOKEN -u CANARY_REPAIR_TOKEN", agent)
        self.assertIn('opencode run --auto --model "$AGENT_MODEL"', agent)
        self.assertNotIn("git push", agent)
        self.assertNotIn("gh pr", agent)
        self.assertIn("heartbeat: agent repair running for", agent)

    def test_every_github_call_is_token_scoped(self) -> None:
        for line in self.wrapper.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or " gh " not in f" {stripped} ":
                continue
            self.assertIn("gh_repair", stripped, stripped)

    def test_token_permission_probe_is_unique_and_runs_before_work(self) -> None:
        self.assertIn("canary-repair-token-preflight-${RUN_KEY}", self.wrapper)
        self.assertIn("git/refs/heads%2F${probe_branch}", self.wrapper)
        call = self.wrapper.index("check_repair_token_permissions\n")
        self.assertLess(call, self.wrapper.index("agent_turn()"))
        self.assertLess(call, self.wrapper.index("run_prepare()"))

    def test_dispatch_sha_is_rejected_before_use(self) -> None:
        crafted = "not-a-sha; echo pwned"
        result = subprocess.run(
            [str(WRAPPER)],
            cwd=ROOT,
            env={**os.environ, "UPSTREAM_SHA_INPUT": crafted},
            text=True,
            capture_output=True,
            check=False,
            timeout=30,
        )
        combined = result.stdout + result.stderr
        self.assertEqual(1, result.returncode)
        self.assertIn("non-40-hex upstream SHA", combined)
        self.assertEqual(1, combined.count("pwned"))

    def test_shell_syntax(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(WRAPPER)], capture_output=True, text=True, check=False
        )
        self.assertEqual(0, result.returncode, result.stderr)

    def test_persistent_runner_scratch_is_scoped_and_pruned(self) -> None:
        self.assertIn('STATE_DIR="$ROOT/.deps/llama-canary-state-${RUN_KEY}"', self.wrapper)
        self.assertIn('TARGET_SHA_FILE="$ROOT/.deps/llama-canary-target-sha"', self.wrapper)
        self.assertIn('printf \'%s\\n\' "$UPSTREAM_SHA" > "$TARGET_SHA_FILE"', self.wrapper)
        self.assertIn('git -C "$ROOT/.deps/llama.cpp" worktree prune', self.wrapper)
        self.assertIn("rm -rf /tmp/llama-old-pin /tmp/llama-repair /tmp/llama-repair-*", self.wrapper)
        self.assertIn("redact_token", self.wrapper)

    def test_runnable_row_carrying_unsupported_reason_is_rejected(self) -> None:
        parity = ROOT / "scripts" / "skippy-llama-parity.py"
        sys.path.insert(0, str(parity.parent))
        try:
            spec = importlib.util.spec_from_file_location("skippy_llama_parity_validate", parity)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        finally:
            sys.path.pop(0)
        for status in ("certified", "candidate", "candidate_stateful"):
            rows = [{"llama_model": "somearch", "status": status, "unsupported_reason": "leftover"}]
            self.assertEqual(module.validate_boundary_registration(rows, {"somearch"}), 1)
        self.assertEqual(
            module.validate_boundary_registration(
                [{"llama_model": "x", "status": "non_causal_aux", "unsupported_reason": "non-causal encoder"}],
                set(),
            ),
            0,
        )


if __name__ == "__main__":
    unittest.main()
