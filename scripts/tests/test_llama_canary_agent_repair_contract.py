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
PUBLISHER = ROOT / "scripts" / "llama-canary-publish.sh"
RUNBOOK = ROOT / "ci" / "llama-canary" / "agent-repair-prompt.md"


class LlamaCanaryDeveloperHarnessContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.wrapper = WRAPPER.read_text(encoding="utf-8")
        self.publisher = PUBLISHER.read_text(encoding="utf-8")

    def test_wrapper_runs_one_agent_then_one_ordered_verification(self) -> None:
        main = self.wrapper[self.wrapper.index("write_repair_pin\n") :]
        self.assertEqual(1, main.count("agent_turn"))
        self.assertLess(main.index("agent_turn"), main.index("run_prepare"))
        self.assertLess(main.index("run_prepare"), main.index("run_full_build"))
        self.assertLess(main.index("run_full_build"), main.index("run_certification"))
        self.assertLess(main.index("run_certification"), main.index("finalize_certified_tree"))
        for obsolete in (
            "MAX_REPAIR_TURNS",
            "PREPARE_REPAIR_TURNS",
            "BUILD_REPAIR_TURNS",
            "CERTIFY_REPAIR_TURNS",
            "report_terminal",
            "while true",
        ):
            self.assertNotIn(obsolete, self.wrapper)

    def test_wrapper_reexecs_natively_before_state_initialization(self) -> None:
        reexec = self.wrapper.index('exec arch -arm64 "${BASH_SOURCE[0]}" "$@"')
        root = self.wrapper.index('ROOT="$(cd')
        self.assertLess(reexec, root)
        self.assertIn("sysctl -n hw.optional.arm64", self.wrapper[:root])

    def test_agent_and_final_verification_have_explicit_budgets(self) -> None:
        self.assertIn(
            'AGENT_TIMEOUT_SECONDS="${CANARY_AGENT_TIMEOUT_SECONDS:-27000}"',
            self.wrapper,
        )
        self.assertIn(
            'VERIFICATION_TIMEOUT_SECONDS="${CANARY_VERIFICATION_TIMEOUT_SECONDS:-14400}"',
            self.wrapper,
        )
        self.assertIn('run_for "agent developer task" "$AGENT_TIMEOUT_SECONDS"', self.wrapper)
        self.assertIn("VERIFICATION_DEADLINE_AT", self.wrapper)
        self.assertIn("scripts/run-command-with-timeout.py", self.wrapper)

        result = subprocess.run(
            [
                str(ROOT / "scripts" / "run-command-with-timeout.py"),
                "--seconds",
                "1",
                "--label",
                "agent developer task",
                "--",
                sys.executable,
                "-c",
                "import time; time.sleep(30)",
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=15,
        )
        self.assertEqual(124, result.returncode)
        self.assertIn("agent developer task timed out after 1s", result.stderr)

    def test_prepare_owns_pin_and_exact_prepared_upstream(self) -> None:
        prepare = self.wrapper[
            self.wrapper.index("run_prepare() {") : self.wrapper.index("run_full_build() {")
        ]
        self.assertIn('scripts/update-llama-pin.sh "$UPSTREAM_SHA"', self.wrapper)
        self.assertIn("verify_repair_pin", prepare)
        self.assertIn("scripts/prepare-llama.sh pinned", prepare)
        self.assertIn(".mesh-llm-upstream-sha", prepare)

    def test_build_gate_is_complete(self) -> None:
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
            self.wrapper.index("run_certification() {") : self.wrapper.index("write_upstream_summary() {")
        ]
        self.assertIn("skippy-llama-parity.py --llama-src .deps/llama.cpp validate", certify)
        self.assertIn("--cadence llama-bump", certify)
        self.assertNotIn("--families", certify)
        self.assertIn("scripts/skippy-canary-live-matrix.sh --prepare", certify)
        self.assertIn("scripts/skippy-family-battery.sh --skip-build --plan", certify)

    def test_agent_has_no_github_credentials_or_publication_authority(self) -> None:
        agent = self.wrapper[
            self.wrapper.index("agent_turn() {") : self.wrapper.index("assert_agent_control_unchanged() {")
        ]
        self.assertIn("-u GH_TOKEN -u GITHUB_TOKEN -u CANARY_REPAIR_TOKEN", agent)
        self.assertIn('opencode run --auto --model "$AGENT_MODEL"', agent)
        self.assertNotIn("git push", self.wrapper)
        self.assertNotIn("gh pr", self.wrapper)
        self.assertNotIn("CANARY_REPAIR_TOKEN:?", self.wrapper)

    def test_agent_cannot_change_harness_or_commit(self) -> None:
        guard = self.wrapper[
            self.wrapper.index("assert_agent_control_unchanged() {") : self.wrapper.index("run_prepare() {")
        ]
        self.assertIn('git rev-parse HEAD', guard)
        self.assertIn('git symbolic-ref -q HEAD', guard)
        self.assertIn("git config --list --show-origin", guard)
        self.assertIn("git status --porcelain=v1 --untracked-files=all", guard)
        for path in (
            ".github",
            ".agents",
            "scripts",
            ".gitattributes",
            "ci/ci.md",
            "ci/llama-canary/agent-repair-prompt.md",
            "ci/llama-canary/family-certified.json",
            "docs/skippy/llama-parity-candidates.json",
        ):
            self.assertIn(path, guard)

    def test_protected_status_detects_untracked_python_startup_hook(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            (repo / "scripts").mkdir()
            (repo / "scripts" / "sitecustomize.py").write_text("raise SystemExit(0)\n")
            result = subprocess.run(
                ["git", "status", "--porcelain=v1", "--untracked-files=all", "--", "scripts"],
                cwd=repo,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn("?? scripts/sitecustomize.py", result.stdout)

    def test_failure_paths_do_not_publish(self) -> None:
        main = self.wrapper[self.wrapper.index("write_repair_pin\n") :]
        self.assertIn("agent task failed or timed out; no canary branch or pull request was published", main)
        self.assertIn("final canary verification failed; no canary branch or pull request was published", main)
        self.assertNotIn("git push", main)

    def test_certified_commit_is_bound_to_verified_tree(self) -> None:
        finalize = self.wrapper[
            self.wrapper.index("finalize_certified_tree() {") : self.wrapper.index("write_repair_pin\n")
        ]
        self.assertIn('BRANCH="llama-canary/repair-${RUN_KEY}-${UPSTREAM_SHA:0:10}"', self.wrapper)
        snapshot = self.wrapper[
            self.wrapper.index("snapshot_candidate_tree() {") : self.wrapper.index("run_prepare() {")
        ]
        self.assertIn('VERIFICATION_TREE="$(git write-tree)"', snapshot)
        self.assertIn("git commit-tree", snapshot)
        materialize = self.wrapper[
            self.wrapper.index("materialize_verification_tree() {") : self.wrapper.index("run_prepare() {")
        ]
        self.assertIn("core.hooksPath=/dev/null", materialize)
        self.assertIn('worktree add --detach "$VERIFY_ROOT" "$CERTIFIED_SHA"', materialize)
        self.assertIn('git rev-parse "${CERTIFIED_SHA}^{tree}"', finalize)
        self.assertIn("certified commit tree changed after final verification", finalize)
        self.assertIn("git status --porcelain --untracked-files=no", finalize)
        self.assertIn("branch=$BRANCH", finalize)
        self.assertIn("head=$CERTIFIED_SHA", finalize)
        self.assertIn("pr_body=$PR_BODY", finalize)
        self.assertIn("git -C \"$TRUSTED_ROOT\" bundle create", finalize)
        self.assertIn("candidate_bundle=$BUNDLE", finalize)

        main = self.wrapper[self.wrapper.index("write_repair_pin\n") :]
        self.assertLess(main.index("snapshot_candidate_tree"), main.index("materialize_verification_tree"))
        self.assertLess(main.index("materialize_verification_tree"), main.index("run_prepare"))
        self.assertLess(main.index("run_certification"), main.index("finalize_certified_tree"))

    def test_verify_mode_restores_tree_identity_from_candidate_commit(self) -> None:
        loader = self.wrapper[
            self.wrapper.index("load_candidate_bundle() {") : self.wrapper.index("cleanup_verification_worktree() {")
        ]
        self.assertIn('CERTIFIED_SHA="$expected_head"', loader)
        self.assertIn('VERIFICATION_TREE="$(git rev-parse "${CERTIFIED_SHA}^{tree}")"', loader)
        self.assertLess(loader.index('CERTIFIED_SHA="$expected_head"'), loader.index("VERIFICATION_TREE="))

    def test_publisher_pushes_only_exact_ready_commit(self) -> None:
        self.assertIn('TOKEN="${CANARY_REPAIR_TOKEN:?CANARY_REPAIR_TOKEN is required}"', self.publisher)
        self.assertIn('BUNDLE="${CANARY_BUNDLE:?CANARY_BUNDLE is required}"', self.publisher)
        self.assertIn("git bundle verify", self.publisher)
        self.assertIn("git bundle list-heads", self.publisher)
        self.assertIn('git rev-parse HEAD', self.publisher)
        self.assertIn("git status --porcelain", self.publisher)
        self.assertIn('GIT_ASKPASS="$ASKPASS"', self.publisher)
        self.assertIn('"HEAD:refs/heads/${BRANCH}"', self.publisher)
        self.assertNotIn("--force", self.publisher)
        self.assertIn("gh pr create", self.publisher)
        self.assertNotIn("--draft", self.publisher)
        self.assertIn('"$remote_head" != "$CERTIFIED_SHA"', self.publisher)
        self.assertIn("cleanup_exact_remote_branch", self.publisher)
        self.assertIn("trap 'cleanup_before_pr $?\' EXIT", self.publisher)
        self.assertLess(self.publisher.index("git push"), self.publisher.index("gh pr create"))
        terminal = self.publisher[self.publisher.index("gh pr create") :]
        self.assertNotIn("gh pr view", terminal)
        self.assertIn("find_exact_ready_pr", terminal)
        self.assertIn("published=1", terminal)

    def test_final_verification_clears_agent_native_products_and_forces_smokes(self) -> None:
        materialize = self.wrapper[
            self.wrapper.index("materialize_verification_tree() {") : self.wrapper.index("run_prepare() {")
        ]
        self.assertIn('rm -rf "$LLAMA_STAGE_BUILD_DIR"', materialize)
        self.assertIn('"$ROOT/target/family-battery/$FAMILY_BATTERY_RUN_ID"', materialize)
        self.assertIn('"$ROOT/target/skippy-stage-rewriter-check"', materialize)
        build = self.wrapper[
            self.wrapper.index("run_full_build() {") : self.wrapper.index("run_certification() {")
        ]
        certify = self.wrapper[
            self.wrapper.index("run_certification() {") : self.wrapper.index("write_upstream_summary() {")
        ]
        self.assertIn("scripts/skippy-ci-smoke.sh", build)
        self.assertIn("scripts/skippy-canary-live-matrix.sh --prepare", certify)
        self.assertNotIn("LLAMA_UPSTREAM_CANARY_SMOKE", build + certify)

    def test_agent_runbook_describes_complete_developer_task(self) -> None:
        runbook = RUNBOOK.read_text(encoding="utf-8")
        self.assertIn("# llama.cpp changed-pin canary developer task", runbook)
        self.assertIn("scripts/prepare-llama.sh pinned", runbook)
        self.assertIn("Run the canonical path repeatedly until it is green", runbook)
        self.assertIn("Leave the finished changes uncommitted", runbook)
        self.assertIn("do not add Actions caching or download logic", runbook)
        for obsolete in ("failed phase", "agent turn", "uncertified draft", "terminal publication"):
            self.assertNotIn(obsolete, runbook)

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
        for script in (WRAPPER, PUBLISHER):
            with self.subTest(script=script.name):
                result = subprocess.run(
                    ["bash", "-n", str(script)], capture_output=True, text=True, check=False
                )
                self.assertEqual(0, result.returncode, result.stderr)

    def test_persistent_runner_scratch_is_scoped_and_pruned(self) -> None:
        self.assertIn('STATE_DIR="$ROOT/.deps/llama-canary-state-${RUN_KEY}"', self.wrapper)
        self.assertIn('TARGET_SHA_FILE="$ROOT/.deps/llama-canary-target-sha"', self.wrapper)
        self.assertIn('git -C "$ROOT/.deps/llama.cpp" worktree prune', self.wrapper)
        self.assertIn("rm -rf /tmp/llama-old-pin /tmp/llama-repair /tmp/llama-repair-*", self.wrapper)

    def test_runnable_row_carrying_unsupported_reason_is_rejected(self) -> None:
        parity = ROOT / "scripts" / "skippy-llama-parity.py"
        sys.path.insert(0, str(parity.parent))
        try:
            spec = importlib.util.spec_from_file_location("skippy_llama_parity_validate", parity)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
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
