from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
GUARD = ROOT / "scripts" / "check-llama-upstream-pin.py"
QUALITY_LANE = ROOT / ".github" / "workflows" / "ci-quality-lane.yml"
PR_QUALITY = ROOT / ".github" / "workflows" / "pr_quality.yml"
PIN_PATH = Path("third_party/llama.cpp/upstream.txt")


def run_git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        check=False,
    )
    if check and result.returncode != 0:
        raise AssertionError(result.stderr)
    return result


class LlamaUpstreamPinGuardTests(unittest.TestCase):
    def create_upstream(self, root: Path) -> tuple[Path, str, str, str]:
        upstream = root / "llama-upstream"
        upstream.mkdir()
        run_git(upstream, "init", "--quiet")
        run_git(upstream, "config", "user.name", "fixture")
        run_git(upstream, "config", "user.email", "fixture@example.com")
        (upstream / "history.txt").write_text("base\n", encoding="utf-8")
        run_git(upstream, "add", "history.txt")
        run_git(upstream, "commit", "--quiet", "-m", "base")
        base_pin = run_git(upstream, "rev-parse", "HEAD").stdout.strip()
        (upstream / "history.txt").write_text("base\nforward\n", encoding="utf-8")
        run_git(upstream, "commit", "--quiet", "-am", "forward")
        forward_pin = run_git(upstream, "rev-parse", "HEAD").stdout.strip()
        (upstream / "history.txt").write_text("base\nforward\nlatest\n", encoding="utf-8")
        run_git(upstream, "commit", "--quiet", "-am", "latest")
        latest_pin = run_git(upstream, "rev-parse", "HEAD").stdout.strip()
        return upstream, base_pin, forward_pin, latest_pin

    def create_mesh_history(self, root: Path, base_pin: str, proposed_pin: str) -> tuple[Path, str, str]:
        mesh = root / "mesh"
        (mesh / PIN_PATH.parent).mkdir(parents=True)
        run_git(mesh, "init", "--quiet")
        run_git(mesh, "config", "user.name", "fixture")
        run_git(mesh, "config", "user.email", "fixture@example.com")
        (mesh / PIN_PATH).write_text(f"{base_pin}\n", encoding="utf-8")
        run_git(mesh, "add", str(PIN_PATH))
        run_git(mesh, "commit", "--quiet", "-m", "base")
        base_revision = run_git(mesh, "rev-parse", "HEAD").stdout.strip()
        (mesh / PIN_PATH).write_text(f"{proposed_pin}\n", encoding="utf-8")
        (mesh / "change.txt").write_text("proposed\n", encoding="utf-8")
        run_git(mesh, "add", "-A")
        run_git(mesh, "commit", "--quiet", "-m", "propose pin")
        head_revision = run_git(mesh, "rev-parse", "HEAD").stdout.strip()
        return mesh, base_revision, head_revision

    def create_stale_mesh_history(
        self,
        root: Path,
        merge_base_pin: str,
        target_pin: str,
        proposed_pin: str,
    ) -> tuple[Path, str, str]:
        mesh = root / "mesh-stale"
        (mesh / PIN_PATH.parent).mkdir(parents=True)
        run_git(mesh, "init", "--quiet")
        run_git(mesh, "config", "user.name", "fixture")
        run_git(mesh, "config", "user.email", "fixture@example.com")
        (mesh / PIN_PATH).write_text(f"{merge_base_pin}\n", encoding="utf-8")
        run_git(mesh, "add", str(PIN_PATH))
        run_git(mesh, "commit", "--quiet", "-m", "merge base")
        merge_base_revision = run_git(mesh, "rev-parse", "HEAD").stdout.strip()
        run_git(mesh, "branch", "target")

        run_git(mesh, "checkout", "--quiet", "-b", "pr")
        (mesh / PIN_PATH).write_text(f"{proposed_pin}\n", encoding="utf-8")
        (mesh / "pr-change.txt").write_text("pr\n", encoding="utf-8")
        run_git(mesh, "add", "-A")
        run_git(mesh, "commit", "--quiet", "-m", "PR change")
        head_revision = run_git(mesh, "rev-parse", "HEAD").stdout.strip()

        run_git(mesh, "checkout", "--quiet", "target")
        (mesh / PIN_PATH).write_text(f"{target_pin}\n", encoding="utf-8")
        (mesh / "target-change.txt").write_text("target\n", encoding="utf-8")
        run_git(mesh, "add", "-A")
        run_git(mesh, "commit", "--quiet", "-m", "target advances pin")
        base_revision = run_git(mesh, "rev-parse", "HEAD").stdout.strip()
        self.assertEqual(
            merge_base_revision,
            run_git(mesh, "merge-base", base_revision, head_revision).stdout.strip(),
        )
        return mesh, base_revision, head_revision

    def create_symlink_pin_history(
        self, root: Path, merge_base_pin: str, dereferenced_pin: str
    ) -> tuple[Path, str, str]:
        mesh = root / "mesh-symlink"
        (mesh / PIN_PATH.parent).mkdir(parents=True)
        run_git(mesh, "init", "--quiet")
        run_git(mesh, "config", "user.name", "fixture")
        run_git(mesh, "config", "user.email", "fixture@example.com")
        (mesh / PIN_PATH).write_text(f"{merge_base_pin}\n", encoding="utf-8")
        run_git(mesh, "add", str(PIN_PATH))
        run_git(mesh, "commit", "--quiet", "-m", "merge base")
        base_revision = run_git(mesh, "rev-parse", "HEAD").stdout.strip()

        (mesh / PIN_PATH).unlink()
        (mesh / merge_base_pin).write_text(f"{dereferenced_pin}\n", encoding="utf-8")
        os.symlink(merge_base_pin, mesh / PIN_PATH)
        run_git(mesh, "add", "-A")
        run_git(mesh, "commit", "--quiet", "-m", "symlink pin")
        head_revision = run_git(mesh, "rev-parse", "HEAD").stdout.strip()
        return mesh, base_revision, head_revision

    def run_guard(self, mesh: Path, base: str, head: str, upstream: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                "python3",
                str(GUARD),
                "--repository",
                str(mesh),
                "--upstream-url",
                str(upstream),
                base,
                head,
            ],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_equal_pin_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream, base_pin, _, _ = self.create_upstream(root)
            mesh, base, head = self.create_mesh_history(root, base_pin, base_pin)
            result = self.run_guard(mesh, base, head, upstream)
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertIn("unchanged", result.stdout)

    def test_descendant_pin_passes_after_fetching_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream, base_pin, _, latest_pin = self.create_upstream(root)
            mesh, base, head = self.create_mesh_history(root, base_pin, latest_pin)
            result = self.run_guard(mesh, base, head, upstream)
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertIn("moves forward", result.stdout)

    def test_ancestor_pin_fails_as_backward(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream, base_pin, _, latest_pin = self.create_upstream(root)
            mesh, base, head = self.create_mesh_history(root, latest_pin, base_pin)
            result = self.run_guard(mesh, base, head, upstream)
            self.assertNotEqual(0, result.returncode)
            self.assertIn("moves", result.stderr)
            self.assertIn("backward", result.stderr)

    def test_stale_pr_uses_merge_base_pin_when_target_advanced(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream, base_pin, forward_pin, _ = self.create_upstream(root)
            mesh, base, head = self.create_stale_mesh_history(
                root,
                merge_base_pin=base_pin,
                target_pin=forward_pin,
                proposed_pin=base_pin,
            )
            result = self.run_guard(mesh, base, head, upstream)
            self.assertEqual(0, result.returncode, result.stderr)
            self.assertIn("unchanged", result.stdout)

    def test_backward_pin_is_compared_to_merge_base(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream, base_pin, _, latest_pin = self.create_upstream(root)
            mesh, base, head = self.create_stale_mesh_history(
                root,
                merge_base_pin=latest_pin,
                target_pin=latest_pin,
                proposed_pin=base_pin,
            )
            result = self.run_guard(mesh, base, head, upstream)
            self.assertNotEqual(0, result.returncode)
            self.assertIn("backward", result.stderr)

    def test_missing_upstream_pin_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream, base_pin, _, _ = self.create_upstream(root)
            missing_pin = "f" * 40
            mesh, base, head = self.create_mesh_history(root, base_pin, missing_pin)
            result = self.run_guard(mesh, base, head, upstream)
            self.assertNotEqual(0, result.returncode)
            self.assertIn("fail closed", result.stderr)

    def test_symlink_pin_is_rejected_before_dereferencing_sibling_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream, base_pin, _, latest_pin = self.create_upstream(root)
            mesh, base, head = self.create_symlink_pin_history(
                root, merge_base_pin=latest_pin, dereferenced_pin=base_pin
            )
            result = self.run_guard(mesh, base, head, upstream)
            self.assertNotEqual(0, result.returncode)
            self.assertIn("regular 100644 blob", result.stderr)

    def test_workflow_runs_guard_from_protected_checkout(self) -> None:
        workflow = QUALITY_LANE.read_text(encoding="utf-8")
        summary = workflow.split("  summary:\n", 1)[1]
        self.assertIn("ref: ${{ github.event.repository.default_branch }}", summary)
        self.assertIn("fetch-depth: 0", summary)
        self.assertIn("persist-credentials: false", summary)
        self.assertIn("if: ${{ github.event_name == 'pull_request' }}", summary)
        self.assertIn('git fetch --no-tags origin "$BASE_SHA" "$HEAD_SHA"', summary)
        self.assertIn(
            'python3 scripts/check-llama-upstream-pin.py "$BASE_SHA" "$HEAD_SHA"',
            summary,
        )
        self.assertNotIn("ref: ${{ inputs.source_sha || github.sha }}", summary)

    def test_pr_base_retargeting_retriggers_the_guard(self) -> None:
        workflow = PR_QUALITY.read_text(encoding="utf-8")
        self.assertIn(
            "types: [opened, synchronize, reopened, ready_for_review, edited]",
            workflow,
        )
        self.assertIn(
            "BASE_SHA: ${{ github.event.pull_request.base.sha }}",
            QUALITY_LANE.read_text(encoding="utf-8"),
        )
        self.assertIn(
            "HEAD_SHA: ${{ github.event.pull_request.head.sha }}",
            QUALITY_LANE.read_text(encoding="utf-8"),
        )


if __name__ == "__main__":
    unittest.main()
