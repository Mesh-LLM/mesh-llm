from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = ROOT / ".github" / "workflows"

_RESOLVE_STEP = "Resolve planned batch crates against the checked-out workspace"

# Both slices plan their Rust batches on the default branch but build the
# revision under test. A branch that predates a workspace member added on the
# default branch must not have that member handed to Cargo, or the whole batch
# dies with "package ID specification <crate> did not match any packages". Each
# workflow resolves the planned batch against the checked-out workspace once and
# every package-specific step consumes that resolved list.
_BATCH_STEPS = (
    ("ci-quality-slice.yml", _RESOLVE_STEP, "Run one Clippy invocation for the batch"),
    ("ci-rust-tests-slice.yml", _RESOLVE_STEP, "Run isolated Cargo tests for the batch"),
)


def _bash_binary() -> str | None:
    """A bash able to run the slices' batch scripts (they use `mapfile`)."""
    candidates = [shutil.which("bash"), "/opt/homebrew/bin/bash"]
    for candidate in candidates:
        if not candidate or not Path(candidate).exists():
            continue
        probe = subprocess.run(
            [candidate, "-c", 'echo "${BASH_VERSINFO[0]}"'],
            capture_output=True,
            text=True,
            check=False,
        )
        major = probe.stdout.strip()
        if probe.returncode == 0 and major.isdigit() and int(major) >= 4:
            return candidate
    return None


BASH = _bash_binary()


def _run_script(workflow: str, step_name: str) -> str:
    workflow_data = yaml.safe_load((WORKFLOWS_DIR / workflow).read_text(encoding="utf-8"))
    for job in workflow_data["jobs"].values():
        for step in job.get("steps", []) or []:
            if step.get("name") == step_name:
                script = step.get("run")
                if not isinstance(script, str):
                    raise AssertionError(f"{workflow}: {step_name!r} has no run script")
                return script
    raise AssertionError(f"{workflow}: no step named {step_name!r}")


def _read_github_output(path: Path) -> dict[str, str]:
    outputs: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            outputs[key] = value
    return outputs


def _stub_cargo(root: Path, members: list[str], metadata_fails: bool) -> Path:
    """A fake Cargo that records its invocations and reports a fixed workspace."""
    bindir = root / "bin"
    bindir.mkdir(parents=True, exist_ok=True)
    stub = bindir / "cargo"
    payload = json.dumps({"packages": [{"name": name} for name in members]})
    stub.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf \'%s\\n\' "$*" >> "$STUB_CARGO_LOG"\n'
        'if [[ "${1:-}" == "metadata" ]]; then\n'
        "  if [[ "
        + ('"1"' if metadata_fails else '"0"')
        + ' == "1" ]]; then\n'
        '    echo "stub: metadata unavailable" >&2\n'
        "    exit 101\n"
        "  fi\n"
        f"  printf '%s\\n' {json.dumps(payload)}\n"
        "  exit 0\n"
        "fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    return bindir


class BatchCrateFilterTest(unittest.TestCase):
    maxDiff = None

    def _execute(
        self,
        workflow: str,
        requested: list[str],
        *,
        members: list[str] | None = None,
        metadata_fails: bool = False,
    ) -> tuple[subprocess.CompletedProcess[str], subprocess.CompletedProcess[str], list[str]]:
        """Run the resolve step then the batch step, as the workflow does."""
        resolve_step, batch_step = self._steps(workflow)
        resolve_script = _run_script(workflow, resolve_step)
        batch_script = _run_script(workflow, batch_step)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bindir = _stub_cargo(root, members or [], metadata_fails)
            log = root / "cargo.log"
            log.write_text("", encoding="utf-8")
            runner_temp = root / "runner-temp"
            runner_temp.mkdir()
            github_output = root / "github-output.txt"
            github_output.write_text("", encoding="utf-8")
            env = dict(os.environ)
            env.update(
                {
                    "PATH": f"{bindir}{os.pathsep}{env['PATH']}",
                    "STUB_CARGO_LOG": str(log),
                    "RUNNER_TEMP": str(runner_temp),
                    "GITHUB_OUTPUT": str(github_output),
                    "PLANNED_BATCH_CRATES": json.dumps(requested),
                }
            )
            resolved = subprocess.run(
                [BASH, "-c", resolve_script],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            outputs = _read_github_output(github_output)
            resolved_crates = outputs.get("crates", json.dumps(requested))
            env["CLIPPY_CRATES"] = resolved_crates
            env["TEST_CRATES"] = resolved_crates
            completed = subprocess.run(
                [BASH, "-c", batch_script],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            return resolved, completed, log.read_text(encoding="utf-8").splitlines()

    @staticmethod
    def _steps(workflow: str) -> tuple[str, str]:
        for candidate, resolve_step, batch_step in _BATCH_STEPS:
            if candidate == workflow:
                return resolve_step, batch_step
        raise AssertionError(f"no batch steps declared for {workflow}")

    @staticmethod
    def _executed_batches(calls: list[str]) -> str:
        """Everything the script asked Cargo to build, excluding `metadata`."""
        return "\n".join(call for call in calls if not call.startswith("metadata"))

    @unittest.skipUnless(BASH, "needs bash >= 4 for mapfile")
    def test_batch_resolves_against_the_checked_out_workspace(self) -> None:
        for workflow, _, _ in _BATCH_STEPS:
            with self.subTest(workflow=workflow):
                resolved, completed, calls = self._execute(
                    workflow,
                    ["mesh-llm-analytics", "mesh-llm-host-runtime"],
                    members=["mesh-llm", "mesh-llm-host-runtime"],
                )
                self.assertEqual(resolved.returncode, 0, resolved.stderr)
                self.assertRegex(resolved.stdout, r"::warning::.*mesh-llm-analytics")
                self.assertEqual(completed.returncode, 0, completed.stderr)
                executed = self._executed_batches(calls)
                self.assertNotIn("mesh-llm-analytics", executed)
                self.assertIn("mesh-llm-host-runtime", executed)

    @unittest.skipUnless(BASH, "needs bash >= 4 for mapfile")
    def test_all_absent_batch_skips_cargo(self) -> None:
        for workflow, _, _ in _BATCH_STEPS:
            with self.subTest(workflow=workflow):
                resolved, completed, calls = self._execute(
                    workflow,
                    ["mesh-llm-analytics", "mesh-llm-not-a-member"],
                    members=["mesh-llm", "mesh-llm-host-runtime"],
                )
                self.assertEqual(resolved.returncode, 0, resolved.stderr)
                self.assertRegex(resolved.stdout, r"::warning::.*mesh-llm-analytics")
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertEqual(self._executed_batches(calls), "")

    @unittest.skipUnless(BASH, "needs bash >= 4 for mapfile")
    def test_failed_metadata_runs_the_planned_batch_unchanged(self) -> None:
        for workflow, _, _ in _BATCH_STEPS:
            with self.subTest(workflow=workflow):
                resolved, completed, calls = self._execute(
                    workflow,
                    ["mesh-llm-analytics"],
                    metadata_fails=True,
                )
                self.assertEqual(resolved.returncode, 0, resolved.stderr)
                self.assertIn("cargo metadata failed", resolved.stdout)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertIn("mesh-llm-analytics", self._executed_batches(calls))

    @unittest.skipUnless(BASH, "needs bash >= 4 for mapfile")
    def test_present_batch_runs_without_warnings(self) -> None:
        for workflow, _, _ in _BATCH_STEPS:
            with self.subTest(workflow=workflow):
                resolved, completed, calls = self._execute(
                    workflow,
                    ["mesh-llm", "mesh-llm-host-runtime"],
                    members=["mesh-llm", "mesh-llm-host-runtime"],
                )
                self.assertEqual(resolved.returncode, 0, resolved.stderr)
                self.assertNotIn("::warning::", resolved.stdout)
                self.assertEqual(completed.returncode, 0, completed.stderr)
                executed = self._executed_batches(calls)
                self.assertIn("mesh-llm", executed)
                self.assertIn("mesh-llm-host-runtime", executed)

    def test_renamed_batches_are_translated_before_the_workspace_filter(self) -> None:
        """A planned batch is translated first, then filtered by this revision.

        The translation maps a pre-extraction package to its extracted owners;
        the filter drops a name the checked-out tree does not have. Every
        package-specific step consumes the filtered list, so a renamed owner is
        matched and a stale plan cannot fail the lane.
        """
        for workflow, resolve_step, batch_step in _BATCH_STEPS:
            with self.subTest(workflow=workflow):
                data = yaml.safe_load((WORKFLOWS_DIR / workflow).read_text(encoding="utf-8"))
                steps = [step for job in data["jobs"].values() for step in job.get("steps", []) or []]
                translate = next(step for step in steps if step.get("id") == "packages")
                resolve = next(step for step in steps if step.get("name") == resolve_step)
                batch = next(step for step in steps if step.get("name") == batch_step)
                self.assertIn("resolve-cargo-packages", translate["uses"])
                self.assertEqual(
                    resolve["env"]["PLANNED_BATCH_CRATES"],
                    "${{ steps.packages.outputs.crates }}",
                )
                self.assertLess(steps.index(translate), steps.index(resolve))
                self.assertLess(steps.index(resolve), steps.index(batch))
                self.assertIn("steps.resolve_batch_crates.outputs.crates", str(batch))


if __name__ == "__main__":
    unittest.main()
