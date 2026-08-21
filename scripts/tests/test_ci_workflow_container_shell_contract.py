from __future__ import annotations

from pathlib import Path
import re
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = ROOT / ".github" / "workflows"

# Bash-only constructs that fail outright under dash/sh, keyed by a short
# human label for the failure message. This is a best-effort sweep, not an
# exhaustive shell grammar -- it exists to catch the same class of mistake
# that already bit this repo twice, not to replace shellcheck.
_BASHISM_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"<<<"), "here-string (<<<)"),
    (re.compile(r"\bpipefail\b"), "set -o pipefail"),
    (re.compile(r"\[\["), "[[ ... ]] test"),
    # Deliberately NOT listed: $(( )) arithmetic expansion. It is POSIX
    # (Shell Command Language 2.6.4) and dash evaluates it correctly, so
    # flagging it would reject valid sh steps and force a spurious
    # `shell: bash`.
    (re.compile(r"\$\{\w+(//|\^\^|,,)"), "parameter expansion (${v//}/${v^^}/${v,,})"),
    (re.compile(r"&>"), "&> redirection"),
    (re.compile(r"\$RANDOM\b"), "$RANDOM"),
    (re.compile(r"(?:^|[;\s])source\s+\S"), "source builtin"),
    (re.compile(r"^\s*function\s+\w+", re.MULTILINE), "function keyword"),
    (re.compile(r"\w+\+=\("), "array += assignment"),
    (re.compile(r"^\s*declare\s+-a\b", re.MULTILINE), "declare -a array"),
    (re.compile(r"\b(?:mapfile|readarray)\b"), "mapfile/readarray"),
    (re.compile(r"\becho\s+-e\b"), "echo -e"),
]


def _load_workflow(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _default_shell(defaults) -> str | None:
    if not isinstance(defaults, dict):
        return None
    run = defaults.get("run")
    if not isinstance(run, dict):
        return None
    shell = run.get("shell")
    return shell.lower() if isinstance(shell, str) else None


class CiWorkflowContainerShellContractTests(unittest.TestCase):
    """GitHub Actions resolves the default `run:` shell inside a `container:`
    job to `sh -e {0}`, not `bash -e {0}` -- bare-metal Linux/macOS runners
    default to bash, so this only changes once a job gains a `container:`
    block. A step using a bashism there fails at runtime with a dash/sh
    syntax error that `actionlint` cannot catch (its shellcheck integration
    assumes bash). This walks every job with a `container:` block and flags
    any `run:` step containing a bashism that has not declared `shell: bash`
    for itself, its job, or its workflow. See ci-web-slice.yml's `ui_e2e`
    preflight (<<<) and website-pages.yml's `Stage Pages artifact`
    (set -euo pipefail) -- both hit this in the same PR."""

    def test_bashisms_in_container_jobs_declare_shell_bash(self) -> None:
        violations = []
        for path in sorted(WORKFLOWS_DIR.glob("*.yml")):
            doc = _load_workflow(path)
            workflow_shell = _default_shell(doc.get("defaults"))
            jobs = doc.get("jobs") or {}
            for job_name, job in jobs.items():
                if not isinstance(job, dict) or not job.get("container"):
                    continue
                job_shell = _default_shell(job.get("defaults")) or workflow_shell
                for step in job.get("steps") or []:
                    if not isinstance(step, dict) or "run" not in step:
                        continue
                    step_shell = step.get("shell")
                    step_shell = step_shell.lower() if isinstance(step_shell, str) else job_shell
                    if step_shell == "bash":
                        continue
                    run_text = step.get("run") or ""
                    step_label = step.get("name", "<unnamed step>")
                    for pattern, label in _BASHISM_PATTERNS:
                        if pattern.search(run_text):
                            violations.append(
                                f"{path.name} :: {job_name} :: {step_label!r} uses "
                                f"{label} but the effective shell in this container "
                                "job is sh, not bash -- add `shell: bash` to the step."
                            )
                            break

        self.assertEqual(
            [],
            violations,
            "Containerized jobs resolve the default `run:` shell to sh, not "
            "bash. These steps use a bashism without declaring `shell: bash`:\n"
            + "\n".join(violations),
        )


if __name__ == "__main__":
    unittest.main()
