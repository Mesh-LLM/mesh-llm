from __future__ import annotations

from pathlib import Path
import re
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]
ACTION_PATH = ROOT / ".github" / "actions" / "compute-changes" / "action.yml"
DERIVE_SCRIPT_PATH = (
    ROOT / ".github" / "actions" / "compute-changes" / "derive-outputs.sh"
)
MAX_RUN_EXPRESSION_LENGTH = 21_000

# GitHub evaluates the complete run scalar after replacing template
# expressions. Use the longest supported event name and full-length SHA values
# so this contract catches the boundary before a real planner run does.
_INPUT_VALUES = {
    "inputs.event_name": "workflow_dispatch",
    "inputs.base_sha": "a" * 40,
    "inputs.head_sha": "b" * 40,
}
_EXPRESSION_RE = re.compile(r"\$\{\{\s*([^{}]+?)\s*\}\}")


def _expanded_run(run_text: str) -> tuple[str, list[str]]:
    unknown: list[str] = []

    def replace(match: re.Match[str]) -> str:
        expression = match.group(1).strip()
        replacement = _INPUT_VALUES.get(expression)
        if replacement is None:
            unknown.append(expression)
            return match.group(0)
        return replacement

    return _EXPRESSION_RE.sub(replace, run_text), unknown


class ComputeChangesExpressionLimitTests(unittest.TestCase):
    """Keep composite-action run scalars below GitHub's template limit.

    GitHub rejects a composite action before any job starts when an expanded
    ``run`` scalar exceeds 21,000 characters. That startup failure is invisible
    to actionlint, so this test measures the exact checked-in action step with
    full-length SHA inputs and reports the step's source/expanded lengths.
    """

    def test_compute_changes_run_scalars_stay_below_github_limit(self) -> None:
        document = yaml.safe_load(ACTION_PATH.read_text(encoding="utf-8"))
        steps = document.get("runs", {}).get("steps", [])
        self.assertIsInstance(steps, list)

        derive_step = next(
            (step for step in steps if step.get("id") == "derive"),
            None,
        )
        self.assertIsNotNone(
            derive_step,
            "compute-changes must retain its Derive outputs step",
        )
        self.assertTrue(
            DERIVE_SCRIPT_PATH.is_file(),
            "Derive outputs must keep its large shell body in the sibling script",
        )

        violations: list[str] = []
        for step in steps:
            if not isinstance(step, dict) or not isinstance(step.get("run"), str):
                continue
            run_text = step["run"]
            expanded, unknown = _expanded_run(run_text)
            step_name = step.get("name", step.get("id", "<unnamed step>"))
            if unknown:
                violations.append(
                    f"{step_name!r} contains unmodelled expressions: {', '.join(unknown)}"
                )
            if len(expanded) >= MAX_RUN_EXPRESSION_LENGTH:
                violations.append(
                    f"{step_name!r}: source={len(run_text)} chars, "
                    f"expanded={len(expanded)} chars, "
                    f"limit={MAX_RUN_EXPRESSION_LENGTH} "
                    f"(over by {len(expanded) - MAX_RUN_EXPRESSION_LENGTH})"
                )

        self.assertEqual(
            [],
            violations,
            "Composite-action run scalars must stay below GitHub's 21,000-"
            "character template limit. Keep large shell-owned derivation in a "
            "sibling script and pass bounded inputs through env:\n"
            + "\n".join(violations),
        )


if __name__ == "__main__":
    unittest.main()
