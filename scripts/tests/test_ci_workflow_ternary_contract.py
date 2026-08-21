from __future__ import annotations

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = ROOT / ".github" / "workflows"

_EXPRESSION_RE = re.compile(r"\$\{\{(.*?)\}\}", re.DOTALL)

# `cond && a || b` is GitHub Actions' idiom for a ternary, built out of
# JS-style short-circuit `&&`/`||`: `&&` returns its second operand only
# when the first is truthy, `||` returns its first operand only when it's
# truthy. That means the `&&` branch (`a`) must never itself be falsy --
# if it is, `cond && a` collapses to the falsy `a`, and the trailing `||`
# then overrides it with `b` unconditionally, regardless of `cond`. The
# three literals that can trigger this are '', "", 0, and false.
_FALSY_AND_BRANCH_RE = re.compile(r"&&\s*(''|\"\"|0|false)\s*\|\|")


class CiWorkflowTernaryContractTests(unittest.TestCase):
    """Catches the `cond && <falsy-literal> || fallback` pitfall in workflow
    expressions: a ternary written with `&&`/`||` where the "true" branch is
    itself a falsy literal always evaluates to the fallback, never the
    intended value. See smoke.yml's `container: image:` line for the case
    this caught in production -- the `gpu-nvidia` branch of that ternary
    (`inputs.runner == 'gpu-nvidia' && '' || url`) could never actually
    produce an empty image."""

    def test_no_ternary_has_a_falsy_literal_in_the_and_branch(self) -> None:
        violations = []
        for path in sorted(WORKFLOWS_DIR.glob("*.yml")):
            text = path.read_text(encoding="utf-8")
            for match in _EXPRESSION_RE.finditer(text):
                expression = match.group(1)
                if _FALSY_AND_BRANCH_RE.search(expression):
                    line = text.count("\n", 0, match.start()) + 1
                    violations.append(
                        f"{path.name}:{line}: {expression.strip()!r} -- the `&&` "
                        "branch is a falsy literal, so `||` always overrides it; "
                        "put the non-empty/truthy value first instead."
                    )

        self.assertEqual(
            [],
            violations,
            "Workflow ternaries must not put a falsy literal ('', \"\", 0, false) "
            "in the `&&` branch -- it makes the `||` fallback unconditional:\n"
            + "\n".join(violations),
        )


if __name__ == "__main__":
    unittest.main()
