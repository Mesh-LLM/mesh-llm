"""`just` runs recipe lines under `sh`, not `bash`.

Recipes that are a plain (non-shebang) body are executed by just's default
shell, which is `sh -cu`. On any host where `/bin/sh` is dash — stock
Debian/Ubuntu, including the aarch64 CUDA build targets — bash-only syntax
does not fail loudly, it takes the wrong branch and keeps going.

A recipe that needs bash must say so with a `#!/usr/bin/env bash` shebang,
which makes just write the body to a file and execute it directly.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Final
import unittest

from scripts.tests.justfile_source import read_justfile_source


ROOT: Final = Path(__file__).resolve().parents[2]
JUSTFILE: Final = ROOT / "Justfile"

# Constructs dash does not implement. Each maps to the failure it produces
# under `sh` so a future reader knows why the entry is here.
BASHISMS: Final = {
    r"\[\[": "[[ ]] test — dash: '[[: not found', then takes the else branch",
    r"<<<": "here-string — dash: 'Syntax error: redirection unexpected'",
    r"\bpipefail\b": "set -o pipefail — dash: 'set: Illegal option -o pipefail'",
    r"\becho -e\b": "echo -e — dash's echo has no -e flag, prints the flag literally",
    r"\$\{[A-Za-z_][A-Za-z0-9_]*\[[@*]\]\}": "array expansion — dash has no arrays",
    r"\bfunction\s+[A-Za-z_]": "`function` keyword — dash only accepts name() { }",
    r"&>": "&> redirect — dash parses this as `&` (background) then `>`",
}

# `$$` is a literal dollar-dollar to just (verified: `just` 1.58 does not
# collapse it), so under `sh` it expands to the shell PID. `$$name` therefore
# silently becomes "<pid>name" rather than the value of `$name`.
DOLLAR_DOLLAR: Final = re.compile(r"\$\$[A-Za-z_{]")

# The [windows] `with-lld` recipe passes a PowerShell script containing `$$`
# sigils through the default shell. It is Windows-only, is not exercised by
# this repo's CI, and its correct form depends on which shell just selects on
# Windows — a separate question from the dash portability this file covers.
# Tracked separately rather than silently swept in with the Linux fixes.
DOLLAR_DOLLAR_EXEMPT: Final = {"with-lld"}


def default_shell_recipes() -> dict[str, str]:
    """Return {recipe_name: body} for recipes just runs under its default shell.

    Recipes whose first non-blank body line is a `#!` shebang are script
    recipes — just executes those with the interpreter they name, so bash
    syntax in them is correct by construction and they are excluded here.
    """
    lines = read_justfile_source(JUSTFILE).splitlines()
    header = re.compile(r"^([a-zA-Z_][\w-]*)(?:\s+[^:]*)?:(?!=)")
    recipes: dict[str, str] = {}
    index = 0
    while index < len(lines):
        match = header.match(lines[index])
        if not match:
            index += 1
            continue
        body: list[str] = []
        cursor = index + 1
        while cursor < len(lines) and (not lines[cursor].strip() or lines[cursor][0].isspace()):
            body.append(lines[cursor])
            cursor += 1
        populated = [line for line in body if line.strip()]
        if populated and not populated[0].strip().startswith("#!"):
            recipes[match.group(1)] = "\n".join(body)
        index = cursor
    return recipes


class JustfileShellPortabilityTests(unittest.TestCase):
    def test_default_shell_recipes_contain_no_bash_only_syntax(self) -> None:
        recipes = default_shell_recipes()
        self.assertGreater(len(recipes), 10, "recipe parser matched suspiciously few recipes")

        offenders: list[str] = []
        for name, body in sorted(recipes.items()):
            for pattern, reason in BASHISMS.items():
                if re.search(pattern, body):
                    offenders.append(f"{name}: {pattern} — {reason}")
        self.assertEqual(
            offenders,
            [],
            "these recipes run under `sh` but use bash-only syntax; either add a "
            "`#!/usr/bin/env bash` shebang to make them script recipes, or rewrite "
            "them in POSIX shell:\n  " + "\n  ".join(offenders),
        )

    def test_default_shell_recipes_do_not_read_variables_through_dollar_dollar(self) -> None:
        offenders = [
            name
            for name, body in sorted(default_shell_recipes().items())
            if name not in DOLLAR_DOLLAR_EXEMPT and DOLLAR_DOLLAR.search(body)
        ]
        self.assertEqual(
            offenders,
            [],
            "`$$name` expands to the shell PID followed by the literal text "
            f"`name`, not to `$name`: {offenders}",
        )

    def test_the_recipe_parser_actually_sees_script_recipes(self) -> None:
        """Guard against the parser silently matching nothing and passing."""
        recipes = default_shell_recipes()
        self.assertNotIn("release-build-cuda", recipes, "script recipe leaked into the sh set")
        self.assertIn("llama-prepare", recipes, "plain recipe missing from the sh set")

    def test_arithmetic_expansion_is_not_flagged_as_a_bashism(self) -> None:
        """$(( )) is POSIX arithmetic expansion; dash implements it natively.

        A recipe using it is portable under the default shell and must not be
        forced into an unneeded `#!/usr/bin/env bash` shebang.
        """
        body = 'attempt="$((attempt + 1))"\necho "$attempt"'
        offenders = [pattern for pattern in BASHISMS if re.search(pattern, body)]
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
