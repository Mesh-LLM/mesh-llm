"""Contract for `scripts/lib/lld.sh`.

lld is a build-speed optimization, so an lld that cannot link must degrade to
the platform linker with a stated reason rather than failing a build that
would otherwise succeed. These tests drive the shell functions directly with
a stub `cc`, so they do not depend on what is installed on the test machine.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from justfile_source import read_justfile_source

ROOT = Path(__file__).resolve().parents[2]
LIB = ROOT / "scripts" / "lib" / "lld.sh"

WORKING_CC = "#!/bin/sh\nexit 0\n"
# Mirrors the real failure shape: lld rejects the SDK's text-based stub and
# every libSystem symbol then comes back undefined.
BROKEN_CC = (
    "#!/bin/sh\n"
    "echo 'ld64.lld: error: could not load TAPI file at "
    "/SDK/usr/lib/libSystem.tbd: malformed file' >&2\n"
    "exit 1\n"
)


def run_with_stub_cc(cc_body: str, snippet: str) -> subprocess.CompletedProcess[str]:
    """Source the library with a stub `cc` first on PATH, then run `snippet`."""
    with tempfile.TemporaryDirectory() as stub_dir:
        cc = Path(stub_dir) / "cc"
        cc.write_text(cc_body, encoding="utf-8")
        cc.chmod(0o755)
        environment = dict(os.environ, PATH=f"{stub_dir}{os.pathsep}{os.environ['PATH']}")
        return subprocess.run(
            ["bash", "-c", f'set -euo pipefail\nsource "{LIB}"\n{snippet}'],
            capture_output=True,
            text=True,
            env=environment,
            check=False,
        )


class LldProbeTests(unittest.TestCase):
    def test_a_linker_that_links_is_usable(self) -> None:
        result = run_with_stub_cc(
            WORKING_CC, 'lld_is_usable lld && echo USABLE || echo UNUSABLE'
        )
        self.assertEqual(result.stdout.strip(), "USABLE", result.stderr)

    def test_a_linker_that_cannot_link_is_not_usable(self) -> None:
        result = run_with_stub_cc(
            BROKEN_CC, 'lld_is_usable lld && echo USABLE || echo UNUSABLE'
        )
        self.assertEqual(result.stdout.strip(), "UNUSABLE", result.stderr)

    def test_the_probe_captures_the_linker_diagnostics(self) -> None:
        """The report has to state the real reason, not guess at one."""
        result = run_with_stub_cc(
            BROKEN_CC, 'lld_is_usable lld || printf %s "$LLD_PROBE_OUTPUT"'
        )
        self.assertIn("could not load TAPI file", result.stdout)

    def test_resolve_prints_nothing_and_reports_when_the_probe_fails(self) -> None:
        result = run_with_stub_cc(BROKEN_CC, 'printf "[%s]" "$(resolve_usable_lld)"')
        self.assertEqual(result.stdout.strip(), "[]")
        self.assertIn("cannot link against the active SDK", result.stderr)
        self.assertIn("using the platform default linker", result.stderr)
        self.assertIn("could not load TAPI file", result.stderr)

    def test_resolve_prints_the_linker_when_the_probe_passes(self) -> None:
        result = run_with_stub_cc(WORKING_CC, 'resolve_usable_lld')
        self.assertNotEqual(result.stdout.strip(), "", result.stderr)

    def test_an_unusable_linker_never_fails_the_caller(self) -> None:
        """`set -e` callers must not be killed by a failed probe."""
        result = run_with_stub_cc(
            BROKEN_CC, 'resolve_usable_lld >/dev/null\necho SURVIVED'
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("SURVIVED", result.stdout)


class WithLldRecipeTests(unittest.TestCase):
    """`just with-lld` must probe, not assume."""

    def setUp(self) -> None:
        self.source = read_justfile_source(ROOT / "justfile")
        start = self.source.index("with-lld *COMMAND:")
        end = self.source.index("with-lld *COMMAND:", start + 1)
        self.recipe = self.source[start:end]

    def test_the_recipe_probes_before_exporting_the_linker(self) -> None:
        self.assertIn('source scripts/lib/lld.sh', self.recipe)
        self.assertIn('lld="$(find_lld)"', self.recipe)
        self.assertIn('if lld_is_usable "$lld"; then', self.recipe)
        self.assertIn('report_unusable_lld "$lld"', self.recipe)

    def test_a_missing_linker_is_still_a_hard_error(self) -> None:
        """Not installed stays fatal with install instructions; only an
        installed-but-unusable linker degrades to a warning."""
        self.assertIn("brew install lld", self.recipe)
        self.assertIn("exit 1", self.recipe)


class CargoConfigTests(unittest.TestCase):
    def test_no_hardcoded_linker_path_survives_in_cargo_config(self) -> None:
        """A checked-in absolute `-fuse-ld=` applies to every cargo
        invocation with no way to check the linker first, and breaks
        `cargo build`/`cargo test` while leaving `cargo check` green."""
        config = (ROOT / ".cargo" / "config.toml").read_text(encoding="utf-8")
        directives = [
            line
            for line in config.splitlines()
            if "fuse-ld" in line and not line.lstrip().startswith("#")
        ]
        self.assertEqual(directives, [], f"unexpected linker directives: {directives}")


if __name__ == "__main__":
    unittest.main()
