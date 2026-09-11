"""Contract for `scripts/lib/lld.sh`.

lld is a build-speed optimization, so an lld that cannot link must degrade to
the platform linker with a stated reason rather than fail a build that would
otherwise succeed. The probe is driven with a stub `cc` so these tests do not
depend on what is installed on the machine running them.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.tests.justfile_source import read_justfile_source

ROOT = Path(__file__).resolve().parents[2]
LIB = ROOT / "scripts" / "lib" / "lld.sh"
JUSTFILE = ROOT / "Justfile"
BUILD_HOST = ROOT / "scripts" / "build-host.sh"

WORKING_CC = "#!/bin/sh\nexit 0\n"
# Mirrors the real failure: lld rejects the SDK's text-based stub, after which
# every libSystem symbol comes back undefined.
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
        env = dict(os.environ, PATH=f"{stub_dir}{os.pathsep}{os.environ['PATH']}")
        return subprocess.run(
            ["bash", "-c", f'set -euo pipefail\nsource "{LIB}"\n{snippet}'],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )


class LldProbeTests(unittest.TestCase):
    def test_a_linker_that_links_is_usable(self) -> None:
        result = run_with_stub_cc(WORKING_CC, 'lld_links lld && echo USABLE || echo UNUSABLE')
        self.assertEqual(result.stdout.strip(), "USABLE", result.stderr)

    def test_a_linker_that_cannot_link_is_not_usable(self) -> None:
        result = run_with_stub_cc(BROKEN_CC, 'lld_links lld && echo USABLE || echo UNUSABLE')
        self.assertEqual(result.stdout.strip(), "UNUSABLE", result.stderr)

    def test_the_probe_keeps_the_linker_diagnostics(self) -> None:
        """The fallback note has to state the real reason, not guess one."""
        result = run_with_stub_cc(BROKEN_CC, 'lld_links lld || printf %s "$LLD_PROBE_OUTPUT"')
        self.assertIn("could not load TAPI file", result.stdout)

    def test_the_probe_passes_the_same_fuse_ld_value_cargo_will_get(self) -> None:
        """Probing one linker and exporting another would prove nothing."""
        recording_cc = "#!/bin/sh\necho \"$@\" >&2\nexit 0\n"
        result = run_with_stub_cc(recording_cc, 'lld_links /opt/x/ld64.lld; printf %s "$LLD_PROBE_OUTPUT"')
        self.assertIn("-fuse-ld=/opt/x/ld64.lld", result.stdout)

    def test_resolve_prints_nothing_and_explains_when_the_probe_fails(self) -> None:
        result = run_with_stub_cc(BROKEN_CC, 'printf "[%s]" "$(resolve_usable_lld)"')
        self.assertEqual(result.stdout.strip(), "[]")
        self.assertIn("cannot link against the active SDK", result.stderr)
        self.assertIn("platform default linker", result.stderr)
        self.assertIn("could not load TAPI file", result.stderr)

    def test_resolve_prints_the_linker_when_the_probe_passes(self) -> None:
        # Skip cleanly where no lld is installed: then there is nothing to resolve.
        found = run_with_stub_cc(WORKING_CC, "find_lld")
        if not found.stdout.strip():
            self.skipTest("no lld installed on this machine")
        result = run_with_stub_cc(WORKING_CC, "resolve_usable_lld")
        self.assertEqual(result.stdout.strip(), found.stdout.strip(), result.stderr)
        self.assertEqual(result.stderr, "")

    def test_an_unusable_linker_never_fails_the_caller(self) -> None:
        """`set -e` callers must survive a failed probe."""
        result = run_with_stub_cc(BROKEN_CC, "resolve_usable_lld >/dev/null\necho SURVIVED")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("SURVIVED", result.stdout)


class CallSiteTests(unittest.TestCase):
    """Every place that hands cargo an lld must probe it first."""

    def test_the_with_lld_recipe_probes_before_exporting_the_linker(self) -> None:
        source = read_justfile_source(JUSTFILE)
        start = source.index("[unix]\nwith-lld *COMMAND:")
        end = source.index("[windows]\nwith-lld *COMMAND:", start)
        recipe = source[start:end]
        self.assertIn("source scripts/lib/lld.sh", recipe)
        self.assertIn('lld="$(find_lld)"', recipe)
        self.assertIn('if lld_links "$lld"; then', recipe)
        self.assertIn('report_lld_fallback "$lld"', recipe)
        # Not installed stays fatal with install instructions; only an
        # installed-but-unusable lld degrades to a note.
        self.assertIn("brew install lld", recipe)
        self.assertIn("exit 1", recipe)

    def test_build_host_probes_before_exporting_the_linker(self) -> None:
        script = BUILD_HOST.read_text(encoding="utf-8")
        self.assertIn('source "$SCRIPT_DIR/lib/lld.sh"', script)
        start = script.index("configure_lld_linker() {")
        end = script.index("configure_rust_cache() {", start)
        function = script[start:end]
        self.assertIn('if lld_links "$lld"; then', function)
        self.assertIn('report_lld_fallback "$lld"', function)
        self.assertNotIn("command -v ld64.lld", function)

    def test_no_unprobed_linker_directive_in_cargo_config(self) -> None:
        """A checked-in `-fuse-ld=` applies to every cargo invocation with no
        way to probe first, and breaks `cargo build`/`cargo test` while
        `cargo check` stays green."""
        config = (ROOT / ".cargo" / "config.toml").read_text(encoding="utf-8")
        directives = [
            line for line in config.splitlines()
            if "fuse-ld" in line and not line.lstrip().startswith("#")
        ]
        self.assertEqual(directives, [], f"unexpected linker directives: {directives}")


if __name__ == "__main__":
    unittest.main()
