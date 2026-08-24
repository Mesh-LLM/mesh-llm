from __future__ import annotations

import re
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RELEASE_VERSION_SCRIPT = ROOT / "scripts" / "release-version.sh"


def known_versions_file() -> Path:
    script = RELEASE_VERSION_SCRIPT.read_text(encoding="utf-8")
    match = re.search(
        r'^known_versions_file="\$REPO_ROOT/(?P<path>[^"]+)"$',
        script,
        re.MULTILINE,
    )
    assert match is not None, "release-version.sh no longer assigns known_versions_file"
    return ROOT / match.group("path")


# Run update_known_mesh_versions in isolation. The script cannot be sourced --
# it validates its own arguments and performs a full release bump at load time
# -- so extract just the function definition and evaluate that.
_INVOKE_HELPER = """
set -euo pipefail
eval "$(sed -n '/^update_known_mesh_versions()/,/^}/p' "$1")"
update_known_mesh_versions "$2" "$3"
"""

class ReleaseVersionScriptTests(unittest.TestCase):
    def test_known_versions_file_defines_the_function_the_script_edits(self) -> None:
        """The script rewrites known_mesh_llm_versions() by regex.

        If the function moves to another module the substitution silently
        matches nothing and the release fails in the metadata job, after the
        tag has been chosen. Keep the path and the definition together.
        """
        target = known_versions_file()
        self.assertTrue(target.is_file(), f"missing {target}")
        self.assertIn(
            "fn known_mesh_llm_versions()",
            target.read_text(encoding="utf-8"),
        )

    def test_update_known_mesh_versions_prepends_the_new_version(self) -> None:
        source = known_versions_file().read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory() as tmp:
            sample = Path(tmp) / "setting_schema.rs"
            sample.write_text(source, encoding="utf-8")
            subprocess.run(
                [
                    "bash",
                    "-c",
                    _INVOKE_HELPER,
                    "bash",
                    str(RELEASE_VERSION_SCRIPT),
                    str(sample),
                    "99.99.99-rc1",
                ],
                check=True,
                capture_output=True,
            )
            updated = sample.read_text(encoding="utf-8")
        self.assertIn('"99.99.99-rc1",', updated)

    def test_update_known_mesh_versions_fails_loudly_on_a_wrong_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sample = Path(tmp) / "elsewhere.rs"
            sample.write_text("// no version list here\n", encoding="utf-8")
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    _INVOKE_HELPER,
                    "bash",
                    str(RELEASE_VERSION_SCRIPT),
                    str(sample),
                    "99.99.99-rc1",
                ],
                capture_output=True,
                text=True,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("known_mesh_llm_versions()", result.stderr)


if __name__ == "__main__":
    unittest.main()
