from __future__ import annotations

from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "rc-release-smoke.sh"


class RcReleaseSmokeScriptTests(unittest.TestCase):
    def test_script_parses_as_bash(self) -> None:
        result = subprocess.run(
            ["bash", "-n", str(SCRIPT)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(result.returncode, 0, result.stderr)

    def test_script_documents_required_green_path_steps(self) -> None:
        contents = SCRIPT.read_text(encoding="utf-8")

        for marker in (
            "runtime list --available",
            "runtime install",
            "runtime list",
            "download",
            "/v1/models",
            "/v1/chat/completions",
            'content.strip() == "rc-ok"',
            "verify no mesh-llm process remains",
        ):
            self.assertIn(marker, contents)


if __name__ == "__main__":
    unittest.main()
