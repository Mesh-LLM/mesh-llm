from __future__ import annotations

from pathlib import Path
from typing import Final
import unittest


ROOT: Final = Path(__file__).resolve().parents[2]
SCRIPT: Final = ROOT / "scripts" / "build-windows.ps1"


class BuildWindowsScriptTests(unittest.TestCase):
    def test_native_runtime_package_out_path_is_git_bash_safe(self) -> None:
        script = SCRIPT.read_text(encoding="utf-8")
        package_call = script[script.index('Invoke-NativeCommand "bash" @(') :]
        package_call = package_call[: package_call.index("\n        )")]

        self.assertIn('"--out", "dist/native-runtimes"', package_call)
        self.assertNotIn('"--out", (Join-Path $repoRoot', package_call)


if __name__ == "__main__":
    unittest.main()
