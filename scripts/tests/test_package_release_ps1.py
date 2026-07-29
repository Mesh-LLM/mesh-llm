from __future__ import annotations

from pathlib import Path
from typing import Final
import unittest


ROOT: Final = Path(__file__).resolve().parents[2]
SCRIPT: Final = ROOT / "scripts" / "package-release.ps1"


class PackageReleasePowerShellTests(unittest.TestCase):
    def test_selector_arguments_are_built_as_tokens_with_optional_cuda_major(self) -> None:
        selector = self.selector_block()

        self.assertIn("$selectorArgs = @(", selector)
        self.assertIn("if (Test-HasValue $cudaMajor)", selector)
        self.assertIn('$selectorArgs += @("--cuda-major", $cudaMajor)', selector)
        self.assertNotIn("--cuda-major $cudaMajor", selector)

    def test_selector_exit_code_is_checked_before_output_normalization(self) -> None:
        selector = self.selector_block()

        exit_code_index = selector.index("$selectorExitCode = $LASTEXITCODE")
        normalize_index = selector.index("$runtimeDir =")
        self.assertLess(exit_code_index, normalize_index)
        self.assertIn("if ($selectorExitCode -ne 0)", selector)

    def test_selector_uses_last_nonempty_trimmed_output_line(self) -> None:
        selector = self.selector_block()

        self.assertIn("Select-Object -Last 1", selector)
        self.assertIn("ForEach-Object { $_.Trim() }", selector)
        self.assertNotIn(").Trim()", selector)

    def selector_block(self) -> str:
        contents = SCRIPT.read_text(encoding="utf-8")
        start = contents.index("$cudaMajor =")
        end = contents.index("$runtimeDestinationRoot =", start)
        return contents[start:end]


if __name__ == "__main__":
    unittest.main()
