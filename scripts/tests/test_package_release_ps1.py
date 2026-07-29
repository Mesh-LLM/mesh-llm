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

    def test_prestamped_host_can_use_checksum_verified_prebuilt_verifier(self) -> None:
        contents = SCRIPT.read_text(encoding="utf-8")
        prestamped_start = contents.index(
            'if ($env:MESH_RELEASE_HOST_PRESTAMPED -eq "1") {',
            contents.index("function Invoke-ReleaseAttestationStamp"),
        )
        prestamped_end = contents.index(
            'if (-not (Test-HasValue $attestationSigningKeyFile))',
            prestamped_start,
        )
        prestamped = contents[prestamped_start:prestamped_end]

        self.assertIn(
            "$attestationVerifier = $env:MESH_RELEASE_ATTESTATION_VERIFIER",
            contents,
        )
        self.assertIn(
            'Assert-FileChecksum -Path $attestationVerifier '
            '-ChecksumPath "${attestationVerifier}.sha256"',
            prestamped,
        )
        self.assertIn(
            "& $attestationVerifier release-attestation inspect",
            prestamped,
        )
        self.assertIn(
            "& cargo run -q -p xtask -- release-attestation inspect",
            prestamped,
        )

    def selector_block(self) -> str:
        contents = SCRIPT.read_text(encoding="utf-8")
        start = contents.index("$cudaMajor =")
        end = contents.index("$runtimeDestinationRoot =", start)
        return contents[start:end]


if __name__ == "__main__":
    unittest.main()
