from __future__ import annotations

from pathlib import Path
from typing import Final
import unittest


ROOT: Final = Path(__file__).resolve().parents[2]
SCRIPT: Final = ROOT / "scripts" / "package-release.ps1"


class PackageReleasePowerShellTests(unittest.TestCase):
    def test_precomposed_product_reuses_exact_verified_tree(self) -> None:
        contents = SCRIPT.read_text(encoding="utf-8")
        precomposed = self.function_block(
            contents,
            "Copy-AndVerifyPrecomposedProduct",
            "$Version = Normalize-RecipeArgument",
        )

        self.assertIn(
            "$precomposedProductDir = $env:MESH_LLM_PRECOMPOSED_PRODUCT_DIR",
            contents,
        )
        self.assertIn(
            "$resolvedSourceDir = Resolve-RepositoryPath $SourceDir",
            precomposed,
        )
        self.assertIn(
            "$attestationPreverified",
            contents,
        )
        self.assertIn(
            "[System.IO.Directory]::GetFileSystemEntries($resolvedSourceDir)",
            precomposed,
        )
        self.assertIn(
            'verify-host-dependencies.py"',
            precomposed,
        )
        self.assertIn(
            'verify-native-runtime-package.sh"',
            precomposed,
        )
        self.assertIn(
            'compose-product-bundle.py"',
            precomposed,
        )
        self.assertIn("--check", precomposed)

    def test_composer_output_accepts_git_for_windows_path(self) -> None:
        contents = SCRIPT.read_text(encoding="utf-8")
        resolver = self.function_block(
            contents,
            "Resolve-RepositoryPath",
            "function Assert-AttestationConfig",
        )

        self.assertIn(
            r"^/(?<drive>[A-Za-z])(?:/(?<tail>.*))?$",
            resolver,
        )
        self.assertIn(
            'GetFullPath("${drive}:\\${tail}")',
            resolver,
        )

    def test_preverified_attestation_requires_immutable_composer_contract(self) -> None:
        contents = SCRIPT.read_text(encoding="utf-8")
        config = self.function_block(
            contents,
            "Assert-AttestationConfig",
            "function Invoke-ReleaseAttestationStamp",
        )
        attestation = self.function_block(
            contents,
            "Invoke-ReleaseAttestationStamp",
            "function Copy-AndVerifyPrecomposedProduct",
        )

        self.assertIn(
            "MESH_RELEASE_ATTESTATION_PREVERIFIED=1 requires a pre-stamped precomposed product",
            config,
        )
        self.assertIn(
            '$env:MESH_RELEASE_HOST_PRESTAMPED -ne "1"',
            config,
        )
        self.assertIn(
            'if ($attestationPreverified -eq "1")',
            attestation,
        )
        self.assertIn(
            "Release attestation: verified by immutable product composer",
            attestation,
        )
        self.assertLess(
            attestation.index('if ($attestationPreverified -eq "1")'),
            attestation.index("cargo run -q -p xtask"),
        )

    def test_legacy_runtime_selection_path_is_preserved(self) -> None:
        contents = SCRIPT.read_text(encoding="utf-8")
        main = contents[contents.index("try {") :]

        self.assertIn("if (Test-HasValue $precomposedProductDir)", main)
        self.assertIn("} else {", main)
        self.assertIn("$selectorArgs = @(", main)
        self.assertIn("failed to select the packaged Windows native runtime", main)

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

    @staticmethod
    def function_block(contents: str, function_name: str, end_marker: str) -> str:
        start = contents.index(f"function {function_name}")
        end = contents.index(end_marker, start)
        return contents[start:end]


if __name__ == "__main__":
    unittest.main()
