from __future__ import annotations

import os
from pathlib import Path
import plistlib
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
VERIFIER = ROOT / "scripts" / "verify-swift-xcframework.py"


FULL_SLICES = [
    ("ios-arm64", "ios", "", ["arm64"]),
    ("ios-arm64_x86_64-simulator", "ios", "simulator", ["arm64", "x86_64"]),
    (
        "ios-arm64_x86_64-maccatalyst",
        "ios",
        "maccatalyst",
        ["arm64", "x86_64"],
    ),
    ("macos-arm64_x86_64", "macos", "", ["arm64", "x86_64"]),
]


class SwiftXCFrameworkVerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.xcframework = self.root / "MeshLLMFFI.xcframework"
        self.xcframework.mkdir()
        self.lipo = self.root / "fake-lipo"
        self.lipo.write_text(
            "#!/usr/bin/env python3\n"
            "from pathlib import Path\n"
            "import sys\n"
            "print(Path(sys.argv[-1]).read_text(encoding='utf-8').strip())\n",
            encoding="utf-8",
        )
        self.lipo.chmod(0o755)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write_framework(
        self,
        identifier: str,
        platform: str,
        variant: str,
        declared_architectures: list[str],
        binary_architectures: list[str] | None = None,
    ) -> dict[str, object]:
        framework = (
            self.xcframework
            / identifier
            / "MeshLLMFFI.framework"
        )
        framework.mkdir(parents=True)
        binary_contents = " ".join(binary_architectures or declared_architectures)

        if platform == "macos" and not variant:
            version = framework / "Versions" / "A"
            (version / "Headers").mkdir(parents=True)
            (version / "Modules").mkdir()
            (version / "Resources").mkdir()
            (version / "MeshLLMFFI").write_text(
                binary_contents,
                encoding="utf-8",
            )
            (version / "Modules" / "module.modulemap").write_text(
                "framework module MeshLLMFFI {}\n",
                encoding="utf-8",
            )
            (version / "Resources" / "Info.plist").write_bytes(
                plistlib.dumps({"CFBundleName": "MeshLLMFFI"}),
            )
            (version / "Resources" / "PrivacyInfo.xcprivacy").write_bytes(
                plistlib.dumps({"NSPrivacyTracking": False}),
            )
            (framework / "Versions" / "Current").symlink_to("A")
            (framework / "MeshLLMFFI").symlink_to(
                "Versions/Current/MeshLLMFFI",
            )
            (framework / "Headers").symlink_to("Versions/Current/Headers")
            (framework / "Modules").symlink_to("Versions/Current/Modules")
            (framework / "Resources").symlink_to("Versions/Current/Resources")
        else:
            (framework / "MeshLLMFFI").write_text(
                binary_contents,
                encoding="utf-8",
            )

        entry: dict[str, object] = {
            "LibraryIdentifier": identifier,
            "LibraryPath": "MeshLLMFFI.framework",
            "SupportedArchitectures": declared_architectures,
            "SupportedPlatform": platform,
        }
        if variant:
            entry["SupportedPlatformVariant"] = variant
        return entry

    def write_info(self, slices: list[tuple[str, str, str, list[str]]]) -> None:
        libraries = [
            self.write_framework(identifier, platform, variant, architectures)
            for identifier, platform, variant, architectures in slices
        ]
        (self.xcframework / "Info.plist").write_bytes(
            plistlib.dumps({"AvailableLibraries": libraries}),
        )

    def run_verifier(self, mode: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["LIPO"] = str(self.lipo)
        return subprocess.run(
            [sys.executable, str(VERIFIER), str(self.xcframework), mode],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )

    def test_accepts_exact_full_architecture_matrix(self) -> None:
        self.write_info(FULL_SLICES)

        result = self.run_verifier("full")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("verified 4 XCFramework slice(s)", result.stdout)

    def test_accepts_exact_host_only_arm64_slice(self) -> None:
        self.write_info(
            [("macos-arm64", "macos", "", ["arm64"])],
        )

        result = self.run_verifier("host-only")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("host-only mode", result.stdout)

    def test_rejects_full_slice_missing_declared_x86_64(self) -> None:
        slices = list(FULL_SLICES)
        slices[1] = (
            "ios-arm64-simulator",
            "ios",
            "simulator",
            ["arm64"],
        )
        self.write_info(slices)

        result = self.run_verifier("full")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unexpected architecture contract", result.stderr)

    def test_rejects_lipo_slices_that_disagree_with_plist(self) -> None:
        libraries = []
        for identifier, platform, variant, architectures in FULL_SLICES:
            binary_architectures = (
                ["arm64"]
                if variant == "maccatalyst"
                else architectures
            )
            libraries.append(
                self.write_framework(
                    identifier,
                    platform,
                    variant,
                    architectures,
                    binary_architectures,
                ),
            )
        (self.xcframework / "Info.plist").write_bytes(
            plistlib.dumps({"AvailableLibraries": libraries}),
        )

        result = self.run_verifier("full")

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("lipo architectures", result.stderr)

    def test_mode_independent_verification_still_requires_macos(self) -> None:
        self.write_info(
            [("ios-arm64", "ios", "", ["arm64"])],
        )
        env = os.environ.copy()
        env["LIPO"] = str(self.lipo)

        result = subprocess.run(
            [sys.executable, str(VERIFIER), str(self.xcframework)],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not contain a macOS framework slice", result.stderr)


if __name__ == "__main__":
    unittest.main()
