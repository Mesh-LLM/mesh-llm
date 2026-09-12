#!/usr/bin/env python3

import pathlib
import unittest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "sdk/swift/scripts/build-xcframework.sh"


class SwiftXcframeworkEnvTests(unittest.TestCase):
    def test_deployment_targets_are_not_globally_exported(self) -> None:
        script = SCRIPT.read_text()

        self.assertNotIn("export IPHONEOS_DEPLOYMENT_TARGET=", script)
        self.assertNotIn("export MACOSX_DEPLOYMENT_TARGET=", script)
        self.assertIn("export -n IPHONEOS_DEPLOYMENT_TARGET MACOSX_DEPLOYMENT_TARGET", script)

    def test_cargo_build_gets_platform_specific_deployment_target(self) -> None:
        script = SCRIPT.read_text()

        self.assertIn("*-apple-darwin)", script)
        self.assertIn('CARGO_ENV+=("MACOSX_DEPLOYMENT_TARGET=$MACOSX_DEPLOYMENT_TARGET")', script)
        self.assertIn("*-apple-ios*)", script)
        self.assertIn('CARGO_ENV+=("IPHONEOS_DEPLOYMENT_TARGET=$IPHONEOS_DEPLOYMENT_TARGET")', script)

    def test_target_mode_stages_one_library_for_parallel_ci(self) -> None:
        script = SCRIPT.read_text()

        self.assertIn('build-xcframework.sh --target <rust-target>', script)
        self.assertIn('MODE="target"', script)
        self.assertIn('stage_target_library "$SELECTED_TARGET"', script)
        self.assertIn('install -m 0644 "$SOURCE_LIBRARY"', script)

    def test_assembly_mode_requires_every_apple_target(self) -> None:
        script = SCRIPT.read_text()

        self.assertIn('build-xcframework.sh --assemble-from <target-artifact-directory>', script)
        self.assertIn('for RUST_TARGET in "${APPLE_TARGETS[@]}"', script)
        self.assertIn('staged Swift target library is missing', script)
        self.assertIn('restore_staged_libraries', script)


if __name__ == "__main__":
    unittest.main()
