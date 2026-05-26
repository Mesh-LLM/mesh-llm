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


if __name__ == "__main__":
    unittest.main()
