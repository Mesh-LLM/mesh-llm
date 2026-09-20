"""Shared Rust/native macOS deployment target and canary propagation."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "scripts/lib/macos-deployment-target.sh"
DEFAULT = (ROOT / "scripts/lib/macos-deployment-target.txt").read_text().strip()


class MacosDeploymentTargetTests(unittest.TestCase):
    def test_shell_default_override_and_non_macos_scope(self):
        for host, override, expected in (
            ("Darwin", None, DEFAULT), ("Darwin", "", DEFAULT), ("Darwin", "14.0", "14.0"),
            ("Linux", None, "unset"), ("Linux", "14.0", "14.0"),
        ):
            with self.subTest(host=host, override=override), tempfile.TemporaryDirectory() as tmp:
                uname = Path(tmp) / "uname"
                uname.write_text(f"#!/bin/sh\necho {host}\n")
                uname.chmod(0o755)
                env = dict(os.environ, PATH=f"{tmp}:{os.environ['PATH']}")
                env.pop("MACOSX_DEPLOYMENT_TARGET", None)
                if override is not None:
                    env["MACOSX_DEPLOYMENT_TARGET"] = override
                result = subprocess.run(
                    ["bash", "-c", 'source "$1"; bash -c \'echo "${MACOSX_DEPLOYMENT_TARGET-unset}"\'', "test", str(HELPER)],
                    env=env, capture_output=True, text=True, check=True,
                )
                self.assertEqual(result.stdout.strip(), expected)

    @unittest.skipUnless(shutil.which("just"), "just is required")
    def test_just_exports_default_and_preserves_override(self):
        for override in (None, "", "14.0"):
            env = dict(os.environ)
            env.pop("MACOSX_DEPLOYMENT_TARGET", None)
            if override is not None:
                env["MACOSX_DEPLOYMENT_TARGET"] = override
            result = subprocess.run(
                ["just", "--command", "bash", "-c", 'echo "$MACOSX_DEPLOYMENT_TARGET"'],
                cwd=ROOT, env=env, capture_output=True, text=True, check=True,
            )
            self.assertEqual(result.stdout.strip(), override or DEFAULT)

    def test_both_canary_jobs_export_target_before_compilation(self):
        workflow = yaml.safe_load((ROOT / ".github/workflows/llama-canary-family-pass.yml").read_text())
        steps = yaml.safe_load((ROOT / ".github/actions/setup-canary-runner/action.yml").read_text())["runs"]["steps"]
        setup = next(i for i, step in enumerate(steps)
                     if "source scripts/lib/macos-deployment-target.sh" in step.get("run", ""))
        self.assertIn('echo "MACOSX_DEPLOYMENT_TARGET=$MACOSX_DEPLOYMENT_TARGET" >> "$GITHUB_ENV"',
                      steps[setup]["run"])
        cache = next(i for i, step in enumerate(steps) if step.get("name") == "Isolate compiler cache identity")
        self.assertLess(setup, cache)
        self.assertIn('macos-deployment-target=%s', steps[cache]["run"])
        build = workflow['jobs']['build']['steps']
        setup_call = next(i for i, step in enumerate(build) if step.get('uses') == './.github/actions/setup-canary-runner')
        compile_call = next(i for i, step in enumerate(build) if step.get('id') == 'build')
        self.assertLess(setup_call, compile_call)
        # Family consumers run producer bytes; no compiler or deployment-target inference.
        self.assertNotIn('cargo ', '\n'.join(step.get('run', '') for step in workflow['jobs']['family']['steps']))



if __name__ == "__main__":
    unittest.main()
