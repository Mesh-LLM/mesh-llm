from __future__ import annotations

import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "llama-upstream-canary.yml"
BATTERY = ROOT / "scripts" / "skippy-family-battery.sh"
SMOKE = ROOT / "scripts" / "skippy-ci-smoke.sh"
FAMILY_MANIFEST = ROOT / "ci" / "llama-canary" / "family-certified.tsv"


def _step_block(workflow: str, name: str) -> str:
    marker = f"      - name: {name}\n"
    start = workflow.index(marker)
    end = workflow.find("\n      - name: ", start + len(marker))
    return workflow[start:] if end == -1 else workflow[start:end]


class LlamaUpstreamCanaryWorkflowTests(unittest.TestCase):
    def test_workflow_builds_binaries_before_skipping_per_lane_builds(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        self.assertIn('- "scripts/family-certify.sh"', workflow)
        self.assertIn("force_certify:", workflow)
        self.assertIn("FORCE_CERTIFY:", workflow)

        build = _step_block(workflow, "Build stage runtime crates")
        self.assertIn("cargo build", build)
        self.assertIn("steps.sha.outputs.certify == 'true'", build)
        for package in ("skippy-correctness", "skippy-server", "llama-spec-bench"):
            self.assertIn(f"-p {package}", build)

        battery = _step_block(
            workflow, "Supported-families certification battery (parity gate)"
        )
        self.assertIn("run: scripts/skippy-family-battery.sh --skip-build", battery)
        self.assertIn("steps.sha.outputs.certify == 'true'", battery)

        capture = _step_block(workflow, "Capture upstream SHAs")
        self.assertIn('"$FORCE_CERTIFY" == "true"', capture)
        self.assertIn('echo "certify=true"', capture)

        forced_report = _step_block(workflow, "Report forced certification result")
        self.assertIn("steps.sha.outputs.changed == 'false'", forced_report)
        self.assertIn("steps.sha.outputs.certify == 'true'", forced_report)

    def test_failed_repair_summary_runs_after_a_failed_step(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        condition = _step_block(workflow, "Report patch-queue failure")
        self.assertIn("failure()", condition)
        self.assertIn("steps.agent_repair.outcome == 'failure'", condition)

    def test_smoke_uses_read_only_prewarmed_family_cache(self) -> None:
        workflow = WORKFLOW.read_text(encoding="utf-8")
        smoke_step = _step_block(workflow, "Skippy smoke tests")
        self.assertNotIn("MODEL_DIR", smoke_step)

        smoke = SMOKE.read_text(encoding="utf-8")
        self.assertIn('DENSE_MODEL_REPO="${DENSE_MODEL_REPO:-Qwen/Qwen3-0.6B-GGUF}"', smoke)
        self.assertIn(
            'RECURRENT_MODEL_REPO="${RECURRENT_MODEL_REPO:-tiiuae/Falcon-H1-1.5B-Instruct-GGUF}"',
            smoke,
        )
        self.assertIn('HF_HUB_CACHE="$HF_CACHE/hub"', smoke)
        self.assertIn("HF_HUB_OFFLINE=1", smoke)
        self.assertIn('hf download "$repo" "$file"', smoke)

        manifest = FAMILY_MANIFEST.read_text(encoding="utf-8")
        self.assertIn(
            "qwen3-dense|Qwen/Qwen3-0.6B-GGUF|Qwen3-0.6B-Q8_0.gguf|", manifest
        )
        self.assertIn(
            "falcon-h1|tiiuae/Falcon-H1-1.5B-Instruct-GGUF|"
            "Falcon-H1-1.5B-Instruct-Q4_K_M.gguf|",
            manifest,
        )


class SkippyFamilyBatteryTests(unittest.TestCase):
    def _dry_run(self, *args: str) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            bin_dir = temp / "bin"
            bin_dir.mkdir()
            for command in ("hf", "jq"):
                executable = bin_dir / command
                executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
                executable.chmod(executable.stat().st_mode | stat.S_IXUSR)

            manifest = temp / "manifest.tsv"
            manifest.write_text(
                "test-family|org/model|model.gguf|Q4_K_M|0|6|fixture||\n",
                encoding="utf-8",
            )
            env = os.environ.copy()
            env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
            return subprocess.run(
                [
                    str(BATTERY),
                    "--manifest",
                    str(manifest),
                    "--dry-run",
                    *args,
                ],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )

    def test_battery_builds_once_then_skips_build_in_each_lane(self) -> None:
        result = self._dry_run()
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(1, result.stdout.count("cargo build -p skippy-correctness"))
        commands = [
            line
            for line in result.stdout.splitlines()
            if line.startswith("scripts/family-certify.sh ")
        ]
        self.assertEqual(1, len(commands))
        self.assertTrue(commands[0].endswith("--require-lanes --skip-build"))

    def test_skip_build_omits_the_one_time_build(self) -> None:
        result = self._dry_run("--skip-build")
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertNotIn("cargo build -p skippy-correctness", result.stdout)


if __name__ == "__main__":
    unittest.main()
