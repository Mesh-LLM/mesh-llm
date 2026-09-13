"""Contract for the native runtime-event gate and its CI lane.

`crates/skippy-runtime/tests/runtime_events_native.rs` is the only test that
exercises the reporter against actual native code. It is env-gated so an
ordinary `cargo test` never touches a native symbol -- which also meant
nothing in CI ever ran it, and the whole native reporter path was covered
only by whoever remembered to run it by hand.

The gate reports a blocked prerequisite as a PASS by design, so the danger
is a lane that looks green while having executed nothing. These tests pin
the two things that prevent that: the script checks for the `executed`
marker itself, and the lane runs the script.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "ci-runtime-events-native-gate.sh"
SLICE = ROOT / ".github" / "workflows" / "ci-linux-runtime-slice.yml"


class GateScriptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.script = SCRIPT.read_text(encoding="utf-8")

    def test_the_script_is_executable(self) -> None:
        self.assertTrue(SCRIPT.stat().st_mode & 0o111, "script must be executable")

    def test_a_blocked_run_fails_instead_of_passing_quietly(self) -> None:
        """The test itself exits 0 on a missing prerequisite. A lane that
        accepted that would be permanently, invisibly green."""
        self.assertIn("grep -q '^executed'", self.script)
        self.assertIn("did not execute", self.script)
        self.assertIn("exit 1", self.script)

    def test_the_evidence_file_is_truncated_before_the_run(self) -> None:
        """A warm workspace could otherwise leave a previous run's
        `executed` marker behind, and the check above would read it."""
        self.assertIn(': >"$EVIDENCE_FILE"', self.script)

    def test_it_sets_every_prerequisite_the_gate_requires(self) -> None:
        for variable in (
            "MESH_LLM_RUNTIME_EVENTS_NATIVE_TEST=1",
            "MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR=",
            "MESH_LLM_RUNTIME_EVENTS_MODEL=",
            "MESH_LLM_RUNTIME_EVENTS_EVIDENCE_FILE=",
        ):
            self.assertIn(variable, self.script)

    def test_it_enables_the_dynamic_native_runtime_feature(self) -> None:
        """Without it the gate panics on an opted-in run, by design."""
        self.assertIn("--features dynamic-native-runtime", self.script)
        self.assertIn("--test runtime_events_native", self.script)

    def test_missing_inputs_are_rejected_rather_than_defaulted(self) -> None:
        self.assertIn("native runtime bundle directory does not exist", self.script)
        self.assertIn("model is missing or empty", self.script)


class GateScriptBehaviorTests(unittest.TestCase):
    """Drive the real script with a stub `cargo`.

    The gate itself needs a built native runtime and a real model, so CI is
    the only place it genuinely runs. What the script does AROUND the gate
    is the part that decides whether a green lane means anything, and that
    is testable here.
    """

    def run_gate(
        self,
        root: Path,
        *,
        cargo_body: str,
        bundle: str | None = None,
        model: str | None = None,
        evidence_seed: str | None = None,
    ) -> subprocess.CompletedProcess[str]:
        stub_bin = root / "stub-bin"
        stub_bin.mkdir(exist_ok=True)
        cargo = stub_bin / "cargo"
        cargo.write_text(cargo_body, encoding="utf-8")
        cargo.chmod(0o755)

        if bundle is None:
            bundle_dir = root / "bundle"
            bundle_dir.mkdir(exist_ok=True)
            bundle = str(bundle_dir)
        if model is None:
            model_path = root / "model.gguf"
            model_path.write_bytes(b"not really a model, the stub never reads it")
            model = str(model_path)

        evidence = root / "evidence.txt"
        if evidence_seed is not None:
            evidence.write_text(evidence_seed, encoding="utf-8")

        return subprocess.run(
            [
                "bash",
                str(SCRIPT),
                "--bundle-dir",
                bundle,
                "--model",
                model,
                "--evidence",
                str(evidence),
            ],
            capture_output=True,
            text=True,
            check=False,
            env={
                **os.environ,
                "PATH": f"{stub_bin}{os.pathsep}{os.environ['PATH']}",
            },
        )

    EXECUTES = (
        "#!/usr/bin/env bash\n"
        'printf \'executed\\n\' >> "$MESH_LLM_RUNTIME_EVENTS_EVIDENCE_FILE"\n'
        "exit 0\n"
    )
    BLOCKS = (
        "#!/usr/bin/env bash\n"
        'printf \'blocked-when-ungated: gate unset\\n\' '
        '>> "$MESH_LLM_RUNTIME_EVENTS_EVIDENCE_FILE"\n'
        "exit 0\n"
    )

    def test_a_gate_that_executed_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_gate(Path(directory), cargo_body=self.EXECUTES)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("executed", result.stdout)

    def test_a_blocked_gate_fails_even_though_the_test_exits_zero(self) -> None:
        """The whole reason the script checks the marker.

        `runtime_events_native.rs` reports a missing prerequisite as a pass
        by design, so a lane that only checked the exit code would be
        permanently, invisibly green.
        """
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_gate(Path(directory), cargo_body=self.BLOCKS)
            self.assertNotEqual(result.returncode, 0)
            output = result.stdout + result.stderr
            self.assertIn("did not execute", output)
            self.assertIn("blocked-when-ungated", output)

    def test_a_stale_executed_marker_cannot_make_a_blocked_run_pass(self) -> None:
        """A warm workspace keeps the evidence file from the last run. If the
        script appended instead of truncating, yesterday's `executed` would
        satisfy today's check."""
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_gate(
                Path(directory),
                cargo_body=self.BLOCKS,
                evidence_seed="executed\nfrom a previous run\n",
            )
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("did not execute", result.stdout + result.stderr)

    def test_a_failing_gate_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_gate(
                Path(directory),
                cargo_body="#!/usr/bin/env bash\nexit 101\n",
            )
            self.assertNotEqual(result.returncode, 0)

    def test_a_missing_bundle_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.run_gate(
                root,
                cargo_body=self.EXECUTES,
                bundle=str(root / "nope"),
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("bundle directory does not exist", result.stderr)

    def test_a_missing_model_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.run_gate(
                root,
                cargo_body=self.EXECUTES,
                model=str(root / "absent.gguf"),
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("model is missing or empty", result.stderr)

    def test_the_gate_receives_every_prerequisite(self) -> None:
        """The env the gate reads, captured from inside the stub rather than
        grepped out of the script."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            captured = root / "env.txt"
            result = self.run_gate(
                root,
                cargo_body=(
                    "#!/usr/bin/env bash\n"
                    f'{{ printf "%s\\n" '
                    '"$MESH_LLM_RUNTIME_EVENTS_NATIVE_TEST" '
                    '"$MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR" '
                    '"$MESH_LLM_RUNTIME_EVENTS_MODEL"; '
                    f'printf "%s\\n" "$*"; }} > {captured}\n'
                    'printf \'executed\\n\' '
                    '>> "$MESH_LLM_RUNTIME_EVENTS_EVIDENCE_FILE"\n'
                ),
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            gate, bundle, model, argv = captured.read_text().splitlines()
            self.assertEqual(gate, "1")
            self.assertTrue(Path(bundle).is_dir())
            self.assertTrue(Path(model).is_file())
            self.assertIn("--features dynamic-native-runtime", argv)
            self.assertIn("--test runtime_events_native", argv)


class LinuxRuntimeSliceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = SLICE.read_text(encoding="utf-8")
        self.workflow = yaml.safe_load(self.source)
        job = self.workflow["jobs"]["linux_runtime"]
        self.steps = {step.get("name"): step for step in job["steps"]}

    def test_the_slice_runs_the_gate(self) -> None:
        step = self.steps["Run native runtime-event gate"]
        self.assertIn("scripts/ci-runtime-events-native-gate.sh", step["run"])

    def test_the_gate_runs_on_cpu_only(self) -> None:
        """The reporter is backend-independent, so a second backend would
        buy a duplicate of the same evidence at the cost of another model
        download and native build."""
        for name in (
            "Restore runtime-event gate model",
            "Run native runtime-event gate",
            "Upload native runtime-event gate evidence",
        ):
            self.assertIn(
                "matrix.runtime.backend == 'cpu'",
                self.steps[name]["if"],
                f"{name} must be CPU-only",
            )

    def test_the_gate_gets_the_bundle_directory_not_the_runtime_directory(self) -> None:
        """`MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR` names the directory that
        CONTAINS runtime-id directories. `prepare-native-runtime-input`
        outputs the runtime directory itself, so the lane has to take its
        parent -- pointing the host at the runtime directory finds no
        runtime at all."""
        step = self.steps["Run native runtime-event gate"]
        self.assertIn(
            'dirname "${{ steps.native_runtime.outputs.runtime_dir }}"', step["run"]
        )

    def test_the_prepare_step_is_addressable(self) -> None:
        self.assertEqual(
            self.steps["Prepare immutable Linux native runtime"]["id"], "native_runtime"
        )

    def test_the_gate_selects_one_artifact_from_the_shared_manifest(self) -> None:
        step = self.steps["Restore runtime-event gate model"]
        self.assertEqual(
            step["with"]["model_manifest"],
            "ci/model-artifacts/manifests/skippy-ci-smoke.json",
        )
        self.assertEqual(step["with"]["model_artifact_id"], "family-qwen3-dense")

    def test_the_gate_model_is_authorized_for_pr_and_main_ci(self) -> None:
        manifest = yaml.safe_load(
            (ROOT / "ci/model-artifacts/manifests/skippy-ci-smoke.json").read_text(
                encoding="utf-8"
            )
        )
        artifact = next(
            row for row in manifest["artifacts"] if row["id"] == "family-qwen3-dense"
        )
        self.assertTrue({"pull-request", "main"}.issubset(artifact["cadences"]))

    def test_evidence_is_uploaded_even_when_the_gate_fails(self) -> None:
        """The evidence file is how a failure is diagnosed, so it must
        survive one."""
        step = self.steps["Upload native runtime-event gate evidence"]
        self.assertIn("!cancelled()", step["if"])
        self.assertEqual(step["with"]["if-no-files-found"], "error")


if __name__ == "__main__":
    unittest.main()
