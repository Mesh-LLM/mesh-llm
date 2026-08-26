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
FAMILY_CERTIFY = ROOT / "scripts" / "family-certify.sh"
STATE_HANDOFF = ROOT / "crates" / "skippy-correctness" / "src" / "runner" / "state_handoff.rs"
KV_PAGES = ROOT / "crates" / "skippy-runtime" / "src" / "kv_pages.rs"
STAGED_GRAPH_PATCH = (
    ROOT
    / "third_party"
    / "llama.cpp"
    / "patches"
    / "0001-Add-staged-model-graph-and-family-support.patch"
)
MODEL_LIFECYCLE_PATCH = (
    ROOT
    / "third_party"
    / "llama.cpp"
    / "patches"
    / "0004-Add-Skippy-model-lifecycle-and-package-support.patch"
)


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
        self.assertIn(
            "LLAMA_BUILD_DIR: ${{ github.workspace }}/.deps/llama-canary-${{ github.run_id }}-${{ github.run_attempt }}",
            workflow,
        )
        self.assertIn(
            "LLAMA_STAGE_BUILD_DIR: ${{ github.workspace }}/.deps/llama-canary-${{ github.run_id }}-${{ github.run_attempt }}",
            workflow,
        )
        self.assertIn("LLAMA_STAGE_BACKEND: metal", workflow)
        self.assertNotIn("mozilla-actions/sccache-action", workflow)
        self.assertNotIn("SCCACHE_GHA_ENABLED", workflow)
        self.assertNotIn("SCCACHE_C_CUSTOM_CACHE_BUSTER", workflow)
        self.assertIn('RUSTC_WRAPPER: ""', workflow)
        self.assertIn('LLAMA_STAGE_USE_SCCACHE: "0"', workflow)
        self.assertNotIn("Show sccache stats", workflow)

        native_build = _step_block(workflow, "Build patched llama.cpp ABI")
        self.assertIn(
            "arch -arm64 bash scripts/build-llama.sh -DCMAKE_OSX_ARCHITECTURES=arm64",
            native_build,
        )

        build = _step_block(workflow, "Build stage runtime crates")
        self.assertIn("cargo build", build)
        self.assertIn("steps.sha.outputs.certify == 'true'", build)
        for package in (
            "skippy-correctness",
            "skippy-server",
            "skippy-model-package",
            "llama-spec-bench",
        ):
            self.assertIn(f"-p {package}", build)

        architecture = _step_block(workflow, "Verify native archive architecture")
        self.assertIn('lipo -archs "$archive"', architecture)
        self.assertIn('[[ "$arches" != "arm64" ]]', architecture)

        battery = _step_block(
            workflow, "Supported-families certification battery (parity gate)"
        )
        self.assertIn("run: scripts/skippy-family-battery.sh --skip-build", battery)
        self.assertIn("steps.sha.outputs.certify == 'true'", battery)
        self.assertIn("FAMILY_BATTERY_RUN_ID:", battery)

        upload = _step_block(workflow, "Upload supported-families battery evidence")
        self.assertIn("if: ${{ !cancelled()", upload)
        self.assertIn("actions/upload-artifact@", upload)
        self.assertIn("target/family-battery/", upload)
        self.assertIn("retention-days: 14", upload)

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
        self.assertIn("'s/^[[:space:]]*path:[[:space:]]*//p'", smoke)
        self.assertIn('CTX_SIZE="${CTX_SIZE:-8192}"', smoke)
        self.assertIn('PROMPT_CTX_SIZE="${PROMPT_CTX_SIZE:-$CTX_SIZE}"', smoke)

        battery = BATTERY.read_text(encoding="utf-8")
        self.assertIn("'s/^[[:space:]]*path:[[:space:]]*//p'", battery)
        self.assertIn("skippy-model-package\" inspect", battery)
        self.assertIn('contains(".nextn.")', battery)
        self.assertIn("--require-native-mtp-draft", battery)
        self.assertIn("startup_timeout_for_bytes", battery)
        self.assertIn("speculative_coding_prompts.jsonl", battery)

        manifest = FAMILY_MANIFEST.read_text(encoding="utf-8")
        self.assertIn(
            "qwen3-dense|Qwen/Qwen3-0.6B-GGUF|Qwen3-0.6B-Q8_0.gguf|", manifest
        )
        self.assertIn(
            "falcon-h1|tiiuae/Falcon-H1-1.5B-Instruct-GGUF|"
            "Falcon-H1-1.5B-Instruct-Q4_K_M.gguf|",
            manifest,
        )

    def test_state_handoff_restores_the_authoritative_continuation_position(self) -> None:
        state_handoff = STATE_HANDOFF.read_text(encoding="utf-8")
        self.assertIn(
            "LocalStatePayload::FullState(bytes) => session.import_full_state_for_token_count(",
            state_handoff,
        )

        kv_pages = KV_PAGES.read_text(encoding="utf-8")
        full_state_start = kv_pages.index("pub fn import_full_state_for_token_count(")
        full_state_end = kv_pages.index("pub fn export_kv_page(", full_state_start)
        self.assertIn("self.set_position(token_count)", kv_pages[full_state_start:full_state_end])

    def test_family_results_have_typed_failure_outcomes(self) -> None:
        certify = FAMILY_CERTIFY.read_text(encoding="utf-8")
        for outcome in (
            "timeout",
            "unsupported",
            "model-invalid",
            "harness",
            "mismatch",
            "runtime-error",
        ):
            self.assertIn(f"printf '{outcome}\\n'", certify)
        self.assertIn("outcome:$outcome", certify)

    def test_runtime_slice_contract_covers_glm4_moe(self) -> None:
        lifecycle = MODEL_LIFECYCLE_PATCH.read_text(encoding="utf-8")
        self.assertIn("model->arch != LLM_ARCH_GLM4_MOE", lifecycle)

    def test_staged_hybrid_memory_preserves_component_partition(self) -> None:
        staged_graph = STAGED_GRAPH_PATCH.read_text(encoding="utf-8")
        attention_default = (
            "filter_attn = [&](uint32_t il) { return !hparams.is_recr(il); };"
        )
        recurrent_default = (
            "filter_recr = [&](uint32_t il) { return hparams.is_recr(il); };"
        )
        stage_intersection = (
            "filter_attn = skippy_stage_memory_filter(std::move(filter_attn), "
            "params.ctx_type);"
        )
        self.assertLess(
            staged_graph.index(attention_default), staged_graph.index(stage_intersection)
        )
        self.assertLess(
            staged_graph.index(recurrent_default), staged_graph.index(stage_intersection)
        )

    def test_certification_captures_stage_child_failures(self) -> None:
        certify = FAMILY_CERTIFY.read_text(encoding="utf-8")
        correctness_start = certify.index("correctness_common=(")
        correctness_end = certify.index("\n)", correctness_start)
        self.assertIn("--child-logs", certify[correctness_start:correctness_end])

    def test_nemotron_uses_a_non_broken_quantization(self) -> None:
        manifest = FAMILY_MANIFEST.read_text(encoding="utf-8")
        nemotron = next(
            line for line in manifest.splitlines() if line.startswith("nemotron|")
        )
        self.assertIn(
            "unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF", nemotron
        )
        self.assertIn("UD-Q4_K_M.gguf|Q4_K_M|", nemotron)
        self.assertNotEqual(
            "ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF",
            nemotron.split("|")[1],
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
            if line.startswith(str(FAMILY_CERTIFY) + " ")
        ]
        self.assertEqual(1, len(commands))
        self.assertTrue(commands[0].strip().endswith("--require-lanes --skip-build"))

    def test_skip_build_omits_the_one_time_build(self) -> None:
        result = self._dry_run("--skip-build")
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertNotIn("cargo build -p skippy-correctness", result.stdout)


if __name__ == "__main__":
    unittest.main()
