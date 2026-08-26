from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "llama-upstream-canary.yml"
BATTERY = ROOT / "scripts" / "skippy-family-battery.sh"
BATTERY_PLANNER = ROOT / "scripts" / "plan-family-battery.py"
SMOKE = ROOT / "scripts" / "skippy-ci-smoke.sh"
FAMILY_MANIFEST = ROOT / "ci" / "llama-canary" / "family-certified.json"
FAMILY_CERTIFY = ROOT / "scripts" / "family-certify.sh"
FAMILY_OUTCOME = ROOT / "scripts" / "lib" / "family-outcome.sh"
TIMEOUT_RUNNER = ROOT / "scripts" / "run-command-with-timeout.py"
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
NEMOTRON_MTP_EMBEDDING_PATCH = (
    ROOT
    / "third_party"
    / "llama.cpp"
    / "patches"
    / "0022-skippy-retain-Nemotron-MTP-shared-embeddings.patch"
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
        self.assertIn('- "scripts/lib/family-outcome.sh"', workflow)
        self.assertIn('- "scripts/run-command-with-timeout.py"', workflow)
        self.assertIn('- "scripts/plan-family-battery.py"', workflow)
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

        family_plan = _step_block(workflow, "Plan and verify family certification cache")
        self.assertIn("python3 scripts/plan-family-battery.py", family_plan)
        self.assertIn("--check-cache", family_plan)
        self.assertIn('--cache-root "$HF_CACHE"', family_plan)
        self.assertIn('--github-output "$GITHUB_OUTPUT"', family_plan)
        self.assertLess(workflow.index(family_plan), workflow.index(native_build))

        build = _step_block(workflow, "Build stage runtime crates")
        self.assertIn("cargo build", build)
        self.assertIn("steps.sha.outputs.certify == 'true'", build)
        self.assertNotIn("-p skippy-ffi", build)
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
        self.assertIn("scripts/skippy-family-battery.sh --skip-build --plan", battery)
        self.assertIn("steps.sha.outputs.certify == 'true'", battery)
        self.assertIn("FAMILY_BATTERY_RUN_ID:", battery)
        self.assertIn("timeout-minutes: 720", battery)

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
        self.assertIn('contains(".nextn.eh_proj")', battery)
        self.assertIn('contains(".nextn.enorm")', battery)
        self.assertIn('contains(".nextn.hnorm")', battery)
        self.assertIn("--require-native-mtp-draft", battery)
        self.assertIn("startup_timeout_for_bytes", battery)
        self.assertIn('.resources.startup_timeout_secs // ""', battery)
        self.assertIn('--activation-width "$activation_width"', battery)
        self.assertIn("speculative_coding_prompts.jsonl", battery)
        self.assertIn('resolve_model "$repo" "$file" "$revision"', battery)
        self.assertIn("resolved-models.tsv", battery)
        self.assertIn("preflight_speculative_corpus", battery)
        self.assertIn("--skip-speculative", battery)
        self.assertIn('if (( MODEL_HAS_MTP == 1 )) && [[ "$speculative_policy" == "mtp-if-present" ]]; then', battery)

        manifest = json.loads(FAMILY_MANIFEST.read_text(encoding="utf-8"))
        by_family = {model["family"]: model for model in manifest["models"]}
        self.assertEqual("Qwen/Qwen3-0.6B-GGUF", by_family["qwen3-dense"]["artifact"]["repo"])
        self.assertEqual(
            ["Falcon-H1-1.5B-Instruct-Q4_K_M.gguf"],
            by_family["falcon-h1"]["artifact"]["files"],
        )
        self.assertEqual(4096, by_family["qwen3-vl"]["execution"]["activation_width"])
        self.assertEqual(600, by_family["qwen3-vl"]["resources"]["startup_timeout_secs"])

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
        classifier = FAMILY_OUTCOME.read_text(encoding="utf-8")
        for outcome in (
            "timeout",
            "unsupported",
            "model-invalid",
            "harness",
            "mismatch",
            "runtime-error",
        ):
            self.assertIn(f"printf '{outcome}\\n'", classifier)
        self.assertIn("outcome:$outcome", certify)
        self.assertNotIn("timed out|timeout|", classifier)

    def test_outcome_classifier_uses_terminal_evidence_not_option_names(self) -> None:
        fixtures = {
            "runtime-error": "+ tool --startup-timeout-secs 900\nlistener disconnected\n",
            "timeout": "stage 1 binary server did not become ready\n",
            "unsupported": "Unsupported: runtime-slice execution is not supported for this model architecture yet\n",
            "model-invalid": "missing tensor blk.5.ssm_in.weight\n",
            "mismatch": "authoritative token mismatch\n",
            "harness": "corpus file does not exist\n",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            log = Path(temp_dir) / "lane.log"
            for expected, evidence in fixtures.items():
                with self.subTest(expected=expected):
                    log.write_text(evidence, encoding="utf-8")
                    result = subprocess.run(
                        [
                            "bash",
                            "-c",
                            'source "$1"; classify_family_outcome fail "$2" ""',
                            "classifier-test",
                            str(FAMILY_OUTCOME),
                            str(log),
                        ],
                        text=True,
                        capture_output=True,
                        check=False,
                    )
                    self.assertEqual(0, result.returncode, result.stderr)
                    self.assertEqual(expected, result.stdout.strip())

    def test_portable_timeout_runner_bounds_the_process_group(self) -> None:
        result = subprocess.run(
            [
                str(TIMEOUT_RUNNER),
                "--seconds",
                "1",
                "--label",
                "fixture",
                "--",
                sys.executable,
                "-c",
                "import time; time.sleep(30)",
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=15,
        )
        self.assertEqual(124, result.returncode)
        self.assertIn("fixture timed out after 1s", result.stderr)

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
        manifest = json.loads(FAMILY_MANIFEST.read_text(encoding="utf-8"))
        nemotron = next(model for model in manifest["models"] if model["family"] == "nemotron")
        self.assertEqual(
            "unsloth/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF",
            nemotron["artifact"]["repo"],
        )
        self.assertIn("UD-Q4_K_M.gguf", nemotron["artifact"]["files"][0])
        self.assertEqual("Q4_K_M", nemotron["artifact"]["selector"])
        self.assertEqual(53, nemotron["execution"]["trunk_layers"] + nemotron["execution"]["mtp_layers"])

    def test_nemotron_final_mtp_stage_retains_shared_embeddings(self) -> None:
        patch = NEMOTRON_MTP_EMBEDDING_PATCH.read_text(encoding="utf-8")
        self.assertIn("llm_kv.arch == LLM_ARCH_NEMOTRON_H_MOE", patch)
        self.assertIn("hparams.n_layer_nextn > 0", patch)
        self.assertIn(
            "g_skippy_filter.layer_end > static_cast<int>(hparams.n_layer())", patch
        )
        self.assertIn("g_skippy_filter.include_output", patch)
        self.assertIn("keep = tn.tensor == LLM_TENSOR_TOKEN_EMBD", patch)

    def test_glm45_air_runtime_range_includes_the_mtp_layer(self) -> None:
        manifest = json.loads(FAMILY_MANIFEST.read_text(encoding="utf-8"))
        glm45_air = next(model for model in manifest["models"] if model["family"] == "glm45-air")
        self.assertEqual(46, glm45_air["execution"]["trunk_layers"])
        self.assertEqual(1, glm45_air["execution"]["mtp_layers"])

    def test_glm47_runtime_range_includes_blk_46(self) -> None:
        manifest = json.loads(FAMILY_MANIFEST.read_text(encoding="utf-8"))
        glm47 = next(model for model in manifest["models"] if model["family"] == "glm47-flash")
        self.assertEqual(47, glm47["execution"]["trunk_layers"])
        self.assertEqual(0, glm47["execution"]["mtp_layers"])


class SkippyFamilyBatteryTests(unittest.TestCase):
    @staticmethod
    def _manifest(model: dict[str, object]) -> dict[str, object]:
        return {
            "schema_version": 1,
            "policy": {
                "profiles": {
                    "full": {
                        "status": "certified",
                        "oracle": "local-monolithic",
                        "required_lanes": [
                            "single-step",
                            "chain",
                            "dtype-matrix",
                            "state-handoff",
                        ],
                    },
                    "package-oracle": {
                        "status": "certified",
                        "oracle": "independent-trace",
                        "required_lanes": [
                            "single-step",
                            "chain",
                            "dtype-matrix",
                            "state-handoff",
                        ],
                    },
                    "graph-only": {
                        "status": "provisional",
                        "oracle": "none",
                        "required_lanes": [
                            "graph-parse",
                            "tensor-ownership",
                            "stage-load",
                        ],
                    },
                },
                "cadences": ["llama-bump", "manual-full", "nightly", "rotating"],
            },
            "models": [model],
        }

    @staticmethod
    def _model(revision: str = "a" * 40) -> dict[str, object]:
        return {
            "family": "test-family",
            "profile": "full",
            "cadences": ["llama-bump", "manual-full"],
            "artifact": {
                "repo": "org/model",
                "revision": revision,
                "files": ["model.gguf"],
                "selector": "Q4_K_M",
            },
            "execution": {
                "trunk_layers": 6,
                "mtp_layers": 0,
                "activation_width": 1024,
                "boundary_sweep_period": 0,
                "speculative_policy": "mtp-if-present",
            },
            "resources": {
                "runner_role": "family-certify",
                "cache_policy": "immutable-local",
                "estimated_model_bytes": 1024,
            },
            "notes": "fixture",
        }

    def _dry_run(self, *args: str) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            bin_dir = temp / "bin"
            bin_dir.mkdir()
            for command in ("hf",):
                executable = bin_dir / command
                executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
                executable.chmod(executable.stat().st_mode | stat.S_IXUSR)

            manifest = temp / "manifest.json"
            manifest.write_text(
                json.dumps(self._manifest(self._model())) + "\n", encoding="utf-8"
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
        self.assertTrue(
            commands[0]
            .strip()
            .endswith("--require-lanes --skip-build --skip-speculative")
        )

    def test_family_filter_limits_the_resolved_dry_run(self) -> None:
        selected = self._dry_run("--families", "test-family")
        self.assertEqual(0, selected.returncode, selected.stderr)
        self.assertIn("--family test-family", selected.stdout)

        omitted = self._dry_run("--families", "another-family")
        self.assertEqual(2, omitted.returncode)
        self.assertIn("unknown selected families: another-family", omitted.stderr)
        self.assertNotIn("--family test-family", omitted.stdout)

    def test_skip_build_omits_the_one_time_build(self) -> None:
        result = self._dry_run("--skip-build")
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertNotIn("cargo build -p skippy-correctness", result.stdout)

    def test_preflight_pins_snapshot_and_limits_speculative_corpus_to_mtp(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            revision = "a" * 40
            model = (
                temp
                / "hf"
                / "hub"
                / "models--org--mtp-model"
                / "snapshots"
                / revision
                / "model.gguf"
            )
            model.parent.mkdir(parents=True)
            model.write_bytes(b"gguf-fixture")

            bin_dir = temp / "bin"
            bin_dir.mkdir()
            hf = bin_dir / "hf"
            hf.write_text('#!/bin/sh\nprintf "path=%s\\n" "$FAKE_MODEL_PATH"\n', encoding="utf-8")
            inspect = bin_dir / "skippy-model-package"
            complete_tensors = [
                {
                    "name": f"blk.{layer}.weight",
                    "layer_index": layer,
                    "role": "layer",
                    "ggml_type": 1,
                    "byte_size": 0,
                }
                for layer in range(5)
            ] + [
                {
                    "name": "blk.5.nextn.eh_proj.weight",
                    "layer_index": 5,
                    "role": "layer",
                    "ggml_type": 1,
                    "byte_size": 1024,
                },
                {
                    "name": "blk.5.nextn.enorm.weight",
                    "layer_index": 5,
                    "role": "layer",
                    "ggml_type": 1,
                    "byte_size": 0,
                },
                {
                    "name": "blk.5.nextn.hnorm.weight",
                    "layer_index": 5,
                    "role": "layer",
                    "ggml_type": 1,
                    "byte_size": 0,
                },
            ]
            complete_scan = json.dumps(
                {"tensor_count": len(complete_tensors), "tensors": complete_tensors},
                separators=(",", ":"),
            )
            inspect.write_text(
                f"#!/bin/sh\nprintf '%s\\n' '{complete_scan}'\n",
                encoding="utf-8",
            )
            spec = bin_dir / "llama-spec-bench"
            spec.write_text(
                "#!/bin/sh\n"
                "while [ \"$#\" -gt 0 ]; do\n"
                "  if [ \"$1\" = --json-out ]; then printf '{}\\n' > \"$2\"; exit 0; fi\n"
                "  shift\n"
                "done\n"
                "exit 1\n",
                encoding="utf-8",
            )
            for name in ("hf", "skippy-model-package", "llama-spec-bench"):
                path = bin_dir / name
                path.chmod(path.stat().st_mode | stat.S_IXUSR)
            for name in ("skippy-correctness", "skippy-server"):
                path = bin_dir / name
                path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
                path.chmod(path.stat().st_mode | stat.S_IXUSR)

            model_policy = self._model(revision)
            model_policy["family"] = "mtp-family"
            model_policy["artifact"] = {
                "repo": "org/mtp-model",
                "revision": revision,
                "files": ["model.gguf"],
                "selector": "Q4_K_M",
            }
            model_policy["execution"] = {
                "trunk_layers": 5,
                "mtp_layers": 1,
                "activation_width": 1024,
                "boundary_sweep_period": 0,
                "speculative_policy": "mtp-if-present",
            }
            manifest = temp / "manifest.json"
            manifest.write_text(
                json.dumps(self._manifest(model_policy)) + "\n", encoding="utf-8"
            )
            artifacts = temp / "artifacts"
            env = os.environ.copy()
            env.pop("HF_CACHE", None)
            env.pop("HF_HUB_OFFLINE", None)
            env.update(
                {
                    "FAKE_MODEL_PATH": str(model),
                    "FAMILY_BATTERY_BIN_DIR": str(bin_dir),
                    "FAMILY_BATTERY_ARTIFACT_ROOT": str(artifacts),
                    "FAMILY_BATTERY_MIN_FREE_GIB": "0",
                    "HF_HOME": str(temp / "hf"),
                    "PATH": f"{bin_dir}{os.pathsep}{env['PATH']}",
                }
            )
            result = subprocess.run(
                [
                    str(BATTERY),
                    "--manifest",
                    str(manifest),
                    "--preflight-only",
                    "--skip-build",
                ],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
                timeout=30,
            )
            self.assertEqual(0, result.returncode, result.stderr)
            run_dir = next(artifacts.iterdir())
            resolved = (run_dir / "resolved-models.tsv").read_text(encoding="utf-8")
            self.assertIn(revision, resolved)
            self.assertIn("|1|1024|5|", resolved)
            mtp_corpus = (run_dir / "mtp-corpus.tsv").read_text(encoding="utf-8")
            self.assertIn("mtp-family", mtp_corpus)
            self.assertTrue((run_dir / "preflight" / "speculative-smoke.json").is_file())

            incomplete_tensors = complete_tensors[:5] + [complete_tensors[5]]
            incomplete_scan = json.dumps(
                {"tensor_count": len(incomplete_tensors), "tensors": incomplete_tensors},
                separators=(",", ":"),
            )
            inspect.write_text(
                f"#!/bin/sh\nprintf '%s\\n' '{incomplete_scan}'\n", encoding="utf-8"
            )
            incomplete_artifacts = temp / "incomplete-artifacts"
            env["FAMILY_BATTERY_ARTIFACT_ROOT"] = str(incomplete_artifacts)
            incomplete = subprocess.run(
                [
                    str(BATTERY),
                    "--manifest",
                    str(manifest),
                    "--preflight-only",
                    "--skip-build",
                ],
                cwd=ROOT,
                env=env,
                text=True,
                capture_output=True,
                check=False,
                timeout=30,
            )
            self.assertEqual(1, incomplete.returncode)
            incomplete_run = next(incomplete_artifacts.iterdir())
            incomplete_corpus = (incomplete_run / "mtp-corpus.tsv").read_text(
                encoding="utf-8"
            )
            self.assertEqual(
                ["family\tmodel_id\tsource_revision\tmodel_path\tmtp_layers"],
                incomplete_corpus.splitlines(),
            )
            self.assertFalse(
                (incomplete_run / "preflight" / "speculative-smoke.json").exists()
            )


if __name__ == "__main__":
    unittest.main()
