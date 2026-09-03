from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "run-event-benchmark-matrix.py"
RUST_OUTPUT_TYPES = (
    ROOT / "crates" / "mesh-llm-commands" / "src" / "gpus" / "tune" / "output_types.rs"
)


def load_module():
    module_name = "run_event_benchmark_matrix"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT)
    module = importlib.util.module_from_spec(spec)
    # Registering in sys.modules BEFORE exec_module is required for
    # `dataclasses` to resolve `cls.__module__` on Python 3.9 (its
    # `_is_type` helper looks the module up via `sys.modules`); omitting
    # this raises `AttributeError: 'NoneType' object has no attribute
    # '__dict__'` the moment the loaded module defines a dataclass.
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_rust_string_literal(raw: str) -> str:
    joined = re.sub(r"\\\s*\n\s*", " ", raw)
    return normalize_whitespace(joined)


class ModeAliasMappingTests(unittest.TestCase):
    def test_production_and_event_disabled_map_to_identical_wire_values(self):
        harness = load_module()
        self.assertEqual(harness.resolve_trial_env_value("production"), "production")
        self.assertEqual(harness.resolve_trial_env_value("event-disabled"), "event-disabled")

    def test_unknown_mode_is_a_hard_error(self):
        harness = load_module()
        with self.assertRaises(ValueError):
            harness.resolve_trial_env_value("bogus")

    def test_valid_modes_matches_the_alias_map_keys(self):
        harness = load_module()
        self.assertEqual(set(harness.VALID_MODES), {"production", "event-disabled"})


class SeedValidationTests(unittest.TestCase):
    def test_accepts_zero_and_u64_max(self):
        harness = load_module()
        harness.validate_seed(0)
        harness.validate_seed(harness.U64_MAX)

    def test_rejects_negative(self):
        harness = load_module()
        with self.assertRaises(ValueError):
            harness.validate_seed(-1)

    def test_rejects_above_u64_max(self):
        harness = load_module()
        with self.assertRaises(ValueError):
            harness.validate_seed(harness.U64_MAX + 1)


class EnvironmentRedactionTests(unittest.TestCase):
    def test_allowlisted_names_persist_normalized_raw_values(self):
        harness = load_module()
        snapshot = harness.capture_environment_snapshot(
            {
                "MESH_LLM_LIFECYCLE_LOG_PARSER": "auto",
                "MESH_LLM_BENCHMARK_TUNE_TRIAL": "1",
                "MESH_LLM_EVENT_SYSTEM_TRIAL_MODE": "event-disabled",
            }
        )
        self.assertEqual(
            snapshot["MESH_LLM_LIFECYCLE_LOG_PARSER"], {"value": "auto", "redacted": False}
        )
        self.assertEqual(
            snapshot["MESH_LLM_BENCHMARK_TUNE_TRIAL"], {"value": True, "redacted": False}
        )
        self.assertEqual(
            snapshot["MESH_LLM_EVENT_SYSTEM_TRIAL_MODE"],
            {"value": "event-disabled", "redacted": False},
        )

    def test_non_allowlisted_mesh_llm_names_are_redacted_to_presence_only(self):
        harness = load_module()
        snapshot = harness.capture_environment_snapshot({"MESH_LLM_CONFIG": "/tmp/secret-path.toml"})
        self.assertEqual(
            snapshot["MESH_LLM_CONFIG"], {"value": harness.REDACTED_PRESENT, "redacted": True}
        )
        self.assertNotIn("/tmp/secret-path.toml", json.dumps(snapshot))

    def test_names_matching_sensitive_pattern_are_always_redacted(self):
        harness = load_module()
        # Simulates a hypothetical future allowlist collision: even a name
        # containing a sensitive substring must redact, defense in depth
        # beyond the hand-curated allowlist.
        snapshot = harness.capture_environment_snapshot({"MESH_LLM_API_TOKEN": "abc123"})
        self.assertTrue(snapshot["MESH_LLM_API_TOKEN"]["redacted"])
        self.assertNotIn("abc123", json.dumps(snapshot))

    def test_non_mesh_llm_names_are_ignored_entirely(self):
        harness = load_module()
        snapshot = harness.capture_environment_snapshot({"HOME": "/Users/example", "PATH": "/usr/bin"})
        self.assertEqual(snapshot, {})


class BinaryIdentityTests(unittest.TestCase):
    def test_missing_binary_reports_none_sha256_and_none_version(self):
        harness = load_module()
        identity = harness.capture_binary_identity(
            Path("/nonexistent/mesh-llm"), run_version=lambda _binary: None
        )
        self.assertIsNone(identity["sha256"])
        self.assertIsNone(identity["version"])
        self.assertTrue(identity["path"].endswith("mesh-llm"))

    def test_existing_file_hashes_deterministically(self):
        harness = load_module()
        with self._temp_file(b"fake-binary-bytes") as path:
            identity = harness.capture_binary_identity(path, run_version=lambda _b: "mesh-llm 0.76.0")
            self.assertIsNotNone(identity["sha256"])
            self.assertEqual(len(identity["sha256"]), 64)
            self.assertEqual(identity["version"], "mesh-llm 0.76.0")

    @staticmethod
    def _temp_file(data: bytes):
        import contextlib
        import tempfile

        @contextlib.contextmanager
        def _ctx():
            with tempfile.NamedTemporaryFile(delete=False) as handle:
                handle.write(data)
                handle.flush()
                path = Path(handle.name)
            try:
                yield path
            finally:
                path.unlink(missing_ok=True)

        return _ctx()


class TrialPlanDeterminismTests(unittest.TestCase):
    def test_same_seed_produces_identical_plans_across_invocations(self):
        harness = load_module()
        plan_a = harness.build_trial_plan(42, 3, 2, ["chat_short", "chat_long"])
        plan_b = harness.build_trial_plan(42, 3, 2, ["chat_short", "chat_long"])
        self.assertEqual(plan_a, plan_b)

    def test_different_seeds_produce_different_prompt_seeds(self):
        harness = load_module()
        plan_a = harness.build_trial_plan(1, 2, 1, ["s"])
        plan_b = harness.build_trial_plan(2, 2, 1, ["s"])
        self.assertNotEqual(
            [entry.prompt_seed for entry in plan_a], [entry.prompt_seed for entry in plan_b]
        )

    def test_plan_shape_covers_primary_then_each_scenario_in_order(self):
        harness = load_module()
        plan = harness.build_trial_plan(1, 2, 3, ["alpha", "beta"])
        scenarios = [entry.scenario for entry in plan]
        self.assertEqual(
            scenarios,
            [harness.PRIMARY_SCENARIO] * 2 + ["alpha"] * 3 + ["beta"] * 3,
        )
        primary_indices = [entry.pair_index for entry in plan if entry.scenario == harness.PRIMARY_SCENARIO]
        self.assertEqual(primary_indices, [0, 1])

    def test_zero_pairs_primary_is_rejected(self):
        harness = load_module()
        with self.assertRaises(ValueError):
            harness.build_trial_plan(1, 0, 1, ["s"])

    def test_zero_pairs_scenario_is_rejected(self):
        harness = load_module()
        with self.assertRaises(ValueError):
            harness.build_trial_plan(1, 1, 0, ["s"])

    def test_no_scenarios_is_rejected(self):
        harness = load_module()
        with self.assertRaises(ValueError):
            harness.build_trial_plan(1, 1, 1, [])

    def test_side_order_first_is_a_valid_mode_for_every_entry(self):
        harness = load_module()
        plan = harness.build_trial_plan(7, 3, 2, ["alpha"])
        for entry in plan:
            self.assertIn(entry.side_order_first, harness.VALID_MODES)

    def test_side_order_first_uses_both_modes_across_many_seeds(self):
        """Side order is minted from the SAME deterministic per-plan rng
        that mints prompt_seed (see build_trial_plan), so re-running with
        the same seed/counts/scenarios always reproduces the same order --
        proven separately by
        test_same_seed_produces_identical_plans_across_invocations, since
        TrialPlanEntry equality now covers side_order_first too. This test
        instead proves the ordering genuinely VARIES rather than being a
        constant: across many different seeds, both modes must appear."""
        harness = load_module()
        observed = {
            harness.build_trial_plan(seed, 1, 1, ["s"])[0].side_order_first for seed in range(50)
        }
        self.assertEqual(observed, set(harness.VALID_MODES))


class DecodeOnlyTokSTests(unittest.TestCase):
    def test_null_when_ttft_is_null(self):
        harness = load_module()
        self.assertIsNone(harness.compute_decode_only_tok_s(100, 5000.0, None))

    def test_null_when_decode_interval_is_zero(self):
        harness = load_module()
        self.assertIsNone(harness.compute_decode_only_tok_s(100, 500.0, 500.0))

    def test_null_when_decode_interval_is_negative(self):
        harness = load_module()
        self.assertIsNone(harness.compute_decode_only_tok_s(100, 500.0, 600.0))

    def test_computed_value_uses_the_epsilon_guarded_interval(self):
        harness = load_module()
        # completion_tokens=100 over a 4.5s decode interval (5.0s total - 0.5s ttft)
        value = harness.compute_decode_only_tok_s(100, 5000.0, 500.0)
        self.assertAlmostEqual(value, 100 / 4.5, places=6)

    def test_never_returns_zero_on_failure_paths(self):
        harness = load_module()
        for args in [(None, 1.0, 1.0), (1, None, 1.0), (1, 1.0, None), (1, 0.0, 5.0)]:
            self.assertIsNone(harness.compute_decode_only_tok_s(*args))


class DecodeTokSTests(unittest.TestCase):
    def test_historical_definition_preserved(self):
        harness = load_module()
        self.assertAlmostEqual(harness.compute_decode_tok_s(100, 2000.0), 50.0)

    def test_null_on_non_positive_elapsed(self):
        harness = load_module()
        self.assertIsNone(harness.compute_decode_tok_s(100, 0.0))
        self.assertIsNone(harness.compute_decode_tok_s(None, 2000.0))


class ModelIdResolutionTests(unittest.TestCase):
    """`--local-model-only` rejects `model: "auto"` with `404 model_not_found`
    (no mesh/routing layer to resolve the alias) -- confirmed against a real
    running binary. `build_chat_request_body` must therefore send the model
    id resolved from `/v1/models`, never a hardcoded `"auto"`."""

    def test_build_chat_request_body_uses_the_passed_model_not_auto(self):
        harness = load_module()
        body = harness.build_chat_request_body("prompt", 16, "local-gguf/sha256-abc")
        self.assertEqual(body["model"], "local-gguf/sha256-abc")
        self.assertNotEqual(body["model"], "auto")

    def test_first_models_list_id_extracts_the_first_entry(self):
        harness = load_module()
        payload = {"object": "list", "data": [{"id": "local-gguf/sha256-abc", "object": "model"}]}
        self.assertEqual(harness.first_models_list_id(payload), "local-gguf/sha256-abc")

    def test_first_models_list_id_is_none_on_empty_data(self):
        harness = load_module()
        self.assertIsNone(harness.first_models_list_id({"object": "list", "data": []}))

    def test_first_models_list_id_is_none_on_malformed_payload(self):
        harness = load_module()
        self.assertIsNone(harness.first_models_list_id({}))
        self.assertIsNone(harness.first_models_list_id({"data": "not-a-list"}))
        self.assertIsNone(harness.first_models_list_id({"data": [{"id": ""}]}))


class SseStreamParsingTests(unittest.TestCase):
    def test_happy_path_extracts_ttft_and_completion_tokens(self):
        # `parse_sse_stream` calls `clock()` exactly once, at the moment
        # the first non-empty content delta is seen -- the fake returns
        # `started_at + 0.25s` on that single call.
        harness = load_module()
        lines = [
            'data: {"choices":[{"delta":{"content":""}}]}\n',
            'data: {"choices":[{"delta":{"content":"hi"}}]}\n',
            'data: {"choices":[{"delta":{}}],"usage":{"completion_tokens":7}}\n',
            "data: [DONE]\n",
        ]
        result = harness.parse_sse_stream(lines, clock=lambda: 0.25, started_at=0.0)
        self.assertEqual(result.completion_tokens, 7)
        self.assertAlmostEqual(result.ttft_ms, 250.0, places=3)
        self.assertFalse(result.malformed)

    def test_split_chunks_and_empty_deltas_do_not_set_ttft(self):
        harness = load_module()
        lines = [
            'data: {"choices":[{"delta":{"content":""}}]}\n',
            'data: {"choices":[{"delta":{}}]}\n',
            'data: {"choices":[{"delta":{"content":"ok"}}],"usage":{"completion_tokens":3}}\n',
            "data: [DONE]\n",
        ]
        result = harness.parse_sse_stream(lines, clock=lambda: 0.1, started_at=0.0)
        self.assertAlmostEqual(result.ttft_ms, 100.0, places=3)
        self.assertEqual(result.completion_tokens, 3)

    def test_malformed_json_line_is_skipped_not_fatal(self):
        harness = load_module()
        lines = [
            "data: {not valid json\n",
            'data: {"choices":[{"delta":{"content":"x"}}],"usage":{"completion_tokens":1}}\n',
            "data: [DONE]\n",
        ]
        result = harness.parse_sse_stream(lines, clock=lambda: 0.05, started_at=0.0)
        self.assertEqual(result.completion_tokens, 1)
        self.assertFalse(result.malformed)

    def test_stream_without_terminal_usage_is_malformed_with_null_tokens(self):
        harness = load_module()
        lines = ['data: {"choices":[{"delta":{"content":"x"}}]}\n', "data: [DONE]\n"]
        result = harness.parse_sse_stream(lines, clock=lambda: 0.0, started_at=0.0)
        self.assertIsNone(result.completion_tokens)
        self.assertTrue(result.malformed)

    def test_non_data_lines_are_ignored(self):
        harness = load_module()
        lines = [
            ": keepalive\n",
            "\n",
            'data: {"choices":[{"delta":{"content":"x"}}],"usage":{"completion_tokens":2}}\n',
            "data: [DONE]\n",
        ]
        result = harness.parse_sse_stream(lines, clock=lambda: 0.0, started_at=0.0)
        self.assertEqual(result.completion_tokens, 2)


class HealthExpectationTests(unittest.TestCase):
    def test_event_disabled_expects_exact_attempted_count_dropped(self):
        harness = load_module()
        results = [_fake_trial_result() for _ in range(5)]
        expectations = harness.summarize_health_expectations("event-disabled", results)
        self.assertEqual(expectations, {"expected_dropped_progress": 5, "expected_dropped_diagnostic": 5})

    def test_production_expects_zero_drops(self):
        harness = load_module()
        results = [_fake_trial_result() for _ in range(5)]
        expectations = harness.summarize_health_expectations("production", results)
        self.assertEqual(expectations, {"expected_dropped_progress": 0, "expected_dropped_diagnostic": 0})


def _fake_trial_result():
    """`summarize_health_expectations` only counts `results` -- content is
    irrelevant, so a plain placeholder is enough."""
    return object()


class ManifestBuildingTests(unittest.TestCase):
    def test_manifest_carries_schema_and_trial_unit_and_expectations(self):
        harness = load_module()
        result = harness.TrialResult(
            scenario=harness.PRIMARY_SCENARIO,
            pair_index=0,
            side_order_first="production",
            status="succeeded",
            completion_tokens=10,
            elapsed_ms=1000.0,
            decode_tok_s=10.0,
            ttft_ms=100.0,
            decode_only_tok_s=11.1,
            setup_ms=50.0,
            readiness_ms=200.0,
            shutdown_ms=25.0,
        )
        manifest = harness.build_manifest(
            binary=Path("/nonexistent/mesh-llm"),
            model="fixture-model",
            mode="event-disabled",
            seed=7,
            pairs_primary=1,
            pairs_scenario=1,
            scenarios=["chat_short"],
            results=[result],
            environ={"MESH_LLM_BENCHMARK_TUNE_TRIAL": "1", "MESH_LLM_EVENT_SYSTEM_TRIAL_MODE": "event-disabled"},
            generated_at="2026-01-01T00:00:00Z",
            run_version=lambda _b: None,
        )
        self.assertEqual(manifest["schema_version"], harness.MANIFEST_SCHEMA_VERSION)
        self.assertEqual(manifest["metrics_schema"], "streaming_v1")
        self.assertEqual(manifest["mode"], "event-disabled")
        self.assertEqual(manifest["trial_unit"], harness.TRIAL_UNIT_DEFINITION)
        self.assertEqual(manifest["expected_dropped_progress"], 1)
        self.assertEqual(manifest["expected_dropped_diagnostic"], 1)
        self.assertEqual(len(manifest["trials"]), 1)
        self.assertEqual(manifest["trials"][0]["status"], "succeeded")

    def test_manifest_records_a_non_default_attempt_number(self):
        harness = load_module()
        manifest = harness.build_manifest(
            binary=Path("/nonexistent/mesh-llm"),
            model="fixture-model",
            mode="production",
            seed=1,
            pairs_primary=1,
            pairs_scenario=1,
            scenarios=["s"],
            results=[],
            environ={},
            attempt=2,
            generated_at="2026-01-01T00:00:00Z",
            run_version=lambda _b: None,
        )
        self.assertEqual(manifest["attempt"], 2)

    def test_manifest_health_is_null_when_not_collected(self):
        """Defect A: no real call site collects health data (see
        summarize_health_expectations's own docstring on what CAN be
        proven without a live console API); build_manifest must report
        that honestly as JSON null, never a silently-zero {}."""
        harness = load_module()
        manifest = harness.build_manifest(
            binary=Path("/nonexistent/mesh-llm"),
            model="fixture-model",
            mode="production",
            seed=1,
            pairs_primary=1,
            pairs_scenario=1,
            scenarios=["s"],
            results=[],
            environ={},
            generated_at="2026-01-01T00:00:00Z",
            run_version=lambda _b: None,
        )
        self.assertIsNone(manifest["health"])

    def test_manifest_preserves_an_explicitly_supplied_health_dict(self):
        """A future caller that CAN collect real health data must have its
        value pass through unchanged -- build_manifest never overwrites a
        caller-supplied health block."""
        harness = load_module()
        supplied = {"terminal_delivery_failed": 0, "dropped_progress": 3, "dropped_diagnostic": 3}
        manifest = harness.build_manifest(
            binary=Path("/nonexistent/mesh-llm"),
            model="fixture-model",
            mode="event-disabled",
            seed=1,
            pairs_primary=1,
            pairs_scenario=1,
            scenarios=["s"],
            results=[],
            environ={},
            generated_at="2026-01-01T00:00:00Z",
            run_version=lambda _b: None,
            health=supplied,
        )
        self.assertEqual(manifest["health"], supplied)


class MainThreadsAttemptIntoManifestTests(unittest.TestCase):
    """`main()` must forward the parsed `--attempt` value into
    `build_manifest` -- a source-level check (rather than invoking `main()`,
    which spawns a REAL trial subprocess by default) mirroring this file's
    existing source-inspection convention (see `HiddenSelectorWiringTests`)."""

    def test_main_passes_attempt_from_args_to_build_manifest(self):
        source = SCRIPT.read_text()
        main_start = source.index("def main(")
        main_body = source[main_start:]
        self.assertIn("attempt=args.attempt", main_body)


class TrialUnitDefinitionMatchesRustSourceTests(unittest.TestCase):
    def test_trial_and_pair_wording_matches_rust_verbatim(self):
        harness = load_module()
        rust_source = RUST_OUTPUT_TYPES.read_text()
        match = re.search(
            r"benchmark_trial_unit_definition\(\).*?trial:\s*\"(?P<trial>.*?)\"\s*\.to_string\(\)"
            r".*?pair:\s*\"(?P<pair>.*?)\"\s*\.to_string\(\)",
            rust_source,
            re.DOTALL,
        )
        self.assertIsNotNone(match, "could not locate benchmark_trial_unit_definition() in the Rust source")
        rust_trial = normalize_rust_string_literal(match.group("trial"))
        rust_pair = normalize_rust_string_literal(match.group("pair"))
        self.assertEqual(normalize_whitespace(harness.TRIAL_UNIT_DEFINITION["trial"]), rust_trial)
        self.assertEqual(normalize_whitespace(harness.TRIAL_UNIT_DEFINITION["pair"]), rust_pair)


class HostAndThermalCaptureTests(unittest.TestCase):
    def test_certification_host_classification_for_macos_arm64(self):
        harness = load_module()
        # capture_host_classification reads the REAL platform; verify the
        # lookup table it consults instead, which is what actually
        # determines certification-host status.
        self.assertEqual(
            harness.CERTIFICATION_HOSTS[("Darwin", "arm64")], "macos-arm64-metal"
        )
        self.assertEqual(
            harness.CERTIFICATION_HOSTS[("Linux", "x86_64")], "linux-x86_64-cuda"
        )

    def test_unknown_host_is_informational_only(self):
        harness = load_module()
        self.assertNotIn(("Windows", "AMD64"), harness.CERTIFICATION_HOSTS)

    def test_thermal_state_capture_never_raises_and_has_available_flag(self):
        harness = load_module()
        state = harness.capture_thermal_state(run_pmset=lambda: None, thermal_root=Path("/nonexistent"))
        self.assertIn("available", state)


class HiddenSelectorWiringTests(unittest.TestCase):
    def test_execute_trial_always_sets_gate_and_selector_together(self):
        harness = load_module()
        source = SCRIPT.read_text()
        needle = 'env[TRIAL_GATE_ENV_NAME] = "1"\n    env[TRIAL_ENV_NAME] = resolve_trial_env_value(mode)'
        self.assertIn(needle, source)


class LocalModelOnlyCliCompatibilityTests(unittest.TestCase):
    """`--local-model-only` rejects `--headless` at CLI validation
    (`validate_local_model_only_options` in
    `crates/mesh-llm-host-runtime/src/runtime/local_model_only.rs`: "never
    starts a console; remove --headless") and never starts a console/
    management API at all ("does not start owner control or management
    APIs", same function) -- so `execute_trial`'s argv must never pass
    `--console` or `--headless` alongside `--local-model-only`, or every
    real trial launch fails at startup before readiness is even polled."""

    def test_argv_never_passes_console_or_headless(self):
        harness = load_module()
        source = SCRIPT.read_text()
        argv_start = source.index("argv = [", source.index("def execute_trial"))
        argv_end = source.index("]", argv_start)
        argv_block = source[argv_start:argv_end]
        self.assertNotIn("--console", argv_block)
        self.assertNotIn("--headless", argv_block)
        self.assertIn("--local-model-only", argv_block)

    def test_argv_always_disables_native_mtp_speculative_decoding(self):
        """Confirmed against a real running binary: native in-model MTP
        crashes every inference request (`llama_decode failed` /
        `backend sampling requires at most one output token per sequence`)
        unless `--speculative-strategy disabled` is passed; `--no-draft`
        does not help (it only covers separate sibling draft-model files)."""
        harness = load_module()
        source = SCRIPT.read_text()
        argv_start = source.index("argv = [", source.index("def execute_trial"))
        argv_end = source.index("]", argv_start)
        argv_block = source[argv_start:argv_end]
        self.assertIn("--speculative-strategy", argv_block)
        self.assertIn('"disabled"', argv_block)


class CliParsingTests(unittest.TestCase):
    def test_help_lists_every_frozen_flag(self):
        harness = load_module()
        parser = harness.build_arg_parser()
        help_text = parser.format_help()
        for flag in (
            "--binary",
            "--model",
            "--output-dir",
            "--pairs-primary",
            "--pairs-scenario",
            "--seed",
            "--mode",
            "--scenario",
        ):
            self.assertIn(flag, help_text)

    def test_mode_choices_are_exactly_production_and_event_disabled(self):
        harness = load_module()
        parser = harness.build_arg_parser()
        args = parser.parse_args(
            [
                "--binary",
                "/bin/true",
                "--model",
                "fixture-model",
                "--output-dir",
                "/tmp/out",
                "--pairs-primary",
                "20",
                "--pairs-scenario",
                "10",
                "--seed",
                "42",
                "--mode",
                "event-disabled",
                "--scenario",
                "chat_short",
            ]
        )
        self.assertEqual(args.mode, "event-disabled")
        self.assertEqual(args.scenarios, ["chat_short"])

    def test_attempt_defaults_to_one(self):
        harness = load_module()
        parser = harness.build_arg_parser()
        args = parser.parse_args(
            [
                "--binary",
                "/bin/true",
                "--model",
                "m",
                "--output-dir",
                "/tmp/out",
                "--pairs-primary",
                "1",
                "--pairs-scenario",
                "1",
                "--seed",
                "1",
                "--mode",
                "production",
                "--scenario",
                "a",
            ]
        )
        self.assertEqual(args.attempt, 1)

    def test_attempt_accepts_explicit_retry_value(self):
        harness = load_module()
        parser = harness.build_arg_parser()
        args = parser.parse_args(
            [
                "--binary",
                "/bin/true",
                "--model",
                "m",
                "--output-dir",
                "/tmp/out",
                "--pairs-primary",
                "1",
                "--pairs-scenario",
                "1",
                "--seed",
                "1",
                "--mode",
                "production",
                "--scenario",
                "a",
                "--attempt",
                "2",
            ]
        )
        self.assertEqual(args.attempt, 2)

    def test_scenario_is_repeatable(self):
        harness = load_module()
        parser = harness.build_arg_parser()
        args = parser.parse_args(
            [
                "--binary",
                "/bin/true",
                "--model",
                "m",
                "--output-dir",
                "/tmp/out",
                "--pairs-primary",
                "1",
                "--pairs-scenario",
                "1",
                "--seed",
                "1",
                "--mode",
                "production",
                "--scenario",
                "a",
                "--scenario",
                "b",
            ]
        )
        self.assertEqual(args.scenarios, ["a", "b"])

    def test_help_exits_zero_via_subprocess(self):
        result = subprocess.run(
            ["python3", str(SCRIPT), "--help"], capture_output=True, text=True, timeout=30, check=False
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("--mode", result.stdout)


class RunTrialPlanInjectionTests(unittest.TestCase):
    def test_run_trial_plan_never_calls_the_real_executor_by_default_in_tests(self):
        """This module's tests must never spawn a real mesh-llm process --
        every call into run_trial_plan below injects a fake executor."""
        harness = load_module()
        plan = harness.build_trial_plan(1, 1, 1, ["s"])
        calls = []

        def fake_executor(binary, model, mode, entry):
            calls.append((binary, model, mode, entry))
            return harness.TrialResult(
                scenario=entry.scenario,
                pair_index=entry.pair_index,
                side_order_first=entry.side_order_first,
                status="succeeded",
                completion_tokens=1,
                elapsed_ms=1.0,
                decode_tok_s=1.0,
                ttft_ms=1.0,
                decode_only_tok_s=1.0,
                setup_ms=1.0,
                readiness_ms=1.0,
                shutdown_ms=1.0,
            )

        results = harness.run_trial_plan(
            Path("/nonexistent"), "model", "production", plan, trial_executor=fake_executor
        )
        self.assertEqual(len(results), len(plan))
        self.assertEqual(len(calls), len(plan))


class SideOrderThreadingTests(unittest.TestCase):
    """`execute_trial` must thread `entry.side_order_first` into every
    `TrialResult` it constructs (all 5 return paths) so the manifest can
    record the deterministic per-pair ordering for the comparator to
    verify -- a source-level check since `execute_trial` spawns a real
    process and is never invoked by this module's own test suite (see the
    module docstring)."""

    def test_execute_trial_threads_side_order_first_into_every_trial_result(self):
        harness = load_module()
        source = SCRIPT.read_text()
        start = source.index("def execute_trial(")
        end = source.index("\ndef ", start + 1)
        body = source[start:end]
        occurrences = body.count("side_order_first=entry.side_order_first")
        self.assertEqual(
            occurrences,
            5,
            f"expected 5 TrialResult(...) constructions to thread side_order_first, found {occurrences}",
        )


if __name__ == "__main__":
    unittest.main()
