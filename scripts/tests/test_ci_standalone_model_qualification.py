"""Contract checks for the composed-standalone model qualification driver."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "skippy/scripts/ci-standalone-model-qualification.py"
SPEC = importlib.util.spec_from_file_location("ci_standalone_model_qualification", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)
contract = driver.contract


class FakeServer:
    def __init__(self, log: Path) -> None:
        self.base = "http://127.0.0.1:1"
        self.model_id = "pinned"
        self.log_path = log
        self.restarts = 0

    def stop(self) -> None:
        pass

    def __enter__(self) -> "FakeServer":
        self.restarts += 1
        return self


def response(prompt_tokens: int, cached: int = 0) -> dict:
    return {"model": "pinned", "choices": [{"message": {"content": "answer"}}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 2,
                      "prompt_tokens_details": {"cached_tokens": cached}}}


class HelperTests(unittest.TestCase):
    def test_usage_tokens_accepts_positive_counts(self) -> None:
        usage = driver.usage_tokens(response(5))
        self.assertEqual(usage["prompt_tokens"], 5)
        self.assertEqual(usage["completion_tokens"], 2)

    def test_usage_tokens_rejects_zero_decode(self) -> None:
        bad = response(5)
        bad["usage"]["completion_tokens"] = 0
        with self.assertRaisesRegex(ValueError, "positive decode"):
            driver.usage_tokens(bad)

    def test_cached_tokens_defaults_to_zero_and_reads_details(self) -> None:
        self.assertEqual(driver.cached_tokens({"usage": {}}), 0)
        self.assertEqual(driver.cached_tokens(response(5, 7)), 7)

    def test_parse_model_requires_four_fields(self) -> None:
        self.assertEqual(driver.parse_model([["a", "b/model.gguf", "c" * 64, "id"]]),
                         [("a", Path("b/model.gguf"), "c" * 64, "id")])


class CaseBatteryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.log = Path(self.temporary.name) / "serve.log"
        self.log.write_text("log", encoding="utf-8")
        self.server = FakeServer(self.log)

    def patch(self, prompts: list[int], cached: int = 0, stream: dict | None = None) -> None:
        counter = {"index": 0}
        original_completion = driver.completion
        original_stream = driver.stream_completion

        def completion(_base: str, _model_id: str, _messages: list, **_options) -> dict:
            index = min(counter["index"], len(prompts) - 1)
            counter["index"] += 1
            return response(prompts[index], cached)

        def restore() -> None:
            driver.completion = original_completion
            driver.stream_completion = original_stream

        driver.completion = completion
        driver.stream_completion = lambda *_a, **_k: stream or {"deltas": 3, "bytes": 40}
        self.addCleanup(restore)

    def test_dense_battery_executes_the_contract_cases(self) -> None:
        self.patch([5, 12, 5])
        observations: dict = {}
        self.assertEqual(driver.dense_cases(self.server, observations),
                         ["load", "prefill-decode", "stream", "continuation"])
        self.assertEqual(observations["prefill_tokens"], 5)
        self.assertEqual(observations["stream_deltas"], 3)

    def test_recurrent_battery_grows_the_prompt(self) -> None:
        self.patch([4, 9, 15])
        observations: dict = {}
        self.assertEqual(driver.recurrent_cases(self.server, observations),
                         ["prefill-decode", "state-preservation"])
        self.assertEqual(observations["state_prompt_tokens"], 15)

    def test_recurrent_battery_rejects_flat_state(self) -> None:
        self.patch([4, 4, 4])
        with self.assertRaisesRegex(ValueError, "did not grow"):
            driver.recurrent_cases(self.server, {})

    def test_kv_battery_requires_a_positive_prefix_hit(self) -> None:
        self.patch([32, 32, 45, 32, 5], cached=0)
        with self.assertRaisesRegex(ValueError, "not served from cache"):
            driver.kv_cases(self.server, {})

    def test_kv_battery_accepts_a_warm_prefix(self) -> None:
        self.patch([32, 32, 45, 32, 5], cached=16)
        observations: dict = {}
        self.assertEqual(sorted(driver.kv_cases(self.server, observations)),
                         ["divergent-prefix", "isolation", "suffix-continuation"])
        self.assertEqual(observations["cached_tokens"], 16)

    def test_moe_battery_requires_stable_prefill(self) -> None:
        self.patch([20, 21, 30])
        with self.assertRaisesRegex(ValueError, "changed the prefill"):
            driver.moe_cases(self.server, {})

    def test_every_suite_driver_names_cover_the_contract(self) -> None:
        drivers = {
            "dense": (driver.dense_cases, ["load", "prefill-decode", "stream", "continuation"], ["restart"]),
            "recurrent": (driver.recurrent_cases, ["prefill-decode", "state-preservation"], ["restart"]),
            "moe": (driver.moe_cases, ["expert-execution", "repeated-restore", "suffix-continuation"], []),
            "kv-cache": (driver.kv_cases, ["isolation", "divergent-prefix", "suffix-continuation"],
                         ["dense-prefix-hit", "recurrent-prefix-hit"]),
        }
        self.assertEqual(set(drivers), set(driver.SUITE_DRIVERS))
        for suite, (_, names, extra) in drivers.items():
            with self.subTest(suite=suite):
                self.assertEqual(set(names) | set(extra), contract.SUITE_CASES[suite])
        self.assertEqual(driver.PREFIX_HIT_CASES["dense"], "dense-prefix-hit")
        self.assertEqual(driver.PREFIX_HIT_CASES["hybrid"], "recurrent-prefix-hit")


class SuiteModelTagTests(unittest.TestCase):
    def models(self, *identifiers: str) -> list[tuple[str, Path, str, str]]:
        return [(identifier, Path(f"{identifier}.gguf"), "c" * 64, identifier)
                for identifier in identifiers]

    def test_dense_suite_accepts_a_model_carrying_its_tag(self) -> None:
        driver.validate_suite_models("dense", self.models("a"), {"a": {"dense", "gguf"}})

    def test_dense_suite_rejects_a_model_without_the_tag(self) -> None:
        with self.assertRaisesRegex(ValueError, "lacks a model tagged"):
            driver.validate_suite_models("dense", self.models("a"), {"a": {"hybrid"}})

    def test_kv_suite_needs_both_dense_and_recurrent_models(self) -> None:
        with self.assertRaisesRegex(ValueError, "lacks a model tagged"):
            driver.validate_suite_models("kv-cache", self.models("a"), {"a": {"dense"}})
        driver.validate_suite_models("kv-cache", self.models("a", "b"),
                                     {"a": {"dense"}, "b": {"hybrid"}})

    def test_unknown_artifact_never_satisfies_a_capability_group(self) -> None:
        with self.assertRaisesRegex(ValueError, "lacks a model tagged"):
            driver.validate_suite_models("dense", self.models("missing"), {})

    def test_prefix_is_long_enough_for_a_reported_cache_hit(self) -> None:
        # Measured on the composed product: a ~130-token repeat still reports
        # cached_tokens=0, while a ~500-token repeat is served from cache on
        # both Metal and CPU. Keep the shared prefix well above that boundary.
        self.assertGreaterEqual(len(driver.PREFIX.split()), 160)


if __name__ == "__main__":
    unittest.main()
