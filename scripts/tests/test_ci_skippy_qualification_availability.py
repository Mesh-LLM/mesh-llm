from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "skippy/scripts/ci-qualification-availability.py"
SPEC = importlib.util.spec_from_file_location("ci_qualification_availability", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
policy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(policy)


class AvailabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = "a" * 40
        self.plan = "b" * 64

    def record(self, row: str, **enablement: str) -> dict:
        values = {name: enablement.get(name, "") for name in sorted(policy.ENABLEMENT_VARIABLES)}
        return policy.availability(self.source, self.plan, row, enablement=values)

    def test_every_core_row_has_one_policy(self) -> None:
        self.assertEqual(set(policy.ALWAYS_AVAILABLE) | set(policy.CONDITIONAL) |
                         set(policy.UNAVAILABLE), policy.contract.CORE_ROWS)
        for row in policy.contract.CORE_ROWS:
            with self.subTest(row=row):
                record = self.record(row)
                self.assertEqual(record["row_id"], row)
                self.assertEqual(record["source_sha"], self.source)
                self.assertEqual(record["plan_digest"], self.plan)
                self.assertEqual(record["policy_source"], "protected-ci")
                self.assertIn(record["state"], {"available", "hardware-unavailable"})

    def test_hosted_rows_are_always_available(self) -> None:
        self.assertEqual(policy.contract.REQUIRED_AVAILABLE_ROWS,
                         set(policy.ALWAYS_AVAILABLE))
        for row in policy.contract.REQUIRED_AVAILABLE_ROWS:
            with self.subTest(row=row):
                self.assertEqual(self.record(row)["state"], "available")

    def test_accelerator_rows_require_exact_lowercase_true(self) -> None:
        conditions = {
            "linux-cuda": "MESH_CUDA_INFERENCE_RUNNER_ENABLED",
            "linux-rocm": "MESH_ROCM_INFERENCE_RUNNER_ENABLED",
            "linux-vulkan": "MESH_VULKAN_INFERENCE_RUNNER_ENABLED",
        }
        for row, variable in conditions.items():
            for value in ("", "TRUE", "True", "1", "false"):
                with self.subTest(row=row, value=value):
                    self.assertEqual(self.record(row, **{variable: value})["state"],
                                     "hardware-unavailable")
            with self.subTest(row=row, value="true"):
                self.assertEqual(self.record(row, **{variable: "true"})["state"], "available")

    def test_windows_accelerators_never_queue(self) -> None:
        for row in policy.UNAVAILABLE:
            with self.subTest(row=row):
                self.assertEqual(self.record(
                    row,
                    MESH_CUDA_INFERENCE_RUNNER_ENABLED="true",
                    MESH_ROCM_INFERENCE_RUNNER_ENABLED="true",
                    MESH_VULKAN_INFERENCE_RUNNER_ENABLED="true",
                )["state"], "hardware-unavailable")

    def test_identity_and_row_are_closed(self) -> None:
        for source, plan, row in (("bad", self.plan, "linux-cpu"),
                                  (self.source, "bad", "linux-cpu"),
                                  (self.source, self.plan, "linux-other")):
            with self.subTest(source=source, plan=plan, row=row):
                with self.assertRaises(ValueError):
                    policy.availability(source, plan, row, enablement={
                        name: "" for name in sorted(policy.ENABLEMENT_VARIABLES)})

    def test_enablement_input_is_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "protected variables"):
            policy.availability(self.source, self.plan, "linux-cuda", enablement={})


if __name__ == "__main__":
    unittest.main()
