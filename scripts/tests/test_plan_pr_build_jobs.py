from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[2]
PLANNER_PATH = ROOT / "scripts" / "plan-pr-build-jobs.py"
PLANNER_SPEC = importlib.util.spec_from_file_location(
    "plan_pr_build_jobs_under_test",
    PLANNER_PATH,
)
if PLANNER_SPEC is None or PLANNER_SPEC.loader is None:
    raise RuntimeError(f"unable to import {PLANNER_PATH}")
PLANNER = importlib.util.module_from_spec(PLANNER_SPEC)
PLANNER_SPEC.loader.exec_module(PLANNER)


def base_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "event_name": "pull_request",
        "all_rust": False,
        "backend_changed": False,
        "benchmarks": False,
        "docs_only": False,
        "inference_artifact_required": False,
        "linux_test_groups_nonempty": False,
        "live_agent_enabled": False,
        "runner_contract_required": False,
        "sdk_smoke_required": False,
        "test_batches_nonempty": False,
        "windows_cpu_required": False,
        "windows_gpu_required": False,
        "affected_crates": [],
    }
    payload.update(overrides)
    return payload


class PrBuildJobPlanTests(unittest.TestCase):
    def test_docs_only_pull_request_has_no_heavy_jobs(self) -> None:
        self.assertEqual(
            PLANNER.required_jobs(base_payload(docs_only=True)),
            [],
        )

    def test_runner_contract_is_independent_of_docs_routing(self) -> None:
        self.assertEqual(
            PLANNER.required_jobs(
                base_payload(
                    docs_only=True,
                    runner_contract_required=True,
                )
            ),
            ["public_runner_image_contract"],
        )

    def test_docs_only_dispatch_does_not_require_public_mesh_without_product(self) -> None:
        jobs = set(
            PLANNER.required_jobs(
                base_payload(
                    docs_only=True,
                    event_name="workflow_dispatch",
                )
            )
        )
        self.assertIn("public_runner_image_contract", jobs)
        self.assertNotIn("linux_cpu_artifact", jobs)
        self.assertNotIn("linux_public_mesh_admission", jobs)

    def test_inference_routes_complete_producer_consumer_chains(self) -> None:
        jobs = set(
            PLANNER.required_jobs(
                base_payload(
                    affected_crates=["mesh-llm"],
                    inference_artifact_required=True,
                    linux_test_groups_nonempty=True,
                    test_batches_nonempty=True,
                )
            )
        )
        expected = {
            "linux_host_input",
            "linux_cpu_runtime_input",
            "linux_cpu_artifact",
            "linux_static_abi_input",
            "rust_crate_tests",
            "linux_test_groups",
            "hf_download_smoke",
            "inference_smoke_tests",
            "two_node_client_serving_smoke",
            "two_node_split_smoke",
            "macos_host_input",
            "macos_metal_runtime_input",
            "macos_cpu_artifact",
            "macos_unit_tests",
        }
        self.assertEqual(jobs, expected)

    def test_benchmark_routes_native_backends_without_cpu_product(self) -> None:
        jobs = set(PLANNER.required_jobs(base_payload(benchmarks=True)))
        self.assertIn("linux_host_input", jobs)
        self.assertNotIn("linux_cpu_runtime_input", jobs)
        self.assertNotIn("linux_cpu_artifact", jobs)
        for backend in ("cuda", "rocm", "vulkan"):
            self.assertIn(f"linux_{backend}_runtime_input", jobs)
            self.assertIn(f"linux_{backend}_product", jobs)
        self.assertTrue(
            {
                "macos_host_input",
                "macos_metal_runtime_input",
                "macos_cpu_artifact",
            }.issubset(jobs)
        )

    def test_sdk_route_reuses_static_abi_and_platform_products(self) -> None:
        jobs = set(
            PLANNER.required_jobs(
                base_payload(
                    inference_artifact_required=True,
                    sdk_smoke_required=True,
                )
            )
        )
        self.assertTrue(
            {
                "linux_host_input",
                "linux_cpu_runtime_input",
                "linux_cpu_artifact",
                "linux_static_abi_input",
                "rust_sdk_smoke",
                "kotlin_sdk_input",
                "kotlin_sdk_smoke",
                "swift_sdk_input",
                "macos_host_input",
                "macos_metal_runtime_input",
                "macos_cpu_artifact",
                "swift_sdk_smoke",
            }.issubset(jobs)
        )

    def test_live_agent_route_requires_both_endpoint_and_relevant_code(self) -> None:
        without_endpoint = set(
            PLANNER.required_jobs(
                base_payload(
                    affected_crates=["mesh-llm-client"],
                    inference_artifact_required=True,
                )
            )
        )
        with_endpoint = set(
            PLANNER.required_jobs(
                base_payload(
                    affected_crates=["mesh-llm-client"],
                    inference_artifact_required=True,
                    live_agent_enabled=True,
                )
            )
        )
        self.assertNotIn("agent_live_smokes", without_endpoint)
        self.assertIn("agent_live_smokes", with_endpoint)
        self.assertIn("two_node_client_serving_smoke", without_endpoint)

    def test_log_store_routes_only_the_windows_storage_privacy_checks(self) -> None:
        jobs = set(
            PLANNER.required_jobs(
                base_payload(affected_crates=["mesh-llm-log-store"])
            )
        )
        self.assertIn("windows_checks", jobs)
        self.assertNotIn("windows_host_input", jobs)
        self.assertNotIn("windows_cpu_runtime_input", jobs)
        self.assertNotIn("windows_gpu_runtime_inputs", jobs)
        self.assertNotIn("windows_cpu_product", jobs)
        self.assertNotIn("windows_gpu_products", jobs)

    def test_manual_dispatch_preserves_public_mesh_and_platform_canaries(self) -> None:
        jobs = set(
            PLANNER.required_jobs(
                base_payload(
                    event_name="workflow_dispatch",
                    sdk_smoke_required=True,
                )
            )
        )
        self.assertIn("linux_public_mesh_admission", jobs)
        self.assertIn("public_runner_image_contract", jobs)
        self.assertIn("linux_cpu_artifact", jobs)
        self.assertIn("linux_cuda_product", jobs)
        self.assertIn("macos_cpu_artifact", jobs)
        self.assertIn("windows_cpu_product", jobs)
        self.assertIn("windows_gpu_products", jobs)
        self.assertNotIn("agent_live_smokes", jobs)

    def test_invalid_or_incomplete_input_fails_closed(self) -> None:
        missing = base_payload()
        del missing["docs_only"]
        with self.assertRaises(PLANNER.PlanError):
            PLANNER.required_jobs(missing)
        with self.assertRaises(PLANNER.PlanError):
            PLANNER.required_jobs(base_payload(docs_only="false"))
        with self.assertRaises(PLANNER.PlanError):
            PLANNER.required_jobs(base_payload(event_name="push"))

    def test_cli_emits_compact_json_and_rejects_invalid_input(self) -> None:
        payload = base_payload(runner_contract_required=True)
        result = subprocess.run(
            [sys.executable, str(PLANNER_PATH)],
            cwd=ROOT,
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            result.stdout,
            '["public_runner_image_contract"]\n',
        )

        invalid = subprocess.run(
            [sys.executable, str(PLANNER_PATH)],
            cwd=ROOT,
            input="{}",
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(invalid.returncode, 2)
        self.assertIn("ERROR: unable to plan PR Builds jobs", invalid.stderr)


if __name__ == "__main__":
    unittest.main()
