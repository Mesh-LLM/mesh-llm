#!/usr/bin/env python3
"""Plan the conditionally required top-level jobs in PR Builds."""

from __future__ import annotations

from collections import Counter
import json
import sys
from typing import Any


BOOLEAN_FIELDS = frozenset(
    {
        "all_rust",
        "backend_changed",
        "benchmarks",
        "docs_only",
        "inference_artifact_required",
        "linux_test_groups_nonempty",
        "live_agent_enabled",
        "runner_contract_required",
        "sdk_smoke_required",
        "test_batches_nonempty",
        "windows_cpu_required",
        "windows_gpu_required",
    }
)

# Keep this in workflow order. Every conditional top-level job must appear
# exactly once; the contract tests compare this table with pr_builds.yml.
JOB_ROUTES: tuple[tuple[str, str], ...] = (
    ("public_runner_image_contract", "runner_contract"),
    ("linux_host_input", "linux_host"),
    ("linux_cpu_runtime_input", "linux_cpu"),
    ("linux_cpu_artifact", "linux_cpu"),
    ("linux_cuda_runtime_input", "linux_backend"),
    ("linux_cuda_product", "linux_backend"),
    ("linux_rocm_runtime_input", "linux_backend"),
    ("linux_rocm_product", "linux_backend"),
    ("linux_vulkan_runtime_input", "linux_backend"),
    ("linux_vulkan_product", "linux_backend"),
    ("linux_static_abi_input", "static_abi"),
    ("rust_crate_tests", "rust_crate_tests"),
    ("linux_test_groups", "linux_test_groups"),
    ("linux_public_mesh_admission", "public_mesh_admission"),
    ("hf_download_smoke", "hf_download"),
    ("inference_smoke_tests", "inference_smoke"),
    ("agent_live_smokes", "agent_live"),
    ("two_node_client_serving_smoke", "two_node_client"),
    ("two_node_split_smoke", "two_node_split"),
    ("rust_sdk_smoke", "linux_sdk_smoke"),
    ("kotlin_sdk_input", "kotlin_sdk_input"),
    ("kotlin_sdk_smoke", "linux_sdk_smoke"),
    ("swift_sdk_input", "swift_sdk"),
    ("macos_host_input", "macos_product"),
    ("macos_metal_runtime_input", "macos_product"),
    ("macos_cpu_artifact", "macos_product"),
    ("swift_sdk_smoke", "swift_sdk"),
    ("macos_unit_tests", "macos_unit_tests"),
    ("windows_checks", "windows_checks"),
    ("windows_host_input", "windows_host"),
    ("windows_cpu_runtime_input", "windows_cpu"),
    ("windows_gpu_runtime_inputs", "windows_gpu"),
    ("windows_cpu_product", "windows_cpu"),
    ("windows_gpu_products", "windows_gpu"),
)


class PlanError(ValueError):
    """Raised when the planner input or static route table is invalid."""


def _validate_payload(payload: object) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise PlanError("planner input must be a JSON object")

    expected_fields = BOOLEAN_FIELDS | {"affected_crates", "event_name"}
    actual_fields = set(payload)
    missing = sorted(expected_fields - actual_fields)
    unknown = sorted(actual_fields - expected_fields)
    if missing:
        raise PlanError(f"planner input is missing fields: {', '.join(missing)}")
    if unknown:
        raise PlanError(f"planner input has unknown fields: {', '.join(unknown)}")

    event_name = payload["event_name"]
    if event_name not in {"pull_request", "workflow_dispatch"}:
        raise PlanError(f"unsupported PR Builds event: {event_name!r}")

    for field in sorted(BOOLEAN_FIELDS):
        if type(payload[field]) is not bool:
            raise PlanError(f"{field} must be a JSON boolean")

    affected_crates = payload["affected_crates"]
    if not isinstance(affected_crates, list) or not all(
        isinstance(crate, str) and crate for crate in affected_crates
    ):
        raise PlanError("affected_crates must be an array of non-empty strings")

    return payload


def _validate_job_routes() -> None:
    job_ids = [job_id for job_id, _route in JOB_ROUTES]
    duplicates = sorted(
        job_id for job_id, count in Counter(job_ids).items() if count != 1
    )
    if duplicates:
        raise PlanError(f"jobs mapped more than once: {', '.join(duplicates)}")


def route_requirements(raw_payload: object) -> dict[str, bool]:
    payload = _validate_payload(raw_payload)
    affected = set(payload["affected_crates"])
    dispatch = payload["event_name"] == "workflow_dispatch"
    eligible = not payload["docs_only"]
    linux_inference = dispatch or payload["inference_artifact_required"]
    macos_inference = linux_inference or payload["benchmarks"]
    linux_backend = (
        dispatch
        or payload["backend_changed"]
        or payload["benchmarks"]
    )

    client_runtime_changed = bool(
        affected & {"mesh-llm", "mesh-llm-client", "openai-frontend"}
    )
    inference_runtime_changed = bool(
        affected
        & {
            "mesh-llm",
            "model-artifact",
            "openai-frontend",
            "skippy-runtime",
            "skippy-server",
        }
    )
    split_runtime_changed = bool(
        affected
        & {
            "mesh-llm",
            "model-artifact",
            "skippy-runtime",
            "skippy-server",
        }
    )

    broad_runtime = dispatch or payload["all_rust"]
    sdk_required = payload["sdk_smoke_required"]
    return {
        "runner_contract": dispatch or payload["runner_contract_required"],
        "linux_host": eligible and (linux_inference or payload["benchmarks"]),
        "linux_cpu": eligible and linux_inference,
        "linux_backend": eligible and linux_backend,
        "static_abi": eligible
        and (
            payload["test_batches_nonempty"]
            or payload["linux_test_groups_nonempty"]
            or sdk_required
        ),
        "rust_crate_tests": eligible and payload["test_batches_nonempty"],
        "linux_test_groups": eligible and payload["linux_test_groups_nonempty"],
        # Preserve the existing operator-only public-mesh path. Its product
        # dependency still has to succeed before GitHub starts the job.
        "public_mesh_admission": eligible and dispatch,
        "hf_download": eligible
        and (
            broad_runtime
            or bool(affected & {"mesh-llm", "model-artifact"})
        ),
        "inference_smoke": eligible
        and linux_inference
        and (broad_runtime or inference_runtime_changed),
        "agent_live": eligible
        and linux_inference
        and payload["live_agent_enabled"]
        and (broad_runtime or client_runtime_changed),
        "two_node_client": eligible
        and linux_inference
        and (broad_runtime or client_runtime_changed),
        "two_node_split": eligible
        and linux_inference
        and (broad_runtime or split_runtime_changed),
        "linux_sdk_smoke": eligible and linux_inference and sdk_required,
        "kotlin_sdk_input": eligible and sdk_required,
        "swift_sdk": eligible and macos_inference and sdk_required,
        "macos_product": eligible and macos_inference,
        "macos_unit_tests": eligible
        and (
            payload["all_rust"]
            or bool(
                affected
                & {"mesh-llm", "mesh-llm-host-runtime", "model-artifact"}
            )
        ),
        "windows_checks": eligible
        and (
            dispatch
            or payload["all_rust"]
            or payload["windows_cpu_required"]
            or payload["windows_gpu_required"]
        ),
        "windows_host": eligible
        and (
            dispatch
            or payload["windows_cpu_required"]
            or payload["windows_gpu_required"]
        ),
        "windows_cpu": eligible
        and (dispatch or payload["windows_cpu_required"]),
        "windows_gpu": eligible
        and (dispatch or payload["windows_gpu_required"]),
    }


def required_jobs(raw_payload: object) -> list[str]:
    _validate_job_routes()
    requirements = route_requirements(raw_payload)
    declared_routes = {route for _job_id, route in JOB_ROUTES}
    unknown_routes = sorted(declared_routes - requirements.keys())
    unused_routes = sorted(requirements.keys() - declared_routes)
    if unknown_routes:
        raise PlanError(f"jobs use undefined routes: {', '.join(unknown_routes)}")
    if unused_routes:
        raise PlanError(f"routes have no jobs: {', '.join(unused_routes)}")
    return [
        job_id
        for job_id, route in JOB_ROUTES
        if requirements[route]
    ]


def main() -> int:
    try:
        payload = json.load(sys.stdin)
        plan = required_jobs(payload)
    except (json.JSONDecodeError, PlanError) as error:
        print(f"ERROR: unable to plan PR Builds jobs: {error}", file=sys.stderr)
        return 2

    print(json.dumps(plan, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
