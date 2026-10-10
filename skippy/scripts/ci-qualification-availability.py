#!/usr/bin/env python3
"""Materialize the closed Skippy hardware policy for one planned core row.

Run this from a protected workflow checkout. The product source and plan are
identities, not authorities for runner selection.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile


SPEC = importlib.util.spec_from_file_location(
    "validate_ci_qualification", Path(__file__).with_name("validate-ci-qualification.py")
)
assert SPEC is not None and SPEC.loader is not None
contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contract)

# runner. The three rows the protected hosted graph always provides.
ALWAYS_AVAILABLE = {
    "linux-cpu": "linux-hosted",
    "macos-metal": "macos-hosted",
    "windows-cpu": "windows-hosted",
}
# accelerator rows. Each is available only when the protected enablement
# variable is exactly true, so an absent GPU runner is recorded as
# hardware-unavailable rather than silently downgraded or faked on CPU.
CONDITIONAL = {
    "linux-cuda": ("gpu-nvidia", "MESH_CUDA_INFERENCE_RUNNER_ENABLED"),
    "linux-rocm": ("gpu-amd", "MESH_ROCM_INFERENCE_RUNNER_ENABLED"),
    "linux-vulkan": ("gpu-nvidia", "MESH_VULKAN_INFERENCE_RUNNER_ENABLED"),
}
UNAVAILABLE = {
    "windows-cuda": "no approved ephemeral Windows CUDA inference runner",
    "windows-rocm": "no approved ephemeral Windows ROCm inference runner",
    "windows-vulkan": "no approved ephemeral Windows Vulkan inference runner",
}
ENABLEMENT_VARIABLES = {
    "MESH_CUDA_INFERENCE_RUNNER_ENABLED",
    "MESH_ROCM_INFERENCE_RUNNER_ENABLED",
    "MESH_VULKAN_INFERENCE_RUNNER_ENABLED",
}


def availability(source_sha: str, plan_digest: str, row_id: str, *,
                 enablement: dict[str, str]) -> dict:
    contract.require(contract.GIT_SHA.fullmatch(source_sha) is not None, "invalid source SHA")
    contract.require(contract.SHA256.fullmatch(plan_digest) is not None, "invalid plan digest")
    contract.catalog_row(row_id)
    contract.require(set(enablement) == ENABLEMENT_VARIABLES,
                     "enablement input must name exactly the protected variables")
    result = {
        "schema_version": 1,
        "source_sha": source_sha,
        "plan_digest": plan_digest,
        "row_id": row_id,
        "policy_source": "protected-ci",
    }
    if row_id in ALWAYS_AVAILABLE:
        result.update(state="available", runner=ALWAYS_AVAILABLE[row_id])
    elif row_id in CONDITIONAL:
        runner, variable = CONDITIONAL[row_id]
        if enablement[variable] == "true":
            result.update(state="available", runner=runner)
        else:
            result.update(state="hardware-unavailable",
                          reason=f"{variable} is not exactly true")
    else:
        result.update(state="hardware-unavailable", reason=UNAVAILABLE[row_id])
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--plan-digest", required=True)
    parser.add_argument("--row-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = availability(
            args.source_sha, args.plan_digest, args.row_id,
            enablement={name: os.environ.get(name, "") for name in sorted(ENABLEMENT_VARIABLES)},
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=args.output.parent, delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(result, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, args.output)
    except (OSError, ValueError, KeyError) as error:
        print(f"Skippy availability rejected: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
