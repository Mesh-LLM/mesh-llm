#!/usr/bin/env python3
"""Enforce the narrow manifest edits allowed to the llama canary agent."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
FAMILY_MANIFEST = Path("ci/llama-canary/family-certified.json")
PARITY_MANIFEST = Path("docs/skippy/llama-parity-candidates.json")
RUNNABLE_NEW_STATUSES = {"candidate", "candidate_stateful", "candidate_multimodal"}
NONRUNNABLE_NEW_STATUSES = {
    "implementation_base",
    "needs_boundary_registration",
    "needs_candidate",
    "needs_runtime_slice_support",
    "no_public_gguf_candidate",
    "non_causal_aux",
    "package_or_remote_only",
}
# New rows classify source coverage only.  Artifact selectors, source
# revisions, integrity records, and execution settings belong to trusted
# manifests and cannot be introduced by the repair agent.
CLASSIFICATION_ROW_KEYS = frozenset(
    {"llama_model", "family", "status", "notes", "unsupported_reason"}
)


class PolicyError(ValueError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PolicyError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise PolicyError(f"{path} must contain a JSON object")
    return value


def load_base_json(base_ref: str, path: Path) -> dict[str, Any]:
    result = subprocess.run(
        ["git", "show", f"{base_ref}:{path.as_posix()}"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise PolicyError(
            f"cannot read {path} from candidate base {base_ref}: {result.stderr.strip()}"
        )
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise PolicyError(f"base {path} is invalid JSON: {error}") from error
    if not isinstance(value, dict):
        raise PolicyError(f"base {path} must contain a JSON object")
    return value


def parity_helpers() -> Any:
    path = ROOT / "scripts/skippy-llama-parity.py"
    spec = importlib.util.spec_from_file_location("skippy_llama_parity_policy", path)
    if spec is None or spec.loader is None:
        raise PolicyError(f"cannot load parity helper from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_family_manifest(before: dict[str, Any], after: dict[str, Any]) -> None:
    before_header = {key: value for key, value in before.items() if key != "models"}
    after_header = {key: value for key, value in after.items() if key != "models"}
    if after_header != before_header:
        raise PolicyError("family certification policy outside model rows changed")

    before_models = before.get("models")
    after_models = after.get("models")
    if not isinstance(before_models, list) or not isinstance(after_models, list):
        raise PolicyError("family certification manifests must contain model arrays")
    if len(after_models) != len(before_models):
        raise PolicyError("family certification roster changed")

    for index, (before_model, after_model) in enumerate(zip(before_models, after_models)):
        if not isinstance(before_model, dict) or not isinstance(after_model, dict):
            raise PolicyError(f"family certification row {index} must be an object")
        family = before_model.get("family", f"row-{index}")
        if after_model.get("family") != family:
            raise PolicyError(f"family certification row identity changed at {family}")

        try:
            before_size = before_model["resources"]["estimated_model_bytes"]
            after_size = after_model["resources"]["estimated_model_bytes"]
        except (KeyError, TypeError) as error:
            raise PolicyError(f"family {family} is missing estimated_model_bytes") from error
        if not isinstance(before_size, int) or isinstance(before_size, bool) or before_size <= 0:
            raise PolicyError(f"base estimated_model_bytes is invalid for {family}")
        if not isinstance(after_size, int) or isinstance(after_size, bool) or after_size <= 0:
            raise PolicyError(f"candidate estimated_model_bytes is invalid for {family}")

        normalized = copy.deepcopy(after_model)
        normalized["resources"]["estimated_model_bytes"] = before_size
        if normalized != before_model:
            raise PolicyError(
                f"family {family} changed outside resources.estimated_model_bytes"
            )


def validate_parity_manifest(
    before: dict[str, Any],
    after: dict[str, Any],
    source_models: set[str],
    boundary_registered: set[str],
) -> None:
    before_header = {key: value for key, value in before.items() if key != "candidates"}
    after_header = {key: value for key, value in after.items() if key != "candidates"}
    if after_header != before_header:
        raise PolicyError("parity policy outside candidate rows changed")

    before_rows = before.get("candidates")
    after_rows = after.get("candidates")
    if not isinstance(before_rows, list) or not isinstance(after_rows, list):
        raise PolicyError("parity manifests must contain candidate arrays")
    if after_rows[: len(before_rows)] != before_rows:
        raise PolicyError("existing parity candidate rows changed or were reordered")

    existing_name_values = [
        row.get("llama_model") if isinstance(row, dict) else None
        for row in before_rows
    ]
    if any(not isinstance(name, str) or not name for name in existing_name_values):
        raise PolicyError("base parity manifest has missing llama_model rows")
    # One llama.cpp source may intentionally have several immutable
    # classification rows for distinct Mesh family variants. Treat the source
    # names as a set when determining which newly observed sources still need
    # classification; the prefix equality check above continues to protect
    # every existing row byte-for-byte.
    existing_names = set(existing_name_values)
    expected_new_names = source_models - existing_names
    new_rows = after_rows[len(before_rows) :]
    new_names = []
    for index, row in enumerate(new_rows):
        if not isinstance(row, dict):
            raise PolicyError(f"new parity row {index} must be an object")
        name = row.get("llama_model")
        if not isinstance(name, str) or not name:
            raise PolicyError(f"new parity row {index} must have a non-empty llama_model")
        new_names.append(name)
    if len(new_names) != len(new_rows) or len(set(new_names)) != len(new_names):
        raise PolicyError("new parity rows must have unique llama_model names")
    if set(new_names) != expected_new_names:
        missing = sorted(expected_new_names - set(new_names))
        unexpected = sorted(set(new_names) - expected_new_names)
        raise PolicyError(
            f"new parity rows do not exactly classify missing sources; "
            f"missing={missing}, unexpected={unexpected}"
        )

    allowed_statuses = RUNNABLE_NEW_STATUSES | NONRUNNABLE_NEW_STATUSES
    for row in new_rows:
        name = row["llama_model"]
        unexpected_keys = sorted(set(row) - CLASSIFICATION_ROW_KEYS)
        if unexpected_keys:
            raise PolicyError(
                f"new parity row {name} may contain classification metadata only; "
                f"disallowed fields: {', '.join(unexpected_keys)}"
            )
        if not isinstance(row.get("family"), str) or not row["family"]:
            raise PolicyError(f"new parity row {name} must have a non-empty family")
        status = row.get("status")
        if not isinstance(status, str) or status not in allowed_statuses:
            raise PolicyError(f"new parity row {name} has disallowed status {status!r}")
        for field in ("notes", "unsupported_reason"):
            if field in row and not isinstance(row[field], str):
                raise PolicyError(f"new parity row {name} has non-string {field}")
        if name in boundary_registered:
            if status not in RUNNABLE_NEW_STATUSES:
                raise PolicyError(
                    f"boundary-registered new source {name} must remain a runnable candidate"
                )
            if row.get("unsupported_reason"):
                raise PolicyError(
                    f"runnable new parity row {name} cannot carry unsupported_reason"
                )
        elif status in RUNNABLE_NEW_STATUSES:
            raise PolicyError(
                f"new runnable parity row {name} lacks begin_block/end_block registration"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ref", required=True)
    parser.add_argument("--llama-src", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        family_before = load_base_json(args.base_ref, FAMILY_MANIFEST)
        parity_before = load_base_json(args.base_ref, PARITY_MANIFEST)
        family_after = load_json(ROOT / FAMILY_MANIFEST)
        parity_after = load_json(ROOT / PARITY_MANIFEST)
        helpers = parity_helpers()
        source_models = set(helpers.pinned_llama_models(args.llama_src))
        boundary_registered = set(helpers.boundary_registered_models(args.llama_src))
        validate_family_manifest(family_before, family_after)
        validate_parity_manifest(
            parity_before,
            parity_after,
            source_models,
            boundary_registered,
        )
    except PolicyError as error:
        print(f"llama canary agent manifest policy failed: {error}", file=sys.stderr)
        return 1
    print("llama canary agent manifest policy passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
