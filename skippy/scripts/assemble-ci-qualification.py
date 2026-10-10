#!/usr/bin/env python3
"""Assemble and validate a Skippy qualification receipt from executed suites."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile


CONTRACT_PATH = Path(__file__).with_name("validate-ci-qualification.py")
SPEC = importlib.util.spec_from_file_location("validate_ci_qualification", CONTRACT_PATH)
assert SPEC is not None and SPEC.loader is not None
contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contract)


def assemble(
    *, product_dir: Path, availability_path: Path, evidence_dir: Path,
    hardware_path: Path | None, source_sha: str, plan_digest: str, row_id: str,
) -> dict:
    product_manifest = product_dir / "product-manifest.json"
    product = contract.load_json(product_manifest)
    availability = contract.load_json(availability_path)
    state = availability.get("state")
    if state not in {"available", "hardware-unavailable"}:
        raise ValueError("unknown hardware availability state")
    if (state == "available") != (hardware_path is not None):
        raise ValueError("hardware evidence is required exactly when execution is available")

    suites = {}
    for name in sorted(contract.REQUIRED_SUITES):
        if state == "hardware-unavailable" and name != "packaging-runtime":
            suites[name] = {"status": "not-executed"}
            continue
        evidence_path = evidence_dir / f"{name}.json"
        evidence = contract.load_json(evidence_path)
        if evidence.get("status") != "passed":
            raise ValueError(f"{name} evidence did not pass")
        suites[name] = {
            "status": "passed",
            "cases": evidence.get("executed_cases"),
            "models": evidence.get("models"),
            "evidence_file": evidence_path.name,
            "evidence_sha256": contract.digest(evidence_path),
        }

    receipt = {
        "schema_version": 1,
        "source_sha": source_sha,
        "plan_digest": plan_digest,
        "row_id": row_id,
        "product_manifest_sha256": contract.digest(product_manifest),
        "availability_sha256": contract.digest(availability_path),
        "status": "qualified" if state == "available" else "hardware-unavailable",
        "hardware": contract.load_json(hardware_path) if hardware_path else None,
        "suites": suites,
    }
    contract.validate_receipt(
        receipt, product, product_manifest, product_dir, availability,
        availability_path, evidence_dir, source_sha=source_sha,
        plan_digest=plan_digest, row_id=row_id,
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product-dir", type=Path, required=True)
    parser.add_argument("--availability", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--hardware", type=Path)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--plan-digest", required=True)
    parser.add_argument("--row-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        receipt = assemble(
            product_dir=args.product_dir, availability_path=args.availability,
            evidence_dir=args.evidence_dir, hardware_path=args.hardware,
            source_sha=args.source_sha, plan_digest=args.plan_digest, row_id=args.row_id,
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=args.output.parent, delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(receipt, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, args.output)
        print(contract.digest(args.output))
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"Skippy qualification assembly rejected: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
