#!/usr/bin/env python3
"""Verify and record the standalone Skippy receipts a Mesh host was built against.

The Mesh host must not be produced from an unqualified Skippy product. This
script consumes the per-row qualification receipts the standalone product slice
produced for the same source and plan, requires the platform's mandatory row to
be qualified, and records every receipt digest so the host product carries its
provenance.
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
    "validate_ci_qualification",
    Path(__file__).resolve().parents[1] / "skippy/scripts/validate-ci-qualification.py",
)
assert SPEC is not None and SPEC.loader is not None
contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(contract)

# Row the protected hosted graph always executes for each platform.
REQUIRED_ROW = {"linux": "linux-cpu", "macos": "macos-metal", "windows": "windows-cpu"}


def verify(receipts_dir: Path, *, source_sha: str, plan_digest: str, platform: str) -> dict:
    contract.require(platform in REQUIRED_ROW, f"unknown platform: {platform}")
    contract.require(contract.GIT_SHA.fullmatch(source_sha) is not None, "invalid source SHA")
    contract.require(contract.SHA256.fullmatch(plan_digest) is not None, "invalid plan digest")
    files = sorted(path for path in receipts_dir.rglob("*.json") if path.is_file())
    contract.require(bool(files), "no standalone qualification receipts were downloaded")
    rows: dict[str, dict] = {}
    records = []
    for path in files:
        receipt = contract.load_json(path)
        contract.require(receipt.get("schema_version") == 1, f"{path.name}: unknown receipt schema")
        contract.require(receipt.get("source_sha") == source_sha,
                         f"{path.name}: receipt belongs to another source revision")
        contract.require(receipt.get("plan_digest") == plan_digest,
                         f"{path.name}: receipt belongs to another CI plan")
        row_id = receipt.get("row_id")
        contract.require(row_id in contract.CORE_ROWS, f"{path.name}: receipt names an unknown core row")
        contract.require(isinstance(row_id, str) and row_id.startswith(f"{platform}-"),
                         f"{path.name}: receipt belongs to another platform")
        contract.require(row_id not in rows, f"{path.name}: duplicate receipt for {row_id}")
        contract.require(receipt.get("status") in {"qualified", "hardware-unavailable"},
                         f"{path.name}: invalid qualification status")
        rows[row_id] = receipt
        records.append({"row_id": row_id, "status": receipt["status"],
                        "receipt_sha256": contract.digest(path)})
    required = REQUIRED_ROW[platform]
    contract.require(required in rows, f"required row {required} produced no receipt")
    contract.require(rows[required].get("status") == "qualified",
                     f"required row {required} is not qualified")
    return {"schema_version": 1, "source_sha": source_sha, "plan_digest": plan_digest,
            "platform": platform, "required_row": required,
            "receipts": sorted(records, key=lambda record: record["row_id"])}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--plan-digest", required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        record = verify(args.receipts, source_sha=args.source_sha,
                        plan_digest=args.plan_digest, platform=args.platform)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=args.output.parent, delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(record, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, args.output)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"standalone qualification receipts rejected: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
