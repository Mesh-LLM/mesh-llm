#!/usr/bin/env python3
"""Validate and record the Cargo packages actually completed by one CI batch."""

import argparse
import json
from pathlib import Path


def validate(selected: list[str], executed: list[str]) -> None:
    for label, packages in (("selected", selected), ("executed", executed)):
        if not isinstance(packages, list) or any(not isinstance(name, str) or not name for name in packages):
            raise ValueError(f"{label} must contain nonempty package names")
        if len(packages) != len(set(packages)):
            raise ValueError(f"{label} contains duplicate packages")
    if set(selected) != set(executed):
        raise ValueError(
            f"Cargo package execution mismatch: missing={sorted(set(selected) - set(executed))}, "
            f"unexpected={sorted(set(executed) - set(selected))}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("test", "clippy"), required=True)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--selected", required=True)
    parser.add_argument("--executed", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    selected = json.loads(args.selected)
    executed = json.loads(args.executed)
    validate(selected, executed)
    receipt = {
        "schema": "mesh-ci-cargo-census-v1",
        "kind": args.kind,
        "batch_id": args.batch_id,
        "source_sha": args.source_sha,
        "selected": sorted(selected),
        "executed": sorted(executed),
    }
    args.out.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{args.kind} batch {args.batch_id}: {len(executed)} packages executed")


if __name__ == "__main__":
    main()
