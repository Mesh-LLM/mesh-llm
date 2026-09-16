#!/usr/bin/env python3
"""Require every primary canary family to own a generated decoder transform."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def verify(manifest_path: Path, family_map_path: Path, report_path: Path) -> list[str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    family_map = json.loads(family_map_path.read_text(encoding="utf-8"))["families"]
    report = json.loads(report_path.read_text(encoding="utf-8"))
    transformed = {
        builder["file"]
        for builder in report.get("builders", [])
        if builder.get("verdict") == "transformable"
        and builder.get("proof", {}).get("execution_scope") == "partitioned_decoder"
    }
    errors: list[str] = []
    for model in manifest.get("models", []):
        family = model.get("family", "")
        sources = family_map.get(family)
        if not sources:
            errors.append(f"{family}: no generated family source mapping")
            continue
        if not any(
            any(path == source or path.endswith(f"/{source}") for path in transformed)
            for source in sources
        ):
            errors.append(f"{family}: no mapped partitioned decoder was transformed")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--family-map", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    errors = verify(args.manifest, args.family_map, args.report)
    if errors:
        raise SystemExit("generator coverage failed:\n  " + "\n  ".join(errors))
    print("every primary canary family owns a transformed partitioned decoder")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
