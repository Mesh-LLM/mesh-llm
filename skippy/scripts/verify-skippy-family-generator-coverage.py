#!/usr/bin/env python3
"""Require every primary canary family to own a generated decoder transform."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

NON_CHAT_CLASSES = {
    "embedding", "rerank", "encoder_decoder", "ocr",
    "speech_synthesis", "speech_recognition",
}


def verify(manifest_path: Path, family_map_path: Path, report_path: Path) -> list[str]:
    """Require decoder transforms only for explicitly classified causal split targets."""
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
        model_class = model.get("class")
        if isinstance(model_class, str) and model_class in NON_CHAT_CLASSES:
            if model.get("profile") not in ("workload-smoke", "workload-oracle"):
                errors.append(f"{family}: non-chat class requires a workload profile")
            continue
        if model_class != "causal_generation":
            errors.append(f"{family}: missing or unknown workload class")
            continue
        if model.get("profile") in ("workload-smoke", "workload-oracle"):
            errors.append(f"{family}: causal split target cannot use a workload profile")
            continue
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
    """Check causal decoder coverage without promoting full-model workload evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--family-map", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    errors = verify(args.manifest, args.family_map, args.report)
    if errors:
        raise SystemExit("generator coverage failed:\n  " + "\n  ".join(errors))
    print("every causal canary split target owns a transformed partitioned decoder")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
