#!/usr/bin/env python3
"""Summarize paired upstream and Depot Registry cold-pull observations."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


SOURCES = ("upstream", "depot")


def load_observations(root: Path) -> list[dict[str, object]]:
    """Load observations from source.

    Returns:
        Loaded data.
    """
    observations: list[dict[str, object]] = []
    for path in sorted(root.rglob("*.json")):
        with path.open(encoding="utf-8") as handle:
            observation = json.load(handle)
        if observation.get("source") not in SOURCES:
            raise ValueError(f"{path}: source must be upstream or depot")
        if not isinstance(observation.get("sample"), int):
            raise ValueError(f"{path}: sample must be an integer")
        if not isinstance(observation.get("elapsed_ms"), int):
            raise ValueError(f"{path}: elapsed_ms must be an integer")
        digest = observation.get("digest")
        if not isinstance(digest, str) or not digest.startswith("sha256:"):
            raise ValueError(f"{path}: digest must be a sha256 digest")
        observations.append(observation)
    return observations


def summarize(
    observations: list[dict[str, object]], minimum_samples: int
) -> dict[str, object]:
    """Execute summarize operation."""
    by_source = {
        source: [item for item in observations if item["source"] == source]
        for source in SOURCES
    }
    for source, items in by_source.items():
        samples = {item["sample"] for item in items}
        if len(items) < minimum_samples or len(samples) != len(items):
            raise ValueError(
                f"{source}: need at least {minimum_samples} unique samples"
            )

    digests = {item["digest"] for item in observations}
    if len(digests) != 1:
        raise ValueError("upstream and Depot observations used different digests")

    upstream_ms = statistics.median(
        int(item["elapsed_ms"]) for item in by_source["upstream"]
    )
    depot_ms = statistics.median(
        int(item["elapsed_ms"]) for item in by_source["depot"]
    )
    improvement_ms = upstream_ms - depot_ms
    improvement_percent = (improvement_ms / upstream_ms) * 100 if upstream_ms else 0
    eligible = improvement_ms >= 10_000 and improvement_percent >= 20
    return {
        "digest": digests.pop(),
        "upstream_median_ms": upstream_ms,
        "depot_median_ms": depot_ms,
        "improvement_ms": improvement_ms,
        "improvement_percent": improvement_percent,
        "eligible": eligible,
        "samples_per_source": {
            source: len(items) for source, items in by_source.items()
        },
    }


def markdown(summary: dict[str, object]) -> str:
    """Execute markdown operation."""
    return "\n".join(
        (
            "## Depot Registry pull-through result",
            "",
            "| Signal | Value |",
            "| --- | ---: |",
            f"| Upstream median | {int(summary['upstream_median_ms']) / 1000:.3f}s |",
            f"| Depot median | {int(summary['depot_median_ms']) / 1000:.3f}s |",
            f"| Improvement | {int(summary['improvement_ms']) / 1000:.3f}s "
            f"({float(summary['improvement_percent']):.1f}%) |",
            f"| Adoption gate | {'pass' if summary['eligible'] else 'fail'} |",
            "",
            f"Digest: `{summary['digest']}`",
            "",
            "The gate requires at least five fresh-runner samples per source, "
            "identical manifest digests, at least 20% improvement, and at least "
            "10 seconds saved at the median.",
        )
    )


def parse_args() -> argparse.Namespace:
    """Parse and validate args.

    Returns:
        Parsed result.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("observations", type=Path)
    parser.add_argument("--minimum-samples", type=int, default=5)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--enforce", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Execute main program logic.

    Returns:
        Exit code.
    """
    args = parse_args()
    result = summarize(load_observations(args.observations), args.minimum_samples)
    report = markdown(result)
    if args.json_out:
        args.json_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.write_text(report + "\n", encoding="utf-8")
    else:
        print(report)
    return 1 if args.enforce and not result["eligible"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
