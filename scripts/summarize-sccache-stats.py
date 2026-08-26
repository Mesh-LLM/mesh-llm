#!/usr/bin/env python3
"""Aggregate downloaded sccache JSON evidence without network access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


class SummaryError(RuntimeError):
    """Raised when downloaded evidence cannot be summarized safely."""


def hit_rate(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "minimum hit rate must be a number between 0 and 1",
        ) from error
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError(
            "minimum hit rate must be a number between 0 and 1",
        )
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate cache hits and misses from downloaded "
            "sccache-stats*.json evidence."
        ),
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="Evidence JSON files or directories to scan recursively.",
    )
    parser.add_argument(
        "--minimum-hit-rate",
        type=hit_rate,
        help="Fail unless aggregate hits / (hits + misses) meets this ratio.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format (default: text).",
    )
    return parser.parse_args()


def discover_evidence(paths: list[Path]) -> list[Path]:
    evidence: set[Path] = set()
    for path in paths:
        if path.is_file():
            evidence.add(path.resolve())
        elif path.is_dir():
            evidence.update(
                candidate.resolve()
                for candidate in path.rglob("sccache-stats*.json")
                if candidate.is_file()
            )
        else:
            raise SummaryError(f"evidence path does not exist: {path}")
    if not evidence:
        raise SummaryError("no sccache-stats*.json evidence files were found")
    return sorted(evidence)


def sum_count_tree(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise SummaryError(f"{field} contains a boolean")
    if isinstance(value, int):
        if value < 0:
            raise SummaryError(f"{field} contains a negative counter")
        return value
    if isinstance(value, dict):
        return sum(
            sum_count_tree(child, f"{field}.{name}")
            for name, child in value.items()
        )
    raise SummaryError(f"{field} must contain only counter maps and integers")


def read_count(path: Path, payload: Any, name: str) -> int:
    if not isinstance(payload, dict):
        raise SummaryError(f"{path}: JSON root must be an object")
    stats = payload.get("stats")
    if not isinstance(stats, dict):
        raise SummaryError(f"{path}: stats must be an object")
    count_map = stats.get(name)
    if not isinstance(count_map, dict):
        raise SummaryError(f"{path}: stats.{name} must be an object")
    counts = count_map.get("counts")
    if not isinstance(counts, dict):
        raise SummaryError(f"{path}: stats.{name}.counts must be an object")
    return sum_count_tree(counts, f"{path}: stats.{name}.counts")


def aggregate(paths: list[Path]) -> tuple[int, int]:
    hits = 0
    misses = 0
    for path in paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise SummaryError(f"{path}: unable to read valid JSON: {error}") from error
        hits += read_count(path, payload, "cache_hits")
        misses += read_count(path, payload, "cache_misses")
    return hits, misses


def render_text(summary: dict[str, Any]) -> str:
    rate = summary["hit_rate"]
    rate_text = "n/a" if rate is None else f"{rate:.2%}"
    lines = [
        "Sccache cache-hit summary",
        f"Evidence files: {summary['file_count']}",
        f"Cache hits: {summary['cache_hits']}",
        f"Cache misses: {summary['cache_misses']}",
        f"Hit rate: {rate_text}",
    ]
    minimum = summary["minimum_hit_rate"]
    if minimum is not None:
        outcome = "PASS" if summary["passed"] else "FAIL"
        lines.append(f"Minimum hit rate: {minimum:.2%} ({outcome})")
    return "\n".join(lines)


def main() -> int:
    arguments = parse_args()
    try:
        evidence = discover_evidence(arguments.paths)
        hits, misses = aggregate(evidence)
    except SummaryError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    requests = hits + misses
    rate = hits / requests if requests else None
    passed = (
        arguments.minimum_hit_rate is None
        or (rate is not None and rate >= arguments.minimum_hit_rate)
    )
    summary = {
        "file_count": len(evidence),
        "cache_hits": hits,
        "cache_misses": misses,
        "cache_requests": requests,
        "hit_rate": rate,
        "minimum_hit_rate": arguments.minimum_hit_rate,
        "passed": passed,
    }
    if arguments.format == "json":
        print(json.dumps(summary, sort_keys=True))
    else:
        print(render_text(summary))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
