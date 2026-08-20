#!/usr/bin/env python3
"""Capture and validate sccache statistics for CI evidence."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ARTIFACT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
EVIDENCE_SCHEMA = "mesh-llm.sccache-stats"
EVIDENCE_SCHEMA_VERSION = 2
REQUIRED_COUNTERS = (
    "compile_requests",
    "requests_executed",
    "compilations",
    "cache_writes",
    "cache_read_errors",
    "cache_write_errors",
)
REQUIRED_COUNT_MAPS = ("cache_hits", "cache_misses", "cache_errors")


class EvidenceError(RuntimeError):
    """Raised when sccache cannot provide trustworthy evidence."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-name", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--github-output", type=Path)
    parser.add_argument(
        "--cache-expectation",
        choices=("cold", "warm", "opportunistic"),
        default="opportunistic",
    )
    parser.add_argument("--minimum-hit-rate", type=float, default=0.0)
    return parser.parse_args()


def assess_cache(
    expectation: str,
    minimum_hit_rate: float,
    counters: dict[str, int],
) -> dict[str, Any]:
    if not 0 <= minimum_hit_rate <= 1:
        raise EvidenceError("minimum hit rate must be between 0 and 1")
    requests = counters["cache_hits"] + counters["cache_misses"]
    rate = counters["cache_hits"] / requests if requests else None
    if expectation == "cold":
        classification, passed = "cold", True
    elif expectation == "opportunistic":
        classification, passed = "opportunistic", True
    elif rate is not None and rate >= minimum_hit_rate:
        classification, passed = "warm-pass", True
    else:
        classification, passed = "warm-failure", False
    return {
        "expectation": expectation,
        "classification": classification,
        "minimum_hit_rate": minimum_hit_rate,
        "hit_rate": rate,
        "cache_requests": requests,
        "passed": passed,
    }


def require_counter(stats: dict[str, Any], name: str) -> int:
    value = stats.get(name)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvidenceError(
            f"sccache JSON field stats.{name} must be a non-negative integer",
        )
    return value


def validate_count_tree(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise EvidenceError(f"sccache JSON field {field} contains a boolean")
    if isinstance(value, int):
        if value < 0:
            raise EvidenceError(
                f"sccache JSON field {field} contains a negative counter",
            )
        return value
    if isinstance(value, dict):
        return sum(
            validate_count_tree(child, f"{field} entry")
            for child in value.values()
        )
    raise EvidenceError(
        f"sccache JSON field {field} must contain only counter maps and integers",
    )


def sanitize_count_map(
    stats: dict[str, Any],
    name: str,
) -> tuple[dict[str, dict[str, int]], int]:
    value = stats.get(name)
    if not isinstance(value, dict):
        raise EvidenceError(f"sccache JSON field stats.{name} must be an object")
    counts = value.get("counts")
    if not isinstance(counts, dict):
        raise EvidenceError(
            f"sccache JSON field stats.{name}.counts must be an object",
        )
    total = validate_count_tree(counts, f"stats.{name}.counts")
    return {"counts": {"total": total}}, total


def sanitize_stats(payload: Any) -> tuple[dict[str, Any], dict[str, int]]:
    if not isinstance(payload, dict):
        raise EvidenceError("sccache JSON root must be an object")
    stats = payload.get("stats")
    if not isinstance(stats, dict):
        raise EvidenceError("sccache JSON field stats must be an object")

    counters = {name: require_counter(stats, name) for name in REQUIRED_COUNTERS}
    sanitized_stats: dict[str, Any] = dict(counters)
    for name in REQUIRED_COUNT_MAPS:
        sanitized_map, total = sanitize_count_map(stats, name)
        sanitized_stats[name] = sanitized_map
        counters[name] = total
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "stats": sanitized_stats,
    }
    return evidence, counters


def run_sccache(arguments: list[str]) -> str:
    result = subprocess.run(
        ["sccache", *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise EvidenceError(
            f"sccache {' '.join(arguments)} failed with "
            f"exit code {result.returncode}",
        )
    return result.stdout


def write_github_outputs(
    destination: Path | None,
    stats_file: Path,
    counters: dict[str, int],
    assessment: dict[str, Any],
) -> None:
    if destination is None:
        return
    with destination.open("a", encoding="utf-8") as output:
        output.write(f"stats_file={stats_file}\n")
        for name in (
            "compile_requests",
            "requests_executed",
            "cache_hits",
            "cache_misses",
            "cache_writes",
            "cache_read_errors",
            "cache_write_errors",
        ):
            output.write(f"{name}={counters[name]}\n")
        output.write(f"hit_rate={assessment['hit_rate'] if assessment['hit_rate'] is not None else ''}\n")
        output.write(f"cache_classification={assessment['classification']}\n")
        output.write(f"cache_passed={str(assessment['passed']).lower()}\n")


def main() -> int:
    arguments = parse_args()
    try:
        if not ARTIFACT_NAME_PATTERN.fullmatch(arguments.artifact_name):
            raise EvidenceError(
                "artifact name must contain only letters, numbers, dots, "
                "underscores, and hyphens",
            )
        if shutil.which("sccache") is None:
            raise EvidenceError("sccache is required to capture build-cache evidence")

        raw_json = run_sccache(
            ["--show-stats", "--stats-format", "json"],
        )
        try:
            payload = json.loads(raw_json)
        except json.JSONDecodeError as error:
            raise EvidenceError(f"sccache returned invalid JSON: {error}") from error
        evidence, counters = sanitize_stats(payload)
        assessment = assess_cache(
            arguments.cache_expectation,
            arguments.minimum_hit_rate,
            counters,
        )
        evidence["assessment"] = assessment

        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        stats_file = arguments.output.resolve()
        write_github_outputs(arguments.github_output, stats_file, counters, assessment)

        print(
            "sccache evidence: "
            f"requests={counters['compile_requests']} "
            f"executed={counters['requests_executed']} "
            f"hits={counters['cache_hits']} "
            f"misses={counters['cache_misses']} "
            f"writes={counters['cache_writes']}",
        )
        print(
            "sccache assessment: "
            f"expectation={assessment['expectation']} "
            f"classification={assessment['classification']} "
            f"hit_rate={assessment['hit_rate']}",
        )
        if counters["compile_requests"] == 0:
            print(
                "::warning title=sccache reported zero compile requests::"
                "Check RUSTC_WRAPPER wiring unless this job fully reused its "
                "restored target cache.",
            )
    except EvidenceError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
