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
    return parser.parse_args()


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
            validate_count_tree(child, f"{field}.{name}")
            for name, child in value.items()
        )
    raise EvidenceError(
        f"sccache JSON field {field} must contain only counter maps and integers",
    )


def require_count_map(stats: dict[str, Any], name: str) -> int:
    value = stats.get(name)
    if not isinstance(value, dict):
        raise EvidenceError(f"sccache JSON field stats.{name} must be an object")
    counts = value.get("counts")
    if not isinstance(counts, dict):
        raise EvidenceError(
            f"sccache JSON field stats.{name}.counts must be an object",
        )
    validate_count_tree(value, f"stats.{name}")
    return validate_count_tree(counts, f"stats.{name}.counts")


def validate_stats(payload: Any) -> dict[str, int]:
    if not isinstance(payload, dict):
        raise EvidenceError("sccache JSON root must be an object")
    stats = payload.get("stats")
    if not isinstance(stats, dict):
        raise EvidenceError("sccache JSON field stats must be an object")

    counters = {name: require_counter(stats, name) for name in REQUIRED_COUNTERS}
    counters.update(
        {name: require_count_map(stats, name) for name in REQUIRED_COUNT_MAPS},
    )
    return counters


def run_sccache(arguments: list[str], *, capture: bool = False) -> str:
    result = subprocess.run(
        ["sccache", *arguments],
        check=False,
        capture_output=capture,
        text=True,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() if capture else ""
        suffix = f": {detail}" if detail else ""
        raise EvidenceError(
            f"sccache {' '.join(arguments)} failed with "
            f"exit code {result.returncode}{suffix}",
        )
    return result.stdout if capture else ""


def write_github_outputs(
    destination: Path | None,
    stats_file: Path,
    counters: dict[str, int],
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

        print("::group::Human-readable sccache statistics", flush=True)
        try:
            run_sccache(["--show-stats"])
        finally:
            print("::endgroup::", flush=True)

        raw_json = run_sccache(
            ["--show-stats", "--stats-format", "json"],
            capture=True,
        )
        try:
            payload = json.loads(raw_json)
        except json.JSONDecodeError as error:
            raise EvidenceError(f"sccache returned invalid JSON: {error}") from error
        counters = validate_stats(payload)

        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            raw_json.rstrip("\n") + "\n",
            encoding="utf-8",
        )
        stats_file = arguments.output.resolve()
        write_github_outputs(arguments.github_output, stats_file, counters)

        print(
            "sccache evidence: "
            f"requests={counters['compile_requests']} "
            f"executed={counters['requests_executed']} "
            f"hits={counters['cache_hits']} "
            f"misses={counters['cache_misses']} "
            f"writes={counters['cache_writes']}",
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
