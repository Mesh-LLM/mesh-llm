#!/usr/bin/env python3
"""Normalize competitive benchmark artifacts and report cohort-matched drift."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
GPU_FIELDS = (
    "name",
    "uuid",
    "compute_capability",
    "driver_version",
    "pci_bus_id",
    "pstate",
    "temperature_c",
    "sm_clock_mhz",
    "memory_clock_mhz",
)
STABLE_GPU_FIELDS = (
    "name",
    "uuid",
    "compute_capability",
    "driver_version",
    "pci_bus_id",
)
CANDIDATE_MESH_ARMS = {"mesh", "mesh-adaptive"}


def stable_hash(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def gpu_fingerprint(artifact: Path) -> dict[str, str | None]:
    path = artifact / "runner-gpu.csv"
    if not path.is_file():
        raise ValueError(f"missing runner GPU fingerprint: {path}")
    rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))
    if len(rows) != 1 or len(rows[0]) != len(GPU_FIELDS):
        raise ValueError(f"runner GPU fingerprint must contain one {len(GPU_FIELDS)}-field row")
    fingerprint = dict(zip(GPU_FIELDS, (value.strip() for value in rows[0]), strict=True))
    missing = [field for field in STABLE_GPU_FIELDS if not fingerprint[field]]
    if missing:
        raise ValueError(
            "runner GPU fingerprint is missing stable fields: " + ", ".join(missing)
        )
    return fingerprint


def normalize(artifact: Path) -> list[dict[str, Any]]:
    provenance_paths = sorted((artifact / "provenance").glob("*.json"))
    if not provenance_paths:
        raise ValueError("benchmark artifact has no provenance JSON")
    provenance_by_platform = {
        path.stem: load_json(path) for path in provenance_paths
    }
    gpu = gpu_fingerprint(artifact)
    stable_hardware = {
        "name": gpu["name"],
        "compute_capability": gpu["compute_capability"],
        "driver_version": gpu["driver_version"],
        "gpu_identity_sha256": stable_hash(
            {"uuid": gpu["uuid"], "pci_bus_id": gpu["pci_bus_id"]}
        ),
    }
    rows: list[dict[str, Any]] = []
    for result_path in sorted((artifact / "trace").glob("*/*/c-*/*/result.json")):
        arm_dir = result_path.parent
        complete_path = arm_dir / "complete.json"
        if not complete_path.is_file():
            continue
        marker = load_json(complete_path)
        cell = marker.get("cell")
        if not isinstance(cell, dict) or marker.get("cell_sha256") != stable_hash(cell):
            raise ValueError(f"invalid complete marker: {complete_path}")
        result = load_json(result_path)
        platform = str(cell["platform"])
        provenance = provenance_by_platform.get(platform)
        if provenance is None:
            raise ValueError(f"missing provenance for platform {platform}")
        model = str(cell["model"])
        arm = str(cell["arm"])
        backend_binary_sha256 = cell.get("binary_sha256")
        cohort = {
            "schema_version": SCHEMA_VERSION,
            "platform": platform,
            "platform_details": provenance.get("platform_details"),
            "hardware": stable_hardware,
            "model": model,
            "model_sha256": provenance.get("models", {}).get(model),
            "arm": arm,
            # Mesh's binary changes on every candidate by definition; it is
            # the subject of the comparison. External backends are controls,
            # so a runtime upgrade must start a fresh cohort rather than
            # masquerading as a performance regression.
            "control_backend_binary_sha256": (
                None if arm in CANDIDATE_MESH_ARMS else backend_binary_sha256
            ),
            "config_sha256": cell.get("config_sha256"),
            "prompt_manifest_sha256": cell.get("manifest_sha256"),
            "native_runtime_directory_sha256": provenance.get(
                "native_runtime_directory_sha256"
            ),
            "comparison_capacity_policy": cell.get("comparison_capacity_policy"),
            "concurrency": int(cell["concurrency"]),
        }
        failed = int(result["failed_requests"])
        successful = int(result["successful_requests"])
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "created_utc": provenance.get("created_utc"),
                "source_sha": provenance.get("mesh_head"),
                "cohort_key": stable_hash(cohort),
                "cohort": cohort,
                "backend_binary_sha256": backend_binary_sha256,
                "observed_gpu_state": {
                    key: gpu[key]
                    for key in (
                        "pstate",
                        "temperature_c",
                        "sm_clock_mhz",
                        "memory_clock_mhz",
                    )
                },
                "prompt_count": int(result["prompt_count"]),
                "successful_requests": successful,
                "failed_requests": failed,
                "output_tokens": int(result["output_tokens"]),
                "measured_wall_ms": float(result["measured_wall_ms"]),
                "output_tokens_per_second": float(
                    result["output_tokens_per_second"]
                ),
                "ttft_ms_mean": float(result["ttft_ms_mean"]),
                "complete": failed == 0 and successful == int(result["prompt_count"]),
                "artifact_result": result_path.relative_to(artifact).as_posix(),
            }
        )
    if not rows:
        raise ValueError("benchmark artifact has no completed Thoughtworks cells")
    return rows


def read_jsonl(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows = []
    paths = sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]
    for source in paths:
        for number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip():
                value = json.loads(line)
                if (
                    not isinstance(value, dict)
                    or value.get("schema_version") != SCHEMA_VERSION
                ):
                    raise ValueError(f"unsupported history row at {source}:{number}")
                rows.append(value)
    return rows


def median_and_mad(values: list[float]) -> tuple[float, float]:
    median = statistics.median(values)
    return median, statistics.median(abs(value - median) for value in values)


def compare(
    current: list[dict[str, Any]],
    history: list[dict[str, Any]],
    throughput_threshold: float,
    ttft_threshold: float,
) -> list[dict[str, Any]]:
    by_cohort: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in history:
        by_cohort[str(row.get("cohort_key"))].append(row)
    comparisons = []
    for row in current:
        baseline = [
            item
            for item in by_cohort[row["cohort_key"]]
            if item.get("complete") and item.get("source_sha") != row.get("source_sha")
        ]
        classification = "insufficient-baseline"
        throughput_delta = None
        ttft_delta = None
        throughput_median = None
        ttft_median = None
        if not row["complete"]:
            classification = "correctness-failure"
        elif len(baseline) >= 3:
            throughput_values = [float(item["output_tokens_per_second"]) for item in baseline]
            ttft_values = [float(item["ttft_ms_mean"]) for item in baseline]
            throughput_median, throughput_mad = median_and_mad(throughput_values)
            ttft_median, ttft_mad = median_and_mad(ttft_values)
            throughput_delta = row["output_tokens_per_second"] / throughput_median - 1
            ttft_delta = row["ttft_ms_mean"] / ttft_median - 1
            throughput_regression = throughput_delta < -throughput_threshold and (
                throughput_median - row["output_tokens_per_second"]
                > max(3 * throughput_mad, throughput_median * 0.01)
            )
            ttft_regression = ttft_delta > ttft_threshold and (
                row["ttft_ms_mean"] - ttft_median
                > max(3 * ttft_mad, ttft_median * 0.01)
            )
            classification = (
                "performance-regression"
                if throughput_regression or ttft_regression
                else "pass"
            )
        comparisons.append(
            {
                "row": row,
                "baseline_count": len(baseline),
                "throughput_median": throughput_median,
                "throughput_delta": throughput_delta,
                "ttft_median": ttft_median,
                "ttft_delta": ttft_delta,
                "classification": classification,
            }
        )
    return comparisons


def percentage(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:+.1f}%"


def write_report(path: Path, comparisons: list[dict[str, Any]]) -> None:
    lines = [
        "# Performance history regression report",
        "",
        "Comparisons are restricted to exact cohort keys. At least three prior complete runs are required; thresholds are report-only unless `--gate` is used.",
        "",
        "| model | backend | c | baseline runs | throughput Δ | TTFT Δ | result |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for comparison in comparisons:
        row = comparison["row"]
        cohort = row["cohort"]
        lines.append(
            "| {model} | {arm} | {concurrency} | {baseline_count} | {throughput} | {ttft} | {classification} |".format(
                model=cohort["model"],
                arm=cohort["arm"],
                concurrency=cohort["concurrency"],
                baseline_count=comparison["baseline_count"],
                throughput=percentage(comparison["throughput_delta"]),
                ttft=percentage(comparison["ttft_delta"]),
                classification=comparison["classification"],
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--throughput-regression", type=float, default=0.05)
    parser.add_argument("--ttft-regression", type=float, default=0.10)
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()
    current = normalize(args.artifact)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in current),
        encoding="utf-8",
    )
    comparisons = compare(
        current,
        read_jsonl(args.baseline),
        args.throughput_regression,
        args.ttft_regression,
    )
    write_report(args.report, comparisons)
    if args.gate and any(
        comparison["classification"]
        in {"correctness-failure", "performance-regression"}
        for comparison in comparisons
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
