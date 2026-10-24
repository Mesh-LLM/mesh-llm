#!/usr/bin/env python3
"""Turn skippy_api_ffi_benchmarks JSON into an old-vs-new comparison table.

    ./skippy_api_ffi_benchmarks --benchmark_repetitions=5 \
        --benchmark_format=json --benchmark_out=out.json
    ./compare_benchmarks.py out.json

Every benchmark family is registered twice: `Family/FFI` (the old C ABI) and
`Family/API` (the skippy_api facade). This script pairs them, reports the
ratio, and lists the geomean. A ratio above 1.0 means the new API is faster.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

UNIT_TO_NS = {"ns": 1.0, "us": 1e3, "ms": 1e6, "s": 1e9}

# google benchmark appends these to the name.
_NOISE = {"real_time", "cpu_time", "manual_time"}
_NOISE_PREFIXES = ("iterations:", "repeats:")
# With --benchmark_repetitions=N it also emits one aggregate row per statistic;
# the individual repetitions are what this script summarises.
_AGGREGATES = ("_mean", "_median", "_stddev", "_cv")


def load(path: Path) -> list[dict]:
    try:
        with path.open() as handle:
            document = json.load(handle)
    except json.JSONDecodeError as failure:
        raise SystemExit(
            f"{path}: not valid JSON ({failure.msg}); the benchmark either "
            "crashed or --benchmark_filter matched no registered benchmark "
            "(google benchmark writes no report at all in that case)"
        ) from failure
    benchmarks = document.get("benchmarks")
    if benchmarks is None:
        raise SystemExit(f"{path}: no 'benchmarks' key; pass the JSON format output")
    if not benchmarks:
        raise SystemExit(
            f"{path}: report contains zero benchmark rows; the filter matched "
            "nothing (names registered with UseRealTime end in /real_time)"
        )
    return benchmarks


def to_ns(entries: list[dict]) -> list[float]:
    return [float(entry["real_time"]) * UNIT_TO_NS[entry["time_unit"]] for entry in entries]


def median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def split(name: str) -> tuple[str, str, str]:
    """Embed/FFI/128/real_time -> ("Embed", "FFI", "128"); SessionCreate/API -> (family, side, "")."""
    parts = [
        part
        for part in name.split("/")
        if part not in _NOISE and not part.startswith(_NOISE_PREFIXES)
    ]
    for index, part in enumerate(parts):
        if part in {"FFI", "API"}:
            return "/".join(parts[:index]), part, "/".join(parts[index + 1 :])
    return name, "", ""


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2

    entries = load(Path(argv[1]))
    grouped: dict[tuple[str, str], dict[str, list[dict]]] = defaultdict(dict)
    # reason -> families that skipped for it
    skipped: dict[str, list[str]] = defaultdict(list)

    for entry in entries:
        if entry["name"].endswith(_AGGREGATES):
            continue
        family, side, arg = split(entry["name"])
        if not side:
            continue
        if entry.get("error_occurred"):
            family = split(entry["name"])[0]
            skipped.setdefault(entry.get("error_message", "skipped"), []).append(family)
            continue
        grouped[(family, arg)].setdefault(side, []).append(entry)

    print(f"{'benchmark':<32} {'FFI':>12} {'API':>12} {'API vs FFI':>12}  verdict")
    print("-" * 86)

    ratios: list[float] = []
    for (family, arg), sides in sorted(grouped.items(), key=lambda item: (item[0][0], _arg_key(item[0][1]))):
        if "FFI" not in sides or "API" not in sides:
            present = "FFI" if "FFI" in sides else "API"
            label = f"{family}/{arg}" if arg else family
            print(
                f"{label:<32} {_format_ns(median(to_ns(sides[present]))):>12} "
                f"{'-':>12} {'':>12}  only {present} ran"
            )
            continue
        ffi_ns = median(to_ns(sides["FFI"]))
        api_ns = median(to_ns(sides["API"]))
        ratio = ffi_ns / api_ns if api_ns > 0 else float("inf")
        ratios.append(ratio)
        change = (ratio - 1.0) * 100.0
        verdict = "new faster" if ratio >= 1.02 else "old faster" if ratio <= 0.98 else "same"
        label = f"{family}/{arg}" if arg else family
        print(
            f"{label:<32} {_format_ns(ffi_ns):>12} {_format_ns(api_ns):>12} "
            f"{ratio:>11.3f}x {verdict} ({change:+.1f}%)"
        )

    if ratios:
        geomean = math.exp(sum(math.log(value) for value in ratios) / len(ratios))
        print("-" * 86)
        print(f"{'geomean':<32} {'':>12} {'':>12} {geomean:>11.3f}x")
        print("\n>1.0 means the skippy_api facade did the same work in less wall time.")

    for reason, families in skipped.items():
        names = list(dict.fromkeys(families))
        shown = ", ".join(names[:6]) + (f", +{len(names) - 6} more" if len(names) > 6 else "")
        print(f"\nskipped ({shown}): {reason}")

    return 0


def _arg_key(arg: str) -> tuple[int, str]:
    try:
        return int(arg), arg
    except ValueError:
        return 0, arg


def _format_ns(value: float) -> str:
    if value >= 1e6:
        return f"{value / 1e6:.3f} ms"
    if value >= 1e3:
        return f"{value / 1e3:.3f} us"
    return f"{value:.1f} ns"


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
