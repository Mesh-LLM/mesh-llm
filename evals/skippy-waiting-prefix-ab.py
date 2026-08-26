#!/usr/bin/env python3
"""Benchmark waiting-request prefix grouping against an exact OLD/NEW pair.

The first long cache operation holds the scheduler worker while two prompt
families arrive in alternating order. With a one-entry prefix cache, FCFS can
repeatedly evict the useful family; prefix-aware waiting order should finish one
family before switching to the other. The runner alternates binary launch
order, retains raw requests and telemetry, and emits JSON plus a PR-ready
Mermaid chart and Markdown table.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import importlib.util
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
SUMMARY_EVENT = "stage.openai_generation_summary"
DEFAULT_FIXTURE_CATALOG = REPO / "evals/skippy-scheduler-fixtures.json"


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_prefix(family: str, blocks: int) -> str:
    rows = [
        f"{family}-context-{index:04d}: src/{family}/module_{index % 37}.rs "
        f"owns invariant {index}; preserve it exactly."
        for index in range(blocks)
    ]
    return (
        f"You are working in repository family {family}. Follow its fixed rules.\n"
        + "\n".join(rows)
    )


def interleaved_prompts(families: int, requests_per_family: int, blocks: int) -> list[dict[str, str]]:
    prefixes = [stable_prefix(f"family-{index}", blocks) for index in range(families)]
    prompts = []
    for request_index in range(requests_per_family):
        for family_index, prefix in enumerate(prefixes):
            prompts.append(
                {
                    "family": f"family-{family_index}",
                    "prompt": (
                        f"{prefix}\nUnique task {request_index}: inspect module_"
                        f"{request_index % 37}.rs and return its invariant only."
                    ),
                }
            )
    return prompts


def read_prompt_manifest(path: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    document = json.loads(path.read_text())
    if not isinstance(document, dict) or not isinstance(document.get("prompts"), list):
        raise ValueError("prompt manifest must be an object with a prompts list")
    prompts = []
    for index, item in enumerate(document["prompts"]):
        if not isinstance(item, dict):
            raise ValueError(f"prompt manifest item {index} must be an object")
        family, prompt = item.get("family"), item.get("prompt")
        if not isinstance(family, str) or not family:
            raise ValueError(f"prompt manifest item {index} needs a nonempty family")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"prompt manifest item {index} needs a nonempty prompt")
        prompts.append({"family": family, "prompt": prompt})
    if not prompts:
        raise ValueError("prompt manifest must contain at least one prompt")
    metadata = document.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("prompt manifest metadata must be an object")
    return prompts, metadata


def apply_fixture_profile(args: argparse.Namespace) -> tuple[dict[str, Any] | None, str | None]:
    if args.fixture_profile is None:
        return None, None
    fixtures = load_module(
        "skippy_scheduler_fixtures", REPO / "evals/skippy-scheduler-fixtures.py"
    )
    catalog_path = args.fixture_catalog.resolve()
    catalog = fixtures.load_catalog(catalog_path)
    selected = fixtures.profile(catalog, args.fixture_profile)
    for key, value in selected["workload"].items():
        setattr(args, key, value)
    return selected, sha256(catalog_path)


def validate_fixture_inputs(
    selected: dict[str, Any],
    model_id: str,
    model_sha256: str,
    prompt_manifest: Path | None,
) -> None:
    expected_model = selected["model"]
    if model_id != expected_model["id"]:
        raise ValueError(
            f"model id must be {expected_model['id']}, got {model_id}"
        )
    if model_sha256 != expected_model["sha256"]:
        raise ValueError(
            "model SHA-256 must be "
            f"{expected_model['sha256']}, got {model_sha256}"
        )
    corpus_kind = selected["corpus"]["kind"]
    if corpus_kind == "hf" and prompt_manifest is None:
        raise ValueError("HF fixture profiles require a prepared prompt manifest")
    if corpus_kind == "synthetic" and prompt_manifest is not None:
        raise ValueError("synthetic fixture profiles do not accept a prompt manifest")


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(round((len(ordered) - 1) * quantile), len(ordered) - 1)
    return ordered[index]


def run_requests(
    prompts: list[dict[str, str]],
    model_id: str,
    output_tokens: int,
    port: int,
    timeout: float,
    stagger_ms: float,
) -> tuple[list[dict[str, Any]], float]:
    workload_started = time.monotonic()

    def make_request(request_id: int, item: dict[str, str]) -> dict[str, Any]:
        target_start = workload_started + request_id * stagger_ms / 1000.0
        remaining = target_start - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        submitted_at = time.monotonic()
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": item["prompt"]}],
            "max_tokens": output_tokens,
            "temperature": 0,
            "seed": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        first_token_at = None
        completion_tokens = 0
        content_events = 0
        cached_tokens = 0
        content_parts: list[str] = []
        connection = None
        try:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
            connection.request(
                "POST",
                "/v1/chat/completions",
                json.dumps(payload),
                {"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            if response.status != 200:
                body = response.read(4096).decode("utf-8", errors="replace")
                return {
                    "request_id": request_id,
                    "family": item["family"],
                    "error": f"HTTP {response.status}: {body}",
                }
            for raw_line in response:
                line = raw_line.strip()
                if not line.startswith(b"data: "):
                    continue
                event_bytes = line[6:]
                if event_bytes == b"[DONE]":
                    break
                try:
                    event = json.loads(event_bytes)
                except json.JSONDecodeError:
                    continue
                usage = event.get("usage")
                if isinstance(usage, dict):
                    if isinstance(usage.get("completion_tokens"), int):
                        completion_tokens = usage["completion_tokens"]
                    details = usage.get("prompt_tokens_details")
                    if isinstance(details, dict) and isinstance(details.get("cached_tokens"), int):
                        cached_tokens = details["cached_tokens"]
                choices = event.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                delta = choices[0].get("delta")
                if not isinstance(delta, dict):
                    continue
                content = delta.get("content") or delta.get("reasoning_content")
                if content:
                    if first_token_at is None:
                        first_token_at = time.monotonic()
                    content_events += 1
                    content_parts.append(content)
            completed_at = time.monotonic()
            if first_token_at is None:
                return {
                    "request_id": request_id,
                    "family": item["family"],
                    "error": "stream completed without generated content",
                }
            predicted = completion_tokens or content_events
            return {
                "request_id": request_id,
                "family": item["family"],
                "prompt_sha256": hashlib.sha256(item["prompt"].encode()).hexdigest(),
                "submitted_ms": (submitted_at - workload_started) * 1000,
                "first_token_ms": (first_token_at - workload_started) * 1000,
                "completed_ms": (completed_at - workload_started) * 1000,
                "ttft_ms": (first_token_at - submitted_at) * 1000,
                "elapsed_ms": (completed_at - submitted_at) * 1000,
                "tpot_ms": (completed_at - first_token_at)
                * 1000
                / max(predicted - 1, 1),
                "tokens_predicted": predicted,
                "cached_tokens": cached_tokens,
                "content": "".join(content_parts),
            }
        except Exception as error:  # noqa: BLE001 - errors belong in the artifact.
            return {
                "request_id": request_id,
                "family": item["family"],
                "error": str(error),
            }
        finally:
            if connection is not None:
                connection.close()

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(prompts)) as executor:
        futures = [
            executor.submit(make_request, request_id, item)
            for request_id, item in enumerate(prompts)
        ]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    makespan_ms = (time.monotonic() - workload_started) * 1000
    results.sort(key=lambda row: row["request_id"])
    return results, makespan_ms


def attributes(event: dict[str, Any]) -> dict[str, Any]:
    value = event.get("attributes")
    return value if isinstance(value, dict) else {}


def summarize(requests: list[dict[str, Any]], events: list[dict[str, Any]], makespan_ms: float) -> dict[str, Any]:
    successful = [row for row in requests if "error" not in row]
    attrs = [attributes(event) for event in events]
    statuses = [str(row.get("skippy.kv.status", "unknown")) for row in attrs]

    def numeric(key: str) -> list[float]:
        return [float(row[key]) for row in attrs if isinstance(row.get(key), (int, float))]

    ttft = [float(row["ttft_ms"]) for row in successful]
    matched = numeric("skippy.kv.matched_prefix_tokens")
    suffix = numeric("skippy.kv.suffix_prefill_tokens")
    service_order = sorted(successful, key=lambda row: row["first_token_ms"])
    family_order = [str(row["family"]) for row in service_order]
    switches = sum(left != right for left, right in zip(family_order, family_order[1:]))
    output_tokens = sum(int(row["tokens_predicted"]) for row in successful)
    return {
        "requests": len(requests),
        "successful": len(successful),
        "errors": [row for row in requests if "error" in row],
        "cache_hits": statuses.count("hit"),
        "cache_misses": statuses.count("miss"),
        "usage_cached_requests": sum(int(row.get("cached_tokens", 0)) > 0 for row in successful),
        "matched_prefix_tokens_total": sum(matched),
        "suffix_prefill_tokens_total": sum(suffix),
        "ttft_ms_p50": percentile(ttft, 0.50),
        "ttft_ms_p95": percentile(ttft, 0.95),
        "makespan_ms": makespan_ms,
        "output_tokens_per_second": output_tokens / (makespan_ms / 1000.0),
        "family_switches": switches,
        "family_service_order": family_order,
    }


def run_cell(
    radix: Any,
    harness: Any,
    version: str,
    binary: Path,
    case: Any,
    round_index: int,
    args: argparse.Namespace,
    prompts: list[dict[str, str]],
) -> dict[str, Any]:
    cell_dir = args.output_dir / f"round-{round_index + 1}-{version}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    config_path = cell_dir / "stage.json"
    log_path = cell_dir / "server.log"
    radix.write_config(
        harness,
        config_path,
        case,
        True,
        args.ctx_size,
        args.lanes,
        args.n_gpu_layers,
    )
    config = json.loads(config_path.read_text())
    config["kv_cache"]["max_entries"] = args.cache_entries
    config["kv_cache"]["shared_prefix_record_limit"] = 1
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    port = harness.free_port()
    cmd = [
        str(binary),
        "serve-openai",
        "--config",
        str(config_path),
        "--bind-addr",
        f"127.0.0.1:{port}",
        "--generation-concurrency",
        str(args.admission_concurrency),
        "--telemetry-level",
        "debug",
    ]
    env = os.environ.copy()
    env["LLAMA_STAGE_BUILD_DIR"] = str(args.native_build.resolve())
    env["SKIPPY_TELEMETRY_STDERR"] = "1"
    env["SKIPPY_NATIVE_MTP_GREEDY_SAMPLING_FASTPATH"] = "1"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            cmd,
            cwd=REPO,
            env=env,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            harness.wait_ready(port, process, 900, path="/v1/models", server_name=version)
            event_start = len(radix.json_events(log_path, SUMMARY_EVENT))
            requests, makespan_ms = run_requests(
                prompts,
                case.model_id,
                args.output_tokens,
                port,
                args.request_timeout_secs,
                args.stagger_ms,
            )
            expected = sum("error" not in row for row in requests)
            events = radix.wait_for_json_events(
                log_path,
                SUMMARY_EVENT,
                event_start + expected,
                process,
            )[event_start:]
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
    return {
        "round": round_index + 1,
        "version": version,
        "binary": str(binary),
        "config": str(config_path),
        "log": str(log_path),
        "requests": requests,
        "summary": summarize(requests, events, makespan_ms),
    }


def aggregate(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for version in ("old", "new"):
        summaries = [cell["summary"] for cell in cells if cell["version"] == version]

        def median(key: str, summaries: list[dict[str, Any]] = summaries) -> float | None:
            values = [float(row[key]) for row in summaries if row.get(key) is not None]
            return statistics.median(values) if values else None

        rows.append(
            {
                "version": version,
                "rounds": len(summaries),
                "requests": sum(int(row["requests"]) for row in summaries),
                "successful": sum(int(row["successful"]) for row in summaries),
                "cache_hits_median": median("cache_hits"),
                "suffix_prefill_tokens_median": median("suffix_prefill_tokens_total"),
                "ttft_ms_p50_median": median("ttft_ms_p50"),
                "ttft_ms_p95_median": median("ttft_ms_p95"),
                "makespan_ms_median": median("makespan_ms"),
                "output_tokens_per_second_median": median("output_tokens_per_second"),
                "family_switches_median": median("family_switches"),
            }
        )
    return rows


def delta(old: float | None, new: float | None) -> float | None:
    if old is None or new is None:
        return None
    if old == 0:
        return 0.0 if new == 0 else None
    return (new - old) / old * 100.0


def format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def format_delta(old: float | None, new: float | None) -> str:
    value = delta(old, new)
    return "n/a" if value is None else f"{value:+.1f}%"


def evaluate_acceptance(
    rows: list[dict[str, Any]], profile: dict[str, Any] | None
) -> dict[str, Any] | None:
    if profile is None:
        return None
    indexed = {row["version"]: row for row in rows}
    old, new = indexed["old"], indexed["new"]
    contract = profile["hardware_acceptance"]
    checks: list[dict[str, Any]] = []
    known_contract_keys = {
        "successful_requests_per_binary",
        "absolute_user_metric_delta_percent_max",
        "suffix_prefill_before_min",
        "family_switch_before_min",
    }
    for prefix in (
        "suffix_prefill",
        "family_switch",
        "ttft_p95",
        "makespan",
        "output_throughput",
    ):
        known_contract_keys.update(
            {
                f"{prefix}_delta_percent",
                f"{prefix}_delta_percent_max",
                f"{prefix}_delta_percent_min",
            }
        )
    unknown_contract_keys = set(contract) - known_contract_keys
    if unknown_contract_keys:
        unknown = ", ".join(sorted(unknown_contract_keys))
        raise ValueError(f"unsupported hardware acceptance keys: {unknown}")

    def record(label: str, actual: float | None, relation: str, threshold: Any) -> None:
        if actual is None:
            passed = False
        elif relation == "eq":
            passed = actual == threshold
        elif relation == "max":
            passed = actual <= threshold
        elif relation == "min":
            passed = actual >= threshold
        elif relation == "range":
            passed = threshold["min"] <= actual <= threshold["max"]
        else:
            raise ValueError(f"unsupported acceptance relation: {relation}")
        checks.append(
            {
                "label": label,
                "actual": actual,
                "relation": relation,
                "threshold": threshold,
                "passed": passed,
            }
        )

    expected_successes = contract["successful_requests_per_binary"]
    record("before successful requests", old["successful"], "eq", expected_successes)
    record("after successful requests", new["successful"], "eq", expected_successes)
    if "suffix_prefill_before_min" in contract:
        record(
            "before suffix prefill pressure",
            old["suffix_prefill_tokens_median"],
            "min",
            contract["suffix_prefill_before_min"],
        )
    if "family_switch_before_min" in contract:
        record(
            "before family-switch pressure",
            old["family_switches_median"],
            "min",
            contract["family_switch_before_min"],
        )
    delta_keys = {
        "suffix_prefill": "suffix_prefill_tokens_median",
        "family_switch": "family_switches_median",
        "ttft_p95": "ttft_ms_p95_median",
        "makespan": "makespan_ms_median",
        "output_throughput": "output_tokens_per_second_median",
    }
    for prefix, row_key in delta_keys.items():
        actual = delta(old[row_key], new[row_key])
        range_key = f"{prefix}_delta_percent"
        max_key = f"{prefix}_delta_percent_max"
        min_key = f"{prefix}_delta_percent_min"
        if range_key in contract:
            record(f"{prefix} delta percent", actual, "range", contract[range_key])
        if max_key in contract:
            record(f"{prefix} delta percent", actual, "max", contract[max_key])
        if min_key in contract:
            record(f"{prefix} delta percent", actual, "min", contract[min_key])
    if "absolute_user_metric_delta_percent_max" in contract:
        threshold = contract["absolute_user_metric_delta_percent_max"]
        for label, row_key in (
            ("TTFT p50", "ttft_ms_p50_median"),
            ("TTFT p95", "ttft_ms_p95_median"),
            ("makespan", "makespan_ms_median"),
            ("output throughput", "output_tokens_per_second_median"),
        ):
            value = delta(old[row_key], new[row_key])
            record(
                f"absolute {label} delta percent",
                abs(value) if value is not None else None,
                "max",
                threshold,
            )
    return {"passed": all(check["passed"] for check in checks), "checks": checks}


def report(rows: list[dict[str, Any]], acceptance: dict[str, Any] | None = None) -> str:
    indexed = {row["version"]: row for row in rows}
    old, new = indexed["old"], indexed["new"]
    old_suffix = old["suffix_prefill_tokens_median"]
    new_suffix = new["suffix_prefill_tokens_median"]
    lines = []
    if old_suffix is not None and new_suffix is not None:
        suffix_ceiling = max(old_suffix, new_suffix, 1)
        lines.extend(
            [
                "```mermaid",
                "xychart-beta",
                '    title "Waiting-prefix A/B: suffix tokens prefetched (lower is better)"',
                '    x-axis ["Before", "After"]',
                f"    y-axis \"tokens\" 0 --> {int(suffix_ceiling * 1.1)}",
                f"    bar [{old_suffix:.0f}, {new_suffix:.0f}]",
                "```",
                "",
            ]
        )
    lines.extend(
        [
        "| Metric | Before | After | Delta |",
        "| --- | ---: | ---: | ---: |",
        ]
    )
    metrics = (
        ("Cache hits / round", "cache_hits_median", False),
        ("Suffix prefill tokens / round", "suffix_prefill_tokens_median", True),
        ("Family switches / round", "family_switches_median", True),
        ("TTFT p50 ms", "ttft_ms_p50_median", True),
        ("TTFT p95 ms", "ttft_ms_p95_median", True),
        ("Makespan ms", "makespan_ms_median", True),
        ("Output tok/s", "output_tokens_per_second_median", False),
    )
    for label, key, _lower_is_better in metrics:
        before, after = old[key], new[key]
        lines.append(
            f"| {label} | {format_metric(before)} | {format_metric(after)} | "
            f"{format_delta(before, after)} |"
        )
    if acceptance is not None:
        lines.extend(
            [
                "",
                f"Fixture acceptance: **{'PASS' if acceptance['passed'] else 'FAIL'}**",
                "",
                "| Check | Actual | Required | Result |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for check in acceptance["checks"]:
            threshold = check["threshold"]
            if isinstance(threshold, dict):
                required = f"{threshold['min']} to {threshold['max']}"
            else:
                operators = {"eq": "=", "max": "≤", "min": "≥"}
                required = f"{operators[check['relation']]} {threshold}"
            actual = format_metric(check["actual"])
            result = "PASS" if check["passed"] else "FAIL"
            lines.append(f"| {check['label']} | {actual} | {required} | {result} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-file", type=Path, required=True)
    parser.add_argument("--old-bin", type=Path, required=True)
    parser.add_argument("--new-bin", type=Path, required=True)
    parser.add_argument("--old-commit", required=True)
    parser.add_argument("--new-commit", required=True)
    parser.add_argument("--native-build", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--fixture-profile",
        help="checked-in workload profile; its shape overrides workload flags",
    )
    parser.add_argument(
        "--fixture-catalog",
        type=Path,
        default=DEFAULT_FIXTURE_CATALOG,
    )
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--families", type=int, default=2)
    parser.add_argument("--requests-per-family", type=int, default=6)
    parser.add_argument("--prefix-blocks", type=int, default=192)
    parser.add_argument(
        "--prompt-manifest",
        type=Path,
        help="JSON object with metadata and a deterministic prompts list",
    )
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--ctx-size", type=int, default=65536)
    parser.add_argument("--lanes", type=int, default=12)
    parser.add_argument(
        "--admission-concurrency",
        type=int,
        default=0,
        help="accepted in-flight requests; 0 admits the full generated workload",
    )
    parser.add_argument("--cache-entries", type=int, default=1)
    parser.add_argument("--stagger-ms", type=float, default=5.0)
    parser.add_argument("--request-timeout-secs", type=float, default=900.0)
    parser.add_argument("--n-gpu-layers", type=int, default=999)
    args = parser.parse_args()
    try:
        fixture_profile, fixture_catalog_sha256 = apply_fixture_profile(args)
    except (json.JSONDecodeError, OSError, ValueError) as error:
        parser.error(f"invalid fixture profile: {error}")
    if min(args.rounds, args.families, args.requests_per_family, args.lanes, args.cache_entries) <= 0:
        parser.error("rounds, families, requests, lanes, and cache entries must be positive")
    if args.admission_concurrency < 0:
        parser.error("admission concurrency must be non-negative")
    if args.stagger_ms < 0:
        parser.error("stagger must be non-negative")
    binaries = {"old": args.old_bin.resolve(), "new": args.new_bin.resolve()}
    for name, binary in binaries.items():
        if not binary.is_file():
            parser.error(f"{name} binary not found: {binary}")
    radix = load_module("skippy_radix_cache_ab", REPO / "evals/skippy-radix-cache-ab.py")
    harness = radix.load_production_harness()
    cases = radix.read_cases(args.case_file)
    if len(cases) != 1:
        parser.error("case file must contain exactly one model")
    case = cases[0]
    model_sha256 = harness.model_sha256(case.model_path)
    if fixture_profile is not None:
        try:
            validate_fixture_inputs(
                fixture_profile,
                case.model_id,
                model_sha256,
                args.prompt_manifest,
            )
        except ValueError as error:
            parser.error(f"fixture input mismatch: {error}")
    prompt_manifest_metadata: dict[str, Any] = {}
    if args.prompt_manifest is not None:
        prompt_manifest = args.prompt_manifest.resolve()
        if not prompt_manifest.is_file():
            parser.error(f"prompt manifest not found: {prompt_manifest}")
        try:
            prompts, prompt_manifest_metadata = read_prompt_manifest(prompt_manifest)
        except (json.JSONDecodeError, OSError, ValueError) as error:
            parser.error(f"invalid prompt manifest: {error}")
        if fixture_profile is not None and fixture_profile["corpus"]["kind"] == "hf":
            expected_manifest_sha = fixture_profile["corpus"]["prompt_manifest_sha256"]
            actual_manifest_sha = sha256(prompt_manifest)
            if actual_manifest_sha != expected_manifest_sha:
                parser.error(
                    "prompt manifest does not match fixture profile: "
                    f"expected {expected_manifest_sha}, got {actual_manifest_sha}"
                )
    else:
        prompt_manifest = None
        prompts = interleaved_prompts(args.families, args.requests_per_family, args.prefix_blocks)
    family_request_counts: dict[str, int] = {}
    for prompt in prompts:
        family = prompt["family"]
        family_request_counts[family] = family_request_counts.get(family, 0) + 1
    requests_per_family = set(family_request_counts.values())
    if args.admission_concurrency == 0:
        args.admission_concurrency = len(prompts)
    if args.admission_concurrency < len(prompts):
        parser.error("admission concurrency must cover the generated workload")
    if args.admission_concurrency > args.lanes:
        parser.error("admission concurrency cannot exceed configured runtime lanes")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cells = []
    for round_index in range(args.rounds):
        versions = ("old", "new") if round_index % 2 == 0 else ("new", "old")
        for version in versions:
            print(f"==> round={round_index + 1} version={version}", flush=True)
            cells.append(
                run_cell(
                    radix,
                    harness,
                    version,
                    binaries[version],
                    case,
                    round_index,
                    args,
                    prompts,
                )
            )
    rows = aggregate(cells)
    acceptance = evaluate_acceptance(rows, fixture_profile)
    result = {
        "metadata": {
            "old": {"commit": args.old_commit, "binary": str(binaries["old"]), "sha256": sha256(binaries["old"])},
            "new": {"commit": args.new_commit, "binary": str(binaries["new"]), "sha256": sha256(binaries["new"])},
            "model_id": case.model_id,
            "model_path": str(case.model_path),
            "model_sha256": model_sha256,
            "rounds": args.rounds,
            "families": len(family_request_counts),
            "requests_per_family": (
                requests_per_family.pop() if len(requests_per_family) == 1 else None
            ),
            "family_request_counts": family_request_counts,
            "prefix_blocks": args.prefix_blocks,
            "prompt_manifest": str(prompt_manifest) if prompt_manifest is not None else None,
            "prompt_manifest_sha256": (
                sha256(prompt_manifest) if prompt_manifest is not None else None
            ),
            "prompt_manifest_metadata": prompt_manifest_metadata,
            "fixture_profile": args.fixture_profile,
            "fixture_catalog": (
                str(args.fixture_catalog.resolve()) if fixture_profile is not None else None
            ),
            "fixture_catalog_sha256": fixture_catalog_sha256,
            "output_tokens": args.output_tokens,
            "lanes": args.lanes,
            "admission_concurrency": args.admission_concurrency,
            "cache_entries": args.cache_entries,
            "stagger_ms": args.stagger_ms,
        },
        "cells": cells,
        "aggregate": rows,
        "acceptance": acceptance,
    }
    (args.output_dir / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output_dir / "report.md").write_text(report(rows, acceptance))
    print(args.output_dir / "comparison.json")
    successful = all(
        cell["summary"]["successful"] == cell["summary"]["requests"] for cell in cells
    )
    accepted = acceptance is None or acceptance["passed"]
    return 0 if successful and accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
