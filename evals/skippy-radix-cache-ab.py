#!/usr/bin/env python3
"""Run a reproducible OLD/NEW Skippy prefix-cache A/B.

The benchmark launches exact release binaries against one model at a time and
records two independent comparisons:

* disabled-cache versus enabled-cache exact-prefix reuse, which measures cache
  lift without attributing unrelated scheduler changes to the cache; and
* same-prefix/different-tail reuse, which is the behavior added by the unified
  token radix tree; and
* a growing coding-agent/tool-result trace whose stable repository prefix is
  reused across turns.

The runner alternates OLD/NEW launch order between rounds, retains every raw
request and telemetry event, checks deterministic output parity, and writes a
JSON result, a Markdown table, and an SVG chart without third-party packages.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
SUMMARY_EVENT = "stage.openai_generation_summary"
RADIX_KEYS = (
    "skippy.kv.radix.namespaces",
    "skippy.kv.radix.nodes",
    "skippy.kv.radix.token_edges",
    "skippy.kv.radix.splits",
    "skippy.kv.radix.resident_entries",
    "skippy.kv.radix.resident_active_refs",
    "skippy.kv.radix.recurrent_entries",
    "skippy.kv.radix.recurrent_active_refs",
    "skippy.kv.radix.resident_evictions",
    "skippy.kv.radix.recurrent_evictions",
)


@dataclass(frozen=True)
class ModelCase:
    key: str
    family: str
    model_id: str
    model_path: Path
    layer_end: int
    payload: str


def load_production_harness() -> Any:
    path = REPO / "evals/skippy-cache-production-bench.py"
    spec = importlib.util.spec_from_file_location("skippy_cache_production_bench", path)
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


def read_cases(path: Path) -> list[ModelCase]:
    payload = json.loads(path.read_text())
    raw_cases = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("case file must contain a non-empty cases array")
    cases = []
    for row in raw_cases:
        case = ModelCase(
            key=str(row["key"]),
            family=str(row["family"]),
            model_id=str(row["model_id"]),
            model_path=Path(os.path.expandvars(row["model_path"])).expanduser().resolve(),
            layer_end=int(row["layer_end"]),
            payload=str(row["payload"]),
        )
        if case.payload not in {"resident-kv", "kv-recurrent", "full-state"}:
            raise ValueError(f"{case.key}: unsupported payload {case.payload!r}")
        if not case.model_path.is_file():
            raise FileNotFoundError(f"{case.key}: model not found: {case.model_path}")
        cases.append(case)
    return cases


def stable_prefix(blocks: int) -> str:
    header = (
        "You are a deterministic coding assistant. Preserve the repository rules, "
        "tool schema, file inventory, and conversation facts below.\n"
    )
    rows = [
        f"context-block-{index:04d}: src/module_{index % 37}.rs owns invariant "
        f"{index}; edits require tests and exact output parity."
        for index in range(blocks)
    ]
    return header + "\n".join(rows)


def prompts(blocks: int) -> dict[str, tuple[str, str]]:
    exact_prefix = f"Exact-prefix workload.\n{stable_prefix(blocks)}"
    divergent_prefix = f"Divergent-prefix workload.\n{stable_prefix(blocks)}"
    exact = f"{exact_prefix}\nTask: identify the owner of invariant 17."
    divergent_a = (
        f"{divergent_prefix}\nTool result alpha: inspect module_17 and return its invariant."
    )
    divergent_b = (
        f"{divergent_prefix}\nTool result beta: inspect module_18 and return its invariant."
    )
    coding_prefix = f"Coding-agent tool loop.\n{stable_prefix(blocks)}"
    coding_warmup = (
        f"{coding_prefix}\nTurn 0 user: inspect the repository root.\n"
        "Turn 0 tool: Cargo.toml and crates/ were found."
    )
    return {
        "exact": (exact, exact),
        "divergent": (divergent_a, divergent_b),
        # A real multi-turn agent appends to its previous full transcript.
        # Use the warmed transcript as the measured base so an exact recurrent
        # checkpoint can be restored before the new tool turns are appended.
        "coding": (coding_warmup, coding_warmup),
    }


def divergent_prompt(base: str, concurrency: int, request_id: int) -> str:
    markers = (
        "amber",
        "birch",
        "cobalt",
        "delta",
        "ember",
        "fjord",
        "garnet",
        "harbor",
        "indigo",
        "juniper",
        "kelp",
        "lilac",
        "marble",
        "nectar",
        "onyx",
        "pearl",
    )
    marker = markers[request_id % len(markers)]
    return (
        f"{base}\nUnique branch {marker}-{concurrency}-{request_id}: "
        "return only the requested invariant."
    )


def coding_agent_prompt(base: str, concurrency: int, request_id: int) -> str:
    turns = [base]
    for turn in range(request_id + 1):
        turns.append(
            f"Turn {turn + 1} user: inspect module_{turn % 37}.rs for invariant {turn}."
        )
        turns.append(
            f"Turn {turn + 1} tool: module_{turn % 37}.rs preserves invariant {turn}; "
            f"trace lane {concurrency}."
        )
    turns.append("Assistant: return the latest invariant only.")
    return "\n".join(turns)


def run_prompt_requests(
    prompts_to_run: list[str],
    model_id: str,
    concurrency: int,
    output_tokens: int,
    port: int,
    timeout: float,
) -> tuple[list[dict[str, Any]], float]:
    """Run deterministic OpenAI SSE requests whose prompts may differ."""

    def make_request(request_id: int, prompt: str) -> dict[str, Any]:
        request_payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": output_tokens,
            "temperature": 0,
            "seed": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        started = time.monotonic()
        first_token_at = None
        content_events = 0
        completion_tokens = 0
        content_parts: list[str] = []
        connection = None
        try:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
            connection.request(
                "POST",
                "/v1/chat/completions",
                json.dumps(request_payload),
                {"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            if response.status != 200:
                body = response.read(4096).decode("utf-8", errors="replace")
                return {
                    "request_id": request_id,
                    "error": f"HTTP {response.status}: {body}",
                    "elapsed_ms": (time.monotonic() - started) * 1000,
                }
            for raw_line in response:
                line = raw_line.strip()
                if not line.startswith(b"data: "):
                    continue
                payload_bytes = line[6:]
                if payload_bytes == b"[DONE]":
                    break
                try:
                    event = json.loads(payload_bytes)
                except json.JSONDecodeError:
                    continue
                usage = event.get("usage")
                if isinstance(usage, dict) and isinstance(
                    usage.get("completion_tokens"), int
                ):
                    completion_tokens = usage["completion_tokens"]
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
            completed = time.monotonic()
            if first_token_at is None:
                return {
                    "request_id": request_id,
                    "error": "stream completed without generated content",
                    "elapsed_ms": (completed - started) * 1000,
                }
            tokens_predicted = completion_tokens or content_events
            return {
                "request_id": request_id,
                "elapsed_ms": (completed - started) * 1000,
                "ttft_ms": (first_token_at - started) * 1000,
                "tpot_ms": (completed - first_token_at)
                * 1000
                / max(tokens_predicted - 1, 1),
                "tokens_predicted": tokens_predicted,
                "content": "".join(content_parts),
                "first_content": content_parts[0],
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            }
        except Exception as error:  # noqa: BLE001 - retain errors in the artifact.
            return {
                "request_id": request_id,
                "error": str(error),
                "elapsed_ms": (time.monotonic() - started) * 1000,
            }
        finally:
            if connection is not None:
                connection.close()

    workload_started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(make_request, request_id, prompt)
            for request_id, prompt in enumerate(prompts_to_run)
        ]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    results.sort(key=lambda result: result["request_id"])
    return results, (time.monotonic() - workload_started) * 1000


def write_config(
    harness: Any,
    path: Path,
    case: ModelCase,
    cache_enabled: bool,
    ctx_size: int,
    lanes: int,
    n_gpu_layers: int,
) -> None:
    config: dict[str, Any] = {
        "run_id": "skippy-radix-cache-ab",
        "topology_id": "skippy-radix-cache-ab-single-stage",
        "model_id": case.model_id,
        "model_path": str(case.model_path),
        "source_model_sha256": harness.model_sha256(case.model_path),
        "stage_id": "stage-0",
        "stage_index": 0,
        "layer_start": 0,
        "layer_end": case.layer_end,
        "ctx_size": ctx_size,
        "lane_count": lanes,
        "n_gpu_layers": n_gpu_layers,
        "filter_tensors_on_load": False,
        "load_mode": "runtime-slice",
        "bind_addr": "127.0.0.1:0",
        "upstream": None,
        "downstream": None,
    }
    if cache_enabled:
        config["kv_cache"] = {
            "mode": "lookup-record",
            "payload": case.payload,
            "max_entries": 64,
            "max_bytes": 0,
            "min_tokens": 64,
            "shared_prefix_stride_tokens": 128,
            "shared_prefix_record_limit": 4,
        }
    path.write_text(json.dumps(config, indent=2) + "\n")


def json_events(path: Path, event_name: str) -> list[dict[str, Any]]:
    events = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") == event_name:
            events.append(event)
    return events


def wait_for_json_events(
    path: Path,
    event_name: str,
    expected_count: int,
    process: subprocess.Popen[str],
    timeout_seconds: float = 30.0,
) -> list[dict[str, Any]]:
    """Wait until asynchronous server telemetry reaches a known boundary."""
    deadline = time.monotonic() + timeout_seconds
    while True:
        events = json_events(path, event_name)
        if len(events) >= expected_count:
            return events
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"server exited with code {return_code} while waiting for "
                f"{event_name}: observed {len(events)}/{expected_count} events"
            )
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"timed out waiting for {event_name}: "
                f"observed {len(events)}/{expected_count} events"
            )
        time.sleep(0.01)


def attributes(event: dict[str, Any]) -> dict[str, Any]:
    value = event.get("attributes")
    return value if isinstance(value, dict) else {}


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(round((len(ordered) - 1) * quantile), len(ordered) - 1)
    return ordered[index]


def summarize_requests(
    requests: list[dict[str, Any]], events: list[dict[str, Any]]
) -> dict[str, Any]:
    successful = [row for row in requests if "error" not in row]
    attrs = [attributes(event) for event in events]
    statuses = [str(row.get("skippy.kv.status", "unknown")) for row in attrs]
    numeric = lambda key: [int(row[key]) for row in attrs if isinstance(row.get(key), (int, float))]
    matched = numeric("skippy.kv.matched_prefix_tokens")
    suffix = numeric("skippy.kv.suffix_prefill_tokens")
    prompt = numeric("llama_stage.prompt_token_count")
    ttft = [float(row["ttft_ms"]) for row in successful]
    tpot = [float(row["tpot_ms"]) for row in successful]
    elapsed = [float(row["elapsed_ms"]) for row in successful]
    outputs_by_prompt: dict[str, set[str]] = {}
    for row in successful:
        prompt_hash = str(row.get("prompt_sha256", "unknown"))
        outputs_by_prompt.setdefault(prompt_hash, set()).add(str(row.get("content", "")))
    return {
        "requests": len(requests),
        "successful": len(successful),
        "errors": [row for row in requests if "error" in row],
        "cache_hits": statuses.count("hit"),
        "cache_misses": statuses.count("miss"),
        "cache_disabled": statuses.count("disabled"),
        "cache_statuses": sorted(set(statuses)),
        "matched_prefix_tokens_median": statistics.median(matched) if matched else None,
        "suffix_prefill_tokens_median": statistics.median(suffix) if suffix else None,
        "prompt_tokens_median": statistics.median(prompt) if prompt else None,
        "matched_prefix_tokens": matched,
        "suffix_prefill_tokens": suffix,
        "prompt_tokens": prompt,
        "ttft_ms": ttft,
        "tpot_ms": tpot,
        "elapsed_ms": elapsed,
        "ttft_ms_p50": percentile(ttft, 0.50),
        "ttft_ms_p99": percentile(ttft, 0.99),
        "tpot_ms_p50": percentile(tpot, 0.50),
        "elapsed_ms_p50": percentile(elapsed, 0.50),
        "outputs": sorted({str(row.get("content", "")) for row in successful}),
        "outputs_by_prompt": {
            prompt_hash: sorted(outputs)
            for prompt_hash, outputs in sorted(outputs_by_prompt.items())
        },
        "first_outputs": sorted({str(row.get("first_content", "")) for row in successful}),
        "radix_final": {
            key: attrs[-1].get(key) for key in RADIX_KEYS if attrs and key in attrs[-1]
        },
    }


def run_server_cell(
    harness: Any,
    version: str,
    binary: Path,
    case: ModelCase,
    cache_enabled: bool,
    round_index: int,
    concurrency_levels: list[int],
    requests_per_level: int,
    output_tokens: int,
    blocks: int,
    ctx_size: int,
    lanes: int,
    n_gpu_layers: int,
    native_build: Path,
    output_dir: Path,
) -> dict[str, Any]:
    mode = "warm" if cache_enabled else "cold"
    cell_name = f"round-{round_index + 1}-{version}-{mode}"
    cell_dir = output_dir / cell_name
    cell_dir.mkdir(parents=True, exist_ok=True)
    config_path = cell_dir / "stage.json"
    log_path = cell_dir / "server.log"
    write_config(
        harness,
        config_path,
        case,
        cache_enabled,
        ctx_size,
        lanes,
        n_gpu_layers,
    )
    port = harness.free_port()
    cmd = [
        str(binary),
        "serve-openai",
        "--config",
        str(config_path),
        "--bind-addr",
        f"127.0.0.1:{port}",
        "--generation-concurrency",
        str(lanes),
        "--telemetry-level",
        "debug",
    ]
    env = os.environ.copy()
    env["LLAMA_STAGE_BUILD_DIR"] = str(native_build)
    env["SKIPPY_TELEMETRY_STDERR"] = "1"
    env["SKIPPY_NATIVE_MTP_GREEDY_SAMPLING_FASTPATH"] = "1"
    request_args = argparse.Namespace(request_timeout_secs=900)
    observations = []
    prompt_pairs = prompts(blocks)
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
            harness.wait_ready(port, process, 900, path="/v1/models", server_name=cell_name)
            for scenario, (warmup_prompt, measured_prompt) in prompt_pairs.items():
                for concurrency in concurrency_levels:
                    # Refresh the intended backing prefix immediately before
                    # each measured level. Earlier concurrency levels record
                    # their own branches and can otherwise turn the next level
                    # into a cache-capacity test instead of a matched A/B.
                    if cache_enabled:
                        warmup_event_start = len(json_events(log_path, SUMMARY_EVENT))
                        warmup_requests, _ = harness.run_openai_concurrent_requests(
                            warmup_prompt, case.model_id, 1, 1, output_tokens, port, request_args
                        )
                        if warmup_requests[0].get("error"):
                            raise RuntimeError(
                                f"{cell_name} {scenario}/n{concurrency} warmup failed: "
                                f"{warmup_requests[0]}"
                            )
                        events = wait_for_json_events(
                            log_path,
                            SUMMARY_EVENT,
                            warmup_event_start + 1,
                            process,
                        )
                        event_start = len(events)
                    else:
                        event_start = len(json_events(log_path, SUMMARY_EVENT))
                    request_count = max(requests_per_level, concurrency)
                    if scenario == "divergent":
                        measured_prompts = [
                            divergent_prompt(measured_prompt, concurrency, request_id)
                            for request_id in range(request_count)
                        ]
                        measured, makespan_ms = run_prompt_requests(
                            measured_prompts,
                            case.model_id,
                            concurrency,
                            output_tokens,
                            port,
                            request_args.request_timeout_secs,
                        )
                    elif scenario == "coding":
                        measured_prompts = [
                            coding_agent_prompt(measured_prompt, concurrency, request_id)
                            for request_id in range(request_count)
                        ]
                        measured, makespan_ms = run_prompt_requests(
                            measured_prompts,
                            case.model_id,
                            concurrency,
                            output_tokens,
                            port,
                            request_args.request_timeout_secs,
                        )
                    else:
                        measured, makespan_ms = harness.run_openai_concurrent_requests(
                            measured_prompt,
                            case.model_id,
                            concurrency,
                            request_count,
                            output_tokens,
                            port,
                            request_args,
                        )
                        prompt_hash = hashlib.sha256(measured_prompt.encode()).hexdigest()
                        for request in measured:
                            request["prompt_sha256"] = prompt_hash
                    expected_summaries = sum("error" not in row for row in measured)
                    events = wait_for_json_events(
                        log_path,
                        SUMMARY_EVENT,
                        event_start + expected_summaries,
                        process,
                    )
                    measured_events = events[event_start:]
                    observations.append(
                        {
                            "scenario": scenario,
                            "concurrency": concurrency,
                            "makespan_ms": makespan_ms,
                            "requests": measured,
                            "summary": summarize_requests(measured, measured_events),
                        }
                    )
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
    suspect_lines = [
        line
        for line in log_path.read_text(errors="replace").splitlines()
        if any(term in line.lower() for term in ("resident_error", "failed to find a memory slot", "panic"))
    ]
    return {
        "round": round_index + 1,
        "version": version,
        "cache": mode,
        "binary": str(binary),
        "config": str(config_path),
        "log": str(log_path),
        "suspect_log_lines": suspect_lines,
        "observations": observations,
    }


def aggregate(cells: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
    for cell in cells:
        for observation in cell["observations"]:
            key = (
                cell["version"],
                cell["cache"],
                observation["scenario"],
                observation["concurrency"],
            )
            buckets.setdefault(key, []).append(observation["summary"])
    rows = []
    for (version, cache, scenario, concurrency), summaries in sorted(buckets.items()):
        def pooled(name: str) -> list[float]:
            return [
                float(value)
                for summary in summaries
                for value in summary.get(name, [])
                if isinstance(value, (int, float))
            ]

        matched = pooled("matched_prefix_tokens")
        suffix = pooled("suffix_prefill_tokens")
        ttft = pooled("ttft_ms")
        tpot = pooled("tpot_ms")

        outputs = sorted({value for row in summaries for value in row["outputs"]})
        outputs_by_prompt: dict[str, set[str]] = {}
        for summary in summaries:
            for prompt_hash, prompt_outputs in summary["outputs_by_prompt"].items():
                outputs_by_prompt.setdefault(prompt_hash, set()).update(prompt_outputs)
        rows.append(
            {
                "version": version,
                "cache": cache,
                "scenario": scenario,
                "concurrency": concurrency,
                "requests": sum(int(row["requests"]) for row in summaries),
                "successful": sum(int(row["successful"]) for row in summaries),
                "cache_hits": sum(int(row["cache_hits"]) for row in summaries),
                "cache_misses": sum(int(row["cache_misses"]) for row in summaries),
                "matched_prefix_tokens_median": statistics.median(matched)
                if matched
                else None,
                "suffix_prefill_tokens_median": statistics.median(suffix)
                if suffix
                else None,
                "ttft_ms_p50": percentile(ttft, 0.50),
                "ttft_ms_p99": percentile(ttft, 0.99),
                "tpot_ms_p50": percentile(tpot, 0.50),
                "outputs": outputs,
                "outputs_by_prompt": {
                    prompt_hash: sorted(prompt_outputs)
                    for prompt_hash, prompt_outputs in sorted(outputs_by_prompt.items())
                },
            }
        )
    by_key = {(row["version"], row["cache"], row["scenario"], row["concurrency"]): row for row in rows}
    for row in rows:
        cold = by_key.get((row["version"], "cold", row["scenario"], row["concurrency"]))
        if (
            row["cache"] == "warm"
            and cold is not None
            and isinstance(cold.get("ttft_ms_p50"), (int, float))
            and isinstance(row.get("ttft_ms_p50"), (int, float))
        ):
            row["cache_lift_ttft_ms"] = cold["ttft_ms_p50"] - row["ttft_ms_p50"]
    return rows


def parity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {(row["version"], row["cache"], row["scenario"], row["concurrency"]): row for row in rows}
    results = []
    for cache in ("cold", "warm"):
        scenarios = sorted({row["scenario"] for row in rows if row["cache"] == cache})
        for scenario in scenarios:
            levels = sorted({row["concurrency"] for row in rows if row["cache"] == cache and row["scenario"] == scenario})
            for concurrency in levels:
                old = indexed.get(("old", cache, scenario, concurrency))
                new = indexed.get(("new", cache, scenario, concurrency))
                if old is not None and new is not None:
                    results.append(
                        {
                            "cache": cache,
                            "scenario": scenario,
                            "concurrency": concurrency,
                            "identical_outputs": old["outputs_by_prompt"]
                            == new["outputs_by_prompt"],
                            "old_outputs_by_prompt": old["outputs_by_prompt"],
                            "new_outputs_by_prompt": new["outputs_by_prompt"],
                        }
                    )
    return results


def cache_preservation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (row["version"], row["cache"], row["scenario"], row["concurrency"]): row
        for row in rows
    }
    results = []
    for version in ("old", "new"):
        scenarios = sorted({row["scenario"] for row in rows if row["version"] == version})
        for scenario in scenarios:
            levels = sorted(
                {
                    row["concurrency"]
                    for row in rows
                    if row["version"] == version and row["scenario"] == scenario
                }
            )
            for concurrency in levels:
                cold = indexed.get((version, "cold", scenario, concurrency))
                warm = indexed.get((version, "warm", scenario, concurrency))
                if cold is not None and warm is not None:
                    results.append(
                        {
                            "version": version,
                            "scenario": scenario,
                            "concurrency": concurrency,
                            "cache_preserves_output": cold["outputs_by_prompt"]
                            == warm["outputs_by_prompt"],
                            "cold_outputs_by_prompt": cold["outputs_by_prompt"],
                            "warm_outputs_by_prompt": warm["outputs_by_prompt"],
                        }
                    )
    return results


def evaluate_gate(case_result: dict[str, Any]) -> dict[str, Any]:
    failures = []
    cells = case_result["cells"]
    rows = case_result["aggregate"]
    payload = case_result["case"]["payload"]
    # Resident KV can borrow the matched portion of a longer native sequence.
    # Recurrent/full-state components are atomic checkpoints. OpenAI chat
    # templating also moves the generation marker when a transcript grows, so
    # neither a divergent user tail nor a later turn is token-prefix-identical
    # to the checkpoint captured for the earlier request. Require exact replay
    # hits; keep the other scenarios visible as measured misses until serving
    # records explicit message-boundary recurrent checkpoints.
    required_hit_scenarios = (
        {"exact", "divergent", "coding"}
        if payload == "resident-kv"
        else {"exact"}
    )
    for cell in cells:
        if cell["suspect_log_lines"]:
            failures.append(
                f"{cell['version']}/{cell['cache']} emitted suspect server log lines"
            )
        for observation in cell["observations"]:
            summary = observation["summary"]
            label = (
                f"{cell['version']}/{cell['cache']}/{observation['scenario']}"
                f"/n{observation['concurrency']}/round-{cell.get('round', 1)}"
            )
            if summary["successful"] != summary["requests"]:
                failures.append(f"{label} did not complete every request")
            if observation["concurrency"] == 1 and any(
                len(prompt_outputs) != 1
                for prompt_outputs in summary["outputs_by_prompt"].values()
            ):
                failures.append(f"{label} produced nondeterministic output for one prompt")
            if (
                cell["cache"] == "warm"
                and observation["scenario"] in required_hit_scenarios
                and observation["concurrency"] == 1
                and summary["cache_hits"] != summary["requests"]
            ):
                failures.append(f"{label} did not report a cache hit for every request")

    preservation = {
        (result["version"], result["scenario"], result["concurrency"]): result
        for result in case_result["cache_output_preservation"]
    }
    for result in case_result["cache_output_preservation"]:
        if (
            result["version"] == "new"
            and result["concurrency"] == 1
            and not result["cache_preserves_output"]
            and preservation.get(("old", result["scenario"], 1), {}).get(
                "cache_preserves_output", True
            )
        ):
            failures.append(
                "NEW cache introduced an N=1 output mismatch absent from OLD for "
                f"{result['scenario']}/n{result['concurrency']}"
            )
    for result in case_result["output_parity"]:
        if (
            result["cache"] == "cold"
            and result["concurrency"] == 1
            and not result["identical_outputs"]
        ):
            failures.append(
                f"OLD/NEW cold output differs for {result['scenario']}/n{result['concurrency']}"
            )

    indexed = {
        (row["version"], row["cache"], row["scenario"], row["concurrency"]): row
        for row in rows
    }
    round_tolerance = max(
        1,
        len(
            {
                cell.get("round", 1)
                for cell in cells
                if cell["version"] == "new" and cell["cache"] == "warm"
            }
        ),
    )
    for scenario in sorted(required_hit_scenarios):
        for concurrency in sorted(
            {
                row["concurrency"]
                for row in rows
                if row["cache"] == "warm"
                and row["scenario"] == scenario
                and row["concurrency"] > 1
            }
        ):
            old = indexed.get(("old", "warm", scenario, concurrency))
            new = indexed.get(("new", "warm", scenario, concurrency))
            if old is None or new is None:
                failures.append(
                    f"missing concurrent warm OLD/NEW pair for {scenario}/n{concurrency}"
                )
            elif new["cache_hits"] + round_tolerance < old["cache_hits"]:
                failures.append(
                    f"NEW warm hit count {new['cache_hits']} regressed beyond the "
                    f"{round_tolerance}-request round tolerance from OLD "
                    f"{old['cache_hits']} for {scenario}/n{concurrency}"
                )
    for concurrency in sorted(
        {
            row["concurrency"]
            for row in rows
            if row["cache"] == "warm" and row["scenario"] == "divergent"
        }
    ):
        old = indexed.get(("old", "warm", "divergent", concurrency))
        new = indexed.get(("new", "warm", "divergent", concurrency))
        if old is None or new is None:
            failures.append(f"missing divergent OLD/NEW pair at concurrency {concurrency}")
            continue
        old_suffix = old.get("suffix_prefill_tokens_median")
        new_suffix = new.get("suffix_prefill_tokens_median")
        if not isinstance(old_suffix, (int, float)) or not isinstance(
            new_suffix, (int, float)
        ):
            failures.append(f"missing divergent suffix telemetry at concurrency {concurrency}")
        elif new_suffix > old_suffix:
            failures.append(
                f"NEW divergent suffix prefill {new_suffix} exceeded OLD {old_suffix} "
                f"at concurrency {concurrency}"
            )
    return {"passed": not failures, "failures": failures}


def markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Version | Cache | Scenario | N | Requests | Hits | Matched tokens | Suffix prefill | p50 TTFT ms | Cache lift ms |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        fmt = lambda key: f"{row[key]:.1f}" if isinstance(row.get(key), (int, float)) else "n/a"
        lines.append(
            f"| {row['version'].upper()} | {row['cache']} | {row['scenario']} | {row['concurrency']} "
            f"| {row['requests']} | {row['cache_hits']} | {fmt('matched_prefix_tokens_median')} "
            f"| {fmt('suffix_prefill_tokens_median')} | {fmt('ttft_ms_p50')} "
            f"| {fmt('cache_lift_ttft_ms')} |"
        )
    return "\n".join(lines) + "\n"


def svg(rows: list[dict[str, Any]]) -> str:
    warm = [row for row in rows if row["cache"] == "warm" and row["scenario"] == "divergent"]
    width, height = 900, 420
    margin = 70
    values = [float(row["matched_prefix_tokens_median"] or 0) for row in warm]
    ceiling = max(values + [1.0])
    bar_width = max(24, int((width - 2 * margin) / max(len(warm), 1) * 0.65))
    gap = (width - 2 * margin) / max(len(warm), 1)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="450" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">Divergent-prefix tokens reused</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#222"/>',
    ]
    colors = {"old": "#8a94a6", "new": "#2f80ed"}
    for index, row in enumerate(warm):
        value = float(row["matched_prefix_tokens_median"] or 0)
        bar_height = (height - 2 * margin) * value / ceiling
        x = margin + index * gap + (gap - bar_width) / 2
        y = height - margin - bar_height
        label = f"{row['version'].upper()} N={row['concurrency']}"
        parts.extend(
            [
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width}" height="{bar_height:.1f}" fill="{colors[row["version"]]}"/>',
                f'<text x="{x + bar_width/2:.1f}" y="{y - 7:.1f}" text-anchor="middle" font-family="sans-serif" font-size="12">{value:.0f}</text>',
                f'<text x="{x + bar_width/2:.1f}" y="{height-margin+20}" text-anchor="middle" font-family="sans-serif" font-size="11">{label}</text>',
            ]
        )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def parse_levels(value: str) -> list[int]:
    levels = sorted({int(raw) for raw in value.split(",") if raw.strip()})
    if not levels or levels[0] <= 0:
        raise ValueError("concurrency must contain positive integers")
    return levels


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-file", type=Path, required=True)
    parser.add_argument("--old-bin", type=Path, required=True)
    parser.add_argument("--new-bin", type=Path, required=True)
    parser.add_argument("--old-commit", required=True)
    parser.add_argument("--new-commit", required=True)
    parser.add_argument("--native-build", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--concurrency", default="1,2,4")
    parser.add_argument("--output-tokens", type=int, default=8)
    parser.add_argument("--prefix-blocks", type=int, default=192)
    parser.add_argument("--ctx-size", type=int, default=32768)
    parser.add_argument("--lanes", type=int, default=4)
    parser.add_argument(
        "--n-gpu-layers",
        type=int,
        default=999,
        help="model layers to offload to the selected accelerator (default: all)",
    )
    parser.add_argument(
        "--require-gates",
        action="store_true",
        help="exit non-zero unless correctness, hit, and divergent-prefix gates pass",
    )
    args = parser.parse_args()
    if args.rounds <= 0 or args.requests <= 0:
        parser.error("rounds and requests must be positive")
    binaries = {"old": args.old_bin.resolve(), "new": args.new_bin.resolve()}
    for name, binary in binaries.items():
        if not binary.is_file():
            parser.error(f"{name} binary not found: {binary}")
    levels = parse_levels(args.concurrency)
    if max(levels) > args.lanes:
        parser.error("maximum concurrency cannot exceed lanes")
    cases = read_cases(args.case_file)
    harness = load_production_harness()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "old": {"commit": args.old_commit, "binary": str(binaries["old"]), "sha256": sha256(binaries["old"])},
        "new": {"commit": args.new_commit, "binary": str(binaries["new"]), "sha256": sha256(binaries["new"])},
        "rounds": args.rounds,
        "requests_per_level_per_round": args.requests,
        "concurrency": levels,
        "prefix_blocks": args.prefix_blocks,
        "ctx_size": args.ctx_size,
        "lanes": args.lanes,
        "n_gpu_layers": args.n_gpu_layers,
    }
    all_results = []
    for case in cases:
        case_dir = args.output_dir / case.key
        cells = []
        for round_index in range(args.rounds):
            versions = ["old", "new"] if round_index % 2 == 0 else ["new", "old"]
            for version in versions:
                for cache_enabled in (False, True):
                    print(f"==> {case.key}: round={round_index + 1} {version} cache={cache_enabled}", flush=True)
                    cells.append(
                        run_server_cell(
                            harness,
                            version,
                            binaries[version],
                            case,
                            cache_enabled,
                            round_index,
                            levels,
                            args.requests,
                            args.output_tokens,
                            args.prefix_blocks,
                            args.ctx_size,
                            args.lanes,
                            args.n_gpu_layers,
                            args.native_build.resolve(),
                            case_dir,
                        )
                    )
        rows = aggregate(cells)
        case_result = {
            "case": {
                "key": case.key,
                "family": case.family,
                "model_id": case.model_id,
                "model_path": str(case.model_path),
                "model_sha256": harness.model_sha256(case.model_path),
                "layer_end": case.layer_end,
                "payload": case.payload,
            },
            "cells": cells,
            "aggregate": rows,
            "output_parity": parity(rows),
            "cache_output_preservation": cache_preservation(rows),
        }
        case_result["gate"] = evaluate_gate(case_result)
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "comparison.json").write_text(json.dumps(case_result, indent=2) + "\n")
        (case_dir / "table.md").write_text(markdown(rows))
        (case_dir / "divergent-prefix.svg").write_text(svg(rows))
        all_results.append(case_result)
    result = {"metadata": metadata, "cases": all_results}
    (args.output_dir / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(args.output_dir / "comparison.json")
    if args.require_gates:
        failures = [
            f"{case_result['case']['key']}: {failure}"
            for case_result in all_results
            for failure in case_result["gate"]["failures"]
        ]
        if failures:
            print("benchmark gate failed:", file=sys.stderr)
            for failure in failures:
                print(f"- {failure}", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
