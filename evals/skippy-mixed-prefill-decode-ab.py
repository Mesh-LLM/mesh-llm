#!/usr/bin/env python3
"""Benchmark mixed prefill/decode scheduling against its exact serial base."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
import random
import socket
import statistics
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(round((len(ordered) - 1) * quantile), len(ordered) - 1)
    return ordered[index]


def delta_percent(before: float, after: float) -> float | None:
    if before == 0:
        return None
    return (after - before) / before * 100.0


def wait_openai(port: int, process: subprocess.Popen[str], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error = "no attempts made"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with status {process.returncode}")
        connection = None
        try:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
            connection.request("GET", "/v1/models")
            response = connection.getresponse()
            response.read()
            if response.status == 200:
                return
            last_error = f"HTTP {response.status}"
        except OSError as error:
            last_error = str(error)
        finally:
            if connection is not None:
                connection.close()
        time.sleep(0.25)
    raise TimeoutError(f"timed out waiting for OpenAI endpoint: {last_error}")


def stop(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def stable_prompt(blocks: int, request_index: int, role: str) -> str:
    rows = [
        f"context-block-{index:04d}: src/module_{index % 37}.rs owns invariant "
        f"{index}; preserve the repository contract exactly."
        for index in range(blocks)
    ]
    task = (
        "Continue a numbered implementation checklist with one item per line."
        if role == "anchor"
        else f"Name the owner of invariant {request_index % blocks}."
    )
    return (
        "You are a deterministic coding assistant. Read this repository context.\n"
        + "\n".join(rows)
        + f"\nRequest {request_index}: {task}"
    )


def read_prompt_manifest(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("prompts"), list):
        raise ValueError("prompt manifest must be an object with a prompts list")
    prompts = []
    for index, item in enumerate(document["prompts"]):
        if not isinstance(item, dict):
            raise ValueError(f"prompt manifest item {index} must be an object")
        prompt = item.get("prompt")
        family = item.get("family")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError(f"prompt manifest item {index} needs a nonempty prompt")
        if not isinstance(family, str) or not family:
            raise ValueError(f"prompt manifest item {index} needs a nonempty family")
        prompts.append(dict(item))
    if not prompts:
        raise ValueError("prompt manifest must contain at least one prompt")
    metadata = document.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("prompt manifest metadata must be an object")
    return prompts, metadata


def write_config(args: argparse.Namespace, path: Path, port: int) -> None:
    config = {
        "run_id": "skippy-mixed-prefill-decode-ab",
        "topology_id": "skippy-mixed-prefill-decode-ab-local",
        "model_id": args.model_id,
        "model_path": str(args.model.resolve()),
        "source_model_sha256": args.model_sha256,
        "stage_id": "stage-0",
        "stage_index": 0,
        "layer_start": 0,
        "layer_end": args.layer_end,
        "ctx_size": args.ctx_size,
        "lane_count": args.lanes,
        "n_batch": args.n_batch,
        "n_ubatch": args.n_ubatch,
        "n_gpu_layers": args.n_gpu_layers,
        "cache_type_k": "f16",
        "cache_type_v": "f16",
        "filter_tensors_on_load": True,
        "load_mode": "runtime-slice",
        "bind_addr": f"127.0.0.1:{port}",
        "upstream": None,
        "downstream": None,
    }
    path.write_text(json.dumps(config, indent=2) + "\n")


def run_request(
    port: int,
    model_id: str,
    role: str,
    request_index: int,
    prompt: str,
    prompt_provenance: dict[str, Any],
    output_tokens: int,
    suppressed_token_ids: tuple[int, ...],
    delay_ms: float,
    epoch: float,
    timeout: float,
) -> dict[str, Any]:
    delay_seconds = delay_ms / 1000.0
    remaining = epoch + delay_seconds - time.monotonic()
    if remaining > 0:
        time.sleep(remaining)
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": output_tokens,
        "temperature": 0,
        "seed": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if suppressed_token_ids:
        payload["logit_bias"] = {
            str(token_id): -100 for token_id in suppressed_token_ids
        }
    started = time.monotonic()
    first_content = None
    previous_content = None
    gaps_ms: list[float] = []
    content: list[str] = []
    completion_tokens = 0
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
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
                "role": role,
                "request_index": request_index,
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
            if isinstance(usage, dict) and isinstance(usage.get("completion_tokens"), int):
                completion_tokens = usage["completion_tokens"]
            choices = event.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            delta = choices[0].get("delta")
            if not isinstance(delta, dict):
                continue
            text = delta.get("content") or delta.get("reasoning_content")
            if not text:
                continue
            arrived = time.monotonic()
            if first_content is None:
                first_content = arrived
            if previous_content is not None:
                gaps_ms.append((arrived - previous_content) * 1000.0)
            previous_content = arrived
            content.append(text)
        completed = time.monotonic()
        if first_content is None:
            return {
                "role": role,
                "request_index": request_index,
                "error": "stream completed without generated content",
            }
        output = "".join(content)
        return {
            "role": role,
            "request_index": request_index,
            "prompt_provenance": prompt_provenance,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "content": output,
            "content_sha256": hashlib.sha256(output.encode()).hexdigest(),
            "completion_tokens": completion_tokens or len(content),
            "ttft_ms": (first_content - started) * 1000.0,
            "elapsed_ms": (completed - started) * 1000.0,
            "content_gaps_ms": gaps_ms,
        }
    except Exception as error:  # noqa: BLE001 - retain failures in the artifact.
        return {"role": role, "request_index": request_index, "error": str(error)}
    finally:
        connection.close()


def scheduler_events(path: Path) -> list[dict[str, Any]]:
    events = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") == "stage.scheduler_iteration":
            events.append(event.get("attributes", {}))
    return events


def summarize_requests(
    requests: list[dict[str, Any]],
    wall_ms: float,
    events: list[dict[str, Any]],
    n_batch: int,
) -> dict[str, Any]:
    successful = [request for request in requests if "error" not in request]
    anchors = [request for request in successful if request["role"] == "anchor"]
    prefills = [request for request in successful if request["role"] == "prefill"]
    anchor_gaps = [gap for request in anchors for gap in request["content_gaps_ms"]]
    completion_tokens = sum(int(request["completion_tokens"]) for request in successful)
    token_counts = [
        int(event.get("skippy.scheduler.prefill_tokens", 0))
        + int(event.get("skippy.scheduler.recompute_tokens", 0))
        + int(event.get("skippy.scheduler.decode_tokens", 0))
        for event in events
    ]
    mixed = [
        event
        for event in events
        if int(event.get("skippy.scheduler.decode_tokens", 0)) > 0
        and (
            int(event.get("skippy.scheduler.prefill_tokens", 0))
            + int(event.get("skippy.scheduler.recompute_tokens", 0))
            > 0
        )
    ]
    prefill_only = [
        event
        for event in events
        if int(event.get("skippy.scheduler.decode_tokens", 0)) == 0
        and (
            int(event.get("skippy.scheduler.prefill_tokens", 0))
            + int(event.get("skippy.scheduler.recompute_tokens", 0))
            > 0
        )
    ]
    decode_only = [
        event
        for event in events
        if int(event.get("skippy.scheduler.decode_tokens", 0)) > 0
        and int(event.get("skippy.scheduler.prefill_tokens", 0)) == 0
        and int(event.get("skippy.scheduler.recompute_tokens", 0)) == 0
    ]

    def request_percentile(
        rows: list[dict[str, Any]], key: str, q: float
    ) -> float | None:
        return percentile([float(row[key]) for row in rows], q)

    return {
        "requests": len(requests),
        "successful_requests": len(successful),
        "errors": len(requests) - len(successful),
        "completion_tokens": completion_tokens,
        "makespan_ms": wall_ms,
        "output_tokens_per_second": completion_tokens / (wall_ms / 1000.0),
        "ttft_ms_p50": request_percentile(successful, "ttft_ms", 0.50),
        "ttft_ms_p95": request_percentile(successful, "ttft_ms", 0.95),
        "anchor_ttft_ms_p50": request_percentile(anchors, "ttft_ms", 0.50),
        "anchor_ttft_ms_p95": request_percentile(anchors, "ttft_ms", 0.95),
        "anchor_gap_ms_p50": percentile(anchor_gaps, 0.50),
        "anchor_gap_ms_p95": percentile(anchor_gaps, 0.95),
        "prefill_ttft_ms_p50": request_percentile(prefills, "ttft_ms", 0.50),
        "prefill_ttft_ms_p95": request_percentile(prefills, "ttft_ms", 0.95),
        "scheduler_iterations": len(events),
        "mixed_iterations": len(mixed),
        "prefill_only_iterations": len(prefill_only),
        "decode_only_iterations": len(decode_only),
        "mean_batch_tokens": statistics.mean(token_counts) if token_counts else 0.0,
        "mean_token_occupancy": (
            statistics.mean(token_counts) / n_batch if token_counts else 0.0
        ),
    }


def launch_cell(
    args: argparse.Namespace,
    version: str,
    binary: Path,
    native_build: Path,
    round_index: int,
) -> dict[str, Any]:
    cell_dir = args.output_dir / f"round-{round_index + 1}-{version}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    binary_port, openai_port = free_port(), free_port()
    config_path = cell_dir / "stage-0.json"
    log_path = cell_dir / "server.log"
    write_config(args, config_path, binary_port)
    command = [
        str(binary),
        "serve-binary",
        "--config",
        str(config_path),
        "--activation-width",
        str(args.activation_width),
        "--activation-wire-dtype",
        args.activation_wire_dtype,
        "--max-inflight",
        str(args.lanes),
        "--telemetry-level",
        "debug",
        "--openai-bind-addr",
        f"127.0.0.1:{openai_port}",
        "--openai-generation-concurrency",
        str(args.lanes),
        "--openai-default-max-tokens",
        str(max(args.anchor_output_tokens, args.prefill_output_tokens)),
        "--openai-prefill-chunk-policy",
        "adaptive-ramp",
        "--openai-prefill-chunk-size",
        str(args.n_ubatch),
        "--openai-prefill-adaptive-start",
        str(args.n_ubatch),
        "--openai-prefill-adaptive-step",
        str(args.n_ubatch),
        "--openai-prefill-adaptive-max",
        str(args.n_ubatch),
    ]
    if version == "new" or not args.adaptive_target_new_only:
        command.extend(
            ["--openai-prefill-adaptive-target-ms", str(args.adaptive_target_ms)]
        )
    environment = os.environ.copy()
    environment["LLAMA_STAGE_BUILD_DIR"] = str(native_build.resolve())
    environment["SKIPPY_TELEMETRY_STDERR"] = "1"
    environment["SKIPPY_NATIVE_MTP_GREEDY_SAMPLING_FASTPATH"] = "1"
    process = None
    with log_path.open("w") as log:
        try:
            process = subprocess.Popen(
                command,
                cwd=Path(__file__).resolve().parents[1],
                env=environment,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            wait_openai(openai_port, process, args.startup_timeout_secs)
            warmup = run_request(
                openai_port,
                args.model_id,
                "prefill",
                -1,
                stable_prompt(16, -1, "prefill"),
                {"family": "synthetic-warmup"},
                4,
                args.suppress_token_id,
                0.0,
                time.monotonic(),
                args.request_timeout_secs,
            )
            if "error" in warmup:
                raise RuntimeError(f"warmup request failed: {warmup['error']}")
            warmup_iteration_count = len(scheduler_events(log_path))
            specs = []
            for index in range(args.anchors):
                specs.append(
                    (
                        "anchor",
                        index,
                        stable_prompt(args.anchor_prompt_blocks, index, "anchor"),
                        {"family": "synthetic-anchor"},
                        args.anchor_output_tokens,
                        0.0,
                    )
                )
            manifest_start = round_index * args.prefills
            round_prefills = (
                args.prefill_prompts[manifest_start : manifest_start + args.prefills]
                if args.prefill_prompts is not None
                else None
            )
            for index in range(args.prefills):
                prompt_record = (
                    round_prefills[index]
                    if round_prefills is not None
                    else {
                        "family": "synthetic-prefill",
                        "prompt": stable_prompt(
                            args.prefill_prompt_blocks,
                            args.anchors + index,
                            "prefill",
                        ),
                    }
                )
                specs.append(
                    (
                        "prefill",
                        args.anchors + index,
                        prompt_record["prompt"],
                        {
                            key: value
                            for key, value in prompt_record.items()
                            if key != "prompt"
                        },
                        args.prefill_output_tokens,
                        args.prefill_delay_ms + index * args.prefill_stagger_ms,
                    )
                )
            epoch = time.monotonic() + 0.25
            with ThreadPoolExecutor(max_workers=len(specs)) as executor:
                futures = [
                    executor.submit(
                        run_request,
                        openai_port,
                        args.model_id,
                        role,
                        request_index,
                        prompt,
                        provenance,
                        output_tokens,
                        args.suppress_token_id,
                        delay_ms,
                        epoch,
                        args.request_timeout_secs,
                    )
                    for role, request_index, prompt, provenance, output_tokens, delay_ms in specs
                ]
                requests = [future.result() for future in futures]
            wall_ms = (time.monotonic() - epoch) * 1000.0
        finally:
            stop(process)
    events = scheduler_events(log_path)[warmup_iteration_count:]
    return {
        "round": round_index + 1,
        "version": version,
        "binary": str(binary),
        "binary_sha256": sha256(binary),
        "native_build": str(native_build),
        "config": str(config_path),
        "log": str(log_path),
        "requests": requests,
        "summary": summarize_requests(requests, wall_ms, events, args.n_batch),
    }


METRICS = (
    "makespan_ms",
    "output_tokens_per_second",
    "ttft_ms_p50",
    "ttft_ms_p95",
    "anchor_ttft_ms_p50",
    "anchor_ttft_ms_p95",
    "anchor_gap_ms_p50",
    "anchor_gap_ms_p95",
    "prefill_ttft_ms_p50",
    "prefill_ttft_ms_p95",
    "scheduler_iterations",
    "mixed_iterations",
    "mean_batch_tokens",
    "mean_token_occupancy",
)


def aggregate(cells: list[dict[str, Any]], version: str) -> dict[str, float]:
    summaries = [cell["summary"] for cell in cells if cell["version"] == version]
    keys = ("successful_requests", "errors", "completion_tokens", *METRICS)
    return {
        key: statistics.median(float(row[key]) for row in summaries)
        for key in keys
        if all(row[key] is not None for row in summaries)
    }


def paired_intervals(cells: list[dict[str, Any]], rounds: int) -> dict[str, Any]:
    result = {}
    rng = random.Random(0)
    for metric in METRICS:
        deltas = []
        for round_index in range(1, rounds + 1):
            old = next(
                cell
                for cell in cells
                if cell["round"] == round_index and cell["version"] == "old"
            )["summary"].get(metric)
            new = next(
                cell
                for cell in cells
                if cell["round"] == round_index and cell["version"] == "new"
            )["summary"].get(metric)
            if old in (None, 0) or new is None:
                continue
            deltas.append(delta_percent(float(old), float(new)))
        if not deltas:
            continue
        bootstrapped = [
            statistics.median(rng.choice(deltas) for _ in deltas) for _ in range(10_000)
        ]
        result[metric] = {
            "round_deltas": deltas,
            "median": statistics.median(deltas),
            "ci95": [percentile(bootstrapped, 0.025), percentile(bootstrapped, 0.975)],
        }
    return result


def parity(cells: list[dict[str, Any]], rounds: int) -> dict[str, Any]:
    mismatches = []
    comparable = 0
    for round_index in range(1, rounds + 1):
        old = next(
            cell
            for cell in cells
            if cell["round"] == round_index and cell["version"] == "old"
        )
        new = next(
            cell
            for cell in cells
            if cell["round"] == round_index and cell["version"] == "new"
        )
        old_requests = {request["request_index"]: request for request in old["requests"]}
        new_requests = {request["request_index"]: request for request in new["requests"]}
        for request_index in sorted(old_requests.keys() & new_requests.keys()):
            old_request = old_requests[request_index]
            new_request = new_requests[request_index]
            if "error" in old_request or "error" in new_request:
                continue
            comparable += 1
            if old_request["content"] != new_request["content"]:
                mismatches.append({"round": round_index, "request": request_index})
    return {
        "comparable_requests": comparable,
        "exact_matches": comparable - len(mismatches),
        "mismatches": mismatches,
    }


def markdown(
    before: dict[str, float],
    after: dict[str, float],
    intervals: dict[str, Any],
    parity_result: dict[str, Any],
) -> str:
    rows = [
        ("Makespan ms", "makespan_ms"),
        ("Output tok/s", "output_tokens_per_second"),
        ("Anchor TTFT p50 ms", "anchor_ttft_ms_p50"),
        ("Anchor TTFT p95 ms", "anchor_ttft_ms_p95"),
        ("Anchor stream gap p50 ms", "anchor_gap_ms_p50"),
        ("Anchor stream gap p95 ms", "anchor_gap_ms_p95"),
        ("Prefill TTFT p50 ms", "prefill_ttft_ms_p50"),
        ("Prefill TTFT p95 ms", "prefill_ttft_ms_p95"),
        ("Scheduler iterations", "scheduler_iterations"),
        ("Mixed iterations", "mixed_iterations"),
        ("Mean batch tokens", "mean_batch_tokens"),
        ("Token-budget occupancy", "mean_token_occupancy"),
    ]
    chart_keys = [
        "makespan_ms",
        "output_tokens_per_second",
        "anchor_ttft_ms_p95",
        "anchor_gap_ms_p95",
        "prefill_ttft_ms_p95",
    ]
    chart_values = [100.0 * after[key] / before[key] for key in chart_keys]
    chart_upper = max(120.0, max(chart_values) * 1.1)
    lines = [
        "```mermaid",
        "xychart-beta",
        '    title "Mixed scheduling (base = 100)"',
        '    x-axis ["Makespan", "Throughput", "Anchor-TTFT-p95", '
        '"Anchor-gap-p95", "Prefill-TTFT-p95"]',
        f'    y-axis "Percent of base" 0 --> {chart_upper:.0f}',
        "    bar [100, 100, 100, 100, 100]",
        "    bar [" + ", ".join(f"{value:.1f}" for value in chart_values) + "]",
        "```",
        "",
        "| Metric | Before | After | Delta | Paired 95% CI |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, key in rows:
        delta = delta_percent(before[key], after[key])
        delta_text = f"{delta:+.1f}%" if delta is not None else "n/a"
        interval = intervals.get(key, {}).get("ci95")
        interval_text = (
            f"[{interval[0]:+.1f}%, {interval[1]:+.1f}%]" if interval else "n/a"
        )
        lines.append(
            f"| {label} | {before[key]:.3f} | {after[key]:.3f} | {delta_text} | {interval_text} |"
        )
    lines.extend(
        [
            "",
            f"Output parity: **{parity_result['exact_matches']}/{parity_result['comparable_requests']} exact matches**.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-binary", type=Path, required=True)
    parser.add_argument("--new-binary", type=Path, required=True)
    parser.add_argument("--old-native-build", type=Path, required=True)
    parser.add_argument("--new-native-build", type=Path, required=True)
    parser.add_argument("--old-commit", required=True)
    parser.add_argument("--new-commit", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--anchors", type=int, default=4)
    parser.add_argument("--prefills", type=int, default=8)
    parser.add_argument("--anchor-prompt-blocks", type=int, default=8)
    parser.add_argument("--prefill-prompt-blocks", type=int, default=256)
    parser.add_argument(
        "--prefill-prompt-manifest",
        type=Path,
        help="JSON object with enough unique trace prompts for every round",
    )
    parser.add_argument("--anchor-output-tokens", type=int, default=128)
    parser.add_argument("--prefill-output-tokens", type=int, default=8)
    parser.add_argument(
        "--suppress-token-id",
        action="append",
        type=int,
        default=[],
        help="token ID to suppress with logit bias; repeat for multiple IDs",
    )
    parser.add_argument("--prefill-delay-ms", type=float, default=100.0)
    parser.add_argument("--prefill-stagger-ms", type=float, default=5.0)
    parser.add_argument("--ctx-size", type=int, default=65536)
    parser.add_argument("--lanes", type=int, default=12)
    parser.add_argument("--n-batch", type=int, default=1024)
    parser.add_argument("--n-ubatch", type=int, default=256)
    parser.add_argument("--layer-end", type=int, required=True)
    parser.add_argument("--activation-width", type=int, required=True)
    parser.add_argument("--activation-wire-dtype", default="f16")
    parser.add_argument("--n-gpu-layers", type=int, default=999)
    parser.add_argument("--adaptive-target-ms", type=float, default=100.0)
    parser.add_argument(
        "--adaptive-target-new-only",
        action="store_true",
        help="pass adaptive-target-ms only to NEW when OLD predates that CLI option",
    )
    parser.add_argument("--startup-timeout-secs", type=float, default=900)
    parser.add_argument("--request-timeout-secs", type=float, default=1800)
    args = parser.parse_args()
    positive = (
        "rounds",
        "anchors",
        "prefills",
        "anchor_prompt_blocks",
        "prefill_prompt_blocks",
        "anchor_output_tokens",
        "prefill_output_tokens",
        "ctx_size",
        "lanes",
        "n_batch",
        "n_ubatch",
        "layer_end",
        "activation_width",
    )
    if any(getattr(args, name) <= 0 for name in positive):
        parser.error(
            "round, workload, batch, model, and activation values must be positive"
        )
    if args.anchors + args.prefills > args.lanes:
        parser.error("anchors plus prefills must not exceed lanes")
    if args.n_ubatch > args.n_batch:
        parser.error("n-ubatch must not exceed n-batch")
    finite_nonnegative = ("prefill_delay_ms", "prefill_stagger_ms")
    if any(
        not math.isfinite(getattr(args, name)) or getattr(args, name) < 0
        for name in finite_nonnegative
    ):
        parser.error("prefill delay and stagger must be finite and non-negative")
    if not math.isfinite(args.adaptive_target_ms) or args.adaptive_target_ms <= 0:
        parser.error("adaptive-target-ms must be finite and positive")
    if any(token_id < 0 for token_id in args.suppress_token_id):
        parser.error("suppress-token-id must be non-negative")
    args.suppress_token_id = tuple(args.suppress_token_id)
    for name in ("old_binary", "new_binary", "model"):
        if not getattr(args, name).is_file():
            parser.error(f"{name.replace('_', '-')} not found: {getattr(args, name)}")
    for name in ("old_native_build", "new_native_build"):
        if not getattr(args, name).is_dir():
            parser.error(f"{name.replace('_', '-')} not found: {getattr(args, name)}")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model = args.model.resolve()
    args.model_sha256 = sha256(args.model)
    args.prefill_prompts = None
    args.prefill_prompt_manifest_metadata = {}
    if args.prefill_prompt_manifest is not None:
        args.prefill_prompt_manifest = args.prefill_prompt_manifest.resolve()
        if not args.prefill_prompt_manifest.is_file():
            parser.error(
                f"prefill-prompt-manifest not found: {args.prefill_prompt_manifest}"
            )
        try:
            args.prefill_prompts, args.prefill_prompt_manifest_metadata = (
                read_prompt_manifest(args.prefill_prompt_manifest)
            )
        except (json.JSONDecodeError, OSError, ValueError) as error:
            parser.error(f"invalid prefill prompt manifest: {error}")
        required_prompts = args.rounds * args.prefills
        if len(args.prefill_prompts) != required_prompts:
            parser.error(
                "prefill prompt manifest must contain exactly "
                f"rounds * prefills = {required_prompts} prompts; "
                f"found {len(args.prefill_prompts)}"
            )
    return args


def main() -> int:
    args = parse_args()
    versions = {
        "old": (args.old_binary.resolve(), args.old_native_build.resolve()),
        "new": (args.new_binary.resolve(), args.new_native_build.resolve()),
    }
    cells = []
    for round_index in range(args.rounds):
        order = ("old", "new") if round_index % 2 == 0 else ("new", "old")
        for version in order:
            print(f"==> round {round_index + 1}/{args.rounds}: {version}", flush=True)
            binary, native_build = versions[version]
            cells.append(launch_cell(args, version, binary, native_build, round_index))
    before = aggregate(cells, "old")
    after = aggregate(cells, "new")
    intervals = paired_intervals(cells, args.rounds)
    parity_result = parity(cells, args.rounds)
    result = {
        "metadata": {
            "old": {
                "commit": args.old_commit,
                "binary": str(versions["old"][0]),
                "sha256": sha256(versions["old"][0]),
                "native_build": str(versions["old"][1]),
            },
            "new": {
                "commit": args.new_commit,
                "binary": str(versions["new"][0]),
                "sha256": sha256(versions["new"][0]),
                "native_build": str(versions["new"][1]),
            },
            "model_id": args.model_id,
            "model_path": str(args.model),
            "model_sha256": args.model_sha256,
            "rounds": args.rounds,
            "shape": {
                "anchors": args.anchors,
                "prefills": args.prefills,
                "anchor_prompt_blocks": args.anchor_prompt_blocks,
                "prefill_prompt_blocks": args.prefill_prompt_blocks,
                "prefill_prompt_manifest": (
                    {
                        "path": str(args.prefill_prompt_manifest),
                        "sha256": sha256(args.prefill_prompt_manifest),
                        "metadata": args.prefill_prompt_manifest_metadata,
                    }
                    if args.prefill_prompt_manifest is not None
                    else None
                ),
                "anchor_output_tokens": args.anchor_output_tokens,
                "prefill_output_tokens": args.prefill_output_tokens,
                "prefill_delay_ms": args.prefill_delay_ms,
                "prefill_stagger_ms": args.prefill_stagger_ms,
                "suppressed_token_ids": list(args.suppress_token_id),
            },
            "runtime": {
                "ctx_size": args.ctx_size,
                "lanes": args.lanes,
                "n_batch": args.n_batch,
                "n_ubatch": args.n_ubatch,
                "layer_end": args.layer_end,
                "activation_width": args.activation_width,
                "adaptive_target_ms": args.adaptive_target_ms,
                "adaptive_target_new_only": args.adaptive_target_new_only,
            },
        },
        "cells": cells,
        "aggregate": {
            "old": before,
            "new": after,
            "delta_percent": {
                key: delta_percent(before[key], after[key])
                for key in before.keys() & after.keys()
                if key not in {"successful_requests", "errors", "completion_tokens"}
            },
        },
        "paired_delta_percent": intervals,
        "output_parity": parity_result,
    }
    comparison_path = args.output_dir / "comparison.json"
    report_path = args.output_dir / "report.md"
    comparison_path.write_text(json.dumps(result, indent=2) + "\n")
    report_path.write_text(markdown(before, after, intervals, parity_result))
    print(report_path.read_text(), end="")
    expected_successes = args.rounds * (args.anchors + args.prefills)
    old_successes = sum(
        cell["summary"]["successful_requests"]
        for cell in cells
        if cell["version"] == "old"
    )
    new_successes = sum(
        cell["summary"]["successful_requests"]
        for cell in cells
        if cell["version"] == "new"
    )
    if old_successes != expected_successes or new_successes != expected_successes:
        return 1
    if parity_result["exact_matches"] != parity_result["comparable_requests"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
