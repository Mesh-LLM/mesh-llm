#!/usr/bin/env python3
"""Benchmark slowest-stage adaptive prefill against an exact OLD/NEW pair.

The runner launches a balanced two-stage chain on one host, discards one
calibration request per cell, and measures cache-disabled long-prefill requests.
It alternates launch order, retains both stage logs and every streamed response,
checks output parity, and emits JSON plus a PR-ready Markdown chart and table.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import socket
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
PREFILL_EVENT = "stage.openai_prefill"
CALIBRATION_EVENT = "stage.openai_prefill_calibration"


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


def delta_percent(before: float | None, after: float | None) -> float | None:
    if before in (None, 0) or after is None:
        return None
    return (after - before) / before * 100.0


def wait_tcp(port: int, process: subprocess.Popen[str], timeout: float, name: str) -> None:
    deadline = time.monotonic() + timeout
    last_error = "no attempts made"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"{name} exited with status {process.returncode}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1) as stream:
                stream.settimeout(1)
                ready = stream.recv(4)
                if ready == (0x5352_4459).to_bytes(4, "little"):
                    return
                last_error = f"unexpected ready bytes: {ready.hex()}"
        except OSError as error:
            last_error = str(error)
            time.sleep(0.25)
    raise TimeoutError(f"timed out waiting for {name}: {last_error}")


def wait_openai(port: int, process: subprocess.Popen[str], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error = "no attempts made"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"stage 0 exited with status {process.returncode}")
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
    raise TimeoutError(f"timed out waiting for stage 0 OpenAI: {last_error}")


def stop(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def stable_prompt(blocks: int, request_index: int) -> str:
    rows = [
        f"context-block-{index:04d}: src/module_{index % 37}.rs owns invariant "
        f"{index}; preserve the repository contract exactly."
        for index in range(blocks)
    ]
    return (
        "You are a deterministic coding assistant. Read this repository context.\n"
        + "\n".join(rows)
        + f"\nTask {request_index}: name the owner of invariant {request_index % blocks}."
    )


def stage_config(
    args: argparse.Namespace,
    stage_index: int,
    layer_start: int,
    layer_end: int,
    bind_port: int,
    peer_port: int,
) -> dict[str, Any]:
    upstream = None
    downstream = None
    if stage_index == 0:
        downstream = {
            "stage_id": "stage-1",
            "stage_index": 1,
            "endpoint": f"tcp://127.0.0.1:{peer_port}",
        }
    else:
        upstream = {
            "stage_id": "stage-0",
            "stage_index": 0,
            "endpoint": f"tcp://127.0.0.1:{peer_port}",
        }
    return {
        "run_id": "skippy-adaptive-prefill-ab",
        "topology_id": "skippy-adaptive-prefill-ab-two-stage",
        "model_id": args.model_id,
        "model_path": str(args.model.resolve()),
        "source_model_sha256": args.model_sha256,
        "stage_id": f"stage-{stage_index}",
        "stage_index": stage_index,
        "layer_start": layer_start,
        "layer_end": layer_end,
        "ctx_size": args.ctx_size,
        "lane_count": 1,
        "n_gpu_layers": args.n_gpu_layers,
        "cache_type_k": "f16",
        "cache_type_v": "f16",
        "filter_tensors_on_load": True,
        "load_mode": "runtime-slice",
        "bind_addr": f"127.0.0.1:{bind_port}",
        "upstream": upstream,
        "downstream": downstream,
    }


def write_configs(
    args: argparse.Namespace,
    cell_dir: Path,
    stage0_port: int,
    stage1_port: int,
) -> tuple[Path, Path]:
    stage0_path = cell_dir / "stage-0.json"
    stage1_path = cell_dir / "stage-1.json"
    stage0 = stage_config(args, 0, 0, args.split_layer, stage0_port, stage1_port)
    stage1 = stage_config(
        args,
        1,
        args.split_layer,
        args.layer_end,
        stage1_port,
        stage0_port,
    )
    stage0_path.write_text(json.dumps(stage0, indent=2) + "\n")
    stage1_path.write_text(json.dumps(stage1, indent=2) + "\n")
    return stage0_path, stage1_path


def run_request(
    port: int,
    model_id: str,
    prompt: str,
    output_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": output_tokens,
        "temperature": 0,
        "seed": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.monotonic()
    first_token = None
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
            return {"error": f"HTTP {response.status}: {body}"}
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
            if text:
                if first_token is None:
                    first_token = time.monotonic()
                content.append(text)
        completed = time.monotonic()
        if first_token is None:
            return {"error": "stream completed without generated content"}
        return {
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "content": "".join(content),
            "content_sha256": hashlib.sha256("".join(content).encode()).hexdigest(),
            "tokens_predicted": completion_tokens or len(content),
            "ttft_ms": (first_token - started) * 1000.0,
            "elapsed_ms": (completed - started) * 1000.0,
        }
    except Exception as error:  # noqa: BLE001 - retain the error in the artifact.
        return {"error": str(error)}
    finally:
        connection.close()


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


def summarize(requests: list[dict[str, Any]], wall_ms: float) -> dict[str, Any]:
    successful = [request for request in requests if "error" not in request]
    ttft = [float(request["ttft_ms"]) for request in successful]
    predicted = sum(int(request["tokens_predicted"]) for request in successful)
    return {
        "requests": len(requests),
        "successful_requests": len(successful),
        "errors": len(requests) - len(successful),
        "ttft_ms_p50": percentile(ttft, 0.50),
        "ttft_ms_p95": percentile(ttft, 0.95),
        "makespan_ms": wall_ms,
        "output_tokens_per_second": predicted / (wall_ms / 1000.0),
    }


def launch_cell(
    args: argparse.Namespace,
    version: str,
    binary: Path,
    round_index: int,
) -> dict[str, Any]:
    cell_dir = args.output_dir / f"round-{round_index + 1}-{version}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    stage0_port, stage1_port, openai_port = free_port(), free_port(), free_port()
    stage0_config, stage1_config = write_configs(
        args, cell_dir, stage0_port, stage1_port
    )
    stage0_log_path = cell_dir / "stage-0.log"
    stage1_log_path = cell_dir / "stage-1.log"
    environment = os.environ.copy()
    environment["LLAMA_STAGE_BUILD_DIR"] = str(args.native_build.resolve())
    environment["SKIPPY_TELEMETRY_STDERR"] = "1"
    environment["SKIPPY_NATIVE_MTP_GREEDY_SAMPLING_FASTPATH"] = "1"
    common = [
        "--activation-width",
        str(args.activation_width),
        "--activation-wire-dtype",
        args.activation_wire_dtype,
        "--max-inflight",
        "2",
        "--reply-credit-limit",
        "1",
        "--async-prefill-forward",
        "--telemetry-level",
        "debug",
    ]
    stage1_command = [
        str(binary),
        "serve-binary",
        "--config",
        str(stage1_config),
        *common,
    ]
    stage0_command = [
        str(binary),
        "serve-binary",
        "--config",
        str(stage0_config),
        *common,
        "--openai-bind-addr",
        f"127.0.0.1:{openai_port}",
        "--openai-generation-concurrency",
        "1",
        "--openai-default-max-tokens",
        str(args.output_tokens),
        "--openai-prefill-chunk-policy",
        "adaptive-ramp",
        "--openai-prefill-chunk-size",
        "256",
        "--openai-prefill-adaptive-start",
        "128",
        "--openai-prefill-adaptive-step",
        "128",
        "--openai-prefill-adaptive-max",
        "384",
    ]
    stage0 = None
    stage1 = None
    with stage0_log_path.open("w") as stage0_log, stage1_log_path.open("w") as stage1_log:
        try:
            stage1 = subprocess.Popen(
                stage1_command,
                cwd=REPO,
                env=environment,
                text=True,
                stdout=stage1_log,
                stderr=subprocess.STDOUT,
            )
            wait_tcp(stage1_port, stage1, args.startup_timeout_secs, "stage 1")
            stage0 = subprocess.Popen(
                stage0_command,
                cwd=REPO,
                env=environment,
                text=True,
                stdout=stage0_log,
                stderr=subprocess.STDOUT,
            )
            wait_openai(openai_port, stage0, args.startup_timeout_secs)
            calibration = run_request(
                openai_port,
                args.model_id,
                stable_prompt(args.prompt_blocks, 0),
                args.output_tokens,
                args.request_timeout_secs,
            )
            if "error" in calibration:
                raise RuntimeError(f"calibration request failed: {calibration['error']}")
            measured = []
            started = time.monotonic()
            for request_index in range(1, args.requests + 1):
                request = run_request(
                    openai_port,
                    args.model_id,
                    stable_prompt(args.prompt_blocks, request_index),
                    args.output_tokens,
                    args.request_timeout_secs,
                )
                request["request_index"] = request_index
                measured.append(request)
            wall_ms = (time.monotonic() - started) * 1000.0
        finally:
            stop(stage0)
            stop(stage1)
    prefill_events = json_events(stage0_log_path, PREFILL_EVENT)
    calibration_events = json_events(stage0_log_path, CALIBRATION_EVENT)
    measured_prefill = prefill_events[1 : 1 + args.requests]
    attributes = [event.get("attributes", {}) for event in measured_prefill]
    summary = summarize(measured, wall_ms)
    summary.update(
        {
            "prefill_chunk_count_median": statistics.median(
                [row["llama_stage.prefill_chunk_count"] for row in attributes]
            ),
            "prefill_min_chunk_size_median": statistics.median(
                [row["llama_stage.prefill_min_chunk_size"] for row in attributes]
            ),
            "prefill_max_chunk_size_median": statistics.median(
                [row["llama_stage.prefill_max_chunk_size"] for row in attributes]
            ),
            "prefill_elapsed_ms_p95": percentile(
                [float(row["llama_stage.elapsed_ms"]) for row in attributes], 0.95
            ),
        }
    )
    latest_calibration = (
        calibration_events[-1].get("attributes", {}) if calibration_events else None
    )
    return {
        "round": round_index + 1,
        "version": version,
        "binary": str(binary),
        "binary_sha256": sha256(binary),
        "stage0_config": str(stage0_config),
        "stage1_config": str(stage1_config),
        "stage0_log": str(stage0_log_path),
        "stage1_log": str(stage1_log_path),
        "calibration_request": calibration,
        "requests": measured,
        "summary": summary,
        "latest_calibration": latest_calibration,
    }


def aggregate(cells: list[dict[str, Any]], version: str) -> dict[str, Any]:
    summaries = [cell["summary"] for cell in cells if cell["version"] == version]
    keys = (
        "successful_requests",
        "errors",
        "ttft_ms_p50",
        "ttft_ms_p95",
        "makespan_ms",
        "output_tokens_per_second",
        "prefill_chunk_count_median",
        "prefill_min_chunk_size_median",
        "prefill_max_chunk_size_median",
        "prefill_elapsed_ms_p95",
    )
    return {key: statistics.median([float(row[key]) for row in summaries]) for key in keys}


def parity(cells: list[dict[str, Any]], rounds: int) -> dict[str, Any]:
    mismatches = []
    comparable = 0
    for round_index in range(1, rounds + 1):
        old = next(cell for cell in cells if cell["round"] == round_index and cell["version"] == "old")
        new = next(cell for cell in cells if cell["round"] == round_index and cell["version"] == "new")
        for old_request, new_request in zip(old["requests"], new["requests"], strict=True):
            if "error" in old_request or "error" in new_request:
                continue
            comparable += 1
            if old_request["content"] != new_request["content"]:
                mismatches.append(
                    {"round": round_index, "request": old_request["request_index"]}
                )
    return {
        "comparable_requests": comparable,
        "exact_matches": comparable - len(mismatches),
        "mismatches": mismatches,
    }


def markdown(before: dict[str, Any], after: dict[str, Any], parity_result: dict[str, Any]) -> str:
    rows = [
        ("TTFT p50 ms", "ttft_ms_p50", True),
        ("TTFT p95 ms", "ttft_ms_p95", True),
        ("Prefill p95 ms", "prefill_elapsed_ms_p95", True),
        ("Makespan ms", "makespan_ms", True),
        ("Output tok/s", "output_tokens_per_second", False),
        ("Prefill chunks / request", "prefill_chunk_count_median", True),
        ("Maximum chunk tokens", "prefill_max_chunk_size_median", None),
    ]
    upper = max(float(before["ttft_ms_p95"]), float(after["ttft_ms_p95"])) * 1.1
    lines = [
        "```mermaid",
        "xychart-beta",
        '    title "Adaptive prefill: p95 TTFT (lower is better)"',
        '    x-axis ["Before", "After"]',
        f'    y-axis "ms" 0 --> {upper:.0f}',
        f'    bar [{before["ttft_ms_p95"]:.1f}, {after["ttft_ms_p95"]:.1f}]',
        "```",
        "",
        "| Metric | Before | After | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for label, key, _lower_is_better in rows:
        delta = delta_percent(float(before[key]), float(after[key]))
        lines.append(
            f"| {label} | {before[key]:.1f} | {after[key]:.1f} | {delta:+.1f}% |"
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
    parser.add_argument("--old-commit", required=True)
    parser.add_argument("--new-commit", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--native-build", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--requests", type=int, default=6)
    parser.add_argument("--prompt-blocks", type=int, default=384)
    parser.add_argument("--output-tokens", type=int, default=8)
    parser.add_argument("--ctx-size", type=int, default=32768)
    parser.add_argument("--layer-end", type=int, default=27)
    parser.add_argument("--split-layer", type=int, default=14)
    parser.add_argument("--activation-width", type=int, default=2048)
    parser.add_argument("--activation-wire-dtype", default="f16")
    parser.add_argument("--n-gpu-layers", type=int, default=999)
    parser.add_argument("--startup-timeout-secs", type=float, default=900)
    parser.add_argument("--request-timeout-secs", type=float, default=900)
    args = parser.parse_args()
    if args.rounds <= 0 or args.requests <= 0 or args.prompt_blocks <= 0:
        parser.error("rounds, requests, and prompt-blocks must be positive")
    if args.split_layer <= 0 or args.split_layer >= args.layer_end:
        parser.error("split-layer must be within the model layer range")
    for name in ("old_binary", "new_binary", "model"):
        path = getattr(args, name)
        if not path.is_file():
            parser.error(f"{name.replace('_', '-')} not found: {path}")
    if not args.native_build.is_dir():
        parser.error(f"native-build not found: {args.native_build}")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model = args.model.resolve()
    args.model_sha256 = sha256(args.model)
    return args


def main() -> int:
    args = parse_args()
    cells = []
    versions = {"old": args.old_binary.resolve(), "new": args.new_binary.resolve()}
    for round_index in range(args.rounds):
        order = ("old", "new") if round_index % 2 == 0 else ("new", "old")
        for version in order:
            print(f"==> round {round_index + 1}/{args.rounds}: {version}", flush=True)
            cells.append(launch_cell(args, version, versions[version], round_index))
    before = aggregate(cells, "old")
    after = aggregate(cells, "new")
    parity_result = parity(cells, args.rounds)
    result = {
        "metadata": {
            "old": {
                "commit": args.old_commit,
                "binary": str(versions["old"]),
                "sha256": sha256(versions["old"]),
            },
            "new": {
                "commit": args.new_commit,
                "binary": str(versions["new"]),
                "sha256": sha256(versions["new"]),
            },
            "model_id": args.model_id,
            "model_path": str(args.model),
            "model_sha256": args.model_sha256,
            "rounds": args.rounds,
            "measured_requests_per_cell": args.requests,
            "discarded_calibration_requests_per_cell": 1,
            "prompt_blocks": args.prompt_blocks,
            "ctx_size": args.ctx_size,
            "split": [0, args.split_layer, args.layer_end],
            "cache": "disabled",
            "prefill_policy": "adaptive-ramp: start=128, step=128, max=384",
        },
        "cells": cells,
        "aggregate": {
            "old": before,
            "new": after,
            "delta_percent": {
                key: delta_percent(float(before[key]), float(after[key]))
                for key in before
                if key not in {"successful_requests", "errors"}
            },
        },
        "output_parity": parity_result,
    }
    comparison_path = args.output_dir / "comparison.json"
    report_path = args.output_dir / "report.md"
    comparison_path.write_text(json.dumps(result, indent=2) + "\n")
    report_path.write_text(markdown(before, after, parity_result))
    print(report_path.read_text(), end="")
    expected_successes = args.rounds * args.requests
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
