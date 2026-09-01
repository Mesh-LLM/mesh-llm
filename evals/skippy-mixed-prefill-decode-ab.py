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


def wait_binary(port: int, process: subprocess.Popen[str], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error = "no attempts made"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"downstream stage exited with status {process.returncode}")
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
    raise TimeoutError(f"timed out waiting for downstream stage: {last_error}")


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


def stage_config(
    args: argparse.Namespace,
    stage_index: int,
    layer_start: int,
    layer_end: int,
    port: int,
    peer_port: int | None,
) -> dict[str, Any]:
    upstream = None
    downstream = None
    if stage_index == 0 and peer_port is not None:
        downstream = {
            "stage_id": "stage-1",
            "stage_index": 1,
            "endpoint": f"tcp://127.0.0.1:{peer_port}",
        }
    elif stage_index == 1 and peer_port is not None:
        upstream = {
            "stage_id": "stage-0",
            "stage_index": 0,
            "endpoint": f"tcp://127.0.0.1:{peer_port}",
        }
    config = {
        "run_id": "skippy-mixed-prefill-decode-ab",
        "topology_id": (
            "skippy-mixed-prefill-decode-ab-two-stage"
            if args.split_layer is not None
            else "skippy-mixed-prefill-decode-ab-local"
        ),
        "model_id": args.model_id,
        "model_path": str(args.model.resolve()),
        "source_model_sha256": args.model_sha256,
        "stage_id": f"stage-{stage_index}",
        "stage_index": stage_index,
        "layer_start": layer_start,
        "layer_end": layer_end,
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
        "upstream": upstream,
        "downstream": downstream,
    }
    return config


def write_configs(
    args: argparse.Namespace,
    cell_dir: Path,
    stage0_port: int,
    stage1_port: int | None,
) -> tuple[Path, Path | None]:
    stage0_path = cell_dir / "stage-0.json"
    stage0_end = args.split_layer or args.layer_end
    stage0_path.write_text(
        json.dumps(stage_config(args, 0, 0, stage0_end, stage0_port, stage1_port), indent=2)
        + "\n"
    )
    if stage1_port is None:
        return stage0_path, None
    stage1_path = cell_dir / "stage-1.json"
    stage1_path.write_text(
        json.dumps(
            stage_config(
                args,
                1,
                args.split_layer,
                args.layer_end,
                stage1_port,
                stage0_port,
            ),
            indent=2,
        )
        + "\n"
    )
    return stage0_path, stage1_path


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


def event_attributes(path: Path, names: set[str]) -> list[dict[str, Any]]:
    events = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        event_name = event.get("event")
        if event_name in names:
            attributes = dict(event.get("attributes", {}))
            attributes["_event"] = event_name
            events.append(attributes)
    return events


def scheduler_events(path: Path) -> list[dict[str, Any]]:
    return event_attributes(
        path,
        {"stage.scheduler_iteration", "stage.scheduler_feature_iteration"},
    )


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
        (
            int(event.get("skippy.scheduler.prefill_tokens", 0))
            + int(event.get("skippy.scheduler.recompute_tokens", 0))
            + int(event.get("skippy.scheduler.decode_tokens", 0))
        )
        if event.get("_event") == "stage.scheduler_iteration"
        else int(event.get("skippy.scheduler.token_count", 0))
        for event in events
    ]
    detailed_events = [
        event
        for event in events
        if event.get("_event") == "stage.scheduler_iteration"
    ]
    mixed = [
        event
        for event in detailed_events
        if int(event.get("skippy.scheduler.decode_tokens", 0)) > 0
        and (
            int(event.get("skippy.scheduler.prefill_tokens", 0))
            + int(event.get("skippy.scheduler.recompute_tokens", 0))
            > 0
        )
    ]
    prefill_only = [
        event
        for event in detailed_events
        if int(event.get("skippy.scheduler.decode_tokens", 0)) == 0
        and (
            int(event.get("skippy.scheduler.prefill_tokens", 0))
            + int(event.get("skippy.scheduler.recompute_tokens", 0))
            > 0
        )
    ]
    decode_only = [
        event
        for event in detailed_events
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
        "scheduler_breakdown_available": bool(detailed_events),
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
    stage0_port, openai_port = free_port(), free_port()
    stage1_port = free_port() if args.split_layer is not None else None
    stage0_config, stage1_config = write_configs(
        args, cell_dir, stage0_port, stage1_port
    )
    stage0_log_path = cell_dir / "stage-0.log"
    stage1_log_path = cell_dir / "stage-1.log"
    common = [
        "--activation-width",
        str(args.activation_width),
        "--activation-wire-dtype",
        args.activation_wire_dtype,
        "--max-inflight",
        str(args.lanes),
        "--telemetry-level",
        "debug",
    ]
    if stage1_config is not None:
        common.extend(["--reply-credit-limit", "1", "--async-prefill-forward"])
    stage0_command = [
        str(binary),
        "serve-binary",
        "--config",
        str(stage0_config),
        *common,
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
        str(args.prefill_adaptive_start),
        "--openai-prefill-adaptive-step",
        str(args.prefill_adaptive_step),
        "--openai-prefill-adaptive-max",
        str(args.prefill_adaptive_max),
    ]
    if version == "new" or not args.adaptive_target_new_only:
        stage0_command.extend(
            ["--openai-prefill-adaptive-target-ms", str(args.adaptive_target_ms)]
        )
    stage1_command = (
        [str(binary), "serve-binary", "--config", str(stage1_config), *common]
        if stage1_config is not None
        else None
    )
    environment = os.environ.copy()
    environment["LLAMA_STAGE_BUILD_DIR"] = str(native_build.resolve())
    environment["SKIPPY_TELEMETRY_STDERR"] = "1"
    environment["SKIPPY_NATIVE_MTP_GREEDY_SAMPLING_FASTPATH"] = "1"
    stage0_process = None
    stage1_process = None
    with (
        stage0_log_path.open("w") as stage0_log,
        stage1_log_path.open("w") as stage1_log,
    ):
        try:
            if stage1_command is not None:
                stage1_process = subprocess.Popen(
                    stage1_command,
                    cwd=Path(__file__).resolve().parents[1],
                    env=environment,
                    text=True,
                    stdout=stage1_log,
                    stderr=subprocess.STDOUT,
                )
                wait_binary(
                    stage1_port,
                    stage1_process,
                    args.startup_timeout_secs,
                )
            stage0_process = subprocess.Popen(
                stage0_command,
                cwd=Path(__file__).resolve().parents[1],
                env=environment,
                text=True,
                stdout=stage0_log,
                stderr=subprocess.STDOUT,
            )
            wait_openai(openai_port, stage0_process, args.startup_timeout_secs)
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
            warmup_iteration_count = len(scheduler_events(stage0_log_path))
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
            stop(stage0_process)
            stop(stage1_process)
    events = scheduler_events(stage0_log_path)[warmup_iteration_count:]
    prefill_events = event_attributes(
        stage0_log_path, {"stage.openai_prefill"}
    )[1:]
    measured_prefills = sorted(
        prefill_events,
        key=lambda row: int(row.get("llama_stage.prefill_token_count", 0)),
        reverse=True,
    )[: args.prefills]

    def median_attribute(name: str) -> float | None:
        values = [int(row[name]) for row in measured_prefills if name in row]
        return statistics.median(values) if values else None

    summary = summarize_requests(requests, wall_ms, events, args.n_batch)
    summary.update(
        {
            "prefill_chunk_count_median": median_attribute(
                "llama_stage.prefill_chunk_count"
            ),
            "prefill_max_chunk_size_median": median_attribute(
                "llama_stage.prefill_max_chunk_size"
            ),
            "prefill_bottleneck_stage_median": median_attribute(
                "llama_stage.prefill_bottleneck_stage_index"
            ),
        }
    )
    return {
        "round": round_index + 1,
        "version": version,
        "binary": str(binary),
        "binary_sha256": sha256(binary),
        "native_build": str(native_build),
        "stage0_config": str(stage0_config),
        "stage1_config": str(stage1_config) if stage1_config is not None else None,
        "stage0_log": str(stage0_log_path),
        "stage1_log": str(stage1_log_path) if stage1_config is not None else None,
        "requests": requests,
        "summary": summary,
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
    "prefill_chunk_count_median",
    "prefill_max_chunk_size_median",
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
        ("Prefill chunks / request", "prefill_chunk_count_median"),
        ("Maximum prefill chunk", "prefill_max_chunk_size_median"),
    ]
    chart_metrics = [
        ("Makespan", "makespan_ms"),
        ("Throughput", "output_tokens_per_second"),
        ("Anchor-TTFT-p95", "anchor_ttft_ms_p95"),
        ("Anchor-gap-p95", "anchor_gap_ms_p95"),
        ("Prefill-TTFT-p95", "prefill_ttft_ms_p95"),
    ]
    chart_values = []
    for label, key in chart_metrics:
        before_value = before.get(key)
        after_value = after.get(key)
        if before_value is None or after_value is None or before_value == 0:
            continue
        chart_values.append((label, 100.0 * after_value / before_value))
    lines = []
    if chart_values:
        chart_upper = max(120.0, max(value for _, value in chart_values) * 1.1)
        labels = ", ".join(json.dumps(label) for label, _ in chart_values)
        values = ", ".join(f"{value:.1f}" for _, value in chart_values)
        lines.extend(
            [
                "```mermaid",
                "xychart-beta",
                '    title "Mixed scheduling (base = 100)"',
                f"    x-axis [{labels}]",
                f'    y-axis "Percent of base" 0 --> {chart_upper:.0f}',
                "    bar [" + ", ".join("100" for _ in chart_values) + "]",
                f"    bar [{values}]",
                "```",
                "",
            ]
        )
    lines.extend(
        [
        "| Metric | Before | After | Delta | Paired 95% CI |",
        "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for label, key in rows:
        before_value = before.get(key)
        after_value = after.get(key)
        if before_value is None or after_value is None:
            continue
        delta = delta_percent(before_value, after_value)
        delta_text = f"{delta:+.1f}%" if delta is not None else "n/a"
        interval = intervals.get(key, {}).get("ci95")
        interval_text = (
            f"[{interval[0]:+.1f}%, {interval[1]:+.1f}%]" if interval else "n/a"
        )
        lines.append(
            f"| {label} | {before_value:.3f} | {after_value:.3f} | {delta_text} | {interval_text} |"
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
    parser.add_argument(
        "--prefill-adaptive-start",
        type=int,
        help="adaptive-ramp starting chunk; defaults to n-ubatch",
    )
    parser.add_argument(
        "--prefill-adaptive-step",
        type=int,
        help="adaptive-ramp chunk increment; defaults to n-ubatch",
    )
    parser.add_argument(
        "--prefill-adaptive-max",
        type=int,
        help="adaptive-ramp maximum chunk; defaults to n-ubatch",
    )
    parser.add_argument("--layer-end", type=int, required=True)
    parser.add_argument(
        "--split-layer",
        type=int,
        help="split into two local pipeline stages after this layer",
    )
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
    if args.split_layer is not None and not 0 < args.split_layer < args.layer_end:
        parser.error("split-layer must be within the model layer range")
    for name in (
        "prefill_adaptive_start",
        "prefill_adaptive_step",
        "prefill_adaptive_max",
    ):
        if getattr(args, name) is None:
            setattr(args, name, args.n_ubatch)
        elif getattr(args, name) <= 0:
            parser.error(f"{name.replace('_', '-')} must be positive")
    if args.prefill_adaptive_start > args.prefill_adaptive_max:
        parser.error("prefill-adaptive-start must not exceed prefill-adaptive-max")
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
                "prefill_adaptive_start": args.prefill_adaptive_start,
                "prefill_adaptive_step": args.prefill_adaptive_step,
                "prefill_adaptive_max": args.prefill_adaptive_max,
                "layer_end": args.layer_end,
                "split_layer": args.split_layer,
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
