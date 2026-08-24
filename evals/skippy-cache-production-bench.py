#!/usr/bin/env python3
"""Run production Skippy cache correctness and llama-server baselines.

This runner intentionally benchmarks only production cache payloads:
ResidentKv and KvRecurrent. FullState is a correctness diagnostic and is not a
performance target.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[1]
HOME = Path.home()


@dataclass(frozen=True)
class Case:
    key: str
    family: str
    model_id: str
    model_path: Path | None
    payload: str
    layer_end: int
    activation_width: int
    ctx_size: int = 512
    n_gpu_layers: int = 0
    prefix_tokens: int = 128
    cache_hit_repeats: int = 3
    stage_load_mode: str = "runtime-slice"
    state_layer_start: int = 0
    state_layer_end: int | None = None
    state_stage_index: int | None = None
    resident_kv_bytes_per_token: int | None = None
    skip_llama_server_reason: str | None = None


@dataclass(frozen=True)
class UseCase:
    key: str
    label: str
    prompt: str
    prefix_tokens: int = 128
    source_dataset: str | None = None
    source_config: str | None = None
    source_split: str | None = None
    source_row: int | None = None


def load_use_cases(path: Path) -> list[UseCase]:
    data = json.loads(path.read_text())
    use_cases = []
    for item in data.get("use_cases", []):
        source = item.get("source", {})
        use_cases.append(
            UseCase(
                key=item["key"],
                label=item["label"],
                prompt=item["prompt"],
                prefix_tokens=int(item.get("prefix_tokens", 128)),
                source_dataset=source.get("dataset"),
                source_config=source.get("config"),
                source_split=source.get("split"),
                source_row=source.get("row_idx"),
            )
        )
    return use_cases


CASES = [
    Case(
        "qwen3_dense",
        "Qwen3 dense",
        "Qwen/Qwen3-0.6B:Q8_0",
        HOME
        / ".cache/huggingface/hub/models--Qwen--Qwen3-0.6B-GGUF/snapshots/23749fefcc72300e3a2ad315e1317431b06b590a/Qwen3-0.6B-Q8_0.gguf",
        "resident-kv",
        28,
        1024,
        resident_kv_bytes_per_token=114_688,
    ),
    Case(
        "llama",
        "Llama",
        "hugging-quants/Llama-3.2-1B-Instruct-Q4_K_M-GGUF:Q4_K_M",
        HOME
        / ".cache/huggingface/hub/models--hugging-quants--Llama-3.2-1B-Instruct-Q4_K_M-GGUF/snapshots/7d1f70022fcab2038000074bd0342e03e1d8b755/llama-3.2-1b-instruct-q4_k_m.gguf",
        "resident-kv",
        16,
        2048,
        resident_kv_bytes_per_token=32_768,
    ),
    Case(
        "deepseek2",
        "DeepSeek2",
        "bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF:Q4_K_M",
        HOME
        / ".cache/huggingface/hub/models--bartowski--DeepSeek-Coder-V2-Lite-Instruct-GGUF/snapshots/8f248fa2072348f77a8bc37754e470de1f61866e/DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf",
        "resident-kv",
        27,
        2048,
        prefix_tokens=64,
        resident_kv_bytes_per_token=276_480,
    ),
    Case(
        "deepseek3",
        "DeepSeek3",
        "unsloth/DeepSeek-V3.2-GGUF:UD-Q4_K_XL",
        HOME
        / ".cache/huggingface/hub/models--meshllm--DeepSeek-V3.2-UD-Q4_K_XL-layers/snapshots/c7d74031a7201334b4550da6537d0b8734b81fe2",
        "resident-kv",
        61,
        7168,
        ctx_size=32,
        prefix_tokens=4,
        stage_load_mode="layer-package",
        state_layer_start=3,
        state_layer_end=4,
        state_stage_index=1,
        skip_llama_server_reason="Layer-package evidence only; no full GGUF is loaded for this DeepSeek3 gate.",
        resident_kv_bytes_per_token=2_176,
    ),
    Case(
        "glm47_flash",
        "GLM-4.7 Flash",
        "unsloth/GLM-4.7-Flash-GGUF:Q4_K_M",
        HOME
        / ".cache/huggingface/hub/models--unsloth--GLM-4.7-Flash-GGUF/snapshots/0d32489ecb9db6d2a4fc93bd27ef01519f95474d/GLM-4.7-Flash-Q4_K_M.gguf",
        "resident-kv",
        47,
        2048,
        prefix_tokens=32,
        resident_kv_bytes_per_token=102_272,
    ),
    Case(
        "glm4",
        "GLM4",
        "meshllm/glm-4-9b-0414-parity-q4_k_m-gguf:Q4_K_M",
        HOME
        / ".cache/huggingface/hub/models--meshllm--glm-4-9b-0414-parity-q4_k_m-gguf/snapshots/b15dd8df3957ace630d34943149a180282db4680/glm-4-9b-0414-q4_k_m.gguf",
        "resident-kv",
        40,
        4096,
        prefix_tokens=32,
        resident_kv_bytes_per_token=40_960,
    ),
    Case(
        "gemma4_a4b",
        "Gemma4 A4B",
        "batiai/Gemma-4-26B-A4B-it-GGUF:Q6_K",
        HOME
        / ".cache/huggingface/hub/models--batiai--Gemma-4-26B-A4B-it-GGUF/snapshots/45ad6023c1c79fe5813b34270bc4d44e392a0d17/google-gemma-4-26B-A4B-it-Q6_K.gguf",
        "resident-kv",
        30,
        2816,
        prefix_tokens=16,
        resident_kv_bytes_per_token=225_280,
    ),
    Case(
        "gemma4_e4b",
        "Gemma4 E4B",
        "unsloth/gemma-4-E4B-it-GGUF:Q4_K_M",
        HOME
        / ".cache/huggingface/hub/models--unsloth--gemma-4-E4B-it-GGUF/snapshots/315e03409eb1cdde302488d66e586dea1e82aad1/gemma-4-E4B-it-Q4_K_M.gguf",
        "resident-kv",
        42,
        2560,
        prefix_tokens=16,
        resident_kv_bytes_per_token=57_344,
    ),
    Case(
        "gemma3",
        "Gemma3",
        "ggml-org/gemma-3-1b-it-GGUF:Q4_K_M",
        HOME
        / ".cache/huggingface/hub/models--ggml-org--gemma-3-1b-it-GGUF/snapshots/f9c28bcd85737ffc5aef028638d3341d49869c27/gemma-3-1b-it-Q4_K_M.gguf",
        "resident-kv",
        26,
        1152,
        resident_kv_bytes_per_token=26_624,
    ),
    Case(
        "gemma2",
        "Gemma2",
        "bartowski/gemma-2-2b-it-GGUF:Q4_K_M",
        HOME
        / ".cache/huggingface/hub/models--bartowski--gemma-2-2b-it-GGUF/snapshots/855f67caed130e1befc571b52bd181be2e858883/gemma-2-2b-it-Q4_K_M.gguf",
        "resident-kv",
        26,
        2304,
        resident_kv_bytes_per_token=106_496,
    ),
    Case(
        "falcon_h1",
        "Falcon-H1",
        "tiiuae/Falcon-H1-1.5B-Instruct-GGUF:Q4_K_M",
        HOME
        / ".cache/huggingface/hub/models--tiiuae--Falcon-H1-1.5B-Instruct-GGUF/snapshots/0d3a6cfe25fb4eeab0153fb8623aac5b69d6bd0a/Falcon-H1-1.5B-Instruct-Q4_K_M.gguf",
        "kv-recurrent",
        24,
        2048,
    ),
    Case(
        "olmo",
        "OLMo",
        "meshllm/olmo-7b-instruct-hf-parity-f16-gguf:F16",
        HOME
        / ".cache/huggingface/hub/models--meshllm--olmo-7b-instruct-hf-parity-f16-gguf/snapshots/846c0ae38aff29ea8fce0959fb406cdcef858bac/olmo-7b-instruct-hf-f16.gguf",
        "resident-kv",
        32,
        4096,
        prefix_tokens=64,
        resident_kv_bytes_per_token=524_288,
    ),
    Case(
        "minimax_m27",
        "MiniMax M2.7",
        "unsloth/MiniMax-M2.7-GGUF:UD-Q2_K_XL",
        HOME
        / ".cache/huggingface/hub/models--unsloth--MiniMax-M2.7-GGUF/snapshots/d2a05ccf69491b03db0cc40b335aec14bdaf7198/UD-Q2_K_XL/MiniMax-M2.7-UD-Q2_K_XL-00001-of-00003.gguf",
        "resident-kv",
        62,
        3072,
        prefix_tokens=16,
        resident_kv_bytes_per_token=253_952,
    ),
    Case(
        "qwen3next",
        "Qwen3Next",
        "bartowski/Qwen_Qwen3-Coder-Next-GGUF:IQ2_XS",
        HOME
        / ".cache/huggingface/hub/models--bartowski--Qwen_Qwen3-Coder-Next-GGUF/snapshots/d32741c4b434bf1f927798d0c093564c7f4e92fd/Qwen_Qwen3-Coder-Next-IQ2_XS.gguf",
        "kv-recurrent",
        48,
        2048,
        prefix_tokens=16,
    ),
]


def http_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 30.0) -> Any:
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["content-type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def wait_ready(
    port: int,
    proc: subprocess.Popen[str],
    timeout: float,
    path: str = "/health",
    server_name: str = "llama-server",
) -> None:
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"{server_name} exited early with code {proc.returncode}")
        try:
            http_json(f"http://127.0.0.1:{port}{path}", timeout=2.0)
            return
        except Exception as exc:  # noqa: BLE001 - readiness loop keeps last error.
            last_error = exc
            time.sleep(0.5)
    raise TimeoutError(f"{server_name} did not become ready: {last_error}")


def warm_mean_ms(runs: list[dict[str, Any]]) -> float | None:
    values = [run.get("elapsed_ms") for run in runs[1:] if isinstance(run.get("elapsed_ms"), (int, float))]
    if not values:
        return None
    return sum(values) / len(values)


def median_ms(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def warm_median_ms(runs: list[dict[str, Any]]) -> float | None:
    values = [run.get("elapsed_ms") for run in runs[1:] if isinstance(run.get("elapsed_ms"), (int, float))]
    return median_ms(values)


def skippy_hit_median_ms(skippy: dict[str, Any]) -> float | None:
    imports = skippy.get("cache_hit_import_ms")
    decodes = skippy.get("cache_hit_decode_ms")
    if not isinstance(imports, list) or not isinstance(decodes, list):
        return None
    values = [
        float(import_ms) + float(decode_ms)
        for import_ms, decode_ms in zip(imports, decodes)
        if isinstance(import_ms, (int, float)) and isinstance(decode_ms, (int, float))
    ]
    return median_ms(values)


def run_correctness(case: Case, args: argparse.Namespace, case_dir: Path, prompt: str | None = None) -> dict[str, Any]:
    report_path = case_dir / "skippy-state-handoff.json"
    cache_hit_repeats = args.cache_hit_repeats or case.cache_hit_repeats
    cmd = [
        str(args.skippy_correctness_bin),
        "state-handoff",
        "--model",
        str(case.model_path),
        "--model-id",
        case.model_id,
        "--layer-end",
        str(case.layer_end),
        "--ctx-size",
        str(case.ctx_size),
        f"--n-gpu-layers={case.n_gpu_layers}",
        "--stage-load-mode",
        case.stage_load_mode,
        "--state-layer-end",
        str(case.state_layer_end or case.layer_end),
        "--state-payload-kind",
        case.payload,
        "--prefix-token-count",
        str(case.prefix_tokens),
        "--cache-hit-repeats",
        str(cache_hit_repeats),
        "--report-out",
        str(report_path),
    ]
    if prompt is not None:
        cmd.extend(["--prompt", prompt])
    if case.state_layer_start:
        cmd.extend(["--state-layer-start", str(case.state_layer_start)])
    if case.state_stage_index is not None:
        cmd.extend(["--state-stage-index", str(case.state_stage_index)])
    if args.runtime_lane_count is not None:
        cmd.extend(["--runtime-lane-count", str(args.runtime_lane_count)])
    if args.borrow_resident_hits:
        cmd.append("--borrow-resident-hits")
    if args.cache_decoded_result_hits:
        cmd.append("--cache-decoded-result-hits")
    env = os.environ.copy()
    env["LLAMA_STAGE_BUILD_DIR"] = str(args.llama_stage_build_dir)
    started = time.monotonic()
    completed = subprocess.run(
        cmd,
        cwd=REPO,
        env=env,
        text=True,
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=args.correctness_timeout_secs,
    )
    elapsed_ms = (time.monotonic() - started) * 1000
    (case_dir / "skippy-state-handoff.log").write_text(completed.stdout)
    if completed.returncode != 0:
        return {
            "status": "fail",
            "exit_code": completed.returncode,
            "elapsed_ms": elapsed_ms,
            "log": str(case_dir / "skippy-state-handoff.log"),
        }
    report = json.loads(report_path.read_text())
    report["runner_elapsed_ms"] = elapsed_ms
    return report


def run_llama_server(case: Case, prompt: str, args: argparse.Namespace, case_dir: Path) -> dict[str, Any]:
    port = free_port()
    log_path = case_dir / "llama-server.log"
    with log_path.open("w") as log:
        cmd = [
            str(args.llama_server_bin),
            "--model",
            str(case.model_path),
            "--ctx-size",
            str(case.ctx_size),
            "--n-gpu-layers",
            str(case.n_gpu_layers),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--parallel",
            str(args.llama_parallel),
            "--no-webui",
        ]
        proc = subprocess.Popen(
            cmd,
            cwd=REPO,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            wait_ready(port, proc, args.server_startup_timeout_secs)
            runs = []
            for index in range(args.llama_repeats):
                payload = {
                    "prompt": prompt,
                    "n_predict": 1,
                    "temperature": 0,
                    "top_k": 1,
                    "cache_prompt": True,
                }
                started = time.monotonic()
                response = http_json(f"http://127.0.0.1:{port}/completion", payload, timeout=args.request_timeout_secs)
                elapsed_ms = (time.monotonic() - started) * 1000
                timings = response.get("timings", {})
                runs.append(
                    {
                        "run": index + 1,
                        "elapsed_ms": elapsed_ms,
                        "content": response.get("content"),
                        "tokens_evaluated": timings.get("tokens_evaluated"),
                        "tokens_predicted": timings.get("tokens_predicted"),
                        "tokens_cached": timings.get("tokens_cached"),
                        "prompt_n": timings.get("prompt_n"),
                        "cache_n": timings.get("cache_n"),
                        "prompt_ms": timings.get("prompt_ms"),
                        "predicted_ms": timings.get("predicted_ms"),
                    }
                )
            return {
                "status": "ok",
                "log": str(log_path),
                "parallel": args.llama_parallel,
                "runs": runs,
                "warm_mean_ms": warm_mean_ms(runs),
                "warm_median_ms": warm_median_ms(runs),
            }
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)


def run_concurrent_requests(
    prompt: str,
    concurrency: int,
    num_requests: int,
    port: int,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], float]:
    """Run multiple concurrent requests and measure per-request timings."""
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    if num_requests < concurrency:
        raise ValueError(
            f"num_requests ({num_requests}) must be at least concurrency ({concurrency})"
        )

    def make_request(request_id: int) -> dict[str, Any]:
        payload = {
            "prompt": prompt,
            "n_predict": 128,
            "temperature": 0,
            "top_k": 1,
            "cache_prompt": True,
            "stream": True,
        }
        started = time.monotonic()
        first_token_time = None
        generated_chunks = 0
        timings: dict[str, Any] = {}
        conn = None
        try:
            import http.client

            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=args.request_timeout_secs)
            conn.request("POST", "/completion", json.dumps(payload), {"Content-Type": "application/json"})
            response = conn.getresponse()

            if response.status != 200:
                return {
                    "request_id": request_id,
                    "error": f"HTTP {response.status}",
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
                if isinstance(event.get("timings"), dict):
                    timings = event["timings"]
                delta = event.get("choices", [{}])[0].get("delta", {})
                content = (
                    event.get("content")
                    or event.get("reasoning_content")
                    or delta.get("content")
                    or delta.get("reasoning_content")
                )
                if content:
                    if first_token_time is None:
                        first_token_time = time.monotonic()
                    generated_chunks += 1

            completed = time.monotonic()
            elapsed_ms = (completed - started) * 1000
            if first_token_time is None:
                return {
                    "request_id": request_id,
                    "error": "stream completed without generated content",
                    "elapsed_ms": elapsed_ms,
                }
            tokens_predicted = timings.get("tokens_predicted")
            if not isinstance(tokens_predicted, int) or tokens_predicted <= 0:
                tokens_predicted = generated_chunks
            generation_intervals = max(tokens_predicted - 1, 1)
            tpot_ms = (completed - first_token_time) * 1000 / generation_intervals
            return {
                "request_id": request_id,
                "elapsed_ms": elapsed_ms,
                "ttft_ms": (first_token_time - started) * 1000,
                "tpot_ms": tpot_ms,
                "tokens_predicted": tokens_predicted,
                "prompt_n": timings.get("prompt_n"),
                "cache_n": timings.get("cache_n"),
            }
        except Exception as exc:
            return {
                "request_id": request_id,
                "error": str(exc),
                "elapsed_ms": (time.monotonic() - started) * 1000,
            }
        finally:
            if conn is not None:
                conn.close()

    workload_started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(make_request, request_id) for request_id in range(num_requests)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    makespan_ms = (time.monotonic() - workload_started) * 1000
    results.sort(key=lambda result: result["request_id"])
    return results, makespan_ms


def run_openai_concurrent_requests(
    prompt: str,
    model_id: str,
    concurrency: int,
    num_requests: int,
    output_tokens: int,
    port: int,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], float]:
    """Run a closed-loop fixed-concurrency sweep against Skippy's OpenAI SSE API."""
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    if num_requests < concurrency:
        raise ValueError(
            f"num_requests ({num_requests}) must be at least concurrency ({concurrency})"
        )

    def make_request(request_id: int) -> dict[str, Any]:
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
        conn = None
        try:
            import http.client

            conn = http.client.HTTPConnection(
                "127.0.0.1",
                port,
                timeout=args.request_timeout_secs,
            )
            conn.request(
                "POST",
                "/v1/chat/completions",
                json.dumps(request_payload),
                {"Content-Type": "application/json"},
            )
            response = conn.getresponse()
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
                if isinstance(usage, dict) and isinstance(usage.get("completion_tokens"), int):
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
            elapsed_ms = (completed - started) * 1000
            if first_token_at is None:
                return {
                    "request_id": request_id,
                    "error": "stream completed without generated content",
                    "elapsed_ms": elapsed_ms,
                }
            tokens_predicted = completion_tokens or content_events
            generation_intervals = max(tokens_predicted - 1, 1)
            return {
                "request_id": request_id,
                "elapsed_ms": elapsed_ms,
                "ttft_ms": (first_token_at - started) * 1000,
                "tpot_ms": (completed - first_token_at) * 1000 / generation_intervals,
                "tokens_predicted": tokens_predicted,
                "content": "".join(content_parts),
                "first_content": content_parts[0],
            }
        except Exception as exc:  # noqa: BLE001 - retain per-request failures in the report.
            return {
                "request_id": request_id,
                "error": str(exc),
                "elapsed_ms": (time.monotonic() - started) * 1000,
            }
        finally:
            if conn is not None:
                conn.close()

    workload_started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(make_request, request_id) for request_id in range(num_requests)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    makespan_ms = (time.monotonic() - workload_started) * 1000
    results.sort(key=lambda result: result["request_id"])
    return results, makespan_ms


def serving_path_output_parity(
    old: dict[str, Any], new: dict[str, Any]
) -> list[dict[str, Any]]:
    """Compare deterministic response text at each concurrency point."""
    old_by_concurrency = {
        row["concurrency"]: row for row in old.get("concurrency_sweep", [])
    }
    new_by_concurrency = {
        row["concurrency"]: row for row in new.get("concurrency_sweep", [])
    }
    parity = []
    for concurrency in sorted(old_by_concurrency.keys() & new_by_concurrency.keys()):
        old_requests = {
            row["request_id"]: row
            for row in old_by_concurrency[concurrency].get("per_request", [])
            if "error" not in row
        }
        new_requests = {
            row["request_id"]: row
            for row in new_by_concurrency[concurrency].get("per_request", [])
            if "error" not in row
        }
        comparable_ids = sorted(old_requests.keys() & new_requests.keys())
        mismatches = [
            request_id
            for request_id in comparable_ids
            if old_requests[request_id].get("content")
            != new_requests[request_id].get("content")
        ]
        first_content_mismatches = [
            request_id
            for request_id in comparable_ids
            if old_requests[request_id].get("first_content")
            != new_requests[request_id].get("first_content")
        ]
        parity.append(
            {
                "concurrency": concurrency,
                "comparable_requests": len(comparable_ids),
                "exact_matches": len(comparable_ids) - len(mismatches),
                "mismatch_request_ids": mismatches,
                "first_content_matches": len(comparable_ids)
                - len(first_content_mismatches),
                "first_content_mismatch_request_ids": first_content_mismatches,
            }
        )
    return parity


_MODEL_SHA256: dict[Path, str] = {}


def model_sha256(path: Path) -> str:
    cached = _MODEL_SHA256.get(path)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as model_file:
        for chunk in iter(lambda: model_file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    _MODEL_SHA256[path] = value
    return value


def write_skippy_benchmark_config(
    case: Case,
    path: Path,
    ctx_size: int,
    lane_count: int,
) -> None:
    if case.model_path is None:
        raise ValueError("Skippy serving benchmark requires a model path")
    config = {
        "run_id": "serving-path-benchmark",
        "topology_id": "serving-path-benchmark-single-stage",
        "model_id": case.model_id,
        "model_path": str(case.model_path),
        "source_model_sha256": model_sha256(case.model_path),
        "stage_id": "stage-0",
        "stage_index": 0,
        "layer_start": 0,
        "layer_end": case.layer_end,
        "ctx_size": ctx_size,
        "lane_count": lane_count,
        "n_gpu_layers": case.n_gpu_layers,
        "filter_tensors_on_load": False,
        "load_mode": "runtime-slice",
        "bind_addr": "127.0.0.1:0",
        "upstream": None,
        "downstream": None,
        "kv_server": None,
    }
    path.write_text(json.dumps(config, indent=2) + "\n")


def run_skippy_serving_path_sweep(
    label: str,
    server_bin: Path,
    case: Case,
    prompt: str,
    concurrency_sweep: list[int],
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    """Start one Skippy binary and collect the full fixed-N scaling curve."""
    lane_count = args.runtime_lane_count or max(concurrency_sweep)
    ctx_size = args.serving_ctx_size or max(case.ctx_size, case.ctx_size * lane_count)
    config_path = output_dir / f"{label}-stage.json"
    log_path = output_dir / f"{label}-server.log"
    write_skippy_benchmark_config(case, config_path, ctx_size, lane_count)
    port = free_port()
    cmd = [
        str(server_bin),
        "serve-openai",
        "--config",
        str(config_path),
        "--bind-addr",
        f"127.0.0.1:{port}",
        "--generation-concurrency",
        str(lane_count),
        "--telemetry-level",
        "debug",
    ]
    env = os.environ.copy()
    env["LLAMA_STAGE_BUILD_DIR"] = str(args.llama_stage_build_dir)
    env["SKIPPY_TELEMETRY_STDERR"] = "1"
    # Exact parity needs a true argmax path; otherwise llama's stateful sampler
    # may seed each concurrent session independently even at temperature 0.
    env["SKIPPY_NATIVE_MTP_GREEDY_SAMPLING_FASTPATH"] = "1"
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=REPO,
            env=env,
            text=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            wait_ready(
                port,
                proc,
                args.server_startup_timeout_secs,
                path="/v1/models",
                server_name=f"Skippy {label}",
            )
            curve = []
            for concurrency in concurrency_sweep:
                print(f"==> {case.key}: {label} serving path concurrency={concurrency}", flush=True)
                per_request, makespan_ms = run_openai_concurrent_requests(
                    prompt,
                    case.model_id,
                    concurrency,
                    max(args.concurrent_requests, concurrency),
                    args.concurrent_output_tokens,
                    port,
                    args,
                )
                curve.append(
                    {
                        "concurrency": concurrency,
                        "requests": len(per_request),
                        "makespan_ms": makespan_ms,
                        "metrics": calculate_goodput(
                            per_request,
                            makespan_ms,
                            args.ttft_slo_ms,
                            args.tpot_slo_ms,
                        ),
                        "per_request": per_request,
                    }
                )
            return {
                "label": label,
                "binary": str(server_bin),
                "config": str(config_path),
                "log": str(log_path),
                "ctx_size": ctx_size,
                "lane_count": lane_count,
                "concurrency_sweep": curve,
            }
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)


def format_ms(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.1f}"
    return "n/a"


def format_bytes(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    if value == 0:
        return "0"
    if value < 1024 * 1024:
        kib = value / 1024
        return f"{kib:.1f} KiB"
    mib = value / (1024 * 1024)
    return f"{mib:.1f} MiB"


def cache_storage_bytes(row: dict[str, Any]) -> int | None:
    skippy = row.get("skippy", {})
    measured = skippy.get("cache_storage_bytes")
    if isinstance(measured, (int, float)):
        return int(measured)
    case = row.get("case", {})
    bytes_per_token = case.get("resident_kv_bytes_per_token")
    prefix_tokens = row.get("prefix_tokens")
    if row.get("payload") == "resident-kv" and isinstance(bytes_per_token, int) and isinstance(prefix_tokens, int):
        return bytes_per_token * prefix_tokens
    return None


def cache_storage_method(row: dict[str, Any]) -> str:
    skippy = row.get("skippy", {})
    if isinstance(skippy.get("cache_storage_bytes"), (int, float)):
        return "measured"
    if cache_storage_bytes(row) is not None:
        return "metadata-derived"
    return "n/a"


def markdown_table(results: list[dict[str, Any]]) -> str:
    include_use_case = any(row.get("use_case") for row in results)
    if include_use_case:
        lines = [
            "| Use case | Family | Payload | Correctness | Prefix tokens | Prompt tokens | llama-server warm median ms | Skippy hit median ms | Speedup | Cache bytes | Size method | Notes |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    else:
        lines = [
            "| Family | Payload | Correctness | Prefix tokens | Prompt tokens | llama-server warm median ms | Skippy hit median ms | Speedup | Cache bytes | Size method | Notes |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    for row in results:
        correctness = row.get("skippy", {}).get("status", "missing")
        llama_warm = row.get("llama_server", {}).get("warm_median_ms")
        if llama_warm is None:
            llama_warm = row.get("llama_server", {}).get("warm_mean_ms")
        skippy_hit = skippy_hit_median_ms(row.get("skippy", {}))
        if skippy_hit is None:
            skippy_hit = row.get("skippy", {}).get("cache_hit_total_ms")
        speedup = None
        if isinstance(llama_warm, (int, float)) and isinstance(skippy_hit, (int, float)) and skippy_hit > 0:
            speedup = llama_warm / skippy_hit
        notes = row.get("notes", "")
        cells = {
            "use_case": row.get("use_case_label", "n/a").replace("|", "/"),
            "family": row["family"],
            "payload": row["payload"],
            "correctness": correctness,
            "prefix": row.get("prefix_tokens", "n/a"),
            "tokens": row.get("benchmark_prompt_token_count", "n/a"),
            "llama": format_ms(llama_warm),
            "skippy": format_ms(skippy_hit),
            "speedup": f"{speedup:.2f}x" if speedup is not None else "n/a",
            "bytes": format_bytes(cache_storage_bytes(row)),
            "method": cache_storage_method(row),
            "notes": notes.replace("|", "/"),
        }
        if include_use_case:
            lines.append(
                "| {use_case} | {family} | `{payload}` | {correctness} | {prefix} | {tokens} | {llama} | {skippy} | {speedup} | {bytes} | {method} | {notes} |".format(
                    **cells
                )
            )
        else:
            lines.append(
                "| {family} | `{payload}` | {correctness} | {prefix} | {tokens} | {llama} | {skippy} | {speedup} | {bytes} | {method} | {notes} |".format(
                    **cells
                )
            )
    return "\n".join(lines) + "\n"


def parse_prefix_sweep(value: str | None) -> list[int | None]:
    if value is None:
        return [None]
    sizes = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        size = int(raw)
        if size <= 0:
            raise SystemExit("--prefix-token-sweep values must be positive")
        sizes.append(size)
    if not sizes:
        raise SystemExit("--prefix-token-sweep did not contain any sizes")
    return sizes


def resolve_case(case: Case, args: argparse.Namespace, use_case: UseCase | None = None) -> Case:
    cache_hit_repeats = args.cache_hit_repeats or case.cache_hit_repeats
    prefix_tokens = args.prefix_tokens
    if prefix_tokens is None and use_case is not None:
        prefix_tokens = use_case.prefix_tokens
    n_gpu_layers = args.n_gpu_layers if args.n_gpu_layers is not None else case.n_gpu_layers
    if (
        prefix_tokens is not None
        or n_gpu_layers != case.n_gpu_layers
        or cache_hit_repeats != case.cache_hit_repeats
    ):
        return Case(
            key=case.key,
            family=case.family,
            model_id=case.model_id,
            model_path=case.model_path,
            payload=case.payload,
            layer_end=case.layer_end,
            activation_width=case.activation_width,
            ctx_size=max(case.ctx_size, prefix_tokens + 128) if prefix_tokens is not None else case.ctx_size,
            n_gpu_layers=n_gpu_layers,
            prefix_tokens=prefix_tokens if prefix_tokens is not None else case.prefix_tokens,
            cache_hit_repeats=cache_hit_repeats,
            stage_load_mode=case.stage_load_mode,
            state_layer_start=case.state_layer_start,
            state_layer_end=case.state_layer_end,
            state_stage_index=case.state_stage_index,
            resident_kv_bytes_per_token=case.resident_kv_bytes_per_token,
            skip_llama_server_reason=case.skip_llama_server_reason,
        )
    return case


def run_case(case: Case, args: argparse.Namespace, use_case: UseCase | None = None) -> dict[str, Any]:
    case_dir = args.output_dir / f"{case.key}-p{case.prefix_tokens}"
    if use_case is not None:
        case_dir = args.output_dir / use_case.key / f"{case.key}-p{case.prefix_tokens}"
    case_dir.mkdir(parents=True, exist_ok=True)
    row: dict[str, Any] = {
        "key": case.key,
        "family": case.family,
        "model_id": case.model_id,
        "model_path": str(case.model_path) if case.model_path else None,
        "payload": case.payload,
        "prefix_tokens": case.prefix_tokens,
        "stage_load_mode": case.stage_load_mode,
        "state_layer_start": case.state_layer_start,
        "state_layer_end": case.state_layer_end or case.layer_end,
        "case": asdict(case) | {"model_path": str(case.model_path) if case.model_path else None},
    }
    if use_case is not None:
        row["use_case"] = use_case.key
        row["use_case_label"] = use_case.label
        row["use_case_source"] = {
            "dataset": use_case.source_dataset,
            "config": use_case.source_config,
            "split": use_case.source_split,
            "row_idx": use_case.source_row,
        }
    if case.model_path is None or not case.model_path.exists():
        row["skippy"] = {"status": "missing-model"}
        row["llama_server"] = {"status": "missing-model"}
        row["notes"] = "No local full GGUF available."
        return row

    print(f"==> {case.key}: Skippy {case.payload}", flush=True)
    try:
        skippy = run_correctness(case, args, case_dir, use_case.prompt if use_case is not None else None)
    except subprocess.TimeoutExpired as exc:
        skippy = {"status": "timeout", "timeout_secs": args.correctness_timeout_secs, "cmd": exc.cmd}
    except Exception as exc:  # noqa: BLE001 - benchmark continues across families.
        skippy = {"status": "error", "error": str(exc)}
    row["skippy"] = skippy
    row["benchmark_prompt_token_count"] = skippy.get("benchmark_prompt_token_count")

    if skippy.get("status") != "pass":
        row["llama_server"] = {"status": "skipped"}
        row["notes"] = "Skipped llama-server baseline because production cache correctness did not pass."
        return row
    if case.skip_llama_server_reason:
        row["llama_server"] = {"status": "skipped", "reason": case.skip_llama_server_reason}
        row["notes"] = case.skip_llama_server_reason
        return row
    if case.stage_load_mode != "runtime-slice":
        row["llama_server"] = {"status": "skipped", "reason": "llama-server requires a full GGUF"}
        row["notes"] = "llama-server baseline skipped because this case uses a layer package."
        return row
    if args.skip_llama_server:
        row["llama_server"] = {"status": "skipped"}
        row["notes"] = "llama-server baseline skipped by request."
        return row

    print(f"==> {case.key}: llama-server baseline", flush=True)
    try:
        row["llama_server"] = run_llama_server(case, skippy["benchmark_prompt_text"], args, case_dir)
        row["notes"] = ""
    except subprocess.TimeoutExpired as exc:
        row["llama_server"] = {"status": "timeout", "timeout_secs": args.request_timeout_secs, "cmd": exc.cmd}
        row["notes"] = "llama-server timed out."
    except Exception as exc:  # noqa: BLE001 - benchmark continues across families.
        row["llama_server"] = {"status": "error", "error": str(exc)}
        row["notes"] = "llama-server baseline failed."
    return row


def parse_concurrency_sweep(value: str) -> list[int]:
    if not value:
        return [1, 2, 4, 8, 16, 32, 64]
    levels = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        level = int(raw)
        if level <= 0:
            raise SystemExit("--concurrency-sweep values must be positive")
        levels.append(level)
    if not levels:
        raise SystemExit("--concurrency-sweep did not contain any levels")
    return levels


def calculate_goodput(
    results: list[dict[str, Any]],
    makespan_ms: float,
    ttft_slo_ms: float,
    tpot_slo_ms: float,
) -> dict[str, Any]:
    """Calculate goodput metrics from per-request results."""
    if not results:
        return {
            "goodput_rps": 0.0,
            "throughput_rps": 0.0,
            "output_tokens_per_second": 0.0,
            "successful_requests": 0,
            "failed_requests": 0,
            "ttft_p50": None,
            "ttft_p99": None,
            "tpot_p50": None,
            "tpot_p99": None,
            "latency_p50": None,
            "latency_p99": None,
        }

    successful = [r for r in results if "error" not in r]
    if not successful:
        empty = calculate_goodput([], makespan_ms, ttft_slo_ms, tpot_slo_ms)
        empty["failed_requests"] = len(results)
        return empty

    ttfts = [r.get("ttft_ms") for r in successful if r.get("ttft_ms") is not None]
    tpots = [r.get("tpot_ms") for r in successful if r.get("tpot_ms") is not None]
    latencies = [r["elapsed_ms"] for r in successful if isinstance(r.get("elapsed_ms"), (int, float))]
    workload_seconds = makespan_ms / 1000.0

    # Goodput: requests/sec that meet SLO
    if ttfts and tpots:
        good = sum(
            1
            for r in successful
            if r.get("ttft_ms", float("inf")) <= ttft_slo_ms
            and r.get("tpot_ms", float("inf")) <= tpot_slo_ms
        )
        goodput_rps = good / workload_seconds if workload_seconds > 0 else 0.0
    else:
        goodput_rps = 0.0

    def percentile(values: list[float], p: float) -> float | None:
        if not values:
            return None
        sorted_vals = sorted(values)
        idx = int(len(sorted_vals) * p / 100)
        idx = min(idx, len(sorted_vals) - 1)
        return sorted_vals[idx]
    
    output_tokens = sum(
        int(r.get("tokens_predicted", 0))
        for r in successful
        if isinstance(r.get("tokens_predicted"), int)
    )
    return {
        "goodput_rps": goodput_rps,
        "ttft_p50": percentile(ttfts, 50),
        "ttft_p99": percentile(ttfts, 99),
        "tpot_p50": percentile(tpots, 50),
        "tpot_p99": percentile(tpots, 99),
        "latency_p50": percentile(latencies, 50),
        "latency_p99": percentile(latencies, 99),
        "throughput_rps": len(successful) / workload_seconds if workload_seconds > 0 else 0.0,
        "output_tokens_per_second": output_tokens / workload_seconds if workload_seconds > 0 else 0.0,
        "successful_requests": len(successful),
        "failed_requests": len(results) - len(successful),
    }


def run_concurrent_benchmark(
    case: Case,
    prompt: str,
    concurrency: int,
    num_requests: int,
    port: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Run concurrent benchmark at a specific concurrency level."""
    results, makespan_ms = run_concurrent_requests(
        prompt,
        concurrency,
        num_requests,
        port,
        args,
    )
    return {
        "case": case.key,
        "concurrency": concurrency,
        "requests": num_requests,
        "makespan_ms": makespan_ms,
        "per_request": results,
        "metrics": calculate_goodput(
            results,
            makespan_ms,
            args.ttft_slo_ms,
            args.tpot_slo_ms,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/skippy-cache-production-bench"))
    parser.add_argument("--case", action="append", help="Run only the named case; may be repeated.")
    parser.add_argument("--skip-llama-server", action="store_true")
    parser.add_argument("--llama-server-bin", type=Path, default=REPO / ".deps/llama-build/build-stage-abi-cpu/bin/llama-server")
    parser.add_argument(
        "--old-skippy-server-bin",
        type=Path,
        help="Old-serving-path skippy-server binary used for the cutover comparison.",
    )
    parser.add_argument(
        "--new-skippy-server-bin",
        type=Path,
        help="New scheduler-serving-path skippy-server binary used for the cutover comparison.",
    )
    parser.add_argument("--skippy-correctness-bin", type=Path, default=REPO / "target/debug/skippy-correctness")
    parser.add_argument("--llama-stage-build-dir", type=Path, default=REPO / ".deps/llama-build/build-stage-abi-cpu")
    parser.add_argument("--correctness-timeout-secs", type=int, default=900)
    parser.add_argument("--server-startup-timeout-secs", type=int, default=600)
    parser.add_argument("--request-timeout-secs", type=int, default=600)
    parser.add_argument("--llama-repeats", type=int, default=3)
    parser.add_argument("--cache-hit-repeats", type=int, help="Override Skippy cache-hit repeats for every selected case.")
    parser.add_argument("--llama-parallel", type=int, default=1)
    parser.add_argument("--runtime-lane-count", type=int)
    parser.add_argument("--n-gpu-layers", type=int, help="Override n_gpu_layers for every selected case.")
    parser.add_argument("--borrow-resident-hits", action="store_true")
    parser.add_argument("--cache-decoded-result-hits", action="store_true")
    parser.add_argument("--prefix-tokens", type=int, help="Override the production prefix-token count for every selected case.")
    parser.add_argument("--use-case", action="append", help="Run one named use case from the corpus; use 'all' for every use case.")
    parser.add_argument(
        "--use-case-corpus",
        type=Path,
        default=REPO / "evals/skippy-usecase-corpus.json",
        help="JSON corpus with HF-derived benchmark use-case prompts.",
    )
    parser.add_argument(
        "--prefix-token-sweep",
        help="Comma-separated prefix-token sizes to run as one benchmark sweep, for example 512,2048,8192.",
    )
    parser.add_argument(
        "--concurrency-sweep",
        help="Comma-separated concurrency levels for sweep, e.g., 1,2,4,8,16,32,64. Default: 1,2,4,8,16,32,64.",
        default="1,2,4,8,16,32,64",
    )
    parser.add_argument(
        "--concurrent-requests",
        type=int,
        default=64,
        help="Number of requests per concurrency level.",
    )
    parser.add_argument(
        "--concurrent-output-tokens",
        type=int,
        default=32,
        help="Generated tokens requested from each old/new Skippy benchmark request.",
    )
    parser.add_argument(
        "--serving-ctx-size",
        type=int,
        help="Shared native context size for old/new Skippy sweeps; defaults to case ctx_size multiplied by the lane ceiling.",
    )
    parser.add_argument(
        "--ttft-slo-ms",
        type=int,
        default=2000,
        help="TTFT SLO in ms for goodput calculation.",
    )
    parser.add_argument(
        "--tpot-slo-ms",
        type=int,
        default=100,
        help="TPOT SLO in ms for goodput calculation.",
    )
    args = parser.parse_args()

    if (args.old_skippy_server_bin is None) != (args.new_skippy_server_bin is None):
        raise SystemExit(
            "--old-skippy-server-bin and --new-skippy-server-bin must be provided together"
        )
    for label, binary in (
        ("old", args.old_skippy_server_bin),
        ("new", args.new_skippy_server_bin),
    ):
        if binary is not None and not binary.is_file():
            raise SystemExit(f"{label} Skippy server binary does not exist: {binary}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = CASES
    if args.case:
        wanted = set(args.case)
        selected = [case for case in CASES if case.key in wanted]
        missing = wanted - {case.key for case in selected}
        if missing:
            raise SystemExit(f"unknown case(s): {', '.join(sorted(missing))}")

    selected_use_cases: list[UseCase | None] = [None]
    if args.use_case:
        use_cases = load_use_cases(args.use_case_corpus)
        wanted_use_cases = set(args.use_case)
        if "all" in wanted_use_cases:
            selected_use_cases = use_cases
        else:
            selected_use_cases = [use_case for use_case in use_cases if use_case.key in wanted_use_cases]
            missing = wanted_use_cases - {use_case.key for use_case in selected_use_cases}
            if missing:
                raise SystemExit(f"unknown use case(s): {', '.join(sorted(missing))}")

    prefix_sweep = parse_prefix_sweep(args.prefix_token_sweep)
    if args.prefix_tokens is not None:
        prefix_sweep = [args.prefix_tokens]
    
    concurrency_sweep = parse_concurrency_sweep(args.concurrency_sweep)
    
    results = []
    benchmark_inputs: list[tuple[Case, UseCase | None, dict[str, Any]]] = []
    for prefix_tokens in prefix_sweep:
        args.prefix_tokens = prefix_tokens
        for use_case in selected_use_cases:
            for case in selected:
                effective_case = resolve_case(case, args, use_case)
                row = run_case(effective_case, args, use_case)
                results.append(row)
                benchmark_inputs.append((effective_case, use_case, row))
                (args.output_dir / "production-cache-bench.json").write_text(json.dumps(results, indent=2))
                (args.output_dir / "production-cache-bench.md").write_text(markdown_table(results))
    
    # Run concurrency sweep for each case
    concurrency_results = []
    for case, use_case, row in benchmark_inputs:
        if case.model_path is None or not case.model_path.exists():
            continue
        if case.stage_load_mode != "runtime-slice":
            continue
        prompt = row.get("skippy", {}).get("benchmark_prompt_text")
        if not isinstance(prompt, str) or not prompt:
            continue

        port = free_port()
        use_case_suffix = f"-{use_case.key}" if use_case is not None else ""
        log_path = (
            args.output_dir
            / f"concurrency-{case.key}-p{case.prefix_tokens}{use_case_suffix}.log"
        )
        with log_path.open("w") as log:
            cmd = [
                str(args.llama_server_bin),
                "--model",
                str(case.model_path),
                "--ctx-size",
                str(case.ctx_size),
                "--n-gpu-layers",
                str(case.n_gpu_layers),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--parallel",
                str(args.llama_parallel),
                "--no-webui",
            ]
            proc = subprocess.Popen(
                cmd,
                cwd=REPO,
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            try:
                wait_ready(port, proc, args.server_startup_timeout_secs)

                case_concurrency_results = []
                for concurrency in concurrency_sweep:
                    print(f"==> {case.key}: concurrency={concurrency}", flush=True)
                    request_count = max(args.concurrent_requests, concurrency)
                    concurrent_results, makespan_ms = run_concurrent_requests(
                        prompt,
                        concurrency,
                        request_count,
                        port,
                        args,
                    )
                    goodput = calculate_goodput(
                        concurrent_results,
                        makespan_ms,
                        args.ttft_slo_ms,
                        args.tpot_slo_ms,
                    )
                    case_concurrency_results.append(
                        {
                            "concurrency": concurrency,
                            "requests": request_count,
                            "makespan_ms": makespan_ms,
                            "per_request": concurrent_results,
                            "goodput": goodput,
                        }
                    )

                concurrency_results.append(
                    {
                        "case": case.key,
                        "prefix_tokens": case.prefix_tokens,
                        "use_case": use_case.key if use_case is not None else None,
                        "concurrency_sweep": case_concurrency_results,
                    }
                )
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=5)

    serving_path_results = []
    if args.old_skippy_server_bin is not None and args.new_skippy_server_bin is not None:
        for case, use_case, row in benchmark_inputs:
            if case.model_path is None or not case.model_path.exists():
                continue
            if case.stage_load_mode != "runtime-slice":
                continue
            prompt = row.get("skippy", {}).get("benchmark_prompt_text")
            if not isinstance(prompt, str) or not prompt:
                continue
            suffix = f"-{use_case.key}" if use_case is not None else ""
            comparison_dir = (
                args.output_dir
                / f"serving-path-{case.key}-p{case.prefix_tokens}{suffix}"
            )
            comparison_dir.mkdir(parents=True, exist_ok=True)
            old_sweep = run_skippy_serving_path_sweep(
                "old",
                args.old_skippy_server_bin,
                case,
                prompt,
                concurrency_sweep,
                args,
                comparison_dir,
            )
            new_sweep = run_skippy_serving_path_sweep(
                "new",
                args.new_skippy_server_bin,
                case,
                prompt,
                concurrency_sweep,
                args,
                comparison_dir,
            )
            comparison = {
                "case": case.key,
                "family": case.family,
                "model_id": case.model_id,
                "prefix_tokens": case.prefix_tokens,
                "use_case": use_case.key if use_case is not None else None,
                "old": old_sweep,
                "new": new_sweep,
                "output_parity": serving_path_output_parity(old_sweep, new_sweep),
            }
            serving_path_results.append(comparison)
            (args.output_dir / "serving-path-comparison.json").write_text(
                json.dumps(serving_path_results, indent=2)
            )

    print(markdown_table(results))
    print(f"Wrote {args.output_dir / 'production-cache-bench.json'}")
    print(f"Wrote {args.output_dir / 'production-cache-bench.md'}")
    
    if concurrency_results:
        (args.output_dir / "concurrency-sweep.json").write_text(json.dumps(concurrency_results, indent=2))
        print(f"Wrote {args.output_dir / 'concurrency-sweep.json'}")
    if serving_path_results:
        print(f"Wrote {args.output_dir / 'serving-path-comparison.json'}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
