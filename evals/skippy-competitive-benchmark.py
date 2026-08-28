#!/usr/bin/env python3
"""Run and report the pinned Mesh-versus-llama.cpp competitive benchmark."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import html
import http.client
import json
import os
import platform as platform_module
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence


REPO = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO / "evals/skippy-competitive-benchmark.json"
PROMPT_GENERATOR = REPO / "evals/skippy-agentic-prompt-manifest.py"
ARMS = ("llama", "mesh")
CELL_RE = re.compile(r"tg-(\d+)-c-(\d+)\.json$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def directory_sha256(path: Path) -> str:
    """Hash relative paths and file bytes without depending on mtimes."""
    digest = hashlib.sha256()
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    if not files:
        raise RuntimeError(f"directory contains no files: {path}")
    for candidate in files:
        relative = candidate.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256(candidate)))
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    temporary.replace(path)


def load_config(path: Path) -> dict[str, Any]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError("competitive benchmark schema_version must be 1")
    baseline = document.get("baseline", {})
    if len(baseline.get("llama_cpp_revision", "")) != 40:
        raise ValueError("baseline needs a pinned llama.cpp revision")
    if not baseline.get("llama_benchy_version"):
        raise ValueError("baseline needs a pinned llama-benchy version")
    concurrency = document.get("concurrency")
    if concurrency != [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        raise ValueError("concurrency must be exactly 1/2/4/8/16/32/64/128/256")
    models = document.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("at least one model is required")
    keys: set[str] = set()
    for model in models:
        required = {
            "key",
            "family",
            "repo",
            "revision",
            "filename",
            "model_id",
            "sha256",
            "tokenizer_sha256",
            "layer_end",
            "synthetic_context_size",
            "cache_payload",
        }
        missing = required - set(model)
        if missing:
            raise ValueError(f"model is missing {', '.join(sorted(missing))}")
        if model["key"] in keys:
            raise ValueError(f"duplicate model key: {model['key']}")
        keys.add(model["key"])
        if (
            len(model["revision"]) != 40
            or len(model["sha256"]) != 64
            or len(model["tokenizer_sha256"]) != 64
        ):
            raise ValueError(
                f"model {model['key']} needs pinned model and tokenizer inputs"
            )
    thoughtworks = document.get("thoughtworks", {})
    dataset = thoughtworks.get("dataset", {})
    selection = thoughtworks.get("selection", {})
    if len(dataset.get("revision", "")) != 40 or len(dataset.get("sha256", "")) != 64:
        raise ValueError("Thoughtworks dataset needs a pinned revision and SHA-256")
    if len(selection.get("manifest_sha256", "")) != 64:
        raise ValueError("Thoughtworks selection needs a pinned manifest SHA-256")
    if len(selection.get("rows", [])) != selection.get("families"):
        raise ValueError("Thoughtworks selection must pin one row per family")
    expected_prompts = selection.get("families", 0) * selection.get(
        "requests_per_family", 0
    )
    if expected_prompts < max(concurrency):
        raise ValueError("Thoughtworks manifest must contain a complete c256 wave")
    return document


def selected_models(config: dict[str, Any], keys: Sequence[str]) -> list[dict[str, Any]]:
    if not keys:
        return list(config["models"])
    wanted = set(keys)
    found = [model for model in config["models"] if model["key"] in wanted]
    missing = wanted - {model["key"] for model in found}
    if missing:
        raise ValueError(f"unknown model(s): {', '.join(sorted(missing))}")
    return found


def prompt_limit(concurrency: int, minimum: int, available: int) -> int:
    target = max(minimum, concurrency)
    if target % concurrency:
        target += concurrency - target % concurrency
    return min(target, available)


def build_plan(
    config: dict[str, Any],
    platforms: Sequence[str],
    models: Sequence[dict[str, Any]],
    workloads: Sequence[str],
) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for platform in platforms:
        for model in models:
            if "synthetic" in workloads:
                for arm in ARMS:
                    for output_tokens in config["synthetic"]["output_tokens"]:
                        for concurrency in config["concurrency"]:
                            cells.append(
                                {
                                    "platform": platform,
                                    "model": model["key"],
                                    "workload": "synthetic",
                                    "arm": arm,
                                    "prompt_tokens": config["synthetic"]["prompt_tokens"],
                                    "output_tokens": output_tokens,
                                    "concurrency": concurrency,
                                }
                            )
            if "thoughtworks" in workloads:
                available = (
                    config["thoughtworks"]["selection"]["families"]
                    * config["thoughtworks"]["selection"]["requests_per_family"]
                )
                minimum = config["thoughtworks"]["minimum_prompts"]
                for index, concurrency in enumerate(config["concurrency"]):
                    arms = ARMS if index % 2 == 0 else tuple(reversed(ARMS))
                    for arm in arms:
                        cells.append(
                            {
                                "platform": platform,
                                "model": model["key"],
                                "workload": "thoughtworks",
                                "arm": arm,
                                "output_tokens": config["thoughtworks"]["output_tokens"],
                                "concurrency": concurrency,
                                "prompt_count": prompt_limit(
                                    concurrency, minimum, available
                                ),
                            }
                        )
    return {
        "schema_version": 1,
        "config_sha256": stable_hash(config),
        "platforms": list(platforms),
        "models": [model["key"] for model in models],
        "workloads": list(workloads),
        "cell_count": len(cells),
        "cells": cells,
    }


def run_checked(command: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[Any]:
    try:
        return subprocess.run(list(command), check=True, **kwargs)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"command failed ({error.returncode}): {' '.join(map(str, command))}"
        ) from error


def verify_file(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    actual = sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"{label} SHA-256 mismatch: expected={expected_sha256} actual={actual}"
        )


def prefetch(args: argparse.Namespace, config: dict[str, Any]) -> None:
    if shutil.which("hf") is None:
        raise RuntimeError("hf CLI is required for prefetch")
    args.model_root.mkdir(parents=True, exist_ok=True)
    for model in selected_models(config, args.model):
        model_dir = args.model_root / model["key"]
        model_dir.mkdir(parents=True, exist_ok=True)
        run_checked(
            [
                "hf",
                "download",
                model["repo"],
                model["filename"],
                "--revision",
                model["revision"],
                "--local-dir",
                str(model_dir),
            ]
        )
        run_checked(
            [
                "hf",
                "cache",
                "verify",
                model["repo"],
                "--revision",
                model["revision"],
                "--local-dir",
                str(model_dir),
            ]
        )
        verify_file(model_dir / model["filename"], model["sha256"], model["key"])

    dataset = config["thoughtworks"]["dataset"]
    args.dataset_root.mkdir(parents=True, exist_ok=True)
    run_checked(
        [
            "hf",
            "download",
            dataset["repo"],
            dataset["filename"],
            "--repo-type",
            "dataset",
            "--revision",
            dataset["revision"],
            "--local-dir",
            str(args.dataset_root),
        ]
    )
    run_checked(
        [
            "hf",
            "cache",
            "verify",
            dataset["repo"],
            "--repo-type",
            "dataset",
            "--revision",
            dataset["revision"],
            "--local-dir",
            str(args.dataset_root),
        ]
    )
    parquet = args.dataset_root / dataset["filename"]
    verify_file(parquet, dataset["sha256"], "Thoughtworks dataset")
    selection = config["thoughtworks"]["selection"]
    command = [
        sys.executable,
        str(PROMPT_GENERATOR),
        "--dataset-file",
        str(parquet),
        "--dataset-revision",
        dataset["revision"],
        "--output",
        str(args.manifest),
        "--families",
        str(selection["families"]),
        "--requests-per-family",
        str(selection["requests_per_family"]),
        "--min-isl",
        str(selection["min_isl"]),
        "--max-isl",
        str(selection["max_isl_exclusive"]),
        "--min-turns",
        str(selection["min_turns"]),
    ]
    for source in selection["sources"]:
        command.extend(["--source-dataset", source])
    run_checked(command)
    verify_manifest(args.manifest, config)
    print(args.manifest)


def verify_manifest(path: Path, config: dict[str, Any]) -> dict[str, Any]:
    selection = config["thoughtworks"]["selection"]
    verify_file(path, selection["manifest_sha256"], "Thoughtworks prompt manifest")
    document = json.loads(path.read_text(encoding="utf-8"))
    metadata = document.get("metadata", {})
    prompts = document.get("prompts")
    if metadata.get("rows") != selection["rows"]:
        raise RuntimeError("Thoughtworks prompt row provenance drifted")
    if not isinstance(prompts, list) or len(prompts) != (
        selection["families"] * selection["requests_per_family"]
    ):
        raise RuntimeError("Thoughtworks prompt count drifted")
    return document


def free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def raise_file_limit(minimum: int = 4096) -> None:
    """Keep the c256 client wave from exhausting a low interactive-shell limit."""
    try:
        import resource
    except ImportError:  # pragma: no cover - resource is present on benchmark hosts.
        return
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = min(max(soft, minimum), hard)
    if target > soft:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
    if target < 1024:
        raise RuntimeError(
            f"open-file limit {target} is too low for the c256 benchmark; need at least 1024"
        )


def git_head(path: Path) -> str:
    result = run_checked(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def write_stage_config(
    path: Path,
    model: dict[str, Any],
    model_path: Path,
    port: int,
    ctx_size: int,
    lanes: int,
    cache: bool,
) -> None:
    value: dict[str, Any] = {
        "run_id": f"competitive-{model['key']}",
        "topology_id": "competitive-single-stage",
        "model_id": model["model_id"],
        "model_path": str(model_path),
        "source_model_sha256": model["sha256"],
        "stage_id": "stage-0",
        "stage_index": 0,
        "layer_start": 0,
        "layer_end": model["layer_end"],
        "ctx_size": ctx_size,
        "lane_count": lanes,
        "n_batch": 2048,
        "n_ubatch": 512,
        "n_gpu_layers": -1,
        "cache_type_k": "f16",
        "cache_type_v": "f16",
        "filter_tensors_on_load": False,
        "native_mtp_enabled": False,
        "load_mode": "runtime-slice",
        "bind_addr": f"127.0.0.1:{port}",
        "upstream": None,
        "downstream": None,
    }
    if cache:
        value["kv_cache"] = {
            "mode": "lookup-record",
            "payload": model["cache_payload"],
            "max_entries": 64,
            "max_bytes": 0,
            "min_tokens": 64,
            "shared_prefix_stride_tokens": 128,
            "shared_prefix_record_limit": 2,
        }
    write_json(path, value)


def server_command(
    arm: str,
    args: argparse.Namespace,
    model: dict[str, Any],
    model_path: Path,
    stage_config: Path,
    port: int,
    ctx_size: int,
    lanes: int,
    output_tokens: int,
    prompt_cache: bool,
) -> list[str]:
    if arm == "mesh":
        return [
            str(args.mesh_binary),
            "serve-openai",
            "--config",
            str(stage_config),
            "--bind-addr",
            f"127.0.0.1:{port}",
            "--model-id",
            model["model_id"],
            "--generation-concurrency",
            str(lanes),
            "--default-max-tokens",
            str(output_tokens),
            "--telemetry-level",
            "summary" if prompt_cache else "off",
        ]
    command = [
        str(args.llama_binary),
        "--model",
        str(model_path),
        "--alias",
        model["model_id"],
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        str(ctx_size),
        "--parallel",
        str(lanes),
        "--batch-size",
        "2048",
        "--ubatch-size",
        "512",
        "--n-gpu-layers",
        "all",
        "--cont-batching",
        "--kv-unified",
        "--no-context-shift",
        "--metrics",
        "--no-webui",
    ]
    if not prompt_cache:
        command.append("--no-cache-prompt")
    return command


def wait_ready(port: int, process: subprocess.Popen[bytes], timeout: float = 600) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited during startup: {process.returncode}")
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
        try:
            connection.request("GET", "/v1/models")
            response = connection.getresponse()
            response.read()
            if response.status == 200:
                return
        except OSError:
            pass
        finally:
            connection.close()
        time.sleep(0.25)
    raise TimeoutError("server did not become ready")


def stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=15)


@contextmanager
def running_server(
    arm: str,
    args: argparse.Namespace,
    model: dict[str, Any],
    model_path: Path,
    output_dir: Path,
    ctx_size: int,
    lanes: int,
    output_tokens: int,
    prompt_cache: bool,
) -> Iterator[int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    port = free_port()
    stage_config = output_dir / "stage.json"
    write_stage_config(
        stage_config, model, model_path, port, ctx_size, lanes, prompt_cache
    )
    command = server_command(
        arm,
        args,
        model,
        model_path,
        stage_config,
        port,
        ctx_size,
        lanes,
        output_tokens,
        prompt_cache,
    )
    write_json(output_dir / "server-command.json", command)
    environment = os.environ.copy()
    environment["LLAMA_STAGE_BUILD_DIR"] = str(args.native_dir)
    environment["SKIPPY_TELEMETRY_STDERR"] = "1" if prompt_cache else "0"
    environment["SKIPPY_NATIVE_MTP_GREEDY_SAMPLING_FASTPATH"] = "1"
    environment.pop("SKIPPY_IDLE_ADMISSION_COALESCE_US", None)
    with (output_dir / "server.log").open("wb") as server_log:
        process = subprocess.Popen(
            command, stdout=server_log, stderr=subprocess.STDOUT, env=environment
        )
        try:
            wait_ready(port, process)
            yield port
        finally:
            stop_process(process)


def model_path(model_root: Path, model: dict[str, Any]) -> Path:
    return model_root / model["key"] / model["filename"]


def preflight_run(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    required = {
        "Mesh release binary": args.mesh_binary,
        "llama.cpp server": args.llama_binary,
        "native runtime directory": args.native_dir,
    }
    if "synthetic" in args.workload:
        required["llama-benchy"] = args.benchy
    for label, path in required.items():
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    models = selected_models(config, args.model)
    for model in models:
        verify_file(model_path(args.model_root, model), model["sha256"], model["key"])
        tokenizer = args.tokenizer_root / model["key"]
        if "synthetic" in args.workload and not tokenizer.exists():
            raise FileNotFoundError(f"tokenizer not found: {tokenizer}")
        if "synthetic" in args.workload:
            tokenizer_hash = directory_sha256(tokenizer)
            if tokenizer_hash != model["tokenizer_sha256"]:
                raise RuntimeError(
                    f"tokenizer {model['key']} SHA-256 mismatch: "
                    f"expected={model['tokenizer_sha256']} actual={tokenizer_hash}"
                )
    manifest = None
    if "thoughtworks" in args.workload:
        manifest = verify_manifest(args.manifest, config)
    llama_head = git_head(args.llama_root)
    if llama_head != config["baseline"]["llama_cpp_revision"]:
        raise RuntimeError(
            "raw llama.cpp revision mismatch: "
            f"expected={config['baseline']['llama_cpp_revision']} actual={llama_head}"
        )
    benchy_version = None
    benchy_sha256 = None
    if "synthetic" in args.workload:
        version_result = run_checked(
            [str(args.benchy), "--version"], text=True, capture_output=True
        )
        benchy_version = (version_result.stdout or version_result.stderr).strip()
        if config["baseline"]["llama_benchy_version"] not in benchy_version:
            raise RuntimeError(
                "llama-benchy version mismatch: "
                f"expected={config['baseline']['llama_benchy_version']} actual={benchy_version}"
            )
        benchy_sha256 = sha256(args.benchy)
    provenance = {
        "created_utc": utc_now(),
        "host": socket.gethostname(),
        "platform": args.platform,
        "platform_details": platform_module.platform(),
        "config_sha256": stable_hash(config),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "prompt_generator_sha256": sha256(PROMPT_GENERATOR),
        "mesh_head": git_head(args.mesh_root),
        "mesh_binary_sha256": sha256(args.mesh_binary),
        "llama_head": llama_head,
        "llama_binary_sha256": sha256(args.llama_binary),
        "llama_benchy_version": benchy_version,
        "llama_benchy_sha256": benchy_sha256,
        "native_runtime_directory_sha256": directory_sha256(args.native_dir),
        "models": {
            model["key"]: model["sha256"] for model in models
        },
        "tokenizers": {
            model["key"]: directory_sha256(args.tokenizer_root / model["key"])
            for model in models
            if "synthetic" in args.workload
        },
        "thoughtworks_manifest_sha256": (
            sha256(args.manifest) if manifest is not None else None
        ),
    }
    return provenance


def load_complete(path: Path, cell_hash: str) -> bool:
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("cell_sha256") == cell_hash
    except (OSError, json.JSONDecodeError):
        return False


def request_completion(
    port: int,
    model_id: str,
    prompt: str,
    output_tokens: int,
    stream: bool,
) -> dict[str, Any]:
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": output_tokens,
        "min_tokens": output_tokens,
        "ignore_eos": True,
        "temperature": 0,
        "seed": 42,
        "stream": stream,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    started = time.monotonic()
    first_token: float | None = None
    content: list[str] = []
    usage: dict[str, Any] = {}
    error: str | None = None
    status = 0
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    try:
        connection.request(
            "POST",
            "/v1/chat/completions",
            json.dumps(payload),
            {"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        status = response.status
        if status != 200:
            error = response.read(4096).decode(errors="replace")
        elif not stream:
            document = json.loads(response.read())
            choices = document.get("choices", [])
            if choices:
                text = choices[0].get("message", {}).get("content") or ""
                if text:
                    first_token = time.monotonic()
                    content.append(text)
            usage = document.get("usage", {})
        else:
            for raw_line in response:
                line = raw_line.strip()
                if not line.startswith(b"data: "):
                    continue
                body = line[6:]
                if body == b"[DONE]":
                    break
                try:
                    event = json.loads(body)
                except json.JSONDecodeError:
                    continue
                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]
                choices = event.get("choices", [])
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                text = delta.get("content") or delta.get("reasoning_content")
                if text:
                    if first_token is None:
                        first_token = time.monotonic()
                    content.append(text)
    except Exception as exc:  # noqa: BLE001 - preserve the failure in the artifact.
        error = str(exc)
    finally:
        connection.close()
    finished = time.monotonic()
    text = "".join(content)
    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    if error is None and completion_tokens and completion_tokens != output_tokens:
        error = f"expected {output_tokens} completion tokens, got {completion_tokens}"
    return {
        "status": status,
        "content": text,
        "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        "completion_tokens": completion_tokens,
        "ttft_ms": None if first_token is None else (first_token - started) * 1000,
        "elapsed_ms": (finished - started) * 1000,
        "error": error,
    }


def parity_probe(port: int, model_id: str, concurrency_values: Sequence[int]) -> dict[str, Any]:
    cells = []
    for concurrency in concurrency_values:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(
                    request_completion,
                    port,
                    model_id,
                    f"Reply with exactly one short sentence about scheduler parity. Case {index}.",
                    32,
                    False,
                )
                for index in range(concurrency)
            ]
            results = []
            for index, future in enumerate(futures):
                result = future.result()
                result["request_index"] = index
                results.append(result)
        cells.append({"concurrency": concurrency, "results": results})
    return {"cells": cells}


def run_synthetic_arm(
    args: argparse.Namespace,
    config: dict[str, Any],
    model: dict[str, Any],
    arm: str,
    provenance: dict[str, Any],
) -> None:
    synthetic = config["synthetic"]
    output_dir = args.output / "data" / args.platform / model["key"] / arm
    cell = {
        "platform": args.platform,
        "model": model["key"],
        "arm": arm,
        "workload": "synthetic",
        "config_sha256": stable_hash(config),
        "binary_sha256": provenance[f"{'mesh' if arm == 'mesh' else 'llama'}_binary_sha256"],
    }
    cell_hash = stable_hash(cell)
    if args.resume and load_complete(output_dir / "complete.json", cell_hash):
        print(f"SKIP synthetic {args.platform} {model['key']} {arm}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    path = model_path(args.model_root, model)
    with running_server(
        arm,
        args,
        model,
        path,
        output_dir,
        model["synthetic_context_size"],
        synthetic["active_lanes"],
        max(synthetic["output_tokens"]),
        False,
    ) as port:
        common = [
            str(args.benchy),
            "--base-url",
            f"http://127.0.0.1:{port}/v1",
            "--api-key",
            "EMPTY",
            "--model",
            model["model_id"],
            "--served-model-name",
            model["model_id"],
            "--tokenizer",
            str(args.tokenizer_root / model["key"]),
            "--pp",
            str(synthetic["prompt_tokens"]),
            "--exact-tg",
            "--extra-body",
            f"temperature={synthetic['temperature']},seed={synthetic['seed']}",
            "--depth",
            "0",
            "--runs",
            str(synthetic["runs"]),
            "--warmup-runs",
            "0",
            "--latency-mode",
            "none",
            "--skip-coherence",
            "--no-adapt-prompt",
            "--no-cache",
            "--no-warmup",
        ]
        warmup = common + ["--tg", "8", "--concurrency", "4", "--format", "json"]
        with (output_dir / "warmup.out").open("wb") as handle:
            subprocess.run(warmup, stdout=handle, stderr=subprocess.STDOUT, check=False)
        status_rows = []
        for output_tokens in synthetic["output_tokens"]:
            for concurrency in config["concurrency"]:
                stem = output_dir / f"tg-{output_tokens}-c-{concurrency}"
                command = common + [
                    "--tg",
                    str(output_tokens),
                    "--concurrency",
                    str(concurrency),
                    "--format",
                    "json",
                    "--save-result",
                    str(stem.with_suffix(".json")),
                    "--emit-progress",
                    str(stem.with_name(stem.name + "-progress.jsonl")),
                ]
                started = utc_now()
                with stem.with_suffix(".out").open("wb") as handle:
                    result = subprocess.run(
                        command, stdout=handle, stderr=subprocess.STDOUT, check=False
                    )
                status_rows.append(
                    {
                        "tg": output_tokens,
                        "concurrency": concurrency,
                        "exit_code": result.returncode,
                        "started_utc": started,
                        "finished_utc": utc_now(),
                    }
                )
        with (output_dir / "status.tsv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=status_rows[0], delimiter="\t")
            writer.writeheader()
            writer.writerows(status_rows)
        write_json(
            output_dir / "parity.json",
            parity_probe(port, model["model_id"], config["concurrency"]),
        )
    write_json(
        output_dir / "complete.json",
        {"cell_sha256": cell_hash, "completed_utc": utc_now(), "cell": cell},
    )


def checkpoint(prompt: str, fraction: float) -> str:
    target = max(1, int(len(prompt) * fraction))
    boundary = prompt.rfind("\n\n", 0, target)
    if boundary < target // 2:
        boundary = prompt.rfind("\n", 0, target)
    if boundary < target // 2:
        boundary = target
    return prompt[:boundary]


def run_trace_cell(
    args: argparse.Namespace,
    config: dict[str, Any],
    model: dict[str, Any],
    arm: str,
    concurrency: int,
    manifest: dict[str, Any],
    provenance: dict[str, Any],
) -> None:
    thoughtworks = config["thoughtworks"]
    prompts = manifest["prompts"]
    limit = prompt_limit(concurrency, thoughtworks["minimum_prompts"], len(prompts))
    selected = prompts[:limit]
    output_dir = (
        args.output
        / "trace"
        / args.platform
        / model["key"]
        / f"c-{concurrency}"
        / arm
    )
    cell = {
        "platform": args.platform,
        "model": model["key"],
        "arm": arm,
        "workload": "thoughtworks",
        "concurrency": concurrency,
        "prompt_count": limit,
        "config_sha256": stable_hash(config),
        "manifest_sha256": thoughtworks["selection"]["manifest_sha256"],
        "binary_sha256": provenance[f"{'mesh' if arm == 'mesh' else 'llama'}_binary_sha256"],
    }
    cell_hash = stable_hash(cell)
    if args.resume and load_complete(output_dir / "complete.json", cell_hash):
        print(f"SKIP thoughtworks {args.platform} {model['key']} c={concurrency} {arm}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    path = model_path(args.model_root, model)
    records: list[dict[str, Any]] = []
    with running_server(
        arm,
        args,
        model,
        path,
        output_dir,
        thoughtworks["context_size"],
        thoughtworks["active_lanes"],
        thoughtworks["output_tokens"],
        True,
    ) as port:
        measured_wall_ms = 0.0
        wall_started = time.monotonic()
        for group_index, group_start in enumerate(range(0, len(selected), concurrency)):
            group = selected[group_start : group_start + concurrency]
            for fraction in thoughtworks["warm_fractions"]:
                for local_index, item in enumerate(group):
                    warm_prompt = checkpoint(item["prompt"], fraction)
                    result = request_completion(
                        port,
                        model["model_id"],
                        warm_prompt,
                        thoughtworks["output_tokens"],
                        True,
                    )
                    result.update(
                        {
                            "phase": f"warm-{round(fraction * 100):02d}",
                            "group_index": group_index,
                            "request_index": group_start + local_index,
                            "family": item.get("family"),
                            "prompt_sha256": hashlib.sha256(
                                warm_prompt.encode()
                            ).hexdigest(),
                        }
                    )
                    records.append(result)
            measured_started = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [
                    executor.submit(
                        request_completion,
                        port,
                        model["model_id"],
                        item["prompt"],
                        thoughtworks["output_tokens"],
                        True,
                    )
                    for item in group
                ]
                measured = [future.result() for future in futures]
            group_wall_ms = (time.monotonic() - measured_started) * 1000
            measured_wall_ms += group_wall_ms
            for local_index, (item, result) in enumerate(zip(group, measured, strict=True)):
                result.update(
                    {
                        "phase": "measured-100",
                        "group_index": group_index,
                        "group_measured_wall_ms": group_wall_ms,
                        "request_index": group_start + local_index,
                        "family": item.get("family"),
                        "prompt_sha256": hashlib.sha256(
                            item["prompt"].encode()
                        ).hexdigest(),
                    }
                )
                records.append(result)
            write_json(output_dir / "requests.partial.json", records)
        wall_ms = (time.monotonic() - wall_started) * 1000
    measured = [record for record in records if record["phase"] == "measured-100"]
    successes = [record for record in measured if not record["error"]]
    output_count = sum(record["completion_tokens"] for record in successes)
    result = {
        "platform": args.platform,
        "model": model["key"],
        "arm": arm,
        "concurrency": concurrency,
        "prompt_count": limit,
        "successful_requests": len(successes),
        "failed_requests": len(measured) - len(successes),
        "output_tokens": output_count,
        "measured_wall_ms": measured_wall_ms,
        "total_wall_ms": wall_ms,
        "output_tokens_per_second": (
            output_count / (measured_wall_ms / 1000) if measured_wall_ms else 0.0
        ),
        "ttft_ms_mean": (
            sum(record["ttft_ms"] for record in successes if record["ttft_ms"] is not None)
            / max(1, sum(record["ttft_ms"] is not None for record in successes))
        ),
    }
    write_json(output_dir / "requests.json", records)
    write_json(output_dir / "result.json", result)
    write_json(
        output_dir / "complete.json",
        {"cell_sha256": cell_hash, "completed_utc": utc_now(), "cell": cell},
    )


def run_benchmark(args: argparse.Namespace, config: dict[str, Any]) -> None:
    raise_file_limit()
    provenance = preflight_run(args, config)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "benchmark-config.json", config)
    existing = args.output / "provenance" / f"{args.platform}.json"
    if existing.is_file():
        previous = json.loads(existing.read_text(encoding="utf-8"))
        immutable = (
            "config_sha256",
            "runner_sha256",
            "prompt_generator_sha256",
            "mesh_binary_sha256",
            "llama_binary_sha256",
            "native_runtime_directory_sha256",
            "tokenizers",
            "thoughtworks_manifest_sha256",
        )
        for key in immutable:
            if previous.get(key) != provenance.get(key):
                raise RuntimeError(f"refusing to mix artifacts with different {key}")
        provenance = {**previous, "last_resumed_utc": utc_now()}
    write_json(existing, provenance)
    models = selected_models(config, args.model)
    plan = build_plan(config, [args.platform], models, args.workload)
    write_json(args.output / "plans" / f"{args.platform}.json", plan)
    manifest = verify_manifest(args.manifest, config) if "thoughtworks" in args.workload else None
    for model in models:
        if "synthetic" in args.workload:
            for arm in ARMS:
                run_synthetic_arm(args, config, model, arm, provenance)
        if "thoughtworks" in args.workload:
            assert manifest is not None
            for index, concurrency in enumerate(config["concurrency"]):
                arms = ARMS if index % 2 == 0 else tuple(reversed(ARMS))
                for arm in arms:
                    run_trace_cell(
                        args, config, model, arm, concurrency, manifest, provenance
                    )
    write_artifact_hashes(args.output)


def write_artifact_hashes(root: Path) -> None:
    output = root / "artifact-sha256.txt"
    lines = []
    for path in sorted(path for path in root.rglob("*") if path.is_file()):
        if path == output:
            continue
        lines.append(f"{sha256(path)}  {path.relative_to(root)}")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_status(path: Path) -> dict[tuple[int, int], int]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            (int(row["tg"]), int(row["concurrency"])): int(row["exit_code"])
            for row in csv.DictReader(handle, delimiter="\t")
        }


def load_synthetic_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    data_root = root / "data"
    if not data_root.exists():
        return rows
    for arm_dir in sorted(data_root.glob("*/*/*")):
        if arm_dir.name not in ARMS:
            continue
        platform = arm_dir.parents[1].name
        model = arm_dir.parent.name
        status = read_status(arm_dir / "status.tsv")
        for path in sorted(arm_dir.glob("tg-*-c-*.json")):
            match = CELL_RE.match(path.name)
            if not match:
                continue
            output_tokens, concurrency = map(int, match.groups())
            document = json.loads(path.read_text(encoding="utf-8"))
            benchmark = document["benchmarks"][0]
            progress_path = path.with_name(path.stem + "-progress.jsonl")
            events = []
            if progress_path.exists():
                events = [
                    json.loads(line)
                    for line in progress_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            ends = [event for event in events if event.get("type") == "request_end"]
            successes = sum(not event.get("error") for event in ends)
            output_text = path.with_suffix(".out").read_text(
                encoding="utf-8", errors="replace"
            )
            http_429 = output_text.count("HTTP 429:")
            rows.append(
                {
                    "platform": platform,
                    "model": model,
                    "arm": arm_dir.name,
                    "tg": output_tokens,
                    "concurrency": concurrency,
                    "exit_code": status.get((output_tokens, concurrency), 0),
                    "throughput": float(benchmark["tg_throughput"]["mean"]),
                    "ttft_ms": float(benchmark["e2e_ttft"]["mean"]),
                    "successful_requests": successes,
                    "expected_requests": concurrency,
                    "failed_requests": max(concurrency - successes, http_429),
                    "http_429": http_429,
                    "complete": successes == concurrency,
                }
            )
    return rows


def load_trace_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((root / "trace").glob("*/*/c-*/*/result.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        result["complete"] = result["failed_requests"] == 0
        rows.append(result)
    return rows


def load_parity_rows(root: Path, concurrency_values: Sequence[int]) -> list[dict[str, Any]]:
    indexed: dict[tuple[str, str, str, int], dict[int, dict[str, Any]]] = {}
    for path in (root / "data").glob("*/*/*/parity.json"):
        arm = path.parent.name
        model = path.parent.parent.name
        platform = path.parent.parent.parent.name
        for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]:
            indexed[(platform, model, arm, cell["concurrency"])] = {
                result["request_index"]: result for result in cell["results"]
            }
    rows = []
    pairs = sorted({(key[0], key[1]) for key in indexed})
    for platform, model in pairs:
        for concurrency in concurrency_values:
            raw = indexed.get((platform, model, "llama", concurrency), {})
            mesh = indexed.get((platform, model, "mesh", concurrency), {})
            indexes = sorted(set(raw) | set(mesh))
            valid = matches = failures = 0
            for index in indexes:
                left = raw.get(index, {})
                right = mesh.get(index, {})
                if left.get("status") != 200 or right.get("status") != 200:
                    failures += 1
                else:
                    valid += 1
                    matches += int(left.get("content_sha256") == right.get("content_sha256"))
            rows.append(
                {
                    "platform": platform,
                    "model": model,
                    "concurrency": concurrency,
                    "matches": matches,
                    "valid_pairs": valid,
                    "failures": failures,
                    "exact_match_pct": 100 * matches / valid if valid else None,
                }
            )
    return rows


def escape(value: Any) -> str:
    return html.escape(str(value))


def polyline(points: Sequence[tuple[float, float]], color: str) -> str:
    if not points:
        return ""
    coordinates = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    circles = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="{color}"/>'
        for x, y in points
    )
    return f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="3"/>{circles}'


def svg_chart(
    title: str,
    rows: Sequence[dict[str, Any]],
    concurrency_values: Sequence[int],
    output: Path,
    delta: bool,
) -> None:
    width, height = 960, 520
    left, top, plot_width, plot_height = 90, 80, 800, 350
    indexed = {(row["arm"], row["concurrency"]): row for row in rows if row["complete"]}
    series: dict[str, list[tuple[int, float]]] = {"llama": [], "mesh": []}
    if delta:
        delta_points = []
        for concurrency in concurrency_values:
            raw = indexed.get(("llama", concurrency))
            mesh = indexed.get(("mesh", concurrency))
            if raw and mesh and raw["throughput"]:
                delta_points.append(
                    (concurrency, 100 * (mesh["throughput"] / raw["throughput"] - 1))
                )
        values = [value for _, value in delta_points] or [0.0]
        y_min, y_max = min(values + [0.0]), max(values + [0.0])
        padding = max((y_max - y_min) * 0.15, 1.0)
        y_min, y_max = y_min - padding, y_max + padding
    else:
        for arm in ARMS:
            for concurrency in concurrency_values:
                row = indexed.get((arm, concurrency))
                if row:
                    series[arm].append((concurrency, row["throughput"]))
        values = [value for points in series.values() for _, value in points] or [1.0]
        y_min, y_max = 0.0, max(values) * 1.08
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="{width / 2}" y="36" text-anchor="middle" font-family="sans-serif" font-size="22" font-weight="700">{escape(title)}</text>',
    ]
    span = max(y_max - y_min, 1e-9)
    for tick in range(6):
        value = y_min + span * tick / 5
        y = top + plot_height - plot_height * (value - y_min) / span
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#e2e8f0"/>')
        suffix = "%" if delta else ""
        parts.append(f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{value:+.1f}{suffix}</text>')
    for index, concurrency in enumerate(concurrency_values):
        x = left + plot_width * index / (len(concurrency_values) - 1)
        parts.append(f'<text x="{x:.1f}" y="{top + plot_height + 24}" text-anchor="middle" font-family="sans-serif" font-size="12">{concurrency}</text>')
    incomplete_mesh = {
        row["concurrency"]
        for row in rows
        if row["arm"] == "mesh" and not row["complete"]
    }
    for concurrency in incomplete_mesh:
        index = concurrency_values.index(concurrency)
        x = left + plot_width * index / (len(concurrency_values) - 1)
        y = top + plot_height - 8
        parts.append(
            f'<line x1="{x - 5:.1f}" y1="{y - 5:.1f}" x2="{x + 5:.1f}" y2="{y + 5:.1f}" stroke="#dc2626" stroke-width="2"/>'
            f'<line x1="{x - 5:.1f}" y1="{y + 5:.1f}" x2="{x + 5:.1f}" y2="{y - 5:.1f}" stroke="#dc2626" stroke-width="2"/>'
        )
    if delta:
        points = []
        for concurrency, value in delta_points:
            index = concurrency_values.index(concurrency)
            x = left + plot_width * index / (len(concurrency_values) - 1)
            y = top + plot_height - plot_height * (value - y_min) / span
            points.append((x, y))
        parts.append(polyline(points, "#dc2626"))
        parts.append('<text x="730" y="480" font-family="sans-serif" font-size="13" fill="#dc2626">Mesh delta</text>')
    else:
        for arm, color in (("llama", "#64748b"), ("mesh", "#0284c7")):
            points = []
            for concurrency, value in series[arm]:
                index = concurrency_values.index(concurrency)
                x = left + plot_width * index / (len(concurrency_values) - 1)
                y = top + plot_height - plot_height * (value - y_min) / span
                points.append((x, y))
            parts.append(polyline(points, color))
        parts.append('<text x="650" y="480" font-family="sans-serif" font-size="13" fill="#64748b">raw llama.cpp</text>')
        parts.append('<text x="790" y="480" font-family="sans-serif" font-size="13" fill="#0284c7">Mesh</text>')
    if incomplete_mesh:
        parts.append('<text x="90" y="480" font-family="sans-serif" font-size="13" fill="#dc2626">× incomplete Mesh cell</text>')
    parts.append('<text x="480" y="505" text-anchor="middle" font-family="sans-serif" font-size="13">Offered concurrency</text>')
    parts.append("</svg>")
    output.write_text("".join(parts), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def report(args: argparse.Namespace, config: dict[str, Any]) -> None:
    synthetic = load_synthetic_rows(args.artifact)
    trace = load_trace_rows(args.artifact)
    parity = load_parity_rows(args.artifact, config["concurrency"])
    summary = args.artifact / "summary"
    charts = summary / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    write_csv(summary / "synthetic.csv", synthetic)
    write_csv(summary / "thoughtworks.csv", trace)
    write_csv(summary / "parity.csv", parity)
    lines = [
        "# Mesh vs raw llama.cpp competitive benchmark",
        "",
        "Pinned matrix: CUDA and Metal; dense, MoE, and recurrent model families; offered concurrency 1/2/4/8/16/32/64/128/256.",
        "",
        "A throughput row is competitive only when the paired deterministic continuation parity gate passes and both arms complete every request in the cell.",
        "",
    ]
    labels = {model["key"]: model["family"] for model in config["models"]}
    platforms = sorted({row["platform"] for row in synthetic + trace})
    models = sorted({row["model"] for row in synthetic + trace})
    for platform in platforms:
        lines.extend([f"## {platform.upper()}", ""])
        for model in models:
            model_synthetic = [row for row in synthetic if row["platform"] == platform and row["model"] == model]
            model_trace = [row for row in trace if row["platform"] == platform and row["model"] == model]
            if not model_synthetic and not model_trace:
                continue
            gate = next((row for row in parity if row["platform"] == platform and row["model"] == model and row["concurrency"] == 1), None)
            gate_pass = bool(gate and gate["failures"] == 0 and gate["valid_pairs"] == 1 and gate["matches"] == 1)
            lines.extend([f"### {labels.get(model, model)}", "", f"Correctness gate: **{'PASS' if gate_pass else 'FAIL OR PENDING'}**.", ""])
            for output_tokens in config["synthetic"]["output_tokens"]:
                rows = [row for row in model_synthetic if row["tg"] == output_tokens]
                if not rows:
                    continue
                slug = f"{platform}-{model}-synthetic-tg-{output_tokens}"
                svg_chart(f"{platform.upper()} {labels.get(model, model)} — pp512/tg{output_tokens}", rows, config["concurrency"], charts / f"{slug}-throughput.svg", False)
                svg_chart(f"{platform.upper()} {labels.get(model, model)} — Mesh delta pp512/tg{output_tokens}", rows, config["concurrency"], charts / f"{slug}-delta.svg", True)
                lines.extend([f"![Synthetic throughput](charts/{slug}-throughput.svg)", "", f"![Synthetic Mesh delta](charts/{slug}-delta.svg)", ""])
            if model_trace:
                trace_rows = [{**row, "throughput": row["output_tokens_per_second"]} for row in model_trace]
                slug = f"{platform}-{model}-thoughtworks"
                svg_chart(f"{platform.upper()} {labels.get(model, model)} — Thoughtworks replay", trace_rows, config["concurrency"], charts / f"{slug}-throughput.svg", False)
                svg_chart(f"{platform.upper()} {labels.get(model, model)} — Thoughtworks Mesh delta", trace_rows, config["concurrency"], charts / f"{slug}-delta.svg", True)
                lines.extend([f"![Thoughtworks throughput](charts/{slug}-throughput.svg)", "", f"![Thoughtworks Mesh delta](charts/{slug}-delta.svg)", ""])
            lines.extend(["| workload | tg | concurrency | raw tok/s | Mesh tok/s | delta | status |", "|---|---:|---:|---:|---:|---:|---|"])
            groups: list[tuple[str, int, list[dict[str, Any]]]] = []
            for output_tokens in config["synthetic"]["output_tokens"]:
                groups.append(("pp512", output_tokens, [row for row in model_synthetic if row["tg"] == output_tokens]))
            groups.append(("Thoughtworks", config["thoughtworks"]["output_tokens"], [{**row, "throughput": row["output_tokens_per_second"]} for row in model_trace]))
            for workload, output_tokens, rows in groups:
                indexed = {(row["arm"], row["concurrency"]): row for row in rows}
                for concurrency in config["concurrency"]:
                    raw = indexed.get(("llama", concurrency))
                    mesh = indexed.get(("mesh", concurrency))
                    if not raw or not mesh:
                        continue
                    complete = raw["complete"] and mesh["complete"]
                    delta = 100 * (mesh["throughput"] / raw["throughput"] - 1) if complete and raw["throughput"] else None
                    status = "valid" if gate_pass and complete else "diagnostic"
                    lines.append(f"| {workload} | {output_tokens} | {concurrency} | {raw['throughput']:.2f} | {mesh['throughput']:.2f} | {'—' if delta is None else f'{delta:+.2f}%'} | {status} |")
            lines.append("")
    if parity:
        lines.extend(["## Exact continuation parity", "", "| platform | model | concurrency | matches | valid pairs | failures | match rate |", "|---|---|---:|---:|---:|---:|---:|"])
        for row in parity:
            rate = "n/a" if row["exact_match_pct"] is None else f"{row['exact_match_pct']:.2f}%"
            lines.append(f"| {row['platform']} | {row['model']} | {row['concurrency']} | {row['matches']} | {row['valid_pairs']} | {row['failures']} | {rate} |")
        lines.append("")
    (summary / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    write_artifact_hashes(args.artifact)
    print(summary / "REPORT.md")


def add_filters(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument(
        "--workload",
        action="append",
        choices=("synthetic", "thoughtworks"),
        default=[],
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="print the immutable benchmark matrix")
    plan.add_argument("--platform", action="append", choices=("cuda", "metal"), default=[])
    add_filters(plan)

    fetch = subparsers.add_parser("prefetch", help="fetch and verify pinned inputs")
    fetch.add_argument("--model-root", type=Path, required=True)
    fetch.add_argument("--dataset-root", type=Path, required=True)
    fetch.add_argument("--manifest", type=Path, required=True)
    fetch.add_argument("--model", action="append", default=[])

    run = subparsers.add_parser("run", help="run or resume one hardware platform")
    run.add_argument("--platform", choices=("cuda", "metal"), required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--model-root", type=Path, required=True)
    run.add_argument("--tokenizer-root", type=Path, required=True)
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--mesh-root", type=Path, required=True)
    run.add_argument("--mesh-binary", type=Path, required=True)
    run.add_argument("--native-dir", type=Path, required=True)
    run.add_argument("--llama-root", type=Path, required=True)
    run.add_argument("--llama-binary", type=Path, required=True)
    run.add_argument("--benchy", type=Path, required=True)
    run.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    add_filters(run)

    render = subparsers.add_parser("report", help="write CSV, SVG, and REPORT.md")
    render.add_argument("--artifact", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = load_config(args.config)
    if hasattr(args, "workload") and not args.workload:
        args.workload = ["synthetic", "thoughtworks"]
    if args.command == "plan":
        platforms = args.platform or ["cuda", "metal"]
        value = build_plan(
            config,
            platforms,
            selected_models(config, args.model),
            args.workload,
        )
        print(json.dumps(value, indent=2))
    elif args.command == "prefetch":
        prefetch(args, config)
    elif args.command == "run":
        run_benchmark(args, config)
    elif args.command == "report":
        report(args, config)
    else:  # pragma: no cover - argparse makes this unreachable.
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
