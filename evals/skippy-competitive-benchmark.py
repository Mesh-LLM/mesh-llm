#!/usr/bin/env python3
"""Run and report the pinned Mesh-versus-llama.cpp competitive benchmark.

Linux runs may opt into vLLM and SGLang comparison arms. Missing optional
runtimes or model inputs are recorded as skips and never weaken the required
raw llama.cpp-versus-Mesh matrix.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import html
import http.client
import json
import math
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
OPTIONAL_ARMS = ("vllm", "sglang")
ADAPTIVE_MESH_ARM = "mesh-adaptive"
KNOWN_ARMS = (*ARMS, ADAPTIVE_MESH_ARM, *OPTIONAL_ARMS)
REPORT_ARM_ORDER = (*ARMS, *OPTIONAL_ARMS)
REPORT_ARM_STYLES = {
    "llama": ("raw llama.cpp", "#64748b"),
    "mesh": ("Mesh", "#0284c7"),
    "vllm": ("vLLM", "#16a34a"),
    "sglang": ("SGLang", "#9333ea"),
}
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
            "vllm_hf_config",
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
            or len(model["vllm_hf_config"].get("revision", "")) != 40
            or len(model["vllm_hf_config"].get("sha256", "")) != 64
            or not model["vllm_hf_config"].get("repo")
        ):
            raise ValueError(
                f"model {model['key']} needs pinned model and tokenizer inputs"
            )
        for field in ("thoughtworks_context_size", "thoughtworks_active_lanes"):
            value = model.get(field)
            if value is not None and (not isinstance(value, int) or value <= 0):
                raise ValueError(f"model {model['key']} needs a positive {field}")
        comparison_support = model.get("comparison_support", {})
        if not isinstance(comparison_support, dict) or not set(
            comparison_support
        ).issubset(OPTIONAL_ARMS):
            raise ValueError(
                f"model {model['key']} comparison_support has an unknown backend"
            )
        for arm, support in comparison_support.items():
            if not isinstance(support, dict) or not isinstance(
                support.get("available"), bool
            ):
                raise ValueError(
                    f"model {model['key']} comparison_support for {arm} "
                    "needs an available boolean"
                )
            if not support["available"] and not support.get("reason"):
                raise ValueError(
                    f"model {model['key']} comparison_support for {arm} "
                    "needs an exclusion reason"
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


def thoughtworks_runtime_shape(
    thoughtworks: dict[str, Any], model: dict[str, Any]
) -> tuple[int, int]:
    return (
        model.get("thoughtworks_context_size", thoughtworks["context_size"]),
        model.get("thoughtworks_active_lanes", thoughtworks["active_lanes"]),
    )


def build_plan(
    config: dict[str, Any],
    platforms: Sequence[str],
    models: Sequence[dict[str, Any]],
    workloads: Sequence[str],
    arms: Sequence[str] = ARMS,
    arms_by_model: dict[str, Sequence[str]] | None = None,
) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for platform in platforms:
        for model in models:
            model_arms = tuple((arms_by_model or {}).get(model["key"], arms))
            if "synthetic" in workloads:
                for arm in model_arms:
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
                trace_context_size, trace_active_lanes = thoughtworks_runtime_shape(
                    config["thoughtworks"], model
                )
                available = (
                    config["thoughtworks"]["selection"]["families"]
                    * config["thoughtworks"]["selection"]["requests_per_family"]
                )
                minimum = config["thoughtworks"]["minimum_prompts"]
                for index, concurrency in enumerate(config["concurrency"]):
                    ordered_arms = (
                        model_arms
                        if index % 2 == 0
                        else tuple(reversed(model_arms))
                    )
                    for arm in ordered_arms:
                        cells.append(
                            {
                                "platform": platform,
                                "model": model["key"],
                                "workload": "thoughtworks",
                                "arm": arm,
                                "output_tokens": config["thoughtworks"]["output_tokens"],
                                "context_size": trace_context_size,
                                "active_lanes": trace_active_lanes,
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
        "arms": list(arms),
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


def served_model_id(arm: str, model: dict[str, Any]) -> str:
    model_id = model["model_id"]
    if arm == "sglang":
        return model_id.replace(":", "-")
    return model_id


def comparison_capacity_policy(
    args: argparse.Namespace, arm: str, ctx_size: int
) -> dict[str, Any] | None:
    if arm not in OPTIONAL_ARMS or not getattr(
        args, "capacity_match_comparison_kv", False
    ):
        return None
    return {"mode": "unified-total-kv-tokens", "token_capacity": ctx_size}


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
    if arm in ("mesh", ADAPTIVE_MESH_ARM):
        command = [
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
            "--generation-queue-capacity",
            "256",
            "--generation-admission-timeout-secs",
            "600",
            "--default-max-tokens",
            str(output_tokens),
            "--telemetry-level",
            "summary" if prompt_cache else "off",
        ]
        if arm == ADAPTIVE_MESH_ARM:
            command.extend(
                [
                    "--adaptive-generation-concurrency",
                    "--adaptive-generation-min-concurrency",
                    "1",
                ]
            )
        return command
    if arm == "vllm":
        command = [
            str(args.vllm_binary),
            "serve",
            str(model_path),
            "--tokenizer",
            str(args.tokenizer_root / model["key"]),
            "--hf-config-path",
            str(args.vllm_hf_config_root / model["key"]),
            "--served-model-name",
            model["model_id"],
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--max-model-len",
            str(ctx_size),
            "--max-num-seqs",
            str(lanes),
            "--load-format",
            "gguf",
            "--quantization",
            "gguf",
        ]
        command.append(
            "--enable-prefix-caching"
            if prompt_cache
            else "--no-enable-prefix-caching"
        )
        if comparison_capacity_policy(args, arm, ctx_size) is not None:
            block_size = 16
            block_count = (ctx_size + block_size - 1) // block_size
            command.extend(
                [
                    "--block-size",
                    str(block_size),
                    "--num-gpu-blocks-override",
                    str(block_count),
                ]
            )
        return command
    if arm == "sglang":
        sglang_path = sglang_model_path(args, model)
        command = [
            str(args.sglang_python),
            "-m",
            "sglang.launch_server",
            "--model-path",
            str(sglang_path),
            "--tokenizer-path",
            str(args.tokenizer_root / model["key"]),
            "--served-model-name",
            served_model_id(arm, model),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--context-length",
            str(ctx_size),
            "--max-running-requests",
            str(lanes),
            "--disable-prefill-cuda-graph",
        ]
        if sglang_path.is_file() and sglang_path.resolve().suffix.lower() == ".gguf":
            command.extend(["--load-format", "gguf", "--quantization", "gguf"])
        if not prompt_cache:
            command.append("--disable-radix-cache")
        if comparison_capacity_policy(args, arm, ctx_size) is not None:
            command.extend(["--max-total-tokens", str(ctx_size)])
        return command
    if arm != "llama":
        raise ValueError(f"unknown benchmark arm: {arm}")
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


def sglang_model_path(args: argparse.Namespace, model: dict[str, Any]) -> Path:
    if args.sglang_model_root is not None:
        return args.sglang_model_root / model["key"]
    return model_path(args.model_root, model)


def resolve_optional_comparisons(
    args: argparse.Namespace, models: Sequence[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    requested = tuple(dict.fromkeys(args.comparison_backend))
    status: dict[str, dict[str, Any]] = {}
    linux_cuda = args.platform == "cuda" and platform_module.system() == "Linux"
    for arm in requested:
        entry: dict[str, Any] = {
            "requested": True,
            "available": False,
            "models": {},
        }
        if not linux_cuda:
            entry["reason"] = "optional comparisons require a Linux CUDA run"
            status[arm] = entry
            continue
        if arm == "vllm":
            executable = args.vllm_binary or shutil.which("vllm")
            if not executable or not Path(executable).is_file():
                entry["reason"] = "vllm executable not found"
                status[arm] = entry
                continue
            args.vllm_binary = Path(executable)
            vllm_python = args.vllm_binary.parent / "python"
            if not vllm_python.is_file():
                entry["reason"] = (
                    "vllm executable must be inside a virtual environment so the "
                    "required GGUF plugin can be verified"
                )
                status[arm] = entry
                continue
            plugin = subprocess.run(
                [
                    str(vllm_python),
                    "-c",
                    "import importlib.metadata; print(importlib.metadata.version('vllm-gguf-plugin'))",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            if plugin.returncode != 0:
                entry["reason"] = "vllm-gguf-plugin is not installed in the vLLM environment"
                status[arm] = entry
                continue
            if args.vllm_hf_config_root is None:
                entry["reason"] = "--vllm-hf-config-root was not provided"
                status[arm] = entry
                continue
            model_status = {}
            for model in models:
                path = args.vllm_hf_config_root / model["key"] / "config.json"
                expected = model["vllm_hf_config"]["sha256"]
                actual = sha256(path) if path.is_file() else None
                support = model.get("comparison_support", {}).get("vllm")
                if support is not None and not support["available"]:
                    model_status[model["key"]] = {
                        "available": False,
                        "config_path": str(path.parent),
                        "config_sha256": actual,
                        "expected_sha256": expected,
                        "repo": model["vllm_hf_config"]["repo"],
                        "revision": model["vllm_hf_config"]["revision"],
                        "reason": support["reason"],
                        "source": "pinned-capability-exclusion",
                    }
                    continue
                model_status[model["key"]] = {
                    "available": actual == expected,
                    "config_path": str(path.parent),
                    "config_sha256": actual,
                    "expected_sha256": expected,
                    "repo": model["vllm_hf_config"]["repo"],
                    "revision": model["vllm_hf_config"]["revision"],
                    "reason": (
                        None
                        if actual == expected
                        else "config.json not found"
                        if actual is None
                        else "config.json SHA-256 mismatch"
                    ),
                }
            version = subprocess.run(
                [str(args.vllm_binary), "--version"],
                text=True,
                capture_output=True,
                check=False,
            )
            entry.update(
                {
                    "available": any(item["available"] for item in model_status.values()),
                    "executable": str(args.vllm_binary),
                    "executable_sha256": sha256(args.vllm_binary),
                    "version": (version.stdout or version.stderr).strip(),
                    "gguf_plugin_version": plugin.stdout.strip(),
                    "models": model_status,
                }
            )
            if not entry["available"]:
                entry["reason"] = "no verified vLLM Hugging Face config exists"
        elif arm == "sglang":
            python = args.sglang_python or Path(sys.executable)
            if not Path(python).is_file():
                entry["reason"] = "SGLang Python executable not found"
                status[arm] = entry
                continue
            args.sglang_python = Path(python)
            imported = subprocess.run(
                [
                    str(args.sglang_python),
                    "-c",
                    "import importlib.metadata; print(importlib.metadata.version('sglang'))",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            if imported.returncode != 0:
                entry["reason"] = "sglang module not importable"
                status[arm] = entry
                continue
            model_status = {}
            for model in models:
                path = sglang_model_path(args, model)
                available = path.is_file() or (
                    path.is_dir() and any(candidate.is_file() for candidate in path.rglob("*"))
                )
                support = model.get("comparison_support", {}).get("sglang")
                if support is not None and not support["available"]:
                    model_status[model["key"]] = {
                        "available": False,
                        "model_path": str(path),
                        "model_sha256": (
                            sha256(path)
                            if available and path.is_file()
                            else directory_sha256(path)
                            if available
                            else None
                        ),
                        "source": "pinned-capability-exclusion",
                        "reason": support["reason"],
                    }
                    continue
                model_status[model["key"]] = {
                    "available": available,
                    "model_path": str(path),
                    "model_sha256": (
                        sha256(path)
                        if available and path.is_file()
                        else directory_sha256(path)
                        if available
                        else None
                    ),
                    "source": (
                        "override"
                        if args.sglang_model_root is not None
                        else "pinned-baseline-gguf"
                    ),
                    "reason": None if available else "model path not found",
                }
            entry.update(
                {
                    "available": any(item["available"] for item in model_status.values()),
                    "python": str(args.sglang_python),
                    "python_sha256": sha256(args.sglang_python),
                    "version": imported.stdout.strip(),
                    "models": model_status,
                }
            )
            if not entry["available"]:
                entry["reason"] = "no SGLang model input exists"
        status[arm] = entry
    return status


def arms_for_model(
    args: argparse.Namespace,
    model: dict[str, Any],
    comparisons: dict[str, dict[str, Any]],
) -> tuple[str, ...]:
    arms = list(ARMS)
    if args.mesh_adaptive:
        arms.append(ADAPTIVE_MESH_ARM)
    for arm in args.comparison_backend:
        entry = comparisons.get(arm, {})
        model_entry = entry.get("models", {}).get(model["key"], {})
        if entry.get("available") and model_entry.get("available"):
            arms.append(arm)
    return tuple(dict.fromkeys(arms))


def required_comparison_errors(
    requested: Sequence[str],
    comparisons: dict[str, dict[str, Any]],
) -> list[str]:
    unavailable = []
    for arm in requested:
        entry = comparisons.get(arm, {})
        if not entry.get("available"):
            unavailable.append(f"{arm}: {entry.get('reason', 'unavailable')}")
            continue
        missing_models = [
            key
            for key, model_entry in entry.get("models", {}).items()
            if not model_entry.get("available")
            and model_entry.get("source") != "pinned-capability-exclusion"
        ]
        if missing_models:
            unavailable.append(
                f"{arm}: missing model inputs for {', '.join(missing_models)}"
            )
    return unavailable


def arm_runtime_sha256(
    arm: str, provenance: dict[str, Any], model_key: str
) -> str:
    if arm in ARMS:
        return provenance[f"{arm}_binary_sha256"]
    if arm == ADAPTIVE_MESH_ARM:
        return stable_hash(
            {
                "mesh_binary_sha256": provenance["mesh_binary_sha256"],
                "policy": "adaptive-generation-hardware-backpressure-v2",
            }
        )
    entry = provenance["optional_comparisons"][arm]
    identity = {
        "arm": arm,
        "version": entry.get("version"),
        "executable_sha256": entry.get("executable_sha256"),
        "python_sha256": entry.get("python_sha256"),
        "model": entry.get("models", {}).get(model_key),
    }
    return stable_hash(identity)


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
    optional_comparisons = resolve_optional_comparisons(args, models)
    if args.require_comparison_backends:
        unavailable = required_comparison_errors(
            args.comparison_backend, optional_comparisons
        )
        if unavailable:
            raise RuntimeError(
                "required comparison backends are unavailable: "
                + "; ".join(unavailable)
            )
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
        "optional_comparisons": optional_comparisons,
        "capacity_match_comparison_kv": args.capacity_match_comparison_kv,
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


def synthetic_benchy_common_command(
    args: argparse.Namespace,
    model: dict[str, Any],
    arm: str,
    benchmark_model_id: str,
    port: int,
    synthetic: dict[str, Any],
) -> list[str]:
    extra_body = [
        f"temperature={synthetic['temperature']}",
        f"seed={synthetic['seed']}",
    ]
    if arm in OPTIONAL_ARMS:
        # SGLang rejects return_token_ids=true on streaming chat completions.
        # vLLM can omit special-token chunks while still reporting the full
        # completion in usage, which makes token-id counting spuriously short.
        # The OpenAI usage count is the common contract for both comparison
        # engines and is what llama-benchy falls back to when this is false.
        extra_body.append("return_token_ids=false")
    return [
        str(args.benchy),
        "--base-url",
        f"http://127.0.0.1:{port}/v1",
        "--api-key",
        "EMPTY",
        "--model",
        benchmark_model_id,
        "--served-model-name",
        benchmark_model_id,
        "--tokenizer",
        str(args.tokenizer_root / model["key"]),
        "--pp",
        str(synthetic["prompt_tokens"]),
        "--exact-tg",
        "--extra-body",
        ",".join(extra_body),
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
        "--exit-on-first-fail",
        "--no-results-on-fail",
    ]


def validate_synthetic_cell(
    stem: Path, expected_requests: int, expected_output_tokens: int
) -> None:
    progress_path = stem.with_name(stem.name + "-progress.jsonl")
    if not progress_path.is_file():
        raise RuntimeError(f"missing progress stream for {stem.name}")
    events = [
        json.loads(line)
        for line in progress_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ends = [event for event in events if event.get("type") == "request_end"]
    failures = [event for event in ends if event.get("error")]
    if len(ends) != expected_requests or failures:
        raise RuntimeError(
            f"{stem.name} completed {len(ends)}/{expected_requests} requests "
            f"with {len(failures)} failures"
        )
    wrong_token_counts = [
        event.get("total_tokens")
        for event in ends
        if event.get("total_tokens") != expected_output_tokens
    ]
    if wrong_token_counts:
        raise RuntimeError(
            f"{stem.name} returned unexpected output token counts: "
            f"{sorted(set(wrong_token_counts), key=lambda value: str(value))}"
        )
    result_path = stem.with_suffix(".json")
    if not result_path.is_file():
        raise RuntimeError(f"missing result JSON for {stem.name}")
    document = json.loads(result_path.read_text(encoding="utf-8"))
    benchmark = document["benchmarks"][0]
    throughput = benchmark.get("tg_throughput", {}).get("mean")
    if (
        not isinstance(throughput, (int, float))
        or isinstance(throughput, bool)
        or not math.isfinite(throughput)
        or throughput <= 0
    ):
        raise RuntimeError(f"{stem.name} has invalid generation throughput: {throughput}")
    if benchmark.get("response_size") != expected_output_tokens:
        raise RuntimeError(
            f"{stem.name} result response_size={benchmark.get('response_size')} "
            f"expected={expected_output_tokens}"
        )


def write_synthetic_status(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fieldnames = (
        "tg",
        "concurrency",
        "exit_code",
        "failure",
        "started_utc",
        "finished_utc",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


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
        "binary_sha256": arm_runtime_sha256(arm, provenance, model["key"]),
        "comparison_capacity_policy": comparison_capacity_policy(
            args, arm, model["synthetic_context_size"]
        ),
    }
    cell_hash = stable_hash(cell)
    if args.resume and load_complete(output_dir / "complete.json", cell_hash):
        print(f"SKIP synthetic {args.platform} {model['key']} {arm}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    path = model_path(args.model_root, model)
    benchmark_model_id = served_model_id(arm, model)
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
        common = synthetic_benchy_common_command(
            args, model, arm, benchmark_model_id, port, synthetic
        )
        warmup = common + ["--tg", "8", "--concurrency", "4", "--format", "json"]
        with (output_dir / "warmup.out").open("wb") as handle:
            warmup_result = subprocess.run(
                warmup, stdout=handle, stderr=subprocess.STDOUT, check=False
            )
        if warmup_result.returncode != 0:
            raise RuntimeError(
                f"llama-benchy warmup failed for {arm} with exit "
                f"{warmup_result.returncode}; see {output_dir / 'warmup.out'}"
            )
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
                failure = ""
                if result.returncode != 0:
                    failure = f"llama-benchy exited {result.returncode}"
                else:
                    try:
                        validate_synthetic_cell(stem, concurrency, output_tokens)
                    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
                        failure = f"invalid llama-benchy artifact: {exc}"
                    except RuntimeError as exc:
                        failure = str(exc)
                status_rows.append(
                    {
                        "tg": output_tokens,
                        "concurrency": concurrency,
                        "exit_code": result.returncode,
                        "failure": failure,
                        "started_utc": started,
                        "finished_utc": utc_now(),
                    }
                )
                write_synthetic_status(output_dir / "status.tsv", status_rows)
                if failure:
                    raise RuntimeError(
                        f"synthetic benchmark failed for {arm} "
                        f"tg={output_tokens} c={concurrency}: {failure}"
                    )
        write_json(
            output_dir / "parity.json",
            parity_probe(port, benchmark_model_id, config["concurrency"]),
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
        "binary_sha256": arm_runtime_sha256(arm, provenance, model["key"]),
        "comparison_capacity_policy": comparison_capacity_policy(
            args,
            arm,
            thoughtworks_runtime_shape(thoughtworks, model)[0],
        ),
    }
    cell_hash = stable_hash(cell)
    if args.resume and load_complete(output_dir / "complete.json", cell_hash):
        print(f"SKIP thoughtworks {args.platform} {model['key']} c={concurrency} {arm}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    path = model_path(args.model_root, model)
    benchmark_model_id = served_model_id(arm, model)
    records: list[dict[str, Any]] = []
    context_size, active_lanes = thoughtworks_runtime_shape(thoughtworks, model)
    with running_server(
        arm,
        args,
        model,
        path,
        output_dir,
        context_size,
        active_lanes,
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
                        benchmark_model_id,
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
                        benchmark_model_id,
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
    write_json(
        args.output / "comparisons" / args.platform / "availability.json",
        provenance["optional_comparisons"],
    )
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
            "optional_comparisons",
            "capacity_match_comparison_kv",
        )
        for key in immutable:
            if previous.get(key) != provenance.get(key):
                raise RuntimeError(f"refusing to mix artifacts with different {key}")
        provenance = {**previous, "last_resumed_utc": utc_now()}
    write_json(existing, provenance)
    models = selected_models(config, args.model)
    arms_by_model = {
        model["key"]: arms_for_model(
            args, model, provenance["optional_comparisons"]
        )
        for model in models
    }
    planned_arms = tuple(
        dict.fromkeys(
            arm
            for model in models
            for arm in arms_by_model[model["key"]]
        )
    )
    plan = build_plan(
        config,
        [args.platform],
        models,
        args.workload,
        planned_arms,
        arms_by_model,
    )
    write_json(args.output / "plans" / f"{args.platform}.json", plan)
    manifest = verify_manifest(args.manifest, config) if "thoughtworks" in args.workload else None
    for model in models:
        if "synthetic" in args.workload:
            model_arms = arms_for_model(
                args, model, provenance["optional_comparisons"]
            )
            for arm in model_arms:
                run_synthetic_arm(args, config, model, arm, provenance)
        if "thoughtworks" in args.workload:
            assert manifest is not None
            for index, concurrency in enumerate(config["concurrency"]):
                model_arms = arms_for_model(
                    args, model, provenance["optional_comparisons"]
                )
                arms = model_arms if index % 2 == 0 else tuple(reversed(model_arms))
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
        if arm_dir.name not in KNOWN_ARMS:
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
            candidates = sorted(
                key[2]
                for key in indexed
                if key[0] == platform
                and key[1] == model
                and key[3] == concurrency
                and key[2] != "llama"
            )
            for candidate_arm in candidates:
                candidate = indexed.get(
                    (platform, model, candidate_arm, concurrency), {}
                )
                indexes = sorted(set(raw) | set(candidate))
                valid = matches = failures = 0
                for index in indexes:
                    left = raw.get(index, {})
                    right = candidate.get(index, {})
                    if left.get("status") != 200 or right.get("status") != 200:
                        failures += 1
                    else:
                        valid += 1
                        matches += int(
                            left.get("content_sha256")
                            == right.get("content_sha256")
                        )
                rows.append(
                    {
                        "platform": platform,
                        "model": model,
                        "arm": candidate_arm,
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
    active_arms = [
        arm
        for arm in REPORT_ARM_ORDER
        if any(row["arm"] == arm for row in rows)
    ]
    series: dict[str, list[tuple[int, float]]] = {
        arm: [] for arm in active_arms
    }
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
        for arm in active_arms:
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
        for arm in active_arms:
            _, color = REPORT_ARM_STYLES[arm]
            points = []
            for concurrency, value in series[arm]:
                index = concurrency_values.index(concurrency)
                x = left + plot_width * index / (len(concurrency_values) - 1)
                y = top + plot_height - plot_height * (value - y_min) / span
                points.append((x, y))
            parts.append(polyline(points, color))
        legend_start = 430 if len(active_arms) > 2 else 650
        legend_step = 125 if len(active_arms) > 2 else 140
        for index, arm in enumerate(active_arms):
            label, color = REPORT_ARM_STYLES[arm]
            x = legend_start + legend_step * index
            parts.append(
                f'<text x="{x}" y="480" font-family="sans-serif" '
                f'font-size="13" fill="{color}">{escape(label)}</text>'
            )
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
    provenance_files = sorted((args.artifact / "provenance").glob("*.json"))
    capacity_modes = {
        bool(json.loads(path.read_text(encoding="utf-8")).get(
            "capacity_match_comparison_kv", False
        ))
        for path in provenance_files
    }
    if capacity_modes == {True}:
        comparison_capacity_note = (
            "Capacity-matched comparison: vLLM and SGLang are capped to the "
            "same total KV-token capacity as the unified Mesh/raw context."
        )
    elif capacity_modes == {False} or not capacity_modes:
        comparison_capacity_note = (
            "Same-hardware comparison: vLLM and SGLang use their default paged "
            "KV capacity; compare peak VRAM separately from throughput."
        )
    else:
        comparison_capacity_note = (
            "Mixed comparison capacity policy detected across platform provenance; "
            "do not aggregate optional-backend results across platforms."
        )
    lines = [
        "# Skippy competitive benchmark",
        "",
        "Pinned matrix: CUDA and Metal; dense, MoE, and recurrent model families; offered concurrency 1/2/4/8/16/32/64/128/256.",
        "",
        "A throughput row is competitive only when the paired deterministic continuation parity gate passes and both arms complete every request in the cell.",
        "",
        comparison_capacity_note,
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
            gate = next((row for row in parity if row["platform"] == platform and row["model"] == model and row["arm"] == "mesh" and row["concurrency"] == 1), None)
            gate_pass = bool(gate and gate["failures"] == 0 and gate["valid_pairs"] == 1 and gate["matches"] == 1)
            lines.extend([f"### {labels.get(model, model)}", "", f"Mesh correctness gate: **{'PASS' if gate_pass else 'FAIL OR PENDING'}**.", ""])
            availability_path = (
                args.artifact
                / "comparisons"
                / platform
                / "availability.json"
            )
            if availability_path.is_file():
                availability = json.loads(
                    availability_path.read_text(encoding="utf-8")
                )
                exclusions = [
                    f"{REPORT_ARM_STYLES[arm][0]}: "
                    f"{entry['models'][model]['reason']}"
                    for arm, entry in availability.items()
                    if model in entry.get("models", {})
                    and not entry["models"][model].get("available")
                    and entry["models"][model].get("reason")
                ]
                if exclusions:
                    lines.extend(
                        [
                            "Pinned exact-input backend exclusions: "
                            + "; ".join(exclusions)
                            + ".",
                            "",
                        ]
                    )
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
            comparison_arms = [
                arm
                for arm in REPORT_ARM_ORDER
                if any(
                    row["arm"] == arm
                    for row in model_synthetic + model_trace
                )
            ]
            candidate_gates = {
                arm: any(
                    row["platform"] == platform
                    and row["model"] == model
                    and row["arm"] == arm
                    and row["concurrency"] == 1
                    and row["failures"] == 0
                    and row["valid_pairs"] == 1
                    and row["matches"] == 1
                    for row in parity
                )
                for arm in comparison_arms
                if arm != "llama"
            }
            arm_headers = " | ".join(
                f"{REPORT_ARM_STYLES[arm][0]} tok/s" for arm in comparison_arms
            )
            lines.extend(
                [
                    f"| workload | tg | concurrency | {arm_headers} | Mesh vs raw | status |",
                    "|---|---:|---:|"
                    + "---:|" * len(comparison_arms)
                    + "---:|---|",
                ]
            )
            groups: list[tuple[str, int, list[dict[str, Any]]]] = []
            for output_tokens in config["synthetic"]["output_tokens"]:
                groups.append(("pp512", output_tokens, [row for row in model_synthetic if row["tg"] == output_tokens]))
            groups.append(("Thoughtworks", config["thoughtworks"]["output_tokens"], [{**row, "throughput": row["output_tokens_per_second"]} for row in model_trace]))
            for workload, output_tokens, rows in groups:
                indexed = {(row["arm"], row["concurrency"]): row for row in rows}
                for concurrency in config["concurrency"]:
                    arm_rows = {
                        arm: indexed.get((arm, concurrency))
                        for arm in comparison_arms
                    }
                    raw = arm_rows.get("llama")
                    mesh = arm_rows.get("mesh")
                    if not raw or not mesh:
                        continue
                    base_complete = raw["complete"] and mesh["complete"]
                    complete = all(
                        row is not None and row["complete"]
                        for row in arm_rows.values()
                    )
                    delta = (
                        100 * (mesh["throughput"] / raw["throughput"] - 1)
                        if base_complete and raw["throughput"]
                        else None
                    )
                    gates_pass = all(candidate_gates.values())
                    status = "valid" if gates_pass and complete else "diagnostic"
                    throughput_cells = " | ".join(
                        "—"
                        if arm_rows[arm] is None
                        else f"{arm_rows[arm]['throughput']:.2f}"
                        for arm in comparison_arms
                    )
                    lines.append(
                        f"| {workload} | {output_tokens} | {concurrency} | "
                        f"{throughput_cells} | "
                        f"{'—' if delta is None else f'{delta:+.2f}%'} | "
                        f"{status} |"
                    )
            lines.append("")
            promotion_arms = sorted(
                {row["arm"] for row in model_synthetic + model_trace}
                - {"llama", "mesh"}
            )
            promotion_rows = []
            for candidate_arm in promotion_arms:
                synthetic_deltas = []
                trace_deltas = []
                complete = True
                for rows, delta_output, expected_keys in (
                    (
                        model_synthetic,
                        synthetic_deltas,
                        {
                            (concurrency, output_tokens)
                            for concurrency in config["concurrency"]
                            for output_tokens in config["synthetic"]["output_tokens"]
                        },
                    ),
                    (
                        [
                            {**row, "throughput": row["output_tokens_per_second"]}
                            for row in model_trace
                        ],
                        trace_deltas,
                        {
                            (concurrency, None)
                            for concurrency in config["concurrency"]
                        },
                    ),
                ):
                    indexed = {
                        (row["arm"], row["concurrency"], row.get("tg")): row
                        for row in rows
                    }
                    candidate_keys = {
                        (concurrency, tg)
                        for arm, concurrency, tg in indexed
                        if arm == candidate_arm
                    }
                    baseline_keys = {
                        (concurrency, tg)
                        for arm, concurrency, tg in indexed
                        if arm == "mesh"
                    }
                    if candidate_keys != expected_keys or baseline_keys != expected_keys:
                        complete = False
                    for concurrency, tg in sorted(candidate_keys):
                        candidate = indexed[(candidate_arm, concurrency, tg)]
                        baseline = indexed.get(("mesh", concurrency, tg))
                        if (
                            baseline is None
                            or not baseline["complete"]
                            or not candidate["complete"]
                            or not baseline["throughput"]
                        ):
                            complete = False
                            continue
                        delta_output.append(
                            100
                            * (
                                candidate["throughput"] / baseline["throughput"]
                                - 1
                            )
                        )
                candidate_gate = next(
                    (
                        row
                        for row in parity
                        if row["platform"] == platform
                        and row["model"] == model
                        and row["arm"] == candidate_arm
                        and row["concurrency"] == 1
                    ),
                    None,
                )
                gate_passed = bool(
                    candidate_gate
                    and candidate_gate["failures"] == 0
                    and candidate_gate["valid_pairs"] == 1
                    and candidate_gate["matches"] == 1
                )
                synthetic_mean = (
                    sum(synthetic_deltas) / len(synthetic_deltas)
                    if synthetic_deltas
                    else None
                )
                trace_mean = (
                    sum(trace_deltas) / len(trace_deltas) if trace_deltas else None
                )
                eligible = bool(
                    complete
                    and gate_passed
                    and synthetic_mean is not None
                    and trace_mean is not None
                    and synthetic_mean > 0
                    and trace_mean > 0
                )
                promotion_rows.append(
                    {
                        "arm": candidate_arm,
                        "synthetic_mean": synthetic_mean,
                        "trace_mean": trace_mean,
                        "eligible": eligible,
                        "gate": gate_passed,
                        "complete": complete,
                    }
                )
            if promotion_rows:
                eligible_rows = [row for row in promotion_rows if row["eligible"]]
                winner = (
                    max(
                        eligible_rows,
                        key=lambda row: row["synthetic_mean"] + row["trace_mean"],
                    )["arm"]
                    if eligible_rows
                    else None
                )
                lines.extend(
                    [
                        "#### Promotion gate",
                        "",
                        "A candidate is promotable only when c1 exact continuation matches raw llama.cpp, every comparable cell completes, and mean throughput beats fixed Mesh in both synthetic and Thoughtworks workloads.",
                        "",
                        "| candidate | synthetic vs Mesh | Thoughtworks vs Mesh | correctness | complete | decision |",
                        "|---|---:|---:|---|---|---|",
                    ]
                )
                for row in promotion_rows:
                    synthetic_delta = (
                        "n/a"
                        if row["synthetic_mean"] is None
                        else f"{row['synthetic_mean']:+.2f}%"
                    )
                    trace_delta = (
                        "n/a"
                        if row["trace_mean"] is None
                        else f"{row['trace_mean']:+.2f}%"
                    )
                    decision = (
                        "**PROMOTION CANDIDATE**"
                        if row["arm"] == winner
                        else "hold"
                    )
                    lines.append(
                        f"| {row['arm']} | {synthetic_delta} | {trace_delta} | "
                        f"{'pass' if row['gate'] else 'fail/pending'} | "
                        f"{'yes' if row['complete'] else 'no'} | {decision} |"
                    )
                lines.append("")
    if parity:
        lines.extend(["## Exact continuation parity", "", "| platform | model | candidate | concurrency | matches | valid pairs | failures | match rate |", "|---|---|---|---:|---:|---:|---:|---:|"])
        for row in parity:
            rate = "n/a" if row["exact_match_pct"] is None else f"{row['exact_match_pct']:.2f}%"
            lines.append(f"| {row['platform']} | {row['model']} | {row['arm']} | {row['concurrency']} | {row['matches']} | {row['valid_pairs']} | {row['failures']} | {rate} |")
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
    run.add_argument(
        "--mesh-adaptive",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="also benchmark Mesh with the staged adaptive generation permit controller",
    )
    run.add_argument(
        "--comparison-backend",
        action="append",
        choices=OPTIONAL_ARMS,
        default=[],
        help="optional Linux CUDA comparison arm; missing runtimes are recorded as skips",
    )
    run.add_argument("--vllm-binary", type=Path)
    run.add_argument(
        "--vllm-hf-config-root",
        type=Path,
        help="verified per-model Hugging Face configs required by the vLLM GGUF plugin",
    )
    run.add_argument("--sglang-python", type=Path)
    run.add_argument(
        "--sglang-model-root",
        type=Path,
        help="optional override containing one SGLang model input per key; defaults to the pinned GGUF inputs",
    )
    run.add_argument(
        "--require-comparison-backends",
        action="store_true",
        help=(
            "fail preflight instead of skipping a requested backend or an "
            "unpinned model incompatibility"
        ),
    )
    run.add_argument(
        "--capacity-match-comparison-kv",
        action="store_true",
        help=(
            "cap vLLM and SGLang to the same total KV-token capacity as the "
            "Mesh/raw unified context"
        ),
    )
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
