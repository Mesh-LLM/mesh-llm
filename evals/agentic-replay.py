#!/usr/bin/env python3
"""Agentic Replay: compare Mesh commits with ordered real-agent trajectories.

The runner creates detached worktrees, builds the release host and native
runtime for every requested ref, replays a deterministic subset of the pinned
Thoughtworks agentic-coding-trajectories corpus, and writes raw evidence,
tables, CSV, and dependency-free SVG charts.

Mesh is deliberately launched without context-size, lane-count, KV-budget, or
backend-tuning arguments. The only serving argument is ``--model``;
``--log-format json`` is observational. Client concurrency is offered by this
runner and is not a Mesh startup setting.
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
import re
import shutil
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


REPO = Path(__file__).resolve().parents[1]
COMPETITIVE_CONFIG = REPO / "evals/skippy-competitive-benchmark.json"
TRAJECTORY_GENERATOR = REPO / "evals/agentic-trajectory-manifest.py"
DEFAULT_BASE_URL = "http://127.0.0.1:9337/v1"
FORBIDDEN_STARTUP_OPTIONS = (
    "--ctx-size",
    "--generation-concurrency",
    "--generation-queue-capacity",
    "--max-vram",
    "--parallel",
)
COLORS = (
    "#0284c7",
    "#dc2626",
    "#16a34a",
    "#7c3aed",
    "#ea580c",
    "#0891b2",
)


@dataclass(frozen=True)
class RefSpec:
    label: str
    ref: str
    commit: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise RuntimeError(f"directory contains no files: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256(item)))
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    temporary.replace(path)


def slug(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    if not normalized:
        raise ValueError(f"cannot derive a safe label from {value!r}")
    return normalized


def git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def parse_ref_specs(repo: Path, values: Sequence[str]) -> list[RefSpec]:
    if len(values) < 2:
        raise ValueError("at least two --ref LABEL=GIT_REF values are required")
    specs: list[RefSpec] = []
    labels: set[str] = set()
    commits: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"ref must use LABEL=GIT_REF syntax: {value}")
        label, ref = value.split("=", 1)
        label, ref = slug(label), ref.strip()
        if not ref:
            raise ValueError(f"empty git ref for label {label}")
        if label in labels:
            raise ValueError(f"duplicate ref label: {label}")
        commit = git(repo, "rev-parse", f"{ref}^{{commit}}")
        if commit in commits:
            raise ValueError(f"multiple labels resolve to commit {commit}")
        labels.add(label)
        commits.add(commit)
        specs.append(RefSpec(label=label, ref=ref, commit=commit))
    return specs


def ab_order(specs: Sequence[RefSpec], passes: int) -> list[tuple[int, RefSpec]]:
    if passes <= 0:
        raise ValueError("passes must be positive")
    ordered: list[tuple[int, RefSpec]] = []
    for pass_index in range(passes):
        pass_specs = specs if pass_index % 2 == 0 else tuple(reversed(specs))
        ordered.extend((pass_index, spec) for spec in pass_specs)
    return ordered


class CommandLog:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: Path,
        log_path: Path,
        env: Optional[dict[str, str]] = None,
    ) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "started_at": utc_now(),
            "cwd": str(cwd),
            "command": list(command),
            "log": str(log_path),
        }
        with self.path.open("a", encoding="utf-8") as command_log:
            command_log.write(json.dumps(event, sort_keys=True) + "\n")
        with log_path.open("w", encoding="utf-8") as output:
            result = subprocess.run(
                list(command),
                cwd=cwd,
                env=env,
                text=True,
                stdout=output,
                stderr=subprocess.STDOUT,
            )
        event["completed_at"] = utc_now()
        event["exit_code"] = result.returncode
        with self.path.open("a", encoding="utf-8") as command_log:
            command_log.write(json.dumps(event, sort_keys=True) + "\n")
        if result.returncode:
            raise RuntimeError(
                f"command failed ({result.returncode}); see {log_path}: "
                + " ".join(command)
            )


def prepare_worktree(repo: Path, root: Path, spec: RefSpec) -> Path:
    path = root / f"{spec.label}-{spec.commit[:10]}"
    if path.exists():
        try:
            actual = git(path, "rev-parse", "HEAD")
        except (subprocess.CalledProcessError, FileNotFoundError) as error:
            raise RuntimeError(f"existing path is not a git worktree: {path}") from error
        if actual != spec.commit:
            raise RuntimeError(
                f"worktree {path} is at {actual}, expected {spec.commit}; "
                "choose a different --worktree-root"
            )
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "-C", str(repo), "worktree", "add", "--detach", str(path), spec.commit],
            check=True,
        )
    if git(path, "status", "--porcelain", "--untracked-files=no"):
        raise RuntimeError(f"benchmark worktree is dirty: {path}")
    return path


def runtime_backend_kind(backend: str) -> str:
    return {"cuda-blackwell": "cuda", "hip": "rocm"}.get(backend, backend)


def find_runtime(root: Path, backend: str) -> Path:
    candidates: list[Path] = []
    for manifest_path in root.glob("*/manifest.json"):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            kind = manifest["runtime"]["backend"]["kind"]
        except (KeyError, json.JSONDecodeError):
            continue
        if kind == runtime_backend_kind(backend):
            candidates.append(manifest_path.parent)
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one {backend} runtime under {root}, found {len(candidates)}"
        )
    return candidates[0]


def build_ref(
    spec: RefSpec,
    worktree: Path,
    backend: str,
    output: Path,
    commands: CommandLog,
    skip_build: bool,
) -> dict[str, Any]:
    binary = worktree / "target/release/mesh-llm"
    runtime_root = worktree / "dist/native-runtimes"
    if not skip_build:
        commands.run(
            ["just", "release-host-build"],
            cwd=worktree,
            log_path=output / "logs" / f"build-{spec.label}-host.log",
        )
        commands.run(
            ["just", "release-runtime-build", backend],
            cwd=worktree,
            log_path=output / "logs" / f"build-{spec.label}-runtime.log",
        )
    if not binary.is_file():
        raise FileNotFoundError(f"release host not found: {binary}")
    runtime = find_runtime(runtime_root, backend)
    actual_head = git(worktree, "rev-parse", "HEAD")
    if actual_head != spec.commit:
        raise RuntimeError(
            f"worktree moved during build: expected {spec.commit}, found {actual_head}"
        )
    return {
        "label": spec.label,
        "ref": spec.ref,
        "commit": spec.commit,
        "worktree": str(worktree),
        "binary": str(binary),
        "binary_sha256": sha256(binary),
        "runtime_root": str(runtime_root),
        "runtime": str(runtime),
        "runtime_sha256": tree_sha256(runtime),
        "backend": backend,
    }


def load_competitive_config() -> dict[str, Any]:
    return json.loads(COMPETITIVE_CONFIG.read_text(encoding="utf-8"))


def verify_dataset(path: Path, expected_sha256: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Thoughtworks parquet not found: {path}")
    actual = sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"Thoughtworks parquet SHA-256 mismatch: expected={expected_sha256} actual={actual}"
        )


def build_trajectory_manifest(args: argparse.Namespace, output: Path) -> dict[str, Any]:
    config = load_competitive_config()
    dataset = config["thoughtworks"]["dataset"]
    verify_dataset(args.dataset_file, dataset["sha256"])
    manifest = output / "inputs" / "thoughtworks-trajectories.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(TRAJECTORY_GENERATOR),
        "--dataset-file",
        str(args.dataset_file),
        "--dataset-revision",
        dataset["revision"],
        "--output",
        str(manifest),
        "--trajectories-per-framework",
        str(args.trajectories_per_framework),
        "--min-isl",
        str(args.min_isl),
        "--max-isl",
        str(args.max_isl),
        "--min-turns",
        str(args.min_turns),
    ]
    for concurrency in args.concurrency:
        command.extend(("--cohort", str(concurrency)))
    for framework in args.framework:
        command.extend(("--framework", framework))
    for source in args.source_dataset:
        command.extend(("--source-dataset", source))
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "prompt generation failed; install DuckDB in this Python environment "
            f"and verify the requested selection window ({args.min_isl}-{args.max_isl})"
        ) from error
    document = json.loads(manifest.read_text(encoding="utf-8"))
    return {
        "dataset": dataset,
        "dataset_file": str(args.dataset_file),
        "dataset_file_sha256": sha256(args.dataset_file),
        "trajectory_generator": str(TRAJECTORY_GENERATOR),
        "trajectory_generator_sha256": sha256(TRAJECTORY_GENERATOR),
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "metadata": document["metadata"],
        "cohorts": document["metadata"]["cohorts"],
    }


def load_trajectory_cohorts(path: Path) -> dict[str, list[dict[str, Any]]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    cohorts = document.get("cohorts")
    if not isinstance(cohorts, dict) or not cohorts:
        raise ValueError("trajectory manifest must contain nonempty cohorts")
    for name, trajectories in cohorts.items():
        if not isinstance(name, str) or not isinstance(trajectories, list) or not trajectories:
            raise ValueError("each trajectory cohort must be a nonempty list")
        for trajectory in trajectories:
            if not isinstance(trajectory.get("session_id"), str):
                raise ValueError(f"cohort {name} contains a trajectory without session_id")
            if not isinstance(trajectory.get("messages"), list):
                raise ValueError(
                    f"trajectory {trajectory.get('session_id')} has no messages list"
                )
    return cohorts


def server_command(binary: Path, model: str) -> list[str]:
    command = [str(binary), "serve", "--model", model, "--log-format", "json"]
    for option in FORBIDDEN_STARTUP_OPTIONS:
        if option in command:
            raise AssertionError(f"default-startup benchmark cannot use {option}")
    return command


def port_is_open(host: str = "127.0.0.1", port: int = 9337) -> bool:
    with socket.socket() as connection:
        connection.settimeout(0.2)
        return connection.connect_ex((host, port)) == 0


def wait_for_model(
    base_url: str,
    timeout: float,
    process: Optional[subprocess.Popen[bytes]] = None,
) -> str:
    if base_url != DEFAULT_BASE_URL:
        raise ValueError(f"default-startup runner requires {DEFAULT_BASE_URL}")
    deadline = time.monotonic() + timeout
    last_error = "not ready"
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"Mesh exited before readiness with status {process.returncode}"
            )
        connection = http.client.HTTPConnection("127.0.0.1", 9337, timeout=5)
        try:
            connection.request("GET", "/v1/models")
            response = connection.getresponse()
            body = response.read()
            if response.status == 200:
                document = json.loads(body)
                models = document.get("data") or []
                if models:
                    return models[0]["id"]
            last_error = f"HTTP {response.status}: {body[:300]!r}"
        except (OSError, json.JSONDecodeError) as error:
            last_error = str(error)
        finally:
            connection.close()
        time.sleep(1)
    raise TimeoutError(f"Mesh did not become ready after {timeout}s: {last_error}")


def percentile(values: Sequence[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    index = min(math.ceil(len(ordered) * fraction) - 1, len(ordered) - 1)
    return ordered[max(index, 0)]


def stream_request(
    request_id: str,
    messages: Sequence[dict[str, Any]],
    tools: Sequence[dict[str, Any]],
    metadata: dict[str, Any],
    model_id: str,
    output_tokens: int,
    timeout: float,
) -> dict[str, Any]:
    started = time.monotonic()
    first_token_at: Optional[float] = None
    completion_tokens = 0
    prompt_tokens = 0
    cached_tokens = 0
    content_events = 0
    content_parts: list[str] = []
    connection = http.client.HTTPConnection("127.0.0.1", 9337, timeout=timeout)
    payload = {
        "model": model_id,
        "messages": list(messages),
        "max_tokens": output_tokens,
        "temperature": 0,
        "seed": 42,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if tools:
        payload["tools"] = list(tools)
    try:
        connection.request(
            "POST",
            "/v1/chat/completions",
            json.dumps(payload),
            {"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        )
        response = connection.getresponse()
        if response.status != 200:
            body = response.read(4096).decode("utf-8", errors="replace")
            return {
                "request_id": request_id,
                **metadata,
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
                completion_tokens = int(usage.get("completion_tokens") or completion_tokens)
                prompt_tokens = int(usage.get("prompt_tokens") or prompt_tokens)
                details = usage.get("prompt_tokens_details")
                if isinstance(details, dict):
                    cached_tokens = int(details.get("cached_tokens") or cached_tokens)
            choices = event.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            delta = choices[0].get("delta")
            if not isinstance(delta, dict):
                continue
            content = delta.get("content") or delta.get("reasoning_content")
            tool_calls = delta.get("tool_calls")
            if content or tool_calls:
                if first_token_at is None:
                    first_token_at = time.monotonic()
                content_events += 1
                if content:
                    content_parts.append(content)
                if tool_calls:
                    content_parts.append(json.dumps(tool_calls, sort_keys=True))
        completed = time.monotonic()
    except Exception as error:  # preserve request-level failures in the artifact
        return {
            "request_id": request_id,
            **metadata,
            "error": f"{type(error).__name__}: {error}",
        }
    finally:
        connection.close()
    if first_token_at is None:
        return {
            "request_id": request_id,
            **metadata,
            "error": "stream completed without generated content",
        }
    if completion_tokens <= 0:
        return {
            "request_id": request_id,
            **metadata,
            "error": "stream completed without completion-token usage",
        }
    return {
        "request_id": request_id,
        **metadata,
        "started": started,
        "first_token_at": first_token_at,
        "completed": completed,
        "ttft_seconds": first_token_at - started,
        "elapsed_seconds": completed - started,
        "generation_seconds": completed - first_token_at,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "content_events": content_events,
        "content_sha256": hashlib.sha256("".join(content_parts).encode()).hexdigest(),
    }


def openai_message(recorded: dict[str, Any]) -> dict[str, Any]:
    message: dict[str, Any] = {
        "role": recorded["role"],
        "content": recorded.get("content") or "",
    }
    tool_calls_json = recorded.get("tool_calls_json")
    if tool_calls_json:
        message["tool_calls"] = json.loads(tool_calls_json)
    tool_call_id = recorded.get("tool_call_id")
    if tool_call_id:
        message["tool_call_id"] = tool_call_id
    return message


def recorded_output_budget(recorded: dict[str, Any], maximum: int) -> int:
    content_length = len(recorded.get("content") or "")
    tool_length = len(recorded.get("tool_calls_json") or "")
    approximate_tokens = math.ceil((content_length + tool_length) / 4)
    return min(max(approximate_tokens, 8), maximum)


def trajectory_tools(trajectory: dict[str, Any]) -> list[dict[str, Any]]:
    names: set[str] = set()
    for recorded in trajectory["messages"]:
        tool_calls_json = recorded.get("tool_calls_json")
        if not tool_calls_json:
            continue
        for tool_call in json.loads(tool_calls_json):
            function = tool_call.get("function")
            if isinstance(function, dict) and isinstance(function.get("name"), str):
                names.add(function["name"])
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": "Tool available in the recorded agent trajectory.",
                "parameters": {"type": "object", "additionalProperties": True},
            },
        }
        for name in sorted(names)
    ]


def replay_trajectory(
    trajectory: dict[str, Any],
    model_id: str,
    max_output_tokens: int,
    timeout: float,
) -> list[dict[str, Any]]:
    history: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    assistant_turn = 0
    tools = trajectory_tools(trajectory)
    for message_index, recorded in enumerate(trajectory["messages"]):
        if recorded["role"] == "assistant":
            requested_output_tokens = recorded_output_budget(
                recorded, max_output_tokens
            )
            metadata = {
                "session_id": trajectory["session_id"],
                "source_dataset": trajectory["source_dataset"],
                "agent_framework": trajectory["agent_framework"],
                "recorded_model": trajectory["recorded_model"],
                "assistant_turn": assistant_turn,
                "recorded_message_index": message_index,
                "history_message_count": len(history),
                "requested_output_tokens": requested_output_tokens,
                "recorded_output_characters": len(recorded.get("content") or ""),
                "available_tools": len(tools),
            }
            result = stream_request(
                f"{trajectory['session_id']}:{assistant_turn}",
                history,
                tools,
                metadata,
                model_id,
                requested_output_tokens,
                timeout,
            )
            results.append(result)
            assistant_turn += 1
        # Continue with the recorded trajectory, not the generated benchmark
        # output, so every experiment arm receives the same ordered history.
        history.append(openai_message(recorded))
    return results


def summarize_requests(requests: Sequence[dict[str, Any]]) -> dict[str, Any]:
    successful = [request for request in requests if "error" not in request]
    ttft = [request["ttft_seconds"] for request in successful]
    completion_tokens = sum(request["completion_tokens"] for request in successful)
    prompt_tokens = sum(request["prompt_tokens"] for request in successful)
    cached_tokens = sum(request["cached_tokens"] for request in successful)
    if successful:
        workload_window = max(request["completed"] for request in successful) - min(
            request["started"] for request in successful
        )
    else:
        workload_window = 0.0
    request_decode_rates = [
        request["completion_tokens"] / request["generation_seconds"]
        for request in successful
        if request["generation_seconds"] > 0
    ]
    return {
        "requests": len(requests),
        "successful_requests": len(successful),
        "failed_requests": len(requests) - len(successful),
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "exact_output_requests": sum(
            request.get("completion_tokens") == request.get("requested_output_tokens")
            for request in successful
        ),
        "ttft_p50_seconds": statistics.median(ttft) if ttft else None,
        "ttft_p95_seconds": percentile(ttft, 0.95),
        "agent_steps_per_second": (
            len(successful) / workload_window if workload_window > 0 else None
        ),
        "workload_output_tokens_per_second": (
            completion_tokens / workload_window if workload_window > 0 else None
        ),
        "mean_request_decode_tokens_per_second": (
            statistics.mean(request_decode_rates) if request_decode_rates else None
        ),
        "cache_pct": 100 * cached_tokens / prompt_tokens if prompt_tokens else None,
    }


def run_trajectory_cell(
    *,
    trajectories: Sequence[dict[str, Any]],
    model_id: str,
    concurrency: int,
    max_output_tokens: int,
    timeout: float,
    raw_path: Path,
) -> dict[str, Any]:
    if not trajectories:
        raise ValueError("trajectory cell cannot be empty")
    requests: list[dict[str, Any]] = []
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(concurrency, len(trajectories))
    ) as pool:
        futures = [
            pool.submit(
                replay_trajectory,
                trajectory,
                model_id,
                max_output_tokens,
                timeout,
            )
            for trajectory in trajectories
        ]
        for future in futures:
            requests.extend(future.result())
    with raw_path.open("w", encoding="utf-8") as raw:
        for request in requests:
            request["concurrency"] = concurrency
            raw.write(json.dumps(request, sort_keys=True) + "\n")
    summary = summarize_requests(requests)
    framework_counts: dict[str, int] = {}
    for trajectory in trajectories:
        framework = trajectory["agent_framework"]
        framework_counts[framework] = framework_counts.get(framework, 0) + 1
    successful_sessions = {
        request["session_id"]
        for request in requests
        if "error" not in request
    }
    failed_sessions = {
        request["session_id"] for request in requests if "error" in request
    }
    summary.update(
        {
            "concurrency": concurrency,
            "trajectories": len(trajectories),
            "successful_trajectories": len(successful_sessions - failed_sessions),
            "failed_trajectories": len(failed_sessions),
            "framework_trajectories": framework_counts,
            "max_output_tokens": max_output_tokens,
            "ordered_replay": True,
        }
    )
    return summary


def isolated_server_env(
    runtime_root: Path, state_dir: Path, hf_home: Optional[Path]
) -> dict[str, str]:
    env = os.environ.copy()
    inherited_home = Path.home()
    home = state_dir / "home"
    home.mkdir(parents=True, exist_ok=True)
    env.update(
        {
            "HOME": str(home),
            "XDG_CACHE_HOME": str(state_dir / "xdg-cache"),
            "XDG_CONFIG_HOME": str(state_dir / "xdg-config"),
            "MESH_LLM_RUNTIME_ROOT": str(state_dir / "runtime"),
            "MESH_LLM_NATIVE_RUNTIME_BUNDLE_DIR": str(runtime_root),
        }
    )
    if hf_home is not None:
        env["HF_HOME"] = str(hf_home)
    elif "HF_HOME" not in env:
        env["HF_HOME"] = str(inherited_home / ".cache/huggingface")
    return env


def start_server(
    build: dict[str, Any],
    model: str,
    state_dir: Path,
    log_path: Path,
    hf_home: Optional[Path],
) -> tuple[subprocess.Popen[bytes], list[str]]:
    if port_is_open():
        raise RuntimeError("TCP 9337 is already in use; stop the existing Mesh instance")
    command = server_command(Path(build["binary"]), model)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("wb")
    process = subprocess.Popen(
        command,
        cwd=Path(build["worktree"]),
        env=isolated_server_env(Path(build["runtime_root"]), state_dir, hf_home),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_handle.close()
    return process, command


def stop_server(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
    deadline = time.monotonic() + 10
    while port_is_open() and time.monotonic() < deadline:
        time.sleep(0.2)
    if port_is_open():
        raise RuntimeError("Mesh stopped but TCP 9337 is still occupied")


def collect_runtime_logs(state_dir: Path, output_dir: Path) -> None:
    runtime_root = state_dir / "runtime"
    if not runtime_root.is_dir():
        return
    for source in runtime_root.rglob("*"):
        if not source.is_file() or "logs" not in source.parts:
            continue
        destination = output_dir / source.relative_to(runtime_root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def run_arm_pass(
    *,
    args: argparse.Namespace,
    output: Path,
    build: dict[str, Any],
    pass_index: int,
    cohorts: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    label = build["label"]
    pass_dir = output / "data" / f"pass-{pass_index + 1}" / label
    state_dir = Path(
        tempfile.mkdtemp(prefix=f"agentic-replay-{label}-pass-{pass_index + 1}-")
    )
    log_path = pass_dir / "mesh.log"
    started_at = utc_now()
    process: Optional[subprocess.Popen[bytes]] = None
    command = server_command(Path(build["binary"]), args.model)
    cells: list[dict[str, Any]] = []
    try:
        process, command = start_server(
            build, args.model, state_dir, log_path, args.hf_home
        )
        model_id = wait_for_model(DEFAULT_BASE_URL, args.startup_timeout, process)
        concurrency_values = (
            args.concurrency
            if pass_index % 2 == 0
            else list(reversed(args.concurrency))
        )
        for concurrency in concurrency_values:
            cell = run_trajectory_cell(
                trajectories=cohorts[str(concurrency)],
                model_id=model_id,
                concurrency=concurrency,
                max_output_tokens=args.max_output_tokens,
                timeout=args.request_timeout,
                raw_path=pass_dir / f"c-{concurrency}-requests.jsonl",
            )
            cells.append(cell)
            write_json(pass_dir / f"c-{concurrency}.json", cell)
    finally:
        try:
            if process is not None:
                stop_server(process)
        finally:
            collect_runtime_logs(state_dir, pass_dir / "native-runtime")
            shutil.rmtree(state_dir, ignore_errors=True)
    return {
        "label": label,
        "ref": build["ref"],
        "commit": build["commit"],
        "pass": pass_index + 1,
        "started_at": started_at,
        "completed_at": utc_now(),
        "server_command": command,
        "server_log": str(log_path),
        "model_id": model_id,
        "cells": cells,
    }


def mean_or_none(values: Iterable[Optional[float]]) -> Optional[float]:
    present = [value for value in values if value is not None]
    return statistics.mean(present) if present else None


def pooled_rows(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    metadata: dict[str, tuple[str, str]] = {}
    for arm_pass in results:
        metadata[arm_pass["label"]] = (arm_pass["ref"], arm_pass["commit"])
        for cell in arm_pass["cells"]:
            groups.setdefault((arm_pass["label"], cell["concurrency"]), []).append(cell)
    rows: list[dict[str, Any]] = []
    for (label, concurrency), cells in sorted(groups.items()):
        ref, commit = metadata[label]
        requests = sum(cell["requests"] for cell in cells)
        successes = sum(cell["successful_requests"] for cell in cells)
        rows.append(
            {
                "label": label,
                "ref": ref,
                "commit": commit,
                "concurrency": concurrency,
                "passes": len(cells),
                "trajectories_per_pass": cells[0]["trajectories"],
                "trajectory_replays": sum(cell["trajectories"] for cell in cells),
                "requests": requests,
                "successful_requests": successes,
                "success_pct": 100 * successes / requests if requests else None,
                "exact_output_pct": 100
                * sum(cell["exact_output_requests"] for cell in cells)
                / successes
                if successes
                else None,
                "agent_steps_per_second": mean_or_none(
                    cell["agent_steps_per_second"] for cell in cells
                ),
                "workload_output_tokens_per_second": mean_or_none(
                    cell["workload_output_tokens_per_second"] for cell in cells
                ),
                "mean_request_decode_tokens_per_second": mean_or_none(
                    cell["mean_request_decode_tokens_per_second"] for cell in cells
                ),
                "ttft_p50_seconds": mean_or_none(cell["ttft_p50_seconds"] for cell in cells),
                "ttft_p95_seconds": mean_or_none(cell["ttft_p95_seconds"] for cell in cells),
                "cache_pct": mean_or_none(cell["cache_pct"] for cell in cells),
            }
        )
    baseline_label = results[0]["label"] if results else None
    baseline = {
        row["concurrency"]: row
        for row in rows
        if row["label"] == baseline_label
    }
    for row in rows:
        reference = baseline.get(row["concurrency"])
        for metric in (
            "agent_steps_per_second",
            "workload_output_tokens_per_second",
            "ttft_p50_seconds",
        ):
            base_value = reference.get(metric) if reference else None
            value = row.get(metric)
            row[f"{metric}_delta_pct"] = (
                100 * (value / base_value - 1)
                if value is not None and base_value not in (None, 0)
                else None
            )
    return rows


def fmt(value: Optional[float], digits: int = 2, suffix: str = "") -> str:
    return "—" if value is None else f"{value:.{digits}f}{suffix}"


def escape(value: Any) -> str:
    return html.escape(str(value))


def svg_chart(
    title: str,
    rows: Sequence[dict[str, Any]],
    labels: Sequence[str],
    metric: str,
    y_label: str,
    output: Path,
) -> None:
    width, height = 960, 540
    left, top, plot_width, plot_height = 100, 80, 790, 350
    concurrency_values = sorted({row["concurrency"] for row in rows})
    values = [row[metric] for row in rows if row.get(metric) is not None]
    y_max = max(max(values) * 1.12, 1e-9) if values else 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        f'<text x="480" y="36" text-anchor="middle" font-family="sans-serif" font-size="22" font-weight="700">{escape(title)}</text>',
    ]
    for tick in range(6):
        value = y_max * tick / 5
        y = top + plot_height - plot_height * tick / 5
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#e2e8f0"/>'
        )
        parts.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{value:.1f}</text>'
        )
    x_denominator = max(len(concurrency_values) - 1, 1)
    for index, concurrency in enumerate(concurrency_values):
        x = left + plot_width * index / x_denominator
        parts.append(
            f'<text x="{x:.1f}" y="{top + plot_height + 24}" text-anchor="middle" font-family="sans-serif" font-size="12">{concurrency}</text>'
        )
    for label_index, label in enumerate(labels):
        color = COLORS[label_index % len(COLORS)]
        points: list[tuple[float, float]] = []
        indexed = {
            row["concurrency"]: row[metric]
            for row in rows
            if row["label"] == label and row.get(metric) is not None
        }
        for index, concurrency in enumerate(concurrency_values):
            if concurrency not in indexed:
                continue
            x = left + plot_width * index / x_denominator
            y = top + plot_height - plot_height * indexed[concurrency] / y_max
            points.append((x, y))
        coordinates = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        parts.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="3"/>'
        )
        parts.extend(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>'
            for x, y in points
        )
        legend_x = 110 + label_index * 150
        parts.append(
            f'<text x="{legend_x}" y="490" font-family="sans-serif" font-size="13" fill="{color}">{escape(label)}</text>'
        )
    parts.extend(
        [
            '<text x="480" y="525" text-anchor="middle" font-family="sans-serif" font-size="13">Client concurrency</text>',
            f'<text x="22" y="255" text-anchor="middle" transform="rotate(-90 22 255)" font-family="sans-serif" font-size="13">{escape(y_label)}</text>',
            "</svg>",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(parts), encoding="utf-8")


def write_report(output: Path, run_document: dict[str, Any]) -> Path:
    rows = pooled_rows(run_document["results"])
    summary = output / "summary"
    charts = summary / "charts"
    summary.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0]) if rows else []
    with (summary / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    labels = [build["label"] for build in run_document["builds"]]
    cohort_metadata = run_document["inputs"]["cohorts"]
    selected_trajectories = sum(
        cohort["trajectory_count"] for cohort in cohort_metadata.values()
    )
    selected_turns = sum(cohort["assistant_turns"] for cohort in cohort_metadata.values())
    svg_chart(
        "Agent-step throughput by commit",
        rows,
        labels,
        "agent_steps_per_second",
        "Completed agent steps / second",
        charts / "agent-step-throughput.svg",
    )
    svg_chart(
        "Median time to first token by commit",
        rows,
        labels,
        "ttft_p50_seconds",
        "Seconds (lower is better)",
        charts / "ttft-p50.svg",
    )
    lines = [
        "# Agentic Replay",
        "",
        f"Generated: `{utc_now()}`",
        "",
        "## Trajectory selection",
        "",
        "| Client concurrency | Whole trajectories | Recorded agent steps | Framework trajectory / step breakdown |",
        "|---:|---:|---:|---|",
    ]
    for concurrency in run_document["config"]["concurrency"]:
        cohort = cohort_metadata[str(concurrency)]
        breakdown = " · ".join(
            f"{framework} {count} / {cohort['framework_assistant_turns'][framework]}"
            for framework, count in cohort["framework_trajectories"].items()
        )
        lines.append(
            f"| {concurrency} | {cohort['trajectory_count']} | "
            f"{cohort['assistant_turns']} | {breakdown} |"
        )
    lines.extend(
        [
            "",
            "## Result table",
            "",
            "| Ref | Commit | C | Trajectories/pass | Agent steps | Success | Steps/s | vs baseline | Output tok/s | TTFT p50 | TTFT p95 | TTFT p50 vs baseline | Cached prompt |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {label} | `{commit}` | {concurrency} | {trajectories} | {requests} | {success} | {steps} | {steps_delta} | {output_tps} | {p50} | {p95} | {p50_delta} | {cache} |".format(
                label=row["label"],
                commit=row["commit"][:10],
                concurrency=row["concurrency"],
                trajectories=row["trajectories_per_pass"],
                requests=row["requests"],
                success=fmt(row["success_pct"], 1, "%"),
                steps=fmt(row["agent_steps_per_second"], 3),
                steps_delta=fmt(row["agent_steps_per_second_delta_pct"], 1, "%"),
                output_tps=fmt(row["workload_output_tokens_per_second"]),
                p50=fmt(row["ttft_p50_seconds"], 3, "s"),
                p95=fmt(row["ttft_p95_seconds"], 3, "s"),
                p50_delta=fmt(row["ttft_p50_seconds_delta_pct"], 1, "%"),
                cache=fmt(row["cache_pct"], 1, "%"),
            )
        )
    lines.extend(
        [
            "",
            "## Charts",
            "",
            "![Agent-step throughput](charts/agent-step-throughput.svg)",
            "",
            "![Median TTFT](charts/ttft-p50.svg)",
            "",
            "## Method",
            "",
            f"- Model: `{run_document['config']['model']}`",
            "- Server startup: `mesh-llm serve --model <model> --log-format json`.",
            "- Mesh chooses context size, execution lanes, KV budget, and backend tuning.",
            f"- Client concurrency: `{','.join(map(str, run_document['config']['concurrency']))}`.",
            f"- Pass order: `{' → '.join(item['label'] for item in run_document['order'])}`.",
            f"- Dataset revision: `{run_document['inputs']['dataset']['revision']}`.",
            f"- Selected trajectories: `{selected_trajectories}` unique whole sessions across disjoint concurrency cohorts.",
            f"- Recorded agent steps: `{selected_turns}` assistant turns per arm pass; each commit replays them once per pass.",
            "- Turns inside a trajectory are strictly sequential. Different trajectories may overlap up to the offered client concurrency.",
            "- Each next request uses the recorded conversation history, so experiment arms receive identical growing prefixes and tool observations.",
            "- Per-turn output budgets approximate each recorded assistant action from its character length, capped by the configured maximum; generated output is measured but never fed into the next turn.",
            f"- Trajectory manifest SHA-256: `{run_document['inputs']['manifest_sha256']}`.",
            "- Raw request records, server logs, build logs, commands, and exact binary/runtime hashes are retained beside this report.",
            "",
            "This report presents measurements and does not make an automatic release decision.",
            "",
        ]
    )
    report_path = summary / "REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    write_json(summary / "comparison.json", rows)
    inventory: list[str] = []
    for path in sorted(item for item in output.rglob("*") if item.is_file()):
        if path.name == "artifact-sha256.txt":
            continue
        inventory.append(f"{sha256(path)}  {path.relative_to(output).as_posix()}")
    (output / "artifact-sha256.txt").write_text("\n".join(inventory) + "\n", encoding="utf-8")
    return report_path


def benchmark_plan(args: argparse.Namespace, specs: Sequence[RefSpec]) -> dict[str, Any]:
    config = load_competitive_config()
    dataset = config["thoughtworks"]["dataset"]
    return {
        "schema_version": 1,
        "repo": str(args.repo),
        "refs": [spec.__dict__ for spec in specs],
        "order": [
            {"pass": pass_index + 1, "label": spec.label, "commit": spec.commit}
            for pass_index, spec in ab_order(specs, args.passes)
        ],
        "build_commands": [
            ["just", "release-host-build"],
            ["just", "release-runtime-build", args.backend],
        ],
        "server_command": ["<release-binary>", "serve", "--model", args.model, "--log-format", "json"],
        "dataset": dataset,
        "selection": {
            "source_datasets": args.source_dataset,
            "frameworks": args.framework,
            "trajectories_per_framework_per_concurrency": args.trajectories_per_framework,
            "unique_trajectory_count": len(args.concurrency)
            * len(args.framework)
            * args.trajectories_per_framework,
            "min_isl": args.min_isl,
            "max_isl_exclusive": args.max_isl,
            "min_turns": args.min_turns,
        },
        "workload": {
            "concurrency": args.concurrency,
            "passes": args.passes,
            "ordered_whole_trajectory_replay": True,
            "max_output_tokens": args.max_output_tokens,
        },
        "outputs": [
            "raw request JSONL",
            "per-cell JSON",
            "server/build logs",
            "comparison CSV/JSON/Markdown",
            "throughput and TTFT SVG charts",
            "SHA-256 inventory",
        ],
    }


def run_benchmark(args: argparse.Namespace) -> Path:
    args.repo = args.repo.resolve()
    args.output = args.output.resolve()
    args.dataset_file = args.dataset_file.resolve()
    if args.hf_home is not None:
        args.hf_home = args.hf_home.resolve()
    specs = parse_ref_specs(args.repo, args.ref)
    plan = benchmark_plan(args, specs)
    args.output.mkdir(parents=True, exist_ok=True)
    existing = args.output / "run.json"
    if existing.exists() and not args.resume:
        raise RuntimeError(f"output already contains run.json; pass --resume: {args.output}")
    write_json(args.output / "plan.json", plan)
    commands = CommandLog(args.output / "commands.jsonl")
    inputs = build_trajectory_manifest(args, args.output)
    cohorts = load_trajectory_cohorts(Path(inputs["manifest"]))
    worktree_root = (
        args.worktree_root or (args.repo.parent / ".agentic-replay-worktrees")
    ).resolve()
    builds = []
    for spec in specs:
        worktree = prepare_worktree(args.repo, worktree_root, spec)
        builds.append(
            build_ref(spec, worktree, args.backend, args.output, commands, args.skip_build)
        )
    run_document: dict[str, Any] = {
        "schema_version": 1,
        "started_at": utc_now(),
        "host": {
            "hostname": socket.gethostname(),
            "platform": sys.platform,
            "python": sys.version,
        },
        "config": {
            "model": args.model,
            "backend": args.backend,
            "concurrency": args.concurrency,
            "passes": args.passes,
            "max_output_tokens": args.max_output_tokens,
        },
        "plan_sha256": stable_hash(plan),
        "inputs": inputs,
        "builds": builds,
        "order": plan["order"],
        "results": [],
    }
    build_by_label = {build["label"]: build for build in builds}
    completed = {
        (item["pass"], item["label"])
        for item in json.loads(existing.read_text(encoding="utf-8")).get("results", [])
    } if existing.exists() and args.resume else set()
    if existing.exists() and args.resume:
        previous = json.loads(existing.read_text(encoding="utf-8"))
        if previous.get("plan_sha256") != run_document["plan_sha256"]:
            raise RuntimeError("cannot resume: plan differs from existing run.json")
        previous_builds = {
            build["label"]: (
                build["commit"],
                build["binary_sha256"],
                build["runtime_sha256"],
            )
            for build in previous.get("builds", [])
        }
        current_builds = {
            build["label"]: (
                build["commit"],
                build["binary_sha256"],
                build["runtime_sha256"],
            )
            for build in builds
        }
        if previous_builds != current_builds:
            raise RuntimeError("cannot resume: built binary or runtime hashes differ")
        if previous.get("inputs", {}).get("manifest_sha256") != inputs["manifest_sha256"]:
            raise RuntimeError("cannot resume: Thoughtworks trajectory manifest differs")
        run_document["results"] = previous["results"]
    for pass_index, spec in ab_order(specs, args.passes):
        key = (pass_index + 1, spec.label)
        if key in completed:
            continue
        result = run_arm_pass(
            args=args,
            output=args.output,
            build=build_by_label[spec.label],
            pass_index=pass_index,
            cohorts=cohorts,
        )
        run_document["results"].append(result)
        write_json(existing, run_document)
    run_document["completed_at"] = utc_now()
    write_json(existing, run_document)
    return write_report(args.output, run_document)


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument(
        "--ref",
        action="append",
        required=True,
        help="repeatable LABEL=GIT_REF; at least two distinct commits",
    )
    parser.add_argument("--model", required=True, help="model URI or local package path")
    parser.add_argument("--backend", default="metal")
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--concurrency", type=int, action="append", default=[])
    parser.add_argument(
        "--trajectories-per-framework",
        type=int,
        required=True,
        help="whole trajectories from each framework in each concurrency cohort",
    )
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--min-isl", type=int, default=8192)
    parser.add_argument("--max-isl", type=int, default=65536)
    parser.add_argument("--min-turns", type=int, default=5)
    parser.add_argument("--framework", action="append", default=[])
    parser.add_argument(
        "--source-dataset",
        action="append",
        default=[],
        help="Thoughtworks source_dataset; defaults to all three pinned sources",
    )


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not args.concurrency:
        args.concurrency = [1, 2, 4]
    if len(set(args.concurrency)) != len(args.concurrency) or any(
        value <= 0 for value in args.concurrency
    ):
        parser.error("--concurrency values must be unique and positive")
    positive = (
        args.passes,
        args.trajectories_per_framework,
        args.max_output_tokens,
        args.min_isl,
        args.min_turns,
    )
    if any(value <= 0 for value in positive):
        parser.error("passes and workload sizes must be positive")
    if args.max_isl <= args.min_isl:
        parser.error("--max-isl must exceed --min-isl")
    if not args.source_dataset:
        args.source_dataset = [
            "swe-smith-claude-3-7-sonnet",
            "kwai-klear-swe-smith-mini",
            "nebius-swe-rebench-openhands",
        ]
    if not args.framework:
        args.framework = ["swe-agent", "mini-swe-agent", "openhands"]
    if len(set(args.framework)) != len(args.framework):
        parser.error("--framework values must be unique")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="print the exact side-effect-free benchmark plan")
    add_common_arguments(plan)
    run = subparsers.add_parser("run", help="build refs, run the matrix, and render the report")
    add_common_arguments(run)
    run.add_argument("--dataset-file", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--worktree-root", type=Path)
    run.add_argument("--hf-home", type=Path)
    run.add_argument("--startup-timeout", type=float, default=1800)
    run.add_argument("--request-timeout", type=float, default=900)
    run.add_argument("--skip-build", action="store_true")
    run.add_argument("--resume", action="store_true")
    report = subparsers.add_parser("report", help="rerender tables and charts from run.json")
    report.add_argument("--artifact", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command in {"plan", "run"}:
        validate_args(args, parser)
    return args


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.command == "plan":
        args.repo = args.repo.resolve()
        specs = parse_ref_specs(args.repo, args.ref)
        print(json.dumps(benchmark_plan(args, specs), indent=2, sort_keys=True))
        return 0
    if args.command == "run":
        print(run_benchmark(args))
        return 0
    artifact = args.artifact.resolve()
    document = json.loads((artifact / "run.json").read_text(encoding="utf-8"))
    print(write_report(artifact, document))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
